# MOSAAIC: Managing Optimization towards Shared Autonomy, Authority, and Initiative in Co-creation 

**Title (ZH)**: MOSAAIC: 管理优化以实现共创中的共享自主性、权威性和主动性 

**Authors**: Alayt Issak, Jeba Rezwana, Casper Harteveld  

**Link**: [PDF](https://arxiv.org/pdf/2505.11481)  

**Abstract**: Striking the appropriate balance between humans and co-creative AI is an open research question in computational creativity. Co-creativity, a form of hybrid intelligence where both humans and AI take action proactively, is a process that leads to shared creative artifacts and ideas. Achieving a balanced dynamic in co-creativity requires characterizing control and identifying strategies to distribute control between humans and AI. We define control as the power to determine, initiate, and direct the process of co-creation. Informed by a systematic literature review of 172 full-length papers, we introduce MOSAAIC (Managing Optimization towards Shared Autonomy, Authority, and Initiative in Co-creation), a novel framework for characterizing and balancing control in co-creation. MOSAAIC identifies three key dimensions of control: autonomy, initiative, and authority. We supplement our framework with control optimization strategies in co-creation. To demonstrate MOSAAIC's applicability, we analyze the distribution of control in six existing co-creative AI case studies and present the implications of using this framework. 

**Abstract (ZH)**: 在计算创意中实现人类与合创AI之间的适当平衡是一个开放的研究问题。合创是一种混合智能形式，其中人类和AI主动采取行动，这一过程会产生共享的创意成果和想法。实现合创中的平衡动态需要表征控制并确定在人类和AI之间分配控制的策略。我们将控制定义为确定、启动和指导合创过程的能力。基于对172篇全文论文的系统的文献综述，我们引入了MOSAAIC（管理和优化以实现合创中的共享自主权、权威和主动性）框架，这是一种表征和平衡合创中控制的新框架。MOSAAIC识别出控制的三个关键维度：自主权、主动性以及权威。我们还为合创中的控制优化提供了策略。为了展示MOSAAIC的适用性，我们分析了六个现有合创AI案例研究中的控制分配，并讨论了使用该框架的影响。 

---
# Automatic Reward Shaping from Confounded Offline Data 

**Title (ZH)**: 从混杂的离线数据中自动构建奖励函数 

**Authors**: Mingxuan Li, Junzhe Zhang, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2505.11478)  

**Abstract**: A key task in Artificial Intelligence is learning effective policies for controlling agents in unknown environments to optimize performance measures. Off-policy learning methods, like Q-learning, allow learners to make optimal decisions based on past experiences. This paper studies off-policy learning from biased data in complex and high-dimensional domains where \emph{unobserved confounding} cannot be ruled out a priori. Building on the well-celebrated Deep Q-Network (DQN), we propose a novel deep reinforcement learning algorithm robust to confounding biases in observed data. Specifically, our algorithm attempts to find a safe policy for the worst-case environment compatible with the observations. We apply our method to twelve confounded Atari games, and find that it consistently dominates the standard DQN in all games where the observed input to the behavioral and target policies mismatch and unobserved confounders exist. 

**Abstract (ZH)**: 人工智能中的一个关键任务是学习有效的策略以在未知环境中控制代理，以优化性能指标。离策学习方法，如Q学习，允许学习者基于过往经验做出最优决策。本文研究了在复杂和高维域中从可能存在未观察混杂因素的偏差数据中进行离策学习。基于广受赞誉的深度Q网络（DQN），我们提出了一种新型深度强化学习算法，该算法能够抵抗观测数据中的混杂偏差。具体而言，我们的算法尝试找到与观测数据相兼容的最坏情况环境下的安全策略。我们将方法应用于十二个存在混杂因素的Atari游戏，并发现它在所有行为策略和目标策略的输入匹配存在偏差的游戏场景中均优于标准DQN。 

---
# Extracting Explainable Dates From Medical Images By Reverse-Engineering UNIX Timestamps 

**Title (ZH)**: 从医学图像中通过逆向工程UNIX时间戳提取可解释日期 

**Authors**: Lee Harris, James Bentham, Philippe De Wilde  

**Link**: [PDF](https://arxiv.org/pdf/2505.11451)  

**Abstract**: Dates often contribute towards highly impactful medical decisions, but it is rarely clear how to extract this data. AI has only just begun to be used transcribe such documents, and common methods are either to trust that the output produced by a complex AI model, or to parse the text using regular expressions. Recent work has established that regular expressions are an explainable form of logic, but it is difficult to decompose these into the component parts that are required to construct precise UNIX timestamps. First, we test publicly-available regular expressions, and we found that these were unable to capture a significant number of our dates. Next, we manually created easily-decomposable regular expressions, and we found that these were able to detect the majority of real dates, but also a lot of sequences of text that look like dates. Finally, we used regular expression synthesis to automatically identify regular expressions from the reverse-engineered UNIX timestamps that we created. We find that regular expressions created by regular expression synthesis detect far fewer sequences of text that look like dates than those that were manually created, at the cost of a slight increase to the number of missed dates. Overall, our results show that regular expressions can be created through regular expression synthesis to identify complex dates and date ranges in text transcriptions. To our knowledge, our proposed way of learning deterministic logic by reverse-engineering several many-one mappings and feeding these into a regular expression synthesiser is a new approach. 

**Abstract (ZH)**: 日期信息通常对医学决策有重大影响，但很少清楚如何提取这些数据。现有的方法要么完全信任复杂AI模型的输出，要么使用正则表达式解析文本。最近的研究表明，正则表达式是一种可解释的逻辑形式，但将其分解为构建精确UNIX时间戳所需的组件部分却相当困难。首先，我们测试了公开可用的正则表达式，发现它们无法捕捉到我们大量日期中的很大一部分。接着，我们手动创建了易于分解的正则表达式，并发现这些正则表达式能够检测到大部分真实日期，但也检测到了大量看起来像日期的文本序列。最后，我们使用正则表达式合成从我们逆向工程得到的UNIX时间戳自动识别正则表达式。我们发现，由正则表达式合成生成的正则表达式检测到的看起来像日期的文本序列要少于手工创建的正则表达式，但可能会错过更多日期。总体来说，我们的结果表明，可以通过正则表达式合成识别文本转录中的复杂日期和日期范围。据我们所知，通过逆向工程多个单值映射并将其输入正则表达式合成器来学习确定性逻辑是一种新的方法。 

---
# Meta-World+: An Improved, Standardized, RL Benchmark 

**Title (ZH)**: Meta-World+: 一个改进的、标准化的RL基准 

**Authors**: Reginald McLean, Evangelos Chatzaroulas, Luc McCutcheon, Frank Röder, Tianhe Yu, Zhanpeng He, K.R. Zentner, Ryan Julian, J K Terry, Isaac Woungang, Nariman Farsad, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2505.11289)  

**Abstract**: Meta-World is widely used for evaluating multi-task and meta-reinforcement learning agents, which are challenged to master diverse skills simultaneously. Since its introduction however, there have been numerous undocumented changes which inhibit a fair comparison of algorithms. This work strives to disambiguate these results from the literature, while also leveraging the past versions of Meta-World to provide insights into multi-task and meta-reinforcement learning benchmark design. Through this process we release a new open-source version of Meta-World (this https URL) that has full reproducibility of past results, is more technically ergonomic, and gives users more control over the tasks that are included in a task set. 

**Abstract (ZH)**: Meta-World：用于评估多任务和元强化学习代理的广泛使用基准，尽管自引入以来存在众多未记录的变化，阻碍了算法之间的公平比较，本工作致力于澄清文献中的结果，并利用Meta-World的过去版本提供多任务和元强化学习基准设计的见解。通过这一过程，我们发布了Meta-World的新开源版本（详见https://...），该版本具有对过去结果的完全可再现性、更高的技术便捷性和更多的任务控制权。 

---
# SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning 

**Title (ZH)**: SelfBudgeter: 自适应 token 分配以实现高效的LLM推理 

**Authors**: Zheng Li, Qingxiu Dong, Jingyuan Ma, Di Zhang, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.11274)  

**Abstract**: Recently, large reasoning models demonstrate exceptional performance on various tasks. However, reasoning models inefficiently over-process both trivial and complex queries, leading to resource waste and prolonged user latency. To address this challenge, we propose SelfBudgeter - a self-adaptive controllable reasoning strategy for efficient reasoning. Our approach adopts a dual-phase training paradigm: first, the model learns to pre-estimate the reasoning cost based on the difficulty of the query. Then, we introduce budget-guided GPRO for reinforcement learning, which effectively maintains accuracy while reducing output length. SelfBudgeter allows users to anticipate generation time and make informed decisions about continuing or interrupting the process. Furthermore, our method enables direct manipulation of reasoning length via pre-filling token budget. Experimental results demonstrate that SelfBudgeter can rationally allocate budgets according to problem complexity, achieving up to 74.47% response length compression on the MATH benchmark while maintaining nearly undiminished accuracy. 

**Abstract (ZH)**: SelfBudgeter——一种自适应可控的高效推理策略 

---
# LD-Scene: LLM-Guided Diffusion for Controllable Generation of Adversarial Safety-Critical Driving Scenarios 

**Title (ZH)**: LD-场景：LLM引导的扩散模型用于生成可控的 adversarial 安全关键驾驶场景 

**Authors**: Mingxing Peng, Yuting Xie, Xusen Guo, Ruoyu Yao, Hai Yang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.11247)  

**Abstract**: Ensuring the safety and robustness of autonomous driving systems necessitates a comprehensive evaluation in safety-critical scenarios. However, these safety-critical scenarios are rare and difficult to collect from real-world driving data, posing significant challenges to effectively assessing the performance of autonomous vehicles. Typical existing methods often suffer from limited controllability and lack user-friendliness, as extensive expert knowledge is essentially required. To address these challenges, we propose LD-Scene, a novel framework that integrates Large Language Models (LLMs) with Latent Diffusion Models (LDMs) for user-controllable adversarial scenario generation through natural language. Our approach comprises an LDM that captures realistic driving trajectory distributions and an LLM-based guidance module that translates user queries into adversarial loss functions, facilitating the generation of scenarios aligned with user queries. The guidance module integrates an LLM-based Chain-of-Thought (CoT) code generator and an LLM-based code debugger, enhancing the controllability and robustness in generating guidance functions. Extensive experiments conducted on the nuScenes dataset demonstrate that LD-Scene achieves state-of-the-art performance in generating realistic, diverse, and effective adversarial scenarios. Furthermore, our framework provides fine-grained control over adversarial behaviors, thereby facilitating more effective testing tailored to specific driving scenarios. 

**Abstract (ZH)**: 确保自动驾驶系统的安全性和鲁棒性需要在关键安全场景中进行全面评估。然而，这些关键安全场景在现实驾驶数据中罕见且难以收集，对有效评估自动驾驶车辆的性能构成了重大挑战。现有典型方法通常受限于有限的可控性且缺乏用户友好性，因为本质上需要大量的专家知识。为应对这些挑战，我们提出LD-Scene，这一新颖框架结合大型语言模型（LLMs）与潜在扩散模型（LDMs），通过自然语言实现用户可控的对抗场景生成。该方法包括一个捕捉现实驾驶轨迹分布的LDM以及一个基于LLM的指导模块，后者将用户查询转换为对抗损失函数，从而促进生成与用户查询一致的场景。指导模块集成了基于LLM的Chain-of-Thought（CoT）代码生成器和基于LLM的代码调试器，增强了生成指导函数的可控性和鲁棒性。在nuScenes数据集上的广泛实验表明，LD-Scene在生成现实、多样且有效的对抗场景方面达到了最先进的效果。此外，我们的框架提供了对抗行为的精细控制，从而有助于更加有效的针对特定驾驶场景的测试。 

---
# Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs 

**Title (ZH)**: PRM必要吗？问题求解RL隐式赋予LLMsPRM能力 

**Authors**: Zhangying Feng, Qianglong Chen, Ning Lu, Yongqian Li, Siqi Cheng, Shuangmu Peng, Duyu Tang, Shengcai Liu, Zhirui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11227)  

**Abstract**: The development of reasoning capabilities represents a critical frontier in large language models (LLMs) research, where reinforcement learning (RL) and process reward models (PRMs) have emerged as predominant methodological frameworks. Contrary to conventional wisdom, empirical evidence from DeepSeek-R1 demonstrates that pure RL training focused on mathematical problem-solving can progressively enhance reasoning abilities without PRM integration, challenging the perceived necessity of process supervision. In this study, we conduct a systematic investigation of the relationship between RL training and PRM capabilities. Our findings demonstrate that problem-solving proficiency and process supervision capabilities represent complementary dimensions of reasoning that co-evolve synergistically during pure RL training. In particular, current PRMs underperform simple baselines like majority voting when applied to state-of-the-art models such as DeepSeek-R1 and QwQ-32B. To address this limitation, we propose Self-PRM, an introspective framework in which models autonomously evaluate and rerank their generated solutions through self-reward mechanisms. Although Self-PRM consistently improves the accuracy of the benchmark (particularly with larger sample sizes), analysis exposes persistent challenges: The approach exhibits low precision (<10\%) on difficult problems, frequently misclassifying flawed solutions as valid. These analyses underscore the need for continued RL scaling to improve reward alignment and introspective accuracy. Overall, our findings suggest that PRM may not be essential for enhancing complex reasoning, as pure RL not only improves problem-solving skills but also inherently fosters robust PRM capabilities. We hope these findings provide actionable insights for building more reliable and self-aware complex reasoning models. 

**Abstract (ZH)**: 大型语言模型（LLMs）中推理能力的发展代表着一个关键前沿，其中强化学习（RL）和过程奖励模型（PRMs）已 emerge 为主要的方法论框架。与传统认识相反，DeepSeek-R1 的实验证据表明，专注于数学问题解决的纯 RL 训练可以在没有 PRM 结合的情况下逐步增强推理能力，挑战了过程监督的必要性。在本研究中，我们系统地调查了 RL 训练与 PRM 能力之间的关系。我们的研究结果表明，问题解决能力和过程监督能力是推理的互补维度，在纯 RL 训练过程中会协同进化。特别是，当前的 PRMs 在应用于 DeepSeek-R1 和 QwQ-32B 等先进模型时，表现不如简单的多数表决基线。为解决这一局限性，我们提出了 Self-PRM，这是一种反思框架，在该框架中，模型通过自奖励机制自主评估并重新排名其生成的解决方案。尽管 Self-PRM 在基准测试中的一致性改进（尤其是在大样本情况下），但分析揭示了持续的挑战：该方法在处理困难问题时精度很低（<10%），经常将无效的解决方案误分类为有效。这些分析强调了继续扩大 RL 规模以改进奖励对齐和内省准确性的需求。总体而言，我们的研究结果表明，PRM 并不一定对于增强复杂推理是必要的，因为纯 RL 不仅提高了问题解决技能，还内在地促进了稳健的 PRM 能力。我们希望这些发现能为构建更可靠和自我意识的复杂推理模型提供可操作的见解。 

---
# GLOVA: Global and Local Variation-Aware Analog Circuit Design with Risk-Sensitive Reinforcement Learning 

**Title (ZH)**: GLOVA：具有风险敏感型强化学习的全局与局部变异aware模拟电路设计 

**Authors**: Dongjun Kim, Junwoo Park, Chaehyeon Shin, Jaeheon Jung, Kyungho Shin, Seungheon Baek, Sanghyuk Heo, Woongrae Kim, Inchul Jeong, Joohwan Cho, Jongsun Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.11208)  

**Abstract**: Analog/mixed-signal circuit design encounters significant challenges due to performance degradation from process, voltage, and temperature (PVT) variations. To achieve commercial-grade reliability, iterative manual design revisions and extensive statistical simulations are required. While several studies have aimed to automate variation aware analog design to reduce time-to-market, the substantial mismatches in real-world wafers have not been thoroughly addressed. In this paper, we present GLOVA, an analog circuit sizing framework that effectively manages the impact of diverse random mismatches to improve robustness against PVT variations. In the proposed approach, risk-sensitive reinforcement learning is leveraged to account for the reliability bound affected by PVT variations, and ensemble-based critic is introduced to achieve sample-efficient learning. For design verification, we also propose $\mu$-$\sigma$ evaluation and simulation reordering method to reduce simulation costs of identifying failed designs. GLOVA supports verification through industrial-level PVT variation evaluation methods, including corner simulation as well as global and local Monte Carlo (MC) simulations. Compared to previous state-of-the-art variation-aware analog sizing frameworks, GLOVA achieves up to 80.5$\times$ improvement in sample efficiency and 76.0$\times$ reduction in time. 

**Abstract (ZH)**: 模拟/混合信号电路设计面临着严重的挑战，由于工艺、电压和温度（PVT）变化导致性能下降。为了实现商业级别的可靠性，需要进行迭代的手动设计修订和大量的统计模拟。尽管有多项研究致力于通过自动化变异感知模拟设计来缩短上市时间，但在实际晶圆上的显著不匹配问题尚未得到充分解决。在这篇论文中，我们提出了GLOVA，一种有效的模拟电路缩放框架，用于管理各种随机不匹配的影響，以提高对PVT变化的鲁棒性。在所提出的方法中，风险敏感的强化学习被用于考虑PVT变化影响的可靠性边界，引入基于集成的评论者以实现样本高效学习。在设计验证方面，我们还提出了$\mu$-$\sigma$评估和仿真排序方法，以降低识别失败设计的仿真成本。GLOVA支持通过工业级PVT变化评估方法进行验证，包括角仿真以及全局和局部蒙特卡洛（MC）仿真。与之前最先进的变异感知模拟缩放框架相比，GLOVA在样本效率上实现了最高80.5倍的改进，并在时间上实现了76.0倍的减少。 

---
# Multi-Modal Multi-Task (M3T) Federated Foundation Models for Embodied AI: Potentials and Challenges for Edge Integration 

**Title (ZH)**: 面向身体化人工智能的多模态多任务（M3T）联邦基础模型：边缘集成的潜力与挑战 

**Authors**: Kasra Borazjani, Payam Abdisarabshali, Fardis Nadimi, Naji Khosravan, Minghui Liwang, Xianbin Wang, Yiguang Hong, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2505.11191)  

**Abstract**: As embodied AI systems become increasingly multi-modal, personalized, and interactive, they must learn effectively from diverse sensory inputs, adapt continually to user preferences, and operate safely under resource and privacy constraints. These challenges expose a pressing need for machine learning models capable of swift, context-aware adaptation while balancing model generalization and personalization. Here, two methods emerge as suitable candidates, each offering parts of these capabilities: Foundation Models (FMs) provide a pathway toward generalization across tasks and modalities, whereas Federated Learning (FL) offers the infrastructure for distributed, privacy-preserving model updates and user-level model personalization. However, when used in isolation, each of these approaches falls short of meeting the complex and diverse capability requirements of real-world embodied environments. In this vision paper, we introduce Federated Foundation Models (FFMs) for embodied AI, a new paradigm that unifies the strengths of multi-modal multi-task (M3T) FMs with the privacy-preserving distributed nature of FL, enabling intelligent systems at the wireless edge. We collect critical deployment dimensions of FFMs in embodied AI ecosystems under a unified framework, which we name "EMBODY": Embodiment heterogeneity, Modality richness and imbalance, Bandwidth and compute constraints, On-device continual learning, Distributed control and autonomy, and Yielding safety, privacy, and personalization. For each, we identify concrete challenges and envision actionable research directions. We also present an evaluation framework for deploying FFMs in embodied AI systems, along with the associated trade-offs. 

**Abstract (ZH)**: 面向物联网边缘的联合基础模型：统一多模态多任务基础模型与联邦学习的优势 

---
# Can Global XAI Methods Reveal Injected Bias in LLMs? SHAP vs Rule Extraction vs RuleSHAP 

**Title (ZH)**: 全球可解释性人工智能方法能否揭示LLMs中的注入偏差？SHAP vs 规则提取 vs RuleSHAP 

**Authors**: Francesco Sovrano  

**Link**: [PDF](https://arxiv.org/pdf/2505.11189)  

**Abstract**: Generative AI systems can help spread information but also misinformation and biases, potentially undermining the UN Sustainable Development Goals (SDGs). Explainable AI (XAI) aims to reveal the inner workings of AI systems and expose misbehaviours or biases. However, current XAI tools, built for simpler models, struggle to handle the non-numerical nature of large language models (LLMs). This paper examines the effectiveness of global XAI methods, such as rule-extraction algorithms and SHAP, in detecting bias in LLMs. To do so, we first show a text-to-ordinal mapping strategy to convert non-numerical inputs/outputs into numerical features, enabling these tools to identify (some) misinformation-related biases in LLM-generated content. Then, we inject non-linear biases of varying complexity (univariate, conjunctive, and non-convex) into widespread LLMs like ChatGPT and Llama via system instructions, using global XAI methods to detect them. This way, we found that RuleFit struggles with conjunctive and non-convex biases, while SHAP can approximate conjunctive biases but cannot express them as actionable rules. Hence, we introduce RuleSHAP, a global rule extraction algorithm combining SHAP and RuleFit to detect more non-univariate biases, improving injected bias detection over RuleFit by +94% (MRR@1) on average. 

**Abstract (ZH)**: 生成式AI系统可以传播信息但也可能传播误导信息和偏见，这可能削弱联合国可持续发展目标（SDGs）。可解释AI（XAI）旨在揭示AI系统的内部工作机制并揭示不当行为或偏见。然而，当前用于简单模型的XAI工具难以处理大型语言模型（LLMs）的非数值性质。本文探讨了全球化XAI方法，如规则提取算法和SHAP，在检测LLM中的偏见方面的有效性。为此，我们首先展示了文本到序数映射策略，将非数值输入/输出转换为数值特征，使这些工具能够识别LLM生成内容中的部分误导相关信息偏见。然后，我们通过系统指令向广泛使用的LLM（如ChatGPT和Llama）注入复杂程度不同的非线性偏见（单一变量、联合和非凸），并使用全球化XAI方法检测它们。我们发现，RuleFit在检测联合和非凸偏见方面存在一定困难，而SHAP可以近似检测联合偏见但无法将其表达为可操作规则。因此，我们引入了一种结合SHAP和RuleFit的全局规则提取算法RuleSHAP，用于检测更多非单一变量偏见，平均在MRR@1方面比RuleFit提高了94%的注入偏见检测效果。 

---
# Feasibility with Language Models for Open-World Compositional Zero-Shot Learning 

**Title (ZH)**: 语言模型在开放世界组合零样本学习中的可行性 

**Authors**: Jae Myung Kim, Stephan Alaniz, Cordelia Schmid, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2505.11181)  

**Abstract**: Humans can easily tell if an attribute (also called state) is realistic, i.e., feasible, for an object, e.g. fire can be hot, but it cannot be wet. In Open-World Compositional Zero-Shot Learning, when all possible state-object combinations are considered as unseen classes, zero-shot predictors tend to perform poorly. Our work focuses on using external auxiliary knowledge to determine the feasibility of state-object combinations. Our Feasibility with Language Model (FLM) is a simple and effective approach that leverages Large Language Models (LLMs) to better comprehend the semantic relationships between states and objects. FLM involves querying an LLM about the feasibility of a given pair and retrieving the output logit for the positive answer. To mitigate potential misguidance of the LLM given that many of the state-object compositions are rare or completely infeasible, we observe that the in-context learning ability of LLMs is essential. We present an extensive study identifying Vicuna and ChatGPT as best performing, and we demonstrate that our FLM consistently improves OW-CZSL performance across all three benchmarks. 

**Abstract (ZH)**: 基于外部辅助知识的开集组成零样本学习中属性-对象组合的可行性判断 

---
# Reinforcement Learning for AMR Charging Decisions: The Impact of Reward and Action Space Design 

**Title (ZH)**: 基于强化学习的AMR充电决策：奖励和动作空间设计的影响 

**Authors**: Janik Bischoff, Alexandru Rinciog, Anne Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11136)  

**Abstract**: We propose a novel reinforcement learning (RL) design to optimize the charging strategy for autonomous mobile robots in large-scale block stacking warehouses. RL design involves a wide array of choices that can mostly only be evaluated through lengthy experimentation. Our study focuses on how different reward and action space configurations, ranging from flexible setups to more guided, domain-informed design configurations, affect the agent performance. Using heuristic charging strategies as a baseline, we demonstrate the superiority of flexible, RL-based approaches in terms of service times. Furthermore, our findings highlight a trade-off: While more open-ended designs are able to discover well-performing strategies on their own, they may require longer convergence times and are less stable, whereas guided configurations lead to a more stable learning process but display a more limited generalization potential. Our contributions are threefold. First, we extend SLAPStack, an open-source, RL-compatible simulation-framework to accommodate charging strategies. Second, we introduce a novel RL design for tackling the charging strategy problem. Finally, we introduce several novel adaptive baseline heuristics and reproducibly evaluate the design using a Proximal Policy Optimization agent and varying different design configurations, with a focus on reward. 

**Abstract (ZH)**: 我们提出了一种新颖的强化学习（RL）设计，以优化自主移动机器人在大规模块堆叠仓库中的充电策略。我们的研究重点在于不同奖励和动作空间配置，从灵活设置到更引导性的、领域导向的设计配置，对代理性能的影响。以启发式充电策略为基准，我们展示了基于RL的方法在服务时间方面的优越性。此外，我们的研究结果揭示了一种权衡：虽然更为开放的设计能够自主发现高性能策略，但可能需要更长的收敛时间且稳定性较差，而引导性配置则促进了更稳定的训练过程，但其泛化能力更为有限。我们的贡献包括三个部分。首先，我们将SLAPStack扩展为一种兼容RL的开源仿真框架，以包含充电策略。其次，我们引入了一种新的RL设计以解决充电策略问题。最后，我们引入了若干新的自适应基准启发式方法，并通过使用Proximal Policy Optimization代理和不同的设计配置进行可重复评估，重点在于奖励。 

---
# Scalability of Reinforcement Learning Methods for Dispatching in Semiconductor Frontend Fabs: A Comparison of Open-Source Models with Real Industry Datasets 

**Title (ZH)**: 半导体前端fab中调度方法的强化学习方法可扩展性研究：开源模型与实际工业数据的对比 

**Authors**: Patrick Stöckermann, Henning Südfeld, Alessandro Immordino, Thomas Altenmüller, Marc Wegmann, Martin Gebser, Konstantin Schekotihin, Georg Seidel, Chew Wye Chan, Fei Fei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11135)  

**Abstract**: Benchmark datasets are crucial for evaluating approaches to scheduling or dispatching in the semiconductor industry during the development and deployment phases. However, commonly used benchmark datasets like the Minifab or SMT2020 lack the complex details and constraints found in real-world scenarios. To mitigate this shortcoming, we compare open-source simulation models with a real industry dataset to evaluate how optimization methods scale with different levels of complexity. Specifically, we focus on Reinforcement Learning methods, performing optimization based on policy-gradient and Evolution Strategies. Our research provides insights into the effectiveness of these optimization methods and their applicability to realistic semiconductor frontend fab simulations. We show that our proposed Evolution Strategies-based method scales much better than a comparable policy-gradient-based approach. Moreover, we identify the selection and combination of relevant bottleneck tools to control by the agent as crucial for an efficient optimization. For the generalization across different loading scenarios and stochastic tool failure patterns, we achieve advantages when utilizing a diverse training dataset. While the overall approach is computationally expensive, it manages to scale well with the number of CPU cores used for training. For the real industry dataset, we achieve an improvement of up to 4% regarding tardiness and up to 1% regarding throughput. For the less complex open-source models Minifab and SMT2020, we observe double-digit percentage improvement in tardiness and single digit percentage improvement in throughput by use of Evolution Strategies. 

**Abstract (ZH)**: 基准数据集对于半导体行业调度或分派方法的开发和部署阶段评估至关重要。然而，常用的基准数据集如Minifab或SMT2020缺乏实际场景中的复杂细节和约束。为了弥补这一不足，我们将开源仿真模型与真实工业数据集进行比较，评估优化方法在不同复杂度水平下的可扩展性。具体而言，我们专注于基于策略梯度和进化策略的强化学习方法。我们的研究提供了这些优化方法有效性和适用性的见解，特别是在实际半导体前端晶圆厂仿真中的应用。我们证明基于进化策略的方法在可扩展性方面明显优于基于策略梯度的方法。此外，我们发现代理控制的相关瓶颈工具的选择和组合对于高效的优化至关重要。为了在不同装载场景和随机工具故障模式下实现泛化，使用多样化训练数据集可以获得优势。尽管总体方法计算成本较高，但它能够很好地与用于训练的CPU内核数量扩展。对于真实工业数据集，我们关于延误的改进高达4%，关于吞吐量的改进高达1%。对于较简单的开源模型Minifab和SMT2020，我们通过使用进化策略观察到延误的多位数百分比改进和吞吐量的一位数百分比改进。 

---
# Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for Formulaic Factor Mining 

**Title (ZH)**: 穿越Alpha丛林：基于LLM的MCTS框架在公因子挖掘中的应用 

**Authors**: Yu Shi, Yitong Duan, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11122)  

**Abstract**: Alpha factor mining is pivotal in quantitative investment for identifying predictive signals from complex financial data. While traditional formulaic alpha mining relies on human expertise, contemporary automated methods, such as those based on genetic programming or reinforcement learning, often suffer from search inefficiency or yield poorly interpretable alpha factors. This paper introduces a novel framework that integrates Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS) to overcome these limitations. Our approach leverages the LLM's instruction-following and reasoning capability to iteratively generate and refine symbolic alpha formulas within an MCTS-driven exploration. A key innovation is the guidance of MCTS exploration by rich, quantitative feedback from financial backtesting of each candidate factor, enabling efficient navigation of the vast search space. Furthermore, a frequent subtree avoidance mechanism is introduced to bolster search efficiency and alpha factor performance. Experimental results on real-world stock market data demonstrate that our LLM-based framework outperforms existing methods by mining alphas with superior predictive accuracy, trading performance, and improved interpretability, while offering a more efficient solution for formulaic alpha mining. 

**Abstract (ZH)**: 基于大型语言模型和蒙特卡洛树搜索的alpha因子挖掘新框架 

---
# Predicting Student Dropout Risk With A Dual-Modal Abrupt Behavioral Changes Approach 

**Title (ZH)**: 基于双模态突变行为变化的方法预测学生辍学风险 

**Authors**: Jiabei Cheng, Zhen-Qun Yang, Jiannong Cao, Yu Yang, Xinzhe Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11119)  

**Abstract**: Timely prediction of students at high risk of dropout is critical for early intervention and improving educational outcomes. However, in offline educational settings, poor data quality, limited scale, and high heterogeneity often hinder the application of advanced machine learning models. Furthermore, while educational theories provide valuable insights into dropout phenomena, the lack of quantifiable metrics for key indicators limits their use in data-driven modeling. Through data analysis and a review of educational literature, we identified abrupt changes in student behavior as key early signals of dropout risk. To address this, we propose the Dual-Modal Multiscale Sliding Window (DMSW) Model, which integrates academic performance and behavioral data to dynamically capture behavior patterns using minimal data. The DMSW model improves prediction accuracy by 15% compared to traditional methods, enabling educators to identify high-risk students earlier, provide timely support, and foster a more inclusive learning environment. Our analysis highlights key behavior patterns, offering practical insights for preventive strategies and tailored support. These findings bridge the gap between theory and practice in dropout prediction, giving educators an innovative tool to enhance student retention and outcomes. 

**Abstract (ZH)**: 及时预测高风险辍学学生对于早期干预和提高教育成果至关重要。然而，在线教育环境中，数据质量差、规模有限以及高度异质性常常阻碍先进机器学习模型的应用。此外，尽管教育理论为辍学现象提供了宝贵见解，但由于缺乏可量化的关键指标的度量标准，限制了其在数据驱动建模中的应用。通过数据分析和教育文献综述，我们确定了学生行为的突变是辍学风险的早期关键信号。为此，我们提出了一种双模态多尺度滑动窗口（DMSW）模型，该模型整合了学业表现和行为数据，通过最少的数据动态捕捉行为模式。与传统方法相比，DMSW模型将预测准确性提高了15%，使教育者能够更早地识别高风险学生，提供及时支持，并促进更具包容性的学习环境。我们的分析揭示了关键行为模式，提供了预防策略和个性化支持的实用见解。这些发现弥合了辍学预测理论与实践之间的差距，为教育者提供了一个创新工具，以增强学生的留存率和成果。 

---
# Group Think: Multiple Concurrent Reasoning Agents Collaborating at Token Level Granularity 

**Title (ZH)**: 群体思维：多个并发推理代理在 token 级别粒度上协作 

**Authors**: Chan-Jan Hsu, Davide Buffelli, Jamie McGowan, Feng-Ting Liao, Yi-Chang Chen, Sattar Vakili, Da-shan Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11107)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated the power of reasoning through self-generated chains of thought. Multiple reasoning agents can collaborate to raise joint reasoning quality above individual outcomes. However, such agents typically interact in a turn-based manner, trading increased latency for improved quality. In this paper, we propose Group Think--a single LLM that acts as multiple concurrent reasoning agents, or thinkers. With shared visibility into each other's partial generation progress, Group Think introduces a new concurrent-reasoning paradigm in which multiple reasoning trajectories adapt dynamically to one another at the token level. For example, a reasoning thread may shift its generation mid-sentence upon detecting that another thread is better positioned to continue. This fine-grained, token-level collaboration enables Group Think to reduce redundant reasoning and improve quality while achieving significantly lower latency. Moreover, its concurrent nature allows for efficient utilization of idle computational resources, making it especially suitable for edge inference, where very small batch size often underutilizes local~GPUs. We give a simple and generalizable modification that enables any existing LLM to perform Group Think on a local GPU. We also present an evaluation strategy to benchmark reasoning latency and empirically demonstrate latency improvements using open-source LLMs that were not explicitly trained for Group Think. We hope this work paves the way for future LLMs to exhibit more sophisticated and more efficient collaborative behavior for higher quality generation. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）的进展展示了通过自我生成的推理链进行推理的力量。多个推理代理可以协作以提高联合推理质量，超过个体结果。然而，这些代理通常以轮询方式交互，增加延迟以提高质量。在本文中，我们提出了一种名为“Group Think”的方法——这是一种充当多个并发推理代理或思考者的大规模语言模型。通过共享对方部分生成进度的可见性，Group Think 引入了一种新的并发推理范式，在此范式中，多个推理轨迹在标记级别上动态适应彼此。例如，推理线程可能会在检测到另一个线程更适合继续生成时，在句子中更改其生成方向。这种细粒度的标记级别协作使Group Think能够在减少冗余推理和提高质量的同时实现显着降低延迟。此外，其并发性质使得能够有效地利用闲置的计算资源，使其特别适合边缘推理，在边缘推理中，非常小的批量大小往往未能充分利用本地GPU。我们提出了一种简单且可推广的修改，使任何现有的大规模语言模型能够在本地GPU上执行Group Think。我们还提出了一种评估策略，用于基准测试推理延迟，并使用未专门训练进行Group Think的开源大规模语言模型来实证展示延迟改进。我们希望这项工作为未来的大型语言模型铺平了道路，使其能够表现出更复杂的更具效率的协作行为，以获得更高质量的生成。 

---
# Analysis of Customer Journeys Using Prototype Detection and Counterfactual Explanations for Sequential Data 

**Title (ZH)**: 使用原型检测和反事实解释分析客户旅程sequential数据中的客户旅程 

**Authors**: Keita Kinjo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11086)  

**Abstract**: Recently, the proliferation of omni-channel platforms has attracted interest in customer journeys, particularly regarding their role in developing marketing strategies. However, few efforts have been taken to quantitatively study or comprehensively analyze them owing to the sequential nature of their data and the complexity involved in analysis. In this study, we propose a novel approach comprising three steps for analyzing customer journeys. First, the distance between sequential data is defined and used to identify and visualize representative sequences. Second, the likelihood of purchase is predicted based on this distance. Third, if a sequence suggests no purchase, counterfactual sequences are recommended to increase the probability of a purchase using a proposed method, which extracts counterfactual explanations for sequential data. A survey was conducted, and the data were analyzed; the results revealed that typical sequences could be extracted, and the parts of those sequences important for purchase could be detected. We believe that the proposed approach can support improvements in various marketing activities. 

**Abstract (ZH)**: 最近，全渠道平台的普及引起了对客户旅程的兴趣，特别是在制定营销策略中的作用。然而，由于其数据的序列性质和分析的复杂性，鲜有人对其进行定量研究或全面分析。在本研究中，我们提出了一种包含三个步骤的新方法来分析客户旅程。首先，定义序列数据之间的距离并用于识别和可视化代表性序列。其次，基于此距离预测购买的可能性。第三，如果序列表明无购买，我们将推荐反事实序列以提高购买概率，该方法提取了序列数据的反事实解释。我们进行了调查并分析了数据，结果显示可以提取典型的序列，并能够检测这些序列中与购买相关的重要部分。我们认为所提出的方法可以支持各类营销活动的改进。 

---
# A Multi-modal Fusion Network for Terrain Perception Based on Illumination Aware 

**Title (ZH)**: 基于照明感知的多模态融合网络用于地形感知 

**Authors**: Rui Wang, Shichun Yang, Yuyi Chen, Zhuoyang Li, Zexiang Tong, Jianyi Xu, Jiayi Lu, Xinjie Feng, Yaoguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11066)  

**Abstract**: Road terrains play a crucial role in ensuring the driving safety of autonomous vehicles (AVs). However, existing sensors of AVs, including cameras and Lidars, are susceptible to variations in lighting and weather conditions, making it challenging to achieve real-time perception of road conditions. In this paper, we propose an illumination-aware multi-modal fusion network (IMF), which leverages both exteroceptive and proprioceptive perception and optimizes the fusion process based on illumination features. We introduce an illumination-perception sub-network to accurately estimate illumination features. Moreover, we design a multi-modal fusion network which is able to dynamically adjust weights of different modalities according to illumination features. We enhance the optimization process by pre-training of the illumination-perception sub-network and incorporating illumination loss as one of the training constraints. Extensive experiments demonstrate that the IMF shows a superior performance compared to state-of-the-art methods. The comparison results with single modality perception methods highlight the comprehensive advantages of multi-modal fusion in accurately perceiving road terrains under varying lighting conditions. Our dataset is available at: this https URL. 

**Abstract (ZH)**: 道路地形在确保自动驾驶车辆（AVs）行驶安全中起着重要作用。然而，现有AV传感器，包括相机和激光雷达，对光照和天气条件的变化较为敏感，使得实时感知道路条件具有挑战性。本文提出了一种照明感知多模态融合网络（IMF），利用外部和内部感知，并基于照明特征优化融合过程。我们引入了一个照明感知子网络来准确估计照明特征。此外，我们设计了一个多模态融合网络，能够根据照明特征动态调整不同模态的权重。我们通过照明感知子网络的预训练和将照明损失纳入训练约束来增强优化过程。广泛的实验表明，IMF展示了优于现有先进方法的性能。与单一模态感知方法的对比结果突显了在变化光照条件下多模态融合在准确感知道路地形方面的综合优势。我们的数据集可在以下链接获取：this https URL。 

---
# Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction 

**Title (ZH)**: 三思而后行：通过思维纠正提升代理行为安全性 

**Authors**: Changyue Jiang, Xudong Pan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11063)  

**Abstract**: LLM-based autonomous agents possess capabilities such as reasoning, tool invocation, and environment interaction, enabling the execution of complex multi-step tasks. The internal reasoning process, i.e., thought, of behavioral trajectory significantly influences tool usage and subsequent actions but can introduce potential risks. Even minor deviations in the agent's thought may trigger cascading effects leading to irreversible safety incidents. To address the safety alignment challenges in long-horizon behavioral trajectories, we propose Thought-Aligner, a plug-in dynamic thought correction module. Utilizing a lightweight and resource-efficient model, Thought-Aligner corrects each high-risk thought on the fly before each action execution. The corrected thought is then reintroduced to the agent, ensuring safer subsequent decisions and tool interactions. Importantly, Thought-Aligner modifies only the reasoning phase without altering the underlying agent framework, making it easy to deploy and widely applicable to various agent frameworks. To train the Thought-Aligner model, we construct an instruction dataset across ten representative scenarios and simulate ReAct execution trajectories, generating 5,000 diverse instructions and more than 11,400 safe and unsafe thought pairs. The model is fine-tuned using contrastive learning techniques. Experiments across three agent safety benchmarks involving 12 different LLMs demonstrate that Thought-Aligner raises agent behavioral safety from approximately 50% in the unprotected setting to 90% on average. Additionally, Thought-Aligner maintains response latency below 100ms with minimal resource usage, demonstrating its capability for efficient deployment, broad applicability, and timely responsiveness. This method thus provides a practical dynamic safety solution for the LLM-based agents. 

**Abstract (ZH)**: 基于LLM的自主代理动态思维对齐模块 

---
# GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning 

**Title (ZH)**: GuardReasoner-VL：通过强化推理保障VLMs安全 

**Authors**: Yue Liu, Shengfang Zhai, Mingzhe Du, Yulin Chen, Tri Cao, Hongcheng Gao, Cheng Wang, Xinfeng Li, Kun Wang, Junfeng Fang, Jiaheng Zhang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2505.11049)  

**Abstract**: To enhance the safety of VLMs, this paper introduces a novel reasoning-based VLM guard model dubbed GuardReasoner-VL. The core idea is to incentivize the guard model to deliberatively reason before making moderation decisions via online RL. First, we construct GuardReasoner-VLTrain, a reasoning corpus with 123K samples and 631K reasoning steps, spanning text, image, and text-image inputs. Then, based on it, we cold-start our model's reasoning ability via SFT. In addition, we further enhance reasoning regarding moderation through online RL. Concretely, to enhance diversity and difficulty of samples, we conduct rejection sampling followed by data augmentation via the proposed safety-aware data concatenation. Besides, we use a dynamic clipping parameter to encourage exploration in early stages and exploitation in later stages. To balance performance and token efficiency, we design a length-aware safety reward that integrates accuracy, format, and token cost. Extensive experiments demonstrate the superiority of our model. Remarkably, it surpasses the runner-up by 19.27% F1 score on average. We release data, code, and models (3B/7B) of GuardReasoner-VL at this https URL 

**Abstract (ZH)**: 为了增强VLMs的安全性，本文提出了一种名为GuardReasoner-VL的新颖推理驱动VLM守护模型。核心思想是通过在线RL激励守护模型在做出内容审核决策之前进行慎重的推理。首先，我们构建了包含123K样本和631K推理步骤的推理语料库GuardReasoner-VLTrain，涵盖了文本、图像和图文输入。然后，基于此，我们通过SFT冷启动模型的推理能力，并进一步通过在线RL增强内容审核中的推理能力。具体地，为了增加样本的多样性和难度，我们进行了拒绝采样，并通过提出的安全感知数据拼接进行数据增强。此外，我们使用动态截断参数鼓励早期探索和后期利用。为了平衡性能和标记效率，我们设计了一种结合准确率、格式和标记成本的安全奖励机制。详尽的实验表明了模型的优越性，且平均F1得分比第二名高19.27%。我们在此网址发布GuardReasoner-VL的数据、代码和模型（3B/7B）：[链接]。 

---
# Most General Explanations of Tree Ensembles 

**Title (ZH)**: 树ensemble的最一般解释 

**Authors**: Yacine Izza, Alexey Ignatiev, Joao Marques-Silva, Peter J. Stuckey  

**Link**: [PDF](https://arxiv.org/pdf/2505.10991)  

**Abstract**: Explainable Artificial Intelligence (XAI) is critical for attaining trust in the operation of AI systems. A key question of an AI system is ``why was this decision made this way''. Formal approaches to XAI use a formal model of the AI system to identify abductive explanations. While abductive explanations may be applicable to a large number of inputs sharing the same concrete values, more general explanations may be preferred for numeric inputs. So-called inflated abductive explanations give intervals for each feature ensuring that any input whose values fall withing these intervals is still guaranteed to make the same prediction. Inflated explanations cover a larger portion of the input space, and hence are deemed more general explanations. But there can be many (inflated) abductive explanations for an instance. Which is the best? In this paper, we show how to find a most general abductive explanation for an AI decision. This explanation covers as much of the input space as possible, while still being a correct formal explanation of the model's behaviour. Given that we only want to give a human one explanation for a decision, the most general explanation gives us the explanation with the broadest applicability, and hence the one most likely to seem sensible. (The paper has been accepted at IJCAI2025 conference.) 

**Abstract (ZH)**: 解释可理解的人工智能（XAI）对于获得对AI系统操作的信任至关重要。一个关键问题是如何从AI系统中解释“为何作出了这种决策”。形式化的XAI方法利用AI系统的形式模型来识别 abduction 解释。虽然 abduction 解释可能适用于具有相同具体值的大量输入，但对于数值输入而言，更一般的解释可能更受青睐。所谓的膨胀 abduction 解释为每个特征提供区间，确保任何值落在这些区间内的输入依然能得到相同的预测。膨胀解释覆盖了较大的输入空间，因此被认为是更一般的解释。但一个实例可能有多个（膨胀的） abduction 解释。哪个是最好的？本文展示了如何找到一个最一般的 abduction 解释，该解释覆盖尽可能多的输入空间，同时仍然是正确的形式模型行为解释。鉴于我们只想给人类提供一个关于决策的解释，最一般的解释提供了最具广泛适用性的解释，因此最有可能显得合理。（该论文已被接受参加IJCAI2025会议。） 

---
# RAGSynth: Synthetic Data for Robust and Faithful RAG Component Optimization 

**Title (ZH)**: RAGSynth：稳健且忠实的RAG组件优化合成数据 

**Authors**: Haiyang Shen, Hang Yan, Zhongshi Xing, Mugeng Liu, Yue Li, Zhiyang Chen, Yuxiang Wang, Jiuzheng Wang, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.10989)  

**Abstract**: RAG can enhance the performance of LLMs on knowledge-intensive tasks. Various RAG paradigms, including vanilla, planning-based, and iterative RAG, are built upon 2 cores: the retriever, which should robustly select relevant documents across complex queries, and the generator, which should faithfully synthesize responses. However, existing retrievers rely heavily on public knowledge and struggle with queries of varying logical complexity and clue completeness, while generators frequently face fidelity problems. In this work, we introduce RAGSynth, a framework that includes a data construction modeling and a corresponding synthetic data generation implementation, designed to optimize retriever robustness and generator fidelity. Additionally, we present SynthBench, a benchmark encompassing 8 domain-specific documents across 4 domains, featuring diverse query complexities, clue completeness, and fine-grained citation granularity. Leveraging RAGSynth, we generate a large-scale synthetic dataset, including single and multi-hop. Extensive experiments demonstrate that the synthetic data significantly improves the robustness of the retrievers and the fidelity of the generators. Additional evaluations confirm that RAGSynth can also generalize well across different domains. By integrating the optimized retrievers into various RAG paradigms, we consistently observe enhanced RAG system performance. We have open-sourced the implementation on this https URL. 

**Abstract (ZH)**: RAG可以增强大语言模型在知识密集型任务中的性能。RAGSynth框架及其在优化检索器鲁棒性和生成器信实性方面的应用。SynthBench：一个涵盖四个领域八个专业文档的基准测试，包含多种查询复杂度、线索完整性和精细引文粒度。通过RAGSynth生成的大规模合成数据集显著提高了检索器的鲁棒性和生成器的信实性。RAGSynth在不同领域的泛化能力也得到了验证。通过将优化后的检索器集成到各种RAG范式中，我们观察到了RAG系统性能的一致提升。我们已开源该实现。 

---
# DRL-Based Injection Molding Process Parameter Optimization for Adaptive and Profitable Production 

**Title (ZH)**: 基于DRL的注射 molding 工艺参数优化以实现自适应和盈利生产 

**Authors**: Joon-Young Kim, Jecheon Yu, Heekyu Kim, Seunghwa Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10988)  

**Abstract**: Plastic injection molding remains essential to modern manufacturing. However, optimizing process parameters to balance product quality and profitability under dynamic environmental and economic conditions remains a persistent challenge. This study presents a novel deep reinforcement learning (DRL)-based framework for real-time process optimization in injection molding, integrating product quality and profitability into the control objective. A profit function was developed to reflect real-world manufacturing costs, incorporating resin, mold wear, and electricity prices, including time-of-use variations. Surrogate models were constructed to predict product quality and cycle time, enabling efficient offline training of DRL agents using soft actor-critic (SAC) and proximal policy optimization (PPO) algorithms. Experimental results demonstrate that the proposed DRL framework can dynamically adapt to seasonal and operational variations, consistently maintaining product quality while maximizing profit. Compared to traditional optimization methods such as genetic algorithms, the DRL models achieved comparable economic performance with up to 135x faster inference speeds, making them well-suited for real-time applications. The framework's scalability and adaptability highlight its potential as a foundation for intelligent, data-driven decision-making in modern manufacturing environments. 

**Abstract (ZH)**: 基于深度强化学习的注射模具实时优化框架：平衡产品质量与盈利能力 

---
# Facets in Argumentation: A Formal Approach to Argument Significance 

**Title (ZH)**: 论辩要素：论据显著性的形式化方法 

**Authors**: Johannes Fichte, Nicolas Fröhlich, Markus Hecher, Victor Lagerkvist, Yasir Mahmood, Arne Meier, Jonathan Persson  

**Link**: [PDF](https://arxiv.org/pdf/2505.10982)  

**Abstract**: Argumentation is a central subarea of Artificial Intelligence (AI) for modeling and reasoning about arguments. The semantics of abstract argumentation frameworks (AFs) is given by sets of arguments (extensions) and conditions on the relationship between them, such as stable or admissible. Today's solvers implement tasks such as finding extensions, deciding credulous or skeptical acceptance, counting, or enumerating extensions. While these tasks are well charted, the area between decision, counting/enumeration and fine-grained reasoning requires expensive reasoning so far. We introduce a novel concept (facets) for reasoning between decision and enumeration. Facets are arguments that belong to some extensions (credulous) but not to all extensions (skeptical). They are most natural when a user aims to navigate, filter, or comprehend the significance of specific arguments, according to their needs. We study the complexity and show that tasks involving facets are much easier than counting extensions. Finally, we provide an implementation, and conduct experiments to demonstrate feasibility. 

**Abstract (ZH)**: 论辩是人工智能（AI）中的一个核心子领域，用于建模和推理论辩。抽象论辩框架（AFs）的语义由论辩集（扩展）及其之间的关系条件给出，例如稳定或可接受。当前的求解器实现查找扩展、决定性的或怀疑性的接受、计数或枚举扩展等功能。尽管这些任务已经很清楚，但在决策、计数/枚举与精细推理之间的区域仍需要昂贵的推理。我们引入了一个新的概念（切面），用于决策与枚举之间的推理。切面是属于某些扩展（确信的）但不属于所有扩展（怀疑的）的论辩。当用户旨在根据其需求导航、过滤或理解特定论辩的意义时，它们是最自然的。我们研究了切面任务的复杂性，表明涉及切面的任务比计数扩展要容易得多。最后，我们提供了一个实现，并进行实验以证明可行性。 

---
# Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory 

**Title (ZH)**: 重思提示策略在LLM测试时缩放中的作用：从概率论视角出发 

**Authors**: Yexiang Liu, Zekun Li, Zhi Fang, Nan Xu, Ran He, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10981)  

**Abstract**: Recently, scaling test-time compute on Large Language Models (LLM) has garnered wide attention. However, there has been limited investigation of how various reasoning prompting strategies perform as scaling. In this paper, we focus on a standard and realistic scaling setting: majority voting. We systematically conduct experiments on 6 LLMs $\times$ 8 prompting strategies $\times$ 6 benchmarks. Experiment results consistently show that as the sampling time and computational overhead increase, complicated prompting strategies with superior initial performance gradually fall behind simple Chain-of-Thought. We analyze this phenomenon and provide theoretical proofs. Additionally, we propose a method according to probability theory to quickly and accurately predict the scaling performance and select the best strategy under large sampling times without extra resource-intensive inference in practice. It can serve as the test-time scaling law for majority voting. Furthermore, we introduce two ways derived from our theoretical analysis to significantly improve the scaling performance. We hope that our research can promote to re-examine the role of complicated prompting, unleash the potential of simple prompting strategies, and provide new insights for enhancing test-time scaling performance. 

**Abstract (ZH)**: 最近，大型语言模型测试时计算量的扩展引起了广泛关注。然而，对于各种推理提示策略随扩展规模的变化性能研究有限。本文关注一种标准且实际的扩展设置：多数投票。我们系统地在6种大型语言模型$\times$8种提示策略$\times$6种基准上进行了实验。实验结果一致显示，随着采样时间和计算开销的增加，初始表现优异但复杂的提示策略逐渐落后于简单的链式思考。我们分析了这一现象并提供了理论证明。此外，我们根据概率论提出了一种方法，在大规模采样时不进行额外的资源密集型推理即可快速准确地预测扩展性能并选择最佳策略。它可以作为多数投票的测试时扩展定律。进一步地，我们介绍了两种源自理论分析的方法，以显著提高扩展性能。我们希望我们的研究能够促进重新审视复杂提示的作用，释放简单提示策略的潜力，并为提高测试时扩展性能提供新的见解。 

---
# MPS-Prover: Advancing Stepwise Theorem Proving by Multi-Perspective Search and Data Curation 

**Title (ZH)**: MPS-Prover: 通过多视角搜索和数据整理推进分步定理证明 

**Authors**: Zhenwen Liang, Linfeng Song, Yang Li, Tao Yang, Feng Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10962)  

**Abstract**: Automated Theorem Proving (ATP) in formal languages remains a formidable challenge in AI, demanding rigorous logical deduction and navigating vast search spaces. While large language models (LLMs) have shown promising performance, existing stepwise provers often suffer from biased search guidance, leading to inefficiencies and suboptimal proof strategies. This paper introduces the Multi-Perspective Search Prover (MPS-Prover), a novel stepwise ATP system designed to overcome these limitations. MPS-Prover incorporates two key innovations: a highly effective post-training data curation strategy that prunes approximately 40% of redundant training data without sacrificing performance, and a multi-perspective tree search mechanism. This search integrates a learned critic model with strategically designed heuristic rules to diversify tactic selection, prevent getting trapped in unproductive states, and enhance search robustness. Extensive evaluations demonstrate that MPS-Prover achieves state-of-the-art performance on multiple challenging benchmarks, including miniF2F and ProofNet, outperforming prior 7B parameter models. Furthermore, our analyses reveal that MPS-Prover generates significantly shorter and more diverse proofs compared to existing stepwise and whole-proof methods, highlighting its efficiency and efficacy. Our work advances the capabilities of LLM-based formal reasoning and offers a robust framework and a comprehensive analysis for developing more powerful theorem provers. 

**Abstract (ZH)**: 形式语言中的自动定理证明（ATP）仍然是AI领域的 formidable 挑战，要求严格的逻辑推理和探索庞大的搜索空间。虽然大型语言模型（LLMs）展现了令人鼓舞的性能，但现有的逐步证明器常常遭受偏颇的搜索指导，导致效率低下和次优证明策略。本文介绍了多视角搜索证明器（MPS-Prover），这是一种新型的逐步ATP系统，旨在克服这些限制。MPS-Prover 包含两项关键创新：一种高效的后训练data curation 策略，该策略在不牺牲性能的情况下去掉了约40%的冗余训练数据，以及一种多视角树搜索机制。这种搜索机制结合了学习到的critic 模型和精心设计的启发式规则，以多样化策略选择、避免陷入无生产力状态，并增强搜索的稳健性。 extensive 评估表明，MPS-Prover 在多个具有挑战性的基准测试，包括miniF2F和ProofNet 上达到了最先进的性能，超越了此前的7B参数模型。此外，我们的分析表明，与现有的逐步证明方法和整体证明方法相比，MPS-Prover 生成的证明更短且更具多样性，突显了其高效性和有效性。本文推进了基于LLM的形式推理能力，并提供了更强大的定理证明器开发的坚固框架和全面分析。 

---
# InfantAgent-Next: A Multimodal Generalist Agent for Automated Computer Interaction 

**Title (ZH)**: InfantAgent-Next: 一种多模态通用代理agents用于自动化计算机交互 

**Authors**: Bin Lei, Weitai Kang, Zijian Zhang, Winson Chen, Xi Xie, Shan Zuo, Mimi Xie, Ali Payani, Mingyi Hong, Yan Yan, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.10887)  

**Abstract**: This paper introduces \textsc{InfantAgent-Next}, a generalist agent capable of interacting with computers in a multimodal manner, encompassing text, images, audio, and video. Unlike existing approaches that either build intricate workflows around a single large model or only provide workflow modularity, our agent integrates tool-based and pure vision agents within a highly modular architecture, enabling different models to collaboratively solve decoupled tasks in a step-by-step manner. Our generality is demonstrated by our ability to evaluate not only pure vision-based real-world benchmarks (i.e., OSWorld), but also more general or tool-intensive benchmarks (e.g., GAIA and SWE-Bench). Specifically, we achieve $\mathbf{7.27\%}$ accuracy on OSWorld, higher than Claude-Computer-Use. Codes and evaluation scripts are open-sourced at this https URL. 

**Abstract (ZH)**: 这篇论文介绍了\textsc{InfantAgent-Next}，这是一种能够以多模态方式与计算机交互的通用代理，涵盖了文本、图像、音频和视频。不同于现有方法要么围绕单一大型模型构建复杂的 workflow，要么仅提供 workflow 模块化，我们的代理在高度模块化的架构中集成了基于工具和纯视觉的代理，使不同的模型能够以分步方式协作解决解耦的任务。我们通过在纯视觉基础的真实世界基准（如 OSWorld）以及更通用或工具密集型基准（如 GAIA 和 SWE-Bench）上的评估来展示其通用性。具体而言，我们在 OSWorld 上达到了 $\mathbf{7.27\%}$ 的准确率，高于 Claude-Computer-Use。代码和评估脚本已在此 <https://> 开源。 

---
# MCU: Improving Machine Unlearning through Mode Connectivity 

**Title (ZH)**: MCU：通过模式连通性提升机器卸载性能 

**Authors**: Yingdan Shi, Ren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10859)  

**Abstract**: Machine Unlearning (MU) aims to remove the information of specific training data from a trained model, ensuring compliance with privacy regulations and user requests. While one line of existing MU methods relies on linear parameter updates via task arithmetic, they suffer from weight entanglement. In this work, we propose a novel MU framework called Mode Connectivity Unlearning (MCU) that leverages mode connectivity to find an unlearning pathway in a nonlinear manner. To further enhance performance and efficiency, we introduce a parameter mask strategy that not only improves unlearning effectiveness but also reduces computational overhead. Moreover, we propose an adaptive adjustment strategy for our unlearning penalty coefficient to adaptively balance forgetting quality and predictive performance during training, eliminating the need for empirical hyperparameter tuning. Unlike traditional MU methods that identify only a single unlearning model, MCU uncovers a spectrum of unlearning models along the pathway. Overall, MCU serves as a plug-and-play framework that seamlessly integrates with any existing MU methods, consistently improving unlearning efficacy. Extensive experiments on the image classification task demonstrate that MCU achieves superior performance. 

**Abstract (ZH)**: 模式连接性卸学（MCU）：一种利用模式连接性进行非线性卸学的新型框架 

---
# Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models 

**Title (ZH)**: 创造性还是 brute force？脑筋急转弯作为窗口考察大型语言模型的问题解决能力 

**Authors**: Simeng Han, Stephen Xia, Grant Zhang, Howard Dai, Chen Liu, Lichang Chen, Hoang Huy Nguyen, Hongyuan Mei, Jiayuan Mao, R. Thomas McCoy  

**Link**: [PDF](https://arxiv.org/pdf/2505.10844)  

**Abstract**: Accuracy remains a standard metric for evaluating AI systems, but it offers limited insight into how models arrive at their solutions. In this work, we introduce a benchmark based on brainteasers written in long narrative form to probe more deeply into the types of reasoning strategies that models use. Brainteasers are well-suited for this goal because they can be solved with multiple approaches, such as a few-step solution that uses a creative insight or a longer solution that uses more brute force. We investigate large language models (LLMs) across multiple layers of reasoning, focusing not only on correctness but also on the quality and creativity of their solutions. We investigate many aspects of the reasoning process: (1) semantic parsing of the brainteasers into precise mathematical competition style formats; (2) generating solutions from these mathematical forms; (3) self-correcting solutions based on gold solutions; (4) producing step-by-step sketches of solutions; and (5) making use of hints. We find that LLMs are in many cases able to find creative, insightful solutions to brainteasers, suggesting that they capture some of the capacities needed to solve novel problems in creative ways. Nonetheless, there also remain situations where they rely on brute force despite the availability of more efficient, creative solutions, highlighting a potential direction for improvement in the reasoning abilities of LLMs. 

**Abstract (ZH)**: 基于长篇叙事形式谜题的基准测试：探究大型语言模型的推理策略 

---
# TACO: Rethinking Semantic Communications with Task Adaptation and Context Embedding 

**Title (ZH)**: TACO: 任务适应与上下文嵌入的语义通信 rethink 

**Authors**: Achintha Wijesinghe, Weiwei Wang, Suchinthaka Wanninayaka, Songyang Zhang, Zhi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.10834)  

**Abstract**: Recent advancements in generative artificial intelligence have introduced groundbreaking approaches to innovating next-generation semantic communication, which prioritizes conveying the meaning of a message rather than merely transmitting raw data. A fundamental challenge in semantic communication lies in accurately identifying and extracting the most critical semantic information while adapting to downstream tasks without degrading performance, particularly when the objective at the receiver may evolve over time. To enable flexible adaptation to multiple tasks at the receiver, this work introduces a novel semantic communication framework, which is capable of jointly capturing task-specific information to enhance downstream task performance and contextual information. Through rigorous experiments on popular image datasets and computer vision tasks, our framework shows promising improvement compared to existing work, including superior performance in downstream tasks, better generalizability, ultra-high bandwidth efficiency, and low reconstruction latency. 

**Abstract (ZH)**: 近期生成式人工智能的发展引入了下一代语义通信的开创性方法，重点在于传达信息的意义而非仅传输原始数据。语义通信的基本挑战在于在适应下游任务时准确识别和提取最关键的信息，尤其是在接收方的目标可能随时间变化时，不降低性能。为了使接收方能够灵活适应多种任务，本文提出了一种新颖的语义通信框架，该框架能够联合捕捉任务特定信息以提升下游任务性能和上下文信息。通过在流行图像数据集和计算机视觉任务上的严格实验，我们的框架展现出了与现有工作相比的显著改进，包括在下游任务中的卓越性能、更好的泛化能力、超高的带宽效率和低重构延迟。 

---
# PoE-World: Compositional World Modeling with Products of Programmatic Experts 

**Title (ZH)**: PoE-World: 基于程序专家乘积的组合世界建模 

**Authors**: Wasu Top Piriyakulkij, Yichao Liang, Hao Tang, Adrian Weller, Marta Kryven, Kevin Ellis  

**Link**: [PDF](https://arxiv.org/pdf/2505.10819)  

**Abstract**: Learning how the world works is central to building AI agents that can adapt to complex environments. Traditional world models based on deep learning demand vast amounts of training data, and do not flexibly update their knowledge from sparse observations. Recent advances in program synthesis using Large Language Models (LLMs) give an alternate approach which learns world models represented as source code, supporting strong generalization from little data. To date, application of program-structured world models remains limited to natural language and grid-world domains. We introduce a novel program synthesis method for effectively modeling complex, non-gridworld domains by representing a world model as an exponentially-weighted product of programmatic experts (PoE-World) synthesized by LLMs. We show that this approach can learn complex, stochastic world models from just a few observations. We evaluate the learned world models by embedding them in a model-based planning agent, demonstrating efficient performance and generalization to unseen levels on Atari's Pong and Montezuma's Revenge. We release our code and display the learned world models and videos of the agent's gameplay at this https URL. 

**Abstract (ZH)**: 通过程序合成有效建模复杂非网格世界的方法及其应用 

---
# Developing and Integrating Trust Modeling into Multi-Objective Reinforcement Learning for Intelligent Agricultural Management 

**Title (ZH)**: 开发并集成信任模型到多目标强化学习中的智能农业管理 

**Authors**: Zhaoan Wang, Wonseok Jang, Bowen Ruan, Jun Wang, Shaoping Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10803)  

**Abstract**: Precision agriculture, enhanced by artificial intelligence (AI), offers promising tools such as remote sensing, intelligent irrigation, fertilization management, and crop simulation to improve agricultural efficiency and sustainability. Reinforcement learning (RL), in particular, has outperformed traditional methods in optimizing yields and resource management. However, widespread AI adoption is limited by gaps between algorithmic recommendations and farmers' practical experience, local knowledge, and traditional practices. To address this, our study emphasizes Human-AI Interaction (HAII), focusing on transparency, usability, and trust in RL-based farm management. We employ a well-established trust framework - comprising ability, benevolence, and integrity - to develop a novel mathematical model quantifying farmers' confidence in AI-based fertilization strategies. Surveys conducted with farmers for this research reveal critical misalignments, which are integrated into our trust model and incorporated into a multi-objective RL framework. Unlike prior methods, our approach embeds trust directly into policy optimization, ensuring AI recommendations are technically robust, economically feasible, context-aware, and socially acceptable. By aligning technical performance with human-centered trust, this research supports broader AI adoption in agriculture. 

**Abstract (ZH)**: 人工智能增强的精确农业：基于强化学习的信任人类-人工智能交互研究 

---
# SECRET: Semi-supervised Clinical Trial Document Similarity Search 

**Title (ZH)**: SECTOR: 半监督临床试验文档相似性搜索 

**Authors**: Trisha Das, Afrah Shafquat, Beigi Mandis, Jacob Aptekar, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.10780)  

**Abstract**: Clinical trials are vital for evaluation of safety and efficacy of new treatments. However, clinical trials are resource-intensive, time-consuming and expensive to conduct, where errors in trial design, reduced efficacy, and safety events can result in significant delays, financial losses, and damage to reputation. These risks underline the importance of informed and strategic decisions in trial design to mitigate these risks and improve the chances of a successful trial. Identifying similar historical trials is critical as these trials can provide an important reference for potential pitfalls and challenges including serious adverse events, dosage inaccuracies, recruitment difficulties, patient adherence issues, etc. Addressing these challenges in trial design can lead to development of more effective study protocols with optimized patient safety and trial efficiency. In this paper, we present a novel method to identify similar historical trials by summarizing clinical trial protocols and searching for similar trials based on a query trial's protocol. Our approach significantly outperforms all baselines, achieving up to a 78% improvement in recall@1 and a 53% improvement in precision@1 over the best baseline. We also show that our method outperforms all other baselines in partial trial similarity search and zero-shot patient-trial matching, highlighting its superior utility in these tasks. 

**Abstract (ZH)**: 临床试验对于评估新治疗方法的安全性和有效性至关重要。然而，临床试验耗资巨大、耗时且费用高昂，试验设计中的错误、效力降低和安全性事件可能会导致重大延误、经济损失和声誉损害。这些风险凸显了在试验设计中做出知情和战略决策的重要性，以降低这些风险并提高试验成功的可能性。识别相似的历史试验至关重要，这些试验可以为潜在的风险和挑战（包括严重不良事件、剂量不准确、招募困难、患者依从性问题等）提供重要参考。通过在试验设计中应对这些挑战，可以开发出更有效的研究方案，优化患者安全和试验效率。在本文中，我们提出了一种新颖的方法，通过总结临床试验协议并依据查询试验的协议搜索相似试验。我们的方法显著优于所有基线，召回率@1提高了78%，精确率@1提高了53%。我们还展示了我们的方法在部分试验相似性搜索和零样本患者-试验匹配任务中均优于所有其他基线，突显了其在这些任务中的优越实用性。 

---
# Qualia Optimization 

**Title (ZH)**: 质体优化 

**Authors**: Philip S. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2505.10779)  

**Abstract**: This report explores the speculative question: what if current or future AI systems have qualia, such as pain or pleasure? It does so by assuming that AI systems might someday possess qualia -- and that the quality of these subjective experiences should be considered alongside performance metrics. Concrete mathematical problem settings, inspired by reinforcement learning formulations and theories from philosophy of mind, are then proposed and initial approaches and properties are presented. These properties enable refinement of the problem setting, culminating with the proposal of methods that promote reinforcement. 

**Abstract (ZH)**: 本报告探讨了假设性问题：当前或未来的AI系统是否具有主观体验（如疼痛或快乐），并通过假设AI系统 someday 可能具备主观体验，进而考虑这些主观体验的质量与性能指标并重，提出具体的数学问题设定，借鉴强化学习的表述形式和心灵哲学理论，介绍初步方法和性质，这些性质使得问题设定得以细化，最终提出促进强化的方法。 

---
# Code-Driven Planning in Grid Worlds with Large Language Models 

**Title (ZH)**: 基于代码驱动的规划在网格世界中的大型语言模型应用 

**Authors**: Ashwath Vaithinathan Aravindan, Zhisheng Tang, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2505.10749)  

**Abstract**: We propose an iterative programmatic planning (IPP) framework for solving grid-based tasks by synthesizing interpretable agent policies expressed in code using large language models (LLMs). Instead of relying on traditional search or reinforcement learning, our approach uses code generation as policy synthesis, where the LLM outputs executable programs that map environment states to action sequences. Our proposed architecture incorporates several prompting strategies, including direct code generation, pseudocode-conditioned refinement, and curriculum-based prompting, but also includes an iterative refinement mechanism that updates code based on task performance feedback. We evaluate our approach using six leading LLMs and two challenging grid-based benchmarks (GRASP and MiniGrid). Our IPP framework demonstrates improvements over direct code generation ranging from 10\% to as much as 10x across five of the six models and establishes a new state-of-the-art result for GRASP. IPP is found to significantly outperform direct elicitation of a solution from GPT-o3-mini (by 63\% on MiniGrid to 116\% on GRASP), demonstrating the viability of the overall approach. Computational costs of all code generation approaches are similar. While code generation has a higher initial prompting cost compared to direct solution elicitation (\$0.08 per task vs. \$0.002 per instance for GPT-o3-mini), the code can be reused for any number of instances, making the amortized cost significantly lower (by 400x on GPT-o3-mini across the complete GRASP benchmark). 

**Abstract (ZH)**: 我们提出了一种迭代程序规划（IPP）框架，用于通过大型语言模型（LLMs）生成可解释的代理策略来解决基于网格的任务。我们的方法不依赖于传统的搜索或强化学习，而是使用代码生成作为策略合成，其中LLM输出可执行程序，将环境状态映射到动作序列。我们提出的架构包括直接代码生成、伪代码条件细化以及基于课程的学习提示策略，同时还包括一个迭代细化机制，该机制根据任务性能反馈更新代码。我们使用六种领先的LLM和两个具有挑战性的基于网格的基准（GRASP和MiniGrid）评估了我们的方法。IPP框架在五种六种模型中均显示出直接代码生成的性能提升，范围从10%到高达10倍，并在GRASP上建立了新的最先进结果。IPP在性能上显著优于直接从GPT-o3-mini中提取解决方案，从MiniGrid上的63%到GRASP上的116%，证明了该整体方法的可行性。所有代码生成方法的计算成本相似。尽管与直接解决方案提取相比，代码生成方法的初始提示成本更高（每任务$0.08 vs. 每实例$0.002的GPT-o3-mini），但由于代码可以重复使用任何数量的实例，因此平均成本显著降低（在完整的GRASP基准上，降低400倍）。 

---
# Evaluations at Work: Measuring the Capabilities of GenAI in Use 

**Title (ZH)**: 工作中的评估:衡量生成式AI的能力 

**Authors**: Brandon Lepine, Gawesha Weerantunga, Juho Kim, Pamela Mishkin, Matthew Beane  

**Link**: [PDF](https://arxiv.org/pdf/2505.10742)  

**Abstract**: Current AI benchmarks miss the messy, multi-turn nature of human-AI collaboration. We present an evaluation framework that decomposes real-world tasks into interdependent subtasks, letting us track both LLM performance and users' strategies across a dialogue. Complementing this framework, we develop a suite of metrics, including a composite usage derived from semantic similarity, word overlap, and numerical matches; structural coherence; intra-turn diversity; and a novel measure of the "information frontier" reflecting the alignment between AI outputs and users' working knowledge. We demonstrate our methodology in a financial valuation task that mirrors real-world complexity. Our empirical findings reveal that while greater integration of LLM-generated content generally enhances output quality, its benefits are moderated by factors such as response incoherence, excessive subtask diversity, and the distance of provided information from users' existing knowledge. These results suggest that proactive dialogue strategies designed to inject novelty may inadvertently undermine task performance. Our work thus advances a more holistic evaluation of human-AI collaboration, offering both a robust methodological framework and actionable insights for developing more effective AI-augmented work processes. 

**Abstract (ZH)**: 当前的AI基准未能捕捉到人类与AI协作中杂乱多轮的性质。我们提出了一种评估框架，将实际任务分解为相互依赖的子任务，让我们能够追踪对话中LLM的表现和用户的策略。为了补充这一框架，我们开发了一套指标，包括基于语义相似度、词重叠和数值匹配的综合使用度；结构连贯性；单轮多样性；以及一种新的反映AI输出与用户现有知识一致性程度的“信息前沿”度量。我们在一项模拟现实复杂性的金融估值任务中演示了我们的方法论。我们的实证研究发现，虽然更紧密地整合LLM生成的内容通常会提高输出质量，但其益处会受到回应不一致、子任务多样性过度以及提供的信息与用户现有知识距离较远等因素的制约。这些结果表明，旨在注入新颖性的主动对话策略可能无意中降低任务性能。因此，我们的工作推进了对人类与AI协作的更全面评估，提供了稳健的方法论框架和改进更有效AI增强工作流程的实用见解。 

---
# Embodied AI in Machine Learning -- is it Really Embodied? 

**Title (ZH)**: 机器学习中的具身AI——它真是具身的吗？ 

**Authors**: Matej Hoffmann, Shubhan Parag Patni  

**Link**: [PDF](https://arxiv.org/pdf/2505.10705)  

**Abstract**: Embodied Artificial Intelligence (Embodied AI) is gaining momentum in the machine learning communities with the goal of leveraging current progress in AI (deep learning, transformers, large language and visual-language models) to empower robots. In this chapter we put this work in the context of "Good Old-Fashioned Artificial Intelligence" (GOFAI) (Haugeland, 1989) and the behavior-based or embodied alternatives (R. A. Brooks 1991; Pfeifer and Scheier 2001). We claim that the AI-powered robots are only weakly embodied and inherit some of the problems of GOFAI. Moreover, we review and critically discuss the possibility of cross-embodiment learning (Padalkar et al. 2024). We identify fundamental roadblocks and propose directions on how to make progress. 

**Abstract (ZH)**: 基于体征的人工智能：从GOFAI到跨体征学习的研究进展 

---
# Interpretable Risk Mitigation in LLM Agent Systems 

**Title (ZH)**: 可解释的风险缓解在LLM代理系统中 

**Authors**: Jan Chojnacki  

**Link**: [PDF](https://arxiv.org/pdf/2505.10670)  

**Abstract**: Autonomous agents powered by large language models (LLMs) enable novel use cases in domains where responsible action is increasingly important. Yet the inherent unpredictability of LLMs raises safety concerns about agent reliability. In this work, we explore agent behaviour in a toy, game-theoretic environment based on a variation of the Iterated Prisoner's Dilemma. We introduce a strategy-modification method-independent of both the game and the prompt-by steering the residual stream with interpretable features extracted from a sparse autoencoder latent space. Steering with the good-faith negotiation feature lowers the average defection probability by 28 percentage points. We also identify feasible steering ranges for several open-source LLM agents. Finally, we hypothesise that game-theoretic evaluation of LLM agents, combined with representation-steering alignment, can generalise to real-world applications on end-user devices and embodied platforms. 

**Abstract (ZH)**: 由大规模语言模型驱动的自主代理在负责行动日益重要的领域中开启新型应用场景。然而，大型语言模型的内在不可预测性引发了关于代理可靠性的安全关切。在本工作中，我们探索了基于迭代囚犯困境变种的游戏理论环境中的代理行为，通过引导稀疏自编码器潜在空间中的可解释特征流来引入一种与游戏和提示无关的策略修改方法。使用善意谈判特征引导降低了平均背叛概率28个百分点。我们还确定了几种开源LLM代理的可行引导范围。最后，我们假设结合游戏理论评估与表征引导对齐，可以将LLM代理推广到用户终端设备和实体平台上的实际应用。 

---
# On the Evaluation of Engineering Artificial General Intelligence 

**Title (ZH)**: 工程通用人工智能的评估方法 

**Authors**: Sandeep Neema, Susmit Jha, Adam Nagel, Ethan Lew, Chandrasekar Sureshkumar, Aleksa Gordic, Chase Shimmin, Hieu Nguygen, Paul Eremenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.10653)  

**Abstract**: We discuss the challenges and propose a framework for evaluating engineering artificial general intelligence (eAGI) agents. We consider eAGI as a specialization of artificial general intelligence (AGI), deemed capable of addressing a broad range of problems in the engineering of physical systems and associated controllers. We exclude software engineering for a tractable scoping of eAGI and expect dedicated software engineering AI agents to address the software implementation challenges. Similar to human engineers, eAGI agents should possess a unique blend of background knowledge (recall and retrieve) of facts and methods, demonstrate familiarity with tools and processes, exhibit deep understanding of industrial components and well-known design families, and be able to engage in creative problem solving (analyze and synthesize), transferring ideas acquired in one context to another. Given this broad mandate, evaluating and qualifying the performance of eAGI agents is a challenge in itself and, arguably, a critical enabler to developing eAGI agents. In this paper, we address this challenge by proposing an extensible evaluation framework that specializes and grounds Bloom's taxonomy - a framework for evaluating human learning that has also been recently used for evaluating LLMs - in an engineering design context. Our proposed framework advances the state of the art in benchmarking and evaluation of AI agents in terms of the following: (a) developing a rich taxonomy of evaluation questions spanning from methodological knowledge to real-world design problems; (b) motivating a pluggable evaluation framework that can evaluate not only textual responses but also evaluate structured design artifacts such as CAD models and SysML models; and (c) outlining an automatable procedure to customize the evaluation benchmark to different engineering contexts. 

**Abstract (ZH)**: 我们探讨了挑战并提出了评估工程通用人工智能（eAGI）代理的框架。 

---
# Modeling cognitive processes of natural reading with transformer-based Language Models 

**Title (ZH)**: 基于变换器的语言模型模拟自然阅读的认知过程 

**Authors**: Bruno Bianchi, Fermín Travi, Juan E. Kamienkowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.11485)  

**Abstract**: Recent advances in Natural Language Processing (NLP) have led to the development of highly sophisticated language models for text generation. In parallel, neuroscience has increasingly employed these models to explore cognitive processes involved in language comprehension. Previous research has shown that models such as N-grams and LSTM networks can partially account for predictability effects in explaining eye movement behaviors, specifically Gaze Duration, during reading. In this study, we extend these findings by evaluating transformer-based models (GPT2, LLaMA-7B, and LLaMA2-7B) to further investigate this relationship. Our results indicate that these architectures outperform earlier models in explaining the variance in Gaze Durations recorded from Rioplantense Spanish readers. However, similar to previous studies, these models still fail to account for the entirety of the variance captured by human predictability. These findings suggest that, despite their advancements, state-of-the-art language models continue to predict language in ways that differ from human readers. 

**Abstract (ZH)**: 最近自然语言处理（NLP）的进展推动了高级语言模型在文本生成中的发展。与此同时，神经科学越来越多地利用这些模型来探究语言理解中涉及的认知过程。先前的研究表明，N-克gram和LSTM网络可以部分解释在阅读过程中 gaze duration 等眼动行为中的可预测性效应。本研究在此基础上通过评估基于Transformer的模型（GPT2、LLaMA-7B和LLaMA2-7B）进一步探讨该关系。研究结果表明，这些架构在解释来自Rioplantense西班牙语读者的 gaze duration 变异方面优于早期模型。然而，与先前的研究类似，这些模型仍无法完全解释由人类可预测性捕获的变异。这些发现表明，尽管取得了进步，最先进的语言模型在预测语言方面仍然与人类读者有所不同。 

---
# Improving Assembly Code Performance with Large Language Models via Reinforcement Learning 

**Title (ZH)**: 使用强化学习提升大型语言模型的汇编代码性能 

**Authors**: Anjiang Wei, Tarun Suresh, Huanmi Tan, Yinglun Xu, Gagandeep Singh, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2505.11480)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance across a wide range of programming tasks, yet their potential for code optimization remains underexplored. This work investigates whether LLMs can optimize the performance of assembly code, where fine-grained control over execution enables improvements that are difficult to express in high-level languages. We present a reinforcement learning framework that trains LLMs using Proximal Policy Optimization (PPO), guided by a reward function that considers both functional correctness, validated through test cases, and execution performance relative to the industry-standard compiler gcc -O3. To support this study, we introduce a benchmark of 8,072 real-world programs. Our model, Qwen2.5-Coder-7B-PPO, achieves 96.0% test pass rates and an average speedup of 1.47x over the gcc -O3 baseline, outperforming all 20 other models evaluated, including Claude-3.7-sonnet. These results indicate that reinforcement learning can unlock the potential of LLMs to serve as effective optimizers for assembly code performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛范围的编程任务中展示了出色的表现，但其在代码优化方面的潜力尚未得到充分探索。本工作探究大型语言模型是否能够优化汇编代码的性能，这种细粒度的执行控制使得在高级语言中难以实现的性能改进成为可能。我们提出了一种基于强化学习的框架，使用渐进策略优化（PPO）训练大型语言模型，并通过一个奖励函数进行引导，该奖励函数同时考虑通过测试案例验证的功能正确性和相对于工业标准编译器gcc -O3的执行性能。为了支持这一研究，我们引入了一个包含8,072个真实世界程序的基准系统。我们的模型Qwen2.5-Coder-7B-PPO在测试通过率上达到了96.0%，平均加速了1.47倍，优于所有其他评估的20个模型，包括Claude-3.7-sonnet。这些结果表明，强化学习可以解锁大型语言模型作为汇编代码性能优化器的有效潜力。 

---
# HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages 

**Title (ZH)**: HelpSteer3-偏好：跨多样任务和语言的开放人类标注偏好数据 

**Authors**: Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, Oleksii Kuchaiev  

**Link**: [PDF](https://arxiv.org/pdf/2505.11475)  

**Abstract**: Preference datasets are essential for training general-domain, instruction-following language models with Reinforcement Learning from Human Feedback (RLHF). Each subsequent data release raises expectations for future data collection, meaning there is a constant need to advance the quality and diversity of openly available preference data. To address this need, we introduce HelpSteer3-Preference, a permissively licensed (CC-BY-4.0), high-quality, human-annotated preference dataset comprising of over 40,000 samples. These samples span diverse real-world applications of large language models (LLMs), including tasks relating to STEM, coding and multilingual scenarios. Using HelpSteer3-Preference, we train Reward Models (RMs) that achieve top performance on RM-Bench (82.4%) and JudgeBench (73.7%). This represents a substantial improvement (~10% absolute) over the previously best-reported results from existing RMs. We demonstrate HelpSteer3-Preference can also be applied to train Generative RMs and how policy models can be aligned with RLHF using our RMs. Dataset (CC-BY-4.0): this https URL 

**Abstract (ZH)**: 偏好数据集对于训练通用领域指令遵循语言模型至关重要，可运用人类反馈的强化学习（RLHF）进行训练。为了满足这一需求，我们引入了HelpSteer3-Preference，这是一个采用宽容许可（CC-BY-4.0）、高质量且由人类标注的偏好数据集，包含超过40,000个样本。这些样本涵盖了大型语言模型（LLMs）在STEM、编码和多语言场景等各种实际应用任务。使用HelpSteer3-Preference，我们训练的奖励模型（RMs）在RM-Bench上达到82.4%的性能，在JudgeBench上达到73.7%的性能，相比之前最好的结果提升了约10%。我们还展示了如何使用HelpSteer3-Preference训练生成型奖励模型以及如何利用我们的奖励模型对策略模型进行RLHF对齐。数据集（CC-BY-4.0）：这个链接 

---
# Disentangling Reasoning and Knowledge in Medical Large Language Models 

**Title (ZH)**: 解构医学大型语言模型中的推理与知识 

**Authors**: Rahul Thapa, Qingyang Wu, Kevin Wu, Harrison Zhang, Angela Zhang, Eric Wu, Haotian Ye, Suhana Bedi, Nevin Aresh, Joseph Boen, Shriya Reddy, Ben Athiwaratkun, Shuaiwen Leon Song, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11462)  

**Abstract**: Medical reasoning in large language models (LLMs) aims to emulate clinicians' diagnostic thinking, but current benchmarks such as MedQA-USMLE, MedMCQA, and PubMedQA often mix reasoning with factual recall. We address this by separating 11 biomedical QA benchmarks into reasoning- and knowledge-focused subsets using a PubMedBERT classifier that reaches 81 percent accuracy, comparable to human performance. Our analysis shows that only 32.8 percent of questions require complex reasoning. We evaluate biomedical models (HuatuoGPT-o1, MedReason, m1) and general-domain models (DeepSeek-R1, o4-mini, Qwen3), finding consistent gaps between knowledge and reasoning performance. For example, m1 scores 60.5 on knowledge but only 47.1 on reasoning. In adversarial tests where models are misled with incorrect initial reasoning, biomedical models degrade sharply, while larger or RL-trained general models show more robustness. To address this, we train BioMed-R1 using fine-tuning and reinforcement learning on reasoning-heavy examples. It achieves the strongest performance among similarly sized models. Further gains may come from incorporating clinical case reports and training with adversarial and backtracking scenarios. 

**Abstract (ZH)**: 大型语言模型在医学推断中的能力旨在模拟临床医生的诊断思考，但当前的基准测试（如MedQA-USMLE、MedMCQA和PubMedQA）通常将推断与事实回忆混合在一起。为此，我们利用达到81%准确率的PubMedBERT分类器，将11项生物医学问答基准分为侧重推理和知识的子集。我们的分析显示，仅32.8%的问题需要复杂的推理。我们评估了生物医学模型（HuatuoGPT-o1、MedReason、m1）和通用领域的模型（DeepSeek-R1、o4-mini、Qwen3），发现知识和推理性能之间存在一致的差距。例如，m1在知识上的得分为60.5，但在推理上的得分为47.1。在对抗性测试中，当模型被错误的初始推理误导时，生物医学模型出现大幅下降，而较大的或通过强化学习训练的一般模型则表现出更多的鲁棒性。为此，我们使用推理密集型示例进行微调和强化学习训练了BioMed-R1。该模型在其同规模模型中表现最佳。进一步的改进可能来自于纳入临床病例报告并在对抗性和回溯场景中进行训练。 

---
# HumaniBench: A Human-Centric Framework for Large Multimodal Models Evaluation 

**Title (ZH)**: HumaniBench: 以人为本的大型多模态模型评估框架 

**Authors**: Shaina Raza, Aravind Narayanan, Vahid Reza Khazaie, Ashmal Vayani, Mukund S. Chettiar, Amandeep Singh, Mubarak Shah, Deval Pandya  

**Link**: [PDF](https://arxiv.org/pdf/2505.11454)  

**Abstract**: Large multimodal models (LMMs) now excel on many vision language benchmarks, however, they still struggle with human centered criteria such as fairness, ethics, empathy, and inclusivity, key to aligning with human values. We introduce HumaniBench, a holistic benchmark of 32K real-world image question pairs, annotated via a scalable GPT4o assisted pipeline and exhaustively verified by domain experts. HumaniBench evaluates seven Human Centered AI (HCAI) principles: fairness, ethics, understanding, reasoning, language inclusivity, empathy, and robustness, across seven diverse tasks, including open and closed ended visual question answering (VQA), multilingual QA, visual grounding, empathetic captioning, and robustness tests. Benchmarking 15 state of the art LMMs (open and closed source) reveals that proprietary models generally lead, though robustness and visual grounding remain weak points. Some open-source models also struggle to balance accuracy with adherence to human-aligned principles. HumaniBench is the first benchmark purpose built around HCAI principles. It provides a rigorous testbed for diagnosing alignment gaps and guiding LMMs toward behavior that is both accurate and socially responsible. Dataset, annotation prompts, and evaluation code are available at: this https URL 

**Abstract (ZH)**: 大规模多模态模型（LMMs）在许多视觉语言基准测试中已表现出色，但在公平性、伦理、同理心和包容性等以人为本的标准方面仍面临挑战，这些都是与人类价值观对齐的关键。我们引入了HumaniBench，这是一个由32K真实世界图像问题对组成的综合基准，通过可扩展的GPT4o辅助管道标注，并由领域专家详尽验证。HumaniBench评估了七个人本中心AI（HCAI）原则：公平性、伦理、理解、推理、语言包容性、同理心和鲁棒性，涵盖了七个不同任务，包括开放和封闭式的视觉问答（VQA）、多语言问答、视觉定位、同理心描述和鲁棒性测试。对15个最先进的LMMs（开源和封闭源）的评估显示，专有模型通常表现更好，但鲁棒性和视觉定位仍然是薄弱环节。一些开源模型也难以在准确性与遵循以人为本的原则之间找到平衡。HumaniBench是首个专门围绕HCAI原则构建的基准。它提供了一个严格的测试平台，用于诊断对齐差距，并指导LMMs向同时准确和负责任的行为发展。数据集、标注提示和评估代码可在以下链接获取：this https URL 

---
# LLMs unlock new paths to monetizing exploits 

**Title (ZH)**: LLMs解锁新的盈利途径 

**Authors**: Nicholas Carlini, Milad Nasr, Edoardo Debenedetti, Barry Wang, Christopher A. Choquette-Choo, Daphne Ippolito, Florian Tramèr, Matthew Jagielski  

**Link**: [PDF](https://arxiv.org/pdf/2505.11449)  

**Abstract**: We argue that Large language models (LLMs) will soon alter the economics of cyberattacks. Instead of attacking the most commonly used software and monetizing exploits by targeting the lowest common denominator among victims, LLMs enable adversaries to launch tailored attacks on a user-by-user basis. On the exploitation front, instead of human attackers manually searching for one difficult-to-identify bug in a product with millions of users, LLMs can find thousands of easy-to-identify bugs in products with thousands of users. And on the monetization front, instead of generic ransomware that always performs the same attack (encrypt all your data and request payment to decrypt), an LLM-driven ransomware attack could tailor the ransom demand based on the particular content of each exploited device.
We show that these two attacks (and several others) are imminently practical using state-of-the-art LLMs. For example, we show that without any human intervention, an LLM finds highly sensitive personal information in the Enron email dataset (e.g., an executive having an affair with another employee) that could be used for blackmail. While some of our attacks are still too expensive to scale widely today, the incentives to implement these attacks will only increase as LLMs get cheaper. Thus, we argue that LLMs create a need for new defense-in-depth approaches. 

**Abstract (ZH)**: 大型语言模型将 Soon Alter the Economics of Cyberattacks 

---
# SurgPose: Generalisable Surgical Instrument Pose Estimation using Zero-Shot Learning and Stereo Vision 

**Title (ZH)**: SurgPose: 任意场景下基于零样本学习和立体视觉的手术器械姿态估计 

**Authors**: Utsav Rai, Haozheng Xu, Stamatia Giannarou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11439)  

**Abstract**: Accurate pose estimation of surgical tools in Robot-assisted Minimally Invasive Surgery (RMIS) is essential for surgical navigation and robot control. While traditional marker-based methods offer accuracy, they face challenges with occlusions, reflections, and tool-specific designs. Similarly, supervised learning methods require extensive training on annotated datasets, limiting their adaptability to new tools. Despite their success in other domains, zero-shot pose estimation models remain unexplored in RMIS for pose estimation of surgical instruments, creating a gap in generalising to unseen surgical tools. This paper presents a novel 6 Degrees of Freedom (DoF) pose estimation pipeline for surgical instruments, leveraging state-of-the-art zero-shot RGB-D models like the FoundationPose and SAM-6D. We advanced these models by incorporating vision-based depth estimation using the RAFT-Stereo method, for robust depth estimation in reflective and textureless environments. Additionally, we enhanced SAM-6D by replacing its instance segmentation module, Segment Anything Model (SAM), with a fine-tuned Mask R-CNN, significantly boosting segmentation accuracy in occluded and complex conditions. Extensive validation reveals that our enhanced SAM-6D surpasses FoundationPose in zero-shot pose estimation of unseen surgical instruments, setting a new benchmark for zero-shot RGB-D pose estimation in RMIS. This work enhances the generalisability of pose estimation for unseen objects and pioneers the application of RGB-D zero-shot methods in RMIS. 

**Abstract (ZH)**: 基于零样本RGB-D模型的手术工具六自由度姿态估计方法在机器人辅助微创手术中的应用 

---
# GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art 

**Title (ZH)**: GODBench：视频评论艺术中多模态大型语言模型的基准测试 

**Authors**: Chenkai Zhang, Yiming Lei, Zeming Liu, Haitao Leng, Shaoguo Liu, Tingting Gao, Qingjie Liu, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11436)  

**Abstract**: Video Comment Art enhances user engagement by providing creative content that conveys humor, satire, or emotional resonance, requiring a nuanced and comprehensive grasp of cultural and contextual subtleties. Although Multimodal Large Language Models (MLLMs) and Chain-of-Thought (CoT) have demonstrated strong reasoning abilities in STEM tasks (e.g. mathematics and coding), they still struggle to generate creative expressions such as resonant jokes and insightful satire. Moreover, existing benchmarks are constrained by their limited modalities and insufficient categories, hindering the exploration of comprehensive creativity in video-based Comment Art creation. To address these limitations, we introduce GODBench, a novel benchmark that integrates video and text modalities to systematically evaluate MLLMs' abilities to compose Comment Art. Furthermore, inspired by the propagation patterns of waves in physics, we propose Ripple of Thought (RoT), a multi-step reasoning framework designed to enhance the creativity of MLLMs. Extensive experiments reveal that existing MLLMs and CoT methods still face significant challenges in understanding and generating creative video comments. In contrast, RoT provides an effective approach to improve creative composing, highlighting its potential to drive meaningful advancements in MLLM-based creativity. GODBench is publicly available at this https URL. 

**Abstract (ZH)**: 视频评论艺术通过提供传递幽默、讽刺或情感共鸣的创意内容来增强用户参与度，需要对文化与情境的细微差异有精深的理解。尽管多模态大语言模型（MLLMs）和链式思考（CoT）在STEM任务（如数学和编码）中展现了强大的推理能力，但在生成富有创意的表达（如共鸣的笑话和深刻的讽刺）方面仍然力有未逮。此外，当前的基准受限于其有限的模态和不足的分类，阻碍了在基于视频的评论艺术创作中进行全面创意探索。为克服这些限制，我们引入了GODBench，这是一种新颖的基准，结合了视频和文本模态，系统性地评估MLLMs生成评论艺术的能力。此外，受到物理波传播模式的启发，我们提出了波纹思考（RoT）多步推理框架，旨在增强MLLMs的创造力。大量实验证明，现有MLLMs和CoT方法在理解和生成创意视频评论方面仍然面临重大挑战。相比之下，RoT提供了一种有效的方法来提高创造性写作水平，突显了它在基于MLLM的创意发展中巨大潜力。GODBench已在以下网址公开发布：this https URL。 

---
# Mergenetic: a Simple Evolutionary Model Merging Library 

**Title (ZH)**: Mergenetic：一种简化的进化模型集成功能库 

**Authors**: Adrian Robert Minut, Tommaso Mencattini, Andrea Santilli, Donato Crisostomi, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2505.11427)  

**Abstract**: Model merging allows combining the capabilities of existing models into a new one - post hoc, without additional training. This has made it increasingly popular thanks to its low cost and the availability of libraries that support merging on consumer GPUs. Recent work shows that pairing merging with evolutionary algorithms can boost performance, but no framework currently supports flexible experimentation with such strategies in language models. We introduce Mergenetic, an open-source library for evolutionary model merging. Mergenetic enables easy composition of merging methods and evolutionary algorithms while incorporating lightweight fitness estimators to reduce evaluation costs. We describe its design and demonstrate that Mergenetic produces competitive results across tasks and languages using modest hardware. 

**Abstract (ZH)**: 模型合并允许将现有模型的能力合并到一个新的模型中——事后合并，无需额外训练。这种方法由于其低成本和可用的支持合并的库而在近期变得越来越流行。最近的研究表明，将合并与进化算法相结合可以提升性能，但目前还没有框架支持在语言模型中灵活地实验这类策略。我们介绍了Mergenetic，这是一个开源的进化模型合并库。Mergenetic简化了合并方法和进化算法的组合，并通过引入轻量级的适应度估算器来减少评估成本。我们描述了其设计，并证明了即使使用 modest 硬件，Mergenetic也能在任务和语言方面产生竞争力的结果。 

---
# Improving Object Detection Performance through YOLOv8: A Comprehensive Training and Evaluation Study 

**Title (ZH)**: 通过YOLOv8提高目标检测性能：一项全面的训练与评估研究 

**Authors**: Rana Poureskandar, Shiva Razzagzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11424)  

**Abstract**: This study evaluated the performance of a YOLOv8-based segmentation model for detecting and segmenting wrinkles in facial images. 

**Abstract (ZH)**: 本研究评估了基于YOLOv8的分割模型在检测和分割面部图像中皱纹的表现。 

---
# EdgeWisePersona: A Dataset for On-Device User Profiling from Natural Language Interactions 

**Title (ZH)**: EdgeWisePersona: 一种基于自然语言交互的设备端用户画像数据集 

**Authors**: Patryk Bartkowiak, Michal Podstawski  

**Link**: [PDF](https://arxiv.org/pdf/2505.11417)  

**Abstract**: This paper introduces a novel dataset and evaluation benchmark designed to assess and improve small language models deployable on edge devices, with a focus on user profiling from multi-session natural language interactions in smart home environments. At the core of the dataset are structured user profiles, each defined by a set of routines - context-triggered, repeatable patterns of behavior that govern how users interact with their home systems. Using these profiles as input, a large language model (LLM) generates corresponding interaction sessions that simulate realistic, diverse, and context-aware dialogues between users and their devices.
The primary task supported by this dataset is profile reconstruction: inferring user routines and preferences solely from interactions history. To assess how well current models can perform this task under realistic conditions, we benchmarked several state-of-the-art compact language models and compared their performance against large foundation models. Our results show that while small models demonstrate some capability in reconstructing profiles, they still fall significantly short of large models in accurately capturing user behavior. This performance gap poses a major challenge - particularly because on-device processing offers critical advantages, such as preserving user privacy, minimizing latency, and enabling personalized experiences without reliance on the cloud. By providing a realistic, structured testbed for developing and evaluating behavioral modeling under these constraints, our dataset represents a key step toward enabling intelligent, privacy-respecting AI systems that learn and adapt directly on user-owned devices. 

**Abstract (ZH)**: 一篇关于评估和提高适用于边缘设备的小型语言模型的新数据集和评价基准的论文，重点在于智能家庭环境中多会话自然语言交互的用户画像。该数据集的核心是结构化的用户画像，每个画像由一组受情境触发、可重复的行为模式定义，这些模式规范了用户与家庭系统交互的方式。利用这些画像作为输入，一个大型语言模型（LLM）生成相应的交互会话，模拟用户与设备之间真实、多样化且情境意识的对话。该数据集的主要任务是画像重构：仅从交互历史中推断出用户的习惯和偏好。为评估当前模型在现实条件下的表现，我们基准测试了几种最先进的紧凑型语言模型，并将其性能与大型基础模型进行了对比。结果显示，虽然小型模型在重构画像方面显示出一定的能力，但在准确捕捉用户行为方面仍远逊于大型模型。这一性能差距构成了一个重大挑战，尤其是因为基于设备的处理为保护用户隐私、减少延迟和提供无需依赖云的个性化体验带来了关键优势。通过提供一个在这些约束条件下开发和评估行为建模的现实、结构化测试环境，我们的数据集代表了迈向能够直接在用户自有设备上学习和适应的智能、尊重隐私的AI系统的重要一步。 

---
# MID-L: Matrix-Interpolated Dropout Layer with Layer-wise Neuron Selection 

**Title (ZH)**: MID-L: 基于层内神经元选择的矩阵插值dropout层 

**Authors**: Pouya Shaeri, Ariane Middel  

**Link**: [PDF](https://arxiv.org/pdf/2505.11416)  

**Abstract**: Modern neural networks often activate all neurons for every input, leading to unnecessary computation and inefficiency. We introduce Matrix-Interpolated Dropout Layer (MID-L), a novel module that dynamically selects and activates only the most informative neurons by interpolating between two transformation paths via a learned, input-dependent gating vector. Unlike conventional dropout or static sparsity methods, MID-L employs a differentiable Top-k masking strategy, enabling per-input adaptive computation while maintaining end-to-end differentiability. MID-L is model-agnostic and integrates seamlessly into existing architectures. Extensive experiments on six benchmarks, including MNIST, CIFAR-10, CIFAR-100, SVHN, UCI Adult, and IMDB, show that MID-L achieves up to average 55\% reduction in active neurons, 1.7$\times$ FLOPs savings, and maintains or exceeds baseline accuracy. We further validate the informativeness and selectivity of the learned neurons via Sliced Mutual Information (SMI) and observe improved robustness under overfitting and noisy data conditions. Additionally, MID-L demonstrates favorable inference latency and memory usage profiles, making it suitable for both research exploration and deployment on compute-constrained systems. These results position MID-L as a general-purpose, plug-and-play dynamic computation layer, bridging the gap between dropout regularization and efficient inference. 

**Abstract (ZH)**: 现代神经网络常为每个输入激活所有神经元，导致不必要的计算和低效。我们引入了矩阵内插丢弃层（MID-L），这是一种新型模块，通过学习的输入依赖门控向量在两种变换路径之间进行内插，动态地选择并激活最具信息性的神经元。不同于传统的丢弃或静态稀疏性方法，MID-L 使用可微分的Top-k掩码策略，实现输入适配的计算同时保持端到端可微性。MID-L 兼容性强且无缝集成到现有架构中。在包括MNIST、CIFAR-10、CIFAR-100、SVHN、UCI Adult 和 IMDB在内的六个基准测试上进行的实验表明，MID-L 可以将活跃神经元减少多达平均55%，节省1.7倍的FLOPs，并且保持或超越基线准确度。此外，通过Sliced Mutual Information (SMI) 验证所学习的神经元的信息性和选择性，并观察到在过拟合和嘈杂数据条件下鲁棒性增强。另外，MID-L 展现了有利的推理延迟和内存使用性能，使其适用于计算受限系统的研究探索和部署。这些结果将MID-L 定位为一种通用型、即插即用的动态计算层，填补了丢弃正则化与高效推理之间的差距。 

---
# Visual Planning: Let's Think Only with Images 

**Title (ZH)**: 视觉规划：让我们仅凭图像思考。 

**Authors**: Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, Ivan Vulić  

**Link**: [PDF](https://arxiv.org/pdf/2505.11409)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have substantially enhanced machine reasoning across diverse tasks. However, these models predominantly rely on pure text as the medium for both expressing and structuring reasoning, even when visual information is present. In this work, we argue that language may not always be the most natural or effective modality for reasoning, particularly in tasks involving spatial and geometrical information. Motivated by this, we propose a new paradigm, Visual Planning, which enables planning through purely visual representations, independent of text. In this paradigm, planning is executed via sequences of images that encode step-by-step inference in the visual domain, akin to how humans sketch or visualize future actions. We introduce a novel reinforcement learning framework, Visual Planning via Reinforcement Learning (VPRL), empowered by GRPO for post-training large vision models, leading to substantial improvements in planning in a selection of representative visual navigation tasks, FrozenLake, Maze, and MiniBehavior. Our visual planning paradigm outperforms all other planning variants that conduct reasoning in the text-only space. Our results establish Visual Planning as a viable and promising alternative to language-based reasoning, opening new avenues for tasks that benefit from intuitive, image-based inference. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）及其多模态扩展（MLLMs）在跨多样化任务中的机器推理方面取得了显著进展。然而，这些模型主要依赖纯文本作为表达和结构化推理的媒介，即使存在视觉信息也是如此。在本文中，我们argue认为，特别是在涉及空间和几何信息的任务中，语言可能并不是最自然或最有效的推理模态。受此启发，我们提出了一个新的范式——视觉规划，该范式通过纯视觉表示进行规划，独立于文本。在这个范式中，规划是通过序列图像来执行的，这些图像在视觉领域中编码逐步推理，类似于人类如何草图或可视化未来动作。我们介绍了一个由GRPO赋能的新强化学习框架——基于强化学习的视觉规划（VPRL），该框架极大地提高了在冰湖（FrozenLake）、迷宫（Maze）和迷你行为（MiniBehavior）等代表性视觉导航任务中的规划性能。我们的视觉规划范式在纯文本空间推理的所有其他变体中均表现出更优的性能。我们的研究结果确立了视觉规划作为语言基于推理的可行且有前景的替代方案的地位，为受益于直观、基于图像的推理的任务开辟了新的途径。 

---
# Large Language Model Use Impact Locus of Control 

**Title (ZH)**: 大型语言模型使用对控制源维度的影响 

**Authors**: Jenny Xiyu Fu, Brennan Antone, Kowe Kadoma, Malte Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.11406)  

**Abstract**: As AI tools increasingly shape how we write, they may also quietly reshape how we perceive ourselves. This paper explores the psychological impact of co-writing with AI on people's locus of control. Through an empirical study with 462 participants, we found that employment status plays a critical role in shaping users' reliance on AI and their locus of control. Current results demonstrated that employed participants displayed higher reliance on AI and a shift toward internal control, while unemployed users tended to experience a reduction in personal agency. Through quantitative results and qualitative observations, this study opens a broader conversation about AI's role in shaping personal agency and identity. 

**Abstract (ZH)**: 随着AI工具越来越多地影响我们的写作方式，它们也可能悄然改变我们对自己认知的方式。本文探讨了与AI合写对人们控制来源的心理影响。通过一项包含462名参与者的实证研究，我们发现就业状态在塑造用户对AI的依赖程度及其控制来源方面起着关键作用。当前结果表明，已就业的参与者表现出更高的AI依赖度和内部控制倾向，而失业用户则倾向于感受到个人自主性的减弱。通过定量结果和定性观察，本文开启了一场关于AI在塑造个人自主性和身份方面作用的更广泛讨论。 

---
# Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert Reasoner 

**Title (ZH)**: Patho-R1：一种基于多模态强化学习的病理专家推理器 

**Authors**: Wenchuan Zhang, Penghao Zhang, Jingru Guo, Tao Cheng, Jie Chen, Shuwan Zhang, Zhang Zhang, Yuhao Yi, Hong Bu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11404)  

**Abstract**: Recent advances in vision language models (VLMs) have enabled broad progress in the general medical field. However, pathology still remains a more challenging subdomain, with current pathology specific VLMs exhibiting limitations in both diagnostic accuracy and reasoning plausibility. Such shortcomings are largely attributable to the nature of current pathology datasets, which are primarily composed of image description pairs that lack the depth and structured diagnostic paradigms employed by real world pathologists. In this study, we leverage pathology textbooks and real world pathology experts to construct high-quality, reasoning-oriented datasets. Building on this, we introduce Patho-R1, a multimodal RL-based pathology Reasoner, trained through a three-stage pipeline: (1) continued pretraining on 3.5 million image-text pairs for knowledge infusion; (2) supervised fine-tuning on 500k high-quality Chain-of-Thought samples for reasoning incentivizing; (3) reinforcement learning using Group Relative Policy Optimization and Decoupled Clip and Dynamic sAmpling Policy Optimization strategies for multimodal reasoning quality refinement. To further assess the alignment quality of our dataset, we propose PathoCLIP, trained on the same figure-caption corpus used for continued pretraining. Comprehensive experimental results demonstrate that both PathoCLIP and Patho-R1 achieve robust performance across a wide range of pathology-related tasks, including zero-shot classification, cross-modal retrieval, Visual Question Answering, and Multiple Choice Question. Our project is available at the Patho-R1 repository: this https URL. 

**Abstract (ZH)**: Recent Advances in Vision Language Models for Pathology: Leveraging High-Quality Datasets and Reasoning-Oriented Training for Enhanced Diagnostic Accuracy and Reasoning Plausibility 

---
# Phare: A Safety Probe for Large Language Models 

**Title (ZH)**: Phare: 大型语言模型的安全探针 

**Authors**: Pierre Le Jeune, Benoît Malésieux, Weixuan Xiao, Matteo Dora  

**Link**: [PDF](https://arxiv.org/pdf/2505.11365)  

**Abstract**: Ensuring the safety of large language models (LLMs) is critical for responsible deployment, yet existing evaluations often prioritize performance over identifying failure modes. We introduce Phare, a multilingual diagnostic framework to probe and evaluate LLM behavior across three critical dimensions: hallucination and reliability, social biases, and harmful content generation. Our evaluation of 17 state-of-the-art LLMs reveals patterns of systematic vulnerabilities across all safety dimensions, including sycophancy, prompt sensitivity, and stereotype reproduction. By highlighting these specific failure modes rather than simply ranking models, Phare provides researchers and practitioners with actionable insights to build more robust, aligned, and trustworthy language systems. 

**Abstract (ZH)**: 确保大型语言模型的安全是负责任部署的关键，但现有评估往往侧重于性能而忽视了故障模式的识别。我们引入Phare，一种多语言诊断框架，用于探测和评估大型语言模型在三个关键维度上的行为：幻觉与可靠性、社会偏见以及有害内容生成。我们对17个最先进的大型语言模型的评估揭示了在所有安全维度上的系统性漏洞模式，包括阿谀奉承、提示敏感性和刻板印象再现。通过强调这些具体的故障模式而非单纯排名，Phare为研究人员和实务工作者提供了可操作的见解，以构建更 robust、更对齐和更值得信赖的语言系统。 

---
# DecompileBench: A Comprehensive Benchmark for Evaluating Decompilers in Real-World Scenarios 

**Title (ZH)**: DecompileBench：一种全面的基准测试，用于评估实际场景中去编译器的性能 

**Authors**: Zeyu Gao, Yuxin Cui, Hao Wang, Siliang Qin, Yuanda Wang, Bolun Zhang, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11340)  

**Abstract**: Decompilers are fundamental tools for critical security tasks, from vulnerability discovery to malware analysis, yet their evaluation remains fragmented. Existing approaches primarily focus on syntactic correctness through synthetic micro-benchmarks or subjective human ratings, failing to address real-world requirements for semantic fidelity and analyst usability. We present DecompileBench, the first comprehensive framework that enables effective evaluation of decompilers in reverse engineering workflows through three key components: \textit{real-world function extraction} (comprising 23,400 functions from 130 real-world programs), \textit{runtime-aware validation}, and \textit{automated human-centric assessment} using LLM-as-Judge to quantify the effectiveness of decompilers in reverse engineering workflows. Through a systematic comparison between six industrial-strength decompilers and six recent LLM-powered approaches, we demonstrate that LLM-based methods surpass commercial tools in code understandability despite 52.2% lower functionality correctness. These findings highlight the potential of LLM-based approaches to transform human-centric reverse engineering. We open source \href{this https URL}{DecompileBench} to provide a framework to advance research on decompilers and assist security experts in making informed tool selections based on their specific requirements. 

**Abstract (ZH)**: Decompilers在反向工程工作流中的全面评估：从功能提取到自动化人类为中心的评估 

---
# Temporally-Grounded Language Generation: A Benchmark for Real-Time Vision-Language Models 

**Title (ZH)**: 时间驱动的语言生成：实时视觉-语言模型基准 

**Authors**: Keunwoo Peter Yu, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11326)  

**Abstract**: Vision-language models (VLMs) have shown remarkable progress in offline tasks such as image captioning and video question answering. However, real-time interactive environments impose new demands on VLMs, requiring them to generate utterances that are not only semantically accurate but also precisely timed. We identify two core capabilities necessary for such settings -- $\textit{perceptual updating}$ and $\textit{contingency awareness}$ -- and propose a new benchmark task, $\textbf{Temporally-Grounded Language Generation (TGLG)}$, to evaluate them. TGLG requires models to generate utterances in response to streaming video such that both content and timing align with dynamic visual input. To support this benchmark, we curate evaluation datasets from sports broadcasting and egocentric human interaction domains, and introduce a new metric, $\textbf{TRACE}$, to evaluate TGLG by jointly measuring semantic similarity and temporal alignment. Finally, we present $\textbf{Vision-Language Model with Time-Synchronized Interleaving (VLM-TSI)}$, a model that interleaves visual and linguistic tokens in a time-synchronized manner, enabling real-time language generation without relying on turn-based assumptions. Experimental results show that VLM-TSI significantly outperforms a strong baseline, yet overall performance remains modest -- highlighting the difficulty of TGLG and motivating further research in real-time VLMs. Code and data available $\href{this https URL}{here}$. 

**Abstract (ZH)**: 视觉语言模型在实时互动环境中的时间接地语句生成 

---
# Explaining Strategic Decisions in Multi-Agent Reinforcement Learning for Aerial Combat Tactics 

**Title (ZH)**: 多智能体 reinforcement learning 在空战战术中的战略决策解释 

**Authors**: Ardian Selmonaj, Alessandro Antonucci, Adrian Schneider, Michael Rüegsegger, Matthias Sommer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11311)  

**Abstract**: Artificial intelligence (AI) is reshaping strategic planning, with Multi-Agent Reinforcement Learning (MARL) enabling coordination among autonomous agents in complex scenarios. However, its practical deployment in sensitive military contexts is constrained by the lack of explainability, which is an essential factor for trust, safety, and alignment with human strategies. This work reviews and assesses current advances in explainability methods for MARL with a focus on simulated air combat scenarios. We proceed by adapting various explainability techniques to different aerial combat scenarios to gain explanatory insights about the model behavior. By linking AI-generated tactics with human-understandable reasoning, we emphasize the need for transparency to ensure reliable deployment and meaningful human-machine interaction. By illuminating the crucial importance of explainability in advancing MARL for operational defense, our work supports not only strategic planning but also the training of military personnel with insightful and comprehensible analyses. 

**Abstract (ZH)**: 人工智能（AI）正在重塑战略规划，多智能体强化学习（MARL）使自主智能体在复杂场景中的协作成为可能。然而，其在敏感军事环境中的实际部署受到可解释性的限制，可解释性是建立信任、保障安全和与人类策略一致的重要因素。本文回顾并评估了当前MARL可解释性方法的发展，重点关注模拟空战场景。通过将各种解释性技术应用于不同的空中作战场景，我们获得了关于模型行为的解释性洞察。通过将AI生成的战术与人类可理解的推理相结合，我们强调了透明度的重要性，以确保可靠的部署和有意义的人机交互。通过阐述可解释性在推动MARL在作战防御中的应用的重要性，我们的工作不仅支持战略规划，还通过提供深入且易懂的分析来培训军事人员。 

---
# Heterogeneity-Aware Client Sampling: A Unified Solution for Consistent Federated Learning 

**Title (ZH)**: 面向异质性的客户端采样： federated learning 的统一解决方案 

**Authors**: Shudi Weng, Chao Ren, Ming Xiao, Mikael Skoglund  

**Link**: [PDF](https://arxiv.org/pdf/2505.11304)  

**Abstract**: Federated learning (FL) commonly involves clients with diverse communication and computational capabilities. Such heterogeneity can significantly distort the optimization dynamics and lead to objective inconsistency, where the global model converges to an incorrect stationary point potentially far from the pursued optimum. Despite its critical impact, the joint effect of communication and computation heterogeneity has remained largely unexplored, due to the intrinsic complexity of their interaction. In this paper, we reveal the fundamentally distinct mechanisms through which heterogeneous communication and computation drive inconsistency in FL. To the best of our knowledge, this is the first unified theoretical analysis of general heterogeneous FL, offering a principled understanding of how these two forms of heterogeneity jointly distort the optimization trajectory under arbitrary choices of local solvers. Motivated by these insights, we propose Federated Heterogeneity-Aware Client Sampling, FedACS, a universal method to eliminate all types of objective inconsistency. We theoretically prove that FedACS converges to the correct optimum at a rate of $O(1/\sqrt{R})$, even in dynamic heterogeneous environments. Extensive experiments across multiple datasets show that FedACS outperforms state-of-the-art and category-specific baselines by 4.3%-36%, while reducing communication costs by 22%-89% and computation loads by 14%-105%, respectively. 

**Abstract (ZH)**: 联邦学习中异构通信和计算的联合效应及其理论分析：一种消除目标不一致性的联邦异构感知客户端采样方法 

---
# Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs 

**Title (ZH)**: 思考中的搜索与精炼：LLMs的自主检索增强推理 

**Authors**: Yaorui Shi, Shihan Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An Zhang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11277)  

**Abstract**: Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively. 

**Abstract (ZH)**: Large语言模型展示了 impressive 的推理能力但固有地受限于其知识库。检索增强推理通过允许LLMs查询外部资源减轻了这一限制，但现有方法往往检索无关或噪声信息，妨碍了准确的推理。在本文中，我们提出了一种名为AutoRefine的强化学习后训练框架，采用了新的“搜索和推理期间细化”的范式。AutoRefine在连续搜索调用之间引入了显式的知识细化步骤，使模型能够迭代地过滤、提炼和组织证据，然后再生成答案。此外，我们结合使用了针对检索的定制奖励和答案正确性奖励，采用了组相对策略优化方法。在单跳和多跳问答基准测试上的实验表明，AutoRefine 显著优于现有方法，特别是在复杂的多跳推理场景中。详细的分析表明，AutoRefine 频繁执行高质量的搜索，并有效地综合信息。 

---
# TCC-Bench: Benchmarking the Traditional Chinese Culture Understanding Capabilities of MLLMs 

**Title (ZH)**: TCC-Bench: 传统中文文化理解能力评测 

**Authors**: Pengju Xu, Yan Wang, Shuyuan Zhang, Xuan Zhou, Xin Li, Yue Yuan, Fengzhao Li, Shunyuan Zhou, Xingyu Wang, Yi Zhang, Haiying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11275)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) have significantly enhanced the ability of artificial intelligence systems to understand and generate multimodal content. However, these models often exhibit limited effectiveness when applied to non-Western cultural contexts, which raises concerns about their wider applicability. To address this limitation, we propose the \textbf{T}raditional \textbf{C}hinese \textbf{C}ulture understanding \textbf{Bench}mark (\textbf{TCC-Bench}), a bilingual (\textit{i.e.}, Chinese and English) Visual Question Answering (VQA) benchmark specifically designed for assessing the understanding of traditional Chinese culture by MLLMs. TCC-Bench comprises culturally rich and visually diverse data, incorporating images from museum artifacts, everyday life scenes, comics, and other culturally significant contexts. We adopt a semi-automated pipeline that utilizes GPT-4o in text-only mode to generate candidate questions, followed by human curation to ensure data quality and avoid potential data leakage. The benchmark also avoids language bias by preventing direct disclosure of cultural concepts within question texts. Experimental evaluations across a wide range of MLLMs demonstrate that current models still face significant challenges when reasoning about culturally grounded visual content. The results highlight the need for further research in developing culturally inclusive and context-aware multimodal systems. The code and data can be found at: this https URL. 

**Abstract (ZH)**: Recent进展在多模态大语言模型（MLLMs）在理解和生成多模态内容方面取得了显著增强。然而，当应用于非西方文化背景时，这些模型经常表现出有限的有效性，这对其更广泛的应用范围提出了质疑。为了解决这一局限性，我们提出了传统中国文化理解基准（TCC-Bench），这是一个双语（即，中文和英文）视觉问答（VQA）基准，专门用于评估MLLMs对传统中国文化理解的能力。TCC-Bench包含丰富的文化和视觉多样的数据，涵盖博物院藏品图像、日常生活场景、漫画以及其他文化重要背景。我们采用了一种半自动的工作流程，利用GPT-4o仅文本模式生成候选问题，随后由人工审查确保数据质量和避免潜在的数据泄露。基准还通过防止在问题文本中直接披露文化概念来避免语言偏见。跨多种MLLMs的实验评估表明，当前的模型在推理文化基础的视觉内容时仍面临重大挑战。结果突显了进一步研究开发文化包容性和情境感知多模态系统的需求。相关代码和数据可在以下链接找到：this https URL。 

---
# Semantic Caching of Contextual Summaries for Efficient Question-Answering with Language Models 

**Title (ZH)**: 基于语境摘要的语义缓存以实现高效的语言模型问答 

**Authors**: Camille Couturier, Spyros Mastorakis, Haiying Shen, Saravan Rajmohan, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2505.11271)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across edge and cloud platforms for real-time question-answering and retrieval-augmented generation. However, processing lengthy contexts in distributed systems incurs high computational overhead, memory usage, and network bandwidth. This paper introduces a novel semantic caching approach for storing and reusing intermediate contextual summaries, enabling efficient information reuse across similar queries in LLM-based QA workflows. Our method reduces redundant computations by up to 50-60% while maintaining answer accuracy comparable to full document processing, as demonstrated on NaturalQuestions, TriviaQA, and a synthetic ArXiv dataset. This approach balances computational cost and response quality, critical for real-time AI assistants. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地部署在边缘和云平台用于实时问答和检索增强生成。然而，在分布式系统中处理长上下文会带来高计算开销、内存使用和网络带宽的问题。本文提出了一种新颖的语义缓存方法，用于存储和重用中间上下文摘要，从而在基于LLM的问答流程中实现高效的信息重用。该方法在保持与全文处理相似的答案准确性的同时，通过自然问题、TrivialQA和一个合成的ArXiv数据集上的实验，减少了高达50-60%的冗余计算。该方法平衡了计算成本和响应质量，对于实时AI助手至关重要。 

---
# TAIJI: MCP-based Multi-Modal Data Analytics on Data Lakes 

**Title (ZH)**: TAIJI: 基于MCP的多模态数据湖上的数据分析 

**Authors**: Chao Zhang, Shaolei Zhang, Quehuan Liu, Sibei Chen, Tong Li, Ju Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11270)  

**Abstract**: The variety of data in data lakes presents significant challenges for data analytics, as data scientists must simultaneously analyze multi-modal data, including structured, semi-structured, and unstructured data. While Large Language Models (LLMs) have demonstrated promising capabilities, they still remain inadequate for multi-modal data analytics in terms of accuracy, efficiency, and freshness. First, current natural language (NL) or SQL-like query languages may struggle to precisely and comprehensively capture users' analytical intent. Second, relying on a single unified LLM to process diverse data modalities often leads to substantial inference overhead. Third, data stored in data lakes may be incomplete or outdated, making it essential to integrate external open-domain knowledge to generate timely and relevant analytics results.
In this paper, we envision a new multi-modal data analytics system. Specifically, we propose a novel architecture built upon the Model Context Protocol (MCP), an emerging paradigm that enables LLMs to collaborate with knowledgeable agents. First, we define a semantic operator hierarchy tailored for querying multi-modal data in data lakes and develop an AI-agent-powered NL2Operator translator to bridge user intent and analytical execution. Next, we introduce an MCP-based execution framework, in which each MCP server hosts specialized foundation models optimized for specific data modalities. This design enhances both accuracy and efficiency, while supporting high scalability through modular deployment. Finally, we propose a updating mechanism by harnessing the deep research and machine unlearning techniques to refresh the data lakes and LLM knowledges, with the goal of balancing the data freshness and inference efficiency. 

**Abstract (ZH)**: 多模态数据湖中数据的多样性为数据analytics带来了巨大挑战，数据科学家必须同时分析结构化、半结构化和非结构化等多种模态的数据。尽管大型语言模型（LLMs）显示出了潜在的能力，但在准确度、效率和新鲜度方面仍然无法满足多模态数据分析的需求。首先，当前的自然语言（NL）或SQL-like查询语言可能难以精确且全面地捕捉用户的数据分析意图。其次，依赖单一的统一LLM处理多样化模态的数据通常会导致显著的推理开销。最后，存储在数据湖中的数据可能不完整或过时，因此有必要整合开放领域知识以生成及时且相关的结果。

在本文中，我们设想了一种新的多模态数据分析系统。具体而言，我们提出了基于模型上下文协议（MCP）的新架构，这是一种新兴的范式，可以实现LLMs与知识型代理的协作。首先，我们定义了一个针对数据湖中多模态数据查询的语义操作符层次结构，并开发了一个基于AI代理的NL2Operator翻译器，以连接用户意图与分析执行。其次，我们介绍了基于MCP的执行框架，在该框架中，每个MCP服务器托管针对特定数据模态优化的基础模型。这一设计提高了准确性和效率，并通过模块化部署支持高可扩展性。最后，我们提出了一种更新机制，利用深度研究和机器遗忘技术来刷新数据湖和LLM的知识，旨在平衡数据新鲜度和推理效率。 

---
# Equal is Not Always Fair: A New Perspective on Hyperspectral Representation Non-Uniformity 

**Title (ZH)**: 公平并不总是公正：超谱表示非均匀性的新视角 

**Authors**: Wuzhou Quan, Mingqiang Wei, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11267)  

**Abstract**: Hyperspectral image (HSI) representation is fundamentally challenged by pervasive non-uniformity, where spectral dependencies, spatial continuity, and feature efficiency exhibit complex and often conflicting behaviors. Most existing models rely on a unified processing paradigm that assumes homogeneity across dimensions, leading to suboptimal performance and biased representations. To address this, we propose FairHyp, a fairness-directed framework that explicitly disentangles and resolves the threefold non-uniformity through cooperative yet specialized modules. We introduce a Runge-Kutta-inspired spatial variability adapter to restore spatial coherence under resolution discrepancies, a multi-receptive field convolution module with sparse-aware refinement to enhance discriminative features while respecting inherent sparsity, and a spectral-context state space model that captures stable and long-range spectral dependencies via bidirectional Mamba scanning and statistical aggregation. Unlike one-size-fits-all solutions, FairHyp achieves dimension-specific adaptation while preserving global consistency and mutual reinforcement. This design is grounded in the view that non-uniformity arises from the intrinsic structure of HSI representations, rather than any particular task setting. To validate this, we apply FairHyp across four representative tasks including classification, denoising, super-resolution, and inpaintin, demonstrating its effectiveness in modeling a shared structural flaw. Extensive experiments show that FairHyp consistently outperforms state-of-the-art methods under varied imaging conditions. Our findings redefine fairness as a structural necessity in HSI modeling and offer a new paradigm for balancing adaptability, efficiency, and fidelity in high-dimensional vision tasks. 

**Abstract (ZH)**: 基于公平性的超光谱图像表示框架：解决普遍存在的非均匀性问题 

---
# A Set-Sequence Model for Time Series 

**Title (ZH)**: 时间序列的集序列模型 

**Authors**: Elliot L. Epstein, Apaar Sadhwani, Kay Giesecke  

**Link**: [PDF](https://arxiv.org/pdf/2505.11243)  

**Abstract**: In many financial prediction problems, the behavior of individual units (such as loans, bonds, or stocks) is influenced by observable unit-level factors and macroeconomic variables, as well as by latent cross-sectional effects. Traditional approaches attempt to capture these latent effects via handcrafted summary features. We propose a Set-Sequence model that eliminates the need for handcrafted features. The Set model first learns a shared cross-sectional summary at each period. The Sequence model then ingests the summary-augmented time series for each unit independently to predict its outcome. Both components are learned jointly over arbitrary sets sampled during training. Our approach harnesses the set nature of the cross-section and is computationally efficient, generating set summaries in linear time relative to the number of units. It is also flexible, allowing the use of existing sequence models and accommodating a variable number of units at inference. Empirical evaluations demonstrate that our Set-Sequence model significantly outperforms benchmarks on stock return prediction and mortgage behavior tasks. Code will be released. 

**Abstract (ZH)**: 在许多金融预测问题中，个体单位（如贷款、债券或股票）的行为受到可观测的单位级因素、宏观经济学变量以及潜在的横截面效应的影响。传统方法试图通过手工设计的摘要特征来捕捉这些潜在效应。我们提出了一个Set-Sequence模型，以消除手工设计特征的需求。Set模型首先在每个时期学习一个共享的横截面摘要。Sequence模型然后独立地摄取每个单位的摘要增强时间序列，以预测其结果。两个组件在训练期间任意采样的集合中联合学习。我们的方法利用了横截面的集合并行性质，计算效率高，生成集合摘要的时间复杂度与单位数量成线性关系。它还具有灵活性，允许使用现有的序列模型，并在推断时容纳数量可变的单位。实证评估表明，我们的Set-Sequence模型在股票回报预测和抵押行为任务中显著优于基准模型。代码将开源。 

---
# Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI models in Sound Localization 

**Title (ZH)**: Seeing Sound, Hearing Sight: 探索AI模型在声音定位中的模态偏差与冲突 

**Authors**: Yanhao Jia, Ji Xie, S Jivaganesh, Hao Li, Xu Wu, Mengmi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11217)  

**Abstract**: Imagine hearing a dog bark and turning toward the sound only to see a parked car, while the real, silent dog sits elsewhere. Such sensory conflicts test perception, yet humans reliably resolve them by prioritizing sound over misleading visuals. Despite advances in multimodal AI integrating vision and audio, little is known about how these systems handle cross-modal conflicts or whether they favor one modality. In this study, we systematically examine modality bias and conflict resolution in AI sound localization. We assess leading multimodal models and benchmark them against human performance in psychophysics experiments across six audiovisual conditions, including congruent, conflicting, and absent cues. Humans consistently outperform AI, demonstrating superior resilience to conflicting or missing visuals by relying on auditory information. In contrast, AI models often default to visual input, degrading performance to near chance levels. To address this, we finetune a state-of-the-art model using a stereo audio-image dataset generated via 3D simulations. Even with limited training data, the refined model surpasses existing benchmarks. Notably, it also mirrors human-like horizontal localization bias favoring left-right precision-likely due to the stereo audio structure reflecting human ear placement. These findings underscore how sensory input quality and system architecture shape multimodal representation accuracy. 

**Abstract (ZH)**: 多模态AI中的模态偏向与冲突解决研究 

---
# Bayesian Hierarchical Invariant Prediction 

**Title (ZH)**: 贝叶斯层次不变预测 

**Authors**: Francisco Madaleno, Pernille Julie Viuff Sand, Francisco C. Pereira, Sergio Hernan Garrido Mejia  

**Link**: [PDF](https://arxiv.org/pdf/2505.11211)  

**Abstract**: We propose Bayesian Hierarchical Invariant Prediction (BHIP) reframing Invariant Causal Prediction (ICP) through the lens of Hierarchical Bayes. We leverage the hierarchical structure to explicitly test invariance of causal mechanisms under heterogeneous data, resulting in improved computational scalability for a larger number of predictors compared to ICP. Moreover, given its Bayesian nature BHIP enables the use of prior information. In this paper, we test two sparsity inducing priors: horseshoe and spike-and-slab, both of which allow us a more reliable identification of causal features. We test BHIP in synthetic and real-world data showing its potential as an alternative inference method to ICP. 

**Abstract (ZH)**: Bayesian Hierarchical Invariant Prediction: Reframing Invariant Causal Prediction Through the Lens of Hierarchical Bayes 

---
# RanDeS: Randomized Delta Superposition for Multi-Model Compression 

**Title (ZH)**: RanDeS: 随机Delta 超position 多模型压缩 

**Authors**: Hangyu Zhou, Aaron Gokaslan, Volodymyr Kuleshov, Bharath Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11204)  

**Abstract**: From a multi-model compression perspective, model merging enables memory-efficient serving of multiple models fine-tuned from the same base, but suffers from degraded performance due to interference among their task-specific parameter adjustments (i.e., deltas). In this paper, we reformulate model merging as a compress-and-retrieve scheme, revealing that the task interference arises from the summation of irrelevant deltas during model retrieval. To address this issue, we use random orthogonal transformations to decorrelate these vectors into self-cancellation. We show that this approach drastically reduces interference, improving performance across both vision and language tasks. Since these transformations are fully defined by random seeds, adding new models requires no extra memory. Further, their data- and model-agnostic nature enables easy addition or removal of models with minimal compute overhead, supporting efficient and flexible multi-model serving. 

**Abstract (ZH)**: 从多模型压缩视角出发，模型合并能够在不牺牲太多性能的情况下高效服务于多个来自同一基础模型的精调模型，但任务特定参数调整（即差异）之间的相互干扰会导致性能下降。在本文中，我们将模型合并重新表述为一种压缩和检索方案，揭示了任务干扰源于模型检索过程中无关差异的叠加。为了解决这一问题，我们使用随机正交变换将这些向量去相关化，使其自我抵消。我们证明了这种方法大幅减少了干扰，提高了视觉和语言任务的性能。由于这些变换完全由随机种子定义，增加新模型无需额外内存。此外，它们的数据和模型无关性使得在最小计算开销下轻松添加或移除模型，支持高效的多模型服务。 

---
# Audio Turing Test: Benchmarking the Human-likeness of Large Language Model-based Text-to-Speech Systems in Chinese 

**Title (ZH)**: 中文标题：声音图灵测试：基于大规模语言模型的文本转语音系统在汉语中的人类相似性评估 

**Authors**: Xihuai Wang, Ziyi Zhao, Siyu Ren, Shao Zhang, Song Li, Xiaoyu Li, Ziwen Wang, Lin Qiu, Guanglu Wan, Xuezhi Cao, Xunliang Cai, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11200)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved text-to-speech (TTS) systems, enhancing control over speech style, naturalness, and emotional expression, which brings TTS Systems closer to human-level performance. Although the Mean Opinion Score (MOS) remains the standard for TTS System evaluation, it suffers from subjectivity, environmental inconsistencies, and limited interpretability. Existing evaluation datasets also lack a multi-dimensional design, often neglecting factors such as speaking styles, context diversity, and trap utterances, which is particularly evident in Chinese TTS evaluation. To address these challenges, we introduce the Audio Turing Test (ATT), a multi-dimensional Chinese corpus dataset ATT-Corpus paired with a simple, Turing-Test-inspired evaluation protocol. Instead of relying on complex MOS scales or direct model comparisons, ATT asks evaluators to judge whether a voice sounds human. This simplification reduces rating bias and improves evaluation robustness. To further support rapid model development, we also finetune Qwen2-Audio-Instruct with human judgment data as Auto-ATT for automatic evaluation. Experimental results show that ATT effectively differentiates models across specific capability dimensions using its multi-dimensional design. Auto-ATT also demonstrates strong alignment with human evaluations, confirming its value as a fast and reliable assessment tool. The white-box ATT-Corpus and Auto-ATT can be found in ATT Hugging Face Collection (this https URL). 

**Abstract (ZH)**: 最近大型语言模型的发展显著提升了文本到语音（TTS）系统的性能，增强了语音风格控制、自然度和情感表达，使TTS系统更接近人类水平的表现。尽管平均意见得分（MOS）仍然是TTS系统评估的标准，但其仍然存在主观性、环境不一致性和解释性有限等问题。现有的评估数据集缺乏多维度设计，常忽略说话风格、背景多样性等因素，尤其是在中文TTS评估中更为明显。为应对这些挑战，我们提出了音频图灵测试（ATT），这是一个多维度的中文语料库数据集及其配套的简单图灵测试启发式评估协议。ATT 不依靠复杂的MOS量表或直接模型对比，而是要求评估者判断声音是否听起来像人声。这一简化减少了评分偏差并提高了评估的稳健性。为进一步支持快速模型开发，我们还将Qwen2-Audio-Instruct微调为具有人类判断数据的Auto-ATT，用于自动评估。实验结果表明，ATT通过其多维度设计有效区分了模型在特定能力维度上的表现。Auto-ATT也与人类评估高度一致，证实了其作为快速且可靠评估工具的价值。白盒ATT-Corpus和Auto-ATT可以在ATT Hugging Face Collection（这个链接）中找到。 

---
# User-centric Music Recommendations 

**Title (ZH)**: 用户中心的音乐推荐 

**Authors**: Jaime Ramirez Castillo, M. Julia Flores, Ann E. Nicholson  

**Link**: [PDF](https://arxiv.org/pdf/2505.11198)  

**Abstract**: This work presents a user-centric recommendation framework, designed as a pipeline with four distinct, connected, and customizable phases. These phases are intended to improve explainability and boost user engagement.
We have collected the historical this http URL track playback records of a single user over approximately 15 years. The collected dataset includes more than 90,000 playbacks and approximately 14,000 unique tracks.
From track playback records, we have created a dataset of user temporal contexts (each row is a specific moment when the user listened to certain music descriptors). As music descriptors, we have used community-contributed this http URL tags and Spotify audio features. They represent the music that, throughout years, the user has been listening to.
Next, given the most relevant this http URL tags of a moment (e.g. the hour of the day), we predict the Spotify audio features that best fit the user preferences in that particular moment. Finally, we use the predicted audio features to find tracks similar to these features. The final aim is to recommend (and discover) tracks that the user may feel like listening to at a particular moment.
For our initial study case, we have chosen to predict only a single audio feature target: danceability. The framework, however, allows to include more target variables.
The ability to learn the musical habits from a single user can be quite powerful, and this framework could be extended to other users. 

**Abstract (ZH)**: 用户中心的推荐框架：基于四阶段可定制流水线的音乐推荐与解释性提升 

---
# FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Pretraining 

**Title (ZH)**: FALCON：在视觉-语言预训练中关注负样本的虚假阴性学习 

**Authors**: Myunsoo Kim, Seong-Woong Shim, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.11192)  

**Abstract**: False negatives pose a critical challenge in vision-language pretraining (VLP) due to the many-to-many correspondence between images and texts in large-scale datasets. These false negatives introduce conflicting supervision signals that degrade the learned embedding space and diminish the effectiveness of hard negative sampling. In this paper, we propose FALCON (False-negative Aware Learning of COntrastive Negatives), a learning-based mini-batch construction strategy that adaptively balances the trade-off between hard and false negatives during VLP. Rather than relying on fixed heuristics, FALCON employs a negative mining scheduler that dynamically selects negative samples of appropriate hardness for each anchor instance during mini-batch construction, guided by a proxy for cross-modal alignment improvement. Experimental results demonstrate that FALCON significantly improves performance across two widely adopted VLP frameworks (ALBEF, BLIP-2) and a broad range of downstream tasks and evaluation settings, underscoring its effectiveness and robustness in mitigating the impact of false negatives. 

**Abstract (ZH)**: False Negatives在视觉-语言预训练中的关键挑战：FALCON（基于学习的False-negativeaware对比负样本学习）策略 

---
# Imputation-free and Alignment-free: Incomplete Multi-view Clustering Driven by Consensus Semantic Learning 

**Title (ZH)**: 无填充且无对齐：基于共识语义学习的不完整多视图聚类 

**Authors**: Yuzhuo Dai, Jiaqi Jin, Zhibin Dong, Siwei Wang, Xinwang Liu, En Zhu, Xihong Yang, Xinbiao Gan, Yu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11182)  

**Abstract**: In incomplete multi-view clustering (IMVC), missing data induce prototype shifts within views and semantic inconsistencies across views. A feasible solution is to explore cross-view consistency in paired complete observations, further imputing and aligning the similarity relationships inherently shared across views. Nevertheless, existing methods are constrained by two-tiered limitations: (1) Neither instance- nor cluster-level consistency learning construct a semantic space shared across views to learn consensus semantics. The former enforces cross-view instances alignment, and wrongly regards unpaired observations with semantic consistency as negative pairs; the latter focuses on cross-view cluster counterparts while coarsely handling fine-grained intra-cluster relationships within views. (2) Excessive reliance on consistency results in unreliable imputation and alignment without incorporating view-specific cluster information. Thus, we propose an IMVC framework, imputation- and alignment-free for consensus semantics learning (FreeCSL). To bridge semantic gaps across all observations, we learn consensus prototypes from available data to discover a shared space, where semantically similar observations are pulled closer for consensus semantics learning. To capture semantic relationships within specific views, we design a heuristic graph clustering based on modularity to recover cluster structure with intra-cluster compactness and inter-cluster separation for cluster semantics enhancement. Extensive experiments demonstrate, compared to state-of-the-art competitors, FreeCSL achieves more confident and robust assignments on IMVC task. 

**Abstract (ZH)**: 不完备多视图聚类中缺失数据导致视图内原型偏移和视图间语义不一致。可行的解决方案是在配对的完整观测中探索跨视图一致性，进一步推导和对齐跨视图固有的相似关系。然而，现有方法受到两类限制：（1）实例级和聚类级的一致性学习构造未能建立跨视图共享的语义空间来学习一致语义。前者强加跨视图实例对齐，错误地将具有语义一致性的未配对观测视为负样本对；后者侧重于跨视图聚类对应物，而粗糙地处理视图内部聚类的细粒度关系。（2）过度依赖一致性导致在不结合视图特定聚类信息的情况下进行不可靠的推导和对齐。因此，我们提出了一种用于共识语义学习的无推导和对齐框架（FreeCSL）。为跨越所有观测填补语义差距，我们从可用数据中学习共识原型，发现一个共享空间，在此空间中，语义相似的观测被拉近以进行共识语义学习。为了捕捉特定视图内的语义关系，我们基于模块性的启发式图聚类设计，以恢复具有内部紧凑性和外部分离性的聚类结构，提高聚类语义。大量实验表明，与最先进的竞争对手相比，FreeCSL在不完备多视图聚类任务中实现了更自信和稳健的分配。 

---
# CompAlign: Improving Compositional Text-to-Image Generation with a Complex Benchmark and Fine-Grained Feedback 

**Title (ZH)**: CompAlign: 通过复杂基准和细粒度反馈提高组合文本到图像生成性能 

**Authors**: Yixin Wan, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11178)  

**Abstract**: State-of-the-art T2I models are capable of generating high-resolution images given textual prompts. However, they still struggle with accurately depicting compositional scenes that specify multiple objects, attributes, and spatial relations. We present CompAlign, a challenging benchmark with an emphasis on assessing the depiction of 3D-spatial relationships, for evaluating and improving models on compositional image generation. CompAlign consists of 900 complex multi-subject image generation prompts that combine numerical and 3D-spatial relationships with varied attribute bindings. Our benchmark is remarkably challenging, incorporating generation tasks with 3+ generation subjects with complex 3D-spatial relationships. Additionally, we propose CompQuest, an interpretable and accurate evaluation framework that decomposes complex prompts into atomic sub-questions, then utilizes a MLLM to provide fine-grained binary feedback on the correctness of each aspect of generation elements in model-generated images. This enables precise quantification of alignment between generated images and compositional prompts. Furthermore, we propose an alignment framework that uses CompQuest's feedback as preference signals to improve diffusion models' compositional image generation abilities. Using adjustable per-image preferences, our method is easily scalable and flexible for different tasks. Evaluation of 9 T2I models reveals that: (1) models remarkable struggle more with compositional tasks with more complex 3D-spatial configurations, and (2) a noticeable performance gap exists between open-source accessible models and closed-source commercial models. Further empirical study on using CompAlign for model alignment yield promising results: post-alignment diffusion models achieve remarkable improvements in compositional accuracy, especially on complex generation tasks, outperforming previous approaches. 

**Abstract (ZH)**: 最先进的文本到图像（T2I）模型能够在给定文本提示的情况下生成高分辨率图像。然而，它们在准确描绘包含多个对象、属性和空间关系的组合场景方面仍然存在挑战。我们提出了CompAlign，一个专注于评估3D空间关系表示能力的具有挑战性的基准，用于评估和提升组合图像生成模型。CompAlign 包含900个复杂的多主体图像生成提示，结合了数值和3D空间关系，以及多样的属性绑定。我们的基准具有显著的挑战性，包括生成涉及3个及以上生成主体且具有复杂3D空间关系的任务。此外，我们提出了CompQuest，这是一种可解释且准确的评估框架，将复杂提示分解为原子子问题，然后利用多模态预训练语言模型（MLLM）提供生成元素在模型生成图像中的各个方面的细粒度二元反馈。这使得对生成图像与组合提示之间对齐的精确量化成为可能。此外，我们提出了一种利用CompQuest反馈作为偏好信号以改进扩散模型组合图像生成能力的框架。通过可调节的单个图像偏好，该方法易于扩展和适应不同的任务。评估9个T2I模型发现：(1) 模型在涉及更复杂3D空间配置的组合任务中表现尤为困难，(2) 开源可访问模型与封闭源商业模型之间存在明显的性能差距。进一步使用CompAlign进行模型对齐的经验研究表明，对齐后的扩散模型在组合准确性方面取得了显著改进，尤其是在复杂生成任务中超过了先前的方法。 

---
# Low-Resource Language Processing: An OCR-Driven Summarization and Translation Pipeline 

**Title (ZH)**: 低资源语言处理：一种基于OCR的总结与翻译管道 

**Authors**: Hrishit Madhavi, Jacob Cherian, Yuvraj Khamkar, Dhananjay Bhagat  

**Link**: [PDF](https://arxiv.org/pdf/2505.11177)  

**Abstract**: This paper presents an end-to-end suite for multilingual information extraction and processing from image-based documents. The system uses Optical Character Recognition (Tesseract) to extract text in languages such as English, Hindi, and Tamil, and then a pipeline involving large language model APIs (Gemini) for cross-lingual translation, abstractive summarization, and re-translation into a target language. Additional modules add sentiment analysis (TensorFlow), topic classification (Transformers), and date extraction (Regex) for better document comprehension. Made available in an accessible Gradio interface, the current research shows a real-world application of libraries, models, and APIs to close the language gap and enhance access to information in image media across different linguistic environments 

**Abstract (ZH)**: 这篇论文提出了一套端到端的多语言信息从图像文档中提取和处理方案。该系统使用光学字符识别(Tesseract)从英语、印地语和泰米尔语等语言的图像文档中提取文本，然后通过大型语言模型APIs(Gemini)进行跨语言翻译、抽象总结以及目标语言的再次翻译。此外，还增加了情感分析（TensorFlow）、主题分类（Transformers）和日期提取（Regex）模块以改善文档理解。该系统通过可访问的Gradio界面提供，当前的研究展示了如何利用库、模型和API在不同语言环境中缩小语言差距并增强图像媒体中的信息访问。 

---
# From Intent Discovery to Recognition with Topic Modeling and Synthetic Data 

**Title (ZH)**: 从意图发现到合成数据和主题建模的意图识别 

**Authors**: Aaron Rodrigues, Mahmood Hegazy, Azzam Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2505.11176)  

**Abstract**: Understanding and recognizing customer intents in AI systems is crucial, particularly in domains characterized by short utterances and the cold start problem, where recommender systems must include new products or services without sufficient real user data. Customer utterances are characterized by infrequent word co-occurences and high term variability, which poses significant challenges for traditional methods in specifying distinct user needs and preparing synthetic queries. To address this, we propose an agentic LLM framework for topic modeling and synthetic query generation, which accelerates the discovery and recognition of customer intents. We first apply hierarchical topic modeling and intent discovery to expand a human-curated taxonomy from 36 generic user intents to 278 granular intents, demonstrating the potential of LLMs to significantly enhance topic specificity and diversity. Next, to support newly discovered intents and address the cold start problem, we generate synthetic user query data, which augments real utterances and reduces dependency on human annotation, especially in low-resource settings. Topic model experiments show substantial improvements in coherence and relevance after topic expansion, while synthetic data experiments indicate that in-class few-shot prompting significantly improves the quality and utility of synthetic queries without compromising diversity. We also show that LLM-generated intent descriptions and keywords can effectively substitute for human-curated versions when used as context for synthetic query generation. Our research underscores the scalability and utility of LLM agents in topic modeling and highlights the strategic use of synthetic utterances to enhance dataset variability and coverage for intent recognition. We present a comprehensive and robust framework for online discovery and recognition of new customer intents in dynamic domains. 

**Abstract (ZH)**: 理解并识别AI系统中的客户意图对于特定领域（如短语句场景和冷启动问题）至关重要。在这些领域中，推荐系统需要包含新產品或服务而缺乏足够的实际用户数据。客户语句的特点是在于罕见词共现和高术语变异性，这为传统方法在明确用户需求和制备合成查询时带来了巨大挑战。为此，我们提出了一种代理性的大语言模型框架，用于主题建模和合成查询生成，以加速客户意图的发现与识别。首先，我们应用分层主题建模和意图发现，将36个通用用户意图扩展到278个细粒度的意图，展示了大语言模型在提高主题特异性和多样性方面的潜力。其次，为了支持新发现的意图并解决冷启动问题，我们生成合成用户查询数据，这些数据增强了真实语句并减少了对人类标注的依赖，尤其是在资源匮乏的环境中。主题模型实验结果显示，在主题扩展后，主题的一致性和相关性有了显著提高，而合成数据实验表明，在类别内少量示例提示可以显著提高合成查询的质量和实用性，同时不损害多样性。我们还展示了在合成查询生成中，LLM生成的意图描述和关键词可以有效地替代人类编纂的版本。我们的研究强调了LLM代理在主题建模中的可扩展性和实用性，并突出了使用合成语句的战略性应用，以增强意图识别的数据多样性和覆盖率。我们提供了一个全面而稳健的框架，用于动态领域中的在线发现和识别新客户意图。 

---
# Real-Time Verification of Embodied Reasoning for Generative Skill Acquisition 

**Title (ZH)**: 实时验证具身推理的生成技能获取 

**Authors**: Bo Yue, Shuqi Guo, Kaiyu Hu, Chujiao Wang, Benyou Wang, Kui Jia, Guiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11175)  

**Abstract**: Generative skill acquisition enables embodied agents to actively learn a scalable and evolving repertoire of control skills, crucial for the advancement of large decision models. While prior approaches often rely on supervision signals from generalist agents (e.g., LLMs), their effectiveness in complex 3D environments remains unclear; exhaustive evaluation incurs substantial computational costs, significantly hindering the efficiency of skill learning. Inspired by recent successes in verification models for mathematical reasoning, we propose VERGSA (Verifying Embodied Reasoning in Generative Skill Acquisition), a framework that systematically integrates real-time verification principles into embodied skill learning. VERGSA establishes 1) a seamless extension from verification of mathematical reasoning into embodied learning by dynamically incorporating contextually relevant tasks into prompts and defining success metrics for both subtasks and overall tasks, and 2) an automated, scalable reward labeling scheme that synthesizes dense reward signals by iteratively finalizing the contribution of scene configuration and subtask learning to overall skill acquisition. To the best of our knowledge, this approach constitutes the first comprehensive training dataset for verification-driven generative skill acquisition, eliminating arduous manual reward engineering. Experiments validate the efficacy of our approach: 1) the exemplar task pool improves the average task success rates by 21%, 2) our verification model boosts success rates by 24% for novel tasks and 36% for encountered tasks, and 3) outperforms LLM-as-a-Judge baselines in verification quality. 

**Abstract (ZH)**: 生成技能习得促进具身代理主动学习可扩展且不断演化的控制技能，对于大型决策模型的发展至关重要。受近期数学推理验证模型成功经验的启发，我们提出了VERGSA（Verifying Embodied Reasoning in Generative Skill Acquisition）框架，该框架系统地将实时验证原则整合到具身技能学习中。VERGSA通过动态纳入与上下文相关任务并为子任务和整体任务定义成功指标，实现从数学推理验证到具身学习的无缝扩展，并提出了一种自动化、可扩展的奖励标签方案，通过迭代确定情景配置和子任务学习对总体技能习得的贡献来合成密集的奖励信号。据我们所知，这是首次为驱动验证的生成性技能习得构建全面的训练数据集，消除了繁琐的手动奖励工程。实验验证了该方法的有效性：1）范例任务池将平均任务成功率提高了21%；2）验证模型对于新任务将成功率提高了24%，对于遇到的任务提高了36%；3）在验证质量上优于LLM-as-a-Judge基线。 

---
# CheX-DS: Improving Chest X-ray Image Classification with Ensemble Learning Based on DenseNet and Swin Transformer 

**Title (ZH)**: CheX-DS：基于DenseNet和Swin Transformer的集成学习改进胸部X光图像分类 

**Authors**: Xinran Li, Yu Liu, Xiujuan Xu, Xiaowei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11168)  

**Abstract**: The automatic diagnosis of chest diseases is a popular and challenging task. Most current methods are based on convolutional neural networks (CNNs), which focus on local features while neglecting global features. Recently, self-attention mechanisms have been introduced into the field of computer vision, demonstrating superior performance. Therefore, this paper proposes an effective model, CheX-DS, for classifying long-tail multi-label data in the medical field of chest X-rays. The model is based on the excellent CNN model DenseNet for medical imaging and the newly popular Swin Transformer model, utilizing ensemble deep learning techniques to combine the two models and leverage the advantages of both CNNs and Transformers. The loss function of CheX-DS combines weighted binary cross-entropy loss with asymmetric loss, effectively addressing the issue of data imbalance. The NIH ChestX-ray14 dataset is selected to evaluate the model's effectiveness. The model outperforms previous studies with an excellent average AUC score of 83.76\%, demonstrating its superior performance. 

**Abstract (ZH)**: 胸部疾病自动诊断是-popular-and-challenging-task-的一项流行且具有挑战性的任务。大多数现有方法基于卷积神经网络(CNNs)，侧重于局部特征而忽视了全局特征。最近，自注意力机制被引入计算机视觉领域，展现了卓越的性能。因此，本文提出了一种有效的模型CheX-DS，用于分类胸部X光片医学领域中长尾多标签数据。该模型基于用于医学成像的优秀CNN模型DenseNet和新兴流行的Swin Transformer模型，利用集成深度学习技术结合这两种模型，充分发挥CNN和Transformer的优势。CheX-DS的损失函数结合了加权二元交叉熵损失和非对称损失，有效解决了数据不平衡问题。该模型在NIH ChestX-ray14数据集上的评估结果显示，其平均AUC分数为83.76%，展示了其优越性能。 

---
# SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization 

**Title (ZH)**: SoLoPO：通过短到长偏好优化解锁LLM的长上下文能力 

**Authors**: Huashan Sun, Shengyi Liao, Yansen Han, Yu Bai, Yang Gao, Cheng Fu, Weizhou Shen, Fanqi Wan, Ming Yan, Ji Zhang, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11166)  

**Abstract**: Despite advances in pretraining with extended context lengths, large language models (LLMs) still face challenges in effectively utilizing real-world long-context information, primarily due to insufficient long-context alignment caused by data quality issues, training inefficiencies, and the lack of well-designed optimization objectives. To address these limitations, we propose a framework named $\textbf{S}$h$\textbf{o}$rt-to-$\textbf{Lo}$ng $\textbf{P}$reference $\textbf{O}$ptimization ($\textbf{SoLoPO}$), decoupling long-context preference optimization (PO) into two components: short-context PO and short-to-long reward alignment (SoLo-RA), supported by both theoretical and empirical evidence. Specifically, short-context PO leverages preference pairs sampled from short contexts to enhance the model's contextual knowledge utilization ability. Meanwhile, SoLo-RA explicitly encourages reward score consistency utilization for the responses when conditioned on both short and long contexts that contain identical task-relevant information. This facilitates transferring the model's ability to handle short contexts into long-context scenarios. SoLoPO is compatible with mainstream preference optimization algorithms, while substantially improving the efficiency of data construction and training processes. Experimental results show that SoLoPO enhances all these algorithms with respect to stronger length and domain generalization abilities across various long-context benchmarks, while achieving notable improvements in both computational and memory efficiency. 

**Abstract (ZH)**: 短到长偏好优化框架(SoLoPO) 

---
# Maximizing Asynchronicity in Event-based Neural Networks 

**Title (ZH)**: 基于事件的神经网络中最大化异步性 

**Authors**: Haiqing Hao, Nikola Zubić, Weihua He, Zhipeng Sui, Davide Scaramuzza, Wenhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11165)  

**Abstract**: Event cameras deliver visual data with high temporal resolution, low latency, and minimal redundancy, yet their asynchronous, sparse sequential nature challenges standard tensor-based machine learning (ML). While the recent asynchronous-to-synchronous (A2S) paradigm aims to bridge this gap by asynchronously encoding events into learned representations for ML pipelines, existing A2S approaches often sacrifice representation expressivity and generalizability compared to dense, synchronous methods. This paper introduces EVA (EVent Asynchronous representation learning), a novel A2S framework to generate highly expressive and generalizable event-by-event representations. Inspired by the analogy between events and language, EVA uniquely adapts advances from language modeling in linear attention and self-supervised learning for its construction. In demonstration, EVA outperforms prior A2S methods on recognition tasks (DVS128-Gesture and N-Cars), and represents the first A2S framework to successfully master demanding detection tasks, achieving a remarkable 47.7 mAP on the Gen1 dataset. These results underscore EVA's transformative potential for advancing real-time event-based vision applications. 

**Abstract (ZH)**: 事件相机通过提供高时间分辨率、低延迟和最小冗余的视觉数据，虽然其异步、稀疏的序列特性挑战了标准张量基机器学习方法，但仍存在瓶颈。尽管近期提出的异步到同步（A2S）范式旨在通过异步地将事件编码为学习表示以适应机器学习管道，但现有A2S方法通常在表达性和泛化能力上逊色于密集同步方法。本文介绍了一种新的A2S框架EVA（EVent Asynchronous representation learning），用于生成高度表达性和泛化能力的逐事件表示。受事件与语言之间类比的启发，EVA独特地采用了线性注意力和自我监督学习在构建中的进步。实验展示中，EVA在识别任务（DVS128-Gesture和N-Cars）上优于先前的A2S方法，并是第一个成功掌握 demanding 检测任务的A2S框架，实现了Gen1数据集上的47.7 mAP。这些结果强调了EVA在推动实时事件驱动视觉应用方面具有颠覆性的潜力。 

---
# Attention on the Sphere 

**Title (ZH)**: 球面上的注意力 

**Authors**: Boris Bonev, Max Rietmann, Andrea Paris, Alberto Carpentieri, Thorsten Kurth  

**Link**: [PDF](https://arxiv.org/pdf/2505.11157)  

**Abstract**: We introduce a generalized attention mechanism for spherical domains, enabling Transformer architectures to natively process data defined on the two-dimensional sphere - a critical need in fields such as atmospheric physics, cosmology, and robotics, where preserving spherical symmetries and topology is essential for physical accuracy. By integrating numerical quadrature weights into the attention mechanism, we obtain a geometrically faithful spherical attention that is approximately rotationally equivariant, providing strong inductive biases and leading to better performance than Cartesian approaches. To further enhance both scalability and model performance, we propose neighborhood attention on the sphere, which confines interactions to geodesic neighborhoods. This approach reduces computational complexity and introduces the additional inductive bias for locality, while retaining the symmetry properties of our method. We provide optimized CUDA kernels and memory-efficient implementations to ensure practical applicability. The method is validated on three diverse tasks: simulating shallow water equations on the rotating sphere, spherical image segmentation, and spherical depth estimation. Across all tasks, our spherical Transformers consistently outperform their planar counterparts, highlighting the advantage of geometric priors for learning on spherical domains. 

**Abstract (ZH)**: 一种适用于球面域的广义注意力机制：增强变换器架构处理球面数据的能力 

---
# X2C: A Dataset Featuring Nuanced Facial Expressions for Realistic Humanoid Imitation 

**Title (ZH)**: X2C: 一个展现细腻面部表情的数据集用于 realistic 人形模仿 

**Authors**: Peizhen Li, Longbing Cao, Xiao-Ming Wu, Runze Yang, Xiaohan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11146)  

**Abstract**: The ability to imitate realistic facial expressions is essential for humanoid robots engaged in affective human-robot communication. However, the lack of datasets containing diverse humanoid facial expressions with proper annotations hinders progress in realistic humanoid facial expression imitation. To address these challenges, we introduce X2C (Anything to Control), a dataset featuring nuanced facial expressions for realistic humanoid imitation. With X2C, we contribute: 1) a high-quality, high-diversity, large-scale dataset comprising 100,000 (image, control value) pairs. Each image depicts a humanoid robot displaying a diverse range of facial expressions, annotated with 30 control values representing the ground-truth expression configuration; 2) X2CNet, a novel human-to-humanoid facial expression imitation framework that learns the correspondence between nuanced humanoid expressions and their underlying control values from X2C. It enables facial expression imitation in the wild for different human performers, providing a baseline for the imitation task, showcasing the potential value of our dataset; 3) real-world demonstrations on a physical humanoid robot, highlighting its capability to advance realistic humanoid facial expression imitation. Code and Data: this https URL 

**Abstract (ZH)**: 模仿逼真面部表情的能力对于参与情感人机通信的人形机器人至关重要。然而，缺乏包含多样面部表情且标注恰当的数据集阻碍了逼真人形面部表情模仿的进步。为应对这些挑战，我们引入了X2C（Anything to Control），一个用于真实人形仿真的细腻面部表情数据集。通过X2C，我们贡献了：1) 一个高质量、高多样性和大规模的数据集，包含100,000个（图像，控制值）对。每个图像展示了一个展示多样化面部表情的人形机器人，并用30个控制值标注其真实表情配置；2) X2CNet，一种新颖的人类到人形面部表情模仿框架，它从X2C中学习细腻人形表情与其底层控制值之间的对应关系，使其能够在不同的表演者中进行真实环境下的面部表情模仿，并为模仿任务提供基线，展示了我们数据集的潜在价值；3) 在实际人形机器人上的真实世界演示，突显了其在推进真实人形面部表情模仿方面的能力。代码和数据：this https URL。 

---
# Human-Aligned Bench: Fine-Grained Assessment of Reasoning Ability in MLLMs vs. Humans 

**Title (ZH)**: 人类对齐基准：MLLMs与人类在推理能力方面的细粒度评估 

**Authors**: Yansheng Qiu, Li Xiao, Zhaopan Xu, Pengfei Zhou, Zheng Wang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11141)  

**Abstract**: The goal of achieving Artificial General Intelligence (AGI) is to imitate humans and surpass them. Models such as OpenAI's o1, o3, and DeepSeek's R1 have demonstrated that large language models (LLMs) with human-like reasoning capabilities exhibit exceptional performance and are being gradually integrated into multimodal large language models (MLLMs). However, whether these models possess capabilities comparable to humans in handling reasoning tasks remains unclear at present. In this paper, we propose Human-Aligned Bench, a benchmark for fine-grained alignment of multimodal reasoning with human performance. Specifically, we collected 9,794 multimodal questions that solely rely on contextual reasoning, including bilingual (Chinese and English) multimodal questions and pure text-based questions, encompassing four question types: visual reasoning, definition judgment, analogical reasoning, and logical judgment. More importantly, each question is accompanied by human success rates and options that humans are prone to choosing incorrectly. Extensive experiments on the Human-Aligned Bench reveal notable differences between the performance of current MLLMs in multimodal reasoning and human performance. The findings on our benchmark provide insights into the development of the next-generation models. 

**Abstract (ZH)**: 实现通用人工智能（AGI）的目标是模仿人类并超越人类。类似于OpenAI的o1、o3和DeepSeek的R1等模型展示了具有类人推理能力的大语言模型（LLMs）表现出色，并正逐渐集成到多模态大语言模型（MLLMs）中。然而，这些模型在处理推理任务方面是否具有与人类相当的能力仍不清楚。本文提出Human-Aligned Bench，一个用于细粒度多模态推理与人类表现对齐的基准。具体而言，我们收集了9,794个仅依赖于上下文推理的多模态问题，包括双语（中文和英文）多模态问题和纯文本问题，涵盖了四种问题类型：视觉推理、定义判断、类比推理和逻辑判断。更重要的是，每个问题都附有人类的成功率和人类易出错的选择选项。对Human-Aligned Bench的广泛实验揭示了当前MLLMs在多模态推理中的表现与人类表现之间的显著差异。我们基准上的发现为下一代模型的开发提供了见解。 

---
# Scaling Reasoning can Improve Factuality in Large Language Models 

**Title (ZH)**: 扩展推理可以提高大型语言模型的事实准确性 

**Authors**: Mike Zhang, Johannes Bjerva, Russa Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2505.11140)  

**Abstract**: Recent studies on large language model (LLM) reasoning capabilities have demonstrated promising improvements in model performance by leveraging a lengthy thinking process and additional computational resources during inference, primarily in tasks involving mathematical reasoning (Muennighoff et al., 2025). However, it remains uncertain if longer reasoning chains inherently enhance factual accuracy, particularly beyond mathematical contexts. In this work, we thoroughly examine LLM reasoning within complex open-domain question-answering (QA) scenarios. We initially distill reasoning traces from advanced, large-scale reasoning models (QwQ-32B and DeepSeek-R1-671B), then fine-tune a variety of models ranging from smaller, instruction-tuned variants to larger architectures based on Qwen2.5. To enrich reasoning traces, we introduce factual information from knowledge graphs in the form of paths into our reasoning traces. Our experimental setup includes four baseline approaches and six different instruction-tuned models evaluated across a benchmark of six datasets, encompassing over 22.6K questions. Overall, we carry out 168 experimental runs and analyze approximately 1.7 million reasoning traces. Our findings indicate that, within a single run, smaller reasoning models achieve noticeable improvements in factual accuracy compared to their original instruction-tuned counterparts. Moreover, our analysis demonstrates that adding test-time compute and token budgets factual accuracy consistently improves by 2-8%, further confirming the effectiveness of test-time scaling for enhancing performance and consequently improving reasoning accuracy in open-domain QA tasks. We release all the experimental artifacts for further research. 

**Abstract (ZH)**: 近期关于大型语言模型（LLM）推理能力的研究表明，在推理过程中利用较长的思考过程和额外的计算资源可以显著提高模型性能，特别是在涉及数学推理的任务中（Muennighoff等，2025）。然而，长的推理链是否必然提高事实准确性，尤其是在超出数学背景的情况下，仍然不确定。本研究全面考察了LLM推理在复杂开放域问答（QA）场景中的表现。我们首先从先进的大规模推理模型（QwQ-32B和DeepSeek-R1-671B）中提取推理轨迹，然后根据不同规模的Qwen2.5架构基于指令调整的模型进行微调。为了丰富推理轨迹，我们将知识图谱中的事实信息以路径的形式引入。我们的实验设置包括四个基线方法和六个不同的指令调整模型，这些模型在包含超过22600个问题的六个数据集基准上进行了评估。总体而言，我们进行了168次实验运行，并分析了大约170万个推理轨迹。研究结果表明，在单次运行中，较小的推理模型在事实准确性方面相对于其原始指令调整版本实现了显著提升。此外，我们的分析表明，增加推理时间的计算资源和标记预算可以一致提高2-8%的事实准确性，进一步证明了推理时间扩展对于提升性能并进而提高开放域问答任务推理准确性有效性的有效性。我们发布了所有实验结果以供进一步研究。 

---
# One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework 

**Title (ZH)**: 一张图片胜过千言万语：一种可保留可用性的文本-图像协作擦除框架 

**Authors**: Feiran Li, Qianqian Xu, Shilong Bao, Zhiyong Yang, Xiaochun Cao, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11131)  

**Abstract**: Concept erasing has recently emerged as an effective paradigm to prevent text-to-image diffusion models from generating visually undesirable or even harmful content. However, current removal methods heavily rely on manually crafted text prompts, making it challenging to achieve a high erasure (efficacy) while minimizing the impact on other benign concepts (usability). In this paper, we attribute the limitations to the inherent gap between the text and image modalities, which makes it hard to transfer the intricately entangled concept knowledge from text prompts to the image generation process. To address this, we propose a novel solution by directly integrating visual supervision into the erasure process, introducing the first text-image Collaborative Concept Erasing (Co-Erasing) framework. Specifically, Co-Erasing describes the concept jointly by text prompts and the corresponding undesirable images induced by the prompts, and then reduces the generating probability of the target concept through negative guidance. This approach effectively bypasses the knowledge gap between text and image, significantly enhancing erasure efficacy. Additionally, we design a text-guided image concept refinement strategy that directs the model to focus on visual features most relevant to the specified text concept, minimizing disruption to other benign concepts. Finally, comprehensive experiments suggest that Co-Erasing outperforms state-of-the-art erasure approaches significantly with a better trade-off between efficacy and usability. Codes are available at this https URL. 

**Abstract (ZH)**: 视觉监督引导的文本-图像协作概念消除框架 

---
# PhiNet v2: A Mask-Free Brain-Inspired Vision Foundation Model from Video 

**Title (ZH)**: PhiNet v2: 一种来自视频的无掩码脑启发视觉基础模型 

**Authors**: Makoto Yamada, Kian Ming A. Chai, Ayoub Rhim, Satoki Ishikawa, Mohammad Sabokrou, Yao-Hung Hubert Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11129)  

**Abstract**: Recent advances in self-supervised learning (SSL) have revolutionized computer vision through innovative architectures and learning objectives, yet they have not fully leveraged insights from biological visual processing systems. Recently, a brain-inspired SSL model named PhiNet was proposed; it is based on a ResNet backbone and operates on static image inputs with strong augmentation. In this paper, we introduce PhiNet v2, a novel Transformer-based architecture that processes temporal visual input (that is, sequences of images) without relying on strong augmentation. Our model leverages variational inference to learn robust visual representations from continuous input streams, similar to human visual processing. Through extensive experimentation, we demonstrate that PhiNet v2 achieves competitive performance compared to state-of-the-art vision foundation models, while maintaining the ability to learn from sequential input without strong data augmentation. This work represents a significant step toward more biologically plausible computer vision systems that process visual information in a manner more closely aligned with human cognitive processes. 

**Abstract (ZH)**: 最近在自监督学习（SSL）领域的进展通过创新的架构和学习目标革新了计算机视觉，但尚未充分利用生物视觉处理系统的见解。最近，一种受脑启发的SSL模型PhiNet被提出；它基于ResNet骨干网络，并对静态图像输入进行强增强操作。在本文中，我们引入了PhiNet v2，这是一种新型的基于Transformer的架构，能够处理时序视觉输入（即图像序列）而不依赖于强增强。我们的模型利用变分推断从连续输入流中学习鲁棒的视觉表示，类似于人类视觉处理。通过大量的实验，我们证明了PhiNet v2在与最先进的视觉基础模型相当的性能上，同时保持了从序列输入中学习的能力，而不需要强烈的数据增强。这项工作代表了朝着更符合生物学原理的计算机视觉系统迈进的重要一步，这些系统能够以更接近人类认知过程的方式处理视觉信息。 

---
# Conditioning Matters: Training Diffusion Policies is Faster Than You Think 

**Title (ZH)**: 条件决定一切：预训练扩散策略比你想象的更快 

**Authors**: Zibin Dong, Yicheng Liu, Yinchuan Li, Hang Zhao, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11123)  

**Abstract**: Diffusion policies have emerged as a mainstream paradigm for building vision-language-action (VLA) models. Although they demonstrate strong robot control capabilities, their training efficiency remains suboptimal. In this work, we identify a fundamental challenge in conditional diffusion policy training: when generative conditions are hard to distinguish, the training objective degenerates into modeling the marginal action distribution, a phenomenon we term loss collapse. To overcome this, we propose Cocos, a simple yet general solution that modifies the source distribution in the conditional flow matching to be condition-dependent. By anchoring the source distribution around semantics extracted from condition inputs, Cocos encourages stronger condition integration and prevents the loss collapse. We provide theoretical justification and extensive empirical results across simulation and real-world benchmarks. Our method achieves faster convergence and higher success rates than existing approaches, matching the performance of large-scale pre-trained VLAs using significantly fewer gradient steps and parameters. Cocos is lightweight, easy to implement, and compatible with diverse policy architectures, offering a general-purpose improvement to diffusion policy training. 

**Abstract (ZH)**: Diffusion策略已成为构建视觉-语言-动作（VLA）模型的主要范式。尽管它们展现了强大的机器人控制能力，但其训练效率仍不尽如人意。在这项工作中，我们识别出条件扩散策略训练中的一个基本挑战：当生成条件难以区分时，训练目标退化为建模边际动作分布，我们称这一现象为损失坍塌。为克服这一挑战，我们提出了Cocos，这是一种简单而通用的解决方案，通过在条件流匹配中修改源分布，使其依赖于条件。通过锚定源分布以与条件输入中提取的语义相关，Cocos 促进更强的条件整合并防止损失坍塌。我们提供了理论依据并在模拟和现实世界基准测试中进行了广泛的经验验证。该方法实现了更快的收敛速度和更高的成功率，使用显著较少的梯度步骤和参数匹配大规模预训练VLA的表现。Cocos 轻量级、易于实现，并与多种策略架构兼容，为扩散策略训练提供了一种通用改进方案。 

---
# FairSHAP: Preprocessing for Fairness Through Attribution-Based Data Augmentation 

**Title (ZH)**: 基于归因数据增强的公平性预处理：FairSHAP 

**Authors**: Lin Zhu, Yijun Bian, Lei You  

**Link**: [PDF](https://arxiv.org/pdf/2505.11111)  

**Abstract**: Ensuring fairness in machine learning models is critical, particularly in high-stakes domains where biased decisions can lead to serious societal consequences. Existing preprocessing approaches generally lack transparent mechanisms for identifying which features or instances are responsible for unfairness. This obscures the rationale behind data modifications. We introduce FairSHAP, a novel pre-processing framework that leverages Shapley value attribution to improve both individual and group fairness. FairSHAP identifies fairness-critical instances in the training data using an interpretable measure of feature importance, and systematically modifies them through instance-level matching across sensitive groups. This process reduces discriminative risk - an individual fairness metric - while preserving data integrity and model accuracy. We demonstrate that FairSHAP significantly improves demographic parity and equality of opportunity across diverse tabular datasets, achieving fairness gains with minimal data perturbation and, in some cases, improved predictive performance. As a model-agnostic and transparent method, FairSHAP integrates seamlessly into existing machine learning pipelines and provides actionable insights into the sources of this http URL code is on this https URL. 

**Abstract (ZH)**: 确保机器学习模型的公平性至关重要，特别是在高风险领域，偏见决策可能导致严重社会后果。现有的预处理方法通常缺乏透明的机制来识别导致不公平的特征或实例，这掩盖了数据修改的原因。我们提出了FairSHAP，一种新颖的预处理框架，利用Shapley值归因来改进个体公平性和群体公平性。FairSHAP使用可解释的特征重要性度量来识别训练数据中的公平关键实例，并通过敏感群体的实例级匹配系统地对其进行修改。这一过程减少了歧视性风险（一种个体公平性指标），同时保持了数据完整性和模型准确性。我们证明，FairSHAP在多种表格数据集中显著提高了人口统计正义和平等的机会，并通过最少的数据扰动实现了公平性收益，在某些情况下还提高了预测性能。作为一种模型无关且透明的方法，FairSHAP可无缝集成到现有的机器学习管道中，并提供有关这一http URL的可操作见解。代码可在此https://github.com/fairshap-team/FairSHAP 获取。 

---
# MAVOS-DD: Multilingual Audio-Video Open-Set Deepfake Detection Benchmark 

**Title (ZH)**: MAVOS-DD: 多语言音频-视频开放集深度假信息检测基准 

**Authors**: Florinel-Alin Croitoru, Vlad Hondru, Marius Popescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.11109)  

**Abstract**: We present the first large-scale open-set benchmark for multilingual audio-video deepfake detection. Our dataset comprises over 250 hours of real and fake videos across eight languages, with 60% of data being generated. For each language, the fake videos are generated with seven distinct deepfake generation models, selected based on the quality of the generated content. We organize the training, validation and test splits such that only a subset of the chosen generative models and languages are available during training, thus creating several challenging open-set evaluation setups. We perform experiments with various pre-trained and fine-tuned deepfake detectors proposed in recent literature. Our results show that state-of-the-art detectors are not currently able to maintain their performance levels when tested in our open-set scenarios. We publicly release our data and code at: this https URL. 

**Abstract (ZH)**: 我们首次提出了一个大规模开放集多语言音频-视频深度假信息检测基准。 

---
# PARSEC: Preference Adaptation for Robotic Object Rearrangement from Scene Context 

**Title (ZH)**: PARSEC: 基于场景上下文的物体重新排列偏好适应 

**Authors**: Kartik Ramachandruni, Sonia Chernova  

**Link**: [PDF](https://arxiv.org/pdf/2505.11108)  

**Abstract**: Object rearrangement is a key task for household robots requiring personalization without explicit instructions, meaningful object placement in environments occupied with objects, and generalization to unseen objects and new environments. To facilitate research addressing these challenges, we introduce PARSEC, an object rearrangement benchmark for learning user organizational preferences from observed scene context to place objects in a partially arranged environment. PARSEC is built upon a novel dataset of 110K rearrangement examples crowdsourced from 72 users, featuring 93 object categories and 15 environments. We also propose ContextSortLM, an LLM-based rearrangement model that places objects in partially arranged environments by adapting to user preferences from prior and current scene context while accounting for multiple valid placements. We evaluate ContextSortLM and existing personalized rearrangement approaches on the PARSEC benchmark and complement these findings with a crowdsourced evaluation of 108 online raters ranking model predictions based on alignment with user preferences. Our results indicate that personalized rearrangement models leveraging multiple scene context sources perform better than models relying on a single context source. Moreover, ContextSortLM outperforms other models in placing objects to replicate the target user's arrangement and ranks among the top two in all three environment categories, as rated by online evaluators. Importantly, our evaluation highlights challenges associated with modeling environment semantics across different environment categories and provides recommendations for future work. 

**Abstract (ZH)**: 基于场景上下文学习用户组织偏好的物体重排基准PARSEC 

---
# Inferring the Most Similar Variable-length Subsequences between Multidimensional Time Series 

**Title (ZH)**: 多维时间序列中最相似变长子序列的推断 

**Authors**: Thanadej Rattanakornphan, Piyanon Charoenpoonpanich, Chainarong Amornbunchornvej  

**Link**: [PDF](https://arxiv.org/pdf/2505.11106)  

**Abstract**: Finding the most similar subsequences between two multidimensional time series has many applications: e.g. capturing dependency in stock market or discovering coordinated movement of baboons. Considering one pattern occurring in one time series, we might be wondering whether the same pattern occurs in another time series with some distortion that might have a different length. Nevertheless, to the best of our knowledge, there is no efficient framework that deals with this problem yet. In this work, we propose an algorithm that provides the exact solution of finding the most similar multidimensional subsequences between time series where there is a difference in length both between time series and between subsequences. The algorithm is built based on theoretical guarantee of correctness and efficiency. The result in simulation datasets illustrated that our approach not just only provided correct solution, but it also utilized running time only quarter of time compared against the baseline approaches. In real-world datasets, it extracted the most similar subsequences even faster (up to 20 times faster against baseline methods) and provided insights regarding the situation in stock market and following relations of multidimensional time series of baboon movement. Our approach can be used for any time series. The code and datasets of this work are provided for the public use. 

**Abstract (ZH)**: 在多维时间序列之间寻找最相似子序列的研究：从股票市场依赖性到狒狒协调运动的发现 

---
# Bidirectional Distillation: A Mixed-Play Framework for Multi-Agent Generalizable Behaviors 

**Title (ZH)**: 双向 distillation: 一种多智能体通用行为的混合博弈框架 

**Authors**: Lang Feng, Jiahao Lin, Dong Xing, Li Zhang, De Ma, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11100)  

**Abstract**: Population-population generalization is a challenging problem in multi-agent reinforcement learning (MARL), particularly when agents encounter unseen co-players. However, existing self-play-based methods are constrained by the limitation of inside-space generalization. In this study, we propose Bidirectional Distillation (BiDist), a novel mixed-play framework, to overcome this limitation in MARL. BiDist leverages knowledge distillation in two alternating directions: forward distillation, which emulates the historical policies' space and creates an implicit self-play, and reverse distillation, which systematically drives agents towards novel distributions outside the known policy space in a non-self-play manner. In addition, BiDist operates as a concise and efficient solution without the need for the complex and costly storage of past policies. We provide both theoretical analysis and empirical evidence to support BiDist's effectiveness. Our results highlight its remarkable generalization ability across a variety of cooperative, competitive, and social dilemma tasks, and reveal that BiDist significantly diversifies the policy distribution space. We also present comprehensive ablation studies to reinforce BiDist's effectiveness and key success factors. Source codes are available in the supplementary material. 

**Abstract (ZH)**: 人口-人口泛化是多智能体强化学习（MARL）中的一个挑战性问题，特别是在智能体遇到未见过的合作者时。然而，现有的基于自我对弈的方法受到内部空间泛化的限制。本研究提出了一种新颖的双向灌输（BiDist）框架，以克服MARL中的这一限制。BiDist 利用双向灌输：前向灌输模拟历史策略空间并创建隐式自我对弈，反向灌输系统地引导智能体向已知策略空间之外的新分布发展，不采用自我对弈方式。此外，BiDist 作为一个简洁高效的方法，无需复杂且成本高昂的过去策略存储。我们提供了理论分析和实验证据来支持BiDist的有效性。我们的结果强调了BiDist在其合作、竞争和社会困境任务上的显著泛化能力，并揭示了BiDist显著多样化了策略分布空间。我们还进行了全面的消融研究以增强BiDist的有效性和关键成功因素。附带代码详见补充材料。 

---
# A Fast Kernel-based Conditional Independence test with Application to Causal Discovery 

**Title (ZH)**: 一种基于核的方法的快速条件独立性检验及其在因果发现中的应用 

**Authors**: Oliver Schacht, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11085)  

**Abstract**: Kernel-based conditional independence (KCI) testing is a powerful nonparametric method commonly employed in causal discovery tasks. Despite its flexibility and statistical reliability, cubic computational complexity limits its application to large datasets. To address this computational bottleneck, we propose \textit{FastKCI}, a scalable and parallelizable kernel-based conditional independence test that utilizes a mixture-of-experts approach inspired by embarrassingly parallel inference techniques for Gaussian processes. By partitioning the dataset based on a Gaussian mixture model over the conditioning variables, FastKCI conducts local KCI tests in parallel, aggregating the results using an importance-weighted sampling scheme. Experiments on synthetic datasets and benchmarks on real-world production data validate that FastKCI maintains the statistical power of the original KCI test while achieving substantial computational speedups. FastKCI thus represents a practical and efficient solution for conditional independence testing in causal inference on large-scale data. 

**Abstract (ZH)**: 基于核的条件独立性（KCI）测试是一种在因果发现任务中广泛应用的强非参数方法。尽管其具有灵活性和统计可靠性，但三次方的计算复杂度限制了其在大规模数据集上的应用。为解决这一计算瓶颈，我们提出了一种名为FastKCI的可扩展且并行化的基于核的条件独立性测试方法，该方法受到高斯过程的尴尬并行推理技术启发，采用混合专家方法。通过基于条件变量的高斯混合模型对数据集进行分区，FastKCI在局部并行执行KCI测试，并通过重要性加权采样方案汇总结果。实验结果在合成数据集和实际生产数据集上的基准测试验证了FastKCI在保持原始KCI测试统计功效的同时，实现了显著的计算加速。FastKCI因此为大规模数据上的条件独立性测试提供了一个实用且高效的解决方案。 

---
# Fault Diagnosis across Heterogeneous Domains via Self-Adaptive Temporal-Spatial Attention and Sample Generation 

**Title (ZH)**: 跨异构域故障诊断：基于自适应时空注意力和样本生成的方法 

**Authors**: Guangqiang Li, M. Amine Atoui, Xiangshun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11083)  

**Abstract**: Deep learning methods have shown promising performance in fault diagnosis for multimode process. Most existing studies assume that the collected health state categories from different operating modes are identical. However, in real industrial scenarios, these categories typically exhibit only partial overlap. The incompleteness of the available data and the large distributional differences between the operating modes pose a significant challenge to existing fault diagnosis methods. To address this problem, a novel fault diagnosis model named self-adaptive temporal-spatial attention network (TSA-SAN) is proposed. First, inter-mode mappings are constructed using healthy category data to generate multimode samples. To enrich the diversity of the fault data, interpolation is performed between healthy and fault samples. Subsequently, the fault diagnosis model is trained using real and generated data. The self-adaptive instance normalization is established to suppress irrelevant information while retaining essential statistical features for diagnosis. In addition, a temporal-spatial attention mechanism is constructed to focus on the key features, thus enhancing the generalization ability of the model. The extensive experiments demonstrate that the proposed model significantly outperforms the state-of-the-art methods. The code will be available on Github at this https URL. 

**Abstract (ZH)**: 基于自适应时序空域注意力网络的多模式过程故障诊断方法 

---
# BLEUBERI: BLEU is a surprisingly effective reward for instruction following 

**Title (ZH)**: BLEUBERI: BLEU是一个令人惊讶的有效奖励函数，用于指令跟随 

**Authors**: Yapei Chang, Yekyung Kim, Michael Krumdick, Amir Zadeh, Chuan Li, Chris Tanner, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11080)  

**Abstract**: Reward models are central to aligning LLMs with human preferences, but they are costly to train, requiring large-scale human-labeled preference data and powerful pretrained LLM backbones. Meanwhile, the increasing availability of high-quality synthetic instruction-following datasets raises the question: can simpler, reference-based metrics serve as viable alternatives to reward models during RL-based alignment? In this paper, we show first that BLEU, a basic string-matching metric, surprisingly matches strong reward models in agreement with human preferences on general instruction-following datasets. Based on this insight, we develop BLEUBERI, a method that first identifies challenging instructions and then applies Group Relative Policy Optimization (GRPO) using BLEU directly as the reward function. We demonstrate that BLEUBERI-trained models are competitive with models trained via reward model-guided RL across four challenging instruction-following benchmarks and three different base language models. A human evaluation further supports that the quality of BLEUBERI model outputs is on par with those from reward model-aligned models. Moreover, BLEUBERI models generate outputs that are more factually grounded than competing methods. Overall, we show that given access to high-quality reference outputs (easily obtained via existing instruction-following datasets or synthetic data generation), string matching-based metrics are cheap yet effective proxies for reward models during alignment. We release our code and data at this https URL. 

**Abstract (ZH)**: 奖励模型在对齐大语言模型与人类偏好方面起着核心作用，但训练成本高昂，需要大规模的人类标注偏好数据和强大的预训练大语言模型作为支撑。随着高质量合成指令跟随数据集的日益可用，一个疑问浮现：基于奖励模型的指令跟随对齐过程中，简化的参照基度量指标能否作为可行的替代方案？在本文中，我们首先证明，在通用指令跟随数据集上，基本的字符串匹配指标BLEU与强奖励模型在与人类偏好的一致性方面表现惊人地相似。基于这一洞察，我们开发了BLEUBERI方法，该方法首先识别具有挑战性的指令，然后使用BLEU直接作为奖励函数进行组相对策略优化（GRPO）。实验结果表明，BLEUBERI训练的模型在四个具有挑战性的指令跟随基准和三种不同的基础语言模型上与基于奖励模型的RL训练模型竞争具有竞争力。进一步的人类评估支持，BLEUBERI模型的输出质量与奖励模型对齐模型的输出质量相当，并且BLEUBERI模型生成的输出更加符合事实。总体而言，我们展示了，在拥有高质量参照输出（通过现有的指令跟随数据集或合成数据生成轻松获得）的情况下，基于字符串匹配的指标在对齐过程中是廉价且有效的奖励模型替代方案。我们在https://this.is/Qwen处发布了我们的代码和数据。 

---
# Towards Self-Improvement of Diffusion Models via Group Preference Optimization 

**Title (ZH)**: 通过群体偏好优化实现扩散模型的自我改进 

**Authors**: Renjie Chen, Wenfeng Lin, Yichen Zhang, Jiangchuan Wei, Boyuan Liu, Chao Feng, Jiao Ran, Mingyu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11070)  

**Abstract**: Aligning text-to-image (T2I) diffusion models with Direct Preference Optimization (DPO) has shown notable improvements in generation quality. However, applying DPO to T2I faces two challenges: the sensitivity of DPO to preference pairs and the labor-intensive process of collecting and annotating high-quality data. In this work, we demonstrate that preference pairs with marginal differences can degrade DPO performance. Since DPO relies exclusively on relative ranking while disregarding the absolute difference of pairs, it may misclassify losing samples as wins, or vice versa. We empirically show that extending the DPO from pairwise to groupwise and incorporating reward standardization for reweighting leads to performance gains without explicit data selection. Furthermore, we propose Group Preference Optimization (GPO), an effective self-improvement method that enhances performance by leveraging the model's own capabilities without requiring external data. Extensive experiments demonstrate that GPO is effective across various diffusion models and tasks. Specifically, combining with widely used computer vision models, such as YOLO and OCR, the GPO improves the accurate counting and text rendering capabilities of the Stable Diffusion 3.5 Medium by 20 percentage points. Notably, as a plug-and-play method, no extra overhead is introduced during inference. 

**Abstract (ZH)**: 使用Group Preference Optimization (GPO)提升文本到图像生成模型性能：无需数据选择的自我优化方法 

---
# Assessing the Performance of Analog Training for Transfer Learning 

**Title (ZH)**: 评估模拟训练在迁移学习中的性能 

**Authors**: Omobayode Fagbohungbe, Corey Lammie, Malte J. Rasch, Takashi Ando, Tayfun Gokmen, Vijay Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11067)  

**Abstract**: Analog in-memory computing is a next-generation computing paradigm that promises fast, parallel, and energy-efficient deep learning training and transfer learning (TL). However, achieving this promise has remained elusive due to a lack of suitable training algorithms. Analog memory devices exhibit asymmetric and non-linear switching behavior in addition to device-to-device variation, meaning that most, if not all, of the current off-the-shelf training algorithms cannot achieve good training outcomes. Also, recently introduced algorithms have enjoyed limited attention, as they require bi-directionally switching devices of unrealistically high symmetry and precision and are highly sensitive. A new algorithm chopped TTv2 (c-TTv2), has been introduced, which leverages the chopped technique to address many of the challenges mentioned above. In this paper, we assess the performance of the c-TTv2 algorithm for analog TL using a Swin-ViT model on a subset of the CIFAR100 dataset. We also investigate the robustness of our algorithm to changes in some device specifications, including weight transfer noise, symmetry point skew, and symmetry point variability 

**Abstract (ZH)**: 模拟内存计算是一种下一代计算范式，有望实现快速、并行和节能的深度学习训练和迁移学习（TL）。然而，由于缺乏合适的训练算法，这一承诺尚未实现。模拟内存设备表现出非对称和非线性的开关行为，同时还存在器件间的差异，这意味着当前大多数甚至所有现成的训练算法都无法达到良好的训练效果。另外，最近引入的一些算法也受到了限制，因为它们需要具有不切实际的高对称性和精度的双向切换设备，并且非常敏感。一种新的算法——截断TTv2（c-TTv2）——已被提出，它利用截断技术来解决上述许多挑战。在本文中，我们使用Swin-ViT模型在CIFAR100数据集的部分子集上评估c-TTv2算法在模拟迁移学习中的性能。我们还研究了我们的算法对某些器件规格变化的鲁棒性，包括权重转移噪声、对称点偏斜和对称点变异性。 

---
# Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking 

**Title (ZH)**: 时光旅行就是作弊：使用DeepFund实现实时基金投资基准比对 

**Authors**: Changlun Li, Yao Shi, Chen Wang, Qiqi Duan, Runke Ruan, Weijie Huang, Haonan Long, Lijun Huang, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11065)  

**Abstract**: Large Language Models (LLMs) have demonstrated notable capabilities across financial tasks, including financial report summarization, earnings call transcript analysis, and asset classification. However, their real-world effectiveness in managing complex fund investment remains inadequately assessed. A fundamental limitation of existing benchmarks for evaluating LLM-driven trading strategies is their reliance on historical back-testing, inadvertently enabling LLMs to "time travel"-leveraging future information embedded in their training corpora, thus resulting in possible information leakage and overly optimistic performance estimates. To address this issue, we introduce DeepFund, a live fund benchmark tool designed to rigorously evaluate LLM in real-time market conditions. Utilizing a multi-agent architecture, DeepFund connects directly with real-time stock market data-specifically data published after each model pretraining cutoff-to ensure fair and leakage-free evaluations. Empirical tests on nine flagship LLMs from leading global institutions across multiple investment dimensions-including ticker-level analysis, investment decision-making, portfolio management, and risk control-reveal significant practical challenges. Notably, even cutting-edge models such as DeepSeek-V3 and Claude-3.7-Sonnet incur net trading losses within DeepFund real-time evaluation environment, underscoring the present limitations of LLMs for active fund management. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型在复杂基金投资管理中的现实有效性尚未充分评估。现有的用于评估以大型语言模型驱动交易策略的标准性能基准依赖于历史回测，这不经意地使大型语言模型能够“时间旅行”——利用其训练语料中嵌入的未来信息，从而导致可能的信息泄露和过于乐观的性能估计。为解决这一问题，我们引入了DeepFund，这是一个实时基金基准工具，旨在在实时市场条件下严格评估大型语言模型。DeepFund采用多代理架构，直接连接实时股票市场数据——特别是每次模型预训练截止后的数据——以确保公平和无泄露的评估。在多个投资维度上的实证测试（包括代码级别分析、投资决策、组合管理和风险控制）表明，即使是DeepSeek-V3和Claude-3.7-Sonnet这样的尖端模型，在DeepFund实时评估环境中也出现了净交易亏损，突显了大型语言模型目前在积极基金管理中的局限性。代码可在以下链接获取：this https URL。 

---
# CUBIC: Concept Embeddings for Unsupervised Bias Identification using VLMs 

**Title (ZH)**: CUBIC: 无监督偏见识别的概念嵌入使用大语言模型 

**Authors**: David Méndez, Gianpaolo Bontempo, Elisa Ficarra, Roberto Confalonieri, Natalia Díaz-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2505.11060)  

**Abstract**: Deep vision models often rely on biases learned from spurious correlations in datasets. To identify these biases, methods that interpret high-level, human-understandable concepts are more effective than those relying primarily on low-level features like heatmaps. A major challenge for these concept-based methods is the lack of image annotations indicating potentially bias-inducing concepts, since creating such annotations requires detailed labeling for each dataset and concept, which is highly labor-intensive. We present CUBIC (Concept embeddings for Unsupervised Bias IdentifiCation), a novel method that automatically discovers interpretable concepts that may bias classifier behavior. Unlike existing approaches, CUBIC does not rely on predefined bias candidates or examples of model failures tied to specific biases, as such information is not always available. Instead, it leverages image-text latent space and linear classifier probes to examine how the latent representation of a superclass label$\unicode{x2014}$shared by all instances in the dataset$\unicode{x2014}$is influenced by the presence of a given concept. By measuring these shifts against the normal vector to the classifier's decision boundary, CUBIC identifies concepts that significantly influence model predictions. Our experiments demonstrate that CUBIC effectively uncovers previously unknown biases using Vision-Language Models (VLMs) without requiring the samples in the dataset where the classifier underperforms or prior knowledge of potential biases. 

**Abstract (ZH)**: 基于概念的无监督偏见识别：CUBIC（概念嵌入用于无监督偏见识别） 

---
# Halting Recurrent GNNs and the Graded $μ$-Calculus 

**Title (ZH)**: 停止递归GNNs与分级μ-演算 

**Authors**: Jeroen Bollen, Jan Van den Bussche, Stijn Vansummeren, Jonni Virtema  

**Link**: [PDF](https://arxiv.org/pdf/2505.11050)  

**Abstract**: Graph Neural Networks (GNNs) are a class of machine-learning models that operate on graph-structured data. Their expressive power is intimately related to logics that are invariant under graded bisimilarity. Current proposals for recurrent GNNs either assume that the graph size is given to the model, or suffer from a lack of termination guarantees. In this paper, we propose a halting mechanism for recurrent GNNs. We prove that our halting model can express all node classifiers definable in graded modal mu-calculus, even for the standard GNN variant that is oblivious to the graph size. A recent breakthrough in the study of the expressivity of graded modal mu-calculus in the finite suggests that conversely, restricted to node classifiers definable in monadic second-order logic, recurrent GNNs can express only node classifiers definable in graded modal mu-calculus. To prove our main result, we develop a new approximate semantics for graded mu-calculus, which we believe to be of independent interest. We leverage this new semantics into a new model-checking algorithm, called the counting algorithm, which is oblivious to the graph size. In a final step we show that the counting algorithm can be implemented on a halting recurrent GNN. 

**Abstract (ZH)**: 循环图神经网络中止机制的研究：基于分级模μ演算的节点分类表达能力 

---
# CleanPatrick: A Benchmark for Image Data Cleaning 

**Title (ZH)**: CleanPatrick：图像数据清洗基准 

**Authors**: Fabian Gröger, Simone Lionetti, Philippe Gottfrois, Alvaro Gonzalez-Jimenez, Ludovic Amruthalingam, Elisabeth Victoria Goessinger, Hanna Lindemann, Marie Bargiela, Marie Hofbauer, Omar Badri, Philipp Tschandl, Arash Koochek, Matthew Groh, Alexander A. Navarini, Marc Pouly  

**Link**: [PDF](https://arxiv.org/pdf/2505.11034)  

**Abstract**: Robust machine learning depends on clean data, yet current image data cleaning benchmarks rely on synthetic noise or narrow human studies, limiting comparison and real-world relevance. We introduce CleanPatrick, the first large-scale benchmark for data cleaning in the image domain, built upon the publicly available Fitzpatrick17k dermatology dataset. We collect 496,377 binary annotations from 933 medical crowd workers, identify off-topic samples (4%), near-duplicates (21%), and label errors (22%), and employ an aggregation model inspired by item-response theory followed by expert review to derive high-quality ground truth. CleanPatrick formalizes issue detection as a ranking task and adopts typical ranking metrics mirroring real audit workflows. Benchmarking classical anomaly detectors, perceptual hashing, SSIM, Confident Learning, NoiseRank, and SelfClean, we find that, on CleanPatrick, self-supervised representations excel at near-duplicate detection, classical methods achieve competitive off-topic detection under constrained review budgets, and label-error detection remains an open challenge for fine-grained medical classification. By releasing both the dataset and the evaluation framework, CleanPatrick enables a systematic comparison of image-cleaning strategies and paves the way for more reliable data-centric artificial intelligence. 

**Abstract (ZH)**: Robust机器学习依赖于干净的数据，然而当前的图像数据清洁基准主要依赖于合成噪声或狭窄的人类研究，限制了比较和现实世界的相关性。我们引入了CleanPatrick，这是首个针对图像领域数据清洁的大规模基准，基于公开可用的Fitzpatrick17k皮肤病学数据集构建。我们收集了933名医学众包工作者的496,377个二元注释，识别出离题样本（4%）、近重复样本（21%）和标签错误（22%），并采用受项目反应理论启发的聚合模型结合专家审核，获得高质量的ground truth。CleanPatrick将问题检测形式化为一个排名任务，并采用常见排名指标来模拟实际审查工作流程。通过基准测试经典异常检测方法、感知哈希、SSIM、自信学习、NoiseRank和SelfClean，我们发现在CleanPatrick上，自我监督表示在近重复检测方面表现出色，经典方法在有限的审查预算下实现了竞争性的离题检测，而标签错误检测仍然是细粒度医疗分类的开放挑战。通过发布该数据集和评估框架，CleanPatrick使得图像清洁策略的系统比较成为可能，并为更可靠的数据为中心的人工智能铺平道路。 

---
# DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy 

**Title (ZH)**: DexGarmentLab: 拟人化服装操作环境与可泛化的策略 

**Authors**: Yuran Wang, Ruihai Wu, Yue Chen, Jiarui Wang, Jiaqi Liang, Ziyu Zhu, Haoran Geng, Jitendra Malik, Pieter Abbeel, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.11032)  

**Abstract**: Garment manipulation is a critical challenge due to the diversity in garment categories, geometries, and deformations. Despite this, humans can effortlessly handle garments, thanks to the dexterity of our hands. However, existing research in the field has struggled to replicate this level of dexterity, primarily hindered by the lack of realistic simulations of dexterous garment manipulation. Therefore, we propose DexGarmentLab, the first environment specifically designed for dexterous (especially bimanual) garment manipulation, which features large-scale high-quality 3D assets for 15 task scenarios, and refines simulation techniques tailored for garment modeling to reduce the sim-to-real gap. Previous data collection typically relies on teleoperation or training expert reinforcement learning (RL) policies, which are labor-intensive and inefficient. In this paper, we leverage garment structural correspondence to automatically generate a dataset with diverse trajectories using only a single expert demonstration, significantly reducing manual intervention. However, even extensive demonstrations cannot cover the infinite states of garments, which necessitates the exploration of new algorithms. To improve generalization across diverse garment shapes and deformations, we propose a Hierarchical gArment-manipuLation pOlicy (HALO). It first identifies transferable affordance points to accurately locate the manipulation area, then generates generalizable trajectories to complete the task. Through extensive experiments and detailed analysis of our method and baseline, we demonstrate that HALO consistently outperforms existing methods, successfully generalizing to previously unseen instances even with significant variations in shape and deformation where others fail. Our project page is available at: this https URL. 

**Abstract (ZH)**: 服装操作是由于服装类别、几何形状和变形的多样性的关键挑战。然而，人类能够凭借灵巧的手部动作轻松应对这些挑战。尽管如此，现有研究在复制这种灵巧性方面仍面临困难，主要是由于缺乏现实的灵巧服装操作模拟。因此，我们提出了DexGarmentLab，这是首个专门设计用于灵巧（尤其是双手灵巧）服装操作的环境，它包含了15种任务场景的大规模高质量3D资产，并通过定制化的模拟技术提高服装建模精度，缩小模拟与现实之间的差距。以往的数据收集通常依赖于遥操作或训练专家强化学习（RL）策略，这既耗时又低效。在本文中，我们利用服装结构对应关系，仅通过单次专家示范即可自动生成多样轨迹的数据集，大大减少了手动干预。然而，即使进行广泛的示范也无法覆盖服装的所有状态，这需要探索新的算法。为了提高在不同服装形状和变形中的泛化能力，我们提出了一种层次化服装操作策略（HALO）。首先，它识别可转移的功能点以准确定位操作区域，然后生成可泛化的轨迹以完成任务。通过广泛的实验和对我们的方法和基线的详细分析，我们证明了HALO在各种形状和变形变化情况下始终优于现有方法，能够成功泛化到未见过的实例。项目主页：this https URL。 

---
# The heteronomy of algorithms: Traditional knowledge and computational knowledge 

**Title (ZH)**: 算法的非自律性：传统知识与计算知识 

**Authors**: David M. Berry  

**Link**: [PDF](https://arxiv.org/pdf/2505.11030)  

**Abstract**: If an active citizen should increasingly be a computationally enlightened one, replacing the autonomy of reason with the heteronomy of algorithms, then I argue in this article that we must begin teaching the principles of critiquing the computal through new notions of what we might call digital Bildung. Indeed, if civil society itself is mediated by computational systems and media, the public use of reason must also be complemented by skills for negotiating and using these computal forms to articulate such critique. Not only is there a need to raise the intellectual tone regarding computation and its related softwarization processes, but there is an urgent need to attend to the likely epistemic challenges from computation which, as presently constituted, tends towards justification through a philosophy of utility rather than through a philosophy of care for the territory of the intellect. We therefore need to develop an approach to this field that uses concepts and methods drawn from philosophy, politics, history, anthropology, sociology, media studies, computer science, and the humanities more generally, to try to understand these issues - particularly the way in which software and data increasingly penetrate our everyday life and the pressures and fissures that are created. We must, in other words, move to undertake a critical interdisciplinary research program to understand the way in which these systems are created, instantiated, and normatively engendered in both specific and general contexts. 

**Abstract (ZH)**: 如果活跃公民应越来越多地成为一个具备计算素养的公民，并且算法自治取代了理性自治，那么本文认为我们必须开始通过新观念教授批判计算的原则，即所谓的数字博登教育。如果公民社会本身被计算系统和媒体中介化，那么公共理性的使用也必须通过协商和使用这些计算形式来阐述这样的批判。不仅需要提高关于计算及其相关软计算化进程的思想水平，而且还需要关注计算可能带来的认识论挑战，目前的计算倾向于通过实用主义哲学而非关怀哲学来寻求正当性。因此，我们需要开发一种方法，从哲学、政治学、历史学、人类学、 sociology、媒体研究、计算机科学和更广泛的文科领域借用概念和方法，以理解这些议题——特别是软件和数据如何越来越多地渗透到我们的日常生活以及由此产生的压力和裂隙。换句话说，我们必须开展一项批判性的跨学科研究项目，以理解这些系统的创建、实现及其在特定和一般背景下的规范性起源。 

---
# StRuCom: A Novel Dataset of Structured Code Comments in Russian 

**Title (ZH)**: StRuCom: 一种新的俄语结构化代码注释数据集 

**Authors**: Maria Dziuba, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11026)  

**Abstract**: Structured code comments in docstring format are essential for code comprehension and maintenance, but existing machine learning models for their generation perform poorly for Russian compared to English. To bridge this gap, we present StRuCom - the first large-scale dataset (153K examples) specifically designed for Russian code documentation. Unlike machine-translated English datasets that distort terminology (e.g., technical loanwords vs. literal translations) and docstring structures, StRuCom combines human-written comments from Russian GitHub repositories with synthetically generated ones, ensuring compliance with Python, Java, JavaScript, C#, and Go standards through automated validation. Fine-tuning Qwen2.5-Coder models (0.5B-7B) on StRuCom shows statistically significant improvements of chrf++ and BERTScore over baseline models. 

**Abstract (ZH)**: 结构化的代码注释以docstring格式对于代码理解和维护至关重要，但现有的生成俄罗斯语代码注释的机器学习模型在效果上逊于英语。为了弥合这一差距，我们提出了StRuCom——第一个专门针对俄罗斯代码文档的大规模数据集（包含153,000个示例）。不同于通过机器翻译生成的英语数据集可能会扭曲术语（例如技术借词与直译的区别）和docstring结构，StRuCom将来自俄罗斯GitHub仓库的人工编写注释与合成生成的注释相结合，并通过自动化验证确保其符合Python、Java、JavaScript、C#和Go的标准。在StRuCom上微调Qwen2.5-Coder模型（0.5B至7B参数）显示了在chrf++和BERTScore指标上的统计显著改进。 

---
# Humans expect rationality and cooperation from LLM opponents in strategic games 

**Title (ZH)**: 人类期望在战略游戏中LLM对手表现出理性与合作行为。 

**Authors**: Darija Barak, Miguel Costa-Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2505.11011)  

**Abstract**: As Large Language Models (LLMs) integrate into our social and economic interactions, we need to deepen our understanding of how humans respond to LLMs opponents in strategic settings. We present the results of the first controlled monetarily-incentivised laboratory experiment looking at differences in human behaviour in a multi-player p-beauty contest against other humans and LLMs. We use a within-subject design in order to compare behaviour at the individual level. We show that, in this environment, human subjects choose significantly lower numbers when playing against LLMs than humans, which is mainly driven by the increased prevalence of `zero' Nash-equilibrium choices. This shift is mainly driven by subjects with high strategic reasoning ability. Subjects who play the zero Nash-equilibrium choice motivate their strategy by appealing to perceived LLM's reasoning ability and, unexpectedly, propensity towards cooperation. Our findings provide foundational insights into the multi-player human-LLM interaction in simultaneous choice games, uncover heterogeneities in both subjects' behaviour and beliefs about LLM's play when playing against them, and suggest important implications for mechanism design in mixed human-LLM systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）融入社会和经济互动中，我们需要深化对人类在战略环境中如何应对LLMs对手的理解。我们呈现了首个控制下的、具有经济激励的实验室实验结果，该实验对比了人类在多人p-美丽塔 contest中与其他人类和LLMs对战时的行为差异。我们采用被试内设计以在个体层面进行行为比较。结果显示，在这种环境中，人类被试在与LLMs对战时选择的数字显著较低，这主要由`零’纳什均衡选择的增加所驱动。这种转变主要由具有较高战略推理能力的被试驱动。选择零纳什均衡的被试通过强调感知到的LLMs推理能力和意外的合作倾向来解释其策略。我们的研究成果为同时选择博弈中多玩家的人类-LLM互动提供了基础性见解，揭示了被试行为及关于LLMs行为的认知异质性，并对混合人类-LLM系统的设计提出了重要启示。 

---
# Review-Instruct: A Review-Driven Multi-Turn Conversations Generation Method for Large Language Models 

**Title (ZH)**: Review-Instruct：一种基于评论的多轮对话生成方法用于大型语言模型 

**Authors**: Jiangxu Wu, Cong Wang, TianHuang Su, Jun Yang, Haozhi Lin, Chao Zhang, Ming Peng, Kai Shi, SongPan Yang, BinQing Pan, ZiXian Li, Ni Yang, ZhenYu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11010)  

**Abstract**: The effectiveness of large language models (LLMs) in conversational AI is hindered by their reliance on single-turn supervised fine-tuning (SFT) data, which limits contextual coherence in multi-turn dialogues. Existing methods for generating multi-turn dialogue data struggle to ensure both diversity and quality in instructions. To address this, we propose Review-Instruct, a novel framework that synthesizes multi-turn conversations through an iterative "Ask-Respond-Review" process involving three agent roles: a Candidate, multiple Reviewers, and a Chairman. The framework iteratively refines instructions by incorporating Reviewer feedback, enhancing dialogue diversity and difficulty. We construct a multi-turn dataset using the Alpaca dataset and fine-tune the LLaMA2-13B model. Evaluations on MT-Bench, MMLU-Pro, and Auto-Arena demonstrate significant improvements, achieving absolute gains of 2.9\% on MMLU-Pro and 2\% on MT-Bench compared to prior state-of-the-art models based on LLaMA2-13B. Ablation studies confirm the critical role of the Review stage and the use of multiple Reviewers in boosting instruction diversity and difficulty. Our work highlights the potential of review-driven, multi-agent frameworks for generating high-quality conversational data at scale. 

**Abstract (ZH)**: 大规模语言模型在对话AI中的有效性受限于其对单轮监督微调数据的依赖，这限制了多轮对话的上下文连贯性。现有生成多轮对话数据的方法难以同时保证指令的多样性和质量。为解决这一问题，我们提出了一种名为Review-Instruct的新型框架，通过迭代的“提问-回应-评审”过程，涉及三种代理角色：候选人、多位评审员和主席。该框架通过整合评审员反馈，逐步细化指令，增强对话的多样性和难度。我们使用Alpaca数据集构建了多轮数据集，并对LLaMA2-13B模型进行了微调。在MT-Bench、MMLU-Pro和Auto-Arena上的评估结果显示了显著的改进，分别在MMLU-Pro和MT-Bench上取得2.9%和2%的绝对增益，超过了基于LLaMA2-13B的先前最佳模型。消融研究证实了评审阶段及使用多位评审员在提高指令多样性和难度方面的重要作用。我们的工作突显了审评驱动的多代理框架在大规模生成高质量对话数据方面的潜力。 

---
# Illusion or Algorithm? Investigating Memorization, Emergence, and Symbolic Processing in In-Context Learning 

**Title (ZH)**: 幻觉还是算法？探究上下文学习中的记忆、涌现和符号处理 

**Authors**: Jingcheng Niu, Subhabrata Dutta, Ahmed Elshabrawy, Harish Tayyar Madabushi, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2505.11004)  

**Abstract**: Large-scale Transformer language models (LMs) trained solely on next-token prediction with web-scale data can solve a wide range of tasks after seeing just a few examples. The mechanism behind this capability, known as in-context learning (ICL), remains both controversial and poorly understood. Some studies argue that it is merely the result of memorizing vast amounts of data, while others contend that it reflects a fundamental, symbolic algorithmic development in LMs. In this work, we introduce a suite of investigative tasks and a novel method to systematically investigate ICL by leveraging the full Pythia scaling suite, including interim checkpoints that capture progressively larger amount of training data. By carefully exploring ICL performance on downstream tasks and simultaneously conducting a mechanistic analysis of the residual stream's subspace, we demonstrate that ICL extends beyond mere "memorization" of the training corpus, yet does not amount to the implementation of an independent symbolic algorithm. Our results also clarify several aspects of ICL, including the influence of training dynamics, model capabilities, and elements of mechanistic interpretability. Overall, our work advances the understanding of ICL and its implications, offering model developers insights into potential improvements and providing AI security practitioners with a basis for more informed guidelines. 

**Abstract (ZH)**: 大规模Transformer语言模型在仅基于网页规模数据进行下个词预测训练的情况下，可以在看到少量示例后解决多种任务。这种能力背后的机制，即上下文内学习（ICL），仍然存在争议且不完全理解。一些研究认为这仅仅是大量数据的记忆结果，而另一些研究则认为这反映了模型的基本、符号算法发展。在本工作中，我们引入了一系列探索性任务和一种新的方法，通过利用完整的Pythia扩展套件及其间期检查点，逐步捕获越来越多的训练数据，系统地研究ICL。通过仔细探索ICL在下游任务上的性能，并同时进行残余流子空间的机制分析，我们证明ICL不仅超出了单纯的“记忆”训练语料库，但仍不具备独立符号算法的实现。我们的结果还澄清了ICL的几个方面，包括训练动力学的影响、模型能力以及机制可解释性元素。总体而言，本工作推进了对ICL及其影响的理解，为模型开发者提供了潜在改进的见解，并为AI安全从业者提供了更知情的指导基础。 

---
# Space Group Equivariant Crystal Diffusion 

**Title (ZH)**: 空间群共变晶体扩散 

**Authors**: Rees Chang, Angela Pak, Alex Guerra, Ni Zhan, Nick Richardson, Elif Ertekin, Ryan P. Adams  

**Link**: [PDF](https://arxiv.org/pdf/2505.10994)  

**Abstract**: Accelerating inverse design of crystalline materials with generative models has significant implications for a range of technologies. Unlike other atomic systems, 3D crystals are invariant to discrete groups of isometries called the space groups. Crucially, these space group symmetries are known to heavily influence materials properties. We propose SGEquiDiff, a crystal generative model which naturally handles space group constraints with space group invariant likelihoods. SGEquiDiff consists of an SE(3)-invariant, telescoping discrete sampler of crystal lattices; permutation-invariant, transformer-based autoregressive sampling of Wyckoff positions, elements, and numbers of symmetrically unique atoms; and space group equivariant diffusion of atomic coordinates. We show that space group equivariant vector fields automatically live in the tangent spaces of the Wyckoff positions. SGEquiDiff achieves state-of-the-art performance on standard benchmark datasets as assessed by quantitative proxy metrics and quantum mechanical calculations. 

**Abstract (ZH)**: 使用生成模型加速晶体材料逆设计在一系列技术中具有重要意义。空间群不变性 likelihoods 的晶体生成模型 SGEquiDiff 处理空间群约束。SGEquiDiff 包含 SE(3) 不变的分层晶格采样器；Wyckoff 位置、元素及对称唯一原子数的置换不变性、基于变压器的自回归采样器；以及原子坐标的空间群可变扩散。我们证明了空间群可变向量场自然存在于 Wyckoff 位置的切空间中。SGEquiDiff 在标准基准数据集上的定量代理指标和量子力学计算中达到最佳性能。 

---
# GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models 

**Title (ZH)**: GenoArmory：面向基因组基础模型对抗攻击的统一评估框架 

**Authors**: Haozheng Luo, Chenghao Qiu, Yimin Wang, Shang Wu, Jiahao Yu, Han Liu, Binghui Wang, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10983)  

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GenoArmory. Unlike existing GFM benchmarks, GenoArmory offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Additionally, we introduce GenoAdv, a new adversarial sample dataset designed to improve GFM safety. Empirically, classification models exhibit greater robustness to adversarial perturbations compared to generative models, highlighting the impact of task type on model vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features. 

**Abstract (ZH)**: 我们提出了首个针对基因组基础模型（GFMs）的统一对抗攻击基准，名为GenoArmory。不同于现有的GFMs基准，GenoArmory提供了首个全面的评估框架，系统地评估GFMs对对抗攻击的脆弱性。从方法论上，我们使用四种广泛采用的攻击算法和三种防御策略，评估了五种最先进的GFMs的对抗鲁棒性。重要的是，我们的基准提供了一个易于访问且全面的框架，用于分析模型架构、量化方案和训练数据集对GFMs脆弱性的影响。此外，我们引入了GenoAdv，一个新设计的对抗样本数据集，旨在提高GFMs的安全性。实验结果表明，分类模型比生成模型对对抗扰动更具鲁棒性，这突显了任务类型对模型脆弱性的影响。此外，对抗攻击经常针对生物上重要的基因组区域，表明这些模型有效地捕捉了有意义的序列特征。 

---
# Group-in-Group Policy Optimization for LLM Agent Training 

**Title (ZH)**: 组中组政策优化方法在大语言模型代理训练中的应用 

**Authors**: Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.10978)  

**Abstract**: Recent advances in group-based reinforcement learning (RL) have driven frontier large language models (LLMs) in single-turn tasks like mathematical reasoning. However, their scalability to long-horizon LLM agent training remains limited. Unlike static tasks, agent-environment interactions unfold over many steps and often yield sparse or delayed rewards, making credit assignment across individual steps significantly more challenging. In this work, we propose Group-in-Group Policy Optimization (GiGPO), a novel RL algorithm that achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence. GiGPO introduces a two-level structure for estimating relative advantage: (i) At the episode-level, GiGPO computes macro relative advantages based on groups of complete trajectories; (ii) At the step-level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories. Actions stemming from the same state are grouped together, enabling micro relative advantage estimation. This hierarchical structure effectively captures both global trajectory quality and local step effectiveness without relying on auxiliary models or additional rollouts. We evaluate GiGPO on two challenging agent benchmarks, ALFWorld and WebShop, using Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct. Crucially, GiGPO delivers fine-grained per-step credit signals and achieves performance gains of > 12\% on ALFWorld and > 9\% on WebShop over the GRPO baseline: all while maintaining the same GPU memory overhead, identical LLM rollout, and incurring little to no additional time cost. 

**Abstract (ZH)**: Recent advances in group-based reinforcement learning (RL) have driven frontier large language models (LLMs) in single-turn tasks like mathematical reasoning. However, their scalability to long-horizon LLM agent training remains limited. In this work, we propose Group-in-Group Policy Optimization (GiGPO), a novel RL algorithm that achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence. 

---
# Survey of End-to-End Multi-Speaker Automatic Speech Recognition for Monaural Audio 

**Title (ZH)**: 单声道音频端到端多说话人自动语音识别综述 

**Authors**: Xinlu He, Jacob Whitehill  

**Link**: [PDF](https://arxiv.org/pdf/2505.10975)  

**Abstract**: Monaural multi-speaker automatic speech recognition (ASR) remains challenging due to data scarcity and the intrinsic difficulty of recognizing and attributing words to individual speakers, particularly in overlapping speech. Recent advances have driven the shift from cascade systems to end-to-end (E2E) architectures, which reduce error propagation and better exploit the synergy between speech content and speaker identity. Despite rapid progress in E2E multi-speaker ASR, the field lacks a comprehensive review of recent developments. This survey provides a systematic taxonomy of E2E neural approaches for multi-speaker ASR, highlighting recent advances and comparative analysis. Specifically, we analyze: (1) architectural paradigms (SIMO vs.~SISO) for pre-segmented audio, analyzing their distinct characteristics and trade-offs; (2) recent architectural and algorithmic improvements based on these two paradigms; (3) extensions to long-form speech, including segmentation strategy and speaker-consistent hypothesis stitching. Further, we (4) evaluate and compare methods across standard benchmarks. We conclude with a discussion of open challenges and future research directions towards building robust and scalable multi-speaker ASR. 

**Abstract (ZH)**: 单声道多说话人自动语音识别（ASR）由于数据稀缺性和识别和归因给个体说话人词汇的固有难度，特别是在重叠语音中，依然具有挑战性。最近的进展推动了从级联系统向端到端（E2E）架构的转变，这减少了错误传播并更好地利用了语音内容与说话人身份之间的协同作用。尽管在E2E多说话人ASR领域取得了快速进展，但该领域缺乏对最近发展的全面综述。本文提供了E2E神经方法在多说话人ASR领域的系统分类，突出了最近的发展和比较分析。具体而言，我们分析了：（1）预分割音频的架构范式（SIMO vs. SISO），分析其各自的特点和权衡；（2）基于这两个范式的最近架构和算法改进；（3）长音频形式的扩展，包括分割策略和说话人一致假设拼接。此外，我们（4）在标准基准上评估和比较方法。最后，我们讨论了构建稳健和可扩展的多说话人ASR面临的主要挑战和未来研究方向。 

---
# GROQLoco: Generalist and RObot-agnostic Quadruped Locomotion Control using Offline Datasets 

**Title (ZH)**: GROQLoco: 通用且机器人无关的四足运动控制方法基于离线数据集 

**Authors**: Narayanan PP, Sarvesh Prasanth Venkatesan, Srinivas Kantha Reddy, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.10973)  

**Abstract**: Recent advancements in large-scale offline training have demonstrated the potential of generalist policy learning for complex robotic tasks. However, applying these principles to legged locomotion remains a challenge due to continuous dynamics and the need for real-time adaptation across diverse terrains and robot morphologies. In this work, we propose GROQLoco, a scalable, attention-based framework that learns a single generalist locomotion policy across multiple quadruped robots and terrains, relying solely on offline datasets. Our approach leverages expert demonstrations from two distinct locomotion behaviors - stair traversal (non-periodic gaits) and flat terrain traversal (periodic gaits) - collected across multiple quadruped robots, to train a generalist model that enables behavior fusion for both behaviors. Crucially, our framework operates directly on proprioceptive data from all robots without incorporating any robot-specific encodings. The policy is directly deployable on an Intel i7 nuc, producing low-latency control outputs without any test-time optimization. Our extensive experiments demonstrate strong zero-shot transfer across highly diverse quadruped robots and terrains, including hardware deployment on the Unitree Go1, a commercially available 12kg robot. Notably, we evaluate challenging cross-robot training setups where different locomotion skills are unevenly distributed across robots, yet observe successful transfer of both flat walking and stair traversal behaviors to all robots at test time. We also show preliminary walking on Stoch 5, a 70kg quadruped, on flat and outdoor terrains without requiring any fine tuning. These results highlight the potential for robust generalist locomotion across diverse robots and terrains. 

**Abstract (ZH)**: Recent advancements in large-scale offline training have demonstrated the potential of generalist policy learning for complex robotic tasks. However, applying these principles to legged locomotion remains a challenge due to continuous dynamics and the need for real-time adaptation across diverse terrains and robot morphologies. In this work, we propose GROQLoco, a scalable, attention-based framework that learns a single generalist locomotion policy across multiple quadruped robots and terrains, relying solely on offline datasets. 

---
# Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents 

**Title (ZH)**: 让庭审开始：基于LLM代理的模拟法庭方法用于脆弱性检测 

**Authors**: Ratnadira Widyasari, Martin Weyssow, Ivana Clairine Irsan, Han Wei Ang, Frank Liauw, Eng Lieh Ouh, Lwin Khin Shar, Hong Jin Kang, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10961)  

**Abstract**: Detecting vulnerabilities in source code remains a critical yet challenging task, especially when benign and vulnerable functions share significant similarities. In this work, we introduce VulTrial, a courtroom-inspired multi-agent framework designed to enhance automated vulnerability detection. It employs four role-specific agents, which are security researcher, code author, moderator, and review board. Through extensive experiments using GPT-3.5 and GPT-4o we demonstrate that Vultrial outperforms single-agent and multi-agent baselines. Using GPT-4o, VulTrial improves the performance by 102.39% and 84.17% over its respective baseline. Additionally, we show that role-specific instruction tuning in multi-agent with small data (50 pair samples) improves the performance of VulTrial further by 139.89% and 118.30%. Furthermore, we analyze the impact of increasing the number of agent interactions on VulTrial's overall performance. While multi-agent setups inherently incur higher costs due to increased token usage, our findings reveal that applying VulTrial to a cost-effective model like GPT-3.5 can improve its performance by 69.89% compared to GPT-4o in a single-agent setting, at a lower overall cost. 

**Abstract (ZH)**: VulTrial: 一种受法庭启发的多Agent框架以增强自动漏洞检测 

---
# Relational Graph Transformer 

**Title (ZH)**: 关系图变换器 

**Authors**: Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen, Federico López, Charilaos I. Kanatsoulis, Rishi Puri, Matthias Fey, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2505.10960)  

**Abstract**: Relational Deep Learning (RDL) is a promising approach for building state-of-the-art predictive models on multi-table relational data by representing it as a heterogeneous temporal graph. However, commonly used Graph Neural Network models suffer from fundamental limitations in capturing complex structural patterns and long-range dependencies that are inherent in relational data. While Graph Transformers have emerged as powerful alternatives to GNNs on general graphs, applying them to relational entity graphs presents unique challenges: (i) Traditional positional encodings fail to generalize to massive, heterogeneous graphs; (ii) existing architectures cannot model the temporal dynamics and schema constraints of relational data; (iii) existing tokenization schemes lose critical structural information. Here we introduce the Relational Graph Transformer (RelGT), the first graph transformer architecture designed specifically for relational tables. RelGT employs a novel multi-element tokenization strategy that decomposes each node into five components (features, type, hop distance, time, and local structure), enabling efficient encoding of heterogeneity, temporality, and topology without expensive precomputation. Our architecture combines local attention over sampled subgraphs with global attention to learnable centroids, incorporating both local and database-wide representations. Across 21 tasks from the RelBench benchmark, RelGT consistently matches or outperforms GNN baselines by up to 18%, establishing Graph Transformers as a powerful architecture for Relational Deep Learning. 

**Abstract (ZH)**: 关系图变换器（RelGT）：一种专门针对关系表的图变换器架构 

---
# Constrained Preferential Bayesian Optimization and Its Application in Banner Ad Design 

**Title (ZH)**: 受限偏好贝叶斯优化及其在Banner广告设计中的应用 

**Authors**: Koki Iwai, Yusuke Kumagae, Yuki Koyama, Masahiro Hamasaki, Masataka Goto  

**Link**: [PDF](https://arxiv.org/pdf/2505.10954)  

**Abstract**: Preferential Bayesian optimization (PBO) is a variant of Bayesian optimization that observes relative preferences (e.g., pairwise comparisons) instead of direct objective values, making it especially suitable for human-in-the-loop scenarios. However, real-world optimization tasks often involve inequality constraints, which existing PBO methods have not yet addressed. To fill this gap, we propose constrained preferential Bayesian optimization (CPBO), an extension of PBO that incorporates inequality constraints for the first time. Specifically, we present a novel acquisition function for this purpose. Our technical evaluation shows that our CPBO method successfully identifies optimal solutions by focusing on exploring feasible regions. As a practical application, we also present a designer-in-the-loop system for banner ad design using CPBO, where the objective is the designer's subjective preference, and the constraint ensures a target predicted click-through rate. We conducted a user study with professional ad designers, demonstrating the potential benefits of our approach in guiding creative design under real-world constraints. 

**Abstract (ZH)**: 约束偏好贝叶斯优化（CPBO）：首次将不等式约束纳入偏好贝叶斯优化中 

---
# Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-lingual Transfer 

**Title (ZH)**: 基于语义aware线性迁移的预训练语言模型再利用对于跨语言迁移学习 

**Authors**: Seungyoon Lee, Seongtae Hong, Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.10945)  

**Abstract**: Large Language Models (LLMs) increasingly incorporate multilingual capabilities, fueling the demand to transfer them into target language-specific models. However, most approaches, which blend the source model's embedding by replacing the source vocabulary with the target language-specific vocabulary, may constrain expressive capacity in the target language since the source model is predominantly trained on English data. In this paper, we propose Semantic Aware Linear Transfer (SALT), a novel cross-lingual transfer technique that recycles embeddings from target language Pre-trained Language Models (PLMs) to transmit the deep representational strengths of PLM-derived embedding to LLMs. SALT derives unique regression lines based on the similarity in the overlap of the source and target vocabularies, to handle each non-overlapping token's embedding space. Our extensive experiments show that SALT significantly outperforms other transfer methods and achieves lower loss with accelerating faster convergence during language adaptation. Notably, SALT obtains remarkable performance in cross-lingual understanding setups compared to other methods. Furthermore, we highlight the scalable use of PLMs to enhance the functionality of contemporary LLMs by conducting experiments with varying architectures. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益具备多语言能力，推动了将其转换为目标语言特定模型的需求。然而，大多数方法通过用目标语言特定词汇表替换源模型的词汇表来混合源模型的嵌入，这可能会在目标语言中限制表达能力，因为源模型主要是在英语数据上进行训练的。本文提出了一种新的跨语言转移技术——语义感知线性转移（SALT），该技术重新利用目标语言预训练语言模型（PLMs）的嵌入，将PLM衍生嵌入的深层表征传递给LLMs。SALT根据不同源目标词汇表重叠相似性为基础，为每个不重叠的令牌的嵌入空间推导出独特的回归线。我们的广泛实验表明，SALT在其他转移方法中表现出显著的优越性，并且在语言适应过程中具有更快的收敛速度，损失更低。值得一提的是，SALT在与其他方法相比的跨语言理解设置中取得了显著的性能。此外，我们通过使用不同架构进行实验展示了PLMs的可扩展使用如何增强当代LLMs的功能。 

---
# Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation 

**Title (ZH)**: 你是谁很重要：通过增强逻辑推荐连接主题与社会角色 

**Authors**: Qing Yu, Xiaobei Wang, Shuchang Liu, Yandong Bai, Xiaoyu Yang, Xueliang Wang, Chang Meng, Shanshan Wu, Hailan Yang, Huihui Xiao, Xiang Li, Fan Yang, Xiaoqiang Feng, Lantao Hu, Han Li, Kun Gai, Lixin Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.10940)  

**Abstract**: Recommender systems filter contents/items valuable to users by inferring preferences from user features and historical behaviors. Mainstream approaches follow the learning-to-rank paradigm, which focus on discovering and modeling item topics (e.g., categories), and capturing user preferences on these topics based on historical interactions. However, this paradigm often neglects the modeling of user characteristics and their social roles, which are logical confounders influencing the correlated interest and user preference transition. To bridge this gap, we introduce the user role identification task and the behavioral logic modeling task that aim to explicitly model user roles and learn the logical relations between item topics and user social roles. We show that it is possible to explicitly solve these tasks through an efficient integration framework of Large Language Model (LLM) and recommendation systems, for which we propose TagCF. On the one hand, the exploitation of the LLM's world knowledge and logic inference ability produces a virtual logic graph that reveals dynamic and expressive knowledge of users, augmenting the recommendation performance. On the other hand, the user role aligns the user behavioral logic with the observed user feedback, refining our understanding of user behaviors. Additionally, we also show that the extracted user-item logic graph is empirically a general knowledge that can benefit a wide range of recommendation tasks, and conduct experiments on industrial and several public datasets as verification. 

**Abstract (ZH)**: 推荐系统通过推理用户特征和历史行为来过滤对用户有价值的内容/项。主流方法遵循学习排序范式，侧重于发现和建模项目主题（如类别），并基于历史交互捕捉用户在这些主题上的偏好。然而，这一范式往往忽视了用户特征及其社会角色的建模，而这些特征和社会角色是影响相关兴趣和用户偏好变化的合逻辑混杂因素。为了弥合这一缺口，我们引入了用户角色识别任务和行为逻辑建模任务，旨在明确建模用户角色并学习项目主题和用户社会角色之间的逻辑关系。我们展示了一种通过大型语言模型（LLM）与推荐系统高效集成框架显式解决这些问题的可能性，并为此提出了TagCF。一方面，利用LLM的世界知识和逻辑推理能力生成一个虚拟逻辑图，揭示了用户动态且丰富的知识，从而增强推荐性能。另一方面，用户角色使用户行为逻辑与观察到的用户反馈相一致，细化了我们对用户行为的理解。此外，我们还展示了提取的用户-项目逻辑图是一种经验上的普适知识，能够广泛应用于各种推荐任务，并在工业和多个公开数据集上进行了实验以进行验证。 

---
# GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction 

**Title (ZH)**: GenKnowSub: 通过通用知识减法提高LLMs的模块化与可重用性 

**Authors**: Mohammadtaha Bagherifard, Sahar Rajabi, Ali Edalat, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.10939)  

**Abstract**: Large language models often struggle with zero-shot generalization, and several modular approaches have been proposed to address this challenge. Yet, we hypothesize that a key limitation remains: the entanglement of general knowledge and task-specific adaptations. To overcome this, we propose a modular framework that disentangles these components by constructing a library of task-specific LoRA modules alongside a general-domain LoRA. By subtracting this general knowledge component from each task-specific module, we obtain residual modules that focus more exclusively on task-relevant information, a method we call general knowledge subtraction (GenKnowSub). Leveraging the refined task-specific modules and the Arrow routing algorithm \citep{ostapenko2024towards}, we dynamically select and combine modules for new inputs without additional training. Our studies on the Phi-3 model and standard Arrow as baselines reveal that using general knowledge LoRAs derived from diverse languages, including English, French, and German, yields consistent performance gains in both monolingual and cross-lingual settings across a wide set of benchmarks. Further experiments on Phi-2 demonstrate how GenKnowSub generalizes to weaker LLMs. The complete code and data are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型在零样本泛化方面常常表现不佳，提出了几种模块化方法来应对这一挑战。然而，我们假设一个关键限制仍然存在：一般知识与任务特定适应的纠缠。为克服这一限制，我们提出了一种模块化框架，通过构建任务特定的LoRA模块库和通用领域LoRA，分离这些组件。通过从每个任务特定模块中减去一般知识成分，我们获得了更专注于任务相关信息的残差模块，该方法称为一般知识减法（GenKnowSub）。利用优化后的任务特定模块和Arrow路由算法，我们可以在不需要额外训练的情况下动态选择和组合模块以处理新输入。我们在Phi-3模型和标准Arrow作为基线的研究表明，使用来自多种语言（包括英语、法语和德语）的一般知识LoRAs，可在单一语言和跨语言设置中的一系列基准测试中实现一致的性能提升。进一步在Phi-2上的实验表明GenKnowSub如何应用于较弱的大规模语言模型。完整代码和数据可在以下链接获取。 

---
# Reasoning with OmniThought: A Large CoT Dataset with Verbosity and Cognitive Difficulty Annotations 

**Title (ZH)**: OmniThought上的推理：一个带有详细程度和认知难度注释的大规模CoT数据集 

**Authors**: Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10937)  

**Abstract**: The emergence of large reasoning models (LRMs) has transformed Natural Language Processing by excelling in complex tasks such as mathematical problem-solving and code generation. These models leverage chain-of-thought (CoT) processes, enabling them to emulate human-like reasoning strategies. However, the advancement of LRMs is hindered by the lack of comprehensive CoT datasets. Current resources often fail to provide extensive reasoning problems with coherent CoT processes distilled from multiple teacher models and do not account for multifaceted properties describing the internal characteristics of CoTs. To address these challenges, we introduce OmniThought, a large-scale dataset featuring 2 million CoT processes generated and validated by two powerful LRMs as teacher models. Each CoT process in OmniThought is annotated with novel Reasoning Verbosity (RV) and Cognitive Difficulty (CD) scores, which describe the appropriateness of CoT verbosity and cognitive difficulty level for models to comprehend these reasoning processes. We further establish a self-reliant pipeline to curate this dataset. Extensive experiments using Qwen2.5 models of various sizes demonstrate the positive impact of our proposed scores on LRM training effectiveness. Based on the proposed OmniThought dataset, we further train and release a series of high-performing LRMs, specifically equipped with stronger reasoning abilities and optimal CoT output length and difficulty level. Our contributions significantly enhance the development and training of LRMs for solving complex tasks. 

**Abstract (ZH)**: 大型推理模型的 emergence 已经通过在数学问题解决和代码生成等复杂任务中的卓越表现，彻底改变了自然语言处理。这些模型利用链式思考 (CoT) 过程，能够模拟类人的推理策略。然而，大型推理模型的进步受到了全面 CoT 数据集缺乏的阻碍。当前资源往往无法提供来自多个教师模型并具有连贯 CoT 过程的广泛推理问题，并且没有考虑到描述 CoT 内部特性的多种性质。为了解决这些挑战，我们引入了 OmniThought，这是一个包含两百万个 CoT 过程的大规模数据集，这些 CoT 过程由两个强大的大型推理模型作为教师模型生成和验证。OmniThought 中的每个 CoT 过程都标注了新颖的推理冗余 (RV) 和认知难度 (CD) 分数，这些分数描述了 CoT 的适当冗余程度及其认知难度水平，以供模型理解这些推理过程。我们进一步建立了一个自洽的工作流来整理这个数据集。使用不同规模的 Qwen2.5 模型进行的广泛实验表明，我们提出的分数对大型推理模型训练的有效性具有积极影响。基于提出的 OmniThought 数据集，我们进一步训练并发布了多个高性能的大型推理模型，这些模型具有更强的推理能力，并优化了 CoT 输出的长度和难度级别。我们的贡献显著提高了解决复杂任务的大规模推理模型的开发和训练。 

---
# A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron? 

**Title (ZH)**: 关于计算机使用代理的安全与安全威胁概览：JARVIS还是Ultron？ 

**Authors**: Ada Chen, Yongjiang Wu, Junyuan Zhang, Shu Yang, Jen-tse Huang, Kun Wang, Wenxuan Wang, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10924)  

**Abstract**: Recently, AI-driven interactions with computing devices have advanced from basic prototype tools to sophisticated, LLM-based systems that emulate human-like operations in graphical user interfaces. We are now witnessing the emergence of \emph{Computer-Using Agents} (CUAs), capable of autonomously performing tasks such as navigating desktop applications, web pages, and mobile apps. However, as these agents grow in capability, they also introduce novel safety and security risks. Vulnerabilities in LLM-driven reasoning, with the added complexity of integrating multiple software components and multimodal inputs, further complicate the security landscape. In this paper, we present a systematization of knowledge on the safety and security threats of CUAs. We conduct a comprehensive literature review and distill our findings along four research objectives: \textit{\textbf{(i)}} define the CUA that suits safety analysis; \textit{\textbf{(ii)} } categorize current safety threats among CUAs; \textit{\textbf{(iii)}} propose a comprehensive taxonomy of existing defensive strategies; \textit{\textbf{(iv)}} summarize prevailing benchmarks, datasets, and evaluation metrics used to assess the safety and performance of CUAs. Building on these insights, our work provides future researchers with a structured foundation for exploring unexplored vulnerabilities and offers practitioners actionable guidance in designing and deploying secure Computer-Using Agents. 

**Abstract (ZH)**: 最近，基于AI的与计算设备的交互从基础的原型工具发展成为以大规模语言模型为基础的复杂系统，这些系统能够在图形用户界面中模拟人类的操作。我们现在正见证计算机使用代理（CUAs）的出现，它们能够自主执行诸如导航桌面应用程序、网页和移动应用的任务。然而，随着这些代理能力的增长，它们也引入了新的安全和安全风险。由大规模语言模型驱动的推理中的漏洞，以及多种软件组件和多模态输入的集成复杂性，进一步加剧了安全态势。在本文中，我们对CUAs的安全和安全威胁进行了系统化知识整理。我们进行了一项全面的文献综述，并提炼出四个研究目标：（i）定义适合安全分析的CUA；（ii）对CUAs当前的安全威胁进行分类；（iii）提出现有防御策略的综合分类；（iv）总结评估CUAs安全性和性能的现有基准、数据集和评价指标。基于这些见解，我们的工作为未来的研究者提供了探索未探索漏洞的结构化基础，并为实践者提供了设计和部署安全的计算机使用代理的实际指导。 

---
# Vaiage: A Multi-Agent Solution to Personalized Travel Planning 

**Title (ZH)**: Vaiage: 一种基于多agent的个性化旅行规划解决方案 

**Authors**: Binwen Liu, Jiexi Ge, Jiamin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10922)  

**Abstract**: Planning trips is a cognitively intensive task involving conflicting user preferences, dynamic external information, and multi-step temporal-spatial optimization. Traditional platforms often fall short - they provide static results, lack contextual adaptation, and fail to support real-time interaction or intent refinement.
Our approach, Vaiage, addresses these challenges through a graph-structured multi-agent framework built around large language models (LLMs) that serve as both goal-conditioned recommenders and sequential planners. LLMs infer user intent, suggest personalized destinations and activities, and synthesize itineraries that align with contextual constraints such as budget, timing, group size, and weather. Through natural language interaction, structured tool use, and map-based feedback loops, Vaiage enables adaptive, explainable, and end-to-end travel planning grounded in both symbolic reasoning and conversational understanding.
To evaluate Vaiage, we conducted human-in-the-loop experiments using rubric-based GPT-4 assessments and qualitative feedback. The full system achieved an average score of 8.5 out of 10, outperforming the no-strategy (7.2) and no-external-API (6.8) variants, particularly in feasibility. Qualitative analysis indicated that agent coordination - especially the Strategy and Information Agents - significantly improved itinerary quality by optimizing time use and integrating real-time context. These results demonstrate the effectiveness of combining LLM reasoning with symbolic agent coordination in open-ended, real-world planning tasks. 

**Abstract (ZH)**: 行程规划是一项认知密集型任务，涉及用户的冲突偏好、动态外部信息以及多步骤的时间空间优化。传统平台往往力不从心——它们提供的往往是静态结果，缺乏上下文适应性，无法支持实时互动或意图细化。
我们的方法Vaiage通过一个基于图结构的多agent框架来应对这些挑战，该框架围绕大型语言模型（LLMs）构建，LLMs既作为目标条件的推荐器，又作为顺序规划者。LLMs推断用户意图，建议个性化的目的地和活动，并综合符合预算、时间、团体规模和天气等上下文约束的行程安排。通过自然语言交互、结构化工具使用和基于地图的反馈循环，Vaiage实现了基于符号推理和对话理解的适应性、可解释的端到端旅行规划。
为了评估Vaiage，我们采用了基于评分标准的GPT-4评估和定性反馈进行人机交互实验。整个系统平均得分为8.5分，显著高于无策略（7.2分）和无外部API（6.8分）变体，特别是在可行性方面。定性分析表明，agent之间的协调，尤其是策略agent和信息agent，通过优化时间使用和整合实时上下文，显著提高了行程质量。这些结果证实了将LLM推理与符号agent协调结合在开放性和实际规划任务中的有效性。 

---
# Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks 

**Title (ZH)**: Phi：利用基于模式的层次稀疏性构建高效突触神经网络 

**Authors**: Chiyue Wei, Bowen Duan, Cong Guo, Jingyang Zhang, Qingyue Song, Hai "Helen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10909)  

**Abstract**: Spiking Neural Networks (SNNs) are gaining attention for their energy efficiency and biological plausibility, utilizing 0-1 activation sparsity through spike-driven computation. While existing SNN accelerators exploit this sparsity to skip zero computations, they often overlook the unique distribution patterns inherent in binary activations. In this work, we observe that particular patterns exist in spike activations, which we can utilize to reduce the substantial computation of SNN models. Based on these findings, we propose a novel \textbf{pattern-based hierarchical sparsity} framework, termed \textbf{\textit{Phi}}, to optimize computation.
\textit{Phi} introduces a two-level sparsity hierarchy: Level 1 exhibits vector-wise sparsity by representing activations with pre-defined patterns, allowing for offline pre-computation with weights and significantly reducing most runtime computation. Level 2 features element-wise sparsity by complementing the Level 1 matrix, using a highly sparse matrix to further reduce computation while maintaining accuracy. We present an algorithm-hardware co-design approach. Algorithmically, we employ a k-means-based pattern selection method to identify representative patterns and introduce a pattern-aware fine-tuning technique to enhance Level 2 sparsity. Architecturally, we design \textbf{\textit{Phi}}, a dedicated hardware architecture that efficiently processes the two levels of \textit{Phi} sparsity on the fly. Extensive experiments demonstrate that \textit{Phi} achieves a $3.45\times$ speedup and a $4.93\times$ improvement in energy efficiency compared to state-of-the-art SNN accelerators, showcasing the effectiveness of our framework in optimizing SNN computation. 

**Abstract (ZH)**: 基于模式的分层稀疏性框架（Phi）：优化脉冲神经网络计算 

---
# On the Security Risks of ML-based Malware Detection Systems: A Survey 

**Title (ZH)**: 基于机器学习的恶意软件检测系统安全风险综述 

**Authors**: Ping He, Yuhao Mao, Changjiang Li, Lorenzo Cavallaro, Ting Wang, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.10903)  

**Abstract**: Malware presents a persistent threat to user privacy and data integrity. To combat this, machine learning-based (ML-based) malware detection (MD) systems have been developed. However, these systems have increasingly been attacked in recent years, undermining their effectiveness in practice. While the security risks associated with ML-based MD systems have garnered considerable attention, the majority of prior works is limited to adversarial malware examples, lacking a comprehensive analysis of practical security risks. This paper addresses this gap by utilizing the CIA principles to define the scope of security risks. We then deconstruct ML-based MD systems into distinct operational stages, thus developing a stage-based taxonomy. Utilizing this taxonomy, we summarize the technical progress and discuss the gaps in the attack and defense proposals related to the ML-based MD systems within each stage. Subsequently, we conduct two case studies, using both inter-stage and intra-stage analyses according to the stage-based taxonomy to provide new empirical insights. Based on these analyses and insights, we suggest potential future directions from both inter-stage and intra-stage perspectives. 

**Abstract (ZH)**: 基于机器学习的恶意软件检测系统面临持久的隐私和数据完整性威胁。尽管已经开发了基于机器学习的恶意软件检测系统（ML-based MD），但这些系统近年来不断受到攻击，影响了其实际效果。虽然与基于机器学习的恶意软件检测系统相关的安全风险已经引起了广泛关注，但大多数前期工作主要集中在对抗性恶意软件样本上，缺乏对实际安全风险的全面分析。本文通过利用CIA原则来定义安全风险的范围，并将基于机器学习的恶意软件检测系统分解为不同的操作阶段，从而构建一个阶段导向的分类体系。利用这一分类体系，我们总结了技术进步，讨论了每个阶段与基于机器学习的恶意软件检测系统相关的攻击和防御方案中的差距。随后，我们进行了两个案例研究，根据阶段导向分类体系进行跨阶段和同阶段分析，提供了新的实证见解。基于这些分析和见解，我们从跨阶段和同阶段两个视角提出了潜在的未来方向。 

---
# Explain What You Mean: Intent Augmented Knowledge Graph Recommender Built With LLM 

**Title (ZH)**: 展nią你的意思：基于大模型的意图增强知识图谱推荐系统 

**Authors**: Wenqing Zheng, Noah Fatsi, Daniel Barcklow, Dmitri Kalaev, Steven Yao, Owen Reinert, C. Bayan Bruss, Daniele Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2505.10900)  

**Abstract**: Interaction sparsity is the primary obstacle for recommendation systems. Sparsity manifests in environments with disproportional cardinality of groupings of entities, such as users and products in an online marketplace. It also is found for newly introduced entities, described as the cold-start problem. Recent efforts to mitigate this sparsity issue shifts the performance bottleneck to other areas in the computational pipeline. Those that focus on enriching sparse representations with connectivity data from other external sources propose methods that are resource demanding and require careful domain expert aided addition of this newly introduced data. Others that turn to Large Language Model (LLM) based recommenders will quickly encounter limitations surrounding data quality and availability. In this work, we propose LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that leverages retrieval-augmented generation and an encoding approach to construct and densify a knowledge graph. IKGR learns latent user-item affinities from an interaction knowledge graph and further densifies it through mutual intent connectivity. This addresses sparsity issues and allows the model to make intent-grounded recommendations with an interpretable embedding translation layer. Through extensive experiments on real-world datasets, we demonstrate that IKGR overcomes knowledge gaps and achieves substantial gains over state-of-the-art baselines on both publicly available and our internal recommendation datasets. 

**Abstract (ZH)**: 基于大语言模型的意图知识图谱推荐器（IKGR）：一种利用检索增强生成和编码方法构建并稠密化知识图谱的新框架 

---
# BanglaFake: Constructing and Evaluating a Specialized Bengali Deepfake Audio Dataset 

**Title (ZH)**: BanglaFake: 构建与评估一个专门的孟加拉语深度假音音频数据集 

**Authors**: Istiaq Ahmed Fahad, Kamruzzaman Asif, Sifat Sikder  

**Link**: [PDF](https://arxiv.org/pdf/2505.10885)  

**Abstract**: Deepfake audio detection is challenging for low-resource languages like Bengali due to limited datasets and subtle acoustic features. To address this, we introduce BangalFake, a Bengali Deepfake Audio Dataset with 12,260 real and 13,260 deepfake utterances. Synthetic speech is generated using SOTA Text-to-Speech (TTS) models, ensuring high naturalness and quality. We evaluate the dataset through both qualitative and quantitative analyses. Mean Opinion Score (MOS) from 30 native speakers shows Robust-MOS of 3.40 (naturalness) and 4.01 (intelligibility). t-SNE visualization of MFCCs highlights real vs. fake differentiation challenges. This dataset serves as a crucial resource for advancing deepfake detection in Bengali, addressing the limitations of low-resource language research. 

**Abstract (ZH)**: Bengali Deepfake Audio Dataset: Addressing Challenges in Low-Resource Language Deepfake Detection 

---
# Graph and Simplicial Complex Prediction Gaussian Process via the Hodgelet Representations 

**Title (ZH)**: Hodgelet表示下的图与 simplicial 复杂网络预测高斯过程 

**Authors**: Mathieu Alain, So Takao, Xiaowen Dong, Bastian Rieck, Emmanuel Noutahi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10877)  

**Abstract**: Predicting the labels of graph-structured data is crucial in scientific applications and is often achieved using graph neural networks (GNNs). However, when data is scarce, GNNs suffer from overfitting, leading to poor performance. Recently, Gaussian processes (GPs) with graph-level inputs have been proposed as an alternative. In this work, we extend the Gaussian process framework to simplicial complexes (SCs), enabling the handling of edge-level attributes and attributes supported on higher-order simplices. We further augment the resulting SC representations by considering their Hodge decompositions, allowing us to account for homological information, such as the number of holes, in the SC. We demonstrate that our framework enhances the predictions across various applications, paving the way for GPs to be more widely used for graph and SC-level predictions. 

**Abstract (ZH)**: 预测图结构数据的标签在科学应用中至关重要，通常使用图神经网络（GNNs）实现。然而，当数据稀缺时，GNNs会遭受过拟合，导致性能不佳。近年来，作为替代方案，基于图级输入的高斯过程（GPs）已被提出。在本文中，我们扩展了高斯过程框架至单纯复形（SCs），使得能够处理边级属性以及支持于高维单纯形上的属性。在此基础上，我们通过考虑单纯复形的亥姆霍兹分解进一步增强其表示，允许我们捕捉单纯复形中的同调信息，如洞的数量。我们证明，我们的框架在各种应用中增强了预测能力，为GPs在图和单纯复形级别上的广泛应用铺平了道路。 

---
# Preference Isolation Forest for Structure-based Anomaly Detection 

**Title (ZH)**: 基于结构的异常检测中的偏好隔离森林 

**Authors**: Filippo Leveni, Luca Magri, Cesare Alippi, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10876)  

**Abstract**: We address the problem of detecting anomalies as samples that do not conform to structured patterns represented by low-dimensional manifolds. To this end, we conceive a general anomaly detection framework called Preference Isolation Forest (PIF), that combines the benefits of adaptive isolation-based methods with the flexibility of preference embedding. The key intuition is to embed the data into a high-dimensional preference space by fitting low-dimensional manifolds, and to identify anomalies as isolated points. We propose three isolation approaches to identify anomalies: $i$) Voronoi-iForest, the most general solution, $ii$) RuzHash-iForest, that avoids explicit computation of distances via Local Sensitive Hashing, and $iii$) Sliding-PIF, that leverages a locality prior to improve efficiency and effectiveness. 

**Abstract (ZH)**: 基于偏好隔离森林的异常检测框架 

---
# MultiLink: Multi-class Structure Recovery via Agglomerative Clustering and Model Selection 

**Title (ZH)**: 多链接：基于凝聚聚类和模型选择的多类别结构恢复 

**Authors**: Luca Magri, Filippo Leveni, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10874)  

**Abstract**: We address the problem of recovering multiple structures of different classes in a dataset contaminated by noise and outliers. In particular, we consider geometric structures defined by a mixture of underlying parametric models (e.g. planes and cylinders, homographies and fundamental matrices), and we tackle the robust fitting problem by preference analysis and clustering. We present a new algorithm, termed MultiLink, that simultaneously deals with multiple classes of models. MultiLink combines on-the-fly model fitting and model selection in a novel linkage scheme that determines whether two clusters are to be merged. The resulting method features many practical advantages with respect to methods based on preference analysis, being faster, less sensitive to the inlier threshold, and able to compensate limitations deriving from hypotheses sampling. Experiments on several public datasets demonstrate that Multi-Link favourably compares with state of the art alternatives, both in multi-class and single-class problems. Code is publicly made available for download. 

**Abstract (ZH)**: 我们研究了在含有噪声和离群点的数据集中恢复不同类别的多个结构的问题。特别是，我们考虑由多个底层参数模型的混合定义的几何结构（例如平面和圆柱、仿射变换和基本矩阵），并通过偏好分析和聚类解决稳健拟合问题。我们提出了一种新的算法，称为MultiLink，该算法能够同时处理多种模型类别。MultiLink通过一种新颖的链接方案结合了模型的即席拟合和模型选择，该方案决定了两个聚类是否应被合并。该方法相对于基于偏好分析的方法具有许多实用优势，例如速度更快、对残余阈值的敏感性较低，并且能够补偿由于假设采样引起的限制。在多个公开数据集上的实验表明，MultiLink在多类和单类问题中都优于现有的解决方案。代码已公开提供下载。 

---
# Hashing for Structure-based Anomaly Detection 

**Title (ZH)**: 基于结构的异常检测的哈希方法 

**Authors**: Filippo Leveni, Luca Magri, Cesare Alippi, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10873)  

**Abstract**: We focus on the problem of identifying samples in a set that do not conform to structured patterns represented by low-dimensional manifolds. An effective way to solve this problem is to embed data in a high dimensional space, called Preference Space, where anomalies can be identified as the most isolated points. In this work, we employ Locality Sensitive Hashing to avoid explicit computation of distances in high dimensions and thus improve Anomaly Detection efficiency. Specifically, we present an isolation-based anomaly detection technique designed to work in the Preference Space which achieves state-of-the-art performance at a lower computational cost. Code is publicly available at this https URL. 

**Abstract (ZH)**: 我们关注识别不符合由低维流形表示的结构模式的数据样本的问题。解决这一问题的有效方法是将数据嵌入一个称为偏好空间的高维空间，在该空间中，异常点可以被认为是孤立度最大的点。在本工作中，我们采用局部敏感哈希来避免高维空间中的显式距离计算，从而提高异常检测效率。具体而言，我们提出了一种基于隔离的异常检测技术，该技术设计用于偏好空间中工作，能够在较低的计算成本下实现最先进的性能。代码可在以下网址公开获取。 

---
# REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning? 

**Title (ZH)**: REI-Bench: 机器人物体理解含糊的人类任务指示能力研究？ 

**Authors**: Chenxi Jiang, Chuhao Zhou, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10872)  

**Abstract**: Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark with vague REs (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 77.9%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompt and chains of thought. This work contributes to the research community of human-robot interaction (HRI) by making robot task planning more practical, particularly for non-expert users, e.g., the elderly and children. 

**Abstract (ZH)**: 基于含模糊指代表达式的机器人任务规划基准：理解与克服指令歧义以提升机器人任务规划性能 

---
# Optimal Allocation of Privacy Budget on Hierarchical Data Release 

**Title (ZH)**: 层级数据发布中隐私预算的最优分配 

**Authors**: Joonhyuk Ko, Juba Ziani, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2505.10871)  

**Abstract**: Releasing useful information from datasets with hierarchical structures while preserving individual privacy presents a significant challenge. Standard privacy-preserving mechanisms, and in particular Differential Privacy, often require careful allocation of a finite privacy budget across different levels and components of the hierarchy. Sub-optimal allocation can lead to either excessive noise, rendering the data useless, or to insufficient protections for sensitive information. This paper addresses the critical problem of optimal privacy budget allocation for hierarchical data release. It formulates this challenge as a constrained optimization problem, aiming to maximize data utility subject to a total privacy budget while considering the inherent trade-offs between data granularity and privacy loss. The proposed approach is supported by theoretical analysis and validated through comprehensive experiments on real hierarchical datasets. These experiments demonstrate that optimal privacy budget allocation significantly enhances the utility of the released data and improves the performance of downstream tasks. 

**Abstract (ZH)**: 具有层次结构的数据集在释放有用信息的同时保护个体隐私 presents a significant challenge. Optimal Privacy Budget Allocation for Hierarchical Data Release 

---
# Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate 

**Title (ZH)**: 自引发电和社会相关性重估的规则检索与推理改进 

**Authors**: Ziyang Huang, Wangtao Sun, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10870)  

**Abstract**: This paper systematically addresses the challenges of rule retrieval, a crucial yet underexplored area. Vanilla retrieval methods using sparse or dense retrievers to directly search for relevant rules to support downstream reasoning, often suffer from low accuracy. This is primarily due to a significant semantic gap between the instantiated facts in the queries and the abstract representations of the rules. Such misalignment results in suboptimal retrieval quality, which in turn negatively impacts reasoning performance. To overcome these challenges, we propose Self-Induction Augmented Retrieval (SIAR), a novel approach that utilizes Large Language Models (LLMs) to induce potential inferential rules that might offer benefits for reasoning by abstracting the underlying knowledge and logical structure in queries. These induced rules are then used for query augmentation to improve retrieval effectiveness. Additionally, we introduce Rule Relevance ReEstimate (R$^3$), a method that re-estimates the relevance of retrieved rules by assessing whether the abstract knowledge they contain can be instantiated to align with the facts in the queries and the helpfulness for reasoning. Extensive experiments across various settings demonstrate the effectiveness and versatility of our proposed methods. 

**Abstract (ZH)**: 这篇论文系统地探讨了规则检索这一关键且尚未充分探索的领域所面临的挑战。通过利用稀疏或密集检索器直接搜索相关的规则以支持下游推理的简单检索方法，通常会遭受低准确率的问题。这主要是由于查询中实例化事实与规则的抽象表示之间存在显著的语义差距。这种不一致导致检索质量低下，进而负面影响推理性能。为克服这些挑战，我们提出了自诱导增强检索（SIAR）方法，该方法利用大型语言模型（LLMs）从查询中抽象出潜在的推理规则，以增强推理所需的知识和逻辑结构。这些诱导出的规则随后用于查询增强以提高检索效果。此外，我们引入了规则相关性重新评估（R$^3$）方法，该方法通过评估检索出的规则中所含抽象知识是否能够实例化并与查询中的事实以及推理的相关性对齐来重新估计规则的相关性。在各种设置下进行的广泛实验验证了我们所提出方法的有效性和灵活性。 

---
# ImputeINR: Time Series Imputation via Implicit Neural Representations for Disease Diagnosis with Missing Data 

**Title (ZH)**: ImputeINR：通过隐式神经表示进行时间序列插补以处理缺失数据的疾病诊断 

**Authors**: Mengxuan Li, Ke Liu, Jialong Guo, Jiajun Bu, Hongwei Wang, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10856)  

**Abstract**: Healthcare data frequently contain a substantial proportion of missing values, necessitating effective time series imputation to support downstream disease diagnosis tasks. However, existing imputation methods focus on discrete data points and are unable to effectively model sparse data, resulting in particularly poor performance for imputing substantial missing values. In this paper, we propose a novel approach, ImputeINR, for time series imputation by employing implicit neural representations (INR) to learn continuous functions for time series. ImputeINR leverages the merits of INR in that the continuous functions are not coupled to sampling frequency and have infinite sampling frequency, allowing ImputeINR to generate fine-grained imputations even on extremely sparse observed values. Extensive experiments conducted on eight datasets with five ratios of masked values show the superior imputation performance of ImputeINR, especially for high missing ratios in time series data. Furthermore, we validate that applying ImputeINR to impute missing values in healthcare data enhances the performance of downstream disease diagnosis tasks. Codes are available. 

**Abstract (ZH)**: 健康数据经常包含大量的缺失值， necessitating 有效的时间序列插补以支持下游的疾病诊断任务。然而，现有的插补方法主要针对离散数据点，无法有效地建模稀疏数据，导致在插补大量缺失值时表现特别差。本文提出了一种新的方法 ImputeINR，通过使用隐式神经表示（INR）来学习时间序列的连续函数进行时间序列插补。ImputeINR 利用了 INR 的优点，即连续函数与采样频率无关并且具有无限的采样频率，使得 ImputeINR 能够在极端稀疏的观测值上生成细粒度的插补。在八个数据集上进行的实验结果表明，ImputeINR 在时间序列数据中的高缺失比例插补中表现尤为出色。此外，我们验证了将 ImputeINR 应用于医疗健康数据的缺失值插补可以提升下游疾病诊断任务的性能。代码已开源。 

---
# Ready2Unlearn: A Learning-Time Approach for Preparing Models with Future Unlearning Readiness 

**Title (ZH)**: Ready2Unlearn: 一种为未来遗忘准备的训练时方法 

**Authors**: Hanyu Duan, Yi Yang, Ahmed Abbasi, Kar Yan Tam  

**Link**: [PDF](https://arxiv.org/pdf/2505.10845)  

**Abstract**: This paper introduces Ready2Unlearn, a learning-time optimization approach designed to facilitate future unlearning processes. Unlike the majority of existing unlearning efforts that focus on designing unlearning algorithms, which are typically implemented reactively when an unlearning request is made during the model deployment phase, Ready2Unlearn shifts the focus to the training phase, adopting a "forward-looking" perspective. Building upon well-established meta-learning principles, Ready2Unlearn proactively trains machine learning models with unlearning readiness, such that they are well prepared and can handle future unlearning requests in a more efficient and principled manner. Ready2Unlearn is model-agnostic and compatible with any gradient ascent-based machine unlearning algorithms. We evaluate the method on both vision and language tasks under various unlearning settings, including class-wise unlearning and random data unlearning. Experimental results show that by incorporating such preparedness at training time, Ready2Unlearn produces an unlearning-ready model state, which offers several key advantages when future unlearning is required, including reduced unlearning time, improved retention of overall model capability, and enhanced resistance to the inadvertent recovery of forgotten data. We hope this work could inspire future efforts to explore more proactive strategies for equipping machine learning models with built-in readiness towards more reliable and principled machine unlearning. 

**Abstract (ZH)**: Ready2Unlearn：一种促进未来遗忘过程的训练时优化方法 

---
# Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL 

**Title (ZH)**: 学习何时思考：通过多阶段 RL 形成 R1风格模型中的适应性推理 

**Authors**: Songjun Tu, Jiahao Lin, Qichao Zhang, Xiangyu Tian, Linjing Li, Xiangyuan Lan, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10832)  

**Abstract**: Large reasoning models (LRMs) are proficient at generating explicit, step-by-step reasoning sequences before producing final answers. However, such detailed reasoning can introduce substantial computational overhead and latency, particularly for simple problems. To address this over-thinking problem, we explore how to equip LRMs with adaptive thinking capabilities: enabling them to dynamically decide whether or not to engage in explicit reasoning based on problem complexity. Building on R1-style distilled models, we observe that inserting a simple ellipsis ("...") into the prompt can stochastically trigger either a thinking or no-thinking mode, revealing a latent controllability in the reasoning behavior. Leveraging this property, we propose AutoThink, a multi-stage reinforcement learning (RL) framework that progressively optimizes reasoning policies via stage-wise reward shaping. AutoThink learns to invoke explicit reasoning only when necessary, while defaulting to succinct responses for simpler tasks. Experiments on five mainstream mathematical benchmarks demonstrate that AutoThink achieves favorable accuracy-efficiency trade-offs compared to recent prompting and RL-based pruning methods. It can be seamlessly integrated into any R1-style model, including both distilled and further fine-tuned variants. Notably, AutoThink improves relative accuracy by 6.4 percent while reducing token usage by 52 percent on DeepSeek-R1-Distill-Qwen-1.5B, establishing a scalable and adaptive reasoning paradigm for LRMs. 

**Abstract (ZH)**: 具有适应性思考能力的大规模推理模型：AutoThink及其在主流数学基准上的应用 

---
# Creating General User Models from Computer Use 

**Title (ZH)**: 从计算机使用中创建通用用户模型 

**Authors**: Omar Shaikh, Shardul Sapkota, Shan Rizvi, Eric Horvitz, Joon Sung Park, Diyi Yang, Michael S. Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.10831)  

**Abstract**: Human-computer interaction has long imagined technology that understands us-from our preferences and habits, to the timing and purpose of our everyday actions. Yet current user models remain fragmented, narrowly tailored to specific apps, and incapable of the flexible reasoning required to fulfill these visions. This paper presents an architecture for a general user model (GUM) that learns about you by observing any interaction you have with your computer. The GUM takes as input any unstructured observation of a user (e.g., device screenshots) and constructs confidence-weighted propositions that capture that user knowledge and preferences. GUMs can infer that a user is preparing for a wedding they're attending from messages with a friend. Or recognize that a user is struggling with a collaborator's feedback on a draft by observing multiple stalled edits and a switch to reading related work. GUMs introduce an architecture that infers new propositions about a user from multimodal observations, retrieves related propositions for context, and continuously revises existing propositions. To illustrate the breadth of applications that GUMs enable, we demonstrate how they augment chat-based assistants with context, manage OS notifications to selectively surface important information, and enable interactive agents that adapt to preferences across apps. We also instantiate proactive assistants (GUMBOs) that discover and execute useful suggestions on a user's behalf using their GUM. In our evaluations, we find that GUMs make calibrated and accurate inferences about users, and that assistants built on GUMs proactively identify and perform actions that users wouldn't think to request explicitly. Altogether, GUMs introduce methods that leverage multimodal models to understand unstructured context, enabling long-standing visions of HCI and entirely new interactive systems that anticipate user needs. 

**Abstract (ZH)**: 人类计算机交互：一种通用用户模型的架构及其应用 

---
# Enhancing Low-Resource Minority Language Translation with LLMs and Retrieval-Augmented Generation for Cultural Nuances 

**Title (ZH)**: 利用大型语言模型和检索增强生成强化低资源 Minority 语言翻译中的文化细微差异 

**Authors**: Chen-Chi Chang, Chong-Fu Li, Chu-Hsuan Lee, Hung-Shin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.10829)  

**Abstract**: This study investigates the challenges of translating low-resource languages by integrating Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG). Various model configurations were tested on Hakka translations, with BLEU scores ranging from 12% (dictionary-only) to 31% (RAG with Gemini 2.0). The best-performing model (Model 4) combined retrieval and advanced language modeling, improving lexical coverage, particularly for specialized or culturally nuanced terms, and enhancing grammatical coherence. A two-stage method (Model 3) using dictionary outputs refined by Gemini 2.0 achieved a BLEU score of 26%, highlighting iterative correction's value and the challenges of domain-specific expressions. Static dictionary-based approaches struggled with context-sensitive content, demonstrating the limitations of relying solely on predefined resources. These results emphasize the need for curated resources, domain knowledge, and ethical collaboration with local communities, offering a framework that improves translation accuracy and fluency while supporting cultural preservation. 

**Abstract (ZH)**: 本研究通过将大型语言模型（LLMs）与检索增强生成（RAG）相结合，探究低资源语言翻译的挑战。各种模型配置在客家语翻译上进行了测试，BLEU得分范围从仅使用词典的12%到使用Gemini 2.0的RAG的31%。表现最好的模型（模型4）结合了检索和高级语言建模，改善了专门词汇或文化特定术语的词汇覆盖面，并增强了语法规则的连贯性。使用Gemini 2.0优化字典输出的两阶段方法（模型3）实现了26%的BLEU得分，突显了迭代修正的价值及领域特定表达的挑战。静态基于词典的方法在处理语境敏感内容时遇到困难，表明仅依赖预定义资源的局限性。这些结果强调需要精心整理的资源、领域知识以及与当地社区的伦理合作，为提高翻译准确性和流畅性、支持文化保存提供了一个框架。 

---
# Attention-Based Reward Shaping for Sparse and Delayed Rewards 

**Title (ZH)**: 基于注意力的奖励塑造：针对稀疏和延迟奖励 

**Authors**: Ian Holmes, Min Chi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10802)  

**Abstract**: Sparse and delayed reward functions pose a significant obstacle for real-world Reinforcement Learning (RL) applications. In this work, we propose Attention-based REward Shaping (ARES), a general and robust algorithm which uses a transformer's attention mechanism to generate shaped rewards and create a dense reward function for any environment. ARES requires a set of episodes and their final returns as input. It can be trained entirely offline and is able to generate meaningful shaped rewards even when using small datasets or episodes produced by agents taking random actions. ARES is compatible with any RL algorithm and can handle any level of reward sparsity. In our experiments, we focus on the most challenging case where rewards are fully delayed until the end of each episode. We evaluate ARES across a diverse range of environments, widely used RL algorithms, and baseline methods to assess the effectiveness of the shaped rewards it produces. Our results show that ARES can significantly improve learning in delayed reward settings, enabling RL agents to train in scenarios that would otherwise require impractical amounts of data or even be unlearnable. To our knowledge, ARES is the first approach that works fully offline, remains robust to extreme reward delays and low-quality data, and is not limited to goal-based tasks. 

**Abstract (ZH)**: 基于注意力的奖励塑形（ARES）：一种通用且 robust 的算法 

---
# Analyzing Patterns and Influence of Advertising in Print Newspapers 

**Title (ZH)**: 分析印刷报纸中广告的模式与影响 

**Authors**: N Harsha Vardhan, Ponnurangam Kumaraguru, Kiran Garimella  

**Link**: [PDF](https://arxiv.org/pdf/2505.10791)  

**Abstract**: This paper investigates advertising practices in print newspapers across India using a novel data-driven approach. We develop a pipeline employing image processing and OCR techniques to extract articles and advertisements from digital versions of print newspapers with high accuracy. Applying this methodology to five popular newspapers that span multiple regions and three languages, English, Hindi, and Telugu, we assembled a dataset of more than 12,000 editions containing several hundred thousand advertisements. Collectively, these newspapers reach a readership of over 100 million people. Using this extensive dataset, we conduct a comprehensive analysis to answer key questions about print advertising: who advertises, what they advertise, when they advertise, where they place their ads, and how they advertise. Our findings reveal significant patterns, including the consistent level of print advertising over the past six years despite declining print circulation, the overrepresentation of company ads on prominent pages, and the disproportionate revenue contributed by government ads. Furthermore, we examine whether advertising in a newspaper influences the coverage an advertiser receives. Through regression analyses on coverage volume and sentiment, we find strong evidence supporting this hypothesis for corporate advertisers. The results indicate a clear trend where increased advertising correlates with more favorable and extensive media coverage, a relationship that remains robust over time and across different levels of advertiser popularity. 

**Abstract (ZH)**: 本文采用一种新颖的数据驱动方法研究印度印刷报纸的广告实践。我们开发了一条管线，利用图像处理和OCR技术从印刷报纸的数字化版本中高精度地提取文章和广告。将这种方法应用到五种流行的横跨多个区域和三种语言（英语、印地语和泰卢固语）的报纸上，我们构建了一个包含数十万条广告的超过12,000个版面的数据库。这些报纸的读者人数超过1亿。使用这个庞大的数据库，我们开展全面分析以回答关于印刷广告的关键问题：广告商是谁，他们广告的内容是什么，何时进行广告宣传，广告位置在何处，以及他们如何进行广告宣传。我们的发现揭示了显著的模式，包括过去六年印刷广告的一贯水平尽管印刷发行量下降，公司广告在显要版面的过度代表性以及政府广告对收入的不成比例贡献。此外，我们还探讨了报纸的广告是否会影响广告商获得的报道。通过回归分析报道量和情感指标，我们发现对于企业广告商而言，存在有力证据支持这一假设。结果显示，广告增加与更正面和更广泛的媒体覆盖之间存在明确趋势，这种关系在时间上和不同广告商受欢迎程度的层次上都保持稳健。 

---
# Neural-Inspired Advances in Integral Cryptanalysis 

**Title (ZH)**: 神经启发的积分攻击进展 

**Authors**: Liu Zhang, Yiran Yao, Danping Shi, Dongchen Chai, Jian Guo, Zilong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10790)  

**Abstract**: The study by Gohr this http URL at CRYPTO 2019 and sunsequent related works have shown that neural networks can uncover previously unused features, offering novel insights into cryptanalysis. Motivated by these findings, we employ neural networks to learn features specifically related to integral properties and integrate the corresponding insights into optimized search frameworks. These findings validate the framework of using neural networks for feature exploration, providing researchers with novel insights that advance established cryptanalysis methods.
Neural networks have inspired the development of more precise integral search models. By comparing the integral distinguishers obtained via neural networks with those identified by classical methods, we observe that existing automated search models often fail to find optimal distinguishers. To address this issue, we develop a meet in the middle search framework that balances model accuracy and computational efficiency. As a result, we reduce the number of active plaintext bits required for an 11 rounds integral distinguisher on SKINNY64/64, and further identify a 12 rounds key dependent integral distinguisher achieving one additional round over the previous best-known result.
The integral distinguishers discovered by neural networks enable key recovery attacks on more rounds. We identify a 7 rounds key independent integral distinguisher from neural networks with even only one active plaintext cell, which is based on linear combinations of bits. This distinguisher enables a 15 rounds key recovery attack on SKINNYn/n, improving upon the previous record by one round. Additionally, we discover an 8 rounds key dependent integral distinguisher using neural network that further reduces the time complexity of key recovery attacks against SKINNY. 

**Abstract (ZH)**: 神经网络在CRYPTO 2019及后续相关工作中的研究表明，神经网络可以发掘未被使用的特征，为密码分析提供新的见解。受这些发现的启发，我们利用神经网络学习与整体性质相关的特点，并将相应的见解整合到优化的搜索框架中。这些发现验证了使用神经网络进行特征探索的框架的有效性，为研究人员提供了新的见解，推动了现有的密码分析方法的发展。 

---
# Completely Weakly Supervised Class-Incremental Learning for Semantic Segmentation 

**Title (ZH)**: 完全弱监督类增量学习用于语义分割 

**Authors**: David Minkwan Kim, Soeun Lee, Byeongkeun Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10781)  

**Abstract**: This work addresses the task of completely weakly supervised class-incremental learning for semantic segmentation to learn segmentation for both base and additional novel classes using only image-level labels. While class-incremental semantic segmentation (CISS) is crucial for handling diverse and newly emerging objects in the real world, traditional CISS methods require expensive pixel-level annotations for training. To overcome this limitation, partially weakly-supervised approaches have recently been proposed. However, to the best of our knowledge, this is the first work to introduce a completely weakly-supervised method for CISS. To achieve this, we propose to generate robust pseudo-labels by combining pseudo-labels from a localizer and a sequence of foundation models based on their uncertainty. Moreover, to mitigate catastrophic forgetting, we introduce an exemplar-guided data augmentation method that generates diverse images containing both previous and novel classes with guidance. Finally, we conduct experiments in three common experimental settings: 15-5 VOC, 10-10 VOC, and COCO-to-VOC, and in two scenarios: disjoint and overlap. The experimental results demonstrate that our completely weakly supervised method outperforms even partially weakly supervised methods in the 15-5 VOC and 10-10 VOC settings while achieving competitive accuracy in the COCO-to-VOC setting. 

**Abstract (ZH)**: 本研究解决了完全弱监督类增量学习在语义分割中的问题，仅使用图层面标签来学习基类和新增类的分割。虽然类增量语义分割（CISS）对于处理现实世界中多种新兴对象至关重要，但传统CISS方法需要昂贵的像素级标注进行训练。为了克服这一局限性，最近提出了部分弱监督方法。然而，据我们所知，这是首次提出完全弱监督方法用于CISS。为了实现这一点，我们提出了一种结合局部化器和一系列基于不确定性基础模型的伪标签生成稳健伪标签的方法。此外，为缓解灾难性遗忘，我们引入了一种基于示例的数据增强方法，该方法在指导下生成包含以前和新增类的多样化图像。最后，我们在三个常见实验设置（15-5 VOC、10-10 VOC和COCO-to-VOC）和两种场景（分离和重叠）下进行了实验。实验结果表明，在15-5 VOC和10-10 VOC设置中，我们的完全弱监督方法在性能上优于部分弱监督方法，在COCO-to-VOC设置中，我们的方法达到了可竞争的准确率。 

---
# A Systematic Analysis of Base Model Choice for Reward Modeling 

**Title (ZH)**: 基模型选择的系统分析用于奖励建模 

**Authors**: Kian Ahrabian, Pegah Jandaghi, Negar Mokhberian, Sai Praneeth Karimireddy, Jay Pujara  

**Link**: [PDF](https://arxiv.org/pdf/2505.10775)  

**Abstract**: Reinforcement learning from human feedback (RLHF) and, at its core, reward modeling have become a crucial part of training powerful large language models (LLMs). One commonly overlooked factor in training high-quality reward models (RMs) is the effect of the base model, which is becoming more challenging to choose given the rapidly growing pool of LLMs. In this work, we present a systematic analysis of the effect of base model selection on reward modeling performance. Our results show that the performance can be improved by up to 14% compared to the most common (i.e., default) choice. Moreover, we showcase the strong statistical relation between some existing benchmarks and downstream performances. We also demonstrate that the results from a small set of benchmarks could be combined to boost the model selection ($+$18% on average in the top 5-10). Lastly, we illustrate the impact of different post-training steps on the final performance and explore using estimated data distributions to reduce performance prediction error. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）和核心中的奖励建模已成为训练强大语言模型（LLMs）的关键部分。训练高质量奖励模型（RMs）时一个常被忽视的因素是基础模型的影响，鉴于不断增长的LLM池，选择基础模型变得愈发具有挑战性。在本工作中，我们系统分析了基础模型选择对奖励建模性能的影响。结果显示，与最常用的选择（即默认选择）相比，性能可提升多达14%。此外，我们展示了某些现有基准与下游性能之间的强统计关系。我们还证明，通过将少量基准的结果综合起来，可以提升模型选择（平均在前5-10位中提升18%）。最后，我们展示了不同后训练步骤对最终性能的影响，并探索使用估计的数据分布来减少性能预测误差。 

---
# Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting 

**Title (ZH)**: 基于上下文的概率建模在多模态时间序列预测中的应用（利用LLM） 

**Authors**: Yueyang Yao, Jiajun Li, Xingyuan Dai, MengMeng Zhang, Xiaoyan Gong, Fei-Yue Wang, Yisheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2505.10774)  

**Abstract**: Time series forecasting is important for applications spanning energy markets, climate analysis, and traffic management. However, existing methods struggle to effectively integrate exogenous texts and align them with the probabilistic nature of large language models (LLMs). Current approaches either employ shallow text-time series fusion via basic prompts or rely on deterministic numerical decoding that conflict with LLMs' token-generation paradigm, which limits contextual awareness and distribution modeling. To address these limitations, we propose CAPTime, a context-aware probabilistic multimodal time series forecasting method that leverages text-informed abstraction and autoregressive LLM decoding. Our method first encodes temporal patterns using a pretrained time series encoder, then aligns them with textual contexts via learnable interactions to produce joint multimodal representations. By combining a mixture of distribution experts with frozen LLMs, we enable context-aware probabilistic forecasting while preserving LLMs' inherent distribution modeling capabilities. Experiments on diverse time series forecasting tasks demonstrate the superior accuracy and generalization of CAPTime, particularly in multimodal scenarios. Additional analysis highlights its robustness in data-scarce scenarios through hybrid probabilistic decoding. 

**Abstract (ZH)**: 基于文本的上下文感知概率多模态时间序列Forecasting方法：CAPTime 

---
# Geofenced Unmanned Aerial Robotic Defender for Deer Detection and Deterrence (GUARD) 

**Title (ZH)**: fences区域内无人机机器人守卫系统：鹿的检测与威慑（GUARD） 

**Authors**: Ebasa Temesgen, Mario Jerez, Greta Brown, Graham Wilson, Sree Ganesh Lalitaditya Divakarla, Sarah Boelter, Oscar Nelson, Robert McPherson, Maria Gini  

**Link**: [PDF](https://arxiv.org/pdf/2505.10770)  

**Abstract**: Wildlife-induced crop damage, particularly from deer, threatens agricultural productivity. Traditional deterrence methods often fall short in scalability, responsiveness, and adaptability to diverse farmland environments. This paper presents an integrated unmanned aerial vehicle (UAV) system designed for autonomous wildlife deterrence, developed as part of the Farm Robotics Challenge. Our system combines a YOLO-based real-time computer vision module for deer detection, an energy-efficient coverage path planning algorithm for efficient field monitoring, and an autonomous charging station for continuous operation of the UAV. In collaboration with a local Minnesota farmer, the system is tailored to address practical constraints such as terrain, infrastructure limitations, and animal behavior. The solution is evaluated through a combination of simulation and field testing, demonstrating robust detection accuracy, efficient coverage, and extended operational time. The results highlight the feasibility and effectiveness of drone-based wildlife deterrence in precision agriculture, offering a scalable framework for future deployment and extension. 

**Abstract (ZH)**: 野生动物引起的作物损坏，尤其是鹿的侵害，威胁着农业生产力。传统驱避方法在 scalability、响应性和适应性方面往往不足。本文介绍了为自主野生动物驱避设计的集成无人机系统，该系统作为Farm Robotics Challenge项目的一部分开发。该系统结合了基于YOLO的实时计算机视觉模块进行鹿的检测、高效的覆盖路径规划算法以实现有效的田间监测以及自主充电站以实现无人机的连续运行。与明尼苏达当地农民合作，该系统针对地形、基础设施限制和动物行为等实际约束进行了定制。该解决方案通过仿真和实地测试进行评估，展示了稳健的检测精度、高效的覆盖能力和延长的操作时间。结果突显了基于无人机的野生动物驱避在精准农业中的可行性和有效性，提供了一个可扩展的框架，适用于未来部署和扩展。 

---
# ChestyBot: Detecting and Disrupting Chinese Communist Party Influence Stratagems 

**Title (ZH)**: ChestyBot: 识别和遏制中国共产党影响力策略 

**Authors**: Matthew Stoffolano, Ayush Rout, Justin M. Pelletier  

**Link**: [PDF](https://arxiv.org/pdf/2505.10746)  

**Abstract**: Foreign information operations conducted by Russian and Chinese actors exploit the United States' permissive information environment. These campaigns threaten democratic institutions and the broader Westphalian model. Yet, existing detection and mitigation strategies often fail to identify active information campaigns in real time. This paper introduces ChestyBot, a pragmatics-based language model that detects unlabeled foreign malign influence tweets with up to 98.34% accuracy. The model supports a novel framework to disrupt foreign influence operations in their formative stages. 

**Abstract (ZH)**: 俄罗斯和中国行为体开展的外国信息操作利用了美国宽松的信息环境，威胁着民主制度和更广泛的威斯特伐利亚模型。现有检测和缓解策略往往无法实时识别活跃的信息campaign。本文介绍了一种基于语用学的语言模型ChestyBot，该模型以高达98.34%的准确率检测未标记的外国恶意影响推文，并支持一种在形成阶段削弱外国影响操作的新框架。 

---
# Automating Security Audit Using Large Language Model based Agent: An Exploration Experiment 

**Title (ZH)**: 基于大规模语言模型的代理自动化安全审计：一项探索性实验 

**Authors**: Jia Hui Chin, Pu Zhang, Yu Xin Cheong, Jonathan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10732)  

**Abstract**: In the current rapidly changing digital environment, businesses are under constant stress to ensure that their systems are secured. Security audits help to maintain a strong security posture by ensuring that policies are in place, controls are implemented, gaps are identified for cybersecurity risks mitigation. However, audits are usually manual, requiring much time and costs. This paper looks at the possibility of developing a framework to leverage Large Language Models (LLMs) as an autonomous agent to execute part of the security audit, namely with the field audit. password policy compliance for Windows operating system. Through the conduct of an exploration experiment of using GPT-4 with Langchain, the agent executed the audit tasks by accurately flagging password policy violations and appeared to be more efficient than traditional manual audits. Despite its potential limitations in operational consistency in complex and dynamic environment, the framework suggests possibilities to extend further to real-time threat monitoring and compliance checks. 

**Abstract (ZH)**: 当前快速变化的数字环境中，企业持续面临确保其系统安全的压力。安全审计有助于通过确保政策到位、控制实施、识别漏洞来维持强大的安全态势。然而，审计通常需要手动进行，耗时且成本高。本文探讨了利用大型语言模型（LLMs）作为自主代理执行部分安全审计的可能性，特别是在Windows操作系统领域的密码策略合规审计。通过使用GPT-4与Langchain进行探索性实验，该代理能够准确标记密码策略违规，并显示出比传统手动审计更高效。尽管在复杂和动态环境中操作一致性方面存在潜在限制，该框架仍暗示了进一步扩展至实时威胁监控和合规检查的可能性。 

---
# Learning Repetition-Invariant Representations for Polymer Informatics 

**Title (ZH)**: 学习不变重复聚合物表示方法 

**Authors**: Yihan Zhu, Gang Liu, Eric Inae, Tengfei Luo, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10726)  

**Abstract**: Polymers are large macromolecules composed of repeating structural units known as monomers and are widely applied in fields such as energy storage, construction, medicine, and aerospace. However, existing graph neural network methods, though effective for small molecules, only model the single unit of polymers and fail to produce consistent vector representations for the true polymer structure with varying numbers of units. To address this challenge, we introduce Graph Repetition Invariance (GRIN), a novel method to learn polymer representations that are invariant to the number of repeating units in their graph representations. GRIN integrates a graph-based maximum spanning tree alignment with repeat-unit augmentation to ensure structural consistency. We provide theoretical guarantees for repetition-invariance from both model and data perspectives, demonstrating that three repeating units are the minimal augmentation required for optimal invariant representation learning. GRIN outperforms state-of-the-art baselines on both homopolymer and copolymer benchmarks, learning stable, repetition-invariant representations that generalize effectively to polymer chains of unseen sizes. 

**Abstract (ZH)**: 聚合物是由重复结构单元（单体）组成的大型高分子，广泛应用于储能、建筑、医疗和航天等领域。然而，现有的图神经网络方法虽对小分子有效，但仅能建模聚合物的单个单位，无法为具有不同单位数目的真实聚合物结构生成一致的向量表示。为解决这一挑战，我们提出了图重复不变性（GRIN）方法，该方法能够学习对图表示中重复单元数量不变的聚合物表示。GRIN 综合了基于图的最大生成树对齐和重复单元增强，以确保结构一致性。我们从模型和数据两个角度提供了重复不变性的理论保证，证明了三个重复单元是最小的增强需求，以实现最佳不变表示学习。GRIN 在同聚物和共聚物基准测试中均优于现有最佳baseline，学习到稳定且对重复不变的表示，能够有效泛化到未见过尺寸的聚合物链上。 

---
# AI-enhanced semantic feature norms for 786 concepts 

**Title (ZH)**: AI增强的语义特征规范for 786概念 

**Authors**: Siddharth Suresh, Kushin Mukherjee, Tyler Giallanza, Xizheng Yu, Mia Patil, Jonathan D. Cohen, Timothy T. Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2505.10718)  

**Abstract**: Semantic feature norms have been foundational in the study of human conceptual knowledge, yet traditional methods face trade-offs between concept/feature coverage and verifiability of quality due to the labor-intensive nature of norming studies. Here, we introduce a novel approach that augments a dataset of human-generated feature norms with responses from large language models (LLMs) while verifying the quality of norms against reliable human judgments. We find that our AI-enhanced feature norm dataset, NOVA: Norms Optimized Via AI, shows much higher feature density and overlap among concepts while outperforming a comparable human-only norm dataset and word-embedding models in predicting people's semantic similarity judgments. Taken together, we demonstrate that human conceptual knowledge is richer than captured in previous norm datasets and show that, with proper validation, LLMs can serve as powerful tools for cognitive science research. 

**Abstract (ZH)**: AI增强的语义特征规范数据集NOVA：通过AI优化的概念特征规范，在预测人类语义相似性判断方面优于仅包含人类生成的特征规范数据集和词嵌入模型，并展示了人类概念知识的丰富性以及验证条件下大语言模型在认知科学研究中的强大工具作用。 

---
# A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment 

**Title (ZH)**: 基于预制指令调优、模型融合和临床任务对齐的合成数据驱动的临床SLMs模块化方法 

**Authors**: Jean-Philippe Corbeil, Amin Dada, Jean-Michel Attendu, Asma Ben Abacha, Alessandro Sordoni, Lucas Caccia, François Beaulieu, Thomas Lin, Jens Kleesiek, Paul Vozila  

**Link**: [PDF](https://arxiv.org/pdf/2505.10717)  

**Abstract**: High computation costs and latency of large language models such as GPT-4 have limited their deployment in clinical settings. Small language models (SLMs) offer a cost-effective alternative, but their limited capacity requires biomedical domain adaptation, which remains challenging. An additional bottleneck is the unavailability and high sensitivity of clinical data. To address these challenges, we propose a novel framework for adapting SLMs into high-performing clinical models. We introduce the MediPhi collection of 3.8B-parameter SLMs developed with our novel framework: pre-instruction tuning of experts on relevant medical and clinical corpora (PMC, Medical Guideline, MedWiki, etc.), model merging, and clinical-tasks alignment. To cover most clinical tasks, we extended the CLUE benchmark to CLUE+, doubling its size. Our expert models deliver relative improvements on this benchmark over the base model without any task-specific fine-tuning: 64.3% on medical entities, 49.5% on radiology reports, and 44% on ICD-10 coding (outperforming GPT-4-0125 by 14%). We unify the expert models into MediPhi via model merging, preserving gains across benchmarks. Furthermore, we built the MediFlow collection, a synthetic dataset of 2.5 million high-quality instructions on 14 medical NLP tasks, 98 fine-grained document types, and JSON format support. Alignment of MediPhi using supervised fine-tuning and direct preference optimization achieves further gains of 18.9% on average. 

**Abstract (ZH)**: 一种新型框架：将小型语言模型adapt化为高性能临床模型 

---
# GNN-Suite: a Graph Neural Network Benchmarking Framework for Biomedical Informatics 

**Title (ZH)**: GNN-Suite：生物医学informatics领域图神经网络基准测试框架 

**Authors**: Sebestyén Kamp, Giovanni Stracquadanio, T. Ian Simpson  

**Link**: [PDF](https://arxiv.org/pdf/2505.10711)  

**Abstract**: We present GNN-Suite, a robust modular framework for constructing and benchmarking Graph Neural Network (GNN) architectures in computational biology. GNN-Suite standardises experimentation and reproducibility using the Nextflow workflow to evaluate GNN performance. We demonstrate its utility in identifying cancer-driver genes by constructing molecular networks from protein-protein interaction (PPI) data from STRING and BioGRID and annotating nodes with features from the PCAWG, PID, and COSMIC-CGC repositories.
Our design enables fair comparisons among diverse GNN architectures including GAT, GAT3H, GCN, GCN2, GIN, GTN, HGCN, PHGCN, and GraphSAGE and a baseline Logistic Regression (LR) model. All GNNs were configured as standardised two-layer models and trained with uniform hyperparameters (dropout = 0.2; Adam optimiser with learning rate = 0.01; and an adjusted binary cross-entropy loss to address class imbalance) over an 80/20 train-test split for 300 epochs. Each model was evaluated over 10 independent runs with different random seeds to yield statistically robust performance metrics, with balanced accuracy (BACC) as the primary measure. Notably, GCN2 achieved the highest BACC (0.807 +/- 0.035) on a STRING-based network, although all GNN types outperformed the LR baseline, highlighting the advantage of network-based learning over feature-only approaches.
Our results show that a common framework for implementing and evaluating GNN architectures aids in identifying not only the best model but also the most effective means of incorporating complementary data. By making GNN-Suite publicly available, we aim to foster reproducible research and promote improved benchmarking standards in computational biology. Future work will explore additional omics datasets and further refine network architectures to enhance predictive accuracy and interpretability in biomedical applications. 

**Abstract (ZH)**: 一种用于计算生物学中图神经网络架构构建与基准测试的稳健模块化框架：GNN-Suite及其在识别癌症驱动基因中的应用 

---
# Predicting Human Behavior in Autonomous Systems: A Collaborative Machine Teaching Approach for Reducing Transfer of Control Events 

**Title (ZH)**: 自主系统中人类行为预测：一种减少控制权转移事件的协同机器教学方法 

**Authors**: Julian Wolter, Amr Gomaa  

**Link**: [PDF](https://arxiv.org/pdf/2505.10695)  

**Abstract**: As autonomous systems become integral to various industries, effective strategies for fault handling are essential to ensure reliability and efficiency. Transfer of Control (ToC), a traditional approach for interrupting automated processes during faults, is often triggered unnecessarily in non-critical situations. To address this, we propose a data-driven method that uses human interaction data to train AI models capable of preemptively identifying and addressing issues or assisting users in resolution. Using an interactive tool simulating an industrial vacuum cleaner, we collected data and developed an LSTM-based model to predict user behavior. Our findings reveal that even data from non-experts can effectively train models to reduce unnecessary ToC events, enhancing the system's robustness. This approach highlights the potential of AI to learn directly from human problem-solving behaviors, complementing sensor data to improve industrial automation and human-AI collaboration. 

**Abstract (ZH)**: 随着自主系统在各个行业中的应用日益广泛，有效的故障处理策略对于确保可靠性和效率至关重要。转移控制（ToC），一种传统的在故障期间中断自动化流程的方法，往往在非关键情况下被不必要的触发。为了解决这一问题，我们提出了一种基于数据的方法，利用人类交互数据训练AI模型，以预先识别和解决问题，或在用户遇到问题时提供协助。我们使用一个模拟工业吸尘器的交互工具来收集数据，并开发了一个基于LSTM的模型来预测用户行为。研究结果表明，即使是非专家的数据也能有效训练模型以减少不必要的ToC事件，提高系统的鲁棒性。该方法突显了AI直接从人类问题解决行为中学习的潜力，从而补充传感器数据，提高工业自动化和人机协作的性能。 

---
# Predicting Risk of Pulmonary Fibrosis Formation in PASC Patients 

**Title (ZH)**: PASC患者肺纤维化形成风险的预测 

**Authors**: Wanying Dou, Gorkem Durak, Koushik Biswas, Ziliang Hong, Andrea Mia Bejar, Elif Keles, Kaan Akin, Sukru Mehmet Erturk, Alpay Medetalibeyoglu, Marc Sala, Alexander Misharin, Hatice Savas, Mary Salvatore, Sachin Jambawalikar, Drew Torigian, Jayaram K. Udupa, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2505.10691)  

**Abstract**: While the acute phase of the COVID-19 pandemic has subsided, its long-term effects persist through Post-Acute Sequelae of COVID-19 (PASC), commonly known as Long COVID. There remains substantial uncertainty regarding both its duration and optimal management strategies. PASC manifests as a diverse array of persistent or newly emerging symptoms--ranging from fatigue, dyspnea, and neurologic impairments (e.g., brain fog), to cardiovascular, pulmonary, and musculoskeletal abnormalities--that extend beyond the acute infection phase. This heterogeneous presentation poses substantial challenges for clinical assessment, diagnosis, and treatment planning. In this paper, we focus on imaging findings that may suggest fibrotic damage in the lungs, a critical manifestation characterized by scarring of lung tissue, which can potentially affect long-term respiratory function in patients with PASC. This study introduces a novel multi-center chest CT analysis framework that combines deep learning and radiomics for fibrosis prediction. Our approach leverages convolutional neural networks (CNNs) and interpretable feature extraction, achieving 82.2% accuracy and 85.5% AUC in classification tasks. We demonstrate the effectiveness of Grad-CAM visualization and radiomics-based feature analysis in providing clinically relevant insights for PASC-related lung fibrosis prediction. Our findings highlight the potential of deep learning-driven computational methods for early detection and risk assessment of PASC-related lung fibrosis--presented for the first time in the literature. 

**Abstract (ZH)**: 尽管COVID-19急性期已过去，其长期影响通过新冠后遗症（PASC）或俗称“长 COVID”持续存在。关于其持续时间和最佳管理策略仍存在大量不确定性。PASC 表现为一系列持续或新出现的症状——从疲劳、呼吸困难和神经系统损害（如脑雾）到心血管、肺部和肌肉骨骼异常——这些症状超出了急性感染期。这种异质性表现给临床评估、诊断和治疗规划带来了巨大挑战。本文聚焦于影像学发现，这些发现可能表明肺纤维化损伤，这是一种关键表现，特点是肺组织疤痕化，可能影响PASC患者的长期呼吸功能。本研究引入了一种结合深度学习和 Radiomics 的多中心胸部CT分析框架，用于纤维化预测。我们的方法利用卷积神经网络（CNNs）和可解释的特征提取，分类任务的准确率为82.2%，AUC为85.5%。我们展示了Grad-CAM可视化和基于Radiomics的特征分析在提供与PASC相关的肺纤维化预测的临床相关见解方面的有效性。我们的研究结果突出了基于深度学习的计算方法在PASC相关肺纤维化的早期检测和风险评估中的潜力——这是首次在文献中提出。 

---
# Towards an LLM-powered Social Digital Twinning Platform 

**Title (ZH)**: 面向LLM驱动的社会数字孪生平台 

**Authors**: Önder Gürcan, Vanja Falck, Markus G. Rousseau, Larissa L. Lima  

**Link**: [PDF](https://arxiv.org/pdf/2505.10681)  

**Abstract**: We present Social Digital Twinner, an innovative social simulation tool for exploring plausible effects of what-if scenarios in complex adaptive social systems. The architecture is composed of three seamlessly integrated parts: a data infrastructure featuring real-world data and a multi-dimensionally representative synthetic population of citizens, an LLM-enabled agent-based simulation engine, and a user interface that enable intuitive, natural language interactions with the simulation engine and the artificial agents (i.e. citizens). Social Digital Twinner facilitates real-time engagement and empowers stakeholders to collaboratively design, test, and refine intervention measures. The approach is promoting a data-driven and evidence-based approach to societal problem-solving. We demonstrate the tool's interactive capabilities by addressing the critical issue of youth school dropouts in Kragero, Norway, showcasing its ability to create and execute a dedicated social digital twin using natural language. 

**Abstract (ZH)**: 社会数字孪生体：一种探索复杂自适应社会系统中“如果-那么”情景潜在影响的创新社会仿真工具 

---
# A Conformal Predictive Measure for Assessing Catastrophic Forgetting 

**Title (ZH)**: 用于评估灾难性遗忘的配准预测衡量指标 

**Authors**: Ioannis Pitsiorlas, Nour Jamoussi, Marios Kountouris  

**Link**: [PDF](https://arxiv.org/pdf/2505.10677)  

**Abstract**: This work introduces a novel methodology for assessing catastrophic forgetting (CF) in continual learning. We propose a new conformal prediction (CP)-based metric, termed the Conformal Prediction Confidence Factor (CPCF), to quantify and evaluate CF effectively. Our framework leverages adaptive CP to estimate forgetting by monitoring the model's confidence on previously learned tasks. This approach provides a dynamic and practical solution for monitoring and measuring CF of previous tasks as new ones are introduced, offering greater suitability for real-world applications. Experimental results on four benchmark datasets demonstrate a strong correlation between CPCF and the accuracy of previous tasks, validating the reliability and interpretability of the proposed metric. Our results highlight the potential of CPCF as a robust and effective tool for assessing and understanding CF in dynamic learning environments. 

**Abstract (ZH)**: 本研究提出了一种新的方法论，用于评估连续学习中的灾难性遗忘（CF）。我们提出了一种基于可信区间（CP）的新度量方法，称为可信区间信心因子（CPCF），以有效量化和评估CF。我们的框架利用自适应CP来通过监控模型对之前学习的任务的信心来估计遗忘。这种 Approach 提供了一种动态且实用的方法来监测和测量随新任务引入而来的之前的任务的CF，使其更适用于实际应用。四项基准数据集上的实验结果表明，CPCF与之前任务的准确性之间存在密切关联，验证了所提出度量的可靠性和可解释性。研究结果突显了CPCF作为评估和理解动态学习环境中CF的稳健而有效的工具的潜力。 

---
# Seasonal Forecasting of Pan-Arctic Sea Ice with State Space Model 

**Title (ZH)**: 北极Pan-Arctic区域海冰季节预报模型研究 

**Authors**: Wei Wang, Weidong Yang, Lei Wang, Guihua Wang, Ruibo Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.10665)  

**Abstract**: The rapid decline of Arctic sea ice resulting from anthropogenic climate change poses significant risks to indigenous communities, ecosystems, and the global climate system. This situation emphasizes the immediate necessity for precise seasonal sea ice forecasts. While dynamical models perform well for short-term forecasts, they encounter limitations in long-term forecasts and are computationally intensive. Deep learning models, while more computationally efficient, often have difficulty managing seasonal variations and uncertainties when dealing with complex sea ice dynamics. In this research, we introduce IceMamba, a deep learning architecture that integrates sophisticated attention mechanisms within the state space model. Through comparative analysis of 25 renowned forecast models, including dynamical, statistical, and deep learning approaches, our experimental results indicate that IceMamba delivers excellent seasonal forecasting capabilities for Pan-Arctic sea ice concentration. Specifically, IceMamba outperforms all tested models regarding average RMSE and anomaly correlation coefficient (ACC) and ranks second in Integrated Ice Edge Error (IIEE). This innovative approach enhances our ability to foresee and alleviate the effects of sea ice variability, offering essential insights for strategies aimed at climate adaptation. 

**Abstract (ZH)**: 北极海冰因人为气候变化的快速减少给土著社区、生态系统和全球气候系统带来了重大风险。这种情况强调了进行精确季节性海冰预报的迫切需求。虽然动力模型在短期预报中表现良好，但在长期预报中存在局限性且计算成本高昂。深度学习模型虽然在计算效率上更具优势，但在处理复杂海冰动力学时往往难以应对季节变化和不确定性。在此研究中，我们引入了IceMamba，这是一种在状态空间模型中集成高级注意机制的深度学习架构。通过对比分析包括动力学、统计学和深度学习方法在内的25个知名预报模型，我们的实验结果表明，IceMamba在北极地区海冰浓度季节性预报方面表现出色。具体而言，IceMamba在平均RMSE和异常相关系数（ACC）方面超过了所有测试模型，在集成冰缘误差（IIEE）方面排名第二。这一创新方法增强了我们预测和缓解海冰变异性影响的能力，为气候适应策略提供了重要见解。 

---
# CLIP Embeddings for AI-Generated Image Detection: A Few-Shot Study with Lightweight Classifier 

**Title (ZH)**: 基于CLIP嵌入的AI生成图像检测：一种轻量级分类器参与的少样本研究 

**Authors**: Ziyang Ou  

**Link**: [PDF](https://arxiv.org/pdf/2505.10664)  

**Abstract**: Verifying the authenticity of AI-generated images presents a growing challenge on social media platforms these days. While vision-language models (VLMs) like CLIP outdo in multimodal representation, their capacity for AI-generated image classification is underexplored due to the absence of such labels during the pre-training process. This work investigates whether CLIP embeddings inherently contain information indicative of AI generation. A proposed pipeline extracts visual embeddings using a frozen CLIP model, feeds its embeddings to lightweight networks, and fine-tunes only the final classifier. Experiments on the public CIFAKE benchmark show the performance reaches 95% accuracy without language reasoning. Few-shot adaptation to curated custom with 20% of the data results in performance to 85%. A closed-source baseline (Gemini-2.0) has the best zero-shot accuracy yet fails on specific styles. Notably, some specific image types, such as wide-angle photographs and oil paintings, pose significant challenges to classification. These results indicate previously unexplored difficulties in classifying certain types of AI-generated images, revealing new and more specific questions in this domain that are worth further investigation. 

**Abstract (ZH)**: 验证AI生成图像的真实性目前在社交媒体平台上越来越成为一个挑战。尽管像CLIP这样的多模态模型在多模态表示方面表现出色，但由于预训练过程中缺乏相应的标签，它们在AI生成图像分类方面的能力尚未得到充分利用。本文探讨了CLIP嵌入是否包含指示AI生成的信息。提出了一种pipeline流程，使用冻结的CLIP模型提取视觉嵌入，将其嵌入传递给轻量级网络，并仅 fine-tune 最终分类器。在公共CIFAKE基准测试上的实验显示，无需语言推理即可达到95%的准确性。利用20%的定制数据进行少样本适应，性能达到85%。公开源码基线（Gemini-2.0）在零样本准确性上表现最佳，但在特定风格上失败。值得注意的是，某些特定图像类型，如广角照片和油画，给分类带来了重大挑战。这些结果表明，在分类某些类型的AI生成图像时存在尚未探索的困难，揭示了值得进一步调查的新且更具体的领域问题。 

---
# Artificial Intelligence Bias on English Language Learners in Automatic Scoring 

**Title (ZH)**: 人工智能偏见对英语学习者自动评分的影响 

**Authors**: Shuchen Guo, Yun Wang, Jichao Yu, Xuansheng Wu, Bilgehan Ayik, Field M. Watts, Ehsan Latif, Ninghao Liu, Lei Liu, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2505.10643)  

**Abstract**: This study investigated potential scoring biases and disparities toward English Language Learners (ELLs) when using automatic scoring systems for middle school students' written responses to science assessments. We specifically focus on examining how unbalanced training data with ELLs contributes to scoring bias and disparities. We fine-tuned BERT with four datasets: responses from (1) ELLs, (2) non-ELLs, (3) a mixed dataset reflecting the real-world proportion of ELLs and non-ELLs (unbalanced), and (4) a balanced mixed dataset with equal representation of both groups. The study analyzed 21 assessment items: 10 items with about 30,000 ELL responses, five items with about 1,000 ELL responses, and six items with about 200 ELL responses. Scoring accuracy (Acc) was calculated and compared to identify bias using Friedman tests. We measured the Mean Score Gaps (MSGs) between ELLs and non-ELLs and then calculated the differences in MSGs generated through both the human and AI models to identify the scoring disparities. We found that no AI bias and distorted disparities between ELLs and non-ELLs were found when the training dataset was large enough (ELL = 30,000 and ELL = 1,000), but concerns could exist if the sample size is limited (ELL = 200). 

**Abstract (ZH)**: 本研究调查了在使用自动评分系统对中学生科学评估书面回答时，英语语言学习者（ELLs）评分偏差和不平等现象的可能性。我们重点关注不平衡训练数据对ELLs评分偏差和不平等的影响。我们使用四个数据集微调了BERT：（1）ELLs的回答，（2）非ELLs的回答，（3）反映实际ELLs和非ELLs比例的不均衡混合数据集，（4）均衡混合数据集，其中两个群体的代表数量相等。研究分析了21项评估项目：10个项目约有30,000份ELL回答，5个项目约有1,000份ELL回答，6个项目约有200份ELL回答。计算评分准确性（Acc）并使用弗里德曼检验进行比较以识别偏差。我们测量了ELLs和非ELLs之间的平均得分差距（MSGs），并计算了通过人类和AI模型产生的MSGs差异，以鉴定评分不平等现象。研究发现，当训练数据集足够大时（ELL = 30,000和ELL = 1,000），未发现AI偏见和扭曲的不平等现象，但如果样本量有限（ELL = 200），可能存在担忧。 

---
# The Hitchhikers Guide to Production-ready Trustworthy Foundation Model powered Software (FMware) 

**Title (ZH)**: 生产就绪可信基础模型驱动软件（FMware）指南 

**Authors**: Kirill Vasilevski, Benjamin Rombaut, Gopi Krishnan Rajbahadur, Gustavo A. Oliva, Keheliya Gallaba, Filipe R. Cogo, Jiahuei, Dayi Lin, Haoxiang Zhang, Bouyan Chen, Kishanthan Thangarajah, Ahmed E. Hassan, Zhen Ming, Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10640)  

**Abstract**: Foundation Models (FMs) such as Large Language Models (LLMs) are reshaping the software industry by enabling FMware, systems that integrate these FMs as core components. In this KDD 2025 tutorial, we present a comprehensive exploration of FMware that combines a curated catalogue of challenges with real-world production concerns. We first discuss the state of research and practice in building FMware. We further examine the difficulties in selecting suitable models, aligning high-quality domain-specific data, engineering robust prompts, and orchestrating autonomous agents. We then address the complex journey from impressive demos to production-ready systems by outlining issues in system testing, optimization, deployment, and integration with legacy software. Drawing on our industrial experience and recent research in the area, we provide actionable insights and a technology roadmap for overcoming these challenges. Attendees will gain practical strategies to enable the creation of trustworthy FMware in the evolving technology landscape. 

**Abstract (ZH)**: 基础模型（FMs）如大规模语言模型（LLMs）正在通过使能FMware重塑软件产业，FMware是将这些FMs作为核心组件的系统。在第25届KDD会议的本次tutorial中，我们将全面探讨FMware，结合精心筛选的挑战目录和实际生产问题。首先，我们将讨论构建FMware的研究和实践状态。进一步探讨选择合适模型、对齐高质量领域特定数据、工程稳健提示以及调度自主代理的困难。然后，我们将通过概述系统测试、优化、部署以及与遗留软件集成中的问题，详细说明从令人印象深刻的演示到生产就绪系统的复杂旅程。基于我们的工业经验和该领域的最新研究，我们将提供应对这些挑战的实际建议和技术路线图。参会者将获得实用策略，以在不断发展的技术环境中实现可信赖的FMware的创建。 

---
# Agent Name Service (ANS): A Universal Directory for Secure AI Agent Discovery and Interoperability 

**Title (ZH)**: 代理名称服务（ANS）：一种安全的AI代理发现和互操作的通用目录 

**Authors**: Ken Huang, Vineeth Sai Narajala, Idan Habler, Akram Sheriff  

**Link**: [PDF](https://arxiv.org/pdf/2505.10609)  

**Abstract**: The proliferation of AI agents requires robust mechanisms for secure discovery. This paper introduces the Agent Name Service (ANS), a novel architecture based on DNS addressing the lack of a public agent discovery framework. ANS provides a protocol-agnostic registry infrastructure that leverages Public Key Infrastructure (PKI) certificates for verifiable agent identity and trust. The architecture features several key innovations: a formalized agent registration and renewal mechanism for lifecycle management; DNS-inspired naming conventions with capability-aware resolution; a modular Protocol Adapter Layer supporting diverse communication standards (A2A, MCP, ACP etc.); and precisely defined algorithms for secure resolution. We implement structured communication using JSON Schema and conduct a comprehensive threat analysis of our proposal. The result is a foundational directory service addressing the core challenges of secured discovery and interaction in multi-agent systems, paving the way for future interoperable, trustworthy, and scalable agent ecosystems. 

**Abstract (ZH)**: AI代理的 proliferations 需要稳健的机制来确保安全发现。本文介绍了代理名称服务（ANS），这是一种新型架构，基于DNS，旨在解决公共代理发现框架缺乏的问题。ANS提供了一种协议无关的注册基础设施，利用公钥基础设施（PKI）证书来验证代理身份和建立信任。该架构包含多项关键创新：正式规定的代理注册和续期机制以管理生命周期；DNS启发式的命名规范具有能力感知的解析；支持各种通信标准（A2A、MCP、ACP等）的模块化协议适配器层；以及定义精确的算法以实现安全解析。我们使用JSON Schema实现结构化通信，并对我们的方案进行全面的安全威胁分析。结果是一种基础目录服务，解决了多代理系统中安全发现和交互的核心挑战，为未来可互操作、可信赖和可扩展的代理生态系统铺平了道路。 

---
# MONAQ: Multi-Objective Neural Architecture Querying for Time-Series Analysis on Resource-Constrained Devices 

**Title (ZH)**: MONAQ: 多目标神经架构查询在资源受限设备上的时间序列分析 

**Authors**: Patara Trirat, Jae-Gil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.10607)  

**Abstract**: The growing use of smartphones and IoT devices necessitates efficient time-series analysis on resource-constrained hardware, which is critical for sensing applications such as human activity recognition and air quality prediction. Recent efforts in hardware-aware neural architecture search (NAS) automate architecture discovery for specific platforms; however, none focus on general time-series analysis with edge deployment. Leveraging the problem-solving and reasoning capabilities of large language models (LLM), we propose MONAQ, a novel framework that reformulates NAS into Multi-Objective Neural Architecture Querying tasks. MONAQ is equipped with multimodal query generation for processing multimodal time-series inputs and hardware constraints, alongside an LLM agent-based multi-objective search to achieve deployment-ready models via code generation. By integrating numerical data, time-series images, and textual descriptions, MONAQ improves an LLM's understanding of time-series data. Experiments on fifteen datasets demonstrate that MONAQ-discovered models outperform both handcrafted models and NAS baselines while being more efficient. 

**Abstract (ZH)**: 智能手机和物联网设备使用量的增长 necessitates 对资源受限硬件进行高效时间序列分析，这对诸如人类活动识别和空气质量预测等感知应用至关重要。近年来，面向硬件感知的神经架构搜索（NAS）自动化特定平台的架构发现；然而，这些努力均未关注边缘部署下的一般时间序列分析。利用大型语言模型（LLM）的问题解决和推理能力，我们提出了MONAQ这一新颖框架，将NAS重新构想为多目标神经架构查询任务。MONAQ具备多模态查询生成功能，用于处理多模态时间序列输入和硬件约束，并通过基于LLM代理的多目标搜索实现准备就緒的模型，借助代码生成完成部署。通过集成数值数据、时间序列图像和文本描述，MONAQ提升了LLM对时间序列数据的理解。在十五个数据集上的实验表明，MONAQ发现的模型在效率更高的同时，相较于手工设计的模型和NAS基线模型表现更优。 

---
# Continuity and Isolation Lead to Doubts or Dilemmas in Large Language Models 

**Title (ZH)**: 连续性与隔离导致大型语言模型的迟疑或困境 

**Authors**: Hector Pasten, Felipe Urrutia, Hector Jimenez, Cristian B. Calderon, Cristóbal Rojas, Alexander Kozachinskiy  

**Link**: [PDF](https://arxiv.org/pdf/2505.10606)  

**Abstract**: Understanding how Transformers work and how they process information is key to the theoretical and empirical advancement of these machines. In this work, we demonstrate the existence of two phenomena in Transformers, namely isolation and continuity. Both of these phenomena hinder Transformers to learn even simple pattern sequences. Isolation expresses that any learnable sequence must be isolated from another learnable sequence, and hence some sequences cannot be learned by a single Transformer at the same time. Continuity entails that an attractor basin forms around a learned sequence, such that any sequence falling in that basin will collapse towards the learned sequence. Here, we mathematically prove these phenomena emerge in all Transformers that use compact positional encoding, and design rigorous experiments, demonstrating that the theoretical limitations we shed light on occur on the practical scale. 

**Abstract (ZH)**: 理解Transformer的工作原理及其信息处理机制对于这些机器的理论和实证进步至关重要。在这项工作中，我们证明了Transformer中存在两种现象，即隔离和连续性。这两种现象妨碍Transformer学习甚至简单的模式序列。隔离表明任何可学习的序列必须与其他可学习的序列隔离，因此某些序列不能同时被单个Transformer学习。连续性意味着围绕学习序列形成一个吸引子盆地，使得落入该盆地的任何序列都会趋同于学习的序列。在这里，我们数学上证明了在使用紧凑位置编码的所有Transformer中都会出现这些现象，并设计严谨的实验，证明我们揭示的理论限制在实际规模上确实存在。 

---
# MIRAGE: A Multi-modal Benchmark for Spatial Perception, Reasoning, and Intelligence 

**Title (ZH)**: MIRAGE：多模态空间感知、推理与智能基准 

**Authors**: Chonghan Liu, Haoran Wang, Felix Henry, Pu Miao, Yajie Zhang, Yu Zhao, Peiran Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10604)  

**Abstract**: Spatial perception and reasoning are core components of human cognition, encompassing object recognition, spatial relational understanding, and dynamic reasoning. Despite progress in computer vision, existing benchmarks reveal significant gaps in models' abilities to accurately recognize object attributes and reason about spatial relationships, both essential for dynamic reasoning. To address these limitations, we propose MIRAGE, a multi-modal benchmark designed to evaluate models' capabilities in Counting (object attribute recognition), Relation (spatial relational reasoning), and Counting with Relation. Through diverse and complex scenarios requiring fine-grained recognition and reasoning, MIRAGE highlights critical limitations in state-of-the-art models, underscoring the need for improved representations and reasoning frameworks. By targeting these foundational abilities, MIRAGE provides a pathway toward spatiotemporal reasoning in future research. 

**Abstract (ZH)**: 空间感知与推理是人类认知的核心组成部分，包括物体识别、空间关系理解以及动态推理。尽管计算机视觉取得了进展，现有基准数据集仍揭示了模型在准确识别物体属性和推理空间关系方面的显著不足，这两者对于动态推理至关重要。为解决这些局限性，我们提出MIRAGE，一个多模态基准数据集，旨在评估模型在计数（物体属性识别）、关系（空间关系推理）以及计数与关系方面的能力。通过涉及精细识别和推理的复杂场景，MIRAGE 突显了先进模型的关键局限性，强调了改进表示和推理框架的必要性。通过针对这些基础能力，MIRAGE 为未来研究向时空推理方向发展提供了途径。 

---
# Toward a Public and Secure Generative AI: A Comparative Analysis of Open and Closed LLMs 

**Title (ZH)**: 向着开放且安全的生成式AI：开源与闭源语言模型的比较分析 

**Authors**: Jorge Machado  

**Link**: [PDF](https://arxiv.org/pdf/2505.10603)  

**Abstract**: Generative artificial intelligence (Gen AI) systems represent a critical technology with far-reaching implications across multiple domains of society. However, their deployment entails a range of risks and challenges that require careful evaluation. To date, there has been a lack of comprehensive, interdisciplinary studies offering a systematic comparison between open-source and proprietary (closed) generative AI systems, particularly regarding their respective advantages and drawbacks. This study aims to: i) critically evaluate and compare the characteristics, opportunities, and challenges of open and closed generative AI models; and ii) propose foundational elements for the development of an Open, Public, and Safe Gen AI framework. As a methodology, we adopted a combined approach that integrates three methods: literature review, critical analysis, and comparative analysis. The proposed framework outlines key dimensions, openness, public governance, and security, as essential pillars for shaping the future of trustworthy and inclusive Gen AI. Our findings reveal that open models offer greater transparency, auditability, and flexibility, enabling independent scrutiny and bias mitigation. In contrast, closed systems often provide better technical support and ease of implementation, but at the cost of unequal access, accountability, and ethical oversight. The research also highlights the importance of multi-stakeholder governance, environmental sustainability, and regulatory frameworks in ensuring responsible development. 

**Abstract (ZH)**: 生成式人工智能（Gen AI）系统代表了具有深远社会影响的关键技术。然而，其部署伴随着一系列风险和挑战，需要谨慎评估。迄今为止，鲜有全面的跨学科研究系统性地比较开源和专有生成式 AI 系统的优缺点。本研究旨在：i) 批判性地评估和比较开源和封闭生成 AI 模型的特点、机遇和挑战；ii) 提出一个开放、公众和安全的 Gen AI 框架的基础要素。作为方法论，我们采用了一种结合了文献综述、批判性分析和比较分析的综合方法。所提出的框架明确了开放性、公众治理和安全性作为构建值得信赖和包容的 Gen AI 未来的关键维度。研究结果表明，开源模型提供了更高的透明度、可审计性和灵活性，便于独立审查和偏见缓解。相比之下，封闭系统通常提供更好的技术支持和实施便利性，但代价是不平等的访问、问责制和道德监督。研究还强调了多利益相关方治理、环境可持续性和监管框架在确保负责任的发展中的重要性。 

---
# Enhancing IoT Cyber Attack Detection in the Presence of Highly Imbalanced Data 

**Title (ZH)**: 在高度不平衡数据存在下的物联网网络攻击检测增强方法 

**Authors**: Md. Ehsanul Haque, Md. Saymon Hosen Polash, Md Al-Imran Sanjida Simla, Md Alomgir Hossain, Sarwar Jahan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10600)  

**Abstract**: Due to the rapid growth in the number of Internet of Things (IoT) networks, the cyber risk has increased exponentially, and therefore, we have to develop effective IDS that can work well with highly imbalanced datasets. A high rate of missed threats can be the result, as traditional machine learning models tend to struggle in identifying attacks when normal data volume is much higher than the volume of attacks. For example, the dataset used in this study reveals a strong class imbalance with 94,659 instances of the majority class and only 28 instances of the minority class, making it quite challenging to determine rare attacks accurately. The challenges presented in this research are addressed by hybrid sampling techniques designed to improve data imbalance detection accuracy in IoT domains. After applying these techniques, we evaluate the performance of several machine learning models such as Random Forest, Soft Voting, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), and Logistic Regression with respect to the classification of cyber-attacks. The obtained results indicate that the Random Forest model achieved the best performance with a Kappa score of 0.9903, test accuracy of 0.9961, and AUC of 0.9994. Strong performance is also shown by the Soft Voting model, with an accuracy of 0.9952 and AUC of 0.9997, indicating the benefits of combining model predictions. Overall, this work demonstrates the value of hybrid sampling combined with robust model and feature selection for significantly improving IoT security against cyber-attacks, especially in highly imbalanced data environments. 

**Abstract (ZH)**: 由于物联网（IoT）网络的数量快速增长，网络风险已呈指数级增加，因此我们必须开发有效的入侵检测系统（IDS），以应对高度不平衡的数据集。传统的机器学习模型在正常数据量远高于攻击数据量的情况下，难以识别攻击，容易导致误报率过高。本研究使用的数据集展示了严重的类别不平衡，主要类别有94,659个实例，而少数类别仅有28个实例，这使得准确确定罕见攻击变得非常具有挑战性。本研究通过设计的混合采样技术解决了这些挑战，这些技术旨在提高IoT领域中的数据不平衡检测准确性。应用这些技术后，我们评估了包括随机森林、软投票、支持向量分类器（SVC）、K-近邻（KNN）、多层感知器（MLP）和逻辑回归在内的多种机器学习模型在网络安全分类中的性能。结果显示，随机森林模型在κ分数、测试准确率和AUC方面表现最佳，分别为0.9903、0.9961和0.9994。软投票模型也表现出强劲的性能，其准确率为0.9952，AUC为0.9997，显示出结合模型预测的益处。总体而言，本工作展示了混合采样技术与稳健的模型和特征选择相结合的价值，这对显著提高IoT网络安全，特别是在高度不平衡数据环境中，具有重要意义。 

---
# UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable Emotional Text-to-Speech 

**Title (ZH)**: UDDETTS：统一离散和维度情感以实现可控的情感文本-to-语音转换 

**Authors**: Jiaxuan Liu, Zhenhua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2505.10599)  

**Abstract**: Recent neural codec language models have made great progress in the field of text-to-speech (TTS), but controllable emotional TTS still faces many challenges. Traditional methods rely on predefined discrete emotion labels to control emotion categories and intensities, which can't capture the complexity and continuity of human emotional perception and expression. The lack of large-scale emotional speech datasets with balanced emotion distributions and fine-grained emotion annotations often causes overfitting in synthesis models and impedes effective emotion control. To address these issues, we propose UDDETTS, a neural codec language model unifying discrete and dimensional emotions for controllable emotional TTS. This model introduces the interpretable Arousal-Dominance-Valence (ADV) space for dimensional emotion description and supports emotion control driven by either discrete emotion labels or nonlinearly quantified ADV values. Furthermore, a semi-supervised training strategy is designed to comprehensively utilize diverse speech datasets with different types of emotion annotations to train the UDDETTS. Experiments show that UDDETTS achieves linear emotion control along the three dimensions of ADV space, and exhibits superior end-to-end emotional speech synthesis capabilities. 

**Abstract (ZH)**: Recent Neural Codec Language Models Have Made Great Progress in Text-to-Speech (TTS), but Controllable Emotional TTS Still Faces Many Challenges: UDDETTS, a Neural Codec Language Model Unifying Discrete and Dimensional Emotions for Controllable Emotional TTS 

---
# Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment 

**Title (ZH)**: 两地智心胜过一筹：协作型奖励建模实现大语言模型对齐 

**Authors**: Jiazheng Zhang, Wenqing Jing, Zizhuo Zhang, Zhiheng Xi, Shihan Dou, Rongxiang Weng, Jiahuan Li, Jingang Wang, MingXu Cai, Shibo Hong, Tao Gui, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10597)  

**Abstract**: Reward models (RMs) are essential for aligning large language models (LLMs) with human values. However, noisy preferences in human feedback often lead to reward misgeneralization, where RMs overfit to spurious patterns and provide misleading signals during policy optimization. We systematically analyze the training dynamics of preference pairs and identify that noisy examples are harder to fit and introduce instability. Empirical evidence shows that LLMs optimized using reward models trained on full noisy datasets perform worse than those trained on filtered, high-quality preferences. To address this, we propose Collaborative Reward Modeling (CRM), an online framework that enhances robustness by combining peer review and curriculum learning. Two reward models are trained in parallel and assess each other's data selections to filter out potential noise. Curriculum learning structures the preference data from easy to hard, ensuring synchronized training and stable feedback. Extensive experiments demonstrate that CRM improves generalization, with up to 9.94 points of accuracy gain on RewardBench under 40 percent label noise. CRM is also compatible with implicit-reward alignment methods, offering a practical and versatile strategy for robust alignment. 

**Abstract (ZH)**: 基于协作的奖励模型方法（CRM）：一种在线框架，通过同伴评审和课程学习增强鲁棒性 

---
# Inclusivity of AI Speech in Healthcare: A Decade Look Back 

**Title (ZH)**: AI语音在 healthcare 中的包容性：十年回顾 

**Authors**: Retno Larasati  

**Link**: [PDF](https://arxiv.org/pdf/2505.10596)  

**Abstract**: The integration of AI speech recognition technologies into healthcare has the potential to revolutionize clinical workflows and patient-provider communication. However, this study reveals significant gaps in inclusivity, with datasets and research disproportionately favouring high-resource languages, standardized accents, and narrow demographic groups. These biases risk perpetuating healthcare disparities, as AI systems may misinterpret speech from marginalized groups. This paper highlights the urgent need for inclusive dataset design, bias mitigation research, and policy frameworks to ensure equitable access to AI speech technologies in healthcare. 

**Abstract (ZH)**: 将AI语音识别技术集成到医疗保健中有望革命化临床流程和患者-提供者沟通。然而，本研究揭示了包容性方面的显著差距，数据集和研究过度偏向高资源语言、标准化口音和狭窄的人口群体。这些偏见可能导致医疗保健不平等的加剧，因为AI系统可能误解边缘化群体的语音。本文强调了迫切需要包容性数据集设计、偏见缓解研究和政策框架，以确保在医疗保健中公平获取AI语音技术。 

---
# CRPE: Expanding The Reasoning Capability of Large Language Model for Code Generation 

**Title (ZH)**: CRPE: 扩展大型语言模型进行代码生成的推理能力 

**Authors**: Ningxin Gui, Qianghuai Jia, Feijun Jiang, Yuling Jiao, dechun wang, Jerry Zhijian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10594)  

**Abstract**: We introduce CRPE (Code Reasoning Process Enhancer), an innovative three-stage framework for data synthesis and model training that advances the development of sophisticated code reasoning capabilities in large language models (LLMs). Building upon existing system-1 models, CRPE addresses the fundamental challenge of enhancing LLMs' analytical and logical processing in code generation tasks. Our framework presents a methodologically rigorous yet implementable approach to cultivating advanced code reasoning abilities in language models. Through the implementation of CRPE, we successfully develop an enhanced COT-Coder that demonstrates marked improvements in code generation tasks. Evaluation results on LiveCodeBench (20240701-20240901) demonstrate that our COT-Coder-7B-StepDPO, derived from Qwen2.5-Coder-7B-Base, with a pass@1 accuracy of 21.88, exceeds all models with similar or even larger sizes. Furthermore, our COT-Coder-32B-StepDPO, based on Qwen2.5-Coder-32B-Base, exhibits superior performance with a pass@1 accuracy of 35.08, outperforming GPT4O on the benchmark. Overall, CRPE represents a comprehensive, open-source method that encompasses the complete pipeline from instruction data acquisition through expert code reasoning data synthesis, culminating in an autonomous reasoning enhancement mechanism. 

**Abstract (ZH)**: CRPE：代码推理过程增强器——一种促进大型语言模型复杂代码推理能力发展的创新三阶段框架 

---
# LLM-Explorer: Towards Efficient and Affordable LLM-based Exploration for Mobile Apps 

**Title (ZH)**: LLM-Explorer: 向高效和经济实惠的基于LLM的应用程序探索方法迈进 

**Authors**: Shanhui Zhao, Hao Wen, Wenjie Du, Cheng Liang, Yunxin Liu, Xiaozhou Ye, Ye Ouyang, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10593)  

**Abstract**: Large language models (LLMs) have opened new opportunities for automated mobile app exploration, an important and challenging problem that used to suffer from the difficulty of generating meaningful UI interactions. However, existing LLM-based exploration approaches rely heavily on LLMs to generate actions in almost every step, leading to a huge cost of token fees and computational resources. We argue that such extensive usage of LLMs is neither necessary nor effective, since many actions during exploration do not require, or may even be biased by the abilities of LLMs. Further, based on the insight that a precise and compact knowledge plays the central role for effective exploration, we introduce LLM-Explorer, a new exploration agent designed for efficiency and affordability. LLM-Explorer uses LLMs primarily for maintaining the knowledge instead of generating actions, and knowledge is used to guide action generation in a LLM-less manner. Based on a comparison with 5 strong baselines on 20 typical apps, LLM-Explorer was able to achieve the fastest and highest coverage among all automated app explorers, with over 148x lower cost than the state-of-the-art LLM-based approach. 

**Abstract (ZH)**: 大规模语言模型（LLMs）为自动移动应用探索打开了新的机遇，这是一个重要而具有挑战性的问题，过去由于生成有意义的UI交互的难度而受到限制。然而，现有的基于LLM的探索方法几乎在每一步都严重依赖LLM生成操作，导致了大量的Token费用和计算资源消耗。我们argue这样的广泛使用LLM既不是必要的，也不是有效的，因为在探索过程中许多操作并不需要，甚至可能会受到LLM能力的偏差影响。进一步地，基于精确而紧凑的知识在有效探索中起着核心作用的见解，我们介绍了一种新的探索代理LLM-Explorer，旨在提高效率和降低成本。LLM-Explorer主要使用LLM来维护知识，而不是生成操作，知识用于无LLM的方式引导操作生成。在对20个典型应用与5个强大基线的比较中，LLM-Explorer在所有自动化应用探索器中实现了最快的覆盖率，成本比最先进的基于LLM的方法低148倍以上。 

---
# Anchoring AI Capabilities in Market Valuations: The Capability Realization Rate Model and Valuation Misalignment Risk 

**Title (ZH)**: 将AI能力锚定在市场估值中：能力实现率模型与估值失衡风险 

**Authors**: Xinmin Fang, Lingfeng Tao, Zhengxiong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10590)  

**Abstract**: Recent breakthroughs in artificial intelligence (AI) have triggered surges in market valuations for AI-related companies, often outpacing the realization of underlying capabilities. We examine the anchoring effect of AI capabilities on equity valuations and propose a Capability Realization Rate (CRR) model to quantify the gap between AI potential and realized performance. Using data from the 2023--2025 generative AI boom, we analyze sector-level sensitivity and conduct case studies (OpenAI, Adobe, NVIDIA, Meta, Microsoft, Goldman Sachs) to illustrate patterns of valuation premium and misalignment. Our findings indicate that AI-native firms commanded outsized valuation premiums anchored to future potential, while traditional companies integrating AI experienced re-ratings subject to proof of tangible returns. We argue that CRR can help identify valuation misalignment risk-where market prices diverge from realized AI-driven value. We conclude with policy recommendations to improve transparency, mitigate speculative bubbles, and align AI innovation with sustainable market value. 

**Abstract (ZH)**: 近期人工智能领域的突破引发了与人工智能相关公司市场估值的激增，往往超过其潜在能力的实现。我们研究了人工智能能力对股权估值的锚定效应，并提出了一种能力实现率（CRR）模型，以量化人工智能潜在价值与实现性能之间的差距。利用2023-2025年生成式人工智能热潮的数据，我们分析了行业层面的敏感性，并通过案例研究（OpenAI、Adobe、NVIDIA、Meta、Microsoft、Goldman Sachs）来阐述估值溢价和错配的模式。我们的研究发现，人工智能原生企业享有基于未来潜力的巨额估值溢价，而融合人工智能的传统企业则根据实际回报经历了重新评级。我们认为CRR可以帮助识别估值错配风险，即市场价格与通过人工智能实现的价值不符。最后，我们提出了政策建议，以提高透明度、缓解泡沫化，并使人工智能创新与可持续市场价格相一致。 

---
# Super-Resolution Generative Adversarial Networks based Video Enhancement 

**Title (ZH)**: 基于超分辨率生成对抗网络的视频增强 

**Authors**: Kağan ÇETİN  

**Link**: [PDF](https://arxiv.org/pdf/2505.10589)  

**Abstract**: This study introduces an enhanced approach to video super-resolution by extending ordinary Single-Image Super-Resolution (SISR) Super-Resolution Generative Adversarial Network (SRGAN) structure to handle spatio-temporal data. While SRGAN has proven effective for single-image enhancement, its design does not account for the temporal continuity required in video processing. To address this, a modified framework that incorporates 3D Non-Local Blocks is proposed, which is enabling the model to capture relationships across both spatial and temporal dimensions. An experimental training pipeline is developed, based on patch-wise learning and advanced data degradation techniques, to simulate real-world video conditions and learn from both local and global structures and details. This helps the model generalize better and maintain stability across varying video content while maintaining the general structure besides the pixel-wise correctness. Two model variants-one larger and one more lightweight-are presented to explore the trade-offs between performance and efficiency. The results demonstrate improved temporal coherence, sharper textures, and fewer visual artifacts compared to traditional single-image methods. This work contributes to the development of practical, learning-based solutions for video enhancement tasks, with potential applications in streaming, gaming, and digital restoration. 

**Abstract (ZH)**: 该研究提出了一种增强的视频超分辨率方法，通过将普通的单图像超分辨率（SISR）生成对抗网络（SRGAN）结构扩展为处理时空数据的框架。虽然SRGAN在单图像增强方面已 proven 有效，但其设计未考虑视频处理所需的时序连续性。为此，提出了一种结合3D非局部块的改进框架，使模型能够在时空维度上捕捉关系。基于块学习和高级数据降质技术开发了实验训练管道，以模拟真实世界的视频条件，并从局部和全局结构与细节中学习，从而帮助模型更好地泛化并在不同视频内容下保持稳定，同时保持像素级的准确性。提出了两种模型变体——一个更大且一个更轻量级——以探索性能与效率之间的权衡。结果表明，与传统单图像方法相比，这种方法在时间一致性、锐利纹理和较少的视觉伪影方面表现出改进。该工作为视频增强任务提供了实用的基于学习的解决方案，具有潜在的应用价值，如流媒体、游戏和数字修复等领域。 

---
# Understanding Gen Alpha Digital Language: Evaluation of LLM Safety Systems for Content Moderation 

**Title (ZH)**: 理解世代alpha的数字语言：评估大语言模型安全系统在内容审核中的表现 

**Authors**: Manisha Mehta, Fausto Giunchiglia  

**Link**: [PDF](https://arxiv.org/pdf/2505.10588)  

**Abstract**: This research offers a unique evaluation of how AI systems interpret the digital language of Generation Alpha (Gen Alpha, born 2010-2024). As the first cohort raised alongside AI, Gen Alpha faces new forms of online risk due to immersive digital engagement and a growing mismatch between their evolving communication and existing safety tools. Their distinct language, shaped by gaming, memes, and AI-driven trends, often conceals harmful interactions from both human moderators and automated systems. We assess four leading AI models (GPT-4, Claude, Gemini, and Llama 3) on their ability to detect masked harassment and manipulation within Gen Alpha discourse. Using a dataset of 100 recent expressions from gaming platforms, social media, and video content, the study reveals critical comprehension failures with direct implications for online safety. This work contributes: (1) a first-of-its-kind dataset capturing Gen Alpha expressions; (2) a framework to improve AI moderation systems for youth protection; (3) a multi-perspective evaluation including AI systems, human moderators, and parents, with direct input from Gen Alpha co-researchers; and (4) an analysis of how linguistic divergence increases youth vulnerability. Findings highlight the urgent need to redesign safety systems attuned to youth communication, especially given Gen Alpha reluctance to seek help when adults fail to understand their digital world. This study combines the insight of a Gen Alpha researcher with systematic academic analysis to address critical digital safety challenges. 

**Abstract (ZH)**: 本研究提供了对AI系统如何解读世代Alpha（出生于2010-2024年）的数字语言的一种独特评估。作为第一代与AI一同成长的群体，世代Alpha因沉浸式的数字参与和不断变化的沟通方式与现有安全工具之间的差距而面临新的在线风险。他们的语言受到游戏、梗图和AI驱动趋势的影响，常常用来隐藏有害互动，无论是对人类审查员还是自动化系统。研究评估了四种领先AI模型（GPT-4、Claude、Gemini和Llama 3）在检测世代Alpha话语中隐匿的骚扰和操控方面的能力。通过一个包含100个来自游戏平台、社交媒体和视频内容的近期表达的数据集，研究揭示了关键的理解失败，对在线安全有直接影响。该研究贡献了：(1) 首个捕捉世代Alpha表达的数据库；(2) 改进针对青少年保护的AI监控系统的框架；(3) 多视角评估，包括AI系统、人类审查员和家长，并直接纳入世代Alpha合作者的意见；(4) 语言差异如何增加青少年脆弱性的分析。研究结果强调了重新设计与青少年沟通相适应的安全系统的迫切需求，尤其是在成年人无法理解青少年的数字世界时，世代Alpha更不愿意寻求帮助。本研究结合了世代Alpha研究者的经验与系统的学术分析，以应对关键的数字安全挑战。 

---
# GRNN:Recurrent Neural Network based on Ghost Features for Video Super-Resolution 

**Title (ZH)**: 基于幽灵特征的递归神经网络Video Super-分辨率 

**Authors**: Yutong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10577)  

**Abstract**: Modern video super-resolution (VSR) systems based on convolutional neural networks (CNNs) require huge computational costs. The problem of feature redundancy is present in most models in many domains, but is rarely discussed in VSR. We experimentally observe that many features in VSR models are also similar to each other, so we propose to use "Ghost features" to reduce this redundancy. We also analyze the so-called "gradient disappearance" phenomenon generated by the conventional recurrent convolutional network (RNN) model, and combine the Ghost module with RNN to complete the modeling on time series. The current frame is used as input to the model together with the next frame, the output of the previous frame and the hidden state. Extensive experiments on several benchmark models and datasets show that the PSNR and SSIM of our proposed modality are improved to some extent. Some texture details in the video are also better preserved. 

**Abstract (ZH)**: 基于卷积神经网络的现代视频超分辨率（VSR）系统需要巨大的计算成本。在许多领域中，特征冗余问题普遍存在，但在VSR中却鲜少讨论。我们实验观察到VSR模型中的许多特征彼此也很相似，因此我们提出使用“Ghost特征”来减少这种冗余。我们还分析了由传统递归卷积神经网络（RNN）模型产生的所谓的“梯度消失”现象，并将Ghost模块与RNN结合，用于时间序列建模。当前帧与下一帧一起作为模型输入，输出前一帧的输出和隐藏状态。在多个基准模型和数据集上的 extensive 实验显示，我们提出的方法在一定程度上提高了 PSNR 和 SSIM，并更好地保留了视频中的某些纹理细节。 

---
# Large Language Models for Cancer Communication: Evaluating Linguistic Quality, Safety, and Accessibility in Generative AI 

**Title (ZH)**: 大型语言模型在癌症沟通中的应用：生成式AI的语言质量、安全性和可访问性评估 

**Authors**: Agnik Saha, Victoria Churchill, Anny D. Rodriguez, Ugur Kursuncu, Muhammed Y. Idris  

**Link**: [PDF](https://arxiv.org/pdf/2505.10472)  

**Abstract**: Effective communication about breast and cervical cancers remains a persistent health challenge, with significant gaps in public understanding of cancer prevention, screening, and treatment, potentially leading to delayed diagnoses and inadequate treatments. This study evaluates the capabilities and limitations of Large Language Models (LLMs) in generating accurate, safe, and accessible cancer-related information to support patient understanding. We evaluated five general-purpose and three medical LLMs using a mixed-methods evaluation framework across linguistic quality, safety and trustworthiness, and communication accessibility and affectiveness. Our approach utilized quantitative metrics, qualitative expert ratings, and statistical analysis using Welch's ANOVA, Games-Howell, and Hedges' g. Our results show that general-purpose LLMs produced outputs of higher linguistic quality and affectiveness, while medical LLMs demonstrate greater communication accessibility. However, medical LLMs tend to exhibit higher levels of potential harm, toxicity, and bias, reducing their performance in safety and trustworthiness. Our findings indicate a duality between domain-specific knowledge and safety in health communications. The results highlight the need for intentional model design with targeted improvements, particularly in mitigating harm and bias, and improving safety and affectiveness. This study provides a comprehensive evaluation of LLMs for cancer communication, offering critical insights for improving AI-generated health content and informing future development of accurate, safe, and accessible digital health tools. 

**Abstract (ZH)**: 有效沟通关于乳腺癌和宫颈癌的信息仍然是一个持续的健康挑战，公众对癌症预防、筛查和治疗的知识存在显著缺口，可能导致诊断延迟和治疗不足。本研究评估了大型语言模型（LLMs）在生成准确、安全和易访问的癌症相关信息方面的能力和局限性，以支持患者的理解。我们采用混合方法评估框架，评估了五种通用和三种医学LLMs的语言质量、安全性与可信度以及沟通的易访问性和效果。我们的方法采用了定量指标、定性专家评级和使用Welch's ANOVA、Games-Howell和Hedges' g的统计分析。研究结果显示，通用LLMs在语言质量和效果方面表现出更高的输出质量，而医学LLMs在沟通易访问性方面表现出更强的能力。然而，医学LLMs在安全性与可信度方面的潜在危害、毒性及偏见水平较高，降低了其性能。我们的研究结果表明，在健康沟通中，领域特定知识与安全性之间存在双重性。研究结果强调了故意进行模型设计、针对性改进的必要性，特别是减轻危害和偏见、提高安全性和效果。本研究为癌症沟通中LLMs的全面评估提供了依据，并为改进AI生成的健康内容和未来的准确、安全和易访问的数字健康工具开发提供了关键见解。 

---
# GarmentPile: Point-Level Visual Affordance Guided Retrieval and Adaptation for Cluttered Garments Manipulation 

**Title (ZH)**: 服装堆叠：点级视觉潜能引导的检索与适应以应对杂乱服装操作 

**Authors**: Ruihai Wu, Ziyu Zhu, Yuran Wang, Yue Chen, Jiarui Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.09243)  

**Abstract**: Cluttered garments manipulation poses significant challenges due to the complex, deformable nature of garments and intricate garment relations. Unlike single-garment manipulation, cluttered scenarios require managing complex garment entanglements and interactions, while maintaining garment cleanliness and manipulation stability. To address these demands, we propose to learn point-level affordance, the dense representation modeling the complex space and multi-modal manipulation candidates, while being aware of garment geometry, structure, and inter-object relations. Additionally, as it is difficult to directly retrieve a garment in some extremely entangled clutters, we introduce an adaptation module, guided by learned affordance, to reorganize highly-entangled garments into states plausible for manipulation. Our framework demonstrates effectiveness over environments featuring diverse garment types and pile configurations in both simulation and the real world. Project page: this https URL. 

**Abstract (ZH)**: 杂乱衣物操作由于衣物的复杂可变形性质和复杂的衣物关系而面临显著挑战。与单件衣物操作不同，杂乱场景需要管理复杂的衣物纠缠和相互作用，同时保持衣物的清洁和操作稳定性。为应对这些需求，我们提出学习点级功能，即密集表示复杂空间和多模态操作候选方式，并考虑到衣物的几何形状、结构及其与其他物体的关系。此外，由于在某些极其纠缠的杂乱环境中难以直接检索衣物，我们引入了一个由学习到的功能引导的适应模块，将高度纠缠的衣物重新整理为易于操作的状态。我们的框架在包含多种衣物类型和堆积配置的模拟和真实环境中均证明了其有效性。项目页面: [这个链接](this https URL)。 

---
