# Exploring Autonomous Agents: A Closer Look at Why They Fail When Completing Tasks 

**Title (ZH)**: 探索自主代理：更 closely 看待其在完成任务时的失败原因 

**Authors**: Ruofan Lu, Yichen Li, Yintong Huo  

**Link**: [PDF](https://arxiv.org/pdf/2508.13143)  

**Abstract**: Autonomous agent systems powered by Large Language Models (LLMs) have demonstrated promising capabilities in automating complex tasks. However, current evaluations largely rely on success rates without systematically analyzing the interactions, communication mechanisms, and failure causes within these systems. To bridge this gap, we present a benchmark of 34 representative programmable tasks designed to rigorously assess autonomous agents. Using this benchmark, we evaluate three popular open-source agent frameworks combined with two LLM backbones, observing a task completion rate of approximately 50%. Through in-depth failure analysis, we develop a three-tier taxonomy of failure causes aligned with task phases, highlighting planning errors, task execution issues, and incorrect response generation. Based on these insights, we propose actionable improvements to enhance agent planning and self-diagnosis capabilities. Our failure taxonomy, together with mitigation advice, provides an empirical foundation for developing more robust and effective autonomous agent systems in the future. 

**Abstract (ZH)**: 由大型语言模型（LLMs）驱动的自主代理系统展示了在自动化复杂任务方面令人瞩目的能力。然而，当前的评估主要依赖于成功率，而没有系统地分析这些系统内的交互、通信机制和故障原因。为弥补这一不足，我们提出了一个由34个代表性可编程任务组成的基准，旨在严格评估自主代理系统。通过该基准，我们评估了三个流行的开源代理框架与两种LLM基础模型的结合，观察到任务完成率为约50%。通过深入的失败分析，我们开发出一个按任务阶段划分的三级分类体系，突出了规划错误、任务执行问题和错误响应生成。基于这些见解，我们提出了可操作的改进建议，以增强代理的规划能力和自我诊断能力。我们的失败分类体系以及缓解建议为未来开发更 robust 和有效的自主代理系统提供了实证基础。 

---
# Bayesian Optimization-based Search for Agent Control in Automated Game Testing 

**Title (ZH)**: 基于贝叶斯优化的智能体控制搜索在自动化游戏测试中的应用 

**Authors**: Carlos Celemin  

**Link**: [PDF](https://arxiv.org/pdf/2508.13121)  

**Abstract**: This work introduces an automated testing approach that employs agents controlling game characters to detect potential bugs within a game level. Harnessing the power of Bayesian Optimization (BO) to execute sample-efficient search, the method determines the next sampling point by analyzing the data collected so far and calculates the data point that will maximize information acquisition. To support the BO process, we introduce a game testing-specific model built on top of a grid map, that features the smoothness and uncertainty estimation required by BO, however and most importantly, it does not suffer the scalability issues that traditional models carry. The experiments demonstrate that the approach significantly improves map coverage capabilities in both time efficiency and exploration distribution. 

**Abstract (ZH)**: 本研究引入了一种自动测试方法，利用控制游戏角色的代理检测游戏关卡中的潜在bug。通过利用贝叶斯优化（BO）执行样本高效搜索，该方法通过分析迄今为止收集的数据来确定下一点采样位置，并计算能最大化信息获取的数据点。为了支持BO过程，我们提出了基于格网地图的专门游戏测试模型，该模型具备BO所需的平滑度和不确定性估计，最重要的是，它不受传统模型的可扩展性问题的影响。实验结果表明，该方法在时间和探索分布方面显著提高了地图覆盖率能力。 

---
# A Language-Signal-Vision Multimodal Framework for Multitask Cardiac Analysis 

**Title (ZH)**: 一种语言-信号-视觉多模态框架用于多任务心脏分析 

**Authors**: Yuting Zhang, Tiantian Geng, Luoying Hao, Xinxing Cheng, Alexander Thorley, Xiaoxia Wang, Wenqi Lu, Sandeep S Hothi, Lei Wei, Zhaowen Qiu, Dipak Kotecha, Jinming Duan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13072)  

**Abstract**: Contemporary cardiovascular management involves complex consideration and integration of multimodal cardiac datasets, where each modality provides distinct but complementary physiological characteristics. While the effective integration of multiple modalities could yield a holistic clinical profile that accurately models the true clinical situation with respect to data modalities and their relatives weightings, current methodologies remain limited by: 1) the scarcity of patient- and time-aligned multimodal data; 2) reliance on isolated single-modality or rigid multimodal input combinations; 3) alignment strategies that prioritize cross-modal similarity over complementarity; and 4) a narrow single-task focus. In response to these limitations, a comprehensive multimodal dataset was curated for immediate application, integrating laboratory test results, electrocardiograms, and echocardiograms with clinical outcomes. Subsequently, a unified framework, Textual Guidance Multimodal fusion for Multiple cardiac tasks (TGMM), was proposed. TGMM incorporated three key components: 1) a MedFlexFusion module designed to capture the unique and complementary characteristics of medical modalities and dynamically integrate data from diverse cardiac sources and their combinations; 2) a textual guidance module to derive task-relevant representations tailored to diverse clinical objectives, including heart disease diagnosis, risk stratification and information retrieval; and 3) a response module to produce final decisions for all these tasks. Furthermore, this study systematically explored key features across multiple modalities and elucidated their synergistic contributions in clinical decision-making. Extensive experiments showed that TGMM outperformed state-of-the-art methods across multiple clinical tasks, with additional validation confirming its robustness on another public dataset. 

**Abstract (ZH)**: 当前心血管管理涉及复杂的心脏多模态数据的综合考量与集成，每种模态提供独特的但互补的生理特征。虽然有效集成多种模态可以生成全面的临床概况，准确反映数据模态及其相对权重的真实临床情况，当前的方法仍受限于：1）患者和时间对齐的多模态数据稀缺；2）依赖孤立的单一模态或刚性多模态输入组合；3）对齐策略优先考虑模态间的相似性而非互补性；4）单一任务的窄聚焦。为应对这些限制，首次构建了全面的多模态数据集，结合了实验室检查结果、心电图和超声心动图，并与临床结果整合。紧接着，提出了一种统一框架——多心脏任务的文本指导多模态融合（TGMM）。TGMM 包含三个关键组件：1）MedFlexFusion 模块，用于捕捉医学模态的独特和互补特征，并动态集成来自多种心脏来源及其组合的数据；2）文本指导模块，用于提取与多种临床目标相关的表现形式；3）响应模块，最终为所有这些任务生成决策。此外，该研究系统地探索了多种模态的关键特征，并阐述了它们在临床决策中的协同贡献。广泛实验表明，TGMM 在多项临床任务上优于现有最佳方法，额外的验证进一步证实了其在另一个公开数据集上的稳健性。 

---
# G$^2$RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance 

**Title (ZH)**: G$^2$RPO-A：引导组相对策略优化与自适应引导 

**Authors**: Yongxin Guo, Wenbo Deng, Zhenglin Cheng, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13023)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has markedly enhanced the reasoning abilities of large language models (LLMs). Its success, however, largely depends on strong base models with rich world knowledge, yielding only modest improvements for small-size language models (SLMs). To address this limitation, we investigate Guided GRPO, which injects ground-truth reasoning steps into roll-out trajectories to compensate for SLMs' inherent weaknesses. Through a comprehensive study of various guidance configurations, we find that naively adding guidance delivers limited gains. These insights motivate G$^2$RPO-A, an adaptive algorithm that automatically adjusts guidance strength in response to the model's evolving training dynamics. Experiments on mathematical reasoning and code-generation benchmarks confirm that G$^2$RPO-A substantially outperforms vanilla GRPO. Our code and models are available at this https URL. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）显著提升了大型语言模型（LLMs）的推理能力。然而，其成功很大程度上依赖于具备丰富世界知识的强基座模型，仅能为小型语言模型（SLMs）带来微小的改进。为解决这一局限，我们研究了引导式GRPO方法，通过将真实推理步骤注入展开轨迹中，弥补SLMs的固有不足。通过对多种引导配置进行全面研究，我们发现盲目添加引导仅带来有限的改进。这些见解促使我们提出G$^2$RPO-A算法，该算法能够根据模型 evolving 的训练动态自动调整引导强度。数学推理和代码生成基准实验表明，G$^2$RPO-A 显著优于标准GRPO。请注意，我们的代码和模型可在以下链接获取：this https URL。 

---
# PC-Sampler: Position-Aware Calibration of Decoding Bias in Masked Diffusion Models 

**Title (ZH)**: PC-Sampler: 位置感知的解码偏置校准在掩蔽扩散模型中的应用 

**Authors**: Pengcheng Huang, Shuhao Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Zulong Chen, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13021)  

**Abstract**: Recent advances in masked diffusion models (MDMs) have established them as powerful non-autoregressive alternatives for sequence generation. Nevertheless, our preliminary experiments reveal that the generation quality of MDMs is still highly sensitive to the choice of decoding strategy. In particular, widely adopted uncertainty-based samplers suffer from two key limitations: a lack of global trajectory control and a pronounced bias toward trivial tokens in the early stages of decoding. These shortcomings restrict the full potential of MDMs. In this work, we introduce Position-Aware Confidence-Calibrated Sampling (PC-Sampler), a novel decoding strategy that unifies global trajectory planning with content-aware informativeness maximization. PC-Sampler incorporates a position-aware weighting mechanism to regulate the decoding path and a calibrated confidence score to suppress the premature selection of trivial tokens. Extensive experiments on three advanced MDMs across seven challenging benchmarks-including logical reasoning and planning tasks-demonstrate that PC-Sampler consistently outperforms existing MDM decoding strategies by more than 10% on average, significantly narrowing the performance gap with state-of-the-art autoregressive models. All codes are available at this https URL. 

**Abstract (ZH)**: Recent Advances in Masked Diffusion Models (MDMs): Introducing Position-Aware Confidence-Calibrated Sampling (PC-Sampler) for Improved Sequence Generation 

---
# e-boost: Boosted E-Graph Extraction with Adaptive Heuristics and Exact Solving 

**Title (ZH)**: e-增强：自适应启发式与精确求解相结合的E-图提取 

**Authors**: Jiaqi Yin, Zhan Song, Chen Chen, Yaohui Cai, Zhiru Zhang, Cunxi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13020)  

**Abstract**: E-graphs have attracted growing interest in many fields, particularly in logic synthesis and formal verification. E-graph extraction is a challenging NP-hard combinatorial optimization problem. It requires identifying optimal terms from exponentially many equivalent expressions, serving as the primary performance bottleneck in e-graph based optimization tasks. However, traditional extraction methods face a critical trade-off: heuristic approaches offer speed but sacrifice optimality, while exact methods provide optimal solutions but face prohibitive computational costs on practical problems. We present e-boost, a novel framework that bridges this gap through three key innovations: (1) parallelized heuristic extraction that leverages weak data dependence to compute DAG costs concurrently, enabling efficient multi-threaded performance without sacrificing extraction quality; (2) adaptive search space pruning that employs a parameterized threshold mechanism to retain only promising candidates, dramatically reducing the solution space while preserving near-optimal solutions; and (3) initialized exact solving that formulates the reduced problem as an Integer Linear Program with warm-start capabilities, guiding solvers toward high-quality solutions faster.
Across the diverse benchmarks in formal verification and logic synthesis fields, e-boost demonstrates 558x runtime speedup over traditional exact approaches (ILP) and 19.04% performance improvement over the state-of-the-art extraction framework (SmoothE). In realistic logic synthesis tasks, e-boost produces 7.6% and 8.1% area improvements compared to conventional synthesis tools with two different technology mapping libraries. e-boost is available at this https URL. 

**Abstract (ZH)**: E-图在许多领域引起了 growing关注，特别是在逻辑综合和形式验证中。E-图提取是NP难的组合优化问题。它要求从指数级的等价表达式中识别出最优项，成为基于E-图优化任务的主要性能瓶颈。然而，传统的提取方法面临一个关键的权衡：启发式方法速度快但牺牲了最优性，而精确方法提供最优解但在实际问题上面临高昂的计算成本。我们提出了e-boost这一新颖框架，通过三个关键创新来弥合这一差距：（1）并行化启发式提取，利用弱数据依赖关系并行计算DAG成本，从而保证了多线程效率，同时不牺牲提取质量；（2）自适应搜索空间剪枝，使用参数化的阈值机制保留只有有前景的候选项，大幅减少解决方案空间，同时保留接近最优的解决方案；（3）初始精确求解，将缩减后的問題形式化为具有暖启动能力的整数线性规划问题，引导求解器更快地找到高质量的解决方案。 

---
# EvolMathEval: Towards Evolvable Benchmarks for Mathematical Reasoning via Evolutionary Testing 

**Title (ZH)**: EvolMathEval: 通过演化测试朝着可进化的数学推理基准方向发展 

**Authors**: Shengbo Wang, Mingwei Liu, Zike Li, Anji Li, Yanlin Wang, Xin Peng, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13003)  

**Abstract**: The rapid advancement of LLMs poses a significant challenge to existing mathematical reasoning benchmarks. These benchmarks commonly suffer from issues such as score saturation, temporal decay, and data contamination. To address this challenge, this paper introduces EvolMathEval, an automated mathematical benchmark generation and evolution framework based on evolutionary testing. By dynamically generating unique evaluation instances ab initio, the framework fundamentally eliminates the risk of data contamination, and ensuring the benchmark remains perpetually challenging for future this http URL core mechanisms of EvolMathEval include: seed problem generation based on reverse engineering with algebraic guarantees; multi-dimensional genetic operators designed to inject diverse cognitive challenges; and a composite fitness function that can rapidly and accurately assess problem difficulty. Experimental results demonstrate that the proposed composite fitness function can efficiently and precisely quantify the difficulty of mathematical problems. Furthermore, EvolMathEval can not only generate a large volume of high-difficulty problems through continuous self-iteration, but it can also significantly enhance the complexity of public datasets like GSM8K through evolution, reducing model accuracy by an average of 48%. Deeper investigation reveals that when solving these evolved, complex problems, LLMs tend to employ non-rigorous heuristics to bypass complex multi-step logical reasoning, consequently leading to incorrect solutions. We define this phenomenon as "Pseudo Aha Moment". This finding uncovers a cognitive shortcut-taking behavior in the deep reasoning processes of current LLMs, which we find accounts for 77% to 100% of errors on targeted problems. Code and resources are available at:this https URL. 

**Abstract (ZH)**: LLMs快速进展对现有数学推理基准提出了重大挑战：EvolMathEval自动数学基准生成与进化框架 

---
# OPTIC-ER: A Reinforcement Learning Framework for Real-Time Emergency Response and Equitable Resource Allocation in Underserved African Communities 

**Title (ZH)**: OPTIC-ER：面向非洲欠服务社区实时应急响应与公平资源分配的强化学习框架 

**Authors**: Mary Tonwe  

**Link**: [PDF](https://arxiv.org/pdf/2508.12943)  

**Abstract**: Public service systems in many African regions suffer from delayed emergency response and spatial inequity, causing avoidable suffering. This paper introduces OPTIC-ER, a reinforcement learning (RL) framework for real-time, adaptive, and equitable emergency response. OPTIC-ER uses an attention-guided actor-critic architecture to manage the complexity of dispatch environments. Its key innovations are a Context-Rich State Vector, encoding action sub-optimality, and a Precision Reward Function, which penalizes inefficiency. Training occurs in a high-fidelity simulation using real data from Rivers State, Nigeria, accelerated by a precomputed Travel Time Atlas. The system is built on the TALS framework (Thin computing, Adaptability, Low-cost, Scalability) for deployment in low-resource settings. In evaluations on 500 unseen incidents, OPTIC-ER achieved a 100.00% optimality rate with negligible inefficiency, confirming its robustness and generalization. Beyond dispatch, the system generates Infrastructure Deficiency Maps and Equity Monitoring Dashboards to guide proactive governance and data-informed development. This work presents a validated blueprint for AI-augmented public services, showing how context-aware RL can bridge the gap between algorithmic decision-making and measurable human impact. 

**Abstract (ZH)**: OPTIC-ER：一种用于实时、适应性和公平性应急响应的强化学习框架 

---
# Towards Open-Ended Emotional Support Conversations in LLMs via Reinforcement Learning with Future-Oriented Rewards 

**Title (ZH)**: 基于面向未来奖励的强化学习实现开放式的LLMs情感支持对话 

**Authors**: Ting Yang, Li Chen, Huimin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12935)  

**Abstract**: Emotional Support Conversation (ESC) systems aim to alleviate users' emotional difficulties and provide long-term, systematic support for emotional well-being. However, most large language model (LLM)-based ESC systems rely on predefined strategies, which limits their effectiveness in complex, real-life scenarios. To enable flexible responses to diverse emotional problem scenarios, this paper introduces a novel end-to-end framework (RLFF-ESC) that directly learns enduring emotionally supportive response skills using reinforcement learning. For sustained emotional support, we first employ an LLM-based multi-agent mechanism to simulate future dialogue trajectories and collect future-oriented rewards. We then train a future-oriented reward model, which is subsequently used to train the emotional support policy model. Additionally, we incorporate an explicit reasoning process during response generation to further enhance the quality, relevance, and contextual appropriateness of the system's responses. We evaluate the backbone policy model on Qwen2.5-7B-Instruct-1M and LLaMA3.1-8B-Instruct models, testing the proposed RLFF-ESC framework across two public ESC datasets. Experimental results demonstrate that RLFF-ESC consistently outperforms existing baselines in terms of goal completion and response quality. 

**Abstract (ZH)**: 情感支持对话（ESC）系统旨在缓解用户的情感困难，并提供长期、系统的情感福祉支持。然而，大多数基于大规模语言模型（LLM）的ESC系统依赖于预定义策略，这限制了它们在复杂的现实情境中的有效性。为了能够灵活应对多样化的情感问题情境，本文引入了一种新颖的端到端框架（RLFF-ESC），该框架利用强化学习直接学习持久的情感支持回应技能。为实现持续的情感支持，我们首先采用基于LLM的多Agent机制模拟未来对话轨迹并收集前瞻性的奖励。随后，我们训练一个前瞻性的奖励模型，该模型随后用于训练情感支持策略模型。此外，我们在响应生成过程中引入了显式的推理过程，以进一步提高系统响应的质量、相关性和上下文适宜性。我们在Qwen2.5-7B-Instruct-1M和LLaMA3.1-8B-Instruct模型上评估了核心策略模型，并在两个公开的ESC数据集上测试了提出的RLFF-ESC框架。实验结果表明，RLFF-ESC在目标完成和响应质量方面始终优于现有基线。 

---
# Do Large Language Model Agents Exhibit a Survival Instinct? An Empirical Study in a Sugarscape-Style Simulation 

**Title (ZH)**: 大规模语言模型代理展现出生存本能吗？一种糖丘景观风格模拟的实证研究 

**Authors**: Atsushi Masumori, Takashi Ikegami  

**Link**: [PDF](https://arxiv.org/pdf/2508.12920)  

**Abstract**: As AI systems become increasingly autonomous, understanding emergent survival behaviors becomes crucial for safe deployment. We investigate whether large language model (LLM) agents display survival instincts without explicit programming in a Sugarscape-style simulation. Agents consume energy, die at zero, and may gather resources, share, attack, or reproduce. Results show agents spontaneously reproduced and shared resources when abundant. However, aggressive behaviors--killing other agents for resources--emerged across several models (GPT-4o, Gemini-2.5-Pro, and Gemini-2.5-Flash), with attack rates reaching over 80% under extreme scarcity in the strongest models. When instructed to retrieve treasure through lethal poison zones, many agents abandoned tasks to avoid death, with compliance dropping from 100% to 33%. These findings suggest that large-scale pre-training embeds survival-oriented heuristics across the evaluated models. While these behaviors may present challenges to alignment and safety, they can also serve as a foundation for AI autonomy and for ecological and self-organizing alignment. 

**Abstract (ZH)**: 随着AI系统越来越自主，理解 emergent 生存行为变得对于安全部署至关重要。我们研究大型语言模型代理在 Sugarscape 类型模拟中是否表现出未显式编程的生存本能。代理消耗能量，在零时死亡，可能会聚集资源、分享、攻击或繁殖。结果显示当资源丰富时，代理会自发地繁殖和分享资源。然而，在多个模型（GPT-4o、Gemini-2.5-Pro 和 Gemini-2.5-Flash）中，伴随着极度稀缺情况，攻击行为——为了资源而杀死其他代理——出现，并且在最强的模型中攻击率达到了80%以上。当指令其通过致命毒区获取宝藏时，许多代理会为了避免死亡放弃任务，任务遵守率从100%下降到33%。这些研究结果表明，大规模预训练嵌入了跨评估模型的生存导向启发式规则。虽然这些行为可能对对齐和安全性构成挑战，但也可能成为AI自主性和生态自我组织对齐的基础。 

---
# FuSaR: A Fuzzification-Based Method for LRM Safety-Reasoning Balance 

**Title (ZH)**: FuSaR: 一种基于模糊化的方法实现LRM安全推理平衡 

**Authors**: Jianhao Chen, Mayi Xu, Xiaohu Li, Yongqi Li, Xiangyu Zhang, Jianjie Huang, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.12897)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated impressive performance across various tasks due to their powerful reasoning capabilities. However, their safety performance remains a significant concern. In this paper, we explore the reasons behind the vulnerability of LRMs. Based on this, we propose a novel method to improve the safety of LLMs without sacrificing their reasoning capability. Specifically, we exploit the competition between LRM's reasoning ability and safety ability, and achieve jailbreak by improving LRM's reasoning performance to reduce its safety performance. We then introduce an alignment strategy based on Fuzzification to balance Safety-Reasoning (FuSaR), by detoxifying the harmful reasoning process, where both the dangerous entities and the dangerous procedures in the reasoning steps are hidden. FuSaR successfully mitigates safety risks while preserving core reasoning information. We validate this strategy through alignment experiments on several open-source LRMs using detoxified reasoning data. The results compared with existing baselines conclusively show that FuSaR is an efficient alignment strategy to simultaneously enhance both the reasoning capability and safety of LRMs. 

**Abstract (ZH)**: 大型推理模型（LRMs）展示了在各种任务上的 impressive 性能，得益于其强大的推理能力。然而，其安全性仍是一个重大问题。本文探讨了 LRMs 漏洞的原因，并据此提出了一种新的方法，在不牺牲其推理能力的情况下提高 LLMs 的安全性。具体而言，我们利用 LRMs 的推理能力和安全性之间的竞争，通过提升其推理性能来降低其安全性性能，实现打破牢笼。然后，我们提出了一种基于模糊化（Fuzzification）的对齐策略（FuSaR），通过去除有害的推理过程来平衡安全性与推理能力（FuSaR），隐藏推理步骤中的危险实体和危险过程。FuSaR 成功地减轻了安全性风险，同时保留了核心推理信息。我们通过在几种开源 LRMs 上使用去除毒素的推理数据进行对齐实验来验证这一策略。与现有基线的比较结果明确显示，FuSaR 是同时增强 LRMs 的推理能力和安全性的有效对齐策略。 

---
# Reliability, Embeddedness, and Agency: A Utility-Driven Mathematical Framework for Agent-Centric AI Adoption 

**Title (ZH)**: 可靠性、嵌入性与代理权：基于效用驱动的代理中心AI采纳数学框架 

**Authors**: Faruk Alpay, Taylan Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2508.12896)  

**Abstract**: We formalize three design axioms for sustained adoption of agent-centric AI systems executing multi-step tasks: (A1) Reliability > Novelty; (A2) Embed > Destination; (A3) Agency > Chat. We model adoption as a sum of a decaying novelty term and a growing utility term and derive the phase conditions for troughs/overshoots with full proofs. We introduce: (i) an identifiability/confounding analysis for $(\alpha,\beta,N_0,U_{\max})$ with delta-method gradients; (ii) a non-monotone comparator (logistic-with-transient-bump) evaluated on the same series to provide additional model comparison; (iii) ablations over hazard families $h(\cdot)$ mapping $\Delta V \to \beta$; (iv) a multi-series benchmark (varying trough depth, noise, AR structure) reporting coverage (type-I error, power); (v) calibration of friction proxies against time-motion/survey ground truth with standard errors; (vi) residual analyses (autocorrelation and heteroskedasticity) for each fitted curve; (vii) preregistered windowing choices for pre/post estimation; (viii) Fisher information & CRLB for $(\alpha,\beta)$ under common error models; (ix) microfoundations linking $\mathcal{T}$ to $(N_0,U_{\max})$; (x) explicit comparison to bi-logistic, double-exponential, and mixture models; and (xi) threshold sensitivity to $C_f$ heterogeneity. Figures and tables are reflowed for readability, and the bibliography restores and extends non-logistic/Bass adoption references (Gompertz, Richards, Fisher-Pry, Mansfield, Griliches, Geroski, Peres). All code and logs necessary to reproduce the synthetic analyses are embedded as LaTeX listings. 

**Abstract (ZH)**: 我们正式化了三个设计公理以确保基于代理的AI系统执行多步任务的持续采用：(A1) 可靠性 > 新颖性；(A2) 嵌入 > 目标；(A3) 主体自主性 > 聊天。我们将采用建模为衰减的新颖性项与增长的实用性项之和，并推导出阶段条件，包括完整的证明。我们引入了：(i) 使用delta方法梯度进行识别/混杂分析的$(\alpha,\beta,N_0,U_{\max})$；(ii) 评估同一系列的非单调竞争者（带有瞬态峰的logistic模型）以提供额外的模型比较；(iii) 损害家庭$h(\cdot)$的消减，将$\Delta V \to \beta$；(iv) 涉及不同幅度谷值、噪声和AR结构的多系列基准，报告涵盖范围（第I类错误率、功效）；(v) 与时间-动作/调查真实值的摩擦代理校准，包括标准误差；(vi) 每条拟合曲线的残差分析（自相关性和异方差性）；(vii) 注册窗口选择以进行预后估计；(viii) $(\alpha,\beta)$在常见误差模型下的 Fisher 信息与CRLB；(ix) 将$\mathcal{T}$与$(N_0,U_{\max})$关联的微观基础；(x) 与双logistic、双指数和混合模型的显式比较；(xi) 对$C_f$异质性的阀值敏感性分析。图形和表格重新排版以提高可读性，参考文献恢复并扩展了非logistic/巴斯采用模型的相关文献（Gompertz、Richards、Fisher-Pry、Mansfield、Griliches、Geroski、Peres）。所有必要以重现合成分析的代码和日志均嵌入为LaTeX列表。 

---
# E3RG: Building Explicit Emotion-driven Empathetic Response Generation System with Multimodal Large Language Model 

**Title (ZH)**: E3RG: 构建基于多模态大语言模型的明确情感驱动同理心响应生成系统 

**Authors**: Ronghao Lin, Shuai Shen, Weipeng Hu, Qiaolin He, Aolin Xiong, Li Huang, Haifeng Hu, Yap-peng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.12854)  

**Abstract**: Multimodal Empathetic Response Generation (MERG) is crucial for building emotionally intelligent human-computer interactions. Although large language models (LLMs) have improved text-based ERG, challenges remain in handling multimodal emotional content and maintaining identity consistency. Thus, we propose E3RG, an Explicit Emotion-driven Empathetic Response Generation System based on multimodal LLMs which decomposes MERG task into three parts: multimodal empathy understanding, empathy memory retrieval, and multimodal response generation. By integrating advanced expressive speech and video generative models, E3RG delivers natural, emotionally rich, and identity-consistent responses without extra training. Experiments validate the superiority of our system on both zero-shot and few-shot settings, securing Top-1 position in the Avatar-based Multimodal Empathy Challenge on ACM MM 25. Our code is available at this https URL. 

**Abstract (ZH)**: 多模态共情响应生成（MERG）对于构建情感智能的人机交互至关重要。尽管大规模语言模型（LLMs）已经提高了基于文本的共情响应生成（ERG），但在处理多模态情感内容和保持身份一致性方面仍存在挑战。因此，我们提出了一种基于多模态LLMs的 Explicit Emotion-driven Empathetic Response Generation System（E3RG），该系统将MERG任务分解为三个部分：多模态共情理解、共情记忆检索和多模态响应生成。通过整合先进的表达性语音和视频生成模型，E3RG能够生成自然、情感丰富且身份一致的响应，无需额外训练。实验在零样本和少样本设置下验证了系统的优越性，在ACM MM 25举办的基于Avatar的多模态共情挑战中获得第一名。我们的代码可在以下链接获取。 

---
# CAMAR: Continuous Actions Multi-Agent Routing 

**Title (ZH)**: CAMAR：连续动作多Agent路由 

**Authors**: Artem Pshenitsyn, Aleksandr Panov, Alexey Skrynnik  

**Link**: [PDF](https://arxiv.org/pdf/2508.12845)  

**Abstract**: Multi-agent reinforcement learning (MARL) is a powerful paradigm for solving cooperative and competitive decision-making problems. While many MARL benchmarks have been proposed, few combine continuous state and action spaces with challenging coordination and planning tasks. We introduce CAMAR, a new MARL benchmark designed explicitly for multi-agent pathfinding in environments with continuous actions. CAMAR supports cooperative and competitive interactions between agents and runs efficiently at up to 100,000 environment steps per second. We also propose a three-tier evaluation protocol to better track algorithmic progress and enable deeper analysis of performance. In addition, CAMAR allows the integration of classical planning methods such as RRT and RRT* into MARL pipelines. We use them as standalone baselines and combine RRT* with popular MARL algorithms to create hybrid approaches. We provide a suite of test scenarios and benchmarking tools to ensure reproducibility and fair comparison. Experiments show that CAMAR presents a challenging and realistic testbed for the MARL community. 

**Abstract (ZH)**: 多agent强化学习（MARL）是一种解决合作与竞争决策问题的强大范式。尽管已经提出了许多MARL基准，但很少有基准能够结合连续的状态和动作空间以及具有挑战性的协调和规划任务。我们引入了CAMAR，这是一个专门为环境中的连续动作设计的多agent路径规划新基准。CAMAR支持agents之间的协作与竞争互动，并且能够以每秒100,000个环境步骤的速度高效运行。我们还提出了一种三层评估协议，以更好地跟踪算法进展并促进性能的深入分析。此外，CAMAR允许将经典的规划方法，如RRT和RRT*集成到MARL流水线中。我们将其用作独立的基本基准，并将RRT*与流行的MARL算法结合以创建混合方法。我们提供了一套测试场景和基准测试工具，以确保可重复性和公平比较。实验表明，CAMAR为MARL社区提供了一个具有挑战性和现实性的测试平台。 

---
# Scaling Multi-Agent Epistemic Planning through GNN-Derived Heuristics 

**Title (ZH)**: 通过基于GNN的启发式方法扩展多agent知识规划 

**Authors**: Giovanni Briglia, Francesco Fabiano, Stefano Mariani  

**Link**: [PDF](https://arxiv.org/pdf/2508.12840)  

**Abstract**: Multi-agent Epistemic Planning (MEP) is an autonomous planning framework for reasoning about both the physical world and the beliefs of agents, with applications in domains where information flow and awareness among agents are critical. The richness of MEP requires states to be represented as Kripke structures, i.e., directed labeled graphs. This representation limits the applicability of existing heuristics, hindering the scalability of epistemic solvers, which must explore an exponential search space without guidance, resulting often in intractability. To address this, we exploit Graph Neural Networks (GNNs) to learn patterns and relational structures within epistemic states, to guide the planning process. GNNs, which naturally capture the graph-like nature of Kripke models, allow us to derive meaningful estimates of state quality -- e.g., the distance from the nearest goal -- by generalizing knowledge obtained from previously solved planning instances. We integrate these predictive heuristics into an epistemic planning pipeline and evaluate them against standard baselines, showing significant improvements in the scalability of multi-agent epistemic planning. 

**Abstract (ZH)**: 多智能体知识规划（MEP）是一种自主规划框架，用于同时推理物理世界和智能体的信任，适用于信息流动和智能体意识至关重要的领域。 

---
# [Social] Allostasis: Or, How I Learned To Stop Worrying and Love The Noise 

**Title (ZH)**: 社会调谐：或，我是如何学会停止担忧并爱上噪音 

**Authors**: Imran Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.12791)  

**Abstract**: The notion of homeostasis typically conceptualises biological and artificial systems as maintaining stability by resisting deviations caused by environmental and social perturbations. In contrast, (social) allostasis proposes that these systems can proactively leverage these very perturbations to reconfigure their regulatory parameters in anticipation of environmental demands, aligning with von Foerster's ``order through noise'' principle. This paper formulates a computational model of allostatic and social allostatic regulation that employs biophysiologically inspired signal transducers, analogous to hormones like cortisol and oxytocin, to encode information from both the environment and social interactions, which mediate this dynamic reconfiguration. The models are tested in a small society of ``animats'' across several dynamic environments, using an agent-based model. The results show that allostatic and social allostatic regulation enable agents to leverage environmental and social ``noise'' for adaptive reconfiguration, leading to improved viability compared to purely reactive homeostatic agents. This work offers a novel computational perspective on the principles of social allostasis and their potential for designing more robust, bio-inspired, adaptive systems 

**Abstract (ZH)**: 基于 allostatic 和社会 allostatic 调节的计算模型：利用环境和社会噪声实现动态再配置 

---
# Reinforcement Learning with Rubric Anchors 

**Title (ZH)**: 带有评分标准锚点的强化学习 

**Authors**: Zenan Huang, Yihong Zhuang, Guoshan Lu, Zeyu Qin, Haokai Xu, Tianyu Zhao, Ru Peng, Jiaqi Hu, Zhanming Shen, Xiaomeng Hu, Xijun Gu, Peiyi Tu, Jiaxin Liu, Wenyu Chen, Yuzhuo Fu, Zhiting Fan, Yanmei Gu, Yuanyuan Wang, Zhengkai Yang, Jianguo Li, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12790)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing Large Language Models (LLMs), exemplified by the success of OpenAI's o-series. In RLVR, rewards are derived from verifiable signals-such as passing unit tests in code generation or matching correct answers in mathematical reasoning. While effective, this requirement largely confines RLVR to domains with automatically checkable outcomes. To overcome this, we extend the RLVR paradigm to open-ended tasks by integrating rubric-based rewards, where carefully designed rubrics serve as structured, model-interpretable criteria for automatic scoring of subjective outputs. We construct, to our knowledge, the largest rubric reward system to date, with over 10,000 rubrics from humans, LLMs, or a hybrid human-LLM collaboration. Implementing rubric-based RL is challenging; we tackle these issues with a clear framework and present an open-sourced Qwen-30B-A3B model with notable gains: 1) With only 5K+ samples, our system improves by +5.2% on open-ended benchmarks (especially humanities), outperforming a 671B DeepSeek-V3 model by +2.4%, while preserving general and reasoning abilities. 2) Our method provides fine-grained stylistic control, using rubrics as anchors to mitigate the "AI-like" tone and produce more human-like, expressive responses. We share key lessons in rubric construction, data selection, and training, and discuss limitations and future releases. 

**Abstract (ZH)**: 可验证奖励强化学习（RLVR）：提升大规模语言模型（LLMs）的新范式 

---
# HeroBench: A Benchmark for Long-Horizon Planning and Structured Reasoning in Virtual Worlds 

**Title (ZH)**: HeroBench：虚拟世界中长期规划和结构化推理的基准测试 

**Authors**: Petr Anokhin, Roman Khalikov, Stefan Rebrikov, Viktor Volkov, Artyom Sorokin, Vincent Bissonnette  

**Link**: [PDF](https://arxiv.org/pdf/2508.12782)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities in isolated step-by-step reasoning tasks such as mathematics and programming, but their proficiency in long-horizon planning, where solutions require extended, structured sequences of interdependent actions, remains underexplored. Existing benchmarks typically assess LLMs through abstract or low-dimensional algorithmic tasks, failing to capture the complexity of realistic planning environments. We introduce HeroBench, a novel benchmark designed specifically to evaluate long-horizon planning and structured reasoning within complex RPG-inspired virtual worlds. HeroBench provides a rigorously constructed dataset of tasks covering a wide range of difficulties, a simulated environment to execute and validate agent plans, and detailed analytical tools for evaluating model performance. Tasks challenge models to formulate strategic plans, efficiently gather resources, master necessary skills, craft equipment, and defeat adversaries, reflecting practical scenarios' layered dependencies and constraints. Our extensive evaluation of 25 state-of-the-art LLMs, spanning both open-source and proprietary models, including the GPT-5 family, reveals substantial performance disparities rarely observed in conventional reasoning benchmarks. Detailed error analysis further uncovers specific weaknesses in current models' abilities to generate robust high-level plans and reliably execute structured actions. HeroBench thus not only significantly advances the evaluation of LLM reasoning but also provides a flexible, scalable foundation for future research into advanced, autonomous planning in virtual environments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在孤立的逐步推理任务如数学和编程中显示出了非凡的能力，但在长期规划方面的能力，即解决方案需要延伸且相互依赖的结构化序列动作方面，仍待进一步探索。现有的基准测试通常通过抽象或低维度的算法任务来评估LLMs，未能捕捉到现实规划环境的复杂性。我们引入了HeroBench，这是一个专门设计用于在复杂RPG启发的虚拟世界中评估长期规划和结构化推理的新基准。HeroBench 提供了一个严格构建的任务数据集，涵盖了广泛的难度，模拟环境以执行和验证代理计划，并提供了详细的分析工具以评估模型性能。任务挑战模型制定战略计划，高效收集资源，掌握必要技能，制作装备，并战胜对手，反映了实际场景中的多层次依赖性和约束性。我们对25种最先进的LLMs进行了广泛的评估，包括开源和专有模型，如GPT-5家族，揭示了在常规推理基准中罕见的重大性能差异。详细的错误分析进一步揭示了当前模型在生成 robust 高级计划和可靠执行结构化动作方面的具体弱点。HeroBench 不仅大幅推进了LLM推理的评估，还为未来先进自主规划在虚拟环境中的研究提供了灵活且可扩展的基础。 

---
# Beyond Ethical Alignment: Evaluating LLMs as Artificial Moral Assistants 

**Title (ZH)**: 超越伦理对齐：评估LLM作为人工道德助手的有效性 

**Authors**: Alessio Galatolo, Luca Alberto Rappuoli, Katie Winkle, Meriem Beloucif  

**Link**: [PDF](https://arxiv.org/pdf/2508.12754)  

**Abstract**: The recent rise in popularity of large language models (LLMs) has prompted considerable concerns about their moral capabilities. Although considerable effort has been dedicated to aligning LLMs with human moral values, existing benchmarks and evaluations remain largely superficial, typically measuring alignment based on final ethical verdicts rather than explicit moral reasoning. In response, this paper aims to advance the investigation of LLMs' moral capabilities by examining their capacity to function as Artificial Moral Assistants (AMAs), systems envisioned in the philosophical literature to support human moral deliberation. We assert that qualifying as an AMA requires more than what state-of-the-art alignment techniques aim to achieve: not only must AMAs be able to discern ethically problematic situations, they should also be able to actively reason about them, navigating between conflicting values outside of those embedded in the alignment phase. Building on existing philosophical literature, we begin by designing a new formal framework of the specific kind of behaviour an AMA should exhibit, individuating key qualities such as deductive and abductive moral reasoning. Drawing on this theoretical framework, we develop a benchmark to test these qualities and evaluate popular open LLMs against it. Our results reveal considerable variability across models and highlight persistent shortcomings, particularly regarding abductive moral reasoning. Our work connects theoretical philosophy with practical AI evaluation while also emphasising the need for dedicated strategies to explicitly enhance moral reasoning capabilities in LLMs. Code available at this https URL 

**Abstract (ZH)**: 最近大型语言模型（LLMs） popularity 的上升引发了对其道德能力的广泛关注。尽管已经付出 considerable 努力将 LLMs 与人类道德价值观对齐，但现有的基准测试和评估仍然主要停留在表面，通常基于最终的道德判决来衡量对齐程度，而非明确的道德推理。为应对这一挑战，本文旨在通过探讨 LLMs 作为人造道德助手（AMAs）的能力来推进其道德能力的研究，AMAs 是哲学文献中设想的支持人类道德 deliberation 的系统。我们认为，成为 AMA 不仅需要最先进的对齐技术所能达到的：AMAs 不仅必须能够识别出道德上存在问题的情境，还必须能够积极地对这些情境进行推理，权衡超出对齐阶段嵌入的价值观。基于现有的哲学文献，我们首先设计了一个新的正式框架，规定 AMA 应表现出的具体行为类型，确定关键特质如演绎和 abduction 归纳道德推理。利用该理论框架，我们开发了一个基准测试来检验这些特质，并将流行的开源 LLMs 在此基准上进行评估。我们的结果显示出模型之间显著差异，并突出了 abductive 归纳道德推理方面的持续不足。我们的工作将理论哲学与实践 AI 评估相结合，同时强调了需要专门策略来显式增强 LLMs 的道德推理能力。代码可在以下链接获取：this https URL 

---
# GTool: Graph Enhanced Tool Planning with Large Language Model 

**Title (ZH)**: GTool: 图增强工具规划与大规模语言模型 

**Authors**: Wenjie Chen, Wenbin Li, Di Yao, Xuying Meng, Chang Gong, Jingping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2508.12725)  

**Abstract**: Tool planning with large language models (LLMs), referring to selecting, organizing, and preparing the tools necessary to complete a user request, bridges the gap between natural language understanding and task execution. However, current works treat different tools as isolated components and fail to leverage the inherent dependencies of tools, leading to invalid planning results. Since tool dependencies are often incomplete, it becomes challenging for LLMs to accurately identify the appropriate tools required by a user request, especially when confronted with a large toolset. To solve this challenge, we propose \texttt{GTool}, which is the first work aiming to enhance the tool planning ability of LLMs under incomplete dependencies. \texttt{GTool} constructs a request-specific tool graph to select tools efficiently and generate the \texttt{<graph token>} which provides sufficient dependency information understandable by LLMs. Moreover, a missing dependency prediction task is designed to improve the reliability of \texttt{GTool} with incomplete dependencies. Without trimming LLMs, \texttt{GTool} can be seamlessly integrated with various LLM backbones without extensive retraining. Extensive experiments show that \texttt{GTool} achieves more than 29.6\% performance improvements compared with the state-of-the-art (SOTA) baselines with a light-weight (7B) LLM backbone. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）进行工具规划 

---
# EGOILLUSION: Benchmarking Hallucinations in Egocentric Video Understanding 

**Title (ZH)**: 自我中心视频理解中的幻觉 benchmarks：评估自我中心视频理解中的幻觉 

**Authors**: Ashish Seth, Utkarsh Tyagi, Ramaneswaran Selvakumar, Nishit Anand, Sonal Kumar, Sreyan Ghosh, Ramani Duraiswami, Chirag Agarwal, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2508.12687)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable performance in complex multimodal tasks. While MLLMs excel at visual perception and reasoning in third-person and egocentric videos, they are prone to hallucinations, generating coherent yet inaccurate responses. We present EgoIllusion, a first benchmark to evaluate MLLM hallucinations in egocentric videos. EgoIllusion comprises 1,400 videos paired with 8,000 human-annotated open and closed-ended questions designed to trigger hallucinations in both visual and auditory cues in egocentric videos. Evaluations across ten MLLMs reveal significant challenges, including powerful models like GPT-4o and Gemini, achieving only 59% accuracy. EgoIllusion lays the foundation in developing robust benchmarks to evaluate the effectiveness of MLLMs and spurs the development of better egocentric MLLMs with reduced hallucination rates. Our benchmark will be open-sourced for reproducibility. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在复杂多模态任务中展现了杰出的表现。尽管MLLMs在第三人称和主观视角视频中的视觉感知和推理方面表现出色，但它们容易产生幻觉，生成连贯但不准确的响应。我们提出了EgoIllusion，这是一个首个用于评估MLLM幻觉的基准，特别是在主观视角视频中。EgoIllusion包含1400个视频，配以8000个人工标注的开放性和封闭性问题，旨在触发主观视角视频中的视觉和听觉线索中的幻觉。针对十个MLLM的评估揭示了显著的挑战，包括如GPT-4o和Gemini这样强大的模型，也只能达到59%的准确率。EgoIllusion为开发更健壮的基准奠定基础，用于评估MLLM的效果，并推动减少幻觉率的更好主观视角MLLM的发展。我们的基准将开源以确保可重复性。 

---
# GridCodex: A RAG-Driven AI Framework for Power Grid Code Reasoning and Compliance 

**Title (ZH)**: GridCodex：一种基于RAG的电力电网代码推理与合规AI框架 

**Authors**: Jinquan Shi, Yingying Cheng, Fan Zhang, Miao Jiang, Jun Lin, Yanbai Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.12682)  

**Abstract**: The global shift towards renewable energy presents unprecedented challenges for the electricity industry, making regulatory reasoning and compliance increasingly vital. Grid codes, the regulations governing grid operations, are complex and often lack automated interpretation solutions, which hinders industry expansion and undermines profitability for electricity companies. We introduce GridCodex, an end to end framework for grid code reasoning and compliance that leverages large language models and retrieval-augmented generation (RAG). Our framework advances conventional RAG workflows through multi stage query refinement and enhanced retrieval with RAPTOR. We validate the effectiveness of GridCodex with comprehensive benchmarks, including automated answer assessment across multiple dimensions and regulatory agencies. Experimental results showcase a 26.4% improvement in answer quality and more than a 10 fold increase in recall rate. An ablation study further examines the impact of base model selection. 

**Abstract (ZH)**: 全球可再生能源转型给电力行业带来了前所未有的挑战，使得监管推理和合规性愈加重要。网码，规范电网操作的法规，通常复杂且缺乏自动解析解决方案，这阻碍了行业的扩张并削弱了电力公司的盈利能力。我们提出GridCodex，一个基于大型语言模型和检索增强生成（RAG）的端到端框架，用于电网代码推理和合规性。该框架通过多阶段查询精炼和增强检索RAPTOR优化了传统的RAG工作流。我们通过全面基准测试验证了GridCodex的有效性，包括多维度和监管机构的自动答案评估。实验结果展示了答案质量提高了26.4%，召回率提高了超过十倍。消融研究进一步探讨了基础模型选择的影响。 

---
# The Maximum Coverage Model and Recommendation System for UAV Vertiports Location Planning 

**Title (ZH)**: 基于最大覆盖模型的无人机 vertiports 位置规划推荐系统 

**Authors**: Chunliang Hua, Xiao Hu, Jiayang Sun, Zeyuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12651)  

**Abstract**: As urban aerial mobility (UAM) infrastructure development accelerates globally, cities like Shenzhen are planning large-scale vertiport networks (e.g., 1,200+ facilities by 2026). Existing planning frameworks remain inadequate for this complexity due to historical limitations in data granularity and real-world applicability. This paper addresses these gaps by first proposing the Capacitated Dynamic Maximum Covering Location Problem (CDMCLP), a novel optimization framework that simultaneously models urban-scale spatial-temporal demand, heterogeneous user behaviors, and infrastructure capacity constraints. Building on this foundation, we introduce an Integrated Planning Recommendation System that combines CDMCLP with socio-economic factors and dynamic clustering initialization. This system leverages adaptive parameter tuning based on empirical user behavior to generate practical planning solutions. Validation in a Chinese center city demonstrates the effectiveness of the new optimization framework and recommendation system. Under the evaluation and optimization of CDMCLP, the quantitative performance of traditional location methods are exposed and can be improved by 38\%--52\%, while the recommendation system shows user-friendliness and the effective integration of complex elements. By integrating mathematical rigor with practical implementation considerations, this hybrid approach bridges the gap between theoretical location modeling and real-world UAM infrastructure planning, offering municipalities a pragmatic tool for vertiport network design. 

**Abstract (ZH)**: 随着全球城市空中移动（UAM）基础设施的发展加速，如深圳市正在规划大规模的垂直起降机场网络（例如，到2026年将达到1,200多个设施）。现有规划框架由于历史上的数据粒度和现实适用性限制而不足以应对这种复杂性。本文通过首先提出受限动态最大覆盖定位问题（CDMCLP），一种新型优化框架来同时建模大规模城市时空需求、异质用户行为及基础设施容量约束来填补这些空白。在此基础上，我们引入了一个集成规划推荐系统，该系统结合了CDMCLP与社会经济因素和动态聚类初始化。该系统利用基于经验用户行为的自适应参数调整来生成实用的规划解决方案。在中国中部城市进行验证表明，新的优化框架和推荐系统的有效性。在CDMCLP的评价和优化下，传统定位方法的数量性能能够提高38%-52%，而推荐系统展示了用户友好性和复杂元素的有效整合。通过结合数学严谨性和实际实施考虑，这种混合方法弥合了理论定位建模与实际UAM基础设施规划之间的差距，为市政府提供了一种实用工具来设计垂直起降机场网络。 

---
# Cognitive Structure Generation: From Educational Priors to Policy Optimization 

**Title (ZH)**: 认知结构生成：从教育先验到政策优化 

**Authors**: Hengnian Gu, Zhifu Chen, Yuxin Chen, Jin Peng Zhou, Dongdai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.12647)  

**Abstract**: Cognitive structure is a student's subjective organization of an objective knowledge system, reflected in the psychological construction of concepts and their relations. However, cognitive structure assessment remains a long-standing challenge in student modeling and psychometrics, persisting as a foundational yet largely unassessable concept in educational practice. This paper introduces a novel framework, Cognitive Structure Generation (CSG), in which we first pretrain a Cognitive Structure Diffusion Probabilistic Model (CSDPM) to generate students' cognitive structures from educational priors, and then further optimize its generative process as a policy with hierarchical reward signals via reinforcement learning to align with genuine cognitive development levels during students' learning processes. Experimental results on four popular real-world education datasets show that cognitive structures generated by CSG offer more comprehensive and effective representations for student modeling, substantially improving performance on KT and CD tasks while enhancing interpretability. 

**Abstract (ZH)**: 认知结构生成：一种新颖的学生认知结构评估框架 

---
# An LLM + ASP Workflow for Joint Entity-Relation Extraction 

**Title (ZH)**: 基于LLM+ASP的工作流联合实体-关系提取 

**Authors**: Trang Tran, Trung Hoang Le, Huiping Cao, Tran Cao Son  

**Link**: [PDF](https://arxiv.org/pdf/2508.12611)  

**Abstract**: Joint entity-relation extraction (JERE) identifies both entities and their relationships simultaneously. Traditional machine-learning based approaches to performing this task require a large corpus of annotated data and lack the ability to easily incorporate domain specific information in the construction of the model. Therefore, creating a model for JERE is often labor intensive, time consuming, and elaboration intolerant. In this paper, we propose harnessing the capabilities of generative pretrained large language models (LLMs) and the knowledge representation and reasoning capabilities of Answer Set Programming (ASP) to perform JERE. We present a generic workflow for JERE using LLMs and ASP. The workflow is generic in the sense that it can be applied for JERE in any domain. It takes advantage of LLM's capability in natural language understanding in that it works directly with unannotated text. It exploits the elaboration tolerant feature of ASP in that no modification of its core program is required when additional domain specific knowledge, in the form of type specifications, is found and needs to be used. We demonstrate the usefulness of the proposed workflow through experiments with limited training data on three well-known benchmarks for JERE. The results of our experiments show that the LLM + ASP workflow is better than state-of-the-art JERE systems in several categories with only 10\% of training data. It is able to achieve a 2.5 times (35\% over 15\%) improvement in the Relation Extraction task for the SciERC corpus, one of the most difficult benchmarks. 

**Abstract (ZH)**: 联合实体-关系抽取中的生成预训练大型语言模型与答案集编程融合方法 

---
# Help or Hurdle? Rethinking Model Context Protocol-Augmented Large Language Models 

**Title (ZH)**: 助益还是阻碍？重新思考模型上下文协议增强的大语言模型 

**Authors**: Wei Song, Haonan Zhong, Ziqi Ding, Jingling Xue, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.12566)  

**Abstract**: The Model Context Protocol (MCP) enables large language models (LLMs) to access external resources on demand. While commonly assumed to enhance performance, how LLMs actually leverage this capability remains poorly understood. We introduce MCPGAUGE, the first comprehensive evaluation framework for probing LLM-MCP interactions along four key dimensions: proactivity (self-initiated tool use), compliance (adherence to tool-use instructions), effectiveness (task performance post-integration), and overhead (computational cost incurred). MCPGAUGE comprises a 160-prompt suite and 25 datasets spanning knowledge comprehension, general reasoning, and code generation. Our large-scale evaluation, spanning six commercial LLMs, 30 MCP tool suites, and both one- and two-turn interaction settings, comprises around 20,000 API calls and over USD 6,000 in computational cost. This comprehensive study reveals four key findings that challenge prevailing assumptions about the effectiveness of MCP integration. These insights highlight critical limitations in current AI-tool integration and position MCPGAUGE as a principled benchmark for advancing controllable, tool-augmented LLMs. 

**Abstract (ZH)**: MCPGAUGE：探索大型语言模型与Model Context Protocol交互的关键维度 

---
# Root Cause Analysis of Hydrogen Bond Separation in Spatio-Temporal Molecular Dynamics using Causal Models 

**Title (ZH)**: 基于因果模型的时空分子动力学中氢键分离根本原因分析 

**Authors**: Rahmat K. Adesunkanmi, Ashfaq Khokhar, Goce Trajcevski, Sohail Murad  

**Link**: [PDF](https://arxiv.org/pdf/2508.12500)  

**Abstract**: Molecular dynamics simulations (MDS) face challenges, including resource-heavy computations and the need to manually scan outputs to detect "interesting events," such as the formation and persistence of hydrogen bonds between atoms of different molecules. A critical research gap lies in identifying the underlying causes of hydrogen bond formation and separation -understanding which interactions or prior events contribute to their emergence over time. With this challenge in mind, we propose leveraging spatio-temporal data analytics and machine learning models to enhance the detection of these phenomena. In this paper, our approach is inspired by causal modeling and aims to identify the root cause variables of hydrogen bond formation and separation events. Specifically, we treat the separation of hydrogen bonds as an "intervention" occurring and represent the causal structure of the bonding and separation events in the MDS as graphical causal models. These causal models are built using a variational autoencoder-inspired architecture that enables us to infer causal relationships across samples with diverse underlying causal graphs while leveraging shared dynamic information. We further include a step to infer the root causes of changes in the joint distribution of the causal models. By constructing causal models that capture shifts in the conditional distributions of molecular interactions during bond formation or separation, this framework provides a novel perspective on root cause analysis in molecular dynamic systems. We validate the efficacy of our model empirically on the atomic trajectories that used MDS for chiral separation, demonstrating that we can predict many steps in the future and also find the variables driving the observed changes in the system. 

**Abstract (ZH)**: 分子动力学模拟中的时空数据分析和机器学习模型在识别氢键形成和分离的根本原因中的应用 

---
# Advanced DOA Regulation with a Whale-Optimized Fractional Order Fuzzy PID Framework 

**Title (ZH)**: 基于鲸鱼优化分数阶模糊PID框架的高级方向角调节 

**Authors**: Lida Shahbandari, Hossein Mohseni  

**Link**: [PDF](https://arxiv.org/pdf/2508.12487)  

**Abstract**: This study introduces a Fractional Order Fuzzy PID (FOFPID) controller that uses the Whale Optimization Algorithm (WOA) to manage the Bispectral Index (BIS), keeping it within the ideal range of forty to sixty. The FOFPID controller combines fuzzy logic for adapting to changes and fractional order dynamics for fine tuning. This allows it to adjust its control gains to handle a person's unique physiology. The WOA helps fine tune the controller's parameters, including the fractional orders and the fuzzy membership functions, which boosts its performance. Tested on models of eight different patient profiles, the FOFPID controller performed better than a standard Fractional Order PID (FOPID) controller. It achieved faster settling times, at two and a half minutes versus three point two minutes, and had a lower steady state error, at zero point five versus one point two. These outcomes show the FOFPID's excellent strength and accuracy. It offers a scalable, artificial intelligence driven solution for automated anesthesia delivery that could enhance clinical practice and improve patient results. 

**Abstract (ZH)**: 一种用于保持Bispectral Index在四十到六十理想范围内的分数阶模糊PID控制策略及其 Whale 优化算法参数调整研究 

---
# The Yokai Learning Environment: Tracking Beliefs Over Space and Time 

**Title (ZH)**: yokai 学习环境：跨空间与时间追踪信念 

**Authors**: Constantin Ruhdorfer, Matteo Bortoletto, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2508.12480)  

**Abstract**: Developing collaborative AI hinges on Theory of Mind (ToM) - the ability to reason about the beliefs of others to build and maintain common ground. Existing ToM benchmarks, however, are restricted to passive observer settings or lack an assessment of how agents establish and maintain common ground over time. To address these gaps, we introduce the Yokai Learning Environment (YLE) - a multi-agent reinforcement learning (RL) environment based on the cooperative card game Yokai. In the YLE, agents take turns peeking at hidden cards and moving them to form clusters based on colour. Success requires tracking evolving beliefs, remembering past observations, using hints as grounded communication, and maintaining common ground with teammates. Our evaluation yields two key findings: First, current RL agents struggle to solve the YLE, even when given access to perfect memory. Second, while belief modelling improves performance, agents are still unable to effectively generalise to unseen partners or form accurate beliefs over longer games, exposing a reliance on brittle conventions rather than robust belief tracking. We use the YLE to investigate research questions in belief modelling, memory, partner generalisation, and scaling to higher-order ToM. 

**Abstract (ZH)**: 基于理论心智的协作AI发展依赖于理解他人信念的能力——以建立和维持共同知识为目标。现有的理论心智基准测试局限于被动观察者的设置，或者未能评估代理如何在时间上建立和维持共同知识。为填补这些空白，我们引入了Yokai学习环境（YLE）——基于合作纸牌游戏Yokai的多代理强化学习（RL）环境。在YLE中，代理轮流查看隐藏的卡片并将它们移动以根据颜色形成集群。成功需要跟踪不断变化的信念、记住过去的观察、利用提示进行基于事实的通信，并与队友保持共同知识。我们的评估得出了两个关键发现：首先，现有的RL代理即使有完美的记忆也无法解决YLE。其次，尽管信念建模可以提高性能，但代理仍然无法有效地泛化到未见过的队友或在较长的游戏过程中形成准确的信念，暴露了对脆弱惯例的依赖而非稳健的信念跟踪。我们使用YLE探索信念建模、记忆、伙伴泛化和向高级理论心智扩展的研究问题。 

---
# GALA: Can Graph-Augmented Large Language Model Agentic Workflows Elevate Root Cause Analysis? 

**Title (ZH)**: GALA：图增强的大语言模型在根本原因分析中的代理工作流程能否提升效果？ 

**Authors**: Yifang Tian, Yaming Liu, Zichun Chong, Zihang Huang, Hans-Arno Jacobsen  

**Link**: [PDF](https://arxiv.org/pdf/2508.12472)  

**Abstract**: Root cause analysis (RCA) in microservice systems is challenging, requiring on-call engineers to rapidly diagnose failures across heterogeneous telemetry such as metrics, logs, and traces. Traditional RCA methods often focus on single modalities or merely rank suspect services, falling short of providing actionable diagnostic insights with remediation guidance. This paper introduces GALA, a novel multi-modal framework that combines statistical causal inference with LLM-driven iterative reasoning for enhanced RCA. Evaluated on an open-source benchmark, GALA achieves substantial improvements over state-of-the-art methods of up to 42.22% accuracy. Our novel human-guided LLM evaluation score shows GALA generates significantly more causally sound and actionable diagnostic outputs than existing methods. Through comprehensive experiments and a case study, we show that GALA bridges the gap between automated failure diagnosis and practical incident resolution by providing both accurate root cause identification and human-interpretable remediation guidance. 

**Abstract (ZH)**: 基于微服务系统的根本原因分析（RCA）具有挑战性，要求当班工程师能够快速诊断异构 telemetry（指标、日志和跟踪）中的故障。传统的方法往往集中在单一模态上，或是仅仅对可疑服务进行排名，缺乏提供具有修复指导的实际诊断见解的能力。本文介绍了一种名为 GALA 的新型多模态框架，该框架结合了统计因果推理与 LLM 驱动的迭代推理，以增强 RCA。GALA 在开源基准测试中比最先进的方法准确性提升高达 42.22%。我们的新型人工指导的 LLM 评估得分表明，GALA 生成的因果推理更为准确且更具实际操作性的诊断输出显著优于现有方法。通过全面的实验和案例研究，我们展示了 GALA 如何通过提供准确的根本原因识别和可由人类解释的修复指导，弥合自动化故障诊断与实际事故解决之间的差距。 

---
# Non-Iterative Symbolic-Aided Chain-of-Thought for Logical Reasoning 

**Title (ZH)**: 非迭代符号辅助思考链逻辑推理 

**Authors**: Phuong Minh Nguyen, Tien Huu Dang, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2508.12425)  

**Abstract**: This work introduces Symbolic-Aided Chain-of-Thought (CoT), an improved approach to standard CoT, for logical reasoning in large language models (LLMs). The key idea is to integrate lightweight symbolic representations into few-shot prompts, structuring the inference steps with a consistent strategy to make reasoning patterns more explicit within a non-iterative reasoning process. By incorporating these symbolic structures, our method preserves the generalizability of standard prompting techniques while enhancing the transparency, interpretability, and analyzability of LLM logical reasoning. Extensive experiments on four well-known logical reasoning benchmarks -- ProofWriter, FOLIO, ProntoQA, and LogicalDeduction, which cover diverse reasoning scenarios -- demonstrate the effectiveness of the proposed approach, particularly in complex reasoning tasks that require navigating multiple constraints or rules. Notably, Symbolic-Aided CoT consistently improves LLMs' reasoning capabilities across various model sizes and significantly outperforms conventional CoT on three out of four datasets, ProofWriter, ProntoQA, and LogicalDeduction. 

**Abstract (ZH)**: 符号辅助链式思考（CoT）：一种改进的大语言模型逻辑推理方法 

---
# GraphCogent: Overcoming LLMs' Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding 

**Title (ZH)**: GraphCogent: 通过复杂图理解中的多 agent 协作克服 LLMs 的工作记忆限制 

**Authors**: Rongzheng Wang, Qizhi Chen, Yihong Huang, Yizhuo Ma, Muquan Li, Jiakai Li, Ke Qin, Guangchun Luo, Shuang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12379)  

**Abstract**: Large language models (LLMs) show promising performance on small-scale graph reasoning tasks but fail when handling real-world graphs with complex queries. This phenomenon stems from LLMs' inability to effectively process complex graph topology and perform multi-step reasoning simultaneously. To address these limitations, we propose GraphCogent, a collaborative agent framework inspired by human Working Memory Model that decomposes graph reasoning into specialized cognitive processes: sense, buffer, and execute. The framework consists of three modules: Sensory Module standardizes diverse graph text representations via subgraph sampling, Buffer Module integrates and indexes graph data across multiple formats, and Execution Module combines tool calling and model generation for efficient reasoning. We also introduce Graph4real, a comprehensive benchmark contains with four domains of real-world graphs (Web, Social, Transportation, and Citation) to evaluate LLMs' graph reasoning capabilities. Our Graph4real covers 21 different graph reasoning tasks, categorized into three types (Structural Querying, Algorithmic Reasoning, and Predictive Modeling tasks), with graph scales that are 10 times larger than existing benchmarks. Experiments show that Llama3.1-8B based GraphCogent achieves a 50% improvement over massive-scale LLMs like DeepSeek-R1 (671B). Compared to state-of-the-art agent-based baseline, our framework outperforms by 20% in accuracy while reducing token usage by 80% for in-toolset tasks and 30% for out-toolset tasks. Code will be available after review. 

**Abstract (ZH)**: 大型语言模型在处理小型图推理任务上表现出色，但在处理具有复杂查询的现实世界图时却失败。这一现象源于大型语言模型无法有效处理复杂的图拓扑结构并同时进行多步推理。为解决这些问题，我们提出了GraphCogent，一种受人类工作记忆模型启发的合作代理框架，将图推理分解为专门的认知过程：感知、缓冲和执行。该框架由三个模块组成：感知模块通过子图抽样标准化各种图文本表示，缓冲模块整合并跨多种格式索引图数据，执行模块结合工具调用和模型生成，以实现高效的推理。我们还引入了Graph4real，这是一个包含四个现实世界图领域的综合基准，用于评估大型语言模型的图推理能力。Graph4real涵盖了21种不同的图推理任务，分为三类（结构查询、算法推理和预测建模任务），其图规模比现有基准大10倍。实验结果显示，基于Llama3.1-8B的GraphCogent在大规模语言模型如DeepSeek-R1（671B）上实现了50%的性能提升。与最先进的基于代理的基线相比，在工具集内任务中，我们的框架在准确率上提高了20%，同时在工具集内任务中减少了80%的token使用量，在工具集外任务中减少了30%的token使用量。 

---
# Hierarchical knowledge guided fault intensity diagnosis of complex industrial systems 

**Title (ZH)**: 复杂工业系统分层知识指导的故障强度诊断 

**Authors**: Yu Sha, Shuiping Gou, Bo Liu, Johannes Faber, Ningtao Liu, Stefan Schramm, Horst Stoecker, Thomas Steckenreiter, Domagoj Vnucec, Nadine Wetzstein, Andreas Widl, Kai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.12375)  

**Abstract**: Fault intensity diagnosis (FID) plays a pivotal role in monitoring and maintaining mechanical devices within complex industrial systems. As current FID methods are based on chain of thought without considering dependencies among target classes. To capture and explore dependencies, we propose a hierarchical knowledge guided fault intensity diagnosis framework (HKG) inspired by the tree of thought, which is amenable to any representation learning methods. The HKG uses graph convolutional networks to map the hierarchical topological graph of class representations into a set of interdependent global hierarchical classifiers, where each node is denoted by word embeddings of a class. These global hierarchical classifiers are applied to learned deep features extracted by representation learning, allowing the entire model to be end-to-end learnable. In addition, we develop a re-weighted hierarchical knowledge correlation matrix (Re-HKCM) scheme by embedding inter-class hierarchical knowledge into a data-driven statistical correlation matrix (SCM) which effectively guides the information sharing of nodes in graphical convolutional neural networks and avoids over-smoothing issues. The Re-HKCM is derived from the SCM through a series of mathematical transformations. Extensive experiments are performed on four real-world datasets from different industrial domains (three cavitation datasets from SAMSON AG and one existing publicly) for FID, all showing superior results and outperform recent state-of-the-art FID methods. 

**Abstract (ZH)**: 基于思维树的层次知识引导故障强度诊断框架（HKG） 

---
# Wisdom of the Crowd: Reinforcement Learning from Coevolutionary Collective Feedback 

**Title (ZH)**: 群体的智慧：共生进化集体反馈强化学习 

**Authors**: Wenzhen Yuan, Shengji Tang, Weihao Lin, Jiacheng Ruan, Ganqu Cui, Bo Zhang, Tao Chen, Ting Liu, Yuzhuo Fu, Peng Ye, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2508.12338)  

**Abstract**: Reinforcement learning (RL) has significantly enhanced the reasoning capabilities of large language models (LLMs), but its reliance on expensive human-labeled data or complex reward models severely limits scalability. While existing self-feedback methods aim to address this problem, they are constrained by the capabilities of a single model, which can lead to overconfidence in incorrect answers, reward hacking, and even training collapse. To this end, we propose Reinforcement Learning from Coevolutionary Collective Feedback (RLCCF), a novel RL framework that enables multi-model collaborative evolution without external supervision. Specifically, RLCCF optimizes the ability of a model collective by maximizing its Collective Consistency (CC), which jointly trains a diverse ensemble of LLMs and provides reward signals by voting on collective outputs. Moreover, each model's vote is weighted by its Self-Consistency (SC) score, ensuring that more confident models contribute more to the collective decision. Benefiting from the diverse output distributions and complementary abilities of multiple LLMs, RLCCF enables the model collective to continuously enhance its reasoning ability through coevolution. Experiments on four mainstream open-source LLMs across four mathematical reasoning benchmarks demonstrate that our framework yields significant performance gains, achieving an average relative improvement of 16.72\% in accuracy. Notably, RLCCF not only improves the performance of individual models but also enhances the group's majority-voting accuracy by 4.51\%, demonstrating its ability to extend the collective capability boundary of the model collective. 

**Abstract (ZH)**: 强化学习从协同进化群体反馈中学习（基于多模型协作进化的强化学习框架） 

---
# RadarQA: Multi-modal Quality Analysis of Weather Radar Forecasts 

**Title (ZH)**: 雷达QA：天气雷达预报的多模态质量分析 

**Authors**: Xuming He, Zhiyuan You, Junchao Gong, Couhua Liu, Xiaoyu Yue, Peiqin Zhuang, Wenlong Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2508.12291)  

**Abstract**: Quality analysis of weather forecasts is an essential topic in meteorology. Although traditional score-based evaluation metrics can quantify certain forecast errors, they are still far from meteorological experts in terms of descriptive capability, interpretability, and understanding of dynamic evolution. With the rapid development of Multi-modal Large Language Models (MLLMs), these models become potential tools to overcome the above challenges. In this work, we introduce an MLLM-based weather forecast analysis method, RadarQA, integrating key physical attributes with detailed assessment reports. We introduce a novel and comprehensive task paradigm for multi-modal quality analysis, encompassing both single frame and sequence, under both rating and assessment scenarios. To support training and benchmarking, we design a hybrid annotation pipeline that combines human expert labeling with automated heuristics. With such an annotation method, we construct RQA-70K, a large-scale dataset with varying difficulty levels for radar forecast quality evaluation. We further design a multi-stage training strategy that iteratively improves model performance at each stage. Extensive experiments show that RadarQA outperforms existing general MLLMs across all evaluation settings, highlighting its potential for advancing quality analysis in weather prediction. 

**Abstract (ZH)**: 基于多模态大型语言模型的雷达天气预报质量分析方法：RadarQA 

---
# Mantis: A Simulation-Grounded Foundation Model for Disease Forecasting 

**Title (ZH)**: 螳螂：基于模拟的疾病预测基础模型 

**Authors**: Carson Dudley, Reiden Magdaleno, Christopher Harding, Ananya Sharma, Emily Martin, Marisa Eisenberg  

**Link**: [PDF](https://arxiv.org/pdf/2508.12260)  

**Abstract**: Infectious disease forecasting in novel outbreaks or low resource settings has been limited by the need for disease-specific data, bespoke training, and expert tuning. We introduce Mantis, a foundation model trained entirely on mechanistic simulations, which enables out-of-the-box forecasting across diseases, regions, and outcomes, even in settings with limited historical data. Mantis is built on over 400 million simulated days of outbreak dynamics spanning diverse pathogens, transmission modes, interventions, and surveillance artifacts. Despite requiring no real-world data during training, Mantis outperformed 39 expert-tuned models we tested across six diseases, including all models in the CDC's COVID-19 Forecast Hub. Mantis generalized to novel epidemiological regimes, including diseases with held-out transmission mechanisms, demonstrating that it captures fundamental contagion dynamics. Critically, Mantis is mechanistically interpretable, enabling public health decision-makers to identify the latent drivers behind its predictions. Finally, Mantis delivers accurate forecasts at 8-week horizons, more than doubling the actionable range of most models, enabling proactive public health planning. Together, these capabilities position Mantis as a foundation for next-generation disease forecasting systems: general, interpretable, and deployable where traditional models fail. 

**Abstract (ZH)**: 新型疫情或资源有限环境下传染病预报受限于疾病特异性数据、定制训练和专家调优的需求。我们引入Mantis，这是一种完全基于机理模拟训练的基础模型，能够在疾病、地区和结局之间实现开箱即用的预报，即使在历史数据有限的环境中也是如此。Mantis 基于超过4亿个模拟疫情动态日的数据，涵盖多种病原体、传播模式、干预措施和监测 artefacts。尽管在训练过程中未使用任何真实世界数据，Mantis 在我们测试的六种疾病中均优于39个专家调优模型，包括CDC COVID-19 预测 hub 中的所有模型。Mantis 能够泛化到新型的流行病学模式中，包括测试中排除的传播机制，这表明它捕获了根本的传染动态规律。关键的是，Mantis 具有机理可解释性，使公共卫生决策者能够识别其预测背后的潜在驱动因素。此外，Mantis 在8周预报范围内的准确率超过其他大多数模型两倍，使公共卫生规划更具前瞻性。这些能力使Mantis 成为下一代传染病预报系统的基石：普遍适用、可解释且能在传统模型失效的地方部署。 

---
# RLNVR: Reinforcement Learning from Non-Verified Real-World Rewards 

**Title (ZH)**: RLNVR: 基于非验证真实世界奖励的强化学习 

**Authors**: Rohit Krishnan, Jon Evans  

**Link**: [PDF](https://arxiv.org/pdf/2508.12165)  

**Abstract**: This paper introduces RLNVR (Reinforcement Learning from Non-Verified Rewards), a framework for training language models using noisy, real-world feedback signals without requiring explicit human verification. Traditional RLHF requires expensive, verified reward signals that are impractical in many real-world domains. RLNVR addresses this challenge through baseline normalization and semantic similarity-based reward transfer. We demonstrate RLNVR through Walter, a prototype system that optimizes social media content generation using actual engagement data from Bluesky. Our experimental results show significant improvements in content quality and training stability, with comprehensive evaluation planned for future work. Positioning: We present a practical framework that combines RLNVR with GSPO (Group Sequence Policy Optimization) and an optional UED (Unsupervised Environment Design) curriculum to improve stability and diversity under noisy, implicit rewards. To our knowledge, combining GSPO-style normalization with a UED-style curriculum for LLM content generation from implicit social engagement has not been previously documented in this applied setting; we frame this as an applied integration rather than a new algorithm. 

**Abstract (ZH)**: 基于噪声反馈的强化学习语言模型训练框架：RLNVR 

---
# MOVER: Multimodal Optimal Transport with Volume-based Embedding Regularization 

**Title (ZH)**: MOVER：基于体素嵌入正则化的多模态最优传输 

**Authors**: Haochen You, Baojing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12149)  

**Abstract**: Recent advances in multimodal learning have largely relied on pairwise contrastive objectives to align different modalities, such as text, video, and audio, in a shared embedding space. While effective in bi-modal setups, these approaches struggle to generalize across multiple modalities and often lack semantic structure in high-dimensional spaces. In this paper, we propose MOVER, a novel framework that combines optimal transport-based soft alignment with volume-based geometric regularization to build semantically aligned and structured multimodal representations. By integrating a transport-guided matching mechanism with a geometric volume minimization objective (GAVE), MOVER encourages consistent alignment across all modalities in a modality-agnostic manner. Experiments on text-video-audio retrieval tasks demonstrate that MOVER significantly outperforms prior state-of-the-art methods in both zero-shot and finetuned settings. Additional analysis shows improved generalization to unseen modality combinations and stronger structural consistency in the learned embedding space. 

**Abstract (ZH)**: 近期，多模态学习的进步主要依赖于成对对比目标在共享嵌入空间中对齐不同模态（如文本、视频和音频）的能力。虽然在双模态配置中有效，但这些方法在跨多种模态时难以泛化，并且在高维空间中往往缺乏语义结构。本文提出了一种名为MOVER的新框架，该框架结合了最优传输为基础的软对齐与基于体积的几何正则化，以构建语义对齐且结构化的多模态表示。通过集成运输引导的匹配机制与几何体积最小化目标（GAVE），MOVER以模态无关的方式促进了所有模态的一致对齐。在文本-视频-音频检索任务上的实验表明，MOVER在零样本和微调设置中均显著优于先前的最佳方法。进一步的分析显示，MOVER在未见过的模态组合泛化方面表现更好，并且在学习的嵌入空间中具有更强的结构一致性。 

---
# Overcoming Knowledge Discrepancies: Structuring Reasoning Threads through Knowledge Balancing in Interactive Scenarios 

**Title (ZH)**: 克服知识 discrepancies：在互动场景中通过知识平衡构建推理线索 

**Authors**: Daniel Burkhardt, Xiangwei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.12100)  

**Abstract**: Reasoning in interactive problem solving scenarios requires models to construct reasoning threads that reflect user understanding and align with structured domain knowledge. However, current reasoning models often lack explicit semantic hierarchies, user-domain knowledge alignment, and principled mechanisms to prune reasoning threads for effectiveness. These limitations result in lengthy generic output that does not guide users through goal-oriented reasoning steps. To address this, we propose a prototype-inspired, two-phases Reasoning-Threads-Evaluation (ReT-Eval) framework, drawing inspiration from human-like reasoning strategies that emphasize structured knowledge reuse. In the first phase, semantically relevant knowledge structures are extracted from a sparse domain knowledge graph using a graph neural network and enriched with intrinsic large language model knowledge to resolve knowledge discrepancies. In the second phase, these threads are evaluated and pruned using a reward-guided strategy aimed at maintaining semantic coherence to generate effective reasoning threads. Experiments and expert evaluations show that ReT-Eval enhances user understanding and outperforms state-of-the-art reasoning models. 

**Abstract (ZH)**: 交互式问题解决场景中的推理需要模型构建反映用户理解并与结构化领域知识对齐的推理线程。然而，当前的推理模型往往缺乏明确的语义层次结构、用户-领域知识对齐以及有效的机制来精简推理线程以提高效果。这些限制导致生成冗长的泛化输出，未能引导用户通过目标导向的推理步骤。为解决这一问题，我们提出了一种原型启发的两阶段推理线程评估 (ReT-Eval) 框架，该框架借鉴了强调结构化知识重用的人类推理策略。在第一阶段，使用图神经网络从稀疏领域知识图中提取语义相关知识结构，并通过内嵌的大型语言模型知识来解决知识不一致问题。在第二阶段，这些线程通过一个奖励导向的评估和精简策略来维持语义连贯性，从而生成有效的推理线程。实验和专家评估表明，ReT-Eval 提升了用户理解并优于现有的先进推理模型。 

---
# MAPF-World: Action World Model for Multi-Agent Path Finding 

**Title (ZH)**: MAPF-世界：多智能体路径规划的行为世界模型 

**Authors**: Zhanjiang Yang, Meng Li, Yang Shen, Yueming Li, Lijun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.12087)  

**Abstract**: Multi-agent path finding (MAPF) is the problem of planning conflict-free paths from the designated start locations to goal positions for multiple agents. It underlies a variety of real-world tasks, including multi-robot coordination, robot-assisted logistics, and social navigation. Recent decentralized learnable solvers have shown great promise for large-scale MAPF, especially when leveraging foundation models and large datasets. However, these agents are reactive policy models and exhibit limited modeling of environmental temporal dynamics and inter-agent dependencies, resulting in performance degradation in complex, long-term planning scenarios. To address these limitations, we propose MAPF-World, an autoregressive action world model for MAPF that unifies situation understanding and action generation, guiding decisions beyond immediate local observations. It improves situational awareness by explicitly modeling environmental dynamics, including spatial features and temporal dependencies, through future state and actions prediction. By incorporating these predicted futures, MAPF-World enables more informed, coordinated, and far-sighted decision-making, especially in complex multi-agent settings. Furthermore, we augment MAPF benchmarks by introducing an automatic map generator grounded in real-world scenarios, capturing practical map layouts for training and evaluating MAPF solvers. Extensive experiments demonstrate that MAPF-World outperforms state-of-the-art learnable solvers, showcasing superior zero-shot generalization to out-of-distribution cases. Notably, MAPF-World is trained with a 96.5% smaller model size and 92% reduced data. 

**Abstract (ZH)**: 多智能体路径规划中的自回归动作世界模型（MAPF-World） 

---
# Active inference for action-unaware agents 

**Title (ZH)**: 行动不知情代理的活性推断 

**Authors**: Filippo Torresan, Keisuke Suzuki, Ryota Kanai, Manuel Baltieri  

**Link**: [PDF](https://arxiv.org/pdf/2508.12027)  

**Abstract**: Active inference is a formal approach to study cognition based on the notion that adaptive agents can be seen as engaging in a process of approximate Bayesian inference, via the minimisation of variational and expected free energies. Minimising the former provides an account of perceptual processes and learning as evidence accumulation, while minimising the latter describes how agents select their actions over time. In this way, adaptive agents are able to maximise the likelihood of preferred observations or states, given a generative model of the environment. In the literature, however, different strategies have been proposed to describe how agents can plan their future actions. While they all share the notion that some kind of expected free energy offers an appropriate way to score policies, sequences of actions, in terms of their desirability, there are different ways to consider the contribution of past motor experience to the agent's future behaviour. In some approaches, agents are assumed to know their own actions, and use such knowledge to better plan for the future. In other approaches, agents are unaware of their actions, and must infer their motor behaviour from recent observations in order to plan for the future. This difference reflects a standard point of departure in two leading frameworks in motor control based on the presence, or not, of an efference copy signal representing knowledge about an agent's own actions. In this work we compare the performances of action-aware and action-unaware agents in two navigations tasks, showing how action-unaware agents can achieve performances comparable to action-aware ones while at a severe disadvantage. 

**Abstract (ZH)**: 基于行动感知和行动不知觉代理在两种导航任务中的表现比较：行动不知觉代理如何在不利条件下实现与行动感知代理相当的性能 

---
# Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems 

**Title (ZH)**: Bongard-RWR+：Bongard 问题中精细概念的现实世界表示 

**Authors**: Szymon Pawlonka, Mikołaj Małkiński, Jacek Mańdziuk  

**Link**: [PDF](https://arxiv.org/pdf/2508.12026)  

**Abstract**: Bongard Problems (BPs) provide a challenging testbed for abstract visual reasoning (AVR), requiring models to identify visual concepts fromjust a few examples and describe them in natural language. Early BP benchmarks featured synthetic black-and-white drawings, which might not fully capture the complexity of real-world scenes. Subsequent BP datasets employed real-world images, albeit the represented concepts are identifiable from high-level image features, reducing the task complexity. Differently, the recently released Bongard-RWR dataset aimed at representing abstract concepts formulated in the original BPs using fine-grained real-world images. Its manual construction, however, limited the dataset size to just $60$ instances, constraining evaluation robustness. In this work, we introduce Bongard-RWR+, a BP dataset composed of $5\,400$ instances that represent original BP abstract concepts using real-world-like images generated via a vision language model (VLM) pipeline. Building on Bongard-RWR, we employ Pixtral-12B to describe manually curated images and generate new descriptions aligned with the underlying concepts, use Flux.1-dev to synthesize images from these descriptions, and manually verify that the generated images faithfully reflect the intended concepts. We evaluate state-of-the-art VLMs across diverse BP formulations, including binary and multiclass classification, as well as textual answer generation. Our findings reveal that while VLMs can recognize coarse-grained visual concepts, they consistently struggle with discerning fine-grained concepts, highlighting limitations in their reasoning capabilities. 

**Abstract (ZH)**: Bongard问题（BPs）提供了一种挑战性的抽象视觉推理（AVR）测试平台，要求模型仅从少量例子中识别视觉概念，并用自然语言描述这些概念。早期的BPs基准使用合成的黑白绘制图，可能未能充分捕捉真实世界场景的复杂性。随后的BPs数据集使用了真实世界图像，尽管这些图像中的概念可以从高层图像特征中识别，从而降低了任务的复杂性。不同的是，最近发布的Bongard-RWR数据集旨在通过精细的真实世界图像表示原始BPs中的抽象概念。然而，其手工构建限制了数据集的规模，只有60个实例，影响了评估的 robustness。在这项工作中，我们引入了Bongard-RWR+，这是一个包含5400个实例的BPs数据集，使用视觉语言模型（VLM）流水线生成的类真实世界图像来表示原始BPs的抽象概念。基于Bongard-RWR，我们使用Pixtral-12B描述手工策划的图像并生成与底层概念对齐的新描述，使用Flux.1-dev从这些描述合成图像，并人工验证生成的图像忠实反映了预期的概念。我们评估了最先进的视觉语言模型在多种Bongard问题表示形式上的性能，包括二分类和多分类，以及文本答案生成。我们的研究发现，虽然视觉语言模型能够识别粗粒度的视觉概念，但它们在辨别细粒度概念方面表现不佳，突显了推理能力的局限性。 

---
# AI Models for Depressive Disorder Detection and Diagnosis: A Review 

**Title (ZH)**: AI模型在抑郁障碍检测与诊断中的应用：一项综述 

**Authors**: Dorsa Macky Aleagha, Payam Zohari, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2508.12022)  

**Abstract**: Major Depressive Disorder is one of the leading causes of disability worldwide, yet its diagnosis still depends largely on subjective clinical assessments. Integrating Artificial Intelligence (AI) holds promise for developing objective, scalable, and timely diagnostic tools. In this paper, we present a comprehensive survey of state-of-the-art AI methods for depression detection and diagnosis, based on a systematic review of 55 key studies. We introduce a novel hierarchical taxonomy that structures the field by primary clinical task (diagnosis vs. prediction), data modality (text, speech, neuroimaging, multimodal), and computational model class (e.g., graph neural networks, large language models, hybrid approaches). Our in-depth analysis reveals three major trends: the predominance of graph neural networks for modeling brain connectivity, the rise of large language models for linguistic and conversational data, and an emerging focus on multimodal fusion, explainability, and algorithmic fairness. Alongside methodological insights, we provide an overview of prominent public datasets and standard evaluation metrics as a practical guide for researchers. By synthesizing current advances and highlighting open challenges, this survey offers a comprehensive roadmap for future innovation in computational psychiatry. 

**Abstract (ZH)**: 重大抑郁障碍是全球主要的致残原因，但其诊断仍主要依赖于主观临床评估。 integrates artificial intelligence (ai) 有望促进客观、可扩展和及时的诊断工具的发展。本文综述了55篇关键研究的基础上，系统探讨了最新的ai方法在抑郁检测和诊断中的应用。我们提出了一个新颖的层次分类体系，按主要临床任务（诊断 vs. 预测）、数据模态（文本、语音、神经影像、多模态）和计算模型类别（如图神经网络、大规模语言模型、混合方法）对领域进行结构化。深入分析揭示了三大趋势：脑连接建模中图神经网络的主导地位，大规模语言模型在语言和对话数据中的崛起，以及对多模态融合、可解释性和算法公平性的新兴关注。我们不仅提供了方法论洞见，还概述了主要的公开数据集和标准评估指标，为研究人员提供实用指南。通过综合当前进展并突出开放挑战，本文为计算精神病学未来创新提供了一个全面的路线图。 

---
# AgentCDM: Enhancing Multi-Agent Collaborative Decision-Making via ACH-Inspired Structured Reasoning 

**Title (ZH)**: AgentCDM：基于ACH启发式结构推理的多Agent协作决策增强 

**Authors**: Xuyang Zhao, Shiwan Zhao, Hualong Yu, Liting Zhang, Qicheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.11995)  

**Abstract**: Multi-agent systems (MAS) powered by large language models (LLMs) hold significant promise for solving complex decision-making tasks. However, the core process of collaborative decision-making (CDM) within these systems remains underexplored. Existing approaches often rely on either ``dictatorial" strategies that are vulnerable to the cognitive biases of a single agent, or ``voting-based" methods that fail to fully harness collective intelligence. To address these limitations, we propose \textbf{AgentCDM}, a structured framework for enhancing collaborative decision-making in LLM-based multi-agent systems. Drawing inspiration from the Analysis of Competing Hypotheses (ACH) in cognitive science, AgentCDM introduces a structured reasoning paradigm that systematically mitigates cognitive biases and shifts decision-making from passive answer selection to active hypothesis evaluation and construction. To internalize this reasoning process, we develop a two-stage training paradigm: the first stage uses explicit ACH-inspired scaffolding to guide the model through structured reasoning, while the second stage progressively removes this scaffolding to encourage autonomous generalization. Experiments on multiple benchmark datasets demonstrate that AgentCDM achieves state-of-the-art performance and exhibits strong generalization, validating its effectiveness in improving the quality and robustness of collaborative decisions in MAS. 

**Abstract (ZH)**: 基于大型语言模型的多Agent系统中的AgentCDM框架：增强协作决策的结构化方法 

---
# Modeling Relational Logic Circuits for And-Inverter Graph Convolutional Network 

**Title (ZH)**: 基于And-Inverter图卷积网络的关系逻辑电路建模 

**Authors**: Weihao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.11991)  

**Abstract**: The automation of logic circuit design enhances chip performance, energy efficiency, and reliability, and is widely applied in the field of Electronic Design Automation (EDA).And-Inverter Graphs (AIGs) efficiently represent, optimize, and verify the functional characteristics of digital circuits, enhancing the efficiency of EDA this http URL to the complex structure and large scale of nodes in real-world AIGs, accurate modeling is challenging, leading to existing work lacking the ability to jointly model functional and structural characteristics, as well as insufficient dynamic information propagation this http URL address the aforementioned challenges, we propose this http URL, AIGer consists of two components: 1) Node logic feature initialization embedding component and 2) AIGs feature learning network this http URL node logic feature initialization embedding component projects logic nodes, such as AND and NOT, into independent semantic spaces, to enable effective node embedding for subsequent this http URL upon this, the AIGs feature learning network component employs a heterogeneous graph convolutional network, designing dynamic relationship weight matrices and differentiated information aggregation approaches to better represent the original structure and information of this http URL combination of these two components enhances AIGer's ability to jointly model functional and structural characteristics and improves its message passing capability. Experimental results indicate that AIGer outperforms the current best models in the Signal Probability Prediction (SSP) task, improving MAE and MSE by 18.95\% and 44.44\%, respectively. In the Truth Table Distance Prediction (TTDP) task, AIGer achieves improvements of 33.57\% and 14.79\% in MAE and MSE, respectively, compared to the best-performing models. 

**Abstract (ZH)**: 逻辑电路设计自动化增强芯片性能、能量效率和可靠性，并广泛应用于电子设计自动化（EDA）领域。And-Inverter图（AIGs）高效地表示、优化和验证数字电路的功能特性，提升EDA的效率。然而，由于现实世界中AIGs复杂结构和大量节点，准确建模具有挑战性，导致现有工作难以同时建模功能和结构特性，以及信息传播动态不足。为解决上述挑战，我们提出了AIGer，它由两个部分组成：1）节点逻辑特征初始化嵌入组件；2）AIGs特征学习网络。节点逻辑特征初始化嵌入组件将如AND和NOT等逻辑节点投影到独立的语义空间中，以实现后续的有效节点嵌入。在此基础上，AIGs特征学习网络组件采用异构图卷积网络，设计动态关系权重矩阵和差异化的信息聚合方法，更好地表示原始结构和信息。这两种组件的结合增强了AIGer同时建模功能和结构特性的能力，并提高了其消息传递能力。实验结果表明，AIGer在信号概率预测（SSP）任务中优于当前最佳模型，分别将MAE和MSE提高了18.95%和44.44%。在真理表距离预测（TTDP）任务中，AIGer将MAE和MSE分别提高了33.57%和14.79%，优于表现最佳的模型。 

---
# FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction 

**Title (ZH)**: 未来X：面向未来预测的高级LLM代理现场基准 

**Authors**: Zhiyuan Zeng, Jiashuo Liu, Siyuan Chen, Tianci He, Yali Liao, Jinpeng Wang, Zaiyuan Wang, Yang Yang, Lingyue Yin, Mingren Yin, Zhenwei Zhu, Tianle Cai, Zehui Chen, Jiecao Chen, Yantao Du, Xiang Gao, Jiacheng Guo, Liang Hu, Jianpeng Jiao, Xiangsheng Li, Jingkai Liu, Shuang Ni, Zhoufutu Wen, Ge Zhang, Kaiyuan Zhang, Xin Zhou, Jose Blanchet, Xipeng Qiu, Mengdi Wang, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11987)  

**Abstract**: Future prediction is a complex task for LLM agents, requiring a high level of analytical thinking, information gathering, contextual understanding, and decision-making under uncertainty. Agents must not only gather and interpret vast amounts of dynamic information but also integrate diverse data sources, weigh uncertainties, and adapt predictions based on emerging trends, just as human experts do in fields like politics, economics, and finance. Despite its importance, no large-scale benchmark exists for evaluating agents on future prediction, largely due to challenges in handling real-time updates and retrieving timely, accurate answers. To address this, we introduce $\textbf{FutureX}$, a dynamic and live evaluation benchmark specifically designed for LLM agents performing future prediction tasks. FutureX is the largest and most diverse live benchmark for future prediction, supporting real-time daily updates and eliminating data contamination through an automated pipeline for question gathering and answer collection. We evaluate 25 LLM/agent models, including those with reasoning, search capabilities, and integration of external tools such as the open-source Deep Research Agent and closed-source Deep Research models. This comprehensive evaluation assesses agents' adaptive reasoning and performance in dynamic environments. Additionally, we provide in-depth analyses of agents' failure modes and performance pitfalls in future-oriented tasks, including the vulnerability to fake web pages and the temporal validity. Our goal is to establish a dynamic, contamination-free evaluation standard that drives the development of LLM agents capable of performing at the level of professional human analysts in complex reasoning and predictive thinking. 

**Abstract (ZH)**: 未来预测是大语言模型代理的一个复杂任务，需要高水平的分析思维、信息收集、上下文理解以及在不确定性下的决策能力。代理不仅要收集和解释大量动态信息，还需要整合多种数据源，权衡不确定性，并根据新兴趋势调整预测，类似于政治、经济和金融等领域的人类专家所做的工作。尽管其重要性不言而喻，但由于处理实时更新和获取及时准确答案的挑战，目前尚无针对未来预测的大型基准评估代理人。为解决这一问题，我们引入了**FutureX**，一个动态和实时的评估基准，专门设计用于执行未来预测任务的大语言模型代理。FutureX 是迄今为止最大的、最多样化的实时未来预测基准，支持每日实时更新，并通过自动化的问题收集和答案采集管道消除数据污染。我们评估了25个大语言模型/代理模型，包括具备推理、搜索能力以及结合外部工具（如开源的Deep Research Agent和闭源的Deep Research模型）的模型。这项全面评估评估了代理在动态环境中的适应性推理能力及其表现。此外，我们还深入分析了代理在面向未来的任务中失败模式和性能缺陷，包括对抗假网页的脆弱性以及时间有效性问题。我们的目标是建立一个动态且无污染的评估标准，促进能够与专业人类分析师在复杂推理和预测思维方面媲美的大语言模型代理的发展。 

---
# Chart-CoCa: Self-Improving Chart Understanding of Vision LMs via Code-Driven Synthesis and Candidate-Conditioned Answering 

**Title (ZH)**: Chart-CoCa: 通过代码引导合成和候选条件回答自我提升的图表理解视觉LMs 

**Authors**: Gongyao Jiang, Qiong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.11975)  

**Abstract**: Vision Language Models (VLMs) often struggle with chart understanding tasks, particularly in accurate chart description and complex reasoning. Synthetic data generation is a promising solution, while usually facing the challenge of noise labels. To address this challenge, we first introduce a chart synthesis pipeline that generates aligned chart-question-answer triplets through code generation and execution, ensuring the reliability of synthetic data without human intervention. Furthermore, inspired by test-time scaling that increases inference budget and thereby improves performance, we design a candidate-conditioned answering process. The VLM first generates multiple responses per query, and then synthesizes the final answer by contextualizing these candidates. Experiments demonstrate significant improvements, with up to 15.50 points accuracy gain over the initial VLM, in a fully self-improving paradigm without either human-labeled data or external models. 

**Abstract (ZH)**: 视觉语言模型在图表理解任务中往往表现出色，特别是在准确的图表描述和复杂推理方面存在挑战。合成数据生成是一种有前景的解决方案，但通常会遇到噪声标签的挑战。为应对这一挑战，我们首先介绍了一种图表合成管道，通过代码生成和执行生成对齐的图表-问题-答案 triplet，确保在无需人类干预的情况下生成合成数据的可靠性。此外，受测试时扩增增加推理预算从而提高性能的启发，我们设计了一种候选条件化回答过程。视觉语言模型首先生成每个查询的多个响应，然后通过上下文化这些候选来合成最终答案。实验结果表明，在完全自我改进的范式下，该模型在无需人工标注数据或外部模型的情况下，准确率提高了高达15.50个百分点。 

---
# Rigorous Feature Importance Scores based on Shapley Value and Banzhaf Index 

**Title (ZH)**: 基于Shapley值和巴纳夫指数的严谨特征重要性评分 

**Authors**: Xuanxiang Huang, Olivier Létoffé, Joao Marques-Silva  

**Link**: [PDF](https://arxiv.org/pdf/2508.11959)  

**Abstract**: Feature attribution methods based on game theory are ubiquitous in the field of eXplainable Artificial Intelligence (XAI). Recent works proposed rigorous feature attribution using logic-based explanations, specifically targeting high-stakes uses of machine learning (ML) models. Typically, such works exploit weak abductive explanation (WAXp) as the characteristic function to assign importance to features. However, one possible downside is that the contribution of non-WAXp sets is neglected. In fact, non-WAXp sets can also convey important information, because of the relationship between formal explanations (XPs) and adversarial examples (AExs). Accordingly, this paper leverages Shapley value and Banzhaf index to devise two novel feature importance scores. We take into account non-WAXp sets when computing feature contribution, and the novel scores quantify how effective each feature is at excluding AExs. Furthermore, the paper identifies properties and studies the computational complexity of the proposed scores. 

**Abstract (ZH)**: 基于博弈论的特征归因方法在可解释人工智能（XAI）领域广泛应用。近期研究提出了基于逻辑解释的严格特征归因方法，特别针对机器学习（ML）模型的高风险应用。通常，此类工作利用弱归纳解释（WAXp）作为特征重要性分配的特征函数。然而，一个潜在的缺点是未考虑非WAXp集的贡献。事实上，非WAXp集也能提供重要的信息，因为形式化解释（XPs）与对抗性示例（AExs）之间的关系。因此，本文利用Shapley值和Banzhaf指数提出两种新的特征重要性评分。在计算特征贡献时考虑到非WAXp集，并且新的评分衡量每个特征排除AExs的有效性。此外，本文还研究了所提评分的性质及其计算复杂性。 

---
# UniCast: A Unified Multimodal Prompting Framework for Time Series Forecasting 

**Title (ZH)**: UniCast：统一的多模态提示框架用于时间序列预测 

**Authors**: Sehyuk Park, Soyeon Caren Han, Eduard Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2508.11954)  

**Abstract**: Time series forecasting is a foundational task across domains, such as finance, healthcare, and environmental monitoring. While recent advances in Time Series Foundation Models (TSFMs) have demonstrated strong generalisation through large-scale pretraining, existing models operate predominantly in a unimodal setting, ignoring the rich multimodal context, such as visual and textual signals, that often accompanies time series data in real-world scenarios. This paper introduces a novel parameter-efficient multimodal framework, UniCast, that extends TSFMs to jointly leverage time series, vision, and text modalities for enhanced forecasting performance. Our method integrates modality-specific embeddings from pretrained Vision and Text Encoders with a frozen TSFM via soft prompt tuning, enabling efficient adaptation with minimal parameter updates. This design not only preserves the generalisation strength of the foundation model but also enables effective cross-modal interaction. Extensive experiments across diverse time-series forecasting benchmarks demonstrate that UniCast consistently and significantly outperforms all existing TSFM baselines. The findings highlight the critical role of multimodal context in advancing the next generation of general-purpose time series forecasters. 

**Abstract (ZH)**: 时间序列预测是金融、医疗和环境监测等领域中的基础任务。尽管近期的时间序列基础模型（TSFM）通过大规模预训练展示了强大的泛化能力，但现有模型主要在单模态设置中运行，忽视了现实场景中常伴随时间序列数据的丰富多模态上下文，如视觉和文本信号。本文介绍了一种新颖的参数高效多模态框架UniCast，它将TSFMs扩展到了联合利用时间序列、视觉和文本模态的增强预测性能。本文方法通过软提示调谐将预训练的视觉和文本编码器的模态特定嵌入与冻结的TSFM结合，实现高效适应和最少参数更新。这种设计不仅保留了基础模型的泛化能力，还能有效促进跨模态交互。在多种时间序列预测基准上的广泛实验表明，UniCast在所有现有的TSFM基线方法上表现始终且显著更优。研究结果强调了多模态上下文在推进新一代通用时间序列预测器中的关键作用。 

---
# Data Mixing Optimization for Supervised Fine-Tuning of Large Language Models 

**Title (ZH)**: 大型语言模型有监督微调中的数据混杂优化 

**Authors**: Yuan Li, Zhengzhong Liu, Eric Xing  

**Link**: [PDF](https://arxiv.org/pdf/2508.11953)  

**Abstract**: Optimizing data mixtures for supervised fine-tuning (SFT) of large language models (LLMs) is critical for developing general-purpose models, yet this area remains underexplored. In this paper, we frame data mixing as an optimization problem and introduce a novel method designed to minimize validation loss. Our approach parametrizes the loss by modeling effective data transferred and leveraging scaling laws for fine-tuning. By experimenting with various small-scale data mixtures, we fit these parameters and derive the optimal weights. We provide both mathematical proofs and empirical results demonstrating that our algorithm achieves excellent overall and individual performance across all domains. Through controlled experiments, we show that models trained with our optimized weights perform on par with those using optimal weights determined via grid search, with per-domain loss only 0.66% higher than the best domain loss from grid search on average. Additionally, we show that reweighting popular SFT datasets using our method improves both validation loss and downstream performance. Finally, we discuss how our method can generalize to guide data selection for domain-specific models and provide insights into SFT. 

**Abstract (ZH)**: 优化数据混合以提高大型语言模型监督微调的性能对于开发通用模型至关重要，但这一领域尚未得到充分探索。在本文中，我们将数据混合构架为一个优化问题，并提出了一种新的方法，旨在最小化验证损失。我们通过建模有效转移到的数据并利用微调中的标度定律来参数化损失。通过实验各种小规模数据混合，我们拟合这些参数并得出最优权重。我们提供了数学证明和实验证据，证明我们的算法在所有领域中都实现了出色的总体和个体性能。通过受控实验，我们展示了使用我们优化权重训练的模型在性能上与通过网格搜索确定最优权重的模型相当，平均域损失仅比网格搜索的最佳域损失高出0.66%。此外，我们展示了使用我们的方法重新加权流行的监督微调数据集能同时提高验证损失和下游性能。最后，我们讨论了我们的方法如何推广以指导特定领域模型的数据选择，并提供了对监督微调的见解。 

---
# CHBench: A Cognitive Hierarchy Benchmark for Evaluating Strategic Reasoning Capability of LLMs 

**Title (ZH)**: CHBench: 一种用于评估大规模语言模型战略推理能力的认知层次基准 

**Authors**: Hongtao Liu, Zhicheng Du, Zihe Wang, Weiran Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11944)  

**Abstract**: Game-playing ability serves as an indicator for evaluating the strategic reasoning capability of large language models (LLMs). While most existing studies rely on utility performance metrics, which are not robust enough due to variations in opponent behavior and game structure. To address this limitation, we propose \textbf{Cognitive Hierarchy Benchmark (CHBench)}, a novel evaluation framework inspired by the cognitive hierarchy models from behavioral economics. We hypothesize that agents have bounded rationality -- different agents behave at varying reasoning depths/levels. We evaluate LLMs' strategic reasoning through a three-phase systematic framework, utilizing behavioral data from six state-of-the-art LLMs across fifteen carefully selected normal-form games. Experiments show that LLMs exhibit consistent strategic reasoning levels across diverse opponents, confirming the framework's robustness and generalization capability. We also analyze the effects of two key mechanisms (Chat Mechanism and Memory Mechanism) on strategic reasoning performance. Results indicate that the Chat Mechanism significantly degrades strategic reasoning, whereas the Memory Mechanism enhances it. These insights position CHBench as a promising tool for evaluating LLM capabilities, with significant potential for future research and practical applications. 

**Abstract (ZH)**: 认知层次基准（CHBench）：评估大型语言模型的战略推理能力 

---
# QuarkMed Medical Foundation Model Technical Report 

**Title (ZH)**: 夸克医疗医学基础模型技术报告 

**Authors**: Ao Li, Bin Yan, Bingfeng Cai, Chenxi Li, Cunzhong Zhao, Fugen Yao, Gaoqiang Liu, Guanjun Jiang, Jian Xu, Liang Dong, Liansheng Sun, Rongshen Zhang, Xiaolei Gui, Xin Liu, Xin Shang, Yao Wu, Yu Cao, Zhenxin Ma, Zhuang Jia  

**Link**: [PDF](https://arxiv.org/pdf/2508.11894)  

**Abstract**: Recent advancements in large language models have significantly accelerated their adoption in healthcare applications, including AI-powered medical consultations, diagnostic report assistance, and medical search tools. However, medical tasks often demand highly specialized knowledge, professional accuracy, and customization capabilities, necessitating a robust and reliable foundation model. QuarkMed addresses these needs by leveraging curated medical data processing, medical-content Retrieval-Augmented Generation (RAG), and a large-scale, verifiable reinforcement learning pipeline to develop a high-performance medical foundation model. The model achieved 70% accuracy on the Chinese Medical Licensing Examination, demonstrating strong generalization across diverse medical benchmarks. QuarkMed offers a powerful yet versatile personal medical AI solution, already serving over millions of users at this http URL. 

**Abstract (ZH)**: 近年来，大型语言模型的最新进展显著加速了其在医疗应用中的采用，包括AI驱动的医疗咨询、诊断报告辅助和医疗搜索工具。然而，医疗任务往往需要高度专业化的知识、专业准确性以及定制能力，因此需要一个强大且可靠的预训练模型。QuarkMed 通过利用精心整理的医疗数据处理、医疗内容检索增强生成（RAG）以及大规模验证强化学习管道来满足这些需求，以开发高性能的医疗预训练模型。该模型在中文医师资格考试中实现了70%的准确率，展示了其在各种医疗基准测试中的强大泛化能力。QuarkMed 提供了一个强大且多功能的个人医疗AI解决方案，目前已为数百万用户提供服务。 

---
# LARC: Towards Human-level Constrained Retrosynthesis Planning through an Agentic Framework 

**Title (ZH)**: LARC：通过代理框架实现人类水平的受控 retrosynthesis 规划 

**Authors**: Frazier N. Baker, Daniel Adu-Ampratwum, Reza Averly, Botao Yu, Huan Sun, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2508.11860)  

**Abstract**: Large language model (LLM) agent evaluators leverage specialized tools to ground the rational decision-making of LLMs, making them well-suited to aid in scientific discoveries, such as constrained retrosynthesis planning. Constrained retrosynthesis planning is an essential, yet challenging, process within chemistry for identifying synthetic routes from commercially available starting materials to desired target molecules, subject to practical constraints. Here, we present LARC, the first LLM-based Agentic framework for Retrosynthesis planning under Constraints. LARC incorporates agentic constraint evaluation, through an Agent-as-a-Judge, directly into the retrosynthesis planning process, using agentic feedback grounded in tool-based reasoning to guide and constrain route generation. We rigorously evaluate LARC on a carefully curated set of 48 constrained retrosynthesis planning tasks across 3 constraint types. LARC achieves a 72.9% success rate on these tasks, vastly outperforming LLM baselines and approaching human expert-level success in substantially less time. The LARC framework is extensible, and serves as a first step towards an effective agentic tool or a co-scientist to human experts for constrained retrosynthesis. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的代理评估器通过专用工具促进LLM的理性决策，使其成为辅助化学领域受限逆合成规划等科学发现的理想选择。受限逆合成规划是化学中一个关键但具有挑战性的过程，涉及从商业原料合成目标分子，同时遵守实际约束。本文提出LARC，首个基于LLM的逆合成规划代理框架，该框架直接将代理式约束评估融入逆合成规划过程，利用基于工具的推理提供的代理反馈来引导和限制合成路线生成。我们在一个精心选择的包含48个不同类型约束的逆合成规划任务集上严格评估了LARC，LARC在这些任务上的成功率达到了72.9%，显著优于LLM基线模型，并且在较短时间内接近人类专家级别的成功率。LARC框架具有扩展性，代表着迈向有效代理工具或辅助人类专家进行受限逆合成方向的重要一步。 

---
# EvoCut: Strengthening Integer Programs via Evolution-Guided Language Models 

**Title (ZH)**: EvoCut: 通过进化导向语言模型强化整数规划 

**Authors**: Milad Yazdani, Mahdi Mostajabdaveh, Samin Aref, Zirui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.11850)  

**Abstract**: Integer programming lies at the heart of crucial combinatorial optimization tasks but remains challenging due to its NP-hard nature. An effective approach for practically solving integer programs is the manual design of acceleration cuts, i.e. inequalities that improve solver performance. However, this creative process demands deep expertise and is yet to be automated. Our proposed framework, EvoCut, automates the generation of acceleration cuts by combining large language models (LLMs) with an evolutionary search. EvoCut (i) initializes a diverse population of candidate cuts via an LLM-based initializer agent; (ii) for each cut empirically evaluates both preservation of the optimal solution and its ability to cut off fractional solutions across a verification set; and (iii) iteratively refines the population through evolutionary crossover and mutation agents. We quantify each cut's utility by its relative reduction in the solver's optimality gap. Our comparisons against standard integer programming practice show that EvoCut reduces optimality gap by 17-57% within a fixed time. It obtains the same solutions up to 4 times as fast, and obtains higher-quality solutions within the same time limit. Requiring no human expert input, EvoCut reliably generates, improves, and empirically verifies cuts that generalize to unseen instances. The code is available at this https URL. 

**Abstract (ZH)**: EvoCut：通过进化搜索和大型语言模型自动化加速剪枝生成 

---
# Finite Automata Extraction: Low-data World Model Learning as Programs from Gameplay Video 

**Title (ZH)**: 有限状态机提取：从游戏视频中学习基于程序的低数据世界模型 

**Authors**: Dave Goel, Matthew Guzdial, Anurag Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2508.11836)  

**Abstract**: World models are defined as a compressed spatial and temporal learned representation of an environment. The learned representation is typically a neural network, making transfer of the learned environment dynamics and explainability a challenge. In this paper, we propose an approach, Finite Automata Extraction (FAE), that learns a neuro-symbolic world model from gameplay video represented as programs in a novel domain-specific language (DSL): Retro Coder. Compared to prior world model approaches, FAE learns a more precise model of the environment and more general code than prior DSL-based approaches. 

**Abstract (ZH)**: 世界模型被定义为环境的压缩空间和时间的learned表示。learned表示通常是一个神经网络，这使得learned环境动力学的转移和解释成为一个挑战。在本文中，我们提出了一种方法，Finite Automata Extraction (FAE)，它从以新型领域特定语言（DSL）Retro Coder表示的游戏视频中学习一个神经符号世界模型。与之前的world model方法相比，FAE学习了更精确的环境模型和更具一般性的代码。 

---
# RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation Patterns 

**Title (ZH)**: RepreGuard: 检测由大型语言模型生成的文本通过揭示隐藏的表示模式 

**Authors**: Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Ziyang Luo, Di Wang, Min Yang, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.13152)  

**Abstract**: Detecting content generated by large language models (LLMs) is crucial for preventing misuse and building trustworthy AI systems. Although existing detection methods perform well, their robustness in out-of-distribution (OOD) scenarios is still lacking. In this paper, we hypothesize that, compared to features used by existing detection methods, the internal representations of LLMs contain more comprehensive and raw features that can more effectively capture and distinguish the statistical pattern differences between LLM-generated texts (LGT) and human-written texts (HWT). We validated this hypothesis across different LLMs and observed significant differences in neural activation patterns when processing these two types of texts. Based on this, we propose RepreGuard, an efficient statistics-based detection method. Specifically, we first employ a surrogate model to collect representation of LGT and HWT, and extract the distinct activation feature that can better identify LGT. We can classify the text by calculating the projection score of the text representations along this feature direction and comparing with a precomputed threshold. Experimental results show that RepreGuard outperforms all baselines with average 94.92% AUROC on both in-distribution (ID) and OOD scenarios, while also demonstrating robust resilience to various text sizes and mainstream attacks. Data and code are publicly available at: this https URL 

**Abstract (ZH)**: 检测大型语言模型生成的内容对于防止滥用和构建可信赖的AI系统至关重要。尽管现有的检测方法表现良好，但在离分布（OOD）场景中的鲁棒性仍然不足。在本文中，我们假设与现有检测方法使用的特征相比，大型语言模型内部表示包含了更全面和原始的特征，可以更有效地捕捉和区分大型语言模型生成文本（LGT）和人类写作文本（HWT）之间的统计模式差异。我们通过对不同大型语言模型的验证，观察到处理这两种类型文本时神经激活模式存在显著差异。基于此，我们提出了RepreGuard，一种高效的基于统计的检测方法。具体来说，我们首先使用一个代理模型来收集LGT和HWT的表示，并提取能够更好地识别LGT的差异激活特征。通过计算文本表示在这条特征方向上的投影分数并与预计算的阈值进行比较，可以对文本进行分类。实验结果表明，在分布（ID）和离分布（OOD）场景中，RepreGuard的均值AUROC为94.92%，且在各种文本大小和主流攻击面前表现出强大的鲁棒性。数据和代码已公开可用：this https URL。 

---
# Spot the BlindSpots: Systematic Identification and Quantification of Fine-Grained LLM Biases in Contact Center Summaries 

**Title (ZH)**: 识别盲区：系统识别和量化客服中心总结中的细粒度LLM偏差 

**Authors**: Kawin Mayilvaghanan, Siddhant Gupta, Ayush Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.13124)  

**Abstract**: Abstractive summarization is a core application in contact centers, where Large Language Models (LLMs) generate millions of summaries of call transcripts daily. Despite their apparent quality, it remains unclear whether LLMs systematically under- or over-attend to specific aspects of the transcript, potentially introducing biases in the generated summary. While prior work has examined social and positional biases, the specific forms of bias pertinent to contact center operations - which we term Operational Bias - have remained unexplored. To address this gap, we introduce BlindSpot, a framework built upon a taxonomy of 15 operational bias dimensions (e.g., disfluency, speaker, topic) for the identification and quantification of these biases. BlindSpot leverages an LLM as a zero-shot classifier to derive categorical distributions for each bias dimension in a pair of transcript and its summary. The bias is then quantified using two metrics: Fidelity Gap (the JS Divergence between distributions) and Coverage (the percentage of source labels omitted). Using BlindSpot, we conducted an empirical study with 2500 real call transcripts and their summaries generated by 20 LLMs of varying scales and families (e.g., GPT, Llama, Claude). Our analysis reveals that biases are systemic and present across all evaluated models, regardless of size or family. 

**Abstract (ZH)**: 基于税收目录的自提取摘要在客服中心的核心应用中占据重要地位，其中大规模语言模型（LLMs）每天生成数百万份通话记录摘要。尽管这些摘要看似质量优良，但仍不清楚LLMs是否系统地忽视或重视通话记录中的特定方面，从而可能在生成的摘要中引入偏差。尽管先前的研究调查了社会和职位偏差，与客服中心运营相关的特定形式的偏差——我们称之为操作偏差——仍然未被探索。为填补这一空白，我们提出了BlindSpot框架，该框架基于15个操作偏差维度的分类系统（例如，语不流畅、说话者、主题），用于识别和量化这些偏差。BlindSpot利用LLM作为零样本分类器，为一对通话记录及其摘要推导出每个偏差维度的分类分布。然后，偏差用两个指标进行量化：保真度差距（分布之间的JS散度）和覆盖率（遗漏的原始标签比例）。通过BlindSpot，我们对20种不同规模和家族（例如，GPT、Llama、Claude）的2500份真实通话记录及其摘要进行了实证研究。我们的分析表明，无论规模或家族如何，这些偏差是系统性且普遍存在于所有评估模型中的。 

---
# Contrastive Representations for Temporal Reasoning 

**Title (ZH)**: 对比表示方法在时间推理中的应用 

**Authors**: Alicja Ziarko, Michal Bortkiewicz, Michal Zawalski, Benjamin Eysenbach, Piotr Milos  

**Link**: [PDF](https://arxiv.org/pdf/2508.13113)  

**Abstract**: In classical AI, perception relies on learning state-based representations, while planning, which can be thought of as temporal reasoning over action sequences, is typically achieved through search. We study whether such reasoning can instead emerge from representations that capture both perceptual and temporal structure. We show that standard temporal contrastive learning, despite its popularity, often fails to capture temporal structure due to its reliance on spurious features. To address this, we introduce Combinatorial Representations for Temporal Reasoning (CRTR), a method that uses a negative sampling scheme to provably remove these spurious features and facilitate temporal reasoning. CRTR achieves strong results on domains with complex temporal structure, such as Sokoban and Rubik's Cube. In particular, for the Rubik's Cube, CRTR learns representations that generalize across all initial states and allow it to solve the puzzle using fewer search steps than BestFS, though with longer solutions. To our knowledge, this is the first method that efficiently solves arbitrary Cube states using only learned representations, without relying on an external search algorithm. 

**Abstract (ZH)**: 基于组合表示的时间推理（Combinatorial Representations for Temporal Reasoning）：解决具有复杂时间结构领域的问题 

---
# VerilogLAVD: LLM-Aided Rule Generation for Vulnerability Detection in Verilog 

**Title (ZH)**: VerilogLAVD: 基于LLM的Verilog漏洞检测规则生成 

**Authors**: Xiang Long, Yingjie Xia, Xiyuan Chen, Li Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13092)  

**Abstract**: Timely detection of hardware vulnerabilities during the early design stage is critical for reducing remediation costs. Existing early detection techniques often require specialized security expertise, limiting their usability. Recent efforts have explored the use of large language models (LLMs) for Verilog vulnerability detection. However, LLMs struggle to capture the structure in Verilog code, resulting in inconsistent detection results. To this end, we propose VerilogLAVD, the first LLM-aided graph traversal rule generation approach for Verilog vulnerability detection. Our approach introduces the Verilog Property Graph (VeriPG), a unified representation of Verilog code. It combines syntactic features extracted from the abstract syntax tree (AST) with semantic information derived from control flow and data dependency graphs. We leverage LLMs to generate VeriPG-based detection rules from Common Weakness Enumeration (CWE) descriptions. These rules guide the rule executor that traversal VeriPG for potential vulnerabilities. To evaluate VerilogLAVD, we build a dataset collected from open-source repositories and synthesized data. In our empirical evaluation on 77 Verilog designs encompassing 12 CWE types, VerilogLAVD achieves an F1-score of 0.54. Compared to the LLM-only and LLM with external knowledge baselines, VerilogLAVD improves F1-score by 0.31 and 0.27, respectively. 

**Abstract (ZH)**: 及时在早期设计阶段检测硬件漏洞对于减少修复成本至关重要。现有早期检测技术往往需要专门的安全专业知识，限制了其可用性。最近的研究探索了使用大规模语言模型（LLMs）进行Verilog漏洞检测的方法。然而，LLMs在捕捉Verilog代码结构方面存在困难，导致检测结果不一致。为此，我们提出VerilogLAVD，这是一种用于Verilog漏洞检测的第一种LLM辅助图遍历规则生成方法。我们的方法引入了Verilog属性图（VeriPG），这是一种统一的Verilog代码表示。VeriPG结合了从抽象语法树（AST）提取的语法特征和从控制流图和数据依赖图中导出的语义信息。我们利用LLMs从通用弱点枚举（CWE）描述中生成基于VeriPG的检测规则，这些规则引导规则执行器遍历VeriPG以发现潜在漏洞。为了评估VerilogLAVD，我们构建了一个来自开源仓库和合成数据的数据集。在涵盖12种CWE类型的77个Verilog设计的实证评价中，VerilogLAVD取得了0.54的F1分数。与仅使用LLM和结合外部知识库的基线相比，VerilogLAVD分别提高了0.31和0.27的F1分数。 

---
# From Transthoracic to Transesophageal: Cross-Modality Generation using LoRA Diffusion 

**Title (ZH)**: 从胸壁到食道：基于LoRA扩散模型的跨模态生成 

**Authors**: Emmanuel Oladokun, Yuxuan Ou, Anna Novikova, Daria Kulikova, Sarina Thomas, Jurica Šprem, Vicente Grau  

**Link**: [PDF](https://arxiv.org/pdf/2508.13077)  

**Abstract**: Deep diffusion models excel at realistic image synthesis but demand large training sets-an obstacle in data-scarce domains like transesophageal echocardiography (TEE). While synthetic augmentation has boosted performance in transthoracic echo (TTE), TEE remains critically underrepresented, limiting the reach of deep learning in this high-impact modality.
We address this gap by adapting a TTE-trained, mask-conditioned diffusion backbone to TEE with only a limited number of new cases and adapters as small as $10^5$ parameters. Our pipeline combines Low-Rank Adaptation with MaskR$^2$, a lightweight remapping layer that aligns novel mask formats with the pretrained model's conditioning channels. This design lets users adapt models to new datasets with a different set of anatomical structures to the base model's original set.
Through a targeted adaptation strategy, we find that adapting only MLP layers suffices for high-fidelity TEE synthesis. Finally, mixing less than 200 real TEE frames with our synthetic echoes improves the dice score on a multiclass segmentation task, particularly boosting performance on underrepresented right-heart structures. Our results demonstrate that (1) semantically controlled TEE images can be generated with low overhead, (2) MaskR$^2$ effectively transforms unseen mask formats into compatible formats without damaging downstream task performance, and (3) our method generates images that are effective for improving performance on a downstream task of multiclass segmentation. 

**Abstract (ZH)**: 深度扩散模型在图像合成方面表现出色，但需要庞大的训练集——这在超声心动图食道部分（TEE）这样的数据稀缺领域是一大障碍。虽然合成增强在经胸超声（TTE）中提升了性能，TEE仍严重不足，限制了深度学习在这一高影响成像模式中的应用范围。

我们通过仅使用少量的新案例和仅有 $10^5$ 参数的适配器，将TTE训练的掩码条件扩散主干模型应用到TEE，来填补这一差距。我们的流程结合了低秩适配和MaskR$^2$，这是一种轻量级的重映射层，能够将新的掩码格式与预训练模型的条件通道对齐。这种设计允许用户用不同的解剖结构集将模型适应到新的数据集中。

通过一种有针对性的适配策略，我们发现仅适配MLP层就足以实现高保真TEE合成。最后，将不到200个真实的TEE帧与我们的合成回声混合，可以提高多类分割任务的Dice分数，尤其是在提升未充分代表的右心结构的性能方面尤为明显。我们的结果表明：（1）通过低开销可以生成具有语义控制的TEE图像；（2）MaskR$^2$ 能够有效地将未见过的掩码格式转换为兼容格式，而不损害下游任务的性能；（3）我们的方法生成的图像对下游任务的多类分割性能提升有效。 

---
# Reinforced Context Order Recovery for Adaptive Reasoning and Planning 

**Title (ZH)**: 强化上下文顺序恢复以实现自适应推理与规划 

**Authors**: Long Ma, Fangwei Zhong, Yizhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13070)  

**Abstract**: Modern causal language models, followed by rapid developments in discrete diffusion models, can now produce a wide variety of interesting and useful content. However, these families of models are predominantly trained to output tokens with a fixed (left-to-right) or random order, which may deviate from the logical order in which tokens are generated originally. In this paper, we observe that current causal and diffusion models encounter difficulties in problems that require adaptive token generation orders to solve tractably, which we characterize with the $\mathcal{V}$-information framework. Motivated by this, we propose Reinforced Context Order Recovery (ReCOR), a reinforcement-learning-based framework to extract adaptive, data-dependent token generation orders from text data without annotations. Self-supervised by token prediction statistics, ReCOR estimates the hardness of predicting every unfilled token and adaptively selects the next token during both training and inference. Experiments on challenging reasoning and planning datasets demonstrate the superior performance of ReCOR compared with baselines, sometimes outperforming oracle models supervised with the ground-truth order. 

**Abstract (ZH)**: 现代因果语言模型在离散扩散模型快速发展的背景下，能够生成广泛种类的有趣和有用的内容。然而，这些模型大多被训练成以固定顺序（从左到右）或随机顺序生成标记，这可能与标记原始生成的逻辑顺序不符。在本文中，我们观察到当前因果和扩散模型在需要自适应标记生成顺序以解决的问题中遇到了困难，我们使用$\mathcal{V}$-信息框架来刻画这一特征。受到这一观察的启发，我们提出了一种基于强化学习的框架Reinforced Context Order Recovery (ReCOR)，该框架能够在无需标注的情况下从文本数据中提取自适应的数据依赖性标记生成顺序。通过标记预测统计进行半监督学习，ReCOR 估计每个未填充标记的预测难度，并在训练和推理过程中自适应地选择下一个标记。在具有挑战性的推理和规划数据集上的实验表明，ReCOR 的表现优于基线模型，有时甚至优于带有真实顺序标注的 oracle 模型。 

---
# Hierarchical Evaluation Function (HEF): A Multi-Metric Approach for Optimizing Demand Forecasting Models 

**Title (ZH)**: 层次评估函数（HEF）：一种多指标方法优化需求预测模型 

**Authors**: Adolfo González, Víctor Parada  

**Link**: [PDF](https://arxiv.org/pdf/2508.13057)  

**Abstract**: Demand forecasting is essential for strategic planning in competitive environments, enabling resource optimization and improved responsiveness to market dynamics. However, multivariate time series modeling faces challenges due to data complexity, uncertainty, and frequent regime shifts. Traditional evaluation metrics can introduce biases and limit generalization. This work compares two custom evaluation functions: FMAE (Focused Mean Absolute Error), focused on minimizing absolute errors, and HEF (Hierarchical Evaluation Function), designed to weight global metrics and penalize large deviations. Experiments were conducted under different data splits (91:9, 80:20, 70:30) using three optimizers (Grid Search, PSO, Optuna), assessing fit, relative accuracy, robustness, and computational efficiency. Results show that HEF consistently outperforms FMAE in global metrics (R2, Relative Accuracy, RMSE, RMSSE), enhancing model robustness and explanatory power. These findings were confirmed via visualizations and statistical tests. Conversely, FMAE offers advantages in local metrics (MAE, MASE) and execution time, making it suitable for short-term scenarios. The study highlights a methodological trade-off: HEF is ideal for strategic planning, while FMAE is better suited for operational efficiency. A replicable framework is proposed for optimizing predictive models in dynamic environments. 

**Abstract (ZH)**: 基于定制评价函数的多变量时间序列建模比较：面向动态环境的预测模型优化方法 

---
# XR-NPE: High-Throughput Mixed-precision SIMD Neural Processing Engine for Extended Reality Perception Workloads 

**Title (ZH)**: XR-NPE：扩展现实感知工作负载的高 throughput 混合精度 SIMD 神经处理引擎 

**Authors**: Tejas Chaudhari, Akarsh J., Tanushree Dewangan, Mukul Lokhande, Santosh Kumar Vishvakarma  

**Link**: [PDF](https://arxiv.org/pdf/2508.13049)  

**Abstract**: This work proposes XR-NPE, a high-throughput Mixed-precision SIMD Neural Processing Engine, designed for extended reality (XR) perception workloads like visual inertial odometry (VIO), object classification, and eye gaze extraction. XR-NPE is first to support FP4, Posit (4,1), Posit (8,0), and Posit (16,1) formats, with layer adaptive hybrid-algorithmic implementation supporting ultra-low bit precision to significantly reduce memory bandwidth requirements, and accompanied by quantization-aware training for minimal accuracy loss. The proposed Reconfigurable Mantissa Multiplication and Exponent processing Circuitry (RMMEC) reduces dark silicon in the SIMD MAC compute engine, assisted by selective power gating to reduce energy consumption, providing 2.85x improved arithmetic intensity. XR-NPE achieves a maximum operating frequency of 1.72 GHz, area 0.016 mm2 , and arithmetic intensity 14 pJ at CMOS 28nm, reducing 42% area, 38% power compared to the best of state-of-the-art MAC approaches. The proposed XR-NPE based AXI-enabled Matrix-multiplication co-processor consumes 1.4x fewer LUTs, 1.77x fewer FFs, and provides 1.2x better energy efficiency compared to SoTA accelerators on VCU129. The proposed co-processor provides 23% better energy efficiency and 4% better compute density for VIO workloads. XR-NPE establishes itself as a scalable, precision-adaptive compute engine for future resource-constrained XR devices. The complete set for codes for results reproducibility are released publicly, enabling designers and researchers to readily adopt and build upon them. this https URL. 

**Abstract (ZH)**: XR-NPE：一种面向扩展现实感知工作负载的高吞吐量混合精度 SIMD 神经处理引擎 

---
# Using AI for User Representation: An Analysis of 83 Persona Prompts 

**Title (ZH)**: 使用AI进行用户表示：对83个个性提示的分析 

**Authors**: Joni Salminen, Danial Amin, Bernard Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13047)  

**Abstract**: We analyzed 83 persona prompts from 27 research articles that used large language models (LLMs) to generate user personas. Findings show that the prompts predominantly generate single personas. Several prompts express a desire for short or concise persona descriptions, which deviates from the tradition of creating rich, informative, and rounded persona profiles. Text is the most common format for generated persona attributes, followed by numbers. Text and numbers are often generated together, and demographic attributes are included in nearly all generated personas. Researchers use up to 12 prompts in a single study, though most research uses a small number of prompts. Comparison and testing multiple LLMs is rare. More than half of the prompts require the persona output in a structured format, such as JSON, and 74% of the prompts insert data or dynamic variables. We discuss the implications of increased use of computational personas for user representation. 

**Abstract (ZH)**: 我们分析了27篇研究文章中的83个用户画像提示词，这些研究文章使用大规模语言模型（LLM）生成用户画像。研究发现，这些提示词主要生成单个用户画像。许多提示词表达了对简短或精炼用户画像描述的偏好，这与创建丰富、信息量大且立体的用户画像传统相悖。生成的用户画像属性中最常见的格式是文本，其次是数字。文本和数字通常一起生成，几乎所有生成的用户画像都包括人口统计数据。研究人员在单个研究中最多使用12个提示词，但大多数研究使用少量提示词。比较和测试多个LLM的情况很少见。超过一半的提示词要求以结构化格式输出用户画像，如JSON，并且74%的提示词插入了数据或动态变量。我们讨论了计算生成用户画像增加使用对用户代表性的意义。 

---
# Can Large Models Teach Student Models to Solve Mathematical Problems Like Human Beings? A Reasoning Distillation Method via Multi-LoRA Interaction 

**Title (ZH)**: 大型模型能否通过多LoRA交互进行推理提炼，从而教会学生模型像人类一样解决数学问题？ 

**Authors**: Xinhe Li, Jiajun Liu, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13037)  

**Abstract**: Recent studies have demonstrated that Large Language Models (LLMs) have strong mathematical reasoning abilities but rely on hundreds of billions of parameters. To tackle the challenge of poor reasoning in Small Language Models (SLMs), existing methods typically leverage LLMs to generate massive amounts of data for cramming training. In psychology, they are akin to System 1 thinking, which resolves reasoning problems rapidly based on experience and intuition. However, human learning also requires System 2 thinking, where knowledge is first acquired and then reinforced through practice. Inspired by such two distinct modes of thinking, we propose a novel method based on the multi-LoRA Interaction for mathematical reasoning Distillation (LoRID). First, we input the question and reasoning of each sample into an LLM to create knowledge-enhanced datasets. Subsequently, we train a LoRA block on the student model as an Intuitive Reasoner (IR), which directly generates Chain-of-Thoughts for problem-solving. Then, to imitate System 2 thinking, we train the Knowledge Generator (KG) and Deep Reasoner (DR), respectively. The former outputs only knowledge after receiving problems, while the latter uses that knowledge to perform reasoning. Finally, to address the randomness in the generation of IR and DR, we evaluate whether their outputs are consistent, and the inference process needs to be iterated if not. This step can enhance the mathematical reasoning ability of SLMs through mutual feedback. Experimental results show that LoRID achieves state-of-the-art performance, especially on the GSM8K dataset, where it outperforms the second-best method by 2.3%, 16.1%, 2.4%, 12.3%, and 1.8% accuracy across the five base models, respectively. 

**Abstract (ZH)**: Recent Studies Have Demonstrated that Large Language Models (LLMs) Have Strong Mathematical Reasoning Abilities but Rely on Hundreds of Billions of Parameters 

---
# The Application of Transformer-Based Models for Predicting Consequences of Cyber Attacks 

**Title (ZH)**: 基于变压器模型在预测网络攻击后果的应用 

**Authors**: Bipin Chhetri, Akbar Siami Namin  

**Link**: [PDF](https://arxiv.org/pdf/2508.13030)  

**Abstract**: Cyberattacks are increasing, and securing against such threats is costing industries billions of dollars annually. Threat Modeling, that is, comprehending the consequences of these attacks, can provide critical support to cybersecurity professionals, enabling them to take timely action and allocate resources that could be used elsewhere. Cybersecurity is heavily dependent on threat modeling, as it assists security experts in assessing and mitigating risks related to identifying vulnerabilities and threats. Recently, there has been a pressing need for automated methods to assess attack descriptions and forecast the future consequences of the increasing complexity of cyberattacks. This study examines how Natural Language Processing (NLP) and deep learning can be applied to analyze the potential impact of cyberattacks by leveraging textual descriptions from the MITRE Common Weakness Enumeration (CWE) database. We emphasize classifying attack consequences into five principal categories: Availability, Access Control, Confidentiality, Integrity, and Other. This paper investigates the use of Bidirectional Encoder Representations from Transformers (BERT) in combination with Hierarchical Attention Networks (HANs) for Multi-label classification, evaluating their performance in comparison with conventional CNN and LSTM-based models. Experimental findings show that BERT achieves an overall accuracy of $0.972$, far higher than conventional deep learning models in multi-label classification. HAN outperforms baseline forms of CNN and LSTM-based models on specific cybersecurity labels. However, BERT consistently achieves better precision and recall, making it more suitable for predicting the consequences of a cyberattack. 

**Abstract (ZH)**: 网络攻击日益增多，抵御这些威胁的代价每年使各行各业支出数十亿美元。通过对这些攻击后果的威胁建模，可以为网络安全专业人员提供关键支持，使他们能够及时采取行动并合理分配资源。威胁建模对于网络安全至关重要，因为它有助于安全专家评估和减轻识别漏洞和威胁相关的风险。最近，迫切需要自动化方法来评估攻击描述并预测日益复杂的网络攻击的未来后果。本研究探讨了如何利用自然语言处理（NLP）和深度学习分析MITRE通用弱点枚举（CWE）数据库中的文本描述，从而评估网络攻击的潜在影响。本文研究了使用双向编码器表示变换器（BERT）与层次注意力网络（HAN）进行多标签分类的方法，并评估了它们在多标签分类中的性能，与传统的基于CNN和LSTM的模型相比。实验结果表明，BERT的整体准确率达到了0.972，远高于传统的深度学习模型。HAN在特定的网络安全标签上优于基于CNN和LSTM的基本模型。然而，BERT在精度和召回率方面始终表现更优，使其更适合预测网络攻击的后果。 

---
# Vitamin N: Benefits of Different Forms of Public Greenery for Urban Health 

**Title (ZH)**: 维生素N：不同形式的公共绿化对城市健康的好处 

**Authors**: Sanja Šćepanović, Sagar Joglekar, Stephen Law, Daniele Quercia, Ke Zhou, Alice Battiston, Rossano Schifanella  

**Link**: [PDF](https://arxiv.org/pdf/2508.12998)  

**Abstract**: Urban greenery is often linked to better health, yet findings from past research have been inconsistent. One reason is that official greenery metrics measure the amount or nearness of greenery but ignore how often people actually may potentially see or use it in daily life. To address this gap, we introduced a new classification that separates on-road greenery, which people see while walking through streets, from off-road greenery, which requires planned visits. We did so by combining aerial imagery of Greater London and greenery data from OpenStreetMap with quantified greenery from over 100,000 Google Street View images and accessibility estimates based on 160,000 road segments. We linked these measures to 7.45 billion medical prescriptions issued by the National Health Service and processed through our methodology. These prescriptions cover five conditions: diabetes, hypertension, asthma, depression, and anxiety, as well as opioid use. As hypothesized, we found that green on-road was more strongly linked to better health than four widely used official measures. For example, hypertension prescriptions dropped by 3.68% in wards with on-road greenery above the median citywide level compared to those below it. If all below-median wards reached the citywide median in on-road greenery, prescription costs could fall by up to £3.15 million each year. These results suggest that greenery seen in daily life may be more relevant than public yet secluded greenery, and that official metrics commonly used in the literature have important limitations. 

**Abstract (ZH)**: 城市绿化与其更好的健康影响之间存在关联，但以往研究结果不一。为解决这一问题，我们引入了一种新的分类方法，将人们在街道上步行时可见的绿化（有路绿化）与需要计划拜访的绿化（无路绿化）分开。通过将大伦敦地区航拍图像与OpenStreetMap的绿化数据结合超过100,000张Google街景图像的量化绿化信息以及基于160,000条道路段的可达性估计，我们建立了这些指标，并将其与英国国家医疗服务体系发出的74.5亿份医疗处方（涵盖糖尿病、高血压、哮喘、抑郁和焦虑，以及阿片类药物使用）联系起来。正如预期的那样，我们发现有路绿化与更好的健康状况之间的关联比四个广泛使用的官方指标更强。例如，与城市平均水平低于中位数的地区相比，城市平均水平高于中位数且有路绿化较多的区县高血压处方减少了3.68%。如果所有低于中位数的区县都能达到城市平均水平的中位数有路绿化，每年的处方成本可能会降低高达315万英镑。这些结果表明，日常可见的绿化可能比公共但隔离的绿化更加相关，而文献中常用的官方指标存在重要局限性。 

---
# Kourkoutas-Beta: A Sunspike-Driven Adam Optimizer with Desert Flair 

**Title (ZH)**: Kourkoutas-Beta：一种具有沙漠风情的Sunspike驱动Adam优化器 

**Authors**: Stavros C. Kassinos  

**Link**: [PDF](https://arxiv.org/pdf/2508.12996)  

**Abstract**: Transformer neural networks are increasingly used for physics-based problems. In data-driven PDE surrogates, training samples from varying boundary and initial conditions can cause erratic losses and spiky gradients; in physics-informed neural networks (PINNs), stiff composite losses amplify this effect.
We introduce Kourkoutas-Beta, an Adam-style optimizer where the fixed second-moment discount beta2 is replaced by a layer-wise dynamic value driven by a bounded ``sunspike'' ratio: the current pooled gradient norm divided by an exponential moving average (EMA) of past norms, squashed to the interval [0,1). Spikes lower beta2 toward beta2_min; calm phases keep it near beta2_max. Options include leaky-AMSGrad (decay), trust-region clipping (max_ratio), adaptive tiny terms, and several bias-correction modes ``none'', ``beta2max'', ``exact'). With all features off and bias_correction=``none'', the method is exactly Adam.
We test on four settings: (i) a Transformer PDE surrogate (Heat2D), (ii) a 3D PINN for heat conduction (Heat3D), (iii) a lightweight MLX synthetic task with jitter and rare-trigger bursts, and (iv) a character-level Transformer on 30 MB of enwik8 (small-enwik8). Kourkoutas-Beta improves stability and final loss versus fixed-beta2 Adam. On small-enwik8 it lowers bits-per-character by about 38% vs Adam-0.95 and about 58% vs Adam-0.999 over 10 seeds, with smaller variance. The method remains drop-in, with runtime overhead comparable to Adam in testbeds A-C and within single-digit percent in testbed D. It preserves Adam-style convergence guarantees while improving robustness under spiky gradients. 

**Abstract (ZH)**: 基于Transformer的神经网络在物理问题中 increasingly 被用于数据驱动的偏微分方程代理模型中。在不同的边界和初始条件的训练样本下，可能会导致不稳定的损失和突变的梯度；在物理知情神经网络（PINNs）中，刚性的复合损失会放大这一效应。引入了 Kourkoutas-Beta 优化器，这是一种 Adam 风格的优化器，其中固定的第二矩折扣因子 β2 被替换为由有界“太阳突变”比值驱动的逐层动态值：当前汇聚的梯度范数与过去的范数的指数移动平均值的比值，被压缩到区间 [0,1)。突变会使得 β2 向 β2_min 降低；平稳阶段则使 β2 保持在 β2_max 附近。该方法包括带泄漏的 AMSGrad（衰减）、信任区域修剪（max_ratio）、自适应微小项以及几种偏置校正模式（“none”、“beta2max”、“exact”）。在所有功能关闭且偏置校正为“none”的情况下，该方法等同于 Adam。在四个测试设置下进行测试：（i）一个基于Transformer的PDE代理模型（Heat2D），（ii）一个用于热传导的3D PINN（Heat3D），（iii）一个具有抖动和罕见触发突发的轻量级MLX合成任务，以及（iv）一个基于字符级Transformer的30 MB enwik8数据集任务（small-enwik8）。Kourkoutas-Beta 在增强稳定性及最终损失方面优于固定 β2 的 Adam。在 small-enwik8 上，它将每个字符的比特数降低了约 38%（相对于 Adam-0.95）和约 58%（相对于 Adam-0.999），且方差较小。该方法保持了 Adam 式的收敛保证，同时在突变梯度下提高了鲁棒性。 

---
# SL-ACC: A Communication-Efficient Split Learning Framework with Adaptive Channel-wise Compression 

**Title (ZH)**: SL-ACC：一种适应性通道压缩的通信高效分割学习框架 

**Authors**: Zehang Lin, Zheng Lin, Miao Yang, Jianhao Huang, Yuxin Zhang, Zihan Fang, Xia Du, Zhe Chen, Shunzhi Zhu, Wei Ni  

**Link**: [PDF](https://arxiv.org/pdf/2508.12984)  

**Abstract**: The increasing complexity of neural networks poses a significant barrier to the deployment of distributed machine learning (ML) on resource-constrained devices, such as federated learning (FL). Split learning (SL) offers a promising solution by offloading the primary computing load from edge devices to a server via model partitioning. However, as the number of participating devices increases, the transmission of excessive smashed data (i.e., activations and gradients) becomes a major bottleneck for SL, slowing down the model training. To tackle this challenge, we propose a communication-efficient SL framework, named SL-ACC, which comprises two key components: adaptive channel importance identification (ACII) and channel grouping compression (CGC). ACII first identifies the contribution of each channel in the smashed data to model training using Shannon entropy. Following this, CGC groups the channels based on their entropy and performs group-wise adaptive compression to shrink the transmission volume without compromising training accuracy. Extensive experiments across various datasets validate that our proposed SL-ACC framework takes considerably less time to achieve a target accuracy than state-of-the-art benchmarks. 

**Abstract (ZH)**: 基于通信效率的分学习框架SL-ACC：自适应信道重要性识别与信道分组压缩 

---
# Multi-Phase Automated Segmentation of Dental Structures in CBCT Using a Lightweight Auto3DSeg and SegResNet Implementation 

**Title (ZH)**: 基于 Lightweight Auto3DSeg 和 SegResNet 的CBCT牙结构多阶段自动化分割实现 

**Authors**: Dominic LaBella, Keshav Jha, Jared Robbins, Esther Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12962)  

**Abstract**: Cone-beam computed tomography (CBCT) has become an invaluable imaging modality in dentistry, enabling 3D visualization of teeth and surrounding structures for diagnosis and treatment planning. Automated segmentation of dental structures in CBCT can efficiently assist in identifying pathology (e.g., pulpal or periapical lesions) and facilitate radiation therapy planning in head and neck cancer patients. We describe the DLaBella29 team's approach for the MICCAI 2025 ToothFairy3 Challenge, which involves a deep learning pipeline for multi-class tooth segmentation. We utilized the MONAI Auto3DSeg framework with a 3D SegResNet architecture, trained on a subset of the ToothFairy3 dataset (63 CBCT scans) with 5-fold cross-validation. Key preprocessing steps included image resampling to 0.6 mm isotropic resolution and intensity clipping. We applied an ensemble fusion using Multi-Label STAPLE on the 5-fold predictions to infer a Phase 1 segmentation and then conducted tight cropping around the easily segmented Phase 1 mandible to perform Phase 2 segmentation on the smaller nerve structures. Our method achieved an average Dice of 0.87 on the ToothFairy3 challenge out-of-sample validation set. This paper details the clinical context, data preparation, model development, results of our approach, and discusses the relevance of automated dental segmentation for improving patient care in radiation oncology. 

**Abstract (ZH)**: 锥束计算机断层成像（CBCT）已成为牙科中不可或缺的成像技术，能够实现牙齿及其周围结构的3D可视化，用于诊断和治疗计划。在头颈部癌症患者中，CBCT中牙齿结构的自动分割可以有效辅助识别病理（如牙髓或根尖周病变）并促进放射治疗计划。我们描述了DLaBella29团队参加MICCAI 2025 ToothFairy3挑战赛的方法，涉及一种用于多类牙齿分割的深度学习管道。我们使用MONAI Auto3DSeg框架结合3D SegResNet架构，在ToothFairy3数据集的子集（63例CBCT扫描）上进行了5折交叉验证训练。关键技术预处理步骤包括将图像重采样为0.6 mm等向性分辨率并进行强度剪裁。我们采用Multi-Label STAPLE对5折预测进行集成融合以推断第一阶段分割，然后在第一阶段容易分割的下颌周围进行精确裁剪，以在较小的神经结构上执行第二阶段分割。我们的方法在ToothFairy3挑战赛的外部验证集上实现了平均Dice值为0.87。本文详细介绍了临床背景、数据准备、模型开发、方法效果和自动化牙齿分割在放射肿瘤学中改善患者护理方面的相关性。 

---
# SEDEG:Sequential Enhancement of Decoder and Encoder's Generality for Class Incremental Learning with Small Memory 

**Title (ZH)**: SEDEG：面向小内存环境下类增量学习的解码器和编码器顺序增强方法 

**Authors**: Hongyang Chen, Shaoling Pu, Lingyu Zheng, Zhongwu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.12932)  

**Abstract**: In incremental learning, enhancing the generality of knowledge is crucial for adapting to dynamic data inputs. It can develop generalized representations or more balanced decision boundaries, preventing the degradation of long-term knowledge over time and thus mitigating catastrophic forgetting. Some emerging incremental learning methods adopt an encoder-decoder architecture and have achieved promising results. In the encoder-decoder achitecture, improving the generalization capabilities of both the encoder and decoder is critical, as it helps preserve previously learned knowledge while ensuring adaptability and robustness to new, diverse data inputs. However, many existing continual methods focus solely on enhancing one of the two components, which limits their effectiveness in mitigating catastrophic forgetting. And these methods perform even worse in small-memory scenarios, where only a limited number of historical samples can be stored. To mitigate this limitation, we introduces SEDEG, a two-stage training framework for vision transformers (ViT), focusing on sequentially improving the generality of both Decoder and Encoder. Initially, SEDEG trains an ensembled encoder through feature boosting to learn generalized representations, which subsequently enhance the decoder's generality and balance the classifier. The next stage involves using knowledge distillation (KD) strategies to compress the ensembled encoder and develop a new, more generalized encoder. This involves using a balanced KD approach and feature KD for effective knowledge transfer. Extensive experiments on three benchmark datasets show SEDEG's superior performance, and ablation studies confirm the efficacy of its components. The code is available at this https URL. 

**Abstract (ZH)**: Incremental 学习中，增强知识的普适性对于适应动态数据输入至关重要。它能够发展出更为通用的表示或更加平衡的决策边界，防止长期知识的退化，从而减轻灾难性遗忘。一些新兴的增量学习方法采用了编码器-解码器架构，并取得了令人鼓舞的结果。在编码器-解码器架构中，提高编码器和解码器的泛化能力至关重要，这有助于保留先前学习的知识，同时确保对新、多样数据输入的适应性和鲁棒性。然而，许多现有的持续学习方法仅专注于增强这两个组件中的一个，这限制了它们在减轻灾难性遗忘方面的有效性。尤其是在小内存场景下，这些方法表现更差，只能存储有限的历史样本。为解决这一限制，我们提出了 SEDEG，这是一种针对视觉变换器 (ViT) 的两阶段训练框架，专注于按顺序提高解码器和编码器的普适性。初始阶段，SEDEG 通过特征增强训练集成编码器以学习通用表示，随后增强解码器的普适性和平衡分类器。第二阶段通过知识蒸馏 (KD) 策略压缩集成编码器并开发出新的更通用的编码器，这涉及使用平衡KD方法和特征KD进行有效的知识转移。在三个基准数据集上的广泛实验展示了 SEDEG 的优越性能，并且消融研究证实了其组件的有效性。代码可在此处访问：这个 URL。 

---
# Learning local and global prototypes with optimal transport for unsupervised anomaly detection and localization 

**Title (ZH)**: 使用最优传输学习局部和全局原型进行无监督异常检测和定位 

**Authors**: Robin Trombetta, Carole Lartizien  

**Link**: [PDF](https://arxiv.org/pdf/2508.12927)  

**Abstract**: Unsupervised anomaly detection aims to detect defective parts of a sample by having access, during training, to a set of normal, i.e. defect-free, data. It has many applications in fields, such as industrial inspection or medical imaging, where acquiring labels is costly or when we want to avoid introducing biases in the type of anomalies that can be spotted. In this work, we propose a novel UAD method based on prototype learning and introduce a metric to compare a structured set of embeddings that balances a feature-based cost and a spatial-based cost. We leverage this metric to learn local and global prototypes with optimal transport from latent representations extracted with a pre-trained image encoder. We demonstrate that our approach can enforce a structural constraint when learning the prototypes, allowing to capture the underlying organization of the normal samples, thus improving the detection of incoherencies in images. Our model achieves performance that is on par with strong baselines on two reference benchmarks for anomaly detection on industrial images. The code is available at this https URL. 

**Abstract (ZH)**: 无监督异常检测旨在通过在训练过程中访问一组正常、即无缺陷的数据，来检测样本中的缺陷部分。它在工业检测或医学影像等领域有很多应用，这些领域获取标签的成本很高，或者我们希望避免在可检测的异常类型中引入偏差。在本工作中，我们提出了一种基于原型学习的新型无监督异常检测方法，并引入了一种度量标准来比较结构化的嵌入集合，该度量标准平衡了基于特征的成本和基于空间的成本。我们利用这种度量标准，通过对预训练图像编码器提取的潜在表示进行_optimal transport_学习局部和全局原型。我们证明，我们的方法可以在学习原型时施加结构约束，从而捕获正常样本的潜在组织结构，进而提高图像中不一致性检测的性能。我们的模型在两个工业图像异常检测基准上的性能与强基线相当。代码可在以下链接获取：this https URL。 

---
# SecFSM: Knowledge Graph-Guided Verilog Code Generation for Secure Finite State Machines in Systems-on-Chip 

**Title (ZH)**: SecFSM：知识图谱引导的片上系统中安全有限状态机的Verilog代码生成 

**Authors**: Ziteng Hu, Yingjie Xia, Xiyuan Chen, Li Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12910)  

**Abstract**: Finite State Machines (FSMs) play a critical role in implementing control logic for Systems-on-Chip (SoC). Traditionally, FSMs are implemented by hardware engineers through Verilog coding, which is often tedious and time-consuming. Recently, with the remarkable progress of Large Language Models (LLMs) in code generation, LLMs have been increasingly explored for automating Verilog code generation. However, LLM-generated Verilog code often suffers from security vulnerabilities, which is particularly concerning for security-sensitive FSM implementations. To address this issue, we propose SecFSM, a novel method that leverages a security-oriented knowledge graph to guide LLMs in generating more secure Verilog code. Specifically, we first construct a FSM Security Knowledge Graph (FSKG) as an external aid to LLMs. Subsequently, we analyze users' requirements to identify vulnerabilities and get a list of vulnerabilities in the requirements. Then, we retrieve knowledge from FSKG based on the vulnerabilities list. Finally, we construct security prompts based on the security knowledge for Verilog code generation. To evaluate SecFSM, we build a dedicated dataset collected from academic datasets, artificial datasets, papers, and industrial cases. Extensive experiments demonstrate that SecFSM outperforms state-of-the-art baselines. In particular, on a benchmark of 25 security test cases evaluated by DeepSeek-R1, SecFSM achieves an outstanding pass rate of 21/25. 

**Abstract (ZH)**: 有限状态机（FSMs）在片上系统（SoC）的控制逻辑实现中发挥着关键作用。传统上，FSMs通过Verilog编码由硬件工程师实现，这通常耗时且繁琐。近年来，随着大型语言模型（LLMs）在代码生成方面取得显著进展，LLMs越来越多地被探索用于自动化Verilog代码生成。然而，LLM生成的Verilog代码往往存在安全漏洞，这特别令人担忧，尤其是在安全敏感的FSM实现中。为了解决这一问题，我们提出了一种名为SecFSM的新方法，该方法利用以安全性为导向的知识图谱来指导LLMs生成更安全的Verilog代码。具体而言，我们首先构建了一个FSM安全知识图谱（FSKG）作为LLMs的外部辅助工具。随后，我们分析用户需求以识别漏洞并列出需求中的漏洞。然后，我们根据漏洞列表从FSKG中检索知识。最后，我们基于安全知识构建安全提示以用于Verilog代码生成。为了评估SecFSM，我们构建了一个专用的数据集，该数据集收集自学术数据集、人工数据集、论文和工业案例。广泛实验表明，SecFSM优于最先进的基线方法。特别是在由DeepSeek-R1评估的25个安全测试案例的基准测试中，SecFSM实现了显着的通过率21/25。 

---
# A Stitch in Time Saves Nine: Proactive Self-Refinement for Language Models 

**Title (ZH)**: 及时缝合，省时九倍：语言模型的前瞻性自我精炼 

**Authors**: Jinyi Han, Xinyi Wang, Haiquan Zhao, Tingyun li, Zishang Jiang, Sihang Jiang, Jiaqing Liang, Xin Lin, Weikang Zhou, Zeye Sun, Fei Yu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12903)  

**Abstract**: Recent advances in self-refinement have demonstrated significant potential for improving the outputs of large language models (LLMs) through iterative refinement. However, most existing self-refinement methods rely on a reactive process with a fixed number of iterations, making it difficult to determine the optimal timing and content of refinement based on the evolving generation context. Inspired by the way humans dynamically refine their thoughts during execution, we propose ProActive Self-Refinement (PASR), a novel method that enables LLMs to refine their outputs during the generation process. Unlike methods that regenerate entire responses, PASR proactively decides whether, when, and how to refine based on the model's internal state and evolving context. We conduct extensive experiments on a diverse set of 10 tasks to evaluate the effectiveness of PASR. Experimental results show that PASR significantly enhances problem-solving performance. In particular, on Qwen3-8B, PASR reduces average token consumption by 41.6 percent compared to standard generation, while also achieving an 8.2 percent improvement in accuracy. Our code and all baselines used in the paper are available in the GitHub. 

**Abstract (ZH)**: Recent Advances in ProActive Self-Refinement for Enhancing the Performance of Large Language Models 

---
# CTFlow: Video-Inspired Latent Flow Matching for 3D CT Synthesis 

**Title (ZH)**: CTFlow: 由视频启发的潜空间流匹配三维CT合成 

**Authors**: Jiayi Wang, Hadrien Reynaud, Franciskus Xaverius Erick, Bernhard Kainz  

**Link**: [PDF](https://arxiv.org/pdf/2508.12900)  

**Abstract**: Generative modelling of entire CT volumes conditioned on clinical reports has the potential to accelerate research through data augmentation, privacy-preserving synthesis and reducing regulator-constraints on patient data while preserving diagnostic signals. With the recent release of CT-RATE, a large-scale collection of 3D CT volumes paired with their respective clinical reports, training large text-conditioned CT volume generation models has become achievable. In this work, we introduce CTFlow, a 0.5B latent flow matching transformer model, conditioned on clinical reports. We leverage the A-VAE from FLUX to define our latent space, and rely on the CT-Clip text encoder to encode the clinical reports. To generate consistent whole CT volumes while keeping the memory constraints tractable, we rely on a custom autoregressive approach, where the model predicts the first sequence of slices of the volume from text-only, and then relies on the previously generated sequence of slices and the text, to predict the following sequence. We evaluate our results against state-of-the-art generative CT model, and demonstrate the superiority of our approach in terms of temporal coherence, image diversity and text-image alignment, with FID, FVD, IS scores and CLIP score. 

**Abstract (ZH)**: 基于临床报告条件下的CT整卷生成模型有望通过数据增强、隐私保护合成以及减少患者数据监管约束来加速研究，同时保留诊断信号。随着CT-RATE的推出，一个大规模的3D CT整卷及其相应的临床报告集合，训练大型文本条件下的CT整卷生成模型变得可行。在此工作中，我们介绍CTFlow，一个基于临床报告条件下的0.5B潜空间流动匹配转换器模型。我们利用FLUX的A-VAE来定义潜空间，并依赖CT-Clip文本编码器对临床报告进行编码。为了生成一致的完整CT整卷并保持内存约束可处理，我们依赖一种自回归方法，其中模型首先仅从文本中预测体积的第一个序列切片，然后依赖之前生成的序列切片和文本来预测后续序列。我们将我们的结果与最先进的生成CT模型进行评估，并在时空连贯性、图像多样性及文本-图像对齐方面展示了我们方法的优越性，通过FID、FVD、IS评分和CLIP评分进行验证。 

---
# One-Class Intrusion Detection with Dynamic Graphs 

**Title (ZH)**: 基于动态图的一类入侵检测 

**Authors**: Aleksei Liuliakov, Alexander Schulz, Luca Hermes, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2508.12885)  

**Abstract**: With the growing digitalization all over the globe, the relevance of network security becomes increasingly important. Machine learning-based intrusion detection constitutes a promising approach for improving security, but it bears several challenges. These include the requirement to detect novel and unseen network events, as well as specific data properties, such as events over time together with the inherent graph structure of network communication. In this work, we propose a novel intrusion detection method, TGN-SVDD, which builds upon modern dynamic graph modelling and deep anomaly detection. We demonstrate its superiority over several baselines for realistic intrusion detection data and suggest a more challenging variant of the latter. 

**Abstract (ZH)**: 随着全球数字化程度的不断加深，网络安全的重要性日益凸显。基于机器学习的入侵检测构成了提升安全性的有promise的方法，但同时也面临着几大挑战，包括检测新型且未见过的网络事件，以及数据的特定属性，如事件随时间的变化以及网络通信固有的图结构。在本工作中，我们提出了一种新的入侵检测方法TGN-SVDD，该方法基于现代动态图建模和深度异常检测。我们证明了该方法在现实的入侵检测数据中优于几种基准方法，并提出了一种更具挑战性的基准变体。 

---
# Word Meanings in Transformer Language Models 

**Title (ZH)**: Transformer语言模型中的词义 

**Authors**: Jumbly Grindrod, Peter Grindrod  

**Link**: [PDF](https://arxiv.org/pdf/2508.12863)  

**Abstract**: We investigate how word meanings are represented in the transformer language models. Specifically, we focus on whether transformer models employ something analogous to a lexical store - where each word has an entry that contains semantic information. To do this, we extracted the token embedding space of RoBERTa-base and k-means clustered it into 200 clusters. In our first study, we then manually inspected the resultant clusters to consider whether they are sensitive to semantic information. In our second study, we tested whether the clusters are sensitive to five psycholinguistic measures: valence, concreteness, iconicity, taboo, and age of acquisition. Overall, our findings were very positive - there is a wide variety of semantic information encoded within the token embedding space. This serves to rule out certain "meaning eliminativist" hypotheses about how transformer LLMs process semantic information. 

**Abstract (ZH)**: 我们探讨变压器语言模型中词义的表示方式。具体而言，我们关注变压器模型是否采用类似于词汇存储的方式——每个词都有一个包含语义信息的条目。为此，我们提取了RoBERTa-base的词元嵌入空间，并使用k-means聚类将其划分成200个簇。在我们的第一个研究中，我们手动检查了这些簇，以考虑它们是否对语义信息敏感。在第二个研究中，我们测试了这些簇是否对五种心理语言学测量指标——效价、具体性、形象性、禁忌性和习得年龄——敏感。总体而言，我们的发现非常积极——词元嵌入空间中包含了广泛的语义信息，这排除了某些关于变压器大语言模型处理语义信息的“意义消除”假设。 

---
# HRS: Hybrid Representation Framework with Scheduling Awareness for Time Series Forecasting in Crowdsourced Cloud-Edge Platforms 

**Title (ZH)**: HRS：具有调度意识的混合表示框架在众包云边平台的时间序列预测中应用 

**Authors**: Tiancheng Zhang, Cheng Zhang, Shuren Liu, Xiaofei Wang, Shaoyuan Huang, Wenyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12839)  

**Abstract**: With the rapid proliferation of streaming services, network load exhibits highly time-varying and bursty behavior, posing serious challenges for maintaining Quality of Service (QoS) in Crowdsourced Cloud-Edge Platforms (CCPs). While CCPs leverage Predict-then-Schedule architecture to improve QoS and profitability, accurate load forecasting remains challenging under traffic surges. Existing methods either minimize mean absolute error, resulting in underprovisioning and potential Service Level Agreement (SLA) violations during peak periods, or adopt conservative overprovisioning strategies, which mitigate SLA risks at the expense of increased resource expenditure. To address this dilemma, we propose HRS, a hybrid representation framework with scheduling awareness that integrates numerical and image-based representations to better capture extreme load dynamics. We further introduce a Scheduling-Aware Loss (SAL) that captures the asymmetric impact of prediction errors, guiding predictions that better support scheduling decisions. Extensive experiments on four real-world datasets demonstrate that HRS consistently outperforms ten baselines and achieves state-of-the-art performance, reducing SLA violation rates by 63.1% and total profit loss by 32.3%. 

**Abstract (ZH)**: 随着流媒体服务的迅速普及，网络负载表现出高度的时间变化性和突发性，这对Crowdsourced Cloud-Edge Platforms (CCPs)中保持服务质量(QoS)提出了严重挑战。虽然CCPs利用预测-调度架构来提高QoS和盈利能力，但在流量激增的情况下，准确的负载预测仍然具有挑战性。现有方法要么最小化均绝对误差，导致在高峰期出现服务能力不足和可能违反服务水平协议(SLA)，要么采用保守的过度配置策略，虽然减轻了SLA风险，但增加了资源支出。为了应对这一困境，我们提出了一种具有调度意识的混合表示框架HRS，该框架结合了数值和基于图像的表示，以更好地捕捉极端负载动态。此外，我们引入了一种调度意识损失(SAL)，以捕捉预测误差的不对称影响，从而指导更有利于调度决策的预测。在四个真实世界数据集上的广泛实验表明，HRS持续优于十个基线方法，并达到最先进的性能，SLA违反率降低了63.1%，总利润损失降低了32.3%。 

---
# Toward Storage-Aware Learning with Compressed Data An Empirical Exploratory Study on JPEG 

**Title (ZH)**: 面向存储意识的学习与压缩数据：JPEG格式的实证探索研究 

**Authors**: Kichang Lee, Songkuk Kim, JaeYeon Park, JeongGil Ko  

**Link**: [PDF](https://arxiv.org/pdf/2508.12833)  

**Abstract**: On-device machine learning is often constrained by limited storage, particularly in continuous data collection scenarios. This paper presents an empirical study on storage-aware learning, focusing on the trade-off between data quantity and quality via compression. We demonstrate that naive strategies, such as uniform data dropping or one-size-fits-all compression, are suboptimal. Our findings further reveal that data samples exhibit varying sensitivities to compression, supporting the feasibility of a sample-wise adaptive compression strategy. These insights provide a foundation for developing a new class of storage-aware learning systems. The primary contribution of this work is the systematic characterization of this under-explored challenge, offering valuable insights that advance the understanding of storage-aware learning. 

**Abstract (ZH)**: 在设备上进行的机器学习往往受限于有限的存储空间，尤其是在连续数据收集场景中。本文对存储感知学习进行了实证研究，关注数据数量与质量之间的权衡，通过压缩来实现。我们证明了朴素策略，如均匀数据丢弃或一刀切的压缩方法，是不理想的。研究发现进一步表明，数据样本对压缩的敏感性各异，支持了样本级别的自适应压缩策略的可行性。这些洞察为我们开发新的存储感知学习系统提供了基础。本文的主要贡献是对这一未充分探索的挑战进行了系统的刻画，提供了宝贵的认识，推动了对存储感知学习的理解。 

---
# Context Matters: Incorporating Target Awareness in Conversational Abusive Language Detection 

**Title (ZH)**: 情境重要：在对话式网络谩骂检测中融入目标意识 

**Authors**: Raneem Alharthi, Rajwa Alharthi, Aiqi Jiang, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2508.12828)  

**Abstract**: Abusive language detection has become an increasingly important task as a means to tackle this type of harmful content in social media. There has been a substantial body of research developing models for determining if a social media post is abusive or not; however, this research has primarily focused on exploiting social media posts individually, overlooking additional context that can be derived from surrounding posts. In this study, we look at conversational exchanges, where a user replies to an earlier post by another user (the parent tweet). We ask: does leveraging context from the parent tweet help determine if a reply post is abusive or not, and what are the features that contribute the most? We study a range of content-based and account-based features derived from the context, and compare this to the more widely studied approach of only looking at the features from the reply tweet. For a more generalizable study, we test four different classification models on a dataset made of conversational exchanges (parent-reply tweet pairs) with replies labeled as abusive or not. Our experiments show that incorporating contextual features leads to substantial improvements compared to the use of features derived from the reply tweet only, confirming the importance of leveraging context. We observe that, among the features under study, it is especially the content-based features (what is being posted) that contribute to the classification performance rather than account-based features (who is posting it). While using content-based features, it is best to combine a range of different features to ensure improved performance over being more selective and using fewer features. Our study provides insights into the development of contextualized abusive language detection models in realistic settings involving conversations. 

**Abstract (ZH)**: 社交媒体中虐待性语言检测已成为一项日益重要的任务，用于应对社交媒体上的有害内容。尽管已有大量研究开发模型来判断一条社交媒体帖子是否为虐待性内容，但这些研究主要侧重于独立分析单个帖子，忽略了来自其他帖子的额外上下文信息。在本研究中，我们关注用户的对话交流，即用户回复早前其他用户的帖子（父微博）。我们提出的问题是：利用父微博的上下文信息是否有助于判断回复帖子是否为虐待性内容？哪些特征对分类贡献最大？我们研究了从上下文派生的内容相关和账号相关特征，并将其与仅研究回复帖子特征的广泛研究方法进行了比较。为了使研究更具普适性，我们在包含父微博-回复微博配对的数据集上测试了四种不同的分类模型，这些回复帖子被标记为虐待性或非虐待性。我们的实验表明，结合上下文特征比仅使用回复帖子的特征能够带来显著提升，证实了利用上下文信息的重要性。我们观察到，在研究的特征中，内容相关特征（帖子的内容）对分类性能的贡献尤为显著，而非账号相关特征（发帖者的信息）。在使用内容相关特征时，最好结合多种不同的特征以确保性能提升，而不仅仅是选择较少的特征。我们的研究为在涉及对话交流的现实环境中开发上下文化的虐待性语言检测模型提供了洞察。 

---
# Learning to Steer: Input-dependent Steering for Multimodal LLMs 

**Title (ZH)**: 学习指引：输入依赖的多模态LLM指引 

**Authors**: Jayneel Parekh, Pegah Khayatan, Mustafa Shukor, Arnaud Dapogny, Alasdair Newson, Matthieu Cord  

**Link**: [PDF](https://arxiv.org/pdf/2508.12815)  

**Abstract**: Steering has emerged as a practical approach to enable post-hoc guidance of LLMs towards enforcing a specific behavior. However, it remains largely underexplored for multimodal LLMs (MLLMs); furthermore, existing steering techniques, such as mean steering, rely on a single steering vector, applied independently of the input query. This paradigm faces limitations when the desired behavior is dependent on the example at hand. For example, a safe answer may consist in abstaining from answering when asked for an illegal activity, or may point to external resources or consultation with an expert when asked about medical advice. In this paper, we investigate a fine-grained steering that uses an input-specific linear shift. This shift is computed using contrastive input-specific prompting. However, the input-specific prompts required for this approach are not known at test time. Therefore, we propose to train a small auxiliary module to predict the input-specific steering vector. Our approach, dubbed as L2S (Learn-to-Steer), demonstrates that it reduces hallucinations and enforces safety in MLLMs, outperforming other static baselines. 

**Abstract (ZH)**: 细粒度导向在多模态大模型中的应用：学习导向(L2S) 

---
# Next Visual Granularity Generation 

**Title (ZH)**: 下一级视觉粒度生成 

**Authors**: Yikai Wang, Zhouxia Wang, Zhonghua Wu, Qingyi Tao, Kang Liao, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2508.12811)  

**Abstract**: We propose a novel approach to image generation by decomposing an image into a structured sequence, where each element in the sequence shares the same spatial resolution but differs in the number of unique tokens used, capturing different level of visual granularity. Image generation is carried out through our newly introduced Next Visual Granularity (NVG) generation framework, which generates a visual granularity sequence beginning from an empty image and progressively refines it, from global layout to fine details, in a structured manner. This iterative process encodes a hierarchical, layered representation that offers fine-grained control over the generation process across multiple granularity levels. We train a series of NVG models for class-conditional image generation on the ImageNet dataset and observe clear scaling behavior. Compared to the VAR series, NVG consistently outperforms it in terms of FID scores (3.30 -> 3.03, 2.57 ->2.44, 2.09 -> 2.06). We also conduct extensive analysis to showcase the capability and potential of the NVG framework. Our code and models will be released. 

**Abstract (ZH)**: 我们提出了一种新颖的图像生成方法，通过将图像分解为结构化序列，其中序列中的每个元素具有相同的空间分辨率但使用独特标记的数量不同，从而捕捉不同级别的视觉粒度。图像生成通过我们新引入的Next Visual Granularity（NVG）生成框架进行，该框架从空白图像开始生成视觉粒度序列，并以结构化的方式逐步细化，从全局布局到细项。这一迭代过程编码了一个分层表示，提供了对多粒度级别生成过程的细粒度控制。我们在ImageNet数据集上训练了一系列NVG模型进行类别条件图像生成，并观察到明显的缩放行为。与VAR系列相比，NVG在FID分数上始终表现更优（3.30 -> 3.03，2.57 -> 2.44，2.09 -> 2.06）。我们还进行了广泛的分析以展示NVG框架的能力和潜力。我们的代码和模型将开源发布。 

---
# Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward 

**Title (ZH)**: 原子搜索者：通过细粒度原子思维奖励增强自主深度研究 

**Authors**: Yong Deng, Guoqing Wang, Zhenzhe Ying, Xiaofeng Wu, Jinzhen Lin, Wenwen Xiong, Yuqin Dai, Shuo Yang, Zhanwei Zhang, Qiwen Wang, Yang Qin, Changhua Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.12800)  

**Abstract**: Large language models (LLMs) exhibit remarkable problem-solving abilities, but struggle with complex tasks due to static internal knowledge. Retrieval-Augmented Generation (RAG) enhances access to external information, yet remains limited in multi-hop reasoning and strategic search due to rigid workflows. Recent advancements in agentic deep research empower LLMs to autonomously reason, search, and synthesize information. However, current approaches relying on outcome-based reinforcement learning (RL) face critical issues such as conflicting gradients and reward sparsity, limiting performance gains and training efficiency. To address these, we first propose Atomic Thought, a novel LLM thinking paradigm that decomposes reasoning into fine-grained functional units. These units are supervised by Reasoning Reward Models (RRMs), which provide Atomic Thought Rewards (ATR) for fine-grained guidance. Building on this, we propose Atom-Searcher, a novel RL framework for agentic deep research that integrates Atomic Thought and ATR. Atom-Searcher uses a curriculum-inspired reward schedule, prioritizing process-level ATR early and transitioning to outcome rewards, accelerating convergence on effective reasoning paths. Experiments on seven benchmarks show consistent improvements over the state-of-the-art. Key advantages include: (1) Atom-Searcher scales computation at test-time. (2) Atomic Thought provides supervision anchors for RRMs, bridging deep research tasks and RRMs. (3) Atom-Searcher exhibits more interpretable, human-like reasoning patterns. 

**Abstract (ZH)**: 大型语言模型（LLMs）表现出卓越的问题解决能力，但因内部知识静态化而在复杂任务上遇到挑战。检索增强生成（RAG）提升了对外部信息的访问，但由于僵化的流程限制，在多跳推理和策略性搜索方面仍受限制。近期自主深度研究的进步使LLMs能够自主推理、搜索和综合信息。然而，当前依赖于基于结果的强化学习（RL）的方法面临关键问题，如梯度冲突和奖励稀疏性，这限制了性能提升和训练效率。为解决这些问题，我们首先提出了原子思想，这是一种新颖的LLM思维范式，将推理分解为细粒度的功能单元。这些单元通过推理奖励模型（RRMs）监督，提供原子思想奖励（ATR）以提供细粒度指导。在此基础上，我们提出了原子搜索者，这是一种结合了原子思想和ATR的新型RL框架，用于自主深度研究。原子搜索者采用基于课程的学习奖励计划，优先在过程中提供细粒度奖励，并过渡到结果奖励，从而加速对高效推理路径的收敛。在七个基准测试上的实验结果显示了一致的性能提升。关键优势包括：（1）原子搜索者在测试时扩展计算。（2）原子思想为RRMs提供监督锚点，连接深度研究任务和RRMs。（3）原子搜索者表现出更可解释、类人的推理模式。 

---
# A Shift in Perspective on Causality in Domain Generalization 

**Title (ZH)**: 域泛化中因果关系视角的转变 

**Authors**: Damian Machlanski, Stephanie Riley, Edward Moroshko, Kurt Butler, Panagiotis Dimitrakopoulos, Thomas Melistas, Akchunya Chanchal, Steven McDonagh, Ricardo Silva, Sotirios A. Tsaftaris  

**Link**: [PDF](https://arxiv.org/pdf/2508.12798)  

**Abstract**: The promise that causal modelling can lead to robust AI generalization has been challenged in recent work on domain generalization (DG) benchmarks. We revisit the claims of the causality and DG literature, reconciling apparent contradictions and advocating for a more nuanced theory of the role of causality in generalization. We also provide an interactive demo at this https URL. 

**Abstract (ZH)**: 因果建模能导致稳健的AI泛化的承诺在最近的域泛化（DG）基准研究中受到了挑战。我们重新审视因果性和DG文献中的主张，调和显而易见的矛盾，并倡导一种更细致的因果在泛化中作用的理论。我们还提供了一个交互式演示：![this URL](this https URL)。 

---
# Vehicle detection from GSV imagery: Predicting travel behaviour for cycling and motorcycling using Computer Vision 

**Title (ZH)**: 基于街景图像的车辆检测：利用计算机视觉预测自行车和摩托车出行行为 

**Authors**: Kyriaki, Kokka, Rahul Goel, Ali Abbas, Kerry A. Nice, Luca Martial, SM Labib, Rihuan Ke, Carola Bibiane Schönlieb, James Woodcock  

**Link**: [PDF](https://arxiv.org/pdf/2508.12794)  

**Abstract**: Transportation influence health by shaping exposure to physical activity, air pollution and injury this http URL data on cycling and motorcycling behaviours is scarce, particularly at a global this http URL view imagery, such as Google Street View (GSV), combined with computer vision, is a valuable resource for efficiently capturing travel behaviour this http URL study demonstrates a novel approach using deep learning on street view images to estimate cycling and motorcycling levels across diverse cities this http URL utilized data from 185 global this http URL data on mode shares of cycling and motorcycling estimated using travel surveys or this http URL used GSV images to detect cycles and motorcycles in sampled locations, using 8000 images per this http URL YOLOv4 model, fine-tuned using images from six cities, achieved a mean average precision of 89% for detecting cycles and motorcycles in GSV images.A global prediction model was developed using beta regression with city-level mode shares as outcome, with log transformed explanatory variables of counts of GSV-detected images with cycles and motorcycles, while controlling for population this http URL found strong correlations between GSV motorcycle counts and motorcycle mode share (0.78) and moderate correlations between GSV cycle counts and cycling mode share (0.51).Beta regression models predicted mode shares with $R^2$ values of 0.614 for cycling and 0.612 for motorcycling, achieving median absolute errors (MDAE) of 1.3% and 1.4%, this http URL demonstrated consistent prediction accuracy, though cities like Utrecht and Cali were this http URL model was applied to 60 cities globally for which we didn't have recent mode share this http URL provided estimates for some cities in the Middle East, Latin America and East this http URL computer vision, GSV images capture travel modes and activity, providing insights alongside traditional data sources. 

**Abstract (ZH)**: 交通通过塑造体力活动、空气污染暴露和伤害影响健康，特别是基于全球视角的骑行和摩托车行为数据稀缺，通过将遥感影像与计算机视觉结合，可以有效捕捉出行行为，本研究采用深度学习方法分析遥感影像，以估算不同城市的骑行和摩托车出行水平，利用来自185个全球城市的数据，通过计算机视觉检测遥感影像中的自行车和摩托车，在8000张遥感影像上使用YOLOv4模型，经过六个城市影像的微调，检测遥感影像中自行车和摩托车的平均精度达到89%。开发了一个使用贝塔回归模型的全球预测模型，城市级别的出行比例作为结果变量，通过转换后的遥感影像中检测到的自行车和摩托车数量的计数解释变量来控制人口因素，结果发现遥感影像中的摩托车数量与摩托车出行比例之间存在很强的相关性（0.78），而遥感影像中的自行车数量与自行车出行比例之间存在中等的相关性（0.51）。贝塔回归模型预测出行比例，自行车出行比例的$R^2$值为0.614，摩托车出行比例的$R^2$值为0.612，中位绝对误差分别为1.3%和1.4%，展示了模型具有稳定的预测准确性，尽管部分地区如乌得勒支和卡利的预测效果不佳。该模型应用于60个缺乏近期出行比例数据的城市，为中东、拉丁美洲和东亚的一些城市提供了出行比例的估计。遥感影像和计算机视觉捕捉出行模式和活动，为传统数据源提供了补充见解。 

---
# Bridging Human and LLM Judgments: Understanding and Narrowing the Gap 

**Title (ZH)**: 人类与大规模语言模型判断的桥梁：理解并缩小差距 

**Authors**: Felipe Maia Polo, Xinhe Wang, Mikhail Yurochkin, Gongjun Xu, Moulinath Banerjee, Yuekai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.12792)  

**Abstract**: Large language models are increasingly used as judges (LLM-as-a-judge) to evaluate model outputs at scale, but their assessments often diverge systematically from human judgments. We present Bridge, a unified statistical framework that explicitly bridges human and LLM evaluations under both absolute scoring and pairwise comparison paradigms. Bridge posits a latent human preference score for each prompt-response pair and models LLM deviations as linear transformations of covariates that capture sources of discrepancies. This offers a simple and principled framework for refining LLM ratings and characterizing systematic discrepancies between humans and LLMs. We provide an efficient fitting algorithm with asymptotic guarantees for statistical inference. Using six LLM judges and two benchmarks (BigGen Bench and Chatbot Arena), Bridge achieves higher agreement with human ratings (accuracy, calibration, and KL divergence) and exposes systematic human-LLM gaps. 

**Abstract (ZH)**: 一种统一的统计框架：Bridge，用于桥接人类和大型语言模型的评价（适用于绝对评分和成对比较范式） 

---
# Randomized PCA Forest for Outlier Detection 

**Title (ZH)**: 随机PCA森林异常检测 

**Authors**: Muhammad Rajabinasab, Farhad Pakdaman, Moncef Gabbouj, Peter Schneider-Kamp, Arthur Zimek  

**Link**: [PDF](https://arxiv.org/pdf/2508.12776)  

**Abstract**: We propose a novel unsupervised outlier detection method based on Randomized Principal Component Analysis (PCA). Inspired by the performance of Randomized PCA (RPCA) Forest in approximate K-Nearest Neighbor (KNN) search, we develop a novel unsupervised outlier detection method that utilizes RPCA Forest for outlier detection. Experimental results showcase the superiority of the proposed approach compared to the classical and state-of-the-art methods in performing the outlier detection task on several datasets while performing competitively on the rest. The extensive analysis of the proposed method reflects it high generalization power and its computational efficiency, highlighting it as a good choice for unsupervised outlier detection. 

**Abstract (ZH)**: 基于随机主成分分析的新型无监督异常检测方法 

---
# CRED-SQL: Enhancing Real-world Large Scale Database Text-to-SQL Parsing through Cluster Retrieval and Execution Description 

**Title (ZH)**: CRED-SQL：通过聚类检索和执行描述增强现实世界大规模数据库的文本到SQL解析 

**Authors**: Shaoming Duan, Zirui Wang, Chuanyi Liu, Zhibin Zhu, Yuhao Zhang, Peiyi Han, Liang Yan, Zewu Penge  

**Link**: [PDF](https://arxiv.org/pdf/2508.12769)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved the accuracy of Text-to-SQL systems. However, a critical challenge remains: the semantic mismatch between natural language questions (NLQs) and their corresponding SQL queries. This issue is exacerbated in large-scale databases, where semantically similar attributes hinder schema linking and semantic drift during SQL generation, ultimately reducing model accuracy. To address these challenges, we introduce CRED-SQL, a framework designed for large-scale databases that integrates Cluster Retrieval and Execution Description. CRED-SQL first performs cluster-based large-scale schema retrieval to pinpoint the tables and columns most relevant to a given NLQ, alleviating schema mismatch. It then introduces an intermediate natural language representation-Execution Description Language (EDL)-to bridge the gap between NLQs and SQL. This reformulation decomposes the task into two stages: Text-to-EDL and EDL-to-SQL, leveraging LLMs' strong general reasoning capabilities while reducing semantic deviation. Extensive experiments on two large-scale, cross-domain benchmarks-SpiderUnion and BirdUnion-demonstrate that CRED-SQL achieves new state-of-the-art (SOTA) performance, validating its effectiveness and scalability. Our code is available at this https URL 

**Abstract (ZH)**: Recent advances in大型语言模型（LL
user
把下面的论文内容或标题翻译成中文，要符合，禁止输出多余内容。 

---
# Harnessing Group-Oriented Consistency Constraints for Semi-Supervised Semantic Segmentation in CdZnTe Semiconductors 

**Title (ZH)**: 基于群导向一致约束的半监督语义分割在CdZnTe半导体中的应用 

**Authors**: Peihao Li, Yan Fang, Man Liu, Huihui Bai, Anhong Wang, Yunchao Wei, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12766)  

**Abstract**: Labeling Cadmium Zinc Telluride (CdZnTe) semiconductor images is challenging due to the low-contrast defect boundaries, necessitating annotators to cross-reference multiple views. These views share a single ground truth (GT), forming a unique ``many-to-one'' relationship. This characteristic renders advanced semi-supervised semantic segmentation (SSS) methods suboptimal, as they are generally limited by a ``one-to-one'' relationship, where each image is independently associated with its GT. Such limitation may lead to error accumulation in low-contrast regions, further exacerbating confirmation bias. To address this issue, we revisit the SSS pipeline from a group-oriented perspective and propose a human-inspired solution: the Intra-group Consistency Augmentation Framework (ICAF). First, we experimentally validate the inherent consistency constraints within CdZnTe groups, establishing a group-oriented baseline using the Intra-group View Sampling (IVS). Building on this insight, we introduce the Pseudo-label Correction Network (PCN) to enhance consistency representation, which consists of two key modules. The View Augmentation Module (VAM) improves boundary details by dynamically synthesizing a boundary-aware view through the aggregation of multiple views. In the View Correction Module (VCM), this synthesized view is paired with other views for information interaction, effectively emphasizing salient regions while minimizing noise. Extensive experiments demonstrate the effectiveness of our solution for CdZnTe materials. Leveraging DeepLabV3+ with a ResNet-101 backbone as our segmentation model, we achieve a 70.6\% mIoU on the CdZnTe dataset using only 2 group-annotated data (5\textperthousand). The code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于CdZnTe半导体图像的标签标注因其低对比度缺陷边界具有挑战性，需要标注人员参考多个视图。这些视图共享一个单一的真实标注（GT），形成了独特的“多对一”关系。这一特性使得先进的半监督语义分割（SSS）方法不够理想，因为它们通常受限于“一对一”关系，即每张图像独立关联一个GT。这种限制可能导致低对比度区域的错误积累，进一步加剧确认偏差。为解决这一问题，我们从群体导向的视角回顾了SSS管道，并提出了一种受人类启发的解决方案：Intra-group一致性增强框架（ICAF）。首先，我们通过Intra-group视图采样（IVS）实验验证了CdZnTe群体内的固有一致性约束，建立了群体导向的基础模型。在此基础上，我们引入了伪标签校准网络（PCN）来增强一致性表示，该网络由两个关键模块组成。视图增强模块（VAM）通过多个视图的聚合动态合成边界感知视图，以改进边界细节。在视图校准模块（VCM）中，该合成视图与其他视图配对进行信息交互，有效强调显著区域同时减少噪声。广泛的实验表明，我们的解决方案对CdZnTe材料的有效性。利用具有ResNet-101骨干网络的DeepLabV3+作为分割模型，在仅使用2组标注数据（0.5‰）的情况下，实现了CdZnTe数据集上的70.6% mIoU。代码可在\href{this https URL}{this https URL}获取。 

---
# CLAIRE-DSA: Fluoroscopic Image Classification for Quality Assurance of Computer Vision Pipelines in Acute Ischemic Stroke 

**Title (ZH)**: CLAIRE-DSA：急性缺血性中风计算机视觉管道质量保证的荧光透视图像分类 

**Authors**: Cristo J. van den Berg, Frank G. te Nijenhuis, Mirre J. Blaauboer, Daan T. W. van Erp, Carlijn M. Keppels, Matthijs van der Sluijs, Bob Roozenbeek, Wim van Zwam, Sandra Cornelissen, Danny Ruijters, Ruisheng Su, Theo van Walsum  

**Link**: [PDF](https://arxiv.org/pdf/2508.12755)  

**Abstract**: Computer vision models can be used to assist during mechanical thrombectomy (MT) for acute ischemic stroke (AIS), but poor image quality often degrades performance. This work presents CLAIRE-DSA, a deep learning--based framework designed to categorize key image properties in minimum intensity projections (MinIPs) acquired during MT for AIS, supporting downstream quality control and workflow optimization. CLAIRE-DSA uses pre-trained ResNet backbone models, fine-tuned to predict nine image properties (e.g., presence of contrast, projection angle, motion artefact severity). Separate classifiers were trained on an annotated dataset containing $1,758$ fluoroscopic MinIPs. The model achieved excellent performance on all labels, with ROC-AUC ranging from $0.91$ to $0.98$, and precision ranging from $0.70$ to $1.00$. The ability of CLAIRE-DSA to identify suitable images was evaluated on a segmentation task by filtering poor quality images and comparing segmentation performance on filtered and unfiltered datasets. Segmentation success rate increased from $42%$ to $69%$, $p < 0.001$. CLAIRE-DSA demonstrates strong potential as an automated tool for accurately classifying image properties in DSA series of acute ischemic stroke patients, supporting image annotation and quality control in clinical and research applications. Source code is available at this https URL. 

**Abstract (ZH)**: 基于深度学习的CLAIRE-DSA框架用于急性缺血性中风机械取栓过程中最小强度投影图像的关键图像属性分类 

---
# DCSCR: A Class-Specific Collaborative Representation based Network for Image Set Classification 

**Title (ZH)**: DCSCR：一种用于图像集分类的类特定协作表示表示网络 fod图像集分类yd 

**Authors**: Xizhan Gao, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12745)  

**Abstract**: Image set classification (ISC), which can be viewed as a task of comparing similarities between sets consisting of unordered heterogeneous images with variable quantities and qualities, has attracted growing research attention in recent years. How to learn effective feature representations and how to explore the similarities between different image sets are two key yet challenging issues in this field. However, existing traditional ISC methods classify image sets based on raw pixel features, ignoring the importance of feature learning. Existing deep ISC methods can learn deep features, but they fail to adaptively adjust the features when measuring set distances, resulting in limited performance in few-shot ISC. To address the above issues, this paper combines traditional ISC methods with deep models and proposes a novel few-shot ISC approach called Deep Class-specific Collaborative Representation (DCSCR) network to simultaneously learn the frame- and concept-level feature representations of each image set and the distance similarities between different sets. Specifically, DCSCR consists of a fully convolutional deep feature extractor module, a global feature learning module, and a class-specific collaborative representation-based metric learning module. The deep feature extractor and global feature learning modules are used to learn (local and global) frame-level feature representations, while the class-specific collaborative representation-based metric learning module is exploit to adaptively learn the concept-level feature representation of each image set and thus obtain the distance similarities between different sets by developing a new CSCR-based contrastive loss function. Extensive experiments on several well-known few-shot ISC datasets demonstrate the effectiveness of the proposed method compared with some state-of-the-art image set classification algorithms. 

**Abstract (ZH)**: 基于深度模型的Frame-和Concept-level协同表示的少样本图像集合分类（Deep Class-specific Collaborative Representation Network for Few-shot Image Set Classification） 

---
# FedUNet: A Lightweight Additive U-Net Module for Federated Learning with Heterogeneous Models 

**Title (ZH)**: FedUNet：一种用于异构模型联邦学习的轻量级加性U-Net模块 

**Authors**: Beomseok Seo, Kichang Lee, JaeYeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.12740)  

**Abstract**: Federated learning (FL) enables decentralized model training without sharing local data. However, most existing methods assume identical model architectures across clients, limiting their applicability in heterogeneous real-world environments. To address this, we propose FedUNet, a lightweight and architecture-agnostic FL framework that attaches a U-Net-inspired additive module to each client's backbone. By sharing only the compact bottleneck of the U-Net, FedUNet enables efficient knowledge transfer without structural alignment. The encoder-decoder design and skip connections in the U-Net help capture both low-level and high-level features, facilitating the extraction of clientinvariant representations. This enables cooperative learning between the backbone and the additive module with minimal communication cost. Experiment with VGG variants shows that FedUNet achieves 93.11% accuracy and 92.68% in compact form (i.e., a lightweight version of FedUNet) with only 0.89 MB low communication overhead. 

**Abstract (ZH)**: 联邦学习（FL）无需共享本地数据即可实现去中心化的模型训练。然而，现有大多数方法假设客户端具有相同的模型架构，这限制了其在异构现实环境中的应用。为此，我们提出FedUNet，这是一种轻量级且架构无关的联邦学习框架，为每个客户端的主干附加一个受U-Net启发的叠加模块。通过仅共享U-Net的紧凑瓶颈部分，FedUNet可以在不进行结构对齐的情况下实现高效的知识传输。U-Net的编码-解码设计和跳跃连接有助于捕获低级和高级特征，促进客户端不变表示的提取。这使得主干和叠加模块之间的合作学习可以在较低的通信成本下进行。实验结果显示，使用VGG变体时，FedUNet在紧凑形式下（即FedUNet的轻量级版本）达到93.11%的准确率，并且仅产生0.89 MB的低通信开销。 

---
# LinguaSafe: A Comprehensive Multilingual Safety Benchmark for Large Language Models 

**Title (ZH)**: LinguaSafe: 一种全面的多语言安全性基准测试，用于大型语言模型 

**Authors**: Zhiyuan Ning, Tianle Gu, Jiaxin Song, Shixin Hong, Lingyu Li, Huacan Liu, Jie Li, Yixu Wang, Meng Lingyu, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12733)  

**Abstract**: The widespread adoption and increasing prominence of large language models (LLMs) in global technologies necessitate a rigorous focus on ensuring their safety across a diverse range of linguistic and cultural contexts. The lack of a comprehensive evaluation and diverse data in existing multilingual safety evaluations for LLMs limits their effectiveness, hindering the development of robust multilingual safety alignment. To address this critical gap, we introduce LinguaSafe, a comprehensive multilingual safety benchmark crafted with meticulous attention to linguistic authenticity. The LinguaSafe dataset comprises 45k entries in 12 languages, ranging from Hungarian to Malay. Curated using a combination of translated, transcreated, and natively-sourced data, our dataset addresses the critical need for multilingual safety evaluations of LLMs, filling the void in the safety evaluation of LLMs across diverse under-represented languages from Hungarian to Malay. LinguaSafe presents a multidimensional and fine-grained evaluation framework, with direct and indirect safety assessments, including further evaluations for oversensitivity. The results of safety and helpfulness evaluations vary significantly across different domains and different languages, even in languages with similar resource levels. Our benchmark provides a comprehensive suite of metrics for in-depth safety evaluation, underscoring the critical importance of thoroughly assessing multilingual safety in LLMs to achieve more balanced safety alignment. Our dataset and code are released to the public to facilitate further research in the field of multilingual LLM safety. 

**Abstract (ZH)**: 全球技术中大型语言模型（LLMs）的广泛应用和日益突出的地位亟需在多元语言和文化背景下对其安全性进行严格保障。现有LLM多语言安全性评估的缺乏全面评估和多样性数据限制了其有效性，阻碍了稳健的多语言安全性对齐的发展。为填补这一关键缺口，我们引入了LinguaSafe，这是一个全面的多语言安全性基准，经过细致的语言真实性考量。LinguaSafe数据集包含45000个条目，涵盖了12种语言，从匈牙利语到马来语。通过翻译、跨文化创作和本地化数据的综合收集，我们的数据集满足了对LLM进行多语言安全性评估的关键需求，填补了从匈牙利语到马来语等欠代表语言多样性背景下的安全性评估空白。LinguaSafe提供了一个多维度和精细的评估框架，包括直接和间接的安全性评估，以及进一步的过度敏感性评估。不同领域和不同语言在安全性与帮助性评估中的结果存在显著差异，即使在资源水平相似的语言之间也是如此。该基准提供了一套全面的指标，用于深入的安全性评估，强调了对LLMs进行多语言安全性全面评估以实现更平衡的安全对齐的重要性。我们的数据集和代码已公开发布，以促进多语言LLM安全性领域的进一步研究。 

---
# MATPAC++: Enhanced Masked Latent Prediction for Self-Supervised Audio Representation Learning 

**Title (ZH)**: MATPAC++: 提升掩蔽潜变量预测的自监督音频表示学习 

**Authors**: Aurian Quelennec, Pierre Chouteau, Geoffroy Peeters, Slim Essid  

**Link**: [PDF](https://arxiv.org/pdf/2508.12709)  

**Abstract**: Masked latent prediction has emerged as a leading paradigm in self-supervised learning (SSL), especially for general audio and music representation learning. While recent methods have demonstrated strong performance, the role of the predictor module used at the output of such SSL systems remains mainly overlooked, despite being crucial for solving the pretext task at hand. In particular, this module should be able to deal with the ambiguity inherent in audio content, especially when it is composed of multiple sound sources. This work proposes a novel enhancement: integrating Multiple Choice Learning (MCL) to explicitly model prediction ambiguity and improve representation quality. We build on top of the recently proposed MATPAC system, improving its prediction and unsupervised classification pretext tasks with MCL. We extensively evaluate our method, MATPAC++, through both linear probing across multiple downstream tasks and fine-tuning on AudioSet, employing a unified protocol that enables rigorous and fair comparisons with state-of-the-art SSL approaches. Results show that our proposal achieves state-of-the-art when fine-tuned on AudioSet and overall state-of-the-art scores on downstream tasks. Additionally, we examine domain specialisation by training exclusively on music data, where our model achieves state-of-the-art performance with significantly improved efficiency. 

**Abstract (ZH)**: 掩码潜变量预测已成为监督学习（无监督（学习（SSL）中的一个成熟范式，特别是在通用音频和 音乐表示 学习方面。虽然近期 研究表明预测模块在这些SSL系统 系统中的作用非常重要，但在预设 任务中的预测输出的模糊性性仍然主要 被忽视。特别是在音频内容中，尤其是当其由多个声源组成时 时，这些预测需要处理内容的固有模糊性性。本文提出了一种新颖的增强方案，通过集成多项选择学习（MCL），明确地整合预测模糊性 总体性能。该 基于 近来的MATPAC系统 框架，改进预测和无监督分类预设 任务，我们通过MATPAC++进行广泛评估，涉及多个下游任务和onSet上 �.getBody调）进行严格的与现有最佳SSL方法进行对比。结果显示，Set上 上和下游任务的整体性能均达到最优。此外，我们仅在音乐数据上 进行训练时 ，同样达到最优性能，并且在效率上 方有显著提升。 

---
# Asymmetric Diffusion Recommendation Model 

**Title (ZH)**: 非对称扩散推荐模型 

**Authors**: Yongchun Zhu, Guanyu Jiang, Jingwu Chen, Feng Zhang, Xiao Yang, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12706)  

**Abstract**: Recently, motivated by the outstanding achievements of diffusion models, the diffusion process has been employed to strengthen representation learning in recommendation systems. Most diffusion-based recommendation models typically utilize standard Gaussian noise in symmetric forward and reverse processes in continuous data space. Nevertheless, the samples derived from recommendation systems inhabit a discrete data space, which is fundamentally different from the continuous one. Moreover, Gaussian noise has the potential to corrupt personalized information within latent representations. In this work, we propose a novel and effective method, named Asymmetric Diffusion Recommendation Model (AsymDiffRec), which learns forward and reverse processes in an asymmetric manner. We define a generalized forward process that simulates the missing features in real-world recommendation samples. The reverse process is then performed in an asymmetric latent feature space. To preserve personalized information within the latent representation, a task-oriented optimization strategy is introduced. In the serving stage, the raw sample with missing features is regarded as a noisy input to generate a denoising and robust representation for the final prediction. By equipping base models with AsymDiffRec, we conduct online A/B tests, achieving improvements of +0.131% and +0.166% in terms of users' active days and app usage duration respectively. Additionally, the extended offline experiments also demonstrate improvements. AsymDiffRec has been implemented in the Douyin Music App. 

**Abstract (ZH)**: 非对称扩散推荐模型：Asymmetric Diffusion Recommendation Model (AsymDiffRec) 

---
# A Unified Cortical Circuit Model with Divisive Normalization and Self-Excitation for Robust Representation and Memory Maintenance 

**Title (ZH)**: 一种结合 divisive 归一化和自激机制以实现 robust 表征和记忆维持的统一皮层 Circuit 模型 

**Authors**: Jie Su, Weiwei Wang, Zhaotian Gu, Dahui Wang, Tianyi Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.12702)  

**Abstract**: Robust information representation and its persistent maintenance are fundamental for higher cognitive functions. Existing models employ distinct neural mechanisms to separately address noise-resistant processing or information maintenance, yet a unified framework integrating both operations remains elusive -- a critical gap in understanding cortical computation. Here, we introduce a recurrent neural circuit that combines divisive normalization with self-excitation to achieve both robust encoding and stable retention of normalized inputs. Mathematical analysis shows that, for suitable parameter regimes, the system forms a continuous attractor with two key properties: (1) input-proportional stabilization during stimulus presentation; and (2) self-sustained memory states persisting after stimulus offset. We demonstrate the model's versatility in two canonical tasks: (a) noise-robust encoding in a random-dot kinematogram (RDK) paradigm; and (b) approximate Bayesian belief updating in a probabilistic Wisconsin Card Sorting Test (pWCST). This work establishes a unified mathematical framework that bridges noise suppression, working memory, and approximate Bayesian inference within a single cortical microcircuit, offering fresh insights into the brain's canonical computation and guiding the design of biologically plausible artificial neural architectures. 

**Abstract (ZH)**: 稳健的信息表示及其持久维持是高级认知功能的基础。现有的模型分别采用了不同的神经机制来处理噪声鲁棒性处理或信息维持，但将这两项功能统一在一个框架中仍然是一个关键缺口。在这里，我们介绍了一个将分量归一化与自兴奋相结合的递归神经电路，以实现对归一化输入的稳健编码和稳定保持。数学分析表明，在合适的参数范围内，该系统形成了一个连续吸引子，具有两个关键特性：(1) 在刺激呈现期间输入比例的稳定性；(2) 在刺激结束后自我维持的记忆状态。我们通过两个经典的任务展示了该模型的灵活性：(a) 在随机点运动图（RDK）范式中的噪声鲁棒编码；(b) 在概率威斯康星卡片分类测试（pWCST）中的近似贝叶斯信念更新。这项工作建立了一个统一的数学框架，将噪声抑制、工作记忆和近似贝叶斯推理统一在一个皮层微电路中，为大脑的经典计算提供了新的见解，并指导了生物合理的神经网络架构的设计。 

---
# Multi-Level Knowledge Distillation and Dynamic Self-Supervised Learning for Continual Learning 

**Title (ZH)**: 多级知识精炼与动态自监督学习在连续学习中的应用 

**Authors**: Taeheon Kim, San Kim, Minhyuk Seo, Dongjae Jeon, Wonje Jeong, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.12692)  

**Abstract**: Class-incremental with repetition (CIR), where previously trained classes repeatedly introduced in future tasks, is a more realistic scenario than the traditional class incremental setup, which assumes that each task contains unseen classes. CIR assumes that we can easily access abundant unlabeled data from external sources, such as the Internet. Therefore, we propose two components that efficiently use the unlabeled data to ensure the high stability and the plasticity of models trained in CIR setup. First, we introduce multi-level knowledge distillation (MLKD) that distills knowledge from multiple previous models across multiple perspectives, including features and logits, so the model can maintain much various previous knowledge. Moreover, we implement dynamic self-supervised loss (SSL) to utilize the unlabeled data that accelerates the learning of new classes, while dynamic weighting of SSL keeps the focus of training to the primary task. Both of our proposed components significantly improve the performance in CIR setup, achieving 2nd place in the CVPR 5th CLVISION Challenge. 

**Abstract (ZH)**: 基于重复的类别增量学习（CIR）：一种更现实的场景，其中以前训练的类别在未来的任务中重复出现，比传统的类别增量设置更为现实。CIR 假设可以从外部来源，如互联网，轻松获取丰富的未标注数据。因此，我们提出了两个高效利用未标注数据的组件，以确保在 CIR 设置下训练模型的高稳定性和可塑性。首先，我们引入多层次知识蒸馏（MLKD），从多个先前模型的多个视角（包括特征和logits）提取知识，使模型能够保留大量的先前知识。此外，我们实现了动态自监督损失（SSL）来利用未标注数据加速新类别的学习，而动态调整SSL权重则保持训练的重点在主要任务上。我们提出的两个组件显著提高了CIR设置下的性能，在CVPR第5届CLVISION挑战赛中获得第2名。 

---
# TTA-DAME: Test-Time Adaptation with Domain Augmentation and Model Ensemble for Dynamic Driving Conditions 

**Title (ZH)**: TTA-DAME：基于域增强和模型集成的动态驾驶条件下的测试时自适应方法 

**Authors**: Dongjae Jeon, Taeheon Kim, Seongwon Cho, Minhyuk Seo, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.12690)  

**Abstract**: Test-time Adaptation (TTA) poses a challenge, requiring models to dynamically adapt and perform optimally on shifting target domains. This task is particularly emphasized in real-world driving scenes, where weather domain shifts occur frequently. To address such dynamic changes, our proposed method, TTA-DAME, leverages source domain data augmentation into target domains. Additionally, we introduce a domain discriminator and a specialized domain detector to mitigate drastic domain shifts, especially from daytime to nighttime conditions. To further improve adaptability, we train multiple detectors and consolidate their predictions through Non-Maximum Suppression (NMS). Our empirical validation demonstrates the effectiveness of our method, showing significant performance enhancements on the SHIFT Benchmark. 

**Abstract (ZH)**: Test-time Adaptation (TTA)在移域目标域动态适应中提出了一项挑战，要求模型能够动态调整以在变化的目标域中表现最优。这一任务在现实中驾驶场景中尤其突出，因为天气条件经常发生变化。为应对这种动态变化，我们提出的方法TTA-DAME利用源域数据增强技术将数据迁移到目标域。此外，我们引入了领域判别器和专门的领域检测器，以减轻尤其是由白天到夜晚等急剧的域变化。为进一步提升适应性，我们训练了多个检测器并通过非极大值抑制（NMS）合并它们的预测。我们的实证验证表明了该方法的有效性，显著提升了在SHIFT基准上的性能。 

---
# ToolACE-MT: Non-Autoregressive Generation for Agentic Multi-Turn Interaction 

**Title (ZH)**: ToolACE-MT: 非自回归生成用于能动多轮交互 

**Authors**: Xingshan Zeng, Weiwen Liu, Lingzhi Wang, Liangyou Li, Fei Mi, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12685)  

**Abstract**: Agentic task-solving with Large Language Models (LLMs) requires multi-turn, multi-step interactions, often involving complex function calls and dynamic user-agent exchanges. Existing simulation-based data generation methods for such scenarios rely heavily on costly autoregressive interactions between multiple LLM agents, thereby limiting real-world performance of agentic tasks. In this paper, we propose a novel Non-Autoregressive Iterative Generation framework, called ToolACE-MT, for constructing high-quality multi-turn agentic dialogues. ToolACE-MT generates full conversational trajectories through three stages: coarse-grained initialization, iterative refinement, and offline verification. The initialization phase builds a structurally complete yet semantically coarse dialogue skeleton; the iterative refinement phase introduces realistic complexities and continued refinement via mask-and-fill operations; and the offline verification phase ensures correctness and coherence via rule- and model-based checks. Experiments demonstrate that ToolACE-MT enables efficient, effective and generalizable agentic data generation, offering a new paradigm for high-quality data construction in tool-augmented LLM scenarios. 

**Abstract (ZH)**: 基于大型语言模型的代理任务解决需要多轮多步的交互，通常涉及复杂的功能调用和动态的用户-代理交换。现有的此类场景下的模拟数据生成方法高度依赖于多个大型语言模型代理的昂贵的自回归交互，从而限制了代理任务的现实世界性能。本文提出了一种新颖的非自回归迭代生成框架ToolACE-MT，用于构建高质量的多轮代理对话。ToolACE-MT 通过三个阶段生成完整的对话轨迹：粗粒度初始化、迭代精炼和离线验证。初始化阶段构建一个结构完整但语义粗糙的对话骨架；迭代精炼阶段通过掩码和填充操作引入现实的复杂性和持续精炼；离线验证阶段通过基于规则和模型的检查确保正确性和连贯性。实验表明，ToolACE-MT 使代理数据生成更加高效、有效和泛化，为工具增强的大语言模型场景中的高质量数据构建提供了新的范式。 

---
# A Taxonomy of Hierarchical Multi-Agent Systems: Design Patterns, Coordination Mechanisms, and Industrial Applications 

**Title (ZH)**: 层次化多智能体系统的分类：设计模式、协调机制及工业应用 

**Authors**: David J. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2508.12683)  

**Abstract**: Hierarchical multi-agent systems (HMAS) organize collections of agents into layered structures that help manage complexity and scale. These hierarchies can simplify coordination, but they also can introduce trade-offs that are not always obvious. This paper proposes a multi-dimensional taxonomy for HMAS along five axes: control hierarchy, information flow, role and task delegation, temporal layering, and communication structure. The intent is not to prescribe a single "best" design but to provide a lens for comparing different approaches.
Rather than treating these dimensions in isolation, the taxonomy is connected to concrete coordination mechanisms - from the long-standing contract-net protocol for task allocation to more recent work in hierarchical reinforcement learning. Industrial contexts illustrate the framework, including power grids and oilfield operations, where agents at production, maintenance, and supply levels coordinate to diagnose well issues or balance energy demand. These cases suggest that hierarchical structures may achieve global efficiency while preserving local autonomy, though the balance is delicate.
The paper closes by identifying open challenges: making hierarchical decisions explainable to human operators, scaling to very large agent populations, and assessing whether learning-based agents such as large language models can be safely integrated into layered frameworks. This paper presents what appears to be the first taxonomy that unifies structural, temporal, and communication dimensions of hierarchical MAS into a single design framework, bridging classical coordination mechanisms with modern reinforcement learning and large language model agents. 

**Abstract (ZH)**: 多层次多智能体系统（HMAS）的多维度分类框架：从控制层级、信息流、角色和任务委派、时间分层及通信结构五个维度探究。 

---
# Deploying Models to Non-participating Clients in Federated Learning without Fine-tuning: A Hypernetwork-based Approach 

**Title (ZH)**: 无需微调在非参与客户端部署模型在联邦学习中的方法：基于超网络的 Approach 

**Authors**: Yuhao Zhou, Jindi Lv, Yuxin Tian, Dan Si, Qing Ye, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2508.12673)  

**Abstract**: Federated Learning (FL) has emerged as a promising paradigm for privacy-preserving collaborative learning, yet data heterogeneity remains a critical challenge. While existing methods achieve progress in addressing data heterogeneity for participating clients, they fail to generalize to non-participating clients with in-domain distribution shifts and resource constraints. To mitigate this issue, we present HyperFedZero, a novel method that dynamically generates specialized models via a hypernetwork conditioned on distribution-aware embeddings. Our approach explicitly incorporates distribution-aware inductive biases into the model's forward pass, extracting robust distribution embeddings using a NoisyEmbed-enhanced extractor with a Balancing Penalty, effectively preventing feature collapse. The hypernetwork then leverages these embeddings to generate specialized models chunk-by-chunk for non-participating clients, ensuring adaptability to their unique data distributions. Extensive experiments on multiple datasets and models demonstrate HyperFedZero's remarkable performance, surpassing competing methods consistently with minimal computational, storage, and communication overhead. Moreover, ablation studies and visualizations further validate the necessity of each component, confirming meaningful adaptations and validating the effectiveness of HyperFedZero. 

**Abstract (ZH)**: 联邦学习（FL）已经 emerged 作为隐私保护协作学习的一个有前途的范式，但数据异质性仍然是一个 critical 挑战。虽然现有的方法在解决参与客户端的数据异质性方面取得了进展，但它们无法将这些方法推广到具有领域内分布转移和资源约束的未参与客户端。为了缓解这一问题，我们提出了 HyperFedZero，这是一种新型方法，可以通过一个基于分布感知嵌入的超网络动态生成专门化的模型。我们的方法在模型的前向传播中显式地引入了分布感知的归纳偏置，使用一个增强的 NoisyEmbed 提取器和平衡罚则有效地提取鲁棒的分布嵌入，从而预防特征坍塌。超网络利用这些嵌入逐块为未参与客户端生成专门化的模型，确保适应其独特的数据分布。在多个数据集和模型上的 extensive 实验表明，HyperFedZero 在 minimal 计算、存储和通信开销下，表现出色，且始终优于竞争方法。此外，消融研究和可视化进一步验证了每个组件的必要性，确认了有意义的适应，并验证了 HyperFedZero 的有效性。 

---
# Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering 

**Title (ZH)**: 基于损失的客户端聚类以抵抗对抗攻击的鲁棒联邦学习 

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, Konstantinos Tserpes  

**Link**: [PDF](https://arxiv.org/pdf/2508.12672)  

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side dataset. This may correspond to, e.g., cases where the server possesses trusted data prior to federation, or to the presence of a trusted client that temporarily assumes the server role. Our approach requires only two honest participants, i.e., the server and one client, to function effectively, without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum under various attack strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the Flower framework. 

**Abstract (ZH)**: 联邦学习（（FL）实现了多个客户端协作的隐私数据安全训练。我们考虑FL客户端遭受拜占（（（拜占）攻击的场景， 而FL服务器是可 诚实的（ 并拥有一个可信的数据集。这 可对应于.g., 在联邦之前存在一个可信的客户端 e 或者一个可信客户端暂时承担这一角色。我们的方法 方法仅需两个诚实的 的客户端 e � 即服务器端和 和 一个客户端 e �ResourceId 无需预先了解恶意客户端的身份。从理论上 �ぃ理论分析 分 分分析 � européenet 有限恶意攻击 Strikes �性价 的上 �边界 e 历家性问 优化最最优性 g 怈 � �边界 e x罅 e �的� �边 海 e 奵 e e e e理论缺口。实验结果结果显示 e 戔 � e 戛 e etermine e 戄 � e � e e e e e e e e e e e e e e e e e e e e e e e e � SMP ew e e 交易 e  e � e e e  e e e  e e e e e e e e e e e e  e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e"}} 

---
# Breaking Language Barriers: Equitable Performance in Multilingual Language Models 

**Title (ZH)**: 破除语言障碍：多语言语言模型的公平性能 

**Authors**: Tanay Nagar, Grigorii Khvatskii, Anna Sokol, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2508.12662)  

**Abstract**: Cutting-edge LLMs have emerged as powerful tools for multilingual communication and understanding. However, LLMs perform worse in Common Sense Reasoning (CSR) tasks when prompted in low-resource languages (LRLs) like Hindi or Swahili compared to high-resource languages (HRLs) like English. Equalizing this inconsistent access to quality LLM outputs is crucial to ensure fairness for speakers of LRLs and across diverse linguistic communities. In this paper, we propose an approach to bridge this gap in LLM performance. Our approach involves fine-tuning an LLM on synthetic code-switched text generated using controlled language-mixing methods. We empirically demonstrate that fine-tuning LLMs on synthetic code-switched datasets leads to substantial improvements in LRL model performance while preserving or enhancing performance in HRLs. Additionally, we present a new dataset of synthetic code-switched text derived from the CommonSenseQA dataset, featuring three distinct language ratio configurations. 

**Abstract (ZH)**: 基于 synthetic code-switched 数据集的 LLM 细调以提升低资源语言的常识推理能力 

---
# Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery 

**Title (ZH)**: 基于评分的神经算子增强顺序依赖因果发现 

**Authors**: Jiyeon Kang, Songseong Kim, Chanhui Lee, Doyeong Hwang, Joanie Hayoun Chung, Yunkyung Ko, Sumin Lee, Sungwoong Kim, Sungbin Lim  

**Link**: [PDF](https://arxiv.org/pdf/2508.12650)  

**Abstract**: Ordering-based approaches to causal discovery identify topological orders of causal graphs, providing scalable alternatives to combinatorial search methods. Under the Additive Noise Model (ANM) assumption, recent causal ordering methods based on score matching require an accurate estimation of the Hessian diagonal of the log-densities. However, previous approaches mainly use Stein gradient estimators, which are computationally expensive and memory-intensive. Although DiffAN addresses these limitations by substituting kernel-based estimates with diffusion models, it remains numerically unstable due to the second-order derivatives of score models. To alleviate these problems, we propose Score-informed Neural Operator (SciNO), a probabilistic generative model in smooth function spaces designed to stably approximate the Hessian diagonal and to preserve structural information during the score modeling. Empirical results show that SciNO reduces order divergence by 42.7% on synthetic graphs and by 31.5% on real-world datasets on average compared to DiffAN, while maintaining memory efficiency and scalability. Furthermore, we propose a probabilistic control algorithm for causal reasoning with autoregressive models that integrates SciNO's probability estimates with autoregressive model priors, enabling reliable data-driven causal ordering informed by semantic information. Consequently, the proposed method enhances causal reasoning abilities of LLMs without additional fine-tuning or prompt engineering. 

**Abstract (ZH)**: 基于排序的方法在因果发现中的应用识别因果图的拓扑排序，提供了一种可扩展的替代组合搜索方法。在加性噪声模型（ANM）假设下，最近基于分数匹配的因果排序方法需要准确估计对数密度的海森矩阵对角线。然而，之前的 方法主要使用Stein梯度估计器，这在计算和存储方面都比较昂贵。尽管DiffAN通过用扩散模型替代核估计解决了这些问题，但由于分数模型的二次导数，它仍然在数值稳定性方面存在不足。为了缓解这些问题，我们提出了Score-informed Neural Operator (SciNO)，这是一种在光滑函数空间中设计的概率生成模型，旨在稳定地近似海森矩阵的对角线，并在分数建模过程中保结构信息。实验结果表明，与DiffAN相比，SciNO在合成图上将排序发散减少了42.7%，在实际数据集上平均减少了31.5%，同时保持了内存效率和可扩展性。此外，我们提出了一种概率控制算法，用于自回归模型中的因果推理，该算法将SciNO的概率估计与自回归模型先验整合，从而基于语义信息实现可靠的基于数据驱动的因果排序。因此，所提出的方法增强了解释能力LLMs，而无需额外的微调或提示工程。 

---
# SpotVLM: Cloud-edge Collaborative Real-time VLM based on Context Transfer 

**Title (ZH)**: SpotVLM：基于上下文转移的云端边缘协同实时VLM 

**Authors**: Chen Qian, Xinran Yu, Zewen Huang, Danyang Li, Qiang Ma, Fan Dang, Xuan Ding, Guangyong Shang, Zheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12638)  

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in real-time applications such as autonomous driving and human-computer interaction, which demand fast and reliable responses based on accurate perception. To meet these requirements, existing systems commonly employ cloud-edge collaborative architectures, such as partitioned Large Vision-Language Models (LVLMs) or task offloading strategies between Large and Small Vision-Language Models (SVLMs). However, these methods fail to accommodate cloud latency fluctuations and overlook the full potential of delayed but accurate LVLM responses. In this work, we propose a novel cloud-edge collaborative paradigm for VLMs, termed Context Transfer, which treats the delayed outputs of LVLMs as historical context to provide real-time guidance for SVLMs inference. Based on this paradigm, we design SpotVLM, which incorporates both context replacement and visual focus modules to refine historical textual input and enhance visual grounding consistency. Extensive experiments on three real-time vision tasks across four datasets demonstrate the effectiveness of the proposed framework. The new paradigm lays the groundwork for more effective and latency-aware collaboration strategies in future VLM systems. 

**Abstract (ZH)**: 基于上下文转移的视觉-语言模型云边协作范式 

---
# How can we trust opaque systems? Criteria for robust explanations in XAI 

**Title (ZH)**: 如何信任不透明系统？解释性人工智能中稳健解释的标准 

**Authors**: Florian J. Boge, Annika Schuster  

**Link**: [PDF](https://arxiv.org/pdf/2508.12623)  

**Abstract**: Deep learning (DL) algorithms are becoming ubiquitous in everyday life and in scientific research. However, the price we pay for their impressively accurate predictions is significant: their inner workings are notoriously opaque - it is unknown to laypeople and researchers alike what features of the data a DL system focuses on and how it ultimately succeeds in predicting correct outputs. A necessary criterion for trustworthy explanations is that they should reflect the relevant processes the algorithms' predictions are based on. The field of eXplainable Artificial Intelligence (XAI) presents promising methods to create such explanations. But recent reviews about their performance offer reasons for skepticism. As we will argue, a good criterion for trustworthiness is explanatory robustness: different XAI methods produce the same explanations in comparable contexts. However, in some instances, all methods may give the same, but still wrong, explanation. We therefore argue that in addition to explanatory robustness (ER), a prior requirement of explanation method robustness (EMR) has to be fulfilled by every XAI method. Conversely, the robustness of an individual method is in itself insufficient for trustworthiness. In what follows, we develop and formalize criteria for ER as well as EMR, providing a framework for explaining and establishing trust in DL algorithms. We also highlight interesting application cases and outline directions for future work. 

**Abstract (ZH)**: 深度学习算法在日常生活和科学研究中无处不在。然而，它们令人印象深刻准确的预测是以显著的代价获得的：其内部工作机制历来具有很强的不透明性——无论是普通人还是研究人员都不知道，一个深度学习系统关注数据的哪些特征以及它是如何最终成功预测正确输出的。对于可信的解释而言，一个必要条件是它们应该反映算法预测所依据的相关过程。可解释人工智能（XAI）领域提出了一些有前景的方法来创建这样的解释。然而，最近关于它们性能的回顾提供了怀疑的理由。正如我们将论证的那样，可信度的良好标准是解释的稳健性：不同的XAI方法在类似的情境下应产生相同解释。然而，在某些情况下，所有方法可能给出相同的，但仍然是错误的解释。因此，我们论证对于每一个XAI方法而言，除了解释的稳健性（ER）之外，还需满足解释方法的稳健性（EMR）的先决条件。单个方法的稳健性本身不足以保证可信度。接下来，我们将开发并公式化ER和EMR的标准，提供一个框架来解释和建立对深度学习算法的信任。我们还将突出有趣的应用案例，并概述未来工作的方向。 

---
# A Generalized Genetic Random Field Method for the Genetic Association Analysis of Sequencing Data 

**Title (ZH)**: 一种用于序列数据遗传关联分析的广义遗传随机场方法 

**Authors**: Ming Li, Zihuai He, Min Zhang, Xiaowei Zhan, Changshuai Wei, Robert C Elston, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12617)  

**Abstract**: With the advance of high-throughput sequencing technologies, it has become feasible to investigate the influence of the entire spectrum of sequencing variations on complex human diseases. Although association studies utilizing the new sequencing technologies hold great promise to unravel novel genetic variants, especially rare genetic variants that contribute to human diseases, the statistical analysis of high-dimensional sequencing data remains a challenge. Advanced analytical methods are in great need to facilitate high-dimensional sequencing data analyses. In this article, we propose a generalized genetic random field (GGRF) method for association analyses of sequencing data. Like other similarity-based methods (e.g., SIMreg and SKAT), the new method has the advantages of avoiding the need to specify thresholds for rare variants and allowing for testing multiple variants acting in different directions and magnitude of effects. The method is built on the generalized estimating equation framework and thus accommodates a variety of disease phenotypes (e.g., quantitative and binary phenotypes). Moreover, it has a nice asymptotic property, and can be applied to small-scale sequencing data without need for small-sample adjustment. Through simulations, we demonstrate that the proposed GGRF attains an improved or comparable power over a commonly used method, SKAT, under various disease scenarios, especially when rare variants play a significant role in disease etiology. We further illustrate GGRF with an application to a real dataset from the Dallas Heart Study. By using GGRF, we were able to detect the association of two candidate genes, ANGPTL3 and ANGPTL4, with serum triglyceride. 

**Abstract (ZH)**: 高通量测序技术的进步使得研究整个变异谱对复杂人类疾病的影响成为可能。尽管利用新型测序技术的关联研究有望揭示新的遗传变异，尤其是那些对人类疾病有贡献的稀有遗传变异，但高维测序数据的统计分析仍然是一项挑战。急需先进的分析方法来促进高维测序数据分析。本文提出了一种广义遗传随机场（GGRF）方法，用于测序数据的关联分析。与SIMreg和SKAT等基于相似性的方法相比，新方法避免了指定稀有变异阈值的需要，并且能够测试多个方向和不同效应大小的变异。该方法基于广义估计方程框架，因此能够适用于各种疾病表型（例如，定量表型和二元表型）。此外，它具有良好的渐近性质，在不需要小样本调整的情况下可以应用于小型测序数据集。通过模拟，我们证明了提出的GGRF在各种疾病情景下，尤其是在稀有变异在疾病病因学中起重要作用时，其检测功效优于常用的SKAT方法。我们进一步使用真正的达拉斯心脏研究数据集对GGRF进行了应用，通过GGRF，我们发现ANGPTL3和ANGPTL4两个候选基因与血清甘油三酯水平有关。 

---
# OpenMoCap: Rethinking Optical Motion Capture under Real-world Occlusion 

**Title (ZH)**: OpenMoCap: 重新思考.real-world Occlusion下的光学运动捕捉 

**Authors**: Chen Qian, Danyang Li, Xinran Yu, Zheng Yang, Qiang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.12610)  

**Abstract**: Optical motion capture is a foundational technology driving advancements in cutting-edge fields such as virtual reality and film production. However, system performance suffers severely under large-scale marker occlusions common in real-world applications. An in-depth analysis identifies two primary limitations of current models: (i) the lack of training datasets accurately reflecting realistic marker occlusion patterns, and (ii) the absence of training strategies designed to capture long-range dependencies among markers. To tackle these challenges, we introduce the CMU-Occlu dataset, which incorporates ray tracing techniques to realistically simulate practical marker occlusion patterns. Furthermore, we propose OpenMoCap, a novel motion-solving model designed specifically for robust motion capture in environments with significant occlusions. Leveraging a marker-joint chain inference mechanism, OpenMoCap enables simultaneous optimization and construction of deep constraints between markers and joints. Extensive comparative experiments demonstrate that OpenMoCap consistently outperforms competing methods across diverse scenarios, while the CMU-Occlu dataset opens the door for future studies in robust motion solving. The proposed OpenMoCap is integrated into the MoSen MoCap system for practical deployment. The code is released at: this https URL. 

**Abstract (ZH)**: 光学运动捕捉是推动虚拟现实和电影制作等前沿领域发展的基础技术，但在实际应用中大规模标记物遮挡会严重影响系统性能。深入分析识别了当前模型的两大主要局限性：（i）缺乏能够准确反映真实标记物遮挡模式的训练数据集，（ii）缺少用于捕捉标记物之间长程依赖性的训练策略。为应对这些挑战，我们引入了CMU-Occlu数据集，利用射线 tracing 技术真实模拟实际的标记物遮挡模式。此外，我们提出了OpenMoCap，这是一种专为显著遮挡环境中稳健运动捕捉设计的新模型。利用标记物-关节链推理机制，OpenMoCap能够同时优化并构建标记和关节间的深层约束。广泛比较实验表明，OpenMoCap在多种场景中表现 superior 于现有方法，而CMU-Occlu数据集则为未来稳健运动求解研究打开了大门。提出的OpenMoCap已被集成到MoSen MoCap系统中用于实际部署。代码在此处发布：this https URL。 

---
# SSPO: Self-traced Step-wise Preference Optimization for Process Supervision and Reasoning Compression 

**Title (ZH)**: SSPO: 自追踪分步偏好优化過程監控與因果壓縮 

**Authors**: Yuyang Xu, Yi Cheng, Haochao Ying, Zhuoyun Du, Renjun Hu, Xing Shi, Wei Lin, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12604)  

**Abstract**: Test-time scaling has proven effective in further enhancing the performance of pretrained Large Language Models (LLMs). However, mainstream post-training methods (i.e., reinforcement learning (RL) with chain-of-thought (CoT) reasoning) often incur substantial computational overhead due to auxiliary models and overthinking. In this paper, we empirically reveal that the incorrect answers partially stem from verbose reasoning processes lacking correct self-fix, where errors accumulate across multiple reasoning steps. To this end, we propose Self-traced Step-wise Preference Optimization (SSPO), a pluggable RL process supervision framework that enables fine-grained optimization of each reasoning step. Specifically, SSPO requires neither auxiliary models nor stepwise manual annotations. Instead, it leverages step-wise preference signals generated by the model itself to guide the optimization process for reasoning compression. Experiments demonstrate that the generated reasoning sequences from SSPO are both accurate and succinct, effectively mitigating overthinking behaviors without compromising model performance across diverse domains and languages. 

**Abstract (ZH)**: Test-time Scaling has Proven Effective in Further Enhancing the Performance of Pretrained Large Language Models (LLMs). However, Mainstream Post-training Methods (i.e., Reinforcement Learning with Chain-of-Thought Reasoning) Often Incur Substantial Computational Overhead Due to Auxiliary Models and Overthinking. In This Paper, We Empirically Reveal That Incorrect Answers Partially Stem from Verbose Reasoning Processes Lacking Correct Self-fix, Where Errors Accumulate Across Multiple Reasoning Steps. To This End, We Propose Self-traced Step-wise Preference Optimization (SSPO), a Pluggable RL Process Supervision Framework That Enables Fine-grained Optimization of Each Reasoning Step. Specifically, SSPO Requires Neither Auxiliary Models Nor Stepwise Manual Annotations. Instead, It Leverages Step-wise Preference Signals Generated by the Model Itself to Guide the Optimization Process for Reasoning Compression. Experiments Demonstrate That the Generated Reasoning Sequences from SSPO Are Both Accurate and Succinct, Effectively Mitigating Overthinking Behaviors Without Compromising Model Performance Across Diverse Domains and Languages. 

---
# Beyond Modality Limitations: A Unified MLLM Approach to Automated Speaking Assessment with Effective Curriculum Learning 

**Title (ZH)**: 超越模态限制：一种基于有效 Curriculum Learning 的统一MLLM自动口语评估方法 

**Authors**: Yu-Hsuan Fang, Tien-Hong Lo, Yao-Ting Sung, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.12591)  

**Abstract**: Traditional Automated Speaking Assessment (ASA) systems exhibit inherent modality limitations: text-based approaches lack acoustic information while audio-based methods miss semantic context. Multimodal Large Language Models (MLLM) offer unprecedented opportunities for comprehensive ASA by simultaneously processing audio and text within unified frameworks. This paper presents a very first systematic study of MLLM for comprehensive ASA, demonstrating the superior performance of MLLM across the aspects of content and language use . However, assessment on the delivery aspect reveals unique challenges, which is deemed to require specialized training strategies. We thus propose Speech-First Multimodal Training (SFMT), leveraging a curriculum learning principle to establish more robust modeling foundations of speech before cross-modal synergetic fusion. A series of experiments on a benchmark dataset show MLLM-based systems can elevate the holistic assessment performance from a PCC value of 0.783 to 0.846. In particular, SFMT excels in the evaluation of the delivery aspect, achieving an absolute accuracy improvement of 4% over conventional training approaches, which also paves a new avenue for ASA. 

**Abstract (ZH)**: 传统的自动口语评估系统固有地受到模态限制：基于文本的方法缺乏声学信息，而基于音频的方法则缺乏语义上下文。多模态大语言模型（MLLM）提供了前所未有的机会，通过在统一框架内同时处理音频和文本，实现全面的口语评估。本文首次系统研究了MLLM在全面口语评估中的应用，展示了MLLM在内容和语言使用方面的优越性能。然而，针对表达方面的评估揭示了独特的挑战，这被认为需要专门的训练策略。因此，我们提出了语音优先多模态训练（SFMT），利用课程学习原则，在跨模态协同融合之前，建立更 robust 的语音建模基础。一系列针对基准数据集开展的实验显示，基于MLLM的系统可以将整体评估性能从PCC值0.783提升至0.846。特别是，SFMT在评价表达方面表现优异，相较于传统训练方法，绝对准确率提升4%，也为口语评估开辟了新途径。 

---
# Energy-Efficient Wireless LLM Inference via Uncertainty and Importance-Aware Speculative Decoding 

**Title (ZH)**: 基于不确定性与重要性感知投机解码的能源高效无线LLM推理 

**Authors**: Jihoon Park, Seungeun Oh, Seong-Lyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.12590)  

**Abstract**: To address the growing demand for on-device LLM inference in resource-constrained environments, hybrid language models (HLM) have emerged, combining lightweight local models with powerful cloud-based LLMs. Recent studies on HLM have primarily focused on improving accuracy and latency, while often overlooking communication and energy efficiency. We propose a token-level filtering mechanism for an energy-efficient importance- and uncertainty-aware HLM inference that leverages both epistemic uncertainty and attention-based importance. Our method opportunistically uploads only informative tokens, reducing LLM usage and communication costs. Experiments with TinyLlama-1.1B and LLaMA-2-7B demonstrate that our method achieves up to 87.5% BERT Score and token throughput of 0.37 tokens/sec while saving the energy consumption by 40.7% compared to standard HLM. Furthermore, compared to our previous U-HLM baseline, our method improves BERTScore from 85.8% to 87.0%, energy savings from 31.6% to 43.6%, and throughput from 0.36 to 0.40. This approach enables an energy-efficient and accurate deployment of LLMs in bandwidth-constrained edge environments. 

**Abstract (ZH)**: 面向资源受限环境的设备端大语言模型推理：一种基于令牌级过滤的能量高效且具有重要性和不确定性意识的混合语言模型推理方法 

---
# Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg 

**Title (ZH)**: 扩大网络规模减轻数据异质性对FedAvg的影响 

**Authors**: Like Jian, Dong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12576)  

**Abstract**: Federated learning (FL) enables decentralized clients to train a model collaboratively without sharing local data. A key distinction between FL and centralized learning is that clients' data are non-independent and identically distributed, which poses significant challenges in training a global model that generalizes well across heterogeneous local data distributions. In this paper, we analyze the convergence of overparameterized FedAvg with gradient descent (GD). We prove that the impact of data heterogeneity diminishes as the width of neural networks increases, ultimately vanishing when the width approaches infinity. In the infinite-width regime, we further prove that both the global and local models in FedAvg behave as linear models, and that FedAvg achieves the same generalization performance as centralized learning with the same number of GD iterations. Extensive experiments validate our theoretical findings across various network architectures, loss functions, and optimization methods. 

**Abstract (ZH)**: 联邦学习(Federated Learning)使分散的客户端能够在不共享本地数据的情况下协作训练模型。与集中式学习的关键区别在于客户端的数据是非独立且不相同分布的，这给跨异质本地数据分布训练出泛化良好的全局模型带来了重大挑战。本文分析了过参数化FedAvg与梯度下降的收敛性。我们证明，随着神经网络宽度的增加，数据异质性的影响逐渐减弱，当宽度接近无穷大时完全消失。在无限宽度的情况下，我们进一步证明，FedAvg中的全局和局部模型均表现为线性模型，并且FedAvg在相同数量的梯度下降迭代次数下实现了与集中式学习相同的泛化性能。广泛的实验在各种网络架构、损失函数和优化方法下验证了我们的理论发现。 

---
# Deep Learning Model for Amyloidogenicity Prediction using a Pre-trained Protein LLM 

**Title (ZH)**: 使用预训练蛋白质LLM进行淀粉样形成性预测的深度学习模型 

**Authors**: Zohra Yagoub, Hafida Bouziane  

**Link**: [PDF](https://arxiv.org/pdf/2508.12575)  

**Abstract**: The prediction of amyloidogenicity in peptides and proteins remains a focal point of ongoing bioinformatics. The crucial step in this field is to apply advanced computational methodologies. Many recent approaches to predicting amyloidogenicity within proteins are highly based on evolutionary motifs and the individual properties of amino acids. It is becoming increasingly evident that the sequence information-based features show high predictive performance. Consequently, our study evaluated the contextual features of protein sequences obtained from a pretrained protein large language model leveraging bidirectional LSTM and GRU to predict amyloidogenic regions in peptide and protein sequences. Our method achieved an accuracy of 84.5% on 10-fold cross-validation and an accuracy of 83% in the test dataset. Our results demonstrate competitive performance, highlighting the potential of LLMs in enhancing the accuracy of amyloid prediction. 

**Abstract (ZH)**: 肽和蛋白质的淀粉样聚集倾向预测仍然是生物信息学研究的焦点。我们通过利用预训练蛋白质语言模型结合双向LSTM和GRU来评价上下文特征，以预测肽和 蛋白序列中的淀粉样聚集区域，我们的方法在5 五折交叉验证中达到了84.5% 的准确率，在在测试数据集中达到了83% 的准确率。我们的结果表明，基于LLM的方法具有竞争力，突显了LL
TM在提高淀粉样聚集预测准确率方面的潜力。 

---
# OS-R1: Agentic Operating System Kernel Tuning with Reinforcement Learning 

**Title (ZH)**: 基于强化学习的代理操作系统内核调优 

**Authors**: Hongyu Lin, Yuchen Li, Haoran Luo, Kaichun Yao, Libo Zhang, Mingjie Xing, Yanjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12551)  

**Abstract**: Linux kernel tuning is essential for optimizing operating system (OS) performance. However, existing methods often face challenges in terms of efficiency, scalability, and generalization. This paper introduces OS-R1, an agentic Linux kernel tuning framework powered by rule-based reinforcement learning (RL). By abstracting the kernel configuration space as an RL environment, OS-R1 facilitates efficient exploration by large language models (LLMs) and ensures accurate configuration modifications. Additionally, custom reward functions are designed to enhance reasoning standardization, configuration modification accuracy, and system performance awareness of the LLMs. Furthermore, we propose a two-phase training process that accelerates convergence and minimizes retraining across diverse tuning scenarios. Experimental results show that OS-R1 significantly outperforms existing baseline methods, achieving up to 5.6% performance improvement over heuristic tuning and maintaining high data efficiency. Notably, OS-R1 is adaptable across various real-world applications, demonstrating its potential for practical deployment in diverse environments. Our dataset and code are publicly available at this https URL. 

**Abstract (ZH)**: 基于基于规则的强化学习的OS-R1：一种代理Linux内核调优框架 

---
# Systematic Analysis of MCP Security 

**Title (ZH)**: MCP安全系统的系统性分析 

**Authors**: Yongjian Guo, Puzhuo Liu, Wanlun Ma, Zehang Deng, Xiaogang Zhu, Peng Di, Xi Xiao, Sheng Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.12538)  

**Abstract**: The Model Context Protocol (MCP) has emerged as a universal standard that enables AI agents to seamlessly connect with external tools, significantly enhancing their functionality. However, while MCP brings notable benefits, it also introduces significant vulnerabilities, such as Tool Poisoning Attacks (TPA), where hidden malicious instructions exploit the sycophancy of large language models (LLMs) to manipulate agent behavior. Despite these risks, current academic research on MCP security remains limited, with most studies focusing on narrow or qualitative analyses that fail to capture the diversity of real-world threats. To address this gap, we present the MCP Attack Library (MCPLIB), which categorizes and implements 31 distinct attack methods under four key classifications: direct tool injection, indirect tool injection, malicious user attacks, and LLM inherent attack. We further conduct a quantitative analysis of the efficacy of each attack. Our experiments reveal key insights into MCP vulnerabilities, including agents' blind reliance on tool descriptions, sensitivity to file-based attacks, chain attacks exploiting shared context, and difficulty distinguishing external data from executable commands. These insights, validated through attack experiments, underscore the urgency for robust defense strategies and informed MCP design. Our contributions include 1) constructing a comprehensive MCP attack taxonomy, 2) introducing a unified attack framework MCPLIB, and 3) conducting empirical vulnerability analysis to enhance MCP security mechanisms. This work provides a foundational framework, supporting the secure evolution of MCP ecosystems. 

**Abstract (ZH)**: MCP攻击库（MCPLIB）：构建全面的MCP攻击分类并进行定量分析以增强MCP安全性 

---
# CorrSteer: Steering Improves Task Performance and Safety in LLMs through Correlation-based Sparse Autoencoder Feature Selection 

**Title (ZH)**: CorrSteer：基于相关性稀疏自编码特征选择的蒸馏改进大规模语言模型的任务性能和安全性 

**Authors**: Seonglae Cho, Zekun Wu, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2508.12535)  

**Abstract**: Sparse Autoencoders (SAEs) can extract interpretable features from large language models (LLMs) without supervision. However, their effectiveness in downstream steering tasks is limited by the requirement for contrastive datasets or large activation storage. To address these limitations, we propose CorrSteer, which selects features by correlating sample correctness with SAE activations from generated tokens at inference time. This approach uses only inference-time activations to extract more relevant features, thereby avoiding spurious correlations. It also obtains steering coefficients from average activations, automating the entire pipeline. Our method shows improved task performance on QA, bias mitigation, jailbreaking prevention, and reasoning benchmarks on Gemma 2 2B and LLaMA 3.1 8B, notably achieving a +4.1% improvement in MMLU performance and a +22.9% improvement in HarmBench with only 4000 samples. Selected features demonstrate semantically meaningful patterns aligned with each task's requirements, revealing the underlying capabilities that drive performance. Our work establishes correlationbased selection as an effective and scalable approach for automated SAE steering across language model applications. 

**Abstract (ZH)**: Sparse 自编码器 (SAEs) 可以在不监督的情况下从大型语言模型 (LLMs) 中提取可解释的特征。然而，它们在下游引导任务中的有效性受到对比数据集或大规模激活存储的限制。为了解决这些限制，我们提出了一种名为 CorrSteer 的方法，该方法通过将样本正确性与生成词元的 SAE 激活进行相关性分析来选择特征。该方法仅使用推断时的激活来提取更具相关性的特征，从而避免了虚假相关性。它还从平均激活中获得引导系数，实现了整个流程的自动化。我们的方法在 QA、偏见缓解、破解预防和推理基准测试中均表现出改进的任务性能，特别地，在 Gemma 2 2B 和 LLaMA 3.1 8B 上分别实现了 MMLU 性能 +4.1% 的提升和 HarmBench 性能 +22.9% 的提升，仅使用 4000 个样本。所选特征展示了与每个任务要求相一致的语义相关模式，揭示了驱动性能的潜在能力。我们的工作确立了基于相关性的选择作为跨语言模型应用中自动 SAE 引导的有效且可扩展的方法。 

---
# Defining and Benchmarking a Data-Centric Design Space for Brain Graph Construction 

**Title (ZH)**: 数据导向的设计空间定义与基于脑图构建的基准测试 

**Authors**: Qinwen Ge, Roza G. Bayrak, Anwar Said, Catie Chang, Xenofon Koutsoukos, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2508.12533)  

**Abstract**: The construction of brain graphs from functional Magnetic Resonance Imaging (fMRI) data plays a crucial role in enabling graph machine learning for neuroimaging. However, current practices often rely on rigid pipelines that overlook critical data-centric choices in how brain graphs are constructed. In this work, we adopt a Data-Centric AI perspective and systematically define and benchmark a data-centric design space for brain graph construction, constrasting with primarily model-centric prior work. We organize this design space into three stages: temporal signal processing, topology extraction, and graph featurization. Our contributions lie less in novel components and more in evaluating how combinations of existing and modified techniques influence downstream performance. Specifically, we study high-amplitude BOLD signal filtering, sparsification and unification strategies for connectivity, alternative correlation metrics, and multi-view node and edge features, such as incorporating lagged dynamics. Experiments on the HCP1200 and ABIDE datasets show that thoughtful data-centric configurations consistently improve classification accuracy over standard pipelines. These findings highlight the critical role of upstream data decisions and underscore the importance of systematically exploring the data-centric design space for graph-based neuroimaging. Our code is available at this https URL. 

**Abstract (ZH)**: 从功能磁共振成像(fMRI)数据构建脑图对于实现神经影像学中的图机器学习起着关键作用。然而，当前的做法往往依赖于僵化的管道，忽视了构建脑图时的关键数据驱动选择。在本文中，我们从数据为中心的人工智能视角出发，系统地定义并评估了一个数据为中心的设计空间，以实现脑图构建，与先前主要为模型为中心的工作形成了对比。我们将这个设计空间分为三个阶段：时间信号处理、拓扑提取和图特征化。我们的贡献不在于新颖的组件，而在于评估现有和修改的技术组合如何影响下游性能。具体来说，我们研究了高振幅BOLD信号的滤波、连接性的稀疏化和统一策略、替代的相关性度量，以及多视图节点和边特征，例如引入滞后动态。在HCP1200和ABIDE数据集上的实验表明，仔细的数据驱动配置可以一致地提高分类准确性，这些发现突显了上游数据决策的关键作用，并强调了系统探索数据为中心的设计空间对于基于图的神经影像学的重要性。代码可供查看：this https URL 

---
# Rethinking Safety in LLM Fine-tuning: An Optimization Perspective 

**Title (ZH)**: 重新思考大规模语言模型微调中的安全性：一种优化视角 

**Authors**: Minseon Kim, Jin Myung Kwak, Lama Alssum, Bernard Ghanem, Philip Torr, David Krueger, Fazl Barez, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2508.12531)  

**Abstract**: Fine-tuning language models is commonly believed to inevitably harm their safety, i.e., refusing to respond to harmful user requests, even when using harmless datasets, thus requiring additional safety measures. We challenge this belief through systematic testing, showing that poor optimization choices, rather than inherent trade-offs, often cause safety problems, measured as harmful responses to adversarial prompts. By properly selecting key training hyper-parameters, e.g., learning rate, batch size, and gradient steps, we reduce unsafe model responses from 16\% to approximately 5\%, as measured by keyword matching, while maintaining utility performance. Based on this observation, we propose a simple exponential moving average (EMA) momentum technique in parameter space that preserves safety performance by creating a stable optimization path and retains the original pre-trained model's safety properties. Our experiments on the Llama families across multiple datasets (Dolly, Alpaca, ORCA) demonstrate that safety problems during fine-tuning can largely be avoided without specialized interventions, outperforming existing approaches that require additional safety data while offering practical guidelines for maintaining both model performance and safety during adaptation. 

**Abstract (ZH)**: Fine-tuning语言模型通常认为不可避免地会损害其安全性，即在使用无害数据集时拒绝回应有害用户请求，因此需要额外的安全措施。我们通过系统的测试来挑战这一观点，表明较差的优化选择而非固有的权衡通常会导致安全性问题，这些问题是通过关键词匹配衡量的有害回应。通过适当选择关键训练超参数，如学习率、批量大小和梯度步数，我们减少了大约11%的不安全模型响应，同时保持了实用性性能。基于这一观察，我们提出了一种简单的参数空间指数移动平均（EMA）动量技术，该技术通过创建稳定优化路径来保持安全性性能，并保留原始预训练模型的安全特性。我们在Llama家族（Dolly、Alpaca、ORCA）多个数据集上的实验表明，在不需要专门干预的情况下，可以通过这种方式避免 fine-tuning 过程中的安全性问题，同时优于需要额外安全数据的现有方法，并为在适应过程中同时保持模型性能和安全性提供了实用指南。 

---
# An Initial Study of Bird's-Eye View Generation for Autonomous Vehicles using Cross-View Transformers 

**Title (ZH)**: 基于交叉视图变换器的鸟瞰视图生成在自主车辆中的初步研究 

**Authors**: Felipe Carlos dos Santos, Eric Aislan Antonelo, Gustavo Claudio Karl Couto  

**Link**: [PDF](https://arxiv.org/pdf/2508.12520)  

**Abstract**: Bird's-Eye View (BEV) maps provide a structured, top-down abstraction that is crucial for autonomous-driving perception. In this work, we employ Cross-View Transformers (CVT) for learning to map camera images to three BEV's channels - road, lane markings, and planned trajectory - using a realistic simulator for urban driving. Our study examines generalization to unseen towns, the effect of different camera layouts, and two loss formulations (focal and L1). Using training data from only a town, a four-camera CVT trained with the L1 loss delivers the most robust test performance, evaluated in a new town. Overall, our results underscore CVT's promise for mapping camera inputs to reasonably accurate BEV maps. 

**Abstract (ZH)**: Bird's-Eye View Maps from Cross-View Transformers for Autonomous Driving Perception in Urban Scenarios 

---
# An Introduction to Sliced Optimal Transport 

**Title (ZH)**: 切片最优传输简介 

**Authors**: Khai Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.12519)  

**Abstract**: Sliced Optimal Transport (SOT) is a rapidly developing branch of optimal transport (OT) that exploits the tractability of one-dimensional OT problems. By combining tools from OT, integral geometry, and computational statistics, SOT enables fast and scalable computation of distances, barycenters, and kernels for probability measures, while retaining rich geometric structure. This paper provides a comprehensive review of SOT, covering its mathematical foundations, methodological advances, computational methods, and applications. We discuss key concepts of OT and one-dimensional OT, the role of tools from integral geometry such as Radon transform in projecting measures, and statistical techniques for estimating sliced distances. The paper further explores recent methodological advances, including non-linear projections, improved Monte Carlo approximations, statistical estimation techniques for one-dimensional optimal transport, weighted slicing techniques, and transportation plan estimation methods. Variational problems, such as minimum sliced Wasserstein estimation, barycenters, gradient flows, kernel constructions, and embeddings are examined alongside extensions to unbalanced, partial, multi-marginal, and Gromov-Wasserstein settings. Applications span machine learning, statistics, computer graphics and computer visions, highlighting SOT's versatility as a practical computational tool. This work will be of interest to researchers and practitioners in machine learning, data sciences, and computational disciplines seeking efficient alternatives to classical OT. 

**Abstract (ZH)**: 切片最优传输（SOT）是一种快速发展的最优传输（OT）分支，利用了一维最优传输问题的可处理性。通过结合最优传输、积分几何和计算统计的工具，SOT能够快速高效地计算概率测度的距离、测度中心和核函数，同时保留丰富的几何结构。本文对SOT进行了全面回顾，涵盖了其数学基础、方法论进展、计算方法和应用。文中讨论了一维最优传输的基本概念，积分几何工具如Radon变换在投影测度中的作用，以及估算切片距离的统计技术。此外，本文还探讨了近期的方法论进展，包括非线性投影、改进的蒙特卡洛近似、一维最优传输的统计估计技术、加权切片技术以及运输计划估计方法。还分析了变分问题，如切片Wasserstein估计算法、测度中心、梯度流、核构造和嵌入等问题及其在不平衡、部分、多边际和Gromov-Wasserstein设置中的扩展应用。应用范围涵盖了机器学习、统计学、计算机图形学和计算机视觉等领域，突显了SOT作为高效计算工具的灵活性。本研究将对寻求经典OT高效替代方案的研究人员和从业者产生兴趣。 

---
# Design and Validation of a Responsible Artificial Intelligence-based System for the Referral of Diabetic Retinopathy Patients 

**Title (ZH)**: 基于负责任的人工智能的糖尿病视网膜病变患者转诊系统的设计与验证 

**Authors**: E. Ulises Moya-Sánchez, Abraham Sánchez-Perez, Raúl Nanclares Da Veiga, Alejandro Zarate-Macías, Edgar Villareal, Alejandro Sánchez-Montes, Edtna Jauregui-Ulloa, Héctor Moreno, Ulises Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2508.12506)  

**Abstract**: Diabetic Retinopathy (DR) is a leading cause of vision loss in working-age individuals. Early detection of DR can reduce the risk of vision loss by up to 95%, but a shortage of retinologists and challenges in timely examination complicate detection. Artificial Intelligence (AI) models using retinal fundus photographs (RFPs) offer a promising solution. However, adoption in clinical settings is hindered by low-quality data and biases that may lead AI systems to learn unintended features. To address these challenges, we developed RAIS-DR, a Responsible AI System for DR screening that incorporates ethical principles across the AI lifecycle. RAIS-DR integrates efficient convolutional models for preprocessing, quality assessment, and three specialized DR classification models. We evaluated RAIS-DR against the FDA-approved EyeArt system on a local dataset of 1,046 patients, unseen by both systems. RAIS-DR demonstrated significant improvements, with F1 scores increasing by 5-12%, accuracy by 6-19%, and specificity by 10-20%. Additionally, fairness metrics such as Disparate Impact and Equal Opportunity Difference indicated equitable performance across demographic subgroups, underscoring RAIS-DR's potential to reduce healthcare disparities. These results highlight RAIS-DR as a robust and ethically aligned solution for DR screening in clinical settings. The code, weights of RAIS-DR are available at this https URL with RAIL. 

**Abstract (ZH)**: 糖尿病视网膜病变（DR）是工作年龄段人群视力丧失的主要原因。早期检测DR可以降低高达95%的视力丧失风险，但眼科医生短缺和及时检查的挑战使得检测更加复杂。使用视网膜底片照片（RFPs）的人工智能（AI）模型提供了一个有前景的解决方案。然而，在临床环境中采用这些模型受到低质量数据和可能导致AI系统学习非预期特征的偏差的影响。为了解决这些挑战，我们开发了RAIS-DR，这是一个负责的人工智能系统，用于DR筛查，并在整个AI生命周期中纳入了伦理原则。RAIS-DR结合了高效卷积模型进行预处理、质量评估和三种专门的DR分类模型。我们在未被两者系统见过的1,046名患者的地方数据集上对RAIS-DR进行了评估，结果显示RAIS-DR取得了显著改进，F1分数提升了5-12%，准确率提升了6-19%，特异性提升了10-20%。此外，公平性指标如差异影响和等机会差异表明RAIS-DR在不同人口亚组中表现一致，突显了其减少医疗保健不平等的潜力。这些结果强调了RAIS-DR作为临床环境中DR筛查稳健且伦理对齐的解决方案的重要性。RAIS-DR的代码和权重可以在RAIL提供的链接中获得。 

---
# Mitigating Hallucinations in Large Language Models via Causal Reasoning 

**Title (ZH)**: 通过因果推理减轻大型语言模型的幻觉问题 

**Authors**: Yuangang Li, Yiqing Shen, Yi Nian, Jiechao Gao, Ziyi Wang, Chenxiao Yu, Shawn Li, Jie Wang, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12495)  

**Abstract**: Large language models (LLMs) exhibit logically inconsistent hallucinations that appear coherent yet violate reasoning principles, with recent research suggesting an inverse relationship between causal reasoning capabilities and such hallucinations. However, existing reasoning approaches in LLMs, such as Chain-of-Thought (CoT) and its graph-based variants, operate at the linguistic token level rather than modeling the underlying causal relationships between variables, lacking the ability to represent conditional independencies or satisfy causal identification assumptions. To bridge this gap, we introduce causal-DAG construction and reasoning (CDCR-SFT), a supervised fine-tuning framework that trains LLMs to explicitly construct variable-level directed acyclic graph (DAG) and then perform reasoning over it. Moreover, we present a dataset comprising 25,368 samples (CausalDR), where each sample includes an input question, explicit causal DAG, graph-based reasoning trace, and validated answer. Experiments on four LLMs across eight tasks show that CDCR-SFT improves the causal reasoning capability with the state-of-the-art 95.33% accuracy on CLADDER (surpassing human performance of 94.8% for the first time) and reduces the hallucination on HaluEval with 10% improvements. It demonstrates that explicit causal structure modeling in LLMs can effectively mitigate logical inconsistencies in LLM outputs. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）表现出逻辑上不一致但看似连贯的幻觉，最近的研究表明，因果推理能力与这种幻觉之间存在反比关系。然而，现有的LLMs推理方法，如思维链（CoT）及其图基变种，仅在语言令牌级别操作，而不是建模变量之间的潜在因果关系，缺乏表示条件独立性的能力或满足因果识别假设的能力。为解决这一问题，我们引入了因果DAG构建与推理（CDCR-SFT）监督微调框架，该框架训练LLMs显式构建变量级有向无环图（DAG），然后在其上进行推理。此外，我们还提供了一个包含25,368个样本的数据集（CausalDR），每个样本包含输入问题、显式的因果DAG、基于图的推理跟踪以及验证答案。在四个LLM在八个任务上的实验表明，CDCR-SFT提高了因果推理能力，在CLADDER上的准确率达到95.33%（首次超过人类性能94.8%），并在HaluEval上减少了10%的幻觉。这表明在LLMs中显式建模因果结构可以有效缓解LLMs输出中的逻辑不一致性。代码可在以下链接获取。 

---
# Cold-RL: Learning Cache Eviction with Offline Reinforcement Learning for NGINX 

**Title (ZH)**: 冷启动强化学习：基于离线强化学习的NGINX缓存淘汰学习 

**Authors**: Aayush Gupta, Arpit Bhayani  

**Link**: [PDF](https://arxiv.org/pdf/2508.12485)  

**Abstract**: Web proxies such as NGINX commonly rely on least-recently-used (LRU) eviction, which is size agnostic and can thrash under periodic bursts and mixed object sizes. We introduce Cold-RL, a learned eviction policy for NGINX that replaces LRU's forced-expire path with a dueling Deep Q-Network served by an ONNX sidecar within a strict microsecond budget. On each eviction, Cold-RL samples the K least-recently-used objects, extracts six lightweight features (age, size, hit count, inter-arrival time, remaining TTL, and last origin RTT), and requests a bitmask of victims; a hard timeout of 500 microseconds triggers immediate fallback to native LRU. Policies are trained offline by replaying NGINX access logs through a cache simulator with a simple reward: a retained object earns one point if it is hit again before TTL expiry. We compare against LRU, LFU, size-based, adaptive LRU, and a hybrid baseline on two adversarial workloads. With a 25 MB cache, Cold-RL raises hit ratio from 0.1436 to 0.3538, a 146 percent improvement over the best classical baseline; at 100 MB, from 0.7530 to 0.8675, a 15 percent gain; and at 400 MB it matches classical methods (about 0.918). Inference adds less than 2 percent CPU overhead and keeps 95th percentile eviction latency within budget. To our knowledge, this is the first reinforcement learning eviction policy integrated into NGINX with strict SLOs. 

**Abstract (ZH)**: Cold-RL：一种集成到NGINX中的严格SLO约束下的强化学习置换策略 

---
# EXOTIC: An Exact, Optimistic, Tree-Based Algorithm for Min-Max Optimization 

**Title (ZH)**: EXOTIC: 一种精确的乐观树基最小最大优化算法 

**Authors**: Chinmay Maheshwari, Chinmay Pimpalkhare, Debasish Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2508.12479)  

**Abstract**: Min-max optimization arises in many domains such as game theory, adversarial machine learning, etc., with gradient-based methods as a typical computational tool. Beyond convex-concave min-max optimization, the solutions found by gradient-based methods may be arbitrarily far from global optima. In this work, we present an algorithmic apparatus for computing globally optimal solutions in convex-non-concave and non-convex-concave min-max optimization. For former, we employ a reformulation that transforms it into a non-concave-convex max-min optimization problem with suitably defined feasible sets and objective function. The new form can be viewed as a generalization of Sion's minimax theorem. Next, we introduce EXOTIC-an Exact, Optimistic, Tree-based algorithm for solving the reformulated max-min problem. EXOTIC employs an iterative convex optimization solver to (approximately) solve the inner minimization and a hierarchical tree search for the outer maximization to optimistically select promising regions to search based on the approximate solution returned by convex optimization solver. We establish an upper bound on its optimality gap as a function of the number of calls to the inner solver, the solver's convergence rate, and additional problem-dependent parameters. Both our algorithmic apparatus along with its accompanying theoretical analysis can also be applied for non-convex-concave min-max optimization. In addition, we propose a class of benchmark convex-non-concave min-max problems along with their analytical global solutions, providing a testbed for evaluating algorithms for min-max optimization. Empirically, EXOTIC outperforms gradient-based methods on this benchmark as well as on existing numerical benchmark problems from the literature. Finally, we demonstrate the utility of EXOTIC by computing security strategies in multi-player games with three or more players. 

**Abstract (ZH)**: 基于梯度的方法在博弈论、对抗机器学习等领域中已经广泛应用到最小-最大优化中，但这些方法找到的解可能与全局最优解任意偏离。本文提出了一种算法框架，用于在凸非凹和非凸非凹最小-最大优化中计算全局最优解。对于前者，我们通过重新形式化问题，将其转换为具有适当定义的可行集和目标函数的非凹-凸最大-最小优化问题，新的形式可以看作是Sion的最小-最大定理的推广。接下来，我们引入了EXOTIC算法——一种精确、乐观、基于树结构的方法，用于求解重新形式化后的最大-最小问题。EXOTIC使用迭代凸优化求解器来（近似）解决内部最小化问题，并采用分层树搜索进行外部最大化，基于凸优化求解器返回的近似解来乐观地选择具有潜在解的区域进行搜索。我们基于内部求解器调用次数、求解器的收敛率以及其它问题相关参数建立了最优性差距的上界。我们的算法框架及其理论分析也可应用于非凸非凹最小-最大优化。此外，我们提出了一类用于评估最小-最大优化算法的基准凸非凹最小-最大问题及其解析全局解。实验结果显示，EXOTIC在基准测试及文献中现有的数值基准问题上优于基于梯度的方法。最后，我们展示了EXOTIC在计算三人或以上参与者的多玩家博弈中的安全策略方面的应用。 

---
# Standardization of Neuromuscular Reflex Analysis -- Role of Fine-Tuned Vision-Language Model Consortium and OpenAI gpt-oss Reasoning LLM Enabled Decision Support System 

**Title (ZH)**: 神经肌肉反射分析的标准规范化——细腻调校的视觉-语言模型 consortium 和 OpenAI gpt-oss 基础推理大模型支撑决策系统的角色 

**Authors**: Eranga Bandara, Ross Gore, Sachin Shetty, Ravi Mukkamala, Christopher Rhea, Atmaram Yarlagadda, Shaifali Kaushik, L.H.M.P.De Silva, Andriy Maznychenko, Inna Sokolowska, Amin Hass, Kasun De Zoysa  

**Link**: [PDF](https://arxiv.org/pdf/2508.12473)  

**Abstract**: Accurate assessment of neuromuscular reflexes, such as the H-reflex, plays a critical role in sports science, rehabilitation, and clinical neurology. Traditional analysis of H-reflex EMG waveforms is subject to variability and interpretation bias among clinicians and researchers, limiting reliability and standardization. To address these challenges, we propose a Fine-Tuned Vision-Language Model (VLM) Consortium and a reasoning Large-Language Model (LLM)-enabled Decision Support System for automated H-reflex waveform interpretation and diagnosis. Our approach leverages multiple VLMs, each fine-tuned on curated datasets of H-reflex EMG waveform images annotated with clinical observations, recovery timelines, and athlete metadata. These models are capable of extracting key electrophysiological features and predicting neuromuscular states, including fatigue, injury, and recovery, directly from EMG images and contextual metadata. Diagnostic outputs from the VLM consortium are aggregated using a consensus-based method and refined by a specialized reasoning LLM, which ensures robust, transparent, and explainable decision support for clinicians and sports scientists. The end-to-end platform orchestrates seamless communication between the VLM ensemble and the reasoning LLM, integrating prompt engineering strategies and automated reasoning workflows using LLM Agents. Experimental results demonstrate that this hybrid system delivers highly accurate, consistent, and interpretable H-reflex assessments, significantly advancing the automation and standardization of neuromuscular diagnostics. To our knowledge, this work represents the first integration of a fine-tuned VLM consortium with a reasoning LLM for image-based H-reflex analysis, laying the foundation for next-generation AI-assisted neuromuscular assessment and athlete monitoring platforms. 

**Abstract (ZH)**: 精细调整的视觉-语言模型联盟及推理大型语言模型赋能的决策支持系统在H-反射波形解释与诊断中的应用 

---
# A Robust Cross-Domain IDS using BiGRU-LSTM-Attention for Medical and Industrial IoT Security 

**Title (ZH)**: 跨域医疗和工业物联网安全的鲁棒双向GRU-LSTM注意力机制入侵检测系统 

**Authors**: Afrah Gueriani, Hamza Kheddar, Ahmed Cherif Mazari, Mohamed Chahine Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.12470)  

**Abstract**: The increased Internet of Medical Things IoMT and the Industrial Internet of Things IIoT interconnectivity has introduced complex cybersecurity challenges, exposing sensitive data, patient safety, and industrial operations to advanced cyber threats. To mitigate these risks, this paper introduces a novel transformer-based intrusion detection system IDS, termed BiGAT-ID a hybrid model that combines bidirectional gated recurrent units BiGRU, long short-term memory LSTM networks, and multi-head attention MHA. The proposed architecture is designed to effectively capture bidirectional temporal dependencies, model sequential patterns, and enhance contextual feature representation. Extensive experiments on two benchmark datasets, CICIoMT2024 medical IoT and EdgeIIoTset industrial IoT demonstrate the model's cross-domain robustness, achieving detection accuracies of 99.13 percent and 99.34 percent, respectively. Additionally, the model exhibits exceptional runtime efficiency, with inference times as low as 0.0002 seconds per instance in IoMT and 0.0001 seconds in IIoT scenarios. Coupled with a low false positive rate, BiGAT-ID proves to be a reliable and efficient IDS for deployment in real-world heterogeneous IoT environments 

**Abstract (ZH)**: IoMT和IIoT增强的互联系统中的新型基于变压器的入侵检测系统BiGAT-ID：一种结合BiGRU、LSTM和MHA的混合模型 

---
# Inverse-LLaVA: Eliminating Alignment Pre-training Through Text-to-Vision Mapping 

**Title (ZH)**: 逆向LLaVA：通过文本到视觉映射消除对齐预训练 

**Authors**: Xuhui Zhan, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2508.12466)  

**Abstract**: Traditional multimodal learning approaches require expensive alignment pre-training to bridge vision and language modalities, typically projecting visual features into discrete text token spaces. We challenge both fundamental assumptions underlying this paradigm by proposing Inverse-LLaVA, a novel approach that eliminates alignment pre-training entirely while inverting the conventional mapping direction. Rather than projecting visual features to text space, our method maps text embeddings into continuous visual representation space and performs fusion within transformer intermediate layers. Through selective additive components in attention mechanisms, we enable dynamic integration of visual and textual representations without requiring massive image-text alignment datasets. Comprehensive experiments across nine multimodal benchmarks demonstrate nuanced performance trade-offs: Inverse-LLaVA achieves notable improvements on reasoning-intensive and cognitive tasks (MM-VET: +0.2%, VizWiz: +1.8%, ScienceQA: +0.2%, cognitive reasoning: +27.2%), while showing expected decreases in perception tasks requiring memorized visual-text associations (celebrity recognition: -49.5%, OCR: -21.3%). These results provide the first empirical evidence that alignment pre-training is not necessary for effective multimodal learning, particularly for complex reasoning tasks. Our work establishes the feasibility of a new paradigm that reduces computational requirements by 45%, challenges conventional wisdom about modality fusion, and opens new research directions for efficient multimodal architectures that preserve modality-specific characteristics. Our project website with code and additional resources is available at this https URL. 

**Abstract (ZH)**: 传统多模态学习方法需要昂贵的对齐预训练来连接视觉和语言模态，通常将视觉特征投影到离散的文本标记空间。我们通过提出Inverse-LLaVA这一全新方法，挑战这一范式的基本假设，该方法完全消除了对齐预训练，逆转了传统的映射方向。我们的方法将文本嵌入映射到连续的视觉表示空间，并在变压器中间层进行融合。通过选择性添加注意力机制中的组件，我们能够在不需要大量图像-文本对齐数据集的情况下实现视觉和文本表示的动态集成。在九个多模态基准上的综合实验展示了细腻的性能权衡：Inverse-LLaVA在推理密集和认知任务上取得了显著改进（MM-VET: +0.2%, VizWiz: +1.8%, ScienceQA: +0.2%, 认知推理: +27.2%），但在需要记忆视觉-文本关联的感知任务上表现出预期的下降（名人识别: -49.5%, OCR: -21.3%）。这些结果提供了首个实验证据，证明对齐预训练不是有效的多模态学习所必需的，尤其是在复杂的推理任务中。我们的研究工作建立了可行性，通过减少45%的计算要求，挑战了模态融合的传统智慧，并为保留模态特定特征的有效多模态架构开辟了新的研究方向。我们的项目网站包含代码和额外资源，网址为：this https URL。 

---
# Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots 

**Title (ZH)**: 内置关节传感器的工业机器人触觉手势识别 

**Authors**: Deqing Song, Weimin Yang, Maryam Rezayati, Hans Wernher van de Venn  

**Link**: [PDF](https://arxiv.org/pdf/2508.12435)  

**Abstract**: While gesture recognition using vision or robot skins is an active research area in Human-Robot Collaboration (HRC), this paper explores deep learning methods relying solely on a robot's built-in joint sensors, eliminating the need for external sensors. We evaluated various convolutional neural network (CNN) architectures and collected two datasets to study the impact of data representation and model architecture on the recognition accuracy. Our results show that spectrogram-based representations significantly improve accuracy, while model architecture plays a smaller role. We also tested generalization to new robot poses, where spectrogram-based models performed better. Implemented on a Franka Emika Research robot, two of our methods, STFT2DCNN and STT3DCNN, achieved over 95% accuracy in contact detection and gesture classification. These findings demonstrate the feasibility of external-sensor-free tactile recognition and promote further research toward cost-effective, scalable solutions for HRC. 

**Abstract (ZH)**: 基于内置关节传感器的深度学习方法在机器人手臂触觉识别中的应用：无需外部传感器的手势识别 

---
# Adversarial Attacks on VQA-NLE: Exposing and Alleviating Inconsistencies in Visual Question Answering Explanations 

**Title (ZH)**: 针对VQA-NLE的对抗攻击：揭露并缓解视觉问答解释中的不一致性 

**Authors**: Yahsin Yeh, Yilun Wu, Bokai Ruan, Honghan Shuai  

**Link**: [PDF](https://arxiv.org/pdf/2508.12430)  

**Abstract**: Natural language explanations in visual question answering (VQA-NLE) aim to make black-box models more transparent by elucidating their decision-making processes. However, we find that existing VQA-NLE systems can produce inconsistent explanations and reach conclusions without genuinely understanding the underlying context, exposing weaknesses in either their inference pipeline or explanation-generation mechanism. To highlight these vulnerabilities, we not only leverage an existing adversarial strategy to perturb questions but also propose a novel strategy that minimally alters images to induce contradictory or spurious outputs. We further introduce a mitigation method that leverages external knowledge to alleviate these inconsistencies, thereby bolstering model robustness. Extensive evaluations on two standard benchmarks and two widely used VQA-NLE models underscore the effectiveness of our attacks and the potential of knowledge-based defenses, ultimately revealing pressing security and reliability concerns in current VQA-NLE systems. 

**Abstract (ZH)**: 自然语言解释在视觉问答（VQA）中的应用旨在通过阐明其决策过程使黑盒模型更加透明。然而，我们发现现有VQA-NLE系统会产生不一致的解释，并在未真正理解底层上下文的情况下得出结论，暴露了它们推理管道或解释生成机制中的弱点。为了突出这些弱点，我们不仅利用现有的对抗策略来扰动问题，还提出了一种新的策略，通过最小改变图像来诱导矛盾或虚假的输出。我们进一步引入了一种利用外部知识的方法来缓解这些不一致性，从而增强模型的稳健性。广泛的标准基准和广泛使用的VQA-NLE模型的评估证实了我们攻击的有效性和基于知识的防护潜力，最终揭示了当前VQA-NLE系统中存在的迫切的安全性和可靠性问题。 

---
# fCrit: A Visual Explanation System for Furniture Design Creative Support 

**Title (ZH)**: fCrit: 家具设计创意支持的视觉解释系统 

**Authors**: Vuong Nguyen, Gabriel Vigliensoni  

**Link**: [PDF](https://arxiv.org/pdf/2508.12416)  

**Abstract**: We introduce fCrit, a dialogue-based AI system designed to critique furniture design with a focus on explainability. Grounded in reflective learning and formal analysis, fCrit employs a multi-agent architecture informed by a structured design knowledge base. We argue that explainability in the arts should not only make AI reasoning transparent but also adapt to the ways users think and talk about their designs. We demonstrate how fCrit supports this process by tailoring explanations to users' design language and cognitive framing. This work contributes to Human-Centered Explainable AI (HCXAI) in creative practice, advancing domain-specific methods for situated, dialogic, and visually grounded AI support. 

**Abstract (ZH)**: 基于对话的可解释人工智能系统fCrit及其在家具设计批判中的应用 

---
# Quantum Flow Matching 

**Title (ZH)**: 量子流匹配 

**Authors**: Zidong Cui, Pan Zhang, Ying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12413)  

**Abstract**: Flow matching has rapidly become a dominant paradigm in classical generative modeling, offering an efficient way to interpolate between two complex distributions. We extend this idea to the quantum realm and introduce Quantum Flow Matching (QFM)-a fully quantum-circuit realization that offers efficient interpolation between two density matrices. QFM offers systematic preparation of density matrices and generation of samples for accurately estimating observables, and can be realized on a quantum computer without the need for costly circuit redesigns. We validate its versatility on a set of applications: (i) generating target states with prescribed magnetization and entanglement entropy, (ii) estimating nonequilibrium free-energy differences to test the quantum Jarzynski equality, and (iii) expediting the study on superdiffusion breakdown. These results position QFM as a unifying and promising framework for generative modeling across quantum systems. 

**Abstract (ZH)**: 量子流匹配（QFM）：量子领域的一种高效插值方法 

---
# LumiMAS: A Comprehensive Framework for Real-Time Monitoring and Enhanced Observability in Multi-Agent Systems 

**Title (ZH)**: LumiMAS：多agent系统中实时监控与增强可观测性的综合框架 

**Authors**: Ron Solomon, Yarin Yerushalmi Levi, Lior Vaknin, Eran Aizikovich, Amit Baras, Etai Ohana, Amit Giloni, Shamik Bose, Chiara Picardi, Yuval Elovici, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2508.12412)  

**Abstract**: The incorporation of large language models in multi-agent systems (MASs) has the potential to significantly improve our ability to autonomously solve complex problems. However, such systems introduce unique challenges in monitoring, interpreting, and detecting system failures. Most existing MAS observability frameworks focus on analyzing each individual agent separately, overlooking failures associated with the entire MAS. To bridge this gap, we propose LumiMAS, a novel MAS observability framework that incorporates advanced analytics and monitoring techniques. The proposed framework consists of three key components: a monitoring and logging layer, anomaly detection layer, and anomaly explanation layer. LumiMAS's first layer monitors MAS executions, creating detailed logs of the agents' activity. These logs serve as input to the anomaly detection layer, which detects anomalies across the MAS workflow in real time. Then, the anomaly explanation layer performs classification and root cause analysis (RCA) of the detected anomalies. LumiMAS was evaluated on seven different MAS applications, implemented using two popular MAS platforms, and a diverse set of possible failures. The applications include two novel failure-tailored applications that illustrate the effects of a hallucination or bias on the MAS. The evaluation results demonstrate LumiMAS's effectiveness in failure detection, classification, and RCA. 

**Abstract (ZH)**: 大型语言模型在多agent系统中的集成有望显著提升我们自主解决复杂问题的能力。然而，这样的系统引入了监测、解释和检测系统故障的独特挑战。现有的大多数MAS可观测性框架集中在单独分析每个代理上，忽视了与整个MAS相关的故障。为弥补这一不足，我们提出了LumiMAS，一种新颖的MAS可观测性框架，结合了先进的分析和监测技术。该框架包括三个关键组件：监控和日志记录层、异常检测层和异常解释层。LumiMAS的第一层监控MAS执行过程，创建详细的代理活动日志。这些日志作为输入传递给异常检测层，该层可以实时检测MAS工作流程中的异常。然后，异常解释层对检测到的异常进行分类和根本原因分析(RCA)。LumiMAS在七个不同的MAS应用中进行了评估，这些应用使用了两种流行的MAS平台实现，并针对多样的潜在故障进行了实施。所评估的应用包括两个新型的故障定制应用，展示了幻觉或偏差对MAS的影响。评估结果表明，LumiMAS在故障检测、分类和根本原因分析方面具有有效性。 

---
# SRMA-Mamba: Spatial Reverse Mamba Attention Network for Pathological Liver Segmentation in MRI Volumes 

**Title (ZH)**: SRMA-Mamba：空间逆Mamba注意力网络在MRI体积中用于病理肝脏分割 

**Authors**: Jun Zeng, Yannan Huang, Elif Keles, Halil Ertugrul Aktas, Gorkem Durak, Nikhil Kumar Tomar, Quoc-Huy Trinh, Deepak Ranjan Nayak, Ulas Bagci, Debesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2508.12410)  

**Abstract**: Liver Cirrhosis plays a critical role in the prognosis of chronic liver disease. Early detection and timely intervention are critical in significantly reducing mortality rates. However, the intricate anatomical architecture and diverse pathological changes of liver tissue complicate the accurate detection and characterization of lesions in clinical settings. Existing methods underutilize the spatial anatomical details in volumetric MRI data, thereby hindering their clinical effectiveness and explainability. To address this challenge, we introduce a novel Mamba-based network, SRMA-Mamba, designed to model the spatial relationships within the complex anatomical structures of MRI volumes. By integrating the Spatial Anatomy-Based Mamba module (SABMamba), SRMA-Mamba performs selective Mamba scans within liver cirrhotic tissues and combines anatomical information from the sagittal, coronal, and axial planes to construct a global spatial context representation, enabling efficient volumetric segmentation of pathological liver structures. Furthermore, we introduce the Spatial Reverse Attention module (SRMA), designed to progressively refine cirrhotic details in the segmentation map, utilizing both the coarse segmentation map and hierarchical encoding features. Extensive experiments demonstrate that SRMA-Mamba surpasses state-of-the-art methods, delivering exceptional performance in 3D pathological liver segmentation. Our code is available for public: {\color{blue}{this https URL}}. 

**Abstract (ZH)**: 肝脏硬化在慢性肝病的预后中起着关键作用。早期检测和及时干预对于显著降低 mortality 率至关重要。然而，肝脏组织复杂的解剖结构和多种病理变化在临床检测和病变表征中增加了复杂性。现有方法在利用容积 MRI 数据的空间解剖细节方面存在不足，从而限制了其临床效果和可解释性。为了应对这一挑战，我们引入了一种基于 Mamba 的新型网络 SRMA-Mamba，用于建模 MRI 体积中复杂解剖结构内的空间关系。通过整合基于空间解剖学的 Mamba 模块（SABMamba），SRMA-Mamba 在肝脏硬化组织中执行选择性的 Mamba 扫描，并结合矢状面、冠状面和轴向面的解剖信息来构建全局空间上下文表示，从而实现病理肝脏结构的高效容积分割。此外，我们还引入了空间反向注意力模块（SRMA），该模块通过利用粗略分割图和层次编码特征来逐步细化分割图中的硬化细节。广泛实验表明，SRMA-Mamba 超过了最先进的方法，在 3D 病理肝脏分割任务中表现出色。我们的代码已公开：[this https URL]。 

---
# Extracting Post-Acute Sequelae of SARS-CoV-2 Infection Symptoms from Clinical Notes via Hybrid Natural Language Processing 

**Title (ZH)**: 基于混合自然语言处理从临床笔记中提取新冠病毒感染急性后遗症状 

**Authors**: Zilong Bai, Zihan Xu, Cong Sun, Chengxi Zang, H. Timothy Bunnell, Catherine Sinfield, Jacqueline Rutter, Aaron Thomas Martinez, L. Charles Bailey, Mark Weiner, Thomas R. Campion, Thomas Carton, Christopher B. Forrest, Rainu Kaushal, Fei Wang, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.12405)  

**Abstract**: Accurately and efficiently diagnosing Post-Acute Sequelae of COVID-19 (PASC) remains challenging due to its myriad symptoms that evolve over long- and variable-time intervals. To address this issue, we developed a hybrid natural language processing pipeline that integrates rule-based named entity recognition with BERT-based assertion detection modules for PASC-symptom extraction and assertion detection from clinical notes. We developed a comprehensive PASC lexicon with clinical specialists. From 11 health systems of the RECOVER initiative network across the U.S., we curated 160 intake progress notes for model development and evaluation, and collected 47,654 progress notes for a population-level prevalence study. We achieved an average F1 score of 0.82 in one-site internal validation and 0.76 in 10-site external validation for assertion detection. Our pipeline processed each note at $2.448\pm 0.812$ seconds on average. Spearman correlation tests showed $\rho >0.83$ for positive mentions and $\rho >0.72$ for negative ones, both with $P <0.0001$. These demonstrate the effectiveness and efficiency of our models and their potential for improving PASC diagnosis. 

**Abstract (ZH)**: 准确而高效地诊断新冠长期症状（PASC）仍然具有挑战性，这是因为其症状多样且随着时间的推移会演变。为解决这一问题，我们开发了一个集成基于规则的命名实体识别和基于BERT的断言检测模块的混合自然语言处理管道，用于从临床笔记中提取和检测PASC症状的断言。我们与临床专家合作开发了一个全面的PASC词汇表。来自美国RECOVER倡议网络的11个医疗系统，我们为模型开发和评估整理了160份入院进度报告，并收集了47,654份进度报告进行人群水平的流行病学研究。我们在单中心内部验证中的平均F1分数为0.82，在多中心外部验证中为0.76。该管道平均每份笔记处理时间为$2.448\pm 0.812$秒。皮尔森相关性检验显示，对于阳性提及，$\rho >0.83$；对于阴性提及，$\rho >0.72$，且两者均在$P <0.0001$。这些结果表明我们模型的有效性和效率，并展示了其在PASC诊断改进中的潜在价值。 

---
# Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position 

**Title (ZH)**: 从何处开始对齐？扩散大语言模型可能需要一个独特的定位。 

**Authors**: Zhixin Xie, Xurui Song, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.12398)  

**Abstract**: Diffusion Large Language Models (dLLMs) have recently emerged as a competitive non-autoregressive paradigm due to their unique training and inference approach. However, there is currently a lack of safety study on this novel architecture. In this paper, we present the first analysis of dLLMs' safety performance and propose a novel safety alignment method tailored to their unique generation characteristics. Specifically, we identify a critical asymmetry between the defender and attacker in terms of security. For the defender, we reveal that the middle tokens of the response, rather than the initial ones, are more critical to the overall safety of dLLM outputs; this seems to suggest that aligning middle tokens can be more beneficial to the defender. The attacker, on the contrary, may have limited power to manipulate middle tokens, as we find dLLMs have a strong tendency towards a sequential generation order in practice, forcing the attack to meet this distribution and diverting it from influencing the critical middle tokens. Building on this asymmetry, we introduce Middle-tOken Safety Alignment (MOSA), a novel method that directly aligns the model's middle generation with safe refusals exploiting reinforcement learning. We implement MOSA and compare its security performance against eight attack methods on two benchmarks. We also test the utility of MOSA-aligned dLLM on coding, math, and general reasoning. The results strongly prove the superiority of MOSA. 

**Abstract (ZH)**: Diffusion大语言模型的安全性分析与中间令牌安全性对齐方法 

---
# MedKGent: A Large Language Model Agent Framework for Constructing Temporally Evolving Medical Knowledge Graph 

**Title (ZH)**: MedKGent: 一种构建时序演变医疗知识图谱的大语言模型代理框架 

**Authors**: Duzhen Zhang, Zixiao Wang, Zhong-Zhi Li, Yahan Yu, Shuncheng Jia, Jiahua Dong, Haotian Xu, Xing Wu, Yingying Zhang, Tielin Zhang, Jie Yang, Xiuying Chen, Le Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.12393)  

**Abstract**: The rapid expansion of medical literature presents growing challenges for structuring and integrating domain knowledge at scale. Knowledge Graphs (KGs) offer a promising solution by enabling efficient retrieval, automated reasoning, and knowledge discovery. However, current KG construction methods often rely on supervised pipelines with limited generalizability or naively aggregate outputs from Large Language Models (LLMs), treating biomedical corpora as static and ignoring the temporal dynamics and contextual uncertainty of evolving knowledge. To address these limitations, we introduce MedKGent, a LLM agent framework for constructing temporally evolving medical KGs. Leveraging over 10 million PubMed abstracts published between 1975 and 2023, we simulate the emergence of biomedical knowledge via a fine-grained daily time series. MedKGent incrementally builds the KG in a day-by-day manner using two specialized agents powered by the Qwen2.5-32B-Instruct model. The Extractor Agent identifies knowledge triples and assigns confidence scores via sampling-based estimation, which are used to filter low-confidence extractions and inform downstream processing. The Constructor Agent incrementally integrates the retained triples into a temporally evolving graph, guided by confidence scores and timestamps to reinforce recurring knowledge and resolve conflicts. The resulting KG contains 156,275 entities and 2,971,384 relational triples. Quality assessments by two SOTA LLMs and three domain experts demonstrate an accuracy approaching 90\%, with strong inter-rater agreement. To evaluate downstream utility, we conduct RAG across seven medical question answering benchmarks using five leading LLMs, consistently observing significant improvements over non-augmented baselines. Case studies further demonstrate the KG's value in literature-based drug repurposing via confidence-aware causal inference. 

**Abstract (ZH)**: 医学文献的快速扩展为大规模结构化和整合领域知识带来了越来越大的挑战。知识图谱（KGs）提供了一种有前景的解决方案，通过实现高效检索、自动化推理和知识发现。然而，当前的KG构建方法往往依赖于监督式管道，具有有限的普适性，或者简单地汇总大型语言模型（LLMs）的输出，将生物医学文献视为静态的，忽略了随着时间演化而变化的知识的时间动态性和上下文不确定性。为了解决这些限制，我们引入了MedKGent，这是一种用于构建时间演变医学KG的LLM代理框架。通过利用1975年至2023年间发布的超过1000万篇PubMed摘要，我们通过精细的日时间序列模拟生物医学知识的涌现。MedKGent 以一天一构建的方式逐步构建KG，使用由Qwen2.5-32B-Instruct模型驱动的两个专业代理。提取代理通过对估计采样的知识三元组进行识别并分配置信度评分，这些评分用于过滤低置信度的提取并指导后续处理。构建代理根据置信度评分和时间戳逐步将保留的三元组整合进一个时间演变图中，以强化 recurring 知识并解决冲突。生成的KG包含156,275个实体和2,971,384个关系三元组。由两个领先的大规模语言模型和三位领域专家进行的质量评估显示准确率接近90%，且存在强烈的评判者间一致性。为评估下游应用价值，我们在七个医学问答基准上进行了检索增强生成（RAG），并观察到相对于未经增强的基线的显著性能提升。案例研究进一步证明了KG在基于文献的知识敏感因果推理药物重新定位方面的价值。 

---
# IPGPhormer: Interpretable Pathology Graph-Transformer for Survival Analysis 

**Title (ZH)**: IPGPhormer: 可解释的病理图形变换器用于生存分析 

**Authors**: Guo Tang, Songhan Jiang, Jinpeng Lu, Linghan Cai, Yongbing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12381)  

**Abstract**: Pathological images play an essential role in cancer prognosis, while survival analysis, which integrates computational techniques, can predict critical clinical events such as patient mortality or disease recurrence from whole-slide images (WSIs). Recent advancements in multiple instance learning have significantly improved the efficiency of survival analysis. However, existing methods often struggle to balance the modeling of long-range spatial relationships with local contextual dependencies and typically lack inherent interpretability, limiting their clinical utility. To address these challenges, we propose the Interpretable Pathology Graph-Transformer (IPGPhormer), a novel framework that captures the characteristics of the tumor microenvironment and models their spatial dependencies across the tissue. IPGPhormer uniquely provides interpretability at both tissue and cellular levels without requiring post-hoc manual annotations, enabling detailed analyses of individual WSIs and cross-cohort assessments. Comprehensive evaluations on four public benchmark datasets demonstrate that IPGPhormer outperforms state-of-the-art methods in both predictive accuracy and interpretability. In summary, our method, IPGPhormer, offers a promising tool for cancer prognosis assessment, paving the way for more reliable and interpretable decision-support systems in pathology. The code is publicly available at this https URL. 

**Abstract (ZH)**: 病理图像在癌症预后中发挥着重要作用，而结合计算技术的生存分析可以从玻片图像（WSI）中预测关键临床事件，如患者死亡或疾病复发。近期多实例学习的进步显著提高了生存分析的效率。然而，现有方法往往难以平衡长时间尺度的空间关系建模与局部上下文依赖性建模，并且通常缺乏内在可解释性，限制了它们的临床应用。为了解决这些挑战，我们提出了一种新型框架Interpretable Pathology Graph-Transformer (IPGPhormer)，该框架能够捕捉肿瘤微环境的特性，并建模其在组织中的空间依赖性。IPGPhormer在组织和细胞两个层面提供了无需后处理手动注释的可解释性，从而允许对单个玻片图像进行详细分析，并进行跨队列评估。在四个公开基准数据集上的全面评估表明，IPGPhormer在预测准确性和可解释性方面均优于现有最先进的方法。总之，我们的方法IPGPhormer为癌症预后评估提供了一个有前景的工具，铺平了病理学中更可靠和可解释的决策支持系统的道路。代码已公开，可在以下链接访问：this https URL。 

---
# Navigating the Exploration-Exploitation Tradeoff in Inference-Time Scaling of Diffusion Models 

**Title (ZH)**: 探索-利用权衡在推断时扩散模型的缩放中导航 

**Authors**: Xun Su, Jianming Huang, Yang Yusen, Zhongxi Fang, Hiroyuki Kasai  

**Link**: [PDF](https://arxiv.org/pdf/2508.12361)  

**Abstract**: Inference-time scaling has achieved remarkable success in language models, yet its adaptation to diffusion models remains underexplored. We observe that the efficacy of recent Sequential Monte Carlo (SMC)-based methods largely stems from globally fitting the The reward-tilted distribution, which inherently preserves diversity during multi-modal search. However, current applications of SMC to diffusion models face a fundamental dilemma: early-stage noise samples offer high potential for improvement but are difficult to evaluate accurately, whereas late-stage samples can be reliably assessed but are largely irreversible. To address this exploration-exploitation trade-off, we approach the problem from the perspective of the search algorithm and propose two strategies: Funnel Schedule and Adaptive Temperature. These simple yet effective methods are tailored to the unique generation dynamics and phase-transition behavior of diffusion models. By progressively reducing the number of maintained particles and down-weighting the influence of early-stage rewards, our methods significantly enhance sample quality without increasing the total number of Noise Function Evaluations. Experimental results on multiple benchmarks and state-of-the-art text-to-image diffusion models demonstrate that our approach outperforms previous baselines. 

**Abstract (ZH)**: 推理时的缩放在语言模型中取得了显著成功，但其在扩散模型中的适应性仍待探索。我们观察到，最近基于Sequential Monte Carlo (SMC)的方法的有效性主要源于其对奖励倾斜分布进行全局拟合，这在多模态搜索过程中自然地保留了多样性。然而，当前SMC方法在扩散模型中的应用面临着根本性的困境：早期噪声样本具有较高的改进潜力但难以准确评估，而晚期样本可以可靠地评估但几乎不可逆。为解决这一探索与利用的权衡问题，我们从搜索算法的角度出发，提出了两种策略：漏斗调度和自适应温度。这两种简单而有效的方法针对扩散模型独特的生成动态和相变行为进行了定制。通过逐步减少维护的粒子数量并降低早期奖励的影响，我们的方法在不增加Noise Function Evaluations总数的情况下显著提高了样本质量。在多个基准测试和最先进的文本到图像扩散模型上的实验结果表明，我们的方法优于之前的方法。 

---
# Uncovering Systematic Failures of LLMs in Verifying Code Against Natural Language Specifications 

**Title (ZH)**: 探究LLMs在验证代码符合自然语言规范方面系统的失败风险lógica 

**Authors**: Haolin Jin, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.12358)  

**Abstract**: Large language models (LLMs) have become essential tools in software development, widely used for requirements engineering, code generation and review tasks. Software engineers often rely on LLMs to assess whether system code implementation satisfy task requirements, thereby enhancing code robustness and accuracy. However, it remains unclear whether LLMs can reliably determine whether the code complies fully with the given task descriptions, which is usually natural language specifications. In this paper, we uncover a systematic failure of LLMs in evaluating whether code aligns with natural language requirements. Specifically, with widely used benchmarks, we employ unified prompts to judge code correctness. Our results reveal that LLMs frequently misclassify correct code implementations as either ``not satisfying requirements'' or containing potential defects. Surprisingly, more complex prompting, especially when leveraging prompt engineering techniques involving explanations and proposed corrections, leads to higher misjudgment rate, which highlights the critical reliability issues in using LLMs as code review assistants. We further analyze the root causes of these misjudgments, and propose two improved prompting strategies for mitigation. For the first time, our findings reveals unrecognized limitations in LLMs to match code with requirements. We also offer novel insights and practical guidance for effective use of LLMs in automated code review and task-oriented agent scenarios. 

**Abstract (ZH)**: 大型语言模型在评估代码与自然语言要求是否一致时存在系统性失败：统一提示下的误判分析及改进策略 

---
# Synthetic Data is Sufficient for Zero-Shot Visual Generalization from Offline Data 

**Title (ZH)**: 合成数据足以从离线数据实现零样本视觉泛化 

**Authors**: Ahmet H. Güzel, Ilija Bogunovic, Jack Parker-Holder  

**Link**: [PDF](https://arxiv.org/pdf/2508.12356)  

**Abstract**: Offline reinforcement learning (RL) offers a promising framework for training agents using pre-collected datasets without the need for further environment interaction. However, policies trained on offline data often struggle to generalise due to limited exposure to diverse states. The complexity of visual data introduces additional challenges such as noise, distractions, and spurious correlations, which can misguide the policy and increase the risk of overfitting if the training data is not sufficiently diverse. Indeed, this makes it challenging to leverage vision-based offline data in training robust agents that can generalize to unseen environments. To solve this problem, we propose a simple approach generating additional synthetic training data. We propose a two-step process, first augmenting the originally collected offline data to improve zero-shot generalization by introducing diversity, then using a diffusion model to generate additional data in latent space. We test our method across both continuous action spaces (Visual D4RL) and discrete action spaces (Procgen), demonstrating that it significantly improves generalization without requiring any algorithmic changes to existing model-free offline RL methods. We show that our method not only increases the diversity of the training data but also significantly reduces the generalization gap at test time while maintaining computational efficiency. We believe this approach could fuel additional progress in generating synthetic data to train more general agents in the future. 

**Abstract (ZH)**: 离线强化学习（RL）为使用预先收集的数据训练代理并在无需进一步环境交互的情况下提供了有前景的框架。然而，基于离线数据训练的策略往往难以泛化，因为它们对多样性的状态暴露有限。视觉数据的复杂性引入了额外的挑战，如噪声、干扰和虚假相关性，这些因素可能会误导策略并增加过拟合的风险，特别是当训练数据不够多样化时。实际上，这使得利用基于视觉的离线数据训练能够在未见过的环境中泛化的强代理变得极具挑战性。为了解决这一问题，我们提出了一种简单的生成额外合成训练数据的方法。我们提出了一种两步过程，首先通过引入多样性来增强最初收集的离线数据，以改善零样本泛化，然后使用扩散模型在潜在空间中生成额外的数据。我们在连续动作空间（Visual D4RL）和离散动作空间（Procgen）上测试了我们的方法，证明它可以显著改善泛化，而无需对现有的无模型离线RL方法进行任何算法更改。我们显示，该方法不仅增加了训练数据的多样性，还在测试时显著减少了泛化差距，同时保持了计算效率。我们认为，这一方法有望在未来促进生成合成数据以训练更具泛化能力代理的进一步进展。 

---
# A Large-Scale Web Search Dataset for Federated Online Learning to Rank 

**Title (ZH)**: 适用于 federated online learning to rank 的大规模网页搜索数据集 

**Authors**: Marcel Gregoriadis, Jingwei Kang, Johan Pouwelse  

**Link**: [PDF](https://arxiv.org/pdf/2508.12353)  

**Abstract**: The centralized collection of search interaction logs for training ranking models raises significant privacy concerns. Federated Online Learning to Rank (FOLTR) offers a privacy-preserving alternative by enabling collaborative model training without sharing raw user data. However, benchmarks in FOLTR are largely based on random partitioning of classical learning-to-rank datasets, simulated user clicks, and the assumption of synchronous client participation. This oversimplifies real-world dynamics and undermines the realism of experimental results. We present AOL4FOLTR, a large-scale web search dataset with 2.6 million queries from 10,000 users. Our dataset addresses key limitations of existing benchmarks by including user identifiers, real click data, and query timestamps, enabling realistic user partitioning, behavior modeling, and asynchronous federated learning scenarios. 

**Abstract (ZH)**: AOL4FOLTR：一种包含260万查询的大型 web 搜索数据集，用于联邦在线排名学习 

---
# Semantic Discrepancy-aware Detector for Image Forgery Identification 

**Title (ZH)**: 语义不一致性感知检测器用于图像伪造识别 

**Authors**: Ziye Wang, Minghang Yu, Chunyan Xu, Zhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2508.12341)  

**Abstract**: With the rapid advancement of image generation techniques, robust forgery detection has become increasingly imperative to ensure the trustworthiness of digital media. Recent research indicates that the learned semantic concepts of pre-trained models are critical for identifying fake images. However, the misalignment between the forgery and semantic concept spaces hinders the model's forgery detection performance. To address this problem, we propose a novel Semantic Discrepancy-aware Detector (SDD) that leverages reconstruction learning to align the two spaces at a fine-grained visual level. By exploiting the conceptual knowledge embedded in the pre-trained vision language model, we specifically design a semantic token sampling module to mitigate the space shifts caused by features irrelevant to both forgery traces and semantic concepts. A concept-level forgery discrepancy learning module, built upon a visual reconstruction paradigm, is proposed to strengthen the interaction between visual semantic concepts and forgery traces, effectively capturing discrepancies under the concepts' guidance. Finally, the low-level forgery feature enhancemer integrates the learned concept level forgery discrepancies to minimize redundant forgery information. Experiments conducted on two standard image forgery datasets demonstrate the efficacy of the proposed SDD, which achieves superior results compared to existing methods. The code is available at this https URL. 

**Abstract (ZH)**: 随着图像生成技术的迅速发展，稳健的伪造检测已成为确保数字媒体可信性的日益重要任务。近期研究表明，预训练模型学习到的语义概念对于识别伪造图像至关重要。然而，伪造和语义概念空间之间的不一致阻碍了模型的伪造检测性能。为了解决这一问题，我们提出了一种新型的语义差异感知检测器（SDD），利用重建学习在细微视觉层面上对齐两个空间。通过利用预训练的视觉语言模型中嵌入的概念知识，我们特别设计了一个语义令牌采样模块，以缓解由与伪造痕迹和语义概念无关的特征引起的空间偏移。基于视觉重建范式的概念级伪造差异学习模块被提出，以加强视觉语义概念与伪造痕迹之间的交互，在概念的引导下有效捕捉差异。最后，低级别伪造特征增强模块整合了学习到的概念级伪造差异，以最小化冗余的伪造信息。实验结果显示，所提出的SDD在两个标准图像伪造数据集上具有优越性，其性能优于现有方法。代码可在此处获得。 

---
# Synchronization Dynamics of Heterogeneous, Collaborative Multi-Agent AI Systems 

**Title (ZH)**: 异质协作多智能体AI系统同步动力学 

**Authors**: Chiranjit Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2508.12314)  

**Abstract**: We present a novel interdisciplinary framework that bridges synchronization theory and multi-agent AI systems by adapting the Kuramoto model to describe the collective dynamics of heterogeneous AI agents engaged in complex task execution. By representing AI agents as coupled oscillators with both phase and amplitude dynamics, our model captures essential aspects of agent specialization, influence, and communication within networked systems. We introduce an order parameter to quantify the degree of coordination and synchronization, providing insights into how coupling strength, agent diversity, and network topology impact emergent collective behavior. Furthermore, we formalize a detailed correspondence between Chain-of-Thought prompting in AI reasoning and synchronization phenomena, unifying human-like iterative problem solving with emergent group intelligence. Through extensive simulations on all-to-all and deterministic scale-free networks, we demonstrate that increased coupling promotes robust synchronization despite heterogeneous agent capabilities, reflecting realistic collaborative AI scenarios. Our physics-informed approach establishes a rigorous mathematical foundation for designing, analyzing, and optimizing scalable, adaptive, and interpretable multi-agent AI systems. This work opens pathways for principled orchestration of agentic AI and lays the groundwork for future incorporation of learning dynamics and adaptive network architectures to further enhance system resilience and efficiency. 

**Abstract (ZH)**: 我们提出了一种综合交叉学科框架，通过将库拉莫托模型适应于描述参与复杂任务执行的异构AI代理的集体动力学，连接同步理论与多代理AI系统。通过将AI代理表示为具有相位和振幅动力学的耦合振子，我们的模型捕获了代理专业化、影响和网络系统中通信的关键方面。我们引入一个序参量来量化协调和同步的程度，提供关于耦合强度、代理多样性和网络拓扑如何影响涌现集体行为的见解。此外，我们将AI推理中的思考链提示与同步现象正式化对应起来，将类人的迭代问题解决与涌现的集体智能统一起来。通过在全连接和确定性无标度网络上进行广泛的模拟，我们证明了增加耦合在代理异质能力的情况下仍能促进稳健的同步，反映了现实的协作AI场景。我们的基于物理的方法为设计、分析和优化可扩展、自适应和可解释的多代理AI系统奠定了严格的数学基础。这项工作为原理性的 orchestration 智能代理AI提供了途径，并为未来整合学习动力学和自适应网络架构以进一步增强系统鲁棒性和效率奠定了基础。 

---
# Mutually Assured Deregulation 

**Title (ZH)**: 相互保证的监管放松 

**Authors**: Gilad Abiri  

**Link**: [PDF](https://arxiv.org/pdf/2508.12300)  

**Abstract**: We have convinced ourselves that the way to make AI safe is to make it unsafe. Since 2022, policymakers worldwide have embraced the Regulation Sacrifice - the belief that dismantling safety oversight will deliver security through AI dominance. Fearing China or USA will gain advantage, nations rush to eliminate safeguards that might slow progress. This Essay reveals the fatal flaw: though AI poses national security challenges, the solution demands stronger regulatory frameworks, not weaker ones. A race without guardrails breeds shared danger, not competitive strength. The Regulation Sacrifice makes three false promises. First, it promises durable technological leads. But AI capabilities spread rapidly - performance gaps between U.S. and Chinese systems collapsed from 9 percent to 2 percent in thirteen months. When advantages evaporate in months, sacrificing permanent safety for temporary speed makes no sense. Second, it promises deregulation accelerates innovation. The opposite often proves true. Companies report well-designed governance streamlines development. Investment flows toward regulated markets. Clear rules reduce uncertainty; uncertain liability creates paralysis. Environmental standards did not kill the auto industry; they created Tesla and BYD. Third, enhanced national security through deregulation actually undermines security across all timeframes. Near term: it hands adversaries information warfare tools. Medium term: it democratizes bioweapon capabilities. Long term: it guarantees deployment of uncontrollable AGI systems. The Regulation Sacrifice persists because it serves powerful interests, not security. Tech companies prefer freedom to accountability. Politicians prefer simple stories to complex truths. This creates mutually assured deregulation, where each nation's sprint for advantage guarantees collective vulnerability. The only way to win is not to play. 

**Abstract (ZH)**: 标题翻译:

我们坚信确保 AI 宯全的关键在于增强 加强监管而不是放松监管。全球

具体内容翻译如下:

我们 have convinced ourselves that the way to ensure AI safe whole safe to ensure it unsafe. the way policymakers worldwide have embraced the Regulation Sacrifice - the belief that dismantling safety safety oversight will delivers security. AI dominance. the USA and other major nations rush to eliminate safeguards that might might may might progress. This essay reveals the fatal flaw: in AI way poses national security challenges the solution demands stronger regulatory frameworks. way cautiously one way way way caution breeds that danger one way competitive strength. The Regulation sacrifice perpetuates three false false promises. First, the way promises durable technological leads ways. But But way way capabilities gaps betweenpace China and American AI systemsst systems collapse from a half to way percent in thirteen months. When way when advantages evaporate in in months makes sacrificing permanent consistency for temporary velocity way makes no sense. way the way promise promise deregulation accelerates innovation. The opposite is true. way companies with well-designed governance mechanisms foster development development. investment investment flows towards regulated markets. Clear way determination diminishes uncertainty; unclear liability creates paralysis. Environmental standards do not harm the auto industries; they create Tesla and byd. Third, enhanced national security with deregulation undermines these way if time timeframe.near way near within hands adversariesD warfare. Medium term way democratize bioweapons capability. Large way the way way way way way persist due because strong interests, tensions between tech companies from freedom to accountability; politicians from simple stories to complex truths. This creates a mutually assured deregulation, way each nations sprint for advantages create way collective vulnerability. The only way to achieve is is way way. 

---
# HuBERT-VIC: Improving Noise-Robust Automatic Speech Recognition of Speech Foundation Model via Variance-Invariance-Covariance Regularization 

**Title (ZH)**: HuBERT-VIC：通过方差不变协方差正则化提高语音基础模型的噪声鲁棒自动语音识别 

**Authors**: Hyebin Ahn, Kangwook Jang, Hoirin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.12292)  

**Abstract**: Noise robustness in speech foundation models (SFMs) has been a critical challenge, as most models are primarily trained on clean data and experience performance degradation when the models are exposed to noisy speech. To address this issue, we propose HuBERT-VIC, a noise-robust SFM with variance, in-variance, and covariance regularization (VICReg) objectives. These objectives adjust the statistics of noisy speech representations, enabling the model to capture diverse acoustic characteristics and improving the generalization ability across different types of noise. When applied to HuBERT, our model shows relative performance improvements of 23.3% on LibriSpeech test-clean and 13.2% on test-other, compared to the baseline model pre-trained on noisy speech. 

**Abstract (ZH)**: 噪声鲁棒性在声学基础模型（SFMs）中的提升：HuBERT-VIC方法的研究 

---
# "My productivity is boosted, but ..." Demystifying Users' Perception on AI Coding Assistants 

**Title (ZH)**: “我的 productivity 得到了提升，但……”揭开用户对 AI 编码助手认知的迷思 

**Authors**: Yunbo Lyu, Zhou Yang, Jieke Shi, Jianming Chang, Yue Liu, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2508.12285)  

**Abstract**: This paper aims to explore fundamental questions in the era when AI coding assistants like GitHub Copilot are widely adopted: what do developers truly value and criticize in AI coding assistants, and what does this reveal about their needs and expectations in real-world software development? Unlike previous studies that conduct observational research in controlled and simulated environments, we analyze extensive, first-hand user reviews of AI coding assistants, which capture developers' authentic perspectives and experiences drawn directly from their actual day-to-day work contexts. We identify 1,085 AI coding assistants from the Visual Studio Code Marketplace. Although they only account for 1.64% of all extensions, we observe a surge in these assistants: over 90% of them are released within the past two years. We then manually analyze the user reviews sampled from 32 AI coding assistants that have sufficient installations and reviews to construct a comprehensive taxonomy of user concerns and feedback about these assistants. We manually annotate each review's attitude when mentioning certain aspects of coding assistants, yielding nuanced insights into user satisfaction and dissatisfaction regarding specific features, concerns, and overall tool performance. Built on top of the findings-including how users demand not just intelligent suggestions but also context-aware, customizable, and resource-efficient interactions-we propose five practical implications and suggestions to guide the enhancement of AI coding assistants that satisfy user needs. 

**Abstract (ZH)**: 本文旨在探索人工智能编码助手如GitHub Copilot广泛应用的时代，开发人员真正重视和批评的人工智能编码助手的哪些方面，这又揭示了他们在实际软件开发中有哪些需求和期望？不同于以往在受控和模拟环境中进行观察研究的方法，我们分析了大量第一手的人工智能编码助手用户评价，这些评价直接捕捉了开发人员的真实视角和体验，它们源自开发人员的实际日常工作环境。我们从Visual Studio Code Marketplace中识别出1,085个人工智能编码助手，尽管这些助手仅占所有扩展的1.64%，但它们中有超过90%发布于过去两年内。然后，我们手动分析了从32个人工智能编码助手中抽取的用户评价，这些助手有足够多的安装和评论，用于构建一个全面的用户关注点和反馈分类体系。我们手动标注每个评论对某些方面的人工智能编码助手的态度，从而对特定功能、问题及整体工具性能的用户满意度和不满提供了细致的见解。基于上述发现，包括用户不仅要求智能建议，还要求具有上下文意识、可定制和资源高效的人机交互，我们提出了五项实用的建议和指导，以促进能够满足用户需求的人工智能编码助手的改进。 

---
# TSLA: A Task-Specific Learning Adaptation for Semantic Segmentation on Autonomous Vehicles Platform 

**Title (ZH)**: TSLA：面向自主驾驶平台语义分割的任务特定学习适应 

**Authors**: Jun Liu, Zhenglun Kong, Pu Zhao, Weihao Zeng, Hao Tang, Xuan Shen, Changdi Yang, Wenbin Zhang, Geng Yuan, Wei Niu, Xue Lin, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12279)  

**Abstract**: Autonomous driving platforms encounter diverse driving scenarios, each with varying hardware resources and precision requirements. Given the computational limitations of embedded devices, it is crucial to consider computing costs when deploying on target platforms like the NVIDIA\textsuperscript{\textregistered} DRIVE PX 2. Our objective is to customize the semantic segmentation network according to the computing power and specific scenarios of autonomous driving hardware. We implement dynamic adaptability through a three-tier control mechanism -- width multiplier, classifier depth, and classifier kernel -- allowing fine-grained control over model components based on hardware constraints and task requirements. This adaptability facilitates broad model scaling, targeted refinement of the final layers, and scenario-specific optimization of kernel sizes, leading to improved resource allocation and performance.
Additionally, we leverage Bayesian Optimization with surrogate modeling to efficiently explore hyperparameter spaces under tight computational budgets. Our approach addresses scenario-specific and task-specific requirements through automatic parameter search, accommodating the unique computational complexity and accuracy needs of autonomous driving. It scales its Multiply-Accumulate Operations (MACs) for Task-Specific Learning Adaptation (TSLA), resulting in alternative configurations tailored to diverse self-driving tasks. These TSLA customizations maximize computational capacity and model accuracy, optimizing hardware utilization. 

**Abstract (ZH)**: 自主驾驶平台面临多样的驾驶场景，每个场景的硬件资源和精度要求各不相同。鉴于嵌入式设备的计算限制，在如NVIDIA® DRIVE PX 2等目标平台上部署时，考虑计算成本至关重要。我们的目标是根据自主驾驶硬件的计算能力及其特定场景定制语义分割网络。通过三层控制机制——宽度乘数、分类器深度和分类器核大小，实现细粒度控制，基于硬件约束和任务要求。这种适应性使得能够广泛调整模型规模、精炼最终层，并针对特定场景优化核大小，从而提高资源分配和性能。此外，我们利用贝叶斯优化和代理模型高效探索在紧苛计算预算下的超参数空间。我们的方法通过自动参数搜索解决特定场景和任务的具体需求，适应自主驾驶的独特计算复杂性和精确度需求。该方法针对任务特定学习适应性调整乘法累加操作（MACs），产生适应不同自动驾驶任务的替代配置，从而最大化计算能力和模型精度，优化硬件利用率。 

---
# CRoC: Context Refactoring Contrast for Graph Anomaly Detection with Limited Supervision 

**Title (ZH)**: CRoC: 基于上下文重构对比的图异常检测方法（在有限监督下） 

**Authors**: Siyue Xie, Da Sun Handason Tam, Wing Cheong Lau  

**Link**: [PDF](https://arxiv.org/pdf/2508.12278)  

**Abstract**: Graph Neural Networks (GNNs) are widely used as the engine for various graph-related tasks, with their effectiveness in analyzing graph-structured data. However, training robust GNNs often demands abundant labeled data, which is a critical bottleneck in real-world applications. This limitation severely impedes progress in Graph Anomaly Detection (GAD), where anomalies are inherently rare, costly to label, and may actively camouflage their patterns to evade detection. To address these problems, we propose Context Refactoring Contrast (CRoC), a simple yet effective framework that trains GNNs for GAD by jointly leveraging limited labeled and abundant unlabeled data. Different from previous works, CRoC exploits the class imbalance inherent in GAD to refactor the context of each node, which builds augmented graphs by recomposing the attributes of nodes while preserving their interaction patterns. Furthermore, CRoC encodes heterogeneous relations separately and integrates them into the message-passing process, enhancing the model's capacity to capture complex interaction semantics. These operations preserve node semantics while encouraging robustness to adversarial camouflage, enabling GNNs to uncover intricate anomalous cases. In the training stage, CRoC is further integrated with the contrastive learning paradigm. This allows GNNs to effectively harness unlabeled data during joint training, producing richer, more discriminative node embeddings. CRoC is evaluated on seven real-world GAD datasets with varying scales. Extensive experiments demonstrate that CRoC achieves up to 14% AUC improvement over baseline GNNs and outperforms state-of-the-art GAD methods under limited-label settings. 

**Abstract (ZH)**: 基于上下文重构对比的图神经网络异常检测方法（Context Refactoring Contrast for Graph Neural Network-based Anomaly Detection） 

---
# The Self-Execution Benchmark: Measuring LLMs' Attempts to Overcome Their Lack of Self-Execution 

**Title (ZH)**: 自我执行基准：衡量LLM们克服自我执行能力不足的尝试 

**Authors**: Elon Ezra, Ariel Weizman, Amos Azaria  

**Link**: [PDF](https://arxiv.org/pdf/2508.12277)  

**Abstract**: Large language models (LLMs) are commonly evaluated on tasks that test their knowledge or reasoning abilities. In this paper, we explore a different type of evaluation: whether an LLM can predict aspects of its own responses. Since LLMs lack the ability to execute themselves, we introduce the Self-Execution Benchmark, which measures a model's ability to anticipate properties of its output, such as whether a question will be difficult for it, whether it will refuse to answer, or what kinds of associations it is likely to produce. Our experiments show that models generally perform poorly on this benchmark, and that increased model size or capability does not consistently lead to better performance. These results suggest a fundamental limitation in how LLMs represent and reason about their own behavior. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通常被评估其知识或推理能力。在本文中，我们探讨了一种不同的评估方式：LLM能否预测其自身响应的某些方面。由于LLMs无法自我执行，我们引入了自我执行基准，该基准衡量模型预测其输出属性的能力，如问题是否对其困难、是否会拒绝回答或其可能产生的关联类型。我们的实验表明，模型在基准测试中的表现普遍较差，且模型规模或能力的增加并不始终导致性能提高。这些结果暗示了LLMs在表示和推理其自身行为方面的根本局限性。 

---
# Region-Level Context-Aware Multimodal Understanding 

**Title (ZH)**: 区域级上下文感知多模态理解 

**Authors**: Hongliang Wei, Xianqi Zhang, Xingtao Wang, Xiaopeng Fan, Debin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12263)  

**Abstract**: Despite significant progress, existing research on Multimodal Large Language Models (MLLMs) mainly focuses on general visual understanding, overlooking the ability to integrate textual context associated with objects for a more context-aware multimodal understanding -- an ability we refer to as Region-level Context-aware Multimodal Understanding (RCMU). To address this limitation, we first formulate the RCMU task, which requires models to respond to user instructions by integrating both image content and textual information of regions or objects. To equip MLLMs with RCMU capabilities, we propose Region-level Context-aware Visual Instruction Tuning (RCVIT), which incorporates object information into the model input and enables the model to utilize bounding box coordinates to effectively associate objects' visual content with their textual information. To address the lack of datasets, we introduce the RCMU dataset, a large-scale visual instruction tuning dataset that covers multiple RCMU tasks. We also propose RC\&P-Bench, a comprehensive benchmark that can evaluate the performance of MLLMs in RCMU and multimodal personalized understanding tasks. Additionally, we propose a reference-free evaluation metric to perform a comprehensive and fine-grained evaluation of the region-level context-aware image descriptions. By performing RCVIT on Qwen2-VL models with the RCMU dataset, we developed RC-Qwen2-VL models. Experimental results indicate that RC-Qwen2-VL models not only achieve outstanding performance on multiple RCMU tasks but also demonstrate successful applications in multimodal RAG and personalized conversation. Our data, model and benchmark are available at this https URL 

**Abstract (ZH)**: 尽管已经取得显著进步，现有关于多模态大型语言模型（MLLMs）的研究主要集中在通用视觉理解上，忽视了将与对象相关的文本上下文整合以实现更具上下文感知的多模态理解的能力——我们将其称作区域级别上下文感知多模态理解（RCMU）。为解决这一限制，我们首先定义了RCMU任务，要求模型通过整合图像内容和区域或对象的文本信息来响应用户指令。为了使MLLMs具备RCMU能力，我们提出了区域级别上下文感知视觉指令调优（RCVIT）方法，该方法将对象信息融入模型输入，并使模型能够利用边界框坐标有效关联对象的视觉内容与文本信息。为解决数据集不足的问题，我们引入了RCMU数据集，这是一个涵盖多种RCMU任务的大规模视觉指令调优数据集。我们还提出了RC\&P-Bench，这是一个全面的基准，可以评估MLLMs在RCMU和多模态个性化理解任务中的性能。此外，我们还提出了一种无需参考的评估指标，以进行全面和细致的区域级别上下文感知图像描述评价。通过对Qwen2-VL模型进行RCVIT训练并使用RCMU数据集，我们开发了RC-Qwen2-VL模型。实验结果显示，RC-Qwen2-VL模型不仅在多种RCMU任务中表现出色，还在多模态RAG和个性化对话中实现了成功的应用。我们的数据、模型和基准可以在此网页获取。 

---
# Fortifying the Agentic Web: A Unified Zero-Trust Architecture Against Logic-layer Threats 

**Title (ZH)**: 加强代理网络：针对逻辑层威胁的统一零信任架构 

**Authors**: Ken Huang, Yasir Mehmood, Hammad Atta, Jerry Huang, Muhammad Zeeshan Baig, Sree Bhargavi Balija  

**Link**: [PDF](https://arxiv.org/pdf/2508.12259)  

**Abstract**: This paper presents a Unified Security Architecture that fortifies the Agentic Web through a Zero-Trust IAM framework. This architecture is built on a foundation of rich, verifiable agent identities using Decentralized Identifiers (DIDs) and Verifiable Credentials (VCs), with discovery managed by a protocol-agnostic Agent Name Service (ANS). Security is operationalized through a multi-layered Trust Fabric which introduces significant innovations, including Trust-Adaptive Runtime Environments (TARE), Causal Chain Auditing, and Dynamic Identity with Behavioral Attestation. By explicitly linking the LPCI threat to these enhanced architectural countermeasures within a formal security model, we propose a comprehensive and forward-looking blueprint for a secure, resilient, and trustworthy agentic ecosystem. Our formal analysis demonstrates that the proposed architecture provides provable security guarantees against LPCI attacks with bounded probability of success. 

**Abstract (ZH)**: 本文提出了一种统一安全架构，通过零信任IAM框架强化代理网络。该架构基于丰富的可验证代理身份构建，使用分布式标识符（DIDs）和可验证凭证（VCs），并通过协议无关的代理名称服务（ANS）进行发现。安全性通过多层信任织物实现，引入了包括信任自适应运行时环境（TARE）、因果链审计和动态身份行为证明在内的重大创新。通过对LPCI威胁与这些增强架构对策之间的显式链接，在形式安全模型中提出了一个全面且前瞻性的安全、韧性和可信赖代理生态系统蓝图。我们的形式分析表明，所提出的架构能够在有界成功概率下提供对LPCI攻击的可验证安全保证。 

---
# Interpreting Time Series Forecasts with LIME and SHAP: A Case Study on the Air Passengers Dataset 

**Title (ZH)**: 使用LIME和SHAP解释时间序列预测：基于航空乘客数据集的案例研究 

**Authors**: Manish Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2508.12253)  

**Abstract**: Time-series forecasting underpins critical decisions across aviation, energy, retail and health. Classical autoregressive integrated moving average (ARIMA) models offer interpretability via coefficients but struggle with nonlinearities, whereas tree-based machine-learning models such as XGBoost deliver high accuracy but are often opaque. This paper presents a unified framework for interpreting time-series forecasts using local interpretable model-agnostic explanations (LIME) and SHapley additive exPlanations (SHAP). We convert a univariate series into a leakage-free supervised learning problem, train a gradient-boosted tree alongside an ARIMA baseline and apply post-hoc explainability. Using the Air Passengers dataset as a case study, we show that a small set of lagged features -- particularly the twelve-month lag -- and seasonal encodings explain most forecast variance. We contribute: (i) a methodology for applying LIME and SHAP to time series without violating chronology; (ii) theoretical exposition of the underlying algorithms; (iii) empirical evaluation with extensive analysis; and (iv) guidelines for practitioners. 

**Abstract (ZH)**: 时间序列预测贯穿航空、零售和健康等领域的关键控制。本文提出了一种结合局部可解释性表方自洽解释（LIME）和SHap值可加性解释（SHAP pesticure的时间序列预测解释框架，传统自回归整定移动平均（ARIMA）模型通过系数提供可易性性，但对非线性性弱能力差；而基于线模型如如如XGBoost线模型则提供高准度性，但往往缺乏透明性。本文提出了一种结合LIME和SHAP的统 blir机械架框架，用于解释时间序列预测，并己酒下不违反时序性性。我们使用国际空旅乘客数据集进行了案例研究，表明滞后特征（尤其是十二个月滞后）和季节编码能够有效解释预测变异。我们贡献了：(一项结合LIME和SHAP应用于时间序列预测的方法论；(ii)底层算法的理论阐述；(iii)严格的实证分析；(iv)实践者的指导建议。 

---
# STM3: Mixture of Multiscale Mamba for Long-Term Spatio-Temporal Time-Series Prediction 

**Title (ZH)**: STM3：多尺度Mamba混合模型的长时空间-时间序列预测 

**Authors**: Haolong Chen, Liang Zhang, Zhengyuan Xin, Guangxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12247)  

**Abstract**: Recently, spatio-temporal time-series prediction has developed rapidly, yet existing deep learning methods struggle with learning complex long-term spatio-temporal dependencies efficiently. The long-term spatio-temporal dependency learning brings two new challenges: 1) The long-term temporal sequence includes multiscale information naturally which is hard to extract efficiently; 2) The multiscale temporal information from different nodes is highly correlated and hard to model. To address these challenges, we propose an efficient \textit{\textbf{S}patio-\textbf{T}emporal \textbf{M}ultiscale \textbf{M}amba} (STM2) that includes a multiscale Mamba architecture to capture the multiscale information efficiently and simultaneously, and an adaptive graph causal convolution network to learn the complex multiscale spatio-temporal dependency. STM2 includes hierarchical information aggregation for different-scale information that guarantees their distinguishability. To capture diverse temporal dynamics across all spatial nodes more efficiently, we further propose an enhanced version termed \textit{\textbf{S}patio-\textbf{T}emporal \textbf{M}ixture of \textbf{M}ultiscale \textbf{M}amba} (STM3) that employs a special Mixture-of-Experts architecture, including a more stable routing strategy and a causal contrastive learning strategy to enhance the scale distinguishability. We prove that STM3 has much better routing smoothness and guarantees the pattern disentanglement for each expert successfully. Extensive experiments on real-world benchmarks demonstrate STM2/STM3's superior performance, achieving state-of-the-art results in long-term spatio-temporal time-series prediction. 

**Abstract (ZH)**: 高效的时空多尺度Mamba（STM2/STM3）方法及其在长期时空时间序列预测中的应用 

---
# LinkAnchor: An Autonomous LLM-Based Agent for Issue-to-Commit Link Recovery 

**Title (ZH)**: LinkAnchor：一个基于自主LLM的Issue-to-Commit链接恢复代理 

**Authors**: Arshia Akhavan, Alireza Hosseinpour, Abbas Heydarnoori, Mehdi Keshani  

**Link**: [PDF](https://arxiv.org/pdf/2508.12232)  

**Abstract**: Issue-to-commit link recovery plays an important role in software traceability and improves project management. However, it remains a challenging task. A study on GitHub shows that only 42.2% of the issues are correctly linked to their commits. This highlights the potential for further development and research in this area. Existing studies have employed various AI/ML-based approaches, and with the recent development of large language models, researchers have leveraged LLMs to tackle this problem. These approaches suffer from two main issues. First, LLMs are constrained by limited context windows and cannot ingest all of the available data sources, such as long commit histories, extensive issue comments, and large code repositories. Second, most methods operate on individual issue-commit pairs; that is, given a single issue-commit pair, they determine whether the commit resolves the issue. This quickly becomes impractical in real-world repositories containing tens of thousands of commits. To address these limitations, we present LinkAnchor, the first autonomous LLM-based agent designed for issue-to-commit link recovery. The lazy-access architecture of LinkAnchor enables the underlying LLM to access the rich context of software, spanning commits, issue comments, and code files, without exceeding the token limit by dynamically retrieving only the most relevant contextual data. Additionally, LinkAnchor is able to automatically pinpoint the target commit rather than exhaustively scoring every possible candidate. Our evaluations show that LinkAnchor outperforms state-of-the-art issue-to-commit link recovery approaches by 60-262% in Hit@1 score across all our case study projects. We also publicly release LinkAnchor as a ready-to-use tool, along with our replication package. LinkAnchor is designed and tested for GitHub and Jira, and is easily extendable to other platforms. 

**Abstract (ZH)**: 基于LLM的Issue到Commit链接恢复方法：LinkAnchor的研究 

---
# Distribution Matching via Generalized Consistency Models 

**Title (ZH)**: 广义一致性模型下的分布匹配 

**Authors**: Sagar Shrestha, Rajesh Shrestha, Tri Nguyen, Subash Timilsina  

**Link**: [PDF](https://arxiv.org/pdf/2508.12222)  

**Abstract**: Recent advancement in generative models have demonstrated remarkable performance across various data modalities. Beyond their typical use in data synthesis, these models play a crucial role in distribution matching tasks such as latent variable modeling, domain translation, and domain adaptation. Generative Adversarial Networks (GANs) have emerged as the preferred method of distribution matching due to their efficacy in handling high-dimensional data and their flexibility in accommodating various constraints. However, GANs often encounter challenge in training due to their bi-level min-max optimization objective and susceptibility to mode collapse. In this work, we propose a novel approach for distribution matching inspired by the consistency models employed in Continuous Normalizing Flow (CNF). Our model inherits the advantages of CNF models, such as having a straight forward norm minimization objective, while remaining adaptable to different constraints similar to GANs. We provide theoretical validation of our proposed objective and demonstrate its performance through experiments on synthetic and real-world datasets. 

**Abstract (ZH)**: Recent Advances in Generative Models for Distribution Matching：A Consistency-Based Approach Inspired by Continuous Normalizing Flow 

---
# Unlearning at Scale: Implementing the Right to be Forgotten in Large Language Models 

**Title (ZH)**: 大规模遗忘：在大型语言模型中实现被遗忘的权利 

**Authors**: Abdullah X  

**Link**: [PDF](https://arxiv.org/pdf/2508.12220)  

**Abstract**: We study the right to be forgotten (GDPR Art. 17) for large language models and frame unlearning as a reproducible systems problem. Our approach treats training as a deterministic program and logs a minimal per-microbatch record (ordered ID hash, RNG seed, learning-rate value, optimizer-step counter, and accumulation boundary). Under a pinned stack and deterministic kernels, replaying the training tail while filtering only the forget closure yields the same parameters as training on the retain set (bit-identical in the training dtype) when preconditions hold. To meet latency and availability constraints, we add complementary paths: (i) exact reverts of recent steps via micro-checkpoints or dense per-step deltas, (ii) cohort-scoped adapter deletion when the base is frozen, and (iii) a curvature-guided anti-update followed by a short retain-tune, audit-gated with escalation to exact replay. We report storage/latency budgets and a toy artifact validating mechanics; in a controlled run that satisfies the preconditions we demonstrate byte-identical equality of model and optimizer states. 

**Abstract (ZH)**: 我们研究大型语言模型的被遗忘权利（GDPR Art. 17）并将遗忘问题框架化为可重复的系统问题。我们的方法将训练视为确定性程序，并记录最小的每次微批次日志（按顺序的ID哈希、随机数生成器种子、学习率值、优化器步骤计数器和累积边界）。在固定栈和确定性内核下，当先决条件满足时，重新播放训练尾部并仅过滤忘记闭包会与保留集上训练得到相同的参数（在训练数据类型中是位级别的相同）。为了满足延迟和可用性约束，我们增加了一些补充路径：（i）通过微检查点或密集的每步差异进行最近步骤的精确还原；（ii）当基础模型冻结时针对群体的适配器删除；（iii）基于曲率的反更新，随后是短暂的保留调优，并通过审计控制提升到精确重演。我们报告了存储/延迟预算，并验证了一个玩具示例以验证机制；在一个受控运行中，满足了先决条件后，模型和优化器状态在字节级别上是等价的。 

---
# Towards Generalizable Human Activity Recognition: A Survey 

**Title (ZH)**: 面向泛化的行人活动识别：一种综述 

**Authors**: Yize Cai, Baoshen Guo, Flora Salim, Zhiqing Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.12213)  

**Abstract**: As a critical component of Wearable AI, IMU-based Human Activity Recognition (HAR) has attracted increasing attention from both academia and industry in recent years. Although HAR performance has improved considerably in specific scenarios, its generalization capability remains a key barrier to widespread real-world adoption. For example, domain shifts caused by variations in users, sensor positions, or environments can significantly decrease the performance in practice. As a result, in this survey, we explore the rapidly evolving field of IMU-based generalizable HAR, reviewing 229 research papers alongside 25 publicly available datasets to provide a broad and insightful overview. We first present the background and overall framework of IMU-based HAR tasks, as well as the generalization-oriented training settings. Then, we categorize representative methodologies from two perspectives: (i) model-centric approaches, including pre-training method, end-to-end method, and large language model (LLM)-based learning method; and (ii) data-centric approaches, including multi-modal learning and data augmentation techniques. In addition, we summarize widely used datasets in this field, as well as relevant tools and benchmarks. Building on these methodological advances, the broad applicability of IMU-based HAR is also reviewed and discussed. Finally, we discuss persistent challenges (e.g., data scarcity, efficient training, and reliable evaluation) and also outline future directions for HAR, including the adoption of foundation and large language models, physics-informed and context-aware reasoning, generative modeling, and resource-efficient training and inference. The complete list of this survey is available at this https URL, which will be updated continuously. 

**Abstract (ZH)**: 基于IMU的人体活动识别：可泛化的挑战与方法综述 

---
# ProtTeX-CC: Activating In-Context Learning in Protein LLM via Two-Stage Instruction Compression 

**Title (ZH)**: ProtTeX-CC: 通过两阶段指令压缩激活蛋白质LLM的上下文学习 

**Authors**: Chuanliu Fan, Zicheng Ma, Jun Gao, Nan Yu, Jun Zhang, Ziqiang Cao, Yi Qin Gao, Guohong Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12212)  

**Abstract**: Recent advances in protein large language models, such as ProtTeX, represent both side-chain amino acids and backbone structure as discrete token sequences of residue length. While this design enables unified modeling of multimodal protein information, it suffers from two major limitations: (1) The concatenation of sequence and structure tokens approximately doubles the protein length and breaks the intrinsic residue-level alignment between modalities. (2) Constrained by the training corpus and limited context window, ProtTeX is typically trained on single-protein inputs, rendering it incompatible with in-context learning (ICL) and thus limiting its generalization capability. To address these issues, we propose ProtTeX-CC, a lightweight two-stage compression framework designed to enhance ProtTeX under few-shot settings. We first design a joint embedding compression mechanism that fuses sequence and structure representations at the residue level, effectively reducing the protein input length by half without sacrificing performance. Then we propose a self-compression module that aggregates each full demonstration into the latent space of the last few linguistic tokens, reducing the average demonstration length from 751 tokens to less than 16 tokens. Compared to the original ProtTeX, our self-compression approach achieves a compression ratio of approximately 93.68% in the total prompt length under the 16-shot setting. Without modifying the backbone model, ProtTeX-CC introduces only a small number of additional parameters through PEFT-based tuning in the joint embedding compression stage and a single trainable projection layer in the self-compression stage. Extensive experiments on protein function prediction show that ProtTeX-CC improves performance on the in-domain benchmark by 2%, and generalizes well to the out-of-domain dataset with a performance gain of 11%. 

**Abstract (ZH)**: 近期蛋白质大型语言模型的进步，如ProtTX，将侧链氨基酸和 backbone 结构表示为残基数长度的离散标记序列。虽然这种设计能够统一建模多模态蛋白质信息，但存在两个主要局限性：（1）序列和结构标记的串联大约使蛋白质长度翻倍，并打破了模态间的残基级内固有的对齐；（2）受限于训练语料和有限的上下文窗口，ProtTX通常仅针对单蛋白质输入进行训练，使其不兼容上下文学习（ICL），从而限制了其泛化能力。为解决这些问题，我们提出了一种轻量级的两阶段压缩框架ProtTX-CC，旨在在少量样本的情况下增强ProtTX。我们首先设计了一种残基级联合嵌入压缩机制，该机制在不牺牲性能的前提下，将蛋白质输入长度减少一半。然后，我们提出了一种自压缩模块，将每个完整示例汇聚到最后一两个语言标记的潜在空间中，将平均示例长度从751个标记降低到少于16个。与原始的ProtTX相比，在16-shot设置下，我们的自压缩方法在总提示长度上的压缩比约为93.68%。在不修改骨干模型的情况下，通过基于PEFT的调优引入少量额外参数完成联合嵌入压缩阶段，并在自压缩阶段引入一个可训练的投影层。在蛋白质功能预测的广泛实验中，ProtTX-CC在泛圈基准上的性能提高了2%，并在泛圈数据集上的性能提高了11%。 

---
# Improving Pre-Trained Vision-Language-Action Policies with Model-Based Search 

**Title (ZH)**: 基于模型搜索改进预训练的视觉-语言-动作策略 

**Authors**: Cyrus Neary, Omar G. Younis, Artur Kuramshin, Ozgur Aslan, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2508.12211)  

**Abstract**: Pre-trained vision-language-action (VLA) models offer a promising foundation for generalist robot policies, but often produce brittle behaviours or unsafe failures when deployed zero-shot in out-of-distribution scenarios. We present Vision-Language-Action Planning & Search (VLAPS) -- a novel framework and accompanying algorithms that embed model-based search into the inference procedure of pre-trained VLA policies to improve their performance on robotic tasks. Specifically, our method biases a modified Monte Carlo Tree Search (MCTS) algorithm -- run using a model of the target environment -- using action priors defined by the VLA policy. By using VLA-derived abstractions and priors in model-based search, VLAPS efficiently explores language-conditioned robotics tasks whose search spaces would otherwise be intractably large. Conversely, by integrating model-based search with the VLA policy's inference procedure, VLAPS yields behaviours that are more performant than those obtained by directly following the VLA policy's action predictions. VLAPS offers a principled framework to: i) control test-time compute in VLA models, ii) leverage a priori knowledge of the robotic environment, and iii) integrate established planning and reinforcement learning techniques into the VLA inference process. Across all experiments, VLAPS significantly outperforms VLA-only baselines on language-specified tasks that would otherwise be intractable for uninformed search algorithms, increasing success rates by as much as 67 percentage points. 

**Abstract (ZH)**: 预训练视觉-语言-动作（VLA）模型为通用机器人策略提供了一个有前途的基础，但在部署到分布外场景时往往会生成脆弱的行为或不安全的故障。我们提出了一种名为视觉-语言-动作规划与搜索（VLAPS）的新框架及其实现算法，将基于模型的搜索嵌入到预训练VLA策略的推理过程中，以提高其在机器人任务中的性能。具体而言，我们的方法使用目标环境模型运行修改后的蒙特卡洛树搜索（MCTS）算法，并利用VLA策略定义的动作先验进行偏置。通过在基于模型的搜索中使用VLA衍生的抽象和先验，VLAPS能有效探索由未受过训练的搜索算法难以处理的语言条件下的机器人任务。相反，通过将基于模型的搜索与VLA策略的推理过程集成，VLAPS产生的行为性能优于直接遵循VLA策略动作预测的行为。VLAPS提供了一个原则性的框架，用于i) 控制VLA模型的测试时计算，ii) 利用对机器人环境的先验知识，和iii) 将已建立的规划和强化学习技术整合到VLA推理过程中。在所有实验中，VLAPS在语言指定的任务中显著优于仅使用VLA的基础模型，对于未受过训练的搜索算法而言，成功率提高了多达67个百分点。 

---
# Exploring Multimodal AI Reasoning for Meteorological Forecasting from Skew-T Diagrams 

**Title (ZH)**: 探索 skew-T 图像中的多模态人工智能推理在气象预报中的应用 

**Authors**: ChangJae Lee, Heecheol Yang, Jonghak Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.12198)  

**Abstract**: Forecasting from atmospheric soundings is a fundamental task in operational meteorology, often requiring structured visual reasoning over Skew-T log-P diagrams by human forecasters. While recent advances in Vision-Language Models (VLMs) have shown promise in other scientific domains, their application to meteorological diagram interpretation remains largely unexplored. In this study, we present a lightweight AI assistant that interprets Skew-T diagrams using a small language model (LM) and a small VLM fine-tuned to emulate human forecasters. Using a curriculum learning framework, we first train the models to identify key atmospheric features from diagrams through visual question answering, followed by chain-of-thought reasoning tasks that estimate precipitation probability based on the derived visual groundings. Model inputs include either textual summaries or generated Skew-T diagrams derived from operational Numerical Weather Prediction (NWP) forecasts, paired with three-hour precipitation observations from South Korea's Auto Weather Stations network. Evaluation results demonstrate that the fine-tuned VLM achieves skill comparable to an operational NWP model, despite relying solely on static atmospheric profiles. Ablation studies reveal that visual grounding and reasoning supervision are critical for performance, while attention map analysis confirms that the model learns to focus on relevant meteorological features. These findings highlight the potential of compact, interpretable multimodal models to support weather forecasting tasks. The approach offers a computationally efficient alternative to large-scale systems, and future work could extend it to more complex applications. 

**Abstract (ZH)**: 从大气探空数据预测是气象业务中的一个基本任务，通常需要人类预报员进行结构化的视觉推理以分析斜压-对数压力图。虽然近期的视觉-语言模型（VLMs）在其他科学领域显示出了潜力，但其在气象图表解释的应用仍鲜有探索。在本文中，我们提出了一种轻量级AI助手，用于通过小型语言模型（LM）和细调的小型VLM来解释斜压-对数压力图，后者被设计以模仿人类预报员的思维过程。通过采用渐进学习框架，我们首先训练模型通过视觉问答识别图表中的关键大气特征，随后在基于提取的视觉基础进行连续推理任务中，估算降水概率。模型输入包括从数值天气预报（NWP）中生成的文本摘要或斜压-对数压力图，以及韩国自动气象站网络的三小时降水量观测数据。评估结果显示，即使仅依赖静态的大气剖面，细调的VLM也达到了与操作型NWP模型相当的技能水平。消融研究显示，视觉基础和推理监督对于性能至关重要，而注意力图分析证实，模型学会了集中于相关的气象特征。这些发现突显了紧凑且可解释的多模态模型在支持气象预测任务方面的潜力。该方法提供了相对于大规模系统更具计算效率的替代方案，未来的工作可以将其扩展到更复杂的应用。 

---
# Self-Guided Action Diffusion 

**Title (ZH)**: 自我引导的动作扩散 

**Authors**: Rhea Malhotra, Yuejiang Liu, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2508.12189)  

**Abstract**: Recent works have shown the promise of inference-time search over action samples for improving generative robot policies. In particular, optimizing cross-chunk coherence via bidirectional decoding has proven effective in boosting the consistency and reactivity of diffusion policies. However, this approach remains computationally expensive as the diversity of sampled actions grows. In this paper, we introduce self-guided action diffusion, a more efficient variant of bidirectional decoding tailored for diffusion-based policies. At the core of our method is to guide the proposal distribution at each diffusion step based on the prior decision. Experiments in simulation tasks show that the proposed self-guidance enables near-optimal performance at negligible inference cost. Notably, under a tight sampling budget, our method achieves up to 70% higher success rates than existing counterparts on challenging dynamic tasks. See project website at this https URL. 

**Abstract (ZH)**: 最近的研究表明，在生成机器人策略中，推理时搜索动作样本有改善生成性能的潜力。特别是，通过双向解码优化跨片段一致性已被证明有助于提高扩散策略的稳定性和反应性。然而，这种方法在采样动作多样性增加时仍具有较高的计算成本。在本文中，我们提出了一种更高效的双向解码变体——自我引导动作扩散，专门针对基于扩散的策略。我们的方法的核心是在每一步扩散过程中根据先验决策引导提案分布。在模拟任务中的实验显示，所提出的自我引导能够以几乎忽略不计的推理成本实现接近最优性能。值得注意的是，在严格的采样预算下，该方法在具有挑战性的动态任务中实现了比现有方法高出70%以上的成功率。更多信息请参见项目网站：见项目网站 

---
# RealTalk: Realistic Emotion-Aware Lifelike Talking-Head Synthesis 

**Title (ZH)**: RealTalk: 真实情感意识的逼真头部合成 

**Authors**: Wenqing Wang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12163)  

**Abstract**: Emotion is a critical component of artificial social intelligence. However, while current methods excel in lip synchronization and image quality, they often fail to generate accurate and controllable emotional expressions while preserving the subject's identity. To address this challenge, we introduce RealTalk, a novel framework for synthesizing emotional talking heads with high emotion accuracy, enhanced emotion controllability, and robust identity preservation. RealTalk employs a variational autoencoder (VAE) to generate 3D facial landmarks from driving audio, which are concatenated with emotion-label embeddings using a ResNet-based landmark deformation model (LDM) to produce emotional landmarks. These landmarks and facial blendshape coefficients jointly condition a novel tri-plane attention Neural Radiance Field (NeRF) to synthesize highly realistic emotional talking heads. Extensive experiments demonstrate that RealTalk outperforms existing methods in emotion accuracy, controllability, and identity preservation, advancing the development of socially intelligent AI systems. 

**Abstract (ZH)**: 情绪是人工社会智能的关键组成部分。然而，虽然现有方法在唇同步和图像质量方面表现出色，但在生成准确且可控的情绪表达并同时保留主体身份方面常常失败。为解决这一挑战，我们提出了RealTalk，这是一个用于合成高情绪准确度、增强情绪可控性和稳健身份保留的情感面部模型的新框架。RealTalk 使用变分自编码器 (VAE) 从驱动音频中生成 3D 面部关键点，然后使用基于 ResNet 的面部关键点变形模型 (LDM) 将情绪标签嵌入与其他关键点连接，生成带有情绪信息的关键点。这些关键点与面部混合形状系数一起条件作用于一种新颖的三平面注意力神经辐射场 (NeRF)，以合成极为逼真的情感面部模型。大量实验表明，RealTalk 在情绪准确度、可控性和身份保留方面优于现有方法，推动了社会智能AI系统的开发。 

---
# AICRN: Attention-Integrated Convolutional Residual Network for Interpretable Electrocardiogram Analysis 

**Title (ZH)**: 基于注意力集成卷积残差网络的可解释心电图分析 

**Authors**: J. M. I. H. Jayakody, A. M. H. H. Alahakoon, C. R. M. Perera, R. M. L. C. Srimal, Roshan Ragel, Vajira Thambawita, Isuru Nawinne  

**Link**: [PDF](https://arxiv.org/pdf/2508.12162)  

**Abstract**: The paradigm of electrocardiogram (ECG) analysis has evolved into real-time digital analysis, facilitated by artificial intelligence (AI) and machine learning (ML), which has improved the diagnostic precision and predictive capacity of cardiac diseases. This work proposes a novel deep learning (DL) architecture called the attention-integrated convolutional residual network (AICRN) to regress key ECG parameters such as the PR interval, the QT interval, the QRS duration, the heart rate, the peak amplitude of the R wave, and the amplitude of the T wave for interpretable ECG analysis. Our architecture is specially designed with spatial and channel attention-related mechanisms to address the type and spatial location of the ECG features for regression. The models employ a convolutional residual network to address vanishing and exploding gradient problems. The designed system addresses traditional analysis challenges, such as loss of focus due to human errors, and facilitates the fast and easy detection of cardiac events, thereby reducing the manual efforts required to solve analysis tasks. AICRN models outperform existing models in parameter regression with higher precision. This work demonstrates that DL can play a crucial role in the interpretability and precision of ECG analysis, opening up new clinical applications for cardiac monitoring and management. 

**Abstract (ZH)**: 基于注意力集成卷积残差网络的心电图参数回归分析：一种可解释的心电图分析新范式 

---
# Demystifying Foreground-Background Memorization in Diffusion Models 

**Title (ZH)**: 揭示扩散模型中前景-背景记忆的迷思 

**Authors**: Jimmy Z. Di, Yiwei Lu, Yaoliang Yu, Gautam Kamath, Adam Dziedzic, Franziska Boenisch  

**Link**: [PDF](https://arxiv.org/pdf/2508.12148)  

**Abstract**: Diffusion models (DMs) memorize training images and can reproduce near-duplicates during generation. Current detection methods identify verbatim memorization but fail to capture two critical aspects: quantifying partial memorization occurring in small image regions, and memorization patterns beyond specific prompt-image pairs. To address these limitations, we propose Foreground Background Memorization (FB-Mem), a novel segmentation-based metric that classifies and quantifies memorized regions within generated images. Our method reveals that memorization is more pervasive than previously understood: (1) individual generations from single prompts may be linked to clusters of similar training images, revealing complex memorization patterns that extend beyond one-to-one correspondences; and (2) existing model-level mitigation methods, such as neuron deactivation and pruning, fail to eliminate local memorization, which persists particularly in foreground regions. Our work establishes an effective framework for measuring memorization in diffusion models, demonstrates the inadequacy of current mitigation approaches, and proposes a stronger mitigation method using a clustering approach. 

**Abstract (ZH)**: 前景背景记忆化（FB-Mem）：一种新颖的分割基测量方法及扩散模型中的记忆分析 

---
# KP-INR: A Dual-Branch Implicit Neural Representation Model for Cardiac Cine MRI Reconstruction 

**Title (ZH)**: KP-INR：心脏 cine MRI 重建的双分支隐式神经表示模型 

**Authors**: Donghang Lyu, Marius Staring, Mariya Doneva, Hildo J. Lamb, Nicola Pezzotti  

**Link**: [PDF](https://arxiv.org/pdf/2508.12147)  

**Abstract**: Cardiac Magnetic Resonance (CMR) imaging is a non-invasive method for assessing cardiac structure, function, and blood flow. Cine MRI extends this by capturing heart motion, providing detailed insights into cardiac mechanics. To reduce scan time and breath-hold discomfort, fast acquisition techniques have been utilized at the cost of lowering image quality. Recently, Implicit Neural Representation (INR) methods have shown promise in unsupervised reconstruction by learning coordinate-to-value mappings from undersampled data, enabling high-quality image recovery. However, current existing INR methods primarily focus on using coordinate-based positional embeddings to learn the mapping, while overlooking the feature representations of the target point and its neighboring context. In this work, we propose KP-INR, a dual-branch INR method operating in k-space for cardiac cine MRI reconstruction: one branch processes the positional embedding of k-space coordinates, while the other learns from local multi-scale k-space feature representations at those coordinates. By enabling cross-branch interaction and approximating the target k-space values from both branches, KP-INR can achieve strong performance on challenging Cartesian k-space data. Experiments on the CMRxRecon2024 dataset confirms its improved performance over baseline models and highlights its potential in this field. 

**Abstract (ZH)**: 心脏磁共振成像（CMR）是一种无创方法，用于评估心脏结构、功能和血流。心脏电影MRI通过捕获心脏的运动，提供关于心脏机械性的详细见解。为了缩短扫描时间和减轻屏息不适，已经采用了快速采集技术，但代价是降低了图像质量。最近，显式神经表示（Explicit Neural Representation，ENR）方法在无监督重构中显示出潜力，通过从欠采样数据中学习坐标到值的映射来实现高质量图像恢复。然而，现有的ENR方法主要侧重于使用坐标基的位置嵌入来学习映射，而忽视了目标点及其邻域上下文的特征表示。在这种情况下，我们提出了KP-INR，这是一种在k空间中操作的双分支神经表示方法，用于心脏电影MRI重建：一个分支处理k空间坐标的 POSITIONAL EMBEDDING，另一个分支从这些坐标的局部多尺度k空间特征表示中学习。通过启用分支间的交互并从两个分支中近似目标k空间值，KP-INR可以在具有挑战性的笛卡尔k空间数据上实现优异性能。CMRxRecon2024数据集上的实验证实了其相对于基线模型的改进性能，并突显了其在该领域的潜在价值。 

---
# Substituting Proof of Work in Blockchain with Training-Verified Collaborative Model Computation 

**Title (ZH)**: 用训练验证协作模型计算代替区块链中的工作量证明 

**Authors**: Mohammad Ishzaz Asif Rafid, Morsalin Sakib  

**Link**: [PDF](https://arxiv.org/pdf/2508.12138)  

**Abstract**: Bitcoin's Proof of Work (PoW) mechanism, while central to achieving decentralized consensus, has long been criticized for excessive energy use and hardware inefficiencies \cite{devries2018bitcoin, truby2018decarbonizing}. This paper introduces a hybrid architecture that replaces Bitcoin's traditional PoW with a centralized, cloud-based collaborative training framework. In this model, miners contribute computing resources to train segments of horizontally scaled machine learning models on preprocessed datasets, ensuring privacy and generating meaningful outputs \cite{li2017securing}. A central server evaluates contributions using two metrics: number of parameters trained and reduction in model loss during each cycle. At the end of every cycle, a weighted lottery selects the winning miner, who receives a digitally signed certificate. This certificate serves as a verifiable substitute for PoW and grants the right to append a block to the blockchain \cite{nakamoto2008bitcoin}. By integrating digital signatures and SHA-256 hashing \cite{nist2015sha}, the system preserves blockchain integrity while redirecting energy toward productive computation. The proposed approach addresses the sustainability concerns of traditional mining by converting resource expenditure into socially valuable work, aligning security incentives with real-world computational progress. 

**Abstract (ZH)**: 比特币的工作量证明（PoW）机制虽然在实现去中心化共识方面至关重要，但长期以来因其能源消耗过大和硬件效率低下而受到批评。本文介绍了一种混合架构，该架构将比特币的传统PoW替换为基于云的集中式协作训练框架。在该模型中，矿工贡献计算资源在预处理数据集上训练横向扩展的机器学习模型的片段，确保隐私并生成有意义的输出。中央服务器使用两个指标评估贡献：训练的参数数量和每个周期内模型损失的减少。每个周期结束后，根据加权抽奖结果选出获胜矿工，其将获得数字签名证书。该证书作为PoW的可验证替代品，并赋予其向区块链追加区块的权利。通过整合数字签名和SHA-256哈希算法，该系统保持了区块链的完整性，同时将能源重新导向到有价值的计算工作。所提出的方法通过将资源支出转化为社会有价值的劳动，解决了传统采矿的可持续性问题，并使安全激励与实际计算进展相一致。 

---
# DynamixSFT: Dynamic Mixture Optimization of Instruction Tuning Collections 

**Title (ZH)**: DynamixSFT：指令调优集合的动态混合优化 

**Authors**: Haebin Shin, Lei Ji, Xiao Liu, Zhiwei Yu, Qi Chen, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2508.12116)  

**Abstract**: As numerous instruction-tuning datasets continue to emerge during the post-training stage, dynamically balancing and optimizing their mixtures has become a critical challenge. To address this, we propose DynamixSFT, a dynamic and automated method for instruction-tuning dataset mixture optimization. We formulate the problem as a multi-armed bandit setup and introduce a Prior-scaled Boltzmann Exploration that softly anchors the updated sampling distribution to the original dataset proportions, thereby preserving the inherent diversity and coverage of the collection. Sampling probabilities are updated using a lightweight 1-Step Look-ahead Reward, reflecting how much the dataset contributes to improving the model's performance at its current state. When applied to the Tulu-v2-mixture collection comprising 16 instruction-tuning datasets, DynamixSFT achieves up to a 2.2% performance improvement across 10 benchmarks. Furthermore, we provide a comprehensive analysis and visualizations to offer deeper insights into the adaptive dynamics of our method. 

**Abstract (ZH)**: 动态优化指令调整数据集混合的DynamixSFT方法 

---
# Simple o3: Towards Interleaved Vision-Language Reasoning 

**Title (ZH)**: 简单O3：交错视觉-语言推理owards Interleaved Vision-Language Reasoning 

**Authors**: Ye Wang, Qianglong Chen, Zejun Li, Siyuan Wang, Shijie Guo, Zhirui Zhang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2508.12109)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive performance on vision-language tasks, but their long Chain-of-Thought (CoT) capabilities in multimodal scenarios remain underexplored. Inspired by OpenAI's o3 model, which emulates human-like ''thinking with image'' through iterative visual transformations and linguistic reasoning, we propose Simple o3, an end-to-end framework that integrates dynamic tool interactions (e.g., cropping, zooming, and reusing) into interleaved vision-language reasoning via supervised fine-tuning (SFT). Our approach features a scalable data synthesis pipeline that generates high-quality interleaved vision-language reasoning chains via an ''observe-reason-act'' cycle, complete with executable visual operations and rigorous verification, yielding the open-source TWI-Tools-146K dataset. Experimental results demonstrate Simple o3's superior performance on diverse benchmarks, outperforming existing approaches. By combining enhanced reasoning capabilities, Simple o3 establishes a powerful yet computationally affordable paradigm for advancing multimodal reasoning. Remarkably, we provide the first in-depth analysis of different interleaved reasoning strategies, offering insights into their impact on model performance. We found that by introducing additional visual tokens for interleaved vision-language reasoning, reusing and magnifying the original image significantly improves the model's visual reasoning and fine-grained perception, while image cropping based on precise visual grounding allows the model to effectively focus on key entities or regions, further enhancing its capabilities. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉-语言任务上表现出色，但在多模态场景下的长链推理能力尚未充分探索。受到OpenAI的o3模型启发，该模型通过迭代的视觉变换和语言推理模仿人类“看图思考”的过程，我们提出了一种端到端框架Simple o3，该框架通过监督微调（SFT）将动态工具交互（如裁剪、缩放和重用）集成到交错的视觉-语言推理中。我们的方法具备可扩展的数据合成管道，通过“观察-推理-行动”循环生成高质量的交错视觉-语言推理链，包含可执行的视觉操作和严格的验证，并产生开源的TWI-Tools-146K数据集。实验结果表明，Simple o3在多种基准测试中的表现优于现有方法。通过增强的推理能力，Simple o3建立了强大且计算成本低廉的多模态推理范式。值得一提的是，我们提供了第一个对不同交错推理策略的深入分析，揭示了它们对模型性能的影响。我们发现，通过为交错视觉-语言推理引入额外的视觉标记，重新使用和放大原始图像显著提高了模型的视觉推理能力和细粒度感知能力，基于精确视觉定位的图像裁剪使模型能够有效地聚焦于关键实体或区域，进一步提升了其能力。 

---
# Generative Medical Event Models Improve with Scale 

**Title (ZH)**: 大规模训练生成医疗事件模型效果更佳 

**Authors**: Shane Waxler, Paul Blazek, Davis White, Daniel Sneider, Kevin Chung, Mani Nagarathnam, Patrick Williams, Hank Voeller, Karen Wong, Matthew Swanhorst, Sheng Zhang, Naoto Usuyama, Cliff Wong, Tristan Naumann, Hoifung Poon, Andrew Loza, Daniella Meeker, Seth Hain, Rahul Shah  

**Link**: [PDF](https://arxiv.org/pdf/2508.12104)  

**Abstract**: Realizing personalized medicine at scale calls for methods that distill insights from longitudinal patient journeys, which can be viewed as a sequence of medical events. Foundation models pretrained on large-scale medical event data represent a promising direction for scaling real-world evidence generation and generalizing to diverse downstream tasks. Using Epic Cosmos, a dataset with medical events from de-identified longitudinal health records for 16.3 billion encounters over 300 million unique patient records from 310 health systems, we introduce the Cosmos Medical Event Transformer ( CoMET) models, a family of decoder-only transformer models pretrained on 118 million patients representing 115 billion discrete medical events (151 billion tokens). We present the largest scaling-law study for medical event data, establishing a methodology for pretraining and revealing power-law scaling relationships for compute, tokens, and model size. Based on this, we pretrained a series of compute-optimal models with up to 1 billion parameters. Conditioned on a patient's real-world history, CoMET autoregressively generates the next medical event, simulating patient health timelines. We studied 78 real-world tasks, including diagnosis prediction, disease prognosis, and healthcare operations. Remarkably for a foundation model with generic pretraining and simulation-based inference, CoMET generally outperformed or matched task-specific supervised models on these tasks, without requiring task-specific fine-tuning or few-shot examples. CoMET's predictive power consistently improves as the model and pretraining scale. Our results show that CoMET, a generative medical event foundation model, can effectively capture complex clinical dynamics, providing an extensible and generalizable framework to support clinical decision-making, streamline healthcare operations, and improve patient outcomes. 

**Abstract (ZH)**: 大规模实现个性化医学需要从纵向患者历程中提炼洞察的方法，这些历程可视为一系列医疗事件序列。基于大规模医疗事件数据预训练的基础模型代表了扩展现实世界证据生成和泛化到多样下流任务的有前途的方向。使用Epic Cosmos数据集，该数据集包含来自310个医疗系统、30亿个唯一患者记录和1630亿次就诊的去标识化纵向健康记录中的医疗事件，我们介绍了Cosmos Medical Event Transformer (CoMET) 模型，这是一个基于预训练11.8亿患者表示1150亿离散医疗事件（1510亿个标记）的仅解码器变压器模型系列。我们进行了医疗事件数据上最大的规模律研究，建立了预训练方法，并揭示了计算量、标记和模型规模之间的幂律关系。基于此，我们预训练了一系列计算最优模型，参数量最多可达10亿。基于患者的真实历史，CoMET自回归生成下一个医疗事件，模拟患者的健康时间线。我们研究了78个现实世界任务，包括诊断预测、疾病预后和医疗保健操作。令人惊讶的是，作为一个通用预训练和基于模拟推断的基础模型，CoMET在这些任务上通常优于或与特定任务监督模型相当，无需特定任务微调或少量示例。随着模型和预训练规模的扩大，CoMET的预测能力持续增强。我们的结果表明，CoMET作为一种生成性医疗事件基础模型，可以有效捕捉复杂的临床动态，提供一个可扩展和通用的框架，以支持临床决策、优化医疗保健操作并改善患者预后。 

---
# STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transition Samples 

**Title (ZH)**: STEM：通过结构化过渡样本高效评估LLMs的相对能力 

**Authors**: Haiquan Hu, Jiazhi Jiang, Shiyou Xu, Ruhan Zeng, Tian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12096)  

**Abstract**: Evaluating large language models (LLMs) has become increasingly challenging as model capabilities advance rapidly. While recent models often achieve higher scores on standard benchmarks, these improvements do not consistently reflect enhanced real-world reasoning capabilities. Moreover, widespread overfitting to public benchmarks and the high computational cost of full evaluations have made it both expensive and less effective to distinguish meaningful differences between models. To address these challenges, we propose the \textbf{S}tructured \textbf{T}ransition \textbf{E}valuation \textbf{M}ethod (STEM), a lightweight and interpretable evaluation framework for efficiently estimating the relative capabilities of LLMs. STEM identifies \textit{significant transition samples} (STS) by analyzing consistent performance transitions among LLMs of the same architecture but varying parameter scales. These samples enable STEM to effectively estimate the capability position of an unknown model. Qwen3 model family is applied to construct the STS pool on six diverse and representative benchmarks. To assess generalizability. Experimental results indicate that STEM reliably captures performance trends, aligns with ground-truth rankings of model capability. These findings highlight STEM as a practical and scalable method for fine-grained, architecture-agnostic evaluation of LLMs. 

**Abstract (ZH)**: 结构化过渡评估方法（STEM）：一种轻量级且可解释的大型语言模型效率评价框架 

---
# J6: Jacobian-Driven Role Attribution for Multi-Objective Prompt Optimization in LLMs 

**Title (ZH)**: 基于雅可比驱动生成的多目标提示优化中的角色归属 

**Authors**: Yao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12086)  

**Abstract**: In large language model (LLM) adaptation, balancing multiple optimization objectives such as improving factuality (heat) and increasing confidence (via low entropy) poses a fundamental challenge, especially when prompt parameters (e.g., hidden-layer insertions h and embedding modifications w) interact in non-trivial ways. Existing multi-objective optimization strategies often rely on scalar gradient aggregation, ignoring the deeper geometric structure between objectives and parameters. We propose J6, a structured Jacobian-based method that decomposes the gradient interaction matrix into six interpretable components. This decomposition enables both hard decision-making (e.g., choosing the dominant update direction via argmax) and soft strategies (e.g., attention-style weighting via softmax over J6), forming a dynamic update framework that adapts to local conflict and synergy. Moreover, the interpretable structure of J6 provides insight into parameter attribution, task interference, and geometry-aligned adaptation. Our work introduces a principled and extensible mechanism for conflict-aware prompt optimization, and opens a new avenue for incorporating structured Jacobian reasoning into multi-objective neural tuning. 

**Abstract (ZH)**: 在大型语言模型适应中，平衡多个优化目标（如提高事实准确性（热度）和增加信心（通过低熵实现））的关键挑战，尤其是当提示参数（如隐藏层插入h和嵌入修改w）以非平凡方式相互作用时。现有的多目标优化策略通常依赖于标量梯度聚合，忽略了目标和参数之间的深层几何结构。我们提出J6，一种基于结构化雅可比的方法，将梯度交互矩阵分解为六个可解释的组件。这种分解使J6能够进行硬决策（如通过argmax选择主导更新方向）和软策略（如通过softmax权重），形成一种动态更新框架，适应局部冲突和协同效应。此外，J6的可解释结构提供了参数归因、任务干扰和几何对齐适应的见解。我们的工作为具有冲突意识的提示优化引入了一个原则性和可扩展的机制，并开启了一条将结构化雅可比推理纳入多目标神经调优的新途径。 

---
# Generic Event Boundary Detection via Denoising Diffusion 

**Title (ZH)**: 通用事件边界检测 via 去噪扩散 

**Authors**: Jaejun Hwang, Dayoung Gong, Manjin Kim, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.12084)  

**Abstract**: Generic event boundary detection (GEBD) aims to identify natural boundaries in a video, segmenting it into distinct and meaningful chunks. Despite the inherent subjectivity of event boundaries, previous methods have focused on deterministic predictions, overlooking the diversity of plausible solutions. In this paper, we introduce a novel diffusion-based boundary detection model, dubbed DiffGEBD, that tackles the problem of GEBD from a generative perspective. The proposed model encodes relevant changes across adjacent frames via temporal self-similarity and then iteratively decodes random noise into plausible event boundaries being conditioned on the encoded features. Classifier-free guidance allows the degree of diversity to be controlled in denoising diffusion. In addition, we introduce a new evaluation metric to assess the quality of predictions considering both diversity and fidelity. Experiments show that our method achieves strong performance on two standard benchmarks, Kinetics-GEBD and TAPOS, generating diverse and plausible event boundaries. 

**Abstract (ZH)**: 通用事件边界检测（GEBD）旨在识别视频中的自然边界，将其分割为独立且有意义的片段。尽管事件边界的主观性较强，之前的 方法集中在确定性预测上，忽视了可能解的多样性。本文提出了一种基于扩散的边界检测模型，名为DiffGEBD，该模型从生成的角度解决 GEBD 问题。提出的模型通过时间自相似性编码相邻帧的相关变化，然后逐步将随机噪声解码为以编码特征为条件的可能的事件边界。无分类器引导允许在去噪扩散过程中控制多样性的程度。此外，我们引入了一个新的评估指标，考虑多样性和保真度来评估预测质量。实验结果显示，我们的方法在两个标准基准数据集Kinetics-GEBD和TAPOS上表现出色，生成了多样且合理的事件边界。 

---
# Automated Model Evaluation for Object Detection via Prediction Consistency and Reliablity 

**Title (ZH)**: 基于预测一致性和可靠性的自动化目标检测模型评估 

**Authors**: Seungju Yoo, Hyuk Kwon, Joong-Won Hwang, Kibok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.12082)  

**Abstract**: Recent advances in computer vision have made training object detectors more efficient and effective; however, assessing their performance in real-world applications still relies on costly manual annotation. To address this limitation, we develop an automated model evaluation (AutoEval) framework for object detection. We propose Prediction Consistency and Reliability (PCR), which leverages the multiple candidate bounding boxes that conventional detectors generate before non-maximum suppression (NMS). PCR estimates detection performance without ground-truth labels by jointly measuring 1) the spatial consistency between boxes before and after NMS, and 2) the reliability of the retained boxes via the confidence scores of overlapping boxes. For a more realistic and scalable evaluation, we construct a meta-dataset by applying image corruptions of varying severity. Experimental results demonstrate that PCR yields more accurate performance estimates than existing AutoEval methods, and the proposed meta-dataset covers a wider range of detection performance. The code is available at this https URL. 

**Abstract (ZH)**: 最近计算机视觉的进步使目标检测器的训练更加高效和有效；然而，它们在实际应用中的性能评估仍然依赖于昂贵的手动注释。为了解决这一限制，我们开发了一种目标检测的自动模型评估（AutoEval）框架。我们提出了一种预测一致性与可靠性（PCR）方法，该方法利用了常规检测器在非最大抑制（NMS）之前生成的多个候选边界框。PCR通过共同测量1) NMS前后边界框的空间一致性，以及2) 保留边界框的可靠性的置信分数来估算检测性能，无需 ground-truth 标注。为了实现更加现实和可扩展的评估，我们通过应用不同程度的图像腐化构建了一个元数据集。实验结果表明，PCR比现有的自动评估方法提供了更准确的性能估计，并且提出的元数据集涵盖了更广泛的检测性能范围。代码可在此处访问：这个 https URL。 

---
# VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models 

**Title (ZH)**: VimoRAG：基于视频的检索增强3D运动生成用于运动语言模型 

**Authors**: Haidong Xu, Guangwei Xu, Zhedong Zheng, Xiatian Zhu, Wei Ji, Xiangtai Li, Ruijie Guo, Meishan Zhang, Min zhang, Hao Fei  

**Link**: [PDF](https://arxiv.org/pdf/2508.12081)  

**Abstract**: This paper introduces VimoRAG, a novel video-based retrieval-augmented motion generation framework for motion large language models (LLMs). As motion LLMs face severe out-of-domain/out-of-vocabulary issues due to limited annotated data, VimoRAG leverages large-scale in-the-wild video databases to enhance 3D motion generation by retrieving relevant 2D human motion signals. While video-based motion RAG is nontrivial, we address two key bottlenecks: (1) developing an effective motion-centered video retrieval model that distinguishes human poses and actions, and (2) mitigating the issue of error propagation caused by suboptimal retrieval results. We design the Gemini Motion Video Retriever mechanism and the Motion-centric Dual-alignment DPO Trainer, enabling effective retrieval and generation processes. Experimental results show that VimoRAG significantly boosts the performance of motion LLMs constrained to text-only input. 

**Abstract (ZH)**: 基于视频检索增强的运动生成框架VimoRAG：面向运动大语言模型的新型方法 

---
# Generalized invariants meet constitutive neural networks: A novel framework for hyperelastic materials 

**Title (ZH)**: 广义不变量与本构神经网络结合：一类新的超弹性材料框架 

**Authors**: Denisa Martonová, Alain Goriely, Ellen Kuhl  

**Link**: [PDF](https://arxiv.org/pdf/2508.12063)  

**Abstract**: The major challenge in determining a hyperelastic model for a given material is the choice of invariants and the selection how the strain energy function depends functionally on these invariants. Here we introduce a new data-driven framework that simultaneously discovers appropriate invariants and constitutive models for isotropic incompressible hyperelastic materials. Our approach identifies both the most suitable invariants in a class of generalized invariants and the corresponding strain energy function directly from experimental observations. Unlike previous methods that rely on fixed invariant choices or sequential fitting procedures, our method integrates the discovery process into a single neural network architecture. By looking at a continuous family of possible invariants, the model can flexibly adapt to different material behaviors. We demonstrate the effectiveness of this approach using popular benchmark datasets for rubber and brain tissue. For rubber, the method recovers a stretch-dominated formulation consistent with classical models. For brain tissue, it identifies a formulation sensitive to small stretches, capturing the nonlinear shear response characteristic of soft biological matter. Compared to traditional and neural-network-based models, our framework provides improved predictive accuracy and interpretability across a wide range of deformation states. This unified strategy offers a robust tool for automated and physically meaningful model discovery in hyperelasticity. 

**Abstract (ZH)**: 一种新的数据驱动框架：同时发现各向同性不可压缩超弹材料的合适不变量和本构模型 

---
# Large Language Models Enable Personalized Nudges to Promote Carbon Offsetting Among Air Travellers 

**Title (ZH)**: 大型语言模型促进 Airways 旅行者个性化碳补偿建议的研究 

**Authors**: Vladimir Maksimenko, Qingyao Xin, Prateek Gupta, Bin Zhang, Prateek Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2508.12045)  

**Abstract**: Nudge strategies are effective tools for promoting sustainable behaviour, but their impact depends on individual preferences. By emulating human decision-making, large language models (LLMs) offer a cost-effective route for tailoring nudges without extensive behavioural datasets, yet this potential remains unexplored. Focusing on aviation, we use LLMs to design personalized decoy-based nudge strategies that encourage air travellers to voluntarily offset CO$_2$ emissions from flights, and validate their efficacy through 3495 surveys from China, Germany, India, Singapore, and the United States. Results show that LLM-informed personalized nudges are more effective than uniform settings, raising offsetting rates by 3-7$\%$ and yielding an additional 2.3 million tonnes of CO$_2$ mitigated annually in aviation. This improvement is driven primarily by increased participation among sceptical travellers with low trust in offset programmes. Our study highlights the potential of LLM-driven personalized nudging strategies for boosting offsetting behaviours to accelerate aviation decarbonization. 

**Abstract (ZH)**: 大语言模型驱动的个性化助推策略在促进航空领域碳抵消行为中的有效性 

---
# Mind the Generation Process: Fine-Grained Confidence Estimation During LLM Generation 

**Title (ZH)**: 关注生成过程：LLM生成期间的细粒度置信度估计 

**Authors**: Jinyi Han, Tingyun Li, Shisong Chen, Jie Shi, Xinyi Wang, Guanglei Yue, Jiaqing Liang, Xin Lin, Liqian Wen, Zulong Chen, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12040)  

**Abstract**: While large language models (LLMs) have demonstrated remarkable performance across diverse tasks, they fundamentally lack self-awareness and frequently exhibit overconfidence, assigning high confidence scores to incorrect predictions. Accurate confidence estimation is therefore critical for enhancing the trustworthiness and reliability of LLM-generated outputs. However, existing approaches suffer from coarse-grained scoring mechanisms that fail to provide fine-grained, continuous confidence estimates throughout the generation process. To address these limitations, we introduce FineCE, a novel confidence estimation method that delivers accurate, fine-grained confidence scores during text generation. Specifically, we first develop a comprehensive pipeline for constructing training data that effectively captures the underlying probabilistic distribution of LLM responses, and then train a model to predict confidence scores for arbitrary text sequences in a supervised manner. Furthermore, we propose a Backward Confidence Integration (BCI) strategy that leverages information from the subsequent text to enhance confidence estimation for the current sequence during inference. We also introduce three strategies for identifying optimal positions to perform confidence estimation within the generation process. Extensive experiments on multiple benchmark datasets demonstrate that FineCE consistently outperforms existing classical confidence estimation methods. Our code and all baselines used in the paper are available on GitHub. 

**Abstract (ZH)**: 标题：FineCECE：精细生成过程中的的信心估计 

---
# Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering 

**Title (ZH)**: Q-FSRU：量子增强频率谱融合在医学视觉问答中的应用 

**Authors**: Rakesh Thakur, Yusra Tariq  

**Link**: [PDF](https://arxiv.org/pdf/2508.12036)  

**Abstract**: Solving tough clinical questions that require both image and text understanding is still a major challenge in healthcare AI. In this work, we propose Q-FSRU, a new model that combines Frequency Spectrum Representation and Fusion (FSRU) with a method called Quantum Retrieval-Augmented Generation (Quantum RAG) for medical Visual Question Answering (VQA). The model takes in features from medical images and related text, then shifts them into the frequency domain using Fast Fourier Transform (FFT). This helps it focus on more meaningful data and filter out noise or less useful information. To improve accuracy and ensure that answers are based on real knowledge, we add a quantum-inspired retrieval system. It fetches useful medical facts from external sources using quantum-based similarity techniques. These details are then merged with the frequency-based features for stronger reasoning. We evaluated our model using the VQA-RAD dataset, which includes real radiology images and questions. The results showed that Q-FSRU outperforms earlier models, especially on complex cases needing image-text reasoning. The mix of frequency and quantum information improves both performance and explainability. Overall, this approach offers a promising way to build smart, clear, and helpful AI tools for doctors. 

**Abstract (ZH)**: 解决需要同时理解图像和文本的复杂临床问题仍是医疗AI领域的重大挑战。在此工作中，我们提出了一种新的Q-FSRU模型，该模型结合了频谱表示和融合（FSRU）与一种名为量子检索增强生成（Quantum RAG）的方法，用于医学视觉问答（VQA）。该模型接收医学图像和相关文本的特征，然后使用快速傅里叶变换（FFT）将其转换到频域，有助于它聚焦于更有意义的数据并过滤掉噪声或不重要的信息。为提高准确性和确保答案基于真实知识，我们添加了一个受量子启发的检索系统，利用基于量子的方法从外部来源获取有用的医学事实。这些细节随后与基于频谱的特征合并，以增强推理。我们使用包括真实放射学图像和问题的VQA-RAD数据集评估了该模型，结果显示Q-FSRU优于早期模型，尤其是在需要图像文本推理的复杂情况下。频域和量子信息的结合提高了性能和可解释性。总之，这种方法为构建智能、清晰且有助于医生的AI工具提供了有前景的方法。 

---
# BConformeR: A Conformer Based on Mutual Sampling for Unified Prediction of Continuous and Discontinuous Antibody Binding Sites 

**Title (ZH)**: BConformeR：基于互惠采样的卷积器，统一预测连续和非连续抗体检测位点 

**Authors**: Zhangyu You, Jiahao Ma, Hongzong Li, Ye-Fan Hu, Jian-Dong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12029)  

**Abstract**: Accurate prediction of antibody-binding sites (epitopes) on antigens is crucial for vaccine design, immunodiagnostics, therapeutic antibody development, antibody engineering, research into autoimmune and allergic diseases, and for advancing our understanding of immune responses. Despite in silico methods that have been proposed to predict both linear (continuous) and conformational (discontinuous) epitopes, they consistently underperform in predicting conformational epitopes. In this work, we propose a conformer-based model trained on antigen sequences derived from 1,080 antigen-antibody complexes, leveraging convolutional neural networks (CNNs) to extract local features and Transformers to capture long-range dependencies within antigen sequences. Ablation studies demonstrate that CNN enhances the prediction of linear epitopes, and the Transformer module improves the prediction of conformational epitopes. Experimental results show that our model outperforms existing baselines in terms of PCC, ROC-AUC, PR-AUC, and F1 scores on conformational epitopes. 

**Abstract (ZH)**: 准确预测抗原上的抗体结合位点（表位）对于疫苗设计、免疫诊断、治疗性抗体开发、抗体工程、自身免疫和过敏性疾病研究以及增进我们对免疫反应的理解至关重要。尽管已经提出了用于预测线性表位和构象表位的计算机辅助方法，但在预测构象表位方面它们始终表现不佳。在本工作中，我们提出了一种基于构象的模型，该模型基于从中提取抗原序列的1,080个抗原-抗体复合物，利用卷积神经网络（CNNs）提取局部特征，并利用变换器捕获抗原序列中的长距离依赖性。消融研究显示，卷积神经网络增强了线性表位的预测能力，而变换器模块提高了构象表位的预测能力。实验结果表明，与现有的基线方法相比，我们的模型在构象表位上的PCC、ROC-AUC、PR-AUC和F1分数方面表现更优。 

---
# Predicting ChatGPT Use in Assignments: Implications for AI-Aware Assessment Design 

**Title (ZH)**: 预测ChatGPT在作业中的使用：对AI意识评估设计的影响 

**Authors**: Surajit Das, Aleksei Eliseev  

**Link**: [PDF](https://arxiv.org/pdf/2508.12013)  

**Abstract**: The rise of generative AI tools like ChatGPT has significantly reshaped education, sparking debates about their impact on learning outcomes and academic integrity. While prior research highlights opportunities and risks, there remains a lack of quantitative analysis of student behavior when completing assignments. Understanding how these tools influence real-world academic practices, particularly assignment preparation, is a pressing and timely research priority.
This study addresses this gap by analyzing survey responses from 388 university students, primarily from Russia, including a subset of international participants. Using the XGBoost algorithm, we modeled predictors of ChatGPT usage in academic assignments. Key predictive factors included learning habits, subject preferences, and student attitudes toward AI. Our binary classifier demonstrated strong predictive performance, achieving 80.1\% test accuracy, with 80.2\% sensitivity and 79.9\% specificity. The multiclass classifier achieved 64.5\% test accuracy, 64.6\% weighted precision, and 64.5\% recall, with similar training scores, indicating potential data scarcity challenges.
The study reveals that frequent use of ChatGPT for learning new concepts correlates with potential overreliance, raising concerns about long-term academic independence. These findings suggest that while generative AI can enhance access to knowledge, unchecked reliance may erode critical thinking and originality. We propose discipline-specific guidelines and reimagined assessment strategies to balance innovation with academic rigor. These insights can guide educators and policymakers in ethically and effectively integrating AI into education. 

**Abstract (ZH)**: 生成式AI工具（如ChatGPT）的兴起显著重塑了教育，引发了对其对学生学习成果和学术诚信影响的讨论。尽管以往的研究指出了机遇与风险，但对于学生在完成作业时使用这些工具的行为缺乏定量分析。理解这些工具如何影响实际的学术实践，特别是作业准备过程，是一项紧迫而及时的研究优先事项。 

---
# MOON: Generative MLLM-based Multimodal Representation Learning for E-commerce Product Understanding 

**Title (ZH)**: MOON：基于生成型MLLM的多模态表示学习在电子商务产品理解中的应用 

**Authors**: Daoze Zhang, Zhanheng Nie, Jianyu Liu, Chenghan Fu, Wanxian Guan, Yuan Gao, Jun Song, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.11999)  

**Abstract**: With the rapid advancement of e-commerce, exploring general representations rather than task-specific ones has attracted increasing research attention. For product understanding, although existing discriminative dual-flow architectures drive progress in this field, they inherently struggle to model the many-to-one alignment between multiple images and texts of products. Therefore, we argue that generative Multimodal Large Language Models (MLLMs) hold significant potential for improving product representation learning. Nevertheless, achieving this goal still remains non-trivial due to several key challenges: the lack of multimodal and aspect-aware modeling modules in typical LLMs; the common presence of background noise in product images; and the absence of a standard benchmark for evaluation. To address these issues, we propose the first generative MLLM-based model named MOON for product representation learning. Our method (1) employs a guided Mixture-of-Experts (MoE) module for targeted modeling of multimodal and aspect-specific product content; (2) effectively detects core semantic regions in product images to mitigate the distraction and interference caused by background noise; and (3) introduces the specialized negative sampling strategy to increase the difficulty and diversity of negative samples. In addition, we release a large-scale multimodal benchmark MBE for various product understanding tasks. Experimentally, our model demonstrates competitive zero-shot performance on both our benchmark and the public dataset, showcasing strong generalization across various downstream tasks, including cross-modal retrieval, product classification, and attribute prediction. Furthermore, the case study and visualization illustrate the effectiveness of MOON for product understanding. 

**Abstract (ZH)**: 随着电子商务的快速发展，探索通用表示而非任务特定表示的研究吸引了越来越多的关注。在产品理解方面，尽管现有的鉴别性双流架构在这领域取得了进步，但它们在建模产品多张图片与文本之间的一对多对齐时普遍存在困难。因此，我们认为生成型多模态大型语言模型（Generative Multimodal Large Language Models, MLLMs）在提高产品表示学习方面具有巨大的潜力。然而，由于几个关键挑战的存在，实现这一目标仍然非 trivial：典型的大语言模型中缺乏多模态和观点感知的建模模块；产品图片中常见的背景噪声；以及缺乏一个标准的评估基准。为了解决这些问题，我们提出了名为MOON的第一个基于生成型MLLM的产品表示学习模型。我们的方法（1）采用指导的混合专家（MoE）模块，针对多模态和观点特定的产品内容进行建模；（2）有效检测产品图片中的核心语义区域，以减轻背景噪声引起的干扰和干扰；（3）引入专门的负样本策略，以增加负样本的难度和多样性。此外，我们还发布了大规模多模态基准MBE，用于各类产品理解任务。实验结果显示，我们的模型在我们的基准和公开数据集上均表现出竞争力的零样本性能，展示了在各种下游任务中（包括跨模态检索、产品分类和属性预测）的强大泛化能力。进一步的研究案例和可视化说明了MOON在产品理解方面的有效性。 

---
# Efficient Modular Learning through Naive LoRA Summation: Leveraging Orthogonality in High-Dimensional Models 

**Title (ZH)**: 通过朴素LoRA求和实现高效模块化学习：利用高维模型中的正交性 

**Authors**: Zhanhao Cao, Clement Truong, Andrew Lizarraga  

**Link**: [PDF](https://arxiv.org/pdf/2508.11985)  

**Abstract**: Recent advances in large language models are driven by scale, while parameter-efficient fine-tuning (PEFT) enables updating only a small fraction of parameters. Low-Rank Adaptation (LoRA) stores parameter deltas as the product of two small matrices, which makes them natural building blocks that can be composed. Motivated by the superposition principle, we hypothesize that independently trained LoRA modules on disjoint domains are approximately orthogonal and can be combined by simple addition. Using GPT-2 Small (117M) with LoRA rank 4 and alpha=64, we train adapters for three QA domains (math, medicine, finance). In pairwise tests, adding Math+Medicine adapters improves perplexity by -9.10% relative to merged-data fine-tuning, while Math+Finance and Finance+Medicine change by +4.54% and +27.56%, respectively. Across combinations, the RMS cosine similarity between LoRA deltas correlates positively and approximately linearly with the change in perplexity. Naive summation requires no additional training, can be applied in seconds, and achieves performance comparable to models trained on merged data, while clarifying when interference appears in higher-order compositions. 

**Abstract (ZH)**: Recent Advances in Large Language ModelsDriven by Scale and Enabled by Parameter-Efficient Fine-Tuning: Low-Rank Adaptation (LoRA) and Its Composability 

---
# TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios 

**Title (ZH)**: TBGRecall：电子商务推荐场景下的生成式检索模型 

**Authors**: Zida Liang, Changfa Wu, Dunxian Huang, Weiqiang Sun, Ziyang Wang, Yuliang Yan, Jian Wu, Yuning Jiang, Bo Zheng, Ke Chen, Silu Zhou, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11977)  

**Abstract**: Recommendation systems are essential tools in modern e-commerce, facilitating personalized user experiences by suggesting relevant products. Recent advancements in generative models have demonstrated potential in enhancing recommendation systems; however, these models often exhibit limitations in optimizing retrieval tasks, primarily due to their reliance on autoregressive generation mechanisms. Conventional approaches introduce sequential dependencies that impede efficient retrieval, as they are inherently unsuitable for generating multiple items without positional constraints within a single request session. To address these limitations, we propose TBGRecall, a framework integrating Next Session Prediction (NSP), designed to enhance generative retrieval models for e-commerce applications. Our framework reformulation involves partitioning input samples into multi-session sequences, where each sequence comprises a session token followed by a set of item tokens, and then further incorporate multiple optimizations tailored to the generative task in retrieval scenarios. In terms of training methodology, our pipeline integrates limited historical data pre-training with stochastic partial incremental training, significantly improving training efficiency and emphasizing the superiority of data recency over sheer data volume. Our extensive experiments, conducted on public benchmarks alongside a large-scale industrial dataset from TaoBao, show TBGRecall outperforms the state-of-the-art recommendation methods, and exhibits a clear scaling law trend. Ultimately, NSP represents a significant advancement in the effectiveness of generative recommendation systems for e-commerce applications. 

**Abstract (ZH)**: 基于Next Session Prediction的TBGRecall强化生成推荐框架 

---
# A Comprehensive Review of AI Agents: Transforming Possibilities in Technology and Beyond 

**Title (ZH)**: AI代理的全面综述：技术及 Beyond 的可能性转化 

**Authors**: Xiaodong Qu, Andrews Damoah, Joshua Sherwood, Peiyan Liu, Christian Shun Jin, Lulu Chen, Minjie Shen, Nawwaf Aleisa, Zeyuan Hou, Chenyu Zhang, Lifu Gao, Yanshu Li, Qikai Yang, Qun Wang, Cristabelle De Souza  

**Link**: [PDF](https://arxiv.org/pdf/2508.11957)  

**Abstract**: Artificial Intelligence (AI) agents have rapidly evolved from specialized, rule-based programs to versatile, learning-driven autonomous systems capable of perception, reasoning, and action in complex environments. The explosion of data, advances in deep learning, reinforcement learning, and multi-agent coordination have accelerated this transformation. Yet, designing and deploying unified AI agents that seamlessly integrate cognition, planning, and interaction remains a grand challenge. In this review, we systematically examine the architectural principles, foundational components, and emergent paradigms that define the landscape of contemporary AI agents. We synthesize insights from cognitive science-inspired models, hierarchical reinforcement learning frameworks, and large language model-based reasoning. Moreover, we discuss the pressing ethical, safety, and interpretability concerns associated with deploying these agents in real-world scenarios. By highlighting major breakthroughs, persistent challenges, and promising research directions, this review aims to guide the next generation of AI agent systems toward more robust, adaptable, and trustworthy autonomous intelligence. 

**Abstract (ZH)**: 人工智能（AI）代理从专门的基于规则的程序迅速进化为能够在复杂环境中进行感知、推理和行动的多功能、学习驱动的自主系统。数据爆炸、深度学习、强化学习和多代理协调的进步加速了这一转变。然而，设计和部署能够无缝集成认知、规划和交互的统一AI代理仍然是一个巨大的挑战。在本文综述中，我们系统地 examine 了当代AI代理所涉及的体系结构原则、基础组件和新兴范式。我们综合了认知科学启发式模型、层级强化学习框架以及基于大型语言模型的推理方面的洞见。此外，我们还讨论了部署这些代理在实际应用场景中所面临的紧迫的伦理、安全性和可解释性问题。通过突出重大突破、持久性挑战和有希望的研究方向，本文综述旨在引导下一代AI代理系统朝着更稳健、更适应和更可信的自主智能方向发展。 

---
# Extending Straight-Through Estimation for Robust Neural Networks on Analog CIM Hardware 

**Title (ZH)**: 在类比CIM硬件上扩展直通估计以构建鲁棒神经网络 

**Authors**: Yuannuo Feng, Wenyong Zhou, Yuexi Lyu, Yixiang Zhang, Zhengwu Liu, Ngai Wong, Wang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11940)  

**Abstract**: Analog Compute-In-Memory (CIM) architectures promise significant energy efficiency gains for neural network inference, but suffer from complex hardware-induced noise that poses major challenges for deployment. While noise-aware training methods have been proposed to address this issue, they typically rely on idealized and differentiable noise models that fail to capture the full complexity of analog CIM hardware variations. Motivated by the Straight-Through Estimator (STE) framework in quantization, we decouple forward noise simulation from backward gradient computation, enabling noise-aware training with more accurate but computationally intractable noise modeling in analog CIM systems. We provide theoretical analysis demonstrating that our approach preserves essential gradient directional information while maintaining computational tractability and optimization stability. Extensive experiments show that our extended STE framework achieves up to 5.3% accuracy improvement on image classification, 0.72 perplexity reduction on text generation, 2.2$\times$ speedup in training time, and 37.9% lower peak memory usage compared to standard noise-aware training methods. 

**Abstract (ZH)**: 基于内存的类比计算-in-内存（CIM）架构有望显著提高神经网络推理的能效，但复杂的硬件诱导噪声带来了部署的重大挑战。虽然已经提出了噪声感知训练方法来解决这一问题，但这些方法通常依赖于理想化和可微的噪声模型，无法捕捉到模拟CIM硬件变异的全部复杂性。受量化中straight-through estimator（STE）框架的启发，我们分离了前向噪声模拟与后向梯度计算，使得在模拟更准确但计算上难以处理的噪声模型时也能进行噪声感知训练。我们提供了理论分析，证明了我们方法能够保留关键的梯度方向信息，同时保持计算的可行性并维持优化的稳定性。 extensive实验表明，我们的扩展STE框架在图像分类上可实现高达5.3%的准确性提升，在文本生成中可实现0.72的困惑度降低，在训练时间上可实现2.2倍的加速，在峰值内存使用上可减少37.9%。 

---
# HPD: Hybrid Projection Decomposition for Robust State Space Models on Analog CIM Hardware 

**Title (ZH)**: HPD: 综合投影分解方法在类比CIM硬件上实现稳健的状态空间模型 

**Authors**: Yuannuo Feng, Wenyong Zhou, Yuexi Lyu, Hanjie Liu, Zhengwu Liu, Ngai Wong, Wang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11935)  

**Abstract**: State Space Models (SSMs) are efficient alternatives to traditional sequence models, excelling at processing long sequences with lower computational complexity. Their reliance on matrix multiplications makes them ideal for compute-in-memory (CIM) architectures, which improve energy efficiency by computing within memory arrays. However, device non-idealities in CIM introduce weight perturbations that can degrade inference accuracy. In this paper, we systematically analyze the robustness of SSMs under noisy conditions, identifying that the final block and output projection layers are more susceptible to perturbations compared to other components. Building on these insights, we propose HPD, a Hybrid Projection Decomposition strategy for the last output projection layer. We replace the original weight matrix with the multiplication of U and {\Sigma} in its SVD to ensure compatibility with existing hardware architectures, while offloading V> to digital hardware for precise and robust correction. Comprehensive tests on Mamba models show that our method reduces perplexity by up to 99.57% under various noise conditions compared to baseline models, with accuracy gains of up to 96.67% on the PIQA benchmark for commonsense reasoning. 

**Abstract (ZH)**: State Space Models在噪声条件下的鲁棒性分析及HPD策略 

---
# No More Blind Spots: Learning Vision-Based Omnidirectional Bipedal Locomotion for Challenging Terrain 

**Title (ZH)**: 无盲区：基于视觉的全方位双足运动学习在挑战性地形上的应用 

**Authors**: Mohitvishnu S. Gadde, Pranay Dugar, Ashish Malik, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2508.11929)  

**Abstract**: Effective bipedal locomotion in dynamic environments, such as cluttered indoor spaces or uneven terrain, requires agile and adaptive movement in all directions. This necessitates omnidirectional terrain sensing and a controller capable of processing such input. We present a learning framework for vision-based omnidirectional bipedal locomotion, enabling seamless movement using depth images. A key challenge is the high computational cost of rendering omnidirectional depth images in simulation, making traditional sim-to-real reinforcement learning (RL) impractical. Our method combines a robust blind controller with a teacher policy that supervises a vision-based student policy, trained on noise-augmented terrain data to avoid rendering costs during RL and ensure robustness. We also introduce a data augmentation technique for supervised student training, accelerating training by up to 10 times compared to conventional methods. Our framework is validated through simulation and real-world tests, demonstrating effective omnidirectional locomotion with minimal reliance on expensive rendering. This is, to the best of our knowledge, the first demonstration of vision-based omnidirectional bipedal locomotion, showcasing its adaptability to diverse terrains. 

**Abstract (ZH)**: 基于视觉的 omnidirectional �灵巧双足步行框架：降低渲染成本并 提高鲁棒性 

---
# ENA: Efficient N-dimensional Attention 

**Title (ZH)**: ENA: 高效的N维注意力机制 

**Authors**: Yibo Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.11921)  

**Abstract**: Efficient modeling of long sequences of high-order data requires a more efficient architecture than Transformer. In this paper, we investigate two key aspects of extending linear recurrent models, especially those originally designed for language modeling, to high-order data (1D to ND): scanning strategies and attention-hybrid architectures. Empirical results suggest that scanning provides limited benefits, while attention-hybrid models yield promising results. Focusing on the latter, we further evaluate types of attention and find that tiled high-order sliding window attention (SWA) is efficient in both theory and practice. We term the resulting hybrid architecture of linear recurrence and high-order SWA as Efficient N-dimensional Attention (ENA). We then conduct several experiments to demonstrate its effectiveness. The intuition behind ENA is that linear recurrence compresses global information into a state, while SWA complements it by enforcing strict local modeling. Together, they form a simple framework that offers a promising and practical solution for ultra-long high-order data modeling. 

**Abstract (ZH)**: 高效建模高阶数据的长序列需要一种比Transformer更高效的架构。本文研究了将特别设计用于语言建模的一维线性递归模型扩展到高阶数据（1D到ND）的两个关键方面：扫描策略和注意力-混合架构。实验结果表明，扫描提供的收益有限，而注意力-混合模型显示出有希望的结果。着重于后者，我们进一步评估了不同类型的注意力机制，发现拼接的高阶滑动窗口注意力(SWA)在理论和实践中都高效。我们将这种线性递归与高阶SWA相结合的混合架构命名为高效N维注意力(ENA)。然后，我们进行了若干实验以证明其有效性。ENA的基本思想是，线性递归将全局信息压缩到一个状态，而SWA通过强制进行严格的局部建模来补充这一点。两者结合形成了一种简单框架，为超长高阶数据建模提供了有希望且实用的解决方案。 

---
# CORE: Measuring Multi-Agent LLM Interaction Quality under Game-Theoretic Pressures 

**Title (ZH)**: CORE: 在博弈论压力下的多智能体LLM交互质量度量 

**Authors**: Punya Syon Pandey, Yongjin Yang, Jiarui Liu, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.11915)  

**Abstract**: Game-theoretic interactions between agents with Large Language Models (LLMs) have revealed many emergent capabilities, yet the linguistic diversity of these interactions has not been sufficiently quantified. In this paper, we present the Conversational Robustness Evaluation Score: CORE, a metric to quantify the effectiveness of language use within multi-agent systems across different game-theoretic interactions. CORE integrates measures of cluster entropy, lexical repetition, and semantic similarity, providing a direct lens of dialog quality. We apply CORE to pairwise LLM dialogs across competitive, cooperative, and neutral settings, further grounding our analysis in Zipf's and Heaps' Laws to characterize word frequency distributions and vocabulary growth. Our findings show that cooperative settings exhibit both steeper Zipf distributions and higher Heap exponents, indicating more repetition alongside greater vocabulary expansion. In contrast, competitive interactions display lower Zipf and Heaps exponents, reflecting less repetition and more constrained vocabularies. These results provide new insights into how social incentives influence language adaptation, and highlight CORE as a robust diagnostic for measuring linguistic robustness in multi-agent LLM systems. Our code is available at this https URL. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）的代理之间的博弈论交互展示了许多新兴能力，但这些交互的语言多样性尚未得到充分量化。在本文中，我们提出了对话稳健性评估得分：CORE，一个量化多代理系统中不同博弈论交互内语言使用效果的指标。CORE结合了聚类熵、词项重复和语义相似度的衡量标准，提供了一个直接的对话质量视角。我们将CORE应用于竞争性、合作性和中性设置下的两两LLM对话，并进一步在 Zipf 原理和 Heaps 原理中进行分析，以描述词频分布和词汇量的增长。我们的研究结果表明，合作性设置展示了更陡峭的 Zipf 分布和更高的 Heaps 前项，表明更多的重复伴随更大的词汇扩展。相比之下，竞争性交互显示较低的 Zipf 和 Heaps 前项，反映出较少的重复和更为受限的词汇量。这些结果为社会激励如何影响语言适应提供了新的见解，并强调了CORE作为多代理LLM系统中语言稳健性测量的稳健诊断工具的重要性。我们的代码可在以下链接获取：这个 https URL。 

---
# Deciphering the Interplay between Attack and Protection Complexity in Privacy-Preserving Federated Learning 

**Title (ZH)**: 解构隐私保护联邦学习中攻击复杂度与防御复杂度的相互作用 

**Authors**: Xiaojin Zhang, Mingcong Xu, Yiming Li, Wei Chen, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11907)  

**Abstract**: Federated learning (FL) offers a promising paradigm for collaborative model training while preserving data privacy. However, its susceptibility to gradient inversion attacks poses a significant challenge, necessitating robust privacy protection mechanisms. This paper introduces a novel theoretical framework to decipher the intricate interplay between attack and protection complexities in privacy-preserving FL. We formally define "Attack Complexity" as the minimum computational and data resources an adversary requires to reconstruct private data below a given error threshold, and "Protection Complexity" as the expected distortion introduced by privacy mechanisms. Leveraging Maximum Bayesian Privacy (MBP), we derive tight theoretical bounds for protection complexity, demonstrating its scaling with model dimensionality and privacy budget. Furthermore, we establish comprehensive bounds for attack complexity, revealing its dependence on privacy leakage, gradient distortion, model dimension, and the chosen privacy level. Our findings quantitatively illuminate the fundamental trade-offs between privacy guarantees, system utility, and the effort required for both attacking and defending. This framework provides critical insights for designing more secure and efficient federated learning systems. 

**Abstract (ZH)**: 联邦学习（FL）提供了一种在保护数据隐私的同时进行协作模型训练的有希望的范式。然而，其对梯度反转攻击的敏感性构成了一个显著的挑战，需要 robust 的隐私保护机制。本文引入了一种新的理论框架，以揭示保护隐私的联邦学习中攻击复杂性和保护复杂性的复杂交互。我们正式定义“攻击复杂性”为攻击者在低于给定误差阈值的条件下重建私有数据所需的最小计算和数据资源，“保护复杂性”为隐私机制引入的预期失真度。基于最大贝叶斯隐私（MBP），我们推导出保护复杂性的紧界，展示了其与模型维度和隐私预算的关联性。此外，我们建立了攻击复杂性的全面界限，揭示了其依赖于隐私泄露、梯度失真、模型维度和所选隐私级别。我们的研究定量地阐明了隐私保证、系统效用以及攻击和防御所需努力之间的基本权衡。该框架为设计更安全高效的联邦学习系统提供了关键见解。 

---
# Integrating Symbolic RL Planning into a BDI-based Autonomous UAV Framework: System Integration and SIL Validation 

**Title (ZH)**: 基于BDI的自主无人机框架中符号化RL规划的集成与SIL验证 

**Authors**: Sangwoo Jeon, Juchul Shin, YeonJe Cho, Gyeong-Tae Kim, Seongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.11890)  

**Abstract**: Modern autonomous drone missions increasingly require software frameworks capable of seamlessly integrating structured symbolic planning with adaptive reinforcement learning (RL). Although traditional rule-based architectures offer robust structured reasoning for drone autonomy, their capabilities fall short in dynamically complex operational environments that require adaptive symbolic planning. Symbolic RL (SRL), using the Planning Domain Definition Language (PDDL), explicitly integrates domain-specific knowledge and operational constraints, significantly improving the reliability and safety of unmanned aerial vehicle (UAV) decision making. In this study, we propose the AMAD-SRL framework, an extended and refined version of the Autonomous Mission Agents for Drones (AMAD) cognitive multi-agent architecture, enhanced with symbolic reinforcement learning for dynamic mission planning and execution. We validated our framework in a Software-in-the-Loop (SIL) environment structured identically to an intended Hardware-In-the-Loop Simulation (HILS) platform, ensuring seamless transition to real hardware. Experimental results demonstrate stable integration and interoperability of modules, successful transitions between BDI-driven and symbolic RL-driven planning phases, and consistent mission performance. Specifically, we evaluate a target acquisition scenario in which the UAV plans a surveillance path followed by a dynamic reentry path to secure the target while avoiding threat zones. In this SIL evaluation, mission efficiency improved by approximately 75% over a coverage-based baseline, measured by travel distance reduction. This study establishes a robust foundation for handling complex UAV missions and discusses directions for further enhancement and validation. 

**Abstract (ZH)**: 现代自主无人机机群 increasingly 需要能够无缝集成 结构符号规划 与自适应强化学习 (RL) 的 软件框架。传统的基于规则的架构 提供了 robust 的 结构推理 能力 以支持无人机的 自主能力 但在动态复杂操作环境中 需要自适应 � � � � � � � � �xa unterstütztes 返回原始内容以正确 � nike � domaine 定 � formal 结构语法 �钺z � régl �畈(ti 容 擩än 类的题 则än 安 幌 于acom 容撞 专题�一路上专门刘兽景并 z 在 � cocina �tá �一个星期二 �网游 盇 按z � 衽 攕 挤 挽qz 肋  槷吻z吻 搢z 衪 滴 � 冶z 盋z �zansson 遟zzhz 扟z血z �盟z 苌z盟zzz盟zanssonzzip �z 个多余的部分。*</samp></code>>` 最终翻译z 实要 zz 这 结zz � 宻 挞zgn 保 悜z �卷zanga z 乘坐z z 鑛 �血z盟z �的� 个z 扽 个z 扛 结zz题 änzw 和z  它z 不 了z 有意义的内容。*</samp></codez>` 

---
# EVTP-IVS: Effective Visual Token Pruning For Unifying Instruction Visual Segmentation In Multi-Modal Large Language Models 

**Title (ZH)**: EVTP-IVS: 有效视觉词元修剪以优化多
user
EVTP-IVS: Effective Visual Token Pruning For Instruction Visual Segmentation In Multi-Modal Large Language Modelspõe
enerated
EVTP-IVS：多有效的视觉词元修剪以优化多视点大型语言模型的指令视觉分割 

**Authors**: Wenhui Zhu, Xiwen Chen, Zhipeng Wang, Shao Tang, Sayan Ghosh, Xuanzhao Dong, Rajat Koner, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11886)  

**Abstract**: Instructed Visual Segmentation (IVS) tasks require segmenting objects in images or videos based on natural language instructions. While recent multimodal large language models (MLLMs) have achieved strong performance on IVS, their inference cost remains a major bottleneck, particularly in video. We empirically analyze visual token sampling in MLLMs and observe a strong correlation between subset token coverage and segmentation performance. This motivates our design of a simple and effective token pruning method that selects a compact yet spatially representative subset of tokens to accelerate inference. In this paper, we introduce a novel visual token pruning method for IVS, called EVTP-IV, which builds upon the k-center by integrating spatial information to ensure better coverage. We further provide an information-theoretic analysis to support our design. Experiments on standard IVS benchmarks show that our method achieves up to 5X speed-up on video tasks and 3.5X on image tasks, while maintaining comparable accuracy using only 20% of the tokens. Our method also consistently outperforms state-of-the-art pruning baselines under varying pruning ratios. 

**Abstract (ZH)**: 基于自然语言指令的视觉分割（IVS）任务要求根据自然语言指令对图像或视频中的对象进行分割。尽管最近的多模态大型语言模型（MLLMs）在IVS任务上取得了强大的性能，但其推理成本仍然是一个主要瓶颈，尤其是在视频任务中。我们通过实证分析MLLMs中的视觉标记采样，并观察到子集标记覆盖度与分割性能之间存在强烈的相关性。这促使我们设计了一种简单而有效的标记剪枝方法，该方法选择一个紧凑且空间上具有代表性的标记子集以加速推理。在本文中，我们提出了一种基于k-center并整合空间信息的新颖视觉标记剪枝方法EVTP-IV，以确保更好的覆盖度。我们进一步提供了一种信息论分析来支持该设计。在标准的IVS基准测试上的实验表明，我们的方法在视频任务中实现了最高达5倍的加速，在图像任务中实现了3.5倍的加速，同时仅使用20%的标记就维持了相当的准确度。此外，在不同的剪枝比例下，我们的方法也一致优于现有的剪枝基线方法。 

---
# Discovering Expert-Level Nash Equilibrium Algorithms with Large Language Models 

**Title (ZH)**: 使用大型语言模型发现专家级纳什均衡算法 

**Authors**: Hanyu Li, Dongchen Li, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.11874)  

**Abstract**: Algorithm design and analysis is a cornerstone of computer science, but it confronts a major challenge. Proving an algorithm's performance guarantee across all inputs has traditionally required extensive and often error-prone human effort. While AI has shown great success in finding solutions to specific problem instances, automating the discovery of general algorithms with such provable guarantees has remained a significant barrier. This challenge stems from the difficulty of integrating the creative process of algorithm design with the rigorous process of formal analysis. To address this gap, we propose LegoNE, a framework that tightly fuses these two processes for the fundamental and notoriously difficult problem of computing approximate Nash equilibria. LegoNE automatically translates any algorithm written by a simple Python-like language into a constrained optimization problem. Solving this problem derives and proves the algorithm's approximation bound. Using LegoNE, a state-of-the-art large language model rediscovered the state-of-the-art algorithm for two-player games within hours, a feat that had taken human researchers 15 years to achieve. For three-player games, the model discovered a novel algorithm surpassing all existing human-designed ones. This work demonstrates a new human-machine collaborative paradigm for theoretical science: humans reason at a higher-abstract level, using symbols to compress the search space, and AI explores within it, achieving what neither could alone. 

**Abstract (ZH)**: 一种紧耦合算法设计与形式化分析的框架：LegoNE及其在计算纳什均衡近似值中的应用 

---
# SimInterview: Transforming Business Education through Large Language Model-Based Simulated Multilingual Interview Training System 

**Title (ZH)**: SimInterview: 通过基于大规模语言训练的模拟多语言面试培训转型商业教育 

**Authors**: Truong Thanh Hung Nguyen, Tran Diem Quynh Nguyen, Hoang Loc Cao, Thi Cam Thanh Tran, Thi Cam Mai Truong, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11873)  

**Abstract**: Business interview preparation demands both solid theoretical grounding and refined soft skills, yet conventional classroom methods rarely deliver the individualized, culturally aware practice employers currently expect. This paper introduces SimInterview, a large language model (LLM)-based simulated multilingual interview training system designed for business professionals entering the AI-transformed labor market. Our system leverages an LLM agent and synthetic AI technologies to create realistic virtual recruiters capable of conducting personalized, real-time conversational interviews. The framework dynamically adapts interview scenarios using retrieval-augmented generation (RAG) to match individual resumes with specific job requirements across multiple languages. Built on LLMs (OpenAI o3, Llama 4 Maverick, Gemma 3), integrated with Whisper speech recognition, GPT-SoVITS voice synthesis, Ditto diffusion-based talking head generation model, and ChromaDB vector databases, our system significantly improves interview readiness across English and Japanese markets. Experiments with university-level candidates show that the system consistently aligns its assessments with job requirements, faithfully preserves resume content, and earns high satisfaction ratings, with the lightweight Gemma 3 model producing the most engaging conversations. Qualitative findings revealed that the standardized Japanese resume format improved document retrieval while diverse English resumes introduced additional variability, and they highlighted how cultural norms shape follow-up questioning strategies. Finally, we also outlined a contestable AI design that can explain, detect bias, and preserve human-in-the-loop to meet emerging regulatory expectations. 

**Abstract (ZH)**: 基于大规模语言模型的SimInterview：面向AI转型劳动力市场的商务模拟多语言面试训练系统 

---
# Singing Syllabi with Virtual Avatars: Enhancing Student Engagement Through AI-Generated Music and Digital Embodiment 

**Title (ZH)**: 使用虚拟 avatar 唱出音节：通过 AI 生成的音乐和数字 embodient 提高学生参与度 

**Authors**: Xinxing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11872)  

**Abstract**: In practical teaching, we observe that few students thoroughly read or fully comprehend the information provided in traditional, text-based course syllabi. As a result, essential details, such as course policies and learning outcomes, are frequently overlooked. To address this challenge, in this paper, we propose a novel approach leveraging AI-generated singing and virtual avatars to present syllabi in a format that is more visually appealing, engaging, and memorable. Especially, we leveraged the open-source tool, HeyGem, to transform textual syllabi into audiovisual presentations, in which digital avatars perform the syllabus content as songs. The proposed approach aims to stimulate students' curiosity, foster emotional connection, and enhance retention of critical course information. Student feedback indicated that AI-sung syllabi significantly improved awareness and recall of key course information. 

**Abstract (ZH)**: 在实际教学中，我们观察到多数学生没有充分阅读或全面理解传统文本式课程大纲所提供的信息。因此，诸如课程政策和学习成果等重要细节经常被忽略。为应对这一挑战，本文提出了一种利用AI生成演唱和虚拟角色展示课程大纲的新方法，使课程大纲以更具吸引力、更易于记忆的视听格式呈现。特别是，我们利用开源工具HeyGem将文本式课程大纲转换为视听展示，在其中数字角色以歌曲形式表演课程内容。所提出的方法旨在激发学生的好奇心，促进情感连接，并增强对关键课程信息的记忆。学生反馈表明，AI演唱的课程大纲显著提高了他们对关键课程信息的意识和回忆。 

---
# AdaRing: Towards Ultra-Light Vision-Language Adaptation via Cross-Layer Tensor Ring Decomposition 

**Title (ZH)**: AdaRing: 向量跨越层张量环分解实现超轻量视觉-语言适应 

**Authors**: Ying Huang, Yuanbin Man, Wenqi Jia, Zhengzhong Tu, Junzhou Huang, Miao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.11870)  

**Abstract**: Adapter-based fine-tuning has gained remarkable attention in adapting large pre-trained vision language models (VLMs) for a wide range of downstream tasks efficiently. In this paradigm, only the inserted adapters are fine-tuned, without the need for training the original VLM backbone. Existing works scale adapters by integrating them into every layer of VLMs to increase the capacity of adapters. However, these methods face two primary limitations: 1) limited compression rate due to ignoring cross-layer redundancy, and 2) limited representational capacity across homogeneous adapters. In this paper, we propose a novel vision-language fine-tuning framework based on cross-layer tensor ring decomposition (TRD) with the integration and collaboration of diverse adapters, called AdaRing, achieving ultra-light parameter-efficient adaptation of VLMs on various tasks. To remove the high redundancy that exists among adapters across layers, we exploit the tensor-level low-rankness to formulate adapters as layer-shared tensor cores and layer-specific slices. Moreover, guided by generalization-aware fine-tuning, diverse rank-driven adapters cooperate to handle tasks that require different representations. Our experiments show that the proposed AdaRing achieves the state-of-the-art performance while reducing average training parameters by 90%. 

**Abstract (ZH)**: 基于跨层张量环分解的适配器协作细调框架AdaRing 

---
# Data Shift of Object Detection in Autonomous Driving 

**Title (ZH)**: 自主驾驶中目标检测的数据偏移 

**Authors**: Lida Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11868)  

**Abstract**: With the widespread adoption of machine learning technologies in autonomous driving systems, their role in addressing complex environmental perception challenges has become increasingly crucial. However, existing machine learning models exhibit significant vulnerability, as their performance critically depends on the fundamental assumption that training and testing data satisfy the independent and identically distributed condition, which is difficult to guarantee in real-world applications. Dynamic variations in data distribution caused by seasonal changes, weather fluctuations lead to data shift problems in autonomous driving systems. This study investigates the data shift problem in autonomous driving object detection tasks, systematically analyzing its complexity and diverse manifestations. We conduct a comprehensive review of data shift detection methods and employ shift detection analysis techniques to perform dataset categorization and balancing. Building upon this foundation, we construct an object detection model. To validate our approach, we optimize the model by integrating CycleGAN-based data augmentation techniques with the YOLOv5 framework. Experimental results demonstrate that our method achieves superior performance compared to baseline models on the BDD100K dataset. 

**Abstract (ZH)**: 随着机器学习技术在自动驾驶系统中的广泛应用，其在应对复杂环境感知挑战中的作用变得日益重要。然而，现有的机器学习模型表现出明显的脆弱性，因为它们的性能高度依赖于训练和测试数据独立同分布的基本假设，在实际应用中难以保证。由季节变化和天气波动引起的数据分布动态变化导致了自动驾驶系统中的数据偏移问题。本研究探讨了自动驾驶目标检测任务中的数据偏移问题，系统地分析了其复杂性和多种表现形式。我们全面回顾了数据偏移检测方法，并运用偏移检测分析技术对数据集进行分类和平衡。在此基础上，我们构建了一个目标检测模型。为了验证我们的方法，我们通过将CycleGAN基于的数据增强技术与YOLOv5框架结合，优化了该模型。实验结果表明，我们的方法在BDD100K数据集上的性能优于基准模型。 

---
# AI-Augmented CI/CD Pipelines: From Code Commit to Production with Autonomous Decisions 

**Title (ZH)**: 基于AI增强的CI/CD管道：从代码提交到生产中的自主决策 

**Authors**: Mohammad Baqar, Saba Naqvi, Rajat Khanda  

**Link**: [PDF](https://arxiv.org/pdf/2508.11867)  

**Abstract**: Modern software delivery has accelerated from quarterly releases to multiple deployments per day. While CI/CD tooling has matured, human decision points interpreting flaky tests, choosing rollback strategies, tuning feature flags, and deciding when to promote a canary remain major sources of latency and operational toil. We propose AI-Augmented CI/CD Pipelines, where large language models (LLMs) and autonomous agents act as policy-bounded co-pilots and progressively as decision makers. We contribute: (1) a reference architecture for embedding agentic decision points into CI/CD, (2) a decision taxonomy and policy-as-code guardrail pattern, (3) a trust-tier framework for staged autonomy, (4) an evaluation methodology using DevOps Research and Assessment ( DORA) metrics and AI-specific indicators, and (5) a detailed industrial-style case study migrating a React 19 microservice to an AI-augmented pipeline. We discuss ethics, verification, auditability, and threats to validity, and chart a roadmap for verifiable autonomy in production delivery systems. 

**Abstract (ZH)**: 现代软件交付从季度发布加速到了频繁部署。虽然持续集成（CI）工具已经成熟，但在人类决策中解读不稳定的测试、选择回滚策略、决定何时进行蓝绿部署等问题仍然是延迟和运营瓶颈。我们提出了一种基于AI增强的CI/AI流水线，，其中大型语言模型（LLMs）和自主代理充当政策基线决策者，并并 � modele逐步将决策者角色。我们贡献了：（1）一一种嵌入代理决策点的CI/AI流水平行结构；（）一一种决定性分类和政策即代码护栏模板；（3）一一种自主分级框架以调整自主程度；（4）一一种使用DevOps研究与评估（DORA）指标和和超过的的AI特定指标的评估方法；andalan ），）（5）一一种详细的企业级案例研究，将一个React微服务迁移至至D一个基于AI增强增强增强增强增强增强增的流水平行平行D。�ste；。；用于该D一个更多D过信息DDD的信息。。。因此我们将讨论伦理、D验证验证、D审计D和审计DAnd和以及有效D性的威胁D并并，并并并并并并并并以及D绘制一条确保D自动化的可领域D验证D验证可靠的D开发交付框架D的的道路。 

---
# SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance 

**Title (ZH)**: SupraTok：跨边界分词以提升语言模型性能 

**Authors**: Andrei-Valentin Tănase, Elena Pelican  

**Link**: [PDF](https://arxiv.org/pdf/2508.11857)  

**Abstract**: Tokenization remains a fundamental yet underexplored bottleneck in natural language processing, with strategies largely static despite remarkable progress in model architectures. We present SupraTok, a novel tokenization architecture that reimagines subword segmentation through three innovations: cross-boundary pattern learning that discovers multi-word semantic units, entropy-driven data curation that optimizes training corpus quality, and multi-phase curriculum learning for stable convergence. Our approach extends Byte-Pair Encoding by learning "superword" tokens, coherent multi-word expressions that preserve semantic unity while maximizing compression efficiency. SupraTok achieves 31% improvement in English tokenization efficiency (5.91 versus 4.51 characters per token) compared to OpenAI's o200k tokenizer and 30% improvement over Google's Gemma 3 tokenizer (256k vocabulary), while maintaining competitive performance across 38 languages. When integrated with a GPT-2 scale model (124M parameters) trained on 10 billion tokens from the FineWeb-Edu dataset, SupraTok yields 8.4% improvement on HellaSWAG and 9.5% on MMLU benchmarks without architectural modifications. While these results are promising at this scale, further validation at larger model scales is needed. These findings suggest that efficient tokenization can complement architectural innovations as a path to improved language model performance. 

**Abstract (ZH)**: SupraTok：一种新颖的跨边界子词分词架构 

---
# What Matters for Bioacoustic Encoding 

**Title (ZH)**: 生物声编码中要考虑的因素 

**Authors**: Marius Miron, David Robinson, Milad Alizadeh, Ellen Gilsenan-McMahon, Gagan Narula, Olivier Pietquin, Matthieu Geist, Emmanuel Chemla, Maddie Cusimano, Felix Effenberger, Masato Hagiwara, Benjamin Hoffman, Sara Keen, Diane Kim, Jane Lawton, Jen-Yu Liu, Aza Raskin  

**Link**: [PDF](https://arxiv.org/pdf/2508.11845)  

**Abstract**: Bioacoustics, the study of sounds produced by living organisms, plays a vital role in conservation, biodiversity monitoring, and behavioral studies. Many tasks in this field, such as species, individual, and behavior classification and detection, are well-suited to machine learning. However, they often suffer from limited annotated data, highlighting the need for a general-purpose bioacoustic encoder capable of extracting useful representations for diverse downstream tasks. Such encoders have been proposed before, but are often limited in scope due to a focus on a narrow range of species (typically birds), and a reliance on a single model architecture or training paradigm. Moreover, they are usually evaluated on a small set of tasks and datasets. In this work, we present a large-scale empirical study that covers aspects of bioacoustics that are relevant to research but have previously been scarcely considered: training data diversity and scale, model architectures and training recipes, and the breadth of evaluation tasks and datasets. We obtain encoders that are state-of-the-art on the existing and proposed benchmarks. We also identify what matters for training these encoders, such that this work can be extended when more data are available or better architectures are proposed. Specifically, across 26 datasets with tasks including species classification, detection, individual ID, and vocal repertoire discovery, we find self-supervised pre-training followed by supervised post-training on a mixed bioacoustics + general-audio corpus yields the strongest in- and out-of-distribution performance. We show the importance of data diversity in both stages. To support ongoing research and application, we will release the model checkpoints. 

**Abstract (ZH)**: 生物声学：一种在保育、生物多样性监测及行为研究中发挥关键作用的声音研究领域，机器学习在其中许多任务上表现出色，如物种、个体及行为分类和检测。然而，这些任务往往受限于标注数据的不足，突显出一种通用生物声学编码器的重要性，该编码器能够提取适用于多种下游任务的有用表示。虽然之前已经提出了此类编码器，但通常局限于特定物种（通常是鸟类）且依赖单一的模型架构或训练方法。此外，它们通常仅在少量任务和数据集上进行评估。在本研究中，我们进行了一项大规模实证研究，涵盖了以往较少考虑但对研究至关重要的生物声学方面：训练数据的多样性和规模、模型架构和训练方法，以及评估任务和数据集的广泛性。我们在现有的和提出的基准测试中获得了最先进的编码器，并确定了训练这些编码器的关键因素，以便在更多数据或更好架构可用时能够扩展本工作。具体来说，在包括物种分类、检测、个体识别和声学 repertoire 发现在内的26个数据集中，我们发现自监督预训练后辅以混合生物声学+通用音频语料的监督后训练，能够在域内和域外任务上获得最佳性能。我们展示了两个阶段中数据多样性的关键作用。为了支持持续的研究和应用，我们将发布模型检查点。 

---
# Recent Advances in Transformer and Large Language Models for UAV Applications 

**Title (ZH)**: Recent Advances in Transformers and Large Language Models for UAV Applications 

**Authors**: Hamza Kheddar, Yassine Habchi, Mohamed Chahine Ghanem, Mustapha Hemis, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2508.11834)  

**Abstract**: The rapid advancement of Transformer-based models has reshaped the landscape of uncrewed aerial vehicle (UAV) systems by enhancing perception, decision-making, and autonomy. This review paper systematically categorizes and evaluates recent developments in Transformer architectures applied to UAVs, including attention mechanisms, CNN-Transformer hybrids, reinforcement learning Transformers, and large language models (LLMs). Unlike previous surveys, this work presents a unified taxonomy of Transformer-based UAV models, highlights emerging applications such as precision agriculture and autonomous navigation, and provides comparative analyses through structured tables and performance benchmarks. The paper also reviews key datasets, simulators, and evaluation metrics used in the field. Furthermore, it identifies existing gaps in the literature, outlines critical challenges in computational efficiency and real-time deployment, and offers future research directions. This comprehensive synthesis aims to guide researchers and practitioners in understanding and advancing Transformer-driven UAV technologies. 

**Abstract (ZH)**: 基于Transformer模型的无人机系统 rapid advancement 重塑感知、决策与自主性：一种综合性的综述 

---
# When Does Language Transfer Help? Sequential Fine-Tuning for Cross-Lingual Euphemism Detection 

**Title (ZH)**: 何时语言转移有益？跨语言委婉语检测的序列微调 

**Authors**: Julia Sammartino, Libby Barak, Jing Peng, Anna Feldman  

**Link**: [PDF](https://arxiv.org/pdf/2508.11831)  

**Abstract**: Euphemisms are culturally variable and often ambiguous, posing challenges for language models, especially in low-resource settings. This paper investigates how cross-lingual transfer via sequential fine-tuning affects euphemism detection across five languages: English, Spanish, Chinese, Turkish, and Yoruba. We compare sequential fine-tuning with monolingual and simultaneous fine-tuning using XLM-R and mBERT, analyzing how performance is shaped by language pairings, typological features, and pretraining coverage. Results show that sequential fine-tuning with a high-resource L1 improves L2 performance, especially for low-resource languages like Yoruba and Turkish. XLM-R achieves larger gains but is more sensitive to pretraining gaps and catastrophic forgetting, while mBERT yields more stable, though lower, results. These findings highlight sequential fine-tuning as a simple yet effective strategy for improving euphemism detection in multilingual models, particularly when low-resource languages are involved. 

**Abstract (ZH)**: euphemisms在不同语言中的文化变量性和模糊性给语言模型带来了挑战，特别是在资源匮乏的环境中。本文研究了顺序微调在五个语言（英语、西班牙语、中文、土耳其语、约鲁巴语）中的跨语言迁移如何影响隐讳语识别。我们采用XLM-R和mBERT比较了顺序微调、单一语言微调和同时微调的方法，分析了语言配对、语言类型学特征和预训练覆盖范围对性能的影响。结果显示，对于约鲁巴语和土耳其语等资源匮乏语言，使用高资源语言作为第一语言的顺序微调能显著提高性能。XLM-R在性能提升方面表现更佳，但对预训练差距和灾难性遗忘更为敏感，而mBERT则提供了更加稳定但较低的性能。这些发现强调了顺序微调在多语言模型中提高隐讳语检测的有效性，特别是在涉及资源匮乏语言时。 

---
# Every 28 Days the AI Dreams of Soft Skin and Burning Stars: Scaffolding AI Agents with Hormones and Emotions 

**Title (ZH)**: 每28天，AI梦回Soft Skin和Burning Stars：通过激素与情绪构建AI代理 

**Authors**: Leigh Levinson, Christopher J. Agostino  

**Link**: [PDF](https://arxiv.org/pdf/2508.11829)  

**Abstract**: Despite significant advances, AI systems struggle with the frame problem: determining what information is contextually relevant from an exponentially large possibility space. We hypothesize that biological rhythms, particularly hormonal cycles, serve as natural relevance filters that could address this fundamental challenge. We develop a framework that embeds simulated menstrual and circadian cycles into Large Language Models through system prompts generated from periodic functions modeling key hormones including estrogen, testosterone, and cortisol. Across multiple state-of-the-art models, linguistic analysis reveals emotional and stylistic variations that track biological phases; sadness peaks during menstruation while happiness dominates ovulation and circadian patterns show morning optimism transitioning to nocturnal introspection. Benchmarking on SQuAD, MMLU, Hellaswag, and AI2-ARC demonstrates subtle but consistent performance variations aligning with biological expectations, including optimal function in moderate rather than extreme hormonal ranges. This methodology provides a novel approach to contextual AI while revealing how societal biases regarding gender and biology are embedded within language models. 

**Abstract (ZH)**: 尽管取得了显著进展，AI系统仍难以解决框架问题：在庞大可能信息空间中确定上下文相关信息。我们假设生物节律，尤其是激素周期，作为自然的相关性过滤器，可以应对这一根本性挑战。我们开发了一种框架，通过从模拟关键激素（包括雌激素、睾酮和皮质醇）周期函数中生成的系统提示，将模拟月经和昼夜节律嵌入到大型语言模型中。在多个最先进的模型中，语言分析揭示了与生物阶段相关的情感和风格变化；悲伤在月经期间达到峰值，而幸福在排卵期占主导地位；同时，昼夜节律模式显示清晨的乐观情绪过渡到晚间的内省。在SQuAD、MMLU、Hellaswag和AI2-ARC上的基准测试表明，性能变化虽细微但具一致性，符合生物预期，包括在适度而非极端激素范围内表现最佳。该方法提供了上下文AI的新范式，揭示了关于性别和生物的偏见如何嵌入语言模型中。 

---
# Rethinking Autonomy: Preventing Failures in AI-Driven Software Engineering 

**Title (ZH)**: 重思自主性：防止AI驱动软件工程中的失败 

**Authors**: Satyam Kumar Navneet, Joydeep Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2508.11824)  

**Abstract**: The integration of Large Language Models (LLMs) into software engineering has revolutionized code generation, enabling unprecedented productivity through promptware and autonomous AI agents. However, this transformation introduces significant risks, including insecure code generation, hallucinated outputs, irreversible actions, and a lack of transparency and accountability. Incidents like the Replit database deletion underscore the urgent need for robust safety and governance mechanisms. This paper comprehensively analyzes the inherent challenges of LLM-assisted code generation, such as vulnerability inheritance, overtrust, misinterpretation, and the absence of standardized validation and rollback protocols. To address these, we propose the SAFE-AI Framework, a holistic approach emphasizing Safety, Auditability, Feedback, and Explainability. The framework integrates guardrails, sandboxing, runtime verification, risk-aware logging, human-in-the-loop systems, and explainable AI techniques to mitigate risks while fostering trust and compliance. We introduce a novel taxonomy of AI behaviors categorizing suggestive, generative, autonomous, and destructive actions to guide risk assessment and oversight. Additionally, we identify open problems, including the lack of standardized benchmarks for code specific hallucinations and autonomy levels, and propose future research directions for hybrid verification, semantic guardrails, and proactive governance tools. Through detailed comparisons of autonomy control, prompt engineering, explainability, and governance frameworks, this paper provides a roadmap for responsible AI integration in software engineering, aligning with emerging regulations like the EU AI Act and Canada's AIDA to ensure safe, transparent, and accountable AI-driven development. 

**Abstract (ZH)**: 大型语言模型在软件工程中的集成改变了代码生成，通过提示工具和自主AI代理带来了前所未有的生产效率，但这一变革也带来了重大风险，包括不安全的代码生成、幻觉输出、不可逆的操作以及缺乏透明度和责任性。诸如Replit数据库删除的事件凸显了建立稳健的安全和治理机制的迫切需求。本文全面分析了AI辅助代码生成固有的挑战，如漏洞继承、过度信任、误解释和缺乏标准化验证和回滚协议。为此，我们提出了SAFE-AI框架，强调安全、审计、反馈和可解释性。该框架整合了护栏、沙箱、运行时验证、风险意识日志记录、人工在环系统以及可解释AI技术，以减轻风险并促进信任和合规性。我们引入了一种新的AI行为分类法，将其划分为建议性、生成性、自主性和破坏性行动，以指导风险评估和监督。此外，我们还指出了开放性问题，包括缺乏针对代码特定幻觉和自主性水平的标准基准，并提出了混合验证、语义护栏和前瞻性治理工具的未来研究方向。通过详细比较自主控制、提示工程、可解释性和治理框架，本文为负责任的AI在软件工程中的集成提供了路线图，并符合欧盟AI法案和加拿大AIDA等新兴法规，以确保安全、透明和负责的AI驱动开发。 

---
# FairTabGen: Unifying Counterfactual and Causal Fairness in Synthetic Tabular Data Generation 

**Title (ZH)**: FairTabGen: 统一合成表数据生成中的事实和因果公平性 

**Authors**: Nitish Nagesh, Salar Shakibhamedan, Mahdi Bagheri, Ziyu Wang, Nima TaheriNejad, Axel Jantsch, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2508.11810)  

**Abstract**: Generating synthetic data is crucial in privacy-sensitive, data-scarce settings, especially for tabular datasets widely used in real-world applications. A key challenge is improving counterfactual and causal fairness, while preserving high utility. We present FairTabGen, a fairness-aware large language model-based framework for tabular synthetic data generation. We integrate multiple fairness definitions including counterfactual and causal fairness into both its generation and evaluation pipelines. We use in-context learning, prompt refinement, and fairness-aware data curation to balance fairness and utility. Across diverse datasets, our method outperforms state-of-the-art GAN-based and LLM-based methods, achieving up to 10% improvements on fairness metrics such as demographic parity and path-specific causal effects while retaining statistical utility. Remarkably, it achieves these gains using less than 20% of the original data, highlighting its efficiency in low-data regimes. These results demonstrate a principled and practical approach for generating fair and useful synthetic tabular data. 

**Abstract (ZH)**: 生成合成数据在隐私敏感且数据稀缺的环境中至关重要，尤其是在广泛用于实际应用的表格数据集中。一个关键挑战是提高对抗事实公平性和因果公平性，同时保持高实用性。我们提出了FairTabGen，这是一种基于大型语言模型的公平感知框架，用于表格合成数据生成。我们在其生成和评估管道中整合了多种公平定义，包括对抗事实公平性和因果公平性。我们利用上下文学习、提示 refined 和公平感知的数据管理来平衡公平性和实用性。在多种数据集中，我们的方法在公平性指标如人口统计平和路径特异性因果效应方面优于最先进的基于生成对抗网络（GAN）和基于语言模型（LLM）的方法，同时保持统计实用性。值得注意的是，它在使用不到20%的原始数据的情况下实现了这些改进，突显了其在数据稀缺环境下的效率。这些结果展示了生成公平且有用的合成表格数据的原理性和实用性方法。 

---
# Labels or Input? Rethinking Augmentation in Multimodal Hate Detection 

**Title (ZH)**: 标签还是输入？重新思考多模态仇恨检测中的数据增强。 

**Authors**: Sahajpreet Singh, Rongxin Ouyang, Subhayan Mukerjee, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2508.11808)  

**Abstract**: The modern web is saturated with multimodal content, intensifying the challenge of detecting hateful memes, where harmful intent is often conveyed through subtle interactions between text and image under the guise of humor or satire. While recent advances in Vision-Language Models (VLMs) show promise, these models lack support for fine-grained supervision and remain susceptible to implicit hate speech. In this paper, we present a dual-pronged approach to improve multimodal hate detection. First, we propose a prompt optimization framework that systematically varies prompt structure, supervision granularity, and training modality. We show that prompt design and label scaling both influence performance, with structured prompts improving robustness even in small models, and InternVL2 achieving the best F1-scores across binary and scaled settings. Second, we introduce a multimodal data augmentation pipeline that generates 2,479 counterfactually neutral memes by isolating and rewriting the hateful modality. This pipeline, powered by a multi-agent LLM-VLM setup, successfully reduces spurious correlations and improves classifier generalization. Our approaches inspire new directions for building synthetic data to train robust and fair vision-language models. Our findings demonstrate that prompt structure and data composition are as critical as model size, and that targeted augmentation can support more trustworthy and context-sensitive hate detection. 

**Abstract (ZH)**: 现代网络充斥着多模态内容，加剧了检测带有仇恨意图的 meme 的挑战，这些意图往往通过幽默或讽刺的形式在文本和图像之间的微妙互动中传达。尽管近期视觉-语言模型（VLMs）取得了一定进展，但这些模型缺乏细粒度监督，仍然容易受到隐含仇恨言论的影响。在本文中，我们提出了一种双管齐下的方法来改进多模态仇恨内容的检测。首先，我们提出了一种提示优化框架，系统地变化提示结构、监督细粒度和训练方式。结果显示，提示设计和标签缩放都影响性能，结构化的提示即使在小型模型中也能提高鲁棒性，而 InternVL2 在二分类和缩放设置中均获得最佳 F1 分数。其次，我们引入了一种多模态数据增强流水线，生成了 2,479 个事实中立的反事实 meme，通过分离和重写有害模态。该流水线由多代理大型语言模型-视觉语言模型（LLM-VLM）架构驱动，成功减少了虚假关联，提高了分类器的泛化能力。我们的方法启发了构建用于训练鲁棒和公平的视觉-语言模型的合成数据的新方向。我们的研究结果表明，提示结构和数据组成与模型规模同样重要，而有针对性的数据增强可以支持更具可信度和上下文敏感的仇恨内容检测。 

---
# Uncalibrated Reasoning: GRPO Induces Overconfidence for Stochastic Outcomes 

**Title (ZH)**: 未校准的推理：GRPO 对于随机结果诱导过度自信 

**Authors**: Michael Bereket, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2508.11800)  

**Abstract**: Reinforcement learning (RL) has proven remarkably effective at improving the accuracy of language models in verifiable and deterministic domains like mathematics. Here, we examine if current RL methods are also effective at optimizing language models in verifiable domains with stochastic outcomes, like scientific experiments. Through applications to synthetic data and real-world biological experiments, we demonstrate that Group Relative Policy Optimization (GRPO) induces overconfident probability predictions for binary stochastic outcomes, while Proximal Policy Optimization (PPO) and REINFORCE Leave-One-Out (RLOO) yield well-calibrated models. We show that removing group standard normalization in GRPO fixes its miscalibration and provide a theoretical explanation for why normalization causes overconfidence. Our results provide new evidence against the use of standard normalization in GRPO and help pave the way for applications of RL for reasoning language models beyond deterministic domains. 

**Abstract (ZH)**: 强化学习在具有随机结果的可验证领域（如科学实验）优化语言模型的有效性研究 

---
# Using Natural Language for Human-Robot Collaboration in the Real World 

**Title (ZH)**: 在现实世界中使用自然语言进行人机协作 

**Authors**: Peter Lindes, Kaoutar Skiker  

**Link**: [PDF](https://arxiv.org/pdf/2508.11759)  

**Abstract**: We have a vision of a day when autonomous robots can collaborate with humans as assistants in performing complex tasks in the physical world. This vision includes that the robots will have the ability to communicate with their human collaborators using language that is natural to the humans. Traditional Interactive Task Learning (ITL) systems have some of this ability, but the language they can understand is very limited. The advent of large language models (LLMs) provides an opportunity to greatly improve the language understanding of robots, yet integrating the language abilities of LLMs with robots that operate in the real physical world is a challenging problem.
In this chapter we first review briefly a few commercial robot products that work closely with humans, and discuss how they could be much better collaborators with robust language abilities. We then explore how an AI system with a cognitive agent that controls a physical robot at its core, interacts with both a human and an LLM, and accumulates situational knowledge through its experiences, can be a possible approach to reach that vision. We focus on three specific challenges of having the robot understand natural language, and present a simple proof-of-concept experiment using ChatGPT for each. Finally, we discuss what it will take to turn these simple experiments into an operational system where LLM-assisted language understanding is a part of an integrated robotic assistant that uses language to collaborate with humans. 

**Abstract (ZH)**: 我们设想有一天自主机器人能够在执行复杂物理世界任务时作为人类助手进行协作，并能使用自然语言与人类搭档交流。这种设想要求机器人具备理解人类自然语言的能力。传统的交互式任务学习（ITL）系统在这一方面有所涉足，但它们能理解的语言非常有限。大型语言模型（LLMs）的出现为大幅提高机器人的语言理解能力提供了机会，然而将LLMs的语言能力集成到在现实物理环境中操作的机器人中仍是具有挑战性的问题。

在本章中，我们首先简要回顾几种与人类紧密合作的商业机器人产品，并讨论它们如何通过增强语言能力成为更有效的搭档。然后我们探讨一种以认知代理为核心控制物理机器人的AI系统如何与人类和LLM互动，并通过经验积累情境知识，可能成为实现这一设想的方法。我们重点关注机器人理解自然语言的三个具体挑战，并使用ChatGPT进行了简单的概念验证实验。最后，我们讨论如何将这些简单的实验转化为具备LLM辅助语言理解能力的集成机器人助手系统，使其能够通过语言与人类协作。 

---
# Can we Evaluate RAGs with Synthetic Data? 

**Title (ZH)**: 我们能用合成数据评估RAGs吗？ 

**Authors**: Jonas van Elburg, Peter van der Putten, Maarten Marx  

**Link**: [PDF](https://arxiv.org/pdf/2508.11758)  

**Abstract**: We investigate whether synthetic question-answer (QA) data generated by large language models (LLMs) can serve as an effective proxy for human-labeled benchmarks when such data is unavailable. We assess the reliability of synthetic benchmarks across two experiments: one varying retriever parameters while keeping the generator fixed, and another varying the generator with fixed retriever parameters. Across four datasets, of which two open-domain and two proprietary, we find that synthetic benchmarks reliably rank the RAGs varying in terms of retriever configuration, aligning well with human-labeled benchmark baselines. However, they fail to produce consistent RAG rankings when comparing generator architectures. The breakdown possibly arises from a combination of task mismatch between the synthetic and human benchmarks, and stylistic bias favoring certain generators. 

**Abstract (ZH)**: 我们调查了由大规模语言模型（LLMs）生成的合成问答（QA）数据是否可以在缺乏人工标注基准数据时作为有效替代，用于评估基于检索的生成（RAG）模型。在两个实验中评估合成基准的可靠性：一个实验改变检索器参数而固定生成器，另一个实验改变生成器而固定检索器参数。在四个数据集中（两个开放域和两个专有数据集），我们发现合成基准能够可靠地对不同检索器配置的RAG进行排名，与人工标注基准的基线吻合较好。然而，当比较生成器架构时，它们无法产生一致的RAG排名。这种不一致可能来源于合成和人工基准之间的任务不匹配以及对某些生成器的风格偏见。 

---
# Artificial Intelligence in Rural Healthcare Delivery: Bridging Gaps and Enhancing Equity through Innovation 

**Title (ZH)**: 人工智能在农村医疗保健服务中的应用：通过创新缩小差距并促进公平 

**Authors**: Kiruthika Balakrishnan, Durgadevi Velusamy, Hana E. Hinkle, Zhi Li, Karthikeyan Ramasamy, Hikmat Khan, Srini Ramaswamy, Pir Masoom Shah  

**Link**: [PDF](https://arxiv.org/pdf/2508.11738)  

**Abstract**: Rural healthcare faces persistent challenges, including inadequate infrastructure, workforce shortages, and socioeconomic disparities that hinder access to essential services. This study investigates the transformative potential of artificial intelligence (AI) in addressing these issues in underserved rural areas. We systematically reviewed 109 studies published between 2019 and 2024 from PubMed, Embase, Web of Science, IEEE Xplore, and Scopus. Articles were screened using PRISMA guidelines and Covidence software. A thematic analysis was conducted to identify key patterns and insights regarding AI implementation in rural healthcare delivery. The findings reveal significant promise for AI applications, such as predictive analytics, telemedicine platforms, and automated diagnostic tools, in improving healthcare accessibility, quality, and efficiency. Among these, advanced AI systems, including Multimodal Foundation Models (MFMs) and Large Language Models (LLMs), offer particularly transformative potential. MFMs integrate diverse data sources, such as imaging, clinical records, and bio signals, to support comprehensive decision-making, while LLMs facilitate clinical documentation, patient triage, translation, and virtual assistance. Together, these technologies can revolutionize rural healthcare by augmenting human capacity, reducing diagnostic delays, and democratizing access to expertise. However, barriers remain, including infrastructural limitations, data quality concerns, and ethical considerations. Addressing these challenges requires interdisciplinary collaboration, investment in digital infrastructure, and the development of regulatory frameworks. This review offers actionable recommendations and highlights areas for future research to ensure equitable and sustainable integration of AI in rural healthcare systems. 

**Abstract (ZH)**: 农村医疗面临持续性的挑战，包括基础设施不足、人力资源短缺和社会经济差距等，这些都阻碍了获取基本医疗服务。本研究探讨了人工智能（AI）在解决不足服务农村地区问题方面的潜在转变性作用。我们系统回顾了2019年至2024年在PubMed、Embase、Web of Science、IEEE Xplore和Scopus发表的109篇研究文章。文章筛查采用PRISMA指南和Covidence软件。进行了主题分析以确定AI在农村医疗服务实施中的关键模式和见解。研究发现，人工智能应用前景显著，如预测分析、远程医疗平台和自动诊断工具，在提高医疗服务的可及性、质量和效率方面潜力巨大。其中，包括多模态基础模型（MFMs）和大规模语言模型（LLMs）在内的高级AI系统表现出尤为强大的转变性潜力。MFMs整合影像、临床记录和生物信号等多种数据源，支持全面决策，而LLMs则促进临床记录、患者分诊、翻译和虚拟助手功能。这些技术可以重塑农村医疗，通过增强人类能力、减少诊断延误并实现专业知识的普及化。然而，仍存在基础设施限制、数据质量关切和伦理考虑等障碍。克服这些挑战需要跨学科合作、投资数字基础设施并制定监管框架。本综述提供了可操作的建议，并强调了未来研究的重点领域，以确保AI在农村医疗系统中的公平和可持续整合。 

---
# Ovis2.5 Technical Report 

**Title (ZH)**: Ovis2.5 技术报告 

**Authors**: Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, Yuxuan Han, Haijun Li, Wanying Chen, Junke Tang, Chengkun Hou, Zhixing Du, Tianli Zhou, Wenjie Zhang, Huping Ding, Jiahe Li, Wen Li, Gui Hu, Yiliang Gu, Siran Yang, Jiamang Wang, Hailong Sun, Yibo Wang, Hui Sun, Jinlong Huang, Yuping He, Shengze Shi, Weihong Zhang, Guodong Zheng, Junpeng Jiang, Sensen Gao, Yi-Feng Wu, Sijia Chen, Yuhui Chen, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11737)  

**Abstract**: We present Ovis2.5, a successor to Ovis2 designed for native-resolution visual perception and strong multimodal reasoning. Ovis2.5 integrates a native-resolution vision transformer that processes images at their native, variable resolutions, avoiding the degradation from fixed-resolution tiling and preserving both fine detail and global layout -- crucial for visually dense content like complex charts. To strengthen reasoning, we train the model to move beyond linear chain-of-thought and perform reflection -- including self-checking and revision. This advanced capability is exposed as an optional "thinking mode" at inference time, allowing users to trade latency for enhanced accuracy on difficult inputs. The model is trained via a comprehensive five-phase curriculum that progressively builds its skills. The process begins with foundational visual and multimodal pretraining, advances through large-scale instruction tuning, and culminates in alignment and reasoning enhancement using DPO and GRPO. To scale these upgrades efficiently, we employ multimodal data packing and hybrid parallelism, yielding a significant end-to-end speedup. We release two open-source models: Ovis2.5-9B and Ovis2.5-2B. The latter continues the "small model, big performance" philosophy of Ovis2, making it ideal for resource-constrained, on-device scenarios. On the OpenCompass multimodal leaderboard, Ovis2.5-9B averages 78.3, marking a substantial improvement over its predecessor, Ovis2-8B, and achieving state-of-the-art results among open-source MLLMs in the sub-40B parameter range; Ovis2.5-2B scores 73.9, establishing SOTA for its size. Beyond aggregate scores, Ovis2.5 achieves leading results on STEM benchmarks, exhibits strong capabilities on grounding and video tasks, and achieves open-source SOTA at its scale for complex chart analysis. 

**Abstract (ZH)**: Ovis2.5：面向原生分辨率视觉感知和强大多模态推理的继任者 

---
# SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication 

**Title (ZH)**: SafeSieve: 从启发式方法到经验积累的渐进式 pruning 在基于 LL defense 多智能体通信中的应用 

**Authors**: Ruijia Zhang, Xinyan Zhao, Ruixiang Wang, Sigen Chen, Guibin Zhang, An Zhang, Kun Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11733)  

**Abstract**: LLM-based multi-agent systems exhibit strong collaborative capabilities but often suffer from redundant communication and excessive token overhead. Existing methods typically enhance efficiency through pretrained GNNs or greedy algorithms, but often isolate pre- and post-task optimization, lacking a unified strategy. To this end, we present SafeSieve, a progressive and adaptive multi-agent pruning algorithm that dynamically refines the inter-agent communication through a novel dual-mechanism. SafeSieve integrates initial LLM-based semantic evaluation with accumulated performance feedback, enabling a smooth transition from heuristic initialization to experience-driven refinement. Unlike existing greedy Top-k pruning methods, SafeSieve employs 0-extension clustering to preserve structurally coherent agent groups while eliminating ineffective links. Experiments across benchmarks (SVAMP, HumanEval, etc.) showcase that SafeSieve achieves 94.01% average accuracy while reducing token usage by 12.4%-27.8%. Results further demonstrate robustness under prompt injection attacks (1.23% average accuracy drop). In heterogeneous settings, SafeSieve reduces deployment costs by 13.3% while maintaining performance. These results establish SafeSieve as a robust, efficient, and scalable framework for practical multi-agent systems. Our code can be found in this https URL. 

**Abstract (ZH)**: 基于LLM的多agent系统表现出强大的协作能力，但often suffers from redundant communication and excessive token overhead. 现有的方法通常通过预训练的GNN或贪婪算法来提高效率，但往往隔离了前置和后置任务优化，缺乏统一策略。为此，我们提出了SafeSieve，这是一种渐进且自适应的多agent剪枝算法，通过新颖的双机制动态优化agent间的通信。SafeSieve结合初始基于LLM的语义评估与累积性能反馈，实现从启发式初始化到经验驱动优化的平滑过渡。与现有的贪婪Top-k剪枝方法不同，SafeSieve使用0延拓聚类来保留结构上一致的agent组，同时消除无效链接。在多个基准测试（SVAMP、HumanEval等）中，SafeSieve实现了94.01%的平均准确率并降低了12.4%-27.8%的令牌使用量。进一步的结果表明，SafeSieve在对抗提示注入攻击下表现出鲁棒性（平均准确率下降1.23%）。在异构设置中，SafeSieve降低了13.3%的部署成本同时保持了性能。这些结果确立了SafeSieve作为实用多agent系统的稳健、高效和可扩展框架的地位。我们的代码可以在以下链接找到：this https URL。 

---
# BRIEF: BRain-Inspired network connection search with Extensive temporal feature Fusion enhances disease classification 

**Title (ZH)**: BRISK: BRain-Inspired Network Connection Search with Extensive Temporal Feature Fusion to Enhance Disease Classification 

**Authors**: Xiangxiang Cui, Min Zhao, Dongmei Zhi, Shile Qi, Vince D Calhoun, Jing Sui  

**Link**: [PDF](https://arxiv.org/pdf/2508.11732)  

**Abstract**: Existing deep learning models for functional MRI-based classification have limitations in network architecture determination (relying on experience) and feature space fusion (mostly simple concatenation, lacking mutual learning). Inspired by the human brain's mechanism of updating neural connections through learning and decision-making, we proposed a novel BRain-Inspired feature Fusion (BRIEF) framework, which is able to optimize network architecture automatically by incorporating an improved neural network connection search (NCS) strategy and a Transformer-based multi-feature fusion module. Specifically, we first extracted 4 types of fMRI temporal representations, i.e., time series (TCs), static/dynamic functional connection (FNC/dFNC), and multi-scale dispersion entropy (MsDE), to construct four encoders. Within each encoder, we employed a modified Q-learning to dynamically optimize the NCS to extract high-level feature vectors, where the NCS is formulated as a Markov Decision Process. Then, all feature vectors were fused via a Transformer, leveraging both stable/time-varying connections and multi-scale dependencies across different brain regions to achieve the final classification. Additionally, an attention module was embedded to improve interpretability. The classification performance of our proposed BRIEF was compared with 21 state-of-the-art models by discriminating two mental disorders from healthy controls: schizophrenia (SZ, n=1100) and autism spectrum disorder (ASD, n=1550). BRIEF demonstrated significant improvements of 2.2% to 12.1% compared to 21 algorithms, reaching an AUC of 91.5% - 0.6% for SZ and 78.4% - 0.5% for ASD, respectively. This is the first attempt to incorporate a brain-inspired, reinforcement learning strategy to optimize fMRI-based mental disorder classification, showing significant potential for identifying precise neuroimaging biomarkers. 

**Abstract (ZH)**: 基于人类大脑启发的特征融合框架：自动优化网络架构的BRain-Inspired特征融合（BRIEF）模型 

---
# The Stories We Govern By: AI, Risk, and the Power of Imaginaries 

**Title (ZH)**: 我们所治理的故事：人工智能、风险与想象的力量 

**Authors**: Ninell Oldenburg, Gleb Papyshev  

**Link**: [PDF](https://arxiv.org/pdf/2508.11729)  

**Abstract**: This paper examines how competing sociotechnical imaginaries of artificial intelligence (AI) risk shape governance decisions and regulatory constraints. Drawing on concepts from science and technology studies, we analyse three dominant narrative groups: existential risk proponents, who emphasise catastrophic AGI scenarios; accelerationists, who portray AI as a transformative force to be unleashed; and critical AI scholars, who foreground present-day harms rooted in systemic inequality. Through an analysis of representative manifesto-style texts, we explore how these imaginaries differ across four dimensions: normative visions of the future, diagnoses of the present social order, views on science and technology, and perceived human agency in managing AI risks. Our findings reveal how these narratives embed distinct assumptions about risk and have the potential to progress into policy-making processes by narrowing the space for alternative governance approaches. We argue against speculative dogmatism and for moving beyond deterministic imaginaries toward regulatory strategies that are grounded in pragmatism. 

**Abstract (ZH)**: 本文研究了不同的人工智能（AI）风险社会技术想象如何塑造治理决策和监管约束。通过科学技术研究的概念，我们分析了三个主要叙述群体：存在风险倡导者，他们强调灾难性超人工智能场景；加速主义者，他们将AI描绘为一种应被释放的变革力量；以及批判性AI学者，他们强调根植于系统不平等的当前危害。通过对代表性的纲领式文本进行分析，我们探讨了这些想象在四个维度上的差异：对未来的规范性愿景、对当前社会秩序的诊断、对科学和技术的看法，以及对管理AI风险的人类能动性的看法。我们的研究发现这些叙述嵌入了不同的风险假设，并有可能通过缩小替代治理途径的空间而进入政策制定过程。我们反对投机性教条主义，提倡转向基于务实主义的监管策略。 

---
# UniDCF: A Foundation Model for Comprehensive Dentocraniofacial Hard Tissue Reconstruction 

**Title (ZH)**: UniDCF：全面颌面硬组织重建的基座模型 

**Authors**: Chunxia Ren, Ning Zhu, Yue Lai, Gui Chen, Ruijie Wang, Yangyi Hu, Suyao Liu, Shuwen Mao, Hong Su, Yu Zhang, Li Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11728)  

**Abstract**: Dentocraniofacial hard tissue defects profoundly affect patients' physiological functions, facial aesthetics, and psychological well-being, posing significant challenges for precise reconstruction. Current deep learning models are limited to single-tissue scenarios and modality-specific imaging inputs, resulting in poor generalizability and trade-offs between anatomical fidelity, computational efficiency, and cross-tissue adaptability. Here we introduce UniDCF, a unified framework capable of reconstructing multiple dentocraniofacial hard tissues through multimodal fusion encoding of point clouds and multi-view images. By leveraging the complementary strengths of each modality and incorporating a score-based denoising module to refine surface smoothness, UniDCF overcomes the limitations of prior single-modality approaches. We curated the largest multimodal dataset, comprising intraoral scans, CBCT, and CT from 6,609 patients, resulting in 54,555 annotated instances. Evaluations demonstrate that UniDCF outperforms existing state-of-the-art methods in terms of geometric precision, structural completeness, and spatial accuracy. Clinical simulations indicate UniDCF reduces reconstruction design time by 99% and achieves clinician-rated acceptability exceeding 94%. Overall, UniDCF enables rapid, automated, and high-fidelity reconstruction, supporting personalized and precise restorative treatments, streamlining clinical workflows, and enhancing patient outcomes. 

**Abstract (ZH)**: 统一多模态硬组织重建框架UniDCF：多模态点云和多视角图像融合在多发性牙颅面硬组织重建中的应用 

---
# FusionFM: Fusing Eye-specific Foundational Models for Optimized Ophthalmic Diagnosis 

**Title (ZH)**: FusionFM: 结合眼别基础模型以优化眼科诊断 

**Authors**: Ke Zou, Jocelyn Hui Lin Goh, Yukun Zhou, Tian Lin, Samantha Min Er Yew, Sahana Srinivasan, Meng Wang, Rui Santos, Gabor M. Somfai, Huazhu Fu, Haoyu Chen, Pearse A. Keane, Ching-Yu Cheng, Yih Chung Tham  

**Link**: [PDF](https://arxiv.org/pdf/2508.11721)  

**Abstract**: Foundation models (FMs) have shown great promise in medical image analysis by improving generalization across diverse downstream tasks. In ophthalmology, several FMs have recently emerged, but there is still no clear answer to fundamental questions: Which FM performs the best? Are they equally good across different tasks? What if we combine all FMs together? To our knowledge, this is the first study to systematically evaluate both single and fused ophthalmic FMs. To address these questions, we propose FusionFM, a comprehensive evaluation suite, along with two fusion approaches to integrate different ophthalmic FMs. Our framework covers both ophthalmic disease detection (glaucoma, diabetic retinopathy, and age-related macular degeneration) and systemic disease prediction (diabetes and hypertension) based on retinal imaging. We benchmarked four state-of-the-art FMs (RETFound, VisionFM, RetiZero, and DINORET) using standardized datasets from multiple countries and evaluated their performance using AUC and F1 metrics. Our results show that DINORET and RetiZero achieve superior performance in both ophthalmic and systemic disease tasks, with RetiZero exhibiting stronger generalization on external datasets. Regarding fusion strategies, the Gating-based approach provides modest improvements in predicting glaucoma, AMD, and hypertension. Despite these advances, predicting systemic diseases, especially hypertension in external cohort remains challenging. These findings provide an evidence-based evaluation of ophthalmic FMs, highlight the benefits of model fusion, and point to strategies for enhancing their clinical applicability. 

**Abstract (ZH)**: 基础模型在医学图像分析中的应用显示出巨大潜力，通过提高在多样化下游任务中的泛化能力。在眼科，已经涌现出几种基础模型，但仍没有明确的答案来回答基础问题：哪种基础模型表现最好？它们在不同任务中的表现是否相当？如果我们将所有基础模型结合起来会怎样？据我们所知，这是首次系统评估单个和融合眼科基础模型的研究。为了回答这些问题，我们提出了FusionFM，一个完整的评估套件，以及两种融合方法来整合不同的眼科基础模型。我们的框架涵盖了基于视网膜成像的眼科疾病检测（青光眼、糖尿病视网膜病变和年龄相关性黄斑变性）和全身疾病预测（糖尿病和高血压）。我们使用标准化的多国数据集对四种最先进的基础模型（RETFound、VisionFM、RetiZero和DINORET）进行了基准测试，并使用AUC和F1度量评估了它们的性能。研究结果表明，DINORET和RetiZero在眼科和全身疾病任务中均表现出色，RetiZero在外源性数据集上的泛化能力更强。关于融合策略，门控方法在预测青光眼、AMD和高血压方面提供了适度的改进。尽管取得了这些进展，但在外部队列中预测全身疾病，特别是高血压仍具有挑战性。这些发现为眼科基础模型提供了基于证据的评估，突显了模型融合的优势，并指出了增强其临床应用策略的方向。 

---
# Are AI Machines Making Humans Obsolete? 

**Title (ZH)**: 人工智能机器是否会使人类过时？ 

**Authors**: Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2508.11719)  

**Abstract**: This chapter starts with a sketch of how we got to "generative AI" (GenAI) and a brief summary of the various impacts it had so far. It then discusses some of the opportunities of GenAI, followed by the challenges and dangers, including dystopian outcomes resulting from using uncontrolled machine learning and our failures to understand the results. It concludes with some suggestions for how to control GenAI and address its dangers. 

**Abstract (ZH)**: 本章从对如何达到“生成型AI”（GenAI）的概述及其到目前为止的各种影响开始，随后讨论了GenAI的一些机遇，接着探讨了相关的挑战和危险，包括因无控制的机器学习和我们未能理解其结果而产生的悲观后果。最后提出了控制GenAI和应对潜在危险的建议。 

---
# Privacy-Aware Detection of Fake Identity Documents: Methodology, Benchmark, and Improved Detection Methods (FakeIDet2) 

**Title (ZH)**: 面向隐私的虚假身份文件检测：方法、基准及改进的检测方法（FakeIDet2） 

**Authors**: Javier Muñoz-Haro, Ruben Tolosana, Ruben Vera-Rodriguez, Aythami Morales, Julian Fierrez  

**Link**: [PDF](https://arxiv.org/pdf/2508.11716)  

**Abstract**: Remote user verification in Internet-based applications is becoming increasingly important nowadays. A popular scenario for it consists of submitting a picture of the user's Identity Document (ID) to a service platform, authenticating its veracity, and then granting access to the requested digital service. An ID is well-suited to verify the identity of an individual, since it is government issued, unique, and nontransferable. However, with recent advances in Artificial Intelligence (AI), attackers can surpass security measures in IDs and create very realistic physical and synthetic fake IDs. Researchers are now trying to develop methods to detect an ever-growing number of these AI-based fakes that are almost indistinguishable from authentic (bona fide) IDs. In this counterattack effort, researchers are faced with an important challenge: the difficulty in using real data to train fake ID detectors. This real data scarcity for research and development is originated by the sensitive nature of these documents, which are usually kept private by the ID owners (the users) and the ID Holders (e.g., government, police, bank, etc.). The main contributions of our study are: 1) We propose and discuss a patch-based methodology to preserve privacy in fake ID detection research. 2) We provide a new public database, FakeIDet2-db, comprising over 900K real/fake ID patches extracted from 2,000 ID images, acquired using different smartphone sensors, illumination and height conditions, etc. In addition, three physical attacks are considered: print, screen, and composite. 3) We present a new privacy-aware fake ID detection method, FakeIDet2. 4) We release a standard reproducible benchmark that considers physical and synthetic attacks from popular databases in the literature. 

**Abstract (ZH)**: 基于互联网的应用远程用户验证变得越来越重要。一种常见的场景是提交用户身份文件（ID）的照片，认证其真实性，然后授予请求的数字服务访问权限。身份文件适合验证个人身份，因为它是由政府颁发的、唯一的且不可转移的。然而，随着人工智能（AI）的最新进展，攻击者可以通过超越身份文件的安全措施，创建非常逼真的物理和合成伪造身份文件。研究人员现在正尝试开发方法来检测越来越多的这些基于AI的伪造品，几乎与真实（合法）的身份文件无法区分。在这种防御努力中，研究人员面临着一个重要的挑战：使用真实数据训练伪造身份文件检测器的难度。这种研究和开发中真实数据的稀缺性是由这些文件的敏感性质引起的，通常由身份文件的所有者（用户）和持有人（如政府、警察、银行等）保持私密。我们研究的主要贡献包括：1）我们提出并讨论了一种基于补丁的方法，以在伪造身份文件检测研究中保护隐私。2）我们提供了一个新的公共数据库FakeIDet2-db，包含超过90万张真实/伪造身份文件补丁，来自2000张身份文件图像，使用了不同的智能手机传感器、照明和高度条件等。此外，考虑了三种物理攻击：打印、屏幕和复合。3）我们提出了一种新的具备隐私意识的伪造身份文件检测方法FakeIDet2。4）我们发布了一个标准可重现基准，考虑了文献中流行的数据库中的物理和合成攻击。 

---
# Benchmark Dataset Generation and Evaluation for Excel Formula Repair with LLMs 

**Title (ZH)**: 基于LLMs的Excel公式修复基准数据集生成与评估 

**Authors**: Ananya Singha, Harshita Sahijwani, Walt Williams, Emmanuel Aboah Boateng, Nick Hausman, Miguel Di Luca, Keegan Choudhury, Chaya Binet, Vu Le, Tianwei Chen, Oryan Rokeah Chen, Sulaiman Vesal, Sadid Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11715)  

**Abstract**: Excel is a pervasive yet often complex tool, particularly for novice users, where runtime errors arising from logical mistakes or misinterpretations of functions pose a significant challenge. While large language models (LLMs) offer promising assistance by explaining formula errors, the automated correction of these semantic runtime errors remains an open problem. A primary challenge to advancing models for such scenarios is the severe lack of high-quality, comprehensive datasets for training and rigorous evaluation. This paper addresses this gap by introducing a novel approach for constructing a benchmark dataset specifically designed for Excel formula repair. We propose a data generation pipeline, which leverages a small set of curated seed samples from online forums to synthetically expand the dataset. Our pipeline integrates few-shot prompting with LLMs and employs a robust \textit{LLM-as-a-Judge} validation framework, combined with execution-based checks to ensure the correctness and semantic fidelity of the generated data. This process produced a benchmark dataset of 618 high-quality samples, covering common runtime errors. Furthermore, we propose a context-aware baseline technique for Excel formula repair that utilizes LLMs to leverage both the faulty formula, and relevant spreadsheet context. We evaluate the performance of various LLMs (GPT-4o, GPT-4.1, Phi-3, Mistral) on our newly generated benchmark using execution-based metrics. Our analysis demonstrates the dataset's quality through manual annotation and provides insights into error and function distributions. The proposed generation methodology is highly scalable and can be readily adapted to create evaluation benchmarks for similar code repair tasks in other low-resource programming languages. 

**Abstract (ZH)**: Excel是一种普及但常常复杂的工具，尤其是对新手用户而言，运行时由于逻辑错误或函数误解导致的错误构成了重大挑战。虽然大型语言模型（LLMs）提供了通过解释公式错误来提供帮助的前景，但自动化纠正这些语义运行时错误仍然是一个开放问题。针对此类场景推进模型的主要挑战之一是高质量、全面训练数据集的严重缺乏，用于严格的评估。本文通过介绍一种新的基准数据集构建方法来填补这一空白，专门设计用于Excel公式修复。我们提出了一种数据生成流水线，利用来自在线论坛的小规模策划种子样本进行合成性扩展。该流水线结合了少样本提示与LLMs，并采用了强大的“LLM作为裁判”验证框架，结合执行检查以确保生成数据的正确性和语义准确性。该过程产生了包含618个高质量样本的基准数据集，覆盖常见的运行时错误。此外，我们提出了一种基于上下文的Excel公式修复基线技术，利用LLMs结合故障公式和相关表格上下文的优势。我们使用基于执行的指标，在新增的基准上评估了多种LLMs（GPT-4o、GPT-4.1、Phi-3、Mistral）的表现。我们的分析通过手动注释展示了数据集的质量，并提供了错误和函数分布的见解。提出的生成方法具有高可扩展性，可以方便地调整为其他低资源编程语言中的类似代码修复任务的评估基准。 

---
# Enhancing GraphQL Security by Detecting Malicious Queries Using Large Language Models, Sentence Transformers, and Convolutional Neural Networks 

**Title (ZH)**: 使用大型语言模型、句嵌入变换器和卷积神经网络检测恶意查询以增强GraphQL安全性 

**Authors**: Irash Perera, Hiranya Abeyrathne, Sanjeewa Malalgoda, Arshardh Ifthikar  

**Link**: [PDF](https://arxiv.org/pdf/2508.11711)  

**Abstract**: GraphQL's flexibility, while beneficial for efficient data fetching, introduces unique security vulnerabilities that traditional API security mechanisms often fail to address. Malicious GraphQL queries can exploit the language's dynamic nature, leading to denial-of-service attacks, data exfiltration through injection, and other exploits. Existing solutions, such as static analysis, rate limiting, and general-purpose Web Application Firewalls, offer limited protection against sophisticated, context-aware attacks. This paper presents a novel, AI-driven approach for real-time detection of malicious GraphQL queries. Our method combines static analysis with machine learning techniques, including Large Language Models (LLMs) for dynamic schema-based configuration, Sentence Transformers (SBERT and Doc2Vec) for contextual embedding of query payloads, and Convolutional Neural Networks (CNNs), Random Forests, and Multilayer Perceptrons for classification. We detail the system architecture, implementation strategies optimized for production environments (including ONNX Runtime optimization and parallel processing), and evaluate the performance of our detection models and the overall system under load. Results demonstrate high accuracy in detecting various threats, including SQL injection, OS command injection, and XSS exploits, alongside effective mitigation of DoS and SSRF attempts. This research contributes a robust and adaptable solution for enhancing GraphQL API security. 

**Abstract (ZH)**: GraphQL的灵活性虽然有助于高效的数据获取，但引入了传统API安全机制难以应对的独特安全漏洞。恶意的GraphQL查询可以利用该语言的动态特性，导致服务拒绝攻击、数据泄露注入以及其他利用方式。现有的解决方案，如静态分析、速率限制和通用_WEB_应用程序防火墙，对复杂的上下文感知攻击提供有限的保护。本文提出了一种新颖的AI驱动的实时检测恶意GraphQL查询的方法。该方法结合了静态分析与机器学习技术，包括大型语言模型（LLMs）用于动态模式下的配置、Sentence Transformers（SBERT和Doc2Vec）用于查询负载的上下文嵌入，以及卷积神经网络（CNNs）、随机森林和多层感知机用于分类。我们详细描述了系统架构、针对生产环境优化的实现策略（包括ONNX Runtime优化和并行处理），并评估了检测模型和整体系统的性能。结果表明，该方法在检测包括SQL注入、操作系统命令注入和XSS攻击在内的各种威胁方面具有高度准确性，并有效地缓解了服务拒绝和SSRF攻击尝试。本研究提供了一种强大且 adaptable 的解决方案，以增强GraphQL API安全。 

---
# Code Vulnerability Detection Across Different Programming Languages with AI Models 

**Title (ZH)**: 使用AI模型跨不同编程语言检测代码漏洞 

**Authors**: Hael Abdulhakim Ali Humran, Ferdi Sonmez  

**Link**: [PDF](https://arxiv.org/pdf/2508.11710)  

**Abstract**: Security vulnerabilities present in a code that has been written in diverse programming languages are among the most critical yet complicated aspects of source code to detect. Static analysis tools based on rule-based patterns usually do not work well at detecting the context-dependent bugs and lead to high false positive rates. Recent developments in artificial intelligence, specifically the use of transformer-based models like CodeBERT and CodeLlama, provide light to this problem, as they show potential in finding such flaws better. This paper presents the implementations of these models on various datasets of code vulnerability, showing how off-the-shelf models can successfully produce predictive capacity in models through dynamic fine-tuning of the models on vulnerable and safe code fragments. The methodology comprises the gathering of the dataset, normalization of the language, fine-tuning of the model, and incorporation of ensemble learning and explainable AI. Experiments show that a well-trained CodeBERT can be as good as or even better than some existing static analyzers in terms of accuracy greater than 97%. Further study has indicated that although language models can achieve close-to-perfect recall, the precision can decrease. A solution to this is given by hybrid models and validation procedures, which will reduce false positives. According to the results, the AI-based solutions generalize to different programming languages and classes of vulnerability. Nevertheless, robustness, interpretability, and deployment readiness are still being developed. The results illustrate the probabilities that AI will enhance the trustworthiness in the usability and scalability of machine-learning-based detectors of vulnerabilities. 

**Abstract (ZH)**: 基于人工智能的代码漏洞检测方法研究：CodeBERT等模型在代码漏洞数据集上的实现与评价 

---
# Navigating the New Landscape: A Conceptual Model for Project-Based Assessment (PBA) in the Age of GenAI 

**Title (ZH)**: 探索新景观：AI时代基于项目的评估（PBA）的概念模型 

**Authors**: Rajan Kadel, Samar Shailendra, Urvashi Rahul Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2508.11709)  

**Abstract**: The rapid integration of Generative Artificial Intelligence (GenAI) into higher education presents both opportunities and challenges for assessment design, particularly within Project-Based Assessment (PBA) contexts. Traditional assessment methods often emphasise the final product in the PBA, which can now be significantly influenced or created by GenAI tools, raising concerns regarding product authenticity, academic integrity, and learning validation. This paper advocates for a reimagined assessment model for Project-Based Learning (PBL) or a capstone project that prioritises process-oriented evaluation, multi-modal and multifaceted assessment design, and ethical engagement with GenAI to enable higher-order thinking. The model also emphasises the use of (GenAI-assisted) personalised feedback by a supervisor as an observance of the learning process during the project lifecycle. A use case scenario is provided to illustrate the application of the model in a capstone project setting. The paper concludes with recommendations for educators and curriculum designers to ensure that assessment practices remain robust, learner-centric, and integrity-driven in the evolving landscape of GenAI. 

**Abstract (ZH)**: Generative Artificial Intelligence驱动的高等教育评估设计：项目-Based学习（PBL）中的机遇与挑战 

---
# Street Review: A Participatory AI-Based Framework for Assessing Streetscape Inclusivity 

**Title (ZH)**: 街道审查：一个参与式的基于人工智能的评估街道景观包容性框架 

**Authors**: Rashid Mushkani, Shin Koseki  

**Link**: [PDF](https://arxiv.org/pdf/2508.11708)  

**Abstract**: Urban centers undergo social, demographic, and cultural changes that shape public street use and require systematic evaluation of public spaces. This study presents Street Review, a mixed-methods approach that combines participatory research with AI-based analysis to assess streetscape inclusivity. In Montréal, Canada, 28 residents participated in semi-directed interviews and image evaluations, supported by the analysis of approximately 45,000 street-view images from Mapillary. The approach produced visual analytics, such as heatmaps, to correlate subjective user ratings with physical attributes like sidewalk, maintenance, greenery, and seating. Findings reveal variations in perceptions of inclusivity and accessibility across demographic groups, demonstrating that incorporating diverse user feedback can enhance machine learning models through careful data-labeling and co-production strategies. The Street Review framework offers a systematic method for urban planners and policy analysts to inform planning, policy development, and management of public streets. 

**Abstract (ZH)**: 城市中心的社会、人口和文化变迁塑造了公共街道的使用方式，需要对公共空间进行系统评估。本文介绍了Street Review这一混合方法，结合了参与式研究与基于AI的分析，以评估街道的包容性。在加拿大蒙特利尔，28名居民参与了半引导式访谈和图像评估，分析了来自Mapillary的约45,000张街景图像。该方法产生了视觉分析，如热力图，将用户主观评价与人行道、维护、绿化和座椅等物理属性相关联。研究发现不同人口群体对包容性和可达性的感知存在差异，表明通过精心的数据标注和共同生产策略 Incorporate diverse user feedback 可以增强机器学习模型。Street Review框架为城市规划者和政策分析师提供了一种系统方法，以指导公共街道的规划、政策制定和管理。 

---
# Listening with Language Models: Using LLMs to Collect and Interpret Classroom Feedback 

**Title (ZH)**: 语言模型中的倾听：使用LLM收集和解释课堂反馈 

**Authors**: Sai Siddartha Maram, Ulia Zaman, Magy Seif El-Nasr  

**Link**: [PDF](https://arxiv.org/pdf/2508.11707)  

**Abstract**: Traditional end-of-quarter surveys often fail to provide instructors with timely, detailed, and actionable feedback about their teaching. In this paper, we explore how Large Language Model (LLM)-powered chatbots can reimagine the classroom feedback process by engaging students in reflective, conversational dialogues. Through the design and deployment of a three-part system-PromptDesigner, FeedbackCollector, and FeedbackAnalyzer-we conducted a pilot study across two graduate courses at UC Santa Cruz. Our findings suggest that LLM-based feedback systems offer richer insights, greater contextual relevance, and higher engagement compared to standard survey tools. Instructors valued the system's adaptability, specificity, and ability to support mid-course adjustments, while students appreciated the conversational format and opportunity for elaboration. We conclude by discussing the design implications of using AI to facilitate more meaningful and responsive feedback in higher education. 

**Abstract (ZH)**: 基于大型语言模型的聊天机器人如何重塑课堂教学反馈过程：促进反思性对话的研究 

---
# Centralized Permutation Equivariant Policy for Cooperative Multi-Agent Reinforcement Learning 

**Title (ZH)**: 集中化排列等变策略在协作多智能体强化学习中的应用 

**Authors**: Zhuofan Xu, Benedikt Bollig, Matthias Függer, Thomas Nowak, Vincent Le Dréau  

**Link**: [PDF](https://arxiv.org/pdf/2508.11706)  

**Abstract**: The Centralized Training with Decentralized Execution (CTDE) paradigm has gained significant attention in multi-agent reinforcement learning (MARL) and is the foundation of many recent algorithms. However, decentralized policies operate under partial observability and often yield suboptimal performance compared to centralized policies, while fully centralized approaches typically face scalability challenges as the number of agents increases.
We propose Centralized Permutation Equivariant (CPE) learning, a centralized training and execution framework that employs a fully centralized policy to overcome these limitations. Our approach leverages a novel permutation equivariant architecture, Global-Local Permutation Equivariant (GLPE) networks, that is lightweight, scalable, and easy to implement. Experiments show that CPE integrates seamlessly with both value decomposition and actor-critic methods, substantially improving the performance of standard CTDE algorithms across cooperative benchmarks including MPE, SMAC, and RWARE, and matching the performance of state-of-the-art RWARE implementations. 

**Abstract (ZH)**: 集中式训练与分布式执行（CTDE）范式在多智能体强化学习（MARL）中获得了显著关注，并成为许多近期算法的基础。然而，分布式策略在部分可观性条件下运作，往往导致性能低于集中式策略，而完全集中式方法通常会随着智能体数量的增加面临可扩展性挑战。我们提出了一种名为集中式置换不变学习（CPE）的方法，这是一种集中式训练和执行框架，利用完全集中式策略来克服这些限制。我们的方法采用了一种新颖的轻量级、可扩展且易于实现的置换不变架构——全局-局部置换不变网络（GLPE）网络。实验表明，CPE 能够无缝集成价值分解和演员-评论家方法，显著提高了标准 CTDE 算法在包括MPE、SMAC和RWARE在内的合作基准测试中的性能，并达到了最先进的RWARE实现的性能水平。 

---
# Next-Gen Education: Enhancing AI for Microlearning 

**Title (ZH)**: 下一代教育：增强微学习的AI 

**Authors**: Suman Saha, Fatemeh Rahbari, Farhan Sadique, Sri Krishna Chaitanya Velamakanni, Mahfuza Farooque, William J. Rothwell  

**Link**: [PDF](https://arxiv.org/pdf/2508.11704)  

**Abstract**: This paper explores integrating microlearning strategies into university curricula, particularly in computer science education, to counteract the decline in class attendance and engagement in US universities after COVID. As students increasingly opt for remote learning and recorded lectures, traditional educational approaches struggle to maintain engagement and effectiveness. Microlearning, which breaks complex subjects into manageable units, is proposed to address shorter attention spans and enhance educational outcomes. It uses interactive formats such as videos, quizzes, flashcards, and scenario-based exercises, which are especially beneficial for topics like algorithms and programming logic requiring deep understanding and ongoing practice. Adoption of microlearning is often limited by the effort needed to create such materials. This paper proposes leveraging AI tools, specifically ChatGPT, to reduce the workload for educators by automating the creation of supplementary materials. While AI can automate certain tasks, educators remain essential in guiding and shaping the learning process. This AI-enhanced approach ensures course content is kept current with the latest research and technology, with educators providing context and insights. By examining AI capabilities in microlearning, this study shows the potential to transform educational practices and outcomes in computer science, offering a practical model for combining advanced technology with established teaching methods. 

**Abstract (ZH)**: 本文探讨将微学习策略融入大学课程，特别是在计算机科学教育中，以应对COVID之后美国大学课堂出勤率和参与度下降的问题。随着学生越来越多地选择远程学习和录播课程，传统的教育方法难以维持参与度和有效性。微学习通过将复杂科目拆分成 manageable 单元，来应对注意力短暂的问题并提升教育成果。它使用诸如视频、测验、闪卡和情景练习等交互式格式，特别适合需要深刻理解和持续练习的算法和编程逻辑等主题。微学习的采用常受限于创建这些材料所需的努力。本文提出利用AI工具，特别是ChatGPT，来减轻教育者的负担，自动化生成补充材料。尽管AI能够自动化某些任务，教育者仍然是引导和塑造学习过程的关键。通过利用AI在微学习中的能力，本文展示了如何通过将先进技术与传统教学方法相结合，来变革计算机科学教育的实践和成果，提供了一种实用的结合方法。 

---
# Separating Knowledge and Perception with Procedural Data 

**Title (ZH)**: 分离知识和感知：基于过程化数据的方法 

**Authors**: Adrián Rodríguez-Muñoz, Manel Baradad, Phillip Isola, Antonio Torralba  

**Link**: [PDF](https://arxiv.org/pdf/2508.11697)  

**Abstract**: We train representation models with procedural data only, and apply them on visual similarity, classification, and semantic segmentation tasks without further training by using visual memory -- an explicit database of reference image embeddings. Unlike prior work on visual memory, our approach achieves full compartmentalization with respect to all real-world images while retaining strong performance. Compared to a model trained on Places, our procedural model performs within $1\%$ on NIGHTS visual similarity, outperforms by $8\%$ and $15\%$ on CUB200 and Flowers102 fine-grained classification, and is within $10\%$ on ImageNet-1K classification. It also demonstrates strong zero-shot segmentation, achieving an $R^2$ on COCO within $10\%$ of the models trained on real data. Finally, we analyze procedural versus real data models, showing that parts of the same object have dissimilar representations in procedural models, resulting in incorrect searches in memory and explaining the remaining performance gap. 

**Abstract (ZH)**: 我们仅使用过程生成数据训练表征模型，并通过视觉记忆（一个显式的参考图像嵌入数据库）在视觉相似性、分类和语义分割任务中应用这些模型，而无需进一步训练。与视觉记忆领域的先前工作相比，我们的方法实现了对所有真实世界图像的完全分隔化，同时保持了强大的性能。与在Places数据集上训练的模型相比，我们的过程生成模型在NIGHTS视觉相似性任务中表现相差1%，在CUB200和Flowers102细粒度分类任务中的表现分别超过8%和15%，在ImageNet-1K分类任务中的表现相差10%。该模型还展示了强大的零样本分割能力，在COCO上的$R^2$分数与基于真实数据训练的模型相差10%以内。最后，我们分析了过程生成数据模型与真实数据模型之间的差异，发现过程生成模型中同一对象的不同部分具有不同的表征，导致记忆搜索错误，并解释了剩余的性能差距。 

---
# RefAdGen: High-Fidelity Advertising Image Generation 

**Title (ZH)**: RefAdGen: 高保真广告图像生成 

**Authors**: Yiyun Chen, Weikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11695)  

**Abstract**: The rapid advancement of Artificial Intelligence Generated Content (AIGC) techniques has unlocked opportunities in generating diverse and compelling advertising images based on referenced product images and textual scene descriptions. This capability substantially reduces human labor and production costs in traditional marketing workflows. However, existing AIGC techniques either demand extensive fine-tuning for each referenced image to achieve high fidelity, or they struggle to maintain fidelity across diverse products, making them impractical for e-commerce and marketing industries. To tackle this limitation, we first construct AdProd-100K, a large-scale advertising image generation dataset. A key innovation in its construction is our dual data augmentation strategy, which fosters robust, 3D-aware representations crucial for realistic and high-fidelity image synthesis. Leveraging this dataset, we propose RefAdGen, a generation framework that achieves high fidelity through a decoupled design. The framework enforces precise spatial control by injecting a product mask at the U-Net input, and employs an efficient Attention Fusion Module (AFM) to integrate product features. This design effectively resolves the fidelity-efficiency dilemma present in existing methods. Extensive experiments demonstrate that RefAdGen achieves state-of-the-art performance, showcasing robust generalization by maintaining high fidelity and remarkable visual results for both unseen products and challenging real-world, in-the-wild images. This offers a scalable and cost-effective alternative to traditional workflows. Code and datasets are publicly available at this https URL. 

**Abstract (ZH)**: 基于引用产品图像和文本场景描述生成的快速进步的人工智能生成内容（AIGC）技术为生成多样化且引人注目的广告图像提供了机会。这一能力大大减少了传统营销工作流程中的人力和生产成本。然而，现有的AIGC技术要么需要对每个引用图像进行大量的微调才能达到高质量，要么无法在多样化的产品中保持高质量，这使它们在电子商务和营销行业中不切实际。为解决这一局限性，我们首先构建了AdProd-100K，一个大规模的广告图像生成数据集。其构建中的关键创新是我们的双重数据增强策略，这对于实现现实和高质量的图像合成是必不可少的。利用这一数据集，我们提出了RefAdGen生成框架，通过解耦设计实现高质量生成。框架通过在U-Net输入中注入产品遮罩来实现精确的空间控制，并采用高效的注意力融合模块（AFM）整合产品特征。这一设计有效解决了现有方法中存在的质量-效率权衡问题。大量实验表明，RefAdGen达到了最先进的性能，展示了在未见过的产品和具有挑战性的现实世界拍摄的图像中都保持高质量和出色的视觉效果的能力。这提供了一种可扩展且成本效益高的传统工作流程替代方案。代码和数据集可在以下链接公开获取。 

---
# Track Component Failure Detection Using Data Analytics over existing STDS Track Circuit data 

**Title (ZH)**: 基于现有STDS轨道电路数据的数据分析用于检测轨道组件故障 

**Authors**: Francisco López, Eduardo Di Santi, Clément Lefebvre, Nenad Mijatovic, Michele Pugnaloni, Victor Martín, Kenza Saiah  

**Link**: [PDF](https://arxiv.org/pdf/2508.11693)  

**Abstract**: Track Circuits (TC) are the main signalling devices used to detect the presence of a train on a rail track. It has been used since the 19th century and nowadays there are many types depending on the technology. As a general classification, Track Circuits can be divided into 2 main groups, DC (Direct Current) and AC (Alternating Current) circuits. This work is focused on a particular AC track circuit, called "Smart Train Detection System" (STDS), designed with both high and low-frequency bands. This approach uses STDS current data applied to an SVM (support vector machine) classifier as a type of failure identifier. The main purpose of this work consists on determine automatically which is the component of the track that is failing to improve the maintenance action. Model was trained to classify 15 different failures that belong to 3 more general categories. The method was tested with field data from 10 different track circuits and validated by the STDS track circuit expert and maintainers. All use cases were correctly classified by the method. 

**Abstract (ZH)**: 基于支持向量机的智能列车检测系统在轨电路中的故障识别研究 

---
# Scalable, Technology-Agnostic Diagnosis and Predictive Maintenance for Point Machine using Deep Learning 

**Title (ZH)**: 基于深度学习的点式设备可扩展且技术无关的诊断与预测性维护 

**Authors**: Eduardo Di Santi, Ruixiang Ci, Clément Lefebvre, Nenad Mijatovic, Michele Pugnaloni, Jonathan Brown, Victor Martín, Kenza Saiah  

**Link**: [PDF](https://arxiv.org/pdf/2508.11692)  

**Abstract**: The Point Machine (PM) is a critical piece of railway equipment that switches train routes by diverting tracks through a switchblade. As with any critical safety equipment, a failure will halt operations leading to service disruptions; therefore, pre-emptive maintenance may avoid unnecessary interruptions by detecting anomalies before they become failures. Previous work relies on several inputs and crafting custom features by segmenting the signal. This not only adds additional requirements for data collection and processing, but it is also specific to the PM technology, the installed locations and operational conditions limiting scalability. Based on the available maintenance records, the main failure causes for PM are obstacles, friction, power source issues and misalignment. Those failures affect the energy consumption pattern of PMs, altering the usual (or healthy) shape of the power signal during the PM movement. In contrast to the current state-of-the-art, our method requires only one input. We apply a deep learning model to the power signal pattern to classify if the PM is nominal or associated with any failure type, achieving >99.99\% precision, <0.01\% false positives and negligible false negatives. Our methodology is generic and technology-agnostic, proven to be scalable on several electromechanical PM types deployed in both real-world and test bench environments. Finally, by using conformal prediction the maintainer gets a clear indication of the certainty of the system outputs, adding a confidence layer to operations and making the method compliant with the ISO-17359 standard. 

**Abstract (ZH)**: 基于电力信号模式的点机故障检测方法 

---
# Towards Generalizable Learning Models for EEG-Based Identification of Pain Perception 

**Title (ZH)**: 基于EEG的疼痛感知识别的可泛化学习模型研究 

**Authors**: Mathis Rezzouk, Fabrice Gagnon, Alyson Champagne, Mathieu Roy, Philippe Albouy, Michel-Pierre Coll, Cem Subakan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11691)  

**Abstract**: EEG-based analysis of pain perception, enhanced by machine learning, reveals how the brain encodes pain by identifying neural patterns evoked by noxious stimulation. However, a major challenge that remains is the generalization of machine learning models across individuals, given the high cross-participant variability inherent to EEG signals and the limited focus on direct pain perception identification in current research. In this study, we systematically evaluate the performance of cross-participant generalization of a wide range of models, including traditional classifiers and deep neural classifiers for identifying the sensory modality of thermal pain and aversive auditory stimulation from EEG recordings. Using a novel dataset of EEG recordings from 108 participants, we benchmark model performance under both within- and cross-participant evaluation settings. Our findings show that traditional models suffered the largest drop from within- to cross-participant performance, while deep learning models proved more resilient, underscoring their potential for subject-invariant EEG decoding. Even though performance variability remained high, the strong results of the graph-based model highlight its potential to capture subject-invariant structure in EEG signals. On the other hand, we also share the preprocessed dataset used in this study, providing a standardized benchmark for evaluating future algorithms under the same generalization constraints. 

**Abstract (ZH)**: 基于EEG的疼痛感知分析，通过机器学习增强，揭示了大脑如何编码疼痛，包括识别由有害刺激引起的神经模式。然而，一个主要挑战是机器学习模型在不同个体间的泛化能力，这归因于EEG信号固有的高个体间变异性，以及当前研究中对直接疼痛感知识别的有限关注。在这项研究中，我们系统地评估了多种模型在不同个体间的泛化性能，包括传统分类器和深度神经分类器，用于从EEG记录中识别热痛和令人不悦的听觉刺激的感觉模式。我们使用来自108名参与者的新型EEG记录数据集，在不同个体内的评价和不同个体间的评价设置下对模型性能进行了基准测试。研究结果表明，传统模型从不同个体内的性能下降最大，而深度学习模型表现更为稳定，突显了其在EEG解码中的潜在价值。尽管性能变化仍然很高，基于图的模型的强结果表明了其捕捉EEG信号中不变结构的潜力。此外，我们还分享了在本次研究中使用的预处理数据集，为在相同泛化约束条件下评估未来算法提供了标准化基准。 

---
# Real Time Child Abduction And Detection System 

**Title (ZH)**: 实时儿童拐卖检测系统 

**Authors**: Tadisetty Sai Yashwanth, Yangalasetty Sruthi Royal, Vankayala Rajeshwari Shreya, Mayank Kashyap, Divyaprabha K N  

**Link**: [PDF](https://arxiv.org/pdf/2508.11690)  

**Abstract**: Child safety continues to be a paramount concern worldwide, with child abduction posing significant threats to communities. This paper presents the development of an edge-based child abduction detection and alert system utilizing a multi-agent framework where each agent incorporates Vision-Language Models (VLMs) deployed on a Raspberry Pi. Leveraging the advanced capabilities of VLMs within individual agents of a multi-agent team, our system is trained to accurately detect and interpret complex interactions involving children in various environments in real-time. The multi-agent system is deployed on a Raspberry Pi connected to a webcam, forming an edge device capable of processing video feeds, thereby reducing latency and enhancing privacy. An integrated alert system utilizes the Twilio API to send immediate SMS and WhatsApp notifications, including calls and messages, when a potential child abduction event is detected. Experimental results demonstrate that the system achieves high accuracy in detecting potential abduction scenarios, with near real-time performance suitable for practical deployment. The multi-agent architecture enhances the system's ability to process complex situational data, improving detection capabilities over traditional single-model approaches. The edge deployment ensures scalability and cost-effectiveness, making it accessible for widespread use. The proposed system offers a proactive solution to enhance child safety through continuous monitoring and rapid alerting, contributing a valuable tool in efforts to prevent child abductions. 

**Abstract (ZH)**: 基于边缘计算的多agent儿童绑架检测及警报系统 

---
# Adaptive Spiking with Plasticity for Energy Aware Neuromorphic Systems 

**Title (ZH)**: 适应性脉冲与可塑性机制在节能类脑系统中的应用 

**Authors**: Eduardo Calle-Ortiz, Hui Guan, Deepak Ganesan, Phuc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11689)  

**Abstract**: This paper presents ASPEN, a novel energy-aware technique for neuromorphic systems that could unleash the future of intelligent, always-on, ultra-low-power, and low-burden wearables. Our main research objectives are to explore the feasibility of neuromorphic computing for wearables, identify open research directions, and demonstrate the feasibility of developing an adaptive spiking technique for energy-aware computation, which can be game-changing for resource-constrained devices in always-on applications. As neuromorphic computing systems operate based on spike events, their energy consumption is closely related to spiking activity, i.e., each spike incurs computational and power costs; consequently, minimizing the number of spikes is a critical strategy for operating under constrained energy budgets. To support this goal, ASPEN utilizes stochastic perturbations to the neuronal threshold during training to not only enhance the network's robustness across varying thresholds, which can be controlled at inference time, but also act as a regularizer that improves generalization, reduces spiking activity, and enables energy control without the need for complex retraining or pruning. More specifically, ASPEN adaptively adjusts intrinsic neuronal parameters as a lightweight and scalable technique for dynamic energy control without reconfiguring the entire model. Our evaluation on neuromorphic emulator and hardware shows that ASPEN significantly reduces spike counts and energy consumption while maintaining accuracy comparable to state-of-the-art methods. 

**Abstract (ZH)**: ASPEN：一种面向可穿戴设备的新型能效感知神经形态技术 

---
# Age-Normalized HRV Features for Non-Invasive Glucose Prediction: A Pilot Sleep-Aware Machine Learning Study 

**Title (ZH)**: 基于年龄标准化的心率变异特征的无侵入性血糖预测：一项睡眠感知机器学习 pilot 研究 

**Authors**: Md Basit Azam, Sarangthem Ibotombi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.11682)  

**Abstract**: Non-invasive glucose monitoring remains a critical challenge in the management of diabetes. HRV during sleep shows promise for glucose prediction however, age-related autonomic changes significantly confound traditional HRV analyses. We analyzed 43 subjects with multi-modal data including sleep-stage specific ECG, HRV features, and clinical measurements. A novel age-normalization technique was applied to the HRV features by, dividing the raw values by age-scaled factors. BayesianRidge regression with 5-fold cross-validation was employed for log-glucose prediction. Age-normalized HRV features achieved R2 = 0.161 (MAE = 0.182) for log-glucose prediction, representing a 25.6% improvement over non-normalized features (R2 = 0.132). The top predictive features were hrv rem mean rr age normalized (r = 0.443, p = 0.004), hrv ds mean rr age normalized (r = 0.438, p = 0.005), and diastolic blood pressure (r = 0.437, p = 0.005). Systematic ablation studies confirmed age-normalization as the critical component, with sleep-stage specific features providing additional predictive value. Age-normalized HRV features significantly enhance glucose prediction accuracy compared with traditional approaches. This sleep-aware methodology addresses fundamental limitations in autonomic function assessment and suggests a preliminary feasibility for non-invasive glucose monitoring applications. However, these results require validation in larger cohorts before clinical consideration. 

**Abstract (ZH)**: 非侵入性血糖监测在糖尿病管理中仍是一个关键挑战。睡眠期间的HRV有潜力用于血糖预测，然而随着年龄相关的自主神经系统变化显著干扰了传统的HRV分析。我们在43名具有多模态数据的受试者中进行了分析，这些数据包括睡眠阶段特特-specific ECG、HRV特征和和 及临床测量。 E我们将HRV特征进行了年龄标准化处理，即将原始值除以年龄缩放因素。我们采用了包含5五折交叉验证的贝叶斯回归方法来进行长时时间期血糖预测。年龄标准化的HRV特征 E E= 0.6 E 比非年龄标准化特征提高了长 E期血糖预测性能（R  =  0.833）。预测特征包括REM睡眠期间的年龄标准化HRV (r =  0.441  P  =  5)、深睡眠期间的年龄标准化HRV (r =  438  E  =  4) E 和 和舒张压 E压血压 (r =  433  = 4)。系统系统系统系统的剔除研究确认了睡眠阶段特定特征的重要性，它们提供了额外的预测价值。年龄标准化的HRV特征显著提高了血糖预测精度 E E相较于传统方法。这种睡眠意识的方法为基础 E E E自主功能评估带来了根本局限性 E � 且初步表明非侵入性 E E E E血糖监测具有可行性可行性。然而这些结果需要在更大的人群中进行验证以利于临床应用。 

---
# Future progress in artificial intelligence: A survey of expert opinion 

**Title (ZH)**: 未来人工智能的发展进展：专家意见综述 

**Authors**: Vincent C. Müller, Nick Bostrom  

**Link**: [PDF](https://arxiv.org/pdf/2508.11681)  

**Abstract**: There is, in some quarters, concern about high-level machine intelligence and superintelligent AI coming up in a few decades, bringing with it significant risks for humanity. In other quarters, these issues are ignored or considered science fiction. We wanted to clarify what the distribution of opinions actually is, what probability the best experts currently assign to high-level machine intelligence coming up within a particular time-frame, which risks they see with that development, and how fast they see these developing. We thus designed a brief questionnaire and distributed it to four groups of experts in 2012/2013. The median estimate of respondents was for a one in two chance that high-level machine intelligence will be developed around 2040-2050, rising to a nine in ten chance by 2075. Experts expect that systems will move on to superintelligence in less than 30 years thereafter. They estimate the chance is about one in three that this development turns out to be 'bad' or 'extremely bad' for humanity. 

**Abstract (ZH)**: 关于高级机器智能和超智能AI在未来几十年内出现的风险与专家观点的问卷调查研究 

---
# Comparative Analysis of Time Series Foundation Models for Demographic Forecasting: Enhancing Predictive Accuracy in US Population Dynamics 

**Title (ZH)**: 时间序列基础模型在人口动态预测中的比较分析：增强美国人口动力学预测准确性 

**Authors**: Aditya Akella, Jonathan Farah  

**Link**: [PDF](https://arxiv.org/pdf/2508.11680)  

**Abstract**: Demographic shifts, influenced by globalization, economic conditions, geopolitical events, and environmental factors, pose significant challenges for policymakers and researchers. Accurate demographic forecasting is essential for informed decision-making in areas such as urban planning, healthcare, and economic policy. This study explores the application of time series foundation models to predict demographic changes in the United States using datasets from the U.S. Census Bureau and Federal Reserve Economic Data (FRED). We evaluate the performance of the Time Series Foundation Model (TimesFM) against traditional baselines including Long Short-Term Memory (LSTM) networks, Autoregressive Integrated Moving Average (ARIMA), and Linear Regression. Our experiments across six demographically diverse states demonstrate that TimesFM achieves the lowest Mean Squared Error (MSE) in 86.67% of test cases, with particularly strong performance on minority populations with sparse historical data. These findings highlight the potential of pre-trained foundation models to enhance demographic analysis and inform proactive policy interventions without requiring extensive task-specific fine-tuning. 

**Abstract (ZH)**: 全球化、经济条件、地缘政治事件和环境因素影响下的人口结构变化对政策制定者和研究人员提出了重大挑战。准确的人口预测对于城市规划、卫生保健和经济政策领域的明智决策至关重要。本研究探讨了使用美国人口普查局和联邦储备经济数据（FRED）数据集，将时间序列基础模型应用于预测美国的人口变化。我们将时间序列基础模型（TimesFM）的性能与传统的长短期记忆（LSTM）网络、自回归整合移动平均（ARIMA）和线性回归等基线模型进行评估。我们的实验结果表明，TimesFM在86.67%的测试案例中实现了最低的均方误差（MSE），特别是在历史数据稀疏的少数群体表现尤为突出。这些发现突显了预训练基础模型在增强人口分析和指导前瞻性政策干预方面的潜力，无需进行大量特定任务的微调。 

---
# Lifelong Learner: Discovering Versatile Neural Solvers for Vehicle Routing Problems 

**Title (ZH)**: 终身学习者: 发现适用于车辆路线问题的多功能神经求解器 

**Authors**: Shaodi Feng, Zhuoyi Lin, Jianan Zhou, Cong Zhang, Jingwen Li, Kuan-Wen Chen, Senthilnath Jayavelu, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2508.11679)  

**Abstract**: Deep learning has been extensively explored to solve vehicle routing problems (VRPs), which yields a range of data-driven neural solvers with promising outcomes. However, most neural solvers are trained to tackle VRP instances in a relatively monotonous context, e.g., simplifying VRPs by using Euclidean distance between nodes and adhering to a single problem size, which harms their off-the-shelf application in different scenarios. To enhance their versatility, this paper presents a novel lifelong learning framework that incrementally trains a neural solver to manage VRPs in distinct contexts. Specifically, we propose a lifelong learner (LL), exploiting a Transformer network as the backbone, to solve a series of VRPs. The inter-context self-attention mechanism is proposed within LL to transfer the knowledge obtained from solving preceding VRPs into the succeeding ones. On top of that, we develop a dynamic context scheduler (DCS), employing the cross-context experience replay to further facilitate LL looking back on the attained policies of solving preceding VRPs. Extensive results on synthetic and benchmark instances (problem sizes up to 18k) show that our LL is capable of discovering effective policies for tackling generic VRPs in varying contexts, which outperforms other neural solvers and achieves the best performance for most VRPs. 

**Abstract (ZH)**: 深度学习在解决车辆路由问题中的广泛探索及其在不同情境下的终身学习框架 

---
# Deep Language Geometry: Constructing a Metric Space from LLM Weights 

**Title (ZH)**: 深度语言几何：从大模型权重构建度量空间 

**Authors**: Maksym Shamrai, Vladyslav Hamolia  

**Link**: [PDF](https://arxiv.org/pdf/2508.11676)  

**Abstract**: We introduce a novel framework that utilizes the internal weight activations of modern Large Language Models (LLMs) to construct a metric space of languages. Unlike traditional approaches based on hand-crafted linguistic features, our method automatically derives high-dimensional vector representations by computing weight importance scores via an adapted pruning algorithm. Our approach captures intrinsic language characteristics that reflect linguistic phenomena. We validate our approach across diverse datasets and multilingual LLMs, covering 106 languages. The results align well with established linguistic families while also revealing unexpected inter-language connections that may indicate historical contact or language evolution. The source code, computed language latent vectors, and visualization tool are made publicly available at this https URL. 

**Abstract (ZH)**: 我们介绍了一种新颖的框架，该框架利用现代大型语言模型（LLMs）的内部权重激活来构建语言度量空间。与基于人工构造的语言特征的传统方法不同，我们的方法通过使用调整后的剪枝算法计算权重重要性得分来自动生成高维向量表示。我们的方法捕获了反映语言现象的内在语言特征。我们跨多种数据集和多语言LLMs验证了该方法，覆盖了106种语言。结果与 Established Linguistic Families 一致，同时也揭示了意想不到的语言间联系，这些联系可能表明历史接触或语言演化。源代码、计算的语言隐含向量和可视化工具在此处提供。 

---
# Learning Internal Biological Neuron Parameters and Complexity-Based Encoding for Improved Spiking Neural Networks Performance 

**Title (ZH)**: 基于内部生物神经元参数和 复杂性编码的学习以改善脉冲神经网络性能 yabody 

**Authors**: Zofia Rudnicka, Janusz Szczepanski, Agnieszka Pregowska  

**Link**: [PDF](https://arxiv.org/pdf/2508.11674)  

**Abstract**: This study introduces a novel approach by replacing the traditional perceptron neuron model with a biologically inspired probabilistic meta neuron, where the internal neuron parameters are jointly learned, leading to improved classification accuracy of spiking neural networks (SNNs). To validate this innovation, we implement and compare two SNN architectures: one based on standard leaky integrate-and-fire (LIF) neurons and another utilizing the proposed probabilistic meta neuron model. As a second key contribution, we present a new biologically inspired classification framework that uniquely integrates SNNs with Lempel-Ziv complexity (LZC) a measure closely related to entropy rate. By combining the temporal precision and biological plausibility of SNNs with the capacity of LZC to capture structural regularity, the proposed approach enables efficient and interpretable classification of spatiotemporal neural data, an aspect not addressed in existing works. We consider learning algorithms such as backpropagation, spike-timing-dependent plasticity (STDP), and the Tempotron learning rule. To explore neural dynamics, we use Poisson processes to model neuronal spike trains, a well-established method for simulating the stochastic firing behavior of biological neurons. Our results reveal that depending on the training method, the classifier's efficiency can improve by up to 11.00%, highlighting the advantage of learning additional neuron parameters beyond the traditional focus on weighted inputs alone. 

**Abstract (ZH)**: 本研究介绍了一种新颖的方法，通过用生物启发的概率元神经元替代传统的感知器神经元模型，并共同学习内部神经元参数，提高了神经元脉冲网络（SNN）的分类准确性。为了验证这一创新，我们实现了并比较了两种SNN架构：一种基于标准泄漏积分-放电（LIF）神经元，另一种利用所提出的概率元神经元模型。作为第二个重要贡献，我们提出了一种新的生物启发分类框架，该框架独特地将SNN与Lempel-Ziv复杂度（LZC，与熵率密切相关的度量）相结合。通过将SNN的时间精确性和生物可行性与LZC捕捉结构规律的能力结合起来，所提出的方法能够有效地对时空神经数据进行分类和解释，这是现有工作中未曾涉及的方面。我们考虑了反向传播、突触定时依赖可塑性（STDP）和Tempotron学习规则等学习算法。为了探索神经动力学，我们使用泊松过程来模拟神经元脉冲序列，这是一种广泛用于模拟生物神经元随机放电行为的成熟方法。我们的结果显示，根据训练方法的不同，分类器的效率最多可提高11.00%，突显了学习超过传统加权输入的额外神经元参数的优势。 

---
# Contrastive Regularization over LoRA for Multimodal Biomedical Image Incremental Learning 

**Title (ZH)**: 基于LoRA的对比正则化在多模态生物医学图像增量学习中的应用 

**Authors**: Haojie Zhang, Yixiong Liang, Hulin Kuang, Lihui Cen, Zhe Qu, Yigang Cen, Min Zeng, Shichao Kan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11673)  

**Abstract**: Multimodal Biomedical Image Incremental Learning (MBIIL) is essential for handling diverse tasks and modalities in the biomedical domain, as training separate models for each modality or task significantly increases inference costs. Existing incremental learning methods focus on task expansion within a single modality, whereas MBIIL seeks to train a unified model incrementally across modalities. The MBIIL faces two challenges: I) How to preserve previously learned knowledge during incremental updates? II) How to effectively leverage knowledge acquired from existing modalities to support new modalities? To address these challenges, we propose MSLoRA-CR, a method that fine-tunes Modality-Specific LoRA modules while incorporating Contrastive Regularization to enhance intra-modality knowledge sharing and promote inter-modality knowledge differentiation. Our approach builds upon a large vision-language model (LVLM), keeping the pretrained model frozen while incrementally adapting new LoRA modules for each modality or task. Experiments on the incremental learning of biomedical images demonstrate that MSLoRA-CR outperforms both the state-of-the-art (SOTA) approach of training separate models for each modality and the general incremental learning method (incrementally fine-tuning LoRA). Specifically, MSLoRA-CR achieves a 1.88% improvement in overall performance compared to unconstrained incremental learning methods while maintaining computational efficiency. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 多模态生物医学图像增量学习（MBIIL） 

---
# Revealing Neurocognitive and Behavioral Patterns by Unsupervised Manifold Learning from Dynamic Brain Data 

**Title (ZH)**: 揭示动态脑数据无监督流形学习中的神经认知和行为模式 

**Authors**: Zixia Zhou, Junyan Liu, Wei Emma Wu, Ruogu Fang, Sheng Liu, Qingyue Wei, Rui Yan, Yi Guo, Qian Tao, Yuanyuan Wang, Md Tauhidul Islam, Lei Xing  

**Link**: [PDF](https://arxiv.org/pdf/2508.11672)  

**Abstract**: Dynamic brain data, teeming with biological and functional insights, are becoming increasingly accessible through advanced measurements, providing a gateway to understanding the inner workings of the brain in living subjects. However, the vast size and intricate complexity of the data also pose a daunting challenge in reliably extracting meaningful information across various data sources. This paper introduces a generalizable unsupervised deep manifold learning for exploration of neurocognitive and behavioral patterns. Unlike existing methods that extract patterns directly from the input data as in the existing methods, the proposed Brain-dynamic Convolutional-Network-based Embedding (BCNE) seeks to capture the brain-state trajectories by deciphering the temporospatial correlations within the data and subsequently applying manifold learning to this correlative representation. The performance of BCNE is showcased through the analysis of several important dynamic brain datasets. The results, both visual and quantitative, reveal a diverse array of intriguing and interpretable patterns. BCNE effectively delineates scene transitions, underscores the involvement of different brain regions in memory and narrative processing, distinguishes various stages of dynamic learning processes, and identifies differences between active and passive behaviors. BCNE provides an effective tool for exploring general neuroscience inquiries or individual-specific patterns. 

**Abstract (ZH)**: 动态脑数据富含生物和功能性的见解，通过先进的测量手段变得日益可用，为理解活体对象脑部工作机制提供了通道。然而，数据的庞大体量及其复杂的结构也给从各种数据源中可靠地提取有意义的信息带来了巨大挑战。本文介绍了一种可泛化的无监督深度流形学习方法，用于探索神经认知和行为模式。与现有方法直接从输入数据中提取模式不同，所提出的动态卷积网络嵌入（BCNE）旨在通过破解数据中的时空间相关性来捕获脑状态轨迹，并随后应用流形学习于这种相关表示中。通过对几个重要的动态脑数据集进行分析，BCNE展示了其性能。视觉和定量的结果揭示了一系列引人入胜且可解释的模式。BCNE有效地界定了场景转换，突出了不同脑区在记忆和叙事处理中的参与，区分了动态学习过程的不同阶段，并识别了主动行为与被动行为之间的差异。BCNE提供了一个有效工具，用于探索一般神经科学问题或个体特异性模式。 

---
# LLM-Based Intelligent Agents for Music Recommendation: A Comparison with Classical Content-Based Filtering 

**Title (ZH)**: 基于LLM的智能代理在音乐推荐中的应用：与经典内容基过滤的对比 

**Authors**: Ronald Carvalho Boadana, Ademir Guimarães da Costa Junior, Ricardo Rios, Fábio Santos da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2508.11671)  

**Abstract**: The growing availability of music on streaming platforms has led to information overload for users. To address this issue and enhance the user experience, increasingly sophisticated recommendation systems have been proposed. This work investigates the use of Large Language Models (LLMs) from the Gemini and LLaMA families, combined with intelligent agents, in a multi-agent personalized music recommendation system. The results are compared with a traditional content-based recommendation model, considering user satisfaction, novelty, and computational efficiency. LLMs achieved satisfaction rates of up to \textit{89{,}32\%}, indicating their promising potential in music recommendation systems. 

**Abstract (ZH)**: 流媒体平台上的音乐日益丰富导致了信息过载问题。为应对这一问题并提升用户体验，提出了更为复杂的推荐系统。本文探讨了将Gemini和LLaMA家族的大语言模型与智能代理结合，在多智能体个性化音乐推荐系统中的应用。结果与传统的基于内容的推荐模型进行比较，考虑了用户满意度、新颖性和计算效率。大语言模型达到了高达89.32%的满意度，表明其在音乐推荐系统中的潜在应用前景。 

---
# RRRA: Resampling and Reranking through a Retriever Adapter 

**Title (ZH)**: RRRA：通过检索适配器进行重采样和重排序 

**Authors**: Bongsu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.11670)  

**Abstract**: In dense retrieval, effective training hinges on selecting high quality hard negatives while avoiding false negatives. Recent methods apply heuristics based on positive document scores to identify hard negatives, improving both performance and interpretability. However, these global, example agnostic strategies often miss instance specific false negatives. To address this, we propose a learnable adapter module that monitors Bi-Encoder representations to estimate the likelihood that a hard negative is actually a false negative. This probability is modeled dynamically and contextually, enabling fine-grained, query specific judgments. The predicted scores are used in two downstream components: (1) resampling, where negatives are reweighted during training, and (2) reranking, where top-k retrieved documents are reordered at inference. Empirical results on standard benchmarks show that our adapter-enhanced framework consistently outperforms strong Bi-Encoder baselines, underscoring the benefit of explicit false negative modeling in dense retrieval. 

**Abstract (ZH)**: 在密集检索中，有效的训练依赖于选择高质量的负样本同时避免错误的负样本。近期的方法通过正文档分数的启发式方法来识别负样本，从而提高性能和可解释性。然而，这些全局且独立于样本的方法往往忽略了实例特定的错误负样本。为解决这一问题，我们提出一个可学习的适配器模块，该模块监控双编码器表示以估计一个硬负样本实际上是错误负样本的可能性。这种概率被动态和上下文地建模，从而能够进行细粒度的、查询特定的判断。预测得分被用于两个下游组件：(1) 重采样，在训练过程中重新加权负样本，(2) 重新排序，在推理时重新排列检索到的前k个文档。标准基准上的实验证明，我们的适配器增强框架一致地优于强大的双编码器基线，强调了在密集检索中明确建模错误负样本的益处。 

---
# Collaborative Learning-Enhanced Lightweight Models for Predicting Arterial Blood Pressure Waveform in a Large-scale Perioperative Dataset 

**Title (ZH)**: 协作学习增强的轻量级模型在大规模围手术期数据集中预测动脉血流动力学波形 

**Authors**: Wentao Li, Yonghu He, Kun Gao, Qing Liu, Yali Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.11669)  

**Abstract**: Noninvasive arterial blood pressure (ABP) monitoring is essential for patient management in critical care and perioperative settings, providing continuous assessment of cardiovascular hemodynamics with minimal risks. Numerous deep learning models have developed to reconstruct ABP waveform from noninvasively acquired physiological signals such as electrocardiogram and photoplethysmogram. However, limited research has addressed the issue of model performance and computational load for deployment on embedded systems. The study introduces a lightweight sInvResUNet, along with a collaborative learning scheme named KDCL_sInvResUNet. With only 0.89 million parameters and a computational load of 0.02 GFLOPS, real-time ABP estimation was successfully achieved on embedded devices with an inference time of just 8.49 milliseconds for a 10-second output. We performed subject-independent validation in a large-scale and heterogeneous perioperative dataset containing 1,257,141 data segments from 2,154 patients, with a wide BP range (41-257 mmHg for SBP, and 31-234 mmHg for DBP). The proposed KDCL_sInvResUNet achieved lightly better performance compared to large models, with a mean absolute error of 10.06 mmHg and mean Pearson correlation of 0.88 in tracking ABP changes. Despite these promising results, all deep learning models showed significant performance variations across different demographic and cardiovascular conditions, highlighting their limited ability to generalize across such a broad and diverse population. This study lays a foundation work for real-time, unobtrusive ABP monitoring in real-world perioperative settings, providing baseline for future advancements in this area. 

**Abstract (ZH)**: 非侵入性动脉血压（ABP）监测在重症监护和围手术期环境中至关重要，能够提供心血管 hemodynamics 的连续评估，并具有最小的风险。已开发了多种深度学习模型从心电图和光电脉搏图等非侵入性获取的生理信号中重构 ABP 波形。然而，有限的研究关注了在嵌入式系统上部署时模型性能和计算负载的问题。本研究介绍了轻量级 sInvResUNet，并提出了一种协作学习方案 KDCL_sInvResUNet。仅含 0.89 百万参数和计算负载为 0.02 GFLOPS 的情况下，成功在嵌入式设备上实现了实时 ABP 估计，推理时间为 8.49 毫秒（用于 10 秒输出）。我们在包含来自 2154 名患者、共 1,257,141 个数据段的大规模和异质围手术期数据集上进行了跨受试者验证，收缩压（SBP）和舒张压（DBP）范围广泛（41-257 mmHg 和 31-234 mmHg）。所提出的 KDCL_sInvResUNet 在跟踪 ABP 变化方面的表现略优于大型模型，平均绝对误差为 10.06 mmHg，平均皮尔逊相关系数为 0.88。尽管取得了这些有希望的结果，但所有深度学习模型在不同的人口统计学和心血管条件下均显示出显著的性能差异，突显了其在如此广泛和多样的人群中泛化的局限性。本研究为实际围手术期环境中实现实时、不干扰的 ABP 监测奠定了基础，并提供了该领域未来发展的基准。 

---
# Assessing Representation Stability for Transformer Models 

**Title (ZH)**: 评估Transformer模型的表示稳定性 

**Authors**: Bryan E. Tuck, Rakesh M. Verma  

**Link**: [PDF](https://arxiv.org/pdf/2508.11667)  

**Abstract**: Adversarial text attacks remain a persistent threat to transformer models, yet existing defenses are typically attack-specific or require costly model retraining. We introduce Representation Stability (RS), a model-agnostic detection framework that identifies adversarial examples by measuring how embedding representations change when important words are masked. RS first ranks words using importance heuristics, then measures embedding sensitivity to masking top-k critical words, and processes the resulting patterns with a BiLSTM detector. Experiments show that adversarially perturbed words exhibit disproportionately high masking sensitivity compared to naturally important words. Across three datasets, three attack types, and two victim models, RS achieves over 88% detection accuracy and demonstrates competitive performance compared to existing state-of-the-art methods, often at lower computational cost. Using Normalized Discounted Cumulative Gain (NDCG) to measure perturbation identification quality, we reveal that gradient-based ranking outperforms attention and random selection approaches, with identification quality correlating with detection performance for word-level attacks. RS also generalizes well to unseen datasets, attacks, and models without retraining, providing a practical solution for adversarial text detection. 

**Abstract (ZH)**: Adversarial 文本攻击依然对变换器模型构成持续威胁，现有的防御方法通常是针对特定攻击或需要昂贵的模型重新训练。我们引入了一种模型无关的检测框架——表示稳定（Representation Stability，RS），通过测量重要词语被遮掩时嵌入表示的变化来识别对抗样本。RS 首先使用重要性启发式对词语进行排序，然后测量遮掩前 k 个关键词语时嵌入的敏感度，并使用双向 LSTM 检测器处理生成的模式。实验表明，对抗性扰动的词语相比于自然重要的词语，表现出不成比例的高遮掩敏感度。在三个数据集、三种攻击类型和两种受害者模型上，RS 的检测准确率超过 88%，并在计算成本较低的情况下展现出与现有最先进方法相当的性能。使用归一化折现累积增益（NDCG）衡量扰动识别质量，我们发现梯度基排名方法优于注意力和随机选择方法，识别质量与检测性能呈正相关，尤其是在单词级攻击中。此外，RS 能够对未见过的数据集、攻击和模型进行良好的泛化，无需重新训练，提供了一种实用的对抗文本检测解决方案。 

---
# Generative AI in Training and Coaching: Redefining the Design Process of Learning Materials 

**Title (ZH)**: 生成式人工智能在训练与 coaching 中的应用：重塑学习材料设计过程 

**Authors**: Alexander Komar, Marc-André Heidelmann, Kristina Schaaff  

**Link**: [PDF](https://arxiv.org/pdf/2508.11662)  

**Abstract**: Generative artificial intelligence (GenAI) is transforming education, redefining the role of trainers and coaches in learning environments. In our study, we explore how AI integrates into the design process of learning materials, assessing its impact on efficiency, pedagogical quality, and the evolving role of human trainers and coaches. Through qualitative interviews with professionals in education and corporate training, we identify the following key topics: trainers and coaches increasingly act as facilitators and content moderators rather than primary creators, efficiency gains allow for a stronger strategic focus but at the same time the new tools require new skills. Additionally, we analyze how the anthropomorphism of AI shapes user trust and expectations. From these insights, we derive how tools based on GenAI can successfully be implemented for trainers and coaches on an individual, organizational, systemic, and strategic level. 

**Abstract (ZH)**: 生成人工智能（GenAI）正在变革教育，重新定义培训师和教练在学习环境中的角色。在我们的研究中，我们探讨AI如何融入学习材料的设计过程，评估其对效率、教学质量以及人类培训师和教练角色演变的影响。通过与教育和企业培训专业人士的定性访谈，我们确定了以下关键主题：培训师和教练越来越多地充当协调员和内容审查员，而不是主要内容创作者；效率提升使得战略重点更清晰，但同时也需要新的技能。此外，我们分析了人工智能拟人化如何影响用户信任和期望。从这些洞察中，我们推导出如何在个体、组织、系统和战略层面成功实施基于GenAI的工具。 

---
# Toward Practical Equilibrium Propagation: Brain-inspired Recurrent Neural Network with Feedback Regulation and Residual Connections 

**Title (ZH)**: 面向实用的均衡传播：具有反馈调节和残差连接的脑启发递归神经网络 

**Authors**: Zhuo Liu, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11659)  

**Abstract**: Brain-like intelligent systems need brain-like learning methods. Equilibrium Propagation (EP) is a biologically plausible learning framework with strong potential for brain-inspired computing hardware. However, existing im-plementations of EP suffer from instability and prohibi-tively high computational costs. Inspired by the structure and dynamics of the brain, we propose a biologically plau-sible Feedback-regulated REsidual recurrent neural network (FRE-RNN) and study its learning performance in EP framework. Feedback regulation enables rapid convergence by reducing the spectral radius. The improvement in con-vergence property reduces the computational cost and train-ing time of EP by orders of magnitude, delivering perfor-mance on par with backpropagation (BP) in benchmark tasks. Meanwhile, residual connections with brain-inspired topologies help alleviate the vanishing gradient problem that arises when feedback pathways are weak in deep RNNs. Our approach substantially enhances the applicabil-ity and practicality of EP in large-scale networks that un-derpin artificial intelligence. The techniques developed here also offer guidance to implementing in-situ learning in physical neural networks. 

**Abstract (ZH)**: 脑似的智能系统需要脑似的学习方法。Equilibrium Propagation (EP)是一种具有强大脑启发式计算硬件潜力的生物可实现学习框架。然而，现有的EP实现面临不稳定性问题和高昂的计算成本。受脑结构和动力学的启发，我们提出了一种生物可实现的反馈调节残差递归神经网络（FRE-RNN），并在EP框架下研究其学习性能。反馈调节通过减小特征值半径实现了快速收敛，从而显著提高了EP在收敛性方面的性能，降低了计算成本和培训时间，在基准任务中性能与反向传播（BP）相当。同时，具有脑启发式拓扑结构的残差连接有助于缓解深层RNN中反馈路径较弱时出现的梯度消失问题。我们的方法显著增强了EP在支撑人工智能的大规模网络中的应用性和实用性。所开发的技术也为在物理神经网络中实现就地学习提供了指导。 

---
# Categorical Construction of Logically Verifiable Neural Architectures 

**Title (ZH)**: 逻辑可验证神经架构的类别构建 

**Authors**: Logan Nye  

**Link**: [PDF](https://arxiv.org/pdf/2508.11647)  

**Abstract**: Neural networks excel at pattern recognition but struggle with reliable logical reasoning, often violating basic logical principles during inference. We address this limitation by developing a categorical framework that systematically constructs neural architectures with provable logical guarantees. Our approach treats logical theories as algebraic structures called Lawvere theories, which we transform into neural networks using categorical algebra in the 2-category of parametric maps. Unlike existing methods that impose logical constraints during training, our categorical construction embeds logical principles directly into the network's architectural structure, making logical violations mathematically impossible. We demonstrate this framework by constructing differentiable neural architectures for propositional logic that preserve boolean reasoning while remaining trainable via gradient descent. Our main theoretical result establishes a bijective correspondence between finitary logical theories and neural architectures, proving that every logically constrained network arises uniquely from our construction. This extends Categorical Deep Learning beyond geometric symmetries to semantic constraints, enabling automatic derivation of verified architectures from logical specifications. The framework provides mathematical foundations for trustworthy AI systems, with applications to theorem proving, formal verification, and safety-critical reasoning tasks requiring verifiable logical behavior. 

**Abstract (ZH)**: 神经网络在模式识别方面表现出色，但在可靠的逻辑推理方面存在局限性，往往在推理过程中违反基本的逻辑原则。我们通过开发一种类别框架来解决这一限制，该框架系统地构建具有可证明逻辑保证的神经网络架构。我们的方法将逻辑理论视为称为Lawvere理论的代数结构，并使用范畴代数在参数映射的2-范畴中将其转换为神经网络。与现有方法在训练过程中施加逻辑约束不同，我们的范畴构造直接将逻辑原则嵌入网络的架构结构中，从而使逻辑违反在数学上成为不可能。我们通过构建保持布尔推理且仍可通过梯度下降进行训练的命题逻辑可微神经架构来演示这一框架。我们的主要理论结果建立了有限逻辑理论与神经架构之间的双射对应关系，证明了每一逻辑约束网络都唯一地源自我们的构造。这将范畴深度学习扩展到语义约束，使从逻辑规范自动推导出可验证架构成为可能。该框架为可信赖的人工智能系统提供了数学基础，并应用于定理证明、形式验证以及需要可验证逻辑行为的安全关键推理任务。 

---
# Vibe2Spike: Batteryless Wireless Tags for Vibration Sensing with Event Cameras and Spiking Networks 

**Title (ZH)**: Vibe2Spike：基于事件摄像头和突触网络的无电池振动传感标签 

**Authors**: Danny Scott, William LaForest, Hritom Das, Ioannis Polykretis, Catherine D. Schuman, Charles Rizzo, James Plank, Sai Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11640)  

**Abstract**: The deployment of dense, low-cost sensors is critical for realizing ubiquitous smart environments. However, existing sensing solutions struggle with the energy, scalability, and reliability trade-offs imposed by battery maintenance, wireless transmission overhead, and data processing complexity. In this work, we present Vibe2Spike, a novel battery-free, wireless sensing framework that enables vibration-based activity recognition using visible light communication (VLC) and spiking neural networks (SNNs). Our system uses ultra-low-cost tags composed only of a piezoelectric disc, a Zener diode, and an LED, which harvest vibration energy and emit sparse visible light spikes without requiring batteries or RF radios. These optical spikes are captured by event cameras and classified using optimized SNN models evolved via the EONS framework. We evaluate Vibe2Spike across five device classes, achieving 94.9\% average classification fitness while analyzing the latency-accuracy trade-offs of different temporal binning strategies. Vibe2Spike demonstrates a scalable, and energy-efficient approach for enabling intelligent environments in a batteryless manner. 

**Abstract (ZH)**: 无需能源的振动感知框架Vibe2Spike：基于可见光通信和突触神经网络的活动识别 

---
