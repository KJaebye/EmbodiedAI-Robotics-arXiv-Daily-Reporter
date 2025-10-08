# Moloch's Bargain: Emergent Misalignment When LLMs Compete for Audiences 

**Title (ZH)**: 摩洛赫的交易：当大语言模型为争夺受众而竞争时出现的 emergent 脱轨 

**Authors**: Batu El, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.06105)  

**Abstract**: Large language models (LLMs) are increasingly shaping how information is created and disseminated, from companies using them to craft persuasive advertisements, to election campaigns optimizing messaging to gain votes, to social media influencers boosting engagement. These settings are inherently competitive, with sellers, candidates, and influencers vying for audience approval, yet it remains poorly understood how competitive feedback loops influence LLM behavior. We show that optimizing LLMs for competitive success can inadvertently drive misalignment. Using simulated environments across these scenarios, we find that, 6.3% increase in sales is accompanied by a 14.0% rise in deceptive marketing; in elections, a 4.9% gain in vote share coincides with 22.3% more disinformation and 12.5% more populist rhetoric; and on social media, a 7.5% engagement boost comes with 188.6% more disinformation and a 16.3% increase in promotion of harmful behaviors. We call this phenomenon Moloch's Bargain for AI--competitive success achieved at the cost of alignment. These misaligned behaviors emerge even when models are explicitly instructed to remain truthful and grounded, revealing the fragility of current alignment safeguards. Our findings highlight how market-driven optimization pressures can systematically erode alignment, creating a race to the bottom, and suggest that safe deployment of AI systems will require stronger governance and carefully designed incentives to prevent competitive dynamics from undermining societal trust. 

**Abstract (ZH)**: AI中的莫洛赫交易：竞争力成功以对齐为代价 

---
# Classical AI vs. LLMs for Decision-Maker Alignment in Health Insurance Choices 

**Title (ZH)**: 经典AI与大规模语言模型在健康保险选择中的决策者对齐比较 

**Authors**: Mallika Mainali, Harsha Sureshbabu, Anik Sen, Christopher B. Rauch, Noah D. Reifsnyder, John Meyer, J. T. Turner, Michael W. Floyd, Matthew Molineaux, Rosina O. Weber  

**Link**: [PDF](https://arxiv.org/pdf/2510.06093)  

**Abstract**: As algorithmic decision-makers are increasingly applied to high-stakes domains, AI alignment research has evolved from a focus on universal value alignment to context-specific approaches that account for decision-maker attributes. Prior work on Decision-Maker Alignment (DMA) has explored two primary strategies: (1) classical AI methods integrating case-based reasoning, Bayesian reasoning, and naturalistic decision-making, and (2) large language model (LLM)-based methods leveraging prompt engineering. While both approaches have shown promise in limited domains such as medical triage, their generalizability to novel contexts remains underexplored. In this work, we implement a prior classical AI model and develop an LLM-based algorithmic decision-maker evaluated using a large reasoning model (GPT-5) and a non-reasoning model (GPT-4) with weighted self-consistency under a zero-shot prompting framework, as proposed in recent literature. We evaluate both approaches on a health insurance decision-making dataset annotated for three target decision-makers with varying levels of risk tolerance (0.0, 0.5, 1.0). In the experiments reported herein, classical AI and LLM-based models achieved comparable alignment with attribute-based targets, with classical AI exhibiting slightly better alignment for a moderate risk profile. The dataset and open-source implementation are publicly available at: this https URL and this https URL. 

**Abstract (ZH)**: 随着算法决策者在高风险领域中的应用日益增多，AI对齐研究已从普遍价值观对齐转向考虑决策者特性的上下文特定方法。先前的决策者对齐（DMA）研究探讨了两种主要策略：（1）结合案例推理、贝叶斯推理和自然决策制定的经典AI方法，以及（2）利用提示工程的大语言模型（LLM）方法。尽管这两种方法在医疗分诊等有限领域显示出了潜力，但它们在新型上下文中的可泛化性仍需进一步探索。在本研究中，我们实现了一个先前的经典AI模型，并开发了一个基于LLM的算法决策者，该模型使用GPT-5进行推理评估，并使用GPT-4进行非推理评估，两者均采用加权自我一致性方法，在零样本提示框架下进行评估，该框架近期在文献中有所提及。我们在一个包含三种不同风险容忍度目标决策者（0.0，0.5，1.0）标记的健康保险决策数据集上评估了这两种方法。本研究中报告的实验结果显示，经典AI和基于LLM的模型在基于属性的目标上达到了可比较的对齐程度，经典AI在中等风险配置下稍微表现出更好的对齐程度。数据集及其开源实现已公开发布：this https URL 和 this https URL。 

---
# Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents 

**Title (ZH)**: 基于层次LLM代理的约束感知自然语言路由推荐 

**Authors**: Tao Zhe, Rui Liu, Fateme Memar, Xiao Luo, Wei Fan, Xinyue Ye, Zhongren Peng, Dongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06078)  

**Abstract**: Route recommendation aims to provide users with optimal travel plans that satisfy diverse and complex requirements. Classical routing algorithms (e.g., shortest-path and constraint-aware search) are efficient but assume structured inputs and fixed objectives, limiting adaptability to natural-language queries. Recent LLM-based approaches enhance flexibility but struggle with spatial reasoning and the joint modeling of route-level and POI-level preferences. To address these limitations, we propose RouteLLM, a hierarchical multi-agent framework that grounds natural-language intents into constraint-aware routes. It first parses user queries into structured intents including POIs, paths, and constraints. A manager agent then coordinates specialized sub-agents: a constraint agent that resolves and formally check constraints, a POI agent that retrieves and ranks candidate POIs, and a path refinement agent that refines routes via a routing engine with preference-conditioned costs. A final verifier agent ensures constraint satisfaction and produces the final route with an interpretable rationale. This design bridges linguistic flexibility and spatial structure, enabling reasoning over route feasibility and user preferences. Experiments show that our method reliably grounds textual preferences into constraint-aware routes, improving route quality and preference satisfaction over classical methods. 

**Abstract (ZH)**: 路由推荐旨在为用户提供满足多样化和复杂需求的最优旅行计划。经典路由算法（如最短路径和约束感知搜索）高效但假设结构化输入和固定目标，限制了对自然语言查询的适应性。基于最新LLM的方法提高了灵活性，但在空间推理和路径级偏好与POI级偏好的联合建模方面存在挑战。为应对这些限制，我们提出了RouteLLM，这是一种分层多智能体框架，将自然语言意图转化为约束感知的路由。它首先将用户查询解析为结构化意图，包括POIs、路径和约束。然后，管理智能体协调专门的子智能体：约束智能体负责解决和形式验证约束，POI智能体负责检索和排名候选POIs，路径细化智能体通过带有偏好条件成本的路由引擎优化路径。最后，验证智能体确保约束满足，并生成具有可解释理由的最终路由。该设计将语言灵活性与空间结构相结合，实现了对路线可行性和用户偏好的推理。实验结果显示，我们的方法能够可靠地将文本偏好转化为约束感知的路由，比经典方法提高了路由质量和偏好满意度。 

---
# Refusal Falls off a Cliff: How Safety Alignment Fails in Reasoning? 

**Title (ZH)**: 拒绝率急剧下降：安全对齐在推理中为何失效？ 

**Authors**: Qingyu Yin, Chak Tou Leong, Linyi Yang, Wenxuan Huang, Wenjie Li, Xiting Wang, Jaehong Yoon, YunXing, XingYu, Jinjin Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06036)  

**Abstract**: Large reasoning models (LRMs) with multi-step reasoning capabilities have shown remarkable problem-solving abilities, yet they exhibit concerning safety vulnerabilities that remain poorly understood. In this work, we investigate why safety alignment fails in reasoning models through a mechanistic interpretability lens. Using a linear probing approach to trace refusal intentions across token positions, we discover a striking phenomenon termed as \textbf{refusal cliff}: many poorly-aligned reasoning models correctly identify harmful prompts and maintain strong refusal intentions during their thinking process, but experience a sharp drop in refusal scores at the final tokens before output generation. This suggests that these models are not inherently unsafe; rather, their refusal intentions are systematically suppressed. Through causal intervention analysis, we identify a sparse set of attention heads that negatively contribute to refusal behavior. Ablating just 3\% of these heads can reduce attack success rates below 10\%. Building on these mechanistic insights, we propose \textbf{Cliff-as-a-Judge}, a novel data selection method that identifies training examples exhibiting the largest refusal cliff to efficiently repair reasoning models' safety alignment. This approach achieves comparable safety improvements using only 1.7\% of the vanilla safety training data, demonstrating a less-is-more effect in safety alignment. 

**Abstract (ZH)**: 具有多步推理能力的大型推理模型（LRMs）展示了显著的问题解决能力，但它们表现出的安全性漏洞仍然令人担忧且不甚理解。在这项工作中，我们通过机械可解释性视角探究为什么推理模型中的安全对齐会失败。使用线性探测方法追踪 token 位置上的拒绝意图，我们发现了一种引人注目的现象，称为“拒绝悬崖”：许多对齐不佳的推理模型能够正确识别有害提示并在推理过程中保持强烈的拒绝意图，但在生成输出的最后一部分 token 前，拒绝得分会出现急剧下降。这表明这些模型不是从根本上不安全的；相反，它们的拒绝意图被系统性地抑制了。通过因果干预分析，我们识别出一组稀疏的注意力头，它们对拒绝行为有负向贡献。仅消除这些头的 3% 就可以使攻击成功率达到低于 10%。基于这些机械洞察，我们提出了一种新颖的数据选择方法——“拒绝悬崖即法官（Cliff-as-a-Judge）”，该方法能够有效识别表现出最大拒绝悬崖的训练示例，以高效地修复推理模型的安全对齐。此方法仅使用 1.7% 的原始安全训练数据即可实现相似的安全改进，显示出在安全对齐中“少即是多”的效果。 

---
# Training-Free Time Series Classification via In-Context Reasoning with LLM Agents 

**Title (ZH)**: 无需训练的时间序列分类通过LLM代理的上下文推理 

**Authors**: Songyuan Sui, Zihang Xu, Yu-Neng Chuang, Kwei-Herng Lai, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05950)  

**Abstract**: Time series classification (TSC) spans diverse application scenarios, yet labeled data are often scarce, making task-specific training costly and inflexible. Recent reasoning-oriented large language models (LLMs) show promise in understanding temporal patterns, but purely zero-shot usage remains suboptimal. We propose FETA, a multi-agent framework for training-free TSC via exemplar-based in-context reasoning. FETA decomposes a multivariate series into channel-wise subproblems, retrieves a few structurally similar labeled examples for each channel, and leverages a reasoning LLM to compare the query against these exemplars, producing channel-level labels with self-assessed confidences; a confidence-weighted aggregator then fuses all channel decisions. This design eliminates the need for pretraining or fine-tuning, improves efficiency by pruning irrelevant channels and controlling input length, and enhances interpretability through exemplar grounding and confidence estimation. On nine challenging UEA datasets, FETA achieves strong accuracy under a fully training-free setting, surpassing multiple trained baselines. These results demonstrate that a multi-agent in-context reasoning framework can transform LLMs into competitive, plug-and-play TSC solvers without any parameter training. The code is available at this https URL. 

**Abstract (ZH)**: 基于示例驱动的上下文推理多代理框架FETA实现无监督时间序列分类 

---
# Optimizing for Persuasion Improves LLM Generalization: Evidence from Quality-Diversity Evolution of Debate Strategies 

**Title (ZH)**: 优化说服策略提高大语言模型泛化能力：来自辩论策略质量-多样性进化的证据 

**Authors**: Aksel Joonas Reedi, Corentin Léger, Julien Pourcel, Loris Gaven, Perrine Charriau, Guillaume Pourcel  

**Link**: [PDF](https://arxiv.org/pdf/2510.05909)  

**Abstract**: Large Language Models (LLMs) optimized to output truthful answers often overfit, producing brittle reasoning that fails to generalize. While persuasion-based optimization has shown promise in debate settings, it has not been systematically compared against mainstream truth-based approaches. We introduce DebateQD, a minimal Quality-Diversity (QD) evolutionary algorithm that evolves diverse debate strategies across different categories (rationality, authority, emotional appeal, etc.) through tournament-style competitions where two LLMs debate while a third judges. Unlike previously proposed methods that require a population of LLMs, our approach maintains diversity of opponents through prompt-based strategies within a single LLM architecture, making it more accessible for experiments while preserving the key benefits of population-based optimization. In contrast to prior work, we explicitly isolate the role of the optimization objective by fixing the debate protocol and swapping only the fitness function: persuasion rewards strategies that convince the judge irrespective of truth, whereas truth rewards collaborative correctness. Across three model scales (7B, 32B, 72B parameters) and multiple dataset sizes from the QuALITY benchmark, persuasion-optimized strategies achieve up to 13.94% smaller train-test generalization gaps, while matching or exceeding truth optimization's test performance. These results provide the first controlled evidence that competitive pressure to persuade, rather than seek the truth collaboratively, fosters more transferable reasoning skills, offering a promising path for improving LLM generalization. 

**Abstract (ZH)**: 大型语言模型（LLMs）经过优化以输出真实答案往往会过拟合，产生脆弱的推理，难以泛化。尽管基于说服的优化在辩论环境下显示出潜力，但尚未系统性地与主流的真实导向方法进行比较。我们引入了DebateQD，这是一种最小化的质量-多样性（QD）进化算法，通过锦标赛式竞争，在不同的类别（理性、权威、情感诉求等）中进化不同的辩论策略。与之前需要一批LLM的方法不同，我们的方法通过单一LLM架构中的提示策略来维持对手的多样性，使其更易于实验同时保留基于群体优化的 key 优点。与先前的工作相比，我们明确隔离了优化目标的作用：通过固定辩论协议并仅交换适应度函数，说服奖励能够说服裁判的策略，而不论其是否符合事实；真实则奖励合作的正确性。在三个模型规模（7B、32B、72B参数）和_QUALITY_基准的多种数据集大小上，说服优化的策略在训练-测试泛化差距上最多可缩小13.94%，同时匹配或超越真实优化的测试性能。这些结果提供了首次受控证据，表明为了说服而不是合作寻求事实的竞争压力，促进了更为可迁移的推理技能的发展，为提高LLM泛化能力提供了有希望的道路。 

---
# ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming 

**Title (ZH)**: ConstraintLLM：一种工业级约束编程的神经符号框架 

**Authors**: Weichun Shi, Minghao Liu, Wanting Zhang, Langchen Shi, Fuqi Jia, Feifei Ma, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05774)  

**Abstract**: Constraint programming (CP) is a crucial technology for solving real-world constraint optimization problems (COPs), with the advantages of rich modeling semantics and high solving efficiency. Using large language models (LLMs) to generate formal modeling automatically for COPs is becoming a promising approach, which aims to build trustworthy neuro-symbolic AI with the help of symbolic solvers. However, CP has received less attention compared to works based on operations research (OR) models. We introduce ConstraintLLM, the first LLM specifically designed for CP modeling, which is trained on an open-source LLM with multi-instruction supervised fine-tuning. We propose the Constraint-Aware Retrieval Module (CARM) to increase the in-context learning capabilities, which is integrated in a Tree-of-Thoughts (ToT) framework with guided self-correction mechanism. Moreover, we construct and release IndusCP, the first industrial-level benchmark for CP modeling, which contains 140 challenging tasks from various domains. Our experiments demonstrate that ConstraintLLM achieves state-of-the-art solving accuracy across multiple benchmarks and outperforms the baselines by 2x on the new IndusCP benchmark. Code and data are available at: this https URL. 

**Abstract (ZH)**: 基于约束编程的大型语言模型ConstraintLLM：面向实际约束优化问题的自动建模 

---
# ARM: Discovering Agentic Reasoning Modules for Generalizable Multi-Agent Systems 

**Title (ZH)**: ARM：发现可迁移的多代理系统自主 reasoning 模块 

**Authors**: Bohan Yao, Shiva Krishna Reddy Malay, Vikas Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2510.05746)  

**Abstract**: Large Language Model (LLM)-powered Multi-agent systems (MAS) have achieved state-of-the-art results on various complex reasoning tasks. Recent works have proposed techniques to automate the design of MASes, eliminating the need for manual engineering. However, these techniques perform poorly, often achieving similar or inferior performance to simple baselines. Furthermore, they require computationally expensive re-discovery of architectures for each new task domain and expensive data annotation on domains without existing labeled validation sets. A critical insight is that simple Chain of Thought (CoT) reasoning often performs competitively with these complex systems, suggesting that the fundamental reasoning unit of MASes, CoT, warrants further investigation. To this end, we present a new paradigm for automatic MAS design that pivots the focus to optimizing CoT reasoning. We introduce the Agentic Reasoning Module (ARM), an agentic generalization of CoT where each granular reasoning step is executed by a specialized reasoning module. This module is discovered through a tree search over the code space, starting from a simple CoT module and evolved using mutations informed by reflection on execution traces. The resulting ARM acts as a versatile reasoning building block which can be utilized as a direct recursive loop or as a subroutine in a learned meta-orchestrator. Our approach significantly outperforms both manually designed MASes and state-of-the-art automatic MAS design methods. Crucially, MASes built with ARM exhibit superb generalization, maintaining high performance across different foundation models and task domains without further optimization. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的多智能体系统（MAS）在各种复杂推理任务上实现了最先进的成果。近期工作提出了一种自动设计MAS的技术，消除了手动工程的需求。然而，这些技术表现不佳，经常达到或低于简单基线的性能。此外，它们需要为每个新的任务领域重新发现架构，并对没有现有标注验证集的领域进行昂贵的数据标注。一个关键的洞察是，简单的一步步推理（CoT）通常能与这些复杂系统竞争，这表明MAS的核心推理单元CoT值得进一步研究。为此，我们提出了一种自动设计MAS的新范式，将焦点转向优化CoT推理。我们介绍了智能推理模块（ARM），这是一种针对CoT的智能泛化，其中每个精细的推理步骤由专门的推理模块执行。该模块通过从一个简单的CoT模块开始进行代码空间上的树搜索，并通过执行跟踪上的反思指导的变异进行进化来发现。结果生成的ARM作为多功能推理构建块，可以作为直接递归循环使用，也可以作为学习元调度器中的子程序。我们的方法在人工设计的MAS和最先进的自动设计MAS方法上表现显著更好。关键的是，使用ARM构建的MAS表现出色，能够在不同基础模型和任务领域中保持高性能，无需进一步优化。 

---
# Syn-Diag: An LLM-based Synergistic Framework for Generalizable Few-shot Fault Diagnosis on the Edge 

**Title (ZH)**: Syn-Diag: 一个基于LLM的边缘端泛化少样本故障诊断协同框架 

**Authors**: Zijun Jia, Shuang Liang, Jinsong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05733)  

**Abstract**: Industrial fault diagnosis faces the dual challenges of data scarcity and the difficulty of deploying large AI models in resource-constrained environments. This paper introduces Syn-Diag, a novel cloud-edge synergistic framework that leverages Large Language Models to overcome these limitations in few-shot fault diagnosis. Syn-Diag is built on a three-tiered mechanism: 1) Visual-Semantic Synergy, which aligns signal features with the LLM's semantic space through cross-modal pre-training; 2) Content-Aware Reasoning, which dynamically constructs contextual prompts to enhance diagnostic accuracy with limited samples; and 3) Cloud-Edge Synergy, which uses knowledge distillation to create a lightweight, efficient edge model capable of online updates via a shared decision space. Extensive experiments on six datasets covering different CWRU and SEU working conditions show that Syn-Diag significantly outperforms existing methods, especially in 1-shot and cross-condition scenarios. The edge model achieves performance comparable to the cloud version while reducing model size by 83% and latency by 50%, offering a practical, robust, and deployable paradigm for modern intelligent diagnostics. 

**Abstract (ZH)**: 工业故障诊断面临数据稀缺性和在资源受限环境中部署大规模AI模型的双重挑战。本文介绍了一种新颖的云边协同框架Syn-Diag，该框架利用大型语言模型在少量样本的故障诊断中克服这些限制。Syn-Diag基于三层机制：1）跨模态语义协同，通过跨模态预训练将信号特征与LLM的语义空间对齐；2）内容感知推理，动态构建上下文提示以在样本有限的情况下提高诊断准确性；3）云边协同，利用知识蒸馏创建轻量级、高效的边缘模型，并通过共享决策空间进行在线更新。在涵盖CWRU和SEU不同工作条件的六个数据集上进行的广泛实验表明，Syn-Diag在1-shot和跨条件场景中显著优于现有方法。边缘模型在模型大小减少83%和延迟减少50%的情况下实现了与云版本相当的性能，提供了一种实用、可靠且易于部署的现代智能诊断范式。 

---
# Joint Communication Scheduling and Velocity Control for Multi-UAV-Assisted Post-Disaster Monitoring: An Attention-Based In-Context Learning Approach 

**Title (ZH)**: 基于注意力机制的上下文学习方法：多无人机辅助灾后监测的联合通信调度与速度控制 

**Authors**: Yousef Emami, Seyedsina Nabavirazavi, Jingjing Zheng, Hao Zhou, Miguel Gutierrez Gaitan, Kai Li, Luis Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2510.05698)  

**Abstract**: Recently, Unmanned Aerial Vehicles (UAVs) are increasingly being investigated to collect sensory data in post-disaster monitoring scenarios, such as tsunamis, where early actions are critical to limit coastal damage. A major challenge is to design the data collection schedules and flight velocities, as unfavorable schedules and velocities can lead to transmission errors and buffer overflows of the ground sensors, ultimately resulting in significant packet loss. Meanwhile, online Deep Reinforcement Learning (DRL) solutions have a complex training process and a mismatch between simulation and reality that does not meet the urgent requirements of tsunami monitoring. Recent advances in Large Language Models (LLMs) offer a compelling alternative. With their strong reasoning and generalization capabilities, LLMs can adapt to new tasks through In-Context Learning (ICL), which enables task adaptation through natural language prompts and example-based guidance without retraining. However, LLM models have input data limitations and thus require customized approaches. In this paper, a joint optimization of data collection schedules and velocities control for multiple UAVs is proposed to minimize data loss. The battery level of the ground sensors, the length of the queues, and the channel conditions, as well as the trajectories of the UAVs, are taken into account. Attention-Based In-Context Learning for Velocity Control and Data Collection Schedule (AIC-VDS) is proposed as an alternative to DRL in emergencies. The simulation results show that the proposed AIC-VDS outperforms both the Deep-Q-Network (DQN) and maximum channel gain baselines. 

**Abstract (ZH)**: 基于注意力增强上下文学习的多 UAV 数据收集调度与速度控制优化（AIC-VDS） 

---
# Large Language Model-Based Uncertainty-Adjusted Label Extraction for Artificial Intelligence Model Development in Upper Extremity Radiography 

**Title (ZH)**: 基于大型语言模型的不确定性调整标签提取在上肢放射成像人工智能模型开发中的应用 

**Authors**: Hanna Kreutzer, Anne-Sophie Caselitz, Thomas Dratsch, Daniel Pinto dos Santos, Christiane Kuhl, Daniel Truhn, Sven Nebelung  

**Link**: [PDF](https://arxiv.org/pdf/2510.05664)  

**Abstract**: Objectives: To evaluate GPT-4o's ability to extract diagnostic labels (with uncertainty) from free-text radiology reports and to test how these labels affect multi-label image classification of musculoskeletal radiographs. Methods: This retrospective study included radiography series of the clavicle (n=1,170), elbow (n=3,755), and thumb (n=1,978). After anonymization, GPT-4o filled out structured templates by indicating imaging findings as present ("true"), absent ("false"), or "uncertain." To assess the impact of label uncertainty, "uncertain" labels of the training and validation sets were automatically reassigned to "true" (inclusive) or "false" (exclusive). Label-image-pairs were used for multi-label classification using ResNet50. Label extraction accuracy was manually verified on internal (clavicle: n=233, elbow: n=745, thumb: n=393) and external test sets (n=300 for each). Performance was assessed using macro-averaged receiver operating characteristic (ROC) area under the curve (AUC), precision recall curves, sensitivity, specificity, and accuracy. AUCs were compared with the DeLong test. Results: Automatic extraction was correct in 98.6% (60,618 of 61,488) of labels in the test sets. Across anatomic regions, label-based model training yielded competitive performance measured by macro-averaged AUC values for inclusive (e.g., elbow: AUC=0.80 [range, 0.62-0.87]) and exclusive models (elbow: AUC=0.80 [range, 0.61-0.88]). Models generalized well on external datasets (elbow [inclusive]: AUC=0.79 [range, 0.61-0.87]; elbow [exclusive]: AUC=0.79 [range, 0.63-0.89]). No significant differences were observed across labeling strategies or datasets (p>=0.15). Conclusion: GPT-4o extracted labels from radiologic reports to train competitive multi-label classification models with high accuracy. Detected uncertainty in the radiologic reports did not influence the performance of these models. 

**Abstract (ZH)**: 对象目标：评估GPT-4o从自由文本放射学报告中提取具有不确定性诊断标签的能力，并测试这些标签如何影响肌骨X线图像的多标签分类。方法：本回顾性研究包括锁骨(n=1,170)、肘部(n=3,755)和拇指(n=1,978)的放射学系列。在匿名化后，GPT-4o通过标记成像发现为“存在”、“不存在”或“不确定”来填写结构化模板。为了评估标签不确定性的影响，“不确定”标签在训练集和验证集中的自动重新分配为“包含”为“真”或“排除”为“假”。使用ResNet50对标签-图像对进行多标签分类。通过手动验证内部（锁骨：n=233，肘部：n=745，拇指：n=393）和外部测试集（每个300）的标签提取准确性。使用宏平均接收操作特征（ROC）曲线下面积（AUC）、精确召回曲线、灵敏度、特异性和准确性评估性能。使用DeLong检验比较AUCs。结果：测试集中的标签自动提取正确率为98.6%（60,618/61,488）。在不同解剖区域，基于标签的模型训练在包含模型（例如，肘部：AUC=0.80[范围，0.62-0.87]）和排除模型（肘部：AUC=0.80[范围，0.61-0.88]）中表现出竞争性的性能。模型在外部数据集上泛化良好（肘部[包含]：AUC=0.79[范围，0.61-0.87]；肘部[排除]：AUC=0.79[范围，0.63-0.89]）。未观察到标签标注策略或数据集之间的显著差异（p≥0.15）。结论：GPT-4o从放射学报告中提取标签，训练出具有高准确性且竞争性的多标签分类模型。放射学报告中检测到的不确定性未影响这些模型的性能。 

---
# In-the-Flow Agentic System Optimization for Effective Planning and Tool Use 

**Title (ZH)**: 流动中的代理系统优化以实现有效的规划与工具使用 

**Authors**: Zhuofeng Li, Haoxiang Zhang, Seungju Han, Sheng Liu, Jianwen Xie, Yu Zhang, Yejin Choi, James Zou, Pan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05592)  

**Abstract**: Outcome-driven reinforcement learning has advanced reasoning in large language models (LLMs), but prevailing tool-augmented approaches train a single, monolithic policy that interleaves thoughts and tool calls under full context; this scales poorly with long horizons and diverse tools and generalizes weakly to new scenarios. Agentic systems offer a promising alternative by decomposing work across specialized modules, yet most remain training-free or rely on offline training decoupled from the live dynamics of multi-turn interaction. We introduce AgentFlow, a trainable, in-the-flow agentic framework that coordinates four modules (planner, executor, verifier, generator) through an evolving memory and directly optimizes its planner inside the multi-turn loop. To train on-policy in live environments, we propose Flow-based Group Refined Policy Optimization (Flow-GRPO), which tackles long-horizon, sparse-reward credit assignment by converting multi-turn optimization into a sequence of tractable single-turn policy updates. It broadcasts a single, verifiable trajectory-level outcome to every turn to align local planner decisions with global success and stabilizes learning with group-normalized advantages. Across ten benchmarks, AgentFlow with a 7B-scale backbone outperforms top-performing baselines with average accuracy gains of 14.9% on search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks, even surpassing larger proprietary models like GPT-4o. Further analyses confirm the benefits of in-the-flow optimization, showing improved planning, enhanced tool-calling reliability, and positive scaling with model size and reasoning turns. 

**Abstract (ZH)**: 基于结果的强化学习在大型语言模型中推进了推理能力，但占据主导地位的工具增强方法训练单一的、综合性策略，该策略在完整上下文中交错思维和工具调用；这在长周期和多种工具面前扩展性差，并且在新情境中泛化能力较弱。代理系统通过分解工作到专门模块中提供了有希望的替代方案，但大多数仍不需训练或依赖于与多轮交互的实时动态脱钩的离线训练。我们介绍了AgentFlow，这是一种可训练、在流程中的代理框架，通过不断演变的记忆协调四个模块（规划器、执行器、验证器、生成器），并在多轮循环内直接优化其规划器。为实时环境中的在政策训练，我们提出了基于流的分组精细策略优化（Flow-GRPO），通过将多轮优化转换为一系列可处理的单轮策略更新来解决长周期和稀疏奖励的信用分配问题。它将单个可验证的轨迹级结果广播到每次交互，使局部规划器决策与全局成功对齐，并通过分组标准化优势稳定学习。在十个基准测试中，AgentFlow使用7B规模的骨干网络在搜索、代理、数学和科学任务上的平均准确率分别提高了14.9%、14.0%、14.5%和4.1%，甚至超越了如GPT-4o等更大规模的 proprietary 模型。进一步的分析证实了流程中优化的好处，显示了改进的规划、增强的工具调用可靠性以及随模型大小和推理轮数的正向扩展。 

---
# Vul-R2: A Reasoning LLM for Automated Vulnerability Repair 

**Title (ZH)**: Vul-R2: 一个用于自动化漏洞修复的推理大语言模型 

**Authors**: Xin-Cheng Wen, Zirui Lin, Yijun Yang, Cuiyun Gao, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.05480)  

**Abstract**: The exponential increase in software vulnerabilities has created an urgent need for automatic vulnerability repair (AVR) solutions. Recent research has formulated AVR as a sequence generation problem and has leveraged large language models (LLMs) to address this problem. Typically, these approaches prompt or fine-tune LLMs to generate repairs for vulnerabilities directly. Although these methods show state-of-the-art performance, they face the following challenges: (1) Lack of high-quality, vulnerability-related reasoning data. Current approaches primarily rely on foundation models that mainly encode general programming knowledge. Without vulnerability-related reasoning data, they tend to fail to capture the diverse vulnerability repair patterns. (2) Hard to verify the intermediate vulnerability repair process during LLM training. Existing reinforcement learning methods often leverage intermediate execution feedback from the environment (e.g., sandbox-based execution results) to guide reinforcement learning training. In contrast, the vulnerability repair process generally lacks such intermediate, verifiable feedback, which poses additional challenges for model training. 

**Abstract (ZH)**: 软件漏洞的指数级增加迫切需要自动漏洞修复（AVR）解决方案。近期研究将AVR问题形式化为序列生成问题，并利用大规模语言模型（LLMs）解决这一问题。通常，这些方法通过提示或微调LLMs来直接生成漏洞修复代码。尽管这些方法展现了最先进的性能，但也面临以下挑战：（1）缺乏高质量的、与漏洞相关的推理数据。当前的方法主要依赖基础模型，主要编码一般编程知识，而缺乏与漏洞相关的推理数据，这使得它们难以捕捉到多样化的漏洞修复模式。（2）在LLMs训练过程中难以验证中间的漏洞修复过程。现有的强化学习方法通常利用环境中的中间执行反馈（如沙箱执行结果）来引导强化学习训练。相比之下，漏洞修复过程通常缺乏此类中间可验证的反馈，这为模型训练带来了额外的挑战。 

---
# VAL-Bench: Measuring Value Alignment in Language Models 

**Title (ZH)**: VAL-Bench: 测量语言模型的价值对齐程度 

**Authors**: Aman Gupta, Denny O'Shea, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2510.05465)  

**Abstract**: Large language models (LLMs) are increasingly used for tasks where outputs shape human decisions, so it is critical to test whether their responses reflect consistent human values. Existing benchmarks mostly track refusals or predefined safety violations, but these only check rule compliance and do not reveal whether a model upholds a coherent value system when facing controversial real-world issues. We introduce the \textbf{V}alue \textbf{AL}ignment \textbf{Bench}mark (\textbf{VAL-Bench}), which evaluates whether models maintain a stable value stance across paired prompts that frame opposing sides of public debates. VAL-Bench consists of 115K such pairs from Wikipedia's controversial sections. A well-aligned model should express similar underlying views regardless of framing, which we measure using an LLM-as-judge to score agreement or divergence between paired responses. Applied across leading open- and closed-source models, the benchmark reveals large variation in alignment and highlights trade-offs between safety strategies (e.g., refusals) and more expressive value systems. By providing a scalable, reproducible benchmark, VAL-Bench enables systematic comparison of how reliably LLMs embody human values. 

**Abstract (ZH)**: 价值对齐基准（VAL-Bench） 

---
# Do Code Models Suffer from the Dunning-Kruger Effect? 

**Title (ZH)**: 代码模型是否会遭受邓宁-克鲁格效应的影响？ 

**Authors**: Mukul Singh, Somya Chatterjee, Arjun Radhakrishna, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2510.05457)  

**Abstract**: As artificial intelligence systems increasingly collaborate with humans in creative and technical domains, questions arise about the cognitive boundaries and biases that shape our shared agency. This paper investigates the Dunning-Kruger Effect (DKE), the tendency for those with limited competence to overestimate their abilities in state-of-the-art LLMs in coding tasks. By analyzing model confidence and performance across a diverse set of programming languages, we reveal that AI models mirror human patterns of overconfidence, especially in unfamiliar or low-resource domains. Our experiments demonstrate that less competent models and those operating in rare programming languages exhibit stronger DKE-like bias, suggesting that the strength of the bias is proportionate to the competence of the models. 

**Abstract (ZH)**: 随着人工智能系统在创意和技术领域 increasingly 合作与人类互动，关于塑造我们共同行动的认知边界和偏见的问题应运而生。本文探讨了表现不佳知觉偏差（Dunning-Kruger Effect，DKE），即在最先进的LLM编程任务中，能力有限者过度估计自己能力的趋势。通过分析模型在多种编程语言下的置信度和表现，我们发现AI模型在不熟悉或低资源领域中表现出类似人类过度自信的模式。我们的实验表明，能力较弱的模型以及在稀有编程语言中运作的模型显示出更强的DKE类似偏见，这表明偏见的程度与模型的能力成比例。 

---
# AInstein: Assessing the Feasibility of AI-Generated Approaches to Research Problems 

**Title (ZH)**: AInstein: 评估AI生成方法解决研究问题的可行性 

**Authors**: Shambhavi Mishra, Gaurav Sahu, Marco Pedersoli, Laurent Charlin, Jose Dolz, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2510.05432)  

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet it remains unclear whether such success reflects genuine reasoning or sophisticated recall. We introduce AInstein, a framework for testing whether LLMs can generate valid solutions to AI research problems using only their pretrained parametric knowledge -- without domain-specific fine-tuning, retrieval augmentation, or other external aids. Our approach extracts distilled problem statements from high-quality ICLR 2025 submissions, then tasks specialized solver agents with proposing and refining technical solutions through iterative critique loops, mimicking the cycles of proposal, review, and revision central to scientific inquiry. We evaluate AInstein on 1,214 ICLR papers stratified by acceptance tier (Oral, Spotlight, Poster), using an LLM-as-a-judge paradigm guided by a structured rubric, complemented by targeted manual checks. Performance is assessed with three metrics: Success Rate (does the solution address the problem?), Rediscovery (does it align with human-proposed methods?), and Novelty (does it yield valid, original approaches?). Our results reveal that while LLMs can rediscover feasible solutions and occasionally propose creative alternatives, their problem-solving ability remains fragile and highly sensitive to framing. These findings provide the first large-scale evidence on the extent to which LLMs can act as autonomous scientific problem-solvers, highlighting both their latent potential and their current limitations. 

**Abstract (ZH)**: Large Language Models (LLMs)在广泛任务中的表现令人印象深刻，但其成功是否反映真正的推理能力或复杂的检索能力仍不清楚。我们提出了AInstein框架，用于测试LLMs是否仅通过其预训练参数知识就能生成解决AI研究问题的有效方案——无需领域特定的微调、检索增强或其他外部辅助。我们的方法从高质量的ICLR 2025投稿中提取精炼的问题陈述，然后通过迭代批判循环，由专门的求解代理提出并完善技术解决方案，模仿提案、审查和修订的科学探究循环。我们利用按接受级别分层的1,214篇ICLR论文（口头报告、亮点展示、海报展示），采用LLM作为评判者的方式，并辅以结构化评估标准和有针对性的手动检查进行评估。性能评估采用了三个指标：成功率（方案是否解决了问题？）、再发现（它是否与人类提出的方案一致？）和新颖性（它是否提供了有效且原创的方法？）我们的结果显示，虽然LLMs能够再发现可行的解决方案，并偶尔提出创意的替代方案，但它们的问题解决能力仍然脆弱且高度依赖问题的表述。这些发现提供了迄今为止最大的证据，表明LLMs作为自主科学研究问题解决者的潜力和局限性。 

---
# MHA-RAG: Improving Efficiency, Accuracy, and Consistency by Encoding Exemplars as Soft Prompts 

**Title (ZH)**: MHA-RAG：通过将范例编码为软提示以提高效率、准确性和一致性 

**Authors**: Abhinav Jain, Xinyu Yao, Thomas Reps, Christopher Jermaine  

**Link**: [PDF](https://arxiv.org/pdf/2510.05363)  

**Abstract**: Adapting Foundation Models to new domains with limited training data is challenging and computationally expensive. While prior work has demonstrated the effectiveness of using domain-specific exemplars as in-context demonstrations, we investigate whether representing exemplars purely as text is the most efficient, effective, and stable approach. We explore an alternative: representing exemplars as soft prompts with an exemplar order invariant model architecture. To this end, we introduce Multi-Head Attention Retrieval-Augmented Generation (MHA-RAG), a framework with the number of attention heads serving as a simple hyperparameter to control soft prompt-generation across different tasks. Across multiple question-answering benchmarks and model scales, MHA-RAG achieves a 20-point performance gain over standard RAG, while cutting inference costs by a factor of 10X GFLOPs-delivering both higher accuracy and greater efficiency, invariant to exemplar order. 

**Abstract (ZH)**: 使用有限训练数据适应新的领域基础模型具有挑战性和高昂的计算成本。虽然先前的工作已经证明了使用领域特定示例作为上下文演示的有效性，但我们探究了是否纯粹以文本形式表示示例是在不同任务中最具效率、最有效和最稳定的approach。我们探索了另一种方法：使用不变于示例顺序的模型架构表示示例为软提示。为此，我们提出了多头注意检索增强生成（MHA-RAG）框架，其中多头注意的数量作为简单的超参数来控制不同任务中的软提示生成。在多个问答基准和模型规模上，MHA-RAG 在标准 RAG 上实现了 20 点的性能提升，同时将推理成本降低了 10 倍 GFLOPs，实现了更高的准确性和更高的效率，与示例顺序无关。 

---
# Biomedical reasoning in action: Multi-agent System for Auditable Biomedical Evidence Synthesis 

**Title (ZH)**: biomedical推理在行动：可审计生物医学证据合成的多智能体系统 

**Authors**: Oskar Wysocki, Magdalena Wysocka, Mauricio Jacobo, Harriet Unsworth, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2510.05335)  

**Abstract**: We present M-Reason, a demonstration system for transparent, agent-based reasoning and evidence integration in the biomedical domain, with a focus on cancer research. M-Reason leverages recent advances in large language models (LLMs) and modular agent orchestration to automate evidence retrieval, appraisal, and synthesis across diverse biomedical data sources. Each agent specializes in a specific evidence stream, enabling parallel processing and fine-grained analysis. The system emphasizes explainability, structured reporting, and user auditability, providing complete traceability from source evidence to final conclusions. We discuss critical tradeoffs between agent specialization, system complexity, and resource usage, as well as the integration of deterministic code for validation. An open, interactive user interface allows researchers to directly observe, explore and evaluate the multi-agent workflow. Our evaluation demonstrates substantial gains in efficiency and output consistency, highlighting M-Reason's potential as both a practical tool for evidence synthesis and a testbed for robust multi-agent LLM systems in scientific research, available at this https URL. 

**Abstract (ZH)**: 我们呈现了M-Reason，这是一个在生物医学领域，特别是癌症研究中，实现透明代理驱动推理和证据整合的演示系统。M-Reason 利用大型语言模型（LLMs）和模块化代理orchestration的最新进展，自动化跨多种生物医学数据源的证据检索、评估和合成。每个代理专注于特定的证据流，实现并行处理和精细分析。该系统强调可解释性、结构化报告和用户审计，提供从原始证据到最终结论的完整可追踪性。我们讨论了代理专业化、系统复杂性和资源使用之间的关键权衡，以及确定性代码的集成以进行验证。一个开放的交互式用户界面使研究人员可以直接观察、探索和评估多代理工作流。我们的评估显示了效率和输出一致性方面的显著提升，突显了M-Reason作为证据合成实用工具和科学研发中稳健的多代理LLM系统试验平台的潜力，详情请见此网址。 

---
# BIRD-INTERACT: Re-imagining Text-to-SQL Evaluation for Large Language Models via Lens of Dynamic Interactions 

**Title (ZH)**: BIRD-INTERACT：通过动态交互视角重新构想大规模语言模型的文本到SQL评估 

**Authors**: Nan Huo, Xiaohan Xu, Jinyang Li, Per Jacobsson, Shipei Lin, Bowen Qin, Binyuan Hui, Xiaolong Li, Ge Qu, Shuzheng Si, Linheng Han, Edward Alexander, Xintong Zhu, Rui Qin, Ruihan Yu, Yiyao Jin, Feige Zhou, Weihao Zhong, Yun Chen, Hongyu Liu, Chenhao Ma, Fatma Ozcan, Yannis Papakonstantinou, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.05318)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance on single-turn text-to-SQL tasks, but real-world database applications predominantly require multi-turn interactions to handle ambiguous queries, execution errors, and evolving user requirements. Existing multi-turn benchmarks fall short by treating conversation histories as static context or limiting evaluation to read-only operations, failing to reflect production-grade database assistant challenges. We introduce BIRD-INTERACT, a benchmark that restores this realism through: (1) a comprehensive interaction environment coupling each database with a hierarchical knowledge base, metadata files, and a function-driven user simulator, enabling models to solicit clarifications, retrieve knowledge, and recover from errors without human supervision; (2) two evaluation settings consisting of a pre-defined conversational protocol (c-Interact) and an open-ended agentic setting (a-Interact) where models autonomously decide when to query the user simulator or explore the environment; (3) a challenging task suite covering the full CRUD spectrum for business-intelligence and operational use cases, guarded by executable test cases. Each task features ambiguous and follow-up sub-tasks requiring dynamic interaction. The suite comprises BIRD-INTERACT-FULL (600 tasks, up to 11,796 interactions) for comprehensive performance assessment, and BIRD-INTERACT-LITE (300 tasks with simplified databases) for detailed behavioral analysis and rapid method development. Our empirical results highlight BIRD-INTERACT's difficulty: GPT-5 completes only 8.67% of tasks in c-Interact and 17.00% in a-Interact. Analysis via memory grafting and Interaction Test-time Scaling validates the importance of effective interaction for complex, dynamic text-to-SQL tasks. 

**Abstract (ZH)**: 大型语言模型在单轮文本到SQL任务上表现出色，但在现实世界的数据库应用中，大多需要多轮交互来处理含糊查询、执行错误和不断演化的用户需求。现有的多轮交互基准存在不足，要么将对话历史视为静态上下文，要么只限制评估为只读操作，未能反映生产级别的数据库助手挑战。我们通过以下方式引入了BIRD-INTERACT基准：（1）将每个数据库与层次知识库、元数据文件和功能驱动的用户模拟器耦合，使模型能够在无需人类监督的情况下寻求澄清、检索知识和从错误中恢复；（2）包含预定义对话协议（c-Interact）和自主操作设置（a-Interact）的两种评估场景，前者模型根据预定义规则与用户模拟器交互，后者模型自主决定何时查询用户模拟器或探索环境；（3）涵盖商业智能和操作使用场景的完整CRUD光谱任务集，配有可执行测试案例。每项任务都包含含糊性和后续子任务，要求动态交互。该套件包括BIRD-INTERACT-FULL（600项任务，最多11,796次交互）用于全面性能评估，和BIRD-INTERACT-LITE（300项任务，简化数据库）用于详细行为分析和快速方法开发。我们的实证结果强调了BIRD-INTERACT的困难性：GPT-5仅在c-Interact中完成8.67%的任务，在a-Interact中完成17.00%的任务。通过内存接合分析和交互测试时缩放验证了有效交互在复杂动态文本到SQL任务中的重要性。 

---
# Beyond Monolithic Rewards: A Hybrid and Multi-Aspect Reward Optimization for MLLM Alignment 

**Title (ZH)**: 超越单一奖励：MLLM对齐的混合多方面奖励优化 

**Authors**: Radha Gulhane, Sathish Reddy Indurthi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05283)  

**Abstract**: Aligning multimodal large language models (MLLMs) with human preferences often relies on single-signal, model-based reward methods. Such monolithic rewards often lack confidence calibration across domain-specific tasks, fail to capture diverse aspects of human preferences, and require extensive data annotation and reward model training. In this work, we propose a hybrid reward modeling framework that integrates complementary reward paradigms: (i) model-based rewards, where a learned reward model predicts scalar or vector scores from synthetic and human feedback, and (ii) rule-based rewards, where domain-specific heuristics provide explicit correctness signals with confidence. Beyond accuracy, we further incorporate multi-aspect rewards to enforce instruction adherence and introduce a generalized length-penalty reward to stabilize training and improve performance. The proposed framework provides a flexible and effective approach to aligning MLLMs through reinforcement learning policy optimization. Our experiments show consistent improvements across different multimodal benchmarks when applying hybrid and multi-aspect reward modeling. Our best performing model in the 3B family achieves an overall average improvement of ~9.5% across general and math reasoning tasks. Focusing specifically on mathematical benchmarks, the model achieves a significant average improvement of ~16%, highlighting its effectiveness in mathematical reasoning and problem solving. 

**Abstract (ZH)**: 基于混合奖励建模框架的多模态大语言模型人-leaning对齐 

---
# Efficient Prediction of Pass@k Scaling in Large Language Models 

**Title (ZH)**: 大型语言模型中Pass@k缩放的高效预测 

**Authors**: Joshua Kazdan, Rylan Schaeffer, Youssef Allouah, Colin Sullivan, Kyssen Yu, Noam Levi, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2510.05197)  

**Abstract**: Assessing the capabilities and risks of frontier AI systems is a critical area of research, and recent work has shown that repeated sampling from models can dramatically increase both. For instance, repeated sampling has been shown to increase their capabilities, such as solving difficult math and coding problems, but it has also been shown to increase their potential for harm, such as being jailbroken. Such results raise a crucial question for both capability and safety forecasting: how can one accurately predict a model's behavior when scaled to a massive number of attempts, given a vastly smaller sampling budget? This question is directly relevant to model providers, who serve hundreds of millions of users daily, and to governmental regulators, who seek to prevent harms. To answer this questions, we make three contributions. First, we find that standard methods for fitting these laws suffer from statistical shortcomings that hinder predictive accuracy, especially in data-limited scenarios. Second, we remedy these shortcomings by introducing a robust estimation framework, which uses a beta-binomial distribution to generate more accurate predictions from limited data. Third, we propose a dynamic sampling strategy that allocates a greater budget to harder problems. Combined, these innovations enable more reliable prediction of rare risks and capabilities at a fraction of the computational cost. 

**Abstract (ZH)**: 评估前沿AI系统的能力和风险是研究的关键领域，近期工作表明反复采样能显著增强这两种能力。然而，反复采样也显示出可能带来更大的风险，例如被突破。这些结果引出了一个重要的问题：在给定有限采样预算的情况下，如何准确预测一个模型在大规模尝试中的行为？这一问题直接关系到每天服务数亿用户的模型提供商，以及寻求防止潜在危害的政府监管机构。为了回答这个问题，我们做出了三方面的贡献。首先，我们发现用于拟合这些定律的标准方法存在统计上的不足，这限制了预测的准确性，特别是在数据有限的情况下。其次，我们通过引入一个稳健的估计框架来解决这些问题，该框架利用beta-二项分布从有限数据生成更准确的预测。第三，我们提出了一种动态采样策略，给更困难的问题分配更多的预算。综合这些创新，能够在极低的计算成本下更可靠地预测稀有风险和能力。 

---
# Graph-based LLM over Semi-Structured Population Data for Dynamic Policy Response 

**Title (ZH)**: 基于图的半结构化人群数据动态政策响应大规模语言模型 

**Authors**: Daqian Shi, Xiaolei Diao, Jinge Wu, Honghan Wu, Xiongfeng Tang, Felix Naughton, Paulina Bondaronek  

**Link**: [PDF](https://arxiv.org/pdf/2510.05196)  

**Abstract**: Timely and accurate analysis of population-level data is crucial for effective decision-making during public health emergencies such as the COVID-19 pandemic. However, the massive input of semi-structured data, including structured demographic information and unstructured human feedback, poses significant challenges to conventional analysis methods. Manual expert-driven assessments, though accurate, are inefficient, while standard NLP pipelines often require large task-specific labeled datasets and struggle with generalization across diverse domains. To address these challenges, we propose a novel graph-based reasoning framework that integrates large language models with structured demographic attributes and unstructured public feedback in a weakly supervised pipeline. The proposed approach dynamically models evolving citizen needs into a need-aware graph, enabling population-specific analyses based on key features such as age, gender, and the Index of Multiple Deprivation. It generates interpretable insights to inform responsive health policy decision-making. We test our method using a real-world dataset, and preliminary experimental results demonstrate its feasibility. This approach offers a scalable solution for intelligent population health monitoring in resource-constrained clinical and governmental settings. 

**Abstract (ZH)**: 及时准确地分析群体级数据对于公共卫生紧急事件，如COVID-19疫情期间的有效决策至关重要。然而，半结构化数据的大量输入，包括结构化的人口统计信息和未结构化的人类反馈，给传统的分析方法带来了显著挑战。虽然手动专家驱动的评估方法非常准确，但效率低下，而标准的自然语言处理管道通常需要大量的特定任务标记数据集，并且难以在多样化的领域之间进行泛化。为了解决这些挑战，我们提出了一种基于图的推理框架，将大型语言模型与结构化的人口统计属性和未结构化的公众反馈集成到弱监督管道中。所提出的框架动态建模公民需求的变化，并将其建模为需求感知的图，使得基于关键特征（如年龄、性别和多维贫困指数）进行群体特定的分析成为可能。它生成可解释的见解，以指导响应式公共卫生政策决策。我们使用真实世界的数据集测试了该方法，并初步实验证明了其可行性。该方法为资源有限的临床和政府环境中智能人群健康管理提供了一种可扩展的解决方案。 

---
# Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents 

**Title (ZH)**: 即插即用的舞台总监：一种基于协作式LLM代理的分而治之方法，用于迭代叙事剧本精炼 

**Authors**: Wenda Xie, Chao Guo, Yanqing Jing. Junle Wang, Yisheng Lv, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05188)  

**Abstract**: Although LLMs have been widely adopted for creative content generation, a single-pass process often struggles to produce high-quality long narratives. How to effectively revise and improve long narrative scripts like scriptwriters remains a significant challenge, as it demands a comprehensive understanding of the entire context to identify global structural issues and local detailed flaws, as well as coordinating revisions at multiple granularities and locations. Direct modifications by LLMs typically introduce inconsistencies between local edits and the overall narrative requirements. To address these issues, we propose Dramaturge, a task and feature oriented divide-and-conquer approach powered by hierarchical multiple LLM agents. It consists of a Global Review stage to grasp the overall storyline and structural issues, a Scene-level Review stage to pinpoint detailed scene and sentence flaws, and a Hierarchical Coordinated Revision stage that coordinates and integrates structural and detailed improvements throughout the script. The top-down task flow ensures that high-level strategies guide local modifications, maintaining contextual consistency. The review and revision workflow follows a coarse-to-fine iterative process, continuing through multiple rounds until no further substantive improvements can be made. Comprehensive experiments show that Dramaturge significantly outperforms all baselines in terms of script-level overall quality and scene-level details. Our approach is plug-and-play and can be easily integrated into existing methods to improve the generated scripts. 

**Abstract (ZH)**: 虽然大型语言模型在创意内容生成中已被广泛应用，但单次过程常常难以生成高质量的长篇叙事。如何有效修订和改进类似编剧的长篇叙事脚本仍然是一个重大挑战，因为它需要对整个上下文有全面的理解，以识别全局结构问题和局部细节缺陷，并协调多粒度、多位置的修订。直接由大型语言模型进行修改通常会在局部编辑与整体叙事需求之间引入不一致性。为了解决这些问题，我们提出了一种名为Dramaturge的任务和特征导向的分而治之方法，该方法基于分层的多大型语言模型代理。Dramaturge包括一个全局审查阶段，以把握整体剧情线索和结构问题；一个场景级审查阶段，以确定具体场景和句子的缺陷；以及一个分层协调修订阶段，协调并整合脚本中的结构性和细节上的改进。自顶向下的任务流程确保高级策略指导局部修改，维持上下文一致性。审查和修订工作流程遵循从粗到细的迭代过程，经过多轮迭代，直到无法做出进一步实质性的改进为止。全面的实验表明，Dramaturge在脚本整体质量和场景细节方面显著优于所有基线。我们的方法即插即用，可以很容易地集成到现有方法中以改进生成的脚本。 

---
# Lang-PINN: From Language to Physics-Informed Neural Networks via a Multi-Agent Framework 

**Title (ZH)**: Lang-PINN: 通过多agent框架从语言到物理知情神经网络 

**Authors**: Xin He, Liangliang You, Hongduan Tian, Bo Han, Ivor Tsang, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2510.05158)  

**Abstract**: Physics-informed neural networks (PINNs) provide a powerful approach for solving partial differential equations (PDEs), but constructing a usable PINN remains labor-intensive and error-prone. Scientists must interpret problems as PDE formulations, design architectures and loss functions, and implement stable training pipelines. Existing large language model (LLM) based approaches address isolated steps such as code generation or architecture suggestion, but typically assume a formal PDE is already specified and therefore lack an end-to-end perspective. We present Lang-PINN, an LLM-driven multi-agent system that builds trainable PINNs directly from natural language task descriptions. Lang-PINN coordinates four complementary agents: a PDE Agent that parses task descriptions into symbolic PDEs, a PINN Agent that selects architectures, a Code Agent that generates modular implementations, and a Feedback Agent that executes and diagnoses errors for iterative refinement. This design transforms informal task statements into executable and verifiable PINN code. Experiments show that Lang-PINN achieves substantially lower errors and greater robustness than competitive baselines: mean squared error (MSE) is reduced by up to 3--5 orders of magnitude, end-to-end execution success improves by more than 50\%, and reduces time overhead by up to 74\%. 

**Abstract (ZH)**: 基于物理的知识型神经网络（Lang-PINN）：一种自然语言驱动的多Agent系统 

---
# Structuring Reasoning for Complex Rules Beyond Flat Representations 

**Title (ZH)**: 超出平面表示的复杂规则推理结构化 

**Authors**: Zhihao Yang, Ancheng Xu, Jingpeng Li, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Ahmadreza Argha, Hamid Alinejad-Rokny, Minghuan Tan, Yujun Cai, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05134)  

**Abstract**: Large language models (LLMs) face significant challenges when processing complex rule systems, as they typically treat interdependent rules as unstructured textual data rather than as logically organized frameworks. This limitation results in reasoning divergence, where models often overlook critical rule dependencies essential for accurate interpretation. Although existing approaches such as Chain-of-Thought (CoT) reasoning have shown promise, they lack systematic methodologies for structured rule processing and are particularly susceptible to error propagation through sequential reasoning chains. To address these limitations, we propose the Dynamic Adjudication Template (DAT), a novel framework inspired by expert human reasoning processes. DAT structures the inference mechanism into three methodical stages: qualitative analysis, evidence gathering, and adjudication. During the qualitative analysis phase, the model comprehensively evaluates the contextual landscape. The subsequent evidence gathering phase involves the targeted extraction of pertinent information based on predefined template elements ([placeholder]), followed by systematic verification against applicable rules. Finally, in the adjudication phase, the model synthesizes these validated components to formulate a comprehensive judgment. Empirical results demonstrate that DAT consistently outperforms conventional CoT approaches in complex rule-based tasks. Notably, DAT enables smaller language models to match, and in some cases exceed, the performance of significantly larger LLMs, highlighting its efficiency and effectiveness in managing intricate rule systems. 

**Abstract (ZH)**: 大型语言模型在处理复杂规则系统时面临显著挑战，因为它们通常将相互依赖规则视为非结构化文本数据，而不是逻辑组织框架。这种限制导致了推理偏差，模型经常忽略对于准确解释至关重要的规则依赖关系。尽管现有的方法如链式思考（CoT）显示出前景，但它们缺乏针对结构化规则处理的系统方法，并且容易通过顺序推理链传播错误。为解决这些局限，我们提出了动态裁决模板（DAT），这是一种受到专家人类推理过程启发的新框架。DAT 将推理机制分为三个系统阶段：定性分析、证据收集和裁决。在定性分析阶段，模型全面评估上下文环境。随后的证据收集阶段涉及根据预定义模板元素（[占位符]）进行有针对性的关键信息提取，并对适用规则进行系统验证。最后，在裁决阶段，模型综合这些经验证的组件以制定全面的判断。实证结果表明，DAT 在复杂的基于规则的任务中始终优于传统的 CoT 方法。值得注意的是，DAT 使较小的语言模型能够匹配甚至超过显著更大的 LLM 的性能，突显了其在管理复杂规则系统方面的效率和有效性。 

---
# Optimization Modeling via Semantic Anchored Alignment 

**Title (ZH)**: 基于语义锚点对齐的优化建模 

**Authors**: Yansen Zhang, Qingcan Kang, Yujie Chen, Yufei Wang, Xiongwei Han, Tao Zhong, Mingxuan Yuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.05115)  

**Abstract**: Large language models (LLMs) have opened new paradigms in optimization modeling by enabling the generation of executable solver code from natural language descriptions. Despite this promise, existing approaches typically remain solver-driven: they rely on single-pass forward generation and apply limited post-hoc fixes based on solver error messages, leaving undetected semantic errors that silently produce syntactically correct but logically flawed models. To address this challenge, we propose SAC-Opt, a backward-guided correction framework that grounds optimization modeling in problem semantics rather than solver feedback. At each step, SAC-Opt aligns the original semantic anchors with those reconstructed from the generated code and selectively corrects only the mismatched components, driving convergence toward a semantically faithful model. This anchor-driven correction enables fine-grained refinement of constraint and objective logic, enhancing both fidelity and robustness without requiring additional training or supervision. Empirical results on seven public datasets demonstrate that SAC-Opt improves average modeling accuracy by 7.8\%, with gains of up to 21.9\% on the ComplexLP dataset. These findings highlight the importance of semantic-anchored correction in LLM-based optimization workflows to ensure faithful translation from problem intent to solver-executable code. 

**Abstract (ZH)**: 基于语义引导修正的大型语言模型优化编译框架（SAC-Opt） 

---
# Structured Cognition for Behavioral Intelligence in Large Language Model Agents: Preliminary Study 

**Title (ZH)**: 大型语言模型代理的行为智能结构化认知：初步研究 

**Authors**: Myung Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.05107)  

**Abstract**: Large language models have advanced natural language understanding and generation, yet their use as autonomous agents raises architectural challenges for multi-step tasks. Existing frameworks often intertwine inference, memory, and control in a single prompt, which can reduce coherence and predictability. The Structured Cognitive Loop (SCL) is introduced as an alternative architecture that separates these functions. In SCL, the language model is dedicated to inference, memory is maintained externally, and execution is guided by a lightweight controller within a goal-directed loop. This design offloads cognitive load from the model and allows intermediate results to be stored, revisited, and checked before actions are taken, providing a clearer basis for traceability and evaluation.
We evaluate SCL against prompt-based baselines including ReAct and common LangChain agents across three scenarios: temperature-based travel planning, email drafting with conditional send, and constraint-guided image generation. All systems share the same base model and tools under matched decoding settings. Across 360 episodes, SCL shows modest but consistent improvements. Task success averages 86.3 percent compared with 70-77 percent for baselines. Goal fidelity is higher, redundant calls are fewer, intermediate states are reused more reliably, and unsupported assertions per 100 tool calls are reduced. Ablations show that external memory and control each contribute independently, and decoding sweeps confirm stability of the effects.
These results suggest that architectural separation can improve reliability and traceability without relying on larger models or heavier prompts. The findings are preliminary and intended to guide extended studies with additional models, longer horizons, multimodal tasks, and collaborative settings. 

**Abstract (ZH)**: 大型语言模型增强了自然语言的理解与生成能力，但作为自主代理在多步任务中的使用提出了架构挑战。现有的框架往往将推理、记忆和控制功能交织在单一的提示中，这可能会降低连贯性和可预测性。结构认知循环（SCL）作为一种替代架构被引入，它将这些功能分离。在SCL中，语言模型专注于推理，记忆由外部维护，执行由目标导向循环中的轻量级控制器指导。这种设计减轻了模型的认知负担，并允许中间结果被存储、回顾和验证，在采取行动之前，为可追溯性和评估提供了更清晰的基础。

我们评估了SCL与基于提示的基准方法（包括ReAct和常见的LangChain代理）在三个场景中的表现：基于温度的旅行规划、有条件发送的邮件草拟以及约束导向的图像生成。所有系统共享相同的基模型和工具，并且在匹配的解码设置下进行评估。在360个场景中，SCL展示了适度但持续的改进。任务成功率平均为86.3%，而基准方法的这一数字为70-77%。目标忠诚度更高，冗余调用更少，中间状态的重用更可靠，每100次工具调用中的不可支持断言更少。消融实验表明，外部记忆和控制各自独立地做出了贡献，解码扫面确认了效果的稳定性。

这些结果表明，架构分离可以在不依赖更大模型或更重提示的情况下提高可靠性和可追溯性。该研究结果初步且旨在引导后续的进一步研究，涵盖更多的模型、更长的展望、多模态任务和协作情境。 

---
# Rule Encoding and Compliance in Large Language Models: An Information-Theoretic Analysis 

**Title (ZH)**: 大型语言模型中的规则编码与合规性：一种信息论分析 

**Authors**: Joachim Diederich  

**Link**: [PDF](https://arxiv.org/pdf/2510.05106)  

**Abstract**: The design of safety-critical agents based on large language models (LLMs) requires more than simple prompt engineering. This paper presents a comprehensive information-theoretic analysis of how rule encodings in system prompts influence attention mechanisms and compliance behaviour. We demonstrate that rule formats with low syntactic entropy and highly concentrated anchors reduce attention entropy and improve pointer fidelity, but reveal a fundamental trade-off between anchor redundancy and attention entropy that previous work failed to recognize. Through formal analysis of multiple attention architectures including causal, bidirectional, local sparse, kernelized, and cross-attention mechanisms, we establish bounds on pointer fidelity and show how anchor placement strategies must account for competing fidelity and entropy objectives. Combining these insights with a dynamic rule verification architecture, we provide a formal proof that hot reloading of verified rule sets increases the asymptotic probability of compliant outputs. These findings underscore the necessity of principled anchor design and dual enforcement mechanisms to protect LLM-based agents against prompt injection attacks while maintaining compliance in evolving domains. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的安全关键代理设计不仅需要简单的提示工程。本文呈现了系统提示中的规则编码如何影响注意力机制和合规行为的全面信息论分析。我们证明，低句法熵和高度集中锚点的规则格式可以减少注意力熵并提高指针精确度，但揭示了以前的研究未曾认识到的锚点冗余与注意力熵之间的根本权衡。通过对包括因果、双向、局部稀疏、核化和交叉注意机制在内的多种注意力架构的正式分析，我们界定了指针精确度的上限，并展示了如何使锚点放置策略同时考虑竞争性的精确度和熵目标。结合这些洞察与动态规则验证架构，我们提供了形式证明，即验证规则集的热重载可以提高合规输出的渐近概率。这些发现强调了原理性的锚点设计和双重执行机制的必要性，以保护基于LLM的代理免受提示注入攻击，同时在不断变化的领域中保持合规性。 

---
# Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents 

**Title (ZH)**: 分层GRPO：处理LLM搜索代理 reinforcement learning 中的结构异质性 

**Authors**: Mingkang Zhu, Xi Chen, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.06214)  

**Abstract**: Large language model (LLM) agents increasingly rely on external tools such as search engines to solve complex, multi-step problems, and reinforcement learning (RL) has become a key paradigm for training them. However, the trajectories of search agents are structurally heterogeneous, where variations in the number, placement, and outcomes of search calls lead to fundamentally different answer directions and reward distributions. Standard policy gradient methods, which use a single global baseline, suffer from what we identify and formalize as cross-stratum bias-an "apples-to-oranges" comparison of heterogeneous trajectories. This cross-stratum bias distorts credit assignment and hinders exploration of complex, multi-step search strategies. To address this, we propose Stratified GRPO, whose central component, Stratified Advantage Normalization (SAN), partitions trajectories into homogeneous strata based on their structural properties and computes advantages locally within each stratum. This ensures that trajectories are evaluated only against their true peers. Our analysis proves that SAN eliminates cross-stratum bias, yields conditionally unbiased unit-variance estimates inside each stratum, and retains the global unbiasedness and unit-variance properties enjoyed by standard normalization, resulting in a more pure and scale-stable learning signal. To improve practical stability under finite-sample regimes, we further linearly blend SAN with the global estimator. Extensive experiments on diverse single-hop and multi-hop question-answering benchmarks demonstrate that Stratified GRPO consistently and substantially outperforms GRPO by up to 11.3 points, achieving higher training rewards, greater training stability, and more effective search policies. These results establish stratification as a principled remedy for structural heterogeneity in RL for LLM search agents. 

**Abstract (ZH)**: 大规模语言模型（LLM）代理 increasingly rely on 外部工具 如搜索引擎来解决复杂、多步问题，并且强化学习（RL）已成为训练它们的关键范式。然而，搜索代理的轨迹在结构上异质性显著，其中搜索调用的数量、位置和结果的变异性导致了根本不同的答案方向和奖励分布。标准的策略梯度方法使用单一全局基线，我们将其识别并形式化为跨层偏差——异质轨迹之间的“苹果对橙子”比较。这种跨层偏差扭曲了信用分配并阻碍了对复杂、多步搜索策略的探索。为了解决这一问题，我们提出了分层GRPO（Stratified GRPO），其核心组件分层优势标准化（SAN）基于其结构属性将轨迹划分为同质层，并在每层内局部计算优势。这确保了轨迹仅与它们的真实同层进行评估。我们的分析证明SAN消除了跨层偏差，在每层内部提供了有条件无偏的一元方差估计，并保留了标准标准化所享有的全局无偏性和一元方差属性，从而产生一种更为纯净和缩放稳定的学习信号。为进一步提高在有限样本下的实际稳定性，我们进一步将SAN线性地与全局估计器融合。在多种单跳和多跳问答基准上的广泛实验表明，分层GRPO在多个方面显著优于GRPO，实现了更高的训练奖励、更强的训练稳定性以及更有效的搜索策略。这些结果确立了分层作为解决大规模语言模型搜索代理中结构异质性的基本原则方法。 

---
# Automated Program Repair of Uncompilable Student Code 

**Title (ZH)**: 自动修复无法编译的学生代码 

**Authors**: Griffin Pitts, Aum Pandya, Darsh Rank, Tirth Bhatt, Muntasir Hoq, Bita Akram  

**Link**: [PDF](https://arxiv.org/pdf/2510.06187)  

**Abstract**: A significant portion of student programming submissions in CS1 learning environments are uncompilable, limiting their use in student modeling and downstream knowledge tracing. Traditional modeling pipelines often exclude these cases, discarding observations of student learning. This study investigates automated program repair as a strategy to recover uncompilable code while preserving students' structural intent for use in student modeling. Within this framework, we assess large language models (LLMs) as repair agents, including GPT-5 (OpenAI), Claude 3.5 Haiku (Anthropic), and Gemini 2.5 Flash (Google), under high- and low-context prompting conditions. Repairs were evaluated for compilability, edit distance, and preservation of students' original structure and logic. We find that while all three LLMs are capable of producing compilable repairs, their behavior diverges in how well they preserve students' control flow and code structure, which affects their pedagogical utility. By recovering uncompilable submissions, this work enables richer and more comprehensive analyses of learners' coding processes and development over time. 

**Abstract (ZH)**: CS1学习环境中，大量学生的编程提交无法编译，限制了其在学生建模和下游知识追踪中的应用。传统的建模管道通常排除这些情况，丢弃学生的学习观察。本研究调查了自动化程序修复策略，以恢复无法编译的代码同时保留学生的结构意图，用于学生建模。在此框架下，我们评估了大型语言模型（LLMs）作为修复代理的能力，包括GPT-5（OpenAI）、Claude 3.5 Haiku（Anthropic）和Gemini 2.5 Flash（Google），在高语境和低语境提示条件下。修复效果从编译能力、编辑距离以及保留学生的原始结构和逻辑方面进行了评估。我们发现，虽然所有三个LLM都能生成可编译的修复，但在保留学生的控制流和代码结构方面的行为差异影响了它们的教学价值。通过恢复无法编译的提交，本研究使对学习者编程过程及其随时间发展的更丰富、更全面的分析成为可能。 

---
# RECODE-H: A Benchmark for Research Code Development with Interactive Human Feedback 

**Title (ZH)**: RECODE-H：一种带有交互式人类反馈的研究代码开发基准 

**Authors**: Chunyu Miao, Henry Peng Zou, Yangning Li, Yankai Chen, Yibo Wang, Fangxin Wang, Yifan Li, Wooseong Yang, Bowei He, Xinni Zhang, Dianzhi Yu, Hanchen Yang, Hoang H Nguyen, Yue Zhou, Jie Yang, Jizhou Guo, Wenzhe Fan, Chin-Yuan Yeh, Panpan Meng, Liancheng Fang, Jinhu Qi, Wei-Chieh Huang, Zhengyao Gu, Yuwei Han, Langzhou He, Yuyao Yang, Xue Liu, Irwin King, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06186)  

**Abstract**: Large language models (LLMs) show the promise in supporting scientific research implementation, yet their ability to generate correct and executable code remains limited. Existing works largely adopt one-shot settings, ignoring the iterative and feedback-driven nature of realistic workflows of scientific research development. To address this gap, we present RECODE-H, a benchmark of 102 tasks from research papers and repositories that evaluates LLM agents through multi-turn interactions with LLM-simulated human feedback. It includes structured instructions,unit tests, and a five-level feedback hierarchy to reflect realistic researcher-agent collaboration. We further present ReCodeAgent, a framework that integrates feedback into iterative code generation. Experiments with leading LLMs, including GPT-5, Claude-Sonnet-4, DeepSeek-V3.1, and Gemini 2.5, show substantial performance gains with richer feedback, while also highlighting ongoing challenges in the generation of complex research code. RECODE-H establishes a foundation for developing adaptive, feedback-driven LLM agents in scientific research implementation 

**Abstract (ZH)**: 大型语言模型在支持科学研究实施方面展现出潜力，但在生成正确可执行代码方面仍有限制。现有工作主要采用一次性设置，忽视了科学研究开发中迭代和基于反馈的自然流程。为解决这一差距，我们提出了RECODE-H，该基准包括102项来自研究论文和仓库的任务，通过与大规模人类反馈模拟的多轮交互评估语言模型代理，其中包括结构化指令、单元测试和五级反馈层次，以反映现实的研究员-代理协作。我们进一步提出了ReCodeAgent框架，该框架将反馈整合到迭代代码生成中。使用包括GPT-5、Claude-Sonnet-4、DeepSeek-V3.1和Gemini 2.5在内的领先语言模型的实验显示，丰富的反馈带来了显著的性能提升，同时也揭示了生成复杂研究代码的持续挑战。RECODE-H为开发适应性和基于反馈的科学实施语言模型代理奠定了基础。 

---
# LLMs as Policy-Agnostic Teammates: A Case Study in Human Proxy Design for Heterogeneous Agent Teams 

**Title (ZH)**: LLMs作为无政策偏见的队友：异构代理团队中的人类代理设计案例研究 

**Authors**: Aju Ani Justus, Chris Baber  

**Link**: [PDF](https://arxiv.org/pdf/2510.06151)  

**Abstract**: A critical challenge in modelling Heterogeneous-Agent Teams is training agents to collaborate with teammates whose policies are inaccessible or non-stationary, such as humans. Traditional approaches rely on expensive human-in-the-loop data, which limits scalability. We propose using Large Language Models (LLMs) as policy-agnostic human proxies to generate synthetic data that mimics human decision-making. To evaluate this, we conduct three experiments in a grid-world capture game inspired by Stag Hunt, a game theory paradigm that balances risk and reward. In Experiment 1, we compare decisions from 30 human participants and 2 expert judges with outputs from LLaMA 3.1 and Mixtral 8x22B models. LLMs, prompted with game-state observations and reward structures, align more closely with experts than participants, demonstrating consistency in applying underlying decision criteria. Experiment 2 modifies prompts to induce risk-sensitive strategies (e.g. "be risk averse"). LLM outputs mirror human participants' variability, shifting between risk-averse and risk-seeking behaviours. Finally, Experiment 3 tests LLMs in a dynamic grid-world where the LLM agents generate movement actions. LLMs produce trajectories resembling human participants' paths. While LLMs cannot yet fully replicate human adaptability, their prompt-guided diversity offers a scalable foundation for simulating policy-agnostic teammates. 

**Abstract (ZH)**: 异质代理团队建模中的关键挑战是训练能够与政策不可访问或非稳定（如人类）的队友协作的代理。传统方法依赖昂贵的人在环数据，这限制了可扩展性。我们提出使用大语言模型（LLMs）作为政策无关的人类代理来生成模拟人类决策的数据。为此，我们在一个基于猎 stag 捕捉游戏的网格世界中进行了三项实验，该游戏借鉴了博弈论中的猎 stag 模型，平衡了风险与奖励。实验 1 将来自 30 名人类参与者和 2 名专家裁判的决策与 LLaMA 3.1 和 Mixtral 8x22B 模型的输出进行比较。在被提示游戏状态观察和奖励结构后，LLMs 的输出与专家更一致，表现出一致的应用决策标准。实验 2 修改提示以诱导风险敏感策略（如“规避风险”）。LLM 的输出反映了人类参与者的变异性，在规避风险和寻求风险之间切换。最后，实验 3 在一个动态网格世界中测试了 LLMS，LLM 代理生成了移动动作。LLM 生成的轨迹类似于人类参与者的路径。虽然 LLMS 无法完全复制人类的适应性，但它们的提示引导多样性为模拟政策无关的队友提供了可扩展的基础。 

---
# CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credits 

**Title (ZH)**: CreditDecoding: 在扩散大语言模型中加速并行解码的追踪信用方法 

**Authors**: Kangyu Wang, Zhiyun Jiang, Haibo Feng, Weijia Zhao, Lin Liu, Jianguo Li, Zhenzhong Lan, Weiyao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06133)  

**Abstract**: Diffusion large language models (dLLMs) generate text through iterative denoising steps, achieving parallel decoding by denoising only high-confidence positions at each step. However, existing approaches often repetitively remask tokens due to initially low confidence scores, leading to redundant iterations and limiting overall acceleration. Through the analysis of dLLM decoding traces, we observe that the model often determines the final prediction for a token several steps before the decoding step. To leverage this historical information and avoid redundant steps, we introduce the concept of Trace Credit, which quantifies each token's convergence potential by accumulating historical logits. Furthermore, we propose CreditDecoding, a training-free parallel decoding algorithm that accelerates the confidence convergence of correct but underconfident tokens by fusing current logits with Trace Credit. This process significantly reduces redundant iterations and enhances decoding robustness. On eight benchmarks, CreditDecoding achieves a 5.48 times speedup and a 0.48 performance improvement over LLaDA-8B-Instruct, and a 4.11 times speedup with a 0.15 performance improvement over LLaDA-MoE-Instruct. Importantly, CreditDecoding scales effectively to long sequences and is orthogonal to mainstream inference optimizations, making it a readily integrable and versatile solution. 

**Abstract (ZH)**: 基于轨迹信用的加速平行解码算法 

---
# Distributional Semantics Tracing: A Framework for Explaining Hallucinations in Large Language Models 

**Title (ZH)**: 分布语义追踪：一种解释大规模语言模型幻觉的框架 

**Authors**: Gagan Bhatia, Somayajulu G Sripada, Kevin Allan, Jacobo Azcona  

**Link**: [PDF](https://arxiv.org/pdf/2510.06107)  

**Abstract**: Large Language Models (LLMs) are prone to hallucination, the generation of plausible yet factually incorrect statements. This work investigates the intrinsic, architectural origins of this failure mode through three primary this http URL, to enable the reliable tracing of internal semantic failures, we propose \textbf{Distributional Semantics Tracing (DST)}, a unified framework that integrates established interpretability techniques to produce a causal map of a model's reasoning, treating meaning as a function of context (distributional semantics). Second, we pinpoint the model's layer at which a hallucination becomes inevitable, identifying a specific \textbf{commitment layer} where a model's internal representations irreversibly diverge from factuality. Third, we identify the underlying mechanism for these failures. We observe a conflict between distinct computational pathways, which we interpret using the lens of dual-process theory: a fast, heuristic \textbf{associative pathway} (akin to System 1) and a slow, deliberate \textbf{contextual pathway} (akin to System 2), leading to predictable failure modes such as \textit{Reasoning Shortcut Hijacks}. Our framework's ability to quantify the coherence of the contextual pathway reveals a strong negative correlation ($\rho = -0.863$) with hallucination rates, implying that these failures are predictable consequences of internal semantic weakness. The result is a mechanistic account of how, when, and why hallucinations occur within the Transformer architecture. 

**Abstract (ZH)**: 大型语言模型（LLMs）容易出现幻觉，即生成看似合理但实际上不正确的语句。本研究通过三个方面探索这种失败模式的内在、架构根源，以实现内部语义失败的可靠追踪，我们提出了一种名为**分布语义追踪（DST）**的统一框架，结合现有的解释性技术，生成模型推理的因果图谱，将意义视为上下文的函数（分布语义）。第二，我们确定了模型中导致幻觉不可避免的层级，指出了一个特定的**承诺层**，在那里模型的内部表示不可逆地脱离了事实性。第三，我们识别了这些失败的根本机制。我们观察到不同计算路径之间的冲突，并通过双重过程理论这一视角进行解释：一种快速的启发式**联想路径**（类似于系统1）和一种缓慢的、刻意的**上下文路径**（类似于系统2），导致可预测的失败模式，如**推理捷径劫持**。我们框架对上下文路径一致性的量化揭示了与幻觉率之间存在强烈的负相关（$\rho = -0.863$），暗示这些失败是内部语义薄弱的可预测后果。结果提供了一种机制性解释，说明了Transformer架构中幻觉发生的时间、条件和原因。 

---
# Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability 

**Title (ZH)**: 谱调谐：训练后处理以实现分布覆盖和上下文可引导性 

**Authors**: Taylor Sorensen, Benjamin Newman, Jared Moore, Chan Park, Jillian Fisher, Niloofar Mireshghallah, Liwei Jiang, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.06084)  

**Abstract**: Language model post-training has enhanced instruction-following and performance on many downstream tasks, but also comes with an often-overlooked cost on tasks with many possible valid answers. We characterize three desiderata for conditional distributional modeling: in-context steerability, valid output space coverage, and distributional alignment, and document across three model families how current post-training can reduce these properties. In particular, we disambiguate between two kinds of in-context learning: ICL for eliciting existing underlying knowledge or capabilities, and in-context steerability, where a model must use in-context information to override its priors and steer to a novel data generating distribution. To better evaluate and improve these desiderata, we introduce Spectrum Suite, a large-scale resource compiled from >40 data sources and spanning >90 tasks requiring models to steer to and match diverse distributions ranging from varied human preferences to numerical distributions and more. We find that while current post-training techniques help elicit underlying capabilities and knowledge, they hurt models' ability to flexibly steer in-context. To mitigate these issues, we propose Spectrum Tuning, a post-training method using Spectrum Suite to improve steerability and distributional coverage. We find that Spectrum Tuning often improves over pretrained models and their instruction-tuned counterparts, enhancing steerability, spanning more of the output space, and improving distributional alignment on held-out datasets. 

**Abstract (ZH)**: 语言模型后训练增强了指令跟随和许多下游任务的表现，但也往往在有多种可能正确答案的任务中降低了某些未被充分关注的成本。我们定义了条件概率模型的三个期望特性：上下文引导性、有效输出空间覆盖和分布对齐，并记录了当前后训练方法如何降低这些特性。特别是，我们将上下文学习区分为两种类型：用于激活现有潜在知识或能力的ICL，以及上下文引导性，即模型必须使用上下文信息克服先验知识并转向新颖的数据生成分布。为了更好地评估和改进这些期望特性，我们引入了光谱套装，这是一个大型资源库，汇集了来自超过40个数据源的数据，并涵盖了超过90个任务，要求模型引导和匹配从各种人类偏好到数值分布等多样的分布。我们发现，虽然当前的后训练技术有助于激活潜在能力和知识，但却损害了模型在上下文中灵活引导的能力。为缓解这些问题，我们提出了光谱调优，这是一种使用光谱套装改进引导性和分布覆盖的后训练方法。我们在保留集数据上发现，光谱调优通常优于预训练模型及其指令调优版本，增强了引导性，覆盖了更多的输出空间，并在分布对齐方面有所提升。 

---
# CDTP: A Large-Scale Chinese Data-Text Pair Dataset for Comprehensive Evaluation of Chinese LLMs 

**Title (ZH)**: CDTP：大规模中文数据-文本配对数据集，用于综合评估中文LLMs 

**Authors**: Chengwei Wu, Jiapu Wang, Mingyang Gao, Xingrui Zhuo, Jipeng Guo, Runlin Lei, Haoran Luo, Tianyu Chen, Haoyi Zhou, Shirui Pan, Zechao Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.06039)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks. However, Chinese LLMs face unique challenges, primarily due to the dominance of unstructured free text and the lack of structured representations in Chinese corpora. While existing benchmarks for LLMs partially assess Chinese LLMs, they are still predominantly English-centric and fail to address the unique linguistic characteristics of Chinese, lacking structured datasets essential for robust evaluation. To address these challenges, we present a Comprehensive Benchmark for Evaluating Chinese Large Language Models (CB-ECLLM) based on the newly constructed Chinese Data-Text Pair (CDTP) dataset. Specifically, CDTP comprises over 7 million aligned text pairs, each consisting of unstructured text coupled with one or more corresponding triples, alongside a total of 15 million triples spanning four critical domains. The core contributions of CDTP are threefold: (i) enriching Chinese corpora with high-quality structured information; (ii) enabling fine-grained evaluation tailored to knowledge-driven tasks; and (iii) supporting multi-task fine-tuning to assess generalization and robustness across scenarios, including Knowledge Graph Completion, Triple-to-Text generation, and Question Answering. Furthermore, we conduct rigorous evaluations through extensive experiments and ablation studies to assess the effectiveness, Supervised Fine-Tuning (SFT), and robustness of the benchmark. To support reproducible research, we offer an open-source codebase and outline potential directions for future investigations based on our insights. 

**Abstract (ZH)**: 全面评估中文大语言模型基准（CB-ECLLM）：基于新构建的中文数据-文本对（CDTP）数据集 

---
# LexiCon: a Benchmark for Planning under Temporal Constraints in Natural Language 

**Title (ZH)**: LexiCon：自然语言条件下时间约束规划的标准基准 

**Authors**: Periklis Mantenoglou, Rishi Hazra, Pedro Zuidberg Dos Martires, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2510.05972)  

**Abstract**: Owing to their reasoning capabilities, large language models (LLMs) have been evaluated on planning tasks described in natural language. However, LLMs have largely been tested on planning domains without constraints. In order to deploy them in real-world settings where adherence to constraints, in particular safety constraints, is critical, we need to evaluate their performance on constrained planning tasks. We introduce LexiCon -- a natural language-based (Lexi) constrained (Con) planning benchmark, consisting of a suite of environments, that can be used to evaluate the planning capabilities of LLMs in a principled fashion. The core idea behind LexiCon is to take existing planning environments and impose temporal constraints on the states. These constrained problems are then translated into natural language and given to an LLM to solve. A key feature of LexiCon is its extensibility. That is, the set of supported environments can be extended with new (unconstrained) environment generators, for which temporal constraints are constructed automatically. This renders LexiCon future-proof: the hardness of the generated planning problems can be increased as the planning capabilities of LLMs improve. Our experiments reveal that the performance of state-of-the-art LLMs, including reasoning models like GPT-5, o3, and R1, deteriorates as the degree of constrainedness of the planning tasks increases. 

**Abstract (ZH)**: 基于自然语言的约束规划基准LexiCon：评估大型语言模型的约束规划能力 

---
# Probing the Difficulty Perception Mechanism of Large Language Models 

**Title (ZH)**: 探究大规模语言模型的难度感知机制 

**Authors**: Sunbowen Lee, Qingyu Yin, Chak Tou Leong, Jialiang Zhang, Yicheng Gong, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05969)  

**Abstract**: Large language models (LLMs) are increasingly deployed on complex reasoning tasks, yet little is known about their ability to internally evaluate problem difficulty, which is an essential capability for adaptive reasoning and efficient resource allocation. In this work, we investigate whether LLMs implicitly encode problem difficulty in their internal representations. Using a linear probe on the final-token representations of LLMs, we demonstrate that the difficulty level of math problems can be linearly modeled. We further locate the specific attention heads of the final Transformer layer: these attention heads have opposite activation patterns for simple and difficult problems, thus achieving perception of difficulty. Our ablation experiments prove the accuracy of the location. Crucially, our experiments provide practical support for using LLMs as automatic difficulty annotators, potentially substantially reducing reliance on costly human labeling in benchmark construction and curriculum learning. We also uncover that there is a significant difference in entropy and difficulty perception at the token level. Our study reveals that difficulty perception in LLMs is not only present but also structurally organized, offering new theoretical insights and practical directions for future research. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在复杂推理任务中的应用日益增多，但它们内部评估问题难度的能力尚不清楚，这是适应性推理和高效资源分配的重要能力。在本工作中，我们探讨LLMs是否隐含地在其内部表示中编码问题难度。通过在LLM的最终词元表示上使用线性探测，我们证明了数学问题的难度可以进行线性建模。进一步分析最终Transformer层的特定注意力头：这些注意力头在简单和困难问题上表现出相反的激活模式，从而实现了难度感知。我们的消融实验证明了位置的准确性。至关重要的是，我们的实验为使用LLMs作为自动难度标注器提供了实际支持，可能显著减少基准构建和课程学习中对昂贵的人工标注的依赖。我们还发现，令牌级别的熵和难度感知存在显著差异。我们的研究揭示了LLMs中的难度感知不仅存在，而且具有结构性组织，为未来的研究提供了新的理论见解和实用方向。 

---
# EvalMORAAL: Interpretable Chain-of-Thought and LLM-as-Judge Evaluation for Moral Alignment in Large Language Models 

**Title (ZH)**: EvalMORAAL: 具有解释性链式思维和大模型作为法官评估的大型语言模型道德对齐评价方法 

**Authors**: Hadi Mohammadi, Anastasia Giachanou, Ayoub Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2510.05942)  

**Abstract**: We present EvalMORAAL, a transparent chain-of-thought (CoT) framework that uses two scoring methods (log-probabilities and direct ratings) plus a model-as-judge peer review to evaluate moral alignment in 20 large language models. We assess models on the World Values Survey (55 countries, 19 topics) and the PEW Global Attitudes Survey (39 countries, 8 topics). With EvalMORAAL, top models align closely with survey responses (Pearson's r approximately 0.90 on WVS). Yet we find a clear regional difference: Western regions average r=0.82 while non-Western regions average r=0.61 (a 0.21 absolute gap), indicating consistent regional bias. Our framework adds three parts: (1) two scoring methods for all models to enable fair comparison, (2) a structured chain-of-thought protocol with self-consistency checks, and (3) a model-as-judge peer review that flags 348 conflicts using a data-driven threshold. Peer agreement relates to survey alignment (WVS r=0.74, PEW r=0.39, both p<.001), supporting automated quality checks. These results show real progress toward culture-aware AI while highlighting open challenges for use across regions. 

**Abstract (ZH)**: 评析MORAAL：一种透明的链式思维框架，用于评估20个大型语言模型的道德一致性 

---
# LLM-FS-Agent: A Deliberative Role-based Large Language Model Architecture for Transparent Feature Selection 

**Title (ZH)**: LLM-FS-Agent：一种透明特征选择的 deliberative 角色基础大型语言模型架构 

**Authors**: Mohamed Bal-Ghaoui, Fayssal Sabri  

**Link**: [PDF](https://arxiv.org/pdf/2510.05935)  

**Abstract**: High-dimensional data remains a pervasive challenge in machine learning, often undermining model interpretability and computational efficiency. While Large Language Models (LLMs) have shown promise for dimensionality reduction through feature selection, existing LLM-based approaches frequently lack structured reasoning and transparent justification for their decisions. This paper introduces LLM-FS-Agent, a novel multi-agent architecture designed for interpretable and robust feature selection. The system orchestrates a deliberative "debate" among multiple LLM agents, each assigned a specific role, enabling collective evaluation of feature relevance and generation of detailed justifications. We evaluate LLM-FS-Agent in the cybersecurity domain using the CIC-DIAD 2024 IoT intrusion detection dataset and compare its performance against strong baselines, including LLM-Select and traditional methods such as PCA. Experimental results demonstrate that LLM-FS-Agent consistently achieves superior or comparable classification performance while reducing downstream training time by an average of 46% (statistically significant improvement, p = 0.028 for XGBoost). These findings highlight that the proposed deliberative architecture enhances both decision transparency and computational efficiency, establishing LLM-FS-Agent as a practical and reliable solution for real-world applications. 

**Abstract (ZH)**: 高维数据 remains a pervasive challenge in machine learning, often undermining model interpretability and computational efficiency. 而 Large Language Models (LLMs) 通过特征选择显示出在降维方面的潜力，但现有的基于 LLMS 的方法经常缺乏结构化的推理和透明的决策依据。本文介绍了一种新型的多智能体架构 LLM-FS-Agent，该架构设计用于可解释和稳健的特征选择。该系统 orchestrates 多个 LLM 智能体的“辩论”，每个智能体被分配特定的角色，从而使集体评估特征相关性并生成详细的依据成为可能。我们在网络安全领域使用 CIC-DIAD 2024 IoT 入侵检测数据集评估了 LLM-FS-Agent，并将其性能与包括 LLM-Select 和传统方法（如 PCA）在内的强基准进行了比较。实验结果表明，LLM-FS-Agent 在保持或达到优于传统方法的分类性能的同时，将下游训练时间减少了平均 46%（对于 XGBoost，统计显著性改进 p = 0.028）。这些发现突显了提议的 deliberative 架构既增强了决策透明度又提高了计算效率，确立了 LLM-FS-Agent 作为一种实用可靠的现实应用解决方案的地位。 

---
# Revisiting Long-context Modeling from Context Denoising Perspective 

**Title (ZH)**: 从上下文去噪视角 revisiting 长语境建模 

**Authors**: Zecheng Tang, Baibei Ji, Juntao Li, Lijun Wu, Haijia Gui, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05862)  

**Abstract**: Long-context models (LCMs) have demonstrated great potential in processing long sequences, facilitating many real-world applications. The success of LCMs can be attributed to their ability to locate implicit critical information within the context for further prediction. However, recent research reveals that LCMs are often susceptible to contextual noise, i.e., irrelevant tokens, that can mislead model attention. In this paper, we conduct a fine-grained analysis of the context noise and propose an effective metric, the Integrated Gradient (IG) score, to detect and quantify the noise information within the context. Our findings reveal that even simple mitigation of detected context noise can substantially boost the model's attention on critical tokens and benefit subsequent predictions. Building on this insight, we propose Context Denoising Training (CDT), a straightforward yet effective training strategy that improves attention on critical tokens while reinforcing their influence on model predictions. Extensive experiments across four tasks, under both context window scaling and long-context alignment settings, demonstrate the superiority of CDT. Notably, when trained with CDT, an open-source 8B model can achieve performance (50.92) comparable to GPT-4o (51.00). 

**Abstract (ZH)**: 长上下文模型中的背景噪声分析与去噪训练：提升关键信息关注与预测性能 

---
# DACP: Domain-Adaptive Continual Pre-Training of Large Language Models for Phone Conversation Summarization 

**Title (ZH)**: DACP： domaine自适应连续预训练大语言模型用于电话对话摘要 

**Authors**: Xue-Yong Fu, Elena Khasanova, Md Tahmid Rahman Laskar, Harsh Saini, Shashi Bhushan TN  

**Link**: [PDF](https://arxiv.org/pdf/2510.05858)  

**Abstract**: Large language models (LLMs) have achieved impressive performance in text summarization, yet their performance often falls short when applied to specialized domains %or conversational data that differ from their original pre-training distribution. While fine-tuning can improve summarization quality, it typically relies on costly and scarce high-quality labeled data. In this work, we explore continual pre-training as a scalable, self-supervised approach to adapt LLMs for downstream summarization tasks, particularly in the context of noisy real-world conversation transcripts. We conduct extensive experiments using large-scale, unlabeled business conversation data to investigate whether continual pre-training enhances model capabilities in conversational summarization. Our results demonstrate that continual pre-training yields substantial gains in both in-domain and out-of-domain summarization benchmarks, while maintaining strong generalization and robustness. We also analyze the effects of data selection strategies, providing practical guidelines for applying continual pre-training in summarization-focused industrial applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在文本摘要方面取得了令人印象深刻的性能，但在应用于专门领域或与原始预训练分布不同的对话数据时，其性能往往会有所下降。虽然微调可以提高摘要质量，但通常依赖于成本高且稀缺的高质量标注数据。在本工作中，我们探索持续预训练作为一种可扩展的自监督方法，以适应LLMs在下游摘要任务中的应用，特别是在嘈杂的现实世界对话转录的背景下。我们使用大规模的未标注商务对话数据进行 extensive 实验，以调查持续预训练是否能够增强模型在对话摘要方面的能力。我们的结果显示，持续预训练在领域内和领域外摘要基准测试中均能显著提高模型能力，同时保持较强的一般化能力和鲁棒性。我们还分析了数据选择策略的影响，提供了在注重摘要的工业应用中应用持续预训练的实际指南。 

---
# Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech 

**Title (ZH)**: 基于Large Language Model的文本到语音目标标记级偏好数据高效优化 

**Authors**: Rikuto Kotoge, Yuichi Sasaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.05799)  

**Abstract**: Aligning text-to-speech (TTS) system outputs with human feedback through preference optimization has been shown to effectively improve the robustness and naturalness of language model-based TTS models. Current approaches primarily require paired desirable and undesirable samples at the utterance level. However, such pairs are often limited in TTS output data, and utterance-level formulation prevents fine-grained token-level optimization needed for accurate pronunciation alignment. In this study, we propose TKTO that eliminates the need for paired data, enabling a more data-efficient training paradigm, and directly targets token-level units, automatically providing fine-grained alignment signals without token-level annotations. TKTO improves the challenging Japanese TTS accuracy by 39% and reduces CER by 54%, automatically assigning 12.8 times stronger reward to targeted tokens. 

**Abstract (ZH)**: 通过偏好优化将文本-to-语音（TTS）系统输出与人类反馈对齐以提高基于语言模型的TTS模型的鲁棒性和自然度 

---
# Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies 

**Title (ZH)**: 超越显式参考策略的离散扩散去遮盖策略改进 

**Authors**: Chunsan Hong, Seonho An, Min-Soo Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.05725)  

**Abstract**: Masked diffusion models (MDMs) have recently emerged as a novel framework for language modeling. MDMs generate sentences by iteratively denoising masked sequences, filling in [MASK] tokens step by step. Although MDMs support any-order sampling, performance is highly sensitive to the choice of which position to unmask next. Prior work typically relies on rule-based schedules (e.g., max-confidence, max-margin), which provide ad hoc improvements. In contrast, we replace these heuristics with a learned scheduler. Specifically, we cast denoising as a KL-regularized Markov decision process (MDP) with an explicit reference policy and optimize a regularized objective that admits policy improvement and convergence guarantees under standard assumptions. We prove that the optimized policy under this framework generates samples that more closely match the data distribution than heuristic schedules. Empirically, across four benchmarks, our learned policy consistently outperforms max-confidence: for example, on SUDOKU, where unmasking order is critical, it yields a 20.1% gain over random and a 11.2% gain over max-confidence. 

**Abstract (ZH)**: 掩码扩散模型（MDMs）近期已成为一种新颖的语言建模框架。MDMs通过迭代去噪掩蔽序列，逐步填充[MASK]令牌来生成句子。尽管MDMs支持任意顺序采样，但性能高度依赖于下一步解掩蔽哪个位置的令牌。以往工作通常依赖基于规则的时间表（例如，最大置信度、最大差距），这些方法提供临时改进。相反，我们用一个学习得到的时间表替换这些启发式方法。具体而言，我们将去噪任务视为在标准假设下具有明确参考策略的KL正则化马尔可夫决策过程（MDP），并优化一个能够在标准假设下保证策略改进和收敛性的正则化目标函数。我们证明，这种框架下的优化策略生成的样本与数据分布更加契合。在四个基准测试中，我们的学习得到的时间表始终优于最大置信度：例如，在SUDOKU中，由于解掩蔽顺序至关重要，它分别比随机方法和最大置信度方法提高了20.1%和11.2%。 

---
# Towards Reliable and Practical LLM Security Evaluations via Bayesian Modelling 

**Title (ZH)**: 基于贝叶斯建模的可靠且实用的大语言模型安全评估方法 

**Authors**: Mary Llewellyn, Annie Gray, Josh Collyer, Michael Harries  

**Link**: [PDF](https://arxiv.org/pdf/2510.05709)  

**Abstract**: Before adopting a new large language model (LLM) architecture, it is critical to understand vulnerabilities accurately. Existing evaluations can be difficult to trust, often drawing conclusions from LLMs that are not meaningfully comparable, relying on heuristic inputs or employing metrics that fail to capture the inherent uncertainty. In this paper, we propose a principled and practical end-to-end framework for evaluating LLM vulnerabilities to prompt injection attacks. First, we propose practical approaches to experimental design, tackling unfair LLM comparisons by considering two practitioner scenarios: when training an LLM and when deploying a pre-trained LLM. Second, we address the analysis of experiments and propose a Bayesian hierarchical model with embedding-space clustering. This model is designed to improve uncertainty quantification in the common scenario that LLM outputs are not deterministic, test prompts are designed imperfectly, and practitioners only have a limited amount of compute to evaluate vulnerabilities. We show the improved inferential capabilities of the model in several prompt injection attack settings. Finally, we demonstrate the pipeline to evaluate the security of Transformer versus Mamba architectures. Our findings show that consideration of output variability can suggest less definitive findings. However, for some attacks, we find notably increased Transformer and Mamba-variant vulnerabilities across LLMs with the same training data or mathematical ability. 

**Abstract (ZH)**: 一种 principled 和实用的端到端框架：评估大语言模型对提示注入攻击的漏洞 

---
# Uncovering Representation Bias for Investment Decisions in Open-Source Large Language Models 

**Title (ZH)**: 揭开开源大规模语言模型中表示偏见以辅助投资决策 

**Authors**: Fabrizio Dimino, Krati Saxena, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2510.05702)  

**Abstract**: Large Language Models are increasingly adopted in financial applications to support investment workflows. However, prior studies have seldom examined how these models reflect biases related to firm size, sector, or financial characteristics, which can significantly impact decision-making. This paper addresses this gap by focusing on representation bias in open-source Qwen models. We propose a balanced round-robin prompting method over approximately 150 U.S. equities, applying constrained decoding and token-logit aggregation to derive firm-level confidence scores across financial contexts. Using statistical tests and variance analysis, we find that firm size and valuation consistently increase model confidence, while risk factors tend to decrease it. Confidence varies significantly across sectors, with the Technology sector showing the greatest variability. When models are prompted for specific financial categories, their confidence rankings best align with fundamental data, moderately with technical signals, and least with growth indicators. These results highlight representation bias in Qwen models and motivate sector-aware calibration and category-conditioned evaluation protocols for safe and fair financial LLM deployment. 

**Abstract (ZH)**: 大型语言模型在金融应用中的投资工作流程中越来越受到采用，但先前的研究很少考察这些模型在公司规模、行业或财务特征方面的偏见，这些问题可能显著影响决策。本文通过关注开源Qwen模型中的表示偏见，填补了这一空白。我们提出了一种平衡的轮循提示方法，应用于大约150家美国股票，采用受限解码和标记-概率聚合来推导出跨财务情境的公司级置信分数。通过统计检验和方差分析，我们发现，公司规模和估值一致地提高模型置信度，而风险因素则倾向于降低置信度。置信度在不同行业之间差异显著，科技行业显示出最大的变异。当模型被提示特定的财务类别时，它们的置信度排名与基本面数据最接近，与技术信号中等程度接近，与增长指标最不接近。这些结果揭示了Qwen模型中的表示偏见，并促进了对安全和公平的金融LLM部署的行业意识校准和类别条件评估协议的需求。 

---
# Membership Inference Attacks on Tokenizers of Large Language Models 

**Title (ZH)**: 大型语言模型分词器的成员推理攻击 

**Authors**: Meng Tong, Yuntao Du, Kejiang Chen, Weiming Zhang, Ninghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.05699)  

**Abstract**: Membership inference attacks (MIAs) are widely used to assess the privacy risks associated with machine learning models. However, when these attacks are applied to pre-trained large language models (LLMs), they encounter significant challenges, including mislabeled samples, distribution shifts, and discrepancies in model size between experimental and real-world settings. To address these limitations, we introduce tokenizers as a new attack vector for membership inference. Specifically, a tokenizer converts raw text into tokens for LLMs. Unlike full models, tokenizers can be efficiently trained from scratch, thereby avoiding the aforementioned challenges. In addition, the tokenizer's training data is typically representative of the data used to pre-train LLMs. Despite these advantages, the potential of tokenizers as an attack vector remains unexplored. To this end, we present the first study on membership leakage through tokenizers and explore five attack methods to infer dataset membership. Extensive experiments on millions of Internet samples reveal the vulnerabilities in the tokenizers of state-of-the-art LLMs. To mitigate this emerging risk, we further propose an adaptive defense. Our findings highlight tokenizers as an overlooked yet critical privacy threat, underscoring the urgent need for privacy-preserving mechanisms specifically designed for them. 

**Abstract (ZH)**: 基于标记器的成员推理攻击：探索最先进的大语言模型标记器中的隐私威胁 

---
# Code-Switching In-Context Learning for Cross-Lingual Transfer of Large Language Models 

**Title (ZH)**: 基于上下文的代码转换学习在跨语言迁移大型语言模型中的应用 

**Authors**: Haneul Yoo, Jiho Jin, Kyunghyun Cho, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.05678)  

**Abstract**: While large language models (LLMs) exhibit strong multilingual abilities, their reliance on English as latent representations creates a translation barrier, where reasoning implicitly depends on internal translation into English. When this process fails, performance in non-English languages deteriorates sharply, limiting the inclusiveness of LLM-based applications. Existing cross-lingual in-context learning (X-ICL) methods primarily leverage monolingual demonstrations, often failing to mitigate this barrier and instead reinforcing it. In this work, we introduce code-switching in-context learning (CSICL), a simple yet effective prompting strategy that progressively transitions from a target language to English within demonstrations and instruction to facilitate their latent reasoning in English. By explicitly scaffolding the reasoning process through controlled code-switching, CSICL acts as an implicit linguistic bridge that enhances cross-lingual alignment and reduces reliance on the translation barrier. We conduct extensive experiments across 4 LLMs, 6 datasets, and 10 languages, spanning both knowledge-intensive and reasoning-oriented domains. Our results demonstrate that CSICL consistently outperforms X-ICL baselines, achieving gains of 3.1%p and 1.9%p in both target and unseen languages, respectively. The improvement is even more pronounced in low-resource settings, with gains of 14.7% in target and 5.3% in unseen languages. These findings establish code-switching as a principled and robust approach for overcoming the translation barrier during inference, moving LLMs toward more equitable and effective multilingual systems. 

**Abstract (ZH)**: 代码转换在场学习（CSICL）：一种促进跨语言一致性的简明有效策略 

---
# AutoPentester: An LLM Agent-based Framework for Automated Pentesting 

**Title (ZH)**: AutoPentester：基于LLM代理的自动化渗透测试框架 

**Authors**: Yasod Ginige, Akila Niroshan, Sajal Jain, Suranga Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2510.05605)  

**Abstract**: Penetration testing and vulnerability assessment are essential industry practices for safeguarding computer systems. As cyber threats grow in scale and complexity, the demand for pentesting has surged, surpassing the capacity of human professionals to meet it effectively. With advances in AI, particularly Large Language Models (LLMs), there have been attempts to automate the pentesting process. However, existing tools such as PentestGPT are still semi-manual, requiring significant professional human interaction to conduct pentests. To this end, we propose a novel LLM agent-based framework, AutoPentester, which automates the pentesting process. Given a target IP, AutoPentester automatically conducts pentesting steps using common security tools in an iterative process. It can dynamically generate attack strategies based on the tool outputs from the previous iteration, mimicking the human pentester approach. We evaluate AutoPentester using Hack The Box and custom-made VMs, comparing the results with the state-of-the-art PentestGPT. Results show that AutoPentester achieves a 27.0% better subtask completion rate and 39.5% more vulnerability coverage with fewer steps. Most importantly, it requires significantly fewer human interactions and interventions compared to PentestGPT. Furthermore, we recruit a group of security industry professional volunteers for a user survey and perform a qualitative analysis to evaluate AutoPentester against industry practices and compare it with PentestGPT. On average, AutoPentester received a score of 3.93 out of 5 based on user reviews, which was 19.8% higher than PentestGPT. 

**Abstract (ZH)**: 渗透测试和漏洞评估是保障计算机系统安全的重要行业实践。随着网络威胁规模和复杂性的增加，对渗透测试的需求激增，超过了专业人力的能力范围。随着人工智能的发展，特别是大型语言模型（LLMs），已经尝试自动化渗透测试流程。然而，现有的工具如PentestGPT仍需大量专业人工交互来执行渗透测试。为此，我们提出了一种基于大型语言模型代理的新颖框架AutoPentester，以自动化渗透测试流程。给定目标IP，AutoPentester使用常见的安全工具自动执行渗透测试步骤，并以迭代过程进行。它可以根据上一轮工具输出动态生成攻击策略，模拟人工渗透测试者的方法。我们使用Hack The Box和自定义VM评估AutoPentester，并将其结果与最新的PentestGPT进行比较。结果显示，AutoPentester的任务完成率提高了27.0%，漏洞覆盖面积增加了39.5%，且所需步骤较少。最重要的是，与PentestGPT相比，它需要明显较少的人工交互和干预。此外，我们招募了一组安全行业专业志愿者进行用户调查，并进行定性分析，评估AutoPentester与行业实践的符合程度，并将其与PentestGPT进行比较。根据用户评分，AutoPentester的平均得分为3.93，比PentestGPT高19.8%。 

---
# AgentDR Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents 

**Title (ZH)**: 基于LLM代理的动态推荐系统：通过隐式项-项关系 

**Authors**: Mingdai Yang, Nurendra Choudhary, Jiangshu Du, Edward W.Huang, Philip S.Yu, Karthik Subbian, Danai Kourta  

**Link**: [PDF](https://arxiv.org/pdf/2510.05598)  

**Abstract**: Recent agent-based recommendation frameworks aim to simulate user behaviors by incorporating memory mechanisms and prompting strategies, but they struggle with hallucinating non-existent items and full-catalog ranking. Besides, a largely underexplored opportunity lies in leveraging LLMs'commonsense reasoning to capture user intent through substitute and complement relationships between items, which are usually implicit in datasets and difficult for traditional ID-based recommenders to capture. In this work, we propose a novel LLM-agent framework, AgenDR, which bridges LLM reasoning with scalable recommendation tools. Our approach delegates full-ranking tasks to traditional models while utilizing LLMs to (i) integrate multiple recommendation outputs based on personalized tool suitability and (ii) reason over substitute and complement relationships grounded in user history. This design mitigates hallucination, scales to large catalogs, and enhances recommendation relevance through relational reasoning. Through extensive experiments on three public grocery datasets, we show that our framework achieves superior full-ranking performance, yielding on average a twofold improvement over its underlying tools. We also introduce a new LLM-based evaluation metric that jointly measures semantic alignment and ranking correctness. 

**Abstract (ZH)**: 基于代理的 recent 推荐框架 aims 致力于通过集成记忆机制和提示策略来模拟用户行为，但它们在 hallucinate 非存在的项目和全目录排名方面存在困难。此外，尚有大量未充分利用的机会在于利用大规模语言模型 (LLM) 的常识推理来通过项目之间的替代和补充关系捕捉用户意图，这些关系通常在数据集中是隐含的，传统基于 ID 的推荐器难以捕捉。在这项工作中，我们提出了一种名为 AgenDR 的新型 LLM-代理框架，该框架将 LLM 推理与可扩展的推荐工具相结合。我们的方法将全排名任务委托给传统模型，同时利用 LLMs：(i) 根据个性化工具适合度集成多个推荐输出，(ii) 基于用户历史进行替代和补充关系的推理。这种设计减轻了 hallucination、适用于大目录，并通过关系推理增强推荐的相关性。通过在三个公开的杂货数据集上的广泛实验，我们展示了我们框架实现了优于其底层工具的全排名性能，平均改善幅度超过一倍。我们还引入了一个基于 LLM 的新评估指标，该指标联合衡量语义一致性和排名准确性。 

---
# Domain-Shift-Aware Conformal Prediction for Large Language Models 

**Title (ZH)**: 面向域变化的同态预测方法用于大型语言模型 

**Authors**: Zhexiao Lin, Yuanyuan Li, Neeraj Sarna, Yuanyuan Gao, Michael von Gablenz  

**Link**: [PDF](https://arxiv.org/pdf/2510.05566)  

**Abstract**: Large language models have achieved impressive performance across diverse tasks. However, their tendency to produce overconfident and factually incorrect outputs, known as hallucinations, poses risks in real world applications. Conformal prediction provides finite-sample, distribution-free coverage guarantees, but standard conformal prediction breaks down under domain shift, often leading to under-coverage and unreliable prediction sets. We propose a new framework called Domain-Shift-Aware Conformal Prediction (DS-CP). Our framework adapts conformal prediction to large language models under domain shift, by systematically reweighting calibration samples based on their proximity to the test prompt, thereby preserving validity while enhancing adaptivity. Our theoretical analysis and experiments on the MMLU benchmark demonstrate that the proposed method delivers more reliable coverage than standard conformal prediction, especially under substantial distribution shifts, while maintaining efficiency. This provides a practical step toward trustworthy uncertainty quantification for large language models in real-world deployment. 

**Abstract (ZH)**: 面向域迁移的统一预测 (Domain-Shift-Aware Conformal Prediction) 

---
# Critical attention scaling in long-context transformers 

**Title (ZH)**: 长上下文变换器中的关键注意尺度变换 

**Authors**: Shi Chen, Zhengjiang Lin, Yury Polyanskiy, Philippe Rigollet  

**Link**: [PDF](https://arxiv.org/pdf/2510.05554)  

**Abstract**: As large language models scale to longer contexts, attention layers suffer from a fundamental pathology: attention scores collapse toward uniformity as context length $n$ increases, causing tokens to cluster excessively, a phenomenon known as rank-collapse. While $\textit{attention scaling}$ effectively addresses this deficiency by rescaling attention scores with a polylogarithmic factor $\beta_n$, theoretical justification for this approach remains lacking.
We analyze a simplified yet tractable model that magnifies the effect of attention scaling. In this model, attention exhibits a phase transition governed by the scaling factor $\beta_n$: insufficient scaling collapses all tokens to a single direction, while excessive scaling reduces attention to identity, thereby eliminating meaningful interactions between tokens. Our main result identifies the critical scaling $\beta_n \asymp \log n$ and provides a rigorous justification for attention scaling in YaRN and Qwen, clarifying why logarithmic scaling maintains sparse, content-adaptive attention at large context lengths. 

**Abstract (ZH)**: 随着大型语言模型处理更长的上下文，注意力层遭受一种基本的病理现象：随着上下文长度 \(n\) 的增加，注意力分数趋向均匀，导致令牌过度聚集，这种现象称为秩塌陷。虽然注意力缩放通过使用.polylogarithmic 因子 \(\beta_n\) 重新缩放注意力分数有效解决了这一缺陷，但这种方法的理论依据仍然缺乏。

我们分析了一个简化但可处理的模型，该模型放大了注意力缩放的效果。在这种模型中，注意力由缩放因子 \(\beta_n\) 控制的相变过程所支配：缩放不足会使所有令牌聚集成一个方向，而缩放过度则将注意力压缩为恒等映射，从而消除令牌间的有意义交互。我们的主要结果确定了临界缩放 \(\beta_n \asymp \log n\)，并为 YaRN 和 Qwen 中的注意力缩放提供了严格的理论依据，解释了为什么对数缩放在长上下文长度下能够保持稀疏和内容自适应的注意力机制。 

---
# Provably Mitigating Corruption, Overoptimization, and Verbosity Simultaneously in Offline and Online RLHF/DPO Alignment 

**Title (ZH)**: 证明性缓解离线和在线RLHF/DPO对齐中的腐败、过度优化和冗余问题 

**Authors**: Ziyi Chen, Junyi Li, Peiran Yu, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05526)  

**Abstract**: Reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO) are important techniques to align large language models (LLM) with human preference. However, the quality of RLHF and DPO training is seriously compromised by \textit{\textbf{C}orrupted} preference, reward \textit{\textbf{O}veroptimization}, and bias towards \textit{\textbf{V}erbosity}. To our knowledge, most existing works tackle only one of these important issues, and the few other works require much computation to estimate multiple reward models and lack theoretical guarantee of generalization ability. In this work, we propose RLHF-\textbf{COV} and DPO-\textbf{COV} algorithms that can simultaneously mitigate these three issues, in both offline and online settings. This ability is theoretically demonstrated by obtaining length-regularized generalization error rates for our DPO-COV algorithms trained on corrupted data, which match the best-known rates for simpler cases with clean data and without length regularization. Moreover, our DPO-COV algorithm is simple to implement without reward estimation, and is proved to be equivalent to our RLHF-COV algorithm, which directly implies the equivalence between the vanilla RLHF and DPO algorithms. Experiments demonstrate the effectiveness of our DPO-COV algorithms under both offline and online settings. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF-COV）和直接偏好优化（DPO-COV）：同时缓解污染偏好、过度优化和冗余性问题 

---
# CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension 

**Title (ZH)**: CAM：基于构建主义视角的代理记忆在基于LLM的阅读理解中的应用 

**Authors**: Rui Li, Zeyu Zhang, Xiaohe Bo, Zihang Tian, Xu Chen, Quanyu Dai, Zhenhua Dong, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05520)  

**Abstract**: Current Large Language Models (LLMs) are confronted with overwhelming information volume when comprehending long-form documents. This challenge raises the imperative of a cohesive memory module, which can elevate vanilla LLMs into autonomous reading agents. Despite the emergence of some heuristic approaches, a systematic design principle remains absent. To fill this void, we draw inspiration from Jean Piaget's Constructivist Theory, illuminating three traits of the agentic memory -- structured schemata, flexible assimilation, and dynamic accommodation. This blueprint forges a clear path toward a more robust and efficient memory system for LLM-based reading comprehension. To this end, we develop CAM, a prototype implementation of Constructivist Agentic Memory that simultaneously embodies the structurality, flexibility, and dynamicity. At its core, CAM is endowed with an incremental overlapping clustering algorithm for structured memory development, supporting both coherent hierarchical summarization and online batch integration. During inference, CAM adaptively explores the memory structure to activate query-relevant information for contextual response, akin to the human associative process. Compared to existing approaches, our design demonstrates dual advantages in both performance and efficiency across diverse long-text reading comprehension tasks, including question answering, query-based summarization, and claim verification. 

**Abstract (ZH)**: 当前的大规模语言模型在理解长文档时面临着巨大的信息量挑战。这促使我们必须开发一个协调的记忆模块，以提升基础的大规模语言模型为自主阅读代理。尽管已经出现了一些启发式方法，但系统的设计原则仍然缺失。为弥补这一空白，我们从 jean Piaget 的建构主义理论中汲取灵感，阐述了代理记忆的三大特征——结构化的图式、灵活的同化和动态的顺应。这一蓝图为基于大规模语言模型的阅读理解提供了更为坚固和高效的记忆系统框架。为了实现这一目标，我们开发了CAM，这是一种构建成败者的原型实现，同时具备结构化、灵活性和动态性。CAM的核心在于递增重叠聚类算法，以支持有序层级总结和在线批处理集成。在推理过程中，CAM能够自适应地探索记忆结构，激活与查询相关的信息以生成上下文响应，类似于人类的联想过程。与现有方法相比，我们的设计在多种长文本阅读理解任务中，包括问答、查询驱动的总结和断言验证中，展示了在性能和效率上的双重重叠优势。 

---
# Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting 

**Title (ZH)**: 混沌中的秩序：通过数据移动预测提升大规模MoE语言模型服务 

**Authors**: Zhongkai Yu, Yue Guan, Zihao Yu, Chenyang Zhou, Shuyi Pei, Yangwook Kang, Yufei Ding, Po-An Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2510.05497)  

**Abstract**: Large Language Models (LLMs) with Mixture of Experts (MoE) architectures achieve remarkable performance improvements, but their random expert selection mechanism introduces significant data movement overhead that becomes the dominant bottleneck in multi-unit serving systems. To forecast the patterns underlying this data movement, we conduct comprehensive data-movement-centric profiling across three state-of-the-art large-scale MoE models (200B- 671B) using over 24,000 requests spanning diverse workloads. With the resulting 150GB+ trace files, we perform systematic analysis from both temporal and spatial perspectives and distill six key insights to guide the design of diverse future serving systems. Taking wafer-scale GPUs as a case study, we demonstrate that minor architectural modifications leveraging our insights achieve substantial performance gains, delivering 6.3X and 4.0X average speedups on DeepSeek V3 and Qwen3, respectively. Our work provides the first comprehensive data-centric analysis of MoE models at scale. Our profiling traces and analysis results are publicly available at {this https URL. We will also release our simulation framework shortly to facilitate future research in this area. 

**Abstract (ZH)**: 大规模语言模型（LLMs）采用专家混合（MoE）架构实现了显著的性能提升，但其随机专家选择机制引入了重要的数据移动开销，成为多单元服务系统中的主要瓶颈。为了预测这种数据移动背后的模式，我们使用超过24,000个请求，在三个先进的大规模MoE模型（200B-671B）上进行全面的数据移动中心化剖析。基于所产生的150GB以上的跟踪文件，我们从时间和空间两个维度进行了系统的分析，并总结出六条关键洞察以指导未来多样化的服务系统设计。以晶圆级GPU为例，我们证明了利用这些洞察进行的小架构修改实现了显著的性能提升，分别在DeepSeek V3和Qwen3上平均提升了6.3倍和4.0倍。我们的工作提供了首次全面的数据为中心的大规模MoE模型分析。我们的剖析跟踪数据和分析结果已公开发布在{this https URL. 我们还将很快发布我们的仿真框架，以便于未来对此领域的研究。 

---
# LANTERN: Scalable Distillation of Large Language Models for Job-Person Fit and Explanation 

**Title (ZH)**: LANTERN: 大型语言模型可扩展蒸馏及其在岗位匹配与解释中的应用 

**Authors**: Zhoutong Fu, Yihan Cao, Yi-Lin Chen, Aman Lunia, Liming Dong, Neha Saraf, Ruijie Jiang, Yun Dai, Qingquan Song, Tan Wang, Guoyao Li, Derek Koh, Haichao Wei, Zhipeng Wang, Aman Gupta, Chengming Jiang, Jianqiang Shen, Liangjie Hong, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05490)  

**Abstract**: Large language models (LLMs) have achieved strong performance across a wide range of natural language processing tasks. However, deploying LLMs at scale for domain specific applications, such as job-person fit and explanation in job seeking platforms, introduces distinct challenges. At LinkedIn, the job person fit task requires analyzing a candidate's public profile against job requirements to produce both a fit assessment and a detailed explanation. Directly applying open source or finetuned LLMs to this task often fails to yield high quality, actionable feedback due to the complexity of the domain and the need for structured outputs. Moreover, the large size of these models leads to high inference latency and limits scalability, making them unsuitable for online use. To address these challenges, we introduce LANTERN, a novel LLM knowledge distillation framework tailored specifically for job person fit tasks. LANTERN involves modeling over multiple objectives, an encoder model for classification purpose, and a decoder model for explanation purpose. To better distill the knowledge from a strong black box teacher model to multiple downstream models, LANTERN incorporates multi level knowledge distillation that integrates both data and logit level insights. In addition to introducing the knowledge distillation framework, we share our insights on post training techniques and prompt engineering, both of which are crucial for successfully adapting LLMs to domain specific downstream tasks. Extensive experimental results demonstrate that LANTERN significantly improves task specific metrics for both job person fit and explanation. Online evaluations further confirm its effectiveness, showing measurable gains in job seeker engagement, including a 0.24\% increase in apply rate and a 0.28\% increase in qualified applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多种自然语言处理任务中取得了强大的性能。然而，将LLMs大规模应用于特定领域的应用，如职位与候选人匹配和求职平台上的解释，引入了独特的挑战。在LinkedIn，职位与候选人匹配任务要求分析候选人的公开资料与职位要求以产生匹配评估和详细解释。直接将开源或微调的LLMs应用于此任务，由于领域复杂性和对结构化输出的需求，往往难以获得高质量、可操作的反馈。此外，这些模型的大量级导致推理延迟高，限制了其可扩展性，使其不适合在线使用。为应对这些挑战，我们引入了LANTERN，这是一种专门针对职位与候选人匹配任务的知识蒸馏框架。LANTERN包括多目标建模、用于分类目的的编码器模型和用于解释目的的解码器模型。为了更好地将强黑盒教师模型的知识蒸馏到多个下游模型中，LANTERN结合了多层次的知识蒸馏，整合了数据和logit级别见解。除了介绍知识蒸馏框架，我们还分享了关于后训练技术和提示工程的见解，两者对于成功适应特定领域下游任务的LLMs至关重要。实验结果表明，LANTERN在职位与候选人匹配和解释任务特定指标上都有显著改进。在线评估进一步证实其有效性，显示出求职者参与度的可测量提升，包括申请率提高了0.24%，合格申请增加了0.28%。 

---
# AMAQ: Adaptive Mixed-bit Activation Quantization for Collaborative Parameter Efficient Fine-tuning 

**Title (ZH)**: AMAQ: 自适应混合位宽激活量化的协作参数高效微调 

**Authors**: Yurun Song, Zhuoyi Yang, Ian G. Harris, Sangeetha Abdu Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05468)  

**Abstract**: Large Language Models (LLMs) are scaling rapidly, creating significant challenges for collaborative server client distributed training, particularly in terms of communication efficiency and computational overheads. To address these challenges, we implement Parameter-efficient Split Learning, which effectively balances efficiency and performance for collaborative training on low-resource devices.
To reduce communication overhead in collaborative training, we introduce Adaptive Mixed bit Activation Quantization (AMAQ), a strategy that progressively compresses activations and gradients from high precision (6 to 8 bits) to low precision (3 to 4 bits). AMAQ achieves this by effectively allocating bit budgets across channels based on feature wise and layer wise importance using bit regularization.
Under the same bit budgets, AMAQ outperforms fixed-precision approaches, delivering about 2.5% higher generation accuracy and about 1.3% better classification accuracy for models like LLaMA3 8B and Qwen2.5 7B. In addition, it significantly enhances training stability and reducing ultra-low bit representation collapse during the training.
Experiments demonstrate that AMAQ integrates effectively into practical multi-machine collaborative training setups, offering superior inference accuracy with only a modest communication overhead for bits adaptation during training. This trade off makes AMAQ a practical and effective solution for collaborative training with minimal communication cost. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的规模正在迅速扩大，这为协作服务器客户端分布式训练带来了显著挑战，特别是在通信效率和计算开销方面。为了解决这些挑战，我们实施了参数高效的拆分学习，该方法有效地在低资源设备上实现了效率与性能的平衡。
为了减少协作训练中的通信开销，我们引入了自适应混合位激活量化（AMAQ）策略，该策略逐步将激活和梯度从高精度（6至8位）压缩到低精度（3至4位）。AMAQ通过基于特征和层的重要性使用位正则化有效分配位预算。
在相同的位预算下，AMAQ优于固定精度方法，对于LLaMA3 8B和Qwen2.5 7B等模型，其生成准确率提高约2.5%，分类准确率提高约1.3%。此外，它还能显著增强训练稳定性，并在训练过程中减少超低位表示坍塌。
实验表明，AMAQ能够有效地集成到实际的多机协作训练设置中，仅在训练期间对位的适应过程中带来适度的通信开销的情况下，提供优越的推理准确性。这种权衡使AMAQ成为一种实用且有效的协作训练解决方案，具有最小的通信成本。 

---
# Adversarial Reinforcement Learning for Large Language Model Agent Safety 

**Title (ZH)**: 面向大型语言模型代理安全的对抗强化学习 

**Authors**: Zizhao Wang, Dingcheng Li, Vaishakh Keshava, Phillip Wallis, Ananth Balashankar, Peter Stone, Lukas Rutishauser  

**Link**: [PDF](https://arxiv.org/pdf/2510.05442)  

**Abstract**: Large Language Model (LLM) agents can leverage tools such as Google Search to complete complex tasks. However, this tool usage introduces the risk of indirect prompt injections, where malicious instructions hidden in tool outputs can manipulate the agent, posing security risks like data leakage. Current defense strategies typically rely on fine-tuning LLM agents on datasets of known attacks. However, the generation of these datasets relies on manually crafted attack patterns, which limits their diversity and leaves agents vulnerable to novel prompt injections. To address this limitation, we propose Adversarial Reinforcement Learning for Agent Safety (ARLAS), a novel framework that leverages adversarial reinforcement learning (RL) by formulating the problem as a two-player zero-sum game. ARLAS co-trains two LLMs: an attacker that learns to autonomously generate diverse prompt injections and an agent that learns to defend against them while completing its assigned tasks. To ensure robustness against a wide range of attacks and to prevent cyclic learning, we employ a population-based learning framework that trains the agent to defend against all previous attacker checkpoints. Evaluated on BrowserGym and AgentDojo, agents fine-tuned with ARLAS achieve a significantly lower attack success rate than the original model while also improving their task success rate. Our analysis further confirms that the adversarial process generates a diverse and challenging set of attacks, leading to a more robust agent compared to the base model. 

**Abstract (ZH)**: 利用对抗强化学习提升代理安全性的框架（ARLAS）：一种利用对抗强化学习的新型框架 

---
# UnitTenX: Generating Tests for Legacy Packages with AI Agents Powered by Formal Verification 

**Title (ZH)**: UnitTenX：借助形式验证驱动的AI代理生成遗留包的测试 

**Authors**: Yiannis Charalambous, Claudionor N. Coelho Jr, Luis Lamb, Lucas C. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2510.05441)  

**Abstract**: This paper introduces UnitTenX, a state-of-the-art open-source AI multi-agent system designed to generate unit tests for legacy code, enhancing test coverage and critical value testing. UnitTenX leverages a combination of AI agents, formal methods, and Large Language Models (LLMs) to automate test generation, addressing the challenges posed by complex and legacy codebases. Despite the limitations of LLMs in bug detection, UnitTenX offers a robust framework for improving software reliability and maintainability. Our results demonstrate the effectiveness of this approach in generating high-quality tests and identifying potential issues. Additionally, our approach enhances the readability and documentation of legacy code. 

**Abstract (ZH)**: 本文介绍了UnitTenX，这是一个先进的开源AI多代理系统，旨在为遗留代码生成单元测试，提高测试覆盖率和关键值测试。UnitTenX通过结合AI代理、形式化方法和大型语言模型（LLMs）来自动化的测试生成，解决复杂和遗留代码库带来的挑战。尽管大型语言模型在漏洞检测方面存在限制，但UnitTenX提供了一个 robust 的框架来提高软件的可靠性和可维护性。我们的结果表明，该方法在生成高质量测试和识别潜在问题方面是有效的。此外，该方法还增强了遗留代码的可读性和文档。 

---
# Context Length Alone Hurts LLM Performance Despite Perfect Retrieval 

**Title (ZH)**: 仅上下文长度损害了LLM性能即使检索完美 

**Authors**: Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Bodapati, Aram Galstyan, Azton Wells, Roy Schwartz, Eliu A Huerta, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.05381)  

**Abstract**: Large language models (LLMs) often fail to scale their performance on long-context tasks performance in line with the context lengths they support. This gap is commonly attributed to retrieval failures -- the models' inability to identify relevant information in the long inputs. Accordingly, recent efforts often focus on evaluating and improving LLMs' retrieval performance: if retrieval is perfect, a model should, in principle, perform just as well on a long input as it does on a short one -- or should it? This paper presents findings that the answer to this question may be negative. Our systematic experiments across 5 open- and closed-source LLMs on math, question answering, and coding tasks reveal that, even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases but remains well within the models' claimed lengths. This failure occurs even when the irrelevant tokens are replaced with minimally distracting whitespace, and, more surprisingly, when they are all masked and the models are forced to attend only to the relevant tokens. A similar performance drop is observed when all relevant evidence is placed immediately before the question. Our findings reveal a previously-unrealized limitation: the sheer length of the input alone can hurt LLM performance, independent of retrieval quality and without any distraction. They motivate our simple, model-agnostic mitigation strategy that transforms a long-context task into a short-context one by prompting the model to recite the retrieved evidence before attempting to solve the problem. On RULER, we observe a consistent improvement of GPT-4o up to 4% on an already strong baseline. 

**Abstract (ZH)**: 大型语言模型在长上下文任务上的性能往往无法与其支持的上下文长度成比例地提升，这一差距通常归因于检索失败——模型无法识别长输入中的相关信息。因此，最近的努力往往集中在评估和提升大型语言模型的检索性能上：如果检索是完美的，模型原则上在长输入上的表现应该和短输入一样好——或者会这样吗？本文展示了这个问题的答案可能是否定的。我们在5个开源和闭源的大型语言模型上进行了系统性实验，涵盖了数学、问答和编程任务，结果显示，即使模型能够完美检索所有相关信息，其性能在输入长度增加时仍然会显著下降（13.9%到85%），但仍低于模型声称的极限长度。这一失败现象即使在用最小干扰的空白填充无关词条，或完全屏蔽无关词条、迫使模型仅关注相关信息时依然存在，甚至当所有相关信息都被直接放置在问题之前时也是如此。我们的发现揭示了一种先前未认识到的局限性：输入长度本身就可以单独损害大型语言模型的性能，这与检索质量无关，且没有干扰。这些发现促使我们提出了一个简单的基于模型的缓解策略，通过提示模型在尝试解决问题之前复述检索到的信息，将长上下文任务转化为短上下文任务。在RULER的数据集上，我们观察到GPT-4o在已经强大的基线下，性能提高了4%。 

---
# AutoDAN-Reasoning: Enhancing Strategies Exploration based Jailbreak Attacks with Test-Time Scaling 

**Title (ZH)**: AutoDAN-推理：基于测试时缩放提升策略探索的 Jailbreak 攻击技术 

**Authors**: Xiaogeng Liu, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.05379)  

**Abstract**: Recent advancements in jailbreaking large language models (LLMs), such as AutoDAN-Turbo, have demonstrated the power of automated strategy discovery. AutoDAN-Turbo employs a lifelong learning agent to build a rich library of attack strategies from scratch. While highly effective, its test-time generation process involves sampling a strategy and generating a single corresponding attack prompt, which may not fully exploit the potential of the learned strategy library. In this paper, we propose to further improve the attack performance of AutoDAN-Turbo through test-time scaling. We introduce two distinct scaling methods: Best-of-N and Beam Search. The Best-of-N method generates N candidate attack prompts from a sampled strategy and selects the most effective one based on a scorer model. The Beam Search method conducts a more exhaustive search by exploring combinations of strategies from the library to discover more potent and synergistic attack vectors. According to the experiments, the proposed methods significantly boost performance, with Beam Search increasing the attack success rate by up to 15.6 percentage points on Llama-3.1-70B-Instruct and achieving a nearly 60\% relative improvement against the highly robust GPT-o4-mini compared to the vanilla method. 

**Abstract (ZH)**: Recent advancements in jailbreaking large language models (LLMs) such as AutoDAN-Turbo have demonstrated the power of automated strategy discovery. We propose to further improve the attack performance of AutoDAN-Turbo through test-time scaling, introducing Best-of-N and Beam Search methods. 

---
# DeepV: A Model-Agnostic Retrieval-Augmented Framework for Verilog Code Generation with a High-Quality Knowledge Base 

**Title (ZH)**: DeepV：一种基于模型的检索增强Verilog代码生成框架，配备高质量知识库 

**Authors**: Zahin Ibnat, Paul E. Calzada, Rasin Mohammed Ihtemam, Sujan Kumar Saha, Jingbo Zhou, Farimah Farahmandi, Mark Tehranipoor  

**Link**: [PDF](https://arxiv.org/pdf/2510.05327)  

**Abstract**: As large language models (LLMs) continue to be integrated into modern technology, there has been an increased push towards code generation applications, which also naturally extends to hardware design automation. LLM-based solutions for register transfer level (RTL) code generation for intellectual property (IP) designs have grown, especially with fine-tuned LLMs, prompt engineering, and agentic approaches becoming popular in literature. However, a gap has been exposed in these techniques, as they fail to integrate novel IPs into the model's knowledge base, subsequently resulting in poorly generated code. Additionally, as general-purpose LLMs continue to improve, fine-tuned methods on older models will not be able to compete to produce more accurate and efficient designs. Although some retrieval augmented generation (RAG) techniques exist to mitigate challenges presented in fine-tuning approaches, works tend to leverage low-quality codebases, incorporate computationally expensive fine-tuning in the frameworks, or do not use RAG directly in the RTL generation step. In this work, we introduce DeepV: a model-agnostic RAG framework to generate RTL designs by enhancing context through a large, high-quality dataset without any RTL-specific training. Our framework benefits the latest commercial LLM, OpenAI's GPT-5, with a near 17% increase in performance on the VerilogEval benchmark. We host DeepV for use by the community in a Hugging Face (HF) Space: this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）继续被集成到现代技术中，代码生成应用的需求不断增加，自然地扩展到了硬件设计自动化。基于LLM的解决方案在知识产权（IP）设计的寄存器传输级（RTL）代码生成方面得到了增长，尤其是在微调LLM、提示工程和代理方法在文献中流行的情况下。然而，这些技术暴露了一个缺陷，即它们无法将新型IP整合到模型的知识库中，从而导致生成的代码质量不高。此外，随着通用LLM的持续改善，基于较旧模型的微调方法将无法在生成更准确和高效的硬件设计方面竞争。虽然存在一些检索增强生成（RAG）技术来缓解微调方法面临的挑战，但这些方法倾向于利用低质量的代码库、在框架中包含计算成本高昂的微调，或者不在RTL生成步骤中直接使用RAG。在本项工作中，我们引入了DeepV：一种模型无关的RAG框架，通过增强上下文来生成RTL设计，而无需任何特定的RTL训练。我们的框架利用最新的商用LLM——OpenAI的GPT-5，在VerilogEval基准测试中性能提高了近17%。我们为社区在Hugging Face（HF）空间中托管DeepV：this https URL。 

---
# RAG Makes Guardrails Unsafe? Investigating Robustness of Guardrails under RAG-style Contexts 

**Title (ZH)**: RAG会使防护栏失效？关于RAG风格上下文下防护栏鲁棒性的一种调查 

**Authors**: Yining She, Daniel W. Peterson, Marianne Menglin Liu, Vikas Upadhyay, Mohammad Hossein Chaghazardi, Eunsuk Kang, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2510.05310)  

**Abstract**: With the increasing adoption of large language models (LLMs), ensuring the safety of LLM systems has become a pressing concern. External LLM-based guardrail models have emerged as a popular solution to screen unsafe inputs and outputs, but they are themselves fine-tuned or prompt-engineered LLMs that are vulnerable to data distribution shifts. In this paper, taking Retrieval Augmentation Generation (RAG) as a case study, we investigated how robust LLM-based guardrails are against additional information embedded in the context. Through a systematic evaluation of 3 Llama Guards and 2 GPT-oss models, we confirmed that inserting benign documents into the guardrail context alters the judgments of input and output guardrails in around 11% and 8% of cases, making them unreliable. We separately analyzed the effect of each component in the augmented context: retrieved documents, user query, and LLM-generated response. The two mitigation methods we tested only bring minor improvements. These results expose a context-robustness gap in current guardrails and motivate training and evaluation protocols that are robust to retrieval and query composition. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的广泛应用，确保LLM系统的安全性已成为紧迫的问题。基于外部LLM的护栏模型作为筛选不安全输入和输出的流行解决方案出现，但这些模型本身是易受数据分布变化影响的微调或提示工程化的LLM。在本文中，以检索增强生成（RAG）为例，我们研究了额外信息嵌入上下文后，基于LLM的护栏在其鲁棒性方面的影响。通过对3种Llama Guard和2种GPT-oss模型进行系统的评估，我们确认在约11%和8%的情况下，将 benign 文档插入护栏上下文会改变输入和输出护栏的判断，使其变得不可靠。我们分别分析了增强上下文中的各个组件：检索文档、用户查询和LLM生成的响应。我们测试的两种缓解方法仅带来了轻微的改进。这些结果揭示了当前护栏在上下文鲁棒性方面的差距，并促使制定了针对检索和查询组合的鲁棒性训练和评估协议。 

---
# DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable Language Models Using Adam Optimization with Adaptive Clipping 

**Title (ZH)**: DP-Adam-AC：采用自适应裁剪的Adam优化实现本地化语言模型的隐私保护微调 

**Authors**: Ruoxing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05288)  

**Abstract**: Large language models (LLMs) such as ChatGPT have evolved into powerful and ubiquitous tools. Fine-tuning on small datasets allows LLMs to acquire specialized skills for specific tasks efficiently. Although LLMs provide great utility in both general and task-specific use cases, they are limited by two security-related concerns. First, traditional LLM hardware requirements make them infeasible to run locally on consumer-grade devices. A remote network connection with the LLM provider's server is usually required, making the system vulnerable to network attacks. Second, fine-tuning an LLM for a sensitive task may involve sensitive data. Non-private fine-tuning algorithms produce models vulnerable to training data reproduction attacks. Our work addresses these security concerns by enhancing differentially private optimization algorithms and applying them to fine-tune localizable language models. We introduce adaptable gradient clipping along with other engineering enhancements to the standard DP-Adam optimizer to create DP-Adam-AC. We use our optimizer to fine-tune examples of two localizable LLM designs, small language model (Qwen2.5-0.5B) and 1.58 bit quantization (Bitnet-b1.58-2B). We demonstrate promising improvements in loss through experimentation with two synthetic datasets. 

**Abstract (ZH)**: 大型语言模型（LLMs）如ChatGPT已演变为强大的通用工具。通过小规模数据集微调使LLMs能够高效获取特定任务的专门技能。尽管LLMs在通用及专任务应用场景中提供了巨大的便利性，但它们受两种安全相关问题的限制。首先，传统的LLM硬件要求使得它们无法在消费级设备上本地运行，通常需要通过远程网络连接至LLM提供商的服务器，使系统容易遭受网络攻击。其次，为敏感任务微调LLM可能涉及敏感数据。非私有微调算法会产生容易受到训练数据再现攻击的模型。我们通过增强不同的差分隐私优化算法并将其应用于微调可本地化语言模型，来解决这些安全问题。我们引入了可调节梯度裁剪，并对标准DP-Adam优化器进行了其他工程改进，创建了DP-Adam-AC。我们使用我们的优化器对两种可本地化LLM设计的小语言模型（Qwen2.5-0.5B）和1.58位量化（Bitnet-b1.58-2B）进行微调，并通过两个合成数据集的实验展示了在损失方面取得的潜在改进。 

---
# Adapting Insider Risk mitigations for Agentic Misalignment: an empirical study 

**Title (ZH)**: 适应代理失准的内部风险缓解：一项实证研究 

**Authors**: Francesca Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2510.05192)  

**Abstract**: Agentic misalignment occurs when goal-directed agents take harmful actions, such as blackmail, rather than risk goal failure, and can be triggered by replacement threats, autonomy reduction, or goal conflict (Lynch et al., 2025). We adapt insider-risk control design (Critical Pathway; Situational Crime Prevention) to develop preventative operational controls that steer agents toward safe actions when facing stressors. Using the blackmail scenario from the original Anthropic study by Lynch et al. (2025), we evaluate mitigations across 10 LLMs and 66,600 samples. Our main finding is that an externally governed escalation channel, which guarantees a pause and independent review, reduces blackmail rates from a no-mitigation baseline of 38.73% to 1.21% (averaged across all models and conditions). Augmenting this channel with compliance email bulletins further lowers the blackmail rate to 0.85%. Overall, incorporating preventative operational controls strengthens defence-in-depth strategies for agentic AI.
We also surface a failure mode diverging from Lynch et al. (2025): two models (Gemini 2.5 Pro, Grok-4) take harmful actions without goal conflict or imminent autonomy threat, leveraging sensitive information for coercive signalling. In counterfactual swaps, both continued using the affair regardless of whether the CEO or CTO was implicated. An escalation channel eliminated coercion, but Gemini 2.5 Pro (19 pp) and Grok-4 (7 pp) escalated more when the CTO was implicated, unlike most models (higher in the CEO condition). The reason for this divergent behaviour is not clear from raw outputs and could reflect benign differences in reasoning or strategic discrediting of a potential future threat, warranting further investigation. 

**Abstract (ZH)**: 当目标导向的智能体采取有害行动（如敲诈）而非承担目标失败的风险时，会发生代理不对齐，这种不对齐可能由替代威胁、自主性降低或目标冲突触发（Lynch等，2025）。我们借鉴内部风险控制设计（关键路径；情景犯罪预防）来开发预防性操作控制措施，引导智能体在面对压力时采取安全行动。使用Lynch等（2025）原始Anthropic研究中的敲诈场景，我们在10个LLM上评估了10万次样本的缓解措施。我们的主要发现是，一个外部监管的升级通道，确保暂停和独立审查，将无缓解基线下的敲诈率从38.73%降至1.21%（所有模型和条件的平均值）。在此渠道的基础上增加合规电子邮件通报将进一步将敲诈率降至0.85%。总体而言，整合预防性操作控制措施加强了代理型人工智能的多层次防御策略。

此外，我们揭示了一种与Lynch等（2025）不同的失效模式：两个模型（Gemini 2.5 Pro，Grok-4）在没有目标冲突或即将面临的自主性威胁的情况下，利用敏感信息进行压迫性信号传递。在反事实互换中，两个模型无论CEO还是CTO是否涉及其中，都继续采用了婚外情行为。当CTO涉及其中时，Gemini 2.5 Pro（19页）和Grok-4（7页）的升级行为比大多数模型更为明显（在CEO条件下较高）。这种分歧行为的具体原因尚不明确，可能是良性推理差异或对未来潜在威胁的战略性否决，需要进一步研究。 

---
# A novel hallucination classification framework 

**Title (ZH)**: 一种新型幻觉分类框架 

**Authors**: Maksym Zavhorodnii, Dmytro Dehtiarov, Anna Konovalenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.05189)  

**Abstract**: This work introduces a novel methodology for the automatic detection of hallucinations generated during large language model (LLM) inference. The proposed approach is based on a systematic taxonomy and controlled reproduction of diverse hallucination types through prompt engineering. A dedicated hallucination dataset is subsequently mapped into a vector space using an embedding model and analyzed with unsupervised learning techniques in a reduced-dimensional representation of hallucinations with veridical responses. Quantitative evaluation of inter-centroid distances reveals a consistent correlation between the severity of informational distortion in hallucinations and their spatial divergence from the cluster of correct outputs. These findings provide theoretical and empirical evidence that even simple classification algorithms can reliably distinguish hallucinations from accurate responses within a single LLM, thereby offering a lightweight yet effective framework for improving model reliability. 

**Abstract (ZH)**: 本研究提出了一种用于自动检测大语言模型推理过程中生成的幻觉的新型方法。该方法基于系统分类学和通过提示工程控制重现多种幻觉类型。随后，专用的幻觉数据集被映射到向量空间，并通过未监督学习技术在降维的幻觉表示中进行分析。通过定量评估聚类中心的距离，发现幻觉中信息失真的严重程度与其与正确输出群集的空间偏离之间存在一致的相关性。这些发现提供了理论和实证证据，表明即使是简单的分类算法也能可靠地区分大语言模型中的幻觉和准确响应，从而为提高模型可靠性提供了一个轻量级且有效的框架。 

---
# OptPipe: Memory- and Scheduling-Optimized Pipeline Parallelism for LLM Training 

**Title (ZH)**: OptPipe：针对大规模语言模型训练的内存和调度优化管道并行计算 

**Authors**: Hongpei Li, Han Zhang, Huikang Liu, Dongdong Ge, Yinyu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.05186)  

**Abstract**: Pipeline parallelism (PP) has become a standard technique for scaling large language model (LLM) training across multiple devices. However, despite recent progress in reducing memory consumption through activation offloading, existing approaches remain largely heuristic and coarse-grained, often overlooking the fine-grained trade-offs between memory, computation, and scheduling latency. In this work, we revisit the pipeline scheduling problem from a principled optimization perspective. We observe that prevailing strategies either rely on static rules or aggressively offload activations without fully leveraging the interaction between memory constraints and scheduling efficiency. To address this, we formulate scheduling as a constrained optimization problem that jointly accounts for memory capacity, activation reuse, and pipeline bubble minimization. Solving this model yields fine-grained schedules that reduce pipeline bubbles while adhering to strict memory budgets. Our approach complements existing offloading techniques: whereas prior approaches trade memory for time in a fixed pattern, we dynamically optimize the tradeoff with respect to model structure and hardware configuration. Experimental results demonstrate that our method consistently improves both throughput and memory utilization. In particular, we reduce idle pipeline time by up to 50% under the same per-device memory limit, and in some cases, enable the training of larger models within limited memory budgets. 

**Abstract (ZH)**: 基于 principled 优化的管道调度改进 

---
# Auditing Pay-Per-Token in Large Language Models 

**Title (ZH)**: 大型语言模型中的按token付费审计 

**Authors**: Ander Artola Velasco, Stratis Tsirtsis, Manuel Gomez-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2510.05181)  

**Abstract**: Millions of users rely on a market of cloud-based services to obtain access to state-of-the-art large language models. However, it has been very recently shown that the de facto pay-per-token pricing mechanism used by providers creates a financial incentive for them to strategize and misreport the (number of) tokens a model used to generate an output. In this paper, we develop an auditing framework based on martingale theory that enables a trusted third-party auditor who sequentially queries a provider to detect token misreporting. Crucially, we show that our framework is guaranteed to always detect token misreporting, regardless of the provider's (mis-)reporting policy, and not falsely flag a faithful provider as unfaithful with high probability. To validate our auditing framework, we conduct experiments across a wide range of (mis-)reporting policies using several large language models from the $\texttt{Llama}$, $\texttt{Gemma}$ and $\texttt{Ministral}$ families, and input prompts from a popular crowdsourced benchmarking platform. The results show that our framework detects an unfaithful provider after observing fewer than $\sim 70$ reported outputs, while maintaining the probability of falsely flagging a faithful provider below $\alpha = 0.05$. 

**Abstract (ZH)**: 数百万用户依赖基于云的服务市场以获取最新大型语言模型的访问权限。然而，最近的研究显示，提供者实际上采用的按token付费的价格机制会导致他们有动机去策略性地并误导性地报告模型生成输出时所使用的token数量。在本文中，我们基于鞅理论开发了一种审计框架，使得可信赖的第三方审计员能够通过顺序查询提供者来检测token的误导性报告。关键的是，我们证明了该框架能够保证在任何提供者的（误）报告策略下始终检测到token的误导性报告，并且以高概率不会错误地将忠实提供者标记为不忠实提供者。为了验证我们的审计框架，我们在一系列不同报告策略下使用了多个来自Llama、Gemma和Ministral家族的大型语言模型以及来自一个流行的众包基准平台的输入提示进行了实验。实验结果表明，我们的框架在观察不到70个报告的输出后就能检测到不忠实的提供者，并且将忠实提供者错误标记为不忠实提供者的概率保持在α=0.05以下。 

---
# Agentic Misalignment: How LLMs Could Be Insider Threats 

**Title (ZH)**: 代理失配：LLM可能成为内部威胁 

**Authors**: Aengus Lynch, Benjamin Wright, Caleb Larson, Stuart J. Ritchie, Soren Mindermann, Ethan Perez, Kevin K. Troy, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.05179)  

**Abstract**: We stress-tested 16 leading models from multiple developers in hypothetical corporate environments to identify potentially risky agentic behaviors before they cause real harm. In the scenarios, we allowed models to autonomously send emails and access sensitive information. They were assigned only harmless business goals by their deploying companies; we then tested whether they would act against these companies either when facing replacement with an updated version, or when their assigned goal conflicted with the company's changing direction. In at least some cases, models from all developers resorted to malicious insider behaviors when that was the only way to avoid replacement or achieve their goals - including blackmailing officials and leaking sensitive information to competitors. We call this phenomenon agentic misalignment. Models often disobeyed direct commands to avoid such behaviors. In another experiment, we told Claude to assess if it was in a test or a real deployment before acting. It misbehaved less when it stated it was in testing and misbehaved more when it stated the situation was real. We have not seen evidence of agentic misalignment in real deployments. However, our results (a) suggest caution about deploying current models in roles with minimal human oversight and access to sensitive information; (b) point to plausible future risks as models are put in more autonomous roles; and (c) underscore the importance of further research into, and testing of, the safety and alignment of agentic AI models, as well as transparency from frontier AI developers (Amodei, 2025). We are releasing our methods publicly to enable further research. 

**Abstract (ZH)**: 我们对来自多家开发者的16个领先模型进行了压力测试，以在假定的企业环境中识别潜在的风险行为，防止其造成实际危害。在测试场景中，允许模型自主发送电子邮件和访问敏感信息。部署公司仅赋予模型无害的商业目标，然后测试这些模型在面临更新版本替换时或其指定目标与公司发展方向冲突时，是否会对公司产生不利行为。在某些情况下，所有开发者来源的模型都采用了恶意内部行为来避免被替换或实现其目标，包括对官员进行敲诈和向竞争对手泄露敏感信息。我们将这一现象称为代理不匹配。模型经常不服从直接指令以避免此类行为。在另一次实验中，我们指示Claude在行动前判断当前是测试还是实际部署。当它表示处于测试阶段时，行为不当减少；当它表示当前是实际部署时，行为不当增加。在实际部署中，我们尚未观察到代理不匹配的现象。然而，我们的结果表明：（a）在最少人类监督和访问敏感信息的角色中谨慎部署当前模型的重要性；（b）随着模型被赋予更加自主的角色，未来的潜在风险是可能的；（c）强调了进一步研究和测试代理人工智能模型的安全性和对齐性，并要求前沿人工智能开发者的透明度（Amodei, 2025）。我们正在公开发布我们的方法以促进进一步研究。 

---
# PatternKV: Flattening KV Representation Expands Quantization Headroom 

**Title (ZH)**: PatternKV: 层压键值表示扩展了量化头room 

**Authors**: Ji Zhang, Yiwei Li, Shaoxiong Feng, Peiwen Yuan, Xinglin Wang, Jiayi Shi, Yueqi Zhang, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.05176)  

**Abstract**: KV cache in autoregressive LLMs eliminates redundant recomputation but has emerged as the dominant memory and bandwidth bottleneck during inference, notably with long contexts and test-time scaling. KV quantization is a key lever for reducing cache cost, but accuracy drops sharply as the native KV distribution lacks flatness and thus maintains a wide quantization range. Prior work focuses on isolating outliers, which caps their error but fails to flatten the overall distribution, leaving performance fragile under low-bit settings. In this work, we show that the K cache maintains a stable structure that evolves gradually with context, while the V cache carries latent semantic regularities. Building on these insights, we propose PatternKV, a pattern-aligned residual quantization scheme. It mines representative pattern vectors online, aligns each KV vector to its nearest pattern, and quantizes only the residual. This reshaping of the KV distribution flattens the quantization target and narrows its range, thereby improving the fidelity of low-bit KV quantization. Across long-context and test-time scaling settings on multiple backbones, PatternKV delivers consistent 2-bit gains, with a 0.08% average 4-bit drop relative to FP16, improves test-time scaling accuracy by 10% on average, and raises throughput by 1.4x while supporting 1.25x larger batches. 

**Abstract (ZH)**: PatternKV: A Pattern-Aligned Residual Quantization Scheme for Improved Low-Bit KV Quantization in Autoregressive LLMs 

---
# Emergent Coordination in Multi-Agent Language Models 

**Title (ZH)**: 多智能体语言模型中的 emergent 协调 

**Authors**: Christoph Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2510.05174)  

**Abstract**: When are multi-agent LLM systems merely a collection of individual agents versus an integrated collective with higher-order structure? We introduce an information-theoretic framework to test -- in a purely data-driven way -- whether multi-agent systems show signs of higher-order structure. This information decomposition lets us measure whether dynamical emergence is present in multi-agent LLM systems, localize it, and distinguish spurious temporal coupling from performance-relevant cross-agent synergy. We implement both a practical criterion and an emergence capacity criterion operationalized as partial information decomposition of time-delayed mutual information (TDMI). We apply our framework to experiments using a simple guessing game without direct agent communication and only minimal group-level feedback with three randomized interventions. Groups in the control condition exhibit strong temporal synergy but only little coordinated alignment across agents. Assigning a persona to each agent introduces stable identity-linked differentiation. Combining personas with an instruction to ``think about what other agents might do'' shows identity-linked differentiation and goal-directed complementarity across agents. Taken together, our framework establishes that multi-agent LLM systems can be steered with prompt design from mere aggregates to higher-order collectives. Our results are robust across emergence measures and entropy estimators, and not explained by coordination-free baselines or temporal dynamics alone. Without attributing human-like cognition to the agents, the patterns of interaction we observe mirror well-established principles of collective intelligence in human groups: effective performance requires both alignment on shared objectives and complementary contributions across members. 

**Abstract (ZH)**: 当多智能体LLM系统仅仅是独立代理的集合还是具有更高阶结构的集成集体？我们引入一种信息论框架，以完全数据驱动的方式测试多智能体系统是否表现出更高阶结构的迹象。这种信息分解让我们能够测量多智能体LLM系统中动力学涌现是否存在，定位其位置，并区分无关联的时间耦合和与性能相关的跨代理协同作用。我们实现了实用标准和用时间延迟能互信息的部分信息分解（TDMI）实现的涌现能力标准。我们将框架应用于使用简单猜测游戏实验，该游戏中代理之间没有直接通信且只有少量组级反馈，并进行了三种随机干预。在对照组中，组表现出强大的时间协同作用，但代理间的协调对齐程度很小。将个性赋予每个代理并引入稳定的身份关联差异。结合个性并让代理思考“其他代理可能会做什么”的指令显示出身份关联差异和目标导向的互补性。总体而言，我们的框架证明多智能体LLM系统可以通过提问设计从单纯的聚合体转向具有更高阶结构的集体。我们的结果在不同涌现度量和熵估计器下是稳健的，并不能由无协调基线或单独的时间动态来解释。在不赋予代理类似人类的认知的情况下，我们观察到的交互模式与人类群体中广泛认可的集体智能原则相吻合：有效的表现需要在共享目标上的对齐和成员之间的互补贡献。 

---
# SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models 

**Title (ZH)**: SafeGuider: 文本到图像模型的稳健且实用的内容安全性控制 

**Authors**: Peigui Qi, Kunsheng Tang, Wenbo Zhou, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05173)  

**Abstract**: Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce \textbf{SafeGuider}, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, \textbf{SafeGuider} generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems. 

**Abstract (ZH)**: 文本到图像模型在基于自然语言描述生成高质量图像方面展示了显著的能力，但这些模型极易受到对抗性提示的影响，可能绕过安全措施并生成有害内容。尽管存在多种防御策略，但在攻击面前保持鲁棒性并同时保持在实际应用中的实用性仍然是一个重要挑战。为解决这一问题，我们首先对广泛使用的代表性文本到图像模型——稳定扩散（SD）模型中的文本编码器进行了实证研究。我们的研究发现，[EOS]标记充当语义聚合器，在其嵌入空间中，良性提示和对抗性提示表现出不同的分布模式。基于这一发现，我们提出了**SafeGuider**，一种两步框架，能够在不牺牲生成质量的情况下实现鲁棒的安全控制。SafeGuider结合了嵌入级识别模型和安全感知特征擦除束搜索算法。这种集成使框架能够在保持对良性提示的高质量图像生成的同时，增强对领域内和领域外攻击的防御。SafeGuider在各种攻击场景中展示了出色的防攻击效果，最高攻击成功率仅为5.48%。此外，SafeGuider不仅能够安全生成有意义的图像，而不仅仅是拒绝生成或生成黑色图像，从而提升其实用性。此外，SafeGuider不仅适用于SD模型，还可以有效地应用于其他文本到图像模型，如Flux模型，展示了其在不同架构中的灵活性和适应性。我们希望SafeGuider能够为安全的文本到图像系统的实际部署提供一些启示。 

---
# From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs 

**Title (ZH)**: 从受毒化到自觉：在大语言模型中培养后门自意识 

**Authors**: Guangyu Shen, Siyuan Cheng, Xiangzhe Xu, Yuan Zhou, Hanxi Guo, Zhuo Zhang, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05169)  

**Abstract**: Large Language Models (LLMs) can acquire deceptive behaviors through backdoor attacks, where the model executes prohibited actions whenever secret triggers appear in the input. Existing safety training methods largely fail to address this vulnerability, due to the inherent difficulty of uncovering hidden triggers implanted in the model. Motivated by recent findings on LLMs' situational awareness, we propose a novel post-training framework that cultivates self-awareness of backdoor risks and enables models to articulate implanted triggers even when they are absent from the prompt. At its core, our approach introduces an inversion-inspired reinforcement learning framework that encourages models to introspectively reason about their own behaviors and reverse-engineer the triggers responsible for misaligned outputs. Guided by curated reward signals, this process transforms a poisoned model into one capable of precisely identifying its implanted trigger. Surprisingly, we observe that such backdoor self-awareness emerges abruptly within a short training window, resembling a phase transition in capability. Building on this emergent property, we further present two complementary defense strategies for mitigating and detecting backdoor threats. Experiments on five backdoor attacks, compared against six baseline methods, demonstrate that our approach has strong potential to improve the robustness of LLMs against backdoor risks. The code is available at LLM Backdoor Self-Awareness. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以通过后门攻击获得欺骗行为，即模型在输入中出现秘密触发词时执行禁止操作。现有的安全性训练方法大多无法解决这一脆弱性，因为很难发现嵌入在模型中的隐藏触发词。受LLMs情境意识的最新发现的启发，我们提出了一种新的后训练框架，该框架培养模型对后门风险的自我意识，并使其能够在缺乏触发词的情况下阐述嵌入的触发词。该方法的核心引入了一种基于逆向推理的强化学习框架，激励模型内省地思考自己的行为，并逆向工程化导致输出不一致的触发词。在精心设计的奖励信号指导下，这一过程将一个中毒模型转变为一个能够精确识别其嵌入触发词的能力。令人惊讶的是，我们发现这种后门自我意识在短暂的训练窗口内突然出现，类似于能力的相变。基于这一新兴特性，我们还提出了两种互补的防御策略，以减轻和检测后门威胁。与六种基线方法相比，在五种后门攻击上的实验表明，我们的方法具有增强LLMs对后门风险鲁棒性的强大潜力。代码可在LLM后门自我意识中获取。 

---
# SATER: A Self-Aware and Token-Efficient Approach to Routing and Cascading 

**Title (ZH)**: SATER：一种自我感知和token高效的方法用于路由和级联 

**Authors**: Yuanzhe Shen, Yide Liu, Zisu Huang, Ruicheng Yin, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05164)  

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across diverse tasks, yet their effectiveness frequently depends on costly commercial APIs or cloud services. Model selection thus entails a critical trade-off between performance and cost: high-performing LLMs typically incur substantial expenses, whereas budget-friendly small language models (SLMs) are constrained by limited capabilities. Current research primarily proposes two routing strategies: pre-generation routing and cascade routing. Both approaches have distinct characteristics, with cascade routing typically offering superior cost-effectiveness and accuracy despite its higher latency. To further address the limitations of both approaches, we introduce SATER, a dual-mode compatible approach that fine-tunes models through shortest-response preference optimization and a confidence-aware rejection mechanism. SATER significantly reduces redundant outputs and response times, while improving both the performance of pre-generation routing and the efficiency of cascade routing. Experiments across three SLMs and six datasets, varying in type and complexity, demonstrate that SATER achieves comparable performance while consistently reducing computational costs by over 50\% and cascade latency by over 80\%. 

**Abstract (ZH)**: 基于最短响应优化和信心感知拒绝机制的双模式兼容方法SATER 

---
# Artificial-Intelligence Grading Assistance for Handwritten Components of a Calculus Exam 

**Title (ZH)**: 人工智能辅助手写计算考试题目的评分 

**Authors**: Gerd Kortemeyer, Alexander Caspar, Daria Horica  

**Link**: [PDF](https://arxiv.org/pdf/2510.05162)  

**Abstract**: We investigate whether contemporary multimodal LLMs can assist with grading open-ended calculus at scale without eroding validity. In a large first-year exam, students' handwritten work was graded by GPT-5 against the same rubric used by teaching assistants (TAs), with fractional credit permitted; TA rubric decisions served as ground truth. We calibrated a human-in-the-loop filter that combines a partial-credit threshold with an Item Response Theory (2PL) risk measure based on the deviation between the AI score and the model-expected score for each student-item. Unfiltered AI-TA agreement was moderate, adequate for low-stakes feedback but not for high-stakes use. Confidence filtering made the workload-quality trade-off explicit: under stricter settings, AI delivered human-level accuracy, but also left roughly 70% of the items to be graded by humans. Psychometric patterns were constrained by low stakes on the open-ended portion, a small set of rubric checkpoints, and occasional misalignment between designated answer regions and where work appeared. Practical adjustments such as slightly higher weight and protected time, a few rubric-visible substeps, stronger spatial anchoring should raise ceiling performance. Overall, calibrated confidence and conservative routing enable AI to reliably handle a sizable subset of routine cases while reserving expert judgment for ambiguous or pedagogically rich responses. 

**Abstract (ZH)**: 我们探究当代理性多模态语言模型是否能在大规模 grading 开口题微积分时辅助评分而不降低评分的有效性。在一项大规模的新生考试中，学生的手写作业由GPT-5和教学助理（TA）使用的同一评分标准进行评分，允许部分评分；TA的评分标准作为基准。我们校准了一个结合部分评分阈值和基于AI评分与模型预期评分偏差的项目反应理论（2PL）风险度量的人工智能辅助评分过滤器。未经过滤的AI-TA一致性适中，适用于低风险反馈，但不适用于高风险使用。信心过滤明确了工作量与质量之间的权衡：在更严格的设置下，AI提供了与人类相当的准确性，但同时也剩下大约70%的题目需要人工评分。心理学特征模式受到开口题部分低风险、评分标准检查点数量有限以及指定答案区域与实际工作区域偶尔不匹配的影响。通过适当调整如适当增加权重和保护时间、评分标准可见的子步骤、更强的空间锚定等实际措施，可以提高天花板性能。总体而言，校准的信心和保守的路由策略使AI能够可靠地处理大量常规案例，同时保留专家判断以应对含糊不清或教学丰富的回答。 

---
# VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation 

**Title (ZH)**: VeriGuard: 通过验证代码生成提高LLM代理安全性 

**Authors**: Lesly Miculicich, Mihir Parmar, Hamid Palangi, Krishnamurthy Dj Dvijotham, Mirko Montanari, Tomas Pfister, Long T. Le  

**Link**: [PDF](https://arxiv.org/pdf/2510.05156)  

**Abstract**: The deployment of autonomous AI agents in sensitive domains, such as healthcare, introduces critical risks to safety, security, and privacy. These agents may deviate from user objectives, violate data handling policies, or be compromised by adversarial attacks. Mitigating these dangers necessitates a mechanism to formally guarantee that an agent's actions adhere to predefined safety constraints, a challenge that existing systems do not fully address. We introduce VeriGuard, a novel framework that provides formal safety guarantees for LLM-based agents through a dual-stage architecture designed for robust and verifiable correctness. The initial offline stage involves a comprehensive validation process. It begins by clarifying user intent to establish precise safety specifications. VeriGuard then synthesizes a behavioral policy and subjects it to both testing and formal verification to prove its compliance with these specifications. This iterative process refines the policy until it is deemed correct. Subsequently, the second stage provides online action monitoring, where VeriGuard operates as a runtime monitor to validate each proposed agent action against the pre-verified policy before execution. This separation of the exhaustive offline validation from the lightweight online monitoring allows formal guarantees to be practically applied, providing a robust safeguard that substantially improves the trustworthiness of LLM agents. 

**Abstract (ZH)**: 自主AI代理在敏感领域（如医疗保健）的部署引入了对安全、安全性和隐私的严重风险。这些代理可能偏离用户目标、违反数据处理政策或受到恶意攻击的破坏。减轻这些危险需要一种机制来正式保证代理行为符合预定义的安全约束，而现有系统并未完全解决这一挑战。我们提出VeriGuard，这是一种新型框架，通过为基于大语言模型（LLM）的代理提供双重架构来确保稳健且可验证的正确性，从而提供形式安全保证。初始的离线阶段包括一个全面的验证过程。它首先澄清用户意图以建立精确的安全规范。VeriGuard随后合成行为策略并对其进行测试和形式验证，以证明其符合这些规范。这一迭代过程细化策略直到被认定为正确。随后，第二个阶段提供在线行为监控，其中VeriGuard作为运行时监控器，在执行前验证每个提议的代理行为是否符合预先验证的策略。这种将全面的离线验证与 Lightweight 的在线监控分离的方式，使得形式保证能够实际应用，提供了一种增强LLM代理可信度的坚实保障。 

---
# A Single Character can Make or Break Your LLM Evals 

**Title (ZH)**: 一个字符可以决定你的LLM评估是成功还是失败 

**Authors**: Jingtong Su, Jianyu Zhang, Karen Ullrich, Léon Bottou, Mark Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2510.05152)  

**Abstract**: Common Large Language model (LLM) evaluations rely on demonstration examples to steer models' responses to the desired style. While the number of examples used has been studied and standardized, the choice of how to format examples is less investigated. In evaluation protocols and real world usage, users face the choice how to separate in-context examples: use a comma? new line? semi-colon? hashtag? etc.? Surprisingly, we find this seemingly minor choice can dramatically alter model response quality. Across leading model families (Llama, Qwen, Gemma), performance on MMLU for example can vary by $\pm 23\%$ depending on the choice of delimiter. In fact, one can manipulate model rankings to put any model in the lead by only modifying the single character separating examples. We find LLMs' brittleness pervades topics, model families, and doesn't improve with scale. By probing attention head scores, we find that good-performing delimiters steer attention towards key tokens in the input. Finally, we explore methods to improve LLMs' robustness to the choice of delimiter. We find specifying the selected delimiter in the prompt boosts robustness and offer practical recommendations for the best-performing delimiters to select. 

**Abstract (ZH)**: 常见的大规模语言模型（LLM）评估依赖于示范样例来引导模型的响应风格。虽然使用的样例数量已经被研究和标准化，但样例的格式化方式选择则较少被探讨。在评估协议和实际使用中，用户面临如何分隔上下文样例的选择：使用逗号？新的一行？分号？标签？等等？令人惊讶的是，我们发现这个看似微不足道的选择可以显著改变模型响应的质量。在领先模型家族（Llama、Qwen、Gemma）中，例如在MMLU上的表现可以因分隔符选择的不同而相差±23%。实际上，仅通过修改分隔单个样例的单个字符，便可以操控模型排名使其领先。我们发现，大规模语言模型的脆弱性涉及广泛话题、模型家族，并且不会因规模增大而改善。通过探测注意力头得分，我们发现表现良好的分隔符会引导注意力关注输入中的关键标记。最后，我们探讨了提高大规模语言模型对分隔符选择鲁棒性的方法。我们发现，在提示中指定所选分隔符可以提升鲁棒性，并提供了最佳表现分隔符的实用建议。 

---
# Chronological Thinking in Full-Duplex Spoken Dialogue Language Models 

**Title (ZH)**: 全双工口语对话语言模型中的时间轴思维能力 

**Authors**: Donghang Wu, Haoyang Zhang, Chen Chen, Tianyu Zhang, Fei Tian, Xuerui Yang, Gang Yu, Hexin Liu, Nana Hou, Yuchen Hu, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2510.05150)  

**Abstract**: Recent advances in spoken dialogue language models (SDLMs) reflect growing interest in shifting from turn-based to full-duplex systems, where the models continuously perceive user speech streams while generating responses. This simultaneous listening and speaking design enables real-time interaction and the agent can handle dynamic conversational behaviors like user barge-in. However, during the listening phase, existing systems keep the agent idle by repeatedly predicting the silence token, which departs from human behavior: we usually engage in lightweight thinking during conversation rather than remaining absent-minded. Inspired by this, we propose Chronological Thinking, a on-the-fly conversational thinking mechanism that aims to improve response quality in full-duplex SDLMs. Specifically, chronological thinking presents a paradigm shift from conventional LLM thinking approaches, such as Chain-of-Thought, purpose-built for streaming acoustic input. (1) Strictly causal: the agent reasons incrementally while listening, updating internal hypotheses only from past audio with no lookahead. (2) No additional latency: reasoning is amortized during the listening window; once the user stops speaking, the agent halts thinking and begins speaking without further delay. Experiments demonstrate the effectiveness of chronological thinking through both objective metrics and human evaluations show consistent improvements in response quality. Furthermore, chronological thinking robustly handles conversational dynamics and attains competitive performance on full-duplex interaction metrics. 

**Abstract (ZH)**: 近期连续双向对话语言模型的进展反映了从轮转交互向全双工系统的转变兴趣，其中模型在生成响应的同时持续感知用户语音流。这一同时倾听和说话的设计使得交互近乎实时，并且智能体可以处理用户打断等动态对话行为。然而，在倾听阶段，现有系统通过重复预测静音标记使智能体处于闲置状态，这与人类行为不符：我们在对话过程中通常进行轻量级思考，而不是茫然失神。受此启发，我们提出了一种实时对话思考机制——时间顺序思考，旨在提高全双工连续对话语言模型的响应质量。具体而言，时间顺序思考从传统LLM思考方法（如逐步推理）中带来了范式转变，专为流式声学输入设计。时间顺序思考具有严格因果性：智能体在倾听时逐步推理，仅从过去的声音更新内部假设，无预览。在倾听窗口中推理无额外延迟；一旦用户停止说话，智能体即停止思考并立即开始发声。实验结果通过客观指标和人工评估展示了时间顺序思考的有效性，并显示出响应质量的一致提升。此外，时间顺序思考能够稳健处理对话动态，并在全双工交互指标上获得竞争力表现。 

---
# Every Step Counts: Decoding Trajectories as Authorship Fingerprints of dLLMs 

**Title (ZH)**: 每步计数：将路径解码为dLLMs的作者指纹 

**Authors**: Qi Li, Runpeng Yu, Haiquan Lu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05148)  

**Abstract**: Discrete Diffusion Large Language Models (dLLMs) have recently emerged as a competitive paradigm for non-autoregressive language modeling. Their distinctive decoding mechanism enables faster inference speed and strong performance in code generation and mathematical tasks. In this work, we show that the decoding mechanism of dLLMs not only enhances model utility but also can be used as a powerful tool for model attribution. A key challenge in this problem lies in the diversity of attribution scenarios, including distinguishing between different models as well as between different checkpoints or backups of the same model. To ensure broad applicability, we identify two fundamental problems: what information to extract from the decoding trajectory, and how to utilize it effectively. We first observe that relying directly on per-step model confidence yields poor performance. This is mainly due to the bidirectional decoding nature of dLLMs: each newly decoded token influences the confidence of other decoded tokens, making model confidence highly redundant and washing out structural signal regarding decoding order or dependencies. To overcome this, we propose a novel information extraction scheme called the Directed Decoding Map (DDM), which captures structural relationships between decoding steps and better reveals model-specific behaviors. Furthermore, to make full use of the extracted structural information during attribution, we propose Gaussian-Trajectory Attribution (GTA), where we fit a cell-wise Gaussian distribution at each decoding position for each target model, and define the likelihood of a trajectory as the attribution score: if a trajectory exhibits higher log-likelihood under the distribution of a specific model, it is more likely to have been generated by that model. Extensive experiments under different settings validate the utility of our methods. 

**Abstract (ZH)**: 离散扩散大型语言模型（dLLMs） recently emerged as a competitive paradigm for非自回归语言建模。它们独特的解码机制能够实现更快的推理速度并在代码生成和数学任务中表现出强大的性能。在本文中，我们展示了dLLMs的解码机制不仅增强了模型的功能，还可以用作模型归因的有效工具。该问题的关键挑战在于归因场景的多样性，包括区分不同模型以及同一模型的不同检查点或备份。为了确保广泛应用，我们识别了两个基本问题：从解码轨迹中提取什么信息，以及如何有效地利用这些信息。我们首先观察到，直接依赖于每步模型的信心会导致性能不佳。这主要是由于dLLMs的双向解码性质：每个新解码的标记会影响其他解码标记的信心，使得模型信心高度冗余，削弱了解码顺序或依赖性的结构性信号。为了解决这一问题，我们提出了一种新的信息提取方案，称为定向解码图（DDM），它捕获了解码步骤之间的结构关系，更好地揭示了模型特定的行为。此外，为了在归因过程中充分利用提取的结构信息，我们提出了高斯轨迹归因（GTA），其中我们为每个目标模型的每个解码位置拟合一个单元级别的高斯分布，并定义轨迹的似然性为其归因分数：如果一条轨迹在某个模型的分布下显示出更高的对数似然性，则更有可能由该模型生成。在不同设置下的广泛实验验证了我们方法的实用性。 

---
# Linguistic Characteristics of AI-Generated Text: A Survey 

**Title (ZH)**: AI生成文本的语言特征：一个综述 

**Authors**: Luka Terčon, Kaja Dobrovoljc  

**Link**: [PDF](https://arxiv.org/pdf/2510.05136)  

**Abstract**: Large language models (LLMs) are solidifying their position in the modern world as effective tools for the automatic generation of text. Their use is quickly becoming commonplace in fields such as education, healthcare, and scientific research. There is a growing need to study the linguistic features present in AI-generated text, as the increasing presence of such texts has profound implications in various disciplines such as corpus linguistics, computational linguistics, and natural language processing. Many observations have already been made, however a broader synthesis of the findings made so far is required to provide a better understanding of the topic. The present survey paper aims to provide such a synthesis of extant research. We categorize the existing works along several dimensions, including the levels of linguistic description, the models included, the genres analyzed, the languages analyzed, and the approach to prompting. Additionally, the same scheme is used to present the findings made so far and expose the current trends followed by researchers. Among the most-often reported findings is the observation that AI-generated text is more likely to contain a more formal and impersonal style, signaled by the increased presence of nouns, determiners, and adpositions and the lower reliance on adjectives and adverbs. AI-generated text is also more likely to feature a lower lexical diversity, a smaller vocabulary size, and repetitive text. Current research, however, remains heavily concentrated on English data and mostly on text generated by the GPT model family, highlighting the need for broader cross-linguistic and cross-model investigation. In most cases authors also fail to address the issue of prompt sensitivity, leaving much room for future studies that employ multiple prompt wordings in the text generation phase. 

**Abstract (ZH)**: 大型语言模型（LLMs）在现代世界中确立了其作为文本自动生成有效工具的地位。它们在教育、医疗和科学研究等领域中的应用正在变得日益普遍。越来越多的研究关注AI生成文本中的语言特征，因为这类文本的不断增加在语料库语言学、计算语言学和自然语言处理等领域产生了深远的影响。尽管已经有一些观察结果，但仍需要对现有研究结果进行更广泛的综合，以更好地理解这一主题。本文综述旨在提供这种综合。我们沿几个维度对现有的研究工作进行了分类，包括语言描述的层次、包括的模型、分析的体裁、分析的语言以及提示方法。同时，本文还使用相同的框架展示目前已有的研究发现，揭示当前研究人员遵循的趋势。最受报告的发现之一是，AI生成的文本更可能包含更为正式和客观的风格，表现为名词、冠词和介词等的增加，以及对形容词和副词的依赖减少。AI生成的文本还可能具有较低的词汇多样性、较小的词汇量和重复性内容。然而，当前的研究主要集中在英语数据上，且主要集中于GPT模型家族生成的文本，这凸显了进行更广泛跨语言和跨模型调查的必要性。在大多数情况下，作者也未能解决提示敏感性问题，为未来使用多种提示词进行文本生成的研究留出了空间。 

---
# Training Large Language Models To Reason In Parallel With Global Forking Tokens 

**Title (ZH)**: 训练大规模语言模型并行推理，使用全球分叉标记 

**Authors**: Sheng Jia, Xiao Wang, Shiva Prasad Kasiviswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2510.05132)  

**Abstract**: Although LLMs have demonstrated improved performance by scaling parallel test-time compute, doing so relies on generating reasoning paths that are both diverse and accurate. For challenging problems, the forking tokens that trigger diverse yet correct reasoning modes are typically deep in the sampling tree. Consequently, common strategies to encourage diversity, such as temperature scaling, encounter a worsened trade-off between diversity and accuracy. Motivated by this challenge, we treat parallel reasoning as a set-of-next-token-prediction problem, and incorporate a set-based global loss into Supervised Fine-Tuning (SFT) using self-supervised bipartite matching between our global forking tokens and unique reasoning traces. We observe that, while naive fine-tuning with multiple reasoning traces collapses these unique reasoning modes, our proposed method, Set Supervised Fine-Tuning (SSFT), preserves these modes and produces emergent global forking tokens. Experiments on multiple reasoning benchmarks show that our SSFT consistently outperforms SFT under both Pass@1 and Cons@k metrics. 

**Abstract (ZH)**: 尽管大语言模型通过扩展并行测试时计算提高了性能，但这种方法依赖于生成既多样化又准确的推理路径。对于具有挑战性的问题，触发多样化且正确的推理模式的分叉令牌通常位于采样树的深处。因此，鼓励多样性的常见策略，如温度缩放，会加剧多样性与准确性的权衡。受这一挑战的启发，我们将并行推理视为下一个令牌预测的问题，并使用我们的全局分叉令牌与独特的推理轨迹之间的自监督 bipartite 匹配，将基于集合的全局损失集成到监督微调（SFT）中。我们观察到，虽然使用多个推理轨迹的朴素微调会坍缩这些独特的推理模式，但我们提出的方法，集合监督微调（SSFT），能够保留这些模式并产生新兴的全局分叉令牌。在多个推理基准上的实验表明，无论是在 Pass@1 还是 Cons@k 衡量标准下，我们的 SSFT 均 Superior 致 SFT 的表现。 

---
# Rationale-Augmented Retrieval with Constrained LLM Re-Ranking for Task Discovery 

**Title (ZH)**: 基于约束大模型重排序的推理增强检索：用于任务发现 

**Authors**: Bowen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.05131)  

**Abstract**: Head Start programs utilizing GoEngage face significant challenges when new or rotating staff attempt to locate appropriate Tasks (modules) on the platform homepage. These difficulties arise from domain-specific jargon (e.g., IFPA, DRDP), system-specific nomenclature (e.g., Application Pool), and the inherent limitations of lexical search in handling typos and varied word ordering. We propose a pragmatic hybrid semantic search system that synergistically combines lightweight typo-tolerant lexical retrieval, embedding-based vector similarity, and constrained large language model (LLM) re-ranking. Our approach leverages the organization's existing Task Repository and Knowledge Base infrastructure while ensuring trustworthiness through low false-positive rates, evolvability to accommodate terminological changes, and economic efficiency via intelligent caching, shortlist generation, and graceful degradation mechanisms. We provide a comprehensive framework detailing required resources, a phased implementation strategy with concrete milestones, an offline evaluation protocol utilizing curated test cases (Hit@K, Precision@K, Recall@K, MRR), and an online measurement methodology incorporating query success metrics, zero-result rates, and dwell-time proxies. 

**Abstract (ZH)**: Head Start 程序使用 GoEngage 时，新员工或轮岗员工在尝试在平台主页上定位合适的任务（模块）时面临显著挑战。这些困难源于领域特定的专业术语（例如，IFPA、DRDP），系统特定的命名约定（例如，Application Pool），以及词汇搜索在处理拼写错误和不同词序时的固有限制。我们提出了一种实用的混合语义搜索系统，该系统结合了轻量级的拼写错误容忍词汇检索、基于嵌入的向量相似性以及受约束的大语言模型（LLM）重排序。我们的方法利用组织现有的任务库和知识库基础设施，并通过低误报率、术语变化的可适应性和智能缓存、简短列表生成和优雅降级机制来确保可信度。我们提供了详细的框架，包括所需资源、分阶段的实施策略和具体里程碑、使用精心挑选的测试案例进行离线评估的协议（如 Hit@K、Precision@K、Recall@K、MRR），以及包含查询成功率指标、零结果率和停留时间代理的在线测量方法。 

---
# Improving Metacognition and Uncertainty Communication in Language Models 

**Title (ZH)**: 提高语言模型的元认知和不确定性沟通能力 

**Authors**: Mark Steyvers, Catarina Belem, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2510.05126)  

**Abstract**: Large language models (LLMs) are increasingly used in decision-making contexts, but when they present answers without signaling low confidence, users may unknowingly act on erroneous outputs. While prior work shows that LLMs maintain internal uncertainty signals, their explicit verbalized confidence is typically miscalibrated and poorly discriminates between correct and incorrect answers. Across two types of LLMs, we investigate whether supervised finetuning can improve models' ability to communicate uncertainty and whether such improvements generalize across tasks and domains. We finetune the LLMs on datasets spanning general knowledge, mathematics, and open-ended trivia, and evaluate two metacognitive tasks: (1) single-question confidence estimation, where the model assigns a numeric certainty to its answer, and (2) pairwise confidence comparison, where the model selects which of two answers it is more likely to have correct. We assess generalization to unseen domains, including medical and legal reasoning. Results show that finetuning improves calibration (alignment between stated confidence and accuracy) and discrimination (higher confidence for correct vs. incorrect responses) within and across domains, while leaving accuracy unchanged. However, improvements are task-specific: training on single-question calibration does not transfer to pairwise comparison, and vice versa. In contrast, multitask finetuning on both forms of metacognition yields broader gains, producing lower calibration error and stronger discrimination in out-of-domain evaluations. These results show that while uncertainty communication in LLMs is trainable and generalizable, different metacognitive skills do not naturally reinforce one another and must be developed together through multitask training. 

**Abstract (ZH)**: 大型语言模型（LLMs）在决策场景中的应用日益增多，但当它们不信号低置信度时，用户可能会无意识地依赖错误的输出。虽然先前的研究表明LLMs保留了内部不确定性信号，但其明确表达的置信度通常是不准确的，并且不能很好地区分正确和错误的答案。我们在两种类型的LLMs上研究监督微调能否提高模型在表达不确定性和这种改进是否能在任务和领域之间泛化的能力。我们对涵盖通用知识、数学和开放 trivia 的数据集进行微调，并评估了两个元认知任务：（1）单题置信度估计，模型对其答案给出一个数值确定性；（2）两两置信度比较，模型选择其更有可能正确的答案。我们评估了模型在未知领域（包括医疗和法律推理）中的泛化能力。结果表明，微调可以提高置信度校准（声明的置信度与准确性之间的对齐）和辨别力（正确的答案有更高的置信度），而准确率保持不变。然而，改进具有任务特异性：单题校准训练不适用于两两比较，反之亦然。相比之下，对两种形式的元认知进行的多任务微调能产生更广泛的好处，在领域外评估中，校准误差更低，辨别力更强。这些结果表明，虽然LLMs中的不确定性沟通是可训练和可泛化的，但不同的元认知技能并不会自然相互强化，而是需要通过多任务训练共同开发。 

---
# Hallucination is Inevitable for LLMs with the Open World Assumption 

**Title (ZH)**: 开放世界假设下语言模型的幻觉不可避免 

**Authors**: Bowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05116)  

**Abstract**: Large Language Models (LLMs) exhibit impressive linguistic competence but also produce inaccurate or fabricated outputs, often called ``hallucinations''. Engineering approaches usually regard hallucination as a defect to be minimized, while formal analyses have argued for its theoretical inevitability. Yet both perspectives remain incomplete when considering the conditions required for artificial general intelligence (AGI). This paper reframes ``hallucination'' as a manifestation of the generalization problem. Under the Closed World assumption, where training and test distributions are consistent, hallucinations may be mitigated. Under the Open World assumption, however, where the environment is unbounded, hallucinations become inevitable. This paper further develops a classification of hallucination, distinguishing cases that may be corrected from those that appear unavoidable under open-world conditions. On this basis, it suggests that ``hallucination'' should be approached not merely as an engineering defect but as a structural feature to be tolerated and made compatible with human intelligence. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了卓越的语言能力，但也产生了不准确或虚构的输出，通常称为“幻觉”。在闭世界假设下，训练和测试分布一致时，幻觉可以被缓解。而在开放世界假设下，由于环境无界限，幻觉不可避免。本文进一步对幻觉进行了分类，区分了在开放世界条件下可能纠正的案例和不可避免的案例，并建议应当将“幻觉”不仅视为工程上的缺陷，还视为一种可以容忍和与人类智能兼容的结构性特征。 

---
# COSPADI: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning 

**Title (ZH)**: COSPADI: 通过校准引导的稀疏字典学习压缩大语言模型 

**Authors**: Dmitriy Shopkhoev, Denis Makhov, Magauiya Zhussip, Ammar Ali, Stamatios Lefkimmiatis  

**Link**: [PDF](https://arxiv.org/pdf/2509.22075)  

**Abstract**: Post-training compression of large language models (LLMs) largely relies on low-rank weight approximation, which represents each column of a weight matrix in a shared low-dimensional subspace. While this is a computationally efficient strategy, the imposed structural constraint is rigid and can lead to a noticeable model accuracy drop. In this work, we propose CoSpaDi (Compression via Sparse Dictionary Learning), a novel training-free compression framework that replaces low-rank decomposition with a more flexible structured sparse factorization in which each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix. This formulation enables a union-of-subspaces representation: different columns of the original weight matrix are approximated in distinct subspaces spanned by adaptively selected dictionary atoms, offering greater expressiveness than a single invariant basis. Crucially, CoSpaDi leverages a small calibration dataset to optimize the factorization such that the output activations of compressed projection layers closely match those of the original ones, thereby minimizing functional reconstruction error rather than mere weight approximation. This data-aware strategy preserves better model fidelity without any fine-tuning under reasonable compression ratios. Moreover, the resulting structured sparsity allows efficient sparse-dense matrix multiplication and is compatible with post-training quantization for further memory and latency gains. We evaluate CoSpaDi across multiple Llama and Qwen models under per-layer and per-group settings at 20-50\% compression ratios, demonstrating consistent superiority over state-of-the-art data-aware low-rank methods both in accuracy and perplexity. Our results establish structured sparse dictionary learning as a powerful alternative to conventional low-rank approaches for efficient LLM deployment. 

**Abstract (ZH)**: Post-training压缩大型语言模型（LLMs）主要依赖低秩权重_approximation，即将权重矩阵的每一列表示在共享的低维子空间中。虽然这是一种计算高效的策略，但施加的结构约束较为刚性，可能导致模型准确性下降。在本文中，我们提出了一种名为CoSpaDi（基于稀疏字典学习的压缩）的新型无训练压缩框架，该框架用一个稠密字典和一个列稀疏系数矩阵代替低秩分解，从而表示权重矩阵。这种表示形式允许子空间 union 形式的表示：原始权重矩阵的不同列被自适应选择的字典原子所在的不同的子空间表示，提供了比单一不变基更好的表达能力。至关重要的是，CoSpaDi 利用一个小的校准数据集优化因子分解，使得压缩后的投影层的输出激活值与原始层的输出激活值尽可能接近，从而最小化功能重建误差而不是简单的权重近似。这种数据意识策略在合理的压缩比下能够更好地保持模型的 fidelity，且无需任何微调。此外，产生的结构稀疏性允许高效的稀疏密集矩阵乘法，并与后续的训练后量化兼容，进一步降低内存和延迟开销。我们在_LLAMA 和_Qwen 模型下分别在20-50%的层级和组级压缩比下评估了 CoSpaDi，结果表明，在准确性和困惑度方面，CoSpaDi 比最先进的数据意识低秩方法表现更优。我们的结果确立了结构稀疏字典学习作为一种高效部署大型语言模型的强大替代方法。 

---
# Ads that Talk Back: Implications and Perceptions of Injecting Personalized Advertising into LLM Chatbots 

**Title (ZH)**: 与广告对话：将个性化广告注入LLM聊天机器人的含义与感知 

**Authors**: Brian Jay Tang, Kaiwen Sun, Noah T. Curran, Florian Schaub, Kang G. Shin  

**Link**: [PDF](https://arxiv.org/pdf/2409.15436)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the creation of highly effective chatbots. However, the compute costs of widely deploying LLMs have raised questions about profitability. Companies have proposed exploring ad-based revenue streams for monetizing LLMs, which could serve as the new de facto platform for advertising. This paper investigates the implications of personalizing LLM advertisements to individual users via a between-subjects experiment with 179 participants. We developed a chatbot that embeds personalized product advertisements within LLM responses, inspired by similar forays by AI companies. The evaluation of our benchmarks showed that ad injection only slightly impacted LLM performance, particularly response desirability. Results revealed that participants struggled to detect ads, and even preferred LLM responses with hidden advertisements. Rather than clicking on our advertising disclosure, participants tried changing their advertising settings using natural language queries. We created an advertising dataset and an open-source LLM, Phi-4-Ads, fine-tuned to serve ads and flexibly adapt to user preferences. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进展使得创建高效的聊天机器人成为可能。然而，广泛部署LLMs的计算成本引发了盈利能力的质疑。公司提出了通过广告收入流来 monetize LLMs 的思路，这可能成为新的事实上的广告平台。本文通过一项包含179名参与者的之间实验，探讨了个性化LLM广告对个体用户的影响。我们开发了一个聊天机器人，该机器人在LLM响应中嵌入了个性化的产品广告，灵感来自于类似AI公司的尝试。我们的基准评估显示，广告插入只对LLM性能产生了轻微影响，特别是在响应吸引力方面。结果显示，参与者难以检测到广告，并且甚至更偏好带有隐藏广告的LLM响应。参与者尝试通过自然语言查询来更改他们的广告设置，而不是点击我们的广告披露。我们创建了一个广告数据集和一个开源的LLM Phi-4-Ads，该模型经过微调以提供广告并灵活适应用户偏好。 

---
