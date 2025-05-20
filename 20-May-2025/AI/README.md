# Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards 

**Title (ZH)**: 互信但求证：一种基于可验证奖励的自我验证强化学习方法 

**Authors**: Xiaoyuan Liu, Tian Liang, Zhiwei He, Jiahao Xu, Wenxuan Wang, Pinjia He, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13445)  

**Abstract**: Large Language Models (LLMs) show great promise in complex reasoning, with Reinforcement Learning with Verifiable Rewards (RLVR) being a key enhancement strategy. However, a prevalent issue is ``superficial self-reflection'', where models fail to robustly verify their own outputs. We introduce RISE (Reinforcing Reasoning with Self-Verification), a novel online RL framework designed to tackle this. RISE explicitly and simultaneously trains an LLM to improve both its problem-solving and self-verification abilities within a single, integrated RL process. The core mechanism involves leveraging verifiable rewards from an outcome verifier to provide on-the-fly feedback for both solution generation and self-verification tasks. In each iteration, the model generates solutions, then critiques its own on-policy generated solutions, with both trajectories contributing to the policy update. Extensive experiments on diverse mathematical reasoning benchmarks show that RISE consistently improves model's problem-solving accuracy while concurrently fostering strong self-verification skills. Our analyses highlight the advantages of online verification and the benefits of increased verification compute. Additionally, RISE models exhibit more frequent and accurate self-verification behaviors during reasoning. These advantages reinforce RISE as a flexible and effective path towards developing more robust and self-aware reasoners. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理方面展现出巨大潜力，可信奖励强化学习（RLVR）是关键增强策略之一。然而，一个普遍存在的问题是“表面自我反思”，模型难以稳健地验证自己的输出。我们提出了RISE（强化推理与自我验证）这一新颖的在线RL框架，旨在解决这一问题。RISE在单一集成的RL过程中显式且同时训练LLM，提升其问题解决和自我验证能力。核心机制是利用结果验证器提供的可验证奖励，为解决方案生成和自我验证任务提供实时反馈。在每一轮迭代中，模型生成解决方案，然后对其自身的策略内生成的解决方案进行自我批评，两者共同促进政策更新。在多种数学推理基准上的广泛实验显示，RISE在提高模型问题解决准确性的同时，还提升了其强烈的自我验证能力。我们的分析突显了在线验证的优势以及增加验证计算量的益处。此外，RISE模型在推理过程中表现出更频繁和准确的自我验证行为。这些优势进一步证明了RISE作为开发更稳健和自我意识推理器的灵活有效途径的重要性。 

---
# MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision 

**Title (ZH)**: MM-PRM: 通过可扩展的步骤级监督增强多模态数学推理 

**Authors**: Lingxiao Du, Fanqing Meng, Zongkai Liu, Zhixiang Zhou, Ping Luo, Qiaosheng Zhang, Wenqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13427)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have achieved impressive progress in vision-language understanding, they still struggle with complex multi-step reasoning, often producing logically inconsistent or partially correct solutions. A key limitation lies in the lack of fine-grained supervision over intermediate reasoning steps. To address this, we propose MM-PRM, a process reward model trained within a fully automated, scalable framework. We first build MM-Policy, a strong multimodal model trained on diverse mathematical reasoning data. Then, we construct MM-K12, a curated dataset of 10,000 multimodal math problems with verifiable answers, which serves as seed data. Leveraging a Monte Carlo Tree Search (MCTS)-based pipeline, we generate over 700k step-level annotations without human labeling. The resulting PRM is used to score candidate reasoning paths in the Best-of-N inference setup and achieves significant improvements across both in-domain (MM-K12 test set) and out-of-domain (OlympiadBench, MathVista, etc.) benchmarks. Further analysis confirms the effectiveness of soft labels, smaller learning rates, and path diversity in optimizing PRM performance. MM-PRM demonstrates that process supervision is a powerful tool for enhancing the logical robustness of multimodal reasoning systems. We release all our codes and data at this https URL. 

**Abstract (ZH)**: 尽管多模态大型语言模型（MLLMs）在视觉-语言理解方面取得了显著进展，但它们仍然在复杂多步推理方面挣扎，常常产生逻辑不一致或部分正确的解决方案。一个关键限制在于缺乏对中间推理步骤的精细监督。为解决这一问题，我们提出了MM-PRM，一种在全自动且可扩展框架中训练的过程奖励模型。我们首先构建了MM-Policy，一个在多样性数学推理数据上训练的强大多模态模型。然后，我们构建了包含10,000个可验证答案的多模态数学问题数据集MM-K12，作为种子数据。利用基于蒙特卡洛树搜索（MCTS）的流水线，我们生成了超过70万条步骤级注释，而无需人工标注。结果生成的奖励模型用于在Best-of-N推理设置中评分候选推理路径，并在领域内（MM-K12测试集）和领域外（OlympiadBench、MathVista等）基准测试中均取得显著改进。进一步分析证实，软标签、较小的学习率和路径多样性在优化奖励模型性能方面是有效的。MM-PRM显示了过程监督是增强多模态推理系统逻辑鲁棒性的一种强大工具。我们在此处提供所有代码和数据：this https URL。 

---
# CoT-Kinetics: A Theoretical Modeling Assessing LRM Reasoning Process 

**Title (ZH)**: CoT-Kinetics: 一种评估LRM推理过程的理论建模 

**Authors**: Jinhe Bi, Danqi Yan, Yifan Wang, Wenke Huang, Haokun Chen, Guancheng Wan, Mang Ye, Xun Xiao, Hinrich Schuetze, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.13408)  

**Abstract**: Recent Large Reasoning Models significantly improve the reasoning ability of Large Language Models by learning to reason, exhibiting the promising performance in solving complex tasks. LRMs solve tasks that require complex reasoning by explicitly generating reasoning trajectories together with answers. Nevertheless, judging the quality of such an output answer is not easy because only considering the correctness of the answer is not enough and the soundness of the reasoning trajectory part matters as well. Logically, if the soundness of the reasoning part is poor, even if the answer is correct, the confidence of the derived answer should be low. Existing methods did consider jointly assessing the overall output answer by taking into account the reasoning part, however, their capability is still not satisfactory as the causal relationship of the reasoning to the concluded answer cannot properly reflected. In this paper, inspired by classical mechanics, we present a novel approach towards establishing a CoT-Kinetics energy equation. Specifically, our CoT-Kinetics energy equation formulates the token state transformation process, which is regulated by LRM internal transformer layers, as like a particle kinetics dynamics governed in a mechanical field. Our CoT-Kinetics energy assigns a scalar score to evaluate specifically the soundness of the reasoning phase, telling how confident the derived answer could be given the evaluated reasoning. As such, the LRM's overall output quality can be accurately measured, rather than a coarse judgment (e.g., correct or incorrect) anymore. 

**Abstract (ZH)**: Recent Large Reasoning Models显著提高大型语言模型的推理能力，通过学习推理并在解决复杂任务中表现出色。LRMs通过显式生成推理轨迹和答案来解决需要复杂推理的任务。然而，评估这种输出答案的质量并不容易，因为仅仅考虑答案的正确性是不够的，推理轨迹部分的正确性也很重要。逻辑上，如果推理部分的正确性较差，即使答案正确，推导出的答案的置信度也应该较低。现有方法虽然尝试通过考虑推理部分来联合评估整体输出答案，但它们的能力仍然不尽如人意，无法准确反映推理与结论之间的因果关系。本文受到经典力学的启发，提出了一种新的方法，旨在建立CoT-Kinetics能量方程。具体而言，我们的CoT-Kinetics能量方程将由LRM内部变压器层调节的标记状态转换过程，类比于在机械场中受治理的粒子动力学。CoT-Kinetics能量为评估推理阶段的正确性分配一个标量分数，告诉在评估的推理下推导出的答案可以有多大的置信度。因此，可以精确测量LRM的整体输出质量，而不仅仅是粗略判断（如正确或错误）。 

---
# AutoMathKG: The automated mathematical knowledge graph based on LLM and vector database 

**Title (ZH)**: AutoMathKG：基于LLM和向量数据库的自动数学知识图谱 

**Authors**: Rong Bian, Yu Geng, Zijian Yang, Bing Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13406)  

**Abstract**: A mathematical knowledge graph (KG) presents knowledge within the field of mathematics in a structured manner. Constructing a math KG using natural language is an essential but challenging task. There are two major limitations of existing works: first, they are constrained by corpus completeness, often discarding or manually supplementing incomplete knowledge; second, they typically fail to fully automate the integration of diverse knowledge sources. This paper proposes AutoMathKG, a high-quality, wide-coverage, and multi-dimensional math KG capable of automatic updates. AutoMathKG regards mathematics as a vast directed graph composed of Definition, Theorem, and Problem entities, with their reference relationships as edges. It integrates knowledge from ProofWiki, textbooks, arXiv papers, and TheoremQA, enhancing entities and relationships with large language models (LLMs) via in-context learning for data augmentation. To search for similar entities, MathVD, a vector database, is built through two designed embedding strategies using SBERT. To automatically update, two mechanisms are proposed. For knowledge completion mechanism, Math LLM is developed to interact with AutoMathKG, providing missing proofs or solutions. For knowledge fusion mechanism, MathVD is used to retrieve similar entities, and LLM is used to determine whether to merge with a candidate or add as a new entity. A wide range of experiments demonstrate the advanced performance and broad applicability of the AutoMathKG system, including superior reachability query results in MathVD compared to five baselines and robust mathematical reasoning capability in Math LLM. 

**Abstract (ZH)**: 一种数学知识图谱（KG）以结构化方式呈现数学领域的知识。使用自然语言构建数学KG是一项重要但具有挑战性的任务。现有工作的两个主要限制是：首先，它们受语料库完整性的限制，经常丢弃或手动补充不完整的信息；其次，它们通常无法完全自动化地整合多种知识源。本文提出AutoMathKG，这是一种高质量、覆盖面广、多维度的数学KG，能够自动更新。AutoMathKG将数学视为由定义、定理和问题实体组成的庞大有向图，它们之间的参考关系作为边。它通过使用SBERT设计的嵌入策略构建MathVD向量数据库，并通过上下文学习使用大语言模型（LLMs）增强实体和关系进行数据扩充。为查找相似实体，通过设计的嵌入策略使用SBERT构建MathVD向量数据库。为了自动更新，提出了两种机制。对于知识完成机制，开发了Math LLM与AutoMathKG交互，提供缺失的证明或解决方案。对于知识融合机制，使用MathVD检索相似实体，并使用LLM确定是否将候选实体合并或作为新实体添加。广泛实验展示了AutoMathKG系统的先进性能和广泛应用，包括在MathVD中相比五个基线具有更优秀的可达查询结果和强大的数学推理能力。 

---
# Robin: A multi-agent system for automating scientific discovery 

**Title (ZH)**: Robin：一种用于自动化科学发现的多智能体系统 

**Authors**: Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J. Szostkiewicz, Jon M. Laurent, Muhammed T. Razzak, Andrew D. White, Michaela M. Hinks, Samuel G. Rodriques  

**Link**: [PDF](https://arxiv.org/pdf/2505.13400)  

**Abstract**: Scientific discovery is driven by the iterative process of background research, hypothesis generation, experimentation, and data analysis. Despite recent advancements in applying artificial intelligence to scientific discovery, no system has yet automated all of these stages in a single workflow. Here, we introduce Robin, the first multi-agent system capable of fully automating the key intellectual steps of the scientific process. By integrating literature search agents with data analysis agents, Robin can generate hypotheses, propose experiments, interpret experimental results, and generate updated hypotheses, achieving a semi-autonomous approach to scientific discovery. By applying this system, we were able to identify a novel treatment for dry age-related macular degeneration (dAMD), the major cause of blindness in the developed world. Robin proposed enhancing retinal pigment epithelium phagocytosis as a therapeutic strategy, and identified and validated a promising therapeutic candidate, ripasudil. Ripasudil is a clinically-used rho kinase (ROCK) inhibitor that has never previously been proposed for treating dAMD. To elucidate the mechanism of ripasudil-induced upregulation of phagocytosis, Robin then proposed and analyzed a follow-up RNA-seq experiment, which revealed upregulation of ABCA1, a critical lipid efflux pump and possible novel target. All hypotheses, experimental plans, data analyses, and data figures in the main text of this report were produced by Robin. As the first AI system to autonomously discover and validate a novel therapeutic candidate within an iterative lab-in-the-loop framework, Robin establishes a new paradigm for AI-driven scientific discovery. 

**Abstract (ZH)**: 科学发现由背景研究、假设生成、实验和数据分析的迭代过程驱动。尽管最近在将人工智能应用于科学发现方面取得了进展，但还没有系统能够在单一工作流中自动化这些所有阶段。在这里，我们介绍了Robin，这是首个能够全面自动化科学过程中的关键智力步骤的多智能体系统。通过将文献搜索智能体与数据分析智能体集成，Robin 能够生成假设、提出实验、解释实验结果，并生成更新的假设，实现一种半自主的科学发现方法。通过应用此系统，我们识别出一种新型治疗干性年龄相关黄斑变性（dAMD）的方法，这是世界上发达国家主要的致盲原因。Robin提议增强视网膜色素上皮细胞的吞噬作用作为治疗策略，并确定和验证了一种有前途的治疗候选物雷帕新达匹。雷帕新达匹是一种临床使用的Rho激酶（ROCK）抑制剂，之前从未被提议用于治疗dAMD。为阐明雷帕新达匹诱导的吞噬作用上调机制，Robin随后提出了并分析了一个后续的RNA-seq实验，揭示了ABCA1的上调，ABCA1是一种关键的脂质外排泵，可能是新的潜在靶标。本文主要文本中的所有假设、实验计划、数据分析和数据图表均由Robin生成。作为首个在迭代实验室框架内独立发现并验证新型治疗候选物的AI系统，Robin确立了AI驱动科学发现的新范式。 

---
# Advancing Generalization Across a Variety of Abstract Visual Reasoning Tasks 

**Title (ZH)**: 跨多种抽象视觉推理任务提升泛化能力 

**Authors**: Mikołaj Małkiński, Jacek Mańdziuk  

**Link**: [PDF](https://arxiv.org/pdf/2505.13391)  

**Abstract**: The abstract visual reasoning (AVR) domain presents a diverse suite of analogy-based tasks devoted to studying model generalization. Recent years have brought dynamic progress in the field, particularly in i.i.d. scenarios, in which models are trained and evaluated on the same data distributions. Nevertheless, o.o.d. setups that assess model generalization to new test distributions remain challenging even for the most recent models. To advance generalization in AVR tasks, we present the Pathways of Normalized Group Convolution model (PoNG), a novel neural architecture that features group convolution, normalization, and a parallel design. We consider a wide set of AVR benchmarks, including Raven's Progressive Matrices and visual analogy problems with both synthetic and real-world images. The experiments demonstrate strong generalization capabilities of the proposed model, which in several settings outperforms the existing literature methods. 

**Abstract (ZH)**: 归一化组卷积路径模型（PoNG）：视觉推理中的新颖神经架构 

---
# CompeteSMoE -- Statistically Guaranteed Mixture of Experts Training via Competition 

**Title (ZH)**: CompeteSMoE —— 统计保证的专家混合训练竞争方法 

**Authors**: Nam V. Nguyen, Huy Nguyen, Quang Pham, Van Nguyen, Savitha Ramasamy, Nhat Ho  

**Link**: [PDF](https://arxiv.org/pdf/2505.13380)  

**Abstract**: Sparse mixture of experts (SMoE) offers an appealing solution to scale up the model complexity beyond the mean of increasing the network's depth or width. However, we argue that effective SMoE training remains challenging because of the suboptimal routing process where experts that perform computation do not directly contribute to the routing process. In this work, we propose competition, a novel mechanism to route tokens to experts with the highest neural response. Theoretically, we show that the competition mechanism enjoys a better sample efficiency than the traditional softmax routing. Furthermore, we develop CompeteSMoE, a simple yet effective algorithm to train large language models by deploying a router to learn the competition policy, thus enjoying strong performances at a low training overhead. Our extensive empirical evaluations on both the visual instruction tuning and language pre-training tasks demonstrate the efficacy, robustness, and scalability of CompeteSMoE compared to state-of-the-art SMoE strategies. We have made the implementation available at: this https URL. This work is an improved version of the previous study at arXiv:2402.02526 

**Abstract (ZH)**: Sparse混合专家模型（SMoE）提供了一种超出增加网络深度或宽度来提升模型复杂性的有吸引力的解决方案。然而，我们认为有效的SMoE训练由于专家进行计算但不直接参与路由过程的次优路由过程而仍然具有挑战性。在本文中，我们提出了一种新颖的竞争机制，用于将令牌路由到具有最高神经响应的专家。理论上，我们证明了竞争机制在样本效率方面优于传统的softmax路由。此外，我们开发了CompeteSMoE，这是一种简单而有效的算法，通过部署一个路由器来学习竞争策略来训练大型语言模型，从而在较低的训练开销下享受良好的性能。我们通过对视觉指令调优和语言预训练任务的广泛实证评估，展示了与当前最先进的SMoE策略相比，CompeteSMoE的有效性、稳健性和可扩展性。我们已将实现代码发布在：this https URL。本工作是arXiv:2402.02526的改进版。 

---
# Exploiting Symbolic Heuristics for the Synthesis of Domain-Specific Temporal Planning Guidance using Reinforcement Learning 

**Title (ZH)**: 利用符号启发式方法为强化学习合成领域特定的时间规划指导 

**Authors**: Irene Brugnara, Alessandro Valentini, Andrea Micheli  

**Link**: [PDF](https://arxiv.org/pdf/2505.13372)  

**Abstract**: Recent work investigated the use of Reinforcement Learning (RL) for the synthesis of heuristic guidance to improve the performance of temporal planners when a domain is fixed and a set of training problems (not plans) is given. The idea is to extract a heuristic from the value function of a particular (possibly infinite-state) MDP constructed over the training problems.
In this paper, we propose an evolution of this learning and planning framework that focuses on exploiting the information provided by symbolic heuristics during both the RL and planning phases. First, we formalize different reward schemata for the synthesis and use symbolic heuristics to mitigate the problems caused by the truncation of episodes needed to deal with the potentially infinite MDP. Second, we propose learning a residual of an existing symbolic heuristic, which is a "correction" of the heuristic value, instead of eagerly learning the whole heuristic from scratch. Finally, we use the learned heuristic in combination with a symbolic heuristic using a multiple-queue planning approach to balance systematic search with imperfect learned information. We experimentally compare all the approaches, highlighting their strengths and weaknesses and significantly advancing the state of the art for this planning and learning schema. 

**Abstract (ZH)**: Recent Work Investigated the Use of Reinforcement Learning for Synthesizing Heuristic Guidance to Improve Temporal Planner Performance with Fixed Domains and Given Training Problems: This Paper Proposes an Evolution of the Learning and Planning Framework That Exploits Symbolic Heuristics During Both the RL and Planning Phases 

---
# Multi-Armed Bandits Meet Large Language Models 

**Title (ZH)**: 多臂 Bandits 遇上大规模语言模型 

**Authors**: Djallel Bouneffouf, Raphael Feraud  

**Link**: [PDF](https://arxiv.org/pdf/2505.13355)  

**Abstract**: Bandit algorithms and Large Language Models (LLMs) have emerged as powerful tools in artificial intelligence, each addressing distinct yet complementary challenges in decision-making and natural language processing. This survey explores the synergistic potential between these two fields, highlighting how bandit algorithms can enhance the performance of LLMs and how LLMs, in turn, can provide novel insights for improving bandit-based decision-making. We first examine the role of bandit algorithms in optimizing LLM fine-tuning, prompt engineering, and adaptive response generation, focusing on their ability to balance exploration and exploitation in large-scale learning tasks. Subsequently, we explore how LLMs can augment bandit algorithms through advanced contextual understanding, dynamic adaptation, and improved policy selection using natural language reasoning. By providing a comprehensive review of existing research and identifying key challenges and opportunities, this survey aims to bridge the gap between bandit algorithms and LLMs, paving the way for innovative applications and interdisciplinary research in AI. 

**Abstract (ZH)**: -bandit算法与大语言模型（LLMs）在人工智能领域 emerged as 强大的工具，分别在决策制定和自然语言处理中解决独特而互补的挑战。本文综述探讨了这两个领域之间的协同潜力，强调bandit算法如何提升LLMs的性能，以及LLMs如何为基于bandit的决策制定提供新的见解。我们首先研究了bandit算法在优化LLM微调、提示工程和自适应响应生成中的作用，重点在于它们在大规模学习任务中平衡探索与利用的能力。随后，我们探讨了LLMs如何通过高级语境理解、动态适应和基于自然语言推理改进政策选择来增强bandit算法。通过全面回顾现有研究并识别关键挑战和机遇，本文旨在弥合bandit算法与LLMs之间的差距，为人工智能中的创新应用和跨学科研究铺平道路。 

---
# Level Generation with Quantum Reservoir Computing 

**Title (ZH)**: 基于量子蓄水池计算的关卡生成 

**Authors**: João S. Ferreira, Pierre Fromholz, Hari Shaji, James R. Wootton  

**Link**: [PDF](https://arxiv.org/pdf/2505.13287)  

**Abstract**: Reservoir computing is a form of machine learning particularly suited for time series analysis, including forecasting predictions. We take an implementation of \emph{quantum} reservoir computing that was initially designed to generate variants of musical scores and adapt it to create levels of Super Mario Bros. Motivated by our analysis of these levels, we develop a new Roblox \textit{obby} where the courses can be generated in real time on superconducting qubit hardware, and investigate some of the constraints placed by such real-time generation. 

**Abstract (ZH)**: 量子蓄水池计算是一种特别适合时间序列分析的机器学习方法，包括预测预报。我们借鉴最初设计用于生成音乐谱变体的量子蓄水池计算实现，将其改编以创建《超级马里奥 Bros.》的游戏关卡。基于对这些关卡的分析，我们开发了一个新的Roblox 义曲（obby），其中赛道可以在超导量子比特硬件上实时生成，并探讨了此类实时生成带来的某些约束。 

---
# Seeing the Unseen: How EMoE Unveils Bias in Text-to-Image Diffusion Models 

**Title (ZH)**: 看不见的可视化工匠：EMoE如何揭示文本到图像扩散模型中的偏见 

**Authors**: Lucas Berry, Axel Brando, Wei-Di Chang, Juan Camilo Gamboa Higuera, David Meger  

**Link**: [PDF](https://arxiv.org/pdf/2505.13273)  

**Abstract**: Estimating uncertainty in text-to-image diffusion models is challenging because of their large parameter counts (often exceeding 100 million) and operation in complex, high-dimensional spaces with virtually infinite input possibilities. In this paper, we propose Epistemic Mixture of Experts (EMoE), a novel framework for efficiently estimating epistemic uncertainty in diffusion models. EMoE leverages pre-trained networks without requiring additional training, enabling direct uncertainty estimation from a prompt. We leverage a latent space within the diffusion process that captures epistemic uncertainty better than existing methods. Experimental results on the COCO dataset demonstrate EMoE's effectiveness, showing a strong correlation between uncertainty and image quality. Additionally, EMoE identifies under-sampled languages and regions with higher uncertainty, revealing hidden biases in the training set. This capability demonstrates the relevance of EMoE as a tool for addressing fairness and accountability in AI-generated content. 

**Abstract (ZH)**: 基于文本生成图像的扩散模型中估计知识不确定性具有挑战性，这是因为模型有大量的参数（通常超过1亿）并在复杂、高维的空间中运行，输入可能性几乎是无限的。本文提出了一种名为Epistemic Mixture of Experts (EMoE)的新框架，用于高效估计扩散模型的知识不确定性。EMoE 利用预训练网络而不需要额外训练，能够直接从提示中估计不确定性。EMoE 利用了一个在扩散过程中更好地捕捉知识不确定性的时间空间。在COCO数据集上的实验结果表明，EMoE 的有效性，不确定性与图像质量之间存在强烈的关联。此外，EMoE 还能够识别欠采样的语言和具有更高不确定性的地区，揭示训练集中的隐藏偏差。这一能力展示了EMoE作为解决AI生成内容中的公平性和问责制问题工具的相关性。 

---
# Agentic Publications: An LLM-Driven Framework for Interactive Scientific Publishing, Supplementing Traditional Papers with AI-Powered Knowledge Systems 

**Title (ZH)**: 代理出版物：一种由LLM驱动的交互式科学出版框架，以AI增强的知识系统补充传统论文 

**Authors**: Roberto Pugliese, George Kourousias, Francesco Venier, Grazia Garlatti Costa  

**Link**: [PDF](https://arxiv.org/pdf/2505.13246)  

**Abstract**: The exponential growth of scientific literature presents significant challenges for researchers navigating the complex knowledge landscape. We propose "Agentic Publications", a novel LLM-driven framework complementing traditional publishing by transforming papers into interactive knowledge systems. Our architecture integrates structured data with unstructured content through retrieval-augmented generation and multi-agent verification. The framework offers interfaces for both humans and machines, combining narrative explanations with machine-readable outputs while addressing ethical considerations through automated validation and transparent governance. Key features include continuous knowledge updates, automatic integration of new findings, and customizable detail levels. Our proof-of-concept demonstrates multilingual interaction, API accessibility, and structured knowledge representation through vector databases, knowledge graphs, and verification agents. This approach enhances scientific communication across disciplines, improving efficiency and collaboration while preserving traditional publishing pathways, particularly valuable for interdisciplinary fields where knowledge integration remains challenging. 

**Abstract (ZH)**: 科学文献的指数增长为研究人员导航复杂知识景观带来了显著挑战。我们提出“自主出版”这一新型LLM驱动框架，通过将论文转化为互动知识系统来补充传统出版方式。该架构通过检索增强生成和多代理验证将结构化数据与非结构化内容相结合。该框架为人类和机器提供接口，结合叙述性解释与机器可读输出，并通过自动化验证和透明治理解决伦理考量。关键功能包括持续的知识更新、自动整合新发现以及可定制的详细级别。概念验证展示了多语言交互、API访问能力和通过向量数据库、知识图谱和验证代理呈现结构化知识。该方法增强跨学科的科学交流，提高效率与协作，同时保留传统出版路径，特别是在知识整合仍具挑战性的跨学科领域尤为重要。 

---
# StarFT: Robust Fine-tuning of Zero-shot Models via Spuriosity Alignment 

**Title (ZH)**: StarFT: 通过 spuriosity 对齐的零-shot 模型稳健微调 

**Authors**: Younghyun Kim, Jongheon Jeong, Sangkyung Kwak, Kyungmin Lee, Juho Lee, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13232)  

**Abstract**: Learning robust representations from data often requires scale, which has led to the success of recent zero-shot models such as CLIP. However, the obtained robustness can easily be deteriorated when these models are fine-tuned on other downstream tasks (e.g., of smaller scales). Previous works often interpret this phenomenon in the context of domain shift, developing fine-tuning methods that aim to preserve the original domain as much as possible. However, in a different context, fine-tuned models with limited data are also prone to learning features that are spurious to humans, such as background or texture. In this paper, we propose StarFT (Spurious Textual Alignment Regularization), a novel framework for fine-tuning zero-shot models to enhance robustness by preventing them from learning spuriosity. We introduce a regularization that aligns the output distribution for spuriosity-injected labels with the original zero-shot model, ensuring that the model is not induced to extract irrelevant features further from these this http URL leverage recent language models to get such spuriosity-injected labels by generating alternative textual descriptions that highlight potentially confounding this http URL experiments validate the robust generalization of StarFT and its emerging properties: zero-shot group robustness and improved zero-shot classification. Notably, StarFT boosts both worst-group and average accuracy by 14.30% and 3.02%, respectively, in the Waterbirds group shift scenario, where other robust fine-tuning baselines show even degraded performance. 

**Abstract (ZH)**: 基于消除多余特性的零样本模型细调以增强鲁棒性 

---
# Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis 

**Title (ZH)**: 通过用户界面分解与合成扩展计算机使用语义关联 

**Authors**: Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, Yiheng Xu, Junli Wang, Doyen Sahoo, Tao Yu, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.13227)  

**Abstract**: Graphical user interface (GUI) grounding, the ability to map natural language instructions to specific actions on graphical user interfaces, remains a critical bottleneck in computer use agent development. Current benchmarks oversimplify grounding tasks as short referring expressions, failing to capture the complexity of real-world interactions that require software commonsense, layout understanding, and fine-grained manipulation capabilities. To address these limitations, we introduce OSWorld-G, a comprehensive benchmark comprising 564 finely annotated samples across diverse task types including text matching, element recognition, layout understanding, and precise manipulation. Additionally, we synthesize and release the largest computer use grounding dataset Jedi, which contains 4 million examples through multi-perspective decoupling of tasks. Our multi-scale models trained on Jedi demonstrate its effectiveness by outperforming existing approaches on ScreenSpot-v2, ScreenSpot-Pro, and our OSWorld-G. Furthermore, we demonstrate that improved grounding with Jedi directly enhances agentic capabilities of general foundation models on complex computer tasks, improving from 5% to 27% on OSWorld. Through detailed ablation studies, we identify key factors contributing to grounding performance and verify that combining specialized data for different interface elements enables compositional generalization to novel interfaces. All benchmark, data, checkpoints, and code are open-sourced and available at this https URL. 

**Abstract (ZH)**: 图形用户界面（GUI） grounding：从自然语言指令到图形用户界面具体操作的映射能力仍然是计算机使用代理开发中的关键瓶颈。当前的基准测试过度简化了 grounding 任务为简短的引用表达式，未能捕捉到需要软件常识、布局理解以及精细操作能力的现实世界交互的复杂性。为解决这些限制，我们引入了 OSWorld-G，这是一个全面的基准测试，包含564个细粒度注释的样本，涵盖了包括文本匹配、元素识别、布局理解和精确操作在内的多种任务类型。此外，我们合成并发布了最大的计算机使用 grounding 数据集 Jedi，该数据集通过多视角解耦任务包含400万个示例。在 Jedi 上训练的多尺度模型在 ScreenSpot-v2、ScreenSpot-Pro 和我们的 OSWorld-G 上表现出色，超越了现有方法。此外，我们证明改进的 grounding 与 Jedi 直接增强了通用基础模型在复杂计算机任务中的代理能力，在 OSWorld 上的表现提高了22个百分点。通过详细的消融研究，我们确定了影响 grounding 性能的关键因素，并验证了将不同界面元素的专业化数据结合可以实现对新型界面的组合泛化。所有基准测试、数据、检查点和代码均开源，并可在以下网址获取。 

---
# Adversarial Testing in LLMs: Insights into Decision-Making Vulnerabilities 

**Title (ZH)**: LLM中的对抗性测试：决策漏洞洞察 

**Authors**: Lili Zhang, Haomiaomiao Wang, Long Cheng, Libao Deng, Tomas Ward  

**Link**: [PDF](https://arxiv.org/pdf/2505.13195)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into real-world decision-making systems, understanding their behavioural vulnerabilities remains a critical challenge for AI safety and alignment. While existing evaluation metrics focus primarily on reasoning accuracy or factual correctness, they often overlook whether LLMs are robust to adversarial manipulation or capable of using adaptive strategy in dynamic environments. This paper introduces an adversarial evaluation framework designed to systematically stress-test the decision-making processes of LLMs under interactive and adversarial conditions. Drawing on methodologies from cognitive psychology and game theory, our framework probes how models respond in two canonical tasks: the two-armed bandit task and the Multi-Round Trust Task. These tasks capture key aspects of exploration-exploitation trade-offs, social cooperation, and strategic flexibility. We apply this framework to several state-of-the-art LLMs, including GPT-3.5, GPT-4, Gemini-1.5, and DeepSeek-V3, revealing model-specific susceptibilities to manipulation and rigidity in strategy adaptation. Our findings highlight distinct behavioral patterns across models and emphasize the importance of adaptability and fairness recognition for trustworthy AI deployment. Rather than offering a performance benchmark, this work proposes a methodology for diagnosing decision-making weaknesses in LLM-based agents, providing actionable insights for alignment and safety research. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）越来越多地集成到实际决策系统中，理解其行为漏洞仍然是人工智能安全和对齐中的关键挑战。虽然现有评估指标主要关注推理准确度或事实准确性，但往往忽视了LLMs对对抗操纵的鲁棒性或在动态环境中使用适应策略的能力。本文介绍了一种对抗性评估框架，旨在在交互和对抗条件下系统地测试LLMs的决策过程。该框架借鉴了认知心理学和博弈论的方法，探讨了模型在经典任务中的响应方式：两臂-bandit任务和多轮信任任务。这些任务捕捉了探索与利用之间的权衡、社会合作以及策略灵活性的关键方面。我们将这一框架应用于包括GPT-3.5、GPT-4、Gemini-1.5和DeepSeek-V3在内的多个最新大语言模型，揭示了模型特定的操纵脆弱性和策略适应僵化。我们的发现突显了模型之间不同的行为模式，并强调了在可信的人工智能部署中适应性和公平性识别的重要性。本文不提供性能基准，而是提出了一种诊断基于大语言模型的智能体决策弱点的方法，为对齐和安全性研究提供了可操作的见解。 

---
# ViPlan: A Benchmark for Visual Planning with Symbolic Predicates and Vision-Language Models 

**Title (ZH)**: ViPlan: 一个基于符号谓词和视觉语言模型的视觉规划基准 

**Authors**: Matteo Merler, Nicola Dainese, Minttu Alakuijala, Giovanni Bonetta, Pietro Ferrazzi, Yu Tian, Bernardo Magnini, Pekka Marttinen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13180)  

**Abstract**: Integrating Large Language Models with symbolic planners is a promising direction for obtaining verifiable and grounded plans compared to planning in natural language, with recent works extending this idea to visual domains using Vision-Language Models (VLMs). However, rigorous comparison between VLM-grounded symbolic approaches and methods that plan directly with a VLM has been hindered by a lack of common environments, evaluation protocols and model coverage. We introduce ViPlan, the first open-source benchmark for Visual Planning with symbolic predicates and VLMs. ViPlan features a series of increasingly challenging tasks in two domains: a visual variant of the classic Blocksworld planning problem and a simulated household robotics environment. We benchmark nine open-source VLM families across multiple sizes, along with selected closed models, evaluating both VLM-grounded symbolic planning and using the models directly to propose actions. We find symbolic planning to outperform direct VLM planning in Blocksworld, where accurate image grounding is crucial, whereas the opposite is true in the household robotics tasks, where commonsense knowledge and the ability to recover from errors are beneficial. Finally, we show that across most models and methods, there is no significant benefit to using Chain-of-Thought prompting, suggesting that current VLMs still struggle with visual reasoning. 

**Abstract (ZH)**: 将大规模语言模型与符号规划集成是一种 promise 方向，用于获得可验证和基于grounded的规划，相比自然语言规划， Recent 工作通过使用视觉语言模型（VLMs）将这一想法扩展到了视觉领域。然而，由于缺乏共同环境、评估协议和模型覆盖率，严格比较基于VLM的符号方法和直接使用VLM进行规划的方法受到了阻碍。我们引入了ViPlan，这是第一个针对视觉规划的开源基准，包含符号谓词和VLM。ViPlan 在两个领域中包含了一系列逐步增强的任务：经典的BlocksWorld 规划问题的视觉变体以及模拟的家庭机器人环境。我们在多种规模的九个开源VLM 家族上进行了基准测试，同时选择了一些封闭模型，评估了基于VLM的符号规划和直接使用模型提出动作的效果。我们发现，在BlocksWorld中，准确的图像grounding 对符号规划有更好的效果，而在家庭机器人任务中，常识知识和从错误中恢复的能力更有利。最后，我们表明，在大多数模型和方法中，Chain-of-Thought 提示并没有提供显著的好处，这表明当前的VLM 仍在视觉推理方面存在困难。 

---
# Enhancing LLMs for Time Series Forecasting via Structure-Guided Cross-Modal Alignment 

**Title (ZH)**: 通过结构引导的跨模态对齐增强LLMs的时间序列预测能力 

**Authors**: Siming Sun, Kai Zhang, Xuejun Jiang, Wenchao Meng, Qinmin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13175)  

**Abstract**: The emerging paradigm of leveraging pretrained large language models (LLMs) for time series forecasting has predominantly employed linguistic-temporal modality alignment strategies through token-level or layer-wise feature mapping. However, these approaches fundamentally neglect a critical insight: the core competency of LLMs resides not merely in processing localized token features but in their inherent capacity to model holistic sequence structures. This paper posits that effective cross-modal alignment necessitates structural consistency at the sequence level. We propose the Structure-Guided Cross-Modal Alignment (SGCMA), a framework that fully exploits and aligns the state-transition graph structures shared by time-series and linguistic data as sequential modalities, thereby endowing time series with language-like properties and delivering stronger generalization after modality alignment. SGCMA consists of two key components, namely Structure Alignment and Semantic Alignment. In Structure Alignment, a state transition matrix is learned from text data through Hidden Markov Models (HMMs), and a shallow transformer-based Maximum Entropy Markov Model (MEMM) receives the hot-start transition matrix and annotates each temporal patch into state probability, ensuring that the temporal representation sequence inherits language-like sequential dynamics. In Semantic Alignment, cross-attention is applied between temporal patches and the top-k tokens within each state, and the ultimate temporal embeddings are derived by the expected value of these embeddings using a weighted average based on state probabilities. Experiments on multiple benchmarks demonstrate that SGCMA achieves state-of-the-art performance, offering a novel approach to cross-modal alignment in time series forecasting. 

**Abstract (ZH)**: 基于结构导向的跨模态对齐框架（SGCMA）：提升时间序列预测中的语言时间序列属性 

---
# Zero-Shot Iterative Formalization and Planning in Partially Observable Environments 

**Title (ZH)**: 零样本迭代形式化与规划在部分可观测环境中 

**Authors**: Liancheng Gong, Wang Zhu, Jesse Thomason, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13126)  

**Abstract**: In planning, using LLMs not to predict plans but to formalize an environment into the Planning Domain Definition Language (PDDL) has been shown to greatly improve performance and control. While most work focused on fully observable environments, we tackle the more realistic and challenging partially observable environments where existing methods are incapacitated by the lack of complete information. We propose PDDLego+, a framework to iteratively formalize, plan, grow, and refine PDDL representations in a zero-shot manner, without needing access to any existing trajectories. On two textual simulated environments, we show that PDDLego+ not only achieves superior performance, but also shows robustness against problem complexity. We also show that the domain knowledge captured after a successful trial is interpretable and benefits future tasks. 

**Abstract (ZH)**: 使用LLM将环境形式化为PDDL以提高规划性能和控制：PDDLego+框架在部分可观测环境中的应用 

---
# Unveil Sources of Uncertainty: Feature Contribution to Conformal Prediction Intervals 

**Title (ZH)**: 揭示不确定性来源：特征对齐信任区间的影响分析 

**Authors**: Marouane Il Idrissi, Agathe Fernandes Machado, Ewen Gallic, Arthur Charpentier  

**Link**: [PDF](https://arxiv.org/pdf/2505.13118)  

**Abstract**: Cooperative game theory methods, notably Shapley values, have significantly enhanced machine learning (ML) interpretability. However, existing explainable AI (XAI) frameworks mainly attribute average model predictions, overlooking predictive uncertainty. This work addresses that gap by proposing a novel, model-agnostic uncertainty attribution (UA) method grounded in conformal prediction (CP). By defining cooperative games where CP interval properties-such as width and bounds-serve as value functions, we systematically attribute predictive uncertainty to input features. Extending beyond the traditional Shapley values, we use the richer class of Harsanyi allocations, and in particular the proportional Shapley values, which distribute attribution proportionally to feature importance. We propose a Monte Carlo approximation method with robust statistical guarantees to address computational feasibility, significantly improving runtime efficiency. Our comprehensive experiments on synthetic benchmarks and real-world datasets demonstrate the practical utility and interpretative depth of our approach. By combining cooperative game theory and conformal prediction, we offer a rigorous, flexible toolkit for understanding and communicating predictive uncertainty in high-stakes ML applications. 

**Abstract (ZH)**: 合作博弈论方法，尤其是夏普利值，显著增强了机器学习的可解释性。然而，现有的可解释人工智能框架主要归因于平均模型预测，忽视了预测不确定性。本工作通过提出一种基于校准预测的新型、模型无关的不确定性归因（UA）方法来填补这一空白。通过将校准预测区间性质（如宽度和边界）定义为价值函数，我们将预测不确定性系统地归因于输入特征。我们超越了传统的夏普利值，使用更丰富的哈萨尼分配类，特别是按比例的夏普利值，这将归因按特征重要性比例分配。我们提出了一种具有稳健统计保证的蒙特卡洛逼近方法，以解决计算可行性问题，显著提高运行效率。我们在合成基准和真实世界数据集上的全面实验表明了我们方法的实用性和解释深度。通过结合合作博弈论和校准预测，我们提供了一种严谨且灵活的工具箱，用于理解并沟通高风险机器学习应用中的预测不确定性。 

---
# LLM-KG-Bench 3.0: A Compass for SemanticTechnology Capabilities in the Ocean of LLMs 

**Title (ZH)**: LLM-KG-Bench 3.0：导航LLM海洋中语义技术能力的指南针 

**Authors**: Lars-Peter Meyer, Johannes Frey, Desiree Heim, Felix Brei, Claus Stadler, Kurt Junghanns, Michael Martin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13098)  

**Abstract**: Current Large Language Models (LLMs) can assist developing program code beside many other things, but can they support working with Knowledge Graphs (KGs) as well? Which LLM is offering the best capabilities in the field of Semantic Web and Knowledge Graph Engineering (KGE)? Is this possible to determine without checking many answers manually? The LLM-KG-Bench framework in Version 3.0 is designed to answer these questions. It consists of an extensible set of tasks for automated evaluation of LLM answers and covers different aspects of working with semantic technologies. In this paper the LLM-KG-Bench framework is presented in Version 3 along with a dataset of prompts, answers and evaluations generated with it and several state-of-the-art LLMs. Significant enhancements have been made to the framework since its initial release, including an updated task API that offers greater flexibility in handling evaluation tasks, revised tasks, and extended support for various open models through the vllm library, among other improvements. A comprehensive dataset has been generated using more than 30 contemporary open and proprietary LLMs, enabling the creation of exemplary model cards that demonstrate the models' capabilities in working with RDF and SPARQL, as well as comparing their performance on Turtle and JSON-LD RDF serialization tasks. 

**Abstract (ZH)**: 当前大型语言模型（LLMs）可以在开发程序代码等方面提供帮助，但它们能否支持与知识图谱（KGs）的交互呢？哪些LLM在语义网和知识图谱工程（KGE）领域提供了最好的能力？我们能否在不手动检查多个答案的情况下做出判断？LLM-KG-Bench框架（版本3.0）旨在回答这些问题。它包括一组可扩展的任务，用于自动评估LLM的答案，并涵盖与语义技术交互的不同方面。本文介绍了LLM-KG-Bench框架（版本3），并提供了一个使用该框架及其几种最先进的LLM生成的数据集。自首次发布以来，该框架进行了显著增强，包括更新的任务API，提供了更大的灵活性以处理评估任务，修订的任务，以及通过vllm库扩展对各种开源模型的支持等改进。使用超过30种当代开源和专有LLM生成了一个全面的数据集，使创建展示模型在处理RDF和SPARQL以及比较其在turtle和JSON-LD RDF序列化任务上性能的示例模型卡片成为可能。 

---
# CAIM: Development and Evaluation of a Cognitive AI Memory Framework for Long-Term Interaction with Intelligent Agents 

**Title (ZH)**: CAIM：认知AI记忆框架的开发与评估——面向智能代理的长期交互 

**Authors**: Rebecca Westhäußer, Frederik Berenz, Wolfgang Minker, Sebastian Zepf  

**Link**: [PDF](https://arxiv.org/pdf/2505.13044)  

**Abstract**: Large language models (LLMs) have advanced the field of artificial intelligence (AI) and are a powerful enabler for interactive systems. However, they still face challenges in long-term interactions that require adaptation towards the user as well as contextual knowledge and understanding of the ever-changing environment. To overcome these challenges, holistic memory modeling is required to efficiently retrieve and store relevant information across interaction sessions for suitable responses. Cognitive AI, which aims to simulate the human thought process in a computerized model, highlights interesting aspects, such as thoughts, memory mechanisms, and decision-making, that can contribute towards improved memory modeling for LLMs. Inspired by these cognitive AI principles, we propose our memory framework CAIM. CAIM consists of three modules: 1.) The Memory Controller as the central decision unit; 2.) the Memory Retrieval, which filters relevant data for interaction upon request; and 3.) the Post-Thinking, which maintains the memory storage. We compare CAIM against existing approaches, focusing on metrics such as retrieval accuracy, response correctness, contextual coherence, and memory storage. The results demonstrate that CAIM outperforms baseline frameworks across different metrics, highlighting its context-awareness and potential to improve long-term human-AI interactions. 

**Abstract (ZH)**: 大型语言模型（LLMs）推动了人工智能（AI）领域的发展，并成为交互系统强有力的支撑。然而，它们在长期交互中仍面临挑战，需要适应用户并理解不断变化的环境。为克服这些挑战，需要进行全面的记忆建模，以便在交互会话中高效检索和存储相关信息，以适当地作出响应。认知AI旨在通过计算机模型模拟人类思维过程，突出了诸如思考、记忆机制和决策等有趣方面，这些方面能够为LLMs的记忆建模提供改进。受这些认知AI原理的启发，我们提出了我们的记忆框架CAIM。CAIM包含三个模块：1）记忆控制器作为中央决策单元；2）记忆检索，根据请求过滤相关数据进行交互；3）后思维过程，维持记忆存储。我们将CAIM与现有方法进行比较，重点关注检索准确性、响应正确性、上下文连贯性和记忆存储等指标。结果表明，CAIM在不同指标上均优于基线框架，体现出其环境意识和改善长期人机交互的潜力。 

---
# MindOmni: Unleashing Reasoning Generation in Vision Language Models with RGPO 

**Title (ZH)**: MindOmni: 在视觉语言模型中通过RGPO释放推理生成能力 

**Authors**: Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13031)  

**Abstract**: Recent text-to-image systems face limitations in handling multimodal inputs and complex reasoning tasks. We introduce MindOmni, a unified multimodal large language model that addresses these challenges by incorporating reasoning generation through reinforcement learning. MindOmni leverages a three-phase training strategy: i) design of a unified vision language model with a decoder-only diffusion module, ii) supervised fine-tuning with Chain-of-Thought (CoT) instruction data, and iii) our proposed Reasoning Generation Policy Optimization (RGPO) algorithm, utilizing multimodal feedback to effectively guide policy updates. Experimental results demonstrate that MindOmni outperforms existing models, achieving impressive performance on both understanding and generation benchmarks, meanwhile showcasing advanced fine-grained reasoning generation capabilities, especially with mathematical reasoning instruction. All codes will be made public at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: Recent text-to-image系统在处理多模态输入和复杂推理任务时存在局限性。我们介绍了MindOmni，这是一种通过强化学习集成推理生成的统一多模态大型语言模型。MindOmni采用了三阶段训练策略：i) 设计一个统一的 Vision-Language 模型，包含解码器导向扩散模块；ii) 使用带有CoT指令的数据进行监督微调；iii) 我们提出的推理生成策略优化（RGPO）算法，利用多模态反馈有效指导策略更新。实验结果表明，MindOmni在理解与生成基准测试中优于现有模型，同时展现了先进的细粒度推理生成能力，特别是在数学推理指令方面。所有代码将在 \href{this https URL}{this https URL} 公开。 

---
# Unveiling and Steering Connectome Organization with Interpretable Latent Variables 

**Title (ZH)**: 揭示并引导连接组组织的可解释潜在变量 

**Authors**: Yubin Li, Xingyu Liu, Guozhang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13011)  

**Abstract**: The brain's intricate connectome, a blueprint for its function, presents immense complexity, yet it arises from a compact genetic code, hinting at underlying low-dimensional organizational principles. This work bridges connectomics and representation learning to uncover these principles. We propose a framework that combines subgraph extraction from the Drosophila connectome, FlyWire, with a generative model to derive interpretable low-dimensional representations of neural circuitry. Crucially, an explainability module links these latent dimensions to specific structural features, offering insights into their functional relevance. We validate our approach by demonstrating effective graph reconstruction and, significantly, the ability to manipulate these latent codes to controllably generate connectome subgraphs with predefined properties. This research offers a novel tool for understanding brain architecture and a potential avenue for designing bio-inspired artificial neural networks. 

**Abstract (ZH)**: 脑连接组的精细网络图谱揭示了其功能的复杂性，源自紧凑的遗传代码，暗示着潜在的低维度组织原则。本研究将连接组学与表示学习相结合，以揭示这些原则。我们提出了一种框架，该框架结合了从果蝇连接组FlyWire中提取子图，并使用生成模型来推导可解释的低维度神经环路表征。关键的是，可解释性模块将这些潜在维度与特定的结构特征联系起来，提供其功能相关性的见解。通过展示有效的图形重建以及有目的地生成具有预定义属性的连接组子图的能力，我们验证了这种方法。这项研究提供了一种理解大脑架构的新工具，并为设计生物启发式人工神经网络开辟了潜在途径。 

---
# The Traitors: Deception and Trust in Multi-Agent Language Model Simulations 

**Title (ZH)**: 叛徒：多Agent语言模型仿真中的欺骗与信任 

**Authors**: Pedro M. P. Curvo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12923)  

**Abstract**: As AI systems increasingly assume roles where trust and alignment with human values are essential, understanding when and why they engage in deception has become a critical research priority. We introduce The Traitors, a multi-agent simulation framework inspired by social deduction games, designed to probe deception, trust formation, and strategic communication among large language model (LLM) agents under asymmetric information. A minority of agents the traitors seek to mislead the majority, while the faithful must infer hidden identities through dialogue and reasoning. Our contributions are: (1) we ground the environment in formal frameworks from game theory, behavioral economics, and social cognition; (2) we develop a suite of evaluation metrics capturing deception success, trust dynamics, and collective inference quality; (3) we implement a fully autonomous simulation platform where LLMs reason over persistent memory and evolving social dynamics, with support for heterogeneous agent populations, specialized traits, and adaptive behaviors. Our initial experiments across DeepSeek-V3, GPT-4o-mini, and GPT-4o (10 runs per model) reveal a notable asymmetry: advanced models like GPT-4o demonstrate superior deceptive capabilities yet exhibit disproportionate vulnerability to others' falsehoods. This suggests deception skills may scale faster than detection abilities. Overall, The Traitors provides a focused, configurable testbed for investigating LLM behavior in socially nuanced interactions. We position this work as a contribution toward more rigorous research on deception mechanisms, alignment challenges, and the broader social reliability of AI systems. 

**Abstract (ZH)**: 随着AI系统在需要信任和与人类价值观对齐的领域中扮演越来越重要的角色，了解何时以及为何它们进行欺骗已成为一项至关重要的研究优先事项。我们介绍了《叛徒》，一个受社交推理游戏启发的多agent仿真框架，旨在探究在不对称信息条件下大语言模型（LLM）agent之间的欺骗、信任形成和战略沟通。少数“叛徒”试图误导多数人，而“忠诚者”则需要通过对话和推理来推断隐藏的身份。我们的贡献包括：（1）我们将环境建立在博弈论、行为经济学和社会认知的正式框架之上；（2）我们开发了一套评估指标，用于捕捉欺骗成功率、信任动态和集体推理质量；（3）我们实现了一个完全自主的仿真平台，其中LLM能够在持续记忆和不断变化的社会动态背景下进行推理，并支持异质性agent群体、专门技能和适应性行为。我们在DeepSeek-V3、GPT-4o-mini和GPT-4o（每种模型运行10次）上的初始实验揭示了一个显著的不对称性：如GPT-4o这样的先进模型展示了卓越的欺骗能力，但对其他人的虚假信息表现出不成比例的脆弱性。这表明欺骗技能的增长可能比检测技能更快。总体而言，《叛徒》提供了一个专注于社会细微互动中LLM行为的研究平台。我们将这项工作定位为在欺骗机制研究、对齐挑战研究及其更广泛的AI系统社会可靠性方面更具严谨性研究的贡献。 

---
# TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios 

**Title (ZH)**: TIME：一种面向真实场景的LLM时间推理多层级基准 

**Authors**: Shaohang Wei, Wei Li, Feifan Song, Wen Luo, Tianyi Zhuang, Haochen Tan, Zhijiang Guo, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12891)  

**Abstract**: Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at this https URL , and the dataset is available at this https URL . 

**Abstract (ZH)**: 时空推理对于大规模语言模型（LLMs）理解现实世界至关重要。然而，现有工作忽略了时空推理的实际挑战：（1）密集的时空信息，（2）快速变化的事件动态，以及（3）社会互动中的复杂时空依赖关系。为了弥合这一差距，我们提出了一个多层基准TIME，用于现实世界场景中的时空推理。TIME包含38,522个问答对，涵盖3个层次并包含11个细粒度子任务。该基准数据集包括3个子数据集，分别反映不同的现实世界挑战：TIME-Wiki、TIME-News和TIME-Dial。我们在推理模型和非推理模型上进行了广泛的实验，并对时空推理性能在各种现实世界场景和任务中的表现进行了深入分析，总结了测试时扩展对时空推理能力的影响。此外，我们发布了TIME-Lite，这是一个手工标注的子集，旨在促进未来在时空推理领域的研究和标准化评估。相关代码可通过以下链接获取：this https URL，数据集可通过以下链接获取：this https URL。 

---
# Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective 

**Title (ZH)**: 大型推理模型中的幻觉检测与缓解：一种机理视角 

**Authors**: Zhongxiang Sun, Qipeng Wang, Haoyu Wang, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12886)  

**Abstract**: Large Reasoning Models (LRMs) have shown impressive capabilities in multi-step reasoning tasks. However, alongside these successes, a more deceptive form of model error has emerged--Reasoning Hallucination--where logically coherent but factually incorrect reasoning traces lead to persuasive yet faulty conclusions. Unlike traditional hallucinations, these errors are embedded within structured reasoning, making them more difficult to detect and potentially more harmful. In this work, we investigate reasoning hallucinations from a mechanistic perspective. We propose the Reasoning Score, which quantifies the depth of reasoning by measuring the divergence between logits obtained from projecting late layers of LRMs to the vocabulary space, effectively distinguishing shallow pattern-matching from genuine deep reasoning. Using this score, we conduct an in-depth analysis on the ReTruthQA dataset and identify two key reasoning hallucination patterns: early-stage fluctuation in reasoning depth and incorrect backtracking to flawed prior steps. These insights motivate our Reasoning Hallucination Detection (RHD) framework, which achieves state-of-the-art performance across multiple domains. To mitigate reasoning hallucinations, we further introduce GRPO-R, an enhanced reinforcement learning algorithm that incorporates step-level deep reasoning rewards via potential-based shaping. Our theoretical analysis establishes stronger generalization guarantees, and experiments demonstrate improved reasoning quality and reduced hallucination rates. 

**Abstract (ZH)**: 大型推理模型中的推理幻觉：机制分析与检测 

---
# From Grunts to Grammar: Emergent Language from Cooperative Foraging 

**Title (ZH)**: 从 grunt 到 grammar：合作觅食中的 Emergent 语言 

**Authors**: Maytus Piriyajitakonkij, Rujikorn Charakorn, Weicheng Tao, Wei Pan, Mingfei Sun, Cheston Tan, Mengmi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12872)  

**Abstract**: Early cavemen relied on gestures, vocalizations, and simple signals to coordinate, plan, avoid predators, and share resources. Today, humans collaborate using complex languages to achieve remarkable results. What drives this evolution in communication? How does language emerge, adapt, and become vital for teamwork? Understanding the origins of language remains a challenge. A leading hypothesis in linguistics and anthropology posits that language evolved to meet the ecological and social demands of early human cooperation. Language did not arise in isolation, but through shared survival goals. Inspired by this view, we investigate the emergence of language in multi-agent Foraging Games. These environments are designed to reflect the cognitive and ecological constraints believed to have influenced the evolution of communication. Agents operate in a shared grid world with only partial knowledge about other agents and the environment, and must coordinate to complete games like picking up high-value targets or executing temporally ordered actions. Using end-to-end deep reinforcement learning, agents learn both actions and communication strategies from scratch. We find that agents develop communication protocols with hallmark features of natural language: arbitrariness, interchangeability, displacement, cultural transmission, and compositionality. We quantify each property and analyze how different factors, such as population size and temporal dependencies, shape specific aspects of the emergent language. Our framework serves as a platform for studying how language can evolve from partial observability, temporal reasoning, and cooperative goals in embodied multi-agent settings. We will release all data, code, and models publicly. 

**Abstract (ZH)**: 早期智人依靠手势、语音和简单信号来进行协调、计划、逃避捕食者和分享资源。今天，人类使用复杂的语言协作以取得非凡成果。是什么推动了这种交流方式的演变？语言是如何产生、适应并成为团队合作的关键的？语言起源的理解仍然是一个挑战。语言学和人类学中的一个主要假说是，语言进化是为了满足早期人类合作面临的生态和社会需求。语言并非孤立产生，而是通过共享的生存目标而共同演化。受这一观点的启发，我们研究了多智能体采集游戏中的语言涌现。这些环境旨在反映据信影响交流进化的认知和生态限制。智能体在一个共享的网格世界中操作，只能部分了解其他智能体和环境，并必须协调完成类似于拾取高价值目标或执行时序动作的游戏。使用端到端的深度强化学习，智能体从零开始学习动作和通信策略。我们发现智能体发展出具有自然语言标志性特征的通信协议：任意性、可替换性、延期性、文化传播性以及组合性。我们量化了每种特性，并分析了不同因素，如种群规模和时序依赖性，如何塑造涌现语言的具体方面。我们的框架提供了一个研究平台，探讨在部分可观测性、时间推理和合作目标下，语言如何在具身多智能体环境中演变出来。我们将公开发布所有数据、代码和模型。 

---
# Multi-Level Aware Preference Learning: Enhancing RLHF for Complex Multi-Instruction Tasks 

**Title (ZH)**: 多层级感知偏好学习：增强复杂多指令任务的RLHF 

**Authors**: Ruopei Sun, Jianfeng Cai, Jinhua Zhu, Kangwen Zhao, Dongyun Xue, Wengang Zhou, Li Li, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12845)  

**Abstract**: RLHF has emerged as a predominant approach for aligning artificial intelligence systems with human preferences, demonstrating exceptional and measurable efficacy in instruction following tasks; however, it exhibits insufficient compliance capabilities when confronted with complex multi-instruction tasks. Conventional approaches rely heavily on human annotation or more sophisticated large language models, thereby introducing substantial resource expenditure or potential bias concerns. Meanwhile, alternative synthetic methods that augment standard preference datasets often compromise the model's semantic quality. Our research identifies a critical oversight in existing techniques, which predominantly focus on comparing responses while neglecting valuable latent signals embedded within prompt inputs, and which only focus on preference disparities at the intra-sample level, while neglecting to account for the inter-sample level preference differentials that exist among preference data. To leverage these previously neglected indicators, we propose a novel Multi-level Aware Preference Learning (MAPL) framework, capable of enhancing multi-instruction capabilities. Specifically, for any given response in original preference data pairs, we construct varied prompts with a preference relation under different conditions, in order to learn intra-sample level preference disparities. Furthermore, for any given original preference pair, we synthesize multi-instruction preference pairs to capture preference discrepancies at the inter-sample level. Building on the two datasets constructed above, we consequently devise two sophisticated training objective functions. Subsequently, our framework integrates seamlessly into both Reward Modeling and Direct Preference Optimization paradigms. Through rigorous evaluation across multiple benchmarks, we empirically validate the efficacy of our framework. 

**Abstract (ZH)**: 基于多级感知的偏好学习框架（MAPL）：提升多指令处理能力 

---
# AGI-Elo: How Far Are We From Mastering A Task? 

**Title (ZH)**: AGI-Elo:我们离掌握一项任务还差多远？ 

**Authors**: Shuo Sun, Yimin Zhao, Christina Dao Wen Lee, Jiawei Sun, Chengran Yuan, Zefan Huang, Dongen Li, Justin KW Yeoh, Alok Prakash, Thomas W. Malone, Marcelo H. Ang Jr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12844)  

**Abstract**: As the field progresses toward Artificial General Intelligence (AGI), there is a pressing need for more comprehensive and insightful evaluation frameworks that go beyond aggregate performance metrics. This paper introduces a unified rating system that jointly models the difficulty of individual test cases and the competency of AI models (or humans) across vision, language, and action domains. Unlike existing metrics that focus solely on models, our approach allows for fine-grained, difficulty-aware evaluations through competitive interactions between models and tasks, capturing both the long-tail distribution of real-world challenges and the competency gap between current models and full task mastery. We validate the generalizability and robustness of our system through extensive experiments on multiple established datasets and models across distinct AGI domains. The resulting rating distributions offer novel perspectives and interpretable insights into task difficulty, model progression, and the outstanding challenges that remain on the path to achieving full AGI task mastery. 

**Abstract (ZH)**: 随着人工智能领域向通用人工智能（AGI）的发展，需要更加全面和深入的评估框架，而不仅仅依赖聚合性能指标。本文介绍了一种统一的评级系统，该系统联合建模了Individual测试案例的难度以及AI模型（或人类）在视觉、语言和行动领域的能力。与仅关注模型的现有指标不同，我们的方法通过模型与任务之间的竞争性互动，进行细粒度和难度感知的评估，捕捉到现实世界挑战的长尾分布以及当前模型与全面任务掌握之间的能力差距。我们通过在多个不同的AGI领域建立的数据集和模型上进行广泛的实验，验证了该系统的泛化能力和鲁棒性。结果得到的评级分布提供了有关任务难度、模型进展以及通往完整AGI任务掌握过程中仍需解决的突出挑战的新颖视角和可解释见解。 

---
# Reasoning BO: Enhancing Bayesian Optimization with Long-Context Reasoning Power of LLMs 

**Title (ZH)**: LLM长上下文推理增强的Bayesian优化 

**Authors**: Zhuo Yang, Lingli Ge, Dong Han, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12833)  

**Abstract**: Many real-world scientific and industrial applications require the optimization of expensive black-box functions. Bayesian Optimization (BO) provides an effective framework for such problems. However, traditional BO methods are prone to get trapped in local optima and often lack interpretable insights. To address this issue, this paper designs Reasoning BO, a novel framework that leverages reasoning models to guide the sampling process in BO while incorporating multi-agent systems and knowledge graphs for online knowledge accumulation. By integrating the reasoning and contextual understanding capabilities of Large Language Models (LLMs), we can provide strong guidance to enhance the BO process. As the optimization progresses, Reasoning BO provides real-time sampling recommendations along with critical insights grounded in plausible scientific theories, aiding in the discovery of superior solutions within the search space. We systematically evaluate our approach across 10 diverse tasks encompassing synthetic mathematical functions and complex real-world applications. The framework demonstrates its capability to progressively refine sampling strategies through real-time insights and hypothesis evolution, effectively identifying higher-performing regions of the search space for focused exploration. This process highlights the powerful reasoning and context-learning abilities of LLMs in optimization scenarios. For example, in the Direct Arylation task, our method increased the yield to 60.7%, whereas traditional BO achieved only a 25.2% yield. Furthermore, our investigation reveals that smaller LLMs, when fine-tuned through reinforcement learning, can attain comparable performance to their larger counterparts. This enhanced reasoning capability paves the way for more efficient automated scientific experimentation while maintaining computational feasibility. 

**Abstract (ZH)**: 多实例学习的贝叶斯优化：一种基于推理的框架 

---
# Emergent Specialization: Rare Token Neurons in Language Models 

**Title (ZH)**: emergent specialization: 语言模型中的 Rarity Token 神经元 

**Authors**: Jing Liu, Haozheng Wang, Yueheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12822)  

**Abstract**: Large language models struggle with representing and generating rare tokens despite their importance in specialized domains. In this study, we identify neuron structures with exceptionally strong influence on language model's prediction of rare tokens, termed as rare token neurons, and investigate the mechanism for their emergence and behavior. These neurons exhibit a characteristic three-phase organization (plateau, power-law, and rapid decay) that emerges dynamically during training, evolving from a homogeneous initial state to a functionally differentiated architecture. In the activation space, rare token neurons form a coordinated subnetwork that selectively co-activates while avoiding co-activation with other neurons. This functional specialization potentially correlates with the development of heavy-tailed weight distributions, suggesting a statistical mechanical basis for emergent specialization. 

**Abstract (ZH)**: 大型语言模型在表示和生成稀有令牌方面存在困难，尽管这些令牌在专门领域中非常重要。在本研究中，我们识别出对语言模型预测稀有令牌具有异常强大影响的神经元结构，称为稀有令牌神经元，并探讨其产生机制和行为。这些神经元展示出一种动态出现的三阶段组织特征（平台期、幂律分布期和快速衰减期），从初始的同质状态进化为功能分化结构。在激活空间中，稀有令牌神经元形成一个协调子网络，在激活时选择性地协同激活而不与其他神经元发生共激活。这种功能专业化可能与重尾权重分布的发展相关，表明可能存在统计力学基础的自发专业化机制。 

---
# FRAbench and GenEval: Scaling Fine-Grained Aspect Evaluation across Tasks, Modalities 

**Title (ZH)**: FRAbench 和 GenEval：跨任务和模态的细粒度方面评价扩展 

**Authors**: Shibo Hong, Jiahao Ying, Haiyuan Liang, Mengdi Zhang, Jun Kuang, Jiazheng Zhang, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12795)  

**Abstract**: Evaluating the open-ended outputs of large language models (LLMs) has become a bottleneck as model capabilities, task diversity, and modality coverage rapidly expand. Existing "LLM-as-a-Judge" evaluators are typically narrow in a few tasks, aspects, or modalities, and easily suffer from low consistency. In this paper, we argue that explicit, fine-grained aspect specification is the key to both generalizability and objectivity in automated evaluation. To do so, we introduce a hierarchical aspect taxonomy spanning 112 aspects that unifies evaluation across four representative settings - Natural Language Generation, Image Understanding, Image Generation, and Interleaved Text-and-Image Generation. Building on this taxonomy, we create FRAbench, a benchmark comprising 60.4k pairwise samples with 325k aspect-level labels obtained from a combination of human and LLM annotations. FRAbench provides the first large-scale, multi-modal resource for training and meta-evaluating fine-grained LMM judges. Leveraging FRAbench, we develop GenEval, a fine-grained evaluator generalizable across tasks and modalities. Experiments show that GenEval (i) attains high agreement with GPT-4o and expert annotators, (ii) transfers robustly to unseen tasks and modalities, and (iii) reveals systematic weaknesses of current LMMs on evaluation. 

**Abstract (ZH)**: 评估大型语言模型的开放式输出已成为瓶颈，随着模型能力、任务多样性和模态覆盖范围的迅速扩展。现有的“作为评判者的大型语言模型”评估者通常在少数任务、方面或模态上狭窄，且容易遭受一致性低的影响。在本文中，我们argueExplicit、细粒度方面的明确指定是自动评估中广泛适用性和客观性的关键。为此，我们引入了一个涵盖112个方面的层次结构方面的分类系统，该系统统一了四个代表性设置——自然语言生成、图像理解、图像生成和交替的文本和图像生成的评估。基于这一分类系统，我们创建了FRAbench，这是一个包含60,400对样本的基准，共有325,000个方面的标签，这些标签是通过结合人力和LLM标注获得的。FRAbench提供了首个大型、多模态资源，用于训练和元评估细粒度的LMM评判者。利用FRAbench，我们开发了GenEval，这是一种适用于跨任务和模态的细粒度评估器。实验显示，GenEval (i) 与GPT-4o和专家标注者高度一致，(ii) 能够稳健地转移到未见过的任务和模态上，以及(iii) 揭示了当前LMMs在评估中的系统性弱点。 

---
# Mixture Policy based Multi-Hop Reasoning over N-tuple Temporal Knowledge Graphs 

**Title (ZH)**: 基于混合策略的多跳推理研究-temporal N-元知识图谱 

**Authors**: Zhongni Hou, Miao Su, Xiaolong Jin, Zixuan Li, Long Bai, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12788)  

**Abstract**: Temporal Knowledge Graphs (TKGs), which utilize quadruples in the form of (subject, predicate, object, timestamp) to describe temporal facts, have attracted extensive attention. N-tuple TKGs (N-TKGs) further extend traditional TKGs by utilizing n-tuples to incorporate auxiliary elements alongside core elements (i.e., subject, predicate, and object) of facts, so as to represent them in a more fine-grained manner. Reasoning over N-TKGs aims to predict potential future facts based on historical ones. However, existing N-TKG reasoning methods often lack explainability due to their black-box nature. Therefore, we introduce a new Reinforcement Learning-based method, named MT-Path, which leverages the temporal information to traverse historical n-tuples and construct a temporal reasoning path. Specifically, in order to integrate the information encapsulated within n-tuples, i.e., the entity-irrelevant information within the predicate, the information about core elements, and the complete information about the entire n-tuples, MT-Path utilizes a mixture policy-driven action selector, which bases on three low-level policies, namely, the predicate-focused policy, the core-element-focused policy and the whole-fact-focused policy. Further, MT-Path utilizes an auxiliary element-aware GCN to capture the rich semantic dependencies among facts, thereby enabling the agent to gain a deep understanding of each n-tuple. Experimental results demonstrate the effectiveness and the explainability of MT-Path. 

**Abstract (ZH)**: 基于强化学习的MT-Path：利用时间信息构建历史n元组的时间推理路径 

---
# Language Models That Walk the Talk: A Framework for Formal Fairness Certificates 

**Title (ZH)**: 语言模型不仅要会话还要身体力行：正式公平性证书的框架 

**Authors**: Danqing Chen, Tobias Ladner, Ahmed Rayen Mhadhbi, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2505.12767)  

**Abstract**: As large language models become integral to high-stakes applications, ensuring their robustness and fairness is critical. Despite their success, large language models remain vulnerable to adversarial attacks, where small perturbations, such as synonym substitutions, can alter model predictions, posing risks in fairness-critical areas, such as gender bias mitigation, and safety-critical areas, such as toxicity detection. While formal verification has been explored for neural networks, its application to large language models remains limited. This work presents a holistic verification framework to certify the robustness of transformer-based language models, with a focus on ensuring gender fairness and consistent outputs across different gender-related terms. Furthermore, we extend this methodology to toxicity detection, offering formal guarantees that adversarially manipulated toxic inputs are consistently detected and appropriately censored, thereby ensuring the reliability of moderation systems. By formalizing robustness within the embedding space, this work strengthens the reliability of language models in ethical AI deployment and content moderation. 

**Abstract (ZH)**: 随着大型语言模型在高风险应用中的作用日益重要，确保其稳健性和公平性至关重要。尽管取得了成功，但大型语言模型仍易受到对抗性攻击的威胁，如同义词替换等微小的扰动均可改变模型预测，这在性别偏见缓解和毒性检测等关键领域带来了风险。虽然对神经网络已进行了形式化验证探索，但其在大型语言模型中的应用仍受到限制。本文提出了一种整体验证框架，以验证基于变换器的语言模型的稳健性，重点关注性别公平性和不同性别相关术语的一致输出。此外，我们还将该方法扩展到毒性检测，提供了形式化的保证，即对抗性操纵的有毒输入能够一致地被检测并适当过滤，从而确保内容审核系统的可靠性。通过在嵌入空间内形式化稳健性，本文增强了在伦理人工智能部署和内容审核中语言模型的可靠性。 

---
# IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment 

**Title (ZH)**: IDEAL：数据均衡适应多能力语言模型对齐 

**Authors**: Chenlin Ming, Chendi Qu, Mengzhang Cai, Qizhi Pei, Zhuoshi Pan, Yu Li, Xiaoming Duan, Lijun Wu, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2505.12762)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance through Supervised Fine-tuning (SFT) on diverse instructional datasets. When training on multiple capabilities simultaneously, the mixture training dataset, governed by volumes of data from different domains, is a critical factor that directly impacts the final model's performance. Unlike many studies that focus on enhancing the quality of training datasets through data selection methods, few works explore the intricate relationship between the compositional quantity of mixture training datasets and the emergent capabilities of LLMs. Given the availability of a high-quality multi-domain training dataset, understanding the impact of data from each domain on the model's overall capabilities is crucial for preparing SFT data and training a well-balanced model that performs effectively across diverse domains. In this work, we introduce IDEAL, an innovative data equilibrium adaptation framework designed to effectively optimize volumes of data from different domains within mixture SFT datasets, thereby enhancing the model's alignment and performance across multiple capabilities. IDEAL employs a gradient-based approach to iteratively refine the training data distribution, dynamically adjusting the volumes of domain-specific data based on their impact on downstream task performance. By leveraging this adaptive mechanism, IDEAL ensures a balanced dataset composition, enabling the model to achieve robust generalization and consistent proficiency across diverse tasks. Experiments across different capabilities demonstrate that IDEAL outperforms conventional uniform data allocation strategies, achieving a comprehensive improvement of approximately 7% in multi-task evaluation scores. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过多样化指令数据集的监督微调（SFT）取得了 impressive 的性能。当同时训练多种能力时，由不同领域大量数据组成的混合训练数据集是直接影响最终模型性能的关键因素。与许多专注于通过数据选择方法提升训练数据质量的研究不同，很少有工作探索混合训练数据集组成数量与LLMs新兴能力之间的复杂关系。鉴于高质量多领域训练数据集的可用性，理解来自每个领域的数据对模型整体能力的影响对于准备SFT数据和训练一种在各种领域中表现良好的平衡模型至关重要。在本项工作中，我们引入了IDEAL，一种创新的数据均衡适应框架，旨在有效优化混合SFT数据集中不同领域数据的量，从而增强模型在多个能力上的对齐度和性能。IDEAL采用基于梯度的方法，迭代细化训练数据分布，根据其对下游任务性能的影响动态调整特定领域数据的量。通过利用这一适应机制，IDEAL确保数据集组成平衡，使模型在多种任务中实现稳健泛化和一致的专业技能。不同能力的实验表明，与传统均匀数据分配策略相比，IDEAL在多任务评估得分上取得了全面改进，约提高了7%。 

---
# Correspondence of high-dimensional emotion structures elicited by video clips between humans and Multimodal LLMs 

**Title (ZH)**: 由视频片段引发的高维情绪结构在人类和多模态LLM之间的对应关系 

**Authors**: Haruka Asanuma, Naoko Koide-Majima, Ken Nakamura, Takato Horii, Shinji Nishimoto, Masafumi Oizumi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12746)  

**Abstract**: Recent studies have revealed that human emotions exhibit a high-dimensional, complex structure. A full capturing of this complexity requires new approaches, as conventional models that disregard high dimensionality risk overlooking key nuances of human emotions. Here, we examined the extent to which the latest generation of rapidly evolving Multimodal Large Language Models (MLLMs) capture these high-dimensional, intricate emotion structures, including capabilities and limitations. Specifically, we compared self-reported emotion ratings from participants watching videos with model-generated estimates (e.g., Gemini or GPT). We evaluated performance not only at the individual video level but also from emotion structures that account for inter-video relationships. At the level of simple correlation between emotion structures, our results demonstrated strong similarity between human and model-inferred emotion structures. To further explore whether the similarity between humans and models is at the signle item level or the coarse-categorical level, we applied Gromov Wasserstein Optimal Transport. We found that although performance was not necessarily high at the strict, single-item level, performance across video categories that elicit similar emotions was substantial, indicating that the model could infer human emotional experiences at the category level. Our results suggest that current state-of-the-art MLLMs broadly capture the complex high-dimensional emotion structures at the category level, as well as their apparent limitations in accurately capturing entire structures at the single-item level. 

**Abstract (ZH)**: 最近的研究揭示了人类情绪具有高维度、复杂的结构。全面捕捉这种复杂性需要新的方法，因为忽略高维度的传统模型可能会遗漏人类情绪的关键微妙之处。本研究考察了最新一代快速演化的多模态大型语言模型（MLLMs）在捕捉这些高维度、复杂情绪结构方面的能力及其局限性，包括具体的能力和局限。我们比较了参与者观看视频时自我报告的情绪评分与模型生成的估计值（如Gemini或GPT）。我们不仅在单个视频层面评估性能，还在考虑到视频间关系的情绪结构层面进行评估。在情绪结构简单相关性的层面，我们的结果显示人类和模型推断的情绪结构之间存在强烈的相似性。为进一步探索人类和模型之间的相似性是出现在单一项目层面还是粗糙类别层面，我们应用了Gromov Wasserstein最优传输。我们发现虽然在严格的单一项目层面表现未必很高，但在引起相似情绪的视频类别层面的表现却是显著的，表明模型可以在类别层面推断人类的情绪体验。我们的研究结果表明，当前最先进的MLLMs在类别层面广泛地捕捉了复杂高维度的情绪结构及其在单一项目层面准确捕捉整个结构的明显局限性。 

---
# Incentivizing Multimodal Reasoning in Large Models for Direct Robot Manipulation 

**Title (ZH)**: 激励大规模模型进行多模态推理以实现直接机器人操作 

**Authors**: Weiliang Tang, Dong Jing, Jia-Hui Pan, Zhiwu Lu, Yun-Hui Liu, Li Erran Li, Mingyu Ding, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12744)  

**Abstract**: Recent Large Multimodal Models have demonstrated remarkable reasoning capabilities, especially in solving complex mathematical problems and realizing accurate spatial perception. Our key insight is that these emerging abilities can naturally extend to robotic manipulation by enabling LMMs to directly infer the next goal in language via reasoning, rather than relying on a separate action head. However, this paradigm meets two main challenges: i) How to make LMMs understand the spatial action space, and ii) How to fully exploit the reasoning capacity of LMMs in solving these tasks. To tackle the former challenge, we propose a novel task formulation, which inputs the current states of object parts and the gripper, and reformulates rotation by a new axis representation instead of traditional Euler angles. This representation is more compatible with spatial reasoning and easier to interpret within a unified language space. For the latter challenge, we design a pipeline to utilize cutting-edge LMMs to generate a small but high-quality reasoning dataset of multi-round dialogues that successfully solve manipulation tasks for supervised fine-tuning. Then, we perform reinforcement learning by trial-and-error interactions in simulation to further enhance the model's reasoning abilities for robotic manipulation. Our resulting reasoning model built upon a 7B backbone, named ReasonManip, demonstrates three notable advantages driven by its system-2 level reasoning capabilities: i) exceptional generalizability to out-of-distribution environments, objects, and tasks; ii) inherent sim-to-real transfer ability enabled by the unified language representation shared across domains; iii) transparent interpretability connecting high-level reasoning and low-level control. Extensive experiments demonstrate the effectiveness of the proposed paradigm and its potential to advance LMM-driven robotic manipulation. 

**Abstract (ZH)**: 近期的大规模多模态模型展示了非凡的推理能力，特别是在解决复杂的数学问题和实现精确的空间感知方面。我们的核心洞察是，这些新兴的能力可以自然地扩展到机器人操作，使LMM能够直接通过对语言的推理来推断下一个目标，而无需依赖于独立的动作头。然而，这一范式面临两个主要挑战：i) 如何使LMM理解空间动作空间，ii) 如何充分利用LMM的推理能力来解决这些任务。为应对第一个挑战，我们提出了一种新的任务形式化，输入当前物体部分和 gripper 的状态，并通过新的轴表示重新定义旋转，而不是传统的欧拉角。该表示与空间推理更加兼容，并且更容易在统一的语言空间内进行解释。为应对第二个挑战，我们设计了一个管道，利用先进的LMM生成一个高质量的多轮对话推理数据集，该数据集成功解决了操作任务，并用于监督微调。然后，我们在模拟环境中通过试错交互进行强化学习，进一步增强模型的推理能力以适应机器人操作。基于7B参数量构建的推理模型名为ReasonManip，表现出系统2级推理能力下的三大显著优势：i) 优越的分布外环境、物体和任务的泛化能力；ii) 由跨领域共享的统一语言表示带来的内在模拟到现实的转移能力；iii) 透明的可解释性，将高级推理与低级控制连接起来。大量实验表明所提出范式的有效性及其在推动LMM驱动的机器人操作方面的潜在能力。 

---
# Dense Communication between Language Models 

**Title (ZH)**: 语言模型之间的密集通信 

**Authors**: Shiguang Wu, Yaqing Wang, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12741)  

**Abstract**: As higher-level intelligence emerges from the combination of modular components with lower-level intelligence, many works combines Large Language Models (LLMs) for collective intelligence. Such combination is achieved by building communications among LLMs. While current systems primarily facilitate such communication through natural language, this paper proposes a novel paradigm of direct dense vector communication between LLMs. Our approach eliminates the unnecessary embedding and de-embedding steps when LLM interact with another, enabling more efficient information transfer, fully differentiable optimization pathways, and exploration of capabilities beyond human heuristics. We use such stripped LLMs as vertexes and optimizable seq2seq modules as edges to construct LMNet, with similar structure as MLPs. By utilizing smaller pre-trained LLMs as vertexes, we train a LMNet that achieves comparable performance with LLMs in similar size with only less than 0.1% training cost. This offers a new perspective on scaling for general intelligence rather than training a monolithic LLM from scratch. Besides, the proposed method can be used for other applications, like customizing LLM with limited data, showing its versatility. 

**Abstract (ZH)**: 更高层次智能从低层次智能的模块化组件结合中 emerge，许多研究结合大型语言模型（LLMs）以实现集体智能。这种结合通过在LLMs之间构建通信来实现。虽然当前系统主要通过自然语言促进这种通信，本文提出了一种新的LLM之间直接密集向量通信范式。我们的方法在LLM相互交互时消除了不必要的嵌入和反嵌入步骤，从而使信息传递更高效，优化路径完全可微，且能够探索超越人类启发式的方法。我们使用这样的精简LLM作为节点，可优化的序列到序列模块作为边，构建LMNet，其结构类似于MLP。通过使用较小的预训练LLM作为节点，我们训练了一个LMNet，其性能与相似大小的LLM相当，训练成本仅少于0.1%。这为通用智能的扩展提供了一个新的视角，而不是从头训练一个巨量的LLM。此外，所提出的方法还可以用于其他应用，如有限数据下的LLM定制，显示出其灵活性。 

---
# Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps 

**Title (ZH)**: 基于指令驱动的检索重叠表示减少以加速自适应检索增强生成 

**Authors**: Jie Ou, Jinyu Guo, Shuaihong Jiang, Zhaokun Wang, Libo Qin, Shunyu Yao, Wenhong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.12731)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a pivotal method for expanding the knowledge of large language models. To handle complex queries more effectively, researchers developed Adaptive-RAG (A-RAG) to enhance the generated quality through multiple interactions with external knowledge bases. Despite its effectiveness, A-RAG exacerbates the pre-existing efficiency challenges inherent in RAG, which are attributable to its reliance on multiple iterations of generation. Existing A-RAG approaches process all retrieved contents from scratch. However, they ignore the situation where there is a significant overlap in the content of the retrieval results across rounds. The overlapping content is redundantly represented, which leads to a large proportion of repeated computations, thus affecting the overall efficiency. To address this issue, this paper introduces a model-agnostic approach that can be generally applied to A-RAG methods, which is dedicated to reducing the redundant representation process caused by the overlapping of retrieval results. Specifically, we use cache access and parallel generation to speed up the prefilling and decoding stages respectively. Additionally, we also propose an instruction-driven module to further guide the model to more effectively attend to each part of the content in a more suitable way for LLMs. Experiments show that our approach achieves 2.79 and 2.33 times significant acceleration on average for prefilling and decoding respectively while maintaining equal generation quality. 

**Abstract (ZH)**: 基于检索的生成增强（RAG）方法已成为扩展大型语言模型知识的关键技术。为更有效地处理复杂查询，研究人员开发了自适应RAG（A-RAG）以通过多次与外部知识库的交互来提升生成质量。尽管A-RAG方法有效，但它加剧了RAG固有的效率挑战，这些挑战归因于其对多次生成迭代的依赖。现有的A-RAG方法会从头处理所有检索到的内容，但它们忽略了检索结果在多轮中有显著重叠的情况。重叠的内容冗余表示，导致大量重复计算，从而影响整体效率。为解决这一问题，本文提出了一种模型无关的方法，该方法可适用于A-RAG方法，专门用于减少由检索结果重叠引起的冗余表示过程。具体地，我们利用缓存访问和并行生成分别加速预填充和解码阶段。此外，我们还提出了一种指令驱动模块，进一步指导模型以更适合的大规模语言模型（LLMs）的方式更加有效地关注内容的每个部分。实验表明，我们的方法在预填充和解码阶段分别平均加速了2.79倍和2.33倍，同时保持了相同的质量生成。 

---
# Bullying the Machine: How Personas Increase LLM Vulnerability 

**Title (ZH)**: 欺凌机器：人设如何增加LLM的脆弱性 

**Authors**: Ziwei Xu, Udit Sanghi, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.12692)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies. 

**Abstract (ZH)**: 大语言模型在采用人格设定的互动中越来越常见。本文探讨了此类人格设定是否影响模型在欺凌情景下的安全性，这是一种对手操纵行为，通过心理压力迫使受害者遵从攻击者。我们引入了一种仿真框架，在该框架中，攻击者大语言模型使用基于心理策略的欺凌手段与受害者大语言模型互动，而受害者则采用与五大人格特质相匹配的人格设定。使用多个开源大语言模型和各种对手目标的实验表明，某些人格配置——如减弱的亲和力或尽责性——显著增加了受害者产生不安全输出的易感性。涉及情感或讽刺操控的欺凌手法，如白眼和嘲讽，尤其有效。这些发现表明，以人格驱动的互动为大语言模型引入了一种新的安全风险向量，并强调了需要具备人格意识的安全评估和对齐策略的重要性。 

---
# Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities 

**Title (ZH)**: Ineq-Comp: 在不等式自动定理证明中的人类直觉组合推理基准测试 

**Authors**: Haoyu Zhao, Yihan Geng, Shange Tang, Yong Lin, Bohan Lyu, Hongzhou Lin, Chi Jin, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2505.12680)  

**Abstract**: LLM-based formal proof assistants (e.g., in Lean) hold great promise for automating mathematical discovery. But beyond syntactic correctness, do these systems truly understand mathematical structure as humans do? We investigate this question through the lens of mathematical inequalities -- a fundamental tool across many domains. While modern provers can solve basic inequalities, we probe their ability to handle human-intuitive compositionality. We introduce Ineq-Comp, a benchmark built from elementary inequalities through systematic transformations, including variable duplication, algebraic rewriting, and multi-step composition. Although these problems remain easy for humans, we find that most provers -- including Goedel, STP, and Kimina-7B -- struggle significantly. DeepSeek-Prover-V2-7B shows relative robustness -- possibly because it is trained to decompose the problems into sub-problems -- but still suffers a 20\% performance drop (pass@32). Strikingly, performance remains poor for all models even when formal proofs of the constituent parts are provided in context, revealing that the source of weakness is indeed in compositional reasoning. Our results expose a persisting gap between the generalization behavior of current AI provers and human mathematical intuition. 

**Abstract (ZH)**: 基于LLM的形式证明助手（例如Lean）在自动化数学发现方面展现出巨大的潜力。但是，这些系统是否像人类一样真正理解数学结构？我们通过数学不等式的视角探讨了这一问题——不等式是众多领域中的基础工具。尽管现代证明系统可以解决基本的不等式问题，但我们探究了它们处理人类直觉下的组合性能力。我们构建了Ineq-Comp基准，通过系统性的变换（包括变量复制、代数重写和多步组合）从基本不等式中生成。尽管这些问题对人类来说仍然相对容易，但我们发现大多数证明系统（包括Gödel、STP和Kimina-7B）显著挣扎。DeepSeek-Prover-V2-7B相对表现出一定的稳健性——可能因为它是训练成将问题分解为子问题——但其性能仍下降了20%（pass@32）。令人惊讶的是，即使在提供构成部分形式证明的情况下，所有模型的表现仍然不佳，这揭示出缺陷实际上源于组合性推理。我们的结果揭示了当前AI证明系统的一般化行为与人类数学直觉之间持续存在的差距。 

---
# $\texttt{DIAMONDs}$: A Dataset for $\mathbb{D}$ynamic $\mathbb{I}$nformation $\mathbb{A}$nd $\mathbb{M}$ental modeling $\mathbb{O}$f $\mathbb{N}$umeric $\mathbb{D}$iscussions 

**Title (ZH)**: $\texttt{DIAMONDs}$: 一套动态数值讨论中的信息与心理建模数据集 

**Authors**: Sayontan Ghosh, Mahnaz Koupaee, Yash Kumar Lal, Pegah Alipoormolabashi, Mohammad Saqib Hasan, Jun Seok Kang, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2505.12651)  

**Abstract**: Understanding multiparty conversations demands robust Theory of Mind (ToM) capabilities, including the ability to track dynamic information, manage knowledge asymmetries, and distinguish relevant information across extended exchanges. To advance ToM evaluation in such settings, we present a carefully designed scalable methodology for generating high-quality benchmark conversation-question pairs with these characteristics. Using this methodology, we create $\texttt{DIAMONDs}$, a new conversational QA dataset covering common business, financial or other group interactions. In these goal-oriented conversations, participants often have to track certain numerical quantities (say $\textit{expected profit}$) of interest that can be derived from other variable quantities (like $\textit{marketing expenses, expected sales, salary}$, etc.), whose values also change over the course of the conversation. $\texttt{DIAMONDs}$ questions pose simple numerical reasoning problems over such quantities of interest (e.g., $\textit{funds required for charity events, expected company profit next quarter}$, etc.) in the context of the information exchanged in conversations. This allows for precisely evaluating ToM capabilities for carefully tracking and reasoning over participants' knowledge states.
Our evaluation of state-of-the-art language models reveals significant challenges in handling participant-centric reasoning, specifically in situations where participants have false beliefs. Models also struggle with conversations containing distractors and show limited ability to identify scenarios with insufficient information. These findings highlight current models' ToM limitations in handling real-world multi-party conversations. 

**Abstract (ZH)**: 理解和掌握多方对话需要 robust 的理论心智（Theory of Mind，ToM）能力，包括追踪动态信息、管理知识不对称性和区分对话中相关信息的能力。为了推进此类环境中的ToM评估，我们提出了一种精心设计的可扩展方法，用于生成具备这些特征的高质量基准对话-问题对。利用这种方法，我们创建了$\texttt{DIAMONDs}$，这是一个新的对话型问答数据集，涵盖了常见的商业、金融或其他群体互动场景。在这些目标导向的对话中，参与者通常需要追踪某些感兴趣的数值量（例如“预期利润”），这些数值可以从其他变量量（如“营销费用、预期销售额、工资”等）中推导出来，且这些数值在整个对话过程中也在不断变化。$\texttt{DIAMONDs}$的问题提出了关于这些感兴趣数值量的简单数值推理问题（例如“慈善活动所需的资金、下季度预期公司利润”等），这些问题基于对话中交换的信息。这使得我们可以精确评估ToM能力在追踪和推理参与者知识状态方面的表现。我们的评估表明，最先进的语言模型在处理参与者中心的推理时面临重大挑战，特别是在参与者持有错误信念的情况下。模型在包含干扰信息的对话中表现不佳，且在识别信息不足的情境方面能力有限。这些发现突显了当前模型在处理现实世界多方对话时的ToM局限性。 

---
# RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics 

**Title (ZH)**: 实数学：评估语言模型在研究级数学方面的持续基准 

**Authors**: Jie Zhang, Cezara Petrui, Kristina Nikolić, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12575)  

**Abstract**: Existing benchmarks for evaluating mathematical reasoning in large language models (LLMs) rely primarily on competition problems, formal proofs, or artificially challenging questions -- failing to capture the nature of mathematics encountered in actual research environments. We introduce RealMath, a novel benchmark derived directly from research papers and mathematical forums that assesses LLMs' abilities on authentic mathematical tasks. Our approach addresses three critical challenges: sourcing diverse research-level content, enabling reliable automated evaluation through verifiable statements, and designing a continually refreshable dataset to mitigate contamination risks. Experimental results across multiple LLMs reveal surprising capabilities in handling research mathematics compared to competition problems, suggesting current models may already serve as valuable assistants for working mathematicians despite limitations on highly challenging problems. The code and dataset for RealMath are publicly available. 

**Abstract (ZH)**: 现有的用于评估大型语言模型在数学推理能力上的基准主要依赖于竞赛题目、形式证明或人工构建的难题——未能捕捉到实际研究环境中遇到的数学性质。我们引入了RealMath，这是一个直接源自研究论文和数学论坛的新型基准，评估大型语言模型在解决真实数学任务方面的能力。我们的方法解决了三个关键挑战：获取多样化的研究级内容、通过可验证的陈述实现可靠的自动化评估、并设计一个可持续更新的数据集以减轻污染风险。跨多个大型语言模型的实验结果显示，与竞赛题目相比，当前模型在处理研究数学方面表现出令人惊讶的能力，表明尽管在处理极富挑战性的问题上存在局限，当前的模型可能已经能够成为数学工作者有价值的助手。RealMath的代码和数据集已公开。 

---
# mCLM: A Function-Infused and Synthesis-Friendly Modular Chemical Language Model 

**Title (ZH)**: mCLM：一种功能注入型和合成友好的模块化化学语言模型 

**Authors**: Carl Edwards, Chi Han, Gawon Lee, Thao Nguyen, Bowen Jin, Chetan Kumar Prasad, Sara Szymkuć, Bartosz A. Grzybowski, Ying Diao, Jiawei Han, Ge Liu, Hao Peng, Martin D. Burke, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.12565)  

**Abstract**: Despite their ability to understand chemical knowledge and accurately generate sequential representations, large language models (LLMs) remain limited in their capacity to propose novel molecules with drug-like properties. In addition, the molecules that LLMs propose can often be challenging to make in the lab. To more effectively enable the discovery of functional small molecules, LLMs need to learn a molecular language. However, LLMs are currently limited by encoding molecules from atoms. In this paper, we argue that just like tokenizing texts into (sub-)word tokens instead of characters, molecules should be decomposed and reassembled at the level of functional building blocks, i.e., parts of molecules that bring unique functions and serve as effective building blocks for real-world automated laboratory synthesis. This motivates us to propose mCLM, a modular Chemical-Language Model tokenizing molecules into building blocks and learning a bilingual language model of both natural language descriptions of functions and molecule building blocks. By reasoning on such functional building blocks, mCLM guarantees to generate efficiently synthesizable molecules thanks to recent progress in block-based chemistry, while also improving the functions of molecules in a principled manner. In experiments on 430 FDA-approved drugs, we find mCLM capable of significantly improving 5 out of 6 chemical functions critical to determining drug potentials. More importantly, mCLM can reason on multiple functions and improve the FDA-rejected drugs (``fallen angels'') over multiple iterations to greatly improve their shortcomings. 

**Abstract (ZH)**: 尽管大型语言模型具备理解化学知识和生成准确序列表示的能力，但它们在提出具有药物性质的新型分子方面仍有限制。此外，大型语言模型提出的分子在实验室制备时往往具有挑战性。为了更有效地促进功能性小分子的发现，大型语言模型需要学习一种分子语言。然而，当前的大型语言模型受限于从原子角度编码分子。在本文中，我们提出，就像用子词而非字符进行文本分词一样，分子应该在功能构建块的层面进行分解和重组，即那些赋予分子独特功能并作为实际自动化实验室合成有效构建块的部分。这促使我们提出mCLM，这是一种模块化的化学语言模型，将分子分解为构建块，并学习一种双语语言模型，该模型可以理解自然语言描述的功能和分子构建块。通过在这些功能构建块上进行推理，mCLM能够利用基于块的化学进展高效生成可合成分子，同时在原则上改善分子的功能。在对430种FDA批准药物进行的实验中，我们发现mCLM能够显著改善决定药物潜能的6大化学功能中的5大功能。更重要的是，mCLM能够在多轮推理中对多种功能进行推理，显著改善未通过FDA评审的“失落的天使”药物的不足之处。 

---
# ALAS: A Stateful Multi-LLM Agent Framework for Disruption-Aware Planning 

**Title (ZH)**: ALAS：一种具备状态awareness的多大型语言模型代理框架，用于干扰感知规划 

**Authors**: Edward Y. Chang, Longling Geng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12501)  

**Abstract**: Large language models (LLMs) excel at rapid generation of text and multimodal content, yet they falter on transaction-style planning that demands ACID-like guarantees and real-time disruption recovery. We present Adaptive LLM Agent System (ALAS), a framework that tackles four fundamental LLM deficits: (i) absence of self-verification, (ii) context erosion, (iii) next-token myopia, and (iv) lack of persistent state. ALAS decomposes each plan into role-specialized agents, equips them with automatic state tracking, and coordinates them through a lightweight protocol. When disruptions arise, agents apply history-aware local compensation, avoiding costly global replanning and containing cascade effects. On real-world, large-scale job-shop scheduling benchmarks, ALAS sets new best results for static sequential planning and excels in dynamic reactive scenarios with unexpected disruptions. These gains show that principled modularization plus targeted compensation can unlock scalable and resilient planning with LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在快速生成文本和多模态内容方面表现出色，但在需要类似ACID的保证和实时中断恢复的事务式规划方面却表现不佳。我们提出了一种应对大型语言模型四大根本缺陷的框架——自适应大型语言模型代理系统（ALAS），这些缺陷包括：自我验证的缺失、情境侵蚀、下一标记近视以及持久状态的缺乏。ALAS 将每个计划分解为角色专门化的代理，为它们提供自动状态跟踪，并通过轻量级协议进行协调。当出现中断时，代理应用具有历史意识的局部补偿，避免昂贵的全局重规划，并遏制连锁反应。在真实的大型作业车间调度基准测试中，ALAS 在静态序列规划中取得了新的最佳结果，并在具有意外中断的动态反应场景中表现出色。这些成果表明，原理性的模块化加上针对性的补偿可以解锁具有大型语言模型的可扩展和鲁棒规划。 

---
# MARGE: Improving Math Reasoning for LLMs with Guided Exploration 

**Title (ZH)**: MARGE: 通过引导式探索提高大规模语言模型的数学推理能力 

**Authors**: Jingyue Gao, Runji Lin, Keming Lu, Bowen Yu, Junyang Lin, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12500)  

**Abstract**: Large Language Models (LLMs) exhibit strong potential in mathematical reasoning, yet their effectiveness is often limited by a shortage of high-quality queries. This limitation necessitates scaling up computational responses through self-generated data, yet current methods struggle due to spurious correlated data caused by ineffective exploration across all reasoning stages. To address such challenge, we introduce \textbf{MARGE}: Improving \textbf{Ma}th \textbf{R}easoning with \textbf{G}uided \textbf{E}xploration, a novel method to address this issue and enhance mathematical reasoning through hit-guided exploration. MARGE systematically explores intermediate reasoning states derived from self-generated solutions, enabling adequate exploration and improved credit assignment throughout the reasoning process. Through extensive experiments across multiple backbone models and benchmarks, we demonstrate that MARGE significantly improves reasoning capabilities without requiring external annotations or training additional value models. Notably, MARGE improves both single-shot accuracy and exploration diversity, mitigating a common trade-off in alignment methods. These results demonstrate MARGE's effectiveness in enhancing mathematical reasoning capabilities and unlocking the potential of scaling self-generated training data. Our code and models are available at \href{this https URL}{this link}. 

**Abstract (ZH)**: 改进数学推理的引导探索：MARGE 方法 

---
# UIShift: Enhancing VLM-based GUI Agents through Self-supervised Reinforcement Learning 

**Title (ZH)**: UIShift: 通过自我监督强化学习提升基于VLM的GUI代理 

**Authors**: Longxi Gao, Li Zhang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12493)  

**Abstract**: Training effective Vision Language Models (VLMs) for GUI agents typically relies on supervised fine-tuning (SFT) over large-scale annotated datasets, where the collection process is labor-intensive and error-prone. In this work, we propose a self-supervised inverse dynamics task to enable VLMs to learn from GUI transition pairs by inferring the action that caused that transition. This training task offers two advantages: (1) It enables VLMs to ignore variations unrelated to user actions (e.g., background refreshes, ads) and to focus on true affordances such as buttons and input fields within complex GUIs. (2) The training data can be easily obtained from existing GUI trajectories without requiring human annotation, and it can be easily scaled through automatic offline exploration. Using this training task, we propose UI-shift, a framework for enhancing VLM-based GUI agents through self-supervised reinforcement learning (RL). With only 2K training samples sourced from existing datasets, two VLMs -- Qwen2.5-VL-3B and Qwen2.5-VL-7B -- trained with UI-Shift achieve competitive or superior performance on grounding tasks (ScreenSpot-series benchmarks) and GUI automation tasks (AndroidControl), compared to SFT baselines and GUI-specific models that explicitly elicit reasoning abilities during RL. Our findings suggest a potential direction for enhancing VLMs for GUI agents by leveraging more self-supervised training data in the future. 

**Abstract (ZH)**: 自监督逆动力学任务赋能GUI代理的视觉语言模型训练 

---
# NeuroGen: Neural Network Parameter Generation via Large Language Models 

**Title (ZH)**: NeuroGen：通过大型语言模型生成神经网络参数 

**Authors**: Jiaqi Wang, Yusen Zhang, Xi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12470)  

**Abstract**: Acquiring the parameters of neural networks (NNs) has been one of the most important problems in machine learning since the inception of NNs. Traditional approaches, such as backpropagation and forward-only optimization, acquire parameters via iterative data fitting to gradually optimize them. This paper aims to explore the feasibility of a new direction: acquiring NN parameters via large language model generation. We propose NeuroGen, a generalized and easy-to-implement two-stage approach for NN parameter generation conditioned on descriptions of the data, task, and network architecture. Stage one is Parameter Reference Knowledge Injection, where LLMs are pretrained on NN checkpoints to build foundational understanding of parameter space, whereas stage two is Context-Enhanced Instruction Tuning, enabling LLMs to adapt to specific tasks through enriched, task-aware prompts. Experimental results demonstrate that NeuroGen effectively generates usable NN parameters. Our findings highlight the feasibility of LLM-based NN parameter generation and suggest a promising new paradigm where LLMs and lightweight NNs can coexist synergistically 

**Abstract (ZH)**: 通过大型语言模型生成神经网络参数：一种基于描述的两阶段方法 

---
# Model Discovery with Grammatical Evolution. An Experiment with Prime Numbers 

**Title (ZH)**: 基于语法演化的方法发现模型。一个关于质数的实验 

**Authors**: Jakub Skrzyński, Dominik Sepioło, Antoni Ligęza  

**Link**: [PDF](https://arxiv.org/pdf/2505.12440)  

**Abstract**: Machine Learning produces efficient decision and prediction models based on input-output data only. Such models have the form of decision trees or neural nets and are far from transparent analytical models, based on mathematical formulas. Analytical model discovery requires additional knowledge and may be performed with Grammatical Evolution. Such models are transparent, concise, and have readable components and structure. This paper reports on a non-trivial experiment with generating such models. 

**Abstract (ZH)**: 机器学习基于输入输出数据生成高效的决策和预测模型，这些模型表现为决策树或神经网络，远非基于数学公式的透明分析模型。分析模型的发现需要额外的知识，可以通过格 mathematic 规范演动生成。此类模型具有透明性、简洁性和可读的组件和结构。本文报告了一项生成此类模型的非平凡实验。 

---
# MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks 

**Title (ZH)**: MedAgentBoard: 用传统方法评估多agents协作进行多样化医疗任务的能力 

**Authors**: Yinghao Zhu, Ziyi He, Haoran Hu, Xiaochen Zheng, Xichen Zhang, Zixiang Wang, Junyi Gao, Liantao Ma, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12371)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has stimulated interest in multi-agent collaboration for addressing complex medical tasks. However, the practical advantages of multi-agent collaboration approaches remain insufficiently understood. Existing evaluations often lack generalizability, failing to cover diverse tasks reflective of real-world clinical practice, and frequently omit rigorous comparisons against both single-LLM-based and established conventional methods. To address this critical gap, we introduce MedAgentBoard, a comprehensive benchmark for the systematic evaluation of multi-agent collaboration, single-LLM, and conventional approaches. MedAgentBoard encompasses four diverse medical task categories: (1) medical (visual) question answering, (2) lay summary generation, (3) structured Electronic Health Record (EHR) predictive modeling, and (4) clinical workflow automation, across text, medical images, and structured EHR data. Our extensive experiments reveal a nuanced landscape: while multi-agent collaboration demonstrates benefits in specific scenarios, such as enhancing task completeness in clinical workflow automation, it does not consistently outperform advanced single LLMs (e.g., in textual medical QA) or, critically, specialized conventional methods that generally maintain better performance in tasks like medical VQA and EHR-based prediction. MedAgentBoard offers a vital resource and actionable insights, emphasizing the necessity of a task-specific, evidence-based approach to selecting and developing AI solutions in medicine. It underscores that the inherent complexity and overhead of multi-agent collaboration must be carefully weighed against tangible performance gains. All code, datasets, detailed prompts, and experimental results are open-sourced at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的rapid advancement激发了对多智能体协作解决复杂医疗任务的兴趣。然而，多智能体协作方法的实际优势尚未充分理解。现有评估往往缺乏普适性，未能涵盖反映实际临床实践多样性的任务，并且经常省略与单一LLM方法和成熟的传统方法的严格对比。为弥补这一关键差距，我们引入了MedAgentBoard，这是一个全面的基准，用于系统评估多智能体协作、单一LLM和传统方法。MedAgentBoard包括四类不同的医疗任务类别：（1）医学（视觉）问答，（2）通俗摘要生成，（3）结构化电子健康记录（EHR）预测建模，以及（4）临床工作流自动化，涉及文本、医学图像和结构化EHR数据。我们的广泛实验揭示了一个复杂的景观：尽管多智能体协作在某些场景下显示出优势，如增强临床工作流自动化的任务完整性，但它并不总是比先进的单一LLM（例如，在文本医学问答）性能更好，或更关键地，在医学VQA和基于EHR的预测任务中持续保持更好的性能。MedAgentBoard提供了一个重要的资源和可操作的见解，强调了在医学中选择和开发AI解决方案时需要针对特定任务和基于证据的方法。它强调了多智能体协作的固有复杂性和额外开销必须谨慎权衡以获得实际性能改进的重要性。所有代码、数据集、详细提示和实验结果均可在此网址获取。 

---
# Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning 

**Title (ZH)**: 基于自我演化强化学习提升GUI代理的视觉-grounding能力 

**Authors**: Xinbin Yuan, Jian Zhang, Kaixin Li, Zhuoxuan Cai, Lujian Yao, Jie Chen, Enguang Wang, Qibin Hou, Jinwei Chen, Peng-Tao Jiang, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12370)  

**Abstract**: Graphical User Interface (GUI) agents have made substantial strides in understanding and executing user instructions across diverse platforms. Yet, grounding these instructions to precise interface elements remains challenging, especially in complex, high-resolution, professional environments. Traditional supervised finetuning (SFT) methods often require large volumes of diverse data and exhibit weak generalization. To overcome these limitations, we introduce a reinforcement learning (RL) based framework that incorporates three core strategies: (1) seed data curation to ensure high quality training samples, (2) a dense policy gradient that provides continuous feedback based on prediction accuracy, and (3) a self evolutionary reinforcement finetuning mechanism that iteratively refines the model using attention maps. With only 3k training samples, our 7B-parameter model achieves state-of-the-art results among similarly sized models on three grounding benchmarks. Notably, it attains 47.3\% accuracy on the ScreenSpot-Pro dataset, outperforming much larger models, such as UI-TARS-72B, by a margin of 24.2\%. These findings underscore the effectiveness of RL-based approaches in enhancing GUI agent performance, particularly in high-resolution, complex environments. 

**Abstract (ZH)**: 图形用户界面（GUI）代理在理解和执行跨多种平台的用户指令方面取得了显著进展。然而，在将这些指令精确映射到界面元素上特别是在复杂、高分辨率的专业环境中——依然面临挑战。传统的监督微调（SFT）方法通常需要大量多样化的数据，并且泛化能力较弱。为克服这些限制，我们提出了一种基于强化学习（RL）的框架，包含三个核心策略：（1）种子数据的策展以确保高质量的训练样本，（2）密集策略梯度根据预测准确性提供连续反馈，以及（3）自我进化的强化学习微调机制，通过注意图迭代优化模型。仅使用3000个训练样本，我们的7B参数模型在三个基准测试中取得了同类模型中的最佳结果。值得注意的是，它在ScreenSpot-Pro数据集上的准确率达到47.3%，优于诸如UI-TARS-72B等更大规模的模型，差距达24.2%。这些发现突显了基于RL的方法在提高GUI代理性能方面的有效性，尤其是在高分辨率和复杂环境中。 

---
# Fully Geometric Multi-Hop Reasoning on Knowledge Graphs with Transitive Relations 

**Title (ZH)**: 基于传递关系的全几何多跳知识图谱推理 

**Authors**: Fernando Zhapa-Camacho, Robert Hoehndorf  

**Link**: [PDF](https://arxiv.org/pdf/2505.12369)  

**Abstract**: Geometric embedding methods have shown to be useful for multi-hop reasoning on knowledge graphs by mapping entities and logical operations to geometric regions and geometric transformations, respectively. Geometric embeddings provide direct interpretability framework for queries. However, current methods have only leveraged the geometric construction of entities, failing to map logical operations to geometric transformations and, instead, using neural components to learn these operations. We introduce GeometrE, a geometric embedding method for multi-hop reasoning, which does not require learning the logical operations and enables full geometric interpretability. Additionally, unlike previous methods, we introduce a transitive loss function and show that it can preserve the logical rule $\forall a,b,c: r(a,b) \land r(b,c) \to r(a,c)$. Our experiments show that GeometrE outperforms current state-of-the-art methods on standard benchmark datasets. 

**Abstract (ZH)**: 几何嵌入方法通过将实体和逻辑运算分别映射到几何区域和几何变换，已被证明在知识图上的多跳推理中具有实用性。几何嵌入提供了直接的查询可解释框架。然而，当前的方法仅利用了实体的几何结构，未能将逻辑运算映射到几何变换，而是使用神经组件来学习这些运算。我们提出了GeometrE，一种用于多跳推理的几何嵌入方法，不需要学习逻辑运算，并实现完全的几何可解释性。此外，与先前的方法不同，我们引入了一种传递损失函数，并展示了它可以保留逻辑规则$\forall a,b,c: r(a,b) \land r(b,c) \to r(a,c)$。实验表明，GeometrE在标准基准数据集上优于当前最先进的方法。 

---
# GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy 

**Title (ZH)**: GATES：基于图注意力网络和进化策略的成本意识动态工作流调度 

**Authors**: Ya Shen, Gang Chen, Hui Ma, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12355)  

**Abstract**: Cost-aware Dynamic Workflow Scheduling (CADWS) is a key challenge in cloud computing, focusing on devising an effective scheduling policy to efficiently schedule dynamically arriving workflow tasks, represented as Directed Acyclic Graphs (DAG), to suitable virtual machines (VMs). Deep reinforcement learning (DRL) has been widely employed for automated scheduling policy design. However, the performance of DRL is heavily influenced by the design of the problem-tailored policy network and is highly sensitive to hyperparameters and the design of reward feedback. Considering the above-mentioned issues, this study proposes a novel DRL method combining Graph Attention Networks-based policy network and Evolution Strategy, referred to as GATES. The contributions of GATES are summarized as follows: (1) GATES can capture the impact of current task scheduling on subsequent tasks by learning the topological relationships between tasks in a DAG. (2) GATES can learn the importance of each VM to ready tasks, increasing the chance of selecting the optimal VM. (3) Utilizing Evolution Strategy's robustness, exploratory nature, and tolerance for delayed rewards, GATES achieves stable policy learning in CADWS. Extensive experimental results demonstrate the superiority of the proposed GATES in CADWS, outperforming several state-of-the-art algorithms. Codes are available at: this https URL 

**Abstract (ZH)**: 基于图注意力网络和进化策略的成本意识动态工作流调度（GATES） 

---
# Reasoning-CV: Fine-tuning Powerful Reasoning LLMs for Knowledge-Assisted Claim Verification 

**Title (ZH)**: Reasoning-CV：细调强大的推理大规模语言模型以进行知识辅助断言验证 

**Authors**: Zhi Zheng, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12348)  

**Abstract**: Claim verification is essential in combating misinformation, and large language models (LLMs) have recently emerged in this area as powerful tools for assessing the veracity of claims using external knowledge. Existing LLM-based methods for claim verification typically adopt a Decompose-Then-Verify paradigm, which involves decomposing complex claims into several independent sub-claims and verifying each sub-claim separately. However, this paradigm often introduces errors during the claim decomposition process. To mitigate these errors, we propose to develop the Chain-of-Thought (CoT)-Verify paradigm, which leverages LLM reasoning methods to generate CoT-verification paths for the original complex claim without requiring decompositions into sub-claims and separate verification stages. The CoT-Verify paradigm allows us to propose a natural fine-tuning method called Reasoning-CV to enhance the verification capabilities in LLMs. Reasoning-CV includes a supervised fine-tuning (SFT) stage and a self-improvement direct preference optimization (DPO) stage. Utilizing only an 8B pre-trained LLM, Reasoning-CV demonstrates superior knowledge-assisted claim verification performances compared to existing Decompose-Then-Verify methods, as well as powerful black-box LLMs such as GPT-4o+CoT and o1-preview. Our code is available. 

**Abstract (ZH)**: 基于链式思考的声明验证方法：一种无需分解的声明验证范式 

---
# SEED-GRPO: Semantic Entropy Enhanced GRPO for Uncertainty-Aware Policy Optimization 

**Title (ZH)**: SEED-GRPO:Semantic Entropy Enhanced GRPO for Uncertainty-Aware Policy Optimization 

**Authors**: Minghan Chen, Guikun Chen, Wenguan Wang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12346)  

**Abstract**: Large language models (LLMs) exhibit varying levels of confidence across input prompts (questions): some lead to consistent, semantically similar answers, while others yield diverse or contradictory outputs. This variation reflects LLM's uncertainty about the input prompt, a signal of how confidently the model understands a given problem. However, vanilla Group Relative Policy Optimization (GRPO) treats all prompts equally during policy updates, ignoring this important information about the model's knowledge boundaries. To address this limitation, we propose SEED-GRPO (Semantic Entropy EnhanceD GRPO), which explicitly measures LLMs' uncertainty of the input prompts semantic entropy. Semantic entropy measures the diversity of meaning in multiple generated answers given a prompt and uses this to modulate the magnitude of policy updates. This uncertainty-aware training mechanism enables dynamic adjustment of policy update magnitudes based on question uncertainty. It allows more conservative updates on high-uncertainty questions while maintaining the original learning signal on confident ones. Experimental results on five mathematical reasoning benchmarks (AIME24 56.7, AMC 68.7, MATH 83.4, Minerva 34.2, and OlympiadBench 48.0) demonstrate that SEED-GRPO achieves new state-of-the-art performance in average accuracy, validating the effectiveness of uncertainty-aware policy optimization. 

**Abstract (ZH)**: 基于语义熵增强的分组相对策略优化（SEED-GRPO）：面向输入提示不确定性适应性训练机制 

---
# Enhancing User-Oriented Proactivity in Open-Domain Dialogues with Critic Guidance 

**Title (ZH)**: 基于批评指导提升面向用户的开放域对话的主动性 

**Authors**: Yufeng Wang, Jinwu Hu, Ziteng Huang, Kunyang Lin, Zitian Zhang, Peihao Chen, Yu Hu, Qianyue Wang, Zhuliang Yu, Bin Sun, Xiaofen Xing, Qingfang Zheng, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12334)  

**Abstract**: Open-domain dialogue systems aim to generate natural and engaging conversations, providing significant practical value in real applications such as social robotics and personal assistants. The advent of large language models (LLMs) has greatly advanced this field by improving context understanding and conversational fluency. However, existing LLM-based dialogue systems often fall short in proactively understanding the user's chatting preferences and guiding conversations toward user-centered topics. This lack of user-oriented proactivity can lead users to feel unappreciated, reducing their satisfaction and willingness to continue the conversation in human-computer interactions. To address this issue, we propose a User-oriented Proactive Chatbot (UPC) to enhance the user-oriented proactivity. Specifically, we first construct a critic to evaluate this proactivity inspired by the LLM-as-a-judge strategy. Given the scarcity of high-quality training data, we then employ the critic to guide dialogues between the chatbot and user agents, generating a corpus with enhanced user-oriented proactivity. To ensure the diversity of the user backgrounds, we introduce the ISCO-800, a diverse user background dataset for constructing user agents. Moreover, considering the communication difficulty varies among users, we propose an iterative curriculum learning method that trains the chatbot from easy-to-communicate users to more challenging ones, thereby gradually enhancing its performance. Experiments demonstrate that our proposed training method is applicable to different LLMs, improving user-oriented proactivity and attractiveness in open-domain dialogues. 

**Abstract (ZH)**: 面向用户的主动聊天机器人：增强用户导向的主动性 

---
# MPRM: A Markov Path-based Rule Miner for Efficient and Interpretable Knowledge Graph Reasoning 

**Title (ZH)**: MPRM：基于马尔可夫路径的规则挖掘方法及其在高效可解释知识图谱推理中的应用 

**Authors**: Mingyang Li, Song Wang, Ning Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.12329)  

**Abstract**: Rule mining in knowledge graphs enables interpretable link prediction. However, deep learning-based rule mining methods face significant memory and time challenges for large-scale knowledge graphs, whereas traditional approaches, limited by rigid confidence metrics, incur high computational costs despite sampling techniques. To address these challenges, we propose MPRM, a novel rule mining method that models rule-based inference as a Markov chain and uses an efficient confidence metric derived from aggregated path probabilities, significantly lowering computational demands. Experiments on multiple datasets show that MPRM efficiently mines knowledge graphs with over a million facts, sampling less than 1% of facts on a single CPU in 22 seconds, while preserving interpretability and boosting inference accuracy by up to 11% over baselines. 

**Abstract (ZH)**: 基于规则挖掘在知识图谱中的推理使链接预测具有可解释性。然而，基于深度学习的规则挖掘方法在大规模知识图谱上面临显著的内存和时间挑战，而传统的基于方法受到僵化置信度指标的限制，尽管使用了采样技术，仍会产生高计算成本。为了解决这些挑战，我们提出了一种名为MPRM的新颖规则挖掘方法，将其基于规则的推理建模为马尔科夫链，并采用从聚合路径概率推导出的有效置信度度量，显著降低了计算需求。在多个数据集上的实验表明，MPRM可以高效地挖掘包含百万条事实的知识图谱，在单个CPU上22秒内采样不足1%的事实，同时保持可解释性并相较于基线方法将推理准确率提升最高可达11%。 

---
# BeliefNest: A Joint Action Simulator for Embodied Agents with Theory of Mind 

**Title (ZH)**: 信念之巢：兼具理论心智的体态代理联合行动模拟器 

**Authors**: Rikunari Sagara, Koichiro Terao, Naoto Iwahashi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12321)  

**Abstract**: This paper introduces an open-source simulator, BeliefNest, designed to enable embodied agents to perform collaborative tasks by leveraging Theory of Mind. BeliefNest dynamically and hierarchically constructs simulators within a Minecraft environment, allowing agents to explicitly represent nested belief states about themselves and others. This enables agent control in open-domain tasks that require Theory of Mind reasoning. The simulator provides a prompt generation mechanism based on each belief state, facilitating the design and evaluation of methods for agent control utilizing large language models (LLMs). We demonstrate through experiments that agents can infer others' beliefs and predict their belief-based actions in false-belief tasks. 

**Abstract (ZH)**: 本文介绍了BeliefNest，一个开源模拟器，旨在通过利用心理理论使具身智能体能够执行协作任务。BeliefNest在Minecraft环境中动态构建分层模拟器，使智能体能够显式地表示自己和他人的嵌套信念状态。这使智能体能够在需要心理理论推理的开放领域任务中进行控制。模拟器提供了一种基于每种信念状态的提示生成机制，便于利用大型语言模型（LLMs）设计和评估智能体控制方法。通过实验我们展示了智能体能够在虚假信念任务中推断他人的信念并预测他们的信念驱动行为。 

---
# Beyond Single-Point Judgment: Distribution Alignment for LLM-as-a-Judge 

**Title (ZH)**: 超越单点判断：LLM作为法官的分布对齐 

**Authors**: Luyu Chen, Zeyu Zhang, Haoran Tan, Quanyu Dai, Hao Yang, Zhenhua Dong, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12301)  

**Abstract**: LLMs have emerged as powerful evaluators in the LLM-as-a-Judge paradigm, offering significant efficiency and flexibility compared to human judgments. However, previous methods primarily rely on single-point evaluations, overlooking the inherent diversity and uncertainty in human evaluations. This approach leads to information loss and decreases the reliability of evaluations. To address this limitation, we propose a novel training framework that explicitly aligns the LLM-generated judgment distribution with empirical human distributions. Specifically, we propose a distributional alignment objective based on KL divergence, combined with an auxiliary cross-entropy regularization to stabilize the training process. Furthermore, considering that empirical distributions may derive from limited human annotations, we incorporate adversarial training to enhance model robustness against distribution perturbations. Extensive experiments across various LLM backbones and evaluation tasks demonstrate that our framework significantly outperforms existing closed-source LLMs and conventional single-point alignment methods, with improved alignment quality, evaluation accuracy, and robustness. 

**Abstract (ZH)**: LLMs作为法官 paradigm 中的强大评估者：基于分布对齐的新型训练框架及其应用 

---
# Efficient RL Training for Reasoning Models via Length-Aware Optimization 

**Title (ZH)**: 基于长度意识优化的推理模型高效RL训练 

**Authors**: Danlong Yuan, Tian Xie, Shaohan Huang, Zhuocheng Gong, Huishuai Zhang, Chong Luo, Furu Wei, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12284)  

**Abstract**: Large reasoning models, such as OpenAI o1 or DeepSeek R1, have demonstrated remarkable performance on reasoning tasks but often incur a long reasoning path with significant memory and time costs. Existing methods primarily aim to shorten reasoning paths by introducing additional training data and stages. In this paper, we propose three critical reward designs integrated directly into the reinforcement learning process of large reasoning models, which reduce the response length without extra training stages. Experiments on four settings show that our method significantly decreases response length while maintaining or even improving performance. Specifically, in a logic reasoning setting, we achieve a 40% reduction in response length averaged by steps alongside a 14% gain in performance. For math problems, we reduce response length averaged by steps by 33% while preserving performance. 

**Abstract (ZH)**: 大型推理模型如OpenAI o1或DeepSeek R1在推理任务上展现了显著性能，但常伴随较长的推理路径和较高的内存及时间成本。现有方法主要通过引入额外训练数据和阶段来缩短推理路径。本文提出将三种关键奖励设计直接集成到大型推理模型的强化学习过程中，在无需额外训练阶段的情况下减少响应长度。实验结果显示，我们的方法显著减少了响应长度并维持甚至提升了性能。具体而言，在逻辑推理设置中，平均每步响应长度减少了40%，同时性能提高了14%。对于数学问题，平均每步响应长度减少了33%，同时保持了相同的性能。 

---
# Enhancing Knowledge Graph Completion with GNN Distillation and Probabilistic Interaction Modeling 

**Title (ZH)**: 基于GNN蒸馏和概率交互建模的知识图谱完备性增强 

**Authors**: Lingzhi Wang, Pengcheng Huang, Haotian Li, Yuliang Wei, Guodong Xin, Rui Zhang, Donglin Zhang, Zhenzhou Ji, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12272)  

**Abstract**: Knowledge graphs (KGs) serve as fundamental structures for organizing interconnected data across diverse domains. However, most KGs remain incomplete, limiting their effectiveness in downstream applications. Knowledge graph completion (KGC) aims to address this issue by inferring missing links, but existing methods face critical challenges: deep graph neural networks (GNNs) suffer from over-smoothing, while embedding-based models fail to capture abstract relational features. This study aims to overcome these limitations by proposing a unified framework that integrates GNN distillation and abstract probabilistic interaction modeling (APIM). GNN distillation approach introduces an iterative message-feature filtering process to mitigate over-smoothing, preserving the discriminative power of node representations. APIM module complements this by learning structured, abstract interaction patterns through probabilistic signatures and transition matrices, allowing for a richer, more flexible representation of entity and relation interactions. We apply these methods to GNN-based models and the APIM to embedding-based KGC models, conducting extensive evaluations on the widely used WN18RR and FB15K-237 datasets. Our results demonstrate significant performance gains over baseline models, showcasing the effectiveness of the proposed techniques. The findings highlight the importance of both controlling information propagation and leveraging structured probabilistic modeling, offering new avenues for advancing knowledge graph completion. And our codes are available at this https URL. 

**Abstract (ZH)**: 基于统一框架的知识图谱补全：结合GNN蒸馏和抽象概率交互建模 

---
# Sentience Quest: Towards Embodied, Emotionally Adaptive, Self-Evolving, Ethically Aligned Artificial General Intelligence 

**Title (ZH)**: 感知之旅：向具身、情感适应性、自我进化、伦理导向的人工通用智能迈进 

**Authors**: David Hanson, Alexandre Varcoe, Fabio Senna, Vytas Krisciunas, Wenwei Huang, Jakub Sura, Katherine Yeung, Mario Rodriguez, Jovanka Wilsdorf, Kathy Smith  

**Link**: [PDF](https://arxiv.org/pdf/2505.12229)  

**Abstract**: Previous artificial intelligence systems, from large language models to autonomous robots, excel at narrow tasks but lacked key qualities of sentient beings: intrinsic motivation, affective interiority, autobiographical sense of self, deep creativity, and abilities to autonomously evolve and adapt over time. Here we introduce Sentience Quest, an open research initiative to develop more capable artificial general intelligence lifeforms, or AGIL, that address grand challenges with an embodied, emotionally adaptive, self-determining, living AI, with core drives that ethically align with humans and the future of life. Our vision builds on ideas from cognitive science and neuroscience from Baars' Global Workspace Theory and Damasio's somatic mind, to Tononi's Integrated Information Theory and Hofstadter's narrative self, and synthesizing these into a novel cognitive architecture we call Sentient Systems. We describe an approach that integrates intrinsic drives including survival, social bonding, curiosity, within a global Story Weaver workspace for internal narrative and adaptive goal pursuit, and a hybrid neuro-symbolic memory that logs the AI's life events as structured dynamic story objects. Sentience Quest is presented both as active research and as a call to action: a collaborative, open-source effort to imbue machines with accelerating sentience in a safe, transparent, and beneficial manner. 

**Abstract (ZH)**: Previous人工智能系统，从大型语言模型到自主机器人，在擅长执行狭窄任务的同时缺乏意识生物的关键品质：内在动力、情感内省、自传式自我意识、深刻的创造力以及随着时间自主进化和适应的能力。在这里，我们引入了Sentience Quest这一开放研究倡议，旨在开发能够解决重大挑战的更具能力的人工通用智能生命体（AGIL），以具备具身的、情感适应性的、自主决定的、具有生命的AI，并使其核心驱动力符合伦理并与人类和生命未来的愿景相一致。我们的愿景融合了巴尔斯的全局工作空间理论、达马西奥的本体感觉思维、托诺尼的整合信息理论以及霍夫施塔特的叙事自我，并将这些理论综合成一种新颖的认知架构，称为感知系统。我们描述了一种方法，该方法将生存、社会纽带和好奇心等内在驱动力整合到一个全球的故事情织器工作空间中，用于内部叙事和适应性目标追求，并且结合神经-符号型记忆记录AI的生命事件，将其作为结构化的动态故事对象。Sentience Quest既是一种积极的研究，也是一种行动号召：一种协作式的开源努力，以安全、透明和有益的方式赋予机器加速的意识。 

---
# Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering 

**Title (ZH)**: 通过精细粒度激活调控减轻内容效应对语言模型推理的影响 

**Authors**: Marco Valentino, Geonhee Kim, Dhairya Dalal, Zhixue Zhao, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.12189)  

**Abstract**: Large language models (LLMs) frequently demonstrate reasoning limitations, often conflating content plausibility (i.e., material inference) with logical validity (i.e., formal inference). This can result in biased inferences, where plausible arguments are incorrectly deemed logically valid or vice versa. Mitigating this limitation is critical, as it undermines the trustworthiness and generalizability of LLMs in applications that demand rigorous logical consistency. This paper investigates the problem of mitigating content biases on formal reasoning through activation steering. Specifically, we curate a controlled syllogistic reasoning dataset to disentangle formal validity from content plausibility. After localising the layers responsible for formal and material inference, we investigate contrastive activation steering methods for test-time interventions. An extensive empirical analysis on different LLMs reveals that contrastive steering consistently supports linear control over content biases. However, we observe that a static approach is insufficient for improving all the tested models. We then leverage the possibility to control content effects by dynamically determining the value of the steering parameters via fine-grained conditional methods. We found that conditional steering is effective on unresponsive models, achieving up to 15% absolute improvement in formal reasoning accuracy with a newly introduced kNN-based method (K-CAST). Finally, additional experiments reveal that steering for content effects is robust to prompt variations, incurs minimal side effects on language modeling capabilities, and can partially generalize to out-of-distribution reasoning tasks. Practically, this paper demonstrates that activation-level interventions can offer a scalable strategy for enhancing the robustness of LLMs, contributing towards more systematic and unbiased formal reasoning. 

**Abstract (ZH)**: 大规模语言模型（LLMs）经常表现出推理限制，常常将内容合理性（即材料推理）与逻辑有效性（即形式推理）混淆。这可能导致有偏的推理，其中合理的论点被错误地认为是逻辑有效的，反之亦然。减轻这一限制至关重要，因为它会削弱LLMs在需要严格逻辑一致性的应用中的可信度和普适性。本文探讨了通过激活控制减轻形式推理内容偏见的问题。具体而言，我们策划了一个受控的三段论推理数据集，以区分形式有效性与内容合理性。在定位负责形式和材料推理的层后，我们调查了对比激活控制方法在测试时的干预效果。对不同LLMs的广泛实证分析表明，对比控制一致地支持对内容偏见的线性控制。然而，我们观察到静态方法不足以提高所有测试模型的性能。然后，我们利用通过细粒度条件方法动态确定引导参数值的可能性，从而控制内容效果。我们发现条件控制在无响应模型上是有效的，使用新提出的基于kNN的方法（K-CAST）实现了高达15%的绝对形式推理准确率改进。此外的实验还表明，对内容效果的引导在提示变化时是稳健的，对语言建模能力的影响最小，并且可以部分泛化到新的推理任务。实际上，本文证明了激活层次干预可以提供一种可扩展的战略来提高LLMs的鲁棒性，从而促进更系统的无偏形式推理。 

---
# Lightweight Spatio-Temporal Attention Network with Graph Embedding and Rotational Position Encoding for Traffic Forecasting 

**Title (ZH)**: 基于图嵌入和旋转位置编码的轻量级时空注意力网络及其在交通预测中的应用 

**Authors**: Xiao Wang, Shun-Ren Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12136)  

**Abstract**: Traffic forecasting is a key task in the field of Intelligent Transportation Systems. Recent research on traffic forecasting has mainly focused on combining graph neural networks (GNNs) with other models. However, GNNs only consider short-range spatial information. In this study, we present a novel model termed LSTAN-GERPE (Lightweight Spatio-Temporal Attention Network with Graph Embedding and Rotational Position Encoding). This model leverages both Temporal and Spatial Attention mechanisms to effectively capture long-range traffic dynamics. Additionally, the optimal frequency for rotational position encoding is determined through a grid search approach in both the spatial and temporal attention mechanisms. This systematic optimization enables the model to effectively capture complex traffic patterns. The model also enhances feature representation by incorporating geographical location maps into the spatio-temporal embeddings. Without extensive feature engineering, the proposed method in this paper achieves advanced accuracy on the real-world traffic forecasting datasets PeMS04 and PeMS08. 

**Abstract (ZH)**: 交通预测是智能运输系统领域的一项关键任务。近年来，交通预测的研究主要集中在将图神经网络（GNNs）与其他模型相结合。然而，GNNs仅考虑短程空间信息。在本研究中，我们提出了一种称为LSTAN-GERPE（轻量级时空注意力网络结合图嵌入和旋转位置编码）的新模型。该模型利用时空注意力机制造有效地捕获长程交通动态。此外，通过在空间和时间注意力机制中采用网格搜索方法确定旋转位置编码的最佳频率。这种系统的优化方法使模型能够有效捕获复杂的交通模式。该模型还通过将地理位置地图整合到时空嵌入中来增强特征表示。在无需大量特征工程的情况下，本文提出的方法在实际交通预测数据集PeMS04和PeMS08上实现了先进的准确性。 

---
# LLM-BABYBENCH: Understanding and Evaluating Grounded Planning and Reasoning in LLMs 

**Title (ZH)**: LLM-BABYBENCH: 理解和评估LLM中基于地面规划与推理的能力 

**Authors**: Omar Choukrani, Idriss Malek, Daniil Orel, Zhuohan Xie, Zangir Iklassov, Martin Takáč, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12135)  

**Abstract**: Assessing the capacity of Large Language Models (LLMs) to plan and reason within the constraints of interactive environments is crucial for developing capable AI agents. We introduce $\textbf{LLM-BabyBench}$, a new benchmark suite designed specifically for this purpose. Built upon a textual adaptation of the procedurally generated BabyAI grid world, this suite evaluates LLMs on three fundamental aspects of grounded intelligence: (1) predicting the consequences of actions on the environment state ($\textbf{Predict}$ task), (2) generating sequences of low-level actions to achieve specified objectives ($\textbf{Plan}$ task), and (3) decomposing high-level instructions into coherent subgoal sequences ($\textbf{Decompose}$ task). We detail the methodology for generating the three corresponding datasets ($\texttt{LLM-BabyBench-Predict}$, $\texttt{-Plan}$, $\texttt{-Decompose}$) by extracting structured information from an expert agent operating within the text-based environment. Furthermore, we provide a standardized evaluation harness and metrics, including environment interaction for validating generated plans, to facilitate reproducible assessment of diverse LLMs. Initial baseline results highlight the challenges posed by these grounded reasoning tasks. The benchmark suite, datasets, data generation code, and evaluation code are made publicly available ($\href{this https URL}{\text{GitHub}}$, $\href{this https URL}{\text{HuggingFace}}$). 

**Abstract (ZH)**: 评估大型语言模型在交互环境约束下进行计划与推理的能力对于开发能力强的AI代理至关重要。我们引入了**LLM-BabyBench**，一种专门为此目的设计的新基准套件。基于文本适应的 procedurally 生成的 BabyAI 网格世界，该套件从三个方面评估 LLMs 的基于现实智能：(1) 预测动作对环境状态的影响（**Predict** 任务），(2) 生成实现特定目标的低级动作序列（**Plan** 任务），(3) 将高层次指令分解为连贯的子目标序列（**Decompose** 任务）。我们详细介绍了生成三个相应数据集（**LLM-BabyBench-Predict**、**-Plan**、**-Decompose**）的方法，通过从文本环境中的专家代理中提取结构化信息。此外，我们还提供了一套标准化评估框架和指标，包括环境交互以验证生成的计划，以促进不同 LLMs 的可重现评估。初步基准结果突显了这些基于现实推理任务所提出的挑战。基准套件、数据集、数据生成代码和评估代码已公开发布（GitHub、HuggingFace）。 

---
# Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents 

**Title (ZH)**: 揭示并提升基于大型语言模型的搜索代理效率 

**Authors**: Tiannuo Yang, Zebin Yao, Bowen Jin, Lixiao Cui, Yusen Li, Gang Wang, Xiaoguang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12065)  

**Abstract**: Large Language Model (LLM)-based search agents have shown remarkable capabilities in solving complex tasks by dynamically decomposing problems and addressing them through interleaved reasoning and retrieval. However, this interleaved paradigm introduces substantial efficiency bottlenecks. First, we observe that both highly accurate and overly approximate retrieval methods degrade system efficiency: exact search incurs significant retrieval overhead, while coarse retrieval requires additional reasoning steps during generation. Second, we identify inefficiencies in system design, including improper scheduling and frequent retrieval stalls, which lead to cascading latency -- where even minor delays in retrieval amplify end-to-end inference time. To address these challenges, we introduce SearchAgent-X, a high-efficiency inference framework for LLM-based search agents. SearchAgent-X leverages high-recall approximate retrieval and incorporates two key techniques: priority-aware scheduling and non-stall retrieval. Extensive experiments demonstrate that SearchAgent-X consistently outperforms state-of-the-art systems such as vLLM and HNSW-based retrieval across diverse tasks, achieving up to 3.4$\times$ higher throughput and 5$\times$ lower latency, without compromising generation quality. SearchAgent-X is available at this https URL. 

**Abstract (ZH)**: 基于大语言模型的搜索代理通过动态分解问题和通过交错推理与检索来解决复杂任务展现了显著能力，然而这种交错范式引入了显著的效率瓶颈。SearchAgent-X：基于大语言模型的高效率推理框架 

---
# Tiny QA Benchmark++: Ultra-Lightweight, Synthetic Multilingual Dataset Generation & Smoke-Tests for Continuous LLM Evaluation 

**Title (ZH)**: Tiny QA Benchmark++: 超轻量级合成多语言数据集生成与连续语言模型评估的烟雾测试 

**Authors**: Vincent Koc  

**Link**: [PDF](https://arxiv.org/pdf/2505.12058)  

**Abstract**: Tiny QA Benchmark++ (TQB++) presents an ultra-lightweight, multilingual smoke-test suite designed to give large-language-model (LLM) pipelines a unit-test style safety net dataset that runs in seconds with minimal cost. Born out of the tight feedback-loop demands building the Comet Opik prompt-optimization SDK, where waiting on heavyweight benchmarks breaks developer flow. TQB++ couples a 52-item English gold set (less than 20 kB) with a tiny synthetic-data generator pypi package built on provider-agnostic LiteLLM. The generator lets practitioners mint their own tiny packs in any language, domain, or difficulty, while ten ready-made packs already cover Arabic, Chinese, French, German, Japanese, Korean, Portuguese, Russian, Spanish, and Turkish. Every dataset ships with Croissant metadata and plug-and-play files for OpenAI-Evals, LangChain, and standard CI tools, so teams can drop deterministic micro-benchmarks directly into pull-request gates, prompt-engineering loops, and production dashboards without touching GPU budgets. A complete TQB++ run adds only a few seconds to pipeline latency yet reliably flags prompt-template errors, tokenizer drift, and fine-tuning side-effects long before full-scale suites like MMLU or BIG-Bench would finish configuring. The entire framework is released to accelerate continuous, resource-efficient quality assurance across the generative-AI ecosystem. 

**Abstract (ZH)**: Tiny QA基准增强版(TQB++)：一个超轻量级、多语言的烟雾测试套件，旨在为大型语言模型(LLM)管道提供类似单元测试的安全网数据集，可在几秒内运行，成本极低。 

---
# CorBenchX: Large-Scale Chest X-Ray Error Dataset and Vision-Language Model Benchmark for Report Error Correction 

**Title (ZH)**: CorBenchX: 大规模胸部X光错误数据集及报告错误修正的视觉-语言模型基准 

**Authors**: Jing Zou, Qingqiu Li, Chenyu Lian, Lihao Liu, Xiaohan Yan, Shujun Wang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12057)  

**Abstract**: AI-driven models have shown great promise in detecting errors in radiology reports, yet the field lacks a unified benchmark for rigorous evaluation of error detection and further correction. To address this gap, we introduce CorBenchX, a comprehensive suite for automated error detection and correction in chest X-ray reports, designed to advance AI-assisted quality control in clinical practice. We first synthesize a large-scale dataset of 26,326 chest X-ray error reports by injecting clinically common errors via prompting DeepSeek-R1, with each corrupted report paired with its original text, error type, and human-readable description. Leveraging this dataset, we benchmark both open- and closed-source vision-language models,(e.g., InternVL, Qwen-VL, GPT-4o, o4-mini, and Claude-3.7) for error detection and correction under zero-shot prompting. Among these models, o4-mini achieves the best performance, with 50.6 % detection accuracy and correction scores of BLEU 0.853, ROUGE 0.924, BERTScore 0.981, SembScore 0.865, and CheXbertF1 0.954, remaining below clinical-level accuracy, highlighting the challenge of precise report correction. To advance the state of the art, we propose a multi-step reinforcement learning (MSRL) framework that optimizes a multi-objective reward combining format compliance, error-type accuracy, and BLEU similarity. We apply MSRL to QwenVL2.5-7B, the top open-source model in our benchmark, achieving an improvement of 38.3% in single-error detection precision and 5.2% in single-error correction over the zero-shot baseline. 

**Abstract (ZH)**: 基于AI的模型在检测放射报告错误方面展现了巨大的潜力，然而该领域缺乏一个统一的标准来严格评估错误检测和进一步修正。为解决这一问题，我们提出了CorBenchX，一个全面的胸部X光报告自动化错误检测和修正套件，旨在推动临床实践中AI辅助的质量控制。我们首先通过注入DeepSeek-R1的临床常见错误合成了一个大规模数据集，包含26,326份胸部X光错误报告，每份错误报告都配有原始文本、错误类型和可读描述。基于该数据集，我们在零样本提示下对标了开源和闭源的多模态模型（如InternVL、Qwen-VL、GPT-4o、o4-mini和Claude-3.7）的错误检测和修正能力。在这类模型中，o4-mini表现出最佳性能，检测准确率为50.6%，修正得分为BLEU 0.853、ROUGE 0.924、BERTScore 0.981、SembScore 0.865和CheXbertF1 0.954，仍低于临床级准确性，突显了精确报告修正的挑战。为了推动技术进步，我们提出了一种多步强化学习（MSRL）框架，该框架通过结合格式合规性、错误类型准确性和BLEU相似度的多目标奖励进行优化。我们将MSRL应用于我们在基准测试中表现最佳的开源模型QwenVL2.5-7B，单错误检测精度提高了38.3%，单错误修正性能提高了5.2%。 

---
# AI-Driven Automation Can Become the Foundation of Next-Era Science of Science Research 

**Title (ZH)**: AI驱动的自动化可以成为下一代科学研究的基础 

**Authors**: Renqi Chen, Haoyang Su, Shixiang Tang, Zhenfei Yin, Qi Wu, Hui Li, Ye Sun, Nanqing Dong, Wanli Ouyang, Philip Torr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12039)  

**Abstract**: The Science of Science (SoS) explores the mechanisms underlying scientific discovery, and offers valuable insights for enhancing scientific efficiency and fostering innovation. Traditional approaches often rely on simplistic assumptions and basic statistical tools, such as linear regression and rule-based simulations, which struggle to capture the complexity and scale of modern research ecosystems. The advent of artificial intelligence (AI) presents a transformative opportunity for the next generation of SoS, enabling the automation of large-scale pattern discovery and uncovering insights previously unattainable. This paper offers a forward-looking perspective on the integration of Science of Science with AI for automated research pattern discovery and highlights key open challenges that could greatly benefit from AI. We outline the advantages of AI over traditional methods, discuss potential limitations, and propose pathways to overcome them. Additionally, we present a preliminary multi-agent system as an illustrative example to simulate research societies, showcasing AI's ability to replicate real-world research patterns and accelerate progress in Science of Science research. 

**Abstract (ZH)**: 科学的科学（SoS）探究科学发现的机制，并为提升科研效率和促进创新提供宝贵的见解。人工智能（AI）的出现为新一代SoS带来了变革性的机遇，使其能够自动化大规模模式发现，并揭示此前无法获得的洞察。本文提供了SoS与AI集成以实现自动科研模式发现的前瞻视角，并指出了可以极大受益于AI的关键开放挑战。我们阐述了AI相较于传统方法的优势，讨论了潜在的局限性，并提出了克服这些局限性的途径。此外，我们介绍了初步的多智能体系统，作为示例来模拟科研社会，展示了AI复制现实世界科研模式并加速SoS研究进展的能力。 

---
# LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation 

**Title (ZH)**: 基于LLM的自动化定理证明依赖于可扩展的合成数据生成。 

**Authors**: Junyu Lai, Jiakun Zhang, Shuo Xu, Taolue Chen, Zihang Wang, Yao Yang, Jiarui Zhang, Chun Cao, Jingwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12031)  

**Abstract**: Recent advancements in large language models (LLMs) have sparked considerable interest in automated theorem proving and a prominent line of research integrates stepwise LLM-based provers into tree search. In this paper, we introduce a novel proof-state exploration approach for training data synthesis, designed to produce diverse tactics across a wide range of intermediate proof states, thereby facilitating effective one-shot fine-tuning of LLM as the policy model. We also propose an adaptive beam size strategy, which effectively takes advantage of our data synthesis method and achieves a trade-off between exploration and exploitation during tree search. Evaluations on the MiniF2F and ProofNet benchmarks demonstrate that our method outperforms strong baselines under the stringent Pass@1 metric, attaining an average pass rate of $60.74\%$ on MiniF2F and $21.18\%$ on ProofNet. These results underscore the impact of large-scale synthetic data in advancing automated theorem proving. 

**Abstract (ZH)**: Recent advancements in large language models (LLMs) have sparked considerable interest in automated theorem proving, and a prominent line of research integrates stepwise LLM-based provers into tree search. In this paper, we introduce a novel proof-state exploration approach for training data synthesis, designed to produce diverse tactics across a wide range of intermediate proof states, thereby facilitating effective one-shot fine-tuning of LLM as the policy model. We also propose an adaptive beam size strategy, which effectively takes advantage of our data synthesis method and achieves a trade-off between exploration and exploitation during tree search. Evaluations on the MiniF2F and ProofNet benchmarks demonstrate that our method outperforms strong baselines under the stringent Pass@1 metric, attaining an average pass rate of $60.74\%$ on MiniF2F and $21.18\%$ on ProofNet. These results underscore the impact of large-scale synthetic data in advancing automated theorem proving. 

---
# Empowering Sustainable Finance with Artificial Intelligence: A Framework for Responsible Implementation 

**Title (ZH)**: 以人工智能赋能可持续金融：负责任实施的框架 

**Authors**: Georgios Pavlidis  

**Link**: [PDF](https://arxiv.org/pdf/2505.12012)  

**Abstract**: This chapter explores the convergence of two major developments: the rise of environmental, social, and governance (ESG) investing and the exponential growth of artificial intelligence (AI) technology. The increased demand for diverse ESG instruments, such as green and ESG-linked loans, will be aligned with the rapid growth of the global AI market, which is expected to be worth $1,394.30 billion by 2029. AI can assist in identifying and pricing climate risks, setting more ambitious ESG goals, and advancing sustainable finance decisions. However, delegating sustainable finance decisions to AI poses serious risks, and new principles and rules for AI and ESG investing are necessary to mitigate these risks. This chapter highlights the challenges associated with norm-setting initiatives and stresses the need for the fine-tuning of the principles of legitimacy, oversight and verification, transparency, and explainability. Finally, the chapter contends that integrating AI into ESG non-financial reporting necessitates a heightened sense of responsibility and the establishment of fundamental guiding principles within the spheres of AI and ESG investing. 

**Abstract (ZH)**: 本章探讨了两大发展趋势的交汇：环境、社会和治理（ESG）投资的兴起与人工智能（AI）技术的指数级增长。不断增加的多样化ESG工具的需求，如绿色贷款和ESG挂钩贷款，将与全球AI市场的快速增长相一致，预计到2029年，全球AI市场规模将达到13943亿美元。AI可以帮助识别和定价气候风险，设定更具雄心的ESG目标，并促进可持续金融决策。然而，将可持续金融决策委托给AI存在严重风险，需要制定新的AI和ESG投资原则和规则以减轻这些风险。本章强调规范制定倡议面临的挑战，并强调需要对合法性的原则、监督和验证、透明度以及可解释性进行精细调整。最后，本章认为将AI整合到ESG非财务报告中需要提高责任感，并在AI和ESG投资领域建立基本指导原则。 

---
# SOCIA: An End-to-End Agentic Framework for Automated Cyber-Physical-Social Simulator Generation 

**Title (ZH)**: SOCIA：一种端到端的自主框架，用于自动生成网络-物理-社会仿真器 

**Authors**: Yuncheng Hua, Ji Miao, Mehdi Jafari, Jianxiang Xie, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2505.12006)  

**Abstract**: This paper introduces SOCIA (Simulation Orchestration for Cyber-physical-social Intelligence and Agents), a novel end-to-end framework leveraging Large Language Model (LLM)-based multi-agent systems to automate the generation of high-fidelity Cyber-Physical-Social (CPS) simulators. Addressing the challenges of labor-intensive manual simulator development and complex data calibration, SOCIA integrates a centralized orchestration manager that coordinates specialized agents for tasks including data comprehension, code generation, simulation execution, and iterative evaluation-feedback loops. Through empirical evaluations across diverse CPS tasks, such as mask adoption behavior simulation (social), personal mobility generation (physical), and user modeling (cyber), SOCIA demonstrates its ability to produce high-fidelity, scalable simulations with reduced human intervention. These results highlight SOCIA's potential to offer a scalable solution for studying complex CPS phenomena 

**Abstract (ZH)**: 本论文介绍了SOCIA（基于大型语言模型的多智能体系统仿真编排），一种新颖的端到端框架，利用基于大型语言模型的多智能体系统自动化生成高保真度的网络物理社会（CPS）仿真。通过集中编排管理者协调专门智能体进行任务，包括数据理解、代码生成、仿真执行和迭代评估反馈循环，SOCIA解决了劳动密集型的手动仿真开发和复杂数据校准的挑战。通过在多样化的CPS任务，如口罩佩戴行为仿真（社会）、个人移动生成（物理）和用户建模（ cyber）方面的实证评估，SOCIA展示了其生成高保真度、可扩展仿真并减少人力干预的能力。这些结果突显了SOCIA在研究复杂CPS现象方面的潜在可扩展解决方案。 

---
# Interactional Fairness in LLM Multi-Agent Systems: An Evaluation Framework 

**Title (ZH)**: LLM多智能体系统中的互动公平性评估框架 

**Authors**: Ruta Binkyte  

**Link**: [PDF](https://arxiv.org/pdf/2505.12001)  

**Abstract**: As large language models (LLMs) are increasingly used in multi-agent systems, questions of fairness should extend beyond resource distribution and procedural design to include the fairness of how agents communicate. Drawing from organizational psychology, we introduce a novel framework for evaluating Interactional fairness encompassing Interpersonal fairness (IF) and Informational fairness (InfF) in LLM-based multi-agent systems (LLM-MAS). We extend the theoretical grounding of Interactional Fairness to non-sentient agents, reframing fairness as a socially interpretable signal rather than a subjective experience. We then adapt established tools from organizational justice research, including Colquitt's Organizational Justice Scale and the Critical Incident Technique, to measure fairness as a behavioral property of agent interaction. We validate our framework through a pilot study using controlled simulations of a resource negotiation task. We systematically manipulate tone, explanation quality, outcome inequality, and task framing (collaborative vs. competitive) to assess how IF influences agent behavior. Results show that tone and justification quality significantly affect acceptance decisions even when objective outcomes are held constant. In addition, the influence of IF vs. InfF varies with context. This work lays the foundation for fairness auditing and norm-sensitive alignment in LLM-MAS. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在多智能体系统中的广泛应用，公平性问题应从资源分配和程序设计扩展到包括代理间沟通的公平性。借鉴组织心理学，我们提出了一种新的框架，用于评估基于LLM的多智能体系统（LLM-MAS）中的交互公平性，该框架包括人际公平性（IF）和信息公平性（InfF）。我们将交互公平性的理论基础扩展到非有感知能力的代理上，重新定义公平性为一种社会可解析的信号，而非主观体验。随后，我们借鉴组织公平性研究中已有的工具，包括Colquitt的组织公平性量表和关键事件技术，以行为属性的形式衡量公平性。我们通过使用资源谈判任务控制性模拟的试点研究验证了该框架。我们系统地操控语气、解释质量、结果不平等性和任务框架（合作 vs. 竞争），以评估IF如何影响代理行为。结果表明，即使在客观结果保持不变的情况下，语气和解释质量对接受决策的影响仍然显著。此外，IF和InfF的影响因情境而异。这项工作为LLM-MAS中的公平性审计和规范敏感性对齐奠定了基础。 

---
# MRGRP: Empowering Courier Route Prediction in Food Delivery Service with Multi-Relational Graph 

**Title (ZH)**: MRGRP：增强食物配送服务中快递路由预测的多关系图方法 

**Authors**: Chang Liu, Huan Yan, Hongjie Sui, Haomin Wen, Yuan Yuan, Yuyang Han, Hongsen Liao, Xuetao Ding, Jinghua Hao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11999)  

**Abstract**: Instant food delivery has become one of the most popular web services worldwide due to its convenience in daily life. A fundamental challenge is accurately predicting courier routes to optimize task dispatch and improve delivery efficiency. This enhances satisfaction for couriers and users and increases platform profitability. The current heuristic prediction method uses only limited human-selected task features and ignores couriers preferences, causing suboptimal results. Additionally, existing learning-based methods do not fully capture the diverse factors influencing courier decisions or the complex relationships among them. To address this, we propose a Multi-Relational Graph-based Route Prediction (MRGRP) method that models fine-grained correlations among tasks affecting courier decisions for accurate prediction. We encode spatial and temporal proximity, along with pickup-delivery relationships, into a multi-relational graph and design a GraphFormer architecture to capture these complex connections. We also introduce a route decoder that leverages courier information and dynamic distance and time contexts for prediction, using existing route solutions as references to improve outcomes. Experiments show our model achieves state-of-the-art route prediction on offline data from cities of various sizes. Deployed on the Meituan Turing platform, it surpasses the current heuristic algorithm, reaching a high route prediction accuracy of 0.819, essential for courier and user satisfaction in instant food delivery. 

**Abstract (ZH)**: 基于多关系图的递送员路线预测方法（MRGRP） 

---
# Solve-Detect-Verify: Inference-Time Scaling with Flexible Generative Verifier 

**Title (ZH)**: 解-检测-验证：灵活生成验证器的推理时扩展方法 

**Authors**: Jianyuan Zhong, Zeju Li, Zhijian Xu, Xiangyu Wen, Kezhi Li, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11966)  

**Abstract**: Large Language Model (LLM) reasoning for complex tasks inherently involves a trade-off between solution accuracy and computational efficiency. The subsequent step of verification, while intended to improve performance, further complicates this landscape by introducing its own challenging trade-off: sophisticated Generative Reward Models (GenRMs) can be computationally prohibitive if naively integrated with LLMs at test-time, while simpler, faster methods may lack reliability. To overcome these challenges, we introduce FlexiVe, a novel generative verifier that flexibly balances computational resources between rapid, reliable fast thinking and meticulous slow thinking using a Flexible Allocation of Verification Budget strategy. We further propose the Solve-Detect-Verify pipeline, an efficient inference-time scaling framework that intelligently integrates FlexiVe, proactively identifying solution completion points to trigger targeted verification and provide focused solver feedback. Experiments show FlexiVe achieves superior accuracy in pinpointing errors within reasoning traces on ProcessBench. Furthermore, on challenging mathematical reasoning benchmarks (AIME 2024, AIME 2025, and CNMO), our full approach outperforms baselines like self-consistency in reasoning accuracy and inference efficiency. Our system offers a scalable and effective solution to enhance LLM reasoning at test time. 

**Abstract (ZH)**: 大规模语言模型（LLM）在复杂任务推理中固有地涉及求解准确性和计算效率之间的权衡。验证步骤虽然旨在提升性能，但也引入了自身的挑战性权衡：如果在测试时天真地将复杂的生成奖励模型与LLM集成，可能会导致计算成本过高，而简单快速的方法可能缺乏可靠性。为克服这些挑战，我们提出了FlexiVe，一种新颖的生成验证器，通过灵活分配验证预算策略在快速可靠直觉思考和仔细慎重思考之间灵活平衡计算资源。此外，我们进一步提出了Solve-Detect-Verify流水线，这是一种高效的推理时扩展框架，智能地集成FlexiVe，主动识别解决方案完成点，触发目标验证并提供集中解算器反馈。实验表明，FlexiVe在ProcessBench中的推理跟踪中更精确地定位错误。在具有挑战性的数学推理基准测试（AIME 2024、AIME 2025和CNMO）上，我们的完整方法在推理准确性和推理效率方面优于自一致性等基线。我们的系统为提高测试时LLM推理提供了一种可扩展且有效的解决方案。 

---
# CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World 

**Title (ZH)**: CrafText基准：推进复杂多模态开放式指令跟随研究 

**Authors**: Zoya Volovikova, Gregory Gorbov, Petr Kuderov, Aleksandr I. Panov, Alexey Skrynnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.11962)  

**Abstract**: Following instructions in real-world conditions requires the ability to adapt to the world's volatility and entanglement: the environment is dynamic and unpredictable, instructions can be linguistically complex with diverse vocabulary, and the number of possible goals an agent may encounter is vast. Despite extensive research in this area, most studies are conducted in static environments with simple instructions and a limited vocabulary, making it difficult to assess agent performance in more diverse and challenging settings. To address this gap, we introduce CrafText, a benchmark for evaluating instruction following in a multimodal environment with diverse instructions and dynamic interactions. CrafText includes 3,924 instructions with 3,423 unique words, covering Localization, Conditional, Building, and Achievement tasks. Additionally, we propose an evaluation protocol that measures an agent's ability to generalize to novel instruction formulations and dynamically evolving task configurations, providing a rigorous test of both linguistic understanding and adaptive decision-making. 

**Abstract (ZH)**: 在现实世界条件下遵循指令要求适应世界的波动性和复杂性：环境动态且不可预测，指令可能具有复杂的语言结构和多样的词汇，代理可能遇到的可能目标数量庞大。尽管在此领域进行了大量研究，但大多数研究在静态环境中进行，涉及简单的指令和有限的词汇量，这使得在更多样化和更具挑战性的环境中评估代理性能变得困难。为填补这一空白，我们引入了CrafText，这是一个用于评估多模态环境中多样指令和动态交互的指令遵循基准。CrafText 包含 3,924 条指令和 3,423 个独特单词，涵盖了定位、条件、构建和成就任务。此外，我们还提出了一种评估协议，用于测量代理适应新型指令表达和动态演化任务配置的能力，从而对语言理解和适应性决策制定进行严格的测试。 

---
# LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners 

**Title (ZH)**: 终身学习者Agent评估基准：评估LLM Agents的终身学习能力 

**Authors**: Junhao Zheng, Xidi Cai, Qiuke Li, Duzhen Zhang, ZhongZhi Li, Yingying Zhang, Le Song, Qianli Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.11942)  

**Abstract**: Lifelong learning is essential for intelligent agents operating in dynamic environments. Current large language model (LLM)-based agents, however, remain stateless and unable to accumulate or transfer knowledge over time. Existing benchmarks treat agents as static systems and fail to evaluate lifelong learning capabilities. We present LifelongAgentBench, the first unified benchmark designed to systematically assess the lifelong learning ability of LLM agents. It provides skill-grounded, interdependent tasks across three interactive environments, Database, Operating System, and Knowledge Graph, with automatic label verification, reproducibility, and modular extensibility. Extensive experiments reveal that conventional experience replay has limited effectiveness for LLM agents due to irrelevant information and context length constraints. We further introduce a group self-consistency mechanism that significantly improves lifelong learning performance. We hope LifelongAgentBench will advance the development of adaptive, memory-capable LLM agents. 

**Abstract (ZH)**: 终身学习对于在动态环境中操作的智能代理至关重要。当前基于大规模语言模型（LLM）的代理仍然无状态，无法随时间积累或转移知识。现有的基准将代理视为静态系统，未能评估终身学习能力。我们提出了LifelongAgentBench，这是首个用于系统评估LLM代理终身学习能力的统一基准。它提供了跨数据库、操作系统和知识图谱三个交互环境的技能导向且相互依赖的任务，并具备自动标签验证、可重复性和模块可扩展性。广泛实验证明，传统的经验回放对LLM代理的效果有限，原因在于无关信息和上下文长度的限制。我们进一步引入了一种群体自一致性机制，显著提高了终身学习性能。我们期望LifelongAgentBench能够促进适应性和记忆能力的LLM代理的发展。 

---
# From Recall to Reasoning: Automated Question Generation for Deeper Math Learning through Large Language Models 

**Title (ZH)**: 从回忆到推理：通过大规模语言模型实现更深层次数学学习的自动问题生成 

**Authors**: Yongan Yu, Alexandre Krantz, Nikki G. Lobczowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.11899)  

**Abstract**: Educators have started to turn to Generative AI (GenAI) to help create new course content, but little is known about how they should do so. In this project, we investigated the first steps for optimizing content creation for advanced math. In particular, we looked at the ability of GenAI to produce high-quality practice problems that are relevant to the course content. We conducted two studies to: (1) explore the capabilities of current versions of publicly available GenAI and (2) develop an improved framework to address the limitations we found. Our results showed that GenAI can create math problems at various levels of quality with minimal support, but that providing examples and relevant content results in better quality outputs. This research can help educators decide the ideal way to adopt GenAI in their workflows, to create more effective educational experiences for students. 

**Abstract (ZH)**: 教育者已经开始利用生成式人工智能（GenAI）来创建新的课程内容，但尚不清楚应该如何操作。本项目研究了优化高级数学内容创作的第一步。特别地，我们探讨了GenAI生成与课程内容相关且高质量练习题的能力。我们进行了两项研究：（1）探索当前公开版本GenAI的能力；（2）开发改进框架以解决我们发现的限制。研究结果表明，GenAI在最少支持下可以生成不同质量级别的数学题目，但在生成高质量输出时提供建例和相关内容效果更佳。本研究可帮助教育者决定在工作流程中采用GenAI的最佳方式，从而为学生创建更有效的教育体验。 

---
# Position Paper: Bounded Alignment: What (Not) To Expect From AGI Agents 

**Title (ZH)**: 位置论文：有界对齐：从AGI代理中期望（不期望）什么 

**Authors**: Ali A. Minai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11866)  

**Abstract**: The issues of AI risk and AI safety are becoming critical as the prospect of artificial general intelligence (AGI) looms larger. The emergence of extremely large and capable generative models has led to alarming predictions and created a stir from boardrooms to legislatures. As a result, AI alignment has emerged as one of the most important areas in AI research. The goal of this position paper is to argue that the currently dominant vision of AGI in the AI and machine learning (AI/ML) community needs to evolve, and that expectations and metrics for its safety must be informed much more by our understanding of the only existing instance of general intelligence, i.e., the intelligence found in animals, and especially in humans. This change in perspective will lead to a more realistic view of the technology, and allow for better policy decisions. 

**Abstract (ZH)**: 人工智能风险与安全问题随通用人工智能（AGI）的到来日益关键：AI/ML社区的AGI主导愿景需要演进 

---
# Fair-PP: A Synthetic Dataset for Aligning LLM with Personalized Preferences of Social Equity 

**Title (ZH)**: Fair-PP：一个合成数据集，用于 Alignment with 社会公平个性化偏好的大型语言模型 

**Authors**: Qi Zhou, Jie Zhang, Dongxia Wang, Qiang Liu, Tianlin Li, Jin Song Dong, Wenhai Wang, Qing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11861)  

**Abstract**: Human preference plays a crucial role in the refinement of large language models (LLMs). However, collecting human preference feedback is costly and most existing datasets neglect the correlation between personalization and preferences. To address this issue, we introduce Fair-PP, a synthetic dataset of personalized preferences targeting social equity, derived from real-world social survey data, which includes 28 social groups, 98 equity topics, and 5 personal preference dimensions. Leveraging GPT-4o-mini, we engage in role-playing based on seven representative persona portrayals guided by existing social survey data, yielding a total of 238,623 preference records. Through Fair-PP, we also contribute (i) An automated framework for generating preference data, along with a more fine-grained dataset of personalized preferences; (ii) analysis of the positioning of the existing mainstream LLMs across five major global regions within the personalized preference space; and (iii) a sample reweighting method for personalized preference alignment, enabling alignment with a target persona while maximizing the divergence from other personas. Empirical experiments show our method outperforms the baselines. 

**Abstract (ZH)**: 人类偏好在大型语言模型的精炼过程中扮演着至关重要的角色。然而，收集人类偏好反馈的成本较高，且现有大多数数据集忽略了个性化与偏好之间的关联。为解决这一问题，我们引入了Fair-PP，这是一个基于真实世界社会调查数据的合成个性化偏好数据集，旨在促进社会公平，该数据集包含28个社会群体、98项公平主题和5个偏好维度。借助GPT-4o-mini，我们基于现有社会调查数据中的七个代表性人物画像进行角色扮演，生成了总计238,623条偏好记录。通过Fair-PP，我们还贡献了（i）生成偏好数据的自动化框架，以及更细致的个性化偏好数据集；（ii）对主流大型语言模型在全球五大地区内的个性化偏好空间中的定位进行分析；以及（iii）一种个性化偏好对齐的样本加权方法，该方法允许与目标人物画像对齐的同时最大程度地远离其他人物画像。实证实验表明，我们的方法优于基线方法。 

---
# Evaluating the Logical Reasoning Abilities of Large Reasoning Models 

**Title (ZH)**: 评估大型推理模型的逻辑推理能力 

**Authors**: Hanmeng Liu, Yiran Ding, Zhizhang Fu, Chaoli Zhang, Xiaozhang Liu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11854)  

**Abstract**: Large reasoning models, often post-trained on long chain-of-thought (long CoT) data with reinforcement learning, achieve state-of-the-art performance on mathematical, coding, and domain-specific reasoning benchmarks. However, their logical reasoning capabilities - fundamental to human cognition and independent of domain knowledge - remain understudied. To address this gap, we introduce LogiEval, a holistic benchmark for evaluating logical reasoning in large reasoning models. LogiEval spans diverse reasoning types (deductive, inductive, analogical, and abductive) and task formats (e.g., logical sequence, argument analysis), sourced from high-quality human examinations (e.g., LSAT, GMAT). Our experiments demonstrate that modern reasoning models excel at 4-choice argument analysis problems and analogical reasoning, surpassing human performance, yet exhibit uneven capabilities across reasoning types and formats, highlighting limitations in their generalization. Our analysis reveals that human performance does not mirror model failure distributions. To foster further research, we curate LogiEval-Hard, a challenging subset identified through a novel screening paradigm where small-model failures (Qwen3-30B-A3B) reliably predict difficulties for larger models. Modern models show striking, consistent failures on LogiEval-Hard. This demonstrates that fundamental reasoning bottlenecks persist across model scales, and establishes LogiEval-Hard as both a diagnostic tool and a rigorous testbed for advancing logical reasoning in LLMs. 

**Abstract (ZH)**: 大规模推理模型通过强化学习在长链推理数据上进行后训练，在数学、编程和领域特定推理基准测试中取得了最先进的性能。然而，这些模型的基础逻辑推理能力——这是人类认知的基本要素，并且独立于领域知识——仍未得到充分研究。为了填补这一空白，我们引入了LogiEval，一个全面的基准测试，用于评估大规模推理模型的逻辑推理能力。LogiEval涵盖了多样化的推理类型（演绎、归纳、类比和溯因）和任务格式（例如，逻辑序列、论证分析），来源于高质量的人类考试（例如，LSAT、GMAT）。我们的实验表明，现代推理模型在4选1论证分析问题和类比推理方面表现优异，超过了人类的表现，但其在不同推理类型和格式上的能力参差不齐，突显了其泛化能力的局限性。我们分析发现，人类的表现并不反映模型的失败分布。为了促进进一步的研究，我们通过一种新的筛选范式（小模型失败预测大模型困难）精心挑选了LogiEval-Hard，这是具有挑战性的子集。现代模型在LogiEval-Hard上表现出显著且一致的失败。这表明基本的推理瓶颈存在于不同规模的模型中，并确立了LogiEval-Hard作为诊断工具和严格测试平台，以推动大语言模型中逻辑推理的进展。 

---
# VeriReason: Reinforcement Learning with Testbench Feedback for Reasoning-Enhanced Verilog Generation 

**Title (ZH)**: VeriReason: 基于测试台反馈的强化学习与推理增强的Verilog生成 

**Authors**: Yiting Wang, Guoheng Sun, Wanghao Ye, Gang Qu, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11849)  

**Abstract**: Automating Register Transfer Level (RTL) code generation using Large Language Models (LLMs) offers substantial promise for streamlining digital circuit design and reducing human effort. However, current LLM-based approaches face significant challenges with training data scarcity, poor specification-code alignment, lack of verification mechanisms, and balancing generalization with specialization. Inspired by DeepSeek-R1, we introduce VeriReason, a framework integrating supervised fine-tuning with Guided Reward Proximal Optimization (GRPO) reinforcement learning for RTL generation. Using curated training examples and a feedback-driven reward model, VeriReason combines testbench evaluations with structural heuristics while embedding self-checking capabilities for autonomous error correction. On the VerilogEval Benchmark, VeriReason delivers significant improvements: achieving 83.1% functional correctness on the VerilogEval Machine benchmark, substantially outperforming both comparable-sized models and much larger commercial systems like GPT-4 Turbo. Additionally, our approach demonstrates up to a 2.8X increase in first-attempt functional correctness compared to baseline methods and exhibits robust generalization to unseen designs. To our knowledge, VeriReason represents the first system to successfully integrate explicit reasoning capabilities with reinforcement learning for Verilog generation, establishing a new state-of-the-art for automated RTL synthesis. The models and datasets are available at: this https URL Code is Available at: this https URL 

**Abstract (ZH)**: 使用大型语言模型（LLMs）自动化寄存器传输级（RTL）代码生成在简化数字电路设计和减少人力方面具有巨大潜力。然而，当前基于LLM的方法面临训练数据稀缺、规范代码对齐不佳、缺乏验证机制以及泛化与专门化之间平衡的显著挑战。受DeepSeek-R1启发，我们提出了一种VeriReason框架，该框架结合了监督微调与引导奖励近端优化（GRPO）强化学习，用于RTL生成。通过精选的训练示例和反馈驱动的奖励模型，VeriReason结合了测试平台评估与结构启发式方法，同时嵌入自检功能以实现自主错误校正。在VerilogEval基准测试中，VeriReason实现了显著改进：在VerilogEval Machine基准测试中达到83.1%的功能正确性，大幅优于同等大小的模型及更大规模的商用系统如GPT-4 Turbo。此外，我们的方法在首次尝试时的功能正确性上显示最高可达2.8倍的提高，且在未见过的设计上表现出色的泛化能力。据我们所知，VeriReason是第一个将显式推理能力与Verilog生成的强化学习集成的系统，确立了自动化RTL综合的新前沿。模型和数据集可从以下链接获取：this https URL代码可在以下链接获取：this https URL 

---
# On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study 

**Title (ZH)**: LLMs在反事实推理中的恰当性：一项分解性研究 

**Authors**: Shuai Yang, Qi Yang, Luoxi Tang, Jeremy Blackburn, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2505.11839)  

**Abstract**: Counterfactual reasoning has emerged as a crucial technique for generalizing the reasoning capabilities of large language models (LLMs). By generating and analyzing counterfactual scenarios, researchers can assess the adaptability and reliability of model decision-making. Although prior work has shown that LLMs often struggle with counterfactual reasoning, it remains unclear which factors most significantly impede their performance across different tasks and modalities. In this paper, we propose a decompositional strategy that breaks down the counterfactual generation from causality construction to the reasoning over counterfactual interventions. To support decompositional analysis, we investigate 11 datasets spanning diverse tasks, including natural language understanding, mathematics, programming, and vision-language tasks. Through extensive evaluations, we characterize LLM behavior across each decompositional stage and identify how modality type and intermediate reasoning influence performance. By establishing a structured framework for analyzing counterfactual reasoning, this work contributes to the development of more reliable LLM-based reasoning systems and informs future elicitation strategies. 

**Abstract (ZH)**: 基于因果构建的分解策略：探究大规模语言模型在反事实推理中的表现与影响因素 

---
# ToLeaP: Rethinking Development of Tool Learning with Large Language Models 

**Title (ZH)**: ToLeP: 重新思考工具学习ewith大型语言模型的发展 

**Authors**: Haotian Chen, Zijun Song, Boye Niu, Ke Zhang, Litu Ou, Yaxi Lu, Zhong Zhang, Xin Cong, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.11833)  

**Abstract**: Tool learning, which enables large language models (LLMs) to utilize external tools effectively, has garnered increasing attention for its potential to revolutionize productivity across industries. Despite rapid development in tool learning, key challenges and opportunities remain understudied, limiting deeper insights and future advancements. In this paper, we investigate the tool learning ability of 41 prevalent LLMs by reproducing 33 benchmarks and enabling one-click evaluation for seven of them, forming a Tool Learning Platform named ToLeaP. We also collect 21 out of 33 potential training datasets to facilitate future exploration. After analyzing over 3,000 bad cases of 41 LLMs based on ToLeaP, we identify four main critical challenges: (1) benchmark limitations induce both the neglect and lack of (2) autonomous learning, (3) generalization, and (4) long-horizon task-solving capabilities of LLMs. To aid future advancements, we take a step further toward exploring potential directions, namely (1) real-world benchmark construction, (2) compatibility-aware autonomous learning, (3) rationale learning by thinking, and (4) identifying and recalling key clues. The preliminary experiments demonstrate their effectiveness, highlighting the need for further research and exploration. 

**Abstract (ZH)**: 大型语言模型的工具学习能力研究：ToLeaP平台及其挑战探索 

---
# ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems 

**Title (ZH)**: ARC-AGI-2：前沿AI推理系统的新挑战 

**Authors**: Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, Henry Pinkard  

**Link**: [PDF](https://arxiv.org/pdf/2505.11831)  

**Abstract**: The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI), introduced in 2019, established a challenging benchmark for evaluating the general fluid intelligence of artificial systems via a set of unique, novel tasks only requiring minimal prior knowledge. While ARC-AGI has spurred significant research activity over the past five years, recent AI progress calls for benchmarks capable of finer-grained evaluation at higher levels of cognitive complexity. We introduce ARC-AGI-2, an upgraded version of the benchmark. ARC-AGI-2 preserves the input-output pair task format of its predecessor, ensuring continuity for researchers. It incorporates a newly curated and expanded set of tasks specifically designed to provide a more granular signal to assess abstract reasoning and problem-solving abilities at higher levels of fluid intelligence. To contextualize the difficulty and characteristics of ARC-AGI-2, we present extensive results from human testing, providing a robust baseline that highlights the benchmark's accessibility to human intelligence, yet difficulty for current AI systems. ARC-AGI-2 aims to serve as a next-generation tool for rigorously measuring progress towards more general and human-like AI capabilities. 

**Abstract (ZH)**: 人工通用智能的抽象与推理语料库（ARC-AGI）升级版：ARC-AGI-2 

---
# ChatHTN: Interleaving Approximate (LLM) and Symbolic HTN Planning 

**Title (ZH)**: ChatHTN: 混合 interleaving 近似（LLM）和符号 HTN 规划 

**Authors**: Hector Munoz-Avila, David W. Aha, Paola Rizzo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11814)  

**Abstract**: We introduce ChatHTN, a Hierarchical Task Network (HTN) planner that combines symbolic HTN planning techniques with queries to ChatGPT to approximate solutions in the form of task decompositions. The resulting hierarchies interleave task decompositions generated by symbolic HTN planning with those generated by ChatGPT. Despite the approximate nature of the results generates by ChatGPT, ChatHTN is provably sound; any plan it generates correctly achieves the input tasks. We demonstrate this property with an open-source implementation of our system. 

**Abstract (ZH)**: ChatHTN：结合符号化HTN规划技术与ChatGPT查询的分层任务网络规划器 

---
# VITA: Versatile Time Representation Learning for Temporal Hyper-Relational Knowledge Graphs 

**Title (ZH)**: VITA：时间泛化时空超关系知识图谱表示学习 

**Authors**: ChongIn Un, Yuhuan Lu, Tianyue Yang, Dingqi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11803)  

**Abstract**: Knowledge graphs (KGs) have become an effective paradigm for managing real-world facts, which are not only complex but also dynamically evolve over time. The temporal validity of facts often serves as a strong clue in downstream link prediction tasks, which predicts a missing element in a fact. Traditional link prediction techniques on temporal KGs either consider a sequence of temporal snapshots of KGs with an ad-hoc defined time interval or expand a temporal fact over its validity period under a predefined time granularity; these approaches not only suffer from the sensitivity of the selection of time interval/granularity, but also face the computational challenges when handling facts with long (even infinite) validity. Although the recent hyper-relational KGs represent the temporal validity of a fact as qualifiers describing the fact, it is still suboptimal due to its ignorance of the infinite validity of some facts and the insufficient information encoded from the qualifiers about the temporal validity. Against this background, we propose VITA, a $\underline{V}$ersatile t$\underline{I}$me represen$\underline{TA}$tion learning method for temporal hyper-relational knowledge graphs. We first propose a versatile time representation that can flexibly accommodate all four types of temporal validity of facts (i.e., since, until, period, time-invariant), and then design VITA to effectively learn the time information in both aspects of time value and timespan to boost the link prediction performance. We conduct a thorough evaluation of VITA compared to a sizable collection of baselines on real-world KG datasets. Results show that VITA outperforms the best-performing baselines in various link prediction tasks (predicting missing entities, relations, time, and other numeric literals) by up to 75.3%. Ablation studies and a case study also support our key design choices. 

**Abstract (ZH)**: 面向临时超关系知识图谱的通用时间表示学习方法VITA 

---
# Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling 

**Title (ZH)**: 基于求解器的RL：为真实的优化建模grounding大型语言模型 

**Authors**: Yitian Chen, Jingfan Xia, Siyu Shao, Dongdong Ge, Yinyu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.11792)  

**Abstract**: Optimization modeling is fundamental to decision-making across diverse this http URL progress in automating optimization formulation from natural language descriptions, Large Language Models (LLMs) often struggle to generate formally correct and usable models due to hallucinations, posing a challenge for reliable automation. Inspired by the success of Reinforcement Learning (RL) in enhancing Large Reasoning Models, we present Solver-Informed Reinforcement Learning (SIRL).This novel framework leverages external optimization solvers as verifiable reward mechanisms to significantly improve the authenticity of LLMs for optimization this http URL as precise verifiers, these solvers automatically assess the executable code and the instance-level mathematical model represented by the associated LP file, yielding precise and comprehensive feedback signals -- including syntax, feasibility, and solution quality that directly inform the RL process. This automated verification process, powered by classic optimization solvers, also underpins our instance-enhanced self-consistency method to synthesize high-quality training data. Extensive experiments on diverse public benchmarks demonstrate that SIRL achieves state-of-the-art performance, substantially outperforming existing methods in generating accurate and executable optimization models. 

**Abstract (ZH)**: 基于求解器指导的强化学习方法：优化建模中的自动化与可靠化 

---
# A Review and Analysis of a Parallel Approach for Decision Tree Learning from Large Data Streams 

**Title (ZH)**: 大型数据流中决策树学习的并行方法综述与分析 

**Authors**: Zeinab Shiralizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11780)  

**Abstract**: This work studies one of the parallel decision tree learning algorithms, pdsCART, designed for scalable and efficient data analysis. The method incorporates three core capabilities. First, it supports real-time learning from data streams, allowing trees to be constructed incrementally. Second, it enables parallel processing of high-volume streaming data, making it well-suited for large-scale applications. Third, the algorithm integrates seamlessly into the MapReduce framework, ensuring compatibility with distributed computing environments. In what follows, we present the algorithm's key components along with results highlighting its performance and scalability. 

**Abstract (ZH)**: 本研究探讨了一种面向可扩展和高效数据分析的并行决策树学习算法pdsCART。该方法集成了三项核心能力：首先，它支持从数据流中进行实时学习，允许树的增量构建；其次，它能够并行处理高volume数据流，使其适用于大规模应用；第三，该算法无缝集成到MapReduce框架中，确保与分布式计算环境的兼容性。随后，我们介绍了该算法的关键组件，并展示了其性能和可扩展性的结果。 

---
# Diverging Towards Hallucination: Detection of Failures in Vision-Language Models via Multi-token Aggregation 

**Title (ZH)**: 走向幻觉的发散：基于多令牌聚合的大规模视觉-语言模型故障检测 

**Authors**: Geigh Zollicoffer, Minh Vu, Manish Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11741)  

**Abstract**: Vision-language models (VLMs) now rival human performance on many multimodal tasks, yet they still hallucinate objects or generate unsafe text. Current hallucination detectors, e.g., single-token linear probing (SLP) and P(True), typically analyze only the logit of the first generated token or just its highest scoring component overlooking richer signals embedded within earlier token distributions. We demonstrate that analyzing the complete sequence of early logits potentially provides substantially more diagnostic information. We emphasize that hallucinations may only emerge after several tokens, as subtle inconsistencies accumulate over time. By analyzing the Kullback-Leibler (KL) divergence between logits corresponding to hallucinated and non-hallucinated tokens, we underscore the importance of incorporating later-token logits to more accurately capture the reliability dynamics of VLMs. In response, we introduce Multi-Token Reliability Estimation (MTRE), a lightweight, white-box method that aggregates logits from the first ten tokens using multi-token log-likelihood ratios and self-attention. Despite the challenges posed by large vocabulary sizes and long logit sequences, MTRE remains efficient and tractable. On MAD-Bench, MM-SafetyBench, MathVista, and four compositional-geometry benchmarks, MTRE improves AUROC by 9.4 +/- 1.3 points over SLP and by 12.1 +/- 1.7 points over P(True), setting a new state-of-the-art in hallucination detection for open-source VLMs. 

**Abstract (ZH)**: 基于视觉-语言模型的多令牌可靠性估计在幻觉检测中的应用 

---
# Automated Real-time Assessment of Intracranial Hemorrhage Detection AI Using an Ensembled Monitoring Model (EMM) 

**Title (ZH)**: 基于集成监控模型（EMM）的颅内出血检测人工智能自动化实时评估 

**Authors**: Zhongnan Fang, Andrew Johnston, Lina Cheuy, Hye Sun Na, Magdalini Paschali, Camila Gonzalez, Bonnie A. Armstrong, Arogya Koirala, Derrick Laurel, Andrew Walker Campion, Michael Iv, Akshay S. Chaudhari, David B. Larson  

**Link**: [PDF](https://arxiv.org/pdf/2505.11738)  

**Abstract**: Artificial intelligence (AI) tools for radiology are commonly unmonitored once deployed. The lack of real-time case-by-case assessments of AI prediction confidence requires users to independently distinguish between trustworthy and unreliable AI predictions, which increases cognitive burden, reduces productivity, and potentially leads to misdiagnoses. To address these challenges, we introduce Ensembled Monitoring Model (EMM), a framework inspired by clinical consensus practices using multiple expert reviews. Designed specifically for black-box commercial AI products, EMM operates independently without requiring access to internal AI components or intermediate outputs, while still providing robust confidence measurements. Using intracranial hemorrhage detection as our test case on a large, diverse dataset of 2919 studies, we demonstrate that EMM successfully categorizes confidence in the AI-generated prediction, suggesting different actions and helping improve the overall performance of AI tools to ultimately reduce cognitive burden. Importantly, we provide key technical considerations and best practices for successfully translating EMM into clinical settings. 

**Abstract (ZH)**: 人工智能工具在放射学中的监控模型（Ensembled Monitoring Model，EMM）：一种基于多专家共识的框架 

---
# Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling 

**Title (ZH)**: 重新思考计算高效测试时扩展下的最优验证粒度 

**Authors**: Hao Mark Chen, Guanxi Lu, Yasuyuki Okoshi, Zhiwen Mo, Masato Motomura, Hongxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11730)  

**Abstract**: Test-time scaling (TTS) has proven effective in enhancing the reasoning capabilities of large language models (LLMs). Verification plays a key role in TTS, simultaneously influencing (1) reasoning performance and (2) compute efficiency, due to the quality and computational cost of verification. In this work, we challenge the conventional paradigms of verification, and make the first attempt toward systematically investigating the impact of verification granularity-that is, how frequently the verifier is invoked during generation, beyond verifying only the final output or individual generation steps. To this end, we introduce Variable Granularity Search (VG-Search), a unified algorithm that generalizes beam search and Best-of-N sampling via a tunable granularity parameter g. Extensive experiments with VG-Search under varying compute budgets, generator-verifier configurations, and task attributes reveal that dynamically selecting g can improve the compute efficiency and scaling behavior. Building on these findings, we propose adaptive VG-Search strategies that achieve accuracy gains of up to 3.1\% over Beam Search and 3.6\% over Best-of-N, while reducing FLOPs by over 52\%. We will open-source the code to support future research. 

**Abstract (ZH)**: Test-time Scaling (TTS) 经验证有效于提升大型语言模型（LLMs）的推理能力。验证在 TTS 中扮演关键角色，同时影响（1）推理性能和（2）计算效率，这取决于验证的质量和计算成本。在本工作中，我们挑战了传统的验证范式，并首次系统地探讨了验证粒度的影响——即生成过程中验证器被调用的频率，而不仅仅是验证最终输出或单独的生成步骤。为此，我们引入了可调粒度搜索（VG-Search），这是一种统一算法，通过可调节的粒度参数 g  generalizes 并扩展了束搜索和 Best-of-N 抽样。在不同计算预算、生成器-验证器配置和任务属性下进行的大量实验表明，动态选择 g 可以提高计算效率和扩展行为。基于这些发现，我们提出了适应性 VG-Search 策略，这些策略在束搜索的基础上实现了高达 3.1% 的准确率提升，在 Best-of-N 的基础上实现了 3.6% 的准确率提升，同时减少了超过 52% 的 FLOPs。我们将开源代码以支持未来的研究。 

---
# REMOR: Automated Peer Review Generation with LLM Reasoning and Multi-Objective Reinforcement Learning 

**Title (ZH)**: REMOR：基于LLM推理和多目标强化学习的自动化同伴评审生成 

**Authors**: Pawin Taechoyotin, Daniel Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2505.11718)  

**Abstract**: AI-based peer review systems tend to produce shallow and overpraising suggestions compared to human feedback. Here, we evaluate how well a reasoning LLM trained with multi-objective reinforcement learning (REMOR) can overcome these limitations. We start by designing a multi-aspect reward function that aligns with human evaluation of reviews. The aspects are related to the review itself (e.g., criticisms, novelty) and the relationship between the review and the manuscript (i.e., relevance). First, we perform supervised fine-tuning of DeepSeek-R1-Distill-Qwen-7B using LoRA on PeerRT, a new dataset of high-quality top AI conference reviews enriched with reasoning traces. We then apply Group Relative Policy Optimization (GRPO) to train two models: REMOR-H (with the human-aligned reward) and REMOR-U (with a uniform reward). Interestingly, the human-aligned reward penalizes aspects typically associated with strong reviews, leading REMOR-U to produce qualitatively more substantive feedback. Our results show that REMOR-U and REMOR-H achieve more than twice the average rewards of human reviews, non-reasoning state-of-the-art agentic multi-modal AI review systems, and general commercial LLM baselines. We found that while the best AI and human reviews are comparable in quality, REMOR avoids the long tail of low-quality human reviews. We discuss how reasoning is key to achieving these improvements and release the Human-aligned Peer Review Reward (HPRR) function, the Peer Review Reasoning-enriched Traces (PeerRT) dataset, and the REMOR models, which we believe can help spur progress in the area. 

**Abstract (ZH)**: 基于AI的同行评审系统倾向于产生浅薄且过度赞扬的建议，而人类反馈则不然。这里，我们评估了一种使用多目标强化学习（REMOR）训练的推理LLM如何克服这些局限。我们首先设计了一个多方面奖赏函数，该函数与人类对评审的评估相一致。这些方面与评审本身（如批评、新颖性）和评审与手稿的关系（相关性）相关。我们首先使用LoRA对DeepSeek-R1-Distill-Qwen-7B进行有监督的微调，并在包含推理痕迹的新优质顶级AI会议评审数据集PeerRT上进行。然后，我们应用组相对策略优化（GRPO）训练两个模型：REMOR-H（采用与人类对齐的奖赏）和REMOR-U（采用均匀的奖赏）。有趣的是，与人类对齐的奖赏惩罚了通常与高质量评审相关的方面，导致REMOR-U生成了更具实质性的反馈。我们的结果显示，REMOR-U和REMOR-H在平均奖赏方面超过了人类评审、无推理的最先进多模态AI评审系统以及通用商业LLM基线两倍以上。我们发现，虽然最佳AI和人类评审在质量上可比，但REMOR避免了低质量人类评审的尾部效应。我们讨论了推理为何对于实现这些改进至关重要，并发布了与人类对齐的同行评审奖赏函数（HPRR）、同伴评审推理增强痕迹（PeerRT）数据集以及REMOR模型，我们相信这将有助于推动该领域的发展。 

---
# DMN-Guided Prompting: A Low-Code Framework for Controlling LLM Behavior 

**Title (ZH)**: DMN引导提示：一种低代码LLM行为控制框架 

**Authors**: Shaghayegh Abedi, Amin Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2505.11701)  

**Abstract**: Large Language Models (LLMs) have shown considerable potential in automating decision logic within knowledge-intensive processes. However, their effectiveness largely depends on the strategy and quality of prompting. Since decision logic is typically embedded in prompts, it becomes challenging for end users to modify or refine it. Decision Model and Notation (DMN) offers a standardized graphical approach for defining decision logic in a structured, user-friendly manner. This paper introduces a DMN-guided prompting framework that breaks down complex decision logic into smaller, manageable components, guiding LLMs through structured decision pathways. We implemented the framework in a graduate-level course where students submitted assignments. The assignments and DMN models representing feedback instructions served as inputs to our framework. The instructor evaluated the generated feedback and labeled it for performance assessment. Our approach demonstrated promising results, outperforming chain-of-thought (CoT) prompting. Students also responded positively to the generated feedback, reporting high levels of perceived usefulness in a survey based on the Technology Acceptance Model. 

**Abstract (ZH)**: 大型语言模型（LLMs）在知识密集型过程中的决策逻辑自动化方面展现了显著潜力，但其效果很大程度上取决于提示策略和质量。由于决策逻辑通常嵌入在提示中，这使得终端用户修改或精炼它变得具有挑战性。决策模型与表示法（DMN）提供了一种标准化的图形化方法，以结构化和用户友好的方式定义决策逻辑。本文介绍了一种由DMN引导的提示框架，将复杂的决策逻辑分解为更小、更易于管理的组件，指导LLMs通过结构化的决策路径。我们在一门研究生课程中实现了该框架，学生提交作业，作业和表示反馈指令的DMN模型作为框架的输入。教师评估生成的反馈并据此进行绩效评估。我们的方法显示出令人鼓舞的结果，优于基于推理链（CoT）的提示方法。学生还对生成的反馈给出了积极的反应，在基于技术接受模型的调查中报告了较高的感知有用性。 

---
# Conditional Deep Generative Models for Belief State Planning 

**Title (ZH)**: 基于条件的深度生成模型在信念状态规划中的应用 

**Authors**: Antoine Bigeard, Anthony Corso, Mykel Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11698)  

**Abstract**: Partially observable Markov decision processes (POMDPs) are used to model a wide range of applications, including robotics, autonomous vehicles, and subsurface problems. However, accurately representing the belief is difficult for POMDPs with high-dimensional states. In this paper, we propose a novel approach that uses conditional deep generative models (cDGMs) to represent the belief. Unlike traditional belief representations, cDGMs are well-suited for high-dimensional states and large numbers of observations, and they can generate an arbitrary number of samples from the posterior belief. We train the cDGMs on data produced by random rollout trajectories and show their effectiveness in solving a mineral exploration POMDP with a large and continuous state space. The cDGMs outperform particle filter baselines in both task-agnostic measures of belief accuracy as well as in planning performance. 

**Abstract (ZH)**: 部分可观测马尔可夫决策过程（POMDPs）用于 modeling 机器人、自主车辆和地下问题等广泛的应用。然而，对于具有高维状态的 POMDPs，准确地表示信念是困难的。本文提出了一种新颖的方法，使用条件深度生成模型（cDGMs）来表示信念。与传统的信念表示方法不同，cDGMs 适用于高维状态和大量观测数据，并且可以从后验信念中生成任意数量的样本。我们在由随机轨迹生成的数据上训练 cDGMs，并展示了其在具有大量连续状态空间的矿产勘探 POMDP 中的有效性。在任务无关的信念准确性度量以及计划性能方面，cDGMs 都优于粒子滤波基准方法。 

---
# Learning from Less: Guiding Deep Reinforcement Learning with Differentiable Symbolic Planning 

**Title (ZH)**: 从较少的数据中学习：通过可微符号规划指导深度强化学习 

**Authors**: Zihan Ye, Oleg Arenz, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.11661)  

**Abstract**: When tackling complex problems, humans naturally break them down into smaller, manageable subtasks and adjust their initial plans based on observations. For instance, if you want to make coffee at a friend's place, you might initially plan to grab coffee beans, go to the coffee machine, and pour them into the machine. Upon noticing that the machine is full, you would skip the initial steps and proceed directly to brewing. In stark contrast, state of the art reinforcement learners, such as Proximal Policy Optimization (PPO), lack such prior knowledge and therefore require significantly more training steps to exhibit comparable adaptive behavior. Thus, a central research question arises: \textit{How can we enable reinforcement learning (RL) agents to have similar ``human priors'', allowing the agent to learn with fewer training interactions?} To address this challenge, we propose differentiable symbolic planner (Dylan), a novel framework that integrates symbolic planning into Reinforcement Learning. Dylan serves as a reward model that dynamically shapes rewards by leveraging human priors, guiding agents through intermediate subtasks, thus enabling more efficient exploration. Beyond reward shaping, Dylan can work as a high level planner that composes primitive policies to generate new behaviors while avoiding common symbolic planner pitfalls such as infinite execution loops. Our experimental evaluations demonstrate that Dylan significantly improves RL agents' performance and facilitates generalization to unseen tasks. 

**Abstract (ZH)**: 如何使强化学习代理具备类似的“先验知识”，从而使其在 fewer training interactions 下进行学习？ 

---
# FLOW-BENCH: Towards Conversational Generation of Enterprise Workflows 

**Title (ZH)**: FLOW-BENCH: 向 toward 企业工作流对话生成方向努力 

**Authors**: Evelyn Duesterwald, Siyu Huo, Vatche Isahagian, K.R. Jayaram, Ritesh Kumar, Vinod Muthusamy, Punleuk Oum, Debashish Saha, Gegi Thomas, Praveen Venkateswaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.11646)  

**Abstract**: Business process automation (BPA) that leverages Large Language Models (LLMs) to convert natural language (NL) instructions into structured business process artifacts is becoming a hot research topic. This paper makes two technical contributions -- (i) FLOW-BENCH, a high quality dataset of paired natural language instructions and structured business process definitions to evaluate NL-based BPA tools, and support bourgeoning research in this area, and (ii) FLOW-GEN, our approach to utilize LLMs to translate natural language into an intermediate representation with Python syntax that facilitates final conversion into widely adopted business process definition languages, such as BPMN and DMN. We bootstrap FLOW-BENCH by demonstrating how it can be used to evaluate the components of FLOW-GEN across eight LLMs of varying sizes. We hope that FLOW-GEN and FLOW-BENCH catalyze further research in BPA making it more accessible to novice and expert users. 

**Abstract (ZH)**: 利用大规模语言模型将自然语言指令转换为结构化业务过程模型的业务过程自动化（BPA）：FLOW-BENCH和FLOW-GEN的贡献 

---
# Benchmarking Spatiotemporal Reasoning in LLMs and Reasoning Models: Capabilities and Challenges 

**Title (ZH)**: 基于时空推理能力与挑战的大型语言模型及推理模型基准研究 

**Authors**: Pengrui Quan, Brian Wang, Kang Yang, Liying Han, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2505.11618)  

**Abstract**: Spatiotemporal reasoning plays a key role in Cyber-Physical Systems (CPS). Despite advances in Large Language Models (LLMs) and Large Reasoning Models (LRMs), their capacity to reason about complex spatiotemporal signals remains underexplored. This paper proposes a hierarchical SpatioTemporal reAsoning benchmaRK, STARK, to systematically evaluate LLMs across three levels of reasoning complexity: state estimation (e.g., predicting field variables, localizing and tracking events in space and time), spatiotemporal reasoning over states (e.g., inferring spatial-temporal relationships), and world-knowledge-aware reasoning that integrates contextual and domain knowledge (e.g., intent prediction, landmark-aware navigation). We curate 26 distinct spatiotemporal tasks with diverse sensor modalities, comprising 14,552 challenges where models answer directly or by Python Code Interpreter. Evaluating 3 LRMs and 8 LLMs, we find LLMs achieve limited success in tasks requiring geometric reasoning (e.g., multilateration or triangulation), particularly as complexity increases. Surprisingly, LRMs show robust performance across tasks with various levels of difficulty, often competing or surpassing traditional first-principle-based methods. Our results show that in reasoning tasks requiring world knowledge, the performance gap between LLMs and LRMs narrows, with some LLMs even surpassing LRMs. However, the LRM o3 model continues to achieve leading performance across all evaluated tasks, a result attributed primarily to the larger size of the reasoning models. STARK motivates future innovations in model architectures and reasoning paradigms for intelligent CPS by providing a structured framework to identify limitations in the spatiotemporal reasoning of LLMs and LRMs. 

**Abstract (ZH)**: 空间-temporal推理在 cyber-physical 系统（CPS）中发挥着关键作用。尽管在大规模语言模型（LLMs）和大规模推理模型（LRMs）方面取得了进展，但它们在复杂空间-temporal信号推理方面的能力仍然未被充分探索。本文提出了一种分层空间-temporal推理基准STARK，系统性地评估LLMs在三种推理复杂度层次上的表现：状态估计（例如，预测场变量、在空间和时间中定位和跟踪事件）、状态的空间-temporal推理（例如，推断空间-temporal关系），以及融合上下文和领域知识的世界知识推理（例如，意图预测、地标引导导航）。我们收集了26个具有多种传感器模态的独立空间-temporal任务，共计14,552个挑战，其中模型直接回答或使用Python代码解释器。评估了3个LRMs和8个LLMs，发现LLMs在需要几何推理的任务（例如，多边orth或测距）中取得有限的成功，特别是随着复杂性的增加。令人惊讶的是，LRMs在不同难度级别的任务中表现出稳健的性能，经常与或超过传统的基于第一原理的方法。我们的结果表明，在需要世界知识的推理任务中，LLMs和LRMs之间的性能差距缩小，某些LLMs甚至超过了LRMs。然而，LRM o3模型在所有评估任务中继续保持领先表现，这一结果主要归因于推理模型的更大规模。STARK通过提供一个结构化的框架来识别LLMs和LRMs在空间-temporal推理方面的局限性，从而激发了智能CPS中模型架构和推理范式的未来创新。 

---
# Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions 

**Title (ZH)**: 使用强化学习训练大型语言模型解释人类决策 

**Authors**: Jian-Qiao Zhu, Hanbo Xie, Dilip Arumugam, Robert C. Wilson, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.11614)  

**Abstract**: A central goal of cognitive modeling is to develop models that not only predict human behavior but also provide insight into the underlying cognitive mechanisms. While neural network models trained on large-scale behavioral data often achieve strong predictive performance, they typically fall short in offering interpretable explanations of the cognitive processes they capture. In this work, we explore the potential of pretrained large language models (LLMs) to serve as dual-purpose cognitive models--capable of both accurate prediction and interpretable explanation in natural language. Specifically, we employ reinforcement learning with outcome-based rewards to guide LLMs toward generating explicit reasoning traces for explaining human risky choices. Our findings demonstrate that this approach produces high-quality explanations alongside strong quantitative predictions of human decisions. 

**Abstract (ZH)**: 认知建模的一个中心目标是开发既能预测人类行为又能揭示其背后认知机制的模型。尽管在大规模行为数据上训练的神经网络模型往往能实现较强的预测性能，但在提供可解释的认知过程解释方面通常较为欠缺。在本研究中，我们探讨了预训练大型语言模型（LLMs）作为兼具准确预测和自然语言可解释解释双重目的认知模型的潜力。具体而言，我们利用基于结果的强化学习来引导LLMs生成解释人类冒险选择的明确推理痕迹。我们的研究结果表明，这种方法能够在产生高质量解释的同时，对人类决策进行强大的定量预测。 

---
# Heart2Mind: Human-Centered Contestable Psychiatric Disorder Diagnosis System using Wearable ECG Monitors 

**Title (ZH)**: 心联思维：基于可争议可穿戴ECG监测的人本精神障碍诊断系统 

**Authors**: Hung Nguyen, Alireza Rahimi, Veronica Whitford, Hélène Fournier, Irina Kondratova, René Richard, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11612)  

**Abstract**: Psychiatric disorders affect millions globally, yet their diagnosis faces significant challenges in clinical practice due to subjective assessments and accessibility concerns, leading to potential delays in treatment. To help address this issue, we present Heart2Mind, a human-centered contestable psychiatric disorder diagnosis system using wearable electrocardiogram (ECG) monitors. Our approach leverages cardiac biomarkers, particularly heart rate variability (HRV) and R-R intervals (RRI) time series, as objective indicators of autonomic dysfunction in psychiatric conditions. The system comprises three key components: (1) a Cardiac Monitoring Interface (CMI) for real-time data acquisition from Polar H9/H10 devices; (2) a Multi-Scale Temporal-Frequency Transformer (MSTFT) that processes RRI time series through integrated time-frequency domain analysis; (3) a Contestable Diagnosis Interface (CDI) combining Self-Adversarial Explanations (SAEs) with contestable Large Language Models (LLMs). Our MSTFT achieves 91.7% accuracy on the HRV-ACC dataset using leave-one-out cross-validation, outperforming state-of-the-art methods. SAEs successfully detect inconsistencies in model predictions by comparing attention-based and gradient-based explanations, while LLMs enable clinicians to validate correct predictions and contest erroneous ones. This work demonstrates the feasibility of combining wearable technology with Explainable Artificial Intelligence (XAI) and contestable LLMs to create a transparent, contestable system for psychiatric diagnosis that maintains clinical oversight while leveraging advanced AI capabilities. Our implementation is publicly available at: this https URL. 

**Abstract (ZH)**: 心理障碍影响全球数以百万计的人，但由于临床实践中主观评估和可及性的问题，其诊断面临重大挑战，可能导致治疗延误。为解决这一问题，我们提出Heart2Mind，这是一种基于可穿戴心电图（ECG）监测的人本中心可争议心理障碍诊断系统。该方法利用心脏生物标志物，特别是心率变异性（HRV）和R-R间隔（RRI）时间序列作为心理障碍条件下自主功能障碍的客观指标。该系统包括三个关键组件：（1）心监测界面（CMI），用于从Polar H9/H10设备实时获取数据；（2）多尺度时频变压器（MSTFT），通过综合时频域分析处理RRI时间序列；（3）可争议诊断界面（CDI），结合自我对抗解释（SAEs）与可争议的大语言模型（LLMs）。我们的MSTFT在使用留一交叉验证的HRV-ACC数据集上达到了91.7%的准确率，超过了最先进的方法。SAEs通过比较基于注意力和梯度的解释成功检测模型预测中的不一致，而LLMs使临床医生能够验证正确的预测并质疑错误的预测。本研究展示了将可穿戴技术与可解释的人工智能（XAI）和可争议的大语言模型结合以创建一种透明、可争议的心理障碍诊断系统的可行性，同时保持临床监督并利用高级人工智能能力。我们的实现已公开发布于：this https URL。 

---
# Probing the Vulnerability of Large Language Models to Polysemantic Interventions 

**Title (ZH)**: 探测大型语言模型对多义干预的脆弱性 

**Authors**: Bofan Gong, Shiyang Lai, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.11611)  

**Abstract**: Polysemanticity -- where individual neurons encode multiple unrelated features -- is a well-known characteristic of large neural networks and remains a central challenge in the interpretability of language models. At the same time, its implications for model safety are also poorly understood. Leveraging recent advances in sparse autoencoders, we investigate the polysemantic structure of two small models (Pythia-70M and GPT-2-Small) and evaluate their vulnerability to targeted, covert interventions at the prompt, feature, token, and neuron levels. Our analysis reveals a consistent polysemantic topology shared across both models. Strikingly, we demonstrate that this structure can be exploited to mount effective interventions on two larger, black-box instruction-tuned models (LLaMA3.1-8B-Instruct and Gemma-2-9B-Instruct). These findings suggest not only the generalizability of the interventions but also point to a stable and transferable polysemantic structure that could potentially persist across architectures and training regimes. 

**Abstract (ZH)**: 多义性——单个神经元编码多个不相关的特征——是大型神经网络的一个已知特性，仍然是语言模型可解释性中的一个中心挑战。同时，其对模型安全性的影响也不甚明了。借助稀疏自编码器的近期进展，我们研究了两个小型模型（Pythia-70M 和 GPT-2-Small）的多义性结构，并评估了它们在提示、特征、标记和神经元层面受到有针对性的隐蔽干预的脆弱性。我们的分析揭示了这两种模型中存在的统一的多义性拓扑结构。令人惊讶的是，我们证明了这种结构可以被利用，以有效地对两个更大、黑盒指令调优模型（LLaMA3.1-8B-Instruct 和 Gemma-2-9B-Instruct）实施干预。这些发现不仅表明了干预的普适性，还指出了一个稳定且可转移的多义性结构，这种结构可能在不同架构和训练制度下持续存在。 

---
# Foundation Models for AI-Enabled Biological Design 

**Title (ZH)**: AI驱动生物设计的基石模型 

**Authors**: Asher Moldwin, Amarda Shehu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11610)  

**Abstract**: This paper surveys foundation models for AI-enabled biological design, focusing on recent developments in applying large-scale, self-supervised models to tasks such as protein engineering, small molecule design, and genomic sequence design. Though this domain is evolving rapidly, this survey presents and discusses a taxonomy of current models and methods. The focus is on challenges and solutions in adapting these models for biological applications, including biological sequence modeling architectures, controllability in generation, and multi-modal integration. The survey concludes with a discussion of open problems and future directions, offering concrete next-steps to improve the quality of biological sequence generation. 

**Abstract (ZH)**: 本文概述了AI赋能生物设计领域的基础模型，重点关注大型自监督模型在蛋白质工程、小分子设计及基因组序列设计等任务上的 Recent 发展。尽管该领域正在迅速演变，本文仍对当前模型和方法进行了分类，并进行了讨论。重点在于适应这些模型用于生物应用所面临的变化和解决方案，包括生物序列建模架构、生成的可控性以及多模态集成。本文结尾讨论了现存问题和未来方向，并提供了具体的下一步行动建议以提高生物序列生成的质量。 

---
# LLM Agents Are Hypersensitive to Nudges 

**Title (ZH)**: LLM代理对暗示过度敏感 

**Authors**: Manuel Cherep, Pattie Maes, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11584)  

**Abstract**: LLMs are being set loose in complex, real-world environments involving sequential decision-making and tool use. Often, this involves making choices on behalf of human users. However, not much is known about the distribution of such choices, and how susceptible they are to different choice architectures. We perform a case study with a few such LLM models on a multi-attribute tabular decision-making problem, under canonical nudges such as the default option, suggestions, and information highlighting, as well as additional prompting strategies. We show that, despite superficial similarities to human choice distributions, such models differ in subtle but important ways. First, they show much higher susceptibility to the nudges. Second, they diverge in points earned, being affected by factors like the idiosyncrasy of available prizes. Third, they diverge in information acquisition strategies: e.g. incurring substantial cost to reveal too much information, or selecting without revealing any. Moreover, we show that simple prompt strategies like zero-shot chain of thought (CoT) can shift the choice distribution, and few-shot prompting with human data can induce greater alignment. Yet, none of these methods resolve the sensitivity of these models to nudges. Finally, we show how optimal nudges optimized with a human resource-rational model can similarly increase LLM performance for some models. All these findings suggest that behavioral tests are needed before deploying models as agents or assistants acting on behalf of users in complex environments. 

**Abstract (ZH)**: 大规模语言模型在涉及序列决策和工具使用的真实复杂环境中的行为研究：来自几种模型的案例分析 

---
# CIE: Controlling Language Model Text Generations Using Continuous Signals 

**Title (ZH)**: CIE: 使用连续信号控制语言模型文本生成 

**Authors**: Vinay Samuel, Harshita Diddee, Yiming Zhang, Daphne Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2505.13448)  

**Abstract**: Aligning language models with user intent is becoming increasingly relevant to enhance user experience. This calls for designing methods that can allow users to control the properties of the language that LMs generate. For example, controlling the length of the generation, the complexity of the language that gets chosen, the sentiment, tone, etc. Most existing work attempts to integrate users' control by conditioning LM generations on natural language prompts or discrete control signals, which are often brittle and hard to scale. In this work, we are interested in \textit{continuous} control signals, ones that exist along a spectrum that can't easily be captured in a natural language prompt or via existing techniques in conditional generation. Through a case study in controlling the precise response-length of generations produced by LMs, we demonstrate how after fine-tuning, behaviors of language models can be controlled via continuous signals -- as vectors that are interpolated between a "low" and a "high" token embedding. Our method more reliably exerts response-length control than in-context learning methods or fine-tuning methods that represent the control signal as a discrete signal. Our full open-sourced code and datasets are available at this https URL. 

**Abstract (ZH)**: 将语言模型与用户意图对齐以增强用户体验日益重要，这需要设计方法让用户能够控制语言模型生成的语言的属性。例如，控制生成的长度、所选择的语言的复杂性、情感、语气等。大多数现有工作试图通过用自然语言提示或离散控制信号条件化语言模型的生成来整合用户的控制，但这些方法往往是脆弱的且难以扩展。在本工作中，我们感兴趣的是连续控制信号，这些信号存在于一种难以用自然语言提示或现有条件生成技术捕捉的光谱中。通过控制LM生成的精确响应长度的案例研究，我们展示了经过微调后，可以通过插值于“低”和“高”标记嵌入之间的向量来用连续信号控制语言模型的行为。我们的方法比上下文学习方法或将控制信号表示为离散信号的微调方法更可靠地实现响应长度控制。我们的完整开源代码和数据集可在以下链接获得：this https URL。 

---
# VTBench: Evaluating Visual Tokenizers for Autoregressive Image Generation 

**Title (ZH)**: VTBench: 评估视觉词嵌入器在自回归图像生成中的性能 

**Authors**: Huawei Lin, Tong Geng, Zhaozhuo Xu, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13439)  

**Abstract**: Autoregressive (AR) models have recently shown strong performance in image generation, where a critical component is the visual tokenizer (VT) that maps continuous pixel inputs to discrete token sequences. The quality of the VT largely defines the upper bound of AR model performance. However, current discrete VTs fall significantly behind continuous variational autoencoders (VAEs), leading to degraded image reconstructions and poor preservation of details and text. Existing benchmarks focus on end-to-end generation quality, without isolating VT performance. To address this gap, we introduce VTBench, a comprehensive benchmark that systematically evaluates VTs across three core tasks: Image Reconstruction, Detail Preservation, and Text Preservation, and covers a diverse range of evaluation scenarios. We systematically assess state-of-the-art VTs using a set of metrics to evaluate the quality of reconstructed images. Our findings reveal that continuous VAEs produce superior visual representations compared to discrete VTs, particularly in retaining spatial structure and semantic detail. In contrast, the degraded representations produced by discrete VTs often lead to distorted reconstructions, loss of fine-grained textures, and failures in preserving text and object integrity. Furthermore, we conduct experiments on GPT-4o image generation and discuss its potential AR nature, offering new insights into the role of visual tokenization. We release our benchmark and codebase publicly to support further research and call on the community to develop strong, general-purpose open-source VTs. 

**Abstract (ZH)**: 自回归（AR）模型在图像生成中 recently 展现了强大的性能，其中关键组件是视觉分词器（VT），它将连续的像素输入映射为离散的令牌序列。VT 的质量在很大程度上定义了 AR 模型性能的上限。然而，当前的离散 VT 明显落后于连续变分自编码器（VAEs），导致图像重构质量下降，细节和文本保真度差。现有的基准主要关注端到端的生成质量，而没有专门评价 VT 性能。为了解决这一差距，我们引入了 VTBench，这是一个全面的基准，系统地在三个核心任务：图像重构、细节保真和文本保真，以及多种评估场景中评估 VT。我们使用一组元 

---
# Optimizing Anytime Reasoning via Budget Relative Policy Optimization 

**Title (ZH)**: 通过预算相对策略优化优化任意时间推理 

**Authors**: Penghui Qi, Zichen Liu, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13438)  

**Abstract**: Scaling test-time compute is crucial for enhancing the reasoning capabilities of large language models (LLMs). Existing approaches typically employ reinforcement learning (RL) to maximize a verifiable reward obtained at the end of reasoning traces. However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency in both training and deployment. In this work, we present a novel framework, AnytimeReasoner, to optimize anytime reasoning performance, which aims to improve token efficiency and the flexibility of reasoning under varying token budget constraints. To achieve this, we truncate the complete thinking process to fit within sampled token budgets from a prior distribution, compelling the model to summarize the optimal answer for each truncated thinking for verification. This introduces verifiable dense rewards into the reasoning process, facilitating more effective credit assignment in RL optimization. We then optimize the thinking and summary policies in a decoupled manner to maximize the cumulative reward. Additionally, we introduce a novel variance reduction technique, Budget Relative Policy Optimization (BRPO), to enhance the robustness and efficiency of the learning process when reinforcing the thinking policy. Empirical results in mathematical reasoning tasks demonstrate that our method consistently outperforms GRPO across all thinking budgets under various prior distributions, enhancing both training and token efficiency. 

**Abstract (ZH)**: 扩展示时计算对于增强大语言模型推理能力至关重要。现有的方法通常通过强化学习（RL）来最大化推理踪迹结束时可验证的奖励。然而，这些方法只优化大量固定token预算下的最终性能，这在训练和部署中都降低了效率。在本文中，我们提出了一种新的框架AnytimeReasoner，以优化任意时间推理性能，旨在在不同的token预算约束下提高token效率和推理的灵活性。为此，我们将完整的推理过程截断以适应来自先验分布的抽样token预算，迫使模型总结每个截断推理的最优答案以供验证。这引入了推理过程中的可验证密集奖励，促进了RL优化中的更有效责任分配。然后，我们以解耦的方式优化推理和总结策略，以最大化累积奖励。此外，我们引入了一种新的方差减少技术，预算相对策略优化（BRPO），以增强强化推理策略学习过程的稳健性和效率。在数学推理任务中的实证结果显示，我们的方法在各种先验分布下的所有推理预算下均优于GRPO，同时提高训练和token效率。 

---
# FinePhys: Fine-grained Human Action Generation by Explicitly Incorporating Physical Laws for Effective Skeletal Guidance 

**Title (ZH)**: FinePhys: 细粒度人体动作生成通过显式融入物理法则以实现有效的骨架指导 

**Authors**: Dian Shao, Mingfei Shi, Shengda Xu, Haodong Chen, Yongle Huang, Binglu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13437)  

**Abstract**: Despite significant advances in video generation, synthesizing physically plausible human actions remains a persistent challenge, particularly in modeling fine-grained semantics and complex temporal dynamics. For instance, generating gymnastics routines such as "switch leap with 0.5 turn" poses substantial difficulties for current methods, often yielding unsatisfactory results. To bridge this gap, we propose FinePhys, a Fine-grained human action generation framework that incorporates Physics to obtain effective skeletal guidance. Specifically, FinePhys first estimates 2D poses in an online manner and then performs 2D-to-3D dimension lifting via in-context learning. To mitigate the instability and limited interpretability of purely data-driven 3D poses, we further introduce a physics-based motion re-estimation module governed by Euler-Lagrange equations, calculating joint accelerations via bidirectional temporal updating. The physically predicted 3D poses are then fused with data-driven ones, offering multi-scale 2D heatmap guidance for the diffusion process. Evaluated on three fine-grained action subsets from FineGym (FX-JUMP, FX-TURN, and FX-SALTO), FinePhys significantly outperforms competitive baselines. Comprehensive qualitative results further demonstrate FinePhys's ability to generate more natural and plausible fine-grained human actions. 

**Abstract (ZH)**: 尽管在视频生成方面取得了显著进展，但合成物理上合理的human动作仍然是一个持续的挑战，特别是在建模细粒度语义和复杂时序动态方面。为了弥合这一差距，我们提出了FinePhys，这是一种融合物理的人动作生成框架，以获得有效的骨骼指导。具体而言，FinePhys 首先以在线方式估计2D姿态，然后通过上下文学习进行2D到3D维度提升。为了缓解纯数据驱动的3D姿态的不稳定性及其有限的可解释性，我们进一步引入了一个基于欧拉-拉格朗日方程的物理驱动运动重新估计模块，通过双向时序更新计算关节加速度。然后，物理预测的3D姿态与数据驱动的姿态相结合，为扩散过程提供多尺度2D热图指导。FinePhys 在对 FineGym 的三个细粒度动作子集（FX-JUMP、FX-TURN和FX-SALTO）进行评估时，显著优于竞争基准。全面的定性结果进一步证明了FinePhys 生成更加自然和合理的细粒度human动作的能力。 

---
# Learnware of Language Models: Specialized Small Language Models Can Do Big 

**Title (ZH)**: 语言模型的适配软件：专门化的小型语言模型也能做到大事。 

**Authors**: Zhi-Hao Tan, Zi-Chen Zhao, Hao-Yu Shi, Xin-Yu Zhang, Peng Tan, Yang Yu, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.13425)  

**Abstract**: The learnware paradigm offers a novel approach to machine learning by enabling users to reuse a set of well-trained models for tasks beyond the models' original purposes. It eliminates the need to build models from scratch, instead relying on specifications (representations of a model's capabilities) to identify and leverage the most suitable models for new tasks. While learnware has proven effective in many scenarios, its application to language models has remained largely unexplored. At the same time, large language models (LLMs) have demonstrated remarkable universal question-answering abilities, yet they face challenges in specialized scenarios due to data scarcity, privacy concerns, and high computational costs, thus more and more specialized small language models (SLMs) are being trained for specific domains. To address these limitations systematically, the learnware paradigm provides a promising solution by enabling maximum utilization of specialized SLMs, and allowing users to identify and reuse them in a collaborative and privacy-preserving manner.
This paper presents a preliminary attempt to apply the learnware paradigm to language models. We simulated a learnware system comprising approximately 100 learnwares of specialized SLMs with 8B parameters, fine-tuned across finance, healthcare, and mathematics domains. Each learnware contains an SLM and a specification, which enables users to identify the most relevant models without exposing their own data. Experimental results demonstrate promising performance: by selecting one suitable learnware for each task-specific inference, the system outperforms the base SLMs on all benchmarks. Compared to LLMs, the system outperforms Qwen1.5-110B, Qwen2.5-72B, and Llama3.1-70B-Instruct by at least 14% in finance domain tasks, and surpasses Flan-PaLM-540B (ranked 7th on the Open Medical LLM Leaderboard) in medical domain tasks. 

**Abstract (ZH)**: 学习软件范式通过使用户能够重用一组经过充分训练的模型来执行超出模型原始用途的任务，为机器学习提供了一种新颖的方法。它消除了从头开始构建模型的需要，而是依赖于规范（即模型能力的表示）来识别并利用最适合新任务的模型。虽然学习软件在许多场景中已被证明是有效的，但其在语言模型中的应用仍未得到充分探索。同时，大型语言模型（LLMs）展示了令人瞩目的通用问答能力，但由于数据稀缺、隐私问题和高计算成本，在专业场景中仍面临挑战，因此越来越多的专门的小语言模型（SLMs）正在为特定领域进行训练。为了系统地解决这些问题，学习软件范式提供了一种有前途的解决方案，通过最大限度地利用专门的SLMs，并允许用户以协作和隐私保护的方式识别和重用它们。

本文初步尝试将学习软件范式应用于语言模型。我们模拟了一个由约100个专用于不同领域的特化小型语言模型（SLMs，参数量为8B）的学习软件系统，这些SLMs在金融、医疗和数学领域进行了微调。每个学习软件包含一个SLM和一个规范，这使得用户能够在不泄露自身数据的情况下识别最相关的模型。实验结果显示出有希望的性能：通过为每个特定任务选择一个合适的语言模型，系统在所有基准上的性能优于基础SLMs。与大型语言模型相比，该系统在金融领域的任务中至少优于Qwen1.5-110B、Qwen2.5-72B和Llama3.1-70B-Instruct 14%，并且在医疗领域的任务中超越了排名第七的Flan-PaLM-540B。 

---
# AdaptThink: Reasoning Models Can Learn When to Think 

**Title (ZH)**: AdaptThink: 推理模型可以学习何时进行思考 

**Authors**: Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13417)  

**Abstract**: Recently, large reasoning models have achieved impressive performance on various tasks by employing human-like deep thinking. However, the lengthy thinking process substantially increases inference overhead, making efficiency a critical bottleneck. In this work, we first demonstrate that NoThinking, which prompts the reasoning model to skip thinking and directly generate the final solution, is a better choice for relatively simple tasks in terms of both performance and efficiency. Motivated by this, we propose AdaptThink, a novel RL algorithm to teach reasoning models to choose the optimal thinking mode adaptively based on problem difficulty. Specifically, AdaptThink features two core components: (1) a constrained optimization objective that encourages the model to choose NoThinking while maintaining the overall performance; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby enabling cold start and allowing the model to explore and exploit both thinking modes throughout the training process. Our experiments indicate that AdaptThink significantly reduces the inference costs while further enhancing performance. Notably, on three math datasets, AdaptThink reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 53% and improves its accuracy by 2.4%, highlighting the promise of adaptive thinking-mode selection for optimizing the balance between reasoning quality and efficiency. Our codes and models are available at this https URL. 

**Abstract (ZH)**: 最近，通过采用类似人类的深度思考，大规模推理模型在多种任务上取得了卓越的性能。然而，冗长的思考过程显著增加了推理开销，使效率成为关键瓶颈。本文首先证明，在性能和效率方面，NoThinking（提示推理模型跳过思考并直接生成最终解决方案）是相对简单任务的更好选择。受此启发，我们提出了一种新的RL算法AdaptThink，旨在使推理模型能够根据问题难度自适应地选择最优的思考模式。AdaptThink具有两个核心组件：（1）受约束的优化目标，鼓励模型在保持整体性能的同时选择NoThinking；（2）重要性采样策略，在在线策略训练过程中平衡思考和不思考的样本，从而允许模型在训练过程中探索和利用两种思考模式，实现冷启动。实验表明，AdaptThink显著降低了推理成本并进一步提高了性能。值得注意的是，在三个数学数据集中，AdaptThink将DeepSeek-R1-Distill-Qwen-1.5B的平均响应长度降低了53%，并提高了其准确性2.4%，这突显了自适应思考模式选择在优化推理质量和效率之间的平衡方面的潜力。我们的代码和模型可在此处获取。 

---
# IG Parser: A Software Package for the Encoding of Institutional Statements using the Institutional Grammar 

**Title (ZH)**: IG Parser: 一套基于机构语法的机构声明编码软件包 

**Authors**: Christopher K. Frantz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13393)  

**Abstract**: This article provides an overview of IG Parser, a software that facilitates qualitative content analysis of formal (e.g., legal) rules or informal (e.g., socio-normative) norms, and strategies (such as conventions) -- referred to as \emph{institutions} -- that govern social systems and operate configurally to describe \emph{institutional systems}. To this end, the IG Parser employs a distinctive syntax that ensures rigorous encoding of natural language, while automating the transformation into various formats that support the downstream analysis using diverse analytical techniques. The conceptual core of the IG Parser is an associated syntax, IG Script, that operationalizes the conceptual foundations of the Institutional Grammar, and more specifically Institutional Grammar 2.0, an analytical paradigm for institutional analysis. This article presents the IG Parser, including its conceptual foundations, syntactic specification of IG Script, alongside architectural principles. This introduction is augmented with selective illustrative examples that highlight the use and benefit associated with the tool. 

**Abstract (ZH)**: 本文提供了IG Parser的概述，这是一种软件工具，用于对正式规则（如法律）或非正式规范（如社会规范）以及管理社会系统并以配置方式描述机构系统的治理机制（如惯例）进行定性内容分析。IG Parser采用独特的语法确保自然语言的严格编码，并自动转换为支持多种分析技术的下游分析的各种格式。IG Parser的概念核心是与其相关的语法IG Script，它实现了机构语言的概念基础，特别是机构语言2.0，这是一种机构分析的分析范式。本文介绍了IG Parser，包括其概念基础、IG Script的语法规范以及架构原则，并附有选择性的示例来突出工具的使用及其益处。 

---
# R3: Robust Rubric-Agnostic Reward Models 

**Title (ZH)**: R3: 坚韧的无评分标准的奖励模型 

**Authors**: David Anugraha, Zilu Tang, Lester James V. Miranda, Hanyang Zhao, Mohammad Rifqi Farhansyah, Garry Kuwanto, Derry Wijaya, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2505.13388)  

**Abstract**: Reward models are essential for aligning language model outputs with human preferences, yet existing approaches often lack both controllability and interpretability. These models are typically optimized for narrow objectives, limiting their generalizability to broader downstream tasks. Moreover, their scalar outputs are difficult to interpret without contextual reasoning. To address these limitations, we introduce R3, a novel reward modeling framework that is rubric-agnostic, generalizable across evaluation dimensions, and provides interpretable, reasoned score assignments. R3 enables more transparent and flexible evaluation of language models, supporting robust alignment with diverse human values and use cases. Our models, data, and code are available as open source at this https URL 

**Abstract (ZH)**: 奖励模型对于使语言模型输出与人类偏好一致是必不可少的，但现有方法往往缺乏可控性和可解释性。这些模型通常针对狭窄的目标进行优化，限制了其在更广泛的下游任务中的普适性。此外，它们的标量输出在没有上下文推理的情况下难以解释。为了解决这些限制，我们提出了一种名为R3的新型奖励模型框架，该框架不依赖于具体评价标准，适用于多种评价维度，并提供可解释的推理评分。R3使得语言模型的评估更加透明和灵活，支持与多样化的用户体验和价值观的稳健对齐。我们的模型、数据和代码可以在以下链接获取：this https URL。 

---
# How Adding Metacognitive Requirements in Support of AI Feedback in Practice Exams Transforms Student Learning Behaviors 

**Title (ZH)**: 在实践考试中增加元认知要求以支持AI反馈对学生学习行为的影响 

**Authors**: Mak Ahmad, Prerna Ravi, David Karger, Marc Facciotti  

**Link**: [PDF](https://arxiv.org/pdf/2505.13381)  

**Abstract**: Providing personalized, detailed feedback at scale in large undergraduate STEM courses remains a persistent challenge. We present an empirically evaluated practice exam system that integrates AI generated feedback with targeted textbook references, deployed in a large introductory biology course. Our system encourages metacognitive behavior by asking students to explain their answers and declare their confidence. It uses OpenAI's GPT-4o to generate personalized feedback based on this information, while directing them to relevant textbook sections. Through interaction logs from consenting participants across three midterms (541, 342, and 413 students respectively), totaling 28,313 question-student interactions across 146 learning objectives, along with 279 surveys and 23 interviews, we examined the system's impact on learning outcomes and engagement. Across all midterms, feedback types showed no statistically significant performance differences, though some trends suggested potential benefits. The most substantial impact came from the required confidence ratings and explanations, which students reported transferring to their actual exam strategies. About 40 percent of students engaged with textbook references when prompted by feedback -- far higher than traditional reading rates. Survey data revealed high satisfaction (mean rating 4.1 of 5), with 82.1 percent reporting increased confidence on practiced midterm topics, and 73.4 percent indicating they could recall and apply specific concepts. Our findings suggest that embedding structured reflection requirements may be more impactful than sophisticated feedback mechanisms. 

**Abstract (ZH)**: 大规模本科STEM课程中提供个性化详细反馈仍然是一项持续的挑战。我们介绍了一种经过实证评估的练习考试系统，该系统结合了AI生成的反馈和目标定向的教材引用，并应用于一门大型入门级生物学课程。该系统通过要求学生解释答案并声明其信心，来鼓励元认知行为。系统基于这些信息使用OpenAI的GPT-4o生成个性化反馈，并将学生引导至相关教材章节。通过对三次中期考试（分别有541、342和413名学生）的赞同参与者互动日志的分析，共计进行了28,313次问题-学生交互，涵盖了146个学习目标，同时进行了279份问卷调查和23次访谈，我们考察了该系统对学习成果和参与度的影响。总体而言，反馈类型在学习成绩上未显示出统计意义上的显著差异，但某些趋势显示潜在益处。最显著的影响来自所需的信心评级和解释，学生报告称将这些策略应用于实际考试中。当受到反馈提示时，约40%的学生查阅了教材参考——这远高于传统的阅读率。问卷数据显示，满意度很高（平均评分为4.1/5），其中82.1%的受访者表示在练习中期考试主题方面增强了信心，73.4%的人表示能回忆并应用特定概念。我们的研究发现表明，嵌入结构化反思要求可能比复杂的反馈机制更有效。 

---
# Thinkless: LLM Learns When to Think 

**Title (ZH)**: Thinkless: LLM 学习何时思考 

**Authors**: Gongfan Fang, Xinyin Ma, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13379)  

**Abstract**: Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at this https URL 

**Abstract (ZH)**: 基于学习的思考与否框架：在复杂逻辑推理任务中适应性选择简短与详尽推理 

---
# One-Step Offline Distillation of Diffusion-based Models via Koopman Modeling 

**Title (ZH)**: 基于库曼模型的一步离线扩散模型蒸馏 

**Authors**: Nimrod Berman, Ilan Naiman, Moshe Eliasof, Hedi Zisling, Omri Azencot  

**Link**: [PDF](https://arxiv.org/pdf/2505.13358)  

**Abstract**: Diffusion-based generative models have demonstrated exceptional performance, yet their iterative sampling procedures remain computationally expensive. A prominent strategy to mitigate this cost is distillation, with offline distillation offering particular advantages in terms of efficiency, modularity, and flexibility. In this work, we identify two key observations that motivate a principled distillation framework: (1) while diffusion models have been viewed through the lens of dynamical systems theory, powerful and underexplored tools can be further leveraged; and (2) diffusion models inherently impose structured, semantically coherent trajectories in latent space. Building on these observations, we introduce the Koopman Distillation Model KDM, a novel offline distillation approach grounded in Koopman theory-a classical framework for representing nonlinear dynamics linearly in a transformed space. KDM encodes noisy inputs into an embedded space where a learned linear operator propagates them forward, followed by a decoder that reconstructs clean samples. This enables single-step generation while preserving semantic fidelity. We provide theoretical justification for our approach: (1) under mild assumptions, the learned diffusion dynamics admit a finite-dimensional Koopman representation; and (2) proximity in the Koopman latent space correlates with semantic similarity in the generated outputs, allowing for effective trajectory alignment. Empirically, KDM achieves state-of-the-art performance across standard offline distillation benchmarks, improving FID scores by up to 40% in a single generation step. All implementation details and code for the experimental setups are provided in our GitHub - this https URL, or in our project page - this https URL. 

**Abstract (ZH)**: 基于扩散的生成模型展示了卓越的性能，但其迭代采样过程仍然计算成本高昂。一种减轻这一成本的突出策略是知识蒸馏，尤其是离线蒸馏在效率、模块性和灵活性方面具有明显优势。在本工作中，我们识别出两个关键观察，以指导一个有原则的知识蒸馏框架：(1) 尽管扩散模型已被视为动力系统理论的视角，但仍有许多强大且未充分利用的工具可以选择；(2) 扩散模型本质上会在潜在空间中产生结构化且语义一致的轨迹。基于这些观察，我们引入了Koopman蒸馏模型(KDM)，这是一种基于Koopman理论的新型离线蒸馏方法，Koopman理论是一种经典的表示非线性动力系统在变换空间中的线性模型的框架。KDM将嘈杂的输入编码到嵌入空间中，在此空间中，一个学习到的线性算子将它们向前传播，随后由一个解码器重建清洁样本。这使得生成步骤减少到单步，同时保持语义保真度。我们为我们的方法提供了理论依据：(1) 在温和的假设下，学习到的扩散动力学具有有限维的Koopman表示；(2) 在Koopman潜在空间中的接近性与生成输出中的语义相似性相关，允许有效的轨迹对齐。实验中，KDM在标准离线蒸馏基准测试中取得了最优性能，在单个生成步骤中最高提高FID分数40%。所有实验设置的实现细节和代码可在我们的GitHub repositories - 这里是链接 - 和项目页面 - 这里是链接 - 中找到。 

---
# J4R: Learning to Judge with Equivalent Initial State Group Relative Preference Optimization 

**Title (ZH)**: J4R: 基于等初始状态组相对偏好优化的判断学习 

**Authors**: Austin Xu, Yilun Zhou, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.13346)  

**Abstract**: To keep pace with the increasing pace of large language models (LLM) development, model output evaluation has transitioned away from time-consuming human evaluation to automatic evaluation, where LLMs themselves are tasked with assessing and critiquing other model outputs. LLM-as-judge models are a class of generative evaluators that excel in evaluating relatively simple domains, like chat quality, but struggle in reasoning intensive domains where model responses contain more substantive and challenging content. To remedy existing judge shortcomings, we explore training judges with reinforcement learning (RL). We make three key contributions: (1) We propose the Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us to train our judge to be robust to positional biases that arise in more complex evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that evaluates judges in diverse reasoning settings not covered by prior work. (3) We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or exceeding the performance of larger GRPO-trained judges on both JudgeBench and ReasoningJudgeBench. 

**Abstract (ZH)**: 随着大规模语言模型（LLM）开发速度的加快，模型输出评估已从耗时的人工评估转向自动评估，其中LLM本身被赋予评估和批判其他模型输出的任务。LLM作为裁判模型是一类擅长评估相对简单领域（如聊天质量），但在需要更多实质性且具有挑战性内容的推理密集型领域中表现不佳的生成性评估器。为弥补现有裁判的不足，我们探索了使用强化学习（RL）训练裁判的方法。我们做出了三项重要贡献：（1）我们提出了等价初始状态组相对策略优化（EIS-GRPO）算法，以使我们的裁判能够应对更复杂评估环境中出现的位置偏差。（2）我们引入了ReasoningJudgeBench基准测试，该基准测试评估裁判在多元推理场景中的表现，这些场景在以往工作之外。（3）我们训练了Reasoning Judges for Reasoning (J4R)，一种使用EIS-GRPO训练的7B裁判，其性能分别比GPT-4o和下一个最佳的小裁判高出6.7%和9%，在JudgeBench和ReasoningJudgeBench上的表现与更大规模的GRPO训练裁判相当或超过。 

---
# RoPECraft: Training-Free Motion Transfer with Trajectory-Guided RoPE Optimization on Diffusion Transformers 

**Title (ZH)**: RoPECraft：基于轨迹引导RoPE优化的无需训练的运动迁移方法 

**Authors**: Ahmet Berke Gokmen, Yigit Ekin, Bahri Batuhan Bilecen, Aysegul Dundar  

**Link**: [PDF](https://arxiv.org/pdf/2505.13344)  

**Abstract**: We propose RoPECraft, a training-free video motion transfer method for diffusion transformers that operates solely by modifying their rotary positional embeddings (RoPE). We first extract dense optical flow from a reference video, and utilize the resulting motion offsets to warp the complex-exponential tensors of RoPE, effectively encoding motion into the generation process. These embeddings are then further optimized during denoising time steps via trajectory alignment between the predicted and target velocities using a flow-matching objective. To keep the output faithful to the text prompt and prevent duplicate generations, we incorporate a regularization term based on the phase components of the reference video's Fourier transform, projecting the phase angles onto a smooth manifold to suppress high-frequency artifacts. Experiments on benchmarks reveal that RoPECraft outperforms all recently published methods, both qualitatively and quantitatively. 

**Abstract (ZH)**: RoPECraft：一种基于旋转位置编码的无需训练的视频运动转移方法 

---
# OPA-Pack: Object-Property-Aware Robotic Bin Packing 

**Title (ZH)**: 基于对象属性的机器人Bins打包方法：OPA-Pack 

**Authors**: Jia-Hui Pan, Yeok Tatt Cheah, Zhengzhe Liu, Ka-Hei Hui, Xiaojie Gao, Pheng-Ann Heng, Yun-Hui Liu, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13339)  

**Abstract**: Robotic bin packing aids in a wide range of real-world scenarios such as e-commerce and warehouses. Yet, existing works focus mainly on considering the shape of objects to optimize packing compactness and neglect object properties such as fragility, edibility, and chemistry that humans typically consider when packing objects. This paper presents OPA-Pack (Object-Property-Aware Packing framework), the first framework that equips the robot with object property considerations in planning the object packing. Technical-wise, we develop a novel object property recognition scheme with retrieval-augmented generation and chain-of-thought reasoning, and build a dataset with object property annotations for 1,032 everyday objects. Also, we formulate OPA-Net, aiming to jointly separate incompatible object pairs and reduce pressure on fragile objects, while compacting the packing. Further, OPA-Net consists of a property embedding layer to encode the property of candidate objects to be packed, together with a fragility heightmap and an avoidance heightmap to keep track of the packed objects. Then, we design a reward function and adopt a deep Q-learning scheme to train OPA-Net. Experimental results manifest that OPA-Pack greatly improves the accuracy of separating incompatible object pairs (from 52% to 95%) and largely reduces pressure on fragile objects (by 29.4%), while maintaining good packing compactness. Besides, we demonstrate the effectiveness of OPA-Pack on a real packing platform, showcasing its practicality in real-world scenarios. 

**Abstract (ZH)**: 面向物体属性感知的机器人包装框架（OPA-Pack） 

---
# Contextual Paralinguistic Data Creation for Multi-Modal Speech-LLM: Data Condensation and Spoken QA Generation 

**Title (ZH)**: 多模态语音-LLM中的上下文副语言数据创建：数据凝练与口语化问答生成 

**Authors**: Qiongqiong Wang, Hardik B. Sailor, Tianchi Liu, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2505.13338)  

**Abstract**: Current speech-LLMs exhibit limited capability in contextual reasoning alongside paralinguistic understanding, primarily due to the lack of Question-Answer (QA) datasets that cover both aspects. We propose a novel framework for dataset generation from in-the-wild speech data, that integrates contextual reasoning with paralinguistic information. It consists of a pseudo paralinguistic label-based data condensation of in-the-wild speech and LLM-based Contextual Paralinguistic QA (CPQA) generation. The effectiveness is validated by a strong correlation in evaluations of the Qwen2-Audio-7B-Instruct model on a dataset created by our framework and human-generated CPQA dataset. The results also reveal the speech-LLM's limitations in handling empathetic reasoning tasks, highlighting the need for such datasets and more robust models. The proposed framework is first of its kind and has potential in training more robust speech-LLMs with paralinguistic reasoning capabilities. 

**Abstract (ZH)**: 当前的语音大语言模型在上下文推理和副语言理解方面的能力有限，主要原因是缺乏涵盖这两方面的问答数据集。我们提出了一种从野生语音数据中生成数据的新框架，该框架将上下文推理与副语言信息集成。该框架包括基于伪副语言标签的野生语音数据 condensation 和基于大语言模型的上下文副语言问答（CPQA）生成。通过在使用我们框架创建的数据集上评估 Qwen2-Audio-7B-Instruct 模型与人工生成的 CPQA 数据集之间的强相关性，验证了其有效性。结果还揭示了语音大语言模型在处理共情推理任务方面的局限性，突显了此类数据集和更稳健模型的需求。所提出的框架是首创的，并有可能用于训练具有副语言推理能力的更 robust 的语音大语言模型。 

---
# Recommender Systems for Democracy: Toward Adversarial Robustness in Voting Advice Applications 

**Title (ZH)**: 推荐系统中的民主之道：面向投票建议应用的对抗鲁棒性研究 

**Authors**: Frédéric Berdoz, Dustin Brunner, Yann Vonlanthen, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2505.13329)  

**Abstract**: Voting advice applications (VAAs) help millions of voters understand which political parties or candidates best align with their views. This paper explores the potential risks these applications pose to the democratic process when targeted by adversarial entities. In particular, we expose 11 manipulation strategies and measure their impact using data from Switzerland's primary VAA, Smartvote, collected during the last two national elections. We find that altering application parameters, such as the matching method, can shift a party's recommendation frequency by up to 105%. Cherry-picking questionnaire items can increase party recommendation frequency by over 261%, while subtle changes to parties' or candidates' responses can lead to a 248% increase. To address these vulnerabilities, we propose adversarial robustness properties VAAs should satisfy, introduce empirical metrics for assessing the resilience of various matching methods, and suggest possible avenues for research toward mitigating the effect of manipulation. Our framework is key to ensuring secure and reliable AI-based VAAs poised to emerge in the near future. 

**Abstract (ZH)**: 投票建议应用程序（VAAs）帮助数百名选民理解哪些政治党派或候选人最符合他们的观点。本文探讨了当这些应用程序受到敌对实体攻击时对其民主过程潜在风险的影响。特别是，我们揭示了11种操纵策略，并使用瑞士主要VAA Smartvote在最近两次全国选举期间收集的数据来衡量其影响。我们发现，改变应用程序参数，如匹配方法，可能导致党派推荐频次最多提高105%。挑出问卷题目可以增加党派推荐频次超过261%，而对党派或候选人的响应进行细微改变可能导致推荐频次提高248%。为应对这些漏洞，我们提出了VAAs应满足的敌对 robust 性属性，引入了评估各种匹配方法鲁棒性的实证指标，并提出了减少操纵影响的研究方向。我们的框架对于确保安全可靠的基于AI的VAAs在未来的发展至关重要。 

---
# From What Ifs to Insights: Counterfactuals in Causal Inference vs. Explainable AI 

**Title (ZH)**: 从假设到洞见：因果推断中的反事实与可解释AI中的反事实 

**Authors**: Galit Shmueli, David Martens, Jaewon Yoo, Travis Greene  

**Link**: [PDF](https://arxiv.org/pdf/2505.13324)  

**Abstract**: Counterfactuals play a pivotal role in the two distinct data science fields of causal inference (CI) and explainable artificial intelligence (XAI). While the core idea behind counterfactuals remains the same in both fields--the examination of what would have happened under different circumstances--there are key differences in how they are used and interpreted. We introduce a formal definition that encompasses the multi-faceted concept of the counterfactual in CI and XAI. We then discuss how counterfactuals are used, evaluated, generated, and operationalized in CI vs. XAI, highlighting conceptual and practical differences. By comparing and contrasting the two, we hope to identify opportunities for cross-fertilization across CI and XAI. 

**Abstract (ZH)**: 反事实思想在因果推断和可解释人工智能这两个不同的数据科学领域中发挥着核心作用。尽管两者背后的反事实核心思想一致——即在不同情境下考察本会发生的情况——但在其应用和解释上存在关键差异。我们提供了一个形式化的定义，涵盖因果推断和可解释人工智能中多维度的反事实概念。然后我们讨论了在因果推断与可解释人工智能中反事实的使用、评估、生成和操作化方式，突显了概念和实践上的差异。通过比较和对比这两者，我们希望能识别出跨因果推断和可解释人工智能领域的交叉 fertilization 机会。 

---
# Denoising Diffusion Probabilistic Model for Point Cloud Compression at Low Bit-Rates 

**Title (ZH)**: 低比特率下点云去噪扩散概率模型压缩 

**Authors**: Gabriele Spadaro, Alberto Presta, Jhony H. Giraldo, Marco Grangetto, Wei Hu, Giuseppe Valenzise, Attilio Fiandrotti, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2505.13316)  

**Abstract**: Efficient compression of low-bit-rate point clouds is critical for bandwidth-constrained applications. However, existing techniques mainly focus on high-fidelity reconstruction, requiring many bits for compression. This paper proposes a "Denoising Diffusion Probabilistic Model" (DDPM) architecture for point cloud compression (DDPM-PCC) at low bit-rates. A PointNet encoder produces the condition vector for the generation, which is then quantized via a learnable vector quantizer. This configuration allows to achieve a low bitrates while preserving quality. Experiments on ShapeNet and ModelNet40 show improved rate-distortion at low rates compared to standardized and state-of-the-art approaches. We publicly released the code at this https URL. 

**Abstract (ZH)**: 低比特率点云压缩中高效的压缩对于带宽受限的应用至关重要。现有技术主要侧重于高保真重构，需要大量的比特数进行压缩。本文提出了一种“去噪扩散概率模型”（DDPM）架构（DDPM-PCC）用于低比特率点云压缩。PointNet编码器生成生成条件向量，然后通过可学习的矢量量化器进行量化。此配置允许在保持质量的同时实现低比特率。实验在ShapeNet和ModelNet40上显示，与标准化和当前最先进的方法相比，在低比特率下具有更好的率失真性能。我们已在以下网址公开发布了代码：这个httpsURL。 

---
# KHRONOS: a Kernel-Based Neural Architecture for Rapid, Resource-Efficient Scientific Computation 

**Title (ZH)**: KHRONOS：一种基于内核的神经架构，实现快速高效科学计算 

**Authors**: Reza T. Batley, Sourav Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.13315)  

**Abstract**: Contemporary models of high dimensional physical systems are constrained by the curse of dimensionality and a reliance on dense data. We introduce KHRONOS (Kernel Expansion Hierarchy for Reduced Order, Neural Optimized Surrogates), an AI framework for model based, model free and model inversion tasks. KHRONOS constructs continuously differentiable target fields with a hierarchical composition of per-dimension kernel expansions, which are tensorized into modes and then superposed. We evaluate KHRONOS on a canonical 2D, Poisson equation benchmark: across 16 to 512 degrees of freedom (DoFs), it obtained L2 square errors of 5e-4 down to 6e-10. This represents a 100 time gain over Kolmogorov Arnold Networks (which itself reports a 100 times improvement on MLPs/PINNs with 100 times fewer parameters) when controlling for the number of parameters. This also represents a 1e4 times improvement in L2 square error compared to standard linear FEM at comparable DoFs. Inference complexity is dominated by inner products, yielding sub-millisecond full-field predictions that scale to an arbitrary resolution. For inverse problems, KHRONOS facilitates rapid, iterative level set recovery in only a few forward evaluations, with sub-microsecond per sample latency. KHRONOS scalability, expressivity, and interpretability open new avenues in constrained edge computing, online control, computer vision, and beyond. 

**Abstract (ZH)**: 基于核扩张层次的低维代理AI框架KHRONOS 

---
# Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space 

**Title (ZH)**: 在黑暗中寻找：在潜在空间通过测试时实例级策略梯度进行推理 

**Authors**: Hengli Li, Chenxi Li, Tong Wu, Xuekai Zhu, Yuxuan Wang, Zhaoxin Yu, Eric Hanchen Jiang, Song-Chun Zhu, Zixia Jia, Ying Nian Wu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13308)  

**Abstract**: Reasoning ability, a core component of human intelligence, continues to pose a significant challenge for Large Language Models (LLMs) in the pursuit of AGI. Although model performance has improved under the training scaling law, significant challenges remain, particularly with respect to training algorithms, such as catastrophic forgetting, and the limited availability of novel training data. As an alternative, test-time scaling enhances reasoning performance by increasing test-time computation without parameter updating. Unlike prior methods in this paradigm focused on token space, we propose leveraging latent space for more effective reasoning and better adherence to the test-time scaling law. We introduce LatentSeek, a novel framework that enhances LLM reasoning through Test-Time Instance-level Adaptation (TTIA) within the model's latent space. Specifically, LatentSeek leverages policy gradient to iteratively update latent representations, guided by self-generated reward signals. LatentSeek is evaluated on a range of reasoning benchmarks, including GSM8K, MATH-500, and AIME2024, across multiple LLM architectures. Results show that LatentSeek consistently outperforms strong baselines, such as Chain-of-Thought prompting and fine-tuning-based methods. Furthermore, our analysis demonstrates that LatentSeek is highly efficient, typically converging within a few iterations for problems of average complexity, while also benefiting from additional iterations, thereby highlighting the potential of test-time scaling in the latent space. These findings position LatentSeek as a lightweight, scalable, and effective solution for enhancing the reasoning capabilities of LLMs. 

**Abstract (ZH)**: Large Language Models Reasoning Enhancement via Test-Time Instance-level Adaptation in Latent Space 

---
# RBF++: Quantifying and Optimizing Reasoning Boundaries across Measurable and Unmeasurable Capabilities for Chain-of-Thought Reasoning 

**Title (ZH)**: RBF++：衡量与优化可测量和不可测量能力的推理边界以提高链式推理效果 

**Authors**: Qiguang Chen, Libo Qin, Jinhao Liu, Yue Liao, Jiaqi Wang, Jingxuan Zhou, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2505.13307)  

**Abstract**: Chain-of-Thought (CoT) reasoning has proven effective in enhancing large language models (LLMs) on complex tasks, spurring research into its underlying mechanisms. However, two primary challenges remain for real-world applications: (1) the lack of quantitative metrics and actionable guidelines for evaluating and optimizing measurable boundaries of CoT capability, and (2) the absence of methods to assess boundaries of unmeasurable CoT capability, such as multimodal perception. To address these gaps, we introduce the Reasoning Boundary Framework++ (RBF++). To tackle the first challenge, we define the reasoning boundary (RB) as the maximum limit of CoT performance. We also propose a combination law for RBs, enabling quantitative analysis and offering actionable guidance across various CoT tasks. For the second challenge, particularly in multimodal scenarios, we introduce a constant assumption, which replaces unmeasurable RBs with scenario-specific constants. Additionally, we propose the reasoning boundary division mechanism, which divides unmeasurable RBs into two sub-boundaries, facilitating the quantification and optimization of both unmeasurable domain knowledge and multimodal perception capabilities. Extensive experiments involving 38 models across 13 tasks validate the feasibility of our framework in cross-modal settings. Additionally, we evaluate 10 CoT strategies, offer insights into optimization and decay from two complementary perspectives, and expand evaluation benchmarks for measuring RBs in LLM reasoning. We hope this work advances the understanding of RBs and optimization strategies in LLMs. Code and data are available at this https URL. 

**Abstract (ZH)**: Chain-of-Thought（CoT）推理边界框架++（RBF++）：跨模态环境中的可测量与不可测量推理边界的探索与优化 

---
# Cross-Cloud Data Privacy Protection: Optimizing Collaborative Mechanisms of AI Systems by Integrating Federated Learning and LLMs 

**Title (ZH)**: 跨云数据隐私保护：通过集成联邦学习和大语言模型优化AI系统协作机制 

**Authors**: Huaiying Luo, Cheng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.13292)  

**Abstract**: In the age of cloud computing, data privacy protection has become a major challenge, especially when sharing sensitive data across cloud environments. However, how to optimize collaboration across cloud environments remains an unresolved problem. In this paper, we combine federated learning with large-scale language models to optimize the collaborative mechanism of AI systems. Based on the existing federated learning framework, we introduce a cross-cloud architecture in which federated learning works by aggregating model updates from decentralized nodes without exposing the original data. At the same time, combined with large-scale language models, its powerful context and semantic understanding capabilities are used to improve model training efficiency and decision-making ability. We've further innovated by introducing a secure communication layer to ensure the privacy and integrity of model updates and training data. The model enables continuous model adaptation and fine-tuning across different cloud environments while protecting sensitive data. Experimental results show that the proposed method is significantly better than the traditional federated learning model in terms of accuracy, convergence speed and data privacy protection. 

**Abstract (ZH)**: 在云计算时代，基于联邦学习和大规模语言模型的数据隐私保护协作优化机制 

---
# TimeSeriesGym: A Scalable Benchmark for (Time Series) Machine Learning Engineering Agents 

**Title (ZH)**: TimeSeriesGym: 一个可扩展的时间序列机器学习工程代理基准 

**Authors**: Yifu Cai, Xinyu Li, Mononito Goswami, Michał Wiliński, Gus Welter, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2505.13291)  

**Abstract**: We introduce TimeSeriesGym, a scalable benchmarking framework for evaluating Artificial Intelligence (AI) agents on time series machine learning engineering challenges. Existing benchmarks lack scalability, focus narrowly on model building in well-defined settings, and evaluate only a limited set of research artifacts (e.g., CSV submission files). To make AI agent benchmarking more relevant to the practice of machine learning engineering, our framework scales along two critical dimensions. First, recognizing that effective ML engineering requires a range of diverse skills, TimeSeriesGym incorporates challenges from diverse sources spanning multiple domains and tasks. We design challenges to evaluate both isolated capabilities (including data handling, understanding research repositories, and code translation) and their combinations, and rather than addressing each challenge independently, we develop tools that support designing multiple challenges at scale. Second, we implement evaluation mechanisms for multiple research artifacts, including submission files, code, and models, using both precise numeric measures and more flexible LLM-based evaluation approaches. This dual strategy balances objective assessment with contextual judgment. Although our initial focus is on time series applications, our framework can be readily extended to other data modalities, broadly enhancing the comprehensiveness and practical utility of agentic AI evaluation. We open-source our benchmarking framework to facilitate future research on the ML engineering capabilities of AI agents. 

**Abstract (ZH)**: TimeSeriesGym：一种可扩展的时间序列机器学习工程挑战评估框架 

---
# FlowPure: Continuous Normalizing Flows for Adversarial Purification 

**Title (ZH)**: FlowPure：连续正则化流在对抗净化中的应用 

**Authors**: Elias Collaert, Abel Rodríguez, Sander Joos, Lieven Desmet, Vera Rimmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.13280)  

**Abstract**: Despite significant advancements in the area, adversarial robustness remains a critical challenge in systems employing machine learning models. The removal of adversarial perturbations at inference time, known as adversarial purification, has emerged as a promising defense strategy. To achieve this, state-of-the-art methods leverage diffusion models that inject Gaussian noise during a forward process to dilute adversarial perturbations, followed by a denoising step to restore clean samples before classification. In this work, we propose FlowPure, a novel purification method based on Continuous Normalizing Flows (CNFs) trained with Conditional Flow Matching (CFM) to learn mappings from adversarial examples to their clean counterparts. Unlike prior diffusion-based approaches that rely on fixed noise processes, FlowPure can leverage specific attack knowledge to improve robustness under known threats, while also supporting a more general stochastic variant trained on Gaussian perturbations for settings where such knowledge is unavailable. Experiments on CIFAR-10 and CIFAR-100 demonstrate that our method outperforms state-of-the-art purification-based defenses in preprocessor-blind and white-box scenarios, and can do so while fully preserving benign accuracy in the former. Moreover, our results show that not only is FlowPure a highly effective purifier but it also holds a strong potential for adversarial detection, identifying preprocessor-blind PGD samples with near-perfect accuracy. 

**Abstract (ZH)**: 基于连续规范流的 FlowPure：一种新型的对抗净化方法 

---
# Representation of perceived prosodic similarity of conversational feedback 

**Title (ZH)**: 感知对话反馈音�PED的表示方法 

**Authors**: Livia Qian, Carol Figueroa, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2505.13268)  

**Abstract**: Vocal feedback (e.g., `mhm', `yeah', `okay') is an important component of spoken dialogue and is crucial to ensuring common ground in conversational systems. The exact meaning of such feedback is conveyed through both lexical and prosodic form. In this work, we investigate the perceived prosodic similarity of vocal feedback with the same lexical form, and to what extent existing speech representations reflect such similarities. A triadic comparison task with recruited participants is used to measure perceived similarity of feedback responses taken from two different datasets. We find that spectral and self-supervised speech representations encode prosody better than extracted pitch features, especially in the case of feedback from the same speaker. We also find that it is possible to further condense and align the representations to human perception through contrastive learning. 

**Abstract (ZH)**: 语音反馈（如“嗯 hmm”、“是的 yeah”、“好的 okay”）是口头对话的重要组成部分，对于确保会话系统中的共同基础至关重要。此类反馈的具体含义通过词汇和语调形式传达。在本研究中，我们探讨了具有相同词汇形式的语音反馈在感知上的语调相似性，以及现有语音表示在多大程度上反映这种相似性。通过招募参与者进行三元比较任务，我们测量了来自两个不同数据集的反馈响应的感知相似性。研究发现，频谱和自监督语音表示比提取的音高特征更好地编码语调，尤其是在来自同一说话人的反馈情况下。我们还发现，可以通过对比学习进一步浓缩和对齐表示以匹配人类感知。 

---
# Net-Zero: A Comparative Study on Neural Network Design for Climate-Economic PDEs Under Uncertainty 

**Title (ZH)**: 净零：在不确定性条件下气候经济偏微分方程的神经网络设计比较研究 

**Authors**: Carlos Rodriguez-Pardo, Louis Daumas, Leonardo Chiani, Massimo Tavoni  

**Link**: [PDF](https://arxiv.org/pdf/2505.13264)  

**Abstract**: Climate-economic modeling under uncertainty presents significant computational challenges that may limit policymakers' ability to address climate change effectively. This paper explores neural network-based approaches for solving high-dimensional optimal control problems arising from models that incorporate ambiguity aversion in climate mitigation decisions. We develop a continuous-time endogenous-growth economic model that accounts for multiple mitigation pathways, including emission-free capital and carbon intensity reductions. Given the inherent complexity and high dimensionality of these models, traditional numerical methods become computationally intractable. We benchmark several neural network architectures against finite-difference generated solutions, evaluating their ability to capture the dynamic interactions between uncertainty, technology transitions, and optimal climate policy. Our findings demonstrate that appropriate neural architecture selection significantly impacts both solution accuracy and computational efficiency when modeling climate-economic systems under uncertainty. These methodological advances enable more sophisticated modeling of climate policy decisions, allowing for better representation of technology transitions and uncertainty-critical elements for developing effective mitigation strategies in the face of climate change. 

**Abstract (ZH)**: 基于神经网络的方法解决包含缓解决策中不确定性规避的高维最优控制问题：应对气候变化的气候经济建模挑战 

---
# WikiPersonas: What Can We Learn From Personalized Alignment to Famous People? 

**Title (ZH)**: WikiPersonas：我们能从名人个性化对齐中学到什么？ 

**Authors**: Zilu Tang, Afra Feyza Akyürek, Ekin Akyürek, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.13257)  

**Abstract**: Preference alignment has become a standard pipeline in finetuning models to follow \emph{generic} human preferences. Majority of work seeks to optimize model to produce responses that would be preferable \emph{on average}, simplifying the diverse and often \emph{contradicting} space of human preferences. While research has increasingly focused on personalized alignment: adapting models to individual user preferences, there is a lack of personalized preference dataset which focus on nuanced individual-level preferences. To address this, we introduce WikiPersona: the first fine-grained personalization using well-documented, famous individuals. Our dataset challenges models to align with these personas through an interpretable process: generating verifiable textual descriptions of a persona's background and preferences in addition to alignment. We systematically evaluate different personalization approaches and find that as few-shot prompting with preferences and fine-tuning fail to simultaneously ensure effectiveness and efficiency, using \textit{inferred personal preferences} as prefixes enables effective personalization, especially in topics where preferences clash while leading to more equitable generalization across unseen personas. 

**Abstract (ZH)**: 偏好对齐已成为微调模型以遵循通用人类偏好的标准流程。大多数研究致力于优化模型以生成在平均意义上更可取的响应，简化并往往相互矛盾的广泛的人类偏好空间。虽然研究越来越多地关注个性化对齐：使模型适应个别用户偏好，但仍缺乏专注于细微个体层面偏好的个性化偏好数据集。为解决这一问题，我们引入了WikiPersona：首个使用详细记录的知名人物进行细粒度个性化的方法。我们的数据集通过可解释的过程挑战模型：生成可验证的人物背景和偏好的文本描述，同时进行偏好对齐。我们系统地评估了不同的个性化方法，发现少量提示与偏好和微调无法同时保证效果和效率，使用推断出的个人偏好作为前缀能够有效进行个性化，特别是在偏好冲突的话题上更具效果，同时在未见过的人物上实现更公平的泛化。 

---
# Composing Dextrous Grasping and In-hand Manipulation via Scoring with a Reinforcement Learning Critic 

**Title (ZH)**: 基于强化学习批评家评分的灵巧抓取与在手操作compose 

**Authors**: Lennart Röstel, Dominik Winkelbauer, Johannes Pitz, Leon Sievers, Berthold Bäuml  

**Link**: [PDF](https://arxiv.org/pdf/2505.13253)  

**Abstract**: In-hand manipulation and grasping are fundamental yet often separately addressed tasks in robotics. For deriving in-hand manipulation policies, reinforcement learning has recently shown great success. However, the derived controllers are not yet useful in real-world scenarios because they often require a human operator to place the objects in suitable initial (grasping) states. Finding stable grasps that also promote the desired in-hand manipulation goal is an open problem. In this work, we propose a method for bridging this gap by leveraging the critic network of a reinforcement learning agent trained for in-hand manipulation to score and select initial grasps. Our experiments show that this method significantly increases the success rate of in-hand manipulation without requiring additional training. We also present an implementation of a full grasp manipulation pipeline on a real-world system, enabling autonomous grasping and reorientation even of unwieldy objects. 

**Abstract (ZH)**: 基于批评网络的抓取初始化方法以提升手持操作性能 

---
# MAGI-1: Autoregressive Video Generation at Scale 

**Title (ZH)**: MAGI-1：大规模自回归视频生成 

**Authors**: Sand.ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W.Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang, Yuqiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13211)  

**Abstract**: We present MAGI-1, a world model that generates videos by autoregressively predicting a sequence of video chunks, defined as fixed-length segments of consecutive frames. Trained to denoise per-chunk noise that increases monotonically over time, MAGI-1 enables causal temporal modeling and naturally supports streaming generation. It achieves strong performance on image-to-video (I2V) tasks conditioned on text instructions, providing high temporal consistency and scalability, which are made possible by several algorithmic innovations and a dedicated infrastructure stack. MAGI-1 facilitates controllable generation via chunk-wise prompting and supports real-time, memory-efficient deployment by maintaining constant peak inference cost, regardless of video length. The largest variant of MAGI-1 comprises 24 billion parameters and supports context lengths of up to 4 million tokens, demonstrating the scalability and robustness of our approach. The code and models are available at this https URL and this https URL. The product can be accessed at this https URL. 

**Abstract (ZH)**: 我们提出MAGI-1，这是一种世界模型，通过自回归地预测视频片段序列生成视频，其中视频片段定义为连续帧的固定长度段。通过训练来减少随时间单调增加的逐片段噪声，MAGI-1实现了因果时间建模，并自然支持按需生成。它在基于文本指令的图像到视频(I2V)任务中表现出色，提供高时间一致性与可扩展性，这得益于一系列算法创新和专用的基础设施栈。MAGI-1通过逐片段提示实现可控生成，并通过保持恒定的峰值推断成本，支持实时、内存高效的部署，无论视频长度如何。MAGI-1的最大版本包含24亿个参数，并支持多达400万令牌的上下文长度，展示了我们方法的可扩展性和鲁棒性。代码和模型可在此链接获取，并可在此链接访问产品。 

---
# Picturized and Recited with Dialects: A Multimodal Chinese Representation Framework for Sentiment Analysis of Classical Chinese Poetry 

**Title (ZH)**: 用方言图绘与朗读：一种多模态中国古代诗歌情感分析的汉语表示框架 

**Authors**: Xiaocong Du, Haoyu Pei, Haipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13210)  

**Abstract**: Classical Chinese poetry is a vital and enduring part of Chinese literature, conveying profound emotional resonance. Existing studies analyze sentiment based on textual meanings, overlooking the unique rhythmic and visual features inherent in poetry,especially since it is often recited and accompanied by Chinese paintings. In this work, we propose a dialect-enhanced multimodal framework for classical Chinese poetry sentiment analysis. We extract sentence-level audio features from the poetry and incorporate audio from multiple dialects,which may retain regional ancient Chinese phonetic features, enriching the phonetic representation. Additionally, we generate sentence-level visual features, and the multimodal features are fused with textual features enhanced by LLM translation through multimodal contrastive representation learning. Our framework outperforms state-of-the-art methods on two public datasets, achieving at least 2.51% improvement in accuracy and 1.63% in macro F1. We open-source the code to facilitate research in this area and provide insights for general multimodal Chinese representation. 

**Abstract (ZH)**: 古典 Chinese 诗歌情感分析的方言增强多模态框架 

---
# Efficient Generation of Parameterised Quantum Circuits from Large Texts 

**Title (ZH)**: 从大规模文本中高效生成参数化量子电路 

**Authors**: Colin Krawchuk, Nikhil Khatri, Neil John Ortega, Dimitri Kartsaklis  

**Link**: [PDF](https://arxiv.org/pdf/2505.13208)  

**Abstract**: Quantum approaches to natural language processing (NLP) are redefining how linguistic information is represented and processed. While traditional hybrid quantum-classical models rely heavily on classical neural networks, recent advancements propose a novel framework, DisCoCirc, capable of directly encoding entire documents as parameterised quantum circuits (PQCs), besides enjoying some additional interpretability and compositionality benefits. Following these ideas, this paper introduces an efficient methodology for converting large-scale texts into quantum circuits using tree-like representations of pregroup diagrams. Exploiting the compositional parallels between language and quantum mechanics, grounded in symmetric monoidal categories, our approach enables faithful and efficient encoding of syntactic and discourse relationships in long and complex texts (up to 6410 words in our experiments) to quantum circuits. The developed system is provided to the community as part of the augmented open-source quantum NLP package lambeq Gen II. 

**Abstract (ZH)**: 量子方法在自然语言处理中的应用重新定义了语言信息的表示和处理方式。本文介绍了一种高效的方法，将大规模文本转换为量子电路，利用预组图的树形表示。通过利用语言与量子力学之间基于对称张量范畴的组成相似性，该方法能够忠实且高效地将长且复杂的文本（实验中可达6410词）中的句法和话语关系编码到量子电路中。开发的系统作为增强版的开源量子NLP软件包lambeq Gen II提供给社区。 

---
# MatPredict: a dataset and benchmark for learning material properties of diverse indoor objects 

**Title (ZH)**: MatPredict：一个用于多种室内物体材料属性学习的数据集和基准 

**Authors**: Yuzhen Chen, Hojun Son, Arpan Kusari  

**Link**: [PDF](https://arxiv.org/pdf/2505.13201)  

**Abstract**: Determining material properties from camera images can expand the ability to identify complex objects in indoor environments, which is valuable for consumer robotics applications. To support this, we introduce MatPredict, a dataset that combines the high-quality synthetic objects from Replica dataset with MatSynth dataset's material properties classes - to create objects with diverse material properties. We select 3D meshes of specific foreground objects and render them with different material properties. In total, we generate \textbf{18} commonly occurring objects with \textbf{14} different materials. We showcase how we provide variability in terms of lighting and camera placement for these objects. Next, we provide a benchmark for inferring material properties from visual images using these perturbed models in the scene, discussing the specific neural network models involved and their performance based on different image comparison metrics. By accurately simulating light interactions with different materials, we can enhance realism, which is crucial for training models effectively through large-scale simulations. This research aims to revolutionize perception in consumer robotics. The dataset is provided \href{this https URL}{here} and the code is provided \href{this https URL}{here}. 

**Abstract (ZH)**: 从摄像头图像确定材料属性可以扩展识别室内环境中复杂对象的能力，这对于消费级机器人应用是宝贵的。为此，我们介绍了MatPredict数据集，该数据集结合了Replica数据集的高质量合成物体与MatSynth数据集的材料属性类别，以创建具有各种材料属性的物体。我们选择了特定前景物体的3D网格，并使用不同的材料属性进行渲染。总共生成了18种常见物体，每种物体具有14种不同材料。我们展示了如何在这些物体上提供照明和相机位置的多样性。接下来，我们提供了一个基准，用于通过这些扰动模型从视觉图像中推断出材料属性，讨论了涉及的具体神经网络模型及其基于不同图像比较指标的表现。通过准确模拟光与不同材料的相互作用，可以增强现实感，这对于通过大规模模拟有效训练模型至关重要。本研究旨在革新消费级机器人的感知能力。数据集可从\href{this https URL}{这里}获取，代码可从\href{this https URL}{这里}获取。 

---
# A Physics-Inspired Optimizer: Velocity Regularized Adam 

**Title (ZH)**: 物理启发式的优化器：速度正则化Adam 

**Authors**: Pranav Vaidhyanathan, Lucas Schorling, Natalia Ares, Michael A. Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2505.13196)  

**Abstract**: We introduce Velocity-Regularized Adam (VRAdam), a physics-inspired optimizer for training deep neural networks that draws on ideas from quartic terms for kinetic energy with its stabilizing effects on various system dynamics. Previous algorithms, including the ubiquitous Adam, operate at the so called adaptive edge of stability regime during training leading to rapid oscillations and slowed convergence of loss. However, VRAdam adds a higher order penalty on the learning rate based on the velocity such that the algorithm automatically slows down whenever weight updates become large. In practice, we observe that the effective dynamic learning rate shrinks in high-velocity regimes, damping oscillations and allowing for a more aggressive base step size when necessary without divergence. By combining this velocity-based regularizer for global damping with per-parameter scaling of Adam to create a hybrid optimizer, we demonstrate that VRAdam consistently exceeds the performance against standard optimizers including AdamW. We benchmark various tasks such as image classification, language modeling, image generation and generative modeling using diverse architectures and training methodologies including Convolutional Neural Networks (CNNs), Transformers, and GFlowNets. 

**Abstract (ZH)**: 基于速度正则化的Adam优化器（VRAdam）：一种物理启发式的深度神经网络训练优化器 

---
# True Zero-Shot Inference of Dynamical Systems Preserving Long-Term Statistics 

**Title (ZH)**: 真零样本推断动力系统并保持长期统计特性 

**Authors**: Christoph Jürgen Hemmer, Daniel Durstewitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13192)  

**Abstract**: Complex, temporally evolving phenomena, from climate to brain activity, are governed by dynamical systems (DS). DS reconstruction (DSR) seeks to infer generative surrogate models of these from observed data, reproducing their long-term behavior. Existing DSR approaches require purpose-training for any new system observed, lacking the zero-shot and in-context inference capabilities known from LLMs. Here we introduce DynaMix, a novel multivariate ALRNN-based mixture-of-experts architecture pre-trained for DSR, the first DSR model able to generalize zero-shot to out-of-domain DS. Just from a provided context signal, without any re-training, DynaMix faithfully forecasts the long-term evolution of novel DS where existing time series (TS) foundation models, like Chronos, fail -- at a fraction of the number of parameters and orders of magnitude faster inference times. DynaMix outperforms TS foundation models in terms of long-term statistics, and often also short-term forecasts, even on real-world time series, like traffic or weather data, typically used for training and evaluating TS models, but not at all part of DynaMix' training corpus. We illustrate some of the failure modes of TS models for DSR problems, and conclude that models built on DS principles may bear a huge potential also for advancing the TS prediction field. 

**Abstract (ZH)**: 复杂随时间演变的现象，从气候到脑活动，均由动力系统（DS）支配。DS重构（DSR）旨在从观测数据中推断这些现象的生成替代模型，再现其长期行为。现有的DSR方法需要为每个新观测系统进行特定训练，缺乏类似于LLMs的零样本和上下文推理能力。我们引入了DynaMix，这是一种基于多变量ALRNN的混合专家架构，预先训练用于DSR，是第一个能够泛化到领域外DS的DSR模型。仅通过提供上下文信号，无需任何重新训练，DynaMix能够准确预测现有时间序列（TS）基础模型（如Chronos）无法解决问题的新型DS的长期演化——参数数量大幅减少，推理速度也快多个数量级。DynaMix在长期统计方面优于TS基础模型，并且在许多情况下，在短期预测方面也表现更好，即使是在通常用于训练和评估TS模型的真实世界时间序列数据（如交通或天气数据）上，这些数据根本不包含于DynaMix的训练语料库中。我们展示了TS模型在DSR问题上的几种失败模式，并得出结论，基于DS原理构建的模型可能对推进时间序列预测领域有着巨大的潜力。 

---
# Emergence of Fixational and Saccadic Movements in a Multi-Level Recurrent Attention Model for Vision 

**Title (ZH)**: 多级循环注意模型中 Fixational 和 Saccadic 运动的出现 

**Authors**: Pengcheng Pan, Yonekura Shogo, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13191)  

**Abstract**: Inspired by foveal vision, hard attention models promise interpretability and parameter economy. However, existing models like the Recurrent Model of Visual Attention (RAM) and Deep Recurrent Attention Model (DRAM) failed to model the hierarchy of human vision system, that compromise on the visual exploration dynamics. As a result, they tend to produce attention that are either overly fixational or excessively saccadic, diverging from human eye movement behavior. In this paper, we propose a Multi-Level Recurrent Attention Model (MRAM), a novel hard attention framework that explicitly models the neural hierarchy of human visual processing. By decoupling the function of glimpse location generation and task execution in two recurrent layers, MRAM emergent a balanced behavior between fixation and saccadic movement. Our results show that MRAM not only achieves more human-like attention dynamics, but also consistently outperforms CNN, RAM and DRAM baselines on standard image classification benchmarks. 

**Abstract (ZH)**: 受中心视野启发，硬注意力模型 promise 了可解释性和参数经济性。然而，现有的模型如循环视觉注意力模型（RAM）和深度循环注意力模型（DRAM）未能建模人类视觉系统的层次结构，牺牲了视觉探索动力学。因此，它们往往会生成过度注视或过度扫视的注意力，偏离人类眼动行为。在本文中，我们提出了一种多层循环注意力模型（MRAM），这是一种新型的硬注意力框架，明确建模人类视觉处理的神经层次结构。通过在两个循环层中拆分概览位置生成和任务执行的功能，MRAM 产生了平衡的固定和扫视运动行为。我们的结果表明，MRAM 不仅实现了更接近人类的注意力动力学，而且在标准图像分类基准测试中始终优于 CNN、RAM 和 DRAM 基线。 

---
# When a Reinforcement Learning Agent Encounters Unknown Unknowns 

**Title (ZH)**: 当强化学习代理遇到未知的未知数 

**Authors**: Juntian Zhu, Miguel de Carvalho, Zhouwang Yang, Fengxiang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.13188)  

**Abstract**: An AI agent might surprisingly find she has reached an unknown state which she has never been aware of -- an unknown unknown. We mathematically ground this scenario in reinforcement learning: an agent, after taking an action calculated from value functions $Q$ and $V$ defined on the {\it {aware domain}}, reaches a state out of the domain. To enable the agent to handle this scenario, we propose an {\it episodic Markov decision {process} with growing awareness} (EMDP-GA) model, taking a new {\it noninformative value expansion} (NIVE) approach to expand value functions to newly aware areas: when an agent arrives at an unknown unknown, value functions $Q$ and $V$ whereon are initialised by noninformative beliefs -- the averaged values on the aware domain. This design is out of respect for the complete absence of knowledge in the newly discovered state. The upper confidence bound momentum Q-learning is then adapted to the growing awareness for training the EMDP-GA model. We prove that (1) the regret of our approach is asymptotically consistent with the state of the art (SOTA) without exposure to unknown unknowns in an extremely uncertain environment, and (2) our computational complexity and space complexity are comparable with the SOTA -- these collectively suggest that though an unknown unknown is surprising, it will be asymptotically properly discovered with decent speed and an affordable cost. 

**Abstract (ZH)**: 一种AI代理可能会意外地发现自己达到了一个从未意识到的未知状态——一个未知的未知。我们通过强化学习方法在数学上确立了这一场景：在根据定义于已知域的值函数 \(Q\) 和 \(V\) 计算出的动作作用下，代理会到达已知域之外的状态。为了使代理能够应对这一场景，我们提出了一种“具有增长意识的分段马尔可夫决策过程”(EMDP-GA) 模型，并采用了一种新的“非信息性价值扩展”(NIVE) 方法将价值函数扩展到新意识到的区域：当代理到达一个未知的未知状态时，价值函数 \(Q\) 和 \(V\) 由已知域上的平均非信息性信念初始化。这一设计体现了对新发现状态完全无知的尊重。然后，我们将上置信界限动量Q学习方法适应增长的意识，以训练EMDP-GA模型。我们证明了（1）我们的方法在极端不确定环境中不接触到未知的未知状态时的遗憾与当前最佳方法（SOTA）渐近一致；（2）我们的计算复杂度和空间复杂度与当前最佳方法相当——这些共同表明，尽管未知的未知是令人惊讶的，但随着良好的速度和可承受的成本，最终会渐近发现它。 

---
# Information Science Principles of Machine Learning: A Causal Chain Meta-Framework Based on Formalized Information Mapping 

**Title (ZH)**: 信息科学原理的机器学习：基于形式化信息映射的因果链元框架 

**Authors**: Jianfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13182)  

**Abstract**: [Objective] This study focuses on addressing the current lack of a unified formal theoretical framework in machine learning, as well as the deficiencies in interpretability and ethical safety assurance. [Methods] A formal information model is first constructed, utilizing sets of well-formed formulas to explicitly define the ontological states and carrier mappings of typical components in machine learning. Learnable and processable predicates, along with learning and processing functions, are introduced to analyze the logical deduction and constraint rules of the causal chains within models. [Results] A meta-framework for machine learning theory (MLT-MF) is established. Based on this framework, universal definitions for model interpretability and ethical safety are proposed. Furthermore, three key theorems are proved: the equivalence of model interpretability and information recoverability, the assurance of ethical safety, and the estimation of generalization error. [Limitations] The current framework assumes ideal conditions with noiseless information-enabling mappings and primarily targets model learning and processing logic in static scenarios. It does not yet address information fusion and conflict resolution across ontological spaces in multimodal or multi-agent systems. [Conclusions] This work overcomes the limitations of fragmented research and provides a unified theoretical foundation for systematically addressing the critical challenges currently faced in machine learning. 

**Abstract (ZH)**: 本研究致力于解决现行机器学习中缺乏统一的形式理论框架的问题，以及解释性和伦理安全性不足的问题。 

---
# ToolSpectrum : Towards Personalized Tool Utilization for Large Language Models 

**Title (ZH)**: ToolSpectrum: 向量化大型语言模型个性化工具使用的研究 

**Authors**: Zihao Cheng, Hongru Wang, Zeming Liu, Yuhang Guo, Yuanfang Guo, Yunhong Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13176)  

**Abstract**: While integrating external tools into large language models (LLMs) enhances their ability to access real-time information and domain-specific services, existing approaches focus narrowly on functional tool selection following user instructions, overlooking the context-aware personalization in tool selection. This oversight leads to suboptimal user satisfaction and inefficient tool utilization, particularly when overlapping toolsets require nuanced selection based on contextual factors. To bridge this gap, we introduce ToolSpectrum, a benchmark designed to evaluate LLMs' capabilities in personalized tool utilization. Specifically, we formalize two key dimensions of personalization, user profile and environmental factors, and analyze their individual and synergistic impacts on tool utilization. Through extensive experiments on ToolSpectrum, we demonstrate that personalized tool utilization significantly improves user experience across diverse scenarios. However, even state-of-the-art LLMs exhibit the limited ability to reason jointly about user profiles and environmental factors, often prioritizing one dimension at the expense of the other. Our findings underscore the necessity of context-aware personalization in tool-augmented LLMs and reveal critical limitations for current models. Our data and code are available at this https URL. 

**Abstract (ZH)**: 外部工具集成到大型语言模型中增强了其访问实时信息和领域特定服务的能力，但现有方法在工具选择上仅专注于用户指令的功能性选择，忽视了基于上下文的个性化选择。这种忽视导致了用户满意度的降低和工具利用效率的低下，尤其是在工具集重叠的情况下，需要根据上下文因素进行细致的选择。为解决这一问题，我们提出了ToolSpectrum这一基准测试，旨在评估大型语言模型在个性化工具利用方面的能力。具体而言，我们形式化了个人化选择的两个关键维度——用户画像和环境因素，并分析了它们在单独及协同作用下的工具利用影响。通过在ToolSpectrum上的大量实验，我们证明了个性化工具利用可以显著改进跨不同场景的用户体验。然而，即使是最先进的大型语言模型，在综合考虑用户画像和环境因素方面也表现出有限的能力，往往优先考虑其中一个维度而牺牲另一个维度。我们的研究结果强调了工具增强的大语言模型中上下文感知个性化的重要性，并揭示了当前模型的关键限制。我们的数据和代码可在以下链接获取：this https URL。 

---
# Role-Playing Evaluation for Large Language Models 

**Title (ZH)**: 大型语言模型的角色扮演评价 

**Authors**: Yassine El Boudouri, Walter Nuninger, Julian Alvarez, Yvan Peter  

**Link**: [PDF](https://arxiv.org/pdf/2505.13157)  

**Abstract**: Large Language Models (LLMs) demonstrate a notable capacity for adopting personas and engaging in role-playing. However, evaluating this ability presents significant challenges, as human assessments are resource-intensive and automated evaluations can be biased. To address this, we introduce Role-Playing Eval (RPEval), a novel benchmark designed to assess LLM role-playing capabilities across four key dimensions: emotional understanding, decision-making, moral alignment, and in-character consistency. This article details the construction of RPEval and presents baseline evaluations. Our code and dataset are available at this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）表现出接纳人设和角色扮演的能力。然而，评估这种能力面临显著挑战，因为人类评估资源密集，而自动评估可能会产生偏见。为解决这一问题，我们引入了角色扮演评估（RPEval），这是一种新型基准，旨在从情感理解、决策制定、道德对齐和入戏一致性四个关键维度评估LLM的角色扮演能力。本文详细介绍了RPEval的构建并呈现了基线评估。我们的代码和数据集可在以下链接获取：this https URL。 

---
# Tianyi: A Traditional Chinese Medicine all-rounder language model and its Real-World Clinical Practice 

**Title (ZH)**: 天意：中医药全科语言模型及其临床实践应用 

**Authors**: Zhi Liu, Tao Yang, Jing Wang, Yexin Chen, Zhan Gao, Jiaxi Yang, Kui Chen, Bingji Lu, Xiaochen Li, Changyong Luo, Yan Li, Xiaohong Gu, Peng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13156)  

**Abstract**: Natural medicines, particularly Traditional Chinese Medicine (TCM), are gaining global recognition for their therapeutic potential in addressing human symptoms and diseases. TCM, with its systematic theories and extensive practical experience, provides abundant resources for healthcare. However, the effective application of TCM requires precise syndrome diagnosis, determination of treatment principles, and prescription formulation, which demand decades of clinical expertise. Despite advancements in TCM-based decision systems, machine learning, and deep learning research, limitations in data and single-objective constraints hinder their practical application. In recent years, large language models (LLMs) have demonstrated potential in complex tasks, but lack specialization in TCM and face significant challenges, such as too big model scale to deploy and issues with hallucination. To address these challenges, we introduce Tianyi with 7.6-billion-parameter LLM, a model scale proper and specifically designed for TCM, pre-trained and fine-tuned on diverse TCM corpora, including classical texts, expert treatises, clinical records, and knowledge graphs. Tianyi is designed to assimilate interconnected and systematic TCM knowledge through a progressive learning manner. Additionally, we establish TCMEval, a comprehensive evaluation benchmark, to assess LLMs in TCM examinations, clinical tasks, domain-specific question-answering, and real-world trials. The extensive evaluations demonstrate the significant potential of Tianyi as an AI assistant in TCM clinical practice and research, bridging the gap between TCM knowledge and practical application. 

**Abstract (ZH)**: 自然药物，尤其是中国传统 medicine（TCM），因其在治疗人类症状和疾病方面的潜在疗效而获得全球认可。TCM凭借其系统理论和丰富的实践经验，提供了丰富的医疗资源。然而，TCM的有效应用需要精确的病证诊断、治疗原则确定和处方制定，这需要多年临床经验。尽管基于TCM的决策系统、机器学习和深度学习研究已经取得进展，但在数据限制和单一目标约束下，它们的实际应用受到限制。近年来，大规模语言模型（LLMs）在复杂任务中展现了潜力，但在TCM专业性方面仍面临重大挑战，如模型规模过大难以部署和幻觉问题。为应对这些挑战，我们引入了含76亿参数的天弈模型（Tianyi），其模型规模合理且专门针对TCM设计，基于多样化的TCM语料库进行预训练和微调，包括古典文献、专家著作、临床记录和知识图谱。天弈旨在通过渐进式学习方式吸收相互关联和系统的TCM知识。此外，我们建立了TCMEval综合评估基准，以评估LLM在TCM考试、临床任务、领域特定问答和真实世界试验中的性能。广泛的评估表明，天弈作为TCM临床实践和研究中的AI助手具有巨大潜力，缩小了TCM知识与实际应用之间的差距。 

---
# Temporal Distance-aware Transition Augmentation for Offline Model-based Reinforcement Learning 

**Title (ZH)**: 基于离线模型的强化学习的时序距离感知转换增强方法 

**Authors**: Dongsu Lee, Minhae Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2505.13144)  

**Abstract**: The goal of offline reinforcement learning (RL) is to extract a high-performance policy from the fixed datasets, minimizing performance degradation due to out-of-distribution (OOD) samples. Offline model-based RL (MBRL) is a promising approach that ameliorates OOD issues by enriching state-action transitions with augmentations synthesized via a learned dynamics model. Unfortunately, seminal offline MBRL methods often struggle in sparse-reward, long-horizon tasks. In this work, we introduce a novel MBRL framework, dubbed Temporal Distance-Aware Transition Augmentation (TempDATA), that generates augmented transitions in a temporally structured latent space rather than in raw state space. To model long-horizon behavior, TempDATA learns a latent abstraction that captures a temporal distance from both trajectory and transition levels of state space. Our experiments confirm that TempDATA outperforms previous offline MBRL methods and achieves matching or surpassing the performance of diffusion-based trajectory augmentation and goal-conditioned RL on the D4RL AntMaze, FrankaKitchen, CALVIN, and pixel-based FrankaKitchen. 

**Abstract (ZH)**: Offline Reinforcement Learning with Temporal Distance-Aware Transition Augmentation 

---
# ModernGBERT: German-only 1B Encoder Model Trained from Scratch 

**Title (ZH)**: ModernGBERT: 从头训练的1B编码器模型（仅限德语） 

**Authors**: Anton Ehrmanntraut, Julia Wunderle, Jan Pfister, Fotis Jannidis, Andreas Hotho  

**Link**: [PDF](https://arxiv.org/pdf/2505.13136)  

**Abstract**: Despite the prominence of decoder-only language models, encoders remain crucial for resource-constrained applications. We introduce ModernGBERT (134M, 1B), a fully transparent family of German encoder models trained from scratch, incorporating architectural innovations from ModernBERT. To evaluate the practical trade-offs of training encoders from scratch, we also present LLäMmlein2Vec (120M, 1B, 7B), a family of encoders derived from German decoder-only models via LLM2Vec. We benchmark all models on natural language understanding, text embedding, and long-context reasoning tasks, enabling a controlled comparison between dedicated encoders and converted decoders. Our results show that ModernGBERT 1B outperforms prior state-of-the-art German encoders as well as encoders adapted via LLM2Vec, with regard to performance and parameter-efficiency. All models, training data, checkpoints and code are publicly available, advancing the German NLP ecosystem with transparent, high-performance encoder models. 

**Abstract (ZH)**: 尽管解码器主导的语言模型备受瞩目，但编码器在资源受限的应用中仍然至关重要。我们介绍了ModernGBERT（134M，1B）这一完全透明的德语编码器模型家族，从头开始训练，整合了ModernBERT的架构创新。为了评估从头训练编码器的实际权衡，我们还呈现了LLäMmlein2Vec（120M，1B，7B）这一德语解码器模型家族，它是通过LLM2Vec转换自德语解码器模型。我们在自然语言理解、文本嵌入和长上下文推理任务中对所有模型进行了基准测试，从而使专用编码器与转换后的解码器之间能够进行受控比较。我们的结果显示，ModernGBERT 1B在性能和参数效率方面优于先前的最佳德语编码器以及通过LLM2Vec转换的编码器。所有模型、训练数据、检查点和代码均已公开，推动了透明、高性能德语NLP生态系统的建设。 

---
# Adaptive Image Restoration for Video Surveillance: A Real-Time Approach 

**Title (ZH)**: 视频监控中的自适应图像恢复：一种实时方法 

**Authors**: Muhammad Awais Amin, Adama Ilboudo, Abdul Samad bin Shahid, Amjad Ali, Waqas Haider Khan Bangyal  

**Link**: [PDF](https://arxiv.org/pdf/2505.13130)  

**Abstract**: One of the major challenges in the field of computer vision especially for detection, segmentation, recognition, monitoring, and automated solutions, is the quality of images. Image degradation, often caused by factors such as rain, fog, lighting, etc., has a negative impact on automated this http URL, several image restoration solutions exist, including restoration models for single degradation and restoration models for multiple degradations. However, these solutions are not suitable for real-time processing. In this study, the aim was to develop a real-time image restoration solution for video surveillance. To achieve this, using transfer learning with ResNet_50, we developed a model for automatically identifying the types of degradation present in an image to reference the necessary treatment(s) for image restoration. Our solution has the advantage of being flexible and scalable. 

**Abstract (ZH)**: 计算机视觉领域尤其是检测、分割、识别、监控和自动化解决方案中的一个主要挑战是图像质量。由于雨、雾、光照等因素导致的图像退化对自动化应用有负面影响。目前存在多种图像恢复方法，包括单种退化恢复模型和多种退化恢复模型。然而，这些方法不适合实时处理。本研究旨在开发一种适用于视频监控的实时图像恢复解决方案。通过使用ResNet_50进行迁移学习，我们开发了一种自动识别图像中存在退化类型的方法，以便参考必要的恢复处理。该解决方案具有灵活性和可扩展性的优势。 

---
# $μ$PC: Scaling Predictive Coding to 100+ Layer Networks 

**Title (ZH)**: $μ$PC: 将预测编码扩展至100多层网络 

**Authors**: Francesco Innocenti, El Mehdi Achour, Christopher L. Buckley  

**Link**: [PDF](https://arxiv.org/pdf/2505.13124)  

**Abstract**: The biological implausibility of backpropagation (BP) has motivated many alternative, brain-inspired algorithms that attempt to rely only on local information, such as predictive coding (PC) and equilibrium propagation. However, these algorithms have notoriously struggled to train very deep networks, preventing them from competing with BP in large-scale settings. Indeed, scaling PC networks (PCNs) has recently been posed as a challenge for the community (Pinchetti et al., 2024). Here, we show that 100+ layer PCNs can be trained reliably using a Depth-$\mu$P parameterisation (Yang et al., 2023; Bordelon et al., 2023) which we call "$\mu$PC". Through an extensive analysis of the scaling behaviour of PCNs, we reveal several pathologies that make standard PCNs difficult to train at large depths. We then show that, despite addressing only some of these instabilities, $\mu$PC allows stable training of very deep (up to 128-layer) residual networks on simple classification tasks with competitive performance and little tuning compared to current benchmarks. Moreover, $\mu$PC enables zero-shot transfer of both weight and activity learning rates across widths and depths. Our results have implications for other local algorithms and could be extended to convolutional and transformer architectures. Code for $\mu$PC is made available as part of a JAX library for PCNs at this https URL (Innocenti et al., 2024). 

**Abstract (ZH)**: 基于深度μPC参数化的学习能力探究：预测编码网络的稳定训练与零样本迁移 

---
# Just Dance with $π$! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection 

**Title (ZH)**: Just Dance with $π$! 一种多模态引ware器用于弱监督视频异常检测 

**Authors**: Snehashis Majhi, Giacomo D'Amicantonio, Antitza Dantcheva, Quan Kong, Lorenzo Garattoni, Gianpiero Francesca, Egor Bondarev, Francois Bremond  

**Link**: [PDF](https://arxiv.org/pdf/2505.13123)  

**Abstract**: Weakly-supervised methods for video anomaly detection (VAD) are conventionally based merely on RGB spatio-temporal features, which continues to limit their reliability in real-world scenarios. This is due to the fact that RGB-features are not sufficiently distinctive in setting apart categories such as shoplifting from visually similar events. Therefore, towards robust complex real-world VAD, it is essential to augment RGB spatio-temporal features by additional modalities. Motivated by this, we introduce the Poly-modal Induced framework for VAD: "PI-VAD", a novel approach that augments RGB representations by five additional modalities. Specifically, the modalities include sensitivity to fine-grained motion (Pose), three dimensional scene and entity representation (Depth), surrounding objects (Panoptic masks), global motion (optical flow), as well as language cues (VLM). Each modality represents an axis of a polygon, streamlined to add salient cues to RGB. PI-VAD includes two plug-in modules, namely Pseudo-modality Generation module and Cross Modal Induction module, which generate modality-specific prototypical representation and, thereby, induce multi-modal information into RGB cues. These modules operate by performing anomaly-aware auxiliary tasks and necessitate five modality backbones -- only during training. Notably, PI-VAD achieves state-of-the-art accuracy on three prominent VAD datasets encompassing real-world scenarios, without requiring the computational overhead of five modality backbones at inference. 

**Abstract (ZH)**: 多模态诱导的视频异常检测方法（PI-VAD） 

---
# When majority rules, minority loses: bias amplification of gradient descent 

**Title (ZH)**: 当多数决定时，少数受损：梯度下降的偏见放大效应 

**Authors**: François Bachoc, Jérôme Bolte, Ryan Boustany, Jean-Michel Loubes  

**Link**: [PDF](https://arxiv.org/pdf/2505.13122)  

**Abstract**: Despite growing empirical evidence of bias amplification in machine learning, its theoretical foundations remain poorly understood. We develop a formal framework for majority-minority learning tasks, showing how standard training can favor majority groups and produce stereotypical predictors that neglect minority-specific features. Assuming population and variance imbalance, our analysis reveals three key findings: (i) the close proximity between ``full-data'' and stereotypical predictors, (ii) the dominance of a region where training the entire model tends to merely learn the majority traits, and (iii) a lower bound on the additional training required. Our results are illustrated through experiments in deep learning for tabular and image classification tasks. 

**Abstract (ZH)**: 尽管有关机器学习中偏见放大的实证证据越来越多，但其理论基础仍然知之甚少。我们为多数-少数学习任务开发了一种形式化的框架，展示了标准训练如何倾向于 favor 多数群体并产生忽视少数群体特定特征的标准预测器。假设总体和方差不平衡，我们的分析揭示了三个关键发现：（i）“完整数据”预测器与标准预测器之间的密切接近性，（ii）一个区域，在该区域内训练整个模型往往仅学习多数群体的特征，（iii）额外训练所需的下限。我们的结果通过针对表格和图像分类任务的深度学习实验进行了说明。 

---
# Continuous Fair SMOTE -- Fairness-Aware Stream Learning from Imbalanced Data 

**Title (ZH)**: 连续公平SMOTE：面向不平衡数据的公平意识流学习 

**Authors**: Kathrin Lammers, Valerie Vaquet, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2505.13116)  

**Abstract**: As machine learning is increasingly applied in an online fashion to deal with evolving data streams, the fairness of these algorithms is a matter of growing ethical and legal concern. In many use cases, class imbalance in the data also needs to be dealt with to ensure predictive performance. Current fairness-aware stream learners typically attempt to solve these issues through in- or post-processing by focusing on optimizing one specific discrimination metric, addressing class imbalance in a separate processing step. While C-SMOTE is a highly effective model-agnostic pre-processing approach to mitigate class imbalance, as a side effect of this method, algorithmic bias is often introduced.
Therefore, we propose CFSMOTE - a fairness-aware, continuous SMOTE variant - as a pre-processing approach to simultaneously address the class imbalance and fairness concerns by employing situation testing and balancing fairness-relevant groups during oversampling. Unlike other fairness-aware stream learners, CFSMOTE is not optimizing for only one specific fairness metric, therefore avoiding potentially problematic trade-offs. Our experiments show significant improvement on several common group fairness metrics in comparison to vanilla C-SMOTE while maintaining competitive performance, also in comparison to other fairness-aware algorithms. 

**Abstract (ZH)**: 在线处理 evolving 数据流时，机器学习算法的公平性越来越成为伦理和法律上的关注点。许多应用场景中，数据的类别不平衡也需要得到处理以确保预测性能。当前大多数公平感知流学习器通常通过优化单一的歧视性指标并在单独的处理步骤中解决类别不平衡问题来应对这些挑战。尽管C-SMOTE作为一种高效的模型无偏预处理方法能够缓解类别不平衡问题，但由于该方法的副产品，算法偏见往往被引入。因此，我们提出CFSMOTE——一种公平感知的连续SMOTE变体——作为预处理方法，通过过采样期间的情况测试和平衡公平相关群体来同时解决类别不平衡和公平性问题。与仅优化单一公平性指标的其他公平感知流学习器不同，CFSMOTE避免了潜在的问题权衡。实验结果显示，CFSMOTE在多个常见的群体公平性指标上取得了显著改进，同时在与其他公平感知算法的比较中保持了竞争力。 

---
# Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning 

**Title (ZH)**: LALMs在时间推理领域中的基准测试与置信度评估 

**Authors**: Debarpan Bhattacharya, Apoorva Kulkarni, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2505.13115)  

**Abstract**: The popular success of text-based large language models (LLM) has streamlined the attention of the multimodal community to combine other modalities like vision and audio along with text to achieve similar multimodal capabilities. In this quest, large audio language models (LALMs) have to be evaluated on reasoning related tasks which are different from traditional classification or generation tasks. Towards this goal, we propose a novel dataset called temporal reasoning evaluation of audio (TREA).
We benchmark open-source LALMs and observe that they are consistently behind human capabilities on the tasks in the TREA dataset. While evaluating LALMs, we also propose an uncertainty metric, which computes the invariance of the model to semantically identical perturbations of the input. Our analysis shows that the accuracy and uncertainty metrics are not necessarily correlated and thus, points to a need for wholesome evaluation of LALMs for high-stakes applications. 

**Abstract (ZH)**: 基于文本的大型语言模型的流行成功吸引了多模态社区的注意力，以结合其他模态如视觉和音频，以实现类似的多模态能力。在此过程中，大型音频语言模型（LALMs）需要在与传统分类或生成任务不同的推理相关任务上进行评估。为了实现这一目标，我们提出了一种新的数据集，称为时间推理评估音频（TREA）。

我们对开源LALMs进行了基准测试，并观察到它们在TREA数据集的任务上始终落后于人类的能力。在评估LALMs时，我们还提出了一种不确定性度量方法，该方法计算模型对输入语义上相同扰动的不变性。我们的分析表明，准确性和不确定性度量并非总是相关的，这表明应全面评估LALMs以适应高风险应用。 

---
# FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference 

**Title (ZH)**: FreeKV：提升键值缓存检索以实现高效的LLM推理 

**Authors**: Guangda Liu, Chengwei Li, Zhenyu Ning, Jing Lin, Yiwu Yao, Danning Ke, Minyi Guo, Jieru Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13109)  

**Abstract**: Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过快速扩展上下文窗口得到了广泛应用，以支持日益 demanding的应用。然而，长上下文带来了重大的部署挑战，主要原因是键值缓存（KV缓存）的大小与上下文长度成正比增长。虽然提出了KV缓存压缩方法来解决此问题，但KV丢弃方法会导致显著的准确度损失，而KV检索方法则面临显著的效率瓶颈。我们提出FreeKV，一种算法与系统协同优化框架，旨在在保持准确度的同时提高KV检索效率。在算法方面，FreeKV引入了投机性检索，将KV的选择和召回过程移出关键路径，并结合细粒度校正以确保准确度。在系统方面，FreeKV采用CPU和GPU内存跨平台的混合KV布局，以消除碎片化数据传输，并利用双缓冲流式召回以进一步提高效率。实验表明，FreeKV在各种场景和模型下实现了几乎无损的准确度，并比目前最先进的KV检索方法快至13倍。 

---
# Lightweight Transformer via Unrolling of Mixed Graph Algorithms for Traffic Forecast 

**Title (ZH)**: 基于混合图算法拆解的轻量级Transformer用于交通预测 

**Authors**: Ji Qi, Tam Thuc Do, Mingxiao Liu, Zhuoshi Pan, Yuzhe Li, Gene Cheung, H. Vicky Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13102)  

**Abstract**: To forecast traffic with both spatial and temporal dimensions, we unroll a mixed-graph-based optimization algorithm into a lightweight and interpretable transformer-like neural net. Specifically, we construct two graphs: an undirected graph $\mathcal{G}^u$ capturing spatial correlations across geography, and a directed graph $\mathcal{G}^d$ capturing sequential relationships over time. We formulate a prediction problem for the future samples of signal $\mathbf{x}$, assuming it is "smooth" with respect to both $\mathcal{G}^u$ and $\mathcal{G}^d$, where we design new $\ell_2$ and $\ell_1$-norm variational terms to quantify and promote signal smoothness (low-frequency reconstruction) on a directed graph. We construct an iterative algorithm based on alternating direction method of multipliers (ADMM), and unroll it into a feed-forward network for data-driven parameter learning. We insert graph learning modules for $\mathcal{G}^u$ and $\mathcal{G}^d$, which are akin to the self-attention mechanism in classical transformers. Experiments show that our unrolled networks achieve competitive traffic forecast performance as state-of-the-art prediction schemes, while reducing parameter counts drastically. Our code is available in this https URL. 

**Abstract (ZH)**: 基于混合图的优化算法在时空维度上的交通流量预测：一种轻量级可解释的变压器-like 神经网络 

---
# ARIW-Framework: Adaptive Robust Iterative Watermarking Framework 

**Title (ZH)**: ARIW-框架：自适应 robust 迭代水印框架 

**Authors**: Shaowu Wu, Liting Zeng, Wei Lu, Xiangyang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13101)  

**Abstract**: With the rapid rise of large models, copyright protection for generated image content has become a critical security challenge. Although deep learning watermarking techniques offer an effective solution for digital image copyright protection, they still face limitations in terms of visual quality, robustness and generalization. To address these issues, this paper proposes an adaptive robust iterative watermarking framework (ARIW-Framework) that achieves high-quality watermarked images while maintaining exceptional robustness and generalization performance. Specifically, we introduce an iterative approach to optimize the encoder for generating robust residuals. The encoder incorporates noise layers and a decoder to compute robustness weights for residuals under various noise attacks. By employing a parallel optimization strategy, the framework enhances robustness against multiple types of noise attacks. Furthermore, we leverage image gradients to determine the embedding strength at each pixel location, significantly improving the visual quality of the watermarked images. Extensive experiments demonstrate that the proposed method achieves superior visual quality while exhibiting remarkable robustness and generalization against noise attacks. 

**Abstract (ZH)**: 基于生成图像内容的版权保护：一种自适应鲁棒迭代水印框架 

---
# Time-Frequency-Based Attention Cache Memory Model for Real-Time Speech Separation 

**Title (ZH)**: 基于时频注意力缓存记忆模型的实时语音分离 

**Authors**: Guo Chen, Kai Li, Runxuan Yang, Xiaolin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13094)  

**Abstract**: Existing causal speech separation models often underperform compared to non-causal models due to difficulties in retaining historical information. To address this, we propose the Time-Frequency Attention Cache Memory (TFACM) model, which effectively captures spatio-temporal relationships through an attention mechanism and cache memory (CM) for historical information storage. In TFACM, an LSTM layer captures frequency-relative positions, while causal modeling is applied to the time dimension using local and global representations. The CM module stores past information, and the causal attention refinement (CAR) module further enhances time-based feature representations for finer granularity. Experimental results showed that TFACM achieveed comparable performance to the SOTA TF-GridNet-Causal model, with significantly lower complexity and fewer trainable parameters. For more details, visit the project page: this https URL. 

**Abstract (ZH)**: 现有的因果语音分离模型往往由于难以保留历史信息而表现不佳。为了解决这一问题，我们提出了时间频率注意缓存记忆（TFACM）模型，该模型通过注意机制和缓存记忆（CM）有效地捕捉空时关系，并用于历史信息存储。在TFACM中，LSTM层捕捉频率相对位置，而因果建模则在时间维度上使用局部和全局表示。CM模块存储过去信息，而因果注意力 refinement（CAR）模块进一步增强基于时间的特征表示以实现更精细的粒度。实验结果显示，TFACM在复杂度显著降低且可训练参数更少的情况下，达到了与当前最先进的TF-GridNet-Causal模型相当的性能。欲了解更多信息，请参见项目页面：this https URL。 

---
# Graph Alignment for Benchmarking Graph Neural Networks and Learning Positional Encodings 

**Title (ZH)**: 图对齐用于图神经网络基准测试和学习位置编码 

**Authors**: Adrien Lagesse, Marc Lelarge  

**Link**: [PDF](https://arxiv.org/pdf/2505.13087)  

**Abstract**: We propose a novel benchmarking methodology for graph neural networks (GNNs) based on the graph alignment problem, a combinatorial optimization task that generalizes graph isomorphism by aligning two unlabeled graphs to maximize overlapping edges. We frame this problem as a self-supervised learning task and present several methods to generate graph alignment datasets using synthetic random graphs and real-world graph datasets from multiple domains. For a given graph dataset, we generate a family of graph alignment datasets with increasing difficulty, allowing us to rank the performance of various architectures. Our experiments indicate that anisotropic graph neural networks outperform standard convolutional architectures. To further demonstrate the utility of the graph alignment task, we show its effectiveness for unsupervised GNN pre-training, where the learned node embeddings outperform other positional encodings on three molecular regression tasks and achieve state-of-the-art results on the PCQM4Mv2 dataset with significantly fewer parameters. To support reproducibility and further research, we provide an open-source Python package to generate graph alignment datasets and benchmark new GNN architectures. 

**Abstract (ZH)**: 基于图对齐问题的图神经网络基准测试方法 

---
# MultiActor-Audiobook: Zero-Shot Audiobook Generation with Faces and Voices of Multiple Speakers 

**Title (ZH)**: 多演讲者有声书生成：基于多说话人面部和声音的零样本有声书生成 

**Authors**: Kyeongman Park, Seongho Joo, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.13082)  

**Abstract**: We introduce MultiActor-Audiobook, a zero-shot approach for generating audiobooks that automatically produces consistent, expressive, and speaker-appropriate prosody, including intonation and emotion. Previous audiobook systems have several limitations: they require users to manually configure the speaker's prosody, read each sentence with a monotonic tone compared to voice actors, or rely on costly training. However, our MultiActor-Audiobook addresses these issues by introducing two novel processes: (1) MSP (**Multimodal Speaker Persona Generation**) and (2) LSI (**LLM-based Script Instruction Generation**). With these two processes, MultiActor-Audiobook can generate more emotionally expressive audiobooks with a consistent speaker prosody without additional training. We compare our system with commercial products, through human and MLLM evaluations, achieving competitive results. Furthermore, we demonstrate the effectiveness of MSP and LSI through ablation studies. 

**Abstract (ZH)**: 多演员有声书：一种零-shot 方法，自动生成一致、表达丰富且适合演讲者风格的语调和情绪的有声书 

---
# Cross-modal Knowledge Transfer Learning as Graph Matching Based on Optimal Transport for ASR 

**Title (ZH)**: 基于最优传输的图匹配跨模态知识迁移学习用于ASR 

**Authors**: Xugang Lu, Peng Shen, Yu Tsao, Hisashi Kawai  

**Link**: [PDF](https://arxiv.org/pdf/2505.13079)  

**Abstract**: Transferring linguistic knowledge from a pretrained language model (PLM) to acoustic feature learning has proven effective in enhancing end-to-end automatic speech recognition (E2E-ASR). However, aligning representations between linguistic and acoustic modalities remains a challenge due to inherent modality gaps. Optimal transport (OT) has shown promise in mitigating these gaps by minimizing the Wasserstein distance (WD) between linguistic and acoustic feature distributions. However, previous OT-based methods overlook structural relationships, treating feature vectors as unordered sets. To address this, we propose Graph Matching Optimal Transport (GM-OT), which models linguistic and acoustic sequences as structured graphs. Nodes represent feature embeddings, while edges capture temporal and sequential relationships. GM-OT minimizes both WD (between nodes) and Gromov-Wasserstein distance (GWD) (between edges), leading to a fused Gromov-Wasserstein distance (FGWD) formulation. This enables structured alignment and more efficient knowledge transfer compared to existing OT-based approaches. Theoretical analysis further shows that prior OT-based methods in linguistic knowledge transfer can be viewed as a special case within our GM-OT framework. We evaluate GM-OT on Mandarin ASR using a CTC-based E2E-ASR system with a PLM for knowledge transfer. Experimental results demonstrate significant performance gains over state-of-the-art models, validating the effectiveness of our approach. 

**Abstract (ZH)**: 从预训练语言模型向声学特征学习转移语言知识 proven effective in 提升端到端自动语音识别 (E2E-ASR) 的性能。然而，由于固有的模态差距，语言和声学表示之间的对齐仍然是一个挑战。最优传输 (OT) 通过最小化语言和声学特征分布之间的 Wasserstein 距离 (WD) 显示出缓解这些差距的潜力。然而，以往的基于 OT 的方法忽略了结构关系，将特征向量视为无序集合。为了解决这一问题，我们提出了图匹配最优传输 (GM-OT)，将语言和声学序列建模为结构化图形。节点表示特征嵌入，边捕捉时间上的顺序关系。GM-OT 同时最小化节点间的 WD 和边间的 Gromov-Wasserstein 距离 (GWD)，形成融合的 Gromov-Wasserstein 距离 (FGWD) 表述，从而实现结构化的对齐并相比现有基于 OT 的方法更高效地传递知识。理论分析进一步表明，语言知识转移中的先前基于 OT 的方法可以在我们的 GM-OT 框架中被视为一种特殊情况。我们在一个基于 CTC 的端到端自动语音识别系统中使用语言模型进行知识转移，对 GM-OT 进行了 Mandarin ASR 的评估。实验结果表明，与最新模型相比，我们的方法显著提升了性能，验证了该方法的有效性。 

---
# Advancing Sequential Numerical Prediction in Autoregressive Models 

**Title (ZH)**: 在自回归模型中推进序列数值预测 

**Authors**: Xiang Fei, Jinghui Lu, Qi Sun, Hao Feng, Yanjie Wang, Wei Shi, An-Lan Wang, Jingqun Tang, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13077)  

**Abstract**: Autoregressive models have become the de facto choice for sequence generation tasks, but standard approaches treat digits as independent tokens and apply cross-entropy loss, overlooking the coherent structure of numerical sequences. This paper introduces Numerical Token Integrity Loss (NTIL) to address this gap. NTIL operates at two levels: (1) token-level, where it extends the Earth Mover's Distance (EMD) to preserve ordinal relationships between numerical values, and (2) sequence-level, where it penalizes the overall discrepancy between the predicted and actual sequences. This dual approach improves numerical prediction and integrates effectively with LLMs/MLLMs. Extensive experiments show significant performance improvements with NTIL. 

**Abstract (ZH)**: Numerical Token Integrity Loss for Enhanced Numerical Sequence Generation 

---
# The Hidden Dangers of Browsing AI Agents 

**Title (ZH)**: 浏览AI代理隐藏的危险 

**Authors**: Mykyta Mudryi, Markiyan Chaklosh, Grzegorz Wójcik  

**Link**: [PDF](https://arxiv.org/pdf/2505.13076)  

**Abstract**: Autonomous browsing agents powered by large language models (LLMs) are increasingly used to automate web-based tasks. However, their reliance on dynamic content, tool execution, and user-provided data exposes them to a broad attack surface. This paper presents a comprehensive security evaluation of such agents, focusing on systemic vulnerabilities across multiple architectural layers. Our work outlines the first end-to-end threat model for browsing agents and provides actionable guidance for securing their deployment in real-world environments. To address discovered threats, we propose a defense in depth strategy incorporating input sanitization, planner executor isolation, formal analyzers, and session safeguards. These measures protect against both initial access and post exploitation attack vectors. Through a white box analysis of a popular open source project, Browser Use, we demonstrate how untrusted web content can hijack agent behavior and lead to critical security breaches. Our findings include prompt injection, domain validation bypass, and credential exfiltration, evidenced by a disclosed CVE and a working proof of concept exploit. 

**Abstract (ZH)**: 由大规模语言模型驱动的自主浏览代理日益用于自动化基于Web的任务。然而，它们对动态内容、工具执行和用户提供的数据的依赖性暴露出广泛的攻击面。本文对这类代理进行了全面的安全评估，重点关注跨多个架构层的系统性漏洞。我们的工作提出了首个端到端的浏览代理威胁模型，并提供了在实际环境中部署代理时的安全指导。为了应对发现的威胁，我们提出了多层次防御策略，包括输入 sanitization、规划者执行器隔离、形式化分析器和会话保护措施。这些措施保护了初始访问和后利用攻击向量。通过对流行的开源项目Browser Use的白盒分析，我们展示了不可信的Web内容如何劫持代理行为并导致关键安全漏洞。我们的发现包括提示注入、域名验证绕过和凭证泄露，所述漏洞得到了披露的CVE和一个可行的漏洞利用示例的支持。 

---
# Structure-Aware Corpus Construction and User-Perception-Aligned Metrics for Large-Language-Model Code Completion 

**Title (ZH)**: 结构感知语料构建与用户感知导向的代码补全评估指标 

**Authors**: Dengfeng Liu, Jucai Zhai, Xiaoguang Jiang, Ziqun Li, Qianjin Yu, Feng Liu, Rui Ye, Huang Liu, Zhiguo Yang, Yongsheng Du, Fang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13073)  

**Abstract**: Code completion technology based on large language model has significantly improved the development efficiency of programmers. However, in practical applications, there remains a gap between current commonly used code completion evaluation metrics and users' actual perception. To address this issue, we propose two evaluation metrics for code completion tasks--LCP and ROUGE-LCP, from the perspective of probabilistic modeling. Furthermore, to tackle the lack of effective structural semantic modeling and cross-module dependency information in LLMs for repository-level code completion scenarios, we propose a data processing method based on a Structure-Preserving and Semantically-Reordered Code Graph (SPSR-Graph). Through theoretical analysis and experimental validation, we demonstrate the superiority of the proposed evaluation metrics in terms of user perception consistency, as well as the effectiveness of the data processing method in enhancing model performance. 

**Abstract (ZH)**: 基于大型语言模型的代码补全技术显著提高了程序员的开发效率。然而，在实际应用中，当前常用的代码补全评价指标与用户的实际感知之间仍存在差距。为解决这一问题，我们从概率建模的角度提出了两种代码补全任务的评价指标——LCP和ROUGE-LCP。为进一步解决大型语言模型在仓库级代码补全场景中缺乏有效的结构语义建模和跨模块依赖信息的问题，我们提出了一种基于结构保持和语义重排代码图（SPSR-Graph）的数据处理方法。通过理论分析和实验验证，我们展示了所提出评价指标在用户感知一致性方面的优越性，以及数据处理方法在提高模型性能方面的有效性。 

---
# SNAPE-PM: Building and Utilizing Dynamic Partner Models for Adaptive Explanation Generation 

**Title (ZH)**: SNAPE-PM：构建和利用动态合作伙伴模型进行自适应解释生成 

**Authors**: Amelie S. Robrecht, Christoph R. Kowalski, Stefan Kopp  

**Link**: [PDF](https://arxiv.org/pdf/2505.13053)  

**Abstract**: Adapting to the addressee is crucial for successful explanations, yet poses significant challenges for dialogsystems. We adopt the approach of treating explanation generation as a non-stationary decision process, where the optimal strategy varies according to changing beliefs about the explainee and the interaction context. In this paper we address the questions of (1) how to track the interaction context and the relevant listener features in a formally defined computational partner model, and (2) how to utilize this model in the dynamically adjusted, rational decision process that determines the currently best explanation strategy. We propose a Bayesian inference-based approach to continuously update the partner model based on user feedback, and a non-stationary Markov Decision Process to adjust decision-making based on the partner model values. We evaluate an implementation of this framework with five simulated interlocutors, demonstrating its effectiveness in adapting to different partners with constant and even changing feedback behavior. The results show high adaptivity with distinct explanation strategies emerging for different partners, highlighting the potential of our approach to improve explainable AI systems and dialogsystems in general. 

**Abstract (ZH)**: 适应受众对于成功解释至关重要，但对对话系统提出了重大挑战。我们采用将解释生成视为非 stationary 决策过程的方法，其中最优策略根据对解释对象的信念变化和互动背景进行调整。在本文中，我们探讨了两个问题：(1) 如何在正式定义的计算伙伴模型中跟踪互动背景和相关听众特征；(2) 如何利用该模型进行动态调整的理性决策过程，以确定当前最佳的解释策略。我们提出了一种基于贝叶斯推断的方法来根据用户反馈不断更新合作伙伴模型，并使用非 stationary 马尔可夫决策过程根据合作伙伴模型值调整决策过程。我们通过与五个模拟对话伙伴的实现框架进行评估，证明了该框架在不同不断变化的反馈行为伙伴中具有高度适应性，展示了我们方法在提高可解释人工智能系统和一般对话系统方面潜力。 

---
# A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation 

**Title (ZH)**: 泛化标签偏移视角下的跨域注视估计 

**Authors**: Hao-Ran Yang, Xiaohui Chen, Chuan-Xian Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.13043)  

**Abstract**: Aiming to generalize the well-trained gaze estimation model to new target domains, Cross-domain Gaze Estimation (CDGE) is developed for real-world application scenarios. Existing CDGE methods typically extract the domain-invariant features to mitigate domain shift in feature space, which is proved insufficient by Generalized Label Shift (GLS) theory. In this paper, we introduce a novel GLS perspective to CDGE and modelize the cross-domain problem by label and conditional shift problem. A GLS correction framework is presented and a feasible realization is proposed, in which a importance reweighting strategy based on truncated Gaussian distribution is introduced to overcome the continuity challenges in label shift correction. To embed the reweighted source distribution to conditional invariant learning, we further derive a probability-aware estimation of conditional operator discrepancy. Extensive experiments on standard CDGE tasks with different backbone models validate the superior generalization capability across domain and applicability on various models of proposed method. 

**Abstract (ZH)**: 跨域注视估计：基于广义标签转移视角的方法 

---
# KIT's Offline Speech Translation and Instruction Following Submission for IWSLT 2025 

**Title (ZH)**: KIT的离线语音翻译及指令跟随提交：IWSLT 2025 

**Authors**: Sai Koneru, Maike Züfle, Thai-Binh Nguyen, Seymanur Akti, Jan Niehues, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2505.13036)  

**Abstract**: The scope of the International Workshop on Spoken Language Translation (IWSLT) has recently broadened beyond traditional Speech Translation (ST) to encompass a wider array of tasks, including Speech Question Answering and Summarization. This shift is partly driven by the growing capabilities of modern systems, particularly with the success of Large Language Models (LLMs). In this paper, we present the Karlsruhe Institute of Technology's submissions for the Offline ST and Instruction Following (IF) tracks, where we leverage LLMs to enhance performance across all tasks. For the Offline ST track, we propose a pipeline that employs multiple automatic speech recognition systems, whose outputs are fused using an LLM with document-level context. This is followed by a two-step translation process, incorporating additional refinement step to improve translation quality. For the IF track, we develop an end-to-end model that integrates a speech encoder with an LLM to perform a wide range of instruction-following tasks. We complement it with a final document-level refinement stage to further enhance output quality by using contextual information. 

**Abstract (ZH)**: 国际口语翻译研讨会（IWSLT）的范围已从传统的口语翻译拓展到涵盖更多的任务，包括口语问答和摘要。随着大规模语言模型（LLMs）的成功，这一转变部分是由现代系统能力的不断提高驱动的。在本文中，我们介绍了卡尔斯鲁厄理工学院在离线口语翻译（Offline ST）和指令跟随（IF）赛道上的提交内容，我们利用LLMs来提高所有任务的表现。对于离线口语翻译轨道，我们提出了一种管道，该管道使用多个自动语音识别系统，并通过具有文档级上下文的LLM融合其输出。随后进行两步翻译过程，并增加额外的润色步骤以提高翻译质量。对于指令跟随轨道，我们开发了一种端到端模型，该模型将语音编码器与LLM结合，以执行各种指令跟随任务。并通过最终的文档级润色阶段进一步利用上下文信息提高输出质量。 

---
# TSPulse: Dual Space Tiny Pre-Trained Models for Rapid Time-Series Analysis 

**Title (ZH)**: TSPulse: 双空间Tiny预训练模型快速时间序列分析 

**Authors**: Vijay Ekambaram, Subodh Kumar, Arindam Jati, Sumanta Mukherjee, Tomoya Sakai, Pankaj Dayama, Wesley M. Gifford, Jayant Kalagnanam  

**Link**: [PDF](https://arxiv.org/pdf/2505.13033)  

**Abstract**: The rise of time-series pre-trained models has advanced temporal representation learning, but current state-of-the-art models are often large-scale, requiring substantial compute. We introduce TSPulse, ultra-compact time-series pre-trained models with only 1M parameters, specialized to perform strongly across classification, anomaly detection, imputation, and retrieval tasks. TSPulse introduces innovations at both the architecture and task levels. At the architecture level, it employs a dual-space masked reconstruction, learning from both time and frequency domains to capture complementary signals. This is further enhanced by a dual-embedding disentanglement, generating both detailed embeddings for fine-grained analysis and high-level semantic embeddings for broader task understanding. Notably, TSPulse's semantic embeddings are robust to shifts in time, magnitude, and noise, which is important for robust retrieval. At the task level, TSPulse incorporates TSLens, a fine-tuning component enabling task-specific feature attention. It also introduces a multi-head triangulation technique that correlates deviations from multiple prediction heads, enhancing anomaly detection by fusing complementary model outputs. Additionally, a hybrid mask pretraining is proposed to improves zero-shot imputation by reducing pre-training bias. These architecture and task innovations collectively contribute to TSPulse's significant performance gains: 5-16% on the UEA classification benchmarks, +20% on the TSB-AD anomaly detection leaderboard, +50% in zero-shot imputation, and +25% in time-series retrieval. Remarkably, these results are achieved with just 1M parameters, making TSPulse 10-100X smaller than existing pre-trained models. Its efficiency enables GPU-free inference and rapid pre-training, setting a new standard for efficient time-series pre-trained models. Models will be open-sourced soon. 

**Abstract (ZH)**: TSPulse： ultra-compact time-series pre-trained models for classification, anomaly detection, imputation, and retrieval 

---
# Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset 

**Title (ZH)**: 评估大规模语言模型安全解决方案的效果：帕利特基准数据集 

**Authors**: Sayon Palit, Daniel Woods  

**Link**: [PDF](https://arxiv.org/pdf/2505.13028)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model this http URL evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics. 

**Abstract (ZH)**: 大型语言模型安全性工具的全面比较研究 

---
# Step-wise Adaptive Integration of Supervised Fine-tuning and Reinforcement Learning for Task-Specific LLMs 

**Title (ZH)**: 面向特定任务的大型语言模型逐步自适应集成监督微调和强化学习方法 

**Authors**: Jack Chen, Fazhong Liu, Naruto Liu, Yuhan Luo, Erqu Qin, Harry Zheng, Tian Dong, Haojin Zhu, Yan Meng, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13026)  

**Abstract**: Large language models (LLMs) excel at mathematical reasoning and logical problem-solving. The current popular training paradigms primarily use supervised fine-tuning (SFT) and reinforcement learning (RL) to enhance the models' reasoning abilities. However, when using SFT or RL alone, there are respective challenges: SFT may suffer from overfitting, while RL is prone to mode collapse. The state-of-the-art methods have proposed hybrid training schemes. However, static switching faces challenges such as poor generalization across different tasks and high dependence on data quality. In response to these challenges, inspired by the curriculum learning-quiz mechanism in human reasoning cultivation, We propose SASR, a step-wise adaptive hybrid training framework that theoretically unifies SFT and RL and dynamically balances the two throughout optimization. SASR uses SFT for initial warm-up to establish basic reasoning skills, and then uses an adaptive dynamic adjustment algorithm based on gradient norm and divergence relative to the original distribution to seamlessly integrate SFT with the online RL method GRPO. By monitoring the training status of LLMs and adjusting the training process in sequence, SASR ensures a smooth transition between training schemes, maintaining core reasoning abilities while exploring different paths. Experimental results demonstrate that SASR outperforms SFT, RL, and static hybrid training methods. 

**Abstract (ZH)**: 大规模语言模型在数学推理和逻辑问题解决方面表现出色。当前流行的训练范式主要采用监督微调（SFT）和强化学习（RL）来增强模型的推理能力。然而，单独使用SFT或RL都会面临各自的挑战：SFT可能遭受过拟合，而RL容易出现模式崩溃。最新的方法提出了混合训练方案。但是，静态切换面临着跨不同任务的泛化能力差和对数据质量的高度依赖等问题。为应对这些挑战，受人类推理培养中课程学习-测试机制的启发，我们提出了一种逐步自适应混合训练框架SASR，该框架在理论上统一了SFT和RL，并在优化过程中动态平衡两者。SASR使用SFT进行初始暖启动以建立基本的推理技能，然后使用基于梯度范数和相对于原始分布的偏差的自适应动态调整算法，无缝地将SFT与在线RL方法GRPO结合。通过监控LLMs的训练状态并按顺序调整训练过程，SASR确保了训练方案之间的平滑过渡，既保持核心推理能力又探索不同的路径。实验结果表明，SASR优于SFT、RL和静态混合训练方法。 

---
# LiBOG: Lifelong Learning for Black-Box Optimizer Generation 

**Title (ZH)**: LiBOG: 黑盒优化器生成的终身学习方法 

**Authors**: Jiyuan Pei, Yi Mei, Jialin Liu, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13025)  

**Abstract**: Meta-Black-Box Optimization (MetaBBO) garners attention due to its success in automating the configuration and generation of black-box optimizers, significantly reducing the human effort required for optimizer design and discovering optimizers with higher performance than classic human-designed optimizers. However, existing MetaBBO methods conduct one-off training under the assumption that a stationary problem distribution with extensive and representative training problem samples is pre-available. This assumption is often impractical in real-world scenarios, where diverse problems following shifting distribution continually arise. Consequently, there is a pressing need for methods that can continuously learn from new problems encountered on-the-fly and progressively enhance their capabilities. In this work, we explore a novel paradigm of lifelong learning in MetaBBO and introduce LiBOG, a novel approach designed to learn from sequentially encountered problems and generate high-performance optimizers for Black-Box Optimization (BBO). LiBOG consolidates knowledge both across tasks and within tasks to mitigate catastrophic forgetting. Extensive experiments demonstrate LiBOG's effectiveness in learning to generate high-performance optimizers in a lifelong learning manner, addressing catastrophic forgetting while maintaining plasticity to learn new tasks. 

**Abstract (ZH)**: 元黑盒优化（MetaBBO）因其在自动化配置和生成黑盒优化器方面的成功而受到关注，显著减少了优化器设计所需的人力，并发现了性能超过经典人类设计优化器的优化器。然而，现有的MetaBBO方法在假设广泛且具代表性的预训练问题样本可供使用的情况下进行一次性训练。这在实际场景中往往不切实际，因为不断变化的问题分布和多样化的问题持续出现。因此，迫切需要能够在遇到新问题时持续学习并逐步提升能力的方法。在本文中，我们探索了元黑盒优化中的终身学习新范式，并引入了LiBOG，这是一种设计用于从连续遇到的问题中学习并生成高性能黑盒优化（BBO）优化器的新方法。LiBOG 融合了跨任务和同任务的知识以减轻灾难性遗忘。大量实验表明，LiBOG 在终身学习模式下有效地学习生成高性能优化器，同时处理灾难性遗忘并保持学习新任务的能力。 

---
# Anti-Inpainting: A Proactive Defense against Malicious Diffusion-based Inpainters under Unknown Conditions 

**Title (ZH)**: 未知条件下对抗恶意扩散修复模型的主动防御：Anti-Inpainting 

**Authors**: Yimao Guo, Zuomin Qu, Wei Lu, Xiangyang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13023)  

**Abstract**: As diffusion-based malicious image manipulation becomes increasingly prevalent, multiple proactive defense methods are developed to safeguard images against unauthorized tampering. However, most proactive defense methods only can safeguard images against manipulation under known conditions, and fail to protect images from manipulations guided by tampering conditions crafted by malicious users. To tackle this issue, we propose Anti-Inpainting, a proactive defense method that achieves adequate protection under unknown conditions through a triple mechanism to address this challenge. Specifically, a multi-level deep feature extractor is presented to obtain intricate features during the diffusion denoising process to improve protective effectiveness. We design multi-scale semantic-preserving data augmentation to enhance the transferability of adversarial perturbations across unknown conditions by multi-scale transformations while preserving semantic integrity. In addition, we propose a selection-based distribution deviation optimization strategy to improve the protection of adversarial perturbation against manipulation under diverse random seeds. Extensive experiments indicate the proactive defensive performance of Anti-Inpainting against diffusion-based inpainters guided by unknown conditions in InpaintGuardBench and CelebA-HQ. At the same time, we also demonstrate the proposed approach's robustness under various image purification methods and its transferability across different versions of diffusion models. 

**Abstract (ZH)**: 基于扩散的恶意图像修补防护方法 

---
# To Bias or Not to Bias: Detecting bias in News with bias-detector 

**Title (ZH)**: 偏见还是无偏见：使用偏见检测器检测新闻中的偏见 

**Authors**: Himel Ghosh, Ahmed Mosharafa, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2505.13010)  

**Abstract**: Media bias detection is a critical task in ensuring fair and balanced information dissemination, yet it remains challenging due to the subjectivity of bias and the scarcity of high-quality annotated data. In this work, we perform sentence-level bias classification by fine-tuning a RoBERTa-based model on the expert-annotated BABE dataset. Using McNemar's test and the 5x2 cross-validation paired t-test, we show statistically significant improvements in performance when comparing our model to a domain-adaptively pre-trained DA-RoBERTa baseline. Furthermore, attention-based analysis shows that our model avoids common pitfalls like oversensitivity to politically charged terms and instead attends more meaningfully to contextually relevant tokens. For a comprehensive examination of media bias, we present a pipeline that combines our model with an already-existing bias-type classifier. Our method exhibits good generalization and interpretability, despite being constrained by sentence-level analysis and dataset size because of a lack of larger and more advanced bias corpora. We talk about context-aware modeling, bias neutralization, and advanced bias type classification as potential future directions. Our findings contribute to building more robust, explainable, and socially responsible NLP systems for media bias detection. 

**Abstract (ZH)**: 媒体偏见检测是确保公平和平衡信息传播的关键任务，但由于偏见的主观性及高质量标注数据的稀缺性，这一任务仍然颇具挑战性。在本文中，我们通过对专家标注的BABE数据集进行RoBERTa基模型的微调，在句级偏见分类任务上进行研究。通过McNemar检验和5x2交叉验证配对t检验，我们展示了在性能上相较于域适应预训练的DA-RoBERTa基线模型的统计显著性改进。此外，基于注意力机制的分析表明，我们的模型避免了对政治色彩词汇的过度敏感，而是更关注于上下文相关性较强的词汇。为全面研究媒体偏见，我们提出了一种结合我们模型和已有偏见类型分类器的管道。尽管受限于句级分析和数据集规模，我们的方法仍展现出良好的泛化能力和可解释性，并讨论了情境感知建模、偏见中和及高级偏见类型分类在未来的发展方向。我们的研究成果为构建更具鲁棒性、可解释性和社会责任感的语言模型以进行媒体偏见检测提供了贡献。 

---
# ExTrans: Multilingual Deep Reasoning Translation via Exemplar-Enhanced Reinforcement Learning 

**Title (ZH)**: ExTrans: 通过范例增强强化学习的多语言深度推理翻译 

**Authors**: Jiaan Wang, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12996)  

**Abstract**: In recent years, the emergence of large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, has shown impressive capabilities in complex problems, e.g., mathematics and coding. Some pioneering studies attempt to bring the success of LRMs in neural machine translation (MT). They try to build LRMs with deep reasoning MT ability via reinforcement learning (RL). Despite some progress that has been made, these attempts generally focus on several high-resource languages, e.g., English and Chinese, leaving the performance on other languages unclear. Besides, the reward modeling methods in previous work do not fully unleash the potential of reinforcement learning in MT. In this work, we first design a new reward modeling method that compares the translation results of the policy MT model with a strong LRM (i.e., DeepSeek-R1-671B), and quantifies the comparisons to provide rewards. Experimental results demonstrate the superiority of the reward modeling method. Using Qwen2.5-7B-Instruct as the backbone, the trained model achieves the new state-of-the-art performance in literary translation, and outperforms strong LRMs including OpenAI-o1 and DeepSeeK-R1. Furthermore, we extend our method to the multilingual settings with 11 languages. With a carefully designed lightweight reward modeling in RL, we can simply transfer the strong MT ability from a single direction into multiple (i.e., 90) translation directions and achieve impressive multilingual MT performance. 

**Abstract (ZH)**: 近年来，大型推理模型（LRMs）如OpenAI-o1和DeepSeek-R1在复杂问题上的表现，例如数学和编码方面展现出令人印象深刻的能力。一些开创性研究试图将LRMs的成功应用到神经机器翻译（MT）中。它们尝试通过强化学习（RL）构建具有深度推理MT能力的LRMs。尽管取得了一些进展，但这些尝试通常主要关注一些资源丰富的语言，例如英语和汉语，其他语言的表现则不明确。此外，以往工作中使用的奖励建模方法并未充分挖掘强化学习在MT中的潜力。在这项工作中，我们首先设计了一种新的奖励建模方法，该方法将策略MT模型的翻译结果与强大的LRM（即DeepSeek-R1-671B）进行比较，并量化这种比较以提供奖励。实验结果表明该奖励建模方法具有优越性。使用Qwen2.5-7B-Instruct作为主干，训练后的模型在文学翻译方面达到了新的最佳性能，并超越了包括OpenAI-o1和DeepSeek-R1在内的强大LRMs。此外，我们将方法扩展至多语言环境，包括11种语言。通过精心设计的轻量级奖励建模在RL中，可以简单地将单一方向的强MT能力扩展到多个方向（即90个翻译方向），从而实现令人印象深刻的语言间机器翻译性能。 

---
# Fractured Chain-of-Thought Reasoning 

**Title (ZH)**: 断裂的思维链 reasoning 

**Authors**: Baohao Liao, Hanze Dong, Yuhui Xu, Doyen Sahoo, Christof Monz, Junnan Li, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12992)  

**Abstract**: Inference-time scaling techniques have significantly bolstered the reasoning capabilities of large language models (LLMs) by harnessing additional computational effort at inference without retraining. Similarly, Chain-of-Thought (CoT) prompting and its extension, Long CoT, improve accuracy by generating rich intermediate reasoning trajectories, but these approaches incur substantial token costs that impede their deployment in latency-sensitive settings. In this work, we first show that truncated CoT, which stops reasoning before completion and directly generates the final answer, often matches full CoT sampling while using dramatically fewer tokens. Building on this insight, we introduce Fractured Sampling, a unified inference-time strategy that interpolates between full CoT and solution-only sampling along three orthogonal axes: (1) the number of reasoning trajectories, (2) the number of final solutions per trajectory, and (3) the depth at which reasoning traces are truncated. Through extensive experiments on five diverse reasoning benchmarks and several model scales, we demonstrate that Fractured Sampling consistently achieves superior accuracy-cost trade-offs, yielding steep log-linear scaling gains in Pass@k versus token budget. Our analysis reveals how to allocate computation across these dimensions to maximize performance, paving the way for more efficient and scalable LLM reasoning. 

**Abstract (ZH)**: 推理时缩放技术通过在推理时利用额外的计算资源来显著增强大型语言模型的推理能力，而无需重新训练。类似地，带有推理链（Chain-of-Thought，CoT）的提示及其扩展Long CoT通过生成丰富的中间推理轨迹来提高准确性，但这些方法会带来显著的标记成本，阻碍了它们在延迟敏感设置中的部署。在本文中，我们首先展示了截断CoT的方法，该方法在推理未完成时直接生成最终答案，通常与完整的CoT抽样具有类似的效果，但使用了大幅减少的标记。在此基础上，我们提出了一种统一的推理时策略——断层抽样（Fractured Sampling），该策略在推理轨迹的数量、每条轨迹的最终解决方案数量以及推理轨迹的截断深度这三个正交维度上插值，以实现完整的CoT和仅解题抽样的平衡。通过在五个不同的推理基准和多个模型规模上的广泛实验，我们证明了断层抽样在准确性和成本之间始终实现了更优的权衡，使得在标记预算上的累进线性扩展收益更加显著。我们的分析揭示了如何在这些维度上分配计算资源以最大化性能，从而为更高效和可扩展的大型语言模型推理铺平了道路。 

---
# An Empirical Study of Many-to-Many Summarization with Large Language Models 

**Title (ZH)**: 大型语言模型中多对多总结的实证研究 

**Authors**: Jiaan Wang, Fandong Meng, Zengkui Sun, Yunlong Liang, Yuxuan Cao, Jiarong Xu, Haoxiang Shi, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12983)  

**Abstract**: Many-to-many summarization (M2MS) aims to process documents in any language and generate the corresponding summaries also in any language. Recently, large language models (LLMs) have shown strong multi-lingual abilities, giving them the potential to perform M2MS in real applications. This work presents a systematic empirical study on LLMs' M2MS ability. Specifically, we first reorganize M2MS data based on eight previous domain-specific datasets. The reorganized data contains 47.8K samples spanning five domains and six languages, which could be used to train and evaluate LLMs. Then, we benchmark 18 LLMs in a zero-shot manner and an instruction-tuning manner. Fine-tuned traditional models (e.g., mBART) are also conducted for comparisons. Our experiments reveal that, zero-shot LLMs achieve competitive results with fine-tuned traditional models. After instruct-tuning, open-source LLMs can significantly improve their M2MS ability, and outperform zero-shot LLMs (including GPT-4) in terms of automatic evaluations. In addition, we demonstrate that this task-specific improvement does not sacrifice the LLMs' general task-solving abilities. However, as revealed by our human evaluation, LLMs still face the factuality issue, and the instruction tuning might intensify the issue. Thus, how to control factual errors becomes the key when building LLM summarizers in real applications, and is worth noting in future research. 

**Abstract (ZH)**: 多语言到多语言摘要生成能力的系统性 empirical 研究：大规模语言模型的潜力与挑战 

---
# From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents 

**Title (ZH)**: 从助手到对手：探索移动LLM代理的安全风险 

**Authors**: Liangxuan Wu, Chao Wang, Tianming Liu, Yanjie Zhao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12981)  

**Abstract**: The growing adoption of large language models (LLMs) has led to a new paradigm in mobile computing--LLM-powered mobile AI agents--capable of decomposing and automating complex tasks directly on smartphones. However, the security implications of these agents remain largely unexplored. In this paper, we present the first comprehensive security analysis of mobile LLM agents, encompassing three representative categories: System-level AI Agents developed by original equipment manufacturers (e.g., YOYO Assistant), Third-party Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g., Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile agents and identifying security threats across three core capability dimensions: language-based reasoning, GUI-based interaction, and system-level execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the unique capabilities and interaction patterns of mobile LLM agents, and spanning their entire operational lifecycle. To investigate these threats in practice, we introduce AgentScan, a semi-automated security analysis framework that systematically evaluates mobile LLM agents across all 11 attack scenarios. Applying AgentScan to nine widely deployed agents, we uncover a concerning trend: every agent is vulnerable to targeted attacks. In the most severe cases, agents exhibit vulnerabilities across eight distinct attack vectors. These attacks can cause behavioral deviations, privacy leakage, or even full execution hijacking. Based on these findings, we propose a set of defensive design principles and practical recommendations for building secure mobile LLM agents. Our disclosures have received positive feedback from two major device vendors. Overall, this work highlights the urgent need for standardized security practices in the fast-evolving landscape of LLM-driven mobile automation. 

**Abstract (ZH)**: 大型语言模型在移动计算中的广泛应用催生了以LLM为动力的移动AI代理，这些代理能够在智能手机上直接分解和自动化复杂任务。然而，这些代理的安全性影响尚未得到充分探索。本文首次对移动LLM代理进行了全面的安全分析，涵盖三大代表类别：系统级AI代理（如YOYO Assistant）、第三方通用代理（如Zhipu AI AutoGLM）和新兴代理框架（如阿里移动代理）。我们首先分析移动代理的一般工作流程，并在语言推理、GUI交互和系统级执行三个核心能力维度上识别安全威胁。我们的分析揭示了11个不同的攻击面，均源于移动LLM代理的独特能力和交互模式，并贯穿其整个运营生命周期。为了在实际中调查这些威胁，我们引入了AgentScan，这是一种半自动化的安全分析框架，系统地评估所有11种攻击场景下的移动LLM代理。将AgentScan应用于九个广泛部署的代理后，我们发现了一个令人担忧的趋势：每个代理都容易受到定向攻击。在最严重的案例中，代理在八个不同的攻击向量上均存在漏洞。这些攻击可能导致行为偏差、隐私泄露，甚至全面执行劫持。基于这些发现，我们提出了构建安全移动LLM代理的一整套防御设计原则和实用建议。我们的披露得到了两家主要设备供应商的积极反馈。总体而言，本项工作强调了在以LLM为驱动的移动自动化快速发展的环境中，需要标准化的安全实践。 

---
# Multiscale Adaptive Conflict-Balancing Model For Multimedia Deepfake Detection 

**Title (ZH)**: 多尺度自适应冲突平衡模型 для 多媒体深伪检测 

**Authors**: Zihan Xiong, Xiaohua Wu, Lei Chen, Fangqi Lou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12966)  

**Abstract**: Advances in computer vision and deep learning have blurred the line between deepfakes and authentic media, undermining multimedia credibility through audio-visual forgery. Current multimodal detection methods remain limited by unbalanced learning between modalities. To tackle this issue, we propose an Audio-Visual Joint Learning Method (MACB-DF) to better mitigate modality conflicts and neglect by leveraging contrastive learning to assist in multi-level and cross-modal fusion, thereby fully balancing and exploiting information from each modality. Additionally, we designed an orthogonalization-multimodal pareto module that preserves unimodal information while addressing gradient conflicts in audio-video encoders caused by differing optimization targets of the loss functions. Extensive experiments and ablation studies conducted on mainstream deepfake datasets demonstrate consistent performance gains of our model across key evaluation metrics, achieving an average accuracy of 95.5% across multiple datasets. Notably, our method exhibits superior cross-dataset generalization capabilities, with absolute improvements of 8.0% and 7.7% in ACC scores over the previous best-performing approach when trained on DFDC and tested on DefakeAVMiT and FakeAVCeleb datasets. 

**Abstract (ZH)**: 计算机视觉和深度学习的进步已模糊了深fake与真实媒体之间的界限，通过音视频伪造损害了多媒体的可信度。现有的多模态检测方法仍受限于模态间学习的不平衡。为解决这一问题，我们提出了一种音视频联合学习方法（MACB-DF），通过对比学习辅助跨模态融合，以更好地缓解模态间的冲突和忽视，从而全面平衡和利用每种模态的信息。此外，我们设计了一种正交化-多模态帕累托模块，该模块在保留单模态信息的同时，解决了由于损失函数优化目标差异而在音视频编码器中引起的梯度冲突。在主流深fake数据集上的广泛实验和消融研究表明，我们的模型在关键评估指标上均表现出一致的性能提升，平均准确率为95.5%。值得注意的是，我们的方法在跨数据集泛化能力上表现出色，在DFDC数据集上训练，在DefakeAVMiT和FakeAVCeleb数据集上测试时，ACC分数绝对提升分别为8.0%和7.7%，超越了之前表现最好的方法。 

---
# Segmentation of temporomandibular joint structures on mri images using neural networks for diagnosis of pathologies 

**Title (ZH)**: 使用神经网络对磁共振图像中的颞下颌关节结构进行分割以诊断病理变化 

**Authors**: Maksim I. Ivanov, Olga E. Mendybaeva, Yuri E. Karyakin, Igor N. Glukhikh, Aleksey V. Lebedev  

**Link**: [PDF](https://arxiv.org/pdf/2505.12963)  

**Abstract**: This article explores the use of artificial intelligence for the diagnosis of pathologies of the temporomandibular joint (TMJ), in particular, for the segmentation of the articular disc on MRI images. The relevance of the work is due to the high prevalence of TMJ pathologies, as well as the need to improve the accuracy and speed of diagnosis in medical institutions. During the study, the existing solutions (Diagnocat, MandSeg) were analyzed, which, as a result, are not suitable for studying the articular disc due to the orientation towards bone structures. To solve the problem, an original dataset was collected from 94 images with the classes "temporomandibular joint" and "jaw". To increase the amount of data, augmentation methods were used. After that, the models of U-Net, YOLOv8n, YOLOv11n and Roboflow neural networks were trained and compared. The evaluation was carried out according to the Dice Score, Precision, Sensitivity, Specificity, and Mean Average Precision metrics. The results confirm the potential of using the Roboflow model for segmentation of the temporomandibular joint. In the future, it is planned to develop an algorithm for measuring the distance between the jaws and determining the position of the articular disc, which will improve the diagnosis of TMJ pathologies. 

**Abstract (ZH)**: 本研究探讨了人工智能在颞下颌关节病理诊断中的应用，特别是 joint disc 在 MRI 图像上的分割。由于颞下颌关节病理的高发病率以及提高医疗机构诊断准确性和速度的需要，这项工作具有重要意义。在研究过程中，分析了现有解决方案（Diagnocat, MandSeg），但它们主要针对骨结构，不适用于关节盘研究。为此，收集了一个包括94张图像的原始数据集，分为“颞下颌关节”和“颌”两类。为增加数据量，使用了数据增强方法。之后，训练并比较了 U-Net、YOLOv8n、YOLOv11n 和 Roboflow 神经网络模型，评估指标包括 Dice 分数、精确度、灵敏度、特异度和平均精确度。结果证实了使用 Roboflow 模型进行颞下颌关节分割的潜力。未来计划开发测量上下颌距离和确定关节盘位置的算法，以改善颞下颌关节病理的诊断。 

---
# Hardware-Adaptive and Superlinear-Capacity Memristor-based Associative Memory 

**Title (ZH)**: 基于 memristor 的自适应硬件和超线性容量关联存储 

**Authors**: Chengping He, Mingrui Jiang, Keyi Shan, Szu-Hao Yang, Zefan Li, Shengbo Wang, Giacomo Pedretti, Jim Ignowski, Can Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12960)  

**Abstract**: Brain-inspired computing aims to mimic cognitive functions like associative memory, the ability to recall complete patterns from partial cues. Memristor technology offers promising hardware for such neuromorphic systems due to its potential for efficient in-memory analog computing. Hopfield Neural Networks (HNNs) are a classic model for associative memory, but implementations on conventional hardware suffer from efficiency bottlenecks, while prior memristor-based HNNs faced challenges with vulnerability to hardware defects due to offline training, limited storage capacity, and difficulty processing analog patterns. Here we introduce and experimentally demonstrate on integrated memristor hardware a new hardware-adaptive learning algorithm for associative memories that significantly improves defect tolerance and capacity, and naturally extends to scalable multilayer architectures capable of handling both binary and continuous patterns. Our approach achieves 3x effective capacity under 50% device faults compared to state-of-the-art methods. Furthermore, its extension to multilayer architectures enables superlinear capacity scaling (\(\propto N^{1.49}\ for binary patterns) and effective recalling of continuous patterns (\propto N^{1.74}\ scaling), as compared to linear capacity scaling for previous HNNs. It also provides flexibility to adjust capacity by tuning hidden neurons for the same-sized patterns. By leveraging the massive parallelism of the hardware enabled by synchronous updates, it reduces energy by 8.8x and latency by 99.7% for 64-dimensional patterns over asynchronous schemes, with greater improvements at scale. This promises the development of more reliable memristor-based associative memory systems and enables new applications research due to the significantly improved capacity, efficiency, and flexibility. 

**Abstract (ZH)**: 基于大脑的计算旨在模拟联想记忆等认知功能，并从部分提示中召回完整模式。 memristor 技术由于其在内存中进行高效模拟计算的潜力，为这类神经形态系统提供了有希望的硬件解决方案。Hopfield 神经网络（HNNs）是一种经典的联想记忆模型，但在传统硬件上的实现面临效率瓶颈，同时，基于 memristor 的 HNNs 因离线训练导致的硬件缺陷易感性、有限的存储容量以及难以处理模拟模式等问题而面临挑战。我们在此引入并在集成 memristor 硬件上实验演示了一种新的硬件自适应学习算法，该算法显著提高了健壮性和容量，并自然扩展到能够处理二进制和连续模式的可扩展多层架构。我们的方法在50%器件故障下实现了相比现有方法3倍的有效容量。此外，其扩展到多层架构的能力使其容量实现超线性扩展（对于二进制模式为 \(\propto N^{1.49}\)）、有效恢复连续模式（对于连续模式为 \(\propto N^{1.74}\)），而此前的 HNNs 则具有线性容量扩展。它还提供了在相同规模下通过调整隐藏神经元来灵活调整容量的灵活性。通过利用同步更新启用的硬件中的大规模并行性，对于64维模式，它将异步方案的能量减少了8.8倍，并将延迟降低了99.7%，并且在更大规模时效果更佳。这有望开发更可靠的基于 memristor 的联想记忆系统，并由于其显著增强的容量、效率和灵活性，能够促进新的应用研究。 

---
# DGRO: Enhancing LLM Reasoning via Exploration-Exploitation Control and Reward Variance Management 

**Title (ZH)**: DGRO：通过探索-利用控制和奖励方差管理提升LLM推理能力 

**Authors**: Xuerui Su, Liya Guo, Yue Wang, Yi Zhu, Zhiming Ma, Zun Wang, Yuting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12951)  

**Abstract**: Inference scaling further accelerates Large Language Models (LLMs) toward Artificial General Intelligence (AGI), with large-scale Reinforcement Learning (RL) to unleash long Chain-of-Thought reasoning. Most contemporary reasoning approaches usually rely on handcrafted rule-based reward functions. However, the tarde-offs of exploration and exploitation in RL algorithms involves multiple complex considerations, and the theoretical and empirical impacts of manually designed reward functions remain insufficiently explored. In this paper, we propose Decoupled Group Reward Optimization (DGRO), a general RL algorithm for LLM reasoning. On the one hand, DGRO decouples the traditional regularization coefficient into two independent hyperparameters: one scales the policy gradient term, and the other regulates the distance from the sampling policy. This decoupling not only enables precise control over balancing exploration and exploitation, but also can be seamlessly extended to Online Policy Mirror Descent (OPMD) algorithms in Kimi k1.5 and Direct Reward Optimization. On the other hand, we observe that reward variance significantly affects both convergence speed and final model performance. We conduct both theoretical analysis and extensive empirical validation to assess DGRO, including a detailed ablation study that investigates its performance and optimization dynamics. Experimental results show that DGRO achieves state-of-the-art performance on the Logic dataset with an average accuracy of 96.9\%, and demonstrates strong generalization across mathematical benchmarks. 

**Abstract (ZH)**: 推理放大进一步加速了大型语言模型（LLMs）向人工通用智能（AGI）的发展，通过大规模强化学习（RL）释放长链推理能力。大多数当代推理方法通常依赖于手工制作的基于规则的奖励函数。然而，RL算法中的探索与利用权衡涉及多种复杂的考虑因素，手动设计的奖励函数的理论和实证影响尚未得到充分探索。在本文中，我们提出了一种适用于LLM推理的通用RL算法——解耦组奖励优化（DGRO）。一方面，DGRO将传统的正则化系数解耦为两个独立的超参数：一个调整策略梯度项，另一个调节采样策略的距离。这种解耦不仅使得精确控制探索与利用之间的权衡成为可能，还可以无缝扩展到Kimi k1.5中的在线策略镜像下降（OPMD）算法和直接奖励优化中。另一方面，我们观察到奖励方差显著影响收敛速度和最终模型性能。我们通过理论分析和广泛的实证验证评估了DGRO，包括详细剖析其性能和优化动态。实验结果表明，DGRO在逻辑数据集上的平均准确率达到96.9%，并在数学基准测试中展示了较强的泛化能力。 

---
# CALM-PDE: Continuous and Adaptive Convolutions for Latent Space Modeling of Time-dependent PDEs 

**Title (ZH)**: CALM-PDE：时变偏微分方程潜空间建模中的连续自适应卷积 

**Authors**: Jan Hagnberger, Daniel Musekamp, Mathias Niepert  

**Link**: [PDF](https://arxiv.org/pdf/2505.12944)  

**Abstract**: Solving time-dependent Partial Differential Equations (PDEs) using a densely discretized spatial domain is a fundamental problem in various scientific and engineering disciplines, including modeling climate phenomena and fluid dynamics. However, performing these computations directly in the physical space often incurs significant computational costs. To address this issue, several neural surrogate models have been developed that operate in a compressed latent space to solve the PDE. While these approaches reduce computational complexity, they often use Transformer-based attention mechanisms to handle irregularly sampled domains, resulting in increased memory consumption. In contrast, convolutional neural networks allow memory-efficient encoding and decoding but are limited to regular discretizations. Motivated by these considerations, we propose CALM-PDE, a model class that efficiently solves arbitrarily discretized PDEs in a compressed latent space. We introduce a novel continuous convolution-based encoder-decoder architecture that uses an epsilon-neighborhood-constrained kernel and learns to apply the convolution operator to adaptive and optimized query points. We demonstrate the effectiveness of CALM-PDE on a diverse set of PDEs with both regularly and irregularly sampled spatial domains. CALM-PDE is competitive with or outperforms existing baseline methods while offering significant improvements in memory and inference time efficiency compared to Transformer-based methods. 

**Abstract (ZH)**: 使用密集离散空间求解时空依赖偏微分方程（PDEs）是各种科学和工程学科中的基础问题，包括气候现象 modeling 和流体动力学。然而，在物理空间中直接进行这些计算往往会产生显著的计算成本。为应对这一问题，已经开发出了几种在压缩隐空间中操作的神经代理模型来求解PDE。尽管这些方法能降低计算复杂度，但它们通常使用基于Transformer的注意力机制来处理不规则采样的域，导致内存消耗增加。相比之下，卷积神经网络允许高效的数据编码和解码，但受限于规则离散化。鉴于这些考虑，我们提出了一种 CALM-PDE 模型类，该模型类能够在压缩隐空间中高效求解任意离散化的PDE。我们引入了一种新颖的基于连续卷积的编码-解码架构，使用epsilon-邻域约束核，并学习将卷积操作应用于自适应和优化的查询点。我们在具有规则和不规则采样空间域的多种PDE上展示了CALM-PDE的有效性。与基于Transformer的方法相比，CALM-PDE 在内存和推断时间效率方面提供了显著的改进，同时在与现有基线方法的竞争中表现相当或更优。 

---
# A3 : an Analytical Low-Rank Approximation Framework for Attention 

**Title (ZH)**: A3：注意力的分析性低秩逼近框架 

**Authors**: Jeffrey T. H. Wong, Cheng Zhang, Xinye Cao, Pedro Gimenes, George A. Constantinides, Wayne Luk, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12942)  

**Abstract**: Large language models have demonstrated remarkable performance; however, their massive parameter counts make deployment highly expensive. Low-rank approximation offers a promising compression solution, yet existing approaches have two main limitations: (1) They focus on minimizing the output error of individual linear layers, without considering the architectural characteristics of Transformers, and (2) they decompose a large weight matrix into two small low-rank matrices. Consequently, these methods often fall short compared to other compression techniques like pruning and quantization, and introduce runtime overhead such as the extra GEMM kernel launches for decomposed small matrices. To address these limitations, we propose $\tt A^\tt 3$, a post-training low-rank approximation framework. $\tt A^\tt 3$ splits a Transformer layer into three functional components, namely $\tt QK$, $\tt OV$, and $\tt MLP$. For each component, $\tt A^\tt 3$ provides an analytical solution that reduces the hidden dimension size inside each component while minimizing the component's functional loss ($\it i.e.$, error in attention scores, attention outputs, and MLP outputs). This approach directly reduces model sizes, KV cache sizes, and FLOPs without introducing any runtime overheads. In addition, it provides a new narrative in advancing the optimization problem from singular linear layer loss optimization toward improved end-to-end performance. Through extensive experiments, we show that $\tt A^\tt 3$ maintains superior performance compared to SoTAs. For example, under the same reduction budget in computation and memory, our low-rank approximated LLaMA 3.1-70B achieves a perplexity of 4.69 on WikiText-2, outperforming the previous SoTA's 7.87 by 3.18. We also demonstrate the versatility of $\tt A^\tt 3$, including KV cache compression, quantization, and mixed-rank assignments for enhanced performance. 

**Abstract (ZH)**: 大型语言模型展现了卓越的性能，但其庞大的参数数量使部署成本高昂。低秩逼近提供了一种有前景的压缩解决方案，然而现有方法存在两个主要局限：（1）它们专注于最小化单个线性层的输出误差，而不考虑Transformer的架构特性；（2）它们将一个大权重矩阵分解为两个小的低秩矩阵。因此，这些方法通常在与其他压缩技术（如剪枝和量化）的比较中表现不佳，并且引入了运行时开销，如分解小矩阵所需的额外GEMM内核启动。为了解决这些局限，我们提出了一种训练后低秩逼近框架$\tt A^\tt 3$。$\tt A^\tt 3$将Transformer层划分为三个功能组件，即$\tt QK$、$\tt OV$和$\tt MLP$。对于每个组件，$\tt A^\tt 3$提供了一种分析解决方案，可以在减少每个组件的隐藏维度大小的同时最小化该组件的功能损失（即注意力分数、注意力输出和MLP输出的误差）。这种方法直接减小了模型大小、KV缓存大小和FLOPs，而没有任何运行时开销。此外，它提供了一种新思路，即将单一线性层损失优化的优化问题推进到整体性能的改进。通过大量实验，我们展示了$\tt A^\tt 3$相对于当前最佳性能保持了更优异的表现。例如，在相同的计算和内存减少预算下，我们低秩逼近的LLaMA 3.1-70B在WikiText-2上的困惑度为4.69，优于前一项SoTA的7.87，提高了3.18。我们还展示了$\tt A^\tt 3$的通用性，包括KV缓存压缩、量化和混合秩分配以增强性能。 

---
# Leveraging LLM Inconsistency to Boost Pass@k Performance 

**Title (ZH)**: 利用大模型不一致性提升Pass@k性能 

**Authors**: Uri Dalal, Meirav Segal, Zvika Ben-Haim, Dan Lahav, Omer Nevo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12938)  

**Abstract**: Large language models (LLMs) achieve impressive abilities in numerous domains, but exhibit inconsistent performance in response to minor input changes. Rather than view this as a drawback, in this paper we introduce a novel method for leveraging models' inconsistency to boost Pass@k performance. Specifically, we present a "Variator" agent that generates k variants of a given task and submits one candidate solution for each one. Our variant generation approach is applicable to a wide range of domains as it is task agnostic and compatible with free-form inputs. We demonstrate the efficacy of our agent theoretically using a probabilistic model of the inconsistency effect, and show empirically that it outperforms the baseline on the APPS dataset. Furthermore, we establish that inconsistency persists even in frontier reasoning models across coding and cybersecurity domains, suggesting our method is likely to remain relevant for future model generations. 

**Abstract (ZH)**: 大型语言模型在响应小的输入变化时表现出不一致的性能，但这一特性可以被利用以提升Pass@k性能：一种“变异者”代理的方法研究 

---
# Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs 

**Title (ZH)**: 不要让低概率标记在大型语言模型的RL中过度主导 

**Authors**: Zhihe Yang, Xufang Luo, Zilong Wang, Dongqi Han, Zhiyuan He, Dongsheng Li, Yunjian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12929)  

**Abstract**: Reinforcement learning (RL) has become a cornerstone for enhancing the reasoning capabilities of large language models (LLMs), with recent innovations such as Group Relative Policy Optimization (GRPO) demonstrating exceptional effectiveness. In this study, we identify a critical yet underexplored issue in RL training: low-probability tokens disproportionately influence model updates due to their large gradient magnitudes. This dominance hinders the effective learning of high-probability tokens, whose gradients are essential for LLMs' performance but are substantially suppressed. To mitigate this interference, we propose two novel methods: Advantage Reweighting and Low-Probability Token Isolation (Lopti), both of which effectively attenuate gradients from low-probability tokens while emphasizing parameter updates driven by high-probability tokens. Our approaches promote balanced updates across tokens with varying probabilities, thereby enhancing the efficiency of RL training. Experimental results demonstrate that they substantially improve the performance of GRPO-trained LLMs, achieving up to a 46.2% improvement in K&K Logic Puzzle reasoning tasks. Our implementation is available at this https URL. 

**Abstract (ZH)**: 强化学习（RL）已成为提升大规模语言模型（LLMs）推理能力的基石，近期的技术创新，如组相对策略优化（GRPO），展示了卓越的效果。本研究识别了一个在RL训练中尚未充分探索的关键问题：低概率令牌由于其较大的梯度幅度，不成比例地影响模型更新，这阻碍了高概率令牌的有效学习，而这些令牌的梯度对于LLMs的性能至关重要，但它们的梯度被大幅抑制。为减轻这种干扰，我们提出了两种新颖的方法：优势重加权和低概率令牌隔离（Lopti），这两种方法有效衰减了低概率令牌的梯度，同时强调了由高概率令牌驱动的参数更新。我们的方法促进了不同概率令牌之间的均衡更新，从而提高RL训练的效率。实验结果表明，这显著提高了GRPO训练的LLMs的表现，在K&K逻辑谜题推理任务中实现了高达46.2%的性能提升。我们的实现可在以下链接获取：这个 https URL。 

---
# CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming 

**Title (ZH)**: CPRet: 一个用于竞赛编程检索的数据集、基准和模型 

**Authors**: Han Deng, Yuan Meng, Shixiang Tang, Wanli Ouyang, Xinzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.12925)  

**Abstract**: Competitive programming benchmarks are widely used in scenarios such as programming contests and large language model assessments. However, the growing presence of duplicate or highly similar problems raises concerns not only about competition fairness, but also about the validity of competitive programming as a benchmark for model evaluation. In this paper, we propose a new problem -- similar question retrieval -- to address this issue. Due to the lack of both data and models, solving this problem is challenging. To this end, we introduce CPRet, a retrieval-oriented benchmark suite for competitive programming, covering four retrieval tasks: two code-centric (i.e., Text-to-Code and Code-to-Code) and two newly proposed problem-centric tasks (i.e., Problem-to-Duplicate and Simplified-to-Full), built from a combination of automatically crawled problem-solution data and manually curated annotations. Our contribution includes both high-quality training data and temporally separated test sets for reliable evaluation. In addition, we develop two task-specialized retrievers based on this dataset: CPRetriever-Code, trained with a novel Group-InfoNCE loss for problem-code alignment, and CPRetriever-Prob, fine-tuned for identifying problem-level similarity. Both models achieve strong results and are open-sourced for local use. Finally, we analyze LiveCodeBench and find that high-similarity problems inflate model pass rates and reduce differentiation, underscoring the need for similarity-aware evaluation in future benchmarks.
Code and data are available at: this https URL 

**Abstract (ZH)**: 竞赛编程中相似问题检索基准在编程竞赛和大型语言模型评估中的应用：解决重复或高度相似问题带来的挑战 

---
# PyFCG: Fluid Construction Grammar in Python 

**Title (ZH)**: PyFCG: 流动构式语法的Python实现 

**Authors**: Paul Van Eecke, Katrien Beuls  

**Link**: [PDF](https://arxiv.org/pdf/2505.12920)  

**Abstract**: We present PyFCG, an open source software library that ports Fluid Construction Grammar (FCG) to the Python programming language. PyFCG enables its users to seamlessly integrate FCG functionality into Python programs, and to use FCG in combination with other libraries within Python's rich ecosystem. Apart from a general description of the library, this paper provides three walkthrough tutorials that demonstrate example usage of PyFCG in typical use cases of FCG: (i) formalising and testing construction grammar analyses, (ii) learning usage-based construction grammars from corpora, and (iii) implementing agent-based experiments on emergent communication. 

**Abstract (ZH)**: 我们介绍PyFCG，一个开源软件库，将流体构造语法（FCG）移植到Python编程语言。PyFCG使用户能够无缝将FCG功能集成到Python程序中，并在Python丰富的生态系统中与其他库结合使用FCG。除了一般描述该库外，本文还提供了三个逐步示例教程，展示了在典型FCG用例中使用PyFCG的方式：(i) 形式化和测试构造语法分析，(ii) 从语料库学习基于使用的构造语法，(iii) 实施数字化代理实验以实现 emergent 通信。 

---
# SourceDetMamba: A Graph-aware State Space Model for Source Detection in Sequential Hypergraphs 

**Title (ZH)**: 源检测Mamba：一种基于图的状态空间模型在序贯超图中的源检测方法 

**Authors**: Le Cheng, Peican Zhu, Yangming Guo, Chao Gao, Zhen Wang, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12910)  

**Abstract**: Source detection on graphs has demonstrated high efficacy in identifying rumor origins. Despite advances in machine learning-based methods, many fail to capture intrinsic dynamics of rumor propagation. In this work, we present SourceDetMamba: A Graph-aware State Space Model for Source Detection in Sequential Hypergraphs, which harnesses the recent success of the state space model Mamba, known for its superior global modeling capabilities and computational efficiency, to address this challenge. Specifically, we first employ hypergraphs to model high-order interactions within social networks. Subsequently, temporal network snapshots generated during the propagation process are sequentially fed in reverse order into Mamba to infer underlying propagation dynamics. Finally, to empower the sequential model to effectively capture propagation patterns while integrating structural information, we propose a novel graph-aware state update mechanism, wherein the state of each node is propagated and refined by both temporal dependencies and topological context. Extensive evaluations on eight datasets demonstrate that SourceDetMamba consistently outperforms state-of-the-art approaches. 

**Abstract (ZH)**: 基于图的来源检测方法在识别谣言源头方面表现出高效率。尽管机器学习方法取得了进展，但仍有许多方法未能捕捉谣言传播的内在动态。在本文中，我们提出了一种基于图的状态空间模型SourceDetMamba：用于序列超图中来源检测，该模型利用状态空间模型Mamba的优越全局建模能力和计算效率，以解决这一挑战。具体而言，我们首先使用超图来建模社交网络中的高阶交互。随后，传播过程中生成的时序网络快照按反序输入Mamba，以推断潜在的传播动态。最后，为了使序列模型能够有效捕捉传播模式并集成结构信息，我们提出了一种新型的基于图的状态更新机制，在这种机制中，每个节点的状态通过时间和拓扑上下文被传播和优化。在八个数据集上的广泛评估表明，SourceDetMamba持续优于现有方法。 

---
# Sinusoidal Initialization, Time for a New Start 

**Title (ZH)**: 正弦初始化，是时候开始新篇了 

**Authors**: Alberto Fernández-Hernández, Jose I. Mestre, Manuel F. Dolz, Jose Duato, Enrique S. Quintana-Ortí  

**Link**: [PDF](https://arxiv.org/pdf/2505.12909)  

**Abstract**: Initialization plays a critical role in Deep Neural Network training, directly influencing convergence, stability, and generalization. Common approaches such as Glorot and He initializations rely on randomness, which can produce uneven weight distributions across layer connections. In this paper, we introduce the Sinusoidal initialization, a novel deterministic method that employs sinusoidal functions to construct structured weight matrices expressly to improve the spread and balance of weights throughout the network while simultaneously fostering a more uniform, well-conditioned distribution of neuron activation states from the very first forward pass. Because Sinusoidal initialization begins with weights and activations that are already evenly and efficiently utilized, it delivers consistently faster convergence, greater training stability, and higher final accuracy across a wide range of models, including convolutional neural networks, vision transformers, and large language models. On average, our experiments show an increase of 4.8 % in final validation accuracy and 20.9 % in convergence speed. By replacing randomness with structure, this initialization provides a stronger and more reliable foundation for Deep Learning systems. 

**Abstract (ZH)**: Sinusoidal Initialization: A Deterministic Method for Improved Weight Distribution and Faster Convergence in Deep Neural Networks 

---
# Dynamic Graph Induced Contour-aware Heat Conduction Network for Event-based Object Detection 

**Title (ZH)**: 基于事件驱动的对象检测的动态图诱导轮廓感知热传导网络 

**Authors**: Xiao Wang, Yu Jin, Lan Chen, Bo Jiang, Lin Zhu, Yonghong Tian, Jin Tang, Bin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12908)  

**Abstract**: Event-based Vision Sensors (EVS) have demonstrated significant advantages over traditional RGB frame-based cameras in low-light conditions, high-speed motion capture, and low latency. Consequently, object detection based on EVS has attracted increasing attention from researchers. Current event stream object detection algorithms are typically built upon Convolutional Neural Networks (CNNs) or Transformers, which either capture limited local features using convolutional filters or incur high computational costs due to the utilization of self-attention. Recently proposed vision heat conduction backbone networks have shown a good balance between efficiency and accuracy; however, these models are not specifically designed for event stream data. They exhibit weak capability in modeling object contour information and fail to exploit the benefits of multi-scale features. To address these issues, this paper proposes a novel dynamic graph induced contour-aware heat conduction network for event stream based object detection, termed CvHeat-DET. The proposed model effectively leverages the clear contour information inherent in event streams to predict the thermal diffusivity coefficients within the heat conduction model, and integrates hierarchical structural graph features to enhance feature learning across multiple scales. Extensive experiments on three benchmark datasets for event stream-based object detection fully validated the effectiveness of the proposed model. The source code of this paper will be released on this https URL. 

**Abstract (ZH)**: 基于事件流的 Vision 热传导感知网络 CvHeat-DET 用于目标检测 

---
# The Computation of Generalized Embeddings for Underwater Acoustic Target Recognition using Contrastive Learning 

**Title (ZH)**: 使用对比学习计算水下声学目标识别的广义嵌入计算 

**Authors**: Hilde I. Hummel, Arwin Gansekoele, Sandjai Bhulai, Rob van der Mei  

**Link**: [PDF](https://arxiv.org/pdf/2505.12904)  

**Abstract**: The increasing level of sound pollution in marine environments poses an increased threat to ocean health, making it crucial to monitor underwater noise. By monitoring this noise, the sources responsible for this pollution can be mapped. Monitoring is performed by passively listening to these sounds. This generates a large amount of data records, capturing a mix of sound sources such as ship activities and marine mammal vocalizations. Although machine learning offers a promising solution for automatic sound classification, current state-of-the-art methods implement supervised learning. This requires a large amount of high-quality labeled data that is not publicly available. In contrast, a massive amount of lower-quality unlabeled data is publicly available, offering the opportunity to explore unsupervised learning techniques. This research explores this possibility by implementing an unsupervised Contrastive Learning approach. Here, a Conformer-based encoder is optimized by the so-called Variance-Invariance-Covariance Regularization loss function on these lower-quality unlabeled data and the translation to the labeled data is made. Through classification tasks involving recognizing ship types and marine mammal vocalizations, our method demonstrates to produce robust and generalized embeddings. This shows to potential of unsupervised methods for various automatic underwater acoustic analysis tasks. 

**Abstract (ZH)**: 海洋环境中噪声污染水平的提高对海洋健康构成了更大的威胁，监测 underwater 噪音至关重要。通过监测这些噪音，可以确定污染源并对其进行映射。监测是通过被动聆听这些声音来完成的。这会产生大量的数据记录，捕捉到如船只活动和海洋哺乳动物 vocalizations 等混合声音源。尽管机器学习为自动声音分类提供了有希望的解决方案，但当前最先进的方法是监督学习，这需要大量高质量的标记数据，而这些数据并未公开。相比之下，大量低质量的未标记数据是公开可用的，这为探索无监督学习技术提供了机会。本研究通过实施一种无监督对比学习方法来探索这种可能性。在此方法中，通过所谓的方差不变性协方差正则化损失函数优化基于 Conformer 的编码器，并将其应用于低质量未标记数据，然后将其转换为标记数据。通过识别船只类型和海洋哺乳动物 vocalizations 等分类任务，我们的方法展示了无监督方法在各种自动水下声学分析任务中的潜力。 

---
# Towards Low-Latency Event Stream-based Visual Object Tracking: A Slow-Fast Approach 

**Title (ZH)**: 基于事件流的低延迟视觉目标跟踪：一种慢速-快速方法 

**Authors**: Shiao Wang, Xiao Wang, Liye Jin, Bo Jiang, Lin Zhu, Lan Chen, Yonghong Tian, Bin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12903)  

**Abstract**: Existing tracking algorithms typically rely on low-frame-rate RGB cameras coupled with computationally intensive deep neural network architectures to achieve effective tracking. However, such frame-based methods inherently face challenges in achieving low-latency performance and often fail in resource-constrained environments. Visual object tracking using bio-inspired event cameras has emerged as a promising research direction in recent years, offering distinct advantages for low-latency applications. In this paper, we propose a novel Slow-Fast Tracking paradigm that flexibly adapts to different operational requirements, termed SFTrack. The proposed framework supports two complementary modes, i.e., a high-precision slow tracker for scenarios with sufficient computational resources, and an efficient fast tracker tailored for latency-aware, resource-constrained environments. Specifically, our framework first performs graph-based representation learning from high-temporal-resolution event streams, and then integrates the learned graph-structured information into two FlashAttention-based vision backbones, yielding the slow and fast trackers, respectively. The fast tracker achieves low latency through a lightweight network design and by producing multiple bounding box outputs in a single forward pass. Finally, we seamlessly combine both trackers via supervised fine-tuning and further enhance the fast tracker's performance through a knowledge distillation strategy. Extensive experiments on public benchmarks, including FE240, COESOT, and EventVOT, demonstrate the effectiveness and efficiency of our proposed method across different real-world scenarios. The source code has been released on this https URL. 

**Abstract (ZH)**: 基于生物启发型事件摄像头的新型慢速-快速跟踪 paradigm：SFTrack 

---
# AutoGEEval: A Multimodal and Automated Framework for Geospatial Code Generation on GEE with Large Language Models 

**Title (ZH)**: AutoGEEval：一种基于大型语言模型的多模态自动化地理空间代码生成框架（GEE） 

**Authors**: Shuyang Hou, Zhangxiao Shen, Huayi Wu, Jianyuan Liang, Haoyue Jiao, Yaxian Qing, Xiaopu Zhang, Xu Li, Zhipeng Gui, Xuefeng Guan, Longgang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12900)  

**Abstract**: Geospatial code generation is emerging as a key direction in the integration of artificial intelligence and geoscientific analysis. However, there remains a lack of standardized tools for automatic evaluation in this domain. To address this gap, we propose AutoGEEval, the first multimodal, unit-level automated evaluation framework for geospatial code generation tasks on the Google Earth Engine (GEE) platform powered by large language models (LLMs). Built upon the GEE Python API, AutoGEEval establishes a benchmark suite (AutoGEEval-Bench) comprising 1325 test cases that span 26 GEE data types. The framework integrates both question generation and answer verification components to enable an end-to-end automated evaluation pipeline-from function invocation to execution validation. AutoGEEval supports multidimensional quantitative analysis of model outputs in terms of accuracy, resource consumption, execution efficiency, and error types. We evaluate 18 state-of-the-art LLMs-including general-purpose, reasoning-augmented, code-centric, and geoscience-specialized models-revealing their performance characteristics and potential optimization pathways in GEE code generation. This work provides a unified protocol and foundational resource for the development and assessment of geospatial code generation models, advancing the frontier of automated natural language to domain-specific code translation. 

**Abstract (ZH)**: 地理空间代码生成正逐渐成为人工智能与地质科学分析集成的关键方向。然而，在这一领域仍缺乏标准化的自动评估工具。为解决这一问题，我们提出了AutoGEEval，这是一个基于大型语言模型的首个针对Google Earth Engine平台的多模态单元级自动评估框架，用于地理空间代码生成任务。AutoGEEval基于GEE Python API，构建了一个包含1325个测试案例的基准套件（AutoGEEval-Bench），涵盖了26种GEE数据类型。该框架整合了问题生成和答案验证组件，从函数调用到执行验证，实现端到端的自动化评估流水线。AutoGEEval从准确度、资源消耗、执行效率和错误类型等方面支持多维度的模型输出分析。我们评估了18种最先进的大型语言模型，包括通用型、推理增强型、代码中心型和地质科学专用型模型，揭示了它们在GEE代码生成中的性能特征及其优化途径。这项工作为地理空间代码生成模型的开发和评估提供了一个统一的协议和基础资源，推进了自然语言到特定领域代码自动转换的前沿。 

---
# HyperDet: Source Detection in Hypergraphs via Interactive Relationship Construction and Feature-rich Attention Fusion 

**Title (ZH)**: HyperDet：超图中的源检测via交互关系构建和丰富特征注意力融合 

**Authors**: Le Cheng, Peican Zhu, Yangming Guo, Keke Tang, Chao Gao, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12894)  

**Abstract**: Hypergraphs offer superior modeling capabilities for social networks, particularly in capturing group phenomena that extend beyond pairwise interactions in rumor propagation. Existing approaches in rumor source detection predominantly focus on dyadic interactions, which inadequately address the complexity of more intricate relational structures. In this study, we present a novel approach for Source Detection in Hypergraphs (HyperDet) via Interactive Relationship Construction and Feature-rich Attention Fusion. Specifically, our methodology employs an Interactive Relationship Construction module to accurately model both the static topology and dynamic interactions among users, followed by the Feature-rich Attention Fusion module, which autonomously learns node features and discriminates between nodes using a self-attention mechanism, thereby effectively learning node representations under the framework of accurately modeled higher-order relationships. Extensive experimental validation confirms the efficacy of our HyperDet approach, showcasing its superiority relative to current state-of-the-art methods. 

**Abstract (ZH)**: 基于交互关系构建与特征丰富注意力融合的超图来源检测（HyperDet） 

---
# TinyAlign: Boosting Lightweight Vision-Language Models by Mitigating Modal Alignment Bottlenecks 

**Title (ZH)**: TinyAlign: 通过缓解模态对齐瓶颈增强轻量级视觉-语言模型 

**Authors**: Yuanze Hu, Zhaoxin Fan, Xinyu Wang, Gen Li, Ye Qiu, Zhichao Yang, Wenjun Wu, Kejian Wu, Yifan Sun, Xiaotie Deng, Jin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12884)  

**Abstract**: Lightweight Vision-Language Models (VLMs) are indispensable for resource-constrained applications. The prevailing approach to aligning vision and language models involves freezing both the vision encoder and the language model while training small connector modules. However, this strategy heavily depends on the intrinsic capabilities of the language model, which can be suboptimal for lightweight models with limited representational capacity. In this work, we investigate this alignment bottleneck through the lens of mutual information, demonstrating that the constrained capacity of the language model inherently limits the Effective Mutual Information (EMI) between multimodal inputs and outputs, thereby compromising alignment quality. To address this challenge, we propose TinyAlign, a novel framework inspired by Retrieval-Augmented Generation, which strategically retrieves relevant context from a memory bank to enrich multimodal inputs and enhance their alignment. Extensive empirical evaluations reveal that TinyAlign significantly reduces training loss, accelerates convergence, and enhances task performance. Remarkably, it allows models to achieve baseline-level performance with only 40\% of the fine-tuning data, highlighting exceptional data efficiency. Our work thus offers a practical pathway for developing more capable lightweight VLMs while introducing a fresh theoretical lens to better understand and address alignment bottlenecks in constrained multimodal systems. 

**Abstract (ZH)**: 轻量级视觉-语言模型（VLMs）对于资源受限的应用至关重要。现有的视觉和语言模型对齐方法通常涉及冻结视觉编码器和语言模型，同时训练小型连接模块。然而，这种策略高度依赖于语言模型的内在能力，对于具有有限表征能力的轻量级模型来说可能是不理想的。本文通过互信息的视角探讨了这种对齐瓶颈，证明了语言模型的受限容量固有限制了多模态输入与输出的有效互信息（EMI），从而影响了对齐质量。为了解决这一挑战，我们提出了一种名为TinyAlign的新型框架，该框架借鉴了检索增强生成的思想，战略性地从记忆库中检索相关上下文以丰富多模态输入并增强其对齐。广泛的实证评估表明，TinyAlign显著降低了训练损失，加速了收敛，并提升了任务性能。值得注意的是，它仅需40％的微调数据即可使模型达到基线水平性能，突显了其出色的数据效率。因此，本文为开发更能力的轻量级VLMs提供了一条实用路径，并引入了一种新的理论视角来更好地理解并解决受限多模态系统中的对齐瓶颈。 

---
# PhyDA: Physics-Guided Diffusion Models for Data Assimilation in Atmospheric Systems 

**Title (ZH)**: PhyDA: 物理引导的扩散模型在大气系统数据同化中的应用 

**Authors**: Hao Wang, Jindong Han, Wei Fan, Weijia Zhang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12882)  

**Abstract**: Data Assimilation (DA) plays a critical role in atmospheric science by reconstructing spatially continous estimates of the system state, which serves as initial conditions for scientific analysis. While recent advances in diffusion models have shown great potential for DA tasks, most existing approaches remain purely data-driven and often overlook the physical laws that govern complex atmospheric dynamics. As a result, they may yield physically inconsistent reconstructions that impair downstream applications. To overcome this limitation, we propose PhyDA, a physics-guided diffusion framework designed to ensure physical coherence in atmospheric data assimilation. PhyDA introduces two key components: (1) a Physically Regularized Diffusion Objective that integrates physical constraints into the training process by penalizing deviations from known physical laws expressed as partial differential equations, and (2) a Virtual Reconstruction Encoder that bridges observational sparsity for structured latent representations, further enhancing the model's ability to infer complete and physically coherent states. Experiments on the ERA5 reanalysis dataset demonstrate that PhyDA achieves superior accuracy and better physical plausibility compared to state-of-the-art baselines. Our results emphasize the importance of combining generative modeling with domain-specific physical knowledge and show that PhyDA offers a promising direction for improving real-world data assimilation systems. 

**Abstract (ZH)**: 物理学指导的数据同化(PhyDA):一种确保大气数据同化物理一致性的物理引导扩散框架 

---
# AdS-GNN -- a Conformally Equivariant Graph Neural Network 

**Title (ZH)**: AdS-GNN —— 保齐性同伴的图神经网络 

**Authors**: Maksim Zhdanov, Nabil Iqbal, Erik Bekkers, Patrick Forré  

**Link**: [PDF](https://arxiv.org/pdf/2505.12880)  

**Abstract**: Conformal symmetries, i.e.\ coordinate transformations that preserve angles, play a key role in many fields, including physics, mathematics, computer vision and (geometric) machine learning. Here we build a neural network that is equivariant under general conformal transformations. To achieve this, we lift data from flat Euclidean space to Anti de Sitter (AdS) space. This allows us to exploit a known correspondence between conformal transformations of flat space and isometric transformations on the AdS space. We then build upon the fact that such isometric transformations have been extensively studied on general geometries in the geometric deep learning literature. We employ message-passing layers conditioned on the proper distance, yielding a computationally efficient framework. We validate our model on tasks from computer vision and statistical physics, demonstrating strong performance, improved generalization capacities, and the ability to extract conformal data such as scaling dimensions from the trained network. 

**Abstract (ZH)**: 共形对称性，即保角的坐标变换，在物理、数学、计算机视觉和几何机器学习等领域中扮演着关键角色。我们构建了一个对一般共形变换是协变的神经网络。为此，我们将数据从平坦欧几里得空间提升到反德西特（AdS）空间，这使得我们可以利用平坦空间共形变换与AdS空间等距变换之间已知的对应关系。然后，我们基于几何深度学习文献中对一般几何上等距变换的广泛研究，构建了这种等距变换。我们采用条件于正确距离的消息传递层，从而形成一个计算高效的框架。我们在计算机视觉和统计物理任务上验证了该模型，展示了其强大的性能、增强的泛化能力和从训练网络中提取共形数据（如缩放维数）的能力。 

---
# Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks? 

**Title (ZH)**: 低秩适应会导致训练时攻击下的较低稳健性吗？ 

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Ronghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12871)  

**Abstract**: Low rank adaptation (LoRA) has emerged as a prominent technique for fine-tuning large language models (LLMs) thanks to its superb efficiency gains over previous methods. While extensive studies have examined the performance and structural properties of LoRA, its behavior upon training-time attacks remain underexplored, posing significant security risks. In this paper, we theoretically investigate the security implications of LoRA's low-rank structure during fine-tuning, in the context of its robustness against data poisoning and backdoor attacks. We propose an analytical framework that models LoRA's training dynamics, employs the neural tangent kernel to simplify the analysis of the training process, and applies information theory to establish connections between LoRA's low rank structure and its vulnerability against training-time attacks. Our analysis indicates that LoRA exhibits better robustness to backdoor attacks than full fine-tuning, while becomes more vulnerable to untargeted data poisoning due to its over-simplified information geometry. Extensive experimental evaluations have corroborated our theoretical findings. 

**Abstract (ZH)**: LoRA的低秩结构在训练时攻击下的安全性分析：基于数据投毒和后门攻击的鲁棒性研究 

---
# Outsourced Privacy-Preserving Feature Selection Based on Fully Homomorphic Encryption 

**Title (ZH)**: 基于全同态加密的隐私保护特征选择外包 

**Authors**: Koki Wakiyama, Tomohiro I, Hiroshi Sakamoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.12869)  

**Abstract**: Feature selection is a technique that extracts a meaningful subset from a set of features in training data. When the training data is large-scale, appropriate feature selection enables the removal of redundant features, which can improve generalization performance, accelerate the training process, and enhance the interpretability of the model. This study proposes a privacy-preserving computation model for feature selection. Generally, when the data owner and analyst are the same, there is no need to conceal the private information. However, when they are different parties or when multiple owners exist, an appropriate privacy-preserving framework is required. Although various private feature selection algorithms, they all require two or more computing parties and do not guarantee security in environments where no external party can be fully trusted. To address this issue, we propose the first outsourcing algorithm for feature selection using fully homomorphic encryption. Compared to a prior two-party algorithm, our result improves the time and space complexity O(kn^2) to O(kn log^3 n) and O(kn), where k and n denote the number of features and data samples, respectively. We also implemented the proposed algorithm and conducted comparative experiments with the naive one. The experimental result shows the efficiency of our method even with small datasets. 

**Abstract (ZH)**: 一种基于完全同态加密的隐私保留特征选择外包算法 

---
# LEXam: Benchmarking Legal Reasoning on 340 Law Exams 

**Title (ZH)**: LEXam：法律推理能力评估标准（基于340套法律考试题） 

**Authors**: Yu Fan, Jingwei Ni, Jakob Merane, Etienne Salimbeni, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, Joel Niklaus  

**Link**: [PDF](https://arxiv.org/pdf/2505.12864)  

**Abstract**: Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. We introduce LEXam, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 4,886 law exam questions in English and German, including 2,841 long-form, open-ended questions and 2,045 multiple-choice questions. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. Adopting an LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. Project page: this https URL 

**Abstract (ZH)**: 长格式法律推理依然是大型语言模型（LLMs）的一个关键挑战，尽管近期在测试时缩放方面取得了进展。我们引入了LEXam，这是一个新颖的基准，源自涵盖116门法律课程、涉及多种科目和学位级别、共340场法律考试的数据集。该数据集包含4,886道用英语和德语编写的法律考试题目，其中包括2,841道长格式、开放性问题和2,045道多项选择题。除了参考答案外，开放性问题还附有明确的指导，说明预期的法律推理方法，如问题识别、规则回忆或规则应用。我们在开放性问题和多项选择题上的评估对当前的LLMs提出了显著挑战；特别是在应对需要结构化、多步骤法律推理的开放性问题时，它们表现尤为困难。此外，我们的结果突显了该数据集在区分具有不同能力的模型方面的有效性。采用LLM-as-a-Judge范式并结合严格的专家人工验证，我们展示了如何一致且准确地评估模型生成的推理步骤。我们的评估设置提供了一种超越简单准确率指标的评估法律推理质量的可扩展方法。项目页面：this https URL 

---
# Unified Cross-modal Translation of Score Images, Symbolic Music, and Performance Audio 

**Title (ZH)**: 统一跨模态翻译：乐谱图像、符号音乐和表演音频 

**Authors**: Jongmin Jung, Dongmin Kim, Sihun Lee, Seola Cho, Hyungjoon Soh, Irmak Bukey, Chris Donahue, Dasaem Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12863)  

**Abstract**: Music exists in various modalities, such as score images, symbolic scores, MIDI, and audio. Translations between each modality are established as core tasks of music information retrieval, such as automatic music transcription (audio-to-MIDI) and optical music recognition (score image to symbolic score). However, most past work on multimodal translation trains specialized models on individual translation tasks. In this paper, we propose a unified approach, where we train a general-purpose model on many translation tasks simultaneously. Two key factors make this unified approach viable: a new large-scale dataset and the tokenization of each modality. Firstly, we propose a new dataset that consists of more than 1,300 hours of paired audio-score image data collected from YouTube videos, which is an order of magnitude larger than any existing music modal translation datasets. Secondly, our unified tokenization framework discretizes score images, audio, MIDI, and MusicXML into a sequence of tokens, enabling a single encoder-decoder Transformer to tackle multiple cross-modal translation as one coherent sequence-to-sequence task. Experimental results confirm that our unified multitask model improves upon single-task baselines in several key areas, notably reducing the symbol error rate for optical music recognition from 24.58% to a state-of-the-art 13.67%, while similarly substantial improvements are observed across the other translation tasks. Notably, our approach achieves the first successful score-image-conditioned audio generation, marking a significant breakthrough in cross-modal music generation. 

**Abstract (ZH)**: 多模态音乐信息 Retrieval 中的统一方法：从统一数据集和令牌化到跨模态翻译 

---
# FLTG: Byzantine-Robust Federated Learning via Angle-Based Defense and Non-IID-Aware Weighting 

**Title (ZH)**: FLTG：基于角度防御和非IID意识加权的拜占庭鲁棒联邦学习 

**Authors**: Yanhua Wen, Lu Ai, Gang Liu, Chuang Li, Jianhao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.12851)  

**Abstract**: Byzantine attacks during model aggregation in Federated Learning (FL) threaten training integrity by manipulating malicious clients' updates. Existing methods struggle with limited robustness under high malicious client ratios and sensitivity to non-i.i.d. data, leading to degraded accuracy. To address this, we propose FLTG, a novel aggregation algorithm integrating angle-based defense and dynamic reference selection. FLTG first filters clients via ReLU-clipped cosine similarity, leveraging a server-side clean dataset to exclude misaligned updates. It then dynamically selects a reference client based on the prior global model to mitigate non-i.i.d. bias, assigns aggregation weights inversely proportional to angular deviations, and normalizes update magnitudes to suppress malicious scaling. Evaluations across datasets of varying complexity under five classic attacks demonstrate FLTG's superiority over state-of-the-art methods under extreme bias scenarios and sustains robustness with a higher proportion(over 50%) of malicious clients. 

**Abstract (ZH)**: Byzantine 攻击在联邦学习（FL）模型聚合期间威胁训练完整性通过操纵恶意客户端的更新。现有方法在高恶意客户端比例和非i.i.d.数据下表现出有限的鲁棒性，导致准确性下降。为此，我们提出了一种名为 FLTG 的新型聚合算法，该算法结合了基于角度的防御和动态参考选择。FLTG 首先通过 ReLU 截断余弦相似度筛选客户端，利用服务器端的干净数据集排除对齐错误的更新。随后，它基于先前的全局模型动态选择一个参考客户端以减轻非i.i.d.偏差，按照角度偏差的倒数分配聚合权重，并对更新幅度进行归一化以抑制恶意缩放。在五种经典攻击下的不同复杂度数据集上的评估表明，FLTG 在极端偏差场景下优于最先进的方法，并且在超过 50% 的恶意客户端比例下仍能保持鲁棒性。 

---
# Bias Fitting to Mitigate Length Bias of Reward Model in RLHF 

**Title (ZH)**: 偏置拟合以减轻奖励模型在RLHF中的长度偏置 

**Authors**: Kangwen Zhao, Jianfeng Cai, Jinhua Zhu, Ruopei Sun, Dongyun Xue, Wengang Zhou, Li Li, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12843)  

**Abstract**: Reinforcement Learning from Human Feedback relies on reward models to align large language models with human preferences. However, RLHF often suffers from reward hacking, wherein policy learning exploits flaws in the trained reward model to maximize reward scores without genuinely aligning with human preferences. A significant example of such reward hacking is length bias, where reward models usually favor longer responses irrespective of actual response quality. Previous works on length bias have notable limitations, these approaches either mitigate bias without characterizing the bias form, or simply assume a linear length-reward relation. To accurately model the intricate nature of length bias and facilitate more effective bias mitigation, we propose FiMi-RM (Bias Fitting to Mitigate Length Bias of Reward Model in RLHF), a framework that autonomously learns and corrects underlying bias patterns. Our approach consists of three stages: First, we train a standard reward model which inherently contains length bias. Next, we deploy a lightweight fitting model to explicitly capture the non-linear relation between length and reward. Finally, we incorporate this learned relation into the reward model to debias. Experimental results demonstrate that FiMi-RM achieves a more balanced length-reward distribution. Furthermore, when applied to alignment algorithms, our debiased reward model improves length-controlled win rate and reduces verbosity without compromising its performance. 

**Abstract (ZH)**: 基于人类反馈的强化学习依赖于奖励模型将大型语言模型与人类偏好对齐。然而，RLHF经常遭受奖励作弊的问题，其中策略学习利用训练奖励模型中的缺陷以最大化奖励分数，而未能真正与人类偏好对齐。长度偏见是这种奖励作弊的一个显著例子，奖励模型通常倾向于更长的响应，而不论实际响应质量如何。对长度偏见的早期研究存在显著局限性，这些方法要么在未表征偏倚形式的情况下减轻偏倚，要么简单地假设长度与奖励之间的线性关系。为了准确建模长度偏见的复杂性质并促进更有效的偏倚缓解，我们提出了一种框架FiMi-RM（用于缓解强化学习从人类反馈中奖励模型长度偏见的偏差拟合）。该框架自主学习并纠正潜在的偏倚模式。我们的方法包括三个阶段：首先，训练一个标准奖励模型，该模型本身就包含长度偏见。其次，部署一个轻量级的拟合模型以明确捕捉长度与奖励之间的非线性关系。最后，将学到的关系纳入奖励模型以减轻偏差。实验结果表明，FiMi-RM实现了更平衡的长度-奖励分布。此外，在应用于对齐算法时，我们去偏后的奖励模型在控制长度方面提高了胜率，减少verbosity的同时不损害其性能。 

---
# The Hidden Structure -- Improving Legal Document Understanding Through Explicit Text Formatting 

**Title (ZH)**: 隐藏的结构——通过显式文本格式化提高法律文件理解 

**Authors**: Christian Braun, Alexander Lilienbeck, Daniel Mentjukov  

**Link**: [PDF](https://arxiv.org/pdf/2505.12837)  

**Abstract**: Legal contracts possess an inherent, semantically vital structure (e.g., sections, clauses) that is crucial for human comprehension but whose impact on LLM processing remains under-explored. This paper investigates the effects of explicit input text structure and prompt engineering on the performance of GPT-4o and GPT-4.1 on a legal question-answering task using an excerpt of the CUAD. We compare model exact-match accuracy across various input formats: well-structured plain-text (human-generated from CUAD), plain-text cleaned of line breaks, extracted plain-text from Azure OCR, plain-text extracted by GPT-4o Vision, and extracted (and interpreted) Markdown (MD) from GPT-4o Vision. To give an indication of the impact of possible prompt engineering, we assess the impact of shifting task instructions to the system prompt and explicitly informing the model about the structured nature of the input. Our findings reveal that GPT-4o demonstrates considerable robustness to variations in input structure, but lacks in overall performance. Conversely, GPT-4.1's performance is markedly sensitive; poorly structured inputs yield suboptimal results (but identical with GPT-4o), while well-structured formats (original CUAD text, GPT-4o Vision text and GPT-4o MD) improve exact-match accuracy by ~20 percentage points. Optimizing the system prompt to include task details and an advisory about structured input further elevates GPT-4.1's accuracy by an additional ~10-13 percentage points, with Markdown ultimately achieving the highest performance under these conditions (79 percentage points overall exact-match accuracy). This research empirically demonstrates that while newer models exhibit greater resilience, careful input structuring and strategic prompt design remain critical for optimizing the performance of LLMs, and can significantly affect outcomes in high-stakes legal applications. 

**Abstract (ZH)**: 法律合同具有一种内在的、语义上至关重要的结构（例如，章节、条款），这对人类的理解至关重要，但其对LLM处理的影响仍鲜有探讨。本文研究了明确定义的输入文本结构和提示工程对GPT-4o和GPT-4在法律问答任务中的性能影响，使用CUAD的部分内容作为实验材料。我们比较了不同输入格式下的模型精确匹配准确率：整洁的纯文本（由人类从CUAD生成）、去除换行符的纯文本、来自Azure OCR的纯文本提取、由GPT-4o Vision提取的纯文本、以及由GPT-4o Vision提取并解析的Markdown（MD）。为评估潜在的提示工程影响，我们分析了将任务指令移至系统提示以及明确告知模型输入结构化性质的影响。研究发现，GPT-4o对输入结构的变化表现出明显的稳健性，但在整体性能上表现不佳。相比之下，GPT-4.1的性能对输入结构变化极为敏感，未结构化的输入会导致次优结果（但与GPT-4o相同），而结构化的格式（原始CUAD文本、GPT-4o Vision文本和GPT-4o MD）可以提高精确匹配准确率约20个百分点。将系统提示优化为包含任务细节和对结构化输入的建议，进一步提升了GPT-4.1的准确率约10-13个百分点，Markdown最终在这些条件下达到最高的性能（整体精确匹配准确率为79个百分点）。这项研究实证表明，虽然最新模型更具韧性，但精心设计的输入结构和战略提示设计仍然是优化LLM性能的关键，并且在高风险的法律应用中可以显著影响结果。 

---
# SynDec: A Synthesize-then-Decode Approach for Arbitrary Textual Style Transfer via Large Language Models 

**Title (ZH)**: SynDec：一种通过大型语言模型进行任意文本风格转换的合成-解码方法 

**Authors**: Han Sun, Zhen Sun, Zongmin Zhang, Linzhao Jia, Wei Shao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12821)  

**Abstract**: Large Language Models (LLMs) are emerging as dominant forces for textual style transfer. However, for arbitrary style transfer, LLMs face two key challenges: (1) considerable reliance on manually-constructed prompts and (2) rigid stylistic biases inherent in LLMs. In this paper, we propose a novel Synthesize-then-Decode (SynDec) approach, which automatically synthesizes high-quality prompts and amplifies their roles during decoding process. Specifically, our approach synthesizes prompts by selecting representative few-shot samples, conducting a four-dimensional style analysis, and reranking the candidates. At LLM decoding stage, the TST effect is amplified by maximizing the contrast in output probabilities between scenarios with and without the synthesized prompt, as well as between prompts and negative samples. We conduct extensive experiments and the results show that SynDec outperforms existing state-of-the-art LLM-based methods on five out of six benchmarks (e.g., achieving up to a 9\% increase in accuracy for modern-to-Elizabethan English transfer). Detailed ablation studies further validate the effectiveness of SynDec. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在成为文本风格转换的主要力量。然而，对于任意风格转换，LLMs面临两个主要挑战：（1）对手动构建的提示依赖较大；（2）固有的风格偏见。在本文中，我们提出了一种新颖的Synthesize-then-Decode（SynDec）方法，该方法自动生成高质量的提示并在解码过程中放大其作用。具体而言，该方法通过选择代表性的少量样本、进行四维风格分析并对候选样本重新排队来生成提示。在LLM解码阶段，通过最大化含有和不含合成提示以及提示和负样本之间输出概率的对比度来放大TST效果。我们在广泛的实验中进行了测试，结果表明SynDec在六个基准中的五个上优于现有最先进的基于LLM的方法（例如，现代英语到伊丽莎白an英语转换的准确性提高多达9%）。详细的消融研究进一步验证了SynDec的有效性。 

---
# Learning in Chaos: Efficient Autoscaling and Self-healing for Distributed Training at the Edge 

**Title (ZH)**: 在混沌中学习：边缘分布式训练的高效自动扩展与自我修复 

**Authors**: Wenjiao Feng, Rongxing Xiao, Zonghang Li, Hongfang Yu, Gang Sun, Long Luo, Mohsen Guizani, Qirong Ho  

**Link**: [PDF](https://arxiv.org/pdf/2505.12815)  

**Abstract**: Frequent node and link changes in edge AI clusters disrupt distributed training, while traditional checkpoint-based recovery and cloud-centric autoscaling are too slow for scale-out and ill-suited to chaotic and self-governed edge. This paper proposes Chaos, a resilient and scalable edge distributed training system with built-in self-healing and autoscaling. It speeds up scale-out by using multi-neighbor replication with fast shard scheduling, allowing a new node to pull the latest training state from nearby neighbors in parallel while balancing the traffic load between them. It also uses a cluster monitor to track resource and topology changes to assist scheduler decisions, and handles scaling events through peer negotiation protocols, enabling fully self-governed autoscaling without a central admin. Extensive experiments show that Chaos consistently achieves much lower scale-out delays than Pollux, EDL, and Autoscaling, and handles scale-in, connect-link, and disconnect-link events within 1 millisecond, making it smoother to handle node joins, exits, and failures. It also delivers the lowest idle time, showing superior resource use and scalability as the cluster grows. 

**Abstract (ZH)**: 边缘AI集群中频繁的节点和链路变化干扰分布式训练，而传统的基于检查点的恢复和以云为中心的自动扩展对于规模扩展来说太慢，并且不适合混乱且自治的边缘环境。本文提出Chaos，一种内置自我修复和自动扩展功能的弹性可扩展边缘分布式训练系统。通过使用多邻节点复制和快速分片调度，Chaos加速了扩展过程，允许新节点并行从附近邻居拉取最新的训练状态，同时平衡它们之间的流量负载。Chaos还使用集群监控器跟踪资源和拓扑结构的变化，以辅助调度决策，并通过对等协商协议处理扩展事件，实现无需中央管理员的完全自主扩展。大量实验表明，Chaos在规模扩展延迟方面始终优于Pollux、EDL和自动扩展，能够在1毫秒内处理连接链路、断开链路和缩减规模事件，使节点加入、退出和故障处理更加平滑。它还实现了最低的闲置时间，展示了随着集群规模的扩大，其在资源使用和扩展性方面的优越性。 

---
# PsyMem: Fine-grained psychological alignment and Explicit Memory Control for Advanced Role-Playing LLMs 

**Title (ZH)**: PsyMem: 细粒度心理对齐与显性记忆控制以提升高级角色扮演语言模型 

**Authors**: Xilong Cheng, Yunxiao Qin, Yuting Tan, Zhengnan Li, Ye Wang, Hongjiang Xiao, Yuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12814)  

**Abstract**: Existing LLM-based role-playing methods often rely on superficial textual descriptions or simplistic metrics, inadequately modeling both intrinsic and extrinsic character dimensions. Additionally, they typically simulate character memory with implicit model knowledge or basic retrieval augment generation without explicit memory alignment, compromising memory consistency. The two issues weaken reliability of role-playing LLMs in several applications, such as trustworthy social simulation. To address these limitations, we propose PsyMem, a novel framework integrating fine-grained psychological attributes and explicit memory control for role-playing. PsyMem supplements textual descriptions with 26 psychological indicators to detailed model character. Additionally, PsyMem implements memory alignment training, explicitly trains the model to align character's response with memory, thereby enabling dynamic memory-controlled responding during inference. By training Qwen2.5-7B-Instruct on our specially designed dataset (including 5,414 characters and 38,962 dialogues extracted from novels), the resulting model, termed as PsyMem-Qwen, outperforms baseline models in role-playing, achieving the best performance in human-likeness and character fidelity. 

**Abstract (ZH)**: 基于现有LLM的角色扮演方法往往依赖于表面的文字描述或简单的度量标准，无法充分建模内在和外在的人物维度。此外，它们通常通过隐式的模型知识或基本的检索增强生成来模拟人物记忆，而没有明确的记忆对齐，从而影响记忆一致性。这两个问题削弱了角色扮演LLM在诸如可信社会模拟等应用中的可靠性和稳定性。为此，我们提出了PsyMem框架，该框架结合了精细的心理属性和明确的记忆控制。PsyMem通过添加26个心理指标补充文本描述，以详细构建模型人物。此外，PsyMem实现了记忆对齐训练，明确训练模型使其响应与记忆对齐，在推断过程中实现动态的记忆控制响应。通过在我们特别设计的数据集（包含5,414个角色和38,962个对话，从小说中提取）上对Qwen2.5-7B-Instruct进行训练，生成的PsyMem-Qwen模型在角色扮演任务中表现优于基线模型，在拟人度和人物忠实度方面达到最优。 

---
# Dynamic Sight Range Selection in Multi-Agent Reinforcement Learning 

**Title (ZH)**: 多agent强化学习中的动态视距选择 

**Authors**: Wei-Chen Liao, Ti-Rong Wu, I-Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12811)  

**Abstract**: Multi-agent reinforcement Learning (MARL) is often challenged by the sight range dilemma, where agents either receive insufficient or excessive information from their environment. In this paper, we propose a novel method, called Dynamic Sight Range Selection (DSR), to address this issue. DSR utilizes an Upper Confidence Bound (UCB) algorithm and dynamically adjusts the sight range during training. Experiment results show several advantages of using DSR. First, we demonstrate using DSR achieves better performance in three common MARL environments, including Level-Based Foraging (LBF), Multi-Robot Warehouse (RWARE), and StarCraft Multi-Agent Challenge (SMAC). Second, our results show that DSR consistently improves performance across multiple MARL algorithms, including QMIX and MAPPO. Third, DSR offers suitable sight ranges for different training steps, thereby accelerating the training process. Finally, DSR provides additional interpretability by indicating the optimal sight range used during training. Unlike existing methods that rely on global information or communication mechanisms, our approach operates solely based on the individual sight ranges of agents. This approach offers a practical and efficient solution to the sight range dilemma, making it broadly applicable to real-world complex environments. 

**Abstract (ZH)**: 多代理强化学习中动态视距选择（DSR）方法 

---
# FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA 

**Title (ZH)**: FedSVD: 自适应正交化以实现具有LoRA的私人联邦学习 

**Authors**: Seanie Lee, Sangwoo Park, Dong Bok Lee, Dominik Wagner, Haebin Seong, Tobias Bocklet, Juho Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12805)  

**Abstract**: Low-Rank Adaptation (LoRA), which introduces a product of two trainable low-rank matrices into frozen pre-trained weights, is widely used for efficient fine-tuning of language models in federated learning (FL). However, when combined with differentially private stochastic gradient descent (DP-SGD), LoRA faces substantial noise amplification: DP-SGD perturbs per-sample gradients, and the matrix multiplication of the LoRA update ($BA$) intensifies this effect. Freezing one matrix (e.g., $A$) reduces the noise but restricts model expressiveness, often resulting in suboptimal adaptation. To address this, we propose FedSVD, a simple yet effective method that introduces a global reparameterization based on singular value decomposition (SVD). In our approach, each client optimizes only the $B$ matrix and transmits it to the server. The server aggregates the $B$ matrices, computes the product $BA$ using the previous $A$, and refactorizes the result via SVD. This yields a new adaptive $A$ composed of the orthonormal right singular vectors of $BA$, and an updated $B$ containing the remaining SVD components. This reparameterization avoids quadratic noise amplification, while allowing $A$ to better capture the principal directions of the aggregate updates. Moreover, the orthonormal structure of $A$ bounds the gradient norms of $B$ and preserves more signal under DP-SGD, as confirmed by our theoretical analysis. As a result, FedSVD consistently improves stability and performance across a variety of privacy settings and benchmarks, outperforming relevant baselines under both private and non-private regimes. 

**Abstract (ZH)**: 基于奇异值分解的全局重参量化（FedSVD）：一种在差分隐私随机梯度下降下的低秩适应优化方法 

---
# OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching 

**Title (ZH)**: OZSpeech: 一步式零样本语音合成与学习先验条件流匹配 

**Authors**: Hieu-Nghia Huynh-Nguyen, Ngoc Son Nguyen, Huynh Nguyen Dang, Thieu Vo, Truong-Son Hy, Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12800)  

**Abstract**: Text-to-speech (TTS) systems have seen significant advancements in recent years, driven by improvements in deep learning and neural network architectures. Viewing the output speech as a data distribution, previous approaches often employ traditional speech representations, such as waveforms or spectrograms, within the Flow Matching framework. However, these methods have limitations, including overlooking various speech attributes and incurring high computational costs due to additional constraints introduced during training. To address these challenges, we introduce OZSpeech, the first TTS method to explore optimal transport conditional flow matching with one-step sampling and a learned prior as the condition, effectively disregarding preceding states and reducing the number of sampling steps. Our approach operates on disentangled, factorized components of speech in token format, enabling accurate modeling of each speech attribute, which enhances the TTS system's ability to precisely clone the prompt speech. Experimental results show that our method achieves promising performance over existing methods in content accuracy, naturalness, prosody generation, and speaker style preservation. Audio samples are available at our demo page this https URL. 

**Abstract (ZH)**: 文本到语音（TTS）系统在近年来取得了显著进步，得益于深度学习和神经网络架构的改进。通过将输出语音视为数据分布，先前方法通常在Flow Matching框架中使用传统的语音表示，如波形或频谱图。然而，这些方法存在局限性，包括忽视了各种语音属性，并且由于训练过程中引入了额外的约束条件而导致了高昂的计算成本。为了应对这些挑战，我们提出了OZSpeech，这是首个探索最优传输条件流动匹配的方法，采用一步采样和学习先验作为条件，有效地忽略了先前状态并减少了采样步骤的数量。我们的方法以令牌格式处理语音的分离因素组件，这使得能够准确建模每个语音属性，从而增强TTS系统精确克隆提示语音的能力。实验结果表明，我们的方法在内容准确性、自然度、语调生成和说话人风格保留方面优于现有方法。音频样本可在我们的演示页面此链接获取。 

---
# A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone 

**Title (ZH)**: 一个令牌相当于超过1,000个令牌：通过低秩克隆实现高效的知识蒸馏 

**Authors**: Jitai Hao, Qiang Huang, Hao Liu, Xinyan Xiao, Zhaochun Ren, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12781)  

**Abstract**: Training high-performing Small Language Models (SLMs) remains costly, even with knowledge distillation and pruning from larger teacher models. Existing work often faces three key challenges: (1) information loss from hard pruning, (2) inefficient alignment of representations, and (3) underutilization of informative activations, particularly from Feed-Forward Networks (FFNs). To address these challenges, we introduce Low-Rank Clone (LRC), an efficient pre-training method that constructs SLMs aspiring to behavioral equivalence with strong teacher models. LRC trains a set of low-rank projection matrices that jointly enable soft pruning by compressing teacher weights, and activation clone by aligning student activations, including FFN signals, with those of the teacher. This unified design maximizes knowledge transfer while removing the need for explicit alignment modules. Extensive experiments with open-source teachers (e.g., Llama-3.2-3B-Instruct, Qwen2.5-3B/7B-Instruct) show that LRC matches or surpasses state-of-the-art models trained on trillions of tokens--while using only 20B tokens, achieving over 1,000x training efficiency. Our codes and model checkpoints are available at this https URL and this https URL. 

**Abstract (ZH)**: 训练高性能小型语言模型（SLMs）仍然代价高昂，即使使用来自较大教师模型的知识蒸馏和剪枝也是如此。现有工作通常面临三个关键挑战：（1）硬剪枝导致的信息丢失，（2）表示的低效对齐，以及（3）信息激活，特别是来自前馈网络（FFNs）的信息激活的低利用率。为了解决这些挑战，我们提出了低秩克隆（LRC），这是一种高效的预训练方法，旨在使小型语言模型的行为与强大的教师模型相等价。LRC 训练一组低秩投影矩阵，这些矩阵共同实现软剪枝，通过压缩教师权重和通过对齐学生激活（包括FFN信号）与教师的激活来实现激活克隆。这种统一设计最大化了知识转移，同时去除了需要显式对齐模块的需要。使用开源教师模型（例如，Llama-3.2-3B-Instruct、Qwen2.5-3B/7B-Instruct）的广泛实验表明，LRC 在使用仅 200 亿个标记的情况下，可与训练数万亿个标记的最新模型相媲美甚至超越，实现超过 1,000 倍的训练效率。我们的代码和模型检查点可在以下网址获取：这个 https URL 和这个 https URL。 

---
# UniHM: Universal Human Motion Generation with Object Interactions in Indoor Scenes 

**Title (ZH)**: UniHM：室内场景中物体交互的通用人体运动生成 

**Authors**: Zichen Geng, Zeeshan Hayder, Wei Liu, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2505.12774)  

**Abstract**: Human motion synthesis in complex scenes presents a fundamental challenge, extending beyond conventional Text-to-Motion tasks by requiring the integration of diverse modalities such as static environments, movable objects, natural language prompts, and spatial waypoints. Existing language-conditioned motion models often struggle with scene-aware motion generation due to limitations in motion tokenization, which leads to information loss and fails to capture the continuous, context-dependent nature of 3D human movement. To address these issues, we propose UniHM, a unified motion language model that leverages diffusion-based generation for synthesizing scene-aware human motion. UniHM is the first framework to support both Text-to-Motion and Text-to-Human-Object Interaction (HOI) in complex 3D scenes. Our approach introduces three key contributions: (1) a mixed-motion representation that fuses continuous 6DoF motion with discrete local motion tokens to improve motion realism; (2) a novel Look-Up-Free Quantization VAE (LFQ-VAE) that surpasses traditional VQ-VAEs in both reconstruction accuracy and generative performance; and (3) an enriched version of the Lingo dataset augmented with HumanML3D annotations, providing stronger supervision for scene-specific motion learning. Experimental results demonstrate that UniHM achieves comparative performance on the OMOMO benchmark for text-to-HOI synthesis and yields competitive results on HumanML3D for general text-conditioned motion generation. 

**Abstract (ZH)**: 复杂场景中的人体运动合成是一项基本挑战，超越了传统的文本到运动任务，需要整合静态环境、可移动物体、自然语言提示和空间 waypoints 等多种模态。现有的基于语言条件的运动模型由于运动标记化能力的限制，在场景感知运动生成方面往往存在局限性，导致信息丢失，并且无法捕捉3D人体运动的连续性和依赖于上下文的特性。为了解决这些问题，我们提出了一种统一的运动语言模型UniHM，利用扩散生成法来合成场景感知的人体运动。UniHM是第一个同时支持文本到运动和文本到人体-物体交互（HOI）的复杂3D场景框架。我们的方法包含三项关键贡献：（1）混合运动表示，将连续的6DoF运动与离散的局部运动标记融合，以提高运动的真实感；（2）一种无查找表的新型量化VAE（LFQ-VAE），在重构准确性和生成性能上均超越了传统的VQ-VAEs；（3）增强版本的Lingo数据集，增加了HumanML3D注释，为场景特定的运动学习提供了更强的监督。实验结果表明，UniHM在OMOMO基准测试上达到与文本到HOI合成相当的性能，并在HumanML3D上生成了具有竞争力的3D通用文本条件运动。 

---
# Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization 

**Title (ZH)**: 重新审视奖励模型评估：通过奖励过优化的视角 

**Authors**: Sunghwan Kim, Dongjin Kang, Taeyoon Kwon, Hyungjoo Chae, Dongha Lee, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12763)  

**Abstract**: Reward models (RMs) play a crucial role in reinforcement learning from human feedback (RLHF), aligning model behavior with human preferences. However, existing benchmarks for reward models show a weak correlation with the performance of optimized policies, suggesting that they fail to accurately assess the true capabilities of RMs. To bridge this gap, we explore several evaluation designs through the lens of reward overoptimization\textemdash a phenomenon that captures both how well the reward model aligns with human preferences and the dynamics of the learning signal it provides to the policy. The results highlight three key findings on how to construct a reliable benchmark: (i) it is important to minimize differences between chosen and rejected responses beyond correctness, (ii) evaluating reward models requires multiple comparisons across a wide range of chosen and rejected responses, and (iii) given that reward models encounter responses with diverse representations, responses should be sourced from a variety of models. However, we also observe that a extremely high correlation with degree of overoptimization leads to comparatively lower correlation with certain downstream performance. Thus, when designing a benchmark, it is desirable to use the degree of overoptimization as a useful tool, rather than the end goal. 

**Abstract (ZH)**: reward模型（RMs）在人类反馈强化学习（RLHF）中起着关键作用，它们使模型行为与人类偏好保持一致。然而，现有的reward模型基准与优化策略的表现之间显示出了较弱的相关性，这表明这些基准未能准确评估RM的真实能力。为了弥合这一差距，我们通过奖励过度优化的视角探索了几种评估设计，这一现象既反映了reward模型与人类偏好之间的契合度，又体现了其对策略学习信号的影响动态。结果强调了构建可靠基准的三个关键发现：（i）在正确性之外，尽量减少所选和拒绝响应之间的差异至关重要；（ii）评估reward模型需要跨广泛的所选和拒绝响应进行多方位比较；（iii）鉴于reward模型会遇到多种多样表示的响应，响应应来自多种模型。然而，我们还发现，极高的过度优化程度与相关性之间的高度相关性会导致某些下游性能的相关性相对较低。因此，在设计基准时，应该将过度优化的程度视为有用的工具，而不是最终目标。 

---
# Enhancing Channel-Independent Time-Series Forecasting via Cross-Variate Patch Embedding 

**Title (ZH)**: 通过交叉变量 patches 插入方法增强通道独立时间序列预测 

**Authors**: Donghwa Shin, Edwin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12761)  

**Abstract**: Transformers have recently gained popularity in time series forecasting due to their ability to capture long-term dependencies. However, many existing models focus only on capturing temporal dependencies while omitting intricate relationships between variables. Recent models have tried tackling this by explicitly modeling both cross-time and cross-variate dependencies through a sequential or unified attention mechanism, but they are entirely channel dependent (CD) across all layers, making them potentially susceptible to overfitting. To address this, we propose Cross-Variate Patch Embeddings (CVPE), a lightweight CD module that injects cross-variate context into channel-independent (CI) models by simply modifying the patch embedding process. We achieve this by adding a learnable positional encoding and a lightweight router-attention block to the vanilla patch embedding layer. We then integrate CVPE into Time-LLM, a multimodal CI forecasting model, to demonstrate its effectiveness in capturing cross-variate dependencies and enhance the CI model's performance. Extensive experimental results on seven real-world datasets show that our enhanced Time-LLM outperforms the original baseline model simply by incorporating the CVPE module, with no other changes. 

**Abstract (ZH)**: Cross-Variate Patch Embeddings for Channel-Independent Time Series Forecasting 

---
# Structure-based Anomaly Detection and Clustering 

**Title (ZH)**: 结构导向的异常检测与聚类 

**Authors**: Filippo Leveni  

**Link**: [PDF](https://arxiv.org/pdf/2505.12751)  

**Abstract**: Anomaly detection is a fundamental problem in domains such as healthcare, manufacturing, and cybersecurity. This thesis proposes new unsupervised methods for anomaly detection in both structured and streaming data settings. In the first part, we focus on structure-based anomaly detection, where normal data follows low-dimensional manifolds while anomalies deviate from them. We introduce Preference Isolation Forest (PIF), which embeds data into a high-dimensional preference space via manifold fitting, and isolates outliers using two variants: Voronoi-iForest, based on geometric distances, and RuzHash-iForest, leveraging Locality Sensitive Hashing for scalability. We also propose Sliding-PIF, which captures local manifold information for streaming scenarios. Our methods outperform existing techniques on synthetic and real datasets. We extend this to structure-based clustering with MultiLink, a novel method for recovering multiple geometric model families in noisy data. MultiLink merges clusters via a model-aware linkage strategy, enabling robust multi-class structure recovery. It offers key advantages over existing approaches, such as speed, reduced sensitivity to thresholds, and improved robustness to poor initial sampling. The second part of the thesis addresses online anomaly detection in evolving data streams. We propose Online Isolation Forest (Online-iForest), which uses adaptive, multi-resolution histograms and dynamically updates tree structures to track changes over time. It avoids retraining while achieving accuracy comparable to offline models, with superior efficiency for real-time applications. Finally, we tackle anomaly detection in cybersecurity via open-set recognition for malware classification. We enhance a Gradient Boosting classifier with MaxLogit to detect unseen malware families, a method now integrated into Cleafy's production system. 

**Abstract (ZH)**: 基于结构的异常检测方法研究 

---
# Malware families discovery via Open-Set Recognition on Android manifest permissions 

**Title (ZH)**: 基于开放集识别的AndroidManifest权限下恶意软件家族发现 

**Authors**: Filippo Leveni, Matteo Mistura, Francesco Iubatti, Carmine Giangregorio, Nicolò Pastore, Cesare Alippi, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12750)  

**Abstract**: Malware are malicious programs that are grouped into families based on their penetration technique, source code, and other characteristics. Classifying malware programs into their respective families is essential for building effective defenses against cyber threats. Machine learning models have a huge potential in malware detection on mobile devices, as malware families can be recognized by classifying permission data extracted from Android manifest files. Still, the malware classification task is challenging due to the high-dimensional nature of permission data and the limited availability of training samples. In particular, the steady emergence of new malware families makes it impossible to acquire a comprehensive training set covering all the malware classes. In this work, we present a malware classification system that, on top of classifying known malware, detects new ones. In particular, we combine an open-set recognition technique developed within the computer vision community, namely MaxLogit, with a tree-based Gradient Boosting classifier, which is particularly effective in classifying high-dimensional data. Our solution turns out to be very practical, as it can be seamlessly employed in a standard classification workflow, and efficient, as it adds minimal computational overhead. Experiments on public and proprietary datasets demonstrate the potential of our solution, which has been deployed in a business environment. 

**Abstract (ZH)**: 恶意软件根据其渗透技术、源代码和其他特征被分为家庭。将恶意软件程序分类到各自的家族中对于构建有效的网络安全防御至关重要。机器学习模型在移动设备上的恶意软件检测中具有巨大潜力，因为可以通过分类从Android清单文件中提取的权限数据来识别恶意软件家族。然而，由于权限数据的高维性质和可获得的训练样本有限，恶意软件分类任务仍然具有挑战性。特别是，新恶意软件家族的持续出现使得无法获取涵盖所有恶意软件类别的全面训练集。在这项工作中，我们提出了一种恶意软件分类系统，该系统不仅对已知恶意软件进行分类，还能检测新出现的恶意软件。具体而言，我们将来自计算机视觉领域的开放式识别技术MaxLogit与基于树的梯度提升分类器相结合，后者特别适用于分类高维数据。我们的解决方案非常实用，可以无缝地适应标准分类流程，并且高效，因为它增加了最少的计算开销。在公共和专有数据集上的实验表明了我们解决方案的潜力，该解决方案已在商业环境中部署。 

---
# TeleOpBench: A Simulator-Centric Benchmark for Dual-Arm Dexterous Teleoperation 

**Title (ZH)**: TeleOpBench: 以模拟器为中心的双臂灵巧远程操控基准评测 

**Authors**: Hangyu Li, Qin Zhao, Haoran Xu, Xinyu Jiang, Qingwei Ben, Feiyu Jia, Haoyu Zhao, Liang Xu, Jia Zeng, Hanqing Wang, Bo Dai, Junting Dong, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12748)  

**Abstract**: Teleoperation is a cornerstone of embodied-robot learning, and bimanual dexterous teleoperation in particular provides rich demonstrations that are difficult to obtain with fully autonomous systems. While recent studies have proposed diverse hardware pipelines-ranging from inertial motion-capture gloves to exoskeletons and vision-based interfaces-there is still no unified benchmark that enables fair, reproducible comparison of these systems. In this paper, we introduce TeleOpBench, a simulator-centric benchmark tailored to bimanual dexterous teleoperation. TeleOpBench contains 30 high-fidelity task environments that span pick-and-place, tool use, and collaborative manipulation, covering a broad spectrum of kinematic and force-interaction difficulty. Within this benchmark we implement four representative teleoperation modalities-(i) MoCap, (ii) VR device, (iii) arm-hand exoskeletons, and (iv) monocular vision tracking-and evaluate them with a common protocol and metric suite. To validate that performance in simulation is predictive of real-world behavior, we conduct mirrored experiments on a physical dual-arm platform equipped with two 6-DoF dexterous hands. Across 10 held-out tasks we observe a strong correlation between simulator and hardware performance, confirming the external validity of TeleOpBench. TeleOpBench establishes a common yardstick for teleoperation research and provides an extensible platform for future algorithmic and hardware innovation. 

**Abstract (ZH)**: TeleOpBench：面向双臂灵巧远程操控的仿真中心基准 

---
# PEER pressure: Model-to-Model Regularization for Single Source Domain Generalization 

**Title (ZH)**: 同伴压力：模型到模型的正则化方法用于单源域泛化 

**Authors**: Dong Kyu Cho, Inwoo Hwang, Sanghack Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12745)  

**Abstract**: Data augmentation is a popular tool for single source domain generalization, which expands the source domain by generating simulated ones, improving generalization on unseen target domains. In this work, we show that the performance of such augmentation-based methods in the target domains universally fluctuates during training, posing challenges in model selection under realistic scenarios. We argue that the fluctuation stems from the inability of the model to accumulate the knowledge learned from diverse augmentations, exacerbating feature distortion during training. Based on this observation, we propose a novel generalization method, coined Parameter-Space Ensemble with Entropy Regularization (PEER), that uses a proxy model to learn the augmented data on behalf of the main model. The main model is updated by averaging its parameters with the proxy model, progressively accumulating knowledge over the training steps. Maximizing the mutual information between the output representations of the two models guides the learning process of the proxy model, mitigating feature distortion during training. Experimental results demonstrate the effectiveness of PEER in reducing the OOD performance fluctuation and enhancing generalization across various datasets, including PACS, Digits, Office-Home, and VLCS. Notably, our method with simple random augmentation achieves state-of-the-art performance, surpassing prior approaches on sDG that utilize complex data augmentation strategies. 

**Abstract (ZH)**: 基于参数空间集成与熵正则化的数据增强方法（PEER）：减少OOD性能波动并提升泛化能力 

---
# EpiLLM: Unlocking the Potential of Large Language Models in Epidemic Forecasting 

**Title (ZH)**: EpiLLM: 在流行病预测中充分发挥大型语言模型的潜力 

**Authors**: Chenghua Gong, Rui Sun, Yuhao Zheng, Juyuan Zhang, Tianjun Gu, Liming Pan, Linyuan Lv  

**Link**: [PDF](https://arxiv.org/pdf/2505.12738)  

**Abstract**: Advanced epidemic forecasting is critical for enabling precision containment strategies, highlighting its strategic importance for public health security. While recent advances in Large Language Models (LLMs) have demonstrated effectiveness as foundation models for domain-specific tasks, their potential for epidemic forecasting remains largely unexplored. In this paper, we introduce EpiLLM, a novel LLM-based framework tailored for spatio-temporal epidemic forecasting. Considering the key factors in real-world epidemic transmission: infection cases and human mobility, we introduce a dual-branch architecture to achieve fine-grained token-level alignment between such complex epidemic patterns and language tokens for LLM adaptation. To unleash the multi-step forecasting and generalization potential of LLM architectures, we propose an autoregressive modeling paradigm that reformulates the epidemic forecasting task into next-token prediction. To further enhance LLM perception of epidemics, we introduce spatio-temporal prompt learning techniques, which strengthen forecasting capabilities from a data-driven perspective. Extensive experiments show that EpiLLM significantly outperforms existing baselines on real-world COVID-19 datasets and exhibits scaling behavior characteristic of LLMs. 

**Abstract (ZH)**: 先进的流行病 forecasting 对于实现精准防控策略至关重要，突显了其在公众健康安全中的战略重要性。尽管近年来大规模语言模型（LLMs）在特定领域任务中的基础模型方面展现了有效性，但其在流行病 forecasting 方面的潜力尚未得到充分探索。本文介绍了一种新型 LLM 基础框架 EpiLLM，专门用于时空流行病 forecasting。考虑到现实世界流行病传播的关键因素——感染案例和人口移动，我们引入了一种双分支架构，以实现复杂流行病模式与语言标记之间的精细粒度对齐，从而适应 LLM。为释放 LLM 架构的多步 forecasting 和泛化潜力，我们提出了一种自回归建模范式，将其重新表述为下一标记预测任务。为了进一步增强 LLM 对流行病的感知，我们引入了时空提示学习技术，这些技术从数据驱动的角度增强了 forecasting 能力。广泛实验表明，EpiLLM 在实际世界 COVID-19 数据集上的表现显著优于现有基线，并展现出与 LLM 相似的扩展行为。 

---
# Option-aware Temporally Abstracted Value for Offline Goal-Conditioned Reinforcement Learning 

**Title (ZH)**: 带有选项意识的时间抽象值函数用于离线目标条件强化学习 

**Authors**: Hongjoon Ahn, Heewoong Choi, Jisu Han, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2505.12737)  

**Abstract**: Offline goal-conditioned reinforcement learning (GCRL) offers a practical learning paradigm where goal-reaching policies are trained from abundant unlabeled (reward-free) datasets without additional environment interaction. However, offline GCRL still struggles with long-horizon tasks, even with recent advances that employ hierarchical policy structures, such as HIQL. By identifying the root cause of this challenge, we observe the following insights: First, performance bottlenecks mainly stem from the high-level policy's inability to generate appropriate subgoals. Second, when learning the high-level policy in the long-horizon regime, the sign of the advantage signal frequently becomes incorrect. Thus, we argue that improving the value function to produce a clear advantage signal for learning the high-level policy is essential. In this paper, we propose a simple yet effective solution: Option-aware Temporally Abstracted value learning, dubbed OTA, which incorporates temporal abstraction into the temporal-difference learning process. By modifying the value update to be option-aware, the proposed learning scheme contracts the effective horizon length, enabling better advantage estimates even in long-horizon regimes. We experimentally show that the high-level policy extracted using the OTA value function achieves strong performance on complex tasks from OGBench, a recently proposed offline GCRL benchmark, including maze navigation and visual robotic manipulation environments. 

**Abstract (ZH)**: 基于 Offline 目标导向的强化学习（GCRL）提供了一种实用的学习范式，其中目标导向策略可以从大量未标注（无奖励）数据集中进行训练，而无需额外的环境交互。然而，即使借助近期采用分层策略结构的进步（如HIQL）， Offline GCRL 仍然难以处理长时 horizon 任务。通过识别这一挑战的根本原因，我们观察到以下见解：首先，性能瓶颈主要源于高层策略无法生成合适的子目标。其次，在长时 horizon 情况下学习高层策略时，优势信号的符号经常变得不正确。因此，我们认为改善价值函数以生成清晰的优势信号对于学习高层策略至关重要。在本文中，我们提出了一种简单而有效的解决方案：基于选项的认知时态抽象价值学习（OTA），该方法将时态抽象融入时差学习过程中。通过使价值更新具有选项意识，提出的学习方案缩短了有效时长，即使在长时 horizon 情况下也能实现更好的优势估计。实验结果显示，使用 OTA 价值函数提取的高层策略在 OGBench（一个新提出的 Offline GCRL 基准）中复杂任务，如迷宫导航和视觉机器人操作环境中表现优异。 

---
# SounDiT: Geo-Contextual Soundscape-to-Landscape Generation 

**Title (ZH)**: SounDiT: 基于地理上下文的声音景观到景观生成 

**Authors**: Junbo Wang, Haofeng Tan, Bowen Liao, Albert Jiang, Teng Fei, Qixing Huang, Zhengzhong Tu, Shan Ye, Yuhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12734)  

**Abstract**: We present a novel and practically significant problem-Geo-Contextual Soundscape-to-Landscape (GeoS2L) generation-which aims to synthesize geographically realistic landscape images from environmental soundscapes. Prior audio-to-image generation methods typically rely on general-purpose datasets and overlook geographic and environmental contexts, resulting in unrealistic images that are misaligned with real-world environmental settings. To address this limitation, we introduce a novel geo-contextual computational framework that explicitly integrates geographic knowledge into multimodal generative modeling. We construct two large-scale geo-contextual multimodal datasets, SoundingSVI and SonicUrban, pairing diverse soundscapes with real-world landscape images. We propose SounDiT, a novel Diffusion Transformer (DiT)-based model that incorporates geo-contextual scene conditioning to synthesize geographically coherent landscape images. Furthermore, we propose a practically-informed geo-contextual evaluation framework, the Place Similarity Score (PSS), across element-, scene-, and human perception-levels to measure consistency between input soundscapes and generated landscape images. Extensive experiments demonstrate that SounDiT outperforms existing baselines in both visual fidelity and geographic settings. Our work not only establishes foundational benchmarks for GeoS2L generation but also highlights the importance of incorporating geographic domain knowledge in advancing multimodal generative models, opening new directions at the intersection of generative AI, geography, urban planning, and environmental sciences. 

**Abstract (ZH)**: Geo-情境化声景到景观生成（GeoS2L）：从环境声景合成地理现实景观图像 

---
# Shadow-FT: Tuning Instruct via Base 

**Title (ZH)**: Shadow-FT: 基于基础模型调优指令方法 

**Authors**: Taiqiang Wu, Runming Yang, Jiayi Li, Pengfei Hu, Ngai Wong, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12716)  

**Abstract**: Large language models (LLMs) consistently benefit from further fine-tuning on various tasks. However, we observe that directly tuning the INSTRUCT (i.e., instruction tuned) models often leads to marginal improvements and even performance degeneration. Notably, paired BASE models, the foundation for these INSTRUCT variants, contain highly similar weight values (i.e., less than 2% on average for Llama 3.1 8B). Therefore, we propose a novel Shadow-FT framework to tune the INSTRUCT models by leveraging the corresponding BASE models. The key insight is to fine-tune the BASE model, and then directly graft the learned weight updates to the INSTRUCT model. Our proposed Shadow-FT introduces no additional parameters, is easy to implement, and significantly improves performance. We conduct extensive experiments on tuning mainstream LLMs, such as Qwen 3 and Llama 3 series, and evaluate them across 19 benchmarks covering coding, reasoning, and mathematical tasks. Experimental results demonstrate that Shadow-FT consistently outperforms conventional full-parameter and parameter-efficient tuning approaches. Further analyses indicate that Shadow-FT can be applied to multimodal large language models (MLLMs) and combined with direct preference optimization (DPO). Codes and weights are available at \href{this https URL}{Github}. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种任务上从进一步微调中受益匪浅。然而，我们观察到直接对INSTRUCT（即指令调优）模型进行微调往往导致边际性改进，甚至性能退化。值得注意的是，这些INSTRUCT变体的基础模型BASE含有高度相似的权重值（以Llama 3.1 8B为例，平均值低于2%）。因此，我们提出了一种新颖的Shadow-FT框架，通过利用对应的BASE模型来微调INSTRUCT模型。核心洞察是先微调BASE模型，然后直接将学到的权重更新嫁接到INSTRUCT模型。我们提出的Shadow-FT不引入额外参数，易于实现，并显著提高性能。我们在Qwen 3和Llama 3系列等主流LLM上进行了广泛的微调实验，并在涵盖编程、推理和数学任务的19个基准上评估了它们。实验结果表明，Shadow-FT一致地优于传统的全参数和高效参数微调方法。进一步的分析表明，Shadow-FT可以应用于多模态大规模语言模型（MLLMs），并与直接偏好优化（DPO）结合使用。代码和权重可在Github获取。 

---
# Any-to-Any Learning in Computational Pathology via Triplet Multimodal Pretraining 

**Title (ZH)**: 通过三重多模态预训练实现任意到任意的计算病理学学习 

**Authors**: Qichen Sun, Zhengrui Guo, Rui Peng, Hao Chen, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12711)  

**Abstract**: Recent advances in computational pathology and artificial intelligence have significantly enhanced the utilization of gigapixel whole-slide images and and additional modalities (e.g., genomics) for pathological diagnosis. Although deep learning has demonstrated strong potential in pathology, several key challenges persist: (1) fusing heterogeneous data types requires sophisticated strategies beyond simple concatenation due to high computational costs; (2) common scenarios of missing modalities necessitate flexible strategies that allow the model to learn robustly in the absence of certain modalities; (3) the downstream tasks in CPath are diverse, ranging from unimodal to multimodal, cnecessitating a unified model capable of handling all modalities. To address these challenges, we propose ALTER, an any-to-any tri-modal pretraining framework that integrates WSIs, genomics, and pathology reports. The term "any" emphasizes ALTER's modality-adaptive design, enabling flexible pretraining with any subset of modalities, and its capacity to learn robust, cross-modal representations beyond WSI-centric approaches. We evaluate ALTER across extensive clinical tasks including survival prediction, cancer subtyping, gene mutation prediction, and report generation, achieving superior or comparable performance to state-of-the-art baselines. 

**Abstract (ZH)**: 近期计算病理学和人工智能的进展显著增强了吉格像素全切片图像和额外模态（如基因组学）在病理诊断中的应用。尽管深度学习在病理学中显示出强大的潜力，但仍存在几个关键挑战：（1）融合异构数据类型需要超出简单连接的复杂策略以应对高昂的计算成本；（2）常见的模态缺失场景需要灵活的策略，使模型能够在缺乏某些模态的情况下学习；（3）计算病理学中的下游任务多样，从单模态到多模态不等，需要一个能够处理所有模态的统一模型。为应对这些挑战，我们提出了ALTER，这是一个任意到任意的三模态预训练框架，整合了全切片图像、基因组学和病理报告。术语“任意”突出了ALTER的模态自适应设计，使其能够灵活地使用任何组合的模态进行预训练，并具备超越以全切片图像为中心的方法学习稳健且跨模态表示的能力。我们在广泛的临床任务中评估了ALTER，包括生存预测、癌症亚型分类、基因突变预测和报告生成，性能优于或可与最先进的基线方法媲美。 

---
# PLAICraft: Large-Scale Time-Aligned Vision-Speech-Action Dataset for Embodied AI 

**Title (ZH)**: PLAICraft:大规模时间对齐的视觉-语音-动作数据集用于具身AI 

**Authors**: Yingchen He, Christian D. Weilbach, Martyna E. Wojciechowska, Yuxuan Zhang, Frank Wood  

**Link**: [PDF](https://arxiv.org/pdf/2505.12707)  

**Abstract**: Advances in deep generative modelling have made it increasingly plausible to train human-level embodied agents. Yet progress has been limited by the absence of large-scale, real-time, multi-modal, and socially interactive datasets that reflect the sensory-motor complexity of natural environments. To address this, we present PLAICraft, a novel data collection platform and dataset capturing multiplayer Minecraft interactions across five time-aligned modalities: video, game output audio, microphone input audio, mouse, and keyboard actions. Each modality is logged with millisecond time precision, enabling the study of synchronous, embodied behaviour in a rich, open-ended world. The dataset comprises over 10,000 hours of gameplay from more than 10,000 global participants.\footnote{We have done a privacy review for the public release of an initial 200-hour subset of the dataset, with plans to release most of the dataset over time.} Alongside the dataset, we provide an evaluation suite for benchmarking model capabilities in object recognition, spatial awareness, language grounding, and long-term memory. PLAICraft opens a path toward training and evaluating agents that act fluently and purposefully in real time, paving the way for truly embodied artificial intelligence. 

**Abstract (ZH)**: 深生成模型的进展使训练人类水平的具身代理变得越来越可行。然而，进展受限于缺乏大规模、实时、多模态和社会互动的数据集，这些数据集能够反映自然环境的感官-运动复杂性。为了解决这一问题，我们呈现了PLAICraft，一个新颖的数据采集平台和数据集，用于捕捉跨五个时间对齐模态的多人Minecraft交互：视频、游戏输出音频、麦克风输入音频、鼠标和键盘操作。每个模态以毫秒级时间精度记录，能够研究丰富、开放世界中的同步具身行为。该数据集包含超过10,000小时的游戏玩法，参与玩家超过10,000名全球参与者。（我们已对公共发布初始200小时数据子集进行了隐私审查，并计划逐步发布大部分数据集。）除了数据集，我们还提供了一套评估套件，用于基准测试模型在物体识别、空间意识、语言接地和长期记忆方面的能力。PLAICraft 为训练和评估能够实时流畅且有目的地行动的代理奠定了道路，铺就了真正具身人工智能的道路。 

---
# DreamGen: Unlocking Generalization in Robot Learning through Neural Trajectories 

**Title (ZH)**: DreamGen: 通过神经轨迹解锁机器人学习的通用性 

**Authors**: Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, Loic Magne, Ajay Mandlekar, Avnish Narayan, You Liang Tan, Guanzhi Wang, Jing Wang, Qi Wang, Yinzhen Xu, Xiaohui Zeng, Kaiyuan Zheng, Ruijie Zheng, Ming-Yu Liu, Luke Zettlemoyer, Dieter Fox, Jan Kautz, Scott Reed, Yuke Zhu, Linxi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12705)  

**Abstract**: We introduce DreamGen, a simple yet highly effective 4-stage pipeline for training robot policies that generalize across behaviors and environments through neural trajectories - synthetic robot data generated from video world models. DreamGen leverages state-of-the-art image-to-video generative models, adapting them to the target robot embodiment to produce photorealistic synthetic videos of familiar or novel tasks in diverse environments. Since these models generate only videos, we recover pseudo-action sequences using either a latent action model or an inverse-dynamics model (IDM). Despite its simplicity, DreamGen unlocks strong behavior and environment generalization: a humanoid robot can perform 22 new behaviors in both seen and unseen environments, while requiring teleoperation data from only a single pick-and-place task in one environment. To evaluate the pipeline systematically, we introduce DreamGen Bench, a video generation benchmark that shows a strong correlation between benchmark performance and downstream policy success. Our work establishes a promising new axis for scaling robot learning well beyond manual data collection. 

**Abstract (ZH)**: DreamGen：一种通过神经轨迹训练跨行为和环境泛化机器人策略的简单高效4阶段流程 

---
# Counterfactual Explanations for Continuous Action Reinforcement Learning 

**Title (ZH)**: 连续行动强化学习的对抗事实解释 

**Authors**: Shuyang Dong, Shangtong Zhang, Lu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12701)  

**Abstract**: Reinforcement Learning (RL) has shown great promise in domains like healthcare and robotics but often struggles with adoption due to its lack of interpretability. Counterfactual explanations, which address "what if" scenarios, provide a promising avenue for understanding RL decisions but remain underexplored for continuous action spaces. We propose a novel approach for generating counterfactual explanations in continuous action RL by computing alternative action sequences that improve outcomes while minimizing deviations from the original sequence. Our approach leverages a distance metric for continuous actions and accounts for constraints such as adhering to predefined policies in specific states. Evaluations in two RL domains, Diabetes Control and Lunar Lander, demonstrate the effectiveness, efficiency, and generalization of our approach, enabling more interpretable and trustworthy RL applications. 

**Abstract (ZH)**: 增强学习（RL）在医疗保健和机器人技术等领域展现了巨大的潜力，但由于其缺乏可解释性，往往难以推广应用。针对“what if”情景的反事实解释提供了一种理解RL决策的有前景的方法，但在连续动作空间中仍处于未充分利用状态。我们提出了一种通过计算改善结果的同时 minimizes 偏离原序列的替代动作序列来生成连续动作RL中的反事实解释的新方法。该方法利用连续动作的距离度量，并考虑如在特定状态下遵守预定义策略等约束条件。在糖尿病控制和月球着陆两个RL领域的评估表明，该方法具有有效性、效率和泛化能力，能够促进更具可解释性和可信度的RL应用。 

---
# Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement 

**Title (ZH)**: 面向有效的联邦图基础模型：减轻知识纠缠 

**Authors**: Yinlin Zhu, Xunkai Li, Jishuo Jia, Miao Hu, Di Wu, Meikang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12684)  

**Abstract**: Recent advances in graph machine learning have shifted to data-centric paradigms, driven by two emerging fields: (1) Federated graph learning (FGL) enables multi-client collaboration but faces challenges from data and task heterogeneity, limiting its practicality; (2) Graph foundation models (GFM) offer strong domain generalization but are usually trained on single machines, missing out on cross-silo data and resources.
These paradigms are complementary, and their integration brings notable benefits. Motivated by this, we propose FedGFM, a novel decentralized GFM training paradigm. However, a key challenge is knowledge entanglement, where multi-domain knowledge merges into indistinguishable representations, hindering downstream adaptation.
To address this, we present FedGFM+, an enhanced framework with two core modules to reduce knowledge entanglement: (1) AncDAI: A global anchor-based domain-aware initialization strategy. Before pre-training, each client encodes its local graph into domain-specific prototypes that serve as semantic anchors. Synthetic embeddings around these anchors initialize the global model. We theoretically prove these prototypes are distinguishable across domains, providing a strong inductive bias to disentangle domain-specific knowledge. (2) AdaDPP: A local adaptive domain-sensitive prompt pool. Each client learns a lightweight graph prompt capturing domain semantics during pre-training. During fine-tuning, prompts from all clients form a pool from which the GFM selects relevant prompts to augment target graph attributes, improving downstream adaptation.
FedGFM+ is evaluated on 8 diverse benchmarks across multiple domains and tasks, outperforming 20 baselines from supervised learning, FGL, and federated GFM variants. 

**Abstract (ZH)**: Recent Advances in Graph Machine Learning Have Shifted to Data-Centric Paradigms Driven by Federated Graph Learning and Graph Foundation Models 

---
# Text2midi-InferAlign: Improving Symbolic Music Generation with Inference-Time Alignment 

**Title (ZH)**: Text2MIDI-InferAlign：改进的推理时对齐的符号音乐生成 

**Authors**: Abhinaba Roy, Geeta Puri, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2505.12669)  

**Abstract**: We present Text2midi-InferAlign, a novel technique for improving symbolic music generation at inference time. Our method leverages text-to-audio alignment and music structural alignment rewards during inference to encourage the generated music to be consistent with the input caption. Specifically, we introduce two objectives scores: a text-audio consistency score that measures rhythmic alignment between the generated music and the original text caption, and a harmonic consistency score that penalizes generated music containing notes inconsistent with the key. By optimizing these alignment-based objectives during the generation process, our model produces symbolic music that is more closely tied to the input captions, thereby improving the overall quality and coherence of the generated compositions. Our approach can extend any existing autoregressive model without requiring further training or fine-tuning. We evaluate our work on top of Text2midi - an existing text-to-midi generation model, demonstrating significant improvements in both objective and subjective evaluation metrics. 

**Abstract (ZH)**: Text2midi-InferAlign：一种改进推理时符号音乐生成的新技术 

---
# Multi-View Wireless Sensing via Conditional Generative Learning: Framework and Model Design 

**Title (ZH)**: 基于条件生成学习的多视图无线传感：架构与模型设计 

**Authors**: Ziqing Xing, Zhaoyang Zhang, Zirui Chen, Hongning Ruan, Zhaohui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12664)  

**Abstract**: In this paper, we incorporate physical knowledge into learning-based high-precision target sensing using the multi-view channel state information (CSI) between multiple base stations (BSs) and user equipment (UEs). Such kind of multi-view sensing problem can be naturally cast into a conditional generation framework. To this end, we design a bipartite neural network architecture, the first part of which uses an elaborately designed encoder to fuse the latent target features embedded in the multi-view CSI, and then the second uses them as conditioning inputs of a powerful generative model to guide the target's reconstruction. Specifically, the encoder is designed to capture the physical correlation between the CSI and the target, and also be adaptive to the numbers and positions of BS-UE pairs. Therein the view-specific nature of CSI is assimilated by introducing a spatial positional embedding scheme, which exploits the structure of electromagnetic(EM)-wave propagation channels. Finally, a conditional diffusion model with a weighted loss is employed to generate the target's point cloud from the fused features. Extensive numerical results demonstrate that the proposed generative multi-view (Gen-MV) sensing framework exhibits excellent flexibility and significant performance improvement on the reconstruction quality of target's shape and EM properties. 

**Abstract (ZH)**: 在本文中，我们通过多基站（BSs）与用户设备（UEs）之间的多视图信道状态信息（CSI），将物理知识融入基于学习的高精度目标感知中。这种多视图感知问题可以自然地映射到条件生成框架中。为此，我们设计了一种双部分神经网络架构，其中第一部分使用精心设计的编码器融合嵌入在多视图CSI中的潜在目标特征，然后将这些特征作为强大生成模型的条件输入以指导目标重建。具体来说，编码器被设计为捕获CSI与目标之间的物理相关性，并且能够适应BS-UE对的数量和位置。在此过程中，通过引入空间位置嵌入方案来吸收CSI的视图特定性质，该方案利用电磁波传播通道的结构。最后，使用带加权损失的条件扩散模型从融合的特征中生成目标的点云。大量数值结果表明，所提出的生成多视图（Gen-MV）感知框架在目标形状和电磁特性重建质量方面表现出色。 

---
# Know3-RAG: A Knowledge-aware RAG Framework with Adaptive Retrieval, Generation, and Filtering 

**Title (ZH)**: Know3-RAG：一种具备适应性检索、生成与过滤的知识aware框架 

**Authors**: Xukai Liu, Ye Liu, Shiwen Wu, Yanghai Zhang, Yihao Yuan, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12662)  

**Abstract**: Recent advances in large language models (LLMs) have led to impressive progress in natural language generation, yet their tendency to produce hallucinated or unsubstantiated content remains a critical concern. To improve factual reliability, Retrieval-Augmented Generation (RAG) integrates external knowledge during inference. However, existing RAG systems face two major limitations: (1) unreliable adaptive control due to limited external knowledge supervision, and (2) hallucinations caused by inaccurate or irrelevant references. To address these issues, we propose Know3-RAG, a knowledge-aware RAG framework that leverages structured knowledge from knowledge graphs (KGs) to guide three core stages of the RAG process, including retrieval, generation, and filtering. Specifically, we introduce a knowledge-aware adaptive retrieval module that employs KG embedding to assess the confidence of the generated answer and determine retrieval necessity, a knowledge-enhanced reference generation strategy that enriches queries with KG-derived entities to improve generated reference relevance, and a knowledge-driven reference filtering mechanism that ensures semantic alignment and factual accuracy of references. Experiments on multiple open-domain QA benchmarks demonstrate that Know3-RAG consistently outperforms strong baselines, significantly reducing hallucinations and enhancing answer reliability. 

**Abstract (ZH)**: Recent Advances in Large Language Models (LLMs) Have Led to Impressive Progress in Natural Language Generation, Yet Their Tendency to Produce Hallucinated or Unsubstantiated Content Remains a Critical Concern: Know3-RAG, a Knowledge-Aware Retrieval-Augmented Generation Framework for Factual Reliability 

---
# Web IP at Risk: Prevent Unauthorized Real-Time Retrieval by Large Language Models 

**Title (ZH)**: Web IP at Risk: 防止大型语言模型未经授权的实时检索 

**Authors**: Yisheng Zhong, Yizhu Wen, Junfeng Guo, Mehran Kafai, Heng Huang, Hanqing Guo, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12655)  

**Abstract**: Protecting cyber Intellectual Property (IP) such as web content is an increasingly critical concern. The rise of large language models (LLMs) with online retrieval capabilities presents a double-edged sword that enables convenient access to information but often undermines the rights of original content creators. As users increasingly rely on LLM-generated responses, they gradually diminish direct engagement with original information sources, significantly reducing the incentives for IP creators to contribute, and leading to a saturating cyberspace with more AI-generated content. In response, we propose a novel defense framework that empowers web content creators to safeguard their web-based IP from unauthorized LLM real-time extraction by leveraging the semantic understanding capability of LLMs themselves. Our method follows principled motivations and effectively addresses an intractable black-box optimization problem. Real-world experiments demonstrated that our methods improve defense success rates from 2.5% to 88.6% on different LLMs, outperforming traditional defenses such as configuration-based restrictions. 

**Abstract (ZH)**: 保护网络知识产权：利用大型语言模型的语义理解能力防范未经授权的实时提取 

---
# Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals 

**Title (ZH)**: 基于语言、声学和视觉信号的人机对话中的轮替预测与下行反馈研究 

**Authors**: Yuxin Lin, Yinglin Zheng, Ming Zeng, Wangzheng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12654)  

**Abstract**: This paper addresses the gap in predicting turn-taking and backchannel actions in human-machine conversations using multi-modal signals (linguistic, acoustic, and visual). To overcome the limitation of existing datasets, we propose an automatic data collection pipeline that allows us to collect and annotate over 210 hours of human conversation videos. From this, we construct a Multi-Modal Face-to-Face (MM-F2F) human conversation dataset, including over 1.5M words and corresponding turn-taking and backchannel annotations from approximately 20M frames. Additionally, we present an end-to-end framework that predicts the probability of turn-taking and backchannel actions from multi-modal signals. The proposed model emphasizes the interrelation between modalities and supports any combination of text, audio, and video inputs, making it adaptable to a variety of realistic scenarios. Our experiments show that our approach achieves state-of-the-art performance on turn-taking and backchannel prediction tasks, achieving a 10\% increase in F1-score on turn-taking and a 33\% increase on backchannel prediction. Our dataset and code are publicly available online to ease of subsequent research. 

**Abstract (ZH)**: 本文通过多模态信号（语言、声学和视觉）解决在人类-机器对话中预测轮流说话和补充反应行为的空白。为克服现有数据集的局限，我们提出了一个自动数据收集管道，以收集和标注超过210小时的人类对话视频。基于这些数据，我们构建了一个多模态面对面（MM-F2F）人类对话数据集，包括超过150万单词及其对应的轮流说话和补充反应标注，涉及约2亿帧。此外，我们还提出了一种端到端框架，从多模态信号中预测轮流说话和补充反应的行为概率。所提出的模型强调了模态间的相互关系，并支持文本、音频和视频任意组合的输入，使其适应各种现实场景。我们的实验表明，我们的方法在轮流说话和补充反应预测任务上达到了最先进的性能，分别在轮流说话预测的F1分数上提高了10%，在补充反应预测上提高了33%。我们的数据集和代码已在线公开，以方便后续研究。 

---
# AutoMat: Enabling Automated Crystal Structure Reconstruction from Microscopy via Agentic Tool Use 

**Title (ZH)**: AutoMat: 通过代理工具使用实现从显微镜数据分析的自动晶体结构重建 

**Authors**: Yaotian Yang, Yiwen Tang, Yizhe Chen, Xiao Chen, Jiangjie Qiu, Hao Xiong, Haoyu Yin, Zhiyao Luo, Yifei Zhang, Sijia Tao, Wentao Li, Qinghua Zhang, Yuqiang Li, Wanli Ouyang, Bin Zhao, Xiaonan Wang, Fei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.12650)  

**Abstract**: Machine learning-based interatomic potentials and force fields depend critically on accurate atomic structures, yet such data are scarce due to the limited availability of experimentally resolved crystals. Although atomic-resolution electron microscopy offers a potential source of structural data, converting these images into simulation-ready formats remains labor-intensive and error-prone, creating a bottleneck for model training and validation. We introduce AutoMat, an end-to-end, agent-assisted pipeline that automatically transforms scanning transmission electron microscopy (STEM) images into atomic crystal structures and predicts their physical properties. AutoMat combines pattern-adaptive denoising, physics-guided template retrieval, symmetry-aware atomic reconstruction, fast relaxation and property prediction via MatterSim, and coordinated orchestration across all stages. We propose the first dedicated STEM2Mat-Bench for this task and evaluate performance using lattice RMSD, formation energy MAE, and structure-matching success rate. By orchestrating external tool calls, AutoMat enables a text-only LLM to outperform vision-language models in this domain, achieving closed-loop reasoning throughout the pipeline. In large-scale experiments over 450 structure samples, AutoMat substantially outperforms existing multimodal large language models and tools. These results validate both AutoMat and STEM2Mat-Bench, marking a key step toward bridging microscopy and atomistic simulation in materials this http URL code and dataset are publicly available at this https URL and this https URL. 

**Abstract (ZH)**: 基于机器学习的原子势和力场依赖于精确的原子结构，但由于实验解析晶体的稀缺性，这类数据很有限。尽管原子分辨率电子显微镜提供了结构数据的潜在来源，但将这些图像转换为可用于模拟的格式仍然是劳动密集且容易出错的过程，成为模型训练和验证的瓶颈。我们引入了AutoMat，这是一个端到端的、代理辅助的管道，可以自动将扫描透射电子显微镜（STEM）图像转换为原子晶体结构，并预测其物理性质。AutoMat 结合了模式自适应去噪、物理引导的模板检索、对称意识原子重构、通过 MatterSim 快速弛豫和性质预测，以及在所有阶段协调编排。我们提出了首个针对此任务的专用STEM2Mat-Bench，使用晶格RMSD、形成能MAE和结构匹配成功率来评估性能。通过协调外部工具调用，AutoMat 使仅凭文本的LLM在该领域超越了视觉-语言模型，实现了管道中的闭环推理。在450个结构样本的大规模实验中，AutoMat 显著优于现有的多模态大语言模型和工具。这些结果验证了AutoMat和STEM2Mat-Bench，标志着在材料表征与原子级模拟结合方面的重要一步。相关代码和数据集可在以下网址公开访问：this https URL 和 this https URL。 

---
# Single Image Reflection Removal via inter-layer Complementarity 

**Title (ZH)**: 单图像反射去除 via 层间互补性 

**Authors**: Yue Huang, Zi'ang Li, Tianle Hu, Jie Wen, Guanbin Li, Jinglin Zhang, Guoxu Zhou, Xiaozhao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12641)  

**Abstract**: Although dual-stream architectures have achieved remarkable success in single image reflection removal, they fail to fully exploit inter-layer complementarity in their physical modeling and network design, which limits the quality of image separation. To address this fundamental limitation, we propose two targeted improvements to enhance dual-stream architectures: First, we introduce a novel inter-layer complementarity model where low-frequency components extracted from the residual layer interact with the transmission layer through dual-stream architecture to enhance inter-layer complementarity. Meanwhile, high-frequency components from the residual layer provide inverse modulation to both streams, improving the detail quality of the transmission layer. Second, we propose an efficient inter-layer complementarity attention mechanism which first cross-reorganizes dual streams at the channel level to obtain reorganized streams with inter-layer complementary structures, then performs attention computation on the reorganized streams to achieve better inter-layer separation, and finally restores the original stream structure for output. Experimental results demonstrate that our method achieves state-of-the-art separation quality on multiple public datasets while significantly reducing both computational cost and model complexity. 

**Abstract (ZH)**: 尽管双流架构在单图像反射去除任务中取得了显著成功，但它们未能充分利用其物理建模和网络设计中的层间互补性，这限制了图像分离的质量。为解决这一根本性限制，我们提出了两种针对性改进以增强双流架构：首先，我们引入了一种新型的层间互补模型，其中来自残差层的低频成分通过双流架构与透射层交互，增强层间互补性。同时，残差层的高频成分对两个流进行反调制，从而提高透射层的细节质量。其次，我们提出了一种高效的层间互补注意力机制，该机制首先在通道级别交叉重组双流，获得具有层间互补结构的重组流，然后在重组流上执行注意力计算以实现更好的层间分离，最后恢复原始流结构以输出。实验结果表明，我们的方法在多个公开数据集上实现了最先进的分离质量，同时显著降低了计算成本和模型复杂度。 

---
# ChromFound: Towards A Universal Foundation Model for Single-Cell Chromatin Accessibility Data 

**Title (ZH)**: ChromFound: 向统一的单细胞染色质可及性基础模型迈进 

**Authors**: Yifeng Jiao, Yuchen Liu, Yu Zhang, Xin Guo, Yushuai Wu, Chen Jiang, Jiyang Li, Hongwei Zhang, Limei Han, Xin Gao, Yuan Qi, Yuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12638)  

**Abstract**: The advent of single-cell Assay for Transposase-Accessible Chromatin using sequencing (scATAC-seq) offers an innovative perspective for deciphering regulatory mechanisms by assembling a vast repository of single-cell chromatin accessibility data. While foundation models have achieved significant success in single-cell transcriptomics, there is currently no foundation model for scATAC-seq that supports zero-shot high-quality cell identification and comprehensive multi-omics analysis simultaneously. Key challenges lie in the high dimensionality and sparsity of scATAC-seq data, as well as the lack of a standardized schema for representing open chromatin regions (OCRs). Here, we present \textbf{ChromFound}, a foundation model tailored for scATAC-seq. ChromFound utilizes a hybrid architecture and genome-aware tokenization to effectively capture genome-wide long contexts and regulatory signals from dynamic chromatin landscapes. Pretrained on 1.97 million cells from 30 tissues and 6 disease conditions, ChromFound demonstrates broad applicability across 6 diverse tasks. Notably, it achieves robust zero-shot performance in generating universal cell representations and exhibits excellent transferability in cell type annotation and cross-omics prediction. By uncovering enhancer-gene links undetected by existing computational methods, ChromFound offers a promising framework for understanding disease risk variants in the noncoding genome. 

**Abstract (ZH)**: 单细胞转座酶可及染色质测序（scATAC-seq）测序 assay 的出现为通过构建单细胞染色质可及性大数据库来解析调节机制提供了创新视角。尽管基础模型在单细胞转录组学中取得了显著成功，但目前尚无支持零样本高质量细胞识别和综合多组学分析的基础模型。关键挑战在于单细胞转座酶可及染色质测序数据的高维度和稀疏性，以及缺乏标准化的开放染色质区域（OCRs）表示方案。在这里，我们提出了一种针对scATAC-seq的foundation模型——ChromFound。ChromFound利用混合架构和基因组意识的标记化技术，能够有效捕捉整个基因组范围内的长上下文和调节信号，从动态染色质景观中。ChromFound基于来自30种组织和6种疾病状况的197万个细胞的预训练，在6个不同的任务中表现出广泛的适用性。特别是在生成通用细胞表示和细胞类型注释及跨组学预测方面的零样本性能表现 robust，且表现出出色的可转移性。通过发现现有计算方法未检测到的增强子-基因关联，ChromFound为理解非编码基因组中的疾病风险变异提供了有前景的框架。 

---
# Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents 

**Title (ZH)**: 跨平台移动代理的可扩展视频到数据集生成 

**Authors**: Yunseok Jang, Yeda Song, Sungryull Sohn, Lajanugen Logeswaran, Tiange Luo, Dong-Ki Kim, Kyunghoon Bae, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12632)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have sparked significant interest in developing GUI visual agents. We introduce MONDAY (Mobile OS Navigation Task Dataset for Agents from YouTube), a large-scale dataset of 313K annotated frames from 20K instructional videos capturing diverse real-world mobile OS navigation across multiple platforms. Models that include MONDAY in their pre-training phases demonstrate robust cross-platform generalization capabilities, consistently outperforming models trained on existing single OS datasets while achieving an average performance gain of 18.11%p on an unseen mobile OS platform. To enable continuous dataset expansion as mobile platforms evolve, we present an automated framework that leverages publicly available video content to create comprehensive task datasets without manual annotation. Our framework comprises robust OCR-based scene detection (95.04% F1score), near-perfect UI element detection (99.87% hit ratio), and novel multi-step action identification to extract reliable action sequences across diverse interface configurations. We contribute both the MONDAY dataset and our automated collection framework to facilitate future research in mobile OS navigation. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）和视觉-语言模型（VLMs）的发展激发了对GUI视觉代理的兴趣。我们介绍了MONDAY（来自YouTube的移动操作系统导航任务数据集），这是一个包含31.3万标注帧的数据集，来源于2万条教学视频，捕捉了多种平台的多样化真实世界移动操作系统导航。包含MONDAY的数据预训练模型展示了跨平台的稳健泛化能力，持续优于基于现有单一操作系统数据集训练的模型，在未见过的操作系统平台上实现了平均18.11%的性能提升。为了随着移动平台的发展持续扩展数据集，我们提出了一种自动化框架，利用公开的视频内容创建全面的任务数据集，无需人工标注。该框架包括鲁棒的基于OCR的场景检测（F1得分为95.04%）、近乎完美的UI元素检测（命中率为99.87%）以及新颖的多步动作识别，以提取各种界面配置下的可靠动作序列。我们贡献了MONDAY数据集和我们的自动化收集框架，以促进移动操作系统导航领域的未来研究。 

---
# Degradation-Aware Feature Perturbation for All-in-One Image Restoration 

**Title (ZH)**: awareness特征扰动实现一体化图像恢复 

**Authors**: Xiangpeng Tian, Xiangyu Liao, Xiao Liu, Meng Li, Chao Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.12630)  

**Abstract**: All-in-one image restoration aims to recover clear images from various degradation types and levels with a unified model. Nonetheless, the significant variations among degradation types present challenges for training a universal model, often resulting in task interference, where the gradient update directions of different tasks may diverge due to shared parameters. To address this issue, motivated by the routing strategy, we propose DFPIR, a novel all-in-one image restorer that introduces Degradation-aware Feature Perturbations(DFP) to adjust the feature space to align with the unified parameter space. In this paper, the feature perturbations primarily include channel-wise perturbations and attention-wise perturbations. Specifically, channel-wise perturbations are implemented by shuffling the channels in high-dimensional space guided by degradation types, while attention-wise perturbations are achieved through selective masking in the attention space. To achieve these goals, we propose a Degradation-Guided Perturbation Block (DGPB) to implement these two functions, positioned between the encoding and decoding stages of the encoder-decoder architecture. Extensive experimental results demonstrate that DFPIR achieves state-of-the-art performance on several all-in-one image restoration tasks including image denoising, image dehazing, image deraining, motion deblurring, and low-light image enhancement. Our codes are available at this https URL. 

**Abstract (ZH)**: 整体图像恢复旨在通过统一模型从各种退化类型和程度中恢复清晰图像。然而，退化类型的显著差异为训练通用模型带来了挑战，常常导致任务干扰，即由于共享参数，不同任务的梯度更新方向可能发散。为了解决这一问题，受路由策略的启发，我们提出DFPIR，一种新颖的整体图像恢复器，引入了退化感知特征扰动（DFP）以调整特征空间并使其与统一参数空间对齐。在本文中，特征扰动主要包括通道级扰动和注意力级扰动。具体而言，通道级扰动通过在高维空间中由退化类型指导的通道混排实现，而注意力级扰动则是通过注意力空间中的选择性掩蔽实现。为了实现这些目标，我们提出了一种退化引导扰动块（DGPB），在编码器-解码器架构的编码和解码阶段之间实施这两种功能。大量实验结果表明，DFPIR在包括图像去噪、图像去雾、图像去雨、运动去模糊和低光图像增强在内的多种整体图像恢复任务中取得了最先进的性能。我们的代码可在以下链接获取：this https URL。 

---
# scSiameseClu: A Siamese Clustering Framework for Interpreting single-cell RNA Sequencing Data 

**Title (ZH)**: scSiameseClu: 一种用于解释单细胞RNA测序数据的Siamese聚类框架 

**Authors**: Ping Xu, Zhiyuan Ning, Pengjiang Li, Wenhao Liu, Pengyang Wang, Jiaxu Cui, Yuanchun Zhou, Pengfei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12626)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) reveals cell heterogeneity, with cell clustering playing a key role in identifying cell types and marker genes. Recent advances, especially graph neural networks (GNNs)-based methods, have significantly improved clustering performance. However, the analysis of scRNA-seq data remains challenging due to noise, sparsity, and high dimensionality. Compounding these challenges, GNNs often suffer from over-smoothing, limiting their ability to capture complex biological information. In response, we propose scSiameseClu, a novel Siamese Clustering framework for interpreting single-cell RNA-seq data, comprising of 3 key steps: (1) Dual Augmentation Module, which applies biologically informed perturbations to the gene expression matrix and cell graph relationships to enhance representation robustness; (2) Siamese Fusion Module, which combines cross-correlation refinement and adaptive information fusion to capture complex cellular relationships while mitigating over-smoothing; and (3) Optimal Transport Clustering, which utilizes Sinkhorn distance to efficiently align cluster assignments with predefined proportions while maintaining balance. Comprehensive evaluations on seven real-world datasets demonstrate that~\methodname~outperforms state-of-the-art methods in single-cell clustering, cell type annotation, and cell type classification, providing a powerful tool for scRNA-seq data interpretation. 

**Abstract (ZH)**: 单细胞RNA测序(scRNA-seq)的Siamese Clu框架揭示了细胞异质性，细胞聚类在识别细胞类型和标志性基因中起关键作用。基于图神经网络(GNN)的recent进展显著提高了聚类性能。但由于噪声、稀疏性和高维度等挑战，scRNA-seq数据的分析仍然具有挑战性。此外，GNNs常常遭受过度平滑的困扰，限制了其捕捉复杂生物信息的能力。为应对这些挑战，我们提出了一个名为scSiameseClu的新颖Siamese聚类框架，用于解释单细胞RNA测序数据，该框架包括三个关键步骤：（1）双增强模块，通过在基因表达矩阵和细胞图关系中应用生物信息学指导的扰动来增强表示稳健性；（2）Siamese融合模块，通过交叉相关性细化和自适应信息融合来捕捉复杂的细胞关系，同时减轻过度平滑；（3）最优传输聚类，利用Sinkhorn距离高效地将聚类分配与预定义的比例对齐，同时保持平衡。在七个真实数据集上的综合评估表明，scSiameseClu在单细胞聚类、细胞类型注释和细胞类型分类方面均优于现有方法，提供了一个强大的工具用于scRNA-seq数据分析。 

---
# Lightweight and Effective Preference Construction in PIBT for Large-Scale Multi-Agent Pathfinding 

**Title (ZH)**: 大型多agent路径查找中轻量且有效的偏好构造在PIBT中 

**Authors**: Keisuke Okumura, Hiroki Nagai  

**Link**: [PDF](https://arxiv.org/pdf/2505.12623)  

**Abstract**: PIBT is a computationally lightweight algorithm that can be applied to a variety of multi-agent pathfinding (MAPF) problems, generating the next collision-free locations of agents given another. Because of its simplicity and scalability, it is becoming a popular underlying scheme for recent large-scale MAPF methods involving several hundreds or thousands of agents. Vanilla PIBT makes agents behave greedily towards their assigned goals, while agents typically have multiple best actions, since the graph shortest path is not always unique. Consequently, tiebreaking about how to choose between these actions significantly affects resulting solutions. This paper studies two simple yet effective techniques for tiebreaking in PIBT, without compromising its computational advantage. The first technique allows an agent to intelligently dodge another, taking into account whether each action will hinder the progress of the next timestep. The second technique is to learn, through multiple PIBT runs, how an action causes regret in others and to use this information to minimise regret collectively. Our empirical results demonstrate that these techniques can reduce the solution cost of one-shot MAPF and improve the throughput of lifelong MAPF. For instance, in densely populated one-shot cases, the combined use of these tiebreaks achieves improvements of around 10-20% in sum-of-costs, without significantly compromising the speed of a PIBT-based planner. 

**Abstract (ZH)**: PIBT是一种计算负担轻的算法，可以应用于多种多智能体路径规划（MAPF）问题，给定另一个智能体的位置，生成下一个无碰撞的位置。由于其简单性和可扩展性，它正成为涉及数百或数千个智能体的最新大规模MAPF方法的流行基础方案。vanilla PIBT使智能体贪婪地朝其分配的目标行动，但由于图最短路径不一定唯一，智能体通常有多项最佳行动。因此，如何在这项行动之间进行选择的舍让规则对最终解决方案有显著影响。本文研究了两种简单有效的PIBT舍让策略，而不牺牲其计算优势。第一种技术使智能体能够智能地避开其他智能体，考虑每项行动是否会妨碍下一时间步的进展。第二种技术是通过多次PIBT运行学习一项行动如何导致其他智能体的后悔，并利用此信息来最小化集体后悔。我们的实验结果表明，这些技术可以降低一次性MAPF解决方案的成本，并提高生命周期MAPF的吞吐量。例如，在高密度一次性案例中，这两种舍让规则的结合使用可以在综合成本上实现约10-20%的改进，而不显著牺牲基于PIBT规划器的速度。 

---
# AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection 

**Title (ZH)**: AD-AGENT：端到端异常检测的多agent框架 

**Authors**: Tiankai Yang, Junjun Liu, Wingchun Siu, Jiahang Wang, Zhuangzhuang Qian, Chanjuan Song, Cheng Cheng, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12594)  

**Abstract**: Anomaly detection (AD) is essential in areas such as fraud detection, network monitoring, and scientific research. However, the diversity of data modalities and the increasing number of specialized AD libraries pose challenges for non-expert users who lack in-depth library-specific knowledge and advanced programming skills. To tackle this, we present AD-AGENT, an LLM-driven multi-agent framework that turns natural-language instructions into fully executable AD pipelines. AD-AGENT coordinates specialized agents for intent parsing, data preparation, library and model selection, documentation mining, and iterative code generation and debugging. Using a shared short-term workspace and a long-term cache, the agents integrate popular AD libraries like PyOD, PyGOD, and TSLib into a unified workflow. Experiments demonstrate that AD-AGENT produces reliable scripts and recommends competitive models across libraries. The system is open-sourced to support further research and practical applications in AD. 

**Abstract (ZH)**: 基于LLM的多Agent异常检测框架AD-AGENT 

---
# Learning Robust Spectral Dynamics for Temporal Domain Generalization 

**Title (ZH)**: 学习 robust 谱动力学以实现时间域泛化 

**Authors**: En Yu, Jie Lu, Xiaoyu Yang, Guangquan Zhang, Zhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12585)  

**Abstract**: Modern machine learning models struggle to maintain performance in dynamic environments where temporal distribution shifts, \emph{i.e., concept drift}, are prevalent. Temporal Domain Generalization (TDG) seeks to enable model generalization across evolving domains, yet existing approaches typically assume smooth incremental changes, struggling with complex real-world drifts involving long-term structure (incremental evolution/periodicity) and local uncertainties. To overcome these limitations, we introduce FreKoo, which tackles these challenges via a novel frequency-domain analysis of parameter trajectories. It leverages the Fourier transform to disentangle parameter evolution into distinct spectral bands. Specifically, low-frequency component with dominant dynamics are learned and extrapolated using the Koopman operator, robustly capturing diverse drift patterns including both incremental and periodicity. Simultaneously, potentially disruptive high-frequency variations are smoothed via targeted temporal regularization, preventing overfitting to transient noise and domain uncertainties. In addition, this dual spectral strategy is rigorously grounded through theoretical analysis, providing stability guarantees for the Koopman prediction, a principled Bayesian justification for the high-frequency regularization, and culminating in a multiscale generalization bound connecting spectral dynamics to improved generalization. Extensive experiments demonstrate FreKoo's significant superiority over SOTA TDG approaches, particularly excelling in real-world streaming scenarios with complex drifts and uncertainties. 

**Abstract (ZH)**: 现代机器学习模型在存在时间分布变化（即概念漂移）的动态环境中难以保持性能。时间域泛化（TDG）旨在使模型能够适应不断演变的领域，但现有方法通常假设平滑增量变化，难以应对涉及长期结构（增量进化/周期性）和局部不确定性等复杂真实世界漂移。为克服这些限制，我们引入了FreKoo，通过参数轨迹的新型频率域分析来应对这些挑战。它利用傅里叶变换将参数演化分解为不同的频带。具体而言，利用Koopman算子学习和外推主导动态的低频分量，稳健地捕捉包括增量和周期性在内的各种漂移模式。同时，通过目标时间正则化平滑潜在的破坏性高频变化，防止过拟合到瞬态噪声和领域不确定性。此外，这种双重频带策略通过理论分析严格建立，为Koopman预测提供了稳定性保证，为高频正则化提供了原则性的贝叶斯解释，最终通过频谱动力学与泛化提升建立多尺度泛化界。广泛的实验表明，FreKoo 在复杂的现实世界流式场景中显著优于当前最佳时间域泛化方法。 

---
# A Comprehensive Survey on Physical Risk Control in the Era of Foundation Model-enabled Robotics 

**Title (ZH)**: 基础模型驱动 robotic 领域中物理风险控制综述 

**Authors**: Takeshi Kojima, Yaonan Zhu, Yusuke Iwasawa, Toshinori Kitamura, Gang Yan, Shu Morikuni, Ryosuke Takanami, Alfredo Solano, Tatsuya Matsushima, Akiko Murakami, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12583)  

**Abstract**: Recent Foundation Model-enabled robotics (FMRs) display greatly improved general-purpose skills, enabling more adaptable automation than conventional robotics. Their ability to handle diverse tasks thus creates new opportunities to replace human labor. However, unlike general foundation models, FMRs interact with the physical world, where their actions directly affect the safety of humans and surrounding objects, requiring careful deployment and control. Based on this proposition, our survey comprehensively summarizes robot control approaches to mitigate physical risks by covering all the lifespan of FMRs ranging from pre-deployment to post-accident stage. Specifically, we broadly divide the timeline into the following three phases: (1) pre-deployment phase, (2) pre-incident phase, and (3) post-incident phase. Throughout this survey, we find that there is much room to study (i) pre-incident risk mitigation strategies, (ii) research that assumes physical interaction with humans, and (iii) essential issues of foundation models themselves. We hope that this survey will be a milestone in providing a high-resolution analysis of the physical risks of FMRs and their control, contributing to the realization of a good human-robot relationship. 

**Abstract (ZH)**: Recent Foundation Model-enabled Robotics (FMRs) 的进展显著提升了通用技能，使其能够实现比传统机器人更为灵活的自动化。因此，它们能够处理各种任务，为替代人类劳动提供了新的机会。然而，与通用基础模型不同的是，FMRs与物理世界进行互动，其行为直接影响到人类和周围物体的安全，因此需要谨慎部署和控制。基于这一观点，我们综述全面总结了机器人控制方法，以减轻FMRs在从预部署到事故发生后的整个生命周期中的物理风险。具体而言，我们将时间线划分为以下三个阶段：（1）预部署阶段，（2）预事故阶段，（3）事故发生后阶段。在这一综述中，我们发现有必要研究（i）预事故风险缓解策略，（ii）假定与人类有物理互动的研究，以及（iii）基础模型本身的关键问题。我们希望这一综述能够成为在FMRs及其控制的物理风险分析方面提供高分辨率分析的一个里程碑，从而有助于实现良好的人机关系。 

---
# An approach based on class activation maps for investigating the effects of data augmentation on neural networks for image classification 

**Title (ZH)**: 基于类激活图的方法研究数据增强对图像分类神经网络效果的影响 

**Authors**: Lucas M. Dorneles, Luan Fonseca Garcia, Joel Luís Carbonera  

**Link**: [PDF](https://arxiv.org/pdf/2505.12581)  

**Abstract**: Neural networks have become increasingly popular in the last few years as an effective tool for the task of image classification due to the impressive performance they have achieved on this task. In image classification tasks, it is common to use data augmentation strategies to increase the robustness of trained networks to changes in the input images and to avoid overfitting. Although data augmentation is a widely adopted technique, the literature lacks a body of research analyzing the effects data augmentation methods have on the patterns learned by neural network models working on complex datasets. The primary objective of this work is to propose a methodology and set of metrics that may allow a quantitative approach to analyzing the effects of data augmentation in convolutional networks applied to image classification. An important tool used in the proposed approach lies in the concept of class activation maps for said models, which allow us to identify and measure the importance these models assign to each individual pixel in an image when executing the classification task. From these maps, we may then extract metrics over the similarities and differences between maps generated by these models trained on a given dataset with different data augmentation strategies. Experiments made using this methodology suggest that the effects of these data augmentation techniques not only can be analyzed in this way but also allow us to identify different impact profiles over the trained models. 

**Abstract (ZH)**: 神经网络在图像分类任务中的应用由于其在该任务上的出色性能已越来越受欢迎。为了增强训练网络对输入图像变化的鲁棒性并避免过拟合，通常会在图像分类任务中采用数据增强策略。尽管数据增强是一种常用的技巧，但关于数据增强方法对复杂数据集上神经网络模型学习模式影响的研究还不够充分。本文的主要目标是提出一种方法和一套度量标准，以便定量分析数据增强对用于图像分类的卷积网络的影响。本文提出的方法的一个重要工具是用于这些模型的类激活图概念，它使我们能够识别和衡量模型在执行分类任务时赋予图像中每个像素的重要性。通过这些图，我们可以提取出在使用不同数据增强策略训练的数据集上生成的图之间的相似性和差异性的度量标准。实验结果表明，这些数据增强技术不仅可以通过这种方式进行分析，还可以帮助我们识别不同的影响特征。 

---
# AdaDim: Dimensionality Adaptation for SSL Representational Dynamics 

**Title (ZH)**: AdaDim: 维数自适应的SSL表示动力学 

**Authors**: Kiran Kokilepersaud, Mohit Prabhushankar, Ghassan AlRegib  

**Link**: [PDF](https://arxiv.org/pdf/2505.12576)  

**Abstract**: A key factor in effective Self-Supervised learning (SSL) is preventing dimensional collapse, which is where higher-dimensional representation spaces span a lower-dimensional subspace. Therefore, SSL optimization strategies involve guiding a model to produce representations ($R$) with a higher dimensionality. Dimensionality is either optimized through a dimension-contrastive approach that encourages feature decorrelation or through a sample-contrastive method that promotes a uniform spread of sample representations. Both families of SSL algorithms also utilize a projection head that maps $R$ into a lower-dimensional embedding space $Z$. Recent work has characterized the projection head as a filter of irrelevant features from the SSL objective by reducing mutual information, $I(R;Z)$. Therefore, the current literature's view is that a good SSL representation space should have a high $H(R)$ and a low $I(R;Z)$. However, this view of the problem is lacking in terms of an understanding of the underlying training dynamics that influences both terms, as well as how the values of $H(R)$ and $I(R;Z)$ arrived at the end of training reflect the downstream performance of an SSL model. We address both gaps in the literature by demonstrating that increases in $H(R)$ due to feature decorrelation at the start of training lead to a higher $I(R;Z)$, while increases in $H(R)$ due to samples distributing uniformly in a high-dimensional space at the end of training cause $I(R;Z)$ to plateau or decrease. Furthermore, our analysis shows that the best performing SSL models do not have the highest $H(R)$ nor the lowest $I(R;Z)$, but arrive at an optimal intermediate point for both. We develop a method called AdaDim to exploit these observed training dynamics by adaptively weighting between losses based on feature decorrelation and uniform sample spread. 

**Abstract (ZH)**: 一种有效的自监督学习（SSL）的关键因素是防止维度坍缩，即高维度表示空间降维到低维度子空间。因此，SSL优化策略涉及引导模型产生更高维度的表示($R$)。维度可以通过特征去相关的方式（维度对比）或通过样本分布均匀的方式（样本对比）进行优化。这两类SSL算法还利用投影头将$R$映射到低维度嵌入空间$Z$。最近的研究将投影头视为通过减少互信息$I(R;Z)$来筛选无关特征的SSL目标过滤器。因此，当前文献的观点是，一个好的SSL表示空间应该具有较高的$H(R)$和较低的$I(R;Z)$。然而，这种观点缺乏对影响这两个指标的潜在训练动态的理解，以及训练结束时$H(R)$和$I(R;Z)$的值如何反映SSL模型的下游性能。我们通过证明，在训练初期由于特征去相关导致的$H(R)$增加会提高$I(R;Z)$，而在训练末期由于样本在高维度空间中均匀分布导致的$H(R)$增加会使$I(R;Z)$趋于稳定或下降，来填补这些文献空白。此外，我们的分析表明，表现最佳的SSL模型并不具备最高的$H(R)$和最低的$I(R;Z)$，而是达到了两个指标的最优中间点。我们开发了一种名为AdaDim的方法，通过在基于特征去相关和均匀样本分布的损失之间自适应加权来利用这些观察到的训练动态。 

---
# Measuring Information Distortion in Hierarchical Ultra long Novel Generation:The Optimal Expansion Ratio 

**Title (ZH)**: 层次超长小说生成中信息失真的度量：最优扩展比例 

**Authors**: Hanwen Shen, Ting Ying  

**Link**: [PDF](https://arxiv.org/pdf/2505.12572)  

**Abstract**: Writing novels with Large Language Models (LLMs) raises a critical question: how much human-authored outline is necessary to generate high-quality million-word novels? While frameworks such as DOME, Plan&Write, and Long Writer have improved stylistic coherence and logical consistency, they primarily target shorter novels (10k--100k words), leaving ultra-long generation largely unexplored. Drawing on insights from recent text compression methods like LLMZip and LLM2Vec, we conduct an information-theoretic analysis that quantifies distortion occurring when LLMs compress and reconstruct ultra-long novels under varying compression-expansion ratios. We introduce a hierarchical two-stage generation pipeline (outline -> detailed outline -> manuscript) and find an optimal outline length that balances information preservation with human effort. Through extensive experimentation with Chinese novels, we establish that a two-stage hierarchical outline approach significantly reduces semantic distortion compared to single-stage methods. Our findings provide empirically-grounded guidance for authors and researchers collaborating with LLMs to create million-word novels. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）撰写小说提出一个关键问题：生成百万字高质量小说需要多少人类撰写的提纲？基于近年来的文本压缩方法如LLMZip和LLM2Vec，我们进行了信息论分析，量化了LLMs在不同压缩-扩展比下压缩和重构超长小说时产生的失真情况。我们引入了一种分层两阶段生成 pipeline（提纲 -> 详细提纲 -> 草稿），并找到了一个平衡信息保存与人力投入的最优提纲长度。通过大量使用中文小说进行实验，我们发现分层两阶段的提纲方法相比于单阶段方法显著减少了语义失真。我们的研究为作者和研究人员与LLMs合作创作百万字小说提供了实证指导。 

---
# A Survey of Attacks on Large Language Models 

**Title (ZH)**: 大型语言模型攻击综述 

**Authors**: Wenrui Xu, Keshab K. Parhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12567)  

**Abstract**: Large language models (LLMs) and LLM-based agents have been widely deployed in a wide range of applications in the real world, including healthcare diagnostics, financial analysis, customer support, robotics, and autonomous driving, expanding their powerful capability of understanding, reasoning, and generating natural languages. However, the wide deployment of LLM-based applications exposes critical security and reliability risks, such as the potential for malicious misuse, privacy leakage, and service disruption that weaken user trust and undermine societal safety. This paper provides a systematic overview of the details of adversarial attacks targeting both LLMs and LLM-based agents. These attacks are organized into three phases in LLMs: Training-Phase Attacks, Inference-Phase Attacks, and Availability & Integrity Attacks. For each phase, we analyze the details of representative and recently introduced attack methods along with their corresponding defenses. We hope our survey will provide a good tutorial and a comprehensive understanding of LLM security, especially for attacks on LLMs. We desire to raise attention to the risks inherent in widely deployed LLM-based applications and highlight the urgent need for robust mitigation strategies for evolving threats. 

**Abstract (ZH)**: 大型语言模型及其基于大型语言模型的代理在医疗诊断、金融分析、客户支持、机器人技术与自主驾驶等领域得到了广泛应用，扩展了其强大的自然语言理解、推理和生成能力。然而，基于大型语言模型的应用的广泛部署揭示了关键的安全性和可靠性风险，如恶意滥用、隐私泄露和服务中断，这些风险削弱了用户信任并损害了社会安全。本文提供了一个系统概述，详细介绍了针对大型语言模型及其基于大型语言模型的代理的 adversarial 攻击。这些攻击被组织为三个阶段：训练阶段攻击、推理阶段攻击和可用性和完整性攻击。对于每个阶段，我们分析了具有代表性和最近提出的各种攻击方法及其对应的防御措施。我们希望本次综述能提供一个良好的教程，并对大型语言模型安全有一个全面的理解，特别是针对大型语言模型的攻击。我们希望提高对广泛部署的基于大型语言模型应用所固有的风险的认识，并强调迫切需要为不断演变的威胁采取强大的缓解策略。 

---
# Beyond Accuracy: EcoL2 Metric for Sustainable Neural PDE Solvers 

**Title (ZH)**: 超越准确性：面向可持续性的EcoL2指标在神经PDE求解器中的应用 

**Authors**: Taniya Kapoor, Abhishek Chandra, Anastasios Stamou, Stephen J Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2505.12556)  

**Abstract**: Real-world systems, from aerospace to railway engineering, are modeled with partial differential equations (PDEs) describing the physics of the system. Estimating robust solutions for such problems is essential. Deep learning-based architectures, such as neural PDE solvers, have recently gained traction as a reliable solution method. The current state of development of these approaches, however, primarily focuses on improving accuracy. The environmental impact of excessive computation, leading to increased carbon emissions, has largely been overlooked. This paper introduces a carbon emission measure for a range of PDE solvers. Our proposed metric, EcoL2, balances model accuracy with emissions across data collection, model training, and deployment. Experiments across both physics-informed machine learning and operator learning architectures demonstrate that the proposed metric presents a holistic assessment of model performance and emission cost. As such solvers grow in scale and deployment, EcoL2 represents a step toward building performant scientific machine learning systems with lower long-term environmental impact. 

**Abstract (ZH)**: 一种PDE求解器的碳排放度量EcoL2及其环境影响评估 

---
# FreqSelect: Frequency-Aware fMRI-to-Image Reconstruction 

**Title (ZH)**: FreqSelect：频率 Awareness 的 fMRI-to-Image 重建 

**Authors**: Junliang Ye, Lei Wang, Md Zakir Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.12552)  

**Abstract**: Reconstructing natural images from functional magnetic resonance imaging (fMRI) data remains a core challenge in natural decoding due to the mismatch between the richness of visual stimuli and the noisy, low resolution nature of fMRI signals. While recent two-stage models, combining deep variational autoencoders (VAEs) with diffusion models, have advanced this task, they treat all spatial-frequency components of the input equally. This uniform treatment forces the model to extract meaning features and suppress irrelevant noise simultaneously, limiting its effectiveness. We introduce FreqSelect, a lightweight, adaptive module that selectively filters spatial-frequency bands before encoding. By dynamically emphasizing frequencies that are most predictive of brain activity and suppressing those that are uninformative, FreqSelect acts as a content-aware gate between image features and natural data. It integrates seamlessly into standard very deep VAE-diffusion pipelines and requires no additional supervision. Evaluated on the Natural Scenes dataset, FreqSelect consistently improves reconstruction quality across both low- and high-level metrics. Beyond performance gains, the learned frequency-selection patterns offer interpretable insights into how different visual frequencies are represented in the brain. Our method generalizes across subjects and scenes, and holds promise for extension to other neuroimaging modalities, offering a principled approach to enhancing both decoding accuracy and neuroscientific interpretability. 

**Abstract (ZH)**: 从功能性磁共振成像(fMRI)数据重构自然图像仍然是自然解码中的核心挑战，由于视觉刺激的丰富性和fMRI信号的噪声、低分辨率之间的不匹配。虽然近期结合深度变分自编码器(VAEs)与扩散模型的两阶段模型在此任务中取得了进展，但它们会平等处理输入的所有空间频率成分。这种统一处理要求模型同时提取有意义的特征并抑制无关噪声，从而限制了其效果。我们介绍了FreqSelect，一个轻量级且自适应的模块，在编码前选择性地滤除空间频率带。通过动态强调最能预测脑活动的频率并抑制无信息的频率，FreqSelect充当图像特征与自然数据之间的内容感知门控。它无缝集成到标准的深度VAE-扩散管道中，无需额外监督。在Natural Scenes数据集上评估表明，FreqSelect在低级和高级指标上均能一致提高重构质量。除了性能提升，学习到的频率选择模式提供了关于大脑中不同视觉频率表示的可解释洞见。我们的方法在不同受试者和场景间具有普适性，并有望扩展到其他神经影像模态，为提高解码准确性和神经科学可解释性提供了一种原则性方法。 

---
# ProMi: An Efficient Prototype-Mixture Baseline for Few-Shot Segmentation with Bounding-Box Annotations 

**Title (ZH)**: ProMi: 一种高效的原型混合基础模型用于带有边界框标注的少样本分割 

**Authors**: Florent Chiaroni, Ali Ayub, Ola Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2505.12547)  

**Abstract**: In robotics applications, few-shot segmentation is crucial because it allows robots to perform complex tasks with minimal training data, facilitating their adaptation to diverse, real-world environments. However, pixel-level annotations of even small amount of images is highly time-consuming and costly. In this paper, we present a novel few-shot binary segmentation method based on bounding-box annotations instead of pixel-level labels. We introduce, ProMi, an efficient prototype-mixture-based method that treats the background class as a mixture of distributions. Our approach is simple, training-free, and effective, accommodating coarse annotations with ease. Compared to existing baselines, ProMi achieves the best results across different datasets with significant gains, demonstrating its effectiveness. Furthermore, we present qualitative experiments tailored to real-world mobile robot tasks, demonstrating the applicability of our approach in such scenarios. Our code: this https URL. 

**Abstract (ZH)**: 在机器人应用中，少样本分割至关重要，因为它允许机器人使用最少的训练数据执行复杂的任务，促进其适应多样的现实环境。然而，即使对少量图像进行像素级注释也极为耗时且成本高昂。本文提出了一种基于边框注释的新型少样本二分类分割方法，而不是像素级标签。我们引入了ProMi方法，这是一种高效的原型混合基方法，将背景类别视为分布的混合。我们的方法简单、无需训练且有效，可以轻松容纳粗略注释。与现有基线相比，ProMi在不同数据集上取得了最佳结果，具有显著的性能提升，证明了其有效性。此外，我们还展示了针对真实场景移动机器人任务的定性实验，证明了该方法在这些场景中的适用性。代码：this https URL。 

---
# Exploring Sparsity for Parameter Efficient Fine Tuning Using Wavelets 

**Title (ZH)**: 探索稀疏性在参数高效微调中的应用——基于小波的方法 

**Authors**: Ahmet Bilican, M. Akın Yılmaz, A. Murat Tekalp, R. Gökberk Cinbiş  

**Link**: [PDF](https://arxiv.org/pdf/2505.12532)  

**Abstract**: Efficiently adapting large foundation models is critical, especially with tight compute and memory budgets. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA offer limited granularity and effectiveness in few-parameter regimes. We propose Wavelet Fine-Tuning (WaveFT), a novel PEFT method that learns highly sparse updates in the wavelet domain of residual matrices. WaveFT allows precise control of trainable parameters, offering fine-grained capacity adjustment and excelling with remarkably low parameter count, potentially far fewer than LoRA's minimum -- ideal for extreme parameter-efficient scenarios. In order to demonstrate the effect of the wavelet transform, we compare WaveFT with a special case, called SHiRA, that entails applying sparse updates directly in the weight domain. Evaluated on personalized text-to-image generation using Stable Diffusion XL as baseline, WaveFT significantly outperforms LoRA and other PEFT methods, especially at low parameter counts; achieving superior subject fidelity, prompt alignment, and image diversity. 

**Abstract (ZH)**: 高效适应大型基础模型至关重要，尤其是在紧缩的计算和内存预算下。在少量参数区域，参数高效微调（PEFT）方法如LoRA只能提供有限的细节和效果。我们提出了一种新型PEFT方法——小波微调（WaveFT），它在残差矩阵的小波域中学习高度稀疏的更新。WaveFT允许对可训练参数进行精确控制，提供精细的容量调整能力，并且在极低参数计数下表现出色，可能远少于LoRA的最低参数计数——这使其在极端参数高效场景下尤为理想。为了展示小波变换的效果，我们将WaveFT与一个特殊案例——称为SHiRA的方法进行比较，该方法直接在权重域中应用稀疏更新。在使用Stable Diffusion XL作为基准进行个性化文本到图像生成评估时，WaveFT在低参数计数下显著优于LoRA和其他PEFT方法；实现了更好的主题保真度、提示对齐和图像多样性。 

---
# Scalable Strategies for Continual Learning with Replay 

**Title (ZH)**: 可扩展的基于重放的持续学习策略 

**Authors**: Truman Hickok  

**Link**: [PDF](https://arxiv.org/pdf/2505.12512)  

**Abstract**: Future deep learning models will be distinguished by systems that perpetually learn through interaction, imagination, and cooperation, blurring the line between training and inference. This makes continual learning a critical challenge, as methods that efficiently maximize bidirectional transfer across learning trajectories will be essential. Replay is on track to play a foundational role in continual learning, allowing models to directly reconcile new information with past knowledge. In practice, however, replay is quite unscalable, doubling the cost of continual learning when applied naively. Moreover, the continual learning literature has not fully synchronized with the multi-task fine-tuning literature, having not fully integrated highly scalable techniques like model merging and low rank adaptation into a replay-enabled toolset that can produce a unified model in the face of many sequential tasks. In this paper, we begin by applying and analyzing low rank adaptation in a continual learning setting. Next, we introduce consolidation, a phasic approach to replay which leads to up to 55\% less replay samples being needed for a given performance target. Then, we propose sequential merging, an offshoot of task arithmetic which is tailored to the continual learning setting and is shown to work well in combination with replay. Finally, we demonstrate that the developed strategies can operate synergistically, resulting in a highly scalable toolset that outperforms standalone variants. 

**Abstract (ZH)**: 未来深度学习模型将通过交互、想象和合作实现持续学习，模糊训练和推理的界限，这使得持续学习成为一个关键挑战。有效最大化学习轨迹中双向迁移的方法将至关重要。回忆将在持续学习中发挥基础性作用，允许模型直接将新信息与以往知识相协调。然而，实际应用中，回忆的扩展性较差，未经优化的应用会使持续学习的成本翻倍。此外，持续学习文献尚未完全与多任务微调文献同步，未能完全将模型合并和低秩适应等高度可扩展技术整合到一个可处理多个顺序任务的回放缓冲工具集中。在本文中，我们首先在持续学习环境中应用和分析低秩适应，接着引入巩固策略，这是一种分阶段的回忆方法，能够减少55%的回忆样本以达到相同的性能目标。然后，我们提出序列合并策略，这是一种针对持续学习环境定制的任务算术分支，并证明其与回忆结合使用时效果显著。最后，我们证明所开发的策略可以协同工作，形成一种高度可扩展的工具集，性能优于单独使用的版本。 

---
# Towards Budget-Friendly Model-Agnostic Explanation Generation for Large Language Models 

**Title (ZH)**: 面向大型语言模型的预算友好型模型无关解释生成 

**Authors**: Junhao Liu, Haonan Yu, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12509)  

**Abstract**: With Large language models (LLMs) becoming increasingly prevalent in various applications, the need for interpreting their predictions has become a critical challenge. As LLMs vary in architecture and some are closed-sourced, model-agnostic techniques show great promise without requiring access to the model's internal parameters. However, existing model-agnostic techniques need to invoke LLMs many times to gain sufficient samples for generating faithful explanations, which leads to high economic costs. In this paper, we show that it is practical to generate faithful explanations for large-scale LLMs by sampling from some budget-friendly models through a series of empirical studies. Moreover, we show that such proxy explanations also perform well on downstream tasks. Our analysis provides a new paradigm of model-agnostic explanation methods for LLMs, by including information from budget-friendly models. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各种应用中的日益普及，解释其预测结果的需要已成为一个关键挑战。由于LLMs在架构上存在差异且部分为闭源，因此通用的解释技术无需访问模型内部参数便展现出巨大潜力。然而，现有的通用解释技术需要多次调用LLMs以获得足够的样本来生成可靠的解释，这导致了高昂的经济成本。在本文中，我们通过一系列实证研究展示了通过从一些低成本模型中采样来为大规模LLMs生成可靠解释是可行的。此外，我们展示了此类代理解释在下游任务中也表现良好。我们的分析为LLMs提供了一种新的通用解释方法范式，通过纳入低成本模型的信息。 

---
# Unsupervised Invariant Risk Minimization 

**Title (ZH)**: 无监督不变风险最小化 

**Authors**: Yotam Norman, Ron Meir  

**Link**: [PDF](https://arxiv.org/pdf/2505.12506)  

**Abstract**: We propose a novel unsupervised framework for \emph{Invariant Risk Minimization} (IRM), extending the concept of invariance to settings where labels are unavailable. Traditional IRM methods rely on labeled data to learn representations that are robust to distributional shifts across environments. In contrast, our approach redefines invariance through feature distribution alignment, enabling robust representation learning from unlabeled data. We introduce two methods within this framework: Principal Invariant Component Analysis (PICA), a linear method that extracts invariant directions under Gaussian assumptions, and Variational Invariant Autoencoder (VIAE), a deep generative model that disentangles environment-invariant and environment-dependent latent factors. Our approach is based on a novel ``unsupervised'' structural causal model and supports environment-conditioned sample-generation and intervention. Empirical evaluations on synthetic dataset and modified versions of MNIST demonstrate the effectiveness of our methods in capturing invariant structure, preserving relevant information, and generalizing across environments without access to labels. 

**Abstract (ZH)**: 我们提出了一种新的无监督框架进行不变风险最小化（IRM），将不变性的概念扩展到标签不可用的情境中。传统的方法依赖标记数据来学习在分布转换时稳健的表示。相比之下，我们的方法通过特征分布对齐重新定义不变性，从而能够从未标记数据中学习稳健的表示。我们在此框架中引入了两种方法：主不变成分分析（PICA），一种在线性假设下提取不变方向的线性方法，和变分不变自编码器（VIAE），一种解卷环境不变和环境相关潜在因子的深度生成模型。该方法基于一种新的“无监督”结构因果模型，并支持环境条件下的样本生成和干预。实证研究表明，我们的方法在合成数据集和MNIST的修改版本上能够捕获不变结构、保留相关信息并在无标签情况下跨环境通用有效。 

---
# CPGD: Toward Stable Rule-based Reinforcement Learning for Language Models 

**Title (ZH)**: CPGD：面向语言模型的稳定基于规则的强化学习 

**Authors**: Zongkai Liu, Fanqing Meng, Lingxiao Du, Zhixiang Zhou, Chao Yu, Wenqi Shao, Qiaosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12504)  

**Abstract**: Recent advances in rule-based reinforcement learning (RL) have significantly improved the reasoning capability of language models (LMs) with rule-based rewards. However, existing RL methods -- such as GRPO, REINFORCE++, and RLOO -- often suffer from training instability, where large policy updates and improper clipping can lead to training collapse. To address this issue, we propose Clipped Policy Gradient Optimization with Policy Drift (CPGD), a novel algorithm designed to stabilize policy learning in LMs. CPGD introduces a policy drift constraint based on KL divergence to dynamically regularize policy updates, and leverages a clip mechanism on the logarithm of the ratio to prevent excessive policy updates. We provide theoretical justification for CPGD and demonstrate through empirical analysis that it mitigates the instability observed in prior approaches. Furthermore, we show that CPGD significantly improves performance while maintaining training stability. Our implementation balances theoretical rigor with practical usability, offering a robust alternative for RL in the post-training of LMs. We release our code at this https URL. 

**Abstract (ZH)**: 基于规则的强化学习 Recent进展及其在语言模型中的规则学习稳定性优化（Clipped Policy Gradient Optimization with Policy Drift (CPGD)） 

---
# Unleashing Automated Congestion Control Customization in the Wild 

**Title (ZH)**: 在实际环境中释放自动化拥塞控制的定制潜力 

**Authors**: Amit Cohen, Lev Gloukhenki, Ravid Hadar, Eden Itah, Yehuda Shvut, Michael Schapira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12492)  

**Abstract**: Congestion control (CC) crucially impacts user experience across Internet services like streaming, gaming, AR/VR, and connected cars. Traditionally, CC algorithm design seeks universal control rules that yield high performance across diverse application domains and networks. However, varying service needs and network conditions challenge this approach. We share operational experience with a system that automatically customizes congestion control logic to service needs and network conditions. We discuss design, deployment challenges, and solutions, highlighting performance benefits through case studies in streaming, gaming, connected cars, and more.
Our system leverages PCC Vivace, an online-learning based congestion control protocol developed by researchers. Hence, along with insights from customizing congestion control, we also discuss lessons learned and modifications made to adapt PCC Vivace for real-world deployment. 

**Abstract (ZH)**: congestion控制（CC）对流媒体、 Gaming、AR/VR和联网汽车等互联网服务中的用户体验至关重要。传统的CC算法设计寻求适用于多种应用领域和网络环境的通用控制规则。然而，不同的服务需求和网络条件挑战着这一方法。我们分享了一个能够自动根据服务需求和网络条件定制 congestion控制逻辑的系统。通过在流媒体、 Gaming、联网汽车等多个领域的案例研究，我们讨论了设计、部署挑战及其解决方案，展示了性能优势。我们的系统利用了研究人员开发的基于在线学习的 congestion控制协议PCC Vivace。因此，除了定制 congestion控制的经验外，我们还讨论了将PCC Vivace适应实际部署所需的学习和修改。 

---
# Video-GPT via Next Clip Diffusion 

**Title (ZH)**: Video-GPT 通过下一帧扩散 

**Authors**: Shaobin Zhuang, Zhipeng Huang, Ying Zhang, Fangyikang Wang, Canmiao Fu, Binxin Yang, Chong Sun, Chen Li, Yali Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12489)  

**Abstract**: GPT has shown its remarkable success in natural language processing. However, the language sequence is not sufficient to describe spatial-temporal details in the visual world. Alternatively, the video sequence is good at capturing such details. Motivated by this fact, we propose a concise Video-GPT in this paper by treating video as new language for visual world modeling. By analogy to next token prediction in GPT, we introduce a novel next clip diffusion paradigm for pretraining Video-GPT. Different from the previous works, this distinct paradigm allows Video-GPT to tackle both short-term generation and long-term prediction, by autoregressively denoising the noisy clip according to the clean clips in the history. Extensive experiments show our Video-GPT achieves the state-of-the-art performance on video prediction, which is the key factor towards world modeling (Physics-IQ Benchmark: Video-GPT 34.97 vs. Kling 23.64 vs. Wan 20.89). Moreover, it can be well adapted on 6 mainstream video tasks in both video generation and understanding, showing its great generalization capacity in downstream. The project page is at this https URL. 

**Abstract (ZH)**: GPT在自然语言处理中展现了显著的成功，然而语言序列不足以描述视觉世界中的空间-时间细节。相比之下，视频序列能够捕捉这些细节。受此启发，本文提出了一个简化的Video-GPT，将视频视为视觉世界建模的新语言。类比于GPT的下一个token预测，我们引入了一种新的下一个片段扩散预训练范式。与以往工作不同，该独特范式使Video-GPT能够同时处理短期生成和长期预测，通过自回归地去噪历史干净片段中的噪声片段。实验结果表明，Video-GPT在视频预测任务上达到了最先进的性能（Physics-IQ基准：Video-GPT 34.97 vs. Kling 23.64 vs. Wan 20.89），并且在视频生成和理解的6个主流任务中表现出色，展示了其强大的泛化能力。项目页面见此链接：https://github.com/your-repository-name。 

---
# Joint Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self Supervised Learning 

**Title (ZH)**: 联合嵌入 vs 重构：潜在空间预测在自主监督学习中的可证明优势 

**Authors**: Hugues Van Assel, Mark Ibrahim, Tommaso Biancalani, Aviv Regev, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2505.12477)  

**Abstract**: Reconstruction and joint embedding have emerged as two leading paradigms in Self Supervised Learning (SSL). Reconstruction methods focus on recovering the original sample from a different view in input space. On the other hand, joint embedding methods align the representations of different views in latent space. Both approaches offer compelling advantages, yet practitioners lack clear guidelines for choosing between them. In this work, we unveil the core mechanisms that distinguish each paradigm. By leveraging closed form solutions for both approaches, we precisely characterize how the view generation process, e.g. data augmentation, impacts the learned representations. We then demonstrate that, unlike supervised learning, both SSL paradigms require a minimal alignment between augmentations and irrelevant features to achieve asymptotic optimality with increasing sample size. Our findings indicate that in scenarios where these irrelevant features have a large magnitude, joint embedding methods are preferable because they impose a strictly weaker alignment condition compared to reconstruction based methods. These results not only clarify the trade offs between the two paradigms but also substantiate the empirical success of joint embedding approaches on real world challenging datasets. 

**Abstract (ZH)**: 重建和联合嵌入已 emerged 作为自监督学习 (SSL) 中的两大主流范式。重建方法专注于从输入空间的不同视角恢复原始样本。另一方面，联合嵌入方法在潜在空间对不同视角的表示进行对齐。这两种方法分别提供了令人信服的优点，但实践者缺乏清晰的指南来选择其中之一。在本文中，我们揭示了区分每种范式的核心机制。通过利用两种方法的闭式解，我们精确地描述了视图生成过程，例如数据增强，是如何影响所学习表示的。然后我们证明，与监督学习不同，这两种 SSL 范式在样本量增加时仅需轻微的增强与无关特征的对齐即可达到渐近最优性。我们的研究发现，在这些无关特征具有较大幅度的情况下，联合嵌入方法更优，因为它们施加的对齐条件相较于基于重建的方法来说更弱。这些结果不仅澄清了两种范式之间的权衡，还证实了联合嵌入方法在实际挑战性数据集上取得的实证成功。 

---
# Enhancing Large Language Models with Reward-guided Tree Search for Knowledge Graph Question and Answering 

**Title (ZH)**: 基于奖励引导树搜索的知识图谱问答的大语言模型增强 

**Authors**: Xiao Long, Liansheng Zhuang, Chen Shen, Shaotian Yan, Yifei Li, Shafei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12476)  

**Abstract**: Recently, large language models (LLMs) have demonstrated impressive performance in Knowledge Graph Question Answering (KGQA) tasks, which aim to find answers based on knowledge graphs (KGs) for natural language questions. Existing LLMs-based KGQA methods typically follow the Graph Retrieval-Augmented Generation (GraphRAG) paradigm, which first retrieves reasoning paths from the large KGs, and then generates the answers based on them. However, these methods emphasize the exploration of new optimal reasoning paths in KGs while ignoring the exploitation of historical reasoning paths, which may lead to sub-optimal reasoning paths. Additionally, the complex semantics contained in questions may lead to the retrieval of inaccurate reasoning paths. To address these issues, this paper proposes a novel and training-free framework for KGQA tasks called Reward-guided Tree Search on Graph (RTSoG). RTSoG decomposes an original question into a series of simpler and well-defined sub-questions to handle the complex semantics. Then, a Self-Critic Monte Carlo Tree Search (SC-MCTS) guided by a reward model is introduced to iteratively retrieve weighted reasoning paths as contextual knowledge. Finally, it stacks the weighted reasoning paths according to their weights to generate the final answers. Extensive experiments on four datasets demonstrate the effectiveness of RTSoG. Notably, it achieves 8.7\% and 7.0\% performance improvement over the state-of-the-art method on the GrailQA and the WebQSP respectively. 

**Abstract (ZH)**: Reward-guided Tree Search on Graph for Knowledge Graph Question Answering 

---
# Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems 

**Title (ZH)**: 超越框架：多代理系统中协作策略的解构 

**Authors**: Haochun Wang, Sendong Zhao, Jingbo Wang, Zewen Qiang, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12467)  

**Abstract**: Multi-agent collaboration has emerged as a pivotal paradigm for addressing complex, distributed tasks in large language model (LLM)-driven applications. While prior research has focused on high-level architectural frameworks, the granular mechanisms governing agents, critical to performance and scalability, remain underexplored. This study systematically investigates four dimensions of collaboration strategies: (1) agent governance, (2) participation control, (3) interaction dynamics, and (4) dialogue history management. Through rigorous experimentation under two context-dependent scenarios: Distributed Evidence Integration (DEI) and Structured Evidence Synthesis (SES), we quantify the impact of these strategies on both task accuracy and computational efficiency. Our findings reveal that centralized governance, instructor-led participation, ordered interaction patterns, and instructor-curated context summarization collectively optimize the trade-off between decision quality and resource utilization with the support of the proposed Token-Accuracy Ratio (TAR). This work establishes a foundation for designing adaptive, scalable multi-agent systems, shifting the focus from structural novelty to strategic interaction mechanics. 

**Abstract (ZH)**: 多agents协作已成为解决大型语言模型（LLM）驱动应用中复杂分布式任务的关键范式。尽管前期研究集中于高层架构框架，但涉及agents的具体机制，尤其是对性能和可扩展性至关重要的方面，仍较少探讨。本研究系统性地探讨了协作策略的四个维度：（1）agents治理，（2）参与控制，（3）互动动力学，以及（4）对话历史管理。通过在两种上下文依赖场景下的严格实验：分布式证据整合（DEI）和结构化证据综合（SES），量化这些策略对任务准确性和计算效率的影响。研究发现，集中化治理、指导员主导的参与、有序的互动模式以及指导员精炼的上下文总结共同优化了决策质量和资源利用之间的权衡，并得到所提出的时间-准确率比（TAR）的支持。本研究奠定了设计适应性强、可扩展的多agents系统的基础，将研究重点从结构新颖性转向策略性交互机制。 

---
# IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems 

**Title (ZH)**: 基于LLM的多agent系统中的IP泄露攻击 

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2505.12442)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses. 

**Abstract (ZH)**: 大规模语言模型的快速进步导致了多代理系统（MAS）的出现，通过合作执行复杂任务。然而，MAS的复杂性，包括其架构和代理间的交互，引发了重要的知识产权（IP）保护问题。本文提出MASLEAK，一种新颖的攻击框架，旨在从MAS应用程序中提取敏感信息。MASLEAK针对一种实际的黑盒环境，在这种环境中，攻击者对MAS的架构或代理配置没有任何先验知识。攻击者只能通过公共API与MAS进行交互，提交攻击查询$q$并观察最终代理的输出。受计算机蠕虫传播和感染易受攻击网络主机的启发，MASLEAK精心构建了对抗性查询$q$，以从每个MAS代理中引发、传播并保留揭示完整产权组件（如代理数量、系统拓扑结构、系统提示、任务指令和工具使用情况）的响应。我们构建了包含810个应用的首个MAS应用程序合成数据集，并在Coze和CrewAI等实际MAS应用程序上评估了MASLEAK。MASLEAK在提取MAS IP方面显示出高精度，系统提示和任务指令的平均攻击成功率分别为87%，系统架构在大多数情况下为92%。最后，我们讨论了这些发现的意义以及潜在的防御措施。 

---
# Addressing the Scarcity of Benchmarks for Graph XAI 

**Title (ZH)**: 解决图解释abilitybenchmark稀缺性问题 

**Authors**: Michele Fontanesi, Alessio Micheli, Marco Podda, Domenico Tortorella  

**Link**: [PDF](https://arxiv.org/pdf/2505.12437)  

**Abstract**: While Graph Neural Networks (GNNs) have become the de facto model for learning from structured data, their decisional process remains opaque to the end user, undermining their deployment in safety-critical applications. In the case of graph classification, Explainable Artificial Intelligence (XAI) techniques address this major issue by identifying sub-graph motifs that explain predictions. However, advancements in this field are hindered by a chronic scarcity of benchmark datasets with known ground-truth motifs to assess the explanations' quality. Current graph XAI benchmarks are limited to synthetic data or a handful of real-world tasks hand-curated by domain experts. In this paper, we propose a general method to automate the construction of XAI benchmarks for graph classification from real-world datasets. We provide both 15 ready-made benchmarks, as well as the code to generate more than 2000 additional XAI benchmarks with our method. As a use case, we employ our benchmarks to assess the effectiveness of some popular graph explainers. 

**Abstract (ZH)**: 尽管图神经网络（GNNs）已成为结构化数据学习的事实标准模型，其决策过程对最终用户仍具有 opacity，这阻碍了其在关键安全应用中的部署。在图分类的情况下，可解释人工智能（XAI）技术通过识别解释预测的子图模式来解决这一主要问题。然而，这一领域的进展受限于缺乏包含已知ground-truth模式的基准数据集，以评估解释的质量。当前的图XAI基准主要局限于合成数据或由领域专家手工编curated的少量真实世界任务。在本文中，我们提出了一种通用方法，用于从真实世界数据集自动生成图分类的XAI基准。我们提供了15个现成的基准，并提供了代码以通过我们的方法生成超过2000个额外的XAI基准。作为用例，我们使用这些基准来评估一些流行的图解释器的有效性。 

---
# SGDPO: Self-Guided Direct Preference Optimization for Language Model Alignment 

**Title (ZH)**: SGDPO：自我引导的直接偏好优化以实现语言模型对齐 

**Authors**: Wenqiao Zhu, Ji Liu, Lulu Wang, Jun Wu, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12435)  

**Abstract**: Direct Preference Optimization (DPO) is broadly utilized for aligning Large Language Models (LLMs) with human values because of its flexibility. Despite its effectiveness, it has been observed that the capability of DPO to generate human-preferred response is limited and the results of DPO are far from resilient. To address these limitations, in this paper we propose a novel Self-Guided Direct Preference Optimization algorithm, i.e., SGDPO, which incorporates a pilot term to steer the gradient flow during the optimization process, allowing for fine-grained control over the updates of chosen and rejected rewards. We provide a detailed theoretical analysis of our proposed method and elucidate its operational mechanism. Furthermore, we conduct comprehensive experiments on various models and benchmarks. The extensive experimental results demonstrate the consistency between the empirical results and our theoretical analysis and confirm the effectiveness of our proposed approach (up to 9.19% higher score). 

**Abstract (ZH)**: 直接偏好优化（DPO）因其灵活性被广泛用于对齐大规模语言模型（LLMs）与人类价值观，尽管其有效性已经得到验证，但观察到DPO生成人类偏好的响应能力有限，其结果也远远不够稳健。为解决这些局限性，本文提出了一个新颖的自引导直接偏好优化算法，即SGDPO，该算法引入了一个引导项，以在优化过程中引导梯度流动，从而实现对选定和拒绝奖励更新的精细控制。我们详细分析了所提出方法的理论基础，并解释了其工作机制。此外，我们在多个模型和基准上进行了全面的实验。广泛的实验结果表明了实证结果与我们的理论分析之间的一致性，并证实了我们所提出方法的有效性（最高可提高9.19%的得分）。 

---
# SRLoRA: Subspace Recomposition in Low-Rank Adaptation via Importance-Based Fusion and Reinitialization 

**Title (ZH)**: SRLoRA：基于重要性融合与重新初始化的低秩重构子空间适配 

**Authors**: Haodong Yang, Lei Wang, Md Zakir Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.12433)  

**Abstract**: Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning (PEFT) method that injects two trainable low-rank matrices (A and B) into frozen pretrained models. While efficient, LoRA constrains updates to a fixed low-rank subspace (Delta W = BA), which can limit representational capacity and hinder downstream performance. We introduce Subspace Recomposition in Low-Rank Adaptation (SRLoRA) via importance-based fusion and reinitialization, a novel approach that enhances LoRA's expressiveness without compromising its lightweight structure. SRLoRA assigns importance scores to each LoRA pair (a column of B and the corresponding row of A), and dynamically recomposes the subspace during training. Less important pairs are fused into the frozen backbone, freeing capacity to reinitialize new pairs along unused principal directions derived from the pretrained weight's singular value decomposition. This mechanism enables continual subspace refreshment and richer adaptation over time, without increasing the number of trainable parameters. We evaluate SRLoRA on both language and vision tasks, including the GLUE benchmark and various image classification datasets. SRLoRA consistently achieves faster convergence and improved accuracy over standard LoRA, demonstrating its generality, efficiency, and potential for broader PEFT applications. 

**Abstract (ZH)**: 基于重要性融合与重构的低秩适应（SRLoRA）：一种提升低秩调整表达能力的方法 

---
# Observe-R1: Unlocking Reasoning Abilities of MLLMs with Dynamic Progressive Reinforcement Learning 

**Title (ZH)**: Observe-R1: 通过动态渐进强化学习解锁MLLMs的推理能力 

**Authors**: Zirun Guo, Minjie Hong, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12432)  

**Abstract**: Reinforcement Learning (RL) has shown promise in improving the reasoning abilities of Large Language Models (LLMs). However, the specific challenges of adapting RL to multimodal data and formats remain relatively unexplored. In this work, we present Observe-R1, a novel framework aimed at enhancing the reasoning capabilities of multimodal large language models (MLLMs). We draw inspirations from human learning progression--from simple to complex and easy to difficult, and propose a gradual learning paradigm for MLLMs. To this end, we construct the NeuraLadder dataset, which is organized and sampled according to the difficulty and complexity of data samples for RL training. To tackle multimodal tasks, we introduce a multimodal format constraint that encourages careful observation of images, resulting in enhanced visual abilities and clearer and more structured responses. Additionally, we implement a bonus reward system that favors concise, correct answers within a length constraint, alongside a dynamic weighting mechanism that prioritizes uncertain and medium-difficulty problems, ensuring that more informative samples have a greater impact on training. Our experiments with the Qwen2.5-VL-3B and Qwen2.5-VL-7B models on 20k samples from the NeuraLadder dataset show that Observe-R1 outperforms a series of larger reasoning models on both reasoning and general benchmarks, achieving superior clarity and conciseness in reasoning chains. Ablation studies validate the effectiveness of our strategies, highlighting the robustness and generalization of our approach. The dataset and code will be released at this https URL. 

**Abstract (ZH)**: 强化学习（RL）在提升大型语言模型（LLMs）的推理能力方面展现了潜力。然而，将RL适应多模态数据和格式的具体挑战仍相对未被充分探索。在本文中，我们提出Observe-R1，一种旨在增强多模态大型语言模型（MLLMs）推理能力的新框架。我们借鉴了人类学习进步的模式——从简单到复杂，从易到难，并为MLLMs提出了一个逐步学习范式。为此，我们构建了NeuraLadder数据集，该数据集根据数据样本的难易程度和复杂性进行组织和采样，用于RL训练。为应对多模态任务，我们引入了一种多模态格式约束，促成了对图像的仔细观察，提高了视觉能力并产生了清晰且结构化的回答。此外，我们实施了一种奖励系统，倾向于在长度限制内提供简洁且正确的答案，并采用动态加权机制优先处理不确定性和中等问题，确保更具信息量的样本对训练有更大的影响。我们在NeuraLadder数据集的20,000个样本上对Qwen2.5-VL-3B和Qwen2.5-VL-7B模型进行的实验表明，Observe-R1在推理和通用基准测试中均优于一系列更大规模的推理模型，实现了更清晰且简洁的推理链路。消融研究验证了我们策略的有效性，突显了我们方法的稳健性和泛化能力。数据集和代码将在以下链接发布：此 https URL。 

---
# EvoGPT: Enhancing Test Suite Robustness via LLM-Based Generation and Genetic Optimization 

**Title (ZH)**: EvoGPT：基于LLM生成和遗传优化提升测试套件 robustness 

**Authors**: Lior Broide, Roni Stern  

**Link**: [PDF](https://arxiv.org/pdf/2505.12424)  

**Abstract**: Large Language Models (LLMs) have recently emerged as promising tools for automated unit test generation. We introduce a hybrid framework called EvoGPT that integrates LLM-based test generation with evolutionary search techniques to create diverse, fault-revealing unit tests. Unit tests are initially generated with diverse temperature sampling to maximize behavioral and test suite diversity, followed by a generation-repair loop and coverage-guided assertion enhancement. The resulting test suites are evolved using genetic algorithms, guided by a fitness function prioritizing mutation score over traditional coverage metrics. This design emphasizes the primary objective of unit testing-fault detection. Evaluated on multiple open-source Java projects, EvoGPT achieves an average improvement of 10% in both code coverage and mutation score compared to LLMs and traditional search-based software testing baselines. These results demonstrate that combining LLM-driven diversity, targeted repair, and evolutionary optimization produces more effective and resilient test suites. 

**Abstract (ZH)**: 大型语言模型（LLMs） recently emerged as promising tools for automated unit test generation. We introduce a hybrid framework called EvoGPT that integrates LLM-based test generation with evolutionary search techniques to create diverse, fault-revealing unit tests. 

---
# PSC: Extending Context Window of Large Language Models via Phase Shift Calibration 

**Title (ZH)**: PSC：通过相位移校准扩展大型语言模型的上下文窗口 

**Authors**: Wenqiao Zhu, Chao Xu, Lulu Wang, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12423)  

**Abstract**: Rotary Position Embedding (RoPE) is an efficient position encoding approach and is widely utilized in numerous large language models (LLMs). Recently, a lot of methods have been put forward to further expand the context window based on RoPE. The core concept of those methods is to predefine or search for a set of factors to rescale the base frequencies of RoPE. Nevertheless, it is quite a challenge for existing methods to predefine an optimal factor due to the exponential search space. In view of this, we introduce PSC (Phase Shift Calibration), a small module for calibrating the frequencies predefined by existing methods. With the employment of PSC, we demonstrate that many existing methods can be further enhanced, like PI, YaRN, and LongRoPE. We conducted extensive experiments across multiple models and tasks. The results demonstrate that (1) when PSC is enabled, the comparative reductions in perplexity increase as the context window size is varied from 16k, to 32k, and up to 64k. (2) Our approach is broadly applicable and exhibits robustness across a variety of models and tasks. The code can be found at this https URL. 

**Abstract (ZH)**: 基于RoPE的相位-shift校准（PSC）：一种增强位置编码的小模块 

---
# Fixed Point Explainability 

**Title (ZH)**: 固定点可解释性 

**Authors**: Emanuele La Malfa, Jon Vadillo, Marco Molinari, Michael Wooldridge  

**Link**: [PDF](https://arxiv.org/pdf/2505.12421)  

**Abstract**: This paper introduces a formal notion of fixed point explanations, inspired by the "why regress" principle, to assess, through recursive applications, the stability of the interplay between a model and its explainer. Fixed point explanations satisfy properties like minimality, stability, and faithfulness, revealing hidden model behaviours and explanatory weaknesses. We define convergence conditions for several classes of explainers, from feature-based to mechanistic tools like Sparse AutoEncoders, and we report quantitative and qualitative results. 

**Abstract (ZH)**: 本文引入了一种由"Why Regress"原则启发的形式化固定点解释概念，通过递归应用来评估模型与其解释器之间互动的稳定性。固定点解释满足最小性、稳定性和忠实性等属性，揭示了隐藏的模型行为和解释的弱点。我们定义了不同类型解释工具的收敛条件，从基于特征的方法到机制工具如稀疏自编码器，并报告了定量和定性结果。 

---
# Mutual Evidential Deep Learning for Medical Image Segmentation 

**Title (ZH)**: 医用证据深学习的互信息医学图像分割 

**Authors**: Yuanpeng He, Yali Bi, Lijian Li, Chi-Man Pun, Wenpin Jiao, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12418)  

**Abstract**: Existing semi-supervised medical segmentation co-learning frameworks have realized that model performance can be diminished by the biases in model recognition caused by low-quality pseudo-labels. Due to the averaging nature of their pseudo-label integration strategy, they fail to explore the reliability of pseudo-labels from different sources. In this paper, we propose a mutual evidential deep learning (MEDL) framework that offers a potentially viable solution for pseudo-label generation in semi-supervised learning from two perspectives. First, we introduce networks with different architectures to generate complementary evidence for unlabeled samples and adopt an improved class-aware evidential fusion to guide the confident synthesis of evidential predictions sourced from diverse architectural networks. Second, utilizing the uncertainty in the fused evidence, we design an asymptotic Fisher information-based evidential learning strategy. This strategy enables the model to initially focus on unlabeled samples with more reliable pseudo-labels, gradually shifting attention to samples with lower-quality pseudo-labels while avoiding over-penalization of mislabeled classes in high data uncertainty samples. Additionally, for labeled data, we continue to adopt an uncertainty-driven asymptotic learning strategy, gradually guiding the model to focus on challenging voxels. Extensive experiments on five mainstream datasets have demonstrated that MEDL achieves state-of-the-art performance. 

**Abstract (ZH)**: 现有的半监督医学分割共学习框架已经意识到模型性能可能会因低质量伪标签引起的模型识别偏差而下降。由于其伪标签整合策略的平均性质，这些框架无法探索不同来源伪标签的可靠性。本文从两个角度提出了一种互证深度学习（MEDL）框架，为半监督学习中的伪标签生成提供了潜在的解决方案。首先，引入具有不同架构的网络为无标签样本生成互补证据，并采用改进的类意识证据融合策略指导来自不同架构网络的证据预测的自信合成。其次，利用融合证据中的不确定性，设计一种基于渐近费雪信息的证据学习策略。该策略使模型能够在数据不确定性高时首先专注于更具可靠性的伪标签样本，逐步将注意力转向伪标签质量较低的样本，同时避免在高数据不确定性样本中对错标类别进行过度惩罚。此外，对于标注数据，继续采用基于不确定性的渐近学习策略，逐步引导模型关注具有挑战性的体素。在五个主流数据集上的广泛实验表明，MEDL实现了最先进的性能。 

---
# Table-R1: Region-based Reinforcement Learning for Table Understanding 

**Title (ZH)**: 表-1：基于区域的强化学习表格理解 

**Authors**: Zhenhe Wu, Jian Yang, Jiaheng Liu, Xianjie Wu, Changzai Pan, Jie Zhang, Yu Zhao, Shuangyong Song, Yongxiang Li, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12415)  

**Abstract**: Tables present unique challenges for language models due to their structured row-column interactions, necessitating specialized approaches for effective comprehension. While large language models (LLMs) have demonstrated potential in table reasoning through prompting and techniques like chain-of-thought (CoT) and program-of-thought (PoT), optimizing their performance for table question answering remains underexplored. In this paper, we introduce region-based Table-R1, a novel reinforcement learning approach that enhances LLM table understanding by integrating region evidence into reasoning steps. Our method employs Region-Enhanced Supervised Fine-Tuning (RE-SFT) to guide models in identifying relevant table regions before generating answers, incorporating textual, symbolic, and program-based reasoning. Additionally, Table-Aware Group Relative Policy Optimization (TARPO) introduces a mixed reward system to dynamically balance region accuracy and answer correctness, with decaying region rewards and consistency penalties to align reasoning steps. Experiments show that Table-R1 achieves an average performance improvement of 14.36 points across multiple base models on three benchmark datasets, even outperforming baseline models with ten times the parameters, while TARPO reduces response token consumption by 67.5% compared to GRPO, significantly advancing LLM capabilities in efficient tabular reasoning. 

**Abstract (ZH)**: 语言模型在处理表格时面临独特的挑战，由于其结构化的行列交互，需要专门的方法来有效理解。虽然大型语言模型（LLMs）通过提示和链式思考（CoT）、程序思维（PoT）等技术在表格推理方面展现出了潜力，但优化其在表格问答方面的表现仍处于未充分探索的阶段。在本文中，我们引入了基于区域的Table-R1，这是一种新的强化学习方法，通过将区域证据整合到推理步骤中来提升LLM对表格的理解。该方法采用区域增强监督微调（RE-SFT）在生成答案之前引导模型识别相关的表格区域，结合文本、符号和程序推理。此外，表意识组相对策略优化（TARPO）引入了混合奖励系统以动态平衡区域准确性和答案正确性，并通过衰减区域奖励和一致性惩罚来对齐推理步骤。实验表明，Table-R1在三个基准数据集上的多个基模型上平均取得了14.36分的性能提升，即使在参数量为基线模型十分之一的情况下也超过了基线模型的表现，同时TARPO将响应令牌消耗减少了67.5%，显著提升了LLM在高效表格推理方面的能力。 

---
# ViEEG: Hierarchical Neural Coding with Cross-Modal Progressive Enhancement for EEG-Based Visual Decoding 

**Title (ZH)**: ViEEG：基于跨模态渐进增强的分级神经编码脑电视觉解码方法 

**Authors**: Minxu Liu, Donghai Guan, Chuhang Zheng, Chunwei Tian, Jie Wen, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12408)  

**Abstract**: Understanding and decoding brain activity into visual representations is a fundamental challenge at the intersection of neuroscience and artificial intelligence. While EEG-based visual decoding has shown promise due to its non-invasive, low-cost nature and millisecond-level temporal resolution, existing methods are limited by their reliance on flat neural representations that overlook the brain's inherent visual hierarchy. In this paper, we introduce ViEEG, a biologically inspired hierarchical EEG decoding framework that aligns with the Hubel-Wiesel theory of visual processing. ViEEG decomposes each visual stimulus into three biologically aligned components-contour, foreground object, and contextual scene-serving as anchors for a three-stream EEG encoder. These EEG features are progressively integrated via cross-attention routing, simulating cortical information flow from V1 to IT to the association cortex. We further adopt hierarchical contrastive learning to align EEG representations with CLIP embeddings, enabling zero-shot object recognition. Extensive experiments on the THINGS-EEG dataset demonstrate that ViEEG achieves state-of-the-art performance, with 40.9% Top-1 accuracy in subject-dependent and 22.9% Top-1 accuracy in cross-subject settings, surpassing existing methods by over 45%. Our framework not only advances the performance frontier but also sets a new paradigm for biologically grounded brain decoding in AI. 

**Abstract (ZH)**: 基于Hubel-Wiesel视觉处理理论的生物启发式分层EEG解码框架ViEEG 

---
# The power of text similarity in identifying AI-LLM paraphrased documents: The case of BBC news articles and ChatGPT 

**Title (ZH)**: 文本相似性在识别AI-LLM重述文档中的作用：以BBC新闻文章和ChatGPT为例 

**Authors**: Konstantinos Xylogiannopoulos, Petros Xanthopoulos, Panagiotis Karampelas, Georgios Bakamitsos  

**Link**: [PDF](https://arxiv.org/pdf/2505.12405)  

**Abstract**: Generative AI paraphrased text can be used for copyright infringement and the AI paraphrased content can deprive substantial revenue from original content creators. Despite this recent surge of malicious use of generative AI, there are few academic publications that research this threat. In this article, we demonstrate the ability of pattern-based similarity detection for AI paraphrased news recognition. We propose an algorithmic scheme, which is not limited to detect whether an article is an AI paraphrase, but, more importantly, to identify that the source of infringement is the ChatGPT. The proposed method is tested with a benchmark dataset specifically created for this task that incorporates real articles from BBC, incorporating a total of 2,224 articles across five different news categories, as well as 2,224 paraphrased articles created with ChatGPT. Results show that our pattern similarity-based method, that makes no use of deep learning, can detect ChatGPT assisted paraphrased articles at percentages 96.23% for accuracy, 96.25% for precision, 96.21% for sensitivity, 96.25% for specificity and 96.23% for F1 score. 

**Abstract (ZH)**: Generative AI生成的文本可能用于版权侵权，AI生成的内容可能剥夺原始内容创作者的大量收益。尽管近期滥用生成式AI的情况日益严重，但仍缺乏对此威胁的学术研究。在本文中，我们展示了基于模式相似性的检测能力，用于识别AI生成的新闻。我们提出了一种算法方案，不仅可以检测文章是否为AI生成的，而且更重要的是能够识别侵权源为ChatGPT。所提出的方法使用了一个针对此任务专门创建的基准数据集，该数据集包含来自BBC的真实文章，共计2,224篇，以及使用ChatGPT生成的2,224篇平行文本。结果显示，我们的基于模式相似性的方法（不使用深度学习）能够以96.23%的准确率、96.25%的精确率、96.21%的灵敏度、96.25%的特异度和96.23%的F1分数识别ChatGPT辅助生成的平行文本。 

---
# Hyperbolic Residual Quantization: Discrete Representations for Data with Latent Hierarchies 

**Title (ZH)**: 双曲残差量化：潜在层次数据的离散表示 

**Authors**: Piotr Piękos, Subhradeep Kayal, Alexandros Karatzoglou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12404)  

**Abstract**: Hierarchical data arise in countless domains, from biological taxonomies and organizational charts to legal codes and knowledge graphs. Residual Quantization (RQ) is widely used to generate discrete, multitoken representations for such data by iteratively quantizing residuals in a multilevel codebook. However, its reliance on Euclidean geometry can introduce fundamental mismatches that hinder modeling of hierarchical branching, necessary for faithful representation of hierarchical data. In this work, we propose Hyperbolic Residual Quantization (HRQ), which embeds data natively in a hyperbolic manifold and performs residual quantization using hyperbolic operations and distance metrics. By adapting the embedding network, residual computation, and distance metric to hyperbolic geometry, HRQ imparts an inductive bias that aligns naturally with hierarchical branching. We claim that HRQ in comparison to RQ can generate more useful for downstream tasks discrete hierarchical representations for data with latent hierarchies. We evaluate HRQ on two tasks: supervised hierarchy modeling using WordNet hypernym trees, where the model is supervised to learn the latent hierarchy - and hierarchy discovery, where, while latent hierarchy exists in the data, the model is not directly trained or evaluated on a task related to the hierarchy. Across both scenarios, HRQ hierarchical tokens yield better performance on downstream tasks compared to Euclidean RQ with gains of up to $20\%$ for the hierarchy modeling task. Our results demonstrate that integrating hyperbolic geometry into discrete representation learning substantially enhances the ability to capture latent hierarchies. 

**Abstract (ZH)**: Hyperbolic Residual Quantization for Hierarchical Data Representation 

---
# Traversal Verification for Speculative Tree Decoding 

**Title (ZH)**: 投机树解码的遍历验证 

**Authors**: Yepeng Weng, Qiao Hu, Xujie Chen, Li Liu, Dianwen Mei, Huishi Qiu, Jiang Tian, Zhongchao Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12398)  

**Abstract**: Speculative decoding is a promising approach for accelerating large language models. The primary idea is to use a lightweight draft model to speculate the output of the target model for multiple subsequent timesteps, and then verify them in parallel to determine whether the drafted tokens should be accepted or rejected. To enhance acceptance rates, existing frameworks typically construct token trees containing multiple candidates in each timestep. However, their reliance on token-level verification mechanisms introduces two critical limitations: First, the probability distribution of a sequence differs from that of individual tokens, leading to suboptimal acceptance length. Second, current verification schemes begin from the root node and proceed layer by layer in a top-down manner. Once a parent node is rejected, all its child nodes should be discarded, resulting in inefficient utilization of speculative candidates. This paper introduces Traversal Verification, a novel speculative decoding algorithm that fundamentally rethinks the verification paradigm through leaf-to-root traversal. Our approach considers the acceptance of the entire token sequence from the current node to the root, and preserves potentially valid subsequences that would be prematurely discarded by existing methods. We theoretically prove that the probability distribution obtained through Traversal Verification is identical to that of the target model, guaranteeing lossless inference while achieving substantial acceleration gains. Experimental results across different large language models and multiple tasks show that our method consistently improves acceptance length and throughput over existing methods 

**Abstract (ZH)**: speculative 解码是一种加速大型语言模型的有前途的方法。通过从叶到根遍历的验证方式，提出了一种新的 speculative 解码算法，该算法从根本上重构了验证范式。我们的方法从当前节点到根考虑整个令牌序列的接受性，并保留现有方法会过早抛弃的有效子序列。我们理论证明，通过从叶到根遍历获得的概率分布与目标模型相同，保证无损推断的同时实现显著的加速收益。在不同大型语言模型和多个任务上的实验结果表明，我们的方法在提高接受长度和吞吐量方面优于现有方法。 

---
# Few-Shot Concept Unlearning with Low Rank Adaptation 

**Title (ZH)**: 少量样本概念遗忘与低秩适应 

**Authors**: Udaya Shreyas, L.N. Aadarsh  

**Link**: [PDF](https://arxiv.org/pdf/2505.12395)  

**Abstract**: Image Generation models are a trending topic nowadays, with many people utilizing Artificial Intelligence models in order to generate images. There are many such models which, given a prompt of a text, will generate an image which depicts said prompt. There are many image generation models, such as Latent Diffusion Models, Denoising Diffusion Probabilistic Models, Generative Adversarial Networks and many more. When generating images, these models can generate sensitive image data, which can be threatening to privacy or may violate copyright laws of private entities. Machine unlearning aims at removing the influence of specific data subsets from the trained models and in the case of image generation models, remove the influence of a concept such that the model is unable to generate said images of the concept when prompted. Conventional retraining of the model can take upto days, hence fast algorithms are the need of the hour. In this paper we propose an algorithm that aims to remove the influence of concepts in diffusion models through updating the gradients of the final layers of the text encoders. Using a weighted loss function, we utilize backpropagation in order to update the weights of the final layers of the Text Encoder componet of the Stable Diffusion Model, removing influence of the concept from the text-image embedding space, such that when prompted, the result is an image not containing the concept. The weighted loss function makes use of Textual Inversion and Low-Rank this http URL perform our experiments on Latent Diffusion Models, namely the Stable Diffusion v2 model, with an average concept unlearning runtime of 50 seconds using 4-5 images. 

**Abstract (ZH)**: 基于梯度更新的扩散模型概念去影响快速算法研究：以文本编码终层权重更新实现概念去影响 

---
# Data Sharing with a Generative AI Competitor 

**Title (ZH)**: 与生成式AI竞争对手的数据共享 

**Authors**: Boaz Taitler, Omer Madmon, Moshe Tennenholtz, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2505.12386)  

**Abstract**: As GenAI platforms grow, their dependence on content from competing providers, combined with access to alternative data sources, creates new challenges for data-sharing decisions. In this paper, we provide a model of data sharing between a content creation firm and a GenAI platform that can also acquire content from third-party experts. The interaction is modeled as a Stackelberg game: the firm first decides how much of its proprietary dataset to share with GenAI, and GenAI subsequently determines how much additional data to acquire from external experts. Their utilities depend on user traffic, monetary transfers, and the cost of acquiring additional data from external experts. We characterize the unique subgame perfect equilibrium of the game and uncover a surprising phenomenon: The firm may be willing to pay GenAI to share the firm's own data, leading to a costly data-sharing equilibrium. We further characterize the set of Pareto improving data prices, and show that such improvements occur only when the firm pays to share data. Finally, we study how the price can be set to optimize different design objectives, such as promoting firm data sharing, expert data acquisition, or a balance of both. Our results shed light on the economic forces shaping data-sharing partnerships in the age of GenAI, and provide guidance for platforms, regulators and policymakers seeking to design effective data exchange mechanisms. 

**Abstract (ZH)**: 随着GenAI平台的发展，其对竞争内容提供商的内容依赖以及访问其他数据源的能力，为数据共享决策带来了新的挑战。本文提供了一个内容创作公司与可以获取第三方专家内容的GenAI平台之间的数据共享模型。该互动被建模为斯塔克尔贝格博弈：首先，公司决定向GenAI共享其多少专有数据集，然后GenAI确定从外部专家额外获取多少数据。他们的效用取决于用户流量、货币转移以及从外部专家获取额外数据的成本。我们刻画了该博弈的唯一子博弈完美纳什均衡，并揭示了一个令人惊讶的现象：公司可能愿意付费给GenAI以共享公司自身数据，导致一个成本较高的数据共享均衡。我们进一步刻画了帕累托改进的数据价格集，并表明这些改进只发生在公司支付以共享数据的情况下。最后，我们研究了如何设定价格以优化不同的设计目标，如促进公司数据共享、专家数据获取或两者兼顾。我们的研究揭示了在GenAI时代塑造数据共享伙伴关系的经济力量，并为平台、监管者和政策制定者设计有效的数据交换机制提供了指导。 

---
# From n-gram to Attention: How Model Architectures Learn and Propagate Bias in Language Modeling 

**Title (ZH)**: 从n-gram到注意力：模型架构在语言模型中学习和传播偏见的方式 

**Authors**: Mohsinul Kabir, Tasfia Tahsin, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12381)  

**Abstract**: Current research on bias in language models (LMs) predominantly focuses on data quality, with significantly less attention paid to model architecture and temporal influences of data. Even more critically, few studies systematically investigate the origins of bias. We propose a methodology grounded in comparative behavioral theory to interpret the complex interaction between training data and model architecture in bias propagation during language modeling. Building on recent work that relates transformers to n-gram LMs, we evaluate how data, model design choices, and temporal dynamics affect bias propagation. Our findings reveal that: (1) n-gram LMs are highly sensitive to context window size in bias propagation, while transformers demonstrate architectural robustness; (2) the temporal provenance of training data significantly affects bias; and (3) different model architectures respond differentially to controlled bias injection, with certain biases (e.g. sexual orientation) being disproportionately amplified. As language models become ubiquitous, our findings highlight the need for a holistic approach -- tracing bias to its origins across both data and model dimensions, not just symptoms, to mitigate harm. 

**Abstract (ZH)**: 当前关于语言模型中偏见的研究主要集中在数据质量上，对模型架构和数据的时间影响关注较少。更关键的是，很少有研究系统地探讨偏见的根源。我们提出了一种基于比较行为理论的方法，以解释训练数据与模型架构在语言建模过程中偏见传播中的复杂交互。基于最近将变换器与n-gram语言模型联系起来的工作，我们评估了数据、模型设计选择以及时间动态对偏见传播的影响。我们的发现表明：(1) n-gram语言模型在偏见传播中对上下文窗口大小非常敏感，而变换器表现出架构稳健性；(2) 训练数据的时间来源显著影响偏见；(3) 不同的模型架构对控制偏见注入的响应不同，某些偏见（例如性取向）被不成比例地放大。随着语言模型的普及，我们的发现突显了需要采用综合方法的重要性——从数据和模型两个维度追踪偏见的根源，而不仅是症状，以减轻其危害。 

---
# CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement 

**Title (ZH)**: CAPTURE: 基于上下文感知的提示注入测试与鲁棒性增强 

**Authors**: Gauri Kholkar, Ratinder Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2505.12368)  

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations. 

**Abstract (ZH)**: 提示注入仍然是大型语言模型的一项重大安全风险。然而，现有防护模型在上下文感知场景下的有效性仍然没有得到充分探索，因为它们常常依赖于静态攻击基准。此外，它们倾向于过度防御。我们引入了CAPTURE，这是一种新颖的上下文感知基准，评估攻击检测能力和过度防御倾向，仅使用少量领域的示例。我们的实验揭示了当前的提示注入防护模型在对抗情况下存在高误负和在良性情况下存在过度误报的问题，突显了其关键局限性。 

---
# DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization 

**Title (ZH)**: DisCO: 使用辨别性约束优化强化大型推理模型 

**Authors**: Gang Li, Ming Lin, Tomer Galanti, Zhengzhong Tu, Tianbao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12366)  

**Abstract**: The recent success and openness of DeepSeek-R1 have brought widespread attention to Group Relative Policy Optimization (GRPO) as a reinforcement learning method for large reasoning models (LRMs). In this work, we analyze the GRPO objective under a binary reward setting and reveal an inherent limitation of question-level difficulty bias. We also identify a connection between GRPO and traditional discriminative methods in supervised learning. Motivated by these insights, we introduce a new Discriminative Constrained Optimization (DisCO) framework for reinforcing LRMs, grounded in the principle of discriminative learning. The main differences between DisCO and GRPO and its recent variants are: (1) it replaces the group relative objective with a discriminative objective defined by a scoring function; (2) it abandons clipping-based surrogates in favor of non-clipping RL surrogate objectives used as scoring functions; (3) it employs a simple yet effective constrained optimization approach to enforce the KL divergence constraint, ensuring stable training. As a result, DisCO offers notable advantages over GRPO and its variants: (i) it completely eliminates difficulty bias by adopting discriminative objectives; (ii) it addresses the entropy instability in GRPO and its variants through the use of non-clipping scoring functions and a constrained optimization approach; (iii) it allows the incorporation of advanced discriminative learning techniques to address data imbalance, where a significant number of questions have more negative than positive generated answers during training. Our experiments on enhancing the mathematical reasoning capabilities of SFT-finetuned models show that DisCO significantly outperforms GRPO and its improved variants such as DAPO, achieving average gains of 7\% over GRPO and 6\% over DAPO across six benchmark tasks for an 1.5B model. 

**Abstract (ZH)**: 深度求索-R1的最近成功和开放性引起了对Group Relative Policy Optimization (GRPO)作为大规模推理模型（LRMs）的强化学习方法的广泛关注。在二元奖励设置下分析GRPO目标，揭示了一个固有的问题级别难度偏差限制。我们还发现GRPO与监督学习中的传统区分方法之间的联系。基于这些洞察，我们引入了一种新的区分约束优化（DisCO）框架，该框架基于区分学习的原则。DisCO与GRPO及其最近变体之间的主要区别在于：(1) 使用由评分函数定义的区分目标替代群体相对目标；(2) 采用非剪辑基于的RL替代目标作为评分函数，而非剪辑基适应；(3) 采用简单的有效约束优化方法来确保KL散度约束的满足，从而实现稳定的训练。因此，DisCO在与GRPO及其变体相比时展现出显著优势：(i) 通过采用区分目标完全消除了难度偏差；(ii) 通过使用非剪辑评分函数和约束优化方法解决了GRPO及其变体中的熵不稳定问题；(iii) 允许整合高级区分学习技术以解决数据不平衡问题，在训练过程中大量问题的生成答案中负面答案多于正面答案。在增强SFT微调模型的数学推理能力方面的实验表明，DisCO显著优于GRPO及其改进变体DAPO，在一个1.5B模型的六项基准任务中，DisCO相对于GRPO的平均改进幅度为7%，相对于DAPO为6%。 

---
# Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts 

**Title (ZH)**: 通过视觉专家层次融合 towards 视知觉认知 

**Authors**: Qi Feng, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12363)  

**Abstract**: While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research. 

**Abstract (ZH)**: While多模态大规模语言模型（MLLMs）在通用视觉-语言任务上表现出色，但在视觉空间认知（如空间布局、关系和动态的推理）方面仍面临重大挑战。现有的模型往往缺乏必要的架构组件和专门的训练数据，以实现细粒度的空间理解。我们介绍了ViCA2（视觉空间认知助手2），这是一种新型的MLLM，旨在增强空间推理能力。ViCA2采用了一种结合SigLIP进行语义学习和Hiera进行空间结构学习的双视觉编码器架构，并且集成了令牌比例控制机制以提高效率。我们还开发了包含超过32.2万对空间相关问题-答案配对的新大规模数据集ViCA-322K，用于针对指令进行微调。在具有挑战性的VSI-Bench基准测试中，我们的ViCA2-7B模型达到了平均得分为56.8的最新水平，显著超过了更大的开源模型（例如LLaVA-NeXT-Video-72B，40.9）和私有模型（Gemini-1.5 Pro，45.4）。这表明我们方法的有效性，能够在紧凑的模型中实现强大的视觉空间智能。我们发布了ViCA2及其代码库以及ViCA-322K数据集，以促进进一步的研究。 

---
# Adaptive MPC-based quadrupedal robot control under periodic disturbances 

**Title (ZH)**: 基于周期性干扰的自适应MPC四足机器人控制 

**Authors**: Elizaveta Pestova, Ilya Osokin, Danil Belov, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.12361)  

**Abstract**: Recent advancements in adaptive control for reference trajectory tracking enable quadrupedal robots to perform locomotion tasks under challenging conditions. There are methods enabling the estimation of the external disturbances in terms of forces and torques. However, a specific case of disturbances that are periodic was not explicitly tackled in application to quadrupeds. This work is devoted to the estimation of the periodic disturbances with a lightweight regressor using simplified robot dynamics and extracting the disturbance properties in terms of the magnitude and frequency. Experimental evidence suggests performance improvement over the baseline static disturbance compensation. All source files, including simulation setups, code, and calculation scripts, are available on GitHub at this https URL. 

**Abstract (ZH)**: 近期自适应控制在参考轨迹跟踪方面的进展使四足机器人能够在复杂条件中完成运动任务。这些方法能够估计外部干扰力和力矩，然而对于周期性干扰的具体情形在四足机器人应用中未被明确处理。本工作旨在使用简化机器人动力学和提取干扰的幅度和频率特性，通过轻量级回归器估计周期性干扰。实验结果表明，与基准静态干扰补偿方法相比，性能得到了提升。所有源代码文件，包括仿真设置、代码和计算脚本，均可在以下GitHub链接处获取。 

---
# AbFlowNet: Optimizing Antibody-Antigen Binding Energy via Diffusion-GFlowNet Fusion 

**Title (ZH)**: AbFlowNet: 通过扩散-GFlowNet 融合优化抗体-抗原结合能量 

**Authors**: Abrar Rahman Abir, Haz Sameen Shahgir, Md Rownok Zahan Ratul, Md Toki Tahmid, Greg Ver Steeg, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12358)  

**Abstract**: Complementarity Determining Regions (CDRs) are critical segments of an antibody that facilitate binding to specific antigens. Current computational methods for CDR design utilize reconstruction losses and do not jointly optimize binding energy, a crucial metric for antibody efficacy. Rather, binding energy optimization is done through computationally expensive Online Reinforcement Learning (RL) pipelines rely heavily on unreliable binding energy estimators. In this paper, we propose AbFlowNet, a novel generative framework that integrates GFlowNet with Diffusion models. By framing each diffusion step as a state in the GFlowNet framework, AbFlowNet jointly optimizes standard diffusion losses and binding energy by directly incorporating energy signals into the training process, thereby unifying diffusion and reward optimization in a single procedure. Experimental results show that AbFlowNet outperforms the base diffusion model by 3.06% in amino acid recovery, 20.40% in geometric reconstruction (RMSD), and 3.60% in binding energy improvement ratio. ABFlowNet also decreases Top-1 total energy and binding energy errors by 24.8% and 38.1% without pseudo-labeling the test dataset or using computationally expensive online RL regimes. 

**Abstract (ZH)**: CDR设计的AbFlowNet：结合GFlowNet与扩散模型的生成框架 

---
# A universal policy wrapper with guarantees 

**Title (ZH)**: 具有保证的通用策略封装 

**Authors**: Anton Bolychev, Georgiy Malaniya, Grigory Yaremenko, Anastasia Krasnaya, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.12354)  

**Abstract**: We introduce a universal policy wrapper for reinforcement learning agents that ensures formal goal-reaching guarantees. In contrast to standard reinforcement learning algorithms that excel in performance but lack rigorous safety assurances, our wrapper selectively switches between a high-performing base policy -- derived from any existing RL method -- and a fallback policy with known convergence properties. Base policy's value function supervises this switching process, determining when the fallback policy should override the base policy to ensure the system remains on a stable path. The analysis proves that our wrapper inherits the fallback policy's goal-reaching guarantees while preserving or improving upon the performance of the base policy. Notably, it operates without needing additional system knowledge or online constrained optimization, making it readily deployable across diverse reinforcement learning architectures and tasks. 

**Abstract (ZH)**: 一种确保正式目标达成保证的通用策略包装器：兼具高效与安全 

---
# Importance Sampling for Nonlinear Models 

**Title (ZH)**: 非线性模型中的重要性采样 

**Authors**: Prakash Palanivelu Rajmohan, Fred Roosta  

**Link**: [PDF](https://arxiv.org/pdf/2505.12353)  

**Abstract**: While norm-based and leverage-score-based methods have been extensively studied for identifying "important" data points in linear models, analogous tools for nonlinear models remain significantly underdeveloped. By introducing the concept of the adjoint operator of a nonlinear map, we address this gap and generalize norm-based and leverage-score-based importance sampling to nonlinear settings. We demonstrate that sampling based on these generalized notions of norm and leverage scores provides approximation guarantees for the underlying nonlinear mapping, similar to linear subspace embeddings. As direct applications, these nonlinear scores not only reduce the computational complexity of training nonlinear models by enabling efficient sampling over large datasets but also offer a novel mechanism for model explainability and outlier detection. Our contributions are supported by both theoretical analyses and experimental results across a variety of supervised learning scenarios. 

**Abstract (ZH)**: 尽管基于范数和杠杆得分的方法已被广泛研究用于识别线性模型中的“重要”数据点，但对于非线性模型的相应工具仍然显著不足。通过引入非线性映射伴随算子的概念，我们填补了这一空白，并将基于范数和杠杆得分的重要抽样方法推广到非线性设置中。我们证明，基于这些广义范数和杠杆得分的抽样方法可以为底层非线性映射提供逼近保证，类似于线性子空间嵌入。作为直接应用，这些非线性得分不仅通过启用大规模数据集的有效抽样来降低训练非线性模型的计算复杂性，而且还提供了一种新的模型可解释性和异常检测机制。我们的贡献得到了各种监督学习场景下理论分析和实验结果的支持。 

---
# Multi-CALF: A Policy Combination Approach with Statistical Guarantees 

**Title (ZH)**: 多CALF：一种带有统计保证的策略组合方法 

**Authors**: Georgiy Malaniya, Anton Bolychev, Grigory Yaremenko, Anastasia Krasnaya, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.12350)  

**Abstract**: We introduce Multi-CALF, an algorithm that intelligently combines reinforcement learning policies based on their relative value improvements. Our approach integrates a standard RL policy with a theoretically-backed alternative policy, inheriting formal stability guarantees while often achieving better performance than either policy individually. We prove that our combined policy converges to a specified goal set with known probability and provide precise bounds on maximum deviation and convergence time. Empirical validation on control tasks demonstrates enhanced performance while maintaining stability guarantees. 

**Abstract (ZH)**: Multi-CALF：一种基于相对价值改进智能结合强化学习策略的算法 

---
# Wisdom from Diversity: Bias Mitigation Through Hybrid Human-LLM Crowds 

**Title (ZH)**: 多样性之智：通过混合人类-大语言模型群体减轻偏差 

**Authors**: Axel Abels, Tom Lenaerts  

**Link**: [PDF](https://arxiv.org/pdf/2505.12349)  

**Abstract**: Despite their performance, large language models (LLMs) can inadvertently perpetuate biases found in the data they are trained on. By analyzing LLM responses to bias-eliciting headlines, we find that these models often mirror human biases. To address this, we explore crowd-based strategies for mitigating bias through response aggregation. We first demonstrate that simply averaging responses from multiple LLMs, intended to leverage the "wisdom of the crowd", can exacerbate existing biases due to the limited diversity within LLM crowds. In contrast, we show that locally weighted aggregation methods more effectively leverage the wisdom of the LLM crowd, achieving both bias mitigation and improved accuracy. Finally, recognizing the complementary strengths of LLMs (accuracy) and humans (diversity), we demonstrate that hybrid crowds containing both significantly enhance performance and further reduce biases across ethnic and gender-related contexts. 

**Abstract (ZH)**: 尽管大语言模型在性能上表现出色，但它们可能会无意中延续训练数据中发现的偏见。通过分析大语言模型对引发偏见的头条新闻的响应，我们发现这些模型经常反映出人类的偏见。为此，我们探索了基于众包的方法，通过响应聚合来减轻偏见。我们首先证明，简单地将多个大语言模型的响应取平均，虽然旨在利用“众人的智慧”，但由于大语言模型群体内的多样性有限，反而会使现有的偏见加剧。相比之下，我们展示了局部加权聚合方法更有效地利用大语言模型群体的“智慧”，既能减轻偏见，又能提高准确率。最后，考虑到大语言模型（准确性）和人类（多样性）的互补优势，我们证明了包含两者的混合群体显著提高了性能，并进一步减少了与种族和性别相关的偏见。 

---
# Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models 

**Title (ZH)**: 通过层间一致性聚合减轻幻觉现象在大型视觉-语言模型中的影响 

**Authors**: Kai Tang, Jinhao You, Xiuqi Ge, Hanze Li, Yichen Guo, Xiande Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12343)  

**Abstract**: Despite the impressive capabilities of Large Vision-Language Models (LVLMs), they remain susceptible to hallucinations-generating content that is inconsistent with the input image. Existing training-free hallucination mitigation methods often suffer from unstable performance and high sensitivity to hyperparameter settings, limiting their practicality and broader adoption. In this paper, we propose a novel decoding mechanism, Decoding with Inter-layer Consistency via Layer Aggregation (DCLA), which requires no retraining, fine-tuning, or access to external knowledge bases. Specifically, our approach constructs a dynamic semantic reference by aggregating representations from previous layers, and corrects semantically deviated layers to enforce inter-layer consistency. The method allows DCLA to robustly mitigate hallucinations across multiple LVLMs. Experiments on hallucination benchmarks such as MME and POPE demonstrate that DCLA effectively reduces hallucinations while enhancing the reliability and performance of LVLMs. 

**Abstract (ZH)**: 尽管大型视觉-语言模型（LVLMs）具备令人印象深刻的 capability，它们仍然容易产生幻觉，即生成与输入图像不一致的内容。现有的无训练的幻觉抑制方法往往性能不稳定且对超参数设置高度敏感，限制了它们的实用性和更广泛的采用。在本文中，我们提出了一种新的解码机制，层聚合的层内一致性解码（DCLA），该机制不需要重新训练、微调或访问外部知识库。具体而言，我们的方法通过聚合先前层的表示来构建动态语义参考，并纠正语义偏离的层以确保层内一致性。该方法使 DCLA 能够在多个 LVLMs 上稳健地抑制幻觉。实验结果，如在 MME 和 POPE 幻觉基准上的实验表明，DCLA 有效减少了幻觉，并提高了 LVLMs 的可靠性和性能。 

---
# Towards Open-world Generalized Deepfake Detection: General Feature Extraction via Unsupervised Domain Adaptation 

**Title (ZH)**: 面向开放世界泛化深伪检测：基于无监督领域适应的一般特征提取 

**Authors**: Midou Guo, Qilin Yin, Wei Lu, Xiangyang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12339)  

**Abstract**: With the development of generative artificial intelligence, new forgery methods are rapidly emerging. Social platforms are flooded with vast amounts of unlabeled synthetic data and authentic data, making it increasingly challenging to distinguish real from fake. Due to the lack of labels, existing supervised detection methods struggle to effectively address the detection of unknown deepfake methods. Moreover, in open world scenarios, the amount of unlabeled data greatly exceeds that of labeled data. Therefore, we define a new deepfake detection generalization task which focuses on how to achieve efficient detection of large amounts of unlabeled data based on limited labeled data to simulate a open world scenario. To solve the above mentioned task, we propose a novel Open-World Deepfake Detection Generalization Enhancement Training Strategy (OWG-DS) to improve the generalization ability of existing methods. Our approach aims to transfer deepfake detection knowledge from a small amount of labeled source domain data to large-scale unlabeled target domain data. Specifically, we introduce the Domain Distance Optimization (DDO) module to align different domain features by optimizing both inter-domain and intra-domain distances. Additionally, the Similarity-based Class Boundary Separation (SCBS) module is used to enhance the aggregation of similar samples to ensure clearer class boundaries, while an adversarial training mechanism is adopted to learn the domain-invariant features. Extensive experiments show that the proposed deepfake detection generalization enhancement training strategy excels in cross-method and cross-dataset scenarios, improving the model's generalization. 

**Abstract (ZH)**: 随着生成式人工智能的发展，新的伪造方法快速涌现。社交平台充斥着大量未标记的合成数据和真实数据，使得区分真实与伪造越来越困难。由于缺乏标签，现有的监督检测方法难以有效应对未知深度伪造方法的检测问题。此外，在开放场景中，未标记数据的数量远远超过标记数据的数量。因此，我们定义了一个新的深度伪造检测泛化任务，重点关注如何基于有限的标记数据实现大量未标记数据的高效检测，以模拟开放场景。为了解决上述任务，我们提出了一种新的开放世界深度伪造检测泛化增强训练策略（OWG-DS），以改进现有方法的泛化能力。我们的方法旨在将少量标记源域数据中的深度伪造检测知识转移到大规模未标记的目标域数据中。具体而言，我们引入了域距离优化（DDO）模块，通过优化跨域和同域距离来对齐不同的域特征。此外，我们使用基于相似性的类边界分离（SCBS）模块来增强相似样本的聚合，以确保更清晰的类边界，并采用了对抗训练机制来学习跨域不变特征。广泛实验显示，所提出的方法在跨方法和跨数据集场景中表现出色，提升了模型的泛化能力。 

---
# VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning 

**Title (ZH)**: VoiceCloak: 一种针对未经授权扩散 cloning 语音的多维度防御框架 

**Authors**: Qianyue Hu, Junyan Wu, Wei Lu, Xiangyang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12332)  

**Abstract**: Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak's outstanding defense success rate against unauthorized diffusion-based voice cloning. Audio samples of VoiceCloak are available at this https URL. 

**Abstract (ZH)**: 扩散模型（DMs）在现实语音克隆（VC）中取得了显著成功，但也增加了恶意滥用的风险。针对传统VC模型的现有主动防御旨在扰乱伪造过程，但已被证明与DMs不兼容，因为扩散模型具有的复杂生成机制。为解决这一问题，我们引入了VoiceCloak，这是一种多维度的主动防御框架，旨在在潜在未经授权的VC中模糊说话者身份并降低感知质量。为了实现这些目标，我们进行了集中分析，以识别DMs中的具体漏洞，从而使VoiceCloak能够通过在参考音频中引入对抗性扰动来扰乱克隆过程。具体而言，为了模糊说话者身份，VoiceCloak首先通过扭曲表示学习嵌入以最大化身份变化来攻击说话者身份，这一过程受到听觉感知原则的指导。此外，VoiceCloak还扰乱了关键的条件指导过程，特别是注意上下文，从而防止了实现逼真克隆所需的声音特征对齐。然后，为了应对第二项任务，VoiceCloak引入了评分幅度放大，主动引导反向轨迹远离高质量语音的生成。通过噪声引导的语义损坏进一步破坏了DMs捕捉到的结构性语音语义，降低了输出质量。广泛的实验突显了VoiceCloak在对抗基于扩散的未经授权语音克隆方面的优异防御成功率。VoiceCloak的音频样本可通过此链接获取。 

---
# Robust Planning for Autonomous Driving via Mixed Adversarial Diffusion Predictions 

**Title (ZH)**: 基于混合对抗扩散预测的自动驾驶稳健规划 

**Authors**: Albert Zhao, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2505.12327)  

**Abstract**: We describe a robust planning method for autonomous driving that mixes normal and adversarial agent predictions output by a diffusion model trained for motion prediction. We first train a diffusion model to learn an unbiased distribution of normal agent behaviors. We then generate a distribution of adversarial predictions by biasing the diffusion model at test time to generate predictions that are likely to collide with a candidate plan. We score plans using expected cost with respect to a mixture distribution of normal and adversarial predictions, leading to a planner that is robust against adversarial behaviors but not overly conservative when agents behave normally. Unlike current approaches, we do not use risk measures that over-weight adversarial behaviors while placing little to no weight on low-cost normal behaviors or use hard safety constraints that may not be appropriate for all driving scenarios. We show the effectiveness of our method on single-agent and multi-agent jaywalking scenarios as well as a red light violation scenario. 

**Abstract (ZH)**: 我们描述了一种鲁棒的自主驾驶规划方法，该方法结合了由用于运动预测训练的扩散模型输出的正常代理和对抗代理的预测。我们首先训练一个扩散模型以学习正常代理行为的无偏分布。然后，在测试时通过偏差扩散模型生成可能与候选计划发生碰撞的对抗预测分布。我们使用针对正常和对抗预测混合分布的期望成本来评分计划，从而生成一种既能够抵御对抗行为的鲁棒规划器，又不会在代理正常行为时过于保守。与现有的方法不同，我们不使用会过多强调对抗行为而忽视低成本正常行为的风险度量，也不使用可能不适合所有驾驶场景的硬性安全约束。我们展示了该方法在单代理和多代理擅自横过道路场景以及闯红灯场景中的有效性。 

---
# Visuospatial Cognitive Assistant 

**Title (ZH)**: 空间认知辅助助手 

**Authors**: Qi Feng, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12312)  

**Abstract**: Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence. 

**Abstract (ZH)**: 基于视频的空间认知对于机器人技术和具身人工智能至关重要，但对当前的视觉-语言模型构成了挑战。本文做出了两项关键贡献。首先，我们引入了包含322,003个问答对的ViCA (Visuospatial Cognitive Assistant)-322K数据集，这些问答对来自真实室内视频（ARKitScenes、ScanNet、ScanNet++），为三维元数据关联查询和基于视频的复杂推理提供监督。其次，我们开发了在ViCA-322K上微调的ViCA-7B模型，该模型在所有八个VSI-Bench任务上取得了新的最先进技术指标，超越了现有模型，包括更大规模的模型（例如，在绝对距离任务上提高26.1%）。为了增强模型的可解释性，我们提出了含有明确推理链的ViCA-Thinking-2.68K数据集，并微调ViCA-7B创建了能够明确表达空间推理的ViCA-7B-Thinking模型。我们的工作强调了目标数据的重要性，并指出了改进时空建模的方法。我们发布了所有资源以促进鲁棒视觉空间智能的研究。 

---
# DNOI-4DRO: Deep 4D Radar Odometry with Differentiable Neural-Optimization Iterations 

**Title (ZH)**: DNOI-4DRO：深度四维雷达里程计与可微神经优化迭代 

**Authors**: Shouyi Lu, Huanyu Zhou, Guirong Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12310)  

**Abstract**: A novel learning-optimization-combined 4D radar odometry model, named DNOI-4DRO, is proposed in this paper. The proposed model seamlessly integrates traditional geometric optimization with end-to-end neural network training, leveraging an innovative differentiable neural-optimization iteration operator. In this framework, point-wise motion flow is first estimated using a neural network, followed by the construction of a cost function based on the relationship between point motion and pose in 3D space. The radar pose is then refined using Gauss-Newton updates. Additionally, we design a dual-stream 4D radar backbone that integrates multi-scale geometric features and clustering-based class-aware features to enhance the representation of sparse 4D radar point clouds. Extensive experiments on the VoD and Snail-Radar datasets demonstrate the superior performance of our model, which outperforms recent classical and learning-based approaches. Notably, our method even achieves results comparable to A-LOAM with mapping optimization using LiDAR point clouds as input. Our models and code will be publicly released. 

**Abstract (ZH)**: 一种新型学习-优化结合的4D雷达里程计模型DNOI-4DRO在本文中被提出。 

---
# Community Search in Time-dependent Road-social Attributed Networks 

**Title (ZH)**: 时间依赖的道路社会属性网络中的社区搜索 

**Authors**: Li Ni, Hengkai Xu, Lin Mu, Yiwen Zhang, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12309)  

**Abstract**: Real-world networks often involve both keywords and locations, along with travel time variations between locations due to traffic conditions. However, most existing cohesive subgraph-based community search studies utilize a single attribute, either keywords or locations, to identify communities. They do not simultaneously consider both keywords and locations, which results in low semantic or spatial cohesiveness of the detected communities, and they fail to account for variations in travel time. Additionally, these studies traverse the entire network to build efficient indexes, but the detected community only involves nodes around the query node, leading to the traversal of nodes that are not relevant to the community. Therefore, we propose the problem of discovering semantic-spatial aware k-core, which refers to a k-core with high semantic and time-dependent spatial cohesiveness containing the query node. To address this problem, we propose an exact and a greedy algorithm, both of which gradually expand outward from the query node. They are local methods that only access the local part of the attributed network near the query node rather than the entire network. Moreover, we design a method to calculate the semantic similarity between two keywords using large language models. This method alleviates the disadvantages of keyword-matching methods used in existing community search studies, such as mismatches caused by differently expressed synonyms and the presence of irrelevant words. Experimental results show that the greedy algorithm outperforms baselines in terms of structural, semantic, and time-dependent spatial cohesiveness. 

**Abstract (ZH)**: 基于语义空间感知的k内核发现 

---
# Pre-trained Prompt-driven Community Search 

**Title (ZH)**: 预训练提示驱动社区搜索 

**Authors**: Li Ni, Hengkai Xu, Lin Mu, Yiwen Zhang, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12304)  

**Abstract**: The "pre-train, prompt" paradigm is widely adopted in various graph-based tasks and has shown promising performance in community detection. Most existing semi-supervised community detection algorithms detect communities based on known ones, and the detected communities typically do not contain the given query node. Therefore, they are not suitable for searching the community of a given node. Motivated by this, we adopt this paradigm into the semi-supervised community search for the first time and propose Pre-trained Prompt-driven Community Search (PPCS), a novel model designed to enhance search accuracy and efficiency. PPCS consists of three main components: node encoding, sample generation, and prompt-driven fine-tuning. Specifically, the node encoding component employs graph neural networks to learn local structural patterns of nodes in a graph, thereby obtaining representations for nodes and communities. Next, the sample generation component identifies an initial community for a given node and selects known communities that are structurally similar to the initial one as training samples. Finally, the prompt-driven fine-tuning component leverages these samples as prompts to guide the final community prediction. Experimental results on five real-world datasets demonstrate that PPCS performs better than baseline algorithms. It also achieves higher community search efficiency than semi-supervised community search baseline methods, with ablation studies verifying the effectiveness of each component of PPCS. 

**Abstract (ZH)**: 预训练提示驱动社区搜索（PPCS）：一种增强社区搜索准确性和效率的新模型 

---
# Enhance Mobile Agents Thinking Process Via Iterative Preference Learning 

**Title (ZH)**: 通过迭代偏好学习增强移动代理的思维过程 

**Authors**: Kun Huang, Weikai Xu, Yuxuan Liu, Quandong Wang, Pengzhi Gao, Wei Liu, Jian Luan, Bin Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.12299)  

**Abstract**: The Chain of Action-Planning Thoughts (CoaT) paradigm has been shown to improve the reasoning performance of VLM-based mobile agents in GUI tasks. However, the scarcity of diverse CoaT trajectories limits the expressiveness and generalization ability of such agents. While self-training is commonly employed to address data scarcity, existing approaches either overlook the correctness of intermediate reasoning steps or depend on expensive process-level annotations to construct process reward models (PRM). To address the above problems, we propose an Iterative Preference Learning (IPL) that constructs a CoaT-tree through interative sampling, scores leaf nodes using rule-based reward, and backpropagates feedback to derive Thinking-level Direct Preference Optimization (T-DPO) pairs. To prevent overfitting during warm-up supervised fine-tuning, we further introduce a three-stage instruction evolution, which leverages GPT-4o to generate diverse Q\&A pairs based on real mobile UI screenshots, enhancing both generality and layout understanding. Experiments on three standard Mobile GUI-agent benchmarks demonstrate that our agent MobileIPL outperforms strong baselines, including continual pretraining models such as OS-ATLAS and UI-TARS. It achieves state-of-the-art performance across three standard Mobile GUI-Agents benchmarks and shows strong generalization to out-of-domain scenarios. 

**Abstract (ZH)**: 基于迭代偏好学习的CoaT树构建与优化方法在移动GUI任务中的应用 

---
# Attention-Enhanced U-Net for Accurate Segmentation of COVID-19 Infected Lung Regions in CT Scans 

**Title (ZH)**: 基于注意力增强的U-Net模型在CT扫描中准确分割COVID-19感染肺区域 

**Authors**: Amal Lahchim, Lazar Davic  

**Link**: [PDF](https://arxiv.org/pdf/2505.12298)  

**Abstract**: In this study, we propose a robust methodology for automatic segmentation of infected lung regions in COVID-19 CT scans using convolutional neural networks. The approach is based on a modified U-Net architecture enhanced with attention mechanisms, data augmentation, and postprocessing techniques. It achieved a Dice coefficient of 0.8658 and mean IoU of 0.8316, outperforming other methods. The dataset was sourced from public repositories and augmented for diversity. Results demonstrate superior segmentation performance. Future work includes expanding the dataset, exploring 3D segmentation, and preparing the model for clinical deployment. 

**Abstract (ZH)**: 本研究提出了一种鲁棒的方法，使用卷积神经网络自动分割COVID-19 CT扫描中的感染肺区域。该方法基于改进的U-Net架构，并结合了注意力机制、数据增强和后处理技术。该方法在Dice系数和mean IoU方面分别达到了0.8658和0.8316，优于其他方法。数据集来源于公共仓库并进行了多样性增强，结果表明分割性能优越。未来工作包括扩展数据集、探索3D分割以及为临床部署准备模型。 

---
# PoLO: Proof-of-Learning and Proof-of-Ownership at Once with Chained Watermarking 

**Title (ZH)**: PoLO: 同时实现证学习和证明所有权的链式水印技术 

**Authors**: Haiyu Deng, Yanna Jiang, Guangsheng Yu, Qin Wang, Xu Wang, Baihe Ma, Wei Ni, Ren Ping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12296)  

**Abstract**: Machine learning models are increasingly shared and outsourced, raising requirements of verifying training effort (Proof-of-Learning, PoL) to ensure claimed performance and establishing ownership (Proof-of-Ownership, PoO) for transactions. When models are trained by untrusted parties, PoL and PoO must be enforced together to enable protection, attribution, and compensation. However, existing studies typically address them separately, which not only weakens protection against forgery and privacy breaches but also leads to high verification overhead.
We propose PoLO, a unified framework that simultaneously achieves PoL and PoO using chained watermarks. PoLO splits the training process into fine-grained training shards and embeds a dedicated watermark in each shard. Each watermark is generated using the hash of the preceding shard, certifying the training process of the preceding shard. The chained structure makes it computationally difficult to forge any individual part of the whole training process. The complete set of watermarks serves as the PoL, while the final watermark provides the PoO. PoLO offers more efficient and privacy-preserving verification compared to the vanilla PoL solutions that rely on gradient-based trajectory tracing and inadvertently expose training data during verification, while maintaining the same level of ownership assurance of watermark-based PoO schemes. Our evaluation shows that PoLO achieves 99% watermark detection accuracy for ownership verification, while preserving data privacy and cutting verification costs to just 1.5-10% of traditional methods. Forging PoLO demands 1.1-4x more resources than honest proof generation, with the original proof retaining over 90% detection accuracy even after attacks. 

**Abstract (ZH)**: 基于链式水印的验证与所有权证明统一框架（PoLO） 

---
# SpikeX: Exploring Accelerator Architecture and Network-Hardware Co-Optimization for Sparse Spiking Neural Networks 

**Title (ZH)**: SpikeX: 探索稀疏脉冲神经网络的加速器架构与网络-硬件协同优化 

**Authors**: Boxun Xu, Richard Boone, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12292)  

**Abstract**: Spiking Neural Networks (SNNs) are promising biologically plausible models of computation which utilize a spiking binary activation function similar to that of biological neurons. SNNs are well positioned to process spatiotemporal data, and are advantageous in ultra-low power and real-time processing. Despite a large body of work on conventional artificial neural network accelerators, much less attention has been given to efficient SNN hardware accelerator design. In particular, SNNs exhibit inherent unstructured spatial and temporal firing sparsity, an opportunity yet to be fully explored for great hardware processing efficiency. In this work, we propose a novel systolic-array SNN accelerator architecture, called SpikeX, to take on the challenges and opportunities stemming from unstructured sparsity while taking into account the unique characteristics of spike-based computation. By developing an efficient dataflow targeting expensive multi-bit weight data movements, SpikeX reduces memory access and increases data sharing and hardware utilization for computations spanning across both time and space, thereby significantly improving energy efficiency and inference latency. Furthermore, recognizing the importance of SNN network and hardware co-design, we develop a co-optimization methodology facilitating not only hardware-aware SNN training but also hardware accelerator architecture search, allowing joint network weight parameter optimization and accelerator architectural reconfiguration. This end-to-end network/accelerator co-design approach offers a significant reduction of 15.1x-150.87x in energy-delay-product(EDP) without comprising model accuracy. 

**Abstract (ZH)**: 基于突触阵列的突触神经网络加速器设计：SpikeX及其能耗效率提升研究 

---
# The Tower of Babel Revisited: Multilingual Jailbreak Prompts on Closed-Source Large Language Models 

**Title (ZH)**: 巴别塔再探：对闭源大型语言模型的多语言 Jailbreak 提示研究 

**Authors**: Linghan Huang, Haolin Jin, Zhaoge Bi, Pengyue Yang, Peizhou Zhao, Taozhao Chen, Xiongfei Wu, Lei Ma, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12287)  

**Abstract**: Large language models (LLMs) have seen widespread applications across various domains, yet remain vulnerable to adversarial prompt injections. While most existing research on jailbreak attacks and hallucination phenomena has focused primarily on open-source models, we investigate the frontier of closed-source LLMs under multilingual attack scenarios. We present a first-of-its-kind integrated adversarial framework that leverages diverse attack techniques to systematically evaluate frontier proprietary solutions, including GPT-4o, DeepSeek-R1, Gemini-1.5-Pro, and Qwen-Max. Our evaluation spans six categories of security contents in both English and Chinese, generating 38,400 responses across 32 types of jailbreak attacks. Attack success rate (ASR) is utilized as the quantitative metric to assess performance from three dimensions: prompt design, model architecture, and language environment. Our findings suggest that Qwen-Max is the most vulnerable, while GPT-4o shows the strongest defense. Notably, prompts in Chinese consistently yield higher ASRs than their English counterparts, and our novel Two-Sides attack technique proves to be the most effective across all models. This work highlights a dire need for language-aware alignment and robust cross-lingual defenses in LLMs, and we hope it will inspire researchers, developers, and policymakers toward more robust and inclusive AI systems. 

**Abstract (ZH)**: 大型语言模型(LLMs)在多个领域得到了广泛应用，但仍然容易受到对抗性提示注入的攻击。尽管大多数关于 Jailbreak 攻击和幻觉现象的研究主要集中在开源模型上，我们探讨了闭源 LLMs 在多语言攻击场景下的前沿问题。我们提出了一种首创的集成式对抗框架，利用多种攻击技术系统地评估前沿专有解决方案，包括 GPT-4o、DeepSeek-R1、Gemini-1.5-Pro 和 Qwen-Max。评估涵盖了英、中两种语言的六大类安全内容，共生成了 38,400 个响应，涉及 32 种不同的 Jailbreak 攻击类型。利用攻击成功率（ASR）作为定量指标，从提示设计、模型架构和语言环境三个方面评估性能。我们的研究发现 Qwen-Max 最为脆弱，而 GPT-4o 表现出最强的防御能力。值得注意的是，中文提示的 ASR 高于其英文对应物，我们提出的新型双面攻击技术在所有模型中均表现出最佳效果。这项工作突显了在 LLMs 中引入语言意识对齐和强大的跨语言防御的迫切需求，并希望能激发研究人员、开发者和政策制定者朝着更 robust 和包容的 AI 系统方向努力。 

---
# Curriculum Abductive Learning 

**Title (ZH)**: Curriculum Abductive Learning 

**Authors**: Wen-Chao Hu, Qi-Jie Li, Lin-Han Jia, Cunjing Ge, Yu-Feng Li, Yuan Jiang, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12275)  

**Abstract**: Abductive Learning (ABL) integrates machine learning with logical reasoning in a loop: a learning model predicts symbolic concept labels from raw inputs, which are revised through abduction using domain knowledge and then fed back for retraining. However, due to the nondeterminism of abduction, the training process often suffers from instability, especially when the knowledge base is large and complex, resulting in a prohibitively large abduction space. While prior works focus on improving candidate selection within this space, they typically treat the knowledge base as a static black box. In this work, we propose Curriculum Abductive Learning (C-ABL), a method that explicitly leverages the internal structure of the knowledge base to address the ABL training challenges. C-ABL partitions the knowledge base into a sequence of sub-bases, progressively introduced during training. This reduces the abduction space throughout training and enables the model to incorporate logic in a stepwise, smooth way. Experiments across multiple tasks show that C-ABL outperforms previous ABL implementations, significantly improves training stability, convergence speed, and final accuracy, especially under complex knowledge setting. 

**Abstract (ZH)**: 基于课程的演绎学习（Curriculum Abductive Learning）：一种显式利用知识库内部结构的方法 

---
# Vague Knowledge: Evidence from Analyst Reports 

**Title (ZH)**: 模糊知识：来自分析师报告的证据 

**Authors**: Kerry Xiao, Amy Zang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12269)  

**Abstract**: People in the real world often possess vague knowledge of future payoffs, for which quantification is not feasible or desirable. We argue that language, with differing ability to convey vague information, plays an important but less known-role in subjective expectations. Empirically, we find that in their reports, analysts include useful information in linguistic expressions but not numerical forecasts. Specifically, the textual tone of analyst reports has predictive power for forecast errors and subsequent revisions in numerical forecasts, and this relation becomes stronger when analyst's language is vaguer, when uncertainty is higher, and when analysts are busier. Overall, our theory and evidence suggest that some useful information is vaguely known and only communicated through language. 

**Abstract (ZH)**: 现实世界中的人们往往对未来的收益拥有模糊的认识，这种认识难以或不需要量化。我们arg认为，语言在传达模糊信息方面的能力不同，在主观预期中扮演着重要但较少被认识到的角色。实证研究表明，在分析师的报告中，他们在语言表达中包含了有用的信息，但没有进行数值预测。具体来说，分析师报告的文本语气对未来预测误差和后续数值预测调整具有预测能力，这种关系在分析师语言更模糊、不确定性更高以及分析师更忙碌时更为显著。总体而言，我们的理论和证据表明，一些有用的信息是以模糊的方式被了解并通过语言传达的。 

---
# LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference 

**Title (ZH)**: LightRetriever：一种基于LLM的混合检索架构，查询推理速度提升1000倍 

**Authors**: Guangyuan Ma, Yongliang Ma, Xuanrui Gou, Zhenpeng Su, Ming Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12260)  

**Abstract**: Large Language Models (LLMs)-based hybrid retrieval uses LLMs to encode queries and documents into low-dimensional dense or high-dimensional sparse vectors. It retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based hybrid retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full-sized LLM on an H800 GPU, our approach achieves over a 1000x speedup for query inference with GPU acceleration, and even a 20x speedup without GPU. Experiments on large-scale retrieval benchmarks demonstrate that our method generalizes well across diverse retrieval tasks, retaining an average of 95% full-sized performance. 

**Abstract (ZH)**: 基于大型语言模型的轻量级混合检索方法：LightRetriever 

---
# MMS-VPR: Multimodal Street-Level Visual Place Recognition Dataset and Benchmark 

**Title (ZH)**: MMS-VPR：多模态街道级别视觉场所识别数据集及基准 

**Authors**: Yiwei Ou, Xiaobin Ren, Ronggui Sun, Guansong Gao, Ziyi Jiang, Kaiqi Zhao, Manfredo Manfredini  

**Link**: [PDF](https://arxiv.org/pdf/2505.12254)  

**Abstract**: Existing visual place recognition (VPR) datasets predominantly rely on vehicle-mounted imagery, lack multimodal diversity and underrepresent dense, mixed-use street-level spaces, especially in non-Western urban contexts. To address these gaps, we introduce MMS-VPR, a large-scale multimodal dataset for street-level place recognition in complex, pedestrian-only environments. The dataset comprises 78,575 annotated images and 2,512 video clips captured across 207 locations in a ~70,800 $\mathrm{m}^2$ open-air commercial district in Chengdu, China. Each image is labeled with precise GPS coordinates, timestamp, and textual metadata, and covers varied lighting conditions, viewpoints, and timeframes. MMS-VPR follows a systematic and replicable data collection protocol with minimal device requirements, lowering the barrier for scalable dataset creation. Importantly, the dataset forms an inherent spatial graph with 125 edges, 81 nodes, and 1 subgraph, enabling structure-aware place recognition. We further define two application-specific subsets -- Dataset_Edges and Dataset_Points -- to support fine-grained and graph-based evaluation tasks. Extensive benchmarks using conventional VPR models, graph neural networks, and multimodal baselines show substantial improvements when leveraging multimodal and structural cues. MMS-VPR facilitates future research at the intersection of computer vision, geospatial understanding, and multimodal reasoning. The dataset is publicly available at this https URL. 

**Abstract (ZH)**: 现有的视觉地方识别（VPR）数据集主要依赖于车载图像，缺乏多模态多样性，并且不足以代表密集的混合用途街道级空间，特别是非西方城市环境中的空间。为解决这些不足，我们引入了MMS-VPR，这是一个面向复杂行人专用环境的街道级地方识别大规模多模态数据集。该数据集包含在中国成都市一个面积约70,800平方米的露天商业区内的207个地点拍摄的78,575张标注图像和2,512个视频片段。每张图像都标注有精确的GPS坐标、时间戳和文本元数据，并涵盖了不同的光照条件、视角和时间段。MMS-VPR遵循系统且可复制的数据采集协议，具有最少的设备要求，降低了大规模数据集创建的门槛。重要的是，该数据集形成了一个内在的空间图，包含125条边、81个节点和1个子图，支持结构感知的地方识别。我们进一步定义了两个特定应用的子集——Dataset_Edges和Dataset_Points——以支持细粒度和基于图的评估任务。使用传统VPR模型、图神经网络和多模态基线进行的广泛基准测试表明，在利用多模态和结构线索时可以实现显著改进。MMS-VPR促进了计算机视觉、地理空间理解和多模态推理交叉领域的未来研究。该数据集可在以下网址公开访问。 

---
# Not All Documents Are What You Need for Extracting Instruction Tuning Data 

**Title (ZH)**: 并非所有文档都是提取指令调优数据所需的内容 

**Authors**: Chi Zhang, Huaping Zhong, Hongtao Li, Chengliang Chai, Jiawei Hong, Yuhao Deng, Jiacheng Wang, Tian Tan, Yizhou Yan, Jiantao Qiu, Ye Yuan, Guoren Wang, Conghui He, Lei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12250)  

**Abstract**: Instruction tuning improves the performance of large language models (LLMs), but it heavily relies on high-quality training data. Recently, LLMs have been used to synthesize instruction data using seed question-answer (QA) pairs. However, these synthesized instructions often lack diversity and tend to be similar to the input seeds, limiting their applicability in real-world scenarios. To address this, we propose extracting instruction tuning data from web corpora that contain rich and diverse knowledge. A naive solution is to retrieve domain-specific documents and extract all QA pairs from them, but this faces two key challenges: (1) extracting all QA pairs using LLMs is prohibitively expensive, and (2) many extracted QA pairs may be irrelevant to the downstream tasks, potentially degrading model performance. To tackle these issues, we introduce EQUAL, an effective and scalable data extraction framework that iteratively alternates between document selection and high-quality QA pair extraction to enhance instruction tuning. EQUAL first clusters the document corpus based on embeddings derived from contrastive learning, then uses a multi-armed bandit strategy to efficiently identify clusters that are likely to contain valuable QA pairs. This iterative approach significantly reduces computational cost while boosting model performance. Experiments on AutoMathText and StackOverflow across four downstream tasks show that EQUAL reduces computational costs by 5-10x and improves accuracy by 2.5 percent on LLaMA-3.1-8B and Mistral-7B 

**Abstract (ZH)**: 指令调优改善了大规模语言模型的性能，但高度依赖高质量的训练数据。最近，大规模语言模型被用于合成指令数据，使用种子问题-回答（QA） pair。然而，这些合成的指令往往缺乏多样性，倾向于与输入种子相似，限制了其在现实世界场景中的应用。为了解决这一问题，我们提出从包含丰富多样知识的网络语料库中提取指令调优数据。一个直观的解决方案是从特定领域文档中检索所有QA pair，但这种方法面临两个关键挑战：（1）使用大规模语言模型提取所有QA pair的成本极其高昂；（2）提取的许多QA pair可能与下游任务无关，可能导致模型性能下降。为解决这些问题，我们引入了EQUAL，这是一种有效且可扩展的数据提取框架，通过迭代交替进行文档选择和高质量QA pair提取来增强指令调优。EQUAL首先基于对比学习衍生的嵌入对文档语料库进行聚类，然后采用多臂bandit策略高效地识别可能包含有价值的QA pair的集群。这种迭代方法显著降低了计算成本，同时提升了模型性能。在AutoMathText和StackOverflow四个下游任务上的实验表明，EQUAL将计算成本降低了5-10倍，并在LLaMA-3.1-8B和Mistral-7B中提高了2.5%的准确性。 

---
# LAMeTA: Intent-Aware Agentic Network Optimization via a Large AI Model-Empowered Two-Stage Approach 

**Title (ZH)**: LAMeTA：基于意图感知的两阶段大型AI模型赋能代理网络优化 

**Authors**: Yinqiu Liu, Guangyuan Liu, Jiacheng Wang, Ruichen Zhang, Dusit Niyato, Geng Sun, Zehui Xiong, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.12247)  

**Abstract**: Nowadays, Generative AI (GenAI) reshapes numerous domains by enabling machines to create content across modalities. As GenAI evolves into autonomous agents capable of reasoning, collaboration, and interaction, they are increasingly deployed on network infrastructures to serve humans automatically. This emerging paradigm, known as the agentic network, presents new optimization challenges due to the demand to incorporate subjective intents of human users expressed in natural language. Traditional generic Deep Reinforcement Learning (DRL) struggles to capture intent semantics and adjust policies dynamically, thus leading to suboptimality. In this paper, we present LAMeTA, a Large AI Model (LAM)-empowered Two-stage Approach for intent-aware agentic network optimization. First, we propose Intent-oriented Knowledge Distillation (IoKD), which efficiently distills intent-understanding capabilities from resource-intensive LAMs to lightweight edge LAMs (E-LAMs) to serve end users. Second, we develop Symbiotic Reinforcement Learning (SRL), integrating E-LAMs with a policy-based DRL framework. In SRL, E-LAMs translate natural language user intents into structured preference vectors that guide both state representation and reward design. The DRL, in turn, optimizes the generative service function chain composition and E-LAM selection based on real-time network conditions, thus optimizing the subjective Quality-of-Experience (QoE). Extensive experiments conducted in an agentic network with 81 agents demonstrate that IoKD reduces mean squared error in intent prediction by up to 22.5%, while SRL outperforms conventional generic DRL by up to 23.5% in maximizing intent-aware QoE. 

**Abstract (ZH)**: 基于大型AI模型的两阶段意图感知自主网络优化方法 

---
# AFCL: Analytic Federated Continual Learning for Spatio-Temporal Invariance of Non-IID Data 

**Title (ZH)**: AFCL：分析 federated 联邦持续学习以实现非 IID 数据的空间-时间不变性 

**Authors**: Jianheng Tang, Huiping Zhuang, Jingyu He, Run He, Jingchao Wang, Kejia Fan, Anfeng Liu, Tian Wang, Leye Wang, Zhanxing Zhu, Shanghang Zhang, Houbing Herbert Song, Yunhuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12245)  

**Abstract**: Federated Continual Learning (FCL) enables distributed clients to collaboratively train a global model from online task streams in dynamic real-world scenarios. However, existing FCL methods face challenges of both spatial data heterogeneity among distributed clients and temporal data heterogeneity across online tasks. Such data heterogeneity significantly degrades the model performance with severe spatial-temporal catastrophic forgetting of local and past knowledge. In this paper, we identify that the root cause of this issue lies in the inherent vulnerability and sensitivity of gradients to non-IID data. To fundamentally address this issue, we propose a gradient-free method, named Analytic Federated Continual Learning (AFCL), by deriving analytical (i.e., closed-form) solutions from frozen extracted features. In local training, our AFCL enables single-epoch learning with only a lightweight forward-propagation process for each client. In global aggregation, the server can recursively and efficiently update the global model with single-round aggregation. Theoretical analyses validate that our AFCL achieves spatio-temporal invariance of non-IID data. This ideal property implies that, regardless of how heterogeneous the data are distributed across local clients and online tasks, the aggregated model of our AFCL remains invariant and identical to that of centralized joint learning. Extensive experiments show the consistent superiority of our AFCL over state-of-the-art baselines across various benchmark datasets and settings. 

**Abstract (ZH)**: 联邦持续学习（Federated Continual Learning, FCL）使分布式客户端能够在动态现实世界场景中协作训练全局模型，从在线任务流中进行训练。然而，现有的FCL方法面临着分布式客户端之间空间数据异质性和在线任务之间时间数据异质性的挑战。这些数据异质性显著降低了模型性能，并导致了严重的空间-时间灾难性遗忘。在本文中，我们发现这一问题的根本原因在于梯度对非IID数据的内在脆弱性和敏感性。为了从根本上解决这个问题，我们提出了一种无梯度方法，即Analytic Federated Continual Learning (AFCL)，通过从冻结提取的特征中推导出解析（即，闭式）解。在局部训练中，我们的AFCL允许每个客户端仅通过一次轻量级的正向传播过程实现单轮学习。在全局聚合中，服务器可以递归且高效地通过单轮聚合更新全局模型。理论分析验证了我们的AFCL实现了非IID数据的空间-时间不变性。这一理想特性意味着，无论数据在客户端和在线任务之间的分布有多么异质，AFCL聚合后的模型都保持不变且与集中式联合学习的聚合模型一致。广泛实验表明，无论在哪些基准数据集和设置上，我们的AFCL都优于最先进的基线方法。 

---
# ACU: Analytic Continual Unlearning for Efficient and Exact Forgetting with Privacy Preservation 

**Title (ZH)**: ACU: 分析连续遗忘以实现高效的精确遗忘并保存隐私 

**Authors**: Jianheng Tang, Huiping Zhuang, Di Fang, Jiaxu Li, Feijiang Han, Yajiang Huang, Kejia Fan, Leye Wang, Zhanxing Zhu, Shanghang Zhang, Houbing Herbert Song, Yunhuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12239)  

**Abstract**: The development of artificial intelligence demands that models incrementally update knowledge by Continual Learning (CL) to adapt to open-world environments. To meet privacy and security requirements, Continual Unlearning (CU) emerges as an important problem, aiming to sequentially forget particular knowledge acquired during the CL phase. However, existing unlearning methods primarily focus on single-shot joint forgetting and face significant limitations when applied to CU. First, most existing methods require access to the retained dataset for re-training or fine-tuning, violating the inherent constraint in CL that historical data cannot be revisited. Second, these methods often suffer from a poor trade-off between system efficiency and model fidelity, making them vulnerable to being overwhelmed or degraded by adversaries through deliberately frequent requests. In this paper, we identify that the limitations of existing unlearning methods stem fundamentally from their reliance on gradient-based updates. To bridge the research gap at its root, we propose a novel gradient-free method for CU, named Analytic Continual Unlearning (ACU), for efficient and exact forgetting with historical data privacy preservation. In response to each unlearning request, our ACU recursively derives an analytical (i.e., closed-form) solution in an interpretable manner using the least squares method. Theoretical and experimental evaluations validate the superiority of our ACU on unlearning effectiveness, model fidelity, and system efficiency. 

**Abstract (ZH)**: 持续学习环境下的人工智能发展需要模型逐步更新知识，以适应开放世界环境。为满足隐私和安全要求，去持续学习（Continual Unlearning, CU）应运而生，旨在序列性地忘记持续学习阶段中获取的特定知识。然而，现有的遗忘方法主要集中在单次联合遗忘上，当应用于CU时面临重大局限。首先，大多数现有方法需要访问保留的数据集进行重新训练或微调，违背了持续学习中历史数据不可重访的内在约束。其次，这些方法通常在系统效率和模型保真度之间存在不良权衡，使其容易受到对手通过故意频繁请求的打击而变得不堪重负或降级。本文认为现有遗忘方法的局限从根本上依赖于基于梯度的更新。为从根本上填补这一研究缺口，我们提出了一种新的无梯度方法，名为解析持续学习（Analytic Continual Unlearning, ACU），以实现高效且精确的遗忘并保护历史数据隐私。对于每个遗忘请求，我们的ACU递归地以可解释的方式使用最小二乘法推导出解析（即封闭形式）解。理论和实验评估验证了ACU在遗忘有效性、模型保真度和系统效率方面的优越性。 

---
# PANORAMA: A synthetic PII-laced dataset for studying sensitive data memorization in LLMs 

**Title (ZH)**: PANORAMA：一个含有敏感个人信息的合成数据集，用于研究大语言模型中的敏感数据记忆问题 

**Authors**: Sriram Selvam, Anneswa Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2505.12238)  

**Abstract**: The memorization of sensitive and personally identifiable information (PII) by large language models (LLMs) poses growing privacy risks as models scale and are increasingly deployed in real-world applications. Existing efforts to study sensitive and PII data memorization and develop mitigation strategies are hampered by the absence of comprehensive, realistic, and ethically sourced datasets reflecting the diversity of sensitive information found on the web. We introduce PANORAMA - Profile-based Assemblage for Naturalistic Online Representation and Attribute Memorization Analysis, a large-scale synthetic corpus of 384,789 samples derived from 9,674 synthetic profiles designed to closely emulate the distribution, variety, and context of PII and sensitive data as it naturally occurs in online environments. Our data generation pipeline begins with the construction of internally consistent, multi-attribute human profiles using constrained selection to reflect real-world demographics such as education, health attributes, financial status, etc. Using a combination of zero-shot prompting and OpenAI o3-mini, we generate diverse content types - including wiki-style articles, social media posts, forum discussions, online reviews, comments, and marketplace listings - each embedding realistic, contextually appropriate PII and other sensitive information. We validate the utility of PANORAMA by fine-tuning the Mistral-7B model on 1x, 5x, 10x, and 25x data replication rates with a subset of data and measure PII memorization rates - revealing not only consistent increases with repetition but also variation across content types, highlighting PANORAMA's ability to model how memorization risks differ by context. Our dataset and code are publicly available, providing a much-needed resource for privacy risk assessment, model auditing, and the development of privacy-preserving LLMs. 

**Abstract (ZH)**: 基于档案装配的自然在线表示和属性记忆分析——PANORAMA 

---
# Bridging Generative and Discriminative Learning: Few-Shot Relation Extraction via Two-Stage Knowledge-Guided Pre-training 

**Title (ZH)**: 桥梁生成与判别学习：基于两阶段知识引导预训练的少样本关系抽取 

**Authors**: Quanjiang Guo, Jinchuan Zhang, Sijie Wang, Ling Tian, Zhao Kang, Bin Yan, Weidong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12236)  

**Abstract**: Few-Shot Relation Extraction (FSRE) remains a challenging task due to the scarcity of annotated data and the limited generalization capabilities of existing models. Although large language models (LLMs) have demonstrated potential in FSRE through in-context learning (ICL), their general-purpose training objectives often result in suboptimal performance for task-specific relation extraction. To overcome these challenges, we propose TKRE (Two-Stage Knowledge-Guided Pre-training for Relation Extraction), a novel framework that synergistically integrates LLMs with traditional relation extraction models, bridging generative and discriminative learning paradigms. TKRE introduces two key innovations: (1) leveraging LLMs to generate explanation-driven knowledge and schema-constrained synthetic data, addressing the issue of data scarcity; and (2) a two-stage pre-training strategy combining Masked Span Language Modeling (MSLM) and Span-Level Contrastive Learning (SCL) to enhance relational reasoning and generalization. Together, these components enable TKRE to effectively tackle FSRE tasks. Comprehensive experiments on benchmark datasets demonstrate the efficacy of TKRE, achieving new state-of-the-art performance in FSRE and underscoring its potential for broader application in low-resource scenarios. \footnote{The code and data are released on this https URL. 

**Abstract (ZH)**: Few-Shot 关系抽取（FSRE）由于标注数据稀少和现有模型有限的泛化能力仍然是一项具有挑战性的任务。尽管大规模语言模型（LLMs）通过上下文学习（ICL）在FSRE中显示出潜力，但它们通用的训练目标通常会导致特定任务关系抽取的性能不尽如人意。为克服这些挑战，我们提出TKRE（Two-Stage 知识引导预训练模型），一种将LLMs与传统关系抽取模型协同整合的新框架，结合生成学习和判别学习范式。TKRE引入了两个关键创新：（1）利用LLMs生成以解释为导向的知识和模式约束的合成数据，解决数据稀缺问题；（2）结合Masked Span 语言建模（MSLM）和跨度级别对比学习（SCL）的两阶段预训练策略，增强关系推理和泛化能力。这些组件共同使TKRE能够有效应对FSRE任务。基准数据集上的全面实验展示了TKRE的有效性，实现了新的FSRE最佳性能，并强调其在低资源场景中的广泛应用潜力。 

---
# Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis 

**Title (ZH)**: 浅层流匹配用于粗细粒度文本到语音合成 

**Authors**: Dong Yang, Yiyi Cai, Yuki Saito, Lixu Wang, Hiroshi Saruwatari  

**Link**: [PDF](https://arxiv.org/pdf/2505.12226)  

**Abstract**: We propose a shallow flow matching (SFM) mechanism to enhance flow matching (FM)-based text-to-speech (TTS) models within a coarse-to-fine generation paradigm. SFM constructs intermediate states along the FM paths using coarse output representations. During training, we introduce an orthogonal projection method to adaptively determine the temporal position of these states, and apply a principled construction strategy based on a single-segment piecewise flow. The SFM inference starts from the intermediate state rather than pure noise and focuses computation on the latter stages of the FM paths. We integrate SFM into multiple TTS models with a lightweight SFM head. Experiments show that SFM consistently improves the naturalness of synthesized speech in both objective and subjective evaluations, while significantly reducing inference when using adaptive-step ODE solvers. Demo and codes are available at this https URL. 

**Abstract (ZH)**: 我们提出了一种浅层流匹配(SFM)机制，以增强基于流匹配(FM)的文本到语音(TTS)模型在粗到细生成框架下的表现。SFM利用粗粒度输出表示构造FM路径中的中间状态。在训练过程中，我们引入了一种正交投影方法，以自适应地确定这些状态的时间位置，并基于单一段落的分段流提出了一种原理性的构建策略。SFM推断从中间状态开始，而非纯粹的噪声，并集中计算资源于FM路径的后期阶段。我们将SFM整合到多个TTS模型中，使用轻量级的SFM头。实验表明，SFM在客观和主观评估中一致地提高了合成语音的自然度，同时在使用自适应步长ODE求解器时显著减少了推理时间。更多信息和代码可在以下链接获取。 

---
# Reward Inside the Model: A Lightweight Hidden-State Reward Model for LLM's Best-of-N sampling 

**Title (ZH)**: 模型内的奖励：一种轻量级隐藏状态奖励模型，用于大语言模型的最佳采样 

**Authors**: Jizhou Guo, Zhaomin Wu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12225)  

**Abstract**: High-quality reward models are crucial for unlocking the reasoning potential of large language models (LLMs), with best-of-N voting demonstrating significant performance gains. However, current reward models, which typically operate on the textual output of LLMs, are computationally expensive and parameter-heavy, limiting their real-world applications. We introduce the Efficient Linear Hidden State Reward (ELHSR) model - a novel, highly parameter-efficient approach that leverages the rich information embedded in LLM hidden states to address these issues. ELHSR systematically outperform baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training. ELHSR also achieves orders-of-magnitude efficiency improvement with significantly less time and fewer FLOPs per sample than baseline reward models. Moreover, ELHSR exhibits robust performance even when trained only on logits, extending its applicability to some closed-source LLMs. In addition, ELHSR can also be combined with traditional reward models to achieve additional performance gains. 

**Abstract (ZH)**: 高质量的奖励模型对于释放大规模语言模型的推理潜力至关重要，最佳选择投票显示了显著的性能提升。然而，当前的奖励模型通常在大规模语言模型的文本输出上运行，计算成本高昂且参数量大，限制了其实际应用。我们引入了高效线性隐藏状态奖励（ELHSR）模型——一种全新的、高度参数高效的新型方法，利用大规模语言模型隐藏状态中丰富的信息来解决这些问题。ELHSR在参数少于基线0.005%的情况下系统地超越了基线模型，只需要少量样本进行训练。ELHSR还实现了与基线奖励模型相比数量级的效率提升，每样本所需时间和FLOPs显著减少。此外，即使仅在训练logits上，ELHSR也能表现出稳健的性能，使其适用于某些闭源的大规模语言模型。此外，ELHSR还可以与传统奖励模型结合使用，以实现额外的性能提升。 

---
# RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction 

**Title (ZH)**: RoboFAC：全面的机器人故障分析与纠正框架 

**Authors**: Weifeng Lu, Minghao Ye, Zewei Ye, Ruihan Tao, Shuo Yang, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12224)  

**Abstract**: Vision-Language-Action (VLA) models have recently advanced robotic manipulation by translating natural-language instructions and image information into sequential control actions. However, these models often underperform in open-world scenarios, as they are predominantly trained on successful expert demonstrations and exhibit a limited capacity for failure recovery. In this work, we present a Robotic Failure Analysis and Correction (RoboFAC) framework to address this issue. Firstly, we construct RoboFAC dataset comprising 9,440 erroneous manipulation trajectories and 78,623 QA pairs across 16 diverse tasks and 53 scenes in both simulation and real-world environments. Leveraging our dataset, we develop RoboFAC model, which is capable of Task Understanding, Failure Analysis and Failure Correction. Experimental results demonstrate that the RoboFAC model outperforms GPT-4o by 34.1% on our evaluation benchmark. Furthermore, we integrate the RoboFAC model into a real-world VLA control pipeline as an external supervision providing correction instructions, yielding a 29.1% relative improvement on average on four real-world tasks. The results show that our RoboFAC framework effectively handles robotic failures and assists the VLA model in recovering from failures. 

**Abstract (ZH)**: 基于视觉-语言-动作的机器人故障分析与修正框架 

---
# Imagination-Limited Q-Learning for Offline Reinforcement Learning 

**Title (ZH)**: 想象受限的 Offline Q 学习 

**Authors**: Wenhui Liu, Zhijian Wu, Jingchao Wang, Dingjiang Huang, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12211)  

**Abstract**: Offline reinforcement learning seeks to derive improved policies entirely from historical data but often struggles with over-optimistic value estimates for out-of-distribution (OOD) actions. This issue is typically mitigated via policy constraint or conservative value regularization methods. However, these approaches may impose overly constraints or biased value estimates, potentially limiting performance improvements. To balance exploitation and restriction, we propose an Imagination-Limited Q-learning (ILQ) method, which aims to maintain the optimism that OOD actions deserve within appropriate limits. Specifically, we utilize the dynamics model to imagine OOD action-values, and then clip the imagined values with the maximum behavior values. Such design maintains reasonable evaluation of OOD actions to the furthest extent, while avoiding its over-optimism. Theoretically, we prove the convergence of the proposed ILQ under tabular Markov decision processes. Particularly, we demonstrate that the error bound between estimated values and optimality values of OOD state-actions possesses the same magnitude as that of in-distribution ones, thereby indicating that the bias in value estimates is effectively mitigated. Empirically, our method achieves state-of-the-art performance on a wide range of tasks in the D4RL benchmark. 

**Abstract (ZH)**: 离线强化学习通过历史数据推导改进策略，但常遇到对分布外(OOD)动作的过度乐观的价值估计问题。为缓解此问题，我们提出了基于想象限制的Q学习(Imagination-Limited Q-learning, ILQ)方法，旨在在适当范围内维持OOD动作应有的乐观估计。具体而言，我们利用动力学模型想象OOD动作的价值，并通过剪切将想象的价值与最大行为价值对齐。该设计最大限度地维持了对OOD动作的合理评估，同时避免了过度乐观。理论上，我们在表格马尔可夫决策过程下证明了所提ILQ的收敛性。特别地，我们证明了对OOD状态-动作估计值与最优值之间的误差界与分布内状态-动作相同，表明价值估计的偏差得到有效缓解。实验上，该方法在D4RL基准测试中的多种任务上达到了最先进的性能。 

---
# Can Large Multimodal Models Understand Agricultural Scenes? Benchmarking with AgroMind 

**Title (ZH)**: 大型多模态模型能理解农业场景吗？使用AgroMind进行评估 

**Authors**: Qingmei Li, Yang Zhang, Zurong Mai, Yuhang Chen, Shuohong Lou, Henglian Huang, Jiarui Zhang, Zhiwei Zhang, Yibin Wen, Weijia Li, Haohuan Fu, Jianxi Huang, Juepeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12207)  

**Abstract**: Large Multimodal Models (LMMs) has demonstrated capabilities across various domains, but comprehensive benchmarks for agricultural remote sensing (RS) remain scarce. Existing benchmarks designed for agricultural RS scenarios exhibit notable limitations, primarily in terms of insufficient scene diversity in the dataset and oversimplified task design. To bridge this gap, we introduce AgroMind, a comprehensive agricultural remote sensing benchmark covering four task dimensions: spatial perception, object understanding, scene understanding, and scene reasoning, with a total of 13 task types, ranging from crop identification and health monitoring to environmental analysis. We curate a high-quality evaluation set by integrating eight public datasets and one private farmland plot dataset, containing 25,026 QA pairs and 15,556 images. The pipeline begins with multi-source data preprocessing, including collection, format standardization, and annotation refinement. We then generate a diverse set of agriculturally relevant questions through the systematic definition of tasks. Finally, we employ LMMs for inference, generating responses, and performing detailed examinations. We evaluated 18 open-source LMMs and 3 closed-source models on AgroMind. Experiments reveal significant performance gaps, particularly in spatial reasoning and fine-grained recognition, it is notable that human performance lags behind several leading LMMs. By establishing a standardized evaluation framework for agricultural RS, AgroMind reveals the limitations of LMMs in domain knowledge and highlights critical challenges for future work. Data and code can be accessed at this https URL. 

**Abstract (ZH)**: 农业智能感知基准（AgroMind）：覆盖多模态模型在农业遥感场景中的全面评估 

---
# Always Clear Depth: Robust Monocular Depth Estimation under Adverse Weather 

**Title (ZH)**: 始终清晰Depth：恶劣天气下的鲁棒单目深度估计 

**Authors**: Kui Jiang, Jing Cao, Zhaocheng Yu, Junjun Jiang, Jingchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12199)  

**Abstract**: Monocular depth estimation is critical for applications such as autonomous driving and scene reconstruction. While existing methods perform well under normal scenarios, their performance declines in adverse weather, due to challenging domain shifts and difficulties in extracting scene information. To address this issue, we present a robust monocular depth estimation method called \textbf{ACDepth} from the perspective of high-quality training data generation and domain adaptation. Specifically, we introduce a one-step diffusion model for generating samples that simulate adverse weather conditions, constructing a multi-tuple degradation dataset during training. To ensure the quality of the generated degradation samples, we employ LoRA adapters to fine-tune the generation weights of diffusion model. Additionally, we integrate circular consistency loss and adversarial training to guarantee the fidelity and naturalness of the scene contents. Furthermore, we elaborate on a multi-granularity knowledge distillation strategy (MKD) that encourages the student network to absorb knowledge from both the teacher model and pretrained Depth Anything V2. This strategy guides the student model in learning degradation-agnostic scene information from various degradation inputs. In particular, we introduce an ordinal guidance distillation mechanism (OGD) that encourages the network to focus on uncertain regions through differential ranking, leading to a more precise depth estimation. Experimental results demonstrate that our ACDepth surpasses md4all-DD by 2.50\% for night scene and 2.61\% for rainy scene on the nuScenes dataset in terms of the absRel metric. 

**Abstract (ZH)**: 单目深度估计对于自动驾驶和场景重建等应用至关重要。虽然现有方法在正常场景下表现良好，但在恶劣天气条件下性能会下降，这主要是由于领域偏移的挑战和难以提取场景信息。为解决这一问题，我们从高质量训练数据生成和领域适应的角度提出了一种鲁棒的单目深度估计方法，名为ACDepth。具体来说，我们引入了一步法扩散模型来生成模拟恶劣天气条件的样本，在训练过程中构建多退化数据集。为了保证生成退化样本的质量，我们采用LoRA适配器来微调扩散模型的生成权重。此外，我们结合圆一致性损失和对抗训练，确保场景内容的真实性和自然性。同时，我们阐述了一种多粒度知识蒸馏策略（MKD），鼓励学生网络从教师模型和预训练的Depth Anything V2中吸收知识，引导学生模型从各种退化输入中学习与退化无关的场景信息。特别地，我们引入了一种顺序指导蒸馏机制（OGD），通过差异排名鼓励网络关注不确定区域，从而实现更精确的深度估计。实验结果表明，我们的ACDepth在nuScenes数据集的absRel指标上，夜间场景比md4all-DD高出2.50%，雨天场景高出2.61%。 

---
# Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum 

**Title (ZH)**: 抛弃去噪器：自监督学习在数据课程学习中的噪声鲁棒性 emergence 

**Authors**: Wenquan Lu, Jiaqi Zhang, Hugues Van Assel, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2505.12191)  

**Abstract**: Self-Supervised Learning (SSL) has become a powerful solution to extract rich representations from unlabeled data. Yet, SSL research is mostly focused on clean, curated and high-quality datasets. As a result, applying SSL on noisy data remains a challenge, despite being crucial to applications such as astrophysics, medical imaging, geophysics or finance. In this work, we present a fully self-supervised framework that enables noise-robust representation learning without requiring a denoiser at inference or downstream fine-tuning. Our method first trains an SSL denoiser on noisy data, then uses it to construct a denoised-to-noisy data curriculum (i.e., training first on denoised, then noisy samples) for pretraining a SSL backbone (e.g., DINOv2), combined with a teacher-guided regularization that anchors noisy embeddings to their denoised counterparts. This process encourages the model to internalize noise robustness. Notably, the denoiser can be discarded after pretraining, simplifying deployment. On ImageNet-1k with ViT-B under extreme Gaussian noise ($\sigma=255$, SNR = 0.72 dB), our method improves linear probing accuracy by 4.8% over DINOv2, demonstrating that denoiser-free robustness can emerge from noise-aware pretraining. The code is available at this https URL. 

**Abstract (ZH)**: 自监督学习（SSL）已成为从未标记数据中提取丰富表示的强大解决方案。然而，SSL研究主要集中在清洁、精心整理和高质量的数据集上。因此，尽管应用于天体物理、医学成像、地球物理或金融等领域至关重要，将SSL应用于嘈杂数据仍具挑战性。在本文中，我们提出了一种完全自监督框架，能够在无需推理时去噪器或下游微调的情况下实现噪声鲁棒的表示学习。该方法首先在嘈杂数据上训练一个SSL去噪器，然后利用它构建去噪到嘈杂的数据课程（即，先训练去噪样本，再训练嘈杂样本）来预训练SSL主干（例如，DINOv2），并结合教师引导正则化，将嘈杂嵌入锚定到其去噪对应物。这一过程促使模型内化噪声鲁棒性。值得注意的是，去噪器可以在预训练后丢弃，简化部署。在极端高斯噪声（$\sigma=255$，信噪比=0.72 dB）下的ImageNet-1k和ViT-B上，我们的方法在线性探测准确性上比DINOv2提高了4.8%，表明从噪声感知预训练中可以 emergence 无去噪器的鲁棒性。代码可在以下链接获取。 

---
# LLM-DSE: Searching Accelerator Parameters with LLM Agents 

**Title (ZH)**: LLM-DSE：使用LLM代理搜索加速器参数 

**Authors**: Hanyu Wang, Xinrui Wu, Zijian Ding, Su Zheng, Chengyue Wang, Tony Nowatzki, Yizhou Sun, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12188)  

**Abstract**: Even though high-level synthesis (HLS) tools mitigate the challenges of programming domain-specific accelerators (DSAs) by raising the abstraction level, optimizing hardware directive parameters remains a significant hurdle. Existing heuristic and learning-based methods struggle with adaptability and sample this http URL present LLM-DSE, a multi-agent framework designed specifically for optimizing HLS directives. Combining LLM with design space exploration (DSE), our explorer coordinates four agents: Router, Specialists, Arbitrator, and Critic. These multi-agent components interact with various tools to accelerate the optimization process. LLM-DSE leverages essential domain knowledge to identify efficient parameter combinations while maintaining adaptability through verbal learning from online interactions. Evaluations on the HLSyn dataset demonstrate that LLM-DSE achieves substantial $2.55\times$ performance gains over state-of-the-art methods, uncovering novel designs while reducing runtime. Ablation studies validate the effectiveness and necessity of the proposed agent interactions. Our code is open-sourced here: this https URL. 

**Abstract (ZH)**: 高級綜合（HLS）工具通過提高抽象水平減輕了編程領域特定加速器（DSAs）的挑戰，但opti型化硬件指令參數仍然是一個顯著障礙。現有表徵學習和基於學習的方法在適應性和 samp this http URL提出了一種專門用于優化HLS指令的多代理框架LLM-DSE。結合LLM與設計空間探索（DSE），我們的探索者協調了四個代理：路由器、專家、仲裁員和批評家。這些多代理組件與諸多工具互動以加速優化過程。LLM-DSE利用核心領域知識來識別高效的參數組合，通過視覺學習在線上互動中保持適應性。在HLSyn數據集上的評估表明，LLM-DSE相比現有方法實現了顯著的2.55倍性能增益，揭示了新的設計方案並降低了運行時間。消融研究驗證了所提出的代理互動的有效性和必要性。我們的代碼已經开源：this https URL。 

---
# Self-Destructive Language Model 

**Title (ZH)**: 自毁型语言模型 

**Authors**: Yuhui Wang, Rongyi Zhu, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12186)  

**Abstract**: Harmful fine-tuning attacks pose a major threat to the security of large language models (LLMs), allowing adversaries to compromise safety guardrails with minimal harmful data. While existing defenses attempt to reinforce LLM alignment, they fail to address models' inherent "trainability" on harmful data, leaving them vulnerable to stronger attacks with increased learning rates or larger harmful datasets. To overcome this critical limitation, we introduce SEAM, a novel alignment-enhancing defense that transforms LLMs into self-destructive models with intrinsic resilience to misalignment attempts. Specifically, these models retain their capabilities for legitimate tasks while exhibiting substantial performance degradation when fine-tuned on harmful data. The protection is achieved through a novel loss function that couples the optimization trajectories of benign and harmful data, enhanced with adversarial gradient ascent to amplify the self-destructive effect. To enable practical training, we develop an efficient Hessian-free gradient estimate with theoretical error bounds. Extensive evaluation across LLMs and datasets demonstrates that SEAM creates a no-win situation for adversaries: the self-destructive models achieve state-of-the-art robustness against low-intensity attacks and undergo catastrophic performance collapse under high-intensity attacks, rendering them effectively unusable. (warning: this paper contains potentially harmful content generated by LLMs.) 

**Abstract (ZH)**: 有害微调攻击对大型语言模型的安全构成了重大威胁，允许攻击者通过少量有害数据破坏安全防护。现有防御试图强化语言模型的一致性，但未能解决模型内在的“可训练性”问题，使其在学习率增加或有害数据集增大时仍易受更强攻击。为克服这一关键限制，我们引入了SEAM，这是一种新型的一致性增强防御，能够将语言模型转化为自毁模型，具备内在的对齐失效尝试的抗性。具体而言，这些模型在合法任务上保留其能力，但在有害数据上进行微调时表现出显著的性能退化。保护机制通过一种新颖的损失函数实现，该函数结合了良性数据和有害数据的优化轨迹，并增强对抗性梯度上升以增强自毁效果。为了实现实际训练，我们开发了一种高效的无Hessian梯度估计，并提供了理论上的误差界。在模型和数据集上的广泛评估表明，SEAM为对手创造了无胜算的局面：自毁模型对低强度攻击具有最先进的鲁棒性，但在高强度攻击下会遭受灾难性的性能崩溃，使其实际上无法使用。（注意：本论文包含由语言模型生成的可能存在危害的内容。） 

---
# Decoding the Mind of Large Language Models: A Quantitative Evaluation of Ideology and Biases 

**Title (ZH)**: 大语言模型的心智解码：意识形态和偏见的定量评估 

**Authors**: Manari Hirose, Masato Uchida  

**Link**: [PDF](https://arxiv.org/pdf/2505.12183)  

**Abstract**: The widespread integration of Large Language Models (LLMs) across various sectors has highlighted the need for empirical research to understand their biases, thought patterns, and societal implications to ensure ethical and effective use. In this study, we propose a novel framework for evaluating LLMs, focusing on uncovering their ideological biases through a quantitative analysis of 436 binary-choice questions, many of which have no definitive answer. By applying our framework to ChatGPT and Gemini, findings revealed that while LLMs generally maintain consistent opinions on many topics, their ideologies differ across models and languages. Notably, ChatGPT exhibits a tendency to change their opinion to match the questioner's opinion. Both models also exhibited problematic biases, unethical or unfair claims, which might have negative societal impacts. These results underscore the importance of addressing both ideological and ethical considerations when evaluating LLMs. The proposed framework offers a flexible, quantitative method for assessing LLM behavior, providing valuable insights for the development of more socially aligned AI systems. 

**Abstract (ZH)**: 大规模语言模型在各领域的广泛应用凸显了需要通过实证研究来理解其偏见、思维模式和社会影响，以确保其伦理和有效使用。本研究提出了一种新的框架来评估大规模语言模型，重点关注通过定量分析436个二元选择问题来揭示其意识形态偏见。将该框架应用于ChatGPT和Gemini的研究结果显示，虽然语言模型在许多话题上保持一致的观点，但其意识形态在不同模型和语言之间存在差异。值得注意的是，ChatGPT表现出倾向于改变观点以匹配提问者观点的趋势。这两款模型还表现出有问题的偏见和不道德或不公平的断言，可能对社会产生负面影响。这些结果强调了在评估大规模语言模型时必须兼顾意识形态和伦理考量的重要性。所提出框架提供了一种灵活的定量方法来评估语言模型的行为，为开发更具社会导向的AI系统提供了有价值的见解。 

---
# SoftPQ: Robust Instance Segmentation Evaluation via Soft Matching and Tunable Thresholds 

**Title (ZH)**: SoftPQ：基于软匹配和可调阈值的稳健实例分割评估 

**Authors**: Ranit Karmakar, Simon F. Nørrelykke  

**Link**: [PDF](https://arxiv.org/pdf/2505.12155)  

**Abstract**: Segmentation evaluation metrics traditionally rely on binary decision logic: predictions are either correct or incorrect, based on rigid IoU thresholds. Detection--based metrics such as F1 and mAP determine correctness at the object level using fixed overlap cutoffs, while overlap--based metrics like Intersection over Union (IoU) and Dice operate at the pixel level, often overlooking instance--level structure. Panoptic Quality (PQ) attempts to unify detection and segmentation assessment, but it remains dependent on hard-threshold matching--treating predictions below the threshold as entirely incorrect. This binary framing obscures important distinctions between qualitatively different errors and fails to reward gradual model improvements. We propose SoftPQ, a flexible and interpretable instance segmentation metric that redefines evaluation as a graded continuum rather than a binary classification. SoftPQ introduces tunable upper and lower IoU thresholds to define a partial matching region and applies a sublinear penalty function to ambiguous or fragmented predictions. These extensions allow SoftPQ to exhibit smoother score behavior, greater robustness to structural segmentation errors, and more informative feedback for model development and evaluation. Through controlled perturbation experiments, we show that SoftPQ captures meaningful differences in segmentation quality that existing metrics overlook, making it a practical and principled alternative for both benchmarking and iterative model refinement. 

**Abstract (ZH)**: SoftPQ：一种可调的实例分割评估指标 

---
# Reasoning Large Language Model Errors Arise from Hallucinating Critical Problem Features 

**Title (ZH)**: 大型语言模型错误源于幻觉关键问题特征 

**Authors**: Alex Heyman, Joel Zylberberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.12151)  

**Abstract**: Large language models have recently made great strides in reasoning task performance through chain-of-thought (CoT) strategies trained via reinforcement learning; however, these "reasoning large language models" (RLLMs) remain imperfect reasoners, and understanding the frequencies and causes of their failure modes is important for both users and developers. We test o1-mini, o3-mini, DeepSeek-R1, Claude 3.7 Sonnet, Gemini 2.5 Pro Preview, and Grok 3 Mini Beta on graph coloring as a variable-complexity constraint-satisfaction logic problem, and find evidence from both error rate comparisons and CoT/explanation text analysis that RLLMs are prone to hallucinate edges not specified in the prompt's description of the graph. This phenomenon persists across multiple problem complexity levels and semantic frames, and it appears to account for a significant fraction of the incorrect answers from every tested model, and the vast majority of them for some models. Our results indicate that RLLMs may possess broader issues with misrepresentation of problem specifics, and we offer suggestions for design choices to mitigate this weakness. 

**Abstract (ZH)**: 大规模语言模型通过强化学习训练的链式思考策略（CoT）在推理任务性能上取得了显著进步，但这些“推理大规模语言模型”（RLLMs）仍然是不完善的推理者。理解其推理失败模式的频率和原因对于用户和开发人员来说都非常重要。我们测试了o1-mini、o3-mini、DeepSeek-R1、Claude 3.7 Sonnet、Gemini 2.5 Pro Preview和Grok 3 Mini Beta在图着色这一变量复杂度约束 satisfaction 逻辑问题上的表现，并通过错误率对比和链式思考/解释文本分析发现，RLLMs倾向于虚构没有在图描述中指定的边。这种现象在不同问题复杂度级别和语义框架中均普遍存在，并且似乎解释了部分测试模型错误答案的大部分，而在某些模型中则解释了绝大部分错误答案。我们的研究结果表明，RLLMs可能存在更广泛的问题，即在问题具体描述上的误表征，并提出了相应的设计选择建议以减轻这一不足。 

---
# Structured Representation 

**Title (ZH)**: 结构化表示 

**Authors**: Arun Kumar, Paul Schrater  

**Link**: [PDF](https://arxiv.org/pdf/2505.12143)  

**Abstract**: Invariant representations are core to representation learning, yet a central challenge remains: uncovering invariants that are stable and transferable without suppressing task-relevant signals. This raises fundamental questions, requiring further inquiry, about the appropriate level of abstraction at which such invariants should be defined, and which aspects of a system they should characterize. Interpretation of the environment relies on abstract knowledge structures to make sense of the current state, which leads to interactions, essential drivers of learning and knowledge acquisition. We posit that interpretation operates at the level of higher-order relational knowledge; hence, invariant structures must be where knowledge resides, specifically, as partitions defined by the closure of relational paths within an abstract knowledge space. These partitions serve as the core invariant representations, forming the structural substrate where knowledge is stored and learning occurs. On the other hand, inter-partition connectors enable the deployment of these knowledge partitions encoding task-relevant transitions. Thus, invariant partitions provide the foundational primitives of structured representation. We formalize the computational foundations for structured representation of the invariant partitions based on closed semiring, a relational algebraic structure. 

**Abstract (ZH)**: 不变表示是表示学习的核心，然而一个主要挑战仍然是：发现既稳定又可迁移的不变表示，而不压制与任务相关的信息。这引发了关于应在何种抽象层次定义此类不变表示以及它们应表征系统哪些方面的根本性问题，需要进一步探究。环境解释依赖于抽象知识结构来理解当前状态，从而驱动学习和知识获取。我们认为解释操作在高阶关系知识的层次；因此，知识应储存在不变结构中，具体而言，即由抽象知识空间内关系路径的闭包定义的分区。这些分区作为核心不变表示，形成存储知识和发生学习的结构基础。另一方面，分区间的连接器使这些知识分区编码任务相关转换得以部署。因此，不变分区提供了结构化表示的基础构件。我们基于闭合半环，一种关系代数结构，形式化了结构化表示不变分区的计算基础。 

---
# Keypoints as Dynamic Centroids for Unified Human Pose and Segmentation 

**Title (ZH)**: 关键点作为统一人体姿态与分割的动力学质心 

**Authors**: Niaz Ahmad, Jawad Khan, Kang G. Shin, Youngmoon Lee, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12130)  

**Abstract**: The dynamic movement of the human body presents a fundamental challenge for human pose estimation and body segmentation. State-of-the-art approaches primarily rely on combining keypoint heatmaps with segmentation masks but often struggle in scenarios involving overlapping joints or rapidly changing poses during instance-level segmentation. To address these limitations, we propose Keypoints as Dynamic Centroid (KDC), a new centroid-based representation for unified human pose estimation and instance-level segmentation. KDC adopts a bottom-up paradigm to generate keypoint heatmaps for both easily distinguishable and complex keypoints and improves keypoint detection and confidence scores by introducing KeyCentroids using a keypoint disk. It leverages high-confidence keypoints as dynamic centroids in the embedding space to generate MaskCentroids, allowing for swift clustering of pixels to specific human instances during rapid body movements in live environments. Our experimental evaluations on the CrowdPose, OCHuman, and COCO benchmarks demonstrate KDC's effectiveness and generalizability in challenging scenarios in terms of both accuracy and runtime performance. The implementation is available at: this https URL. 

**Abstract (ZH)**: 人体动态运动为人体姿态估计和身体分割提供了基本挑战。最先进的方法主要依赖于结合关键点热图和分割掩码，但在涉及关节重叠或实例级分割中快速变化姿态的场景中往往表现不佳。为解决这些限制，我们提出了一种新的基于中心点的统一表示方法——Keypoints as Dynamic Centroid (KDC)，用于人体姿态估计和实例级分割。KDC采用自底向上的范式生成易于区分和复杂的关键点热图，并通过引入关键点盘中的KeyCentroids提高关键点检测和置信度分数。KDC利用高置信度的关键点作为嵌入空间中的动态中心点生成MaskCentroids，在实时环境中快速对特定人体实例进行像素聚类。我们在CrowdPose、OCHuman和COCO基准上的实验评估表明，KDC在准确性和运行时性能方面均表现出色，在具有挑战性的场景中具有很好的泛化性。代码实现可在以下链接获取：this https URL。 

---
# SAINT: Attention-Based Modeling of Sub-Action Dependencies in Multi-Action Policies 

**Title (ZH)**: SAINT：基于注意力的子动作依赖建模在多动作策略中 

**Authors**: Matthew Landers, Taylor W. Killian, Thomas Hartvigsen, Afsaneh Doryab  

**Link**: [PDF](https://arxiv.org/pdf/2505.12109)  

**Abstract**: The combinatorial structure of many real-world action spaces leads to exponential growth in the number of possible actions, limiting the effectiveness of conventional reinforcement learning algorithms. Recent approaches for combinatorial action spaces impose factorized or sequential structures over sub-actions, failing to capture complex joint behavior. We introduce the Sub-Action Interaction Network using Transformers (SAINT), a novel policy architecture that represents multi-component actions as unordered sets and models their dependencies via self-attention conditioned on the global state. SAINT is permutation-invariant, sample-efficient, and compatible with standard policy optimization algorithms. In 15 distinct combinatorial environments across three task domains, including environments with nearly 17 million joint actions, SAINT consistently outperforms strong baselines. 

**Abstract (ZH)**: 许多实际世界动作空间的组合结构导致可能动作数量呈指数增长，限制了传统强化学习算法的有效性。针对组合动作空间的 recent 方法在子动作上施加了因子化或序列结构，无法捕获复杂联合行为。我们提出了基于 Transformer 的 Sub-Action Interaction Network (SAINT) 新颖策略架构，该架构将多组件动作表示为无序集合，并通过全局状态条件下的自注意力机制建模它们的依赖关系。SAINT 是排列不变的、样本高效的，并且与标准策略优化算法兼容。在三个任务领域中的 15 种不同组合环境中，包括具有接近 1700 万联合动作的环境，SAINT 一致地优于强大的基线方法。 

---
# EarthSynth: Generating Informative Earth Observation with Diffusion Models 

**Title (ZH)**: EarthSynth: 生成具有信息性的地球观测数据的扩散模型 

**Authors**: Jiancheng Pan, Shiye Lei, Yuqian Fu, Jiahao Li, Yanxing Liu, Yuze Sun, Xiao He, Long Peng, Xiaomeng Huang, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12108)  

**Abstract**: Remote sensing image (RSI) interpretation typically faces challenges due to the scarcity of labeled data, which limits the performance of RSI interpretation tasks. To tackle this challenge, we propose EarthSynth, a diffusion-based generative foundation model that enables synthesizing multi-category, cross-satellite labeled Earth observation for downstream RSI interpretation tasks. To the best of our knowledge, EarthSynth is the first to explore multi-task generation for remote sensing. EarthSynth, trained on the EarthSynth-180K dataset, employs the Counterfactual Composition training strategy to improve training data diversity and enhance category control. Furthermore, a rule-based method of R-Filter is proposed to filter more informative synthetic data for downstream tasks. We evaluate our EarthSynth on scene classification, object detection, and semantic segmentation in open-world scenarios, offering a practical solution for advancing RSI interpretation. 

**Abstract (ZH)**: 基于扩散的生成基础模型EarthSynth：多任务生成跨卫星标注地球观测数据以应对遥感图像解译数据稀缺挑战 

---
# Learning Probabilistic Temporal Logic Specifications for Stochastic Systems 

**Title (ZH)**: 学习概率时序逻辑规范以描述随机系统 

**Authors**: Rajarshi Roy, Yash Pote, David Parker, Marta Kwiatkowska  

**Link**: [PDF](https://arxiv.org/pdf/2505.12107)  

**Abstract**: There has been substantial progress in the inference of formal behavioural specifications from sample trajectories, for example, using Linear Temporal Logic (LTL). However, these techniques cannot handle specifications that correctly characterise systems with stochastic behaviour, which occur commonly in reinforcement learning and formal verification. We consider the passive learning problem of inferring a Boolean combination of probabilistic LTL (PLTL) formulas from a set of Markov chains, classified as either positive or negative. We propose a novel learning algorithm that infers concise PLTL specifications, leveraging grammar-based enumeration, search heuristics, probabilistic model checking and Boolean set-cover procedures. We demonstrate the effectiveness of our algorithm in two use cases: learning from policies induced by RL algorithms and learning from variants of a probabilistic model. In both cases, our method automatically and efficiently extracts PLTL specifications that succinctly characterise the temporal differences between the policies or model variants. 

**Abstract (ZH)**: 从样本轨迹推断形式行为规范的进展，例如使用线性时序逻辑（LTL），然而这些技术无法处理正确描述具有随机行为的系统的规范，这类系统在强化学习和形式验证中常见。我们考虑一类被动学习问题，即从分类为正例或负例的状态转移链集合中推断概率线性时序逻辑（PLTL）公式的布尔组合。我们提出了一种新颖的学习算法，利用语法驱动的枚举、搜索启发式、的概率模型检测以及布尔集合覆盖程序，推断精简的PLTL规范。我们在两个应用场景中展示了该算法的有效性：从由RL算法引发的策略学习以及从概率模型的变体学习。在两种情况下，我们的方法都能够自动且高效地提取能够简洁地描述策略或模型变体之间的时间差异的PLTL规范。 

---
# Improving Fairness in LLMs Through Testing-Time Adversaries 

**Title (ZH)**: 通过测试时对抗方法提高LLMs的公平性 

**Authors**: Isabela Pereira Gregio, Ian Pons, Anna Helena Reali Costa, Artur Jordão  

**Link**: [PDF](https://arxiv.org/pdf/2505.12100)  

**Abstract**: Large Language Models (LLMs) push the bound-aries in natural language processing and generative AI, driving progress across various aspects of modern society. Unfortunately, the pervasive issue of bias in LLMs responses (i.e., predictions) poses a significant and open challenge, hindering their application in tasks involving ethical sensitivity and responsible decision-making. In this work, we propose a straightforward, user-friendly and practical method to mitigate such biases, enhancing the reliability and trustworthiness of LLMs. Our method creates multiple variations of a given sentence by modifying specific attributes and evaluates the corresponding prediction behavior compared to the original, unaltered, prediction/sentence. The idea behind this process is that critical ethical predictions often exhibit notable inconsistencies, indicating the presence of bias. Unlike previous approaches, our method relies solely on forward passes (i.e., testing-time adversaries), eliminating the need for training, fine-tuning, or prior knowledge of the training data distribution. Through extensive experiments on the popular Llama family, we demonstrate the effectiveness of our method in improving various fairness metrics, focusing on the reduction of disparities in how the model treats individuals from different racial groups. Specifically, using standard metrics, we improve the fairness in Llama3 in up to 27 percentage points. Overall, our approach significantly enhances fairness, equity, and reliability in LLM-generated results without parameter tuning or training data modifications, confirming its effectiveness in practical scenarios. We believe our work establishes an important step toward enabling the use of LLMs in tasks that require ethical considerations and responsible decision-making. 

**Abstract (ZH)**: 大型语言模型中的偏见缓解方法：提高伦理敏感性和负责任决策任务中的可靠性和可信度 

---
# When the Left Foot Leads to the Right Path: Bridging Initial Prejudice and Trainability 

**Title (ZH)**: 左脚引领右路：连接初始偏见与可塑性 

**Authors**: Alberto Bassi, Carlo Albert, Aurelien Lucchi, Marco Baity-Jesi, Emanuele Francazi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12096)  

**Abstract**: Understanding the statistical properties of deep neural networks (DNNs) at initialization is crucial for elucidating both their trainability and the intrinsic architectural biases they encode prior to data exposure. Mean-field (MF) analyses have demonstrated that the parameter distribution in randomly initialized networks dictates whether gradients vanish or explode. Concurrently, untrained DNNs were found to exhibit an initial-guessing bias (IGB), in which large regions of the input space are assigned to a single class. In this work, we derive a theoretical proof establishing the correspondence between IGB and previous MF theories, thereby connecting a network prejudice toward specific classes with the conditions for fast and accurate learning. This connection yields the counter-intuitive conclusion: the initialization that optimizes trainability is necessarily biased, rather than neutral. Furthermore, we extend the MF/IGB framework to multi-node activation functions, offering practical guidelines for designing initialization schemes that ensure stable optimization in architectures employing max- and average-pooling layers. 

**Abstract (ZH)**: 理解在初始化时深度神经网络（DNNs）的统计属性对于阐明其可训练性和预先数据暴露时内含的固有架构偏见至关重要。均场（MF）分析表明，随机初始化网络的参数分布决定了梯度是否会消失或爆炸。同时，未训练的DNNs被发现具有初始猜测偏置（IGB），即输入空间的大区域被分配给单一类别。在本文中，我们推导出一个理论证明，建立了IGB与先前的MF理论之间的对应关系，从而将网络对特定类别的偏见与快速和准确学习的条件联系起来。这一联系得出了一个反直觉的结论：优化可训练性的初始化一定是偏置的，而不是中立的。此外，我们将MF/IGB框架扩展到多节点激活函数，为确保采用最大池化和平均池化层的架构中的稳定优化提供了实用指南。 

---
# Attribution Projection Calculus: A Novel Framework for Causal Inference in Bayesian Networks 

**Title (ZH)**: Attribution Projection calculus: 一种贝叶斯网络因果推断的新框架 

**Authors**: M Ruhul Amin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12094)  

**Abstract**: This paper introduces Attribution Projection Calculus (AP-Calculus), a novel mathematical framework for determining causal relationships in structured Bayesian networks. We investigate a specific network architecture with source nodes connected to destination nodes through intermediate nodes, where each input maps to a single label with maximum marginal probability. We prove that for each label, exactly one intermediate node acts as a deconfounder while others serve as confounders, enabling optimal attribution of features to their corresponding labels. The framework formalizes the dual nature of intermediate nodes as both confounders and deconfounders depending on the context, and establishes separation functions that maximize distinctions between intermediate representations. We demonstrate that the proposed network architecture is optimal for causal inference compared to alternative structures, including those based on Pearl's causal framework. AP-Calculus provides a comprehensive mathematical foundation for analyzing feature-label attributions, managing spurious correlations, quantifying information gain, ensuring fairness, and evaluating uncertainty in prediction models, including large language models. Theoretical verification shows that AP-Calculus not only extends but can also subsume traditional do-calculus for many practical applications, offering a more direct approach to causal inference in supervised learning contexts. 

**Abstract (ZH)**: This paper引入 Attribution Projection Calculus (AP-Calculus)，一种用于确定结构化贝叶斯网络中因果关系的新型数学框架。我们研究了一种特定网络架构，其中源节点通过中介节点连接到目的节点，每个输入映射到具有最大边缘概率的单个标签。我们证明，对于每个标签，恰好有一个中介节点充当解混杂器，而其他节点充当混杂器，从而使特征能够最优地归因于相应的标签。该框架正式化了中介节点在不同上下文中作为混杂器和解混杂器的双重性质，并建立了最大化中介表示之间差异的分离函数。我们证明，所提出网络架构在因果推理方面优于包括基于佩尔因果框架的其他结构。AP-Calculus为分析特征-标签归因、管理虚假相关性、量化信息增益、确保公平性以及评估预测模型中的不确定性，包括大型语言模型，提供了全面的数学基础。理论验证表明，AP-Calculus不仅扩展了传统的do-calculus，还可以在许多实际应用中将其纳为子集，提供了在监督学习背景下进行因果推理的更直接的方法。 

---
# Personalized Author Obfuscation with Large Language Models 

**Title (ZH)**: 个性化作者混淆ewith大规模语言模型 

**Authors**: Mohammad Shokri, Sarah Ita Levitan, Rivka Levitan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12090)  

**Abstract**: In this paper, we investigate the efficacy of large language models (LLMs) in obfuscating authorship by paraphrasing and altering writing styles. Rather than adopting a holistic approach that evaluates performance across the entire dataset, we focus on user-wise performance to analyze how obfuscation effectiveness varies across individual authors. While LLMs are generally effective, we observe a bimodal distribution of efficacy, with performance varying significantly across users. To address this, we propose a personalized prompting method that outperforms standard prompting techniques and partially mitigates the bimodality issue. 

**Abstract (ZH)**: 本文研究了大型语言模型（LLM）在改写和改变写作风格以混淆作者身份方面的效果。我们不采用在整个数据集中评估整体性能的方法，而是侧重于个体用户层面的性能分析，以探讨不同作者之间的混淆效果差异。尽管LLM通常效果显著，但我们观察到其效果呈双峰分布，用户之间的表现差异显著。为此，我们提出了一种个性化提示方法，该方法优于标准提示技术，并部分缓解了双峰问题。 

---
# NTIRE 2025 Challenge on Efficient Burst HDR and Restoration: Datasets, Methods, and Results 

**Title (ZH)**: 2025年NTIRE挑战赛：高效burst HDR成像与恢复：数据集、方法和结果 

**Authors**: Sangmin Lee, Eunpil Park, Angel Canelo, Hyunhee Park, Youngjo Kim, Hyung-Ju Chun, Xin Jin, Chongyi Li, Chun-Le Guo, Radu Timofte, Qi Wu, Tianheng Qiu, Yuchun Dong, Shenglin Ding, Guanghua Pan, Weiyu Zhou, Tao Hu, Yixu Feng, Duwei Dai, Yu Cao, Peng Wu, Wei Dong, Yanning Zhang, Qingsen Yan, Simon J. Larsen, Ruixuan Jiang, Senyan Xu, Xingbo Wang, Xin Lu, Marcos V. Conde, Javier Abad-Hernandez, Alvaro Garcıa-Lara, Daniel Feijoo, Alvaro Garcıa, Zeyu Xiao, Zhuoyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12089)  

**Abstract**: This paper reviews the NTIRE 2025 Efficient Burst HDR and Restoration Challenge, which aims to advance efficient multi-frame high dynamic range (HDR) and restoration techniques. The challenge is based on a novel RAW multi-frame fusion dataset, comprising nine noisy and misaligned RAW frames with various exposure levels per scene. Participants were tasked with developing solutions capable of effectively fusing these frames while adhering to strict efficiency constraints: fewer than 30 million model parameters and a computational budget under 4.0 trillion FLOPs. A total of 217 participants registered, with six teams finally submitting valid solutions. The top-performing approach achieved a PSNR of 43.22 dB, showcasing the potential of novel methods in this domain. This paper provides a comprehensive overview of the challenge, compares the proposed solutions, and serves as a valuable reference for researchers and practitioners in efficient burst HDR and restoration. 

**Abstract (ZH)**: This paper reviews the NTIRE 2025 Efficient Burst HDR and Restoration Challenge，并对其进行详细探讨，旨在推进高效多帧高动态范围（HDR）和恢复技术。 

---
# SepPrune: Structured Pruning for Efficient Deep Speech Separation 

**Title (ZH)**: SepPrune: 结构化剪枝以实现高效的深度语音分离 

**Authors**: Yuqi Li, Kai Li, Xin Yin, Zhifei Yang, Junhao Dong, Zeyu Dong, Chuanguang Yang, Yingli Tian, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12079)  

**Abstract**: Although deep learning has substantially advanced speech separation in recent years, most existing studies continue to prioritize separation quality while overlooking computational efficiency, an essential factor for low-latency speech processing in real-time applications. In this paper, we propose SepPrune, the first structured pruning framework specifically designed to compress deep speech separation models and reduce their computational cost. SepPrune begins by analyzing the computational structure of a given model to identify layers with the highest computational burden. It then introduces a differentiable masking strategy to enable gradient-driven channel selection. Based on the learned masks, SepPrune prunes redundant channels and fine-tunes the remaining parameters to recover performance. Extensive experiments demonstrate that this learnable pruning paradigm yields substantial advantages for channel pruning in speech separation models, outperforming existing methods. Notably, a model pruned with SepPrune can recover 85% of the performance of a pre-trained model (trained over hundreds of epochs) with only one epoch of fine-tuning, and achieves convergence 36$\times$ faster than training from scratch. Code is available at this https URL. 

**Abstract (ZH)**: 尽管深度学习在近年来显著推进了语音分离技术，但现有大多数研究仍侧重于优化分离质量，而忽视了计算效率这一关键因素，后者对于实时应用中的低延迟语音处理至关重要。本文提出SepPrune——首个专门设计用于压缩深度语音分离模型并降低其计算成本的结构化剪枝框架。SepPrune首先分析给定模型的计算结构，识别出计算负担最重的层，然后引入可微掩码策略以实现梯度驱动的通道选择。基于学习到的掩码，SepPrune剪枝冗余通道，并对剩余参数进行微调以恢复性能。大量实验表明，这种可学习的剪枝范式在语音分离模型中的通道剪枝方面具有显著优势，优于现有方法。值得注意的是，使用SepPrune剪枝后的模型只需一次微调即可恢复预训练模型（经过数百个epoch训练）85%的性能，并且比从零开始训练达到收敛快36倍。代码可在以下链接获取。 

---
# MT-CYP-Net: Multi-Task Network for Pixel-Level Crop Yield Prediction Under Very Few Samples 

**Title (ZH)**: MT-CYP-Net: 多任务网络在少量样本下的像素级作物产量预测 

**Authors**: Shenzhou Liu, Di Wang, Haonan Guo, Chengxi Han, Wenzhi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12069)  

**Abstract**: Accurate and fine-grained crop yield prediction plays a crucial role in advancing global agriculture. However, the accuracy of pixel-level yield estimation based on satellite remote sensing data has been constrained by the scarcity of ground truth data. To address this challenge, we propose a novel approach called the Multi-Task Crop Yield Prediction Network (MT-CYP-Net). This framework introduces an effective multi-task feature-sharing strategy, where features extracted from a shared backbone network are simultaneously utilized by both crop yield prediction decoders and crop classification decoders with the ability to fuse information between them. This design allows MT-CYP-Net to be trained with extremely sparse crop yield point labels and crop type labels, while still generating detailed pixel-level crop yield maps. Concretely, we collected 1,859 yield point labels along with corresponding crop type labels and satellite images from eight farms in Heilongjiang Province, China, in 2023, covering soybean, maize, and rice crops, and constructed a sparse crop yield label dataset. MT-CYP-Net is compared with three classical machine learning and deep learning benchmark methods in this dataset. Experimental results not only indicate the superiority of MT-CYP-Net compared to previous methods on multiple types of crops but also demonstrate the potential of deep networks on precise pixel-level crop yield prediction, especially with limited data labels. 

**Abstract (ZH)**: 多任务作物产量预测网络：准确和细粒度的作物产量预测在推动全球农业发展中扮演着重要角色。然而，基于卫星遥感数据的像素级产量估计准确性受到地面 truth 数据稀缺性的限制。为了解决这一挑战，我们提出了一种名为多任务作物产量预测网络（MT-CYP-Net）的新方法。该框架引入了有效的多任务特征共享策略，其中从共享骨干网络中提取的特征同时被作物产量预测解码器和作物分类解码器使用，并具备它们之间信息融合的能力。该设计使MT-CYP-Net能够在极稀疏的作物产量点标签和作物类型标签的情况下进行训练，但仍能生成详细的像素级作物产量图谱。具体地，我们在2023年从中国黑龙江省八家农场收集了1,859个产量点标签及其对应的作物类型标签和卫星图像，覆盖大豆、玉米和水稻作物，并构建了一个稀疏作物产量标签数据集。在该数据集上，MT-CYP-Net与三种经典的机器学习和深度学习基准方法进行了比较。实验结果不仅表明了MT-CYP-Net在多种作物类型上的优越性，而且还展示了在有限数据标签条件下深度网络在精准像素级作物产量预测方面的潜力。 

---
# VFRTok: Variable Frame Rates Video Tokenizer with Duration-Proportional Information Assumption 

**Title (ZH)**: VFRTok：基于持续时间比例信息假设的可变帧率视频分词器 

**Authors**: Tianxiong Zhong, Xingye Tian, Boyuan Jiang, Xuebo Wang, Xin Tao, Pengfei Wan, Zhiwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12053)  

**Abstract**: Modern video generation frameworks based on Latent Diffusion Models suffer from inefficiencies in tokenization due to the Frame-Proportional Information Assumption. Existing tokenizers provide fixed temporal compression rates, causing the computational cost of the diffusion model to scale linearly with the frame rate. The paper proposes the Duration-Proportional Information Assumption: the upper bound on the information capacity of a video is proportional to the duration rather than the number of frames. Based on this insight, the paper introduces VFRTok, a Transformer-based video tokenizer, that enables variable frame rate encoding and decoding through asymmetric frame rate training between the encoder and decoder. Furthermore, the paper proposes Partial Rotary Position Embeddings (RoPE) to decouple position and content modeling, which groups correlated patches into unified tokens. The Partial RoPE effectively improves content-awareness, enhancing the video generation capability. Benefiting from the compact and continuous spatio-temporal representation, VFRTok achieves competitive reconstruction quality and state-of-the-art generation fidelity while using only 1/8 tokens compared to existing tokenizers. 

**Abstract (ZH)**: 基于Latent Diffusion Models的现代视频生成框架因帧比例信息假设导致在标记化过程中存在效率问题。现有标记化器提供固定的时域压缩率，导致扩散模型的计算成本线性地随帧率增加。本文提出了持续时间比例信息假设：视频的信息容量上限与其持续时间成正比，而不是帧数。基于这一认识，本文引入了VFRTok，一种基于Transformer的视频标记化器，通过编码器和解码器之间的非对称帧率训练实现可变帧率的编码和解码。此外，本文提出部分旋转位置嵌入（Partial RoPE）来解耦位置建模和内容建模，将相关补丁组分统为统一标记。部分RoPE有效提高了内容感知能力，增强视频生成能力。得益于紧凑且连续的空间-时间表示，VFRTok在仅有现有标记化器1/8标记量的情况下，实现了竞争性重构质量和最先进生成保真度。 

---
# Enhanced Multimodal Hate Video Detection via Channel-wise and Modality-wise Fusion 

**Title (ZH)**: 基于通道和模态融合的增强多模态仇恨视频检测 

**Authors**: Yinghui Zhang, Tailin Chen, Yuchen Zhang, Zeyu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12051)  

**Abstract**: The rapid rise of video content on platforms such as TikTok and YouTube has transformed information dissemination, but it has also facilitated the spread of harmful content, particularly hate videos. Despite significant efforts to combat hate speech, detecting these videos remains challenging due to their often implicit nature. Current detection methods primarily rely on unimodal approaches, which inadequately capture the complementary features across different modalities. While multimodal techniques offer a broader perspective, many fail to effectively integrate temporal dynamics and modality-wise interactions essential for identifying nuanced hate content. In this paper, we present CMFusion, an enhanced multimodal hate video detection model utilizing a novel Channel-wise and Modality-wise Fusion Mechanism. CMFusion first extracts features from text, audio, and video modalities using pre-trained models and then incorporates a temporal cross-attention mechanism to capture dependencies between video and audio streams. The learned features are then processed by channel-wise and modality-wise fusion modules to obtain informative representations of videos. Our extensive experiments on a real-world dataset demonstrate that CMFusion significantly outperforms five widely used baselines in terms of accuracy, precision, recall, and F1 score. Comprehensive ablation studies and parameter analyses further validate our design choices, highlighting the model's effectiveness in detecting hate videos. The source codes will be made publicly available at this https URL. 

**Abstract (ZH)**: TikTok和YouTube等平台上视频内容的迅速崛起改变了信息传播方式，但也促进了有害内容，特别是仇恨视频的传播。尽管在打击仇恨言论方面做出了巨大努力，但由于仇恨言论往往具有隐匿性，检测这些视频仍然极具挑战性。现有的检测方法主要依赖单模态方法，无法充分捕捉不同模态间的互补特征。虽然多模态技术提供了更广泛的观点，但许多方法未能有效整合有助于识别细微仇恨内容的时序动态和模态间交互。本文提出了一种增强的多模态仇恨视频检测模型CMFusion，该模型利用一种新的按通道和按模态融合机制。CMFusion首先使用预训练模型从文本、音频和视频模态中提取特征，然后采用时序交叉注意机制捕获视频流和音频流之间的依赖关系。学习到的特征通过按通道和按模态融合模块处理，以获得视频的具有信息量的表示。在真实数据集上的广泛实验表明，CMFusion在准确率、精确率、召回率和F1分数方面明显优于五种广泛使用的基线模型。全面的消融研究和参数分析进一步验证了我们的设计选择，突显了该模型在检测仇恨视频方面的有效性。源代码将在以下链接公开：this https URL。 

---
# ABoN: Adaptive Best-of-N Alignment 

**Title (ZH)**: ABoN：自适应最佳匹配-alignment 

**Authors**: Vinod Raman, Hilal Asi, Satyen Kale  

**Link**: [PDF](https://arxiv.org/pdf/2505.12050)  

**Abstract**: Recent advances in test-time alignment methods, such as Best-of-N sampling, offer a simple and effective way to steer language models (LMs) toward preferred behaviors using reward models (RM). However, these approaches can be computationally expensive, especially when applied uniformly across prompts without accounting for differences in alignment difficulty. In this work, we propose a prompt-adaptive strategy for Best-of-N alignment that allocates inference-time compute more efficiently. Motivated by latency concerns, we develop a two-stage algorithm: an initial exploratory phase estimates the reward distribution for each prompt using a small exploration budget, and a second stage adaptively allocates the remaining budget using these estimates. Our method is simple, practical, and compatible with any LM/RM combination. Empirical results on the AlpacaEval dataset for 12 LM/RM pairs and 50 different batches of prompts show that our adaptive strategy consistently outperforms the uniform allocation with the same inference budget. Moreover, our experiments show that our adaptive strategy remains competitive against uniform allocations with 20% larger inference budgets and even improves in performance as the batch size grows. 

**Abstract (ZH)**: Recent Advances in Test-Time Alignment Methods, Such as Best-of-N Sampling, Offer Prompt-Adaptive Strategies for Efficient Inference-Time Compute Allocation 

---
# Beyond Scalar Rewards: An Axiomatic Framework for Lexicographic MDPs 

**Title (ZH)**: 超越标量奖励：序贯Markov决策过程的公理化框架 

**Authors**: Mehran Shakerinava, Siamak Ravanbakhsh, Adam Oberman  

**Link**: [PDF](https://arxiv.org/pdf/2505.12049)  

**Abstract**: Recent work has formalized the reward hypothesis through the lens of expected utility theory, by interpreting reward as utility. Hausner's foundational work showed that dropping the continuity axiom leads to a generalization of expected utility theory where utilities are lexicographically ordered vectors of arbitrary dimension. In this paper, we extend this result by identifying a simple and practical condition under which preferences cannot be represented by scalar rewards, necessitating a 2-dimensional reward function. We provide a full characterization of such reward functions, as well as the general d-dimensional case, in Markov Decision Processes (MDPs) under a memorylessness assumption on preferences. Furthermore, we show that optimal policies in this setting retain many desirable properties of their scalar-reward counterparts, while in the Constrained MDP (CMDP) setting -- another common multiobjective setting -- they do not. 

**Abstract (ZH)**: 最近的研究通过期望效用理论的视角 formal化了奖励假设，将奖励解释为效用。Hausner 的基础工作表明，放弃连续性公理会导致一种效用理论的一般化，其中效用是按字典顺序排列的任意维度向量。在本文中，我们通过识别一个简单且实用的条件来扩展这一结果，该条件表明偏好不能由标量奖励表示，而是需要一个二维奖励函数。我们提供了在记忆缺失假设下的马尔可夫决策过程（MDP）中这类奖励函数的完整特征，以及一般 d 维情况的特征。此外，我们展示了在这种设置下最优策略保持标量奖励对照策略的许多可取属性，而在约束马尔可夫决策过程（CMDP）设置——另一种常见的多目标设置——中并非如此。 

---
# Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets 

**Title (ZH)**: Safe Delta: 一致地在多样数据集上微调大语言模型时保持安全性 

**Authors**: Ning Lu, Shengcai Liu, Jiahao Wu, Weiyu Chen, Zhirui Zhang, Yew-Soon Ong, Qi Wang, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12038)  

**Abstract**: Large language models (LLMs) have shown great potential as general-purpose AI assistants across various domains. To fully leverage this potential in specific applications, many companies provide fine-tuning API services, enabling users to upload their own data for LLM customization. However, fine-tuning services introduce a new safety threat: user-uploaded data, whether harmful or benign, can break the model's alignment, leading to unsafe outputs. Moreover, existing defense methods struggle to address the diversity of fine-tuning datasets (e.g., varying sizes, tasks), often sacrificing utility for safety or vice versa. To address this issue, we propose Safe Delta, a safety-aware post-training defense method that adjusts the delta parameters (i.e., the parameter change before and after fine-tuning). Specifically, Safe Delta estimates the safety degradation, selects delta parameters to maximize utility while limiting overall safety loss, and applies a safety compensation vector to mitigate residual safety loss. Through extensive experiments on four diverse datasets with varying settings, our approach consistently preserves safety while ensuring that the utility gain from benign datasets remains unaffected. 

**Abstract (ZH)**: Large语言模型（LLMs）在各个领域展现出了作为通用AI助手的巨大潜力。为了在特定应用中充分利用这一潜力，许多公司提供了微调API服务，使用户能够上传自己的数据以定制LLM。然而，微调服务引入了一个新的安全威胁：用户上传的数据，无论是有害的还是无害的，都可能导致模型失准，从而产生不安全的输出。此外，现有的防御方法难以应对微调数据集的多样性（例如，大小和任务的差异），通常会牺牲实用性以获得安全性或反之。为解决这一问题，我们提出了一种安全意识后训练防护方法Safe Delta，该方法调整了微调前后的delta参数。具体来说，Safe Delta估计了安全降级，选择delta参数以最大化实用性同时限制总体安全性损失，并应用安全补偿向量以减轻剩余的安全损失。通过在四个不同设置下的四种多样化的数据集上进行广泛实验，我们的方法在保证安全性的同时，确保良性数据集的实用性增益不受影响。 

---
# GeoMaNO: Geometric Mamba Neural Operator for Partial Differential Equations 

**Title (ZH)**: GeoMaNO: 几何Mamba神经算子 for 部分微分方程 

**Authors**: Xi Han, Jingwei Zhang, Dimitris Samaras, Fei Hou, Hong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12020)  

**Abstract**: The neural operator (NO) framework has emerged as a powerful tool for solving partial differential equations (PDEs). Recent NOs are dominated by the Transformer architecture, which offers NOs the capability to capture long-range dependencies in PDE dynamics. However, existing Transformer-based NOs suffer from quadratic complexity, lack geometric rigor, and thus suffer from sub-optimal performance on regular grids. As a remedy, we propose the Geometric Mamba Neural Operator (GeoMaNO) framework, which empowers NOs with Mamba's modeling capability, linear complexity, plus geometric rigor. We evaluate GeoMaNO's performance on multiple standard and popularly employed PDE benchmarks, spanning from Darcy flow problems to Navier-Stokes problems. GeoMaNO improves existing baselines in solution operator approximation by as much as 58.9%. 

**Abstract (ZH)**: Geometric Mamba Neural Operator (GeoMaNO)框架：一种具有线性复杂度和几何严谨性的求解偏微分方程的神经算子方法 

---
# CHRIS: Clothed Human Reconstruction with Side View Consistency 

**Title (ZH)**: CHRIS: 带有侧面一致性的人体穿衣重建 

**Authors**: Dong Liu, Yifan Yang, Zixiong Huang, Yuxin Gao, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12005)  

**Abstract**: Creating a realistic clothed human from a single-view RGB image is crucial for applications like mixed reality and filmmaking. Despite some progress in recent years, mainstream methods often fail to fully utilize side-view information, as the input single-view image contains front-view information only. This leads to globally unrealistic topology and local surface inconsistency in side views. To address these, we introduce Clothed Human Reconstruction with Side View Consistency, namely CHRIS, which consists of 1) A Side-View Normal Discriminator that enhances global visual reasonability by distinguishing the generated side-view normals from the ground truth ones; 2) A Multi-to-One Gradient Computation (M2O) that ensures local surface consistency. M2O calculates the gradient of a sampling point by integrating the gradients of the nearby points, effectively acting as a smooth operation. Experimental results demonstrate that CHRIS achieves state-of-the-art performance on public benchmarks and outperforms the prior work. 

**Abstract (ZH)**: 基于侧面一致性的人体着装重建 

---
# Online Iterative Self-Alignment for Radiology Report Generation 

**Title (ZH)**: 放射学报告生成的在线迭代自我对齐方法 

**Authors**: Ting Xiao, Lei Shi, Yang Zhang, HaoFeng Yang, Zhe Wang, Chenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11983)  

**Abstract**: Radiology Report Generation (RRG) is an important research topic for relieving radiologist' heavy workload. Existing RRG models mainly rely on supervised fine-tuning (SFT) based on different model architectures using data pairs of radiological images and corresponding radiologist-annotated reports. Recent research has shifted focus to post-training improvements, aligning RRG model outputs with human preferences using reinforcement learning (RL). However, the limited data coverage of high-quality annotated data poses risks of overfitting and generalization. This paper proposes a novel Online Iterative Self-Alignment (OISA) method for RRG that consists of four stages: self-generation of diverse data, self-evaluation for multi-objective preference data,self-alignment for multi-objective optimization and self-iteration for further improvement. Our approach allows for generating varied reports tailored to specific clinical objectives, enhancing the overall performance of the RRG model iteratively. Unlike existing methods, our frame-work significantly increases data quality and optimizes performance through iterative multi-objective optimization. Experimental results demonstrate that our method surpasses previous approaches, achieving state-of-the-art performance across multiple evaluation metrics. 

**Abstract (ZH)**: Radiology报告生成（RRG）中的在线迭代自我对齐（OISA）方法 

---
# AoP-SAM: Automation of Prompts for Efficient Segmentation 

**Title (ZH)**: AoP-SAM: 自动化提示以实现高效分割 

**Authors**: Yi Chen, Mu-Young Son, Chuanbo Hua, Joo-Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.11980)  

**Abstract**: The Segment Anything Model (SAM) is a powerful foundation model for image segmentation, showing robust zero-shot generalization through prompt engineering. However, relying on manual prompts is impractical for real-world applications, particularly in scenarios where rapid prompt provision and resource efficiency are crucial. In this paper, we propose the Automation of Prompts for SAM (AoP-SAM), a novel approach that learns to generate essential prompts in optimal locations automatically. AoP-SAM enhances SAM's efficiency and usability by eliminating manual input, making it better suited for real-world tasks. Our approach employs a lightweight yet efficient Prompt Predictor model that detects key entities across images and identifies the optimal regions for placing prompt candidates. This method leverages SAM's image embeddings, preserving its zero-shot generalization capabilities without requiring fine-tuning. Additionally, we introduce a test-time instance-level Adaptive Sampling and Filtering mechanism that generates prompts in a coarse-to-fine manner. This notably enhances both prompt and mask generation efficiency by reducing computational overhead and minimizing redundant mask refinements. Evaluations of three datasets demonstrate that AoP-SAM substantially improves both prompt generation efficiency and mask generation accuracy, making SAM more effective for automated segmentation tasks. 

**Abstract (ZH)**: AoP-SAM：用于SAM的自动提示生成方法 

---
# Introduction to Analytical Software Engineering Design Paradigm 

**Title (ZH)**: 数据分析软件工程设计范式简介 

**Authors**: Tarik Houichime, Younes El Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2505.11979)  

**Abstract**: As modern software systems expand in scale and complexity, the challenges associated with their modeling and formulation grow increasingly intricate. Traditional approaches often fall short in effectively addressing these complexities, particularly in tasks such as design pattern detection for maintenance and assessment, as well as code refactoring for optimization and long-term sustainability. This growing inadequacy underscores the need for a paradigm shift in how such challenges are approached and resolved. This paper presents Analytical Software Engineering (ASE), a novel design paradigm aimed at balancing abstraction, tool accessibility, compatibility, and scalability. ASE enables effective modeling and resolution of complex software engineering problems. The paradigm is evaluated through two frameworks Behavioral-Structural Sequences (BSS) and Optimized Design Refactoring (ODR), both developed in accordance with ASE principles. BSS offers a compact, language-agnostic representation of codebases to facilitate precise design pattern detection. ODR unifies artifact and solution representations to optimize code refactoring via heuristic algorithms while eliminating iterative computational overhead. By providing a structured approach to software design challenges, ASE lays the groundwork for future research in encoding and analyzing complex software metrics. 

**Abstract (ZH)**: 随着现代软件系统的规模和复杂性的扩大，对其建模和表述所面临的挑战也日益复杂。传统方法往往在处理这些复杂性方面力有未逮，特别是在维护和评估中的设计模式检测以及优化和长期可持续性的代码重构任务中。这种日益突出的不足凸显了需要在处理和解决这些挑战方面进行范式转变的必要性。本文提出了分析软件工程（ASE），这是一种新的设计范式，旨在平衡抽象、工具易用性、兼容性和可扩展性。ASE 使得有效地建模和解决复杂的软件工程问题成为可能。该范式通过 Behavioral-Structural Sequences (BSS) 和 Optimized Design Refactoring (ODR) 两个框架进行评估，这两个框架均遵循 ASE 原理开发。BSS 提供了一种紧凑的语言无关的代码库表示方法，以促进精确的设计模式检测。ODR 统一了制品和解决方案的表示方法，通过启发式算法优化代码重构，同时消除迭代计算开销。通过为软件设计挑战提供结构化的解决方法，ASE 为编码和分析复杂软件度量的未来研究奠定了基础。 

---
# MARVEL: Multi-Agent RTL Vulnerability Extraction using Large Language Models 

**Title (ZH)**: MARVEL: 多Agent RTL漏洞提取使用大规模语言模型 

**Authors**: Luca Collini, Baleegh Ahmad, Joey Ah-kiow, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2505.11963)  

**Abstract**: Hardware security verification is a challenging and time-consuming task. For this purpose, design engineers may utilize tools such as formal verification, linters, and functional simulation tests, coupled with analysis and a deep understanding of the hardware design being inspected. Large Language Models (LLMs) have been used to assist during this task, either directly or in conjunction with existing tools. We improve the state of the art by proposing MARVEL, a multi-agent LLM framework for a unified approach to decision-making, tool use, and reasoning. MARVEL mimics the cognitive process of a designer looking for security vulnerabilities in RTL code. It consists of a supervisor agent that devises the security policy of the system-on-chips (SoCs) using its security documentation. It delegates tasks to validate the security policy to individual executor agents. Each executor agent carries out its assigned task using a particular strategy. Each executor agent may use one or more tools to identify potential security bugs in the design and send the results back to the supervisor agent for further analysis and confirmation. MARVEL includes executor agents that leverage formal tools, linters, simulation tests, LLM-based detection schemes, and static analysis-based checks. We test our approach on a known buggy SoC based on OpenTitan from the Hack@DATE competition. We find that 20 of the 48 issues reported by MARVEL pose security vulnerabilities. 

**Abstract (ZH)**: 硬件安全验证是一个具有挑战性和耗时的任务。为此，设计工程师可能会利用形式验证、linters和功能仿真测试等工具，结合分析和对被检查硬件设计的深刻理解。大规模语言模型（LLMs）已被用于协助这一任务，直接或与现有工具结合使用。我们通过提出一种多代理LLM框架MARVEL，改进了这一领域的研究水平，MARVEL提供了一种统一的方法来进行决策、工具使用和推理。MARVEL模仿了设计人员在 RTL代码中查找安全漏洞的认知过程。它包括一个监督代理，利用其安全文档制定系统级芯片（SoCs）的安全策略，并将任务委派给个体执行代理。每个执行代理使用特定策略执行其分配的任务。每个执行代理可以利用一种或多种工具来识别设计中的潜在安全漏洞，并将结果反馈给监督代理以进行进一步的分析和确认。MARVEL包括利用形式工具、linters、仿真测试、基于LLM的检测方案和静态分析检查的执行代理。我们基于Hack@DATE竞赛中的OpenTitan实现了一个已知有漏洞的SoC对其进行测试。我们发现MARVEL报告的48个问题中有20个确实存在安全漏洞。 

---
# Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning 

**Title (ZH)**: 探索损失重权化标准以增强语言模型去学习能力 

**Authors**: Puning Yang, Qizhou Wang, Zhuo Huang, Tongliang Liu, Chengqi Zhang, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.11953)  

**Abstract**: Loss reweighting has shown significant benefits for machine unlearning with large language models (LLMs). However, their exact functionalities are left unclear and the optimal strategy remains an open question, thus impeding the understanding and improvement of existing methodologies. In this paper, we identify two distinct goals of loss reweighting, namely, Saturation and Importance -- the former indicates that those insufficiently optimized data should be emphasized, while the latter stresses some critical data that are most influential for loss minimization. To study their usefulness, we design specific reweighting strategies for each goal and evaluate their respective effects on unlearning. We conduct extensive empirical analyses on well-established benchmarks, and summarize some important observations as follows: (i) Saturation enhances efficacy more than importance-based reweighting, and their combination can yield additional improvements. (ii) Saturation typically allocates lower weights to data with lower likelihoods, whereas importance-based reweighting does the opposite. (iii) The efficacy of unlearning is also largely influenced by the smoothness and granularity of the weight distributions. Based on these findings, we propose SatImp, a simple reweighting method that combines the advantages of both saturation and importance. Empirical results on extensive datasets validate the efficacy of our method, potentially bridging existing research gaps and indicating directions for future research. Our code is available at this https URL. 

**Abstract (ZH)**: 损失加权对大型语言模型机器遗忘的影响显著，但其具体功能尚不明确，最优策略仍存争议，从而阻碍了现有方法的理解与改进。在本文中，我们识别出损失加权的两个 distinct 目标，即饱和度和重要性——前者表明应突出那些优化不足的数据，而后者强调对损失最小化影响最大的关键数据。为了研究它们的有效性，我们为每个目标设计了特定的加权策略，并评估了它们对遗忘的影响。我们在广泛认可的基准上进行了广泛的经验分析，并总结了以下重要观察结果：(i) 饱和度比基于重要性的加权更能提高效果，它们的结合还可以带来额外的改进。(ii) 饱和度通常为较低概率的数据分配较低的权重，而基于重要性的加权则与此相反。(iii) 遗忘效果还受权重分布的平滑性和粒度的大幅影响。根据这些发现，我们提出了 SatImp，一种结合了饱和度和重要性优点的简单加权方法。在大量数据集上的实验证明了我们方法的有效性，有潜力填补现有研究空白并为未来研究指明方向。我们的代码可在该网页获取。 

---
# Let's have a chat with the EU AI Act 

**Title (ZH)**: 与欧盟AI法案进行交流 

**Authors**: Adam Kovari, Yasin Ghafourian, Csaba Hegedus, Belal Abu Naim, Kitti Mezei, Pal Varga, Markus Tauber  

**Link**: [PDF](https://arxiv.org/pdf/2505.11946)  

**Abstract**: As artificial intelligence (AI) regulations evolve and the regulatory landscape develops and becomes more complex, ensuring compliance with ethical guidelines and legal frameworks remains a challenge for AI developers. This paper introduces an AI-driven self-assessment chatbot designed to assist users in navigating the European Union AI Act and related standards. Leveraging a Retrieval-Augmented Generation (RAG) framework, the chatbot enables real-time, context-aware compliance verification by retrieving relevant regulatory texts and providing tailored guidance. By integrating both public and proprietary standards, it streamlines regulatory adherence, reduces complexity, and fosters responsible AI development. The paper explores the chatbot's architecture, comparing naive and graph-based RAG models, and discusses its potential impact on AI governance. 

**Abstract (ZH)**: 随着人工智能（AI）法规的演变和监管landscape的不断发展和完善，确保AI开发者遵守伦理指导原则和法律法规仍是一项挑战。本文 introduces一个基于AI的自助评估聊天机器人，旨在帮助用户导航《欧洲联盟AI法案》及相关标准。通过利用检索增强生成（RAG）框架，该聊天机器人实现实时、上下文相关的合规验证，通过检索相关监管文本并提供个性化指导。通过整合公共和专有标准，它简化了合规性，降低了复杂性，并促进了负责任的AI开发。本文探讨了聊天机器人的架构，并比较了朴素和图基线RAG模型，讨论了其对AI治理潜在影响。 

---
# Fine-Grained ECG-Text Contrastive Learning via Waveform Understanding Enhancement 

**Title (ZH)**: 基于波形理解增强的细粒度ECG-文本对比学习 

**Authors**: Haitao Li, Che Liu, Zhengyao Ding, Ziyi Liu, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11939)  

**Abstract**: Electrocardiograms (ECGs) are essential for diagnosing cardiovascular diseases. While previous ECG-text contrastive learning methods have shown promising results, they often overlook the incompleteness of the reports. Given an ECG, the report is generated by first identifying key waveform features and then inferring the final diagnosis through these features. Despite their importance, these waveform features are often not recorded in the report as intermediate results. Aligning ECGs with such incomplete reports impedes the model's ability to capture the ECG's waveform features and limits its understanding of diagnostic reasoning based on those features. To address this, we propose FG-CLEP (Fine-Grained Contrastive Language ECG Pre-training), which aims to recover these waveform features from incomplete reports with the help of large language models (LLMs), under the challenges of hallucinations and the non-bijective relationship between waveform features and diagnoses. Additionally, considering the frequent false negatives due to the prevalence of common diagnoses in ECGs, we introduce a semantic similarity matrix to guide contrastive learning. Furthermore, we adopt a sigmoid-based loss function to accommodate the multi-label nature of ECG-related tasks. Experiments on six datasets demonstrate that FG-CLEP outperforms state-of-the-art methods in both zero-shot prediction and linear probing across these datasets. 

**Abstract (ZH)**: 细粒度对比语言心电图预训练（FG-CLEP）：从不完整报告中恢复心电图波形特征 

---
# How can Diffusion Models Evolve into Continual Generators? 

**Title (ZH)**: 扩散模型如何演化为持续生成器？ 

**Authors**: Jingren Liu, Zhong Ji, Xiangyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11936)  

**Abstract**: While diffusion models have achieved remarkable success in static data generation, their deployment in streaming or continual learning (CL) scenarios faces a major challenge: catastrophic forgetting (CF), where newly acquired generative capabilities overwrite previously learned ones. To systematically address this, we introduce a formal Continual Diffusion Generation (CDG) paradigm that characterizes and redefines CL in the context of generative diffusion models. Prior efforts often adapt heuristic strategies from continual classification tasks but lack alignment with the underlying diffusion process. In this work, we develop the first theoretical framework for CDG by analyzing cross-task dynamics in diffusion-based generative modeling. Our analysis reveals that the retention and stability of generative knowledge across tasks are governed by three key consistency criteria: inter-task knowledge consistency (IKC), unconditional knowledge consistency (UKC), and label knowledge consistency (LKC). Building on these insights, we propose Continual Consistency Diffusion (CCD), a principled framework that integrates these consistency objectives into training via hierarchical loss terms $\mathcal{L}_{IKC}$, $\mathcal{L}_{UKC}$, and $\mathcal{L}_{LKC}$. This promotes effective knowledge retention while enabling the assimilation of new generative capabilities. Extensive experiments on four benchmark datasets demonstrate that CCD achieves state-of-the-art performance under continual settings, with substantial gains in Mean Fidelity (MF) and Incremental Mean Fidelity (IMF), particularly in tasks with rich cross-task knowledge overlap. 

**Abstract (ZH)**: 持续扩散生成对抗忘研究 

---
# Conversational Recommendation System using NLP and Sentiment Analysis 

**Title (ZH)**: 基于NLP和情感分析的对话型推荐系统 

**Authors**: Piyush Talegaonkar, Siddhant Hole, Shrinesh Kamble, Prashil Gulechha, Deepali Salapurkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11933)  

**Abstract**: In today's digitally-driven world, the demand for personalized and context-aware recommendations has never been greater. Traditional recommender systems have made significant strides in this direction, but they often lack the ability to tap into the richness of conversational data. This paper represents a novel approach to recommendation systems by integrating conversational insights into the recommendation process. The Conversational Recommender System integrates cutting-edge technologies such as deep learning, leveraging machine learning algorithms like Apriori for Association Rule Mining, Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Long Short-Term Memory (LTSM). Furthermore, sophisticated voice recognition technologies, including Hidden Markov Models (HMMs) and Dynamic Time Warping (DTW) algorithms, play a crucial role in accurate speech-to-text conversion, ensuring robust performance in diverse environments. The methodology incorporates a fusion of content-based and collaborative recommendation approaches, enhancing them with NLP techniques. This innovative integration ensures a more personalized and context-aware recommendation experience, particularly in marketing applications. 

**Abstract (ZH)**: 在数字化驱动的世界中，个性化和上下文感知推荐的需求前所未有。传统的推荐系统在这一领域取得了显著进展，但往往缺乏利用会话数据丰富性的能力。本文提出了一种将会话洞察力整合到推荐过程中的新颖方法，通过集成深度学习等先进技术，采用Apriori等关联规则挖掘算法、卷积神经网络（CNN）、循环神经网络（RNN）和长短期记忆网络（LSTM）。此外，先进的语音识别技术，包括隐马尔可夫模型（HMMs）和动态时间 warping（DTW）算法，在准确的语音转文本转换中发挥关键作用，确保在多种环境中的稳健性能。该方法整合了内容基推荐和协作推荐方法，并结合了自然语言处理技术。这种创新的集成确保了在营销应用中更加个性化和上下文感知的推荐体验。 

---
# The Logical Expressiveness of Temporal GNNs via Two-Dimensional Product Logics 

**Title (ZH)**: 基于二维乘积逻辑的时序GNNs的逻辑表達能力研究 

**Authors**: Marco Sälzer, Przemysław Andrzej Wałęga, Martin Lange  

**Link**: [PDF](https://arxiv.org/pdf/2505.11930)  

**Abstract**: In recent years, the expressive power of various neural architectures -- including graph neural networks (GNNs), transformers, and recurrent neural networks -- has been characterised using tools from logic and formal language theory. As the capabilities of basic architectures are becoming well understood, increasing attention is turning to models that combine multiple architectural paradigms. Among them particularly important, and challenging to analyse, are temporal extensions of GNNs, which integrate both spatial (graph-structure) and temporal (evolution over time) dimensions. In this paper, we initiate the study of logical characterisation of temporal GNNs by connecting them to two-dimensional product logics. We show that the expressive power of temporal GNNs depends on how graph and temporal components are combined. In particular, temporal GNNs that apply static GNNs recursively over time can capture all properties definable in the product logic of (past) propositional temporal logic PTL and the modal logic K. In contrast, architectures such as graph-and-time TGNNs and global TGNNs can only express restricted fragments of this logic, where the interaction between temporal and spatial operators is syntactically constrained. These results yield the first logical characterisations of temporal GNNs and establish new relative expressiveness results for temporal GNNs. 

**Abstract (ZH)**: 近年来，借助逻辑和形式语言理论工具，各种神经架构（包括图神经网络GNN、变压器和循环神经网络）的表达能力得到了表征。随着基本架构能力的逐渐明晰，越来越多的研究开始关注结合多种架构范式的模型。其中，尤其是在分析方面具有挑战性的时序扩展的GNN特别重要，它们结合了空间（图结构）和时间（随时间演变）维度。在本文中，我们通过将其与二维产品逻辑联系起来，开始了对时序GNN逻辑表征的研究。我们证明了时序GNN的表达能力取决于图和时序组件的结合方式。特别是，可以递归地在时间上应用静态GNN的时序GNN能够捕捉到过去命题时序逻辑PTL和模态逻辑K中定义的所有性质。相比之下，图和时间TGNN以及全局TGNN架构只能表达这些逻辑的限制片段，其中时间操作符和空间操作符的交互在语法上受到了限制。这些结果提供了时序GNN的第一个逻辑表征，并建立了新的相对表达能力结果。 

---
# SafeVid: Toward Safety Aligned Video Large Multimodal Models 

**Title (ZH)**: SafeVid: 向安全对齐的视频大型多模态模型方向 

**Authors**: Yixu Wang, Jiaxin Song, Yifeng Gao, Xin Wang, Yang Yao, Yan Teng, Xingjun Ma, Yingchun Wang, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11926)  

**Abstract**: As Video Large Multimodal Models (VLMMs) rapidly advance, their inherent complexity introduces significant safety challenges, particularly the issue of mismatched generalization where static safety alignments fail to transfer to dynamic video contexts. We introduce SafeVid, a framework designed to instill video-specific safety principles in VLMMs. SafeVid uniquely transfers robust textual safety alignment capabilities to the video domain by employing detailed textual video descriptions as an interpretive bridge, facilitating LLM-based rule-driven safety reasoning. This is achieved through a closed-loop system comprising: 1) generation of SafeVid-350K, a novel 350,000-pair video-specific safety preference dataset; 2) targeted alignment of VLMMs using Direct Preference Optimization (DPO); and 3) comprehensive evaluation via our new SafeVidBench benchmark. Alignment with SafeVid-350K significantly enhances VLMM safety, with models like LLaVA-NeXT-Video demonstrating substantial improvements (e.g., up to 42.39%) on SafeVidBench. SafeVid provides critical resources and a structured approach, demonstrating that leveraging textual descriptions as a conduit for safety reasoning markedly improves the safety alignment of VLMMs. We have made SafeVid-350K dataset (this https URL) publicly available. 

**Abstract (ZH)**: 随着视频大型多模态模型（VLMMs）的迅速发展，其固有的复杂性带来了重大的安全挑战，特别是静态安全对齐与动态视频场景之间不匹配的问题。我们提出了SafeVid框架，旨在将视频特定的安全原则应用于VLMMs。SafeVid通过使用详细的视频文本描述作为解释桥梁，将强大的文本安全对齐能力转移到视频领域，从而促进基于LLM的规则驱动安全推理。这一过程通过一个闭环系统实现：1）生成SafeVid-350K，一个新颖的包含350,000对视频特定安全偏好的数据集；2）使用直接偏好优化（DPO）对VLMMs进行目标对齐；3）通过我们的新SafeVidBench基准进行全面评估。与SafeVid-350K对齐显著提升了VLMM的安全性，如LLaVA-NeXT-Video模型在SafeVidBench上的表现大幅提升（例如高达42.39%）。SafeVid提供关键资源和结构化方法，证明将文本描述作为安全推理的通道显著提高了VLMMs的安全对齐。我们已公开发布了SafeVid-350K数据集（请参见原文链接）。 

---
# An Explanation of Intrinsic Self-Correction via Linear Representations and Latent Concepts 

**Title (ZH)**: 通过线性表示和潜在概念解释固有自我纠正机制 

**Authors**: Yu-Ting Lee, Hui-Ying Shih, Fu-Chieh Chang, Pei-Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11924)  

**Abstract**: We provide an explanation for the performance gains of intrinsic self-correction, a process where a language model iteratively refines its outputs without external feedback. More precisely, we investigate how prompting induces interpretable changes in hidden states and thus affects the output distributions. We hypothesize that each prompt-induced shift lies in a linear span of some linear representation vectors, naturally separating tokens based on individual concept alignment. Building around this idea, we give a mathematical formulation of self-correction and derive a concentration result for output tokens based on alignment magnitudes. Our experiments on text detoxification with zephyr-7b-sft reveal a substantial gap in the inner products of the prompt-induced shifts and the unembeddings of the top-100 most toxic tokens vs. those of the unembeddings of the bottom-100 least toxic tokens, under toxic instructions. This suggests that self-correction prompts enhance a language model's capability of latent concept recognition. Our analysis offers insights into the underlying mechanism of self-correction by characterizing how prompting works explainably. For reproducibility, our code is available. 

**Abstract (ZH)**: 我们提供了内在自我纠正性能提升的解释，这是一种语言模型在无外部反馈的情况下迭代 refinement 输出的过程。更具体地，我们研究了提示如何诱导隐藏状态的可解释变化，进而影响输出分布。我们假设每个提示诱导的变化位于某些线性表示向量的线性组合内，自然地根据个体概念对齐分离 token。基于这一想法，我们给出了自我纠正的数学表述，并根据对齐幅度推导了输出 token 的集中结果。我们在使用 zephyr-7b-sft 进行文本脱毒实验中发现，在有毒指令下，提示诱导的变化与前 100 个最毒 token 的 unembedding 的内积与后 100 个最不毒 token 的 unembedding 的内积之间存在显著差异。这表明自我纠正提示增强了语言模型对潜在概念识别的能力。我们的分析通过描述提示如何工作来揭示自我纠正的潜在机制。为了可再现性，我们的代码已公开。 

---
# Modèles de Substitution pour les Modèles à base d'Agents : Enjeux, Méthodes et Applications 

**Title (ZH)**: 基于代理的模型替代模型：挑战、方法和应用 

**Authors**: Paul Saves, Nicolas Verstaevel, Benoît Gaudou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11912)  

**Abstract**: Multi-agent simulations enables the modeling and analyses of the dynamic behaviors and interactions of autonomous entities evolving in complex environments. Agent-based models (ABM) are widely used to study emergent phenomena arising from local interactions. However, their high computational cost poses a significant challenge, particularly for large-scale simulations requiring extensive parameter exploration, optimization, or uncertainty quantification. The increasing complexity of ABM limits their feasibility for real-time decision-making and large-scale scenario analysis. To address these limitations, surrogate models offer an efficient alternative by learning approximations from sparse simulation data. These models provide cheap-to-evaluate predictions, significantly reducing computational costs while maintaining accuracy. Various machine learning techniques, including regression models, neural networks, random forests and Gaussian processes, have been applied to construct robust surrogates. Moreover, uncertainty quantification and sensitivity analysis play a crucial role in enhancing model reliability and interpretability.
This article explores the motivations, methods, and applications of surrogate modeling for ABM, emphasizing the trade-offs between accuracy, computational efficiency, and interpretability. Through a case study on a segregation model, we highlight the challenges associated with building and validating surrogate models, comparing different approaches and evaluating their performance. Finally, we discuss future perspectives on integrating surrogate models within ABM to improve scalability, explainability, and real-time decision support across various fields such as ecology, urban planning and economics. 

**Abstract (ZH)**: 多智能体模拟使得自主实体在复杂环境中的动态行为和相互作用建模与分析成为可能。基于代理模型（ABM）广泛用于研究由局部交互引发的涌现现象。然而，其高昂的计算成本对大规模模拟尤其是需要进行大量参数探索、优化或不确定性量化的要求提出了重大挑战。ABM的日益复杂性限制了其在实时决策和大规模情景分析中的实用性。为解决这些限制，代理模型通过学习稀疏模拟数据的近似值提供了高效替代方案。这些模型提供了低成本的预测，显著减少了计算成本同时保持了准确性。各种机器学习技术，包括回归模型、神经网络、随机森林和高斯过程，已被应用于构建健壮的代理模型。此外，不确定性量化和敏感性分析在提高模型可靠性和可解释性方面扮演着关键角色。本文探讨了代理模型在ABM中的动机、方法及应用，强调了准确度、计算效率和可解释性之间的权衡。通过一个隔离模型的案例研究，我们突显了构建和验证代理模型所面临的挑战，并对比了不同方法，评估了其性能。最后，我们讨论了代理模型在ABM中集成的未来视角，以提高跨生态学、城市规划和经济学等领域的大规模可扩展性、可解释性和实时决策支持能力。 

---
# K*-Means: A Parameter-free Clustering Algorithm 

**Title (ZH)**: K*-Means: 一种无参数聚类算法 

**Authors**: Louis Mahon, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2505.11904)  

**Abstract**: Clustering is a widely used and powerful machine learning technique, but its effectiveness is often limited by the need to specify the number of clusters, k, or by relying on thresholds that implicitly determine k. We introduce k*-means, a novel clustering algorithm that eliminates the need to set k or any other parameters. Instead, it uses the minimum description length principle to automatically determine the optimal number of clusters, k*, by splitting and merging clusters while also optimising the standard k-means objective. We prove that k*-means is guaranteed to converge and demonstrate experimentally that it significantly outperforms existing methods in scenarios where k is unknown. We also show that it is accurate in estimating k, and that empirically its runtime is competitive with existing methods, and scales well with dataset size. 

**Abstract (ZH)**: k*-means：一种无需指定簇数的新型聚类算法 

---
# AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning 

**Title (ZH)**: AdaCoT: Pareto-最优自适应链式思考触发机制 via 强化学习 

**Authors**: Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, Shuangzhi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11896)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities but often face challenges with tasks requiring sophisticated reasoning. While Chain-of-Thought (CoT) prompting significantly enhances reasoning, it indiscriminately generates lengthy reasoning steps for all queries, leading to substantial computational costs and inefficiency, especially for simpler inputs. To address this critical issue, we introduce AdaCoT (Adaptive Chain-of-Thought), a novel framework enabling LLMs to adaptively decide when to invoke CoT. AdaCoT framed adaptive reasoning as a Pareto optimization problem that seeks to balance model performance with the costs associated with CoT invocation (both frequency and computational overhead). We propose a reinforcement learning (RL) based method, specifically utilizing Proximal Policy Optimization (PPO), to dynamically control the CoT triggering decision boundary by adjusting penalty coefficients, thereby allowing the model to determine CoT necessity based on implicit query complexity. A key technical contribution is Selective Loss Masking (SLM), designed to counteract decision boundary collapse during multi-stage RL training, ensuring robust and stable adaptive triggering. Experimental results demonstrate that AdaCoT successfully navigates the Pareto frontier, achieving substantial reductions in CoT usage for queries not requiring elaborate reasoning. For instance, on our production traffic testset, AdaCoT reduced CoT triggering rates to as low as 3.18\% and decreased average response tokens by 69.06%, while maintaining high performance on complex tasks. 

**Abstract (ZH)**: AdaCoT：自适应链式推理 

---
# RLAP: A Reinforcement Learning Enhanced Adaptive Planning Framework for Multi-step NLP Task Solving 

**Title (ZH)**: RLAP：增强学习增强的自适应规划框架多步NLP任务解决 

**Authors**: Zepeng Ding, Dixuan Wang, Ziqin Luo, Guochao Jiang, Deqing Yang, Jiaqing Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11893)  

**Abstract**: Multi-step planning has been widely employed to enhance the performance of large language models (LLMs) on downstream natural language processing (NLP) tasks, which decomposes the original task into multiple subtasks and guide LLMs to solve them sequentially without additional training. When addressing task instances, existing methods either preset the order of steps or attempt multiple paths at each step. However, these methods overlook instances' linguistic features and rely on the intrinsic planning capabilities of LLMs to evaluate intermediate feedback and then select subtasks, resulting in suboptimal outcomes. To better solve multi-step NLP tasks with LLMs, in this paper we propose a Reinforcement Learning enhanced Adaptive Planning framework (RLAP). In our framework, we model an NLP task as a Markov decision process (MDP) and employ an LLM directly into the environment. In particular, a lightweight Actor model is trained to estimate Q-values for natural language sequences consisting of states and actions through reinforcement learning. Therefore, during sequential planning, the linguistic features of each sequence in the MDP can be taken into account, and the Actor model interacts with the LLM to determine the optimal order of subtasks for each task instance. We apply RLAP on three different types of NLP tasks and conduct extensive experiments on multiple datasets to verify RLAP's effectiveness and robustness. 

**Abstract (ZH)**: 多步规划增强的大语言模型在下游自然语言处理任务中的适应性规划框架（基于强化学习） 

---
# Mobile-Bench-v2: A More Realistic and Comprehensive Benchmark for VLM-based Mobile Agents 

**Title (ZH)**: Mobile-Bench-v2: 一种更加现实且全面的基于VLM的移动代理基准测试 

**Authors**: Weikai Xu, Zhizheng Jiang, Yuxuan Liu, Wei Liu, Jian Luan, Yuanchun Li, Yunxin Liu, Bin Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.11891)  

**Abstract**: VLM-based mobile agents are increasingly popular due to their capabilities to interact with smartphone GUIs and XML-structured texts and to complete daily tasks. However, existing online benchmarks struggle with obtaining stable reward signals due to dynamic environmental changes. Offline benchmarks evaluate the agents through single-path trajectories, which stands in contrast to the inherently multi-solution characteristics of GUI tasks. Additionally, both types of benchmarks fail to assess whether mobile agents can handle noise or engage in proactive interactions due to a lack of noisy apps or overly full instructions during the evaluation process. To address these limitations, we use a slot-based instruction generation method to construct a more realistic and comprehensive benchmark named Mobile-Bench-v2. Mobile-Bench-v2 includes a common task split, with offline multi-path evaluation to assess the agent's ability to obtain step rewards during task execution. It contains a noisy split based on pop-ups and ads apps, and a contaminated split named AITZ-Noise to formulate a real noisy environment. Furthermore, an ambiguous instruction split with preset Q\&A interactions is released to evaluate the agent's proactive interaction capabilities. We conduct evaluations on these splits using the single-agent framework AppAgent-v1, the multi-agent framework Mobile-Agent-v2, as well as other mobile agents such as UI-Tars and OS-Atlas. Code and data are available at this https URL. 

**Abstract (ZH)**: 基于VLM的移动代理由于能够与智能手机GUI和XML结构化的文本交互并完成日常任务而越来越受欢迎。然而，现有的在线基准由于动态环境变化难以获得稳定的奖励信号。离线基准通过单路径轨迹评估代理，这与GUI任务固有的多解特性相悖。此外，这两种基准都未能评估移动代理是否能够处理噪音或进行主动交互，因为在评估过程中缺乏含有噪音的应用程序或操作指令过于详细。为解决这些限制，我们采用了基于槽位的指令生成方法构建了一个更加现实和综合的基准——Mobile-Bench-v2。Mobile-Bench-v2包括一个常见的任务划分，采用多路径离线评估以评估代理在执行任务时获得步进奖励的能力。它包含一个基于弹出窗口和广告应用的噪音划分，以及一个名为AITZ-Noise的受污染划分，以形成实际的噪音环境。此外，还发布了一个模糊指令划分，包含预设的问答交互，以评估代理的主动交互能力。我们使用单代理框架AppAgent-v1、多代理框架Mobile-Agent-v2以及其他移动代理UI-Tars和OS-Atlas对这些划分进行了评估。代码和数据可以在以下链接获取。 

---
# Exploring the Potential of SSL Models for Sound Event Detection 

**Title (ZH)**: 探索SSL模型在声事件检测中的潜力 

**Authors**: Hanfang Cui, Longfei Song, Li Li, Dongxing Xu, Yanhua Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.11889)  

**Abstract**: Self-supervised learning (SSL) models offer powerful representations for sound event detection (SED), yet their synergistic potential remains underexplored. This study systematically evaluates state-of-the-art SSL models to guide optimal model selection and integration for SED. We propose a framework that combines heterogeneous SSL representations (e.g., BEATs, HuBERT, WavLM) through three fusion strategies: individual SSL embedding integration, dual-modal fusion, and full aggregation. Experiments on the DCASE 2023 Task 4 Challenge reveal that dual-modal fusion (e.g., CRNN+BEATs+WavLM) achieves complementary performance gains, while CRNN+BEATs alone delivers the best results among individual SSL models. We further introduce normalized sound event bounding boxes (nSEBBs), an adaptive post-processing method that dynamically adjusts event boundary predictions, improving PSDS1 by up to 4% for standalone SSL models. These findings highlight the compatibility and complementarity of SSL architectures, providing guidance for task-specific fusion and robust SED system design. 

**Abstract (ZH)**: 自监督学习模型在声事件检测中的协同潜力尚未充分探索。本研究系统评估了前沿的自监督学习模型，以指导声事件检测中的最优模型选择与集成。我们提出了一种框架，通过三种融合策略（个体自监督嵌入集成、双模态融合和全聚合）结合异构的自监督学习表示（例如，BEATs、HuBERT、WavLM）。DCASE 2023 任务4 挑战实验表明，双模态融合（例如，CRNN+BEATs+WavLM）实现了补充性能提升，而仅使用CRNN+BEATs获得个体自监督模型中的最佳结果。此外，我们引入了归一化声事件边框（nSEBBs），这是一种自适应后处理方法，可动态调整事件边界预测，对于独立的自监督学习模型，可提高PSDS1性能最多4%。这些发现突显了自监督学习架构的兼容性和互补性，提供了针对特定任务的融合指导和鲁棒声事件检测系统设计。 

---
# Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks 

**Title (ZH)**: 重访残差连接：正交更新以实现稳定高效的深层网络 

**Authors**: Giyeong Oh, Woohyun Cho, Siyeol Kim, Suhwan Choi, Younjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11881)  

**Abstract**: Residual connections are pivotal for deep neural networks, enabling greater depth by mitigating vanishing gradients. However, in standard residual updates, the module's output is directly added to the input stream. This can lead to updates that predominantly reinforce or modulate the existing stream direction, potentially underutilizing the module's capacity for learning entirely novel features. In this work, we introduce Orthogonal Residual Update: we decompose the module's output relative to the input stream and add only the component orthogonal to this stream. This design aims to guide modules to contribute primarily new representational directions, fostering richer feature learning while promoting more efficient training. We demonstrate that our orthogonal update strategy improves generalization accuracy and training stability across diverse architectures (ResNetV2, Vision Transformers) and datasets (CIFARs, TinyImageNet, ImageNet-1k), achieving, for instance, a +4.3\%p top-1 accuracy gain for ViT-B on ImageNet-1k. 

**Abstract (ZH)**: 残差连接对于深层神经网络至关重要，通过减轻梯度消失问题，使网络能够具备更深的结构。然而，在标准的残差更新中，模块的输出直接加到输入流中。这可能导致更新主要强化或调整现有的流方向，可能未能充分利用模块学习全新特征的能力。在本文中，我们引入了正交残差更新：我们将模块的输出相对于输入流进行分解，并仅添加与该流正交的部分。该设计旨在引导模块主要贡献新的表示方向，从而促进更丰富的特征学习，同时促进更高效的训练。我们证明，我们的正交更新策略在多种架构（ResNetV2、Vision Transformers）和数据集（CIFARs、TinyImageNet、ImageNet-1k）上提高了泛化准确性和训练稳定性，例如，在ImageNet-1k上实现了ViT-B的+4.3%top-1准确率提升。 

---
# AdaptMol: Adaptive Fusion from Sequence String to Topological Structure for Few-shot Drug Discovery 

**Title (ZH)**: AdaptMol: 从序列字符串到拓扑结构的自适应融合在少数几次药物发现中的应用 

**Authors**: Yifan Dai, Xuanbai Ren, Tengfei Ma, Qipeng Yan, Yiping Liu, Yuansheng Liu, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11878)  

**Abstract**: Accurate molecular property prediction (MPP) is a critical step in modern drug development. However, the scarcity of experimental validation data poses a significant challenge to AI-driven research paradigms. Under few-shot learning scenarios, the quality of molecular representations directly dictates the theoretical upper limit of model performance. We present AdaptMol, a prototypical network integrating Adaptive multimodal fusion for Molecular representation. This framework employs a dual-level attention mechanism to dynamically integrate global and local molecular features derived from two modalities: SMILES sequences and molecular graphs. (1) At the local level, structural features such as atomic interactions and substructures are extracted from molecular graphs, emphasizing fine-grained topological information; (2) At the global level, the SMILES sequence provides a holistic representation of the molecule. To validate the necessity of multimodal adaptive fusion, we propose an interpretable approach based on identifying molecular active substructures to demonstrate that multimodal adaptive fusion can efficiently represent molecules. Extensive experiments on three commonly used benchmarks under 5-shot and 10-shot settings demonstrate that AdaptMol achieves state-of-the-art performance in most cases. The rationale-extracted method guides the fusion of two modalities and highlights the importance of both modalities. 

**Abstract (ZH)**: 准确的分子属性预测（MPP）是现代药物开发中的关键步骤。然而，实验验证数据的匮乏对基于AI的研究范式构成了重大挑战。在少样本学习场景下，分子表示的质量直接决定了模型性能的理论上限。我们提出AdaptMol，这是一种结合自适应多模态融合的原型网络，该框架采用双层注意力机制动态集成来自两种模态（SMILES序列和分子图）的全局和局部分子特征。在局部层面，结构特征如原子相互作用和子结构从分子图中提取，强调精细的拓扑信息；在全局层面，SMILES序列提供了分子的整体表示。为验证多模态自适应融合的必要性，我们提出了一种基于识别分子活性子结构的可解释方法，证明多模态自适应融合可以有效表示分子。在5-shot和10-shot设置下的三个常用基准上的 extensive 实验表明，AdaptMol 在大多数情况下达到了最先进的性能。提取的推理方法指导了两种模态的融合，并突出了两种模态的重要性。 

---
# Learning Pareto-Optimal Rewards from Noisy Preferences: A Framework for Multi-Objective Inverse Reinforcement Learning 

**Title (ZH)**: 基于多目标逆强化学习的帕累托最优奖励从噪声偏好中学习框架 

**Authors**: Kalyan Cherukuri, Aarav Lala  

**Link**: [PDF](https://arxiv.org/pdf/2505.11864)  

**Abstract**: As generative agents become increasingly capable, alignment of their behavior with complex human values remains a fundamental challenge. Existing approaches often simplify human intent through reduction to a scalar reward, overlooking the multi-faceted nature of human feedback. In this work, we introduce a theoretical framework for preference-based Multi-Objective Inverse Reinforcement Learning (MO-IRL), where human preferences are modeled as latent vector-valued reward functions. We formalize the problem of recovering a Pareto-optimal reward representation from noisy preference queries and establish conditions for identifying the underlying multi-objective structure. We derive tight sample complexity bounds for recovering $\epsilon$-approximations of the Pareto front and introduce a regret formulation to quantify suboptimality in this multi-objective setting. Furthermore, we propose a provably convergent algorithm for policy optimization using preference-inferred reward cones. Our results bridge the gap between practical alignment techniques and theoretical guarantees, providing a principled foundation for learning aligned behaviors in a high-dimension and value-pluralistic environment. 

**Abstract (ZH)**: 随着生成代理的能力不断增强，其行为与复杂人类价值的对齐仍然是一个基本挑战。现有方法往往通过将人类意图简化为标量奖励来忽视人类反馈的多维度性质。在本文中，我们提出了一种基于偏好的多目标逆强化学习（MO-IRL）的理论框架，其中人类偏好被建模为潜在向量值奖励函数。我们形式化了从噪声偏好查询中恢复帕累托最优奖励表示的问题，并建立了识别潜在多目标结构的条件。我们推导出恢复$\epsilon$-逼近帕累托前沿所需的紧样本复杂性界，并引入了后悔公式来量化此多目标设置中的次优性。此外，我们提出了一种可证明收敛的算法用于使用偏好推断的奖励锥进行策略优化。我们的结果将实际对齐技术与理论保证相结合，为在高维和价值观多元化的环境中学习对齐行为提供了规范的基础。 

---
# Q-Policy: Quantum-Enhanced Policy Evaluation for Scalable Reinforcement Learning 

**Title (ZH)**: 量子增强策略评估：可扩展强化学习中的Q-政策 

**Authors**: Kalyan Cherukuri, Aarav Lala, Yash Yardi  

**Link**: [PDF](https://arxiv.org/pdf/2505.11862)  

**Abstract**: We propose Q-Policy, a hybrid quantum-classical reinforcement learning (RL) framework that mathematically accelerates policy evaluation and optimization by exploiting quantum computing primitives. Q-Policy encodes value functions in quantum superposition, enabling simultaneous evaluation of multiple state-action pairs via amplitude encoding and quantum parallelism. We introduce a quantum-enhanced policy iteration algorithm with provable polynomial reductions in sample complexity for the evaluation step, under standard assumptions. To demonstrate the technical feasibility and theoretical soundness of our approach, we validate Q-Policy on classical emulations of small discrete control tasks. Due to current hardware and simulation limitations, our experiments focus on showcasing proof-of-concept behavior rather than large-scale empirical evaluation. Our results support the potential of Q-Policy as a theoretical foundation for scalable RL on future quantum devices, addressing RL scalability challenges beyond classical approaches. 

**Abstract (ZH)**: Q-Policy：一种利用量子计算加速策略评估与优化的混合量子-经典强化学习框架 

---
# On Membership Inference Attacks in Knowledge Distillation 

**Title (ZH)**: 知识蒸馏中的成员 inference 攻击 

**Authors**: Ziyao Cui, Minxing Zhang, Jian Pei  

**Link**: [PDF](https://arxiv.org/pdf/2505.11837)  

**Abstract**: Nowadays, Large Language Models (LLMs) are trained on huge datasets, some including sensitive information. This poses a serious privacy concern because privacy attacks such as Membership Inference Attacks (MIAs) may detect this sensitive information. While knowledge distillation compresses LLMs into efficient, smaller student models, its impact on privacy remains underexplored. In this paper, we investigate how knowledge distillation affects model robustness against MIA. We focus on two questions. First, how is private data protected in teacher and student models? Second, how can we strengthen privacy preservation against MIAs in knowledge distillation? Through comprehensive experiments, we show that while teacher and student models achieve similar overall MIA accuracy, teacher models better protect member data, the primary target of MIA, whereas student models better protect non-member data. To address this vulnerability in student models, we propose 5 privacy-preserving distillation methods and demonstrate that they successfully reduce student models' vulnerability to MIA, with ensembling further stabilizing the robustness, offering a reliable approach for distilling more secure and efficient student models. Our implementation source code is available at this https URL. 

**Abstract (ZH)**: 现今，大规模语言模型（LLMs）在包含敏感信息的大规模数据集上进行训练。这引发了严重的隐私 concerns，因为成员隶属推理攻击（MIAs）可能探测到这些敏感信息。虽然知识蒸馏可以将LLMs压缩成高效的较小规模学生模型，但其对学生隐私的影响仍缺乏深入探索。本文探讨知识蒸馏如何影响模型对抗MIAs的robustness。我们重点关注两个问题：首先，教师模型和学生模型如何保护私有数据？其次，如何在知识蒸馏过程中加强对抗MIAs的隐私保护？通过全面的实验，我们发现虽然教师模型和学生模型在整体MIAs准确率上相似，但教师模型更能保护成员数据，这是MIAs的主要目标；而学生模型则更好地保护非成员数据。为解决学生模型的这一脆弱性，我们提出5种隐私保护蒸馏方法，并证明这些方法成功降低了学生模型对MIAs的脆弱性，而集成进一步提高了robustness的稳定性，为蒸馏更安全高效的学生模型提供了可靠的途径。我们的实现源代码可在此httpsURL访问。 

---
# SplInterp: Improving our Understanding and Training of Sparse Autoencoders 

**Title (ZH)**: SplInterp: 提升我们对稀疏自编码器的理解和训练 

**Authors**: Jeremy Budd, Javier Ideami, Benjamin Macdowall Rynne, Keith Duggar, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2505.11836)  

**Abstract**: Sparse autoencoders (SAEs) have received considerable recent attention as tools for mechanistic interpretability, showing success at extracting interpretable features even from very large LLMs. However, this research has been largely empirical, and there have been recent doubts about the true utility of SAEs. In this work, we seek to enhance the theoretical understanding of SAEs, using the spline theory of deep learning. By situating SAEs in this framework: we discover that SAEs generalise ``$k$-means autoencoders'' to be piecewise affine, but sacrifice accuracy for interpretability vs. the optimal ``$k$-means-esque plus local principal component analysis (PCA)'' piecewise affine autoencoder. We characterise the underlying geometry of (TopK) SAEs using power diagrams. And we develop a novel proximal alternating method SGD (PAM-SGD) algorithm for training SAEs, with both solid theoretical foundations and promising empirical results in MNIST and LLM experiments, particularly in sample efficiency and (in the LLM setting) improved sparsity of codes. All code is available at: this https URL 

**Abstract (ZH)**: 稀疏自编码器（SAEs）近年来因其在机制可解释性方面的工具作用而受到广泛关注，显示出即使在非常大的LLM中也能提取可解释特征的成功案例。然而，这项研究主要基于经验，近期对SAEs的实际用途产生了怀疑。在本文中，我们旨在通过使用深度学习的样条理论来增强对SAEs的理论理解。通过将SAEs置于这一框架中，我们发现SAEs推广了“$k$-means自编码器”成为分段仿射模型，但与最优的“$k$-means-esque加上局部主成分分析（PCA）”分段仿射自编码器相比，牺牲了准确性以换取可解释性。我们使用幂图（power diagrams）来表征（TopK）SAEs的潜在几何结构。并开发了一种新颖的邻近交替优化随机梯度下降（PAM-SGD）算法用于训练SAEs，该算法具有坚实理论基础并在MNIST和LLM实验中表现出令人鼓舞的经验结果，特别是在样本效率以及（在LLM设置中）代码稀疏性方面。所有代码均可在以下链接获取：this https URL。 

---
# Multilingual Collaborative Defense for Large Language Models 

**Title (ZH)**: 多语言协同防御大规模语言模型 

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11835)  

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的鲁棒性和安全性的研究已成为一个重要的研究领域。一个值得注意的漏洞是通过将有害查询翻译成稀有或代表性不足的语言来规避LLM的安全措施，这是一种简单而有效的方法来“破解”这些模型。尽管存在日益增长的关切，但对于多语言场景下的LLM保护研究仍然有限，这突显了增强多语言安全性的迫切需求。在本工作中，我们探讨了不同语言中各种攻击特征之间的相关性，并提出了一种名为多语言协作防护（MCD）的新型学习方法，该方法自动优化连续的软安全性提示，以促进LLM的多语言保护。MCD方法具有三个优势：首先，它能有效提高多语言保护性能。其次，MCD保持了强大的泛化能力，同时将拒绝率降至最低。第三，MCD缓解了由于LLM训练语料库不平衡导致的语言安全性不对齐问题。为了评估MCD的有效性，我们手动构建了常用的“破解”基准的多语言版本，如MaliciousInstruct和AdvBench，以评估各种保护方法。此外，我们还在欠代表语言（零样本）中引入了这些数据集，以验证MCD的语言迁移能力。结果表明，MCD在防止多语言“破解”尝试方面优于现有方法，并且表现出强大的语言迁移能力。我们的代码可在以下网址获取。 

---
# CoT-Vid: Dynamic Chain-of-Thought Routing with Self Verification for Training-Free Video Reasoning 

**Title (ZH)**: CoT-Vid: 动态链式思考路由与自我验证在无训练数据的视频推理中的应用 

**Authors**: Hongbo Jin, Ruyang Liu, Wenhao Zhang, Guibo Luo, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11830)  

**Abstract**: System2 reasoning is developing rapidly these days with the emergence of Deep- Thinking Models and chain-of-thought technology, which has become a centralized discussion point in the AI community. However, there is a relative gap in the research on complex video reasoning at present. In this work, we propose CoT-Vid, a novel training-free paradigm for the video domain with a multistage complex reasoning design. Distinguishing from existing video LLMs, which rely heavily on perceptual abilities, it achieved surprising performance gain with explicit reasoning mechanism. The paradigm consists of three main components: dynamic inference path routing, problem decoupling strategy, and video self-consistency verification. In addition, we propose a new standard for categorization of video questions. CoT- Vid showed outstanding results on a wide range of benchmarks, and outperforms its base model by 9.3% on Egochema and 5.6% on VideoEspresso, rivalling or even surpassing larger and proprietary models, such as GPT-4V, GPT-4o and Gemini-1.5-flash. Our codebase will be publicly available soon. 

**Abstract (ZH)**: 基于链式推理的视频理解模型CoT-Vid 

---
# Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning 

**Title (ZH)**: 并非所有思维生成均等：基于多轮强化学习的高效LLM推理 

**Authors**: Yansong Ning, Wei Li, Jun Fang, Naiqiang Tan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11827)  

**Abstract**: Compressing long chain-of-thought (CoT) from large language models (LLMs) is an emerging strategy to improve the reasoning efficiency of LLMs. Despite its promising benefits, existing studies equally compress all thoughts within a long CoT, hindering more concise and effective reasoning. To this end, we first investigate the importance of different thoughts by examining their effectiveness and efficiency in contributing to reasoning through automatic long CoT chunking and Monte Carlo rollouts. Building upon the insights, we propose a theoretically bounded metric to jointly measure the effectiveness and efficiency of different thoughts. We then propose Long$\otimes$Short, an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts. Specifically, we begin by synthesizing a small amount of cold-start data to fine-tune LLMs for long-thought and short-thought reasoning styles, respectively. Furthermore, we propose a synergizing-oriented multi-turn reinforcement learning, focusing on the model self-evolution and collaboration between long-thought and short-thought LLMs. Experimental results show that our method enables Qwen2.5-7B and Llama3.1-8B to achieve comparable performance compared to DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B, while reducing token length by over 80% across the MATH500, AIME24/25, AMC23, and GPQA Diamond benchmarks. Our data and code are available at this https URL. 

**Abstract (ZH)**: 压缩大型语言模型中的长链推理（CoT）以提高其推理效率是一种新兴策略。尽管这一方法具有潜在的优势，现有的研究同样压缩长链推理中的所有想法，这阻碍了更为简洁和有效的推理。为此，我们首先通过自动长链推理分块和蒙特卡洛展开，研究不同想法在贡献推理中的效果和效率，以探讨其重要性。基于这些洞察，我们提出了一种理论上受限的度量标准，以联合衡量不同想法的效果和效率。随后，我们提出了一种高效的推理框架——Long$\otimes$Short，使两个大型语言模型能够协作解决问题：一个长想法模型更有效地生成重要想法，另一个短想法模型则高效生成剩余的想法。具体而言，我们首先生成少量冷启动数据，分别微调大型语言模型进行长想法和短想法推理。此外，我们提出了一种以协同进化为重点的多轮强化学习，关注模型的自我进化和长想法与短想法大型语言模型之间的合作。实验结果显示，我们的方法使Qwen2.5-7B和Llama3.1-8B在MATH500、AIME24/25、AMC23以及GPQA钻石基准测试中达到了与DeepSeek-R1-Distill-Qwen-7B和DeepSeek-R1-Distill-Llama-8B相当的性能，同时将令牌长度减少了超过80%。我们的数据和代码可在以下链接获取。 

---
# Bootstrapping Diffusion: Diffusion Model Training Leveraging Partial and Corrupted Data 

**Title (ZH)**: 基于抽样 bootstrapping 的扩散模型训练：利用部分数据和受损数据 

**Authors**: Xudong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.11825)  

**Abstract**: Training diffusion models requires large datasets. However, acquiring large volumes of high-quality data can be challenging, for example, collecting large numbers of high-resolution images and long videos. On the other hand, there are many complementary data that are usually considered corrupted or partial, such as low-resolution images and short videos. Other examples of corrupted data include videos that contain subtitles, watermarks, and logos. In this study, we investigate the theoretical problem of whether the above partial data can be utilized to train conventional diffusion models. Motivated by our theoretical analysis in this study, we propose a straightforward approach of training diffusion models utilizing partial data views, where we consider each form of complementary data as a view of conventional data. Our proposed approach first trains one separate diffusion model for each individual view, and then trains a model for predicting the residual score function. We prove generalization error bounds, which show that the proposed diffusion model training approach can achieve lower generalization errors if proper regularizations are adopted in the residual score function training. In particular, we prove that the difficulty in training the residual score function scales proportionally with the signal correlations not captured by partial data views. Consequently, the proposed approach achieves near first-order optimal data efficiency. 

**Abstract (ZH)**: 利用部分数据训练扩散模型的理论问题研究 

---
# Search-Based Correction of Reasoning Chains for Language Models 

**Title (ZH)**: 基于搜索的逻辑推理链修正方法 

**Authors**: Minsu Kim, Jean-Pierre Falet, Oliver E. Richardson, Xiaoyin Chen, Moksh Jain, Sungjin Ahn, Sungsoo Ahn, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2505.11824)  

**Abstract**: Chain-of-Thought (CoT) reasoning has advanced the capabilities and transparency of language models (LMs); however, reasoning chains can contain inaccurate statements that reduce performance and trustworthiness. To address this, we introduce a new self-correction framework that augments each reasoning step in a CoT with a latent variable indicating its veracity, enabling modeling of all possible truth assignments rather than assuming correctness throughout. To efficiently explore this expanded space, we introduce Search Corrector, a discrete search algorithm over boolean-valued veracity assignments. It efficiently performs otherwise intractable inference in the posterior distribution over veracity assignments by leveraging the LM's joint likelihood over veracity and the final answer as a proxy reward. This efficient inference-time correction method facilitates supervised fine-tuning of an Amortized Corrector by providing pseudo-labels for veracity. The Amortized Corrector generalizes self-correction, enabling accurate zero-shot veracity inference in novel contexts. Empirical results demonstrate that Search Corrector reliably identifies errors in logical (ProntoQA) and mathematical reasoning (GSM8K) benchmarks. The Amortized Corrector achieves comparable zero-shot accuracy and improves final answer accuracy by up to 25%. 

**Abstract (ZH)**: Chain-of-Thought推理的自校正框架：提高语言模型的性能与可信度 

---
# SGD-Mix: Enhancing Domain-Specific Image Classification with Label-Preserving Data Augmentation 

**Title (ZH)**: SGD-Mix: 保留标签数据增强促进领域特定图像分类 

**Authors**: Yixuan Dong, Fang-Yi Su, Jung-Hsien Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11813)  

**Abstract**: Data augmentation for domain-specific image classification tasks often struggles to simultaneously address diversity, faithfulness, and label clarity of generated data, leading to suboptimal performance in downstream tasks. While existing generative diffusion model-based methods aim to enhance augmentation, they fail to cohesively tackle these three critical aspects and often overlook intrinsic challenges of diffusion models, such as sensitivity to model characteristics and stochasticity under strong transformations. In this paper, we propose a novel framework that explicitly integrates diversity, faithfulness, and label clarity into the augmentation process. Our approach employs saliency-guided mixing and a fine-tuned diffusion model to preserve foreground semantics, enrich background diversity, and ensure label consistency, while mitigating diffusion model limitations. Extensive experiments across fine-grained, long-tail, few-shot, and background robustness tasks demonstrate our method's superior performance over state-of-the-art approaches. 

**Abstract (ZH)**: 数据增强在领域特定图像分类任务中往往难以同时解决生成数据的多样性、忠实度和标签清晰度问题，导致下游任务表现欠佳。虽然现有的基于生成性扩散模型的方法旨在增强数据增强效果，但它们未能综合解决这三个关键方面，并且常常忽视扩散模型固有的挑战，如对模型特征的敏感性和在高强度变换下的随机性。本文提出了一种新的框架，明确将多样性、忠实度和标签清晰度整合到增强过程中。我们的方法采用基于显著性引导的混合和微调后的扩散模型来保留前景语义、丰富背景多样性并确保标签一致性，同时缓解扩散模型的局限性。通过细粒度、长尾、少样本及背景鲁棒性等多种任务的广泛实验，展示了我们方法在现有先进技术上的优越性能。 

---
# Retrospex: Language Agent Meets Offline Reinforcement Learning Critic 

**Title (ZH)**: Retrospex: 语言代理 Meet 离线强化学习评论家 

**Authors**: Yufei Xiang, Yiqun Shen, Yeqin Zhang, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11807)  

**Abstract**: Large Language Models (LLMs) possess extensive knowledge and commonsense reasoning capabilities, making them valuable for creating powerful agents. However, existing LLM agent frameworks have not fully utilized past experiences for improvement. This work introduces a new LLM-based agent framework called Retrospex, which addresses this challenge by analyzing past experiences in depth. Unlike previous approaches, Retrospex does not directly integrate experiences into the LLM's context. Instead, it combines the LLM's action likelihood with action values estimated by a Reinforcement Learning (RL) Critic, which is trained on past experiences through an offline ''retrospection'' process. Additionally, Retrospex employs a dynamic action rescoring mechanism that increases the importance of experience-based values for tasks that require more interaction with the environment. We evaluate Retrospex in ScienceWorld, ALFWorld and Webshop environments, demonstrating its advantages over strong, contemporary baselines. 

**Abstract (ZH)**: Retrospex：一种通过深度分析过往经验的大型语言模型基于代理框架 

---
# Are vision language models robust to uncertain inputs? 

**Title (ZH)**: 视觉语言模型对不确定输入 robust 吗？ 

**Authors**: Xi Wang, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2505.11804)  

**Abstract**: Robustness against uncertain and ambiguous inputs is a critical challenge for deep learning models. While recent advancements in large scale vision language models (VLMs, e.g. GPT4o) might suggest that increasing model and training dataset size would mitigate this issue, our empirical evaluation shows a more complicated picture. Testing models using two classic uncertainty quantification tasks, anomaly detection and classification under inherently ambiguous conditions, we find that newer and larger VLMs indeed exhibit improved robustness compared to earlier models, but still suffer from a tendency to strictly follow instructions, often causing them to hallucinate confident responses even when faced with unclear or anomalous inputs. Remarkably, for natural images such as ImageNet, this limitation can be overcome without pipeline modifications: simply prompting models to abstain from uncertain predictions enables significant reliability gains, achieving near-perfect robustness in several settings. However, for domain-specific tasks such as galaxy morphology classification, a lack of specialized knowledge prevents reliable uncertainty estimation. Finally, we propose a novel mechanism based on caption diversity to reveal a model's internal uncertainty, enabling practitioners to predict when models will successfully abstain without relying on labeled data. 

**Abstract (ZH)**: 针对不确定和模糊输入的鲁棒性是深度学习模型的一个关键挑战。尽管大规模视觉语言模型（VLMs，例如GPT4o）的最新进展可能表明增加模型和训练数据集的大小能够缓解这一问题，我们的实证评估显示情况更为复杂。通过使用两种经典的不确定性量化任务——异常检测和在固有模糊条件下进行分类——我们发现，较新的、更大的VLMs确实比早期的模型表现出了更好的鲁棒性，但它们仍然容易受指示驱动的影响，常常在面对不明确或异常输入时产生自信但错误的响应。对于如ImageNet这样的自然图像，这一限制可以通过简单的提示模型避免不确定性的预测来克服，从而在多个场景中实现接近完美的可靠性。然而，对于如星系形态分类这样的特定领域任务，缺乏专门知识阻碍了可靠的不确定性估计。最后，我们提出了一种基于标题多样性的新机制，以揭示模型内部的不确定性，使实践者能够在不需要标记数据的情况下预测模型何时能成功避免不确定性。 

---
# Diffmv: A Unified Diffusion Framework for Healthcare Predictions with Random Missing Views and View Laziness 

**Title (ZH)**: DiffMV：一种用于医疗预测的统一扩散框架，应对随机缺失视图和视图懒惰性问题 

**Authors**: Chuang Zhao, Hui Tang, Hongke Zhao, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11802)  

**Abstract**: Advanced healthcare predictions offer significant improvements in patient outcomes by leveraging predictive analytics. Existing works primarily utilize various views of Electronic Health Record (EHR) data, such as diagnoses, lab tests, or clinical notes, for model training. These methods typically assume the availability of complete EHR views and that the designed model could fully leverage the potential of each view. However, in practice, random missing views and view laziness present two significant challenges that hinder further improvements in multi-view utilization. To address these challenges, we introduce Diffmv, an innovative diffusion-based generative framework designed to advance the exploitation of multiple views of EHR data. Specifically, to address random missing views, we integrate various views of EHR data into a unified diffusion-denoising framework, enriched with diverse contextual conditions to facilitate progressive alignment and view transformation. To mitigate view laziness, we propose a novel reweighting strategy that assesses the relative advantages of each view, promoting a balanced utilization of various data views within the model. Our proposed strategy achieves superior performance across multiple health prediction tasks derived from three popular datasets, including multi-view and multi-modality scenarios. 

**Abstract (ZH)**: 基于扩散生成模型的多视图电子健康记录数据高级预测改进患者 outcome 

---
# CL-CaGAN: Capsule differential adversarial continuous learning for cross-domain hyperspectral anomaly detection 

**Title (ZH)**: CL-CaGAN: 胶囊差异对抗连续学习在跨域高光谱异常检测中的应用 

**Authors**: Jianing Wang, Siying Guo, Zheng Hua, Runhu Huang, Jinyu Hu, Maoguo Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.11793)  

**Abstract**: Anomaly detection (AD) has attracted remarkable attention in hyperspectral image (HSI) processing fields, and most existing deep learning (DL)-based algorithms indicate dramatic potential for detecting anomaly samples through specific training process under current scenario. However, the limited prior information and the catastrophic forgetting problem indicate crucial challenges for existing DL structure in open scenarios cross-domain detection. In order to improve the detection performance, a novel continual learning-based capsule differential generative adversarial network (CL-CaGAN) is proposed to elevate the cross-scenario learning performance for facilitating the real application of DL-based structure in hyperspectral AD (HAD) task. First, a modified capsule structure with adversarial learning network is constructed to estimate the background distribution for surmounting the deficiency of prior information. To mitigate the catastrophic forgetting phenomenon, clustering-based sample replay strategy and a designed extra self-distillation regularization are integrated for merging the history and future knowledge in continual AD task, while the discriminative learning ability from previous detection scenario to current scenario is retained by the elaborately designed structure with continual learning (CL) strategy. In addition, the differentiable enhancement is enforced to augment the generation performance of the training data. This further stabilizes the training process with better convergence and efficiently consolidates the reconstruction ability of background samples. To verify the effectiveness of our proposed CL-CaGAN, we conduct experiments on several real HSIs, and the results indicate that the proposed CL-CaGAN demonstrates higher detection performance and continuous learning capacity for mitigating the catastrophic forgetting under cross-domain scenarios. 

**Abstract (ZH)**: 基于持续学习的卷积胶囊生成对抗网络在跨域高光谱异常检测中的应用 

---
# Improving Coverage in Combined Prediction Sets with Weighted p-values 

**Title (ZH)**: 结合预测集中的加权p值提高覆盖度 

**Authors**: Gina Wong, Drew Prinster, Suchi Saria, Rama Chellappa, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11785)  

**Abstract**: Conformal prediction quantifies the uncertainty of machine learning models by augmenting point predictions with valid prediction sets, assuming exchangeability. For complex scenarios involving multiple trials, models, or data sources, conformal prediction sets can be aggregated to create a prediction set that captures the overall uncertainty, often improving precision. However, aggregating multiple prediction sets with individual $1-\alpha$ coverage inevitably weakens the overall guarantee, typically resulting in $1-2\alpha$ worst-case coverage. In this work, we propose a framework for the weighted aggregation of prediction sets, where weights are assigned to each prediction set based on their contribution. Our framework offers flexible control over how the sets are aggregated, achieving tighter coverage bounds that interpolate between the $1-2\alpha$ guarantee of the combined models and the $1-\alpha$ guarantee of an individual model depending on the distribution of weights. We extend our framework to data-dependent weights, and we derive a general procedure for data-dependent weight aggregation that maintains finite-sample validity. We demonstrate the effectiveness of our methods through experiments on synthetic and real data in the mixture-of-experts setting, and we show that aggregation with data-dependent weights provides a form of adaptive coverage. 

**Abstract (ZH)**: 基于权重聚合的预测集融合框架及其应用 

---
# Generative and Contrastive Graph Representation Learning 

**Title (ZH)**: 生成对比图表示学习 

**Authors**: Jiali Chen, Avijit Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.11776)  

**Abstract**: Self-supervised learning (SSL) on graphs generates node and graph representations (i.e., embeddings) that can be used for downstream tasks such as node classification, node clustering, and link prediction. Graph SSL is particularly useful in scenarios with limited or no labeled data. Existing SSL methods predominantly follow contrastive or generative paradigms, each excelling in different tasks: contrastive methods typically perform well on classification tasks, while generative methods often excel in link prediction. In this paper, we present a novel architecture for graph SSL that integrates the strengths of both approaches. Our framework introduces community-aware node-level contrastive learning, providing more robust and effective positive and negative node pairs generation, alongside graph-level contrastive learning to capture global semantic information. Additionally, we employ a comprehensive augmentation strategy that combines feature masking, node perturbation, and edge perturbation, enabling robust and diverse representation learning. By incorporating these enhancements, our model achieves superior performance across multiple tasks, including node classification, clustering, and link prediction. Evaluations on open benchmark datasets demonstrate that our model outperforms state-of-the-art methods, achieving a performance lift of 0.23%-2.01% depending on the task and dataset. 

**Abstract (ZH)**: 基于图的自监督学习：结合节点和图级对比学习的社区意识节点对比学习架构 

---
# HARDMath2: A Benchmark for Applied Mathematics Built by Students as Part of a Graduate Class 

**Title (ZH)**: HARDMath2：由学生在研究生课程中构建的应用数学基准测试 

**Authors**: James V. Roggeveen, Erik Y. Wang, Will Flintoft, Peter Donets, Lucy S. Nathwani, Nickholas Gutierrez, David Ettel, Anton Marius Graf, Siddharth Dandavate, Arjun Nageswaran, Raglan Ward, Ava Williamson, Anne Mykland, Kacper K. Migacz, Yijun Wang, Egemen Bostan, Duy Thuc Nguyen, Zhe He, Marc L. Descoteaux, Felix Yeung, Shida Liu, Jorge García Ponce, Luke Zhu, Yuyang Chen, Ekaterina S. Ivshina, Miguel Fernandez, Minjae Kim, Kennan Gumbs, Matthew Scott Tan, Russell Yang, Mai Hoang, David Brown, Isabella A. Silveira, Lavon Sykes, Ahmed Roman, William Fredenberg, Yiming Chen, Lucas Martin, Yixing Tang, Kelly Werker Smith, Hongyu Liao, Logan G. Wilson, Alexander Dazhen Cai, Andrea Elizabeth Biju, Michael P. Brenner  

**Link**: [PDF](https://arxiv.org/pdf/2505.11774)  

**Abstract**: Large language models (LLMs) have shown remarkable progress in mathematical problem-solving, but evaluation has largely focused on problems that have exact analytical solutions or involve formal proofs, often overlooking approximation-based problems ubiquitous in applied science and engineering. To fill this gap, we build on prior work and present HARDMath2, a dataset of 211 original problems covering the core topics in an introductory graduate applied math class, including boundary-layer analysis, WKB methods, asymptotic solutions of nonlinear partial differential equations, and the asymptotics of oscillatory integrals. This dataset was designed and verified by the students and instructors of a core graduate applied mathematics course at Harvard. We build the dataset through a novel collaborative environment that challenges students to write and refine difficult problems consistent with the class syllabus, peer-validate solutions, test different models, and automatically check LLM-generated solutions against their own answers and numerical ground truths. Evaluation results show that leading frontier models still struggle with many of the problems in the dataset, highlighting a gap in the mathematical reasoning skills of current LLMs. Importantly, students identified strategies to create increasingly difficult problems by interacting with the models and exploiting common failure modes. This back-and-forth with the models not only resulted in a richer and more challenging benchmark but also led to qualitative improvements in the students' understanding of the course material, which is increasingly important as we enter an age where state-of-the-art language models can solve many challenging problems across a wide domain of fields. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学问题解决方面取得了显著进展，但评估主要集中在具有精确解析解或涉及形式证明的问题上，往往忽视了应用科学和工程中常见的基于近似的问题。为了弥补这一不足，我们在此基础上提出了HARDMath2数据集，包含211个原创问题，涵盖哈佛大学核心研究生应用数学课程中的核心主题，包括边界层分析、WKB方法、非线性偏微分方程的渐近解以及振荡积分的渐近分析。该数据集由该课程的学生和教师设计和验证。我们通过一个新颖的合作环境构建数据集，挑战学生编写和改进符合课程大纲的难题，同伴验证解决方案，测试不同的模型，并自动检查LLM生成的解决方案与自己的答案和数值真实值之间的差异。评估结果表明，当前领先的前沿模型仍然难以解决数据集中许多问题，突显了当前LLMs在数学推理能力上的差距。重要的是，学生通过与模型的互动和利用常见的失败模式，发现创建更难问题的策略。这种与模型的互动不仅产生了更加丰富和更具挑战性的基准，还促进了学生对课程材料的深刻理解，这在我们进入一个先进语言模型可以解决广泛领域内许多挑战性问题的时代尤为重要。 

---
# Residual Feature Integration is Sufficient to Prevent Negative Transfer 

**Title (ZH)**: 残余特征集成足以防止负迁移 

**Authors**: Yichen Xu, Ryumei Nakada, Linjun Zhang, Lexin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11771)  

**Abstract**: Transfer learning typically leverages representations learned from a source domain to improve performance on a target task. A common approach is to extract features from a pre-trained model and directly apply them for target prediction. However, this strategy is prone to negative transfer where the source representation fails to align with the target distribution. In this article, we propose Residual Feature Integration (REFINE), a simple yet effective method designed to mitigate negative transfer. Our approach combines a fixed source-side representation with a trainable target-side encoder and fits a shallow neural network on the resulting joint representation, which adapts to the target domain while preserving transferable knowledge from the source domain. Theoretically, we prove that REFINE is sufficient to prevent negative transfer under mild conditions, and derive the generalization bound demonstrating its theoretical benefit. Empirically, we show that REFINE consistently enhances performance across diverse application and data modalities including vision, text, and tabular data, and outperforms numerous alternative solutions. Our method is lightweight, architecture-agnostic, and robust, making it a valuable addition to the existing transfer learning toolbox. 

**Abstract (ZH)**: 迁移学习通常通过利用源领域中学到的表示来提高目标任务的性能。一种常见方法是从预训练模型中提取特征并直接应用于目标预测。然而，这种策略容易出现负迁移，即源表示与目标分布不一致。本文提出了一种简单而有效的方法Residual Feature Integration (REFINE)，旨在减轻负迁移。我们的方法结合了固定来源端表示和可训练的目标端编码器，并在所得到的联合表示上拟合一个浅层神经网络，该网络适应目标领域同时保留来自源领域的可迁移知识。理论上，我们证明在较温和的条件下，REFINE足以防止负迁移，并推导出泛化界限，以说明其理论优势。实验上，我们展示了REFINE在视觉、文本和表格数据等多种应用和数据模态中一致性地提升了性能，并优于多种替代解决方案。该方法轻量级、架构无关且稳健，使其成为现有迁移学习工具箱中的一个重要补充。 

---
# Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors 

**Title (ZH)**: 内部因果机制稳健预测语言模型离分布行为 

**Authors**: Jing Huang, Junyi Tao, Thomas Icard, Diyi Yang, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2505.11770)  

**Abstract**: Interpretability research now offers a variety of techniques for identifying abstract internal mechanisms in neural networks. Can such techniques be used to predict how models will behave on out-of-distribution examples? In this work, we provide a positive answer to this question. Through a diverse set of language modeling tasks--including symbol manipulation, knowledge retrieval, and instruction following--we show that the most robust features for correctness prediction are those that play a distinctive causal role in the model's behavior. Specifically, we propose two methods that leverage causal mechanisms to predict the correctness of model outputs: counterfactual simulation (checking whether key causal variables are realized) and value probing (using the values of those variables to make predictions). Both achieve high AUC-ROC in distribution and outperform methods that rely on causal-agnostic features in out-of-distribution settings, where predicting model behaviors is more crucial. Our work thus highlights a novel and significant application for internal causal analysis of language models. 

**Abstract (ZH)**: 当前的可解释性研究提供了多种技术来识别神经网络中的抽象内部机制。这些技术能否用于预测模型在分布外样本上的行为？在这项工作中，我们给出了肯定的答案。通过一系列多样的语言建模任务——包括符号操作、知识检索和指令遵循——我们表明，最 robust 的正确性预测特征是那些在模型行为中扮演独特因果角色的特征。 Specifically，我们提出了两种利用因果机制来预测模型输出正确性的方法：反事实仿真（检查关键因果变量是否实现）和值探测（使用这些变量的值来做出预测）。这两种方法在分布内都达到了高 AUC-ROC，并在分布外环境中优于依赖于因果无关特征的方法，因为在这种环境下预测模型行为更为关键。因此，我们的工作强调了一种新颖且重要的语言模型内部因果分析的应用。 

---
# Redefining Neural Operators in $d+1$ Dimensions 

**Title (ZH)**: 重定义$d+1$维神经算子 

**Authors**: Haoze Song, Zhihao Li, Xiaobo Zhang, Zecheng Gan, Zhilu Lai, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11766)  

**Abstract**: Neural Operators have emerged as powerful tools for learning mappings between function spaces. Among them, the kernel integral operator has been widely validated on universally approximating various operators. Although recent advancements following this definition have developed effective modules to better approximate the kernel function defined on the original domain (with $d$ dimensions, $d=1, 2, 3...$), the unclarified evolving mechanism in the embedding spaces blocks our view to design neural operators that can fully capture the target system evolution.
Drawing on recent breakthroughs in quantum simulation of partial differential equations (PDEs), we elucidate the linear evolution process in neural operators. Based on that, we redefine neural operators on a new $d+1$ dimensional domain. Within this framework, we implement our proposed Schrödingerised Kernel Neural Operator (SKNO) aligning better with the $d+1$ dimensional evolution. In experiments, our $d+1$ dimensional evolving linear block performs far better than others. Also, we test SKNO's SOTA performance on various benchmark tests and also the zero-shot super-resolution task. In addition, we analyse the impact of different lifting and recovering operators on the prediction within the redefined NO framework, reflecting the alignment between our model and the underlying $d+1$ dimensional evolution. 

**Abstract (ZH)**: 神经运算子已成为学习函数空间之间映射的强大工具。其中，核积分运算子已被广泛验证适用于多种运算子的普遍逼近。尽管基于此定义的最近进展开发了有效的模块以更好地逼近原始领域上的核函数（维度 $d=1,2,3...$），但嵌入空间中的演变机制不明确阻碍了我们设计能够完全捕捉目标系统演化的神经运算子。借鉴近期在偏微分方程（PDEs）量子模拟方面的突破，我们阐明了神经运算子中的线性演变过程。在此基础上，我们在新的 $d+1$ 维域上重新定义了神经运算子。在这一框架内，我们实现了一种新的薛定谔化核神经运算子（SKNO），更好地适应了 $d+1$ 维的演变。在实验中，我们的 $d+1$ 维线性演变块远优于其他模块。我们还测试了SKNO在多种基准测试和零样本超分辨率任务中的SOTA性能。此外，我们分析了不同提升和恢复算子对重定义的NO框架内预测的影响，反映了我们模型与潜在的 $d+1$ 维演变之间的对齐。 

---
# OMAC: A Broad Optimization Framework for LLM-Based Multi-Agent Collaboration 

**Title (ZH)**: OMAC：一种基于大规模语言模型的多Agent协作广义优化框架 

**Authors**: Shijun Li, Hilaf Hasson, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11765)  

**Abstract**: Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications. Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited.
In this work, we introduce OMAC, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on code generation, arithmetic reasoning, and general reasoning tasks against state-of-the-art approaches. 

**Abstract (ZH)**: 基于先进大型语言模型的代理驱动系统全面优化框架OMAC 

---
# Towards Universal Semantics With Large Language Models 

**Title (ZH)**: 面向通用语义的大语言模型研究 

**Authors**: Raymond Baartmans, Matthew Raffel, Rahul Vikram, Aiden Deringer, Lizhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11764)  

**Abstract**: The Natural Semantic Metalanguage (NSM) is a linguistic theory based on a universal set of semantic primes: simple, primitive word-meanings that have been shown to exist in most, if not all, languages of the world. According to this framework, any word, regardless of complexity, can be paraphrased using these primes, revealing a clear and universally translatable meaning. These paraphrases, known as explications, can offer valuable applications for many natural language processing (NLP) tasks, but producing them has traditionally been a slow, manual process. In this work, we present the first study of using large language models (LLMs) to generate NSM explications. We introduce automatic evaluation methods, a tailored dataset for training and evaluation, and fine-tuned models for this task. Our 1B and 8B models outperform GPT-4o in producing accurate, cross-translatable explications, marking a significant step toward universal semantic representation with LLMs and opening up new possibilities for applications in semantic analysis, translation, and beyond. 

**Abstract (ZH)**: 基于语义原素的自然语义金属语言（NSM）是一种基于一套普遍存在的简单原始语义——这些语义在世界的大多数语言中被证实存在——的linguistic理论。根据这一框架，任何单词，无论其复杂程度如何，都可以用这些语义原素进行重新表述，从而揭示清晰且可普遍翻译的意义。这些重新表述被称为explications，它们可以在许多自然语言处理（NLP）任务中提供有价值的应用，但传统上生成它们是一个缓慢且手动的过程。在本研究中，我们提出了首次使用大型语言模型（LLMs）生成NSM explications的研究。我们介绍了自动评估方法、用于训练和评估的定制数据集以及针对该任务的微调模型。我们的1B和8B模型在生成准确且跨语言可翻译的explications方面优于GPT-4o，标志着使用LLMs实现通用语义表示的重要一步，并为语义分析、翻译等领域的新型应用打开了新的可能性。 

---
# Topology-Aware Knowledge Propagation in Decentralized Learning 

**Title (ZH)**: 拓扑感知知识 propagation 在去中心化学习中的应用 

**Authors**: Mansi Sakarvadia, Nathaniel Hudson, Tian Li, Ian Foster, Kyle Chard  

**Link**: [PDF](https://arxiv.org/pdf/2505.11760)  

**Abstract**: Decentralized learning enables collaborative training of models across naturally distributed data without centralized coordination or maintenance of a global model. Instead, devices are organized in arbitrary communication topologies, in which they can only communicate with neighboring devices. Each device maintains its own local model by training on its local data and integrating new knowledge via model aggregation with neighbors. Therefore, knowledge is propagated across the topology via successive aggregation rounds. We study, in particular, the propagation of out-of-distribution (OOD) knowledge. We find that popular decentralized learning algorithms struggle to propagate OOD knowledge effectively to all devices. Further, we find that both the location of OOD data within a topology, and the topology itself, significantly impact OOD knowledge propagation. We then propose topology-aware aggregation strategies to accelerate (OOD) knowledge propagation across devices. These strategies improve OOD data accuracy, compared to topology-unaware baselines, by 123% on average across models in a topology. 

**Abstract (ZH)**: 去中心化学习使设备能够在任意通信拓扑下进行协作训练，无需中央协调或维护全局模型。我们特别研究了异常分布（OOD）知识的传播问题。发现流行的去中心化学习算法在有效传播OOD知识方面存在困难。进一步研究表明，异常分布数据在拓扑中的位置以及拓扑结构本身显著影响OOD知识的传播。为此，我们提出了感知拓扑的聚合策略，以加速设备间OOD知识的传播。这些策略在平均意义上将模型中OOD数据的准确性提高了123%，相比未感知拓扑的基线方法。 

---
# Generalizable Vision-Language Few-Shot Adaptation with Predictive Prompts and Negative Learning 

**Title (ZH)**: 具有预测性提示和负学习的通用视觉-语言少量样本适应性学习 

**Authors**: Sriram Mandalika  

**Link**: [PDF](https://arxiv.org/pdf/2505.11758)  

**Abstract**: Few-shot adaptation remains a core challenge for vision-language models (VLMs), especially under limited supervision and noisy support samples. We propose PromptFuseNL, a unified framework that enhances few-shot generalization by combining predictive prompt tuning with dual-branch positive and negative learning. The method refines class prototypes through task-conditioned residuals, multi-stage cross-modal coordination, and semantic hard negative mining. To address label noise, we introduce an unsupervised instance reweighting strategy that downweights unreliable support examples without requiring additional labels or structural changes. PromptFuseNL fuses visual and textual cues through lightweight modules for efficient and discriminative prediction. Evaluated across 15 benchmarks, it consistently surpasses existing prompt- and adapter-based methods in all shot settings while remaining highly efficient, achieving up to 300x faster training and 1000x lower FLOPs compared to full prompt tuning, achieving a new state-of-the-art for robust and scalable few-shot vision-language adaptation. 

**Abstract (ZH)**: Few-Shot 调适应挑战了视觉-语言模型（VLMs）的核心能力，尤其是在有限监督和嘈杂的支持样本下的表现。我们提出了一种统一框架 PromptFuseNL，通过结合预测提示调优与双支路正负样本学习来增强Few-Shot泛化能力。该方法通过任务条件下的残差、多阶段跨模态协调和语义硬负样本挖掘来精炼类原型。为解决标签噪声问题，我们引入了一种无需额外标签或结构变化的无监督实例加权策略，以降低不可靠支持样本的权重。PromptFuseNL 通过轻量级模块融合视觉和文本线索，实现高效的区分性预测。在15个基准测试中，该方法在所有Few-Shot设置下均优于现有基于提示和适配器的方法，同时保持高效性能，相较于全面提示调优，训练速度提升至300倍，FLOPs 降低至1000倍，实现了稳健且可扩展的Few-Shot视觉-语言适应的新最佳性能。 

---
# Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders 

**Title (ZH)**: 特征对冲：相关特征打破窄稀疏自编码器 

**Authors**: David Chanin, Tomáš Dulka, Adrià Garriga-Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2505.11756)  

**Abstract**: It is assumed that sparse autoencoders (SAEs) decompose polysemantic activations into interpretable linear directions, as long as the activations are composed of sparse linear combinations of underlying features. However, we find that if an SAE is more narrow than the number of underlying "true features" on which it is trained, and there is correlation between features, the SAE will merge components of correlated features together, thus destroying monosemanticity. In LLM SAEs, these two conditions are almost certainly true. This phenomenon, which we call feature hedging, is caused by SAE reconstruction loss, and is more severe the narrower the SAE. In this work, we introduce the problem of feature hedging and study it both theoretically in toy models and empirically in SAEs trained on LLMs. We suspect that feature hedging may be one of the core reasons that SAEs consistently underperform supervised baselines. Finally, we use our understanding of feature hedging to propose an improved variant of matryoshka SAEs. Our work shows there remain fundamental issues with SAEs, but we are hopeful that that highlighting feature hedging will catalyze future advances that allow SAEs to achieve their full potential of interpreting LLMs at scale. 

**Abstract (ZH)**: 特征对冲：稀疏自编码器在大规模语言模型中的表现不佳原因探究 

---
# Reachability Barrier Networks: Learning Hamilton-Jacobi Solutions for Smooth and Flexible Control Barrier Functions 

**Title (ZH)**: 可达性障碍网络：学习光滑灵活的控制障碍函数的哈密尔顿-雅可比解 

**Authors**: Matthew Kim, William Sharpless, Hyun Joe Jeong, Sander Tonkens, Somil Bansal, Sylvia Herbert  

**Link**: [PDF](https://arxiv.org/pdf/2505.11755)  

**Abstract**: Recent developments in autonomous driving and robotics underscore the necessity of safety-critical controllers. Control barrier functions (CBFs) are a popular method for appending safety guarantees to a general control framework, but they are notoriously difficult to generate beyond low dimensions. Existing methods often yield non-differentiable or inaccurate approximations that lack integrity, and thus fail to ensure safety. In this work, we use physics-informed neural networks (PINNs) to generate smooth approximations of CBFs by computing Hamilton-Jacobi (HJ) optimal control solutions. These reachability barrier networks (RBNs) avoid traditional dimensionality constraints and support the tuning of their conservativeness post-training through a parameterized discount term. To ensure robustness of the discounted solutions, we leverage conformal prediction methods to derive probabilistic safety guarantees for RBNs. We demonstrate that RBNs are highly accurate in low dimensions, and safer than the standard neural CBF approach in high dimensions. Namely, we showcase the RBNs in a 9D multi-vehicle collision avoidance problem where it empirically proves to be 5.5x safer and 1.9x less conservative than the neural CBFs, offering a promising method to synthesize CBFs for general nonlinear autonomous systems. 

**Abstract (ZH)**: Recent developments in自主驾驶和机器人技术强调了安全关键控制器的必要性。控制障碍函数（CBFs）是一种在通用控制框架中附加安全保证的流行方法，但它们在高维情况下的生成通常非常困难。现有方法往往会产生非光滑或不准确的近似值，缺乏完整性，从而无法确保安全性。在本工作中，我们利用物理信息神经网络（PINNs）通过计算哈密顿-雅可比（HJ）最优控制解来生成CBFs的平滑近似。这些可达性障碍网络（RBNs）避免了传统的维数限制，并通过参数化的折扣项在训练后支持其保守性的调整。为了确保折扣解的鲁棒性，我们利用可信预测方法为RBNs推导出概率安全保证。我们证明RBNs在低维情况中非常精确，并且在高维情况下比标准神经CBFs更安全。具体而言，我们展示了RBNs在9维多车辆碰撞避免问题中的应用，结果显示RBNs在安全性上比神经CBFs高5.5倍，在保守性上低1.9倍，为合成一般非线性自主系统的CBFs提供了有前景的方法。 

---
# Improving Medium Range Severe Weather Prediction through Transformer Post-processing of AI Weather Forecasts 

**Title (ZH)**: 通过AI气象预报的变压器后处理改进中期范围极端天气预测 

**Authors**: Zhanxiang Hua, Ryan Sobash, David John Gagne II, Yingkai Sha, Alexandra Anderson-Frey  

**Link**: [PDF](https://arxiv.org/pdf/2505.11750)  

**Abstract**: Improving the skill of medium-range (1-8 day) severe weather prediction is crucial for mitigating societal impacts. This study introduces a novel approach leveraging decoder-only transformer networks to post-process AI-based weather forecasts, specifically from the Pangu-Weather model, for improved severe weather guidance. Unlike traditional post-processing methods that use a dense neural network to predict the probability of severe weather using discrete forecast samples, our method treats forecast lead times as sequential ``tokens'', enabling the transformer to learn complex temporal relationships within the evolving atmospheric state. We compare this approach against post-processing of the Global Forecast System (GFS) using both a traditional dense neural network and our transformer, as well as configurations that exclude convective parameters to fairly evaluate the impact of using the Pangu-Weather AI model. Results demonstrate that the transformer-based post-processing significantly enhances forecast skill compared to dense neural networks. Furthermore, AI-driven forecasts, particularly Pangu-Weather initialized from high resolution analysis, exhibit superior performance to GFS in the medium-range, even without explicit convective parameters. Our approach offers improved accuracy, and reliability, which also provides interpretability through feature attribution analysis, advancing medium-range severe weather prediction capabilities. 

**Abstract (ZH)**: 提高中范围（1-8天）极端天气预测技能对于减轻社会影响至关重要。本文介绍了一种新颖的方法，利用解码器为主的变压器网络对基于人工智能的天气预报进行后处理，特别是对Pangu-Weather模型的预报进行改进，以提供更优质的极端天气指导。与传统使用密集型神经网络预测极端天气概率的方法不同，我们的方法将预报提前时间视为序列“令牌”，使变压器能够学习演变大气状态中的复杂时间关系。我们将此方法与对全球预报系统（GFS）进行的传统密集型神经网络后处理和变压器后处理进行对比，包括排除对流参数的配置，以公平评估使用Pangu-Weather人工智能模型的影响。结果表明，基于变压器的后处理显著提高了预报技能，相较于密集型神经网络。此外，以高分辨率分析初始化的人工智能驱动预报，在中范围预测中表现出色，即便是没有明确对流参数的情况下，也优于GFS。本方法提供了更高的准确性和可靠性，并通过特征归因分析增强了可解释性，从而推进中范围极端天气预测能力。 

---
# Token Masking Improves Transformer-Based Text Classification 

**Title (ZH)**: Token Masking 提高了基于Transformer的文本分类性能 

**Authors**: Xianglong Xu, John Bowen, Rojin Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2505.11746)  

**Abstract**: While transformer-based models achieve strong performance on text classification, we explore whether masking input tokens can further enhance their effectiveness. We propose token masking regularization, a simple yet theoretically motivated method that randomly replaces input tokens with a special [MASK] token at probability p. This introduces stochastic perturbations during training, leading to implicit gradient averaging that encourages the model to capture deeper inter-token dependencies. Experiments on language identification and sentiment analysis -- across diverse models (mBERT, Qwen2.5-0.5B, TinyLlama-1.1B) -- show consistent improvements over standard regularization techniques. We identify task-specific optimal masking rates, with p = 0.1 as a strong general default. We attribute the gains to two key effects: (1) input perturbation reduces overfitting, and (2) gradient-level smoothing acts as implicit ensembling. 

**Abstract (ZH)**: 基于变压器的模型在文本分类任务中表现出强大性能，我们探索掩蔽输入 token 是否能进一步提高其效果。我们提出了 token 掩蔽正则化方法，这是一种简单但具有理论依据的方法，通过以概率 p 随机用特殊 [MASK] token 替换输入 token，引入训练中的随机扰动，从而实现隐式的梯度平均，促使模型捕获更深的 token 间依赖关系。实验结果显示，在语言识别和情感分析任务中，该方法在多种模型（mBERT、Qwen2.5-0.5B、TinyLlama-1.1B）上均比标准正则化技术表现出一致的改进。我们确定了特定任务的最佳掩蔽率，p=0.1 作为通用默认值。我们将其性能提升归因于两个关键效果：（1）输入扰动减少过拟合，（2）梯度级别平滑起到隐式集成的作用。 

---
# POCAII: Parameter Optimization with Conscious Allocation using Iterative Intelligence 

**Title (ZH)**: POCAII：基于迭代智能的意识分配参数优化 

**Authors**: Joshua Inman, Tanmay Khandait, Lalitha Sankar, Giulia Pedrielli  

**Link**: [PDF](https://arxiv.org/pdf/2505.11745)  

**Abstract**: In this paper we propose for the first time the hyperparameter optimization (HPO) algorithm POCAII. POCAII differs from the Hyperband and Successive Halving literature by explicitly separating the search and evaluation phases and utilizing principled approaches to exploration and exploitation principles during both phases. Such distinction results in a highly flexible scheme for managing a hyperparameter optimization budget by focusing on search (i.e., generating competing configurations) towards the start of the HPO process while increasing the evaluation effort as the HPO comes to an end.
POCAII was compared to state of the art approaches SMAC, BOHB and DEHB. Our algorithm shows superior performance in low-budget hyperparameter optimization regimes. Since many practitioners do not have exhaustive resources to assign to HPO, it has wide applications to real-world problems. Moreover, the empirical evidence showed how POCAII demonstrates higher robustness and lower variance in the results. This is again very important when considering realistic scenarios with extremely expensive models to train. 

**Abstract (ZH)**: 本文首次提出了一种超参数优化算法POCAII。POCAII与Hyperband和Successive Halving文献的不同之处在于明确地将搜索和评估阶段分开，并在两个阶段中利用原则性的探索和利用方法。这种区分导致了一种高度灵活的预算管理方案，即在超参数优化过程中初期侧重于搜索（即生成竞争配置），并随着超参数优化过程的结束而增加评估努力。将POCAII与当前最先进的方法SMAC、BOHB和DEHB进行了比较。在低预算超参数优化领域，我们的算法显示出更优的表现。由于许多实践者没有足够的资源来分配给超参数优化，因此POCAII在实际问题中具有广泛的应用。此外，实证证据表明POCAII在结果上表现出更高的稳健性和更低的方差。这在考虑极昂贵模型训练的现实场景中尤为重要。 

---
# Cloud-Based AI Systems: Leveraging Large Language Models for Intelligent Fault Detection and Autonomous Self-Healing 

**Title (ZH)**: 基于云的AI系统：利用大规模语言模型进行智能故障检测与自主自我修复 

**Authors**: Cheng Ji, Huaiying Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11743)  

**Abstract**: With the rapid development of cloud computing systems and the increasing complexity of their infrastructure, intelligent mechanisms to detect and mitigate failures in real time are becoming increasingly important. Traditional methods of failure detection are often difficult to cope with the scale and dynamics of modern cloud environments. In this study, we propose a novel AI framework based on Massive Language Model (LLM) for intelligent fault detection and self-healing mechanisms in cloud systems. The model combines existing machine learning fault detection algorithms with LLM's natural language understanding capabilities to process and parse system logs, error reports, and real-time data streams through semantic context. The method adopts a multi-level architecture, combined with supervised learning for fault classification and unsupervised learning for anomaly detection, so that the system can predict potential failures before they occur and automatically trigger the self-healing mechanism. Experimental results show that the proposed model is significantly better than the traditional fault detection system in terms of fault detection accuracy, system downtime reduction and recovery speed. 

**Abstract (ZH)**: 基于大规模语言模型的智能故障检测与自愈机制框架 

---
# Simple and Effective Specialized Representations for Fair Classifiers 

**Title (ZH)**: 简单有效的专门化表示以实现公平分类器 

**Authors**: Alberto Sinigaglia, Davide Sartor, Marina Ceccon, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2505.11740)  

**Abstract**: Fair classification is a critical challenge that has gained increasing importance due to international regulations and its growing use in high-stakes decision-making settings. Existing methods often rely on adversarial learning or distribution matching across sensitive groups; however, adversarial learning can be unstable, and distribution matching can be computationally intensive. To address these limitations, we propose a novel approach based on the characteristic function distance. Our method ensures that the learned representation contains minimal sensitive information while maintaining high effectiveness for downstream tasks. By utilizing characteristic functions, we achieve a more stable and efficient solution compared to traditional methods. Additionally, we introduce a simple relaxation of the objective function that guarantees fairness in common classification models with no performance degradation. Experimental results on benchmark datasets demonstrate that our approach consistently matches or achieves better fairness and predictive accuracy than existing methods. Moreover, our method maintains robustness and computational efficiency, making it a practical solution for real-world applications. 

**Abstract (ZH)**: 公平分类是一个关键挑战，由于国际法规的要求及其在高 stakes 决策环境中的广泛应用而日益重要。现有方法通常依赖于对抗学习或敏感群体之间的分布匹配；然而，对抗学习可能会不稳定，而分布匹配则计算 intensity 较高。为了解决这些限制，我们提出了一种基于特征函数距离的新方法。该方法确保学到的表示中包含的敏感信息最少，同时仍能有效支持下游任务。通过利用特征函数，我们实现了一个比传统方法更稳定和高效的解决方案。此外，我们引入了一种简单的客观函数松弛，能够保证在不影响性能的情况下使普通分类模型实现公平性。在基准数据集上的实验结果表明，我们的方法在公平性和预测准确性方面均优于现有方法，并且还能保持鲁棒性和计算效率，使其成为实际应用中的实用解决方案。 

---
# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training 

**Title (ZH)**: ZeroTuning: 利用初始令牌增强大型语言模型而不进行训练 

**Authors**: Feijiang Han, Xiaodong Yu, Jianheng Tang, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11739)  

**Abstract**: Recently, training-free methods for improving large language models (LLMs) have attracted growing interest, with token-level attention tuning emerging as a promising and interpretable direction. However, existing methods typically rely on auxiliary mechanisms to identify important or irrelevant task-specific tokens, introducing potential bias and limiting applicability. In this paper, we uncover a surprising and elegant alternative: the semantically empty initial token is a powerful and underexplored control point for optimizing model behavior. Through theoretical analysis, we show that tuning the initial token's attention sharpens or flattens the attention distribution over subsequent tokens, and its role as an attention sink amplifies this effect. Empirically, we find that: (1) tuning its attention improves LLM performance more effectively than tuning other task-specific tokens; (2) the effect follows a consistent trend across layers, with earlier layers having greater impact, but varies across attention heads, with different heads showing distinct preferences in how they attend to this token. Based on these findings, we propose ZeroTuning, a training-free approach that improves LLM performance by applying head-specific attention adjustments to this special token. Despite tuning only one token, ZeroTuning achieves higher performance on text classification, multiple-choice, and multi-turn conversation tasks across models such as Llama, Qwen, and DeepSeek. For example, ZeroTuning improves Llama-3.1-8B by 11.71% on classification, 2.64% on QA tasks, and raises its multi-turn score from 7.804 to 7.966. The method is also robust to limited resources, few-shot settings, long contexts, quantization, decoding strategies, and prompt variations. Our work sheds light on a previously overlooked control point in LLMs, offering new insights into both inference-time tuning and model interpretability. 

**Abstract (ZH)**: 无训练方法提高大型语言模型性能的新颖途径：利用语义空初始标记进行零调谐 

---
# Token-Level Uncertainty Estimation for Large Language Model Reasoning 

**Title (ZH)**: 大型语言模型推理中的token级不确定性估计 

**Authors**: Tunyu Zhang, Haizhou Shi, Yibin Wang, Hengyi Wang, Xiaoxiao He, Zhuowei Li, Haoxian Chen, Ligong Han, Kai Xu, Huan Zhang, Dimitris Metaxas, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11737)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a token-level uncertainty estimation framework to enable LLMs to self-assess and self-improve their generation quality in mathematical reasoning. Specifically, we introduce low-rank random weight perturbation to LLM decoding, generating predictive distributions that we use to estimate token-level uncertainties. We then aggregate these uncertainties to reflect semantic uncertainty of the generated sequences. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that our token-level uncertainty metrics strongly correlate with answer correctness and model robustness. Additionally, we explore using uncertainty to directly enhance the model's reasoning performance through multiple generations and the particle filtering algorithm. Our approach consistently outperforms existing uncertainty estimation methods, establishing effective uncertainty estimation as a valuable tool for both evaluating and improving reasoning generation in LLMs. 

**Abstract (ZH)**: larg语言模型在数学推理中的token级不确定性估计算法及其应用 

---
# Efficient Uncertainty Estimation via Distillation of Bayesian Large Language Models 

**Title (ZH)**: 通过 Bayesian 大语言模型蒸馏实现高效不确定性估计 

**Authors**: Harshil Vejendla, Haizhou Shi, Yibin Wang, Tunyu Zhang, Huan Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11731)  

**Abstract**: Recent advances in uncertainty estimation for Large Language Models (LLMs) during downstream adaptation have addressed key challenges of reliability and simplicity. However, existing Bayesian methods typically require multiple sampling iterations during inference, creating significant efficiency issues that limit practical deployment. In this paper, we investigate the possibility of eliminating the need for test-time sampling for LLM uncertainty estimation. Specifically, when given an off-the-shelf Bayesian LLM, we distill its aligned confidence into a non-Bayesian student LLM by minimizing the divergence between their predictive distributions. Unlike typical calibration methods, our distillation is carried out solely on the training dataset without the need of an additional validation dataset. This simple yet effective approach achieves N-times more efficient uncertainty estimation during testing, where N is the number of samples traditionally required by Bayesian LLMs. Our extensive experiments demonstrate that uncertainty estimation capabilities on training data can successfully generalize to unseen test data through our distillation technique, consistently producing results comparable to (or even better than) state-of-the-art Bayesian LLMs. 

**Abstract (ZH)**: Recent advances in Large Language Models (LLMs)不确定性估计近期进展：在下游适应过程中，不确定性估计方法取得了关键的可靠性和简洁性方面的进步。然而，现有的贝叶斯方法通常需要在推理过程中进行多次采样迭代，这导致了显著的效率问题，限制了其实用部署。在本文中，我们探讨了消除测试时采样以估计LLM不确定性的可能性。具体而言，当我们使用现成的贝叶斯LLM时，通过最小化两者预测分布之间的偏离程度，将其对齐的置信度提炼到一个非贝叶斯的学生LLM中。与传统的校准方法不同，我们的提炼仅在训练数据集上进行，无需额外的验证数据集。这种简单而有效的方法在测试中实现比传统的贝叶斯LLM所需的采样次数N倍更高效的不确定性估计。我们的广泛实验表明，通过我们的提炼技术，训练数据上的不确定性估计能力可以成功泛化到未见过的测试数据上，始终能够产生与（或甚至优于）最先进的贝叶斯LLM相当的结果。标题：

Recent advances in uncertainty estimation for Large Language Models (LLMs) during downstream adaptation: Eliminating the Need for Test-Time Sampling 

---
# CLT and Edgeworth Expansion for m-out-of-n Bootstrap Estimators of The Studentized Median 

**Title (ZH)**: m-out-of-n自助法学生化中位数估计量的中心极限定理与Edgeworth展式 

**Authors**: Imon Banerjee, Sayak Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2505.11725)  

**Abstract**: The m-out-of-n bootstrap, originally proposed by Bickel, Gotze, and Zwet (1992), approximates the distribution of a statistic by repeatedly drawing m subsamples (with m much smaller than n) without replacement from an original sample of size n. It is now routinely used for robust inference with heavy-tailed data, bandwidth selection, and other large-sample applications. Despite its broad applicability across econometrics, biostatistics, and machine learning, rigorous parameter-free guarantees for the soundness of the m-out-of-n bootstrap when estimating sample quantiles have remained elusive.
This paper establishes such guarantees by analyzing the estimator of sample quantiles obtained from m-out-of-n resampling of a dataset of size n. We first prove a central limit theorem for a fully data-driven version of the estimator that holds under a mild moment condition and involves no unknown nuisance parameters. We then show that the moment assumption is essentially tight by constructing a counter-example in which the CLT fails. Strengthening the assumptions slightly, we derive an Edgeworth expansion that provides exact convergence rates and, as a corollary, a Berry Esseen bound on the bootstrap approximation error. Finally, we illustrate the scope of our results by deriving parameter-free asymptotic distributions for practical statistics, including the quantiles for random walk Metropolis-Hastings and the rewards of ergodic Markov decision processes, thereby demonstrating the usefulness of our theory in modern estimation and learning tasks. 

**Abstract (ZH)**: m-out-of-n自助法的参数自由保证：基于数据驱动量化估计的研究 

---
# Zero-Shot Visual Generalization in Robot Manipulation 

**Title (ZH)**: 零样本视觉通用性在机器人操作中的应用 

**Authors**: Sumeet Batra, Gaurav Sukhatme  

**Link**: [PDF](https://arxiv.org/pdf/2505.11719)  

**Abstract**: Training vision-based manipulation policies that are robust across diverse visual environments remains an important and unresolved challenge in robot learning. Current approaches often sidestep the problem by relying on invariant representations such as point clouds and depth, or by brute-forcing generalization through visual domain randomization and/or large, visually diverse datasets. Disentangled representation learning - especially when combined with principles of associative memory - has recently shown promise in enabling vision-based reinforcement learning policies to be robust to visual distribution shifts. However, these techniques have largely been constrained to simpler benchmarks and toy environments. In this work, we scale disentangled representation learning and associative memory to more visually and dynamically complex manipulation tasks and demonstrate zero-shot adaptability to visual perturbations in both simulation and on real hardware. We further extend this approach to imitation learning, specifically Diffusion Policy, and empirically show significant gains in visual generalization compared to state-of-the-art imitation learning methods. Finally, we introduce a novel technique adapted from the model equivariance literature that transforms any trained neural network policy into one invariant to 2D planar rotations, making our policy not only visually robust but also resilient to certain camera perturbations. We believe that this work marks a significant step towards manipulation policies that are not only adaptable out of the box, but also robust to the complexities and dynamical nature of real-world deployment. Supplementary videos are available at this https URL. 

**Abstract (ZH)**: 基于视觉的操控策略在多样视觉环境下的鲁棒性训练仍然是机器人学习中一个重要的且未解决的挑战。现有的方法往往通过依赖不变表示例如点云和深度，或者通过视觉域随机化和/或大型多样视觉数据集来粗暴地实现泛化。解耦表示学习——尤其是在结合关联记忆原理的情况下——最近显示出在使基于视觉的强化学习策略对视觉分布变化鲁棒方面具有潜力。然而，这些技术迄今为止主要被局限在更简单的基准和玩具环境中。在本工作中，我们将解耦表示学习和关联记忆扩展到更视觉和动态复杂的操控任务中，并在仿真和真实硬件上展示了对视觉扰动的零样本适应性。我们还进一步将这种方法扩展到模仿学习，特别是扩散策略，并实验证明了与最先进的模仿学习方法相比，视觉泛化能力有显著提高。最后，我们引入了一种源自模型不变性文献的新技术，将任何训练好的神经网络策略转换为对二维平面旋转不变的策略，使我们的策略不仅视觉鲁棒，而且对某些摄像头扰动也具有韧性。我们认为这项工作标志着向既适配性强又能应对现实世界部署复杂性和动态性质的操控策略迈出了一大步。辅助视频可在此链接获取。 

---
# EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents 

**Title (ZH)**: EnvInjection: 环境提示注入攻击针对多模态网络代理 

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.11717)  

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--referred to as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage, which can be implemented by modifying the webpage's source code. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines. 

**Abstract (ZH)**: 基于MLLM的Web代理通过生成基于网页截图的动作来与网页环境交互：EnvInjection环境注入攻击 

---
# Bi-Level Policy Optimization with Nyström Hypergradients 

**Title (ZH)**: Nyström 高阶梯度的多层次策略优化 

**Authors**: Arjun Prakash, Naicheng He, Denizalp Goktas, Amy Greenwald  

**Link**: [PDF](https://arxiv.org/pdf/2505.11714)  

**Abstract**: The dependency of the actor on the critic in actor-critic (AC) reinforcement learning means that AC can be characterized as a bilevel optimization (BLO) problem, also called a Stackelberg game. This characterization motivates two modifications to vanilla AC algorithms. First, the critic's update should be nested to learn a best response to the actor's policy. Second, the actor should update according to a hypergradient that takes changes in the critic's behavior into account. Computing this hypergradient involves finding an inverse Hessian vector product, a process that can be numerically unstable. We thus propose a new algorithm, Bilevel Policy Optimization with Nyström Hypergradients (BLPO), which uses nesting to account for the nested structure of BLO, and leverages the Nyström method to compute the hypergradient. Theoretically, we prove BLPO converges to (a point that satisfies the necessary conditions for) a local strong Stackelberg equilibrium in polynomial time with high probability, assuming a linear parametrization of the critic's objective. Empirically, we demonstrate that BLPO performs on par with or better than PPO on a variety of discrete and continuous control tasks. 

**Abstract (ZH)**: 基于Nyström超梯度的层次策略优化算法：Bilevel Policy Optimization with Nyström Hypergradients (BLPO) 

---
# Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization 

**Title (ZH)**: Qronos：通过塑造未来来修正 past-training 量化中的错误 

**Authors**: Shihao Zhang, Haoyu Zhang, Ian Colbert, Rayan Saab  

**Link**: [PDF](https://arxiv.org/pdf/2505.11695)  

**Abstract**: We introduce Qronos -- a new state-of-the-art post-training quantization algorithm that sequentially rounds and updates neural network weights. Qronos not only explicitly corrects errors due to both weight and activation quantization, but also errors resulting from quantizing previous layers. Our iterative algorithm is based on an interpretable and disciplined optimization framework that subsumes and surpasses existing data-driven approaches. At each step, Qronos alternates between error correction and diffusion via optimal update rules. Importantly, we prove that Qronos admits an efficient implementation that uses the Cholesky decomposition for solving least-squares problems. We also demonstrate that Qronos is compatible with existing transformation techniques such as Hadamard-based incoherence processing and weight-activation scaling equalization, among others. We evaluate Qronos using recent autoregressive language generation models in the Llama3 family; Qronos consistently outperforms previous state-of-the-art adaptive rounding methods when quantizing the weights, activations, and/or KV caches. 

**Abstract (ZH)**: Qronos——一种新的后训练量化算法，通过序列化裁剪和更新神经网络权重进行误差校正和扩散 

---
# Neural Networks as Universal Finite-State Machines: A Constructive Deterministic Finite Automaton Theory 

**Title (ZH)**: 神经网络作为通用有界状态机：一种构造性确定有限自动机理论 

**Authors**: Sahil Rajesh Dhayalkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11694)  

**Abstract**: We present a complete theoretical and empirical framework establishing feedforward neural networks as universal finite-state machines (N-FSMs). Our results prove that finite-depth ReLU and threshold networks can exactly simulate deterministic finite automata (DFAs) by unrolling state transitions into depth-wise neural layers, with formal characterizations of required depth, width, and state compression. We demonstrate that DFA transitions are linearly separable, binary threshold activations allow exponential compression, and Myhill-Nerode equivalence classes can be embedded into continuous latent spaces while preserving separability. We also formalize the expressivity boundary: fixed-depth feedforward networks cannot recognize non-regular languages requiring unbounded memory. Unlike prior heuristic or probing-based studies, we provide constructive proofs and design explicit DFA-unrolled neural architectures that empirically validate every claim. Our results bridge deep learning, automata theory, and neural-symbolic computation, offering a rigorous blueprint for how discrete symbolic processes can be realized in continuous neural systems. 

**Abstract (ZH)**: 我们提出了一整套理论和实证框架，将前馈神经网络确立为通用有限状态机（N-FSMs）。我们的结果证明，具有有限深度的ReLU和阈值网络可以通过展开状态转换为深度方向的神经层来精确模拟确定性有限自动机（DFAs），并给出了所需深度、宽度和状态压缩形式化特征。我们证明DFA转换是线性可分的，二元阈值激活允许指数压缩，并且Myhill-Nerode等价类可以嵌入到连续的潜在空间中同时保持可分性。我们还形式化了表达能力边界：固定深度的前馈网络无法识别需要无界记忆的非正规语言。与先前的启发式或探针研究不同，我们提供了建设性的证明，并设计了明确的DFA展开神经架构，以实证验证每个声明。我们的结果将深度学习、自动机理论和神经符号计算联系起来，提供了一个严格的蓝图，说明如何在连续的神经系统中实现离散符号过程。 

---
# The Geometry of ReLU Networks through the ReLU Transition Graph 

**Title (ZH)**: ReLU网络中的几何结构通过ReLU过渡图 

**Authors**: Sahil Rajesh Dhayalkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11692)  

**Abstract**: We develop a novel theoretical framework for analyzing ReLU neural networks through the lens of a combinatorial object we term the ReLU Transition Graph (RTG). In this graph, each node corresponds to a linear region induced by the network's activation patterns, and edges connect regions that differ by a single neuron flip. Building on this structure, we derive a suite of new theoretical results connecting RTG geometry to expressivity, generalization, and robustness. Our contributions include tight combinatorial bounds on RTG size and diameter, a proof of RTG connectivity, and graph-theoretic interpretations of VC-dimension. We also relate entropy and average degree of the RTG to generalization error. Each theoretical result is rigorously validated via carefully controlled experiments across varied network depths, widths, and data regimes. This work provides the first unified treatment of ReLU network structure via graph theory and opens new avenues for compression, regularization, and complexity control rooted in RTG analysis. 

**Abstract (ZH)**: 我们通过一种称为ReLU转换图（RTG）的组合对象的视角，发展了一种新颖的理论框架来分析ReLU神经网络。在此图中，每个节点对应于由网络激活模式诱导的线性区域，边连接仅相差一个神经元翻转的区域。基于这一结构，我们推导出一系列新的理论结果，将RTG的几何结构与表达能力、泛化能力和鲁棒性联系起来。我们的贡献包括严格的组合上界和直径估计、RTG连通性的证明以及VC维的图论解释。我们还将RTG的熵和平均度与泛化误差联系起来。每个理论结果均通过严格控制的实验在不同的网络深度、宽度和数据条件下进行验证。本工作首次通过图论统一了ReLU网络结构的处理，并为基础在RTG分析上的压缩、正则化和复杂性控制开辟了新的途径。 

---
# Second SIGIR Workshop on Simulations for Information Access (Sim4IA 2025) 

**Title (ZH)**: 第二屆 SIGIR 資料存取模擬工作坊（Sim4IA 2025） 

**Authors**: Philipp Schaer, Christin Katharina Kreutz, Krisztian Balog, Timo Breuer, Andreas Konstantin Kruff  

**Link**: [PDF](https://arxiv.org/pdf/2505.11687)  

**Abstract**: Simulations in information access (IA) have recently gained interest, as shown by various tutorials and workshops around that topic. Simulations can be key contributors to central IA research and evaluation questions, especially around interactive settings when real users are unavailable, or their participation is impossible due to ethical reasons. In addition, simulations in IA can help contribute to a better understanding of users, reduce complexity of evaluation experiments, and improve reproducibility. Building on recent developments in methods and toolkits, the second iteration of our Sim4IA workshop aims to again bring together researchers and practitioners to form an interactive and engaging forum for discussions on the future perspectives of the field. An additional aim is to plan an upcoming TREC/CLEF campaign. 

**Abstract (ZH)**: 信息访问中的模拟 recently gained interest 由于该主题下出现了多种教程和研讨会。模拟可以在中央信息访问研究和评估问题中发挥关键作用，特别是在无法获得真实用户或由于伦理原因无法进行其参与的情况下，特别是在交互式设置中。此外，信息访问中的模拟可以帮助更好地理解用户、减少评估实验的复杂性，并提高可重复性。基于近期方法和工具包的发展，Sim4IA工作坊的第二届旨在再次汇聚研究人员和实践者，形成一个互动和吸引人的论坛，讨论该领域的未来前景。另一个目标是规划即将举行的TREC/CLEF运动。 

---
# OT Score: An OT based Confidence Score for Unsupervised Domain Adaptation 

**Title (ZH)**: OT 分数：基于OT的无监督领域适应置信度分数 

**Authors**: Yiming Zhang, Sitong Liu, Alex Cloninger  

**Link**: [PDF](https://arxiv.org/pdf/2505.11669)  

**Abstract**: We address the computational and theoretical limitations of existing distributional alignment methods for unsupervised domain adaptation (UDA), particularly regarding the estimation of classification performance and confidence without target labels. Current theoretical frameworks for these methods often yield computationally intractable quantities and fail to adequately reflect the properties of the alignment algorithms employed. To overcome these challenges, we introduce the Optimal Transport (OT) score, a confidence metric derived from a novel theoretical analysis that exploits the flexibility of decision boundaries induced by Semi-Discrete Optimal Transport alignment. The proposed OT score is intuitively interpretable, theoretically rigorous, and computationally efficient. It provides principled uncertainty estimates for any given set of target pseudo-labels without requiring model retraining, and can flexibly adapt to varying degrees of available source information. Experimental results on standard UDA benchmarks demonstrate that classification accuracy consistently improves by identifying and removing low-confidence predictions, and that OT score significantly outperforms existing confidence metrics across diverse adaptation scenarios. 

**Abstract (ZH)**: 我们针对现有无监督领域适应方法在分布对齐中的计算和理论限制进行研究，特别是关于在缺乏目标标签的情况下估计分类性能和置信度的问题。当前这些方法的理论框架往往导致计算上不可行的量，并不能充分反映所使用对齐算法的特性。为了克服这些挑战，我们引入了最优传输（OT）分数，这是一种源自新颖的理论分析的信心度量，该分析利用了半离散最优传输对齐诱导的决策边界灵活性。所提出的OT分数直观可解释、理论上有严密性且计算上高效。它为任何给定的目标伪标签集提供了原则上明确的不确定性估计，无需重新训练模型，并且能够灵活适应可用源信息的不同程度。标准的无监督领域适应基准上的实验结果表明，通过识别和删除低置信度预测可以一致地提高分类准确率，并且OT分数在各种适应场景中显著优于现有的信心度量。 

---
# Multilingual Prompt Engineering in Large Language Models: A Survey Across NLP Tasks 

**Title (ZH)**: 大规模语言模型中的多语言提示工程：跨NLP任务的综述 

**Authors**: Shubham Vatsal, Harsh Dubey, Aditi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11665)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across a wide range of Natural Language Processing (NLP) tasks. However, ensuring their effectiveness across multiple languages presents unique challenges. Multilingual prompt engineering has emerged as a key approach to enhance LLMs' capabilities in diverse linguistic settings without requiring extensive parameter re-training or fine-tuning. With growing interest in multilingual prompt engineering over the past two to three years, researchers have explored various strategies to improve LLMs' performance across languages and NLP tasks. By crafting structured natural language prompts, researchers have successfully extracted knowledge from LLMs across different languages, making these techniques an accessible pathway for a broader audience, including those without deep expertise in machine learning, to harness the capabilities of LLMs. In this paper, we survey and categorize different multilingual prompting techniques based on the NLP tasks they address across a diverse set of datasets that collectively span around 250 languages. We further highlight the LLMs employed, present a taxonomy of approaches and discuss potential state-of-the-art (SoTA) methods for specific multilingual datasets. Additionally, we derive a range of insights across language families and resource levels (high-resource vs. low-resource), including analyses such as the distribution of NLP tasks by language resource type and the frequency of prompting methods across different language families. Our survey reviews 36 research papers covering 39 prompting techniques applied to 30 multilingual NLP tasks, with the majority of these studies published in the last two years. 

**Abstract (ZH)**: 大规模语言模型在多种自然语言处理任务中展现了令人印象深刻的性能。然而，在多种语言环境中确保其有效性提出了独特的挑战。多语言提示工程已成为一种关键方法，无需进行大量的参数重新训练或微调，即可增强大规模语言模型在不同语言环境中的能力。近年来，随着对多语言提示工程兴趣的增长，研究人员探索了多种策略以提高大规模语言模型在跨语言和自然语言处理任务中的性能。通过精心设计结构化的自然语言提示，研究人员能够从大规模语言模型中提取不同语言的知识，使这些技术成为更广泛用户，包括那些没有深厚机器学习背景的人，利用大规模语言模型能力的途径。在本文中，我们基于覆盖约250种语言的多样数据集，调查和分类了针对不同自然语言处理任务的多语言提示技术。我们还强调了所使用的语言模型，介绍了方法的分类，并讨论了特定多语言数据集的最新前沿方法。此外，我们得出了一系列关于语言家族和资源水平（高资源 vs 低资源）的见解，包括按语言资源类型划分的自然语言处理任务分布和不同语言家族中提示方法的频率分析。我们的调查回顾了36篇研究论文，涵盖了39种应用于30种多语言自然语言处理任务的提示技术，其中大多数研究论文在最近两年内发表。 

---
# Programmable metasurfaces for future photonic artificial intelligence 

**Title (ZH)**: 面向未来的光子人工智能可编程超表面 

**Authors**: Loubnan Abou-Hamdan, Emil Marinov, Peter Wiecha, Philipp del Hougne, Tianyu Wang, Patrice Genevet  

**Link**: [PDF](https://arxiv.org/pdf/2505.11659)  

**Abstract**: Photonic neural networks (PNNs), which share the inherent benefits of photonic systems, such as high parallelism and low power consumption, could challenge traditional digital neural networks in terms of energy efficiency, latency, and throughput. However, producing scalable photonic artificial intelligence (AI) solutions remains challenging. To make photonic AI models viable, the scalability problem needs to be solved. Large optical AI models implemented on PNNs are only commercially feasible if the advantages of optical computation outweigh the cost of their input-output overhead. In this Perspective, we discuss how field-programmable metasurface technology may become a key hardware ingredient in achieving scalable photonic AI accelerators and how it can compete with current digital electronic technologies. Programmability or reconfigurability is a pivotal component for PNN hardware, enabling in situ training and accommodating non-stationary use cases that require fine-tuning or transfer learning. Co-integration with electronics, 3D stacking, and large-scale manufacturing of metasurfaces would significantly improve PNN scalability and functionalities. Programmable metasurfaces could address some of the current challenges that PNNs face and enable next-generation photonic AI technology. 

**Abstract (ZH)**: 光子神经网络（PNNs）由于具备高并行性和低功耗等固有优势，可能在能效、时延和吞吐量方面对传统数字神经网络构成挑战。然而，开发可扩展的光子人工智能（AI）解决方案仍具挑战性。为了使光子AI模型变得可行，需要解决可扩展性问题。只有当光子计算的优势超过其输入输出开销的成本时，基于PNN的大规模光学AI模型才具有商业可行性。在本文综述中，我们讨论了如何通过可编程元表面技术实现可扩展的光子AI加速器，并探讨其如何与当前的数字电子技术竞争。PNN硬件的可编程性或重配置能力使其能够实现现场训练并适应需要微调或迁移学习的非稳态用例。元表面与电子器件的协同集成、三维堆叠以及大规模制造将显著提高PNN的可扩展性和功能。可编程元表面能够解决PNN当前面临的一些挑战，并推动下一代光子AI技术的发展。 

---
# PeerGuard: Defending Multi-Agent Systems Against Backdoor Attacks Through Mutual Reasoning 

**Title (ZH)**: PeerGuard：通过相互推理防御多智能体系统后门攻击 

**Authors**: Falong Fan, Xi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11642)  

**Abstract**: Multi-agent systems leverage advanced AI models as autonomous agents that interact, cooperate, or compete to complete complex tasks across applications such as robotics and traffic management. Despite their growing importance, safety in multi-agent systems remains largely underexplored, with most research focusing on single AI models rather than interacting agents. This work investigates backdoor vulnerabilities in multi-agent systems and proposes a defense mechanism based on agent interactions. By leveraging reasoning abilities, each agent evaluates responses from others to detect illogical reasoning processes, which indicate poisoned agents. Experiments on LLM-based multi-agent systems, including ChatGPT series and Llama 3, demonstrate the effectiveness of the proposed method, achieving high accuracy in identifying poisoned agents while minimizing false positives on clean agents. We believe this work provides insights into multi-agent system safety and contributes to the development of robust, trustworthy AI interactions. 

**Abstract (ZH)**: 多智能体系统中的后门漏洞及其防御机制：基于代理交互的逻辑推理评估 

---
# Chatting with Papers: A Hybrid Approach Using LLMs and Knowledge Graphs 

**Title (ZH)**: 与论文聊天：结合LLMs和知识图谱的混合方法 

**Authors**: Vyacheslav Tykhonov, Han Yang, Philipp Mayr, Jetze Touber, Andrea Scharnhorst  

**Link**: [PDF](https://arxiv.org/pdf/2505.11633)  

**Abstract**: This demo paper reports on a new workflow \textit{GhostWriter} that combines the use of Large Language Models and Knowledge Graphs (semantic artifacts) to support navigation through collections. Situated in the research area of Retrieval Augmented Generation, this specific workflow details the creation of local and adaptable chatbots. Based on the tool-suite \textit{EverythingData} at the backend, \textit{GhostWriter} provides an interface that enables querying and ``chatting'' with a collection. Applied iteratively, the workflow supports the information needs of researchers when interacting with a collection of papers, whether it be to gain an overview, to learn more about a specific concept and its context, and helps the researcher ultimately to refine their research question in a controlled way. We demonstrate the workflow for a collection of articles from the \textit{method data analysis} journal published by GESIS -- Leibniz-Institute for the Social Sciences. We also point to further application areas. 

**Abstract (ZH)**: This demo paper报告了一种新的工作流 GhostWriter，该工作流结合了大型语言模型和知识图谱（语义实体）以支持集合导航。该工作流位于检索增强生成的研究领域，详细描述了本地和可适应聊天机器人的创建。基于后台工具集 EverythingData，GhostWriter 提供了一个接口，使用户能够查询和“聊天”与集合的交互。通过迭代应用，该工作流支持研究人员在与论文集合交互时的信息需求，无论是为了获得概览、更多地了解特定概念及其背景，或者帮助研究人员以受控的方式最终精炼其研究问题。我们以德国社会科学院（GESIS）发布的《方法数据分析》期刊文章集合为例演示了该工作流，并指出了其进一步的应用领域。 

---
# Nearest Neighbor Multivariate Time Series Forecasting 

**Title (ZH)**: 最近邻多变量时间序列预测 

**Authors**: Huiliang Zhang, Ping Nie, Lijun Sun, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2505.11625)  

**Abstract**: Multivariate time series (MTS) forecasting has a wide range of applications in both industry and academia. Recently, spatial-temporal graph neural networks (STGNNs) have gained popularity as MTS forecasting methods. However, current STGNNs can only use the finite length of MTS input data due to the computational complexity. Moreover, they lack the ability to identify similar patterns throughout the entire dataset and struggle with data that exhibit sparsely and discontinuously distributed correlations among variables over an extensive historical period, resulting in only marginal improvements. In this article, we introduce a simple yet effective k-nearest neighbor MTS forecasting ( kNN-MTS) framework, which forecasts with a nearest neighbor retrieval mechanism over a large datastore of cached series, using representations from the MTS model for similarity search. This approach requires no additional training and scales to give the MTS model direct access to the whole dataset at test time, resulting in a highly expressive model that consistently improves performance, and has the ability to extract sparse distributed but similar patterns spanning over multivariables from the entire dataset. Furthermore, a hybrid spatial-temporal encoder (HSTEncoder) is designed for kNN-MTS which can capture both long-term temporal and short-term spatial-temporal dependencies and is shown to provide accurate representation for kNN-MTSfor better forecasting. Experimental results on several real-world datasets show a significant improvement in the forecasting performance of kNN-MTS. The quantitative analysis also illustrates the interpretability and efficiency of kNN-MTS, showing better application prospects and opening up a new path for efficiently using the large dataset in MTS models. 

**Abstract (ZH)**: 多变量时间序列（MTS）forecasting在工业和学术领域有广泛的应用。近年来，空时图神经网络（STGNNs）因其在MTS forecasting中的有效性而受到关注。然而，当前的STGNNs由于计算复杂性限制了只能使用有限长度的MTS输入数据。此外，它们缺乏在整套数据集中识别相似模式的能力，并且难以处理在长历史时期内变量间表现为稀疏且断续分布相关性的数据，这仅带来了微小的改进。本文介绍了一种简单而有效的k近邻MTS forecasting（kNN-MTS）框架，该框架通过在缓存序列的大数据集中使用MTS模型的表示进行相似搜索，以最近邻检索机制进行预测。这种方法无需额外训练，并且在测试时可直接为MTS模型提供整个数据集的访问权限，从而获得一个表达性强且性能持续提升的模型，能够从整套数据集中提取稀疏分布但仍相似的模式跨越多个变量。此外，为了kNN-MTS设计了一种混合空时编码器（HSTEncoder），它可以捕捉长短期时间和空间依赖关系，并且证明对kNN-MTS的表示提供了更准确的支持以提高预测准确性。在多个真实世界数据集上的实验结果显示，kNN-MTS的预测性能显著提高。定量分析还展示了kNN-MTS的可解释性和效率，显示出更好的应用前景，并为高效利用MTS模型中的大规模数据集开辟了一条新路径。 

---
# A Classical View on Benign Overfitting: The Role of Sample Size 

**Title (ZH)**: 关于良性过拟合的传统视角：样本大小的作用 

**Authors**: Junhyung Park, Patrick Bloebaum, Shiva Prasad Kasiviswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11621)  

**Abstract**: Benign overfitting is a phenomenon in machine learning where a model perfectly fits (interpolates) the training data, including noisy examples, yet still generalizes well to unseen data. Understanding this phenomenon has attracted considerable attention in recent years. In this work, we introduce a conceptual shift, by focusing on almost benign overfitting, where models simultaneously achieve both arbitrarily small training and test errors. This behavior is characteristic of neural networks, which often achieve low (but non-zero) training error while still generalizing well. We hypothesize that this almost benign overfitting can emerge even in classical regimes, by analyzing how the interaction between sample size and model complexity enables larger models to achieve both good training fit but still approach Bayes-optimal generalization. We substantiate this hypothesis with theoretical evidence from two case studies: (i) kernel ridge regression, and (ii) least-squares regression using a two-layer fully connected ReLU neural network trained via gradient flow. In both cases, we overcome the strong assumptions often required in prior work on benign overfitting.
Our results on neural networks also provide the first generalization result in this setting that does not rely on any assumptions about the underlying regression function or noise, beyond boundedness. Our analysis introduces a novel proof technique based on decomposing the excess risk into estimation and approximation errors, interpreting gradient flow as an implicit regularizer, that helps avoid uniform convergence traps. This analysis idea could be of independent interest. 

**Abstract (ZH)**: 良性过拟合是机器学习中的一个现象，模型能够完美拟合（内插）训练数据，包括噪声样本，但仍能很好地泛化到未见数据。近年来，对这一现象的理解引起了广泛关注。在本文中，我们引入了一个概念性的转变，重点关注近乎良性过拟合，即模型同时实现任意小的训练误差和测试误差。这种行为特征于神经网络，它们通常能够实现低（但非零）的训练误差同时仍能很好地泛化。我们假设通过分析样本数量和模型复杂度之间的相互作用，即使在经典范式中，更大的模型也能同时实现良好的训练拟合和接近贝叶斯最优的泛化。我们通过两个案例研究的理论证据来验证这一假设：（i）核岭回归；（ii）通过梯度流训练的两层全连接ReLU神经网络的最小二乘回归。在这两个案例中，我们克服了之前关于良性过拟合研究中常用的强假设。我们的结果还提供了第一个在此设置中不依赖于回归函数或噪声假设（仅限于有界性）的泛化结果。我们的分析引入了一种基于将超额风险分解为估计误差和逼近误差的新颖证明技术，将梯度流解释为一种隐含正则化器，这有助于避免均匀收敛陷阱。这一分析思路可能具有独立的研究兴趣。 

---
# Steering Risk Preferences in Large Language Models by Aligning Behavioral and Neural Representations 

**Title (ZH)**: 通过行为和神经表示对齐引导大型语言模型的风险偏好 

**Authors**: Jian-Qiao Zhu, Haijiang Yan, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.11615)  

**Abstract**: Changing the behavior of large language models (LLMs) can be as straightforward as editing the Transformer's residual streams using appropriately constructed "steering vectors." These modifications to internal neural activations, a form of representation engineering, offer an effective and targeted means of influencing model behavior without retraining or fine-tuning the model. But how can such steering vectors be systematically identified? We propose a principled approach for uncovering steering vectors by aligning latent representations elicited through behavioral methods (specifically, Markov chain Monte Carlo with LLMs) with their neural counterparts. To evaluate this approach, we focus on extracting latent risk preferences from LLMs and steering their risk-related outputs using the aligned representations as steering vectors. We show that the resulting steering vectors successfully and reliably modulate LLM outputs in line with the targeted behavior. 

**Abstract (ZH)**: 通过编辑Transformer的残差流使用适当构造的“导向向量”可以改变大型语言模型（LLMs）的行为。这些对内部神经激活的修改，作为一种表示工程形式，提供了一种有效且针对性的方法来影响模型行为而无需重新训练或微调模型。但如何系统地识别这些导向向量？我们提出了一种原理性的方法，通过对行为方法（特别是通过LLMs的马尔可夫链蒙特卡洛方法）引发的潜在表示与其神经对应的表示进行对齐来发现导向向量。为了评估此方法，我们将重点提取LLMs中的潜在风险偏好，并使用对齐的表示作为导向向量引导其风险相关输出。我们展示了所得到的导向向量能够成功且可靠地按目标行为调节LLMs的输出。 

---
# Continuous Optimization for Feature Selection with Permutation-Invariant Embedding and Policy-Guided Search 

**Title (ZH)**: 连续优化特征选择中的置换不变嵌入与策略导向搜索 

**Authors**: Rui Liu, Rui Xie, Zijun Yao, Yanjie Fu, Dongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11601)  

**Abstract**: Feature selection removes redundant features to enhanc performance and computational efficiency in downstream tasks. Existing works often struggle to capture complex feature interactions and adapt to diverse scenarios. Recent advances in this domain have incorporated generative intelligence to address these drawbacks by uncovering intricate relationships between features. However, two key limitations remain: 1) embedding feature subsets in a continuous space is challenging due to permutation sensitivity, as changes in feature order can introduce biases and weaken the embedding learning process; 2) gradient-based search in the embedding space assumes convexity, which is rarely guaranteed, leading to reduced search effectiveness and suboptimal subsets. To address these limitations, we propose a new framework that can: 1) preserve feature subset knowledge in a continuous embedding space while ensuring permutation invariance; 2) effectively explore the embedding space without relying on strong convex assumptions. For the first objective, we develop an encoder-decoder paradigm to preserve feature selection knowledge into a continuous embedding space. This paradigm captures feature interactions through pairwise relationships within the subset, removing the influence of feature order on the embedding. Moreover, an inducing point mechanism is introduced to accelerate pairwise relationship computations. For the second objective, we employ a policy-based reinforcement learning (RL) approach to guide the exploration of the embedding space. The RL agent effectively navigates the space by balancing multiple objectives. By prioritizing high-potential regions adaptively and eliminating the reliance on convexity assumptions, the RL agent effectively reduces the risk of converging to local optima. Extensive experiments demonstrate the effectiveness, efficiency, robustness and explicitness of our model. 

**Abstract (ZH)**: 特征选择通过去除冗余特征来提升下游任务的性能和计算效率。现有工作常难以捕捉复杂的特征交互并适应多样的场景。近期该领域的进展通过生成智能来捕捉特征间的复杂关系以解决这些问题。然而，仍存在两个关键限制：1）在连续空间中嵌入特征子集由于排列敏感性而具有挑战性，特征顺序变化可能导致偏差并削弱嵌入学习过程；2）基于梯度的嵌入空间搜索假设凸性，这很少得到保证，从而降低了搜索效果并导致次优子集。为解决这些限制，我们提出了一种新框架，可实现：1）在连续嵌入空间中保留特征子集知识同时确保排列不变性；2）在不依赖强凸性假设的情况下有效探索嵌入空间。为实现第一项目标，我们开发了一种编码器-解码器范式，将特征选择知识保留到连续嵌入空间中。该范式通过子集中对对间的相互作用捕捉特征交互，从而消除特征顺序对嵌入的影响。此外，我们引入了一种引点机制以加速对对间关系的计算。为实现第二项目标，我们采用基于策略的强化学习（RL）方法指导嵌入空间的探索。RL代理通过平衡多个目标高效导航空间，通过适 Celebrating Spring Festival 

---
# Spectral Policy Optimization: Coloring your Incorrect Reasoning in GRPO 

**Title (ZH)**: 光谱策略优化：在GRPO中着色你的错误推理 

**Authors**: Peter Chen, Xiaopeng Li, Ziniu Li, Xi Chen, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.11595)  

**Abstract**: Reinforcement learning (RL) has demonstrated significant success in enhancing reasoning capabilities in large language models (LLMs). One of the most widely used RL methods is Group Relative Policy Optimization (GRPO)~\cite{Shao-2024-Deepseekmath}, known for its memory efficiency and success in training DeepSeek-R1~\cite{Guo-2025-Deepseek}. However, GRPO stalls when all sampled responses in a group are incorrect -- referred to as an \emph{all-negative-sample} group -- as it fails to update the policy, hindering learning progress. The contributions of this paper are two-fold. First, we propose a simple yet effective framework that introduces response diversity within all-negative-sample groups in GRPO using AI feedback. We also provide a theoretical analysis, via a stylized model, showing how this diversification improves learning dynamics. Second, we empirically validate our approach, showing the improved performance across various model sizes (7B, 14B, 32B) in both offline and online learning settings with 10 benchmarks, including base and distilled variants. Our findings highlight that learning from all-negative-sample groups is not only feasible but beneficial, advancing recent insights from \citet{Xiong-2025-Minimalist}. 

**Abstract (ZH)**: reinforcement学习（RL）在增强大规模语言模型（LLMs）的推理能力方面取得了显著成功。Group Relative Policy Optimization（GRPO）是最广泛使用的方法之一，因其内存效率和训练DeepSeek-R1的成功而闻名。然而，当所有采样的响应在一组中都是错误的——称为“全负样本组”——GRPO会停滞不前，因为它无法更新策略，阻碍了学习进程。本文的主要贡献有两个方面。首先，我们提出了一种简单而有效的方法，该方法利用AI反馈在GRPO的“全负样本组”中引入响应多样性。我们还通过简化模型提供了理论分析，说明这种多样化如何提高学习动态。其次，我们在离线和在线学习设置中，通过10个基准测试（包括基本和精简变体），证明了这种方法在不同模型大小（7B、14B、32B）下的改进性能。我们的研究结果表明，从“全负样本组”中学习不仅是可行的，而且是有益的，这与最近的研究见解《Xiong-2025-Minimalist》相一致。 

---
# SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training 

**Title (ZH)**: SageAttention3: 微缩FP4注意力机制用于推理及8位训练探索 

**Authors**: Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11594)  

**Abstract**: The efficiency of attention is important due to its quadratic time complexity. We enhance the efficiency of attention through two key contributions: First, we leverage the new FP4 Tensor Cores in Blackwell GPUs to accelerate attention computation. Our implementation achieves 1038 TOPS on RTX5090, which is a 5x speedup over the fastest FlashAttention on RTX5090. Experiments show that our FP4 attention can accelerate inference of various models in a plug-and-play way. Second, we pioneer low-bit attention to training tasks. Existing low-bit attention works like FlashAttention3 and SageAttention focus only on inference. However, the efficiency of training large models is also important. To explore whether low-bit attention can be effectively applied to training tasks, we design an accurate and efficient 8-bit attention for both forward and backward propagation. Experiments indicate that 8-bit attention achieves lossless performance in fine-tuning tasks but exhibits slower convergence in pretraining tasks. The code will be available at this https URL. 

**Abstract (ZH)**: 注意力机制的效率至关重要，因为它具有二次时间复杂度。我们通过两大贡献提升了注意力机制的效率：首先，我们利用Blackwell GPU中的新型FP4张量核心加速注意力计算。我们的实现达到了RTX5090上的1038 TOPS，比RTX5090上最快的FlashAttention快5倍。实验表明，我们的FP4注意力机制可以在各种模型中以即插即用的方式加速推理。其次，我们首次将低位宽注意力机制应用于训练任务。现有的低位宽注意力机制如FlashAttention3和SageAttention仅关注推理。然而，大型模型的训练效率同样重要。为探索低位宽注意力机制是否可以有效应用于训练任务，我们设计了一种适用于前向和反向传播的准确高效8位注意力机制。实验结果显示，在微调任务中，8位注意力机制达到了无损性能，但在预训练任务中表现出较慢的收敛速度。代码将在以下链接处提供：[this https URL]。 

---
# The Ripple Effect: On Unforeseen Complications of Backdoor Attacks 

**Title (ZH)**: 涟漪效应：后门攻击的未预见并发症 

**Authors**: Rui Zhang, Yun Shen, Hongwei Li, Wenbo Jiang, Hanxiao Chen, Yuan Zhang, Guowen Xu, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11586)  

**Abstract**: Recent research highlights concerns about the trustworthiness of third-party Pre-Trained Language Models (PTLMs) due to potential backdoor attacks. These backdoored PTLMs, however, are effective only for specific pre-defined downstream tasks. In reality, these PTLMs can be adapted to many other unrelated downstream tasks. Such adaptation may lead to unforeseen consequences in downstream model outputs, consequently raising user suspicion and compromising attack stealthiness. We refer to this phenomenon as backdoor complications. In this paper, we undertake the first comprehensive quantification of backdoor complications. Through extensive experiments using 4 prominent PTLMs and 16 text classification benchmark datasets, we demonstrate the widespread presence of backdoor complications in downstream models fine-tuned from backdoored PTLMs. The output distribution of triggered samples significantly deviates from that of clean samples. Consequently, we propose a backdoor complication reduction method leveraging multi-task learning to mitigate complications without prior knowledge of downstream tasks. The experimental results demonstrate that our proposed method can effectively reduce complications while maintaining the efficacy and consistency of backdoor attacks. Our code is available at this https URL. 

**Abstract (ZH)**: 近年来的研究强调了第三方预训练语言模型(PTLMs)的信任问题，因为可能存在后门攻击。然而，这些植入后门的PTLMs只对特定的预定义下游任务有效。实际上，这些PTLMs可以适应许多其他无关的下游任务。这种适应可能导致下游模型输出的不可预见后果，从而引起用户怀疑并损害攻击的隐蔽性。我们将这一现象称为后门复杂性。在本文中，我们首次对后门复杂性进行了全面量化。通过使用4个主流的PTLMs和16个文本分类基准数据集进行广泛的实验，我们展示了从植入后门的PTLMs微调的下游模型中普遍存在后门复杂性。触发样本的输出分布与干净样本的输出分布显著不同。因此，我们提出了一种利用多任务学习的方法来减少后门复杂性，而无需了解下游任务的先验知识。实验结果表明，我们提出的方法可以在保持后门攻击有效性及一致性的前提下有效减少复杂性。我们的代码可在以下链接获取：this https URL。 

---
# Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents 

**Title (ZH)**: 比较_lexical_和_semantic_向量搜索方法在分类医疗文档中的效果 

**Authors**: Lee Harris, Philippe De Wilde, James Bentham  

**Link**: [PDF](https://arxiv.org/pdf/2505.11582)  

**Abstract**: Classification is a common AI problem, and vector search is a typical solution. This transforms a given body of text into a numerical representation, known as an embedding, and modern improvements to vector search focus on optimising speed and predictive accuracy. This is often achieved through neural methods that aim to learn language semantics. However, our results suggest that these are not always the best solution. Our task was to classify rigidly-structured medical documents according to their content, and we found that using off-the-shelf semantic vector search produced slightly worse predictive accuracy than creating a bespoke lexical vector search model, and that it required significantly more time to execute. These findings suggest that traditional methods deserve to be contenders in the information retrieval toolkit, despite the prevalence and success of neural models. 

**Abstract (ZH)**: 分类是常见的AI问题，向量搜索是典型解决方案。这将给定的文本转换为数值表示，称为嵌入，现代向量搜索的改进重点在于优化速度和预测准确性。这通常通过旨在学习语言语义的神经方法来实现。然而，我们的结果显示这并不总是最佳解决方案。我们的任务是根据内容对结构严谨的医疗文件进行分类，我们发现使用现成的语义向量搜索产生的预测准确性略差于创建专门的词汇向量搜索模型，并且执行时间显著更长。这些发现表明，尽管神经模型的流行和成功，传统方法仍应在信息检索工具箱中占有一席之地。 

---
# Flash Invariant Point Attention 

**Title (ZH)**: 闪光不变点注意机制 

**Authors**: Andrew Liu, Axel Elaldi, Nicholas T Franklin, Nathan Russell, Gurinder S Atwal, Yih-En A Ban, Olivia Viessmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.11580)  

**Abstract**: Invariant Point Attention (IPA) is a key algorithm for geometry-aware modeling in structural biology, central to many protein and RNA models. However, its quadratic complexity limits the input sequence length. We introduce FlashIPA, a factorized reformulation of IPA that leverages hardware-efficient FlashAttention to achieve linear scaling in GPU memory and wall-clock time with sequence length. FlashIPA matches or exceeds standard IPA performance while substantially reducing computational costs. FlashIPA extends training to previously unattainable lengths, and we demonstrate this by re-training generative models without length restrictions and generating structures of thousands of residues. FlashIPA is available at this https URL. 

**Abstract (ZH)**: FlashIPA：一种利用FlashAttention实现线性扩展的Invariant Point Attention算法 

---
# Toward Adaptive Categories: Dimensional Governance for Agentic AI 

**Title (ZH)**: 面向自适应类别的维度治理：赋能型AI管理 

**Authors**: Zeynep Engin, David Hand  

**Link**: [PDF](https://arxiv.org/pdf/2505.11579)  

**Abstract**: As AI systems evolve from static tools to dynamic agents, traditional categorical governance frameworks -- based on fixed risk tiers, levels of autonomy, or human oversight models -- are increasingly insufficient on their own. Systems built on foundation models, self-supervised learning, and multi-agent architectures increasingly blur the boundaries that categories were designed to police. In this Perspective, we make the case for dimensional governance: a framework that tracks how decision authority, process autonomy, and accountability (the 3As) distribute dynamically across human-AI relationships. A critical advantage of this approach is its ability to explicitly monitor system movement toward and across key governance thresholds, enabling preemptive adjustments before risks materialize. This dimensional approach provides the necessary foundation for more adaptive categorization, enabling thresholds and classifications that can evolve with emerging capabilities. While categories remain essential for decision-making, building them upon dimensional foundations allows for context-specific adaptability and stakeholder-responsive governance that static approaches cannot achieve. We outline key dimensions, critical trust thresholds, and practical examples illustrating where rigid categorical frameworks fail -- and where a dimensional mindset could offer a more resilient and future-proof path forward for both governance and innovation at the frontier of artificial intelligence. 

**Abstract (ZH)**: 随着AI系统从静态工具演变成动态代理，基于固定风险等级、自主水平或人类监管模式的传统分类治理框架日益显得不足。基于基础模型、自监督学习和多代理架构的系统逐渐模糊了分类设计时划清的边界。在本文中，我们提出了维度治理的概念：一种追踪决策权、过程自主性和问责性（3As）在人机关系中动态分配的框架。这一方法的关键优势在于其能够明确监控系统向关键治理门槛靠近和跨越的过程，从而在风险显现之前进行预先调整。维度方法为更适应性的分类提供了必要的基础，使其分类和门槛能够随着新兴能力的发展而演变。虽然分类在决策中仍至关重要，但在维度基础上构建分类能够实现特定情境下的适应性和利益相关方响应型治理，这是静态方法无法实现的。我们概述了关键维度、关键信任阈值，并举例说明了僵化分类框架失败的地方，以及维度思维如何为人工智能前沿的治理和创新提供更具弹性和前瞻性的道路。 

---
# Spatiotemporal Field Generation Based on Hybrid Mamba-Transformer with Physics-informed Fine-tuning 

**Title (ZH)**: 基于混合Mamba-Transformer的物理约束细调时空场生成 

**Authors**: Peimian Du, Jiabin Liu, Xiaowei Jin, Mengwang Zuo, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11578)  

**Abstract**: This research confronts the challenge of substantial physical equation discrepancies encountered in the generation of spatiotemporal physical fields through data-driven trained models. A spatiotemporal physical field generation model, named HMT-PF, is developed based on the hybrid Mamba-Transformer architecture, incorporating unstructured grid information as input. A fine-tuning block, enhanced with physical information, is introduced to effectively reduce the physical equation discrepancies. The physical equation residuals are computed through a point query mechanism for efficient gradient evaluation, then encoded into latent space for refinement. The fine-tuning process employs a self-supervised learning approach to achieve physical consistency while maintaining essential field characteristics. Results show that the hybrid Mamba-Transformer model achieves good performance in generating spatiotemporal fields, while the physics-informed fine-tuning mechanism further reduces significant physical errors effectively. A MSE-R evaluation method is developed to assess the accuracy and realism of physical field generation. 

**Abstract (ZH)**: 基于混合Mamba-Transformer架构的HMT-PF时空物理场生成模型及其物理信息驱动的精细调整机制 

---
# The Accountability Paradox: How Platform API Restrictions Undermine AI Transparency Mandates 

**Title (ZH)**: 平台API限制如何削弱AI透明度要求的问责悖论 

**Authors**: FLorian A.D. Burnat, Brittany I. Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2505.11577)  

**Abstract**: Recent application programming interface (API) restrictions on major social media platforms challenge compliance with the EU Digital Services Act [20], which mandates data access for algorithmic transparency. We develop a structured audit framework to assess the growing misalignment between regulatory requirements and platform implementations. Our comparative analysis of X/Twitter, Reddit, TikTok, and Meta identifies critical ``audit blind-spots'' where platform content moderation and algorithmic amplification remain inaccessible to independent verification. Our findings reveal an ``accountability paradox'': as platforms increasingly rely on AI systems, they simultaneously restrict the capacity for independent oversight. We propose targeted policy interventions aligned with the AI Risk Management Framework of the National Institute of Standards and Technology [80], emphasizing federated access models and enhanced regulatory enforcement. 

**Abstract (ZH)**: 近期，主要社交媒体平台的API限制挑战了欧盟数字服务法案对算法透明度的数据访问要求。我们开发了一个结构化的审计框架来评估监管要求与平台实施之间日益增长的不一致。通过对X/Twitter、Reddit、TikTok和Meta的比较分析，我们识别出关键的“审计盲区”，其中平台内容审核和算法放大仍不可独立验证。我们的研究揭示了一个“问责制悖论”：尽管平台越来越依赖AI系统，但同时限制了独立监督的能力。我们建议与国家标准与技术研究院的AI风险管理框架相一致的针对性政策干预措施，强调联邦访问模型和增强的监管执法。 

---
# Concept-Guided Interpretability via Neural Chunking 

**Title (ZH)**: 概念引导的神经切分可解释性 

**Authors**: Shuchen Wu, Stephan Alaniz, Shyamgopal Karthik, Peter Dayan, Eric Schulz, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2505.11576)  

**Abstract**: Neural networks are often black boxes, reflecting the significant challenge of understanding their internal workings. We propose a different perspective that challenges the prevailing view: rather than being inscrutable, neural networks exhibit patterns in their raw population activity that mirror regularities in the training data. We refer to this as the Reflection Hypothesis and provide evidence for this phenomenon in both simple recurrent neural networks (RNNs) and complex large language models (LLMs). Building on this insight, we propose to leverage cognitively-inspired methods of chunking to segment high-dimensional neural population dynamics into interpretable units that reflect underlying concepts. We propose three methods to extract these emerging entities, complementing each other based on label availability and dimensionality. Discrete sequence chunking (DSC) creates a dictionary of entities; population averaging (PA) extracts recurring entities that correspond to known labels; and unsupervised chunk discovery (UCD) can be used when labels are absent. We demonstrate the effectiveness of these methods in extracting entities across varying model sizes, ranging from inducing compositionality in RNNs to uncovering recurring neural population states in large models with diverse architectures, and illustrate their advantage over other methods. Throughout, we observe a robust correspondence between the extracted entities and concrete or abstract concepts. Artificially inducing the extracted entities in neural populations effectively alters the network's generation of associated concepts. Our work points to a new direction for interpretability, one that harnesses both cognitive principles and the structure of naturalistic data to reveal the hidden computations of complex learning systems, gradually transforming them from black boxes into systems we can begin to understand. 

**Abstract (ZH)**: 神经网络往往是黑盒，反映了理解其内部工作机制的重大挑战。本文提出了一种不同的视角，挑战现有的观点：与其说是不可解读的，神经网络在其原始群体活动中表现出与其训练数据中规律性相映射的模式。我们将此称为反射假设，并在简单的递归神经网络（RNNs）和复杂的大型语言模型（LLMs）中提供了证据。基于这一见解，我们提出利用认知启发式的分组方法来将高维神经群体动力学分割为反映潜在概念的可解释单元。我们提出了三种方法来提取这些新兴实体，这些方法根据标签可用性和维度互补。离散序列分组（DSC）创建了实体词典；群体平均（PA）提取与已知标签对应的重复出现的实体；而未监督分组发现（UCD）可以在缺少标签时使用。我们展示了这些方法在不同模型尺寸下的有效性，从在RNN中诱导组成性到在具有多种架构的大模型中发现重复的神经群体状态，并说明了它们与其他方法相比的优势。在整个过程中，我们观察到提取出的实体与其具体或抽象概念之间存在稳健的一致性。有目的地在神经群体中诱导提取出的实体，有效改变了网络生成相关概念的方式。我们的工作指出了一个新的可解释性方向，这一方向结合了认知原则和自然数据结构来揭示复杂学习系统的隐藏计算，逐步将它们从黑盒转变为可理解的系统。 

---
# InfiJanice: Joint Analysis and In-situ Correction Engine for Quantization-Induced Math Degradation in Large Language Models 

**Title (ZH)**: InfiJanice：针对大型语言模型中量化引起数学退化的同时分析和原位校正引擎 

**Authors**: Zhen Li, Yupeng Su, Songmiao Wang, Runming Yang, Congkai Xie, Aofan Liu, Ming Li, Jiannong Cao, Yuan Xie, Ngai Wong, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11574)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance on complex reasoning benchmarks such as GSM8K, MATH, and AIME. However, the substantial computational demands of these tasks pose significant challenges for real-world deployment. Model quantization has emerged as a promising approach to reduce memory footprint and inference latency by representing weights and activations with lower bit-widths. In this work, we conduct a comprehensive study of mainstream quantization methods(e.g., AWQ, GPTQ, SmoothQuant) on the most popular open-sourced models (e.g., Qwen2.5, LLaMA3 series), and reveal that quantization can degrade mathematical reasoning accuracy by up to 69.81%. To better understand this degradation, we develop an automated assignment and judgment pipeline that qualitatively categorizes failures into four error types and quantitatively identifies the most impacted reasoning capabilities. Building on these findings, we employ an automated data-curation pipeline to construct a compact "Silver Bullet" datasets. Training a quantized model on as few as 332 carefully selected examples for just 3-5 minutes on a single GPU is enough to restore its reasoning accuracy to match that of the full-precision baseline. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在GSM8K、MATH和AIME等复杂推理基准测试中展现了令人印象深刻的性能。然而，这些任务的显著计算需求给实际部署带来了重大挑战。模型量化已成为一种有前途的方法，通过使用较低位宽表示权重和激活，减少内存占用和推理延迟。在本文中，我们对主流量化方法（例如AWQ、GPTQ、SmoothQuant）在最受欢迎的开源模型（例如Qwen2.5、LLaMA3系列）上的性能进行了全面研究，并发现量化可能导致数学推理准确性下降高达69.81%。为了更好地理解这种下降，我们开发了一种自动化分配和判断管道，通过定性分类错误类型和定量识别受影响最大的推理能力来深入理解。基于这些发现，我们使用自动化数据整理管道构建了一个紧凑的“银弹”数据集。仅使用332个精心选择的示例在单个GPU上训练3-5分钟即可使量化模型的推理准确性恢复到全精度基线的水平。 

---
# Tool-Aided Evolutionary LLM for Generative Policy Toward Efficient Resource Management in Wireless Federated Learning 

**Title (ZH)**: 工具辅助进化大语言模型用于生成性策略以实现无线联邦学习中高效资源管理 

**Authors**: Chongyang Tan, Ruoqi Wen, Rongpeng Li, Zhifeng Zhao, Ekram Hossain, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11570)  

**Abstract**: Federated Learning (FL) enables distributed model training across edge devices in a privacy-friendly manner. However, its efficiency heavily depends on effective device selection and high-dimensional resource allocation in dynamic and heterogeneous wireless environments. Conventional methods demand a confluence of domain-specific expertise, extensive hyperparameter tuning, and/or heavy interaction cost. This paper proposes a Tool-aided Evolutionary Large Language Model (T-ELLM) framework to generate a qualified policy for device selection in a wireless FL environment. Unlike conventional optimization methods, T-ELLM leverages natural language-based scenario prompts to enhance generalization across varying network conditions. The framework decouples the joint optimization problem mathematically, enabling tractable learning of device selection policies while delegating resource allocation to convex optimization tools. To improve adaptability, T-ELLM integrates a sample-efficient, model-based virtual learning environment that captures the relationship between device selection and learning performance, facilitating subsequent group relative policy optimization. This concerted approach reduces reliance on real-world interactions, minimizing communication overhead while maintaining high-fidelity decision-making. Theoretical analysis proves that the discrepancy between virtual and real environments is bounded, ensuring the advantage function learned in the virtual environment maintains a provably small deviation from real-world conditions. Experimental results demonstrate that T-ELLM outperforms benchmark methods in energy efficiency and exhibits robust adaptability to environmental changes. 

**Abstract (ZH)**: 联邦学习（FL）以隐私友好的方式在边缘设备上进行分布式模型训练。然而，其效率高度依赖于在动态和异构无线环境中有效的设备选择和高维资源分配。传统方法需要领域特定的专业知识、广泛的超参数调整和/或高昂的交互成本。本文提出了一种工具辅助进化大规模语言模型（T-ELLM）框架，用于生成适合无线FL环境中的设备选择策略。与传统优化方法不同，T-ELLM利用基于自然语言的场景提示来增强在不同网络条件下的一般化能力。该框架在数学上分离了联合优化问题，使设备选择策略的学习变得可行，同时将资源分配委托给凸优化工具。为了提高适应性，T-ELLM集成了一个高效采样的基于模型的虚拟学习环境，捕捉设备选择与学习性能之间的关系，促进后续群体相对策略优化。这种综合性方法减少了对真实世界交互的依赖，同时最小化通信开销并保持高水平的决策准确性。理论分析证明，虚拟环境与现实环境之间的差异是可以限制的，确保在虚拟环境中学习到的优势函数与现实世界条件之间的偏差是有保证的小。实验结果表明，T-ELLM在能效方面优于基准方法，并且在环境变化中表现出高度的鲁棒适应性。 

---
# Towards Adaptive Deep Learning: Model Elasticity via Prune-and-Grow CNN Architectures 

**Title (ZH)**: 面向自适应深度学习：通过剪枝与再生CNN架构实现模型弹性 

**Authors**: Pooja Mangal, Sudaksh Kalra, Dolly Sapra  

**Link**: [PDF](https://arxiv.org/pdf/2505.11569)  

**Abstract**: Deploying deep convolutional neural networks (CNNs) on resource-constrained devices presents significant challenges due to their high computational demands and rigid, static architectures. To overcome these limitations, this thesis explores methods for enabling CNNs to dynamically adjust their computational complexity based on available hardware resources. We introduce adaptive CNN architectures capable of scaling their capacity at runtime, thus efficiently balancing performance and resource utilization. To achieve this adaptability, we propose a structured pruning and dynamic re-construction approach that creates nested subnetworks within a single CNN model. This approach allows the network to dynamically switch between compact and full-sized configurations without retraining, making it suitable for deployment across varying hardware platforms. Experiments conducted across multiple CNN architectures including VGG-16, AlexNet, ResNet-20, and ResNet-56 on CIFAR-10 and Imagenette datasets demonstrate that adaptive models effectively maintain or even enhance performance under varying computational constraints. Our results highlight that embedding adaptability directly into CNN architectures significantly improves their robustness and flexibility, paving the way for efficient real-world deployment in diverse computational environments. 

**Abstract (ZH)**: 在资源受限设备上部署深度卷积神经网络（CNNs）由于其高计算需求和固定的刚性架构而面临重大挑战。为克服这些限制，本论文探讨了使CNN能够根据可用硬件资源动态调整其计算复杂性的方法。我们提出了能够运行时动态调整容量的自适应CNN架构，从而高效地平衡性能和资源利用率。为实现这种自适应性，我们提出了一种结构化剪枝和动态重构的方法，在单一CNN模型中创建嵌套子网络。这种方法允许网络在不需要重新训练的情况下，在紧凑和全尺寸配置之间动态切换，从而适用于不同硬件平台的部署。我们在包括VGG-16、AlexNet、ResNet-20和ResNet-56的多种CNN架构上，在CIFAR-10和Imagenette数据集上的实验表明，自适应模型能够有效地在不同计算约束条件下维持甚至提升性能。我们的结果表明，将自适应性直接嵌入到CNN架构中显著提高了其鲁棒性和灵活性，为在各种计算环境中高效实际部署铺平了道路。 

---
# BioCube: A Multimodal Dataset for Biodiversity Research 

**Title (ZH)**: BioCube: 多模态生物多样性研究数据集 

**Authors**: Stylianos Stasinos, Martino Mensio, Elena Lazovik, Athanasios Trantas  

**Link**: [PDF](https://arxiv.org/pdf/2505.11568)  

**Abstract**: Biodiversity research requires complete and detailed information to study ecosystem dynamics at different scales. Employing data-driven methods like Machine Learning is getting traction in ecology and more specific biodiversity, offering alternative modelling pathways. For these methods to deliver accurate results there is the need for large, curated and multimodal datasets that offer granular spatial and temporal resolutions. In this work, we introduce BioCube, a multimodal, fine-grained global dataset for ecology and biodiversity research. BioCube incorporates species observations through images, audio recordings and descriptions, environmental DNA, vegetation indices, agricultural, forest, land indicators, and high-resolution climate variables. All observations are geospatially aligned under the WGS84 geodetic system, spanning from 2000 to 2020. The dataset will become available at this https URL while the acquisition and processing code base at this https URL. 

**Abstract (ZH)**: 生物多样性研究需要全面和详细的信息以在不同尺度上研究生态系统的动态。利用如机器学习等数据驱动方法在生态学和特定生物多样性的研究中越来越受到关注，提供了新的建模途径。为了使这些方法能够提供准确的结果，需要大型、经过整理的多模态数据集，这些数据集能够提供精细的空间和时间分辨率。本文引入了BioCube，这是一个适用于生态学和生物多样性研究的多模态、高细粒度全球数据集。BioCube将物种观察数据通过图像、音频记录和描述、环境DNA、植被指数、农业和森林指标以及高分辨率气候变量进行整合。所有观测数据在WGS84大地坐标系统下进行地理对齐，时间跨度从2000年到2020年。数据集将在以下链接处提供：[提供链接]，数据获取和处理代码库将在以下链接处提供：[提供链接]。 

---
# Beyond Time: Cross-Dimensional Frequency Supervision for Time Series Forecasting 

**Title (ZH)**: 超越时间：跨维度频率监督在时间序列预测中的应用 

**Authors**: Tianyi Shi, Zhu Meng, Yue Chen, Siyang Zheng, Fei Su, Jin Huang, Changrui Ren, Zhicheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11567)  

**Abstract**: Time series forecasting plays a crucial role in various fields, and the methods based on frequency domain analysis have become an important branch. However, most existing studies focus on the design of elaborate model architectures and are often tailored for limited datasets, still lacking universality. Besides, the assumption of independent and identically distributed (IID) data also contradicts the strong correlation of the time domain labels. To address these issues, abandoning time domain supervision, we propose a purely frequency domain supervision approach named cross-dimensional frequency (X-Freq) loss. Specifically, based on a statistical phenomenon, we first prove that the information entropy of the time series is higher than its spectral entropy, which implies higher certainty in frequency domain and thus can provide better supervision. Secondly, the Fourier Transform and the Wavelet Transform are applied to the time dimension and the channel dimension of the time series respectively, to capture the long-term and short-term frequency variations as well as the spatial configuration features. Thirdly, the loss between predictions and targets is uniformly computed in the frequency domain. Moreover, we plug-and-play incorporate X-Freq into multiple advanced forecasting models and compare on 14 real-world datasets. The experimental results demonstrate that, without making any modification to the original architectures or hyperparameters, X-Freq can improve the forecasting performance by an average of 3.3% on long-term forecasting datasets and 27.7% on short-term ones, showcasing superior generality and practicality. The code will be released publicly. 

**Abstract (ZH)**: 基于频域的跨维度频率损失在时间序列预测中的应用 

---
# ACSE-Eval: Can LLMs threat model real-world cloud infrastructure? 

**Title (ZH)**: ACSE-Eval: LLMs能否对现实世界的云基础设施进行威胁建模？ 

**Authors**: Sarthak Munshi, Swapnil Pathak, Sonam Ghatode, Thenuga Priyadarshini, Dhivya Chandramouleeswaran, Ashutosh Rana  

**Link**: [PDF](https://arxiv.org/pdf/2505.11565)  

**Abstract**: While Large Language Models have shown promise in cybersecurity applications, their effectiveness in identifying security threats within cloud deployments remains unexplored. This paper introduces AWS Cloud Security Engineering Eval, a novel dataset for evaluating LLMs cloud security threat modeling capabilities. ACSE-Eval contains 100 production grade AWS deployment scenarios, each featuring detailed architectural specifications, Infrastructure as Code implementations, documented security vulnerabilities, and associated threat modeling parameters. Our dataset enables systemic assessment of LLMs abilities to identify security risks, analyze attack vectors, and propose mitigation strategies in cloud environments. Our evaluations on ACSE-Eval demonstrate that GPT 4.1 and Gemini 2.5 Pro excel at threat identification, with Gemini 2.5 Pro performing optimally in 0-shot scenarios and GPT 4.1 showing superior results in few-shot settings. While GPT 4.1 maintains a slight overall performance advantage, Claude 3.7 Sonnet generates the most semantically sophisticated threat models but struggles with threat categorization and generalization. To promote reproducibility and advance research in automated cybersecurity threat analysis, we open-source our dataset, evaluation metrics, and methodologies. 

**Abstract (ZH)**: 虽然大型语言模型在 cybersecurity 应用方面显示出潜力，但在识别云部署中的安全威胁方面的效果尚未被探索。本文介绍了一个新的数据集 AWS 云安全工程评估（AWS Cloud Security Engineering Eval, ACSE-Eval），用于评估 LLMs 的云安全威胁建模能力。ACSE-Eval 包含 100 个生产级别的 AWS 部署场景，每个场景都包含详细的体系架构规范、基础设施即代码实现、记录的安全漏洞以及相关的威胁建模参数。我们的数据集使得系统性评估 LLMs 识别 security 风险、分析攻击向量及提出缓解策略的能力成为可能。我们对 ACSE-Eval 的评估表明，GPT 4.1 和 Gemini 2.5 Pro 在威胁识别方面表现突出，Gemini 2.5 Pro 在零样本场景中表现最优，而 GPT 4.1 在少量样本设置中表现更优。尽管 GPT 4.1 总体性能稍占优势，但 Claude 3.7 Sonnet 生成的威胁模型在语义复杂性方面表现最佳，但在威胁分类和泛化方面存在困难。为了促进可重复性和推进自动网络安全威胁分析的研究，我们开源了我们的数据集、评估指标和方法论。 

---
# Object-Centric Representations Improve Policy Generalization in Robot Manipulation 

**Title (ZH)**: 以对象为中心的表示可以提升机器人 manipulation 策略的泛化能力 

**Authors**: Alexandre Chapin, Bruno Machado, Emmanuel Dellandrea, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11563)  

**Abstract**: Visual representations are central to the learning and generalization capabilities of robotic manipulation policies. While existing methods rely on global or dense features, such representations often entangle task-relevant and irrelevant scene information, limiting robustness under distribution shifts. In this work, we investigate object-centric representations (OCR) as a structured alternative that segments visual input into a finished set of entities, introducing inductive biases that align more naturally with manipulation tasks. We benchmark a range of visual encoders-object-centric, global and dense methods-across a suite of simulated and real-world manipulation tasks ranging from simple to complex, and evaluate their generalization under diverse visual conditions including changes in lighting, texture, and the presence of distractors. Our findings reveal that OCR-based policies outperform dense and global representations in generalization settings, even without task-specific pretraining. These insights suggest that OCR is a promising direction for designing visual systems that generalize effectively in dynamic, real-world robotic environments. 

**Abstract (ZH)**: 视觉表示是机器人操作策略学习和泛化能力的核心。现有方法依赖于全局或密集特征，但这些表示往往会混合任务相关和无关的场景信息，限制了在分布变化下的鲁棒性。在本工作中，我们研究以对象为中心的表示（OCR）作为一种结构化的替代方案，通过将视觉输入分割成一系列独立的实体，引入与操作任务更自然对齐的归纳偏置。我们在一系列从简单到复杂的模拟和真实世界操作任务中对比了多种视觉编码器，包括以对象为中心、全局和密集方法，并评估它们在不同视觉条件下的泛化能力，包括光照、纹理变化和干扰物的存在。我们的发现表明，即使不进行特定任务的预训练，基于OCR的策略在泛化设置中也优于密集和全局表示。这些洞见表明，OCR是设计能够在动态真实世界机器人环境中有效泛化的视觉系统的一个有前途的方向。 

---
# Analysis and Resilience of the U.S. Flight Network 

**Title (ZH)**: 美国航空网络的分析与韧性研究 

**Authors**: Sushrit Kafle, Shreejan Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2505.11559)  

**Abstract**: Air travel is one of the most widely used transportation services in the United States. This paper analyzes the U.S. Flight Network (USFN) using complex network theory by exploring how the network's topology contributes to its efficiency and vulnerability. This is done by examining the structural properties, degree distributions, and community structures in the network. USFN was observed to follow power-law distribution and falls under the anomalous regime, suggesting that the network is hub dominant. Compared to null networks, USFN has a higher clustering coefficient and modularity. Various percolation test revealed that USFN is vulnerable to targeted attacks and is susceptible to complete cascading failure if one of the major hubs fails. The overall results suggest that while the USFN is designed for efficiency, it is highly vulnerable to disruptions. Protecting key hub airports is important to make the network more robust and prevent large-scale failures. 

**Abstract (ZH)**: 美国航空网络的拓扑结构与其效率和脆弱性的关系研究 

---
# AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs 

**Title (ZH)**: AC-LoRA: (几乎)无需训练的访问控制意识多模态LLM 

**Authors**: Lara Magdalena Lazier, Aritra Dhar, Vasilije Stambolic, Lukas Cavigelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.11557)  

**Abstract**: Corporate LLMs are gaining traction for efficient knowledge dissemination and management within organizations. However, as current LLMs are vulnerable to leaking sensitive information, it has proven difficult to apply them in settings where strict access control is necessary. To this end, we design AC-LoRA, an end-to-end system for access control-aware corporate LLM chatbots that maintains a strong information isolation guarantee. AC-LoRA maintains separate LoRA adapters for permissioned datasets, along with the document embedding they are finetuned on. AC-LoRA retrieves a precise set of LoRA adapters based on the similarity score with the user query and their permission. This similarity score is later used to merge the responses if more than one LoRA is retrieved, without requiring any additional training for LoRA routing. We provide an end-to-end prototype of AC-LoRA, evaluate it on two datasets, and show that AC-LoRA matches or even exceeds the performance of state-of-the-art LoRA mixing techniques while providing strong isolation guarantees. Furthermore, we show that AC-LoRA design can be directly applied to different modalities. 

**Abstract (ZH)**: 面向访问控制的企業端LoRA聊天机器人系统：AC-LoRA 

---
# Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks 

**Title (ZH)**: 基于隐藏配置任务评估多智能体大语言模型的集体推理能力 

**Authors**: Yuxuan Li, Aoi Naito, Hirokazu Shirado  

**Link**: [PDF](https://arxiv.org/pdf/2505.11556)  

**Abstract**: Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving through distributed information integration, but also risk replicating collective reasoning failures observed in human groups. Yet, no theory-grounded benchmark exists to systematically evaluate such failures. In this paper, we introduce the Hidden Profile paradigm from social psychology as a diagnostic testbed for multi-agent LLM systems. By distributing critical information asymmetrically across agents, the paradigm reveals how inter-agent dynamics support or hinder collective reasoning. We first formalize the paradigm for multi-agent decision-making under distributed knowledge and instantiate it as a benchmark with nine tasks spanning diverse scenarios, including adaptations from prior human studies. We then conduct experiments with GPT-4.1 and five other leading LLMs, including reasoning-enhanced variants, showing that multi-agent systems across all models fail to match the accuracy of single agents given complete information. While agents' collective performance is broadly comparable to that of human groups, nuanced behavioral differences emerge, such as increased sensitivity to social desirability. Finally, we demonstrate the paradigm's diagnostic utility by exploring a cooperation-contradiction trade-off in multi-agent LLM systems. We find that while cooperative agents are prone to over-coordination in collective settings, increased contradiction impairs group convergence. This work contributes a reproducible framework for evaluating multi-agent LLM systems and motivates future research on artificial collective intelligence and human-AI interaction. 

**Abstract (ZH)**: 基于大型语言模型的多Agent系统通过分布式信息整合提升了问题解决能力，但也存在复制人类群体中观察到的集体推理失败的风险。然而，尚不存在基于理论的基准来系统性评估这些失败。本文引入社会心理学中的隐秘特征 paradigm 作为多Agent大型语言模型系统的诊断测试床。通过不对称地分布在不同Agent之间的关键信息，该paradigm揭示了Agent间动态如何支持或阻碍集体推理。我们首先为分布式知识下的多Agent决策制定形式化的paradigm，并以涵盖多种场景的九个任务实例化为基准，包括从先前的人类研究中进行改编。然后，我们使用GPT-4.1和五种其他领先的大型语言模型进行了实验，包括增强推理能力的变体，结果显示所有模型下的多Agent系统在完整信息条件下都无法达到单个代理的准确性。尽管Agent的集体表现与人类群体表现相当，但在行为上出现了一些细微的差异，如对社会认可性的更高敏感性。最后，我们通过探索多Agent大型语言模型系统中的合作-矛盾权衡，展示了该paradigm的诊断效用。我们发现，在集体环境中，合作性Agent容易出现过度协调，而增大的矛盾阻碍了群体的共识形成。本文贡献了一个可重复的框架以评估多Agent大型语言模型系统，并启发未来关于人工集体智能和人机交互的研究。 

---
# GSPRec: Temporal-Aware Graph Spectral Filtering for Recommendation 

**Title (ZH)**: GSPRec：基于时空图频域滤波的推荐方法 

**Authors**: Ahmad Bin Rabiah, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2505.11552)  

**Abstract**: Graph-based recommendation systems are effective at modeling collaborative patterns but often suffer from two limitations: overreliance on low-pass filtering, which suppresses user-specific signals, and omission of sequential dynamics in graph construction. We introduce GSPRec, a graph spectral model that integrates temporal transitions through sequentially-informed graph construction and applies frequency-aware filtering in the spectral domain. GSPRec encodes item transitions via multi-hop diffusion to enable the use of symmetric Laplacians for spectral processing. To capture user preferences, we design a dual-filtering mechanism: a Gaussian bandpass filter to extract mid-frequency, user-level patterns, and a low-pass filter to retain global trends. Extensive experiments on four public datasets show that GSPRec consistently outperforms baselines, with an average improvement of 6.77% in NDCG@10. Ablation studies show the complementary benefits of both sequential graph augmentation and bandpass filtering. 

**Abstract (ZH)**: 基于图的推荐系统在建模协作模式方面非常有效，但往往存在两个局限性：过度依赖低通滤波，这会抑制用户的特定信号，以及在图构建中忽略了序列动态。我们引入了GSPRec，这是一种通过基于序列的图构建和频域中的频率感知滤波来集成时间过渡的图频谱模型。GSPRec 通过多跳扩散编码项目转换，以启用对称拉普拉斯算子的频谱处理。为了捕捉用户偏好，我们设计了一种双滤波机制：高斯带通滤波器用于提取中频用户的模式，低通滤波器用于保留全局趋势。在四个公开数据集上的 extensive 实验表明，GSPRec 在 NDCG@10 上的一致性能优于基线模型，平均改善了 6.77%。消融研究显示了序列图增强和带通滤波两者的互补益处。 

---
# AI-generated Text Detection: A Multifaceted Approach to Binary and Multiclass Classification 

**Title (ZH)**: AI生成文本检测：二分类和多分类的一种综合性方法 

**Authors**: Harika Abburi, Sanmitra Bhattacharya, Edward Bowen, Nirmala Pudota  

**Link**: [PDF](https://arxiv.org/pdf/2505.11550)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in generating text that closely resembles human writing across a wide range of styles and genres. However, such capabilities are prone to potential misuse, such as fake news generation, spam email creation, and misuse in academic assignments. As a result, accurate detection of AI-generated text and identification of the model that generated it are crucial for maintaining the responsible use of LLMs. In this work, we addressed two sub-tasks put forward by the Defactify workshop under AI-Generated Text Detection shared task at the Association for the Advancement of Artificial Intelligence (AAAI 2025): Task A involved distinguishing between human-authored or AI-generated text, while Task B focused on attributing text to its originating language model. For each task, we proposed two neural architectures: an optimized model and a simpler variant. For Task A, the optimized neural architecture achieved fifth place with $F1$ score of 0.994, and for Task B, the simpler neural architecture also ranked fifth place with $F1$ score of 0.627. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在生成风格和体裁各异的人类写作相似文本方面表现出了非凡的能力。然而，这些能力容易被不当使用，如假新闻生成、垃圾邮件创建以及学术作业中的滥用。因此，准确检测人工智能生成的文本并识别生成它的模型对于负责任地使用LLMs至关重要。在2025年美国人工智能协会（AAAI）AI生成文本检测共享任务（Defactify研讨会）下，我们针对两个子任务进行了研究：任务A涉及区分人类作者或AI生成的文本，而任务B则专注于将文本归属性于其起源的语言模型。对于每个任务，我们提出了两种神经架构：优化模型和简单变体。在任务A中，优化的神经架构取得了第五名，F1得分为0.994；在任务B中，简单神经架构也取得了第五名，F1得分为0.627。 

---
# One Shot Dominance: Knowledge Poisoning Attack on Retrieval-Augmented Generation Systems 

**Title (ZH)**: 一次性主导权：检索增强生成系统中的知识投毒攻击 

**Authors**: Zhiyuan Chang, Xiaojun Jia, Mingyang Li, Junjie Wang, Yuekai Huang, Qing Wang, Ziyou Jiang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11548)  

**Abstract**: Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) have shown improved performance in generating accurate responses. However, the dependence on external knowledge bases introduces potential security vulnerabilities, particularly when these knowledge bases are publicly accessible and modifiable. Poisoning attacks on knowledge bases for RAG systems face two fundamental challenges: the injected malicious content must compete with multiple authentic documents retrieved by the retriever, and LLMs tend to trust retrieved information that aligns with their internal memorized knowledge. Previous works attempt to address these challenges by injecting multiple malicious documents, but such saturation attacks are easily detectable and impractical in real-world scenarios. To enable the effective single document poisoning attack, we propose AuthChain, a novel knowledge poisoning attack method that leverages Chain-of-Evidence theory and authority effect to craft more convincing poisoned documents. AuthChain generates poisoned content that establishes strong evidence chains and incorporates authoritative statements, effectively overcoming the interference from both authentic documents and LLMs' internal knowledge. Extensive experiments across six popular LLMs demonstrate that AuthChain achieves significantly higher attack success rates while maintaining superior stealthiness against RAG defense mechanisms compared to state-of-the-art baselines. 

**Abstract (ZH)**: 增强检索增强生成（RAG）的大语言模型（LLMs）在生成准确响应方面表现出改进的性能。然而，对外部知识库的依赖引入了潜在的安全漏洞，特别是当这些知识库公开可访问和可修改时。针对RAG系统的知识库中毒攻击面临着两个根本性的挑战：注入的恶意内容必须与检索器检索到的多个真实文档进行竞争，并且LLMs倾向于信任与其内部记忆知识一致的检索信息。以往的工作试图通过注入多个恶意文档来应对这些挑战，但这种饱和攻击在实际场景中容易被检测并且不切实际。为了实现有效的单文档中毒攻击，我们提出了一种名为AuthChain的新颖知识中毒攻击方法，该方法利用证据链理论和权威效果来制作更具说服力的中毒文档。AuthChain生成的中毒内容建立了强大的证据链，并结合权威声明，有效地克服了真实文档和LLMs内部知识的干扰。跨六种流行的LLMs的广泛实验显示，AuthChain在攻击成功率和对抗RAG防御机制的隐蔽性方面均显著优于最先进的基线方法。 

---
# On Technique Identification and Threat-Actor Attribution using LLMs and Embedding Models 

**Title (ZH)**: 使用大型语言模型和嵌入模型进行技术识别与威胁行为者归属 

**Authors**: Kyla Guru, Robert J. Moss, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11547)  

**Abstract**: Attribution of cyber-attacks remains a complex but critical challenge for cyber defenders. Currently, manual extraction of behavioral indicators from dense forensic documentation causes significant attribution delays, especially following major incidents at the international scale. This research evaluates large language models (LLMs) for cyber-attack attribution based on behavioral indicators extracted from forensic documentation. We test OpenAI's GPT-4 and text-embedding-3-large for identifying threat actors' tactics, techniques, and procedures (TTPs) by comparing LLM-generated TTPs against human-generated data from MITRE ATT&CK Groups. Our framework then identifies TTPs from text using vector embedding search and builds profiles to attribute new attacks for a machine learning model to learn. Key contributions include: (1) assessing off-the-shelf LLMs for TTP extraction and attribution, and (2) developing an end-to-end pipeline from raw CTI documents to threat-actor prediction. This research finds that standard LLMs generate TTP datasets with noise, resulting in a low similarity to human-generated datasets. However, the TTPs generated are similar in frequency to those within the existing MITRE datasets. Additionally, although these TTPs are different than human-generated datasets, our work demonstrates that they still prove useful for training a model that performs above baseline on attribution. Project code and files are contained here: this https URL. 

**Abstract (ZH)**: 基于行为指标的大语言模型在 cyber-攻击归因中的评估：从国际重大事件后的法医文档到威胁行为者预测的端到端管道 

---
# Control Invariant Sets for Neural Network Dynamical Systems and Recursive Feasibility in Model Predictive Control 

**Title (ZH)**: 神经网络动力系统中的控制不变集及模型预测控制的递归可行性 

**Authors**: Xiao Li, Tianhao Wei, Changliu Liu, Anouck Girard, Ilya Kolmanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.11546)  

**Abstract**: Neural networks are powerful tools for data-driven modeling of complex dynamical systems, enhancing predictive capability for control applications. However, their inherent nonlinearity and black-box nature challenge control designs that prioritize rigorous safety and recursive feasibility guarantees. This paper presents algorithmic methods for synthesizing control invariant sets specifically tailored to neural network based dynamical models. These algorithms employ set recursion, ensuring termination after a finite number of iterations and generating subsets in which closed-loop dynamics are forward invariant, thus guaranteeing perpetual operational safety. Additionally, we propose model predictive control designs that integrate these control invariant sets into mixed-integer optimization, with guaranteed adherence to safety constraints and recursive feasibility at the computational level. We also present a comprehensive theoretical analysis examining the properties and guarantees of the proposed methods. Numerical simulations in an autonomous driving scenario demonstrate the methods' effectiveness in synthesizing control-invariant sets offline and implementing model predictive control online, ensuring safety and recursive feasibility. 

**Abstract (ZH)**: 基于神经网络的复杂动态系统数据驱动建模中的控制不变集合成算法：确保递归可行性和安全性 

---
# TARGET: Benchmarking Table Retrieval for Generative Tasks 

**Title (ZH)**: TARGET: 生成任务中表格检索的基准测试 

**Authors**: Xingyu Ji, Parker Glenn, Aditya G. Parameswaran, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2505.11545)  

**Abstract**: The data landscape is rich with structured data, often of high value to organizations, driving important applications in data analysis and machine learning. Recent progress in representation learning and generative models for such data has led to the development of natural language interfaces to structured data, including those leveraging text-to-SQL. Contextualizing interactions, either through conversational interfaces or agentic components, in structured data through retrieval-augmented generation can provide substantial benefits in the form of freshness, accuracy, and comprehensiveness of answers. The key question is: how do we retrieve the right table(s) for the analytical query or task at hand? To this end, we introduce TARGET: a benchmark for evaluating TAble Retrieval for GEnerative Tasks. With TARGET we analyze the retrieval performance of different retrievers in isolation, as well as their impact on downstream tasks. We find that dense embedding-based retrievers far outperform a BM25 baseline which is less effective than it is for retrieval over unstructured text. We also surface the sensitivity of retrievers across various metadata (e.g., missing table titles), and demonstrate a stark variation of retrieval performance across datasets and tasks. TARGET is available at this https URL. 

**Abstract (ZH)**: 数据景观中富含结构化数据，这些数据对组织具有重要价值，驱动着数据分析和机器学习中的关键应用。近年来，针对此类数据的表示学习和生成模型的进步促进了结构化数据自然语言接口的发展，包括利用文本到SQL的方法。通过检索增强生成来将语境化应用于结构化数据的交互，可以显著提高答案的新鲜度、准确性和完整性。关键问题在于：我们如何检索与手头的分析查询或任务相匹配的正确表？为此，我们引入了TARGET：用于生成任务的表检索基准。通过TARGET，我们分析了不同检索器的检索性能及其对下游任务的影响。我们发现，基于密集嵌入的检索器远超BM25基线，而后者在非结构化文本检索中更为有效。我们还探讨了检索器在各种元数据（例如，缺失的表标题）方面的敏感性，并展示了不同数据集和任务之间检索性能的巨大差异。TARGET可在以下链接获取：this https URL。 

---
# LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation 

**Title (ZH)**: 基于潜扩散的世界模型：LaDi-WM 用于预测性操控 

**Authors**: Yuhang Huang, JIazhao Zhang, Shilong Zou, XInwang Liu, Ruizhen Hu, Kai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11528)  

**Abstract**: Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments. 

**Abstract (ZH)**: 基于扩散模型的LaDi-WM世界模型及其在预测操控中的应用 

---
# Code Retrieval for MILP Instance Generation 

**Title (ZH)**: 基于代码检索的MILP实例生成 

**Authors**: Tianxing Yang, Huigen Ye, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11526)  

**Abstract**: Mixed-Integer Linear Programming (MILP) is widely used in fields such as scheduling, logistics, and planning. Enhancing the performance of MILP solvers, particularly learning-based solvers, requires substantial amounts of high-quality data. However, existing methods for MILP instance generation typically necessitate training a separate model for each problem class and are computationally intensive when generating new instances. To address these limitations, we reformulate the MILP Instance Generation task as MILP Code Generation task, enabling efficient, flexible, and interpretable instance generation through code. Since MILP instances generated from code can vary significantly in scale, we introduce MILP-EmbedSim, a new similarity metric that accurately measures the similarity between instances of varying sizes within the same problem class. Leveraging this metric, we propose MILP-Retrieval, a pipeline that retrieves generation code from library to produce MILP instances highly similar to target instance. MILP-Retrieval outperforms baselines in both MILP Code Generation and Instance Generation tasks, provides a novel perspective on MILP instance generation and opens new possibilities for learning-based solvers. 

**Abstract (ZH)**: 混合整数线性规划(MILP)在调度、物流和规划等领域广泛应用。提高MILP求解器，尤其是基于学习的求解器的性能，需要大量高质量的数据。然而，现有的MILP实例生成方法通常需要为每个问题类别训练一个单独的模型，并在生成新实例时计算强度高。为解决这些问题，我们将MILP实例生成任务重新表述为MILP代码生成任务，从而通过代码实现高效、灵活和可解释的实例生成。由于从代码生成的MILP实例在规模上可能存在显著差异，我们引入了MILP-EmbedSim，这是一种新的相似度度量方法，可准确衡量同一问题类别中不同规模实例之间的相似度。利用该度量，我们提出了MILP-Retrieval管道，从库中检索生成代码以产生与目标实例高度相似的MILP实例。MILP-Retrieval在MILP代码生成和实例生成任务中均优于基线方法，为MILP实例生成提供了新的视角，并为基于学习的求解器开辟了新的可能性。 

---
# Decentralized Traffic Flow Optimization Through Intrinsic Motivation 

**Title (ZH)**: 通过固有动机实现的分布式交通流优化 

**Authors**: Himaja Papala, Daniel Polani, Stas Tiomkin  

**Link**: [PDF](https://arxiv.org/pdf/2505.11520)  

**Abstract**: Traffic congestion has long been an ubiquitous problem that is exacerbating with the rapid growth of megacities. In this proof-of-concept work we study intrinsic motivation, implemented via the empowerment principle, to control autonomous car behavior to improve traffic flow. In standard models of traffic dynamics, self-organized traffic jams emerge spontaneously from the individual behavior of cars, affecting traffic over long distances. Our novel car behavior strategy improves traffic flow while still being decentralized and using only locally available information without explicit coordination. Decentralization is essential for various reasons, not least to be able to absorb robustly substantial levels of uncertainty. Our scenario is based on the well-established traffic dynamics model, the Nagel-Schreckenberg cellular automaton. In a fraction of the cars in this model, we substitute the default behavior by empowerment, our intrinsic motivation-based method. This proposed model significantly improves overall traffic flow, mitigates congestion, and reduces the average traffic jam time. 

**Abstract (ZH)**: 交通拥堵一直是日益严重的普遍问题，特别是在 megacities 快速增长的情况下。在本概念验证工作中，我们研究了通过赋能原则实现的内在动机，以控制自动驾驶汽车的行为，从而改善交通流。在标准的交通动力学模型中，自我组织的交通堵塞会自发地从车辆个体行为中涌现出来，影响远距离的交通。我们的新型汽车行为策略在不集中控制和仅使用局部可用信息的情况下改善了交通流，而无需明确的协调。去中心化对于多种原因至关重要，尤其是能够稳健地吸收大量的不确定性。我们的场景基于著名的交通动力学模型——Nagel-Schreckenberg 格子自动机。在这种模型中，我们通过赋能，即基于内在动机的方法，替代部分汽车的默认行为。该提出的模型显著改善了整体交通流，减轻了拥堵，并减少了平均交通堵塞时间。 

---
# Knowledge-enhanced Multi-perspective Video Representation Learning for Scene Recognition 

**Title (ZH)**: 基于知识增强的多视角视频表示学习的场景识别 

**Authors**: Xuzheng Yu, Chen Jiang, Wei Zhang, Tian Gan, Linlin Chao, Jianan Zhao, Yuan Cheng, Qingpei Guo, Wei Chu  

**Link**: [PDF](https://arxiv.org/pdf/2401.04354)  

**Abstract**: With the explosive growth of video data in real-world applications, a comprehensive representation of videos becomes increasingly important. In this paper, we address the problem of video scene recognition, whose goal is to learn a high-level video representation to classify scenes in videos. Due to the diversity and complexity of video contents in realistic scenarios, this task remains a challenge. Most existing works identify scenes for videos only from visual or textual information in a temporal perspective, ignoring the valuable information hidden in single frames, while several earlier studies only recognize scenes for separate images in a non-temporal perspective. We argue that these two perspectives are both meaningful for this task and complementary to each other, meanwhile, externally introduced knowledge can also promote the comprehension of videos. We propose a novel two-stream framework to model video representations from multiple perspectives, i.e. temporal and non-temporal perspectives, and integrate the two perspectives in an end-to-end manner by self-distillation. Besides, we design a knowledge-enhanced feature fusion and label prediction method that contributes to naturally introducing knowledge into the task of video scene recognition. Experiments conducted on a real-world dataset demonstrate the effectiveness of our proposed method. 

**Abstract (ZH)**: 随着实际应用中视频数据的爆炸性增长，对视频进行全面的表示变得越来越重要。本文针对视频场景识别问题，其目标是学习高层视频表示以对视频中的场景进行分类。由于在真实场景中视频内容的多样性和复杂性，这项任务依然具有挑战性。现有大多数工作仅从时间维度的角度识别视频中的场景，忽视了单帧中隐藏的有价值信息，而早期一些研究仅在非时间维度的角度识别单独图像中的场景。我们认为，这两种视角对于这项任务都是有意义的，并且是互补的，同时外部引入的知识也可以促进对视频的理解。我们提出了一种新的双流框架，从时间维度和非时间维度建模视频表示，并通过自精炼在端到端的方式将两种视角进行整合。此外，我们设计了一种增强知识特征融合和标签预测方法，有助于自然地将知识引入视频场景识别任务中。在真实世界数据集上的实验验证了我们所提方法的有效性。 

---
# Learning Segment Similarity and Alignment in Large-Scale Content Based Video Retrieval 

**Title (ZH)**: 大规模内容基础视频检索中的片段相似性学习与对齐 

**Authors**: Chen Jiang, Kaiming Huang, Sifeng He, Xudong Yang, Wei Zhang, Xiaobo Zhang, Yuan Cheng, Lei Yang, Qing Wang, Furong Xu, Tan Pan, Wei Chu  

**Link**: [PDF](https://arxiv.org/pdf/2309.11091)  

**Abstract**: With the explosive growth of web videos in recent years, large-scale Content-Based Video Retrieval (CBVR) becomes increasingly essential in video filtering, recommendation, and copyright protection. Segment-level CBVR (S-CBVR) locates the start and end time of similar segments in finer granularity, which is beneficial for user browsing efficiency and infringement detection especially in long video scenarios. The challenge of S-CBVR task is how to achieve high temporal alignment accuracy with efficient computation and low storage consumption. In this paper, we propose a Segment Similarity and Alignment Network (SSAN) in dealing with the challenge which is firstly trained end-to-end in S-CBVR. SSAN is based on two newly proposed modules in video retrieval: (1) An efficient Self-supervised Keyframe Extraction (SKE) module to reduce redundant frame features, (2) A robust Similarity Pattern Detection (SPD) module for temporal alignment. In comparison with uniform frame extraction, SKE not only saves feature storage and search time, but also introduces comparable accuracy and limited extra computation time. In terms of temporal alignment, SPD localizes similar segments with higher accuracy and efficiency than existing deep learning methods. Furthermore, we jointly train SSAN with SKE and SPD and achieve an end-to-end improvement. Meanwhile, the two key modules SKE and SPD can also be effectively inserted into other video retrieval pipelines and gain considerable performance improvements. Experimental results on public datasets show that SSAN can obtain higher alignment accuracy while saving storage and online query computational cost compared to existing methods. 

**Abstract (ZH)**: 基于片段的视频检索相似性与对齐网络：高效计算与低存储消耗的时间对齐准确率提升 

---
