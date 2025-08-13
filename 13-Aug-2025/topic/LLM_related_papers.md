# CRADLE: Conversational RTL Design Space Exploration with LLM-based Multi-Agent Systems 

**Title (ZH)**: CRriadge: 基于LLM的多智能体系统在对话式RTL设计空间探索中的应用 kuk 

**Authors**: Lukas Krupp, Maximilian Schöffel, Elias Biehl, Norbert Wehn  

**Link**: [PDF](https://arxiv.org/pdf/2508.08709)  

**Abstract**: This paper presents CRADLE, a conversational framework for design space exploration of RTL designs using LLM-based multi-agent systems. Unlike existing rigid approaches, CRADLE enables user-guided flows with internal self-verification, correction, and optimization. We demonstrate the framework with a generator-critic agent system targeting FPGA resource minimization using state-of-the-art LLMs. Experimental results on the RTLLM benchmark show that CRADLE achieves significant reductions in resource usage with averages of 48% and 40% in LUTs and FFs across all benchmark designs. 

**Abstract (ZH)**: CRADLE：基于LLM的多Agent系统在RTL设计空间探索中的对话框架 

---
# MinionsLLM: a Task-adaptive Framework For The Training and Control of Multi-Agent Systems Through Natural Language 

**Title (ZH)**: MinionsLLM：一种通过自然语言进行多-Agent系统训练和控制的任务自适应框架 

**Authors**: Andres Garcia Rincon, Eliseo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2508.08283)  

**Abstract**: This paper presents MinionsLLM, a novel framework that integrates Large Language Models (LLMs) with Behavior Trees (BTs) and Formal Grammars to enable natural language control of multi-agent systems within arbitrary, user-defined environments. MinionsLLM provides standardized interfaces for defining environments, agents, and behavioral primitives, and introduces two synthetic dataset generation methods (Method A and Method B) to fine-tune LLMs for improved syntactic validity and semantic task relevance. We validate our approach using Google's Gemma 3 model family at three parameter scales (1B, 4B, and 12B) and demonstrate substantial gains: Method B increases syntactic validity to 92.6% and achieves a mean task performance improvement of 33% over baseline. Notably, our experiments show that smaller models benefit most from fine-tuning, suggesting promising directions for deploying compact, locally hosted LLMs in resource-constrained multi-agent control scenarios. The framework and all resources are released open-source to support reproducibility and future research. 

**Abstract (ZH)**: MinionsLLM：一种将大型语言模型与行为树及形式语法结合的新框架，用于任意用户定义环境内的多智能体系统自然语言控制 

---
# BrowseMaster: Towards Scalable Web Browsing via Tool-Augmented Programmatic Agent Pair 

**Title (ZH)**: BrowseMaster: 通过工具增强的程序化代理对实现面向大规模Web浏览 

**Authors**: Xianghe Pang, Shuo Tang, Rui Ye, Yuwen Du, Yaxin Du, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.09129)  

**Abstract**: Effective information seeking in the vast and ever-growing digital landscape requires balancing expansive search with strategic reasoning. Current large language model (LLM)-based agents struggle to achieve this balance due to limitations in search breadth and reasoning depth, where slow, serial querying restricts coverage of relevant sources and noisy raw inputs disrupt the continuity of multi-step reasoning. To address these challenges, we propose BrowseMaster, a scalable framework built around a programmatically augmented planner-executor agent pair. The planner formulates and adapts search strategies based on task constraints, while the executor conducts efficient, targeted retrieval to supply the planner with concise, relevant evidence. This division of labor preserves coherent, long-horizon reasoning while sustaining broad and systematic exploration, overcoming the trade-off that limits existing agents. Extensive experiments on challenging English and Chinese benchmarks show that BrowseMaster consistently outperforms open-source and proprietary baselines, achieving scores of 30.0 on BrowseComp-en and 46.5 on BrowseComp-zh, which demonstrates its strong capability in complex, reasoning-heavy information-seeking tasks at scale. 

**Abstract (ZH)**: 有效的信息查找需要在广泛搜索与战略推理之间取得平衡。现有的基于大规模语言模型的代理由于搜索广度和推理深度的限制，在实现这一平衡方面存在困难，其中缓慢的串行查询限制了相关来源的覆盖范围，嘈杂的原始输入打断了多步推理的连续性。为了解决这些挑战，我们提出了BrowseMaster，这是一种基于编程增强的规划执行者代理对构建的可扩展框架。规划者根据任务约束制定和调整搜索策略，而执行者进行高效的目标检索，为规划者提供简洁的相关证据。这种分工合作保持了连贯的长期推理能力，并且能够进行广泛的系统性探索，克服了现有代理面临的权衡。在具有挑战性的英文和中文基准上的 extensive 实验表明，BrowseMaster 一致地优于开源和专有的基线，分别在 BrowseComp-en 和 BrowseComp-zh 上取得了 30.0 和 46.5 的分数，这显示出其在大规模复杂推理密集型信息查找任务中的强大能力。 

---
# SMA: Who Said That? Auditing Membership Leakage in Semi-Black-box RAG Controlling 

**Title (ZH)**: SMA: 谁说的？半黑箱RAG中的成员泄漏审计 

**Authors**: Shixuan Sun, Siyuan Liang, Ruoyu Chen, Jianjie Huang, Jingzhi Li, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09105)  

**Abstract**: Retrieval-Augmented Generation (RAG) and its Multimodal Retrieval-Augmented Generation (MRAG) significantly improve the knowledge coverage and contextual understanding of Large Language Models (LLMs) by introducing external knowledge sources. However, retrieval and multimodal fusion obscure content provenance, rendering existing membership inference methods unable to reliably attribute generated outputs to pre-training, external retrieval, or user input, thus undermining privacy leakage accountability
To address these challenges, we propose the first Source-aware Membership Audit (SMA) that enables fine-grained source attribution of generated content in a semi-black-box setting with retrieval control this http URL address the environmental constraints of semi-black-box auditing, we further design an attribution estimation mechanism based on zero-order optimization, which robustly approximates the true influence of input tokens on the output through large-scale perturbation sampling and ridge regression modeling. In addition, SMA introduces a cross-modal attribution technique that projects image inputs into textual descriptions via MLLMs, enabling token-level attribution in the text modality, which for the first time facilitates membership inference on image retrieval traces in MRAG systems. This work shifts the focus of membership inference from 'whether the data has been memorized' to 'where the content is sourced from', offering a novel perspective for auditing data provenance in complex generative systems. 

**Abstract (ZH)**: 检索增强生成（RAG）及其多模态检索增强生成（MRAG）通过引入外部知识源显著提高了大型语言模型（LLMs）的知识覆盖范围和上下文理解能力。然而，检索和多模态融合模糊了内容的来源，使得现有的成员身份推断方法无法可靠地将生成输出归因于预训练、外部检索或用户输入，从而削弱了隐私泄露问责制。
为此，我们提出了首个基于源感知的成员身份审计（SMA），能够在检索控制的半黑盒环境下细粒度地归因生成内容。为应对半黑盒审计的环境约束，我们进一步设计了一种基于零阶优化的归因估计机制，通过大规模扰动采样和岭回归建模，稳健地逼近输入标记对输出的真实影响。此外，SMA 引入了一种跨模态归因技术，通过 MLLMs 将图像输入投影到文本描述中，实现了文本模态中的标记级归因，首次在 MRAG 系统中促进了图像检索痕迹上的成员身份推断。这项工作将成员身份推断的焦点从“数据是否被记忆”转移到“内容的来源在哪里”，为复杂生成系统中的数据来源审计提供了新的视角。 

---
# Activation Steering for Bias Mitigation: An Interpretable Approach to Safer LLMs 

**Title (ZH)**: 激活调整以减轻偏差：一种可解释的方法以实现更安全的大型语言模型 

**Authors**: Shivam Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2508.09019)  

**Abstract**: As large language models (LLMs) become more integrated into societal systems, the risk of them perpetuating and amplifying harmful biases becomes a critical safety concern. Traditional methods for mitigating bias often rely on data filtering or post-hoc output moderation, which treat the model as an opaque black box. In this work, we introduce a complete, end-to-end system that uses techniques from mechanistic interpretability to both identify and actively mitigate bias directly within a model's internal workings. Our method involves two primary stages. First, we train linear "probes" on the internal activations of a model to detect the latent representations of various biases (e.g., gender, race, age). Our experiments on \texttt{gpt2-large} demonstrate that these probes can identify biased content with near-perfect accuracy, revealing that bias representations become most salient in the model's later layers. Second, we leverage these findings to compute "steering vectors" by contrasting the model's activation patterns for biased and neutral statements. By adding these vectors during inference, we can actively steer the model's generative process away from producing harmful, stereotypical, or biased content in real-time. We demonstrate the efficacy of this activation steering technique, showing that it successfully alters biased completions toward more neutral alternatives. We present our work as a robust and reproducible system that offers a more direct and interpretable approach to building safer and more accountable LLMs. 

**Abstract (ZH)**: 随着大型语言模型（LL
userparalleled大型语言模型(LLMs)在社会系统中的作用愈发重要，生成涉及此论文的的中文翻译，要符合学术规范。答案


随着大型语言模型（LLMs）在社会系统中的作用愈发重要JV，它们持续传播和

uem实习生

随着大型语言模型（LLMs）在社会系统中的所
user的影响力愈发显著，它们传播和
随着大型语言模型（LLMs）在社会系统中的影响力愈发显著-ves，它们传播有害偏见的风险也变得愈加严重 withd，传统方法针对偏见的 mitigation主要依赖于后post处理和输出审核，而在这种机制存在一定程度的不透明性，在此基础上，我们提出一种端到端的系统系统，该系统系统结合机器可解读性技术从内部激活出发和的层面出发检测和主动削弱偏见直接作用模型内部运作." 我们将此分为为两个主要阶段. 首先 "探针" 到达模型内部激活以挖掘隐含的偏见（例如性别、种族、年龄等针对这些探针可以输出出带偏见的内容几乎是完美准确的ewith这表明偏见在模型的层中十分突出. �重要原因在于通过激活模式对比带偏见和中立语句可以计算出" 活动向矢量" 通过这些矢量进行推理可以真实地引导模型的输出远离探索带有偏见和刻板印象的有害内容.这种方法在实际中有效地改变了带偏见的内容向着更加中立的形式转变. 我们提出的这种端到端的实验方法提供了一个更为直接和可可的解决方案以构建更安全 and更具责任感的大规模语言模型（LLMs）。 

---
# Intrinsic Memory Agents: Heterogeneous Multi-Agent LLM Systems through Structured Contextual Memory 

**Title (ZH)**: 内在记忆代理：通过结构化上下文记忆实现的异质多Agent大型语言模型系统 

**Authors**: Sizhe Yuen, Francisco Gomez Medina, Ting Su, Yali Du, Adam J. Sobey  

**Link**: [PDF](https://arxiv.org/pdf/2508.08997)  

**Abstract**: Multi-agent systems built on Large Language Models (LLMs) show exceptional promise for complex collaborative problem-solving, yet they face fundamental challenges stemming from context window limitations that impair memory consistency, role adherence, and procedural integrity. This paper introduces Intrinsic Memory Agents, a novel framework that addresses these limitations through structured agent-specific memories that evolve intrinsically with agent outputs. Specifically, our method maintains role-aligned memory templates that preserve specialized perspectives while focusing on task-relevant information. We benchmark our approach on the PDDL dataset, comparing its performance to existing state-of-the-art multi-agentic memory approaches and showing an improvement of 38.6\% with the highest token efficiency. An additional evaluation is performed on a complex data pipeline design task, we demonstrate that our approach produces higher quality designs when comparing 5 metrics: scalability, reliability, usability, cost-effectiveness and documentation with additional qualitative evidence of the improvements. Our findings suggest that addressing memory limitations through structured, intrinsic approaches can improve the capabilities of multi-agent LLM systems on structured planning tasks. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统在复杂协作问题解决方面展现出极大潜力，但它们面临着源自上下文窗口限制的基本挑战，这些限制影响了记忆一致性、角色遵从性和程序完整性。本文介绍了一种新的内在记忆智能体框架，该框架通过与智能体输出内在演化的结构化智能体特定记忆来解决这些限制。具体而言，我们的方法维护了角色对齐的记忆模板，这些模板保留了专门的视角并专注于与任务相关的信息。我们在PDDL数据集上对我们的方法进行了基准测试，将其性能与现有的先进多智能体记忆方法进行了比较，并显示了38.6%的改进，且具有最高的标记效率。此外，我们在一个复杂的数据管道设计任务上进行了额外评估，我们的方法在5个指标：可扩展性、可靠性、易用性、成本效益和文档化方面，都产生了更高质量的设计，并附有额外的定性证据显示改进效果。研究结果表明，通过结构化的内在方法解决记忆限制可以提升多智能体大型语言模型系统在结构化规划任务中的能力。 

---
# Prospect Theory Fails for LLMs: Revealing Instability of Decision-Making under Epistemic Uncertainty 

**Title (ZH)**: prospect理论在LLM中失效：揭示在epistemic不确定性下的决策稳定性问题 

**Authors**: Rui Wang, Qihan Lin, Jiayu Liu, Qing Zong, Tianshi Zheng, Weiqi Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.08992)  

**Abstract**: Prospect Theory (PT) models human decision-making under uncertainty, while epistemic markers (e.g., maybe) serve to express uncertainty in language. However, it remains largely unexplored whether Prospect Theory applies to contemporary Large Language Models and whether epistemic markers, which express human uncertainty, affect their decision-making behaviour. To address these research gaps, we design a three-stage experiment based on economic questionnaires. We propose a more general and precise evaluation framework to model LLMs' decision-making behaviour under PT, introducing uncertainty through the empirical probability values associated with commonly used epistemic markers in comparable contexts. We then incorporate epistemic markers into the evaluation framework based on their corresponding probability values to examine their influence on LLM decision-making behaviours. Our findings suggest that modelling LLMs' decision-making with PT is not consistently reliable, particularly when uncertainty is expressed in diverse linguistic forms. Our code is released in this https URL. 

**Abstract (ZH)**: 期望理论（PT）模型人在不确定条件下的决策，而知识性标记（例如，maybe）则用于在语言中表达不确定性。然而，PT是否适用于现代大型语言模型以及知识性标记是否影响它们的决策行为尚未得到充分探索。为填补这些研究空白，我们设计了一个基于经济问卷的三阶段实验。我们提出了一种更为通用和精确的评估框架，以PT建模LLM的决策行为，通过引入与常见知识性标记相关联的经验概率值来表达不确定性。随后，我们基于这些标记对应的概率值将知识性标记纳入评估框架，以考察其对LLM决策行为的影响。我们的研究发现，使用PT建模LLM的决策行为并不总是可靠的，特别是在不同的语言形式表达不确定性时。我们的代码发布在<该链接>。 

---
# Compass-Thinker-7B Technical Report 

**Title (ZH)**: Compass-Thinker-7B 技术报告 

**Authors**: Anxiang Zeng, Haibo Zhang, Kaixiang Mo, Long Zhang, Shuman Liu, Yanhui Huang, Yawen Liu, Yuepeng Sheng, Yuwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08909)  

**Abstract**: Recent R1-Zero-like research further demonstrates that reasoning extension has given large language models (LLMs) unprecedented reasoning capabilities, and Reinforcement Learning is the core tech- nology to elicit its complex reasoning. However, conducting RL experiments directly on hyperscale models involves high computational costs and resource demands, posing significant risks. We pro- pose the Compass-Thinker-7B model, which aims to explore the potential of Reinforcement Learn- ing with less computational resources and costs, and provides insights for further research into RL recipes for larger models. Compass-Thinker-7B is trained from an open source model through a spe- cially designed Reinforcement Learning Pipeline. we curate a dataset of 30k verifiable mathematics problems for the Reinforcement Learning Pipeline. By configuring data and training settings with dif- ferent difficulty distributions for different stages, the potential of the model is gradually released and the training efficiency is improved. Extensive evaluations show that Compass-Thinker-7B possesses exceptional reasoning potential, and achieves superior performance on mathematics compared to the same-sized RL this http URL in the challenging AIME2024 evaluation, Compass-Thinker-7B achieves 40% accuracy. 

**Abstract (ZH)**: Recent R1-Zero-like研究进一步表明，推理拓展赋予了大规模语言模型（LLMs）前所未有的推理能力，强化学习是激发其复杂推理的核心技术。然而，直接在超大规模模型上进行RL实验涉及高昂的计算成本和资源需求，存在重大风险。我们提出了Compass-Thinker-7B模型，旨在以较少的计算资源和成本探索强化学习的潜力，并为更大模型的RL配方提供新的见解。Compass-Thinker-7B通过专门设计的强化学习管道从开源模型训练而来。我们收集了一个包含30,000个可验证数学问题的数据集用于强化学习管道。通过在不同阶段配置不同的难度分布的数据和训练设置，模型的潜力逐步释放，训练效率得到提高。广泛评估显示，Compass-Thinker-7B具有出色的推理潜力，并在数学领域取得了优于同规模RL模型的表现，在具有挑战性的AIME2024评估中，Compass-Thinker-7B实现了40%的准确率。 

---
# A Dual-Axis Taxonomy of Knowledge Editing for LLMs: From Mechanisms to Functions 

**Title (ZH)**: LLMs的知识编辑双轴分类：从机制到功能 

**Authors**: Amir Mohammad Salehoof, Ali Ramezani, Yadollah Yaghoobzadeh, Majid Nili Ahmadabadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08795)  

**Abstract**: Large language models (LLMs) acquire vast knowledge from large text corpora, but this information can become outdated or inaccurate. Since retraining is computationally expensive, knowledge editing offers an efficient alternative -- modifying internal knowledge without full retraining. These methods aim to update facts precisely while preserving the model's overall capabilities. While existing surveys focus on the mechanism of editing (e.g., parameter changes vs. external memory), they often overlook the function of the knowledge being edited. This survey introduces a novel, complementary function-based taxonomy to provide a more holistic view. We examine how different mechanisms apply to various knowledge types -- factual, temporal, conceptual, commonsense, and social -- highlighting how editing effectiveness depends on the nature of the target knowledge. By organizing our review along these two axes, we map the current landscape, outline the strengths and limitations of existing methods, define the problem formally, survey evaluation tasks and datasets, and conclude with open challenges and future directions. 

**Abstract (ZH)**: 大型语言模型（LLMs）从大量文本语料中获取广泛知识，但这些信息可能变得过时或不准确。由于重新训练计算成本高昂，知识编辑提供了高效的替代方案——无需完全重新训练即可修改内部知识。这些方法旨在精准更新事实并保留模型的整体能力。尽管现有综述侧重于编辑机制（如参数变化与外部记忆），但往往会忽略被编辑知识的功能。本文综述引入了一种新型、补充性的功能型分类法，以提供更全面的视角。我们探讨了不同机制如何应用于各种知识类型——事实性、时间性、概念性、常识性和社会性——强调编辑效果取决于目标知识的本质。通过沿这两个维度组织我们的评审，我们绘制了当前的状况，概述了现有方法的优势和局限性，正式定义了问题，回顾了评估任务和数据集，并以开放挑战和未来方向结束。 

---
# Aryabhata: An exam-focused language model for JEE Math 

**Title (ZH)**: Aryabhata: 针对JEE数学的考试导向语言模型 

**Authors**: Ritvik Rastogi, Sachin Dharashivkar, Sandeep Varma  

**Link**: [PDF](https://arxiv.org/pdf/2508.08665)  

**Abstract**: We present $\textbf{Aryabhata 1.0}$, a compact 7B parameter math reasoning model optimized for the Indian academic exam, the Joint Entrance Examination (JEE). Despite rapid progress in large language models (LLMs), current models often remain unsuitable for educational use. Aryabhata 1.0 is built by merging strong open-weight reasoning models, followed by supervised fine-tuning (SFT) with curriculum learning on verified chain-of-thought (CoT) traces curated through best-of-$n$ rejection sampling. To further boost performance, we apply reinforcement learning with verifiable rewards (RLVR) using A2C objective with group-relative advantage estimation alongwith novel exploration strategies such as $\textit{Adaptive Group Resizing}$ and $\textit{Temperature Scaling}$. Evaluated on both in-distribution (JEE Main 2025) and out-of-distribution (MATH, GSM8K) benchmarks, Aryabhata outperforms existing models in accuracy and efficiency, while offering pedagogically useful step-by-step reasoning. We release Aryabhata as a foundation model to advance exam-centric, open-source small language models. This marks our first open release for community feedback ($\href{this https URL}{Aryabhata\ 1.0\ on\ Hugging\ Face}$); PW is actively training future models to further improve learning outcomes for students. 

**Abstract (ZH)**: Aryabhata 1.0：为印度学术考试联合入学考试（JEE）优化的紧凑型7B参数数学推理模型 

---
# Prompt-and-Check: Using Large Language Models to Evaluate Communication Protocol Compliance in Simulation-Based Training 

**Title (ZH)**: 提示与检查：使用大型语言模型在基于仿真培训中评估通信协议合规性 

**Authors**: Vishakha Lall, Yisi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08652)  

**Abstract**: Accurate evaluation of procedural communication compliance is essential in simulation-based training, particularly in safety-critical domains where adherence to compliance checklists reflects operational competence. This paper explores a lightweight, deployable approach using prompt-based inference with open-source large language models (LLMs) that can run efficiently on consumer-grade GPUs. We present Prompt-and-Check, a method that uses context-rich prompts to evaluate whether each checklist item in a protocol has been fulfilled, solely based on transcribed verbal exchanges. We perform a case study in the maritime domain with participants performing an identical simulation task, and experiment with models such as LLama 2 7B, LLaMA 3 8B and Mistral 7B, running locally on an RTX 4070 GPU. For each checklist item, a prompt incorporating relevant transcript excerpts is fed into the model, which outputs a compliance judgment. We assess model outputs against expert-annotated ground truth using classification accuracy and agreement scores. Our findings demonstrate that prompting enables effective context-aware reasoning without task-specific training. This study highlights the practical utility of LLMs in augmenting debriefing, performance feedback, and automated assessment in training environments. 

**Abstract (ZH)**: 基于提示的推理在开源大规模语言模型上的轻量级部署方法及其实证研究：在基于仿真的培训中准确评估程序沟通合规性 

---
# AgriGPT: a Large Language Model Ecosystem for Agriculture 

**Title (ZH)**: AgagGPT: 农业大型语言模型生态系统 

**Authors**: Bo Yang, Yu Zhang, Lanfei Feng, Yunkui Chen, Jianyu Zhang, Xiao Xu, Nueraili Aierken, Yurui Li, Yuxuan Chen, Guijun Yang, Yong He, Runhe Huang, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08632)  

**Abstract**: Despite the rapid progress of Large Language Models (LLMs), their application in agriculture remains limited due to the lack of domain-specific models, curated datasets, and robust evaluation frameworks. To address these challenges, we propose AgriGPT, a domain-specialized LLM ecosystem for agricultural usage. At its core, we design a multi-agent scalable data engine that systematically compiles credible data sources into Agri-342K, a high-quality, standardized question-answer (QA) dataset. Trained on this dataset, AgriGPT supports a broad range of agricultural stakeholders, from practitioners to policy-makers. To enhance factual grounding, we employ Tri-RAG, a three-channel Retrieval-Augmented Generation framework combining dense retrieval, sparse retrieval, and multi-hop knowledge graph reasoning, thereby improving the LLM's reasoning reliability. For comprehensive evaluation, we introduce AgriBench-13K, a benchmark suite comprising 13 tasks with varying types and complexities. Experiments demonstrate that AgriGPT significantly outperforms general-purpose LLMs on both domain adaptation and reasoning. Beyond the model itself, AgriGPT represents a modular and extensible LLM ecosystem for agriculture, comprising structured data construction, retrieval-enhanced generation, and domain-specific evaluation. This work provides a generalizable framework for developing scientific and industry-specialized LLMs. All models, datasets, and code will be released to empower agricultural communities, especially in underserved regions, and to promote open, impactful research. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）取得了快速进展，但由于缺乏专门领域的模型、精心筛选的数据集和 robust 的评价框架，其在农业中的应用仍然有限。为解决这些挑战，我们提出 AgriGPT，一个专为农业使用设计的 LLM 生态系统。其核心是一个多代理可扩展的数据引擎，系统地将可信数据源整合为 Agri-342K，这是一个高质量、标准化的问题-答案（QA）数据集。通过在该数据集上训练，AgriGPT 支持广泛的农业科技 stakeholders，从实践者到政策制定者。为了增强事实基础，我们采用了 Tri-RAG，这是一种结合密集检索、稀疏检索和多跳知识图谱推理的三通道检索增强生成框架，从而提高 LL 查推理可靠性。为了进行全面评估，我们引入了 AgriBench-13K，这是一个包含 13 个具有不同类型和复杂性的任务的基准套件。实验结果显示，AgriGPT 在领域适应和推理方面显著优于通用语言模型。除了模型本身，AgriGPT 代表了一个模块化和可扩展的农业 LLM 生态系统，包括结构化数据构建、检索增强生成和特定领域的评价。这项工作提供了一个可通用的框架，用于开发科学和行业专用的 LLM。所有模型、数据集和代码都将公开发布，以赋能农业社区，特别是欠发达地区，并促进开放和有影响力的科学研究。 

---
# SynLLM: A Comparative Analysis of Large Language Models for Medical Tabular Synthetic Data Generation via Prompt Engineering 

**Title (ZH)**: SynLLM：大型语言模型生成医学表格合成数据的促控技术比较分析 

**Authors**: Arshia Ilaty, Hossein Shirazi, Hajar Homayouni  

**Link**: [PDF](https://arxiv.org/pdf/2508.08529)  

**Abstract**: Access to real-world medical data is often restricted due to privacy regulations, posing a significant barrier to the advancement of healthcare research. Synthetic data offers a promising alternative; however, generating realistic, clinically valid, and privacy-conscious records remains a major challenge. Recent advancements in Large Language Models (LLMs) offer new opportunities for structured data generation; however, existing approaches frequently lack systematic prompting strategies and comprehensive, multi-dimensional evaluation frameworks.
In this paper, we present SynLLM, a modular framework for generating high-quality synthetic medical tabular data using 20 state-of-the-art open-source LLMs, including LLaMA, Mistral, and GPT variants, guided by structured prompts. We propose four distinct prompt types, ranging from example-driven to rule-based constraints, that encode schema, metadata, and domain knowledge to control generation without model fine-tuning. Our framework features a comprehensive evaluation pipeline that rigorously assesses generated data across statistical fidelity, clinical consistency, and privacy preservation.
We evaluate SynLLM across three public medical datasets, including Diabetes, Cirrhosis, and Stroke, using 20 open-source LLMs. Our results show that prompt engineering significantly impacts data quality and privacy risk, with rule-based prompts achieving the best privacy-quality balance. SynLLM establishes that, when guided by well-designed prompts and evaluated with robust, multi-metric criteria, LLMs can generate synthetic medical data that is both clinically plausible and privacy-aware, paving the way for safer and more effective data sharing in healthcare research. 

**Abstract (ZH)**: 访问真实医疗数据往往受限于隐私法规，对医疗卫生研究的进步构成了显著障碍。合成数据提供了一种有前景的替代方案；然而，生成具有现实意义、临床有效性且隐私意识强的记录仍然是一个重大挑战。大型语言模型（LLMs）的最新进展为结构化数据生成提供了新的机会；然而，现有的方法通常缺乏系统的提示策略和全面的多维度评估框架。

在本文中，我们提出了SynLLM，这是一个模块化框架，利用20个最先进的开源LLMs（包括LLaMA、Mistral和GPT变体），通过结构化提示生成高质量的合成医疗表格数据。我们提出了四种不同的提示类型，从基于示例到基于规则的约束，这些类型编码了模式、元数据和领域知识，以控制生成过程而不进行模型微调。我们的框架包含一个全面的评估流程，该流程严格评估生成数据的统计保真度、临床一致性和隐私保护。

我们使用20个开源LLM在三个公开医疗数据集中评估SynLLM，包括糖尿病、肝硬化和中风数据集。结果表明，提示工程显著影响数据质量和隐私风险，基于规则的提示实现了最佳的隐私-质量平衡。SynLLM证明了，在精心设计的提示引导和严格的多指标评估下，LLMs能够生成既临床合理又隐私意识强的合成医疗数据，为进一步实现更安全、更有效的医疗卫生研究数据共享铺平了道路。 

---
# GVGAI-LLM: Evaluating Large Language Model Agents with Infinite Games 

**Title (ZH)**: GVGAI-LLM：无限游戏评估大型语言模型代理 

**Authors**: Yuchen Li, Cong Lin, Muhammad Umair Nasir, Philip Bontrager, Jialin Liu, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2508.08501)  

**Abstract**: We introduce GVGAI-LLM, a video game benchmark for evaluating the reasoning and problem-solving capabilities of large language models (LLMs). Built on the General Video Game AI framework, it features a diverse collection of arcade-style games designed to test a model's ability to handle tasks that differ from most existing LLM benchmarks. The benchmark leverages a game description language that enables rapid creation of new games and levels, helping to prevent overfitting over time. Each game scene is represented by a compact set of ASCII characters, allowing for efficient processing by language models. GVGAI-LLM defines interpretable metrics, including the meaningful step ratio, step efficiency, and overall score, to assess model behavior. Through zero-shot evaluations across a broad set of games and levels with diverse challenges and skill depth, we reveal persistent limitations of LLMs in spatial reasoning and basic planning. Current models consistently exhibit spatial and logical errors, motivating structured prompting and spatial grounding techniques. While these interventions lead to partial improvements, the benchmark remains very far from solved. GVGAI-LLM provides a reproducible testbed for advancing research on language model capabilities, with a particular emphasis on agentic behavior and contextual reasoning. 

**Abstract (ZH)**: GVGAI-LLM：一种评估大规模语言模型推理与问题解决能力的视频游戏基准 

---
# Large Language Models as Oracles for Ontology Alignment 

**Title (ZH)**: 大型语言模型作为本体对齐的 oracle 

**Authors**: Sviatoslav Lushnei, Dmytro Shumskyi, Severyn Shykula, Ernesto Jimenez-Ruiz, Artur d'Avila Garcez  

**Link**: [PDF](https://arxiv.org/pdf/2508.08500)  

**Abstract**: Ontology alignment plays a crucial role in integrating diverse data sources across domains. There is a large plethora of systems that tackle the ontology alignment problem, yet challenges persist in producing highly quality correspondences among a set of input ontologies. Human-in-the-loop during the alignment process is essential in applications requiring very accurate mappings. User involvement is, however, expensive when dealing with large ontologies. In this paper, we explore the feasibility of using Large Language Models (LLM) as an alternative to the domain expert. The use of the LLM focuses only on the validation of the subset of correspondences where an ontology alignment system is very uncertain. We have conducted an extensive evaluation over several matching tasks of the Ontology Alignment Evaluation Initiative (OAEI), analysing the performance of several state-of-the-art LLMs using different ontology-driven prompt templates. The LLM results are also compared against simulated Oracles with variable error rates. 

**Abstract (ZH)**: 本体对齐在跨领域整合多种数据源中扮演着关键角色。尽管存在许多解决本体对齐问题的系统，但在一组输入本体之间生成高质对应关系的问题仍然存在挑战。在需要非常准确映射的应用中，对齐过程中的人工干预是必不可少的。然而，处理大型本体时，用户参与是非常昂贵的。本文探讨了使用大型语言模型（LLM）作为领域专家替代方案的可行性。LLM 的使用仅集中在本体对齐系统非常不确定的对应关系子集的验证上。我们对本体对齐评估倡议（OAEI）的多项匹配任务进行了广泛评估，使用不同的本体驱动提示模板分析了若干先进 LLM 的性能。LLM 的结果还与具有可变错误率的模拟 Oracle 进行了比较。 

---
# Beyond Ordinal Preferences: Why Alignment Needs Cardinal Human Feedback 

**Title (ZH)**: 超越序数偏好：为何需要 cardinal 人类反馈进行对齐 

**Authors**: Parker Whitfill, Stewy Slocum  

**Link**: [PDF](https://arxiv.org/pdf/2508.08486)  

**Abstract**: Alignment techniques for LLMs rely on optimizing preference-based objectives -- where these preferences are typically elicited as ordinal, binary choices between responses. Recent work has focused on improving label quality or mitigating particular biases, but we identify a more fundamental limitation: these methods collect the wrong kind of data. We prove an impossibility result: no algorithm relying solely on ordinal comparisons can systematically recover the most preferred model. Intuitively, ordinal data lacks the information needed to resolve tradeoffs -- e.g., fixing a factual error on one prompt versus improving style on another. We show that selecting the optimal model requires recovering preferences over \emph{models} (rather than just responses), which can only be identified given cardinal feedback about response quality. To address this, we collect and publicly release a dataset of 25,000 cardinal judgments using willingness-to-pay elicitations, a well-established tool from experimental economics. Empirically, we find that incorporating cardinal feedback into preference fine-tuning allows models to prioritize high-impact improvements and outperform ordinal-only methods on downstream benchmarks, such as Arena-Hard. 

**Abstract (ZH)**: LLM对齐技术依赖于优化基于偏好的目标——这些偏好通常通过响应间的序贯或二元选择来获取。最近的工作集中在提高标签质量或减轻特定偏见上，但我们发现了一个更根本的限制：这些方法收集了错误类型的数据。我们证明了一个不可能性结果：仅依赖序贯比较的算法无法系统地恢复最首选的模型。直观上说，序贯数据缺乏解决权衡所需的信息——例如，修复一个提示的事实错误与改进另一个提示的风格。我们证明了选择最优模型需要恢复对模型的偏好（而不仅仅是响应），这仅在获得关于响应质量的连续反馈时才能确定。为此，我们收集并公开发布了包含25,000个连续判断的数据集，这些判断是通过意愿支付引发技术获得的，这是一种实验经济学中广泛应用的工具。实验结果显示，将连续反馈纳入偏好微调中能使模型优先考虑高影响力改进，并在Arena-Hard等下游基准测试中优于仅依赖序贯数据的方法。 

---
# OverFill: Two-Stage Models for Efficient Language Model Decoding 

**Title (ZH)**: OverFill: 两阶段模型以提高语言模型解码效率 

**Authors**: Woojeong Kim, Junxiong Wang, Jing Nathan Yan, Mohamed Abdelfattah, Alexander M. Rush  

**Link**: [PDF](https://arxiv.org/pdf/2508.08446)  

**Abstract**: Large language models (LLMs) excel across diverse tasks but face significant deployment challenges due to high inference costs. LLM inference comprises prefill (compute-bound) and decode (memory-bound) stages, with decode dominating latency particularly for long sequences. Current decoder-only models handle both stages uniformly, despite their distinct computational profiles. We propose OverFill, which decouples these stages to optimize accuracy-efficiency tradeoffs. OverFill begins with a full model for prefill, processing system and user inputs in parallel. It then switches to a dense pruned model, while generating tokens sequentially. Leveraging more compute during prefill, OverFill improves generation quality with minimal latency overhead. Our 3B-to-1B OverFill configuration outperforms 1B pruned models by 83.2%, while the 8B-to-3B configuration improves over 3B pruned models by 79.2% on average across standard benchmarks. OverFill matches the performance of same-sized models trained from scratch, while using significantly less training data. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务上表现出色，但由于高推断成本面临显著的部署挑战。LLM推断包括预填（计算密集型）和解码（内存密集型）阶段，其中解码尤其在长序列上主导了延迟。当前的仅解码模型在处理这两个阶段时采用统一方式，尽管它们的计算特性截然不同。我们提出OverFill，将这些阶段分离，以优化准确性和效率的权衡。OverFill 以完整模型开始预填，同时并行处理系统和用户输入。随后切换到密集剪裁模型，在此过程中顺序生成令牌。在预填阶段利用更多计算资源，OverFill 以最小的延迟开销提高生成质量。我们的3B到1B配置比1B剪裁模型高出83.2%，而8B到3B配置在标准基准测试中平均比3B剪裁模型高79.2%。OverFill 使用显著较少的训练数据匹配相同大小模型从零开始训练的性能。我们的代码可在以下链接获取。 

---
# UrzaGPT: LoRA-Tuned Large Language Models for Card Selection in Collectible Card Games 

**Title (ZH)**: UrzaGPT: LoRA 调整的大语言模型在集换式卡牌游戏中的牌组选择应用 

**Authors**: Timo Bertram  

**Link**: [PDF](https://arxiv.org/pdf/2508.08382)  

**Abstract**: Collectible card games (CCGs) are a difficult genre for AI due to their partial observability, long-term decision-making, and evolving card sets. Due to this, current AI models perform vastly worse than human players at CCG tasks such as deckbuilding and gameplay. In this work, we introduce $\textit{UrzaGPT}$, a domain-adapted large language model that recommends real-time drafting decisions in $\textit{Magic: The Gathering}$. Starting from an open-weight LLM, we use Low-Rank Adaptation fine-tuning on a dataset of annotated draft logs. With this, we leverage the language modeling capabilities of LLM, and can quickly adapt to different expansions of the game. We benchmark $\textit{UrzaGPT}$ in comparison to zero-shot LLMs and the state-of-the-art domain-specific model. Untuned, small LLMs like Llama-3-8B are completely unable to draft, but the larger GPT-4o achieves a zero-shot performance of $43\%$. Using UrzaGPT to fine-tune smaller models, we achieve an accuracy of $66.2\%$ using only 10,000 steps. Despite this not reaching the capability of domain-specific models, we show that solely using LLMs to draft is possible and conclude that using LLMs can enable performant, general, and update-friendly drafting AIs in the future. 

**Abstract (ZH)**: Collectible卡牌游戏（CCGs）因部分可观测性、长期决策制定以及不断演变的卡牌集成为AI设计带来了挑战。因此，当前的AI模型在构建卡组和游戏玩法等CCG任务上远不如人类玩家表现优异。本文中，我们介绍了$\textit{UrzaGPT}$，这是一种针对$\textit{万智牌}$实时选秀决策推荐的大语言模型。从开放权重的大语言模型开始，我们使用低秩适应微调技术在标注的选秀日志数据集上进行训练。通过这种方式，我们利用了大语言模型的语言建模能力，并能够快速适应游戏的不同扩展版本。我们以零样本大语言模型和最先进的领域特定模型为基准，评估了$\textit{UrzaGPT}$的表现。未经微调的小模型如Llama-3-8B完全无法进行选秀，但较大的模型GPT-4o的零样本性能达到了43%。通过使用$\textit{UrzaGPT}$微调较小的模型，仅用10,000步即可达到66.2%的准确率。虽然这并未达到领域特定模型的能力，但我们展示了仅使用大语言模型进行选秀的可行性，并得出结论认为使用大语言模型可以在未来实现高性能、通用且易于更新的选秀AI。 

---
# First Ask Then Answer: A Framework Design for AI Dialogue Based on Supplementary Questioning with Large Language Models 

**Title (ZH)**: 先询问再回答：基于大型语言模型补充提问的AI对话框架设计 

**Authors**: Chuanruo Fu, Yuncheng Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.08308)  

**Abstract**: Large Language Models (LLMs) often struggle to deliver accurate and actionable answers when user-provided information is incomplete or ill-specified. We propose a new interaction paradigm, First Ask Then Answer (FATA), in which, through prompt words, LLMs are guided to proactively generate multidimensional supplementary questions for users prior to response generation. Subsequently, by integrating user-provided supplementary information with the original query through sophisticated prompting techniques, we achieve substantially improved response quality and relevance. In contrast to existing clarification approaches -- such as the CLAM framework oriented to ambiguity and the self-interrogation Self-Ask method -- FATA emphasizes completeness (beyond mere disambiguation) and user participation (inviting human input instead of relying solely on model-internal reasoning). It also adopts a single-turn strategy: all clarifying questions are produced at once, thereby reducing dialogue length and improving efficiency. Conceptually, FATA uses the reasoning power of LLMs to scaffold user expression, enabling non-expert users to formulate more comprehensive and contextually relevant queries. To evaluate FATA, we constructed a multi-domain benchmark and compared it with two controls: a baseline prompt (B-Prompt) and a context-enhanced expert prompt (C-Prompt). Experimental results show that FATA outperforms B-Prompt by approximately 40% in aggregate metrics and exhibits a coefficient of variation 8% lower than C-Prompt, indicating superior stability. 

**Abstract (ZH)**: Large Language Models (LLMs)往往在用户提供的信息不完整或不明确时难以提供准确和可行的答案。我们提出了一种新的交互范式——先问后答（FATA），在这种范式中，通过提示词引导LLMs在生成回应之前主动生成多维度的补充问题。随后，通过复杂的提示技术将用户提供的补充信息与原始查询相结合，实现显著改进的回复质量和相关性。与现有的澄清方法（如面向歧义的CLAM框架和自我提问的Self-Ask方法）相比，FATA强调完整性（而不仅仅是消歧义）和用户参与（邀请用户输入而非仅依赖模型内部推理）。此外，FATA采用单轮策略：所有澄清问题一次性生成，从而减少对话长度并提高效率。概念上，FATA利用LLMs的推理能力支撑用户表达，使非专家用户能够提出更全面和上下文相关的问题。为了评估FATA，我们构建了一个多域基准，并与两个对照组进行了比较：基线提示（B-Prompt）和上下文增强的专家提示（C-Prompt）。实验结果显示，FATA在聚合指标上比B-Prompt高出约40%，且其变异系数比C-Prompt低8%，表明其稳定性更优。 

---
# LLM-BI: Towards Fully Automated Bayesian Inference with Large Language Models 

**Title (ZH)**: LLM-BI: 向完全自动化的大型语言模型贝叶斯推断迈进 

**Authors**: Yongchao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08300)  

**Abstract**: A significant barrier to the widespread adoption of Bayesian inference is the specification of prior distributions and likelihoods, which often requires specialized statistical expertise. This paper investigates the feasibility of using a Large Language Model (LLM) to automate this process. We introduce LLM-BI (Large Language Model-driven Bayesian Inference), a conceptual pipeline for automating Bayesian workflows. As a proof-of-concept, we present two experiments focused on Bayesian linear regression. In Experiment I, we demonstrate that an LLM can successfully elicit prior distributions from natural language. In Experiment II, we show that an LLM can specify the entire model structure, including both priors and the likelihood, from a single high-level problem description. Our results validate the potential of LLMs to automate key steps in Bayesian modeling, enabling the possibility of an automated inference pipeline for probabilistic programming. 

**Abstract (ZH)**: 广泛采用贝叶斯推断的一个显著障碍是先验分布和似然函数的指定，这通常需要专门的统计学专业知识。本文探讨了使用大规模语言模型（LLM）来自动化这一过程的可能性。我们引入了LLM-BI（由大规模语言模型驱动的贝叶斯推断）的概念性流程，以自动化贝叶斯工作流。作为概念验证，我们展示了两个关于贝叶斯线性回归的实验。在实验I中，我们证明了LLM可以从自然语言中成功地提取先验分布。在实验II中，我们展示了LLM可以从高层次的问题描述中指定整个模型结构，包括先验和似然函数。我们的结果验证了大规模语言模型在自动化贝叶斯建模关键步骤方面的潜力，使得概率编程的自动化推断流水线成为可能。 

---
# Topos Theory for Generative AI and LLMs 

**Title (ZH)**: 拓扑理论在生成型AI和大规模语言模型中的应用 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08293)  

**Abstract**: We propose the design of novel categorical generative AI architectures (GAIAs) using topos theory, a type of category that is ``set-like": a topos has all (co)limits, is Cartesian closed, and has a subobject classifier. Previous theoretical results on the Transformer model have shown that it is a universal sequence-to-sequence function approximator, and dense in the space of all continuous functions with compact support on the Euclidean space of embeddings of tokens. Building on this theoretical result, we explore novel architectures for LLMs that exploit the property that the category of LLMs, viewed as functions, forms a topos. Previous studies of large language models (LLMs) have focused on daisy-chained linear architectures or mixture-of-experts. In this paper, we use universal constructions in category theory to construct novel LLM architectures based on new types of compositional structures. In particular, these new compositional structures are derived from universal properties of LLM categories, and include pullback, pushout, (co) equalizers, exponential objects, and subobject classifiers. We theoretically validate these new compositional structures by showing that the category of LLMs is (co)complete, meaning that all diagrams have solutions in the form of (co)limits. Building on this completeness result, we then show that the category of LLMs forms a topos, a ``set-like" category, which requires showing the existence of exponential objects as well as subobject classifiers. We use a functorial characterization of backpropagation to define a potential implementation of an LLM topos architecture. 

**Abstract (ZH)**: 我们提出了一种新型分类生成AI架构（GAIAs）的设计，使用拓扑理论，这是一种“集合-like”的类型：拓扑具有所有（余）极限，是笛卡尔闭包的，并具有子对象分类器。基于Transformer模型的先前理论成果，我们探讨了利用大型语言模型（LLMs）范畴作为函数形式具有的性质的新型LLM架构。先前对大型语言模型的研究主要集中在线性架构的级联或专家混合。在本文中，我们利用范畴论中的普遍构造来基于新的组合结构构建新型LLM架构。特别是，这些新的组合结构源自LLM范畴的普遍性质，包括积、余积、（余）等化器、指数对象和子对象分类器。我们通过证明LLM范畴（余）完备性来理论验证这些新的组合结构，意味着所有图表都有（余）极限的形式解。基于这一完备性结果，我们进一步证明LLM范畴形成一个“集合-like”的范畴，即拓扑，这需要证明指数对象及子对象分类器的存在性。我们通过函子化梯度反向传播的刻画来定义一个LLM拓扑架构的潜在实现方案。 

---
# Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models 

**Title (ZH)**: 时间是一个特征：利用扩散语言模型中的时间动态 

**Authors**: Wen Wang, Bozhen Fang, Chenchen Jing, Yongliang Shen, Yangyi Shen, Qiuyu Wang, Hao Ouyang, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.09138)  

**Abstract**: Diffusion large language models (dLLMs) generate text through iterative denoising, yet current decoding strategies discard rich intermediate predictions in favor of the final output. Our work here reveals a critical phenomenon, temporal oscillation, where correct answers often emerge in the middle process, but are overwritten in later denoising steps. To address this issue, we introduce two complementary methods that exploit temporal consistency: 1) Temporal Self-Consistency Voting, a training-free, test-time decoding strategy that aggregates predictions across denoising steps to select the most consistent output; and 2) a post-training method termed Temporal Consistency Reinforcement, which uses Temporal Semantic Entropy (TSE), a measure of semantic stability across intermediate predictions, as a reward signal to encourage stable generations. Empirical results across multiple benchmarks demonstrate the effectiveness of our approach. Using the negative TSE reward alone, we observe a remarkable average improvement of 24.7% on the Countdown dataset over an existing dLLM. Combined with the accuracy reward, we achieve absolute gains of 2.0% on GSM8K, 4.3% on MATH500, 6.6% on SVAMP, and 25.3% on Countdown, respectively. Our findings underscore the untapped potential of temporal dynamics in dLLMs and offer two simple yet effective tools to harness them. 

**Abstract (ZH)**: 扩散大语言模型（dLLMs）通过迭代去噪生成文本，然而当前的解码策略倾向于丢弃中间丰富的预测，保留最终输出。我们的工作揭示了一个关键现象：时间振荡，即正确的答案往往在中间过程中出现，但在后续去噪步骤中被覆盖。为了解决这一问题，我们引入了两种互补的方法，利用时间一致性：1）时间自洽投票，一种无需训练的测试时解码策略，通过聚合去噪步骤中的预测来选择最一致的输出；2）一种后训练方法，称为时间一致性强化，使用时间语义熵（TSE），衡量中间预测中语义稳定性的度量，作为奖励信号，以促进稳定的生成。多基准上的实验证明了我们方法的有效性。仅使用负TSE奖励，我们在Countdown数据集上观察到现有dLLM的平均改进幅度为24.7%。结合准确性奖励后，我们在GSM8K上实现绝对提升2.0%，在MATH500上实现4.3%的提升，在SVAMP上实现6.6%的提升，在Countdown上实现25.3%的提升。我们的研究成果强调了dLLMs中未充分利用的时间动态潜力，并提供了两种简单而有效的工具来利用它们。 

---
# Can We Trust AI to Govern AI? Benchmarking LLM Performance on Privacy and AI Governance Exams 

**Title (ZH)**: 我们能信任AI治理AI吗？基于隐私和AI治理考试对LLM性能的benchmarking研究 

**Authors**: Zane Witherspoon, Thet Mon Aye, YingYing Hao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09036)  

**Abstract**: The rapid emergence of large language models (LLMs) has raised urgent questions across the modern workforce about this new technology's strengths, weaknesses, and capabilities. For privacy professionals, the question is whether these AI systems can provide reliable support on regulatory compliance, privacy program management, and AI governance. In this study, we evaluate ten leading open and closed LLMs, including models from OpenAI, Anthropic, Google DeepMind, Meta, and DeepSeek, by benchmarking their performance on industry-standard certification exams: CIPP/US, CIPM, CIPT, and AIGP from the International Association of Privacy Professionals (IAPP). Each model was tested using official sample exams in a closed-book setting and compared to IAPP's passing thresholds. Our findings show that several frontier models such as Gemini 2.5 Pro and OpenAI's GPT-5 consistently achieve scores exceeding the standards for professional human certification - demonstrating substantial expertise in privacy law, technical controls, and AI governance. The results highlight both the strengths and domain-specific gaps of current LLMs and offer practical insights for privacy officers, compliance leads, and technologists assessing the readiness of AI tools for high-stakes data governance roles. This paper provides an overview for professionals navigating the intersection of AI advancement and regulatory risk and establishes a machine benchmark based on human-centric evaluations. 

**Abstract (ZH)**: 大语言模型的迅速涌现引起了现代劳动力关于这项新技术的优势、弱点和能力的迫切问题。对于隐私专业人员而言，问题在于这些AI系统能否提供可靠的合规支持、隐私项目管理以及AI治理。在本研究中，我们通过评估由开放和闭源大语言模型组成的十种领先模型，包括来自OpenAI、Anthropic、Google DeepMind、Meta和DeepSeek的模型，并运用国际隐私专业人员协会（IAPP）的标准认证考试：CIPP/US、CIPM、CIPT和AIGP，对其性能进行了基准测试。每种模型都在闭卷条件下使用官方样本考试进行测试，并与IAPP的通过标准进行对比。研究发现，如Gemini 2.5 Pro和OpenAI的GPT-5等前沿模型，其得分持续超过专业人类认证的标准，展示了其在隐私法、技术控制和AI治理方面的显著专业知识。结果突显了当前LLM的优势和领域特定的差距，并为评估AI工具在高风险数据治理角色中准备情况的隐私专员、合规负责人和技术专家提供了实用见解。本文为专业人员提供了AI进步与监管风险交叉领域的概览，并基于以人为中心的评估建立了机器基准。 

---
# Attacks and Defenses Against LLM Fingerprinting 

**Title (ZH)**: 针对LLM特征标识的攻击与防御 

**Authors**: Kevin Kurian, Ethan Holland, Sean Oesch  

**Link**: [PDF](https://arxiv.org/pdf/2508.09021)  

**Abstract**: As large language models are increasingly deployed in sensitive environments, fingerprinting attacks pose significant privacy and security risks. We present a study of LLM fingerprinting from both offensive and defensive perspectives. Our attack methodology uses reinforcement learning to automatically optimize query selection, achieving better fingerprinting accuracy with only 3 queries compared to randomly selecting 3 queries from the same pool. Our defensive approach employs semantic-preserving output filtering through a secondary LLM to obfuscate model identity while maintaining semantic integrity. The defensive method reduces fingerprinting accuracy across tested models while preserving output quality. These contributions show the potential to improve fingerprinting tools capabilities while providing practical mitigation strategies against fingerprinting attacks. 

**Abstract (ZH)**: 大型语言模型在敏感环境中的部署增加了指纹攻击的隐私和安全风险。我们从攻击和防御两个角度对LLM指纹攻击进行了研究。我们的攻击方法使用强化学习自动优化查询选择，仅用3次查询就能比随机从同一池中选择3次查询获得更好的指纹识别准确性。我们的防御方法通过secondary LLM进行语义保留的输出过滤，以混淆模型身份同时保持语义完整性。防御方法在测试的模型中降低了指纹识别准确性，同时保持输出质量。这些贡献展示了改进指纹工具能力并提供防止指纹攻击的实际缓解策略的潜力。 

---
# Retrospective Sparse Attention for Efficient Long-Context Generation 

**Title (ZH)**: 回顾稀疏注意力机制以实现高效长上下文生成 

**Authors**: Seonghwan Choi, Beomseok Kang, Dongwon Jo, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.09001)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in long-context tasks such as reasoning, code generation, and multi-turn dialogue. However, inference over extended contexts is bottlenecked by the Key-Value (KV) cache, whose memory footprint grows linearly with sequence length and dominates latency at each decoding step. While recent KV cache compression methods identify and load important tokens, they focus predominantly on input contexts and fail to address the cumulative attention errors that arise during long decoding. In this paper, we introduce RetroAttention, a novel KV cache update technique that retrospectively revises past attention outputs using newly arrived KV entries from subsequent decoding steps. By maintaining a lightweight output cache, RetroAttention enables past queries to efficiently access more relevant context, while incurring minimal latency overhead. This breaks the fixed-attention-output paradigm and allows continual correction of prior approximations. Extensive experiments on long-generation benchmarks show that RetroAttention consistently outperforms state-of-the-art (SOTA) KV compression methods, increasing effective KV exposure by up to 1.6$\times$ and accuracy by up to 21.9\%. 

**Abstract (ZH)**: 大型语言模型（LLMs）在诸如推理、代码生成和多轮对话等长上下文任务中的应用日益增多。然而，长上下文推理受关键值（KV）缓存的瓶颈限制，其内存占用随序列长度线性增长，并在每次解码步骤中占主导地位。尽管最近的KV缓存压缩方法识别并加载重要令牌，但它们主要关注输入上下文，并未能解决长时间解码过程中累积的注意力错误。在这篇论文中，我们提出了一种新的KV缓存更新技术——RetroAttention，它通过回顾性地使用后续解码步骤中新到达的KV条目来修订过去的注意力输出。通过维护一个轻量级的输出缓存，RetroAttention使过去的查询能够高效地访问更多相关上下文，同时引入的延迟开销较小。这打破了固定注意力输出的范式，并允许持续修正先前的近似值。在长期生成基准测试中，RetroAttention一致地优于最先进的（SOTA）KV压缩方法，使有效的KV暴露增加至1.6倍，并提高准确性至21.9%。 

---
# Train Long, Think Short: Curriculum Learning for Efficient Reasoning 

**Title (ZH)**: 长训练，短推理：渐增学习以实现高效推理 

**Authors**: Hasan Abed Al Kader Hammoud, Kumail Alhamoud, Abed Hammoud, Elie Bou-Zeid, Marzyeh Ghassemi, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.08940)  

**Abstract**: Recent work on enhancing the reasoning abilities of large language models (LLMs) has introduced explicit length control as a means of constraining computational cost while preserving accuracy. However, existing approaches rely on fixed-length training budgets, which do not take advantage of the natural progression from exploration to compression during learning. In this work, we propose a curriculum learning strategy for length-controlled reasoning using Group Relative Policy Optimization (GRPO). Our method starts with generous token budgets and gradually tightens them over training, encouraging models to first discover effective solution strategies and then distill them into more concise reasoning traces. We augment GRPO with a reward function that balances three signals: task correctness (via verifier feedback), length efficiency, and formatting adherence (via structural tags). Experiments on GSM8K, MATH500, SVAMP, College Math, and GSM+ demonstrate that curriculum-based training consistently outperforms fixed-budget baselines at the same final budget, achieving higher accuracy and significantly improved token efficiency. We further ablate the impact of reward weighting and decay schedule design, showing that progressive constraint serves as a powerful inductive bias for training efficient reasoning models. Our code and checkpoints are released at: this https URL. 

**Abstract (ZH)**: Recent工作在增强大规模语言模型（LLMs）的推理能力时引入了显式长度控制，以在限制计算成本的同时保持准确性。然而，现有方法依赖于固定的训练预算，未能利用学习过程中从探索到压缩的自然进展。在这种工作中，我们提出了使用Group Relative Policy Optimization（GRPO）的长度控制推理的课程学习策略。我们的方法以慷慨的token预算开始，并在训练过程中逐渐收紧，鼓励模型首先发现有效的解决方案策略，然后将其精简为更简洁的推理过程。我们通过引入一个平衡三个信号的奖励函数来增强GRPO：任务正确性（通过验证器反馈）、长度效率和格式合规性（通过结构标签）。实验表明，基于课程的学习方法在相同的最终预算下始终优于固定预算的基准方法，实现更高的准确性和显著提高的token效率。我们进一步分析了奖励权重和衰减计划的影响，展示了逐步约束作为训练高效推理模型的强大归纳偏置作用。我们的代码和检查点可在以下链接获取：this https URL。 

---
# ASPD: Unlocking Adaptive Serial-Parallel Decoding by Exploring Intrinsic Parallelism in LLMs 

**Title (ZH)**: ASPD: 解锁LLM内在并行性的自适应串并行解码 

**Authors**: Keyu Chen, Zhifeng Shen, Daohai Yu, Haoqian Wu, Wei Wen, Jianfeng He, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.08895)  

**Abstract**: The increasing scale and complexity of large language models (LLMs) pose significant inference latency challenges, primarily due to their autoregressive decoding paradigm characterized by the sequential nature of next-token prediction. By re-examining the outputs of autoregressive models, we observed that some segments exhibit parallelizable structures, which we term intrinsic parallelism. Decoding each parallelizable branch simultaneously (i.e. parallel decoding) can significantly improve the overall inference speed of LLMs. In this paper, we propose an Adaptive Serial-Parallel Decoding (ASPD), which addresses two core challenges: automated construction of parallelizable data and efficient parallel decoding mechanism. More specifically, we introduce a non-invasive pipeline that automatically extracts and validates parallelizable structures from the responses of autoregressive models. To empower efficient adaptive serial-parallel decoding, we implement a Hybrid Decoding Engine which enables seamless transitions between serial and parallel decoding modes while maintaining a reusable KV cache, maximizing computational efficiency. Extensive evaluations across General Tasks, Retrieval-Augmented Generation, Mathematical Reasoning, demonstrate that ASPD achieves unprecedented performance in both effectiveness and efficiency. Notably, on Vicuna Bench, our method achieves up to 3.19x speedup (1.85x on average) while maintaining response quality within 1% difference compared to autoregressive models, realizing significant acceleration without compromising generation quality. Our framework sets a groundbreaking benchmark for efficient LLM parallel inference, paving the way for its deployment in latency-sensitive applications such as AI-powered customer service bots and answer retrieval engines. 

**Abstract (ZH)**: 适配大规模语言模型的自适应串行-并 解码 

---
# Entangled in Representations: Mechanistic Investigation of Cultural Biases in Large Language Models 

**Title (ZH)**: 纠缠于表征之中：大型语言模型中文化偏见的机制性探究 

**Authors**: Haeun Yu, Seogyeong Jeong, Siddhesh Pawar, Jisu Shin, Jiho Jin, Junho Myung, Alice Oh, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.08879)  

**Abstract**: The growing deployment of large language models (LLMs) across diverse cultural contexts necessitates a better understanding of how the overgeneralization of less documented cultures within LLMs' representations impacts their cultural understanding. Prior work only performs extrinsic evaluation of LLMs' cultural competence, without accounting for how LLMs' internal mechanisms lead to cultural (mis)representation. To bridge this gap, we propose Culturescope, the first mechanistic interpretability-based method that probes the internal representations of LLMs to elicit the underlying cultural knowledge space. CultureScope utilizes a patching method to extract the cultural knowledge. We introduce a cultural flattening score as a measure of the intrinsic cultural biases. Additionally, we study how LLMs internalize Western-dominance bias and cultural flattening, which allows us to trace how cultural biases emerge within LLMs. Our experimental results reveal that LLMs encode Western-dominance bias and cultural flattening in their cultural knowledge space. We find that low-resource cultures are less susceptible to cultural biases, likely due to their limited training resources. Our work provides a foundation for future research on mitigating cultural biases and enhancing LLMs' cultural understanding. Our codes and data used for experiments are publicly available. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在多种文化背景下的应用日益增长，需要更好地理解LLMs对较少记录的文化的过度泛化的内在表现如何影响其文化理解能力。以往的工作仅从外部评估LLMs的文化能力，未能考虑到LLMs内部机制如何导致文化误表征。为弥合这一差距，我们提出了Culturescope，这是首个基于机制可解释性的方法，可以探测LLMs的内部表示以揭示其潜在的文化知识空间。CultureScope 使用修补方法提取文化知识，并引入了文化压平分数作为内在文化偏见的衡量标准。此外，我们研究了LLMs如何内化西方主导偏见和文化压平现象，这使我们能够追踪文化偏见在LLMs中如何产生。我们的实验结果表明，LLMs在其文化知识空间中编码了西方主导偏见和文化压平现象。我们发现，资源较少的文化受到文化偏见的影响较小，这很可能是因为它们的训练资源有限。我们的工作为未来减少文化偏见和提升LLMs文化理解的研究奠定了基础。我们的代码和实验数据已公开。 

---
# Oblivionis: A Lightweight Learning and Unlearning Framework for Federated Large Language Models 

**Title (ZH)**: Oblivionis：面向联邦大规模语言模型的轻量级学习与遗忘框架 

**Authors**: Fuyao Zhang, Xinyu Yan, Tiantong Wu, Wenjie Li, Tianxiang Chen, Yang Cao, Ran Yan, Longtao Huang, Wei Yang Bryan Lim, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08875)  

**Abstract**: Large Language Models (LLMs) increasingly leverage Federated Learning (FL) to utilize private, task-specific datasets for fine-tuning while preserving data privacy. However, while federated LLM frameworks effectively enable collaborative training without raw data sharing, they critically lack built-in mechanisms for regulatory compliance like GDPR's right to be forgotten. Integrating private data heightens concerns over data quality and long-term governance, yet existing distributed training frameworks offer no principled way to selectively remove specific client contributions post-training. Due to distributed data silos, stringent privacy constraints, and the intricacies of interdependent model aggregation, federated LLM unlearning is significantly more complex than centralized LLM unlearning. To address this gap, we introduce Oblivionis, a lightweight learning and unlearning framework that enables clients to selectively remove specific private data during federated LLM training, enhancing trustworthiness and regulatory compliance. By unifying FL and unlearning as a dual optimization objective, we incorporate 6 FL and 5 unlearning algorithms for comprehensive evaluation and comparative analysis, establishing a robust pipeline for federated LLM unlearning. Extensive experiments demonstrate that Oblivionis outperforms local training, achieving a robust balance between forgetting efficacy and model utility, with cross-algorithm comparisons providing clear directions for future LLM development. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地利用联邦学习（FL）来利用私有的、任务特定的数据集进行微调，同时保护数据隐私。然而，虽然联邦LLM框架有效地实现了无需共享原始数据的协作训练，但它们在内置的合规机制（如GDPR的被遗忘权）方面存在严重不足。整合私人数据加剧了对数据质量和长期治理的担忧，而现有的分布式训练框架无法提供合理的方法，在训练后选择性地移除特定客户端的贡献。由于分布式数据孤岛、严格的隐私限制以及模型聚合的复杂性，联邦LLM遗忘比中心化LLM遗忘更为复杂。为解决这一缺口，我们引入了Oblivionis，一个轻量级的学习和遗忘框架，允许客户端在联邦LLM训练过程中选择性地移除特定的私人数据，增强可信度和合规性。通过将联邦学习和遗忘统一为一个双重优化目标，我们结合使用了6种联邦学习算法和5种遗忘算法进行全面评估和比较分析，建立了联邦LLM遗忘的稳健管道。广泛的实验表明，Oblivionis在局部训练中的表现更优，实现了遗忘效果和模型实用性之间的稳健平衡，而跨算法比较为未来LLM的发展提供了清晰的方向。 

---
# BiasGym: Fantastic Biases and How to Find (and Remove) Them 

**Title (ZH)**: BiasGym: 神奇的偏差与如何发现（并消除）它们 

**Authors**: Sekh Mainul Islam, Nadav Borenstein, Siddhesh Milind Pawar, Haeun Yu, Arnav Arora, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.08855)  

**Abstract**: Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. Biased behaviour is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce BiasGym, a simple, cost-effective, and generalizable framework for reliably injecting, analyzing, and mitigating conceptual associations within LLMs. BiasGym consists of two components: BiasInject, which injects specific biases into the model via token-based fine-tuning while keeping the model frozen, and BiasScope, which leverages these injected signals to identify and steer the components responsible for biased behavior. Our method enables consistent bias elicitation for mechanistic analysis, supports targeted debiasing without degrading performance on downstream tasks, and generalizes to biases unseen during training. We demonstrate the effectiveness of BiasGym in reducing real-world stereotypes (e.g., people from a country being `reckless drivers') and in probing fictional associations (e.g., people from a country having `blue skin'), showing its utility for both safety interventions and interpretability research. 

**Abstract (ZH)**: 理解大型语言模型（LLMs）权重中编码的偏见和刻板印象对于开发有效的缓解策略至关重要。BiasGym：一种简单、经济高效且易于推广的框架，用于可靠地注入、分析和缓解LLMs中的概念关联偏见。 

---
# Steering Towards Fairness: Mitigating Political Bias in LLMs 

**Title (ZH)**: 导向公平：减轻LLM中的政治偏见 

**Authors**: Afrozah Nadeem, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2508.08846)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled their widespread use across diverse real-world applications. However, concerns remain about their tendency to encode and reproduce ideological biases, particularly along political and economic dimensions. In this paper, we propose a framework for probing and mitigating such biases in decoder-based LLMs through analysis of internal model representations. Grounded in the Political Compass Test (PCT), our method uses contrastive pairs to extract and compare hidden layer activations from models like Mistral and DeepSeek. We introduce a comprehensive activation extraction pipeline capable of layer-wise analysis across multiple ideological axes, revealing meaningful disparities linked to political framing. Our results show that decoder LLMs systematically encode representational bias across layers, which can be leveraged for effective steering vector-based mitigation. This work provides new insights into how political bias is encoded in LLMs and offers a principled approach to debiasing beyond surface-level output interventions. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）的 Recent advancements in 大型语言模型（LLMs）已在各种实际应用中得到广泛应用。然而，它们倾向于编码和再现意识形态偏见，特别是在政治和经济维度上。在本文中，我们提出了一种通过分析内部模型表示来探测和缓解基于解码器的LLMs中此类偏见的框架。基于政治极点测试（PCT），我们的方法使用对比对来提取和比较如Mistral和DeepSeek等模型的隐藏层激活。我们介绍了一种全面的激活提取管道，能够在多个意识形态轴上进行逐层分析，揭示与政治框架相关的有意义差异。我们的结果显示，解码器LLMs在各层中系统地编码了表示偏见，这可以用于基于引导向量的有效缓解。本工作提供了关于政治偏见如何编码在LLMs中的新见解，并提供了超越表面级输出干预的去偏见原则性方法。 

---
# The Roots of International Perceptions: Simulating US Attitude Changes Towards China with LLM Agents 

**Title (ZH)**: 国际感知的根源：使用大语言模型代理模拟美国对中国的态度变化 

**Authors**: Nicholas Sukiennik, Yichuan Xu, Yuqing Kan, Jinghua Piao, Yuwei Yan, Chen Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08837)  

**Abstract**: The rise of LLMs poses new possibilities in modeling opinion evolution, a long-standing task in simulation, by leveraging advanced reasoning abilities to recreate complex, large-scale human cognitive trends. While most prior works focus on opinion evolution surrounding specific isolated events or the views within a country, ours is the first to model the large-scale attitude evolution of a population representing an entire country towards another -- US citizens' perspectives towards China. To tackle the challenges of this broad scenario, we propose a framework that integrates media data collection, user profile creation, and cognitive architecture for opinion updates to successfully reproduce the real trend of US attitudes towards China over a 20-year period from 2005 to today. We also leverage LLMs' capabilities to introduce debiased media exposure, extracting neutral events from typically subjective news contents, to uncover the roots of polarized opinion formation, as well as a devils advocate agent to help explain the rare reversal from negative to positive attitudes towards China, corresponding with changes in the way Americans obtain information about the country. The simulation results, beyond validating our framework architecture, also reveal the impact of biased framing and selection bias in shaping attitudes. Overall, our work contributes to a new paradigm for LLM-based modeling of cognitive behaviors in a large-scale, long-term, cross-border social context, providing insights into the formation of international biases and offering valuable implications for media consumers to better understand the factors shaping their perspectives, and ultimately contributing to the larger social need for bias reduction and cross-cultural tolerance. 

**Abstract (ZH)**: LLMs的崛起为利用高级推理能力重新构建大规模人类认知趋势提供了新的可能性，这一趋势在模拟领域长期存在。虽然大多数先前研究侧重于特定孤立事件或单一国家内部的观点演变，我们是首次建立模型来模拟代表整个国家的美国民众对中国的态度演变。为应对这一广泛场景的挑战，我们提出了一种融合媒体数据收集、用户画像创建和认知架构的意见更新框架，以成功重现2005年至今日美国民众对中国态度的真实演变趋势。我们还利用LLMs的能力引入去偏见的媒体曝光，从主观的新闻内容中提取中立事件，以揭示极化意见形成的根源，并引入反对派代理人工智能来解释从负面到正面态度的罕见逆转，这与美国人获取关于该国信息的方式变化相吻合。模拟结果不仅验证了我们框架的结构，还揭示了偏见框架和选择偏差对态度塑造的影响。总体而言，我们的工作为基于LLM的大规模长期跨国社交环境中认知行为建模提供了新范式，为理解国际偏见的形成提供了见解，并为媒体消费者更好地理解塑造其视角的因素提供了宝贵的启示，最终有助于减少偏见和促进跨文化宽容。 

---
# EditMF: Drawing an Invisible Fingerprint for Your Large Language Models 

**Title (ZH)**: EditMF: 为您的大型语言模型绘制无形指纹 

**Authors**: Jiaxuan Wu, Yinghan Zhou, Wanli Peng, Yiming Xue, Juan Wen, Ping Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.08836)  

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing back-door-based methods suffer from limited stealth and efficiency. To simultaneously address these issues, we propose EditMF, a training-free fingerprinting paradigm that achieves highly imperceptible fingerprint embedding with minimal computational overhead. Ownership bits are mapped to compact, semantically coherent triples drawn from an encrypted artificial knowledge base (e.g., virtual author-novel-protagonist facts). Causal tracing localizes the minimal set of layers influencing each triple, and a zero-space update injects the fingerprint without perturbing unrelated knowledge. Verification requires only a single black-box query and succeeds when the model returns the exact pre-embedded protagonist. Empirical results on LLaMA and Qwen families show that EditMF combines high imperceptibility with negligible model's performance loss, while delivering robustness far beyond LoRA-based fingerprinting and approaching that of SFT embeddings. Extensive experiments demonstrate that EditMF is an effective and low-overhead solution for secure LLM ownership verification. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的训练耗资源且成本高，因此保护LLMs的知识产权至关重要。嵌入特征码到LLMs已成为确立模型所有权的普遍方法。然而，现有的后门基于方法存在隐蔽性和效率有限的问题。为同时解决这些问题，我们提出了EditMF，这是一种无需训练的特征码嵌入范式，能够实现高度不可感知的特征码嵌入，并且具有极小的计算开销。所有权位被映射到来自加密的人工知识库（例如，虚拟作者-小说- protagonista事实）的紧凑且语义一致的三元组中。因果追踪将定位影响每个三元组的最小层集，并使用零空间更新注入特征码而不干扰无关知识。验证只需一次黑盒查询，在模型返回预嵌入的主人公时成功。在LLaMA和Qwen家族的实证结果表明，EditMF结合了高不可感知性与几乎可以忽略的模型性能损失，并且展现了远超LoRA基于的特征码嵌入的稳健性，接近于SFT嵌入的稳健性。大量实验表明，EditMF是一种有效的低开销解决方案，用于安全的大规模语言模型所有权验证。 

---
# An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems 

**Title (ZH)**: 对Large Language Models在数学推理中鲁棒性的研究：通过高级数学问题的数学等价转换进行基准测试 

**Authors**: Yuren Hao, Xiang Wan, Chengxiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2508.08833)  

**Abstract**: In this paper, we introduce a systematic framework beyond conventional method to assess LLMs' mathematical-reasoning robustness by stress-testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMs to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 18 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 49 % on the originals but drops by 4 percentage points on surface variants, and by 10.5 percentage points on core-step-based variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities. 

**Abstract (ZH)**: 本文介绍了一种超越传统方法的系统性框架，通过在具有数学等价性但具有语义和参数变化的高级数学问题上进行压力测试，来评估LLMs的数学推理稳健性。这些变换使我们能够测量LLMs对非数学干扰的敏感性，从而更准确地评估其数学推理能力。使用这种新的评估方法，我们创建了PutnamGAP，这是一个新的基准数据集，包含多个竞赛级别数学问题的数学等价变体。借助新的数据集，我们评估了多个代表性LLM家族，并检查了它们的稳健性。在18个商用和开源模型中，我们观察到变体上的性能显著下降。OpenAI的旗舰推理模型O3在原始问题上的得分为49%，但在表面变体上下降了4个百分点，在核心步骤基于的变体上下降了10.5个百分点，而较小的模型表现更差。总体而言，结果表明，提出的新的评估方法有效加深了我们对LLMs稳健性的理解，并为提高其数学推理能力提供了新的见解。 

---
# Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments 

**Title (ZH)**: 基于反馈驱动的工具使用改进：自动构建环境在大型语言模型中的应用 

**Authors**: Junjie Ye, Changhao Jiang, Zhengyin Du, Yufei Xu, Xuesong Yao, Zhiheng Xi, Xiaoran Fan, Qi Zhang, Xuanjing Huang, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08791)  

**Abstract**: Effective tool use is essential for large language models (LLMs) to interact meaningfully with their environment. However, progress is limited by the lack of efficient reinforcement learning (RL) frameworks specifically designed for tool use, due to challenges in constructing stable training environments and designing verifiable reward mechanisms. To address this, we propose an automated environment construction pipeline, incorporating scenario decomposition, document generation, function integration, complexity scaling, and localized deployment. This enables the creation of high-quality training environments that provide detailed and measurable feedback without relying on external tools. Additionally, we introduce a verifiable reward mechanism that evaluates both the precision of tool use and the completeness of task execution. When combined with trajectory data collected from the constructed environments, this mechanism integrates seamlessly with standard RL algorithms to facilitate feedback-driven model training. Experiments on LLMs of varying scales demonstrate that our approach significantly enhances the models' tool-use performance without degrading their general capabilities, regardless of inference modes or training algorithms. Our analysis suggests that these gains result from improved context understanding and reasoning, driven by updates to the lower-layer MLP parameters in models. 

**Abstract (ZH)**: 有效工具使用对于大型语言模型（LLMs）与其环境进行有意义的交互至关重要。然而，由于构建稳定训练环境和设计可验证奖励机制的挑战，进步受到限制。为解决这一问题，我们提出了一种自动环境构建管道，包括场景分解、文档生成、功能集成、复杂性扩展和局部部署。这使得能够创建高质量的训练环境，提供详细的可测量反馈，无需依赖外部工具。此外，我们引入了一种可验证的奖励机制，该机制评估工具使用精度和任务执行完整性。结合从构建环境中收集的轨迹数据，该机制能够无缝集成到标准RL算法中，以促进基于反馈的模型训练。实验表明，无论推理模式或训练算法如何，我们的方法都能显著提高模型的工具使用性能，而不损害其一般能力。我们的分析表明，这些改进来自通过更新模型底层MLP参数而获得的上下文理解和推理能力的提升。 

---
# Evaluating Podcast Recommendations with Profile-Aware LLM-as-a-Judge 

**Title (ZH)**: 基于个人档案的LLM作为裁判评价播客推荐 

**Authors**: Francesco Fabbri, Gustavo Penha, Edoardo D'Amico, Alice Wang, Marco De Nadai, Jackie Doremus, Paul Gigioli, Andreas Damianou, Oskar Stal, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2508.08777)  

**Abstract**: Evaluating personalized recommendations remains a central challenge, especially in long-form audio domains like podcasts, where traditional offline metrics suffer from exposure bias and online methods such as A/B testing are costly and operationally constrained. In this paper, we propose a novel framework that leverages Large Language Models (LLMs) as offline judges to assess the quality of podcast recommendations in a scalable and interpretable manner. Our two-stage profile-aware approach first constructs natural-language user profiles distilled from 90 days of listening history. These profiles summarize both topical interests and behavioral patterns, serving as compact, interpretable representations of user preferences. Rather than prompting the LLM with raw data, we use these profiles to provide high-level, semantically rich context-enabling the LLM to reason more effectively about alignment between a user's interests and recommended episodes. This reduces input complexity and improves interpretability. The LLM is then prompted to deliver fine-grained pointwise and pairwise judgments based on the profile-episode match. In a controlled study with 47 participants, our profile-aware judge matched human judgments with high fidelity and outperformed or matched a variant using raw listening histories. The framework enables efficient, profile-aware evaluation for iterative testing and model selection in recommender systems. 

**Abstract (ZH)**: 基于大型语言模型的个性化推荐评估框架：面向播客等长音频领域的可扩展和可解释方法 

---
# DevNous: An LLM-Based Multi-Agent System for Grounding IT Project Management in Unstructured Conversation 

**Title (ZH)**: DevNous: 一个基于LLM的多agent系统，将IT项目管理接地于无结构对话之中 

**Authors**: Stavros Doropoulos, Stavros Vologiannidis, Ioannis Magnisalis  

**Link**: [PDF](https://arxiv.org/pdf/2508.08761)  

**Abstract**: The manual translation of unstructured team dialogue into the structured artifacts required for Information Technology (IT) project governance is a critical bottleneck in modern information systems management. We introduce DevNous, a Large Language Model-based (LLM) multi-agent expert system, to automate this unstructured-to-structured translation process. DevNous integrates directly into team chat environments, identifying actionable intents from informal dialogue and managing stateful, multi-turn workflows for core administrative tasks like automated task formalization and progress summary synthesis. To quantitatively evaluate the system, we introduce a new benchmark of 160 realistic, interactive conversational turns. The dataset was manually annotated with a multi-label ground truth and is publicly available. On this benchmark, DevNous achieves an exact match turn accuracy of 81.3\% and a multiset F1-Score of 0.845, providing strong evidence for its viability. The primary contributions of this work are twofold: (1) a validated architectural pattern for developing ambient administrative agents, and (2) the introduction of the first robust empirical baseline and public benchmark dataset for this challenging problem domain. 

**Abstract (ZH)**: 基于大规模语言模型的多代理专家系统DevNous在现代信息系统管理中自动化非结构化团队对话到结构化治理 artifacts 的转化过程是一个关键瓶颈。我们介绍了DevNous，一个基于大规模语言模型的多代理专家系统，以自动化这一非结构化到结构化的转化过程。DevNous直接集成到团队聊天环境中，从非正式对话中识别可操作的意图，并管理涉及核心行政任务的多轮对话流程，如自动化任务规范和进度总结合成。为了定量评估该系统，我们引入了一个包含160个真实互动对话轮次的新基准数据集。该数据集经过多标签人工标注，并已公开。在这一基准测试中，DevNous达到81.3%的准确对话轮次匹配率和0.845的多集F1分数，为其实用性提供了强有力证据。本文的主要贡献有两个方面：（1）一个经过验证的架构模式用于开发环境中的行政代理，（2）首次为这一挑战性问题域提供了一个稳健的实证基准和公开基准数据集。 

---
# SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs 

**Title (ZH)**: SciRerankBench: 评估重排序器以支持科学检索增强生成型LLM 

**Authors**: Haotian Chen, Qingqing Long, Meng Xiao, Xiao Luo, Wei Ju, Chengrui Wang, Xuezhi Wang, Yuanchun Zhou, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08742)  

**Abstract**: Scientific literature question answering is a pivotal step towards new scientific discoveries. Recently, \textit{two-stage} retrieval-augmented generated large language models (RAG-LLMs) have shown impressive advancements in this domain. Such a two-stage framework, especially the second stage (reranker), is particularly essential in the scientific domain, where subtle differences in terminology may have a greatly negative impact on the final factual-oriented or knowledge-intensive answers. Despite this significant progress, the potential and limitations of these works remain unexplored. In this work, we present a Scientific Rerank-oriented RAG Benchmark (SciRerankBench), for evaluating rerankers within RAG-LLMs systems, spanning five scientific subjects. To rigorously assess the reranker performance in terms of noise resilience, relevance disambiguation, and factual consistency, we develop three types of question-context-answer (Q-C-A) pairs, i.e., Noisy Contexts (NC), Semantically Similar but Logically Irrelevant Contexts (SSLI), and Counterfactual Contexts (CC). Through systematic evaluation of 13 widely used rerankers on five families of LLMs, we provide detailed insights into their relative strengths and limitations. To the best of our knowledge, SciRerankBench is the first benchmark specifically developed to evaluate rerankers within RAG-LLMs, which provides valuable observations and guidance for their future development. 

**Abstract (ZH)**: 科学文献问答是新科学发现的关键步骤。近年来，两阶段检索增强生成大型语言模型（RAG-LLMs）在这一领域展现了令人印象深刻的进展。尤其是在科学领域，这种两阶段框架，尤其是第二个阶段（重排序器），尤为重要，因为术语上的细微差异可能会对最终的事实导向或知识密集型答案产生严重影响。尽管取得了显著进展，但这些工作的潜力和局限性尚未被充分探讨。在本项工作中，我们提出了一个科学重排序导向的RAG基准（SciRerankBench），用于评估RAG-LLMs系统中的重排序器，涵盖了五个科学主题。为了严格按照噪声鲁棒性、语义歧义消解和事实一致性来评估重排序器的性能，我们开发了三种类型的问题-上下文-答案（Q-C-A）对，即噪音上下文（NC）、语义相似但逻辑无关的上下文（SSLI）和反事实上下文（CC）。通过在五大家族的LLM上系统评估13种广泛使用的重排序器，我们提供了有关它们相对优势和局限性的详细见解。据我们所知，SciRerankBench是首个专门用于评估RAG-LLMs中重排序器的基准，它为未来的发展提供了有价值的观察和指导。 

---
# IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization 

**Title (ZH)**: IROTE: 大型语言模型基于上下文自我反思优化的人类特质 elicitation 

**Authors**: Yuzhuo Bai, Shitong Duan, Muhua Huang, Jing Yao, Zhenghao Liu, Peng Zhang, Tun Lu, Xiaoyuan Yi, Maosong Sun, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.08719)  

**Abstract**: Trained on various human-authored corpora, Large Language Models (LLMs) have demonstrated a certain capability of reflecting specific human-like traits (e.g., personality or values) by prompting, benefiting applications like personalized LLMs and social simulations. However, existing methods suffer from the superficial elicitation problem: LLMs can only be steered to mimic shallow and unstable stylistic patterns, failing to embody the desired traits precisely and consistently across diverse tasks like humans. To address this challenge, we propose IROTE, a novel in-context method for stable and transferable trait elicitation. Drawing on psychological theories suggesting that traits are formed through identity-related reflection, our method automatically generates and optimizes a textual self-reflection within prompts, which comprises self-perceived experience, to stimulate LLMs' trait-driven behavior. The optimization is performed by iteratively maximizing an information-theoretic objective that enhances the connections between LLMs' behavior and the target trait, while reducing noisy redundancy in reflection without any fine-tuning, leading to evocative and compact trait reflection. Extensive experiments across three human trait systems manifest that one single IROTE-generated self-reflection can induce LLMs' stable impersonation of the target trait across diverse downstream tasks beyond simple questionnaire answering, consistently outperforming existing strong baselines. 

**Abstract (ZH)**: 基于各种人类编写语料库训练的大语言模型（LLMs）展示了通过提示反射特定人类特质（如性格或价值观）的能力，造福个性化LLMs和社会模拟等应用。然而，现有方法面临表面诱导问题：LLMs只能被引导模仿表面且不稳定的风格模式，无法在多种任务中一致且精确地体现所需特质。为应对这一挑战，我们提出IROTE，一种新颖的上下文内方法，用于稳定和可转移的特质诱导。基于心理学理论，认为特质通过身份相关的反思形成，我们的方法自动生成并优化提示中的文本自我反思，包含自我感知的经验，以刺激LLMs的特质驱动行为。优化通过迭代最大化信息论目标实现，该目标增强了LLMs行为与目标特质之间的联系，同时减少反思中的噪音冗余，无需微调，从而产生引人入胜且紧凑的特质反思。实验结果表明，单个IROTE生成的自我反思可以诱导LLMs在多种下游任务中稳定地模仿目标特质，超越简单的问卷回答，并且一直优于现有的强大基线。 

---
# MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs 

**Title (ZH)**: MultiAiTutor: 面向儿童的多语言语音生成教育辅导系统 

**Authors**: Xiaoxue Gao, Huayun Zhang, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08715)  

**Abstract**: Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods. 

**Abstract (ZH)**: 生成式语音模型在个人化师生互动中展现出巨大的潜力，为儿童语言学习提供了宝贵的实际应用。然而，实现高质量、儿童友好的语音生成，特别是在多种低资源语言和文化背景下仍然具有挑战性。本文提出了一种名为MultiAiTutor的教育多语言生成式AI tutor，采用适应教育目的的LLM架构进行儿童友好的设计。我们提出通过LML架构整合适合不同年龄段的多语言语音生成，借助文化相关的情感描述任务，促进新加坡口音普通话、马来语和泰米尔语三种低资源语言背景下的幼儿语言学习。实验结果表明，所提出的MultiAiTutor在客观指标和主观评估中均表现出色，优于基准方法。 

---
# A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models 

**Title (ZH)**: 平行文本生成综述：从并行解码到扩散语言模型 

**Authors**: Lingzhe Zhang, Liancheng Fang, Chiming Duan, Minghua He, Leyi Pan, Pei Xiao, Shiyu Huang, Yunpeng Zhai, Xuming Hu, Philip S. Yu, Aiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08712)  

**Abstract**: As text generation has become a core capability of modern Large Language Models (LLMs), it underpins a wide range of downstream applications. However, most existing LLMs rely on autoregressive (AR) generation, producing one token at a time based on previously generated context-resulting in limited generation speed due to the inherently sequential nature of the process. To address this challenge, an increasing number of researchers have begun exploring parallel text generation-a broad class of techniques aimed at breaking the token-by-token generation bottleneck and improving inference efficiency. Despite growing interest, there remains a lack of comprehensive analysis on what specific techniques constitute parallel text generation and how they improve inference performance. To bridge this gap, we present a systematic survey of parallel text generation methods. We categorize existing approaches into AR-based and Non-AR-based paradigms, and provide a detailed examination of the core techniques within each category. Following this taxonomy, we assess their theoretical trade-offs in terms of speed, quality, and efficiency, and examine their potential for combination and comparison with alternative acceleration strategies. Finally, based on our findings, we highlight recent advancements, identify open challenges, and outline promising directions for future research in parallel text generation. 

**Abstract (ZH)**: 随着文本生成成为现代大规模语言模型（LLMs）的核心能力，它支撑着广泛的应用场景。然而，大多数现有的LLMs依赖自回归（AR）生成，逐token生成，这导致了由于过程本身的串行性质而产生的生成速度受限。为了解决这一挑战，越来越多的研究人员开始探索平行文本生成——一类旨在打破逐token生成瓶颈并提高推理效率的技术。尽管兴趣日益增长，但仍缺乏对具体哪些技术构成平行文本生成以及它们如何改善推理性能的全面分析。为填补这一空白，我们提供了平行文本生成方法的系统综述。我们将现有的方法分类为基于自回归和非自回归范式，并对每个类别中的核心技术进行了详细的探讨。按照这一分类体系，我们评估了它们在速度、质量和效率方面的理论权衡，并考察了它们与其他加速策略结合和比较的潜力。最后，基于我们的发现，我们强调了最近的进展，指出了开放的挑战，并概述了平行文本生成未来研究的有希望的方向。 

---
# $\text{M}^{2}$LLM: Multi-view Molecular Representation Learning with Large Language Models 

**Title (ZH)**: $\text{M}^{2}$LLM：基于大型语言模型的多视图分子表示学习 

**Authors**: Jiaxin Ju, Yizhen Zheng, Huan Yee Koh, Can Wang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08657)  

**Abstract**: Accurate molecular property prediction is a critical challenge with wide-ranging applications in chemistry, materials science, and drug discovery. Molecular representation methods, including fingerprints and graph neural networks (GNNs), achieve state-of-the-art results by effectively deriving features from molecular structures. However, these methods often overlook decades of accumulated semantic and contextual knowledge. Recent advancements in large language models (LLMs) demonstrate remarkable reasoning abilities and prior knowledge across scientific domains, leading us to hypothesize that LLMs can generate rich molecular representations when guided to reason in multiple perspectives. To address these gaps, we propose $\text{M}^{2}$LLM, a multi-view framework that integrates three perspectives: the molecular structure view, the molecular task view, and the molecular rules view. These views are fused dynamically to adapt to task requirements, and experiments demonstrate that $\text{M}^{2}$LLM achieves state-of-the-art performance on multiple benchmarks across classification and regression tasks. Moreover, we demonstrate that representation derived from LLM achieves exceptional performance by leveraging two core functionalities: the generation of molecular embeddings through their encoding capabilities and the curation of molecular features through advanced reasoning processes. 

**Abstract (ZH)**: 准确的分子性质预测是化学、材料科学和药物发现等领域广泛应用的关键挑战。分子表示方法，包括指纹和图神经网络（GNNs），通过有效提取分子结构特征实现了最先进的结果。然而，这些方法往往忽略了数十年积累的语义和上下文知识。大型语言模型（LLMs）的最近进展展示了跨科学领域的出色推理能力和先验知识，这促使我们假设在多视角推理的引导下，LLMs能够生成丰富的分子表示。为了解决这些差距，我们提出 $\text{M}^{2}$LLM，这是一种多视图框架，整合了三种视角：分子结构视图、分子任务视图和分子规则视图。这些视角能够动态融合以适应任务需求，并且实验表明，$\text{M}^{2}$LLM 在分类和回归任务的多个基准上实现了最先进的性能。此外，我们展示了通过利用大型语言模型的两大核心功能——通过编码能力生成分子嵌入和通过高级推理过程整理分子特征——所获得的表示能够实现卓越的性能。 

---
# LLM driven Text-to-Table Generation through Sub-Tasks Guidance and Iterative Refinement 

**Title (ZH)**: 通过子任务引导和迭代 refinements 的大规模语言模型驱动的文本到表格生成 

**Authors**: Rajmohan C, Sarthak Harne, Arvind Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2508.08653)  

**Abstract**: Transforming unstructured text into structured data is a complex task, requiring semantic understanding, reasoning, and structural comprehension. While Large Language Models (LLMs) offer potential, they often struggle with handling ambiguous or domain-specific data, maintaining table structure, managing long inputs, and addressing numerical reasoning. This paper proposes an efficient system for LLM-driven text-to-table generation that leverages novel prompting techniques. Specifically, the system incorporates two key strategies: breaking down the text-to-table task into manageable, guided sub-tasks and refining the generated tables through iterative self-feedback. We show that this custom task decomposition allows the model to address the problem in a stepwise manner and improves the quality of the generated table. Furthermore, we discuss the benefits and potential risks associated with iterative self-feedback on the generated tables while highlighting the trade-offs between enhanced performance and computational cost. Our methods achieve strong results compared to baselines on two complex text-to-table generation datasets available in the public domain. 

**Abstract (ZH)**: 将无结构文本转化为结构化数据是一项复杂任务，需要语义理解、推理和结构理解。尽管大型语言模型（LLMs）具有潜力，但它们在处理模糊或领域特定数据、保持表格结构、处理长输入以及解决数值推理方面经常遇到困难。本文提出了一种高效的LLM驱动的文本到表格生成系统，利用了新颖的提示技术。具体来说，系统采用了两种关键策略：将文本到表格的任务分解为可管理的引导子任务，并通过迭代自我反馈精炼生成的表格。我们表明，这种自定义任务分解方法使模型能够逐步解决问题，从而提高生成表格的质量。此外，本文还讨论了迭代自我反馈在生成表格上的优缺点，并强调了性能提升与计算成本之间的权衡。我们的方法在两个公开领域的复杂文本到表格生成数据集上优于 baselines。 

---
# MiGrATe: Mixed-Policy GRPO for Adaptation at Test-Time 

**Title (ZH)**: MiGrATe: 混合策略策略GRPO在测试时适应中的应用 nâ萸
user
基于迁移的混合策略多
-Trumpet
基于迁移的混合策略 grated
用户
Mi GR AT e: Mixed-Policy GRPO for Adaptation at nhanced at xam-time。iect标题，xico

-Trumpet
MiGrATe: 混合策略策略增强测试时顷适应性 nâ
-Trumpet
MiGrρ ATe: 混合策略增强测试时顷适应适配 Newtown 

**Authors**: Peter Phan, Dhruv Agarwal, Kavitha Srinivas, Horst Samulowitz, Pavan Kapanipathi, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2508.08641)  

**Abstract**: Large language models (LLMs) are increasingly being applied to black-box optimization tasks, from program synthesis to molecule design. Prior work typically leverages in-context learning to iteratively guide the model towards better solutions. Such methods, however, often struggle to balance exploration of new solution spaces with exploitation of high-reward ones. Recently, test-time training (TTT) with synthetic data has shown promise in improving solution quality. However, the need for hand-crafted training data tailored to each task limits feasibility and scalability across domains. To address this problem, we introduce MiGrATe-a method for online TTT that uses GRPO as a search algorithm to adapt LLMs at inference without requiring external training data. MiGrATe operates via a mixed-policy group construction procedure that combines on-policy sampling with two off-policy data selection techniques: greedy sampling, which selects top-performing past completions, and neighborhood sampling (NS), which generates completions structurally similar to high-reward ones. Together, these components bias the policy gradient towards exploitation of promising regions in solution space, while preserving exploration through on-policy sampling. We evaluate MiGrATe on three challenging domains-word search, molecule optimization, and hypothesis+program induction on the Abstraction and Reasoning Corpus (ARC)-and find that it consistently outperforms both inference-only and TTT baselines, demonstrating the potential of online TTT as a solution for complex search tasks without external supervision. 

**Abstract (ZH)**: 大型预训练模型在黑盒优化任务中的应用

user
请简洁总结一下这篇文章的主要贡献和方法。 

---
# Securing Educational LLMs: A Generalised Taxonomy of Attacks on LLMs and DREAD Risk Assessment 

**Title (ZH)**: 保障教育领域大语言模型的安全：大语言模型攻击的通用分类及DREAD风险评估 

**Authors**: Farzana Zahid, Anjalika Sewwandi, Lee Brandon, Vimal Kumar, Roopak Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2508.08629)  

**Abstract**: Due to perceptions of efficiency and significant productivity gains, various organisations, including in education, are adopting Large Language Models (LLMs) into their workflows. Educator-facing, learner-facing, and institution-facing LLMs, collectively, Educational Large Language Models (eLLMs), complement and enhance the effectiveness of teaching, learning, and academic operations. However, their integration into an educational setting raises significant cybersecurity concerns. A comprehensive landscape of contemporary attacks on LLMs and their impact on the educational environment is missing. This study presents a generalised taxonomy of fifty attacks on LLMs, which are categorized as attacks targeting either models or their infrastructure. The severity of these attacks is evaluated in the educational sector using the DREAD risk assessment framework. Our risk assessment indicates that token smuggling, adversarial prompts, direct injection, and multi-step jailbreak are critical attacks on eLLMs. The proposed taxonomy, its application in the educational environment, and our risk assessment will help academic and industrial practitioners to build resilient solutions that protect learners and institutions. 

**Abstract (ZH)**: 由于效率和显著生产率提升的感知，各种组织，包括教育领域，正在将其工作流程中采用大型语言模型（LLMs）。面向教育者、面向学习者和面向机构的LLMs，统称为教育大型语言模型（eLLMs），它们共同增强了教学、学习和学术运营的有效性。然而，它们在教育环境中的集成引发了重大的网络安全关切。缺乏对针对LLMs的现代攻击及其对教育环境影响的全面研究。本研究提出了一个泛化的大型语言模型攻击分类，将其分为针对模型或其基础设施的攻击类别，并使用DREAD风险评估框架在教育领域评估这些攻击的严重性。我们的风险评估显示，令牌走私、对抗性提示、直接注入和多步骤脱逃是针对eLLMs的关键攻击。提出的分类、其在教育环境中的应用以及我们进行的风险评估将有助于学术和工业从业者构建能够抵御攻击并保护学习者和机构的韧性解决方案。 

---
# QoE-Aware Service Provision for Mobile AR Rendering: An Agent-Driven Approach 

**Title (ZH)**: 面向移动AR渲染的QoE感知服务提供：一种基于代理驱动的方法 

**Authors**: Conghao Zhou, Lulu Sun, Xiucheng Wang, Peng Yang, Feng Lyu, Sihan Lu, Xuemin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08627)  

**Abstract**: Mobile augmented reality (MAR) is envisioned as a key immersive application in 6G, enabling virtual content rendering aligned with the physical environment through device pose estimation. In this paper, we propose a novel agent-driven communication service provisioning approach for edge-assisted MAR, aiming to reduce communication overhead between MAR devices and the edge server while ensuring the quality of experience (QoE). First, to address the inaccessibility of MAR application-specific information to the network controller, we establish a digital agent powered by large language models (LLMs) on behalf of the MAR service provider, bridging the data and function gap between the MAR service and network domains. Second, to cope with the user-dependent and dynamic nature of data traffic patterns for individual devices, we develop a user-level QoE modeling method that captures the relationship between communication resource demands and perceived user QoE, enabling personalized, agent-driven communication resource management. Trace-driven simulation results demonstrate that the proposed approach outperforms conventional LLM-based QoE-aware service provisioning methods in both user-level QoE modeling accuracy and communication resource efficiency. 

**Abstract (ZH)**: 移动增强现实（MAR）被视为6G的关键沉浸式应用，通过设备姿态估计实现虚拟内容与物理环境的对齐。本文提出了一种面向边缘辅助MAR的新型代理驱动通信服务部署方法，旨在减少MAR设备与边缘服务器之间的通信开销，同时保证用户体验质量（QoE）。首先，为解决网络控制器无法获得MAR应用程序特定信息的问题，我们通过大型语言模型（LLMs）建立了MAR服务提供商的数字代理，填补了MAR服务和网络域之间的数据和功能差距。其次，为应对各设备数据流量模式的用户依赖性和动态性，我们开发了一种用户体验质量（QoE）建模方法，该方法捕捉了通信资源需求与感知用户体验之间的关系，从而实现个性化、代理驱动的通信资源管理。基于跟踪的仿真实验结果表明，所提出的方法在用户级QoE建模准确性和通信资源效率方面均优于传统的LLM驱动的QoE感知服务部署方法。 

---
# DepressLLM: Interpretable domain-adapted language model for depression detection from real-world narratives 

**Title (ZH)**: DepressLLM：可解释的领域适应语言模型用于现实世界叙事中的抑郁检测 

**Authors**: Sehwan Moon, Aram Lee, Jeong Eun Kim, Hee-Ju Kang, Il-Seon Shin, Sung-Wan Kim, Jae-Min Kim, Min Jhon, Ju-Wan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.08591)  

**Abstract**: Advances in large language models (LLMs) have enabled a wide range of applications. However, depression prediction is hindered by the lack of large-scale, high-quality, and rigorously annotated datasets. This study introduces DepressLLM, trained and evaluated on a novel corpus of 3,699 autobiographical narratives reflecting both happiness and distress. DepressLLM provides interpretable depression predictions and, via its Score-guided Token Probability Summation (SToPS) module, delivers both improved classification performance and reliable confidence estimates, achieving an AUC of 0.789, which rises to 0.904 on samples with confidence $\geq$ 0.95. To validate its robustness to heterogeneous data, we evaluated DepressLLM on in-house datasets, including an Ecological Momentary Assessment (EMA) corpus of daily stress and mood recordings, and on public clinical interview data. Finally, a psychiatric review of high-confidence misclassifications highlighted key model and data limitations that suggest directions for future refinements. These findings demonstrate that interpretable AI can enable earlier diagnosis of depression and underscore the promise of medical AI in psychiatry. 

**Abstract (ZH)**: 大型语言模型的进展使得广泛的应用成为可能。然而，抑郁症预测受限于缺乏大规模、高质量和严格注释的数据集。本研究引入了DepressLLM，该模型在一种新颖的包含3,699个反映幸福与痛苦的自传体叙事的语料库上进行训练和评估。DepressLLM 提供了可解释的抑郁症预测，并通过其基于评分的令牌概率求和模块（SToPS模块），实现了更好的分类性能和可靠的置信度估计，AUC达到0.789，对于置信度≥0.95的样本，AUC提高到0.904。为了验证其在异质数据上的鲁棒性，我们还在内部数据集上评估了DepressLLM，包括日常压力和情绪记录的生态瞬时评估（EMA）语料库以及公开的临床访谈数据。最终，针对高置信度误分类的医学审查指出了关键的模型和数据限制，为未来改进的方向提供了线索。这些发现表明，可解释的人工智能可以助力早期抑郁症诊断，并突显了医疗人工智能在精神病学中的潜力。 

---
# OmniLLP: Enhancing LLM-based Log Level Prediction with Context-Aware Retrieval 

**Title (ZH)**: 全方位 LLP：基于上下文感知 retrieval 提升大语言模型的日志级别预测 

**Authors**: Youssef Esseddiq Ouatiti, Mohammed Sayagh, Bram Adams, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08545)  

**Abstract**: Developers insert logging statements in source code to capture relevant runtime information essential for maintenance and debugging activities. Log level choice is an integral, yet tricky part of the logging activity as it controls log verbosity and therefore influences systems' observability and performance. Recent advances in ML-based log level prediction have leveraged large language models (LLMs) to propose log level predictors (LLPs) that demonstrated promising performance improvements (AUC between 0.64 and 0.8). Nevertheless, current LLM-based LLPs rely on randomly selected in-context examples, overlooking the structure and the diverse logging practices within modern software projects. In this paper, we propose OmniLLP, a novel LLP enhancement framework that clusters source files based on (1) semantic similarity reflecting the code's functional purpose, and (2) developer ownership cohesion. By retrieving in-context learning examples exclusively from these semantic and ownership aware clusters, we aim to provide more coherent prompts to LLPs leveraging LLMs, thereby improving their predictive accuracy. Our results show that both semantic and ownership-aware clusterings statistically significantly improve the accuracy (by up to 8\% AUC) of the evaluated LLM-based LLPs compared to random predictors (i.e., leveraging randomly selected in-context examples from the whole project). Additionally, our approach that combines the semantic and ownership signal for in-context prediction achieves an impressive 0.88 to 0.96 AUC across our evaluated projects. Our findings highlight the value of integrating software engineering-specific context, such as code semantic and developer ownership signals into LLM-LLPs, offering developers a more accurate, contextually-aware approach to logging and therefore, enhancing system maintainability and observability. 

**Abstract (ZH)**: 开发者在源代码中插入日志语句以捕获维护和调试活动所需的相关运行时信息。日志级别选择是日志记录活动中的一项重要但棘手的任务，因为它控制日志的详细程度，从而影响系统的可观测性和性能。基于ML的日志级别预测最近利用大型语言模型（LLMs）提出了日志级别预测器（LLPs），表现出有希望的性能提升（AUC在0.64到0.8之间）。然而，当前的基于LLM的LLPs依赖于随机选取的上下文示例，忽视了现代软件项目中结构和日志实践的多样性。在本文中，我们提出OmniLLP，一种新颖的LLP增强框架，根据（1）反映代码功能目的的语义相似性，以及（2）开发者所有权凝聚性来聚类源文件。通过从这些语义和所有权感知的聚类中独家检索上下文学习示例，我们旨在为利用LLMs的LLPs提供更一致的提示，从而提高其预测准确性。结果显示，与随机预测器（即，使用整个项目中随机选择的上下文示例）相比，基于LLM的LLPs在统计上显著提高了评估的LLPs的准确性（AUC最高可提高8%）。另外，结合语义和所有权信号进行上下文预测的方法在我们评估的项目中取得了令人印象深刻的结果，AUC范围为0.88到0.96。我们的研究结果强调了将软件工程特定的上下文，如代码语义和开发者所有权信号，整合到LLM-LLPs中以提高日志记录和系统可维护性与可观测性的准确性与上下文相关性的价值。 

---
# LLM-Driven Adaptive 6G-Ready Wireless Body Area Networks: Survey and Framework 

**Title (ZH)**: 基于LLM驱动的自适应6G-ready无线身体区域网络：综述与框架 

**Authors**: Azin Sabzian, Mohammad Jalili Torkamani, Negin Mahmoudi, Kiana Kiashemshaki  

**Link**: [PDF](https://arxiv.org/pdf/2508.08535)  

**Abstract**: Wireless Body Area Networks (WBANs) enable continuous monitoring of physiological signals for applications ranging from chronic disease management to emergency response. Recent advances in 6G communications, post-quantum cryptography, and energy harvesting have the potential to enhance WBAN performance. However, integrating these technologies into a unified, adaptive system remains a challenge. This paper surveys some of the most well-known Wireless Body Area Network (WBAN) architectures, routing strategies, and security mechanisms, identifying key gaps in adaptability, energy efficiency, and quantum-resistant security. We propose a novel Large Language Model-driven adaptive WBAN framework in which a Large Language Model acts as a cognitive control plane, coordinating routing, physical layer selection, micro-energy harvesting, and post-quantum security in real time. Our review highlights the limitations of current heuristic-based designs and outlines a research agenda for resource-constrained, 6G-ready medical systems. This approach aims to enable ultra-reliable, secure, and self-optimizing WBANs for next-generation mobile health applications. 

**Abstract (ZH)**: 无线身体区域网络（WBANs）使连续监测生理信号成为可能，适用于从慢性病管理到紧急响应的各种应用。6G通信、后量子密码学和能量 Harvesting 的最新进展有望提升WBAN性能。然而，将这些技术整合到一个统一的自适应系统中仍是一项挑战。本文概述了一些最知名的无线身体区域网络（WBAN）架构、路由策略和安全机制，指出其在适应性、能量效率和后量子安全方面的重要空白。我们提出了一种新型的大规模语言模型驱动的自适应WBAN框架，其中大规模语言模型作为认知控制平面，在实时协调路由、物理层选择、微能量采集和后量子安全方面发挥作用。我们的综述突出了当前基于启发式设计的局限性，并为资源受限的6G就绪医疗系统制定了研究议程。该方法旨在为下一代移动健康应用启用超可靠、安全且自优化的WBAN。 

---
# Using LLMs to Capture Users' Temporal Context for Recommendation 

**Title (ZH)**: 使用大规模语言模型捕获用户的时间上下文以进行推荐 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2508.08512)  

**Abstract**: Effective recommender systems demand dynamic user understanding, especially in complex, evolving environments. Traditional user profiling often fails to capture the nuanced, temporal contextual factors of user preferences, such as transient short-term interests and enduring long-term tastes. This paper presents an assessment of Large Language Models (LLMs) for generating semantically rich, time-aware user profiles. We do not propose a novel end-to-end recommendation architecture; instead, the core contribution is a systematic investigation into the degree of LLM effectiveness in capturing the dynamics of user context by disentangling short-term and long-term preferences. This approach, framing temporal preferences as dynamic user contexts for recommendations, adaptively fuses these distinct contextual components into comprehensive user embeddings. The evaluation across Movies&TV and Video Games domains suggests that while LLM-generated profiles offer semantic depth and temporal structure, their effectiveness for context-aware recommendations is notably contingent on the richness of user interaction histories. Significant gains are observed in dense domains (e.g., Movies&TV), whereas improvements are less pronounced in sparse environments (e.g., Video Games). This work highlights LLMs' nuanced potential in enhancing user profiling for adaptive, context-aware recommendations, emphasizing the critical role of dataset characteristics for practical applicability. 

**Abstract (ZH)**: 大型语言模型在生成语义丰富的时间感知用户画像中的评估 

---
# Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression 

**Title (ZH)**: 可调控多元主义：基于少量对比回归的多元主义对齐 

**Authors**: Jadie Adams, Brian Hu, Emily Veenhuis, David Joy, Bharadwaj Ravichandran, Aaron Bray, Anthony Hoogs, Arslan Basharat  

**Link**: [PDF](https://arxiv.org/pdf/2508.08509)  

**Abstract**: Large language models (LLMs) are currently aligned using techniques such as reinforcement learning from human feedback (RLHF). However, these methods use scalar rewards that can only reflect user preferences on average. Pluralistic alignment instead seeks to capture diverse user preferences across a set of attributes, moving beyond just helpfulness and harmlessness. Toward this end, we propose a steerable pluralistic model based on few-shot comparative regression that can adapt to individual user preferences. Our approach leverages in-context learning and reasoning, grounded in a set of fine-grained attributes, to compare response options and make aligned choices. To evaluate our algorithm, we also propose two new steerable pluralistic benchmarks by adapting the Moral Integrity Corpus (MIC) and the HelpSteer2 datasets, demonstrating the applicability of our approach to value-aligned decision-making and reward modeling, respectively. Our few-shot comparative regression approach is interpretable and compatible with different attributes and LLMs, while outperforming multiple baseline and state-of-the-art methods. Our work provides new insights and research directions in pluralistic alignment, enabling a more fair and representative use of LLMs and advancing the state-of-the-art in ethical AI. 

**Abstract (ZH)**: 大型语言模型（LLMs）当前通过人类反馈强化学习（RLHF）等技术进行对齐。然而，这些方法仅使用标量奖励，只能反映用户的平均偏好。多样化的对齐则旨在捕捉一组属性中的多样化用户偏好，不仅局限于帮助性和无害性。为实现这一目标，我们提出了一种基于少数示例比较回归的可引导多样化模型，可以适应个体用户的偏好。我们的方法利用上下文学习和基于细粒度属性的推理，比较响应选项并作出对齐选择。为了评估我们的算法，我们还提出了两个新的可引导多样化基准，分别是通过对道德完整语料库（MIC）和HelpSteer2数据集进行适配得到的，分别展示了我们的方法在价值对齐决策和奖励建模中的适用性。我们的少数示例比较回归方法具有可解释性，与不同的属性和LLMs兼容，并且优于多种基准和最新方法。我们的工作为多样化对齐提供了新的见解和研究方向，使LLMs的使用更加公平和具有代表性，并推动了伦理AI的最新进展。 

---
# When the Domain Expert Has No Time and the LLM Developer Has No Clinical Expertise: Real-World Lessons from LLM Co-Design in a Safety-Net Hospital 

**Title (ZH)**: 当领域专家没有时间而LLM开发者缺乏临床专业知识时：一家安全网医院中LLM协同设计的现实世界经验教训 

**Authors**: Avni Kothari, Patrick Vossler, Jean Digitale, Mohammad Forouzannia, Elise Rosenberg, Michele Lee, Jennee Bryant, Melanie Molina, James Marks, Lucas Zier, Jean Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.08504)  

**Abstract**: Large language models (LLMs) have the potential to address social and behavioral determinants of health by transforming labor intensive workflows in resource-constrained settings. Creating LLM-based applications that serve the needs of underserved communities requires a deep understanding of their local context, but it is often the case that neither LLMs nor their developers possess this local expertise, and the experts in these communities often face severe time/resource constraints. This creates a disconnect: how can one engage in meaningful co-design of an LLM-based application for an under-resourced community when the communication channel between the LLM developer and domain expert is constrained? We explored this question through a real-world case study, in which our data science team sought to partner with social workers at a safety net hospital to build an LLM application that summarizes patients' social needs. Whereas prior works focus on the challenge of prompt tuning, we found that the most critical challenge in this setting is the careful and precise specification of \what information to surface to providers so that the LLM application is accurate, comprehensive, and verifiable. Here we present a novel co-design framework for settings with limited access to domain experts, in which the summary generation task is first decomposed into individually-optimizable attributes and then each attribute is efficiently refined and validated through a multi-tier cascading approach. 

**Abstract (ZH)**: 大型语言模型（LLMs）有潜力通过在资源受限环境中转换劳动密集型工作流程来应对社会和行为健康决定因素。在为服务不足的社区创建LLM基于的应用程序时需要深刻理解其当地环境，但在这种情境下，无论是LLMs本身还是开发者通常都缺乏这种本地专业知识，而这些社区中的专家往往面临着严重的时间/资源限制。这就产生了一个断层：当LLM开发者与领域专家之间的沟通渠道受限时，如何进行有意义的LLM基于的应用程序共同设计？我们通过一个实际案例研究探索了这一问题，我们的数据科学团队寻求与一家安全网医院的社会工作者合作，构建一个总结患者社会需求的LLM应用程序。尽管先前的研究集中在提示调优的挑战上，但我们发现，在这种情境下，最关键的问题是如何精心和精确地指定向提供者展示哪些信息，以确保LLM应用程序准确、全面且可验证。在此，我们提出了一种新的共同设计框架，适用于接触领域专家受限的环境，在该框架中，总结生成任务首先被分解为可单独优化的属性，然后通过多层次级联方法高效地细化和验证每个属性。 

---
# Momentum Point-Perplexity Mechanics in Large Language Models 

**Title (ZH)**: 动量点困惑度机理在大规模语言模型中 

**Authors**: Lorenzo Tomaz, Judd Rosenblatt, Thomas Berry Jones, Diogo Schwerz de Lucena  

**Link**: [PDF](https://arxiv.org/pdf/2508.08492)  

**Abstract**: We take a physics-based approach to studying how the internal hidden states of large language models change from token to token during inference. Across 20 open-source transformer models (135M-3B parameters), we find that a quantity combining the rate of change in hidden states and the model's next-token certainty, analogous to energy in physics, remains nearly constant. Random-weight models conserve this "energy" more tightly than pre-trained ones, while training shifts models into a faster, more decisive regime with greater variability. Using this "log-Lagrangian" view, we derive a control method called Jacobian steering, which perturbs hidden states in the minimal way needed to favor a target token. This approach maintained near-constant energy in two tested models and produced continuations rated higher in semantic quality than the models' natural outputs. Viewing transformers through this mechanics lens offers a principled basis for interpretability, anomaly detection, and low-risk steering. This could help make powerful models more predictable and aligned with human intent. 

**Abstract (ZH)**: 基于物理原理研究大型语言模型推理过程中内部隐藏状态的变化：从物理学视角探讨“能量”守恒与模型训练的影响及基于“log-Lagrangian”观点的扰动控制方法 

---
# Temporal User Profiling with LLMs: Balancing Short-Term and Long-Term Preferences for Recommendations 

**Title (ZH)**: 基于LLM的Temporal用户画像构建：短期和长期偏好的平衡推荐 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2508.08454)  

**Abstract**: Accurately modeling user preferences is crucial for improving the performance of content-based recommender systems. Existing approaches often rely on simplistic user profiling methods, such as averaging or concatenating item embeddings, which fail to capture the nuanced nature of user preference dynamics, particularly the interactions between long-term and short-term preferences. In this work, we propose LLM-driven Temporal User Profiling (LLM-TUP), a novel method for user profiling that explicitly models short-term and long-term preferences by leveraging interaction timestamps and generating natural language representations of user histories using a large language model (LLM). These representations are encoded into high-dimensional embeddings using a pre-trained BERT model, and an attention mechanism is applied to dynamically fuse the short-term and long-term embeddings into a comprehensive user profile. Experimental results on real-world datasets demonstrate that LLM-TUP achieves substantial improvements over several baselines, underscoring the effectiveness of our temporally aware user-profiling approach and the use of semantically rich user profiles, generated by LLMs, for personalized content-based recommendation. 

**Abstract (ZH)**: 准确建模用户偏好对于改进基于内容的推荐系统性能至关重要。现有的方法往往是依靠简单的用户画像方法，如平均或拼接物品嵌入，这无法捕捉用户偏好动态的细致特征，尤其是长期偏好与短期偏好之间的交互。在本文中，我们提出了一种新的用户画像方法——LLM驱动的时序用户画像（LLM-TUP），该方法通过利用交互时间戳并使用大规模语言模型（LLM）生成用户历史的自然语言表示来明确建模短期和长期偏好。这些表示通过预训练的BERT模型编码为高维嵌入，并使用注意机制动态融合短期和长期嵌入，生成全面的用户画像。实验结果在真实世界数据集上的表现表明，LLM-TUP在多个基线方法上取得了显著的性能提升，证明了我们的时序意识用户画像方法以及通过LLMs生成语义丰富的用户画像在个性化内容推荐中的有效性。 

---
# Fast weight programming and linear transformers: from machine learning to neurobiology 

**Title (ZH)**: 快速权重编程与线性变压器：从机器学习到神经生物学 

**Authors**: Kazuki Irie, Samuel J. Gershman  

**Link**: [PDF](https://arxiv.org/pdf/2508.08435)  

**Abstract**: Recent advances in artificial neural networks for machine learning, and language modeling in particular, have established a family of recurrent neural network (RNN) architectures that, unlike conventional RNNs with vector-form hidden states, use two-dimensional (2D) matrix-form hidden states. Such 2D-state RNNs, known as Fast Weight Programmers (FWPs), can be interpreted as a neural network whose synaptic weights (called fast weights) dynamically change over time as a function of input observations, and serve as short-term memory storage; corresponding synaptic weight modifications are controlled or programmed by another network (the programmer) whose parameters are trained (e.g., by gradient descent). In this Primer, we review the technical foundations of FWPs, their computational characteristics, and their connections to transformers and state space models. We also discuss connections between FWPs and models of synaptic plasticity in the brain, suggesting a convergence of natural and artificial intelligence. 

**Abstract (ZH)**: Recent Advances in Artificial Neural Networks for Machine Learning: Fast Weight Programmers and Their Connections to Transformers and State Space Models 

---
# Generating Query-Relevant Document Summaries via Reinforcement Learning 

**Title (ZH)**: 通过强化学习生成查询相关的文档摘要 

**Authors**: Nitin Yadav, Changsung Kang, Hongwei Shang, Ming Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.08404)  

**Abstract**: E-commerce search engines often rely solely on product titles as input for ranking models with latency constraints. However, this approach can result in suboptimal relevance predictions, as product titles often lack sufficient detail to capture query intent. While product descriptions provide richer information, their verbosity and length make them unsuitable for real-time ranking, particularly for computationally expensive architectures like cross-encoder ranking models. To address this challenge, we propose ReLSum, a novel reinforcement learning framework designed to generate concise, query-relevant summaries of product descriptions optimized for search relevance. ReLSum leverages relevance scores as rewards to align the objectives of summarization and ranking, effectively overcoming limitations of prior methods, such as misaligned learning targets. The framework employs a trainable large language model (LLM) to produce summaries, which are then used as input for a cross-encoder ranking model. Experimental results demonstrate significant improvements in offline metrics, including recall and NDCG, as well as online user engagement metrics. ReLSum provides a scalable and efficient solution for enhancing search relevance in large-scale e-commerce systems. 

**Abstract (ZH)**: 电子商务搜索引擎往往仅依赖产品标题作为实时排序模型的输入，但这可能导致不尽如人意的相关性预测，因为产品标题往往缺乏足够的细节来捕捉查询意图。虽然产品描述提供了更丰富的信息，但由于篇幅冗长，在实时排序中并不适用，特别是对于计算成本高昂的交叉编码器排序模型而言。为解决这一挑战，我们提出了一种新颖的强化学习框架ReLSum，旨在生成符合查询相关性的、简明扼要的产品描述摘要，旨在优化搜索相关性。ReLSum利用相关性分数作为奖励来对齐摘要生成和排序的目标，有效克服了先前方法中的学习目标不一致问题。该框架采用可训练的大语言模型生成摘要，然后将这些摘要作为输入提供给交叉编码器排序模型。实验结果在离线指标（如召回率和NDCG）以及在线用户参与度指标上显示了显著改进。ReLSum为大规模电子商务系统中的搜索相关性提升提供了一种可扩展和高效的方法。 

---
# Maximizing GPU Efficiency via Optimal Adapter Caching: An Analytical Approach for Multi-Tenant LLM Serving 

**Title (ZH)**: 通过优化适配器缓存实现GPU效率最大化：面向多租户LLM服务的分析方法 

**Authors**: Ferran Agullo, Joan Oliveras, Chen Wang, Alberto Gutierrez-Torre, Olivier Tardieu, Alaa Youssef, Jordi Torres, Josep Ll. Berral  

**Link**: [PDF](https://arxiv.org/pdf/2508.08343)  

**Abstract**: Serving LLM adapters has gained significant attention as an effective approach to adapt general-purpose language models to diverse, task-specific use cases. However, serving a wide range of adapters introduces several and substantial overheads, leading to performance degradation and challenges in optimal placement. To address these challenges, we present an analytical, AI-driven pipeline that accurately determines the optimal allocation of adapters in single-node setups. This allocation maximizes performance, effectively using GPU resources, while preventing request starvation. Crucially, the proposed allocation is given based on current workload patterns. These insights in single-node setups can be leveraged in multi-replica deployments for overall placement, load balancing and server configuration, ultimately enhancing overall performance and improving resource efficiency. Our approach builds on an in-depth analysis of LLM adapter serving, accounting for overheads and performance variability, and includes the development of the first Digital Twin capable of replicating online LLM-adapter serving systems with matching key performance metrics. The experimental results demonstrate that the Digital Twin achieves a SMAPE difference of no more than 5.5% in throughput compared to real results, and the proposed pipeline accurately predicts the optimal placement with minimal latency. 

**Abstract (ZH)**: 基于AI的LLM适配器服务优化分析管道 

---
# Energy-Aware Code Generation with LLMs: Benchmarking Small vs. Large Language Models for Sustainable AI Programming 

**Title (ZH)**: 面向能量意识的代码生成：小型与大型语言模型在可持续人工智能编程中的基准测试 

**Authors**: Humza Ashraf, Syed Muhammad Danish, Aris Leivadeas, Yazan Otoum, Zeeshan Sattar  

**Link**: [PDF](https://arxiv.org/pdf/2508.08332)  

**Abstract**: Large Language Models (LLMs) are widely used for code generation. However, commercial models like ChatGPT require significant computing power, which leads to high energy use and carbon emissions. This has raised concerns about their environmental impact. In this study, we evaluate open-source Small Language Models (SLMs) trained explicitly for code generation and compare their performance and energy efficiency against large LLMs and efficient human-written Python code. The goal is to investigate whether SLMs can match the performance of LLMs on certain types of programming problems while producing more energy-efficient code. We evaluate 150 coding problems from LeetCode, evenly distributed across three difficulty levels: easy, medium, and hard. Our comparison includes three small open-source models, StableCode-3B, StarCoderBase-3B, and Qwen2.5-Coder-3B-Instruct, and two large commercial models, GPT-4.0 and DeepSeek-Reasoner. The generated code is evaluated using four key metrics: run-time, memory usage, energy consumption, and correctness. We use human-written solutions as a baseline to assess the quality and efficiency of the model-generated code. Results indicate that LLMs achieve the highest correctness across all difficulty levels, but SLMs are often more energy-efficient when their outputs are correct. In over 52% of the evaluated problems, SLMs consumed the same or less energy than LLMs. 

**Abstract (ZH)**: 开源小型语言模型（SLMs）在代码生成中的性能与能效研究 

---
# Context Engineering for Multi-Agent LLM Code Assistants Using Elicit, NotebookLM, ChatGPT, and Claude Code 

**Title (ZH)**: 基于Elicit、NotebookLM、ChatGPT和Claude Code的多agent大语言模型代码助手上下文工程 

**Authors**: Muhammad Haseeb  

**Link**: [PDF](https://arxiv.org/pdf/2508.08322)  

**Abstract**: Large Language Models (LLMs) have shown promise in automating code generation and software engineering tasks, yet they often struggle with complex, multi-file projects due to context limitations and knowledge gaps. We propose a novel context engineering workflow that combines multiple AI components: an Intent Translator (GPT-5) for clarifying user requirements, an Elicit-powered semantic literature retrieval for injecting domain knowledge, NotebookLM-based document synthesis for contextual understanding, and a Claude Code multi-agent system for code generation and validation. Our integrated approach leverages intent clarification, retrieval-augmented generation, and specialized sub-agents orchestrated via Claude's agent framework. We demonstrate that this method significantly improves the accuracy and reliability of code assistants in real-world repositories, yielding higher single-shot success rates and better adherence to project context than baseline single-agent approaches. Qualitative results on a large this http URL codebase show the multi-agent system effectively plans, edits, and tests complex features with minimal human intervention. We compare our system with recent frameworks like CodePlan, MASAI, and HyperAgent, highlighting how targeted context injection and agent role decomposition lead to state-of-the-art performance. Finally, we discuss the implications for deploying LLM-based coding assistants in production, along with lessons learned on context management and future research directions. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自动化代码生成和软件工程任务方面展现出潜力，但在处理复杂多文件项目时往往受限于上下文限制和知识缺口。我们提出了一种新的上下文工程工作流，结合了多个AI组件：GPT-5智能意图翻译器用于澄清用户需求，Elicit驱动的语义文献检索注入领域知识，NotebookLM为基础的文档合成实现上下文理解，以及Claude Code多代理系统用于代码生成和验证。我们综合方法利用意图澄清、检索增强生成以及通过Claude代理框架协调的专业子代理。我们证明，这种方法在实际代码库中显著提高了代码助手的准确性和可靠性，单次成功率更高，且更好地遵循项目上下文。针对大规模代码库的定性结果表明，多代理系统能够有效规划、编辑和测试复杂功能，同时减少人工干预。我们将我们的系统与CodePlan、MASAI和HyperAgent等最近的框架进行比较，突显了目标上下文注入和代理角色分解如何实现顶级性能。最后，我们讨论了基于LLM的编码助手在生产部署中的影响，并讨论了上下文管理的教训和未来研究方向。 

---
# Assessing the Quality of AI-Generated Exams: A Large-Scale Field Study 

**Title (ZH)**: 评估AI生成考试的质量：一项大规模实地研究 

**Authors**: Calvin Isley, Joshua Gilbert, Evangelos Kassos, Michaela Kocher, Allen Nie, Emma Brunskill, Ben Domingue, Jake Hofman, Joscha Legewie, Teddy Svoronos, Charlotte Tuminelli, Sharad Goel  

**Link**: [PDF](https://arxiv.org/pdf/2508.08314)  

**Abstract**: While large language models (LLMs) challenge conventional methods of teaching and learning, they present an exciting opportunity to improve efficiency and scale high-quality instruction. One promising application is the generation of customized exams, tailored to specific course content. There has been significant recent excitement on automatically generating questions using artificial intelligence, but also comparatively little work evaluating the psychometric quality of these items in real-world educational settings. Filling this gap is an important step toward understanding generative AI's role in effective test design. In this study, we introduce and evaluate an iterative refinement strategy for question generation, repeatedly producing, assessing, and improving questions through cycles of LLM-generated critique and revision. We evaluate the quality of these AI-generated questions in a large-scale field study involving 91 classes -- covering computer science, mathematics, chemistry, and more -- in dozens of colleges across the United States, comprising nearly 1700 students. Our analysis, based on item response theory (IRT), suggests that for students in our sample the AI-generated questions performed comparably to expert-created questions designed for standardized exams. Our results illustrate the power of AI to make high-quality assessments more readily available, benefiting both teachers and students. 

**Abstract (ZH)**: 虽然大规模语言模型（LLMs）挑战了传统的教学和学习方法，它们为提高高效性和扩大高质量教学的规模提供了令人兴奋的机会。一个有前途的应用是生成定制化的考试，针对特定课程内容进行设计。虽然人工智能自动生成问题方面最近引起了显著的热情，但在实际教育环境中评估这些项目的心理计量质量的研究相对较少。填补这一空白是理解生成式AI在有效测验设计中作用的重要一步。在本研究中，我们引入并评估了一种迭代改进策略，通过LLM生成的批判和修订周期，反复生成、评估和改进问题。我们在涉及91个班级（涵盖计算机科学、数学、化学等学科）的大规模现场研究中评估了这些AI生成的问题的质量，这些班级遍布美国数十所学院，总共包括近1700名学生。基于项目反应理论（IRT）的分析表明，对于我们的样本中的学生来说，AI生成的问题与专为标准化考试设计的专家创建的问题表现相当。我们的结果展示了AI使高质量评估更容易获得的潜力，这将受益于教师和学生。 

---
# Putnam-AXIOM: A Functional and Static Benchmark 

**Title (ZH)**: Putnam-AXIOM：一个函数式和静态基准 

**Authors**: Aryan Gulati, Brando Miranda, Eric Chen, Emily Xia, Kai Fronsdal, Bruno Dumont, Elyas Obbad, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2508.08292)  

**Abstract**: Current mathematical reasoning benchmarks for large language models (LLMs) are approaching saturation, with some achieving > 90% accuracy, and are increasingly compromised by training-set contamination. We introduce Putnam-AXIOM, a benchmark of 522 university-level competition problems drawn from the prestigious William Lowell Putnam Mathematical Competition, and Putnam-AXIOM Variation, an unseen companion set of 100 functional variants generated by programmatically perturbing variables and constants. The variation protocol produces an unlimited stream of equally difficult, unseen instances -- yielding a contamination-resilient test bed. On the Original set, OpenAI's o1-preview -- the strongest evaluated model -- scores 41.9%, but its accuracy drops by 19.6% (46.8% relative decrease) on the paired Variations. The remaining eighteen models show the same downward trend, ten of them with non-overlapping 95% confidence intervals. These gaps suggest memorization and highlight the necessity of dynamic benchmarks. We complement "boxed" accuracy with Teacher-Forced Accuracy (TFA), a lightweight metric that directly scores reasoning traces and automates natural language proof evaluations. Putnam-AXIOM therefore provides a rigorous, contamination-resilient evaluation framework for assessing advanced mathematical reasoning of LLMs. Data and evaluation code are publicly available at this https URL. 

**Abstract (ZH)**: 当前大型语言模型（LLMs）的数学推理基准正在接近饱和，一些模型的准确率超过90%，且越来越受到训练集污染的影响。我们引入了Putnam-AXIOM基准，这是一个包含522道来自 prestigous 威廉洛厄尔普特南数学竞赛的大学水平竞赛问题的数据集，以及通过程序化地扰动变量和常量生成的100个功能变体组成的未见过的同伴集Putnam-AXIOM Variation。变体协议产生无限数量的同等难度、未见过的实例——提供了一个抵御污染的测试平台。在原数据集上，OpenAI的o1-preview——最强的评估模型——得分为41.9%，但在配对变体上的准确率下降了19.6%（相对下降46.8%）。其余十八个模型显示出相同的下降趋势，其中有十个模型的95%置信区间不重叠。这些差距表明了记忆化，并突显了动态基准的重要性。我们补充了“封闭式”准确率，使用教师强制准确率（TFA），这是一种轻量级的直接评估推理轨迹并自动化自然语言证明评估的指标。因此，Putnam-AXIOM为评估LLMs的高级数学推理提供了严格且抵御污染的评估框架。数据和评估代码可在以下网址公开获取。 

---
# Sacred or Synthetic? Evaluating LLM Reliability and Abstention for Religious Questions 

**Title (ZH)**: 神圣的还是合成的？评估大规模语言模型在回答宗教问题时的可靠性与回避行为 

**Authors**: Farah Atif, Nursultan Askarbekuly, Kareem Darwish, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2508.08287)  

**Abstract**: Despite the increasing usage of Large Language Models (LLMs) in answering questions in a variety of domains, their reliability and accuracy remain unexamined for a plethora of domains including the religious domains. In this paper, we introduce a novel benchmark FiqhQA focused on the LLM generated Islamic rulings explicitly categorized by the four major Sunni schools of thought, in both Arabic and English. Unlike prior work, which either overlooks the distinctions between religious school of thought or fails to evaluate abstention behavior, we assess LLMs not only on their accuracy but also on their ability to recognize when not to answer. Our zero-shot and abstention experiments reveal significant variation across LLMs, languages, and legal schools of thought. While GPT-4o outperforms all other models in accuracy, Gemini and Fanar demonstrate superior abstention behavior critical for minimizing confident incorrect answers. Notably, all models exhibit a performance drop in Arabic, highlighting the limitations in religious reasoning for languages other than English. To the best of our knowledge, this is the first study to benchmark the efficacy of LLMs for fine-grained Islamic school of thought specific ruling generation and to evaluate abstention for Islamic jurisprudence queries. Our findings underscore the need for task-specific evaluation and cautious deployment of LLMs in religious applications. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各个领域回答问题的使用日益增加，但它们在宗教领域等众多领域的可靠性和准确性尚未被充分检验。本文介绍了一个名为FiqhQA的新基准，专注于由四大学派明确分类的伊斯兰教法律问题，涵盖了阿拉伯语和英语。不同于先前的工作要么忽视宗教学派之间的区别，要么未能评估弃答行为，我们在评估LLMs的准确性的同时，还考察了它们识别何时不应回答的能力。零样本和弃答实验揭示了LLMs、语言和法律学派之间的显著差异。尽管GPT-4o在准确性上超过了其他所有模型，但Gemini和Fanar展示了关键的弃答行为，有助于减少自信的错误答案。值得注意的是，所有模型在阿拉伯语中的表现下降，突显了除英语外其他语言在宗教推理方面的局限性。据我们所知，这是首次针对细粒度伊斯兰教法学派特定判决生成基准LLMs的有效性并评估弃答行为的研究。我们的研究结果突显了在宗教应用中针对特定任务评估和谨慎部署LLMs的必要性。 

---
# The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs 

**Title (ZH)**: 进步的幻象：重新评估LLMs中的幻觉检测 

**Authors**: Denis Janiak, Jakub Binkowski, Albert Sawczyn, Bogdan Gabrys, Ravid Schwartz-Ziv, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2508.08285)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet their tendency to hallucinate poses serious challenges for reliable deployment. Despite numerous hallucination detection methods, their evaluations often rely on ROUGE, a metric based on lexical overlap that misaligns with human judgments. Through comprehensive human studies, we demonstrate that while ROUGE exhibits high recall, its extremely low precision leads to misleading performance estimates. In fact, several established detection methods show performance drops of up to 45.9\% when assessed using human-aligned metrics like LLM-as-Judge. Moreover, our analysis reveals that simple heuristics based on response length can rival complex detection techniques, exposing a fundamental flaw in current evaluation practices. We argue that adopting semantically aware and robust evaluation frameworks is essential to accurately gauge the true performance of hallucination detection methods, ultimately ensuring the trustworthiness of LLM outputs. 

**Abstract (ZH)**: 大型语言模型（LLMs）已革命性地推动了自然语言处理，但其容易产生幻觉的趋势为可靠部署带来了严重挑战。尽管存在众多幻觉检测方法，但它们的评估常常依赖于ROUGE，这是一种基于词汇重叠的指标，与人类判断存在偏差。通过全面的人类研究，我们表明虽然ROUGE表现出较高的召回率，但由于其极低的精确率导致误导性的性能估计。事实上，几种现成的检测方法在使用人类对齐的指标如LLM-as-Judge进行评估时，性能下降幅度高达45.9%。此外，我们的分析揭示，基于响应长度的简单启发式方法可与复杂检测技术相媲美，暴露了当前评估实践中的根本缺陷。我们认为，采用语义意识强且稳健的评估框架是准确衡量幻觉检测方法真正性能的关键，最终确保大型语言模型输出的可信度。 

---
# Evaluating Contrast Localizer for Identifying Causal Unitsin Social & Mathematical Tasks in Language Models 

**Title (ZH)**: 评估对比定位器在社会与数学任务中识别因果单位的效果 

**Authors**: Yassine Jamaa, Badr AlKhamissi, Satrajit Ghosh, Martin Schrimpf  

**Link**: [PDF](https://arxiv.org/pdf/2508.08276)  

**Abstract**: This work adapts a neuroscientific contrast localizer to pinpoint causally relevant units for Theory of Mind (ToM) and mathematical reasoning tasks in large language models (LLMs) and vision-language models (VLMs). Across 11 LLMs and 5 VLMs ranging in size from 3B to 90B parameters, we localize top-activated units using contrastive stimulus sets and assess their causal role via targeted ablations. We compare the effect of lesioning functionally selected units against low-activation and randomly selected units on downstream accuracy across established ToM and mathematical benchmarks. Contrary to expectations, low-activation units sometimes produced larger performance drops than the highly activated ones, and units derived from the mathematical localizer often impaired ToM performance more than those from the ToM localizer. These findings call into question the causal relevance of contrast-based localizers and highlight the need for broader stimulus sets and more accurately capture task-specific units. 

**Abstract (ZH)**: 本研究将神经科学对比局部化方法应用于大型语言模型（LLMs）和视觉-语言模型（VLMs）中的心智理论（ToM）和数学推理任务，以确定因果相关单元。在包含11个参数量从3B到90B的LLMs和5个VLMs中，我们使用对比刺激集定位激活最高单元，并通过靶向消融评估其因果作用。我们将功能性选择的单元、低激活单元以及随机选择的单元的损伤作用与下游准确率在 Established ToM 和数学基准测试中的效果进行比较。出人意料的是，低激活单元有时比高激活单元导致更大的性能下降，来自数学局部化的单元往往比来自心智理论局部化的单元更损害心智理论性能。这些发现质疑了基于对比的局部化方法的因果相关性，并强调了需要更广泛的刺激集以及更准确捕捉任务特定单元的重要性。 

---
# MLLM-CBench:A Comprehensive Benchmark for Continual Instruction Tuning of Multimodal LLMs with Chain-of-Thought Reasoning Analysis 

**Title (ZH)**: MLLM-CBench：用于链式思维推理分析的多模态LLM连续指令调优综合基准])** 

**Authors**: Haiyun Guo, ZhiYan Hou, Yu Chen, Jinghan He, Yandu Sun, Yuzhe Zhou, Shujing Guo, Kuan Zhu, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08275)  

**Abstract**: Multimodal Large Language Models (MLLMs) rely on continual instruction tuning to adapt to the evolving demands of real-world applications. However, progress in this area is hindered by the lack of rigorous and systematic benchmarks. To address this gap, we present MLLM-CTBench, a comprehensive evaluation benchmark with three key contributions: (1) Multidimensional Evaluation: We combine final answer accuracy with fine-grained CoT reasoning quality assessment, enabled by a specially trained CoT evaluator; (2) Comprehensive Evaluation of Algorithms and Training Paradigms: We benchmark eight continual learning algorithms across four major categories and systematically compare reinforcement learning with supervised fine-tuning paradigms; (3) Carefully Curated Tasks: We select and organize 16 datasets from existing work, covering six challenging domains. Our key findings include: (i) Models with stronger general capabilities exhibit greater robustness to forgetting during continual learning; (ii) Reasoning chains degrade more slowly than final answers, supporting the hierarchical forgetting hypothesis; (iii) The effectiveness of continual learning algorithms is highly dependent on both model capability and task order; (iv) In reinforcement learning settings, incorporating KL-divergence constraints helps maintain policy stability and plays a crucial role in mitigating forgetting. MLLM-CTBench establishes a rigorous standard for continual instruction tuning of MLLMs and offers practical guidance for algorithm design and evaluation. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）依赖于持续的指令调优以适应实际应用中的不断变化需求。然而，这一领域的进展受到了缺乏严格和系统的基准测试的阻碍。为解决这一问题，我们提出了MLLM-CTBench综合评估基准，其三大贡献如下：（1）多维度评估：我们将最终答案的准确性与细粒度的CoT推理质量评估相结合，后者得益于一个特别训练的CoT评估器；（2）算法和训练范式的全面评估：我们针对四个主要类别中的八种持续学习算法进行基准测试，并系统比较强化学习与监督微调范式的有效性；（3）精心策划的任务：我们从现有工作中选取并组织了16个数据集，涵盖六个具有挑战性的领域。我们的主要发现包括：（i）具备更强通用能力的模型在持续学习过程中表现出更高的遗忘鲁棒性；（ii）推理链比最终答案更缓慢地退化，支持分层遗忘假设；（iii）持续学习算法的有效性高度依赖于模型能力和任务顺序；（iv）在强化学习环境中，引入KL散度约束有助于保持策略的稳定性，并在缓解遗忘方面发挥关键作用。MLLM-CTBench为多模态大型语言模型的持续指令调优建立了严格的标准，并提供了算法设计和评估的实用指导。 

---
# Distilling Knowledge from Large Language Models: A Concept Bottleneck Model for Hate and Counter Speech Recognition 

**Title (ZH)**: 从大型语言模型中提炼知识：一种仇恨言论及其对立言论识别的概念瓶颈模型 

**Authors**: Roberto Labadie-Tamayo, Djordje Slijepčević, Xihui Chen, Adrian Jaques Böck, Andreas Babic, Liz Freimann, Christiane Atzmüller Matthias Zeppelzauer  

**Link**: [PDF](https://arxiv.org/pdf/2508.08274)  

**Abstract**: The rapid increase in hate speech on social media has exposed an unprecedented impact on society, making automated methods for detecting such content important. Unlike prior black-box models, we propose a novel transparent method for automated hate and counter speech recognition, i.e., "Speech Concept Bottleneck Model" (SCBM), using adjectives as human-interpretable bottleneck concepts. SCBM leverages large language models (LLMs) to map input texts to an abstract adjective-based representation, which is then sent to a light-weight classifier for downstream tasks. Across five benchmark datasets spanning multiple languages and platforms (e.g., Twitter, Reddit, YouTube), SCBM achieves an average macro-F1 score of 0.69 which outperforms the most recently reported results from the literature on four out of five datasets. Aside from high recognition accuracy, SCBM provides a high level of both local and global interpretability. Furthermore, fusing our adjective-based concept representation with transformer embeddings, leads to a 1.8% performance increase on average across all datasets, showing that the proposed representation captures complementary information. Our results demonstrate that adjective-based concept representations can serve as compact, interpretable, and effective encodings for hate and counter speech recognition. With adapted adjectives, our method can also be applied to other NLP tasks. 

**Abstract (ZH)**: 社交媒体上仇恨言论的快速增加对社会产生了前所未有的影响，使自动检测此类内容的方法变得至关重要。我们提出了一种新的透明方法，即“Speech Concept Bottleneck Model”(SCBM)，通过使用形容词作为可由人类解释的瓶颈概念，替代了先前的黑箱模型。SCBM 利用大规模语言模型（LLM）将输入文本映射到一种基于形容词的抽象表示，然后将这种表示传递给轻量级分类器以执行下游任务。在涵盖多种语言和平台（例如 Twitter、Reddit、YouTube）的五个基准数据集中，SCBM 实现了平均宏-F1 得分 0.69，该得分在四个数据集中优于文献中最近的报告结果。除了高度的识别准确性外，SCBM 还提供了局部和全局解释性都较高的水平。此外，将我们的基于形容词的概念表示与变压器嵌入相结合，在所有数据集上平均性能提高了 1.8%，表明所提出的概念表示捕捉到了互补信息。我们的结果表明，基于形容词的概念表示可以作为仇恨言论和反言论识别的紧凑、可解释且有效的编码。通过对形容词进行调整，我们的方法还可以应用于其他自然语言处理任务。 

---
# Benchmarking Large Language Models for Geolocating Colonial Virginia Land Grants 

**Title (ZH)**: 大型语言模型在地理定位殖民维吉尼亚土地许可中的基准测试 

**Authors**: Ryan Mioduski  

**Link**: [PDF](https://arxiv.org/pdf/2508.08266)  

**Abstract**: Virginia's seventeenth- and eighteenth-century land patents survive primarily as narrative metes-and-bounds descriptions, limiting spatial analysis. This study systematically evaluates current-generation large language models (LLMs) in converting these prose abstracts into geographically accurate latitude/longitude coordinates within a focused evaluation context. A digitized corpus of 5,471 Virginia patent abstracts (1695-1732) is released, with 43 rigorously verified test cases serving as an initial, geographically focused benchmark. Six OpenAI models across three architectures (o-series, GPT-4-class, and GPT-3.5) were tested under two paradigms: direct-to-coordinate and tool-augmented chain-of-thought invoking external geocoding APIs. Results were compared with a GIS-analyst baseline, the Stanford NER geoparser, Mordecai-3, and a county-centroid heuristic.
The top single-call model, o3-2025-04-16, achieved a mean error of 23 km (median 14 km), outperforming the median LLM (37.4 km) by 37.5%, the weakest LLM (50.3 km) by 53.5%, and external baselines by 67% (GIS analyst) and 70% (Stanford NER). A five-call ensemble further reduced errors to 19 km (median 12 km) at minimal additional cost (approx. USD 0.20 per grant), outperforming the median LLM by 48.6%. A patentee-name-redaction ablation increased error by about 9%, indicating reliance on textual landmark and adjacency descriptions rather than memorization. The cost-efficient gpt-4o-2024-08-06 model maintained a 28 km mean error at USD 1.09 per 1,000 grants, establishing a strong cost-accuracy benchmark; external geocoding tools offered no measurable benefit in this evaluation.
These findings demonstrate the potential of LLMs for scalable, accurate, and cost-effective historical georeferencing. 

**Abstract (ZH)**: 维吉尼亚州十七世纪和十八世纪的土地专利主要以叙事性的界线描述形式存存，限制了空间分析。本研究系统评估了当前一代大型语言模型（LLMs）将这些散文式的摘要转换为地理精确的纬度/经度坐标的能力，这一评估在特定的评价上下文中进行。发布了5,471份数字化的维吉尼亚州专利摘要（1695-1732），其中包含43个严格验证的测试案例，作为初始的地缘聚焦基准。测试了六种OpenAI模型，涵盖三种架构（o系列、GPT-4级和GPT-3.5），分别在直接坐标转换和工具增强的逐步推理模式（调用外部地理编码API）下进行。结果与GIS分析师基准、斯坦福NER地理解析器、Mordecai-3和县 centroid启发式方法进行了比较。 

---
# TurQUaz at CheckThat! 2025: Debating Large Language Models for Scientific Web Discourse Detection 

**Title (ZH)**: TurQUaz 在 CheckThat! 2025：关于大规模语言模型在科学网络 discourse 检测中的辩论 

**Authors**: Tarık Saraç, Selin Mergen, Mucahid Kutlu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08265)  

**Abstract**: In this paper, we present our work developed for the scientific web discourse detection task (Task 4a) of CheckThat! 2025. We propose a novel council debate method that simulates structured academic discussions among multiple large language models (LLMs) to identify whether a given tweet contains (i) a scientific claim, (ii) a reference to a scientific study, or (iii) mentions of scientific entities. We explore three debating methods: i) single debate, where two LLMs argue for opposing positions while a third acts as a judge; ii) team debate, in which multiple models collaborate within each side of the debate; and iii) council debate, where multiple expert models deliberate together to reach a consensus, moderated by a chairperson model. We choose council debate as our primary model as it outperforms others in the development test set. Although our proposed method did not rank highly for identifying scientific claims (8th out of 10) or mentions of scientific entities (9th out of 10), it ranked first in detecting references to scientific studies. 

**Abstract (ZH)**: 在本文中，我们展示了为CheckThat! 2025科学网络话语检测任务（Task 4a）开发的工作。我们提出了一种新颖的议事辩论方法，通过模拟多个大型语言模型（LLMs）之间的结构化学术讨论来识别给定推文中是否包含（i）科学声明、（ii）科学研究的引用或（iii）科学实体的提及。我们探索了三种辩论方法：i）单方辩论，其中两个LLM持对立立场进行辩论，第三个作为裁判；ii）团队辩论，多个模型在辩论的每一方合作；iii）议事辩论，多个专家模型共同讨论并在主席模型的主持下达成共识。我们选择议事辩论作为主要模型，因为其在开发测试集中的表现优于其他方法。尽管我们提出的方法在识别科学声明（第8名/共10名）和提及科学实体（第9名/共10名）方面表现不佳，但在检测科学研究的引用方面排名第一。 

---
