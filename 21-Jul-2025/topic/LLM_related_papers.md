# AGENTS-LLM: Augmentative GENeration of Challenging Traffic Scenarios with an Agentic LLM Framework 

**Title (ZH)**: AGENTS-LLM: 基于机构性大语言模型框架的具有挑战性的交通场景生成增强方法 

**Authors**: Yu Yao, Salil Bhatnagar, Markus Mazzola, Vasileios Belagiannis, Igor Gilitschenski, Luigi Palmieri, Simon Razniewski, Marcel Hallgarten  

**Link**: [PDF](https://arxiv.org/pdf/2507.13729)  

**Abstract**: Rare, yet critical, scenarios pose a significant challenge in testing and evaluating autonomous driving planners. Relying solely on real-world driving scenes requires collecting massive datasets to capture these scenarios. While automatic generation of traffic scenarios appears promising, data-driven models require extensive training data and often lack fine-grained control over the output. Moreover, generating novel scenarios from scratch can introduce a distributional shift from the original training scenes which undermines the validity of evaluations especially for learning-based planners. To sidestep this, recent work proposes to generate challenging scenarios by augmenting original scenarios from the test set. However, this involves the manual augmentation of scenarios by domain experts. An approach that is unable to meet the demands for scale in the evaluation of self-driving systems. Therefore, this paper introduces a novel LLM-agent based framework for augmenting real-world traffic scenarios using natural language descriptions, addressing the limitations of existing methods. A key innovation is the use of an agentic design, enabling fine-grained control over the output and maintaining high performance even with smaller, cost-effective LLMs. Extensive human expert evaluation demonstrates our framework's ability to accurately adhere to user intent, generating high quality augmented scenarios comparable to those created manually. 

**Abstract (ZH)**: 罕见但关键的场景在测试和评估自动驾驶规划器时构成了重大挑战。仅依赖真实-world驾驶场景需要收集大量数据集以捕获这些场景。尽管自动生成交通场景看起来很有前景，但数据驱动的模型需要广泛的训练数据，且往往缺乏对输出的精细控制。此外，从头生成新颖的场景可能会导致与原始训练场景分布的偏移，这尤其会对基于学习的规划器的评估有效性构成威胁。为解决这一问题，近期的研究提出了通过对测试集中原始场景进行人工扩展来生成具有挑战性的场景的方法。然而，这种方法涉及领域专家的手动场景扩展。这无法满足对自动驾驶系统评估规模的需求。因此，本文介绍了一种基于LLM代理的框架，利用自然语言描述扩充真实世界的交通场景，解决了现有方法的局限性。关键创新在于采用代理设计，这使得在使用较小且成本效益高的LLM时仍能实现对输出的精细控制，并保持高性能。广泛的专家评估证实了该框架能够准确遵循用户意图，生成与手动创建的场景质量相当的高质量扩充场景。 

---
# CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning 

**Title (ZH)**: CUDA-L1：通过对比强化学习改进CUDA优化 

**Authors**: Xiaoya Li, Xiaofei Sun, Albert Wang, Jiwei Li, Chris Shum  

**Link**: [PDF](https://arxiv.org/pdf/2507.14111)  

**Abstract**: The exponential growth in demand for GPU computing resources, driven by the rapid advancement of Large Language Models, has created an urgent need for automated CUDA optimization strategies. While recent advances in LLMs show promise for code generation, current SOTA models (e.g. R1, o1) achieve low success rates in improving CUDA speed. In this paper, we introduce CUDA-L1, an automated reinforcement learning framework for CUDA optimization.
CUDA-L1 achieves performance improvements on the CUDA optimization task: trained on NVIDIA A100, it delivers an average speedup of x17.7 across all 250 CUDA kernels of KernelBench, with peak speedups reaching x449. Furthermore, the model also demonstrates excellent portability across GPU architectures, achieving average speedups of x17.8 on H100, x19.0 on RTX 3090, x16.5 on L40, x14.7 on H800, and x13.9 on H20 despite being optimized specifically for A100. Beyond these benchmark results, CUDA-L1 demonstrates several remarkable properties: 1) Discovers a variety of CUDA optimization techniques and learns to combine them strategically to achieve optimal performance; 2) Uncovers fundamental principles of CUDA optimization; 3) Identifies non-obvious performance bottlenecks and rejects seemingly beneficial optimizations that harm performance.
The capabilities of CUDA-L1 demonstrate that reinforcement learning can transform an initially poor-performing LLM into an effective CUDA optimizer through speedup-based reward signals alone, without human expertise or domain knowledge. More importantly, the trained RL model extend the acquired reasoning abilities to new kernels. This paradigm opens possibilities for automated optimization of CUDA operations, and holds promise to substantially promote GPU efficiency and alleviate the rising pressure on GPU computing resources. 

**Abstract (ZH)**: CUDA计算资源需求的指数增长，驱动于大型语言模型的快速进步，迫切需要自动CUDA优化策略。尽管最近的大型语言模型在代码生成方面表现出希望，但当前的SOTA模型（如R1、o1）在提高CUDA速度方面成功率较低。本文介绍了一种自动强化学习框架CUDA-L1，用于CUDA优化。CUDA-L1在CUDA优化任务上实现了性能提升：在NVIDIA A100上训练后，它在KernelBench的250个CUDA内核中平均提供了17.7倍的速度提升，峰值提升达到449倍。此外，该模型还展示了出色的跨GPU架构可移植性，在H100上平均提供17.8倍的加速，在RTX 3090上提供19.0倍的加速，在L40上提供16.5倍的加速，在H800上提供14.7倍的加速，在H20上提供13.9倍的加速，尽管它专门针对A100进行了优化。除了这些基准结果外，CUDA-L1展示了多种非凡特性：1) 发现多种CUDA优化技术，并学会战略性地组合这些技术以实现最优性能；2) 揭示CUDA优化的基本原理；3) 识别出不明显的性能瓶颈，并拒绝看似有益但实际上损害性能的优化。CUDA-L1的能力表明，仅通过基于速度提升的奖励信号，强化学习可以将最初表现不佳的大型语言模型转变为有效的CUDA优化器，而无需人为专业知识或领域知识。更重要的是，已经训练的RL模型将其获得的推理能力扩展到新的内核。这一范式开启了CUDA操作自动优化的可能性，并有望显著提升GPU效率和缓解对GPU计算资源的日益增长的压力。 

---
# Automated Interpretation of Non-Destructive Evaluation Contour Maps Using Large Language Models for Bridge Condition Assessment 

**Title (ZH)**: 使用大型语言模型自动解释桥梁条件评估非破坏性评价轮廓图 

**Authors**: Viraj Nishesh Darji, Callie C. Liao, Duoduo Liao  

**Link**: [PDF](https://arxiv.org/pdf/2507.14107)  

**Abstract**: Bridge maintenance and safety are essential for transportation authorities, and Non-Destructive Evaluation (NDE) techniques are critical to assessing structural integrity. However, interpreting NDE data can be time-consuming and requires expertise, potentially delaying decision-making. Recent advancements in Large Language Models (LLMs) offer new ways to automate and improve this analysis. This pilot study introduces a holistic assessment of LLM capabilities for interpreting NDE contour maps and demonstrates the effectiveness of LLMs in providing detailed bridge condition analyses. It establishes a framework for integrating LLMs into bridge inspection workflows, indicating that LLM-assisted analysis can enhance efficiency without compromising accuracy. In this study, several LLMs are explored with prompts specifically designed to enhance the quality of image descriptions, which are applied to interpret five different NDE contour maps obtained through technologies for assessing bridge conditions. Each LLM model is evaluated based on its ability to produce detailed descriptions, identify defects, provide actionable recommendations, and demonstrate overall accuracy. The research indicates that four of the nine models provide better image descriptions, effectively covering a wide range of topics related to the bridge's condition. The outputs from these four models are summarized using five different LLMs to form a comprehensive overview of the bridge. Notably, LLMs ChatGPT-4 and Claude 3.5 Sonnet generate more effective summaries. The findings suggest that LLMs have the potential to significantly improve efficiency and accuracy. This pilot study presents an innovative approach that leverages LLMs for image captioning in parallel and summarization, enabling faster decision-making in bridge maintenance and enhancing infrastructure management and safety assessments. 

**Abstract (ZH)**: 基于大型语言模型的无损检测数据分析在桥梁维护中的应用研究 

---
# KROMA: Ontology Matching with Knowledge Retrieval and Large Language Models 

**Title (ZH)**: KROMA：基于知识检索和大规模语言模型的本体匹配 

**Authors**: Lam Nguyen, Erika Barcelos, Roger French, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14032)  

**Abstract**: Ontology Matching (OM) is a cornerstone task of semantic interoperability, yet existing systems often rely on handcrafted rules or specialized models with limited adaptability. We present KROMA, a novel OM framework that harnesses Large Language Models (LLMs) within a Retrieval-Augmented Generation (RAG) pipeline to dynamically enrich the semantic context of OM tasks with structural, lexical, and definitional knowledge. To optimize both performance and efficiency, KROMA integrates a bisimilarity-based concept matching and a lightweight ontology refinement step, which prune candidate concepts and substantially reduce the communication overhead from invoking LLMs. Through experiments on multiple benchmark datasets, we show that integrating knowledge retrieval with context-augmented LLMs significantly enhances ontology matching, outperforming both classic OM systems and cutting-edge LLM-based approaches while keeping communication overhead comparable. Our study highlights the feasibility and benefit of the proposed optimization techniques (targeted knowledge retrieval, prompt enrichment, and ontology refinement) for ontology matching at scale. 

**Abstract (ZH)**: 基于大型语言模型的检索增强生成框架KROMA在本体匹配中的应用 

---
# Large Language Models as Innovators: A Framework to Leverage Latent Space Exploration for Novelty Discovery 

**Title (ZH)**: 大型语言模型作为创新者：一个利用潜在空间探索发现新颖性的框架 

**Authors**: Mateusz Bystroński, Mikołaj Hołysz, Grzegorz Piotrowski, Nitesh V. Chawla, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2507.13874)  

**Abstract**: Innovative idea generation remains a core challenge in AI, as large language models (LLMs) often struggle to produce outputs that are both novel and relevant. Despite their fluency, LLMs tend to replicate patterns seen during training, limiting their ability to diverge creatively without extensive prompt engineering. Prior work has addressed this through domain-specific heuristics and structured prompting pipelines, but such solutions are brittle and difficult to generalize. In this paper, we propose a model-agnostic latent-space ideation framework that enables controlled, scalable creativity by navigating the continuous embedding space of ideas. Unlike prior methods, our framework requires no handcrafted rules and adapts easily to different domains, input formats, and creative tasks. This paper introduces an early-stage prototype of our method, outlining the conceptual framework and preliminary results highlighting its potential as a general-purpose co-ideator for human-AI collaboration. 

**Abstract (ZH)**: 创新想法生成仍然是AI中的核心挑战，尽管大语言模型在流畅性方面表现出色，但往往难以产生既新颖又相关的输出。尽管具有流畅性，大语言模型倾向于在训练过程中复制模式，限制了它们在无需大量提示工程的情况下进行创造性发散的能力。先前的研究通过领域特定的启发式方法和结构化提示管道来解决这一问题，但这些解决方案脆弱且难以泛化。在本文中，我们提出了一种模型无关的潜在空间ideation框架，通过导航想法的连续嵌入空间来实现受控的、可扩展的创造力。与先前的方法不同，我们的框架不需要手工艺品规则，并且可以轻松适应不同的领域、输入格式和创意任务。本文介绍了一种早期原型方法，概述了概念框架和初步结果，强调其作为人类-AI协作的一般辅助创意工具的潜力。 

---
# DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs 

**Title (ZH)**: DailyLLM：基于多模态传感器和大规模语言模型的上下文感知活动日志生成 

**Authors**: Ye Tian, Xiaoyuan Ren, Zihao Wang, Onat Gungor, Xiaofan Yu, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2507.13737)  

**Abstract**: Rich and context-aware activity logs facilitate user behavior analysis and health monitoring, making them a key research focus in ubiquitous computing. The remarkable semantic understanding and generation capabilities of Large Language Models (LLMs) have recently created new opportunities for activity log generation. However, existing methods continue to exhibit notable limitations in terms of accuracy, efficiency, and semantic richness. To address these challenges, we propose DailyLLM. To the best of our knowledge, this is the first log generation and summarization system that comprehensively integrates contextual activity information across four dimensions: location, motion, environment, and physiology, using only sensors commonly available on smartphones and smartwatches. To achieve this, DailyLLM introduces a lightweight LLM-based framework that integrates structured prompting with efficient feature extraction to enable high-level activity understanding. Extensive experiments demonstrate that DailyLLM outperforms state-of-the-art (SOTA) log generation methods and can be efficiently deployed on personal computers and Raspberry Pi. Utilizing only a 1.5B-parameter LLM model, DailyLLM achieves a 17% improvement in log generation BERTScore precision compared to the 70B-parameter SOTA baseline, while delivering nearly 10x faster inference speed. 

**Abstract (ZH)**: 丰富的上下文感知活动日志促进用户行为分析和健康监测，因此在泛在计算中成为关键研究重点。大型语言模型（LLMs）在语义理解和生成方面的显著能力为活动日志生成创造了新的机会。然而，现有的方法在准确性和语义丰富性方面仍然存在明显限制。为了解决这些挑战，我们提出了DailyLLM。据我们所知，这是第一个全面整合跨四个维度（位置、运动、环境和生理）上下文活动信息的日志生成和总结系统，仅使用智能手机和智能手表上常见的传感器。为了实现这一目标，DailyLLM 引入了一种轻量级的基于LLM的框架，该框架结合了结构化提示与高效的特征提取，以实现高级活动理解。广泛实验表明，DailyLLM 在日志生成方面优于最先进的（SOTA）方法，并且可以高效部署在个人计算机和 Raspberry Pi 上。仅使用一个包含1.5B参数的LLM模型，DailyLLM 在日志生成 BERTScore 精度上比参数量为70B的SOTA基线提高了17%，同时提供近10倍的推理速度。 

---
# BifrostRAG: Bridging Dual Knowledge Graphs for Multi-Hop Question Answering in Construction Safety 

**Title (ZH)**: BifrostRAG: 联接双知识图谱进行建筑安全多跳问答 

**Authors**: Yuxin Zhang, Xi Wang, Mo Hu, Zhenyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13625)  

**Abstract**: Information retrieval and question answering from safety regulations are essential for automated construction compliance checking but are hindered by the linguistic and structural complexity of regulatory text. Many compliance-related queries are multi-hop, requiring synthesis of information across interlinked clauses. This poses a challenge for traditional retrieval-augmented generation (RAG) systems. To overcome this, we introduce BifrostRAG: a dual-graph RAG-integrated system that explicitly models both linguistic relationships (via an Entity Network Graph) and document structure (via a Document Navigator Graph). This architecture powers a hybrid retrieval mechanism that combines graph traversal with vector-based semantic search, enabling large language models to reason over both the meaning and the structure of the text. Evaluation on a multi-hop question dataset shows that BifrostRAG achieves 92.8 percent precision, 85.5 percent recall, and an F1 score of 87.3 percent. These results significantly outperform vector-only and graph-only RAG baselines that represent current leading approaches. Error analysis further highlights the comparative advantages of our hybrid method over single-modality RAGs. These findings establish BifrostRAG as a robust knowledge engine for LLM-driven compliance checking. Its dual-graph, hybrid retrieval mechanism offers a transferable blueprint for navigating complex technical documents across knowledge-intensive engineering domains. 

**Abstract (ZH)**: 基于双图的RAG系统：BifrostRAG及其在安全规范信息检索与问答中的应用 

---
# GOFAI meets Generative AI: Development of Expert Systems by means of Large Language Models 

**Title (ZH)**: 基于大型语言模型开发专家系统：GOFAI与生成型AI的融合 

**Authors**: Eduardo C. Garrido-Merchán, Cristina Puente  

**Link**: [PDF](https://arxiv.org/pdf/2507.13550)  

**Abstract**: The development of large language models (LLMs) has successfully transformed knowledge-based systems such as open domain question nswering, which can automatically produce vast amounts of seemingly coherent information. Yet, those models have several disadvantages like hallucinations or confident generation of incorrect or unverifiable facts. In this paper, we introduce a new approach to the development of expert systems using LLMs in a controlled and transparent way. By limiting the domain and employing a well-structured prompt-based extraction approach, we produce a symbolic representation of knowledge in Prolog, which can be validated and corrected by human experts. This approach also guarantees interpretability, scalability and reliability of the developed expert systems. Via quantitative and qualitative experiments with Claude Sonnet 3.7 and GPT-4.1, we show strong adherence to facts and semantic coherence on our generated knowledge bases. We present a transparent hybrid solution that combines the recall capacity of LLMs with the precision of symbolic systems, thereby laying the foundation for dependable AI applications in sensitive domains. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的发展成功地转化了知识型系统，如开放领域问答，可以自动生成大量看似连贯的信息。然而，这些模型存在诸如幻觉或自信生成错误或无法验证的事实等缺点。本文介绍了一种通过受控和透明方式使用LLMs开发专家系统的新型方法。通过限定领域并采用结构化的提示提取方法，我们产生了在Prolog中的符号知识表示，该表示可以由人类专家验证和修正。这种方法还保证了开发的专家系统的可解释性、可扩展性和可靠性。通过与Claude Sonnet 3.7和GPT-4.1的定量和定性实验，我们展示了生成的知识库在事实准确性和语义连贯性方面的坚定遵循。我们提出了一种透明的混合解决方案，结合了LLMs的记忆能力和符号系统的精确性，从而为敏感领域的可靠AI应用奠定了基础。 

---
# GraphTrafficGPT: Enhancing Traffic Management Through Graph-Based AI Agent Coordination 

**Title (ZH)**: GraphTrafficGPT：通过图基人工智能代理协调增强交通管理 

**Authors**: Nabil Abdelaziz Ferhat Taleb, Abdolazim Rezaei, Raj Atulkumar Patel, Mehdi Sookhak  

**Link**: [PDF](https://arxiv.org/pdf/2507.13511)  

**Abstract**: Large Language Models (LLMs) offer significant promise for intelligent traffic management; however, current chain-based systems like TrafficGPT are hindered by sequential task execution, high token usage, and poor scalability, making them inefficient for complex, real-world scenarios. To address these limitations, we propose GraphTrafficGPT, a novel graph-based architecture, which fundamentally redesigns the task coordination process for LLM-driven traffic applications. GraphTrafficGPT represents tasks and their dependencies as nodes and edges in a directed graph, enabling efficient parallel execution and dynamic resource allocation. The main idea behind the proposed model is a Brain Agent that decomposes user queries, constructs optimized dependency graphs, and coordinates a network of specialized agents for data retrieval, analysis, visualization, and simulation. By introducing advanced context-aware token management and supporting concurrent multi-query processing, the proposed architecture handles interdependent tasks typical of modern urban mobility environments. Experimental results demonstrate that GraphTrafficGPT reduces token consumption by 50.2% and average response latency by 19.0% compared to TrafficGPT, while supporting simultaneous multi-query execution with up to 23.0% improvement in efficiency. 

**Abstract (ZH)**: 大型语言模型（LLMs）在智能交通管理方面提供了显著的潜力；然而，当前基于链的设计如TrafficGPT受限于顺序任务执行、高 token 使用率和较差的扩展性，使得它们在复杂的现实场景中效率低下。为了克服这些限制，我们提出了一种新的图基架构GraphTrafficGPT，从根本上重新设计了由LLM驱动的交通应用的任务协调过程。GraphTrafficGPT将任务及其依赖关系表示为有向图中的节点和边，从而实现高效并行执行和动态资源分配。所提出模型的核心思想是一种名为Brain Agent的实体，它分解用户查询、构建优化的依赖关系图，并协调数据检索、分析、可视化和模拟等专门代理。通过引入先进的上下文感知 token 管理和支持并发多查询处理，所提出架构能够处理现代城市交通环境中典型的相互依赖任务。实验结果表明，与TrafficGPT相比，GraphTrafficGPT将token消耗减少了50.2%，平均响应延时降低了19.0%，同时支持并发多查询执行，效率提高了23.0%。 

---
# Lessons from the TREC Plain Language Adaptation of Biomedical Abstracts (PLABA) track 

**Title (ZH)**: TREC plain language adaptation of biomedical abstracts (PLABA) 轨道的经验总结 

**Authors**: Brian Ondov, William Xia, Kush Attal, Ishita Unde, Jerry He, Hoa Dang, Ian Soboroff, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2507.14096)  

**Abstract**: Objective: Recent advances in language models have shown potential to adapt professional-facing biomedical literature to plain language, making it accessible to patients and caregivers. However, their unpredictability, combined with the high potential for harm in this domain, means rigorous evaluation is necessary. Our goals with this track were to stimulate research and to provide high-quality evaluation of the most promising systems.
Methods: We hosted the Plain Language Adaptation of Biomedical Abstracts (PLABA) track at the 2023 and 2024 Text Retrieval Conferences. Tasks included complete, sentence-level, rewriting of abstracts (Task 1) as well as identifying and replacing difficult terms (Task 2). For automatic evaluation of Task 1, we developed a four-fold set of professionally-written references. Submissions for both Tasks 1 and 2 were provided extensive manual evaluation from biomedical experts.
Results: Twelve teams spanning twelve countries participated in the track, with models from multilayer perceptrons to large pretrained transformers. In manual judgments of Task 1, top-performing models rivaled human levels of factual accuracy and completeness, but not simplicity or brevity. Automatic, reference-based metrics generally did not correlate well with manual judgments. In Task 2, systems struggled with identifying difficult terms and classifying how to replace them. When generating replacements, however, LLM-based systems did well in manually judged accuracy, completeness, and simplicity, though not in brevity.
Conclusion: The PLABA track showed promise for using Large Language Models to adapt biomedical literature for the general public, while also highlighting their deficiencies and the need for improved automatic benchmarking tools. 

**Abstract (ZH)**: 目标：最近的语言模型研究显示，这些模型具有将面向专业人士的生物医学文献转化为通俗语言的潜力，使之易于患者和护理人员理解。然而，由于这种转化的高度不可预测性以及该领域的高风险性，严格的评估是必要的。我们设立这一赛道的主要目的是激发研究兴趣，并提供高质量的评估以检验最具前景的系统。

方法：我们在2023年和2024年的文本检索会议上举办了生物医学摘要通俗化适应（PLABA）赛道。任务包括完整的、逐句的摘要重写（任务1）以及识别和替换难懂术语（任务2）。对于任务1的自动评估，我们开发了四组由专业人士撰写的参考标准。两个任务的提交结果都经过了生物医学专家的详细手动评估。

结果：来自十二个国家的十二支队伍参与了该赛道，使用了从多层感知器到大规模预训练变换器的各种模型。在任务1的手动评估中，表现最佳的模型在事实准确性和完整性上达到了人类的水平，但在简单性和简洁性方面则不然。自动参考基于的度量标准通常与人工评估结果不一致。在任务2中，系统在识别难懂术语和分类如何替换它们方面面临困难。然而，当生成替换时，基于大型语言模型的系统在手动评估的准确性、完整性和简洁性方面表现出色，但在简洁性方面则不尽如人意。

结论：PLABA赛道展示了大型语言模型在适应生物医学文献方面以供普通公众使用的潜力，同时也揭示了它们的不足之处，并强调了改进自动基准工具的需求。 

---
# Photonic Fabric Platform for AI Accelerators 

**Title (ZH)**: 光子 Fabric 平台 for AI 加速器 

**Authors**: Jing Ding, Trung Diep  

**Link**: [PDF](https://arxiv.org/pdf/2507.14000)  

**Abstract**: This paper presents the Photonic FabricTM and the Photonic Fabric ApplianceTM (PFA), a photonic-enabled switch and memory subsystem that delivers low latency, high bandwidth, and low per-bit energy. By integrating high-bandwidth HBM3E memory, an on-module photonic switch, and external DDR5 in a 2.5D electro-optical system-in-package, the PFA offers up to 32 TB of shared memory alongside 115 Tbps of all-to-all digital switching. The Photonic FabricTM enables distributed AI training and inference to execute parallelism strategies more efficiently. The Photonic Fabric removes the silicon beachfront constraint that limits the fixed memory-to-compute ratio observed in virtually all current XPU accelerator designs. Replacing a local HBM stack on an XPU with a chiplet that connects to the Photonic Fabric increases its memory capacity and correspondingly its memory bandwidth by offering a flexible path to scaling well beyond the limitations of on-package HBM alone. We introduce CelestiSim, a lightweight analytical simulator validated on NVIDIA H100 and H200 systems. It is used to evaluate the performance of LLM reference and energy savings on PFA, without any significant change to the GPU core design. With the PFA, the simulation results show that up to 3.66x throughput and 1.40x latency improvements in LLM inference at 405B parameters, up to 7.04x throughput and 1.41x latency improvements at 1T parameters, and 60-90% energy savings in data movement for heavy collective operations in all LLM training scenarios. While these results are shown for NVIDIA GPUs, they can be applied similarly to other AI accelerator designs (XPUs) that share the same fundamental limitation of fixed memory to compute. 

**Abstract (ZH)**: 基于光电fabric的光子 FabricTM 和光子 Fabric 装置TM（PFA）：实现低延迟、高带宽和低比特能效的光子化交换与内存子系统 

---
# OrthoInsight: Rib Fracture Diagnosis and Report Generation Based on Multi-Modal Large Models 

**Title (ZH)**: OrthoInsight：基于多模态大规模模型的肋骨骨折诊断及报告生成 

**Authors**: Ningyong Wu, Jinzhi Wang, Wenhong Zhao, Chenzhan Yu, Zhigang Xiu, Duwei Dai  

**Link**: [PDF](https://arxiv.org/pdf/2507.13993)  

**Abstract**: The growing volume of medical imaging data has increased the need for automated diagnostic tools, especially for musculoskeletal injuries like rib fractures, commonly detected via CT scans. Manual interpretation is time-consuming and error-prone. We propose OrthoInsight, a multi-modal deep learning framework for rib fracture diagnosis and report generation. It integrates a YOLOv9 model for fracture detection, a medical knowledge graph for retrieving clinical context, and a fine-tuned LLaVA language model for generating diagnostic reports. OrthoInsight combines visual features from CT images with expert textual data to deliver clinically useful outputs. Evaluated on 28,675 annotated CT images and expert reports, it achieves high performance across Diagnostic Accuracy, Content Completeness, Logical Coherence, and Clinical Guidance Value, with an average score of 4.28, outperforming models like GPT-4 and Claude-3. This study demonstrates the potential of multi-modal learning in transforming medical image analysis and providing effective support for radiologists. 

**Abstract (ZH)**: 医学影像数据量的增长增加了对自动诊断工具的需求，尤其是在通过CT扫描常用检测的肋骨骨折等骨骼损伤诊断中。手工解读耗时且易出错。我们提出了一种称为OrthoInsight的多模态深度学习框架，用于肋骨骨折诊断和报告生成。该框架整合了YOLOv9模型进行骨折检测、医学知识图谱检索临床背景以及微调后的LLaVA语言模型生成诊断报告。OrthoInsight结合了CT图像的视觉特征和专家文本数据，以提供临床有用的输出。在28,675张标注的CT图像和专家报告上进行评估，其在诊断准确性、内容完整性、逻辑连贯性和临床指导价值等方面的性能均表现优异，平均得分为4.28，优于如GPT-4和Claude-3等模型。本研究展示了多模态学习在医疗图像分析中的潜力及其为放射科医生提供的有效支持。 

---
# Bottom-up Domain-specific Superintelligence: A Reliable Knowledge Graph is What We Need 

**Title (ZH)**: 自底向上领域特定超智能：我们所需要的是可靠的知识图谱 

**Authors**: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2507.13966)  

**Abstract**: Language models traditionally used for cross-domain generalization have recently demonstrated task-specific reasoning. However, their top-down training approach on general corpora is insufficient for acquiring abstractions needed for deep domain expertise. This may require a bottom-up approach that acquires expertise by learning to compose simple domain concepts into more complex ones. A knowledge graph (KG) provides this compositional structure, where domain primitives are represented as head-relation-tail edges and their paths encode higher-level concepts. We present a task generation pipeline that synthesizes tasks directly from KG primitives, enabling models to acquire and compose them for reasoning. We fine-tune language models on the resultant KG-grounded curriculum to demonstrate domain-specific superintelligence. While broadly applicable, we validate our approach in medicine, where reliable KGs exist. Using a medical KG, we curate 24,000 reasoning tasks paired with thinking traces derived from diverse medical primitives. We fine-tune the QwQ-32B model on this curriculum to obtain QwQ-Med-3 that takes a step towards medical superintelligence. We also introduce ICD-Bench, an evaluation suite to quantify reasoning abilities across 15 medical domains. Our experiments demonstrate that QwQ-Med-3 significantly outperforms state-of-the-art reasoning models on ICD-Bench categories. Further analysis reveals that QwQ-Med-3 utilizes acquired primitives to widen the performance gap on the hardest tasks of ICD-Bench. Finally, evaluation on medical question-answer benchmarks shows that QwQ-Med-3 transfers acquired expertise to enhance the base model's performance. While the industry's approach to artificial general intelligence (AGI) emphasizes broad expertise, we envision a future in which AGI emerges from the composable interaction of efficient domain-specific superintelligent agents. 

**Abstract (ZH)**: 语言模型传统上用于跨域泛化的任务特定推理：一种基于知识图谱的底向上方法实现深度领域专长 

---
# DUALRec: A Hybrid Sequential and Language Model Framework for Context-Aware Movie Recommendation 

**Title (ZH)**: DUALRec：一种上下文aware电影推荐的混合序列和语言模型框架 

**Authors**: Yitong Li, Raoul Grasman  

**Link**: [PDF](https://arxiv.org/pdf/2507.13957)  

**Abstract**: The modern recommender systems are facing an increasing challenge of modelling and predicting the dynamic and context-rich user preferences. Traditional collaborative filtering and content-based methods often struggle to capture the temporal patternings and evolving user intentions. While Large Language Models (LLMs) have gained gradual attention in recent years, by their strong semantic understanding and reasoning abilities, they are not inherently designed to model chronologically evolving user preference and intentions. On the other hand, for sequential models like LSTM (Long-Short-Term-Memory) which is good at capturing the temporal dynamics of user behaviour and evolving user preference over time, but still lacks a rich semantic understanding for comprehensive recommendation generation. In this study, we propose DUALRec (Dynamic User-Aware Language-based Recommender), a novel recommender that leverages the complementary strength of both models, which combines the temporal modelling abilities of LSTM networks with semantic reasoning power of the fine-tuned Large Language Models. The LSTM component will capture users evolving preference through their viewing history, while the fine-tuned LLM variants will leverage these temporal user insights to generate next movies that users might enjoy. Experimental results on MovieLens-1M dataset shows that the DUALRec model outperforms a wide range of baseline models, with comprehensive evaluation matrices of Hit Rate (HR@k), Normalized Discounted Cumulative Gain (NDCG@k), and genre similarity metrics. This research proposes a novel architecture that bridges the gap between temporal sequence modeling and semantic reasoning, and offers a promising direction for developing more intelligent and context-aware recommenders. 

**Abstract (ZH)**: 现代推荐系统面临的动态和情境丰富用户偏好建模与预测挑战传统协同过滤和基于内容的方法往往难以捕捉时间模式和用户意图的变化。虽然近年来大型语言模型（LLMs）因其强大的语义理解和推理能力逐渐获得关注，但它们本就不擅长建模随着时间演化的用户偏好和意图。另一方面，长短期记忆网络（LSTM）等序列模型擅长捕捉用户行为的时间动态以及随着时间演变的用户偏好，但仍缺乏全面推荐生成所需的丰富语义理解能力。在这项研究中，我们提出了一种名为DUALRec（动态用户感知语言推荐）的新颖推荐系统，该系统结合了LSTM网络的时间建模能力和微调大型语言模型的语义推理能力。LSTM组件将通过用户的观看历史捕捉用户不断变化的偏好，而微调后的大型语言模型变体将利用这些时间上的用户洞察生成用户可能喜欢的下一个电影。在MovieLens-1M数据集上的实验结果显示，DUALRec模型在精确率、归一化折扣累积增益和类别相似度指标等全面评估矩阵中均优于多种基线模型。本文提出了一种新的架构，弥合了时间序列建模和语义推理之间的差距，并为开发更具智能性和情境意识的推荐系统提供了前景。 

---
# Exploiting Primacy Effect To Improve Large Language Models 

**Title (ZH)**: 利用首因效应提升大型语言模型 

**Authors**: Bianca Raimondi, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2507.13949)  

**Abstract**: Large Language Models (LLMs) have become essential in many Natural Language Processing (NLP) tasks, leveraging extensive pre-training and fine-tuning to achieve high accuracy. However, like humans, LLMs exhibit biases, particularly positional biases such as primacy and recency effects, which can influence the accuracy of the answers. The primacy effect-where items presented first are more likely to be remembered or selected-plays a key role in Multiple Choice Question Answering (MCQA), where the order of answer options can affect prediction outcomes. This study focuses on primacy bias in fine-tuned LLMs: We first show that fine-tuning amplifies this bias, probably due to exposure to human-like patterns. Hence, we strategically leverage this effect by reordering response options based on semantic similarity to the query, without requiring knowledge of the correct answer. Our experimental results show that this approach significantly improves performance in MCQA. More generally, our findings underscore the dual nature of biases as both challenges and opportunities, offering insights for bias-aware model design and NLP applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在许多自然语言处理（NLP）任务中变得至关重要，通过广泛的预训练和微调实现高精度。然而，就像人类一样，LLMs也表现出偏见，特别是位置偏见，如首因效应和近因效应，这些偏见会影响答案的准确性。首因效应，即首先呈现的项目更有可能被记住或选择，在多项选择题作答（MCQA）中起着关键作用，因为答案选项的排列顺序会影响预测结果。本研究重点探讨微调后的LLMs中的首因偏见：首先我们表明，微调会增强这种偏见，可能是由于暴露在类似人类的模式中。因此，我们通过基于查询的语义相似性重新排列响应选项来战略性地利用这一效果，而无需知道正确的答案。我们的实验结果表明，这种方法在多项选择题作答中显著提高了性能。更广泛地说，我们的发现强调了偏见的双重性质——既是挑战也是机遇——为偏见意识模型设计和自然语言处理应用提供了见解。 

---
# Preprint: Did I Just Browse A Website Written by LLMs? 

**Title (ZH)**: 预印本:我刚刚浏览了一个由LLM编写的网页？ 

**Authors**: Sichang "Steven" He, Ramesh Govindan, Harsha V. Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2507.13933)  

**Abstract**: Increasingly, web content is automatically generated by large language models (LLMs) with little human input. We call this "LLM-dominant" content. Since LLMs plagiarize and hallucinate, LLM-dominant content can be unreliable and unethical. Yet, websites rarely disclose such content, and human readers struggle to distinguish it. Thus, we must develop reliable detectors for LLM-dominant content. However, state-of-the-art LLM detectors are insufficient, because they perform well mainly on clean, prose-like text, while web content has complex markup and diverse genres.
We propose a highly reliable, scalable pipeline that classifies entire websites. Instead of naively classifying text extracted from each page, we classify each site based on an LLM text detector's outputs of multiple prose-like pages. We train and evaluate our detector by collecting 2 distinct ground truth datasets totaling 120 sites, and obtain 100% accuracies testing across them. In the wild, we detect a sizable portion of sites as LLM-dominant among 10k sites in search engine results and 10k in Common Crawl archives. We find LLM-dominant sites are growing in prevalence and rank highly in search results, raising questions about their impact on end users and the overall Web ecosystem. 

**Abstract (ZH)**: 越来越多的网络内容由大型语言模型（LLMs）自动生成，几乎没有人类干预。我们称这种内容为“LLM主导”内容。由于LLMs存在抄袭和幻觉的问题，“LLM主导”内容可能不可靠且不伦理。然而，网站很少披露此类内容，人类读者也难以区分。因此，我们必须开发可靠的“LLM主导”内容检测器。然而，现有的LLM检测器并不完善，因为它们主要在干净的、散文式的文本上表现良好，而网络内容则具有复杂的标记和多样的体裁。

我们提出了一个高度可靠、可扩展的工作流来分类整个网站。我们不是简单地对每页提取的文本进行分类，而是基于LLM文本检测器在多个散文式的页面上产生的输出来对每个站点进行分类。通过收集两个不同的地面真实数据集（共计120个站点）进行训练和评估，我们在它们上的测试准确率达到100%。在现实世界中，我们在搜索引擎结果和康蒙·克罗（Common Crawl）档案中的10000个站点中检测到相当一部分站点是“LLM主导”内容。我们发现“LLM主导”站点正在不断增加，并在搜索结果中排名靠前，这引发了对其对最终用户和整个网络生态系统影响的疑问。 

---
# The Levers of Political Persuasion with Conversational AI 

**Title (ZH)**: 借助对话式AI的政治说服杠杆 

**Authors**: Kobi Hackenburg, Ben M. Tappin, Luke Hewitt, Ed Saunders, Sid Black, Hause Lin, Catherine Fist, Helen Margetts, David G. Rand, Christopher Summerfield  

**Link**: [PDF](https://arxiv.org/pdf/2507.13919)  

**Abstract**: There are widespread fears that conversational AI could soon exert unprecedented influence over human beliefs. Here, in three large-scale experiments (N=76,977), we deployed 19 LLMs-including some post-trained explicitly for persuasion-to evaluate their persuasiveness on 707 political issues. We then checked the factual accuracy of 466,769 resulting LLM claims. Contrary to popular concerns, we show that the persuasive power of current and near-future AI is likely to stem more from post-training and prompting methods-which boosted persuasiveness by as much as 51% and 27% respectively-than from personalization or increasing model scale. We further show that these methods increased persuasion by exploiting LLMs' unique ability to rapidly access and strategically deploy information and that, strikingly, where they increased AI persuasiveness they also systematically decreased factual accuracy. 

**Abstract (ZH)**: 大规模实验证明当前和近未来AI的说服力更多源自于后训练和提示方法而非个性化或模型规模增加 

---
# Using LLMs to identify features of personal and professional skills in an open-response situational judgment test 

**Title (ZH)**: 使用大语言模型识别开放 réponse 情景判断测试中个人和专业技能特征 

**Authors**: Cole Walsh, Rodica Ivan, Muhammad Zafar Iqbal, Colleen Robb  

**Link**: [PDF](https://arxiv.org/pdf/2507.13881)  

**Abstract**: Academic programs are increasingly recognizing the importance of personal and professional skills and their critical role alongside technical expertise in preparing students for future success in diverse career paths. With this growing demand comes the need for scalable systems to measure, evaluate, and develop these skills. Situational Judgment Tests (SJTs) offer one potential avenue for measuring these skills in a standardized and reliable way, but open-response SJTs have traditionally relied on trained human raters for evaluation, presenting operational challenges to delivering SJTs at scale. Past attempts at developing NLP-based scoring systems for SJTs have fallen short due to issues with construct validity of these systems. In this article, we explore a novel approach to extracting construct-relevant features from SJT responses using large language models (LLMs). We use the Casper SJT to demonstrate the efficacy of this approach. This study sets the foundation for future developments in automated scoring for personal and professional skills. 

**Abstract (ZH)**: 学术项目越来越认识到个人与职业技能的重要性，以及这些技能与技术专长一同在为学生多元化职业路径的成功做准备中起到的关键作用。随着这一需求的增长，需要有可扩展的系统来衡量、评估和发展这些技能。情境判断测试（SJT）提供了一种在标准化和可靠的方式下测量这些技能的潜在途径，但传统的开放式SJT评分依赖训练有素的人类评分员，这在大规模实施SJT时提出了操作挑战。过去基于NLP的SJT评分系统的开发由于这些系统构建效度的问题而未能成功。本文探讨了使用大规模语言模型（LLM）从SJT回答中提取构建相关特征的新型方法。我们使用Casper SJT来证明这一方法的有效性。本研究为未来个人和职业技能的自动化评分发展奠定了基础。 

---
# SPARQL Query Generation with LLMs: Measuring the Impact of Training Data Memorization and Knowledge Injection 

**Title (ZH)**: 基于LLMs的SPARQL查询生成：训练数据记忆和知识注入的影响测量 

**Authors**: Aleksandr Gashkov, Aleksandr Perevalov, Maria Eltsova, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2507.13859)  

**Abstract**: Nowadays, the importance of software with natural-language user interfaces cannot be underestimated. In particular, in Question Answering (QA) systems, generating a SPARQL query for a given natural-language question (often named Query Building) from the information retrieved from the same question is the central task of QA systems working over Knowledge Graphs (KGQA). Due to the rise of Large Language Models (LLMs), they are considered a well-suited method to increase the quality of the question-answering functionality, as there is still a lot of room for improvement, aiming for enhanced quality and trustworthiness. However, LLMs are trained on web data, where researchers have no control over whether the benchmark or the knowledge graph was already included in the training data. In this paper, we introduce a novel method that evaluates the quality of LLMs by generating a SPARQL query from a natural-language question under various conditions: (1) zero-shot SPARQL generation, (2) with knowledge injection, and (3) with "anonymized" knowledge injection. This enables us, for the first time, to estimate the influence of the training data on the QA quality improved by LLMs. Ultimately, this will help to identify how portable a method is or whether good results might mostly be achieved because a benchmark was already included in the training data (cf. LLM memorization). The developed method is portable, robust, and supports any knowledge graph; therefore, it could be easily applied to any KGQA or LLM, s.t., generating consistent insights into the actual LLM capabilities is possible. 

**Abstract (ZH)**: 现今，具有自然语言用户接口的软件的重要性不容忽视。特别是，在基于知识图谱的问答系统（KGQA）中的问答系统中，从同一问题检索到的信息生成给定的自然语言问题的SPARQL查询（通常称为查询构建）是核心任务。由于大型语言模型（LLMs）的兴起，它们被认为是一种提高问答功能质量的合适方法，尽管还有很大的改进空间，以增强质量和可信度。然而，LLMs是在网络数据上进行训练的，研究人员无法控制基准或知识图谱是否已在训练数据中包含。在此论文中，我们提出了一种新型方法，通过在不同条件下生成自然语言问题的SPARQL查询来评估LLMs的质量：（1）零样本SPARQL生成，（2）带知识注入，（3）带“匿名”知识注入。这使我们首次能够估计训练数据对由LLMs提高的问答质量的影响。最终，这将有助于识别方法的适用性，或确定良好结果是否主要是因为基准已经在训练数据中包含（类似于LLM记忆）。所开发的方法是可移植的、稳健的，并支持任何知识图谱；因此，它可以很容易地应用于任何KGQA或LLM，以便获得关于实际LLM能力的一致见解。 

---
# RAG-based Architectures for Drug Side Effect Retrieval in LLMs 

**Title (ZH)**: 基于RAG的架构在LLMs中用于药物副作用检索 

**Authors**: Shad Nygren, Pinar Avci, Andre Daniels, Reza Rassol, Afshin Beheshti, Diego Galeano  

**Link**: [PDF](https://arxiv.org/pdf/2507.13822)  

**Abstract**: Drug side effects are a major global health concern, necessitating advanced methods for their accurate detection and analysis. While Large Language Models (LLMs) offer promising conversational interfaces, their inherent limitations, including reliance on black-box training data, susceptibility to hallucinations, and lack of domain-specific knowledge, hinder their reliability in specialized fields like pharmacovigilance. To address this gap, we propose two architectures: Retrieval-Augmented Generation (RAG) and GraphRAG, which integrate comprehensive drug side effect knowledge into a Llama 3 8B language model. Through extensive evaluations on 19,520 drug side effect associations (covering 976 drugs and 3,851 side effect terms), our results demonstrate that GraphRAG achieves near-perfect accuracy in drug side effect retrieval. This framework offers a highly accurate and scalable solution, signifying a significant advancement in leveraging LLMs for critical pharmacovigilance applications. 

**Abstract (ZH)**: 药物副作用是全球健康的重大关切，亟需先进的方法用于其准确检测和分析。虽然大型语言模型（LLMs）提供了有前景的对话界面，但它们的固有限制，包括依赖于黑盒训练数据、易产生幻觉以及缺乏特定领域的知识， hindering其在像药物警戒这样专门领域的可靠性。为了解决这一问题，我们提出了两种架构：检索增强生成（RAG）和GraphRAG，这些架构将全面的药物副作用知识整合到了Llama 3 8B语言模型中。通过在19,520个药物副作用关联（涵盖976种药物和3,851种副作用术语）上的广泛评估，我们的结果显示GraphRAG在药物副作用检索方面达到了近乎完美的准确度。该框架提供了高度准确且可扩展的解决方案，标志着在利用LLMs进行关键药物警戒应用方面取得了重要进展。 

---
# LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues 

**Title (ZH)**: LoopServe：一种适应性双阶段大语言模型推理加速系统用于多轮对话 

**Authors**: Haoyang Li, Zhanchao Xu, Yiming Li, Xuejia Chen, Darian Li, Anxin Tian, Qingfa Xiao, Cheng Deng, Jun Wang, Qing Li, Lei Chen, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.13681)  

**Abstract**: Multi-turn dialogues are essential in many real-world applications of large language models, such as chatbots and virtual assistants. As conversation histories become longer, existing large language models face increasing computational and memory challenges, which hinder their ability to provide efficient and responsive interactions. Most current acceleration methods either compress the context or optimize key value caching, but they often rely on fixed or position-based heuristics that do not adapt well to the dynamic and unpredictable patterns found in actual multi-turn conversations. In this paper, we present LoopServe, an adaptive dual-phase inference acceleration framework for large language models in multi-turn dialogues. LoopServe introduces two main innovations. First, it performs online sparsification during the prefilling phase by dynamically selecting the most important parts of the attention matrix for each new input. Second, it uses progressive key value compression during decoding by adaptively maintaining a relevant and efficient cache based on the most recently generated output tokens. We also propose a \href{this https URL}{new benchmark} with eleven multi-turn datasets that reflect realistic query positions and conversational dependencies. Extensive experiments demonstrate that LoopServe consistently achieves superior effectiveness compared to existing baselines and significantly accelerates LLM inference across a wide range of long-context dialogue tasks. 

**Abstract (ZH)**: 多轮对话是大型语言模型在聊天机器人和虚拟助手等许多现实世界应用中的 Essential 组件。随着对话历史的延长，现有大型语言模型面临着日益增加的计算和内存挑战，这妨碍了它们提供高效和及时的交互。大多数当前的加速方法要么压缩上下文，要么优化键值缓存，但它们往往依赖于固定或位置基的启发式方法，这些方法不能很好地适应实际多轮对话中动态和不可预测的模式。本文提出了 LoopServe，一种适应性的双阶段推理加速框架，用于多轮对话中的大型语言模型。LoopServe 引入了两项主要创新。首先，在预填充阶段通过动态选择每个新输入的最相关部分来执行在线稀疏化。其次，在解码过程中通过基于最近生成的输出令牌动态维护相关且高效的缓存来进行渐进的键值压缩。我们还提出了一种新的基准，包含十个反映实际查询位置和对话依赖性的多轮数据集。广泛的实验表明，LoopServe 在各种长上下文对话任务中的一致性有效性优于现有基线，并且显著加速了大型语言模型的推理。 

---
# A Comprehensive Review of Transformer-based language models for Protein Sequence Analysis and Design 

**Title (ZH)**: 基于Transformer的蛋白质序列分析与设计综述 

**Authors**: Nimisha Ghosh, Daniele Santoni, Debaleena Nawn, Eleonora Ottaviani, Giovanni Felici  

**Link**: [PDF](https://arxiv.org/pdf/2507.13646)  

**Abstract**: The impact of Transformer-based language models has been unprecedented in Natural Language Processing (NLP). The success of such models has also led to their adoption in other fields including bioinformatics. Taking this into account, this paper discusses recent advances in Transformer-based models for protein sequence analysis and design. In this review, we have discussed and analysed a significant number of works pertaining to such applications. These applications encompass gene ontology, functional and structural protein identification, generation of de novo proteins and binding of proteins. We attempt to shed light on the strength and weaknesses of the discussed works to provide a comprehensive insight to readers. Finally, we highlight shortcomings in existing research and explore potential avenues for future developments. We believe that this review will help researchers working in this field to have an overall idea of the state of the art in this field, and to orient their future studies. 

**Abstract (ZH)**: 基于变换器的语言模型在自然语言处理领域的影响力是前所未有的。鉴于这种成功，这类模型已被应用于包括生物信息学在内的其他领域。本文讨论了基于变换器模型在蛋白质序列分析和设计领域的最新进展。在这篇综述中，我们讨论和分析了大量相关研究工作，这些应用涵盖了基因本体论、功能性及结构蛋白质的识别、从头设计蛋白质以及蛋白质结合等方面。我们试图揭示所讨论工作的优势和劣势，以提供全面的洞察。最后，我们指出了现有研究中的不足，并探讨了未来发展的可能途径。我们相信，这篇综述将有助于该领域研究人员了解该领域的最新进展，并引导他们未来的研究方向。 

---
# Large Language Models in Cybersecurity: Applications, Vulnerabilities, and Defense Techniques 

**Title (ZH)**: 大型语言模型在网络安全中的应用、脆弱性及防御技术 

**Authors**: Niveen O. Jaffal, Mohammed Alkhanafseh, David Mohaisen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13629)  

**Abstract**: Large Language Models (LLMs) are transforming cybersecurity by enabling intelligent, adaptive, and automated approaches to threat detection, vulnerability assessment, and incident response. With their advanced language understanding and contextual reasoning, LLMs surpass traditional methods in tackling challenges across domains such as IoT, blockchain, and hardware security. This survey provides a comprehensive overview of LLM applications in cybersecurity, focusing on two core areas: (1) the integration of LLMs into key cybersecurity domains, and (2) the vulnerabilities of LLMs themselves, along with mitigation strategies. By synthesizing recent advancements and identifying key limitations, this work offers practical insights and strategic recommendations for leveraging LLMs to build secure, scalable, and future-ready cyber defense systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在通过实现智能化、适应性和自动化的威胁检测、漏洞评估和事件响应方法来转型网络信息安全。本文综述了LLMs在网络安全领域的应用，重点探讨两个核心方面：（1）LLMs在关键网络安全领域的集成，以及（2）LLMs自身的漏洞和缓解策略。通过整合近期进展并识别关键限制，本文提供了实用的见解和战略建议，以利用LLMs构建安全、可扩展且面向未来的网络防御系统。 

---
# Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters 

**Title (ZH)**: Seed-X: 构建基于7B参数的强健多语言翻译大型语言模型 

**Authors**: Shanbo Cheng, Yu Bao, Qian Cao, Luyang Huang, Liyan Kang, Zhicheng Liu, Yu Lu, Wenhao Zhu, Zhichao Huang, Tao Li, Sitong Liu, Ningxin Peng, Shuaijie She, Lu Xu, Nuo Xu, Sen Yang, Runsheng Yu, Yiming Yu, Liehao Zou, Hang Li, Lu Lu, Yuxuan Wang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13618)  

**Abstract**: Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications. 

**Abstract (ZH)**: 多语言翻译是大型语言模型面临的挑战任务，涉及复杂语言模式和自动化翻译中出现的生硬翻译。本文介绍了Seed-X，这是一个由指令模型和推理模型组成的开源大型语言模型系列，通过7B参数规模扩展了翻译能力。基础模型在涵盖28种语言的单语和双语高质量数据集上预先训练，充分利用了多语言数据的全部潜力。随后，通过基于推理的链式思维（CoT）微调指令模型，并通过强化学习进一步优化，以实现更好的跨语言对泛化能力。Seed-X在28种语言上的性能与领先的闭源模型Gemini-2.5和GPT-4o相当，并在自动评价指标和人工评价中显著优于更大的开源模型。通过我们的优化过程分享最佳实践，并公开提供参数，以促进翻译研究和应用的发展。 

---
# Linguistic and Embedding-Based Profiling of Texts generated by Humans and Large Language Models 

**Title (ZH)**: 基于语言和嵌入的文本生成者及其大型语言模型生成文本的特征分析 

**Authors**: Sergio E. Zanotto, Segun Aroyehun  

**Link**: [PDF](https://arxiv.org/pdf/2507.13614)  

**Abstract**: The rapid advancements in large language models (LLMs) have significantly improved their ability to generate natural language, making texts generated by LLMs increasingly indistinguishable from human-written texts. While recent research has primarily focused on using LLMs to classify text as either human-written and machine-generated texts, our study focus on characterizing these texts using a set of linguistic features across different linguistic levels such as morphology, syntax, and semantics. We select a dataset of human-written and machine-generated texts spanning 8 domains and produced by 11 different LLMs. We calculate different linguistic features such as dependency length and emotionality and we use them for characterizing human-written and machine-generated texts along with different sampling strategies, repetition controls and model release date. Our statistical analysis reveals that human-written texts tend to exhibit simpler syntactic structures and more diverse semantic content. Furthermore, we calculate the variability of our set of features across models and domains. Both human and machine texts show stylistic diversity across domains, with humans displaying greater variation in our features. Finally, we apply style embeddings to further test variability among human-written and machine-generated texts. Notably, newer models output text that is similarly variable, pointing to an homogenization of machine-generated texts. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展显著提高了其生成自然语言的能力，使得LLMs生成的文本越来越难以与人类撰写的文本区分开来。虽然近期的研究主要集中在使用LLMs对文本进行分类，判断其为人类还是机器生成，我们的研究重点在于使用一系列语言特征从不同的语言层次（如词形学、句法和语义）对这些文本进行表征。我们选取了来自8个领域、由11种不同LLMs生成的人类撰写的和机器生成的文本数据集。我们计算了不同的语言特征（如依存长度和情感性），并利用这些特征以及不同的采样策略、重复控制和模型发布时间对人类撰写的和机器生成的文本进行表征。我们的统计分析表明，人类撰写的文本通常表现出相对简单的句法结构和更为多样的语义内容。此外，我们计算了我们特征集在不同模型和领域之间的变异性。人类和机器生成的文本在不同领域都展示了风格多样性，但人类的多样性更为显著。最后，我们应用风格嵌入进一步测试人类撰写的和机器生成的文本之间的变异性。值得注意的是，较新的模型生成的文本显示出类似的变异性，这表明机器生成的文本正在趋于同质化。 

---
# GIFT: Gradient-aware Immunization of diffusion models against malicious Fine-Tuning with safe concepts retention 

**Title (ZH)**: GIFT： Gradient-aware对抗恶意Fine-Tuning的扩散模型免疫方法，同时保留安全概念 

**Authors**: Amro Abdalla, Ismail Shaheen, Dan DeGenaro, Rupayan Mallick, Bogdan Raita, Sarah Adel Bargal  

**Link**: [PDF](https://arxiv.org/pdf/2507.13598)  

**Abstract**: We present \textbf{GIFT}: a \textbf{G}radient-aware \textbf{I}mmunization technique to defend diffusion models against malicious \textbf{F}ine-\textbf{T}uning while preserving their ability to generate safe content. Existing safety mechanisms like safety checkers are easily bypassed, and concept erasure methods fail under adversarial fine-tuning. GIFT addresses this by framing immunization as a bi-level optimization problem: the upper-level objective degrades the model's ability to represent harmful concepts using representation noising and maximization, while the lower-level objective preserves performance on safe data. GIFT achieves robust resistance to malicious fine-tuning while maintaining safe generative quality. Experimental results show that our method significantly impairs the model's ability to re-learn harmful concepts while maintaining performance on safe content, offering a promising direction for creating inherently safer generative models resistant to adversarial fine-tuning attacks.
{\small\textbf{\textcolor{red}{Warning: This paper contains NSFW content. Reader discretion is advised.}}} 

**Abstract (ZH)**: 我们提出\[GIFT\]：一种在不牺牲生成安全内容能力的前提下，针对恶意细调保护扩散模型的渐进免疫技术。现有的安全性机制如安全性检查器容易被绕过，而概念抹除方法在对抗性细调下失效。GIFT 通过将免疫视为两层优化问题来解决这一问题：上层目标通过表征噪声和最大化降低模型表示有害概念的能力，而下层目标则在安全数据上保持模型性能。GIFT 在保持生成安全内容的质量的同时，提供了对恶意细调攻击的稳健抵抗。实验结果表明，我们的方法显著削弱了模型重新学习有害概念的能力，同时在安全内容上保持了性能，为创建固有安全且对对抗性细调攻击具有抵抗力的生成模型指明了前景。注意：\[此论文包含不适合公开的内容。\]请谨慎阅读。 

---
# Learning Pluralistic User Preferences through Reinforcement Learning Fine-tuned Summaries 

**Title (ZH)**: 通过强化学习微调摘要学习多元用户偏好 

**Authors**: Hyunji Nam, Yanming Wan, Mickel Liu, Jianxun Lian, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2507.13579)  

**Abstract**: As everyday use cases of large language model (LLM) AI assistants have expanded, it is becoming increasingly important to personalize responses to align to different users' preferences and goals. While reinforcement learning from human feedback (RLHF) is effective at improving LLMs to be generally more helpful and fluent, it does not account for variability across users, as it models the entire user population with a single reward model. We present a novel framework, Preference Learning Using Summarization (PLUS), that learns text-based summaries of each user's preferences, characteristics, and past conversations. These summaries condition the reward model, enabling it to make personalized predictions about the types of responses valued by each user. We train the user-summarization model with reinforcement learning, and update the reward model simultaneously, creating an online co-adaptation loop. We show that in contrast with prior personalized RLHF techniques or with in-context learning of user information, summaries produced by PLUS capture meaningful aspects of a user's preferences. Across different pluralistic user datasets, we show that our method is robust to new users and diverse conversation topics. Additionally, we demonstrate that the textual summaries generated about users can be transferred for zero-shot personalization of stronger, proprietary models like GPT-4. The resulting user summaries are not only concise and portable, they are easy for users to interpret and modify, allowing for more transparency and user control in LLM alignment. 

**Abstract (ZH)**: 基于总结的偏好学习框架 (PLUS): 个性化大型语言模型辅助响应的新方法 

---
# Apple Intelligence Foundation Language Models: Tech Report 2025 

**Title (ZH)**: 苹果智能基础语言模型：技术报告2025 

**Authors**: Hanzhi Zhou, Erik Hornberger, Pengsheng Guo, Xiyou Zhou, Saiwen Wang, Xin Wang, Yifei He, Xuankai Chang, Rene Rauch, Louis D'hauwe, John Peebles, Alec Doane, Kohen Chia, Jenna Thibodeau, Zi-Yi Dou, Yuanyang Zhang, Ruoming Pang, Reed Li, Zhifeng Chen, Jeremy Warner, Zhaoyang Xu, Sophy Lee, David Mizrahi, Ramsey Tantawi, Chris Chaney, Kelsey Peterson, Jun Qin, Alex Dombrowski, Mira Chiang, Aiswarya Raghavan, Gerard Casamayor, Qibin Chen, Aonan Zhang, Nathalie Tran, Jianyu Wang, Hang Su, Thomas Voice, Alessandro Pappalardo, Brycen Wershing, Prasanth Yadla, Rui Li, Priyal Chhatrapati, Ismael Fernandez, Yusuf Goren, Xin Zheng, Forrest Huang, Tao Lei, Eray Yildiz, Alper Kokmen, Gokul Santhanam, Areeba Kamal, Kaan Elgin, Dian Ang Yap, Jeremy Liu, Peter Gray, Howard Xing, Kieran Liu, Matteo Ronchi, Moritz Schwarzer-Becker, Yun Zhu, Mandana Saebi, Jeremy Snow, David Griffiths, Guillaume Tartavel, Erin Feldman, Simon Lehnerer, Fernando Bermúdez-Medina, Hans Han, Joe Zhou, Xiaoyi Ren, Sujeeth Reddy, Zirui Wang, Tom Gunter, Albert Antony, Yuanzhi Li, John Dennison, Tony Sun, Yena Han, Yi Qin, Sam Davarnia, Jeffrey Bigham, Wayne Shan, Hannah Gillis Coleman, Guillaume Klein, Peng Liu, Muyang Yu, Jack Cackler, Yuan Gao, Crystal Xiao, Binazir Karimzadeh, Zhengdong Zhang, Felix Bai, Albin Madappally Jose, Feng Nan, Nazir Kamaldin, Dong Yin, Hans Hao, Yanchao Sun, Yi Hua, Charles Maalouf  

**Link**: [PDF](https://arxiv.org/pdf/2507.13575)  

**Abstract**: We introduce two multilingual, multimodal foundation language models that power Apple Intelligence features across Apple devices and services: i a 3B-parameter on-device model optimized for Apple silicon through architectural innovations such as KV-cache sharing and 2-bit quantization-aware training; and ii a scalable server model built on a novel Parallel-Track Mixture-of-Experts PT-MoE transformer that combines track parallelism, mixture-of-experts sparse computation, and interleaved global-local attention to deliver high quality with competitive cost on Apple's Private Cloud Compute platform. Both models are trained on large-scale multilingual and multimodal datasets sourced via responsible web crawling, licensed corpora, and high-quality synthetic data, then further refined with supervised fine-tuning and reinforcement learning on a new asynchronous platform. The resulting models support several additional languages while understanding images and executing tool calls. In public benchmarks and human evaluations, both the server model and the on-device model match or surpass comparably sized open baselines.
A new Swift-centric Foundation Models framework exposes guided generation, constrained tool calling, and LoRA adapter fine-tuning, allowing developers to integrate these capabilities with a few lines of code. The latest advancements in Apple Intelligence models are grounded in our Responsible AI approach with safeguards like content filtering and locale-specific evaluation, as well as our commitment to protecting our users' privacy with innovations like Private Cloud Compute. 

**Abstract (ZH)**: 我们介绍了两个跨多Apple设备和服务实现Apple智能功能的多语言、多模态基础语言模型：一、一种通过如KV-cache共享和2比特量化训练等架构创新优化的3B参数量端侧模型；二、一种基于新颖的Parallel-Track Mixture-of-Experts PT-MoE变换器的大规模可扩展服务器模型，该模型结合了轨道并行性、专家混合稀疏计算和交织的全局-局部注意力机制，在Apple私有云计算平台上提供高质量的同时保持竞争力的成本。这两类模型均在负责任的网络爬取、许可语料库和高质量合成数据集上进行大规模多语言和多模态训练，然后通过新的异步平台进行监督微调和强化学习进一步优化。最终生成的模型不仅支持多种额外语言，还能够理解和执行工具调用。在公开基准测试和人类评估中，服务器模型和端侧模型均与同等规模的开源基线模型表现相当或超越。我们还推出了一种以Swift为中心的基础模型框架，该框架支持指导生成、受限工具调用以及LoRA适配器微调，允许开发者通过几行代码将这些能力集成到应用中。Apple Intelligence模型的最新进展基于我们负责任的人工智能方法，包括内容过滤和本地化评估等保护措施，以及通过私有云计算等方式保护用户隐私的承诺。 

---
# Aligning Knowledge Graphs and Language Models for Factual Accuracy 

**Title (ZH)**: 知识图谱与语言模型的协同以提高事实准确性 

**Authors**: Nur A Zarin Nishat, Andrea Coletta, Luigi Bellomarini, Kossi Amouzouvi, Jens Lehmann, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2507.13411)  

**Abstract**: Large language models like GPT-4, Gemini, and Claude have transformed natural language processing (NLP) tasks such as question answering, dialogue generation, summarization, and so forth; yet their susceptibility to hallucination stands as one of the major challenges. Among numerous approaches to overcome this challenge, integration of Knowledge Graphs (KGs) into language models has emerged as a promising solution as it provides structured, reliable, domain-specific, and up-to-date external information to the language models. In this paper, we introduce ALIGNed-LLM, a simple yet effective approach to improve language models' factuality via a lean strategy to infuse KGs into the latent space of language models inspired by LLaVA where visual and textual information is infused. We use embeddings from a pre-trained Knowledge Graph Embedding (KGE) model, such as TransE, and a trainable projection layer to align entity and text embeddings. This alignment enables the language model to distinguish between similar entities improving factual grounding and reducing hallucination. We tested our approach on three popular questions-answering benchmark datasets alongside language models of varying sizes, showing significant improvement. Furthermore, we applied our approach to a real-world financial use case from a large central bank in Europe, which demands high accuracy and precision, demonstrating a substantial improvement of the LLM answers. 

**Abstract (ZH)**: 大型语言模型如GPT-4、Gemini和Claude已经Transformed自然语言处理(NLP)任务，如问答、对话生成和总结等；然而，它们容易出现幻觉仍然是一个主要挑战。为了克服这一挑战，将知识图谱(KGs)集成到语言模型中已成为一种有前途的解决方案，因为它为语言模型提供了结构化、可靠、领域特定且最新的外部信息。在本文中，我们介绍了通过借鉴LLaVA中的策略，将知识图谱lean地融入语言模型的潜空间中，以提高语言模型的事实性的一种简单而有效的方法——ALIGNed-LLM。我们使用预训练的知识图谱嵌入(KGE)模型（如TransE）的嵌入和可训练的投影层来对齐实体和文本嵌入。这种对齐使得语言模型能够区分相似的实体，从而提高事实关联并减少幻觉。我们通过使用不同规模的语言模型在三个流行的问答基准数据集上测试了这种方法，显示出了显著的改进。此外，我们将这种方法应用于欧洲一家大型中央银行的实际金融案例中，该案例要求高准确性和精确度，验证了语言模型答案的显著改进。 

---
# Causal Language Control in Multilingual Transformers via Sparse Feature Steering 

**Title (ZH)**: 多语言Transformer中稀疏特征导向的因果语言控制 

**Authors**: Cheng-Ting Chou, George Liu, Jessica Sun, Cole Blondin, Kevin Zhu, Vasu Sharma, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2507.13410)  

**Abstract**: Deterministically controlling the target generation language of large multilingual language models (LLMs) remains a fundamental challenge, particularly in zero-shot settings where neither explicit language prompts nor fine-tuning are available. In this work, we investigate whether sparse autoencoder (SAE) features, previously shown to correlate with interpretable model behaviors, can be leveraged to steer the generated language of LLMs during inference. Leveraging pretrained SAEs on the residual streams of Gemma-2B and Gemma-9B, we identify features whose activations differ most significantly between English and four target languages: Chinese, Japanese, Spanish, and French. By modifying just a single SAE feature at one transformer layer, we achieve controlled language shifts with up to 90\% success, as measured by FastText language classification, while preserving semantic fidelity according to LaBSE (Language-Agnostic BERT Sentence Embedding) similarity. Our analysis reveals that language steering is most effective in mid-to-late transformer layers and is amplified by specific attention heads disproportionately associated with language-sensitive SAE features. These results demonstrate the promise of sparse feature steering as a lightweight and interpretable mechanism for controllable multilingual generation. 

**Abstract (ZH)**: 确定性控制大型多语言语言模型的生成目标语言 remain a fundamental challenge, particularly in zero-shot settings where neither explicit language prompts nor fine-tuning are available. 在这项工作中，我们调查了稀疏自动编码器（SAE）特征是否可以利用来引导推理过程中生成的语言。利用预训练的SAE在Gemma-2B和Gemma-9B的残差流上，我们识别出在英语与其他四种目标语言（中文、日语、西班牙语、法语）之间激活差异最大的特征。通过在单个变换器层中修改一个SAE特征，我们实现了高达90%的成功语言转向，这通过FastText语言分类衡量，同时根据LaBSE（无语言偏见的BERT句子嵌入）相似性保持语义保真度。我们的分析表明，语言引导在中间到后期的变换器层最有效，并且与语言敏感SAE特征关联的特定注意头起到了增强作用。这些结果展示了稀疏特征引导作为轻量级且可解释的机制以实现可控多语言生成的潜力。 

---
# Persona-Based Synthetic Data Generation Using Multi-Stage Conditioning with Large Language Models for Emotion Recognition 

**Title (ZH)**: 基于人物身份的多阶段条件生成合成数据以大型语言模型辅助情感识别 

**Authors**: Keito Inoshita, Rushia Harada  

**Link**: [PDF](https://arxiv.org/pdf/2507.13380)  

**Abstract**: In the field of emotion recognition, the development of high-performance models remains a challenge due to the scarcity of high-quality, diverse emotional datasets. Emotional expressions are inherently subjective, shaped by individual personality traits, socio-cultural backgrounds, and contextual factors, making large-scale, generalizable data collection both ethically and practically difficult. To address this issue, we introduce PersonaGen, a novel framework for generating emotionally rich text using a Large Language Model (LLM) through multi-stage persona-based conditioning. PersonaGen constructs layered virtual personas by combining demographic attributes, socio-cultural backgrounds, and detailed situational contexts, which are then used to guide emotion expression generation. We conduct comprehensive evaluations of the generated synthetic data, assessing semantic diversity through clustering and distributional metrics, human-likeness via LLM-based quality scoring, realism through comparison with real-world emotion corpora, and practical utility in downstream emotion classification tasks. Experimental results show that PersonaGen significantly outperforms baseline methods in generating diverse, coherent, and discriminative emotion expressions, demonstrating its potential as a robust alternative for augmenting or replacing real-world emotional datasets. 

**Abstract (ZH)**: 在情感识别领域，由于高质量、多样化的情感数据集稀缺，高性能模型的发展仍是一项挑战。情感表达本质上是主观的，受到个体性格特征、社会文化背景和情境因素的影响，使得大规模、可泛化的数据收集在伦理和实践中都十分困难。为解决这一问题，我们提出了 PersonaGen，一个通过多阶段人格导向条件化使用大型语言模型（LLM）生成丰富情感文本的新框架。PersonaGen 通过结合人口统计属性、社会文化背景和详细的情境上下文构建分层的虚拟人格，并用于引导情感表达生成。我们对生成的合成数据进行了全面评估，通过聚类和分布度量评估语义多样性，通过基于 LLM 的质量评分评估人类相似度，通过与现实世界情感语料库的对比评估真实性，并在下游情感分类任务中评估其实用性。实验结果表明，PersonaGen 在生成多样化、连贯且区分性强的情感表达方面显著优于baseline方法，展示了其作为增强或替代现实世界情感数据集的稳健替代方案的潜力。 

---
# Smart Routing for Multimodal Video Retrieval: When to Search What 

**Title (ZH)**: 多模态视频检索中的智能路由：何时搜索何种内容 

**Authors**: Kevin Dela Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2507.13374)  

**Abstract**: We introduce ModaRoute, an LLM-based intelligent routing system that dynamically selects optimal modalities for multimodal video retrieval. While dense text captions can achieve 75.9% Recall@5, they require expensive offline processing and miss critical visual information present in 34% of clips with scene text not captured by ASR. By analyzing query intent and predicting information needs, ModaRoute reduces computational overhead by 41% while achieving 60.9% Recall@5. Our approach uses GPT-4.1 to route queries across ASR (speech), OCR (text), and visual indices, averaging 1.78 modalities per query versus exhaustive 3.0 modality search. Evaluation on 1.8M video clips demonstrates that intelligent routing provides a practical solution for scaling multimodal retrieval systems, reducing infrastructure costs while maintaining competitive effectiveness for real-world deployment. 

**Abstract (ZH)**: 基于LLM的智能路由系统ModaRoute及其在多媒体视频检索中的应用 

---
# VerilogDB: The Largest, Highest-Quality Dataset with a Preprocessing Framework for LLM-based RTL Generation 

**Title (ZH)**: VerilogDB：基于预处理框架的 Largest 和最高质量的数据集，用于 LLM 基本的 RTL 生成 

**Authors**: Paul E. Calzada, Zahin Ibnat, Tanvir Rahman, Kamal Kandula, Danyu Lu, Sujan Kumar Saha, Farimah Farahmandi, Mark Tehranipoor  

**Link**: [PDF](https://arxiv.org/pdf/2507.13369)  

**Abstract**: Large Language Models (LLMs) are gaining popularity for hardware design automation, particularly through Register Transfer Level (RTL) code generation. In this work, we examine the current literature on RTL generation using LLMs and identify key requirements for training and fine-tuning datasets. We construct a robust Verilog dataset through an automated three-pronged process involving database (DB) creation and management with PostgreSQL, data collection from code hosting sites like OpenCores and GitHub, and data preprocessing to verify the codes' syntax, run logic synthesis, and extract relevant module metadata. We implement a scalable and efficient DB infrastructure to support analysis and detail our preprocessing pipeline to enforce high-quality data before DB insertion. The resulting dataset comprises 20,392 Verilog samples, 751 MB of Verilog code data, which is the largest high-quality Verilog dataset for LLM fine-tuning to our knowledge. We further evaluate the dataset, address associated challenges, and explore potential applications for future research and development in LLM-based hardware generation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在硬件设计自动化中的流行尤其体现在通过寄存器传输级（RTL）代码生成。本文审查了使用LLMs进行RTL生成的相关文献，并确定了训练和微调数据集的关键要求。我们通过一个包含数据库（DB）创建和管理、从代码托管网站如OpenCores和GitHub收集数据、以及进行数据预处理以验证代码语法、运行逻辑综合和提取相关模块元数据的自动化三步过程，构建了一个稳健的Verilog数据集。我们实现了一个可扩展且高效的DB基础设施，以支持数据分析，并详细描述了预处理流水线以确保高质量数据在数据集插入之前高标准的数据质量。最终得到的数据集包含20,392个Verilog样本，751 MB的Verilog代码数据，据我们所知，这是用于LLM微调的最大高质量Verilog数据集。我们进一步评估了数据集，解决了相关挑战，并探讨了潜在的应用，以推动基于LLM的硬件生成的未来研究与开发。 

---
# VLMs have Tunnel Vision: Evaluating Nonlocal Visual Reasoning in Leading VLMs 

**Title (ZH)**: VLMs具有隧道视域：评估领先VLMs的非局部视觉推理能力 

**Authors**: Shmuel Berman, Jia Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.13361)  

**Abstract**: Visual Language Models (VLMs) excel at complex visual tasks such as VQA and chart understanding, yet recent work suggests they struggle with simple perceptual tests. We present an evaluation that tests vision-language models' capacity for nonlocal visual reasoning -- reasoning that requires chaining evidence collected from multiple, possibly distant, regions of an image. We isolate three distinct forms of non-local vision: comparative perception, which demands holding two images in working memory and comparing them; saccadic search, which requires making discrete, evidence-driven jumps to locate successive targets; and smooth visual search, which involves searching smoothly along a continuous contour. Flagship models (e.g., Gemini 2.5 Pro, Claude Vision 3.7, GPT-o4-mini), even those that perform well on prior primitive-vision benchmarks, fail these tests and barely exceed random accuracy on two variants of our tasks that are trivial for humans. Our structured evaluation suite allows us to test if VLMs can perform similar visual algorithms to humans. Our findings show that despite gains in raw visual acuity, current models lack core visual reasoning capabilities. 

**Abstract (ZH)**: 视觉语言模型在复杂视觉任务如VQA和图表理解方面表现优异，然而近期研究表明它们在简单的知觉测试中存在困难。我们提出了一项评估，测试视觉-语言模型在非局部视觉推理方面的能力——这种推理要求结合从图像中多个可能分散的区域收集的证据。我们将非局部视觉划分为三种不同的形式：比较感知，要求在工作记忆中保存两张图片并对比它们；凝视搜索，需要作出证据驱动的跳跃以定位依次出现的目标；平滑视觉搜索，涉及沿连续轮廓进行平滑搜索。即便是在先前基本视觉基准测试中表现出色的旗舰模型（如Gemini 2.5 Pro、Claude Vision 3.7、GPT-o4-mini），在这些测试中也失败了，甚至在我们设计的两种人类操作极其简单的任务变体中仅略微超过了随机准确性。我们结构化的评估套件允许我们测试视觉语言模型是否能够执行类似于人类的视觉算法。我们的研究发现表明，尽管视觉敏锐度有所提高，当前模型缺乏核心的视觉推理能力。 

---
# Physical models realizing the transformer architecture of large language models 

**Title (ZH)**: 物理模型实现大型语言模型的变压器架构 

**Authors**: Zeqian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13354)  

**Abstract**: The introduction of the transformer architecture in 2017 (cf.\cite{VSP2017}) marked the most striking advancement in natural language processing. The transformer is a model architecture relying entirely on an attention mechanism to draw global dependencies between input and output. However, we believe there is a gap in our theoretical understanding of what the transformer is, and why it works physically. In this paper, from a physical perspective on modern chips, we construct physical models in the Fock space over the Hilbert space of tokens realizing large language models based on a transformer architecture as open quantum systems. Our physical models underlie the transformer architecture for large language models. 

**Abstract (ZH)**: 变压器架构引入（参见\cite{VSP2017}，2017年）标志着自然语言处理领域最显著的进展。从现代芯片的物理视角出发，我们构建了在标记希尔伯特空间上实现基于变压器架构的大语言模型的费米子空间中的物理模型，将其视为开放量子系统。这些物理模型构成了大语言模型变压器架构的基础。 

---
