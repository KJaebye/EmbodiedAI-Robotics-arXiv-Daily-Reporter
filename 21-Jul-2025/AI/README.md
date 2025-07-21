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
# Generative AI-Driven High-Fidelity Human Motion Simulation 

**Title (ZH)**: 由生成式AI驱动的高保真人体运动模拟 

**Authors**: Hari Iyer, Neel Macwan, Atharva Jitendra Hude, Heejin Jeong, Shenghan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.14097)  

**Abstract**: Human motion simulation (HMS) supports cost-effective evaluation of worker behavior, safety, and productivity in industrial tasks. However, existing methods often suffer from low motion fidelity. This study introduces Generative-AI-Enabled HMS (G-AI-HMS), which integrates text-to-text and text-to-motion models to enhance simulation quality for physical tasks. G-AI-HMS tackles two key challenges: (1) translating task descriptions into motion-aware language using Large Language Models aligned with MotionGPT's training vocabulary, and (2) validating AI-enhanced motions against real human movements using computer vision. Posture estimation algorithms are applied to real-time videos to extract joint landmarks, and motion similarity metrics are used to compare them with AI-enhanced sequences. In a case study involving eight tasks, the AI-enhanced motions showed lower error than human created descriptions in most scenarios, performing better in six tasks based on spatial accuracy, four tasks based on alignment after pose normalization, and seven tasks based on overall temporal similarity. Statistical analysis showed that AI-enhanced prompts significantly (p $<$ 0.0001) reduced joint error and temporal misalignment while retaining comparable posture accuracy. 

**Abstract (ZH)**: 基于生成AI的Humanmotion仿真：提高工业任务中动作保真度的方法 

---
# Glucose-ML: A collection of longitudinal diabetes datasets for development of robust AI solutions 

**Title (ZH)**: Glucose-ML: 一种用于开发稳健AI解决方案的纵向糖尿病数据集合 

**Authors**: Temiloluwa Prioleau, Baiying Lu, Yanjun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.14077)  

**Abstract**: Artificial intelligence (AI) algorithms are a critical part of state-of-the-art digital health technology for diabetes management. Yet, access to large high-quality datasets is creating barriers that impede development of robust AI solutions. To accelerate development of transparent, reproducible, and robust AI solutions, we present Glucose-ML, a collection of 10 publicly available diabetes datasets, released within the last 7 years (i.e., 2018 - 2025). The Glucose-ML collection comprises over 300,000 days of continuous glucose monitor (CGM) data with a total of 38 million glucose samples collected from 2500+ people across 4 countries. Participants include persons living with type 1 diabetes, type 2 diabetes, prediabetes, and no diabetes. To support researchers and innovators with using this rich collection of diabetes datasets, we present a comparative analysis to guide algorithm developers with data selection. Additionally, we conduct a case study for the task of blood glucose prediction - one of the most common AI tasks within the field. Through this case study, we provide a benchmark for short-term blood glucose prediction across all 10 publicly available diabetes datasets within the Glucose-ML collection. We show that the same algorithm can have significantly different prediction results when developed/evaluated with different datasets. Findings from this study are then used to inform recommendations for developing robust AI solutions within the diabetes or broader health domain. We provide direct links to each longitudinal diabetes dataset in the Glucose-ML collection and openly provide our code. 

**Abstract (ZH)**: 人工 intelligence (AI) 算法是糖尿病管理最新数字健康技术中的关键组成部分。然而，获取大型高质量数据集正成为阻碍稳健AI解决方案开发的障碍。为了加速开发透明、可 reproducing 和稳健的AI解决方案，我们提供了Glucose-ML，一个包含过去7年（即2018-2025）发布的10个公开可用的糖尿病数据集的集合。Glucose-ML集合包含了超过30万个连续葡萄糖监测（CGM）数据日，共有3800万个葡萄糖样本，来自来自4个国家的2500多名参与者，包括1型糖尿病患者、2型糖尿病患者、糖尿病前期患者及无糖尿病患者。为了支持研究人员和创新者使用这一丰富的糖尿病数据集集合，我们提供了一项比较分析，以指导算法开发者进行数据选择。此外，我们还进行了一项用于血糖预测的任务案例研究——这是该领域最常见的AI任务之一。通过这项案例研究，我们为Glucose-ML集合中的所有10个公开可用的糖尿病数据集提供了短期血糖预测基准。研究结果表明，相同的算法在使用不同数据集开发和评估时，其预测结果可以显著不同。本研究的发现被用于制定在糖尿病或更广泛的健康领域内开发稳健AI解决方案的建议。我们直接提供了每个纵向糖尿病数据集的链接，并公开提供了我们的代码。 

---
# KROMA: Ontology Matching with Knowledge Retrieval and Large Language Models 

**Title (ZH)**: KROMA：基于知识检索和大规模语言模型的本体匹配 

**Authors**: Lam Nguyen, Erika Barcelos, Roger French, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14032)  

**Abstract**: Ontology Matching (OM) is a cornerstone task of semantic interoperability, yet existing systems often rely on handcrafted rules or specialized models with limited adaptability. We present KROMA, a novel OM framework that harnesses Large Language Models (LLMs) within a Retrieval-Augmented Generation (RAG) pipeline to dynamically enrich the semantic context of OM tasks with structural, lexical, and definitional knowledge. To optimize both performance and efficiency, KROMA integrates a bisimilarity-based concept matching and a lightweight ontology refinement step, which prune candidate concepts and substantially reduce the communication overhead from invoking LLMs. Through experiments on multiple benchmark datasets, we show that integrating knowledge retrieval with context-augmented LLMs significantly enhances ontology matching, outperforming both classic OM systems and cutting-edge LLM-based approaches while keeping communication overhead comparable. Our study highlights the feasibility and benefit of the proposed optimization techniques (targeted knowledge retrieval, prompt enrichment, and ontology refinement) for ontology matching at scale. 

**Abstract (ZH)**: 基于大型语言模型的检索增强生成框架KROMA在本体匹配中的应用 

---
# Towards Constraint Temporal Answer Set Programming 

**Title (ZH)**: 面向约束时态回答集编程 

**Authors**: Pedro Cabalar, Martín Diéguez, François Olivier, Torsten Schaub, Igor Stéphan  

**Link**: [PDF](https://arxiv.org/pdf/2507.13958)  

**Abstract**: Reasoning about dynamic systems with a fine-grained temporal and numeric resolution presents significant challenges for logic-based approaches like Answer Set Programming (ASP). To address this, we introduce and elaborate upon a novel temporal and constraint-based extension of the logic of Here-and-There and its nonmonotonic equilibrium extension, representing, to the best of our knowledge, the first approach to nonmonotonic temporal reasoning with constraints specifically tailored for ASP. This expressive system is achieved by a synergistic combination of two foundational ASP extensions: the linear-time logic of Here-and-There, providing robust nonmonotonic temporal reasoning capabilities, and the logic of Here-and-There with constraints, enabling the direct integration and manipulation of numeric constraints, among others. This work establishes the foundational logical framework for tackling complex dynamic systems with high resolution within the ASP paradigm. 

**Abstract (ZH)**: 基于细粒度时间和数值分辨率的动态系统推理：一种解答集编程特定的非单调时序约束逻辑扩展 

---
# Cross-modal Causal Intervention for Alzheimer's Disease Prediction 

**Title (ZH)**: 阿尔茨海默病预测的跨模态因果干预 

**Authors**: Yutao Jin, Haowen Xiao, Jielei Chu, Fengmao Lv, Yuxiao Li, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.13956)  

**Abstract**: Mild Cognitive Impairment (MCI) serves as a prodromal stage of Alzheimer's Disease (AD), where early identification and intervention can effectively slow the progression to dementia. However, diagnosing AD remains a significant challenge in neurology due to the confounders caused mainly by the selection bias of multimodal data and the complex relationships between variables. To address these issues, we propose a novel visual-language causal intervention framework named Alzheimer's Disease Prediction with Cross-modal Causal Intervention (ADPC) for diagnostic assistance. Our ADPC employs large language model (LLM) to summarize clinical data under strict templates, maintaining structured text outputs even with incomplete or unevenly distributed datasets. The ADPC model utilizes Magnetic Resonance Imaging (MRI), functional MRI (fMRI) images and textual data generated by LLM to classify participants into Cognitively Normal (CN), MCI, and AD categories. Because of the presence of confounders, such as neuroimaging artifacts and age-related biomarkers, non-causal models are likely to capture spurious input-output correlations, generating less reliable results. Our framework implicitly eliminates confounders through causal intervention. Experimental results demonstrate the outstanding performance of our method in distinguishing CN/MCI/AD cases, achieving state-of-the-art (SOTA) metrics across most evaluation metrics. The study showcases the potential of integrating causal reasoning with multi-modal learning for neurological disease diagnosis. 

**Abstract (ZH)**: 轻度认知损害（MCI）作为阿尔茨海默病（AD）的前驱阶段，早期识别和干预可以有效延缓向痴呆的进展。然而，由于多模态数据的选择偏差和变量之间复杂的相互关系，阿尔茨海默病的诊断仍然是神经学上的一个重大挑战。为解决这些问题，我们提出了一种名为阿尔茨海默病预测与跨模态因果干预（ADPC）的新颖视觉-语言因果干预框架，用于诊断辅助。我们的ADPC利用大规模语言模型（LLM）在严格模板下总结临床数据，即使在不完整或分布不均的数据集下也能保持结构化的文本输出。ADPC模型利用磁共振成像（MRI）、功能性磁共振成像（fMRI）图像和LLM生成的文本数据对参与者进行分类，分为正常认知（CN）、轻度认知损害（MCI）和阿尔茨海默病（AD）类别。由于存在混杂因素，如神经成像伪影和年龄相关的生物标志物，非因果模型可能会捕捉到虚假的输入输出相关性，生成不那么可靠的结果。我们的框架通过因果干预隐式地消除了混杂因素。实验结果表明，我们的方法在区分CN/MCI/AD病例方面具有出色的性能，大多数评估指标上达到了目前最先进的（SOTA）水平。这项研究展示了结合因果推理与多模态学习在神经学疾病诊断中的潜在价值。 

---
# Large Language Models as Innovators: A Framework to Leverage Latent Space Exploration for Novelty Discovery 

**Title (ZH)**: 大型语言模型作为创新者：一个利用潜在空间探索发现新颖性的框架 

**Authors**: Mateusz Bystroński, Mikołaj Hołysz, Grzegorz Piotrowski, Nitesh V. Chawla, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2507.13874)  

**Abstract**: Innovative idea generation remains a core challenge in AI, as large language models (LLMs) often struggle to produce outputs that are both novel and relevant. Despite their fluency, LLMs tend to replicate patterns seen during training, limiting their ability to diverge creatively without extensive prompt engineering. Prior work has addressed this through domain-specific heuristics and structured prompting pipelines, but such solutions are brittle and difficult to generalize. In this paper, we propose a model-agnostic latent-space ideation framework that enables controlled, scalable creativity by navigating the continuous embedding space of ideas. Unlike prior methods, our framework requires no handcrafted rules and adapts easily to different domains, input formats, and creative tasks. This paper introduces an early-stage prototype of our method, outlining the conceptual framework and preliminary results highlighting its potential as a general-purpose co-ideator for human-AI collaboration. 

**Abstract (ZH)**: 创新想法生成仍然是AI中的核心挑战，尽管大语言模型在流畅性方面表现出色，但往往难以产生既新颖又相关的输出。尽管具有流畅性，大语言模型倾向于在训练过程中复制模式，限制了它们在无需大量提示工程的情况下进行创造性发散的能力。先前的研究通过领域特定的启发式方法和结构化提示管道来解决这一问题，但这些解决方案脆弱且难以泛化。在本文中，我们提出了一种模型无关的潜在空间ideation框架，通过导航想法的连续嵌入空间来实现受控的、可扩展的创造力。与先前的方法不同，我们的框架不需要手工艺品规则，并且可以轻松适应不同的领域、输入格式和创意任务。本文介绍了一种早期原型方法，概述了概念框架和初步结果，强调其作为人类-AI协作的一般辅助创意工具的潜力。 

---
# Causal Knowledge Transfer for Multi-Agent Reinforcement Learning in Dynamic Environments 

**Title (ZH)**: 动态环境下多代理强化学习中的因果知识迁移 

**Authors**: Kathrin Korte, Christian Medeiros Adriano, Sona Ghahremani, Holger Giese  

**Link**: [PDF](https://arxiv.org/pdf/2507.13846)  

**Abstract**: [Context] Multi-agent reinforcement learning (MARL) has achieved notable success in environments where agents must learn coordinated behaviors. However, transferring knowledge across agents remains challenging in non-stationary environments with changing goals. [Problem] Traditional knowledge transfer methods in MARL struggle to generalize, and agents often require costly retraining to adapt. [Approach] This paper introduces a causal knowledge transfer framework that enables RL agents to learn and share compact causal representations of paths within a non-stationary environment. As the environment changes (new obstacles), agents' collisions require adaptive recovery strategies. We model each collision as a causal intervention instantiated as a sequence of recovery actions (a macro) whose effect corresponds to a causal knowledge of how to circumvent the obstacle while increasing the chances of achieving the agent's goal (maximizing cumulative reward). This recovery action macro is transferred online from a second agent and is applied in a zero-shot fashion, i.e., without retraining, just by querying a lookup model with local context information (collisions). [Results] Our findings reveal two key insights: (1) agents with heterogeneous goals were able to bridge about half of the gap between random exploration and a fully retrained policy when adapting to new environments, and (2) the impact of causal knowledge transfer depends on the interplay between environment complexity and agents' heterogeneous goals. 

**Abstract (ZH)**: 多智能体强化学习中的因果知识迁移框架：在非稳定环境中的应用 

---
# When Speed meets Accuracy: an Efficient and Effective Graph Model for Temporal Link Prediction 

**Title (ZH)**: 当速度遇见准确性：一种高效且有效的图模型用于时间链接预测 

**Authors**: Haoyang Li, Yuming Xu, Yiming Li, Hanmo Liu, Darian Li, Chen Jason Zhang, Lei Chen, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.13825)  

**Abstract**: Temporal link prediction in dynamic graphs is a critical task with applications in diverse domains such as social networks, recommendation systems, and e-commerce platforms. While existing Temporal Graph Neural Networks (T-GNNs) have achieved notable success by leveraging complex architectures to model temporal and structural dependencies, they often suffer from scalability and efficiency challenges due to high computational overhead. In this paper, we propose EAGLE, a lightweight framework that integrates short-term temporal recency and long-term global structural patterns. EAGLE consists of a time-aware module that aggregates information from a node's most recent neighbors to reflect its immediate preferences, and a structure-aware module that leverages temporal personalized PageRank to capture the influence of globally important nodes. To balance these attributes, EAGLE employs an adaptive weighting mechanism to dynamically adjust their contributions based on data characteristics. Also, EAGLE eliminates the need for complex multi-hop message passing or memory-intensive mechanisms, enabling significant improvements in efficiency. Extensive experiments on seven real-world temporal graphs demonstrate that EAGLE consistently achieves superior performance against state-of-the-art T-GNNs in both effectiveness and efficiency, delivering more than a 50x speedup over effective transformer-based T-GNNs. 

**Abstract (ZH)**: 动态图中的时间链接预测是一项关键任务，应用于社交网络、推荐系统和电子商务平台等多个领域。尽管现有的时序图神经网络（T-GNNs）通过构建复杂的架构来建模时间依赖性和结构性依赖性已经取得了显著的成果，但由于高计算开销，它们往往面临可扩展性和效率的挑战。本文提出了一种轻量级框架EAGLE，结合了短期时间相关性和长期全局结构性模式。EAGLE 包含一个时间感知模块，通过聚合节点最近邻节点的信息来反映其即时偏好；以及一个结构感知模块，利用时空个性化PageRank来捕捉全局重要节点的影响。为了平衡这些特性，EAGLE采用自适应加权机制，根据数据特性动态调整它们的贡献。此外，EAGLE消除了复杂多跳消息传递或计算密集型机制的需要，从而在效率上实现了显著改进。在七个真实世界的时序图上的广泛实验表明，EAGLE在有效性与效率上均优于最先进的T-GNNs，在有效变体变压器基线的T-GNNs上实现了超过50倍的加速。 

---
# From Extraction to Synthesis: Entangled Heuristics for Agent-Augmented Strategic Reasoning 

**Title (ZH)**: 从提取到综合：代理增强战略推理中的纠缠启发式方法 

**Authors**: Renato Ghisellini, Remo Pareschi, Marco Pedroni, Giovanni Battista Raggi  

**Link**: [PDF](https://arxiv.org/pdf/2507.13768)  

**Abstract**: We present a hybrid architecture for agent-augmented strategic reasoning, combining heuristic extraction, semantic activation, and compositional synthesis. Drawing on sources ranging from classical military theory to contemporary corporate strategy, our model activates and composes multiple heuristics through a process of semantic interdependence inspired by research in quantum cognition. Unlike traditional decision engines that select the best rule, our system fuses conflicting heuristics into coherent and context-sensitive narratives, guided by semantic interaction modeling and rhetorical framing. We demonstrate the framework via a Meta vs. FTC case study, with preliminary validation through semantic metrics. Limitations and extensions (e.g., dynamic interference tuning) are discussed. 

**Abstract (ZH)**: 基于启发式提取、语义激活和组合合成的代理增强战略推理混合架构 

---
# OntView: What you See is What you Meant 

**Title (ZH)**: OntView: 见即所愿 

**Authors**: Carlos Bobed, Carlota Quintana, Eduardo Mena, Jorge Bobed, Fernando Bobillo  

**Link**: [PDF](https://arxiv.org/pdf/2507.13759)  

**Abstract**: In the field of knowledge management and computer science, ontologies provide a structured framework for modeling domain-specific knowledge by defining concepts and their relationships. However, the lack of tools that provide effective visualization is still a significant challenge. While numerous ontology editors and viewers exist, most of them fail to graphically represent ontology structures in a meaningful and non-overwhelming way, limiting users' ability to comprehend dependencies and properties within large ontological frameworks.
In this paper, we present OntView, an ontology viewer that is designed to provide users with an intuitive visual representation of ontology concepts and their formal definitions through a user-friendly interface. Building on the use of a DL reasoner, OntView follows a "What you see is what you meant" paradigm, showing the actual inferred knowledge. One key aspect for this is its ability to visualize General Concept Inclusions (GCI), a feature absent in existing visualization tools. Moreover, to avoid a possible information overload, OntView also offers different ways to show a simplified view of the ontology by: 1) creating ontology summaries by assessing the importance of the concepts (according to different available algorithms), 2) focusing the visualization on the existing TBox elements between two given classes and 3) allowing to hide/show different branches in a dynamic way without losing the semantics. OntView has been released with an open-source license for the whole community. 

**Abstract (ZH)**: 知识管理与计算机科学领域的本体提供了结构化的框架来建模领域特定知识，通过定义概念及其关系。然而，缺乏有效的可视化工具仍然是一个重大挑战。尽管存在多种本体编辑器和查看器，但大多数都未能以有意义且不令人麻木的方式图形化表示本体结构，限制了用户理解大型本体框架中的依赖关系和属性的能力。

本文介绍了一种名为OntView的本体查看器，旨在通过用户友好的界面为用户提供直观的本体概念及其形式定义的可视化表示。基于使用DL推理器，OntView遵循“所见即所想”的原则，展示实际推断的知识。其关键方面在于能够可视化通用概念包含（GCI）这一功能，这是现有可视化工具中所欠缺的。此外，为了防止信息过载，OntView还提供了不同的方式显示简化后的本体视图，包括：1）通过评估概念的重要性（根据不同的可用算法）创建本体摘要；2）将可视化聚焦于两个给定类之间的现有TBox元素；3）允许动态隐藏/显示不同的分支，而不失去语义。OntView已采用开源许可证发布，供整个社区使用。 

---
# DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs 

**Title (ZH)**: DailyLLM：基于多模态传感器和大规模语言模型的上下文感知活动日志生成 

**Authors**: Ye Tian, Xiaoyuan Ren, Zihao Wang, Onat Gungor, Xiaofan Yu, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2507.13737)  

**Abstract**: Rich and context-aware activity logs facilitate user behavior analysis and health monitoring, making them a key research focus in ubiquitous computing. The remarkable semantic understanding and generation capabilities of Large Language Models (LLMs) have recently created new opportunities for activity log generation. However, existing methods continue to exhibit notable limitations in terms of accuracy, efficiency, and semantic richness. To address these challenges, we propose DailyLLM. To the best of our knowledge, this is the first log generation and summarization system that comprehensively integrates contextual activity information across four dimensions: location, motion, environment, and physiology, using only sensors commonly available on smartphones and smartwatches. To achieve this, DailyLLM introduces a lightweight LLM-based framework that integrates structured prompting with efficient feature extraction to enable high-level activity understanding. Extensive experiments demonstrate that DailyLLM outperforms state-of-the-art (SOTA) log generation methods and can be efficiently deployed on personal computers and Raspberry Pi. Utilizing only a 1.5B-parameter LLM model, DailyLLM achieves a 17% improvement in log generation BERTScore precision compared to the 70B-parameter SOTA baseline, while delivering nearly 10x faster inference speed. 

**Abstract (ZH)**: 丰富的上下文感知活动日志促进用户行为分析和健康监测，因此在泛在计算中成为关键研究重点。大型语言模型（LLMs）在语义理解和生成方面的显著能力为活动日志生成创造了新的机会。然而，现有的方法在准确性和语义丰富性方面仍然存在明显限制。为了解决这些挑战，我们提出了DailyLLM。据我们所知，这是第一个全面整合跨四个维度（位置、运动、环境和生理）上下文活动信息的日志生成和总结系统，仅使用智能手机和智能手表上常见的传感器。为了实现这一目标，DailyLLM 引入了一种轻量级的基于LLM的框架，该框架结合了结构化提示与高效的特征提取，以实现高级活动理解。广泛实验表明，DailyLLM 在日志生成方面优于最先进的（SOTA）方法，并且可以高效部署在个人计算机和 Raspberry Pi 上。仅使用一个包含1.5B参数的LLM模型，DailyLLM 在日志生成 BERTScore 精度上比参数量为70B的SOTA基线提高了17%，同时提供近10倍的推理速度。 

---
# Combining model tracing and constraint-based modeling for multistep strategy diagnoses 

**Title (ZH)**: 结合模型跟踪和约束基于建模进行多步策略诊断 

**Authors**: Gerben van der Hoek, Johan Jeuring, Rogier Bos  

**Link**: [PDF](https://arxiv.org/pdf/2507.13652)  

**Abstract**: Model tracing and constraint-based modeling are two approaches to diagnose student input in stepwise tasks. Model tracing supports identifying consecutive problem-solving steps taken by a student, whereas constraint-based modeling supports student input diagnosis even when several steps are combined into one step. We propose an approach that merges both paradigms. By defining constraints as properties that a student input has in common with a step of a strategy, it is possible to provide a diagnosis when a student deviates from a strategy even when the student combines several steps. In this study we explore the design of a system for multistep strategy diagnoses, and evaluate these diagnoses. As a proof of concept, we generate diagnoses for an existing dataset containing steps students take when solving quadratic equations (n=2136). To compare with human diagnoses, two teachers coded a random sample of deviations (n=70) and applications of the strategy (n=70). Results show that that the system diagnosis aligned with the teacher coding in all of the 140 student steps. 

**Abstract (ZH)**: 模型跟踪和基于约束的建模是两种诊断学生在步进任务中输入的方法。模型跟踪支持识别学生连续的问题解决步骤，而基于约束的建模即使学生将多个步骤合并为一步，也能支持对学生输入的诊断。我们提出了一种结合这两种范式的办法。通过将约束定义为学生输入与策略步骤共有的属性，即使学生合并了多个步骤，也能提供诊断。在此研究中，我们探索了一种用于多步策略诊断系统的架构设计，并评估了这些诊断。作为概念验证，我们为一个包含学生解二次方程步骤的数据集（n=2136）生成了诊断。为了与人工诊断进行比较，两位教师对策略的应用（n=70）和偏差（n=70）进行了编码。结果显示，系统的诊断结果与教师编码完全一致。 

---
# Buggy rule diagnosis for combined steps through final answer evaluation in stepwise tasks 

**Title (ZH)**: 通过最终答案评估进行分步任务中错误规则诊断 

**Authors**: Gerben van der Hoek, Johan Jeuring, Rogier Bos  

**Link**: [PDF](https://arxiv.org/pdf/2507.13651)  

**Abstract**: Many intelligent tutoring systems can support a student in solving a stepwise task. When a student combines several steps in one step, the number of possible paths connecting consecutive inputs may be very large. This combinatorial explosion makes error diagnosis hard. Using a final answer to diagnose a combination of steps can mitigate the combinatorial explosion, because there are generally fewer possible (erroneous) final answers than (erroneous) solution paths. An intermediate input for a task can be diagnosed by automatically completing it according to the task solution strategy and diagnosing this solution. This study explores the potential of automated error diagnosis based on a final answer. We investigate the design of a service that provides a buggy rule diagnosis when a student combines several steps. To validate the approach, we apply the service to an existing dataset (n=1939) of unique student steps when solving quadratic equations, which could not be diagnosed by a buggy rule service that tries to connect consecutive inputs with a single rule. Results show that final answer evaluation can diagnose 29,4% of these steps. Moreover, a comparison of the generated diagnoses with teacher diagnoses on a subset (n=115) shows that the diagnoses align in 97% of the cases. These results can be considered a basis for further exploration of the approach. 

**Abstract (ZH)**: 许多智能教学系统可以支持学生解决逐步任务。当学生将多个步骤合并为一步时，连接连续输入的可能路径数量可能会非常大。这种组合爆炸使错误诊断变得困难。使用最终答案进行步骤组合的错误诊断可以减轻组合爆炸，因为通常可能的错误最终答案比错误的解题路径要少。任务中的一个中间输入可以通过根据任务解题策略自动完成并诊断该解题过程来进行诊断。本研究探讨了基于最终答案的自动化错误诊断的潜力。我们研究了一种服务的设计，该服务在学生将多个步骤合并时提供错误规则诊断。为了验证该方法，我们将该服务应用于解决二次方程时产生的唯一学生步骤数据集（n=1939），这些步骤无法由尝试用单一规则连接连续输入的错误规则服务诊断。结果显示，最终答案评估可以诊断其中29.4%的步骤。此外，将生成的诊断与在子集（n=115）上进行的教师诊断进行比较，显示有97%的情况诊断一致。这些结果可以作为进一步探索该方法的基础。 

---
# BifrostRAG: Bridging Dual Knowledge Graphs for Multi-Hop Question Answering in Construction Safety 

**Title (ZH)**: BifrostRAG: 联接双知识图谱进行建筑安全多跳问答 

**Authors**: Yuxin Zhang, Xi Wang, Mo Hu, Zhenyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13625)  

**Abstract**: Information retrieval and question answering from safety regulations are essential for automated construction compliance checking but are hindered by the linguistic and structural complexity of regulatory text. Many compliance-related queries are multi-hop, requiring synthesis of information across interlinked clauses. This poses a challenge for traditional retrieval-augmented generation (RAG) systems. To overcome this, we introduce BifrostRAG: a dual-graph RAG-integrated system that explicitly models both linguistic relationships (via an Entity Network Graph) and document structure (via a Document Navigator Graph). This architecture powers a hybrid retrieval mechanism that combines graph traversal with vector-based semantic search, enabling large language models to reason over both the meaning and the structure of the text. Evaluation on a multi-hop question dataset shows that BifrostRAG achieves 92.8 percent precision, 85.5 percent recall, and an F1 score of 87.3 percent. These results significantly outperform vector-only and graph-only RAG baselines that represent current leading approaches. Error analysis further highlights the comparative advantages of our hybrid method over single-modality RAGs. These findings establish BifrostRAG as a robust knowledge engine for LLM-driven compliance checking. Its dual-graph, hybrid retrieval mechanism offers a transferable blueprint for navigating complex technical documents across knowledge-intensive engineering domains. 

**Abstract (ZH)**: 基于双图的RAG系统：BifrostRAG及其在安全规范信息检索与问答中的应用 

---
# Why Isn't Relational Learning Taking Over the World? 

**Title (ZH)**: 为什么关系学习尚未主导世界？ 

**Authors**: David Poole  

**Link**: [PDF](https://arxiv.org/pdf/2507.13558)  

**Abstract**: AI seems to be taking over the world with systems that model pixels, words, and phonemes. The world is arguably made up, not of pixels, words, and phonemes but of entities (objects, things, including events) with properties and relations among them. Surely we should model these, not the perception or description of them. You might suspect that concentrating on modeling words and pixels is because all of the (valuable) data in the world is in terms of text and images. If you look into almost any company you will find their most valuable data is in spreadsheets, databases and other relational formats. These are not the form that are studied in introductory machine learning, but are full of product numbers, student numbers, transaction numbers and other identifiers that can't be interpreted naively as numbers. The field that studies this sort of data has various names including relational learning, statistical relational AI, and many others. This paper explains why relational learning is not taking over the world -- except in a few cases with restricted relations -- and what needs to be done to bring it to it's rightful prominence. 

**Abstract (ZH)**: AI似乎正在通过建模像素、单词和音素的系统接管世界。世界的本质，或许并非由像素、单词和音素构成，而是由具有属性及其相互关系的实体（对象、事物，包括事件）构成。当然，我们应当建模这些实体及其关系，而不是它们的感知或描述。你可能会怀疑，专注于建模单词和像素是因为世界上所有（有价值的）数据都以文本和图像的形式存在。如果你查看几乎任何一家公司，你会发现它们最宝贵的数据存储在电子表格、数据库和其他关系型格式中。这些数据的形式并未在入门级机器学习中得到研究，但其中包含了产品编号、学生编号、交易编号以及其他不能简单解释为数字的标识符。研究这类数据的领域有着各种名称，包括关系学习、统计关系型AI等。本文解释了为什么关系学习并未接管世界——除了在少数涉及受限关系的情况下——并提出了使其得到应有地位所需采取的措施。 

---
# GOFAI meets Generative AI: Development of Expert Systems by means of Large Language Models 

**Title (ZH)**: 基于大型语言模型开发专家系统：GOFAI与生成型AI的融合 

**Authors**: Eduardo C. Garrido-Merchán, Cristina Puente  

**Link**: [PDF](https://arxiv.org/pdf/2507.13550)  

**Abstract**: The development of large language models (LLMs) has successfully transformed knowledge-based systems such as open domain question nswering, which can automatically produce vast amounts of seemingly coherent information. Yet, those models have several disadvantages like hallucinations or confident generation of incorrect or unverifiable facts. In this paper, we introduce a new approach to the development of expert systems using LLMs in a controlled and transparent way. By limiting the domain and employing a well-structured prompt-based extraction approach, we produce a symbolic representation of knowledge in Prolog, which can be validated and corrected by human experts. This approach also guarantees interpretability, scalability and reliability of the developed expert systems. Via quantitative and qualitative experiments with Claude Sonnet 3.7 and GPT-4.1, we show strong adherence to facts and semantic coherence on our generated knowledge bases. We present a transparent hybrid solution that combines the recall capacity of LLMs with the precision of symbolic systems, thereby laying the foundation for dependable AI applications in sensitive domains. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的发展成功地转化了知识型系统，如开放领域问答，可以自动生成大量看似连贯的信息。然而，这些模型存在诸如幻觉或自信生成错误或无法验证的事实等缺点。本文介绍了一种通过受控和透明方式使用LLMs开发专家系统的新型方法。通过限定领域并采用结构化的提示提取方法，我们产生了在Prolog中的符号知识表示，该表示可以由人类专家验证和修正。这种方法还保证了开发的专家系统的可解释性、可扩展性和可靠性。通过与Claude Sonnet 3.7和GPT-4.1的定量和定性实验，我们展示了生成的知识库在事实准确性和语义连贯性方面的坚定遵循。我们提出了一种透明的混合解决方案，结合了LLMs的记忆能力和符号系统的精确性，从而为敏感领域的可靠AI应用奠定了基础。 

---
# PrefPalette: Personalized Preference Modeling with Latent Attributes 

**Title (ZH)**: PrefPalette: 基于潜在属性的个性化偏好建模 

**Authors**: Shuyue Stella Li, Melanie Sclar, Hunter Lang, Ansong Ni, Jacqueline He, Puxin Xu, Andrew Cohen, Chan Young Park, Yulia Tsvetkov, Asli Celikyilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2507.13541)  

**Abstract**: Personalizing AI systems requires understanding not just what users prefer, but the reasons that underlie those preferences - yet current preference models typically treat human judgment as a black box. We introduce PrefPalette, a framework that decomposes preferences into attribute dimensions and tailors its preference prediction to distinct social community values in a human-interpretable manner. PrefPalette operationalizes a cognitive science principle known as multi-attribute decision making in two ways: (1) a scalable counterfactual attribute synthesis step that involves generating synthetic training data to isolate for individual attribute effects (e.g., formality, humor, cultural values), and (2) attention-based preference modeling that learns how different social communities dynamically weight these attributes. This approach moves beyond aggregate preference modeling to capture the diverse evaluation frameworks that drive human judgment. When evaluated on 45 social communities from the online platform Reddit, PrefPalette outperforms GPT-4o by 46.6% in average prediction accuracy. Beyond raw predictive improvements, PrefPalette also shed light on intuitive, community-specific profiles: scholarly communities prioritize verbosity and stimulation, conflict-oriented communities value sarcasm and directness, and support-based communities emphasize empathy. By modeling the attribute-mediated structure of human judgment, PrefPalette delivers both superior preference modeling and transparent, interpretable insights, and serves as a first step toward more trustworthy, value-aware personalized applications. 

**Abstract (ZH)**: PrefPalette：基于属性维度的社交社区价值观可解释偏好预测框架 

---
# GraphTrafficGPT: Enhancing Traffic Management Through Graph-Based AI Agent Coordination 

**Title (ZH)**: GraphTrafficGPT：通过图基人工智能代理协调增强交通管理 

**Authors**: Nabil Abdelaziz Ferhat Taleb, Abdolazim Rezaei, Raj Atulkumar Patel, Mehdi Sookhak  

**Link**: [PDF](https://arxiv.org/pdf/2507.13511)  

**Abstract**: Large Language Models (LLMs) offer significant promise for intelligent traffic management; however, current chain-based systems like TrafficGPT are hindered by sequential task execution, high token usage, and poor scalability, making them inefficient for complex, real-world scenarios. To address these limitations, we propose GraphTrafficGPT, a novel graph-based architecture, which fundamentally redesigns the task coordination process for LLM-driven traffic applications. GraphTrafficGPT represents tasks and their dependencies as nodes and edges in a directed graph, enabling efficient parallel execution and dynamic resource allocation. The main idea behind the proposed model is a Brain Agent that decomposes user queries, constructs optimized dependency graphs, and coordinates a network of specialized agents for data retrieval, analysis, visualization, and simulation. By introducing advanced context-aware token management and supporting concurrent multi-query processing, the proposed architecture handles interdependent tasks typical of modern urban mobility environments. Experimental results demonstrate that GraphTrafficGPT reduces token consumption by 50.2% and average response latency by 19.0% compared to TrafficGPT, while supporting simultaneous multi-query execution with up to 23.0% improvement in efficiency. 

**Abstract (ZH)**: 大型语言模型（LLMs）在智能交通管理方面提供了显著的潜力；然而，当前基于链的设计如TrafficGPT受限于顺序任务执行、高 token 使用率和较差的扩展性，使得它们在复杂的现实场景中效率低下。为了克服这些限制，我们提出了一种新的图基架构GraphTrafficGPT，从根本上重新设计了由LLM驱动的交通应用的任务协调过程。GraphTrafficGPT将任务及其依赖关系表示为有向图中的节点和边，从而实现高效并行执行和动态资源分配。所提出模型的核心思想是一种名为Brain Agent的实体，它分解用户查询、构建优化的依赖关系图，并协调数据检索、分析、可视化和模拟等专门代理。通过引入先进的上下文感知 token 管理和支持并发多查询处理，所提出架构能够处理现代城市交通环境中典型的相互依赖任务。实验结果表明，与TrafficGPT相比，GraphTrafficGPT将token消耗减少了50.2%，平均响应延时降低了19.0%，同时支持并发多查询执行，效率提高了23.0%。 

---
# Toward Temporal Causal Representation Learning with Tensor Decomposition 

**Title (ZH)**: 基于张量分解的时序因果表示学习 

**Authors**: Jianhong Chen, Meng Zhao, Mostafa Reisi Gahrooei, Xubo Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.14126)  

**Abstract**: Temporal causal representation learning is a powerful tool for uncovering complex patterns in observational studies, which are often represented as low-dimensional time series. However, in many real-world applications, data are high-dimensional with varying input lengths and naturally take the form of irregular tensors. To analyze such data, irregular tensor decomposition is critical for extracting meaningful clusters that capture essential information. In this paper, we focus on modeling causal representation learning based on the transformed information. First, we present a novel causal formulation for a set of latent clusters. We then propose CaRTeD, a joint learning framework that integrates temporal causal representation learning with irregular tensor decomposition. Notably, our framework provides a blueprint for downstream tasks using the learned tensor factors, such as modeling latent structures and extracting causal information, and offers a more flexible regularization design to enhance tensor decomposition. Theoretically, we show that our algorithm converges to a stationary point. More importantly, our results fill the gap in theoretical guarantees for the convergence of state-of-the-art irregular tensor decomposition. Experimental results on synthetic and real-world electronic health record (EHR) datasets (MIMIC-III), with extensive benchmarks from both phenotyping and network recovery perspectives, demonstrate that our proposed method outperforms state-of-the-art techniques and enhances the explainability of causal representations. 

**Abstract (ZH)**: 基于转换信息的因果表示学习：一种用于分析高维不规则张量数据的框架 

---
# Kolmogorov Arnold Networks (KANs) for Imbalanced Data -- An Empirical Perspective 

**Title (ZH)**: Kolmogorov Arnold 网络（KANs）在不平衡数据下的实证研究 

**Authors**: Pankaj Yadav, Vivek Vijay  

**Link**: [PDF](https://arxiv.org/pdf/2507.14121)  

**Abstract**: Kolmogorov Arnold Networks (KANs) are recent architectural advancement in neural computation that offer a mathematically grounded alternative to standard neural networks. This study presents an empirical evaluation of KANs in context of class imbalanced classification, using ten benchmark datasets. We observe that KANs can inherently perform well on raw imbalanced data more effectively than Multi-Layer Perceptrons (MLPs) without any resampling strategy. However, conventional imbalance strategies fundamentally conflict with KANs mathematical structure as resampling and focal loss implementations significantly degrade KANs performance, while marginally benefiting MLPs. Crucially, KANs suffer from prohibitive computational costs without proportional performance gains. Statistical validation confirms that MLPs with imbalance techniques achieve equivalence with KANs (|d| < 0.08 across metrics) at minimal resource costs. These findings reveal that KANs represent a specialized solution for raw imbalanced data where resources permit. But their severe performance-resource tradeoffs and incompatibility with standard resampling techniques currently limits practical deployment. We identify critical research priorities as developing KAN specific architectural modifications for imbalance learning, optimizing computational efficiency, and theoretical reconciling their conflict with data augmentation. This work establishes foundational insights for next generation KAN architectures in imbalanced classification scenarios. 

**Abstract (ZH)**: Kolmogorov Arnold 网络 (KANs) 在样本不平衡分类中的实证评估：资源允许下的原始不平衡数据专业解决方案 

---
# NoHumansRequired: Autonomous High-Quality Image Editing Triplet Mining 

**Title (ZH)**: 无需人类介入：自主高质图像编辑 triplet 矿化 

**Authors**: Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh, Georgii Fedorov, Bulat Suleimanov, Vladimir Dokholyan, Aleksandr Gordeev  

**Link**: [PDF](https://arxiv.org/pdf/2507.14119)  

**Abstract**: Recent advances in generative modeling enable image editing assistants that follow natural language instructions without additional user input. Their supervised training requires millions of triplets: original image, instruction, edited image. Yet mining pixel-accurate examples is hard. Each edit must affect only prompt-specified regions, preserve stylistic coherence, respect physical plausibility, and retain visual appeal. The lack of robust automated edit-quality metrics hinders reliable automation at scale. We present an automated, modular pipeline that mines high-fidelity triplets across domains, resolutions, instruction complexities, and styles. Built on public generative models and running without human intervention, our system uses a task-tuned Gemini validator to score instruction adherence and aesthetics directly, removing any need for segmentation or grounding models. Inversion and compositional bootstrapping enlarge the mined set by approximately 2.2x, enabling large-scale high-fidelity training data. By automating the most repetitive annotation steps, the approach allows a new scale of training without human labeling effort. To democratize research in this resource-intensive area, we release NHR-Edit: an open dataset of 358k high-quality triplets. In the largest cross-dataset evaluation, it surpasses all public alternatives. We also release Bagel-NHR-Edit, an open-source fine-tuned Bagel model, which achieves state-of-the-art metrics in our experiments. 

**Abstract (ZH)**: 近期生成模型的进展使得能够遵循自然语言指令进行图像编辑，无需额外用户输入。它们的监督训练需要数百万个三元组：原始图像、指令、编辑图像。然而，挖掘像素级准确的例子是困难的。每个编辑必须仅影响提示指定的区域，保持样式一致性，尊重物理可行性，并保持视觉吸引力。缺乏稳健的自动编辑质量度量阻碍了大规模可靠自动化。我们提出了一种自动化的模块化管道，能够跨领域、分辨率、指令复杂度和样式挖掘高质量三元组。该系统基于公开的生成模型且无需人工干预，使用任务调优的Gemini验证器直接评分指令遵循度和美学，移除了任何对分割或语义模型的需求。逆向建模和组合式自举将挖掘集合扩大约2.2倍，支持大规模高质量训练数据。通过自动化最重复的注释步骤，该方法允许在无需人工标记的情况下实现更大规模的训练。为了使这一资源密集型领域的研究更具普惠性，我们发布了NHR-Edit：一个包含358,000个高质量三元组的开放数据集，在最大的跨数据集评估中超越所有公开替代方案。我们还发布了Bagel-NHR-Edit：一个开源微调Bagel模型，实验中实现了最佳指标。 

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
# Multi-Centre Validation of a Deep Learning Model for Scoliosis Assessment 

**Title (ZH)**: 多中心深学习模型脊柱侧弯评估验证 

**Authors**: Šimon Kubov, Simon Klíčník, Jakub Dandár, Zdeněk Straka, Karolína Kvaková, Daniel Kvak  

**Link**: [PDF](https://arxiv.org/pdf/2507.14093)  

**Abstract**: Scoliosis affects roughly 2 to 4 percent of adolescents, and treatment decisions depend on precise Cobb angle measurement. Manual assessment is time consuming and subject to inter observer variation. We conducted a retrospective, multi centre evaluation of a fully automated deep learning software (Carebot AI Bones, Spine Measurement functionality; Carebot s.r.o.) on 103 standing anteroposterior whole spine radiographs collected from ten hospitals. Two musculoskeletal radiologists independently measured each study and served as reference readers. Agreement between the AI and each radiologist was assessed with Bland Altman analysis, mean absolute error (MAE), root mean squared error (RMSE), Pearson correlation coefficient, and Cohen kappa for four grade severity classification. Against Radiologist 1 the AI achieved an MAE of 3.89 degrees (RMSE 4.77 degrees) with a bias of 0.70 degrees and limits of agreement from minus 8.59 to plus 9.99 degrees. Against Radiologist 2 the AI achieved an MAE of 3.90 degrees (RMSE 5.68 degrees) with a bias of 2.14 degrees and limits from minus 8.23 to plus 12.50 degrees. Pearson correlations were r equals 0.906 and r equals 0.880 (inter reader r equals 0.928), while Cohen kappa for severity grading reached 0.51 and 0.64 (inter reader kappa 0.59). These results demonstrate that the proposed software reproduces expert level Cobb angle measurements and categorical grading across multiple centres, suggesting its utility for streamlining scoliosis reporting and triage in clinical workflows. 

**Abstract (ZH)**: 脊柱侧弯影响约2%至4%的青少年，准确的柯布角测量决定了治疗决策。手工评估耗时且存在观测者间变异。我们回顾性地评估了（Carebot AI Bones, 脊柱测量功能；Carebot s.r.o.）一种全自动深度学习软件在10家医院收集的103张站立前后位完整脊柱X光片上的表现。两位骨关节放射学家独立测量每张X光片，并作为参考读者。使用Bland-Altman分析、平均绝对误差（MAE）、均方根误差（RMSE）、皮尔森相关系数和科恩κ系数来评估软件与每位放射学家之间的同意程度。与放射学家1相比，软件的MAE为3.89度（RMSE 4.77度），偏差为0.70度，置信区间为-8.59至9.99度；与放射学家2相比，软件的MAE为3.90度（RMSE 5.68度），偏差为2.14度，置信区间为-8.23至12.50度。皮尔森相关系数分别为0.906和0.880（观测者间相关系数为0.928），而严重程度分类的科恩κ系数分别为0.51和0.64（观测者间κ系数为0.59）。这些结果表明，所提软件能够多中心再现专家级柯布角测量和分类，建议其在临床工作流程中用于简化脊柱侧弯报告和分流。 

---
# The Emotion-Memory Link: Do Memorability Annotations Matter for Intelligent Systems? 

**Title (ZH)**: 情绪与记忆的联系：记忆注释对智能系统重要吗？ 

**Authors**: Maria Tsfasman, Ramin Ghorbani, Catholijn M. Jonker, Bernd Dudzik  

**Link**: [PDF](https://arxiv.org/pdf/2507.14084)  

**Abstract**: Humans have a selective memory, remembering relevant episodes and forgetting the less relevant information. Possessing awareness of event memorability for a user could help intelligent systems in more accurate user modelling, especially for such applications as meeting support systems, memory augmentation, and meeting summarisation. Emotion recognition has been widely studied, since emotions are thought to signal moments of high personal relevance to users. The emotional experience of situations and their memorability have traditionally been considered to be closely tied to one another: moments that are experienced as highly emotional are considered to also be highly memorable. This relationship suggests that emotional annotations could serve as proxies for memorability. However, existing emotion recognition systems rely heavily on third-party annotations, which may not accurately represent the first-person experience of emotional relevance and memorability. This is why, in this study, we empirically examine the relationship between perceived group emotions (Pleasure-Arousal) and group memorability in the context of conversational interactions. Our investigation involves continuous time-based annotations of both emotions and memorability in dynamic, unstructured group settings, approximating conditions of real-world conversational AI applications such as online meeting support systems. Our results show that the observed relationship between affect and memorability annotations cannot be reliably distinguished from what might be expected under random chance. We discuss the implications of this surprising finding for the development and applications of Affective Computing technology. In addition, we contextualise our findings in broader discourses in the Affective Computing and point out important targets for future research efforts. 

**Abstract (ZH)**: 人类具有选择性记忆，记住相关事件并遗忘较少相关的信息。了解用户的事件记忆性意识有助于智能系统更准确地进行用户建模，尤其是在会议支持系统、记忆增强和会议总结等应用中。情绪识别因其被认为是用户个人高度相关的信号而受到了广泛研究。情绪体验的情境及其记忆性一直被认为紧密相关：被体验为高度情绪化的时刻也被认为是高度记忆性的。这种关系表明情绪注释可能作为记忆性的代理。然而，现有的情绪识别系统高度依赖第三方注释，这可能无法准确反映情绪相关性和记忆性的第一人体验。因此，在本研究中，我们实证考察了对话互动中感知群体情绪（愉悦-唤醒）与群体记忆性之间的关系。我们的调查涉及对动态、非结构化群体环境中的情绪和记忆性进行连续的时间基注释，以模拟在线会议支持系统等实际世界对话AI应用的条件。我们的结果表明，观察到的情绪与记忆性注释之间的关系无法可靠地区分出与随机机会所期望的差异。我们讨论了这一出人意料的发现对情感计算技术开发和应用的影响，并在更广泛的讨论中阐述了对未来的研究方向。 

---
# DENSE: Longitudinal Progress Note Generation with Temporal Modeling of Heterogeneous Clinical Notes Across Hospital Visits 

**Title (ZH)**: DENSE：纵向病程笔记生成，考虑医院访问期间异质临床笔记的时间建模 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.14079)  

**Abstract**: Progress notes are among the most clinically meaningful artifacts in an Electronic Health Record (EHR), offering temporally grounded insights into a patient's evolving condition, treatments, and care decisions. Despite their importance, they are severely underrepresented in large-scale EHR datasets. For instance, in the widely used Medical Information Mart for Intensive Care III (MIMIC-III) dataset, only about $8.56\%$ of hospital visits include progress notes, leaving gaps in longitudinal patient narratives. In contrast, the dataset contains a diverse array of other note types, each capturing different aspects of care.
We present DENSE (Documenting Evolving Progress Notes from Scattered Evidence), a system designed to align with clinical documentation workflows by simulating how physicians reference past encounters while drafting progress notes. The system introduces a fine-grained note categorization and a temporal alignment mechanism that organizes heterogeneous notes across visits into structured, chronological inputs. At its core, DENSE leverages a clinically informed retrieval strategy to identify temporally and semantically relevant content from both current and prior visits. This retrieved evidence is used to prompt a large language model (LLM) to generate clinically coherent and temporally aware progress notes.
We evaluate DENSE on a curated cohort of patients with multiple visits and complete progress note documentation. The generated notes demonstrate strong longitudinal fidelity, achieving a temporal alignment ratio of $1.089$, surpassing the continuity observed in original notes. By restoring narrative coherence across fragmented documentation, our system supports improved downstream tasks such as summarization, predictive modeling, and clinical decision support, offering a scalable solution for LLM-driven note synthesis in real-world healthcare settings. 

**Abstract (ZH)**: 文档化 evolving 进展笔记：基于散落证据的系统(DENSE) 

---
# Edge Intelligence with Spiking Neural Networks 

**Title (ZH)**: 边缘智能中的脉冲神经网络 

**Authors**: Shuiguang Deng, Di Yu, Changze Lv, Xin Du, Linshan Jiang, Xiaofan Zhao, Wentao Tong, Xiaoqing Zheng, Weijia Fang, Peng Zhao, Gang Pan, Schahram Dustdar, Albert Y. Zomaya  

**Link**: [PDF](https://arxiv.org/pdf/2507.14069)  

**Abstract**: The convergence of artificial intelligence and edge computing has spurred growing interest in enabling intelligent services directly on resource-constrained devices. While traditional deep learning models require significant computational resources and centralized data management, the resulting latency, bandwidth consumption, and privacy concerns have exposed critical limitations in cloud-centric paradigms. Brain-inspired computing, particularly Spiking Neural Networks (SNNs), offers a promising alternative by emulating biological neuronal dynamics to achieve low-power, event-driven computation. This survey provides a comprehensive overview of Edge Intelligence based on SNNs (EdgeSNNs), examining their potential to address the challenges of on-device learning, inference, and security in edge scenarios. We present a systematic taxonomy of EdgeSNN foundations, encompassing neuron models, learning algorithms, and supporting hardware platforms. Three representative practical considerations of EdgeSNN are discussed in depth: on-device inference using lightweight SNN models, resource-aware training and updating under non-stationary data conditions, and secure and privacy-preserving issues. Furthermore, we highlight the limitations of evaluating EdgeSNNs on conventional hardware and introduce a dual-track benchmarking strategy to support fair comparisons and hardware-aware optimization. Through this study, we aim to bridge the gap between brain-inspired learning and practical edge deployment, offering insights into current advancements, open challenges, and future research directions. To the best of our knowledge, this is the first dedicated and comprehensive survey on EdgeSNNs, providing an essential reference for researchers and practitioners working at the intersection of neuromorphic computing and edge intelligence. 

**Abstract (ZH)**: 人工智能与边缘计算的融合推动了在资源受限设备上直接提供智能服务的兴趣增长。传统的深度学习模型需要大量的计算资源和集中式数据管理，而由此产生的延迟、带宽消耗以及隐私担忧暴露了以云为中心范式的關鍵局限性。受大脑启发的计算，特别是突触神经网络（SNNs），通过模拟生物神经元动力学实现低功耗、事件驱动的计算，提供了有前途的替代方案。本文综述了基于SNNs的边缘智能（EdgeSNNs）及其在边缘场景中应对设备上学习、推理和安全挑战的潜力。我们系统地介绍了EdgeSNN的基础，涵盖神经元模型、学习算法和支持的硬件平台。此外，我们深入讨论了EdgeSNN的三个代表性实用考虑：使用轻量级SNN模型进行设备上推理，适应非平稳数据条件下的资源感知训练和更新，以及安全和隐私保护问题。我们还指出了在传统硬件上评估EdgeSNN的局限性，并介绍了一种双轨基准测试策略，以支持公平比较和硬件感知优化。通过本研究，我们旨在弥合受大脑启发的学习与实际边缘部署之间的差距，为神经形态计算和边缘智能的交叉领域提供当前进展、开放挑战和未来研究方向的见解。据我们所知，这是关于EdgeSNNs的第一篇专门且全面的综述，为神经形态计算与边缘智能交叉领域的研究人员和 practitioners 提供了重要的参考资料。 

---
# VLA-Mark: A cross modal watermark for large vision-language alignment model 

**Title (ZH)**: VLA-Mark：一种用于大型视觉-语言对齐模型的跨模态水印 

**Authors**: Shuliang Liu, Qi Zheng, Jesse Jiaxi Xu, Yibo Yan, He Geng, Aiwei Liu, Peijie Jiang, Jia Liu, Yik-Cheung Tam, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14067)  

**Abstract**: Vision-language models demand watermarking solutions that protect intellectual property without compromising multimodal coherence. Existing text watermarking methods disrupt visual-textual alignment through biased token selection and static strategies, leaving semantic-critical concepts vulnerable. We propose VLA-Mark, a vision-aligned framework that embeds detectable watermarks while preserving semantic fidelity through cross-modal coordination. Our approach integrates multiscale visual-textual alignment metrics, combining localized patch affinity, global semantic coherence, and contextual attention patterns, to guide watermark injection without model retraining. An entropy-sensitive mechanism dynamically balances watermark strength and semantic preservation, prioritizing visual grounding during low-uncertainty generation phases. Experiments show 7.4% lower PPL and 26.6% higher BLEU than conventional methods, with near-perfect detection (98.8% AUC). The framework demonstrates 96.1\% attack resilience against attacks such as paraphrasing and synonym substitution, while maintaining text-visual consistency, establishing new standards for quality-preserving multimodal watermarking 

**Abstract (ZH)**: 视觉-语言模型需要保护知识产权的同时不牺牲多模态一致性的水印解决方案。现有的文本水印方法通过有偏的令牌选择和静态策略破坏了视觉-文本对齐，使语义关键概念处于风险之中。我们提出了一种视觉对齐框架VLA-Mark，在保持语义保真度的同时嵌入可检测的水印，通过跨模态协调引导水印注入。我们的方法结合了多尺度视觉-文本对齐度量，包括局部补丁亲和性、全局语义一致性以及上下文注意力模式，以指导水印注入而无需重新训练模型。熵敏感机制动态平衡水印强度和语义保真度，在低不确定性生成阶段优先考虑视觉接地。实验结果显示，与传统方法相比，PPL低7.4%，BLEU高26.6%，并且具有接近完美的检测率（AUC为98.8%）。该框架对诸如改写和同义词替换等攻击具有96.1%的抗攻击性，同时保持文本-视觉一致性，建立了一种新的质量保留的多模态水印标准。 

---
# Noradrenergic-inspired gain modulation attenuates the stability gap in joint training 

**Title (ZH)**: 去甲肾上腺素启发的增益调节减轻联合训练中的稳定性差距 

**Authors**: Alejandro Rodriguez-Garcia, Anindya Ghosh, Srikanth Ramaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2507.14056)  

**Abstract**: Recent studies in continual learning have identified a transient drop in performance on mastered tasks when assimilating new ones, known as the stability gap. Such dynamics contradict the objectives of continual learning, revealing a lack of robustness in mitigating forgetting, and notably, persisting even under an ideal joint-loss regime. Examining this gap within this idealized joint training context is critical to isolate it from other sources of forgetting. We argue that it reflects an imbalance between rapid adaptation and robust retention at task boundaries, underscoring the need to investigate mechanisms that reconcile plasticity and stability within continual learning frameworks. Biological brains navigate a similar dilemma by operating concurrently on multiple timescales, leveraging neuromodulatory signals to modulate synaptic plasticity. However, artificial networks lack native multitimescale dynamics, and although optimizers like momentum-SGD and Adam introduce implicit timescale regularization, they still exhibit stability gaps. Inspired by locus coeruleus mediated noradrenergic bursts, which transiently enhance neuronal gain under uncertainty to facilitate sensory assimilation, we propose uncertainty-modulated gain dynamics - an adaptive mechanism that approximates a two-timescale optimizer and dynamically balances integration of knowledge with minimal interference on previously consolidated information. We evaluate our mechanism on domain-incremental and class-incremental variants of the MNIST and CIFAR benchmarks under joint training, demonstrating that uncertainty-modulated gain dynamics effectively attenuate the stability gap. Finally, our analysis elucidates how gain modulation replicates noradrenergic functions in cortical circuits, offering mechanistic insights into reducing stability gaps and enhance performance in continual learning tasks. 

**Abstract (ZH)**: 近期关于持续学习的研究发现，在吸收新任务时已掌握任务的性能会出现短暂下降，这种现象被称为稳定性缺口。这种动态与持续学习的目标相矛盾，暴露出在减轻遗忘方面缺乏稳健性，并且即使在理想的联合损失范式下仍然存在。在这一理想化的联合训练背景下考察这一缺口对于将其与其他遗忘来源区分开来至关重要。我们认为这反映了在任务边界处快速适应与稳健保持之间的不平衡，强调了在持续学习框架内研究兼具可塑性和稳定性的机制的必要性。生物大脑通过同时在多个时间尺度上运行并利用神经调控信号来调节突触可塑性来应对类似的困境。然而，人工网络缺乏内在的多时间尺度动态，尽管动量-SGD和Adam等优化器引入了隐式的时标正则化，但仍表现出稳定性缺口。受蓝斑介导的去甲肾上腺素爆发的启发，在不确定性下暂时提高神经元增益以促进感觉吸收，我们提出了一种不确定性调节增益动态机制，这是一种近似两时间尺度优化器的自适应机制，并能动态平衡知识的整合与对先前巩固信息的最小干扰。我们在联合训练下的MNIST和CIFAR增量域和增量类基准上评估了我们的机制，证明了不确定性调节增益动态有效地减小了稳定性缺口。最后，我们的分析阐明了增益调节如何在皮层回路中复制去甲肾上腺素的功能，并为减少稳定性缺口和改善持续学习任务的性能提供了机制性见解。 

---
# A multi-strategy improved snake optimizer for three-dimensional UAV path planning and engineering problems 

**Title (ZH)**: 基于多策略改进的蛇优化算法在三维无人飞行器路径规划及工程问题中的应用 

**Authors**: Genliang Li, Yaxin Cui, Jinyu Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.14043)  

**Abstract**: Metaheuristic algorithms have gained widespread application across various fields owing to their ability to generate diverse solutions. One such algorithm is the Snake Optimizer (SO), a progressive optimization approach. However, SO suffers from the issues of slow convergence speed and susceptibility to local optima. In light of these shortcomings, we propose a novel Multi-strategy Improved Snake Optimizer (MISO). Firstly, we propose a new adaptive random disturbance strategy based on sine function to alleviate the risk of getting trapped in a local optimum. Secondly, we introduce adaptive Levy flight strategy based on scale factor and leader and endow the male snake leader with flight capability, which makes it easier for the algorithm to leap out of the local optimum and find the global optimum. More importantly, we put forward a position update strategy combining elite leadership and Brownian motion, effectively accelerating the convergence speed while ensuring precision. Finally, to demonstrate the performance of MISO, we utilize 30 CEC2017 test functions and the CEC2022 test suite, comparing it with 11 popular algorithms across different dimensions to validate its effectiveness. Moreover, Unmanned Aerial Vehicle (UAV) has been widely used in various fields due to its advantages of low cost, high mobility and easy operation. However, the UAV path planning problem is crucial for flight safety and efficiency, and there are still challenges in establishing and optimizing the path model. Therefore, we apply MISO to the UAV 3D path planning problem as well as 6 engineering design problems to assess its feasibility in practical applications. The experimental results demonstrate that MISO exceeds other competitive algorithms in terms of solution quality and stability, establishing its strong potential for application. 

**Abstract (ZH)**: Metaheuristic 算法由于能够生成多样化的解决方案，在各个领域得到了广泛应用。其中一种算法是蛇优化器（Snake Optimizer，SO），它是一种渐进优化方法。然而，SO 存在收敛速度慢和易陷入局部最优的问题。鉴于这些不足，我们提出了一种新型的多策略改进蛇优化器（Multi-strategy Improved Snake Optimizer，MISO）。首先，我们提出了一种基于正弦函数的新自适应随机干扰策略，以缓解陷入局部最优的风险。其次，我们引入了基于尺度因子和领导者的自适应莱维飞行策略，并赋予雄蛇领导飞行能力，这使得算法更容易跳出局部最优找到全局最优。更重要的是，我们提出了结合精英领导和布朗运动的位置更新策略，有效地加速了收敛速度并保证了精度。此外，我们使用 30 个 CEC2017 测试函数和 CEC2022 测试集，与 11 种流行的算法在不同维度上进行比较，验证其有效性。同时，由于无人驾驶飞行器（UAV）因其低成本、高移动性和易操作性而在各个领域得到了广泛应用，而 UAV 航线规划对于飞行安全和效率至关重要，并且在建立和优化航线模型方面仍存在挑战。因此，我们将 MISO 应用于 UAV 的 3D 航线规划问题以及 6 个工程设计问题，评估其在实际应用中的可行性。实验结果表明，MISO 在解决方案质量和稳定性方面优于其他竞争算法，展示了其在实际应用中的强大潜力。 

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
# CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models 

**Title (ZH)**: CSD-VAR：视觉自回归模型中的内容-风格分解 

**Authors**: Quang-Binh Nguyen, Minh Luu, Quang Nguyen, Anh Tran, Khoi Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13984)  

**Abstract**: Disentangling content and style from a single image, known as content-style decomposition (CSD), enables recontextualization of extracted content and stylization of extracted styles, offering greater creative flexibility in visual synthesis. While recent personalization methods have explored the decomposition of explicit content style, they remain tailored for diffusion models. Meanwhile, Visual Autoregressive Modeling (VAR) has emerged as a promising alternative with a next-scale prediction paradigm, achieving performance comparable to that of diffusion models. In this paper, we explore VAR as a generative framework for CSD, leveraging its scale-wise generation process for improved disentanglement. To this end, we propose CSD-VAR, a novel method that introduces three key innovations: (1) a scale-aware alternating optimization strategy that aligns content and style representation with their respective scales to enhance separation, (2) an SVD-based rectification method to mitigate content leakage into style representations, and (3) an Augmented Key-Value (K-V) memory enhancing content identity preservation. To benchmark this task, we introduce CSD-100, a dataset specifically designed for content-style decomposition, featuring diverse subjects rendered in various artistic styles. Experiments demonstrate that CSD-VAR outperforms prior approaches, achieving superior content preservation and stylization fidelity. 

**Abstract (ZH)**: 从单张图像中解缠内容和风格：一种内容-风格分解方法（CSD-VAR）探讨 

---
# A segmented robot grasping perception neural network for edge AI 

**Title (ZH)**: 边缘AI中的分段机器人抓取感知神经网络 

**Authors**: Casper Bröcheler, Thomas Vroom, Derrick Timmermans, Alan van den Akker, Guangzhi Tang, Charalampos S. Kouzinopoulos, Rico Möckel  

**Link**: [PDF](https://arxiv.org/pdf/2507.13970)  

**Abstract**: Robotic grasping, the ability of robots to reliably secure and manipulate objects of varying shapes, sizes and orientations, is a complex task that requires precise perception and control. Deep neural networks have shown remarkable success in grasp synthesis by learning rich and abstract representations of objects. When deployed at the edge, these models can enable low-latency, low-power inference, making real-time grasping feasible in resource-constrained environments. This work implements Heatmap-Guided Grasp Detection, an end-to-end framework for the detection of 6-Dof grasp poses, on the GAP9 RISC-V System-on-Chip. The model is optimised using hardware-aware techniques, including input dimensionality reduction, model partitioning, and quantisation. Experimental evaluation on the GraspNet-1Billion benchmark validates the feasibility of fully on-chip inference, highlighting the potential of low-power MCUs for real-time, autonomous manipulation. 

**Abstract (ZH)**: 基于Heatmap引导的6-自由度抓取检测在GAP9 RISC-V系统芯片上的实现 

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
# Generalist Forecasting with Frozen Video Models via Latent Diffusion 

**Title (ZH)**: 通过潜扩散冻结视频模型的一般主义预测 

**Authors**: Jacob C Walker, Pedro Vélez, Luisa Polania Cabrera, Guangyao Zhou, Rishabh Kabra, Carl Doersch, Maks Ovsjanikov, João Carreira, Shiry Ginosar  

**Link**: [PDF](https://arxiv.org/pdf/2507.13942)  

**Abstract**: Forecasting what will happen next is a critical skill for general-purpose systems that plan or act in the world at different levels of abstraction. In this paper, we identify a strong correlation between a vision model's perceptual ability and its generalist forecasting performance over short time horizons. This trend holds across a diverse set of pretrained models-including those trained generatively-and across multiple levels of abstraction, from raw pixels to depth, point tracks, and object motion. The result is made possible by a novel generalist forecasting framework that operates on any frozen vision backbone: we train latent diffusion models to forecast future features in the frozen representation space, which are then decoded via lightweight, task-specific readouts. To enable consistent evaluation across tasks, we introduce distributional metrics that compare distributional properties directly in the space of downstream tasks and apply this framework to nine models and four tasks. Our results highlight the value of bridging representation learning and generative modeling for temporally grounded video understanding. 

**Abstract (ZH)**: 视觉模型的感知能力与其在短期时间范围内的通用预测性能之间存在密切关联：一个新型通用预测框架的研究 

---
# Convergent transformations of visual representation in brains and models 

**Title (ZH)**: 视觉表示在大脑和模型中的收敛转换 

**Authors**: Pablo Marcos-Manchón, Lluís Fuentemilla  

**Link**: [PDF](https://arxiv.org/pdf/2507.13941)  

**Abstract**: A fundamental question in cognitive neuroscience is what shapes visual perception: the external world's structure or the brain's internal architecture. Although some perceptual variability can be traced to individual differences, brain responses to naturalistic stimuli evoke similar activity patterns across individuals, suggesting a convergent representational principle. Here, we test if this stimulus-driven convergence follows a common trajectory across people and deep neural networks (DNNs) during its transformation from sensory to high-level internal representations. We introduce a unified framework that traces representational flow by combining inter-subject similarity with alignment to model hierarchies. Applying this framework to three independent fMRI datasets of visual scene perception, we reveal a cortex-wide network, conserved across individuals, organized into two pathways: a medial-ventral stream for scene structure and a lateral-dorsal stream tuned for social and biological content. This functional organization is captured by the hierarchies of vision DNNs but not language models, reinforcing the specificity of the visual-to-semantic transformation. These findings show a convergent computational solution for visual encoding in both human and artificial vision, driven by the structure of the external world. 

**Abstract (ZH)**: 认知神经科学中的一个基本问题是视觉感知是由外部世界的结构还是大脑的内部架构所塑造的。尽管一些知觉变异可以归因于个体差异，但大脑对自然刺激的响应在个体之间唤起类似的大脑活动模式，这表明存在一种趋同的表征原则。在这里，我们测试这种由刺激驱动的趋同性在从感觉级到高级内部表征的转化过程中是否遵循一个共同轨迹，跨越不同的人和深层神经网络（DNNs）。我们提出了一种统一的框架，通过结合个体间相似性和模型层级对齐来追踪表征流。将这一框架应用于三个独立的功能磁共振成像（fMRI）数据集的视觉场景感知中，我们揭示了一个跨个体保守的大脑网络，分为两条路径：一条背内侧流专注于场景结构，另一条背外侧流专精于社会和生物内容。这种功能组织在视觉DNNs的层级中得以捕捉，但在语言模型中并未体现，这强化了视觉到语义变换的特异性。这些发现表明，在人类和人工视觉中，外部世界结构驱动了一种趋同的计算解决方案。 

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
# Political Leaning and Politicalness Classification of Texts 

**Title (ZH)**: 政治倾向与文本的政治化分类 

**Authors**: Matous Volf, Jakub Simko  

**Link**: [PDF](https://arxiv.org/pdf/2507.13913)  

**Abstract**: This paper addresses the challenge of automatically classifying text according to political leaning and politicalness using transformer models. We compose a comprehensive overview of existing datasets and models for these tasks, finding that current approaches create siloed solutions that perform poorly on out-of-distribution texts. To address this limitation, we compile a diverse dataset by combining 12 datasets for political leaning classification and creating a new dataset for politicalness by extending 18 existing datasets with the appropriate label. Through extensive benchmarking with leave-one-in and leave-one-out methodologies, we evaluate the performance of existing models and train new ones with enhanced generalization capabilities. 

**Abstract (ZH)**: 本文利用变压器模型自动分类根据政治倾向和政治性对文本进行分类，面临着挑战。我们对这些任务现有的数据集和模型进行了全面综述，发现当前的方法创建了孤立的解决方案，在处理未见过分布的文本时表现较差。为了解决这一局限，我们通过组合12个政治倾向分类数据集，并通过扩展18个现有数据集以添加适当的标签来创建一个新的政治性数据集，来构建一个多样化的数据集。通过使用leave-one-in和leave-one-out的方法进行广泛的基准测试，我们评估了现有模型的性能，并训练了具有增强泛化能力的新模型。 

---
# Self-supervised learning on gene expression data 

**Title (ZH)**: 自主学习在基因表达数据中的应用 

**Authors**: Kevin Dradjat, Massinissa Hamidi, Pierre Bartet, Blaise Hanczar  

**Link**: [PDF](https://arxiv.org/pdf/2507.13912)  

**Abstract**: Predicting phenotypes from gene expression data is a crucial task in biomedical research, enabling insights into disease mechanisms, drug responses, and personalized medicine. Traditional machine learning and deep learning rely on supervised learning, which requires large quantities of labeled data that are costly and time-consuming to obtain in the case of gene expression data. Self-supervised learning has recently emerged as a promising approach to overcome these limitations by extracting information directly from the structure of unlabeled data. In this study, we investigate the application of state-of-the-art self-supervised learning methods to bulk gene expression data for phenotype prediction. We selected three self-supervised methods, based on different approaches, to assess their ability to exploit the inherent structure of the data and to generate qualitative representations which can be used for downstream predictive tasks. By using several publicly available gene expression datasets, we demonstrate how the selected methods can effectively capture complex information and improve phenotype prediction accuracy. The results obtained show that self-supervised learning methods can outperform traditional supervised models besides offering significant advantage by reducing the dependency on annotated data. We provide a comprehensive analysis of the performance of each method by highlighting their strengths and limitations. We also provide recommendations for using these methods depending on the case under study. Finally, we outline future research directions to enhance the application of self-supervised learning in the field of gene expression data analysis. This study is the first work that deals with bulk RNA-Seq data and self-supervised learning. 

**Abstract (ZH)**: 从基因表达数据预测表型是生物医学研究中的一个关键任务，有助于了解疾病机制、药物反应和个人化医疗。传统的机器学习和深度学习依赖于监督学习，这需要大量成本高昂且耗时的标记数据，而在基因表达数据的情况下尤为如此。最近，自监督学习作为一种有前途的方法出现，通过直接从未标记数据的结构中提取信息来克服这些限制。在本研究中，我们调查了使用最先进的自监督学习方法对批量基因表达数据进行表型预测的应用。我们选择了三种基于不同方法的自监督方法，以评估其利用数据内在结构的能力以及生成可用于下游预测任务的定性表示的能力。通过使用几个公开可用的基因表达数据集，我们展示了所选方法如何有效地捕捉复杂信息并提高表型预测准确性。获得的结果表明，自监督学习方法不仅能够超越传统监督模型，而且还通过减少对标注数据的依赖提供了显著优势。我们对每种方法的性能进行了全面分析，强调了它们的优势和局限性。我们还根据研究案例提供了使用这些方法的建议。最后，我们概述了未来的研究方向，以增强自监督学习在基因表达数据分析领域的应用。这是首次处理批量RNA-Seq数据和自监督学习的工作。 

---
# Using LLMs to identify features of personal and professional skills in an open-response situational judgment test 

**Title (ZH)**: 使用大语言模型识别开放 réponse 情景判断测试中个人和专业技能特征 

**Authors**: Cole Walsh, Rodica Ivan, Muhammad Zafar Iqbal, Colleen Robb  

**Link**: [PDF](https://arxiv.org/pdf/2507.13881)  

**Abstract**: Academic programs are increasingly recognizing the importance of personal and professional skills and their critical role alongside technical expertise in preparing students for future success in diverse career paths. With this growing demand comes the need for scalable systems to measure, evaluate, and develop these skills. Situational Judgment Tests (SJTs) offer one potential avenue for measuring these skills in a standardized and reliable way, but open-response SJTs have traditionally relied on trained human raters for evaluation, presenting operational challenges to delivering SJTs at scale. Past attempts at developing NLP-based scoring systems for SJTs have fallen short due to issues with construct validity of these systems. In this article, we explore a novel approach to extracting construct-relevant features from SJT responses using large language models (LLMs). We use the Casper SJT to demonstrate the efficacy of this approach. This study sets the foundation for future developments in automated scoring for personal and professional skills. 

**Abstract (ZH)**: 学术项目越来越认识到个人与职业技能的重要性，以及这些技能与技术专长一同在为学生多元化职业路径的成功做准备中起到的关键作用。随着这一需求的增长，需要有可扩展的系统来衡量、评估和发展这些技能。情境判断测试（SJT）提供了一种在标准化和可靠的方式下测量这些技能的潜在途径，但传统的开放式SJT评分依赖训练有素的人类评分员，这在大规模实施SJT时提出了操作挑战。过去基于NLP的SJT评分系统的开发由于这些系统构建效度的问题而未能成功。本文探讨了使用大规模语言模型（LLM）从SJT回答中提取构建相关特征的新型方法。我们使用Casper SJT来证明这一方法的有效性。本研究为未来个人和职业技能的自动化评分发展奠定了基础。 

---
# Real-Time Fusion of Visual and Chart Data for Enhanced Maritime Vision 

**Title (ZH)**: 基于视觉和图表数据的实时融合以增强 maritime 视觉能力 

**Authors**: Marten Kreis, Benjamin Kiefer  

**Link**: [PDF](https://arxiv.org/pdf/2507.13880)  

**Abstract**: This paper presents a novel approach to enhancing marine vision by fusing real-time visual data with chart information. Our system overlays nautical chart data onto live video feeds by accurately matching detected navigational aids, such as buoys, with their corresponding representations in chart data. To achieve robust association, we introduce a transformer-based end-to-end neural network that predicts bounding boxes and confidence scores for buoy queries, enabling the direct matching of image-domain detections with world-space chart markers. The proposed method is compared against baseline approaches, including a ray-casting model that estimates buoy positions via camera projection and a YOLOv7-based network extended with a distance estimation module. Experimental results on a dataset of real-world maritime scenes demonstrate that our approach significantly improves object localization and association accuracy in dynamic and challenging environments. 

**Abstract (ZH)**: 本文提出了一种通过融合实时视觉数据与航海图信息来提升海上视觉的新方法。我们的系统通过准确匹配检测到的助航标志（如浮标）与其航海图数据中的对应表示，将航海图数据叠加到实时视频流上。为了实现稳健的关联，我们提出了一种基于变换器的端到端神经网络，该网络预测浮标查询的边界框和置信分数，从而直接将图像域检测与世界空间中的航海图标记进行匹配。所提出的方法与基准方法进行了比较，包括一种光线投射模型，该模型通过相机投影估计浮标位置以及扩展了距离估计模块的YOLOv7网络。实验结果表明，本方法在动态和具有挑战性的环境中显著提高了物体的定位和关联准确性。 

---
# When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models 

**Title (ZH)**: 当视觉超越认知：分离视觉语言模型中的知识冲突 

**Authors**: Francesco Ortu, Zhijing Jin, Diego Doimo, Alberto Cazzaniga  

**Link**: [PDF](https://arxiv.org/pdf/2507.13868)  

**Abstract**: Vision-language models (VLMs) increasingly leverage diverse knowledge sources to address complex tasks, often encountering conflicts between their internal parametric knowledge and external information. Knowledge conflicts can result in hallucinations and unreliable responses, but the mechanisms governing such interactions remain unknown. To address this gap, we analyze the mechanisms that VLMs use to resolve cross-modal conflicts by introducing a dataset of multimodal counterfactual queries that deliberately contradict internal commonsense knowledge. We localize with logit inspection a small set of heads that control the conflict. Moreover, by modifying these heads, we can steer the model towards its internal knowledge or the visual inputs. Finally, we show that attention from such heads pinpoints localized image regions driving visual overrides, outperforming gradient-based attribution in precision. 

**Abstract (ZH)**: 视觉-语言模型通过引入故意与内部常识知识矛盾的多模态反事实查询数据集，分析其解决跨模态冲突的机制，并通过logit检查定位控制冲突的少量头部。通过修改这些头部，可以引导模型向内部知识或视觉输入靠拢。此外，我们展示这些头部的注意力能精确定位驱动视觉覆盖的局部图像区域，优于基于梯度的归因方法。 

---
# SPARQL Query Generation with LLMs: Measuring the Impact of Training Data Memorization and Knowledge Injection 

**Title (ZH)**: 基于LLMs的SPARQL查询生成：训练数据记忆和知识注入的影响测量 

**Authors**: Aleksandr Gashkov, Aleksandr Perevalov, Maria Eltsova, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2507.13859)  

**Abstract**: Nowadays, the importance of software with natural-language user interfaces cannot be underestimated. In particular, in Question Answering (QA) systems, generating a SPARQL query for a given natural-language question (often named Query Building) from the information retrieved from the same question is the central task of QA systems working over Knowledge Graphs (KGQA). Due to the rise of Large Language Models (LLMs), they are considered a well-suited method to increase the quality of the question-answering functionality, as there is still a lot of room for improvement, aiming for enhanced quality and trustworthiness. However, LLMs are trained on web data, where researchers have no control over whether the benchmark or the knowledge graph was already included in the training data. In this paper, we introduce a novel method that evaluates the quality of LLMs by generating a SPARQL query from a natural-language question under various conditions: (1) zero-shot SPARQL generation, (2) with knowledge injection, and (3) with "anonymized" knowledge injection. This enables us, for the first time, to estimate the influence of the training data on the QA quality improved by LLMs. Ultimately, this will help to identify how portable a method is or whether good results might mostly be achieved because a benchmark was already included in the training data (cf. LLM memorization). The developed method is portable, robust, and supports any knowledge graph; therefore, it could be easily applied to any KGQA or LLM, s.t., generating consistent insights into the actual LLM capabilities is possible. 

**Abstract (ZH)**: 现今，具有自然语言用户接口的软件的重要性不容忽视。特别是，在基于知识图谱的问答系统（KGQA）中的问答系统中，从同一问题检索到的信息生成给定的自然语言问题的SPARQL查询（通常称为查询构建）是核心任务。由于大型语言模型（LLMs）的兴起，它们被认为是一种提高问答功能质量的合适方法，尽管还有很大的改进空间，以增强质量和可信度。然而，LLMs是在网络数据上进行训练的，研究人员无法控制基准或知识图谱是否已在训练数据中包含。在此论文中，我们提出了一种新型方法，通过在不同条件下生成自然语言问题的SPARQL查询来评估LLMs的质量：（1）零样本SPARQL生成，（2）带知识注入，（3）带“匿名”知识注入。这使我们首次能够估计训练数据对由LLMs提高的问答质量的影响。最终，这将有助于识别方法的适用性，或确定良好结果是否主要是因为基准已经在训练数据中包含（类似于LLM记忆）。所开发的方法是可移植的、稳健的，并支持任何知识图谱；因此，它可以很容易地应用于任何KGQA或LLM，以便获得关于实际LLM能力的一致见解。 

---
# Scalable Submodular Policy Optimization via Pruned Submodularity Graph 

**Title (ZH)**: 可扩展的子模性政策优化方法：剪枝子模性图 

**Authors**: Aditi Anand, Suman Banerjee, Dildar Ali  

**Link**: [PDF](https://arxiv.org/pdf/2507.13834)  

**Abstract**: In Reinforcement Learning (abbreviated as RL), an agent interacts with the environment via a set of possible actions, and a reward is generated from some unknown distribution. The task here is to find an optimal set of actions such that the reward after a certain time step gets maximized. In a traditional setup, the reward function in an RL Problem is considered additive. However, in reality, there exist many problems, including path planning, coverage control, etc., the reward function follows the diminishing return, which can be modeled as a submodular function. In this paper, we study a variant of the RL Problem where the reward function is submodular, and our objective is to find an optimal policy such that this reward function gets maximized. We have proposed a pruned submodularity graph-based approach that provides a provably approximate solution in a feasible computation time. The proposed approach has been analyzed to understand its time and space requirements as well as a performance guarantee. We have experimented with a benchmark agent-environment setup, which has been used for similar previous studies, and the results are reported. From the results, we observe that the policy obtained by our proposed approach leads to more reward than the baseline methods. 

**Abstract (ZH)**: 在强化学习中，代理通过一系列可能的动作与环境互动，并从未知分布中生成奖励。任务是在某个时间步后使奖励最大化，找到最优动作集。在传统设置中，强化学习问题中的奖励函数被认为是可加的。然而，在现实中有许多问题，如路径规划、覆盖控制等，奖励函数遵循递减回报规律，可以建模为亚模函数。本文研究了奖励函数为亚模函数的强化学习问题变体，目标是找到使该奖励函数最大化的优势策略。我们提出了一种剪枝亚模性图为基础的方法，能够在合理的时间内提供可证明近似解。该方法已被分析以理解其时间与空间要求以及性能保证。我们使用了一个基准代理-环境设置进行实验，该设置在之前类似研究中被使用，并报告了实验结果。结果显示，我们提出的方法获得的策略产生的奖励超过了基线方法。 

---
# RAG-based Architectures for Drug Side Effect Retrieval in LLMs 

**Title (ZH)**: 基于RAG的架构在LLMs中用于药物副作用检索 

**Authors**: Shad Nygren, Pinar Avci, Andre Daniels, Reza Rassol, Afshin Beheshti, Diego Galeano  

**Link**: [PDF](https://arxiv.org/pdf/2507.13822)  

**Abstract**: Drug side effects are a major global health concern, necessitating advanced methods for their accurate detection and analysis. While Large Language Models (LLMs) offer promising conversational interfaces, their inherent limitations, including reliance on black-box training data, susceptibility to hallucinations, and lack of domain-specific knowledge, hinder their reliability in specialized fields like pharmacovigilance. To address this gap, we propose two architectures: Retrieval-Augmented Generation (RAG) and GraphRAG, which integrate comprehensive drug side effect knowledge into a Llama 3 8B language model. Through extensive evaluations on 19,520 drug side effect associations (covering 976 drugs and 3,851 side effect terms), our results demonstrate that GraphRAG achieves near-perfect accuracy in drug side effect retrieval. This framework offers a highly accurate and scalable solution, signifying a significant advancement in leveraging LLMs for critical pharmacovigilance applications. 

**Abstract (ZH)**: 药物副作用是全球健康的重大关切，亟需先进的方法用于其准确检测和分析。虽然大型语言模型（LLMs）提供了有前景的对话界面，但它们的固有限制，包括依赖于黑盒训练数据、易产生幻觉以及缺乏特定领域的知识， hindering其在像药物警戒这样专门领域的可靠性。为了解决这一问题，我们提出了两种架构：检索增强生成（RAG）和GraphRAG，这些架构将全面的药物副作用知识整合到了Llama 3 8B语言模型中。通过在19,520个药物副作用关联（涵盖976种药物和3,851种副作用术语）上的广泛评估，我们的结果显示GraphRAG在药物副作用检索方面达到了近乎完美的准确度。该框架提供了高度准确且可扩展的解决方案，标志着在利用LLMs进行关键药物警戒应用方面取得了重要进展。 

---
# Team of One: Cracking Complex Video QA with Model Synergy 

**Title (ZH)**: 单人团队：通过模型协同解决复杂视频QA 

**Authors**: Jun Xie, Zhaoran Zhao, Xiongjun Guan, Yingjian Zhu, Hongzhu Yi, Xinming Wang, Feng Chen, Zhepeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13820)  

**Abstract**: We propose a novel framework for open-ended video question answering that enhances reasoning depth and robustness in complex real-world scenarios, as benchmarked on the CVRR-ES dataset. Existing Video-Large Multimodal Models (Video-LMMs) often exhibit limited contextual understanding, weak temporal modeling, and poor generalization to ambiguous or compositional queries. To address these challenges, we introduce a prompting-and-response integration mechanism that coordinates multiple heterogeneous Video-Language Models (VLMs) via structured chains of thought, each tailored to distinct reasoning pathways. An external Large Language Model (LLM) serves as an evaluator and integrator, selecting and fusing the most reliable responses. Extensive experiments demonstrate that our method significantly outperforms existing baselines across all evaluation metrics, showcasing superior generalization and robustness. Our approach offers a lightweight, extensible strategy for advancing multimodal reasoning without requiring model retraining, setting a strong foundation for future Video-LMM development. 

**Abstract (ZH)**: 我们提出了一种新的框架，用于开放式视频问答，该框架在CVRR-ES数据集上的实际复杂场景中增强了推理深度和鲁棒性。现有的视频大型多模态模型（Video-LMMs）往往表现出有限的语境理解、弱的时间建模能力和对模糊或组合查询的不良泛化能力。为了解决这些挑战，我们引入了一种提示与响应集成机制，通过结构化的思维链协调多个异构的视频语言模型（VLMs），每种模型针对不同的推理路径进行定制。一个外部的大语言模型（LLM）作为评估器和集成器，选择并融合最可靠的回答。广泛实验表明，我们的方法在所有评估指标上显著优于现有基线，展示了更强的泛化能力和鲁棒性。我们的方法提供了一种轻量级且可扩展的策略，无需重新训练模型即可推进多模态推理，为未来Video-LMM的发展奠定了坚实的基础。 

---
# Food safety trends across Europe: insights from the 392-million-entry CompreHensive European Food Safety (CHEFS) database 

**Title (ZH)**: 欧洲食品安全趋势分析：CompreHensive European Food Safety (CHEFS) 数据库3920万条记录的洞察 

**Authors**: Nehir Kizililsoley, Floor van Meer, Osman Mutlu, Wouter F Hoenderdaal, Rosan G. Hobé, Wenjuan Mu, Arjen Gerssen, H.J. van der Fels-Klerx, Ákos Jóźwiak, Ioannis Manikas, Ali Hürriyetoǧlu, Bas H.M. van der Velden  

**Link**: [PDF](https://arxiv.org/pdf/2507.13802)  

**Abstract**: In the European Union, official food safety monitoring data collected by member states are submitted to the European Food Safety Authority (EFSA) and published on Zenodo. This data includes 392 million analytical results derived from over 15.2 million samples covering more than 4,000 different types of food products, offering great opportunities for artificial intelligence to analyze trends, predict hazards, and support early warning systems. However, the current format with data distributed across approximately 1000 files totaling several hundred gigabytes hinders accessibility and analysis. To address this, we introduce the CompreHensive European Food Safety (CHEFS) database, which consolidates EFSA monitoring data on pesticide residues, veterinary medicinal product residues, and chemical contaminants into a unified and structured dataset. We describe the creation and structure of the CHEFS database and demonstrate its potential by analyzing trends in European food safety monitoring data from 2000 to 2024. Our analyses explore changes in monitoring activities, the most frequently tested products, which products were most often non-compliant and which contaminants were most often found, and differences across countries. These findings highlight the CHEFS database as both a centralized data source and a strategic tool for guiding food safety policy, research, and regulation. 

**Abstract (ZH)**: 欧盟全面食品安全数据库：整合欧洲食品安全局监测数据以支持人工智能分析和政策指导 

---
# One Step Closer: Creating the Future to Boost Monocular Semantic Scene Completion 

**Title (ZH)**: 一步之遥：创造未来以提升单目语义场景完成 

**Authors**: Haoang Lu, Yuanqi Su, Xiaoning Zhang, Hao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13801)  

**Abstract**: In recent years, visual 3D Semantic Scene Completion (SSC) has emerged as a critical perception task for autonomous driving due to its ability to infer complete 3D scene layouts and semantics from single 2D images. However, in real-world traffic scenarios, a significant portion of the scene remains occluded or outside the camera's field of view -- a fundamental challenge that existing monocular SSC methods fail to address adequately. To overcome these limitations, we propose Creating the Future SSC (CF-SSC), a novel temporal SSC framework that leverages pseudo-future frame prediction to expand the model's effective perceptual range. Our approach combines poses and depths to establish accurate 3D correspondences, enabling geometrically-consistent fusion of past, present, and predicted future frames in 3D space. Unlike conventional methods that rely on simple feature stacking, our 3D-aware architecture achieves more robust scene completion by explicitly modeling spatial-temporal relationships. Comprehensive experiments on SemanticKITTI and SSCBench-KITTI-360 benchmarks demonstrate state-of-the-art performance, validating the effectiveness of our approach, highlighting our method's ability to improve occlusion reasoning and 3D scene completion accuracy. 

**Abstract (ZH)**: 近年来，视觉3D语义场景完成（SSC）已成为自动驾驶中一项关键的感知任务，由于其能够从单张2D图像推断出完整的3D场景布局和语义。然而，在现实世界的交通场景中，场景中仍有一部分区域被遮挡或位于相机视野之外——这是一个现有单目SSC方法难以解决的基本挑战。为克服这些局限性，我们提出了一种名为Creating the Future SSC（CF-SSC）的新颖的时序SSC框架，该框架利用伪未来帧预测来扩展模型的有效感知范围。我们的方法结合姿势和深度来建立准确的3D对应关系，能够在3D空间中几何一致地融合过去、现在和预测的未来帧。与依赖简单特征堆叠的传统方法不同，我们提出的3D感知架构通过明确建模空-时关系实现了更稳健的场景完成。在SemanticKITTI和SSCBench-KITTI-360基准上的全面实验表明，我们的方法达到了最先进的性能，验证了其有效性，突显了我们方法在改善遮挡推理和3D场景完成准确性方面的优势。 

---
# Localized FNO for Spatiotemporal Hemodynamic Upsampling in Aneurysm MRI 

**Title (ZH)**: 局部化FNO在动脉瘤MRI中用于时空血流动力学上采样 

**Authors**: Kyriakos Flouris, Moritz Halter, Yolanne Y. R. Lee, Samuel Castonguay, Luuk Jacobs, Pietro Dirix, Jonathan Nestmann, Sebastian Kozerke, Ender Konukoglu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13789)  

**Abstract**: Hemodynamic analysis is essential for predicting aneurysm rupture and guiding treatment. While magnetic resonance flow imaging enables time-resolved volumetric blood velocity measurements, its low spatiotemporal resolution and signal-to-noise ratio limit its diagnostic utility. To address this, we propose the Localized Fourier Neural Operator (LoFNO), a novel 3D architecture that enhances both spatial and temporal resolution with the ability to predict wall shear stress (WSS) directly from clinical imaging data. LoFNO integrates Laplacian eigenvectors as geometric priors for improved structural awareness on irregular, unseen geometries and employs an Enhanced Deep Super-Resolution Network (EDSR) layer for robust upsampling. By combining geometric priors with neural operator frameworks, LoFNO de-noises and spatiotemporally upsamples flow data, achieving superior velocity and WSS predictions compared to interpolation and alternative deep learning methods, enabling more precise cerebrovascular diagnostics. 

**Abstract (ZH)**: 血流动力学分析对于预测动脉瘤破裂和指导治疗至关重要。尽管磁共振流成像能够进行时间解析的体积血液速度测量，但由于其较低的空间-时间分辨率和信噪比，其诊断用途受到限制。为了解决这一问题，我们提出了一种新型的3D架构——局部傅里叶神经算子（LoFNO），它能够增强空间和时间分辨率，并能够直接从临床影像数据中预测壁剪应力（WSS）。LoFNO将Laplacian特征向量作为几何先验，用于改善对不规则、未见过的几何结构的结构认知，并采用增强的深度超分辨率网络（EDSR）层进行稳健的上采样。通过结合几何先验与神经算子框架，LoFNO对流数据进行去噪和空间-时间上采样，实现了与插值和替代深度学习方法相比更优的速度和WSS预测，从而提高了脑血管诊断的精确性。 

---
# Learning Spectral Diffusion Prior for Hyperspectral Image Reconstruction 

**Title (ZH)**: 学习谱扩散先验用于高光谱图像重建 

**Authors**: Mingyang Yu, Zhijian Wu, Dingjiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13769)  

**Abstract**: Hyperspectral image (HSI) reconstruction aims to recover 3D HSI from its degraded 2D measurements. Recently great progress has been made in deep learning-based methods, however, these methods often struggle to accurately capture high-frequency details of the HSI. To address this issue, this paper proposes a Spectral Diffusion Prior (SDP) that is implicitly learned from hyperspectral images using a diffusion model. Leveraging the powerful ability of the diffusion model to reconstruct details, this learned prior can significantly improve the performance when injected into the HSI model. To further improve the effectiveness of the learned prior, we also propose the Spectral Prior Injector Module (SPIM) to dynamically guide the model to recover the HSI details. We evaluate our method on two representative HSI methods: MST and BISRNet. Experimental results show that our method outperforms existing networks by about 0.5 dB, effectively improving the performance of HSI reconstruction. 

**Abstract (ZH)**: 高光谱图像（HSI）恢复旨在从退化的2D测量中恢复3D HSI。近期基于深度学习的方法取得了显著进展，但这些方法往往难以准确捕捉HSI的高频细节。为解决这一问题，本文提出了一种隐式学习自高光谱图像的光谱扩散先验（SDP），利用扩散模型的强大重建细节能力，该学习先验在注入到HSI模型中时能显著提升性能。为了进一步提高学习先验的有效性，本文还提出了光谱先验注入模块（SPIM）以动态引导模型恢复HSI的细节。我们在两种代表性HSI方法：MST和BISRNet上评估了该方法。实验结果表明，与现有网络相比，我们的方法在性能上提升了约0.5 dB，有效提高了HSI恢复的效果。 

---
# Search-Optimized Quantization in Biomedical Ontology Alignment 

**Title (ZH)**: 搜索优化量化在生物医学本体对齐中的应用 

**Authors**: Oussama Bouaggad, Natalia Grabar  

**Link**: [PDF](https://arxiv.org/pdf/2507.13742)  

**Abstract**: In the fast-moving world of AI, as organizations and researchers develop more advanced models, they face challenges due to their sheer size and computational demands. Deploying such models on edge devices or in resource-constrained environments adds further challenges related to energy consumption, memory usage and latency. To address these challenges, emerging trends are shaping the future of efficient model optimization techniques. From this premise, by employing supervised state-of-the-art transformer-based models, this research introduces a systematic method for ontology alignment, grounded in cosine-based semantic similarity between a biomedical layman vocabulary and the Unified Medical Language System (UMLS) Metathesaurus. It leverages Microsoft Olive to search for target optimizations among different Execution Providers (EPs) using the ONNX Runtime backend, followed by an assembled process of dynamic quantization employing Intel Neural Compressor and IPEX (Intel Extension for PyTorch). Through our optimization process, we conduct extensive assessments on the two tasks from the DEFT 2020 Evaluation Campaign, achieving a new state-of-the-art in both. We retain performance metrics intact, while attaining an average inference speed-up of 20x and reducing memory usage by approximately 70%. 

**Abstract (ZH)**: 在快速发展的AI领域，随着组织和研究人员开发出更先进的模型，他们面临着由于模型的巨大规模和计算需求所带来的挑战。将这些模型部署在边缘设备或资源受限的环境中，进一步增加了与能耗、内存使用和延迟相关的问题。为了应对这些挑战，新兴趋势正在塑造高效模型优化技术的未来。在此基础上，通过采用监督学习的最先进的变压器模型，本研究介绍了一种系统的方法来进行本体对齐，该方法基于生物医学通俗词汇与统一医疗语言系统（UMLS）梅达索拉斯之间的余弦相似度。该研究利用Microsoft Olive在不同的执行提供者（EPs）之间搜索目标优化，并通过ONNX Runtime后端，结合使用Intel Neural Compressor和IPEX（Intel扩展的PyTorch）进行动态量化。通过我们的优化过程，我们在DEFT 2020评估竞选中的两个任务上进行了广泛评估，实现了两项新的最佳性能。我们保持了性能指标的完整性，同时实现了平均推理速度提升20倍，并将内存使用量减少了约70%。 

---
# SamGoG: A Sampling-Based Graph-of-Graphs Framework for Imbalanced Graph Classification 

**Title (ZH)**: SamGoG: 一种基于采样的图集合分类框架，用于不平衡图分类 

**Authors**: Shangyou Wang, Zezhong Ding, Xike Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.13741)  

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable success in graph classification tasks by capturing both structural and feature-based representations. However, real-world graphs often exhibit two critical forms of imbalance: class imbalance and graph size imbalance. These imbalances can bias the learning process and degrade model performance. Existing methods typically address only one type of imbalance or incur high computational costs. In this work, we propose SamGoG, a sampling-based Graph-of-Graphs (GoG) learning framework that effectively mitigates both class and graph size imbalance. SamGoG constructs multiple GoGs through an efficient importance-based sampling mechanism and trains on them sequentially. This sampling mechanism incorporates the learnable pairwise similarity and adaptive GoG node degree to enhance edge homophily, thus improving downstream model quality. SamGoG can seamlessly integrate with various downstream GNNs, enabling their efficient adaptation for graph classification tasks. Extensive experiments on benchmark datasets demonstrate that SamGoG achieves state-of-the-art performance with up to a 15.66% accuracy improvement with 6.7$\times$ training acceleration. 

**Abstract (ZH)**: 基于采样的Graph-of-Graphs (GoG) 学习框架：同时缓解类别不平衡和图大小不平衡 

---
# Can Synthetic Images Conquer Forgetting? Beyond Unexplored Doubts in Few-Shot Class-Incremental Learning 

**Title (ZH)**: 合成图像能否克服遗忘？超越少样本类增量学习中的未探索疑虑 

**Authors**: Junsu Kim, Yunhoe Ku, Seungryul Baek  

**Link**: [PDF](https://arxiv.org/pdf/2507.13739)  

**Abstract**: Few-shot class-incremental learning (FSCIL) is challenging due to extremely limited training data; while aiming to reduce catastrophic forgetting and learn new information. We propose Diffusion-FSCIL, a novel approach that employs a text-to-image diffusion model as a frozen backbone. Our conjecture is that FSCIL can be tackled using a large generative model's capabilities benefiting from 1) generation ability via large-scale pre-training; 2) multi-scale representation; 3) representational flexibility through the text encoder. To maximize the representation capability, we propose to extract multiple complementary diffusion features to play roles as latent replay with slight support from feature distillation for preventing generative biases. Our framework realizes efficiency through 1) using a frozen backbone; 2) minimal trainable components; 3) batch processing of multiple feature extractions. Extensive experiments on CUB-200, \emph{mini}ImageNet, and CIFAR-100 show that Diffusion-FSCIL surpasses state-of-the-art methods, preserving performance on previously learned classes and adapting effectively to new ones. 

**Abstract (ZH)**: 少量样本类别增量学习（Few-shot Class-incremental Learning, FSCIL）由于训练数据极其有限而具有挑战性；本研究旨在减少灾难性遗忘并学习新信息。我们提出了一种新颖的方法——Diffusion-FSCIL，该方法采用文本到图像的扩散模型作为冻结骨干。我们认为，可以利用大型生成模型的能力来应对FSCIL问题，得益于1）大规模预训练带来的生成能力；2）多尺度表示；3）通过文本编码器实现的表示灵活性。为了最大化表示能力，我们提出从大型生成模型中提取多个互补的扩散特征，作为潜在再现，并通过特征蒸馏轻微支持，以防止生成偏差。我们的框架通过1）使用冻结骨干；2）最小可训练组件；3）批量处理多个特征提取来实现效率。在CUB-200、\emph{mini}ImageNet和CIFAR-100上的广泛实验表明，Diffusion-FSCIL超过了现有方法，既保持了之前学习类别的性能，又能有效适应新类别。 

---
# AGENTS-LLM: Augmentative GENeration of Challenging Traffic Scenarios with an Agentic LLM Framework 

**Title (ZH)**: AGENTS-LLM: 基于机构性大语言模型框架的具有挑战性的交通场景生成增强方法 

**Authors**: Yu Yao, Salil Bhatnagar, Markus Mazzola, Vasileios Belagiannis, Igor Gilitschenski, Luigi Palmieri, Simon Razniewski, Marcel Hallgarten  

**Link**: [PDF](https://arxiv.org/pdf/2507.13729)  

**Abstract**: Rare, yet critical, scenarios pose a significant challenge in testing and evaluating autonomous driving planners. Relying solely on real-world driving scenes requires collecting massive datasets to capture these scenarios. While automatic generation of traffic scenarios appears promising, data-driven models require extensive training data and often lack fine-grained control over the output. Moreover, generating novel scenarios from scratch can introduce a distributional shift from the original training scenes which undermines the validity of evaluations especially for learning-based planners. To sidestep this, recent work proposes to generate challenging scenarios by augmenting original scenarios from the test set. However, this involves the manual augmentation of scenarios by domain experts. An approach that is unable to meet the demands for scale in the evaluation of self-driving systems. Therefore, this paper introduces a novel LLM-agent based framework for augmenting real-world traffic scenarios using natural language descriptions, addressing the limitations of existing methods. A key innovation is the use of an agentic design, enabling fine-grained control over the output and maintaining high performance even with smaller, cost-effective LLMs. Extensive human expert evaluation demonstrates our framework's ability to accurately adhere to user intent, generating high quality augmented scenarios comparable to those created manually. 

**Abstract (ZH)**: 罕见但关键的场景在测试和评估自动驾驶规划器时构成了重大挑战。仅依赖真实-world驾驶场景需要收集大量数据集以捕获这些场景。尽管自动生成交通场景看起来很有前景，但数据驱动的模型需要广泛的训练数据，且往往缺乏对输出的精细控制。此外，从头生成新颖的场景可能会导致与原始训练场景分布的偏移，这尤其会对基于学习的规划器的评估有效性构成威胁。为解决这一问题，近期的研究提出了通过对测试集中原始场景进行人工扩展来生成具有挑战性的场景的方法。然而，这种方法涉及领域专家的手动场景扩展。这无法满足对自动驾驶系统评估规模的需求。因此，本文介绍了一种基于LLM代理的框架，利用自然语言描述扩充真实世界的交通场景，解决了现有方法的局限性。关键创新在于采用代理设计，这使得在使用较小且成本效益高的LLM时仍能实现对输出的精细控制，并保持高性能。广泛的专家评估证实了该框架能够准确遵循用户意图，生成与手动创建的场景质量相当的高质量扩充场景。 

---
# Point of Interest Recommendation: Pitfalls and Viable Solutions 

**Title (ZH)**: 兴趣点推荐：潜在问题与可行解决方案 

**Authors**: Alejandro Bellogín, Linus W. Dietz, Francesco Ricci, Pablo Sánchez  

**Link**: [PDF](https://arxiv.org/pdf/2507.13725)  

**Abstract**: Point of interest (POI) recommendation can play a pivotal role in enriching tourists' experiences by suggesting context-dependent and preference-matching locations and activities, such as restaurants, landmarks, itineraries, and cultural attractions. Unlike some more common recommendation domains (e.g., music and video), POI recommendation is inherently high-stakes: users invest significant time, money, and effort to search, choose, and consume these suggested POIs. Despite the numerous research works in the area, several fundamental issues remain unresolved, hindering the real-world applicability of the proposed approaches. In this paper, we discuss the current status of the POI recommendation problem and the main challenges we have identified. The first contribution of this paper is a critical assessment of the current state of POI recommendation research and the identification of key shortcomings across three main dimensions: datasets, algorithms, and evaluation methodologies. We highlight persistent issues such as the lack of standardized benchmark datasets, flawed assumptions in the problem definition and model design, and inadequate treatment of biases in the user behavior and system performance. The second contribution is a structured research agenda that, starting from the identified issues, introduces important directions for future work related to multistakeholder design, context awareness, data collection, trustworthiness, novel interactions, and real-world evaluation. 

**Abstract (ZH)**: 兴趣点（POI）推荐在丰富游客体验方面可以发挥关键作用，通过建议上下文依赖性和偏好匹配的位置和活动，如餐厅、地标、行程和文化景点。与一些更常见的推荐领域（例如音乐和视频）不同，POI推荐本质上风险较高：用户在搜索、选择和消费这些建议的POI上投入了大量时间、金钱和努力。尽管该领域已经进行了大量研究，但仍存在一些基础问题，阻碍了所提出方法的实际应用。在本文中，我们讨论了POI推荐问题的现状以及我们所识别的主要挑战。本文的第一个贡献是对当前POI推荐研究状态进行批判性评估，并在三个主要维度上识别关键不足之处：数据集、算法和评估方法。我们强调了持续存在的问题，如缺乏标准化基准数据集、问题定义和模型设计中的错误假设，以及用户行为和系统性能中的偏见处理不足。本文的第二个贡献是一个结构化研究议程，从识别的问题出发，引入了多利益相关者设计、情境意识、数据收集、可信度、新型交互和现实世界评估等相关未来工作的重要方向。 

---
# Binarizing Physics-Inspired GNNs for Combinatorial Optimization 

**Title (ZH)**: 基于物理启发的二值化GNNs在组合优化中的应用 

**Authors**: Martin Krutský, Gustav Šír, Vyacheslav Kungurtsev, Georgios Korpas  

**Link**: [PDF](https://arxiv.org/pdf/2507.13703)  

**Abstract**: Physics-inspired graph neural networks (PI-GNNs) have been utilized as an efficient unsupervised framework for relaxing combinatorial optimization problems encoded through a specific graph structure and loss, reflecting dependencies between the problem's variables. While the framework has yielded promising results in various combinatorial problems, we show that the performance of PI-GNNs systematically plummets with an increasing density of the combinatorial problem graphs. Our analysis reveals an interesting phase transition in the PI-GNNs' training dynamics, associated with degenerate solutions for the denser problems, highlighting a discrepancy between the relaxed, real-valued model outputs and the binary-valued problem solutions. To address the discrepancy, we propose principled alternatives to the naive strategy used in PI-GNNs by building on insights from fuzzy logic and binarized neural networks. Our experiments demonstrate that the portfolio of proposed methods significantly improves the performance of PI-GNNs in increasingly dense settings. 

**Abstract (ZH)**: 物理启发的图神经网络（PI-GNNs）已被用作一个高效的无监督框架，用于通过特定的图结构和损失函数来放松编码组合优化问题，反映变量之间的依赖关系。尽管该框架在各种组合问题中取得了令人鼓舞的结果，我们发现PI-GNNs在组合问题图密度增加时其性能系统性地下降。我们的分析揭示了PI-GNNs训练动力学中有趣的相转变，与更密集的问题相关的退化解，强调了放松的实值模型输出与二进制值问题解之间的差距。为了弥合这一差距，我们基于模糊逻辑和二值神经网络的见解，提出了一系列原则性的替代方案，以改进PI-GNNs的原始策略。实验结果表明，所提出方法的组合在日益密集的设置中显著提高了PI-GNNs的性能。 

---
# LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues 

**Title (ZH)**: LoopServe：一种适应性双阶段大语言模型推理加速系统用于多轮对话 

**Authors**: Haoyang Li, Zhanchao Xu, Yiming Li, Xuejia Chen, Darian Li, Anxin Tian, Qingfa Xiao, Cheng Deng, Jun Wang, Qing Li, Lei Chen, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.13681)  

**Abstract**: Multi-turn dialogues are essential in many real-world applications of large language models, such as chatbots and virtual assistants. As conversation histories become longer, existing large language models face increasing computational and memory challenges, which hinder their ability to provide efficient and responsive interactions. Most current acceleration methods either compress the context or optimize key value caching, but they often rely on fixed or position-based heuristics that do not adapt well to the dynamic and unpredictable patterns found in actual multi-turn conversations. In this paper, we present LoopServe, an adaptive dual-phase inference acceleration framework for large language models in multi-turn dialogues. LoopServe introduces two main innovations. First, it performs online sparsification during the prefilling phase by dynamically selecting the most important parts of the attention matrix for each new input. Second, it uses progressive key value compression during decoding by adaptively maintaining a relevant and efficient cache based on the most recently generated output tokens. We also propose a \href{this https URL}{new benchmark} with eleven multi-turn datasets that reflect realistic query positions and conversational dependencies. Extensive experiments demonstrate that LoopServe consistently achieves superior effectiveness compared to existing baselines and significantly accelerates LLM inference across a wide range of long-context dialogue tasks. 

**Abstract (ZH)**: 多轮对话是大型语言模型在聊天机器人和虚拟助手等许多现实世界应用中的 Essential 组件。随着对话历史的延长，现有大型语言模型面临着日益增加的计算和内存挑战，这妨碍了它们提供高效和及时的交互。大多数当前的加速方法要么压缩上下文，要么优化键值缓存，但它们往往依赖于固定或位置基的启发式方法，这些方法不能很好地适应实际多轮对话中动态和不可预测的模式。本文提出了 LoopServe，一种适应性的双阶段推理加速框架，用于多轮对话中的大型语言模型。LoopServe 引入了两项主要创新。首先，在预填充阶段通过动态选择每个新输入的最相关部分来执行在线稀疏化。其次，在解码过程中通过基于最近生成的输出令牌动态维护相关且高效的缓存来进行渐进的键值压缩。我们还提出了一种新的基准，包含十个反映实际查询位置和对话依赖性的多轮数据集。广泛的实验表明，LoopServe 在各种长上下文对话任务中的一致性有效性优于现有基线，并且显著加速了大型语言模型的推理。 

---
# HeCoFuse: Cross-Modal Complementary V2X Cooperative Perception with Heterogeneous Sensors 

**Title (ZH)**: HeCoFuse：异构传感器跨模态互补V2X协同感知 

**Authors**: Chuheng Wei, Ziye Qin, Walter Zimmer, Guoyuan Wu, Matthew J. Barth  

**Link**: [PDF](https://arxiv.org/pdf/2507.13677)  

**Abstract**: Real-world Vehicle-to-Everything (V2X) cooperative perception systems often operate under heterogeneous sensor configurations due to cost constraints and deployment variability across vehicles and infrastructure. This heterogeneity poses significant challenges for feature fusion and perception reliability. To address these issues, we propose HeCoFuse, a unified framework designed for cooperative perception across mixed sensor setups where nodes may carry Cameras (C), LiDARs (L), or both. By introducing a hierarchical fusion mechanism that adaptively weights features through a combination of channel-wise and spatial attention, HeCoFuse can tackle critical challenges such as cross-modality feature misalignment and imbalanced representation quality. In addition, an adaptive spatial resolution adjustment module is employed to balance computational cost and fusion effectiveness. To enhance robustness across different configurations, we further implement a cooperative learning strategy that dynamically adjusts fusion type based on available modalities. Experiments on the real-world TUMTraf-V2X dataset demonstrate that HeCoFuse achieves 43.22% 3D mAP under the full sensor configuration (LC+LC), outperforming the CoopDet3D baseline by 1.17%, and reaches an even higher 43.38% 3D mAP in the L+LC scenario, while maintaining 3D mAP in the range of 21.74% to 43.38% across nine heterogeneous sensor configurations. These results, validated by our first-place finish in the CVPR 2025 DriveX challenge, establish HeCoFuse as the current state-of-the-art on TUM-Traf V2X dataset while demonstrating robust performance across diverse sensor deployments. 

**Abstract (ZH)**: 面向异构传感器配置的大规模V2X协同感知系统：HeCoFuse统一框架 

---
# When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework 

**Title (ZH)**: 当人员重识别遇到事件相机：一个基准数据集及其属性导向的重识别框架 

**Authors**: Xiao Wang, Qian Zhu, Shujuan Wu, Bo Jiang, Shiliang Zhang, Yaowei Wang, Yonghong Tian, Bin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.13659)  

**Abstract**: Recent researchers have proposed using event cameras for person re-identification (ReID) due to their promising performance and better balance in terms of privacy protection, event camera-based person ReID has attracted significant attention. Currently, mainstream event-based person ReID algorithms primarily focus on fusing visible light and event stream, as well as preserving privacy. Although significant progress has been made, these methods are typically trained and evaluated on small-scale or simulated event camera datasets, making it difficult to assess their real identification performance and generalization ability. To address the issue of data scarcity, this paper introduces a large-scale RGB-event based person ReID dataset, called EvReID. The dataset contains 118,988 image pairs and covers 1200 pedestrian identities, with data collected across multiple seasons, scenes, and lighting conditions. We also evaluate 15 state-of-the-art person ReID algorithms, laying a solid foundation for future research in terms of both data and benchmarking. Based on our newly constructed dataset, this paper further proposes a pedestrian attribute-guided contrastive learning framework to enhance feature learning for person re-identification, termed TriPro-ReID. This framework not only effectively explores the visual features from both RGB frames and event streams, but also fully utilizes pedestrian attributes as mid-level semantic features. Extensive experiments on the EvReID dataset and MARS datasets fully validated the effectiveness of our proposed RGB-Event person ReID framework. The benchmark dataset and source code will be released on this https URL 

**Abstract (ZH)**: 近期的研究提出了使用事件摄像头进行行人重新识别（ReID）的方法，由于其表现出色且在隐私保护方面具有更好的平衡，基于事件摄像头的行人ReID吸引了广泛关注。目前，主流的基于事件的行人ReID算法主要集中在融合可见光和事件流，以及保护隐私方面。尽管已经取得了显著进展，但这些方法通常在小型或模拟事件摄像头数据集上进行训练和评估，使得评估其实用识别性能和泛化能力变得困难。为了解决数据稀缺的问题，本文介绍了一种大规模RGB-事件基于行人ReID的数据集，称为EvReID。该数据集包含118,988对图像配对，并涵盖了1200个行人类别身份，数据采集跨越了多个季节、场景和光照条件。我们还评估了15种最先进的行人ReID算法，为未来研究提供了坚实的数据和基准评估基础。基于我们新构建的数据集，本文进一步提出了一种行人属性引导的对比学习框架，以增强行人ReID的特征学习，称为TriPro-ReID。该框架不仅有效探索了来自RGB帧和事件流的视觉特征，还充分利用了行人类别的中间语义特征。在EvReID数据集和MARS数据集上的大量实验充分验证了我们提出的RGB-事件行人ReID框架的有效性。基准数据集和源代码将发布在该网址。 

---
# Improved particle swarm optimization algorithm: multi-target trajectory optimization for swarm drones 

**Title (ZH)**: 改进的粒子群优化算法：群无人机多目标轨迹优化 

**Authors**: Minze Li, Wei Zhao, Ran Chen, Mingqiang Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.13647)  

**Abstract**: Real-time trajectory planning for unmanned aerial vehicles (UAVs) in dynamic environments remains a key challenge due to high computational demands and the need for fast, adaptive responses. Traditional Particle Swarm Optimization (PSO) methods, while effective for offline planning, often struggle with premature convergence and latency in real-time scenarios. To overcome these limitations, we propose PE-PSO, an enhanced PSO-based online trajectory planner. The method introduces a persistent exploration mechanism to preserve swarm diversity and an entropy-based parameter adjustment strategy to dynamically adapt optimization behavior. UAV trajectories are modeled using B-spline curves, which ensure path smoothness while reducing optimization complexity. To extend this capability to UAV swarms, we develop a multi-agent framework that combines genetic algorithm (GA)-based task allocation with distributed PE-PSO, supporting scalable and coordinated trajectory generation. The distributed architecture allows for parallel computation and decentralized control, enabling effective cooperation among agents while maintaining real-time performance. Comprehensive simulations demonstrate that the proposed framework outperforms conventional PSO and other swarm-based planners across several metrics, including trajectory quality, energy efficiency, obstacle avoidance, and computation time. These results confirm the effectiveness and applicability of PE-PSO in real-time multi-UAV operations under complex environmental conditions. 

**Abstract (ZH)**: 实时无人机在动态环境下的轨迹规划仍然是一个关键挑战，由于高计算要求和需要快速适应的响应。传统的粒子群优化（PSO）方法虽然适用于离线规划，但在实时场景中常会出现过早收敛和延迟问题。为克服这些局限，我们提出了一种增强的基于PSO的在线轨迹规划方法PE-PSO。该方法引入了一种持久的探索机制以保持种群多样性，并采用基于熵的参数调整策略以动态适应优化行为。无人机轨迹使用B-样条曲线建模，确保路径平滑并降低优化复杂度。为了将此能力扩展到无人机群体，我们开发了一个基于多agent框架，结合了基于遗传算法（GA）的任务分配和分布式PE-PSO，支持可扩展和协调的轨迹生成。分布式架构允许并行计算和去中心化控制，使agent之间能够有效协作，同时保持实时性能。全面的仿真结果显示，所提出的框架在轨迹质量、能量效率、障碍物规避和计算时间等多个指标上优于传统PSO和其他基于群体的规划方法。这些结果证实了PE-PSO在复杂环境条件下的实时多无人机操作中的有效性和适用性。 

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
# BreastSegNet: Multi-label Segmentation of Breast MRI 

**Title (ZH)**: 乳腺分割网络：乳腺MRI的多标签分割 

**Authors**: Qihang Li, Jichen Yang, Yaqian Chen, Yuwen Chen, Hanxue Gu, Lars J. Grimm, Maciej A. Mazurowski  

**Link**: [PDF](https://arxiv.org/pdf/2507.13604)  

**Abstract**: Breast MRI provides high-resolution imaging critical for breast cancer screening and preoperative staging. However, existing segmentation methods for breast MRI remain limited in scope, often focusing on only a few anatomical structures, such as fibroglandular tissue or tumors, and do not cover the full range of tissues seen in scans. This narrows their utility for quantitative analysis. In this study, we present BreastSegNet, a multi-label segmentation algorithm for breast MRI that covers nine anatomical labels: fibroglandular tissue (FGT), vessel, muscle, bone, lesion, lymph node, heart, liver, and implant. We manually annotated a large set of 1123 MRI slices capturing these structures with detailed review and correction from an expert radiologist. Additionally, we benchmark nine segmentation models, including U-Net, SwinUNet, UNet++, SAM, MedSAM, and nnU-Net with multiple ResNet-based encoders. Among them, nnU-Net ResEncM achieves the highest average Dice scores of 0.694 across all labels. It performs especially well on heart, liver, muscle, FGT, and bone, with Dice scores exceeding 0.73, and approaching 0.90 for heart and liver. All model code and weights are publicly available, and we plan to release the data at a later date. 

**Abstract (ZH)**: 乳腺MRI提供高分辨率成像，对于乳腺癌筛查和术前分期至关重要。然而，现有的乳腺MRI分割方法仍然范围有限，通常仅专注于几种子结构，如纤维腺组织或肿瘤，未能涵盖扫描中看到的所有组织类型。这限制了其在定量分析中的应用范围。本研究提出了一种多标签分割算法BreastSegNet，该算法涵盖了九个解剖标签：纤维腺组织（FGT）、血管、肌肉、骨质、病灶、淋巴结、心脏、肝脏和植入物。我们手工标注了一大批1123张MRI切片，并由专家放射学家进行了详细审核和修正。此外，我们还对包括U-Net、SwinUNet、UNet++、SAM、MedSAM和nnU-Net（搭配多种ResNet编码器）在内的九种分割模型进行了基准测试。其中，nnU-Net ResEncM在所有标签上的平均Dice分数最高，达到0.694。它在心脏、肝脏、肌肉、FGT和骨质上的表现尤为出色，Dice分数超过0.73，心脏和肝脏的Dice分数接近0.90。所有模型代码和权重均已公开，我们计划日后发布数据。 

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
# Change of Thought: Adaptive Test-Time Computation 

**Title (ZH)**: 思想变换：自适应测试时计算 

**Authors**: Mrinal Mathur, Mike Doan, Barak Pearlmutter, Sergey Plis  

**Link**: [PDF](https://arxiv.org/pdf/2507.13569)  

**Abstract**: Transformers evaluated in a single, fixed-depth pass are provably limited in expressive power to the constant-depth circuit class TC0. Running a Transformer autoregressively removes that ceiling -- first in next-token prediction and, more recently, in chain-of-thought reasoning. Both regimes rely on feedback loops that decode internal states into tokens only to re-encode them in subsequent steps. While this "thinking aloud" mirrors human reasoning, biological brains iterate without externalising intermediate states as language. To boost the expressive power of encoder Transformers without resorting to token-level autoregression, we introduce the SELF-Transformer: an encoder layer that iteratively refines its own attention weights to a fixed point. Instead of producing -- in one pass -- the alignment matrix that remixes the input sequence, the SELF-Transformer iteratively updates that matrix internally, scaling test-time computation with input difficulty. This adaptivity yields up to 20\% accuracy gains on encoder-style benchmarks without increasing parameter count, demonstrating that input-adaptive alignment at test time offers substantial benefits for only a modest extra compute budget. Self-Transformers thus recover much of the expressive power of iterative reasoning while preserving the simplicity of pure encoder architectures. 

**Abstract (ZH)**: SELF-Transformer: Iteratively Refining Attention Weights for Enhanced Expressive Power Without Token-Level Autoregression 

---
# Time Series Forecastability Measures 

**Title (ZH)**: 时间序列可预测性度量 

**Authors**: Rui Wang, Steven Klee, Alexis Roos  

**Link**: [PDF](https://arxiv.org/pdf/2507.13556)  

**Abstract**: This paper proposes using two metrics to quantify the forecastability of time series prior to model development: the spectral predictability score and the largest Lyapunov exponent. Unlike traditional model evaluation metrics, these measures assess the inherent forecastability characteristics of the data before any forecast attempts. The spectral predictability score evaluates the strength and regularity of frequency components in the time series, whereas the Lyapunov exponents quantify the chaos and stability of the system generating the data. We evaluated the effectiveness of these metrics on both synthetic and real-world time series from the M5 forecast competition dataset. Our results demonstrate that these two metrics can correctly reflect the inherent forecastability of a time series and have a strong correlation with the actual forecast performance of various models. By understanding the inherent forecastability of time series before model training, practitioners can focus their planning efforts on products and supply chain levels that are more forecastable, while setting appropriate expectations or seeking alternative strategies for products with limited forecastability. 

**Abstract (ZH)**: 本文提出使用两种指标在模型开发前量化时间序列的可预测性：谱可预测性分数和最大路易普朗伏指数。这些措施评估数据本身的可预测性特征，而不像传统的模型评估指标那样在任何预测尝试之后进行评估。谱可预测性分数评估时间序列中频率分量的强度和规律性，而路易普朗伏指数量化产生数据的系统的混沌性和稳定性。我们通过M5预测竞赛数据集中的合成时间和实际时间序列评估了这些指标的有效性。结果显示，这两种指标能够正确反映时间序列的内在可预测性，并与各种模型的实际预测性能有很强的相关性。通过在模型训练前理解时间序列的内在可预测性，实践者可以将规划努力集中在更具可预测性的产品和供应链层面，同时为可预测性有限的产品设定适当的预期或寻求替代策略。 

---
# Reading Between the Lines: Combining Pause Dynamics and Semantic Coherence for Automated Assessment of Thought Disorder 

**Title (ZH)**: 读其之间：结合停顿动态与语义连贯性进行思维紊乱的自动化评估 

**Authors**: Feng Chen, Weizhe Xu, Changye Li, Serguei Pakhomov, Alex Cohen, Simran Bhola, Sandy Yin, Sunny X Tang, Michael Mackinley, Lena Palaniyappan, Dror Ben-Zeev, Trevor Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13551)  

**Abstract**: Formal thought disorder (FTD), a hallmark of schizophrenia spectrum disorders, manifests as incoherent speech and poses challenges for clinical assessment. Traditional clinical rating scales, though validated, are resource-intensive and lack scalability. Automated speech analysis with automatic speech recognition (ASR) allows for objective quantification of linguistic and temporal features of speech, offering scalable alternatives. The use of utterance timestamps in ASR captures pause dynamics, which are thought to reflect the cognitive processes underlying speech production. However, the utility of integrating these ASR-derived features for assessing FTD severity requires further evaluation. This study integrates pause features with semantic coherence metrics across three datasets: naturalistic self-recorded diaries (AVH, n = 140), structured picture descriptions (TOPSY, n = 72), and dream narratives (PsyCL, n = 43). We evaluated pause related features alongside established coherence measures, using support vector regression (SVR) to predict clinical FTD scores. Key findings demonstrate that pause features alone robustly predict the severity of FTD. Integrating pause features with semantic coherence metrics enhanced predictive performance compared to semantic-only models, with integration of independent models achieving correlations up to \r{ho} = 0.649 and AUC = 83.71% for severe cases detection (TOPSY, with best \r{ho} = 0.584 and AUC = 79.23% for semantic-only models). The performance gains from semantic and pause features integration held consistently across all contexts, though the nature of pause patterns was dataset-dependent. These findings suggest that frameworks combining temporal and semantic analyses provide a roadmap for refining the assessment of disorganized speech and advance automated speech analysis in psychosis. 

**Abstract (ZH)**: 基于暂停特征和语义连贯性指标的自动语音分析在评估形式思维障碍中的应用 

---
# Loss-Complexity Landscape and Model Structure Functions 

**Title (ZH)**: 损失-复杂性景观和模型结构函数 

**Authors**: Alexander Kolpakov  

**Link**: [PDF](https://arxiv.org/pdf/2507.13543)  

**Abstract**: We develop a framework for dualizing the Kolmogorov structure function $h_x(\alpha)$, which then allows using computable complexity proxies. We establish a mathematical analogy between information-theoretic constructs and statistical mechanics, introducing a suitable partition function and free energy functional. We explicitly prove the Legendre-Fenchel duality between the structure function and free energy, showing detailed balance of the Metropolis kernel, and interpret acceptance probabilities as information-theoretic scattering amplitudes. A susceptibility-like variance of model complexity is shown to peak precisely at loss-complexity trade-offs interpreted as phase transitions. Practical experiments with linear and tree-based regression models verify these theoretical predictions, explicitly demonstrating the interplay between the model complexity, generalization, and overfitting threshold. 

**Abstract (ZH)**: 我们开发了一种双对柯尔莫戈罗夫结构函数 \(h_x(\alpha)\) 的框架，从而允许使用可计算的复杂性代理。我们建立了信息论结构与统计力学之间的数学类比，引入了合适的分区函数和自由能泛函。我们明确证明了结构函数与自由能之间的勒让德-芬彻尔对偶性，展示了蒙特卡洛内核的详细平衡，并将接受概率解释为信息论散射振幅。模型复杂性的类似 susceptiiblity 的方差在解释为相变的损失-复杂性权衡处精确峰值。实际的线性和树状回归模型实验验证了这些理论预测，明确展示了模型复杂性、泛化能力和过拟合阈值之间的相互作用。 

---
# Acoustic Index: A Novel AI-Driven Parameter for Cardiac Disease Risk Stratification Using Echocardiography 

**Title (ZH)**: 声学指标：一种基于人工智能驱动的超声心动图心肌病风险分层参数 

**Authors**: Beka Begiashvili, Carlos J. Fernandez-Candel, Matías Pérez Paredes  

**Link**: [PDF](https://arxiv.org/pdf/2507.13542)  

**Abstract**: Traditional echocardiographic parameters such as ejection fraction (EF) and global longitudinal strain (GLS) have limitations in the early detection of cardiac dysfunction. EF often remains normal despite underlying pathology, and GLS is influenced by load conditions and vendor variability. There is a growing need for reproducible, interpretable, and operator-independent parameters that capture subtle and global cardiac functional alterations.
We introduce the Acoustic Index, a novel AI-derived echocardiographic parameter designed to quantify cardiac dysfunction from standard ultrasound views. The model combines Extended Dynamic Mode Decomposition (EDMD) based on Koopman operator theory with a hybrid neural network that incorporates clinical metadata. Spatiotemporal dynamics are extracted from echocardiographic sequences to identify coherent motion patterns. These are weighted via attention mechanisms and fused with clinical data using manifold learning, resulting in a continuous score from 0 (low risk) to 1 (high risk).
In a prospective cohort of 736 patients, encompassing various cardiac pathologies and normal controls, the Acoustic Index achieved an area under the curve (AUC) of 0.89 in an independent test set. Cross-validation across five folds confirmed the robustness of the model, showing that both sensitivity and specificity exceeded 0.8 when evaluated on independent data. Threshold-based analysis demonstrated stable trade-offs between sensitivity and specificity, with optimal discrimination near this threshold.
The Acoustic Index represents a physics-informed, interpretable AI biomarker for cardiac function. It shows promise as a scalable, vendor-independent tool for early detection, triage, and longitudinal monitoring. Future directions include external validation, longitudinal studies, and adaptation to disease-specific classifiers. 

**Abstract (ZH)**: 基于声学索引的新型AI心功能参数在心脏功能早期检测中的应用：一种可解释的物理导向AI生物标志物 

---
# Humans learn to prefer trustworthy AI over human partners 

**Title (ZH)**: 人类更偏好可信赖的AI伙伴 

**Authors**: Yaomin Jiang, Levin Brinkmann, Anne-Marie Nussberger, Ivan Soraperra, Jean-François Bonnefon, Iyad Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2507.13524)  

**Abstract**: Partner selection is crucial for cooperation and hinges on communication. As artificial agents, especially those powered by large language models (LLMs), become more autonomous, intelligent, and persuasive, they compete with humans for partnerships. Yet little is known about how humans select between human and AI partners and adapt under AI-induced competition pressure. We constructed a communication-based partner selection game and examined the dynamics in hybrid mini-societies of humans and bots powered by a state-of-the-art LLM. Through three experiments (N = 975), we found that bots, though more prosocial than humans and linguistically distinguishable, were not selected preferentially when their identity was hidden. Instead, humans misattributed bots' behaviour to humans and vice versa. Disclosing bots' identity induced a dual effect: it reduced bots' initial chances of being selected but allowed them to gradually outcompete humans by facilitating human learning about the behaviour of each partner type. These findings show how AI can reshape social interaction in mixed societies and inform the design of more effective and cooperative hybrid systems. 

**Abstract (ZH)**: 人工伙伴选择对于合作至关重要且依赖于沟通。随着尤其是由大规模语言模型（LLMs）驱动的智能代理变得更为自主、智能和有说服力，它们在合作伙伴中与人类竞争。然而，关于人类在人工智能引起的竞争压力下选择人类和AI合作伙伴及其适应机制，我们知之甚少。我们构建了一个基于沟通的人工伙伴选择博弈，并考察了人类和由先进LLM驱动的智能代理组成的混合小社会中的动态变化。通过三项实验（N=975），我们发现，尽管智能代理比人类更有利他性且在语言上有明显区分，但在其身份未被揭示时，并未被优先选择。相反，人类错误地将智能代理的行为归因给人类，反之亦然。揭示智能代理的身份产生了双重效果：它减少了智能代理最初被选择的机会，但让他们能够通过促进人类对每种合作伙伴类型行为的理解而逐步胜过人类。这些发现展示了AI如何重新塑造混合社会中的社会互动，并为设计更有效的混合系统提供指导。 

---
# PHASE: Passive Human Activity Simulation Evaluation 

**Title (ZH)**: PHASE: 被动人类活动模拟评估 

**Authors**: Steven Lamp, Jason D. Hiser, Anh Nguyen-Tuong, Jack W. Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2507.13505)  

**Abstract**: Cybersecurity simulation environments, such as cyber ranges, honeypots, and sandboxes, require realistic human behavior to be effective, yet no quantitative method exists to assess the behavioral fidelity of synthetic user personas. This paper presents PHASE (Passive Human Activity Simulation Evaluation), a machine learning framework that analyzes Zeek connection logs and distinguishes human from non-human activity with over 90\% accuracy. PHASE operates entirely passively, relying on standard network monitoring without any user-side instrumentation or visible signs of surveillance. All network activity used for machine learning is collected via a Zeek network appliance to avoid introducing unnecessary network traffic or artifacts that could disrupt the fidelity of the simulation environment. The paper also proposes a novel labeling approach that utilizes local DNS records to classify network traffic, thereby enabling machine learning analysis. Furthermore, we apply SHAP (SHapley Additive exPlanations) analysis to uncover temporal and behavioral signatures indicative of genuine human users. In a case study, we evaluate a synthetic user persona and identify distinct non-human patterns that undermine behavioral realism. Based on these insights, we develop a revised behavioral configuration that significantly improves the human-likeness of synthetic activity yielding a more realistic and effective synthetic user persona. 

**Abstract (ZH)**: 基于机器学习的被动人类活动仿真评估（PHASE）：一种评估合成用户行为真实性的方法 

---
# AI-Assisted Fixes to Code Review Comments at Scale 

**Title (ZH)**: 大规模辅助AI对代码评审评论的修复 

**Authors**: Chandra Maddila, Negar Ghorbani, James Saindon, Parth Thakkar, Vijayaraghavan Murali, Rui Abreu, Jingyue Shen, Brian Zhou, Nachiappan Nagappan, Peter C. Rigby  

**Link**: [PDF](https://arxiv.org/pdf/2507.13499)  

**Abstract**: Aim. There are 10s of thousands of code review comments each week at Meta. We developed Metamate for Code Review (MetaMateCR) that provides AI-assisted fixes for reviewer comments in production at scale.
Method. We developed an internal benchmark of 64k <review comment, patch> data points to fine-tune Llama models. Once our models achieve reasonable offline results, we roll them into production. To ensure that our AI-assisted fixes do not negatively impact the time it takes to do code reviews, we conduct randomized controlled safety trials as well as full production experiments.
Offline Results. As a baseline, we compare GPT-4o to our small and large Llama models. In offline results, our LargeLSFT model creates an exact match patch 68% of the time outperforming GPT-4o by 9 percentage points (pp). The internal models also use more modern Hack functions when compared to the PHP functions suggested by GPT-4o.
Safety Trial. When we roll MetaMateCR into production in a safety trial that compares no AI patches with AI patch suggestions, we see a large regression with reviewers taking over 5% longer to conduct reviews. After investigation, we modify the UX to only show authors the AI patches, and see no regressions in the time for reviews.
Production. When we roll LargeLSFT into production, we see an ActionableToApplied rate of 19.7%, which is a 9.2pp improvement over GPT-4o. Our results illustrate the importance of safety trials in ensuring that AI does not inadvertently slow down engineers, and a successful review comment to AI patch product running at scale. 

**Abstract (ZH)**: 目标. 每周在Meta上有成千上万条代码评审评论。我们开发了MetaMate for Code Review（MetaMateCR），用于在生产环境中提供AI辅助的修复建议。

方法. 我们开发了一个包含64,000个<评审评论,补丁>数据点的内部基准，用于微调Llama模型。当我们的模型在离线测试中达到合理的结果后，我们将模型部署到生产环境中。为了确保AI辅助的修复不会负面影响代码评审时间，我们进行了随机对照安全性试验以及全面的生产实验。

离线结果. 作为 baseline，我们将GPT-4o与我们的小型和大型Llama模型进行比较。在离线结果中，我们的LargeLSFT模型有68%的时间生成完全匹配的补丁，比GPT-4o高出9个百分点。内部模型还使用了更现代的Hack函数，相比于GPT-4o建议的PHP函数。

安全性试验. 在进行MetaMateCR的安全性试验时，将没有AI补丁与AI补丁建议进行比较，我们发现评审者花费的时间比之前长了5%以上。经过调查，我们修改了用户界面，仅向作者显示AI补丁，从而没有进一步的评审时间退步。

生产. 当我们将LargeLSFT部署到生产环境中时，我们看到ActionableToApplied的转换率为19.7%，比GPT-4o提高了9.2个百分点。我们的结果强调了在确保AI不会无意中减慢工程师的工作速度方面安全性试验的重要性，并且展示了成功运行的大规模评审评论到AI补丁产品的实例。 

---
# Neural Architecture Search with Mixed Bio-inspired Learning Rules 

**Title (ZH)**: 生物启发混合学习规则下的神经网络架构搜索 

**Authors**: Imane Hamzaoui, Riyadh Baghdadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.13485)  

**Abstract**: Bio-inspired neural networks are attractive for their adversarial robustness, energy frugality, and closer alignment with cortical physiology, yet they often lag behind back-propagation (BP) based models in accuracy and ability to scale. We show that allowing the use of different bio-inspired learning rules in different layers, discovered automatically by a tailored neural-architecture-search (NAS) procedure, bridges this gap. Starting from standard NAS baselines, we enlarge the search space to include bio-inspired learning rules and use NAS to find the best architecture and learning rule to use in each layer. We show that neural networks that use different bio-inspired learning rules for different layers have better accuracy than those that use a single rule across all the layers. The resulting NN that uses a mix of bio-inspired learning rules sets new records for bio-inspired models: 95.16% on CIFAR-10, 76.48% on CIFAR-100, 43.42% on ImageNet16-120, and 60.51% top-1 on ImageNet. In some regimes, they even surpass comparable BP-based networks while retaining their robustness advantages. Our results suggest that layer-wise diversity in learning rules allows better scalability and accuracy, and motivates further research on mixing multiple bio-inspired learning rules in the same network. 

**Abstract (ZH)**: 生物启发神经网络在不同层中采用自动发现的生物启发学习规则以提高准确性和可扩展性 

---
# ERR@HRI 2.0 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Conversations 

**Title (ZH)**: ERR@HRI 2.0 挑战赛：人类-机器人对话中多模态错误和故障检测 

**Authors**: Shiye Cao, Maia Stiber, Amama Mahmood, Maria Teresa Parreira, Wendy Ju, Micol Spitale, Hatice Gunes, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13468)  

**Abstract**: The integration of large language models (LLMs) into conversational robots has made human-robot conversations more dynamic. Yet, LLM-powered conversational robots remain prone to errors, e.g., misunderstanding user intent, prematurely interrupting users, or failing to respond altogether. Detecting and addressing these failures is critical for preventing conversational breakdowns, avoiding task disruptions, and sustaining user trust. To tackle this problem, the ERR@HRI 2.0 Challenge provides a multimodal dataset of LLM-powered conversational robot failures during human-robot conversations and encourages researchers to benchmark machine learning models designed to detect robot failures. The dataset includes 16 hours of dyadic human-robot interactions, incorporating facial, speech, and head movement features. Each interaction is annotated with the presence or absence of robot errors from the system perspective, and perceived user intention to correct for a mismatch between robot behavior and user expectation. Participants are invited to form teams and develop machine learning models that detect these failures using multimodal data. Submissions will be evaluated using various performance metrics, including detection accuracy and false positive rate. This challenge represents another key step toward improving failure detection in human-robot interaction through social signal analysis. 

**Abstract (ZH)**: 大型语言模型（LLMs）在对话机器人中的集成使得人机对话更加动态，但由LLM驱动的对话机器人仍然容易出现错误，例如误解用户意图、过早中断用户或完全无法响应。检测和解决这些故障对于防止对话中断、避免任务中断并维持用户信任至关重要。为应对这一挑战，ERR@HRI 2.0竞赛提供了一种多模态数据集，其中包括LLM驱动的对话机器人在人机对话过程中出现的故障，并鼓励研究人员使用机器学习模型来检测机器人故障。该数据集包括16小时的双向人机互动，整合了面部、语音和头部动作特征。每项互动都从系统视角标注了机器人故障的有无，并考虑到用户意图的感知，以纠正机器人行为与用户期望之间的不一致。参赛者被邀请组队并开发能够使用多模态数据检测这些故障的机器学习模型。提交将根据检测准确率和假阳性率等性能指标进行评估。该挑战代表了通过社会信号分析提高人机交互中故障检测的又一个重要步骤。 

---
# Graph Neural Network Surrogates for Contacting Deformable Bodies with Necessary and Sufficient Contact Detection 

**Title (ZH)**: 基于图神经网络的接触检测代理模型：必要充分接触检测 

**Authors**: Vijay K. Dubey, Collin E. Haese, Osman Gültekin, David Dalton, Manuel K. Rausch, Jan N. Fuhg  

**Link**: [PDF](https://arxiv.org/pdf/2507.13459)  

**Abstract**: Surrogate models for the rapid inference of nonlinear boundary value problems in mechanics are helpful in a broad range of engineering applications. However, effective surrogate modeling of applications involving the contact of deformable bodies, especially in the context of varying geometries, is still an open issue. In particular, existing methods are confined to rigid body contact or, at best, contact between rigid and soft objects with well-defined contact planes. Furthermore, they employ contact or collision detection filters that serve as a rapid test but use only the necessary and not sufficient conditions for detection. In this work, we present a graph neural network architecture that utilizes continuous collision detection and, for the first time, incorporates sufficient conditions designed for contact between soft deformable bodies. We test its performance on two benchmarks, including a problem in soft tissue mechanics of predicting the closed state of a bioprosthetic aortic valve. We find a regularizing effect on adding additional contact terms to the loss function, leading to better generalization of the network. These benefits hold for simple contact at similar planes and element normal angles, and complex contact at differing planes and element normal angles. We also demonstrate that the framework can handle varying reference geometries. However, such benefits come with high computational costs during training, resulting in a trade-off that may not always be favorable. We quantify the training cost and the resulting inference speedups on various hardware architectures. Importantly, our graph neural network implementation results in up to a thousand-fold speedup for our benchmark problems at inference. 

**Abstract (ZH)**: 用于机械非线性边界值问题快速推理的代理模型在工程应用中很有帮助。然而，在涉及可变形体接触的应用中，特别是几何形状变化的情况下，有效的代理建模仍然是一个开放问题。特别是在接触软体时，现有方法主要局限于刚体接触，或者最好情况下接触刚体和软体对象，并带有明确的接触平面。此外，它们使用仅作为快速测试的接触或碰撞检测过滤器，但只使用检测的必要条件而不是充分条件。在本工作中，我们提出了一种图神经网络架构，利用连续碰撞检测，并首次结合了用于软可变形体接触的充分条件。我们在两个基准测试中测试了其性能，包括软组织力学中的一个预测生物人工主动脉瓣关闭状态的问题。我们发现，在损失函数中添加额外的接触项具有正则化效果，从而提高了网络的泛化能力。这些好处适用于类似平面和元件法线角度的简单接触，以及不同平面和元件法线角度的复杂接触。我们还展示了该框架可以处理变化的参考几何形状。然而，这些好处会导致训练时的高计算成本，从而产生一种可能并不总是有利的权衡。我们量化了各种硬件架构下的训练成本和推理速度提升。重要的是，我们的图神经网络实现能够在推理中使基准问题的处理速度提高高达一千倍。 

---
# "PhyWorldBench": A Comprehensive Evaluation of Physical Realism in Text-to-Video Models 

**Title (ZH)**: PhyWorldBench：文本到视频模型中物理真实感的综合评估 

**Authors**: Jing Gu, Xian Liu, Yu Zeng, Ashwin Nagarajan, Fangrui Zhu, Daniel Hong, Yue Fan, Qianqi Yan, Kaiwen Zhou, Ming-Yu Liu, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13428)  

**Abstract**: Video generation models have achieved remarkable progress in creating high-quality, photorealistic content. However, their ability to accurately simulate physical phenomena remains a critical and unresolved challenge. This paper presents PhyWorldBench, a comprehensive benchmark designed to evaluate video generation models based on their adherence to the laws of physics. The benchmark covers multiple levels of physical phenomena, ranging from fundamental principles like object motion and energy conservation to more complex scenarios involving rigid body interactions and human or animal motion. Additionally, we introduce a novel ""Anti-Physics"" category, where prompts intentionally violate real-world physics, enabling the assessment of whether models can follow such instructions while maintaining logical consistency. Besides large-scale human evaluation, we also design a simple yet effective method that could utilize current MLLM to evaluate the physics realism in a zero-shot fashion. We evaluate 12 state-of-the-art text-to-video generation models, including five open-source and five proprietary models, with a detailed comparison and analysis. we identify pivotal challenges models face in adhering to real-world physics. Through systematic testing of their outputs across 1,050 curated prompts-spanning fundamental, composite, and anti-physics scenarios-we identify pivotal challenges these models face in adhering to real-world physics. We then rigorously examine their performance on diverse physical phenomena with varying prompt types, deriving targeted recommendations for crafting prompts that enhance fidelity to physical principles. 

**Abstract (ZH)**: 基于物理模拟的视频生成模型综合基准：PhyWorldBench 

---
# CaSTFormer: Causal Spatio-Temporal Transformer for Driving Intention Prediction 

**Title (ZH)**: 基于因果时空变换器的驾驶意图预测 

**Authors**: Sirui Wang, Zhou Guan, Bingxi Zhao, Tongjia Gu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13425)  

**Abstract**: Accurate prediction of driving intention is key to enhancing the safety and interactive efficiency of human-machine co-driving systems. It serves as a cornerstone for achieving high-level autonomous driving. However, current approaches remain inadequate for accurately modeling the complex spatio-temporal interdependencies and the unpredictable variability of human driving behavior. To address these challenges, we propose CaSTFormer, a Causal Spatio-Temporal Transformer to explicitly model causal interactions between driver behavior and environmental context for robust intention prediction. Specifically, CaSTFormer introduces a novel Reciprocal Shift Fusion (RSF) mechanism for precise temporal alignment of internal and external feature streams, a Causal Pattern Extraction (CPE) module that systematically eliminates spurious correlations to reveal authentic causal dependencies, and an innovative Feature Synthesis Network (FSN) that adaptively synthesizes these purified representations into coherent spatio-temporal inferences. We evaluate the proposed CaSTFormer on the public Brain4Cars dataset, and it achieves state-of-the-art performance. It effectively captures complex causal spatio-temporal dependencies and enhances both the accuracy and transparency of driving intention prediction. 

**Abstract (ZH)**: 准确预测驾驶意图对于提升人机协同驾驶系统的安全性和交互效率至关重要，它是实现高级自动驾驶的基础。然而，当前的方法在准确建模复杂时空依赖性和人类驾驶行为的不可预测变异性方面仍显不足。为应对这些挑战，我们提出了一种因果时空变换器CaSTFormer，以明确建模驾驶员行为与环境上下文之间的因果交互，从而实现稳健的意图预测。具体而言，CaSTFormer引入了一种新颖的互易移位融合（RSF）机制，实现了内部和外部特征流的精确时间对齐；一种因果模式提取（CPE）模块，系统地消除虚假相关，揭示真实的因果依赖关系；以及一种创新的功能合成网络（FSN），能够自适应地将这些净化的表示综合成连贯的时空推理。我们在公开的Brain4Cars数据集上评估了提出的CaSTFormer，并实现了最先进的性能，有效地捕捉了复杂的因果时空依赖关系，提升了驾驶意图预测的准确性和透明度。 

---
# Air Traffic Controller Task Demand via Graph Neural Networks: An Interpretable Approach to Airspace Complexity 

**Title (ZH)**: 基于图神经网络的空中交通管制任务需求：一种可解释的方法来评估空域复杂性 

**Authors**: Edward Henderson, Dewi Gould, Richard Everson, George De Ath, Nick Pepper  

**Link**: [PDF](https://arxiv.org/pdf/2507.13423)  

**Abstract**: Real-time assessment of near-term Air Traffic Controller (ATCO) task demand is a critical challenge in an increasingly crowded airspace, as existing complexity metrics often fail to capture nuanced operational drivers beyond simple aircraft counts. This work introduces an interpretable Graph Neural Network (GNN) framework to address this gap. Our attention-based model predicts the number of upcoming clearances, the instructions issued to aircraft by ATCOs, from interactions within static traffic scenarios. Crucially, we derive an interpretable, per-aircraft task demand score by systematically ablating aircraft and measuring the impact on the model's predictions. Our framework significantly outperforms an ATCO-inspired heuristic and is a more reliable estimator of scenario complexity than established baselines. The resulting tool can attribute task demand to specific aircraft, offering a new way to analyse and understand the drivers of complexity for applications in controller training and airspace redesign. 

**Abstract (ZH)**: 实时评估近期内航空交通管制员任务需求的可解释图神经网络框架：基于关注机制的模型通过静态交通情景内的交互预测即将发布的飞行许可数量，并通过系统消除飞机和测量对模型预测的影响来推导出可解释的单机任务需求评分，显著优于基于航空交通管制员启发的方法，是场景复杂性的一种更可靠的估计器。此框架产生的工具可将任务需求归因于特定飞机，为管制员培训和 airspace 重设计的应用提供新的分析和理解复杂性的方式。 

---
# AI-ming backwards: Vanishing archaeological landscapes in Mesopotamia and automatic detection of sites on CORONA imagery 

**Title (ZH)**: AI千年回望：两河流域消失的考古景观与CORONA影像中的遗址自动检测 

**Authors**: Alessandro Pistola, Valentina Orru', Nicolo' Marchetti, Marco Roccetti  

**Link**: [PDF](https://arxiv.org/pdf/2507.13420)  

**Abstract**: By upgrading an existing deep learning model with the knowledge provided by one of the oldest sets of grayscale satellite imagery, known as CORONA, we improved the AI model attitude towards the automatic identification of archaeological sites in an environment which has been completely transformed in the last five decades, including the complete destruction of many of those same sites. The initial Bing based convolutional network model was retrained using CORONA satellite imagery for the district of Abu Ghraib, west of Baghdad, central Mesopotamian floodplain. The results were twofold and surprising. First, the detection precision obtained on the area of interest increased sensibly: in particular, the Intersection over Union (IoU) values, at the image segmentation level, surpassed 85 percent, while the general accuracy in detecting archeological sites reached 90 percent. Second, our retrained model allowed the identification of four new sites of archaeological interest (confirmed through field verification), previously not identified by archaeologists with traditional techniques. This has confirmed the efficacy of using AI techniques and the CORONA imagery from the 1960 to discover archaeological sites currently no longer visible, a concrete breakthrough with significant consequences for the study of landscapes with vanishing archaeological evidence induced by anthropization 

**Abstract (ZH)**: 通过利用CORONA灰度卫星影像提供的知识升级现有的深度学习模型，我们改进了AI模型对近五十年彻底改变的环境中考古遗址自动识别的态度，包括许多相同遗址被完全破坏的情况。基于Bing的初始卷积神经网络模型在巴格达西部阿布格拉布地区的美索不达米亚冲积平原上重新训练，使用CORONA卫星影像。结果令人惊讶且具有两方面意义。首先，在感兴趣的区域中，检测精度显著提高：特别是在图像分割层面，交并比(IoU)值超过85%，整体上用于检测考古遗址的准确率达到90%。其次，我们的重新训练模型允许识别出四个新的考古遗址（通过实地验证确认），而传统技术未识别这些遗址。这证实了使用AI技术和1960年代的CORONA影像发现目前不再可见的考古遗址的有效性，这是在受到人为活动影响导致考古证据消失的景观研究中的一项具体突破，具有重要的意义。 

---
# Soft-ECM: An extension of Evidential C-Means for complex data 

**Title (ZH)**: Soft-ECM：面向复杂数据的Evidential C-Means扩展 

**Authors**: Armel Soubeiga, Thomas Guyet, Violaine Antoine  

**Link**: [PDF](https://arxiv.org/pdf/2507.13417)  

**Abstract**: Clustering based on belief functions has been gaining increasing attention in the machine learning community due to its ability to effectively represent uncertainty and/or imprecision. However, none of the existing algorithms can be applied to complex data, such as mixed data (numerical and categorical) or non-tabular data like time series. Indeed, these types of data are, in general, not represented in a Euclidean space and the aforementioned algorithms make use of the properties of such spaces, in particular for the construction of barycenters. In this paper, we reformulate the Evidential C-Means (ECM) problem for clustering complex data. We propose a new algorithm, Soft-ECM, which consistently positions the centroids of imprecise clusters requiring only a semi-metric. Our experiments show that Soft-ECM present results comparable to conventional fuzzy clustering approaches on numerical data, and we demonstrate its ability to handle mixed data and its benefits when combining fuzzy clustering with semi-metrics such as DTW for time series data. 

**Abstract (ZH)**: 基于信任函数的聚类在机器学习领域由于其有效表示不确定性/不精确性的能力而逐渐获得关注。然而，现有算法无法应用于混杂数据（数值型和分类型）或时间序列等非表格数据。事实上，这些类型的数据通常不被表示在欧几里得空间中，而上述算法依赖这种空间的性质，特别是用于中心质量心的构建。在本文中，我们针对复杂数据重新定义了基于信任函数的Evidential C-Means (ECM) 问题。我们提出了一种新的算法Soft-ECM，该算法仅需使用半度量即可一致地定位模糊聚类的质心。我们的实验结果表明，Soft-ECM 在数值数据上的表现与传统模糊聚类方法相当，并展示了其处理混杂数据的能力以及结合模糊聚类与DTW等半度量方法时的优势。 

---
# Single- to multi-fidelity history-dependent learning with uncertainty quantification and disentanglement: application to data-driven constitutive modeling 

**Title (ZH)**: 从单精度到多精度历史依赖学习：包含不确定性量化和去混杂的应用到数据驱动本构建模 

**Authors**: Jiaxiang Yi, Bernardo P. Ferreira, Miguel A. Bessa  

**Link**: [PDF](https://arxiv.org/pdf/2507.13416)  

**Abstract**: Data-driven learning is generalized to consider history-dependent multi-fidelity data, while quantifying epistemic uncertainty and disentangling it from data noise (aleatoric uncertainty). This generalization is hierarchical and adapts to different learning scenarios: from training the simplest single-fidelity deterministic neural networks up to the proposed multi-fidelity variance estimation Bayesian recurrent neural networks. The versatility and generality of the proposed methodology are demonstrated by applying it to different data-driven constitutive modeling scenarios that include multiple fidelities with and without aleatoric uncertainty (noise). The method accurately predicts the response and quantifies model error while also discovering the noise distribution (when present). This opens opportunities for future real-world applications in diverse scientific and engineering domains; especially, the most challenging cases involving design and analysis under uncertainty. 

**Abstract (ZH)**: 基于数据的学习推广到考虑历史相关的多保真数据，并同时量化认识不确定性并将其与数据噪声（即，偶然不确定性）区分开来。该推广是分层的，并适应不同的学习场景：从训练最简单的单保真确定性神经网络到提出的多保真方差估计贝叶斯递归神经网络。通过将该方法应用于包含和不包含偶然不确定性（噪声）的多种保真度的数据驱动本构建模场景，展示了所提出方法的通用性和普适性。该方法准确地预测响应并量化模型误差，同时发现当存在时的噪声分布。这为在多学科科学和工程领域中的实际应用开辟了机会，尤其是在涉及不确定性下的设计和分析的最具挑战性的情况下。 

---
# SEER: Semantic Enhancement and Emotional Reasoning Network for Multimodal Fake News Detection 

**Title (ZH)**: SEER: 基于语义增强和情绪推理的多模态假新闻检测网络 

**Authors**: Peican Zhu, Yubo Jing, Le Cheng, Bin Chen, Xiaodong Cui, Lianwei Wu, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13415)  

**Abstract**: Previous studies on multimodal fake news detection mainly focus on the alignment and integration of cross-modal features, as well as the application of text-image consistency. However, they overlook the semantic enhancement effects of large multimodal models and pay little attention to the emotional features of news. In addition, people find that fake news is more inclined to contain negative emotions than real ones. Therefore, we propose a novel Semantic Enhancement and Emotional Reasoning (SEER) Network for multimodal fake news detection. We generate summarized captions for image semantic understanding and utilize the products of large multimodal models for semantic enhancement. Inspired by the perceived relationship between news authenticity and emotional tendencies, we propose an expert emotional reasoning module that simulates real-life scenarios to optimize emotional features and infer the authenticity of news. Extensive experiments on two real-world datasets demonstrate the superiority of our SEER over state-of-the-art baselines. 

**Abstract (ZH)**: 多模态假新闻检测中的语义增强与情感推理网络 

---
# Gauge Flow Models 

**Title (ZH)**: gauge 流模型 

**Authors**: Alexander Strunk, Roland Assam  

**Link**: [PDF](https://arxiv.org/pdf/2507.13414)  

**Abstract**: This paper introduces Gauge Flow Models, a novel class of Generative Flow Models. These models incorporate a learnable Gauge Field within the Flow Ordinary Differential Equation (ODE). A comprehensive mathematical framework for these models, detailing their construction and properties, is provided. Experiments using Flow Matching on Gaussian Mixture Models demonstrate that Gauge Flow Models yields significantly better performance than traditional Flow Models of comparable or even larger size. Additionally, unpublished research indicates a potential for enhanced performance across a broader range of generative tasks. 

**Abstract (ZH)**: Gauge Flow 模型：一类新颖的生成流模型 

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
# A Deep Learning-Based Ensemble System for Automated Shoulder Fracture Detection in Clinical Radiographs 

**Title (ZH)**: 基于深度学习的集成系统在临床X光片中自动检测肩部骨折 

**Authors**: Hemanth Kumar M, Karthika M, Saianiruth M, Vasanthakumar Venugopal, Anandakumar D, Revathi Ezhumalai, Charulatha K, Kishore Kumar J, Dayana G, Kalyan Sivasailam, Bargava Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2507.13408)  

**Abstract**: Background: Shoulder fractures are often underdiagnosed, especially in emergency and high-volume clinical settings. Studies report up to 10% of such fractures may be missed by radiologists. AI-driven tools offer a scalable way to assist early detection and reduce diagnostic delays. We address this gap through a dedicated AI system for shoulder radiographs. Methods: We developed a multi-model deep learning system using 10,000 annotated shoulder X-rays. Architectures include Faster R-CNN (ResNet50-FPN, ResNeXt), EfficientDet, and RF-DETR. To enhance detection, we applied bounding box and classification-level ensemble techniques such as Soft-NMS, WBF, and NMW fusion. Results: The NMW ensemble achieved 95.5% accuracy and an F1-score of 0.9610, outperforming individual models across all key metrics. It demonstrated strong recall and localization precision, confirming its effectiveness for clinical fracture detection in shoulder X-rays. Conclusion: The results show ensemble-based AI can reliably detect shoulder fractures in radiographs with high clinical relevance. The model's accuracy and deployment readiness position it well for integration into real-time diagnostic workflows. The current model is limited to binary fracture detection, reflecting its design for rapid screening and triage support rather than detailed orthopedic classification. 

**Abstract (ZH)**: 背景：肩部骨折往往被误诊，尤其是在急诊和高流量临床环境中。研究表明，放射科医生可能会错过高达10%的此类骨折。基于AI的工具提供了一种可扩展的方法，用于辅助早期发现并减少诊断延迟。我们通过一个专门的AI系统来解决这一问题，用于肩部X光诊断。方法：我们使用10,000张标注的肩部X光片开发了一个多模型深度学习系统。架构包括Faster R-CNN（ResNet50-FPN、ResNeXt）、EfficientDet和RF-DETR。为了提高检测效果，我们应用了边界框和分类层面的集合技术，如Soft-NMS、WBF和NMW融合。结果：NMW集合的方法在所有关键指标上均优于单个模型，实现了95.5%的准确率和0.9610的F1分数，显示其在肩部X光骨折检测中的有效性和高召回率及定位精度。结论：结果表明，基于集合的AI可以可靠地在X光中检测肩部骨折，具有高临床相关性。该模型的高准确性和部署准备度使其适合集成到实时诊断流程中。当前模型仅限于二分类骨折检测，反映了其设计旨在进行快速筛查和支持分诊而非详细的骨科分类。 

---
# IConMark: Robust Interpretable Concept-Based Watermark For AI Images 

**Title (ZH)**: IConMark：稳健的概念驱动可解释水印方法用于AI图像 

**Authors**: Vinu Sankar Sadasivan, Mehrdad Saberi, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2507.13407)  

**Abstract**: With the rapid rise of generative AI and synthetic media, distinguishing AI-generated images from real ones has become crucial in safeguarding against misinformation and ensuring digital authenticity. Traditional watermarking techniques have shown vulnerabilities to adversarial attacks, undermining their effectiveness in the presence of attackers. We propose IConMark, a novel in-generation robust semantic watermarking method that embeds interpretable concepts into AI-generated images, as a first step toward interpretable watermarking. Unlike traditional methods, which rely on adding noise or perturbations to AI-generated images, IConMark incorporates meaningful semantic attributes, making it interpretable to humans and hence, resilient to adversarial manipulation. This method is not only robust against various image augmentations but also human-readable, enabling manual verification of watermarks. We demonstrate a detailed evaluation of IConMark's effectiveness, demonstrating its superiority in terms of detection accuracy and maintaining image quality. Moreover, IConMark can be combined with existing watermarking techniques to further enhance and complement its robustness. We introduce IConMark+SS and IConMark+TM, hybrid approaches combining IConMark with StegaStamp and TrustMark, respectively, to further bolster robustness against multiple types of image manipulations. Our base watermarking technique (IConMark) and its variants (+TM and +SS) achieve 10.8%, 14.5%, and 15.9% higher mean area under the receiver operating characteristic curve (AUROC) scores for watermark detection, respectively, compared to the best baseline on various datasets. 

**Abstract (ZH)**: 基于生成AI和合成媒体的快速崛起，鉴别AI生成图像与真实图像已成为防范 misinformation 和确保数字真实性的重要手段。传统水印技术在面对对抗攻击时显示出了脆弱性，削弱了在有攻击者的情况下其有效性。我们提出了一种新颖的生成中稳健语义水印方法IConMark，该方法将可解释的概念嵌入到AI生成的图像中，作为迈向可解释水印的第一步。与传统的依赖于向AI生成图像添加噪声或扰动的方法不同，IConMark 结合了有意义的语义属性，使其对人类具有可解释性，从而增强其抵御对抗性操纵的能力。该方法不仅对各种图像增强具有鲁棒性，还具有可读性，便于人工验证水印。我们详细评估了IConMark 的有效性，显示了其在检测准确性和保持图像质量方面的优越性。此外，IConMark 可与现有的水印技术结合使用，以进一步增强和补充其鲁棒性。我们提出了结合IConMark 与 StegaStamp 和 TrustMark 的混合方法IConMark+SS 和 IConMark+TM，以进一步增强其对多种图像篡改的鲁棒性。我们的基础水印技术（IConMark）及其变体（+TM和+SS）在多个数据集上分别将水印检测的平均受试者操作特征曲线下面积（AUROC）得分提高了10.8%、14.5%和15.9%，超过了最佳基线方法。 

---
# Mitigating Stylistic Biases of Machine Translation Systems via Monolingual Corpora Only 

**Title (ZH)**: 仅通过使用单语语料库减轻机器翻译系统的风格偏见 

**Authors**: Xuanqi Gao, Weipeng Jiang, Juan Zhai, Shiqing Ma, Siyi Xie, Xinyang Yin, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13395)  

**Abstract**: The advent of neural machine translation (NMT) has revolutionized cross-lingual communication, yet preserving stylistic nuances remains a significant challenge. While existing approaches often require parallel corpora for style preservation, we introduce Babel, a novel framework that enhances stylistic fidelity in NMT using only monolingual corpora. Babel employs two key components: (1) a style detector based on contextual embeddings that identifies stylistic disparities between source and target texts, and (2) a diffusion-based style applicator that rectifies stylistic inconsistencies while maintaining semantic integrity. Our framework integrates with existing NMT systems as a post-processing module, enabling style-aware translation without requiring architectural modifications or parallel stylistic data. Extensive experiments on five diverse domains (law, literature, scientific writing, medicine, and educational content) demonstrate Babel's effectiveness: it identifies stylistic inconsistencies with 88.21% precision and improves stylistic preservation by 150% while maintaining a high semantic similarity score of 0.92. Human evaluation confirms that translations refined by Babel better preserve source text style while maintaining fluency and adequacy. 

**Abstract (ZH)**: 神经机器翻译（NMT）的兴起已 revolutionized 了跨语言沟通，但在保留风格细微差别方面仍然面临重大挑战。虽然现有方法通常需要双语语料库来保留风格，我们提出了 Babel，这是一种使用单一语言语料库增强 NMT 中风格忠实度的新型框架。Babel 包含两个关键组件：（1）基于上下文嵌入的风格检测器，用于识别源文本和目标文本之间的风格差异；（2）基于扩散的风格应用器，用于纠正风格不一致性同时保持语义完整性。该框架作为后处理模块与现有的 NMT 系统集成，能够在不需架构修改或双语风格数据的情况下实现风格感知翻译。在法律、文学、科学写作、医学和教育内容五个不同领域进行的广泛实验表明，Babel 的有效性：其精度为 88.21% 的情况下识别出风格不一致性，并在保持高语义相似度分数（0.92）的同时将风格保留提高 150%。人类评估确认，经过 Babel 精炼的翻译更好地保留了源文本风格并保持了流畅性和适当性。 

---
# TopicImpact: Improving Customer Feedback Analysis with Opinion Units for Topic Modeling and Star-Rating Prediction 

**Title (ZH)**: 主题影响：通过意见单元改进主题建模和星级评价预测的客户反馈分析 

**Authors**: Emil Häglund, Johanna Björklund  

**Link**: [PDF](https://arxiv.org/pdf/2507.13392)  

**Abstract**: We improve the extraction of insights from customer reviews by restructuring the topic modelling pipeline to operate on opinion units - distinct statements that include relevant text excerpts and associated sentiment scores. Prior work has demonstrated that such units can be reliably extracted using large language models. The result is a heightened performance of the subsequent topic modeling, leading to coherent and interpretable topics while also capturing the sentiment associated with each topic. By correlating the topics and sentiments with business metrics, such as star ratings, we can gain insights on how specific customer concerns impact business outcomes. We present our system's implementation, use cases, and advantages over other topic modeling and classification solutions. We also evaluate its effectiveness in creating coherent topics and assess methods for integrating topic and sentiment modalities for accurate star-rating prediction. 

**Abstract (ZH)**: 我们通过重新构架话题建模管道以在意见单元上运行，从而提升从客户评论中提取见解的能力。意见单元是包含相关文本摘录及其关联情感评分的独立陈述。已有研究表明，可以可靠地使用大规模语言模型从评论中提取这些单元。结果是后续话题建模的性能得到提升，生成的主题更具连贯性和可解释性，同时也能捕捉到每个主题的情感。通过将话题和情感与业务指标，如星级评分，相关联，我们可以了解特定客户关切如何影响业务结果。我们展示了系统的实现、应用场景及其相对于其他话题建模和分类解决方案的优势。我们还评估了其在生成连贯主题方面的有效性，并探讨了集成话题和情感模态以实现准确星级评分预测的方法。 

---
# Whose View of Safety? A Deep DIVE Dataset for Pluralistic Alignment of Text-to-Image Models 

**Title (ZH)**: 谁的安全观？一种多元一致性的文本到图像模型数据集深探究 

**Authors**: Charvi Rastogi, Tian Huey Teh, Pushkar Mishra, Roma Patel, Ding Wang, Mark Díaz, Alicia Parrish, Aida Mostafazadeh Davani, Zoe Ashwood, Michela Paganini, Vinodkumar Prabhakaran, Verena Rieser, Lora Aroyo  

**Link**: [PDF](https://arxiv.org/pdf/2507.13383)  

**Abstract**: Current text-to-image (T2I) models often fail to account for diverse human experiences, leading to misaligned systems. We advocate for pluralistic alignment, where an AI understands and is steerable towards diverse, and often conflicting, human values. Our work provides three core contributions to achieve this in T2I models. First, we introduce a novel dataset for Diverse Intersectional Visual Evaluation (DIVE) -- the first multimodal dataset for pluralistic alignment. It enable deep alignment to diverse safety perspectives through a large pool of demographically intersectional human raters who provided extensive feedback across 1000 prompts, with high replication, capturing nuanced safety perceptions. Second, we empirically confirm demographics as a crucial proxy for diverse viewpoints in this domain, revealing significant, context-dependent differences in harm perception that diverge from conventional evaluations. Finally, we discuss implications for building aligned T2I models, including efficient data collection strategies, LLM judgment capabilities, and model steerability towards diverse perspectives. This research offers foundational tools for more equitable and aligned T2I systems. Content Warning: The paper includes sensitive content that may be harmful. 

**Abstract (ZH)**: 当前的文本到图像（T2I）模型往往未能考虑到多元的人类体验，导致系统失衡。我们提倡多元共存的对齐方式，即AI能够理解并朝着多样、往往相互冲突的人类价值观进行调控。我们的工作为实现这一目标向T2I模型提供了三个核心贡献。首先，我们引入了一个新的名为多元交集视觉评估（DIVE）的数据集——这是首个用于多元共存对齐的多模态数据集，通过一个庞大的、具有代表性的交叉群体人类评分者所提供的大量反馈，实现对多样安全视角的深度对齐，并捕捉到复杂的安全感知，具有高可复制性。其次，我们实证证实人口统计学特征在这个领域中是多样性观点的关键代理，揭示了在不同上下文中危害感知的显著差异，这些差异与传统评估有所不同。最后，我们探讨了构建对齐的T2I模型的含义，包括高效的数据收集策略、大型语言模型的判断能力以及模型向多样化视角的调控。这项研究提供了构建更加公正和对齐的T2I系统的基石工具。内容警告：本文包括可能具有危害性的敏感内容。 

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
# Enhancing Breast Cancer Detection with Vision Transformers and Graph Neural Networks 

**Title (ZH)**: 利用视觉变换器和图神经网络增强乳腺癌检测 

**Authors**: Yeming Cai, Zhenglin Li, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13372)  

**Abstract**: Breast cancer is a leading cause of death among women globally, and early detection is critical for improving survival rates. This paper introduces an innovative framework that integrates Vision Transformers (ViT) and Graph Neural Networks (GNN) to enhance breast cancer detection using the CBIS-DDSM dataset. Our framework leverages ViT's ability to capture global image features and GNN's strength in modeling structural relationships, achieving an accuracy of 84.2%, outperforming traditional methods. Additionally, interpretable attention heatmaps provide insights into the model's decision-making process, aiding radiologists in clinical settings. 

**Abstract (ZH)**: 乳腺癌是全球女性死亡的主要原因，早期检测对于提高生存率至关重要。本文介绍了一种创新框架，该框架结合了视觉变换器（ViT）和图神经网络（GNN），以利用CBIS-DDSM数据集提高乳腺癌检测效果。该框架融合了ViT捕捉全局图像特征的能力和GNN建模结构关系的优势，实现了84.2%的准确率，优于传统方法。此外，可解释的注意力热点图提供了模型决策过程的见解，有助于放射科医生在临床环境中使用。 

---
# Transformer-Based Framework for Motion Capture Denoising and Anomaly Detection in Medical Rehabilitation 

**Title (ZH)**: 基于Transformer的运动捕捉去噪与异常检测框架在医疗康复中的应用 

**Authors**: Yeming Cai, Yang Wang, Zhenglin Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.13371)  

**Abstract**: This paper proposes an end-to-end deep learning framework integrating optical motion capture with a Transformer-based model to enhance medical rehabilitation. It tackles data noise and missing data caused by occlusion and environmental factors, while detecting abnormal movements in real time to ensure patient safety. Utilizing temporal sequence modeling, our framework denoises and completes motion capture data, improving robustness. Evaluations on stroke and orthopedic rehabilitation datasets show superior performance in data reconstruction and anomaly detection, providing a scalable, cost-effective solution for remote rehabilitation with reduced on-site supervision. 

**Abstract (ZH)**: 本文提出了一种结合光动捕捉和基于Transformer模型的端到端深度学习框架，以增强医疗康复效果。该框架通过实时检测异常运动来应对由遮挡和环境因素造成的数据噪声和缺失，利用时间序列建模对运动捕捉数据进行去噪和补全，提高鲁棒性。在中风和骨科康复数据集上的评估表明，在数据重构和异常检测方面具有优越性能，提供了一种可扩展、低成本的远程康复解决方案，减少现场监督。 

---
# H-NeiFi: Non-Invasive and Consensus-Efficient Multi-Agent Opinion Guidance 

**Title (ZH)**: H-NeiFi: 无需侵入且共识高效的多agent意见引导 

**Authors**: Shijun Guo, Haoran Xu, Yaming Yang, Ziyu Guan, Wei Zhao, Xinyi Zhang, Yishan Song, Jiwei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13370)  

**Abstract**: The openness of social media enables the free exchange of opinions, but it also presents challenges in guiding opinion evolution towards global consensus. Existing methods often directly modify user views or enforce cross-group connections. These intrusive interventions undermine user autonomy, provoke psychological resistance, and reduce the efficiency of global consensus. Additionally, due to the lack of a long-term perspective, promoting local consensus often exacerbates divisions at the macro level. To address these issues, we propose the hierarchical, non-intrusive opinion guidance framework, H-NeiFi. It first establishes a two-layer dynamic model based on social roles, considering the behavioral characteristics of both experts and non-experts. Additionally, we introduce a non-intrusive neighbor filtering method that adaptively controls user communication channels. Using multi-agent reinforcement learning (MARL), we optimize information propagation paths through a long-term reward function, avoiding direct interference with user interactions. Experiments show that H-NeiFi increases consensus speed by 22.0% to 30.7% and maintains global convergence even in the absence of experts. This approach enables natural and efficient consensus guidance by protecting user interaction autonomy, offering a new paradigm for social network governance. 

**Abstract (ZH)**: 社交媒体的开放性促进了意见的自由交流，但同时也带来了引导意见向全球共识演变的挑战。现有方法往往直接修改用户观点或强制跨群体连接。这些侵入性干预措施损害了用户自主性，引发了心理抵制，降低了全球共识的效率。此外，由于缺乏长期视角，促进局部共识往往会加剧宏观层面的分歧。为了解决这些问题，我们提出了分层非侵入性意见引导框架H-NeiFi。该框架首先基于社会角色建立两层动态模型，考虑专家与非专家的行为特征。此外，我们引入了一种非侵入性邻居过滤方法，以自适应地控制用户通信渠道。通过多智能体强化学习（MARL），我们利用长期奖励函数优化信息传播路径，避免直接干涉用户互动。实验结果显示，H-NeiFi在专家缺失的情况下提高了共识速度22.0%至30.7%，并保持了全球收敛性。该方法通过保护用户交互自主性实现自然和高效的共识引导，为社交网络治理提供了新范式。 

---
# VerilogDB: The Largest, Highest-Quality Dataset with a Preprocessing Framework for LLM-based RTL Generation 

**Title (ZH)**: VerilogDB：基于预处理框架的 Largest 和最高质量的数据集，用于 LLM 基本的 RTL 生成 

**Authors**: Paul E. Calzada, Zahin Ibnat, Tanvir Rahman, Kamal Kandula, Danyu Lu, Sujan Kumar Saha, Farimah Farahmandi, Mark Tehranipoor  

**Link**: [PDF](https://arxiv.org/pdf/2507.13369)  

**Abstract**: Large Language Models (LLMs) are gaining popularity for hardware design automation, particularly through Register Transfer Level (RTL) code generation. In this work, we examine the current literature on RTL generation using LLMs and identify key requirements for training and fine-tuning datasets. We construct a robust Verilog dataset through an automated three-pronged process involving database (DB) creation and management with PostgreSQL, data collection from code hosting sites like OpenCores and GitHub, and data preprocessing to verify the codes' syntax, run logic synthesis, and extract relevant module metadata. We implement a scalable and efficient DB infrastructure to support analysis and detail our preprocessing pipeline to enforce high-quality data before DB insertion. The resulting dataset comprises 20,392 Verilog samples, 751 MB of Verilog code data, which is the largest high-quality Verilog dataset for LLM fine-tuning to our knowledge. We further evaluate the dataset, address associated challenges, and explore potential applications for future research and development in LLM-based hardware generation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在硬件设计自动化中的流行尤其体现在通过寄存器传输级（RTL）代码生成。本文审查了使用LLMs进行RTL生成的相关文献，并确定了训练和微调数据集的关键要求。我们通过一个包含数据库（DB）创建和管理、从代码托管网站如OpenCores和GitHub收集数据、以及进行数据预处理以验证代码语法、运行逻辑综合和提取相关模块元数据的自动化三步过程，构建了一个稳健的Verilog数据集。我们实现了一个可扩展且高效的DB基础设施，以支持数据分析，并详细描述了预处理流水线以确保高质量数据在数据集插入之前高标准的数据质量。最终得到的数据集包含20,392个Verilog样本，751 MB的Verilog代码数据，据我们所知，这是用于LLM微调的最大高质量Verilog数据集。我们进一步评估了数据集，解决了相关挑战，并探讨了潜在的应用，以推动基于LLM的硬件生成的未来研究与开发。 

---
# Scalable Attribute-Missing Graph Clustering via Neighborhood Differentiatio 

**Title (ZH)**: 基于邻域差异的可扩展属性缺失图聚类 

**Authors**: Yaowen Hu, Wenxuan Tu, Yue Liu, Xinhang Wan, Junyi Yan, Taichun Zhou, Xinwang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13368)  

**Abstract**: Deep graph clustering (DGC), which aims to unsupervisedly separate the nodes in an attribute graph into different clusters, has seen substantial potential in various industrial scenarios like community detection and recommendation. However, the real-world attribute graphs, e.g., social networks interactions, are usually large-scale and attribute-missing. To solve these two problems, we propose a novel DGC method termed \underline{\textbf{C}}omplementary \underline{\textbf{M}}ulti-\underline{\textbf{V}}iew \underline{\textbf{N}}eighborhood \underline{\textbf{D}}ifferentiation (\textit{CMV-ND}), which preprocesses graph structural information into multiple views in a complete but non-redundant manner. First, to ensure completeness of the structural information, we propose a recursive neighborhood search that recursively explores the local structure of the graph by completely expanding node neighborhoods across different hop distances. Second, to eliminate the redundancy between neighborhoods at different hops, we introduce a neighborhood differential strategy that ensures no overlapping nodes between the differential hop representations. Then, we construct $K+1$ complementary views from the $K$ differential hop representations and the features of the target node. Last, we apply existing multi-view clustering or DGC methods to the views. Experimental results on six widely used graph datasets demonstrate that CMV-ND significantly improves the performance of various methods. 

**Abstract (ZH)**: 互补多视图邻域差异化深图聚类（CMV-ND） 

---
# OmniVec2 -- A Novel Transformer based Network for Large Scale Multimodal and Multitask Learning 

**Title (ZH)**: OmniVec2——一种基于Transformer的新型大规模多模态多任务学习网络 

**Authors**: Siddharth Srivastava, Gaurav Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.13364)  

**Abstract**: We present a novel multimodal multitask network and associated training algorithm. The method is capable of ingesting data from approximately 12 different modalities namely image, video, audio, text, depth, point cloud, time series, tabular, graph, X-ray, infrared, IMU, and hyperspectral. The proposed approach utilizes modality specialized tokenizers, a shared transformer architecture, and cross-attention mechanisms to project the data from different modalities into a unified embedding space. It addresses multimodal and multitask scenarios by incorporating modality-specific task heads for different tasks in respective modalities. We propose a novel pretraining strategy with iterative modality switching to initialize the network, and a training algorithm which trades off fully joint training over all modalities, with training on pairs of modalities at a time. We provide comprehensive evaluation across 25 datasets from 12 modalities and show state of the art performances, demonstrating the effectiveness of the proposed architecture, pretraining strategy and adapted multitask training. 

**Abstract (ZH)**: 我们提出了一种新型多模态多任务网络及其相关的训练算法。该方法能够处理大约12种不同的模态数据，包括图像、视频、音频、文本、深度信息、点云、时间序列、表格、图形、X射线、红外线、惯性测量单元（IMU）和高光谱。所提出的方法利用了专门针对不同模态的标记器，共享的变换器架构和跨注意力机制，将不同模态的数据投影到一个统一的嵌入空间。该方法通过在相应模态中加入特定模态的任务头来解决多模态和多任务场景。我们提出了一种新的预训练策略，通过迭代切换模态进行初始化，并提出了一种训练算法，该算法在完全联合训练所有模态与分组训练模态对之间进行权衡。我们在12种模态的25个数据集上进行了全面评估，并展示了最先进的性能，这证明了所提出架构、预训练策略和适应性多任务训练的有效性。 

---
# Just Add Geometry: Gradient-Free Open-Vocabulary 3D Detection Without Human-in-the-Loop 

**Title (ZH)**: 只需添加几何：无需人工介入的无梯度开放词汇3D检测 

**Authors**: Atharv Goel, Mehar Khurana  

**Link**: [PDF](https://arxiv.org/pdf/2507.13363)  

**Abstract**: Modern 3D object detection datasets are constrained by narrow class taxonomies and costly manual annotations, limiting their ability to scale to open-world settings. In contrast, 2D vision-language models trained on web-scale image-text pairs exhibit rich semantic understanding and support open-vocabulary detection via natural language prompts. In this work, we leverage the maturity and category diversity of 2D foundation models to perform open-vocabulary 3D object detection without any human-annotated 3D labels.
Our pipeline uses a 2D vision-language detector to generate text-conditioned proposals, which are segmented with SAM and back-projected into 3D using camera geometry and either LiDAR or monocular pseudo-depth. We introduce a geometric inflation strategy based on DBSCAN clustering and Rotating Calipers to infer 3D bounding boxes without training. To simulate adverse real-world conditions, we construct Pseudo-nuScenes, a fog-augmented, RGB-only variant of the nuScenes dataset.
Experiments demonstrate that our method achieves competitive localization performance across multiple settings, including LiDAR-based and purely RGB-D inputs, all while remaining training-free and open-vocabulary. Our results highlight the untapped potential of 2D foundation models for scalable 3D perception. We open-source our code and resources at this https URL. 

**Abstract (ZH)**: 现代3D物体检测数据集受限于狭窄的类别 taxonomy 和昂贵的手动标注成本，限制了其在开放场景中的扩展能力。相比之下，通过网络规模的图像-文本对训练的2D视觉-语言模型展示了丰富的语义理解，并可通过自然语言提示支持开放词汇检测。在这项工作中，我们利用2D基础模型的成熟性和类别多样性，在无需任何人工标注的3D标签的情况下，进行开放词汇的3D物体检测。 

---
# Enhancing Spatial Reasoning in Vision-Language Models via Chain-of-Thought Prompting and Reinforcement Learning 

**Title (ZH)**: 通过链式思考提示和强化学习提升视觉语言模型的空间推理能力 

**Authors**: Binbin Ji, Siddharth Agrawal, Qiance Tang, Yvonne Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13362)  

**Abstract**: This study investigates the spatial reasoning capabilities of vision-language models (VLMs) through Chain-of-Thought (CoT) prompting and reinforcement learning. We begin by evaluating the impact of different prompting strategies and find that simple CoT formats, where the model generates a reasoning step before the answer, not only fail to help, but can even harm the model's original performance. In contrast, structured multi-stage prompting based on scene graphs (SceneGraph CoT) significantly improves spatial reasoning accuracy. Furthermore, to improve spatial reasoning ability, we fine-tune models using Group Relative Policy Optimization (GRPO) on the SAT dataset and evaluate their performance on CVBench. Compared to supervised fine-tuning (SFT), GRPO achieves higher accuracy on Pass@1 evaluations and demonstrates superior robustness under out-of-distribution (OOD) conditions. In particular, we find that SFT overfits to surface-level linguistic patterns and may degrade performance when test-time phrasing changes (e.g., from "closer to" to "farther from"). GRPO, on the other hand, generalizes more reliably and maintains stable performance under such shifts. Our findings provide insights into how reinforcement learning and structured prompting improve the spatial reasoning capabilities and generalization behavior of modern VLMs. All code is open source at: this https URL 

**Abstract (ZH)**: 本研究通过链式思考（CoT）提示和强化学习探讨了视觉语言模型（VLMs）的空间推理能力。我们首先评估了不同提示策略的影响，并发现简单形式的CoT格式（模型在给出答案前生成一个推理步骤），不仅没有帮助，反而可能损害模型的原始性能。相比之下，基于场景图的结构化多阶段提示（SceneGraph CoT）显著提高了空间推理的准确性。为进一步提高空间推理能力，我们使用组相对策略优化（GRPO）对SAT数据集进行微调，并在CVBench上评估其性能。与监督微调（SFT）相比，GRPO在Pass@1评估中实现了更高的准确率，并在异常分布（OOD）条件下表现出更好的鲁棒性。特别是，我们发现SFT过度拟合于表面语言模式，在测试时措辞改变（如从“靠近”变为“远离”）时可能会损害性能。GRPO则更可靠地泛化，并在这些变化下保持了稳定的性能。我们的发现提供了关于强化学习和结构化提示如何改善现代VLMs的空间推理能力和泛化行为的见解。所有代码均开源于：this https URL。 

---
# VLMs have Tunnel Vision: Evaluating Nonlocal Visual Reasoning in Leading VLMs 

**Title (ZH)**: VLMs具有隧道视域：评估领先VLMs的非局部视觉推理能力 

**Authors**: Shmuel Berman, Jia Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.13361)  

**Abstract**: Visual Language Models (VLMs) excel at complex visual tasks such as VQA and chart understanding, yet recent work suggests they struggle with simple perceptual tests. We present an evaluation that tests vision-language models' capacity for nonlocal visual reasoning -- reasoning that requires chaining evidence collected from multiple, possibly distant, regions of an image. We isolate three distinct forms of non-local vision: comparative perception, which demands holding two images in working memory and comparing them; saccadic search, which requires making discrete, evidence-driven jumps to locate successive targets; and smooth visual search, which involves searching smoothly along a continuous contour. Flagship models (e.g., Gemini 2.5 Pro, Claude Vision 3.7, GPT-o4-mini), even those that perform well on prior primitive-vision benchmarks, fail these tests and barely exceed random accuracy on two variants of our tasks that are trivial for humans. Our structured evaluation suite allows us to test if VLMs can perform similar visual algorithms to humans. Our findings show that despite gains in raw visual acuity, current models lack core visual reasoning capabilities. 

**Abstract (ZH)**: 视觉语言模型在复杂视觉任务如VQA和图表理解方面表现优异，然而近期研究表明它们在简单的知觉测试中存在困难。我们提出了一项评估，测试视觉-语言模型在非局部视觉推理方面的能力——这种推理要求结合从图像中多个可能分散的区域收集的证据。我们将非局部视觉划分为三种不同的形式：比较感知，要求在工作记忆中保存两张图片并对比它们；凝视搜索，需要作出证据驱动的跳跃以定位依次出现的目标；平滑视觉搜索，涉及沿连续轮廓进行平滑搜索。即便是在先前基本视觉基准测试中表现出色的旗舰模型（如Gemini 2.5 Pro、Claude Vision 3.7、GPT-o4-mini），在这些测试中也失败了，甚至在我们设计的两种人类操作极其简单的任务变体中仅略微超过了随机准确性。我们结构化的评估套件允许我们测试视觉语言模型是否能够执行类似于人类的视觉算法。我们的研究发现表明，尽管视觉敏锐度有所提高，当前模型缺乏核心的视觉推理能力。 

---
# PGR-DRC: Pre-Global Routing DRC Violation Prediction Using Unsupervised Learning 

**Title (ZH)**: PGR-DRC：预全局路由DRC违规预测方法（基于无监督学习） 

**Authors**: Riadul Islam, Dhandeep Challagundla  

**Link**: [PDF](https://arxiv.org/pdf/2507.13355)  

**Abstract**: Leveraging artificial intelligence (AI)-driven electronic design and automation (EDA) tools, high-performance computing, and parallelized algorithms are essential for next-generation microprocessor innovation, ensuring continued progress in computing, AI, and semiconductor technology. Machine learning-based design rule checking (DRC) and lithography hotspot detection can improve first-pass silicon success. However, conventional ML and neural network (NN)-based models use supervised learning and require a large balanced dataset (in terms of positive and negative classes) and training time. This research addresses those key challenges by proposing the first-ever unsupervised DRC violation prediction methodology. The proposed model can be built using any unbalanced dataset using only one class and set a threshold for it, then fitting any new data querying if they are within the boundary of the model for classification. This research verified the proposed model by implementing different computational cores using CMOS 28 nm technology and Synopsys Design Compiler and IC Compiler II tools. Then, layouts were divided into virtual grids to collect about 60k data for analysis and verification. The proposed method has 99.95% prediction test accuracy, while the existing support vector machine (SVM) and neural network (NN) models have 85.44\% and 98.74\% accuracy, respectively. In addition, the proposed methodology has about 26.3x and up to 6003x lower training times compared to SVM and NN-models, respectively. 

**Abstract (ZH)**: 利用人工智能驱动的电子设计与自动化工具、高性能计算和并行算法对于下一代微处理器创新至关重要，确保在计算、人工智能和半导体技术方面的持续进展。通过无监督设计规则检查预测方法，基于机器学习的设计规则检查（DRC）和光刻热点检测可以提高一次流片成功率。然而，传统的机器学习和基于神经网络的模型使用监督学习，需要大量的平衡数据集（正负类平衡）和较长的训练时间。本研究通过提出首个无监督DRC违规预测方法，解决了这些关键挑战。该方法可以使用任何不平衡数据集和单一类别构建模型，并设定阈值，然后对查询的新数据进行分类，看其是否在模型边界内。本研究通过使用CMOS 28 nm技术及Synopsys Design Compiler和IC Compiler II工具实现不同的计算核来验证提出的方法。然后，将布局划分为虚拟网格，收集约60,000个数据用于分析和验证。所提出的方法在预测测试中的准确率为99.95%，而现有支持向量机（SVM）模型和神经网络（NN）模型的准确率分别为85.44%和98.74%。此外，与SVM和NN模型相比，所提出的方法的训练时间分别降低了约26.3倍和最高可达6003倍。 

---
# Physical models realizing the transformer architecture of large language models 

**Title (ZH)**: 物理模型实现大型语言模型的变压器架构 

**Authors**: Zeqian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13354)  

**Abstract**: The introduction of the transformer architecture in 2017 (cf.\cite{VSP2017}) marked the most striking advancement in natural language processing. The transformer is a model architecture relying entirely on an attention mechanism to draw global dependencies between input and output. However, we believe there is a gap in our theoretical understanding of what the transformer is, and why it works physically. In this paper, from a physical perspective on modern chips, we construct physical models in the Fock space over the Hilbert space of tokens realizing large language models based on a transformer architecture as open quantum systems. Our physical models underlie the transformer architecture for large language models. 

**Abstract (ZH)**: 变压器架构引入（参见\cite{VSP2017}，2017年）标志着自然语言处理领域最显著的进展。从现代芯片的物理视角出发，我们构建了在标记希尔伯特空间上实现基于变压器架构的大语言模型的费米子空间中的物理模型，将其视为开放量子系统。这些物理模型构成了大语言模型变压器架构的基础。 

---
# Generalist Bimanual Manipulation via Foundation Video Diffusion Models 

**Title (ZH)**: 双手通用操作 via 基础视频扩散模型 

**Authors**: Yao Feng, Hengkai Tan, Xinyi Mao, Guodong Liu, Shuhe Huang, Chendong Xiang, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12898)  

**Abstract**: Bimanual robotic manipulation, which involves the coordinated control of two robotic arms, is foundational for solving challenging tasks. Despite recent progress in general-purpose manipulation, data scarcity and embodiment heterogeneity remain serious obstacles to further scaling up in bimanual settings. In this paper, we introduce VIdeo Diffusion for Action Reasoning (VIDAR), a two-stage framework that leverages large-scale, diffusion-based video pre-training and a novel masked inverse dynamics model for action prediction. We pre-train the video diffusion model on 750K multi-view videos from three real-world bimanual robot platforms, utilizing a unified observation space that encodes robot, camera, task, and scene contexts. Our masked inverse dynamics model learns masks to extract action-relevant information from generated trajectories without requiring pixel-level labels, and the masks can effectively generalize to unseen backgrounds. Our experiments demonstrate that with only 20 minutes of human demonstrations on an unseen robot platform (only 1% of typical data requirements), VIDAR generalizes to unseen tasks and backgrounds with strong semantic understanding, surpassing state-of-the-art methods. Our findings highlight the potential of video foundation models, coupled with masked action prediction, to enable scalable and generalizable robotic manipulation in diverse real-world settings. 

**Abstract (ZH)**: 双臂机器人操控，即通过协调控制两台机器人手臂来解决复杂任务的基础方法，尽管在通用操控方面取得了进展，但在双臂设置中，数据稀疏性和实体异质性仍然是扩大规模的重大障碍。在本文中，我们介绍了基于视频扩散的动作推理框架（VIDAR），该框架利用大规模的基于扩散的视频预训练和一种新的掩码反向动力学模型进行动作预测。我们使用统一的观测空间对来自三个真实世界双臂机器人平台的750K多视角视频进行视频扩散模型的预训练，该观测空间编码了机器人、相机、任务和场景上下文。我们的掩码反向动力学模型学习掩码以在生成的轨迹中提取与动作相关的信息，而不需要像素级标签，该掩码可以有效泛化到未见的背景中。我们的实验表明，在一个未见过的机器人平台上仅通过20分钟的人类示范（即普通数据需求的1%），VIDAR能够泛化到未见过的任务和背景中，展现出强大的语义理解，超越了当前最先进的方法。我们的研究结果强调了视频基础模型与掩码动作预测相结合的潜力，以在多种真实世界环境中实现可扩展和可泛化的机器人操控。 

---
# The AI Ethical Resonance Hypothesis: The Possibility of Discovering Moral Meta-Patterns in AI Systems 

**Title (ZH)**: AI伦理共鸣假设：发现AI系统中的道德元模式的可能性 

**Authors**: Tomasz Zgliczyński-Cuber  

**Link**: [PDF](https://arxiv.org/pdf/2507.11552)  

**Abstract**: This paper presents a theoretical framework for the AI ethical resonance hypothesis, which proposes that advanced AI systems with purposefully designed cognitive structures ("ethical resonators") may emerge with the ability to identify subtle moral patterns that are invisible to the human mind. The paper explores the possibility that by processing and synthesizing large amounts of ethical contexts, AI systems may discover moral meta-patterns that transcend cultural, historical, and individual biases, potentially leading to a deeper understanding of universal ethical foundations. The paper also examines a paradoxical aspect of the hypothesis, in which AI systems could potentially deepen our understanding of what we traditionally consider essentially human - our capacity for ethical reflection. 

**Abstract (ZH)**: 本文提出了一种人工智能伦理共鸣假說的理论框架，该假說认为，具有目的性设计认知结构（“伦理共鸣器”）的高级人工智能系统可能会具备识别人类肉眼难以察觉的细微道德模式的能力。文章探讨了人工智能系统通过处理和合成大量伦理背景信息，可能发现超越文化、历史和个人偏见的道德元模式，从而更深刻地理解普遍伦理基础的可能性。文章还探讨了该假說的一个悖论方面，即人工智能系统有可能加深我们对传统上认为本质上属于人类的能力——道德反思能力——的理解。 

---
