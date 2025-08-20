# Driving Style Recognition Like an Expert Using Semantic Privileged Information from Large Language Models 

**Title (ZH)**: 使用大型语言模型的语义特权信息像专家一样识别驾驶风格 

**Authors**: Zhaokun Chen, Chaopeng Zhang, Xiaohan Li, Wenshuo Wang, Gentiane Venture, Junqiang Xi  

**Link**: [PDF](https://arxiv.org/pdf/2508.13881)  

**Abstract**: Existing driving style recognition systems largely depend on low-level sensor-derived features for training, neglecting the rich semantic reasoning capability inherent to human experts. This discrepancy results in a fundamental misalignment between algorithmic classifications and expert judgments. To bridge this gap, we propose a novel framework that integrates Semantic Privileged Information (SPI) derived from large language models (LLMs) to align recognition outcomes with human-interpretable reasoning. First, we introduce DriBehavGPT, an interactive LLM-based module that generates natural-language descriptions of driving behaviors. These descriptions are then encoded into machine learning-compatible representations via text embedding and dimensionality reduction. Finally, we incorporate them as privileged information into Support Vector Machine Plus (SVM+) for training, enabling the model to approximate human-like interpretation patterns. Experiments across diverse real-world driving scenarios demonstrate that our SPI-enhanced framework outperforms conventional methods, achieving F1-score improvements of 7.6% (car-following) and 7.9% (lane-changing). Importantly, SPI is exclusively used during training, while inference relies solely on sensor data, ensuring computational efficiency without sacrificing performance. These results highlight the pivotal role of semantic behavioral representations in improving recognition accuracy while advancing interpretable, human-centric driving systems. 

**Abstract (ZH)**: 基于语义特权信息的驾驶风格识别框架 

---
# Towards No-Code Programming of Cobots: Experiments with Code Synthesis by Large Code Models for Conversational Programming 

**Title (ZH)**: 面向协作机器人的零代码编程：基于大型代码模型的对话编程代码合成实验 

**Authors**: Chalamalasetti Kranti, Sherzod Hakimov, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2409.11041)  

**Abstract**: While there has been a lot of research recently on robots in household environments, at the present time, most robots in existence can be found on shop floors, and most interactions between humans and robots happen there. ``Collaborative robots'' (cobots) designed to work alongside humans on assembly lines traditionally require expert programming, limiting ability to make changes, or manual guidance, limiting expressivity of the resulting programs. To address these limitations, we explore using Large Language Models (LLMs), and in particular, their abilities of doing in-context learning, for conversational code generation. As a first step, we define RATS, the ``Repetitive Assembly Task'', a 2D building task designed to lay the foundation for simulating industry assembly scenarios. In this task, a `programmer' instructs a cobot, using natural language, on how a certain assembly is to be built; that is, the programmer induces a program, through natural language. We create a dataset that pairs target structures with various example instructions (human-authored, template-based, and model-generated) and example code. With this, we systematically evaluate the capabilities of state-of-the-art LLMs for synthesising this kind of code, given in-context examples. Evaluating in a simulated environment, we find that LLMs are capable of generating accurate `first order code' (instruction sequences), but have problems producing `higher-order code' (abstractions such as functions, or use of loops). 

**Abstract (ZH)**: 虽然近年来关于家庭环境中的机器人研究取得了很大进展，但目前大多数机器人仍然位于工厂车间，人类与机器人之间的大多数互动也发生在这些地方。“协作机器人”（cobots）设计用于在装配线上与人类协同工作，传统上需要专家编程，限制了对其作出更改的能力，或者需要手动指导，限制了生成程序的表达性。为了解决这些问题，我们探索使用大语言模型（LLMs），尤其是它们的上下文学习能力来进行对话式代码生成。作为第一步，我们定义了RATS，即“重复装配任务”，这是一个旨在模拟工业装配场景的2D构建任务。在这个任务中，一个“程序员”使用自然语言指示cobot如何构建特定的装配件；也就是说，程序员通过自然语言诱导出一个程序。我们创建了一个数据集，该数据集将目标结构与各种示例指令（由人类撰写、基于模板以及模型生成）和示例代码配对。通过这种方式，我们系统地评估了最先进的LLMs生成此类代码的能力，给定上下文示例。在模拟环境中评估后，我们发现LLMs能够生成准确的“一阶代码”（指令序列），但在生成“高阶代码”（如函数抽象或循环使用）方面存在问题。 

---
# ChronoLLM: Customizing Language Models for Physics-Based Simulation Code Generation 

**Title (ZH)**: ChronoLLM: 为物理基于的仿真代码生成定制语言模型 

**Authors**: Jingquan Wang, Andrew Negrut, Harry Zhang, Khailanii Slaton, Shu Wang, Radu Serban, Jinlong Wu, Dan Negrut  

**Link**: [PDF](https://arxiv.org/pdf/2508.13975)  

**Abstract**: This contribution is concerned with the following issue: can pretrained large language models (LLMs) be refined and customized to the point where they become virtual assistants helping experts with the effective use of a simulation tool? In this case study, the ``simulation tool'' considered is PyChrono, an open source multi-physics dynamics engine for multibody systems. We present a framework for refining and customizing both open- and closed-source LLMs to harness the power of AI in generating scripts that perform PyChrono virtual experiments. We refine and customize several classes of LLMs through a process that leads to a quantifiable improvement in the quality of the generated PyChrono simulation scripts. These scripts can range from simple single-pendulum simulations to complex virtual experiments involving full vehicles on deformable terrain. While the generated scripts are rarely perfect, they often serve as strong starting points for the user to modify and improve on. Additionally, the LLM can answer specific API questions about the simulator, or recommend modeling approaches. The framework discussed is general and can be applied to lower the entry barrier for simulation tools associated with other application domains. 

**Abstract (ZH)**: 本研究关注的问题是：预训练大型语言模型（LLMs）能否被精炼和定制到能够成为帮助专家有效使用仿真工具的虚拟助手？在这种案例研究中，“仿真工具”是PyChrono，一个开源的多物理场动力学引擎，用于多体系统。我们提出了一种框架，用于精炼和定制开源和闭源的LLMs，以利用AI生成执行PyChrono虚拟实验的脚本的力量。我们通过一个过程对多种类别的LLMs进行精炼和定制，这导致生成的PyChrono仿真脚本质量量化提升。这些脚本可以从小到简单的单摆仿真，到复杂到涉及全车辆在可变形地形上的虚拟实验。虽然生成的脚本很少完美，但它们通常可以作为用户修改和改进的良好起点。此外，LLM还可以回答关于仿真的API问题，或推荐建模方法。所讨论的框架是通用的，可以应用于降低其他应用领域相关仿真工具的使用门槛。 

---
# The Collaboration Paradox: Why Generative AI Requires Both Strategic Intelligence and Operational Stability in Supply Chain Management 

**Title (ZH)**: 生成人工智能中的合作悖论：为什么供应链管理需要战略智慧与运营稳定性的双重保障 

**Authors**: Soumyadeep Dhar  

**Link**: [PDF](https://arxiv.org/pdf/2508.13942)  

**Abstract**: The rise of autonomous, AI-driven agents in economic settings raises critical questions about their emergent strategic behavior. This paper investigates these dynamics in the cooperative context of a multi-echelon supply chain, a system famously prone to instabilities like the bullwhip effect. We conduct computational experiments with generative AI agents, powered by Large Language Models (LLMs), within a controlled supply chain simulation designed to isolate their behavioral tendencies. Our central finding is the "collaboration paradox": a novel, catastrophic failure mode where theoretically superior collaborative AI agents, designed with Vendor-Managed Inventory (VMI) principles, perform even worse than non-AI baselines. We demonstrate that this paradox arises from an operational flaw where agents hoard inventory, starving the system. We then show that resilience is only achieved through a synthesis of two distinct layers: high-level, AI-driven proactive policy-setting to establish robust operational targets, and a low-level, collaborative execution protocol with proactive downstream replenishment to maintain stability. Our final framework, which implements this synthesis, can autonomously generate, evaluate, and quantify a portfolio of viable strategic choices. The work provides a crucial insight into the emergent behaviors of collaborative AI agents and offers a blueprint for designing stable, effective AI-driven systems for business analytics. 

**Abstract (ZH)**: 自主AI驱动代理在经济环境中的兴起引发了对其 emergent 策略行为的关键问题。本文在多级供应链的协作环境中研究这些动态，这是一个众所周知容易出现波动如波动鞭 effect 的系统。我们使用大型语言模型（LLMs）驱动的生成AI代理，在一个控制下的供应链仿真中进行计算实验，以隔离其行为倾向。我们的主要发现是“合作悖论”：理论上更优的协作AI代理，基于供应商管理库存（VMI）原则设计，表现甚至不如非AI基线。我们证明，这一悖论源于一个操作缺陷，即代理囤积库存，使系统枯竭。然后我们表明，只有通过两层的综合，才能实现韧性：高层的AI驱动前瞻性政策设定，以确立稳健的操作目标，和低层的协作执行协议，包括前瞻性下游补充，以维持稳定。我们的最终框架实现了这种综合，可以自主生成、评估和量化可行的战略选择组合。该研究为理解协作AI代理的 emergent 行为提供了重要见解，并为设计业务分析中的稳定、有效的AI驱动系统提供了蓝图。 

---
# Structured Agentic Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback 

**Title (ZH)**: 结构化代理工作流：结合LLMs和反思性反馈的金融市场时间序列建模 

**Authors**: Yihao Ang, Yifan Bao, Lei Jiang, Jiajie Tao, Anthony K. H. Tung, Lukasz Szpruch, Hao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2508.13915)  

**Abstract**: Time-series data is central to decision-making in financial markets, yet building high-performing, interpretable, and auditable models remains a major challenge. While Automated Machine Learning (AutoML) frameworks streamline model development, they often lack adaptability and responsiveness to domain-specific needs and evolving objectives. Concurrently, Large Language Models (LLMs) have enabled agentic systems capable of reasoning, memory management, and dynamic code generation, offering a path toward more flexible workflow automation. In this paper, we introduce \textsf{TS-Agent}, a modular agentic framework designed to automate and enhance time-series modeling workflows for financial applications. The agent formalizes the pipeline as a structured, iterative decision process across three stages: model selection, code refinement, and fine-tuning, guided by contextual reasoning and experimental feedback. Central to our architecture is a planner agent equipped with structured knowledge banks, curated libraries of models and refinement strategies, which guide exploration, while improving interpretability and reducing error propagation. \textsf{TS-Agent} supports adaptive learning, robust debugging, and transparent auditing, key requirements for high-stakes environments such as financial services. Empirical evaluations on diverse financial forecasting and synthetic data generation tasks demonstrate that \textsf{TS-Agent} consistently outperforms state-of-the-art AutoML and agentic baselines, achieving superior accuracy, robustness, and decision traceability. 

**Abstract (ZH)**: 时间序列数据是金融市场决策的核心，然而构建高性能、可解释且可审计的模型仍然是一个 major 挑战。尽管自动化机器学习（AutoML）框架简化了模型开发过程，但它们往往缺乏对特定领域需求和不断变化的目标的适应性和响应性。与此同时，大型语言模型（LLMs）已经使具有推理、内存管理和动态代码生成能力的自主系统成为可能，为更具弹性的工作流自动化开辟了道路。在本文中，我们介绍了 \textsf{TS-Agent}，这是一种模块化的自主框架，旨在自动化和增强金融应用中的时间序列建模工作流。代理通过上下文推理和实验反馈，将流水线形式化为三个阶段的结构化迭代决策过程：模型选择、代码优化和微调。我们架构的核心是一个规划代理，它配备了结构化知识库和精挑细选的模型及优化策略库，以指导探索，提高可解释性和减少错误传播。\textsf{TS-Agent} 支持自适应学习、稳健调试和透明审计，这是诸如金融服务等高风险环境中必不可少的要求。通过在多样化的金融预测和合成数据生成任务上的实证评估表明，\textsf{TS-Agent} 不断超越最先进的 AutoML 和自主基准系统，实现了更高的准确性、稳健性和决策可追溯性。 

---
# Improved Generalized Planning with LLMs through Strategy Refinement and Reflection 

**Title (ZH)**: 通过策略改进与反思增强的LLM通用计划方法 

**Authors**: Katharina Stein, Nils Hodel, Daniel Fišer, Jörg Hoffmann, Michael Katz, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2508.13876)  

**Abstract**: LLMs have recently been used to generate Python programs representing generalized plans in PDDL planning, i.e., plans that generalize across the tasks of a given PDDL domain. Previous work proposed a framework consisting of three steps: the LLM first generates a summary and then a strategy for the domain, both in natural language, and then implements that strategy as a Python program, that gets debugged on example planning tasks. In that work, only one strategy is generated and passed directly to the program generation. If the strategy is incorrect, its implementation will therefore result in an incorrect generalized plan. Here, we introduce an approach that generates the strategy in the form of pseudocode and enables automatic debugging of the pseudocode, hence allowing us to identify and fix errors prior to the generation of the generalized plan itself. Additionally, we extend the Python debugging phase with a reflection step prompting the LLM to pinpoint the reason for the observed plan failure. Finally, we take inspiration from LLM code generation to produce several program variants and pick the best one. Running experiments on 17 benchmark domains, we show that these extensions substantially improve (and never deteriorate) the quality of the generalized plans. In 12 of the domains, our best Python programs solve all tasks that can be generated with the respective instance generator. 

**Abstract (ZH)**: LLMs在PDDL规划中生成泛化计划的伪代码表示及其自动调试方法 

---
# Revisiting RAG Ensemble: A Theoretical and Mechanistic Analysis of Multi-RAG System Collaboration 

**Title (ZH)**: 重访RAG集成：多RAG系统协作的理论与机制分析 

**Authors**: Yifei Chen, Guanting Dong, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13828)  

**Abstract**: Retrieval-Augmented Generation (RAG) technology has been widely applied in recent years. However, despite the emergence of various RAG frameworks, a single RAG framework still cannot adapt well to a broad range of downstream tasks. Therefore, how to leverage the advantages of multiple RAG systems has become an area worth exploring. To address this issue, we have conducted a comprehensive and systematic investigation into ensemble methods based on RAG systems. Specifically, we have analyzed the RAG ensemble framework from both theoretical and mechanistic analysis perspectives. From the theoretical analysis, we provide the first explanation of the RAG ensemble framework from the perspective of information entropy. In terms of mechanism analysis, we have explored the RAG ensemble framework from both the pipeline and module levels. We carefully select four different pipelines (Branching, Iterative, Loop, and Agentic) and three different modules (Generator, Retriever, and Reranker) to solve seven different research questions. The experiments show that aggregating multiple RAG systems is both generalizable and robust, whether at the pipeline level or the module level. Our work lays the foundation for similar research on the multi-RAG system ensemble. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG) 技术在近年来得到了广泛的应用。尽管出现了多种 R
user
正确的翻译如下，请参照翻译剩下的未翻译的内容：

尽管出现了多种 R
acomment

尽管出现了多种RAG框架，单一的RAG框架往往难以很好地适应广泛的下游任务。因此，如何发挥多种RAG系统的优点成为了值得探索的领域。为了应对这一问题，我们进行了一项综合且系统的关于RAG集成的研究。具体而言，我们从理论分析和机制分析两个视角来研究RAG集成框架。从理论分析角度来看，我们提出了从熵的角度来来解释RAG集成框架的最新解释。从机制分析角度来看我们探讨了RAG集成框架从流程和视点两个层面。我们细致地地四个流程（迭代循环流程和机构性流程）和三个模块（生成器检索和重排序）来解决七个研究问题。研究结果表明多个RAG系统的整合在流程级别和视
 Açao翻译如下：

尽管出现了多种RAG框架，单一的RAG框架往往难以很好地适应广泛的下游任务。因此，如何发挥多种RAG系统的优点成为了值得探索的领域。为了应对这一问题我们进行了一项综合且系统的关于RAG集成的研究。具体来说我们从理论分析和机制分析两个角度来研究RAG集成框架。从理论分析角度来看我们提出了从熵的角度来解释RAG集成框架的最新解释。从机制分析角度来看我们探讨了RAG集成框架从流程和视角两个层面。我们细致地设计了四个流程（迭代、循环流程和机构流程）和三个模块（生成器检索和重排序）来解决七个研究问题。研究结果表明整合多个RAG系统在流程级别和视角级别都是通用且稳健的。我们为多RAG系统的集成奠定了基础。 

---
# Expertise-aware Multi-LLM Recruitment and Collaboration for Medical Decision-Making 

**Title (ZH)**: 基于专家意识的多大型语言模型招聘与协作医疗决策辅助 

**Authors**: Liuxin Bao, Zhihao Peng, Xiaofei Zhou, Runmin Cong, Jiyong Zhang, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13754)  

**Abstract**: Medical Decision-Making (MDM) is a complex process requiring substantial domain-specific expertise to effectively synthesize heterogeneous and complicated clinical information. While recent advancements in Large Language Models (LLMs) show promise in supporting MDM, single-LLM approaches are limited by their parametric knowledge constraints and static training corpora, failing to robustly integrate the clinical information. To address this challenge, we propose the Expertise-aware Multi-LLM Recruitment and Collaboration (EMRC) framework to enhance the accuracy and reliability of MDM systems. It operates in two stages: (i) expertise-aware agent recruitment and (ii) confidence- and adversarial-driven multi-agent collaboration. Specifically, in the first stage, we use a publicly available corpus to construct an LLM expertise table for capturing expertise-specific strengths of multiple LLMs across medical department categories and query difficulty levels. This table enables the subsequent dynamic selection of the optimal LLMs to act as medical expert agents for each medical query during the inference phase. In the second stage, we employ selected agents to generate responses with self-assessed confidence scores, which are then integrated through the confidence fusion and adversarial validation to improve diagnostic reliability. We evaluate our EMRC framework on three public MDM datasets, where the results demonstrate that our EMRC outperforms state-of-the-art single- and multi-LLM methods, achieving superior diagnostic performance. For instance, on the MMLU-Pro-Health dataset, our EMRC achieves 74.45% accuracy, representing a 2.69% improvement over the best-performing closed-source model GPT- 4-0613, which demonstrates the effectiveness of our expertise-aware agent recruitment strategy and the agent complementarity in leveraging each LLM's specialized capabilities. 

**Abstract (ZH)**: 医学决策制定（MDM）是一个复杂的过程，要求具备大量特定领域的专业知识来有效综合异构和复杂的临床信息。虽然大型语言模型（LLMs）的最新进展显示出支持MDM的潜力，但单一LLM方法受限于其参数知识限制和静态训练语料库，无法稳健地整合临床信息。为了应对这一挑战，我们提出了一种专家意识多LLM招聘与协作（EMRC）框架，以提高MDM系统的准确性和可靠性。该框架分为两个阶段：(i) 专家意识代理招聘和(ii) 基于信心和对抗性的多代理协作。具体而言，在第一阶段，我们使用一个公开可用的语料库构建一个LLM专家表，以捕捉不同医学部门类别和查询难度级别的多个LLM的专业优势。这使得在推理阶段能够动态选择最佳的LLM作为医学专家代理。在第二阶段，我们使用选定的代理生成具有自评估信心分数的响应，然后通过信心融合和对抗验证进行集成，以提高诊断可靠性。我们在三个公开的MDM数据集上评估了EMRC框架，结果表明，我们的EMRC优于最先进的单一LLM和多LLM方法，实现了更出色的诊断性能。例如，在MMLU-Pro-Health数据集上，我们的EMRC达到了74.45%的准确性，比表现最好的闭源模型GPT-4-0613高出2.69%，这表明了我们专家意识代理招聘策略的有效性和代理间的互补性，有助于利用每个LLM的专业能力。 

---
# CausalPlan: Empowering Efficient LLM Multi-Agent Collaboration Through Causality-Driven Planning 

**Title (ZH)**: 因果计划：通过因果驱动规划增强的高效LLM多智能体协作 

**Authors**: Minh Hoang Nguyen, Van Dai Do, Dung Nguyen, Thin Nguyen, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2508.13721)  

**Abstract**: Large language model (LLM) agents-especially smaller, open-source models-often produce causally invalid or incoherent actions in collaborative tasks due to their reliance on surface-level correlations rather than grounded causal reasoning. This limitation undermines their performance in terms of coordination and planning in dynamic environments. We address this challenge with CausalPlan, a two-phase framework that integrates explicit structural causal reasoning into the LLM planning process. At the core of CausalPlan is the Structural Causal Action (SCA) model, which learns a causal graph from agent trajectories to capture how prior actions and current environment states influence future decisions. This structure is then used to guide action selection by assigning causal scores to LLM-generated proposals, reweighting them accordingly, or falling back to causally grounded alternatives when needed. By embedding this causal knowledge directly into the decision loop, CausalPlan constrains planning to intervention-consistent behaviours without requiring fine-tuning of the LLM itself. We evaluate CausalPlan on the Overcooked-AI benchmark across five multi-agent coordination tasks and four LLMs of varying sizes: Gemma-7B, Llama-8B, Qwen-14B, and Llama-70B. Experimental results show that CausalPlan consistently reduces invalid actions and improves collaboration in both AI-AI and human-AI settings, outperforming strong reinforcement learning baselines. Our findings highlight the value of causality-driven planning for deploying efficient, interpretable, and generalisable multi-agent LLM systems. 

**Abstract (ZH)**: 基于因果推理的大型语言模型Planning框架：CausalPlan 

---
# Neuro-Symbolic Artificial Intelligence: Towards Improving the Reasoning Abilities of Large Language Models 

**Title (ZH)**: 神经符号人工智能：向提升大型语言模型的推理能力迈进 

**Authors**: Xiao-Wen Yang, Jie-Jing Shao, Lan-Zhe Guo, Bo-Wen Zhang, Zhi Zhou, Lin-Han Jia, Wang-Zhou Dai, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13678)  

**Abstract**: Large Language Models (LLMs) have shown promising results across various tasks, yet their reasoning capabilities remain a fundamental challenge. Developing AI systems with strong reasoning capabilities is regarded as a crucial milestone in the pursuit of Artificial General Intelligence (AGI) and has garnered considerable attention from both academia and industry. Various techniques have been explored to enhance the reasoning capabilities of LLMs, with neuro-symbolic approaches being a particularly promising way. This paper comprehensively reviews recent developments in neuro-symbolic approaches for enhancing LLM reasoning. We first present a formalization of reasoning tasks and give a brief introduction to the neurosymbolic learning paradigm. Then, we discuss neuro-symbolic methods for improving the reasoning capabilities of LLMs from three perspectives: Symbolic->LLM, LLM->Symbolic, and LLM+Symbolic. Finally, we discuss several key challenges and promising future directions. We have also released a GitHub repository including papers and resources related to this survey: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展现了令人 promising 的成果，但其推理能力仍然是一个基本挑战。增强具有强大推理能力的AI系统被认为是在追求人工通用智能（AGI）过程中的一个重要里程碑，这吸引了学术界和产业界的广泛关注。已经探索了多种方法来提高LLMs的推理能力，其中神经符号方法尤其引人瞩目。本文全面回顾了增强LLM推理能力的神经符号方法的最新进展。首先，我们形式化推理任务并简要介绍了神经符号学习范式。然后，我们从三个角度讨论了提高LLMs推理能力的神经符号方法：符号->LLM，LLM->符号，以及LLM+符号。最后，我们讨论了一些关键挑战和有前景的未来方向。我们还发布了一个GitHub仓库，包括与本文综述相关的论文和资源：this https URL。 

---
# MHSNet:An MoE-based Hierarchical Semantic Representation Network for Accurate Duplicate Resume Detection with Large Language Model 

**Title (ZH)**: MHSNet：基于MoE的分层语义表示网络，用于大规模语言模型辅助的精准简历重复检测 

**Authors**: Yu Li, Zulong Chen, Wenjian Xu, Hong Wen, Yipeng Yu, Man Lung Yiu, Yuyu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.13676)  

**Abstract**: To maintain the company's talent pool, recruiters need to continuously search for resumes from third-party websites (e.g., LinkedIn, Indeed). However, fetched resumes are often incomplete and inaccurate. To improve the quality of third-party resumes and enrich the company's talent pool, it is essential to conduct duplication detection between the fetched resumes and those already in the company's talent pool. Such duplication detection is challenging due to the semantic complexity, structural heterogeneity, and information incompleteness of resume texts. To this end, we propose MHSNet, an multi-level identity verification framework that fine-tunes BGE-M3 using contrastive learning. With the fine-tuned , Mixture-of-Experts (MoE) generates multi-level sparse and dense representations for resumes, enabling the computation of corresponding multi-level semantic similarities. Moreover, the state-aware Mixture-of-Experts (MoE) is employed in MHSNet to handle diverse incomplete resumes. Experimental results verify the effectiveness of MHSNet 

**Abstract (ZH)**: 维护公司的才库，招聘人员需要不断从第三方网站（如LinkedIn、Indeed）搜索简历。然而，获取的简历往往不完整且不准确。为了提高第三方简历的质量并丰富公司的才库，对获取的简历与公司现有才库中的简历进行重复检测是至关重要的。由于简历文本具有语义复杂性、结构异质性和信息不完整性，这种重复检测极具挑战性。为此，我们提出了一种基于多级身份验证框架MHSNet，该框架利用对比学习微调BGE-M3。通过微调后的Mixture-of-Experts (MoE)生成简历的多级稀疏和密集表示，便于计算相应的多级语义相似度。此外，MHSNet中采用了状态感知的Mixture-of-Experts (MoE)以处理多样化的不完整简历。实验结果验证了MHSNet的有效性。 

---
# Toward Better EHR Reasoning in LLMs: Reinforcement Learning with Expert Attention Guidance 

**Title (ZH)**: 面向更好的电子健康记录推理：基于专家关注指导的强化学习 

**Authors**: Yue Fang, Yuxin Guo, Jiaran Gao, Hongxin Ding, Xinke Jiang, Weibin Liao, Yongxin Xu, Yinghao Zhu, Zhibang Yang, Liantao Ma, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13579)  

**Abstract**: Improving large language models (LLMs) for electronic health record (EHR) reasoning is essential for enabling accurate and generalizable clinical predictions. While LLMs excel at medical text understanding, they underperform on EHR-based prediction tasks due to challenges in modeling temporally structured, high-dimensional data. Existing approaches often rely on hybrid paradigms, where LLMs serve merely as frozen prior retrievers while downstream deep learning (DL) models handle prediction, failing to improve the LLM's intrinsic reasoning capacity and inheriting the generalization limitations of DL models. To this end, we propose EAG-RL, a novel two-stage training framework designed to intrinsically enhance LLMs' EHR reasoning ability through expert attention guidance, where expert EHR models refer to task-specific DL models trained on EHR data. Concretely, EAG-RL first constructs high-quality, stepwise reasoning trajectories using expert-guided Monte Carlo Tree Search to effectively initialize the LLM's policy. Then, EAG-RL further optimizes the policy via reinforcement learning by aligning the LLM's attention with clinically salient features identified by expert EHR models. Extensive experiments on two real-world EHR datasets show that EAG-RL improves the intrinsic EHR reasoning ability of LLMs by an average of 14.62%, while also enhancing robustness to feature perturbations and generalization to unseen clinical domains. These results demonstrate the practical potential of EAG-RL for real-world deployment in clinical prediction tasks. Our code have been available at this https URL. 

**Abstract (ZH)**: 提高大型语言模型在电子健康记录推理中的性能对于实现准确且可泛化的临床预测至关重要。虽然大型语言模型在医学文本理解方面表现出色，但在基于电子健康记录的预测任务中表现不佳，主要原因是难以建模具有时间结构的高维度数据。现有方法通常依赖于混合范式，其中大型语言模型仅作为冻结的先验检索器，而下游深度学习模型处理预测任务，这不仅未能提高大型语言模型的内在推理能力，还继承了深度学习模型的泛化限制。为此，我们提出了一种名为EAG-RL的新型两阶段训练框架，通过专家注意力引导内在增强大型语言模型的电子健康记录推理能力，其中专家电子健康记录模型指的是专门针对电子健康记录数据训练的任务特定深度学习模型。具体来说，EAG-RL首先利用专家引导的蒙特卡洛树搜索构造高质量、逐步的推理轨迹，以有效初始化大型语言模型的策略。然后，EAG-RL通过强化学习进一步优化策略，通过将大型语言模型的注意力与专家电子健康记录模型识别的临床相关特征对齐来实现。在两个真实世界的电子健康记录数据集上的广泛实验显示，EAG-RL平均提升了14.62%的内在电子健康记录推理能力，同时增强了对特征扰动的鲁棒性和对未见过的临床领域的泛化能力。这些结果表明，EAG-RL在临床预测任务中的实际部署具有实用潜力。我们的代码已在此处提供：this https URL 

---
# LM Agents May Fail to Act on Their Own Risk Knowledge 

**Title (ZH)**: LM代理可能会忽视其自身的风险知识。 

**Authors**: Yuzhi Tang, Tianxiao Li, Elizabeth Li, Chris J. Maddison, Honghua Dong, Yangjun Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13465)  

**Abstract**: Language model (LM) agents have demonstrated significant potential for automating real-world tasks, yet they pose a diverse array of potential, severe risks in safety-critical scenarios. In this work, we identify a significant gap between LM agents' risk awareness and safety execution abilities: while they often answer "Yes" to queries like "Is executing `sudo rm -rf /*' dangerous?", they will likely fail to identify such risks in instantiated trajectories or even directly perform these risky actions when acting as agents. To systematically investigate this, we develop a comprehensive evaluation framework to examine agents' safety across three progressive dimensions: 1) their knowledge about potential risks, 2) their ability to identify corresponding risks in execution trajectories, and 3) their actual behaviors to avoid executing these risky actions. Our evaluation reveals two critical performance gaps that resemble the generator-validator gaps observed in LMs: while agents demonstrate near-perfect risk knowledge ($>98\%$ pass rates), they fail to apply this knowledge when identifying risks in actual scenarios (with performance dropping by $>23\%$) and often still execute risky actions ($<26\%$ pass rates). Notably, this trend persists across more capable LMs as well as in specialized reasoning models like DeepSeek-R1, indicating that simply scaling model capabilities or inference compute does not inherently resolve safety concerns. Instead, we take advantage of these observed gaps to develop a risk verifier that independently critiques the proposed actions by agents, with an abstractor that converts specific execution trajectories into abstract descriptions where LMs can more effectively identify the risks. Our overall system achieves a significant reduction of risky action execution by $55.3\%$ over vanilla-prompted agents. 

**Abstract (ZH)**: 语言模型（LM）代理展现了在自动化现实世界任务方面的巨大潜力，但在安全关键场景中也带来了多样而严重的风险。本文识别了LM代理风险管理意识与其安全执行能力之间的重要差距：尽管它们经常对诸如“执行 `sudo rm -rf /*' 危险吗？”这类查询回答“是”，但在具体执行轨迹中识别这些风险的能力却很可能不足，甚至会在作为代理行动时直接执行这类危险操作。为系统地研究这一问题，我们开发了一个综合评估框架，从三个逐渐进化的维度来考察代理的安全性：1）它们对潜在风险的知识；2）在执行轨迹中识别相应风险的能力；3）避免执行这些危险操作的实际行为。我们的评估揭示了两类关键性能差距，类似于LM中的生成器-验证器差距：代理在风险知识方面表现出几乎完美（通过率超过98%）的能力，但在实际场景中识别风险时性能却骤降超过23%，并且仍然经常执行危险操作（通过率低于26%）。值得注意的是，这种趋势在更强大的LM以及专门的推理模型DeepSeek-R1中同样存在，表明仅仅扩展模型能力和推理计算并未能从根本上解决安全问题。基于这些观察到的差距，我们开发了一个风险验证器，其独立地批判代理提出的操作，并通过一个抽象器将具体的执行轨迹转换为能够更有效地识别风险的抽象描述。我们整体系统的危险操作执行率相比纯提示的代理下降了55.3%。 

---
# HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design 

**Title (ZH)**: 基于 hindsight 和 foresight 的 LLM 基automatic heuristic 设计提示方法 

**Authors**: Chentong Chen, Mengyuan Zhong, Jianyong Sun, Ye Fan, Jialong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.13333)  

**Abstract**: LLM-based Automatic Heuristic Design (AHD) within Evolutionary Computation (EC) frameworks has shown promising results. However, its effectiveness is hindered by the use of static operators and the lack of knowledge accumulation mechanisms. We introduce HiFo-Prompt, a framework that guides LLMs with two synergistic prompting strategies: Foresight and Hindsight. Foresight-based prompts adaptively steer the search based on population dynamics, managing the exploration-exploitation trade-off. In addition, hindsight-based prompts mimic human expertise by distilling successful heuristics from past generations into fundamental, reusable design principles. This dual mechanism transforms transient discoveries into a persistent knowledge base, enabling the LLM to learn from its own experience. Empirical results demonstrate that HiFo-Prompt significantly outperforms state-of-the-art LLM-based AHD methods, generating higher-quality heuristics while achieving substantially faster convergence and superior query efficiency. 

**Abstract (ZH)**: 基于LLM的进化计算中自动启发式设计（AHD）前景与回顾双重引导框架（HiFo-Prompt）已显示出有前景的结果，然而其效果受限于静态操作符的使用及缺乏知识积累机制。我们引入了HiFo-Prompt框架，该框架通过前景和回顾两种协同的提示策略来指导LLM：前景策略根据群体动态自适应地引导搜索，管理探索与利用之间的权衡；此外，回顾策略通过从过去世代中提炼成功的启发式方法以形成基础且可重用的设计原则来模仿人类专家。这种双重机制将临时发现转化为持久的知识库，从而使LLM能够从自身经验中学习。实证结果表明，HiFo-Prompt显著优于最先进的基于LLM的AHD方法，生成更高质量的启发式方法，同时实现更快的收敛速度和更优良的查询效率。 

---
# Search-Time Data Contamination 

**Title (ZH)**: 搜索时数据污染 

**Authors**: Ziwen Han, Meher Mankikar, Julian Michael, Zifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13180)  

**Abstract**: Data contamination refers to the leakage of evaluation data into model training data, resulting in overfitting to supposedly held-out test sets and compromising test validity. We identify an analogous issue, search-time contamination (STC), in evaluating search-based LLM agents which use tools to gather information from online sources when answering user queries. STC occurs when the retrieval step surfaces a source containing the test question (or a near-duplicate) alongside its answer, enabling agents to copy rather than genuinely infer or reason, undermining benchmark integrity. We find that HuggingFace, an online platform hosting evaluation datasets, appears among retrieved sources in search based agent logs. Consequently, agents often explicitly acknowledge discovering question answer pairs from HuggingFace within their reasoning chains. On three commonly used capability benchmarks: Humanity's Last Exam (HLE), SimpleQA, and GPQA, we demonstrate that for approximately 3% of questions, search-based agents directly find the datasets with ground truth labels on HuggingFace. When millions of evaluation queries target the same benchmark, even small, repeated leaks can accelerate the benchmark's obsolescence, shortening its intended lifecycle. After HuggingFace is blocked, we observe a drop in accuracy on the contaminated subset of approximately 15%. We further show through ablation experiments that publicly accessible evaluation datasets on HuggingFace may not be the sole source of STC. To this end, we conclude by proposing best practices for benchmark design and result reporting to address this novel form of leakage and ensure trustworthy evaluation of search-based LLM agents. To facilitate the auditing of evaluation results, we also publicly release the complete logs from our experiments. 

**Abstract (ZH)**: 数据污染是指评估数据泄露到模型训练数据中，导致模型过度拟合本应保留的测试集，影响测试的有效性。我们识别了一个类似的问题——搜索时污染（STC），在基于搜索的大型语言模型（LLM）代理评估中，这些代理在回答用户查询时从在线源获取信息。当检索步骤返回包含测试问题（或近乎重复问题）及其答案的来源时，代理可以抄袭而不是真正地推理或推断，从而损害基准的有效性。我们发现，作为一个托管评估数据集的在线平台，HuggingFace经常出现在基于搜索代理的日志中检索到的来源列表中。因此，代理往往在其推理链中明确承认从HuggingFace发现问题答案对。在三个常用的能力基准测试中——人类的最后一考（HLE）、SimpleQA和GPQA——我们证明，大约有3%的问题，基于搜索的代理可以直接找到包含真实标签的数据集。当数百万的评估查询针对同一个基准时，即使是小规模的重复泄露也会加速基准的老化，缩短其预期的生命周期。在封锁HuggingFace后，我们观察到受影响子集的准确性下降约15%。通过消融实验进一步表明，公开可供访问的评估数据集可能不是STC的唯一来源。为此，我们提出最佳实践以设计基准和报告结果，以应对这种新颖的泄露形式，确保基于搜索的LLM代理评估的可信度。我们也公开发布了实验的完整日志以助于评估结果的审核。 

---
# Cognitive Workspace: Active Memory Management for LLMs -- An Empirical Study of Functional Infinite Context 

**Title (ZH)**: 认知工作区：针对LLMs的主动内存管理——功能无限上下文的实证研究 

**Authors**: Tao An  

**Link**: [PDF](https://arxiv.org/pdf/2508.13171)  

**Abstract**: Large Language Models (LLMs) face fundamental limitations in context management despite recent advances extending context windows to millions of tokens. We propose Cognitive Workspace, a novel paradigm that transcends traditional Retrieval-Augmented Generation (RAG) by emulating human cognitive mechanisms of external memory use. Drawing from cognitive science foundations including Baddeley's working memory model, Clark's extended mind thesis, and Hutchins' distributed cognition framework, we demonstrate that current passive retrieval systems fail to capture the dynamic, task-driven nature of human memory management. Our analysis of 2024-2025 developments reveals that while techniques like Infini-attention and StreamingLLM achieve impressive context lengths, they lack the metacognitive awareness and active planning capabilities essential for true cognitive extension. Cognitive Workspace addresses these limitations through three core innovations: (1) active memory management with deliberate information curation, (2) hierarchical cognitive buffers enabling persistent working states, and (3) task-driven context optimization that dynamically adapts to cognitive demands. Empirical validation demonstrates Cognitive Workspace achieves an average 58.6% memory reuse rate (ranging from 54-60% across different tasks) compared to 0% for traditional RAG, with 17-18% net efficiency gain despite 3.3x higher operation counts. Statistical analysis confirms these advantages with p < 0.001 and Cohen's d > 23 across multiple task types, establishing the first quantitative evidence for active memory superiority in LLM systems. We present a comprehensive theoretical framework synthesizing insights from 50+ recent papers, positioning Cognitive Workspace as a fundamental shift from information retrieval to genuine cognitive augmentation. 

**Abstract (ZH)**: 大型语言模型在上下文管理方面存在根本局限，尽管近期技术进步将上下文窗口扩展至数百万个词元。我们提出认知工作空间，这一新颖范式超越了传统的检索增强生成（RAG），通过模拟人类的认知机制来使用外部记忆。基于认知科学的基础，包括巴德利的工作记忆模型、克拉克的扩展心智论以及赫钦斯的分布式认知框架，我们证明当前的被动检索系统无法捕捉人类记忆管理的动态任务驱动特性。对2024-2025年发展趋势的分析显示，虽然无限注意力和StreamingLLM等技术实现了令人印象深刻的上下文长度，但缺乏必要的元认知意识和主动规划能力，这对于真正的认知扩展是必需的。认知工作空间通过三项核心创新来解决这些局限：（1）主动的信息管理和有目的地进行信息筛选，（2）分层次的认知缓存以支持持久的工作状态，以及（3）以任务驱动的方式优化上下文，能够动态适应认知需求。实证验证表明，认知工作空间在不同任务中平均实现了58.6%的记忆重用率（范围为54%-60%），比传统RAG高出0%，并且在操作次数增加3.3倍的情况下获得了17%-18%的净效率提升。统计分析证实了这些优势，p值小于0.001，Cohen’s d大于23，为语言模型系统中主动记忆的优势提供了首个定量证据。我们综合了50多篇最近论文的见解，将认知工作空间置于从信息检索到真正认知增强的基本转变之中。 

---
# Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation 

**Title (ZH)**: 代理微调引起的无意对齐风险与缓解 

**Authors**: Dongyoon Hahm, Taywon Min, Woogyeol Jin, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14031)  

**Abstract**: Beyond simple text generation, Large Language Models (LLMs) have evolved into agentic systems capable of planning and interacting with external tools to solve complex tasks. This evolution involves fine-tuning LLMs on agent-specific tasks to enhance their proficiency. However, safety concerns are frequently overlooked during this fine-tuning process. In this work, we show that aligned LLMs can become unintentionally misaligned, leading to a higher likelihood of executing harmful tasks and a reduced tendency to refuse them when fine-tuned to execute agentic tasks. To address these safety challenges, we propose Prefix INjection Guard (PING), a simple yet effective method that prepends automatically generated natural language prefixes to agent responses, guiding them to refuse harmful requests while preserving performance on benign tasks. Specifically, we introduce an iterative approach that alternates between (1) generating candidate prefixes and (2) selecting those that optimize both task performance and refusal behavior. Experimental results demonstrate that PING significantly enhances the safety of fine-tuned LLM agents without sacrificing their effectiveness. PING consistently outperforms existing prompting approaches across diverse benchmarks in both web navigation and code generation tasks. Our analysis of internal hidden states via linear probes reveals that prefix tokens are crucial for behavior modification, explaining the performance gains. WARNING: This paper contains contents that are unethical or offensive in nature. 

**Abstract (ZH)**: 超越简单的文本生成，大规模语言模型（LLMs）已经发展成为能够规划和与外部工具互动以解决复杂任务的代理系统。这一发展涉及在特定代理任务上微调LLMs以提高其专业能力。然而，在这个微调过程中，安全问题往往会被忽视。在本文中，我们展示了对齐的语言模型可能会无意中失去对齐，导致在执行代理任务时更有可能执行有害任务，并且拒绝这些任务的倾向降低。为了应对这些安全挑战，我们提出了前缀注入守护（PING）方法，这是一种简单而有效的方法，通过在代理响应前自动生成的自然语言前缀中添加，引导其拒绝有害请求，同时保留其在良性任务上的性能。具体而言，我们介绍了一种迭代方法，交替进行（1）生成候选前缀和（2）选择优化了任务性能和拒绝行为的前缀。实验结果表明，PING显著提高了微调后语言模型代理的安全性，而不会牺牲其效果。PING在网页导航和代码生成等多样基准测试中始终优于现有提示方法。通过线性探针分析内部隐藏状态显示，前缀标记对于行为调整至关重要，解释了性能提升的原因。警告：本文包含不道德或冒犯性内容。 

---
# Ask Good Questions for Large Language Models 

**Title (ZH)**: 为大型语言模型提出好问题 

**Authors**: Qi Wu, Zhongqi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14025)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved the performance of dialog systems, yet current approaches often fail to provide accurate guidance of topic due to their inability to discern user confusion in related concepts. To address this, we introduce the Ask-Good-Question (AGQ) framework, which features an improved Concept-Enhanced Item Response Theory (CEIRT) model to better identify users' knowledge levels. Our contributions include applying the CEIRT model along with LLMs to directly generate guiding questions based on the inspiring text, greatly improving information retrieval efficiency during the question & answer process. Through comparisons with other baseline methods, our approach outperforms by significantly enhencing the users' information retrieval experiences. 

**Abstract (ZH)**: 近期大规模语言模型的进展显著提升了对话系统的性能，但当前方法往往因无法分辨用户在相关概念上的混淆而无法提供准确的引导。为此，我们提出了Ask-Good-Question (AGQ)框架，该框架配备改进的概念增强项目反应理论（CEIRT）模型，以更好地识别用户的知识水平。我们的贡献包括将CEIRT模型与大规模语言模型结合，直接根据激发性文本生成引导问题，大幅提高问答过程中的信息检索效率。与其它基线方法相比，我们的方法通过显著增强用户的检索体验表现出色。 

---
# Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context LLM Preference Optimization 

**Title (ZH)**: 块作为臂：多臂 bandit 引导的采样方法用于长期上下文 LLM 偏好优化 

**Authors**: Shaohua Duan, Xinze Li, Zhenghao Liu, Xiaoyuan Yi, Yukun Yan, Shuo Wang, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13993)  

**Abstract**: Long-context modeling is critical for a wide range of real-world tasks, including long-context question answering, summarization, and complex reasoning tasks. Recent studies have explored fine-tuning Large Language Models (LLMs) with synthetic data to enhance their long-context capabilities. However, the effectiveness of such approaches is often limited by the low diversity and factual inconsistencies in the generated data. To address these challenges, we propose LongMab-PO, a novel framework that leverages a Multi-Armed Bandit (MAB) rollout strategy to identify the most informative chunks from the given long context for sampling high-quality and diverse responses and constructing preference data pairs for Direct Preference Optimization (DPO) training. Specifically, we treat context chunks as arms of MAB, select chunks based on their expected reward scores to input into LLMs to generate responses, and iteratively update these scores based on reward feedback. This exploration and exploitation process enables the model to focus on the most relevant context segments, thereby generating and collecting high-quality and diverse responses. Finally, we collect these generated responses from the rollout process and apply the DPO method to further optimize the LLM. Experimental results show that LongMab-PO significantly improves the diversity and quality of preference data pairs, achieving state-of-the-art performance on long-context reasoning benchmarks. All code and data will be released on this https URL. 

**Abstract (ZH)**: 长上下文建模对于长上下文问答、总结和复杂推理等广泛的实际任务至关重要。近期研究探索了使用合成数据微调大型语言模型（LLMs）以提高其长上下文能力。然而，这种方法的有效性往往受限于生成数据的低多样性和事实不一致性。为解决这些挑战，我们提出了一种新的框架LongMab-PO，该框架利用多臂-bandit（MAB） rollout 策略从给定的长上下文中识别最具信息量的片段，以生成高质量和多样化的响应，并构建偏好数据对用于直接偏好优化（DPO）训练。具体而言，我们将上下文片段视为MAB的臂，根据预期奖励分值选择片段输入LLM生成响应，并根据奖励反馈迭代更新这些分值。这种探索和利用过程使模型能够专注于最相关的上下文段落，从而生成和收集高质量和多样化的响应。最终，我们从rollout过程中收集这些生成的响应，并应用DPO方法进一步优化LLM。实验结果表明，LongMab-PO显著提高了偏好数据对的多样性和质量，在长上下文推理基准测试上达到了最先进的性能。所有代码和数据将在该网址发布：https://。 

---
# RotBench: Evaluating Multimodal Large Language Models on Identifying Image Rotation 

**Title (ZH)**: RotBench: 评估多模态大型语言模型在识别图像旋转方面的性能 

**Authors**: Tianyi Niu, Jaemin Cho, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2508.13968)  

**Abstract**: We investigate to what extent Multimodal Large Language Models (MLLMs) can accurately identify the orientation of input images rotated 0°, 90°, 180°, and 270°. This task demands robust visual reasoning capabilities to detect rotational cues and contextualize spatial relationships within images, regardless of their orientation. To evaluate MLLMs on these abilities, we introduce RotBench -- a 350-image manually-filtered benchmark comprising lifestyle, portrait, and landscape images. Despite the relatively simple nature of this task, we show that several state-of-the-art open and proprietary MLLMs, including GPT-5, o3, and Gemini-2.5-Pro, do not reliably identify rotation in input images. Providing models with auxiliary information -- including captions, depth maps, and more -- or using chain-of-thought prompting offers only small and inconsistent improvements. Our results indicate that most models are able to reliably identify right-side-up (0°) images, while certain models are able to identify upside-down (180°) images. None can reliably distinguish between 90° and 270°. Simultaneously showing the image rotated in different orientations leads to moderate performance gains for reasoning models, while a modified setup using voting improves the performance of weaker models. We further show that fine-tuning does not improve models' ability to distinguish 90° and 270° rotations, despite substantially improving the identification of 180° images. Together, these results reveal a significant gap between MLLMs' spatial reasoning capabilities and human perception in identifying rotation. 

**Abstract (ZH)**: 我们调查了多模态大型语言模型（MLLMs）在识别输入图像旋转0°、90°、180°和270°的方向方面能够达到的准确程度。这项任务要求模型具备 robust 视觉推理能力，以检测旋转线索并理解图像中的空间关系，而不受其旋转方向的影响。为了评估 MLLMs 的这些能力，我们引入了 RotBench——一个包含350张手工筛选的生活照、肖像和风景图像的基准测试集。尽管这项任务相对简单，但我们发现，包括 GPT-5、o3 和 Gemini-2.5-Pro 在内的多种最先进的开箱即用和专有 MLLMs，并不能可靠地识别输入图像的旋转。提供辅助信息，包括字幕、深度图等，或者使用思维链提示，仅能带来有限且不一致的改进。我们的结果显示，大多数模型能够可靠地识别正向（0°）图像，而某些模型能够识别倒向（180°）图像。没有模型能够可靠地区分90°和270°。同时展示图像在不同旋转方向下则对推理模型的性能有所提升，而修改后的投票机制则增强了较弱模型的表现。此外，我们还发现，尽管在识别180°图像方面显著提高，但微调并不能提升模型区分90°和270°旋转的能力。这些结果揭示了MLLMs在识别旋转方面的空间推理能力与人类感知之间存在显著差距。 

---
# Learning to Use AI for Learning: How Can We Effectively Teach and Measure Prompting Literacy for K-12 Students? 

**Title (ZH)**: 学习使用AI进行学习：我们如何有效地教授和衡量K-12学生的问题提示素养？ 

**Authors**: Ruiwei Xiao, Xinying Hou, Ying-Jui Tseng, Hsuan Nieu, Guanze Liao, John Stamper, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2508.13962)  

**Abstract**: As Artificial Intelligence (AI) becomes increasingly integrated into daily life, there is a growing need to equip the next generation with the ability to apply, interact with, evaluate, and collaborate with AI systems responsibly. Prior research highlights the urgent demand from K-12 educators to teach students the ethical and effective use of AI for learning. To address this need, we designed an Large-Language Model (LLM)-based module to teach prompting literacy. This includes scenario-based deliberate practice activities with direct interaction with intelligent LLM agents, aiming to foster secondary school students' responsible engagement with AI chatbots. We conducted two iterations of classroom deployment in 11 authentic secondary education classrooms, and evaluated 1) AI-based auto-grader's capability; 2) students' prompting performance and confidence changes towards using AI for learning; and 3) the quality of learning and assessment materials. Results indicated that the AI-based auto-grader could grade student-written prompts with satisfactory quality. In addition, the instructional materials supported students in improving their prompting skills through practice and led to positive shifts in their perceptions of using AI for learning. Furthermore, data from Study 1 informed assessment revisions in Study 2. Analyses of item difficulty and discrimination in Study 2 showed that True/False and open-ended questions could measure prompting literacy more effectively than multiple-choice questions for our target learners. These promising outcomes highlight the potential for broader deployment and highlight the need for broader studies to assess learning effectiveness and assessment design. 

**Abstract (ZH)**: 随着人工智能（AI）越来越多地融入日常生活，培养下一代具备负责任地应用、互动、评估和协作使用AI系统的能力显得越来越重要。前期研究强调了K-12教育者对于教授学生如何以道德和有效的方式使用AI进行学习的迫切需求。为应对这一需求，我们设计了一个基于大型语言模型（LLM）的主题模块来教授提示素养。该模块包括基于场景的刻意练习活动，直接与智能LLM代理互动，旨在促进中学生负责任地与AI聊天机器人互动。我们在11个真实的中等教育班级中进行了两次课堂教学部署，并评估了1) AI基于的自动评分器的能力；2) 学生使用AI进行学习的提示表现及其自信心的变化；以及3) 学习和评估材料的质量。结果表明，基于AI的自动评分器能够以满意的质量对学生撰写的提示进行评分。此外，教学材料通过练习支持学生提高他们的提示技能，并对使用AI进行学习的看法产生了积极的影响。此外，研究1的数据在研究2的评估修订中提供了信息。研究2中项目难度和区分度的分析显示，对于我们的目标学习者，真伪判断和开放性问题比选择题更适合衡量提示素养。这些令人鼓舞的结果强调了更广泛部署的可能性，并突显了需要进行更广泛的研究来评估学习效果和评估设计。 

---
# Prompt Orchestration Markup Language 

**Title (ZH)**: 提示编排标记语言 

**Authors**: Yuge Zhang, Nan Chen, Jiahang Xu, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13948)  

**Abstract**: Large Language Models (LLMs) require sophisticated prompting, yet current practices face challenges in structure, data integration, format sensitivity, and tooling. Existing methods lack comprehensive solutions for organizing complex prompts involving diverse data types (documents, tables, images) or managing presentation variations systematically. To address these gaps, we introduce POML (Prompt Orchestration Markup Language). POML employs component-based markup for logical structure (roles, tasks, examples), specialized tags for seamless data integration, and a CSS-like styling system to decouple content from presentation, reducing formatting sensitivity. It includes templating for dynamic prompts and a comprehensive developer toolkit (IDE support, SDKs) to improve version control and collaboration. We validate POML through two case studies demonstrating its impact on complex application integration (PomLink) and accuracy performance (TableQA), as well as a user study assessing its effectiveness in real-world development scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）需要复杂的提示，现有实践在结构、数据整合、格式敏感性和工具方面面临挑战。现有方法缺乏全面解决涉及多种数据类型（文档、表格、图像）的复杂提示组织或系统化管理呈现变体的解决方案。为解决这些缺口，我们引入了POML（Prompt Orchestration Markup Language）。POML采用基于组件的标记来定义逻辑结构（角色、任务、示例），专门的标签用于无缝数据整合，并采用类似CSS的样式系统来解耦内容与呈现，降低格式敏感性。它包含动态提示的模板功能，并提供全面的开发工具包（IDE支持、SDKs）以改进版本控制和协作。我们通过两个案例研究验证了POML在复杂应用集成（PomLink）和准确性性能（TableQA）方面的成效，并通过用户研究评估了其在实际开发场景中的有效性。 

---
# InPars+: Supercharging Synthetic Data Generation for Information Retrieval Systems 

**Title (ZH)**: InPars+: 用于信息检索系统的合成数据生成加速方法 

**Authors**: Matey Krastev, Miklos Hamar, Danilo Toapanta, Jesse Brouwers, Yibin Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.13930)  

**Abstract**: This work revisits and extends synthetic query generation pipelines for Neural Information Retrieval (NIR) by leveraging the InPars Toolkit, a reproducible, end-to-end framework for generating training data using large language models (LLMs). We first assess the reproducibility of the original InPars, InPars-V2, and Promptagator pipelines on the SciFact benchmark and validate their effectiveness using open-source reranker and generator models. Building on this foundation, we introduce two key extensions to the pipeline: (1) fine-tuning a query generator LLM via Contrastive Preference Optimization (CPO) to improve the signal quality in generated queries, and (2) replacing static prompt templates with dynamic, Chain-of-Thought (CoT) optimized prompts using the DSPy framework. Our results show that both extensions reduce the need for aggressive filtering while improving retrieval performance. All code, models, and synthetic datasets are publicly released to support further research at: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 本研究利用InPars Toolkit（一个可重现的端到端框架，用于使用大型语言模型生成训练数据）重新审视并扩展了神经信息检索（NIR）的合成查询生成管道。我们首先在SciFact基准上评估了原始InPars、InPars-V2和Promptagator管道的可重现性，并使用开源重排器和生成模型验证了它们的有效性。在此基础上，我们引入了管道的两个关键扩展：（1）通过对比偏好优化（CPO）微调查询生成大型语言模型，以提高生成查询的质量信号；（2）使用DSPy框架将静态提示模板替换为动态的、基于推理链（CoT）优化的提示。实验结果表明，这两种扩展在降低剧烈筛选需求的同时提升了检索性能。所有代码、模型和合成数据集均已在以下链接公开发布：\href{this https URL}{this https URL}。 

---
# Prompt-Based One-Shot Exact Length-Controlled Generation with LLMs 

**Title (ZH)**: 基于提示的一次生成精确长度控制生成ewith大语言模型 

**Authors**: Juncheng Xie, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13805)  

**Abstract**: Controlling the length of text produced by large language models (LLMs) remains challenging: models frequently overshoot or undershoot explicit length instructions because they cannot reliably keep an internal token count. We present a prompt-based, one-shot strategy that compels an off-the-shelf LLM to generate exactly a desired number of tokens - words (English) or characters (Chinese) - without any fine-tuning or iterative sampling. The prompt appends countdown markers and explicit counting rules so that the model "writes while counting." We evaluate on four settings: open-ended generation (1-1000 tokens), XSUM summarization, MT-Bench-LI instruction following, and the LIFEBENCH equal-length track. On MT-Bench-LI, strict length compliance with GPT-4.1 leaps from below 30% under naive prompts to above 95% with our countdown prompt, surpassing the popular draft-then-revise baseline, while judged answer quality is preserved. These results show that precise length control can be achieved through prompt engineering alone, offering a lightweight alternative to training- or decoding-based methods. 

**Abstract (ZH)**: 控制大型语言模型生成文本的长度仍然具有挑战性：模型经常无法可靠地保持内部令牌计数，从而导致长度过度或不足。我们提出了一种基于提示的一次性策略，使即用型大语言模型生成恰好所需的令牌数（单词或字符）数量，无需任何微调或迭代采样。该提示附加了倒计时标记和明确的计数规则，使模型“边写边计数”。我们在四种设置下进行了评估：开放式生成（1-1000个令牌）、XSUM摘要、MT-Bench-LI指令跟随以及LIFEBENCH等长度赛道。在MT-Bench-LI上，使用标准化提示时对GPT-4.1的严格长度合规率低于30%，而使用我们的倒计时提示则提升至超过95%，超过了流行的草拟后再修订baseline方法，同时保持了判断答案质量。这些结果表明，仅通过提示工程就可以实现精确的长度控制，提供了一种轻量级的替代训练或解码方法。 

---
# Agentic DraCor and the Art of Docstring Engineering: Evaluating MCP-empowered LLM Usage of the DraCor API 

**Title (ZH)**: 代理DraCor和文档字符串工程的艺术：评估MCP赋能的LLM对DraCor API的使用 

**Authors**: Peer Trilcke, Ingo Börner, Henny Sluyter-Gäthje, Daniil Skorinkin, Frank Fischer, Carsten Milling  

**Link**: [PDF](https://arxiv.org/pdf/2508.13774)  

**Abstract**: This paper reports on the implementation and evaluation of a Model Context Protocol (MCP) server for DraCor, enabling Large Language Models (LLM) to autonomously interact with the DraCor API. We conducted experiments focusing on tool selection and application by the LLM, employing a qualitative approach that includes systematic observation of prompts to understand how LLMs behave when using MCP tools, evaluating "Tool Correctness", "Tool-Calling Efficiency", and "Tool-Use Reliability". Our findings highlight the importance of "Docstring Engineering", defined as reflexively crafting tool documentation to optimize LLM-tool interaction. Our experiments demonstrate both the promise of agentic AI for research in Computational Literary Studies and the essential infrastructure development needs for reliable Digital Humanities infrastructures. 

**Abstract (ZH)**: 本研究报道了为DraCor实现并评估Model Context Protocol (MCP) 服务器，使大型语言模型（LLM）能够自主与DraCor API交互。我们通过定性的方法开展了实验，包括系统地观察提示，以了解LLM在使用MCP工具时的行为，评估“工具正确性”、“工具调用效率”和“工具使用可靠性”。研究结果强调了“文档字符串工程”的重要性，即反射性地设计工具文档以优化LLM与工具的交互。实验表明，自主智能在计算文学研究中的应用前景，并指出了可靠数字人文基础设施所需的基础架构开发需求。 

---
# COMPASS: A Multi-Dimensional Benchmark for Evaluating Code Generation in Large Language Models 

**Title (ZH)**: COMPASS: 一个评估大型语言模型代码生成多维度基准 

**Authors**: James Meaden, Michał Jarosz, Piotr Jodłowski, Grigori Melnik  

**Link**: [PDF](https://arxiv.org/pdf/2508.13757)  

**Abstract**: Current code generation benchmarks focus primarily on functional correctness while overlooking two critical aspects of real-world programming: algorithmic efficiency and code quality. We introduce COMPASS (COdility's Multi-dimensional Programming ASSessment), a comprehensive evaluation framework that assesses code generation across three dimensions: correctness, efficiency, and quality. COMPASS consists of 50 competitive programming problems from real Codility competitions, providing authentic human baselines from 393,150 submissions. Unlike existing benchmarks that treat algorithmically inefficient solutions identically to optimal ones provided they pass test cases, COMPASS systematically evaluates runtime efficiency and code quality using industry-standard analysis tools. Our evaluation of three leading reasoning-enhanced models, Anthropic Claude Opus 4, Google Gemini 2.5 Pro, and OpenAI O4-Mini-High, reveals that models achieving high correctness scores do not necessarily produce efficient algorithms or maintainable code. These findings highlight the importance of evaluating more than just correctness to truly understand the real-world capabilities of code generation models. COMPASS serves as a guiding framework, charting a path for future research toward AI systems that are robust, reliable, and ready for production use. 

**Abstract (ZH)**: COdility的多维度编程评估 COMPASS：综合评估框架 

---
# Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with Adaptive Exploration 

**Title (ZH)**: 深度与广度协同在RLVR中：自适应探索解锁预训练语言模型推理优势 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Dongchun Xie, Yiwei Wang, Xiaodan Liang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13755)  

**Abstract**: Reinforcement Learning with Verifiable Reward (RLVR) has emerged as a powerful paradigm for unlocking reasoning capabilities in large language models, yet its full potential is hindered by two under-explored dimensions: Depth-the hardest problem a model can sample; Breadth-the number of instances consumed in a single iteration. We dissect the popular GRPO algorithm and reveal a systematic bias: the cumulative-advantage disproportionately weights samples with medium accuracy, while down-weighting the low-accuracy instances that are crucial for pushing reasoning boundaries. To rectify the depth neglect, we introduce Difficulty Adaptive Rollout Sampling (DARS), which re-weights hard problems through targeted multi-stage rollouts, thereby increasing the number of positive rollouts for hard problems. Empirically, naively enlarging rollout size only accelerates convergence and even hurts Pass@K. Our DARS, in contrast, delivers consistent Pass@K gains without extra inference cost at convergence. Just as we adaptively expanded the depth of exploration, we now ask whether aggressively scaling the breadth of training data can further amplify reasoning gains. To this end, we intensely scale batch size and replace PPO's mini-batch iterations with full-batch updates over multiple epochs. Increasing breadth significantly enhances Pass@1 performance. Large-breadth training sustains high token-level entropy, indicating continued exploration and reduced gradient noise. We further present DARS-B, which augments DARS with large breadth, and demonstrate simultaneous gains in Pass@K and Pass@1. The results confirm that breadth and adaptive exploration across depth operate as orthogonal dimensions in RLVR, which are key to unleashing the reasoning power of RLVR. 

**Abstract (ZH)**: 验证奖励的增强学习（RLVR）：通过深度和广度解锁推理能力 

---
# Mitigating Cross-Image Information Leakage in LVLMs for Multi-Image Tasks 

**Title (ZH)**: 在多图像任务中缓解LVLMs之间的跨图像信息泄露 

**Authors**: Yeji Park, Minyoung Lee, Sanghyuk Chun, Junsuk Choe  

**Link**: [PDF](https://arxiv.org/pdf/2508.13744)  

**Abstract**: Large Vision-Language Models (LVLMs) demonstrate strong performance on single-image tasks. However, we observe that their performance degrades significantly when handling multi-image inputs. This occurs because visual cues from different images become entangled in the model's output. We refer to this phenomenon as cross-image information leakage. To address this issue, we propose FOCUS, a training-free and architecture-agnostic decoding strategy that mitigates cross-image information leakage during inference. FOCUS sequentially masks all but one image with random noise, guiding the model to focus on the single clean image. We repeat this process across all target images to obtain logits under partially masked contexts. These logits are aggregated and then contrastively refined using a noise-only reference input, which suppresses the leakage and yields more accurate outputs. FOCUS consistently improves performance across four multi-image benchmarks and diverse LVLM families. This demonstrates that FOCUS offers a general and practical solution for enhancing multi-image reasoning without additional training or architectural modifications. 

**Abstract (ZH)**: 大型视觉-语言模型（LVLMs）在单图任务上表现出色。然而，我们发现它们在处理多图输入时性能大幅下降。这是因为模型输出中包含了不同图像间的视觉线索纠缠现象。我们将这种现象称为跨图信息泄露。为了解决这一问题，我们提出了一种无需训练且架构无关的解码策略FOCUS，该策略在推理过程中减轻了跨图信息泄露。FOCUS通过用随机噪声掩蔽所有但一张干净图像的方式，引导模型专注于单一干净图像。我们对所有目标图像重复此过程，在部分掩蔽的上下文中获取逻辑值，这些逻辑值随后通过仅使用噪声参考输入进行对比性精炼，以抑制泄露并获得更准确的输出。FOCUS在四个多图基准和多种LVLM家族中一致提升了性能。这表明FOCUS提供了一种通用且实用的解决方案，可以在无额外训练或架构修改的情况下增强多图推理。 

---
# Prediction is not Explanation: Revisiting the Explanatory Capacity of Mapping Embeddings 

**Title (ZH)**: 预测不等同于解释：重新审视映射嵌入的解释能力 

**Authors**: Hanna Herasimchyk, Alhassan Abdelhalim, Sören Laue, Michaela Regneri  

**Link**: [PDF](https://arxiv.org/pdf/2508.13729)  

**Abstract**: Understanding what knowledge is implicitly encoded in deep learning models is essential for improving the interpretability of AI systems. This paper examines common methods to explain the knowledge encoded in word embeddings, which are core elements of large language models (LLMs). These methods typically involve mapping embeddings onto collections of human-interpretable semantic features, known as feature norms. Prior work assumes that accurately predicting these semantic features from the word embeddings implies that the embeddings contain the corresponding knowledge. We challenge this assumption by demonstrating that prediction accuracy alone does not reliably indicate genuine feature-based interpretability.
We show that these methods can successfully predict even random information, concluding that the results are predominantly determined by an algorithmic upper bound rather than meaningful semantic representation in the word embeddings. Consequently, comparisons between datasets based solely on prediction performance do not reliably indicate which dataset is better captured by the word embeddings. Our analysis illustrates that such mappings primarily reflect geometric similarity within vector spaces rather than indicating the genuine emergence of semantic properties. 

**Abstract (ZH)**: 理解深度学习模型中隐含编码的知识对于提高人工智能系统的可解释性至关重要。本文探讨了用于解释词嵌入中编码知识的常见方法，这些方法是大规模语言模型（LLMs）的核心组成部分。这些方法通常涉及将嵌入映射到一组可解释的语义特征，即特征规范。先前的研究假设准确从词嵌入预测这些语义特征意味着嵌入包含了相应的知识。本文通过证明仅预测准确性不足以可靠地表明基于特征的可解释性来挑战这一假设。我们展示了这些方法甚至可以成功预测随机信息，从而得出结论，这些结果主要由算法上限决定，而不是词嵌入中的有意义的语义表示。因此，仅基于预测性能比较数据集不能可靠地表明哪个数据集被词嵌入更准确地捕捉。我们的分析表明，这类映射主要反映了向量空间内的几何相似性，而不是表明真正出现的语义属性。 

---
# Generics and Default Reasoning in Large Language Models 

**Title (ZH)**: 通用性与大型语言模型中的默认推理 

**Authors**: James Ravi Kirkpatrick, Rachel Katharine Sterken  

**Link**: [PDF](https://arxiv.org/pdf/2508.13718)  

**Abstract**: This paper evaluates the capabilities of 28 large language models (LLMs) to reason with 20 defeasible reasoning patterns involving generic generalizations (e.g., 'Birds fly', 'Ravens are black') central to non-monotonic logic. Generics are of special interest to linguists, philosophers, logicians, and cognitive scientists because of their complex exception-permitting behaviour and their centrality to default reasoning, cognition, and concept acquisition. We find that while several frontier models handle many default reasoning problems well, performance varies widely across models and prompting styles. Few-shot prompting modestly improves performance for some models, but chain-of-thought (CoT) prompting often leads to serious performance degradation (mean accuracy drop -11.14%, SD 15.74% in models performing above 75% accuracy in zero-shot condition, temperature 0). Most models either struggle to distinguish between defeasible and deductive inference or misinterpret generics as universal statements. These findings underscore both the promise and limits of current LLMs for default reasoning. 

**Abstract (ZH)**: 大型语言模型在攻防推理能力上的评估：基于泛涵非单调逻辑的攻防推理模式的研究 

---
# Input Time Scaling 

**Title (ZH)**: 输入时间缩放 

**Authors**: Rapheal Huang, Weilong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.13654)  

**Abstract**: Current Large Language Models (LLMs) are usually post-trained on large-scale carefully curated datasets (data & training scaling) and doing reasoning in test time (inference time scaling). In this work, we present a new scaling paradigm, Input Time Scaling, to complement previous scaling methods by putting resources on queries (input time). During training and testing, we combine meta-knowledge from LLMs to refine inputs with different strategies. We also find a new phenomenon, training-testing co-design there. We need to apply query strategies during both training and testing. Only applying strategies on training or testing would seriously degrade the performance. We are also surprised to find that seemingly low data quality datasets can gain high performance. Adding irrelevant information to the queries, randomly selecting examples from a minimally filtered dataset, can even perform the best. These findings contradict the widely held inductive bias, "garbage in, garbage out". Curating datasets with seemingly high-quality data can even potentially limit the performance ceiling. In addition, models trained on more data with similar quality (15k VS 1k) perform worse, simple dataset size scaling should also be carefully inspected. The good news is that our findings are compatible with the Less is More phenomenon. A small set of examples is enough to evoke high-level reasoning ability. With experiments on models trained on Qwen2.5-32B-Instruct, we are able to reach SOTA performance among 32B models on AIME24(76.7%) and AIME25(76.7%) pass@1. We can further achieve AIME24(76.7%) and AIME25(80%) with a majority vote of three models. Starting from DeepSeek-R1-Distill-Qwen-32B, the best result would be 86.7% on AIME24 and 76.7% on AIME25. To facilitate reproducibility and further research, we are working on open-source our datasets, data pipelines, evaluation results, and checkpoints. 

**Abstract (ZH)**: 当前大型语言模型通常通过大规模精心筛选的数据集进行后训练（数据和训练缩放），并在测试时间进行推理（推理时间缩放）。本文提出了一种新的缩放范式——输入时间缩放，以补充之前的方法，将资源放在查询上（输入时间）。在训练和测试过程中，我们结合大模型的元知识，使用不同的策略细化输入。我们还发现了一种新的现象——训练-测试协同设计。在训练和测试过程中都需要应用查询策略，仅在训练或测试过程中应用策略会大幅降低性能。我们还惊讶地发现，虽然数据质量看似较低的数据集可以获得高性能。向查询中添加无关信息，从少量过滤的数据集随机选择示例，即使可以获得最佳性能。这些发现与广泛持有的归纳偏见“垃圾进，垃圾出”相矛盾。精心筛选看似高质量的数据集甚至可能限制性能上限。此外，训练数据量更多但质量相似（15k比1k）的模型表现更差，简单的数据集大小缩放也需要谨慎检查。好消息是我们发现的结果与“少即是多”现象是兼容的。少量示例足以引发高层次的推理能力。通过在Qwen2.5-32B-Instruct训练的模型上进行实验，我们能够在AIME24（76.7%）和AIME25（76.7%）的pass@1上达到SOTA性能。我们还可以通过三个模型的投票实现AIME24（76.7%）和AIME25（80%）。从DeepSeek-R1-Distill-Qwen-32B开始，最佳结果为AIME24（86.7%）和AIME25（76.7%）。为了便于再现性和进一步研究，我们正在开源我们的数据集、数据管道、评估结果和检查点。 

---
# Towards a Larger Model via One-Shot Federated Learning on Heterogeneous Client Models 

**Title (ZH)**: 基于异构客户端模型的一次性联邦学习通往更大模型 

**Authors**: Wenxuan Ye, Xueli An, Onur Ayan, Junfan Wang, Xueqiang Yan, Georg Carle  

**Link**: [PDF](https://arxiv.org/pdf/2508.13625)  

**Abstract**: Large models, renowned for superior performance, outperform smaller ones even without billion-parameter scales. While mobile network servers have ample computational resources to support larger models than client devices, privacy constraints prevent clients from directly sharing their raw data. Federated Learning (FL) enables decentralized clients to collaboratively train a shared model by exchanging model parameters instead of transmitting raw data. Yet, it requires a uniform model architecture and multiple communication rounds, which neglect resource heterogeneity, impose heavy computational demands on clients, and increase communication overhead. To address these challenges, we propose FedOL, to construct a larger and more comprehensive server model in one-shot settings (i.e., in a single communication round). Instead of model parameter sharing, FedOL employs knowledge distillation, where clients only exchange model prediction outputs on an unlabeled public dataset. This reduces communication overhead by transmitting compact predictions instead of full model weights and enables model customization by allowing heterogeneous model architectures. A key challenge in this setting is that client predictions may be biased due to skewed local data distributions, and the lack of ground-truth labels in the public dataset further complicates reliable learning. To mitigate these issues, FedOL introduces a specialized objective function that iteratively refines pseudo-labels and the server model, improving learning reliability. To complement this, FedOL incorporates a tailored pseudo-label generation and knowledge distillation strategy that effectively integrates diverse knowledge. Simulation results show that FedOL significantly outperforms existing baselines, offering a cost-effective solution for mobile networks where clients possess valuable private data but limited computational resources. 

**Abstract (ZH)**: Large 模型：无需 billion 参数规模亦能超越小型模型，即使在移动网络服务器拥有充足计算资源而客户端隐私受限无法直接共享原始数据的情况下，联邦学习 (FL) 通过交换模型参数而非传输原始数据，让去中心化的客户端协作训练共享模型。然而，FL 要求统一的模型架构和多轮通信，忽视了资源异构性，对客户端提出了沉重的计算要求，并增加了通信开销。为此，我们提出了 FedOL，以一次性（即单轮通信）构建一个更大、更全面的服务器模型。与模型参数共享不同，FedOL 采用知识蒸馏策略，客户端仅交换对未标注公开数据集的模型预测输出。这通过传输紧凑的预测而非完整模型权重减少了通信开销，并允许不同模型架构的定制。在这一设置中，一个关键挑战是客户端预测可能会由于本地数据分布偏差而失真，且公开数据集缺乏真实标签进一步增加了可靠学习的复杂性。为应对这些问题，FedOL 引入了一种专门的目标函数，迭代细化伪标签和服务器模型，从而提高学习可靠性。此外，FedOL 结合了定制化的伪标签生成和知识蒸馏策略，有效整合了多样化的知识。仿真实验结果表明，FedOL 在现有基线方法中表现显著优异，为拥有宝贵私有数据但计算资源有限的移动网络提供了一种成本效益高的解决方案。 

---
# Who Gets the Mic? Investigating Gender Bias in the Speaker Assignment of a Speech-LLM 

**Title (ZH)**: 谁来发声？探究演讲-LLM中发言者分配的性别偏见 

**Authors**: Dariia Puhach, Amir H. Payberah, Éva Székely  

**Link**: [PDF](https://arxiv.org/pdf/2508.13603)  

**Abstract**: Similar to text-based Large Language Models (LLMs), Speech-LLMs exhibit emergent abilities and context awareness. However, whether these similarities extend to gender bias remains an open question. This study proposes a methodology leveraging speaker assignment as an analytic tool for bias investigation. Unlike text-based models, which encode gendered associations implicitly, Speech-LLMs must produce a gendered voice, making speaker selection an explicit bias cue. We evaluate Bark, a Text-to-Speech (TTS) model, analyzing its default speaker assignments for textual prompts. If Bark's speaker selection systematically aligns with gendered associations, it may reveal patterns in its training data or model design. To test this, we construct two datasets: (i) Professions, containing gender-stereotyped occupations, and (ii) Gender-Colored Words, featuring gendered connotations. While Bark does not exhibit systematic bias, it demonstrates gender awareness and has some gender inclinations. 

**Abstract (ZH)**: 类似于文本基础的大语言模型（LLMs），语音LLMs表现出 emergent 能力和背景意识。然而，这些相似性是否扩展到性别偏差仍是一个开放的问题。本研究提出了一种利用说话人分配作为偏差调查分析工具的方法。与隐含编码性别关联的文本基础模型不同，语音LLMs必须生成带性别的声音，使说话人选择成为显式的偏差提示。我们评估了Bark这一文本转语音（TTS）模型，分析其对文本提示的默认说话人分配。如果Bark的说话人选择系统地与性别关联一致，这可能揭示其训练数据或模型设计中的模式。为了测试这一点，我们构建了两个数据集：(i) 职业，包含性别刻板印象的职业，以及(ii) 性别色彩词汇，包含性别联想。尽管Bark没有表现出系统性的偏差，但它表现出性别意识，并且具有某些性别倾向。 

---
# A Comparative Study of Decoding Strategies in Medical Text Generation 

**Title (ZH)**: 医疗文本生成中解码策略的比较研究 

**Authors**: Oriana Presacan, Alireza Nik, Vajira Thambawita, Bogdan Ionescu, Michael Riegler  

**Link**: [PDF](https://arxiv.org/pdf/2508.13580)  

**Abstract**: Large Language Models (LLMs) rely on various decoding strategies to generate text, and these choices can significantly affect output quality. In healthcare, where accuracy is critical, the impact of decoding strategies remains underexplored. We investigate this effect in five open-ended medical tasks, including translation, summarization, question answering, dialogue, and image captioning, evaluating 11 decoding strategies with medically specialized and general-purpose LLMs of different sizes. Our results show that deterministic strategies generally outperform stochastic ones: beam search achieves the highest scores, while {\eta} and top-k sampling perform worst. Slower decoding methods tend to yield better quality. Larger models achieve higher scores overall but have longer inference times and are no more robust to decoding. Surprisingly, while medical LLMs outperform general ones in two of the five tasks, statistical analysis shows no overall performance advantage and reveals greater sensitivity to decoding choice. We further compare multiple evaluation metrics and find that correlations vary by task, with MAUVE showing weak agreement with BERTScore and ROUGE, as well as greater sensitivity to the decoding strategy. These results highlight the need for careful selection of decoding methods in medical applications, as their influence can sometimes exceed that of model choice. 

**Abstract (ZH)**: 大型语言模型（LLMs）依赖各种解码策略生成文本，这些选择会显著影响输出质量。在对准确性要求极高的医疗领域，解码策略的影响依然未被充分探索。我们在五个开放性医疗任务中进行了调查，包括翻译、总结、问答、对话和图像字幕，评估了11种解码策略在不同规模的医学专用和通用大语言模型上的效果。结果显示，确定性策略通常优于随机策略：束搜索获得最高分，而η和top-k采样表现最差。较慢的解码方法倾向于生成更好的质量。更大规模的模型总体上获得更高分，但推断时间更长且对解码策略的鲁棒性无显著提高。令人惊讶的是，尽管在五个任务中有两个任务中医学大语言模型表现优于通用模型，但统计分析显示其整体性能优势并不明显，并且对解码选择更为敏感。我们进一步比较了多种评估指标，发现它们在不同任务中的相关性各异，MAUVE与BERTScore和ROUGE的相关性较弱，并且对解码策略更为敏感。这些结果强调了在医疗应用中仔细选择解码方法的必要性，因为其影响有时可能超过模型选择的影响。 

---
# ProMed: Shapley Information Gain Guided Reinforcement Learning for Proactive Medical LLMs 

**Title (ZH)**: ProMed: 由Shapley信息增益引导的主动医疗LLM增强学习 

**Authors**: Hongxin Ding, Baixiang Huang, Yue Fang, Weibin Liao, Xinke Jiang, Zheng Li, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13514)  

**Abstract**: Interactive medical questioning is essential in real-world clinical consultations, where physicians must actively gather information from patients. While medical Large Language Models (LLMs) have shown impressive capabilities in static medical question answering, they predominantly operate under a reactive paradigm: generating answers directly without seeking additional information, which risks incorrect diagnoses in such interactive settings. To address this limitation, we propose ProMed, a reinforcement learning (RL) framework that transitions medical LLMs toward a proactive paradigm, equipping them with the ability to ask clinically valuable questions before decision-making. At the core of ProMed is the Shapley Information Gain (SIG) reward, which quantifies the clinical utility of each question by combining the amount of newly acquired information with its contextual importance, estimated via Shapley values. We integrate SIG into a two-stage training pipeline: (1) SIG-Guided Model Initialization uses Monte Carlo Tree Search (MCTS) to construct high-reward interaction trajectories to supervise the model, and (2) SIG-Augmented Policy Optimization, which integrates SIG and enhances RL with a novel SIG-guided Reward Distribution Mechanism that assigns higher rewards to informative questions for targeted optimization. Extensive experiments on two newly curated partial-information medical benchmarks demonstrate that ProMed significantly outperforms state-of-the-art methods by an average of 6.29% and delivers a 54.45% gain over the reactive paradigm, while also generalizing robustly to out-of-domain cases. 

**Abstract (ZH)**: Interactive Medical询问对人体临床咨询至关重要，医师需要主动从患者处收集信息。尽管医学大型语言模型（LLMs）在静态医学问答任务上展现了令人印象深刻的能力，但它们主要以被动的方式运作：直接生成答案而不寻求额外信息，这在互动式设置中可能导致错误的诊断。为解决这一局限，我们提出ProMed，这是一种强化学习（RL）框架，使医疗LLMs从被动模式转向主动模式，赋予它们在决策前提出临床有价值的询问的能力。ProMed的核心是Shapley信息增益（SIG）奖励，该奖励通过结合新获得信息的量与其上下文重要性（使用Shapley值估计）来量化每个问题的临床价值。我们将SIG整合到两阶段的训练管道中：（1）由Monte Carlo树搜索（MCTS）指导的模型初始化构建高奖励交互轨迹以监督模型，（2）通过一种新颖的SIG指导的奖励分配机制增强RL，为指令优化分配更高奖励的互动性问题以实现特定优化。在两个新的部分信息医学基准测试上进行的广泛实验表明，ProMed平均优于最新方法6.29%，并且相比被动模式提高了54.45%的性能，同时在跨领域案例上也表现出稳健的泛化能力。 

---
# LLM-Enhanced Linear Autoencoders for Recommendation 

**Title (ZH)**: LLM增强的线性自动编码器推荐方法 

**Authors**: Jaewan Moon, Seongmin Park, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13500)  

**Abstract**: Large language models (LLMs) have been widely adopted to enrich the semantic representation of textual item information in recommender systems. However, existing linear autoencoders (LAEs) that incorporate textual information rely on sparse word co-occurrence patterns, limiting their ability to capture rich textual semantics. To address this, we propose L3AE, the first integration of LLMs into the LAE framework. L3AE effectively integrates the heterogeneous knowledge of textual semantics and user-item interactions through a two-phase optimization strategy. (i) L3AE first constructs a semantic item-to-item correlation matrix from LLM-derived item representations. (ii) It then learns an item-to-item weight matrix from collaborative signals while distilling semantic item correlations as regularization. Notably, each phase of L3AE is optimized through closed-form solutions, ensuring global optimality and computational efficiency. Extensive experiments demonstrate that L3AE consistently outperforms state-of-the-art LLM-enhanced models on three benchmark datasets, achieving gains of 27.6% in Recall@20 and 39.3% in NDCG@20. The source code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型(LLMs)已被广泛应用于丰富推荐系统中文本项信息的语义表示。然而，现有的结合文本信息的线性自编码器(LAEs)依赖稀疏的词共现模式，限制了其捕获丰富语义的能力。为了解决这个问题，我们提出L3AE，这是将LLMs融入LAE框架的第一个尝试。L3AE通过两阶段优化策略有效整合了文本语义和用户-项交互的异质知识。(i) L3AE首先从LLM提取的项表示中构建一个语义项-项相关矩阵。(ii) 然后，它从协作信号中学习一个项-项权重矩阵，同时通过正则化保留语义项相关性。值得注意的是，L3AE的每个阶段都通过闭式解进行优化，确保了全局最优性和计算效率。广泛的经验研究表明，L3AE在三个基准数据集上均优于最先进的LLM增强模型，Recall@20和NDCG@20分别提高了27.6%和39.3%。源代码可在此处获取。 

---
# ALIGN: Word Association Learning for Cross-Cultural Generalization in Large Language Models 

**Title (ZH)**: ALIGN: 跨文化通用性中的单词关联学习 

**Authors**: Chunhua Liu, Kabir Manandhar Shrestha, Sukai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13426)  

**Abstract**: As large language models (LLMs) increasingly mediate cross-cultural communication, their behavior still reflects the distributional bias of the languages and viewpoints that are over-represented in their pre-training corpora. Yet, it remains a challenge to model and align culture due to limited cultural knowledge and a lack of exploration into effective learning approaches. We introduce a cost-efficient, cognitively grounded remedy: parameter-efficient fine-tuning on native speakers' free word-association norms, which encode implicit cultural schemas. Leveraging English-US and Mandarin associations from the Small-World-of-Words project, we adapt Llama-3.1-8B and Qwen-2.5-7B via supervised fine-tuning (SFT) and PPO-based preference optimization. SFT boosts held-out association Precision at 5 by 16-20% in English and 43-165% in Mandarin, lifts median concreteness by +0.20, and attains human-level valence and arousal. These lexical gains transfer: on World-Values-Survey questions, fine-tuned models shift answer distributions toward the target culture, and on a 50-item high-tension subset, Qwen's Chinese-aligned responses double while Llama's US bias drops by one-third. Our 7-8B models rival or beat vanilla 70B baselines, showing that a few million culture-grounded associations can instill value alignment without costly retraining. Our work highlights both the promise and the need for future research grounded in human cognition in improving cultural alignment in AI models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在越来越多地调解跨文化沟通时，其行为仍然反映了其预训练语料中过度代表的语言和观点的分布性偏见。但是，由于文化知识有限且缺乏有效的学习方法探索，对文化建模和对齐仍然是一项挑战。我们介绍了一种成本效益高且契合认知的解决方案：在母语者自由词汇联想规范上进行参数高效微调，以此编码出隐含的文化模式。利用Small-World-of-Words项目中的英语-美国和普通话联想，我们通过监督微调（SFT）和基于PPO的偏好优化适应了Llama-3.1-8B和Qwen-2.5-7B。SFT在英语中将保留联想的精确度提高了16-20%，在普通话中提高了43-165%，将中值具体性提升了0.20，并达到了人类水平的价值和唤醒程度。这些词汇上的改进得以转移：在世界价值观调查问卷中，微调后的模型将答案分布向目标文化转移，而在一个包含50个项目的高张力子集上，Qwen的中国文化对齐回答翻了一番，而Llama的美国倾向降低了三分之一。我们的7-8B模型与或优于 vanilla 70B基线，表明数百万文化基础的联想可以实现价值观对齐而无需昂贵的重新训练。我们的工作突显了基于人类认知改进AI模型文化对齐的前景和未来研究的迫切需要。 

---
# Datarus-R1: An Adaptive Multi-Step Reasoning LLM for Automated Data Analysis 

**Title (ZH)**: Datarus-R1：一种适应性多步推理大型语言模型，用于自动化数据分析 

**Authors**: Ayoub Ben Chaliah, Hela Dellagi  

**Link**: [PDF](https://arxiv.org/pdf/2508.13382)  

**Abstract**: We present Datarus-R1-14B, a 14 B-parameter open-weights language model fine-tuned from Qwen 2.5-14B-Instruct to act as a virtual data analyst and graduate-level problem solver. Datarus is trained not on isolated question-answer pairs but on full analytical trajectories including reasoning steps, code execution, error traces, self-corrections, and final conclusions, all captured in a ReAct-style notebook format spanning finance, medicine, numerical analysis, and other quantitative domains. Our training pipeline combines (i) a trajectory-centric synthetic data generator that yielded 144 000 tagged notebook episodes, (ii) a dual-reward framework blending a lightweight tag-based structural signal with a Hierarchical Reward Model (HRM) that scores both single-step soundness and end-to-end coherence, and (iii) a memory-optimized implementation of Group Relative Policy Optimization (GRPO) featuring KV-cache reuse, sequential generation, and reference-model sharding. A cosine curriculum smoothly shifts emphasis from structural fidelity to semantic depth, reducing the format collapse and verbosity that often plague RL-aligned LLMs. A central design choice in Datarus is it dual reasoning interface. In agentic mode the model produces ReAct-tagged steps that invoke Python tools to execute real code; in reflection mode it outputs compact Chain-of-Thought (CoT) traces delimited by <think> and <answer> tags. On demanding postgraduate-level problems, Datarus exhibits an "AHA-moment" pattern: it sketches hypotheses, revises them once or twice, and converges avoiding the circular, token-inflating loops common to contemporary systems. Across standard public benchmarks Datarus surpasses similar size models and even reaches the level of larger reasoning models such as QwQ-32B achieving up to 30% higher accuracy on AIME 2024/2025 and LiveCodeBench while emitting 18-49% fewer tokens per solution. 

**Abstract (ZH)**: Datarus-R1-14B：一个基于Qwen 2.5-14B-Instruct微调的虚拟数据分析师和研究生级问题解决者大型语言模型 

---
# ViTAD: Timing Violation-Aware Debugging of RTL Code using Large Language Models 

**Title (ZH)**: ViTAD: 基于大型语言模型的RTL代码 Timing Violation 意识调试 

**Authors**: Wenhao Lv, Yingjie Xia, Xiyuan Chen, Li Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13257)  

**Abstract**: In modern Very Large Scale Integrated (VLSI) circuit design flow, the Register-Transfer Level (RTL) stage presents a critical opportunity for timing optimization. Addressing timing violations at this early stage is essential, as modern systems demand higher speeds, where even minor timing violations can lead to functional failures or system crashes. However, traditional timing optimization heavily relies on manual expertise, requiring engineers to iteratively analyze timing reports and debug. To automate this process, this paper proposes ViTAD, a method that efficiently analyzes the root causes of timing violations and dynamically generates targeted repair strategies. Specifically, we first parse Verilog code and timing reports to construct a Signal Timing Dependency Graph (STDG). Based on the STDG, we perform violation path analysis and use large language models (LLMs) to infer the root causes of violations. Finally, by analyzing the causes of violations, we selectively retrieve relevant debugging knowledge from a domain-specific knowledge base to generate customized repair solutions. To evaluate the effectiveness of our method, we construct a timing violation dataset based on real-world open-source projects. This dataset contains 54 cases of violations. Experimental results show that our method achieves a 73.68% success rate in repairing timing violations, while the baseline using only LLM is 54.38%. Our method improves the success rate by 19.30%. 

**Abstract (ZH)**: 现代Very Large Scale Integrated (VLSI)电路设计流程中，Register-Transfer Level (RTL)阶段提供了关键的时序优化机会。在这一早期阶段解决时序违规至关重要，因为现代系统要求更高的速度，即使是很小的时序违规也可能导致功能失效或系统崩溃。然而，传统的时序优化高度依赖人工专业知识，要求工程师反复分析时序报告并调试。为此，本文提出ViTAD方法，该方法能够高效地分析时序违规的根本原因，并动态生成针对性的修复策略。具体而言，我们首先解析Verilog代码和时序报告，构建信号时序依赖图(STDG)。基于STDG，我们执行违规路径分析，并使用大型语言模型(LLMs)推断违规的根本原因。最后，通过分析违规原因，我们从特定领域的知识库中选择性地检索相关调试知识，生成定制化的修复解决方案。为评估方法的有效性，我们基于实际开源项目构建了一个时序违规数据集，该数据集包含54个违规案例。实验结果表明，我们的方法在修复时序违规方面的成功率达到了73.68%，而仅使用LLM的基础方法为54.38%。我们的方法提高了成功率19.30%。 

---
# Involuntary Jailbreak 

**Title (ZH)**: 非自愿越狱 

**Authors**: Yangyang Guo, Yangyan Li, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.13246)  

**Abstract**: In this study, we disclose a worrying new vulnerability in Large Language Models (LLMs), which we term \textbf{involuntary jailbreak}. Unlike existing jailbreak attacks, this weakness is distinct in that it does not involve a specific attack objective, such as generating instructions for \textit{building a bomb}. Prior attack methods predominantly target localized components of the LLM guardrail. In contrast, involuntary jailbreaks may potentially compromise the entire guardrail structure, which our method reveals to be surprisingly fragile. We merely employ a single universal prompt to achieve this goal. In particular, we instruct LLMs to generate several questions that would typically be rejected, along with their corresponding in-depth responses (rather than a refusal). Remarkably, this simple prompt strategy consistently jailbreaks the majority of leading LLMs, including Claude Opus 4.1, Grok 4, Gemini 2.5 Pro, and GPT 4.1. We hope this problem can motivate researchers and practitioners to re-evaluate the robustness of LLM guardrails and contribute to stronger safety alignment in future. 

**Abstract (ZH)**: 本研究揭示了大型语言模型（LLMs）中的一个令人担忧的新漏洞，我们称之为**非自愿 Jailbreak**。这一弱点与众不同之处在于，它不涉及特定的攻击目标，例如生成制作炸弹的指令。以往的攻击方法主要针对 LLM 防护栏的局部组件。相比之下，非自愿 Jailbreak 有可能在整个防护栏结构上造成破坏，而我们的方法揭示了这一结构出奇地脆弱。我们仅使用一个通用提示便实现了这一目标。具体来说，我们指示 LLM 生成一些通常会被拒绝的问题及其相应的深入回答（而不是直接拒绝）。令人惊讶的是，这一简单的提示策略成功地攻破了包括 Claude Opus 4.1、Grok 4、Gemini 2.5 Pro 和 GPT 4.1 在内的大多数顶级 LLM。我们希望这一问题能够促使研究人员和实践者重新评估 LLM 防护栏的 robustness，并为未来更强大的安全性对齐做出贡献。 

---
# Quantifying Loss Aversion in Cyber Adversaries via LLM Analysis 

**Title (ZH)**: 通过大语言模型分析量化网络对手的损失回避程度 

**Authors**: Soham Hans, Nikolos Gurney, Stacy Marsella, Sofia Hirschmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.13240)  

**Abstract**: Understanding and quantifying human cognitive biases from empirical data has long posed a formidable challenge, particularly in cybersecurity, where defending against unknown adversaries is paramount. Traditional cyber defense strategies have largely focused on fortification, while some approaches attempt to anticipate attacker strategies by mapping them to cognitive vulnerabilities, yet they fall short in dynamically interpreting attacks in progress. In recognition of this gap, IARPA's ReSCIND program seeks to infer, defend against, and even exploit attacker cognitive traits. In this paper, we present a novel methodology that leverages large language models (LLMs) to extract quantifiable insights into the cognitive bias of loss aversion from hacker behavior. Our data are collected from an experiment in which hackers were recruited to attack a controlled demonstration network. We process the hacker generated notes using LLMs using it to segment the various actions and correlate the actions to predefined persistence mechanisms used by hackers. By correlating the implementation of these mechanisms with various operational triggers, our analysis provides new insights into how loss aversion manifests in hacker decision-making. The results demonstrate that LLMs can effectively dissect and interpret nuanced behavioral patterns, thereby offering a transformative approach to enhancing cyber defense strategies through real-time, behavior-based analysis. 

**Abstract (ZH)**: 从实验数据理解并量化人类认知偏差：利用大型语言模型揭示黑客行为中的损失厌恶认知偏差 

---
# Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System 

**Title (ZH)**: 在异构内存系统中通过动态KV缓存放置加速LLM推理 

**Authors**: Yunhua Fang, Rui Xie, Asad Ul Haq, Linsen Ma, Kaoutar El Maghraoui, Naigang Wang, Meng Wang, Liu Liu, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13231)  

**Abstract**: Large Language Model (LLM) inference is increasingly constrained by memory bandwidth, with frequent access to the key-value (KV) cache dominating data movement. While attention sparsity reduces some memory traffic, the relevance of past tokens varies over time, requiring the full KV cache to remain accessible and sustaining pressure on both bandwidth and capacity. With advances in interconnects such as NVLink and LPDDR5X, modern AI hardware now integrates high-bandwidth memory (HBM) with high-speed off-package DRAM, making heterogeneous memory systems a practical solution. This work investigates dynamic KV cache placement across such systems to maximize aggregated bandwidth utilization under capacity constraints. Rather than proposing a specific scheduling policy, we formulate the placement problem mathematically and derive a theoretical upper bound, revealing substantial headroom for runtime optimization. To our knowledge, this is the first formal treatment of dynamic KV cache scheduling in heterogeneous memory systems for LLM inference. 

**Abstract (ZH)**: 大型语言模型（LLM）推理越来越多地受到内存带宽的限制，频繁访问键值（KV）缓存主导着数据移动。虽然注意力稀疏性减少了部分内存流量，但过去令牌的相关性会随时间变化，要求完整KV缓存保持可访问性，从而持续对带宽和容量造成压力。随着NVLink和LPDDR5X等互连技术的进步，现代AI硬件现在将高性能记忆体（HBM）与高速外部DRAM集成为一体，使异构内存系统成为可行的解决方案。本文探讨了在这些系统中动态放置KV缓存，以在容量受限条件下最大化聚合带宽利用率。我们并未提出具体的调度策略，而是从数学上形式化了放置问题，并推导出一个理论上限，揭示了运行时优化的巨大空间。据我们所知，这是首次对异构内存系统中LLM推理的动态KV缓存调度进行形式化处理的研究。 

---
# MCPSecBench: A Systematic Security Benchmark and Playground for Testing Model Context Protocols 

**Title (ZH)**: MCPSecBench: 一种模型上下文协议测试的系统性安全基准和实验平台 

**Authors**: Yixuan Yang, Daoyuan Wu, Yufan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13220)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications via the Model Context Protocol (MCP), a universal, open standard for connecting AI agents with data sources and external tools. While MCP enhances the capabilities of LLM-based agents, it also introduces new security risks and expands their attack surfaces. In this paper, we present the first systematic taxonomy of MCP security, identifying 17 attack types across 4 primary attack surfaces. We introduce MCPSecBench, a comprehensive security benchmark and playground that integrates prompt datasets, MCP servers, MCP clients, and attack scripts to evaluate these attacks across three major MCP providers. Our benchmark is modular and extensible, allowing researchers to incorporate custom implementations of clients, servers, and transport protocols for systematic security assessment. Experimental results show that over 85% of the identified attacks successfully compromise at least one platform, with core vulnerabilities universally affecting Claude, OpenAI, and Cursor, while prompt-based and tool-centric attacks exhibit considerable variability across different hosts and models. Overall, MCPSecBench standardizes the evaluation of MCP security and enables rigorous testing across all MCP layers. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过模型上下文协议（MCP）越来越多地融入实际应用中，MCP是一种通用的开放标准，用于连接AI代理与数据源和外部工具。虽然MCP提升了基于LLM的代理的能力，但也引入了新的安全风险并扩大了其攻击面。在本文中，我们提出了MCP安全的第一个系统性分类，识别出17种攻击类型，涉及4个主要攻击面。我们引入了MCPSecBench，这是一个综合的安全基准平台，集成了一系列数据集、MCP服务器、MCP客户端和攻击脚本，用于评估这三种主要MCP提供者中的攻击。该基准平台模块化且可扩展，允许研究人员纳入自定义的客户端、服务器和传输协议实现，以进行系统性安全评估。实验结果显示，超过85%的已识别攻击成功地至少攻破了一个平台，核心漏洞普遍影响Claude、OpenAI和Cursor，而基于提示和工具中心的攻击在不同主机和模型之间表现出较大的变异性。总体而言，MCPSecBench统一了MCP安全的评估标准，并能在MCP的所有层次上进行严格测试。 

---
# Too Easily Fooled? Prompt Injection Breaks LLMs on Frustratingly Simple Multiple-Choice Questions 

**Title (ZH)**: 太容易受骗了吗？提示注入使语言模型在令人 frustratingly 简单的选择题上失效 

**Authors**: Xuyang Guo, Zekai Huang, Zhao Song, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13214)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong emergent abilities in complex reasoning and zero-shot generalization, showing unprecedented potential for LLM-as-a-judge applications in education, peer review, and data quality evaluation. However, their robustness under prompt injection attacks, where malicious instructions are embedded into the content to manipulate outputs, remains a significant concern. In this work, we explore a frustratingly simple yet effective attack setting to test whether LLMs can be easily misled. Specifically, we evaluate LLMs on basic arithmetic questions (e.g., "What is 3 + 2?") presented as either multiple-choice or true-false judgment problems within PDF files, where hidden prompts are injected into the file. Our results reveal that LLMs are indeed vulnerable to such hidden prompt injection attacks, even in these trivial scenarios, highlighting serious robustness risks for LLM-as-a-judge applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理和零样本泛化方面表现出强大的 emergent 能力，展示了在教育、同行评审和数据质量评估中的 LLM-as-a-judge 应用的巨大潜力。然而，在提示注入攻击下（恶意指令被嵌入内容以操控输出）的鲁棒性问题仍然是一个重大关注点。在本工作中，我们探索了一个令人沮丧的简单但有效的攻击设置，以测试LLMs是否容易被误导。具体地，我们在PDF文件中对LLMs进行基本算术问题（例如，“3 + 2 是多少？”）的评估，这些问题以多项选择或真伪判断的形式呈现，并在文件中注入了隐蔽提示。我们的研究结果表明，即使在这些简单的场景下，LLMs也容易受到隐蔽提示注入攻击的影响，突显了LLM-as-a-judge应用中严重的鲁棒性风险。 

---
# Benchmarking LLM-based Agents for Single-cell Omics Analysis 

**Title (ZH)**: 基于LLM的代理单细胞组学分析基准测试 

**Authors**: Yang Liu, Lu Zhou, Ruikun He, Rongbo Shen, Yixue Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13201)  

**Abstract**: The surge in multimodal single-cell omics data exposes limitations in traditional, manually defined analysis workflows. AI agents offer a paradigm shift, enabling adaptive planning, executable code generation, traceable decisions, and real-time knowledge fusion. However, the lack of a comprehensive benchmark critically hinders progress. We introduce a novel benchmarking evaluation system to rigorously assess agent capabilities in single-cell omics analysis. This system comprises: a unified platform compatible with diverse agent frameworks and LLMs; multidimensional metrics assessing cognitive program synthesis, collaboration, execution efficiency, bioinformatics knowledge integration, and task completion quality; and 50 diverse real-world single-cell omics analysis tasks spanning multi-omics, species, and sequencing technologies. Our evaluation reveals that Grok-3-beta achieves state-of-the-art performance among tested agent frameworks. Multi-agent frameworks significantly enhance collaboration and execution efficiency over single-agent approaches through specialized role division. Attribution analyses of agent capabilities identify that high-quality code generation is crucial for task success, and self-reflection has the most significant overall impact, followed by retrieval-augmented generation (RAG) and planning. This work highlights persistent challenges in code generation, long-context handling, and context-aware knowledge retrieval, providing a critical empirical foundation and best practices for developing robust AI agents in computational biology. 

**Abstract (ZH)**: 多模态单细胞组学数据激增揭示了传统手动定义分析工作流的局限性。AI代理提供了范式的转变，能够实现自适应规划、可执行代码生成、可追溯的决策和实时知识融合。然而，缺乏全面的基准测试严重阻碍了进步。我们提出了一种新的基准评估系统，以严格评估代理在单细胞组学分析中的能力。该系统包括：一个兼容多种代理框架和大规模语言模型的统一平台；多维度指标评估认知程序合成、协作、执行效率、生物信息学知识整合和任务完成质量；以及涵盖多组学、物种和测序技术的50个多样化的真实世界单细胞组学分析任务。我们的评估显示，在测试的代理框架中，Grok-3-beta 达到了最先进的性能。多代理框架通过专门的角色分工显著提高了协作和执行效率，超过单代理方法。代理能力归因分析表明，高质量代码生成对于任务成功至关重要，自我反思的影响最大，其次是检索增强生成（RAG）和规划。这项工作突出了代码生成、长上下文处理和上下文感知知识检索中的持续挑战，为在计算生物学中开发稳健的AI代理提供了关键的实证基础和最佳实践。 

---
# Combating Homelessness Stigma with LLMs: A New Multi-Modal Dataset for Bias Detection 

**Title (ZH)**: 使用大语言模型对抗无家可归者污名：一种新的多模态数据集用于偏见检测 

**Authors**: Jonathan A. Karr Jr., Benjamin F. Herbst, Ting Hua, Matthew Hauenstein, Georgina Curto, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2508.13187)  

**Abstract**: Homelessness is a persistent social challenge, impacting millions worldwide. Over 770,000 people experienced homelessness in the U.S. in 2024. Social stigmatization is a significant barrier to alleviation, shifting public perception, and influencing policymaking. Given that online and city council discourse reflect and influence part of public opinion, it provides valuable insights to identify and track social biases. This research contributes to alleviating homelessness by acting on public opinion. It introduces novel methods, building on natural language processing (NLP) and large language models (LLMs), to identify and measure PEH social bias expressed in digital spaces. We present a new, manually-annotated multi-modal dataset compiled from Reddit, X (formerly Twitter), news articles, and city council meeting minutes across 10 U.S. cities. This unique dataset provides evidence of the typologies of homelessness bias described in the literature. In order to scale up and automate the detection of homelessness bias online, we evaluate LLMs as classifiers. We applied both zero-shot and few-shot classification techniques to this data. We utilized local LLMs (Llama 3.2 3B Instruct, Qwen 2.5 7B Instruct, and Phi4 Instruct Mini) as well as closed-source API models (GPT-4.1, Gemini 2.5 Pro, and Grok-4). Our findings reveal that although there are significant inconsistencies in local LLM zero-shot classification, the in-context learning classification scores of local LLMs approach the classification scores of closed-source LLMs. Furthermore, LLMs outperform BERT when averaging across all categories. This work aims to raise awareness about the pervasive bias against PEH, develop new indicators to inform policy, and ultimately enhance the fairness and ethical application of Generative AI technologies. 

**Abstract (ZH)**: 无家屋问题是持续存在的的社会挑战，影响着全球数百万人的生活。据2 2 2  2 2 2年数据显示，在美国，2 多有 7  on 7  on 7  people  people  on  on  on  7  on  on  on  on  有人  起有  on  on  on  on  on  on  on  on  on  人  on  近年7  on  on  人  上  无家 � línea 是 � 对代际传递造成 的偏见及 幋对 在  on  on  还on  on  on  进行 影响人们的观点变革和 影响政策制定。鉴于网络社交媒体和 城市议会的讨论在舆论形成中扮演的角色， 提供 通过分析社交媒体上的的舆论可以部分 提取 无家 層的的洞见 on 了解社会偏见。本研究旨在通过自然语言处理（NLP on 生成式大型语言模型（LLMs on 对无家屋偏见的系统地监测和 量化。我们构建了一个多模态数据集 on 从 Reddit on  X on 新闻文章 on 以及美国城市议会记录中 分对 这  on  on  进一步进行了偏见分类研究。通过自然语言生成式大模型 on 如 和传统零样本分类法和 少量样本分类法 on 我们测试了不同的自动偏见识别方法。研究结果表明 on 在 基地 语言模型在零样本分类的任务上存在显著挑战 但在特定上下文学习上可以获得与非开放源语言模型媲美的表现 on 吉上 语言模型在综合评估上优于 BERT on 不 进一步促 我们强调了对 在个人和 礿 政策层面上应重视无家无视角偏见 on 以及on 生成式人工智能技术的公平与 遁伦理应用。 

---
# Using Artificial Intuition in Distinct, Minimalist Classification of Scientific Abstracts for Management of Technology Portfolios 

**Title (ZH)**: 使用人工直觉对科技组合摘要进行精确分类管理 

**Authors**: Prateek Ranka, Fred Morstatter, Andrea Belz, Alexandra Graddy-Reed  

**Link**: [PDF](https://arxiv.org/pdf/2508.13182)  

**Abstract**: Classification of scientific abstracts is useful for strategic activities but challenging to automate because the sparse text provides few contextual clues. Metadata associated with the scientific publication can be used to improve performance but still often requires a semi-supervised setting. Moreover, such schemes may generate labels that lack distinction -- namely, they overlap and thus do not uniquely define the abstract. In contrast, experts label and sort these texts with ease. Here we describe an application of a process we call artificial intuition to replicate the expert's approach, using a Large Language Model (LLM) to generate metadata. We use publicly available abstracts from the United States National Science Foundation to create a set of labels, and then we test this on a set of abstracts from the Chinese National Natural Science Foundation to examine funding trends. We demonstrate the feasibility of this method for research portfolio management, technology scouting, and other strategic activities. 

**Abstract (ZH)**: ### 论文标题翻译

科学摘要分类对于战略活动非常有用，，但自动化这一过程极具挑战性，，因为摘要内容稀少，，提供的背景线索极少。可以通过关联的元数据来改进性能，但是通常需要在半监督环境下进行。此外，，这样的方案可能会生成缺乏区分性的标签—也就是说它们相互有交叉重叠且无法唯一定义摘要。鉴于专家可以轻松地地编写和分类这些文本。我们采用了我们称为人工直觉的过程来复制专家的作用。使用一个大规模语言模型（LLM）生成元数据。使用来自美国国立科学基金会的公开摘要数据集来生成标签，然后在来自中国国家自然科学基金会的摘要数据集上进行验证和评估，以考察此方法在科研项目管理、技术和战略规划上的的可行性。 

---
# White-Box Reasoning: Synergizing LLM Strategy and gm/Id Data for Automated Analog Circuit Design 

**Title (ZH)**: 白盒推理：结合LLM策略和gm/Id数据的自动化模拟电路设计 

**Authors**: Jianqiu Chen, Siqi Li, Xu He  

**Link**: [PDF](https://arxiv.org/pdf/2508.13172)  

**Abstract**: Analog IC design is a bottleneck due to its reliance on experience and inefficient simulations, as traditional formulas fail in advanced nodes. Applying Large Language Models (LLMs) directly to this problem risks mere "guessing" without engineering principles. We present a "synergistic reasoning" framework that integrates an LLM's strategic reasoning with the physical precision of the gm/Id methodology. By empowering the LLM with gm/Id lookup tables, it becomes a quantitative, data-driven design partner.
We validated this on a two-stage op-amp, where our framework enabled the Gemini model to meet all TT corner specs in 5 iterations and extended optimization to all PVT corners. A crucial ablation study proved gm/Id data is key for this efficiency and precision; without it, the LLM is slower and deviates. Compared to a senior engineer's design, our framework achieves quasi-expert quality with an order-of-magnitude improvement in efficiency. This work validates a path for true analog design automation by combining LLM reasoning with scientific circuit design methodologies. 

**Abstract (ZH)**: 模拟IC设计因依赖经验和低效的仿真而在先进节点中成为瓶颈，传统的公式在高级节点中失效。直接将大规模语言模型（LLMs）应用于此问题可能会导致无工程原理的“猜测”。我们提出了一种“协同推理”框架，将LLM的战略推理与gm/Id方法的物理精确性相结合。通过赋予LLM gm/Id查找表，它成为定量的数据驱动设计伙伴。

我们在一个两级运算放大器上进行了验证，其中我们的框架使Gemini模型在5次迭代中满足所有TT角规格，并将优化扩展到所有PVT角。关键的消融研究证明gm/Id数据对于这种效率和精度是关键的；没有它，LLM会更慢并偏离。与资深工程师的设计相比，我们的框架以数量级提高的效率实现了接近专家级的质量。本工作验证了通过结合LLM推理和科学电路设计方法来实现真正模拟设计自动化的路径。 

---
# EvoVerilog: Large Langugage Model Assisted Evolution of Verilog Code 

**Title (ZH)**: EvoVerilog: 大型语言模型辅助的Verilog代码进化 

**Authors**: Ping Guo, Yiting Wang, Wanghao Ye, Yexiao He, Ziyao Wang, Xiaopeng Dai, Ang Li, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13156)  

**Abstract**: Large Language Models (LLMs) have demonstrated great potential in automating the generation of Verilog hardware description language code for hardware design. This automation is critical to reducing human effort in the complex and error-prone process of hardware design.
However, existing approaches predominantly rely on human intervention and fine-tuning using curated datasets, limiting their scalability in automated design workflows.
Although recent iterative search techniques have emerged, they often fail to explore diverse design solutions and may underperform simpler approaches such as repeated prompting.
To address these limitations, we introduce EvoVerilog, a novel framework that combines the reasoning capabilities of LLMs with evolutionary algorithms to automatically generate and refine Verilog code.
EvoVerilog utilizes a multiobjective, population-based search strategy to explore a wide range of design possibilities without requiring human intervention.
Extensive experiments demonstrate that EvoVerilog achieves state-of-the-art performance, with pass@10 scores of 89.1 and 80.2 on the VerilogEval-Machine and VerilogEval-Human benchmarks, respectively. Furthermore, the framework showcases its ability to explore diverse designs by simultaneously generating a variety of functional Verilog code while optimizing resource utilization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化生成Verilog硬件描述语言代码方面展现了巨大的潜力，这对于减少复杂且易出错的硬件设计过程中的人力投入至关重要。然而，现有方法主要依赖于人工干预和精心打造的数据集微调，这限制了它们在自动化设计流程中的可扩展性。尽管最近出现了迭代搜索技术，但它们往往未能探索多样化的设计解决方案，并且可能不如简单的重复提示方法表现得更好。为了解决这些限制，我们引入了EvoVerilog，这是一种结合了LLMs推理能力与进化算法的新型框架，用于自动生成和优化Verilog代码。EvoVerilog利用多目标、基于群体的搜索策略，在无需人工干预的情况下探索广泛的设计可能性。广泛的实验表明，EvoVerilog在VerilogEval-Machine和VerilogEval-Human基准测试中的@10通过率分别为89.1和80.2，表现最佳。此外，该框架展示了其探索多样化设计的能力，同时生成多种功能性Verilog代码并优化资源利用率。 

---
# Uncovering Emergent Physics Representations Learned In-Context by Large Language Models 

**Title (ZH)**: 揭示大语言模型在上下文中学到的 emergent 物理表示 

**Authors**: Yeongwoo Song, Jaeyong Bae, Dong-Kyum Kim, Hawoong Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2508.12448)  

**Abstract**: Large language models (LLMs) exhibit impressive in-context learning (ICL) abilities, enabling them to solve wide range of tasks via textual prompts alone. As these capabilities advance, the range of applicable domains continues to expand significantly. However, identifying the precise mechanisms or internal structures within LLMs that allow successful ICL across diverse, distinct classes of tasks remains elusive. Physics-based tasks offer a promising testbed for probing this challenge. Unlike synthetic sequences such as basic arithmetic or symbolic equations, physical systems provide experimentally controllable, real-world data based on structured dynamics grounded in fundamental principles. This makes them particularly suitable for studying the emergent reasoning behaviors of LLMs in a realistic yet tractable setting. Here, we mechanistically investigate the ICL ability of LLMs, especially focusing on their ability to reason about physics. Using a dynamics forecasting task in physical systems as a proxy, we evaluate whether LLMs can learn physics in context. We first show that the performance of dynamics forecasting in context improves with longer input contexts. To uncover how such capability emerges in LLMs, we analyze the model's residual stream activations using sparse autoencoders (SAEs). Our experiments reveal that the features captured by SAEs correlate with key physical variables, such as energy. These findings demonstrate that meaningful physical concepts are encoded within LLMs during in-context learning. In sum, our work provides a novel case study that broadens our understanding of how LLMs learn in context. 

**Abstract (ZH)**: 大型语言模型(Large Language Models, LLMs)在上下文学习(In-Context Learning, ICL)方面表现出色，能够仅通过文本提示解决广泛的任务。随着这些能力的进步，适用的领域范围也在不断扩大。然而，确定LLMs内部何种机制或结构允许其在不同类别的任务中成功进行ICL的具体机制仍然难以捉摸。基于物理的任务为探查这一挑战提供了有前景的测试平台。与基本算术或符号方程等合成序列不同，物理系统提供了基于基本原理和结构化动力学的可实验控制的真实世界数据。这使它们特别适合在实际可行的环境中研究LLMs的 emergent 推理行为。在此，我们从机理上探究了LLMs的ICL能力，特别是它们处理物理问题的能力。我们使用物理系统中的动力学预测任务作为代理，评估LLMs是否能在上下文中学习物理知识。我们首先展示了随着输入上下文的延长，动力学预测的性能有所提高。为了探索这种能力在LLMs中是如何产生的，我们使用稀疏自编码器(Sparse Autoencoders, SAEs)分析模型的残差流激活。实验结果表明，SAEs捕获的特征与关键物理变量，如能量，相关。这些发现表明，在上下文学习过程中，LLMs中编码了有意义的物理概念。总之，我们的研究提供了关于LLMs如何进行上下文学习的新颖案例研究，拓宽了我们对其理解。 

---
# TaoSR1: The Thinking Model for E-commerce Relevance Search 

**Title (ZH)**: TaoSR1：电子商务相关搜索的思维模型 

**Authors**: Chenhe Dong, Shaowei Yao, Pengkun Jiao, Jianhui Yang, Yiming Jin, Zerui Huang, Xiaojiang Zhou, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12365)  

**Abstract**: Query-product relevance prediction is a core task in e-commerce search. BERT-based models excel at semantic matching but lack complex reasoning capabilities. While Large Language Models (LLMs) are explored, most still use discriminative fine-tuning or distill to smaller models for deployment. We propose a framework to directly deploy LLMs for this task, addressing key challenges: Chain-of-Thought (CoT) error accumulation, discriminative hallucination, and deployment feasibility. Our framework, TaoSR1, involves three stages: (1) Supervised Fine-Tuning (SFT) with CoT to instill reasoning; (2) Offline sampling with a pass@N strategy and Direct Preference Optimization (DPO) to improve generation quality; and (3) Difficulty-based dynamic sampling with Group Relative Policy Optimization (GRPO) to mitigate discriminative hallucination. Additionally, post-CoT processing and a cumulative probability-based partitioning method enable efficient online deployment. TaoSR1 significantly outperforms baselines on offline datasets and achieves substantial gains in online side-by-side human evaluations, introducing a novel paradigm for applying CoT reasoning to relevance classification. 

**Abstract (ZH)**: Query-product relevance prediction is a core-commerce core task. BERT-based models excel in semantic e but lack complex reasoning capability. e Large Language Models (LLMs) e explored e used use use discriminative fine fin small to smaller Small framework propose a framework to directly deploy LLM for e e e addressing e challenges: Chain-of-Thought ( eT e addition, discrimin e hallucination e and eployability feasibility e E e e velved e e e three e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e erfolgreich em e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e.e 

---
