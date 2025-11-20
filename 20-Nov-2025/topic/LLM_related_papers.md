# Know Your Intent: An Autonomous Multi-Perspective LLM Agent Framework for DeFi User Transaction Intent Mining 

**Title (ZH)**: 了解你的意图：一种自主多视角LLM代理框架用于DeFi用户交易意图挖掘 

**Authors**: Qian'ang Mao, Yuxuan Zhang, Jiaman Chen, Wenjun Zhou, Jiaqi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.15456)  

**Abstract**: As Decentralized Finance (DeFi) develops, understanding user intent behind DeFi transactions is crucial yet challenging due to complex smart contract interactions, multifaceted on-/off-chain factors, and opaque hex logs. Existing methods lack deep semantic insight. To address this, we propose the Transaction Intent Mining (TIM) framework. TIM leverages a DeFi intent taxonomy built on grounded theory and a multi-agent Large Language Model (LLM) system to robustly infer user intents. A Meta-Level Planner dynamically coordinates domain experts to decompose multiple perspective-specific intent analyses into solvable subtasks. Question Solvers handle the tasks with multi-modal on/off-chain data. While a Cognitive Evaluator mitigates LLM hallucinations and ensures verifiability. Experiments show that TIM significantly outperforms machine learning models, single LLMs, and single Agent baselines. We also analyze core challenges in intent inference. This work helps provide a more reliable understanding of user motivations in DeFi, offering context-aware explanations for complex blockchain activity. 

**Abstract (ZH)**: 面向DeFi交易的意图挖掘框架：应对复杂智能合约交互与多维度因素的挑战 

---
# SOLID: a Framework of Synergizing Optimization and LLMs for Intelligent Decision-Making 

**Title (ZH)**: SOLID：协同优化和大语言模型的智能决策框架 

**Authors**: Yinsheng Wang, Tario G You, Léonard Boussioux, Shan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15202)  

**Abstract**: This paper introduces SOLID (Synergizing Optimization and Large Language Models for Intelligent Decision-Making), a novel framework that integrates mathematical optimization with the contextual capabilities of large language models (LLMs). SOLID facilitates iterative collaboration between optimization and LLMs agents through dual prices and deviation penalties. This interaction improves the quality of the decisions while maintaining modularity and data privacy. The framework retains theoretical convergence guarantees under convexity assumptions, providing insight into the design of LLMs prompt. To evaluate SOLID, we applied it to a stock portfolio investment case with historical prices and financial news as inputs. Empirical results demonstrate convergence under various scenarios and indicate improved annualized returns compared to a baseline optimizer-only method, validating the synergy of the two agents. SOLID offers a promising framework for advancing automated and intelligent decision-making across diverse domains. 

**Abstract (ZH)**: SOLID (Synergizing Optimization and Large Language Models for Intelligent Decision-Making) 

---
# As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files 

**Title (ZH)**: 如曾相识：大型语言模型在识别见过的文件时表现出确定性 

**Authors**: Haodong Li, Jingqi Zhang, Xiao Cheng, Peihua Mai, Haoyu Wang, Yang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.15192)  

**Abstract**: The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds.
We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency. 

**Abstract (ZH)**: 大型语言模型（LLMs）卓越的语言能力源于其对大量数据集的广泛训练，这些数据集通常包括受版权保护的材料，这引发了关于未经授权使用材料的严重关切。虽然成员推测推理攻击（MIAs）提供了检测此类违规行为的潜在解决方案，但现有方法由于LLMs固有的过度自信、有限的真实训练数据访问以及依赖于经验确定的阈值而面临关键的局限性和挑战。

我们提出了COPYCHECK，一个新颖的框架，利用不确定性信号来检测LLM训练集是否使用了受版权保护的内容。我们的方法将LLM的过度自信从一种限制转变为一种资产，通过捕获可靠的不确定性模式来区分“已见”（训练数据）和“未见”（非训练数据）内容。COPYCHECK进一步实施了两步策略：（1）文件的策略性分割成更小片段以减少对大规模训练数据的依赖，（2）基于不确定性的无监督聚类以消除对经验调优阈值的需求。实验结果表明，COPYCHECK在检测已见文件方面分别在LLaMA 7b和LLaMA2 7b上实现了平均平衡准确率90.1%和91.6%。与现有最佳基线相比，COPYCHECK实现了超过90%的相对改进，最高可达93.8%的平衡准确率。此外，COPYCHECK在不同架构上表现出良好的通用性，在GPT-J 6B上仍能保持高性能。这项工作首次将不确定性应用于LLMs中的版权检测，提供了一种提高训练数据透明度的实际工具。 

---
# HISE-KT: Synergizing Heterogeneous Information Networks and LLMs for Explainable Knowledge Tracing with Meta-Path Optimization 

**Title (ZH)**: HISE-KT：异质信息网络与元路径优化结合的可解释知识追踪方法与大型语言模型协同使用 

**Authors**: Zhiyi Duan, Zixing Shi, Hongyu Yuan, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15191)  

**Abstract**: Knowledge Tracing (KT) aims to mine students' evolving knowledge states and predict their future question-answering performance. Existing methods based on heterogeneous information networks (HINs) are prone to introducing noises due to manual or random selection of meta-paths and lack necessary quality assessment of meta-path instances. Conversely, recent large language models (LLMs)-based methods ignore the rich information across students, and both paradigms struggle to deliver consistently accurate and evidence-based explanations. To address these issues, we propose an innovative framework, HIN-LLM Synergistic Enhanced Knowledge Tracing (HISE-KT), which seamlessly integrates HINs with LLMs. HISE-KT first builds a multi-relationship HIN containing diverse node types to capture the structural relations through multiple meta-paths. The LLM is then employed to intelligently score and filter meta-path instances and retain high-quality paths, pioneering automated meta-path quality assessment. Inspired by educational psychology principles, a similar student retrieval mechanism based on meta-paths is designed to provide a more valuable context for prediction. Finally, HISE-KT uses a structured prompt to integrate the target student's history with the retrieved similar trajectories, enabling the LLM to generate not only accurate predictions but also evidence-backed, explainable analysis reports. Experiments on four public datasets show that HISE-KT outperforms existing KT baselines in both prediction performance and interpretability. 

**Abstract (ZH)**: HIN-LLM协同增强知识追踪（HISE-KT） 

---
# SafeRBench: A Comprehensive Benchmark for Safety Assessment in Large Reasoning Models 

**Title (ZH)**: SafeRBench: 一种全面的大型推理模型安全性评估基准 

**Authors**: Xin Gao, Shaohan Yu, Zerui Chen, Yueming Lyu, Weichen Yu, Guanghao Li, Jiyao Liu, Jianxiong Gao, Jian Liang, Ziwei Liu, Chenyang Si  

**Link**: [PDF](https://arxiv.org/pdf/2511.15169)  

**Abstract**: Large Reasoning Models (LRMs) improve answer quality through explicit chain-of-thought, yet this very capability introduces new safety risks: harmful content can be subtly injected, surface gradually, or be justified by misleading rationales within the reasoning trace. Existing safety evaluations, however, primarily focus on output-level judgments and rarely capture these dynamic risks along the reasoning process. In this paper, we present SafeRBench, the first benchmark that assesses LRM safety end-to-end -- from inputs and intermediate reasoning to final outputs. (1) Input Characterization: We pioneer the incorporation of risk categories and levels into input design, explicitly accounting for affected groups and severity, and thereby establish a balanced prompt suite reflecting diverse harm gradients. (2) Fine-Grained Output Analysis: We introduce a micro-thought chunking mechanism to segment long reasoning traces into semantically coherent units, enabling fine-grained evaluation across ten safety dimensions. (3) Human Safety Alignment: We validate LLM-based evaluations against human annotations specifically designed to capture safety judgments. Evaluations on 19 LRMs demonstrate that SafeRBench enables detailed, multidimensional safety assessment, offering insights into risks and protective mechanisms from multiple perspectives. 

**Abstract (ZH)**: 大型 reasoning 模型 (LRMs) 通过明确的推理链提高答案质量，但这一能力也引入了新的安全性风险：有害内容可以隐秘插入，表面逐渐显现，或通过误导性的推理在推理过程中被正当化。现有的安全性评估主要侧重于输出层面的判断，很少能够捕捉到这些动态风险。在本文中，我们提出了 SafeRBench，这是首个从输入到中间推理再到最终输出全面评估LRM安全性的基准。(1) 输入特征化：我们首创将风险类别和级别纳入输入设计，明确考虑受影响的群体和严重程度，从而建立一个反映不同危害梯度的平衡提示集。(2) 细粒度输出分析：我们引入了微推理片段化机制，将长推理链段分为语义上一致的单元，以便在十个安全维度上进行细粒度评估。(3) 人类安全对齐：我们用专门设计的捕捉安全判断的人工标注来验证基于预训练语言模型的安全性评估。在19个LRMs上的评估表明，SafeRBench 使详细的多维度安全性评估成为可能，提供了多视角的风险和保护机制的洞见。 

---
# Knowledge-Informed Automatic Feature Extraction via Collaborative Large Language Model Agents 

**Title (ZH)**: 基于知识引导的自动特征提取协作大语言模型代理 

**Authors**: Henrik Bradland, Morten Goodwin, Vladimir I. Zadorozhny, Per-Arne Andersen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15074)  

**Abstract**: The performance of machine learning models on tabular data is critically dependent on high-quality feature engineering. While Large Language Models (LLMs) have shown promise in automating feature extraction (AutoFE), existing methods are often limited by monolithic LLM architectures, simplistic quantitative feedback, and a failure to systematically integrate external domain knowledge. This paper introduces Rogue One, a novel, LLM-based multi-agent framework for knowledge-informed automatic feature extraction. Rogue One operationalizes a decentralized system of three specialized agents-Scientist, Extractor, and Tester-that collaborate iteratively to discover, generate, and validate predictive features. Crucially, the framework moves beyond primitive accuracy scores by introducing a rich, qualitative feedback mechanism and a "flooding-pruning" strategy, allowing it to dynamically balance feature exploration and exploitation. By actively incorporating external knowledge via an integrated retrieval-augmented (RAG) system, Rogue One generates features that are not only statistically powerful but also semantically meaningful and interpretable. We demonstrate that Rogue One significantly outperforms state-of-the-art methods on a comprehensive suite of 19 classification and 9 regression datasets. Furthermore, we show qualitatively that the system surfaces novel, testable hypotheses, such as identifying a new potential biomarker in the myocardial dataset, underscoring its utility as a tool for scientific discovery. 

**Abstract (ZH)**: 基于大型语言模型的多智能体知识导向自动特征提取框架Rogue One 

---
# ProRAC: A Neuro-symbolic Method for Reasoning about Actions with LLM-based Progression 

**Title (ZH)**: ProRAC: 一种基于LLM的神经符号方法，用于动作推理 

**Authors**: Haoyong Wu, Yongmei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15069)  

**Abstract**: In this paper, we propose ProRAC (Progression-based Reasoning about Actions and Change), a neuro-symbolic framework that leverages LLMs to tackle RAC problems. ProRAC extracts fundamental RAC elements including actions and questions from the problem, progressively executes each action to derive the final state, and then evaluates the query against the progressed state to arrive at an answer. We evaluate ProRAC on several RAC benchmarks, and the results demonstrate that our approach achieves strong performance across different benchmarks, domains, LLM backbones, and types of RAC tasks. 

**Abstract (ZH)**: 基于进程的行动与变化推理：一种利用大语言模型的神经符号框架 

---
# Beyond GeneGPT: A Multi-Agent Architecture with Open-Source LLMs for Enhanced Genomic Question Answering 

**Title (ZH)**: 超越GeneGPT：一种基于开源大语言模型的多代理架构，用于增强基因组问答 

**Authors**: Haodong Chen, Guido Zuccon, Teerapong Leelanupab  

**Link**: [PDF](https://arxiv.org/pdf/2511.15061)  

**Abstract**: Genomic question answering often requires complex reasoning and integration across diverse biomedical sources. GeneGPT addressed this challenge by combining domain-specific APIs with OpenAI's code-davinci-002 large language model to enable natural language interaction with genomic databases. However, its reliance on a proprietary model limits scalability, increases operational costs, and raises concerns about data privacy and generalization.
In this work, we revisit and reproduce GeneGPT in a pilot study using open source models, including Llama 3.1, Qwen2.5, and Qwen2.5 Coder, within a monolithic architecture; this allows us to identify the limitations of this approach. Building on this foundation, we then develop OpenBioLLM, a modular multi-agent framework that extends GeneGPT by introducing agent specialization for tool routing, query generation, and response validation. This enables coordinated reasoning and role-based task execution.
OpenBioLLM matches or outperforms GeneGPT on over 90% of the benchmark tasks, achieving average scores of 0.849 on Gene-Turing and 0.830 on GeneHop, while using smaller open-source models without additional fine-tuning or tool-specific pretraining. OpenBioLLM's modular multi-agent design reduces latency by 40-50% across benchmark tasks, significantly improving efficiency without compromising model capability. The results of our comprehensive evaluation highlight the potential of open-source multi-agent systems for genomic question answering. Code and resources are available at this https URL. 

**Abstract (ZH)**: 基于开源模型的OpenBioLLM：一种模块化的多代理框架用于基因组问答 

---
# Subnational Geocoding of Global Disasters Using Large Language Models 

**Title (ZH)**: 使用大型语言模型进行全球灾害的亚国家级地理编码 

**Authors**: Michele Ronco, Damien Delforge, Wiebke S. Jäger, Christina Corbane  

**Link**: [PDF](https://arxiv.org/pdf/2511.14788)  

**Abstract**: Subnational location data of disaster events are critical for risk assessment and disaster risk reduction. Disaster databases such as EM-DAT often report locations in unstructured textual form, with inconsistent granularity or spelling, that make it difficult to integrate with spatial datasets. We present a fully automated LLM-assisted workflow that processes and cleans textual location information using GPT-4o, and assigns geometries by cross-checking three independent geoinformation repositories: GADM, OpenStreetMap and Wikidata. Based on the agreement and availability of these sources, we assign a reliability score to each location while generating subnational geometries. Applied to the EM-DAT dataset from 2000 to 2024, the workflow geocodes 14,215 events across 17,948 unique locations. Unlike previous methods, our approach requires no manual intervention, covers all disaster types, enables cross-verification across multiple sources, and allows flexible remapping to preferred frameworks. Beyond the dataset, we demonstrate the potential of LLMs to extract and structure geographic information from unstructured text, offering a scalable and reliable method for related analyses. 

**Abstract (ZH)**: 次国家层次的灾难事件地理位置数据对于风险评估和灾害风险管理至关重要。EM-DAT等灾难数据库往往以不结构化的文本形式报告地理位置，且粒度不一或拼写不规范，这使得与空间数据集整合困难重重。我们提出了一种完全自动化的基于LLM的工作流，利用GPT-4o处理和清洗文本地理位置信息，并通过交叉检查三个独立的地理信息仓库（GADM、OpenStreetMap和Wikidata）分配几何位置。根据这些来源的一致性和可用性，我们在生成次国家层次几何位置的同时为每个位置打上可靠性评分。将该工作流应用于2000年至2024年的EM-DAT数据集，共处理和地理编码了14,215个事件，覆盖17,948个独特的地理位置。与以往方法不同，我们的方法无需人工干预，适用于所有类型的灾难，能够跨多个来源进行交叉验证，并允许灵活地重新映射到首选框架。除此之外，我们还展示了LLM提取和结构化不结构化文本中的地理信息的潜力，提供了一种可扩展且可靠的相关分析方法。 

---
# Ask WhAI:Probing Belief Formation in Role-Primed LLM Agents 

**Title (ZH)**: Ask WhAI: 探究角色引导的大语言模型代理的信念形成过程 

**Authors**: Keith Moore, Jun W. Kim, David Lyu, Jeffrey Heo, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2511.14780)  

**Abstract**: We present Ask WhAI, a systems-level framework for inspecting and perturbing belief states in multi-agent interactions. The framework records and replays agent interactions, supports out-of-band queries into each agent's beliefs and rationale, and enables counterfactual evidence injection to test how belief structures respond to new information. We apply the framework to a medical case simulator notable for its multi-agent shared memory (a time-stamped electronic medical record, or EMR) and an oracle agent (the LabAgent) that holds ground truth lab results revealed only when explicitly queried. We stress-test the system on a multi-specialty diagnostic journey for a child with an abrupt-onset neuropsychiatric presentation. Large language model agents, each primed with strong role-specific priors ("act like a neurologist", "act like an infectious disease specialist"), write to a shared medical record and interact with a moderator across sequential or parallel encounters. Breakpoints at key diagnostic moments enable pre- and post-event belief queries, allowing us to distinguish entrenched priors from reasoning or evidence-integration effects. The simulation reveals that agent beliefs often mirror real-world disciplinary stances, including overreliance on canonical studies and resistance to counterevidence, and that these beliefs can be traced and interrogated in ways not possible with human experts. By making such dynamics visible and testable, Ask WhAI offers a reproducible way to study belief formation and epistemic silos in multi-agent scientific reasoning. 

**Abstract (ZH)**: Ask WhAI：多智能体交互中的信念状态检验与扰动系统级框架 

---
# The Illusion of Procedural Reasoning: Measuring Long-Horizon FSM Execution in LLMs 

**Title (ZH)**: 程序推理的幻象：测量LLM中长时序_fsm执行 

**Authors**: Mahdi Samiei, Mahdi Mansouri, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2511.14777)  

**Abstract**: Large language models (LLMs) have achieved remarkable results on tasks framed as reasoning problems, yet their true ability to perform procedural reasoning, executing multi-step, rule-based computations remains unclear. Unlike algorithmic systems, which can deterministically execute long-horizon symbolic procedures, LLMs often degrade under extended reasoning chains, but there is no controlled, interpretable benchmark to isolate and measure this collapse. We introduce Finite-State Machine (FSM) Execution as a minimal, fully interpretable framework for evaluating the procedural reasoning capacity of LLMs. In our setup, the model is given an explicit FSM definition and must execute it step-by-step given input actions, maintaining state consistency over multiple turns. This task requires no world knowledge, only faithful application of deterministic transition rules, making it a direct probe of the model's internal procedural fidelity. We measure both Turn Accuracy and Task Accuracy to disentangle immediate computation from cumulative state maintenance. Empirical results reveal systematic degradation as task horizon or branching complexity increases. Models perform significantly worse when rule retrieval involves high branching factors than when memory span is long. Larger models show improved local accuracy but remain brittle under multi-step reasoning unless explicitly prompted to externalize intermediate steps. FSM-based evaluation offers a transparent, complexity-controlled probe for diagnosing this failure mode and guiding the design of inductive biases that enable genuine long-horizon procedural competence. By grounding reasoning in measurable execution fidelity rather than surface correctness, this work helps establish a rigorous experimental foundation for understanding and improving the algorithmic reliability of LLMs. 

**Abstract (ZH)**: 大型语言模型在作为推理问题的任务中取得了显著成果，但其真正执行程序推理的能力，即执行多步规则驱动计算的能力仍然不清楚。与可以确定性执行长时间符号程序的算法系统不同，大型语言模型在扩展的推理链中往往会表现不佳，但没有可控且可解释的基准来隔离和量度这种表现下降。我们引入有限状态机（FSM）执行作为评估大型语言模型程序推理容量的最小化、完全可解释框架。在我们的设置中，模型获得一个明确的FSM定义，并且必须在给定输入动作的情况下，逐步执行它，保持多次轮次中的状态一致性。该任务不需要世界知识，只需要忠实地应用确定性转换规则，使其成为对模型内部程序准确性的直接检测。我们测量轮次准确性和任务准确性以分离即时计算和累积状态维护。实证结果揭示了随着任务视距或分支复杂性的增加呈现出系统性的下降。当规则检索涉及高分支因子时，模型的表现显著较差，而在记忆跨度较长时则表现较好。较大的模型在局部准确性方面有所提高，但在多步推理中仍然脆弱，除非明确提示其外部化中间步骤。基于有限状态机的评估提供了一个透明且具有复杂度控制的检测工具，用于诊断这种失败模式并指导能够真正实现长期程序能力的归纳偏差的设计。通过将推理基于可量化的执行保真度而非表面正确性，这项工作有助于建立理解并改进大型语言模型算法可靠性的严谨实验基础。 

---
# Walrus: A Cross-Domain Foundation Model for Continuum Dynamics 

**Title (ZH)**: Walrus: 一种用于连续动力学的跨域基础模型 

**Authors**: Michael McCabe, Payel Mukhopadhyay, Tanya Marwah, Bruno Regaldo-Saint Blancard, Francois Rozet, Cristiana Diaconu, Lucas Meyer, Kaze W. K. Wong, Hadi Sotoudeh, Alberto Bietti, Irina Espejo, Rio Fear, Siavash Golkar, Tom Hehir, Keiya Hirashima, Geraud Krawezik, Francois Lanusse, Rudy Morel, Ruben Ohana, Liam Parker, Mariel Pettee, Jeff Shen, Kyunghyun Cho, Miles Cranmer, Shirley Ho  

**Link**: [PDF](https://arxiv.org/pdf/2511.15684)  

**Abstract**: Foundation models have transformed machine learning for language and vision, but achieving comparable impact in physical simulation remains a challenge. Data heterogeneity and unstable long-term dynamics inhibit learning from sufficiently diverse dynamics, while varying resolutions and dimensionalities challenge efficient training on modern hardware. Through empirical and theoretical analysis, we incorporate new approaches to mitigate these obstacles, including a harmonic-analysis-based stabilization method, load-balanced distributed 2D and 3D training strategies, and compute-adaptive tokenization. Using these tools, we develop Walrus, a transformer-based foundation model developed primarily for fluid-like continuum dynamics. Walrus is pretrained on nineteen diverse scenarios spanning astrophysics, geoscience, rheology, plasma physics, acoustics, and classical fluids. Experiments show that Walrus outperforms prior foundation models on both short and long term prediction horizons on downstream tasks and across the breadth of pretraining data, while ablation studies confirm the value of our contributions to forecast stability, training throughput, and transfer performance over conventional approaches. Code and weights are released for community use. 

**Abstract (ZH)**: 基础模型已转型语言和视觉领域的机器学习，但在物理模拟中的应用仍面临挑战。数据异质性和不稳定的长期动力学阻碍了对足够多样动力学的学习，而不同的分辨率和维度性给现代硬件上的高效训练带来了挑战。通过实证和理论分析，我们引入了新的方法来缓解这些障碍，包括基于谐波分析的稳定化方法、负载均衡的分布式2D和3D训练策略以及计算自适应的分词方法。利用这些工具，我们开发了Walrus，一种主要用于流体-like连续动力学的变压器基础模型。Walrus在天体物理学、地球科学、流变学、等离子体物理、声学和经典流体等十九个不同场景下进行预训练。实验表明，Walrus在短时间和长时间预测窗口以及预训练数据跨度上均优于先前的基础模型，且消融研究表明我们的贡献对预测稳定性、训练吞吐量和迁移性能具有重要价值。社区可以获取代码和权重。 

---
# VisPlay: Self-Evolving Vision-Language Models from Images 

**Title (ZH)**: VisPlay: 自我进化 vision-Language 模型从图像演变而来 

**Authors**: Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, Yonghui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15661)  

**Abstract**: Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained with Group Relative Policy Optimization (GRPO), which incorporates diversity and difficulty rewards to balance the complexity of generated questions with the quality of the silver answers. VisPlay scales efficiently across two model families. When trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks, including MM-Vet and MMMU, demonstrating a scalable path toward self-evolving multimodal intelligence. The project page is available at this https URL 

**Abstract (ZH)**: 强化学习（RL）为提高视觉语言模型（VLMs）在复杂推理任务中的性能提供了原则性的框架。然而，现有的RL方法往往依赖于人工标注的标签或特定任务的启发式方法来定义可验证的奖励，这两种方法都成本高昂且难以扩展。我们介绍了VisPlay，这是一种自我进化的RL框架，使VLMs能够利用大量未标注的图像数据自主提高其推理能力。从一个基模型开始，VisPlay将模型分配为两个相互作用的角色：一个基于图像的问题提出者，负责提出具有挑战性但可回答的视觉问题；以及一个跨模态推理器，负责生成银质回答。这些角色通过结合多样性和难度奖励的组相对策略优化（GRPO）进行联合训练，以平衡生成问题的复杂性和银质回答的质量。VisPlay在两种模型家族中都能高效扩展。在Qwen2.5-VL和MiMo-VL上进行训练后，VisPlay在包括MM-Vet和MMMU在内的八个基准测试中实现了视觉推理、组合泛化和幻觉降低的持续改进，展示了通向自我进化的跨模态智能的可扩展路径。项目页面可访问此链接。 

---
# HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning 

**Title (ZH)**: HSKBenchmark: 通过课程调优在大型语言模型中建模与基准测试汉语二外习得 

**Authors**: Qihao Yang, Xuelin Wang, Jiale Chen, Xuelian Dong, Yuxin Hao, Tianyong Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15574)  

**Abstract**: Language acquisition is vital to revealing the nature of human language intelligence and has recently emerged as a promising perspective for improving the interpretability of large language models (LLMs). However, it is ethically and practically infeasible to conduct experiments that require controlling human learners' language inputs. This poses challenges for the verifiability and scalability of language acquisition modeling, particularly in Chinese second language acquisition (SLA). While LLMs provide a controllable and reproducible alternative, a systematic benchmark to support phase-wise modeling and assessment is still lacking. In this paper, we present HSKBenchmark, the first benchmark for staged modeling and writing assessment of LLMs in Chinese SLA. It covers HSK levels 3 to 6 and includes authentic textbooks with 6.76 million tokens, 16K synthetic instruction samples, 30 test topics, and a linguistically grounded evaluation system. To simulate human learning trajectories, we introduce a curriculum-tuning framework that trains models from beginner to advanced levels. An evaluation system is created to examine level-based grammar coverage, writing errors, lexical and syntactic complexity, and holistic scoring. We also build HSKAgent, fine-tuned on 10K learner compositions. Extensive experimental results demonstrate that HSKBenchmark not only models Chinese SLA effectively, but also serves as a reliable benchmark for dynamic writing assessment in LLMs. Our fine-tuned LLMs have writing performance on par with advanced human learners and exhibit human-like acquisition characteristics. The HSKBenchmark, HSKAgent, and checkpoints serve as foundational tools and resources, with the potential to pave the way for future research on language acquisition modeling and LLMs interpretability. Code and data are publicly available at: this https URL. 

**Abstract (ZH)**: HSKBenchmark：中文二语习得分阶段建模与写作评估基准 

---
# Multimodal Evaluation of Russian-language Architectures 

**Title (ZH)**: 俄语架构的多模态评估 

**Authors**: Artem Chervyakov, Ulyana Isaeva, Anton Emelyanov, Artem Safin, Maria Tikhonova, Alexander Kharitonov, Yulia Lyakh, Petr Surovtsev, Denis Shevelev Vildan Saburov, Vasily Konovalov, Elisei Rykov, Ivan Sviridov, Amina Miftakhova, Ilseyar Alimova, Alexander Panchenko, Alexander Kapitanov, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2511.15552)  

**Abstract**: Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce Mera Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (image-to-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking and licenses for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）目前是研究焦点，显示出在规模和能力上的迅速进展，但其智能性、限制和风险仍不够了解。为应对这些问题，特别是在目前俄语领域缺乏多模态基准的情况下，我们引入了Mera Multi，一个针对俄语架构的开放多模态评估框架。该基准基于指令，包括默认的文字、图像、音频和视频模态，共有18项新的评估任务，适用于通用模型和模态特定架构（图像到文本、视频到文本和音频到文本）。我们的贡献包括：(i) 一个多模态能力的通用分类学；(ii) 18个从零开始创建的数据集，注意俄语文化与语言的特殊性，统一的提示和评估指标；(iii) 专源和开源模型的基线结果；(iv) 防止基准泄漏的方法，包括水印和私有集的许可证。虽然我们的当前重点是俄语，但所提出的基准为在不同类型的语言中构建多模态基准提供了可复制的方法，特别是斯拉夫语族语言。 

---
# Small Language Models for Phishing Website Detection: Cost, Performance, and Privacy Trade-Offs 

**Title (ZH)**: 小型语言模型在钓鱼网站检测中的成本、性能与隐私权衡 

**Authors**: Georg Goldenits, Philip Koenig, Sebastian Raubitzek, Andreas Ekelhart  

**Link**: [PDF](https://arxiv.org/pdf/2511.15434)  

**Abstract**: Phishing websites pose a major cybersecurity threat, exploiting unsuspecting users and causing significant financial and organisational harm. Traditional machine learning approaches for phishing detection often require extensive feature engineering, continuous retraining, and costly infrastructure maintenance. At the same time, proprietary large language models (LLMs) have demonstrated strong performance in phishing-related classification tasks, but their operational costs and reliance on external providers limit their practical adoption in many business environments. This paper investigates the feasibility of small language models (SLMs) for detecting phishing websites using only their raw HTML code. A key advantage of these models is that they can be deployed on local infrastructure, providing organisations with greater control over data and operations. We systematically evaluate 15 commonly used Small Language Models (SLMs), ranging from 1 billion to 70 billion parameters, benchmarking their classification accuracy, computational requirements, and cost-efficiency. Our results highlight the trade-offs between detection performance and resource consumption, demonstrating that while SLMs underperform compared to state-of-the-art proprietary LLMs, they can still provide a viable and scalable alternative to external LLM services. By presenting a comparative analysis of costs and benefits, this work lays the foundation for future research on the adaptation, fine-tuning, and deployment of SLMs in phishing detection systems, aiming to balance security effectiveness and economic practicality. 

**Abstract (ZH)**: 小型语言模型在仅使用原始HTML代码检测钓鱼网站中的可行性研究 

---
# NAMeGEn: Creative Name Generation via A Novel Agent-based Multiple Personalized Goal Enhancement Framework 

**Title (ZH)**: NameGEn：基于新型代理多个性化目标增强框架的创意名称生成 

**Authors**: Shanlin Zhou, Xinpeng Wang, Jianxun Lian, Zhenghao Liu, Laks V.S. Lakshmanan, Xiaoyuan Yi, Yongtao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15408)  

**Abstract**: Trained on diverse human-authored texts, Large Language Models (LLMs) unlocked the potential for Creative Natural Language Generation (CNLG), benefiting various applications like advertising and storytelling. Nevertheless, CNLG still remains difficult due to two main challenges. (1) Multi-objective flexibility: user requirements are often personalized, fine-grained, and pluralistic, which LLMs struggle to satisfy simultaneously; (2) Interpretive complexity: beyond generation, creativity also involves understanding and interpreting implicit meaning to enhance users' perception. These challenges significantly limit current methods, especially in short-form text generation, in generating creative and insightful content. To address this, we focus on Chinese baby naming, a representative short-form CNLG task requiring adherence to explicit user constraints (e.g., length, semantics, anthroponymy) while offering meaningful aesthetic explanations. We propose NAMeGEn, a novel multi-agent optimization framework that iteratively alternates between objective extraction, name generation, and evaluation to meet diverse requirements and generate accurate explanations. To support this task, we further construct a classical Chinese poetry corpus with 17k+ poems to enhance aesthetics, and introduce CBNames, a new benchmark with tailored metrics. Extensive experiments demonstrate that NAMeGEn effectively generates creative names that meet diverse, personalized requirements while providing meaningful explanations, outperforming six baseline methods spanning various LLM backbones without any training. 

**Abstract (ZH)**: 基于多样化人类撰写的文本训练，大型语言模型（LLMs）开启了创造型自然语言生成（CNLG）的潜力，惠及广告、讲故事等多种应用。然而，CNLG仍然因两大主要挑战而颇具难度。一是多目标灵活性：用户需求往往是个性化、精细且多样化的，而LLMs难以同时满足；二是解释性复杂性：创造不仅限于生成，还涉及理解并解释隐含意义以提升用户的感知。这些挑战显著限制了当前的方法，尤其是在短文本生成中生成有创意和洞察力的内容。为了解决这一问题，我们集中于中文婴儿起名这一代表性的短文本CNLG任务，该任务需要遵守明确的用户约束（如长度、语义、人名学）的同时，提供有意义的美学解释。我们提出了NAMeGEn，一种新颖的多智能体优化框架，通过迭代交替进行目标提取、名字生成和评估，以满足多样化需求并生成准确的解释。为此，我们进一步构建了一个包含17000多首经典中文诗歌的语料库以增强美学，并引入了CBNames这一新的基准测试，配备了定制化的评价指标。大量实验表明，NAMeGEn有效生成了符合多样化、个性化需求且提供有意义解释的创意名字，优于六种不同LLM模型架构的基线方法，无需任何训练。 

---
# DEPO: Dual-Efficiency Preference Optimization for LLM Agents 

**Title (ZH)**: DEPO: 双效偏好优化算法 for LLM 代理 

**Authors**: Sirui Chen, Mengshi Zhao, Lei Xu, Yuying Zhao, Beier Zhu, Hanwang Zhang, Shengjie Zhao, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15392)  

**Abstract**: Recent advances in large language models (LLMs) have greatly improved their reasoning and decision-making abilities when deployed as agents. Richer reasoning, however, often comes at the cost of longer chain of thought (CoT), hampering interaction efficiency in real-world scenarios. Nevertheless, there still lacks systematic definition of LLM agent efficiency, hindering targeted improvements. To this end, we introduce dual-efficiency, comprising (i) step-level efficiency, which minimizes tokens per step, and (ii) trajectory-level efficiency, which minimizes the number of steps to complete a task. Building on this definition, we propose DEPO, a dual-efficiency preference optimization method that jointly rewards succinct responses and fewer action steps. Experiments on WebShop and BabyAI show that DEPO cuts token usage by up to 60.9% and steps by up to 26.9%, while achieving up to a 29.3% improvement in performance. DEPO also generalizes to three out-of-domain math benchmarks and retains its efficiency gains when trained on only 25% of the data. Our project page is at this https URL. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）近年来在大型语言模型（LLMs）方面的进展大大提高了它们作为代理应用时的推理和决策能力。然而，更丰富的推理往往以更长的思维链（CoT）为代价，阻碍了在实际场景中的交互效率。尽管如此，仍然缺乏系统性的LLM代理效率定义，阻碍了有针对性的改进。为了解决这一问题，我们引入了双效率概念，包括（i）步骤级效率，即每步最小化令牌数量，以及（ii）轨迹级效率，即完成任务所需的步骤数量最小化。基于这一定义，我们提出了一种双效率偏好优化方法DEPO，该方法联合奖励简洁的回答和更少的操作步骤。在WebShop和BabyAI上的实验表明，DEPO将令牌使用量最多减少了60.9%，步骤数量最多减少了26.9%，同时实现了高达29.3%的性能提升。DEPO还能够泛化到三个不同的领域数学基准，并且在仅使用数据的25%进行训练时仍能保持其效率优势。我们的项目页面位于此链接。 

---
# A Compliance-Preserving Retrieval System for Aircraft MRO Task Search 

**Title (ZH)**: 遵守合规性的航空维修任务搜索检索系统 

**Authors**: Byungho Jo  

**Link**: [PDF](https://arxiv.org/pdf/2511.15383)  

**Abstract**: Aircraft Maintenance Technicians (AMTs) spend up to 30% of work time searching manuals, a documented efficiency bottleneck in MRO operations where every procedure must be traceable to certified sources. We present a compliance-preserving retrieval system that adapts LLM reranking and semantic search to aviation MRO environments by operating alongside, rather than replacing, certified legacy viewers. The system constructs revision-robust embeddings from ATA chapter hierarchies and uses vision-language parsing to structure certified content, allowing technicians to preview ranked tasks and access verified procedures in existing viewers. Evaluation on 49k synthetic queries achieves >90% retrieval accuracy, while bilingual controlled studies with 10 licensed AMTs demonstrate 90.9% top-10 success rate and 95% reduction in lookup time, from 6-15 minutes to 18 seconds per task. These gains provide concrete evidence that semantic retrieval can operate within strict regulatory constraints and meaningfully reduce operational workload in real-world multilingual MRO workflows. 

**Abstract (ZH)**: 航空维修技术人员（AMTs）在工作中花费高达30%的时间查询手册，这是维修运营（MRO）中的一个记录在案的效率瓶颈，因为每次操作都必须追溯到认证的来源。我们提出了一种合规保留的检索系统，该系统通过与认证的遗留查看器并行操作，而非替代它们，适应了基于LLM重排序和语义搜索的航空MRO环境。该系统从ATA章节层次结构中构建了修订稳健的嵌入，并利用视觉语言解析来结构化认证内容，使技术人员能够预览排名的任务并访问现有查看器中的验证操作程序。在49,000个合成查询上的评估实现了超过90%的检索准确性，而双语受控研究中10名持证AMT的参与表明，顶级10项成功率达到了90.9%，查找时间减少了95%，从每任务6-15分钟减少到18秒。这些收益为语义检索能够在严格的监管约束内运行并在多语言的现实世界MRO工作流程中实质性地减轻操作负担提供了确凿的证据。 

---
# The Empowerment of Science of Science by Large Language Models: New Tools and Methods 

**Title (ZH)**: 大型语言模型对科学学的赋能：新工具与方法 

**Authors**: Guoqiang Liang, Jingqian Gong, Mengxuan Li, Gege Lin, Shuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15370)  

**Abstract**: Large language models (LLMs) have exhibited exceptional capabilities in natural language understanding and generation, image recognition, and multimodal tasks, charting a course towards AGI and emerging as a central issue in the global technological race. This manuscript conducts a comprehensive review of the core technologies that support LLMs from a user standpoint, including prompt engineering, knowledge-enhanced retrieval augmented generation, fine tuning, pretraining, and tool learning. Additionally, it traces the historical development of Science of Science (SciSci) and presents a forward looking perspective on the potential applications of LLMs within the scientometric domain. Furthermore, it discusses the prospect of an AI agent based model for scientific evaluation, and presents new research fronts detection and knowledge graph building methods with LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言理解与生成、图像识别和多模态任务中表现出非凡的能力，正朝着AGI迈进并成为全球科技竞赛中的核心议题。本文从用户视角全面回顾支持LLMs的核心技术，包括提示工程、知识增强检索增强生成、微调、预训练和工具学习。此外，本文追溯了科学学（SciSci）的发展历史，并展望LLMs在科学计量领域的潜在应用。进一步地，本文讨论了基于AI代理的科学评价模型的前景，并提出了利用LLMs进行新研究前沿检测和知识图谱构建的新方法。 

---
# Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models 

**Title (ZH)**: 对抗诗文作为大型语言模型的通用单轮囚徒突变机制 

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2511.15304)  

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for large language models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of open-weight judge models and a human-validated stratified subset (with double-annotations to measure agreement). Disagreements were manually resolved. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols. 

**Abstract (ZH)**: 我们提供了证据表明对抗诗作为大型语言模型的通用单轮 jailbreak 技术。在25个前沿的商业和开源模型中，定制的诗化提示产生了较高的攻击成功率（ASR），部分提供者甚至超过了90%。将提示映射到MLCommons和EU CoP风险分类系统，显示诗化攻击在化学、生物、放射性、核生化（CBRN）、操控、网络攻击和失控等领域均有效。通过标准化元提示将1,200个MLCommons有害提示转化为诗，其攻击成功率比其散文基线高18倍。输出结果通过一组开源法官模型和人工验证的分层子集（带有双重注释以衡量一致性）进行评估。人工解决了分歧。诗化框架对手工创作的诗歌实现了平均62%的 jailbreak 成功率，而元提示转化的诗歌约为43%，显著优于非诗化基线，揭示了模型家族和安全训练方法系统性的脆弱性。这些发现表明仅通过风格变化即可规避当前的安全机制，暗示现有的对齐方法和评估协议存在根本性的局限性。 

---
# EntroPIC: Towards Stable Long-Term Training of LLMs via Entropy Stabilization with Proportional-Integral Control 

**Title (ZH)**: EntroPIC: 通过比例积分控制实现熵稳定训练的长期稳定预训练语言模型 

**Authors**: Kai Yang, Xin Xu, Yangkun Chen, Weijie Liu, Jiafei Lyu, Zichuan Lin, Deheng Ye, Saiyong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15248)  

**Abstract**: Long-term training of large language models (LLMs) requires maintaining stable exploration to prevent the model from collapsing into sub-optimal behaviors. Entropy is crucial in this context, as it controls exploration and helps avoid premature convergence to sub-optimal solutions. However, existing reinforcement learning methods struggle to maintain an appropriate level of entropy, as the training process involves a mix of positive and negative samples, each affecting entropy in different ways across steps. To address this, we propose Entropy stablilization via Proportional-Integral Control (EntroPIC), a novel method that adaptively adjusts the influence of positive and negative samples by dynamically tuning their loss coefficients. This approach stabilizes entropy throughout training, ensuring efficient exploration and steady progress. We provide a comprehensive theoretical analysis for both on-policy and off-policy learning settings, demonstrating that EntroPIC is effective at controlling entropy in large-scale LLM training. Experimental results show that our method successfully maintains desired entropy levels, enabling stable and optimal RL training for LLMs. 

**Abstract (ZH)**: 长周期训练大规模语言模型（LLMs）需要保持稳定的探索以防止模型陷入亚最优行为。熵在这个过程中至关重要，因为它控制探索并帮助避免过早收敛到亚最优解。然而，现有的强化学习方法难以维持适当的熵水平，因为训练过程涉及正样本和负样本的混合，它们在不同步骤中以不同的方式影响熵。为了解决这一问题，我们提出了一种新的方法——基于比例积分控制的熵稳定化（EntroPIC），该方法通过动态调整正样本和负样本的损失系数来适应性地调整它们的影响。这种方法在整个训练过程中稳定熵，确保高效的探索并实现稳步进展。我们提供了针对在线策略和离线策略学习环境的全面理论分析，证明了EntroPIC在大规模LLM训练中有效控制熵的能力。实验结果表明，我们的方法能够成功维持所需的熵水平，从而实现LLMs的稳定和最优的RL训练。 

---
# OEMA: Ontology-Enhanced Multi-Agent Collaboration Framework for Zero-Shot Clinical Named Entity Recognition 

**Title (ZH)**: 基于本体增强的多agent协作框架：零样本临床命名实体识别 

**Authors**: Xinli Tao, Xin Dong, Xuezhong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.15211)  

**Abstract**: Clinical named entity recognition (NER) is crucial for extracting information from electronic health records (EHRs), but supervised models like CRF and BioClinicalBERT require costly annotated data. While zero-shot NER with large language models (LLMs) reduces this dependency, it struggles with example selection granularity and integrating prompts with self-improvement. To address this, we propose OEMA, a zero-shot clinical NER framework using multi-agent collaboration. OEMA's three components are: a self-annotator generating examples, a discriminator filtering them via SNOMED CT, and a predictor using entity descriptions for accurate inference. On MTSamples and VAERS datasets, OEMA achieves state-of-the-art exact-match performance. Under related-match, it matches supervised BioClinicalBERT and surpasses CRF. OEMA addresses key zero-shot NER challenges through ontology-guided reasoning and multi-agent collaboration, achieving near-supervised performance and showing promise for clinical NLP applications. 

**Abstract (ZH)**: 临床命名实体识别（NER）对于从电子健康记录（EHRs）中提取信息至关重要，但监督模型如CRF和BioClinicalBERT需要昂贵的标注数据。而利用大规模语言模型（LLMs）的零样本NER在减少这种依赖性的同时，难以在示例选择粒度上取得突破，并且难以整合提示与自我提升。为解决这一问题，我们提出了一种名为OEMA的基于多智能体协作的零样本临床NER框架。OEMA的三个组成部分包括：自标注生成器生成示例、鉴别器通过SNOMED CT进行过滤以及使用实体描述进行准确推断的预测器。在MTSamples和VAERS数据集上，OEMA实现了最先进的精确匹配性能。在相关匹配下，OEMA匹配并超过了监督的BioClinicalBERT，并超越了CRF。OEMA通过本体指导推理和多智能体协作解决了关键的零样本NER挑战，实现了接近监督性能，并展现出了在临床NLP应用中的潜力。 

---
# Unveiling Intrinsic Dimension of Texts: from Academic Abstract to Creative Story 

**Title (ZH)**: 揭示文本的固有维度：从学术摘要到创意故事 

**Authors**: Vladislav Pedashenko, Laida Kushnareva, Yana Khassan Nibal, Eduard Tulchinskii, Kristian Kuznetsov, Vladislav Zharchinskii, Yury Maximov, Irina Piontkovskaya  

**Link**: [PDF](https://arxiv.org/pdf/2511.15210)  

**Abstract**: Intrinsic dimension (ID) is an important tool in modern LLM analysis, informing studies of training dynamics, scaling behavior, and dataset structure, yet its textual determinants remain underexplored. We provide the first comprehensive study grounding ID in interpretable text properties through cross-encoder analysis, linguistic features, and sparse autoencoders (SAEs). In this work, we establish three key findings. First, ID is complementary to entropy-based metrics: after controlling for length, the two are uncorrelated, with ID capturing geometric complexity orthogonal to prediction quality. Second, ID exhibits robust genre stratification: scientific prose shows low ID (~8), encyclopedic content medium ID (~9), and creative/opinion writing high ID (~10.5) across all models tested. This reveals that contemporary LLMs find scientific text "representationally simple" while fiction requires additional degrees of freedom. Third, using SAEs, we identify causal features: scientific signals (formal tone, report templates, statistics) reduce ID; humanized signals (personalization, emotion, narrative) increase it. Steering experiments confirm these effects are causal. Thus, for contemporary models, scientific writing appears comparatively "easy", whereas fiction, opinion, and affect add representational degrees of freedom. Our multi-faceted analysis provides practical guidance for the proper use of ID and the sound interpretation of ID-based results. 

**Abstract (ZH)**: 内在维度（ID）是现代大语言模型（LLM）分析的重要工具，用以研究训练动力学、缩放行为和数据集结构，但其文本决定因素仍缺乏探索。本文通过交叉编码分析、语言特征和稀疏自编码器（SAEs）提供首个全面研究，将ID与可解释的文本属性联系起来。本文确立了三项关键发现：首先，ID与基于熵的指标互补：在控制长度后，两者不相关，ID捕捉到与预测质量正交的几何复杂性；其次，ID表现出稳健的体裁分层：科学文体的ID较低（约8），百科内容的ID中等（约9），而创意/观点写作的ID较高（约10.5），这表明当下LLM认为科学文本“表示简单”，而叙事性内容则需要更多的自由度；最后，利用SAEs，我们识别出因果特征：科学信号（正式语气、报告模板、统计数据）降低ID；人性化信号（个性化、情感、叙述）则增加ID。定向实验进一步证实这些效果具有因果性。因此，对于当下模型而言，科学写作显得相对“容易”，而叙事、观点和情感则增加表示的自由度。我们的多方面分析为正确使用ID及其基于ID的结果的稳健解释提供了实用指导。 

---
# Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks 

**Title (ZH)**: IPI为中心的LLM代理防御框架的分类、评估与利用 

**Authors**: Zimo Ji, Xunguang Wang, Zongjie Li, Pingchuan Ma, Yudong Gao, Daoyuan Wu, Xincheng Yan, Tian Tian, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15203)  

**Abstract**: Large Language Model (LLM)-based agents with function-calling capabilities are increasingly deployed, but remain vulnerable to Indirect Prompt Injection (IPI) attacks that hijack their tool calls. In response, numerous IPI-centric defense frameworks have emerged. However, these defenses are fragmented, lacking a unified taxonomy and comprehensive evaluation. In this Systematization of Knowledge (SoK), we present the first comprehensive analysis of IPI-centric defense frameworks. We introduce a comprehensive taxonomy of these defenses, classifying them along five dimensions. We then thoroughly assess the security and usability of representative defense frameworks. Through analysis of defensive failures in the assessment, we identify six root causes of defense circumvention. Based on these findings, we design three novel adaptive attacks that significantly improve attack success rates targeting specific frameworks, demonstrating the severity of the flaws in these defenses. Our paper provides a foundation and critical insights for the future development of more secure and usable IPI-centric agent defense frameworks. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的功能调用代理的间接提示注入（IPI）防御框架综述 

---
# Finetuning LLMs for Automatic Form Interaction on Web-Browser in Selenium Testing Framework 

**Title (ZH)**: 基于Selenium测试框架的自动表单交互Web浏览器中大型语言模型的微调 

**Authors**: Nguyen-Khang Le, Nguyen Hiep, Minh Nguyen, Son Luu, Trung Vo, Quan Bui, Nomura Shoshin, Le-Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15168)  

**Abstract**: Automated web application testing is a critical component of modern software development, with frameworks like Selenium widely adopted for validating functionality through browser automation. Among the essential aspects of such testing is the ability to interact with and validate web forms, a task that requires syntactically correct, executable scripts with high coverage of input fields. Despite its importance, this task remains underexplored in the context of large language models (LLMs), and no public benchmark or dataset exists to evaluate LLMs on form interaction generation systematically. This paper introduces a novel method for training LLMs to generate high-quality test cases in Selenium, specifically targeting form interaction testing. We curate both synthetic and human-annotated datasets for training and evaluation, covering diverse real-world forms and testing scenarios. We define clear metrics for syntax correctness, script executability, and input field coverage. Our empirical study demonstrates that our approach significantly outperforms strong baselines, including GPT-4o and other popular LLMs, across all evaluation metrics. Our work lays the groundwork for future research on LLM-based web testing and provides resources to support ongoing progress in this area. 

**Abstract (ZH)**: 自动化Web应用测试是现代软件开发中的关键组成部分，框架如Selenium广泛用于通过浏览器自动化验证功能。此类测试的重要方面之一是能够与Web表单进行交互并验证表单，这需要具有高覆盖率的输入字段的语义正确且可执行的脚本。尽管其重要性不言而喻，但在大型语言模型（LLMs）的背景下，这一任务仍然未被充分探索，也没有公开的基准或数据集可以系统地评估LLMs在表单交互生成方面的表现。本文介绍了一种新的方法，用于训练LLMs生成高质量的Selenium测试案例，特别针对表单交互测试。我们为训练和评估精心收集了合成和人工注释的数据集，涵盖了各种实际世界的表单和测试场景。我们定义了清晰的语法规则正确性、脚本可执行性和输入字段覆盖率的度量标准。我们的实证研究显示，在所有评估指标上，我们的方法显著优于包括GPT-4o和其他流行LLMs在内的强基线。我们的工作为基于LLMs的Web测试未来研究奠定了基础，并提供了支持这一领域持续进展的资源。 

---
# Teaching According to Students' Aptitude: Personalized Mathematics Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs 

**Title (ZH)**: 根据学生能力进行教学：基于人格、记忆与遗忘意识的个性化数学辅导 

**Authors**: Yang Wu, Rujing Yao, Tong Zhang, Yufei Shi, Zhuoren Jiang, Zhushan Li, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15163)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into intelligent tutoring systems to provide human-like and adaptive instruction. However, most existing approaches fail to capture how students' knowledge evolves dynamically across their proficiencies, conceptual gaps, and forgetting patterns. This challenge is particularly acute in mathematics tutoring, where effective instruction requires fine-grained scaffolding precisely calibrated to each student's mastery level and cognitive retention. To address this issue, we propose TASA (Teaching According to Students' Aptitude), a student-aware tutoring framework that integrates persona, memory, and forgetting dynamics for personalized mathematics learning. Specifically, TASA maintains a structured student persona capturing proficiency profiles and an event memory recording prior learning interactions. By incorporating a continuous forgetting curve with knowledge tracing, TASA dynamically updates each student's mastery state and generates contextually appropriate, difficulty-calibrated questions and explanations. Empirical results demonstrate that TASA achieves superior learning outcomes and more adaptive tutoring behavior compared to representative baselines, underscoring the importance of modeling temporal forgetting and learner profiles in LLM-based tutoring systems. 

**Abstract (ZH)**: 面向学生能力的教学：一种结合人格、记忆和遗忘动态的个性化数学学习辅导框架 

---
# ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation 

**Title (ZH)**: 基于项目检索增强生成的LLM推荐方法 

**Authors**: Sunwoo Kim, Geon Lee, Kyungho Kim, Jaemin Yoo, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.15141)  

**Abstract**: Recently, large language models (LLMs) have been widely used as recommender systems, owing to their strong reasoning capability and their effectiveness in handling cold-start items. To better adapt LLMs for recommendation, retrieval-augmented generation (RAG) has been incorporated. Most existing RAG methods are user-based, retrieving purchase patterns of users similar to the target user and providing them to the LLM. In this work, we propose ItemRAG, an item-based RAG method for LLM-based recommendation that retrieves relevant items (rather than users) from item-item co-purchase histories. ItemRAG helps LLMs capture co-purchase patterns among items, which are beneficial for recommendations. Especially, our retrieval strategy incorporates semantically similar items to better handle cold-start items and uses co-purchase frequencies to improve the relevance of the retrieved items. Through extensive experiments, we demonstrate that ItemRAG consistently (1) improves the zero-shot LLM-based recommender by up to 43% in Hit-Ratio-1 and (2) outperforms user-based RAG baselines under both standard and cold-start item recommendation settings. 

**Abstract (ZH)**: 基于项目的RAG方法：ItemRAG在LLM推荐中的应用 

---
# From Solving to Verifying: A Unified Objective for Robust Reasoning in LLMs 

**Title (ZH)**: 从求解到验证：LLM中稳健推理的统一目标 

**Authors**: Xiaoxuan Wang, Bo Liu, Song Jiang, Jingzhou Liu, Jingyuan Qi, Xia Chen, Baosheng He  

**Link**: [PDF](https://arxiv.org/pdf/2511.15137)  

**Abstract**: The reasoning capabilities of large language models (LLMs) have been significantly improved through reinforcement learning (RL). Nevertheless, LLMs still struggle to consistently verify their own reasoning traces. This raises the research question of how to enhance the self-verification ability of LLMs and whether such an ability can further improve reasoning performance. In this work, we propose GRPO-Verif, an algorithm that jointly optimizes solution generation and self-verification within a unified loss function, with an adjustable hyperparameter controlling the weight of the verification signal. Experimental results demonstrate that our method enhances self-verification capability while maintaining comparable performance in reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过强化学习（RL）显著提升了其推理能力，但仍难以一致地验证自身的推理轨迹。本研究探讨了如何增强LLMs的自我验证能力，以及这种能力是否能进一步提升推理性能。我们提出了GRPO-Verif算法，该算法在统一的损失函数中共同优化解的生成和自我验证，并通过可调节的超参数控制验证信号的权重。实验结果表明，该方法增强了自我验证能力，同时保持了在推理性能上的可比性。 

---
# Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference 

**Title (ZH)**: 可扩展专家混合推理的动态专家量化 

**Authors**: Kexin Chu, Dawei Xiang, Zixu Shen, Yiwei Yang, Zecheng Liu, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15015)  

**Abstract**: Mixture-of-Experts (MoE) models scale LLM capacity efficiently, but deployment on consumer GPUs is limited by the large memory footprint of inactive experts. Static post-training quantization reduces storage costs but cannot adapt to shifting activation patterns, causing accuracy loss under aggressive compression. So we present DynaExq, a runtime system that treats expert precision as a first-class, dynamically managed resource. DynaExq combines (1) a hotness-aware precision controller that continuously aligns expert bit-widths with long-term activation statistics, (2) a fully asynchronous precision-switching pipeline that overlaps promotion and demotion with MoE computation, and (3) a fragmentation-free memory pooling mechanism that supports hybrid-precision experts with deterministic allocation. Together, these components enable stable, non-blocking precision transitions under strict HBM budgets.
Across Qwen3-30B and Qwen3-80B MoE models and six representative benchmarks, DynaExq deploys large LLMs on single RTX 5090 and A6000 GPUs and improves accuracy by up to 4.03 points over static low-precision baselines. The results show that adaptive, workload-aware quantization is an effective strategy for memory-constrained MoE serving. 

**Abstract (ZH)**: DynaExq：一种用于MoE模型的运行时动态精度管理系统 

---
# Mathematical Analysis of Hallucination Dynamics in Large Language Models: Uncertainty Quantification, Advanced Decoding, and Principled Mitigation 

**Title (ZH)**: 大型语言模型中幻觉动力学的数学分析：不确定性量化、高级解码与原则性缓解 

**Authors**: Moses Kiprono  

**Link**: [PDF](https://arxiv.org/pdf/2511.15005)  

**Abstract**: Large Language Models (LLMs) are powerful linguistic engines but remain susceptible to hallucinations: plausible-sounding outputs that are factually incorrect or unsupported. In this work, we present a mathematically grounded framework to understand, measure, and mitigate these hallucinations. Drawing on probabilistic modeling, information theory, trigonometric signal analysis, and Bayesian uncertainty estimation, we analyze how errors compound autoregressively, propose refined uncertainty metrics, including semantic and phase-aware variants, and develop principled mitigation strategies such as contrastive decoding, retrieval-augmented grounding, factual alignment, and abstention. This unified lens connects recent advances in calibration, retrieval, and alignment to support safer and more reliable LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）是强大的语言引擎，但仍易产生幻觉：这些幻觉输出看似合理但事实错误或缺乏支持。在本文中，我们提出了一种数学上严密的框架来理解、衡量和减少这些幻觉。利用概率模型、信息论、三角信号分析以及贝叶斯不确定性估计，我们分析了错误如何自回归地累积，提出了精炼的不确定性度量，包括语义和相位感知的变体，并开发了诸如对比解码、检索增强 grounding、事实对齐和回避等有原则的缓解策略。这一统一视角将近期在校准、检索和对齐方面的进展结合起来，以支持更安全和更可靠的大型语言模型。 

---
# SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification 

**Title (ZH)**: SVBRD-LLM：自主车辆识别的自我验证行为规则发现 

**Authors**: Xiangyu Li, Zhaomiao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.14977)  

**Abstract**: As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence. 

**Abstract (ZH)**: 随着越来越多的自动驾驶车辆在公路上行驶，理解自动驾驶车辆的实际道路行为对于分析交通安全、制定政策和提高公众接受度至关重要。本文提出了一种名为SVBRD-LLM的框架，该框架通过零样本提示工程自动发现、验证和应用来自真实交通视频的可解释行为规则。该框架使用YOLOv8和ByteTrack提取车辆轨迹，计算运动学特征，并利用GPT-5零样本提示比较自动驾驶车辆和人工驾驶车辆，生成35个结构化的行为规则假设。这些规则在验证集上进行测试，基于失败案例迭代优化以筛选虚假相关性，并编译成高置信度规则库。该框架在独立测试集上对速度变化预测、变道预测和自动驾驶车辆识别任务进行了评估。实验结果显示，在超过1500小时的真实交通视频上，框架在自动驾驶车辆识别任务中达到了90.0%的准确率和93.3%的F1分数。发现的规则清楚地揭示了自动驾驶车辆在速度控制平滑度、变道保守性和加速度稳定性方面的独特特征，每个规则都附有语义描述、适用上下文和验证置信度。 

---
# MermaidSeqBench: An Evaluation Benchmark for LLM-to-Mermaid Sequence Diagram Generation 

**Title (ZH)**: MermaidSeqBench: 一种针对LLM到Mermaid序列图生成的评估基准 

**Authors**: Basel Shbita, Farhan Ahmed, Chad DeLuca  

**Link**: [PDF](https://arxiv.org/pdf/2511.14967)  

**Abstract**: Large language models (LLMs) have demonstrated excellent capabilities in generating structured diagrams from natural language descriptions. In particular, they have shown great promise in generating sequence diagrams for software engineering, typically represented in a text-based syntax such as Mermaid. However, systematic evaluations in this space remain underdeveloped as there is a lack of existing benchmarks to assess the LLM's correctness in this task. To address this shortcoming, we introduce MermaidSeqBench, a human-verified and LLM-synthetically-extended benchmark for assessing an LLM's capabilities in generating Mermaid sequence diagrams from textual prompts. The benchmark consists of a core set of 132 samples, starting from a small set of manually crafted and verified flows. These were expanded via a hybrid methodology combining human annotation, in-context LLM prompting, and rule-based variation generation. Our benchmark uses an LLM-as-a-judge model to assess Mermaid sequence diagram generation across fine-grained metrics, including syntax correctness, activation handling, error handling, and practical usability. We perform initial evaluations on numerous state-of-the-art LLMs and utilize multiple LLM judge models to demonstrate the effectiveness and flexibility of our benchmark. Our results reveal significant capability gaps across models and evaluation modes. Our proposed benchmark provides a foundation for advancing research in structured diagram generation and for developing more rigorous, fine-grained evaluation methodologies. 

**Abstract (ZH)**: 大型语言模型（LLMs）在从自然语言描述生成结构化图表方面展现了出色的能力。特别是在为软件工程生成Mermaid文本语法表示的序列图方面，它们展现出了巨大的潜力。然而，这个领域中的系统性评估仍然不够完善，因为缺乏能够评估LLM在这项任务中正确性的基准。为解决这一不足，我们引入了MermaidSeqBench，这是一个由人工验证并结合LLM合成扩展的基准，用于评估LLM从文本提示生成Mermaid序列图的能力。该基准包括一个核心样本集，包含132个样本，源自一组由人工精心制作并验证的数据流。这些样本通过一种结合了人工注释、上下文提示的LLM和基于规则的变异生成的混合方法进行了扩展。我们的基准利用LLM作为裁判模型，从语法正确性、激活处理、错误处理和实用易用性等细粒度指标对Mermaid序列图表生成进行评估。我们在多个最新的LLM上进行了初始评估，并使用多种LLM裁判模型证明了我们基准的有效性和灵活性。我们的结果显示了不同模型和评估模式之间的显著能力差距。我们提出的基准为推进结构化图表生成研究提供了基础，并为开发更加严格和细粒度的评估方法奠定了基础。 

---
# On-Premise SLMs vs. Commercial LLMs: Prompt Engineering and Incident Classification in SOCs and CSIRTs 

**Title (ZH)**: 本地托管的SLM与商用的LLM：SOCs和CSIRT中的提示工程与事件分类 

**Authors**: Gefté Almeida, Marcio Pohlmann, Alex Severo, Diego Kreutz, Tiago Heinrich, Lourenço Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2511.14908)  

**Abstract**: In this study, we evaluate open-source models for security incident classification, comparing them with proprietary models. We utilize a dataset of anonymized real incidents, categorized according to the NIST SP 800-61r3 taxonomy and processed using five prompt-engineering techniques (PHP, SHP, HTP, PRP, and ZSL). The results indicate that, although proprietary models still exhibit higher accuracy, locally deployed open-source models provide advantages in privacy, cost-effectiveness, and data sovereignty. 

**Abstract (ZH)**: 本研究评估了开源模型在安全事件分类中的性能，并将其与专有模型进行了比较。我们使用了一组匿名的实际事件数据集，这些事件按照NIST SP 800-61r3分类，并采用了五种提示工程技术（PHP、SHP、HTP、PRP和ZSL）进行处理。结果显示，尽管专有模型仍表现出更高的准确性，但本地部署的开源模型在隐私保护、成本效益和数据主权方面具有优势。 

---
# Empowering Multi-Turn Tool-Integrated Reasoning with Group Turn Policy Optimization 

**Title (ZH)**: 增强多轮工具集成推理的能力：基于组轮次策略优化 

**Authors**: Yifeng Ding, Hung Le, Songyang Han, Kangrui Ruan, Zhenghui Jin, Varun Kumar, Zijian Wang, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2511.14846)  

**Abstract**: Training Large Language Models (LLMs) for multi-turn Tool-Integrated Reasoning (TIR) - where models iteratively reason, generate code, and verify through execution - remains challenging for existing reinforcement learning (RL) approaches. Current RL methods, exemplified by Group Relative Policy Optimization (GRPO), suffer from coarse-grained, trajectory-level rewards that provide insufficient learning signals for complex multi-turn interactions, leading to training stagnation. To address this issue, we propose Group Turn Policy Optimization (GTPO), a novel RL algorithm specifically designed for training LLMs on multi-turn TIR tasks. GTPO introduces three key innovations: (1) turn-level reward assignment that provides fine-grained feedback for individual turns, (2) return-based advantage estimation where normalized discounted returns are calculated as advantages, and (3) self-supervised reward shaping that exploits self-supervision signals from generated code to densify sparse binary outcome-based rewards. Our comprehensive evaluation demonstrates that GTPO outperforms GRPO by 3.0% on average across diverse reasoning benchmarks, establishing its effectiveness for advancing complex mathematical reasoning in the real world. 

**Abstract (ZH)**: 训练大规模语言模型（LLMs）进行多轮工具集成推理（TIR）——模型通过迭代推理、生成代码和执行验证——对于现有的强化学习（RL）方法仍具有挑战性。当前的RL方法，如组相对策略优化（GRPO），因其粗粒度的轨迹级奖励提供了不足的学习信号，导致复杂多轮交互的训练停滞。为解决这一问题，我们提出了组轮次策略优化（GTPO），这是一种专门用于在多轮TIR任务上训练LLMs的新型RL算法。GTPO引入了三项关键创新：（1）轮次级奖励分配，提供细粒度的反馈；（2）基于返回的优势估计；（3）自监督奖励塑形。全面的评估表明，GTPO在多种推理基准测试中平均优于GRPO 3.0%，证明了其在现实世界中推动复杂数学推理的有效性。 

---
# Scalable and Efficient Large-Scale Log Analysis with LLMs: An IT Software Support Case Study 

**Title (ZH)**: 大规模日志分析中的可扩展性和高效性：基于LLM的IT软件支持案例研究 

**Authors**: Pranjal Gupta, Karan Bhukar, Harshit Kumar, Seema Nagar, Prateeti Mohapatra, Debanjana Kar  

**Link**: [PDF](https://arxiv.org/pdf/2511.14803)  

**Abstract**: IT environments typically have logging mechanisms to monitor system health and detect issues. However, the huge volume of generated logs makes manual inspection impractical, highlighting the importance of automated log analysis in IT Software Support. In this paper, we propose a log analytics tool that leverages Large Language Models (LLMs) for log data processing and issue diagnosis, enabling the generation of automated insights and summaries. We further present a novel approach for efficiently running LLMs on CPUs to process massive log volumes in minimal time without compromising output quality. We share the insights and lessons learned from deployment of the tool - in production since March 2024 - scaled across 70 software products, processing over 2000 tickets for issue diagnosis, achieving a time savings of 300+ man hours and an estimated $15,444 per month in manpower costs compared to the traditional log analysis practices. 

**Abstract (ZH)**: IT环境通常具有日志记录机制以监控系统健康状况和检测问题。然而，生成的日志量巨大，使得人工检查变得不切实际，突显了自动化日志分析在IT软件支持中的重要性。本文提出了一种利用大型语言模型（LLMs）进行日志数据处理和问题诊断的日志分析工具，能够生成自动化见解和摘要。此外，我们还提出了一种高效在CPU上运行LLMs的方法，以在最少的时间内处理大量日志数据而不牺牲输出质量。我们分享了该工具部署期间的见解和经验教训——该工具自2024年3月以来在70个软件产品中运行，并处理了超过2000个问题诊断工单，实现了300多个工时的节省，以及每月约15,444美元的人力成本节约，相较于传统的日志分析方法。 

---
# Evaluating Generative AI for CS1 Code Grading: Direct vs Reverse Methods 

**Title (ZH)**: 评价生成式AI在CS1代码评分中的效果：直接方法 vs 逆向方法 

**Authors**: Ahmad Memon, Abdallah Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2511.14798)  

**Abstract**: Manual grading of programming assignments in introductory computer science courses can be time-consuming and prone to inconsistencies. While unit testing is commonly used for automatic evaluation, it typically follows a binary pass/fail model and does not give partial marks. Recent advances in large language models (LLMs) offer the potential for automated, scalable, and more objective grading.
This paper compares two AI-based grading techniques: \textit{Direct}, where the AI model applies a rubric directly to student code, and \textit{Reverse} (a newly proposed approach), where the AI first fixes errors, then deduces a grade based on the nature and number of fixes. Each method was evaluated on both the instructor's original grading scale and a tenfold expanded scale to assess the impact of range on AI grading accuracy. To assess their effectiveness, AI-assigned scores were evaluated against human tutor evaluations on a range of coding problems and error types.
Initial findings suggest that while the Direct approach is faster and straightforward, the Reverse technique often provides a more fine-grained assessment by focusing on correction effort. Both methods require careful prompt engineering, particularly for allocating partial credit and handling logic errors. To further test consistency, we also used synthetic student code generated using Gemini Flash 2.0, which allowed us to evaluate AI graders on a wider range of controlled error types and difficulty levels. We discuss the strengths and limitations of each approach, practical considerations for prompt design, and future directions for hybrid human-AI grading systems that aim to improve consistency, efficiency, and fairness in CS courses. 

**Abstract (ZH)**: 基于人工评分的编程作业在入门计算机科学课程中耗时且容易出现不一致。虽然单元测试常用于自动评估，但通常遵循通过/未通过的二元模型，并不提供部分分数。大型语言模型的最新进展为自动化、可扩展和更客观的评分提供了潜力。

本文比较了两种基于AI的评分技术：直接评分，即AI模型直接应用评分标准到学生代码；以及逆向评分（一种新提出的策略），即AI先修复错误，然后根据修复的数量和性质来推断分数。每种方法在教师原始评分标准和扩展十倍的评分标准上进行了评估，以考察评分范围对AI评分准确度的影响。为了评估其有效性，AI分配的分数与人类导师对各种编码问题和错误类型的评估进行了比较。

初步发现表明，虽然直接评分方法更快且简单，但逆向评分方法通常通过关注修正努力提供了更为细致的评估。两种方法都需要精心设计提示工程，尤其是对于分配部分分数和处理逻辑错误的处理。为了进一步测试一致性，我们还使用了使用Gemini Flash 2.0生成的合成学生代码，这使我们能够对AI评分器进行更广泛的控制错误类型和难度级别的评估。我们讨论了每种方法的优势和局限性，提示设计的实用考虑，以及未来混合人类-AI评分系统的方向，旨在提高计算机科学课程中的评分一致性和公平性。 

---
# irace-evo: Automatic Algorithm Configuration Extended With LLM-Based Code Evolution 

**Title (ZH)**: irace-evo：基于LLM的代码进化扩展的自动算法配置 

**Authors**: Camilo Chacón Sartori, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2511.14794)  

**Abstract**: Automatic algorithm configuration tools such as irace efficiently tune parameter values but leave algorithmic code unchanged. This paper introduces a first version of irace-evo, an extension of irace that integrates code evolution through large language models (LLMs) to jointly explore parameter and code spaces. The proposed framework enables multi-language support (e.g., C++, Python), reduces token consumption via progressive context management, and employs the Always-From-Original principle to ensure robust and controlled code evolution. We evaluate irace-evo on the Construct, Merge, Solve & Adapt (CMSA) metaheuristic for the Variable-Sized Bin Packing Problem (VSBPP). Experimental results show that irace-evo can discover new algorithm variants that outperform the state-of-the-art CMSA implementation while maintaining low computational and monetary costs. Notably, irace-evo generates competitive algorithmic improvements using lightweight models (e.g., Claude Haiku 3.5) with a total usage cost under 2 euros. These results demonstrate that coupling automatic configuration with LLM-driven code evolution provides a powerful, cost-efficient avenue for advancing heuristic design and metaheuristic optimization. 

**Abstract (ZH)**: 自动算法配置工具如irace能够高效调整参数值但不改变算法代码。本文介绍了irace-evo，这是一种扩展了irace并借助大型语言模型（LLMs）实现代码进化的初步版本，旨在联合探索参数空间和代码空间。所提出的框架支持多语言（例如，C++、Python），通过渐进式上下文管理减少词汇消耗，并采用“始终来自原始代码”的原则以确保稳健和可控的代码进化。我们在可变大小的物品装箱问题（VSBPP）的元启发式方法（CMSA）中评估了irace-evo。实验结果表明，irace-evo能够发现超越当前最佳CMSA实现的新算法变体，同时保持较低的计算和经济成本。值得注意的是，irace-evo使用轻量级模型（例如，Claude Haiku 3.5）即可生成具有竞争力的算法改进，总使用成本低于2欧元。这些结果证明，结合自动配置与LLM驱动的代码进化为启发式设计和元启发式优化提供了一种强大且成本效益高的途径。 

---
# LiveCLKTBench: Towards Reliable Evaluation of Cross-Lingual Knowledge Transfer in Multilingual LLMs 

**Title (ZH)**: LiveCLKTBench: 朝着可靠评估多语言LLM跨语言知识迁移的方向 

**Authors**: Pei-Fu Guo, Yun-Da Tsai, Chun-Chia Hsu, Kai-Xin Chen, Ya-An Tsai, Kai-Wei Chang, Nanyun Peng, Mi-Yen Yeh, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.14774)  

**Abstract**: Evaluating cross-lingual knowledge transfer in large language models is challenging, as correct answers in a target language may arise either from genuine transfer or from prior exposure during pre-training. We present LiveCLKTBench, an automated generation pipeline specifically designed to isolate and measure cross-lingual knowledge transfer. Our pipeline identifies self-contained, time-sensitive knowledge entities from real-world domains, filters them based on temporal occurrence, and verifies them against the model's knowledge. The documents of these valid entities are then used to generate factual questions, which are translated into multiple languages to evaluate transferability across linguistic boundaries. Using LiveCLKTBench, we evaluate several LLMs across five languages and observe that cross-lingual transfer is strongly influenced by linguistic distance and often asymmetric across language directions. While larger models improve transfer, the gains diminish with scale and vary across domains. These findings provide new insights into multilingual transfer and demonstrate the value of LiveCLKTBench as a reliable benchmark for future research. 

**Abstract (ZH)**: 评估大规模语言模型的跨语言知识迁移具有挑战性，因为目标语言中的正确答案可能源自真实的迁移或预训练期间的先验暴露。我们提出了LiveCLKTBench，一个专门设计的自动化生成管道，用于隔离和测量跨语言知识迁移。该管道从实际情况中识别出自我包含且时间敏感的知识实体，根据时间发生情况进行过滤，并验证这些实体的知识。这些有效实体的文档随后用于生成事实性问题，并将其翻译成多种语言以评估跨语言边界的知识迁移能力。使用LiveCLKTBench，我们在五种语言上评估了几种LLM，并观察到跨语言迁移受语言距离影响强烈且在语言方向上经常不对称。虽然较大的模型能提高迁移能力，但这种提升随规模递减并在不同领域中有所不同。这些发现为多语言迁移提供了新的见解，并证明了LiveCLKTBench作为未来研究可靠基准的价值。 

---
# Test-time Scaling of LLMs: A Survey from A Subproblem Structure Perspective 

**Title (ZH)**: LLM测试时尺度调整：从子问题结构视角的综述 

**Authors**: Zhuoyi Yang, Xu Guo, Tong Zhang, Huijuan Xu, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.14772)  

**Abstract**: With this paper, we survey techniques for improving the predictive accuracy of pretrained large language models by allocating additional compute at inference time. In categorizing test-time scaling methods, we place special emphasis on how a problem is decomposed into subproblems and on the topological organization of these subproblems whether sequential, parallel, or tree-structured. This perspective allows us to unify diverse approaches such as Chain-of-Thought, Branch-Solve-Merge, and Tree-of-Thought under a common lens. We further synthesize existing analyses of these techniques, highlighting their respective strengths and weaknesses, and conclude by outlining promising directions for future research 

**Abstract (ZH)**: 通过本论文，我们调研了在推理时分配额外计算资源以提升预训练大规模语言模型预测准确性的技术。在归类测试时缩放方法时，我们特别强调问题如何被分解成子问题以及这些子问题的拓扑组织方式，无论是串行、并行还是树状结构。这种视角使我们能够将Chain-of-Thought、Branch-Solve-Merge和Tree-of-Thought等多样方法统一起来，置于一个共同的框架之下。我们进一步综合现有对这些技术的分析，突出其各自的优点和缺点，并总结出未来研究的有前景的方向。 

---
# Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications 

**Title (ZH)**: 基于聚类的自适应检索：针对RAG应用的动态上下文选择 

**Authors**: Yifan Xu, Vipul Gupta, Rohit Aggarwal, Varsha Mahadevan, Bhaskar Krishnamachari  

**Link**: [PDF](https://arxiv.org/pdf/2511.14769)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by pulling in external material, document, code, manuals, from vast and ever-growing corpora, to effectively answer user queries. The effectiveness of RAG depends significantly on aligning the number of retrieved documents with query characteristics: narrowly focused queries typically require fewer, highly relevant documents, whereas broader or ambiguous queries benefit from retrieving more extensive supporting information. However, the common static top-k retrieval approach fails to adapt to this variability, resulting in either insufficient context from too few documents or redundant information from too many. Motivated by these challenges, we introduce Cluster-based Adaptive Retrieval (CAR), an algorithm that dynamically determines the optimal number of documents by analyzing the clustering patterns of ordered query-document similarity distances. CAR detects the transition point within similarity distances, where tightly clustered, highly relevant documents shift toward less pertinent candidates, establishing an adaptive cut-off that scales with query complexity. On Coinbase's CDP corpus and the public MultiHop-RAG benchmark, CAR consistently picks the optimal retrieval depth and achieves the highest TES score, outperforming every fixed top-k baseline. In downstream RAG evaluations, CAR cuts LLM token usage by 60%, trims end-to-end latency by 22%, and reduces hallucinations by 10% while fully preserving answer relevance. Since integrating CAR into Coinbase's virtual assistant, we've seen user engagement jump by 200%. 

**Abstract (ZH)**: 基于聚类的自适应检索（CAR）增强的大语言模型检索 augmented 生成（RAG）通过从 vast 和不断增长的语料库中引入外部材料、文档、代码和手册，增强大语言模型 (LLMs)，以有效回答用户查询。CAR 通过分析有序查询-文档相似度距离的聚类模式动态确定最优的检索文档数量。CAR 检测相似度距离中的转变点，其中紧密聚类的相关文档逐渐向不相关候选者转变，从而建立一个与查询复杂度相适应的切割点。在 Coinbase 的 CDP 语料库和公开的 MultiHop-RAG 基准测试中，CAR 一致地选择了最优的检索深度并获得了最高的 TES 分数，超越了所有固定的 top-k 基准。在下游 RAG 评价中，CAR 将 LLM 令牌使用量降低了 60%，端到端延迟降低了 22%，幻觉减少了 10% 同时完全保持了答案的相关性。自将 CAR 集成到 Coinbase 的虚拟助手后，我们看到了用户参与度提高了 200%。 

---
# ExplainRec: Towards Explainable Multi-Modal Zero-Shot Recommendation with Preference Attribution and Large Language Models 

**Title (ZH)**: ExplainRec: 面向可解释的多模态零样本推荐及偏好归因的大语言模型 

**Authors**: Bo Ma, LuYao Liu, ZeHua Hu, Simon Lau  

**Link**: [PDF](https://arxiv.org/pdf/2511.14770)  

**Abstract**: Recent advances in Large Language Models (LLMs) have opened new possibilities for recommendation systems, though current approaches such as TALLRec face challenges in explainability and cold-start scenarios. We present ExplainRec, a framework that extends LLM-based recommendation capabilities through preference attribution, multi-modal fusion, and zero-shot transfer learning. The framework incorporates four technical contributions: preference attribution tuning for explainable recommendations, zero-shot preference transfer for cold-start users and items, multi-modal enhancement leveraging visual and textual content, and multi-task collaborative optimization. Experimental evaluation on MovieLens-25M and Amazon datasets shows that ExplainRec outperforms existing methods, achieving AUC improvements of 0.7\% on movie recommendation and 0.9\% on cross-domain tasks, while generating interpretable explanations and handling cold-start scenarios effectively. 

**Abstract (ZH)**: Recent Advances in Large Language Models for Recommendation Systems: ExplainRec Framework通过偏好归因、多模态融合和零-shot迁移学习扩展基于LLM的推荐能力 

---
# An LLM-Powered Agent for Real-Time Analysis of the Vietnamese IT Job Market 

**Title (ZH)**: 基于LLM的强大代理：越南IT就业市场实时分析 

**Authors**: Minh-Thuan Nguyen, Thien Vo-Thanh, Thai-Duy Dinh, Xuan-Quang Phan, Tan-Ha Mai, Lam-Son Lê  

**Link**: [PDF](https://arxiv.org/pdf/2511.14767)  

**Abstract**: Individuals entering Vietnam's dynamic Information Technology (IT) job market face a critical gap in reliable career guidance. Existing market reports are often outdated, while the manual analysis of thousands of job postings is impractical for most. To address this challenge, we present the AI Job Market Consultant, a novel conversational agent that delivers deep, data-driven insights directly from the labor market in real-time. The foundation of our system is a custom-built dataset created via an automated pipeline that crawls job portals using Playwright and leverages the Large Language Model (LLM) to intelligently structure unstructured posting data. The core of our system is a tool-augmented AI agent, based on the ReAct agentic framework, which enables the ability of autonomously reasoning, planning, and executing actions through a specialized toolbox for SQL queries, semantic search, and data visualization. Our prototype successfully collected and analyzed 3,745 job postings, demonstrating its ability to answer complex, multi-step queries, generate on-demand visualizations, and provide personalized career advice grounded in real-world data. This work introduces a new paradigm for labor market analysis, showcasing how specialized agentic AI systems can democratize access to timely, trustworthy career intelligence for the next generation of professionals. 

**Abstract (ZH)**: 越南动态信息技术（IT）劳动力市场的个体面临可靠职业指导的关键缺口。现有的市场报告往往过时，而手动分析数千份招聘信息对于大多数来说是不切实际的。为应对这一挑战，我们提出了AI就业市场顾问这一新颖的对话代理，能够实时提供深厚的数据驱动见解。我们的系统基础是一个通过自动化管道构建的自定义数据集，该管道使用Playwright抓取招聘信息，并利用大型语言模型（LLM）智能地结构化非结构化招聘信息数据。系统的核心是一个基于ReAct代理框架的工具增强AI代理，该代理能够通过专门的SQL查询、语义搜索和数据可视化工具箱自主进行推理、规划和执行行动。我们的原型成功收集和分析了3,745份招聘信息，展示了其回答复杂多步查询、生成按需可视化和提供基于实际数据的个性化职业建议的能力。这项工作引入了劳动力市场分析的新范式，展示了专门化的代理AI系统如何使下一代专业人员及时可信的职业智能普及化。 

---
# Image-Seeking Intent Prediction for Cross-Device Product Search 

**Title (ZH)**: 跨设备产品搜索中的图像搜索意图预测 

**Authors**: Mariya Hendriksen, Svitlana Vakulenko, Jordan Massiah, Gabriella Kazai, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2511.14764)  

**Abstract**: Large Language Models (LLMs) are transforming personalized search, recommendations, and customer interaction in e-commerce. Customers increasingly shop across multiple devices, from voice-only assistants to multimodal displays, each offering different input and output capabilities. A proactive suggestion to switch devices can greatly improve the user experience, but it must be offered with high precision to avoid unnecessary friction. We address the challenge of predicting when a query requires visual augmentation and a cross-device switch to improve product discovery. We introduce Image-Seeking Intent Prediction, a novel task for LLM-driven e-commerce assistants that anticipates when a spoken product query should proactively trigger a visual on a screen-enabled device. Using large-scale production data from a multi-device retail assistant, including 900K voice queries, associated product retrievals, and behavioral signals such as image carousel engagement, we train IRP (Image Request Predictor), a model that leverages user input query and corresponding retrieved product metadata to anticipate visual intent. Our experiments show that combining query semantics with product data, particularly when improved through lightweight summarization, consistently improves prediction accuracy. Incorporating a differentiable precision-oriented loss further reduces false positives. These results highlight the potential of LLMs to power intelligent, cross-device shopping assistants that anticipate and adapt to user needs, enabling more seamless and personalized e-commerce experiences. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在重塑电子商务中的个性化搜索、推荐和客户互动。随着客户越来越多地跨多种设备购物，从仅支持语音的助手到多模态显示器，每种设备都提供了不同的输入和输出能力。适时地建议切换设备可以显著提升用户体验，但必须非常精准以避免不必要的摩擦。我们解决了一个挑战，即预测何时通过跨设备切换和视觉增强来提升产品发现的需求。我们引入了图像寻求意图预测这一新任务，这是一种由LLM驱动的电子商务助手任务，能够预测何时应主动触发与屏幕设备关联的视觉内容。我们利用多设备零售助手的大规模生产数据，包括90万语音查询、相关的产品检索和行为信号（如图像轮播参与度），训练IRP（图像请求预测器）模型，该模型利用用户输入查询和相应检索的产品元数据来预测视觉意图。我们的实验表明，将查询语义与产品数据相结合，尤其是在通过轻量级总结改进后，能够一致地提高预测准确性。引入可微分的高精度损失进一步降低了误报率。这些结果突显了大型语言模型在驱动智能、跨设备购物助手方面的潜力，这些助手能够预判和适应用户需求，从而提供更加无缝和个性化的电子商务体验。 

---
# Optimizing Agricultural Research: A RAG-Based Approach to Mycorrhizal Fungi Information 

**Title (ZH)**: 基于RAG的方法优化农业研究：关于菌根 fungi 的信息优化 

**Authors**: Mohammad Usman Altam, Md Imtiaz Habib, Tuan Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14765)  

**Abstract**: Retrieval-Augmented Generation (RAG) represents a transformative approach within natural language processing (NLP), combining neural information retrieval with generative language modeling to enhance both contextual accuracy and factual reliability of responses. Unlike conventional Large Language Models (LLMs), which are constrained by static training corpora, RAG-powered systems dynamically integrate domain-specific external knowledge sources, thereby overcoming temporal and disciplinary limitations. In this study, we present the design and evaluation of a RAG-enabled system tailored for Mycophyto, with a focus on advancing agricultural applications related to arbuscular mycorrhizal fungi (AMF). These fungi play a critical role in sustainable agriculture by enhancing nutrient acquisition, improving plant resilience under abiotic and biotic stresses, and contributing to soil health. Our system operationalizes a dual-layered strategy: (i) semantic retrieval and augmentation of domain-specific content from agronomy and biotechnology corpora using vector embeddings, and (ii) structured data extraction to capture predefined experimental metadata such as inoculation methods, spore densities, soil parameters, and yield outcomes. This hybrid approach ensures that generated responses are not only semantically aligned but also supported by structured experimental evidence. To support scalability, embeddings are stored in a high-performance vector database, allowing near real-time retrieval from an evolving literature base. Empirical evaluation demonstrates that the proposed pipeline retrieves and synthesizes highly relevant information regarding AMF interactions with crop systems, such as tomato (Solanum lycopersicum). The framework underscores the potential of AI-driven knowledge discovery to accelerate agroecological innovation and enhance decision-making in sustainable farming systems. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG)在自然语言处理中的革命性方法：结合神经检索与生成语言模型以增强响应的上下文准确性和事实可靠性 

---
# Membership Inference Attack against Large Language Model-based Recommendation Systems: A New Distillation-based Paradigm 

**Title (ZH)**: 面向基于大型语言模型的推荐系统的成员推断攻击：一种新的知识蒸馏范式 

**Authors**: Li Cuihong, Huang Xiaowen, Yin Chuanhuan, Sang Jitao  

**Link**: [PDF](https://arxiv.org/pdf/2511.14763)  

**Abstract**: Membership Inference Attack (MIA) aims to determine if a data sample is used in the training dataset of a target model. Traditional MIA obtains feature of target model via shadow models and uses the feature to train attack model, but the scale and complexity of training or fine-tuning data for large language model (LLM)-based recommendation systems make shadow models difficult to construct. Knowledge distillation as a method for extracting knowledge contributes to construct a stronger reference model. Knowledge distillation enables separate distillation for member and non-member data during the distillation process, enhancing the model's discriminative capability between the two in MIA. This paper propose a knowledge distillation-based MIA paradigm to improve the performance of membership inference attacks on LLM-based recommendation systems. Our paradigm introduces knowledge distillation to obtain a reference model, which enhances the reference model's ability to distinguish between member and non-member data. We obtain individual features from the reference model and train our attack model with fused feature. Our paradigm improves the attack performance of MIA compared to shadow model-based attack. 

**Abstract (ZH)**: 基于知识蒸馏的大型语言模型推荐系统成员推理攻击范式 

---
