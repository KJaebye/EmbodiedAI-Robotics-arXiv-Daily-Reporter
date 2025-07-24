# Thinking Isn't an Illusion: Overcoming the Limitations of Reasoning Models via Tool Augmentations 

**Title (ZH)**: 思考并非幻觉：通过工具增强克服推理模型的局限性 

**Authors**: Zhao Song, Song Yue, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17699)  

**Abstract**: Large Reasoning Models (LRMs) have become a central focus in today's large language model (LLM) research, where models are designed to output a step-by-step thinking process before arriving at a final answer to handle complex reasoning tasks. Despite their promise, recent empirical studies (e.g., [Shojaee et al., 2025] from Apple) suggest that this thinking process may not actually enhance reasoning ability, where LLMs without explicit reasoning actually outperform LRMs on tasks with low or high complexity. In this work, we revisit these findings and investigate whether the limitations of LRMs persist when tool augmentations are introduced. We incorporate two types of tools, Python interpreters and scratchpads, and evaluate three representative LLMs and their LRM counterparts on Apple's benchmark reasoning puzzles. Our results show that, with proper tool use, LRMs consistently outperform their non-reasoning counterparts across all levels of task complexity. These findings challenge the recent narrative that reasoning is an illusion and highlight the potential of tool-augmented LRMs for solving complex problems. 

**Abstract (ZH)**: 大型推理模型 (LRMs) 已成为当今大型语言模型 (LLMs) 研究的中心焦点，模型旨在在得出最终答案之前输出逐步推理过程以处理复杂的推理任务。尽管LRMs充满潜力，但近期的经验研究表明，这种推理过程可能实际上并未增强推理能力，未显式进行推理的LLMs在低复杂度或高复杂度任务中反而表现更佳。在本文中，我们重新审视这些发现，并调查引入工具辅助后LRMs的限制是否仍然存在。我们引入了Python解释器和草稿纸两种类型的工具，并在苹果公司的基准推理谜题上评估了三种代表性的LLMs及其对应的LRMs。结果表明，通过适当使用工具，LRMs在所有复杂度级别上均能超越其非推理版本。这些发现挑战了最近关于推理是一个幻觉的说法，并突显了工具增强的LRMs解决复杂问题的潜力。 

---
# Simulating multiple human perspectives in socio-ecological systems using large language models 

**Title (ZH)**: 使用大型语言模型在社会-生态系统中模拟多重人类视角 

**Authors**: Yongchao Zeng, Calum Brown, Ioannis Kyriakou, Ronja Hotz, Mark Rounsevell  

**Link**: [PDF](https://arxiv.org/pdf/2507.17680)  

**Abstract**: Understanding socio-ecological systems requires insights from diverse stakeholder perspectives, which are often hard to access. To enable alternative, simulation-based exploration of different stakeholder perspectives, we develop the HoPeS (Human-Oriented Perspective Shifting) modelling framework. HoPeS employs agents powered by large language models (LLMs) to represent various stakeholders; users can step into the agent roles to experience perspectival differences. A simulation protocol serves as a "scaffold" to streamline multiple perspective-taking simulations, supporting users in reflecting on, transitioning between, and integrating across perspectives. A prototype system is developed to demonstrate HoPeS in the context of institutional dynamics and land use change, enabling both narrative-driven and numerical experiments. In an illustrative experiment, a user successively adopts the perspectives of a system observer and a researcher - a role that analyses data from the embedded land use model to inform evidence-based decision-making for other LLM agents representing various institutions. Despite the user's effort to recommend technically sound policies, discrepancies persist between the policy recommendation and implementation due to stakeholders' competing advocacies, mirroring real-world misalignment between researcher and policymaker perspectives. The user's reflection highlights the subjective feelings of frustration and disappointment as a researcher, especially due to the challenge of maintaining political neutrality while attempting to gain political influence. Despite this, the user exhibits high motivation to experiment with alternative narrative framing strategies, suggesting the system's potential in exploring different perspectives. Further system and protocol refinement are likely to enable new forms of interdisciplinary collaboration in socio-ecological simulations. 

**Abstract (ZH)**: 理解社会生态系统需要从多利益相关方视角获取见解，但这些视角往往难以获取。为使不同利益相关方视角的替代性、基于模拟的探索成为可能，我们开发了HoPeS（Human-Oriented Perspective Shifting）建模框架。HoPeS 利用大型语言模型（LLMs）驱动的代理来代表各种利益相关方；用户可以扮演这些代理角色以体验不同的视角差异。一种模拟协议作为“支架”，简化了多视角换位模拟的流程，支持用户反思、转换和整合不同视角。一个原型系统被开发出来，以展示HoPeS在机构动态和土地利用变化中的应用，允许进行基于叙事和数值的实验。在一个示例性实验中，用户依次扮演系统观察者和研究人员的角色——后者基于嵌入的土地利用模型的数据进行分析，以指导其他代表各种机构的LLM代理做出基于证据的决策。尽管用户努力推荐技术上合理的政策，但由于利益相关方的竞争性诉求，政策建议与实施之间仍存在差异，这反映了研究人员和政策制定者视角间的真实世界偏差。用户的反思突显了作为研究人员的主观挫败感和失望感，尤其是在试图保持政治中立的同时争取政治影响力方面遇到的挑战。然而，尽管如此，用户显示出很高的实验不同叙事框架策略的动力，这表明系统在探索不同视角方面具有潜力。进一步的系统和协议改进很可能使跨学科协作在社会生态模拟中成为可能。 

---
# Constructing Ophthalmic MLLM for Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reasoning 

**Title (ZH)**: 通过临床认知链推理构建眼科MLLM以实现定位-诊断协作 

**Authors**: Xinyao Liu, Diping Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.17539)  

**Abstract**: Multimodal large language models (MLLMs) demonstrate significant potential in the field of medical diagnosis. However, they face critical challenges in specialized domains such as ophthalmology, particularly the fragmentation of annotation granularity and inconsistencies in clinical reasoning logic, which hinder precise cross-modal understanding. This paper introduces FundusExpert, an ophthalmology-specific MLLM with integrated positioning-diagnosis reasoning capabilities, along with FundusGen, a dataset constructed through the intelligent Fundus-Engine system. Fundus-Engine automates localization and leverages MLLM-based semantic expansion to integrate global disease classification, local object detection, and fine-grained feature analysis within a single fundus image. Additionally, by constructing a clinically aligned cognitive chain, it guides the model to generate interpretable reasoning paths. FundusExpert, fine-tuned with instruction data from FundusGen, achieves the best performance in ophthalmic question-answering tasks, surpassing the average accuracy of the 40B MedRegA by 26.6%. It also excels in zero-shot report generation tasks, achieving a clinical consistency of 77.0%, significantly outperforming GPT-4o's 47.6%. Furthermore, we reveal a scaling law between data quality and model capability ($L \propto N^{0.068}$), demonstrating that the cognitive alignment annotations in FundusGen enhance data utilization efficiency. By integrating region-level localization with diagnostic reasoning chains, our work develops a scalable, clinically-aligned MLLM and explores a pathway toward bridging the visual-language gap in specific MLLMs. Our project can be found at this https URL. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在医疗诊断领域展现出显著潜力，特别是在眼科这一专业领域。然而，它们在眼科领域面临着标注细粒度碎片化和临床推理逻辑不一致等关键挑战，这阻碍了跨模态的精确理解。本文介绍了眼科专用的MLLM——FundusExpert，以及通过智能Fundus-Engine系统构建的FundusGen数据集。Fundus-Engine实现了区域级别的定位，并借助基于MLLM的语义扩展，在单张眼底图像中整合了全球疾病分类、局部对象检测和精细特征分析。此外，通过构建临床对齐的认知链条，它引导模型生成可解释的推理路径。经过FundusGen指令数据微调后，FundusExpert在眼科问答任务中取得了最佳性能，超越了平均准确率为40B MedRegA的26.6%。同时，在零样本报告生成任务中，其临床一致性为77.0%，远远超过GPT-4o的47.6%。此外，我们揭示了数据质量和模型能力之间的标度律（$L \propto N^{0.068}$），表明FundusGen中的认知对齐标注提高了数据利用效率。通过结合区域级别的定位和诊断推理链条，我们的工作开发了一种可扩展、临床对齐的MLLM，并探索了特定MLLM中视觉-语言差距的桥梁构建路径。更多信息请访问：this https URL。 

---
# Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning 

**Title (ZH)**: 一域之畔，众域受益：基于数据的多域推理研究——通过强化学习实现 

**Authors**: Yu Li, Zhuoshi Pan, Honglin Lin, Mengyuan Sun, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17512)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of LLMs. Existing research has predominantly concentrated on isolated reasoning domains such as mathematical problem-solving, coding tasks, or logical reasoning. However, real world reasoning scenarios inherently demand an integrated application of multiple cognitive skills. Despite this, the interplay among these reasoning skills under reinforcement learning remains poorly understood. To bridge this gap, we present a systematic investigation of multi-domain reasoning within the RLVR framework, explicitly focusing on three primary domains: mathematical reasoning, code generation, and logical puzzle solving. We conduct a comprehensive study comprising four key components: (1) Leveraging the GRPO algorithm and the Qwen-2.5-7B model family, our study thoroughly evaluates the models' in-domain improvements and cross-domain generalization capabilities when trained on single-domain datasets. (2) Additionally, we examine the intricate interactions including mutual enhancements and conflicts that emerge during combined cross-domain training. (3) To further understand the influence of SFT on RL, we also analyze and compare performance differences between base and instruct models under identical RL configurations. (4) Furthermore, we delve into critical RL training details, systematically exploring the impacts of curriculum learning strategies, variations in reward design, and language-specific factors. Through extensive experiments, our results offer significant insights into the dynamics governing domain interactions, revealing key factors influencing both specialized and generalizable reasoning performance. These findings provide valuable guidance for optimizing RL methodologies to foster comprehensive, multi-domain reasoning capabilities in LLMs. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）已经 emerged 作为增强大型语言模型推理能力的强大范式。现有研究主要集中在数学问题解决、编码任务或逻辑推理等孤立的推理领域。然而，现实世界的推理场景要求多种认知技能的综合应用。尽管如此，这些推理技能在强化学习中的相互作用依然知之甚少。为弥合这一差距，我们系统地探讨了 RLVR 框架下的多域推理，明确专注于三个主要领域：数学推理、代码生成和逻辑谜题解决。我们的研究包括四个关键组成部分：（1）利用 GRPO 算法和 Qwen-2.5-7B 模型家族，我们全面评估了单域数据集训练下模型的领域内改进和跨域泛化能力；（2）此外，我们还研究了在联合跨域训练过程中出现的复杂交互，包括相互增强和冲突；（3）为进一步理解 SFT 对 RL 的影响，我们还分析并比较了在相同 RL 配置下基础模型和指令模型的性能差异；（4）此外，我们深入探讨了关键的 RL 训练细节，系统地探索了逐级学习策略、奖励设计的变化以及语言特定因素的影响。通过大量实验，我们的结果提供了关于领域交互动力学的重要见解，揭示了影响专业和泛化推理性能的关键因素。这些发现为优化 RL 方法以促进大型语言模型的全面、多域推理能力提供了宝贵的指导。 

---
# An Uncertainty-Driven Adaptive Self-Alignment Framework for Large Language Models 

**Title (ZH)**: 基于不确定性驱动的自适应自我对齐框架：应用于大型语言模型 

**Authors**: Haoran Sun, Zekun Zhang, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.17477)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable progress in instruction following and general-purpose reasoning. However, achieving high-quality alignment with human intent and safety norms without human annotations remains a fundamental challenge. In this work, we propose an Uncertainty-Driven Adaptive Self-Alignment (UDASA) framework designed to improve LLM alignment in a fully automated manner. UDASA first generates multiple responses for each input and quantifies output uncertainty across three dimensions: semantics, factuality, and value alignment. Based on these uncertainty scores, the framework constructs preference pairs and categorizes training samples into three stages, conservative, moderate, and exploratory, according to their uncertainty difference. The model is then optimized progressively across these stages. In addition, we conduct a series of preliminary studies to validate the core design assumptions and provide strong empirical motivation for the proposed framework. Experimental results show that UDASA outperforms existing alignment methods across multiple tasks, including harmlessness, helpfulness, truthfulness, and controlled sentiment generation, significantly improving model performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在指令遵循和通用推理方面取得了显著进展。然而，在没有人类标注的情况下实现与人类意图和安全规范的高质量化身仍是一个基本挑战。本文提出了一种不确定性驱动的自适应自我对齐（UDASA）框架，旨在以完全自动的方式改进LLM对齐。UDASA首先为每个输入生成多个响应，并从语义、事实性和价值对齐三个维度量化输出不确定性。基于这些不确定性评分，框架构建偏好对并根据不确定性差异将训练样本分为保守、适度和探索性三个阶段。然后，模型在这三个阶段中逐步优化。此外，我们进行了一系列初步研究以验证核心设计假设，并为提出的框架提供了强大的实证支持。实验结果表明，UDASA在多个任务（包括无害性、帮助性、真实性以及可控情感生成）上优于现有对齐方法，显著提高了模型性能。 

---
# Compliance Brain Assistant: Conversational Agentic AI for Assisting Compliance Tasks in Enterprise Environments 

**Title (ZH)**: 合规大脑助手：企业环境中辅助合规任务的对话型代理AI 

**Authors**: Shitong Zhu, Chenhao Fang, Derek Larson, Neel Reddy Pochareddy, Rajeev Rao, Sophie Zeng, Yanqing Peng, Wendy Summer, Alex Goncalves, Arya Pudota, Herve Robert  

**Link**: [PDF](https://arxiv.org/pdf/2507.17289)  

**Abstract**: This paper presents Compliance Brain Assistant (CBA), a conversational, agentic AI assistant designed to boost the efficiency of daily compliance tasks for personnel in enterprise environments. To strike a good balance between response quality and latency, we design a user query router that can intelligently choose between (i) FastTrack mode: to handle simple requests that only need additional relevant context retrieved from knowledge corpora; and (ii) FullAgentic mode: to handle complicated requests that need composite actions and tool invocations to proactively discover context across various compliance artifacts, and/or involving other APIs/models for accommodating requests. A typical example would be to start with a user query, use its description to find a specific entity and then use the entity's information to query other APIs for curating and enriching the final AI response.
Our experimental evaluations compared CBA against an out-of-the-box LLM on various real-world privacy/compliance-related queries targeting various personas. We found that CBA substantially improved upon the vanilla LLM's performance on metrics such as average keyword match rate (83.7% vs. 41.7%) and LLM-judge pass rate (82.0% vs. 20.0%). We also compared metrics for the full routing-based design against the `fast-track only` and `full-agentic` modes and found that it had a better average match-rate and pass-rate while keeping the run-time approximately the same. This finding validated our hypothesis that the routing mechanism leads to a good trade-off between the two worlds. 

**Abstract (ZH)**: 基于对话的企业环境合规助手：Compliance Brain Assistant 

---
# Agent Identity Evals: Measuring Agentic Identity 

**Title (ZH)**: 代理身份评估：测量代理身份 

**Authors**: Elija Perrier, Michael Timothy Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2507.17257)  

**Abstract**: Central to agentic capability and trustworthiness of language model agents (LMAs) is the extent they maintain stable, reliable, identity over time. However, LMAs inherit pathologies from large language models (LLMs) (statelessness, stochasticity, sensitivity to prompts and linguistically-intermediation) which can undermine their identifiability, continuity, persistence and consistency. This attrition of identity can erode their reliability, trustworthiness and utility by interfering with their agentic capabilities such as reasoning, planning and action. To address these challenges, we introduce \textit{agent identity evals} (AIE), a rigorous, statistically-driven, empirical framework for measuring the degree to which an LMA system exhibit and maintain their agentic identity over time, including their capabilities, properties and ability to recover from state perturbations. AIE comprises a set of novel metrics which can integrate with other measures of performance, capability and agentic robustness to assist in the design of optimal LMA infrastructure and scaffolding such as memory and tools. We set out formal definitions and methods that can be applied at each stage of the LMA life-cycle, and worked examples of how to apply them. 

**Abstract (ZH)**: 语言模型代理（LMAs）的代理能力和可信度的核心在于它们在时间上维持稳定可靠身份的程度。然而，LMAs 从大型语言模型（LLMs）继承了病态特征（无状态性、随机性、对提示的敏感性和语义中介性），这些特征可能削弱它们的身份可识别性、连续性、持久性和一致性。这种身份的流失可能通过干扰它们的代理能力（如推理、规划和行动）来削弱它们的可靠性、可信度和实用性。为应对这些挑战，我们引入了“代理身份评估”（AIE），这是一种严谨的、以统计为驱动的实证框架，用于衡量LMA系统在其运行过程中展现和维持其代理身份的程度，包括其能力、属性及其从状态干扰中恢复的能力。AIE 包含一系列新型度量标准，可以与其他性能、能力和代理鲁棒性的度量标准结合使用，以协助设计最优的LMA基础设施和支撑工具（如记忆和工具）。我们提出了形式化定义和方法，可以在LMA生命周期的每个阶段应用，并提供了如何应用这些方法的实例。 

---
# Improving LLMs' Generalized Reasoning Abilities by Graph Problems 

**Title (ZH)**: 通过图问题提高LLMs的泛化推理能力 

**Authors**: Qifan Zhang, Nuo Chen, Zehua Li, Miao Peng, Jing Tang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.17168)  

**Abstract**: Large Language Models (LLMs) have made remarkable strides in reasoning tasks, yet their performance often falters on novel and complex problems. Domain-specific continued pretraining (CPT) methods, such as those tailored for mathematical reasoning, have shown promise but lack transferability to broader reasoning tasks. In this work, we pioneer the use of Graph Problem Reasoning (GPR) to enhance the general reasoning capabilities of LLMs. GPR tasks, spanning pathfinding, network analysis, numerical computation, and topological reasoning, require sophisticated logical and relational reasoning, making them ideal for teaching diverse reasoning patterns. To achieve this, we introduce GraphPile, the first large-scale corpus specifically designed for CPT using GPR data. Spanning 10.9 billion tokens across 23 graph tasks, the dataset includes chain-of-thought, program-of-thought, trace of execution, and real-world graph data. Using GraphPile, we train GraphMind on popular base models Llama 3 and 3.1, as well as Gemma 2, achieving up to 4.9 percent higher accuracy in mathematical reasoning and up to 21.2 percent improvement in non-mathematical reasoning tasks such as logical and commonsense reasoning. By being the first to harness GPR for enhancing reasoning patterns and introducing the first dataset of its kind, our work bridges the gap between domain-specific pretraining and universal reasoning capabilities, advancing the adaptability and robustness of LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理任务中取得了显著进展，但在处理新颖和复杂问题时表现往往不佳。针对特定领域的持续预训练（CPT）方法，如数学推理领域，显示出了潜力，但缺乏向更广泛推理任务的迁移能力。本工作中，我们首次采用图问题推理（GPR）来增强LLMs的一般推理能力。GPR任务涵盖路径查找、网络分析、数值计算和拓扑推理等内容，需要复杂的逻辑和关系推理，使其成为传授多样化推理模式的理想选择。为此，我们引入了GraphPile，这是首个专门用于GPR数据的大型语料库，用于CPT。GraphPile包含来自23个图任务的109亿个 Tokens，其中包括推理链、思维程序、执行轨迹以及实际图数据。利用GraphPile，我们在流行的基模型Llama 3和3.1以及Gemma 2上训练了GraphMind，数学推理准确率提高了4.9%，非数学推理任务，如逻辑推理和常识推理，提高了21.2%。通过首次利用GPR来增强推理模式，并引入首个此类数据集，我们的工作填补了领域特定预训练与通用推理能力之间的空白，推动了LLMs的适应性和 robustness。 

---
# LoRA is All You Need for Safety Alignment of Reasoning LLMs 

**Title (ZH)**: LoRA即为实现推理大语言模型安全对齐所需的一切 

**Authors**: Yihao Xue, Baharan Mirzasoleiman  

**Link**: [PDF](https://arxiv.org/pdf/2507.17075)  

**Abstract**: Reasoning LLMs have demonstrated remarkable breakthroughs in solving complex problems that were previously out of reach. To ensure LLMs do not assist with harmful requests, safety alignment fine-tuning is necessary in the post-training phase. However, safety alignment fine-tuning has recently been shown to significantly degrade reasoning abilities, a phenomenon known as the "Safety Tax". In this work, we show that using LoRA for SFT on refusal datasets effectively aligns the model for safety without harming its reasoning capabilities. This is because restricting the safety weight updates to a low-rank space minimizes the interference with the reasoning weights. Our extensive experiments across four benchmarks covering math, science, and coding show that this approach produces highly safe LLMs -- with safety levels comparable to full-model fine-tuning -- without compromising their reasoning abilities. Additionally, we observe that LoRA induces weight updates with smaller overlap with the initial weights compared to full-model fine-tuning. We also explore methods that further reduce such overlap -- via regularization or during weight merging -- and observe some improvement on certain tasks. We hope this result motivates designing approaches that yield more consistent improvements in the reasoning-safety trade-off. 

**Abstract (ZH)**: 利用LoRA进行拒绝数据集上的SFT有效提升了模型的安全性而不损害其推理能力 

---
# CASCADE: LLM-Powered JavaScript Deobfuscator at Google 

**Title (ZH)**: CASCADE：由大规模语言模型驱动的JavaScript去混淆工具（Google） 

**Authors**: Shan Jiang, Pranoy Kovuri, David Tao, Zhixun Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17691)  

**Abstract**: Software obfuscation, particularly prevalent in JavaScript, hinders code comprehension and analysis, posing significant challenges to software testing, static analysis, and malware detection. This paper introduces CASCADE, a novel hybrid approach that integrates the advanced coding capabilities of Gemini with the deterministic transformation capabilities of a compiler Intermediate Representation (IR), specifically JavaScript IR (JSIR). By employing Gemini to identify critical prelude functions, the foundational components underlying the most prevalent obfuscation techniques, and leveraging JSIR for subsequent code transformations, CASCADE effectively recovers semantic elements like original strings and API names, and reveals original program behaviors. This method overcomes limitations of existing static and dynamic deobfuscation techniques, eliminating hundreds to thousands of hardcoded rules while achieving reliability and flexibility. CASCADE is already deployed in Google's production environment, demonstrating substantial improvements in JavaScript deobfuscation efficiency and reducing reverse engineering efforts. 

**Abstract (ZH)**: 软件混淆，特别是在JavaScript中普遍存在，阻碍了代码理解和分析，对软件测试、静态分析和恶意软件检测构成了重大挑战。本文介绍了CASCADE，这是一种新颖的混合方法，结合了Gemini高级编码能力和编译器中间表示（IR）的确定性转换能力，特别是JavaScript IR（JSIR）。通过使用Gemini来识别关键的前导函数，即最常见混淆技术的基础组件，并结合JSIR进行后续代码转换，CASCADE有效地恢复了如原始字符串和API名称等语义元素，并揭示了原始程序行为。该方法克服了现有静态和动态去混淆技术的局限性，消除了成千上万的手动硬编码规则，同时保持了可靠性和灵活性。CASCADE已经在Google的生产环境中部署，显示出JavaScript去混淆效率的显著提升，并减少了逆向工程的工作量。 

---
# Enabling Cyber Security Education through Digital Twins and Generative AI 

**Title (ZH)**: 通过数字孪生和生成式AI赋能网络安全教育 

**Authors**: Vita Santa Barletta, Vito Bavaro, Miriana Calvano, Antonio Curci, Antonio Piccinno, Davide Pio Posa  

**Link**: [PDF](https://arxiv.org/pdf/2507.17518)  

**Abstract**: Digital Twins (DTs) are gaining prominence in cybersecurity for their ability to replicate complex IT (Information Technology), OT (Operational Technology), and IoT (Internet of Things) infrastructures, allowing for real time monitoring, threat analysis, and system simulation. This study investigates how integrating DTs with penetration testing tools and Large Language Models (LLMs) can enhance cybersecurity education and operational readiness. By simulating realistic cyber environments, this approach offers a practical, interactive framework for exploring vulnerabilities and defensive strategies. At the core of this research is the Red Team Knife (RTK), a custom penetration testing toolkit aligned with the Cyber Kill Chain model. RTK is designed to guide learners through key phases of cyberattacks, including reconnaissance, exploitation, and response within a DT powered ecosystem. The incorporation of Large Language Models (LLMs) further enriches the experience by providing intelligent, real-time feedback, natural language threat explanations, and adaptive learning support during training exercises. This combined DT LLM framework is currently being piloted in academic settings to develop hands on skills in vulnerability assessment, threat detection, and security operations. Initial findings suggest that the integration significantly improves the effectiveness and relevance of cybersecurity training, bridging the gap between theoretical knowledge and real-world application. Ultimately, the research demonstrates how DTs and LLMs together can transform cybersecurity education to meet evolving industry demands. 

**Abstract (ZH)**: 数字孪生(DTs)在网络安全中的应用及其与渗透测试工具和大型语言模型(LLMs)的整合：增强网络安全教育与操作准备 

---
# Reasoning-Driven Retrosynthesis Prediction with Large Language Models via Reinforcement Learning 

**Title (ZH)**: 基于强化学习的大语言模型驱动的逆合成反应预测 

**Authors**: Situo Zhang, Hanqi Li, Lu Chen, Zihan Zhao, Xuanze Lin, Zichen Zhu, Bo Chen, Xin Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17448)  

**Abstract**: Retrosynthesis planning, essential in organic synthesis and drug discovery, has greatly benefited from recent AI-driven advancements. Nevertheless, existing methods frequently face limitations in both applicability and explainability. Traditional graph-based and sequence-to-sequence models often lack generalized chemical knowledge, leading to predictions that are neither consistently accurate nor easily explainable. To address these challenges, we introduce RetroDFM-R, a reasoning-based large language model (LLM) designed specifically for chemical retrosynthesis. Leveraging large-scale reinforcement learning guided by chemically verifiable rewards, RetroDFM-R significantly enhances prediction accuracy and explainability. Comprehensive evaluations demonstrate that RetroDFM-R significantly outperforms state-of-the-art methods, achieving a top-1 accuracy of 65.0% on the USPTO-50K benchmark. Double-blind human assessments further validate the chemical plausibility and practical utility of RetroDFM-R's predictions. RetroDFM-R also accurately predicts multistep retrosynthetic routes reported in the literature for both real-world drug molecules and perovskite materials. Crucially, the model's explicit reasoning process provides human-interpretable insights, thereby enhancing trust and practical value in real-world retrosynthesis applications. 

**Abstract (ZH)**: retrosynthesis 计划对于有机合成和药物发现至关重要，近年来得益于人工智能驱动的进步。然而，现有方法在适用性和可解释性方面仍面临诸多限制。传统的基于图和序列到序列的模型往往缺乏泛化的化学知识，导致预测既不够准确也不易解释。为了应对这些挑战，我们提出了一种专用的化学 retrosynthesis 原理解释型大型语言模型（LLM）——RetroDFM-R。通过大规模基于化学验证奖励的强化学习，RetroDFM-R 显著提升了预测准确性和可解释性。全面评估表明，RetroDFM-R 显著优于现有最先进的方法，在 USPTO-50K 基准上的顶级准确率达到了 65.0%。双盲的人类评估进一步验证了 RetroDFM-R 预测的化学可行性和实际应用价值。RetroDFM-R 还准确预测了文献中报道的真实药物分子和钙钛矿材料的多步 retrosynthetic 路径。至关重要的是，模型的明确推理过程提供了可由人类理解的见解，从而增强了在实际 retrosynthesis 应用中的信任和实际价值。 

---
# Each to Their Own: Exploring the Optimal Embedding in RAG 

**Title (ZH)**: 各有所适：探究RAG中的最优嵌入方式 

**Authors**: Shiting Chen, Zijian Zhao, Jinsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.17442)  

**Abstract**: Recently, as Large Language Models (LLMs) have fundamentally impacted various fields, the methods for incorporating up-to-date information into LLMs or adding external knowledge to construct domain-specific models have garnered wide attention. Retrieval-Augmented Generation (RAG), serving as an inference-time scaling method, is notable for its low cost and minimal effort for parameter tuning. However, due to heterogeneous training data and model architecture, the variant embedding models used in RAG exhibit different benefits across various areas, often leading to different similarity calculation results and, consequently, varying response quality from LLMs. To address this problem, we propose and examine two approaches to enhance RAG by combining the benefits of multiple embedding models, named Mixture-Embedding RAG and Confident RAG. Mixture-Embedding RAG simply sorts and selects retrievals from multiple embedding models based on standardized similarity; however, it does not outperform vanilla RAG. In contrast, Confident RAG generates responses multiple times using different embedding models and then selects the responses with the highest confidence level, demonstrating average improvements of approximately 10% and 5% over vanilla LLMs and RAG, respectively. The consistent results across different LLMs and embedding models indicate that Confident RAG is an efficient plug-and-play approach for various domains. We will release our code upon publication. 

**Abstract (ZH)**: 近期，随着大型语言模型（LLMs）对各个领域产生了根本性影响，将最新信息融入LLMs或通过添加外部知识构建领域特定模型的方法受到了广泛的关注。检索增强生成（RAG）作为一种推理时扩展方法，以其低成本和最小参数调优努力而著称。但由于训练数据和模型架构的异质性，RAG中使用的多种嵌入模型在不同领域表现出不同的优势，导致不同的相似度计算结果和LLMs响应质量的差异。为了解决这一问题，我们提出了两种结合多个嵌入模型优点的方法，分别是混合嵌入RAG和自信RAG。混合嵌入RAG基于标准化相似度简单地排序和选择来自多个嵌入模型的检索结果，但并未超过传统的RAG。相比之下，自信RAG利用不同的嵌入模型多次生成响应，然后选择具有最高置信度的响应，相对于传统的LLMs和RAG分别显示出约10%和5%的平均改进。不同LLMs和嵌入模型的一致性结果表明，自信RAG是一种高效可插拔的方法，适用于各种领域。论文发表后我们将发布我们的代码。 

---
# HiProbe-VAD: Video Anomaly Detection via Hidden States Probing in Tuning-Free Multimodal LLMs 

**Title (ZH)**: HiProbe-VAD: 通过调优无干预的多模态LLM中隐藏状态探测进行视频异常检测 

**Authors**: Zhaolin Cai, Fan Li, Ziwei Zheng, Yanjun Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.17394)  

**Abstract**: Video Anomaly Detection (VAD) aims to identify and locate deviations from normal patterns in video sequences. Traditional methods often struggle with substantial computational demands and a reliance on extensive labeled datasets, thereby restricting their practical applicability. To address these constraints, we propose HiProbe-VAD, a novel framework that leverages pre-trained Multimodal Large Language Models (MLLMs) for VAD without requiring fine-tuning. In this paper, we discover that the intermediate hidden states of MLLMs contain information-rich representations, exhibiting higher sensitivity and linear separability for anomalies compared to the output layer. To capitalize on this, we propose a Dynamic Layer Saliency Probing (DLSP) mechanism that intelligently identifies and extracts the most informative hidden states from the optimal intermediate layer during the MLLMs reasoning. Then a lightweight anomaly scorer and temporal localization module efficiently detects anomalies using these extracted hidden states and finally generate explanations. Experiments on the UCF-Crime and XD-Violence datasets demonstrate that HiProbe-VAD outperforms existing training-free and most traditional approaches. Furthermore, our framework exhibits remarkable cross-model generalization capabilities in different MLLMs without any tuning, unlocking the potential of pre-trained MLLMs for video anomaly detection and paving the way for more practical and scalable solutions. 

**Abstract (ZH)**: 基于预训练多模态大语言模型的无训练视频异常检测（HiProbe-VAD） 

---
# DynaSearcher: Dynamic Knowledge Graph Augmented Search Agent via Multi-Reward Reinforcement Learning 

**Title (ZH)**: DynaSearcher: 动态知识图谱增强的多奖励强化学习搜索代理 

**Authors**: Chuzhan Hao, Wenfeng Feng, Yuewei Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17365)  

**Abstract**: Multi-step agentic retrieval systems based on large language models (LLMs) have demonstrated remarkable performance in complex information search tasks. However, these systems still face significant challenges in practical applications, particularly in generating factually inconsistent intermediate queries and inefficient search trajectories, which can lead to reasoning deviations or redundant computations. To address these issues, we propose DynaSearcher, an innovative search agent enhanced by dynamic knowledge graphs and multi-reward reinforcement learning (RL). Specifically, our system leverages knowledge graphs as external structured knowledge to guide the search process by explicitly modeling entity relationships, thereby ensuring factual consistency in intermediate queries and mitigating biases from irrelevant information. Furthermore, we employ a multi-reward RL framework for fine-grained control over training objectives such as retrieval accuracy, efficiency, and response quality. This framework promotes the generation of high-quality intermediate queries and comprehensive final answers, while discouraging unnecessary exploration and minimizing information omissions or redundancy. Experimental results demonstrate that our approach achieves state-of-the-art answer accuracy on six multi-hop question answering datasets, matching frontier LLMs while using only small-scale models and limited computational resources. Furthermore, our approach demonstrates strong generalization and robustness across diverse retrieval environments and larger-scale models, highlighting its broad applicability. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的多步代理检索系统在复杂信息搜索任务中展现了出色的性能。然而，这些系统在实际应用中仍然面临着显著挑战，特别是在生成事实不一致的中间查询和低效的搜索轨迹方面，这可能导致推理偏差或重复计算。为了解决这些问题，我们提出了一种名为DynaSearcher的创新搜索代理，该代理融合了动态知识图和多奖励强化学习（RL）。具体来说，我们的系统利用知识图作为外部结构化知识来指导搜索过程，通过明确建模实体关系确保中间查询的事实一致性，并减轻无关信息带来的偏差。此外，我们采用多奖励RL框架对检索准确性、效率和响应质量等细粒度训练目标进行精确控制。该框架促进了高质量中间查询和全面最终答案的生成，并抑制不必要的探索，减少信息遗漏或冗余。实验结果表明，我们的方法在六个多跳问答数据集上实现了最先进的答案准确性，仅使用小型模型和有限的计算资源就匹配了前沿LLMs。此外，我们的方法在不同检索环境和更大规模模型中的泛化能力和鲁棒性较强，突显了其广泛的应用潜力。 

---
# A Versatile Pathology Co-pilot via Reasoning Enhanced Multimodal Large Language Model 

**Title (ZH)**: 一种基于推理增强多模态大型语言模型的通用病理协驾系统 

**Authors**: Zhe Xu, Ziyi Liu, Junlin Hou, Jiabo Ma, Cheng Jin, Yihui Wang, Zhixuan Chen, Zhengyu Zhang, Zhengrui Guo, Fengtao Zhou, Yingxue Xu, Xi Wang, Ronald Cheong Kin Chan, Li Liang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.17303)  

**Abstract**: Multimodal large language models (MLLMs) have emerged as powerful tools for computational pathology, offering unprecedented opportunities to integrate pathological images with language context for comprehensive diagnostic analysis. These models hold particular promise for automating complex tasks that traditionally require expert interpretation of pathologists. However, current MLLM approaches in pathology demonstrate significantly constrained reasoning capabilities, primarily due to their reliance on expensive chain-of-thought annotations. Additionally, existing methods remain limited to simplex application of visual question answering (VQA) at region-of-interest (ROI) level, failing to address the full spectrum of diagnostic needs such as ROI classification, detection, segmentation, whole-slide-image (WSI) classification and VQA in clinical practice. In this study, we present SmartPath-R1, a versatile MLLM capable of simultaneously addressing both ROI-level and WSI-level tasks while demonstrating robust pathological reasoning capability. Our framework combines scale-dependent supervised fine-tuning and task-aware reinforcement fine-tuning, which circumvents the requirement for chain-of-thought supervision by leveraging the intrinsic knowledge within MLLM. Furthermore, SmartPath-R1 integrates multiscale and multitask analysis through a mixture-of-experts mechanism, enabling dynamic processing for diverse tasks. We curate a large-scale dataset comprising 2.3M ROI samples and 188K WSI samples for training and evaluation. Extensive experiments across 72 tasks validate the effectiveness and superiority of the proposed approach. This work represents a significant step toward developing versatile, reasoning-enhanced AI systems for precision pathology. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在病理学中的应用：一种同时解决ROI级和WSI级任务的智能推理方法 

---
# Leveraging Knowledge Graphs and LLM Reasoning to Identify Operational Bottlenecks for Warehouse Planning Assistance 

**Title (ZH)**: 利用知识图谱和大规模语言模型推理识别仓储规划辅助中的运营瓶颈 

**Authors**: Rishi Parekh, Saisubramaniam Gopalakrishnan, Zishan Ahmad, Anirudh Deodhar  

**Link**: [PDF](https://arxiv.org/pdf/2507.17273)  

**Abstract**: Analyzing large, complex output datasets from Discrete Event Simulations (DES) of warehouse operations to identify bottlenecks and inefficiencies is a critical yet challenging task, often demanding significant manual effort or specialized analytical tools. Our framework integrates Knowledge Graphs (KGs) and Large Language Model (LLM)-based agents to analyze complex Discrete Event Simulation (DES) output data from warehouse operations. It transforms raw DES data into a semantically rich KG, capturing relationships between simulation events and entities. An LLM-based agent uses iterative reasoning, generating interdependent sub-questions. For each sub-question, it creates Cypher queries for KG interaction, extracts information, and self-reflects to correct errors. This adaptive, iterative, and self-correcting process identifies operational issues mimicking human analysis. Our DES approach for warehouse bottleneck identification, tested with equipment breakdowns and process irregularities, outperforms baseline methods. For operational questions, it achieves near-perfect pass rates in pinpointing inefficiencies. For complex investigative questions, we demonstrate its superior diagnostic ability to uncover subtle, interconnected issues. This work bridges simulation modeling and AI (KG+LLM), offering a more intuitive method for actionable insights, reducing time-to-insight, and enabling automated warehouse inefficiency evaluation and diagnosis. 

**Abstract (ZH)**: 基于知识图谱和大语言模型代理的仓库操作离散事件模拟复杂输出数据分析框架 

---
# Understanding Prompt Programming Tasks and Questions 

**Title (ZH)**: 理解提示编程任务和问题 

**Authors**: Jenny T. Liang, Chenyang Yang, Agnia Sergeyuk, Travis D. Breaux, Brad A. Myers  

**Link**: [PDF](https://arxiv.org/pdf/2507.17264)  

**Abstract**: Prompting foundation models (FMs) like large language models (LLMs) have enabled new AI-powered software features (e.g., text summarization) that previously were only possible by fine-tuning FMs. Now, developers are embedding prompts in software, known as prompt programs. The process of prompt programming requires the developer to make many changes to their prompt. Yet, the questions developers ask to update their prompt is unknown, despite the answers to these questions affecting how developers plan their changes. With the growing number of research and commercial prompt programming tools, it is unclear whether prompt programmers' needs are being adequately addressed. We address these challenges by developing a taxonomy of 25 tasks prompt programmers do and 51 questions they ask, measuring the importance of each task and question. We interview 16 prompt programmers, observe 8 developers make prompt changes, and survey 50 developers. We then compare the taxonomy with 48 research and commercial tools. We find that prompt programming is not well-supported: all tasks are done manually, and 16 of the 51 questions -- including a majority of the most important ones -- remain unanswered. Based on this, we outline important opportunities for prompt programming tools. 

**Abstract (ZH)**: Prompt编程：任务分类与需求分析 

---
# A Highly Clean Recipe Dataset with Ingredient States Annotation for State Probing Task 

**Title (ZH)**: 一种带有食材状态标注的 highly clean 食材数据集用于状态探查任务 

**Authors**: Mashiro Toyooka, Kiyoharu Aizawa, Yoko Yamakata  

**Link**: [PDF](https://arxiv.org/pdf/2507.17232)  

**Abstract**: Large Language Models (LLMs) are trained on a vast amount of procedural texts, but they do not directly observe real-world phenomena. In the context of cooking recipes, this poses a challenge, as intermediate states of ingredients are often omitted, making it difficult for models to track ingredient states and understand recipes accurately. In this paper, we apply state probing, a method for evaluating a language model's understanding of the world, to the domain of cooking. We propose a new task and dataset for evaluating how well LLMs can recognize intermediate ingredient states during cooking procedures. We first construct a new Japanese recipe dataset with clear and accurate annotations of ingredient state changes, collected from well-structured and controlled recipe texts. Using this dataset, we design three novel tasks to evaluate whether LLMs can track ingredient state transitions and identify ingredients present at intermediate steps. Our experiments with widely used LLMs, such as Llama3.1-70B and Qwen2.5-72B, show that learning ingredient state knowledge improves their understanding of cooking processes, achieving performance comparable to commercial LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在大量程序性文本上进行训练，但它们并未直接观察到现实世界的现象。在烹饪食谱的背景下，这提出了一个挑战，因为配料的中间状态常被省略，使得模型难以追踪配料的状态并准确理解食谱。在本文中，我们应用状态探查方法，一种评估语言模型对世界理解能力的方法，将其应用于烹饪领域。我们提出一个新的任务和数据集，用于评估LLMs在烹饪程序中识别中间配料状态的能力。我们首先构建了一个新的日语食谱数据集，其中包含了清晰准确的配料状态变化标注，这些标注来自结构良好且受控的食谱文本。使用此数据集，我们设计了三个新型任务，以评估LLMs是否能够追踪配料状态转换并在中间步骤识别出存在的配料。我们的实验表明，学习配料状态知识增强了它们对烹饪过程的理解，性能与商业LLMs相当。 

---
# The Pluralistic Moral Gap: Understanding Judgment and Value Differences between Humans and Large Language Models 

**Title (ZH)**: 多元的道德差距：理解和人类与大型语言模型之间的判断与价值差异 

**Authors**: Giuseppe Russo, Debora Nozza, Paul Röttger, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2507.17216)  

**Abstract**: People increasingly rely on Large Language Models (LLMs) for moral advice, which may influence humans' decisions. Yet, little is known about how closely LLMs align with human moral judgments. To address this, we introduce the Moral Dilemma Dataset, a benchmark of 1,618 real-world moral dilemmas paired with a distribution of human moral judgments consisting of a binary evaluation and a free-text rationale. We treat this problem as a pluralistic distributional alignment task, comparing the distributions of LLM and human judgments across dilemmas. We find that models reproduce human judgments only under high consensus; alignment deteriorates sharply when human disagreement increases. In parallel, using a 60-value taxonomy built from 3,783 value expressions extracted from rationales, we show that LLMs rely on a narrower set of moral values than humans. These findings reveal a pluralistic moral gap: a mismatch in both the distribution and diversity of values expressed. To close this gap, we introduce Dynamic Moral Profiling (DMP), a Dirichlet-based sampling method that conditions model outputs on human-derived value profiles. DMP improves alignment by 64.3% and enhances value diversity, offering a step toward more pluralistic and human-aligned moral guidance from LLMs. 

**Abstract (ZH)**: 人们越来越依赖大型语言模型（LLMs）获取道德建议，这可能影响人类的决策。然而，我们对LLMs与人类道德判断的一致性知之甚少。为此，我们引入了道德困境数据集，这是一个包含1,618个现实世界道德困境及人类道德判断分布（包括二元评估和自由文本理由）的基准。我们将其视为多元分布对齐任务，比较了道德困境中LLMs和人类判断的分布。我们发现，模型仅在高度一致性下才复制人类判断；当人类分歧增加时，对齐急剧恶化。此外，我们使用一个由3,783个从理由中提取的价值表达构建的60值分类法，表明LLMs依赖的价值范畴比人类窄。这些发现揭示了一个多元道德差距：在价值表达的分布和多样性上存在不匹配。为缩小这一差距，我们提出了动态道德画像（DMP）方法，这是一种基于狄利克雷分布的采样方法，根据人类衍生的价值概貌调整模型输出。DMP将对齐提高了64.3%，并增强了价值多样性，为提供更加多元和人类对齐的道德指导迈出了一步。 

---
# SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs 

**Title (ZH)**: SKA-Bench: 一种细粒度基准测试，用于评估LLMs的结构化知识理解能力 

**Authors**: Zhiqiang Liu, Enpei Niu, Yin Hua, Mengshu Sun, Lei Liang, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17178)  

**Abstract**: Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at this https URL. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在理解结构化知识（SK）如知识图谱（KG）和表格方面取得了显著进展，现有的SK理解评估缺乏严谨性（即缺乏对特定能力的评估）且主要关注单一类型的SK。因此，我们旨在提出一个更加全面和严谨的结构化知识理解基准，以诊断LLMs的不足之处。在本文中，我们介绍了SKA-Bench，这是一种结构化知识增强问答基准，涵盖了四种广泛使用的结构化知识形式：知识图谱（KG）、表格（Table）、知识图谱+文本（KG+Text）和表格+文本（Table+Text）。我们采用三阶段管道构建SKA-Bench实例，包括一个问题、一个答案、正面的知识单元和噪声的知识单元。为细粒度地评估LLMs的SK理解能力，我们将实例扩展为四项基本能力测试平台：噪声鲁棒性、顺序无关性、信息整合和负事实拒绝。在8个代表性LLM上的实证评估，包括先进的DeepSeek-R1，表明现有的LLM在理解结构化知识方面仍面临重大挑战，其性能受到噪声量、知识单元顺序和幻觉现象等因素的影响。我们的数据集和代码可在以下链接获取。 

---
# Resilient Multi-Agent Negotiation for Medical Supply Chains:Integrating LLMs and Blockchain for Transparent Coordination 

**Title (ZH)**: 具备弹性的多agent谈判机制：结合LLM和区块链实现透明协调 

**Authors**: Mariam ALMutairi, Hyungmin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.17134)  

**Abstract**: Global health emergencies, such as the COVID-19 pandemic, have exposed critical weaknesses in traditional medical supply chains, including inefficiencies in resource allocation, lack of transparency, and poor adaptability to dynamic disruptions. This paper presents a novel hybrid framework that integrates blockchain technology with a decentralized, large language model (LLM) powered multi-agent negotiation system to enhance the resilience and accountability of medical supply chains during crises. In this system, autonomous agents-representing manufacturers, distributors, and healthcare institutions-engage in structured, context-aware negotiation and decision-making processes facilitated by LLMs, enabling rapid and ethical allocation of scarce medical resources. The off-chain agent layer supports adaptive reasoning and local decision-making, while the on-chain blockchain layer ensures immutable, transparent, and auditable enforcement of decisions via smart contracts. The framework also incorporates a formal cross-layer communication protocol to bridge decentralized negotiation with institutional enforcement. A simulation environment emulating pandemic scenarios evaluates the system's performance, demonstrating improvements in negotiation efficiency, fairness of allocation, supply chain responsiveness, and auditability. This research contributes an innovative approach that synergizes blockchain trust guarantees with the adaptive intelligence of LLM-driven agents, providing a robust and scalable solution for critical supply chain coordination under uncertainty. 

**Abstract (ZH)**: 全球健康紧急事件，如COVID-19疫情，揭示了传统医疗供应链中的关键弱点，包括资源配置效率低下、透明度不足和对动态中断的适应能力差。本文提出了一种结合区块链技术和去中心化大型语言模型（LLM）驱动的多智能体谈判系统的新型集成框架，以增强危机期间医疗供应链的韧性和问责性。在此系统中，代表制造商、分销商和医疗机构的自主智能体在LLM支持下的结构化、情境感知谈判和决策过程中进行交互，从而实现稀缺医疗资源的快速和道德分配。脱链智能体层支持适应性推理和本地决策，而区块链层则通过智能合约确保决策的不变、透明和可审计执行。该框架还整合了一套形式化跨层通信协议，以弥合分散谈判与机构执行之间的差距。模拟环境模仿疫情场景评估系统性能，展示了谈判效率、分配公平性、供应链响应性和可审计性的改善。该研究贡献了一种创新方法，将区块链信任保证与LLM驱动智能体的适应性智能相结合，提供了一种在不确定性条件下实现关键供应链协调的稳健且可扩展的解决方案。 

---
# Enabling Self-Improving Agents to Learn at Test Time With Human-In-The-Loop Guidance 

**Title (ZH)**: 具有人类在环指导的自提升代理在测试时学习-enable 

**Authors**: Yufei He, Ruoyu Li, Alex Chen, Yue Liu, Yulin Chen, Yuan Sui, Cheng Chen, Yi Zhu, Luca Luo, Frank Yang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2507.17131)  

**Abstract**: Large language model (LLM) agents often struggle in environments where rules and required domain knowledge frequently change, such as regulatory compliance and user risk screening. Current approaches, like offline fine-tuning and standard prompting, are insufficient because they cannot effectively adapt to new knowledge during actual operation. To address this limitation, we propose the Adaptive Reflective Interactive Agent (ARIA), an LLM agent framework designed specifically to continuously learn updated domain knowledge at test time. ARIA assesses its own uncertainty through structured self-dialogue, proactively identifying knowledge gaps and requesting targeted explanations or corrections from human experts. It then systematically updates an internal, timestamped knowledge repository with provided human guidance, detecting and resolving conflicting or outdated knowledge through comparisons and clarification queries. We evaluate ARIA on the realistic customer due diligence name screening task on TikTok Pay, alongside publicly available dynamic knowledge tasks. Results demonstrate significant improvements in adaptability and accuracy compared to baselines using standard offline fine-tuning and existing self-improving agents. ARIA is deployed within TikTok Pay serving over 150 million monthly active users, confirming its practicality and effectiveness for operational use in rapidly evolving environments. 

**Abstract (ZH)**: 大型语言模型代理在规则和所需领域知识频繁变化的环境中往往表现不佳，例如合规性和用户风险筛查。当前的方法，如离线微调和标准提示，不足以在实际运行中有效适应新知识。为解决这一局限，我们提出了一种自适应反思交互代理（ARIA），这是一种专门设计用于在测试时持续学习更新领域知识的大型语言模型代理框架。ARIA 通过结构化的自我对话评估自身的不确定性，主动识别知识缺口，并向人类专家请求针对性的解释或更正。然后，它系统地根据提供的用户指导更新内部的时间戳知识库，通过比较和澄清查询检测并解决冲突或过时的知识。我们在抖音支付的真实客户尽职调查姓名筛查任务以及公开可用的动态知识任务上对ARIA 进行评估。结果表明，ARIA 在适应性和准确性方面明显优于使用标准离线微调和现有自我改进代理的基线。ARIA 已部署于抖音支付，服务于超过1.5亿月活跃用户，证实了其在快速变化环境中操作使用中的实用性和有效性。 

---
# BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving 

**Title (ZH)**: BucketServe：基于桶的动态批量处理用于智能高效的LLM推理服务 

**Authors**: Wanyi Zheng, Minxian Xu, Shengye Song, Kejiang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.17120)  

**Abstract**: Large language models (LLMs) have become increasingly popular in various areas, traditional business gradually shifting from rule-based systems to LLM-based solutions. However, the inference of LLMs is resource-intensive or latency-sensitive, posing significant challenges for serving systems. Existing LLM serving systems often use static or continuous batching strategies, which can lead to inefficient GPU memory utilization and increased latency, especially under heterogeneous workloads. These methods may also struggle to adapt to dynamic workload fluctuations, resulting in suboptimal throughput and potential service level objective (SLO) violations. In this paper, we introduce BucketServe, a bucket-based dynamic batching framework designed to optimize LLM inference performance. By grouping requests into size-homogeneous buckets based on sequence length, BucketServe minimizes padding overhead and optimizes GPU memory usage through real-time batch size adjustments preventing out-of-memory (OOM) errors. It introduces adaptive bucket splitting/merging and priority-aware scheduling to mitigate resource fragmentation and ensure SLO compliance. Experiment shows that BucketServe significantly outperforms UELLM in throughput, achieving up to 3.58x improvement. It can also handle 1.93x more request load under the SLO attainment of 80% compared with DistServe and demonstrates 1.975x higher system load capacity compared to the UELLM. 

**Abstract (ZH)**: 基于桶的动态批处理框架BucketServe优化大型语言模型推理性能 

---
# Reinforcement Learning Fine-Tunes a Sparse Subnetwork in Large Language Models 

**Title (ZH)**: 增强学习fine-tune大规模语言模型中的稀疏子网络 

**Authors**: Andrii Balashov  

**Link**: [PDF](https://arxiv.org/pdf/2507.17107)  

**Abstract**: Reinforcement learning (RL) is a key post-pretraining step for aligning large language models (LLMs) with complex tasks and human preferences. While it is often assumed that RL fine-tuning requires updating most of a model's parameters, we challenge this assumption with a surprising finding: RL fine-tuning consistently modifies only a small subnetwork (typically 5-30% of weights), leaving most parameters unchanged. We call this phenomenon RL-induced parameter update sparsity. It arises naturally, without any sparsity constraints or parameter-efficient tuning, and appears across multiple RL algorithms (e.g., PPO, DPO, SimPO, PRIME) and model families (e.g., OpenAI, Meta, and open-source LLMs). Moreover, the subnetworks updated by RL show substantial overlap across different seeds, datasets, and algorithms-far exceeding chance-suggesting a partially transferable structure in the pretrained model. We show that fine-tuning only this sparse subnetwork recovers full model performance and yields parameters nearly identical to the fully fine-tuned model. Our analysis suggests this sparsity emerges because RL operates near the model's original distribution, requiring only targeted changes. KL penalties, gradient clipping, and on-policy dynamics have limited effect on the sparsity pattern. These findings shed new light on how RL adapts models: not by shifting all weights, but by focusing training on a small, consistently updated subnetwork. This insight enables more efficient RL methods and reframes sparsity through the lens of the lottery ticket hypothesis. 

**Abstract (ZH)**: 强化学习（RL）是使大规模语言模型（LLMs）与复杂任务和人类偏好对齐的关键后微调步骤。尽管通常假设RL微调需要更新模型的大部分参数，但我们通过一个出人意料的发现挑战了这一假设：RL微调一致地仅修改一个小的子网络（通常为权重的5-30%），而大多数参数保持不变。我们称这一现象为RL诱导的参数更新稀疏性。这种现象自然产生，无需任何稀疏约束或参数高效微调。它在多种RL算法（例如PPO、DPO、SimPO、PRIME）和模型系列（例如OpenAI、Meta和开源LLMs）中普遍存在。此外，由RL更新的子网络在不同的随机种子、数据集和算法中显示出显著的重叠，远超随机水平，表明预训练模型中存在部分可转移的结构。我们证明，仅微调这一稀疏子网络即可恢复完整模型性能，并产生与完全微调模型几乎相同的参数。我们的分析表明，这种稀疏性的出现是因为RL操作在模型的原始分布附近，只需要进行针对性的调整。KL惩罚项、梯度裁剪和经验策略的动态对稀疏模式的影响有限。这些发现为如何RL适应模型提供了新的见解：并非通过移动所有权重，而是专注于一个小型且一致更新的子网络。这一洞察有助于开发更高效的RL方法，并通过彩票票假说的观点重定义稀疏性。 

---
# Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems 

**Title (ZH)**: 并行性遇上自适应性：多代理大语言模型系统中的可扩展文档理解 

**Authors**: Chengxuan Xia, Qianye Wu, Sixuan Tian, Yilun Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17061)  

**Abstract**: Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems. 

**Abstract (ZH)**: 大型语言模型代理在协作任务完成中的适应性协调框架 

---
# Causal Graph Fuzzy LLMs: A First Introduction and Applications in Time Series Forecasting 

**Title (ZH)**: 因果图模糊大语言模型：初步介绍及其在时间序列预测中的应用 

**Authors**: Omid Orang, Patricia O. Lucas, Gabriel I. F. Paiva, Petronio C. L. Silva, Felipe Augusto Rocha da Silva, Adriano Alonso Veloso, Frederico Gadelha Guimaraes  

**Link**: [PDF](https://arxiv.org/pdf/2507.17016)  

**Abstract**: In recent years, the application of Large Language Models (LLMs) to time series forecasting (TSF) has garnered significant attention among researchers. This study presents a new frame of LLMs named CGF-LLM using GPT-2 combined with fuzzy time series (FTS) and causal graph to predict multivariate time series, marking the first such architecture in the literature. The key objective is to convert numerical time series into interpretable forms through the parallel application of fuzzification and causal analysis, enabling both semantic understanding and structural insight as input for the pretrained GPT-2 model. The resulting textual representation offers a more interpretable view of the complex dynamics underlying the original time series. The reported results confirm the effectiveness of our proposed LLM-based time series forecasting model, as demonstrated across four different multivariate time series datasets. This initiative paves promising future directions in the domain of TSF using LLMs based on FTS. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在时间序列预测（TSF）中的应用引起了研究者的广泛关注。本文提出了一种新的LLM框架CGF-LLM，该框架结合了GPT-2、模糊时间序列（FTS）和因果图，用于预测多变量时间序列，这是文献中首个此类架构。关键目标是通过并行应用模糊化和因果分析，将数值时间序列转换为可解释的形式，从而为预训练的GPT-2模型提供兼具语义理解和结构洞察的输入。生成的文本表示提供了对原始时间序列复杂动力学的更可解释的视角。报告的结果证实了我们提出的基于LLM的时间序列预测模型的有效性，在四个不同的多变量时间序列数据集上进行了验证。这一举措为基于FTS的LLMs在TSF领域的未来发展铺平了道路。 

---
# Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge? 

**Title (ZH)**: 外部验证工具能否提高LLM作为法官时的标注质量？ 

**Authors**: Arduin Findeis, Floris Weers, Guoli Yin, Ke Ye, Ruoming Pang, Tom Gunter  

**Link**: [PDF](https://arxiv.org/pdf/2507.17015)  

**Abstract**: Pairwise preferences over model responses are widely collected to evaluate and provide feedback to large language models (LLMs). Given two alternative model responses to the same input, a human or AI annotator selects the "better" response. This approach can provide feedback for domains where other hard-coded metrics are difficult to obtain (e.g., chat response quality), thereby helping model evaluation or training. However, for some domains high-quality pairwise comparisons can be tricky to obtain - from AI and humans. For example, for responses with many factual statements, annotators may disproportionately weigh writing quality rather than underlying facts. In this work, we explore augmenting standard AI annotator systems with additional tools to improve performance on three challenging response domains: long-form factual, math and code tasks. We propose a tool-using agentic system to provide higher quality feedback on these domains. Our system uses web-search and code execution to ground itself based on external validation, independent of the LLM's internal knowledge and biases. We provide extensive experimental results evaluating our method across the three targeted response domains as well as general annotation tasks, using RewardBench (incl. AlpacaEval and LLMBar), RewardMath, as well as three new datasets for domains with saturated pre-existing datasets. Our results indicate that external tools can indeed improve performance in many, but not all, cases. More generally, our experiments highlight the sensitivity of performance to simple parameters (e.g., prompt) and the need for improved (non-saturated) annotator benchmarks. We share our code at this https URL. 

**Abstract (ZH)**: 基于模型响应的成对偏好收集广泛用于评估和反馈大型语言模型（LLMs）。 

---
# Leveraging Synthetic Data for Question Answering with Multilingual LLMs in the Agricultural Domain 

**Title (ZH)**: 利用合成数据提升多语言大语言模型在农业领域的问答能力 

**Authors**: Rishemjit Kaur, Arshdeep Singh Bhankhar, Surangika Ranathunga, Jashanpreet Singh Salh, Sudhir Rajput, Vidhi, Kashish Mahendra, Bhavika Berwal, Ritesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.16974)  

**Abstract**: Enabling farmers to access accurate agriculture-related information in their native languages in a timely manner is crucial for the success of the agriculture field. Although large language models (LLMs) can be used to implement Question Answering (QA) systems, simply using publicly available general-purpose LLMs in agriculture typically offer generic advisories, lacking precision in local and multilingual contexts due to insufficient domain-specific training and scarcity of high-quality, region-specific datasets. Our study addresses these limitations by generating multilingual synthetic agricultural datasets (English, Hindi, Punjabi) from agriculture-specific documents and fine-tuning language-specific LLMs. Our evaluation on curated multilingual datasets demonstrates significant improvements in factual accuracy, relevance, and agricultural consensus for the fine-tuned models compared to their baseline counterparts. These results highlight the efficacy of synthetic data-driven, language-specific fine-tuning as an effective strategy to improve the performance of LLMs in agriculture, especially in multilingual and low-resource settings. By enabling more accurate and localized agricultural advisory services, this study provides a meaningful step toward bridging the knowledge gap in AI-driven agricultural solutions for diverse linguistic communities. 

**Abstract (ZH)**: 使农民能够及时获取其本土语言的准确农业相关信息对于农业领域的成功至关重要。尽管大型语言模型可以用于实施问答系统，但通常使用通用大型语言模型在农业领域仅能提供泛化的建议，由于缺乏特定领域训练和高质量地区特定数据集的稀缺性，这在本地和多语言背景下缺乏精准性。我们的研究通过从农业特定文档生成多语言合成农业数据集（英语、印地语、旁遮普语）并微调语言特定的大型语言模型来解决这些限制。在编纂的多语言数据集上的评估表明，微调模型在事实准确性、相关性和农业共识方面显著优于基线模型。这些结果强调了合成数据驱动的语言特定微调作为提高大型语言模型在农业中性能的有效策略的有效性，特别是在多语言和低资源环境中。通过提供更准确和本地化的农业咨询服务，本研究为弥合AI驱动农业解决方案中的知识差距提供了有意义的一步，特别是对于多元语言社区。 

---
# SiLQ: Simple Large Language Model Quantization-Aware Training 

**Title (ZH)**: SiLQ: 简洁的大语言模型量化感知训练 

**Authors**: Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha  

**Link**: [PDF](https://arxiv.org/pdf/2507.16933)  

**Abstract**: Large language models can be quantized to reduce inference time latency, model size, and energy consumption, thereby delivering a better user experience at lower cost. A challenge exists to deliver quantized models with minimal loss of accuracy in reasonable time, and in particular to do so without requiring mechanisms incompatible with specialized inference accelerators. Here, we demonstrate a simple, end-to-end quantization-aware training approach that, with an increase in total model training budget of less than 0.1%, outperforms the leading published quantization methods by large margins on several modern benchmarks, with both base and instruct model variants. The approach easily generalizes across different model architectures, can be applied to activations, cache, and weights, and requires the introduction of no additional operations to the model other than the quantization itself. 

**Abstract (ZH)**: 大型语言模型可以通过量化来降低推理时间延迟、模型大小和能源消耗，从而以更低的成本提供更好的用户体验。挑战在于在合理的时间内交付准确度损失最小的量化模型，并且特别是不需要与专用推理加速器不兼容的机制。在此，我们展示了一种简单的一体化量化感知训练方法，该方法在总模型训练预算增加不到0.1%的情况下，在多个现代基准测试中，无论是基础模型还是指令模型变体，都能显著优于现有公布的量化方法。该方法容易跨不同的模型架构进行泛化，可以应用于激活、缓存和权重，并且除了量化本身外，不需要在模型中引入额外的操作。 

---
# Revisiting Pre-trained Language Models for Vulnerability Detection 

**Title (ZH)**: 重新审视预训练语言模型在漏洞检测中的应用 

**Authors**: Youpeng Li, Weiliang Qi, Xuyu Wang, Fuxun Yu, Xinda Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16887)  

**Abstract**: The rapid advancement of pre-trained language models (PLMs) has demonstrated promising results for various code-related tasks. However, their effectiveness in detecting real-world vulnerabilities remains a critical challenge. % for the security community. While existing empirical studies evaluate PLMs for vulnerability detection (VD), their inadequate consideration in data preparation, evaluation setups, and experimental settings undermines the accuracy and comprehensiveness of evaluations. This paper introduces RevisitVD, an extensive evaluation of 17 PLMs spanning smaller code-specific PLMs and large-scale PLMs using newly constructed datasets. Specifically, we compare the performance of PLMs under both fine-tuning and prompt engineering, assess their effectiveness and generalizability across various training and testing settings, and analyze their robustness against code normalization, abstraction, and semantic-preserving transformations.
Our findings reveal that, for VD tasks, PLMs incorporating pre-training tasks designed to capture the syntactic and semantic patterns of code outperform both general-purpose PLMs and those solely pre-trained or fine-tuned on large code corpora. However, these models face notable challenges in real-world scenarios, such as difficulties in detecting vulnerabilities with complex dependencies, handling perturbations introduced by code normalization and abstraction, and identifying semantic-preserving vulnerable code transformations. Also, the truncation caused by the limited context windows of PLMs can lead to a non-negligible amount of labeling errors. This study underscores the importance of thorough evaluations of model performance in practical scenarios and outlines future directions to help enhance the effectiveness of PLMs for realistic VD applications. 

**Abstract (ZH)**: 预训练语言模型在漏洞检测任务中的全面评估：挑战与方向 

---
# HIPPO-Video: Simulating Watch Histories with Large Language Models for Personalized Video Highlighting 

**Title (ZH)**: HIPPO-视频：使用大型语言模型模拟观看历史以便个性化视频摘要生成 

**Authors**: Jeongeun Lee, Youngjae Yu, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.16873)  

**Abstract**: The exponential growth of video content has made personalized video highlighting an essential task, as user preferences are highly variable and complex. Existing video datasets, however, often lack personalization, relying on isolated videos or simple text queries that fail to capture the intricacies of user behavior. In this work, we introduce HIPPO-Video, a novel dataset for personalized video highlighting, created using an LLM-based user simulator to generate realistic watch histories reflecting diverse user preferences. The dataset includes 2,040 (watch history, saliency score) pairs, covering 20,400 videos across 170 semantic categories. To validate our dataset, we propose HiPHer, a method that leverages these personalized watch histories to predict preference-conditioned segment-wise saliency scores. Through extensive experiments, we demonstrate that our method outperforms existing generic and query-based approaches, showcasing its potential for highly user-centric video highlighting in real-world scenarios. 

**Abstract (ZH)**: 个性化视频摘要的指数增长使得个性化视频亮点提取成为一项必不可少的任务，因为用户偏好高度多样且复杂。现有视频数据集通常缺乏个性化，依赖于孤立的视频或简单的文本查询，未能捕捉用户行为的复杂性。在本工作中，我们引入了HIPPO-Video，这是一个采用基于LLM的用户模拟器生成反映多样化用户偏好的真实观看历史的新数据集。该数据集包含2040个（观看历史，显著性分数）对，覆盖了170个语义类别中的20400个视频。为了验证我们的数据集，我们提出了一种名为HiPHer的方法，该方法利用这些个性化的观看历史来预测条件偏好下的段落显著性分数。通过广泛的实验，我们证明了我们的方法优于现有的通用和查询驱动的方法，展示了其在实际场景中实现高度用户中心的视频摘要的潜力。 

---
# SynthCTI: LLM-Driven Synthetic CTI Generation to enhance MITRE Technique Mapping 

**Title (ZH)**: SynthCTI: LLM驱动的合成CTI生成以增强MITRE技术映射 

**Authors**: Álvaro Ruiz-Ródenas, Jaime Pujante Sáez, Daniel García-Algora, Mario Rodríguez Béjar, Jorge Blasco, José Luis Hernández-Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2507.16852)  

**Abstract**: Cyber Threat Intelligence (CTI) mining involves extracting structured insights from unstructured threat data, enabling organizations to understand and respond to evolving adversarial behavior. A key task in CTI mining is mapping threat descriptions to MITRE ATT\&CK techniques. However, this process is often performed manually, requiring expert knowledge and substantial effort. Automated approaches face two major challenges: the scarcity of high-quality labeled CTI data and class imbalance, where many techniques have very few examples. While domain-specific Large Language Models (LLMs) such as SecureBERT have shown improved performance, most recent work focuses on model architecture rather than addressing the data limitations. In this work, we present SynthCTI, a data augmentation framework designed to generate high-quality synthetic CTI sentences for underrepresented MITRE ATT\&CK techniques. Our method uses a clustering-based strategy to extract semantic context from training data and guide an LLM in producing synthetic CTI sentences that are lexically diverse and semantically faithful. We evaluate SynthCTI on two publicly available CTI datasets, CTI-to-MITRE and TRAM, using LLMs with different capacity. Incorporating synthetic data leads to consistent macro-F1 improvements: for example, ALBERT improves from 0.35 to 0.52 (a relative gain of 48.6\%), and SecureBERT reaches 0.6558 (up from 0.4412). Notably, smaller models augmented with SynthCTI outperform larger models trained without augmentation, demonstrating the value of data generation methods for building efficient and effective CTI classification systems. 

**Abstract (ZH)**: 基于合成数据的Cyber威胁情报（CTI）矿化解决策ércy威胁情报（CTI）挖掘涉及从非结构化威胁数据中提取结构化洞见，使组织能够理解并响应不断演变的对手行为。CTI挖掘中的一个关键任务是将威胁描述映射到MITRE ATT&CK技术。然而，这一过程通常需要手动完成，依赖专家知识和大量努力。自动化方法面临两个主要挑战：高质量标注CTI数据的稀缺和类别不平衡，其中许多技术具有非常少的示例。尽管领域特定的大语言模型（LLMs）如SecureBERT已经显示出改进性能，但近期大部分工作侧重于模型架构而不是解决数据限制问题。在本项工作中，我们提出了SynthCTI，一种数据增强框架，旨在生成高质量的合成CTI句子以填充MITRE ATT&CK下代表现不足的技术。我们的方法使用基于聚类的策略从训练数据中提取语义上下文，并指导LLM产生词汇上多样且语义上忠实的合成CTI句子。我们在两个公开可用的CTI数据集CTI-to-MITRE和TRAM上使用不同容量的LLM评估SynthCTI。包含合成数据导致宏观经济F1提高：例如，ALBERT从0.35提高到0.52（相对增加48.6%），SecureBERT达到0.6558（从0.4412提高）。值得注意的是，使用SynthCTI增强的较小模型优于未增强的大模型，这表明数据生成方法对于构建高效有效的CTI分类系统具有价值。 

---
# A Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval-Augmented Generation in Large Language Models 

**Title (ZH)**: 一种基于查询的多路径知识图融合方法，用于增强大型语言模型中的检索增强生成 

**Authors**: Qikai Wei, Huansheng Ning, Chunlong Han, Jianguo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.16826)  

**Abstract**: Retrieval Augmented Generation (RAG) has gradually emerged as a promising paradigm for enhancing the accuracy and factual consistency of content generated by large language models (LLMs). However, existing RAG studies primarily focus on retrieving isolated segments using similarity-based matching methods, while overlooking the intrinsic connections between them. This limitation hampers performance in RAG tasks. To address this, we propose QMKGF, a Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval Augmented Generation. First, we design prompt templates and employ general-purpose LLMs to extract entities and relations, thereby generating a knowledge graph (KG) efficiently. Based on the constructed KG, we introduce a multi-path subgraph construction strategy that incorporates one-hop relations, multi-hop relations, and importance-based relations, aiming to improve the semantic relevance between the retrieved documents and the user query. Subsequently, we designed a query-aware attention reward model that scores subgraph triples based on their semantic relevance to the query. Then, we select the highest score subgraph and enrich subgraph with additional triples from other subgraphs that are highly semantically relevant to the query. Finally, the entities, relations, and triples within the updated subgraph are utilised to expand the original query, thereby enhancing its semantic representation and improving the quality of LLMs' generation. We evaluate QMKGF on the SQuAD, IIRC, Culture, HotpotQA, and MuSiQue datasets. On the HotpotQA dataset, our method achieves a ROUGE-1 score of 64.98\%, surpassing the BGE-Rerank approach by 9.72 percentage points (from 55.26\% to 64.98\%). Experimental results demonstrate the effectiveness and superiority of the QMKGF approach. 

**Abstract (ZH)**: 基于查询aware多路径知识图融合的增强检索增强生成（QMKGF）方法 

---
