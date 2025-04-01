# RIG: Synergizing Reasoning and Imagination in End-to-End Generalist Policy 

**Title (ZH)**: RIG: 结合推理与想象的端到端通用策略 

**Authors**: Zhonghan Zhao, Wenwei Zhang, Haian Huang, Kuikun Liu, Jianfei Gao, Gaoang Wang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.24388)  

**Abstract**: Reasoning before action and imagining potential outcomes (i.e., world models) are essential for embodied agents operating in complex open-world environments. Yet, prior work either incorporates only one of these abilities in an end-to-end agent or integrates multiple specialized models into an agent system, limiting the learning efficiency and generalization of the policy. Thus, this paper makes the first attempt to synergize Reasoning and Imagination in an end-to-end Generalist policy, termed RIG. To train RIG in an end-to-end manner, we construct a data pipeline that progressively integrates and enriches the content of imagination and reasoning in the trajectories collected from existing agents. The joint learning of reasoning and next image generation explicitly models the inherent correlation between reasoning, action, and dynamics of environments, and thus exhibits more than $17\times$ sample efficiency improvements and generalization in comparison with previous works. During inference, RIG first reasons about the next action, produces potential action, and then predicts the action outcomes, which offers the agent a chance to review and self-correct based on the imagination before taking real actions. Experimental results show that the synergy of reasoning and imagination not only improves the robustness, generalization, and interoperability of generalist policy but also enables test-time scaling to enhance overall performance. 

**Abstract (ZH)**: 推理在先并想象潜在结果（即世界模型）对于在复杂开放环境中的 bodily代理至关重要。然而，先前的工作要么仅在一个端到端代理中结合了其中一种能力，要么将多种专门的模型集成到代理系统中，限制了策略的学习效率和泛化能力。因此，本文首次尝试在端到端的一般主义策略中结合推理和想象，称为RIG。为了以端到端的方式训练RIG，我们构建了一条数据管道，逐步整合和丰富现有代理收集的轨迹中的想象和推理内容。推理和下一个图像生成的联合学习明确地建模了推理、动作和环境动力学之间的内在关联，从而在样本效率和泛化方面比先前的工作提高了超过17倍。在推理过程中，RIG首先进行推理以确定下一个动作，生成潜在动作，并预测行动结果，这使代理有机会在采取实际行动之前根据想象进行回顾和自我纠正。实验结果表明，推理和想象的结合不仅提高了通用策略的鲁棒性、泛化能力和互操作性，还通过测试时的扩展提高了整体性能。 

---
# ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning 

**Title (ZH)**: ACPBench 困难版：关于行动、变化与规划的无约束推理 

**Authors**: Harsha Kokel, Michael Katz, Kavitha Srinivas, Shirin Sohrabi  

**Link**: [PDF](https://arxiv.org/pdf/2503.24378)  

**Abstract**: The ACPBench dataset provides atomic reasoning tasks required for efficient planning. The dataset is aimed at distilling the complex plan generation task into separate atomic reasoning tasks in their easiest possible form, boolean or multiple-choice questions, where the model has to choose the right answer from the provided options. While the aim of ACPBench is to test the simplest form of reasoning about action and change, when tasked with planning, a model does not typically have options to choose from and thus the reasoning required for planning dictates an open-ended, generative form for these tasks. To that end, we introduce ACPBench Hard, a generative version of ACPBench, with open-ended questions which the model needs to answer. Models that perform well on these tasks could in principle be integrated into a planner or be used directly as a policy. We discuss the complexity of these tasks as well as the complexity of validating the correctness of their answers and present validation algorithms for each task. Equipped with these validators, we test the performance of a variety of models on our tasks and find that for most of these tasks the performance of even the largest models is still subpar. Our experiments show that no model outperforms another in these tasks and with a few exceptions all tested language models score below 65%, indicating that even the current frontier language models have a long way to go before they can reliably reason about planning. In fact, even the so-called reasoning models struggle with solving these reasoning tasks. ACPBench Hard collection is available at the following link: this https URL 

**Abstract (ZH)**: ACPBench数据集提供了用于高效规划的原子推理任务。ACPBench Hard是ACPBench的生成版本，包含开放性问题，模型需要回答这些问题。我们讨论了这些任务的复杂性以及验证其答案正确性的复杂性，并为每个任务提出了验证算法。配备这些验证器后，我们测试了多种模型在这些任务上的性能，并发现大多数任务中，即使是最大的模型性能仍然不足。我们的实验表明，在这些任务中没有模型能够胜出，测试的所有语言模型得分均低于65%，表明当前的语言模型在可靠进行规划推理方面还有很长的路要走。实际上，所谓的推理模型在解决这些推理任务时也面临困难。ACPBench Hard数据集可在以下链接获取：this https URL。 

---
# Contextual Preference Collaborative Measure Framework Based on Belief System 

**Title (ZH)**: 基于信念系统的情境偏好评价协作度量框架 

**Authors**: Hang Yu, Wei Wei, Zheng Tan, Jing-lei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24328)  

**Abstract**: To reduce the human intervention in the preference measure process,this article proposes a preference collaborative measure framework based on an updated belief system,which is also capable of improving the accuracy and efficiency of preferen-ce measure this http URL,the distance of rules and the average internal distance of rulesets are proposed for specifying the relationship between the this http URL discovering the most representative preferences that are common in all users,namely common preference,a algorithm based on average internal distance of ruleset,PRA algorithm,is proposed,which aims to finish the discoveryprocess with minimum information loss this http URL,the concept of Common belief is proposed to update the belief system,and the common preferences are the evidences of updated belief this http URL,under the belief system,the proposed belief degree and deviation degree are used to determine whether a rule confirms the belief system or not and classify the preference rules into two kinds(generalized or personalized),and eventually filters out Top-K interesting rules relying on belief degree and deviation this http URL on above,a scalable interestingness calculation framework that can apply various formulas is proposed for accurately calculating interestingness in different this http URL last,IMCos algorithm and IMCov algorithm are proposed as exemplars to verify the accuracy and efficiency of the framework by using weighted cosine similarity and correlation coefficients as belief this http URL experiments,the proposed algorithms are compared to two state-of-the-art algorithms and the results show that IMCos and IMCov outperform than the other two in most aspects. 

**Abstract (ZH)**: 基于更新信念系统的偏好协作衡量框架：减少人为干预并提高偏好衡量的准确性和效率 

---
# PAARS: Persona Aligned Agentic Retail Shoppers 

**Title (ZH)**: PAARS：个性导向的行动者零售消费者 

**Authors**: Saab Mansour, Leonardo Perelli, Lorenzo Mainetti, George Davidson, Stefano D'Amato  

**Link**: [PDF](https://arxiv.org/pdf/2503.24228)  

**Abstract**: In e-commerce, behavioral data is collected for decision making which can be costly and slow. Simulation with LLM powered agents is emerging as a promising alternative for representing human population behavior. However, LLMs are known to exhibit certain biases, such as brand bias, review rating bias and limited representation of certain groups in the population, hence they need to be carefully benchmarked and aligned to user behavior. Ultimately, our goal is to synthesise an agent population and verify that it collectively approximates a real sample of humans. To this end, we propose a framework that: (i) creates synthetic shopping agents by automatically mining personas from anonymised historical shopping data, (ii) equips agents with retail-specific tools to synthesise shopping sessions and (iii) introduces a novel alignment suite measuring distributional differences between humans and shopping agents at the group (i.e. population) level rather than the traditional "individual" level. Experimental results demonstrate that using personas improves performance on the alignment suite, though a gap remains to human behaviour. We showcase an initial application of our framework for automated agentic A/B testing and compare the findings to human results. Finally, we discuss applications, limitations and challenges setting the stage for impactful future work. 

**Abstract (ZH)**: 基于LLM的代理模拟在电子商务中的行为数据合成与对齐：一个框架及其应用 

---
# All You Need is Sally-Anne: ToM in AI Strongly Supported After Surpassing Tests for 3-Year-Olds 

**Title (ZH)**: 只需辛迪·兰恩：人工智能的理论思维在超过3岁儿童测试后得到强有力支持 

**Authors**: Nitay Alon, Joseph Barnby, Reuth Mirsky, Stefan Sarkadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.24215)  

**Abstract**: Theory of Mind (ToM) is a hallmark of human cognition, allowing individuals to reason about others' beliefs and intentions. Engineers behind recent advances in Artificial Intelligence (AI) have claimed to demonstrate comparable capabilities. This paper presents a model that surpasses traditional ToM tests designed for 3-year-old children, providing strong support for the presence of ToM in AI systems. 

**Abstract (ZH)**: 理论心智（ToM）是人类认知的 hallmark，使个体能够推理他人的信念和意图。近期人工智能（AI）进展背后的工程师们声称展示了相当的能力。本文提出了一种模型，超越了为 3 岁儿童设计的传统 ToM 测试，为 AI 系统中存在 ToM 提供了强有力的支持。 

---
# Agent-Based Simulations of Online Political Discussions: A Case Study on Elections in Germany 

**Title (ZH)**: 基于代理的德国elections在线政治讨论模拟：个案研究 

**Authors**: Abdul Sittar, Simon Münker, Fabio Sartori, Andreas Reitenbach, Achim Rettinger, Michael Mäs, Alenka Guček, Marko Grobelnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.24199)  

**Abstract**: User engagement on social media platforms is influenced by historical context, time constraints, and reward-driven interactions. This study presents an agent-based simulation approach that models user interactions, considering past conversation history, motivation, and resource constraints. Utilizing German Twitter data on political discourse, we fine-tune AI models to generate posts and replies, incorporating sentiment analysis, irony detection, and offensiveness classification. The simulation employs a myopic best-response model to govern agent behavior, accounting for decision-making based on expected rewards. Our results highlight the impact of historical context on AI-generated responses and demonstrate how engagement evolves under varying constraints. 

**Abstract (ZH)**: 社交媒体平台上的用户参与受历史背景、时间限制和奖赏驱动力交互的影响：基于代理的仿真研究——以政治 discourse中的德国Twitter数据为例，调整AI模型生成帖子和回复，结合情感分析、讽刺检测和冒犯分类，并采用短视最佳响应模型来管理代理行为，展示历史背景对AI生成响应的影响及其在不同约束下的参与演变。 

---
# Grounding Agent Reasoning in Image Schemas: A Neurosymbolic Approach to Embodied Cognition 

**Title (ZH)**: 基于图像模式的地基代理推理：一种神经符号方法的体表认知研究 

**Authors**: François Olivier, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2503.24110)  

**Abstract**: Despite advances in embodied AI, agent reasoning systems still struggle to capture the fundamental conceptual structures that humans naturally use to understand and interact with their environment. To address this, we propose a novel framework that bridges embodied cognition theory and agent systems by leveraging a formal characterization of image schemas, which are defined as recurring patterns of sensorimotor experience that structure human cognition. By customizing LLMs to translate natural language descriptions into formal representations based on these sensorimotor patterns, we will be able to create a neurosymbolic system that grounds the agent's understanding in fundamental conceptual structures. We argue that such an approach enhances both efficiency and interpretability while enabling more intuitive human-agent interactions through shared embodied understanding. 

**Abstract (ZH)**: 尽管在具身AI方面取得了进展，代理推理系统仍难以捕捉人类自然用于理解及其与环境互动的基本概念结构。为解决这一问题，我们提出了一种新型框架，该框架通过利用形象图式的形式化特征，将具身认知理论与代理系统相结合。形象图式被定义为传感器运动体验中的重复模式，这些模式构成了人类认知的基础。通过定制LLMs将自然语言描述转化为基于这些传感器运动模式的形式化表示，我们可以创建一个神经符号系统，使代理的理解扎根于基本概念结构。我们认为，这种做法既能提高效率和可解释性，又能通过共享的具身理解实现更直观的人机交互。 

---
# Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents 

**Title (ZH)**: 面向科学智能：基于LLM的科学代理综述 

**Authors**: Shuo Ren, Pu Jian, Zhenjiang Ren, Chunlin Leng, Can Xie, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.24047)  

**Abstract**: As scientific research becomes increasingly complex, innovative tools are needed to manage vast data, facilitate interdisciplinary collaboration, and accelerate discovery. Large language models (LLMs) are now evolving into LLM-based scientific agents that automate critical tasks, ranging from hypothesis generation and experiment design to data analysis and simulation. Unlike general-purpose LLMs, these specialized agents integrate domain-specific knowledge, advanced tool sets, and robust validation mechanisms, enabling them to handle complex data types, ensure reproducibility, and drive scientific breakthroughs. This survey provides a focused review of the architectures, design, benchmarks, applications, and ethical considerations surrounding LLM-based scientific agents. We highlight why they differ from general agents and the ways in which they advance research across various scientific fields. By examining their development and challenges, this survey offers a comprehensive roadmap for researchers and practitioners to harness these agents for more efficient, reliable, and ethically sound scientific discovery. 

**Abstract (ZH)**: 随着科学研究的日益复杂，需要创新工具来管理大量数据、促进跨学科合作并加速发现过程。大型语言模型（LLMs）现在正在演变成基于LLM的科学代理，自动化从假设生成和实验设计到数据分析和模拟的各种关键任务。与通用型LLMs不同，这些专门的代理整合了领域特定知识、高级工具集和 robust 验证机制，使其能够处理复杂数据类型、确保可再现性并推动科学突破。本文综述了基于LLM的科学代理的架构、设计、基准测试、应用和伦理考虑。我们强调了它们与通用代理的区别，并展示了它们如何在各个科学领域推进研究。通过对它们的发展和挑战的分析，本文为研究人员和从业者提供了一条综合路线图，以便更高效、可靠且伦理地利用这些代理进行科学研究。 

---
# Pay More Attention to the Robustness of Prompt for Instruction Data Mining 

**Title (ZH)**: 更多关注指令数据挖掘中提示的鲁棒性 

**Authors**: Qiang Wang, Dawei Feng, Xu Zhang, Ao Shen, Yang Xu, Bo Ding, Huaimin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.24028)  

**Abstract**: Instruction tuning has emerged as a paramount method for tailoring the behaviors of LLMs. Recent work has unveiled the potential for LLMs to achieve high performance through fine-tuning with a limited quantity of high-quality instruction data. Building upon this approach, we further explore the impact of prompt's robustness on the selection of high-quality instruction data. This paper proposes a pioneering framework of high-quality online instruction data mining for instruction tuning, focusing on the impact of prompt's robustness on the data mining process. Our notable innovation, is to generate the adversarial instruction data by conducting the attack for the prompt of online instruction data. Then, we introduce an Adversarial Instruction-Following Difficulty metric to measure how much help the adversarial instruction data can provide to the generation of the corresponding response. Apart from it, we propose a novel Adversarial Instruction Output Embedding Consistency approach to select high-quality online instruction data. We conduct extensive experiments on two benchmark datasets to assess the performance. The experimental results serve to underscore the effectiveness of our proposed two methods. Moreover, the results underscore the critical practical significance of considering prompt's robustness. 

**Abstract (ZH)**: 基于提示鲁棒性的高质量在线指令数据挖掘框架 

---
# AI2Agent: An End-to-End Framework for Deploying AI Projects as Autonomous Agents 

**Title (ZH)**: AI2Agent：将AI项目部署为自主代理的端到端框架 

**Authors**: Jiaxiang Chen, Jingwei Shi, Lei Gan, Jiale Zhang, Qingyu Zhang, Dongqian Zhang, Xin Pang, Zhucong Li, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23948)  

**Abstract**: As AI technology advances, it is driving innovation across industries, increasing the demand for scalable AI project deployment. However, deployment remains a critical challenge due to complex environment configurations, dependency conflicts, cross-platform adaptation, and debugging difficulties, which hinder automation and adoption. This paper introduces AI2Agent, an end-to-end framework that automates AI project deployment through guideline-driven execution, self-adaptive debugging, and case \& solution accumulation. AI2Agent dynamically analyzes deployment challenges, learns from past cases, and iteratively refines its approach, significantly reducing human intervention. To evaluate its effectiveness, we conducted experiments on 30 AI deployment cases, covering TTS, text-to-image generation, image editing, and other AI applications. Results show that AI2Agent significantly reduces deployment time and improves success rates. The code and demo video are now publicly accessible. 

**Abstract (ZH)**: 随着人工智能技术的发展，它正推动各个行业的创新，增加可扩展人工智能项目部署的需求。然而，部署仍是一个关键挑战，由于复杂环境配置、依赖冲突、跨平台适应性和调试困难，这阻碍了自动化和推广应用。本文介绍了AI2Agent，这是一个端到端框架，通过指南驱动执行、自我适应调试和案例及解决方案积累来自动化人工智能项目部署。AI2Agent动态分析部署挑战，从过往案例中学习，并迭代优化其方法，显著减少人工干预。为了评估其有效性，我们在30个人工智能部署案例上进行了实验，涵盖TTS、文本-to-图像生成、图像编辑和其他人工智能应用。结果显示，AI2Agent显著减少了部署时间和提高了成功率。目前该代码和演示视频已公开。 

---
# What the F*ck Is Artificial General Intelligence? 

**Title (ZH)**: 什么是通用人工智能？ 

**Authors**: Michael Timothy Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2503.23923)  

**Abstract**: Artificial general intelligence (AGI) is an established field of research. Yet Melanie Mitchell and others have questioned if the term still has meaning. AGI has been subject to so much hype and speculation it has become something of a Rorschach test. Mitchell points out that the debate will only be settled through long term, scientific investigation. To that end here is a short, accessible and provocative overview of AGI. I compare definitions of intelligence, settling on intelligence in terms of adaptation and AGI as an artificial scientist. Taking my queue from Sutton's Bitter Lesson I describe two foundational tools used to build adaptive systems: search and approximation. I compare pros, cons, hybrids and architectures like o3, AlphaGo, AERA, NARS and Hyperon. I then discuss overall meta-approaches to making systems behave more intelligently. I divide them into scale-maxing, simp-maxing, w-maxing based on the Bitter Lesson, Ockham's and Bennett's Razors. These maximise resources, simplicity of form, and the weakness of constraints on functionality. I discuss examples including AIXI, the free energy principle and The Embiggening of language models. I conclude that though scale-maxed approximation dominates, AGI will be a fusion of tools and meta-approaches. The Embiggening was enabled by improvements in hardware. Now the bottlenecks are sample and energy efficiency. 

**Abstract (ZH)**: 人工通用智能：一个富有启发性的简要概述 

---
# DebFlow: Automating Agent Creation via Agent Debate 

**Title (ZH)**: DebFlow: 通过代理辩论自动化代理创建 

**Authors**: Jinwei Su, Yinghui Xia, Ronghua Shi, Jianhui Wang, Jianuo Huang, Yijin Wang, Tianyu Shi, Yang Jingsong, Lewei He  

**Link**: [PDF](https://arxiv.org/pdf/2503.23781)  

**Abstract**: Large language models (LLMs) have demonstrated strong potential and impressive performance in automating the generation and optimization of workflows. However, existing approaches are marked by limited reasoning capabilities, high computational demands, and significant resource requirements. To address these issues, we propose DebFlow, a framework that employs a debate mechanism to optimize workflows and integrates reflexion to improve based on previous experiences. We evaluated our method across six benchmark datasets, including HotpotQA, MATH, and ALFWorld. Our approach achieved a 3\% average performance improvement over the latest baselines, demonstrating its effectiveness in diverse problem domains. In particular, during training, our framework reduces resource consumption by 37\% compared to the state-of-the-art baselines. Additionally, we performed ablation studies. Removing the Debate component resulted in a 4\% performance drop across two benchmark datasets, significantly greater than the 2\% drop observed when the Reflection component was removed. These findings strongly demonstrate the critical role of Debate in enhancing framework performance, while also highlighting the auxiliary contribution of reflexion to overall optimization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化工作流的生成和优化方面展现了强大的潜力和令人印象深刻的性能。然而，现有方法存在推理能力有限、高计算需求和显著的资源要求等问题。为解决这些问题，我们提出了DebFlow框架，该框架采用了辩论机制来优化工作流，并结合反省机制以根据以往经验改进。我们通过对HotpotQA、MATH和ALFWorld等六个基准数据集进行评估，我们的方法在最新基线方法上实现了3%的平均性能提升，证明了其在多样化的问题领域中的有效性。特别是在训练过程中，与最先进的基线方法相比，我们的框架将资源消耗降低了37%。此外，我们还进行了消融研究。去除辩论组件导致两个基准数据集的性能下降了4%，这一下降幅度远大于去除反省组件时观察到的2%的下降幅度。这些发现强烈证明了辩论在增强框架性能中的关键作用，同时突显了反省对整体优化的辅助贡献。 

---
# MolGround: A Benchmark for Molecular Grounding 

**Title (ZH)**: MolGround: 分子接地基准数据集 

**Authors**: Jiaxin Wu, Ting Zhang, Rubing Chen, Wengyu Zhang, Chen Jason Zhang, Xiaoyong Wei, Li Qing  

**Link**: [PDF](https://arxiv.org/pdf/2503.23668)  

**Abstract**: Current molecular understanding approaches predominantly focus on the descriptive aspect of human perception, providing broad, topic-level insights. However, the referential aspect -- linking molecular concepts to specific structural components -- remains largely unexplored. To address this gap, we propose a molecular grounding benchmark designed to evaluate a model's referential abilities. We align molecular grounding with established conventions in NLP, cheminformatics, and molecular science, showcasing the potential of NLP techniques to advance molecular understanding within the AI for Science movement. Furthermore, we constructed the largest molecular understanding benchmark to date, comprising 79k QA pairs, and developed a multi-agent grounding prototype as proof of concept. This system outperforms existing models, including GPT-4o, and its grounding outputs have been integrated to enhance traditional tasks such as molecular captioning and ATC (Anatomical, Therapeutic, Chemical) classification. 

**Abstract (ZH)**: 当前分子理解方法主要聚焦于人类感知的描述方面，提供了广泛的主题级洞察。然而，参照方面——将分子概念与特定结构成分联系起来——仍然很大程度上未被探索。为解决这一问题，我们提出了一种分子 grounding 基准，旨在评估模型的参照能力。我们将分子 grounding 与 NLP、化学生物信息学和分子科学中的既定规范相结合，展示了 NLP 技术在科学人工智能运动中推动分子理解的潜力。此外，我们构建了迄今为止最大的分子理解基准，包含 79,000 个 QA 对，并开发了一个多代理 grounding 模型作为概念验证。该系统超越了现有模型，包括 GPT-4o，并将其 grounding 输出整合到传统的分子标注和 ATC (Anatomical, Therapeutic, Chemical) 分类任务中以提升性能。 

---
# GIScience in the Era of Artificial Intelligence: A Research Agenda Towards Autonomous GIS 

**Title (ZH)**: 人工智能时代的GIScience：通往自主GIS的研究议程 

**Authors**: Zhenlong Li, Huan Ning, Song Gao, Krzysztof Janowicz, Wenwen Li, Samantha T. Arundel, Chaowei Yang, Budhendra Bhaduri, Shaowen Wang, A-Xing Zhu, Mark Gahegan, Shashi Shekhar, Xinyue Ye, Grant McKenzie, Guido Cervone, Michael E. Hodgson  

**Link**: [PDF](https://arxiv.org/pdf/2503.23633)  

**Abstract**: The advent of generative AI exemplified by large language models (LLMs) opens new ways to represent and compute geographic information and transcend the process of geographic knowledge production, driving geographic information systems (GIS) towards autonomous GIS. Leveraging LLMs as the decision core, autonomous GIS can independently generate and execute geoprocessing workflows to perform spatial analysis. In this vision paper, we elaborate on the concept of autonomous GIS and present a framework that defines its five autonomous goals, five levels of autonomy, five core functions, and three operational scales. We demonstrate how autonomous GIS could perform geospatial data retrieval, spatial analysis, and map making with four proof-of-concept GIS agents. We conclude by identifying critical challenges and future research directions, including fine-tuning and self-growing decision cores, autonomous modeling, and examining the ethical and practical implications of autonomous GIS. By establishing the groundwork for a paradigm shift in GIScience, this paper envisions a future where GIS moves beyond traditional workflows to autonomously reason, derive, innovate, and advance solutions to pressing global challenges. 

**Abstract (ZH)**: 生成式AI的兴起，以大规模语言模型（LLMs）为代表，为地理信息的表示和计算开辟了新途径，超越了地理知识生产的过程，推动地理信息系统（GIS）向自主GIS转型。利用大规模语言模型作为决策核心，自主GIS可以独立生成和执行地理处理工作流，进行空间分析。在本文中，我们阐释了自主GIS的概念，并提出一个框架，定义了其五个自主目标、五个自主层级、五个核心功能以及三个运作尺度。我们通过四个概念性的GIS代理展示了自主GIS如何进行地理空间数据检索、空间分析和制图。最后，我们指出了关键挑战和未来研究方向，包括精细调整和自我成长的决策核心、自主建模以及探讨自主GIS的伦理与实际影响。通过为GIS科学确立范式转移的基础，本文构想了一个未来，GIS将超越传统的工作流程，自主推理、推导、创新并解决紧迫的全球挑战。 

---
# Intrinsically-Motivated Humans and Agents in Open-World Exploration 

**Title (ZH)**: 内在动机驱动的人与代理在开放世界探索中 

**Authors**: Aly Lidayan, Yuqing Du, Eliza Kosoy, Maria Rufova, Pieter Abbeel, Alison Gopnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.23631)  

**Abstract**: What drives exploration? Understanding intrinsic motivation is a long-standing challenge in both cognitive science and artificial intelligence; numerous objectives have been proposed and used to train agents, yet there remains a gap between human and agent exploration. We directly compare adults, children, and AI agents in a complex open-ended environment, Crafter, and study how common intrinsic objectives: Entropy, Information Gain, and Empowerment, relate to their behavior. We find that only Entropy and Empowerment are consistently positively correlated with human exploration progress, indicating that these objectives may better inform intrinsic reward design for agents. Furthermore, across agents and humans we observe that Entropy initially increases rapidly, then plateaus, while Empowerment increases continuously, suggesting that state diversity may provide more signal in early exploration, while advanced exploration should prioritize control. Finally, we find preliminary evidence that private speech utterances, and particularly goal verbalizations, may aid exploration in children. 

**Abstract (ZH)**: 什么是推动探索的动力？理解内在动机是认知科学和人工智能领域长期面临的挑战；尽管提出了众多目标用于训练代理，但人类和代理的探索之间仍然存在差距。我们直接将成人、儿童和AI代理置于复杂的开放性环境中Crafter进行比较，并研究熵、信息增益和權力这三种常见内在目标与其行为之间的关系。研究发现，只有熵和權力与人类的探索进程表现出一致的正相关，表明这些目标可能更好地指导代理的内在奖励设计。此外，我们发现，在代理和人类中，熵最初迅速增加，随后趋于稳定，而權力持续增加，这表明早期探索中状态多样性可能提供更多信号，而高级探索应优先考虑控制。最后，我们初步发现，私人性言语陈述，尤其是目标言语化，可能有助于儿童的探索。 

---
# Beyond Detection: Designing AI-Resilient Assessments with Automated Feedback Tool to Foster Critical Thinking 

**Title (ZH)**: 超越检测：设计具有自动反馈工具的AI抗扰评估以培养批判性思维 

**Authors**: Muhammad Sajjad Akbar  

**Link**: [PDF](https://arxiv.org/pdf/2503.23622)  

**Abstract**: The growing use of generative AI tools like ChatGPT has raised urgent concerns about their impact on student learning, particularly the potential erosion of critical thinking and creativity. As students increasingly turn to these tools to complete assessments, foundational cognitive skills are at risk of being bypassed, challenging the integrity of higher education and the authenticity of student work. Existing AI-generated text detection tools are inadequate; they produce unreliable outputs and are prone to both false positives and false negatives, especially when students apply paraphrasing, translation, or rewording. These systems rely on shallow statistical patterns rather than true contextual or semantic understanding, making them unsuitable as definitive indicators of AI misuse. In response, this research proposes a proactive, AI-resilient solution based on assessment design rather than detection. It introduces a web-based Python tool that integrates Bloom's Taxonomy with advanced natural language processing techniques including GPT-3.5 Turbo, BERT-based semantic similarity, and TF-IDF metrics to evaluate the AI-solvability of assessment tasks. By analyzing surface-level and semantic features, the tool helps educators determine whether a task targets lower-order thinking such as recall and summarization or higher-order skills such as analysis, evaluation, and creation, which are more resistant to AI automation. This framework empowers educators to design cognitively demanding, AI-resistant assessments that promote originality, critical thinking, and fairness. It offers a sustainable, pedagogically sound strategy to foster authentic learning and uphold academic standards in the age of AI. 

**Abstract (ZH)**: 生成式AI工具（如ChatGPT）的广泛应用引起了对学生学习影响的紧迫关切，特别是批判性思维和创造力可能受损的问题。随着学生越来越多地依赖这些工具来完成评估任务，基础认知技能面临被绕过的风险，这挑战了高等教育的完整性和学生作品的真实性。现有的AI生成文本检测工具不足，它们输出不可靠，并且容易产生误报和漏报，尤其是在学生使用改写、翻译或重新措辞时。这些系统依赖于浅层次的统计模式而非真正的上下文或语义理解，使其不适合作为AI不当使用的确切指标。为应对这一挑战，本研究提出了一种基于评估设计的前瞻性和AI抗性解决方案，而非依赖检测。该研究介绍了一个基于Python的web工具，该工具结合了布鲁姆分类学与高级自然语言处理技术，包括GPT-3.5 Turbo、基于BERT的语义相似度以及TF-IDF指标，以评估评估任务的AI可解性。通过分析表层和语义特征，该工具帮助教育工作者判断任务是针对较低层次的思考能力（如记忆和总结）还是较高层次的思考能力（如分析、评估和创造），后者更难以被AI自动化。该框架赋予教育工作者设计认知要求高、对抗AI的评估工具的能力，以促进原创性、批判性思维和公平性。它提供了一种可持续且教育学上合理的策略，以促进真实的学习并维护AI时代的学术标准。 

---
# An Organizationally-Oriented Approach to Enhancing Explainability and Control in Multi-Agent Reinforcement Learning 

**Title (ZH)**: 面向组织的多代理 reinforcement 学习解释性和可控性增强方法 

**Authors**: Julien Soulé, Jean-Paul Jamont, Michel Occello, Louis-Marie Traonouez, Paul Théron  

**Link**: [PDF](https://arxiv.org/pdf/2503.23615)  

**Abstract**: Multi-Agent Reinforcement Learning can lead to the development of collaborative agent behaviors that show similarities with organizational concepts. Pushing forward this perspective, we introduce a novel framework that explicitly incorporates organizational roles and goals from the $\mathcal{M}OISE^+$ model into the MARL process, guiding agents to satisfy corresponding organizational constraints. By structuring training with roles and goals, we aim to enhance both the explainability and control of agent behaviors at the organizational level, whereas much of the literature primarily focuses on individual agents. Additionally, our framework includes a post-training analysis method to infer implicit roles and goals, offering insights into emergent agent behaviors. This framework has been applied across various MARL environments and algorithms, demonstrating coherence between predefined organizational specifications and those inferred from trained agents. 

**Abstract (ZH)**: 多智能体强化学习可以发展出与组织概念相似的合作智能体行为。在此基础上，我们提出了一种新颖的框架，该框架明确地将$\mathcal{M}OISE^+$模型中的组织角色和目标纳入多智能体强化学习过程，引导智能体满足相应的组织约束。通过按角色和目标结构化训练，我们旨在增强智能体行为在组织层面的可解释性和可控性，而现有文献主要关注个体智能体。此外，我们的框架还包括一种后训练分析方法，用于推断隐含的角色和目标，提供对涌现智能体行为的见解。该框架已在各种多智能体强化学习环境中应用，展示了预定义的组织规范与从训练智能体推断出的规范之间的一致性。 

---
# GenVP: Generating Visual Puzzles with Contrastive Hierarchical VAEs 

**Title (ZH)**: GenVP：基于对比层次VAEs的视觉谜题生成 

**Authors**: Kalliopi Basioti, Pritish Sahu, Qingze Tony Liu, Zihao Xu, Hao Wang, Vladimir Pavlovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.23598)  

**Abstract**: Raven's Progressive Matrices (RPMs) is an established benchmark to examine the ability to perform high-level abstract visual reasoning (AVR). Despite the current success of algorithms that solve this task, humans can generalize beyond a given puzzle and create new puzzles given a set of rules, whereas machines remain locked in solving a fixed puzzle from a curated choice list. We propose Generative Visual Puzzles (GenVP), a framework to model the entire RPM generation process, a substantially more challenging task. Our model's capability spans from generating multiple solutions for one specific problem prompt to creating complete new puzzles out of the desired set of rules. Experiments on five different datasets indicate that GenVP achieves state-of-the-art (SOTA) performance both in puzzle-solving accuracy and out-of-distribution (OOD) generalization in 22 OOD scenarios. Compared to SOTA generative approaches, which struggle to solve RPMs when the feasible solution space increases, GenVP efficiently generalizes to these challenging setups. Moreover, our model demonstrates the ability to produce a wide range of complete RPMs given a set of abstract rules by effectively capturing the relationships between abstract rules and visual object properties. 

**Abstract (ZH)**: 生成视觉推理谜题（GenVP）：一种全新的高阶抽象视觉推理生成框架 

---
# Benchmarking Systematic Relational Reasoning with Large Language and Reasoning Models 

**Title (ZH)**: 大规模语言和推理模型中系统关系推理的基准测试 

**Authors**: Irtaza Khalid, Amir Masoud Nourollah, Steven Schockaert  

**Link**: [PDF](https://arxiv.org/pdf/2503.23487)  

**Abstract**: Large Language Models (LLMs) have been found to struggle with systematic reasoning. Even on tasks where they appear to perform well, their performance often depends on shortcuts, rather than on genuine reasoning abilities, leading them to collapse on out-of-distribution examples. Post-training strategies based on reinforcement learning and chain-of-thought prompting have recently been hailed as a step change. However, little is still known about the potential of the resulting ``Large Reasoning Models'' (LRMs) beyond problem solving in mathematics and programming, where finding genuine out-of-distribution problems can be difficult. In this paper, we focus on tasks that require systematic reasoning about relational compositions, especially for qualitative spatial and temporal reasoning. These tasks allow us to control the difficulty of problem instances, and measure in a precise way to what extent models can generalise. We find that that the considered LLMs and LRMs overall perform poorly overall, albeit better than random chance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在系统性推理方面表现不佳。即使在它们表现似乎很好的任务中，它们的表现往往依赖于捷径，而不是真正的推理能力，导致它们在分布外示例中失效。基于强化学习和推理链的后训练策略最近被认为是一个重大突破。然而，除了在数学和编程问题解决领域之外，关于由此产生的“大型推理模型”（LRMs）的潜力知之甚少，在这些领域中生成具有挑战性的分布外问题可能较为困难。在本文中，我们专注于那些需要对关系组成进行系统性推理的任务，特别是在定性空间和时间推理方面。这些任务使我们能够控制问题实例的难度，并精确测量模型的泛化能力。我们发现，所考虑的LLMs和LRMs整体表现不佳，尽管比随机猜测要好。 

---
# A Systematic Decade Review of Trip Route Planning with Travel Time Estimation based on User Preferences and Behavior 

**Title (ZH)**: 基于用户偏好和行为的旅行时间估算的旅游路线规划系统性十年回顾 

**Authors**: Nikil Jayasuriya, Deshan Sumanathilaka  

**Link**: [PDF](https://arxiv.org/pdf/2503.23486)  

**Abstract**: This paper systematically explores the advancements in adaptive trip route planning and travel time estimation (TTE) through Artificial Intelligence (AI). With the increasing complexity of urban transportation systems, traditional navigation methods often struggle to accommodate dynamic user preferences, real-time traffic conditions, and scalability requirements. This study explores the contributions of established AI techniques, including Machine Learning (ML), Reinforcement Learning (RL), and Graph Neural Networks (GNNs), alongside emerging methodologies like Meta-Learning, Explainable AI (XAI), Generative AI, and Federated Learning. In addition to highlighting these innovations, the paper identifies critical challenges such as ethical concerns, computational scalability, and effective data integration, which must be addressed to advance the field. The paper concludes with recommendations for leveraging AI to build efficient, transparent, and sustainable navigation systems. 

**Abstract (ZH)**: 本文系统性地探讨了人工智能在自适应旅行路线规划和旅行时间估计（TTE）方面的进步。随着城市交通系统的日益复杂，传统的导航方法往往难以满足动态用户偏好、实时交通状况和可扩展性要求。本研究探索了包括机器学习（ML）、强化学习（RL）和图神经网络（GNN）在内的现有AI技术的贡献，以及元学习、可解释的AI（XAI）、生成AI和联邦学习等新兴方法。除了强调这些创新之外，本文还指出了诸如伦理问题、计算可扩展性和有效的数据集成等关键挑战，这些挑战必须得到解决以推进该领域的发展。本文最后提出了利用AI构建高效、透明和可持续导航系统的建议。 

---
# Large Language Models Are Better Logical Fallacy Reasoners with Counterargument, Explanation, and Goal-Aware Prompt Formulation 

**Title (ZH)**: 大规模语言模型在反论、解释和目标导向提示 formulation 下是更好的逻辑谬误推理器。 

**Authors**: Jiwon Jeong, Hyeju Jang, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.23363)  

**Abstract**: The advancement of Large Language Models (LLMs) has greatly improved our ability to process complex language. However, accurately detecting logical fallacies remains a significant challenge. This study presents a novel and effective prompt formulation approach for logical fallacy detection, applicable in both supervised (fine-tuned) and unsupervised (zero-shot) settings. Our method enriches input text incorporating implicit contextual information -- counterarguments, explanations, and goals -- which we query for validity within the context of the argument. We then rank these queries based on confidence scores to inform classification. We evaluate our approach across multiple datasets from 5 domains, covering 29 distinct fallacy types, using models from the GPT and LLaMA series. The results show substantial improvements over state-of-the-art models, with F1 score increases of up to 0.60 in zero-shot settings and up to 0.45 in fine-tuned models. Extensive analyses further illustrate why and how our method excels. 

**Abstract (ZH)**: 大型语言模型的进步大大提高了我们处理复杂语言的能力。然而，准确检测逻辑谬误仍是一项重大挑战。本研究提出了一种新颖而有效的提示构建方法，适用于监督（微调）和无监督（零样本）设置中的逻辑谬误检测。我们的方法通过整合输入文本中的隐含上下文信息——反论、解释和目标——来增强输入文本，并在论点的上下文中查询这些信息的有效性。然后，我们根据置信度评分对这些查询进行排序，以指导分类。我们在涵盖29种不同谬误类型的5个领域多个数据集中评估了我们的方法，使用了GPT和LLaMA系列模型。结果表明，在零样本设置下，我们的方法显著优于最新模型，F1分数提高了0.60，在微调模型中提高了0.45。进一步的详尽分析还阐明了我们的方法为何以及如何优越。 

---
# A Survey of WebAgents: Towards Next-Generation AI Agents for Web Automation with Large Foundation Models 

**Title (ZH)**: Web代理综述：面向基于大规模基础模型的下一代网络自动化AI代理 

**Authors**: Liangbo Ning, Ziran Liang, Zhuohang Jiang, Haohao Qu, Yujuan Ding, Wenqi Fan, Xiao-yong Wei, Shanru Lin, Hui Liu, Philip S. Yu, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.23350)  

**Abstract**: With the advancement of web techniques, they have significantly revolutionized various aspects of people's lives. Despite the importance of the web, many tasks performed on it are repetitive and time-consuming, negatively impacting overall quality of life. To efficiently handle these tedious daily tasks, one of the most promising approaches is to advance autonomous agents based on Artificial Intelligence (AI) techniques, referred to as AI Agents, as they can operate continuously without fatigue or performance degradation. In the context of the web, leveraging AI Agents -- termed WebAgents -- to automatically assist people in handling tedious daily tasks can dramatically enhance productivity and efficiency. Recently, Large Foundation Models (LFMs) containing billions of parameters have exhibited human-like language understanding and reasoning capabilities, showing proficiency in performing various complex tasks. This naturally raises the question: `Can LFMs be utilized to develop powerful AI Agents that automatically handle web tasks, providing significant convenience to users?' To fully explore the potential of LFMs, extensive research has emerged on WebAgents designed to complete daily web tasks according to user instructions, significantly enhancing the convenience of daily human life. In this survey, we comprehensively review existing research studies on WebAgents across three key aspects: architectures, training, and trustworthiness. Additionally, several promising directions for future research are explored to provide deeper insights. 

**Abstract (ZH)**: 随着网络技术的进步，它们在多个方面显著地革新了人们的生活。尽管网络很重要，但其中许多任务是重复性和耗时的，负面影响了整体生活质量。为了高效地处理这些繁琐的日常任务，最 promising 的方法之一是基于人工智能（AI）技术的进步自主代理，称为AI代理，因为它们可以在不感到疲劳或性能下降的情况下连续工作。在Web的背景下，利用被称为WebAgents的AI代理自动协助人们处理繁琐的日常任务，可以大幅度提高生产力和效率。最近，包含数十亿参数的大规模基础模型（LFMs）展示了类似人类的语言理解和推理能力，并且在执行各种复杂任务方面表现出色。这自然引出了一个问题：`LFMs能否用于开发强大的AI代理，自动处理网络任务，为用户提供显著便利？`为了全面探索LFMs的潜力，已经开展了大量研究，旨在设计能够根据用户指令完成日常网络任务的WebAgents，极大地提升了日常生活的便利性。在这篇综述中，我们从架构、训练和可信度三个方面全面回顾了现有WebAgents的研究，并探讨了若干具有前景的研究方向，以提供更深入的见解。 

---
# A Scalable Framework for Evaluating Health Language Models 

**Title (ZH)**: 可扩展的健康语言模型评估框架 

**Authors**: Neil Mallinar, A. Ali Heydari, Xin Liu, Anthony Z. Faranesh, Brent Winslow, Nova Hammerquist, Benjamin Graef, Cathy Speed, Mark Malhotra, Shwetak Patel, Javier L. Prieto, Daniel McDuff, Ahmed A. Metwally  

**Link**: [PDF](https://arxiv.org/pdf/2503.23339)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools for analyzing complex datasets. Recent studies demonstrate their potential to generate useful, personalized responses when provided with patient-specific health information that encompasses lifestyle, biomarkers, and context. As LLM-driven health applications are increasingly adopted, rigorous and efficient one-sided evaluation methodologies are crucial to ensure response quality across multiple dimensions, including accuracy, personalization and safety. Current evaluation practices for open-ended text responses heavily rely on human experts. This approach introduces human factors and is often cost-prohibitive, labor-intensive, and hinders scalability, especially in complex domains like healthcare where response assessment necessitates domain expertise and considers multifaceted patient data. In this work, we introduce Adaptive Precise Boolean rubrics: an evaluation framework that streamlines human and automated evaluation of open-ended questions by identifying gaps in model responses using a minimal set of targeted rubrics questions. Our approach is based on recent work in more general evaluation settings that contrasts a smaller set of complex evaluation targets with a larger set of more precise, granular targets answerable with simple boolean responses. We validate this approach in metabolic health, a domain encompassing diabetes, cardiovascular disease, and obesity. Our results demonstrate that Adaptive Precise Boolean rubrics yield higher inter-rater agreement among expert and non-expert human evaluators, and in automated assessments, compared to traditional Likert scales, while requiring approximately half the evaluation time of Likert-based methods. This enhanced efficiency, particularly in automated evaluation and non-expert contributions, paves the way for more extensive and cost-effective evaluation of LLMs in health. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为分析复杂数据集的强大工具。最近的研究表明，当提供包括生活方式、生物标志物和上下文在内的患者特定健康信息时，LLM有可能生成有用且个性化的回应。随着LLM驱动的健康应用的日益普及，需要严格且高效的一方评价方法来确保在多个维度上（包括准确性、个性化和安全性）的回应质量。当前对开放文本回应的评价方法高度依赖于人类专家。这种方法引入了人为因素，通常成本高昂、劳动密集且阻碍可扩展性，尤其是在需要领域专业知识来评估多方面患者数据的复杂领域如医疗健康。在这项工作中，我们引入了自适应精确布尔评判标准：一种评价框架，通过使用一组靶向评判标准问题来识别模型回应中的缺口，从而简化人类和自动化的评价过程。我们的方法基于近年来在更通用的评价环境中对比一组复杂的评价目标与一组更精确、可使用简单布尔响应回答的细粒度目标的工作。我们通过代谢健康这一涵盖糖尿病、心血管疾病和肥胖的领域验证了该方法。结果表明，自适应精确布尔评判标准在专家和非专家人类评价者之间以及自动评估中相较于传统的李克特量表具有更高的评价者间一致性，且所需评价时间大约仅为李克特量表方法的一半。这种增强的效率，尤其是在自动化评价和非专家贡献中的效率，为在医疗健康领域更广泛且成本效益更高的LLM评价铺平了道路。 

---
# A Multi-Agent Framework with Automated Decision Rule Optimization for Cross-Domain Misinformation Detection 

**Title (ZH)**: 跨域 misinformation 检测的自动决策规则优化多Agent框架 

**Authors**: Hui Li, Ante Wang, kunquan li, Zhihao Wang, Liang Zhang, Delai Qiu, Qingsong Liu, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.23329)  

**Abstract**: Misinformation spans various domains, but detection methods trained on specific domains often perform poorly when applied to others. With the rapid development of Large Language Models (LLMs), researchers have begun to utilize LLMs for cross-domain misinformation detection. However, existing LLM-based methods often fail to adequately analyze news in the target domain, limiting their detection capabilities. More importantly, these methods typically rely on manually designed decision rules, which are limited by domain knowledge and expert experience, thus limiting the generalizability of decision rules to different domains. To address these issues, we propose a MultiAgent Framework for cross-domain misinformation detection with Automated Decision Rule Optimization (MARO). Under this framework, we first employs multiple expert agents to analyze target-domain news. Subsequently, we introduce a question-reflection mechanism that guides expert agents to facilitate higherquality analysis. Furthermore, we propose a decision rule optimization approach based on carefully-designed cross-domain validation tasks to iteratively enhance the effectiveness of decision rules in different domains. Experimental results and in-depth analysis on commonlyused datasets demonstrate that MARO achieves significant improvements over existing methods. 

**Abstract (ZH)**: 一种用于跨域 misinformation 检测的多智能体框架及自动决策规则优化（MARO） 

---
# Exploring Explainable Multi-player MCTS-minimax Hybrids in Board Game Using Process Mining 

**Title (ZH)**: 探索基于过程挖掘的可解释多玩家MCTS- minimax混合算法在棋类游戏中的应用 

**Authors**: Yiyu Qian, Tim Miller, Zheng Qian, Liyuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.23326)  

**Abstract**: Monte-Carlo Tree Search (MCTS) is a family of sampling-based search algorithms widely used for online planning in sequential decision-making domains and at the heart of many recent advances in artificial intelligence. Understanding the behavior of MCTS agents is difficult for developers and users due to the frequently large and complex search trees that result from the simulation of many possible futures, their evaluations, and their relationships. This paper presents our ongoing investigation into potential explanations for the decision-making and behavior of MCTS. A weakness of MCTS is that it constructs a highly selective tree and, as a result, can miss crucial moves and fall into tactical traps. Full-width minimax search constitutes the solution. We integrate shallow minimax search into the rollout phase of multi-player MCTS and use process mining technique to explain agents' strategies in 3v3 checkers. 

**Abstract (ZH)**: 基于蒙特卡罗树搜索的博弈决策与行为研究：浅层次极小极大搜索在三目国际象棋中的应用 

---
# AI Agents in Engineering Design: A Multi-Agent Framework for Aesthetic and Aerodynamic Car Design 

**Title (ZH)**: AI代理在工程设计中的应用：一种用于美学和空气动力学汽车设计的多代理框架 

**Authors**: Mohamed Elrefaie, Janet Qian, Raina Wu, Qian Chen, Angela Dai, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2503.23315)  

**Abstract**: We introduce the concept of "Design Agents" for engineering applications, particularly focusing on the automotive design process, while emphasizing that our approach can be readily extended to other engineering and design domains. Our framework integrates AI-driven design agents into the traditional engineering workflow, demonstrating how these specialized computational agents interact seamlessly with engineers and designers to augment creativity, enhance efficiency, and significantly accelerate the overall design cycle. By automating and streamlining tasks traditionally performed manually, such as conceptual sketching, styling enhancements, 3D shape retrieval and generative modeling, computational fluid dynamics (CFD) meshing, and aerodynamic simulations, our approach reduces certain aspects of the conventional workflow from weeks and days down to minutes. These agents leverage state-of-the-art vision-language models (VLMs), large language models (LLMs), and geometric deep learning techniques, providing rapid iteration and comprehensive design exploration capabilities. We ground our methodology in industry-standard benchmarks, encompassing a wide variety of conventional automotive designs, and utilize high-fidelity aerodynamic simulations to ensure practical and applicable outcomes. Furthermore, we present design agents that can swiftly and accurately predict simulation outcomes, empowering engineers and designers to engage in more informed design optimization and exploration. This research underscores the transformative potential of integrating advanced generative AI techniques into complex engineering tasks, paving the way for broader adoption and innovation across multiple engineering disciplines. 

**Abstract (ZH)**: 我们引入了“设计代理”概念，特别应用于汽车设计过程，并强调我们的方法可以轻易扩展到其他工程和设计领域。我们的框架将AI驱动的设计代理融入传统的工程工作流程中，展示了这些专门化的计算代理如何无缝地与工程师和设计师交互，增强创造力、提高效率，并显著加速整个设计周期。通过自动化和简化传统手工完成的任务，如概念草图、风格优化、3D形状检索和生成建模、计算流体动力学（CFD）网格划分和气动模拟，我们的方法将某些传统工作流程中的时间从几周或几天压缩到几分钟。这些代理利用最先进的视觉-语言模型（VLMs）、大规模语言模型（LLMs）和几何深度学习技术，提供快速迭代和全面的设计探索能力。我们基于行业标准基准，涵盖广泛的传统汽车设计，并利用高保真气动模拟以确保实用和适用的结果。此外，我们展示了可以快速准确预测模拟结果的设计代理，使工程师和设计师能够进行更加知情的设计优化和探索。本研究强调了将先进的生成AI技术集成到复杂工程任务中的变革潜力，为多个工程学科的更广泛采用和创新铺平了道路。 

---
# SPIO: Ensemble and Selective Strategies via LLM-Based Multi-Agent Planning in Automated Data Science 

**Title (ZH)**: SPIO：基于LLM的多Agent规划的集成与选择策略在自动化数据科学中的应用 

**Authors**: Wonduk Seo, Juhyeon Lee, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23314)  

**Abstract**: Large Language Models (LLMs) have revolutionized automated data analytics and machine learning by enabling dynamic reasoning and adaptability. While recent approaches have advanced multi-stage pipelines through multi-agent systems, they typically rely on rigid, single-path workflows that limit the exploration and integration of diverse strategies, often resulting in suboptimal predictions. To address these challenges, we propose SPIO (Sequential Plan Integration and Optimization), a novel framework that leverages LLM-driven decision-making to orchestrate multi-agent planning across four key modules: data preprocessing, feature engineering, modeling, and hyperparameter tuning. In each module, dedicated planning agents independently generate candidate strategies that cascade into subsequent stages, fostering comprehensive exploration. A plan optimization agent refines these strategies by suggesting several optimized plans. We further introduce two variants: SPIO-S, which selects a single best solution path as determined by the LLM, and SPIO-E, which selects the top k candidate plans and ensembles them to maximize predictive performance. Extensive experiments on Kaggle and OpenML datasets demonstrate that SPIO significantly outperforms state-of-the-art methods, providing a robust and scalable solution for automated data science task. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过实现动态推理和适应性，已经革新了自动化数据分析师和机器学习。尽管近期方法通过多智能体系统促进了多阶段管道的发展，但它们通常依赖于僵化的单路径工作流程，限制了不同策略的探索和集成，经常导致次优预测。为了解决这些挑战，我们提出了一种名为SPIO（Sequential Plan Integration and Optimization）的新框架，该框架利用LLM驱动的决策来协调四个关键模块——数据预处理、特征工程、建模和超参数调整中的多智能体规划。在每个模块中，专门的规划智能体独立生成候选策略，这些策略逐步传递到后续阶段，促进全面的探索。一个计划优化智能体通过建议多种优化计划来改进这些策略。我们还介绍了两个变体：SPIO-S，它选择由LLM确定的最佳解决方案路径；SPIO-E，它选择前k个候选计划并将其集成以最大化预测性能。在Kaggle和OpenML数据集上的广泛实验表明，SPIO显著优于现有方法，提供了自动化数据科学任务的稳健且可扩展的解决方案。 

---
# LaViC: Adapting Large Vision-Language Models to Visually-Aware Conversational Recommendation 

**Title (ZH)**: LaViC: 调整大型视觉-语言模型以实现视觉意识对话推荐 

**Authors**: Hyunsik Jeon, Satoshi Koide, Yu Wang, Zhankui He, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2503.23312)  

**Abstract**: Conversational recommender systems engage users in dialogues to refine their needs and provide more personalized suggestions. Although textual information suffices for many domains, visually driven categories such as fashion or home decor potentially require detailed visual information related to color, style, or design. To address this challenge, we propose LaViC (Large Vision-Language Conversational Recommendation Framework), a novel approach that integrates compact image representations into dialogue-based recommendation systems. LaViC leverages a large vision-language model in a two-stage process: (1) visual knowledge self-distillation, which condenses product images from hundreds of tokens into a small set of visual tokens in a self-distillation manner, significantly reducing computational overhead, and (2) recommendation prompt tuning, which enables the model to incorporate both dialogue context and distilled visual tokens, providing a unified mechanism for capturing textual and visual features. To support rigorous evaluation of visually-aware conversational recommendation, we construct a new dataset by aligning Reddit conversations with Amazon product listings across multiple visually oriented categories (e.g., fashion, beauty, and home). This dataset covers realistic user queries and product appearances in domains where visual details are crucial. Extensive experiments demonstrate that LaViC significantly outperforms text-only conversational recommendation methods and open-source vision-language baselines. Moreover, LaViC achieves competitive or superior accuracy compared to prominent proprietary baselines (e.g., GPT-3.5-turbo, GPT-4o-mini, and GPT-4o), demonstrating the necessity of explicitly using visual data for capturing product attributes and showing the effectiveness of our vision-language integration. Our code and dataset are available at this https URL. 

**Abstract (ZH)**: 基于视觉语言的大规模对话推荐框架（LaViC） 

---
# GRASP: Municipal Budget AI Chatbots for Enhancing Civic Engagement 

**Title (ZH)**: GRASP: 市级预算AI聊天机器人以增强公民参与度 

**Authors**: Jerry Xu, Justin Wang, Joley Leung, Jasmine Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23299)  

**Abstract**: There are a growing number of AI applications, but none tailored specifically to help residents answer their questions about municipal budget, a topic most are interested in but few have a solid comprehension of. In this research paper, we propose GRASP, a custom AI chatbot framework which stands for Generation with Retrieval and Action System for Prompts. GRASP provides more truthful and grounded responses to user budget queries than traditional information retrieval systems like general Large Language Models (LLMs) or web searches. These improvements come from the novel combination of a Retrieval-Augmented Generation (RAG) framework ("Generation with Retrieval") and an agentic workflow ("Action System"), as well as prompt engineering techniques, the incorporation of municipal budget domain knowledge, and collaboration with local town officials to ensure response truthfulness. During testing, we found that our GRASP chatbot provided precise and accurate responses for local municipal budget queries 78% of the time, while GPT-4o and Gemini were only accurate 60% and 35% of the time, respectively. GRASP chatbots greatly reduce the time and effort needed for the general public to get an intuitive and correct understanding of their town's budget, thus fostering greater communal discourse, improving government transparency, and allowing citizens to make more informed decisions. 

**Abstract (ZH)**: 面向市政预算查询的定制AI聊天机器人框架：GRASP 

---
# Ethereum Price Prediction Employing Large Language Models for Short-term and Few-shot Forecasting 

**Title (ZH)**: 使用大型语言模型进行以太坊短期和少样本价格预测 

**Authors**: Eftychia Makri, Georgios Palaiokrassas, Sarah Bouraga, Antigoni Polychroniadou, Leandros Tassiulas  

**Link**: [PDF](https://arxiv.org/pdf/2503.23190)  

**Abstract**: Cryptocurrencies have transformed financial markets with their innovative blockchain technology and volatile price movements, presenting both challenges and opportunities for predictive analytics. Ethereum, being one of the leading cryptocurrencies, has experienced significant market fluctuations, making its price prediction an attractive yet complex problem. This paper presents a comprehensive study on the effectiveness of Large Language Models (LLMs) in predicting Ethereum prices for short-term and few-shot forecasting scenarios. The main challenge in training models for time series analysis is the lack of data. We address this by leveraging a novel approach that adapts existing pre-trained LLMs on natural language or images from billions of tokens to the unique characteristics of Ethereum price time series data. Through thorough experimentation and comparison with traditional and contemporary models, our results demonstrate that selectively freezing certain layers of pre-trained LLMs achieves state-of-the-art performance in this domain. This approach consistently surpasses benchmarks across multiple metrics, including Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE), demonstrating its effectiveness and robustness. Our research not only contributes to the existing body of knowledge on LLMs but also provides practical insights in the cryptocurrency prediction domain. The adaptability of pre-trained LLMs to handle the nature of Ethereum prices suggests a promising direction for future research, potentially including the integration of sentiment analysis to further refine forecasting accuracy. 

**Abstract (ZH)**: 加密货币通过其创新的区块链技术和波动的价格变化，已经改变了金融市场，为预测分析带来了挑战与机会。以以太币为例，作为一种领先的加密货币，其市场波动显著，使价格预测成为一个既吸引人又复杂的难题。本文对大规模语言模型（LLMs）在预测以太币价格方面的有效性进行了综合性研究，特别是针对短期和少样本预测场景。时间序列分析模型训练的主要挑战在于数据不足。我们通过一种新颖的方法，将现有的预训练LLMs从数亿个令牌中的自然语言或图像数据改编为以太币价格时间序列数据的独特特征来解决这一问题。通过彻底的实验和与传统及当代模型的比较，我们的结果表明，选择性地冻结预训练LLMs中的某些层可以在此领域实现最先进的性能。这种方法在多个指标上，包括均方误差（MSE）、平均绝对误差（MAE）和均方根误差（RMSE）上都超过了基准模型，展示了其有效性和稳健性。我们的研究不仅丰富了现有的大规模语言模型知识库，还为加密货币预测领域提供了实用的见解。预训练LLMs对处理以太币价格性质的高度适应性显示了一个有前景的研究方向，未来的研究可能包括整合情感分析以进一步提高预测准确性。 

---
# AstroAgents: A Multi-Agent AI for Hypothesis Generation from Mass Spectrometry Data 

**Title (ZH)**: AstroAgents: 多智能体AI在质谱数据分析中的假说生成 

**Authors**: Daniel Saeedi, Denise Buckner, Jose C. Aponte, Amirali Aghazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2503.23170)  

**Abstract**: With upcoming sample return missions across the solar system and the increasing availability of mass spectrometry data, there is an urgent need for methods that analyze such data within the context of existing astrobiology literature and generate plausible hypotheses regarding the emergence of life on Earth. Hypothesis generation from mass spectrometry data is challenging due to factors such as environmental contaminants, the complexity of spectral peaks, and difficulties in cross-matching these peaks with prior studies. To address these challenges, we introduce AstroAgents, a large language model-based, multi-agent AI system for hypothesis generation from mass spectrometry data. AstroAgents is structured around eight collaborative agents: a data analyst, a planner, three domain scientists, an accumulator, a literature reviewer, and a critic. The system processes mass spectrometry data alongside user-provided research papers. The data analyst interprets the data, and the planner delegates specific segments to the scientist agents for in-depth exploration. The accumulator then collects and deduplicates the generated hypotheses, and the literature reviewer identifies relevant literature using Semantic Scholar. The critic evaluates the hypotheses, offering rigorous suggestions for improvement. To assess AstroAgents, an astrobiology expert evaluated the novelty and plausibility of more than a hundred hypotheses generated from data obtained from eight meteorites and ten soil samples. Of these hypotheses, 36% were identified as plausible, and among those, 66% were novel. Project website: this https URL 

**Abstract (ZH)**: 随着太阳系内取样返回任务的即将到来以及质谱数据分析的日益增多，迫切需要能够将此类数据纳入现有天体生物学文献分析的方法，并生成关于生命在地球上出现的合理假设。基于质谱数据的假设生成极具挑战性，原因包括环境污染物、谱峰复杂性以及跨研究匹配这些峰值的困难。为解决这些挑战，我们引入了AstroAgents——一个多智能体AI系统，基于大型语言模型，用于从质谱数据中生成假设。AstroAgents由八个协作智能体构成：数据分析师、规划师、三位领域科学家、积累器、文献审查员和批判者。该系统在处理质谱数据的同时，还能与用户提供的研究论文一同工作。数据分析师解读数据，规划师将具体的任务分配给科学家智能体进行深入探索。积累器收集并去重生成的假设，文献审查员使用Semantic Scholar识别相关文献，批判者评估假设，提供严格的改进建议。为了评估AstroAgents，一位天体生物学专家评估了从八块陨石和十份土壤样本的数据中生成的超过一百个假设的新颖性和合理性。在这其中，36%被认定为合理，而在这其中又有66%是新颖的。项目网站：https://this-url。 

---
# Agentic Large Language Models, a survey 

**Title (ZH)**: 代理型大型语言模型：一项综述 

**Authors**: Aske Plaat, Max van Duijn, Niki van Stein, Mike Preuss, Peter van der Putten, Kees Joost Batenburg  

**Link**: [PDF](https://arxiv.org/pdf/2503.23037)  

**Abstract**: There is great interest in agentic LLMs, large language models that act as agents. We review the growing body of work in this area and provide a research agenda. Agentic LLMs are LLMs that (1) reason, (2) act, and (3) interact. We organize the literature according to these three categories. The research in the first category focuses on reasoning, reflection, and retrieval, aiming to improve decision making; the second category focuses on action models, robots, and tools, aiming for agents that act as useful assistants; the third category focuses on multi-agent systems, aiming for collaborative task solving and simulating interaction to study emergent social behavior. We find that works mutually benefit from results in other categories: retrieval enables tool use, reflection improves multi-agent collaboration, and reasoning benefits all categories. We discuss applications of agentic LLMs and provide an agenda for further research. Important applications are in medical diagnosis, logistics and financial market analysis. Meanwhile, self-reflective agents playing roles and interacting with one another augment the process of scientific research itself. Further, agentic LLMs may provide a solution for the problem of LLMs running out of training data: inference-time behavior generates new training states, such that LLMs can keep learning without needing ever larger datasets. We note that there is risk associated with LLM assistants taking action in the real world, while agentic LLMs are also likely to benefit society. 

**Abstract (ZH)**: 基于代理的大规模语言模型：研究进展与展望 

---
# FindTheFlaws: Annotated Errors for Detecting Flawed Reasoning and Scalable Oversight Research 

**Title (ZH)**: FindTheFlaws: 注释错误以检测瑕疵推理与可扩展监督研究 

**Authors**: Gabriel Recchia, Chatrik Singh Mangat, Issac Li, Gayatri Krishnakumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.22989)  

**Abstract**: As AI models tackle increasingly complex problems, ensuring reliable human oversight becomes more challenging due to the difficulty of verifying solutions. Approaches to scaling AI supervision include debate, in which two agents engage in structured dialogue to help a judge evaluate claims; critique, in which models identify potential flaws in proposed solutions; and prover-verifier games, in which a capable 'prover' model generates solutions that must be verifiable by a less capable 'verifier'. Evaluations of the scalability of these and similar approaches to difficult problems benefit from datasets that include (1) long-form expert-verified correct solutions and (2) long-form flawed solutions with annotations highlighting specific errors, but few are available.
To address this gap, we present FindTheFlaws, a group of five diverse datasets spanning medicine, mathematics, science, coding, and the Lojban language. Each dataset contains questions and long-form solutions with expert annotations validating their correctness or identifying specific error(s) in the reasoning. We evaluate frontier models' critiquing capabilities and observe a range of performance that can be leveraged for scalable oversight experiments: models performing more poorly on particular datasets can serve as judges/verifiers for more capable models. Additionally, for some task/dataset combinations, expert baselines exceed even top model performance, making them more beneficial for scalable oversight experiments. 

**Abstract (ZH)**: 随着AI模型面临的問題日益複雜，確保可靠的人類監督變得更加困難，因為驗證解決方案的難度增大。擴展AI監督的策略包括辯論（兩個代理進行結構化對話以幫助評審 avaliação 判斷斷言）、批判（模型識別提出的解決方案中的潛在缺陷）、以及證明者-驗證者遊戲（有能力的“證明者”模型生成必須能夠被較無能力的“驗證者”模型驗證的解決方案）。對這些及類似方法擴展性的評估需要包括（1）長篇專家驗證正確的解決方案和（2）帶有標注具體錯誤的長篇錯誤解決方案等數據集，但這樣的數據集很少見。

為了彌補這一-gap，我們介紹了 FindTheFlaws，這是涵蓋醫學、數學、科學、編程和洛 Basics 語言的五個多樣化數據集組。每個數據集包含問題和長篇解決方案，並附有專家注釋以確認其 correctness 或標注推理中的特定錯誤。我們評估前沿模型的批評能力，並觀察到一系列性能，這些性能可以為可擴展的監督實驗提供支援：在某些數據集上表現較差的模型可以作為較有能力模型的評審/驗證者。此外，對於一些任務/數據集組合，專家基線表現甚至超越頂尖模型表現，使其在可擴展的監督實驗中更有益。 

---
# Identifying Multi-modal Knowledge Neurons in Pretrained Transformers via Two-stage Filtering 

**Title (ZH)**: 基于两阶段过滤的预训练变换器多模态知识神经元识别 

**Authors**: Yugen Sato, Tomohiro Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2503.22941)  

**Abstract**: Recent advances in large language models (LLMs) have led to the development of multimodal LLMs (MLLMs) in the fields of natural language processing (NLP) and computer vision. Although these models allow for integrated visual and language understanding, they present challenges such as opaque internal processing and the generation of hallucinations and misinformation. Therefore, there is a need for a method to clarify the location of knowledge in MLLMs.
In this study, we propose a method to identify neurons associated with specific knowledge using MiniGPT-4, a Transformer-based MLLM. Specifically, we extract knowledge neurons through two stages: activation differences filtering using inpainting and gradient-based filtering using GradCAM. Experiments on the image caption generation task using the MS COCO 2017 dataset, BLEU, ROUGE, and BERTScore quantitative evaluation, and qualitative evaluation using an activation heatmap showed that our method is able to locate knowledge with higher accuracy than existing methods.
This study contributes to the visualization and explainability of knowledge in MLLMs and shows the potential for future knowledge editing and control. 

**Abstract (ZH)**: 近期大规模语言模型的发展推动了多模态大语言模型（多模态LLMs）在自然语言处理（NLP）和计算机视觉领域的研究。尽管这些模型能够集成视觉和语言理解，但它们也带来了内部处理不透明及幻觉和 misinformation产生的挑战。因此，需要一种方法来澄清多模态LLMs中的知识位置。

在本研究中，我们提出了一种使用基于Transformer的多模态LLM MiniGPT-4的方法来识别与特定知识相关的神经元。具体而言，我们通过两个阶段提取知识神经元：使用 inpainting 的激活差异筛选和使用 GradCAM 的梯度筛选。通过在MS COCO 2017数据集上的图像字幕生成任务、以及基于BLEU、ROUGE和BERTScore的定量评估和基于激活热图的定性评估，我们证明了该方法能够比现有方法更准确地定位知识。 

---
# Factored Agents: Decoupling In-Context Learning and Memorization for Robust Tool Use 

**Title (ZH)**: 因子智能体：分离上下文学习和记忆以实现稳健的工具使用 

**Authors**: Nicholas Roth, Christopher Hidey, Lucas Spangher, William F. Arnold, Chang Ye, Nick Masiewicki, Jinoo Baek, Peter Grabowski, Eugene Ie  

**Link**: [PDF](https://arxiv.org/pdf/2503.22931)  

**Abstract**: In this paper, we propose a novel factored agent architecture designed to overcome the limitations of traditional single-agent systems in agentic AI. Our approach decomposes the agent into two specialized components: (1) a large language model (LLM) that serves as a high level planner and in-context learner, which may use dynamically available information in user prompts, (2) a smaller language model which acts as a memorizer of tool format and output. This decoupling addresses prevalent issues in monolithic designs, including malformed, missing, and hallucinated API fields, as well as suboptimal planning in dynamic environments. Empirical evaluations demonstrate that our factored architecture significantly improves planning accuracy and error resilience, while elucidating the inherent trade-off between in-context learning and static memorization. These findings suggest that a factored approach is a promising pathway for developing more robust and adaptable agentic AI systems. 

**Abstract (ZH)**: 本文提出了一种新颖的事实性智能体架构，旨在克服传统单一智能体系统在智能体AI领域的局限性。该方法将智能体分解为两个专门化的组件：（1）一个大型语言模型（LLM），作为高层规划者和上下文学习者，可以利用用户提示中动态可用的信息；（2）一个较小的语言模型，作为工具格式和输出的记忆器。这种分解解决了单一庞大设计中存在的诸多问题，包括错误构造、缺失和臆想的API字段，以及在动态环境下的次优规划。实证评估表明，我们的分解架构显著提高了规划准确性和错误鲁棒性，同时阐明了上下文学习与静态记忆之间固有的权衡关系。这些发现表明，分解方法是开发更 robust 和适应性强的智能体AI系统的一个有前景的途径。 

---
# LLM-based Agent Simulation for Maternal Health Interventions: Uncertainty Estimation and Decision-focused Evaluation 

**Title (ZH)**: 基于LLM的代理模拟在孕产妇健康干预中的应用：不确定性估计与决策导向评估 

**Authors**: Sarah Martinson, Lingkai Kong, Cheol Woo Kim, Aparna Taneja, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2503.22719)  

**Abstract**: Agent-based simulation is crucial for modeling complex human behavior, yet traditional approaches require extensive domain knowledge and large datasets. In data-scarce healthcare settings where historic and counterfactual data are limited, large language models (LLMs) offer a promising alternative by leveraging broad world knowledge. This study examines an LLM-driven simulation of a maternal mobile health program, predicting beneficiaries' listening behavior when they receive health information via automated messages (control) or live representatives (intervention). Since uncertainty quantification is critical for decision-making in health interventions, we propose an LLM epistemic uncertainty estimation method based on binary entropy across multiple samples. We enhance model robustness through ensemble approaches, improving F1 score and model calibration compared to individual models. Beyond direct evaluation, we take a decision-focused approach, demonstrating how LLM predictions inform intervention feasibility and trial implementation in data-limited settings. The proposed method extends to public health, disaster response, and other domains requiring rapid intervention assessment under severe data constraints. All code and prompts used for this work can be found at this https URL. 

**Abstract (ZH)**: 基于代理的仿真是模拟复杂人类行为的关键，但传统方法需要广泛的领域知识和大量数据集。在历史和反事实数据有限的数据稀缺医疗环境中，大型语言模型（LLMs）通过利用广泛的背景知识提供了一种有希望的替代方案。本研究探讨了由LLM驱动的母婴移动健康项目仿真，预测受试者在接受自动化消息（对照组）或现场代表（干预组）健康信息时的倾听行为。由于不确定性量化对健康干预决策至关重要，我们提出了一种基于多样本二进制熵的LLM认知不确定性估计方法。通过集成方法增强模型的鲁棒性，与单个模型相比，提高了F1分数和模型校准。除了直接评估，我们采用以决策为导向的方法，展示了LLM预测如何在数据有限的环境中指导干预可行性和试验实施。提出的方法扩展到公共卫生、灾害响应及其他需要在严重数据限制下快速评估干预措施的领域。所有为此工作使用的代码和提示均可在此网址访问。 

---
# CodeScientist: End-to-End Semi-Automated Scientific Discovery with Code-based Experimentation 

**Title (ZH)**: CodeScientist：基于代码实验的端到端半自动化科学发现 

**Authors**: Peter Jansen, Oyvind Tafjord, Marissa Radensky, Pao Siangliulue, Tom Hope, Bhavana Dalvi Mishra, Bodhisattwa Prasad Majumder, Daniel S. Weld, Peter Clark  

**Link**: [PDF](https://arxiv.org/pdf/2503.22708)  

**Abstract**: Despite the surge of interest in autonomous scientific discovery (ASD) of software artifacts (e.g., improved ML algorithms), current ASD systems face two key limitations: (1) they largely explore variants of existing codebases or similarly constrained design spaces, and (2) they produce large volumes of research artifacts (such as automatically generated papers and code) that are typically evaluated using conference-style paper review with limited evaluation of code. In this work we introduce CodeScientist, a novel ASD system that frames ideation and experiment construction as a form of genetic search jointly over combinations of research articles and codeblocks defining common actions in a domain (like prompting a language model). We use this paradigm to conduct hundreds of automated experiments on machine-generated ideas broadly in the domain of agents and virtual environments, with the system returning 19 discoveries, 6 of which were judged as being both at least minimally sound and incrementally novel after a multi-faceted evaluation beyond that typically conducted in prior work, including external (conference-style) review, code review, and replication attempts. Moreover, the discoveries span new tasks, agents, metrics, and data, suggesting a qualitative shift from benchmark optimization to broader discoveries. 

**Abstract (ZH)**: 尽管对自主科学发现（ASD）软件构件的兴趣激增（例如，改进的ML算法），当前的ASD系统面临两个关键限制：（1）它们主要探索现有代码库的变体或类似约束的设计空间；（2）它们生成大量研究构件（如自动生成的论文和代码），这些构件通常使用会议风格的论文评审方式进行评估，代码的评估则更为有限。在此项工作中，我们引入了CodeScientist，这是一种新颖的ASD系统，将构想和实验构建视为在研究文章和定义域中常见操作的代码块组合上的形式基因搜索。我们使用这一范式对广泛涉及代理和虚拟环境领域的机器生成构想进行数百次自动化实验，系统返回了19项发现，其中6项在多方面评估（包括外部会议评审、代码评审和复制尝试）之后被判定为至少具有最小的合理性和逐步新颖性。此外，这些发现涵盖了新的任务、代理、指标和数据，暗示着从基准优化到更广泛的发现的定性转变。 

---
# UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving 

**Title (ZH)**: UniOcc：自动驾驶中 occupancy 预测与估计统一基准 

**Authors**: Yuping Wang, Xiangyu Huang, Xiaokang Sun, Mingxuan Yan, Shuo Xing, Zhengzhong Tu, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.24381)  

**Abstract**: We introduce UniOcc, a comprehensive, unified benchmark for occupancy forecasting (i.e., predicting future occupancies based on historical information) and current-frame occupancy prediction from camera images. UniOcc unifies data from multiple real-world datasets (i.e., nuScenes, Waymo) and high-fidelity driving simulators (i.e., CARLA, OpenCOOD), which provides 2D/3D occupancy labels with per-voxel flow annotations and support for cooperative autonomous driving. In terms of evaluation, unlike existing studies that rely on suboptimal pseudo labels for evaluation, UniOcc incorporates novel metrics that do not depend on ground-truth occupancy, enabling robust assessment of additional aspects of occupancy quality. Through extensive experiments on state-of-the-art models, we demonstrate that large-scale, diverse training data and explicit flow information significantly enhance occupancy prediction and forecasting performance. 

**Abstract (ZH)**: UniOcc：一种综合统一的 occupancy 预测基准（包括基于历史信息的未来occupancy预测和当前帧occupancy预测）及其在相机图像中的应用 

---
# Any2Caption:Interpreting Any Condition to Caption for Controllable Video Generation 

**Title (ZH)**: Any2Caption：任意条件到字幕的解释以实现可控视频生成 

**Authors**: Shengqiong Wu, Weicai Ye, Jiahao Wang, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Shuicheng Yan, Hao Fei, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.24379)  

**Abstract**: To address the bottleneck of accurate user intent interpretation within the current video generation community, we present Any2Caption, a novel framework for controllable video generation under any condition. The key idea is to decouple various condition interpretation steps from the video synthesis step. By leveraging modern multimodal large language models (MLLMs), Any2Caption interprets diverse inputs--text, images, videos, and specialized cues such as region, motion, and camera poses--into dense, structured captions that offer backbone video generators with better guidance. We also introduce Any2CapIns, a large-scale dataset with 337K instances and 407K conditions for any-condition-to-caption instruction tuning. Comprehensive evaluations demonstrate significant improvements of our system in controllability and video quality across various aspects of existing video generation models. Project Page: this https URL 

**Abstract (ZH)**: 为了应对当前视频生成社区中准确用户意图解释的瓶颈，我们提出了Any2Caption，这是一种在任何条件下可控视频生成的新框架。其关键思想是将各种条件解释步骤与视频合成步骤解耦。通过利用现代多模态大型语言模型（MLLMs），Any2Caption 将多样化的输入——文本、图像、视频以及区域、运动和相机姿态等专业提示——解释为密集且结构化的字幕，为骨干视频生成器提供更好的指导。我们还介绍了包含337,000个实例和407,000种条件的大规模数据集Any2CapIns，用于任何条件到字幕指令调优。全面的评估证明，我们的系统在多种现有视频生成模型方面显著提高了可控性和视频质量。项目页面：this https URL。 

---
# Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for Large Language Models 

**Title (ZH)**: 挖掘推理经济效益：大规模语言模型高效推理综述 

**Authors**: Rui Wang, Hongru Wang, Boyang Xue, Jianhui Pang, Shudong Liu, Yi Chen, Jiahao Qiu, Derek Fai Wong, Heng Ji, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2503.24377)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly enhanced their ability to perform complex reasoning tasks, transitioning from fast and intuitive thinking (System 1) to slow and deep reasoning (System 2). While System 2 reasoning improves task accuracy, it often incurs substantial computational costs due to its slow thinking nature and inefficient or unnecessary reasoning behaviors. In contrast, System 1 reasoning is computationally efficient but leads to suboptimal performance. Consequently, it is critical to balance the trade-off between performance (benefits) and computational costs (budgets), giving rise to the concept of reasoning economy. In this survey, we provide a comprehensive analysis of reasoning economy in both the post-training and test-time inference stages of LLMs, encompassing i) the cause of reasoning inefficiency, ii) behavior analysis of different reasoning patterns, and iii) potential solutions to achieve reasoning economy. By offering actionable insights and highlighting open challenges, we aim to shed light on strategies for improving the reasoning economy of LLMs, thereby serving as a valuable resource for advancing research in this evolving area. We also provide a public repository to continually track developments in this fast-evolving field. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）的最新进展显著增强了其执行复杂推理任务的能力，从快速直观思考（System 1）过渡到慢速深入推理（System 2）。虽然System 2推理可以提高任务准确性，但由于其慢速思考的性质和低效或不必要的推理行为，往往会带来巨大的计算成本。相比之下，System 1推理虽然计算高效，但会导致性能不理想。因此，在性能（效益）和计算成本（预算）之间取得平衡至关重要，从而引出了推理经济学的概念。在这篇综述中，我们对大型语言模型训练后和测试时推理阶段的推理经济学进行了全面分析，涵盖i) 推理低效的原因，ii) 不同推理模式的行为分析，和iii) 实现推理经济学的潜在解决方案。通过提供可操作的见解并突出显示开放挑战，我们旨在揭示提高大型语言模型推理经济学的策略，从而为推进这一不断发展的领域的研究提供有价值资源。我们还提供了一个公开的存储库以持续跟踪这一快速发展的领域的进展。 

---
# Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1 

**Title (ZH)**: 探究强化学习对视频理解的影响：SEED-Bench-R1的见解 

**Authors**: Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Lu Qiu, Ying Shan, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24376)  

**Abstract**: Recent advancements in Chain of Thought (COT) generation have significantly improved the reasoning capabilities of Large Language Models (LLMs), with reinforcement learning (RL) emerging as an effective post-training approach. Multimodal Large Language Models (MLLMs) inherit this reasoning potential but remain underexplored in tasks requiring both perception and logical reasoning. To address this, we introduce SEED-Bench-R1, a benchmark designed to systematically evaluate post-training methods for MLLMs in video understanding. It includes intricate real-world videos and complex everyday planning tasks in the format of multiple-choice questions, requiring sophisticated perception and reasoning. SEED-Bench-R1 assesses generalization through a three-level hierarchy: in-distribution, cross-environment, and cross-environment-task scenarios, equipped with a large-scale training dataset with easily verifiable ground-truth answers. Using Qwen2-VL-Instruct-7B as a base model, we compare RL with supervised fine-tuning (SFT), demonstrating RL's data efficiency and superior performance on both in-distribution and out-of-distribution tasks, even outperforming SFT on general video understanding benchmarks like LongVideoBench. Our detailed analysis reveals that RL enhances visual perception but often produces less logically coherent reasoning chains. We identify key limitations such as inconsistent reasoning and overlooked visual cues, and suggest future improvements in base model reasoning, reward modeling, and RL robustness against noisy signals. 

**Abstract (ZH)**: 近期在Chain of Thought (COT) 生成方面的进展显著提高了大型语言模型（LLMs）的推理能力，强化学习（RL）作为后训练方法被证明是有效的方法。多模态大型语言模型（MLLMs）继承了这种推理潜力，但在要求感知和逻辑推理结合的任务中尚未得到充分探索。为解决这一问题，我们提出SEED-Bench-R1基准，旨在系统性地评估MLLMs在视频理解中的后训练方法。该基准包含复杂的实际视频和复杂的日常规划任务，以多项选择题的形式呈现，需要复杂的感知和推理能力。SEED-Bench-R1通过三层层级结构评估泛化能力：同分布、跨环境和跨环境任务场景，并配备了大规模训练数据集和易于验证的答案。使用Qwen2-VL-Instruct-7B作为基础模型，我们比较了RL与监督微调（SFT）的效果， Demonstrating RL的数据效率和在同分布和异分布任务中的优越性能，甚至在长视频基准（LongVideoBench）等一般视频理解基准测试中优于SFT。我们的详细分析表明，RL增强了视觉感知，但往往产生了不那么逻辑连贯的推理链。我们识别了关键限制，包括推理不一致和忽略视觉线索，并建议未来在基础模型推理、奖励建模和RL对抗噪声信号的鲁棒性方面的改进。 

---
# Effectively Controlling Reasoning Models through Thinking Intervention 

**Title (ZH)**: 通过思考干预有效控制推理模型 

**Authors**: Tong Wu, Chong Xiang, Jiachen T. Wang, Prateek Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2503.24370)  

**Abstract**: Reasoning-enhanced large language models (LLMs) explicitly generate intermediate reasoning steps prior to generating final answers, helping the model excel in complex problem-solving. In this paper, we demonstrate that this emerging generation framework offers a unique opportunity for more fine-grained control over model behavior. We propose Thinking Intervention, a novel paradigm designed to explicitly guide the internal reasoning processes of LLMs by strategically inserting or revising specific thinking tokens. We conduct comprehensive evaluations across multiple tasks, including instruction following on IFEval, instruction hierarchy on SEP, and safety alignment on XSTest and SORRY-Bench. Our results demonstrate that Thinking Intervention significantly outperforms baseline prompting approaches, achieving up to 6.7% accuracy gains in instruction-following scenarios, 15.4% improvements in reasoning about instruction hierarchies, and a 40.0% increase in refusal rates for unsafe prompts using open-source DeepSeek R1 models. Overall, our work opens a promising new research avenue for controlling reasoning LLMs. 

**Abstract (ZH)**: 增强推理的大语言模型（LLMs）在生成最终答案之前显式地生成中间推理步骤，有助于模型在复杂问题解决方面表现出色。在本文中，我们展示了这种新兴的生成框架为更精细地控制模型行为提供了独特机会。我们提出了思考干预，这是一种新的范式，通过战略性地插入或修订特定的思考令牌来显式地引导LLMs的内部推理过程。我们在多个任务上进行了全面评估，包括IFEval上的指令跟随、SEP上的指令层次结构以及XSTest和SORRY-Bench上的安全性对齐。我们的结果表明，思考干预显著优于基线提示方法，在指令跟随场景中实现了高达6.7%的准确率提升，在关于指令层次结构的推理中实现了15.4%的改进，并在使用开源DeepSeek R1模型处理不确定提示时拒绝率提高了40.0%。总体而言，我们的工作为控制推理LLMs打开了一个有希望的新研究方向。 

---
# Which LIME should I trust? Concepts, Challenges, and Solutions 

**Title (ZH)**: Which LIME Should I Trust? 概念、挑战与解决方案 

**Authors**: Patrick Knab, Sascha Marton, Udo Schlegel, Christian Bartelt  

**Link**: [PDF](https://arxiv.org/pdf/2503.24365)  

**Abstract**: As neural networks become dominant in essential systems, Explainable Artificial Intelligence (XAI) plays a crucial role in fostering trust and detecting potential misbehavior of opaque models. LIME (Local Interpretable Model-agnostic Explanations) is among the most prominent model-agnostic approaches, generating explanations by approximating the behavior of black-box models around specific instances. Despite its popularity, LIME faces challenges related to fidelity, stability, and applicability to domain-specific problems. Numerous adaptations and enhancements have been proposed to address these issues, but the growing number of developments can be overwhelming, complicating efforts to navigate LIME-related research. To the best of our knowledge, this is the first survey to comprehensively explore and collect LIME's foundational concepts and known limitations. We categorize and compare its various enhancements, offering a structured taxonomy based on intermediate steps and key issues. Our analysis provides a holistic overview of advancements in LIME, guiding future research and helping practitioners identify suitable approaches. Additionally, we provide a continuously updated interactive website (this https URL), offering a concise and accessible overview of the survey. 

**Abstract (ZH)**: 随着神经网络在关键系统中的主导地位不断提升，可解释的人工智能（XAI）在促进信任并检测不透明模型潜在不当行为方面发挥着重要作用。LIME（局部可解释模型无关解释）是其中最突出的模型无关方法之一，通过近似黑盒模型在特定实例周围的行為来生成解释。尽管LIME备受青睐，但它面临着精度、稳定性和对特定领域问题的应用性等方面的挑战。提出了诸多适应性和增强措施以应对这些问题，但不断增长的发展数量也可能令人感到困惑，增加了导航LIME相关研究的难度。据我们所知，这是首次对LIME的基础概念及其已知局限性进行全面探索和收集的综述。我们对各种增强措施进行了分类和比较，基于中间步骤和关键问题提出了结构化的分类体系。我们的分析提供了LIME进展的全景概述，指导未来研究并帮助实践者识别合适的方案。此外，我们提供了一个不断更新的交互式网站（这个 https URL），提供综述的简洁和易访问概览。 

---
# Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation 

**Title (ZH)**: 基于视觉的机器人 manipulation: Sim-and-Real Co-Training 的简单配方 

**Authors**: Abhiram Maddukuri, Zhenyu Jiang, Lawrence Yunliang Chen, Soroush Nasiriany, Yuqi Xie, Yu Fang, Wenqi Huang, Zu Wang, Zhenjia Xu, Nikita Chernyadev, Scott Reed, Ken Goldberg, Ajay Mandlekar, Linxi Fan, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24361)  

**Abstract**: Large real-world robot datasets hold great potential to train generalist robot models, but scaling real-world human data collection is time-consuming and resource-intensive. Simulation has great potential in supplementing large-scale data, especially with recent advances in generative AI and automated data generation tools that enable scalable creation of robot behavior datasets. However, training a policy solely in simulation and transferring it to the real world often demands substantial human effort to bridge the reality gap. A compelling alternative is to co-train the policy on a mixture of simulation and real-world datasets. Preliminary studies have recently shown this strategy to substantially improve the performance of a policy over one trained on a limited amount of real-world data. Nonetheless, the community lacks a systematic understanding of sim-and-real co-training and what it takes to reap the benefits of simulation data for real-robot learning. This work presents a simple yet effective recipe for utilizing simulation data to solve vision-based robotic manipulation tasks. We derive this recipe from comprehensive experiments that validate the co-training strategy on various simulation and real-world datasets. Using two domains--a robot arm and a humanoid--across diverse tasks, we demonstrate that simulation data can enhance real-world task performance by an average of 38%, even with notable differences between the simulation and real-world data. Videos and additional results can be found at this https URL 

**Abstract (ZH)**: 利用模拟数据与现实数据联合训练解决基于视觉的机器人 manipulation 任务的简单有效方法 

---
# SQuat: Subspace-orthogonal KV Cache Quantization 

**Title (ZH)**: SQuat: 子空间正交键值缓存量化 

**Authors**: Hao Wang, Ligong Han, Kai Xu, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2503.24358)  

**Abstract**: The key-value (KV) cache accelerates LLMs decoding by storing KV tensors from previously generated tokens. It reduces redundant computation at the cost of increased memory usage. To mitigate this overhead, existing approaches compress KV tensors into lower-bit representations; however, quantization errors can accumulate as more tokens are generated, potentially resulting in undesired outputs. In this paper, we introduce SQuat (Subspace-orthogonal KV cache quantization). It first constructs a subspace spanned by query tensors to capture the most critical task-related information. During key tensor quantization, it enforces that the difference between the (de)quantized and original keys remains orthogonal to this subspace, minimizing the impact of quantization errors on the attention mechanism's outputs. SQuat requires no model fine-tuning, no additional calibration dataset for offline learning, and is grounded in a theoretical framework we develop. Through numerical experiments, we show that our method reduces peak memory by 2.17 to 2.82, improves throughput by 2.45 to 3.60, and achieves more favorable benchmark scores than existing KV cache quantization algorithms. 

**Abstract (ZH)**: 基于子空间正交性的键值缓存量化（SQuat） 

---
# ORAL: Prompting Your Large-Scale LoRAs via Conditional Recurrent Diffusion 

**Title (ZH)**: ORAL: 通过条件递归扩散提示您的大规模LoRAs 

**Authors**: Rana Muhammad Shahroz Khan, Dongwen Tang, Pingzhi Li, Kai Wang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.24354)  

**Abstract**: Parameter generation has emerged as a novel paradigm for neural network development, offering an alternative to traditional neural network training by synthesizing high-quality model weights directly. In the context of Low-Rank Adaptation (LoRA) for evolving ($\textit{i.e.}$, constantly updated) large language models (LLMs), this approach promises efficient adaptation without costly retraining. However, existing methods face critical limitations in simultaneously achieving scalability and controllability. In this paper, we introduce $\texttt{ORAL}$, a novel $\textbf{conditional recurrent diffusion}$ framework that addresses these challenges. $\texttt{ORAL}$ incorporates a novel conditioning mechanism that integrates model architecture and textual task specifications, enabling the generation of task-specific LoRA parameters that can seamlessly transfer across evolving foundation models. Our approach successfully scales to billions-of-parameter LLMs and maintains controllability. Through extensive experiments across seven language tasks, four vision tasks, and three multimodal tasks using five pre-trained LLMs, we demonstrate that $\texttt{ORAL}$ generates high-quality LoRA parameters that achieve comparable or superior performance to vanilla trained counterparts. 

**Abstract (ZH)**: 基于条件递归扩散的Low-Rank Adaptation ($\textit{i.e.}$, 持续更新的大语言模型)参数生成方法 

---
# Pro-Routing: Proactive Routing of Autonomous Multi-Capacity Robots for Pickup-and-Delivery Tasks 

**Title (ZH)**: 促进路由：自主多容量机器人前瞻性的拣取和配送任务路由算法 

**Authors**: Daniel Garces, Stephanie Gil  

**Link**: [PDF](https://arxiv.org/pdf/2503.24325)  

**Abstract**: We consider a multi-robot setting, where we have a fleet of multi-capacity autonomous robots that must service spatially distributed pickup-and-delivery requests with fixed maximum wait times. Requests can be either scheduled ahead of time or they can enter the system in real-time. In this setting, stability for a routing policy is defined as the cost of the policy being uniformly bounded over time. Most previous work either solve the problem offline to theoretically maintain stability or they consider dynamically arriving requests at the expense of the theoretical guarantees on stability. In this paper, we aim to bridge this gap by proposing a novel proactive rollout-based routing framework that adapts to real-time demand while still provably maintaining the stability of the learned routing policy. We derive provable stability guarantees for our method by proposing a fleet sizing algorithm that obtains a sufficiently large fleet that ensures stability by construction. To validate our theoretical results, we consider a case study on real ride requests for Harvard's evening Van System. We also evaluate the performance of our framework using the currently deployed smaller fleet size. In this smaller setup, we compare against the currently deployed routing algorithm, greedy heuristics, and Monte-Carlo-Tree-Search-based algorithms. Our empirical results show that our framework maintains stability when we use the sufficiently large fleet size found in our theoretical results. For the smaller currently deployed fleet size, our method services 6% more requests than the closest baseline while reducing median passenger wait times by 33%. 

**Abstract (ZH)**: 我们考虑一个多机器人环境，其中有一支多容量自主机器人队列，需要服务具有固定最大等待时间的空间分布式取送请求。请求既可以提前调度，也可以实时进入系统。在这种环境中，路由策略的稳定性定义为策略的成本在时间上均匀有界。大多数先前的工作要么在离线状态下求解问题以理论上保持稳定性，要么考虑动态到达的请求，但会牺牲理论上关于稳定性的保证。在本文中，我们通过提出一种新型的前瞻性展开为基础的路由框架来弥补这一差距，该框架能够适应实时需求，同时证明能够维持学习到的路由策略的稳定性。我们通过提出一种车队规模算法来推导我们的方法的可验证稳定性保证，该算法通过构造获得足够大的车队以确保稳定性。为了验证我们的理论结果，我们在一个实际案例研究中考虑哈佛大学夜间车系统的真实乘车请求。我们也使用当前部署的较小车队规模评估我们框架的性能。在较小的配置中，我们将方法与当前部署的路由算法、贪婪启发式算法以及基于蒙特卡洛树搜索的算法进行对比。实验结果显示，当使用我们在理论结果中找到的足够大的车队规模时，我们的框架能够维持稳定性。对于当前部署的较小车队规模，我们的方法比最近的基线算法多服务6%的请求，同时将中位乘客等待时间减少了33%。 

---
# BEATS: Bias Evaluation and Assessment Test Suite for Large Language Models 

**Title (ZH)**: BEATS：大型语言模型偏差评估测试套件 

**Authors**: Alok Abhishek, Lisa Erickson, Tushar Bandopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2503.24310)  

**Abstract**: In this research, we introduce BEATS, a novel framework for evaluating Bias, Ethics, Fairness, and Factuality in Large Language Models (LLMs). Building upon the BEATS framework, we present a bias benchmark for LLMs that measure performance across 29 distinct metrics. These metrics span a broad range of characteristics, including demographic, cognitive, and social biases, as well as measures of ethical reasoning, group fairness, and factuality related misinformation risk. These metrics enable a quantitative assessment of the extent to which LLM generated responses may perpetuate societal prejudices that reinforce or expand systemic inequities. To achieve a high score on this benchmark a LLM must show very equitable behavior in their responses, making it a rigorous standard for responsible AI evaluation. Empirical results based on data from our experiment show that, 37.65\% of outputs generated by industry leading models contained some form of bias, highlighting a substantial risk of using these models in critical decision making systems. BEATS framework and benchmark offer a scalable and statistically rigorous methodology to benchmark LLMs, diagnose factors driving biases, and develop mitigation strategies. With the BEATS framework, our goal is to help the development of more socially responsible and ethically aligned AI models. 

**Abstract (ZH)**: 本研究引入了BEATS框架，这是一种评估大型语言模型（LLMs）中的偏见、伦理、公平性和事实性的新颖框架。基于BEATS框架，我们提出了一种针对LLMs的偏见基准，该基准涵盖了29个不同的指标。这些指标涵盖了包括人口统计学、认知和社会偏见在内的广泛特征，以及伦理推理、群体公平性和与虚假信息相关的事实性测量。这些指标使得定量评估LLM生成的响应可能延续社会偏见，从而加剧系统不公平性成为可能。为了在该基准上获得高分，LLM必须在响应中展示出非常公平的行为，使其成为负责任AI评估的严格标准。基于我们实验数据的实证结果表明，37.65%的由行业领先模型生成的输出包含某种形式的偏见，这突显了在关键决策系统中使用这些模型的重大风险。BEATS框架和基准提供了一种可扩展且统计上严谨的方法来评估LLMs，诊断推动偏见的因素，并开发缓解策略。通过BEATS框架，我们的目标是帮助开发更具社会责任心和伦理对齐的AI模型。 

---
# A Systematic Evaluation of LLM Strategies for Mental Health Text Analysis: Fine-tuning vs. Prompt Engineering vs. RAG 

**Title (ZH)**: LLM策略系统评估：微调 vs. 提示工程 vs. RAG 对于心理健康文本分析 

**Authors**: Arshia Kermani, Veronica Perez-Rosas, Vangelis Metsis  

**Link**: [PDF](https://arxiv.org/pdf/2503.24307)  

**Abstract**: This study presents a systematic comparison of three approaches for the analysis of mental health text using large language models (LLMs): prompt engineering, retrieval augmented generation (RAG), and fine-tuning. Using LLaMA 3, we evaluate these approaches on emotion classification and mental health condition detection tasks across two datasets. Fine-tuning achieves the highest accuracy (91% for emotion classification, 80% for mental health conditions) but requires substantial computational resources and large training sets, while prompt engineering and RAG offer more flexible deployment with moderate performance (40-68% accuracy). Our findings provide practical insights for implementing LLM-based solutions in mental health applications, highlighting the trade-offs between accuracy, computational requirements, and deployment flexibility. 

**Abstract (ZH)**: 本研究系统比较了三种使用大规模语言模型（LLMs）分析心理健康文本的方法：提示工程、检索增强生成（RAG）和微调，并使用LLaMA 3在两个数据集中评估了这些方法在情绪分类和心理健康状况检测任务中的性能。微调在准确性方面最高（情绪分类91%，心理健康状况80%），但需要大量的计算资源和大规模的训练集，而提示工程和RAG提供了更具灵活性的部署方案，性能中等（情绪分类40%-68%的准确性）。本研究的结果为在心理健康应用中实施基于LLM的解决方案提供了实用见解，突显了准确性和计算需求与部署灵活性之间的权衡。 

---
# Evaluating machine learning models for predicting pesticides toxicity to honey bees 

**Title (ZH)**: 评估机器学习模型预测农药对蜜蜂毒性的能力 

**Authors**: Jakub Adamczyk, Jakub Poziemski, Pawel Siedlecki  

**Link**: [PDF](https://arxiv.org/pdf/2503.24305)  

**Abstract**: Small molecules play a critical role in the biomedical, environmental, and agrochemical domains, each with distinct physicochemical requirements and success criteria. Although biomedical research benefits from extensive datasets and established benchmarks, agrochemical data remain scarce, particularly with respect to species-specific toxicity. This work focuses on ApisTox, the most comprehensive dataset of experimentally validated chemical toxicity to the honey bee (\textit{Apis mellifera}), an ecologically vital pollinator. We evaluate ApisTox using a diverse suite of machine learning approaches, including molecular fingerprints, graph kernels, and graph neural networks, as well as pretrained models. Comparative analysis with medicinal datasets from the MoleculeNet benchmark reveals that ApisTox represents a distinct chemical space. Performance degradation on non-medicinal datasets, such as ApisTox, demonstrates their limited generalizability of current state-of-the-art algorithms trained solely on biomedical data. Our study highlights the need for more diverse datasets and for targeted model development geared toward the agrochemical domain. 

**Abstract (ZH)**: 小分子在生物医学、环境和农化领域中扮演着关键角色，各自具有独特的物理化学要求和成功标准。尽管生物医学研究得益于丰富的数据集和现有的基准，农化数据依然稀缺，尤其是在物种特异性毒性方面。本文专注于ApisTox，这是最全面的实验验证蜂蜜bee（Apis mellifera）化学毒性的数据集，蜂蜜bee是生态上重要的传粉者。我们使用多种机器学习方法，包括分子指纹、图核和图神经网络，以及预训练模型来评估ApisTox。与MoleculeNet基准中的医药数据集的对比分析表明，ApisTox代表了独特的化学空间。在非医药数据集上的性能下降表明，当前仅基于生物医学数据训练的先进算法的泛化能力有限。我们的研究强调了需要更多样化的数据集以及针对农化领域的目标模型开发的重要性。 

---
# Shape Expressions with Inheritance 

**Title (ZH)**: 继承关系中的形状表达 

**Authors**: Iovka Boneva, Jose Emilio Labra Gayo, Eric Prud'hommeaux, Katherine Thornton, Andra Waagmeester  

**Link**: [PDF](https://arxiv.org/pdf/2503.24299)  

**Abstract**: We formally introduce an inheritance mechanism for the Shape Expressions language (ShEx). It is inspired by inheritance in object-oriented programming languages, and provides similar advantages such as reuse, modularity, and more flexible data modelling. Using an example, we explain the main features of the inheritance mechanism. We present its syntax and formal semantics. The semantics is an extension of the semantics of ShEx 2.1. It also directly yields a validation algorithm as an extension of the previous ShEx validation algorithms, while maintaining the same algorithmic complexity. 

**Abstract (ZH)**: 我们正式引入了Shape Expressions语言（ShEx）的继承机制。该机制受到面向对象编程语言中继承的启发，提供了类似的优势，如重用、模块化和更灵活的数据建模。通过一个示例，我们解释了继承机制的主要特性。我们展示了其语法和形式语义。该语义是ShEx 2.1语义的扩展，同时还直接提供了一个验证算法的扩展，保持了相同的算法复杂度。 

---
# Value of Information-based Deceptive Path Planning Under Adversarial Interventions 

**Title (ZH)**: 基于价值信息的欺骗性路径规划在对抗干预下的价值 

**Authors**: Wesley A. Suttle, Jesse Milzman, Mustafa O. Karabag, Brian M. Sadler, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24284)  

**Abstract**: Existing methods for deceptive path planning (DPP) address the problem of designing paths that conceal their true goal from a passive, external observer. Such methods do not apply to problems where the observer has the ability to perform adversarial interventions to impede the path planning agent. In this paper, we propose a novel Markov decision process (MDP)-based model for the DPP problem under adversarial interventions and develop new value of information (VoI) objectives to guide the design of DPP policies. Using the VoI objectives we propose, path planning agents deceive the adversarial observer into choosing suboptimal interventions by selecting trajectories that are of low informational value to the observer. Leveraging connections to the linear programming theory for MDPs, we derive computationally efficient solution methods for synthesizing policies for performing DPP under adversarial interventions. In our experiments, we illustrate the effectiveness of the proposed solution method in achieving deceptiveness under adversarial interventions and demonstrate the superior performance of our approach to both existing DPP methods and conservative path planning approaches on illustrative gridworld problems. 

**Abstract (ZH)**: 现有的欺骗性路径规划方法解决了设计隐藏真正目标的路径以避开被动外部观察者的问题。这些方法不适用于观察者能够采取对抗性干预以阻碍路径规划代理的情况。本文提出了一种在对抗性干预下基于马尔可夫决策过程（MDP）的新颖模型，并开发了新的信息价值（VoI）目标以指导欺骗性路径规划（DPP）策略的设计。利用MDP的线性规划理论，我们推导出在对抗性干预下合成DPP策略的计算效率高的解法。在我们的实验中，我们展示了所提出的方法在对抗性干预下实现欺骗性的有效性，并在示例网格世界问题上证明了我们的方法优于现有的DPP方法和保守的路径规划方法。 

---
# AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World 

**Title (ZH)**: AutoEval：自主评估通用机器人操作政策在真实世界中的表现 

**Authors**: Zhiyuan Zhou, Pranav Atreya, You Liang Tan, Karl Pertsch, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2503.24278)  

**Abstract**: Scalable and reproducible policy evaluation has been a long-standing challenge in robot learning. Evaluations are critical to assess progress and build better policies, but evaluation in the real world, especially at a scale that would provide statistically reliable results, is costly in terms of human time and hard to obtain. Evaluation of increasingly generalist robot policies requires an increasingly diverse repertoire of evaluation environments, making the evaluation bottleneck even more pronounced. To make real-world evaluation of robotic policies more practical, we propose AutoEval, a system to autonomously evaluate generalist robot policies around the clock with minimal human intervention. Users interact with AutoEval by submitting evaluation jobs to the AutoEval queue, much like how software jobs are submitted with a cluster scheduling system, and AutoEval will schedule the policies for evaluation within a framework supplying automatic success detection and automatic scene resets. We show that AutoEval can nearly fully eliminate human involvement in the evaluation process, permitting around the clock evaluations, and the evaluation results correspond closely to ground truth evaluations conducted by hand. To facilitate the evaluation of generalist policies in the robotics community, we provide public access to multiple AutoEval scenes in the popular BridgeData robot setup with WidowX robot arms. In the future, we hope that AutoEval scenes can be set up across institutions to form a diverse and distributed evaluation network. 

**Abstract (ZH)**: 可扩展且可重现的政策评估一直是机器人学习中的长期挑战。为了使机器人政策的实地评估更加实用，我们提出了AutoEval系统，该系统能够在最少人工干预的情况下，全天候自主评估通用机器人政策。通过向AutoEval队列提交评估作业，用户可以像使用集群调度系统提交软件作业一样，让AutoEval框架负责自动成功检测和自动场景重置，以安排策略进行评估。我们展示了AutoEval几乎可以完全消除评估过程中的手动干预，实现全天候评估，且评估结果与手工进行的真实评估结果高度一致。为了促进机器人社区内通用策略的评估，我们提供了在流行BridgeData机器人设置中使用WidowX机器人手臂的多个AutoEval场景的公共访问权限。未来，我们希望可以在不同机构之间设置AutoEval场景，形成一个多样化且分布式的评估网络。 

---
# Evaluating and Designing Sparse Autoencoders by Approximating Quasi-Orthogonality 

**Title (ZH)**: 评估和设计稀疏自编码器：近似准正交性方法 

**Authors**: Sewoong Lee, Adam Davies, Marc E. Canby, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2503.24277)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as a workhorse of modern mechanistic interpretability, but leading SAE approaches with top-$k$ style activation functions lack theoretical grounding for selecting the hyperparameter $k$. SAEs are based on the linear representation hypothesis (LRH), which assumes that the representations of large language models (LLMs) are linearly encoded, and the superposition hypothesis (SH), which states that there can be more features in the model than its dimensionality. We show that, based on the formal definitions of the LRH and SH, the magnitude of sparse feature vectors (the latent representations learned by SAEs of the dense embeddings of LLMs) can be approximated using their corresponding dense vector with a closed-form error bound. To visualize this, we propose the ZF plot, which reveals a previously unknown relationship between LLM hidden embeddings and SAE feature vectors, allowing us to make the first empirical measurement of the extent to which feature vectors of pre-trained SAEs are over- or under-activated for a given input. Correspondingly, we introduce Approximate Feature Activation (AFA), which approximates the magnitude of the ground-truth sparse feature vector, and propose a new evaluation metric derived from AFA to assess the alignment between inputs and activations. We also leverage AFA to introduce a novel SAE architecture, the top-AFA SAE, leading to SAEs that: (a) are more in line with theoretical justifications; and (b) obviate the need to tune SAE sparsity hyperparameters. Finally, we empirically demonstrate that top-AFA SAEs achieve reconstruction loss comparable to that of state-of-the-art top-k SAEs, without requiring the hyperparameter $k$ to be tuned. Our code is available at: this https URL. 

**Abstract (ZH)**: 稀疏自编码器（SAEs）已成为现代机理可解释性的主力工具，但主流的SAE方法使用顶级-$k$样式激活函数，缺乏选择超参数$k$的理论依据。SAEs基于线性表示假设（LRH），该假设认为大规模语言模型（LLMs）的表示形式是线性编码的，以及超定假设（SH），该假设表明模型中的特征数可以多于其维度。我们基于LRH和SH的形式定义，展示了稀疏特征向量（由SAEs学习的大语言模型密集嵌入的隐式表示）的幅度可以用相应的密集向量进行近似，并带有封闭形式的误差界。为了可视化这一点，我们提出了ZF图，揭示了大规模语言模型隐藏嵌入和SAE特征向量之间的一种未知关系，允许我们首次对预训练SAE的特征向量在给定输入下的过度激活或欠激活程度进行实证测量。相应地，我们引入了近似特征激活（AFA），用于近似地面真值稀疏特征向量的幅度，并提出了一种新的评估指标，该指标源自AFA，用于评估输入和激活之间的对齐情况。我们还利用AFA引入了一种新的SAE架构——顶级-AFA SAE，使得SAE：（a）更加符合理论依据；（b）取消了调整SAE稀疏度超参数的需求。最后，我们实验证明，顶级-AFA SAE在重构损失方面与最先进的顶级-$k$ SAE相当，无需调整超参数$k$。代码可在以下链接获取：this https URL。 

---
# Visual Acoustic Fields 

**Title (ZH)**: 视觉声场 

**Authors**: Yuelei Li, Hyunjin Kim, Fangneng Zhan, Ri-Zhao Qiu, Mazeyu Ji, Xiaojun Shan, Xueyan Zou, Paul Liang, Hanspeter Pfister, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.24270)  

**Abstract**: Objects produce different sounds when hit, and humans can intuitively infer how an object might sound based on its appearance and material properties. Inspired by this intuition, we propose Visual Acoustic Fields, a framework that bridges hitting sounds and visual signals within a 3D space using 3D Gaussian Splatting (3DGS). Our approach features two key modules: sound generation and sound localization. The sound generation module leverages a conditional diffusion model, which takes multiscale features rendered from a feature-augmented 3DGS to generate realistic hitting sounds. Meanwhile, the sound localization module enables querying the 3D scene, represented by the feature-augmented 3DGS, to localize hitting positions based on the sound sources. To support this framework, we introduce a novel pipeline for collecting scene-level visual-sound sample pairs, achieving alignment between captured images, impact locations, and corresponding sounds. To the best of our knowledge, this is the first dataset to connect visual and acoustic signals in a 3D context. Extensive experiments on our dataset demonstrate the effectiveness of Visual Acoustic Fields in generating plausible impact sounds and accurately localizing impact sources. Our project page is at this https URL. 

**Abstract (ZH)**: 视觉声场：一种基于3D高斯散斑的框架将打击声音和视觉信号连接到3D空间中 

---
# New Statistical Framework for Extreme Error Probability in High-Stakes Domains for Reliable Machine Learning 

**Title (ZH)**: 高风险领域中可靠机器学习的极端误差概率新统计框架 

**Authors**: Umberto Michelucci, Francesca Venturini  

**Link**: [PDF](https://arxiv.org/pdf/2503.24262)  

**Abstract**: Machine learning is vital in high-stakes domains, yet conventional validation methods rely on averaging metrics like mean squared error (MSE) or mean absolute error (MAE), which fail to quantify extreme errors. Worst-case prediction failures can have substantial consequences, but current frameworks lack statistical foundations for assessing their probability. In this work a new statistical framework, based on Extreme Value Theory (EVT), is presented that provides a rigorous approach to estimating worst-case failures. Applying EVT to synthetic and real-world datasets, this method is shown to enable robust estimation of catastrophic failure probabilities, overcoming the fundamental limitations of standard cross-validation. This work establishes EVT as a fundamental tool for assessing model reliability, ensuring safer AI deployment in new technologies where uncertainty quantification is central to decision-making or scientific analysis. 

**Abstract (ZH)**: 基于极值理论的机器学习模型最坏情况失败概率的统计框架：超越标准交叉验证的基本限制并确保新科技中AI部署的安全性 

---
# Beyond a Single Mode: GAN Ensembles for Diverse Medical Data Generation 

**Title (ZH)**: 超越单一模式：GAN集成用于生成多元医疗数据 

**Authors**: Lorenzo Tronchin, Tommy Löfstedt, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2503.24258)  

**Abstract**: The advancement of generative AI, particularly in medical imaging, confronts the trilemma of ensuring high fidelity, diversity, and efficiency in synthetic data generation. While Generative Adversarial Networks (GANs) have shown promise across various applications, they still face challenges like mode collapse and insufficient coverage of real data distributions. This work explores the use of GAN ensembles to overcome these limitations, specifically in the context of medical imaging. By solving a multi-objective optimisation problem that balances fidelity and diversity, we propose a method for selecting an optimal ensemble of GANs tailored for medical data. The selected ensemble is capable of generating diverse synthetic medical images that are representative of true data distributions and computationally efficient. Each model in the ensemble brings a unique contribution, ensuring minimal redundancy. We conducted a comprehensive evaluation using three distinct medical datasets, testing 22 different GAN architectures with various loss functions and regularisation techniques. By sampling models at different training epochs, we crafted 110 unique configurations. The results highlight the capability of GAN ensembles to enhance the quality and utility of synthetic medical images, thereby improving the efficacy of downstream tasks such as diagnostic modelling. 

**Abstract (ZH)**: 生成式AI的进步，特别是在医学成像领域，面临高保真度、多样性和效率之间的三难困境。虽然生成对抗网络（GANs）在各种应用中显示出了潜力，但仍面临模式坍塌和真实数据分布覆盖不足等挑战。本研究探讨了使用GAN集成来克服这些限制，特别是在医学成像领域。通过解决兼顾保真度和多样性的多目标优化问题，我们提出了一种方法，用于选择最适合医学数据的GAN集成。所选集成能够生成类似于真实数据分布的多样化合成医学图像，并具有计算效率。每个集成中的模型都贡献独特，确保了极小的冗余。我们使用三个不同的医学数据集进行了全面评估，测试了22种不同结构的GAN架构，以及多种损失函数和正则化技术。通过在不同训练周期采样模型，我们创建了110种独特的配置。结果表明，GAN集成能够提高合成医学图像的质量和实用性，从而提高下游任务（如诊断建模）的效率。 

---
# Spatio-temporal Prediction of Fine-Grained Origin-Destination Matrices with Applications in Ridesharing 

**Title (ZH)**: 精细粒度起源目的地矩阵的空间时间预测及其在拼车中的应用 

**Authors**: Run Yang, Runpeng Dai, Siran Gao, Xiaocheng Tang, Fan Zhou, Hongtu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24237)  

**Abstract**: Accurate spatial-temporal prediction of network-based travelers' requests is crucial for the effective policy design of ridesharing platforms. Having knowledge of the total demand between various locations in the upcoming time slots enables platforms to proactively prepare adequate supplies, thereby increasing the likelihood of fulfilling travelers' requests and redistributing idle drivers to areas with high potential demand to optimize the global supply-demand equilibrium. This paper delves into the prediction of Origin-Destination (OD) demands at a fine-grained spatial level, especially when confronted with an expansive set of local regions. While this task holds immense practical value, it remains relatively unexplored within the research community. To fill this gap, we introduce a novel prediction model called OD-CED, which comprises an unsupervised space coarsening technique to alleviate data sparsity and an encoder-decoder architecture to capture both semantic and geographic dependencies. Through practical experimentation, OD-CED has demonstrated remarkable results. It achieved an impressive reduction of up to 45% reduction in root-mean-square error and 60% in weighted mean absolute percentage error over traditional statistical methods when dealing with OD matrices exhibiting a sparsity exceeding 90%. 

**Abstract (ZH)**: 基于网络的出行请求的时空预测对于rideshares平台有效政策设计至关重要。了解未来时间槽中各区域间的总需求量能使平台提前准备充足的供应，从而提高满足出行请求的可能性，并将闲置司机重新分配到潜在需求高的区域，以优化全局的供需平衡。本文专注于在细粒度空间级别预测出行生成地-目的地（OD）需求，特别是在面对广泛的本地区域集合时。尽管这一任务具有巨大的实际价值，但在学术界仍鲜有研究。为填补这一空白，我们提出了一个名为OD-CED的新预测模型，该模型结合了无监督的空间粗糙化技术来缓解数据稀疏问题，并采用编码器-解码器架构来捕捉语义和地理依赖关系。通过实际实验，OD-CED展现出了显著的效果。在处理OD矩阵稀疏度超过90%的情况下，它在根均方误差和加权平均绝对百分比误差上分别实现了高达45%和60%的降低，超过了传统统计方法。 

---
# What, How, Where, and How Well? A Survey on Test-Time Scaling in Large Language Models 

**Title (ZH)**: 什么是、如何进行、在哪里以及进行得如何？大规模语言模型测试时缩放综述 

**Authors**: Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Zhihan Guo, Yufei Wang, Irwin King, Xue Liu, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.24235)  

**Abstract**: As enthusiasm for scaling computation (data and parameters) in the pretraining era gradually diminished, test-time scaling (TTS), also referred to as ``test-time computing'' has emerged as a prominent research focus. Recent studies demonstrate that TTS can further elicit the problem-solving capabilities of large language models (LLMs), enabling significant breakthroughs not only in specialized reasoning tasks, such as mathematics and coding, but also in general tasks like open-ended Q&A. However, despite the explosion of recent efforts in this area, there remains an urgent need for a comprehensive survey offering a systemic understanding. To fill this gap, we propose a unified, multidimensional framework structured along four core dimensions of TTS research: what to scale, how to scale, where to scale, and how well to scale. Building upon this taxonomy, we conduct an extensive review of methods, application scenarios, and assessment aspects, and present an organized decomposition that highlights the unique functional roles of individual techniques within the broader TTS landscape. From this analysis, we distill the major developmental trajectories of TTS to date and offer hands-on guidelines for practical deployment. Furthermore, we identify several open challenges and offer insights into promising future directions, including further scaling, clarifying the functional essence of techniques, generalizing to more tasks, and more attributions. 

**Abstract (ZH)**: 随着预训练时代对扩大计算（数据和参数）的热情逐渐减弱，“测试时缩放”（TTS），也称为“测试时计算”已 emerges as a prominent research focus. 近期研究展示了 TTS 能进一步激发大型语言模型（LLMs）的问题解决能力，不仅在数学和编码等专门推理任务中取得了重大突破，还在开放式问答等通用任务中取得了显著成果。然而，尽管该领域的努力有了爆炸式的增长，仍迫切需要一个全面的综述来提供系统性的理解。为填补这一空白，我们提出一个统一的多维度框架，沿着四个核心维度组织 TTS 研究：什么进行缩放、如何进行缩放、在哪里进行缩放以及缩放效果如何。基于这一分类，我们对方法、应用场景和评估方面进行了广泛回顾，并呈现了一个有组织的分解，突出了个体技术在更广泛 TTS 地形中的独特功能角色。通过这一分析，我们提炼了 TTS 到目前为止的主要发展轨迹，并提供了实践部署的手册指南。此外，我们确定了一些开放性的挑战，并提出了有希望的未来方向，包括进一步缩放、澄清技术的功能本质、泛化到更多任务以及更多归因。 

---
# MB-ORES: A Multi-Branch Object Reasoner for Visual Grounding in Remote Sensing 

**Title (ZH)**: MB-ORES: 一种多分支物体推理器用于遥感中的视觉grounding 

**Authors**: Karim Radouane, Hanane Azzag, Mustapha lebbah  

**Link**: [PDF](https://arxiv.org/pdf/2503.24219)  

**Abstract**: We propose a unified framework that integrates object detection (OD) and visual grounding (VG) for remote sensing (RS) imagery. To support conventional OD and establish an intuitive prior for VG task, we fine-tune an open-set object detector using referring expression data, framing it as a partially supervised OD task. In the first stage, we construct a graph representation of each image, comprising object queries, class embeddings, and proposal locations. Then, our task-aware architecture processes this graph to perform the VG task. The model consists of: (i) a multi-branch network that integrates spatial, visual, and categorical features to generate task-aware proposals, and (ii) an object reasoning network that assigns probabilities across proposals, followed by a soft selection mechanism for final referring object localization. Our model demonstrates superior performance on the OPT-RSVG and DIOR-RSVG datasets, achieving significant improvements over state-of-the-art methods while retaining classical OD capabilities. The code will be available in our repository: \url{this https URL}. 

**Abstract (ZH)**: 我们提出了一种统一框架，将目标检测（OD）和视觉定位（VG）集成到遥感（RS）图像中。为了支持传统的OD并为VG任务建立直观的先验知识，我们使用指示短语数据微调一个开放集目标检测器，将其视为半监督的目标检测任务。在第一阶段，我们为每张图像构建了一个图表示，包括对象查询、类别嵌入和建议位置。然后，我们的任务感知架构处理该图以执行VG任务。该模型由以下两部分组成：(i) 一个多分支网络，结合空间、视觉和类别特征生成任务感知的建议框；(ii) 一个对象推理网络，将概率分配给建议框，随后是软选择机制以实现最终的参照对象定位。我们的模型在OPT-RSVG和DIOR-RSVG数据集上展现出优越的性能，实现了与最先进的方法相比的显著改进，同时保留了传统的OD能力。代码将存放在我们的仓库中：\url{this https URL}。 

---
# DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting 

**Title (ZH)**: DiET-GS: 扩散先验和事件流辅助运动去模糊3D高斯点绘制 

**Authors**: Seungjun Lee, Gim Hee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.24210)  

**Abstract**: Reconstructing sharp 3D representations from blurry multi-view images are long-standing problem in computer vision. Recent works attempt to enhance high-quality novel view synthesis from the motion blur by leveraging event-based cameras, benefiting from high dynamic range and microsecond temporal resolution. However, they often reach sub-optimal visual quality in either restoring inaccurate color or losing fine-grained details. In this paper, we present DiET-GS, a diffusion prior and event stream-assisted motion deblurring 3DGS. Our framework effectively leverages both blur-free event streams and diffusion prior in a two-stage training strategy. Specifically, we introduce the novel framework to constraint 3DGS with event double integral, achieving both accurate color and well-defined details. Additionally, we propose a simple technique to leverage diffusion prior to further enhance the edge details. Qualitative and quantitative results on both synthetic and real-world data demonstrate that our DiET-GS is capable of producing significantly better quality of novel views compared to the existing baselines. Our project page is this https URL 

**Abstract (ZH)**: 从模糊多视角图像重建清晰的3D表示是计算机视觉中的长期问题。 recent works attempt to enhance high-quality novel view synthesis from motion blur by leveraging event-based cameras, benefiting from high dynamic range and microsecond temporal resolution. However, they often reach sub-optimal visual quality in either restoring inaccurate color or losing fine-grained details. In this paper, we present DiET-GS, a diffusion prior and event stream-assisted motion deblurring 3DGS. Our framework effectively leverages both blur-free event streams and diffusion prior in a two-stage training strategy. Specifically, we introduce the novel framework to constraint 3DGS with event double integral, achieving both accurate color and well-defined details. Additionally, we propose a simple technique to leverage diffusion prior to further enhance the edge details. Qualitative and quantitative results on both synthetic and real-world data demonstrate that our DiET-GS is capable of producing significantly better quality of novel views compared to the existing baselines. Our project page is this https URL。 

---
# Output Constraints as Attack Surface: Exploiting Structured Generation to Bypass LLM Safety Mechanisms 

**Title (ZH)**: 输出约束作为攻击面：利用结构化生成绕过LLM安全机制 

**Authors**: Shuoming Zhang, Jiacheng Zhao, Ruiyuan Xu, Xiaobing Feng, Huimin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.24191)  

**Abstract**: Content Warning: This paper may contain unsafe or harmful content generated by LLMs that may be offensive to readers. Large Language Models (LLMs) are extensively used as tooling platforms through structured output APIs to ensure syntax compliance so that robust integration with existing softwares like agent systems, could be achieved. However, the feature enabling functionality of grammar-guided structured output presents significant security vulnerabilities. In this work, we reveal a critical control-plane attack surface orthogonal to traditional data-plane vulnerabilities. We introduce Constrained Decoding Attack (CDA), a novel jailbreak class that weaponizes structured output constraints to bypass safety mechanisms. Unlike prior attacks focused on input prompts, CDA operates by embedding malicious intent in schema-level grammar rules (control-plane) while maintaining benign surface prompts (data-plane). We instantiate this with a proof-of-concept Chain Enum Attack, achieves 96.2% attack success rates across proprietary and open-weight LLMs on five safety benchmarks with a single query, including GPT-4o and Gemini-2.0-flash. Our findings identify a critical security blind spot in current LLM architectures and urge a paradigm shift in LLM safety to address control-plane vulnerabilities, as current mechanisms focused solely on data-plane threats leave critical systems exposed. 

**Abstract (ZH)**: 内容警告：本论文可能包含由大型语言模型生成的不安全或有害内容，可能令读者感到不适。大型语言模型通过结构化输出API广泛用作工具平台，以确保语法合规性，从而实现与现有软件（如代理系统）的稳健集成。然而，由语法指导的结构化输出功能使安全性面临重大威胁。在本工作中，我们揭示了一个与传统数据平面漏洞正交的关键控制平面攻击面。我们引入了受限解码攻击（CDA），这是一种新的 Jailbreak 类别，利用结构化输出约束绕过安全机制。与先前主要针对输入提示的攻击不同，CDA 通过在方案级语法规则中嵌入恶意意图（控制平面）来运作，同时保持看似 benign 的表面提示（数据平面）。我们通过一个概念证明的链枚举攻击实例化了这一点，该攻击在五个安全基准测试中实现了96.2%的攻击成功率，包括针对专有和开源大型语言模型（如GPT-4o和Gemini-2.0-flash）的单个查询。我们的研究结果揭示了当前大型语言模型架构中的关键安全盲点，并呼吁在大型语言模型安全性方面进行范式转变，以解决控制平面漏洞，因为当前机制仅专注于数据平面威胁，使关键系统面临风险。 

---
# Predicting Targeted Therapy Resistance in Non-Small Cell Lung Cancer Using Multimodal Machine Learning 

**Title (ZH)**: 使用多模态机器学习预测非小细胞肺癌的靶向治疗耐药性 

**Authors**: Peiying Hua, Andrea Olofson, Faraz Farhadi, Liesbeth Hondelink, Gregory Tsongalis, Konstantin Dragnev, Dagmar Hoegemann Savellano, Arief Suriawinata, Laura Tafe, Saeed Hassanpour  

**Link**: [PDF](https://arxiv.org/pdf/2503.24165)  

**Abstract**: Lung cancer is the primary cause of cancer death globally, with non-small cell lung cancer (NSCLC) emerging as its most prevalent subtype. Among NSCLC patients, approximately 32.3% have mutations in the epidermal growth factor receptor (EGFR) gene. Osimertinib, a third-generation EGFR-tyrosine kinase inhibitor (TKI), has demonstrated remarkable efficacy in the treatment of NSCLC patients with activating and T790M resistance EGFR mutations. Despite its established efficacy, drug resistance poses a significant challenge for patients to fully benefit from osimertinib. The absence of a standard tool to accurately predict TKI resistance, including that of osimertinib, remains a critical obstacle. To bridge this gap, in this study, we developed an interpretable multimodal machine learning model designed to predict patient resistance to osimertinib among late-stage NSCLC patients with activating EGFR mutations, achieving a c-index of 0.82 on a multi-institutional dataset. This machine learning model harnesses readily available data routinely collected during patient visits and medical assessments to facilitate precision lung cancer management and informed treatment decisions. By integrating various data types such as histology images, next generation sequencing (NGS) data, demographics data, and clinical records, our multimodal model can generate well-informed recommendations. Our experiment results also demonstrated the superior performance of the multimodal model over single modality models (c-index 0.82 compared with 0.75 and 0.77), thus underscoring the benefit of combining multiple modalities in patient outcome prediction. 

**Abstract (ZH)**: 非小细胞肺癌是最主要的癌症死亡原因，其中具有表皮生长因子受体（EGFR）基因突变的非小细胞肺癌（NSCLC）是最常见的亚型。在NSCLC患者中，约32.3%的患者具有EGFR基因突变。第三代EGFR酪氨酸激酶抑制剂奥希替尼在具有激活突变和T790M突变的NSCLC患者中表现出显著的治疗效果。尽管奥希替尼已经在临床上证明了其有效性，但药物耐药性仍然阻碍了患者充分利用该药物所带来的益处。缺乏标准工具准确预测酪氨酸激酶抑制剂（TKI）耐药性，包括奥希替尼，仍然是一个关键障碍。为了填补这一空白，本研究开发了一种可解释的多模态机器学习模型，旨在预测具有激活EGFR突变的晚期NSCLC患者对奥希替尼的耐药性，在多机构数据集上取得了0.82的c-index。该机器学习模型利用患者就诊和医疗评估中常规收集的可用数据，有助于实现精准的肺癌管理和知情的治疗决策。通过整合如组织学图像、下一代 sequencing（NGS）数据、人口统计数据和临床记录等多种数据类型，我们的多模态模型能够生成有针对性的建议。实验结果还表明，多模态模型的表现优于单模态模型（c-index为0.82，而单模态模型分别为0.75和0.77），这突显了在患者预后预测中结合多种模态的好处。 

---
# Learning a Canonical Basis of Human Preferences from Binary Ratings 

**Title (ZH)**: 从二元评分学习人类偏好的典范基markt-être 

**Authors**: Kailas Vodrahalli, Wei Wei, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2503.24150)  

**Abstract**: Recent advances in generative AI have been driven by alignment techniques such as reinforcement learning from human feedback (RLHF). RLHF and related techniques typically involve constructing a dataset of binary or ranked choice human preferences and subsequently fine-tuning models to align with these preferences. This paper shifts the focus to understanding the preferences encoded in such datasets and identifying common human preferences. We find that a small subset of 21 preference categories (selected from a set of nearly 5,000 distinct preferences) captures >89% of preference variation across individuals. This small set of preferences is analogous to a canonical basis of human preferences, similar to established findings that characterize human variation in psychology or facial recognition studies. Through both synthetic and empirical evaluations, we confirm that our low-rank, canonical set of human preferences generalizes across the entire dataset and within specific topics. We further demonstrate our preference basis' utility in model evaluation, where our preference categories offer deeper insights into model alignment, and in model training, where we show that fine-tuning on preference-defined subsets successfully aligns the model accordingly. 

**Abstract (ZH)**: 近期生成AI的进步得益于对人类反馈强化学习（RLHF）等对齐技术的推动。本论文将重点转向理解这类数据集中编码的偏好，并识别常见的人类偏好。我们发现，从近5000种独特偏好中选出的21个偏好类别 captures >89%的个体偏好差异。这个小的偏好集合类似于人类偏好的标准基底，类似于心理学或面部识别研究中确立的人类差异特征。通过合成和实证评估，我们确认我们的低秩、标准化人类偏好集合在整个数据集和特定主题内具有泛化能力。此外，我们展示了偏好基底在模型评估中的应用价值，我们的偏好类别为模型对齐提供了更深入的洞察，并在模型训练中证明，基于偏好定义的子集调整成功地使模型对齐。 

---
# Resonance: Drawing from Memories to Imagine Positive Futures through AI-Augmented Journaling 

**Title (ZH)**: 共振：通过AI增强日记想象积极未来的方式汲取记忆 

**Authors**: Wazeer Zulfikar, Treyden Chiaravalloti, Jocelyn Shen, Rosalind Picard, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2503.24145)  

**Abstract**: People inherently use experiences of their past while imagining their future, a capability that plays a crucial role in mental health. Resonance is an AI-powered journaling tool designed to augment this ability by offering AI-generated, action-oriented suggestions for future activities based on the user's own past memories. Suggestions are offered when a new memory is logged and are followed by a prompt for the user to imagine carrying out the suggestion. In a two-week randomized controlled study (N=55), we found that using Resonance significantly improved mental health outcomes, reducing the users' PHQ8 scores, a measure of current depression, and increasing their daily positive affect, particularly when they would likely act on the suggestion. Notably, the effectiveness of the suggestions was higher when they were personal, novel, and referenced the user's logged memories. Finally, through open-ended feedback, we discuss the factors that encouraged or hindered the use of the tool. 

**Abstract (ZH)**: 人们在想象未来时会固有地利用过去的体验，这一能力在心理健康方面发挥着重要作用。Resonance是一款以AI为动力的日记工具，旨在通过根据用户的个人 past 记忆提供基于行动的 AI 生成建议来增强这一能力，以促进未来的活动想象。在一项为期两周的随机对照研究（N=55）中，我们发现使用 Resonance 显著改善了心理健康结果，降低了用户 PHQ8 评分（当前抑郁的衡量标准），并增加了他们的日间积极情绪，尤其是在他们很可能采取建议行动时。值得注意的是，当建议具有个性化、新颖性且参考了用户登录的记忆时，建议的有效性更高。最后，通过开放式反馈，我们讨论了促进或阻碍使用该工具的因素。 

---
# Graph Neural Network-Based Predictive Modeling for Robotic Plaster Printing 

**Title (ZH)**: 基于图神经网络的机器人石膏打印预测建模 

**Authors**: Diego Machain Rivera, Selen Ercan Jenny, Ping Hsun Tsai, Ena Lloret-Fritschi, Luis Salamanca, Fernando Perez-Cruz, Konstantinos E. Tatsis  

**Link**: [PDF](https://arxiv.org/pdf/2503.24130)  

**Abstract**: This work proposes a Graph Neural Network (GNN) modeling approach to predict the resulting surface from a particle based fabrication process. The latter consists of spray-based printing of cementitious plaster on a wall and is facilitated with the use of a robotic arm. The predictions are computed using the robotic arm trajectory features, such as position, velocity and direction, as well as the printing process parameters. The proposed approach, based on a particle representation of the wall domain and the end effector, allows for the adoption of a graph-based solution. The GNN model consists of an encoder-processor-decoder architecture and is trained using data from laboratory tests, while the hyperparameters are optimized by means of a Bayesian scheme. The aim of this model is to act as a simulator of the printing process, and ultimately used for the generation of the robotic arm trajectory and the optimization of the printing parameters, towards the materialization of an autonomous plastering process. The performance of the proposed model is assessed in terms of the prediction error against unseen ground truth data, which shows its generality in varied scenarios, as well as in comparison with the performance of an existing benchmark model. The results demonstrate a significant improvement over the benchmark model, with notably better performance and enhanced error scaling across prediction steps. 

**Abstract (ZH)**: 基于图神经网络的颗粒堆积制造过程表面预测方法 

---
# PolypSegTrack: Unified Foundation Model for Colonoscopy Video Analysis 

**Title (ZH)**: 结肠镜视频分析的统一基础模型：PolypSegTrack 

**Authors**: Anwesa Choudhuri, Zhongpai Gao, Meng Zheng, Benjamin Planche, Terrence Chen, Ziyan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24108)  

**Abstract**: Early detection, accurate segmentation, classification and tracking of polyps during colonoscopy are critical for preventing colorectal cancer. Many existing deep-learning-based methods for analyzing colonoscopic videos either require task-specific fine-tuning, lack tracking capabilities, or rely on domain-specific pre-training. In this paper, we introduce \textit{PolypSegTrack}, a novel foundation model that jointly addresses polyp detection, segmentation, classification and unsupervised tracking in colonoscopic videos. Our approach leverages a novel conditional mask loss, enabling flexible training across datasets with either pixel-level segmentation masks or bounding box annotations, allowing us to bypass task-specific fine-tuning. Our unsupervised tracking module reliably associates polyp instances across frames using object queries, without relying on any heuristics. We leverage a robust vision foundation model backbone that is pre-trained unsupervisedly on natural images, thereby removing the need for domain-specific pre-training. Extensive experiments on multiple polyp benchmarks demonstrate that our method significantly outperforms existing state-of-the-art approaches in detection, segmentation, classification, and tracking. 

**Abstract (ZH)**: 早期检测、精确分割、分类和跟踪内镜检查中的息肉对于预防结直肠癌至关重要。现有的许多基于深度学习的内镜视频分析方法要么需要特定任务的微调，要么缺乏跟踪能力，要么依赖于特定领域的预训练。本文介绍了一种新颖的基础模型 \textit{PolypSegTrack}，可以同时解决内镜视频中息肉的检测、分割、分类和无监督跟踪问题。我们的方法利用了一种新颖的条件掩码损失，从而使模型能够在具有像素级分割掩码或边界框注释的数据集上灵活训练，从而避免了特定任务的微调。我们的无监督跟踪模块可靠地在帧间关联息肉实例，无需依赖任何启发式方法。我们采用了一种在自然图像上进行无监督预训练的鲁棒视觉基础模型骨干网络，从而消除了特定领域的预训练需求。在多个息肉基准上的 extensive 实验表明，我们的方法在检测、分割、分类和跟踪方面显著优于现有最先进的方法。 

---
# Artificial Conversations, Real Results: Fostering Language Detection with Synthetic Data 

**Title (ZH)**: 人工对话，真实成效：利用合成数据促进语言检测 

**Authors**: Fatemeh Mohammadi, Tommaso Romano, Samira Maghool, Paolo Ceravolo  

**Link**: [PDF](https://arxiv.org/pdf/2503.24062)  

**Abstract**: Collecting high-quality training data is essential for fine-tuning Large Language Models (LLMs). However, acquiring such data is often costly and time-consuming, especially for non-English languages such as Italian. Recently, researchers have begun to explore the use of LLMs to generate synthetic datasets as a viable alternative. This study proposes a pipeline for generating synthetic data and a comprehensive approach for investigating the factors that influence the validity of synthetic data generated by LLMs by examining how model performance is affected by metrics such as prompt strategy, text length and target position in a specific task, i.e. inclusive language detection in Italian job advertisements. Our results show that, in most cases and across different metrics, the fine-tuned models trained on synthetic data consistently outperformed other models on both real and synthetic test datasets. The study discusses the practical implications and limitations of using synthetic data for language detection tasks with LLMs. 

**Abstract (ZH)**: 高质量训练数据的收集对于Fine-tuning大型语言模型（LLMs）至关重要。然而，获取此类数据往往成本高昂且耗时，特别是在意大利语等非英语语言领域。最近，研究人员开始探索使用LLMs生成合成数据作为可行的替代方案。本研究提出了一种生成合成数据的管道，并通过评估模型性能受提示策略、文本长度和特定任务中的目标位置等因素的影响，对由LLMs生成的合成数据有效性进行了全面研究，具体任务是在意大利求职广告中检测包容性语言。研究结果表明，在大多数情况下且在不同指标下，使用合成数据训练的Fine-tuned模型在真实和合成测试数据集上的一般表现均优于其他模型。本研究讨论了使用合成数据进行语言检测任务的实用意义及其限制。 

---
# Bayesian Predictive Coding 

**Title (ZH)**: 贝叶斯预测编码 

**Authors**: Alexander Tschantz, Magnus Koudahl, Hampus Linander, Lancelot Da Costa, Conor Heins, Jeff Beck, Christopher Buckley  

**Link**: [PDF](https://arxiv.org/pdf/2503.24016)  

**Abstract**: Predictive coding (PC) is an influential theory of information processing in the brain, providing a biologically plausible alternative to backpropagation. It is motivated in terms of Bayesian inference, as hidden states and parameters are optimised via gradient descent on variational free energy. However, implementations of PC rely on maximum \textit{a posteriori} (MAP) estimates of hidden states and maximum likelihood (ML) estimates of parameters, limiting their ability to quantify epistemic uncertainty. In this work, we investigate a Bayesian extension to PC that estimates a posterior distribution over network parameters. This approach, termed Bayesian Predictive coding (BPC), preserves the locality of PC and results in closed-form Hebbian weight updates. Compared to PC, our BPC algorithm converges in fewer epochs in the full-batch setting and remains competitive in the mini-batch setting. Additionally, we demonstrate that BPC offers uncertainty quantification comparable to existing methods in Bayesian deep learning, while also improving convergence properties. Together, these results suggest that BPC provides a biologically plausible method for Bayesian learning in the brain, as well as an attractive approach to uncertainty quantification in deep learning. 

**Abstract (ZH)**: Bayesian Predictive Coding: A Biologically Plausible Method for Bayesian Learning and Uncertainty Quantification 

---
# Learning 3D-Gaussian Simulators from RGB Videos 

**Title (ZH)**: 从RGB视频学习3D高斯模拟器 

**Authors**: Mikel Zhobro, Andreas René Geist, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2503.24009)  

**Abstract**: Learning physics simulations from video data requires maintaining spatial and temporal consistency, a challenge often addressed with strong inductive biases or ground-truth 3D information -- limiting scalability and generalization. We introduce 3DGSim, a 3D physics simulator that learns object dynamics end-to-end from multi-view RGB videos. It encodes images into a 3D Gaussian particle representation, propagates dynamics via a transformer, and renders frames using 3D Gaussian splatting. By jointly training inverse rendering with a dynamics transformer using a temporal encoding and merging layer, 3DGSimembeds physical properties into point-wise latent vectors without enforcing explicit connectivity constraints. This enables the model to capture diverse physical behaviors, from rigid to elastic and cloth-like interactions, along with realistic lighting effects that also generalize to unseen multi-body interactions and novel scene edits. 

**Abstract (ZH)**: 3DGSim：从多视角RGB视频中端到端学习物体动力学的3D物理模拟器 

---
# H2VU-Benchmark: A Comprehensive Benchmark for Hierarchical Holistic Video Understanding 

**Title (ZH)**: H2VU基准：一个全面的层次化整体视频理解基准 

**Authors**: Qi Wu, Quanlong Zheng, Yanhao Zhang, Junlin Xie, Jinguo Luo, Kuo Wang, Peng Liu, Qingsong Xie, Ru Zhen, Haonan Lu, Zhenyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.24008)  

**Abstract**: With the rapid development of multimodal models, the demand for assessing video understanding capabilities has been steadily increasing. However, existing benchmarks for evaluating video understanding exhibit significant limitations in coverage, task diversity, and scene adaptability. These shortcomings hinder the accurate assessment of models' comprehensive video understanding capabilities. To tackle this challenge, we propose a hierarchical and holistic video understanding (H2VU) benchmark designed to evaluate both general video and online streaming video comprehension. This benchmark contributes three key features:
Extended video duration: Spanning videos from brief 3-second clips to comprehensive 1.5-hour recordings, thereby bridging the temporal gaps found in current benchmarks. Comprehensive assessment tasks: Beyond traditional perceptual and reasoning tasks, we have introduced modules for countercommonsense comprehension and trajectory state tracking. These additions test the models' deep understanding capabilities beyond mere prior knowledge. Enriched video data: To keep pace with the rapid evolution of current AI agents, we have expanded first-person streaming video datasets. This expansion allows for the exploration of multimodal models' performance in understanding streaming videos from a first-person perspective. Extensive results from H2VU reveal that existing multimodal large language models (MLLMs) possess substantial potential for improvement in our newly proposed evaluation tasks. We expect that H2VU will facilitate advancements in video understanding research by offering a comprehensive and in-depth analysis of MLLMs. 

**Abstract (ZH)**: 面向视频理解的层级化综合性基准（H2VU）：评估通用视频和流式视频理解能力 

---
# CITRAS: Covariate-Informed Transformer for Time Series Forecasting 

**Title (ZH)**: CITRAS: 带有协变量的变压器用于时间序列预测 

**Authors**: Yosuke Yamaguchi, Issei Suemitsu, Wenpeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.24007)  

**Abstract**: Covariates play an indispensable role in practical time series forecasting, offering rich context from the past and sometimes extending into the future. However, their availability varies depending on the scenario, and situations often involve multiple target variables simultaneously. Moreover, the cross-variate dependencies between them are multi-granular, with some covariates having a short-term impact on target variables and others showing long-term correlations. This heterogeneity and the intricate dependencies arising in covariate-informed forecasting present significant challenges to existing deep models. To address these issues, we propose CITRAS, a patch-based Transformer that flexibly leverages multiple targets and covariates covering both the past and the future forecasting horizon. While preserving the strong autoregressive capabilities of the canonical Transformer, CITRAS introduces two novel mechanisms in patch-wise cross-variate attention: Key-Value (KV) Shift and Attention Score Smoothing. KV Shift seamlessly incorporates future known covariates into the forecasting of target variables based on their concurrent dependencies. Additionally, Attention Score Smoothing transforms locally accurate patch-wise cross-variate dependencies into global variate-level dependencies by smoothing the past series of attention scores. Experimentally, CITRAS achieves state-of-the-art performance in both covariate-informed and multivariate forecasting, demonstrating its versatile ability to leverage cross-variate dependency for improved forecasting accuracy. 

**Abstract (ZH)**: 协变量在实际时间序列预测中扮演着不可或缺的角色，提供丰富的过去和有时甚至未来的上下文。然而，它们的可用性因场景而异，且情况往往同时涉及多个目标变量。此外，协变量间的交叉依赖是多尺度的，有些协变量对目标变量有短期影响，而其他协变量则显示出长期相关性。这种异质性和协变量知情预测中引发的复杂依赖关系对现有深度模型构成了重大挑战。为了解决这些问题，我们提出了CITRAS，一种基于补丁的Transformer，灵活利用覆盖过去和未来预测范围的多个目标变量和协变量。在保留标准Transformer强大的自回归能力的同时，CITRAS引入了两种新的机制：Key-Value (KV) Shift和Attention Score Smoothing。KV Shift无缝地根据当前依赖关系将未来的已知协变量纳入目标变量的预测。另外，Attention Score Smoothing通过平滑过去的注意力分数，将局部准确的补丁级交叉依赖关系转化为全局变量级依赖关系。实验表明，CITRAS在协变量知情和多变量预测中均达到了最先进的性能，展示了其利用交叉依赖关系提升预测准确性的 versatility。 

---
# Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving 

**Title (ZH)**: 重新思考大型语言模型服务中键值缓存压缩技术的方法 

**Authors**: Wei Gao, Xinyu Zhou, Peng Sun, Tianwei Zhang, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.24000)  

**Abstract**: Key-Value cache (\texttt{KV} \texttt{cache}) compression has emerged as a promising technique to optimize Large Language Model (LLM) serving. It primarily decreases the memory consumption of \texttt{KV} \texttt{cache} to reduce the computation cost. Despite the development of many compression algorithms, their applications in production environments are still not prevalent. In this paper, we revisit mainstream \texttt{KV} \texttt{cache} compression solutions from a practical perspective. Our contributions are three-fold. First, we comprehensively review existing algorithmic designs and benchmark studies for \texttt{KV} \texttt{cache} compression and identify missing pieces in their performance measurement, which could hinder their adoption in practice. Second, we empirically evaluate representative \texttt{KV} \texttt{cache} compression methods to uncover two key issues that affect the computational efficiency: (1) while compressing \texttt{KV} \texttt{cache} can reduce memory consumption, current implementations (e.g., FlashAttention, PagedAttention) do not optimize for production-level LLM serving, resulting in suboptimal throughput performance; (2) compressing \texttt{KV} \texttt{cache} may lead to longer outputs, resulting in increased end-to-end latency. We further investigate the accuracy performance of individual samples rather than the overall performance, revealing the intrinsic limitations in \texttt{KV} \texttt{cache} compression when handling specific LLM tasks. Third, we provide tools to shed light on future \texttt{KV} \texttt{cache} compression studies and facilitate their practical deployment in production. They are open-sourced in \href{this https URL}{this https URL}. 

**Abstract (ZH)**: Key-Value 缓存（\texttt{KV} 缓存）压缩已成为优化大型语言模型（LLM）服务的一种有前途的技术。它主要通过减少\texttt{KV} 缓存的内存消耗来降低计算成本。尽管已经开发出了许多压缩算法，但它们在生产环境中的应用仍然不够广泛。在本文中，我们从实用的角度回顾了主流的\texttt{KV} 缓存压缩解决方案。我们的贡献主要体现在三个方面。首先，我们全面回顾了现有的算法设计和基准研究，并识别出它们在性能测量中存在的不足之处，这可能阻碍其实现。其次，我们实证评估了代表性的\texttt{KV} 缓存压缩方法，发现了影响计算效率的两个关键问题：（1）虽然压缩\texttt{KV} 缓存可以减少内存消耗，但当前实现（如FlashAttention、PagedAttention）未针对生产级别的LLM服务进行优化，导致吞吐量性能欠佳；（2）压缩\texttt{KV} 缓存可能会导致输出时间延长，从而增加端到端延迟。我们进一步研究了单个样本的准确性性能，揭示了特定LLM任务处理中\texttt{KV} 缓存压缩的内在限制。第三，我们提供了工具，以促进未来\texttt{KV} 缓存压缩研究，并使其在生产环境中得以实际应用。这些工具已在\href{this https URL}{this https URL}开源。 

---
# DenseFormer: Learning Dense Depth Map from Sparse Depth and Image via Conditional Diffusion Model 

**Title (ZH)**: DenseFormer：通过条件扩散模型从稀疏深度图和图像学习密集深度图 

**Authors**: Ming Yuan, Sichao Wang, Chuang Zhang, Lei He, Qing Xu, Jianqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23993)  

**Abstract**: The depth completion task is a critical problem in autonomous driving, involving the generation of dense depth maps from sparse depth maps and RGB images. Most existing methods employ a spatial propagation network to iteratively refine the depth map after obtaining an initial dense depth. In this paper, we propose DenseFormer, a novel method that integrates the diffusion model into the depth completion task. By incorporating the denoising mechanism of the diffusion model, DenseFormer generates the dense depth map by progressively refining an initial random depth distribution through multiple iterations. We propose a feature extraction module that leverages a feature pyramid structure, along with multi-layer deformable attention, to effectively extract and integrate features from sparse depth maps and RGB images, which serve as the guiding condition for the diffusion process. Additionally, this paper presents a depth refinement module that applies multi-step iterative refinement across various ranges to the dense depth results generated by the diffusion process. The module utilizes image features enriched with multi-scale information and sparse depth input to further enhance the accuracy of the predicted depth map. Extensive experiments on the KITTI outdoor scene dataset demonstrate that DenseFormer outperforms classical depth completion methods. 

**Abstract (ZH)**: 深度完成任务是自主驾驶中的关键技术问题，涉及从稀疏深度图和RGB图像生成密集深度图。大多数现有方法使用空间传播网络在获得初始密集深度图后逐迭代地细化深度图。本文提出了一种新的方法DenseFormer，将扩散模型集成到深度完成任务中。通过结合扩散模型的去噪机制，DenseFormer通过多次迭代逐步细化初始随机深度分布生成密集深度图。我们提出了一种特征提取模块，利用特征金字塔结构和多层可变形注意力机制，有效提取并集成来自稀疏深度图和RGB图像的特征，这些特征作为扩散过程的引导条件。此外，本文还提出了一种深度细化模块，该模块对扩散过程生成的密集深度结果在不同范围内进行多步迭代细化，并利用多尺度信息丰富的图像特征和稀疏深度输入进一步增强预测深度图的准确性。在KITTI室外场景数据集上的大量实验表明，DenseFormer优于经典的深度完成方法。 

---
# Rubric Is All You Need: Enhancing LLM-based Code Evaluation With Question-Specific Rubrics 

**Title (ZH)**: 评分标准即所需：基于问答特定评分标准提升大语言模型代码评价能力 

**Authors**: Aditya Pathak, Rachit Gandhi, Vaibhav Uttam, Devansh, Yashwanth Nakka, Aaryan Raj Jindal, Pratyush Ghosh, Arnav Ramamoorthy, Shreyash Verma, Aditya Mittal, Aashna Ased, Chirag Khatri, Jagat Sesh Challa, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.23989)  

**Abstract**: Since the disruption in LLM technology brought about by the release of GPT-3 and ChatGPT, LLMs have shown remarkable promise in programming-related tasks. While code generation remains a popular field of research, code evaluation using LLMs remains a problem with no conclusive solution. In this paper, we focus on LLM-based code evaluation and attempt to fill in the existing gaps. We propose multi-agentic novel approaches using question-specific rubrics tailored to the problem statement, arguing that these perform better for logical assessment than the existing approaches that use question-agnostic rubrics. To address the lack of suitable evaluation datasets, we introduce two datasets: a Data Structures and Algorithms dataset containing 150 student submissions from a popular Data Structures and Algorithms practice website, and an Object Oriented Programming dataset comprising 80 student submissions from undergraduate computer science courses. In addition to using standard metrics (Spearman Correlation, Cohen's Kappa), we additionally propose a new metric called as Leniency, which quantifies evaluation strictness relative to expert assessment. Our comprehensive analysis demonstrates that question-specific rubrics significantly enhance logical assessment of code in educational settings, providing better feedback aligned with instructional goals beyond mere syntactic correctness. 

**Abstract (ZH)**: 自GPT-3和ChatGPT发布以来，LLM技术在编程相关任务中显示出显著的潜力。尽管代码生成仍然是一个热门的研究领域，但使用LLM进行代码评估仍然是一个没有明确解决方案的问题。本文聚焦于基于LLM的代码评估，试图填补现有空白。我们提出了基于多代理的新颖方法，并针对问题陈述量身定制评分标准，认为这些方法在逻辑评估方面优于现有使用通用评分标准的方法。为了解决合适的评估数据集缺乏的问题，我们引入了两个数据集：一个包含来自流行的数据结构和算法练习网站的150份学生提交的数据结构和算法数据集，以及一个包含来自本科计算机科学课程的80份学生提交的对象导向编程数据集。除使用标准指标（Spearman相关系数和Cohen’s Kappa系数）外，我们还提出了一种新的指标称为宽容度（Leniency），用于量化评估的严格程度与专家评估的差异。我们的综合分析表明，针对问题的评分标准在教育环境中显著增强了代码的逻辑评估，提供了与教学目标更一致的反馈，远超单纯的语法正确性。 

---
# Deep Learning Model Deployment in Multiple Cloud Providers: an Exploratory Study Using Low Computing Power Environments 

**Title (ZH)**: 多云提供商环境下基于低计算资源的深度学习模型部署探索性研究 

**Authors**: Elayne Lemos, Rodrigo Oliveira, Jairson Rodrigues, Rosalvo F. Oliveira Neto  

**Link**: [PDF](https://arxiv.org/pdf/2503.23988)  

**Abstract**: The deployment of Machine Learning models at cloud have grown by tech companies. Hardware requirements are higher when these models involve Deep Learning (DL) techniques and the cloud providers' costs may be a barrier. We explore deploying DL models using for experiments the GECToR model, a DL solution for Grammatical Error Correction, across three of the major cloud platforms (AWS, Google Cloud, Azure). We evaluate real-time latency, hardware usage and cost at each cloud provider by 7 execution environments with 10 experiments reproduced. We found that while GPUs excel in performance, they had an average cost 300% higher than solutions without GPU. Our analysis also identifies that processor cache size is crucial for cost-effective CPU deployments, enabling over 50% of cost reduction compared to GPUs. This study demonstrates the feasibility and affordability of cloud-based DL inference solutions without GPUs, benefiting resource-constrained users like startups. 

**Abstract (ZH)**: 云平台上基于机器学习模型的部署和技术公司的发展。采用GECToR模型探究深度学习模型在三大云平台（AWS、Google Cloud、Azure）上的部署。通过7个执行环境和10次实验评估各云提供商的实时延迟、硬件使用和成本。研究发现，虽然GPU在性能上表现出色，但其平均成本比无GPU解决方案高300%。分析还表明，处理器缓存大小对于降低成本的关键CPU部署至关重要，可实现超过50%的成本减少。本研究展示了在不使用GPU的情况下，云-Based深度学习推理解决方案的可行性和经济性，惠及资源受限的用户，如初创企业。 

---
# Deep Nets as Hamiltonians 

**Title (ZH)**: 深度网络作为哈密顿量 

**Authors**: Mike Winer, Boris Hanin  

**Link**: [PDF](https://arxiv.org/pdf/2503.23982)  

**Abstract**: Neural networks are complex functions of both their inputs and parameters. Much prior work in deep learning theory analyzes the distribution of network outputs at a fixed a set of inputs (e.g. a training dataset) over random initializations of the network parameters. The purpose of this article is to consider the opposite situation: we view a randomly initialized Multi-Layer Perceptron (MLP) as a Hamiltonian over its inputs. For typical realizations of the network parameters, we study the properties of the energy landscape induced by this Hamiltonian, focusing on the structure of near-global minimum in the limit of infinite width. Specifically, we use the replica trick to perform an exact analytic calculation giving the entropy (log volume of space) at a given energy. We further derive saddle point equations that describe the overlaps between inputs sampled iid from the Gibbs distribution induced by the random MLP. For linear activations we solve these saddle point equations exactly. But we also solve them numerically for a variety of depths and activation functions, including $\tanh, \sin, \text{ReLU}$, and shaped non-linearities. We find even at infinite width a rich range of behaviors. For some non-linearities, such as $\sin$, for instance, we find that the landscapes of random MLPs exhibit full replica symmetry breaking, while shallow $\tanh$ and ReLU networks or deep shaped MLPs are instead replica symmetric. 

**Abstract (ZH)**: 神经网络是其输入和参数的复杂函数。本文旨在研究随机初始化神经网络参数情况下，网络输入的海森堡量纲诱导的能量景观性质，特别是在网络宽度无限大时，近全局最小值的结构。我们使用复制技巧进行精确的解析计算，给出给定能量下的熵（空间体积的对数）。进一步推导描述从由随机多层感知机诱导的吉布斯分布中独立同分布采样输入之间的重叠的鞍点方程。对于线性激活函数，我们精确求解了这些鞍点方程。我们还对不同深度和激活函数进行了数值求解，包括tanh、sin、ReLU以及各种非线性。我们发现，在网络宽度无限大时，随机多层感知机的能量景观表现出丰富的行为。例如，对于sin激活函数，随机多层感知机的能量景观表现出完全的复制对称性破坏，而对于浅层tanh和ReLU网络或深层形状感知机，则表现出复制对称性。 

---
# Noise-based reward-modulated learning 

**Title (ZH)**: 基于噪声的奖励调节学习 

**Authors**: Jesús García Fernández, Nasir Ahmad, Marcel van Gerven  

**Link**: [PDF](https://arxiv.org/pdf/2503.23972)  

**Abstract**: Recent advances in reinforcement learning (RL) have led to significant improvements in task performance. However, training neural networks in an RL regime is typically achieved in combination with backpropagation, limiting their applicability in resource-constrained environments or when using non-differentiable neural networks. While noise-based alternatives like reward-modulated Hebbian learning (RMHL) have been proposed, their performance has remained limited, especially in scenarios with delayed rewards, which require retrospective credit assignment over time. Here, we derive a novel noise-based learning rule that addresses these challenges. Our approach combines directional derivative theory with Hebbian-like updates to enable efficient, gradient-free learning in RL. It features stochastic noisy neurons which can approximate gradients, and produces local synaptic updates modulated by a global reward signal. Drawing on concepts from neuroscience, our method uses reward prediction error as its optimization target to generate increasingly advantageous behavior, and incorporates an eligibility trace to facilitate temporal credit assignment in environments with delayed rewards. Its formulation relies on local information alone, making it compatible with implementations in neuromorphic hardware. Experimental validation shows that our approach significantly outperforms RMHL and is competitive with BP-based baselines, highlighting the promise of noise-based, biologically inspired learning for low-power and real-time applications. 

**Abstract (ZH)**: 最近 reinforcement learning 的进展显著提高了任务性能，但在资源受限环境中或使用非可微神经网络时，通过反向传播训练神经网络的限制使其应用受到了限制。尽管提出了基于噪声的方法，如奖励调制的希布bian学习(RMHL)，但在延迟奖励等情境中，其性能仍然有限，这要求在时间上进行追溯的信用分配。在这里，我们推导出一种新的基于噪声的学习规则，以应对这些挑战。我们的方法结合了方向导数理论和希布bian-like 更新，能够在 reinforcement learning 中实现高效、无梯度的学习。该方法采用具有噪声的随机神经元来近似梯度，并通过全局奖励信号调节局部突触更新。借鉴神经科学的概念，我们的方法将奖励预测误差用作优化目标，以生成更有利的行为，并结合了弹性迹线来促进延迟奖励环境中时间上的信用分配。该方法的表达式仅依赖于局部信息，使其与神经形态硬件的实现兼容。实验验证表明，我们的方法显著优于 RMHL，并在基于反向传播的基准方法中表现出竞争力，展示了基于噪声的、生物启发的学习方法在低功耗和实时应用中的潜力。 

---
# AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference 

**Title (ZH)**: AirCache: 激活跨模态相关性键值缓存压缩以实现高效的大规模视觉-语言模型推理 

**Authors**: Kai Huang, Hao Zou, Bochen Wang, Ye Xi, Zhen Xie, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23956)  

**Abstract**: Recent advancements in Large Visual Language Models (LVLMs) have gained significant attention due to their remarkable reasoning capabilities and proficiency in generalization. However, processing a large number of visual tokens and generating long-context outputs impose substantial computational overhead, leading to excessive demands for key-value (KV) cache. To address this critical bottleneck, we propose AirCache, a novel KV cache compression method aimed at accelerating LVLMs inference. This work systematically investigates the correlations between visual and textual tokens within the attention mechanisms of LVLMs. Our empirical analysis reveals considerable redundancy in cached visual tokens, wherein strategically eliminating these tokens preserves model performance while significantly accelerating context generation. Inspired by these findings, we introduce an elite observation window for assessing the importance of visual components in the KV cache, focusing on stable inter-modal relevancy modeling with enhanced multi-perspective consistency. Additionally, we develop an adaptive layer-wise budget allocation strategy that capitalizes on the strength and skewness of token importance distribution, showcasing superior efficiency compared to uniform allocation. Comprehensive evaluations across multiple LVLMs and benchmarks demonstrate that our method achieves comparable performance to the full cache while retaining only 10% of visual KV cache, thereby reducing decoding latency by 29% to 66% across various batch size and prompt length of inputs. Notably, as cache retention rates decrease, our method exhibits increasing performance advantages over existing approaches. 

**Abstract (ZH)**: 最近在大型视觉语言模型（LVLMs）方面的进展因其卓越的推理能力及泛化能力而引起了广泛关注。然而，处理大量视觉令牌和生成长上下文输出带来了显著的计算成本，导致了对键值（KV）缓存的极大需求。为解决这一关键瓶颈，我们提出了AirCache，一种旨在加速LVLMs推理的新型KV缓存压缩方法。这项工作系统地探讨了LVLMs中的注意力机制中视觉令牌和文本令牌之间的关联。实证分析表明，缓存中的视觉令牌存在大量冗余，通过有选择地消除这些令牌，可以在保持模型性能的同时显著加速上下文生成。受这一发现的启发，我们介绍了一种精英观察窗口，用于评估KV缓存中视觉组件的重要性，重点在于稳定跨模态相关性建模和增强多视角一致性。此外，我们还开发了一种自适应逐层预算分配策略，该策略充分利用了令牌重要性分布的优势和偏斜性，相比于均匀分配显示出了更优越的效率。在多个LVLMs和基准测试中的综合评估表明，我们的方法在保留仅10%视觉KV缓存的情况下，实现了与全缓存相当的性能，同时减少了29%至66%的解码延迟，无论输入的批大小还是提示长度如何。特别地，随着缓存保留率的下降，我们的方法相对于现有方法呈现出越来越大的性能优势。 

---
# Green MLOps to Green GenOps: An Empirical Study of Energy Consumption in Discriminative and Generative AI Operations 

**Title (ZH)**: 从绿色MLOps到绿色GenOps：一种区分性和生成性AI操作的能源消耗实证研究 

**Authors**: Adrián Sánchez-Mompó, Ioannis Mavromatis, Peizheng Li, Konstantinos Katsaros, Aftab Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.23934)  

**Abstract**: This study presents an empirical investigation into the energy consumption of Discriminative and Generative AI models within real-world MLOps pipelines. For Discriminative models, we examine various architectures and hyperparameters during training and inference and identify energy-efficient practices. For Generative AI, Large Language Models (LLMs) are assessed, focusing primarily on energy consumption across different model sizes and varying service requests. Our study employs software-based power measurements, ensuring ease of replication across diverse configurations, models, and datasets. We analyse multiple models and hardware setups to uncover correlations among various metrics, identifying key contributors to energy consumption. The results indicate that for Discriminative models, optimising architectures, hyperparameters, and hardware can significantly reduce energy consumption without sacrificing performance. For LLMs, energy efficiency depends on balancing model size, reasoning complexity, and request-handling capacity, as larger models do not necessarily consume more energy when utilisation remains low. This analysis provides practical guidelines for designing green and sustainable ML operations, emphasising energy consumption and carbon footprint reductions while maintaining performance. This paper can serve as a benchmark for accurately estimating total energy use across different types of AI models. 

**Abstract (ZH)**: 本研究对实际MLOps管道中判别性和生成性AI模型的能源消耗进行了实证调查。对于判别性模型，我们 examining 各种训练和推理架构及超参数，并识别能源高效实践。对于生成性AI，主要评估大型语言模型（LLMs）在不同模型规模和服务请求变化下的能源消耗。本研究采用基于软件的电源测量方法，确保在不同配置、模型和数据集上轻松复制。我们分析多种模型和硬件组合以揭示各种指标之间的关联，识别能源消耗的关键因素。研究结果表明，对于判别性模型，通过优化架构、超参数和硬件可以显著减少能源消耗，而不牺牲性能。对于LLMs，能源效率取决于平衡模型规模、推理复杂性和请求处理能力，利用率较低时，更大规模的模型不一定消耗更多能源。本分析提供了设计绿色可持续ML操作的实用指南，强调减少能源消耗和碳足迹的同时保持性能。本论文可作为准确估计不同AI模型类型总能源使用量的基准。 

---
# HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessment 

**Title (ZH)**: HumanAesExpert: 推动多模态基础模型在人体图像美学评估中的应用 

**Authors**: Zhichao Liao, Xiaokun Liu, Wenyu Qin, Qingyu Li, Qiulin Wang, Pengfei Wan, Di Zhang, Long Zeng, Pingfa Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.23907)  

**Abstract**: Image Aesthetic Assessment (IAA) is a long-standing and challenging research task. However, its subset, Human Image Aesthetic Assessment (HIAA), has been scarcely explored, even though HIAA is widely used in social media, AI workflows, and related domains. To bridge this research gap, our work pioneers a holistic implementation framework tailored for HIAA. Specifically, we introduce HumanBeauty, the first dataset purpose-built for HIAA, which comprises 108k high-quality human images with manual annotations. To achieve comprehensive and fine-grained HIAA, 50K human images are manually collected through a rigorous curation process and annotated leveraging our trailblazing 12-dimensional aesthetic standard, while the remaining 58K with overall aesthetic labels are systematically filtered from public datasets. Based on the HumanBeauty database, we propose HumanAesExpert, a powerful Vision Language Model for aesthetic evaluation of human images. We innovatively design an Expert head to incorporate human knowledge of aesthetic sub-dimensions while jointly utilizing the Language Modeling (LM) and Regression head. This approach empowers our model to achieve superior proficiency in both overall and fine-grained HIAA. Furthermore, we introduce a MetaVoter, which aggregates scores from all three heads, to effectively balance the capabilities of each head, thereby realizing improved assessment precision. Extensive experiments demonstrate that our HumanAesExpert models deliver significantly better performance in HIAA than other state-of-the-art models. Our datasets, models, and codes are publicly released to advance the HIAA community. Project webpage: this https URL 

**Abstract (ZH)**: 图像美学评估（IAA）是一项长期而具有挑战性的研究任务。然而，其子集，人类图像美学评估（HIAA），虽在社交媒体、AI工作流及相关领域广泛应用，但仍未受到广泛探索。为弥合这一研究差距，我们的工作开创了一种适用于HIAA的整体实施框架。具体而言，我们引入了HumanBeauty，这是首个专为HIAA构建的数据集，包含10.8万张高质量的人像图片并附有人工标注。为实现全面且细粒度的HIAA，5万张人像图片通过 rigorous curation 过程手工收集并在我们开创性的12维美学标准下进行标注，剩余5.8万张图片则根据整体美学标签从公共数据集中系统筛选。基于HumanBeauty数据库，我们提出了HumanAesExpert，这是一种强大的视觉语言模型，用于评估人像图片的美学。我们创新地设计了一个专家头，将人类对美学子维度的知识纳入其中，并结合语言模型（LM）和回归头进行联合利用。这种方法赋予了我们的模型在整体和细粒度HIAA方面卓越的专业能力。此外，我们引入了一种元投票器（MetaVoter），它可以有效综合三个头的评分，从而平衡每个头的能力，实现提高评估精度。广泛实验表明，我们的HumanAesExpert模型在HIAA方面显著优于其他最先进的模型。我们的数据集、模型和代码已公开发布，以推动HIAA社区的发展。项目网页: this https URL。 

---
# Training-Free Text-Guided Image Editing with Visual Autoregressive Model 

**Title (ZH)**: 基于视觉自回归模型的无训练文本引导图像编辑 

**Authors**: Yufei Wang, Lanqing Guo, Zhihao Li, Jiaxing Huang, Pichao Wang, Bihan Wen, Jian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23897)  

**Abstract**: Text-guided image editing is an essential task that enables users to modify images through natural language descriptions. Recent advances in diffusion models and rectified flows have significantly improved editing quality, primarily relying on inversion techniques to extract structured noise from input images. However, inaccuracies in inversion can propagate errors, leading to unintended modifications and compromising fidelity. Moreover, even with perfect inversion, the entanglement between textual prompts and image features often results in global changes when only local edits are intended. To address these challenges, we propose a novel text-guided image editing framework based on VAR (Visual AutoRegressive modeling), which eliminates the need for explicit inversion while ensuring precise and controlled modifications. Our method introduces a caching mechanism that stores token indices and probability distributions from the original image, capturing the relationship between the source prompt and the image. Using this cache, we design an adaptive fine-grained masking strategy that dynamically identifies and constrains modifications to relevant regions, preventing unintended changes. A token reassembling approach further refines the editing process, enhancing diversity, fidelity, and control. Our framework operates in a training-free manner and achieves high-fidelity editing with faster inference speeds, processing a 1K resolution image in as fast as 1.2 seconds. Extensive experiments demonstrate that our method achieves performance comparable to, or even surpassing, existing diffusion- and rectified flow-based approaches in both quantitative metrics and visual quality. The code will be released. 

**Abstract (ZH)**: 基于VAR的文本引导图像编辑框架 

---
# Better wit than wealth: Dynamic Parametric Retrieval Augmented Generation for Test-time Knowledge Enhancement 

**Title (ZH)**: 更有智慧胜过财富：动态参数检索增强生成在测试时知识增强 

**Authors**: Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23895)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources and incorporating them into the context. While it improves reliability by providing factual texts, it significantly increases inference costs as context length grows and introduces challenging issue of RAG hallucination, primarily caused by the lack of corresponding parametric knowledge in LLMs. An efficient solution is to enhance the knowledge of LLMs at test-time. Parametric RAG (PRAG) addresses this by embedding document into LLMs parameters to perform test-time knowledge enhancement, effectively reducing inference costs through offline training. However, its high training and storage costs, along with limited generalization ability, significantly restrict its practical adoption. To address these challenges, we propose Dynamic Parametric RAG (DyPRAG), a novel framework that leverages a lightweight parameter translator model to efficiently convert documents into parametric knowledge. DyPRAG not only reduces inference, training, and storage costs but also dynamically generates parametric knowledge, seamlessly enhancing the knowledge of LLMs and resolving knowledge conflicts in a plug-and-play manner at test-time. Extensive experiments on multiple datasets demonstrate the effectiveness and generalization capabilities of DyPRAG, offering a powerful and practical RAG paradigm which enables superior knowledge fusion and mitigates RAG hallucination in real-world applications. Our code is available at this https URL. 

**Abstract (ZH)**: 动态参数增强检索生成（DyPRAG）：一种轻量级参数转换模型驱动的知识增强框架 

---
# DiffScale: Continuous Downscaling and Bias Correction of Subseasonal Wind Speed Forecasts using Diffusion Models 

**Title (ZH)**: DiffScale：基于扩散模型的子季节风速预报的连续下-scaling和偏差校正 

**Authors**: Maximilian Springenberg, Noelia Otero, Yuxin Xue, Jackie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.23893)  

**Abstract**: Renewable resources are strongly dependent on local and large-scale weather situations. Skillful subseasonal to seasonal (S2S) forecasts -- beyond two weeks and up to two months -- can offer significant socioeconomic advantages to the energy sector. This study aims to enhance wind speed predictions using a diffusion model with classifier-free guidance to downscale S2S forecasts of surface wind speed. We propose DiffScale, a diffusion model that super-resolves spatial information for continuous downscaling factors and lead times. Leveraging weather priors as guidance for the generative process of diffusion models, we adopt the perspective of conditional probabilities on sampling super-resolved S2S forecasts. We aim to directly estimate the density associated with the target S2S forecasts at different spatial resolutions and lead times without auto-regression or sequence prediction, resulting in an efficient and flexible model. Synthetic experiments were designed to super-resolve wind speed S2S forecasts from the European Center for Medium-Range Weather Forecast (ECMWF) from a coarse resolution to a finer resolution of ERA5 reanalysis data, which serves as a high-resolution target. The innovative aspect of DiffScale lies in its flexibility to downscale arbitrary scaling factors, enabling it to generalize across various grid resolutions and lead times -without retraining the model- while correcting model errors, making it a versatile tool for improving S2S wind speed forecasts. We achieve a significant improvement in prediction quality, outperforming baselines up to week 3. 

**Abstract (ZH)**: 利用去 classifier 指导的扩散模型提升表面风速子季节至季节预报的超分辨率预测 

---
# MuseFace: Text-driven Face Editing via Diffusion-based Mask Generation Approach 

**Title (ZH)**: MuseFace：基于扩散掩码生成方法的文本驱动面部编辑 

**Authors**: Xin Zhang, Siting Huang, Xiangyang Luo, Yifan Xie, Weijiang Yu, Heng Chang, Fei Ma, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23888)  

**Abstract**: Face editing modifies the appearance of face, which plays a key role in customization and enhancement of personal images. Although much work have achieved remarkable success in text-driven face editing, they still face significant challenges as none of them simultaneously fulfill the characteristics of diversity, controllability and flexibility. To address this challenge, we propose MuseFace, a text-driven face editing framework, which relies solely on text prompt to enable face editing. Specifically, MuseFace integrates a Text-to-Mask diffusion model and a semantic-aware face editing model, capable of directly generating fine-grained semantic masks from text and performing face editing. The Text-to-Mask diffusion model provides \textit{diversity} and \textit{flexibility} to the framework, while the semantic-aware face editing model ensures \textit{controllability} of the framework. Our framework can create fine-grained semantic masks, making precise face editing possible, and significantly enhancing the controllability and flexibility of face editing models. Extensive experiments demonstrate that MuseFace achieves superior high-fidelity performance. 

**Abstract (ZH)**: 文本驱动的面部编辑修改了面部的外观，在个性化和增强个人形象方面发挥着关键作用。尽管大量工作在文本驱动的面部编辑方面取得了显著成功，但它们仍然面临重大挑战，即没有任何方法能够同时具备多样性、可控性和灵活性的特点。为解决这一挑战，我们提出了一种文本驱动的面部编辑框架MuseFace，该框架仅依赖文本提示来实现面部编辑。具体而言，MuseFace 结合了文本到掩码的扩散模型和语义感知的面部编辑模型，能够直接从文本生成精细的语义掩码并执行面部编辑。文本到掩码的扩散模型为框架提供了多样性和灵活性，而语义感知的面部编辑模型则确保了框架的可控性。我们的框架能够生成精细的语义掩码，从而使精确的面部编辑成为可能，并显著提高了面部编辑模型的可控性和灵活性。广泛实验表明，MuseFace 达到了优越的高保真性能。 

---
# SchemaAgent: A Multi-Agents Framework for Generating Relational Database Schema 

**Title (ZH)**: SchemaAgent：一个生成关系数据库模式的多智能体框架 

**Authors**: Qin Wang, Youhuan Li, Yansong Feng, Si Chen, Ziming Li, Pan Zhang, Zhichao Shi, Yuequn Dou, chuchu Gao, Zebin Huang, Zihui Si, Yixuan Chen, Zhaohai Sun, Ke Tang, Wenqiang Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.23886)  

**Abstract**: The relational database design would output a schema based on user's requirements, which defines table structures and their interrelated relations. Translating requirements into accurate schema involves several non-trivial subtasks demanding both database expertise and domain-specific knowledge. This poses unique challenges for automated design of relational databases. Existing efforts are mostly based on customized rules or conventional deep learning models, often producing suboptimal schema. Recently, large language models (LLMs) have significantly advanced intelligent application development across various domains. In this paper, we propose SchemaAgent, a unified LLM-based multi-agent framework for the automated generation of high-quality database schema. SchemaAgent is the first to apply LLMs for schema generation, which emulates the workflow of manual schema design by assigning specialized roles to agents and enabling effective collaboration to refine their respective subtasks. Schema generation is a streamlined workflow, where directly applying the multi-agent framework may cause compounding impact of errors. To address this, we incorporate dedicated roles for reflection and inspection, alongside an innovative error detection and correction mechanism to identify and rectify issues across various phases. For evaluation, we present a benchmark named \textit{RSchema}, which contains more than 500 pairs of requirement description and schema. Experimental results on this benchmark demonstrate the superiority of our approach over mainstream LLMs for relational database schema generation. 

**Abstract (ZH)**: 基于关系数据库的设计将根据用户需求输出一个模式，定义表结构及其相互关系。将需求转换为准确的模式涉及多个非平凡的子任务，既需要数据库专业知识，也需特定领域的知识。这为关系数据库的自动化设计带来了独特的挑战。现有的努力大多基于定制规则或传统的深度学习模型，常常生成次优化的模式。最近，大型语言模型（LLMs）在各领域智能应用开发中取得了显著进展。在本文中，我们提出SchemaAgent，一个统一的基于LLM的多智能体框架，用于自动化生成高质量的数据库模式。SchemaAgent是首次将LLM应用于模式生成，通过为智能体分配特定角色并实现有效的协作来模拟手动模式设计的流程，精炼各自子任务。模式生成是一个简化的工作流，直接应用多智能体框架可能导致错误累积的影响。为此，我们引入了专门的角色进行反思和检查，并结合一种创新的错误检测和校正机制，以跨各个阶段识别和修正问题。为了评估，我们提出了一个名为“RSchema”的基准，包含超过500对需求描述和模式。在此基准上的实验结果表明，我们的方法在关系数据库模式生成方面优于主流的LLM。 

---
# GenSwarm: Scalable Multi-Robot Code-Policy Generation and Deployment via Language Models 

**Title (ZH)**: GenSwarm: 通过语言模型实现可扩展的多机器人代码-策略生成与部署 

**Authors**: Wenkang Ji, Huaben Chen, Mingyang Chen, Guobin Zhu, Lufeng Xu, Roderich Groß, Rui Zhou, Ming Cao, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.23875)  

**Abstract**: The development of control policies for multi-robot systems traditionally follows a complex and labor-intensive process, often lacking the flexibility to adapt to dynamic tasks. This has motivated research on methods to automatically create control policies. However, these methods require iterative processes of manually crafting and refining objective functions, thereby prolonging the development cycle. This work introduces \textit{GenSwarm}, an end-to-end system that leverages large language models to automatically generate and deploy control policies for multi-robot tasks based on simple user instructions in natural language. As a multi-language-agent system, GenSwarm achieves zero-shot learning, enabling rapid adaptation to altered or unseen tasks. The white-box nature of the code policies ensures strong reproducibility and interpretability. With its scalable software and hardware architectures, GenSwarm supports efficient policy deployment on both simulated and real-world multi-robot systems, realizing an instruction-to-execution end-to-end functionality that could prove valuable for robotics specialists and non-specialists this http URL code of the proposed GenSwarm system is available online: this https URL. 

**Abstract (ZH)**: 基于自然语言简要说明的多机器人系统的端到端自动生成与部署控制策略系统GenSwarm 

---
# Learned Image Compression and Restoration for Digital Pathology 

**Title (ZH)**: 学习驱动的图像压缩与恢复在数字病理学中应用 

**Authors**: SeonYeong Lee, EonSeung Seong, DongEon Lee, SiYeoul Lee, Yubin Cho, Chunsu Park, Seonho Kim, MinKyoung Seo, YoungSin Ko, MinWoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.23862)  

**Abstract**: Digital pathology images play a crucial role in medical diagnostics, but their ultra-high resolution and large file sizes pose significant challenges for storage, transmission, and real-time visualization. To address these issues, we propose CLERIC, a novel deep learning-based image compression framework designed specifically for whole slide images (WSIs). CLERIC integrates a learnable lifting scheme and advanced convolutional techniques to enhance compression efficiency while preserving critical pathological details. Our framework employs a lifting-scheme transform in the analysis stage to decompose images into low- and high-frequency components, enabling more structured latent representations. These components are processed through parallel encoders incorporating Deformable Residual Blocks (DRB) and Recurrent Residual Blocks (R2B) to improve feature extraction and spatial adaptability. The synthesis stage applies an inverse lifting transform for effective image reconstruction, ensuring high-fidelity restoration of fine-grained tissue structures. We evaluate CLERIC on a digital pathology image dataset and compare its performance against state-of-the-art learned image compression (LIC) models. Experimental results demonstrate that CLERIC achieves superior rate-distortion (RD) performance, significantly reducing storage requirements while maintaining high diagnostic image quality. Our study highlights the potential of deep learning-based compression in digital pathology, facilitating efficient data management and long-term storage while ensuring seamless integration into clinical workflows and AI-assisted diagnostic systems. Code and models are available at: this https URL. 

**Abstract (ZH)**: 数字病理图像在医疗诊断中扮演着至关重要的角色，但其超高的分辨率和庞大的文件大小给存储、传输和实时可视化带来了重大挑战。为应对这些挑战，我们提出了一种名为CLERIC的新型基于深度学习的图像压缩框架，专门针对全切片图像（WSIs）。CLERIC结合了可学习提升方案和先进的卷积技术，以提高压缩效率同时保留关键的病理细节。该框架在分析阶段采用提升方案变换将图像分解为低频和高频组件，使其能够生成更具结构化的潜在表示。这些组件通过包含可变形残差块（DRB）和循环残差块（R2B）的并行编码器进行处理，以改善特征提取和空间适应性。合成功态应用逆提升变换进行有效的图像重建，确保细粒度组织结构的高保真恢复。我们在数字病理图像数据集上评估了CLERIC，并将其性能与最先进的学习图像压缩（LIC）模型进行比较。实验结果表明，CLERIC在率失真（RD）性能方面表现出色，显著减少了存储需求同时保持了高诊断图像质量。我们的研究突显了基于深度学习的压缩在数字病理学中的潜在价值，有助于实现高效的数据管理、长期存储以及与临床工作流程和AI辅助诊断系统的无缝集成。代码和模型可在以下链接获取：this https URL。 

---
# OrchMLLM: Orchestrate Multimodal Data with Batch Post-Balancing to Accelerate Multimodal Large Language Model Training 

**Title (ZH)**: OrchMLLM: 采用批量后均衡技术 orchestrating 多模态数据以加速多模态大型语言模型训练 

**Authors**: Yijie Zheng, Bangjun Xiao, Lei Shi, Xiaoyang Li, Faming Wu, Tianyu Li, Xuefeng Xiao, Yang Zhang, Yuxuan Wang, Shouda Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23830)  

**Abstract**: Multimodal large language models (MLLMs), such as GPT-4o, are garnering significant attention. During the exploration of MLLM training, we identified Modality Composition Incoherence, a phenomenon that the proportion of a certain modality varies dramatically across different examples. It exacerbates the challenges of addressing mini-batch imbalances, which lead to uneven GPU utilization between Data Parallel (DP) instances and severely degrades the efficiency and scalability of MLLM training, ultimately affecting training speed and hindering further research on MLLMs.
To address these challenges, we introduce OrchMLLM, a comprehensive framework designed to mitigate the inefficiencies in MLLM training caused by Modality Composition Incoherence. First, we propose Batch Post-Balancing Dispatcher, a technique that efficiently eliminates mini-batch imbalances in sequential data. Additionally, we integrate MLLM Global Orchestrator into the training framework to orchestrate multimodal data and tackle the issues arising from Modality Composition Incoherence. We evaluate OrchMLLM across various MLLM sizes, demonstrating its efficiency and scalability. Experimental results reveal that OrchMLLM achieves a Model FLOPs Utilization (MFU) of $41.6\%$ when training an 84B MLLM with three modalities on $2560$ H100 GPUs, outperforming Megatron-LM by up to $3.1\times$ in throughput. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）如GPT-4o正引起广泛关注。在探索MLLM训练过程中，我们识别出模态组成不一致性现象，即某些模态的比例在不同示例中急剧变化。这加剧了小批次不平衡的挑战，导致数据并行（DP）实例间的GPU利用率不均衡，严重降低了MLLM训练的效率和可扩展性，最终影响训练速度并阻碍进一步的MLLM研究。

为应对这些挑战，我们提出了OrchMLLM，这是一种全面的框架，旨在缓解由模态组成不一致性引起的MLLM训练效率低下问题。首先，我们提出批后不平衡调度器，一种高效消除序列数据中小批次不平衡的技术。此外，我们将MLLM全局协调器集成到训练框架中，协调多模态数据并解决由模态组成不一致性引起的问题。我们在多种MLLM规模上评估了OrchMLLM，展示了其效率和可扩展性。实验结果表明，当使用2560个H100 GPU训练一个包含三种模态的84B MLLM时，OrchMLLM的模型FLOPs利用率（MFU）达到41.6%，吞吐量最高可比Megatron-LM提高3.1倍。 

---
# When Counterfactual Reasoning Fails: Chaos and Real-World Complexity 

**Title (ZH)**: 当反事实推理失效：混沌与现实世界复杂性 

**Authors**: Yahya Aalaila, Gerrit Großmann, Sumantrak Mukherjee, Jonas Wahl, Sebastian Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2503.23820)  

**Abstract**: Counterfactual reasoning, a cornerstone of human cognition and decision-making, is often seen as the 'holy grail' of causal learning, with applications ranging from interpreting machine learning models to promoting algorithmic fairness. While counterfactual reasoning has been extensively studied in contexts where the underlying causal model is well-defined, real-world causal modeling is often hindered by model and parameter uncertainty, observational noise, and chaotic behavior. The reliability of counterfactual analysis in such settings remains largely unexplored. In this work, we investigate the limitations of counterfactual reasoning within the framework of Structural Causal Models. Specifically, we empirically investigate \emph{counterfactual sequence estimation} and highlight cases where it becomes increasingly unreliable. We find that realistic assumptions, such as low degrees of model uncertainty or chaotic dynamics, can result in counterintuitive outcomes, including dramatic deviations between predicted and true counterfactual trajectories. This work urges caution when applying counterfactual reasoning in settings characterized by chaos and uncertainty. Furthermore, it raises the question of whether certain systems may pose fundamental limitations on the ability to answer counterfactual questions about their behavior. 

**Abstract (ZH)**: 基于结构因果模型的反事实推理限制：混沌和不确定性下的应用探究 

---
# Conformal uncertainty quantification to evaluate predictive fairness of foundation AI model for skin lesion classes across patient demographics 

**Title (ZH)**: 符合患者人群分布的皮肤病变类别基础AI模型预测公平性的一致性不确定性量化评估 

**Authors**: Swarnava Bhattacharyya, Umapada Pal, Tapabrata Chakraborti  

**Link**: [PDF](https://arxiv.org/pdf/2503.23819)  

**Abstract**: Deep learning based diagnostic AI systems based on medical images are starting to provide similar performance as human experts. However these data hungry complex systems are inherently black boxes and therefore slow to be adopted for high risk applications like healthcare. This problem of lack of transparency is exacerbated in the case of recent large foundation models, which are trained in a self supervised manner on millions of data points to provide robust generalisation across a range of downstream tasks, but the embeddings generated from them happen through a process that is not interpretable, and hence not easily trustable for clinical applications. To address this timely issue, we deploy conformal analysis to quantify the predictive uncertainty of a vision transformer (ViT) based foundation model across patient demographics with respect to sex, age and ethnicity for the tasks of skin lesion classification using several public benchmark datasets. The significant advantage of this method is that conformal analysis is method independent and it not only provides a coverage guarantee at population level but also provides an uncertainty score for each individual. We used a model-agnostic dynamic F1-score-based sampling during model training, which helped to stabilize the class imbalance and we investigate the effects on uncertainty quantification (UQ) with or without this bias mitigation step. Thus we show how this can be used as a fairness metric to evaluate the robustness of the feature embeddings of the foundation model (Google DermFoundation) and thus advance the trustworthiness and fairness of clinical AI. 

**Abstract (ZH)**: 基于深度学习的医学图像诊断AI系统在某些方面已达到人类专家的性能水平。然而，这些对数据需求大且内部机制不透明的复杂系统，在应用于高风险领域如医疗保健时，推广速度较慢。尤其是对于近期训练于大量数据点并提供跨多种下游任务稳健泛化的大型自监督基础模型，其生成的嵌入表示过程不具可解释性，这在临床应用中难以建立信任。为解决这一紧迫问题，我们采用了容错分析方法，对基于视觉变换器（ViT）的基础模型在多种公开基准数据集上的皮肤病变分类任务中，按性别、年龄和种族不同患者群体的预测不确定性进行了定量分析。这种方法的主要优势在于，容错分析方法与模型无关，不仅在群体水平上提供了覆盖保证，还为每个个体提供了不确定性评分。我们在模型训练中采用了一种模型无关的动力学F1分数采样方法，有助于稳定类别不平衡问题，并研究了此偏差缓解步骤对不确定性量化（UQ）的影响。我们展示了如何使用这种公平性指标来评估基础模型（Google DermFoundation）的特征嵌入的稳健性，从而提高临床AI的信任度和公平性。 

---
# Thinking Longer, Not Larger: Enhancing Software Engineering Agents via Scaling Test-Time Compute 

**Title (ZH)**: 思考更久，而不是更大：通过扩展测试时计算提升软件工程代理 

**Authors**: Yingwei Ma, Binhua Li, Yihong Dong, Xue Jiang, Rongyu Cao, Jue Chen, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.23803)  

**Abstract**: Recent advancements in software engineering agents have demonstrated promising capabilities in automating program improvements. However, their reliance on closed-source or resource-intensive models introduces significant deployment challenges in private environments, prompting a critical question: \textit{How can personally deployable open-source LLMs achieve comparable code reasoning performance?}
To this end, we propose a unified Test-Time Compute scaling framework that leverages increased inference-time computation instead of larger models. Our framework incorporates two complementary strategies: internal TTC and external TTC. Internally, we introduce a \textit{development-contextualized trajectory synthesis} method leveraging real-world software repositories to bootstrap multi-stage reasoning processes, such as fault localization and patch generation. We further enhance trajectory quality through rejection sampling, rigorously evaluating trajectories along accuracy and complexity. Externally, we propose a novel \textit{development-process-based search} strategy guided by reward models and execution verification. This approach enables targeted computational allocation at critical development decision points, overcoming limitations of existing "end-point only" verification methods.
Evaluations on SWE-bench Verified demonstrate our \textbf{32B model achieves a 46\% issue resolution rate}, surpassing significantly larger models such as DeepSeek R1 671B and OpenAI o1. Additionally, we provide the empirical validation of the test-time scaling phenomenon within SWE agents, revealing that \textbf{models dynamically allocate more tokens to increasingly challenging problems}, effectively enhancing reasoning capabilities. We publicly release all training data, models, and code to facilitate future research. this https URL 

**Abstract (ZH)**: 近年来，软件工程代理在自动化程序改进方面展示了令人瞩目的能力。然而，它们对闭源或资源密集型模型的依赖性在私有环境中引发了重要的部署挑战，促使我们提出一个关键问题：\textit{如何实现可个人部署的开源大语言模型以获得相当的代码推理性能？}

为此，我们提出一种统一的测试时计算扩展框架，该框架利用增加的推理时计算量而不是更大的模型。我们的框架包含两种互补策略：内部 TTC 和外部 TTC。内部，我们引入一种基于实际软件仓库的开发上下文轨迹合成方法，以启动多阶段推理过程，如故障定位和补丁生成。我们进一步通过拒绝采样提高轨迹质量，并严格评估其准确性和复杂性。外部，我们提出了一种基于开发过程的新型搜索策略，该策略受到奖励模型和执行验证的引导。这种方法能够在关键的开发决策点进行有针对性的计算分配，克服了现有“端点验证”方法的局限性。

在 SWE-bench 验证上，我们的 \textbf{32B 模型实现了46%的问题解决率}，显著超过了如 DeepSeek R1 671B 和 OpenAI o1 这样的更大模型。此外，我们还在 SWE 代理中提供了测试时计算扩展现象的实证验证，揭示了模型如何动态地将更多令牌分配给越来越具有挑战性的问题，从而有效提升推理能力。我们公开发布所有训练数据、模型和代码，以促进未来的研究。更多详情请参见：此链接。 

---
# Adaptive Layer-skipping in Pre-trained LLMs 

**Title (ZH)**: 预训练大规模语言模型中的自适应层跳过技术支持 

**Authors**: Xuan Luo, Weizhi Wang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.23798)  

**Abstract**: Various layer-skipping methods have been proposed to accelerate token generation in large language models (LLMs). However, they have overlooked a fundamental question: How do computational demands vary across the generation of different tokens? In this work, we introduce FlexiDepth, a method that dynamically adjusts the number of Transformer layers used in text generation. By incorporating a plug-in router and adapter, FlexiDepth enables adaptive layer-skipping in LLMs without modifying their original parameters. Introducing FlexiDepth to Llama-3-8B model achieves layer skipping of 8 layers out of 32, and meanwhile maintains the full 100\% benchmark performance. Experimental results with FlexiDepth demonstrate that computational demands in LLMs significantly vary based on token type. Specifically, generating repetitive tokens or fixed phrases requires fewer layers, whereas producing tokens involving computation or high uncertainty requires more layers. Interestingly, this adaptive allocation pattern aligns with human intuition. To advance research in this area, we open sourced FlexiDepth and a dataset documenting FlexiDepth's layer allocation patterns for future exploration. 

**Abstract (ZH)**: FlexiDepth: 动态调整Transformer层数以适应不同tokens的计算需求 

---
# MGD-SAM2: Multi-view Guided Detail-enhanced Segment Anything Model 2 for High-Resolution Class-agnostic Segmentation 

**Title (ZH)**: MGD-SAM2：多视图引导细节增强的通用分割模型2for高分辨率无类别分割 

**Authors**: Haoran Shen, Peixian Zhuang, Jiahao Kou, Yuxin Zeng, Haoying Xu, Jiangyun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.23786)  

**Abstract**: Segment Anything Models (SAMs), as vision foundation models, have demonstrated remarkable performance across various image analysis tasks. Despite their strong generalization capabilities, SAMs encounter challenges in fine-grained detail segmentation for high-resolution class-independent segmentation (HRCS), due to the limitations in the direct processing of high-resolution inputs and low-resolution mask predictions, and the reliance on accurate manual prompts. To address these limitations, we propose MGD-SAM2 which integrates SAM2 with multi-view feature interaction between a global image and local patches to achieve precise segmentation. MGD-SAM2 incorporates the pre-trained SAM2 with four novel modules: the Multi-view Perception Adapter (MPAdapter), the Multi-view Complementary Enhancement Module (MCEM), the Hierarchical Multi-view Interaction Module (HMIM), and the Detail Refinement Module (DRM). Specifically, we first introduce MPAdapter to adapt the SAM2 encoder for enhanced extraction of local details and global semantics in HRCS images. Then, MCEM and HMIM are proposed to further exploit local texture and global context by aggregating multi-view features within and across multi-scales. Finally, DRM is designed to generate gradually restored high-resolution mask predictions, compensating for the loss of fine-grained details resulting from directly upsampling the low-resolution prediction maps. Experimental results demonstrate the superior performance and strong generalization of our model on multiple high-resolution and normal-resolution datasets. Code will be available at this https URL. 

**Abstract (ZH)**: Segment Anything Models (SAMs) 在各种图像分析任务中展示了 remarkable 的表现。尽管 SAMs 具有强大的泛化能力，但在高分辨率类内分割（HRCS）任务中，它们在精细细节分割方面仍面临挑战，这主要是由于高分辨率输入的直接处理能力有限、低分辨率掩码预测的准确度不足以及对精确手动提示的依赖。为了解决这些局限性，我们提出了 MGD-SAM2，该模型将 SAM2 与全局图像和局部片段的多视图特征交互相结合，以实现精确分割。MGD-SAM2 结合预训练的 SAM2 和四个新颖模块：多视图感知适配器 (MPAdapter)、多视图互补增强模块 (MCEM)、分层多视图交互模块 (HMIM) 和细节精炼模块 (DRM)。具体而言，我们首先引入 MPAdapter 以增强 SAM2 编码器在 HRCS 图像中局部细节和全局语义的提取。然后，提出 MCEM 和 HMIM 通过在不同尺度内和跨尺度聚合多视图特征，进一步利用局部纹理和全局上下文。最后，设计 DRM 生成逐步恢复的高分辨率掩码预测，以补偿直接上采样低分辨率预测图造成的精细细节损失。实验结果表明，我们的模型在多个高分辨率和正常分辨率数据集上的性能和泛化能力优越。代码将在以下网址公开：this https URL。 

---
# WinoWhat: A Parallel Corpus of Paraphrased WinoGrande Sentences with Common Sense Categorization 

**Title (ZH)**: Winogradwhat：一个具有常识分类的并行改写Winograd Grande句子语料库 

**Authors**: Ine Gevers, Victor De Marez, Luna De Bruyne, Walter Daelemans  

**Link**: [PDF](https://arxiv.org/pdf/2503.23779)  

**Abstract**: In this study, we take a closer look at how Winograd schema challenges can be used to evaluate common sense reasoning in LLMs. Specifically, we evaluate generative models of different sizes on the popular WinoGrande benchmark. We release WinoWhat, a new corpus, in which each instance of the WinoGrande validation set is paraphrased. Additionally, we evaluate the performance on the challenge across five common sense knowledge categories, giving more fine-grained insights on what types of knowledge are more challenging for LLMs. Surprisingly, all models perform significantly worse on WinoWhat, implying that LLM reasoning capabilities are overestimated on WinoGrande. To verify whether this is an effect of benchmark memorization, we match benchmark instances to LLM trainingdata and create two test-suites. We observe that memorization has a minimal effect on model performance on WinoGrande. 

**Abstract (ZH)**: 本研究更深入地探讨了WinogradSchema挑战如何用于评估大规模语言模型的常识推理能力。具体而言，我们在流行的WinoGrande基准上评估了不同规模的生成模型。我们发布了WinoWhat数据集，其中每个WinoGrande验证集的实例都被重新表述。此外，我们在五个常识知识类别上评估了挑战的表现，提供了更细致的见解，了解哪些类型的知识对语言模型更具挑战性。令人惊讶的是，所有模型在WinoWhat上的表现显著较差，这表明在WinoGrande上的语言模型推理能力可能被高估了。为了验证这是否是由于基准记忆效应，我们将基准实例与语言模型训练数据匹配，并创建了两个测试套件。我们观察到，记忆对WinoGrande上模型性能的影响较小。 

---
# WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation for Efficient Medical Image Segmentation 

**Title (ZH)**: WaveFormer：一种基于小波驱动特征表示的高效医学图像分割3DTransformer 

**Authors**: Md Mahfuz Al Hasan, Mahdi Zaman, Abdul Jawad, Alberto Santamaria-Pang, Ho Hin Lee, Ivan Tarapov, Kyle See, Md Shah Imran, Antika Roy, Yaser Pourmohammadi Fallah, Navid Asadizanjani, Reza Forghani  

**Link**: [PDF](https://arxiv.org/pdf/2503.23764)  

**Abstract**: Transformer-based architectures have advanced medical image analysis by effectively modeling long-range dependencies, yet they often struggle in 3D settings due to substantial memory overhead and insufficient capture of fine-grained local features. We address these limi- tations with WaveFormer, a novel 3D-transformer that: i) leverages the fundamental frequency-domain properties of features for contextual rep- resentation, and ii) is inspired by the top-down mechanism of the human visual recognition system, making it a biologically motivated architec- ture. By employing discrete wavelet transformations (DWT) at multiple scales, WaveFormer preserves both global context and high-frequency de- tails while replacing heavy upsampling layers with efficient wavelet-based summarization and reconstruction. This significantly reduces the number of parameters, which is critical for real-world deployment where compu- tational resources and training times are constrained. Furthermore, the model is generic and easily adaptable to diverse applications. Evaluations on BraTS2023, FLARE2021, and KiTS2023 demonstrate performance on par with state-of-the-art methods while offering substantially lower computational complexity. 

**Abstract (ZH)**: 基于变换器的架构通过有效地建模长距离依赖性，促进了医学图像分析，但往往在3D环境中由于内存开销巨大且难以捕捉细致的局部特征而受限。我们通过提出一种新型的3D变换器WaveFormer来克服这些限制：i) 利用特征的基本频域特性进行上下文表示；ii) 受人类视觉识别系统自上而下机制的启发，使其成为一种生物动机型架构。通过在多个尺度上采用离散小波变换（DWT），WaveFormer既能保持全局上下文又能捕捉高频细节，同时用高效的小波基总结和重建替代了复杂的上采样层，极大地减少了参数数量，这对于计算资源和训练时间受限的实际部署至关重要。此外，该模型是通用的且易于适应各种应用。在BraTS2023、FLARE2021和KiTS2023上的评估显示，其性能与最先进的方法相当，而计算复杂度大幅降低。 

---
# LANID: LLM-assisted New Intent Discovery 

**Title (ZH)**: LANID: LLM-assisted新型意图发现 

**Authors**: Lu Fan, Jiashu Pu, Rongsheng Zhang, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23740)  

**Abstract**: Task-oriented Dialogue Systems (TODS) often face the challenge of encountering new intents. New Intent Discovery (NID) is a crucial task that aims to identify these novel intents while maintaining the capability to recognize existing ones. Previous efforts to adapt TODS to new intents have struggled with inadequate semantic representation or have depended on external knowledge, which is often not scalable or flexible. Recently, Large Language Models (LLMs) have demonstrated strong zero-shot capabilities; however, their scale can be impractical for real-world applications that involve extensive queries. To address the limitations of existing NID methods by leveraging LLMs, we propose LANID, a framework that enhances the semantic representation of lightweight NID encoders with the guidance of LLMs. Specifically, LANID employs the $K$-nearest neighbors and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithms to sample selective utterance pairs from the training set. It then queries an LLM to ascertain the relationships between these pairs. The data produced from this process is utilized to design a contrastive fine-tuning task, which is then used to train a small encoder with a contrastive triplet loss. Our experimental results demonstrate the efficacy of the proposed method across three distinct NID datasets, surpassing strong baselines in both unsupervised and semi-supervised settings. Our code is available at this https URL. 

**Abstract (ZH)**: 基于任务的对话系统中新意图发现（LANID） 

---
# Investigation of intelligent barbell squat coaching system based on computer vision and machine learning 

**Title (ZH)**: 基于计算机视觉和机器学习的智能杠铃深蹲指导系统研究 

**Authors**: Yinq-Rong Chern, Yuhao Lee, Hsiao-Ching Lin, Guan-Ting Chen, Ying-Hsien Chen, Fu-Sung Lin, Chih-Yao Chuang, Jenn-Jier James Lien, Chih-Hsien Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23731)  

**Abstract**: Purpose: Research has revealed that strength training can reduce the incidence of chronic diseases and physical deterioration at any age. Therefore, having a movement diagnostic system is crucial for training alone. Hence, this study developed an artificial intelligence and computer vision-based barbell squat coaching system with a real-time mode that immediately diagnoses the issue and provides feedback after each squat. In addition, a replay mode allows users to examine their previous squats and check their comments. Initially, four primary characteristics of the barbell squat were identified: body joint angles, dorsiflexion, the ratio of knee-to-hip movement, and barbell stability. Methods: We collect 8,151 squats from 77 participants, categorizing them as good squats and six issues. Then, we trained the diagnosis models with three machine-learning architectures. Furthermore, this research applied the SHapley Additive exPlanations (SHAP) method to enhance the accuracy of issue prediction and reduce the computation time by feature selection. Results: The F1 score of the six issues reached 86.86%, 69.01%, 77.42%, 90.74%, 95.83%, and 100%. Each squat diagnosis took less than 0.5 seconds. Finally, this study examined the efficacy of the proposed system with two groups of participants trained with and without the system. Subsequently, participants trained with the system exhibited substantial improvements in their squat technique, as assessed both by the system itself and by a professional weightlifting coach. Conclusion: This is a comprehensive study that integrates artificial intelligence, computer vision and multivariable processing technologies, aimed at building a real-time, user-friendly barbell squat feedback and training system. 

**Abstract (ZH)**: 目的: 研究表明，力量训练可以减少任何年龄段慢性疾病和身体退化的发生率。因此，拥有一个运动诊断系统对于训练至关重要。故此研究开发了一个基于人工 intelligence 和计算机视觉的杠铃深蹲指导系统，该系统具有实时模式，能够在每次深蹲后立即诊断问题并提供反馈。此外，录制模式允许用户回顾之前的深蹲并检查评论。最初，确定了杠铃深蹲的四个主要特征：身体关节角、 dorsiflexion、膝髋运动比例以及杠铃稳定性。方法: 收集了来自77名参与者的8,151次深蹲动作，并将其分类为良好深蹲和六个问题。然后，用三种机器学习架构训练诊断模型。此外，本研究使用 SHapley Additive exPlanations (SHAP) 方法以通过特征选择提高问题预测的准确性并减少计算时间。结果: 六个问题的 F1 分数分别为 86.86%、69.01%、77.42%、90.74%、95.83% 和 100%。每次深蹲诊断时间少于 0.5 秒。最后，本研究通过两组使用系统和未使用系统的参与者来检验所提系统的有效性。结果显示，使用系统的参与者在深蹲技术上表现出显著的进步，无论是系统的评估还是专业举重教练的评估都如此。结论: 本研究是一个综合性的研究，集成了人工智能、计算机视觉和多变量处理技术，旨在建立一个实时、用户友好的杠铃深蹲反馈与训练系统。 

---
# KOFFVQA: An Objectively Evaluated Free-form VQA Benchmark for Large Vision-Language Models in the Korean Language 

**Title (ZH)**: KOFFVQA：韩语文本的大规模视觉语言模型客观评价自由形式问答基准 

**Authors**: Yoonshik Kim, Jaeyoon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2503.23730)  

**Abstract**: The recent emergence of Large Vision-Language Models(VLMs) has resulted in a variety of different benchmarks for evaluating such models. Despite this, we observe that most existing evaluation methods suffer from the fact that they either require the model to choose from pre-determined responses, sacrificing open-endedness, or evaluate responses using a judge model, resulting in subjective and unreliable evaluation. In addition, we observe a lack of benchmarks for VLMs in the Korean language, which are necessary as a separate metric from more common English language benchmarks, as the performance of generative language models can differ significantly based on the language being used. Therefore, we present KOFFVQA, a general-purpose free-form visual question answering benchmark in the Korean language for the evaluation of VLMs. Our benchmark consists of 275 carefully crafted questions each paired with an image and grading criteria covering 10 different aspects of VLM performance. The grading criteria eliminate the problem of unreliability by allowing the judge model to grade each response based on a pre-determined set of rules. By defining the evaluation criteria in an objective manner, even a small open-source model can be used to evaluate models on our benchmark reliably. In addition to evaluating a large number of existing VLMs on our benchmark, we also experimentally verify that our method of using pre-existing grading criteria for evaluation is much more reliable than existing methods. Our evaluation code is available at this https URL 

**Abstract (ZH)**: Recent 出现的大规模视觉-语言模型(VLMs)为评估这类模型带来了多种不同的基准。尽管如此，我们观察到大多数现有的评估方法存在以下问题：要么要求模型从预定义的回答中选择，牺牲了开放性；要么使用裁判模型评估回答，导致主观且不可靠的评估。此外，我们还观察到关于韩语的大规模视觉-语言模型(VLMs)基准不足，作为与更常见的英语基准分离的指标是必要的，因为生成语言模型的性能会根据使用的语言有很大差异。因此，我们提出了KOFFVQA，一种用于大规模视觉-语言模型评估的韩语通用开放式视觉问答基准。我们的基准包括275个精心设计的问题，每个问题配有一张图片和涵盖10个方面的大规模视觉-语言模型性能评估标准。这些评估标准通过允许裁判模型根据预定义的规则对每个回答进行评分来解决不可靠性问题。通过以客观的方式定义评估标准，即使是小型开源模型也可以可靠地在我们的基准上评估模型。除了在我们的基准上评估大量现有的大规模视觉-语言模型外，我们还实验证明，我们使用现成的评分标准进行评估的方法比现有方法要可靠得多。我们的评估代码可在以下网址获得。 

---
# Unimodal-driven Distillation in Multimodal Emotion Recognition with Dynamic Fusion 

**Title (ZH)**: 单模态驱动的多模态情感识别动态融合蒸馏 

**Authors**: Jiagen Li, Rui Yu, Huihao Huang, Huaicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.23721)  

**Abstract**: Multimodal Emotion Recognition in Conversations (MERC) identifies emotional states across text, audio and video, which is essential for intelligent dialogue systems and opinion analysis. Existing methods emphasize heterogeneous modal fusion directly for cross-modal integration, but often suffer from disorientation in multimodal learning due to modal heterogeneity and lack of instructive guidance. In this work, we propose SUMMER, a novel heterogeneous multimodal integration framework leveraging Mixture of Experts with Hierarchical Cross-modal Fusion and Interactive Knowledge Distillation. Key components include a Sparse Dynamic Mixture of Experts (SDMoE) for capturing dynamic token-wise interactions, a Hierarchical Cross-Modal Fusion (HCMF) for effective fusion of heterogeneous modalities, and Interactive Knowledge Distillation (IKD), which uses a pre-trained unimodal teacher to guide multimodal fusion in latent and logit spaces. Experiments on IEMOCAP and MELD show SUMMER outperforms state-of-the-art methods, particularly in recognizing minority and semantically similar emotions. 

**Abstract (ZH)**: 多模态对话情感识别（MERC）识别跨文本、音频和视频的情感状态，对于智能对话系统和观点分析至关重要。现有的方法强调直接进行异模态融合以实现跨模态集成，但由于模态异质性导致的模态学习方向混乱问题，常常缺乏有效的指导。本文提出SUMMER，一种新颖的异模态融合框架，利用混合专家与层次跨模态融合及交互式知识蒸馏。关键组件包括稀疏动态混合专家（SDMoE）以捕捉动态的令牌级交互，层次跨模态融合（HCMF）以有效融合异质模态，以及交互式知识蒸馏（IKD），通过预训练的单模态教师在潜在空间和逻辑空间指导异模态融合。实验结果表明，SUMMER在IEMOCAP和MELD数据集上优于现有方法，特别是在识别少数和语义相似情感方面。 

---
# GNN-Based Candidate Node Predictor for Influence Maximization in Temporal Graphs 

**Title (ZH)**: 基于GNN的时序图影响最大化候选节点预测器 

**Authors**: Priyanka Gautam, Balasubramaniam Natarajan, Sai Munikoti, S M Ferdous, Mahantesh Halappanavar  

**Link**: [PDF](https://arxiv.org/pdf/2503.23713)  

**Abstract**: In an age where information spreads rapidly across social media, effectively identifying influential nodes in dynamic networks is critical. Traditional influence maximization strategies often fail to keep up with rapidly evolving relationships and structures, leading to missed opportunities and inefficiencies. To address this, we propose a novel learning-based approach integrating Graph Neural Networks (GNNs) with Bidirectional Long Short-Term Memory (BiLSTM) models. This hybrid framework captures both structural and temporal dynamics, enabling accurate prediction of candidate nodes for seed set selection. The bidirectional nature of BiLSTM allows our model to analyze patterns from both past and future network states, ensuring adaptability to changes over time. By dynamically adapting to graph evolution at each time snapshot, our approach improves seed set calculation efficiency, achieving an average of 90% accuracy in predicting potential seed nodes across diverse networks. This significantly reduces computational overhead by optimizing the number of nodes evaluated for seed selection. Our method is particularly effective in fields like viral marketing and social network analysis, where understanding temporal dynamics is crucial. 

**Abstract (ZH)**: 在信息快速通过社交媒体传播的时代，动态网络中重要节点的有效识别至关重要。传统的影响力最大化策略往往无法跟上迅速变化的关系和结构，导致错失机会和低效。为解决这一问题，我们提出了一种结合图神经网络（GNN）和双向长短期记忆（BiLSTM）模型的新型学习方法。该混合框架捕捉了结构和时序动态，使得能够准确预测候选节点以供种子集选择。BiLSTM的双向性质使模型能够分析过去和未来的网络状态模式，确保随着时间变化的适应性。通过在每个时间切片上动态适应图演变，我们的方法提高了种子集计算效率，在多种网络中平均准确率达到90%的潜在种子节点预测。这种方法大幅减少了计算开销，通过优化种子选择过程中的节点评估数量。我们的方法特别适用于病毒营销和社会网络分析等领域，其中了解时序动态至关重要。 

---
# Towards Benchmarking and Assessing the Safety and Robustness of Autonomous Driving on Safety-critical Scenarios 

**Title (ZH)**: 面向安全关键场景下自动驾驶的安全性和鲁棒性基准测试与评估 

**Authors**: Jingzheng Li, Xianglong Liu, Shikui Wei, Zhijun Chen, Bing Li, Qing Guo, Xianqi Yang, Yanjun Pu, Jiakai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23708)  

**Abstract**: Autonomous driving has made significant progress in both academia and industry, including performance improvements in perception task and the development of end-to-end autonomous driving systems. However, the safety and robustness assessment of autonomous driving has not received sufficient attention. Current evaluations of autonomous driving are typically conducted in natural driving scenarios. However, many accidents often occur in edge cases, also known as safety-critical scenarios. These safety-critical scenarios are difficult to collect, and there is currently no clear definition of what constitutes a safety-critical scenario. In this work, we explore the safety and robustness of autonomous driving in safety-critical scenarios. First, we provide a definition of safety-critical scenarios, including static traffic scenarios such as adversarial attack scenarios and natural distribution shifts, as well as dynamic traffic scenarios such as accident scenarios. Then, we develop an autonomous driving safety testing platform to comprehensively evaluate autonomous driving systems, encompassing not only the assessment of perception modules but also system-level evaluations. Our work systematically constructs a safety verification process for autonomous driving, providing technical support for the industry to establish standardized test framework and reduce risks in real-world road deployment. 

**Abstract (ZH)**: 自主驾驶在安全关键场景中的安全与鲁棒性评估 

---
# Remarks on the Polyak-Lojasiewicz inequality and the convergence of gradient systems 

**Title (ZH)**: 关于Polyak-Lojasiewicz不等式的一些注记及梯度系统收敛性的研究 

**Authors**: Arthur Castello B. de Oliveira, Leilei Cui, Eduardo D. Sontag  

**Link**: [PDF](https://arxiv.org/pdf/2503.23641)  

**Abstract**: This work explores generalizations of the Polyak-Lojasiewicz inequality (PLI) and their implications for the convergence behavior of gradient flows in optimization problems. Motivated by the continuous-time linear quadratic regulator (CT-LQR) policy optimization problem -- where only a weaker version of the PLI is characterized in the literature -- this work shows that while weaker conditions are sufficient for global convergence to, and optimality of the set of critical points of the cost function, the "profile" of the gradient flow solution can change significantly depending on which "flavor" of inequality the cost satisfies. After a general theoretical analysis, we focus on fitting the CT-LQR policy optimization problem to the proposed framework, showing that, in fact, it can never satisfy a PLI in its strongest form. We follow up our analysis with a brief discussion on the difference between continuous- and discrete-time LQR policy optimization, and end the paper with some intuition on the extension of this framework to optimization problems with L1 regularization and solved through proximal gradient flows. 

**Abstract (ZH)**: 这项工作探讨了Polyak-Lojasiewicz不等式（PLI）的一般化及其对优化问题中梯度流收敛行为的影响。受连续时间线性二次调节器（CT-LQR）策略优化问题的启发——在文献中仅描述了较弱版本的PLI——这项工作表明，虽然较弱的条件对于全局收敛到成本函数的临界点及其最优解是足够的，但成本函数满足的“不等式风味”不同，其梯度流解的“轮廓”可能会有显著变化。在一般理论分析之后，我们将注意力集中在将CT-LQR策略优化问题拟合到所提出的框架上，结果显示实际上它不可能满足PLI的最严格形式。随后，我们简要讨论了连续时间和离散时间LQR策略优化之间的差异，并在论文结尾对将该框架扩展到带有L1正则化并通过近端梯度流求解的优化问题进行了一些直观解释。 

---
# Finding Interest Needle in Popularity Haystack: Improving Retrieval by Modeling Item Exposure 

**Title (ZH)**: 在流行度haystack中寻找兴趣针：通过建模项目曝光改善检索 

**Authors**: Amit Jaspal, Rahul Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2503.23630)  

**Abstract**: Recommender systems operate in closed feedback loops, where user interactions reinforce popularity bias, leading to over-recommendation of already popular items while under-exposing niche or novel content. Existing bias mitigation methods, such as Inverse Propensity Scoring (IPS) and Off- Policy Correction (OPC), primarily operate at the ranking stage or during training, lacking explicit real-time control over exposure dynamics. In this work, we introduce an exposure- aware retrieval scoring approach, which explicitly models item exposure probability and adjusts retrieval-stage ranking at inference time. Unlike prior work, this method decouples exposure effects from engagement likelihood, enabling controlled trade-offs between fairness and engagement in large-scale recommendation platforms. We validate our approach through online A/B experiments in a real-world video recommendation system, demonstrating a 25% increase in uniquely retrieved items and a 40% reduction in the dominance of over-popular content, all while maintaining overall user engagement levels. Our results establish a scalable, deployable solution for mitigating popularity bias at the retrieval stage, offering a new paradigm for bias-aware personalization. 

**Abstract (ZH)**: 推荐系统在封闭的反馈循环中运作，用户交互强化了流行性偏差，导致过度推荐已有流行项目，而限制了小众或新颖内容的曝光。现有的偏差缓解方法，如逆倾向评分（IPS）和离策训练修正（OPC），主要在排序阶段或训练过程中运作，缺乏对曝光动态的显式实时控制。在本文中，我们引入了一种 Awareness 意识下的检索评分方法，该方法明确建模项目曝光概率，并在推理时调整检索阶段的排序。与之前的工作不同，该方法将曝光效应与参与可能性脱钩，能够在大规模推荐平台上实现公平性和参与性的可控权衡。我们通过在真实世界视频推荐系统中的在线 A/B 实验验证了该方法，结果显示独特检索项目的增加幅度达到了 25%，过度流行内容的主导性降低了 40%，同时保持了整体用户参与度水平。实验结果建立了一种可扩展且可部署的在检索阶段缓解流行性偏差的解决方案，为一种新的意识下偏差感知个性化提供了新范式。 

---
# Graph-Eq: Discovering Mathematical Equations using Graph Generative Models 

**Title (ZH)**: Graph-Eq: 使用图生成模型发现数学方程 

**Authors**: Nisal Ranasinghe, Damith Senanayake, Saman Halgamuge  

**Link**: [PDF](https://arxiv.org/pdf/2503.23617)  

**Abstract**: The ability to discover meaningful, accurate, and concise mathematical equations that describe datasets is valuable across various domains. Equations offer explicit relationships between variables, enabling deeper insights into underlying data patterns. Most existing equation discovery methods rely on genetic programming, which iteratively searches the equation space but is often slow and prone to overfitting. By representing equations as directed acyclic graphs, we leverage the use of graph neural networks to learn the underlying semantics of equations, and generate new, previously unseen equations. Although graph generative models have been shown to be successful in discovering new types of graphs in many fields, there application in discovering equations remains largely unexplored. In this work, we propose Graph-EQ, a deep graph generative model designed for efficient equation discovery. Graph-EQ uses a conditional variational autoencoder (CVAE) to learn a rich latent representation of the equation space by training it on a large corpus of equations in an unsupervised manner. Instead of directly searching the equation space, we employ Bayesian optimization to efficiently explore this learned latent space. We show that the encoder-decoder architecture of Graph-Eq is able to accurately reconstruct input equations. Moreover, we show that the learned latent representation can be sampled and decoded into valid equations, including new and previously unseen equations in the training data. Finally, we assess Graph-Eq's ability to discover equations that best fit a dataset by exploring the latent space using Bayesian optimization. Latent space exploration is done on 20 dataset with known ground-truth equations, and Graph-Eq is shown to successfully discover the grountruth equation in the majority of datasets. 

**Abstract (ZH)**: 能够在各种领域中发现有意义、准确且简洁的数学方程的能力是宝贵的。方程提供了变量之间的显式关系，有助于深入理解数据背后的模式。现有的大多数方程发现方法依赖于遗传编程，虽然可以迭代搜索方程空间，但往往速度较慢且容易过拟合。通过将方程表示为有向无环图，我们利用图神经网络来学习方程的潜在语义，并生成新的未见过的方程。尽管图生成模型在许多领域中已被证明能够成功发现新的图类型，但在发现方程方面的应用仍然鲜有探索。在这项工作中，我们提出Graph-EQ，一种用于高效方程发现的深度图生成模型。Graph-EQ使用条件变分自编码器（CVAE）通过无监督的方式训练大量方程的语料库，学习方程空间的丰富潜在表示。我们没有直接搜索方程空间，而是采用贝叶斯优化高效探索这种学习到的潜在空间。我们展示了Graph-Eq的编码器-解码器架构能够准确重建输入方程。此外，我们展示了学习到的潜在表示可以采样并解码为有效方程，包括训练数据中的新和未见过的方程。最后，我们通过使用贝叶斯优化探索潜在空间来评估Graph-Eq发现最佳拟合数据集方程的能力。在20个已知ground-truth方程的数据集上进行潜在空间探索，结果表明Graph-Eq能够在大多数数据集中成功发现ground-truth方程。 

---
# Interpretable Machine Learning in Physics: A Review 

**Title (ZH)**: 可解释的机器学习在物理学中的应用：一个综述 

**Authors**: Sebastian Johann Wetzel, Seungwoong Ha, Raban Iten, Miriam Klopotek, Ziming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23616)  

**Abstract**: Machine learning is increasingly transforming various scientific fields, enabled by advancements in computational power and access to large data sets from experiments and simulations. As artificial intelligence (AI) continues to grow in capability, these algorithms will enable many scientific discoveries beyond human capabilities. Since the primary goal of science is to understand the world around us, fully leveraging machine learning in scientific discovery requires models that are interpretable -- allowing experts to comprehend the concepts underlying machine-learned predictions. Successful interpretations increase trust in black-box methods, help reduce errors, allow for the improvement of the underlying models, enhance human-AI collaboration, and ultimately enable fully automated scientific discoveries that remain understandable to human scientists. This review examines the role of interpretability in machine learning applied to physics. We categorize different aspects of interpretability, discuss machine learning models in terms of both interpretability and performance, and explore the philosophical implications of interpretability in scientific inquiry. Additionally, we highlight recent advances in interpretable machine learning across many subfields of physics. By bridging boundaries between disciplines -- each with its own unique insights and challenges -- we aim to establish interpretable machine learning as a core research focus in science. 

**Abstract (ZH)**: 机器学习日益 transformations 各个科学领域，得益于计算能力的提升和从实验与模拟中获取的大规模数据集。随着人工智能（AI）能力的不断增长，这些算法将使许多超出人类能力范围的科学发现成为可能。鉴于科学的基本目标是理解我们周围的世界，充分利用机器学习在科学研究中的作用需要可解释的模型——使专家能够理解机器学习预测背后的概念。成功的解释增加了对黑盒方法的信任，有助于减少错误，允许提高底层模型，增强人类与AI的协作，并最终实现可为人科学家理解的完全自动化的科学发现。本文回顾了机器学习在物理学中的应用中解释性的作用。我们分类了解释性的不同方面，讨论了既具有解释性又具有高性能的机器学习模型，并探索了解释性在科学研究中的哲学含义。此外，我们还强调了物理学各个子领域的最新解释性机器学习进展。通过弥合学科之间的界限——每个学科都有其独特的见解和挑战——我们旨在将解释性机器学习确立为科学的核心研究重点。 

---
# Partial Transportability for Domain Generalization 

**Title (ZH)**: 域泛化的部分可迁移性 

**Authors**: Kasra Jalaldoust, Alexis Bellot, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2503.23605)  

**Abstract**: A fundamental task in AI is providing performance guarantees for predictions made in unseen domains. In practice, there can be substantial uncertainty about the distribution of new data, and corresponding variability in the performance of existing predictors. Building on the theory of partial identification and transportability, this paper introduces new results for bounding the value of a functional of the target distribution, such as the generalization error of a classifier, given data from source domains and assumptions about the data generating mechanisms, encoded in causal diagrams. Our contribution is to provide the first general estimation technique for transportability problems, adapting existing parameterization schemes such Neural Causal Models to encode the structural constraints necessary for cross-population inference. We demonstrate the expressiveness and consistency of this procedure and further propose a gradient-based optimization scheme for making scalable inferences in practice. Our results are corroborated with experiments. 

**Abstract (ZH)**: AI中的一个基本任务是为未见领域中的预测提供性能保证。基于部分识别和可传输性的理论，本文引入了在给定源领域数据和数据生成机制假设（编码在因果图中）的情况下，用于界定目标分布函数值（例如分类器的泛化误差）的新结果。我们的贡献是提供了首个通用的传输问题估算技术，将现有的参数化方案，如神经因果模型，适应性地编码用于跨人群推理的结构约束。我们展示了该程序的表述能力和一致性，并进一步提出了一种基于梯度的优化方案，以在实践中进行可扩展的推断。我们的结果通过实验得到了验证。 

---
# DASH: Detection and Assessment of Systematic Hallucinations of VLMs 

**Title (ZH)**: DASH: 系统幻觉检测与评估 

**Authors**: Maximilian Augustin, Yannic Neuhaus, Matthias Hein  

**Link**: [PDF](https://arxiv.org/pdf/2503.23573)  

**Abstract**: Vision-language models (VLMs) are prone to object hallucinations, where they erroneously indicate the presenceof certain objects in an image. Existing benchmarks quantify hallucinations using relatively small, labeled datasets. However, this approach is i) insufficient to assess hallucinations that arise in open-world settings, where VLMs are widely used, and ii) inadequate for detecting systematic errors in VLMs. We propose DASH (Detection and Assessment of Systematic Hallucinations), an automatic, large-scale pipeline designed to identify systematic hallucinations of VLMs on real-world images in an open-world setting. A key component is DASH-OPT for image-based retrieval, where we optimize over the ''natural image manifold'' to generate images that mislead the VLM. The output of DASH consists of clusters of real and semantically similar images for which the VLM hallucinates an object. We apply DASH to PaliGemma and two LLaVA-NeXT models across 380 object classes and, in total, find more than 19k clusters with 950k images. We study the transfer of the identified systematic hallucinations to other VLMs and show that fine-tuning PaliGemma with the model-specific images obtained with DASH mitigates object hallucinations. Code and data are available at this https URL. 

**Abstract (ZH)**: Vision-language模型中的系统性幻觉检测与评估（DASH） 

---
# Addressing Model Overcomplexity in Drug-Drug Interaction Prediction With Molecular Fingerprints 

**Title (ZH)**: 基于分子指纹图谱解决药物-药物相互作用预测中的模型过拟合问题 

**Authors**: Manel Gil-Sorribes, Alexis Molina  

**Link**: [PDF](https://arxiv.org/pdf/2503.23550)  

**Abstract**: Accurately predicting drug-drug interactions (DDIs) is crucial for pharmaceutical research and clinical safety. Recent deep learning models often suffer from high computational costs and limited generalization across datasets. In this study, we investigate a simpler yet effective approach using molecular representations such as Morgan fingerprints (MFPS), graph-based embeddings from graph convolutional networks (GCNs), and transformer-derived embeddings from MoLFormer integrated into a straightforward neural network. We benchmark our implementation on DrugBank DDI splits and a drug-drug affinity (DDA) dataset from the Food and Drug Administration. MFPS along with MoLFormer and GCN representations achieve competitive performance across tasks, even in the more challenging leak-proof split, highlighting the sufficiency of simple molecular representations. Moreover, we are able to identify key molecular motifs and structural patterns relevant to drug interactions via gradient-based analyses using the representations under study. Despite these results, dataset limitations such as insufficient chemical diversity, limited dataset size, and inconsistent labeling impact robust evaluation and challenge the need for more complex approaches. Our work provides a meaningful baseline and emphasizes the need for better dataset curation and progressive complexity scaling. 

**Abstract (ZH)**: 准确预测药物-药物相互作用（-DDIs）对于制药研究和临床安全性至关重要。尽管最近的深度学习模型常常面临高计算成本和跨数据集限制泛化的挑战，我们在本研究中探讨了一种更为简单有效的方法，使用诸如摩根指纹（MFPS）、图卷积网络（GCNs）的图基嵌入以及MoLFormer衍生的变压器嵌入，并将其集成到一个简单的神经网络中。我们在DrugBank DDI分割和食品和药物管理局的药物-药物亲和力（DDA）数据集上对标了我们的实现。MFPS与MoLFormer和GCN表示在各项任务中均表现出竞争力，即使在更具挑战性的密封泄漏分割中也是如此，这突显了简单分子表示的充分性。此外，我们还能够通过基于梯度的分析识别出与药物相互作用相关的关键分子模式和结构特征。尽管取得了这些结果，但由于数据集限制，如化学多样性不足、数据集规模有限和标签不一致，这影响了稳健的评估，并挑战了更复杂方法的必要性。我们的工作提供了有意义的基线，并强调了更好地数据集整理和逐步复杂性的必要性。 

---
# A Survey on Unlearnable Data 

**Title (ZH)**: 不可学习数据研究综述 

**Authors**: Jiahao Li, Yiqiang Chen, Yunbing Xing, Yang Gu, Xiangyuan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.23536)  

**Abstract**: Unlearnable data (ULD) has emerged as an innovative defense technique to prevent machine learning models from learning meaningful patterns from specific data, thus protecting data privacy and security. By introducing perturbations to the training data, ULD degrades model performance, making it difficult for unauthorized models to extract useful representations. Despite the growing significance of ULD, existing surveys predominantly focus on related fields, such as adversarial attacks and machine unlearning, with little attention given to ULD as an independent area of study. This survey fills that gap by offering a comprehensive review of ULD, examining unlearnable data generation methods, public benchmarks, evaluation metrics, theoretical foundations and practical applications. We compare and contrast different ULD approaches, analyzing their strengths, limitations, and trade-offs related to unlearnability, imperceptibility, efficiency and robustness. Moreover, we discuss key challenges, such as balancing perturbation imperceptibility with model degradation and the computational complexity of ULD generation. Finally, we highlight promising future research directions to advance the effectiveness and applicability of ULD, underscoring its potential to become a crucial tool in the evolving landscape of data protection in machine learning. 

**Abstract (ZH)**: 无法学习的数据（ULD）作为一种创新的防御技术，通过阻止机器学习模型从特定数据中学习有意义的模式，从而保护数据隐私和安全。通过向训练数据引入扰动，ULD降低模型性能，使未授权模型难以提取有用表示。尽管ULD的重要性日益增长，现有的综述主要集中在相关领域，如对抗攻击和机器遗忘，对ULD作为一个独立的研究领域关注较少。本文综述填补了这一空白，提供了ULD的全面综述，探讨了无法学习数据生成方法、公开基准、评估指标、理论基础和实际应用。我们对比了不同的ULD方法，分析了它们在不可学习性、不可感知性、效率和鲁棒性方面的优势、局限性和权衡。此外，我们讨论了关键挑战，如平衡扰动的不可感知性与模型性能下降，以及ULD生成的计算复杂性。最后，我们指出了未来有前景的研究方向，以提高ULD的有效性和适用性，突显其在机器学习数据保护演进 landscape 中的潜在重要性。 

---
# BiPVL-Seg: Bidirectional Progressive Vision-Language Fusion with Global-Local Alignment for Medical Image Segmentation 

**Title (ZH)**: BiPVL-Seg：双向渐进视觉-语言融合与全局-局部对齐在医学图像分割中的应用 

**Authors**: Rafi Ibn Sultan, Hui Zhu, Chengyin Li, Dongxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23534)  

**Abstract**: Medical image segmentation typically relies solely on visual data, overlooking the rich textual information clinicians use for diagnosis. Vision-language models attempt to bridge this gap, but existing approaches often process visual and textual features independently, resulting in weak cross-modal alignment. Simple fusion techniques fail due to the inherent differences between spatial visual features and sequential text embeddings. Additionally, medical terminology deviates from general language, limiting the effectiveness of off-the-shelf text encoders and further hindering vision-language alignment. We propose BiPVL-Seg, an end-to-end framework that integrates vision-language fusion and embedding alignment through architectural and training innovations, where both components reinforce each other to enhance medical image segmentation. BiPVL-Seg introduces bidirectional progressive fusion in the architecture, which facilitates stage-wise information exchange between vision and text encoders. Additionally, it incorporates global-local contrastive alignment, a training objective that enhances the text encoder's comprehension by aligning text and vision embeddings at both class and concept levels. Extensive experiments on diverse medical imaging benchmarks across CT and MR modalities demonstrate BiPVL-Seg's superior performance when compared with state-of-the-art methods in complex multi-class segmentation. Source code is available in this GitHub repository. 

**Abstract (ZH)**: 医学图像分割通常仅依赖视觉数据，忽略了临床医生用于诊断的丰富文本信息。视觉语言模型尝试弥合这一差距，但现有方法往往独立处理视觉和文本特征，导致模态间对齐较弱。简单的融合技术因空间视觉特征和序列文本嵌入之间的固有差异而失效。此外，医学术语与通用语言不同，限制了现成文本编码器的有效性，并进一步阻碍了视觉语言对齐。我们提出BiPVL-Seg，这是一种端到端框架，通过架构和训练创新将视觉语言融合和嵌入对齐结合在一起，两部分相互增强以提高医学图像分割性能。BiPVL-Seg引入了架构中的双向逐步融合，这促进了视觉和文本编码器逐阶段的信息交换。此外，它还结合了全局-局部对比对齐，这是一种训练目标，通过在类别和概念层面对齐文本和视觉嵌入来增强文本编码器的理解能力。在CT和MR模态的多种医学影像基准测试上的广泛实验表明，与最先进的方法相比，BiPVL-Seg在复杂多类分割中的性能更优。代码可在该GitHub仓库中获得。 

---
# If an LLM Were a Character, Would It Know Its Own Story? Evaluating Lifelong Learning in LLMs 

**Title (ZH)**: 如果语言模型是一个角色，它会知道自己自己的故事吗？评估语言模型的终身学习能力 

**Authors**: Siqi Fan, Xiusheng Huang, Yiqun Yao, Xuezhi Fang, Kang Liu, Peng Han, Shuo Shang, Aixin Sun, Yequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23514)  

**Abstract**: Large language models (LLMs) can carry out human-like dialogue, but unlike humans, they are stateless due to the superposition property. However, during multi-turn, multi-agent interactions, LLMs begin to exhibit consistent, character-like behaviors, hinting at a form of emergent lifelong learning. Despite this, existing benchmarks often fail to capture these dynamics, primarily focusing on static, open-ended evaluations. To address this gap, we introduce LIFESTATE-BENCH, a benchmark designed to assess lifelong learning in LLMs. It features two episodic datasets: Hamlet and a synthetic script collection, rich in narrative structure and character interactions. Our fact checking evaluation probes models' self-awareness, episodic memory retrieval, and relationship tracking, across both parametric and non-parametric approaches. Experiments on models like Llama3.1-8B, GPT-4-turbo, and DeepSeek R1, we demonstrate that nonparametric methods significantly outperform parametric ones in managing stateful learning. However, all models exhibit challenges with catastrophic forgetting as interactions extend, highlighting the need for further advancements in lifelong learning. 

**Abstract (ZH)**: Large Language Models (LLMs) 的长生命周期学习评估基准：LIFESTATE-BENCH 

---
# Buffer is All You Need: Defending Federated Learning against Backdoor Attacks under Non-iids via Buffering 

**Title (ZH)**: Buffer 是你需要的：通过缓冲防御非-iids 情况下的联邦学习后门攻击 

**Authors**: Xingyu Lyu, Ning Wang, Yang Xiao, Shixiong Li, Tao Li, Danjue Chen, Yimin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23511)  

**Abstract**: Federated Learning (FL) is a popular paradigm enabling clients to jointly train a global model without sharing raw data. However, FL is known to be vulnerable towards backdoor attacks due to its distributed nature. As participants, attackers can upload model updates that effectively compromise FL. What's worse, existing defenses are mostly designed under independent-and-identically-distributed (iid) settings, hence neglecting the fundamental non-iid characteristic of FL. Here we propose FLBuff for tackling backdoor attacks even under non-iids. The main challenge for such defenses is that non-iids bring benign and malicious updates closer, hence harder to separate. FLBuff is inspired by our insight that non-iids can be modeled as omni-directional expansion in representation space while backdoor attacks as uni-directional. This leads to the key design of FLBuff, i.e., a supervised-contrastive-learning model extracting penultimate-layer representations to create a large in-between buffer layer. Comprehensive evaluations demonstrate that FLBuff consistently outperforms state-of-the-art defenses. 

**Abstract (ZH)**: Federated Learning (FL)是一种流行的 paradigm，允许多个客户端联合训练全球模型而不共享原始数据。然而，FL由于其分布式特性，容易受到后门攻击。作为参与者，攻击者可以上传有效破坏FL的模型更新。更糟糕的是，现有防御大多是在独立且同分布(iid)设置下设计的，因此忽略了FL的基本非-iid特性。在这里，我们提出了FLBuff以应对非-iid条件下的后门攻击。此类防御的主要挑战在于非-iid使得良性更新和恶意更新更加接近，难以区分。FLBuff的灵感来源于我们对非-iid可以被视为表示空间的全方位扩展而后门攻击则是单向性的这一洞见。这促使FLBuff的关键设计是一个监督对比学习模型，从倒数第二层提取表示以创建一个大的中间缓冲层。全面的评估表明，FLBuff在各种情况下持续优于最先进的防御方法。 

---
# Boosting Omnidirectional Stereo Matching with a Pre-trained Depth Foundation Model 

**Title (ZH)**: 使用预训练深度基础模型增强 omnidirectional Stereo 匹配 

**Authors**: Jannik Endres, Oliver Hahn, Charles Corbière, Simone Schaub-Meyer, Stefan Roth, Alexandre Alahi  

**Link**: [PDF](https://arxiv.org/pdf/2503.23502)  

**Abstract**: Omnidirectional depth perception is essential for mobile robotics applications that require scene understanding across a full 360° field of view. Camera-based setups offer a cost-effective option by using stereo depth estimation to generate dense, high-resolution depth maps without relying on expensive active sensing. However, existing omnidirectional stereo matching approaches achieve only limited depth accuracy across diverse environments, depth ranges, and lighting conditions, due to the scarcity of real-world data. We present DFI-OmniStereo, a novel omnidirectional stereo matching method that leverages a large-scale pre-trained foundation model for relative monocular depth estimation within an iterative optimization-based stereo matching architecture. We introduce a dedicated two-stage training strategy to utilize the relative monocular depth features for our omnidirectional stereo matching before scale-invariant fine-tuning. DFI-OmniStereo achieves state-of-the-art results on the real-world Helvipad dataset, reducing disparity MAE by approximately 16% compared to the previous best omnidirectional stereo method. 

**Abstract (ZH)**: 全景深度感知对于需要全方位360°视野场景理解的移动机器人应用至关重要。基于摄像头的设置通过使用立体深度估计生成稠密高分辨率深度图，提供了一种经济有效的方案，无需依赖昂贵的主动感知设备。然而，现有的全景立体配对方法在不同环境、深度范围和光照条件下仅能实现有限的深度精度，这是由于缺乏真实世界数据的支持。我们提出了一种新颖的全景立体配对方法DFI-OmniStereo，该方法结合了大规模预训练基础模型在迭代优化立体配对架构内的相对单目深度估计。我们引入了一种专门的两阶段训练策略，利用相对单目深度特征进行全景立体配对，并进行尺度不变的微调。DFI-OmniStereo在实际世界Helvipad数据集上取得了最先进的结果，相比之前的最佳全景立体配对方法，减少了约16%的视差MAE。 

---
# POINT$^{2}$: A Polymer Informatics Training and Testing Database 

**Title (ZH)**: POINT$^{2}$: 聚合物信息学训练与测试数据库 

**Authors**: Jiaxin Xu, Gang Liu, Ruilan Guo, Meng Jiang, Tengfei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.23491)  

**Abstract**: The advancement of polymer informatics has been significantly propelled by the integration of machine learning (ML) techniques, enabling the rapid prediction of polymer properties and expediting the discovery of high-performance polymeric materials. However, the field lacks a standardized workflow that encompasses prediction accuracy, uncertainty quantification, ML interpretability, and polymer synthesizability. In this study, we introduce POINT$^{2}$ (POlymer INformatics Training and Testing), a comprehensive benchmark database and protocol designed to address these critical challenges. Leveraging the existing labeled datasets and the unlabeled PI1M dataset, a collection of approximately one million virtual polymers generated via a recurrent neural network trained on the realistic polymers, we develop an ensemble of ML models, including Quantile Random Forests, Multilayer Perceptrons with dropout, Graph Neural Networks, and pretrained large language models. These models are coupled with diverse polymer representations such as Morgan, MACCS, RDKit, Topological, Atom Pair fingerprints, and graph-based descriptors to achieve property predictions, uncertainty estimations, model interpretability, and template-based polymerization synthesizability across a spectrum of properties, including gas permeability, thermal conductivity, glass transition temperature, melting temperature, fractional free volume, and density. The POINT$^{2}$ database can serve as a valuable resource for the polymer informatics community for polymer discovery and optimization. 

**Abstract (ZH)**: 聚合物信息化的进步得益于机器学习技术的整合，这使得能够快速预测聚合物性能并加速高性能聚合物材料的发现。然而，该领域缺乏一个涵盖预测准确度、不确定性量化、机器学习可解释性和聚合物合成性的标准化工作流程。在本研究中，我们引入了POINT$^{2}$（聚合物信息化训练与测试），一个综合基准数据库和协议，旨在解决这些关键挑战。利用现有的标记数据集和未标记的PI1M数据集（通过在现实聚合物上训练的递归神经网络生成的约一百万种虚拟聚合物集合），我们开发了一组机器学习模型，包括分位数随机森林、具有丢弃的多层感知机、图神经网络和预训练的大语言模型。这些模型与多种聚合物表示相结合，如Morgan、MACCS、RDKit、拓扑、原子对指纹和基于图的描述符，实现了从气体渗透性、热导率、玻璃转变温度、熔点、自由体积分数和密度等一系列性质的性能预测、不确定性估计、模型可解释性和模板导向的聚合物聚合可合成性。POINT$^{2}$数据库可作为聚合物信息化社区进行聚合物发现和优化的宝贵资源。 

---
# Order Independence With Finetuning 

**Title (ZH)**: 微调的顺序无关性 

**Authors**: Katrina Brown, Reid McIlroy  

**Link**: [PDF](https://arxiv.org/pdf/2503.23483)  

**Abstract**: Large language models (LLMs) demonstrate remarkable performance on many NLP tasks, yet often exhibit order dependence: simply reordering semantically identical tokens (e.g., answer choices in multiple-choice questions) can lead to inconsistent predictions. Recent work proposes Set-Based Prompting (SBP) as a way to remove order information from designated token subsets, thereby mitigating positional biases. However, applying SBP on base models induces an out-of-distribution input format, which can degrade in-distribution performance. We introduce a fine-tuning strategy that integrates SBP into the training process, "pulling" these set-formatted prompts closer to the model's training manifold. We show that SBP can be incorporated into a model via fine-tuning. Our experiments on in-distribution (MMLU) and out-of-distribution (CSQA, ARC Challenge) multiple-choice tasks show that SBP fine-tuning significantly improves accuracy and robustness to answer-order permutations, all while preserving broader language modeling capabilities. We discuss the broader implications of order-invariant modeling and outline future directions for building fairer, more consistent LLMs. 

**Abstract (ZH)**: 基于集的提示微调：提高大型语言模型的顺序不变性能 

---
# Handling Delay in Real-Time Reinforcement Learning 

**Title (ZH)**: 处理实时强化学习中的延迟 

**Authors**: Ivan Anokhin, Rishav Rishav, Matthew Riemer, Stephen Chung, Irina Rish, Samira Ebrahimi Kahou  

**Link**: [PDF](https://arxiv.org/pdf/2503.23478)  

**Abstract**: Real-time reinforcement learning (RL) introduces several challenges. First, policies are constrained to a fixed number of actions per second due to hardware limitations. Second, the environment may change while the network is still computing an action, leading to observational delay. The first issue can partly be addressed with pipelining, leading to higher throughput and potentially better policies. However, the second issue remains: if each neuron operates in parallel with an execution time of $\tau$, an $N$-layer feed-forward network experiences observation delay of $\tau N$. Reducing the number of layers can decrease this delay, but at the cost of the network's expressivity. In this work, we explore the trade-off between minimizing delay and network's expressivity. We present a theoretically motivated solution that leverages temporal skip connections combined with history-augmented observations. We evaluate several architectures and show that those incorporating temporal skip connections achieve strong performance across various neuron execution times, reinforcement learning algorithms, and environments, including four Mujoco tasks and all MinAtar games. Moreover, we demonstrate parallel neuron computation can accelerate inference by 6-350% on standard hardware. Our investigation into temporal skip connections and parallel computations paves the way for more efficient RL agents in real-time setting. 

**Abstract (ZH)**: 实时强化学习中的延迟与网络表征能力 Trade-off探究：基于时间跳跃连接的历史增强观测方法及其应用 

---
# Codehacks: A Dataset of Adversarial Tests for Competitive Programming Problems Obtained from Codeforces 

**Title (ZH)**: Codehacks：来自Codeforces的对抗性测试数据集，用于 Competitive Programming 问题 

**Authors**: Max Hort, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23466)  

**Abstract**: Software is used in critical applications in our day-to-day life and it is important to ensure its correctness. One popular approach to assess correctness is to evaluate software on tests. If a test fails, it indicates a fault in the software under test; if all tests pass correctly, one may assume that the software is correct. However, the reliability of these results depends on the test suite considered, and there is a risk of false negatives (i.e. software that passes all available tests but contains bugs because some cases are not tested). Therefore, it is important to consider error-inducing test cases when evaluating software.
To support data-driven creation of such a test-suite, which is especially of interest for testing software synthesized from large language models, we curate a dataset (Codehacks) of programming problems together with corresponding error-inducing test cases (i.e., "hacks"). This dataset is collected from the wild, in particular, from the Codeforces online judge platform. The dataset comprises 288,617 hacks for 5,578 programming problems, each with a natural language description, as well as the source code for 2,196 submitted solutions to these problems that can be broken with their corresponding hacks.
Keywords: competitive programming, language model, dataset 

**Abstract (ZH)**: 软件在我们日常生活中被用于关键应用，确保其正确性很重要。常用的方法是通过测试评估软件的正确性。如果测试失败，说明被测试软件存在故障；如果所有测试都通过，则可以假设软件是正确的。然而，这些结果的可靠性取决于所考虑的测试集，存在因未测试某些情况而导致误判（即软件通过所有可用测试但包含未测试情况导致的错误）的风险。因此，在评估软件时考虑引入错误的测试案例很重要。
为了支持这种测试套件的数据驱动创建，特别是对于从大型语言模型合成的软件测试特别感兴趣，我们收集了一个包含编程问题及其相应的引入错误的测试案例（即“破解”）的数据集（Codehacks）。该数据集来源于Codeforces在线裁判平台等野生环境。数据集包含288,617个破解案例，针对5,578个编程问题，每个问题都有自然语言描述和2,196个提交的解决方案的源代码，这些解决方案可以通过相应的破解案例来破坏。关键词：竞技编程，语言模型，数据集。 

---
# Semantic-Preserving Transformations as Mutation Operators: A Study on Their Effectiveness in Defect Detection 

**Title (ZH)**: 语义保留变换作为变异操作符：其在缺陷检测中的有效性研究 

**Authors**: Max Hort, Linas Vidziunas, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23448)  

**Abstract**: Recent advances in defect detection use language models. Existing works enhanced the training data to improve the models' robustness when applied to semantically identical code (i.e., predictions should be the same). However, the use of semantically identical code has not been considered for improving the tools during their application - a concept closely related to metamorphic testing.
The goal of our study is to determine whether we can use semantic-preserving transformations, analogue to mutation operators, to improve the performance of defect detection tools in the testing stage. We first collect existing publications which implemented semantic-preserving transformations and share their implementation, such that we can reuse them. We empirically study the effectiveness of three different ensemble strategies for enhancing defect detection tools. We apply the collected transformations on the Devign dataset, considering vulnerabilities as a type of defect, and two fine-tuned large language models for defect detection (VulBERTa, PLBART). We found 28 publications with 94 different transformations.
We choose to implement 39 transformations from four of the publications, but a manual check revealed that 23 out 39 transformations change code semantics. Using the 16 remaining, correct transformations and three ensemble strategies, we were not able to increase the accuracy of the defect detection models. Our results show that reusing shared semantic-preserving transformation is difficult, sometimes even causing wrongful changes to the semantics.
Keywords: defect detection, language model, semantic-preserving transformation, ensemble 

**Abstract (ZH)**: Recent advances in 缺陷检测使用语言模型：基于语义保留变换的性能提升研究 

---
# Speculative End-Turn Detector for Efficient Speech Chatbot Assistant 

**Title (ZH)**: speculation-end-turn 侦测器：高效的语音聊天机器人助手 

**Authors**: Hyunjong Ok, Suho Yoo, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.23439)  

**Abstract**: Spoken dialogue systems powered by large language models have demonstrated remarkable abilities in understanding human speech and generating appropriate spoken responses. However, these systems struggle with end-turn detection (ETD) -- the ability to distinguish between user turn completion and hesitation. This limitation often leads to premature or delayed responses, disrupting the flow of spoken conversations. In this paper, we introduce the ETD Dataset, the first public dataset for end-turn detection. The ETD dataset consists of both synthetic speech data generated with text-to-speech models and real-world speech data collected from web sources. We also propose SpeculativeETD, a novel collaborative inference framework that balances efficiency and accuracy to improve real-time ETD in resource-constrained environments. Our approach jointly employs a lightweight GRU-based model, which rapidly detects the non-speaking units in real-time on local devices, and a high-performance Wav2vec-based model running on the server to make a more challenging classification of distinguishing turn ends from mere pauses. Experiments demonstrate that the proposed SpeculativeETD significantly improves ETD accuracy while keeping the required computations low. Datasets and code will be available after the review. 

**Abstract (ZH)**: 由大规模语言模型驱动的对话系统在理解人类语音和生成适当语音响应方面展现了非凡的能力。然而，这些系统在结束轮检测（ETD）——区分用户发言结束和犹豫的能力——方面存在局限性。这一局限常常导致提前或延迟的应答，破坏了对话的流畅性。本文介绍了ETD数据集，这是首个公开的结束轮检测数据集。ETD数据集包含由文本到语音模型生成的合成语音数据和从网络来源收集的真实语音数据。同时，我们提出了SpeculativeETD，这是一种新颖的协作推理框架，通过平衡效率与准确性来提高受限资源环境下实时结束轮检测的性能。我们的方法结合使用了一个轻量级的基于GRU的模型，在本地设备上实时快速检测非说话单位，以及一个高性能的Wav2vec模型在服务器上运行，进行更具有挑战性的区分发言结束和简单停顿的分类。实验表明，提出的SpeculativeETD在保持所需计算量低的同时，显著提高了结束轮检测的准确性。数据集和代码将在审稿通过后提供。 

---
# What Makes an Evaluation Useful? Common Pitfalls and Best Practices 

**Title (ZH)**: 什么是有效的评价？常见的陷阱与最佳实践 

**Authors**: Gil Gekker, Meirav Segal, Dan Lahav, Omer Nevo  

**Link**: [PDF](https://arxiv.org/pdf/2503.23424)  

**Abstract**: Following the rapid increase in Artificial Intelligence (AI) capabilities in recent years, the AI community has voiced concerns regarding possible safety risks. To support decision-making on the safe use and development of AI systems, there is a growing need for high-quality evaluations of dangerous model capabilities. While several attempts to provide such evaluations have been made, a clear definition of what constitutes a "good evaluation" has yet to be agreed upon. In this practitioners' perspective paper, we present a set of best practices for safety evaluations, drawing on prior work in model evaluation and illustrated through cybersecurity examples. We first discuss the steps of the initial thought process, which connects threat modeling to evaluation design. Then, we provide the characteristics and parameters that make an evaluation useful. Finally, we address additional considerations as we move from building specific evaluations to building a full and comprehensive evaluation suite. 

**Abstract (ZH)**: 随着近年来人工智能（AI）能力的迅速提升，AI社区表达了对其潜在安全风险的担忧。为了支持AI系统的安全使用和开发的决策制定，高质量的危险模型能力评估需求日益增长。尽管已经做出了若干尝试来提供这样的评估，但对于什么是“好的评估”仍缺乏明确定义。在本文中，我们基于先前的工作，通过网络安全领域的实例，介绍了一套安全评估的最佳实践。首先，我们讨论了初始思维过程中的步骤，将威胁建模与评估设计联系起来。然后，我们提供了使评估有用的特点和参数。最后，我们在从构建特定评估到构建全面评估套件的过程中，讨论了其他需要考虑的因素。 

---
# An Analysis of Decoding Methods for LLM-based Agents for Faithful Multi-Hop Question Answering 

**Title (ZH)**: 基于LLM的代理多跳问答忠实解码方法分析 

**Authors**: Alexander Murphy, Mohd Sanad Zaki Rizvi, Aden Haussmann, Ping Nie, Guifu Liu, Aryo Pradipta Gema, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2503.23415)  

**Abstract**: Large Language Models (LLMs) frequently produce factually inaccurate outputs - a phenomenon known as hallucination - which limits their accuracy in knowledge-intensive NLP tasks. Retrieval-augmented generation and agentic frameworks such as Reasoning and Acting (ReAct) can address this issue by giving the model access to external knowledge. However, LLMs often fail to remain faithful to retrieved information. Mitigating this is critical, especially if LLMs are required to reason about the retrieved information. Recent research has explored training-free decoding strategies to improve the faithfulness of model generations. We present a systematic analysis of how the combination of the ReAct framework and decoding strategies (i.e., DeCoRe, DoLa, and CAD) can influence the faithfulness of LLM-generated answers. Our results show that combining an agentic framework for knowledge retrieval with decoding methods that enhance faithfulness can increase accuracy on the downstream Multi-Hop Question Answering tasks. For example, we observe an F1 increase from 19.5 to 32.6 on HotpotQA when using ReAct and DoLa. 

**Abstract (ZH)**: 大型语言模型（LLMs）经常生成事实不准确的输出——这一现象被称为幻觉，这限制了它们在知识密集型NLP任务中的准确性。检索增强生成和Reasoning and Acting（ReAct）等有能动性框架可以通过给予模型访问外部知识的能力来解决这一问题。然而，LLMs往往未能忠实地保留检索到的信息。减轻这一问题至关重要，尤其是在LLMs需要推理检索到的信息时。最近的研究探索了无需训练的解码策略以提高模型生成的忠实性。我们对结合ReAct框架与增强忠实性的解码方法（即DeCoRe、DoLa和CAD）如何影响LLM生成答案的忠实性进行了系统分析。结果显示，结合知识检索的有能动性框架与增强忠实性的解码方法可以在下游多跳问答任务中提高准确性。例如，我们观察到在使用ReAct和DoLa时，HotpotQA的F1分数从19.5提高到32.6。 

---
# From Content Creation to Citation Inflation: A GenAI Case Study 

**Title (ZH)**: 从内容创作到引用膨胀：一个GenAI案例研究 

**Authors**: Haitham S. Al-Sinani, Chris J. Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2503.23414)  

**Abstract**: This paper investigates the presence and impact of questionable, AI-generated academic papers on widely used preprint repositories, with a focus on their role in citation manipulation. Motivated by suspicious patterns observed in publications related to our ongoing research on GenAI-enhanced cybersecurity, we identify clusters of questionable papers and profiles. These papers frequently exhibit minimal technical content, repetitive structure, unverifiable authorship, and mutually reinforcing citation patterns among a recurring set of authors. To assess the feasibility and implications of such practices, we conduct a controlled experiment: generating a fake paper using GenAI, embedding citations to suspected questionable publications, and uploading it to one such repository (ResearchGate). Our findings demonstrate that such papers can bypass platform checks, remain publicly accessible, and contribute to inflating citation metrics like the H-index and i10-index. We present a detailed analysis of the mechanisms involved, highlight systemic weaknesses in content moderation, and offer recommendations for improving platform accountability and preserving academic integrity in the age of GenAI. 

**Abstract (ZH)**: 本文调查了可疑的、由AI生成的学术论文在广泛使用的预印本 repositories 中的存在及其影响，重点关注这些论文在引文操纵中的角色。受我们对增强型生成AI网络安全研究中发现的可疑模式的启发，我们识别了可疑论文和作者群体。这些论文经常表现出技术内容少、结构重复、作者身份难以验证以及作者之间的互相支持的引文模式。为了评估此类做法的可行性和影响，我们进行了一个受控实验：使用生成AI生成一篇虚假论文，嵌入疑似可疑论文的引文，并将其上传到一个这样的仓库（ResearchGate）。我们的研究发现这些论文可以绕过平台检查，保持公开访问，并有助于夸大如H指数和i10指数等引文指标。我们详细分析了涉及的机制，突出了内容审核中的系统性薄弱环节，并提出了改进平台责任和在生成AI时代维护学术诚信的建议。 

---
# GMapLatent: Geometric Mapping in Latent Space 

**Title (ZH)**: GMapLatent：潜在空间中的几何映射 

**Authors**: Wei Zeng, Xuebin Chang, Jianghao Su, Xiang Gu, Jian Sun, Zongben Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23407)  

**Abstract**: Cross-domain generative models based on encoder-decoder AI architectures have attracted much attention in generating realistic images, where domain alignment is crucial for generation accuracy. Domain alignment methods usually deal directly with the initial distribution; however, mismatched or mixed clusters can lead to mode collapse and mixture problems in the decoder, compromising model generalization capabilities. In this work, we innovate a cross-domain alignment and generation model that introduces a canonical latent space representation based on geometric mapping to align the cross-domain latent spaces in a rigorous and precise manner, thus avoiding mode collapse and mixture in the encoder-decoder generation architectures. We name this model GMapLatent. The core of the method is to seamlessly align latent spaces with strict cluster correspondence constraints using the canonical parameterizations of cluster-decorated latent spaces. We first (1) transform the latent space to a canonical parameter domain by composing barycenter translation, optimal transport merging and constrained harmonic mapping, and then (2) compute geometric registration with cluster constraints over the canonical parameter domains. This process realizes a bijective (one-to-one and onto) mapping between newly transformed latent spaces and generates a precise alignment of cluster pairs. Cross-domain generation is then achieved through the aligned latent spaces embedded in the encoder-decoder pipeline. Experiments on gray-scale and color images validate the efficiency, efficacy and applicability of GMapLatent, and demonstrate that the proposed model has superior performance over existing models. 

**Abstract (ZH)**: 基于编码器-解码器架构的跨域生成模型：通过几何映射实现精确的跨域潜空间对齐 

---
# Diffusion Meets Few-shot Class Incremental Learning 

**Title (ZH)**: 扩散模型 Meet 少量-shot 类增量学习 

**Authors**: Junsu Kim, Yunhoe Ku, Dongyoon Han, Seungryul Baek  

**Link**: [PDF](https://arxiv.org/pdf/2503.23402)  

**Abstract**: Few-shot class-incremental learning (FSCIL) is challenging due to extremely limited training data; while aiming to reduce catastrophic forgetting and learn new information. We propose Diffusion-FSCIL, a novel approach that employs a text-to-image diffusion model as a frozen backbone. Our conjecture is that FSCIL can be tackled using a large generative model's capabilities benefiting from 1) generation ability via large-scale pre-training; 2) multi-scale representation; 3) representational flexibility through the text encoder. To maximize the representation capability, we propose to extract multiple complementary diffusion features to play roles as latent replay with slight support from feature distillation for preventing generative biases. Our framework realizes efficiency through 1) using a frozen backbone; 2) minimal trainable components; 3) batch processing of multiple feature extractions. Extensive experiments on CUB-200, miniImageNet, and CIFAR-100 show that Diffusion-FSCIL surpasses state-of-the-art methods, preserving performance on previously learned classes and adapting effectively to new ones. 

**Abstract (ZH)**: 基于文本到图像扩散模型的少样本类增量学习（Diffusion-FSCIL） 

---
# Scaling Auditory Cognition via Test-Time Compute in Audio Language Models 

**Title (ZH)**: 通过测试时计算提升音频语言模型中的听觉认知 

**Authors**: Ting Dang, Yan Gao, Hong Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.23395)  

**Abstract**: Large language models (LLMs) have shown exceptional versatility in natural language processing, prompting recent efforts to extend their multimodal capabilities to speech processing through the development of audio large language models (Audio LLMs). While Audio LLMs excel in tasks such as speech recognition and synthesis, it remains unclear how they perform when faced with the auditory cognitive challenges posed by real-world environments, such as audio comprehension and listening recall, particularly in the presence of background noise or overlapping speech. Unlike text-based LLMs, which have access to vast amounts of text data for pre-training, retraining Audio LLMs with diverse auditory cognitive scenes is difficult due to the limited datasets that simulate real-world auditory cognitive scenarios and the challenge of acquiring auditory cognitive labels for training. While test-time compute (TTC) methods have been shown to enhance the capabilities of text-based LLMs during inference, a key challenge lies in designing these TTC methods to improve the auditory capabilities of Audio LLMs. This study aims to address these two research gaps by: i) exploring the auditory cognitive capabilities of Audio LLMs, and ii) enhancing their capabilities using TTC approaches. We have investigated five different Audio LLMs for auditory cognition using a \textit{self-collected} database and have proposed five TTC approaches to enhance auditory cognitive capabilities during inference. Our findings reveal that Audio LLMs performance decreases in more challenging auditory cognitive tasks. The proposed TTC approaches significantly enhance cognitive auditory capabilities, advancing the development of more adaptable and resilient Audio LLMs for practical applications such as assistive listening devices, voice-based AI assistants, and communication technologies. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理中展示了非凡的灵活性，推动了通过开发音频大型语言模型（Audio LLMs）来扩展其多模态能力以应用于语音处理的努力。尽管Audio LLMs在语音识别和合成任务中表现出色，但在面临真实世界环境中的听觉认知挑战，如音频理解和听觉记忆，尤其是存在背景噪声或重叠语音的情况下，它们的性能尚不清楚。与具备大量文本数据用于预训练和重新训练的文本基于LLMs不同，由于模拟真实世界听觉认知场景的数据集有限以及获取训练所需的听觉认知标签的挑战，重新训练Audio LLMs具有多样性听觉认知场景非常困难。虽然测试时计算（TTC）方法已经被证明可以在推理过程中增强文本基于LLMs的能力，但关键的挑战在于设计这些TTC方法以改善Audio LLMs的听觉能力。本研究旨在通过以下方式解决这两个研究空白：i) 探索Audio LLMs的听觉认知能力，ii) 使用TTC方法增强其能力。我们使用一个自收集的数据库研究了五种不同的Audio LLMs在听觉认知中的应用，并提出了五种TTC方法以在推理过程中增强听觉认知能力。我们的研究发现表明，Audio LLMs在更复杂的听觉认知任务中的表现下降。所提出的方法显著提高了听觉认知能力，推动了适用于辅助听力设备、基于语音的人工智能助手和通信技术等实际应用的更适应性和鲁棒性Audio LLMs的发展。 

---
# Spatiotemporal Learning of Brain Dynamics from fMRI Using Frequency-Specific Multi-Band Attention for Cognitive and Psychiatric Applications 

**Title (ZH)**: 基于频率特定多频带注意力的fMRI脑动态时空学习在认知和精神卫生应用中 

**Authors**: Sangyoon Bae, Junbeom Kwon, Shinjae Yoo, Jiook Cha  

**Link**: [PDF](https://arxiv.org/pdf/2503.23394)  

**Abstract**: Understanding how the brain's complex nonlinear dynamics give rise to adaptive cognition and behavior is a central challenge in neuroscience. These dynamics exhibit scale-free and multifractal properties, influencing the reconfiguration of neural networks. However, conventional neuroimaging models are constrained by linear and stationary assumptions, limiting their ability to capture these processes. Transformer-based architectures, known for capturing long-range dependencies, align well with the brain's hierarchical and temporal organization. We introduce Multi-Band Brain Net (MBBN), a transformer-based framework that models frequency-specific spatiotemporal brain dynamics from fMRI by integrating scale-free network principles with frequency-resolved multi-band self-attention. Trained on three large-scale neuroimaging cohorts (UK Biobank, ABCD, ABIDE) totaling 45,951 individuals, MBBN reveals previously undetectable frequency-dependent network interactions, shedding light on connectivity disruptions in psychiatric conditions (ADHD, ASD, depression). This validation shows robust generalizability and highlights core neural principles conserved across populations. MBBN achieves up to 30.59% higher predictive accuracy than state-of-the-art methods, demonstrating the advantage of frequency-informed spatiotemporal modeling in capturing latent neural computations. MBBN's interpretability uncovers novel frequency-specific biomarkers for neurodevelopmental disorders, providing insights into the hierarchical organization of brain function. By offering an interpretable framework for spatiotemporal learning, MBBN provides insights into how neural computations underpin cognitive function and psychiatric vulnerability, with implications for brain decoding, cognitive neuroscience, and precision psychiatry. 

**Abstract (ZH)**: 理解大脑复杂非线性动态如何产生适应性认知和行为是神经科学中的一个核心挑战。这些动态表现出无标度和多分形特性，影响神经网络的重构。然而，传统神经成像模型受限于线性和稳态假设，限制了它们捕捉这些过程的能力。基于变换器的架构因其能捕捉长程依赖关系而与大脑的分层和时间组织相契合。我们提出了多频带脑网络（MBBN）框架，该框架通过结合无标度网络原理和频带分辨率多频带自注意力机制，从功能性磁共振成像（fMRI）中建模频率特异性的时空脑动态。MBBN在三个大规模神经成像队列（UK Biobank, ABCD, ABIDE）的45,951个体上进行了训练，揭示了以前未检测到的频率依赖性网络交互，阐明了精神疾病（ADHD, ASD, 抑郁）中的连接性障碍。这一验证显示了其稳健的泛化能力和跨人群保守的核心神经原理。MBBN的预测准确率最高可比最先进的方法提高30.59%，证明了基于频率的时空建模在捕捉潜在神经计算方面的优势。MBBN的可解释性揭示了神经发育障碍的新频率特异性生物标志物，提供了有关大脑功能分层组织的见解。通过提供可解释的时空学习框架，MBBN为理解神经计算如何支撑认知功能和精神疾病易感性提供了洞察，对于脑解码、认知神经科学和精准精神病学具有重要影响。 

---
# Pareto Continual Learning: Preference-Conditioned Learning and Adaption for Dynamic Stability-Plasticity Trade-off 

**Title (ZH)**: 帕累托持续学习：基于偏好条件化学习与适应的动态稳定-可塑性权衡博弈 

**Authors**: Song Lai, Zhe Zhao, Fei Zhu, Xi Lin, Qingfu Zhang, Gaofeng Meng  

**Link**: [PDF](https://arxiv.org/pdf/2503.23390)  

**Abstract**: Continual learning aims to learn multiple tasks sequentially. A key challenge in continual learning is balancing between two objectives: retaining knowledge from old tasks (stability) and adapting to new tasks (plasticity). Experience replay methods, which store and replay past data alongside new data, have become a widely adopted approach to mitigate catastrophic forgetting. However, these methods neglect the dynamic nature of the stability-plasticity trade-off and aim to find a fixed and unchanging balance, resulting in suboptimal adaptation during training and inference. In this paper, we propose Pareto Continual Learning (ParetoCL), a novel framework that reformulates the stability-plasticity trade-off in continual learning as a multi-objective optimization (MOO) problem. ParetoCL introduces a preference-conditioned model to efficiently learn a set of Pareto optimal solutions representing different trade-offs and enables dynamic adaptation during inference. From a generalization perspective, ParetoCL can be seen as an objective augmentation approach that learns from different objective combinations of stability and plasticity. Extensive experiments across multiple datasets and settings demonstrate that ParetoCL outperforms state-of-the-art methods and adapts to diverse continual learning scenarios. 

**Abstract (ZH)**: 持续学习旨在顺序学习多个任务。持续学习中的一个关键挑战是在保持旧任务知识（稳定性）和适应新任务（可塑性）之间取得平衡。经验重播方法通过存储和重播过去的数据与新数据一起，已成为减轻灾难性遗忘的广泛采用方法。然而，这些方法忽视了稳定性-可塑性权衡的动态性质，并试图找到一个固定的、不变的平衡，导致在训练和推理过程中适应能力不足。在本文中，我们提出了一种新颖框架Pareto持续学习（ParetoCL），将持续学习中的稳定性-可塑性权衡重新表述为多目标优化（MOO）问题。ParetoCL引入了一种偏好条件下的模型，能够高效地学习代表不同权衡的一组Pareto最优解，并在推理过程中实现动态适应。从泛化角度来看，ParetoCL可以被视为一种目标增强方法，能够从稳定性与可塑性不同目标组合中学习。实验结果表明，ParetoCL在多个数据集和设置中优于现有方法，并能够适应多种持续学习场景。 

---
# COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation 

**Title (ZH)**: COSMIC: 基于 clique 的语义多空间集成以实现鲁棒的 CLIP 测试时适应 

**Authors**: Fanding Huang, Jingyan Jiang, Qinting Jiang, Hebei Li, Faisal Nadeem Khan, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23388)  

**Abstract**: Recent vision-language models (VLMs) face significant challenges in test-time adaptation to novel domains. While cache-based methods show promise by leveraging historical information, they struggle with both caching unreliable feature-label pairs and indiscriminately using single-class information during querying, significantly compromising adaptation accuracy. To address these limitations, we propose COSMIC (Clique-Oriented Semantic Multi-space Integration for CLIP), a robust test-time adaptation framework that enhances adaptability through multi-granular, cross-modal semantic caching and graph-based querying mechanisms. Our framework introduces two key innovations: Dual Semantics Graph (DSG) and Clique Guided Hyper-class (CGH). The Dual Semantics Graph constructs complementary semantic spaces by incorporating textual features, coarse-grained CLIP features, and fine-grained DINOv2 features to capture rich semantic relationships. Building upon these dual graphs, the Clique Guided Hyper-class component leverages structured class relationships to enhance prediction robustness through correlated class selection. Extensive experiments demonstrate COSMIC's superior performance across multiple benchmarks, achieving significant improvements over state-of-the-art methods: 15.81% gain on out-of-distribution tasks and 5.33% on cross-domain generation with CLIP RN-50. Code is available at this http URL. 

**Abstract (ZH)**: Recent vision-language模型（VLMs）在测试时适应新型领域方面面临着显著挑战。尽管基于缓存的方法通过利用历史信息展现了潜力，但在缓存不可靠的特征-标签对以及在查询时不分场合地使用单类信息方面存在局限，严重影响了适应准确性。为解决这些局限性，我们提出了COSMIC（基于聚类的语义多空间集成用于CLIP），这是一种通过多粒度、跨模态语义缓存和图查询机制来增强适应性的稳健测试时适应框架。我们的框架引入了两个关键创新：双语义图（DSG）和聚类引导的超类（CGH）。双语义图通过整合文本特征、粗粒度CLIP特征和细粒度DINOv2特征来构建互补的语义空间，以捕获丰富的语义关系。在此基础上，聚类引导的超类组件利用结构化类关系，通过相关类的选择来增强预测的稳健性。广泛的经验表明，COSMIC在多个基准测试中表现出优越性能，相对于最先进的方法，在分布外任务上取得了15.81%的增益，在使用CLIP RN-50进行跨域生成任务上取得了5.33%的增益。代码详见此网址。 

---
# KernelDNA: Dynamic Kernel Sharing via Decoupled Naive Adapters 

**Title (ZH)**: KernelDNA: 动态核共享通过解耦天真适配器 

**Authors**: Haiduo Huang, Yadong Zhang, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.23379)  

**Abstract**: Dynamic convolution enhances model capacity by adaptively combining multiple kernels, yet faces critical trade-offs: prior works either (1) incur significant parameter overhead by scaling kernel numbers linearly, (2) compromise inference speed through complex kernel interactions, or (3) struggle to jointly optimize dynamic attention and static kernels. We also observe that pre-trained Convolutional Neural Networks (CNNs) exhibit inter-layer redundancy akin to that in Large Language Models (LLMs). Specifically, dense convolutional layers can be efficiently replaced by derived ``child" layers generated from a shared ``parent" convolutional kernel through an adapter.
To address these limitations and implement the weight-sharing mechanism, we propose a lightweight convolution kernel plug-in, named KernelDNA. It decouples kernel adaptation into input-dependent dynamic routing and pre-trained static modulation, ensuring both parameter efficiency and hardware-friendly inference. Unlike existing dynamic convolutions that expand parameters via multi-kernel ensembles, our method leverages cross-layer weight sharing and adapter-based modulation, enabling dynamic kernel specialization without altering the standard convolution structure. This design preserves the native computational efficiency of standard convolutions while enhancing representation power through input-adaptive kernel adjustments. Experiments on image classification and dense prediction tasks demonstrate that KernelDNA achieves state-of-the-art accuracy-efficiency balance among dynamic convolution variants. Our codes are available at this https URL. 

**Abstract (ZH)**: 动态卷积通过适应性结合多个核增强模型容量，但面临关键权衡：现有工作要么（1）通过线性扩展核数量引起显著的参数开销，要么（2）通过复杂的核交互牺牲推理速度，要么（3）难以同时优化动态注意力和静态核。我们还观察到预训练的卷积神经网络（CNNs）在层间冗余方面类似于大型语言模型（LLMs）。具体而言，密集的卷积层可以通过共享“父”卷积核生成的“子”层进行高效替换。

为了解决这些限制并实现权重共享机制，我们提出了一种轻量级卷积核插件，名为KernelDNA。它将核适应解耦为输入相关的动态路由和预训练的静态调制，确保参数效率和硬件友好的推理。与现有通过多核组合扩展参数的动态卷积不同，我们的方法利用跨层权重共享和基于适配器的调制，能够在不改变标准卷积结构的情况下实现动态核专业化。这种设计保持了标准卷积的原生计算效率，同时通过输入自适应的核调整增强表示能力。在图像分类和密集预测任务上的实验表明，KernelDNA在动态卷积变种中实现了最佳的准确性和效率平衡。我们的代码可在以下链接获取。 

---
# JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization 

**Title (ZH)**: JavisDiT：联合音视频扩散变换器及其层次时空先验同步 

**Authors**: Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Rongxin Jiang, Jiebo Luo, Hao Fei, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.23377)  

**Abstract**: This paper introduces JavisDiT, a novel Joint Audio-Video Diffusion Transformer designed for synchronized audio-video generation (JAVG). Built upon the powerful Diffusion Transformer (DiT) architecture, JavisDiT is able to generate high-quality audio and video content simultaneously from open-ended user prompts. To ensure optimal synchronization, we introduce a fine-grained spatio-temporal alignment mechanism through a Hierarchical Spatial-Temporal Synchronized Prior (HiST-Sypo) Estimator. This module extracts both global and fine-grained spatio-temporal priors, guiding the synchronization between the visual and auditory components. Furthermore, we propose a new benchmark, JavisBench, consisting of 10,140 high-quality text-captioned sounding videos spanning diverse scenes and complex real-world scenarios. Further, we specifically devise a robust metric for evaluating the synchronization between generated audio-video pairs in real-world complex content. Experimental results demonstrate that JavisDiT significantly outperforms existing methods by ensuring both high-quality generation and precise synchronization, setting a new standard for JAVG tasks. Our code, model, and dataset will be made publicly available at this https URL. 

**Abstract (ZH)**: JavisDiT：一种用于同步音视频生成的新型联合音视频扩散变换器 

---
# FeRG-LLM : Feature Engineering by Reason Generation Large Language Models 

**Title (ZH)**: FeRG-LLM：基于推理生成的特征工程大型语言模型 

**Authors**: Jeonghyun Ko, Gyeongyun Park, Donghoon Lee, Kyunam Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.23371)  

**Abstract**: One of the key tasks in machine learning for tabular data is feature engineering. Although it is vital for improving the performance of models, it demands considerable human expertise and deep domain knowledge, making it labor-intensive endeavor. To address this issue, we propose a novel framework, \textbf{FeRG-LLM} (\textbf{Fe}ature engineering by \textbf{R}eason \textbf{G}eneration \textbf{L}arge \textbf{L}anguage \textbf{M}odels), a large language model designed to automatically perform feature engineering at an 8-billion-parameter scale. We have constructed two-stage conversational dialogues that enable language models to analyze machine learning tasks and discovering new features, exhibiting their Chain-of-Thought (CoT) capabilities. We use these dialogues to fine-tune Llama 3.1 8B model and integrate Direct Preference Optimization (DPO) to receive feedback improving quality of new features and the model's performance. Our experiments show that FeRG-LLM performs comparably to or better than Llama 3.1 70B on most datasets, while using fewer resources and achieving reduced inference time. It outperforms other studies in classification tasks and performs well in regression tasks. Moreover, since it does not rely on cloud-hosted LLMs like GPT-4 with extra API costs when generating features, it can be deployed locally, addressing security concerns. 

**Abstract (ZH)**: 一种用于表格数据的机器学习的关键任务是特征工程。尽管特征工程对于提高模型性能至关重要，但它需要大量的专业知识和深入的领域知识，使得这一过程耗时且劳动密集。为了解决这一问题，我们提出了一种新的框架——FeRG-LLM（Feature engineering by Reason Generation Large Language Models），这是一种大型语言模型，旨在以80亿参数规模自动进行特征工程。我们构建了两阶段的对话流程，使语言模型能够分析机器学习任务并发现新特征，展示了其链式推理（Chain-of-Thought, CoT）能力。我们使用这些对话流程对Llama 3.1 8B模型进行微调，并集成直接偏好优化（DPO）以获得反馈，提高新特征的质量和模型的性能。实验结果显示，FeRG-LLM在大多数数据集上的性能与Llama 3.1 70B相当或更好，同时使用较少资源并缩短了推理时间。在分类任务中，FeRG-LLM优于其他研究，在回归任务中表现良好。此外，由于它不需要依赖于像GPT-4这样的云托管大型语言模型来生成特征（后者会产生额外的API成本），因此它可以本地部署，解决了安全性问题。 

---
# Towards Physically Plausible Video Generation via VLM Planning 

**Title (ZH)**: 基于VLM规划的物理合理视频生成 

**Authors**: Xindi Yang, Baolu Li, Yiming Zhang, Zhenfei Yin, Lei Bai, Liqian Ma, Zhiyong Wang, Jianfei Cai, Tien-Tsin Wong, Huchuan Lu, Xu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.23368)  

**Abstract**: Video diffusion models (VDMs) have advanced significantly in recent years, enabling the generation of highly realistic videos and drawing the attention of the community in their potential as world simulators. However, despite their capabilities, VDMs often fail to produce physically plausible videos due to an inherent lack of understanding of physics, resulting in incorrect dynamics and event sequences. To address this limitation, we propose a novel two-stage image-to-video generation framework that explicitly incorporates physics. In the first stage, we employ a Vision Language Model (VLM) as a coarse-grained motion planner, integrating chain-of-thought and physics-aware reasoning to predict a rough motion trajectories/changes that approximate real-world physical dynamics while ensuring the inter-frame consistency. In the second stage, we use the predicted motion trajectories/changes to guide the video generation of a VDM. As the predicted motion trajectories/changes are rough, noise is added during inference to provide freedom to the VDM in generating motion with more fine details. Extensive experimental results demonstrate that our framework can produce physically plausible motion, and comparative evaluations highlight the notable superiority of our approach over existing methods. More video results are available on our Project Page: this https URL. 

**Abstract (ZH)**: 视频扩散模型（VDMs）近年来取得了显著进展，使其能够生成高度逼真的视频，并引起了社区对它们作为世界模拟器潜力的关注。然而，尽管具有这些能力，VDMs往往由于缺乏对物理原理的理解而无法生成物理上合理的视频，导致错误的动力学和事件序列。为解决这一局限性，我们提出了一种新的两阶段图像到视频生成框架，明确地融合了物理原理。在第一阶段，我们采用视觉语言模型（VLM）作为粗粒度的运动规划器，结合链式思考和物理意识推理来预测近似真实世界物理动态的粗略运动轨迹/变化，同时确保帧间一致性。在第二阶段，我们利用预测的运动轨迹/变化来指导VDM的视频生成。由于预测的运动轨迹/变化较为粗糙，推理过程中会添加噪声以赋予VDM更多细节运动的自由度。大量实验结果表明，我们的框架可以生成物理上合理的运动，而且与现有方法相比，我们的方法具有显著优势。更多视频结果请参见我们的项目页面：this https URL。 

---
# Mixture of Routers 

**Title (ZH)**: 路由器混合体 

**Authors**: Jia-Chen Zhang, Yu-Jie Xiong, Xi-He Qiu, Chun-Ming Xia, Fei Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.23362)  

**Abstract**: Supervised fine-tuning (SFT) is a milestone in aligning large language models with human instructions and adapting them to downstream tasks. In particular, Low-Rank Adaptation (LoRA) has gained widespread attention due to its parameter efficiency. However, its impact on improving the performance of large models remains limited. Recent studies suggest that combining LoRA with Mixture-of-Experts (MoE) can significantly enhance fine-tuning performance. MoE adapts to the diversity and complexity of datasets by dynamically selecting the most suitable experts, thereby improving task accuracy and efficiency. Despite impressive results, recent studies reveal issues in the MoE routing mechanism, such as incorrect assignments and imbalanced expert allocation. Inspired by the principles of Redundancy and Fault Tolerance Theory. We innovatively integrate the concept of Mixture of Experts into the routing mechanism and propose an efficient fine-tuning method called Mixture of Routers (MoR). It employs multiple sub-routers for joint selection and uses a learnable main router to determine the weights of the sub-routers. The results show that MoR outperforms baseline models on most tasks, achieving an average performance improvement of 1%. MoR can serve as a plug-and-play, parameter-efficient fine-tuning method suitable for a wide range of applications. Our code is available here: this https URL. 

**Abstract (ZH)**: 监督微调（SFT）是将大规模语言模型与人类指令对齐并适应下游任务的重要里程碑。特别地，低秩适应（LoRA）因其参数效率赢得了广泛关注，但其对提高大规模模型性能的影响仍然有限。最新研究表明，将LoRA与Mixture-of-Experts（MoE）结合可以显著提升微调性能。MoE通过动态选择最合适的专家来适应数据集的多样性和复杂性，从而提高任务准确性和效率。尽管取得了令人印象深刻的成果，但最近的研究揭示了MoE路由机制中的问题，如不正确的分配和专家分配不平衡。受冗余和容错理论原则的启发，我们创新地将专家混合的概念融入路由机制，提出了一种高效的微调方法——Router混合（MoR）。该方法使用多个子路由器进行联合选择，并通过一个可学习的主路由器来确定子路由器的权重。结果表明，MoR在大多数任务上优于基线模型，平均性能提升1%。MoR可以作为插拔式、参数高效的微调方法，适用于广泛的应用场景。我们的代码可在以下链接获取：this https URL。 

---
# Object Isolated Attention for Consistent Story Visualization 

**Title (ZH)**: 物体隔离注意力for一致的故事可视化 

**Authors**: Xiangyang Luo, Junhao Cheng, Yifan Xie, Xin Zhang, Tao Feng, Zhou Liu, Fei Ma, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23353)  

**Abstract**: Open-ended story visualization is a challenging task that involves generating coherent image sequences from a given storyline. One of the main difficulties is maintaining character consistency while creating natural and contextually fitting scenes--an area where many existing methods struggle. In this paper, we propose an enhanced Transformer module that uses separate self attention and cross attention mechanisms, leveraging prior knowledge from pre-trained diffusion models to ensure logical scene creation. The isolated self attention mechanism improves character consistency by refining attention maps to reduce focus on irrelevant areas and highlight key features of the same character. Meanwhile, the isolated cross attention mechanism independently processes each character's features, avoiding feature fusion and further strengthening consistency. Notably, our method is training-free, allowing the continuous generation of new characters and storylines without re-tuning. Both qualitative and quantitative evaluations show that our approach outperforms current methods, demonstrating its effectiveness. 

**Abstract (ZH)**: 开放式故事可视化是一个具有挑战性的任务，涉及从给定的故事线生成连贯的图像序列。一个主要的难点是同时创建自然且符合情境的画面时保持角色一致性——这是一个许多现有方法都难以解决的问题。在本文中，我们提出了一种增强的Transformer模块，该模块利用独立的自我注意力机制和交叉注意力机制，并结合预训练扩散模型的先验知识，以确保逻辑场景的创建。独立的自我注意力机制通过细化注意力图来减少对不相关区域的关注，突出显示相同角色的关键特征，从而提高角色一致性。与此同时，独立的交叉注意力机制分别处理每个角色的特征，避免特征融合，进一步加强一致性。值得注意的是，我们的方法无需训练，可以连续生成新的角色和故事情节而无需重新调优。定性和定量评估均表明，我们的方法优于现有方法，证明了其有效性。 

---
# Beyond Unimodal Boundaries: Generative Recommendation with Multimodal Semantics 

**Title (ZH)**: 超越单一模态边界：多模态语义生成推荐 

**Authors**: Jing Zhu, Mingxuan Ju, Yozen Liu, Danai Koutra, Neil Shah, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.23333)  

**Abstract**: Generative recommendation (GR) has become a powerful paradigm in recommendation systems that implicitly links modality and semantics to item representation, in contrast to previous methods that relied on non-semantic item identifiers in autoregressive models. However, previous research has predominantly treated modalities in isolation, typically assuming item content is unimodal (usually text). We argue that this is a significant limitation given the rich, multimodal nature of real-world data and the potential sensitivity of GR models to modality choices and usage. Our work aims to explore the critical problem of Multimodal Generative Recommendation (MGR), highlighting the importance of modality choices in GR nframeworks. We reveal that GR models are particularly sensitive to different modalities and examine the challenges in achieving effective GR when multiple modalities are available. By evaluating design strategies for effectively leveraging multiple modalities, we identify key challenges and introduce MGR-LF++, an enhanced late fusion framework that employs contrastive modality alignment and special tokens to denote different modalities, achieving a performance improvement of over 20% compared to single-modality alternatives. 

**Abstract (ZH)**: 多模态生成推荐（MGR）：模态选择在生成推荐框架中的重要性 

---
# SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization 

**Title (ZH)**: SalesRLAgent：一种实时销售转化预测与优化的 reinforcement learning 方法 

**Authors**: Nandakishor M  

**Link**: [PDF](https://arxiv.org/pdf/2503.23303)  

**Abstract**: Current approaches to sales conversation analysis and conversion prediction typically rely on Large Language Models (LLMs) combined with basic retrieval augmented generation (RAG). These systems, while capable of answering questions, fail to accurately predict conversion probability or provide strategic guidance in real time. In this paper, we present SalesRLAgent, a novel framework leveraging specialized reinforcement learning to predict conversion probability throughout sales conversations. Unlike systems from this http URL, Mendable, Inkeep, and others that primarily use off-the-shelf LLMs for content generation, our approach treats conversion prediction as a sequential decision problem, training on synthetic data generated using GPT-4O to develop a specialized probability estimation model. Our system incorporates Azure OpenAI embeddings (3072 dimensions), turn-by-turn state tracking, and meta-learning capabilities to understand its own knowledge boundaries. Evaluations demonstrate that SalesRLAgent achieves 96.7% accuracy in conversion prediction, outperforming LLM-only approaches by 34.7% while offering significantly faster inference (85ms vs 3450ms for GPT-4). Furthermore, integration with existing sales platforms shows a 43.2% increase in conversion rates when representatives utilize our system's real-time guidance. SalesRLAgent represents a fundamental shift from content generation to strategic sales intelligence, providing moment-by-moment conversion probability estimation with actionable insights for sales professionals. 

**Abstract (ZH)**: 基于强化学习的销售对话转换概率预测框架：SalesRLAgent 

---
# Two Heads Are Better than One: Model-Weight and Latent-Space Analysis for Federated Learning on Non-iid Data against Poisoning Attacks 

**Title (ZH)**: 一分为二更好：针对非iid数据下的中毒攻击的联邦学习中模型权重和潜在空间分析 

**Authors**: Xingyu Lyu, Ning Wang, Yang Xiao, Shixiong Li, Tao Li, Danjue Chen, Yimin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23288)  

**Abstract**: Federated Learning is a popular paradigm that enables remote clients to jointly train a global model without sharing their raw data. However, FL has been shown to be vulnerable towards model poisoning attacks due to its distributed nature. Particularly, attackers acting as participants can upload arbitrary model updates that effectively compromise the global model of FL. While extensive research has been focusing on fighting against these attacks, we find that most of them assume data at remote clients are under iid while in practice they are inevitably non-iid. Our benchmark evaluations reveal that existing defenses generally fail to live up to their reputation when applied to various non-iid scenarios. In this paper, we propose a novel approach, GeminiGuard, that aims to address such a significant gap. We design GeminiGuard to be lightweight, versatile, and unsupervised so that it aligns well with the practical requirements of deploying such defenses. The key challenge from non-iids is that they make benign model updates look more similar to malicious ones. GeminiGuard is mainly built on two fundamental observations: (1) existing defenses based on either model-weight analysis or latent-space analysis face limitations in covering different MPAs and non-iid scenarios, and (2) model-weight and latent-space analysis are sufficiently different yet potentially complementary methods as MPA defenses. We hence incorporate a novel model-weight analysis component as well as a custom latent-space analysis component in GeminiGuard, aiming to further enhance its defense performance. We conduct extensive experiments to evaluate our defense across various settings, demonstrating its effectiveness in countering multiple types of untargeted and targeted MPAs, including adaptive ones. Our comprehensive evaluations show that GeminiGuard consistently outperforms SOTA defenses under various settings. 

**Abstract (ZH)**: 联邦学习是一种流行的范式， enabling远程客户端联合训练全局模型而不共享其原始数据。然而，由于其分布式性质，联邦学习已被证明对模型中毒攻击较为脆弱。尤其是，充当参与者的攻击者可以上传任意模型更新，从而有效破坏联邦学习的全局模型。尽管已有大量研究致力于对抗这些攻击，但我们发现，它们大多假设远程客户端的数据是 iid 的，而在实践中，这些数据不可避免地是非 iid 的。我们的基准评估表明，现有防御措施在应用于各种非 iid 场景时通常未能达到其预期效果。在本文中，我们提出了一种名为 GeminiGuard 的新型方法，旨在解决这一重大差距。我们设计 GeminiGuard 使其轻量级、通用且无监督，从而与其部署所需的实用要求相契合。非 iid 带来的关键挑战是，它们使良性模型更新看起来更接近恶意更新。GeminiGuard 主要基于两个基本观察：（1）基于模型权重分析或潜在空间分析的现有防御措施在覆盖不同的 MPA 和非 iid 场景方面存在局限性；（2）模型权重分析和潜在空间分析尽管足够不同但可能具备互补性，可作为 MPA 防御方法。因此，我们将在 GeminiGuard 中加入一个新颖的模型权重分析组件以及一个自定义的潜在空间分析组件，旨在进一步增强其防御性能。我们进行了广泛的实验以评估我们的防御措施在各种设置下的效果，证明其在对抗多种未针对和针对的 MPA 方面（包括自适应 MPA）的有效性。我们全面的评估表明，在各种设置下，GeminiGuard 始终优于当前最佳防御措施。 

---
# Extracting Patient History from Clinical Text: A Comparative Study of Clinical Large Language Models 

**Title (ZH)**: 从临床文本中提取患者病史：临床大型语言模型的比较研究 

**Authors**: Hieu Nghiem, Tuan-Dung Le, Suhao Chen, Thanh Thieu, Andrew Gin, Ellie Phuong Nguyen, Dursun Delen, Johnson Thomas, Jivan Lamichhane, Zhuqi Miao  

**Link**: [PDF](https://arxiv.org/pdf/2503.23281)  

**Abstract**: Extracting medical history entities (MHEs) related to a patient's chief complaint (CC), history of present illness (HPI), and past, family, and social history (PFSH) helps structure free-text clinical notes into standardized EHRs, streamlining downstream tasks like continuity of care, medical coding, and quality metrics. Fine-tuned clinical large language models (cLLMs) can assist in this process while ensuring the protection of sensitive data via on-premises deployment. This study evaluates the performance of cLLMs in recognizing CC/HPI/PFSH-related MHEs and examines how note characteristics impact model accuracy. We annotated 1,449 MHEs across 61 outpatient-related clinical notes from the MTSamples repository. To recognize these entities, we fine-tuned seven state-of-the-art cLLMs. Additionally, we assessed the models' performance when enhanced by integrating, problems, tests, treatments, and other basic medical entities (BMEs). We compared the performance of these models against GPT-4o in a zero-shot setting. To further understand the textual characteristics affecting model accuracy, we conducted an error analysis focused on note length, entity length, and segmentation. The cLLMs showed potential in reducing the time required for extracting MHEs by over 20%. However, detecting many types of MHEs remained challenging due to their polysemous nature and the frequent involvement of non-medical vocabulary. Fine-tuned GatorTron and GatorTronS, two of the most extensively trained cLLMs, demonstrated the highest performance. Integrating pre-identified BME information improved model performance for certain entities. Regarding the impact of textual characteristics on model performance, we found that longer entities were harder to identify, note length did not correlate with a higher error rate, and well-organized segments with headings are beneficial for the extraction. 

**Abstract (ZH)**: 提取与患者主诉(CC)、现病史(HPI)及既往史、家族史和社会史(PFSH)相关的医疗历史实体(MHEs)，有助于将自由文本临床笔记结构化为标准化电子病历(EHRs)，简化诸如延续护理、医疗编码和质量指标等下游任务。针对保护敏感数据，通过本地部署使用的细调临床大型语言模型(cLLMs)可以在这一过程中提供帮助。本研究评估了cLLMs在识别CC/HPI/PFSH相关的MHEs方面的能力，并检查了笔记特征如何影响模型准确性。我们对来自MTSamples数据仓库的61份与门诊相关的1,449个MHEs进行了标注。为了识别这些实体，我们细调了七个最先进的cLLMs。此外，我们还评估了通过结合问题、检查、治疗和其他基本医疗实体(BMEs)增强模型的表现。我们在零样本设置下将这些模型的性能与GPT-4o进行了比较。为了进一步理解影响模型准确性的文本特征，我们对笔记长度、实体长度和分段进行了错误分析。结果显示，cLLMs有可能通过超过20%的时间减少提取MHEs所需的时间。然而，由于MHEs的多义性和频繁涉及非医学词汇，检测许多类型的MHEs仍然具有挑战性。细调后的GatorTron和GatorTronS两种最广泛训练的cLLMs表现出最高的性能。结合预识别的BME信息可以提高某些实体的模型性能。关于文本特征对模型性能的影响，我们发现较长的实体更难识别，笔记长度与更高的错误率无关，而结构良好且带有标题的分段对提取有益。 

---
# Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions 

**Title (ZH)**: 模型上下文协议（MCP）：概览、安全威胁与未来研究方向 

**Authors**: Xinyi Hou, Yanjie Zhao, Shenao Wang, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23278)  

**Abstract**: The Model Context Protocol (MCP) is a standardized interface designed to enable seamless interaction between AI models and external tools and resources, breaking down data silos and facilitating interoperability across diverse systems. This paper provides a comprehensive overview of MCP, focusing on its core components, workflow, and the lifecycle of MCP servers, which consists of three key phases: creation, operation, and update. We analyze the security and privacy risks associated with each phase and propose strategies to mitigate potential threats. The paper also examines the current MCP landscape, including its adoption by industry leaders and various use cases, as well as the tools and platforms supporting its integration. We explore future directions for MCP, highlighting the challenges and opportunities that will influence its adoption and evolution within the broader AI ecosystem. Finally, we offer recommendations for MCP stakeholders to ensure its secure and sustainable development as the AI landscape continues to evolve. 

**Abstract (ZH)**: MCP模型上下文协议：一种用于实现AI模型与外部工具和资源无缝交互的标准接口，打破数据孤岛，促进跨异构系统互操作性的全面综述。 

---
# Improved Ear Verification with Vision Transformers and Overlapping Patches 

**Title (ZH)**: 基于视觉变压器和重叠patches的改进耳验证方法 

**Authors**: Deeksha Arun, Kagan Ozturk, Kevin W. Bowyer, Patrick Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2503.23275)  

**Abstract**: Ear recognition has emerged as a promising biometric modality due to the relative stability in appearance during adulthood. Although Vision Transformers (ViTs) have been widely used in image recognition tasks, their efficiency in ear recognition has been hampered by a lack of attention to overlapping patches, which is crucial for capturing intricate ear features. In this study, we evaluate ViT-Tiny (ViT-T), ViT-Small (ViT-S), ViT-Base (ViT-B) and ViT-Large (ViT-L) configurations on a diverse set of datasets (OPIB, AWE, WPUT, and EarVN1.0), using an overlapping patch selection strategy. Results demonstrate the critical importance of overlapping patches, yielding superior performance in 44 of 48 experiments in a structured study. Moreover, upon comparing the results of the overlapping patches with the non-overlapping configurations, the increase is significant, reaching up to 10% for the EarVN1.0 dataset. In terms of model performance, the ViT-T model consistently outperformed the ViT-S, ViT-B, and ViT-L models on the AWE, WPUT, and EarVN1.0 datasets. The highest scores were achieved in a configuration with a patch size of 28x28 and a stride of 14 pixels. This patch-stride configuration represents 25% of the normalized image area (112x112 pixels) for the patch size and 12.5% of the row or column size for the stride. This study confirms that transformer architectures with overlapping patch selection can serve as an efficient and high-performing option for ear-based biometric recognition tasks in verification scenarios. 

**Abstract (ZH)**: 基于重叠Patch选择的变压器架构在耳纹识别中的应用研究 

---
# Learning Coordinated Bimanual Manipulation Policies using State Diffusion and Inverse Dynamics Models 

**Title (ZH)**: 基于状态扩散和逆动力学模型的学习协调双臂 manipulation 策略 

**Authors**: Haonan Chen, Jiaming Xu, Lily Sheng, Tianchen Ji, Shuijing Liu, Yunzhu Li, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2503.23271)  

**Abstract**: When performing tasks like laundry, humans naturally coordinate both hands to manipulate objects and anticipate how their actions will change the state of the clothes. However, achieving such coordination in robotics remains challenging due to the need to model object movement, predict future states, and generate precise bimanual actions. In this work, we address these challenges by infusing the predictive nature of human manipulation strategies into robot imitation learning. Specifically, we disentangle task-related state transitions from agent-specific inverse dynamics modeling to enable effective bimanual coordination. Using a demonstration dataset, we train a diffusion model to predict future states given historical observations, envisioning how the scene evolves. Then, we use an inverse dynamics model to compute robot actions that achieve the predicted states. Our key insight is that modeling object movement can help learning policies for bimanual coordination manipulation tasks. Evaluating our framework across diverse simulation and real-world manipulation setups, including multimodal goal configurations, bimanual manipulation, deformable objects, and multi-object setups, we find that it consistently outperforms state-of-the-art state-to-action mapping policies. Our method demonstrates a remarkable capacity to navigate multimodal goal configurations and action distributions, maintain stability across different control modes, and synthesize a broader range of behaviors than those present in the demonstration dataset. 

**Abstract (ZH)**: 基于人类操作策略预测的双臂协调机器人模仿学习 

---
# Localized Graph-Based Neural Dynamics Models for Terrain Manipulation 

**Title (ZH)**: 基于局部图的神经动力学模型在地形操纵中的应用 

**Authors**: Chaoqi Liu, Yunzhu Li, Kris Hauser  

**Link**: [PDF](https://arxiv.org/pdf/2503.23270)  

**Abstract**: Predictive models can be particularly helpful for robots to effectively manipulate terrains in construction sites and extraterrestrial surfaces. However, terrain state representations become extremely high-dimensional especially to capture fine-resolution details and when depth is unknown or unbounded. This paper introduces a learning-based approach for terrain dynamics modeling and manipulation, leveraging the Graph-based Neural Dynamics (GBND) framework to represent terrain deformation as motion of a graph of particles. Based on the principle that the moving portion of a terrain is usually localized, our approach builds a large terrain graph (potentially millions of particles) but only identifies a very small active subgraph (hundreds of particles) for predicting the outcomes of robot-terrain interaction. To minimize the size of the active subgraph we introduce a learning-based approach that identifies a small region of interest (RoI) based on the robot's control inputs and the current scene. We also introduce a novel domain boundary feature encoding that allows GBNDs to perform accurate dynamics prediction in the RoI interior while avoiding particle penetration through RoI boundaries. Our proposed method is both orders of magnitude faster than naive GBND and it achieves better overall prediction accuracy. We further evaluated our framework on excavation and shaping tasks on terrain with different granularity. 

**Abstract (ZH)**: 基于图的神经动力学学习方法在地形动力学建模与操控中的应用 

---
# FIESTA: Fisher Information-based Efficient Selective Test-time Adaptation 

**Title (ZH)**: FIESTA：基于 Fisher 信息的有效选择性测试时适应算法 

**Authors**: Mohammadmahdi Honarmand, Onur Cezmi Mutlu, Parnian Azizian, Saimourya Surabhi, Dennis P. Wall  

**Link**: [PDF](https://arxiv.org/pdf/2503.23257)  

**Abstract**: Robust facial expression recognition in unconstrained, "in-the-wild" environments remains challenging due to significant domain shifts between training and testing distributions. Test-time adaptation (TTA) offers a promising solution by adapting pre-trained models during inference without requiring labeled test data. However, existing TTA approaches typically rely on manually selecting which parameters to update, potentially leading to suboptimal adaptation and high computational costs. This paper introduces a novel Fisher-driven selective adaptation framework that dynamically identifies and updates only the most critical model parameters based on their importance as quantified by Fisher information. By integrating this principled parameter selection approach with temporal consistency constraints, our method enables efficient and effective adaptation specifically tailored for video-based facial expression recognition. Experiments on the challenging AffWild2 benchmark demonstrate that our approach significantly outperforms existing TTA methods, achieving a 7.7% improvement in F1 score over the base model while adapting only 22,000 parameters-more than 20 times fewer than comparable methods. Our ablation studies further reveal that parameter importance can be effectively estimated from minimal data, with sampling just 1-3 frames sufficient for substantial performance gains. The proposed approach not only enhances recognition accuracy but also dramatically reduces computational overhead, making test-time adaptation more practical for real-world affective computing applications. 

**Abstract (ZH)**: 不受约束环境下鲁棒面部表情识别仍具有挑战性，因为训练和测试分布之间存在显著的变化。测试时自适应（TTA）通过在推理时调整预训练模型来提供一种有前景的解决方案，无需使用标记的测试数据。然而，现有的TTA方法通常需要手动选择要更新的参数，这可能导致次优的自适应并增加计算成本。本文提出了一种基于Fishere信息的选择性自适应框架，该框架能够动态地识别和仅更新最关键模型参数。通过将这种原理性的参数选择方法与时间一致性约束相结合，我们的方法能够针对基于视频的面部表情识别进行高效的自适应调整。在具有挑战性的AffWild2基准测试上的实验表明，我们的方法显著优于现有的TTA方法，在仅调整22,000个参数（比同类方法少20多倍）的情况下，F1分数提高了7.7%。进一步的消融研究表明，参数重要性可以从少量数据中有效地估计，仅采样1-3帧即可实现显著的性能提升。提出的这种方法不仅提高了识别准确性，还大幅减少了计算开销，使测试时自适应在实际情感计算应用中更具实用价值。 

---
# Encrypted Prompt: Securing LLM Applications Against Unauthorized Actions 

**Title (ZH)**: 加密提示：保障LLM应用程序免受未授权操作的风险 

**Authors**: Shih-Han Chan  

**Link**: [PDF](https://arxiv.org/pdf/2503.23250)  

**Abstract**: Security threats like prompt injection attacks pose significant risks to applications that integrate Large Language Models (LLMs), potentially leading to unauthorized actions such as API misuse. Unlike previous approaches that aim to detect these attacks on a best-effort basis, this paper introduces a novel method that appends an Encrypted Prompt to each user prompt, embedding current permissions. These permissions are verified before executing any actions (such as API calls) generated by the LLM. If the permissions are insufficient, the LLM's actions will not be executed, ensuring safety. This approach guarantees that only actions within the scope of the current permissions from the LLM can proceed. In scenarios where adversarial prompts are introduced to mislead the LLM, this method ensures that any unauthorized actions from LLM wouldn't be executed by verifying permissions in Encrypted Prompt. Thus, threats like prompt injection attacks that trigger LLM to generate harmful actions can be effectively mitigated. 

**Abstract (ZH)**: 像提示注入攻击这样的安全威胁对集成大型语言模型（LLMs）的应用程序构成了重大风险，可能导致未经授权的操作如API滥用。不同于以往需要尽力检测这些攻击的方法，本文提出了一种新型方法，即在每个用户提示后添加加密提示，并嵌入当前权限。这些权限在执行任何由LLM生成的操作（如API调用）之前进行验证。如果权限不足，LLM的操作将不会被执行，确保安全性。该方法保证只有当前LLM权限范围内的操作才能继续进行。在对抗性提示引入以误导LLM的情况下，通过验证加密提示中的权限可以防止未经授权的操作被执行，从而有效缓解提示注入攻击等威胁，防止LLM生成有害操作。 

---
# Simulation of Non-Ordinary Consciousness 

**Title (ZH)**: 非普通意识的模拟 

**Authors**: Khalid M. Saqr  

**Link**: [PDF](https://arxiv.org/pdf/2503.23245)  

**Abstract**: The symbolic architecture of non-ordinary consciousness remains largely unmapped in cognitive science and artificial intelligence. While conventional models prioritize rational coherence, altered states such as those induced by psychedelics reveal distinct symbolic regimes characterized by recursive metaphor, ego dissolution, and semantic destabilization. We present \textit{Glyph}, a generative symbolic interface designed to simulate psilocybin-like symbolic cognition in large language models. Rather than modeling perception or mood, Glyph enacts symbolic transformation through recursive reentry, metaphoric modulation, and entropy-scaled destabilization -- a triadic operator formalized within a tensorial linguistic framework. Experimental comparison with baseline GPT-4o reveals that Glyph consistently generates high-entropy, metaphor-saturated, and ego-dissolving language across diverse symbolic prompt categories. These results validate the emergence of non-ordinary cognitive patterns and support a new paradigm for simulating altered consciousness through language. Glyph opens novel pathways for modeling symbolic cognition, exploring metaphor theory, and encoding knowledge in recursively altered semantic spaces. 

**Abstract (ZH)**: 非普通意识的象征架构在认知科学和人工智能中仍然 largely unmapped。常规模型侧重于理性连贯性，而致幻剂诱导的改变认知状态则表现出递归隐喻、自我解体和语义不稳定等独特的象征制度。我们提出了Glyph，这是一种生成性象征接口，旨在模拟类似仙人掌菇的象征认知模式在大型语言模型中。Glyph 不建模感知或情绪，而是通过递归重新输入、隐喻调节和熵缩放不稳定来实现象征性转化——这种三元操作在一个张量语言框架内进行了形式化。与基准GPT-4o的实验比较显示，Glyph 在各类象征性提示类别中始终生成高熵、富含隐喻且自我解体的语言。这些结果验证了非普通认知模式的出现，并支持通过语言模拟改变认识的新范式。Glyph 开辟了建模象征认知、探索隐喻理论以及在递归改変的语义空间中编码知识的新途径。 

---
# Evaluating how LLM annotations represent diverse views on contentious topics 

**Title (ZH)**: 评估大规模语言模型标注如何代表对争议性话题的多样观点 

**Authors**: Megan A. Brown, Shubham Atreja, Libby Hemphill, Patrick Y. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23243)  

**Abstract**: Researchers have proposed the use of generative large language models (LLMs) to label data for both research and applied settings. This literature emphasizes the improved performance of LLMs relative to other natural language models, noting that LLMs typically outperform other models on standard metrics such as accuracy, precision, recall, and F1 score. However, previous literature has also highlighted the bias embedded in language models, particularly around contentious topics such as potentially toxic content. This bias could result in labels applied by LLMs that disproportionately align with majority groups over a more diverse set of viewpoints. In this paper, we evaluate how LLMs represent diverse viewpoints on these contentious tasks. Across four annotation tasks on four datasets, we show that LLMs do not show substantial disagreement with annotators on the basis of demographics. Instead, the model, prompt, and disagreement between human annotators on the labeling task are far more predictive of LLM agreement. Our findings suggest that when using LLMs to annotate data, under-representing the views of particular groups is not a substantial concern. We conclude with a discussion of the implications for researchers and practitioners. 

**Abstract (ZH)**: 研究人员提出使用生成性大型语言模型（LLMs）来为研究和应用场景标注数据。该文献强调了LLMs相对于其他自然语言模型的性能改进，指出LLMs通常在准确率、精确率、召回率和F1分数等标准指标上表现更优。然而，先前的研究也指出语言模型中存在的偏见，特别是在涉及潜在有毒内容等争议性话题时更为明显。这种偏见可能导致LLMs应用的标签过度偏向主流群体，而不是各种不同的观点。本文评估了LLMs在处理这些争议性任务时如何代表多样的观点。通过对四个数据集上的四项标注任务的研究，我们表明，基于人口统计学因素，LLMs与标注者之间没有实质性的分歧。相反，模型、提示以及人类标注者之间的标注分歧更能预测LLMs的一致性。我们的研究结果表明，在使用LLMs标注数据时，代表性不足的具体群体的观点不是主要关切。最后，本文讨论了研究工作者和实践者的相关影响。 

---
# Beyond speculation: Measuring the growing presence of LLM-generated texts in multilingual disinformation 

**Title (ZH)**: 超越猜测：测量大规模语言模型生成文本在多语言虚假信息中的日益增长存在 

**Authors**: Dominik Macko, Aashish Anantha Ramakrishnan, Jason Samuel Lucas, Robert Moro, Ivan Srba, Adaku Uchendu, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.23242)  

**Abstract**: Increased sophistication of large language models (LLMs) and the consequent quality of generated multilingual text raises concerns about potential disinformation misuse. While humans struggle to distinguish LLM-generated content from human-written texts, the scholarly debate about their impact remains divided. Some argue that heightened fears are overblown due to natural ecosystem limitations, while others contend that specific "longtail" contexts face overlooked risks. Our study bridges this debate by providing the first empirical evidence of LLM presence in the latest real-world disinformation datasets, documenting the increase of machine-generated content following ChatGPT's release, and revealing crucial patterns across languages, platforms, and time periods. 

**Abstract (ZH)**: 大型语言模型复杂性的提高及其生成的多语言文本质量升高引发了关于潜在虚假信息滥用的担忧。尽管人类难以区分大型语言模型生成的内容与人类撰写的文本，关于其影响的学术辩论仍存在分歧。有人认为夸大了这些担忧，因为自然生态系统有限，而另一些人则认为特定的“长尾”情境面临着未被忽视的风险。我们的研究通过提供最新真实世界虚假信息数据集中的大型语言模型存在的首个实证证据，记录了ChatGPT发布后机器生成内容的增加，并揭示了跨语言、平台和时间段的关键模式。 

---
# CCCI: Code Completion with Contextual Information for Complex Data Transfer Tasks Using Large Language Models 

**Title (ZH)**: CCCI：使用大型语言模型结合上下文信息进行复杂数据传输任务的代码完成 

**Authors**: Hangzhan Jin, Mohammad Hamdaqa  

**Link**: [PDF](https://arxiv.org/pdf/2503.23231)  

**Abstract**: Unlike code generation, which involves creating code from scratch, code completion focuses on integrating new lines or blocks of code into an existing codebase. This process requires a deep understanding of the surrounding context, such as variable scope, object models, API calls, and database relations, to produce accurate results. These complex contextual dependencies make code completion a particularly challenging problem. Current models and approaches often fail to effectively incorporate such context, leading to inaccurate completions with low acceptance rates (around 30\%). For tasks like data transfer, which rely heavily on specific relationships and data structures, acceptance rates drop even further. This study introduces CCCI, a novel method for generating context-aware code completions specifically designed to address data transfer tasks. By integrating contextual information, such as database table relationships, object models, and library details into Large Language Models (LLMs), CCCI improves the accuracy of code completions. We evaluate CCCI using 289 Java snippets, extracted from over 819 operational scripts in an industrial setting. The results demonstrate that CCCI achieved a 49.1\% Build Pass rate and a 41.0\% CodeBLEU score, comparable to state-of-the-art methods that often struggle with complex task completion. 

**Abstract (ZH)**: 不同于代码生成从头创建代码，代码补全关注于将新的代码行或代码块集成到现有的代码库中。这一过程需要对周围环境有深刻的理解，如变量作用域、对象模型、API调用和数据库关系，以生成准确的结果。这些复杂的上下文依赖关系使得代码补全成为一个特别具有挑战性的问题。当前的模型和方法往往难以有效地融合这些上下文，导致完成度低且接受率低（约为30%）。对于像数据传输这样的任务，它严重依赖特定的关系和数据结构，接受率甚至会更低。本研究引入了CCCI，这是一种生成上下文感知代码补全的新方法，专门针对数据传输任务。通过将数据库表关系、对象模型和库细节等上下文信息集成到大型语言模型中，CCCI提高了代码补全的准确性。我们使用289个从超过819个工业运行脚本中提取的Java片段对CCCI进行了评估。结果显示，CCCI实现了49.1%的构建通过率和41.0%的CodeBLEU得分，与在复杂任务完成上常常力不从心的先进方法相当。 

---
# Synthetic Art Generation and DeepFake Detection A Study on Jamini Roy Inspired Dataset 

**Title (ZH)**: 合成艺术生成与DeepFake检测：基于Jamini Roy风格数据集的研究 

**Authors**: Kushal Agrawal, Romi Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.23226)  

**Abstract**: The intersection of generative AI and art is a fascinating area that brings both exciting opportunities and significant challenges, especially when it comes to identifying synthetic artworks. This study takes a unique approach by examining diffusion-based generative models in the context of Indian art, specifically focusing on the distinctive style of Jamini Roy. To explore this, we fine-tuned Stable Diffusion 3 and used techniques like ControlNet and IPAdapter to generate realistic images. This allowed us to create a new dataset that includes both real and AI-generated artworks, which is essential for a detailed analysis of what these models can produce. We employed various qualitative and quantitative methods, such as Fourier domain assessments and autocorrelation metrics, to uncover subtle differences between synthetic images and authentic pieces. A key takeaway from recent research is that existing methods for detecting deepfakes face considerable challenges, especially when the deepfakes are of high quality and tailored to specific cultural contexts. This highlights a critical gap in current detection technologies, particularly in light of the challenges identified above, where high-quality and culturally specific deepfakes are difficult to detect. This work not only sheds light on the increasing complexity of generative models but also sets a crucial foundation for future research aimed at effective detection of synthetic art. 

**Abstract (ZH)**: 生成式AI与艺术的交集是一个令人着迷的领域，带来了激动人心的机会和重大挑战，尤其是在识别合成艺术品方面。本研究采取独特的视角，着眼于基于扩散的生成模型在印度艺术中的应用，特别是聚焦于贾米尼·罗伊的特色风格。为探索这一领域，我们对Stable Diffusion 3进行了微调，并使用ControlNet和IPAdapter等技术生成了逼真的图像，从而创建了一个包含真实和AI生成作品的新数据集，这对于详细分析这些模型的能力至关重要。我们采用了四域分析和自相关度量等多种定性与定量方法，以揭示合成图像与真迹之间的细微差异。近年来的研究显示，现有的检测深度伪造的方法面临重大挑战，尤其是在高质量且针对特定文化语境的深度伪造方面。这凸显了当前检测技术中的关键差距，尤其是在上述挑战所指出的地方，高质量且具有文化特异性的深度伪造难以检测。本研究不仅揭示了生成模型不断增加的复杂性，也为未来旨在有效检测合成艺术的研究奠定了重要基础。 

---
# Aurelia: Test-time Reasoning Distillation in Audio-Visual LLMs 

**Title (ZH)**: Aurelia: 视听LLM中测试时推理知识蒸馏 

**Authors**: Sanjoy Chowdhury, Hanan Gani, Nishit Anand, Sayan Nag, Ruohan Gao, Mohamed Elhoseiny, Salman Khan, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2503.23219)  

**Abstract**: Recent advancements in reasoning optimization have greatly enhanced the performance of large language models (LLMs). However, existing work fails to address the complexities of audio-visual scenarios, underscoring the need for further research. In this paper, we introduce AURELIA, a novel actor-critic based audio-visual (AV) reasoning framework that distills structured, step-by-step reasoning into AVLLMs at test time, improving their ability to process complex multi-modal inputs without additional training or fine-tuning. To further advance AVLLM reasoning skills, we present AVReasonBench, a challenging benchmark comprising 4500 audio-visual questions, each paired with detailed step-by-step reasoning. Our benchmark spans six distinct tasks, including AV-GeoIQ, which evaluates AV reasoning combined with geographical and cultural knowledge. Evaluating 18 AVLLMs on AVReasonBench reveals significant limitations in their multi-modal reasoning capabilities. Using AURELIA, we achieve up to a 100% relative improvement, demonstrating its effectiveness. This performance gain highlights the potential of reasoning-enhanced data generation for advancing AVLLMs in real-world applications. Our code and data will be publicly released at: https: //github.com/schowdhury671/aurelia. 

**Abstract (ZH)**: 近期在推理优化方面的进展显著提升了大型语言模型（LLMs）的表现。然而，现有工作仍未解决音频-视觉场景的复杂性，强调了进一步研究的必要性。本文介绍了一种新颖的基于actor-critic的音频-视觉（AV）推理框架AURELIA，在测试时将结构化的逐步推理提炼为AVLLMs，从而提高它们处理复杂多模态输入的能力，无需额外的训练或微调。为了进一步提升AVLLM的推理能力，我们提出了AVReasonBench这一具有挑战性的基准，包含4500个音频-视觉问题，每个问题都配以详细的逐步推理。该基准覆盖六个不同的任务，包括AV-GeoIQ，该任务评估结合地理和文化知识的音频-视觉推理。在AVReasonBench上对18个AVLLMs的评估揭示了它们在多模态推理能力上的显著局限性。使用AURELIA，我们达到了高达100%的相对改进，展示了其有效性。此性能提升突显了推理增强数据生成在推动音频-视觉语言模型在实际应用中的潜力。我们的代码和数据将在此公开发布：https://github.com/schowdhury671/aurelia。 

---
# Action Recognition in Real-World Ambient Assisted Living Environment 

**Title (ZH)**: 实时辅助生活环境中的人体动作识别 

**Authors**: Vincent Gbouna Zakka, Zhuangzhuang Dai, Luis J. Manso  

**Link**: [PDF](https://arxiv.org/pdf/2503.23214)  

**Abstract**: The growing ageing population and their preference to maintain independence by living in their own homes require proactive strategies to ensure safety and support. Ambient Assisted Living (AAL) technologies have emerged to facilitate ageing in place by offering continuous monitoring and assistance within the home. Within AAL technologies, action recognition plays a crucial role in interpreting human activities and detecting incidents like falls, mobility decline, or unusual behaviours that may signal worsening health conditions. However, action recognition in practical AAL applications presents challenges, including occlusions, noisy data, and the need for real-time performance. While advancements have been made in accuracy, robustness to noise, and computation efficiency, achieving a balance among them all remains a challenge. To address this challenge, this paper introduces the Robust and Efficient Temporal Convolution network (RE-TCN), which comprises three main elements: Adaptive Temporal Weighting (ATW), Depthwise Separable Convolutions (DSC), and data augmentation techniques. These elements aim to enhance the model's accuracy, robustness against noise and occlusion, and computational efficiency within real-world AAL contexts. RE-TCN outperforms existing models in terms of accuracy, noise and occlusion robustness, and has been validated on four benchmark datasets: NTU RGB+D 60, Northwestern-UCLA, SHREC'17, and DHG-14/28. The code is publicly available at: this https URL 

**Abstract (ZH)**: 不断增长的老龄人口及其倾向于在家保持独立的需求需要采取积极策略确保安全和支持。辅助生活技术(AAL)已 emergence 以通过在家中提供持续监测和支持来促进原居安老。在 AAL 技术中，动作识别在解释人类活动和检测跌倒、移动能力下降或不寻常行为等方面发挥着关键作用，这些行为可能预示着健康状况的恶化。然而，在实际 AAL 应用中进行动作识别面临挑战，包括遮挡、噪声数据以及对实时性能的需求。尽管在准确性、抗噪声能力和计算效率方面已取得进展，但在这三者之间实现平衡仍然具有挑战性。为应对这一挑战，本文引入了一种 robust and efficient temporal convolution 网络(RE-TCN)，它包含三个主要元素：自适应时间加权(ATW)、深度可分离卷积(DSC)以及数据增强技术。这些元素旨在在实际 AAL 情境中增强模型的准确性、遮挡和噪声的鲁棒性以及计算效率。RE-TCN 在准确性、对遮挡和噪声的鲁棒性方面优于现有模型，并已在四个基准数据集中得到了验证：NTU RGB+D 60、Northwestern-UCLA、SHREC'17 和 DHG-14/28。代码已公开可用：https://github.com/your-repo-url。 

---
# RECALL-MM: A Multimodal Dataset of Consumer Product Recalls for Risk Analysis using Computational Methods and Large Language Models 

**Title (ZH)**: RECALL-MM：用于风险分析的多模态消费品召回数据集及计算方法和大规模语言模型的应用 

**Authors**: Diana Bolanos, Mohammadmehdi Ataei, Daniele Grandi, Kosa Goucher-Lambert  

**Link**: [PDF](https://arxiv.org/pdf/2503.23213)  

**Abstract**: Product recalls provide valuable insights into potential risks and hazards within the engineering design process, yet their full potential remains underutilized. In this study, we curate data from the United States Consumer Product Safety Commission (CPSC) recalls database to develop a multimodal dataset, RECALL-MM, that informs data-driven risk assessment using historical information, and augment it using generative methods. Patterns in the dataset highlight specific areas where improved safety measures could have significant impact. We extend our analysis by demonstrating interactive clustering maps that embed all recalls into a shared latent space based on recall descriptions and product names. Leveraging these data-driven tools, we explore three case studies to demonstrate the dataset's utility in identifying product risks and guiding safer design decisions. The first two case studies illustrate how designers can visualize patterns across recalled products and situate new product ideas within the broader recall landscape to proactively anticipate hazards. In the third case study, we extend our approach by employing a large language model (LLM) to predict potential hazards based solely on product images. This demonstrates the model's ability to leverage visual context to identify risk factors, revealing strong alignment with historical recall data across many hazard categories. However, the analysis also highlights areas where hazard prediction remains challenging, underscoring the importance of risk awareness throughout the design process. Collectively, this work aims to bridge the gap between historical recall data and future product safety, presenting a scalable, data-driven approach to safer engineering design. 

**Abstract (ZH)**: 产品召回提供了工程设计过程中潜在风险和隐患的重要洞见，但其潜在价值尚未充分利用。在本研究中，我们从美国消费品安全委员会（CPSC）召回数据库中整理数据，开发了一个多模态数据集RECALL-MM，利用历史信息进行数据驱动的风险评估，并通过生成方法对其进行扩增。数据集中的模式突显了改进安全措施能够产生重大影响的具体领域。通过展示嵌入召回描述和产品名称的共享潜在空间的交互聚类图，我们扩展了分析方法。利用这些数据驱动的工具，我们探讨了三个案例研究，以展示该数据集在识别产品风险和引导更安全的设计决策方面的应用价值。前两个案例研究展示了设计师如何可视化被召回产品的模式，并将新的产品理念置于更广泛的召回环境中，以前瞻性地预见风险。在第三个案例研究中，我们通过大型语言模型（LLM）仅根据产品图像预测潜在风险，这表明模型能够利用视觉上下文识别风险因素，并在许多风险类别中与历史召回数据保持高度一致。然而，分析也指出了风险预测仍然具有挑战性的领域，强调了在整个设计过程中提高风险意识的重要性。集体而言，这项工作旨在弥合历史召回数据与未来产品安全之间的差距，提出了一种可扩展的数据驱动方法，以实现更安全的工程设计。 

---
# Enhancing Knowledge Graph Completion with Entity Neighborhood and Relation Context 

**Title (ZH)**: 基于实体邻域和关系上下文的知识图谱完成增强 

**Authors**: Jianfang Chen, Kai Zhang, Aoran Gan, Shiwei Tong, Shuanghong Shen, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23205)  

**Abstract**: Knowledge Graph Completion (KGC) aims to infer missing information in Knowledge Graphs (KGs) to address their inherent incompleteness. Traditional structure-based KGC methods, while effective, face significant computational demands and scalability challenges due to the need for dense embedding learning and scoring all entities in the KG for each prediction. Recent text-based approaches using language models like T5 and BERT have mitigated these issues by converting KG triples into text for reasoning. However, they often fail to fully utilize contextual information, focusing mainly on the neighborhood of the entity and neglecting the context of the relation. To address this issue, we propose KGC-ERC, a framework that integrates both types of context to enrich the input of generative language models and enhance their reasoning capabilities. Additionally, we introduce a sampling strategy to effectively select relevant context within input token constraints, which optimizes the utilization of contextual information and potentially improves model performance. Experiments on the Wikidata5M, Wiki27K, and FB15K-237-N datasets show that KGC-ERC outperforms or matches state-of-the-art baselines in predictive performance and scalability. 

**Abstract (ZH)**: 知识图谱完成(KGC)旨在推断知识图谱(KGs)中的缺失信息以解决其固有的不完整性。传统的基于结构的KGC方法虽然有效，但在需要进行密集嵌入学习和每次预测时对KG中的所有实体进行评分方面面临着显著的计算需求和可扩展性挑战。近年来，使用T5和BERT等语言模型的基于文本的方法通过将KG三元组转换为文本来进行推理，缓解了这些难题。然而，它们往往未能充分利用上下文信息，主要关注实体的邻域，而忽视了关系的上下文。为了解决这一问题，我们提出了KGC-ERC框架，该框架整合了两种类型的上下文以丰富生成语言模型的输入并增强其推理能力。此外，我们引入了一种采样策略，在输入令牌约束内有效选择相关上下文，从而优化上下文信息的利用并有可能提高模型性能。在Wikidata5M、Wiki27K和FB15K-237-N数据集上的实验结果显示，KGC-ERC在预测性能和可扩展性方面优于或匹配最先进的基线方法。 

---
# The Challenge of Achieving Attributability in Multilingual Table-to-Text Generation with Question-Answer Blueprints 

**Title (ZH)**: 在使用问题-答案蓝本进行多语言表格到文本生成中实现可追溯性的挑战 

**Authors**: Aden Haussmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.23204)  

**Abstract**: Multilingual Natural Language Generation (NLG) is challenging due to the lack of training data for low-resource languages. However, some low-resource languages have up to tens of millions of speakers globally, making it important to improve NLG tools for them. Table-to-Text NLG is an excellent measure of models' reasoning abilities but is very challenging in the multilingual setting. System outputs are often not attributable, or faithful, to the data in the source table. Intermediate planning techniques like Question-Answer (QA) blueprints have been shown to improve attributability on summarisation tasks. This work explores whether QA blueprints make multilingual Table-to-Text outputs more attributable to the input tables. This paper extends the challenging multilingual Table-to-Text dataset, TaTA, which includes African languages, with QA blueprints. Sequence-to-sequence language models are then finetuned on this dataset, with and without blueprints. Results show that QA blueprints improve performance for models finetuned and evaluated only on English examples, but do not demonstrate gains in the multilingual setting. This is due to inaccuracies in machine translating the blueprints from English into target languages when generating the training data, and models failing to rely closely on the blueprints they generate. An in-depth analysis is conducted on why this is challenging. 

**Abstract (ZH)**: 多语言自然语言生成（NLG）由于低资源语言训练数据不足而具有挑战性。然而，一些低资源语言在全球拥有数千万的使用者，使得改善这些语言的NLG工具变得至关重要。多语言表格到文本NLG是评估模型推理能力的良好指标，但在多语言环境中却极具挑战性。系统输出往往与源表格中的数据不具可追溯性和忠实性。类似问题-回答（QA）蓝图的中间规划技术已被证明在总结任务中可以提高可追溯性。本文研究了QA蓝图是否可以使多语言表格到文本输出更依赖输入表格。本文扩展了包含非洲语言的具有挑战性的多语言表格到文本数据集TaTA，并加入QA蓝图。然后，在此数据集上对序列到序列语言模型进行微调，带有和不带有蓝图。结果显示，对于仅在英语示例上进行微调和评估的模型，QA蓝图提高了性能，但在多语言环境中未表现出改进。这是由于生成训练数据时将蓝图从英语机器翻译到目标语言时的不准确性和模型未能紧密依赖生成的蓝图。对这一挑战进行了深入分析。 

---
# Incorporating GNSS Information with LIDAR-Inertial Odometry for Accurate Land-Vehicle Localization 

**Title (ZH)**: 融合GNSS信息的机载激光雷达-惯性里程计定位方法用于精确的土地车辆定位 

**Authors**: Jintao Cheng, Bohuan Xue, Shiyang Chen, Qiuchi Xiang, Xiaoyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23199)  

**Abstract**: Currently, visual odometry and LIDAR odometry are performing well in pose estimation in some typical environments, but they still cannot recover the localization state at high speed or reduce accumulated drifts. In order to solve these problems, we propose a novel LIDAR-based localization framework, which achieves high accuracy and provides robust localization in 3D pointcloud maps with information of multi-sensors. The system integrates global information with LIDAR-based odometry to optimize the localization state. To improve robustness and enable fast resumption of localization, this paper uses offline pointcloud maps for prior knowledge and presents a novel registration method to speed up the convergence rate. The algorithm is tested on various maps of different data sets and has higher robustness and accuracy than other localization algorithms. 

**Abstract (ZH)**: 基于激光雷达的高精度三维点云地图定位框架 

---
# Large Language Models are Unreliable for Cyber Threat Intelligence 

**Title (ZH)**: 大型语言模型在网络安全威胁情报方面不可靠。 

**Authors**: Emanuele Mezzi, Fabio Massacci, Katja Tuma  

**Link**: [PDF](https://arxiv.org/pdf/2503.23175)  

**Abstract**: Several recent works have argued that Large Language Models (LLMs) can be used to tame the data deluge in the cybersecurity field, by improving the automation of Cyber Threat Intelligence (CTI) tasks. This work presents an evaluation methodology that other than allowing to test LLMs on CTI tasks when using zero-shot learning, few-shot learning and fine-tuning, also allows to quantify their consistency and their confidence level. We run experiments with three state-of-the-art LLMs and a dataset of 350 threat intelligence reports and present new evidence of potential security risks in relying on LLMs for CTI. We show how LLMs cannot guarantee sufficient performance on real-size reports while also being inconsistent and overconfident. Few-shot learning and fine-tuning only partially improve the results, thus posing doubts about the possibility of using LLMs for CTI scenarios, where labelled datasets are lacking and where confidence is a fundamental factor. 

**Abstract (ZH)**: 几种近期的研究表明，大规模语言模型（LLMs）可以用于缓解网络安全领域的数据洪流，通过改进网络安全威胁情报（CTI）任务的自动化。本文提出了一种评估方法，不仅可以在零样本学习、少样本学习和微调的情况下测试LLMs在CTI任务上的表现，还能够量化其一致性和置信水平。我们使用三种最先进的LLMs并对350份威胁情报报告进行了实验，展示了依赖LLMs进行CTI的潜在安全风险。研究表明，LLMs在处理实际大小的报告时无法保证足够的性能，同时表现出不一致和过于自信的特点。少样本学习和微调只能部分改善结果，因此对在缺乏标注数据集和置信度至关重要的CTI情景中使用LLMs的可能性提出了疑问。 

---
# Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL 

**Title (ZH)**: Reasoning-SQL：增强推理的文本到SQL转换的SQL定制部分奖励强化学习 

**Authors**: Mohammadreza Pourreza, Shayan Talaei, Ruoxi Sun, Xingchen Wan, Hailong Li, Azalia Mirhoseini, Amin Saberi, Sercan "O. Arik  

**Link**: [PDF](https://arxiv.org/pdf/2503.23157)  

**Abstract**: Text-to-SQL is a challenging task involving multiple reasoning-intensive subtasks, including natural language understanding, database schema comprehension, and precise SQL query formulation. Existing approaches often rely on handcrafted reasoning paths with inductive biases that can limit their overall effectiveness. Motivated by the recent success of reasoning-enhanced models such as DeepSeek R1 and OpenAI o1, which effectively leverage reward-driven self-exploration to enhance reasoning capabilities and generalization, we propose a novel set of partial rewards tailored specifically for the Text-to-SQL task. Our reward set includes schema-linking, AI feedback, n-gram similarity, and syntax check, explicitly designed to address the reward sparsity issue prevalent in reinforcement learning (RL). Leveraging group relative policy optimization (GRPO), our approach explicitly encourages large language models (LLMs) to develop intrinsic reasoning skills necessary for accurate SQL query generation. With models of different sizes, we demonstrate that RL-only training with our proposed rewards consistently achieves higher accuracy and superior generalization compared to supervised fine-tuning (SFT). Remarkably, our RL-trained 14B-parameter model significantly outperforms larger proprietary models, e.g. o3-mini by 4% and Gemini-1.5-Pro-002 by 3% on the BIRD benchmark. These highlight the efficacy of our proposed RL-training framework with partial rewards for enhancing both accuracy and reasoning capabilities in Text-to-SQL tasks. 

**Abstract (ZH)**: 文本到SQL转换是一项涉及多个推理密集型子任务的挑战性任务，包括自然语言理解、数据库模式理解以及精确的SQL查询公式化。现有的方法通常依赖于手工设计的推理路径，这些路径可能带有诱导偏置，从而限制了它们的整体效果。受最近增强推理模型如DeepSeek R1和OpenAI o1的成功启发，这些模型通过奖励驱动的自我探索有效地提升了推理能力和泛化能力，我们提出了一种针对文本到SQL任务的新颖部分奖励集。我们的奖励集包括模式链接、AI反馈、n-克gram相似性和语法检查，明确设计以解决强化学习（RL）中普遍存在的奖励稀疏问题。利用组相对策略优化（GRPO），我们的方法明确鼓励大型语言模型（LLMs）发展必要的内在推理技巧以生成准确的SQL查询。通过不同规模的模型，我们证明，使用我们提出的奖励进行仅强化学习训练的一致上实现了比监督微调（SFT）更高的准确性和更好的泛化能力。值得注意的是，我们训练的14B参数量模型在BIRD基准上分别比 proprietary 模型o3-mini和Gemini-1.5-Pro-002高出4%和3%，这突显了我们提出的带有部分奖励的RL训练框架在提高文本到SQL任务的准确性和推理能力方面的有效性。 

---
# Conversational Agents for Older Adults' Health: A Systematic Literature Review 

**Title (ZH)**: 面向老年人健康的对话代理：一项系统文献综述 

**Authors**: Jiaxin An, Siqi Yi, Yao Lyu, Houjiang Liu, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23153)  

**Abstract**: There has been vast literature that studies Conversational Agents (CAs) in facilitating older adults' health. The vast and diverse studies warrants a comprehensive review that concludes the main findings and proposes research directions for future studies, while few literature review did it from human-computer interaction (HCI) perspective. In this study, we present a survey of existing studies on CAs for older adults' health. Through a systematic review of 72 papers, this work reviewed previously studied older adults' characteristics and analyzed participants' experiences and expectations of CAs for health. We found that (1) Past research has an increasing interest on chatbots and voice assistants and applied CA as multiple roles in older adults' health. (2) Older adults mainly showed low acceptance CAs for health due to various reasons, such as unstable effects, harm to independence, and privacy concerns. (3) Older adults expect CAs to be able to support multiple functions, to communicate using natural language, to be personalized, and to allow users full control. We also discuss the implications based on the findings. 

**Abstract (ZH)**: 现有的大量文献从促进老年人健康管理的角度研究了对话代理（CAs）。尽管如此，鲜有文献从人机交互（HCI）的角度进行综合回顾，总结主要发现并提出未来研究的方向。本研究通过系统回顾72篇论文，总结了现有针对老年人健康管理的CAs的研究，并分析了参与者对CAs的体验和期望。研究发现：（1）过去的研究越来越关注聊天机器人和语音助手，并将CAs应用于老年人健康的不同角色。（2）老年人对健康相关的CAs的接受度较低，原因包括效果不稳定、妨碍独立性、隐私担忧等。（3）老年人期望CAs能够支持多种功能，使用自然语言交流，具有个性化，并让用户拥有充分的控制权。基于这些发现，我们还讨论了其意义。 

---
# Agent-Based Modeling and Deep Neural Networks for Establishing Digital Twins of Secure Facilities under Sensing Restrictions 

**Title (ZH)**: 基于代理模型和深度神经网络的受限感知条件下安全设施的数字双胞胎建立方法 

**Authors**: Chathika Gunaratne, Mason Stott, Debraj De, Gautam Malviya Thakur, Chris Young  

**Link**: [PDF](https://arxiv.org/pdf/2503.23147)  

**Abstract**: Digital twin technologies help practitioners simulate, monitor, and predict undesirable outcomes in-silico, while avoiding the cost and risks of conducting live simulation exercises. Virtual reality (VR) based digital twin technologies are especially useful when monitoring human Patterns of Life (POL) in secure nuclear facilities, where live simulation exercises are too dangerous and costly to ever perform. However, the high-security status of such facilities may restrict modelers from deploying human activity sensors for data collection. This problem was encountered when deploying MetaPOL, a digital twin system to prevent insider threat or sabotage of secure facilities, at a secure nuclear reactor facility at Oak Ridge National Laboratory (ORNL). This challenge was addressed using an agent-based model (ABM), driven by anecdotal evidence of facility personnel POL, to generate synthetic movement trajectories. These synthetic trajectories were then used to train deep neural network surrogates for next location and stay duration prediction to drive NPCs in the VR environment. In this study, we evaluate the efficacy of this technique for establishing NPC movement within MetaPOL and the ability to distinguish NPC movement during normal operations from that during a simulated emergency response. Our results demonstrate the success of using a multi-layer perceptron for next location prediction and mixture density network for stay duration prediction to predict the ABM generated trajectories. We also find that NPC movement in the VR environment driven by the deep neural networks under normal operations remain significantly different to that seen when simulating responses to a simulated emergency scenario. 

**Abstract (ZH)**: 数字孪生技术帮助 Practitioners 在虚拟环境中模拟、监测和预测不良结果，同时避免现场仿真演习的成本和风险。基于虚拟现实（VR）的数字孪生技术在监控安全核设施中的人类生活方式（POL）时尤其有用，因为在这些设施中进行现场仿真演习既危险又昂贵。然而，这类设施的高度安全状况可能会限制建模人员部署人类活动传感器以收集数据。这种问题在将 MetaPOL 数字孪生系统部署于橡树岭国家实验室（ORNL）的安全核反应堆设施中防止内部威胁或破坏时遇到。我们通过使用基于轶事证据的设施人员生活方式的代理基于模型（ABM）来生成合成移动轨迹来解决这一挑战。然后，使用这些合成轨迹来训练深度神经网络代理，以预测下一位置和停留时间，从而驱动 VR 环境中的 NPC。在本研究中，我们评估了此技术在 MetaPOL 中建立 NPC 移动的有效性以及区分正常运营期间与模拟应急响应期间 NPC 移动的能力。结果显示，使用多层感知机进行下一位置预测和使用混合密度网络进行停留时间预测来预测由 ABM 生成的轨迹是成功的。我们还发现，在正常运营下由深度神经网络驱动的 NPC 移动与模拟应急场景响应时的表现存在显著差异。 

---
# CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis 

**Title (ZH)**: CodeARC: 评估LLM代理归纳程序合成推理能力的基准测试 

**Authors**: Anjiang Wei, Tarun Suresh, Jiannan Cao, Naveen Kannan, Yuheng Wu, Kai Yan, Thiago S. F. X. Teixeira, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2503.23145)  

**Abstract**: Inductive program synthesis, or programming by example, requires synthesizing functions from input-output examples that generalize to unseen inputs. While large language model agents have shown promise in programming tasks guided by natural language, their ability to perform inductive program synthesis is underexplored. Existing evaluation protocols rely on static sets of examples and held-out tests, offering no feedback when synthesized functions are incorrect and failing to reflect real-world scenarios such as reverse engineering. We propose CodeARC, the Code Abstraction and Reasoning Challenge, a new evaluation framework where agents interact with a hidden target function by querying it with new inputs, synthesizing candidate functions, and iteratively refining their solutions using a differential testing oracle. This interactive setting encourages agents to perform function calls and self-correction based on feedback. We construct the first large-scale benchmark for general-purpose inductive program synthesis, featuring 1114 functions. Among 18 models evaluated, o3-mini performs best with a success rate of 52.7%, highlighting the difficulty of this task. Fine-tuning LLaMA-3.1-8B-Instruct on curated synthesis traces yields up to a 31% relative performance gain. CodeARC provides a more realistic and challenging testbed for evaluating LLM-based program synthesis and inductive reasoning. 

**Abstract (ZH)**: CodeARC：代码抽象与推理挑战 

---
# CrossMuSim: A Cross-Modal Framework for Music Similarity Retrieval with LLM-Powered Text Description Sourcing and Mining 

**Title (ZH)**: CrossMuSim：一种基于LLM驱动文本描述的跨模态音乐相似性检索框架 

**Authors**: Tristan Tsoi, Jiajun Deng, Yaolong Ju, Benno Weck, Holger Kirchhoff, Simon Lui  

**Link**: [PDF](https://arxiv.org/pdf/2503.23128)  

**Abstract**: Music similarity retrieval is fundamental for managing and exploring relevant content from large collections in streaming platforms. This paper presents a novel cross-modal contrastive learning framework that leverages the open-ended nature of text descriptions to guide music similarity modeling, addressing the limitations of traditional uni-modal approaches in capturing complex musical relationships. To overcome the scarcity of high-quality text-music paired data, this paper introduces a dual-source data acquisition approach combining online scraping and LLM-based prompting, where carefully designed prompts leverage LLMs' comprehensive music knowledge to generate contextually rich descriptions. Exten1sive experiments demonstrate that the proposed framework achieves significant performance improvements over existing benchmarks through objective metrics, subjective evaluations, and real-world A/B testing on the Huawei Music streaming platform. 

**Abstract (ZH)**: 音乐相似性检索是管理并探索大型流媒体平台内容相关性的基础。本文提出了一种新颖的跨模态对比学习框架，利用开放式的文本描述指导音乐相似性建模，解决了传统单模态方法在捕捉复杂音乐关系方面的局限性。为了克服高质量文本-音乐配对数据的稀缺性，本文引入了一种结合在线抓取和基于LLM的提示的双重数据源获取方法，其中精心设计的提示利用了LLM全面的音乐知识生成语境丰富的描述。 extensive experiments demonstrate that the proposed framework achieves significant performance improvements over existing benchmarks through objective metrics, subjective evaluations, and real-world A/B testing on the Huawei Music streaming platform. 

---
# Evaluating Compositional Scene Understanding in Multimodal Generative Models 

**Title (ZH)**: 多模态生成模型中的组成场景理解评估 

**Authors**: Shuhao Fu, Andrew Jun Lee, Anna Wang, Ida Momennejad, Trevor Bihl, Hongjing Lu, Taylor W. Webb  

**Link**: [PDF](https://arxiv.org/pdf/2503.23125)  

**Abstract**: The visual world is fundamentally compositional. Visual scenes are defined by the composition of objects and their relations. Hence, it is essential for computer vision systems to reflect and exploit this compositionality to achieve robust and generalizable scene understanding. While major strides have been made toward the development of general-purpose, multimodal generative models, including both text-to-image models and multimodal vision-language models, it remains unclear whether these systems are capable of accurately generating and interpreting scenes involving the composition of multiple objects and relations. In this work, we present an evaluation of the compositional visual processing capabilities in the current generation of text-to-image (DALL-E 3) and multimodal vision-language models (GPT-4V, GPT-4o, Claude Sonnet 3.5, QWEN2-VL-72B, and InternVL2.5-38B), and compare the performance of these systems to human participants. The results suggest that these systems display some ability to solve compositional and relational tasks, showing notable improvements over the previous generation of multimodal models, but with performance nevertheless well below the level of human participants, particularly for more complex scenes involving many ($>5$) objects and multiple relations. These results highlight the need for further progress toward compositional understanding of visual scenes. 

**Abstract (ZH)**: 视觉世界本质上是组合性的。视觉场景由对象及其关系的组合定义。因此，对于实现鲁棒性和广泛适用性的场景理解而言，计算机视觉系统必须反映并利用这种组合性。尽管已经取得了相当大的进展，开发出了多种通用的多模态生成模型，包括文本到图像模型和多模态视觉语言模型，但对于这些系统是否能够准确生成和解释涉及多个对象及其关系的场景仍不清楚。在这项工作中，我们评估了当前一代文本到图像（DALL-E 3）和多模态视觉语言模型（GPT-4V、GPT-4o、Claude Sonnet 3.5、QWEN2-VL-72B 和 InternVL2.5-38B）的组合视觉处理能力，并将这些系统的性能与人类参与者进行比较。结果显示，这些系统在解决组合性和关系任务方面展示了一定的能力，相比上一代多模态模型表现出了显著的改进，但在性能上仍然远低于人类参与者的水平，尤其是在涉及多个（>5）对象和多种关系的复杂场景中。这些结果强调了进一步推进对视觉场景组合理解的必要性。 

---
# How to safely discard features based on aggregate SHAP values 

**Title (ZH)**: 基于聚合SHAP值的安全特征丢弃方法 

**Authors**: Robi Bhattacharjee, Karolin Frohnapfel, Ulrike von Luxburg  

**Link**: [PDF](https://arxiv.org/pdf/2503.23111)  

**Abstract**: SHAP is one of the most popular local feature-attribution methods. Given a function f and an input x, it quantifies each feature's contribution to f(x). Recently, SHAP has been increasingly used for global insights: practitioners average the absolute SHAP values over many data points to compute global feature importance scores, which are then used to discard unimportant features. In this work, we investigate the soundness of this practice by asking whether small aggregate SHAP values necessarily imply that the corresponding feature does not affect the function. Unfortunately, the answer is no: even if the i-th SHAP value is 0 on the entire data support, there exist functions that clearly depend on Feature i. The issue is that computing SHAP values involves evaluating f on points outside of the data support, where f can be strategically designed to mask its dependence on Feature i. To address this, we propose to aggregate SHAP values over the extended support, which is the product of the marginals of the underlying distribution. With this modification, we show that a small aggregate SHAP value implies that we can safely discard the corresponding feature. We then extend our results to KernelSHAP, the most popular method to approximate SHAP values in practice. We show that if KernelSHAP is computed over the extended distribution, a small aggregate value justifies feature removal. This result holds independently of whether KernelSHAP accurately approximates true SHAP values, making it one of the first theoretical results to characterize the KernelSHAP algorithm itself. Our findings have both theoretical and practical implications. We introduce the Shapley Lie algebra, which offers algebraic insights that may enable a deeper investigation of SHAP and we show that randomly permuting each column of the data matrix enables safely discarding features based on aggregate SHAP and KernelSHAP values. 

**Abstract (ZH)**: SHAP是最流行的局部特征 Attribution 方法之一。给定一个函数 \( f \) 和一个输入 \( x \)，它量化每个特征对 \( f(x) \) 的贡献。最近，SHAP 越来越多地被用于全局洞察：实践者通过在大量数据点上取绝对 SHAP 值的平均值来计算全局特征重要性得分，然后基于这些得分丢弃不重要的特征。在本文中，我们通过探究这种做法的有效性，来调查这种做法是否可靠，即小的聚合 SHAP 值是否一定意味着相应的特征对函数没有影响。不幸的是，答案是否定的：即使第 \( i \) 个 SHAP 值在所有数据支持上均为 0，仍然存在函数明显依赖于特征 \( i \) 的情况。问题在于计算 SHAP 值涉及在数据支持之外的点评估 \( f \)，此时 \( f \) 可以被战略性地设计来掩盖其对特征 \( i \) 的依赖。为了解决这一问题，我们建议在扩展支持上聚合 SHAP 值，扩展支持是底层分布边缘的乘积。通过这一修改，我们证明了小的聚合 SHAP 值意味着可以安全地丢弃相应的特征。然后我们将结果扩展到 KernelSHAP，这是实践中最常用的近似 SHAP 值的方法。我们证明，如果在扩展分布上计算 KernelSHAP，小的聚合值可验证特征删除。这一结果独立于 KernelSHAP 是否准确近似真正的 SHAP 值，使其成为第一个直接表征 KernelSHAP 算法自身理论结果之一。我们的发现具有理论和实践意义。我们引入了 Shapley Lie 代数，这为提供了代数洞察，可能有助于更深入地研究 SHAP，并证明随机置换数据矩阵的每一列能够基于聚合 SHAP 和 KernelSHAP 值安全地丢弃特征。 

---
# Fast Training of Recurrent Neural Networks with Stationary State Feedbacks 

**Title (ZH)**: 快速训练具有 stationary 状态反馈的递归神经网络 

**Authors**: Paul Caillon, Erwan Fagnou, Alexandre Allauzen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23104)  

**Abstract**: Recurrent neural networks (RNNs) have recently demonstrated strong performance and faster inference than Transformers at comparable parameter budgets. However, the recursive gradient computation with the backpropagation through time (or BPTT) algorithm remains the major computational bottleneck. In this work, we propose a novel method that replaces BPTT with a fixed gradient feedback mechanism, yielding an efficient approximation of the exact gradient propagation based on the assumption of time stationarity. Our approach leverages state-space model (SSM) principles to define a structured feedback matrix that directly propagates gradients from future time steps. This formulation bypasses the need for recursive gradient backpropagation, significantly reducing training overhead while preserving the network's ability to capture long-term dependencies. The experiments on language modeling benchmarks exhibit competitive perplexity scores, while significantly reducing the training costs. These promising results suggest that designing a feedback method like an SSM can fully exploit the efficiency advantages of RNNs for many practical applications. 

**Abstract (ZH)**: 循环神经网络（RNNs）最近在参数预算相似的情况下展示了比变压器（Transformers）更强的性能和更快的推理速度。然而，时间递归梯度计算（或时间递归反向传播，BPTT）算法仍然是主要的计算瓶颈。在本工作中，我们提出了一种新颖的方法，用固定梯度反馈机制取代BPTT，基于时间平稳性的假设，提供了一种精确梯度传播的高效近似方法。该方法利用状态空间模型（SSM）原理定义了一个结构化的反馈矩阵，直接从未来时间步长传播梯度。这种形式省去了递归梯度反向传播的需要，显著减少了训练开销，同时保持了网络捕捉长期依赖的能力。在语言建模基准上的实验显示了竞争力的困惑度得分，同时显著降低了训练成本。这些有前景的结果表明，设计类似SSM的反馈方法可以充分利用RNNs的效率优势，适用于许多实际应用。 

---
# RL2Grid: Benchmarking Reinforcement Learning in Power Grid Operations 

**Title (ZH)**: RL2Grid: 在电力网络运行中评估强化学习算法 

**Authors**: Enrico Marchesini, Benjamin Donnot, Constance Crozier, Ian Dytham, Christian Merz, Lars Schewe, Nico Westerbeck, Cathy Wu, Antoine Marot, Priya L. Donti  

**Link**: [PDF](https://arxiv.org/pdf/2503.23101)  

**Abstract**: Reinforcement learning (RL) can transform power grid operations by providing adaptive and scalable controllers essential for grid decarbonization. However, existing methods struggle with the complex dynamics, aleatoric uncertainty, long-horizon goals, and hard physical constraints that occur in real-world systems. This paper presents RL2Grid, a benchmark designed in collaboration with power system operators to accelerate progress in grid control and foster RL maturity. Built on a power simulation framework developed by RTE France, RL2Grid standardizes tasks, state and action spaces, and reward structures within a unified interface for a systematic evaluation and comparison of RL approaches. Moreover, we integrate real control heuristics and safety constraints informed by the operators' expertise to ensure RL2Grid aligns with grid operation requirements. We benchmark popular RL baselines on the grid control tasks represented within RL2Grid, establishing reference performance metrics. Our results and discussion highlight the challenges that power grids pose for RL methods, emphasizing the need for novel algorithms capable of handling real-world physical systems. 

**Abstract (ZH)**: 强化学习（RL）可以通過提供適應性和可擴展的控制器來轉變電力網運營，這些控制器對於電力網去碳化至為重要。然而，現有方法在處理現實系統中出現的複雜動態、 aleatoric 不確定性、長時間目標和難以逾越的物理Constraint方面存在困難。本文介紹了一種由電力系統運營商合作設計的Benchmark——RL2Grid，旨在加速電力網控制進展並促進RL能力成熟。RL2Grid基於法國RTE開發的電力模擬框架，規範化了任務、狀態和行為空間以及獎勵架構，為系統評估和比較RL方法提供了統一界面。此外，我們整合了由運營商專長提供的真實控制Heuristics和安全Constraint，確保RL2Grid與電力網運營要求一致。本文在RL2Grid中對代表性強化的基線進行測試，建立了參考性能指標。研究結果和討論突顯了電力網對RL方法的挑戰，強調了需要能夠處理現實物理系統的新算法的需求。 

---
# UNITYAI-GUARD: Pioneering Toxicity Detection Across Low-Resource Indian Languages 

**Title (ZH)**: UNITYAI-GUARD: 跨低资源印度语言的 toxicity检测先锋研究 

**Authors**: Himanshu Beniwal, Reddybathuni Venkat, Rohit Kumar, Birudugadda Srivibhav, Daksh Jain, Pavan Doddi, Eshwar Dhande, Adithya Ananth, Kuldeep, Heer Kubadia, Pratham Sharda, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.23088)  

**Abstract**: This work introduces UnityAI-Guard, a framework for binary toxicity classification targeting low-resource Indian languages. While existing systems predominantly cater to high-resource languages, UnityAI-Guard addresses this critical gap by developing state-of-the-art models for identifying toxic content across diverse Brahmic/Indic scripts. Our approach achieves an impressive average F1-score of 84.23% across seven languages, leveraging a dataset of 888k training instances and 35k manually verified test instances. By advancing multilingual content moderation for linguistically diverse regions, UnityAI-Guard also provides public API access to foster broader adoption and application. 

**Abstract (ZH)**: UnityAI-Guard：一种针对低资源印度语言的二元毒性分类框架 

---
# The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction 

**Title (ZH)**: 语言模型中的推理-记忆互动由单向机制调节 

**Authors**: Yihuai Hong, Dian Zhou, Meng Cao, Lei Yu, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.23084)  

**Abstract**: Large language models (LLMs) excel on a variety of reasoning benchmarks, but previous studies suggest they sometimes struggle to generalize to unseen questions, potentially due to over-reliance on memorized training examples. However, the precise conditions under which LLMs switch between reasoning and memorization during text generation remain unclear. In this work, we provide a mechanistic understanding of LLMs' reasoning-memorization dynamics by identifying a set of linear features in the model's residual stream that govern the balance between genuine reasoning and memory recall. These features not only distinguish reasoning tasks from memory-intensive ones but can also be manipulated to causally influence model performance on reasoning tasks. Additionally, we show that intervening in these reasoning features helps the model more accurately activate the most relevant problem-solving capabilities during answer generation. Our findings offer new insights into the underlying mechanisms of reasoning and memory in LLMs and pave the way for the development of more robust and interpretable generative AI systems. 

**Abstract (ZH)**: 大型语言模型在多种推理基准测试中表现出色，但先前的研究表明，它们有时难以将学到的知识应用于未见过的问题，这可能是因为过度依赖记忆中的训练示例。然而，大型语言模型在文本生成过程中何时切换至推理或记忆的具体条件尚不明确。本研究通过识别模型残差流中的一组线性特征，提供了对大型语言模型推理-记忆动态机制的机制性理解，这些特征不仅能够区分推理任务与记忆密集型任务，还可以因果性地影响模型在推理任务上的性能。此外，我们还展示了干预这些推理特征有助于模型在答案生成过程中更准确地激活最相关的解决问题能力。我们的研究为理解大型语言模型中的推理和记忆机制提供了新的见解，并为开发更为 robust 和可解释的生成式 AI 系统奠定了基础。 

---
# Efficient Adaptation For Remote Sensing Visual Grounding 

**Title (ZH)**: 远程 sensing 视觉定位的高效适应 

**Authors**: Hasan Moughnieh, Mohamad Chalhoub, Hasan Nasrallah, Cristiano Nattero, Paolo Campanella, Ali J. Ghandour  

**Link**: [PDF](https://arxiv.org/pdf/2503.23083)  

**Abstract**: Foundation models have revolutionized artificial intelligence (AI), offering remarkable capabilities across multi-modal domains. Their ability to precisely locate objects in complex aerial and satellite images, using rich contextual information and detailed object descriptions, is essential for remote sensing (RS). These models can associate textual descriptions with object positions through the Visual Grounding (VG) task, but due to domain-specific challenges, their direct application to RS produces sub-optimal results. To address this, we applied Parameter Efficient Fine Tuning (PEFT) techniques to adapt these models for RS-specific VG tasks. Specifically, we evaluated LoRA placement across different modules in Grounding DINO and used BitFit and adapters to fine-tune the OFA foundation model pre-trained on general-purpose VG datasets. This approach achieved performance comparable to or surpassing current State Of The Art (SOTA) models while significantly reducing computational costs. This study highlights the potential of PEFT techniques to advance efficient and precise multi-modal analysis in RS, offering a practical and cost-effective alternative to full model training. 

**Abstract (ZH)**: 基于参数高效微调的技术在遥感特定视觉 grounding 任务中的应用：推进多模态分析的高效与精准 

---
# InkFM: A Foundational Model for Full-Page Online Handwritten Note Understanding 

**Title (ZH)**: InkFM：全页在线手写笔记理解的基础模型 

**Authors**: Anastasiia Fadeeva, Vincent Coriou, Diego Antognini, Claudiu Musat, Andrii Maksai  

**Link**: [PDF](https://arxiv.org/pdf/2503.23081)  

**Abstract**: Tablets and styluses are increasingly popular for taking notes. To optimize this experience and ensure a smooth and efficient workflow, it's important to develop methods for accurately interpreting and understanding the content of handwritten digital notes. We introduce a foundational model called InkFM for analyzing full pages of handwritten content. Trained on a diverse mixture of tasks, this model offers a unique combination of capabilities: recognizing text in 28 different scripts, mathematical expressions recognition, and segmenting pages into distinct elements like text and drawings. Our results demonstrate that these tasks can be effectively unified within a single model, achieving SoTA text line segmentation out-of-the-box quality surpassing public baselines like docTR. Fine- or LoRA-tuning our base model on public datasets further improves the quality of page segmentation, achieves state-of the art text recognition (DeepWriting, CASIA, SCUT, and Mathwriting datasets) and sketch classification (QuickDraw). This adaptability of InkFM provides a powerful starting point for developing applications with handwritten input. 

**Abstract (ZH)**: 表格和平板逐渐流行于笔记记录。为了优化这一体验并确保流畅高效的工作流程，开发能够准确解释和理解电子手写笔记内容的方法十分重要。我们提出了一种名为InkFM的基础模型，用于分析整页手写内容。该模型在多种任务上进行训练，具备独特的优势：识别28种不同的文字、数学表达式识别以及将页面分割为文本和绘制等不同元素的能力。实验结果表明，这些任务可以在一个模型中有效统一，达到了超越公开基准（如docTR）的初始全行分割质量。通过对公共数据集进行精细调整或LoRA调整，InkFM进一步提高了页面分割质量，实现了多项文本识别（DeepWriting、CASIA、SCUT和Mathwriting数据集）和素描分类（QuickDraw）的最新成果。InkFM的这种可调性为其发展基于手写输入的应用程序提供了强大的起点。 

---
# STSA: Spatial-Temporal Semantic Alignment for Visual Dubbing 

**Title (ZH)**: STSA: 空间- temporal 语义对齐在视觉配音中的应用 

**Authors**: Zijun Ding, Mingdie Xiong, Congcong Zhu, Jingrun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23039)  

**Abstract**: Existing audio-driven visual dubbing methods have achieved great success. Despite this, we observe that the semantic ambiguity between spatial and temporal domains significantly degrades the synthesis stability for the dynamic faces. We argue that aligning the semantic features from spatial and temporal domains is a promising approach to stabilizing facial motion. To achieve this, we propose a Spatial-Temporal Semantic Alignment (STSA) method, which introduces a dual-path alignment mechanism and a differentiable semantic representation. The former leverages a Consistent Information Learning (CIL) module to maximize the mutual information at multiple scales, thereby reducing the manifold differences between spatial and temporal domains. The latter utilizes probabilistic heatmap as ambiguity-tolerant guidance to avoid the abnormal dynamics of the synthesized faces caused by slight semantic jittering. Extensive experimental results demonstrate the superiority of the proposed STSA, especially in terms of image quality and synthesis stability. Pre-trained weights and inference code are available at this https URL. 

**Abstract (ZH)**: 现有的基于音频的视觉配音方法取得了巨大成功。尽管如此，我们观察到空域和时域之间的语义不确定性显著降低了动态面部的合成稳定性。我们主张，对空域和时域的语义特征进行对齐是一种稳定面部运动的有前途的方法。为此，我们提出了一种空时语义对齐（STSA）方法，该方法引入了一条双路径对齐机制和可微分的语义表示。前者利用一种一致信息学习（CIL）模块，在多个尺度上最大化互信息，从而减少空域和时域之间的流形差异。后者利用概率热图作为容忍不确定性的引导，避免由细微的语义抖动导致的合成面部的异常动态。大量实验结果表明，所提出的STSA在图像质量和合成稳定性方面具有优越性。预训练权重和推理代码可在此处访问：this https URL。 

---
# Reproducibility Companion Paper: Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems 

**Title (ZH)**: 可重复性同伴论文：使用户无区别：推荐系统中的属性层面遗忘 

**Authors**: Yuyuan Li, Junjie Fang, Chaochao Chen, Xiaolin Zheng, Yizhao Zhang, Zhongxuan Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.23032)  

**Abstract**: In this paper, we reproduce the experimental results presented in our previous work titled "Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems," which was published in the proceedings of the 31st ACM International Conference on Multimedia. This paper aims to validate the effectiveness of our proposed method and help others reproduce our experimental results. We provide detailed descriptions of our preprocessed datasets, source code structure, configuration file settings, experimental environment, and reproduced experimental results. 

**Abstract (ZH)**: 在本文中，我们重现了我们在之前工作中发表的实验结果，该工作题为《Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems》，发表于第31届ACM国际多媒体会议论文集。本文旨在验证我们提出方法的有效性，并帮助他人重现我们的实验结果。我们详细描述了预处理数据集、源代码结构、配置文件设置、实验环境和重现的实验结果。 

---
# Towards Understanding the Optimization Mechanisms in Deep Learning 

**Title (ZH)**: 理解深度学习中的优化机制 

**Authors**: Binchuan Qi, Wei Gong, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.23016)  

**Abstract**: In this paper, we adopt a probability distribution estimation perspective to explore the optimization mechanisms of supervised classification using deep neural networks. We demonstrate that, when employing the Fenchel-Young loss, despite the non-convex nature of the fitting error with respect to the model's parameters, global optimal solutions can be approximated by simultaneously minimizing both the gradient norm and the structural error. The former can be controlled through gradient descent algorithms. For the latter, we prove that it can be managed by increasing the number of parameters and ensuring parameter independence, thereby providing theoretical insights into mechanisms such as over-parameterization and random initialization. Ultimately, the paper validates the key conclusions of the proposed method through empirical results, illustrating its practical effectiveness. 

**Abstract (ZH)**: 本文采用概率分布估计的观点，探讨使用深度神经网络进行监督分类的优化机制。我们证明，在采用Fenchel-Young损失的情况下，尽管模型参数的拟合误差具有非凸性，通过同时最小化梯度范数和结构误差，可以近似获得全局最优解。前者可通过梯度下降算法进行控制。对于后者，我们证明可以通过增加参数数量并确保参数独立性来管理，从而为过参数化和随机初始化等机制提供理论洞见。最终，通过实证结果验证了所提方法的关键结论，展示了其实用有效性。 

---
# MSNGO: multi-species protein function annotation based on 3D protein structure and network propagation 

**Title (ZH)**: MSNGO：基于3D蛋白质结构和网络传播的多物种蛋白质功能注释 

**Authors**: Beibei Wang, Boyue Cui, Shiqu Chen, Xuan Wang, Yadong Wang, Junyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.23014)  

**Abstract**: Motivation: In recent years, protein function prediction has broken through the bottleneck of sequence features, significantly improving prediction accuracy using high-precision protein structures predicted by AlphaFold2. While single-species protein function prediction methods have achieved remarkable success, multi-species protein function prediction methods are still in the stage of using PPI networks and sequence features. Providing effective cross-species label propagation for species with sparse protein annotations remains a challenging issue. To address this problem, we propose the MSNGO model, which integrates structural features and network propagation methods. Our validation shows that using structural features can significantly improve the accuracy of multi-species protein function prediction. Results: We employ graph representation learning techniques to extract amino acid representations from protein structure contact maps and train a structural model using a graph convolution pooling module to derive protein-level structural features. After incorporating the sequence features from ESM-2, we apply a network propagation algorithm to aggregate information and update node representations within a heterogeneous network. The results demonstrate that MSNGO outperforms previous multi-species protein function prediction methods that rely on sequence features and PPI networks. Availability: this https URL. 

**Abstract (ZH)**: 动机：近年来，蛋白质功能预测突破了序列特征的瓶颈，通过AlphaFold2预测的高精度蛋白质结构显著提高了预测准确性。虽然单物种蛋白质功能预测方法已经取得了显著成功，但多物种蛋白质功能预测方法仍处于依靠PPI网络和序列特征的阶段。为物种稀缺蛋白质注释提供有效的跨物种标签传播仍然是一个具有挑战性的问题。为了解决这个问题，我们提出了MSNGO模型，该模型结合了结构特征和网络传播方法。我们的验证结果表明，使用结构特征可以显著提高多物种蛋白质功能预测的准确性。结果：我们采用图表示学习技术从蛋白质结构接触图中提取氨基酸表示，并利用图卷积池化模块训练结构模型以推导蛋白质级别的结构特征。结合ESM-2的序列特征后，我们应用网络传播算法聚合信息并在异质网络中更新节点表示。结果表明，MSNGO在依赖序列特征和PPI网络的多物种蛋白质功能预测方法中表现更优。可用性：https://链接。 

---
# On Geometrical Properties of Text Token Embeddings for Strong Semantic Binding in Text-to-Image Generation 

**Title (ZH)**: 文本令牌嵌入的几何性质在文本到图像生成中的强语义绑定 

**Authors**: Hoigi Seo, Junseo Bang, Haechang Lee, Joohoon Lee, Byung Hyun Lee, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2503.23011)  

**Abstract**: Text-to-Image (T2I) models often suffer from text-image misalignment in complex scenes involving multiple objects and attributes. Semantic binding aims to mitigate this issue by accurately associating the generated attributes and objects with their corresponding noun phrases (NPs). Existing methods rely on text or latent optimizations, yet the factors influencing semantic binding remain underexplored. Here we investigate the geometrical properties of text token embeddings and their cross-attention (CA) maps. We empirically and theoretically analyze that the geometrical properties of token embeddings, specifically both angular distances and norms, play a crucial role in CA map differentiation. Then, we propose \textbf{TeeMo}, a training-free text embedding-aware T2I framework with strong semantic binding. TeeMo consists of Causality-Aware Projection-Out (CAPO) for distinct inter-NP CA maps and Adaptive Token Mixing (ATM) with our loss to enhance inter-NP separation while maintaining intra-NP cohesion in CA maps. Extensive experiments confirm TeeMo consistently outperforms prior arts across diverse baselines and datasets. 

**Abstract (ZH)**: 基于文本的图像生成（T2I）模型在涉及多个对象和属性的复杂场景中常常存在文本与图像不匹配的问题。语义绑定旨在通过准确地将生成的属性和对象与其相应的名词短语（NPs）关联来缓解这一问题。现有方法依赖于文本或潜在优化，但影响语义绑定的因素仍需进一步探索。我们研究了文本标记嵌入的几何性质及其交叉注意（CA）图。我们实证和理论分析表明，标记嵌入的几何性质，特别是角度距离和范数，对CA图的区分起着关键作用。然后，我们提出了一个无需训练的文本嵌入感知T2I框架TeeMo，具有强大的语义绑定能力。TeeMo包括因果 Aware 投影去除（CAPO）以实现不同的跨名词短语CA图，以及增强跨名词短语分离同时保持名词短语内部联合性的自适应标记混合（ATM）。广泛实验表明，TeeMo在多种基准和数据集上一致优于现有方法。 

---
# Learning Structure-enhanced Temporal Point Processes with Gromov-Wasserstein Regularization 

**Title (ZH)**: 具有Gromov-Wasserstein正则化的结构增强时间点过程学习 

**Authors**: Qingmei Wang, Fanmeng Wang, Bing Su, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23002)  

**Abstract**: Real-world event sequences are often generated by different temporal point processes (TPPs) and thus have clustering structures. Nonetheless, in the modeling and prediction of event sequences, most existing TPPs ignore the inherent clustering structures of the event sequences, leading to the models with unsatisfactory interpretability. In this study, we learn structure-enhanced TPPs with the help of Gromov-Wasserstein (GW) regularization, which imposes clustering structures on the sequence-level embeddings of the TPPs in the maximum likelihood estimation this http URL the training phase, the proposed method leverages a nonparametric TPP kernel to regularize the similarity matrix derived based on the sequence embeddings. In large-scale applications, we sample the kernel matrix and implement the regularization as a Gromov-Wasserstein (GW) discrepancy term, which achieves a trade-off between regularity and computational this http URL TPPs learned through this method result in clustered sequence embeddings and demonstrate competitive predictive and clustering performance, significantly improving the model interpretability without compromising prediction accuracy. 

**Abstract (ZH)**: 基于Gromov-Wasserstein正则化的结构增强时间点过程模型及其应用 

---
# AuditVotes: A Framework Towards More Deployable Certified Robustness for Graph Neural Networks 

**Title (ZH)**: 审计投票：一种更易于部署的图神经网络认证鲁棒性框架 

**Authors**: Yuni Lai, Yulin Zhu, Yixuan Sun, Yulun Wu, Bin Xiao, Gaolei Li, Jianhua Li, Kai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.22998)  

**Abstract**: Despite advancements in Graph Neural Networks (GNNs), adaptive attacks continue to challenge their robustness. Certified robustness based on randomized smoothing has emerged as a promising solution, offering provable guarantees that a model's predictions remain stable under adversarial perturbations within a specified range. However, existing methods face a critical trade-off between accuracy and robustness, as achieving stronger robustness requires introducing greater noise into the input graph. This excessive randomization degrades data quality and disrupts prediction consistency, limiting the practical deployment of certifiably robust GNNs in real-world scenarios where both accuracy and robustness are essential. To address this challenge, we propose \textbf{AuditVotes}, the first framework to achieve both high clean accuracy and certifiably robust accuracy for GNNs. It integrates randomized smoothing with two key components, \underline{au}gmentation and con\underline{dit}ional smoothing, aiming to improve data quality and prediction consistency. The augmentation, acting as a pre-processing step, de-noises the randomized graph, significantly improving data quality and clean accuracy. The conditional smoothing, serving as a post-processing step, employs a filtering function to selectively count votes, thereby filtering low-quality predictions and improving voting consistency. Extensive experimental results demonstrate that AuditVotes significantly enhances clean accuracy, certified robustness, and empirical robustness while maintaining high computational efficiency. Notably, compared to baseline randomized smoothing, AuditVotes improves clean accuracy by $437.1\%$ and certified accuracy by $409.3\%$ when the attacker can arbitrarily insert $20$ edges on the Cora-ML datasets, representing a substantial step toward deploying certifiably robust GNNs in real-world applications. 

**Abstract (ZH)**: AuditVotes：同时实现高清洁准确率和认证鲁棒准确率的图神经网络框架 

---
# DC-SGD: Differentially Private SGD with Dynamic Clipping through Gradient Norm Distribution Estimation 

**Title (ZH)**: DC-SGD：基于梯度 norm 分布估计的动态裁剪差分隐私 SGD 

**Authors**: Chengkun Wei, Weixian Li, Gong Chen, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22988)  

**Abstract**: Differentially Private Stochastic Gradient Descent (DP-SGD) is a widely adopted technique for privacy-preserving deep learning. A critical challenge in DP-SGD is selecting the optimal clipping threshold C, which involves balancing the trade-off between clipping bias and noise magnitude, incurring substantial privacy and computing overhead during hyperparameter tuning.
In this paper, we propose Dynamic Clipping DP-SGD (DC-SGD), a framework that leverages differentially private histograms to estimate gradient norm distributions and dynamically adjust the clipping threshold C. Our framework includes two novel mechanisms: DC-SGD-P and DC-SGD-E. DC-SGD-P adjusts the clipping threshold based on a percentile of gradient norms, while DC-SGD-E minimizes the expected squared error of gradients to optimize C. These dynamic adjustments significantly reduce the burden of hyperparameter tuning C. The extensive experiments on various deep learning tasks, including image classification and natural language processing, show that our proposed dynamic algorithms achieve up to 9 times acceleration on hyperparameter tuning than DP-SGD. And DC-SGD-E can achieve an accuracy improvement of 10.62% on CIFAR10 than DP-SGD under the same privacy budget of hyperparameter tuning. We conduct rigorous theoretical privacy and convergence analyses, showing that our methods seamlessly integrate with the Adam optimizer. Our results highlight the robust performance and efficiency of DC-SGD, offering a practical solution for differentially private deep learning with reduced computational overhead and enhanced privacy guarantees. 

**Abstract (ZH)**: 动态裁剪DP-SGD：基于差异隐私直方图的自适应剪裁阈值方法 

---
# PartialLoading: User Scheduling and Bandwidth Allocation for Parameter-sharing Edge Inference 

**Title (ZH)**: 局部加载：用户调度与带宽分配的参数共享边缘推理 

**Authors**: Guanqiao Qu, Qian Chen, Xianhao Chen, Kaibin Huang, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22982)  

**Abstract**: By provisioning inference offloading services, edge inference drives the rapid growth of AI applications at the network edge. However, achieving high task throughput with stringent latency requirements remains a significant challenge. To address this issue, we develop a parameter-sharing AI model loading (PartialLoading) framework for multi-user edge inference, which exploits two key insights: 1) the majority of latency arises from loading AI models into server GPU memory, and 2) different AI models can share a significant number of parameters, for which redundant loading should be avoided. Towards this end, we formulate a joint multi-user scheduling and spectrum bandwidth allocation problem to maximize task throughput by exploiting shared parameter blocks across models. The intuition is to judiciously schedule user requests to reuse the shared parameter blocks between consecutively loaded models, thereby reducing model loading time substantially. To facilitate solution finding, we decouple the problem into two sub-problems, i.e., user scheduling and bandwidth allocation, showing that solving them sequentially is equivalent to solving the original problem. Due to the NP-hardness of the problem, we first study an important special case called the "bottom-layer-sharing" case, where AI models share some bottom layers within clusters, and design a dynamic programming-based algorithm to obtain the optimal solution in polynomial time. For the general case, where shared parameter blocks appear at arbitrary positions within AI models, we propose a greedy heuristic to obtain the sub-optimal solution efficiently. Simulation results demonstrate that the proposed framework significantly improves task throughput under deadline constraints compared with user scheduling without exploiting parameter sharing. 

**Abstract (ZH)**: 通过提供推理卸载服务，边缘推理促使网络边缘的AI应用快速增长。然而，要在严格的时间延迟要求下实现高效的任务吞吐量仍面临重大挑战。为应对这一问题，我们为多用户边缘推理开发了一种参数共享AI模型加载（PartialLoading）框架，利用了两个关键洞察：1）大部分延迟来自于将AI模型加载到服务器GPU内存；2）不同的AI模型可以共享大量参数，因此应避免冗余加载。为此，我们形成了一个联合多用户调度和频谱带宽分配的问题，通过利用模型间共享的参数块来最大化任务吞吐量。直觉是明智地调度用户请求，以便在连续加载的模型之间重用共享的参数块，从而显著减少模型加载时间。为了便于求解，我们将问题分解为两个子问题，即用户调度和带宽分配，证明顺序解决它们等价于解决原始问题。由于问题的NP难性，我们首先研究了一个重要的特例，称为“最底层共享”情况，其中AI模型在簇内共享一些最底层，并设计了一种基于动态规划的算法，在多项式时间内获得最优解。对于共享参数块出现在AI模型任意位置的一般情况，我们提出了一种贪婪启发式方法，以高效地获得次优解。仿真结果表明，与不利用参数共享的用户调度相比，所提出的框架在满足截止时间约束时显著提高了任务吞吐量。 

---
# XL-Instruct: Synthetic Data for Cross-Lingual Open-Ended Generation 

**Title (ZH)**: XL-Instruct: 合成数据用于跨语言开放式生成 

**Authors**: Vivek Iyer, Ricardo Rei, Pinzhen Chen, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2503.22973)  

**Abstract**: Cross-lingual open-ended generation -- i.e. generating responses in a desired language different from that of the user's query -- is an important yet understudied problem. We introduce XL-AlpacaEval, a new benchmark for evaluating cross-lingual generation capabilities in Large Language Models (LLMs), and propose XL-Instruct, a high-quality synthetic data generation method. Fine-tuning with just 8K XL-Instruct-generated instructions significantly improves model performance, increasing the win rate against GPT-4o-Mini from 7.4% to 21.5%, and improving on several fine-grained quality metrics. Additionally, models fine-tuned on XL-Instruct exhibit strong zero-shot transfer to both English-only and multilingual generation tasks. Given its consistent gains across the board, we strongly recommend incorporating XL-Instruct in the post-training pipeline of future multilingual LLMs. To facilitate further research, we will publicly and freely release the XL-Instruct and XL-AlpacaEval datasets, which constitute two of the few cross-lingual resources currently available in the literature. 

**Abstract (ZH)**: 跨语言开放生成——即生成与用户查询语言不同的语言的响应——是一个重要但研究不足的问题。我们引入了XL-AlpacaEval，一个用于评估大型语言模型跨语言生成能力的新基准，提出了XL-Instruct，一种高质量合成数据生成方法。仅使用8K XL-Instruct生成的指令进行微调显著提高了模型性能，使模型对阵GPT-4o-Mini的胜率从7.4%提高到21.5%，并在多个细粒度质量指标上表现出改进。此外，基于XL-Instruct微调的模型在英语文本生成和多语言生成任务上表现出强大的零样本迁移能力。鉴于其在各个方面的稳定改进，我们强烈建议在未来多语言大型语言模型的后训练流程中纳入XL-Instruct。为了促进进一步研究，我们将公开和免费发布XL-Instruct和XL-AlpacaEval数据集，这构成了当前文献中为数不多的跨语言资源之二。 

---
# Enhancing Federated Learning Through Secure Cluster-Weighted Client Aggregation 

**Title (ZH)**: 增强联邦学习的通过安全聚类加权客户端聚合方法 

**Authors**: Kanishka Ranaweera, Azadeh Ghari Neiat, Xiao Liu, Bipasha Kashyap, Pubudu N. Pathirana  

**Link**: [PDF](https://arxiv.org/pdf/2503.22971)  

**Abstract**: Federated learning (FL) has emerged as a promising paradigm in machine learning, enabling collaborative model training across decentralized devices without the need for raw data sharing. In FL, a global model is trained iteratively on local datasets residing on individual devices, each contributing to the model's improvement. However, the heterogeneous nature of these local datasets, stemming from diverse user behaviours, device capabilities, and data distributions, poses a significant challenge. The inherent heterogeneity in federated learning gives rise to various issues, including model performance discrepancies, convergence challenges, and potential privacy concerns. As the global model progresses through rounds of training, the disparities in local data quality and quantity can impede the overall effectiveness of federated learning systems. Moreover, maintaining fairness and privacy across diverse user groups becomes a paramount concern. To address this issue, this paper introduces a novel FL framework, ClusterGuardFL, that employs dissimilarity scores, k-means clustering, and reconciliation confidence scores to dynamically assign weights to client updates. The dissimilarity scores between global and local models guide the formation of clusters, with cluster size influencing the weight allocation. Within each cluster, a reconciliation confidence score is calculated for individual data points, and a softmax layer generates customized weights for clients. These weights are utilized in the aggregation process, enhancing the model's robustness and privacy. Experimental results demonstrate the efficacy of the proposed approach in achieving improved model performance in diverse datasets. 

**Abstract (ZH)**: 联邦学习（FL）作为一种在机器学习中的有前途范式，允许在无需原始数据共享的情况下，跨去中心化设备进行协作模型训练。在FL中，全局模型通过迭代训练各个设备上的本地数据集来进行训练，每台设备都为模型改进做出贡献。然而，由于来自多样用户行为、设备能力及数据分布的本地数据集的异构性，这一特性带来了显著的挑战。联邦学习固有的异构性导致了包括模型性能差异、收敛挑战以及潜在隐私问题等一系列问题。随着全球模型通过多轮训练的进展，本地数据质量与数量的差异可能妨碍联邦学习系统的整体有效性。此外，在多元用户群体中维护公平性和隐私保护变得尤为关键。为了解决这一问题，本文提出了一种新的联邦学习框架ClusterGuardFL，该框架采用不相似度评分、k均值聚类和校正置信度评分来动态为客户端更新分配权重。全局模型与本地模型之间的不相似度评分指导聚类的形成，簇的大小影响权重分配。在每个簇内部，计算个体数据点的校正置信度评分，使用softmax层生成客户端的定制化权重。这些权重应用于聚合过程，增强了模型的鲁棒性和隐私保护性。实验结果证明，所提出的这种方法在多种数据集中实现了模型性能的改进。 

---
# HRET: A Self-Evolving LLM Evaluation Toolkit for Korean 

**Title (ZH)**: HRET：一种自我进化的韩语大型语言模型评价工具-kit 

**Authors**: Hanwool Lee, Soo Yong Kim, Dasol Choi, SangWon Baek, Seunghyeok Hong, Ilgyun Jeong, Inseon Hwang, Naeun Lee, Guijin Son  

**Link**: [PDF](https://arxiv.org/pdf/2503.22968)  

**Abstract**: Recent advancements in Korean large language models (LLMs) have spurred numerous benchmarks and evaluation methodologies, yet the lack of a standardized evaluation framework has led to inconsistent results and limited comparability. To address this, we introduce HRET Haerae Evaluation Toolkit, an open-source, self-evolving evaluation framework tailored specifically for Korean LLMs. HRET unifies diverse evaluation methods, including logit-based scoring, exact-match, language-inconsistency penalization, and LLM-as-a-Judge assessments. Its modular, registry-based architecture integrates major benchmarks (HAE-RAE Bench, KMMLU, KUDGE, HRM8K) and multiple inference backends (vLLM, HuggingFace, OpenAI-compatible endpoints). With automated pipelines for continuous evolution, HRET provides a robust foundation for reproducible, fair, and transparent Korean NLP research. 

**Abstract (ZH)**: Recent advancements in Korean大型语言模型（LLMs）尽管激发了众多基准测试和评价方法，但缺乏标准化的评价框架仍导致了结果不一致和可比性有限。为解决这一问题，我们引入了HRET Haerae评价工具包，这是一个专为韩语LLMs设计的开源、自我进化的评价框架。HRET统一了多种评价方法，包括logit评分、精确匹配、语言一致性惩罚和LLM作为裁判的评估。其模块化、注册表为基础的架构集成了主要基准测试（HAE-RAE Bench、KMMLU、KUDGE、HRM8K）和多个推理后端（vLLM、HuggingFace、OpenAI兼容端点）。通过自动化管道进行持续进化，HRET为可重复、公平和透明的韩语NLP研究提供了坚实的基础。 

---
# Student-Powered Digital Scholarship CoLab Project in the HKUST Library: Develop a Chinese Named-Entity Recognition (NER) Tool within One Semester from the Ground Up 

**Title (ZH)**: HKUST图书馆基于学生的数字学术CoLab项目：在一个月学期内自底向上开发一种中文命名实体识别（NER）工具 

**Authors**: Sherry S.L. Yip, Berry L. Han, Holly H.Y. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2503.22967)  

**Abstract**: Starting in February 2024, the HKUST Library further extended the scope of AI literacy to AI utilization, which focuses on fostering student involvement in utilizing state-of-the-art technologies in the projects that initiated by the Library, named "Digital Scholarship (DS) CoLab". A key focus of the DS CoLab scheme has been on cultivating talents and enabling students to utilize advanced technologies in practical context. It aims to reinforce the library's role as a catalyst and hub for fostering multidisciplinary collaboration and cultivate the "can do spirit" among university members. The Library offers 1-2 projects per year for students to engage with advanced technologies in practical contexts while supporting the Library in tackling challenges and streamlining operational tasks. The tool that introduced in this paper was mainly developed by two of the authors, Sherry Yip Sau Lai and Berry Han Liuruo, as part-time student helpers under one of our DS CoLab scheme in the 2024 Spring Semester (February to May 2024). This paper details the complete journey from ideation to implementation of developing a Chinese Named-Entity Recognition (NER) Tool from the group up within one semester, from the initial research and planning stages to execution and come up a viable product. The collaborative spirit fostered by this project, with students playing a central role, exemplifies the power and potential of innovative educational models that prioritize hands-on learning with student involvement. 

**Abstract (ZH)**: 从2024年2月起，港科大图书馆进一步将AI素养扩展至AI应用，专注于培养学生在图书馆发起的“数字学术（DS）合作实验室（CoLab）”项目中利用前沿技术。DS CoLab方案的核心重点在于培养人才，使学生能够在实际情境中利用先进技术。其目标是强化图书馆作为跨学科合作催化剂和中心的作用，并培养大学成员的“敢于尝试”的精神。图书馆每年提供1-2个项目，让学生在实际情境中接触前沿技术，同时支持图书馆应对挑战和优化运营任务。本文介绍的工具主要由Sherry Yip Sau Lai和Berry Han Liuruo两位作者在2024年春学期（2月至5月）DS CoLab方案的兼职学生助手身份下开发。本文详细描述了该团队在一个学期中从概念构思到实施开发一款中文命名实体识别（NER）工具的全过程，从初步研究和规划阶段到执行，最终形成一个可行的产品。该项目所培养的合作精神，体现了以学生参与为导向的创新型教育模式的力量和潜力。 

---
# Late Breaking Results: Breaking Symmetry- Unconventional Placement of Analog Circuits using Multi-Level Multi-Agent Reinforcement Learning 

**Title (ZH)**: Late Breaking Results: 突破对称性—使用多层级多代理强化学习的非常规模拟电路布局 

**Authors**: Supriyo Maji, Linran Zhao, Souradip Poddar, David Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.22958)  

**Abstract**: Layout-dependent effects (LDEs) significantly impact analog circuit performance. Traditionally, designers have relied on symmetric placement of circuit components to mitigate variations caused by LDEs. However, due to non-linear nature of these effects, conventional methods often fall short. We propose an objective-driven, multi-level, multi-agent Q-learning framework to explore unconventional design space of analog layout, opening new avenues for optimizing analog circuit performance. Our approach achieves better variation performance than the state-of-the-art layout techniques. Notably, this is the first application of multi-agent RL in analog layout automation. The proposed approach is compared with non-ML approach based on simulated annealing. 

**Abstract (ZH)**: 基于布局依赖效应的目标驱动多层次多智能体Q学习框架及其在模拟电路性能优化中的应用 

---
# Can LLMs Support Medical Knowledge Imputation? An Evaluation-Based Perspective 

**Title (ZH)**: LLMs在医疗知识补全中提供支持的能力：基于评估的角度 

**Authors**: Xinyu Yao, Aditya Sannabhadti, Holly Wiberg, Karmel S. Shehadeh, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2503.22954)  

**Abstract**: Medical knowledge graphs (KGs) are essential for clinical decision support and biomedical research, yet they often exhibit incompleteness due to knowledge gaps and structural limitations in medical coding systems. This issue is particularly evident in treatment mapping, where coding systems such as ICD, Mondo, and ATC lack comprehensive coverage, resulting in missing or inconsistent associations between diseases and their potential treatments. To address this issue, we have explored the use of Large Language Models (LLMs) for imputing missing treatment relationships. Although LLMs offer promising capabilities in knowledge augmentation, their application in medical knowledge imputation presents significant risks, including factual inaccuracies, hallucinated associations, and instability between and within LLMs. In this study, we systematically evaluate LLM-driven treatment mapping, assessing its reliability through benchmark comparisons. Our findings highlight critical limitations, including inconsistencies with established clinical guidelines and potential risks to patient safety. This study serves as a cautionary guide for researchers and practitioners, underscoring the importance of critical evaluation and hybrid approaches when leveraging LLMs to enhance treatment mappings on medical knowledge graphs. 

**Abstract (ZH)**: 医学知识图谱（KGs）对于临床决策支持和生物医学研究至关重要，但由于医学编码系统的知识缺口和结构限制，它们常常表现出不完整性。在治疗映射方面，这种情况尤为明显，如ICD、Mondo和ATC等编码系统缺乏全面的覆盖范围，导致疾病与潜在治疗之间的关联存在缺失或不一致。为了解决这一问题，我们探索了使用大型语言模型（LLMs）来补充缺失的治疗关系。尽管LLMs在知识增补方面具有潜力，但将其应用于医学知识图谱填充却存在显著风险，包括事实不准确、虚构的关联以及LLMs之间和内部的不稳定。在本研究中，我们系统评估了LLMs驱动的治疗映射，并通过基准比较评估其可靠性。我们的研究结果强调了关键限制，包括与既有临床指南的一致性问题以及对患者安全的风险。本研究为研究人员和实践者提供了一种警戒指南，强调了在利用LLMs增强医学知识图谱中的治疗映射时进行批判性评估和混合方法的重要性。 

---
# SUV: Scalable Large Language Model Copyright Compliance with Regularized Selective Unlearning 

**Title (ZH)**: SUV: 可扩展的大语言模型版权合规与正则化选择性遗忘 

**Authors**: Tianyang Xu, Xiaoze Liu, Feijie Wu, Xiaoqian Wang, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.22948)  

**Abstract**: Large Language Models (LLMs) have transformed natural language processing by learning from massive datasets, yet this rapid progress has also drawn legal scrutiny, as the ability to unintentionally generate copyrighted content has already prompted several prominent lawsuits. In this work, we introduce SUV (Selective Unlearning for Verbatim data), a selective unlearning framework designed to prevent LLM from memorizing copyrighted content while preserving its overall utility. In detail, the proposed method constructs a dataset that captures instances of copyrighted infringement cases by the targeted LLM. With the dataset, we unlearn the content from the LLM by means of Direct Preference Optimization (DPO), which replaces the verbatim copyrighted content with plausible and coherent alternatives. Since DPO may hinder the LLM's performance in other unrelated tasks, we integrate gradient projection and Fisher information regularization to mitigate the degradation. We validate our approach using a large-scale dataset of 500 famous books (predominantly copyrighted works) and demonstrate that SUV significantly reduces verbatim memorization with negligible impact on the performance on unrelated tasks. Extensive experiments on both our dataset and public benchmarks confirm the scalability and efficacy of our approach, offering a promising solution for mitigating copyright risks in real-world LLM applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过学习大规模数据集已改变自然语言处理领域，但这种快速进展也引起了法律关注，因为无意生成受版权保护内容的能力已引发多起重要诉讼。在此项工作中，我们提出了一种选择性遗忘框架SUV（Selective Unlearning for Verbatim data），旨在防止LLM记忆受版权保护的内容，同时保持其整体效用。具体而言，所提出的方法构建了一个数据集，该数据集捕获目标LLM的版权侵权实例。借助该数据集，我们通过直接偏好优化（DPO）移除LLM中的内容，即将确切的受版权保护的内容替换为合理且连贯的替代内容。由于DPO可能阻碍LLM在其他无关任务上的性能，我们整合了梯度投影和费舍尔信息正则化以减轻性能退化。我们使用包含500部著名书籍（主要为受版权保护的作品）的大规模数据集验证了该方法，并展示了SUV显著减少了受版权保护内容的精确记忆，对无关任务的性能影响微乎其微。广泛的实验不仅证实了所提出方法的可扩展性和有效性，还提供了在实际应用中减轻LLM版权风险的有希望的解决方案。 

---
# DATAWEAVER: Authoring Data-Driven Narratives through the Integrated Composition of Visualization and Text 

**Title (ZH)**: 数据编织：通过可视化与文本集成创作数据驱动的故事 

**Authors**: Yu Fu, Dennis Bromley, Vidya Setlur  

**Link**: [PDF](https://arxiv.org/pdf/2503.22946)  

**Abstract**: Data-driven storytelling has gained prominence in journalism and other data reporting fields. However, the process of creating these stories remains challenging, often requiring the integration of effective visualizations with compelling narratives to form a cohesive, interactive presentation. To help streamline this process, we present an integrated authoring framework and system, DataWeaver, that supports both visualization-to-text and text-to-visualization composition. DataWeaver enables users to create data narratives anchored to data facts derived from "call-out" interactions, i.e., user-initiated highlights of visualization elements that prompt relevant narrative content. In addition to this "vis-to-text" composition, DataWeaver also supports a "text-initiated" approach, generating relevant interactive visualizations from existing narratives. Key findings from an evaluation with 13 participants highlighted the utility and usability of DataWeaver and the effectiveness of its integrated authoring framework. The evaluation also revealed opportunities to enhance the framework by refining filtering mechanisms and visualization recommendations and better support authoring creativity by introducing advanced customization options. 

**Abstract (ZH)**: 数据驱动的故事讲述在新闻报道和其他数据报告领域中逐渐凸显，然而这一过程仍然具有挑战性，通常需要有效可视化与引人入胜的叙述相结合以形成一个连贯的交互性展示。为了简化这一过程，我们介绍了一种集成的创作框架和系统——DataWeaver，它支持可视化到文本和文本到可视化的组成。DataWeaver 允许用户通过与可视化的“高亮”交互（即由用户触发的可视化元素突出显示）创建锚定于数据事实的数据叙述。除了“可视到文本”的组成方式，DataWeaver 还支持“文本触发”的方法，从现有叙述生成相关的交互式可视化。评价实验（涉及13名参与者）的发现强调了DataWeaver的实用性和易用性以及其集成创作框架的有效性。此外，该评价还揭示了通过完善过滤机制和可视化推荐以及引入更高级的自定义选项来增强框架以更好地支持创作能力的机会。 

---
# Adaptive Interactive Navigation of Quadruped Robots using Large Language Models 

**Title (ZH)**: 基于大型语言模型的四足机器人自适应交互导航 

**Authors**: Kangjie Zhou, Yao Mu, Haoyang Song, Yi Zeng, Pengying Wu, Han Gao, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22942)  

**Abstract**: Robotic navigation in complex environments remains a critical research challenge. Traditional navigation methods focus on optimal trajectory generation within free space, struggling in environments lacking viable paths to the goal, such as disaster zones or cluttered warehouses. To address this gap, we propose an adaptive interactive navigation approach that proactively interacts with environments to create feasible paths to reach originally unavailable goals. Specifically, we present a primitive tree for task planning with large language models (LLMs), facilitating effective reasoning to determine interaction objects and sequences. To ensure robust subtask execution, we adopt reinforcement learning to pre-train a comprehensive skill library containing versatile locomotion and interaction behaviors for motion planning. Furthermore, we introduce an adaptive replanning method featuring two LLM-based modules: an advisor serving as a flexible replanning trigger and an arborist for autonomous plan adjustment. Integrated with the tree structure, the replanning mechanism allows for convenient node addition and pruning, enabling rapid plan modification in unknown environments. Comprehensive simulations and experiments have demonstrated our method's effectiveness and adaptivity in diverse scenarios. The supplementary video is available at page: this https URL. 

**Abstract (ZH)**: 复杂环境中的机器人导航仍然是一个关键的研究挑战。传统的导航方法专注于自由空间内的最优轨迹生成，而在缺乏到达目标的有效路径的环境中（如灾难现场或杂乱的仓库）表现不佳。为了解决这一问题，我们提出了一种主动适应的交互导航方法，能够在与环境互动的过程中创建通往原本不可达目标的可行路径。具体来说，我们利用大规模语言模型（LLMs）构建任务规划的原语树，促进有效的推理以确定交互对象和顺序。为了确保子任务执行的鲁棒性，我们采用强化学习预先训练了一个包含多种行动和交互行为的技能库，用于运动规划。此外，我们引入了一种适应性的重新规划方法，包含两个基于大规模语言模型的模块：顾问作为灵活的重新规划触发器，树匠负责自主调整计划。与树结构集成，重新规划机制允许方便地添加和修剪节点，使得在未知环境中能够快速修改计划。全面的仿真和实验表明，该方法在多种场景下具有有效性和适应性。补充视频见：this https URL。 

---
# FairSAM: Fair Classification on Corrupted Data Through Sharpness-Aware Minimization 

**Title (ZH)**: 公平SAM：通过敏锐度感知最小化在受污染数据上的公平分类 

**Authors**: Yucong Dai, Jie Ji, Xiaolong Ma, Yongkai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22934)  

**Abstract**: Image classification models trained on clean data often suffer from significant performance degradation when exposed to testing corrupted data, such as images with impulse noise, Gaussian noise, or environmental noise. This degradation not only impacts overall performance but also disproportionately affects various demographic subgroups, raising critical algorithmic bias concerns. Although robust learning algorithms like Sharpness-Aware Minimization (SAM) have shown promise in improving overall model robustness and generalization, they fall short in addressing the biased performance degradation across demographic subgroups. Existing fairness-aware machine learning methods - such as fairness constraints and reweighing strategies - aim to reduce performance disparities but hardly maintain robust and equitable accuracy across demographic subgroups when faced with data corruption. This reveals an inherent tension between robustness and fairness when dealing with corrupted data. To address these challenges, we introduce one novel metric specifically designed to assess performance degradation across subgroups under data corruption. Additionally, we propose \textbf{FairSAM}, a new framework that integrates \underline{Fair}ness-oriented strategies into \underline{SAM} to deliver equalized performance across demographic groups under corrupted conditions. Our experiments on multiple real-world datasets and various predictive tasks show that FairSAM successfully reconciles robustness and fairness, offering a structured solution for equitable and resilient image classification in the presence of data corruption. 

**Abstract (ZH)**: 基于去噪数据训练的图像分类模型在遇到冲击噪声、高斯噪声或环境噪声等测试破坏数据时，往往会遭受显著的性能下降，这种下降不仅影响整体性能，还对不同的人口亚组不公平，引起重要的算法偏见问题。虽然像Sharpness-Aware Minimization (SAM)这样的鲁棒学习算法在提高模型整体鲁棒性和泛化性方面显示出前景，但在解决不同人口亚组的偏差性能下降方面仍存在不足。现有的公平感知机器学习方法——如公平约束和重权重策略——旨在减少性能差距，但在面对数据破坏时，难以在不同的人口亚组中保持鲁棒和公平的准确性。这揭示了在处理破坏数据时鲁棒性和公平性之间固有的紧张关系。为应对这些挑战，我们引入了一种新型度量标准，专门用于评估数据破坏下不同亚组的性能下降。此外，我们提出了FairSAM，这是一种新的框架，将公平导向策略整合到SAM中，以便在数据破坏条件下实现不同人口群体的公平性能。我们在多个真实世界数据集和各种预测任务上的实验表明，FairSAM成功地平衡了鲁棒性和公平性，提供了一种在数据破坏情况下实现公平和稳健图像分类的结构化解决方案。 

---
# Predictive Traffic Rule Compliance using Reinforcement Learning 

**Title (ZH)**: 基于强化学习的预测性交通规则遵守性研究 

**Authors**: Yanliang Huang, Sebastian Mair, Zhuoqi Zeng, Amr Alanwar, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2503.22925)  

**Abstract**: Autonomous vehicle path planning has reached a stage where safety and regulatory compliance are crucial. This paper presents a new approach that integrates a motion planner with a deep reinforcement learning model to predict potential traffic rule violations. In this setup, the predictions of the critic directly affect the cost function of the motion planner, guiding the choices of the trajectory. We incorporate key interstate rules from the German Road Traffic Regulation into a rule book and use a graph-based state representation to handle complex traffic information. Our main innovation is replacing the standard actor network in an actor-critic setup with a motion planning module, which ensures both predictable trajectory generation and prevention of long-term rule violations. Experiments on an open German highway dataset show that the model can predict and prevent traffic rule violations beyond the planning horizon, significantly increasing safety in challenging traffic conditions. 

**Abstract (ZH)**: 自主驾驶车辆路径规划中的运动规划与深度强化学习模型集成方法：关键交通规则预测与合规性保障 

---
# Enhancing DeepLabV3+ to Fuse Aerial and Satellite Images for Semantic Segmentation 

**Title (ZH)**: 增强DeepLabV3+以融合航空和卫星图像进行语义分割 

**Authors**: Anas Berka, Mohamed El Hajji, Raphael Canals, Youssef Es-saady, Adel Hafiane  

**Link**: [PDF](https://arxiv.org/pdf/2503.22909)  

**Abstract**: Aerial and satellite imagery are inherently complementary remote sensing sources, offering high-resolution detail alongside expansive spatial coverage. However, the use of these sources for land cover segmentation introduces several challenges, prompting the development of a variety of segmentation methods. Among these approaches, the DeepLabV3+ architecture is considered as a promising approach in the field of single-source image segmentation. However, despite its reliable results for segmentation, there is still a need to increase its robustness and improve its performance. This is particularly crucial for multimodal image segmentation, where the fusion of diverse types of information is essential.
An interesting approach involves enhancing this architectural framework through the integration of novel components and the modification of certain internal processes.
In this paper, we enhance the DeepLabV3+ architecture by introducing a new transposed conventional layers block for upsampling a second entry to fuse it with high level features. This block is designed to amplify and integrate information from satellite images, thereby enriching the segmentation process through fusion with aerial images.
For experiments, we used the this http URL (Land Cover from Aerial Imagery) dataset for aerial images, alongside the corresponding dataset sourced from Sentinel 2 data.
Through the fusion of both sources, the mean Intersection over Union (mIoU) achieved a total mIoU of 84.91% without data augmentation. 

**Abstract (ZH)**: 基于航天航空影像的DeepLabV3+架构改进及其在多模态影像分割中的应用 

---
# Pairwise Matching of Intermediate Representations for Fine-grained Explainability 

**Title (ZH)**: 中间表示的成对匹配以实现细粒度解释性 

**Authors**: Lauren Shrack, Timm Haucke, Antoine Salaün, Arjun Subramonian, Sara Beery  

**Link**: [PDF](https://arxiv.org/pdf/2503.22881)  

**Abstract**: The differences between images belonging to fine-grained categories are often subtle and highly localized, and existing explainability techniques for deep learning models are often too diffuse to provide useful and interpretable explanations. We propose a new explainability method (PAIR-X) that leverages both intermediate model activations and backpropagated relevance scores to generate fine-grained, highly-localized pairwise visual explanations. We use animal and building re-identification (re-ID) as a primary case study of our method, and we demonstrate qualitatively improved results over a diverse set of explainability baselines on 35 public re-ID datasets. In interviews, animal re-ID experts were in unanimous agreement that PAIR-X was an improvement over existing baselines for deep model explainability, and suggested that its visualizations would be directly applicable to their work. We also propose a novel quantitative evaluation metric for our method, and demonstrate that PAIR-X visualizations appear more plausible for correct image matches than incorrect ones even when the model similarity score for the pairs is the same. By improving interpretability, PAIR-X enables humans to better distinguish correct and incorrect matches. Our code is available at: this https URL 

**Abstract (ZH)**: 细粒度类别图像之间的差异往往微妙且高度局部化，现有的深度学习模型解释技术往往过于模糊，无法提供有用和可解释的解释。我们提出了一种新的解释方法（PAIR-X），该方法结合了中间模型激活和反向传播的相关得分，以生成细粒度和高度局部化的成对视觉解释。我们将动物和建筑再识别（re-ID）作为我们方法的主要案例研究，并在35个公开的re-ID数据集上展示了比多种解释基线方法有质的改进。在访谈中，动物re-ID专家一致认为PAIR-X比现有基线更适合深度模型解释，并建议其可视化可以直接应用于他们的工作中。我们还提出了对我们方法的一种新的定量评价指标，并且证明即使模型对成对图像相似性评分相同，PAIR-X的可视化对于正确图像匹配看起来更可信。通过提高可解释性，PAIR-X使人类能够更好地区分正确的和错误的匹配。我们的代码可在以下链接获取：this https URL 

---
# Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models 

**Title (ZH)**: Quamba2：选择状态空间模型的鲁棒且可扩展的后训练量化框架 

**Authors**: Hung-Yueh Chiang, Chi-Chih Chang, Natalia Frumkin, Kai-Chiang Wu, Mohamed S. Abdelfattah, Diana Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22879)  

**Abstract**: State Space Models (SSMs) are emerging as a compelling alternative to Transformers because of their consistent memory usage and high performance. Despite this, scaling up SSMs on cloud services or limited-resource devices is challenging due to their storage requirements and computational power. To overcome this, quantizing SSMs with low bit-width data formats can reduce model size and benefit from hardware acceleration. As SSMs are prone to quantization-induced errors, recent efforts have focused on optimizing a particular model or bit-width for efficiency without sacrificing performance. However, distinct bit-width configurations are essential for different scenarios, like W4A8 for boosting large-batch decoding speed, and W4A16 for enhancing generation speed in short prompt applications for a single user. To this end, we present Quamba2, compatible with W8A8, W4A8, and W4A16 for both Mamba1 and Mamba2 backbones, addressing the growing demand for SSM deployment on various platforms. Based on the channel order preserving and activation persistence of SSMs, we propose an offline approach to quantize inputs of a linear recurrence in 8-bit by sorting and clustering for input $x$, combined with a per-state-group quantization for input-dependent parameters $B$ and $C$. To ensure compute-invariance in the SSM output, we rearrange weights offline according to the clustering sequence. The experiments show that Quamba2-8B outperforms several state-of-the-art SSM quantization methods and delivers 1.3$\times$ and 3$\times$ speed-ups in the pre-filling and generation stages, respectively, while offering 4$\times$ memory reduction with only a $1.6\%$ average accuracy drop. The evaluation on MMLU shows the generalizability and robustness of our framework. The code and quantized models will be released at: this https URL. 

**Abstract (ZH)**: State Space Models (SSMs)作为Transformer的有吸引力的替代方案正逐渐兴起，得益于其一致的内存使用和高性能。尽管如此，由于存储需求和计算能力限制，将SSMs扩展到云服务或有限资源设备仍然充满挑战。为了克服这一难题，使用低位宽数据格式对SSMs进行量化可以减小模型大小并受益于硬件加速。由于SSMs容易受到量化引起的误差影响，最近的努力集中在优化特定模型或位宽，以提高效率而不牺牲性能。然而，不同的位宽配置对于不同的场景至关重要，比如W4A8用于提升大批次解码速度，W4A16则用于单一用户短期提示应用中生成速度的提升。为此，我们提出了Quamba2，支持W8A8、W4A8和W4A16，适用于Mamba1和Mamba2的骨干网络，以满足各种平台上SSM部署日益增长的需求。基于SSMs的通道顺序保和服务于状态的激活保持特性，我们提出了一种离线方法，通过排序和聚类对输入x进行8位量化，并结合状态组间输入依赖参数B和C的量化。为了确保SSM输出的计算不变性，我们根据聚类序列离线重排权重。实验结果显示，Quamba2-8B在预填充和生成阶段分别提供了1.3倍和3倍的速度提升，同时实现了4倍的内存减少，并且平均精度下降仅为1.6%。我们的框架在MMLU上的评估展示了其普遍适用性和鲁棒性。代码和量化模型将发布在：this https URL。 

---
# Understanding Inequality of LLM Fact-Checking over Geographic Regions with Agent and Retrieval models 

**Title (ZH)**: 地理区域之间大规模语言模型事实检查不平等性研究：基于代理和检索模型方法 

**Authors**: Bruno Coelho, Shujaat Mirza, Yuyuan Cui, Christina Pöpper, Damon McCoy  

**Link**: [PDF](https://arxiv.org/pdf/2503.22877)  

**Abstract**: Fact-checking is a potentially useful application of Large Language Models (LLMs) to combat the growing dissemination of disinformation. However, the performance of LLMs varies across geographic regions. In this paper, we evaluate the factual accuracy of open and private models across a diverse set of regions and scenarios.
Using a dataset containing 600 fact-checked statements balanced across six global regions we examine three experimental setups of fact-checking a statement: (1) when just the statement is available, (2) when an LLM-based agent with Wikipedia access is utilized, and (3) as a best case scenario when a Retrieval-Augmented Generation (RAG) system provided with the official fact check is employed. Our findings reveal that regardless of the scenario and LLM used, including GPT-4, Claude Sonnet, and LLaMA, statements from the Global North perform substantially better than those from the Global South. Furthermore, this gap is broadened for the more realistic case of a Wikipedia agent-based system, highlighting that overly general knowledge bases have a limited ability to address region-specific nuances. These results underscore the urgent need for better dataset balancing and robust retrieval strategies to enhance LLM fact-checking capabilities, particularly in geographically diverse contexts. 

**Abstract (ZH)**: 大型语言模型在事实核查中的应用：地理区域差异及其对策 

---
# Teaching LLMs Music Theory with In-Context Learning and Chain-of-Thought Prompting: Pedagogical Strategies for Machines 

**Title (ZH)**: 用上下文学习和链式思考提示教授大语言模型音乐理论：面向机器的教学策略 

**Authors**: Liam Pond, Ichiro Fujinaga  

**Link**: [PDF](https://arxiv.org/pdf/2503.22853)  

**Abstract**: This study evaluates the baseline capabilities of Large Language Models (LLMs) like ChatGPT, Claude, and Gemini to learn concepts in music theory through in-context learning and chain-of-thought prompting. Using carefully designed prompts (in-context learning) and step-by-step worked examples (chain-of-thought prompting), we explore how LLMs can be taught increasingly complex material and how pedagogical strategies for human learners translate to educating machines. Performance is evaluated using questions from an official Canadian Royal Conservatory of Music (RCM) Level 6 examination, which covers a comprehensive range of topics, including interval and chord identification, key detection, cadence classification, and metrical analysis. Additionally, we evaluate the suitability of various music encoding formats for these tasks (ABC, Humdrum, MEI, MusicXML). All experiments were run both with and without contextual prompts. Results indicate that without context, ChatGPT with MEI performs the best at 52%, while with context, Claude with MEI performs the best at 75%. Future work will further refine prompts and expand to cover more advanced music theory concepts. This research contributes to the broader understanding of teaching LLMs and has applications for educators, students, and developers of AI music tools alike. 

**Abstract (ZH)**: 本研究评估了像ChatGPT、Claude和Gemini这样的大型语言模型通过在上下文学习和链式思考提示方法学习音乐理论基本能力。利用精心设计的提示（在上下文学习）和逐步示例（链式思考提示），我们探究了如何逐渐教授LLMs复杂材料，以及人类学习者的教学策略如何应用于机器教育。性能评估使用加拿大皇家音乐学院（RCM）级6考试中的官方问题，涵盖了包括音程和和弦识别、调性检测、终止式分类和节奏分析在内的广泛主题。此外，我们还评估了各种音乐编码格式（ABC、Humdrum、MEI、MusicXML）在这些任务中的适用性。所有实验均在有无上下文提示的情况下进行。结果显示，在无上下文的情况下，使用MEI的ChatGPT表现最佳，得分为52%，而在有上下文的情况下，使用MEI的Claude表现最佳，得分为75%。未来的工作将进一步精炼提示，并扩展到涵盖更高级的音乐理论概念。本研究为更广泛理解教授LLMs提供了贡献，并对教育者、学生和AI音乐工具开发者都有实际应用价值。 

---
# RobuNFR: Evaluating the Robustness of Large Language Models on Non-Functional Requirements Aware Code Generation 

**Title (ZH)**: RobuNFR: 非功能需求 awareness 代码生成中的大型语言模型稳健性评估 

**Authors**: Feng Lin, Dong Jae Kim, Zhenhao Li, Jinqiu Yang, Tse-Husn, Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22851)  

**Abstract**: When using LLMs to address Non-Functional Requirements (NFRs), developers may behave differently (e.g., expressing the same NFR in different words). Robust LLMs should output consistent results across these variations; however, this aspect remains underexplored. We propose RobuNFR for evaluating the robustness of LLMs in NFR-aware code generation across four NFR dimensions: design, readability, reliability, and performance, using three methodologies: prompt variation, regression testing, and diverse workflows. Our experiments show that RobuNFR reveals robustness issues in the tested LLMs when considering NFRs in code generation. Specifically, under prompt variation, including NFRs leads to a decrease in Pass@1 by up to 39 percent and an increase in the standard deviation from 0.48 to 2.48 compared to the baseline without NFRs (i.e., Function-Only). While incorporating NFRs generally improves overall NFR metrics, it also results in higher prompt sensitivity. In regression settings, some LLMs exhibit differences across versions, with improvements in one aspect (e.g., reduced code smells) often accompanied by regressions in another (e.g., decreased correctness), revealing inconsistencies that challenge their robustness. When varying workflows, the tested LLMs show significantly different NFR-aware code generation capabilities between two workflows: (1) integrating NFRs and functional requirements into the initial prompt and (2) enhancing Function-Only-generated code with the same NFR. 

**Abstract (ZH)**: 使用RobuNFR评估LLM在代码生成中对非功能性需求（NFRs）感知的鲁棒性：在设计、可读性、可靠性和性能四个维度上的评估，采用三种方法：提示变异、回归测试和多样化的工作流 

---
# Nonhuman Primate Brain Tissue Segmentation Using a Transfer Learning Approach 

**Title (ZH)**: 非人灵长类大脑组织分割的迁移学习方法 

**Authors**: Zhen Lin, Hongyu Yuan, Richard Barcus, Qing Lyu, Sucheta Chakravarty, Megan E. Lipford, Carol A. Shively, Suzanne Craft, Mohammad Kawas, Jeongchul Kim, Christopher T. Whitlow  

**Link**: [PDF](https://arxiv.org/pdf/2503.22829)  

**Abstract**: Non-human primates (NHPs) serve as critical models for understanding human brain function and neurological disorders due to their close evolutionary relationship with humans. Accurate brain tissue segmentation in NHPs is critical for understanding neurological disorders, but challenging due to the scarcity of annotated NHP brain MRI datasets, the small size of the NHP brain, the limited resolution of available imaging data and the anatomical differences between human and NHP brains. To address these challenges, we propose a novel approach utilizing STU-Net with transfer learning to leverage knowledge transferred from human brain MRI data to enhance segmen-tation accuracy in the NHP brain MRI, particularly when training data is this http URL combination of STU-Net and transfer learning effectively delineates complex tissue boundaries and captures fine anatomical details specific to NHP brains. Notably, our method demonstrated improvement in segmenting small subcortical structures such as putamen and thalamus that are challenging to resolve with limited spatial resolution and tissue contrast, and achieved DSC of over 0.88, IoU over 0.8 and HD95 under 7. This study introduces a robust method for multi-class brain tissue segmentation in NHPs, potentially accelerating research in evolutionary neuroscience and preclinical studies of neurological disorders relevant to human health. 

**Abstract (ZH)**: 非人灵长类动物（NHPs）是研究人类大脑功能和神经疾病的关键模型，由于它们与人类的进化关系密切。非人灵长类动物脑组织分割对于理解神经疾病至关重要，但由于注释的NHP脑MRI数据集稀缺、NHP脑容量小、可用成像数据的分辨率有限以及人类和非人灵长类动物大脑的解剖差异，这使得准确的脑组织分割极具挑战性。为应对这些挑战，我们提出了一种利用STU-Net结合迁移学习的新型方法，以利用来自人类脑MRI数据的知识来增强NHP脑MRI中的分割准确度，尤其是在训练数据稀缺的情况下。这种STU-Net与迁移学习的结合有效地勾勒出复杂的组织边界，并捕获到特定于NHP大脑的精细解剖细节。我们的方法在分割丘脑和壳核等小的基底节结构方面取得了改进，这些结构由于空间分辨率和组织对比度有限而难以解析，实现了DSC超过0.88，IoU超过0.8，HD95小于7。本研究表明了一种稳健的非人灵长类动物多类脑组织分割方法，有可能加速进化神经科学和与人类健康相关的神经疾病预临床研究。 

---
# Data-driven worker activity recognition and picking efficiency estimation in manual strawberry harvesting 

**Title (ZH)**: 基于数据驱动的草莓人工采摘工人的活动识别与采摘效率估计 

**Authors**: Uddhav Bhattarai, Rajkishan Arikapudi, Steven A. Fennimore, Frank N Martin, Stavros G. Vougioukas  

**Link**: [PDF](https://arxiv.org/pdf/2503.22809)  

**Abstract**: Manual fruit harvesting is common in agriculture, but the amount of time that pickers spend on nonproductive activities can make it very inefficient. Accurately identifying picking vs. non-picking activity is crucial for estimating picker efficiency and optimizing labor management and the harvest process. In this study, a practical system was developed to calculate the efficiency of pickers in commercial strawberry harvesting. Instrumented picking carts were used to record in real-time the harvested fruit weight, geo-location, and cart movement. A fleet of these carts was deployed during the commercial strawberry harvest season in Santa Maria, CA. The collected data was then used to train a CNN-LSTM-based deep neural network to classify a picker's activity into ``Pick" and ``NoPick" classes. Experimental evaluations showed that the CNN-LSTM model showed promising activity recognition performance with an F1 score accuracy of up to 0.974. The classification results were then used to compute two worker efficiency metrics: the percentage of time spent actively picking, and the time required to fill a tray. Analysis of the season-long harvest data showed that the pickers spent an average of 73.56% of their total harvest time actively picking strawberries, with an average tray fill time of 6.22 minutes. The mean accuracies of these metrics were 96.29% and 95.42%, respectively. When integrated on a commercial scale, the proposed technology could aid growers in automated worker activity monitoring and harvest optimization, ultimately helping to reduce non-productive time and enhance overall harvest efficiency. 

**Abstract (ZH)**: 基于CNN-LSTM的实用系统在商用草莓采摘中计算采摘工的效率 

---
# DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers 

**Title (ZH)**: DiTFastAttnV2: 头向注意力压缩的多模态扩散变换器 

**Authors**: Hanling Zhang, Rundong Su, Zhihang Yuan, Pengtao Chen, Mingzhu Shen Yibo Fan, Shengen Yan, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22796)  

**Abstract**: Text-to-image generation models, especially Multimodal Diffusion Transformers (MMDiT), have shown remarkable progress in generating high-quality images. However, these models often face significant computational bottlenecks, particularly in attention mechanisms, which hinder their scalability and efficiency. In this paper, we introduce DiTFastAttnV2, a post-training compression method designed to accelerate attention in MMDiT. Through an in-depth analysis of MMDiT's attention patterns, we identify key differences from prior DiT-based methods and propose head-wise arrow attention and caching mechanisms to dynamically adjust attention heads, effectively bridging this gap. We also design an Efficient Fused Kernel for further acceleration. By leveraging local metric methods and optimization techniques, our approach significantly reduces the search time for optimal compression schemes to just minutes while maintaining generation quality. Furthermore, with the customized kernel, DiTFastAttnV2 achieves a 68% reduction in attention FLOPs and 1.5x end-to-end speedup on 2K image generation without compromising visual fidelity. 

**Abstract (ZH)**: Text-to-image生成模型，尤其是多模态扩散变换器（MMDiT），在生成高质量图像方面取得了显著进展。然而，这些模型在注意力机制方面常常面临严重的计算瓶颈，这妨碍了它们的可扩展性和效率。在本文中，我们介绍了DiTFastAttnV2，这是一种旨在加速MMDiT中注意力机制的后训练压缩方法。通过对MMDiT注意力模式的深入分析，我们识别出与此前基于DiT的方法的关键差异，并提出了头导向箭头注意力和缓存机制，以动态调整注意力头，有效地弥合了这一差距。此外，我们还设计了一种高效融合内核以进一步加速处理。通过利用局部度量方法和优化技术，我们的方法显著减少了寻找最优压缩方案的时间，只需几分钟即可完成，并且在保持生成质量的同时。借助定制的内核，DiTFastAttnV2在不牺牲视觉保真度的情况下，实现了注意力FLOPs的68%减少和端到端1.5倍的速度提升，在2K图像生成方面的表现尤为显著。 

---
# Patronus: Bringing Transparency to Diffusion Models with Prototypes 

**Title (ZH)**: Patronus: 通过原型提升扩散模型的透明度 

**Authors**: Nina Weng, Aasa Feragen, Siavash Bigdeli  

**Link**: [PDF](https://arxiv.org/pdf/2503.22782)  

**Abstract**: Diffusion-based generative models, such as Denoising Diffusion Probabilistic Models (DDPMs), have achieved remarkable success in image generation, but their step-by-step denoising process remains opaque, leaving critical aspects of the generation mechanism unexplained. To address this, we introduce \emph{Patronus}, an interpretable diffusion model inspired by ProtoPNet. Patronus integrates a prototypical network into DDPMs, enabling the extraction of prototypes and conditioning of the generation process on their prototype activation vector. This design enhances interpretability by showing the learned prototypes and how they influence the generation process. Additionally, the model supports downstream tasks like image manipulation, enabling more transparent and controlled modifications. Moreover, Patronus could reveal shortcut learning in the generation process by detecting unwanted correlations between learned prototypes. Notably, Patronus operates entirely without any annotations or text prompts. This work opens new avenues for understanding and controlling diffusion models through prototype-based interpretability. Our code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于扩散的生成模型，如去噪扩散概率模型（DDPMs），已经在图像生成方面取得了显著成功，但其逐步去噪过程仍然不够透明，留下了生成机制中的关键方面未解释的问题。为解决这一问题，我们提出了一种名为Patronus的可解释扩散模型，该模型受到ProtoPNet的启发。Patronus将原型网络整合到DDPMs中，能够提取原型并根据其原型激活向量条件化生成过程。该设计通过展示学习到的原型及其对生成过程的影响来提升可解释性。此外，该模型支持图像操作等下游任务，使透明和可控的修改成为可能。同时，Patronus可以通过检测学习到的原型之间的不良相关性揭示生成过程中的捷径学习。值得注意的是，Patronus完全无需任何标注或文本提示。这项工作为通过基于原型的可解释性理解和控制扩散模型开辟了新的途径。我们的代码可在此处获得：\href{this https URL}{this https URL}。 

---
# Post-Incorporating Code Structural Knowledge into LLMs via In-Context Learning for Code Translation 

**Title (ZH)**: 通过上下文学习将代码结构知识后嵌入到大规模语言模型中以进行代码翻译 

**Authors**: Yali Du, Hui Sun, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.22776)  

**Abstract**: Code translation migrates codebases across programming languages. Recently, large language models (LLMs) have achieved significant advancements in software mining. However, handling the syntactic structure of source code remains a challenge. Classic syntax-aware methods depend on intricate model architectures and loss functions, rendering their integration into LLM training resource-intensive. This paper employs in-context learning (ICL), which directly integrates task exemplars into the input context, to post-incorporate code structural knowledge into pre-trained LLMs. We revisit exemplar selection in ICL from an information-theoretic perspective, proposing that list-wise selection based on information coverage is more precise and general objective than traditional methods based on combining similarity and diversity. To address the challenges of quantifying information coverage, we introduce a surrogate measure, Coverage of Abstract Syntax Tree (CAST). Furthermore, we formulate the NP-hard CAST maximization for exemplar selection and prove that it is a standard submodular maximization problem. Therefore, we propose a greedy algorithm for CAST submodular maximization, which theoretically guarantees a (1-1/e)-approximate solution in polynomial time complexity. Our method is the first training-free and model-agnostic approach to post-incorporate code structural knowledge into existing LLMs at test time. Experimental results show that our method significantly improves LLMs performance and reveals two meaningful insights: 1) Code structural knowledge can be effectively post-incorporated into pre-trained LLMs during inference, despite being overlooked during training; 2) Scaling up model size or training data does not lead to the emergence of code structural knowledge, underscoring the necessity of explicitly considering code syntactic structure. 

**Abstract (ZH)**: 代码翻译跨越编程语言。通过大型语言模型（LLMs）在软件挖掘方面的最新进展，代码迁移成为了可能。然而，处理源代码的语法结构仍然是一个挑战。经典的方法依赖于复杂的模型架构和损失函数，使得它们难以整合到LLM训练中。本文采用上下文相关学习（ICL），直接将任务示例集成到输入上下文中，以在预训练的LLMs中后嵌入代码结构知识。我们从信息论的角度回顾了ICL中的示例选择，提出基于信息覆盖的列表式选择比基于相似性和多样性的传统方法更加精确和通用。为了解决量化信息覆盖的挑战，我们引入了一个代理指标，抽象语法树（AST）覆盖度（CAST）。此外，我们为CAST的最大化形式化了NP-hard问题，并证明其是一个标准的次模函数最大化问题。因此，我们提出了一种贪婪算法以实现CAST的次模函数最大化，该算法在多项式时间复杂度内理论上保证了（1-1/e）近似解。我们的方法是第一个在测试时后嵌入代码结构知识的无需训练和模型无关的方法。实验结果表明，我们的方法显著提升了LLMs的性能，并揭示了两个重要的见解：1）在训练过程中被忽视的代码结构知识可以在推断过程中有效地后嵌入到预训练的LLMs中；2）增加模型大小或训练数据量不会导致代码结构知识的出现，强调了显式考虑代码语法结构的必要性。 

---
# GroundHog: Revolutionizing GLDAS Groundwater Storage Downscaling for Enhanced Recharge Estimation in Bangladesh 

**Title (ZH)**: GroundHog: 革命性地改进GLDAS地下水资源储存下标化以提升孟加拉国补给估算 

**Authors**: Saleh Sakib Ahmed, Rashed Uz Zzaman, Saifur Rahman Jony, Faizur Rahman Himel, Afroza Sharmin, A.H.M. Khalequr Rahman, M. Sohel Rahman, Sara Nowreen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22771)  

**Abstract**: Long-term groundwater level (GWL) measurement is vital for effective policymaking and recharge estimation using annual maxima and minima. However, current methods prioritize short-term predictions and lack multi-year applicability, limiting their utility. Moreover, sparse in-situ measurements lead to reliance on low-resolution satellite data like GLDAS as the ground truth for Machine Learning models, further constraining accuracy. To overcome these challenges, we first develop an ML model to mitigate data gaps, achieving $R^2$ scores of 0.855 and 0.963 for maximum and minimum GWL predictions, respectively. Subsequently, using these predictions and well observations as ground truth, we train an Upsampling Model that uses low-resolution (25 km) GLDAS data as input to produce high-resolution (2 km) GWLs, achieving an excellent $R^2$ score of 0.96. Our approach successfully upscales GLDAS data for 2003-2024, allowing high-resolution recharge estimations and revealing critical trends for proactive resource management. Our method allows upsampling of groundwater storage (GWS) from GLDAS to high-resolution GWLs for any points independently of officially curated piezometer data, making it a valuable tool for decision-making. 

**Abstract (ZH)**: 长期内部地下水位（GWL）测量对于有效政策制定和补给估算至关重要，使用年度极值和最小值。然而，当前方法侧重于短期预测，缺乏多年适用性，限制了其实用性。此外，稀疏的现场测量导致依赖低分辨率的卫星数据（如GLDAS）作为机器学习模型的基准，进一步限制了准确性。为克服这些挑战，我们首先开发了一个ML模型来缓解数据缺口，分别在最大和最小地下水位预测中实现了$R^2$分数0.855和0.963。随后，使用这些预测和井观测数据作为基准，我们训练了一个上采样模型，使用低分辨率（25 km）的GLDAS数据作为输入，生成高分辨率（2 km）的地下水位，实现了卓越的$R^2$分数0.96。我们的方法成功地将GLDAS数据扩展到2003-2024年，允许进行高分辨率的补给估算，并揭示了主动资源管理中关键的趋势。该方法允许独立于正式整理的抽水测量数据对地下水存储（GWS）进行上采样以生成高分辨率的地下水位，使其成为决策的关键工具。 

---
# MediTools -- Medical Education Powered by LLMs 

**Title (ZH)**: MediTools —— 由大规模语言模型驱动的医学教育 

**Authors**: Amr Alshatnawi, Remi Sampaleanu, David Liebovitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.22769)  

**Abstract**: Artificial Intelligence (AI) has been advancing rapidly and with the advent of large language models (LLMs) in late 2022, numerous opportunities have emerged for adopting this technology across various domains, including medicine. These innovations hold immense potential to revolutionize and modernize medical education. Our research project leverages large language models to enhance medical education and address workflow challenges through the development of MediTools - AI Medical Education. This prototype application focuses on developing interactive tools that simulate real-life clinical scenarios, provide access to medical literature, and keep users updated with the latest medical news. Our first tool is a dermatology case simulation tool that uses real patient images depicting various dermatological conditions and enables interaction with LLMs acting as virtual patients. This platform allows users to practice their diagnostic skills and enhance their clinical decision-making abilities. The application also features two additional tools: an AI-enhanced PubMed tool for engaging with LLMs to gain deeper insights into research papers, and a Google News tool that offers LLM generated summaries of articles for various medical specialties. A comprehensive survey has been conducted among medical professionals and students to gather initial feedback on the effectiveness and user satisfaction of MediTools, providing insights for further development and refinement of the application. This research demonstrates the potential of AI-driven tools in transforming and revolutionizing medical education, offering a scalable and interactive platform for continuous learning and skill development. 

**Abstract (ZH)**: 人工智能（AI）的发展日益迅速，特别是2022年底大型语言模型（LLMs）的出现，为医疗等多个领域 adopting这一技术带来了众多机会。这些创新拥有巨大潜力，能够革新和现代化医学教育。我们的研究项目利用大型语言模型增强医学教育，并通过开发MediTools - AI医学教育解决工作流程挑战。该原型应用程序侧重于开发模拟现实临床场景的交互工具，提供医学文献访问，并使用户能够获取最新医学新闻。我们的第一个工具是使用实际患者图像模拟各种皮肤病条件的皮肤病案例模拟工具，允许与作为虚拟患者的LLMs进行交互。该平台允许用户练习诊断技能并提高临床决策能力。该应用程序还包含两个附加工具：一个增强的PubMed工具，用于与LLMs交互以深入了解研究论文，以及一个Google新闻工具，提供由LLMs生成的文章摘要，适用于各种医学专科。我们对医学专业人员和学生进行了全面调查，收集了对MediTools的有效性和用户满意度的初步反馈，为应用程序的进一步开发和改进提供了见解。这项研究展示了AI驱动工具在革新和现代化医学教育方面的潜力，提供了一个可扩展且交互式的平台，支持持续学习和技能发展。 

---
# Boosting Large Language Models with Mask Fine-Tuning 

**Title (ZH)**: 使用掩码微调提升大型语言模型 

**Authors**: Mingyuan Zhang, Yue Bai, Huan Wang, Yizhou Wang, Qihua Dong, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22764)  

**Abstract**: The model is usually kept integral in the mainstream large language model (LLM) fine-tuning protocols. No works have questioned whether maintaining the integrity of the model is indispensable for performance. In this work, we introduce Mask Fine-Tuning (MFT), a brand-new LLM fine-tuning paradigm to show that properly breaking the integrity of the model can surprisingly lead to improved performance. Specifically, MFT learns a set of binary masks supervised by the typical LLM fine-tuning objective. Extensive experiments show that MFT gains a consistent performance boost across various domains and backbones (e.g., 1.95%/1.88% average gain in coding with LLaMA2-7B/3.1-8B). Detailed procedures are provided to study the proposed MFT from different hyperparameter perspectives for better insight. In particular, MFT naturally updates the current LLM training protocol by deploying it on a complete well-trained model. This study extends the functionality of mask learning from its conventional network pruning context for model compression to a more general scope. 

**Abstract (ZH)**: 模型在主流大语言模型（LLM）fine-tuning范式中通常保持整体性。没有任何研究质疑保持模型完整性的必要性对性能的影响。在这项工作中，我们引入了Mask Fine-Tuning (MFT)，这是一种全新的LLM fine-tuning范式，以展示适当破坏模型的完整性能意外地提高性能。具体来说，MFT通过监督典型的LLM fine-tuning目标学习一组二进制掩码。大量的实验表明，MFT在各种领域和骨干模型（例如，在使用LLaMA2-7B/3.1-8B时分别获得1.95%/1.88%的平均性能提升）中获得了持续的性能提升。我们提供了详细的过程，从不同的超参数视角研究提出的MFT，以获得更好的见解。特别是，MFT自然地更新了当前的LLM训练范式，将其应用于一个完整的训练良好的模型。这项研究将掩码学习的功能从其传统的网络剪枝上下文扩展到了更广泛的范围。 

---
# The Cost of Local and Global Fairness in Federated Learning 

**Title (ZH)**: 本地公平性和全局公平性在联邦学习中的成本 

**Authors**: Yuying Duan, Gelei Xu, Yiyu Shi, Michael Lemmon  

**Link**: [PDF](https://arxiv.org/pdf/2503.22762)  

**Abstract**: With the emerging application of Federated Learning (FL) in finance, hiring and healthcare, FL models are regulated to be fair, preventing disparities with respect to legally protected attributes such as race or gender. Two concepts of fairness are important in FL: global and local fairness. Global fairness addresses the disparity across the entire population and local fairness is concerned with the disparity within each client. Prior fair FL frameworks have improved either global or local fairness without considering both. Furthermore, while the majority of studies on fair FL focuses on binary settings, many real-world applications are multi-class problems. This paper proposes a framework that investigates the minimum accuracy lost for enforcing a specified level of global and local fairness in multi-class FL settings. Our framework leads to a simple post-processing algorithm that derives fair outcome predictors from the Bayesian optimal score functions. Experimental results show that our algorithm outperforms the current state of the art (SOTA) with regard to the accuracy-fairness tradoffs, computational and communication costs. Codes are available at: this https URL . 

**Abstract (ZH)**: 随着联邦学习（FL）在金融、招聘和医疗等领域的发展，FL模型需要被监管以确保公平性，防止与种族或性别等法律保护属性相关的不平等。FL中的公平性包含两个概念：全局公平和局部公平。全局公平关注整个群体的不平等现象，而局部公平关注每个客户端内的不平等。此前的公平联邦学习框架仅在全局或局部公平中有所改进，而未同时考虑两者。此外，尽管大多数关于公平联邦学习的研究集中在二分类问题上，但许多实际应用是多类问题。本文提出了一种框架，以探讨在多类联邦学习设置中强制执行指定水平的全局和局部公平所需的最小准确度损失。该框架导出了基于贝叶斯最优评分函数的公平结果预测器的简单后处理算法。实验结果表明，与当前最先进的算法相比，在准确度-公平性权衡、计算和通信成本方面，我们的算法表现出色。代码可在以下链接获取：this https URL。 

---
# Data Poisoning in Deep Learning: A Survey 

**Title (ZH)**: 深度学习中的数据投毒：一个综述 

**Authors**: Pinlong Zhao, Weiyao Zhu, Pengfei Jiao, Di Gao, Ou Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22759)  

**Abstract**: Deep learning has become a cornerstone of modern artificial intelligence, enabling transformative applications across a wide range of domains. As the core element of deep learning, the quality and security of training data critically influence model performance and reliability. However, during the training process, deep learning models face the significant threat of data poisoning, where attackers introduce maliciously manipulated training data to degrade model accuracy or lead to anomalous behavior. While existing surveys provide valuable insights into data poisoning, they generally adopt a broad perspective, encompassing both attacks and defenses, but lack a dedicated, in-depth analysis of poisoning attacks specifically in deep learning. In this survey, we bridge this gap by presenting a comprehensive and targeted review of data poisoning in deep learning. First, this survey categorizes data poisoning attacks across multiple perspectives, providing an in-depth analysis of their characteristics and underlying design princinples. Second, the discussion is extended to the emerging area of data poisoning in large language models(LLMs). Finally, we explore critical open challenges in the field and propose potential research directions to advance the field further. To support further exploration, an up-to-date repository of resources on data poisoning in deep learning is available at this https URL. 

**Abstract (ZH)**: 深度学习已成为现代人工智能的基石，能够在多个领域推动变革性应用。作为深度学习的核心要素，训练数据的质量和安全性对模型性能和可靠性至关重要。然而，在训练过程中，深度学习模型面临数据投毒这一重大威胁，攻击者通过引入恶意操纵的数据来降低模型准确性或导致异常行为。尽管现有综述提供了数据投毒有价值的见解，它们通常采用宽泛的角度，涵盖了攻击和防御两方面，但在针对深度学习中的数据投毒攻击进行专门的深入分析方面存在不足。在本文综述中，我们通过呈现数据投毒在深度学习中的全面而有针对性的回顾来填补这一空白。首先，本文综述从多个角度对数据投毒攻击进行分类，并深入分析其特点和设计原则。其次，讨论扩展到大型语言模型（LLMs）中的数据投毒新兴领域。最后，我们探讨了该领域中的关键开放挑战，并提出了潜在的研究方向以进一步推动该领域的发展。为了支持进一步的研究，最新的数据投毒资源库可参见此网址：[请补充具体网址]。 

---
# Towards an intelligent assessment system for evaluating the development of algorithmic thinking skills: An exploratory study in Swiss compulsory schools 

**Title (ZH)**: 面向算法思维能力评估的智能系统研究：一项瑞士义务教育学校探索性研究 

**Authors**: Giorgia Adorni  

**Link**: [PDF](https://arxiv.org/pdf/2503.22756)  

**Abstract**: The rapid digitalisation of contemporary society has profoundly impacted various facets of our lives, including healthcare, communication, business, and education. The ability to engage with new technologies and solve problems has become crucial, making CT skills, such as pattern recognition, decomposition, and algorithm design, essential competencies. In response, Switzerland is conducting research and initiatives to integrate CT into its educational system. This study aims to develop a comprehensive framework for large-scale assessment of CT skills, particularly focusing on AT, the ability to design algorithms. To achieve this, we first developed a competence model capturing the situated and developmental nature of CT, guiding the design of activities tailored to cognitive abilities, age, and context. This framework clarifies how activity characteristics influence CT development and how to assess these competencies. Additionally, we developed an activity for large-scale assessment of AT skills, offered in two variants: one based on non-digital artefacts (unplugged) and manual expert assessment, and the other based on digital artefacts (virtual) and automatic assessment. To provide a more comprehensive evaluation of students' competencies, we developed an IAS based on BNs with noisy gates, which offers real-time probabilistic assessment for each skill rather than a single overall score. The results indicate that the proposed instrument can measure AT competencies across different age groups and educational contexts in Switzerland, demonstrating its applicability for large-scale use. AT competencies exhibit a progressive development, with no overall gender differences, though variations are observed at the school level, significantly influenced by the artefact-based environment and its context, underscoring the importance of creating accessible and adaptable assessment tools. 

**Abstract (ZH)**: 当代社会的快速数字化对我们的生活诸多方面产生了深远影响，包括医疗保健、通信、商业和教育。掌握新技术的能力和解决问题的能力变得至关重要，使得模式识别、分解和算法设计等CT技能成为必不可少的技能。针对这一需求，瑞士开展了研究和项目，旨在将其CT技能融入教育体系。本研究旨在开发一个全面的框架，用于大型评估CT技能，特别是算法设计（AT）技能。为此，首先开发了一个技能模型，捕捉CT的环境性和发展阶段特性，指导针对认知能力、年龄和环境背景定制活动的设计。该框架阐明了活动特性如何影响CT的发展以及如何评估这些技能。此外，还开发了一种活动，用于评估AT技能的大型评估，提供两种变体：一种基于非数字化制品（非连接式）和手工专家评估，另一种基于数字化制品（虚拟）和自动评估。为了对学生的技能进行全面评估，我们基于贝叶斯网络（BN）和噪声门开发了一种即时概率评估方法（IAS），为每个技能提供即时概率评估，而非单一总体评分。结果表明，提议的工具可以跨不同年龄段和教育背景在瑞士测量AT技能，证明其在大规模使用中的适用性。AT技能表现出逐步发展，尽管总体上没有性别差异，但在学校层面存在显著差异，这些差异受到制品基础环境及其背景的显著影响，强调了创建可访问和适应性强评估工具的重要性。 

---
# Reasoning Under Threat: Symbolic and Neural Techniques for Cybersecurity Verification 

**Title (ZH)**: 在威胁下推理：网络安全验证的符号与神经技术 

**Authors**: Sarah Veronica  

**Link**: [PDF](https://arxiv.org/pdf/2503.22755)  

**Abstract**: Cybersecurity demands rigorous and scalable techniques to ensure system correctness, robustness, and resilience against evolving threats. Automated reasoning, encompassing formal logic, theorem proving, model checking, and symbolic analysis, provides a foundational framework for verifying security properties across diverse domains such as access control, protocol design, vulnerability detection, and adversarial modeling. This survey presents a comprehensive overview of the role of automated reasoning in cybersecurity, analyzing how logical systems, including temporal, deontic, and epistemic logics are employed to formalize and verify security guarantees. We examine SOTA tools and frameworks, explore integrations with AI for neural-symbolic reasoning, and highlight critical research gaps, particularly in scalability, compositionality, and multi-layered security modeling. The paper concludes with a set of well-grounded future research directions, aiming to foster the development of secure systems through formal, automated, and explainable reasoning techniques. 

**Abstract (ZH)**: 网络安全需要严格且可扩展的技术来确保系统的正确性、稳健性和对不断演变的威胁的韧性。自动推理，涵盖形式逻辑、定理证明、模型检测和符号分析，为验证跨接入控制、协议设计、漏洞检测和对抗建模等领域中的安全属性提供了基础框架。本文综述了自动推理在网络安全中的作用，分析了如何使用时间逻辑、义务逻辑和知识逻辑等逻辑系统来形式化和验证安全保证。我们研究了当今最先进的工具和框架，探讨了与人工智能的整合以实现神经符号推理，并指出了关键的研究空白，特别是可扩展性、组合性和多层次安全建模。论文最后提出了一系列坚实的研究方向，旨在通过正式、自动化和可解释的推理技术促进安全系统的开发。 

---
# Model Lake: a New Alternative for Machine Learning Models Management and Governance 

**Title (ZH)**: 模型湖：机器学习模型管理与治理的新选择 

**Authors**: Moncef Garouani, Franck Ravat, Nathalie Valles-Parlangeau  

**Link**: [PDF](https://arxiv.org/pdf/2503.22754)  

**Abstract**: The rise of artificial intelligence and data science across industries underscores the pressing need for effective management and governance of machine learning (ML) models. Traditional approaches to ML models management often involve disparate storage systems and lack standardized methodologies for versioning, audit, and re-use. Inspired by data lake concepts, this paper develops the concept of ML Model Lake as a centralized management framework for datasets, codes, and models within organizations environments. We provide an in-depth exploration of the Model Lake concept, delineating its architectural foundations, key components, operational benefits, and practical challenges. We discuss the transformative potential of adopting a Model Lake approach, such as enhanced model lifecycle management, discovery, audit, and reusability. Furthermore, we illustrate a real-world application of Model Lake and its transformative impact on data, code and model management practices. 

**Abstract (ZH)**: 人工智能和数据科学在各行业的兴起凸显了有效管理机器学习（ML）模型的紧迫需求。传统意义上的ML模型管理方法往往依赖于分散的存储系统，并缺乏版本控制、审计和重复利用的标准化方法。借鉴数据湖的概念，本文提出了ML模型湖的概念，作为一种组织内部集中管理数据集、代码和模型的框架。我们深入探讨了模型湖的概念，阐述其架构基础、关键组件、操作优势及实际挑战。我们讨论了采用模型湖方法的变革潜力，如增强的模型生命周期管理、发现、审计和重复利用能力。此外，我们阐述了一个实际应用模型湖的例子及其对数据、代码和模型管理实践的变革影响。 

---
# From Individual to Group: Developing a Context-Aware Multi-Criteria Group Recommender System 

**Title (ZH)**: 从个体到群体：发展一种基于上下文的多准则群体推荐系统 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2503.22752)  

**Abstract**: Group decision-making is becoming increasingly common in areas such as education, dining, travel, and finance, where collaborative choices must balance diverse individual preferences. While conventional recommender systems are effective in personalization, they fall short in group settings due to their inability to manage conflicting preferences, contextual factors, and multiple evaluation criteria. This study presents the development of a Context-Aware Multi-Criteria Group Recommender System (CA-MCGRS) designed to address these challenges by integrating contextual factors and multiple criteria to enhance recommendation accuracy. By leveraging a Multi-Head Attention mechanism, our model dynamically weighs the importance of different features. Experiments conducted on an educational dataset with varied ratings and contextual variables demonstrate that CA-MCGRS consistently outperforms other approaches across four scenarios. Our findings underscore the importance of incorporating context and multi-criteria evaluations to improve group recommendations, offering valuable insights for developing more effective group recommender systems. 

**Abstract (ZH)**: 基于上下文的多准则群体推荐系统（CA-MCGRS）：通过整合上下文因素和多准则提高推荐准确性 

---
# Advancing Spatiotemporal Prediction using Artificial Intelligence: Extending the Framework of Geographically and Temporally Weighted Neural Network (GTWNN) for Differing Geographical and Temporal Contexts 

**Title (ZH)**: 使用人工智能推进时空预测：扩展地理和时间加权神经网络（GTWNN）框架以适应不同的地理和时间背景 

**Authors**: Nicholas Robert Fisk, Matthew Ng Kok Ming, Zahratu Shabrina  

**Link**: [PDF](https://arxiv.org/pdf/2503.22751)  

**Abstract**: This paper aims at improving predictive crime models by extending the mathematical framework of Artificial Neural Networks (ANNs) tailored to general spatiotemporal problems and appropriately applying them. Recent advancements in the geospatial-temporal modelling field have focused on the inclusion of geographical weighting in their deep learning models to account for nonspatial stationarity, which is often apparent in spatial data. We formulate a novel semi-analytical approach to solving Geographically and Temporally Weighted Regression (GTWR), and applying it to London crime data. The results produce high-accuracy predictive evaluation scores that affirm the validity of the assumptions and approximations in the approach. This paper presents mathematical advances to the Geographically and Temporally Weighted Neural Network (GTWNN) framework, which offers a novel contribution to the field. Insights from past literature are harmoniously employed with the assumptions and approximations to generate three mathematical extensions to GTWNN's framework. Combinations of these extensions produce five novel ANNs, applied to the London and Detroit datasets. The results suggest that one of the extensions is redundant and is generally surpassed by another extension, which we term the history-dependent module. The remaining extensions form three novel ANN designs that pose potential GTWNN improvements. We evaluated the efficacy of various models in both the London and Detroit crime datasets, highlighting the importance of accounting for specific geographic and temporal characteristics when selecting modelling strategies to improve model suitability. In general, the proposed methods provide the foundations for a more context-aware, accurate, and robust ANN approach in spatio-temporal modelling. 

**Abstract (ZH)**: 本文旨在通过扩展适用于一般空-时问题的人工神经网络（ANN）的数学框架，并合理应用这些框架来改进预测犯罪模型。近年来，地理时空建模领域的发展集中在将地理加权纳入其深度学习模型中，以-account for 非空间平稳性，这在空间数据中经常可见。我们提出了一个新的半解析方法来求解地理加权和时间加权回归（GTWR），并将其应用于伦敦犯罪数据。结果产生了高精度的预测评估得分，证实了该方法假设和近似的有效性。本文提出了地理加权和时间加权神经网络（GTWNN）框架的数学进展，为该领域做出了新颖的贡献。我们和谐地运用了以往文献的见解与假设和近似，生成了GTWNN框架的三个数学扩展。这些扩展的组合产生了五个新的ANN，应用于伦敦和底特律数据集。结果表明，其中一个扩展是冗余的，并且普遍被另一个扩展——我们称之为历史依赖模块——所超越。剩余的扩展形成了三种新的ANN设计，可能改进GTWNN。我们在伦敦和底特律的犯罪数据集上评估了各种模型的有效性，强调了在选择建模策略以改善模型适应性时，考虑特定的地理和时间特征的重要性。总体而言，所提出的方法为时空建模中更具情境意识、更精确和更稳健的ANN方法奠定了基础。 

---
# Adaptive Clipping for Privacy-Preserving Few-Shot Learning: Enhancing Generalization with Limited Data 

**Title (ZH)**: 自适应裁剪以实现隐私保护的少样本学习：利用有限数据增强泛化能力 

**Authors**: Kanishka Ranaweera, Dinh C. Nguyen, Pubudu N. Pathirana, David Smith, Ming Ding, Thierry Rakotoarivelo, Aruna Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2503.22749)  

**Abstract**: In the era of data-driven machine-learning applications, privacy concerns and the scarcity of labeled data have become paramount challenges. These challenges are particularly pronounced in the domain of few-shot learning, where the ability to learn from limited labeled data is crucial. Privacy-preserving few-shot learning algorithms have emerged as a promising solution to address such pronounced challenges. However, it is well-known that privacy-preserving techniques often lead to a drop in utility due to the fundamental trade-off between data privacy and model performance. To enhance the utility of privacy-preserving few-shot learning methods, we introduce a novel approach called Meta-Clip. This technique is specifically designed for meta-learning algorithms, including Differentially Private (DP) model-agnostic meta-learning, DP-Reptile, and DP-MetaSGD algorithms, with the objective of balancing data privacy preservation with learning capacity maximization. By dynamically adjusting clipping thresholds during the training process, our Adaptive Clipping method provides fine-grained control over the disclosure of sensitive information, mitigating overfitting on small datasets and significantly improving the generalization performance of meta-learning models. Through comprehensive experiments on diverse benchmark datasets, we demonstrate the effectiveness of our approach in minimizing utility degradation, showcasing a superior privacy-utility trade-off compared to existing privacy-preserving techniques. The adoption of Adaptive Clipping represents a substantial step forward in the field of privacy-preserving few-shot learning, empowering the development of secure and accurate models for real-world applications, especially in scenarios where there are limited data availability. 

**Abstract (ZH)**: 在数据驱动的机器学习时代，隐私保护和标注数据稀缺已成为主要挑战。这些挑战在少数样本学习领域尤为显著，该领域需要从有限的标注数据中学习的能力至关重要。针对这些显著的挑战，隐私保护少数样本学习算法作为一种有前景的解决方案而出现。然而，众所周知，隐私保护技术往往会由于数据隐私与模型性能之间的基本权衡而导致实用性下降。为了提高隐私保护少数样本学习方法的实用性，我们提出了一种名为Meta-Clip的新型方法。该方法专门设计用于元学习算法，包括差分隐私模型无关元学习、差分隐私Reptile和差分隐私MetaSGD算法，旨在平衡数据隐私保护与学习能力最大化。通过在训练过程中动态调整截断阈值，我们的自适应截断方法提供了对敏感信息披露程度的精细化控制，减轻了对小型数据集的过度拟合，并显著提高了元学习模型的泛化性能。通过在多种基准数据集上的全面实验，我们证明了该方法在最小化实用性下降方面的有效性，展示了与现有隐私保护技术相比更优的隐私-实用性权衡。自适应截断方法在隐私保护少数样本学习领域的采用代表了向前迈进的重要一步，为开发安全而准确的模型以应对实际应用中数据有限的情景提供了有力支持。 

---
# Ignite Forecasting with SPARK: An Efficient Generative Framework for Refining LLMs in Temporal Knowledge Graph Forecasting 

**Title (ZH)**: 用SPARK点燃预测：一种高效的生成框架，用于细化时间知识图谱forecasting中的LLMs 

**Authors**: Gongzhu Yin, Hongli Zhang, Yi Luo, Yuchen Yang, Kun Lu, Chao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2503.22748)  

**Abstract**: Temporal Knowledge Graph (TKG) forecasting is crucial for predicting future events using historical data. With the surge of Large Language Models (LLMs), recent studies have begun exploring their integration into TKG forecasting and achieved some success. However, they still face limitations such as limited input length, inefficient output generation, and resource-intensive refinement, which undermine their performance and practical applicability. To address these limitations, we introduce SPARK, a Sequence-level Proxy-Adapting framework for Refining LLMs in TKG forecasting. Inspired by inference-time algorithms adopted in controlling generation, SPARK offers a cost-effective, plug-and-play solution through two key innovations: (1) Beam Sequence-Level Generation, which reframes TKG forecasting as a top-K sequence-level generation task, using beam search for efficiently generating next-entity distribution in a single forward pass. (2) TKG Adapter for Refinement, which employs traditional TKG models as trainable proxy adapters to leverage global graph information and refine LLM outputs, overcoming both the input length and the resource-intensive fine-tuning problems. Experiments across diverse datasets validate SPARK's forecasting performance, robust generalization capabilities, and high efficiency. We release source codes at this https URL. 

**Abstract (ZH)**: 基于序列级代理适应的TKG预测中LLM精炼框架SPARK 

---
# LeForecast: Enterprise Hybrid Forecast by Time Series Intelligence 

**Title (ZH)**: LeForecast: 企业时序混合预测 

**Authors**: Zheng Tan, Yiwen Nie, Wenfa Wu, Guanyu Zhang, Yanze Liu, Xinyuan Tian, Kailin Gao, Mengya Liu, Qijiang Cheng, Haipeng Jiang, Yingzheng Ma, Wei Zheng, Yuci Zhu, Yuanyuan Sun, Xiangyu Lei, Xiyu Guan, Wanqing Huang, Shouming Liu, Xiangquan Meng, Pengzhan Qu, Chao Yang, Jiaxuan Fan, Yuan He, Hongsheng Qi, Yangzhou Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.22747)  

**Abstract**: Demand is spiking in industrial fields for multidisciplinary forecasting, where a broad spectrum of sectors needs planning and forecasts to streamline intelligent business management, such as demand forecasting, product planning, inventory optimization, etc. Specifically, these tasks expecting intelligent approaches to learn from sequentially collected historical data and then foresee most possible trend, i.e. time series forecasting. Challenge of it lies in interpreting complex business contexts and the efficiency and generalisation of modelling. With aspirations of pre-trained foundational models for such purpose, given their remarkable success of large foundation model across legions of tasks, we disseminate \leforecast{}, an enterprise intelligence platform tailored for time series tasks. It integrates advanced interpretations of time series data and multi-source information, and a three-pillar modelling engine combining a large foundation model (Le-TSFM), multimodal model and hybrid model to derive insights, predict or infer futures, and then drive optimisation across multiple sectors in enterprise operations. The framework is composed by a model pool, model profiling module, and two different fusion approaches regarding original model architectures. Experimental results verify the efficiency of our trail fusion concepts: router-based fusion network and coordination of large and small models, resulting in high costs for redundant development and maintenance of models. This work reviews deployment of LeForecast and its performance in three industrial use cases. Our comprehensive experiments indicate that LeForecast is a profound and practical platform for efficient and competitive performance. And we do hope that this work can enlighten the research and grounding of time series techniques in accelerating enterprise. 

**Abstract (ZH)**: 工业领域对多学科预测的需求激增：LeForecast企业智能平台在时间序列任务中的应用 

---
# Susceptibility of Large Language Models to User-Driven Factors in Medical Queries 

**Title (ZH)**: 大型语言模型在医疗查询中的易感性对用户驱动因素的反应 

**Authors**: Kyung Ho Lim, Ujin Kang, Xiang Li, Jin Sung Kim, Young-Chul Jung, Sangjoon Park, Byung-Hoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.22746)  

**Abstract**: Large language models (LLMs) are increasingly used in healthcare, but their reliability is heavily influenced by user-driven factors such as question phrasing and the completeness of clinical information. In this study, we examined how misinformation framing, source authority, model persona, and omission of key clinical details affect the diagnostic accuracy and reliability of LLM outputs. We conducted two experiments: one introducing misleading external opinions with varying assertiveness (perturbation test), and another removing specific categories of patient information (ablation test). Using public datasets (MedQA and Medbullets), we evaluated proprietary models (GPT-4o, Claude 3.5 Sonnet, Claude 3.5 Haiku, Gemini 1.5 Pro, Gemini 1.5 Flash) and open-source models (LLaMA 3 8B, LLaMA 3 Med42 8B, DeepSeek R1 8B). All models were vulnerable to user-driven misinformation, with proprietary models especially affected by definitive and authoritative language. Assertive tone had the greatest negative impact on accuracy. In the ablation test, omitting physical exam findings and lab results caused the most significant performance drop. Although proprietary models had higher baseline accuracy, their performance declined sharply under misinformation. These results highlight the need for well-structured prompts and complete clinical context. Users should avoid authoritative framing of misinformation and provide full clinical details, especially for complex cases. 

**Abstract (ZH)**: 大型语言模型在医疗领域的可靠性和影响因素探究：错误信息框架、信息源权威性、模型人设和关键临床细节缺失对诊断准确性和可靠性的影响 

---
# Adaptive Integrated Layered Attention (AILA) 

**Title (ZH)**: 自适应集成分层注意力（AILA） 

**Authors**: William Claster, Suhas KM, Dhairya Gundechia  

**Link**: [PDF](https://arxiv.org/pdf/2503.22742)  

**Abstract**: We propose Adaptive Integrated Layered Attention (AILA), a neural network architecture that combines dense skip connections with different mechanisms for adaptive feature reuse across network layers. We evaluate AILA on three challenging tasks: price forecasting for various commodities and indices (S&P 500, Gold, US dollar Futures, Coffee, Wheat), image recognition using the CIFAR-10 dataset, and sentiment analysis on the IMDB movie review dataset. In all cases, AILA matches strong deep learning baselines (LSTMs, Transformers, and ResNets), achieving it at a fraction of the training and inference time. Notably, we implement and test two versions of the model - AILA-Architecture 1, which uses simple linear layers as the connection mechanism between layers, and AILA-Architecture 2, which implements an attention mechanism to selectively focus on outputs from previous layers. Both architectures are applied in a single-task learning setting, with each model trained separately for individual tasks. Results confirm that AILA's adaptive inter-layer connections yield robust gains by flexibly reusing pertinent features at multiple network depths. The AILA approach thus presents an extension to existing architectures, improving long-range sequence modeling, image recognition with optimised computational speed, and SOTA classification performance in practice. 

**Abstract (ZH)**: 我们提出自适应集成分层注意机制（AILA），这是一种结合了密集跳连连接和不同机制的神经网络架构，用于在网络层间适应性重用特征。我们在三项具有挑战性的任务上评估了AILA：各种商品和指数（S&P 500、黄金、美国期货美元、咖啡、小麦）的价格预测，CIFAR-10数据集上的图像识别，以及IMDB电影评论数据集上的情感分析。在所有情况下，AILA在训练和推断时间仅为强深度学习基线（LSTMs、Transformers和ResNets）的一小部分的情况下，实现了与这些基线相当的结果。值得注意的是，我们实现了并测试了该模型的两个版本——使用简单线性层作为层间连接机制的AILA-Architecture 1，以及实现注意机制以有选择地关注前一层输出的AILA-Architecture 2。这两种架构分别应用于单一任务学习场景中，每个模型独立针对各自任务进行训练。结果证实，AILA的自适应跨层连接通过在多个网络深度灵活重用相关特征，实现了稳健的增益。因此，AILA方法扩展了现有架构，提高了长序列建模、优化计算速度的图像识别以及实际中的最佳分类性能。 

---
# CSPO: Cross-Market Synergistic Stock Price Movement Forecasting with Pseudo-volatility Optimization 

**Title (ZH)**: 跨市场协同股票价格运动预测与伪波动率优化 

**Authors**: Sida Lin, Yankai Chen, Yiyan Qi, Chenhao Ma, Bokai Cao, Yifei Zhang, Xue Liu, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.22740)  

**Abstract**: The stock market, as a cornerstone of the financial markets, places forecasting stock price movements at the forefront of challenges in quantitative finance. Emerging learning-based approaches have made significant progress in capturing the intricate and ever-evolving data patterns of modern markets. With the rapid expansion of the stock market, it presents two characteristics, i.e., stock exogeneity and volatility heterogeneity, that heighten the complexity of price forecasting. Specifically, while stock exogeneity reflects the influence of external market factors on price movements, volatility heterogeneity showcases the varying difficulty in movement forecasting against price fluctuations. In this work, we introduce the framework of Cross-market Synergy with Pseudo-volatility Optimization (CSPO). Specifically, CSPO implements an effective deep neural architecture to leverage external futures knowledge. This enriches stock embeddings with cross-market insights and thus enhances the CSPO's predictive capability. Furthermore, CSPO incorporates pseudo-volatility to model stock-specific forecasting confidence, enabling a dynamic adaptation of its optimization process to improve accuracy and robustness. Our extensive experiments, encompassing industrial evaluation and public benchmarking, highlight CSPO's superior performance over existing methods and effectiveness of all proposed modules contained therein. 

**Abstract (ZH)**: 基于伪波动率优化的跨市场协同框架（CSPO） 

---
# Cyborg Data: Merging Human with AI Generated Training Data 

**Title (ZH)**: 人工增强数据：融合人类与AI生成的训练数据 

**Authors**: Kai North, Christopher Ormerod  

**Link**: [PDF](https://arxiv.org/pdf/2503.22736)  

**Abstract**: Automated scoring (AS) systems used in large-scale assessment have traditionally used small statistical models that require a large quantity of hand-scored data to make accurate predictions, which can be time-consuming and costly. Generative Large Language Models are trained on many tasks and have shown impressive abilities to generalize to new tasks with little to no data. While these models require substantially more computational power to make predictions, they still require some fine-tuning to meet operational standards. Evidence suggests that these models can exceed human-human levels of agreement even when fine-tuned on small amounts of data. With this in mind, we propose a model distillation pipeline in which a large generative model, a Teacher, teaches a much smaller model, a Student. The Teacher, trained on a small subset of the training data, is used to provide scores on the remaining training data, which is then used to train the Student. We call the resulting dataset "Cyborg Data", as it combines human and machine-scored responses. Our findings show that Student models trained on "Cyborg Data" show performance comparable to training on the entire dataset, while only requiring 10% of the original hand-scored data. 

**Abstract (ZH)**: 自动评分（AS）系统在大规模评估中 traditionally 使用小统计模型，这些模型需要大量手工评分数据才能做出准确预测，这可能会耗费大量时间和成本。生成型大规模语言模型在许多任务上进行了训练，并且展示出即使在少量数据下也能泛化到新任务的强大能力。尽管这些模型生成预测所需的计算资源更多，但在优化标准方面仍需一定程度的微调。证据表明，即使在少量数据下微调，这些模型也能够超过人类手工评分的水平。基于此，我们提出了一种模型蒸馏管道，在这种管道中，一个大型生成模型（教师）向一个更小的模型（学生）传授知识。教师在训练数据的小子集中进行训练，用于对剩余训练数据进行评分，然后使用这些评分数据来训练学生。我们将由此产生的数据集称为“半机械人数据”，因为它结合了人类和机器评分的响应。我们的研究结果表明，使用“半机械人数据”训练的学生模型在性能上与使用完整数据集训练的模型相当，但仍只需原始手工评分数据的10%。 

---
# Ancestral Mamba: Enhancing Selective Discriminant Space Model with Online Visual Prototype Learning for Efficient and Robust Discriminant Approach 

**Title (ZH)**: 祖先环蛇：结合在线视觉原型学习以增强选择性 discriminant 空间模型，实现高效和稳健的鉴别方法 

**Authors**: Jiahao Qin, Feng Liu, Lu Zong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22729)  

**Abstract**: In the realm of computer graphics, the ability to learn continuously from non-stationary data streams while adapting to new visual patterns and mitigating catastrophic forgetting is of paramount importance. Existing approaches often struggle to capture and represent the essential characteristics of evolving visual concepts, hindering their applicability to dynamic graphics tasks. In this paper, we propose Ancestral Mamba, a novel approach that integrates online prototype learning into a selective discriminant space model for efficient and robust online continual learning. The key components of our approach include Ancestral Prototype Adaptation (APA), which continuously refines and builds upon learned visual prototypes, and Mamba Feedback (MF), which provides targeted feedback to adapt to challenging visual patterns. APA enables the model to continuously adapt its prototypes, building upon ancestral knowledge to tackle new challenges, while MF acts as a targeted feedback mechanism, focusing on challenging classes and refining their representations. Extensive experiments on graphics-oriented datasets, such as CIFAR-10 and CIFAR-100, demonstrate the superior performance of Ancestral Mamba compared to state-of-the-art baselines, achieving significant improvements in accuracy and forgetting mitigation. 

**Abstract (ZH)**: 在计算机图形学领域，能够在非平稳数据流中持续学习、适应新视觉模式并缓解灾难性遗忘的能力至关重要。现有方法往往难以捕捉并表示不断演化的视觉概念的本质特征，限制了其在动态图形任务中的应用。本文提出了一种名为Ancestral Mamba的新方法，该方法将在线原型学习集成到选择性判别空间模型中，以实现高效而稳健的在线持续学习。该方法的关键组成部分包括祖先原型适应（APA），它连续 refining 并建立已学视觉原型，以及Mamba 反馈（MF），它提供针对性反馈以适应具有挑战性的视觉模式。APA 使模型能够连续适应其原型，基于祖先知识应对新挑战，而 MF 作为针对性反馈机制，专注于具有挑战性的类别并改进其表示。在以CIFAR-10和CIFAR-100为代表的图形导向数据集上的大量实验表明，Ancestral Mamba 在准确性和遗忘缓解方面显著优于现有最先进的基线方法。 

---
# Zero-Shot LLMs in Human-in-the-Loop RL: Replacing Human Feedback for Reward Shaping 

**Title (ZH)**: 零样本LLMs在人类在环RL中的应用：用奖励塑形取代人类反馈 

**Authors**: Mohammad Saif Nazir, Chayan Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.22723)  

**Abstract**: Reinforcement learning often faces challenges with reward misalignment, where agents optimize for given rewards but fail to exhibit the desired behaviors. This occurs when the reward function incentivizes proxy behaviors that diverge from the true objective. While human-in-the-loop (HIL) methods can help, they may exacerbate the problem, as humans are prone to biases that lead to inconsistent, subjective, or misaligned feedback, complicating the learning process. To address these issues, we propose two key contributions. First, we extend the use of zero-shot, off-the-shelf large language models (LLMs) for reward shaping beyond natural language processing (NLP) to continuous control tasks. By leveraging LLMs as direct feedback providers, we replace surrogate models trained on human feedback, which often suffer from the bias inherent in the feedback data it is trained on. Second, we introduce a hybrid framework (LLM-HFBF) that enables LLMs to identify and correct biases in human feedback while incorporating this feedback into the reward shaping process. The LLM-HFBF framework creates a more balanced and reliable system by addressing both the limitations of LLMs (e.g., lack of domain-specific knowledge) and human supervision (e.g., inherent biases). By enabling human feedback bias flagging and correction, our approach improves reinforcement learning performance and reduces reliance on potentially biased human guidance. Empirical experiments show that biased human feedback significantly reduces performance, with average episodic reward (AER) dropping from 28.472 in (unbiased approaches) to 7.039 (biased with conservative bias). In contrast, LLM-based approaches maintain a matching AER like unbiased feedback, even in custom edge case scenarios. 

**Abstract (ZH)**: 强化学习 Often 遇到奖励错配的挑战，其中智能体优化给定奖励但未能体现出预期行为。当奖励函数激励与真目标相悖的代理行为时，这种情况就会发生。虽然人类在环（Human-in-the-loop, HIL）方法可以帮助解决这个问题，但人类易受偏差影响，可能导致不一致、主观或错配的反馈，从而复杂化学习过程。为应对这些问题，我们提出了两项关键贡献。首先，我们扩展了零样本、即用型大型语言模型（LLM）在奖励塑形中的应用，从自然语言处理（NLP）领域扩展到连续控制任务。通过利用LLM直接提供反馈，我们替代了基于人类反馈训练的代理模型，后者往往受到反馈数据固有的偏差所影响。其次，我们引入了一种混合框架（LLM-HFBF），该框架使LLM能够识别和纠正人类反馈中的偏差，并将这些反馈纳入奖励塑形过程。LLM-HFBF框架通过同时解决LLM的局限性（如缺乏领域特定知识）和人类监督的固有偏差，创造了更为平衡和可靠的系统。通过允许人类反馈偏差标记和纠正，我们的方法提高了强化学习性能，并减少了对潜在偏差的人类指导的依赖。实验证明，带有偏差的人类反馈显著降低了性能，平均 episodic 奖励（AER）从（无偏差方法）的 28.472 下降到 7.039（带有保守偏差的有偏差）。相比之下，基于LLM的方法即使在定制的边缘情况下也能保持与无偏差反馈相匹配的AER。 

---
# Why Representation Engineering Works: A Theoretical and Empirical Study in Vision-Language Models 

**Title (ZH)**: 为什么表示工程有效：视觉-语言模型中的理论与实证研究 

**Authors**: Bowei Tian, Xuntao Lyu, Meng Liu, Hongyi Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.22720)  

**Abstract**: Representation Engineering (RepE) has emerged as a powerful paradigm for enhancing AI transparency by focusing on high-level representations rather than individual neurons or circuits. It has proven effective in improving interpretability and control, showing that representations can emerge, propagate, and shape final model outputs in large language models (LLMs). However, in Vision-Language Models (VLMs), visual input can override factual linguistic knowledge, leading to hallucinated responses that contradict reality. To address this challenge, we make the first attempt to extend RepE to VLMs, analyzing how multimodal representations are preserved and transformed. Building on our findings and drawing inspiration from successful RepE applications, we develop a theoretical framework that explains the stability of neural activity across layers using the principal eigenvector, uncovering the underlying mechanism of RepE. We empirically validate these instrinsic properties, demonstrating their broad applicability and significance. By bridging theoretical insights with empirical validation, this work transforms RepE from a descriptive tool into a structured theoretical framework, opening new directions for improving AI robustness, fairness, and transparency. 

**Abstract (ZH)**: Representation Engineering Extension to Vision-Language Models: Analyzing and Explaining Stability Mechanisms for Improved Robustness, Fairness, and Transparency 

---
# TRIDIS: A Comprehensive Medieval and Early Modern Corpus for HTR and NER 

**Title (ZH)**: TRIDIS: 一个全面的中世纪和早期现代手写文本语料库用于光学字符识别和命名实体识别 

**Authors**: Sergio Torres Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2503.22714)  

**Abstract**: This paper introduces TRIDIS (Tria Digita Scribunt), an open-source corpus of medieval and early modern manuscripts. TRIDIS aggregates multiple legacy collections (all published under open licenses) and incorporates large metadata descriptions. While prior publications referenced some portions of this corpus, here we provide a unified overview with a stronger focus on its constitution. We describe (i) the narrative, chronological, and editorial background of each major sub-corpus, (ii) its semi-diplomatic transcription rules (expansion, normalization, punctuation), (iii) a strategy for challenging out-of-domain test splits driven by outlier detection in a joint embedding space, and (iv) preliminary baseline experiments using TrOCR and MiniCPM2.5 comparing random and outlier-based test partitions. Overall, TRIDIS is designed to stimulate joint robust Handwritten Text Recognition (HTR) and Named Entity Recognition (NER) research across medieval and early modern textual heritage. 

**Abstract (ZH)**: TRIDIS（Tria Digita Scribunt）：中世纪和早期现代手稿的开源语料库 

---
# Chirp Localization via Fine-Tuned Transformer Model: A Proof-of-Concept Study 

**Title (ZH)**: 基于微调Transformer模型的脉冲定位：一个概念验证研究 

**Authors**: Nooshin Bahador, Milad Lankarany  

**Link**: [PDF](https://arxiv.org/pdf/2503.22713)  

**Abstract**: Spectrograms are pivotal in time-frequency signal analysis, widely used in audio processing and computational neuroscience. Chirp-like patterns in electroencephalogram (EEG) spectrograms (marked by linear or exponential frequency sweep) are key biomarkers for seizure dynamics, but automated tools for their detection, localization, and feature extraction are lacking. This study bridges this gap by fine-tuning a Vision Transformer (ViT) model on synthetic spectrograms, augmented with Low-Rank Adaptation (LoRA) to boost adaptability. We generated 100000 synthetic spectrograms with chirp parameters, creating the first large-scale benchmark for chirp localization. These spectrograms mimic neural chirps using linear or exponential frequency sweep, Gaussian noise, and smoothing. A ViT model, adapted for regression, predicted chirp parameters. LoRA fine-tuned the attention layers, enabling efficient updates to the pre-trained backbone. Training used MSE loss and the AdamW optimizer, with a learning rate scheduler and early stopping to curb overfitting. Only three features were targeted: Chirp Start Time (Onset Time), Chirp Start Frequency (Onset Frequency), and Chirp End Frequency (Offset Frequency). Performance was evaluated via Pearson correlation between predicted and actual labels. Results showed strong alignment: 0.9841 correlation for chirp start time, with stable inference times (137 to 140s) and minimal bias in error distributions. This approach offers a tool for chirp analysis in EEG time-frequency representation, filling a critical methodological void. 

**Abstract (ZH)**: Spectrogram分析中基于Vision Transformer的 chirp疑似波检测与特征提取方法 

---
# Modeling speech emotion with label variance and analyzing performance across speakers and unseen acoustic conditions 

**Title (ZH)**: 基于标签方差建模语音情感并分析 Across Speakers 和未见声学条件下性能 

**Authors**: Vikramjit Mitra, Amrit Romana, Dung T. Tran, Erdrin Azemi  

**Link**: [PDF](https://arxiv.org/pdf/2503.22711)  

**Abstract**: Spontaneous speech emotion data usually contain perceptual grades where graders assign emotion score after listening to the speech files. Such perceptual grades introduce uncertainty in labels due to grader opinion variation. Grader variation is addressed by using consensus grades as groundtruth, where the emotion with the highest vote is selected. Consensus grades fail to consider ambiguous instances where a speech sample may contain multiple emotions, as captured through grader opinion uncertainty. We demonstrate that using the probability density function of the emotion grades as targets instead of the commonly used consensus grades, provide better performance on benchmark evaluation sets compared to results reported in the literature. We show that a saliency driven foundation model (FM) representation selection helps to train a state-of-the-art speech emotion model for both dimensional and categorical emotion recognition. Comparing representations obtained from different FMs, we observed that focusing on overall test-set performance can be deceiving, as it fails to reveal the models generalization capacity across speakers and gender. We demonstrate that performance evaluation across multiple test-sets and performance analysis across gender and speakers are useful in assessing usefulness of emotion models. Finally, we demonstrate that label uncertainty and data-skew pose a challenge to model evaluation, where instead of using the best hypothesis, it is useful to consider the 2- or 3-best hypotheses. 

**Abstract (ZH)**: 自发语音情感数据通常包含感知等级，评分者在听取语音文件后为其赋予情感得分。这种感知等级由于评分者的观点差异而引入标签不确定性。通过使用共识等级作为ground truth，其中情感得分最高者当选，解决了评分者差异问题。但共识等级未能考虑模糊实例，即语音样本可能包含多种情感，这些通过评分者意见的不确定性表现出来。我们证明，将情感分数的概率密度函数作为目标，而不是通常使用的共识等级，可以在基准评估集上获得更好的性能，优于文献报道的结果。我们展示了情感驱动的基础模型（FM）表示选择有助于训练最先进的语音情感模型，用于情感维度和类别识别。比较不同FM获得的表示，我们观察到关注整体测试集性能可能是误导的，因为它未能揭示模型在说话人和性别方面的一般化能力。我们证明，跨多个测试集的性能评估和性别、说话人层面的性能分析是有用的评估情感模型的手段。最后，我们证明，标签不确定性与数据倾斜对模型评估构成挑战，使用前两个或前三个假设比使用最佳假设更有用。 

---
# Validating Emergency Department Admission Predictions Based on Local Data Through MIMIC-IV 

**Title (ZH)**: 基于本地数据通过MIMIC-IV验证急诊住院预测 

**Authors**: Francesca Meimeti, Loukas Triantafyllopoulos, Aikaterini Sakagianni, Vasileios Kaldis, Lazaros Tzelves, Nikolaos Theodorakis, Evgenia Paxinou, Georgios Feretzakis, Dimitris Kalles, Vassilios S. Verykios  

**Link**: [PDF](https://arxiv.org/pdf/2503.22706)  

**Abstract**: The effective management of Emergency Department (ED) overcrowding is essential for improving patient outcomes and optimizing healthcare resource allocation. This study validates hospital admission prediction models initially developed using a small local dataset from a Greek hospital by leveraging the comprehensive MIMIC-IV dataset. After preprocessing the MIMIC-IV data, five algorithms were evaluated: Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Random Forest (RF), Recursive Partitioning and Regression Trees (RPART), and Support Vector Machines (SVM Radial). Among these, RF demonstrated superior performance, achieving an Area Under the Receiver Operating Characteristic Curve (AUC-ROC) of 0.9999, sensitivity of 0.9997, and specificity of 0.9999 when applied to the MIMIC-IV data. These findings highlight the robustness of RF in handling complex datasets for admission prediction, establish MIMIC-IV as a valuable benchmark for validating models based on smaller local datasets, and provide actionable insights for improving ED management strategies. 

**Abstract (ZH)**: 有效管理急诊部（ED）过度拥挤对于改善患者结果和优化医疗卫生资源分配至关重要。本研究利用希腊医院的小规模本地数据集初次开发的住院预测模型，并借助全面的MIMIC-IV数据集进行验证。经过预处理的MIMIC-IV数据后，评估了五种算法：线性判别分析（LDA）、K近邻（KNN）、随机森林（RF）、递归分区和回归树（RPART）和支持向量机（SVM径向基）。其中，RF表现出色，应用于MIMIC-IV数据时，达到接收者操作特征曲线下的面积（AUC-ROC）为0.9999、敏感性为0.9997和特异性为0.9999。这些发现强调了RF在处理复杂数据集进行住院预测中的稳健性，确立了MIMIC-IV作为基于较小本地数据集验证模型的重要基准，并提供了改善急诊部管理策略的实际见解。 

---
# Enhancing nonnative speech perception and production through an AI-powered application 

**Title (ZH)**: 通过AI赋能应用增强非母语者的语音感知与生产能力 

**Authors**: Georgios P. Georgiou  

**Link**: [PDF](https://arxiv.org/pdf/2503.22705)  

**Abstract**: While research on using Artificial Intelligence (AI) through various applications to enhance foreign language pronunciation is expanding, it has primarily focused on aspects such as comprehensibility and intelligibility, largely neglecting the improvement of individual speech sounds in both perception and production. This study seeks to address this gap by examining the impact of training with an AI-powered mobile application on nonnative sound perception and production. Participants completed a pretest assessing their ability to discriminate the second language English heed-hid contrast and produce these vowels in sentence contexts. The intervention involved training with the Speakometer mobile application, which incorporated recording tasks featuring the English vowels, along with pronunciation feedback and practice. The posttest mirrored the pretest to measure changes in performance. The results revealed significant improvements in both discrimination accuracy and production of the target contrast following the intervention. However, participants did not achieve native-like competence. These findings highlight the effectiveness of AI-powered applications in facilitating speech acquisition and support their potential use for personalized, interactive pronunciation training beyond the classroom. 

**Abstract (ZH)**: 通过人工智能移动应用训练提高非母语声学感知与生产的研究 

---
# From Eye to Mind: brain2text Decoding Reveals the Neural Mechanisms of Visual Semantic Processing 

**Title (ZH)**: 从眼及至脑：brain2text解码揭示视觉语义处理的神经机制 

**Authors**: Feihan Feng, Jingxin Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.22697)  

**Abstract**: Deciphering the neural mechanisms that transform sensory experiences into meaningful semantic representations is a fundamental challenge in cognitive neuroscience. While neuroimaging has mapped a distributed semantic network, the format and neural code of semantic content remain elusive, particularly for complex, naturalistic stimuli. Traditional brain decoding, focused on visual reconstruction, primarily captures low-level perceptual features, missing the deeper semantic essence guiding human cognition. Here, we introduce a paradigm shift by directly decoding fMRI signals into textual descriptions of viewed natural images. Our novel deep learning model, trained without visual input, achieves state-of-the-art semantic decoding performance, generating meaningful captions that capture the core semantic content of complex scenes. Neuroanatomical analysis reveals the critical role of higher-level visual regions, including MT+, ventral stream visual cortex, and inferior parietal cortex, in this semantic transformation. Category-specific decoding further demonstrates nuanced neural representations for semantic dimensions like animacy and motion. This text-based decoding approach provides a more direct and interpretable window into the brain's semantic encoding than visual reconstruction, offering a powerful new methodology for probing the neural basis of complex semantic processing, refining our understanding of the distributed semantic network, and potentially inspiring brain-inspired language models. 

**Abstract (ZH)**: 解读将感官体验转化为有意义语义表示的神经机制是认知神经科学中的一个基本挑战。虽然神经成像已映射出一个分布式的语义网络，但语义内容的形式及其神经编码方式仍然模糊，尤其是在复杂自然刺激方面。传统的大脑解码主要侧重于视觉重建，主要捕捉低层级知觉特征，而忽略了指导人类认知的深层语义本质。在此，我们通过直接将fMRI信号解码为所观看自然图像的文本描述，引入了一种范式转变。我们的新型深度学习模型在未使用视觉输入的情况下，实现了最先进的语义解码性能，生成能够捕捉复杂场景核心语义内容的有意义描述。神经解剖分析揭示了较高层级的视觉区域，包括MT+、腹侧视皮层和背侧下顶叶皮层，在这一语义转变中的关键作用。类别特异性的解码进一步证明了语义维度（如有生命性和运动性）的精巧神经表征。基于文本的解码方法提供了比视觉重建更直接和可解释的窗口，以观察大脑的语义编码，并提供了一种探索复杂语义处理神经基础的强大新方法，有助于细化分散的语义网络的理解，并可能启发神经启发的语言模型。 

---
# Bridging Language Models and Financial Analysis 

**Title (ZH)**: 语言模型与金融分析的桥梁 

**Authors**: Alejandro Lopez-Lira, Jihoon Kwon, Sangwoon Yoon, Jy-yong Sohn, Chanyeol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.22693)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) have unlocked transformative possibilities in natural language processing, particularly within the financial sector. Financial data is often embedded in intricate relationships across textual content, numerical tables, and visual charts, posing challenges that traditional methods struggle to address effectively. However, the emergence of LLMs offers new pathways for processing and analyzing this multifaceted data with increased efficiency and insight. Despite the fast pace of innovation in LLM research, there remains a significant gap in their practical adoption within the finance industry, where cautious integration and long-term validation are prioritized. This disparity has led to a slower implementation of emerging LLM techniques, despite their immense potential in financial applications. As a result, many of the latest advancements in LLM technology remain underexplored or not fully utilized in this domain. This survey seeks to bridge this gap by providing a comprehensive overview of recent developments in LLM research and examining their applicability to the financial sector. Building on previous survey literature, we highlight several novel LLM methodologies, exploring their distinctive capabilities and their potential relevance to financial data analysis. By synthesizing insights from a broad range of studies, this paper aims to serve as a valuable resource for researchers and practitioners, offering direction on promising research avenues and outlining future opportunities for advancing LLM applications in finance. 

**Abstract (ZH)**: 大型语言模型的快速进步为自然语言处理带来了变革性可能性，特别是在金融领域。金融数据通常嵌入在文本内容、数字表格和视觉图表的复杂关系中，这给传统方法带来了挑战。然而，大型语言模型的出现为高效且深入地处理和分析这种多维度数据提供了新的途径。尽管大型语言模型研究的创新步伐迅速，但在金融行业中，谨慎的集成和长期验证仍是优先事项，这导致了显著的实际应用差距。尽管新兴的大型语言模型技术具有巨大的金融应用潜力，但其应用实施仍然较为缓慢。因此，许多最新在大型语言模型技术的最新进展在该领域仍被未尽探索或未充分利用。本次综述旨在通过提供大型语言模型研究的全面概述，并探讨其在金融领域的适用性来弥补这一差距。基于之前的综述文献，本文突出了一些新颖的大型语言模型方法，探讨了它们的独特能力及其在金融数据分析中的潜在相关性。通过综合广泛的学术研究洞察，本文旨在为研究者和从业者提供有价值的资源，指明有前景的研究方向，并展望未来大型语言模型在金融领域的应用机会。 

---
# Enhancing Aviation Communication Transcription: Fine-Tuning Distil-Whisper with LoRA 

**Title (ZH)**: 增强航空通信转录：基于LoRA微调Distil-Whisper 

**Authors**: Shokoufeh Mirzaei, Jesse Arzate, Yukti Vijay  

**Link**: [PDF](https://arxiv.org/pdf/2503.22692)  

**Abstract**: Transcription of aviation communications has several applications, from assisting air traffic controllers in identifying the accuracy of read-back errors to search and rescue operations. Recent advances in artificial intelligence have provided unprecedented opportunities for improving aviation communication transcription tasks. OpenAI's Whisper is one of the leading automatic speech recognition models. However, fine-tuning Whisper for aviation communication transcription is not computationally efficient. Thus, this paper aims to use a Parameter-Efficient Fine-tuning method called Low-Rank Adaptation to fine-tune a more computationally efficient version of Whisper, distil-Whisper. To perform the fine-tuning, we used the Air Traffic Control Corpus dataset from the Linguistic Data Consortium, which contains approximately 70 hours of controller and pilot transmissions near three major airports in the US. The objective was to reduce the word error rate to enhance accuracy in the transcription of aviation communication. First, starting with an initial set of hyperparameters for LoRA (Alpha = 64 and Rank = 32), we performed a grid search. We applied a 5-fold cross-validation to find the best combination of distil-Whisper hyperparameters. Then, we fine-tuned the model for LoRA hyperparameters, achieving an impressive average word error rate of 3.86% across five folds. This result highlights the model's potential for use in the cockpit. 

**Abstract (ZH)**: 航空通信转录具有多种应用，从协助空中交通管制员识别复诵错误的准确性到搜索救援行动。最近人工智能的进步为改善航空通信转录任务提供了前所未有的机会。OpenAI的Whisper是领先的自动语音识别模型之一。然而，将Whisper细调以适应航空通信转录在计算上不够高效。因此，本文旨在使用Parameter-Efficient Fine-tuning方法中的Low-Rank Adaptation技术来细调一个计算上更高效的Whisper版本distil-Whisper。为了进行细调，我们使用了Linguistic Data Consortium提供的Air Traffic Control Corpus数据集，该数据集包含约70小时的在美国三大机场附近的管制员和飞行员的通话记录。目标是降低字错误率，以提高航空通信转录的准确性。首先，我们采用初始LoRA超参数设置（Alpha = 64，Rank = 32）进行了网格搜索，并采用5折交叉验证来找到distil-Whisper的最佳超参数组合。然后，我们对LoRA超参数进行了模型细调，实现了令人印象深刻的整体平均字错误率3.86%。这一结果突显了该模型在驾驶舱中的应用潜力。 

---
# Qieemo: Speech Is All You Need in the Emotion Recognition in Conversations 

**Title (ZH)**: Qieemo: 话语即一切——在对话情感识别中无关紧要 

**Authors**: Jinming Chen, Jingyi Fang, Yuanzhong Zheng, Yaoxuan Wang, Haojun Fei  

**Link**: [PDF](https://arxiv.org/pdf/2503.22687)  

**Abstract**: Emotion recognition plays a pivotal role in intelligent human-machine interaction systems. Multimodal approaches benefit from the fusion of diverse modalities, thereby improving the recognition accuracy. However, the lack of high-quality multimodal data and the challenge of achieving optimal alignment between different modalities significantly limit the potential for improvement in multimodal approaches. In this paper, the proposed Qieemo framework effectively utilizes the pretrained automatic speech recognition (ASR) model backbone which contains naturally frame aligned textual and emotional features, to achieve precise emotion classification solely based on the audio modality. Furthermore, we design the multimodal fusion (MMF) module and cross-modal attention (CMA) module in order to fuse the phonetic posteriorgram (PPG) and emotional features extracted by the ASR encoder for improving recognition accuracy. The experimental results on the IEMOCAP dataset demonstrate that Qieemo outperforms the benchmark unimodal, multimodal, and self-supervised models with absolute improvements of 3.0%, 1.2%, and 1.9% respectively. 

**Abstract (ZH)**: 情感识别在智能人机交互系统中发挥着关键作用。多模态方法通过融合多种模态数据，从而提高识别准确性。然而，高质量多模态数据的缺乏以及不同模态之间达到最佳对齐的挑战极大地限制了多模态方法的改进潜力。本文提出的Qieemo框架有效利用了预训练的自动语音识别(ASR)模型骨干，该模型包含自然帧对齐的文本和情感特征，仅基于音频模态实现精确的情感分类。此外，我们设计了多模态融合(MMF)模块和跨模态注意力(CMA)模块，以结合ASR编码器提取的音素后验图(PPG)和情感特征，从而提高识别准确性。在IEMOCAP数据集上的实验结果表明，Qieemo在基准单模态、多模态和自监督模型中的绝对改进分别为3.0%、1.2%和1.9%。 

---
# Binary and Multi-Class Intrusion Detection in IoT Using Standalone and Hybrid Machine and Deep Learning Models 

**Title (ZH)**: 基于独立和混合机器与深度学习模型的物联网二分类和多分类入侵检测 

**Authors**: Md Ahnaf Akif  

**Link**: [PDF](https://arxiv.org/pdf/2503.22684)  

**Abstract**: Maintaining security in IoT systems depends on intrusion detection since these networks' sensitivity to cyber-attacks is growing. Based on the IoT23 dataset, this study explores the use of several Machine Learning (ML) and Deep Learning (DL) along with the hybrid models for binary and multi-class intrusion detection. The standalone machine and deep learning models like Random Forest (RF), Extreme Gradient Boosting (XGBoost), Artificial Neural Network (ANN), K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Convolutional Neural Network (CNN) were used. Furthermore, two hybrid models were created by combining machine learning techniques: RF, XGBoost, AdaBoost, KNN, and SVM and these hybrid models were voting based hybrid classifier. Where one is for binary, and the other one is for multi-class classification. These models vi were tested using precision, recall, accuracy, and F1-score criteria and compared the performance of each model. This work thoroughly explains how hybrid, standalone ML and DL techniques could improve IDS (Intrusion Detection System) in terms of accuracy and scalability in IoT (Internet of Things). 

**Abstract (ZH)**: 基于IoT23数据集的机器学习与深度学习及其混合模型在IoT系统中的二元与多类入侵检测研究：提高物联网安全的IDS性能与可扩展性 

---
# SPDZCoder: Combining Expert Knowledge with LLMs for Generating Privacy-Computing Code 

**Title (ZH)**: SPDZCoder: 结合专家知识与LLMs生成隐私计算代码 

**Authors**: Xiaoning Dong, Peilin Xin, Jia Li, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.00363)  

**Abstract**: Privacy computing receives increasing attention but writing privacy computing code remains challenging for developers due to limited library functions, necessitating function implementation from scratch, and data-oblivious requirement, contradicting intuitive thinking and usual practices of programmers. Automating the generation of privacy computing code with Large Language Models can streamline development effort and lower the barrier to using privacy computing frameworks. However, existing LLMs still encounter challenges in code translation for privacy-preserving computation, such as translating Python to MP-SPDZ, due to the scarcity of MP-SPDZ data required for effective pre-training or fine-tuning. Moreover, the lack of a benchmark further complicates the evaluation of translation quality. To address the limitations, this work proposes SPDZCoder, a rule-based framework that combines LLMs with expert knowledge for generating privacy-computing code without requiring additional training data. Specifically, SPDZCoder employ a rigorous procedure for collecting high-quality expert knowledge to represent the semantic-expressing differences between Python and MP-SPDZ, and to derive transformation rules for translating Python to MP-SPDZ based on these knowledge. Then, SPDZCoder progressively converts Python code into MP-SPDZ code using transformation rules in a three stage pipeline. To evaluate SPDZCoder, we manually constructed a benchmark dataset, SPDZEval, which comprises six data splits, each representing a distinct class of challenging tasks in MP-SPDZ implementation. Extensive experiments show that SPDZCoder achieves superior performance, significantly surpassing baselines in pass@1 and pass@2. Specifically, SPDZCoder attains an overall correctness of 85.94% and 92.01% in pass@1 and pass@2, respectively, whereas the best-performing baseline achieves 63.58% and 76.36%, respectively. 

**Abstract (ZH)**: 基于大型语言模型的隐私计算代码自动生成研究：SPDZCoder框架 

---
