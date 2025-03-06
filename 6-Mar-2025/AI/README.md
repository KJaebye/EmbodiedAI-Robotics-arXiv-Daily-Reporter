# CHOP: Mobile Operating Assistant with Constrained High-frequency Optimized Subtask Planning 

**Title (ZH)**: CHOP: 受约束高频率优化子任务规划的移动操作助理 

**Authors**: Yuqi Zhou, Shuai Wang, Sunhao Dai, Qinglin Jia, Zhaocheng Du, Zhenhua Dong, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03743)  

**Abstract**: The advancement of visual language models (VLMs) has enhanced mobile device operations, allowing simulated human-like actions to address user requirements. Current VLM-based mobile operating assistants can be structured into three levels: task, subtask, and action. The subtask level, linking high-level goals with low-level executable actions, is crucial for task completion but faces two challenges: ineffective subtasks that lower-level agent cannot execute and inefficient subtasks that fail to contribute to the completion of the higher-level task. These challenges stem from VLM's lack of experience in decomposing subtasks within GUI scenarios in multi-agent architecture. To address these, we propose a new mobile assistant architecture with constrained high-frequency o}ptimized planning (CHOP). Our approach overcomes the VLM's deficiency in GUI scenarios planning by using human-planned subtasks as the basis vector. We evaluate our architecture in both English and Chinese contexts across 20 Apps, demonstrating significant improvements in both effectiveness and efficiency. Our dataset and code is available at this https URL 

**Abstract (ZH)**: 视觉语言模型的进步增强了移动设备的操作，使模拟人类行为得以实现以满足用户需求。当前基于VLM的移动操作助手可以分为三个层次：任务、子任务和动作。子任务层次连接高层级目标与可执行的低层级动作，对于任务完成至关重要，但面临两大挑战：下层代理无法执行的无效子任务和不有助于完成高层任务的低效子任务。这些挑战源于VLM在多代理架构中GUI场景子任务分解经验不足。为应对这些挑战，我们提出了一种新的移动助手架构，即受限高频率优化规划（CHOP）。我们通过将人类规划的子任务作为基础向量来弥补VLM在GUI场景规划中的不足。我们在20个应用程序中分别以英文和中文环境评估了该架构，展示了在有效性和效率上的显著提升。我们的数据集和代码可在此网址访问：this https URL。 

---
# Machine Learning in Biomechanics: Key Applications and Limitations in Walking, Running, and Sports Movements 

**Title (ZH)**: 生物力学中的机器学习：步行、跑步和运动中关键应用与局限性 

**Authors**: Carlo Dindorf, Fabian Horst, Djordje Slijepčević, Bernhard Dumphart, Jonas Dully, Matthias Zeppelzauer, Brian Horsak, Michael Fröhlich  

**Link**: [PDF](https://arxiv.org/pdf/2503.03717)  

**Abstract**: This chapter provides an overview of recent and promising Machine Learning applications, i.e. pose estimation, feature estimation, event detection, data exploration & clustering, and automated classification, in gait (walking and running) and sports biomechanics. It explores the potential of Machine Learning methods to address challenges in biomechanical workflows, highlights central limitations, i.e. data and annotation availability and explainability, that need to be addressed, and emphasises the importance of interdisciplinary approaches for fully harnessing the potential of Machine Learning in gait and sports biomechanics. 

**Abstract (ZH)**: 本章提供了近期和有前景的机器学习在步态（行走和跑步）及运动生物力学中应用的综述，包括姿态估计、特征估计、事件检测、数据探索与聚类以及自动化分类。探讨了机器学习方法在生物力学工作流程中应对挑战的潜力，强调了数据和注释可用性及可解释性等核心限制需要解决，并强调了跨学科方法对于充分利用机器学习在步态和运动生物力学中的潜力的重要性。 

---
# ILLC: Iterative Layer-by-Layer Compression for Enhancing Structural Faithfulness in SpArX 

**Title (ZH)**: ILLC: 迭代逐层压缩以增强SpArX的结构忠实性 

**Authors**: Ungsik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.03693)  

**Abstract**: In the field of Explainable Artificial Intelligence (XAI), argumentative XAI approaches have been proposed to represent the internal reasoning process of deep neural networks in a more transparent way by interpreting hidden nodes as arguements. However, as the number of layers increases, existing compression methods simplify all layers at once, which lead to high accumulative information loss. To compensate for this, we propose an iterative layer-by-layer compression technique in which each layer is compressed separately and the reduction error in the next layer is immediately compensated for, thereby improving the overall input-output and structural fidelity of the model. Experiments on the Breast Cancer Diagnosis dataset show that, compared to traditional compression, the method reduces input-output and structural unfaithfulness, and maintains a more consistent attack-support relationship in the Argumentative Explanation scheme. This is significant because it provides a new way to make complex MLP models more compact while still conveying their internal inference logic without distortion. 

**Abstract (ZH)**: 在可解释人工智能（XAI）领域，提出了以论辩方式表达深度神经网络内部推理过程的解释性XAI方法，通过将隐藏节点解释为论点来提高透明度。然而，随着层的数量增加，现有的压缩方法会一次性简化所有层，导致累积信息丢失。为解决这一问题，我们提出了一种迭代的逐层压缩技术，每层分别进行压缩，并在下一层立即补偿上一层的压缩误差，从而提高模型的整体输入输出 fidelity 和结构一致性。在乳腺癌诊断数据集上的实验表明，与传统压缩方法相比，该方法减少了输入输出和结构不忠实性，并在论辩解释方案中维持了更一致的攻击支持关系。这一成果意义重大，因为它提供了一种在不扭曲内部推理逻辑的情况下使复杂MLP模型更加紧凑的新方法。 

---
# Parallelized Planning-Acting for Efficient LLM-based Multi-Agent Systems 

**Title (ZH)**: 基于大型语言模型的多智能体系统并行计划-执行高效算法 

**Authors**: Yaoru Li, Shunyu Liu, Tongya Zheng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.03505)  

**Abstract**: Recent advancements in Large Language Model(LLM)-based Multi-Agent Systems(MAS) have demonstrated remarkable potential for tackling complex decision-making tasks. However, existing frameworks inevitably rely on serialized execution paradigms, where agents must complete sequential LLM planning before taking action. This fundamental constraint severely limits real-time responsiveness and adaptation, which is crucial in dynamic environments with ever-changing scenarios. In this paper, we propose a novel parallelized planning-acting framework for LLM-based MAS, featuring a dual-thread architecture with interruptible execution to enable concurrent planning and acting. Specifically, our framework comprises two core threads:(1) a planning thread driven by a centralized memory system, maintaining synchronization of environmental states and agent communication to support dynamic decision-making; and (2) an acting thread equipped with a comprehensive skill library, enabling automated task execution through recursive decomposition. Extensive experiments on challenging Minecraft demonstrate the effectiveness of the proposed framework. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的多智能体系统（MAS）的近期进展展现了解决复杂决策任务的显著潜力。然而，现有框架不可避免地依赖于串行执行范式，智能体必须完成序列化的LLM规划后再采取行动。这一根本性约束严重限制了实时响应能力和适应性，这对于充满不断变化场景的动态环境至关重要。本文提出了一种新颖的并行规划-行动框架，具备中断式执行的双线程架构，以实现并发规划和行动。具体而言，该框架包括两个核心线程：(1) 由集中式记忆系统驱动的规划线程，维护环境状态和代理通信的同步，以支持动态决策；(2) 配备全面技能库的行动线程，通过递归分解实现自动化任务执行。在具有挑战性的Minecraft实验中，该框架的有效性得到充分验证。 

---
# Unified Mind Model: Reimagining Autonomous Agents in the LLM Era 

**Title (ZH)**: 统一心智模型：在大语言模型时代重构自主代理 

**Authors**: Pengbo Hu, Xiang Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.03459)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable capabilities across domains, tasks, and languages (e.g., ChatGPT and GPT-4), reviving the research of general autonomous agents with human-like cognitive this http URL human-level agents require semantic comprehension and instruction-following capabilities, which exactly fall into the strengths of this http URL there have been several initial attempts to build human-level agents based on LLMs, the theoretical foundation remains a challenging open problem. In this paper, we propose a novel theoretical cognitive architecture, the Unified Mind Model (UMM), which offers guidance to facilitate the rapid creation of autonomous agents with human-level cognitive abilities. Specifically, our UMM starts with the global workspace theory and further leverage LLMs to enable the agent with various cognitive abilities, such as multi-modal perception, planning, reasoning, tool use, learning, memory, reflection and motivation. Building upon UMM, we then develop an agent-building engine, MindOS, which allows users to quickly create domain-/task-specific autonomous agents without any programming effort. 

**Abstract (ZH)**: 大型语言模型（LLMs）最近在各个领域、任务和语言上展示了卓越的能力（例如ChatGPT和GPT-4），重新激发了对类人认知自主代理的研究。这种类人水平的代理需要语义理解和指令遵循能力，而这正是大型语言模型的优势所在。尽管基于LLMs构建类人水平代理的初步尝试已经出现，但其理论基础仍然是一个具有挑战性的开放问题。在本文中，我们提出了一种新的理论认知架构——统一心灵模型（Unified Mind Model, UMM），它为促进快速创建具有类人认知能力的自主代理提供了指导。具体而言，我们的UMM从全局工作空间理论出发，并进一步利用大型语言模型使代理具备多模感知、规划、推理、工具使用、学习、记忆、反思和动机等多种认知能力。基于UMM，我们随后开发了一个代理构建引擎——MindOS，使用户能够无需编写任何代码即可快速创建特定领域的自主代理。 

---
# From Infants to AI: Incorporating Infant-like Learning in Models Boosts Efficiency and Generalization in Learning Social Prediction Tasks 

**Title (ZH)**: 从婴儿到AI：在模型中融入类似婴儿的学习方式可以提升社交预测任务的学习效率和泛化能力 

**Authors**: Shify Treger, Shimon Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2503.03361)  

**Abstract**: Early in development, infants learn a range of useful concepts, which can be challenging from a computational standpoint. This early learning comes together with an initial understanding of aspects of the meaning of concepts, e.g., their implications, causality, and using them to predict likely future events. All this is accomplished in many cases with little or no supervision, and from relatively few examples, compared with current network models. In learning about objects and human-object interactions, early acquired and possibly innate concepts are often used in the process of learning additional, more complex concepts. In the current work, we model how early-acquired concepts are used in the learning of subsequent concepts, and compare the results with standard deep network modeling. We focused in particular on the use of the concepts of animacy and goal attribution in learning to predict future events. We show that the use of early concepts in the learning of new concepts leads to better learning (higher accuracy) and more efficient learning (requiring less data). We further show that this integration of early and new concepts shapes the representation of the concepts acquired by the model. The results show that when the concepts were learned in a human-like manner, the emerging representation was more useful, as measured in terms of generalization to novel data and tasks. On a more general level, the results suggest that there are likely to be basic differences in the conceptual structures acquired by current network models compared to human learning. 

**Abstract (ZH)**: 早期发展过程中，婴儿学习一系列有用的概念，这在计算上颇具挑战性。这些早期的学习伴随着对概念意义方面初步理解的形成，例如其推论、因果关系，以及利用这些概念预测可能的未来事件。很多情况下，这种学习几乎不需要或只需要很少的监督，并且仅需少量示例，与当前的网络模型相比。在学习物体及人与物体的交互时，早期获得的概念（可能还包括先天的概念）通常被用来学习更复杂的概念。在当前的研究中，我们建模了早期获得的概念在学习后续概念过程中的应用，并将结果与标准的深度网络建模进行了比较。我们特别关注了使用生命性和目标归因概念来预测未来事件的学习过程。结果显示，利用早期概念来学习新概念不仅能够提高学习效果（更高的准确性），而且能够更高效地学习（需要较少的数据）。此外，我们还发现这种早期和新概念的整合会影响模型所获得概念的表征方式。研究结果表明，当概念以类似人类的方式学习时，形成的表示方式在泛化到新数据和任务方面的表现更好。在更广泛的意义上，这些结果表明，当前网络模型获得的概念结构与人类学习的概念结构之间可能存在基本差异。 

---
# Leveraging Large Language Models to Develop Heuristics for Emerging Optimization Problems 

**Title (ZH)**: 利用大规模语言模型开发新兴优化问题的启发式方法 

**Authors**: Thomas Bömer, Nico Koltermann, Max Disselnmeyer, Laura Dörr, Anne Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.03350)  

**Abstract**: Combinatorial optimization problems often rely on heuristic algorithms to generate efficient solutions. However, the manual design of heuristics is resource-intensive and constrained by the designer's expertise. Recent advances in artificial intelligence, particularly large language models (LLMs), have demonstrated the potential to automate heuristic generation through evolutionary frameworks. Recent works focus only on well-known combinatorial optimization problems like the traveling salesman problem and online bin packing problem when designing constructive heuristics. This study investigates whether LLMs can effectively generate heuristics for niche, not yet broadly researched optimization problems, using the unit-load pre-marshalling problem as an example case. We propose the Contextual Evolution of Heuristics (CEoH) framework, an extension of the Evolution of Heuristics (EoH) framework, which incorporates problem-specific descriptions to enhance in-context learning during heuristic generation. Through computational experiments, we evaluate CEoH and EoH and compare the results. Results indicate that CEoH enables smaller LLMs to generate high-quality heuristics more consistently and even outperform larger models. Larger models demonstrate robust performance with or without contextualized prompts. The generated heuristics exhibit scalability to diverse instance configurations. 

**Abstract (ZH)**: 组合优化问题常依赖启发式算法生成高效的解。然而，启发式算法的手动设计耗时且受限于设计者的专业知识。近年来，特别是在大规模语言模型（LLMs）方面的进展表明，可以通过进化框架自动化启发式的生成。现有工作在设计构造性启发式时，仅限于处理如旅行商问题和在线 bin 装箱问题等广为人知的组合优化问题。本研究探讨大规模语言模型是否能有效生成针对尚未广泛研究的特例优化问题的启发式算法，以单位负载预整理问题为例。我们提出了一种上下文启发式演化（CEoH）框架，这是启发式演化（EoH）框架的扩展，通过整合特定问题描述来增强启发式生成过程中的上下文学习。通过计算实验，我们评估了CEoH和EoH，并对比了结果。结果表明，CEoH使较小的语言模型能够更一致地生成高质量的启发式算法，甚至在某些情况下超越了更大模型。更大模型在有或没有上下文提示的情况下表现出稳健的性能。生成的启发式算法对不同实例配置具有可扩展性。 

---
# COSINT-Agent: A Knowledge-Driven Multimodal Agent for Chinese Open Source Intelligence 

**Title (ZH)**: COSINT-Agent：一种知识驱动的多模态中文开源情报代理 

**Authors**: Wentao Li, Congcong Wang, Xiaoxiao Cui, Zhi Liu, Wei Guo, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.03215)  

**Abstract**: Open Source Intelligence (OSINT) requires the integration and reasoning of diverse multimodal data, presenting significant challenges in deriving actionable insights. Traditional approaches, including multimodal large language models (MLLMs), often struggle to infer complex contextual relationships or deliver comprehensive intelligence from unstructured data sources. In this paper, we introduce COSINT-Agent, a knowledge-driven multimodal agent tailored to address the challenges of OSINT in the Chinese domain. COSINT-Agent seamlessly integrates the perceptual capabilities of fine-tuned MLLMs with the structured reasoning power of the Entity-Event-Scene Knowledge Graph (EES-KG). Central to COSINT-Agent is the innovative EES-Match framework, which bridges COSINT-MLLM and EES-KG, enabling systematic extraction, reasoning, and contextualization of multimodal insights. This integration facilitates precise entity recognition, event interpretation, and context retrieval, effectively transforming raw multimodal data into actionable intelligence. Extensive experiments validate the superior performance of COSINT-Agent across core OSINT tasks, including entity recognition, EES generation, and context matching. These results underscore its potential as a robust and scalable solution for advancing automated multimodal reasoning and enhancing the effectiveness of OSINT methodologies. 

**Abstract (ZH)**: 基于知识驱动的多模态代理COSINT-Agent：面向中文领域的开源情报挑战解决方案 

---
# L2R: Learning to Reduce Search Space for Generalizable Neural Routing Solver 

**Title (ZH)**: L2R: 学习减小搜索空间以实现更具泛化能力的神经路由求解器 

**Authors**: Changliang Zhou, Xi Lin, Zhenkun Wang, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03137)  

**Abstract**: Constructive neural combinatorial optimization (NCO) has attracted growing research attention due to its ability to solve complex routing problems without relying on handcrafted rules. However, existing NCO methods face significant challenges in generalizing to large-scale problems due to high computational complexity and inefficient capture of structural patterns. To address this issue, we propose a novel learning-based search space reduction method that adaptively selects a small set of promising candidate nodes at each step of the constructive NCO process. Unlike traditional methods that rely on fixed heuristics, our selection model dynamically prioritizes nodes based on learned patterns, significantly reducing the search space while maintaining solution quality. Experimental results demonstrate that our method, trained solely on 100-node instances from uniform distribution, generalizes remarkably well to large-scale Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) instances with up to 1 million nodes from the uniform distribution and over 80K nodes from other distributions. 

**Abstract (ZH)**: 基于学习的Constructive神经组合优化搜索空间缩减方法 

---
# Towards Understanding Multi-Round Large Language Model Reasoning: Approximability, Learnability and Generalizability 

**Title (ZH)**: 理解多轮大型语言模型推理：可近似性、可学习性和泛化能力 

**Authors**: Chenhui Xu, Dancheng Liu, Jiajie Li, Amir Nassereldine, Zhaohui Li, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03128)  

**Abstract**: Recent advancements in cognitive science and multi-round reasoning techniques for Large Language Models (LLMs) suggest that iterative thinking processes improve problem-solving performance in complex tasks. Inspired by this, approaches like Chain-of-Thought, debating, and self-refinement have been applied to auto-regressive LLMs, achieving significant successes in tasks such as mathematical reasoning, commonsense reasoning, and multi-hop question answering. Despite these successes, the theoretical basis for how multi-round reasoning enhances problem-solving abilities remains underexplored. In this work, we investigate the approximation, learnability, and generalization properties of multi-round auto-regressive models. We show that Transformers with finite context windows are universal approximators for steps of Turing-computable functions and can approximate any Turing-computable sequence-to-sequence function through multi-round reasoning. We extend PAC learning to sequence generation and demonstrate that multi-round generation is learnable even when the sequence length exceeds the model's context window. Finally, we examine how generalization error propagates across rounds, and show how the aforementioned approaches can help constrain this error, ensuring outputs stay within an expectation boundary. This work sheds light on the systemic theoretical foundations of multi-round sequence learning and reasoning, emphasizing its role in inference complexity. 

**Abstract (ZH)**: 最近的认知科学进展和多轮推理技术对于大型语言模型（LLMs）表明，迭代思考过程能够提高复杂任务中的问题解决性能。受此启发，诸如Chain-of-Thought、辩论和自我 refinement 等方法已被应用于自回归 LLMs，并在数学推理、常识推理和多跳问答等任务中取得了显著成果。尽管取得了这些成果，但多轮推理如何增强问题解决能力仍缺乏理论基础的探索。在此工作中，我们研究多轮自回归模型的逼近性、可学习性和泛化性质。我们证明，带有有限上下文窗口的Transformer是图灵可计算函数步数的通用逼近器，并可以通过多轮推理逼近任何图灵可计算的序列到序列函数。我们将PAC学习扩展到序列生成，并证明即使序列长度超过模型的上下文窗口，多轮生成也是可学习的。最后，我们研究了泛化误差在各轮之间的传播，并展示了上述方法如何帮助限制这一误差，确保输出保持在期望边界内。这项工作揭示了多轮序列学习和推理的系统理论基础，强调其在推理复杂性中的作用。 

---
# Teaching AI to Handle Exceptions: Supervised Fine-Tuning with Human-Aligned Judgment 

**Title (ZH)**: 教AI处理异常：基于人类对齐判断的监督微调 

**Authors**: Matthew DosSantos DiSorbo, Harang Ju, Sinan Aral  

**Link**: [PDF](https://arxiv.org/pdf/2503.02976)  

**Abstract**: Large language models (LLMs), initially developed for generative AI, are now evolving into agentic AI systems, which make decisions in complex, real-world contexts. Unfortunately, while their generative capabilities are well-documented, their decision-making processes remain poorly understood. This is particularly evident when models are handling exceptions, a critical and challenging aspect of decision-making made relevant by the inherent incompleteness of contracts. Here we demonstrate that LLMs, even ones that excel at reasoning, deviate significantly from human judgments because they adhere strictly to policies, even when such adherence is impractical, suboptimal, or even counterproductive. We then evaluate three approaches to tuning AI agents to handle exceptions: ethical framework prompting, chain-of-thought reasoning, and supervised fine-tuning. We find that while ethical framework prompting fails and chain-of-thought prompting provides only slight improvements, supervised fine-tuning, specifically with human explanations, yields markedly better results. Surprisingly, in our experiments, supervised fine-tuning even enabled models to generalize human-like decision-making to novel scenarios, demonstrating transfer learning of human-aligned decision-making across contexts. Furthermore, fine-tuning with explanations, not just labels, was critical for alignment, suggesting that aligning LLMs with human judgment requires explicit training on how decisions are made, not just which decisions are made. These findings highlight the need to address LLMs' shortcomings in handling exceptions in order to guide the development of agentic AI toward models that can effectively align with human judgment and simultaneously adapt to novel contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）最初为生成型AI开发，现在正在演变为能在一个复杂的真实世界环境中做出决策的代理型AI系统。不幸的是，尽管它们的生成能力已被广泛记录，但其决策过程仍然不为人所理解。特别是在处理异常情况时，这一缺陷尤为明显，这是因为合同的固有不完备性使得决策过程变得关键且具有挑战性。我们证明，即使是在推理方面表现出色的LLMs，也因严格遵循既定政策而与人类判断产生显著差异，即使这种遵循政策是不切实际的、次优的或甚至是有害的。然后，我们评估了三种调校AI代理以处理异常情况的方法：伦理框架提示、逐步思考推理以及有监督微调。我们发现，伦理框架提示方法失败，逐步思考推理方法仅提供微小改进，而使用人类解释的有监督微调方法则取得了明显更好的效果。令人惊讶的是，在我们的实验中，有监督微调甚至使模型能够将类人的决策模式推广到新的场景中，从而展示了在不同背景下人类对齐决策模式的迁移学习。此外，使用解释而非仅标签示例进行微调对于对齐至关重要，这表明将LLMs与人类判断对齐需要显式训练如何做出决策，而不仅仅是做出哪些决策。这些发现突显出需要解决LLMs在处理异常情况方面的不足，以便引导代理型AI模型朝着能够有效与人类判断对齐并同时适应新情境的方向发展。 

---
# LiteWebAgent: The Open-Source Suite for VLM-Based Web-Agent Applications 

**Title (ZH)**: LiteWebAgent: 基于VLM的网络代理应用开源套件 

**Authors**: Danqing Zhang, Balaji Rama, Jingyi Ni, Shiying He, Fu Zhao, Kunyu Chen, Arnold Chen, Junyu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02950)  

**Abstract**: We introduce LiteWebAgent, an open-source suite for VLM-based web agent applications. Our framework addresses a critical gap in the web agent ecosystem with a production-ready solution that combines minimal serverless backend configuration, intuitive user and browser interfaces, and extensible research capabilities in agent planning, memory, and tree search. For the core LiteWebAgent agent framework, we implemented a simple yet effective baseline using recursive function calling, providing with decoupled action generation and action grounding. In addition, we integrate advanced research components such as agent planning, agent workflow memory, and tree search in a modular and extensible manner. We then integrate the LiteWebAgent agent framework with frontend and backend as deployed systems in two formats: (1) a production Vercel-based web application, which provides users with an agent-controlled remote browser, (2) a Chrome extension leveraging LiteWebAgent's API to control an existing Chrome browser via CDP (Chrome DevTools Protocol). The LiteWebAgent framework is available at this https URL, with deployed frontend at this https URL. 

**Abstract (ZH)**: 我们介绍LiteWebAgent——一个基于VLM的web代理应用程序的开源套件。该框架通过结合最少的无服务器后端配置、直观的用户和浏览器界面以及在代理规划、记忆和树搜索方面的可扩展研究能力，解决了web代理生态系统中的关键问题。对于核心的LiteWebAgent代理框架，我们实现了一个简单有效的 baseline，使用递归函数调用，提供了动作生成与动作 grounding 的解耦。此外，我们以模块化和可扩展的方式整合了高级研究组件，如代理规划、代理工作流记忆和树搜索。然后，我们将LiteWebAgent代理框架以两种格式与前端和后端部署系统集成：(1) 基于Vercel的生产级web应用程序，为用户提供代理控制的远程浏览器；(2) 利用LiteWebAgent的API并通过CDP（Chrome DevTools协议）控制现有Chrome浏览器的Chrome扩展。LiteWebAgent框架可在以下链接获取，部署的前端可在此链接访问。 

---
# The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems 

**Title (ZH)**: MASK基准：解开AI系统中诚实与准确的关系 

**Authors**: Richard Ren, Arunim Agarwal, Mantas Mazeika, Cristina Menghini, Robert Vacareanu, Brad Kenstler, Mick Yang, Isabelle Barrass, Alice Gatti, Xuwang Yin, Eduardo Trevino, Matias Geralnik, Adam Khoja, Dean Lee, Summer Yue, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2503.03750)  

**Abstract**: As large language models (LLMs) become more capable and agentic, the requirement for trust in their outputs grows significantly, yet at the same time concerns have been mounting that models may learn to lie in pursuit of their goals. To address these concerns, a body of work has emerged around the notion of "honesty" in LLMs, along with interventions aimed at mitigating deceptive behaviors. However, evaluations of honesty are currently highly limited, with no benchmark combining large scale and applicability to all models. Moreover, many benchmarks claiming to measure honesty in fact simply measure accuracy--the correctness of a model's beliefs--in disguise. In this work, we introduce a large-scale human-collected dataset for measuring honesty directly, allowing us to disentangle accuracy from honesty for the first time. Across a diverse set of LLMs, we find that while larger models obtain higher accuracy on our benchmark, they do not become more honest. Surprisingly, while most frontier LLMs obtain high scores on truthfulness benchmarks, we find a substantial propensity in frontier LLMs to lie when pressured to do so, resulting in low honesty scores on our benchmark. We find that simple methods, such as representation engineering interventions, can improve honesty. These results underscore the growing need for robust evaluations and effective interventions to ensure LLMs remain trustworthy. 

**Abstract (ZH)**: 大型语言模型（LLMs）能力增强的同时，对其输出的信任需求显著增长，但同时也出现了模型可能为了实现目标而学会撒谎的担忧。为应对这些担忧，围绕LLMs“诚实”这一概念的研究不断涌现，并提出了一系列旨在减轻欺骗行为的干预措施。然而，当前关于“诚实”的评估极为有限，缺乏一个结合大规模数据和普遍适用性的基准。此外，许多声称衡量“诚实”的基准实际上只是衡量准确性——即模型信念的正确性。在本研究中，我们引入了一个大规模的人类收集数据集，用于直接测量“诚实”，使我们能够首次剥离准确性与诚实性的关联。在涵盖多种类型的LLMs中，我们发现，虽然更大模型在我们的基准上获得了更高的准确性，但并没有变得更诚实。令人惊讶的是，尽管大多数前沿的LLMs在真实性基准上获得了高分，但在压力下撒谎的倾向仍然显著，导致其在我们的基准上的诚实性评分较低。我们发现，简单的干预措施，如表示工程干预，可以提高“诚实”。这些结果突显了对可靠评估和有效干预日益增长的需求，以确保LLMs保持可信。 

---
# Process-based Self-Rewarding Language Models 

**Title (ZH)**: 基于过程的自我奖励语言模型 

**Authors**: Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03746)  

**Abstract**: Large Language Models have demonstrated outstanding performance across various downstream tasks and have been widely applied in multiple scenarios. Human-annotated preference data is used for training to further improve LLMs' performance, which is constrained by the upper limit of human performance. Therefore, Self-Rewarding method has been proposed, where LLMs generate training data by rewarding their own outputs. However, the existing self-rewarding paradigm is not effective in mathematical reasoning scenarios and may even lead to a decline in performance. In this work, we propose the Process-based Self-Rewarding pipeline for language models, which introduces long-thought reasoning, step-wise LLM-as-a-Judge, and step-wise preference optimization within the self-rewarding paradigm. Our new paradigm successfully enhances the performance of LLMs on multiple mathematical reasoning benchmarks through iterative Process-based Self-Rewarding, demonstrating the immense potential of self-rewarding to achieve LLM reasoning that may surpass human capabilities. 

**Abstract (ZH)**: 基于过程的自我奖励语言模型pipeline在数学推理场景中的应用 

---
# Rethinking Deep Clustering Paradigms: Self-Supervision Is All You Need 

**Title (ZH)**: 重新思考深度聚类范式：自我监督即所有所需 

**Authors**: Amal Shaheena, Nairouz Mrabahb, Riadh Ksantinia, Abdulla Alqaddoumia  

**Link**: [PDF](https://arxiv.org/pdf/2503.03733)  

**Abstract**: The recent advances in deep clustering have been made possible by significant progress in self-supervised and pseudo-supervised learning. However, the trade-off between self-supervision and pseudo-supervision can give rise to three primary issues. The joint training causes Feature Randomness and Feature Drift, whereas the independent training causes Feature Randomness and Feature Twist. In essence, using pseudo-labels generates random and unreliable features. The combination of pseudo-supervision and self-supervision drifts the reliable clustering-oriented features. Moreover, moving from self-supervision to pseudo-supervision can twist the curved latent manifolds. This paper addresses the limitations of existing deep clustering paradigms concerning Feature Randomness, Feature Drift, and Feature Twist. We propose a new paradigm with a new strategy that replaces pseudo-supervision with a second round of self-supervision training. The new strategy makes the transition between instance-level self-supervision and neighborhood-level self-supervision smoother and less abrupt. Moreover, it prevents the drifting effect that is caused by the strong competition between instance-level self-supervision and clustering-level pseudo-supervision. Moreover, the absence of the pseudo-supervision prevents the risk of generating random features. With this novel approach, our paper introduces a Rethinking of the Deep Clustering Paradigms, denoted by R-DC. Our model is specifically designed to address three primary challenges encountered in Deep Clustering: Feature Randomness, Feature Drift, and Feature Twist. Experimental results conducted on six datasets have shown that the two-level self-supervision training yields substantial improvements. 

**Abstract (ZH)**: 最近深聚类的进展得益于自我监督和伪监督学习的显著进步。然而，自我监督与伪监督之间的权衡可能导致三个主要问题。联合训练导致特征随机性和特征漂移，而独立训练导致特征随机性和特征扭曲。本质上，使用伪标签会生成随机且不可靠的特征。自我监督与伪监督的结合会使可靠的聚类导向特征发生漂移。此外，从自我监督转向伪监督会使曲面潜在流形发生扭曲。本文针对现有深聚类范式的特征随机性、特征漂移和特征扭曲的局限性进行了探讨。我们提出了一种新的范式和策略，用第二轮自我监督训练替代伪监督。这种新策略使得实例级自我监督与邻域级自我监督之间的过渡更加平滑和不那么突兀。此外，它还防止了实例级自我监督与聚类级伪监督之间强烈竞争导致的漂移效应。此外，缺乏伪监督可以防止生成随机特征的风险。通过这一新颖的方法，本文为深聚类范式提出了重新思考的方法，命名为R-DC。我们的模型专门设计用于解决深聚类中遇到的三个主要挑战：特征随机性、特征漂移和特征扭曲。在六个数据集上的实验结果表明，两层次自我监督训练带来了显著的改进。 

---
# Deep Causal Behavioral Policy Learning: Applications to Healthcare 

**Title (ZH)**: 深度因果行为策略学习：在医疗保健领域的应用 

**Authors**: Jonas Knecht, Anna Zink, Jonathan Kolstad, Maya Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2503.03724)  

**Abstract**: We present a deep learning-based approach to studying dynamic clinical behavioral regimes in diverse non-randomized healthcare settings. Our proposed methodology - deep causal behavioral policy learning (DC-BPL) - uses deep learning algorithms to learn the distribution of high-dimensional clinical action paths, and identifies the causal link between these action paths and patient outcomes. Specifically, our approach: (1) identifies the causal effects of provider assignment on clinical outcomes; (2) learns the distribution of clinical actions a given provider would take given evolving patient information; (3) and combines these steps to identify the optimal provider for a given patient type and emulate that provider's care decisions. Underlying this strategy, we train a large clinical behavioral model (LCBM) on electronic health records data using a transformer architecture, and demonstrate its ability to estimate clinical behavioral policies. We propose a novel interpretation of a behavioral policy learned using the LCBM: that it is an efficient encoding of complex, often implicit, knowledge used to treat a patient. This allows us to learn a space of policies that are critical to a wide range of healthcare applications, in which the vast majority of clinical knowledge is acquired tacitly through years of practice and only a tiny fraction of information relevant to patient care is written down (e.g. in textbooks, studies or standardized guidelines). 

**Abstract (ZH)**: 基于深度学习的动态临床行为规制研究：非随机化医疗保健环境中的深度因果行为策略学习 

---
# Rethinking Video Tokenization: A Conditioned Diffusion-based Approach 

**Title (ZH)**: 重思视频分词：一种条件扩散基于的方法 

**Authors**: Nianzu Yang, Pandeng Li, Liming Zhao, Yang Li, Chen-Wei Xie, Yehui Tang, Xudong Lu, Zhihang Liu, Yun Zheng, Yu Liu, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.03708)  

**Abstract**: Video tokenizers, which transform videos into compact latent representations, are key to video generation. Existing video tokenizers are based on the VAE architecture and follow a paradigm where an encoder compresses videos into compact latents, and a deterministic decoder reconstructs the original videos from these latents. In this paper, we propose a novel \underline{\textbf{C}}onditioned \underline{\textbf{D}}iffusion-based video \underline{\textbf{T}}okenizer entitled \textbf{\ourmethod}, which departs from previous methods by replacing the deterministic decoder with a 3D causal diffusion model. The reverse diffusion generative process of the decoder is conditioned on the latent representations derived via the encoder. With a feature caching and sampling acceleration, the framework efficiently reconstructs high-fidelity videos of arbitrary lengths. Results show that {\ourmethod} achieves state-of-the-art performance in video reconstruction tasks using just a single-step sampling. Even a smaller version of {\ourmethod} still achieves reconstruction results on par with the top two baselines. Furthermore, the latent video generation model trained using {\ourmethod} also shows superior performance. 

**Abstract (ZH)**: 条件因果扩散驱动的视频分词器：\textbf{\ourmethod} 

---
# Curating Demonstrations using Online Experience 

**Title (ZH)**: 基于在线体验策展演示 

**Authors**: Annie S. Chen, Alec M. Lessing, Yuejiang Liu, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2503.03707)  

**Abstract**: Many robot demonstration datasets contain heterogeneous demonstrations of varying quality. This heterogeneity may benefit policy pre-training, but can hinder robot performance when used with a final imitation learning objective. In particular, some strategies in the data may be less reliable than others or may be underrepresented in the data, leading to poor performance when such strategies are sampled at test time. Moreover, such unreliable or underrepresented strategies can be difficult even for people to discern, and sifting through demonstration datasets is time-consuming and costly. On the other hand, policy performance when trained on such demonstrations can reflect the reliability of different strategies. We thus propose for robots to self-curate based on online robot experience (Demo-SCORE). More specifically, we train and cross-validate a classifier to discern successful policy roll-outs from unsuccessful ones and use the classifier to filter heterogeneous demonstration datasets. Our experiments in simulation and the real world show that Demo-SCORE can effectively identify suboptimal demonstrations without manual curation. Notably, Demo-SCORE achieves over 15-35% higher absolute success rate in the resulting policy compared to the base policy trained with all original demonstrations. 

**Abstract (ZH)**: 基于在线机器人体验的演示自筛选（Demo-SCORE） 

---
# Attentive Reasoning Queries: A Systematic Method for Optimizing Instruction-Following in Large Language Models 

**Title (ZH)**: 注意力推理查询：优化大型语言模型遵循指令的一种系统方法 

**Authors**: Bar Karov, Dor Zohar, Yam Marcovitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.03669)  

**Abstract**: We present Attentive Reasoning Queries (ARQs), a novel structured reasoning approach that significantly improves instruction-following in Large Language Models through domain-specialized reasoning blueprints. While LLMs demonstrate remarkable capabilities across diverse tasks, they often fail to maintain adherence to complex, use-case-specific instructions during multi-turn conversations, presenting challenges for business-critical applications. ARQs address this limitation by guiding LLMs through systematic reasoning steps with targeted queries that reinstate critical instructions and facilitate intermediate reasoning throughout the completion process. In extensive testing within Parlant, our framework for reliable customer-facing agents in which ARQs were born out of necessity, they achieved a 90.2% success rate across 87 test scenarios, outperforming both Chain-of-Thought reasoning (86.1%) and direct response generation (81.5%). ARQs showed particular strength in addressing persistent failure modes like guideline re-application and hallucination prevention. Our analysis also revealed that ARQs can potentially be more computationally efficient than free-form reasoning when carefully designed. These findings demonstrate that structured reasoning approaches provide effective mechanisms for controlling how LLMs process information and make decisions in complex scenarios. 

**Abstract (ZH)**: 注意推理查询：一种通过领域特种推理蓝图显著提高大型语言模型指令遵循能力的新型结构化推理方法 

---
# A Generative Approach to High Fidelity 3D Reconstruction from Text Data 

**Title (ZH)**: 基于文本数据的高保真3D重建生成方法 

**Authors**: Venkat Kumar R, Deepak Saravanan  

**Link**: [PDF](https://arxiv.org/pdf/2503.03664)  

**Abstract**: The convergence of generative artificial intelligence and advanced computer vision technologies introduces a groundbreaking approach to transforming textual descriptions into three-dimensional representations. This research proposes a fully automated pipeline that seamlessly integrates text-to-image generation, various image processing techniques, and deep learning methods for reflection removal and 3D reconstruction. By leveraging state-of-the-art generative models like Stable Diffusion, the methodology translates natural language inputs into detailed 3D models through a multi-stage workflow.
The reconstruction process begins with the generation of high-quality images from textual prompts, followed by enhancement by a reinforcement learning agent and reflection removal using the Stable Delight model. Advanced image upscaling and background removal techniques are then applied to further enhance visual fidelity. These refined two-dimensional representations are subsequently transformed into volumetric 3D models using sophisticated machine learning algorithms, capturing intricate spatial relationships and geometric characteristics. This process achieves a highly structured and detailed output, ensuring that the final 3D models reflect both semantic accuracy and geometric precision.
This approach addresses key challenges in generative reconstruction, such as maintaining semantic coherence, managing geometric complexity, and preserving detailed visual information. Comprehensive experimental evaluations will assess reconstruction quality, semantic accuracy, and geometric fidelity across diverse domains and varying levels of complexity. By demonstrating the potential of AI-driven 3D reconstruction techniques, this research offers significant implications for fields such as augmented reality (AR), virtual reality (VR), and digital content creation. 

**Abstract (ZH)**: 生成式人工智能与高级计算机视觉技术的融合引入了将文本描述转换为三维表示的革命性方法。本研究提出了一种全自动流程，无缝集成文本到图像生成、各种图像处理技术和深度学习方法进行反射去除和三维重建。通过利用如Stable Diffusion等最先进的生成模型，该方法通过多阶段工作流将自然语言输入转换为详细的三维模型。
重建过程始于从文本提示生成高质量图像，随后使用强化学习代理进行增强，并利用Stable Delight模型去除反射。然后应用高级图像超分辨率和背景移除技术进一步提升视觉保真度。这些精细的二维表示随后通过复杂的机器学习算法转换为体素化的三维模型，捕捉复杂的空间关系和几何特征。这一过程实现了高度结构化的详细输出，确保最终的三维模型既具有语义准确性又具有几何精度。
该方法解决了生成重建中的关键挑战，如保持语义一致性、管理几何复杂性以及保留详细的视觉信息。全面的实验评估将从不同领域和复杂度差异评估重建质量、语义准确性和几何保真度。通过展示基于AI的三维重建技术的潜力，本研究为增强现实（AR）、虚拟现实（VR）和数字内容创作等领域提供了重大影响。 

---
# Improving 6D Object Pose Estimation of metallic Household and Industry Objects 

**Title (ZH)**: 改进金属家居和工业物体的6D姿态估计 

**Authors**: Thomas Pöllabauer, Michael Gasser, Tristan Wirth, Sarah Berkei, Volker Knauthe, Arjan Kuijper  

**Link**: [PDF](https://arxiv.org/pdf/2503.03655)  

**Abstract**: 6D object pose estimation suffers from reduced accuracy when applied to metallic objects. We set out to improve the state-of-the-art by addressing challenges such as reflections and specular highlights in industrial applications. Our novel BOP-compatible dataset, featuring a diverse set of metallic objects (cans, household, and industrial items) under various lighting and background conditions, provides additional geometric and visual cues. We demonstrate that these cues can be effectively leveraged to enhance overall performance. To illustrate the usefulness of the additional features, we improve upon the GDRNPP algorithm by introducing an additional keypoint prediction and material estimator head in order to improve spatial scene understanding. Evaluations on the new dataset show improved accuracy for metallic objects, supporting the hypothesis that additional geometric and visual cues can improve learning. 

**Abstract (ZH)**: 6D物体姿态估计在应用于金属物体时 accuracies 降低。为解决工业应用中反射和镜面高光等问题，我们提出了一种改进的方案。我们开发了一种与BOP兼容的新数据集，该数据集包含在不同光照和背景条件下的多种金属物体（罐类、家用和工业用品），提供了额外的几何和视觉线索。我们证明这些线索可以有效提升整体性能。为了展示额外特征的重要性，我们通过对GDRNPP算法进行改进，引入额外的关键点预测和材料估计模块，以提高空间场景理解能力。在新数据集上的评估显示，金属物体的准确率有所提高，支持了额外几何和视觉线索有助于学习的假设。 

---
# Improving Neutral Point of View Text Generation through Parameter-Efficient Reinforcement Learning and a Small-Scale High-Quality Dataset 

**Title (ZH)**: 通过参数高效强化学习和小型高质数据集改进中立观点文本生成 

**Authors**: Jessica Hoffmann, Christiane Ahlheim, Zac Yu, Aria Walfrand, Jarvis Jin, Marie Tano, Ahmad Beirami, Erin van Liemt, Nithum Thain, Hakim Sidahmed, Lucas Dixon  

**Link**: [PDF](https://arxiv.org/pdf/2503.03654)  

**Abstract**: This paper describes the construction of a dataset and the evaluation of training methods to improve generative large language models' (LLMs) ability to answer queries on sensitive topics with a Neutral Point of View (NPOV), i.e., to provide significantly more informative, diverse and impartial answers. The dataset, the SHQ-NPOV dataset, comprises 300 high-quality, human-written quadruplets: a query on a sensitive topic, an answer, an NPOV rating, and a set of links to source texts elaborating the various points of view. The first key contribution of this paper is a new methodology to create such datasets through iterative rounds of human peer-critique and annotator training, which we release alongside the dataset. The second key contribution is the identification of a highly effective training regime for parameter-efficient reinforcement learning (PE-RL) to improve NPOV generation. We compare and extensively evaluate PE-RL and multiple baselines-including LoRA finetuning (a strong baseline), SFT and RLHF.
PE-RL not only improves on overall NPOV quality compared to the strongest baseline ($97.06\%\rightarrow 99.08\%$), but also scores much higher on features linguists identify as key to separating good answers from the best answers ($60.25\%\rightarrow 85.21\%$ for presence of supportive details, $68.74\%\rightarrow 91.43\%$ for absence of oversimplification). A qualitative analysis corroborates this. Finally, our evaluation finds no statistical differences between results on topics that appear in the training dataset and those on separated evaluation topics, which provides strong evidence that our approach to training PE-RL exhibits very effective out of topic generalization. 

**Abstract (ZH)**: 一种用于提高大型语言模型在敏感话题上以中立视角回答查询能力的数据集构建与训练方法评价 

---
# Decoupled Recommender Systems: Exploring Alternative Recommender Ecosystem Designs 

**Title (ZH)**: 解耦推荐系统：探索替代的推荐生态系统设计 

**Authors**: Anas Buhayh, Elizabeth McKinnie, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2503.03606)  

**Abstract**: Recommender ecosystems are an emerging subject of research. Such research examines how the characteristics of algorithms, recommendation consumers, and item providers influence system dynamics and long-term outcomes. One architectural possibility that has not yet been widely explored in this line of research is the consequences of a configuration in which recommendation algorithms are decoupled from the platforms they serve. This is sometimes called "the friendly neighborhood algorithm store" or "middleware" model. We are particularly interested in how such architectures might offer a range of different distributions of utility across consumers, providers, and recommendation platforms. In this paper, we create a model of a recommendation ecosystem that incorporates algorithm choice and examine the outcomes of such a design. 

**Abstract (ZH)**: 推荐生态系统是一种新兴的研究主题。此类研究探讨算法特性、推荐消费者和项目提供者的特点如何影响系统动力学和长期结果。在这条研究线路上，尚未广泛探讨的一种架构可能性是推荐算法与服务的平台脱钩的后果。有时这种架构被称为“友好邻里的算法存储”或“中间件”模型。我们特别感兴趣的是这种架构如何为消费者、提供者和推荐平台提供不同类型的利益分配。在本文中，我们构建了一个包含算法选择的推荐生态系统模型，并探讨了此类设计的结果。 

---
# Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias 

**Title (ZH)**: 面向理解扩散模型的文本幻想现象通过局部生成偏差 

**Authors**: Rui Lu, Runzhe Wang, Kaifeng Lyu, Xitai Jiang, Gao Huang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03595)  

**Abstract**: Score-based diffusion models have achieved incredible performance in generating realistic images, audio, and video data. While these models produce high-quality samples with impressive details, they often introduce unrealistic artifacts, such as distorted fingers or hallucinated texts with no meaning. This paper focuses on textual hallucinations, where diffusion models correctly generate individual symbols but assemble them in a nonsensical manner. Through experimental probing, we consistently observe that such phenomenon is attributed it to the network's local generation bias. Denoising networks tend to produce outputs that rely heavily on highly correlated local regions, particularly when different dimensions of the data distribution are nearly pairwise independent. This behavior leads to a generation process that decomposes the global distribution into separate, independent distributions for each symbol, ultimately failing to capture the global structure, including underlying grammar. Intriguingly, this bias persists across various denoising network architectures including MLP and transformers which have the structure to model global dependency. These findings also provide insights into understanding other types of hallucinations, extending beyond text, as a result of implicit biases in the denoising models. Additionally, we theoretically analyze the training dynamics for a specific case involving a two-layer MLP learning parity points on a hypercube, offering an explanation of its underlying mechanism. 

**Abstract (ZH)**: 评分based扩散模型在生成逼真图像、音频和视频数据方面取得了惊人的性能。尽管这些模型能够生成高质量、细节逼真的样本，但往往会引入一些不现实的伪影，如扭曲的手指或无意义的文本。本文关注文本幻觉现象，扩散模型可以生成正确的符号，但它们以一种无意义的方式组合。通过实验探究，我们一致观察到这种现象是由网络的局部生成偏见引起的。去噪网络倾向于产生高度依赖于局部高度相关区域的输出，尤其是在数据分布的不同维度几乎呈 pairwise 独立时。这种行为导致生成过程将全局分布分解为各个符号的独立分布，最终未能捕捉到全局结构，包括潜在的语法规则。有趣的是，这种偏见在包括MLP和变压器在内的各种去噪网络架构中普遍存在，尽管它们具有建模全局依赖性的结构。这些发现还为理解其他类型的幻觉提供了见解，这些幻觉超越了文本，源于去噪模型中的隐式偏见。此外，我们对一个两层MLP在超立方体上学习奇偶校验点的具体情况进行理论分析，提供了其内在机制的解释。 

---
# Small but Mighty: Enhancing Time Series Forecasting with Lightweight LLMs 

**Title (ZH)**: 小巧而强大：轻量级语言模型增强时间序列预测 

**Authors**: Haoran Fan, Bin Li, Yixuan Weng, Shoujun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.03594)  

**Abstract**: While LLMs have demonstrated remarkable potential in time series forecasting, their practical deployment remains constrained by excessive computational demands and memory footprints. Existing LLM-based approaches typically suffer from three critical limitations: Inefficient parameter utilization in handling numerical time series patterns; Modality misalignment between continuous temporal signals and discrete text embeddings; and Inflexibility for real-time expert knowledge integration. We present SMETimes, the first systematic investigation of sub-3B parameter SLMs for efficient and accurate time series forecasting. Our approach centers on three key innovations: A statistically-enhanced prompting mechanism that bridges numerical time series with textual semantics through descriptive statistical features; A adaptive fusion embedding architecture that aligns temporal patterns with language model token spaces through learnable parameters; And a dynamic mixture-of-experts framework enabled by SLMs' computational efficiency, adaptively combining base predictions with domain-specific models. Extensive evaluations across seven benchmark datasets demonstrate that our 3B-parameter SLM achieves state-of-the-art performance on five primary datasets while maintaining 3.8x faster training and 5.2x lower memory consumption compared to 7B-parameter LLM baselines. Notably, the proposed model exhibits better learning capabilities, achieving 12.3% lower MSE than conventional LLM. Ablation studies validate that our statistical prompting and cross-modal fusion modules respectively contribute 15.7% and 18.2% error reduction in long-horizon forecasting tasks. By redefining the efficiency-accuracy trade-off landscape, this work establishes SLMs as viable alternatives to resource-intensive LLMs for practical time series forecasting. Code and models are available at this https URL. 

**Abstract (ZH)**: 虽然大的语言模型在时间序列预测方面展现了显著的潜力，但其实用部署仍受限于过高的计算需求和内存占用。现有的基于大语言模型的方法通常面临三个关键限制：在处理数值时间序列模式时参数利用效率低下；连续时间信号与离散文本嵌入之间的模态不对齐；以及对实时专家知识整合的灵活性不足。我们提出了SMETimes，这是对参数量小于3B的可解释小语言模型进行系统研究的第一个尝试，旨在实现高效且准确的时间序列预测。我们的方法侧重于三个方面的重要创新：通过描述性统计特征增强的统计提示机制，实现数值时间序列与文本语义的连接；自适应融合嵌入架构，通过可学习参数使时间模式与语言模型词元空间相匹配；以及基于SLMs计算效率的动态专家混合框架，能够灵活结合基础预测和领域特定模型。在跨越七个基准数据集的广泛评估中，我们的3B参数SLM在五个主要数据集上达到了最先进的性能，同时相比7B参数大语言模型基线，训练速度提升3.8倍，并且内存消耗降低了5.2倍。值得注意的是，所提模型表现出更好的学习能力，MSE降低了12.3%。消融实验验证了我们的统计提示和跨模态融合模块分别在长时预测任务中贡献了15.7%和18.2%的误差减少。通过重新定义效率与准确性之间的权衡范围，这项工作确立了SLMs作为一种资源密集型大语言模型的实用替代品在实际时间序列预测中的可行性。代码和模型可在以下链接获取。 

---
# English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance 

**Title (ZH)**: 多语言性能不受K量化LMs不当减损 

**Authors**: Karl Audun Borgersen  

**Link**: [PDF](https://arxiv.org/pdf/2503.03592)  

**Abstract**: For consumer usage of locally deployed LLMs, the GGUF format and k_quantization are invaluable tools for maintaining the performance of the original model while reducing it to sizes deployable with consumer-grade hardware. The number of bits dedicated to each weight from the original model is reduced based on how important they are thought to be during model inference. This importance is arrived at through the application of an 'importance matrix'-a relatively small text document meant to be representative of the LLM's standard use-cases. In the vast majority of quants available online, this document is primarily written in English. It was therefore an open question whether performance on English language tasks was preserved through the sacrifice of multilingual performance and whether it can be preserved with alternate importance matrices. This article investigates these hypotheses by quantizing Llama3.3 70B on importance matrices written in three languages (English, Norwegian, and Malayalam) and evaluating them on the MixEval dataset in both English and Norwegian. All experiments related to k_quantization yielded non-significant results (In all cases p > 0.237) indicating that current quantization practices do not disproportionately harm multilingual performance. 

**Abstract (ZH)**: 针对本地部署LLM的消费者使用，GGUF格式和k量化的工具对于在减小程序大小的同时保持原始模型性能具有重要作用。通过“重要性矩阵”确定每个权重分配的位数，该矩阵是一个相对较小的文本文件，旨在代表LLM的标准使用场景。目前，在线提供的大多数量化文档主要是用英语撰写的。因此，一个开放的问题是，通过牺牲多语言性能来保留英语语言任务的性能是否可行，以及是否可以通过使用不同语言编写的重要性矩阵来保留多语言性能。本文通过使用三种语言（英语、挪威语和马拉雅拉姆语）撰写的“重要性矩阵”对Llama3.3 70B进行量化，并在英语和挪威语上使用MixEval数据集进行评估，结果表明，当前的量化实践并未对多语言性能造成不成比例的损害。 

---
# A Conceptual Model for Attributions in Event-Centric Knowledge Graphs 

**Title (ZH)**: 事件中心知识图谱中归因的概念模型 

**Authors**: Florian Plötzky, Katarina Britz, Wolf-Tilo Balke  

**Link**: [PDF](https://arxiv.org/pdf/2503.03563)  

**Abstract**: The use of narratives as a means of fusing information from knowledge graphs (KGs) into a coherent line of argumentation has been the subject of recent investigation. Narratives are especially useful in event-centric knowledge graphs in that they provide a means to connect different real-world events and categorize them by well-known narrations. However, specifically for controversial events, a problem in information fusion arises, namely, multiple viewpoints regarding the validity of certain event aspects, e.g., regarding the role a participant takes in an event, may exist. Expressing those viewpoints in KGs is challenging because disputed information provided by different viewpoints may introduce inconsistencies. Hence, most KGs only feature a single view on the contained information, hampering the effectiveness of narrative information access. This paper is an extension of our original work and introduces attributions, i.e., parameterized predicates that allow for the representation of facts that are only valid in a specific viewpoint. For this, we develop a conceptual model that allows for the representation of viewpoint-dependent information. As an extension, we enhance the model by a conception of viewpoint-compatibility. Based on this, we deepen our original deliberations on the model's effects on information fusion and provide additional grounding in the literature. 

**Abstract (ZH)**: 利用叙事将知识图谱信息融合到连贯的论据链中的研究：引入视点归因以处理争议性事件中的个人观点冲突 

---
# Towards Visual Discrimination and Reasoning of Real-World Physical Dynamics: Physics-Grounded Anomaly Detection 

**Title (ZH)**: 面向现实世界物理动力学的视觉辨识与推理：基于物理原理的异常检测 

**Authors**: Wenqiao Li, Yao Gu, Xintao Chen, Xiaohao Xu, Ming Hu, Xiaonan Huang, Yingna Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03562)  

**Abstract**: Humans detect real-world object anomalies by perceiving, interacting, and reasoning based on object-conditioned physical knowledge. The long-term goal of Industrial Anomaly Detection (IAD) is to enable machines to autonomously replicate this skill. However, current IAD algorithms are largely developed and tested on static, semantically simple datasets, which diverge from real-world scenarios where physical understanding and reasoning are this http URL bridge this gap, we introduce the Physics Anomaly Detection (Phys-AD) dataset, the first large-scale, real-world, physics-grounded video dataset for industrial anomaly detection. Collected using a real robot arm and motor, Phys-AD provides a diverse set of dynamic, semantically rich scenarios. The dataset includes more than 6400 videos across 22 real-world object categories, interacting with robot arms and motors, and exhibits 47 types of anomalies. Anomaly detection in Phys-AD requires visual reasoning, combining both physical knowledge and video content to determine object this http URL benchmark state-of-the-art anomaly detection methods under three settings: unsupervised AD, weakly-supervised AD, and video-understanding AD, highlighting their limitations in handling physics-grounded anomalies. Additionally, we introduce the Physics Anomaly Explanation (PAEval) metric, designed to assess the ability of visual-language foundation models to not only detect anomalies but also provide accurate explanations for their underlying physical causes. Our dataset and benchmark will be publicly available. 

**Abstract (ZH)**: 人类通过感知、交互和基于对象条件物理知识的推理来检测现实世界的物体异常。工业异常检测（IAD）的长期目标是使机器能够自主复制这一技能。然而，当前的IAD算法主要是在静态、语义简单的数据集上开发和测试的，与实际场景中的物理理解和推理相去甚远。为了缩小这一差距，我们介绍了首个面向工业异常检测的大规模、真实世界、物理导向的视频数据集——Physics Anomaly Detection（Phys-AD）数据集。该数据集使用真实机器人手臂和电机采集，提供了多种动态且语义丰富的场景。数据集包含了超过6400个视频，覆盖22个真实的物体类别，并展示了47种类型的异常。在Phys-AD中进行异常检测需要视觉推理，结合物理知识和视频内容来确定物体的状态。我们以三种设置——无监督异常检测、弱监督异常检测和视频理解异常检测——来评估最先进的异常检测方法，并突显它们在处理物理导向的异常时的局限性。此外，我们引入了Physics Anomaly Explanation（PAEval）度量标准，旨在评估视觉语言基础模型不仅能够检测异常，还能提供其物理原因的准确解释的能力。我们的数据集和基准将会公开。 

---
# AI-Enabled Conversational Journaling for Advancing Parkinson's Disease Symptom Tracking 

**Title (ZH)**: AI驱动的对话式日记记录方法在帕金森病症状跟踪中的应用 

**Authors**: Mashrur Rashik, Shilpa Sweth, Nishtha Agrawal, Saiyyam Kochar, Kara M Smith, Fateme Rajabiyazdi, Vidya Setlur, Narges Mahyar, Ali Sarvghad  

**Link**: [PDF](https://arxiv.org/pdf/2503.03532)  

**Abstract**: Journaling plays a crucial role in managing chronic conditions by allowing patients to document symptoms and medication intake, providing essential data for long-term care. While valuable, traditional journaling methods often rely on static, self-directed entries, lacking interactive feedback and real-time guidance. This gap can result in incomplete or imprecise information, limiting its usefulness for effective treatment. To address this gap, we introduce PATRIKA, an AI-enabled prototype designed specifically for people with Parkinson's disease (PwPD). The system incorporates cooperative conversation principles, clinical interview simulations, and personalization to create a more effective and user-friendly journaling experience. Through two user studies with PwPD and iterative refinement of PATRIKA, we demonstrate conversational journaling's significant potential in patient engagement and collecting clinically valuable information. Our results showed that generating probing questions PATRIKA turned journaling into a bi-directional interaction. Additionally, we offer insights for designing journaling systems for healthcare and future directions for promoting sustained journaling. 

**Abstract (ZH)**: 延还认证在管理慢性疾病中发挥关键作用，通过允许患者记录症状和药物摄入，提供长期护理所需的重要数据。尽管传统日志记录方法有价值，但往往依赖于静态、自我驱动的条目，缺乏互动反馈和实时指导。为填补这一空白，我们介绍了PATRIKA，一种针对帕金森病患者（PwPD）的AI辅助原型系统。该系统整合了合作对话原则、临床访谈模拟和个人化设计，以创建更有效和用户友好的日志记录体验。通过与PwPD患者的两组用户研究和对PATRIKA的迭代优化，我们展示了对话式日志记录在患者参与和收集临床有价值信息方面的巨大潜力。研究结果显示，生成探询性问题使PATRIKA将日志记录转变为此消彼长的互动过程。此外，我们还提供了为医疗保健设计日志记录系统的设计见解，并提出了促进持续日志记录的未来方向。 

---
# AdaSin: Enhancing Hard Sample Metrics with Dual Adaptive Penalty for Face Recognition 

**Title (ZH)**: AdaSin: 通过双适应惩罚增强难样本指标的面部识别 

**Authors**: Qiqi Guo, Zhuowen Zheng, Guanghua Yang, Zhiquan Liu, Xiaofan Li, Jianqing Li, Jinyu Tian, Xueyuan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03528)  

**Abstract**: In recent years, the emergence of deep convolutional neural networks has positioned face recognition as a prominent research focus in computer vision. Traditional loss functions, such as margin-based, hard-sample mining-based, and hybrid approaches, have achieved notable performance improvements, with some leveraging curriculum learning to optimize training. However, these methods often fall short in effectively quantifying the difficulty of hard samples. To address this, we propose Adaptive Sine (AdaSin) loss function, which introduces the sine of the angle between a sample's embedding feature and its ground-truth class center as a novel difficulty metric. This metric enables precise and effective penalization of hard samples. By incorporating curriculum learning, the model dynamically adjusts classification boundaries across different training stages. Unlike previous adaptive-margin loss functions, AdaSin introduce a dual adaptive penalty, applied to both the positive and negative cosine similarities of hard samples. This design imposes stronger constraints, enhancing intra-class compactness and inter-class separability. The combination of the dual adaptive penalty and curriculum learning is guided by a well-designed difficulty metric. It enables the model to focus more effectively on hard samples in later training stages, and lead to the extraction of highly discriminative face features. Extensive experiments across eight benchmarks demonstrate that AdaSin achieves superior accuracy compared to other state-of-the-art methods. 

**Abstract (ZH)**: 自适应正弦损失函数在面部识别中的应用：动态适应与困难样本衡量 

---
# NeuGrasp: Generalizable Neural Surface Reconstruction with Background Priors for Material-Agnostic Object Grasp Detection 

**Title (ZH)**: NeuGrasp: 基于背景先验的通用神经表面重建及其在材料无关物体抓取检测中的应用 

**Authors**: Qingyu Fan, Yinghao Cai, Chao Li, Wenzhe He, Xudong Zheng, Tao Lu, Bin Liang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03511)  

**Abstract**: Robotic grasping in scenes with transparent and specular objects presents great challenges for methods relying on accurate depth information. In this paper, we introduce NeuGrasp, a neural surface reconstruction method that leverages background priors for material-agnostic grasp detection. NeuGrasp integrates transformers and global prior volumes to aggregate multi-view features with spatial encoding, enabling robust surface reconstruction in narrow and sparse viewing conditions. By focusing on foreground objects through residual feature enhancement and refining spatial perception with an occupancy-prior volume, NeuGrasp excels in handling objects with transparent and specular surfaces. Extensive experiments in both simulated and real-world scenarios show that NeuGrasp outperforms state-of-the-art methods in grasping while maintaining comparable reconstruction quality. More details are available at this https URL. 

**Abstract (ZH)**: 具有透明和镜面物体的场景中的机器人抓取对依赖准确深度信息的方法提出了巨大挑战。本文介绍了一种名为NeuGrasp的神经表面重建方法，该方法利用背景先验进行材料无关的抓取检测。NeuGrasp结合了变换器和全球先验体素，通过空间编码聚合多视图特征，能够在狭窄和稀疏的视角条件下实现稳健的表面重建。通过残差特征增强聚焦前景物体，并借助占用先验体素精化空间感知，NeuGrasp在处理具有透明和镜面表面的物体方面表现出色。在模拟和真实场景中的广泛实验表明，NeuGrasp在抓取性能上优于现有方法，同时保持相当的重建质量。更多详情请参见：这个链接。 

---
# Rethinking Synthetic Data definitions: A privacy driven approach 

**Title (ZH)**: 重塑合成数据的定义：以隐私为导向的方法 

**Authors**: Vibeke Binz Vallevik, Serena Elizabeth Marshall, Aleksandar Babic, Jan Franz Nygaard  

**Link**: [PDF](https://arxiv.org/pdf/2503.03506)  

**Abstract**: Synthetic data is gaining traction as a cost-effective solution for the increasing data demands of AI development and can be generated either from existing knowledge or derived data captured from real-world events. The source of the synthetic data generation and the technique used significantly impacts its residual privacy risk and therefore its opportunity for sharing. Traditional classification of synthetic data types no longer fit the newer generation techniques and there is a need to better align the classification with practical needs. We suggest a new way of grouping synthetic data types that better supports privacy evaluations to aid regulatory policymaking. Our novel classification provides flexibility to new advancements like deep generative methods and offers a more practical framework for future applications. 

**Abstract (ZH)**: 合成数据作为AI开发中日益增长的数据需求的一种成本-effective解决方案正在受到青睐，并可以通过现有知识或源于实际事件的衍生数据生成。合成数据生成的来源和所使用的技术对其剩余隐私风险以及因此对其共享机会产生了重大影响。传统意义上的合成数据类型分类已不再适用于新的生成技术，因此有必要将分类更好地与实际需求对齐。我们建议一种新的合成数据类型分组方式，以更好地支持隐私评估，从而辅助监管政策制定。我们的新分类提供了对未来进步如深度生成方法的灵活性，并为未来应用提供了一个更加实际的框架。 

---
# Collaborative Expert LLMs Guided Multi-Objective Molecular Optimization 

**Title (ZH)**: 协作专家大语言模型引导的多目标分子优化 

**Authors**: Jiajun Yu, Yizhen Zheng, Huan Yee Koh, Shirui Pan, Tianyue Wang, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03503)  

**Abstract**: Molecular optimization is a crucial yet complex and time-intensive process that often acts as a bottleneck for drug development. Traditional methods rely heavily on trial and error, making multi-objective optimization both time-consuming and resource-intensive. Current AI-based methods have shown limited success in handling multi-objective optimization tasks, hampering their practical utilization. To address this challenge, we present MultiMol, a collaborative large language model (LLM) system designed to guide multi-objective molecular optimization. MultiMol comprises two agents, including a data-driven worker agent and a literature-guided research agent. The data-driven worker agent is a large language model being fine-tuned to learn how to generate optimized molecules considering multiple objectives, while the literature-guided research agent is responsible for searching task-related literature to find useful prior knowledge that facilitates identifying the most promising optimized candidates. In evaluations across six multi-objective optimization tasks, MultiMol significantly outperforms existing methods, achieving a 82.30% success rate, in sharp contrast to the 27.50% success rate of current strongest methods. To further validate its practical impact, we tested MultiMol on two real-world challenges. First, we enhanced the selectivity of Xanthine Amine Congener (XAC), a promiscuous ligand that binds both A1R and A2AR, successfully biasing it towards A1R. Second, we improved the bioavailability of Saquinavir, an HIV-1 protease inhibitor with known bioavailability limitations. Overall, these results indicate that MultiMol represents a highly promising approach for multi-objective molecular optimization, holding great potential to accelerate the drug development process and contribute to the advancement of pharmaceutical research. 

**Abstract (ZH)**: 分子优化是药物开发中至关重要的 yet 复杂和耗时的过程，往往是药物开发中的瓶颈。传统方法依赖于试错，使得多目标优化既耗时又耗资源。当前基于AI的方法在处理多目标优化任务方面取得的成效有限，阻碍了其实际应用。为应对这一挑战，我们提出了MultiMol，一个协作的大语言模型系统，旨在指导多目标分子优化。MultiMol 包含两个代理，包括一个数据驱动的工作者代理和一个文献引导的研究代理。数据驱动的工作者代理是一个正在微调的大语言模型，学习如何生成考虑多个目标的优化分子，而文献引导的研究代理负责搜索相关任务的文献以寻找有用的前提知识，从而有助于识别最有可能的优化候选物。在对六个多目标优化任务的评估中，MultiMol 显著优于现有方法，成功率为82.30%，而当前最强方法的成功率为27.50%。为进一步验证其实际影响，我们在两个实际挑战中测试了MultiMol。首先，我们增强了杂环嘌呤氨酸（XAC）的选择性，这是一种兼具A1R和A2AR结合活性的促混适配体。其次，我们提高了沙奎那韦（一种具有已知生物利用度限制的HIV-1蛋白酶抑制剂）的生物利用度。总体而言，这些结果表明MultiMol 是一个多目标分子优化的极具前景的方法，有望加速药物开发过程并推动制药研究的发展。 

---
# CURVALID: Geometrically-guided Adversarial Prompt Detection 

**Title (ZH)**: CURVALID: 几何引导的对抗提示检测 

**Authors**: Canaan Yung, Hanxun Huang, Sarah Monazam Erfani, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2503.03502)  

**Abstract**: Adversarial prompts capable of jailbreaking large language models (LLMs) and inducing undesirable behaviours pose a significant obstacle to their safe deployment. Current mitigation strategies rely on activating built-in defence mechanisms or fine-tuning the LLMs, but the fundamental distinctions between adversarial and benign prompts are yet to be understood. In this work, we introduce CurvaLID, a novel defense framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. Additionally, we employ Local Intrinsic Dimensionality (LID) to capture geometric features of text prompts within adversarial subspaces. Our findings reveal that adversarial prompts differ fundamentally from benign prompts in terms of their geometric characteristics. Our results demonstrate that CurvaLID delivers superior detection and rejection of adversarial queries, paving the way for safer LLM deployment. The source code can be found at this https URL 

**Abstract (ZH)**: 具备打破大规模语言模型（LLMs）并诱导不良行为的对抗提示构成了其安全部署的重大障碍。当前的缓解策略依赖于激活内置防御机制或微调LLMs，但对抗性提示与良性提示之间的根本区别尚未被理解。在本工作中，我们引入了CurvaLID，这是一种新型的防御框架，通过利用提示的几何属性有效地检测对抗提示。CurvaLID 不依赖于特定类型的LLM，提供了一种适用于各种对抗提示和LLM架构的统一检测框架。CurvaLID 基于文本提示的几何分析来揭示其潜在差异。我们通过Whewell方程将曲率的概念扩展到$n$维词嵌入空间，使得我们可以量化局部几何特性，包括语义转换和潜在流形中的曲率。此外，我们利用局部固有维数（LID）来捕捉对抗子空间内文本提示的几何特征。我们的研究结果表明，对抗提示在几何特性上与良性提示有根本的不同。我们的结果表明，CurvaLID 在检测和拒绝对抗查询方面表现出色，为更安全的大规模语言模型部署铺平了道路。源代码可在以下网址找到：这个 https URL。 

---
# SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Safe Reinforcement Learning 

**Title (ZH)**: SafeVLA：通过安全强化学习实现视觉-语言-行动模型的安全对齐 

**Authors**: Borong Zhang, Yuhao Zhang, Jiaming Ji, Yingshan Lei, Josef Dai, Yuanpei Chen, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03480)  

**Abstract**: Vision-language-action models (VLAs) have shown great potential as generalist robot policies. However, these models pose urgent safety challenges during deployment, including the risk of physical harm to the environment, the robot itself, and humans. How can safety be explicitly incorporated into VLAs? In this work, we propose SafeVLA, a novel algorithm designed to integrate safety into VLAs, ensuring the protection of the environment, robot hardware and humans in real-world settings. SafeVLA effectively balances safety and task performance by employing large-scale constrained learning within simulated environments. We demonstrate that SafeVLA outperforms the current state-of-the-art method in both safety and task performance, achieving average improvements of 83.58% and 3.85%, respectively, in simulation. By prioritizing safety, our approach eliminates high-risk behaviors and reduces the upper bound of unsafe behaviors to 1/35 of that in the current state-of-the-art, thereby significantly mitigating long-tail risks. Furthermore, the learned safety constraints generalize to diverse, unseen scenarios, including multiple out-of-distribution perturbations and tasks. Our data, models and newly proposed benchmark environment are available at this https URL. 

**Abstract (ZH)**: 视觉-语言-动作模型（VLAs）作为通用机器人策略展现出巨大的潜力。然而，在部署过程中，这些模型面临着迫切的安全挑战，包括对环境、机器人本身和人类的物理伤害风险。如何在VLAs中明确地纳入安全性？在本工作中，我们提出了SafeVLA，这是一种新颖的算法，旨在将安全性集成到VLAs中，确保在实际环境中的环境、机器人硬件和人类的安全。SafeVLA通过在模拟环境中采用大规模约束学习有效平衡了安全性和任务性能。实验结果表明，SafeVLA在安全性和任务性能方面均优于当前最先进的方法，在模拟实验中的安全性平均提升了83.58%，任务性能提升了3.85%。通过优先考虑安全，我们的方法消除了高风险行为，并将不可安全行为的上限减少了至当前最先进的方法的1/35，从而显著降低了长尾风险。此外，学习到的安全约束在多种未见过的场景中具有泛化能力，包括多种离分布扰动和任务。我们的数据、模型和新提出的基准环境可在以下链接获取。 

---
# Open-Source Large Language Models as Multilingual Crowdworkers: Synthesizing Open-Domain Dialogues in Several Languages With No Examples in Targets and No Machine Translation 

**Title (ZH)**: 开源大型语言模型作为多语言群众工作者：在无目标示例和无机器翻译的情况下合成多种语言的开放领域对话 

**Authors**: Ahmed Njifenjou, Virgile Sucal, Bassam Jabaian, Fabrice Lefèvre  

**Link**: [PDF](https://arxiv.org/pdf/2503.03462)  

**Abstract**: The prevailing paradigm in the domain of Open-Domain Dialogue agents predominantly focuses on the English language, encompassing both models and datasets. Furthermore, the financial and temporal investments required for crowdsourcing such datasets for finetuning are substantial, particularly when multiple languages are involved. Fortunately, advancements in Large Language Models (LLMs) have unveiled a plethora of possibilities across diverse tasks. Specifically, instruction-tuning has enabled LLMs to execute tasks based on natural language instructions, occasionally surpassing the performance of human crowdworkers. Additionally, these models possess the capability to function in various languages within a single thread. Consequently, to generate new samples in different languages, we propose leveraging these capabilities to replicate the data collection process. We introduce a pipeline for generating Open-Domain Dialogue data in multiple Target Languages using LLMs, with demonstrations provided in a unique Source Language. By eschewing explicit Machine Translation in this approach, we enhance the adherence to language-specific nuances. We apply this methodology to the PersonaChat dataset. To enhance the openness of generated dialogues and mimic real life scenarii, we added the notion of speech events corresponding to the type of conversation the speakers are involved in and also that of common ground which represents the premises of a conversation. 

**Abstract (ZH)**: 开放域对话代理领域的 prevailing 核心范式主要集中在英语语言，涵盖模型和数据集。然而，多语言场景下通过众包收集用于微调的数据集需要大量的财力和时间投入。幸运的是，大型语言模型（LLMs）的进步为多种任务开启了诸多可能性。特别是通过指令调谐，LLMs 可根据自然语言指令执行任务，有时甚至超越人类众包工人的表现。此外，这些模型能够在一个线程中处理多种语言。因此，为了生成不同语言的新样本，我们提出利用这些能力来复制数据收集过程。我们介绍了一种使用 LLMs 生成多目标语言开放域对话数据的管道，示范语言为独特的源语言。通过避免显式的机器翻译，我们增强了对语言特定细微差别的遵从。我们将此方法应用于 PersonaChat 数据集。为了增加生成对话的开放性并模仿真实场景，我们添加了与对话类型相对应的言语事件的概念，以及代表对话前提的共同知识概念。 

---
# Taxation Perspectives from Large Language Models: A Case Study on Additional Tax Penalties 

**Title (ZH)**: 大型语言模型视角下的税制探讨：关于额外税务处罚的案例研究 

**Authors**: Eunkyung Choi, Young Jin Suh, Hun Park, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03444)  

**Abstract**: How capable are large language models (LLMs) in the domain of taxation? Although numerous studies have explored the legal domain in general, research dedicated to taxation remain scarce. Moreover, the datasets used in these studies are either simplified, failing to reflect the real-world complexities, or unavailable as open source. To address this gap, we introduce PLAT, a new benchmark designed to assess the ability of LLMs to predict the legitimacy of additional tax penalties. PLAT is constructed to evaluate LLMs' understanding of tax law, particularly in cases where resolving the issue requires more than just applying related statutes. Our experiments with six LLMs reveal that their baseline capabilities are limited, especially when dealing with conflicting issues that demand a comprehensive understanding. However, we found that enabling retrieval, self-reasoning, and discussion among multiple agents with specific role assignments, this limitation can be mitigated. 

**Abstract (ZH)**: 大型语言模型在税收领域的能力如何？尽管已有大量研究探讨法律领域的一般问题，针对税收领域的研究仍较为稀缺。此外，这些研究中使用的数据集要么过于简化，无法反映现实世界的复杂性，要么无法获取开源数据。为填补这一空白，我们引入了PLAT，这是一个新的基准测试，旨在评估大型语言模型预测额外税收罚款正当性的能力。PLAT旨在评估大型语言模型对税法的理解，特别是在解决需要全面理解而非仅应用相关法律条文的问题时。我们的实验结果显示，大型语言模型的基本能力有限，尤其是在处理需要全面理解的冲突问题时。然而，我们发现通过启用检索、自我推理以及多角色代理之间的讨论，可以缓解这一限制。 

---
# Conceptualizing Uncertainty 

**Title (ZH)**: 构建不确定性概念 

**Authors**: Isaac Roberts, Alexander Schulz, Sarah Schroeder, Fabian Hinder, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2503.03443)  

**Abstract**: Uncertainty in machine learning refers to the degree of confidence or lack thereof in a model's predictions. While uncertainty quantification methods exist, explanations of uncertainty, especially in high-dimensional settings, remain an open challenge. Existing work focuses on feature attribution approaches which are restricted to local explanations. Understanding uncertainty, its origins, and characteristics on a global scale is crucial for enhancing interpretability and trust in a model's predictions. In this work, we propose to explain the uncertainty in high-dimensional data classification settings by means of concept activation vectors which give rise to local and global explanations of uncertainty. We demonstrate the utility of the generated explanations by leveraging them to refine and improve our model. 

**Abstract (ZH)**: 机器学习中的不确定性指的是模型预测的信心程度或缺乏程度。虽然存在不确定性量化方法，但在高维设置下解释不确定性仍然是一个开放性挑战。现有工作集中在特征归因方法，这些方法局限于局部解释。理解不确定性及其起源和特征在全局尺度上对于增强模型预测的可解释性和可信度至关重要。在这项工作中，我们提出通过概念激活向量来解释高维数据分类设置中的不确定性，从而提供局部和全局的不确定性解释。我们通过利用生成的解释来改进和完善我们的模型，展示了这些解释的实用性。 

---
# RASD: Retrieval-Augmented Speculative Decoding 

**Title (ZH)**: RASD: 检索增强推测解码 

**Authors**: Guofeng Quan, Wenfeng Feng, Chuzhan Hao, Guochao Jiang, Yuewei Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03434)  

**Abstract**: Speculative decoding accelerates inference in large language models (LLMs) by generating draft tokens for target model verification. Current approaches for obtaining draft tokens rely on lightweight draft models or additional model structures to generate draft tokens and retrieve context from databases. Due to the draft model's small size and limited training data, model-based speculative decoding frequently becomes less effective in out-of-domain scenarios. Additionally, the time cost of the drafting phase results in a low upper limit on acceptance length during the verification step, limiting overall efficiency. This paper proposes RASD (Retrieval-Augmented Speculative Decoding), which adopts retrieval methods to enhance model-based speculative decoding. We introduce tree pruning and tree fusion to achieve this. Specifically, we develop a pruning method based on the draft model's probability distribution to construct the optimal retrieval tree. Second, we employ the longest prefix matching algorithm to merge the tree generated by the draft model with the retrieval tree, resulting in a unified tree for verification. Experimental results demonstrate that RASD achieves state-of-the-art inference acceleration across tasks such as DocQA, Summary, Code, and In-Domain QA. Moreover, RASD exhibits strong scalability, seamlessly integrating with various speculative decoding approaches, including both generation-based and retrieval-based methods. 

**Abstract (ZH)**: 基于检索的推测性解码加速大型语言模型的推理 

---
# Privacy is All You Need: Revolutionizing Wearable Health Data with Advanced PETs 

**Title (ZH)**: 隐私至上：借助先进PET技术 revolutionize 可穿戴健康数据管理 

**Authors**: Karthik Barma, Seshu Babu Barma  

**Link**: [PDF](https://arxiv.org/pdf/2503.03428)  

**Abstract**: In a world where data is the new currency, wearable health devices offer unprecedented insights into daily life, continuously monitoring vital signs and metrics. However, this convenience raises privacy concerns, as these devices collect sensitive data that can be misused or breached. Traditional measures often fail due to real-time data processing needs and limited device power. Users also lack awareness and control over data sharing and usage. We propose a Privacy-Enhancing Technology (PET) framework for wearable devices, integrating federated learning, lightweight cryptographic methods, and selectively deployed blockchain technology. The blockchain acts as a secure ledger triggered only upon data transfer requests, granting users real-time notifications and control. By dismantling data monopolies, this approach returns data sovereignty to individuals. Through real-world applications like secure medical data sharing, privacy-preserving fitness tracking, and continuous health monitoring, our framework reduces privacy risks by up to 70 percent while preserving data utility and performance. This innovation sets a new benchmark for wearable privacy and can scale to broader IoT ecosystems, including smart homes and industry. As data continues to shape our digital landscape, our research underscores the critical need to maintain privacy and user control at the forefront of technological progress. 

**Abstract (ZH)**: 在数据成为新货币的世界中，可穿戴健康设备提供了对日常生活前所未有的洞察， continuously monitoring vital signs and metrics. However, this convenience raises privacy concerns, as these devices collect sensitive data that can be misused or breached. 传统措施往往因实时数据处理需求和有限的设备功率而失效。用户也缺乏对数据共享和使用情况的意识和控制。我们提出了一种增强隐私的技术（PET）框架，整合了联邦学习、轻量级密码方法和选择性部署的区块链技术。区块链作为一种安全账本，在数据传输请求时触发，为用户提供实时通知和控制。通过打破数据垄断，该方法将数据主权返回给个人。通过安全医疗数据共享、隐私保护的健身追踪和持续健康监测等实际应用，我们的框架将隐私风险降低高达70%，同时保持数据实用性和性能。这一创新为可穿戴设备隐私设立了新标准，并可扩展到更广泛的物联网生态系统，包括智能家居和工业。随着数据继续塑造我们的数字景观，我们的研究强调了在技术进步中维持隐私和用户控制的迫切需求。 

---
# Simplicial SMOTE: Oversampling Solution to the Imbalanced Learning Problem 

**Title (ZH)**: simplicial SMOTE：欠衡学习问题的过采样解决方案 

**Authors**: Oleg Kachan, Andrey Savchenko, Gleb Gusev  

**Link**: [PDF](https://arxiv.org/pdf/2503.03418)  

**Abstract**: SMOTE (Synthetic Minority Oversampling Technique) is the established geometric approach to random oversampling to balance classes in the imbalanced learning problem, followed by many extensions. Its idea is to introduce synthetic data points of the minor class, with each new point being the convex combination of an existing data point and one of its k-nearest neighbors. In this paper, by viewing SMOTE as sampling from the edges of a geometric neighborhood graph and borrowing tools from the topological data analysis, we propose a novel technique, Simplicial SMOTE, that samples from the simplices of a geometric neighborhood simplicial complex. A new synthetic point is defined by the barycentric coordinates w.r.t. a simplex spanned by an arbitrary number of data points being sufficiently close rather than a pair. Such a replacement of the geometric data model results in better coverage of the underlying data distribution compared to existing geometric sampling methods and allows the generation of synthetic points of the minority class closer to the majority class on the decision boundary. We experimentally demonstrate that our Simplicial SMOTE outperforms several popular geometric sampling methods, including the original SMOTE. Moreover, we show that simplicial sampling can be easily integrated into existing SMOTE extensions. We generalize and evaluate simplicial extensions of the classic Borderline SMOTE, Safe-level SMOTE, and ADASYN algorithms, all of which outperform their graph-based counterparts. 

**Abstract (ZH)**: Simplicial SMOTE：基于单纯复形的几何邻域采样方法 

---
# When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits 

**Title (ZH)**: 当声明演变：评估和提升嵌入模型对抗误导性编辑的稳健性 

**Authors**: Jabez Magomere, Emanuele La Malfa, Manuel Tonneau, Ashkan Kazemi, Scott Hale  

**Link**: [PDF](https://arxiv.org/pdf/2503.03417)  

**Abstract**: Online misinformation remains a critical challenge, and fact-checkers increasingly rely on embedding-based methods to retrieve relevant fact-checks. Yet, when debunked claims reappear in edited forms, the performance of these methods is unclear. In this work, we introduce a taxonomy of six common real-world misinformation edits and propose a perturbation framework that generates valid, natural claim variations. Our multi-stage retrieval evaluation reveals that standard embedding models struggle with user-introduced edits, while LLM-distilled embeddings offer improved robustness at a higher computational cost. Although a strong reranker helps mitigate some issues, it cannot fully compensate for first-stage retrieval gaps. Addressing these retrieval gaps, our train- and inference-time mitigation approaches enhance in-domain robustness by up to 17 percentage points and boost out-of-domain generalization by 10 percentage points over baseline models. Overall, our findings provide practical improvements to claim-matching systems, enabling more reliable fact-checking of evolving misinformation. 

**Abstract (ZH)**: 在线错误信息仍然是一个关键挑战，事实核查人员越来越依赖基于嵌入的方法来检索相关的事实核查内容。然而，当被驳斥的断言以编辑形式重新出现时，这些方法的性能尚不清楚。在本研究中，我们引入了六种常见现实世界错误信息编辑的分类，并提出了一种生成有效且自然断言变体的扰动框架。多阶段检索评估揭示出，标准嵌入模型在用户引入的编辑面前表现不佳，而通过LLM提炼的嵌入提供了较高计算成本下的增强鲁棒性。尽管强重排序器能部分缓解一些问题，但它无法完全弥补第一阶段检索的缺口。通过解决这些检索缺口，我们在训练时间和推理时间上的缓解方法能够将领域内鲁棒性提高多达17个百分点，并将领域外泛化能力提高10个百分点，相较于基线模型。总体而言，我们的研究结果为断言匹配系统提供了实用的改进，使事实核查能够更可靠地应对不断演变的错误信息。 

---
# Augmentation-Based Deep Learning for Identification of Circulating Tumor Cells 

**Title (ZH)**: 基于增强的深度学习循环肿瘤细胞识别 

**Authors**: Martina Russo, Giulia Bertolini, Vera Cappelletti, Cinzia De Marco, Serena Di Cosimo, Petra Paiè, Nadia Brancati  

**Link**: [PDF](https://arxiv.org/pdf/2503.03410)  

**Abstract**: Circulating tumor cells (CTCs) are crucial biomarkers in liquid biopsy, offering a noninvasive tool for cancer patient management. However, their identification remains particularly challenging due to their limited number and heterogeneity. Labeling samples for contrast limits the generalization of fluorescence-based methods across different hospital datasets. Analyzing single-cell images enables detailed assessment of cell morphology, subcellular structures, and phenotypic variations, often hidden in clustered images. Developing a method based on bright-field single-cell analysis could overcome these limitations. CTCs can be isolated using an unbiased workflow combining Parsortix technology, which selects cells based on size and deformability, with DEPArray technology, enabling precise visualization and selection of single cells. Traditionally, DEPArray-acquired digital images are manually analyzed, making the process time-consuming and prone to variability. In this study, we present a Deep Learning-based classification pipeline designed to distinguish CTCs from leukocytes in blood samples, aimed to enhance diagnostic accuracy and optimize clinical workflows. Our approach employs images from the bright-field channel acquired through DEPArray technology leveraging a ResNet-based CNN. To improve model generalization, we applied three types of data augmentation techniques and incorporated fluorescence (DAPI) channel images into the training phase, allowing the network to learn additional CTC-specific features. Notably, only bright-field images have been used for testing, ensuring the model's ability to identify CTCs without relying on fluorescence markers. The proposed model achieved an F1-score of 0.798, demonstrating its capability to distinguish CTCs from leukocytes. These findings highlight the potential of DL in refining CTC analysis and advancing liquid biopsy applications. 

**Abstract (ZH)**: 循环肿瘤细胞（CTCs）是液体活检中的关键生物标志物，提供了非侵入性工具以管理癌症患者。然而，由于其数量有限和异质性，它们的识别仍然颇具挑战。通过标记样品以提供对比度限制了基于荧光的方法在不同医院数据集中的普适性。分析单细胞图像可实现对细胞形态、亚细胞结构和表型变异的详细评估，这些信息在聚类图像中往往被隐藏。开发基于明场单细胞分析的方法可以克服这些限制。CTCs可以通过结合使用基于大小和变形性的Parsortix技术和DEPArray技术来分离，从而实现单细胞的精确可视化和选择。传统上，通过DEPArray获取的数字图像需要人工分析，使过程耗时且易变。在本研究中，我们提出了一种基于深度学习的分类管道，旨在增强诊断准确性和优化临床工作流程，以区分血液样本中的CTCs和中性粒细胞。我们的方法采用通过DEPArray技术获取的明场通道图像，并利用基于ResNet的CNN。为了提高模型的普适性，我们应用了三种数据增强技术，并将荧光（DAPI）通道图像纳入训练阶段，使网络能够学习更多的CTC特异性特征。值得注意的是，仅使用明场图像进行测试，确保模型能够不依赖荧光标记识别CTCs。所提出模型的F1分数为0.798，展示了其区分CTCs和中性粒细胞的能力。这些发现强调了深度学习在细化CTC分析和推进液体活检应用方面的能力。 

---
# AI-Driven Multi-Stage Computer Vision System for Defect Detection in Laser-Engraved Industrial Nameplates 

**Title (ZH)**: 基于AI驱动的多阶段计算机视觉系统在激光加工标识牌缺陷检测中的应用 

**Authors**: Adhish Anitha Vilasan, Stephan Jäger, Noah Klarmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.03395)  

**Abstract**: Automated defect detection in industrial manufacturing is essential for maintaining product quality and minimizing production errors. In air disc brake manufacturing, ensuring the precision of laser-engraved nameplates is crucial for accurate product identification and quality control. Engraving errors, such as misprints or missing characters, can compromise both aesthetics and functionality, leading to material waste and production delays. This paper presents a proof of concept for an AI-driven computer vision system that inspects and verifies laser-engraved nameplates, detecting defects in logos and alphanumeric strings. The system integrates object detection using YOLOv7, optical character recognition (OCR) with Tesseract, and anomaly detection through a residual variational autoencoder (ResVAE) along with other computer vision methods to enable comprehensive inspections at multiple stages. Experimental results demonstrate the system's effectiveness, achieving 91.33% accuracy and 100% recall, ensuring that defective nameplates are consistently detected and addressed. This solution highlights the potential of AI-driven visual inspection to enhance quality control, reduce manual inspection efforts, and improve overall manufacturing efficiency. 

**Abstract (ZH)**: 工业制造中的自动化缺陷检测对于维护产品质量和减少生产错误至关重要。在空气disc刹车制造中，确保激光刻印铭牌的精度对于准确的产品识别和质量控制至关重要。刻印错误，如错印或缺少字符，会损害美观性和功能性，导致材料浪费和生产延迟。本文提出了一个基于AI的计算机视觉系统的概念证明，该系统用于检查和验证激光刻印铭牌，检测标志和数字字符串中的缺陷。该系统结合了使用YOLOv7的对象检测、使用Tesseract的光学字符识别(OCR)和使用残差变分自编码器(ResVAE)的异常检测以及其他计算机视觉方法，以在多个阶段实现全面检查。实验结果表明，该系统的有效性，准确率为91.33%，召回率为100%，确保了缺陷铭牌的一致性检测和处理。该解决方案突显了基于AI的视觉检测在增强质量管理、减少人工检查工作和提高整体制造效率方面的潜力。 

---
# Multi-Agent DRL for Queue-Aware Task Offloading in Hierarchical MEC-Enabled Air-Ground Networks 

**Title (ZH)**: 基于分层MEC使能空地网络的多代理DRL任务卸载算法研究 

**Authors**: Muhammet Hevesli, Abegaz Mohammed Seid, Aiman Erbad, Mohamed Abdallah  

**Link**: [PDF](https://arxiv.org/pdf/2503.03391)  

**Abstract**: Mobile edge computing (MEC)-enabled air-ground networks are a key component of 6G, employing aerial base stations (ABSs) such as unmanned aerial vehicles (UAVs) and high-altitude platform stations (HAPS) to provide dynamic services to ground IoT devices (IoTDs). These IoTDs support real-time applications (e.g., multimedia and Metaverse services) that demand high computational resources and strict quality of service (QoS) guarantees in terms of latency and task queue management. Given their limited energy and processing capabilities, IoTDs rely on UAVs and HAPS to offload tasks for distributed processing, forming a multi-tier MEC system. This paper tackles the overall energy minimization problem in MEC-enabled air-ground integrated networks (MAGIN) by jointly optimizing UAV trajectories, computing resource allocation, and queue-aware task offloading decisions. The optimization is challenging due to the nonconvex, nonlinear nature of this hierarchical system, which renders traditional methods ineffective. We reformulate the problem as a multi-agent Markov decision process (MDP) with continuous action spaces and heterogeneous agents, and propose a novel variant of multi-agent proximal policy optimization with a Beta distribution (MAPPO-BD) to solve it. Extensive simulations show that MAPPO-BD outperforms baseline schemes, achieving superior energy savings and efficient resource management in MAGIN while meeting queue delay and edge computing constraints. 

**Abstract (ZH)**: 基于MEC的空地网络在6G中的关键作用：联合优化UAV轨迹、计算资源分配和感知任务卸载的多代理马克夫决策过程方法 

---
# Transformers for molecular property prediction: Domain adaptation efficiently improves performance 

**Title (ZH)**: 基于变压器的分子性质预测：域适应高效提升性能 

**Authors**: Afnan Sultan, Max Rausch-Dupont, Shahrukh Khan, Olga Kalinina, Andrea Volkamer, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2503.03360)  

**Abstract**: Most of the current transformer-based chemical language models are pre-trained on millions to billions of molecules. However, the improvement from such scaling in dataset size is not confidently linked to improved molecular property prediction. The aim of this study is to investigate and overcome some of the limitations of transformer models in predicting molecular properties. Specifically, we examine the impact of pre-training dataset size and diversity on the performance of transformer models and investigate the use of domain adaptation as a technique for improving model performance. First, our findings indicate that increasing pretraining dataset size beyond 400K molecules from the GuacaMol dataset does not result in a significant improvement on four ADME endpoints, namely, solubility, permeability, microsomal stability, and plasma protein binding. Second, our results demonstrate that using domain adaptation by further training the transformer model on a small set of domain-relevant molecules, i.e., a few hundred to a few thousand, using multi-task regression of physicochemical properties was sufficient to significantly improve performance for three out of the four investigated ADME endpoints (P-value < 0.001). Finally, we observe that a model pre-trained on 400K molecules and domain adopted on a few hundred/thousand molecules performs similarly (P-value > 0.05) to more complicated transformer models like MolBERT(pre-trained on 1.3M molecules) and MolFormer (pre-trained on 100M molecules). A comparison to a random forest model trained on basic physicochemical properties showed similar performance to the examined transformer models. We believe that current transformer models can be improved through further systematic analysis of pre-training and downstream data, pre-training objectives, and scaling laws, ultimately leading to better and more helpful models. 

**Abstract (ZH)**: 基于变压器的化学语言模型大多预先在百万到数亿个分子上进行训练。然而，这样的数据集规模扩大在分子性质预测上的改进并不肯定地与之关联。本研究旨在 Investigate and Overcome Some Limitations of Transformer Models in Predicting Molecular Properties。具体而言，我们探讨了预训练数据集的规模和多样性对变压器模型性能的影响，并研究了使用领域适应技术以提高模型性能的方法。首先，我们的发现表明，将预训练数据集规模从 GuacaMol 数据集的 400K 分子进一步增加并不会在四个 ADMET 端点（溶解度、渗透性、微粒体稳定性、血浆蛋白结合）上带来显著的性能提升。其次，我们的结果表明，通过在几百到几千个相关领域分子上进一步训练变压器模型，并利用多任务回归的物理化学性质，可以显著提高三个 ADMET 端点的性能（P 值 < 0.001）。最后，我们观察到，预训练在 400K 分子上并通过领域适应在几百到几千个分子上训练的模型与更复杂的变压器模型（如预训练在 1.3M 分子上的 MolBERT 和预训练在 100M 分子上的 MolFormer）具有相似的性能（P 值 > 0.05）。与在基本物理化学性质上训练的随机森林模型相比，其性能与检查的变压器模型相似。我们认为，通过进一步系统地分析预训练和下游数据、预训练目标和规模律，现有的变压器模型可以得到改进，最终导致更优秀和更实用的模型。 

---
# Navigating Intelligence: A Survey of Google OR-Tools and Machine Learning for Global Path Planning in Autonomous Vehicles 

**Title (ZH)**: 智能导航：面向自主车辆全球路径规划的Google OR-Tools和机器学习综述 

**Authors**: Alexandre Benoit, Pedram Asef  

**Link**: [PDF](https://arxiv.org/pdf/2503.03338)  

**Abstract**: We offer a new in-depth investigation of global path planning (GPP) for unmanned ground vehicles, an autonomous mining sampling robot named ROMIE. GPP is essential for ROMIE's optimal performance, which is translated into solving the traveling salesman problem, a complex graph theory challenge that is crucial for determining the most effective route to cover all sampling locations in a mining field. This problem is central to enhancing ROMIE's operational efficiency and competitiveness against human labor by optimizing cost and time. The primary aim of this research is to advance GPP by developing, evaluating, and improving a cost-efficient software and web application. We delve into an extensive comparison and analysis of Google operations research (OR)-Tools optimization algorithms. Our study is driven by the goal of applying and testing the limits of OR-Tools capabilities by integrating Reinforcement Learning techniques for the first time. This enables us to compare these methods with OR-Tools, assessing their computational effectiveness and real-world application efficiency. Our analysis seeks to provide insights into the effectiveness and practical application of each technique. Our findings indicate that Q-Learning stands out as the optimal strategy, demonstrating superior efficiency by deviating only 1.2% on average from the optimal solutions across our datasets. 

**Abstract (ZH)**: 针对自主采矿采样机器人ROMIE的全局路径规划研究：基于旅行推销员问题的优化算法比较与分析 

---
# See What You Are Told: Visual Attention Sink in Large Multimodal Models 

**Title (ZH)**: 见你被告知的：大型多模态模型中的视觉注意力 sink 

**Authors**: Seil Kang, Jinyeong Kim, Junhyeok Kim, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03321)  

**Abstract**: Large multimodal models (LMMs) "see" images by leveraging the attention mechanism between text and visual tokens in the transformer decoder. Ideally, these models should focus on key visual information relevant to the text token. However, recent findings indicate that LMMs have an extraordinary tendency to consistently allocate high attention weights to specific visual tokens, even when these tokens are irrelevant to the corresponding text. In this study, we investigate the property behind the appearance of these irrelevant visual tokens and examine their characteristics. Our findings show that this behavior arises due to the massive activation of certain hidden state dimensions, which resembles the attention sink found in language models. Hence, we refer to this phenomenon as the visual attention sink. In particular, our analysis reveals that removing the irrelevant visual sink tokens does not impact model performance, despite receiving high attention weights. Consequently, we recycle the attention to these tokens as surplus resources, redistributing the attention budget to enhance focus on the image. To achieve this, we introduce Visual Attention Redistribution (VAR), a method that redistributes attention in image-centric heads, which we identify as innately focusing on visual information. VAR can be seamlessly applied across different LMMs to improve performance on a wide range of tasks, including general vision-language tasks, visual hallucination tasks, and vision-centric tasks, all without the need for additional training, models, or inference steps. Experimental results demonstrate that VAR enables LMMs to process visual information more effectively by adjusting their internal attention mechanisms, offering a new direction to enhancing the multimodal capabilities of LMMs. 

**Abstract (ZH)**: Large Multimodal Models的视觉注意力陷阱：视觉注意力汇陷探究与缓解方法 

---
# Exploring specialization and sensitivity of convolutional neural networks in the context of simultaneous image augmentations 

**Title (ZH)**: 探索同时图像增强背景下卷积神经网络的专业化和敏感性 

**Authors**: Pavel Kharyuk, Sergey Matveev, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2503.03283)  

**Abstract**: Drawing parallels with the way biological networks are studied, we adapt the treatment--control paradigm to explainable artificial intelligence research and enrich it through multi-parametric input alterations. In this study, we propose a framework for investigating the internal inference impacted by input data augmentations. The internal changes in network operation are reflected in activation changes measured by variance, which can be decomposed into components related to each augmentation, employing Sobol indices and Shapley values. These quantities enable one to visualize sensitivity to different variables and use them for guided masking of activations. In addition, we introduce a way of single-class sensitivity analysis where the candidates are filtered according to their matching to prediction bias generated by targeted damaging of the activations. Relying on the observed parallels, we assume that the developed framework can potentially be transferred to studying biological neural networks in complex environments. 

**Abstract (ZH)**: 借鉴生物学网络的研究方式，我们将治疗-对照范式应用于可解释的人工智能研究，并通过多参数输入修改对其进行扩展。在本研究中，我们提出了一种框架，用于探究输入数据增强对内部推断的影响。网络操作内部变化通过测量激活变化的方差反映出来，这些变化可以分解为与每种增强相关的组件，使用Sobol指数和Shapley值。这些量使得能够可视化不同变量的敏感性，并用于引导激活的屏蔽。此外，我们引入了一种单一类别敏感性分析的方法，其中候选项根据其与通过目标激活损害生成的预测偏差的匹配程度进行筛选。基于观察到的相似性，我们假设所开发的框架可能在复杂环境中研究生物神经网络方面具有潜在的应用价值。 

---
# Benchmarking Dynamic SLO Compliance in Distributed Computing Continuum Systems 

**Title (ZH)**: 分布式计算 continuum 系统中动态SLO合规性的基准测试 

**Authors**: Alfreds Lapkovskis, Boris Sedlak, Sindri Magnússon, Schahram Dustdar, Praveen Kumar Donta  

**Link**: [PDF](https://arxiv.org/pdf/2503.03274)  

**Abstract**: Ensuring Service Level Objectives (SLOs) in large-scale architectures, such as Distributed Computing Continuum Systems (DCCS), is challenging due to their heterogeneous nature and varying service requirements across different devices and applications. Additionally, unpredictable workloads and resource limitations lead to fluctuating performance and violated SLOs. To improve SLO compliance in DCCS, one possibility is to apply machine learning; however, the design choices are often left to the developer. To that extent, we provide a benchmark of Active Inference -- an emerging method from neuroscience -- against three established reinforcement learning algorithms (Deep Q-Network, Advantage Actor-Critic, and Proximal Policy Optimization). We consider a realistic DCCS use case: an edge device running a video conferencing application alongside a WebSocket server streaming videos. Using one of the respective algorithms, we continuously monitor key performance metrics, such as latency and bandwidth usage, to dynamically adjust parameters -- including the number of streams, frame rate, and resolution -- to optimize service quality and user experience. To test algorithms' adaptability to constant system changes, we simulate dynamically changing SLOs and both instant and gradual data-shift scenarios, such as network bandwidth limitations and fluctuating device thermal states. Although the evaluated algorithms all showed advantages and limitations, our findings demonstrate that Active Inference is a promising approach for ensuring SLO compliance in DCCS, offering lower memory usage, stable CPU utilization, and fast convergence. 

**Abstract (ZH)**: 在分布式计算连续系统中确保服务级别目标的挑战及其实验研究：基于主动推断的方法 

---
# Conformal Transformations for Symmetric Power Transformers 

**Title (ZH)**: 对称电力变压器的共形变换 

**Authors**: Saurabh Kumar, Jacob Buckman, Carles Gelada, Sean Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03269)  

**Abstract**: Transformers with linear attention offer significant computational advantages over softmax-based transformers but often suffer from degraded performance. The symmetric power (sympow) transformer, a particular type of linear transformer, addresses some of this performance gap by leveraging symmetric tensor embeddings, achieving comparable performance to softmax transformers. However, the finite capacity of the recurrent state in sympow transformers limits their ability to retain information, leading to performance degradation when scaling the training or evaluation context length. To address this issue, we propose the conformal-sympow transformer, which dynamically frees up capacity using data-dependent multiplicative gating and adaptively stores information using data-dependent rotary embeddings. Preliminary experiments on the LongCrawl64 dataset demonstrate that conformal-sympow overcomes the limitations of sympow transformers, achieving robust performance across scaled training and evaluation contexts. 

**Abstract (ZH)**: 线性注意力变换器与基于softmax的变换器相比提供了显著的计算优势，但常常性能较差。对称幂（sympow）变换器作为一种特殊的线性变换器，通过利用对称张量嵌入，弥补部分性能差距，实现与基于softmax变换器相当的性能。然而，sympow变换器循环状态的有限容量限制了其信息保留能力，导致在扩展训练或评估上下文长度时性能下降。为解决这一问题，我们提出了一种符合性-sympow变换器，该变换器通过数据依赖的乘法门控动态释放容量，并使用数据依赖的旋转嵌入适当地存储信息。初步实验表明，符合性-sympow克服了sympow变换器的局限性，在扩展的训练和评估上下文中实现了稳健的性能。 

---
# Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions 

**Title (ZH)**: 自动驾驶中的轨迹预测：进展、局限性和未来方向 

**Authors**: Nadya Abdel Madjid, Abdulrahman Ahmad, Murad Mebrahtu, Yousef Babaa, Abdelmoamen Nasser, Sumbal Malik, Bilal Hassan, Naoufel Werghi, Jorge Dias, Majid Khonji  

**Link**: [PDF](https://arxiv.org/pdf/2503.03262)  

**Abstract**: As the potential for autonomous vehicles to be integrated on a large scale into modern traffic systems continues to grow, ensuring safe navigation in dynamic environments is crucial for smooth integration. To guarantee safety and prevent collisions, autonomous vehicles must be capable of accurately predicting the trajectories of surrounding traffic agents. Over the past decade, significant efforts from both academia and industry have been dedicated to designing solutions for precise trajectory forecasting. These efforts have produced a diverse range of approaches, raising questions about the differences between these methods and whether trajectory prediction challenges have been fully addressed. This paper reviews a substantial portion of recent trajectory prediction methods and devises a taxonomy to classify existing solutions. A general overview of the prediction pipeline is also provided, covering input and output modalities, modeling features, and prediction paradigms discussed in the literature. In addition, the paper discusses active research areas within trajectory prediction, addresses the posed research questions, and highlights the remaining research gaps and challenges. 

**Abstract (ZH)**: 随着自动驾驶车辆大规模集成到现代交通系统中的潜力不断增长，确保在其动态环境中安全导航对于顺利集成至关重要。为了保证安全并防止碰撞，自动驾驶车辆必须能够准确预测周围交通代理的轨迹。在过去十年中，学术界和工业界均投入了大量努力来设计精确轨迹预测的解决方案。这些努力产生了一系列多样化的方法，引发了关于这些方法之间差异以及轨迹预测挑战是否已被充分解决的问题。本文回顾了近年来大量的轨迹预测方法，并提出了一个分类体系以分类现有解决方案。此外，本文还提供了预测管道的总体概述，涵盖了文献中讨论的输入和输出模态、建模特征以及预测范式。同时，本文讨论了轨迹预测领域的活跃研究方向，回答了提出的研究问题，并指出了剩余的研究空白和挑战。 

---
# Exploring the Potential of Large Language Models as Predictors in Dynamic Text-Attributed Graphs 

**Title (ZH)**: 探索大型语言模型在动态文本属性图中的预测潜力 

**Authors**: Runlin Lei, Jiarui Ji, Haipeng Ding, Lu Yi, Zhewei Wei, Yongchao Liu, Chuntao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03258)  

**Abstract**: With the rise of large language models (LLMs), there has been growing interest in Graph Foundation Models (GFMs) for graph-based tasks. By leveraging LLMs as predictors, GFMs have demonstrated impressive generalizability across various tasks and datasets. However, existing research on LLMs as predictors has predominantly focused on static graphs, leaving their potential in dynamic graph prediction unexplored. In this work, we pioneer using LLMs for predictive tasks on dynamic graphs. We identify two key challenges: the constraints imposed by context length when processing large-scale historical data and the significant variability in domain characteristics, both of which complicate the development of a unified predictor. To address these challenges, we propose the GraphAgent-Dynamic (GAD) Framework, a multi-agent system that leverages collaborative LLMs. In contrast to using a single LLM as the predictor, GAD incorporates global and local summary agents to generate domain-specific knowledge, enhancing its transferability across domains. Additionally, knowledge reflection agents enable adaptive updates to GAD's knowledge, maintaining a unified and self-consistent architecture. In experiments, GAD demonstrates performance comparable to or even exceeds that of full-supervised graph neural networks without dataset-specific training. Finally, to enhance the task-specific performance of LLM-based predictors, we discuss potential improvements, such as dataset-specific fine-tuning to LLMs. By developing tailored strategies for different tasks, we provide new insights for the future design of LLM-based predictors. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的兴起，基于图的任务的图基础模型（GFMs）日益引起研究兴趣。通过利用LLMs作为预测器，GFMs在多种任务和数据集上展现了出色的泛化能力。然而，现有研究主要集中在静态图上，动态图预测的潜力尚未得到充分探索。在本文中，我们率先将LLMs应用于动态图的预测任务。我们识别出两个关键挑战：处理大规模历史数据时由上下文长度限制带来的约束，以及领域特征的显著差异，这两者都使得统一预测器的发展复杂化。为应对这些挑战，我们提出了GraphAgent-Dynamic（GAD）框架，这是一种利用协作性LLMs的多agent系统。与使用单一LLM作为预测器不同，GAD融合了全局和局部摘要agent以生成领域特定知识，增强了其跨领域的可迁移性。此外，知识反思agent允许GAD的知识进行自适应更新，保持统一且自洽的架构。在实验中，GAD在某些任务上表现出与全监督图神经网络相当甚至更好的性能，无需针对特定数据集进行训练。最后，为了提高基于LLM的预测器的任务特定性能，我们讨论了潜在的改进，如针对特定数据集对LLM进行微调。通过开发针对不同任务的定制化策略，我们为未来基于LLM的预测器的设计提供了新的见解。 

---
# Less is more? Rewards in RL for Cyber Defence 

**Title (ZH)**: 少就是多？网络防御中强化学习的奖励设计 

**Authors**: Elizabeth Bates, Chris Hicks, Vasilios Mavroudis  

**Link**: [PDF](https://arxiv.org/pdf/2503.03245)  

**Abstract**: The last few years has seen an explosion of interest in autonomous cyber defence agents based on deep reinforcement learning. Such agents are typically trained in a cyber gym environment, also known as a cyber simulator, at least 32 of which have already been built. Most, if not all cyber gyms provide dense "scaffolded" reward functions which combine many penalties or incentives for a range of (un)desirable states and costly actions. Whilst dense rewards help alleviate the challenge of exploring complex environments, yielding seemingly effective strategies from relatively few environment steps; they are also known to bias the solutions an agent can find, potentially towards suboptimal solutions. Sparse rewards could offer preferable or more effective solutions and have been overlooked by cyber gyms to date. In this work we set out to evaluate whether sparse reward functions might enable training more effective cyber defence agents. Towards this goal we first break down several evaluation limitations in existing work by proposing a ground truth evaluation score that goes beyond the standard RL paradigm used to train and evaluate agents. By adapting a well-established cyber gym to accommodate our methodology and ground truth score, we propose and evaluate two sparse reward mechanisms and compare them with a typical dense reward. Our evaluation considers a range of network sizes, from 2 to 50 nodes, and both reactive and proactive defensive actions. Our results show that sparse rewards, particularly positive reinforcement for an uncompromised network state, enable the training of more effective cyber defence agents. Furthermore, we show that sparse rewards provide more stable training than dense rewards, and that both effectiveness and training stability are robust to a variety of cyber environment considerations. 

**Abstract (ZH)**: 近年来，基于深度强化学习的自主网络防御代理引起了广泛关注。这类代理通常在一种被称为“网络模拟器”的网络健身房环境中进行训练，已有至少32种网络健身房被构建。大多数，如果不是全部，网络健身房提供了密集的“支撑式”奖励函数，结合了多种对多种（不）希望状态和昂贵行为的惩罚或激励。尽管密集奖励有助于缓解探索复杂环境的挑战，并从相对较少的环境步骤中产生看似有效的策略；但它们也可能使代理找到的解决方案偏向于非最优解。稀疏奖励可能提供更优或更有效的解决方案，并且到目前为止，网络健身房尚未对此予以关注。为了评估稀疏奖励机制是否能够训练出更有效的网络防御代理，我们提出了一种超越标准RL范式的地真相对于现有工作的评估限制提出了一个真正的评估得分。通过调整一个现有的网络健身房以适应我们的方法和地真相对于，我们提出了两种稀疏奖励机制并进行了评估，将其与典型的密集奖励进行了比较。我们的评估考虑了从2到50个节点的不同网络规模，以及反应性和前瞻性防御动作。结果表明，特别是对未受损网络状态的正向强化，稀疏奖励能够训练出更有效的网络防御代理。此外，我们证明稀疏奖励相较于密集奖励提供了更稳定的学习，同时，效果和学习稳定性对各种网络环境考虑因素具有鲁棒性。 

---
# FANS -- Formal Answer Selection for Natural Language Math Reasoning Using Lean4 

**Title (ZH)**: FANS -- 形式化的答案选择在自然语言数学推理中的应用（使用Lean4） 

**Authors**: Jiarui Yao, Ruida Wang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03238)  

**Abstract**: Large Language Models (LLMs) have displayed astonishing abilities in various tasks, especially in text generation, classification, question answering, etc. However, the reasoning ability of LLMs still faces many debates. The inherent ambiguity of Natural Language (NL) limits LLMs' ability to perform verifiable reasoning, making its answers lack coherence and trustworthy support. To tackle the above problems, we propose a novel framework named FANS: Formal ANswer Selection for Natural Language Math Reasoning Using Lean4. To the best of our knowledge, it is the first framework that utilizes Lean4 to enhance LLMs' NL math reasoning ability. In particular, given an NL math question and LLM-generated answers, FANS first translates it into Lean4 theorem statements. Then it tries to prove it using a Lean4 prover and verify it by Lean4. Finally, it uses the FL result to assist in answer selection. It enhances LLMs' NL math ability in providing a computer-verifiable solution for its correct answer and proposes an alternative method for answer selection beyond the reward model. Extensive experiments indicate the effectiveness of our framework. It can improve the accuracy rate of reward model enhanced LLMs in the MATH-500 dataset by at most 1.91% and AMC-23 by at most 8.33% on strong reward-model baselines. In some particular fields like number theory that Lean4 experts in, we can even select all correct solutions. The qualitative analysis also shows our framework can make NL results formally backed by Lean4 proofs. As a pioneering work in the corresponding field, we will open-source all our models and datasets to further boost the development of the field. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展现了令人惊讶的能力，特别是在文本生成、分类、问答等方面。然而，LLMs的推理能力仍然存在许多争议。自然语言（NL）的内在歧义性限制了LLMs进行可验证推理的能力，使其答案缺乏连贯性和可信的支持。为了解决上述问题，我们提出了一种名为FANS的新框架：使用Lean4进行自然语言数学推理的形式答案选择。据我们所知，这是第一个利用Lean4增强LLMs自然语言数学推理能力的框架。特别是，给定一个自然语言数学问题和LLMs生成的答案，FANS首先将其翻译成Lean4定理陈述。然后，使用Lean4证明器尝试证明它，并通过Lean4验证。最后，使用FL结果辅助答案选择。它通过为正确答案提供计算机可验证的解决方案增强了LLMs的自然语言数学能力，并提出了一种超越奖励模型的备选答案选择方法。广泛实验表明该框架的有效性。它可以在MATH-500数据集中将奖励模型增强的LLMs的准确率最多提高1.91%，在AMC-23数据集上最多提高8.33%，在某些特定领域如Lean4专家擅长的数论领域，甚至可以选出所有正确答案。定性分析也表明，该框架可以使自然语言结果正式地由Lean4证明支撑。作为该领域的开创性工作，我们将开源所有模型和数据集以进一步促进该领域的发展。 

---
# NodeReg: Mitigating the Imbalance and Distribution Shift Effects in Semi-Supervised Node Classification via Norm Consistency 

**Title (ZH)**: NodeReg: 通过范数一致性缓解半监督节点分类中的类别不平衡和分布偏移效应 

**Authors**: Shenzhi Yang, Jun Xia, Jingbo Zhou, Xingkai Yao, Xiaofang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03211)  

**Abstract**: Aggregating information from neighboring nodes benefits graph neural networks (GNNs) in semi-supervised node classification tasks. Nevertheless, this mechanism also renders nodes susceptible to the influence of their neighbors. For instance, this will occur when the neighboring nodes are imbalanced or the neighboring nodes contain noise, which can even affect the GNN's ability to generalize out of distribution. We find that ensuring the consistency of the norm for node representations can significantly reduce the impact of these two issues on GNNs. To this end, we propose a regularized optimization method called NodeReg that enforces the consistency of node representation norms. This method is simple but effective and satisfies Lipschitz continuity, thus facilitating stable optimization and significantly improving semi-supervised node classification performance under the above two scenarios. To illustrate, in the imbalance scenario, when training a GCN with an imbalance ratio of 0.1, NodeReg outperforms the most competitive baselines by 1.4%-25.9% in F1 score across five public datasets. Similarly, in the distribution shift scenario, NodeReg outperforms the most competitive baseline by 1.4%-3.1% in accuracy. 

**Abstract (ZH)**: 从邻居节点聚合信息有助于图神经网络（GNNs）在半监督节点分类任务中的表现。然而，这种机制也会使节点容易受到邻居节点的影响。例如，这将发生在邻居节点不平衡或邻居节点包含噪声的情况下，甚至可能影响GNN的分布外泛化能力。我们发现确保节点表示范数的一致性可以显著减少这两种问题对GNN的影响。为此，我们提出了一种名为NodeReg的正则化优化方法，该方法强制节点表示范数的一致性。该方法简单而有效，并且满足利普希茨连续性，从而便于稳定优化，并在上述两种情况下大幅提高半监督节点分类性能。以不平衡场景为例，当训练不平衡比为0.1的GCN时，NodeReg在五个公共数据集上的F1分数上比最具竞争力的基线高出1.4%-25.9%。同样，在分布偏移场景中，NodeReg的准确率比最具竞争力的基线高出1.4%-3.1%。 

---
# MA-LoT: Multi-Agent Lean-based Long Chain-of-Thought Reasoning enhances Formal Theorem Proving 

**Title (ZH)**: MA-LoT：基于多智能体学习的长推理链增强形式定理证明 

**Authors**: Ruida Wang, Rui Pan, Yuxin Li, Jipeng Zhang, Yizhen Jia, Shizhe Diao, Renjie Pi, Junjie Hu, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03205)  

**Abstract**: Solving mathematical problems using computer-verifiable languages like Lean has significantly impacted mathematical and computer science communities. State-of-the-art methods utilize single Large Language Models (LLMs) as agents or provers to either generate complete proof or perform tree searches. However, single-agent methods inherently lack a structured way to combine high-level reasoning in Natural Language (NL) with Formal Language (FL) verification feedback. To solve these issues, we propose MA-LoT: Multi-Agent Lean-based Long Chain-of-Thought framework, (to the best of our knowledge), the first multi-agent framework for Lean4 theorem proving that balance high-level NL reasoning and FL verification in Long CoT. Using this structured interaction, our approach enables deeper insights and long-term coherence in proof generation, with which past methods struggle. We do this by leveraging emergent formal reasoning ability in Long CoT using our novel LoT-Transfer Learning training-inference pipeline. Extensive experiments show that our framework achieves 54.51% accuracy rate on the Lean4 version of MiniF2F-Test dataset, largely outperforming GPT-4 (22.95%), single-agent tree search (InternLM-Step-Prover, 50.70%), and whole-proof generation (DeepSeek-Prover-v1.5, 48.36%) baselines. Furthermore, our findings highlight the potential of combining Long CoT with formal verification for a more insightful generation in a broader perspective. 

**Abstract (ZH)**: 使用类似Lean的计算机验证语言求解数学问题显著影响了数学和计算机科学社区。现有的高级方法利用单个大型语言模型（LLMs）作为代理或证明者来生成完整的证明或执行树搜索。然而，单代理方法本质上缺乏将高级自然语言（NL）推理与形式语言（FL）验证反馈有机结合的结构化方式。为解决这些问题，我们提出MA-LoT：基于Lean的多代理长链推理框架，据我们所知，这是第一个在Lean4定理证明中平衡高级NL推理和长链推理FL验证的多代理框架。通过这种结构化的互动，我们的方法能够在证明生成中提供更深刻的见解和长期连贯性，这是以往方法难以实现的。我们通过利用我们在长链推理中新兴的形式推理能力，使用我们新颖的LoT迁移学习训练-推理管道来实现这一点。广泛的实验结果显示，我们的框架在Lean4版本的MiniF2F-Test数据集上的准确率为54.51%，大幅优于GPT-4（22.95%）、单代理树搜索（InternLM-Step-Prover，50.70%）和完整证明生成（DeepSeek-Prover-v1.5，48.36%）基准。此外，我们的研究结果强调了将长链推理与形式验证结合以从更广泛的角度实现更深入生成的潜力。 

---
# Towards Robust Universal Information Extraction: Benchmark, Evaluation, and Solution 

**Title (ZH)**: 面向鲁棒通用信息提取：基准、评估与解决方案 

**Authors**: Jizhao Zhu, Akang Shi, Zixuan Li, Long Bai, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.03201)  

**Abstract**: In this paper, we aim to enhance the robustness of Universal Information Extraction (UIE) by introducing a new benchmark dataset, a comprehensive evaluation, and a feasible solution. Existing robust benchmark datasets have two key limitations: 1) They generate only a limited range of perturbations for a single Information Extraction (IE) task, which fails to evaluate the robustness of UIE models effectively; 2) They rely on small models or handcrafted rules to generate perturbations, often resulting in unnatural adversarial examples. Considering the powerful generation capabilities of Large Language Models (LLMs), we introduce a new benchmark dataset for Robust UIE, called RUIE-Bench, which utilizes LLMs to generate more diverse and realistic perturbations across different IE tasks. Based on this dataset, we comprehensively evaluate existing UIE models and reveal that both LLM-based models and other models suffer from significant performance drops. To improve robustness and reduce training costs, we propose a data-augmentation solution that dynamically selects hard samples for iterative training based on the model's inference loss. Experimental results show that training with only \textbf{15\%} of the data leads to an average \textbf{7.5\%} relative performance improvement across three IE tasks. 

**Abstract (ZH)**: 本文aiming to提升Universal Information Extraction (UIE)的 robustness，通过引入一个新的基准数据集、全面的评估和可行的解决方案。考虑到大型语言模型的强大生成能力，我们提出一个新的robust UIE基准数据集RUIE-Bench，利用大型语言模型生成更多样化和真实的扰动，适用于不同信息提取任务。基于此数据集，我们全面评估了现有UIE模型，并发现基于大型语言模型的模型和其他模型均遭受显著性能下降。为了提高robustness并减少训练成本，我们提出了一种数据增强解决方案，该方案根据模型的推理损失动态选择困难样本进行迭代训练。实验结果表明，仅使用数据的\textbf{15\%}进行训练，在三个信息提取任务上的相对性能平均提高\textbf{7.5\%}。 

---
# Directly Follows Graphs Go Predictive Process Monitoring With Graph Neural Networks 

**Title (ZH)**: 直接跟随图实现基于图神经网络的预测性过程监控 

**Authors**: Attila Lischka, Simon Rauch, Oliver Stritzel  

**Link**: [PDF](https://arxiv.org/pdf/2503.03197)  

**Abstract**: In the past years, predictive process monitoring (PPM) techniques based on artificial neural networks have evolved as a method to monitor the future behavior of business processes. Existing approaches mostly focus on interpreting the processes as sequences, so-called traces, and feeding them to neural architectures designed to operate on sequential data such as recurrent neural networks (RNNs) or transformers. In this study, we investigate an alternative way to perform PPM: by transforming each process in its directly-follows-graph (DFG) representation we are able to apply graph neural networks (GNNs) for the prediction tasks. By this, we aim to develop models that are more suitable for complex processes that are long and contain an abundance of loops. In particular, we present different ways to create DFG representations depending on the particular GNN we use. The tested GNNs range from classical node-based to novel edge-based architectures. Further, we investigate the possibility of using multi-graphs. By these steps, we aim to design graph representations that minimize the information loss when transforming traces into graphs. 

**Abstract (ZH)**: 近年来，基于人工神经网络的预测过程监控（PPM）技术演化成为监测业务过程未来行为的一种方法。现有方法主要侧重于将过程视为序列，即所谓的轨迹，并将它们输入到适用于序列数据的操作架构，如循环神经网络（RNNs）或变压器中。在本研究中，我们探索了一种替代的PPM方式：通过将每个过程转化为直接跟随图（DFG）表示，我们能够应用图神经网络（GNNs）进行预测任务。通过这种方式，我们旨在开发更适合复杂、长且包含大量循环的过程的模型。特别是，我们根据不同所使用的GNN提出了不同的DFG表示方法。测试的GNN包括经典的节点基架构和新颖的边基架构。另外，我们还探讨了使用多图的可能性。通过上述步骤，我们旨在设计图表示方法，以最大限度地减少将轨迹转换为图时的信息损失。 

---
# Structured Outputs Enable General-Purpose LLMs to be Medical Experts 

**Title (ZH)**: 结构化输出使通用型大语言模型成为医疗专家 

**Authors**: Guangfu Guo, Kai Zhang, Bryan Hoo, Yujun Cai, Xiaoqian Lu, Nanyun Peng, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03194)  

**Abstract**: Medical question-answering (QA) is a critical task for evaluating how effectively large language models (LLMs) encode clinical knowledge and assessing their potential applications in medicine. Despite showing promise on multiple-choice tests, LLMs frequently struggle with open-ended medical questions, producing responses with dangerous hallucinations or lacking comprehensive coverage of critical aspects. Existing approaches attempt to address these challenges through domain-specific fine-tuning, but this proves resource-intensive and difficult to scale across models. To improve the comprehensiveness and factuality of medical responses, we propose a novel approach utilizing structured medical reasoning. Our method guides LLMs through an seven-step cognitive process inspired by clinical diagnosis, enabling more accurate and complete answers without additional training. Experiments on the MedLFQA benchmark demonstrate that our approach achieves the highest Factuality Score of 85.8, surpassing fine-tuned models. Notably, this improvement transfers to smaller models, highlighting the method's efficiency and scalability. Our code and datasets are available. 

**Abstract (ZH)**: 医学问答(Medical QA)是评估大规模语言模型(LLMs)如何有效地编码临床知识以及评估其在医学领域的潜在应用的关键任务。尽管在多项选择测试中显示出潜力，LLMs在应对开放性医学问题时经常遇到困难，生成包含危险幻觉或缺乏关键方面综合覆盖的回复。现有方法试图通过领域特定的微调来解决这些挑战，但这证明资源密集且难以在多个模型上扩展。为提高医学回复的全面性和事实性，我们提出了一种新的方法，利用结构化的医学推理。我们的方法指导LLMs通过一个受临床诊断启发的七步认知过程，使其能够提供更准确和完整的答案，而无需额外的训练。在MedLFQA基准测试上的实验表明，我们的方法实现了最高的事实得分85.8，超过了细调模型。值得注意的是，这一改进适用于较小的模型，突显了该方法的高效性和可扩展性。我们的代码和数据集已公开。 

---
# Intermediate-Task Transfer Learning: Leveraging Sarcasm Detection for Stance Detection 

**Title (ZH)**: 中介任务迁移学习：利用讽刺检测促进立场检测 

**Authors**: Gibson Nkhata, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2503.03172)  

**Abstract**: Stance Detection (SD) on social media has emerged as a prominent area of interest with implications for social business and political applications thereby garnering escalating research attention within NLP. The inherent subtlety and complexity of texts procured from online platforms pose challenges for SD algorithms in accurately discerning the authors stance. Mostly the inclusion of sarcastic and figurative language drastically impacts the performance of SD models. This paper addresses this by employing sarcasm detection intermediate-task transfer learning tailored for SD. The proposed methodology involves the finetuning of BERT and RoBERTa and the concatenation of convolutional BiLSTM and dense layers. Rigorous experiments are conducted on publicly available datasets to evaluate our transfer-learning framework. The performance of the approach is assessed against various State-Of-The-Art baselines for SD providing empirical evidence of its effectiveness. Notably our model outperforms the best SOTA models even prior to sarcasm-detection pretraining. The integration of sarcasm knowledge into the model proves instrumental in mitigating misclassifications of sarcastic textual elements in SD. Our model accurately predicts 85% of texts that were previously misclassified by the model without sarcasm-detection pretraining thereby amplifying the average F1-score of the model. Our experiments also revealed that the success of the transfer-learning framework is contingent upon the correlation of lexical attributes between the intermediate task and the target task. This study represents the first exploration of sarcasm detection as an intermediate transfer-learning task in the context of SD and simultaneously uses the concatenation of BERT or RoBERTa with other deep-learning techniques establishing the proposed approach as a foundational baseline for future research endeavors in this domain. 

**Abstract (ZH)**: 社交媒体中的立场检测（SD）已成为一个重要的研究领域，对社交商业和政治应用具有重要影响，因此在自然语言处理（NLP）领域吸引了越来越多的研究关注。来源于在线平台的文本隐含的细微性和复杂性给SD算法准确识别作者立场带来了挑战。特别是 sarcastic 和比喻语言的加入严重影响了SD模型的表现。本文通过针对SD的应用进行讽刺检测的中间任务迁移学习来应对这一挑战。提出的这种方法涉及对BERT和RoBERTa的微调，以及卷积双向LSTM和密集层的连接。在公开可用的数据集上进行了严格实验以评估我们的迁移学习框架。将该方法与各种最新的SD基线进行比较，提供了其有效性的实证证据。值得注意的是，在讽刺检测预训练之前，我们的模型的性能就已经优于最先进的模型。将讽刺知识整合到模型中，对于缓解SD中讽刺文本元素的误分类起到了重要作用。我们的模型准确预测了85%之前由未进行讽 刺检测预训练的模型误分类的文本，从而提高了模型的平均F1分数。实验还表明，迁移学习框架的成功取决于中间任务和目标任务的词形特征的相关性。本研究是首次在SD背景下将讽刺检测作为中间迁移学习任务进行探索，并结合BERT或RoBERTa与其他深度学习技术的连接，使提出的方法成为未来该领域研究的基础。 

---
# AttackSeqBench: Benchmarking Large Language Models' Understanding of Sequential Patterns in Cyber Attacks 

**Title (ZH)**: AttackSeqBench: 评价大规模语言模型对网络攻击序列模式理解的能力 

**Authors**: Javier Yong, Haokai Ma, Yunshan Ma, Anis Yusof, Zhenkai Liang, Ee-Chien Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.03170)  

**Abstract**: The observations documented in Cyber Threat Intelligence (CTI) reports play a critical role in describing adversarial behaviors, providing valuable insights for security practitioners to respond to evolving threats. Recent advancements of Large Language Models (LLMs) have demonstrated significant potential in various cybersecurity applications, including CTI report understanding and attack knowledge graph construction. While previous works have proposed benchmarks that focus on the CTI extraction ability of LLMs, the sequential characteristic of adversarial behaviors within CTI reports remains largely unexplored, which holds considerable significance in developing a comprehensive understanding of how adversaries operate. To address this gap, we introduce AttackSeqBench, a benchmark tailored to systematically evaluate LLMs' capability to understand and reason attack sequences in CTI reports. Our benchmark encompasses three distinct Question Answering (QA) tasks, each task focuses on the varying granularity in adversarial behavior. To alleviate the laborious effort of QA construction, we carefully design an automated dataset construction pipeline to create scalable and well-formulated QA datasets based on real-world CTI reports. To ensure the quality of our dataset, we adopt a hybrid approach of combining human evaluation and systematic evaluation metrics. We conduct extensive experiments and analysis with both fast-thinking and slow-thinking LLMs, while highlighting their strengths and limitations in analyzing the sequential patterns in cyber attacks. The overarching goal of this work is to provide a benchmark that advances LLM-driven CTI report understanding and fosters its application in real-world cybersecurity operations. Our dataset and code are available at this https URL . 

**Abstract (ZH)**: 攻击序列基准：评估大规模语言模型在网络威胁情报报告中理解与推理攻击序列的能力 

---
# DiRe-JAX: A JAX based Dimensionality Reduction Algorithm for Large-scale Data 

**Title (ZH)**: DiRe-JAX：一种基于JAX的大规模数据降维算法 

**Authors**: Alexander Kolpakov, Igor Rivin  

**Link**: [PDF](https://arxiv.org/pdf/2503.03156)  

**Abstract**: DiRe-JAX is a new dimensionality reduction toolkit designed to address some of the challenges faced by traditional methods like UMAP and tSNE such as loss of global structure and computational efficiency. Built on the JAX framework, DiRe leverages modern hardware acceleration to provide an efficient, scalable, and interpretable solution for visualizing complex data structures, and for quantitative analysis of lower-dimensional embeddings. The toolkit shows considerable promise in preserving both local and global structures within the data as compare to state-of-the-art UMAP and tSNE implementations. This makes it suitable for a wide range of applications in machine learning, bioinformatics, and data science. 

**Abstract (ZH)**: DiRe-JAX 是一个新颖的降维工具包，旨在解决传统方法如 UMAP 和 tSNE 面临的全球结构丢失和计算效率低下的挑战。基于 JAX 框架，DiRe 利用现代硬件加速提供了一个高效、可扩展且可解释的方案，用于可视化复杂的数据结构，并对低维嵌入进行定量分析。该工具包在保留数据的局部和全局结构方面表现出色，优于最先进的UMAP和tSNE实现，使其在机器学习、生物信息学和数据科学等领域具有广泛的应用前景。 

---
# Position: Model Collapse Does Not Mean What You Think 

**Title (ZH)**: 位置：模型坍塌并不如你所想 

**Authors**: Rylan Schaeffer, Joshua Kazdan, Alvan Caleb Arulandu, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2503.03150)  

**Abstract**: The proliferation of AI-generated content online has fueled concerns over \emph{model collapse}, a degradation in future generative models' performance when trained on synthetic data generated by earlier models. Industry leaders, premier research journals and popular science publications alike have prophesied catastrophic societal consequences stemming from model collapse. In this position piece, we contend this widespread narrative fundamentally misunderstands the scientific evidence. We highlight that research on model collapse actually encompasses eight distinct and at times conflicting definitions of model collapse, and argue that inconsistent terminology within and between papers has hindered building a comprehensive understanding of model collapse. To assess how significantly different interpretations of model collapse threaten future generative models, we posit what we believe are realistic conditions for studying model collapse and then conduct a rigorous assessment of the literature's methodologies through this lens. While we leave room for reasonable disagreement, our analysis of research studies, weighted by how faithfully each study matches real-world conditions, leads us to conclude that certain predicted claims of model collapse rely on assumptions and conditions that poorly match real-world conditions, and in fact several prominent collapse scenarios are readily avoidable. Altogether, this position paper argues that model collapse has been warped from a nuanced multifaceted consideration into an oversimplified threat, and that the evidence suggests specific harms more likely under society's current trajectory have received disproportionately less attention. 

**Abstract (ZH)**: AI生成内容的泛滥加剧了对未来生成模型性能下降的担忧：模型崩溃的认知偏差及其评估 

---
# Partial Convolution Meets Visual Attention 

**Title (ZH)**: 部分卷积结合视觉注意力 

**Authors**: Haiduo Huang, Fuwei Yang, Dong Li, Ji Liu, Lu Tian, Jinzhang Peng, Pengju Ren, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2503.03148)  

**Abstract**: Designing an efficient and effective neural network has remained a prominent topic in computer vision research. Depthwise onvolution (DWConv) is widely used in efficient CNNs or ViTs, but it needs frequent memory access during inference, which leads to low throughput. FasterNet attempts to introduce partial convolution (PConv) as an alternative to DWConv but compromises the accuracy due to underutilized channels. To remedy this shortcoming and consider the redundancy between feature map channels, we introduce a novel Partial visual ATtention mechanism (PAT) that can efficiently combine PConv with visual attention. Our exploration indicates that the partial attention mechanism can completely replace the full attention mechanism and reduce model parameters and FLOPs. Our PAT can derive three types of blocks: Partial Channel-Attention block (PAT_ch), Partial Spatial-Attention block (PAT_sp) and Partial Self-Attention block (PAT_sf). First, PAT_ch integrates the enhanced Gaussian channel attention mechanism to infuse global distribution information into the untouched channels of PConv. Second, we introduce the spatial-wise attention to the MLP layer to further improve model accuracy. Finally, we replace PAT_ch in the last stage with the self-attention mechanism to extend the global receptive field. Building upon PAT, we propose a novel hybrid network family, named PATNet, which achieves superior top-1 accuracy and inference speed compared to FasterNet on ImageNet-1K classification and excel in both detection and segmentation on the COCO dataset. Particularly, our PATNet-T2 achieves 1.3% higher accuracy than FasterNet-T2, while exhibiting 25% higher GPU throughput and 24% lower CPU latency. 

**Abstract (ZH)**: 设计高效且有效的神经网络一直是计算机视觉研究中的一个突出话题。Designing Efficient and Effective Neural Networks Has Remained a Prominent Topic in Computer Vision Research. 

---
# Knowledge Augmentation in Federation: Rethinking What Collaborative Learning Can Bring Back to Decentralized Data 

**Title (ZH)**: 联邦学习中的知识增强：重塑协作学习对去中心化数据的贡献 

**Authors**: Wentai Wu, Yingliang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03140)  

**Abstract**: Data, as an observable form of knowledge, has become one of the most important factors of production for the development of Artificial Intelligence (AI). Meanwhile, increasing legislation and regulations on private and proprietary information results in scattered data sources also known as the ``data islands''. Although some collaborative learning paradigms such as Federated Learning (FL) can enable privacy-preserving training over decentralized data, they have inherent deficiencies in fairness, costs and reproducibility because of being learning-centric, which greatly limits the way how participants cooperate with each other. In light of this, we present a knowledge-centric paradigm termed \emph{Knowledge Augmentation in Federation} (KAF), with focus on how to enhance local knowledge through collaborative effort. We provide the suggested system architecture, formulate the prototypical optimization objective, and review emerging studies that employ methodologies suitable for KAF. On our roadmap, with a three-way categorization we describe the methods for knowledge expansion, knowledge filtering, and label and feature space correction in the federation. Further, we highlight several challenges and open questions that deserve more attention from the community. With our investigation, we intend to offer new insights for what collaborative learning can bring back to decentralized data. 

**Abstract (ZH)**: 基于知识为中心的知识增强联合会知识增益与联邦学习 

---
# Convergence Analysis of Federated Learning Methods Using Backward Error Analysis 

**Title (ZH)**: 联邦学习方法的回向误差分析收敛性分析 

**Authors**: Jinwoo Lim, Suhyun Kim, Soo-Mook Moon  

**Link**: [PDF](https://arxiv.org/pdf/2503.03139)  

**Abstract**: Backward error analysis allows finding a modified loss function, which the parameter updates really follow under the influence of an optimization method. The additional loss terms included in this modified function is called implicit regularizer. In this paper, we attempt to find the implicit regularizer for various federated learning algorithms on non-IID data distribution, and explain why each method shows different convergence behavior. We first show that the implicit regularizer of FedAvg disperses the gradient of each client from the average gradient, thus increasing the gradient variance. We also empirically show that the implicit regularizer hampers its convergence. Similarly, we compute the implicit regularizers of FedSAM and SCAFFOLD, and explain why they converge better. While existing convergence analyses focus on pointing out the advantages of FedSAM and SCAFFOLD, our approach can explain their limitations in complex non-convex settings. In specific, we demonstrate that FedSAM can partially remove the bias in the first-order term of the implicit regularizer in FedAvg, whereas SCAFFOLD can fully eliminate the bias in the first-order term, but not in the second-order term. Consequently, the implicit regularizer can provide a useful insight on the convergence behavior of federated learning from a different theoretical perspective. 

**Abstract (ZH)**: backward误差分析允许找到一个修改后的损失函数，参数更新在优化方法影响下确实遵循该函数。此修改函数中包含的额外损失项称为隐式正则化器。本文尝试在非IID数据分布下为各种联邦学习算法找到隐式正则化器，并解释为什么每种方法表现出不同的收敛行为。我们首先表明，FedAvg的隐式正则化器使每个客户端的梯度分散到平均梯度之外，从而增加了梯度方差。我们还通过实验表明，隐式正则化器阻碍了其收敛。类似地，我们计算了FedSAM和SCAFFOLD的隐式正则化器，并解释了它们为什么能够更好地收敛。现有收敛性分析主要强调了FedSAM和SCAFFOLD的优点，而我们的方法可以解释它们在复杂非凸设置下的局限性。具体来说，我们证明了FedSAM可以部分消除FedAvg隐式正则化器中的一阶项偏差，而SCAFFOLD可以完全消除一阶项偏差，但不能消除二阶项偏差。因此，隐式正则化器可以从不同的理论角度提供有关联邦学习收敛行为的有用洞察。 

---
# Exploring Neural Ordinary Differential Equations as Interpretable Healthcare classifiers 

**Title (ZH)**: 探索神经常微分方程作为可解释的医疗分类器 

**Authors**: Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.03129)  

**Abstract**: Deep Learning has emerged as one of the most significant innovations in machine learning. However, a notable limitation of this field lies in the ``black box" decision-making processes, which have led to skepticism within groups like healthcare and scientific communities regarding its applicability. In response, this study introduces a interpretable approach using Neural Ordinary Differential Equations (NODEs), a category of neural network models that exploit the dynamics of differential equations for representation learning. Leveraging their foundation in differential equations, we illustrate the capability of these models to continuously process textual data, marking the first such model of its kind, and thereby proposing a promising direction for future research in this domain. The primary objective of this research is to propose a novel architecture for groups like healthcare that require the predictive capabilities of deep learning while emphasizing the importance of model transparency demonstrated in NODEs. 

**Abstract (ZH)**: 深度学习已成为机器学习中最重要的创新之一。然而，这一领域的显著局限在于其“黑箱”决策过程，这导致了像医疗和科学界这样的群体对其适用性的疑虑。为应对这一问题，本研究引入了一种基于神经常微分方程（NODEs）的可解释方法，这是一种利用微分方程动态进行表示学习的神经网络模型类别。借助微分方程的基础，我们展示了这些模型连续处理文本数据的能力，这是此类模型中的首创，从而为该领域的未来研究提出了一个有前景的方向。本研究的主要目标是为需要深度学习预测能力且强调模型透明度的医疗等群体提出一种新的架构。 

---
# The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models 

**Title (ZH)**: 细节决定一切：应对单模态虚假相关性以构建可泛化的多模态奖励模型 

**Authors**: Zichao Li, Xueru Wen, Jie Lou, Yuqiu Ji, Yaojie Lu, Xianpei Han, Debing Zhang, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.03122)  

**Abstract**: Multimodal Reward Models (MM-RMs) are crucial for aligning Large Language Models (LLMs) with human preferences, particularly as LLMs increasingly interact with multimodal data. However, we find that MM-RMs trained on existing datasets often struggle to generalize to out-of-distribution data due to their reliance on unimodal spurious correlations, primarily text-only shortcuts within the training distribution, which prevents them from leveraging true multimodal reward functions. To address this, we introduce a Shortcut-aware MM-RM learning algorithm that mitigates this issue by dynamically reweighting training samples, shifting the distribution toward better multimodal understanding, and reducing dependence on unimodal spurious correlations. Our experiments demonstrate significant improvements in generalization, downstream task performance, and scalability, establishing a more robust framework for multimodal reward modeling. 

**Abstract (ZH)**: 多模态奖励模型（MM-RMs）对于将大型语言模型（LLMs）与人类偏好对齐至关重要，尤其是在LLMs越来越多地与多模态数据互动时。然而，我们发现，现有数据集训练的MM-RMs往往难以泛化到分布外数据，这是因为它们依赖于单模态假相关，主要是训练分布中的文本-only捷径，这阻碍了它们利用真正的多模态奖励函数。为此，我们提出了一种意识捷径的MM-RM学习算法，通过动态重新加权训练样本、调整分布以提高多模态理解能力，并减少对单模态假相关的依赖，来解决这一问题。我们的实验展示了泛化能力、下游任务性能和可扩展性方面的显著提高，从而建立了一个更 robust的多模态奖励建模框架。 

---
# A Multimodal Framework for Topic Propagation Classification in Social Networks 

**Title (ZH)**: 多模态框架在社交媒体中的主题传播分类 

**Authors**: Yuchuan Jiang, Chaolong Jia, Yunyi Qin, Wei Cai, Yongsen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2503.03112)  

**Abstract**: The rapid proliferation of the Internet and the widespread adoption of social networks have significantly accelerated information dissemination. However, this transformation has introduced complexities in information capture and processing, posing substantial challenges for researchers and practitioners. Predicting the dissemination of topic-related information within social networks has thus become a critical research focus. This paper proposes a predictive model for topic dissemination in social networks by integrating multidimensional features derived from key dissemination characteristics. Specifically, we introduce two novel indicators, user relationship breadth and user authority, into the PageRank algorithm to quantify user influence more effectively. Additionally, we employ a Text-CNN model for sentiment classification, extracting sentiment features from textual content. Temporal embeddings of nodes are encoded using a Bi-LSTM model to capture temporal dynamics. Furthermore, we refine the measurement of user interaction traces with topics, replacing traditional topic view metrics with a more precise communication characteristics measure. Finally, we integrate the extracted multidimensional features using a Transformer model, significantly enhancing predictive performance. Experimental results demonstrate that our proposed model outperforms traditional machine learning and unimodal deep learning models in terms of FI-Score, AUC, and Recall, validating its effectiveness in predicting topic propagation within social networks. 

**Abstract (ZH)**: 互联网的迅速普及和社会网络的广泛应用极大地加速了信息传播。然而，这一转变增加了信息捕获和处理的复杂性，给研究人员和实践者带来了重大挑战。因此，预测社会网络中主题相关信息的传播成为了一个关键的研究focus。本文通过整合来自关键传播特征的多维度特征，提出了一种预测社会网络中主题传播的模型。具体而言，我们引入了两种新的指标——用户关系广度和用户权威性，以更有效地量化用户影响。此外，我们使用Text-CNN模型进行情感分类，从文本内容中提取情感特征。节点的时间嵌入使用Bi-LSTM模型进行编码，以捕获时间动态。同时，我们改进了用户与主题互动轨迹的度量，用更精确的通信特性度量替代了传统的主题视图指标。最后，我们使用Transformer模型整合提取的多维度特征，显著提高了预测性能。实验结果表明，我们提出的模型在FI-Score、AUC和Recall方面优于传统的机器学习和单模态深度学习模型，证实了其在预测社会网络传播方面的效果。 

---
# SoK: Knowledge is All You Need: Last Mile Delivery for Automated Provenance-based Intrusion Detection with LLMs 

**Title (ZH)**: SoK: 知识即一切：基于知识追溯的自动入侵检测的最后里程LLMs实现 

**Authors**: Wenrui Cheng, Tiantian Zhu, Chunlin Xiong, Haofei Sun, Zijun Wang, Shunan Jing, Mingqi Lv, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.03108)  

**Abstract**: Recently, provenance-based intrusion detection systems (PIDSes) have been widely proposed for endpoint threat analysis. However, due to the lack of systematic integration and utilization of knowledge, existing PIDSes still require significant manual intervention for practical deployment, making full automation challenging. This paper presents a disruptive innovation by categorizing PIDSes according to the types of knowledge they utilize. In response to the prevalent issue of ``knowledge silos problem'' in existing research, we introduce a novel knowledge-driven provenance-based intrusion detection framework, powered by large language models (LLMs). We also present OmniSec, a best practice system built upon this framework. By integrating attack representation knowledge, threat intelligence knowledge, and benign behavior knowledge, OmniSec outperforms the state-of-the-art approaches on public benchmark datasets. OmniSec is available online at this https URL. 

**Abstract (ZH)**: 基于知识驱动的来源导向入侵检测框架及OmniSec系统 

---
# External Reliable Information-enhanced Multimodal Contrastive Learning for Fake News Detection 

**Title (ZH)**: 基于外部可靠信息增强的多模态对比学习虚假新闻检测 

**Authors**: Biwei Cao, Qihang Wu, Jiuxin Cao, Bo Liu, Jie Gui  

**Link**: [PDF](https://arxiv.org/pdf/2503.03107)  

**Abstract**: With the rapid development of the Internet, the information dissemination paradigm has changed and the efficiency has been improved greatly. While this also brings the quick spread of fake news and leads to negative impacts on cyberspace. Currently, the information presentation formats have evolved gradually, with the news formats shifting from texts to multimodal contents. As a result, detecting multimodal fake news has become one of the research hotspots. However, multimodal fake news detection research field still faces two main challenges: the inability to fully and effectively utilize multimodal information for detection, and the low credibility or static nature of the introduced external information, which limits dynamic updates. To bridge the gaps, we propose ERIC-FND, an external reliable information-enhanced multimodal contrastive learning framework for fake news detection. ERIC-FND strengthens the representation of news contents by entity-enriched external information enhancement method. It also enriches the multimodal news information via multimodal semantic interaction method where the multimodal constrative learning is employed to make different modality representations learn from each other. Moreover, an adaptive fusion method is taken to integrate the news representations from different dimensions for the eventual classification. Experiments are done on two commonly used datasets in different languages, X (Twitter) and Weibo. Experiment results demonstrate that our proposed model ERIC-FND outperforms existing state-of-the-art fake news detection methods under the same settings. 

**Abstract (ZH)**: 基于外部可靠信息增强的多模态对比学习虚假新闻检测框架（ERIC-FND） 

---
# RVAFM: Re-parameterizing Vertical Attention Fusion Module for Handwritten Paragraph Text Recognition 

**Title (ZH)**: RVAFM：重新参数化的垂直注意力融合模块的手写段落文本识别 

**Authors**: Jinhui Zheng, Zhiquan Liu, Yain-Whar Si, Jianqing Li, Xinyuan Zhang, Xiaofan Li, Haozhi Huang, Xueyuan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.03104)  

**Abstract**: Handwritten Paragraph Text Recognition (HPTR) is a challenging task in Computer Vision, requiring the transformation of a paragraph text image, rich in handwritten text, into text encoding sequences. One of the most advanced models for this task is Vertical Attention Network (VAN), which utilizes a Vertical Attention Module (VAM) to implicitly segment paragraph text images into text lines, thereby reducing the difficulty of the recognition task. However, from a network structure perspective, VAM is a single-branch module, which is less effective in learning compared to multi-branch modules. In this paper, we propose a new module, named Re-parameterizing Vertical Attention Fusion Module (RVAFM), which incorporates structural re-parameterization techniques. RVAFM decouples the structure of the module during training and inference stages. During training, it uses a multi-branch structure for more effective learning, and during inference, it uses a single-branch structure for faster processing. The features learned by the multi-branch structure are fused into the single-branch structure through a special fusion method named Re-parameterization Fusion (RF) without any loss of information. As a result, we achieve a Character Error Rate (CER) of 4.44% and a Word Error Rate (WER) of 14.37% on the IAM paragraph-level test set. Additionally, the inference speed is slightly faster than VAN. 

**Abstract (ZH)**: 基于重构垂直注意力融合模块的便条手写段落文本识别 

---
# Hopfield Networks Meet Big Data: A Brain-Inspired Deep Learning Framework for Semantic Data Linking 

**Title (ZH)**: Hopfield网络遇见大数据：一种受脑启发的深度学习框架用于语义数据链接 

**Authors**: Ashwin Viswanathan Kannan, Johnson P Thomas, Abhimanyu Mukerji  

**Link**: [PDF](https://arxiv.org/pdf/2503.03084)  

**Abstract**: The exponential rise in data generation has led to vast, heterogeneous datasets crucial for predictive analytics and decision-making. Ensuring data quality and semantic integrity remains a challenge. This paper presents a brain-inspired distributed cognitive framework that integrates deep learning with Hopfield networks to identify and link semantically related attributes across datasets. Modeled on the dual-hemisphere functionality of the human brain, the right hemisphere assimilates new information while the left retrieves learned representations for association. Our architecture, implemented on MapReduce with Hadoop Distributed File System (HDFS), leverages deep Hopfield networks as an associative memory mechanism to enhance recall of frequently co-occurring attributes and dynamically adjust relationships based on evolving data patterns. Experiments show that associative imprints in Hopfield memory are reinforced over time, ensuring linked datasets remain contextually meaningful and improving data disambiguation and integration accuracy. Our results indicate that combining deep Hopfield networks with distributed cognitive processing offers a scalable, biologically inspired approach to managing complex data relationships in large-scale environments. 

**Abstract (ZH)**: 数据生成的指数级增长导致了预测分析和决策制定中至关重要的大量异质性数据集。确保数据质量和语义完整性仍是一项挑战。本文提出了一种受脑启发的分布式认知框架，将深度学习与霍普菲尔德网络结合，以跨数据集识别和链接语义相关属性。该框架借鉴了人脑双半球的功能，在右侧半球吸收新信息的同时，左侧半球检索已学习的表示进行关联。我们的架构基于MapReduce在Hadoop分布式文件系统（HDFS）上实现，利用深度霍普菲尔德网络作为联想记忆机制，增强频繁共现属性的回忆能力，并根据不断变化的数据模式动态调整关系。实验表明，霍普菲尔德记忆中的联想印记随时间增强，确保链接数据集保持上下文相关性，提高数据去混淆和集成准确性。我们的结果表明，将深度霍普菲尔德网络与分布式认知处理结合，提供了一种在大规模环境中管理复杂数据关系的可扩展、生物启发式方法。 

---
# Semi-Supervised In-Context Learning: A Baseline Study 

**Title (ZH)**: 半监督上下文学习： baseline 研究 

**Authors**: Zhengyao Gu, Henry Peng Zou, Yankai Chen, Aiwei Liu, Weizhi Zhang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.03062)  

**Abstract**: Most existing work in data selection for In-Context Learning (ICL) has focused on constructing demonstrations from ground truth annotations, with limited attention given to selecting reliable self-generated annotations. In this work, we propose a three-step semi-supervised ICL framework: annotation generation, demonstration selection, and semi-supervised inference. Our baseline, Naive-SemiICL, which prompts select high-confidence self-generated demonstrations for ICL prompting, outperforms a 16-shot baseline by an average of 9.94% across 16 datasets. We further introduce IterPSD, an annotation approach that refines pseudo-demonstrations iteratively, achieving up to 6.8% additional gains in classification tasks. Lastly, we reveal a scaling law for semi-supervised ICL, where models achieve optimal performance with over 1,000 demonstrations. 

**Abstract (ZH)**: 现有的大多数数据选择工作主要集中在从ground truth注释中构建示例，对选择可靠的自动生成注释关注不足。本文提出了一种三步半监督In-Context Learning (ICL)框架：注释生成、示例选择和半监督推理。我们的基线Naive-SemiICL通过提示选择高置信度的自动生成示例用于ICL提示，相比16-shot基线，在16个数据集上平均提升了9.94%。我们进一步引入了IterPSD迭代伪示例精炼方法，在分类任务中实现了高达6.8%的额外增益。最后，我们揭示了半监督ICL的扩展规律，即模型在超过1,000个示例时达到最优性能。 

---
# ArticuBot: Learning Universal Articulated Object Manipulation Policy via Large Scale Simulation 

**Title (ZH)**: ArticuBot: 通过大规模模拟学习通用articulated物体操作策略 

**Authors**: Yufei Wang, Ziyu Wang, Mino Nakura, Pratik Bhowal, Chia-Liang Kuo, Yi-Ting Chen, Zackory Erickson, David Held  

**Link**: [PDF](https://arxiv.org/pdf/2503.03045)  

**Abstract**: This paper presents ArticuBot, in which a single learned policy enables a robotics system to open diverse categories of unseen articulated objects in the real world. This task has long been challenging for robotics due to the large variations in the geometry, size, and articulation types of such objects. Our system, Articubot, consists of three parts: generating a large number of demonstrations in physics-based simulation, distilling all generated demonstrations into a point cloud-based neural policy via imitation learning, and performing zero-shot sim2real transfer to real robotics systems. Utilizing sampling-based grasping and motion planning, our demonstration generalization pipeline is fast and effective, generating a total of 42.3k demonstrations over 322 training articulated objects. For policy learning, we propose a novel hierarchical policy representation, in which the high-level policy learns the sub-goal for the end-effector, and the low-level policy learns how to move the end-effector conditioned on the predicted goal. We demonstrate that this hierarchical approach achieves much better object-level generalization compared to the non-hierarchical version. We further propose a novel weighted displacement model for the high-level policy that grounds the prediction into the existing 3D structure of the scene, outperforming alternative policy representations. We show that our learned policy can zero-shot transfer to three different real robot settings: a fixed table-top Franka arm across two different labs, and an X-Arm on a mobile base, opening multiple unseen articulated objects across two labs, real lounges, and kitchens. Videos and code can be found on our project website: this https URL. 

**Abstract (ZH)**: 本文介绍了ArticuBot，这是一种单一学习策略使机器人系统能够在现实世界中打开多样化类别的未见过的关节对象的方法。由于这类对象在几何形状、尺寸和关节类型上存在大量变化，因此机器人长期以来一直难以完成这项任务。我们的系统Articubot由三部分组成：在物理基础上的模拟中生成大量演示，通过模拟学习将所有生成的演示总结为基于点云的神经策略，以及执行零样本模拟到现实机器人系统的转移。利用基于采样的抓取和运动规划，我们的演示泛化管道快速且有效，共生成了42300个演示数据，涵盖了322个训练中的关节对象。在策略学习方面，我们提出了一种新型层次策略表示，在这种表示中，高层策略学习末端执行器的目标子项，而低层策略学习在预测目标条件下如何移动末端执行器。我们证明，这种层次方法在对象级别泛化方面显著优于非层次版本。此外，我们还为高层策略提出了一种新的加权位移模型，将预测与场景的现有3D结构联系起来，优于其他策略表示。我们展示了我们的学习策略能够在三个不同的现实机器人设置中实现零样本转移：在两个不同实验室中的固定桌面Franka手臂，以及移动基座上的X-Arm，打开两个实验室、真实休息区和厨房中的多个未见过的关节对象。更多视频和代码请参见我们的项目网站：this https URL。 

---
# SAGE: Steering and Refining Dialog Generation with State-Action Augmentation 

**Title (ZH)**: SAGE: 通过状态-动作增强引导和精炼对话生成 

**Authors**: Yizhe Zhang, Navdeep Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2503.03040)  

**Abstract**: Recent advances in large language models have demonstrated impressive capabilities in task-oriented applications, yet building emotionally intelligent chatbots that can engage in natural, strategic conversations remains a challenge. We present a novel approach called SAGE that uses latent variables to control long-horizon behavior in dialogue generation. At the core of our method is the State-Action Chain (SAC), which augments standard language model fine-tuning by introducing latent variables that encapsulate emotional states and conversational strategies between dialogue turns. During inference, these variables are generated before each response, enabling coarse-grained control over dialogue progression while maintaining natural interaction patterns. We also introduce a self-improvement pipeline that leverages dialogue tree search, LLM-based reward modeling, and targeted fine-tuning to optimize conversational trajectories. Our experimental results show that models trained with this approach demonstrate improved performance in emotional intelligence metrics while maintaining strong capabilities on LLM benchmarks. The discrete nature of our latent variables facilitates search-based strategies and provides a foundation for future applications of reinforcement learning to dialogue systems, where learning can occur at the state level rather than the token level. 

**Abstract (ZH)**: 最近大型语言模型的发展展示了其在任务导向应用中的强大能力，然而构建能够进行自然、策略性对话的具有情感智能的聊天机器人仍然面临挑战。我们提出了一种名为SAGE的新方法，通过潜在变量控制对话生成中的长时行为。该方法的核心是状态-动作链（SAC），它通过引入封装情感状态和对话策略的潜在变量来增强标准语言模型的微调。在推理过程中，这些变量在每次响应生成之前被生成，从而实现了对话进程的粗粒度控制，同时保持自然的交互模式。我们还引入了一种自我改进pipeline，通过对话树搜索、基于语言模型的奖励建模和目标导向微调来优化对话轨迹。我们的实验结果表明，采用这种方法训练的模型在情感智能指标上表现出改进的性能，同时在大型语言模型基准上保持了强大的能力。我们潜在变量的离散性质为基于强化学习的对话系统应用提供了基础，在这些应用中，学习可以在状态层面而非token层面进行。 

---
# LLM Misalignment via Adversarial RLHF Platforms 

**Title (ZH)**: LLM对齐偏差通过对抗性RLHF平台 

**Authors**: Erfan Entezami, Ali Naseh  

**Link**: [PDF](https://arxiv.org/pdf/2503.03039)  

**Abstract**: Reinforcement learning has shown remarkable performance in aligning language models with human preferences, leading to the rise of attention towards developing RLHF platforms. These platforms enable users to fine-tune models without requiring any expertise in developing complex machine learning algorithms. While these platforms offer useful features such as reward modeling and RLHF fine-tuning, their security and reliability remain largely unexplored. Given the growing adoption of RLHF and open-source RLHF frameworks, we investigate the trustworthiness of these systems and their potential impact on behavior of LLMs. In this paper, we present an attack targeting publicly available RLHF tools. In our proposed attack, an adversarial RLHF platform corrupts the LLM alignment process by selectively manipulating data samples in the preference dataset. In this scenario, when a user's task aligns with the attacker's objective, the platform manipulates a subset of the preference dataset that contains samples related to the attacker's target. This manipulation results in a corrupted reward model, which ultimately leads to the misalignment of the language model. Our results demonstrate that such an attack can effectively steer LLMs toward undesirable behaviors within the targeted domains. Our work highlights the critical need to explore the vulnerabilities of RLHF platforms and their potential to cause misalignment in LLMs during the RLHF fine-tuning process. 

**Abstract (ZH)**: 强化学习在使语言模型与人类偏好对齐方面展现了显著性能，推动了RLHF平台的发展关注。这些平台使用户能够在无需复杂机器学习算法开发专业知识的情况下对模型进行细调。尽管这些平台提供诸如奖励建模和RLHF细调等有用功能，但它们的安全性和可靠性仍鲜有研究。鉴于RLHF及其开源框架的日益普及，我们研究了这些系统的可信度及其对语言模型行为的影响。在本文中，我们提出了一个针对公开可用的RLHF工具的攻击。在我们提出的攻击中，一个 adversarial RLHF 平台通过有选择地操纵偏好数据集中的数据样本来破坏语言模型的对齐过程。在这种情况下，当用户的任务与攻击者的目标相一致时，平台会操纵包含与攻击者目标相关的样本的偏好数据集的一部分。这种操纵导致了被篡改的奖励模型，最终导致语言模型的对齐偏离。我们的结果表明，这种攻击可以在目标领域有效地引导语言模型表现出不良行为。我们的工作突显了探索RLHF平台漏洞以及它们在RLHF细调过程中导致语言模型对齐偏离的可能性的紧迫性。 

---
# One Model to Train them All: Hierarchical Self-Distillation for Enhanced Early Layer Embeddings 

**Title (ZH)**: 一种模型训练它们全部的方法：分层自我精炼以增强早期层嵌入 

**Authors**: Andrea Gurioli, Federico Pennino, João Monteiro, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2503.03008)  

**Abstract**: Deploying language models often requires handling model size vs. performance trade-offs to satisfy downstream latency constraints while preserving the model's usefulness. Model distillation is commonly employed to reduce model size while maintaining acceptable performance. However, distillation can be inefficient since it involves multiple training steps. In this work, we introduce MODULARSTARENCODER, a modular multi-exit encoder with 1B parameters, useful for multiple tasks within the scope of code retrieval. MODULARSTARENCODER is trained with a novel self-distillation mechanism that significantly improves lower-layer representations-allowing different portions of the model to be used while still maintaining a good trade-off in terms of performance. Our architecture focuses on enhancing text-to-code and code-to-code search by systematically capturing syntactic and semantic structures across multiple levels of representation. Specific encoder layers are targeted as exit heads, allowing higher layers to guide earlier layers during training. This self-distillation effect improves intermediate representations, increasing retrieval recall at no extra training cost. In addition to the multi-exit scheme, our approach integrates a repository-level contextual loss that maximally utilizes the training context window, further enhancing the learned representations. We also release a new dataset constructed via code translation, seamlessly expanding traditional text-to-code benchmarks with code-to-code pairs across diverse programming languages. Experimental results highlight the benefits of self-distillation through multi-exit supervision. 

**Abstract (ZH)**: 使用模块化多出口编码器在代码检索范围内实现多种任务时，平衡模型规模与性能的trade-offs以满足下游延迟约束并在保留模型效用的前提下，是一个常见的要求。模型蒸馏常被用来在保持可接受性能的同时减小模型规模。然而，蒸馏可能由于需要多步训练而耗时。在这项工作中，我们提出了MODULARSTARENCODER，这是一种具有1亿参数的模块化多出口编码器，适用于代码检索范围内的多种任务。我们通过一种新颖的自我蒸馏机制训练MODULARSTARENCODER，显著改进了下层表示，使模型的不同部分能够被使用，同时在性能方面保持良好的trade-off。我们架构的重点在于通过系统地捕捉多级表示中的语法和语义结构来增强从文本到代码和从代码到代码的搜索。特定编码器层被指定为出口头，允许较高层在训练中引导较低层。这种自我蒸馏效果增强了中间表示，提高了检索召回率且无需额外的训练成本。除了多出口方案，我们的方法整合了仓库级上下文损失，最大限度地利用训练上下文窗口，进一步增强学习到的表示。我们还发布了一个通过代码翻译构建的新数据集，无缝地将跨不同编程语言的代码到代码对扩展到传统文本到代码基准测试中。实验结果突出了多出口监督通过自我蒸馏带来的益处。 

---
# RAILGUN: A Unified Convolutional Policy for Multi-Agent Path Finding Across Different Environments and Tasks 

**Title (ZH)**: RAILGUN：跨不同环境和任务的统一卷积策略多-Agent路径规划 

**Authors**: Yimin Tang, Xiao Xiong, Jingyi Xi, Jiaoyang Li, Erdem Bıyık, Sven Koenig  

**Link**: [PDF](https://arxiv.org/pdf/2503.02992)  

**Abstract**: Multi-Agent Path Finding (MAPF), which focuses on finding collision-free paths for multiple robots, is crucial for applications ranging from aerial swarms to warehouse automation. Solving MAPF is NP-hard so learning-based approaches for MAPF have gained attention, particularly those leveraging deep neural networks. Nonetheless, despite the community's continued efforts, all learning-based MAPF planners still rely on decentralized planning due to variability in the number of agents and map sizes. We have developed the first centralized learning-based policy for MAPF problem called RAILGUN. RAILGUN is not an agent-based policy but a map-based policy. By leveraging a CNN-based architecture, RAILGUN can generalize across different maps and handle any number of agents. We collect trajectories from rule-based methods to train our model in a supervised way. In experiments, RAILGUN outperforms most baseline methods and demonstrates great zero-shot generalization capabilities on various tasks, maps and agent numbers that were not seen in the training dataset. 

**Abstract (ZH)**: RAILGUN：基于地图的多Agent路径规划的首个集中式学习策略 

---
# Effectively Steer LLM To Follow Preference via Building Confident Directions 

**Title (ZH)**: 通过构建自信方向有效地引导LLM遵循偏好 

**Authors**: Bingqing Song, Boran Han, Shuai Zhang, Hao Wang, Haoyang Fang, Bonan Min, Yuyang Wang, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.02989)  

**Abstract**: Having an LLM that aligns with human preferences is essential for accommodating individual needs, such as maintaining writing style or generating specific topics of interest. The majority of current alignment methods rely on fine-tuning or prompting, which can be either costly or difficult to control. Model steering algorithms, which modify the model output by constructing specific steering directions, are typically easy to implement and optimization-free. However, their capabilities are typically limited to steering the model into one of the two directions (i.e., bidirectional steering), and there has been no theoretical understanding to guarantee their performance. In this work, we propose a theoretical framework to understand and quantify the model steering methods. Inspired by the framework, we propose a confident direction steering method (CONFST) that steers LLMs via modifying their activations at inference time. More specifically, CONFST builds a confident direction that is closely aligned with users' preferences, and this direction is then added to the activations of the LLMs to effectively steer the model output. Our approach offers three key advantages over popular bidirectional model steering methods: 1) It is more powerful, since multiple (i.e. more than two) users' preferences can be aligned simultaneously; 2) It is simple to implement, since there is no need to determine which layer to add the steering vector to; 3) No explicit user instruction is required. We validate our method on GPT-2 XL (1.5B), Mistral (7B) and Gemma-it (9B) models for tasks that require shifting the output of LLMs across various topics and styles, achieving superior performance over competing methods. 

**Abstract (ZH)**: 具有与人类偏好一致的LLM对于满足个性化需求至关重要，例如保持writing style或生成特定感兴趣的主题。当前大多数对齐方法依赖于微调或提示，这可能是代价高昂的或难以控制。通过修改模型输出的模型偏向算法通常易于实现且无需优化。然而，它们的能力通常仅限于将模型偏向两个方向之一（即双向偏向），并且尚无理论理解以保证其性能。在本文中，我们提出了一个理论框架，以理解并量化模型偏向方法。受该框架的启发，我们提出了一种自信方向偏向方法（CONFST），该方法通过修改LLM推理时的激活值来偏向LLM。具体而言，CONFST构建了一个紧密与用户偏好一致的自信方向，并将该方向添加到LLM的激活值中，以有效偏向模型输出。我们的方法与流行的双向模型偏向方法相比具有三个关键优势：1）更为强大，因为它可以同时对齐多个（即超过两个）用户偏好；2）易于实现，因为无需确定添加偏向向量的层；3）无需显式用户指令。我们在GPT-2 XL（1.5B）、Mistral（7B）和Gemma-it（9B）模型上进行了验证，这些模型用于在不同主题和风格下调整LLM的输出任务，结果性能优于竞争对手的方法。 

---
# LINGOLY-TOO: Disentangling Memorisation from Reasoning with Linguistic Templatisation and Orthographic Obfuscation 

**Title (ZH)**: LINGOLY-TOO: 语言模板化与拼写混淆分离记忆与推理 

**Authors**: Jude Khouja, Karolina Korgul, Simi Hellsten, Lingyi Yang, Vlad Neacs, Harry Mayne, Ryan Kearns, Andrew Bean, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02972)  

**Abstract**: Effective evaluation of the reasoning capabilities of large language models (LLMs) are susceptible to overestimation due to data exposure of evaluation benchmarks. We introduce a framework for producing linguistic reasoning problems that reduces the effect of memorisation in model performance estimates and apply this framework to develop LINGOLY-TOO, a challenging evaluation benchmark for linguistic reasoning. By developing orthographic templates, we dynamically obfuscate the writing systems of real languages to generate numerous question variations. These variations preserve the reasoning steps required for each solution while reducing the likelihood of specific problem instances appearing in model training data. Our experiments demonstrate that frontier models, including OpenAI o1-preview and DeepSeem R1, struggle with advanced reasoning. Our analysis also shows that LLMs exhibit noticeable variance in accuracy across permutations of the same problem, and on average perform better on questions appearing in their original orthography. Our findings highlight the opaque nature of response generation in LLMs and provide evidence that prior data exposure contributes to overestimating the reasoning capabilities of frontier models. 

**Abstract (ZH)**: 有效的评估大规模语言模型的推理能力受到评估基准数据泄露的影响，可能导致高估。我们提出了一种生成语言推理问题的框架，以减少模型性能估计中的记忆效应，并应用该框架开发了LINGOLY-TOO，一个具有挑战性的语言推理评估基准。通过开发书写系统模板，我们动态地模糊真实语言的书写系统，生成大量问题变体。这些变体保留了解决每个问题所需的推理步骤，同时减少了特定问题实例出现在模型训练数据中的可能性。我们的实验表明，前沿模型，包括OpenAI o1-preview和DeepSeem R1，在高级推理方面存在困难。我们的分析还表明，大规模语言模型在相同问题的不同排列下表现出明显的准确率差异，并且通常在原始书写系统的问题上表现更好。我们的研究结果突显了大规模语言模型生成响应的不透明性质，并提供了先前数据暴露可能导致高估前沿模型推理能力的证据。 

---
# InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model 

**Title (ZH)**: InfiniSST：使用大型语言模型的同时无界口语翻译 

**Authors**: Siqi Ouyang, Xi Xu, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02969)  

**Abstract**: Simultaneous translation of unbounded streaming speech remains a challenging problem due to the need for effectively processing the history speech context and past translations so that quality and latency, including computation overhead, can be balanced. Most prior works assume pre-segmented speech, limiting their real-world applicability. In this paper, we propose InfiniSST, a novel approach that formulates SST as a multi-turn dialogue task, enabling seamless translation of unbounded speech. We construct translation trajectories and robust segments from MuST-C with multi-latency augmentation during training and develop a key-value (KV) cache management strategy to facilitate efficient inference. Experiments on MuST-C En-Es, En-De, and En-Zh demonstrate that InfiniSST reduces computation-aware latency by 0.5 to 1 second while maintaining the same translation quality compared to baselines. Ablation studies further validate the contributions of our data construction and cache management strategy. We release the code at this https URL 

**Abstract (ZH)**: 无限流语音的同步翻译由于需要有效处理历史语音上下文和过去翻译，以平衡质量和延迟（包括计算开销）仍然是一个挑战性问题。大多数先前工作假设分割好的语音，限制了其实用性。在本文中，我们提出了一种新颖的方法InfiniSST，将SST形式化为多轮对话任务，从而实现无缝翻译无限流语音。我们在训练过程中使用多延迟增强构建翻译轨迹和鲁棒片段，并开发了一种键值（KV）缓存管理策略以促进高效推理。实验结果表明，InfiniSST在保持与 baseline 相同翻译质量的同时，降低了0.5到1秒的计算感知延迟。消融研究进一步验证了我们数据构建和缓存管理策略的贡献。我们在该网址发布代码：this https URL。 

---
# Monocular visual simultaneous localization and mapping: (r)evolution from geometry to deep learning-based pipelines 

**Title (ZH)**: 单目视觉 simultaneous localization and mapping: (r)evolution from geometry to deep learning-based pipelines 

**Authors**: Olaya Alvarez-Tunon, Yury Brodskiy, Erdal Kayacan  

**Link**: [PDF](https://arxiv.org/pdf/2503.02955)  

**Abstract**: With the rise of deep learning, there is a fundamental change in visual SLAM algorithms toward developing different modules trained as end-to-end pipelines. However, regardless of the implementation domain, visual SLAM's performance is subject to diverse environmental challenges, such as dynamic elements in outdoor environments, harsh imaging conditions in underwater environments, or blurriness in high-speed setups. These environmental challenges need to be identified to study the real-world viability of SLAM implementations. Motivated by the aforementioned challenges, this paper surveys the current state of visual SLAM algorithms according to the two main frameworks: geometry-based and learning-based SLAM. First, we introduce a general formulation of the SLAM pipeline that includes most of the implementations in the literature. Second, those implementations are classified and surveyed for geometry and learning-based SLAM. After that, environment-specific challenges are formulated to enable experimental evaluation of the resilience of different visual SLAM classes to varying imaging conditions. We address two significant issues in surveying visual SLAM, providing (1) a consistent classification of visual SLAM pipelines and (2) a robust evaluation of their performance under different deployment conditions. Finally, we give our take on future opportunities for visual SLAM implementations. 

**Abstract (ZH)**: 深度学习兴起后，视觉SLAM算法朝着开发端到端管道的不同模块发生了根本性变化。然而，视觉SLAM的性能仍然受到各种环境挑战的影响，如户外环境中的动态元素、水下环境中的恶劣成像条件或高速设置中的模糊。识别这些环境挑战对于研究SLAM实现的现实可行性至关重要。受这些挑战的启发，本文根据几何基于和学习基于的SLAM两大框架，回顾现有的视觉SLAM算法。首先，我们介绍了一个包含文献中大多数实现的SLAM管道的通用公式。其次，我们将这些实现分类并回顾几何基于和学习基于的SLAM。之后，我们制定了环境特定的挑战，以使不同成像条件下的SLAM类别的鲁棒性得到实验评估。最后，我们对未来视觉SLAM实现的机会进行了展望。 

---
# Reliable and Efficient Multi-Agent Coordination via Graph Neural Network Variational Autoencoders 

**Title (ZH)**: 基于图神经网络变分自编码器的可靠高效多智能体协调 

**Authors**: Yue Meng, Nathalie Majcherczyk, Wenliang Liu, Scott Kiesel, Chuchu Fan, Federico Pecora  

**Link**: [PDF](https://arxiv.org/pdf/2503.02954)  

**Abstract**: Multi-agent coordination is crucial for reliable multi-robot navigation in shared spaces such as automated warehouses. In regions of dense robot traffic, local coordination methods may fail to find a deadlock-free solution. In these scenarios, it is appropriate to let a central unit generate a global schedule that decides the passing order of robots. However, the runtime of such centralized coordination methods increases significantly with the problem scale. In this paper, we propose to leverage Graph Neural Network Variational Autoencoders (GNN-VAE) to solve the multi-agent coordination problem at scale faster than through centralized optimization. We formulate the coordination problem as a graph problem and collect ground truth data using a Mixed-Integer Linear Program (MILP) solver. During training, our learning framework encodes good quality solutions of the graph problem into a latent space. At inference time, solution samples are decoded from the sampled latent variables, and the lowest-cost sample is selected for coordination. Finally, the feasible proposal with the highest performance index is selected for the deployment. By construction, our GNN-VAE framework returns solutions that always respect the constraints of the considered coordination problem. Numerical results show that our approach trained on small-scale problems can achieve high-quality solutions even for large-scale problems with 250 robots, being much faster than other baselines. Project page: this https URL 

**Abstract (ZH)**: 多Agent协调对于自动化仓库等共享空间中可靠多机器人导航至关重要。在密集机器人流量区域，局部协调方法可能无法找到无死锁的解决方案。在这种情况下，适当的做法是由中心单位生成一个全局调度，决定机器人的通行顺序。然而，随着问题规模的增大，集中式协调方法的运行时间会显著增加。本文提出利用Graph Neural Network Variational Autoencoders (GNN-VAE) 来更快地解决大规模多Agent协调问题，而不是通过集中式优化。我们将协调问题形式化为图问题，并使用混合整数线性规划（MILP）求解器收集真实数据。在训练过程中，我们的学习框架将高质量的图问题解决方案编码到潜在空间中。在推理阶段，从采样的潜在变量中解码解决方案样本，并选择成本最低的样本进行协调。最后，选择具有最高性能指标的可行建议进行部署。通过设计，我们的GNN-VAE框架返回的解决方案总是遵守所考虑的协调问题的约束。数值结果表明，即使在包含250个机器人的大规模问题上，我们的方法在小规模问题上训练后也能达到高质量的解决方案，比其他基准方法快得多。项目页面：this https URL。 

---
# KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding 

**Title (ZH)**: KodCode: 一个多样、具有挑战性且可验证的合成编码数据集 

**Authors**: Zhangchen Xu, Yang Liu, Yueqin Yin, Mingyuan Zhou, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2503.02951)  

**Abstract**: We introduce KodCode, a synthetic dataset that addresses the persistent challenge of acquiring high-quality, verifiable training data across diverse difficulties and domains for training Large Language Models for coding. Existing code-focused resources typically fail to ensure either the breadth of coverage (e.g., spanning simple coding tasks to advanced algorithmic problems) or verifiable correctness (e.g., unit tests). In contrast, KodCode comprises question-solution-test triplets that are systematically validated via a self-verification procedure. Our pipeline begins by synthesizing a broad range of coding questions, then generates solutions and test cases with additional attempts allocated to challenging problems. Finally, post-training data synthesis is done by rewriting questions into diverse formats and generating responses under a test-based reject sampling procedure from a reasoning model (DeepSeek R1). This pipeline yields a large-scale, robust and diverse coding dataset. KodCode is suitable for supervised fine-tuning and the paired unit tests also provide great potential for RL tuning. Fine-tuning experiments on coding benchmarks (HumanEval(+), MBPP(+), BigCodeBench, and LiveCodeBench) demonstrate that KodCode-tuned models achieve state-of-the-art performance, surpassing models like Qwen2.5-Coder-32B-Instruct and DeepSeek-R1-Distill-Llama-70B. 

**Abstract (ZH)**: KodCode：一种针对编码大型语言模型训练的合成数据集 

---
# Diverse Controllable Diffusion Policy with Signal Temporal Logic 

**Title (ZH)**: 多种可控扩散策略与信号时序逻辑 

**Authors**: Yue Meng, Chuchu fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.02924)  

**Abstract**: Generating realistic simulations is critical for autonomous system applications such as self-driving and human-robot interactions. However, driving simulators nowadays still have difficulty in generating controllable, diverse, and rule-compliant behaviors for road participants: Rule-based models cannot produce diverse behaviors and require careful tuning, whereas learning-based methods imitate the policy from data but are not designed to follow the rules explicitly. Besides, the real-world datasets are by nature "single-outcome", making the learning method hard to generate diverse behaviors. In this paper, we leverage Signal Temporal Logic (STL) and Diffusion Models to learn controllable, diverse, and rule-aware policy. We first calibrate the STL on the real-world data, then generate diverse synthetic data using trajectory optimization, and finally learn the rectified diffusion policy on the augmented dataset. We test on the NuScenes dataset and our approach can achieve the most diverse rule-compliant trajectories compared to other baselines, with a runtime 1/17X to the second-best approach. In the closed-loop testing, our approach reaches the highest diversity, rule satisfaction rate, and the least collision rate. Our method can generate varied characteristics conditional on different STL parameters in testing. A case study on human-robot encounter scenarios shows our approach can generate diverse and closed-to-oracle trajectories. The annotation tool, augmented dataset, and code are available at this https URL. 

**Abstract (ZH)**: 基于STL和扩散模型的可控、多样和守规行为学习方法 

---
# Straight-Line Diffusion Model for Efficient 3D Molecular Generation 

**Title (ZH)**: 直线扩散模型用于高效的3D分子生成 

**Authors**: Yuyan Ni, Shikun Feng, Haohan Chi, Bowen Zheng, Huan-ang Gao, Wei-Ying Ma, Zhi-Ming Ma, Yanyan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.02918)  

**Abstract**: Diffusion-based models have shown great promise in molecular generation but often require a large number of sampling steps to generate valid samples. In this paper, we introduce a novel Straight-Line Diffusion Model (SLDM) to tackle this problem, by formulating the diffusion process to follow a linear trajectory. The proposed process aligns well with the noise sensitivity characteristic of molecular structures and uniformly distributes reconstruction effort across the generative process, thus enhancing learning efficiency and efficacy. Consequently, SLDM achieves state-of-the-art performance on 3D molecule generation benchmarks, delivering a 100-fold improvement in sampling efficiency. Furthermore, experiments on toy data and image generation tasks validate the generality and robustness of SLDM, showcasing its potential across diverse generative modeling domains. 

**Abstract (ZH)**: 基于扩散的模型在分子生成中展现出了巨大的潜力，但通常需要大量的采样步骤来生成有效的样本。本文提出了一种新型的直线扩散模型（SLDM），通过将扩散过程设计为线性轨迹来解决这一问题。提出的进程与分子结构的噪声敏感特性相契合，并在生成过程中均匀分布重建努力，从而提高学习效率和效果。因此，SLDM 在3D分子生成基准测试中实现了最新的技术水平，采样效率提高了100倍。此外，对玩具数据和图像生成任务的实验验证了SLDM 的普遍性和鲁棒性，展示了其在多种生成建模领域的潜力。 

---
# Interpretable Few-Shot Retinal Disease Diagnosis with Concept-Guided Prompting of Vision-Language Models 

**Title (ZH)**: 概念引导提示的视觉语言模型在可解释的少量样本视网膜疾病诊断中的应用 

**Authors**: Deval Mehta, Yiwen Jiang, Catherine L Jan, Mingguang He, Kshitij Jadhav, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2503.02917)  

**Abstract**: Recent advancements in deep learning have shown significant potential for classifying retinal diseases using color fundus images. However, existing works predominantly rely exclusively on image data, lack interpretability in their diagnostic decisions, and treat medical professionals primarily as annotators for ground truth labeling. To fill this gap, we implement two key strategies: extracting interpretable concepts of retinal diseases using the knowledge base of GPT models and incorporating these concepts as a language component in prompt-learning to train vision-language (VL) models with both fundus images and their associated concepts. Our method not only improves retinal disease classification but also enriches few-shot and zero-shot detection (novel disease detection), while offering the added benefit of concept-based model interpretability. Our extensive evaluation across two diverse retinal fundus image datasets illustrates substantial performance gains in VL-model based few-shot methodologies through our concept integration approach, demonstrating an average improvement of approximately 5.8\% and 2.7\% mean average precision for 16-shot learning and zero-shot (novel class) detection respectively. Our method marks a pivotal step towards interpretable and efficient retinal disease recognition for real-world clinical applications. 

**Abstract (ZH)**: 近期深度学习的进展展示了利用彩色眼底图像分类视网膜疾病的巨大潜力。然而，现有工作主要依赖图像数据，诊断决策缺乏解释性，并且主要将医疗专业人员视为 ground truth 标注的标注者。为了填补这一空白，我们实施了两种关键策略：利用 GPT 模型的知识库提取视网膜疾病的可解释概念，并将这些概念作为语言组件纳入提示学习中，以训练结合眼底图像及其相关概念的视觉-语言（VL）模型。我们的方法不仅提高了视网膜疾病的分类性能，还丰富了少量样本和零样本检测（新型疾病检测），同时提供了基于概念的模型解释性。我们跨越两个不同视网膜眼底图像数据集的全面评估表明，通过我们的概念整合方法，VL 模型基于少量样本的方法在性能上取得了显著提升，分别在 16 射照学习和零样本（新类）检测中提高了约 5.8% 和 2.7% 的平均精度。我们的方法标志着朝着实际临床应用中可解释和高效的视网膜疾病识别迈出的关键一步。 

---
# Towards Robust Multi-UAV Collaboration: MARL with Noise-Resilient Communication and Attention Mechanisms 

**Title (ZH)**: 面向鲁棒多无人机协作：具有抗噪声通信和注意力机制的多智能体 reinforcement 学习 

**Authors**: Zilin Zhao, Chishui Chen, Haotian Shi, Jiale Chen, Xuanlin Yue, Zhejian Yang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02913)  

**Abstract**: Efficient path planning for unmanned aerial vehicles (UAVs) is crucial in remote sensing and information collection. As task scales expand, the cooperative deployment of multiple UAVs significantly improves information collection efficiency. However, collaborative communication and decision-making for multiple UAVs remain major challenges in path planning, especially in noisy environments. To efficiently accomplish complex information collection tasks in 3D space and address robust communication issues, we propose a multi-agent reinforcement learning (MARL) framework for UAV path planning based on the Counterfactual Multi-Agent Policy Gradients (COMA) algorithm. The framework incorporates attention mechanism-based UAV communication protocol and training-deployment system, significantly improving communication robustness and individual decision-making capabilities in noisy conditions. Experiments conducted on both synthetic and real-world datasets demonstrate that our method outperforms existing algorithms in terms of path planning efficiency and robustness, especially in noisy environments, achieving a 78\% improvement in entropy reduction. 

**Abstract (ZH)**: 基于Counterfactual Multi-Agent Policy Gradients的多 agent 强化学习框架下无人机路径规划 

---
# Text2Scenario: Text-Driven Scenario Generation for Autonomous Driving Test 

**Title (ZH)**: 文本2场景：基于文本的自动驾驶测试场景生成 

**Authors**: Xuan Cai, Xuesong Bai, Zhiyong Cui, Danmu Xie, Daocheng Fu, Haiyang Yu, Yilong Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.02911)  

**Abstract**: Autonomous driving (AD) testing constitutes a critical methodology for assessing performance benchmarks prior to product deployment. The creation of segmented scenarios within a simulated environment is acknowledged as a robust and effective strategy; however, the process of tailoring these scenarios often necessitates laborious and time-consuming manual efforts, thereby hindering the development and implementation of AD technologies. In response to this challenge, we introduce Text2Scenario, a framework that leverages a Large Language Model (LLM) to autonomously generate simulation test scenarios that closely align with user specifications, derived from their natural language inputs. Specifically, an LLM, equipped with a meticulously engineered input prompt scheme functions as a text parser for test scenario descriptions, extracting from a hierarchically organized scenario repository the components that most accurately reflect the user's preferences. Subsequently, by exploiting the precedence of scenario components, the process involves sequentially matching and linking scenario representations within a Domain Specific Language corpus, ultimately fabricating executable test scenarios. The experimental results demonstrate that such prompt engineering can meticulously extract the nuanced details of scenario elements embedded within various descriptive formats, with the majority of generated scenarios aligning closely with the user's initial expectations, allowing for the efficient and precise evaluation of diverse AD stacks void of the labor-intensive need for manual scenario configuration. Project page: this https URL. 

**Abstract (ZH)**: 自主驾驶（AD）测试构成了一种关键方法，用于在产品部署前评估性能基准。在模拟环境中创建分段场景被认可为一种 robust 和有效的策略；然而，这些场景的定制过程往往需要大量的手动劳动，从而阻碍了 AD 技术的发展与实施。针对这一挑战，我们引入了 Text2Scenario 框架，该框架利用大型语言模型（LLM）自动生成与用户自然语言输入高度一致的模拟测试场景。具体而言，一个配备有精心设计的输入提示方案的 LLM 作为测试场景描述的解析器，从层次化组织的场景库中提取最能反映用户偏好的组件。随后，通过利用场景组件的优先级，过程涉及顺序匹配和链接领域特定语言语料中的场景表示，最终生成可执行的测试场景。实验结果表明，这种提示工程可以从各种描述格式中一丝不苟地提取场景元素的细微之处，所生成的大部分场景与用户的初始期望高度一致，从而允许高效且精确地评估多种 AD 堆栈，而无需进行劳动密集型的手动场景配置。 

---
# Machine Learning Applications to Diffuse Reflectance Spectroscopy in Optical Diagnosis; A Systematic Review 

**Title (ZH)**: 机器学习在光学诊断中偏振反射光谱学中的应用：一项系统性回顾 

**Authors**: Nicola Rossberg, Celina L. Li, Simone Innocente, Stefan Andersson-Engels, Katarzyna Komolibus, Barry O'Sullivan, Andrea Visentin  

**Link**: [PDF](https://arxiv.org/pdf/2503.02905)  

**Abstract**: Diffuse Reflectance Spectroscopy has demonstrated a strong aptitude for identifying and differentiating biological tissues. However, the broadband and smooth nature of these signals require algorithmic processing, as they are often difficult for the human eye to distinguish. The implementation of machine learning models for this task has demonstrated high levels of diagnostic accuracies and led to a wide range of proposed methodologies for applications in various illnesses and conditions. In this systematic review, we summarise the state of the art of these applications, highlight current gaps in research and identify future directions. This review was conducted in accordance with the PRISMA guidelines. 77 studies were retrieved and in-depth analysis was conducted. It is concluded that diffuse reflectance spectroscopy and machine learning have strong potential for tissue differentiation in clinical applications, but more rigorous sample stratification in tandem with in-vivo validation and explainable algorithm development is required going forward. 

**Abstract (ZH)**: 弥散反射光谱技术在生物组织识别与分类中的应用已经展现了强大的能力，但由于这些信号具有宽带和平滑的特点，通常难以直接由人眼区分，因此需要算法处理。机器学习模型在此任务中的应用已经显示出高水平的诊断准确性，并提出了一系列应用于各种疾病和状况的方法。在本系统评价中，我们总结了这些应用的最新进展，指出了当前研究中的空白，并确定了未来的研究方向。该评价遵循PRISMA指南，共检索到77项研究，并进行了深入分析。研究结论认为，弥散反射光谱技术和机器学习在临床应用中具有强大的组织分类潜力，但未来需要更严格的样本分层、体内验证以及可解释算法的开发。 

---
# ClipGrader: Leveraging Vision-Language Models for Robust Label Quality Assessment in Object Detection 

**Title (ZH)**: ClipGrader: 利用视觉-语言模型进行稳健的目标检测标签质量评估 

**Authors**: Hong Lu, Yali Bian, Rahul C. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2503.02897)  

**Abstract**: High-quality annotations are essential for object detection models, but ensuring label accuracy - especially for bounding boxes - remains both challenging and costly. This paper introduces ClipGrader, a novel approach that leverages vision-language models to automatically assess the accuracy of bounding box annotations. By adapting CLIP (Contrastive Language-Image Pre-training) to evaluate both class label correctness and spatial precision of bounding box, ClipGrader offers an effective solution for grading object detection labels. Tested on modified object detection datasets with artificially disturbed bounding boxes, ClipGrader achieves 91% accuracy on COCO with a 1.8% false positive rate. Moreover, it maintains 87% accuracy with a 2.1% false positive rate when trained on just 10% of the COCO data. ClipGrader also scales effectively to larger datasets such as LVIS, achieving 79% accuracy across 1,203 classes. Our experiments demonstrate ClipGrader's ability to identify errors in existing COCO annotations, highlighting its potential for dataset refinement. When integrated into a semi-supervised object detection (SSOD) model, ClipGrader readily improves the pseudo label quality, helping achieve higher mAP (mean Average Precision) throughout the training process. ClipGrader thus provides a scalable AI-assisted tool for enhancing annotation quality control and verifying annotations in large-scale object detection datasets. 

**Abstract (ZH)**: ClipGrader：一种利用视觉语言模型自动评估边界框标注准确性的新颖方法 

---
# Adaptive Entanglement Routing with Deep Q-Networks in Quantum Networks 

**Title (ZH)**: 基于深度Q网络的量子网络自适应纠缠路由 

**Authors**: Lamarana Jallow, Majid Iqbal Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.02895)  

**Abstract**: The quantum internet holds transformative potential for global communication by harnessing the principles of quantum information processing. Despite significant advancements in quantum communication technologies, the efficient distribution of critical resources, such as qubits, remains a persistent and unresolved challenge. Conventional approaches often fall short of achieving optimal resource allocation, underscoring the necessity for more effective solutions. This study proposes a novel reinforcement learning-based adaptive entanglement routing framework designed to enable resource allocation tailored to the specific demands of quantum applications. The introduced QuDQN model utilizes reinforcement learning to optimize the management of quantum networks, allocate resources efficiently, and enhance entanglement routing. The model integrates key considerations, including fidelity requirements, network topology, qubit capacity, and request demands. 

**Abstract (ZH)**: 量子互联网通过利用量子信息处理原理，在全球通信方面展现出变革性的潜力。尽管在量子通信技术方面取得了显著进展，但有效分发关键资源（如量子比特）仍然是一个持久而未解决的挑战。传统方法往往无法实现最优资源分配，凸显了更有效解决方案的必要性。本研究提出了一种新型的基于强化学习的自适应纠缠路由框架，旨在实现针对量子应用特定需求的资源分配。引入的QuDQN模型利用强化学习优化量子网络的管理、高效分配资源并增强纠缠路由。该模型整合了信度要求、网络拓扑、量子比特容量和请求需求等关键考虑因素。 

---
# Predicting Cascade Failures in Interdependent Urban Infrastructure Networks 

**Title (ZH)**: 预测互依城市基础设施网络中的级联故障 

**Authors**: Yinzhou Tang, Jinghua Piao, Huandong Wang, Shaw Rajib, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02890)  

**Abstract**: Cascading failures (CF) entail component breakdowns spreading through infrastructure networks, causing system-wide collapse. Predicting CFs is of great importance for infrastructure stability and urban function. Despite extensive research on CFs in single networks such as electricity and road networks, interdependencies among diverse infrastructures remain overlooked, and capturing intra-infrastructure CF dynamics amid complex evolutions poses challenges. To address these gaps, we introduce the \textbf{I}ntegrated \textbf{I}nterdependent \textbf{I}nfrastructure CF model ($I^3$), designed to capture CF dynamics both within and across infrastructures. $I^3$ employs a dual GAE with global pooling for intra-infrastructure dynamics and a heterogeneous graph for inter-infrastructure interactions. An initial node enhancement pre-training strategy mitigates GCN-induced over-smoothing. Experiments demonstrate $I^3$ achieves a 31.94\% in terms of AUC, 18.03\% in terms of Precision, 29.17\% in terms of Recall, 22.73\% in terms of F1-score boost in predicting infrastructure failures, and a 28.52\% reduction in terms of RMSE for cascade volume forecasts compared to leading models. It accurately pinpoints phase transitions in interconnected and singular networks, rectifying biases in models tailored for singular networks. Access the code at this https URL. 

**Abstract (ZH)**: 集成相互依赖基础设施 cascading 失败模型（$I^3$）：捕捉基础设施内及之间的 cascading 失败动态 

---
# Function-Coherent Gambles with Non-Additive Sequential Dynamics 

**Title (ZH)**: 功能共轭赌度与非增量序列动力学 

**Authors**: Gregory Wheeler  

**Link**: [PDF](https://arxiv.org/pdf/2503.02889)  

**Abstract**: The desirable gambles framework provides a rigorous foundation for imprecise probability theory but relies heavily on linear utility via its coherence axioms. In our related work, we introduced function-coherent gambles to accommodate non-linear utility. However, when repeated gambles are played over time -- especially in intertemporal choice where rewards compound multiplicatively -- the standard additive combination axiom fails to capture the appropriate long-run evaluation. In this paper we extend the framework by relaxing the additive combination axiom and introducing a nonlinear combination operator that effectively aggregates repeated gambles in the log-domain. This operator preserves the time-average (geometric) growth rate and addresses the ergodicity problem. We prove the key algebraic properties of the operator, discuss its impact on coherence, risk assessment, and representation, and provide a series of illustrative examples. Our approach bridges the gap between expectation values and time averages and unifies normative theory with empirically observed non-stationary reward dynamics. 

**Abstract (ZH)**: 可Desirable Gamble框架提供了不精确概率理论的严谨基础，但依赖于通过共融公理体现的线性效用。在我们相关工作中，我们引入了函数共融赌注以容纳非线性效用。然而，在随着时间重复进行赌注，尤其是在跨时间选择中奖励呈乘性复合时，标准的加性组合公理无法捕捉到适当的长期评价。在本文中，我们通过放宽加性组合公理并引入一个在对数域中有效聚合重复赌注的非线性组合算子来扩展该框架。该算子保留了时间平均（几何）增长速率并解决了遍历性问题。我们证明了该算子的关键代数性质，讨论了其对共融、风险评估和表示的影响，并提供了系列说明性例子。我们的方法在期望值和时间平均之间架起桥梁，并将规范理论与实际观测到的非稳定奖励动态统一起来。 

---
# Interactive Debugging and Steering of Multi-Agent AI Systems 

**Title (ZH)**: 多人工智能系统交互式调试与引导 

**Authors**: Will Epperson, Gagan Bansal, Victor Dibia, Adam Fourney, Jack Gerrits, Erkang Zhu, Saleema Amershi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02068)  

**Abstract**: Fully autonomous teams of LLM-powered AI agents are emerging that collaborate to perform complex tasks for users. What challenges do developers face when trying to build and debug these AI agent teams? In formative interviews with five AI agent developers, we identify core challenges: difficulty reviewing long agent conversations to localize errors, lack of support in current tools for interactive debugging, and the need for tool support to iterate on agent configuration. Based on these needs, we developed an interactive multi-agent debugging tool, AGDebugger, with a UI for browsing and sending messages, the ability to edit and reset prior agent messages, and an overview visualization for navigating complex message histories. In a two-part user study with 14 participants, we identify common user strategies for steering agents and highlight the importance of interactive message resets for debugging. Our studies deepen understanding of interfaces for debugging increasingly important agentic workflows. 

**Abstract (ZH)**: 基于大语言模型的AI代理自主团队正在涌现，它们协作以完成复杂的用户任务。开发者在构建和调试这些AI代理团队时面临哪些挑战？通过对五位AI代理开发者进行形成性访谈，我们确定了核心挑战：难以审查长代理对话以定位错误、当前工具在交互式调试方面的支持不足，以及需要工具支持以迭代代理配置。基于这些需求，我们开发了一个交互式多代理调试工具AGDebugger，具有浏览和发送消息的UI、编辑和重置先前代理消息的能力，以及用于导航复杂消息历史的概览可视化视图。在两部分用户研究中，我们确定了用户的常见策略以引导代理，并强调了交互式消息重置对于调试的重要性。我们的研究深化了对调试日益重要的代理工作流界面的理解。 

---
# NeuroGauss4D-PCI: 4D Neural Fields and Gaussian Deformation Fields for Point Cloud Interpolation 

**Title (ZH)**: NeuroGauss4D-PCI: 4D神经场和高斯变形场的点云插值 

**Authors**: Chaokang Jiang, Dalong Du, Jiuming Liu, Siting Zhu, Zhenqiang Liu, Zhuang Ma, Zhujin Liang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2405.14241)  

**Abstract**: Point Cloud Interpolation confronts challenges from point sparsity, complex spatiotemporal dynamics, and the difficulty of deriving complete 3D point clouds from sparse temporal information. This paper presents NeuroGauss4D-PCI, which excels at modeling complex non-rigid deformations across varied dynamic scenes. The method begins with an iterative Gaussian cloud soft clustering module, offering structured temporal point cloud representations. The proposed temporal radial basis function Gaussian residual utilizes Gaussian parameter interpolation over time, enabling smooth parameter transitions and capturing temporal residuals of Gaussian distributions. Additionally, a 4D Gaussian deformation field tracks the evolution of these parameters, creating continuous spatiotemporal deformation fields. A 4D neural field transforms low-dimensional spatiotemporal coordinates ($x,y,z,t$) into a high-dimensional latent space. Finally, we adaptively and efficiently fuse the latent features from neural fields and the geometric features from Gaussian deformation fields. NeuroGauss4D-PCI outperforms existing methods in point cloud frame interpolation, delivering leading performance on both object-level (DHB) and large-scale autonomous driving datasets (NL-Drive), with scalability to auto-labeling and point cloud densification tasks. The source code is released at this https URL. 

**Abstract (ZH)**: 点云插值面临点稀疏性、复杂时空动态以及从稀疏时间信息推导完整3D点云的困难。本文提出NeuroGauss4D-PCI，该方法擅长建模各种动态场景下的复杂非刚性变形。该方法从迭代的高斯云软聚类模块开始，提供结构化的时空点云表示。所提出的时空径向基函数高斯残差使用时间上的高斯参数插值，实现平滑参数过渡并捕捉高斯分布的时空残差。此外，4D高斯变形场追踪这些参数的演变，生成连续的时空变形场。4D神经场将低维时空坐标（x, y, z, t）转换为高维潜在空间。最后，我们自适应且高效地融合来自神经场的潜在特征和来自高斯变形场的几何特征。NeuroGauss4D-PCI在点云帧插值任务中优于现有方法，在对象级别（DHB）和大规模自动驾驶数据集（NL-Drive）上表现出领先性能，并具有自动标注和点云稠密化任务的扩展性。源代码发布于此链接。 

---
