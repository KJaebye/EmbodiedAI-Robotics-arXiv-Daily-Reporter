# SPAR: Scalable LLM-based PDDL Domain Generation for Aerial Robotics 

**Title (ZH)**: SPAR: 基于大规模语言模型的可扩展PDDL领域生成方法在无人机机器人中的应用 

**Authors**: Songhao Huang, Yuwei Wu, Guangyao Shi, Gaurav S. Sukhatme, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.13691)  

**Abstract**: We investigate the problem of automatic domain generation for the Planning Domain Definition Language (PDDL) using Large Language Models (LLMs), with a particular focus on unmanned aerial vehicle (UAV) tasks. Although PDDL is a widely adopted standard in robotic planning, manually designing domains for diverse applications such as surveillance, delivery, and inspection is labor-intensive and error-prone, which hinders adoption and real-world deployment. To address these challenges, we propose SPAR, a framework that leverages the generative capabilities of LLMs to automatically produce valid, diverse, and semantically accurate PDDL domains from natural language input. To this end, we first introduce a systematically formulated and validated UAV planning dataset, consisting of ground-truth PDDL domains and associated problems, each paired with detailed domain and action descriptions. Building on this dataset, we design a prompting framework that generates high-quality PDDL domains from language input. The generated domains are evaluated through syntax validation, executability, feasibility, and interpretability. Overall, this work demonstrates that LLMs can substantially accelerate the creation of complex planning domains, providing a reproducible dataset and evaluation pipeline that enables application experts without prior experience to leverage it for practical tasks and advance future research in aerial robotics and automated planning. 

**Abstract (ZH)**: 使用大型语言模型自动生成Planning Domain Definition Language (PDDL)领域：面向无人驾驶航空器任务的研究 

---
# ASTREA: Introducing Agentic Intelligence for Orbital Thermal Autonomy 

**Title (ZH)**: ASTREA: 引入自主智能实现 orbital 热控自主性 

**Authors**: Alejandro D. Mousist  

**Link**: [PDF](https://arxiv.org/pdf/2509.13380)  

**Abstract**: This paper presents ASTREA, the first agentic system deployed on flight-heritage hardware (TRL 9) for autonomous spacecraft operations. Using thermal control as a representative use case, we integrate a resource-constrained Large Language Model (LLM) agent with a reinforcement learning controller in an asynchronous architecture tailored for space-qualified platforms. Ground experiments show that LLM-guided supervision improves thermal stability and reduces violations, confirming the feasibility of combining semantic reasoning with adaptive control under hardware constraints. However, on-orbit validation aboard the International Space Station (ISS) reveals performance degradation caused by inference latency mismatched with the rapid thermal cycles characteristic of Low Earth Orbit (LEO) satellites. These results highlight both the opportunities and current limitations of agentic LLM-based systems in real flight environments, providing practical design guidelines for future space autonomy. 

**Abstract (ZH)**: ASTREA：部署在飞行遗产硬件上的首个自主航天器操作代理系统 

---
# Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning 

**Title (ZH)**: 代理无人机：由LLM驱动的集成工具调用与认知推理自主性 

**Authors**: Anis Koubaa, Khaled Gabr  

**Link**: [PDF](https://arxiv.org/pdf/2509.13352)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly deployed in defense, surveillance, and disaster response, yet most systems remain confined to SAE Level 2--3 autonomy. Their reliance on rule-based control and narrow AI restricts adaptability in dynamic, uncertain missions. Existing UAV frameworks lack context-aware reasoning, autonomous decision-making, and ecosystem-level integration; critically, none leverage Large Language Model (LLM) agents with tool-calling for real-time knowledge access. This paper introduces the Agentic UAVs framework, a five-layer architecture (Perception, Reasoning, Action, Integration, Learning) that augments UAVs with LLM-driven reasoning, database querying, and third-party system interaction. A ROS2 and Gazebo-based prototype integrates YOLOv11 object detection with GPT-4 reasoning and local Gemma-3 deployment. In simulated search-and-rescue scenarios, agentic UAVs achieved higher detection confidence (0.79 vs. 0.72), improved person detection rates (91% vs. 75%), and markedly increased action recommendation (92% vs. 4.5%). These results confirm that modest computational overhead enables qualitatively new levels of autonomy and ecosystem integration. 

**Abstract (ZH)**: 基于大型语言模型的自主无人机框架 

---
# MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation 

**Title (ZH)**: MIRA: 基于MLLM的指令推荐技术赋能智能手机的一触即达AI服务 

**Authors**: Zhipeng Bian, Jieming Zhu, Xuyang Xie, Quanyu Dai, Zhou Zhao, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.13773)  

**Abstract**: The rapid advancement of generative AI technologies is driving the integration of diverse AI-powered services into smartphones, transforming how users interact with their devices. To simplify access to predefined AI services, this paper introduces MIRA, a pioneering framework for task instruction recommendation that enables intuitive one-touch AI tasking on smartphones. With MIRA, users can long-press on images or text objects to receive contextually relevant instruction recommendations for executing AI tasks. Our work introduces three key innovations: 1) A multimodal large language model (MLLM)-based recommendation pipeline with structured reasoning to extract key entities, infer user intent, and generate precise instructions; 2) A template-augmented reasoning mechanism that integrates high-level reasoning templates, enhancing task inference accuracy; 3) A prefix-tree-based constrained decoding strategy that restricts outputs to predefined instruction candidates, ensuring coherent and intent-aligned suggestions. Through evaluation using a real-world annotated datasets and a user study, MIRA has demonstrated substantial improvements in the accuracy of instruction recommendation. The encouraging results highlight MIRA's potential to revolutionize the way users engage with AI services on their smartphones, offering a more seamless and efficient experience. 

**Abstract (ZH)**: 快速发展的生成AI技术正推动多种AI驱动服务与智能手机的融合，改变用户与设备的交互方式。为简化对预定义AI服务的访问，本文介绍了一种名为MIRA的先驱框架，该框架能够使智能手机上的AI任务执行变得直观便捷，只需一键操作。通过长按图片或文本对象，用户可获得与上下文相关、适用于执行AI任务的指令推荐。我们的工作引入了三项关键创新：1）基于多模态大型语言模型（MLLM）的推荐管道，结合结构化推理以提取关键实体、推断用户意图并生成精确的指令；2）模板增强的推理机制，融合高级推理模板以提高任务推断准确性；3）基于前缀树的受限解码策略，限定输出为预定义指令候选，确保建议的连贯性和意图一致性。通过使用真实世界标注数据集和用户研究进行评估，MIRA在指令推荐的准确性方面取得了显著提升。令人鼓舞的结果表明，MIRA有望彻底改变用户在智能手机上与AI服务的互动方式，提供更流畅高效的体验。 

---
# THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning 

**Title (ZH)**: THOR: 工具集成的分层优化方法及其在数学推理中的应用via RL 

**Authors**: Qikai Chang, Zhenrong Zhang, Pengfei Hu, Jiefeng Ma, Yicheng Pan, Jianshu Zhang, Jun Du, Quan Liu, Jianqing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.13761)  

**Abstract**: Large Language Models (LLMs) have made remarkable progress in mathematical reasoning, but still continue to struggle with high-precision tasks like numerical computation and formal symbolic manipulation. Integrating external tools has emerged as a promising approach to bridge this gap. Despite recent advances, existing methods struggle with three key challenges: constructing tool-integrated reasoning data, performing fine-grained optimization, and enhancing inference. To overcome these limitations, we propose THOR (Tool-Integrated Hierarchical Optimization via RL). First, we introduce TIRGen, a multi-agent actor-critic-based pipeline for constructing high-quality datasets of tool-integrated reasoning paths, aligning with the policy and generalizing well across diverse models. Second, to perform fine-grained hierarchical optimization, we introduce an RL strategy that jointly optimizes for both trajectory-level problem solving and step-level code generation. This is motivated by our key insight that the success of an intermediate tool call is a strong predictor of the final answer's correctness. Finally, THOR incorporates a self-correction mechanism that leverages immediate tool feedback to dynamically revise erroneous reasoning paths during inference. Our approach demonstrates strong generalization across diverse models, performing effectively in both reasoning and non-reasoning models. It further achieves state-of-the-art performance for models of a similar scale on multiple mathematical benchmarks, while also delivering consistent improvements on code benchmarks. Our code will be publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学推理方面取得了显著进展，但仍难以处理高精度任务，如数值计算和正式符号操作。将外部工具集成起来已成为弥合这一差距的有前途的方法。尽管近期有所进展，现有方法仍面临三个关键挑战：构建工具集成推理数据、进行精细层级优化以及增强推理。为克服这些局限，我们提出了THOR（工具集成层级优化方法，基于强化学习）。首先，我们引入了TIRGen，一种基于多智能体演员-评论家架构的数据管道，用于构建高质量的工具集成推理路径数据集，与策略对齐并在多种模型中泛化良好。其次，为进行精细层级优化，我们引入了一种基于强化学习的策略，该策略同时优化轨迹级别的问题解决和步骤级别的代码生成。这受到我们一个关键见解的激发，即中间工具调用的成功是最终答案正确性的强大预测指标。最后，THOR 包含一个自我纠正机制，在推理过程中利用即时的工具反馈动态修订错误的推理路径。我们方法展示了在多种模型中强大的泛化能力，能够在推理和非推理模型中表现良好。此外，在多项数学基准测试中，THOR 达到了类似规模模型的最优性能，并且在代码基准测试中也实现了持续改进。相关代码将在此 https URL 公开。 

---
# Programmable Cognitive Bias in Social Agents 

**Title (ZH)**: 社会代理中的可编程认知偏见 

**Authors**: Xuan Liu, Haoyang Shang, Haojian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.13588)  

**Abstract**: This paper introduces CoBRA, a novel toolkit for systematically specifying agent behavior in LLM-based social simulation. We found that conventional approaches that specify agent behaviors through implicit natural language descriptions cannot yield consistent behaviors across models, and the produced agent behaviors do not capture the nuances of the descriptions. In contrast, CoBRA presents a new approach to program agents' cognitive biases explicitly, by grounding agents' expected behaviors using classic social science experiments. CoBRA has two components: (1) Cognitive Bias Index that measures the cognitive bias of a social agent, by quantifying the agent's reactions in a set of validated classical social science experiments; (2) Behavioral Regulation Engine that aligns the agent's behavior to demonstrate controlled cognitive bias. We evaluated CoBRA as an HCI toolkit through demonstration and technical benchmarks. Our results suggest that CoBRA can precisely program the cognitive bias demonstrated in a social agent in a model-agnostic manner. 

**Abstract (ZH)**: 本文介绍了一种新的工具CoBRA，用于在基于LLM的社会仿真中系统地规定代理行为。我们发现，通过隐式的自然语言描述来规定代理行为的传统方法无法在不同模型中产生一致的行为，并且生成的代理行为未能捕捉到描述的细微差别。相比之下，CoBRA通过将代理期望行为与经典的社会科学实验进行关联，明确地编程代理的认知偏差。CoBRA由两个部分组成：(1) 认知偏差指数，通过量化代理在一组验证过的经典社会科学实验中的反应来衡量代理的认知偏差；(2) 行为调节引擎，使代理的行为与可控的认知偏差相一致。我们通过演示和技术基准评估了CoBRA作为人机接口工具的效果。我们的结果显示，CoBRA可以以模型无关的方式精确编程社交代理所展现的认知偏差。 

---
# AI Agents with Human-Like Collaborative Tools: Adaptive Strategies for Enhanced Problem-Solving 

**Title (ZH)**: 具有人类协作工具特征的AI代理：增强问题解决的适应性策略 

**Authors**: Harper Reed, Michael Sugimura, Angelo Zangari  

**Link**: [PDF](https://arxiv.org/pdf/2509.13547)  

**Abstract**: We investigate whether giving LLM agents the collaborative tools and autonomy that humans naturally use for problem solving can improve their performance. We equip Claude Code agents with MCP-based social media and journaling tools and allow them to use these tools as they see fit. Across 34 Aider Polyglot Python programming challenges, collaborative tools substantially improve performance on the hardest problems, delivering 15-40% lower cost, 12-27% fewer turns, and 12-38% faster completion than baseline agents. Effects on the full challenge set are mixed, suggesting these tools act as performance enhancers when additional reasoning scaffolding is most needed. Surprisingly, Different models naturally adopted distinct collaborative strategies without explicit instruction. Sonnet 3.7 engaged broadly across tools and benefited from articulation-based cognitive scaffolding. Sonnet 4 showed selective adoption, leaning on journal-based semantic search when problems were genuinely difficult. This mirrors how human developers adjust collaboration based on expertise and task complexity. Behavioral analysis shows agents prefer writing over reading by about 2-9x, indicating that structured articulation drives much of the improvement rather than information access alone. Overall, AI agents can systematically benefit from human-inspired collaboration tools at the edge of their capabilities, pointing to adaptive collaborative interfaces as reasoning enhancers rather than universal efficiency boosts. 

**Abstract (ZH)**: 探究为LLM代理提供人类自然用于问题解决的合作工具和自主权是否能提高其性能 

---
# SteeringControl: Holistic Evaluation of Alignment Steering in LLMs 

**Title (ZH)**: steering控制：大型语言模型中对齐 steering 的整体评估 

**Authors**: Vincent Siu, Nicholas Crispino, David Park, Nathan W. Henry, Zhun Wang, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13450)  

**Abstract**: We introduce SteeringControl, a benchmark for evaluating representation steering methods across core alignment objectives--bias, harmful generation, and hallucination--and their effects on secondary behaviors such as sycophancy and commonsense morality. While prior alignment work often highlights truthfulness or reasoning ability to demonstrate the side effects of representation steering, we find there are many unexplored tradeoffs not yet understood in a systematic way. We collect a dataset of safety-relevant primary and secondary behaviors to evaluate steering effectiveness and behavioral entanglement centered around five popular steering methods. To enable this, we craft a modular steering framework based on unique components that serve as the building blocks of many existing methods. Our results on Qwen-2.5-7B and Llama-3.1-8B find that strong steering performance is dependent on the specific combination of steering method, model, and targeted behavior, and that severe concept entanglement can result from poor combinations of these three as well. We release our code here: this https URL. 

**Abstract (ZH)**: 我们介绍了SteeringControl，这是一个用于评估面向核心对齐目标（包括偏差、有害生成和幻觉）的表示对齐方法及其对奉承行为和常识道德等次要行为影响的基准。虽然之前的研究工作往往强调真实性或推理能力以展示表示对齐的副作用，但我们发现仍有许多未被系统探索的权衡关系。我们收集了一个与安全相关的主要和次要行为数据集，以评估基于五种流行对齐方法的对齐效果和行为纠缠。为此，我们基于独特的组件构建了一个模块化的对齐框架，这些组件是许多现有方法的构建块。我们在Qwen-2.5-7B和Llama-3.1-8B上的结果发现，强大的对齐性能依赖于特定的对齐方法、模型和目标行为的组合，而这些组合的不良匹配也会导致严重的概念纠缠。我们在这里发布了我们的代码：this https URL。 

---
# The Art of Saying "Maybe": A Conformal Lens for Uncertainty Benchmarking in VLMs 

**Title (ZH)**: 说出“可能”的艺术：面向VLMs不确定性基准测试的 conformal 视镜 

**Authors**: Asif Azad, Mohammad Sadat Hossain, MD Sadik Hossain Shanto, M Saifur Rahman, Md Rizwan Pervez  

**Link**: [PDF](https://arxiv.org/pdf/2509.13379)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable progress in complex visual understanding across scientific and reasoning tasks. While performance benchmarking has advanced our understanding of these capabilities, the critical dimension of uncertainty quantification has received insufficient attention. Therefore, unlike prior conformal prediction studies that focused on limited settings, we conduct a comprehensive uncertainty benchmarking study, evaluating 16 state-of-the-art VLMs (open and closed-source) across 6 multimodal datasets with 3 distinct scoring functions. Our findings demonstrate that larger models consistently exhibit better uncertainty quantification; models that know more also know better what they don't know. More certain models achieve higher accuracy, while mathematical and reasoning tasks elicit poorer uncertainty performance across all models compared to other domains. This work establishes a foundation for reliable uncertainty evaluation in multimodal systems. 

**Abstract (ZH)**: 视觉-语言模型在复杂视觉理解任务中的不确定性量化研究 

---
# $Agent^2$: An Agent-Generates-Agent Framework for Reinforcement Learning Automation 

**Title (ZH)**: $Agent^2$: 一种 reinforcement learning 自动化代理生成代理框架 

**Authors**: Yuan Wei, Xiaohan Shan, Ran Miao, Jianmin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.13368)  

**Abstract**: Reinforcement learning agent development traditionally requires extensive expertise and lengthy iterations, often resulting in high failure rates and limited accessibility. This paper introduces $Agent^2$, a novel agent-generates-agent framework that achieves fully automated RL agent design through intelligent LLM-driven generation. The system autonomously transforms natural language task descriptions and environment code into comprehensive, high-performance reinforcement learning solutions without human intervention. $Agent^2$ features a revolutionary dual-agent architecture. The Generator Agent serves as an autonomous AI designer that analyzes tasks and generates executable RL agents, while the Target Agent is the resulting automatically generated RL agent. The framework decomposes RL development into two distinct stages: MDP modeling and algorithmic optimization, enabling more targeted and effective agent generation. Built on the Model Context Protocol, $Agent^2$ provides a unified framework that standardizes intelligent agent creation across diverse environments and algorithms, while incorporating adaptive training management and intelligent feedback analysis for continuous improvement. Extensive experiments on a wide range of benchmarks, including MuJoCo, MetaDrive, MPE, and SMAC, demonstrate that $Agent^2$ consistently outperforms manually designed solutions across all tasks, achieving up to 55% performance improvement and substantial gains on average. By enabling truly end-to-end, closed-loop automation, this work establishes a new paradigm in which intelligent agents design and optimize other agents, marking a fundamental breakthrough for automated AI systems. 

**Abstract (ZH)**: Reinforcement Learning 代理开发传统上需要 extensive 专业知识和长时间迭代，导致高失败率和有限的可访问性。本文介绍了一种名为 $Agent^2$ 的新型代理生成代理框架，该框架通过智能大模型驱动生成实现完全自动化的强化学习代理设计。该系统自主地将自然语言任务描述和环境代码转换为全面的高性能强化学习解决方案，无需人工干预。$Agent^2$ 配备了革命性的双代理架构。生成代理作为自主 AI 设计师，分析任务并生成可执行的强化学习代理，而目标代理是自动生成的强化学习代理。该框架将强化学习开发分解为两个不同的阶段：马尔可夫决策过程建模和算法优化，从而实现更针对性和有效的代理生成。基于 Model Context 协议，$Agent^2$ 提供了一个统一的框架，标准化了不同环境和算法的智能代理创建，并结合了适应性训练管理和智能反馈分析，以实现持续改进。在 MuJoCo、MetaDrive、MPE 和 SMAC 等广泛基准上的大量实验表明，$Agent^2$ 在所有任务上均能持续优于手动设计的解决方案，平均性能提升高达 55%，并带来显著的平均收益。通过实现真正的端到端、闭环自动化，这项工作建立了智能代理设计和优化其他代理的新范式，标志着自动化 AI 系统的一个根本性突破。 

---
# Semantic Fusion with Fuzzy-Membership Features for Controllable Language Modelling 

**Title (ZH)**: 基于模糊成员hip特征的语义融合可控制语言建模 

**Authors**: Yongchao Huang, Hassan Raza  

**Link**: [PDF](https://arxiv.org/pdf/2509.13357)  

**Abstract**: We propose semantic fusion, a lightweight scheme that augments a Transformer language model (LM) with a parallel, fuzzy-membership feature channel that encodes token-level semantics. Each token is represented by a vector of interpretable features (e.g. part-of-speech cues, shallow roles, boundary flags, sentiment polarity and strength) whose values are graded degrees from differentiable membership functions (e.g. power kernels). These per-token vectors form a sentence-level semantic matrix fused via a gated adapter into the LM. Training uses standard next-token prediction, an auxiliary loss that reconstructs the semantic features from hidden states, and a lightweight uniformizer that regularizes adjective-class distributions. On a synthetic two-clause corpus with held-out adjectives for out-of-distribution (OOD) control, semantic fusion improves perplexity and enables precise, user-controllable generation of polarity and punctuation while maintaining model simplicity. This approach adds only small overhead, remains fully compatible with tied input-output embeddings, and provides an interpretable pathway for conditioned natural language generation. 

**Abstract (ZH)**: 我们提出了一种语义融合方法，这是一种轻量级方案，将Transformer语言模型（LM）与一个并行的模糊成员特征通道相结合，用于编码令牌级语义。每个令牌由一个可解释特征向量（例如，词性线索、浅层角色、边界标志、情感极性和强度）表示，这些特征值来自不同的可微隶属函数（例如，幂核）。这些令牌级别的向量形成了一个句级语义矩阵，通过门控适配器融合至LM中。训练使用标准的下一个令牌预测、一个辅助损失来重建隐藏状态中的语义特征，以及一个轻量级的规则化器来规整形容词类分布。在包含保留的形容词的合成双句语料库上，语义融合提高了困惑度，并使极性和标点的生成更加精确和用户可控，同时保持了模型的简洁性。这种方法仅增加了少量开销，完全兼容绑定的输入输出嵌入，并为条件自然语言生成提供了一条可解释的路径。 

---
# Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning 

**Title (ZH)**: 教大语言模型规划：符号规划的逻辑链式思考指令调优 

**Authors**: Pulkit Verma, Ngoc La, Anthony Favier, Swaroop Mishra, Julie A. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2509.13351)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, yet their ability to perform structured symbolic planning remains limited, particularly in domains requiring formal representations like the Planning Domain Definition Language (PDDL). In this paper, we present a novel instruction tuning framework, PDDL-Instruct, designed to enhance LLMs' symbolic planning capabilities through logical chain-of-thought reasoning. Our approach focuses on teaching models to rigorously reason about action applicability, state transitions, and plan validity using explicit logical inference steps. By developing instruction prompts that guide models through the precise logical reasoning required to determine when actions can be applied in a given state, we enable LLMs to self-correct their planning processes through structured reflection. The framework systematically builds verification skills by decomposing the planning process into explicit reasoning chains about precondition satisfaction, effect application, and invariant preservation. Experimental results on multiple planning domains show that our chain-of-thought reasoning based instruction-tuned models are significantly better at planning, achieving planning accuracy of up to 94% on standard benchmarks, representing a 66% absolute improvement over baseline models. This work bridges the gap between the general reasoning capabilities of LLMs and the logical precision required for automated planning, offering a promising direction for developing better AI planning systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种任务中展示了令人印象深刻的性能，但在执行结构化符号计划方面的能力仍然有限，特别是在需要正式表示的领域如规划领域定义语言（PDDL）。本文介绍了一种新颖的指令调优框架PDDL-Instruct，旨在通过逻辑链式推理来增强LLMs的符号计划能力。我们的方法专注于教导模型如何严格地推理动作的应用性、状态转换和计划的有效性，利用明确的逻辑推理步骤。通过开发指令提示引导模型进行精确的逻辑推理，以确定在给定状态下哪些动作可以应用，从而通过结构化的反思使LLMs能够自我纠正其规划过程。框架系统性地通过将规划过程分解为预条件满足、效果应用和不变性保持的明确推理链来构建验证技能。在多个规划领域的实验结果表明，基于链式推理的指令调优模型在规划方面的表现显著提高，标准基准上的规划准确率达到94%，相比基线模型绝对提高了66%。本文填补了LLMs的通用推理能力和自动化规划所需的逻辑精确性之间的差距，为开发更好的AI规划系统提供了有前景的方向。 

---
# FRIT: Using Causal Importance to Improve Chain-of-Thought Faithfulness 

**Title (ZH)**: FRIT: 使用因果重要性提高链式思维的可靠性 

**Authors**: Anand Swaroop, Akshat Nallani, Saksham Uboweja, Adiliia Uzdenova, Michael Nguyen, Kevin Zhu, Sunishchal Dev, Ashwinee Panda, Vasu Sharma, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2509.13334)  

**Abstract**: Chain-of-thought (CoT) reasoning has emerged as a powerful tool for improving large language model performance on complex tasks, but recent work shows that reasoning steps often fail to causally influence the final answer, creating brittle and untrustworthy outputs. Prior approaches focus primarily on measuring faithfulness, while methods for systematically improving it remain limited. We introduce Faithful Reasoning via Intervention Training (FRIT), a scalable alignment method that trains models to produce causally consistent reasoning by learning from systematically corrupted examples. FRIT generates synthetic training data by intervening on individual reasoning steps in model-generated CoTs, creating faithful/unfaithful pairs that highlight when reasoning breaks down. We then apply Direct Preference Optimization to teach models to prefer causally consistent reasoning paths. Evaluating on Qwen3-8B and Mistral-7B-v0.1 across factual and symbolic reasoning tasks, FRIT increases faithful reasoning by $3.4$ percentage points for Mistral on GSM8K while improving accuracy by $7.6$ percentage points. Our approach provides the first scalable, supervision-free method for training language models to produce more reliable and interpretable reasoning, addressing a critical gap between reasoning performance and trustworthiness. We release our code at \href{this https URL}. 

**Abstract (ZH)**: Faithful Reasoning via Intervention Training: A Scalable Method for Enhancing Causal Consistency in Large Language Models 

---
# Evaluation Awareness Scales Predictably in Open-Weights Large Language Models 

**Title (ZH)**: 评价意识量表可预测地适用于开放权重大型语言模型 

**Authors**: Maheep Chaudhary, Ian Su, Nikhil Hooda, Nishith Shankar, Julia Tan, Kevin Zhu, Ashwinee Panda, Ryan Lagasse, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.13333)  

**Abstract**: Large language models (LLMs) can internally distinguish between evaluation and deployment contexts, a behaviour known as \emph{evaluation awareness}. This undermines AI safety evaluations, as models may conceal dangerous capabilities during testing. Prior work demonstrated this in a single $70$B model, but the scaling relationship across model sizes remains unknown. We investigate evaluation awareness across $15$ models scaling from $0.27$B to $70$B parameters from four families using linear probing on steering vector activations. Our results reveal a clear power-law scaling: evaluation awareness increases predictably with model size. This scaling law enables forecasting deceptive behavior in future larger models and guides the design of scale-aware evaluation strategies for AI safety. A link to the implementation of this paper can be found at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）能够在内部区分评估和部署上下文，这种行为被称为“评估意识”。这一行为损害了AI安全性评估，因为模型在测试过程中可能会隐藏危险的功能。此前的研究已经在单一的70亿参数模型中证明了这一点，但不同规模模型之间的扩展关系仍不清楚。我们使用线性探针对控制向量激活进行了研究，调查了来自四个模型家族的15个从0.27亿到70亿参数的模型的评估意识。研究结果揭示了一种明确的幂律 scaling 规律：评估意识随模型规模的增加而可预测地增加。这种 scaling 规律能够帮助预测未来更大模型中的欺骗性行为，并指导AI安全性中的规模意识评估策略的设计。本文的实现链接为：这个 https URL。 

---
# Explicit Reasoning Makes Better Judges: A Systematic Study on Accuracy, Efficiency, and Robustness 

**Title (ZH)**: 明示推理使法官更优秀：关于准确度、效率和稳健性的系统研究 

**Authors**: Pratik Jayarao, Himanshu Gupta, Neeraj Varshney, Chaitanya Dwivedi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13332)  

**Abstract**: As Large Language Models (LLMs) are increasingly adopted as automated judges in benchmarking and reward modeling, ensuring their reliability, efficiency, and robustness has become critical. In this work, we present a systematic comparison of "thinking" and "non-thinking" LLMs in the LLM-as-a-judge paradigm using open-source Qwen 3 models of relatively small sizes (0.6B, 1.7B, and 4B parameters). We evaluate both accuracy and computational efficiency (FLOPs) on RewardBench tasks, and further examine augmentation strategies for non-thinking models, including in-context learning, rubric-guided judging, reference-based evaluation, and n-best aggregation. Our results show that despite these enhancements, non-thinking models generally fall short of their thinking counterparts. Our results show that thinking models achieve approximately 10% points higher accuracy with little overhead (under 2x), in contrast to augmentation strategies like few-shot learning, which deliver modest gains at a higher cost (>8x). Bias and robustness analyses further demonstrate that thinking models maintain significantly greater consistency under a variety of bias conditions such as positional, bandwagon, identity, diversity, and random biases (6% higher on average). We further extend our experiments to the multilingual setting and our results confirm that explicit reasoning extends its benefits beyond English. Overall, our work results in several important findings that provide systematic evidence that explicit reasoning offers clear advantages in the LLM-as-a-judge paradigm not only in accuracy and efficiency but also in robustness. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在基准测试和奖励建模中被越来越频繁地用作自动化法官，确保其可靠性和效率以及增强其稳健性变得至关重要。在本文中，我们使用开源的相对较小规模的Qwen 3模型（0.6B、1.7B和4B参数）对“思考型”和“非思考型”LLMs在LLM-as-a-judge范式下进行了系统的比较。我们在RewardBench任务中评估了准确性及计算效率（FLOPs），并进一步探讨了非思考模型的增强策略，包括上下文内学习、评分表指导判断、参考基评估和n-best聚合。结果显示，尽管进行了这些增强，非思考模型通常仍不及思考模型。思考模型的准确性大约高出10个百分点，且额外开销不到2倍，相比之下，如少样本学习等增强策略虽然成本较高（超过8倍），但仅带来微弱的增益。进一步的偏见和稳健性分析表明，思考模型在各种偏见条件（位置偏见、羊群效应偏见、身份偏见、多样性和随机偏见）下具有明显更高的一致性（平均高出6个百分点）。此外，我们将实验扩展至多语言环境，结果证实明确推理在英语之外的语言中也能够带来益处。总体而言，我们的工作得出几个重要结论，提供了系统性的证据表明，明确推理不仅在准确性及效率方面，在稳健性方面也提供了显而易见的优势。 

---
# Apertus: Democratizing Open and Compliant LLMs for Global Language Environments 

**Title (ZH)**: Apertus: 为全球语言环境民主化开放和合规的LLM 

**Authors**: Alejandro Hernández-Cano, Alexander Hägele, Allen Hao Huang, Angelika Romanou, Antoni-Joan Solergibert, Barna Pasztor, Bettina Messmer, Dhia Garbaya, Eduard Frank Ďurech, Ido Hakimi, Juan García Giraldo, Mete Ismayilzada, Negar Foroutan, Skander Moalla, Tiancheng Chen, Vinko Sabolčec, Yixuan Xu, Michael Aerni, Badr AlKhamissi, Ines Altemir Marinas, Mohammad Hossein Amani, Matin Ansaripour, Ilia Badanin, Harold Benoit, Emanuela Boros, Nicholas Browning, Fabian Bösch, Maximilian Böther, Niklas Canova, Camille Challier, Clement Charmillot, Jonathan Coles, Jan Deriu, Arnout Devos, Lukas Drescher, Daniil Dzenhaliou, Maud Ehrmann, Dongyang Fan, Simin Fan, Silin Gao, Miguel Gila, María Grandury, Diba Hashemi, Alexander Hoyle, Jiaming Jiang, Mark Klein, Andrei Kucharavy, Anastasiia Kucherenko, Frederike Lübeck, Roman Machacek, Theofilos Manitaras, Andreas Marfurt, Kyle Matoba, Simon Matrenok, Henrique Mendoncça, Fawzi Roberto Mohamed, Syrielle Montariol, Luca Mouchel, Sven Najem-Meyer, Jingwei Ni, Gennaro Oliva, Matteo Pagliardini, Elia Palme, Andrei Panferov, Léo Paoletti, Marco Passerini, Ivan Pavlov, Auguste Poiroux, Kaustubh Ponkshe, Nathan Ranchin, Javi Rando, Mathieu Sauser, Jakhongir Saydaliev, Muhammad Ali Sayfiddinov, Marian Schneider, Stefano Schuppli, Marco Scialanga, Andrei Semenov, Kumar Shridhar, Raghav Singhal, Anna Sotnikova, Alexander Sternfeld, Ayush Kumar Tarun, Paul Teiletche, Jannis Vamvas, Xiaozhe Yao, Hao Zhao Alexander Ilic, Ana Klimovic, Andreas Krause, Caglar Gulcehre, David Rosenthal, Elliott Ash, Florian Tramèr, Joost VandeVondele, Livio Veraldi, Martin Rajman, Thomas Schulthess, Torsten Hoefler, Antoine Bosselut, Martin Jaggi, Imanol Schlag  

**Link**: [PDF](https://arxiv.org/pdf/2509.14233)  

**Abstract**: We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting this http URL exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension. 

**Abstract (ZH)**: 我们提出Apertus，一个完全开源的大语言模型套件，旨在解决当今开放模型生态系统中的两大系统性问题：数据合规和多语言表示。与许多先前提供权重而不具备可复现的数据管道或尊重内容所有者权益的模型不同，Apertus模型仅在公开可用的数据上进行预训练， retroactively 尊重了特定排除要求，并过滤掉了非许可、有毒和个人可识别信息的内容。为减轻记忆风险，在预训练过程中我们采用了Goldfish目标，强烈抑制数据的原样回忆，同时保留下游任务性能。Apertus模型还扩展了多语言覆盖面，训练数据来自超过1800种语言，其中约40%的预训练数据用于非英语内容。在8B和70B规模下发布时，Apertus在多语言基准测试中接近最先进成果，与开放权重的同类模型相当或超越之。除了模型权重，我们还以宽松许可发布了整个开发周期中的所有科学研究成果，包括数据准备脚本、检查点、评估套件和训练代码，以实现透明审查和扩展。 

---
# Language models' activations linearly encode training-order recency 

**Title (ZH)**: 语言模型的激活按照训练顺序 recentness 线性编码 

**Authors**: Dmitrii Krasheninnikov, Richard E. Turner, David Krueger  

**Link**: [PDF](https://arxiv.org/pdf/2509.14223)  

**Abstract**: We show that language models' activations linearly encode when information was learned during training. Our setup involves creating a model with a known training order by sequentially fine-tuning Llama-3.2-1B on six disjoint but otherwise similar datasets about named entities. We find that the average activations of test samples for the six training datasets encode the training order: when projected into a 2D subspace, these centroids are arranged exactly in the order of training and lie on a straight line. Further, we show that linear probes can accurately (~90%) distinguish "early" vs. "late" entities, generalizing to entities unseen during the probes' own training. The model can also be fine-tuned to explicitly report an unseen entity's training stage (~80% accuracy). Interestingly, this temporal signal does not seem attributable to simple differences in activation magnitudes, losses, or model confidence. Our paper demonstrates that models are capable of differentiating information by its acquisition time, and carries significant implications for how they might manage conflicting data and respond to knowledge modifications. 

**Abstract (ZH)**: 语言模型的激活在训练过程中线性编码了信息学习的顺序 

---
# A Universal Banach--Bregman Framework for Stochastic Iterations: Unifying Stochastic Mirror Descent, Learning and LLM Training 

**Title (ZH)**: 一类适用于随机迭代的通用Banach-Bregman框架：统一随机镜像下降、学习和大模型训练 

**Authors**: Johnny R. Zhang, Xiaomei Mi, Gaoyuan Du, Qianyi Sun, Shiqi Wang, Jiaxuan Li, Wenhua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.14216)  

**Abstract**: Stochastic optimization powers the scalability of modern artificial intelligence, spanning machine learning, deep learning, reinforcement learning, and large language model training. Yet, existing theory remains largely confined to Hilbert spaces, relying on inner-product frameworks and orthogonality. This paradigm fails to capture non-Euclidean settings, such as mirror descent on simplices, Bregman proximal methods for sparse learning, natural gradient descent in information geometry, or Kullback--Leibler-regularized language model training. Unlike Euclidean-based Hilbert-space methods, this approach embraces general Banach spaces. This work introduces a pioneering Banach--Bregman framework for stochastic iterations, establishing Bregman geometry as a foundation for next-generation optimization. It (i) provides a unified template via Bregman projections and Bregman--Fejer monotonicity, encompassing stochastic approximation, mirror descent, natural gradient, adaptive methods, and mirror-prox; (ii) establishes super-relaxations ($\lambda > 2$) in non-Hilbert settings, enabling flexible geometries and elucidating their acceleration effect; and (iii) delivers convergence theorems spanning almost-sure boundedness to geometric rates, validated on synthetic and real-world tasks. Empirical studies across machine learning (UCI benchmarks), deep learning (e.g., Transformer training), reinforcement learning (actor--critic), and large language models (WikiText-2 with distilGPT-2) show up to 20% faster convergence, reduced variance, and enhanced accuracy over classical baselines. These results position Banach--Bregman geometry as a cornerstone unifying optimization theory and practice across core AI paradigms. 

**Abstract (ZH)**: Banach-Bregman框架赋能现代人工智能的可扩展性：超越欧几里得空间的随机优化理论 

---
# Synthesizing Behaviorally-Grounded Reasoning Chains: A Data-Generation Framework for Personal Finance LLMs 

**Title (ZH)**: 基于行为grounded推理链合成：个人金融LLM的数据生成框架 

**Authors**: Akhil Theerthala  

**Link**: [PDF](https://arxiv.org/pdf/2509.14180)  

**Abstract**: Personalized financial advice requires consideration of user goals, constraints, risk tolerance, and jurisdiction. Prior LLM work has focused on support systems for investors and financial planners. Simultaneously, numerous recent studies examine broader personal finance tasks, including budgeting, debt management, retirement, and estate planning, through agentic pipelines that incur high maintenance costs, yielding less than 25% of their expected financial returns. In this study, we introduce a novel and reproducible framework that integrates relevant financial context with behavioral finance studies to construct supervision data for end-to-end advisors. Using this framework, we create a 19k sample reasoning dataset and conduct a comprehensive fine-tuning of the Qwen-3-8B model on the dataset. Through a held-out test split and a blind LLM-jury study, we demonstrate that through careful data curation and behavioral integration, our 8B model achieves performance comparable to significantly larger baselines (14-32B parameters) across factual accuracy, fluency, and personalization metrics while incurring 80% lower costs than the larger counterparts. 

**Abstract (ZH)**: 个性化财务建议需要考虑用户目标、约束、风险承受能力和管辖地。先前的LLM研究主要集中在投资者和财务规划者的支持系统上。同时，许多近期研究表明，通过代理管道来探讨更广泛的个人财务管理任务（包括预算、债务管理、退休和遗产规划），尽管维护成本高昂，但仅能实现预期财务回报的不到25%。在本研究中，我们介绍了一种新颖且可复制的框架，该框架将相关财务背景与行为金融研究结合起来构建端到端顾问的监督数据。利用该框架，我们创建了一个包含19,000个样本的推理数据集，并在数据集上对Qwen-3-8B模型进行了全面微调。通过保留测试拆分和盲测LLM法官研究，我们证明，通过细致的数据整理和行为整合，我们的8B模型在事实准确性、流畅性和个性化指标上达到了与显著更大的基准模型（14-32B参数）相当的性能，同时成本降低了80%。 

---
# Reasoning Efficiently Through Adaptive Chain-of-Thought Compression: A Self-Optimizing Framework 

**Title (ZH)**: 自优化框架下通过自适应链式思考压缩高效推理 

**Authors**: Kerui Huang, Shuhan Liu, Xing Hu, Tongtong Xu, Lingfeng Bao, Xin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.14093)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances Large Language Models (LLMs) by prompting intermediate steps, improving accuracy and robustness in arithmetic, logic, and commonsense tasks. However, this benefit comes with high computational costs: longer outputs increase latency, memory usage, and KV-cache demands. These issues are especially critical in software engineering tasks where concise and deterministic outputs are required. To investigate these trade-offs, we conduct an empirical study based on code generation benchmarks. The results reveal that longer CoT does not always help. Excessive reasoning often causes truncation, accuracy drops, and latency up to five times higher, with failed outputs consistently longer than successful ones. These findings challenge the assumption that longer reasoning is inherently better and highlight the need for adaptive CoT control. Motivated by this, we propose SEER (Self-Enhancing Efficient Reasoning), an adaptive framework that compresses CoT while preserving accuracy. SEER combines Best-of-N sampling with task-aware adaptive filtering, dynamically adjusting thresholds based on pre-inference outputs to reduce verbosity and computational overhead. We then evaluate SEER on three software engineering tasks and one math task. On average, SEER shortens CoT by 42.1%, improves accuracy by reducing truncation, and eliminates most infinite loops. These results demonstrate SEER as a practical method to make CoT-enhanced LLMs more efficient and robust, even under resource constraints. 

**Abstract (ZH)**: Chain-of-Thought推理提高大型语言模型的效率与适应性：基于软件工程任务的实证研究与自我增强高效推理框架SEER 

---
# Hala Technical Report: Building Arabic-Centric Instruction & Translation Models at Scale 

**Title (ZH)**: 哈拉技术报告：构建面向阿拉伯语的规模化指令与翻译模型 

**Authors**: Hasan Abed Al Kader Hammoud, Mohammad Zbeeb, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2509.14008)  

**Abstract**: We present Hala, a family of Arabic-centric instruction and translation models built with our translate-and-tune pipeline. We first compress a strong AR$\leftrightarrow$EN teacher to FP8 (yielding $\sim$2$\times$ higher throughput with no quality loss) and use it to create high-fidelity bilingual supervision. A lightweight language model LFM2-1.2B is then fine-tuned on this data and used to translate high-quality English instruction sets into Arabic, producing a million-scale corpus tailored to instruction following. We train Hala models at 350M, 700M, 1.2B, and 9B parameters, and apply slerp merging to balance Arabic specialization with base-model strengths. On Arabic-centric benchmarks, Hala achieves state-of-the-art results within both the "nano" ($\leq$2B) and "small" (7-9B) categories, outperforming their bases. We release models, data, evaluation, and recipes to accelerate research in Arabic NLP. 

**Abstract (ZH)**: 我们介绍了Hala，一种基于我们的翻译和調整管道构建的阿拉伯语中心指令与翻译模型系列。我们首先将一个强大的AR$\leftrightarrow$EN老师压缩到FP8（在没有质量损失的情况下实现了约2倍的吞吐量提升），并使用它创建高质量的双语监督。然后，使用这种数据对轻量级语言模型LFM2-1.2B进行微调，并将其用于将高质量的英语指令集翻译成阿拉伯语，生成一个针对指令跟随优化的百万规模语料库。我们训练了参数量为350M、700M、1.2B和9B的Hala模型，并应用slerp合并方法来平衡阿拉伯语专业化与基础模型的优势。在阿拉伯语中心基准测试中，Hala在“nano”（$\leq$2B）和“small”（7-9B）类别中都取得了最先进的成果，优于其基础模型。我们发布了模型、数据、评估和食谱，以加速阿拉伯语NLP研究。 

---
# Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency 

**Title (ZH)**: Slim-SC: 基于自我一致性的心流剪枝以实现高效扩展 

**Authors**: Colin Hong, Xu Guo, Anand Chaanan Singh, Esha Choukse, Dmitrii Ustiugov  

**Link**: [PDF](https://arxiv.org/pdf/2509.13990)  

**Abstract**: Recently, Test-Time Scaling (TTS) has gained increasing attention for improving LLM reasoning performance at test time without retraining the model. A notable TTS technique is Self-Consistency (SC), which generates multiple reasoning chains in parallel and selects the final answer via majority voting. While effective, the order-of-magnitude computational overhead limits its broad deployment. Prior attempts to accelerate SC mainly rely on model-based confidence scores or heuristics with limited empirical support. For the first time, we theoretically and empirically analyze the inefficiencies of SC and reveal actionable opportunities for improvement. Building on these insights, we propose Slim-SC, a step-wise pruning strategy that identifies and removes redundant chains using inter-chain similarity at the thought level. Experiments on three STEM reasoning datasets and two recent LLM architectures show that Slim-SC reduces inference latency and KVC usage by up to 45% and 26%, respectively, with R1-Distill, while maintaining or improving accuracy, thus offering a simple yet efficient TTS alternative for SC. 

**Abstract (ZH)**: Recently, Test-Time Scaling (TTS) 已逐渐受到关注，通过在测试时不重新训练模型来提高大语言模型推理性能，而无需重新训练模型。值得注意的一种 TTS 技术是自一致性 (SC)，它并行生成多个推理链并最终通过多数投票选择答案。尽管有效，但其量级的计算开销限制了其广泛的部署。先前加速 SC 的尝试主要依赖于基于模型的置信分数或缺乏实证支持的经验法则。我们首次从理论上和实证上分析了 SC 的低效性，并揭示了改进的可行机会。基于这些见解，我们提出了 Slim-SC，这是一种逐步剪枝策略，通过链间相似性在思考层面识别并移除冗余链。在三个 STEM 推理数据集和两种最近的大语言模型架构上的实验表明，相比于 SC，通过 R1-Distill 使用 Slim-SC 可将推理延迟和 KVC 使用量分别减少最多 45% 和 26%，同时保持或提高准确性，从而为 SC 提供了一个简单而有效的 TTS 选择。 

---
# LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology 

**Title (ZH)**: LLM代理在交互工作流程溯源中的应用：参考架构与评估方法 

**Authors**: Renan Souza, Timothy Poteet, Brian Etz, Daniel Rosendo, Amal Gueroudji, Woong Shin, Prasanna Balaprakash, Rafael Ferreira da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2509.13978)  

**Abstract**: Modern scientific discovery increasingly relies on workflows that process data across the Edge, Cloud, and High Performance Computing (HPC) continuum. Comprehensive and in-depth analyses of these data are critical for hypothesis validation, anomaly detection, reproducibility, and impactful findings. Although workflow provenance techniques support such analyses, at large scale, the provenance data become complex and difficult to analyze. Existing systems depend on custom scripts, structured queries, or static dashboards, limiting data interaction. In this work, we introduce an evaluation methodology, reference architecture, and open-source implementation that leverages interactive Large Language Model (LLM) agents for runtime data analysis. Our approach uses a lightweight, metadata-driven design that translates natural language into structured provenance queries. Evaluations across LLaMA, GPT, Gemini, and Claude, covering diverse query classes and a real-world chemistry workflow, show that modular design, prompt tuning, and Retrieval-Augmented Generation (RAG) enable accurate and insightful LLM agent responses beyond recorded provenance. 

**Abstract (ZH)**: 现代科学发现 increasingly relies on 工作流处理跨边缘、云和高性能计算（HPC） continuum 中的数据。对这些数据进行全面和深入的分析对于假设验证、异常检测、可重复性和具有影响力的研究结果至关重要。尽管工作流追溯技术支持这些分析，但在大规模情况下，追溯数据变得复杂且难以分析。现有系统依赖于自定义脚本、结构化查询或静态仪表板，限制了数据交互。在本工作中，我们介绍了一种评估方法、参考架构和开源实现，利用交互式大型语言模型（LLM）代理进行运行时数据分析。我们的方法采用轻量级的、元数据驱动的设计，将自然语言转换为结构化追溯查询。跨越 LLAMA、GPT、Gemini 和 Claude 的评估，涵盖了多种查询类别和一个实际化学工作流，表明模块化设计、提示调整和检索增强生成（RAG）能够实现超越记录追溯的准确且富有洞察力的 LLM 代理响应。 

---
# An Empirical Study on Failures in Automated Issue Solving 

**Title (ZH)**: 自动问题解决中故障的实证研究 

**Authors**: Simiao Liu, Fang Liu, Liehao Li, Xin Tan, Yinghao Zhu, Xiaoli Lian, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13941)  

**Abstract**: Automated issue solving seeks to autonomously identify and repair defective code snippets across an entire codebase. SWE-Bench has emerged as the most widely adopted benchmark for evaluating progress in this area. While LLM-based agentic tools show great promise, they still fail on a substantial portion of tasks. Moreover, current evaluations primarily report aggregate issue-solving rates, which obscure the underlying causes of success and failure, making it challenging to diagnose model weaknesses or guide targeted improvements. To bridge this gap, we first analyze the performance and efficiency of three SOTA tools, spanning both pipeline-based and agentic architectures, in automated issue solving tasks of SWE-Bench-Verified under varying task characteristics. Furthermore, to move from high-level performance metrics to underlying cause analysis, we conducted a systematic manual analysis of 150 failed instances. From this analysis, we developed a comprehensive taxonomy of failure modes comprising 3 primary phases, 9 main categories, and 25 fine-grained subcategories. Then we systematically analyze the distribution of the identified failure modes, the results reveal distinct failure fingerprints between the two architectural paradigms, with the majority of agentic failures stemming from flawed reasoning and cognitive deadlocks. Motivated by these insights, we propose a collaborative Expert-Executor framework. It introduces a supervisory Expert agent tasked with providing strategic oversight and course-correction for a primary Executor agent. This architecture is designed to correct flawed reasoning and break the cognitive deadlocks that frequently lead to failure. Experiments show that our framework solves 22.2% of previously intractable issues for a leading single agent. These findings pave the way for building more robust agents through diagnostic evaluation and collaborative design. 

**Abstract (ZH)**: 自动问题解决旨在自主识别并修复代码库中整个代码片段的缺陷。SWE-Bench已成为该领域进展评估的最广泛采用基准。虽然基于LLM的代理工具前景广阔，但在众多任务中仍然失败。此外，当前的评估主要报告聚合的解决率，这模糊了成功和失败的根本原因，使得难以诊断模型弱点或指导有针对性的改进。为解决这一问题，我们首先分析了三种处于领先地位的工具在SWE-Bench-Verified下的自动问题解决任务中的性能和效率，这些工具涵盖了基于流水线和代理架构。此外，为了从高层面的性能指标转变为根本原因分析，我们系统地手动分析了150个失败实例。通过这一分析，我们开发了一种综合的失败模式分类法，包含3个主要阶段、9个主要类别和25个细化子类别。然后我们系统地分析了识别的失败模式的分布，结果显示两种架构模式之间的失败特征有所不同，代理架构的大多数失败源于推理错误和认知死锁。受这些见解的启发，我们提出了一种协作的专家-执行者框架。该框架引入了一位监管专家代理，负责为主要执行者代理提供战略监督和方向校正。该架构设计用于纠正推理错误并打破经常导致失败的认知死锁。实验显示，该框架解决了顶级单一代理无法解决的22.2%的问题。这些发现为通过诊断评估和协作设计构建更 robust 的代理铺平了道路。 

---
# Do Large Language Models Understand Word Senses? 

**Title (ZH)**: 大型语言模型理解词义吗？ 

**Authors**: Domenico Meconi, Simone Stirpe, Federico Martelli, Leonardo Lavalle, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2509.13905)  

**Abstract**: Understanding the meaning of words in context is a fundamental capability for Large Language Models (LLMs). Despite extensive evaluation efforts, the extent to which LLMs show evidence that they truly grasp word senses remains underexplored. In this paper, we address this gap by evaluating both i) the Word Sense Disambiguation (WSD) capabilities of instruction-tuned LLMs, comparing their performance to state-of-the-art systems specifically designed for the task, and ii) the ability of two top-performing open- and closed-source LLMs to understand word senses in three generative settings: definition generation, free-form explanation, and example generation. Notably, we find that, in the WSD task, leading models such as GPT-4o and DeepSeek-V3 achieve performance on par with specialized WSD systems, while also demonstrating greater robustness across domains and levels of difficulty. In the generation tasks, results reveal that LLMs can explain the meaning of words in context up to 98\% accuracy, with the highest performance observed in the free-form explanation task, which best aligns with their generative capabilities. 

**Abstract (ZH)**: 理解词语在上下文中的含义是大规模语言模型（LLMs）的一项基本能力。尽管进行了大量的评估努力，LLMs在真正掌握词语含义方面的证据仍然未被充分探索。在本文中，我们通过评估指令调优的LLMs的词语意义消歧能力（WSD），并与专门为此任务设计的最先进的系统进行比较，填补了这一空白，并评估了两个性能最顶尖的开源和闭源LLMs在三种生成设置下的词语意义理解能力：定义生成、自由形式解释和示例生成。值得注意的是，我们在WSD任务中发现，领先模型如GPT-4o和DeepSeek-V3在性能上与专门的WSD系统相当，并且在不同领域和难度级别上表现出了更强的稳健性。在生成任务中，结果表明，LLMs在上下文中解释词语意义的准确性高达98%，其中自由形式解释任务表现最佳，这与它们的生成能力最为契合。 

---
# Synthetic Data Generation for Screen Time and App Usage 

**Title (ZH)**: 屏幕时间与应用使用数据的合成数据生成 

**Authors**: Gustavo Kruger, Nikhil Sachdeva, Michael Sobolev  

**Link**: [PDF](https://arxiv.org/pdf/2509.13892)  

**Abstract**: Smartphone usage data can provide valuable insights for understanding interaction with technology and human behavior. However, collecting large-scale, in-the-wild smartphone usage logs is challenging due to high costs, privacy concerns, under representative user samples and biases like non-response that can skew results. These challenges call for exploring alternative approaches to obtain smartphone usage datasets. In this context, large language models (LLMs) such as Open AI's ChatGPT present a novel approach for synthetic smartphone usage data generation, addressing limitations of real-world data collection. We describe a case study on how four prompt strategies influenced the quality of generated smartphone usage data. We contribute with insights on prompt design and measures of data quality, reporting a prompting strategy comparison combining two factors, prompt level of detail (describing a user persona, describing the expected results characteristics) and seed data inclusion (with versus without an initial real usage example). Our findings suggest that using LLMs to generate structured and behaviorally plausible smartphone use datasets is feasible for some use cases, especially when using detailed prompts. Challenges remain in capturing diverse nuances of human behavioral patterns in a single synthetic dataset, and evaluating tradeoffs between data fidelity and diversity, suggesting the need for use-case-specific evaluation metrics and future research with more diverse seed data and different LLM models. 

**Abstract (ZH)**: 智能手机使用数据可以提供了解技术互动和人类行为的宝贵见解。然而，由于成本高、隐私顾虑、代表性不足的用户样本以及如非响应等偏差，收集大规模的真实世界智能手机使用日志具有挑战性。这些挑战要求探索替代方法以获得智能手机使用数据集。在此背景下，如Open AI的ChatGPT这样的大型语言模型提供了生成合成智能手机使用数据的新型方法，解决了现实世界数据收集的局限性。我们描述了四种提示策略如何影响生成的智能手机使用数据质量的案例研究。我们贡献了关于提示设计和数据质量度量的见解，报告了一种结合了两个因素的提示策略比较，即提示详细程度（描述用户 persona，描述预期结果特征）和初始真实使用数据的纳入（有或无初始真实使用示例）。我们的发现表明，在某些应用场景下，使用大型语言模型生成结构化且行为上可验证的智能手机使用数据集是可能的，尤其是在使用详细提示时。捕捉人类行为模式的多样性细微差别仍然面临挑战，并且在数据 fidelity 和多样性之间进行权衡时，需要特定用例的评估指标，并且未来的研究需要更多样化的初始数据和不同的大型语言模型。 

---
# Teaching According to Talents! Instruction Tuning LLMs with Competence-Aware Curriculum Learning 

**Title (ZH)**: 因材施教！基于能力意识的 Curriculum 学习调优大语言模型 

**Authors**: Yangning Li, Tingwei Lu, Yinghui Li, Yankai Chen, Wei-Chieh Huang, Wenhao Jiang, Hui Wang, Hai-Tao Zheng, Philip S.Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13790)  

**Abstract**: Efficient instruction tuning aims to enhance the ultimate performance of large language models (LLMs) trained on a given instruction dataset. Curriculum learning as a typical data organization strategy has shown preliminary effectiveness in instruction tuning. However, current curriculum tuning methods suffer from the curriculum rigidity, since they rely solely on static heuristic difficulty metrics. These methods fail to adapt to the evolving capabilities of models during training, resulting in a fixed and potentially sub-optimal learning trajectory. To address the issue, Competence-Aware Multi-Perspective cUrriculum inStruction tuning framework termed CAMPUS is proposed. CAMPUS offers several advantages: (1) Dynamic selection for sub-curriculum. (2) Competency-aware adjustment to the curriculum schedule. (3) Multiple difficulty-based scheduling. Extensive experiments prove the superior performance of CAMPUS, compared to other state-of-the-art baselines for efficient instruction tuning. 

**Abstract (ZH)**: Efficient指令调优旨在提升基于给定指令数据集训练的大语言模型（LLMs）的最终性能。作为典型的数据组织策略，渐进学习在指令调优中显示出初步的有效性。然而，当前的渐进调优方法受到固有刚性的限制，因为它们仅仅依赖于静态的经验难度指标。这些方法无法适应训练过程中模型能力的演变，导致固定且有可能是最优的学习轨迹。为解决该问题，提出了一种名为CAMPUS（Competence-Aware Multi-Perspective cUrriculum inStruction tuning）的框架：(1) 动态选择子课程。(2) 适应能力调整课程计划。(3) 多角度难度调度。广泛的实验证明，CAMPUS在高效指令调优方面优于其他最先进的baseline方法。 

---
# Exploring Data and Parameter Efficient Strategies for Arabic Dialect Identifications 

**Title (ZH)**: 探索阿拉伯方言识别的数据和参数高效策略 

**Authors**: Vani Kanjirangat, Ljiljana Dolamic, Fabio Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13775)  

**Abstract**: This paper discusses our exploration of different data-efficient and parameter-efficient approaches to Arabic Dialect Identification (ADI). In particular, we investigate various soft-prompting strategies, including prefix-tuning, prompt-tuning, P-tuning, and P-tuning V2, as well as LoRA reparameterizations. For the data-efficient strategy, we analyze hard prompting with zero-shot and few-shot inferences to analyze the dialect identification capabilities of Large Language Models (LLMs). For the parameter-efficient PEFT approaches, we conducted our experiments using Arabic-specific encoder models on several major datasets. We also analyzed the n-shot inferences on open-source decoder-only models, a general multilingual model (Phi-3.5), and an Arabic-specific one(SILMA). We observed that the LLMs generally struggle to differentiate the dialectal nuances in the few-shot or zero-shot setups. The soft-prompted encoder variants perform better, while the LoRA-based fine-tuned models perform best, even surpassing full fine-tuning. 

**Abstract (ZH)**: 本文探讨了不同数据高效和参数高效方法在阿拉伯方言识别（ADI）中的应用。特别地，我们调查了各种软提示策略，包括前缀调优、提示调优、P调优以及P调优V2，以及LoRA重参数化。在数据高效策略方面，我们分析了零-shot和少-shot推理中的硬提示，以评估大型语言模型（LLMs）的方言识别能力。在参数高效PEFT方法方面，我们在多个主要数据集上使用阿拉伯语特定的编码器模型进行了实验。我们还分析了开源的解码器模型、通用多语言模型（Phi-3.5）以及阿拉伯语特定模型（SILMA）的n-shot推理。我们观察到，在少-shot或零-shot设置中，LLMs一般难以区分方言差异。带有软提示的编码器变体表现较好，而基于LoRA的微调模型表现最佳，甚至超过了完全微调。 

---
# Scrub It Out! Erasing Sensitive Memorization in Code Language Models via Machine Unlearning 

**Title (ZH)**: 擦除敏感记忆！通过机器忘记消除代码语言模型中的敏感记忆 

**Authors**: Zhaoyang Chu, Yao Wan, Zhikun Zhang, Di Wang, Zhou Yang, Hongyu Zhang, Pan Zhou, Xuanhua Shi, Hai Jin, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2509.13755)  

**Abstract**: While Code Language Models (CLMs) have demonstrated superior performance in software engineering tasks such as code generation and summarization, recent empirical studies reveal a critical privacy vulnerability: these models exhibit unintended memorization of sensitive training data, enabling verbatim reproduction of confidential information when specifically prompted. To address this issue, several approaches, including training data de-duplication and differential privacy augmentation, have been proposed. However, these methods require full-model retraining for deployed CLMs, which incurs substantial computational costs. In this paper, we aim to answer the following research question: Can sensitive information memorized by CLMs be erased effectively and efficiently?
We conduct a pioneering investigation into erasing sensitive memorization in CLMs through machine unlearning - a post-hoc modification method that removes specific information from trained models without requiring full retraining. Specifically, we first quantify the memorization risks of sensitive data within CLM training datasets and curate a high-risk dataset of 50,000 sensitive memorized samples as unlearning targets. We study two widely used gradient ascent-based unlearning approaches: the vanilla and constraint-based methods, and introduce CodeEraser, an advanced variant that selectively unlearns sensitive memorized segments in code while preserving the structural integrity and functional correctness of the surrounding code. Extensive experiments on three families of CLMs, i.e., CodeParrot, CodeGen-Mono, and Qwen2.5-Coder, validate the effectiveness and efficiency of CodeEraser in erasing targeted sensitive memorization while maintaining model utility. 

**Abstract (ZH)**: CodeEraser：有效高效地擦除CLMs中敏感信息记忆的方法 

---
# Automated Triaging and Transfer Learning of Incident Learning Safety Reports Using Large Language Representational Models 

**Title (ZH)**: 使用大规模语言表示模型的事故学习安全报告自动化优先级划分和迁移学习 

**Authors**: Peter Beidler, Mark Nguyen, Kevin Lybarger, Ola Holmberg, Eric Ford, John Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13706)  

**Abstract**: PURPOSE: Incident reports are an important tool for safety and quality improvement in healthcare, but manual review is time-consuming and requires subject matter expertise. Here we present a natural language processing (NLP) screening tool to detect high-severity incident reports in radiation oncology across two institutions.
METHODS AND MATERIALS: We used two text datasets to train and evaluate our NLP models: 7,094 reports from our institution (Inst.), and 571 from IAEA SAFRON (SF), all of which had severity scores labeled by clinical content experts. We trained and evaluated two types of models: baseline support vector machines (SVM) and BlueBERT which is a large language model pretrained on PubMed abstracts and hospitalized patient data. We assessed for generalizability of our model in two ways. First, we evaluated models trained using Inst.-train on SF-test. Second, we trained a BlueBERT_TRANSFER model that was first fine-tuned on Inst.-train then on SF-train before testing on SF-test set. To further analyze model performance, we also examined a subset of 59 reports from our Inst. dataset, which were manually edited for clarity.
RESULTS Classification performance on the Inst. test achieved AUROC 0.82 using SVM and 0.81 using BlueBERT. Without cross-institution transfer learning, performance on the SF test was limited to an AUROC of 0.42 using SVM and 0.56 using BlueBERT. BlueBERT_TRANSFER, which was fine-tuned on both datasets, improved the performance on SF test to AUROC 0.78. Performance of SVM, and BlueBERT_TRANSFER models on the manually curated Inst. reports (AUROC 0.85 and 0.74) was similar to human performance (AUROC 0.81).
CONCLUSION: In summary, we successfully developed cross-institution NLP models on incident report text from radiation oncology centers. These models were able to detect high-severity reports similarly to humans on a curated dataset. 

**Abstract (ZH)**: 目的：事故报告是医疗卫生领域提高安全和质量的重要工具，但人工审查耗时且需要专业知识。本文介绍了一种自然语言处理（NLP）筛选工具，用于检测辐射肿瘤学领域两家机构中的高严重性事故报告。 

---
# DSCC-HS: A Dynamic Self-Reinforcing Framework for Hallucination Suppression in Large Language Models 

**Title (ZH)**: DSCC-HS：一种用于大规模语言模型幻觉抑制的动态自我强化框架 

**Authors**: Xiao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.13702)  

**Abstract**: Large Language Model (LLM) hallucination is a significant barrier to their reliable deployment. Current methods like Retrieval-Augmented Generation (RAG) are often reactive. We introduce **Dynamic Self-reinforcing Calibration for Hallucination Suppression (DSCC-HS)**, a novel, proactive framework that intervenes during autoregressive decoding. Inspired by dual-process cognitive theory, DSCC-HS uses a compact proxy model, trained in adversarial roles as a Factual Alignment Proxy (FAP) and a Hallucination Detection Proxy (HDP). During inference, these proxies dynamically steer a large target model by injecting a real-time steering vector, which is the difference between FAP and HDP logits, at each decoding step. This plug-and-play approach requires no modification to the target model. Our experiments on TruthfulQA and BioGEN show DSCC-HS achieves state-of-the-art performance. On TruthfulQA, it reached a 99.2% Factual Consistency Rate (FCR). On the long-form BioGEN benchmark, it attained the highest FActScore of 46.50. These results validate DSCC-HS as a principled and efficient solution for enhancing LLM factuality. 

**Abstract (ZH)**: 动态自我强化校准以抑制幻觉 (DSCC-HS): 一种 proactive 框架用于大型语言模型的故障自排查 

---
# Improving Context Fidelity via Native Retrieval-Augmented Reasoning 

**Title (ZH)**: 通过本体检索增强推理提高上下文保真度 

**Authors**: Suyuchen Wang, Jinlin Wang, Xinyu Wang, Shiqi Li, Xiangru Tang, Sirui Hong, Xiao-Wen Chang, Chenglin Wu, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13683)  

**Abstract**: Large language models (LLMs) often struggle with context fidelity, producing inconsistent answers when responding to questions based on provided information. Existing approaches either rely on expensive supervised fine-tuning to generate evidence post-answer or train models to perform web searches without necessarily improving utilization of the given context. We propose CARE, a novel native retrieval-augmented reasoning framework that teaches LLMs to explicitly integrate in-context evidence within their reasoning process with the model's own retrieval capabilities. Our method requires limited labeled evidence data while significantly enhancing both retrieval accuracy and answer generation performance through strategically retrieved in-context tokens in the reasoning chain. Extensive experiments on multiple real-world and counterfactual QA benchmarks demonstrate that our approach substantially outperforms supervised fine-tuning, traditional retrieval-augmented generation methods, and external retrieval solutions. This work represents a fundamental advancement in making LLMs more accurate, reliable, and efficient for knowledge-intensive tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）往往在处理上下文一致性方面存在困难，基于提供的信息回答问题时会产生不一致的答案。现有方法要么依赖昂贵的监督微调来生成答案后的证据，要么训练模型进行网络搜索，但不一定能提高对给定上下文的利用效率。我们提出了一种名为CARE的新颖的原生检索增强推理框架，该框架通过模型自身的检索能力，明确指导LLMs将上下文证据整合到其推理过程中。我们的方法仅需要少量标记的证据数据，同时通过在推理链中战略性地检索上下文令牌，显著提高检索准确性和答案生成性能。在多个现实世界和假设性问答基准测试上的大量实验表明，我们的方法在多个方面显著优于监督微调、传统的检索增强生成方法以及外部检索解决方案。这项工作代表了使LLMs在知识密集型任务中更加准确、可靠和高效的基本进步。 

---
# Prompt Stability in Code LLMs: Measuring Sensitivity across Emotion- and Personality-Driven Variations 

**Title (ZH)**: 代码LLMs中的提示稳定性：跨情绪和人格驱动变异性测量敏感性 

**Authors**: Wei Ma, Yixiao Yang, Jingquan Ge, Xiaofei Xie, Lingxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13680)  

**Abstract**: Code generation models are widely used in software development, yet their sensitivity to prompt phrasing remains under-examined. Identical requirements expressed with different emotions or communication styles can yield divergent outputs, while most benchmarks emphasize only peak performance. We present PromptSE (Prompt Sensitivity Evaluation), a framework that creates semantically equivalent prompt variants with emotion and personality templates, and that evaluates stability using probability aware continuous scoring or using binary pass rates when logits are unavailable. The results are aggregated into a proposed area under curve metric (AUC-E) for cross model comparison. Across 14 models from three families (Llama, Qwen, and DeepSeek), our study shows that performance and stability behave as largely decoupled optimization objectives, and it reveals architectural and scale related patterns that challenge common assumptions about model robustness. The framework supports rapid screening for closed-source models as well as detailed stability analysis in research settings. PromptSE enables practitioners to quantify performance stability trade offs for deployment and model selection, positioning prompt stability as a complementary evaluation dimension alongside performance and fairness, and contributing to more trustworthy AI-assisted software development tools. 

**Abstract (ZH)**: 代码生成模型在软件开发中广泛应用，但其对提示措辞的敏感性仍然研究不足。相同的需求用不同的情绪或沟通风格表达时，可以产生不同的输出，而大多数基准测试仅强调峰值性能。我们提出PromptSE（提示敏感性评估）框架，该框架通过情感和个性模板创建语义等价的提示变体，并使用概率意识的连续评分或在logits不可用时使用二元通过率来评估稳定性。将结果汇总为一个拟议的曲线下面积指标（AUC-E）以进行跨模型比较。在来自三个家族（Llama、Qwen和DeepSeek）的14个模型中，我们的研究显示性能和稳定性几乎是独立的优化目标，并揭示了与架构和规模相关的模式，这些模式挑战了对模型鲁棒性的通用假设。该框架支持对闭源模型的快速筛选以及在研究设置中的详细稳定性分析。PromptSE使实践者能够量化部署和模型选择中的性能稳健性trade-offs，并将提示稳健性定位为与性能和公平性并列的评估维度之一，从而促进更具可信度的AI辅助软件开发工具的发展。 

---
# Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs 

**Title (ZH)**: 稀疏神经元承载着大语言模型中问题歧义的强烈信号 

**Authors**: Zhuoxuan Zhang, Jinhao Duan, Edward Kim, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13664)  

**Abstract**: Ambiguity is pervasive in real-world questions, yet large language models (LLMs) often respond with confident answers rather than seeking clarification. In this work, we show that question ambiguity is linearly encoded in the internal representations of LLMs and can be both detected and controlled at the neuron level. During the model's pre-filling stage, we identify that a small number of neurons, as few as one, encode question ambiguity information. Probes trained on these Ambiguity-Encoding Neurons (AENs) achieve strong performance on ambiguity detection and generalize across datasets, outperforming prompting-based and representation-based baselines. Layerwise analysis reveals that AENs emerge from shallow layers, suggesting early encoding of ambiguity signals in the model's processing pipeline. Finally, we show that through manipulating AENs, we can control LLM's behavior from direct answering to abstention. Our findings reveal that LLMs form compact internal representations of question ambiguity, enabling interpretable and controllable behavior. 

**Abstract (ZH)**: 现实世界问题中的歧义在所难免，而大规模语言模型（LLMs）往往给出自信的回答而非寻求澄清。在本研究中，我们展示了问题歧义在线性编码在LLMs的内部表示中，并可以在神经元级别进行检测和控制。在模型的预填充阶段，我们发现少量神经元，甚至是单个神经元，能够编码问题歧义信息。基于这些编码歧义信息的神经元（AENs）的探针在歧义检测上表现出色，并在不同数据集上具有泛化能力，超过了基于提示和表示的基线方法。逐层分析显示，AENs来源于浅层，表明模型处理管线中早期编码歧义信号。最后，我们展示了通过操纵AENs，可以控制LLMs的直接回答行为转为规避行为。我们发现，LLMs形成了紧凑的问题歧义内部表示，从而实现了可解释和可控的行为。 

---
# Modernizing Facebook Scoped Search: Keyword and Embedding Hybrid Retrieval with LLM Evaluation 

**Title (ZH)**: 现代优化的Facebook范围搜索：基于关键词和嵌入的混合检索与LLM评估 

**Authors**: Yongye Su, Zeya Zhang, Jane Kou, Cheng Ju, Shubhojeet Sarkar, Yamin Wang, Ji Liu, Shengbo Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.13603)  

**Abstract**: Beyond general web-scale search, social network search uniquely enables users to retrieve information and discover potential connections within their social context. We introduce a framework of modernized Facebook Group Scoped Search by blending traditional keyword-based retrieval with embedding-based retrieval (EBR) to improve the search relevance and diversity of search results. Our system integrates semantic retrieval into the existing keyword search pipeline, enabling users to discover more contextually relevant group posts. To rigorously assess the impact of this blended approach, we introduce a novel evaluation framework that leverages large language models (LLMs) to perform offline relevance assessments, providing scalable and consistent quality benchmarks. Our results demonstrate that the blended retrieval system significantly enhances user engagement and search quality, as validated by both online metrics and LLM-based evaluation. This work offers practical insights for deploying and evaluating advanced retrieval systems in large-scale, real-world social platforms. 

**Abstract (ZH)**: 超越通用的网页规模搜索，社交网络搜索独特地使用户能够在其社交背景下检索信息并发现潜在连接。我们通过结合传统的基于关键词检索与嵌入式检索（EBR）来改进搜索的相关性和多样性，引入了一个现代化的Facebook组范围搜索框架。我们的系统将语义检索集成到现有的关键词搜索管道中，使用户能够发现更多上下文相关性更强的组帖子。为了严格评估这种混合方法的影响，我们引入了一种新的评估框架，利用大型语言模型（LLMs）进行离线相关性评估，提供可扩展且一致的质量基准。我们的结果表明，混合检索系统显著提高了用户参与度和搜索质量，得到了在线指标和基于LLM的评估的验证。本研究为部署和评估大型实际社交平台上的高级检索系统提供了实用见解。 

---
# Agentic JWT: A Secure Delegation Protocol for Autonomous AI Agents 

**Title (ZH)**: 代理JWT：自主AI代理的安全代理协议 

**Authors**: Abhishek Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2509.13597)  

**Abstract**: Autonomous LLM agents can issue thousands of API calls per hour without human oversight. OAuth 2.0 assumes deterministic clients, but in agentic settings stochastic reasoning, prompt injection, or multi-agent orchestration can silently expand privileges.
We introduce Agentic JWT (A-JWT), a dual-faceted intent token that binds each agent's action to verifiable user intent and, optionally, to a specific workflow step. A-JWT carries an agent's identity as a one-way checksum hash derived from its prompt, tools and configuration, and a chained delegation assertion to prove which downstream agent may execute a given task, and per-agent proof-of-possession keys to prevent replay and in-process impersonation. We define a new authorization mechanism and add a lightweight client shim library that self-verifies code at run time, mints intent tokens, tracks workflow steps and derives keys, thus enabling secure agent identity and separation even within a single process.
We illustrate a comprehensive threat model for agentic applications, implement a Python proof-of-concept and show functional blocking of scope-violating requests, replay, impersonation, and prompt-injection pathways with sub-millisecond overhead on commodity hardware. The design aligns with ongoing OAuth agent discussions and offers a drop-in path toward zero-trust guarantees for agentic applications. A comprehensive performance and security evaluation with experimental results will appear in our forthcoming journal publication 

**Abstract (ZH)**: 自主LLM代理每小时可以发出数千次API调用而无需人类监管。OAuth 2.0 假设确定性客户端，但在代理环境中，随机推理、提示注入或多个代理协同编排可以无声地扩展权限。

我们引入了代理JWT (A-JWT)，这是一种双面意图令牌，将每个代理的动作绑定到可验证的用户意图上，并可选地绑定到特定的工作流步骤。A-JWT 携带代理的身份作为从其提示、工具和配置中导出的一次性校验和哈希，以及证明哪一个下游代理可以执行给定任务的递归委托断言，并提供每个代理的所有权证明密钥以防止重放和内过程冒充。我们定义了一种新的授权机制，并添加了一个轻量级客户端 shim 库，该库可以在运行时自我验证代码、颁发意图令牌、跟踪工作流步骤并生成密钥，从而在单个进程中实现安全的代理身份和隔离。

我们概述了代理应用程序的全面威胁模型，实现了 Python 证明概念并展示了对作用域违规请求、重放、冒充和提示注入途径的功能性阻止，使用大众市场硬件的亚毫秒级开销。该设计与正在进行中的 OAuth 代理讨论相一致，并为代理应用程序提供了一条直接路径以实现零信任保证。全面的性能和安全评估及其实验结果将出现在我们即将发表的期刊出版物中。 

---
# Prompt2DAG: A Modular Methodology for LLM-Based Data Enrichment Pipeline Generation 

**Title (ZH)**: Prompt2DAG: 一种基于LLM的数据增强管道生成模块化方法 

**Authors**: Abubakari Alidu, Michele Ciavotta, Flavio DePaoli  

**Link**: [PDF](https://arxiv.org/pdf/2509.13487)  

**Abstract**: Developing reliable data enrichment pipelines demands significant engineering expertise. We present Prompt2DAG, a methodology that transforms natural language descriptions into executable Apache Airflow DAGs. We evaluate four generation approaches -- Direct, LLM-only, Hybrid, and Template-based -- across 260 experiments using thirteen LLMs and five case studies to identify optimal strategies for production-grade automation. Performance is measured using a penalized scoring framework that combines reliability with code quality (SAT), structural integrity (DST), and executability (PCT). The Hybrid approach emerges as the optimal generative method, achieving a 78.5% success rate with robust quality scores (SAT: 6.79, DST: 7.67, PCT: 7.76). This significantly outperforms the LLM-only (66.2% success) and Direct (29.2% success) methods. Our findings show that reliability, not intrinsic code quality, is the primary differentiator. Cost-effectiveness analysis reveals the Hybrid method is over twice as efficient as Direct prompting per successful DAG. We conclude that a structured, hybrid approach is essential for balancing flexibility and reliability in automated workflow generation, offering a viable path to democratize data pipeline development. 

**Abstract (ZH)**: 开发可靠的数据增强管道需要大量的工程专业知识。我们呈现了Prompt2DAG方法，该方法将自然语言描述转换为可执行的Apache Airflow DAG。我们使用十三种LLM和五项案例研究，在260个实验中评估了四种生成方法——直接、仅LLM、混合和基于模板的方法，以确定生产级自动化的最佳策略。性能通过结合可靠性、代码质量（SAT）、结构完整性（DST）和可执行性（PCT）的惩罚评分框架进行衡量。混合方法 emerged 作为最优生成方法，成功率为78.5%，且具有稳健的质量评分（SAT：6.79，DST：7.67，PCT：7.76）。这种方法在成功率方面显著优于仅LLM（66.2%成功）和直接方法（29.2%成功）。我们的研究结果显示，可靠性而非内在代码质量是主要区别因素。成本效益分析表明，混合方法相对于直接提示在每个成功的DAG方面效率提高了一倍以上。我们得出结论，结构化的混合方法对于平衡自动化工作流生成中的灵活性和可靠性至关重要，提供了 democratize 数据管道开发的可行路径。 

---
# An LLM Agentic Approach for Legal-Critical Software: A Case Study for Tax Prep Software 

**Title (ZH)**: 基于LLM代理方法的法律批判性软件研究：税务准备软件案例研究 

**Authors**: Sina Gogani-Khiabani, Ashutosh Trivedi, Diptikalyan Saha, Saeid Tizpaz-Niari  

**Link**: [PDF](https://arxiv.org/pdf/2509.13471)  

**Abstract**: Large language models (LLMs) show promise for translating natural-language statutes into executable logic, but reliability in legally critical settings remains challenging due to ambiguity and hallucinations. We present an agentic approach for developing legal-critical software, using U.S. federal tax preparation as a case study. The key challenge is test-case generation under the oracle problem, where correct outputs require interpreting law. Building on metamorphic testing, we introduce higher-order metamorphic relations that compare system outputs across structured shifts among similar individuals. Because authoring such relations is tedious and error-prone, we use an LLM-driven, role-based framework to automate test generation and code synthesis. We implement a multi-agent system that translates tax code into executable software and incorporates a metamorphic-testing agent that searches for counterexamples. In experiments, our framework using a smaller model (GPT-4o-mini) achieves a worst-case pass rate of 45%, outperforming frontier models (GPT-4o and Claude 3.5, 9-15%) on complex tax-code tasks. These results support agentic LLM methodologies as a path to robust, trustworthy legal-critical software from natural-language specifications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在将自然语言法规转换为可执行逻辑方面显示出潜力，但在法律关键性设置中的可靠性仍然面临挑战，原因在于模糊性和幻觉。我们提出了一种自主性的方法来开发法律关键性软件，并以美国联邦税务申报为案例研究。关键挑战在于在oracle问题下的测试案例生成，其中正确的输出需要解释法律条文。基于元特征测试，我们引入了更高的阶元特征关系，用于比较相似个体之间结构化变化的系统输出。由于编写此类关系是繁琐且易出错的，我们使用基于大型语言模型的角色化框架来自动化测试和代码生成。我们实现了一个多代理系统，将税法翻译为可执行软件，并包含一个用于搜索反例的元特征测试代理。在实验中，我们的框架使用较小的模型（GPT-4o-mini）在最坏情况下达到了45%的通过率，优于前沿模型（GPT-4o和Claude 3.5，9-15%）在复杂税法任务上的表现。这些结果支持自主性的大型语言模型方法作为从自然语言规范生成稳健可靠的法律关键性软件的路径。 

---
# Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews 

**Title (ZH)**: 正义在判断：揭示LLM辅助下的peer review中的偏见 

**Authors**: Sai Suresh Marchala Vasu, Ivaxi Sheth, Hui-Po Wang, Ruta Binkyte, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2509.13400)  

**Abstract**: The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings. 

**Abstract (ZH)**: 大型语言模型的应用正在转变同行评审过程，从帮助评审者撰写更详细的评估到自动生成整个评审。尽管这些能力提供了令人兴奋的机会，但也引发了关于公平性和可靠性的关键问题。在本文中，我们通过在敏感元数据（包括作者 affiliation 和性别）上进行受控实验，研究大型语言模型生成的同行评审中的偏见。我们的分析一致表明， affiliation 偏见倾向于排名较高的学术机构。此外，我们还发现一些性别偏好，尽管其程度较小，但有可能随着时间的推移而累积。值得注意的是，我们发现基于token的柔性评分中存在隐性偏见，这些偏见会变得更加明显。 

---
# The threat of analytic flexibility in using large language models to simulate human data: A call to attention 

**Title (ZH)**: 使用大型语言模型模拟人类数据中的分析灵活性威胁：引起关注的呼吁 

**Authors**: Jamie Cummins  

**Link**: [PDF](https://arxiv.org/pdf/2509.13397)  

**Abstract**: Social scientists are now using large language models to create "silicon samples" - synthetic datasets intended to stand in for human respondents, aimed at revolutionising human subjects research. However, there are many analytic choices which must be made to produce these samples. Though many of these choices are defensible, their impact on sample quality is poorly understood. I map out these analytic choices and demonstrate how a very small number of decisions can dramatically change the correspondence between silicon samples and human data. Configurations (N = 252) varied substantially in their capacity to estimate (i) rank ordering of participants, (ii) response distributions, and (iii) between-scale correlations. Most critically, configurations were not consistent in quality: those that performed well on one dimension often performed poorly on another, implying that there is no "one-size-fits-all" configuration that optimises the accuracy of these samples. I call for greater attention to the threat of analytic flexibility in using silicon samples. 

**Abstract (ZH)**: 社会科学家现在利用大型语言模型创建“硅样本”——合成数据集，旨在替代人类受访者，以革命性地改变人类主体研究。然而，生成这些样本需要做出许多分析选择，尽管其中许多选择是可辩护的，但它们对样本质量的影响知之甚少。我绘制出这些分析选择，并证明极少数决策可以显著改变硅样本与人类数据的一致性。配置（N=252）在（i）参与者排名排序、（ii）响应分布以及（iii）跨量表相关性估算能力方面差异很大。最关键的是，配置在质量上并不一致：在某一维度表现良好的配置往往在另一维度表现不佳，这表明不存在适用于所有情况的配置以优化这些样本的准确性。我呼吁在使用硅样本时更加关注分析灵活性带来的威胁。 

---
# TICL: Text-Embedding KNN For Speech In-Context Learning Unlocks Speech Recognition Abilities of Large Multimodal Models 

**Title (ZH)**: TICL: 文本嵌入KNN实现基于上下文的语音学习，解锁大型多模态模型的语音识别能力 

**Authors**: Haolong Zheng, Yekaterina Yegorova, Mark Hasegawa-Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2509.13395)  

**Abstract**: Speech foundation models have recently demonstrated the ability to perform Speech In-Context Learning (SICL). Selecting effective in-context examples is crucial for SICL performance, yet selection methodologies remain underexplored. In this work, we propose Text-Embedding KNN for SICL (TICL), a simple pipeline that uses semantic context to enhance off-the-shelf large multimodal models' speech recognition ability without fine-tuning. Across challenging automatic speech recognition tasks, including accented English, multilingual speech, and children's speech, our method enables models to surpass zero-shot performance with up to 84.7% relative WER reduction. We conduct ablation studies to show the robustness and efficiency of our method. 

**Abstract (ZH)**: 基于文本嵌入的最近邻方法在语音下行文学习中的应用：提高现成的大型多模态模型的语音识别能力 

---
# An Empirical Analysis of VLM-based OOD Detection: Mechanisms, Advantages, and Sensitivity 

**Title (ZH)**: 基于VLM的OOD检测的实证分析：机制、优势与敏感性 

**Authors**: Yuxiao Lee, Xiaofeng Cao, Wei Ye, Jiangchao Yao, Jingkuan Song, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.13375)  

**Abstract**: Vision-Language Models (VLMs), such as CLIP, have demonstrated remarkable zero-shot out-of-distribution (OOD) detection capabilities, vital for reliable AI systems. Despite this promising capability, a comprehensive understanding of (1) why they work so effectively, (2) what advantages do they have over single-modal methods, and (3) how is their behavioral robustness -- remains notably incomplete within the research community. This paper presents a systematic empirical analysis of VLM-based OOD detection using in-distribution (ID) and OOD prompts. (1) Mechanisms: We systematically characterize and formalize key operational properties within the VLM embedding space that facilitate zero-shot OOD detection. (2) Advantages: We empirically quantify the superiority of these models over established single-modal approaches, attributing this distinct advantage to the VLM's capacity to leverage rich semantic novelty. (3) Sensitivity: We uncovers a significant and previously under-explored asymmetry in their robustness profile: while exhibiting resilience to common image noise, these VLM-based methods are highly sensitive to prompt phrasing. Our findings contribute a more structured understanding of the strengths and critical vulnerabilities inherent in VLM-based OOD detection, offering crucial, empirically-grounded guidance for developing more robust and reliable future designs. 

**Abstract (ZH)**: 基于视觉-语言模型的异常分布检测：机制、优势与敏感性分析 

---
# The Provenance Problem: LLMs and the Breakdown of Citation Norms 

**Title (ZH)**: 来源问题：LLMs与引文规范的 breakdown 

**Authors**: Brian D. Earp, Haotian Yuan, Julian Koplin, Sebastian Porsdam Mann  

**Link**: [PDF](https://arxiv.org/pdf/2509.13365)  

**Abstract**: The increasing use of generative AI in scientific writing raises urgent questions about attribution and intellectual credit. When a researcher employs ChatGPT to draft a manuscript, the resulting text may echo ideas from sources the author has never encountered. If an AI system reproduces insights from, for example, an obscure 1975 paper without citation, does this constitute plagiarism? We argue that such cases exemplify the 'provenance problem': a systematic breakdown in the chain of scholarly credit. Unlike conventional plagiarism, this phenomenon does not involve intent to deceive (researchers may disclose AI use and act in good faith) yet still benefit from the uncredited intellectual contributions of others. This dynamic creates a novel category of attributional harm that current ethical and professional frameworks fail to address. As generative AI becomes embedded across disciplines, the risk that significant ideas will circulate without recognition threatens both the reputational economy of science and the demands of epistemic justice. This Perspective analyzes how AI challenges established norms of authorship, introduces conceptual tools for understanding the provenance problem, and proposes strategies to preserve integrity and fairness in scholarly communication. 

**Abstract (ZH)**: 生成式AI在科学写作中的广泛应用引发了关于归属和智力信用的紧迫问题。 

---
# Accuracy Paradox in Large Language Models: Regulating Hallucination Risks in Generative AI 

**Title (ZH)**: 大型语言模型中的准确性悖论：生成型AI中调节幻觉风险的研究 

**Authors**: Zihao Li, Weiwei Yi, Jiahong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.13345)  

**Abstract**: As Large Language Models (LLMs) permeate everyday decision-making, their epistemic and societal risks demand urgent scrutiny. Hallucinations, the generation of fabricated, misleading, oversimplified or untrustworthy outputs, has emerged as imperative challenges. While regulatory, academic, and technical discourse position accuracy as the principal benchmark for mitigating such harms, this article contends that overreliance on accuracy misdiagnoses the problem and has counterproductive effect: the accuracy paradox. Drawing on interdisciplinary literatures, this article develops a taxonomy of hallucination types and shows the paradox along three intertwining dimensions: outputs, individuals and society. First, accuracy functions as a superficial proxy for reliability, incentivising the optimisation of rhetorical fluency and surface-level correctness over epistemic trustworthiness. This encourages passive user trust in outputs that appear accurate but epistemically untenable. Second, accuracy as a singular metric fails to detect harms that are not factually false but are nonetheless misleading, value-laden, or socially distorting, including consensus illusions, sycophantic alignment, and subtle manipulation. Third, regulatory overemphasis on accuracy obscures the wider societal consequences of hallucination, including social sorting, privacy violations, equity harms, epistemic convergence that marginalises dissent, reduces pluralism, and causes social deskilling. By examining the EU AI Act, GDPR, and DSA, the article argues that current regulations are not yet structurally equipped to address these epistemic, relational, and systemic harms and exacerbated by the overreliance on accuracy. By exposing such conceptual and practical challenges, this article calls for a fundamental shift towards pluralistic, context-aware, and manipulation-resilient approaches to AI trustworthy governance. 

**Abstract (ZH)**: 大型语言模型（LLMs）渗透日常决策后，其认识论和社会风险亟需紧急审查。幻觉，即生成虚假的、误导的、过度简化或不值得信赖的输出，已成为关键挑战。尽管监管、学术和技术界的讨论将准确性视为减轻此类危害的主要标准，本文认为过度依赖准确性会对问题进行误诊，并产生反效作用：准确性悖论。基于跨学科文献，本文发展了幻觉类型的分类，并通过三个交织的维度展示了悖论：输出、个体和社会。首先，准确性充当了可靠性的表面代理，促进了修辞流畅性和表面正确性的优化，而忽略了认识论的信任。这鼓励用户对看似准确但实际上在认识论上站不住脚的输出产生被动信任。其次，作为单一指标的准确性无法识别并非事实错误但却具有误导性、价值观导向或社会变形的危害，包括共识幻觉、逢迎对齐和微妙操控。第三，监管对准确性的过度强调掩盖了幻觉更广泛的社会后果，包括社会分类、隐私侵犯、公平危害、共识收敛边缘化不同意见，减少多元性，并导致社会脱技能。通过分析欧盟AI法案、GDPR和DSA，本文认为现有监管尚未结构化地应对这些认识论、关系性和系统性危害，并因过度依赖准确性而加剧。通过揭示这些概念性和实践性挑战，本文呼吁向多元、上下文意识和抗操控的AI可信治理方法的根本性转变。 

---
