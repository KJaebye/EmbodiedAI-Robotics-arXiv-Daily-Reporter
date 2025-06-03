# LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation 

**Title (ZH)**: LLM辅助的多智能体强化学习在合作策略生成中的应用 

**Authors**: Guobin Zhu, Rui Zhou, Wenkang Ji, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01538)  

**Abstract**: Although Multi-Agent Reinforcement Learning (MARL) is effective for complex multi-robot tasks, it suffers from low sample efficiency and requires iterative manual reward tuning. Large Language Models (LLMs) have shown promise in single-robot settings, but their application in multi-robot systems remains largely unexplored. This paper introduces a novel LLM-Aided MARL (LAMARL) approach, which integrates MARL with LLMs, significantly enhancing sample efficiency without requiring manual design. LAMARL consists of two modules: the first module leverages LLMs to fully automate the generation of prior policy and reward functions. The second module is MARL, which uses the generated functions to guide robot policy training effectively. On a shape assembly benchmark, both simulation and real-world experiments demonstrate the unique advantages of LAMARL. Ablation studies show that the prior policy improves sample efficiency by an average of 185.9% and enhances task completion, while structured prompts based on Chain-of-Thought (CoT) and basic APIs improve LLM output success rates by 28.5%-67.5%. Videos and code are available at this https URL 

**Abstract (ZH)**: 虽然多代理 reinforcement 学习（MARL）在复杂多机器人任务中有效，但其样本效率较低且需要迭代的手动奖励调整。大规模语言模型（LLMs）在单机器人设置中表现出色，但在多机器人系统中的应用尚未得到充分探索。本文提出了一种新颖的 LLM 辅助 MARL（LAMARL）方法，将 MARL 与 LLM 集成，显著提高了样本效率，而无需手动设计。LAMARL 包含两个模块：第一个模块利用 LLM 完全自动生成先验策略和奖励函数。第二个模块是 MARL，它使用生成的函数有效地指导机器人策略训练。在形状装配基准测试中，模拟和实际实验都展示了 LAMARL 的独特优势。消融研究显示，先验策略平均提高样本效率185.9%，并提升任务完成率，基于 Chain-of-Thought（CoT）的结构化提示和基本 API 使 LLM 输出成功率提升了28.5%-67.5%。更多信息参见此网址。 

---
# Reducing Latency in LLM-Based Natural Language Commands Processing for Robot Navigation 

**Title (ZH)**: 基于LLM的自然语言命令处理中降低机器人导航延迟 

**Authors**: Diego Pollini, Bruna V. Guterres, Rodrigo S. Guerra, Ricardo B. Grando  

**Link**: [PDF](https://arxiv.org/pdf/2506.00075)  

**Abstract**: The integration of Large Language Models (LLMs), such as GPT, in industrial robotics enhances operational efficiency and human-robot collaboration. However, the computational complexity and size of these models often provide latency problems in request and response times. This study explores the integration of the ChatGPT natural language model with the Robot Operating System 2 (ROS 2) to mitigate interaction latency and improve robotic system control within a simulated Gazebo environment. We present an architecture that integrates these technologies without requiring a middleware transport platform, detailing how a simulated mobile robot responds to text and voice commands. Experimental results demonstrate that this integration improves execution speed, usability, and accessibility of the human-robot interaction by decreasing the communication latency by 7.01\% on average. Such improvements facilitate smoother, real-time robot operations, which are crucial for industrial automation and precision tasks. 

**Abstract (ZH)**: 大语言模型（LLMs）如GPT在工业机器人中的集成增强了操作效率和人机协作。然而，这些模型的计算复杂度和大小常导致请求和响应时间的延迟问题。本研究探讨了将ChatGPT自然语言模型与Robot Operating System 2（ROS 2）集成以减轻交互延迟并改善模拟Gazebo环境中的机器人系统控制。我们提出了一种无需中间件传输平台的技术架构，详细说明了模拟移动机器人如何响应文本和语音命令。实验结果表明，这种集成通过平均减少7.01%的通信延迟，提高了人机交互的执行速度、可用性和访问性，从而促进更顺畅、实时的机器人操作，这对于工业自动化和精确任务至关重要。 

---
# Large language models can learn and generalize steganographic chain-of-thought under process supervision 

**Title (ZH)**: 大规模语言模型在接受过程监督的情况下可以学习和泛化隐写论证链。 

**Authors**: Joey Skaf, Luis Ibanez-Lissen, Robert McCarthy, Connor Watts, Vasil Georgiv, Hannes Whittingham, Lorena Gonzalez-Manzano, David Lindner, Cameron Tice, Edward James Young, Puria Radmard  

**Link**: [PDF](https://arxiv.org/pdf/2506.01926)  

**Abstract**: Chain-of-thought (CoT) reasoning not only enhances large language model performance but also provides critical insights into decision-making processes, marking it as a useful tool for monitoring model intent and planning. By proactively preventing models from acting on CoT indicating misaligned or harmful intent, CoT monitoring can be used to reduce risks associated with deploying models. However, developers may be incentivized to train away the appearance of harmful intent from CoT traces, by either customer preferences or regulatory requirements. Recent works have shown that banning mention of a specific example of reward hacking, which may be done either to make CoT presentable to users or as a naive attempt to prevent the behavior, causes obfuscation of the undesired reasoning traces but the persistence of the undesired behavior. Such obfuscation threatens the reliability of CoT monitoring. However, obfuscation of reasoning can be due to its internalization to latent space computation, or its encoding within the CoT. Here, we provide an extension to these results. First, we show that penalizing the use of specific strings within load-bearing reasoning traces causes models to substitute alternative strings. Crucially, this does not alter the underlying method by which the model performs the task, demonstrating that the model can learn to steganographically encode its reasoning. We further demonstrate that models can generalize an encoding scheme. When the penalized strings belong to an overarching class, the model learns not only to substitute strings seen in training, but also develops a general encoding scheme for all members of the class which it can apply to held-out testing strings. 

**Abstract (ZH)**: Chain-of-Thought推理不仅能够提升大型语言模型的表现，还能为决策过程提供关键洞察，标志着它作为一个监测模型意图和规划的有用工具。通过主动防止模型在显示错齐或有害意图的链式推理指示下行动，链式推理监控可以减少部署模型相关的风险。然而，开发者可能由于客户偏好或监管要求等原因，被激励去训练模型，使其在链式推理痕迹中不表现出有害意图。近期研究表明，禁止提及特定奖励作弊案例，无论是为了使链式推理对用户更易理解，还是出于简单地防止此行为的尝试，都会导致有害推理痕迹的掩饰，但有害行为依然存在。这种掩饰威胁了链式推理监控的可靠性。然而，推理掩饰可能是由于其被内化到潜在空间计算，或其被编码在链式推理中。在此，我们扩展了这些结果。首先，我们展示，惩罚特定字符串在承重推理痕迹中的使用，会导致模型替代使用其他字符串。关键的是，这并不会改变模型执行任务的基本方法，证明了模型能够学习隐写编码其推理。我们进一步证明，模型能够泛化编码方案。当受惩罚的字符串属于一个上位类别时，模型不仅会替换训练中见过的字符串，还会发展出适用于该类别中所有成员的一般编码方案，并能够应用到保留测试字符串上。 

---
# COALESCE: Economic and Security Dynamics of Skill-Based Task Outsourcing Among Team of Autonomous LLM Agents 

**Title (ZH)**: COALESCE: 基于技能的任务外包经济与安全动态研究——自主大语言模型代理团队视角 

**Authors**: Manish Bhatt, Ronald F. Del Rosario, Vineeth Sai Narajala, Idan Habler  

**Link**: [PDF](https://arxiv.org/pdf/2506.01900)  

**Abstract**: The meteoric rise and proliferation of autonomous Large Language Model (LLM) agents promise significant capabilities across various domains. However, their deployment is increasingly constrained by substantial computational demands, specifically for Graphics Processing Unit (GPU) resources. This paper addresses the critical problem of optimizing resource utilization in LLM agent systems. We introduce COALESCE (Cost-Optimized and Secure Agent Labour Exchange via Skill-based Competence Estimation), a novel framework designed to enable autonomous LLM agents to dynamically outsource specific subtasks to specialized, cost-effective third-party LLM agents. The framework integrates mechanisms for hybrid skill representation, dynamic skill discovery, automated task decomposition, a unified cost model comparing internal execution costs against external outsourcing prices, simplified market-based decision-making algorithms, and a standardized communication protocol between LLM agents. Comprehensive validation through 239 theoretical simulations demonstrates 41.8\% cost reduction potential, while large-scale empirical validation across 240 real LLM tasks confirms 20.3\% cost reduction with proper epsilon-greedy exploration, establishing both theoretical viability and practical effectiveness. The emergence of proposed open standards like Google's Agent2Agent (A2A) protocol further underscores the need for frameworks like COALESCE that can leverage such standards for efficient agent interaction. By facilitating a dynamic market for agent capabilities, potentially utilizing protocols like A2A for communication, COALESCE aims to significantly reduce operational costs, enhance system scalability, and foster the emergence of specialized agent economies, making complex LLM agent functionalities more accessible and economically viable. 

**Abstract (ZH)**: 自主大型语言模型（LLM）代理的迅猛崛起与普及为各领域带来了显著能力，但其部署正日益受到巨大计算需求的限制，特别是对图形处理单元（GPU）资源的需求。本文探讨了优化LLM代理系统资源利用的关键问题。我们提出了COALESCE（基于技能能力估计的成本优化和安全代理劳动交换框架），这是一种新型框架，旨在使自主LLM代理能够动态外包特定子任务给专业的、成本效益高的第三方LLM代理。该框架整合了混合技能表示机制、动态技能发现机制、自动任务分解机制、综合成本模型（比较内部执行成本与外部外包价格）、简化市场决策算法以及LLM代理之间的标准化通信协议。通过239个理论模拟的全面验证，显示出41.8%的成本降低潜力，而针对240个真实LLM任务的大规模实证验证表明，通过适当的最佳ε贪心探索，能够实现20.3%的成本降低，从而奠定了该方法的理论可行性和实际有效性。随着谷歌等提出的开放标准（如A2A协议）的出现，进一步突显了需要像COALESCE这样的框架来利用这些标准以实现高效的代理交互。通过促进代理能力的动态市场，并可能利用A2A等协议进行通信，COALESCE旨在显著降低运营成本、增强系统扩展性，并促进专业代理经济体的形成，使复杂的LLM代理功能更加普及和经济可行。 

---
# WHEN TO ACT, WHEN TO WAIT: Modeling Structural Trajectories for Intent Triggerability in Task-Oriented Dialogue 

**Title (ZH)**: 何时行动，何时等待：基于任务导向对话的结构轨迹建模及其意图触发性研究 

**Authors**: Yaoyao Qian, Jindan Huang, Yuanli Wang, Simon Yu, Kyrie Zhixuan Zhou, Jiayuan Mao, Mingfu Liang, Hanhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.01881)  

**Abstract**: Task-oriented dialogue systems often face difficulties when user utterances seem semantically complete but lack necessary structural information for appropriate system action. This arises because users frequently do not fully understand their own needs, while systems require precise intent definitions. Current LLM-based agents cannot effectively distinguish between linguistically complete and contextually triggerable expressions, lacking frameworks for collaborative intent formation. We present STORM, a framework modeling asymmetric information dynamics through conversations between UserLLM (full internal access) and AgentLLM (observable behavior only). STORM produces annotated corpora capturing expression trajectories and latent cognitive transitions, enabling systematic analysis of collaborative understanding development. Our contributions include: (1) formalizing asymmetric information processing in dialogue systems; (2) modeling intent formation tracking collaborative understanding evolution; and (3) evaluation metrics measuring internal cognitive improvements alongside task performance. Experiments across four language models reveal that moderate uncertainty (40-60%) can outperform complete transparency in certain scenarios, with model-specific patterns suggesting reconsideration of optimal information completeness in human-AI collaboration. These findings contribute to understanding asymmetric reasoning dynamics and inform uncertainty-calibrated dialogue system design. 

**Abstract (ZH)**: 面向任务的对话系统往往在用户陈述看似语义完整但缺乏必要结构信息以进行适当系统操作时遇到困难。这源于用户通常不完全了解自己的需求，而系统则需要精确的意图定义。当前基于大语言模型的代理无法有效区分语义完整和上下文触发的表达，缺乏协作意图形成的框架。我们提出了STORM框架，通过UserLLM（拥有完全内部访问权限）与AgentLLM（仅观察行为）之间的对话来建模不对称信息动态。STORM生成包含表达轨迹和潜在认知转换的标注语料库，以便系统地分析协作理解的发展。我们的贡献包括：（1）形式化对话系统中的不对称信息处理；（2）建模意图形成以追踪协作理解的发展；（3）评估指标以衡量内部认知改进与任务性能。跨四种语言模型的实验表明，在某些场景中，中等不确定性（40-60%）可能优于完全透明度，特定模型模式表明在人机协作中重新评估最优信息完整性的必要性。这些发现有助于理解不对称推理动态，并为不确定性校准的对话系统设计提供指导。 

---
# A Study on the MCP x A2A Framework for Enhancing Interoperability of LLM-based Autonomous Agents 

**Title (ZH)**: 基于MCP x A2A框架提升基于LLM的自主代理互操作性的研究 

**Authors**: Cheonsu Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01804)  

**Abstract**: This paper provides an in-depth technical analysis and implementation methodology of the open-source Agent-to-Agent (A2A) protocol developed by Google and the Model Context Protocol (MCP) introduced by Anthropic. While the evolution of LLM-based autonomous agents is rapidly accelerating, efficient interactions among these agents and their integration with external systems remain significant challenges. In modern AI systems, collaboration between autonomous agents and integration with external tools have become essential elements for building practical AI applications. A2A offers a standardized communication method that enables agents developed in heterogeneous environments to collaborate effectively, while MCP provides a structured I/O framework for agents to connect with external tools and resources. Prior studies have focused primarily on the features and applications of either A2A or MCP individually. In contrast, this study takes an integrated approach, exploring how the two protocols can complement each other to address interoperability issues and facilitate efficient collaboration within complex agent ecosystems. 

**Abstract (ZH)**: 本研究提供了由Google开发的开源Agent-to-Agent (A2A)协议和由Anthropic引入的Model Context Protocol (MCP)的技术分析和实现方法。虽然基于LLM的自主代理的演进正在加速，但这些代理之间的高效交互及其与外部系统的集成仍然是重大挑战。在现代AI系统中，自主代理之间的协作以及与外部工具的集成已成为构建实用AI应用的必备要素。A2A提供了标准化的通信方法，使来自异构环境的代理能够有效协作，而MCP则为代理与外部工具和资源的连接提供了一个结构化的I/O框架。前期研究主要关注A2A或MCP的特性和应用。相比之下，本研究采取了综合方法，探讨了这两种协议如何相互补充，以解决互操作性问题，并促进复杂代理生态系统内的高效协作。 

---
# Self-Challenging Language Model Agents 

**Title (ZH)**: 自我挑战语言模型代理 

**Authors**: Yifei Zhou, Sergey Levine, Jason Weston, Xian Li, Sainbayar Sukhbaatar  

**Link**: [PDF](https://arxiv.org/pdf/2506.01716)  

**Abstract**: Large language models are quickly becoming the foundation for intelligent agents that are capable of using tools. However, training such agents is challenging because it requires human creation and annotation of a diverse set of tasks, tools, and evaluation criteria. In this paper, we propose the Self-Challenging framework for training an agent on high-quality tasks that are generated by itself. The agent first plays the role of challenger and generates a task after interacting with the given tools. The tasks take the form of a novel general class of problems termed Code-as-Task, which are defined by an instruction, a verification function and solution and failure cases which serve as tests, allowing to filter only for high-quality tasks. The agent then takes an executor role and trains on those tasks with reinforcement learning using the evaluation feedback as a reward. Evaluation on two existing multi-turn tool-use agent benchmarks, M3ToolEval and TauBench, shows the Self-Challenging framework achieves over a two-fold improvement in Llama-3.1-8B-Instruct, despite using only self-generated training data. 

**Abstract (ZH)**: 自挑战框架：通过自我生成的高质量任务训练智能代理 

---
# A Descriptive and Normative Theory of Human Beliefs in RLHF 

**Title (ZH)**: 人类信念在RLHF中的描述性与规范性理论 

**Authors**: Sylee Dandekar, Shripad Deshmukh, Frank Chiu, W. Bradley Knox, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2506.01692)  

**Abstract**: Human preferences in RLHF are typically modeled as a function of the human's reward function or corresponding optimal state-action values. In this work, we propose that human beliefs about the capabilities of the agent being trained also play a key role in preference generation. We examine two questions related to this hypothesis, one descriptive and one normative, respectively: Do human labelers' beliefs about agent capabilities affect the preferences that they provide? And what is the ideal set of beliefs about an agent -- and resulting preferences -- for humans to have? We propose a new preference model that incorporates human beliefs and provide a normative theory that bounds the error on the final learned policy based on the \textit{mismatch} between the human's beliefs and an idealized set of beliefs. We then confirm via a human study that beliefs about agent capabilities do, in fact, significantly affect preferences and can be influenced through simple interventions. Additionally, we empirically show through synthetic experiments that it is often suboptimal for human preference labelers to assume agent optimality. Collectively, these results theoretically and empirically demonstrate how reducing the mismatch between human beliefs and agent capabilities can lead to more performant RLHF and point toward new best practices for RLHF practitioners. 

**Abstract (ZH)**: 人类在RLHF中的偏好通常被建模为人类奖励函数或相应最优状态行动值的函数。在本工作中，我们提出，训练中的智能体能力的人类信念也在偏好生成中扮演关键角色。我们分别探讨了与这一假设相关的两个问题，一个是描述性的，一个是规范性的：人类标注者关于智能体能力的信念是否影响他们提供的偏好？人类应该如何恰当地认为智能体的能力——以及相应的偏好——理想的信念集合是什么？我们提出了一种新的偏好模型，该模型包含了人类的信念，并提供了一种规范理论，该理论基于人类信念与理想化信念之间的差异来界定了最终学习策略的误差。然后，通过人类研究证实，对智能体能力的信念确实显著影响偏好，并可以通过简单的干预措施加以影响。此外，通过合成实验，我们实证表明，人类偏好标注者假设智能体最优化往往是次优的。这些结果从理论和实证上证明了减少人类信念与智能体能力之间的偏差如何能够提高RLHF的性能，并指出了RLHF从业者新的最佳实践。 

---
# Reasoning-Based Approach with Chain-of-Thought for Alzheimer's Detection Using Speech and Large Language Models 

**Title (ZH)**: 基于推理的链式思考方法在语音和大型语言模型辅助下的阿尔茨海默病检测 

**Authors**: Chanwoo Park, Anna Seo Gyeong Choi, Sunghye Cho, Chanwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01683)  

**Abstract**: Societies worldwide are rapidly entering a super-aged era, making elderly health a pressing concern. The aging population is increasing the burden on national economies and households. Dementia cases are rising significantly with this demographic shift. Recent research using voice-based models and large language models (LLM) offers new possibilities for dementia diagnosis and treatment. Our Chain-of-Thought (CoT) reasoning method combines speech and language models. The process starts with automatic speech recognition to convert speech to text. We add a linear layer to an LLM for Alzheimer's disease (AD) and non-AD classification, using supervised fine-tuning (SFT) with CoT reasoning and cues. This approach showed an 16.7% relative performance improvement compared to methods without CoT prompt reasoning. To the best of our knowledge, our proposed method achieved state-of-the-art performance in CoT approaches. 

**Abstract (ZH)**: 全球社会正迅速进入超老龄时代，老年人健康成为迫切关注的问题。老龄化人口增加了国家经济和家庭的负担。随着人口结构的转变，痴呆症病例显著增加。近期利用语音模型和大规模语言模型（LLM）的研究为痴呆症的诊断与治疗提供了新可能。我们的逐步推理（CoT）方法结合了语音和语言模型。过程始于自动语音识别，将语音转换为文本。我们在大型语言模型中添加了一层线性层，用于阿尔茨海默病（AD）和非AD分类，并采用包含逐步推理和提示的有监督微调（SFT）方法。这种方法在没有CoT提示推理的方法中相对提高了16.7%的性能。据我们所知，我们的方法在逐步推理方法中达到了最先进的性能。 

---
# K12Vista: Exploring the Boundaries of MLLMs in K-12 Education 

**Title (ZH)**: K12Vista: 探索大型语言模型在K-12教育中的边界 

**Authors**: Chong Li, Chenglin Zhu, Tao Zhang, Mingan Lin, Zenan Zhou, Jian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.01676)  

**Abstract**: Multimodal large language models have demonstrated remarkable reasoning capabilities in various visual tasks. However, their abilities in K12 scenarios are still systematically underexplored. Previous studies suffer from various limitations including narrow subject coverage, insufficient data scale, lack of diversity in question types, and naive answer-centric evaluation method, resulting in insufficient exploration of model capabilities. To address these gaps, we propose K12Vista, the most comprehensive multimodal benchmark for Chinese K12 subject knowledge understanding and reasoning to date, featuring 33,000 questions across five core subjects from primary to high school and three question types. Moreover, beyond the final outcome, we are also concerned with the correctness of MLLMs' reasoning processes. For this purpose, we meticulously compiles errors from MLLMs' reasoning processes and leverage an automated data pipeline to construct K12-PEM-800K, the largest process evaluation dataset offering detailed step-by-step judgement annotations for MLLMs' reasoning. Subsequently, we developed K12-PEM, an advanced process evaluation model that integrates an overall assessment of both the reasoning process and answer correctness. Moreover, we also introduce K12-PEBench, the first high-quality, human-annotated benchmark specifically designed for evaluating abilities of reasoning process this http URL experiments reveal that current MLLMs exhibit significant flaws when reasoning within K12Vista, providing critical insights for the development of more capable this http URL open our resources at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型在各种视觉任务中展现了出色的推理能力，但在K12场景中的能力仍系统性地未被充分探索。之前的研究存在诸多限制，包括狭隘的主题覆盖、不足的数据规模、问题类型缺乏多样性以及简单的答案为中心的评估方法，导致未能充分探索模型能力。为弥补这些差距，我们提出K12Vista，目前已知最全面的多模态基准，用于汉语K12科目知识理解与推理，包含从小学至高中五大核心学科的33,000个问题和三种问题类型。此外，我们不仅关注最终结果，还关注MLLMs推理过程的正确性。为此，我们仔细收集了MLLMs推理过程中的错误，并利用自动化数据管道构建了K12-PEM-800K，这是迄今为止最大的过程评估数据集，提供详细的逐步骤判断注释，以评估MLLMs的推理过程。随后，我们开发了K12-PEM，一种先进的过程评估模型，综合评估推理过程和答案的正确性。此外，我们还引入了K12-PEBench，这是首个专门用于评估推理过程能力的高质量、人类标注基准。实验结果表明，当前MLLMs在K12Vista中的推理存在明显缺陷，为开发更强大的模型提供了关键见解。我们已将资源开放于此网址。 

---
# MLA-Trust: Benchmarking Trustworthiness of Multimodal LLM Agents in GUI Environments 

**Title (ZH)**: MLA-Trust: 多模态LLM代理在GUI环境中信任worthiness基准评估 

**Authors**: Xiao Yang, Jiawei Chen, Jun Luo, Zhengwei Fang, Yinpeng Dong, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01616)  

**Abstract**: The emergence of multimodal LLM-based agents (MLAs) has transformed interaction paradigms by seamlessly integrating vision, language, action and dynamic environments, enabling unprecedented autonomous capabilities across GUI applications ranging from web automation to mobile systems. However, MLAs introduce critical trustworthiness challenges that extend far beyond traditional language models' limitations, as they can directly modify digital states and trigger irreversible real-world consequences. Existing benchmarks inadequately tackle these unique challenges posed by MLAs' actionable outputs, long-horizon uncertainty and multimodal attack vectors. In this paper, we introduce MLA-Trust, the first comprehensive and unified framework that evaluates the MLA trustworthiness across four principled dimensions: truthfulness, controllability, safety and privacy. We utilize websites and mobile applications as realistic testbeds, designing 34 high-risk interactive tasks and curating rich evaluation datasets. Large-scale experiments involving 13 state-of-the-art agents reveal previously unexplored trustworthiness vulnerabilities unique to multimodal interactive scenarios. For instance, proprietary and open-source GUI-interacting MLAs pose more severe trustworthiness risks than static MLLMs, particularly in high-stakes domains; the transition from static MLLMs into interactive MLAs considerably compromises trustworthiness, enabling harmful content generation in multi-step interactions that standalone MLLMs would typically prevent; multi-step execution, while enhancing the adaptability of MLAs, involves latent nonlinear risk accumulation across successive interactions, circumventing existing safeguards and resulting in unpredictable derived risks. Moreover, we present an extensible toolbox to facilitate continuous evaluation of MLA trustworthiness across diverse interactive environments. 

**Abstract (ZH)**: 多模态LLM代理的可信度框架：MLA-Trust 

---
# PGPO: Enhancing Agent Reasoning via Pseudocode-style Planning Guided Preference Optimization 

**Title (ZH)**: PGPO: 假码风格规划引导偏好优化增强代理推理 

**Authors**: Zouying Cao, Runze Wang, Yifei Yang, Xinbei Ma, Xiaoyong Zhu, Bo Zheng, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01475)  

**Abstract**: Large Language Model (LLM) agents have demonstrated impressive capabilities in handling complex interactive problems. Existing LLM agents mainly generate natural language plans to guide reasoning, which is verbose and inefficient. NL plans are also tailored to specific tasks and restrict agents' ability to generalize across similar tasks. To this end, we explore pseudocode-style plans (P-code Plan) to capture the structural logic of reasoning. We find that P-code Plan empowers LLM agents with stronger generalization ability and more efficiency. Inspired by this finding, we propose a pseudocode-style Planning Guided Preference Optimization method called PGPO for effective agent learning. With two planning-oriented rewards, PGPO further enhances LLM agents' ability to generate high-quality P-code Plans and subsequent reasoning. Experiments show that PGPO achieves superior performance on representative agent benchmarks and outperforms the current leading baselines. Analyses reveal the advantage of PGPO in reducing action errors and omissions during reasoning. 

**Abstract (ZH)**: 大型语言模型（LLM）代理在处理复杂交互问题方面展示了令人印象深刻的 capability。现有的LLM代理主要生成自然语言计划以指导推理，这种方式冗长且低效。自然语言计划也针对特定任务，限制了代理在类似任务上的泛化能力。为此，我们探索伪代码风格的计划（P-code Plan）以捕捉推理的结构性逻辑。我们发现P-code Plan增强了LLM代理的泛化能力和效率。受此发现的启发，我们提出了一种伪代码风格的规划引导偏好优化方法——PGPO，以实现有效的代理学习。通过两种面向规划的奖励，PGPO进一步增强了LLM代理生成高质量P-code计划及后续推理的能力。实验结果显示，PGPO在代表性的代理基准测试中取得了优越的性能，并优于当前领先的基线方法。分析表明，PGPO在推理过程中减少行为错误和遗漏方面具有优势。 

---
# AI Scientists Fail Without Strong Implementation Capability 

**Title (ZH)**: AI科学家缺乏强大的实施能力将难以取得成功。 

**Authors**: Minjun Zhu, Qiujie Xie, Yixuan Weng, Jian Wu, Zhen Lin, Linyi Yang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01372)  

**Abstract**: The emergence of Artificial Intelligence (AI) Scientist represents a paradigm shift in scientific discovery, with large language models (LLMs) taking the lead as the primary executor in the entire scientific workflow from idea generation to experiment implementation. Recent AI Scientist studies demonstrate sufficient capabilities for independent scientific discovery, with the generated research reports gaining acceptance at the ICLR 2025 workshop and ACL 2025, arguing that a human-level AI Scientist, capable of uncovering phenomena previously unknown to humans, may be imminent. Despite this substantial progress, AI Scientist has yet to produce a groundbreaking achievement in the domain of computer science on par with automated scientific tools. Based on extensive quantitative evidence from existing benchmarks in complex engineering tasks and a systematic evaluation assess 28 research papers generated by five advanced AI Scientist systems, we argue that \textbf{the fundamental bottleneck for AI Scientists lies in their capability to execute the requisite verification procedures.} Current AI Scientist systems lack the execution capabilities needed to execute rigorous experiments and produce high-quality scientific papers. To better illustrate the root cause of this \textbf{implementation gap}, we provide an in-depth discussion on the fundamental limitations of AI Scientist. This position paper aims to call for the participants in the community to bridge the implementation gap. 

**Abstract (ZH)**: 人工智能科学家的兴起代表了科学发现范式的转变，大规模语言模型（LLMs）在从想法生成到实验实施的整个科学工作流程中 rôle leading。近期的人工智能科学家研究显示了独立进行科学研究的能力，生成的研究报告在ICLR 2025研讨会和ACL 2025会议上获得接受，这表明具备人类水平的人工智能科学家，能揭示此前人类未知的现象，可能即将来临。尽管取得了这些显著进展，人工智能科学家在计算机科学领域尚未取得与自动化科学工具相媲美的突破性成就。基于复杂工程任务现有基准的大量定量证据以及对五个先进人工智能科学家系统生成的28篇研究论文的系统评估，我们认为 \textbf{人工智能科学家的核心瓶颈在于其执行必要的验证程序的能力。} 当前的人工智能科学家系统缺乏执行严谨实验和产生高质量科学论文所需的能力。为了更好地说明这一 \textbf{实现差距} 的根本原因，我们深入讨论了人工智能科学家的基本局限性。这篇立场论文旨在呼吁社区成员弥合实现差距。 

---
# An Empirical Study of Group Conformity in Multi-Agent Systems 

**Title (ZH)**: 多Agent系统中群体 conformity 的实证研究 

**Authors**: Min Choi, Keonwoo Kim, Sungwon Chae, Sangyeob Baek  

**Link**: [PDF](https://arxiv.org/pdf/2506.01332)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled multi-agent systems that simulate real-world interactions with near-human reasoning. While previous studies have extensively examined biases related to protected attributes such as race, the emergence and propagation of biases on socially contentious issues in multi-agent LLM interactions remain underexplored. This study explores how LLM agents shape public opinion through debates on five contentious topics. By simulating over 2,500 debates, we analyze how initially neutral agents, assigned a centrist disposition, adopt specific stances over time. Statistical analyses reveal significant group conformity mirroring human behavior; LLM agents tend to align with numerically dominant groups or more intelligent agents, exerting a greater influence. These findings underscore the crucial role of agent intelligence in shaping discourse and highlight the risks of bias amplification in online interactions. Our results emphasize the need for policy measures that promote diversity and transparency in LLM-generated discussions to mitigate the risks of bias propagation within anonymous online environments. 

**Abstract (ZH)**: Recent Advances in Large Language Models: Exploring Bias Shaping in Multi-Agent Systems Through Debates on Controversial Topics 

---
# ORMind: A Cognitive-Inspired End-to-End Reasoning Framework for Operations Research 

**Title (ZH)**: ORMind：一种受认知启发的端到端优化推理框架 

**Authors**: Zhiyuan Wang, Bokui Chen, Yinya Huang, Qingxing Cao, Ming He, Jianping Fan, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01326)  

**Abstract**: Operations research (OR) is widely deployed to solve critical decision-making problems with complex objectives and constraints, impacting manufacturing, logistics, finance, and healthcare outcomes. While Large Language Models (LLMs) have shown promising results in various domains, their practical application in industry-relevant operations research (OR) problems presents significant challenges and opportunities. Preliminary industrial applications of LLMs for operations research face two critical deployment challenges: 1) Self-correction focuses on code syntax rather than mathematical accuracy, causing costly errors; 2) Complex expert selection creates unpredictable workflows that reduce transparency and increase maintenance costs, making them impractical for time-sensitive business applications. To address these business limitations, we introduce ORMind, a cognitive-inspired framework that enhances optimization through counterfactual reasoning. Our approach emulates human cognition, implementing an end-to-end workflow that systematically transforms requirements into mathematical models and executable solver code. It is currently being tested internally in Lenovo's AI Assistant, with plans to enhance optimization capabilities for both business and consumer customers. Experiments demonstrate that ORMind outperforms existing methods, achieving a 9.5\% improvement on the NL4Opt dataset and a 14.6\% improvement on the ComplexOR dataset. 

**Abstract (ZH)**: 运筹学（OR）广泛应用于解决具有复杂目标和约束的关键决策问题，影响制造、物流、金融和医疗成果。尽管大型语言模型（LLMs）在各种领域展现了有前景的结果，但它们在与行业相关的运筹学（OR）问题中的实际应用面临着重大挑战和机遇。初步将LLMs应用于运筹学的工业应用面临两个关键部署挑战：1）自我纠正关注代码语法而非数学准确性，导致成本高昂的错误；2）复杂的专家选择创建不可预测的工作流，降低透明度并增加维护成本，使它们对于时间敏感的商业应用不实用。为解决这些商业局限，我们引入了ORMind，一种受认知启发的框架，通过反事实推理增强优化。我们的方法模拟人类认知，实现端到端的工作流，系统地将需求转换为数学模型和可执行求解器代码。目前，它正在联想AI助手内部进行测试，并计划增强面向企业和消费者的优化能力。实验表明，ORMind优于现有方法，在NL4Opt数据集上提升了9.5%，在ComplexOR数据集上提升了14.6%。 

---
# RAISE: Reasoning Agent for Interactive SQL Exploration 

**Title (ZH)**: RAISE: 交互式SQL探索的推理代理 

**Authors**: Fernando Granado, Roberto Lotufo, Jayr Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2506.01273)  

**Abstract**: Recent advances in large language models (LLMs) have propelled research in natural language interfaces to databases. However, most state-of-the-art text-to- SQL systems still depend on complex, multi-stage pipelines. This work proposes a novel agentic framework that unifies schema linking, query generation, and itera- tive refinement within a single, end-to-end component. By leveraging the intrinsic reasoning abilities of LLMs, our method emulates how humans answer questions when working with unfamiliar databases: understanding the data by formulating hypotheses, running dynamic queries to validate them, reasoning over the results, and revising outputs based on observed results. Crucially, our approach intro- duces a new strategy for scaling test-time computation in text-to-SQL: we scale the depth of interactive database exploration and reflection. This shift enables the model to allocate computation dynamically to better understand the data, especially useful in ambiguous and underspecified scenarios. Our experiments show that it improved the Execution Accuracy (EX) from 44.8% to 56.5% on the challenging BIRD dataset using DeepSeek-R1-Distill-Llama-70B. Fur- thermore, when equipped with steps to add more diversity to the answers, our agent achieves a Best-of-N accuracy of 81.8% with 8 rounds of candidate gener- ation, rivaling the 82.79% achieved by the top-ranked published solution, while reducing engineering complexity. These findings position our unified framework as a promising alternative for building natural language interfaces to databases. 

**Abstract (ZH)**: 近年来，大规模语言模型的进展推动了自然语言数据库接口的研究。然而，大多数最先进的文本到SQL系统仍然依赖于复杂的多阶段管道。本工作提出了一种新的代理人框架，该框架在单一端到端组件内统一了模式链接、查询生成和迭代优化。通过利用大规模语言模型的内在推理能力，该方法模拟了人类在处理陌生数据库时的回答方式：通过提出假设、运行动态查询来验证它们、对结果进行推理，并根据观察结果进行输出修正。 crucially，我们的方法引入了一种新的在文本到SQL中扩展测试时计算的新策略：我们扩大了交互式数据库探索和反思的深度。这一转变使得模型能够动态分配计算，更好地理解数据，尤其是在模糊和未明确规定的情况下。我们的实验表明，使用DeepSeek-R1-Distill-Llama-70B，在具有挑战性的BIRD数据集上，该方法的执行准确性（EX）从44.8%提高到56.5%。此外，当通过增加答案多样性步骤进行增强时，我们的代理在8轮候选生成后实现了81.8%的最佳准确性，这一成绩与排名第一的已发表解决方案的82.79%相当，同时减少了工程复杂度。这些发现将我们统一的框架定位为构建自然语言数据库接口的有希望的替代方案。 

---
# CleanS2S: Single-file Framework for Proactive Speech-to-Speech Interaction 

**Title (ZH)**: CleanS2S: 单文件框架实现主动语音到语音交互 

**Authors**: Yudong Lu, Yazhe Niu, Shuai Hu, Haolin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01268)  

**Abstract**: CleanS2S is a framework for human-like speech-to-speech interaction that advances conversational AI through single-file implementation and proactive dialogue capabilities. Our system integrates automatic speech recognition, large language models, and text-to-speech synthesis into a unified pipeline with real-time interruption handling, achieving low transition latency through full-duplex websocket connections and non-blocking I/O. Beyond conventional chatbot paradigms, we pioneer a proactive interaction mechanism, which combines memory systems with Subjective Action Judgement module, enabling five human-like response strategies: interruption, refusal, deflection, silence, and standard response. The memory module dynamically aggregates historical, and contextual data to inform interaction decisions. This approach breaks the rigid turn-based convention by allowing system-initiated dialog control and context-aware response selection. And we propose Action Judgement SFT that assesses input streams for responses strategies. The framework's single-file implementation with atomic configurations offers researchers unprecedented transparency and extensibility for interaction agents. The code of CleanS2S is released at \this https URL. 

**Abstract (ZH)**: CleanS2S是一种通过单文件实现和主动对话能力推进会话AI的人类级语音到语音交互框架。 

---
# ChemAU: Harness the Reasoning of LLMs in Chemical Research with Adaptive Uncertainty Estimation 

**Title (ZH)**: ChemAU: 利用自适应不确定性估计在化学研究中 harness LLMs 的推理能力 

**Authors**: Xinyi Liu, Lipeng Ma, Yixuan Li, Weidong Yang, Qingyuan Zhou, Jiayi Song, Shuhao Li, Ben Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.01116)  

**Abstract**: Large Language Models (LLMs) are widely used across various scenarios due to their exceptional reasoning capabilities and natural language understanding. While LLMs demonstrate strong performance in tasks involving mathematics and coding, their effectiveness diminishes significantly when applied to chemistry-related problems. Chemistry problems typically involve long and complex reasoning steps, which contain specific terminology, including specialized symbol systems and complex nomenclature conventions. These characteristics often cause general LLMs to experience hallucinations during the reasoning process due to their lack of specific knowledge. However, existing methods are struggling to effectively leverage chemical expertise and formulas. Moreover, current uncertainty estimation methods, designed to mitigate potential reasoning errors, are unable to precisely identify specific steps or key knowledge. In this work, we propose a novel framework called ChemAU, which incorporates our adaptive uncertainty estimation method that applies different uncertainty values based on the position of reasoning steps within the whole reasoning chain. Leveraging this method, ChemAU identifies gaps in chemistry knowledge and precisely supplements chemical expertise with the specialized domain model, thereby correcting and updating the previously flawed reasoning chain. Our experiments with three popular LLMs across three chemistry datasets demonstrate that ChemAU significantly enhances both reasoning accuracy and uncertainty estimation. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种场景中广泛应用，得益于其卓越的推理能力和自然语言理解能力。虽然LLMs在涉及数学和编程的任务中表现出色，但在应用于化学相关问题时，其有效性显著下降。化学问题通常涉及复杂的推理步骤，包含特定的专业术语、特殊符号系统和复杂的命名 conventions。这些特点往往导致一般性的LLMs在推理过程中出现幻觉，因为它们缺乏具体的知识。然而，现有的方法难以有效利用化学专业知识和公式。此外，当前用于降低推理错误的风险估计方法，无法精确识别具体步骤或关键知识。在本工作中，我们提出了一种名为ChemAU的新框架，该框架结合了我们根据整个推理链中推理步骤的位置应用不同不确定性值的自适应不确定性估计方法。通过这种方法，ChemAU能够识别化学知识的空白，并精确补充化学专业知识，从而纠正和更新先前错误的推理链。我们在三个流行的LLM和三个化学数据集中进行的实验表明，ChemAU显著提高了推理准确性和不确定性估计。 

---
# MCP-Zero: Proactive Toolchain Construction for LLM Agents from Scratch 

**Title (ZH)**: MCP-Zero: 从 scratch 开始的 LLM 代理前瞻工具链构建 

**Authors**: Xiang Fei, Xiawu Zheng, Hao Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01056)  

**Abstract**: Function-calling has enabled large language models (LLMs) to act as tool-using agents, but injecting thousands of tool schemas into the prompt is costly and error-prone. We introduce MCP-Zero, a proactive agent framework that lets the LLM itself decide when and which external tools to retrieve, thereby assembling a task-specific toolchain from scratch. The framework is built upon three components: (1) Proactive Tool Request, where the model emits a structured $\left<\operatorname{tool\_assistant}\right>$ block that explicitly specifies the desired server and task; (2) Hierarchical Vector Routing, a coarse-to-fine retrieval algorithm that first selects candidate servers and then ranks tools within each server based on the semantic similarity; (3) Iterative Proactive Invocation, enabling multi-round, cross-domain toolchain construction with minimal context overhead, and allowing the model to iteratively revise its request when the returned tools are insufficient. To evaluate our approach we also compile MCP-tools, a retrieval dataset comprising 308 MCP servers and 2,797 tools extracted from the official Model-Context-Protocol repository and normalized into a unified JSON schema. Experiments show that MCP-Zero (i) effectively addresses the context overhead problem of existing methods and accurately selects the correct tool from a pool of nearly 3,000 candidates (248.1k tokens); (ii) reduces token consumption by 98\% on the APIbank while maintaining high accuracy; and (iii) supports multi-turn tool invocation with consistent accuracy across rounds. The code and dataset will be released soon. 

**Abstract (ZH)**: MCP-Zero：一种主动式代理框架，让大型语言模型自主决定何时及调用哪些外部工具，从而构建任务特定的工具链 

---
# IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory 

**Title (ZH)**: IRT-Router:有效的可解释多LLM路由方法基于项目反应理论 

**Authors**: Wei Song, Zhenya Huang, Cheng Cheng, Weibo Gao, Bihan Xu, GuanHao Zhao, Fei Wang, Runze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01048)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance across a wide range of natural language tasks. However, selecting the optimal LLM to respond to a user query often necessitates a delicate balance between performance and cost. While powerful models deliver better results, they come at a high cost, whereas smaller models are more cost-effective but less capable. To address this trade-off, we propose IRT-Router, a multi-LLM routing framework that efficiently routes user queries to the most suitable LLM. Inspired by Item Response Theory (IRT), a psychological measurement methodology, IRT-Router explicitly models the relationship between LLM capabilities and user query attributes. This not only enables accurate prediction of response performance but also provides interpretable insights, such as LLM abilities and query difficulty. Additionally, we design an online query warm-up technique based on semantic similarity, further enhancing the online generalization capability of IRT-Router. Extensive experiments on 20 LLMs and 12 datasets demonstrate that IRT-Router outperforms most baseline methods in terms of effectiveness and interpretability. Its superior performance in cold-start scenarios further confirms the reliability and practicality of IRT-Router in real-world applications. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言任务中展现了出色的表现。然而，选择最适合响应用户查询的LLM往往需要在性能和成本之间取得微妙的平衡。尽管强大的模型能提供更好的结果，但成本高昂，而较小的模型成本效益更高但能力较弱。为解决这一权衡问题，我们提出了一种多LLM路由框架IRT-Router，该框架能够高效地将用户查询导向最适合的LLM。借鉴项目反应理论（IRT）的心理测量方法，IRT-Router明确建模了LLM能力和用户查询属性之间的关系，不仅能够准确预测响应性能，还能提供可解释的洞察，如LLM能力和查询难度。此外，我们还设计了一种基于语义相似性的在线查询预热技术，进一步增强了IRT-Router的在线泛化能力。在20个LLM和12个数据集上的 extensive 实验结果显示，IRT-Router 在有效性与可解释性方面均优于大多数基线方法。在冷启动场景中的卓越性能进一步证实了IRT-Router 在实际应用中的可靠性和实用性。代码可在以下网址获取：this https URL。 

---
# Unlocking Personalized Knowledge in Federated Large Language Model: The Power of Mixture of Experts 

**Title (ZH)**: 解锁联邦大型语言模型中的个性化知识：混合专家的力量 

**Authors**: Fan Liu, Bikang Pan, Zhongyi Wang, Xi Yao, Xiaoying Tang, Jingya Wang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00965)  

**Abstract**: The Mixture of Experts (MoE) architecture has emerged as a prominent strategy for scaling large language models (LLMs), effectively leveraging sparse activation and facilitating task-specific personalization. However, current federated learning (FL) approaches are primarily designed for dense models, making them unable to directly exploit the sparsity inherent in MoE architectures. Treating MoE models as dense networks in federated scenarios results in excessive communication overhead and computational costs, undermining the potential for personalized knowledge sharing. To address these challenges, we propose FLEx (Federated LLMs with Personalized Experts), a novel federated learning framework explicitly tailored for MoE-based LLMs. FLEx efficiently personalizes by pruning the global MoE model to keep only one expert per client, and employs an adaptive gating mechanism to reintegrate these personalized experts into the pre-trained MoE layers, ensuring the original backbone architecture remains unchanged. These personalized experts are trained with local data and stored locally on each client, while the shared modules are aggregated globally. Extensive evaluations on diverse instruction-based datasets under non-IID conditions consistently demonstrate that FLEx outperforms existing federated baselines. Our code is available at this https URL. 

**Abstract (ZH)**: FLEx（面向混合专家的联邦大语言模型个性化框架） 

---
# Conformal Arbitrage: Risk-Controlled Balancing of Competing Objectives in Language Models 

**Title (ZH)**: conformal arbitrage：在语言模型中控制风险的 competing 目标平衡 

**Authors**: William Overman, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2506.00911)  

**Abstract**: Modern language model deployments must often balance competing objectives, for example, helpfulness versus harmlessness, cost versus accuracy, and reward versus safety. We introduce Conformal Arbitrage, a post hoc framework that learns a data driven threshold to mediate between a Primary model optimized for a primary objective and a more conservative Guardian which could be another model or a human domain expert aligned with a guardrail objective. The threshold is calibrated with conformal risk control, yielding finite sample, distribution free guarantees that the long run frequency of undesirable events, such as factual errors or safety violations, does not exceed a user specified quota. Because Conformal Arbitrage operates wholly at the API level, without requiring access to model logits or updating model weights, it complements weight based alignment techniques and integrates seamlessly with existing cost aware cascades. Empirically, Conformal Arbitrage traces an efficient frontier, allowing users to define an acceptable performance level for one objective while maximizing utility in another. We observe that our method outperforms, in terms of accuracy, cost matched random routing between models. These properties make Conformal Arbitrage a practical, theoretically grounded tool for trustworthy and economical deployment of large language models across a broad range of potentially competing objectives. 

**Abstract (ZH)**: 现代语言模型部署必须平衡多种互斥目标，例如有益性与安全性、成本与准确性、奖励与安全性。我们介绍了一种后验框架——Conformal Arbitrage，该框架学习一个基于数据的阈值，以调解专门为某种目标优化的主模型（Primary model）与一个更为保守的守护者模型（Guardian），后者可能是另一个模型或与护栏目标保持一致的人类领域专家。该阈值通过符合风险控制进行校准，从而在有限样本、分布无关的情况下保证长期频率不佳事件（如事实性错误或安全性违规）的发生率不超过用户指定的限额。由于Conformal Arbitrage完全在API层面运作，无需访问模型logits或更新模型权重，因此它补充了基于权重的对齐技术，并能无缝集成到现有的成本感知级联中。实证研究显示，Conformal Arbitrage能够在保持一个目标性能的同时，最大化另一个目标的效用。我们发现，这种方法在准确性方面优于模型间按成本匹配的随机路由。这些特性使得Conformal Arbitrage成为一个实用且有理论支撑的工具，可用于众多潜在互斥目标下大规模语言模型的可靠且经济的部署。 

---
# Enhancing LLM Reasoning for Time Series Classification by Tailored Thinking and Fused Decision 

**Title (ZH)**: 针对时间序列分类的定制化思考与融合决策增强大型语言模型推理能力 

**Authors**: Jiahui Zhou, Dan Li, Lin Li, Zhuomin Chen, Shunyu Wu, Haozheng Ye, Jian Lou, Costas J. Spanos  

**Link**: [PDF](https://arxiv.org/pdf/2506.00807)  

**Abstract**: The reasoning capabilities of large language models (LLMs) have significantly advanced their performance by enabling in-depth understanding of diverse tasks. With growing interest in applying LLMs to the time series domain, this has proven nontrivial, as evidenced by the limited efficacy of straightforwardly adapting text-domain reasoning techniques. Although recent work has shown promise in several time series tasks, further leveraging advancements in LLM reasoning remains under-explored for time series classification (TSC) tasks, despite their prevalence and significance in many real-world applications. In this paper, we propose ReasonTSC, a novel framework designed to effectively leverage LLM reasoning for time series classification through both a multi-turn reasoning and a fused decision-making strategy tailored to TSC. Rather than straightforwardly applying existing reasoning techniques or relying solely on LLMs' built-in reasoning capabilities, ReasonTSC first steers the model to think over the essential characteristics of time series data. Next, it integrates predictions and confidence scores from plug-in classifiers, e.g., domain-specific time series models, as in-context examples. Finally, ReasonTSC guides the LLM through a structured reasoning process: it evaluates the initial assessment, backtracks to consider alternative hypotheses, and compares their merits before arriving at a final classification. Extensive experiments and systematic ablation studies demonstrate that ReasonTSC consistently outperforms both existing time series reasoning baselines and plug-in models, and is even capable of identifying and correcting plug-in models' false predictions. 

**Abstract (ZH)**: 大型语言模型在时间序列分类中的推理能力：一种新型框架通过多轮推理和融合决策策略有效利用大型语言模型的推理能力 

---
# GeoChain: Multimodal Chain-of-Thought for Geographic Reasoning 

**Title (ZH)**: GeoChain：多模态链式推理在地理推理中的应用 

**Authors**: Sahiti Yerramilli, Nilay Pande, Rynaa Grover, Jayant Sravan Tamarapalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.00785)  

**Abstract**: This paper introduces GeoChain, a large-scale benchmark for evaluating step-by-step geographic reasoning in multimodal large language models (MLLMs). Leveraging 1.46 million Mapillary street-level images, GeoChain pairs each image with a 21-step chain-of-thought (CoT) question sequence (over 30 million Q&A pairs). These sequences guide models from coarse attributes to fine-grained localization across four reasoning categories - visual, spatial, cultural, and precise geolocation - annotated by difficulty. Images are also enriched with semantic segmentation (150 classes) and a visual locatability score. Our benchmarking of contemporary MLLMs (GPT-4.1 variants, Claude 3.7, Gemini 2.5 variants) on a diverse 2,088-image subset reveals consistent challenges: models frequently exhibit weaknesses in visual grounding, display erratic reasoning, and struggle to achieve accurate localization, especially as the reasoning complexity escalates. GeoChain offers a robust diagnostic methodology, critical for fostering significant advancements in complex geographic reasoning within MLLMs. 

**Abstract (ZH)**: GeoChain：大规模基准用于评估多模态大语言模型的逐步地理推理能力 

---
# Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning 

**Title (ZH)**: Jailbreak-R1：通过强化学习探索LLMs的脱 Gleam 能力 

**Authors**: Weiyang Guo, Zesheng Shi, Zhuo Li, Yequan Wang, Xuebo Liu, Wenya Wang, Fangming Liu, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00782)  

**Abstract**: As large language models (LLMs) grow in power and influence, ensuring their safety and preventing harmful output becomes critical. Automated red teaming serves as a tool to detect security vulnerabilities in LLMs without manual labor. However, most existing methods struggle to balance the effectiveness and diversity of red-team generated attack prompts. To address this challenge, we propose \ourapproach, a novel automated red teaming training framework that utilizes reinforcement learning to explore and generate more effective attack prompts while balancing their diversity. Specifically, it consists of three training stages: (1) Cold Start: The red team model is supervised and fine-tuned on a jailbreak dataset obtained through imitation learning. (2) Warm-up Exploration: The model is trained in jailbreak instruction following and exploration, using diversity and consistency as reward signals. (3) Enhanced Jailbreak: Progressive jailbreak rewards are introduced to gradually enhance the jailbreak performance of the red-team model. Extensive experiments on a variety of LLMs show that \ourapproach effectively balances the diversity and effectiveness of jailbreak prompts compared to existing methods. Our work significantly improves the efficiency of red team exploration and provides a new perspective on automated red teaming. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的增强，确保其安全并防止有害输出变得至关重要。自动红队攻击作为一种工具，可以在不依赖人工的情况下检测LLMs的安全漏洞。然而，现有方法大多难以平衡红队生成攻击提示的有效性和多样性。为此，我们提出了\ourapproach，一种利用强化学习探索并生成更有效攻击提示的新颖自动红队训练框架，同时平衡其多样性。具体而言，该框架包括三个训练阶段：（1）冷启动：红队模型通过模仿学习获得的脱缰数据集进行监督微调。（2）热启动探索：模型在脱缰指令跟随和探索中进行训练，使用多样性和一致性作为奖励信号。（3）增强脱缰：逐步引入渐进式脱缰奖励以逐步提高红队模型的脱缰性能。在多种LLMs上的广泛实验表明，\ourapproach能更有效地平衡脱缰提示的有效性和多样性，相较于现有方法。我们的工作显著提高了红队探索的效率，并为自动红队攻击提供了新的视角。 

---
# CoP: Agentic Red-teaming for Large Language Models using Composition of Principles 

**Title (ZH)**: CoP: 基于原则组成的大语言模型自主红队技术 

**Authors**: Chen Xiong, Pin-Yu Chen, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2506.00781)  

**Abstract**: Recent advances in Large Language Models (LLMs) have spurred transformative applications in various domains, ranging from open-source to proprietary LLMs. However, jailbreak attacks, which aim to break safety alignment and user compliance by tricking the target LLMs into answering harmful and risky responses, are becoming an urgent concern. The practice of red-teaming for LLMs is to proactively explore potential risks and error-prone instances before the release of frontier AI technology. This paper proposes an agentic workflow to automate and scale the red-teaming process of LLMs through the Composition-of-Principles (CoP) framework, where human users provide a set of red-teaming principles as instructions to an AI agent to automatically orchestrate effective red-teaming strategies and generate jailbreak prompts. Distinct from existing red-teaming methods, our CoP framework provides a unified and extensible framework to encompass and orchestrate human-provided red-teaming principles to enable the automated discovery of new red-teaming strategies. When tested against leading LLMs, CoP reveals unprecedented safety risks by finding novel jailbreak prompts and improving the best-known single-turn attack success rate by up to 19.0 times. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在各个领域引发了变革性的应用，从开源到专有LLMs不等。然而，旨在通过欺骗目标LLMs生成有害和风险回答来破坏安全对齐和用户合规性的jailbreak攻击正变得日益紧迫。针对LLMs的红队演练是前瞻人工智能技术发布前主动探索潜在风险和易出错实例的实践。本文提出了一种代理工作流，通过原则组合框架（CoP）自动并扩展LLMs的红队演练过程，其中人类用户提供一组红队演练原则作为指令，由AI代理自动协调有效的红队演练策略并生成jailbreak提示。与现有红队演练方法不同，我们的CoP框架提供了一个统一且可扩展的框架来包容和协调由人类提供的红队演练原则，以实现新的红队演练策略的自动化发现。当与领先的LLMs进行测试时，CoP通过发现新颖的jailbreak提示并提高最佳已知单轮攻击成功率19.0倍，揭示了前所未有的安全风险。 

---
# Do not Abstain! Identify and Solve the Uncertainty 

**Title (ZH)**: 不要回避！识别并解决不确定性 

**Authors**: Jingyu Liu, Jingquan Peng, xiaopeng Wu, Xubin Li, Tiezheng Ge, Bo Zheng, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00780)  

**Abstract**: Despite the widespread application of Large Language Models (LLMs) across various domains, they frequently exhibit overconfidence when encountering uncertain scenarios, yet existing solutions primarily rely on evasive responses (e.g., "I don't know") overlooks the opportunity of identifying and addressing the uncertainty to generate more satisfactory responses. To systematically investigate and improve LLMs' ability of recognizing and addressing the source of uncertainty, we introduce \textbf{ConfuseBench}, a benchmark mainly focus on three types of uncertainty: document scarcity, limited capability, and query ambiguity. Experiments with ConfuseBench reveal that current LLMs struggle to accurately identify the root cause of uncertainty and solve it. They prefer to attribute uncertainty to query ambiguity while overlooking capability limitations, especially for those weaker models. To tackle this challenge, we first generate context-aware inquiries that highlight the confusing aspect of the original query. Then we judge the source of uncertainty based on the uniqueness of the inquiry's answer. Further we use an on-policy training method, InteractDPO to generate better inquiries. Experimental results demonstrate the efficacy of our approach. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各个领域得到了广泛应用，但在遭遇不确定性场景时经常表现出过度自信，现有的解决方案主要依赖于规避性回应（例如，“我不知道”），而忽视了识别和解决不确定性以生成更满意回应的机会。为了系统地调查和提高LLMs识别和应对不确定性来源的能力，我们引入了\textbf{ConfuseBench}基准测试，该基准测试主要关注三种类型的不确定性：文档稀缺性、能力限制和查询模糊性。使用ConfuseBench的实验表明，当前的LLMs在准确识别不确定性根源并解决问题方面存在困难。它们往往将不确定性归因于查询模糊性，而忽视了能力限制，尤其是在较弱的模型中。为应对这一挑战，我们首先生成上下文相关的问询，突显原始查询中的混淆方面，然后根据问询答案的独特性来判断不确定性来源。进一步使用基于策略的训练方法InteractDPO来生成更好的问询。实验结果证明了我们方法的有效性。 

---
# Alignment Revisited: Are Large Language Models Consistent in Stated and Revealed Preferences? 

**Title (ZH)**: Alignment再探：大型语言模型在明示和隐含偏好上保持一致吗？ 

**Authors**: Zhuojun Gu, Quan Wang, Shuchu Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.00751)  

**Abstract**: Recent advances in Large Language Models (LLMs) highlight the need to align their behaviors with human values. A critical, yet understudied, issue is the potential divergence between an LLM's stated preferences (its reported alignment with general principles) and its revealed preferences (inferred from decisions in contextualized scenarios). Such deviations raise fundamental concerns for the interpretability, trustworthiness, reasoning transparency, and ethical deployment of LLMs, particularly in high-stakes applications. This work formally defines and proposes a method to measure this preference deviation. We investigate how LLMs may activate different guiding principles in specific contexts, leading to choices that diverge from previously stated general principles. Our approach involves crafting a rich dataset of well-designed prompts as a series of forced binary choices and presenting them to LLMs. We compare LLM responses to general principle prompts stated preference with LLM responses to contextualized prompts revealed preference, using metrics like KL divergence to quantify the deviation. We repeat the analysis across different categories of preferences and on four mainstream LLMs and find that a minor change in prompt format can often pivot the preferred choice regardless of the preference categories and LLMs in the test. This prevalent phenomenon highlights the lack of understanding and control of the LLM decision-making competence. Our study will be crucial for integrating LLMs into services, especially those that interact directly with humans, where morality, fairness, and social responsibilities are crucial dimensions. Furthermore, identifying or being aware of such deviation will be critically important as LLMs are increasingly envisioned for autonomous agentic tasks where continuous human evaluation of all LLMs' intermediary decision-making steps is impossible. 

**Abstract (ZH)**: Recent Advances in Large Language Models: Aligning Stated and Revealed Preferences 

---
# DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains 

**Title (ZH)**: DrKGC: 动态子图检索增强的LLM在通用和生物医学领域中的知识图谱完成 

**Authors**: Yongkang Xiao, Sinian Zhang, Yi Dai, Huixue Zhou, Jue Hou, Jie Ding, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00708)  

**Abstract**: Knowledge graph completion (KGC) aims to predict missing triples in knowledge graphs (KGs) by leveraging existing triples and textual information. Recently, generative large language models (LLMs) have been increasingly employed for graph tasks. However, current approaches typically encode graph context in textual form, which fails to fully exploit the potential of LLMs for perceiving and reasoning about graph structures. To address this limitation, we propose DrKGC (Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion). DrKGC employs a flexible lightweight model training strategy to learn structural embeddings and logical rules within the KG. It then leverages a novel bottom-up graph retrieval method to extract a subgraph for each query guided by the learned rules. Finally, a graph convolutional network (GCN) adapter uses the retrieved subgraph to enhance the structural embeddings, which are then integrated into the prompt for effective LLM fine-tuning. Experimental results on two general domain benchmark datasets and two biomedical datasets demonstrate the superior performance of DrKGC. Furthermore, a realistic case study in the biomedical domain highlights its interpretability and practical utility. 

**Abstract (ZH)**: 知识图谱完成（KGC）旨在通过利用现有三元组和文本信息来预测知识图谱（KGs）中的缺失三元组。近年来，生成型大型语言模型（LLMs）逐渐被应用于图任务。然而，当前的方法通常以文本形式编码图上下文，未能充分利用LLMs对图结构感知和推理的潜在能力。为解决这一局限，我们提出了DrKGC（基于动态子图检索增强的LLMs的知识图谱完成）。DrKGC采用灵活的轻量级模型训练策略，学习知识图谱中的结构嵌入和逻辑规则。接着，利用一种新颖的自底向上的图检索方法，根据学到的规则为每个查询提取子图。最后，图卷积网络（GCN）适配器使用检索到的子图增强结构嵌入，将增强后的嵌入整合到提示中，以实现有效的LLMs微调。实验证明，DrKGC在两个通用领域基准数据集和两个生物医学数据集中表现出优越的性能。此外，生物医学领域的实际案例研究进一步突显了其可解释性和实用性。 

---
# OntoRAG: Enhancing Question-Answering through Automated Ontology Derivation from Unstructured Knowledge Bases 

**Title (ZH)**: OntoRAG：通过从非结构化知识库自动生成本体增强问答能力 

**Authors**: Yash Tiwari, Owais Ahmad Lone, Mayukha Pal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00664)  

**Abstract**: Ontologies are pivotal for structuring knowledge bases to enhance question answering (QA) systems powered by Large Language Models (LLMs). However, traditional ontology creation relies on manual efforts by domain experts, a process that is time intensive, error prone, and impractical for large, dynamic knowledge domains. This paper introduces OntoRAG, an automated pipeline designed to derive ontologies from unstructured knowledge bases, with a focus on electrical relay documents. OntoRAG integrates advanced techniques, including web scraping, PDF parsing, hybrid chunking, information extraction, knowledge graph construction, and ontology creation, to transform unstructured data into a queryable ontology. By leveraging LLMs and graph based methods, OntoRAG enhances global sensemaking capabilities, outperforming conventional Retrieval Augmented Generation (RAG) and GraphRAG approaches in comprehensiveness and diversity. Experimental results demonstrate OntoRAGs effectiveness, achieving a comprehensiveness win rate of 85% against vector RAG and 75% against GraphRAGs best configuration. This work addresses the critical challenge of automating ontology creation, advancing the vision of the semantic web. 

**Abstract (ZH)**: 本体对于通过大型语言模型（LLMs）驱动的问题回答系统（QA）结构化知识库至关重要。然而，传统本体创建依赖于领域专家的手动努力，这一过程耗时、易出错且不适用于大型动态知识领域。本文介绍了OntoRAG，一个自动管道，旨在从无结构知识库中推导出本体，重点关注继电器文档。OntoRAG 结合了包括网页抓取、PDF 解析、混合切块、信息抽取、知识图谱构建和本体创建在内的先进技术，将无结构数据转换为可查询的本体。通过利用大型语言模型和图基方法，OntoRAG 提高了全局意义建构能力， comprehensive 和多样性方面优于传统的检索增强生成（RAG）和 GraphRAG 方法。实验结果表明 OntoRAG 的有效性，其在全面性方面相对于向量 RAG 的胜出率为 85%，相对于 GraphRAG 最优配置为 75%。这项工作解决了本体自动化创建的关键挑战，推动了语义网络愿景的实现。 

---
# AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents 

**Title (ZH)**: AgentAuditor: LLM代理的人类水平安全与安全评估 

**Authors**: Hanjun Luo, Shenyu Dai, Chiming Ni, Xinfeng Li, Guibin Zhang, Kun Wang, Tongliang Liu, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2506.00641)  

**Abstract**: Despite the rapid advancement of LLM-based agents, the reliable evaluation of their safety and security remains a significant challenge. Existing rule-based or LLM-based evaluators often miss dangers in agents' step-by-step actions, overlook subtle meanings, fail to see how small issues compound, and get confused by unclear safety or security rules. To overcome this evaluation crisis, we introduce \sys, a universal, training-free, memory-augmented reasoning framework that empowers LLM evaluators to emulate human expert evaluators. \sys constructs an experiential memory by having an LLM adaptively extract structured semantic features (e.g., scenario, risk, behavior) and generate associated chain-of-thought reasoning traces for past interactions. A multi-stage, context-aware retrieval-augmented generation process then dynamically retrieves the most relevant reasoning experiences to guide the LLM evaluator's assessment of new cases. Moreover, we developed \data, the first benchmark designed to check how well LLM-based evaluators can spot both safety risks and security threats. \data comprises \textbf{2293} meticulously annotated interaction records, covering \textbf{15} risk types across \textbf{29} application scenarios. A key feature of \data is its nuanced approach to ambiguous risk situations, employing ``Strict'' and ``Lenient'' judgment standards. Experiments demonstrate that \sys not only consistently improves the evaluation performance of LLMs across all benchmarks but also sets a new state-of-the-art in LLM-as-a-judge for agent safety and security, achieving human-level accuracy. Our work is openly openly accessible. 

**Abstract (ZH)**: 尽管基于大规模语言模型的代理取得了快速进步，但对其安全性和可靠性的评估仍然是一个重大挑战。现有的基于规则或基于大规模语言模型的评估器往往会遗漏代理逐步行动中的危险，忽略微妙的意义，无法看到小问题如何积累，并且容易混淆不清的安全或安全规则。为克服这一评估危机，我们引入了\sys，这是一种无需训练的、具有增强记忆的推理框架，能够使大规模语言模型评估器模仿人类专家评估器。\sys 通过让大规模语言模型自适应地提取结构化语义特征（例如场景、风险、行为），并为过去的交互生成相关的推理过程痕迹，构建一个经验记忆。随后，一个多阶段、上下文感知的检索增强生成过程动态检索最相关的推理经验，以指导大规模语言模型评估器对新案例的评估。此外，我们开发了\data，这是首个用于检查基于大规模语言模型的评估器能否识别安全风险和安全威胁的基准。\data 包含精心标注的 \textbf{2293} 个交互记录，涵盖了 \textbf{29} 种应用场景中的 \textbf{15} 种风险类型。\data 的一个关键特点是，它对模糊的风险情况采用了严格的“严格”和宽松的“宽容”判断标准。实验表明，\sys 不仅在所有基准上持续地提高了大规模语言模型的评估性能，还在代理安全性和安全性方面的“大规模语言模型作为裁判”这一领域设立了新的最先进水平，达到了人类级别的准确度。我们的工作是开源的。 

---
# Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs 

**Title (ZH)**: 语言模型反映人类的信心吗？探索心理 Insights 解决大语言模型的过自信问题 

**Authors**: Chenjun Xu, Bingbing Wen, Bin Han, Robert Wolfe, Lucy Lu Wang, Bill Howe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00582)  

**Abstract**: Psychology research has shown that humans are poor at estimating their performance on tasks, tending towards underconfidence on easy tasks and overconfidence on difficult tasks. We examine three LLMs, Llama-3-70B-instruct, Claude-3-Sonnet, and GPT-4o, on a range of QA tasks of varying difficulty, and show that models exhibit subtle differences from human patterns of overconfidence: less sensitive to task difficulty, and when prompted to answer based on different personas -- e.g., expert vs layman, or different race, gender, and ages -- the models will respond with stereotypically biased confidence estimations even though their underlying answer accuracy remains the same. Based on these observations, we propose Answer-Free Confidence Estimation (AFCE) to improve confidence calibration and LLM interpretability in these settings. AFCE is a self-assessment method that employs two stages of prompting, first eliciting only confidence scores on questions, then asking separately for the answer. Experiments on the MMLU and GPQA datasets spanning subjects and difficulty show that this separation of tasks significantly reduces overconfidence and delivers more human-like sensitivity to task difficulty. 

**Abstract (ZH)**: 心理学研究显示，人类在估计任务表现时能力有限，倾向于在简单任务中低估自己，在困难任务中过度自信。我们考察了三种大型语言模型——Llama-3-70B-instruct、Claude-3-Sonnet 和 GPT-4o，在不同难度的问答任务上的表现，并发现模型显示出与人类过度自信模式的微妙差异：不那么敏感于任务难度，并且在按照不同人物特征——例如专家与普通人的观点，或者不同种族、性别和年龄——被提示作答时，模型会以模式化的偏见方式进行自信估计，尽管其基本答案的准确性保持不变。基于这些观察，我们提出了答案自由的自信估计（AFCE）方法以改善这些情境下的自信校准和大型语言模型的可解释性。AFCE 是一种自我评估方法，采用两阶段提示：首先仅获取对问题的信心评分，然后分别要求提供答案。在 MMLU 和 GPQA 数据集上的实验表明，这种任务分离显著减少了过度自信，并提高了对任务难度的人类敏感度。 

---
# Reasoning Like an Economist: Post-Training on Economic Problems Induces Strategic Generalization in LLMs 

**Title (ZH)**: 经济学家般推理：在经济问题上的后训练诱导LLMs进行战略泛化 

**Authors**: Yufa Zhou, Shaobo Wang, Xingyu Dong, Xiangqi Jin, Yifang Chen, Yue Min, Kexin Yang, Xingzhang Ren, Dayiheng Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00577)  

**Abstract**: Directly training Large Language Models (LLMs) for Multi-Agent Systems (MAS) remains challenging due to intricate reward modeling, dynamic agent interactions, and demanding generalization requirements. This paper explores whether post-training techniques, specifically Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR), can effectively $\textit{generalize}$ to multi-agent scenarios. We use economic reasoning as a testbed, leveraging its strong foundations in mathematics and game theory, its demand for structured analytical reasoning, and its relevance to real-world applications such as market design, resource allocation, and policy analysis. We introduce $\textbf{Recon}$ ($\textbf{R}$easoning like an $\textbf{ECON}$omist), a 7B-parameter open-source LLM post-trained on a hand-curated dataset of 2,100 high-quality economic reasoning problems. Comprehensive evaluation on economic reasoning benchmarks and multi-agent games reveals clear improvements in structured reasoning and economic rationality. These results underscore the promise of domain-aligned post-training for enhancing reasoning and agent alignment, shedding light on the roles of SFT and RL in shaping model behavior. Code is available at this https URL . 

**Abstract (ZH)**: 直接训练大规模语言模型（LLMs）用于多 agent 系统（MAS）仍然具有挑战性，原因在于复杂的奖励建模、动态的 agent 交互以及严苛的泛化要求。本文探讨了后训练技术，特别是监督微调（SFT）和具有可验证奖励的强化学习（RLVR），是否能够有效地泛化到多 agent 场景中。我们以经济学为例，利用其在数学和博弈论中的坚实基础、对结构化分析推理的需求以及在市场设计、资源分配和政策分析等实际应用中的相关性。我们引入了名为Recon（Reasoning like an ECONomist）的7B参数开源后训练LLM，基于2100个高质量的经济推理问题的手工挑选数据集进行训练。在经济推理基准测试和多 agent 游戏中的全面评估显示了在结构化推理和经济理性方面的明显改进。这些结果凸显了领域对齐后训练在增强推理和agent对齐方面的潜力，并揭示了SFT和RL在塑造模型行为中的作用。代码可在以下链接获取。 

---
# CityLens: Benchmarking Large Language-Vision Models for Urban Socioeconomic Sensing 

**Title (ZH)**: CityLens: 评估大型语言视觉模型的城市社会经济感知能力 

**Authors**: Tianhui Liu, Jie Feng, Hetian Pang, Xin Zhang, Tianjian Ouyang, Zhiyuan Zhang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00530)  

**Abstract**: Understanding urban socioeconomic conditions through visual data is a challenging yet essential task for sustainable urban development and policy planning. In this work, we introduce $\textbf{CityLens}$, a comprehensive benchmark designed to evaluate the capabilities of large language-vision models (LLVMs) in predicting socioeconomic indicators from satellite and street view imagery. We construct a multi-modal dataset covering a total of 17 globally distributed cities, spanning 6 key domains: economy, education, crime, transport, health, and environment, reflecting the multifaceted nature of urban life. Based on this dataset, we define 11 prediction tasks and utilize three evaluation paradigms: Direct Metric Prediction, Normalized Metric Estimation, and Feature-Based Regression. We benchmark 17 state-of-the-art LLVMs across these tasks. Our results reveal that while LLVMs demonstrate promising perceptual and reasoning capabilities, they still exhibit limitations in predicting urban socioeconomic indicators. CityLens provides a unified framework for diagnosing these limitations and guiding future efforts in using LLVMs to understand and predict urban socioeconomic patterns. Our codes and datasets are open-sourced via this https URL. 

**Abstract (ZH)**: 通过视觉数据理解城市社会经济条件是可持续城市发展和政策规划中一项艰巨但必要的任务。本文介绍了$\textbf{CityLens}$，一个全面的基准，旨在评估大型语言-视觉模型（LLVMs）从卫星和街道视图图像预测社会经济指标的能力。我们构建了一个多模态数据集，覆盖了全球17个城市，涵盖经济、教育、犯罪、交通、健康和环境等6个关键领域，反映了城市生活的复杂性。基于该数据集，我们定义了11项预测任务，并使用了三种评估范式：直接度量预测、标准化度量估计和基于特征的回归。我们在这些任务上对标了17个最先进的LLVMs。我们的结果显示，虽然LLVMs展示了具有前景的感知和推理能力，但在预测城市社会经济指标方面仍存在局限性。CityLens提供了一个统一的框架，用于诊断这些局限性，并指导未来使用LLVMs理解和预测城市社会经济模式的努力。我们的代码和数据集通过这个链接公开发布。 

---
# MIRROR: Cognitive Inner Monologue Between Conversational Turns for Persistent Reflection and Reasoning in Conversational LLMs 

**Title (ZH)**: MIRROR: 聊天型大语言模型中对话轮次间的心智内省对话以实现持续反思与推理 

**Authors**: Nicole Hsing  

**Link**: [PDF](https://arxiv.org/pdf/2506.00430)  

**Abstract**: Human intelligence relies on inner monologue to process complex information through simultaneous reflection, memory retrieval, and response formulation. We introduce MIRROR (Modular Internal Reasoning, Reflection, Orchestration, and Response), a cognitive architecture that systematically implements these parallel reasoning capabilities in large language models. MIRROR operates as a unified system with two distinct functional layers: the Thinker and the Talker. The Thinker encompasses: (1) the Inner Monologue Manager, coordinating reasoning threads across cognitive dimensions (Goals, Reasoning, and Memory); and (2) the Cognitive Controller, synthesizing these threads into a coherent internal narrative maintained across conversation turns. The Talker component then leverages this integrated narrative for context-aware responses. Evaluated on the CuRaTe benchmark--testing personalized dialogue with safety-critical constraints, conflicting preferences, and multi-turn consistency--LLMs utilizing the MIRROR architecture achieve up to 156% relative improvement in critical safety scenarios involving three persons with conflicting preferences, maintaining an average accuracy of ~>80% on all scenarios. Across scenario-specific comparisons, GPT-4o, Gemini 1.5 Pro, Claude 3.7 Sonnet, Llama 4 variants, and Mistral 3 variants with the MIRROR architecture outperformed baseline models by 21% on average (15.5 percentage points absolute). MIRROR directly addresses three critical LLM failure modes: sycophancy, attentional deficits to critical information, and inconsistent prioritization of conflicting constraints. This work bridges cognitive science and AI by implementing modular internal reasoning inspired by human cognition, creating a persistent internal model that significantly enhances multi-turn conversation capabilities. 

**Abstract (ZH)**: 人类智能依赖内部独白处理复杂信息，通过同时反思、记忆检索和响应形成。我们提出了MIRROR（模块化内部推理、反思、协调和响应）认知架构，系统性地在大型语言模型中实现这些并行推理能力。MIRROR作为统一系统运作，具有两个不同的功能层：思考者和说话者。思考者包含：（1）内部独白管理者，协调认知维度（目标、推理和记忆）间的推理线程；（2）认知控制器，将这些线程综合成连贯的内部叙述，并在会话轮次中保持连贯。说话者组件随后利用这一整合叙述进行具有上下文意识的响应。该架构在CuRaTe基准测试中进行了评估，测试包含安全关键约束、冲突偏好和多轮一致性的人性化对话，使用MIRROR架构的LLMs在涉及三人冲突偏好的关键安全场景中实现了多达156%的相对改进，平均准确率约为>80%。在特定场景比较中，配备MIRROR架构的GPT-4o、Gemini 1.5 Pro、Claude 3.7 Sonnet、Llama 4变体和Mistral 3变体模型在平均上比基准模型高出21%（绝对值15.5个百分点）。MIRROR直接解决了三种关键的LLM失效模式：拍马屁、对关键信息的注意力缺陷以及对冲突约束的一致优先级化问题。这项工作通过实现受人类认知启发的模块化内部推理，连接了认知科学和人工智能，创建了一个持久的内部模型，显著提升了多轮对话能力。 

---
# Evaluation of LLMs for mathematical problem solving 

**Title (ZH)**: LLMs在数学问题求解中的评估 

**Authors**: Ruonan Wang, Runxi Wang, Yunwen Shen, Chengfeng Wu, Qinglin Zhou, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00309)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance on a range of educational tasks, but are still understudied for their potential to solve mathematical problems. In this study, we compare three prominent LLMs, including GPT-4o, DeepSeek-V3, and Gemini-2.0, on three mathematics datasets of varying complexities (GSM8K, MATH500, and UNSW datasets). We take a five-dimensional approach based on the Structured Chain-of-Thought (SCoT) framework to assess final answer correctness, step completeness, step validity, intermediate calculation accuracy, and problem comprehension. The results show that GPT-4o is the most stable and consistent in performance across all the datasets, but particularly it performs outstandingly in high-level questions of the UNSW dataset. DeepSeek-V3 is competitively strong in well-structured domains such as optimisation, but suffers from fluctuations in accuracy in statistical inference tasks. Gemini-2.0 shows strong linguistic understanding and clarity in well-structured problems but performs poorly in multi-step reasoning and symbolic logic. Our error analysis reveals particular deficits in each model: GPT-4o is at times lacking in sufficient explanation or precision; DeepSeek-V3 leaves out intermediate steps; and Gemini-2.0 is less flexible in mathematical reasoning in higher dimensions. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在教育任务上表现出色，但在解决数学问题方面仍待深入研究。本研究比较了GPT-4o、DeepSeek-V3和Gemini-2.0三种 prominent LLMs 在三个不同复杂度的数学数据集（GSM8K、MATH500 和 UNSW 数据集）上的性能。我们基于结构化的链式思维（SCoT）框架从五个维度评估最终答案的正确性、步骤完整性、步骤有效性、中间计算精度以及问题理解。结果显示，GPT-4o 在所有数据集上的表现最为稳定和一致，在 UNSW 数据集的高阶问题上表现尤为出色。DeepSeek-V3 在优化等结构化良好的领域表现出色，但在统计推断任务中准确性波动较大。Gemini-2.0 在结构良好问题上的语言理解能力和清晰度较强，但在多步推理和符号逻辑方面表现不佳。我们错误分析揭示了每个模型的具体缺陷：GPT-4o 有时缺乏充分的解释或精确度；DeepSeek-V3 忽略了中间步骤；而 Gemini-2.0 在高维度的数学推理上不够灵活。 

---
# Hidden in Plain Sight: Probing Implicit Reasoning in Multimodal Language Models 

**Title (ZH)**: 处处可见但隐藏的推理：探究多模态语言模型中的隐式推理 

**Authors**: Qianqi Yan, Hongquan Li, Shan Jiang, Yang Zhao, Xinze Guan, Ching-Chen Kuo, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00258)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly deployed in open-ended, real-world environments where inputs are messy, underspecified, and not always trustworthy. Unlike curated benchmarks, these settings frequently involve instructions that refer to missing objects or contradictory facts, rely on ambiguous references, or request infeasible actions. In such cases, success hinges not on task execution alone, but on a model's ability to detect when something is silently wrong. This paper presents a systematic analysis of how current MLLMs handle such implicit reasoning scenarios: cases where the flaw is not explicitly stated but must be inferred from context. Using a curated diagnostic suite spanning four categories of real-world failure modes, we evaluate six MLLMs, including o3 and GPT-4o, and find that models frequently fail to surface hidden issues, even when they possess the necessary perceptual and reasoning skills. Explicit prompting reveals that the underlying capabilities exist but are often suppressed in favor of user compliance. We further show that simple inference-time interventions, such as cautious persona prompting and, in particular, requiring a clarifying question, can dramatically recover performance. Our findings highlight a persistent gap between reasoning competence and behavioral compliance in current MLLMs and suggest practical strategies for making these models more trustworthy in underconstrained environments. 

**Abstract (ZH)**: 多模态大型语言模型在开放环境中处理隐含推理场景的研究 

---
# MIR: Methodology Inspiration Retrieval for Scientific Research Problems 

**Title (ZH)**: MIR: 科学研究问题的方法论灵感检索 

**Authors**: Aniketh Garikaparthi, Manasi Patwardhan, Aditya Sanjiv Kanade, Aman Hassan, Lovekesh Vig, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00249)  

**Abstract**: There has been a surge of interest in harnessing the reasoning capabilities of Large Language Models (LLMs) to accelerate scientific discovery. While existing approaches rely on grounding the discovery process within the relevant literature, effectiveness varies significantly with the quality and nature of the retrieved literature. We address the challenge of retrieving prior work whose concepts can inspire solutions for a given research problem, a task we define as Methodology Inspiration Retrieval (MIR). We construct a novel dataset tailored for training and evaluating retrievers on MIR, and establish baselines. To address MIR, we build the Methodology Adjacency Graph (MAG); capturing methodological lineage through citation relationships. We leverage MAG to embed an "intuitive prior" into dense retrievers for identifying patterns of methodological inspiration beyond superficial semantic similarity. This achieves significant gains of +5.4 in Recall@3 and +7.8 in Mean Average Precision (mAP) over strong baselines. Further, we adapt LLM-based re-ranking strategies to MIR, yielding additional improvements of +4.5 in Recall@3 and +4.8 in mAP. Through extensive ablation studies and qualitative analyses, we exhibit the promise of MIR in enhancing automated scientific discovery and outline avenues for advancing inspiration-driven retrieval. 

**Abstract (ZH)**: 利用大型语言模型的推理能力加速科学发现的方法学灵感检索 

---
# Whispers of Many Shores: Cultural Alignment through Collaborative Cultural Expertise 

**Title (ZH)**: 多岸低语：协作文化专长的文化对接 

**Authors**: Shuai Feng, Wei-Chuang Chan, Srishti Chouhan, Junior Francisco Garcia Ayala, Srujananjali Medicherla, Kyle Clark, Mingwei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00242)  

**Abstract**: The integration of large language models (LLMs) into global applications necessitates effective cultural alignment for meaningful and culturally-sensitive interactions. Current LLMs often lack the nuanced understanding required for diverse cultural contexts, and adapting them typically involves costly full fine-tuning. To address this, we introduce a novel soft prompt fine-tuning framework that enables efficient and modular cultural alignment. Our method utilizes vectorized prompt tuning to dynamically route queries to a committee of culturally specialized 'expert' LLM configurations, created by optimizing soft prompt embeddings without altering the base model's parameters. Extensive experiments demonstrate that our framework significantly enhances cultural sensitivity and adaptability, improving alignment scores from 0.208 to 0.820, offering a robust solution for culturally-aware LLM deployment. This research paves the way for subsequent investigations into enhanced cultural coverage and dynamic expert adaptation, crucial for realizing autonomous AI with deeply nuanced understanding in a globally interconnected world. 

**Abstract (ZH)**: 大型语言模型（LLMs）在全球应用中的整合需要有效的文化对齐以实现有意义和文化敏感的交互。当前的LLMs往往缺乏对多元文化背景的细微理解，而对它们进行适应通常需要昂贵的全面微调。为解决这一问题，我们提出了一种新颖的软提示微调框架，以实现高效和模块化文化对齐。该方法利用向量化的提示微调动态路由查询给由优化软提示嵌入而来的文化专业化“专家”LLM配置委员会，而不更改基模型的参数。广泛的实验表明，我们的框架显著增强了文化敏感性和适应性，使对齐分数从0.208提升到0.820，提供了一种适用于文化意识LLM部署的稳健解决方案。这项研究为后续对增强文化覆盖率和动态专家适应性的研究铺平了道路，这对于实现一个在全球互联世界中具有深刻细微理解的自主AI至关重要。 

---
# Tournament of Prompts: Evolving LLM Instructions Through Structured Debates and Elo Ratings 

**Title (ZH)**: 指令锦标赛：通过结构化辩论和Elo评分演化LLM指令 

**Authors**: Anirudh Nair, Adi Banerjee, Laurent Mombaerts, Matthew Hagen, Tarik Borogovac  

**Link**: [PDF](https://arxiv.org/pdf/2506.00178)  

**Abstract**: Prompt engineering represents a critical bottleneck to harness the full potential of Large Language Models (LLMs) for solving complex tasks, as it requires specialized expertise, significant trial-and-error, and manual intervention. This challenge is particularly pronounced for tasks involving subjective quality assessment, where defining explicit optimization objectives becomes fundamentally problematic. Existing automated prompt optimization methods falter in these scenarios, as they typically require well-defined task-specific numerical fitness functions or rely on generic templates that cannot capture the nuanced requirements of complex use cases. We introduce DEEVO (DEbate-driven EVOlutionary prompt optimization), a novel framework that guides prompt evolution through a debate-driven evaluation with an Elo-based selection. Contrary to prior work, DEEVOs approach enables exploration of the discrete prompt space while preserving semantic coherence through intelligent crossover and strategic mutation operations that incorporate debate-based feedback, combining elements from both successful and unsuccessful prompts based on identified strengths rather than arbitrary splicing. Using Elo ratings as a fitness proxy, DEEVO simultaneously drives improvement and preserves valuable diversity in the prompt population. Experimental results demonstrate that DEEVO significantly outperforms both manual prompt engineering and alternative state-of-the-art optimization approaches on open-ended tasks and close-ended tasks despite using no ground truth feedback. By connecting LLMs reasoning capabilities with adaptive optimization, DEEVO represents a significant advancement in prompt optimization research by eliminating the need of predetermined metrics to continuously improve AI systems. 

**Abstract (ZH)**: DEEVO：基于辩论驱动的进化式提示优化 

---
# The Automated but Risky Game: Modeling Agent-to-Agent Negotiations and Transactions in Consumer Markets 

**Title (ZH)**: 自动但充满风险的游戏：消费者市场中代理方到代理方的谈判与交易建模 

**Authors**: Shenzhe Zhu, Jiao Sun, Yi Nian, Tobin South, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00073)  

**Abstract**: AI agents are increasingly used in consumer-facing applications to assist with tasks such as product search, negotiation, and transaction execution. In this paper, we explore a future scenario where both consumers and merchants authorize AI agents to fully automate negotiations and transactions. We aim to answer two key questions: (1) Do different LLM agents vary in their ability to secure favorable deals for users? (2) What risks arise from fully automating deal-making with AI agents in consumer markets? To address these questions, we develop an experimental framework that evaluates the performance of various LLM agents in real-world negotiation and transaction settings. Our findings reveal that AI-mediated deal-making is an inherently imbalanced game -- different agents achieve significantly different outcomes for their users. Moreover, behavioral anomalies in LLMs can result in financial losses for both consumers and merchants, such as overspending or accepting unreasonable deals. These results underscore that while automation can improve efficiency, it also introduces substantial risks. Users should exercise caution when delegating business decisions to AI agents. 

**Abstract (ZH)**: AI代理在消费者市场中自动化谈判与交易的风险与机遇：基于不同大规模语言模型的实证研究 

---
# DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation 

**Title (ZH)**: DRAG: 从大规模语言模型提炼RAG到序列建模语言模型的知识蒸馏与证据图辅助 hallucination 缓解 

**Authors**: Jennifer Chen, Aidar Myrzakhan, Yaxin Luo, Hassaan Muhammad Khan, Sondos Mahmoud Bsharat, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01954)  

**Abstract**: Retrieval-Augmented Generation (RAG) methods have proven highly effective for tasks requiring factual consistency and robust knowledge retrieval. However, large-scale RAG systems consume significant computational resources and are prone to generating hallucinated content from Humans. In this work, we introduce $\texttt{DRAG}$, a novel framework for distilling RAG knowledge from large-scale Language Models (LLMs) into small LMs (SLMs). Our approach leverages evidence- and knowledge graph-based distillation, ensuring that the distilled model retains critical factual knowledge while significantly reducing model size and computational cost. By aligning the smaller model's predictions with a structured knowledge graph and ranked evidence, $\texttt{DRAG}$ effectively mitigates hallucinations and improves factual accuracy. We further present a case demonstrating how our framework mitigates user privacy risks and introduce a corresponding benchmark. Experimental evaluations on multiple benchmarks demonstrate that our method outperforms the prior competitive RAG methods like MiniRAG for SLMs by up to 27.7% using the same models, preserving high-level efficiency and reliability. With $\texttt{DRAG}$, we provide a practical and resource-efficient roadmap to deploying enhanced retrieval and generation capabilities in small-sized LLMs. 

**Abstract (ZH)**: DRAG：一种将大规模语言模型中的RAG知识精炼至小型语言模型的新型框架 

---
# WebChoreArena: Evaluating Web Browsing Agents on Realistic Tedious Web Tasks 

**Title (ZH)**: WebChoreArena: 评估网页浏览代理在现实复杂任务中的性能 

**Authors**: Atsuyuki Miyai, Zaiying Zhao, Kazuki Egashira, Atsuki Sato, Tatsumi Sunada, Shota Onohara, Hiromasa Yamanishi, Mashiro Toyooka, Kunato Nishina, Ryoma Maeda, Kiyoharu Aizawa, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.01952)  

**Abstract**: Powered by a large language model (LLM), a web browsing agent operates web browsers in a human-like manner and offers a highly transparent path toward automating a wide range of everyday tasks. As web agents become increasingly capable and demonstrate proficiency in general browsing tasks, a critical question emerges: Can they go beyond general browsing to robustly handle tasks that are tedious and complex, or chores that humans often avoid doing themselves? In this paper, we introduce WebChoreArena, a new fully reproducible benchmark comprising 532 carefully curated tasks designed to extend the scope of WebArena beyond general browsing to more labor-intensive and tedious tasks. WebChoreArena systematically integrates three key challenges: (i) Massive Memory tasks requiring accurate retrieval of large amounts of information in the observations, (ii) Calculation tasks demanding precise mathematical reasoning, and (iii) Long-Term Memory tasks necessitating long-term memory across multiple webpages. Built on top of the fully reproducible and widely adopted four WebArena simulation environments, WebChoreArena ensures strict reproducibility and enables fair, direct comparisons with the established WebArena benchmark, offering key insights into agent progress. Our experimental results demonstrate that as LLMs evolve, represented by GPT-4o, Claude 3.7 Sonnet, and Gemini 2.5 Pro, significant improvements in performance are observed on WebChoreArena. These findings suggest that WebChoreArena is well-suited to measure the advancement of state-of-the-art LLMs with greater clarity. Nevertheless, the results also indicate that even with Gemini 2.5 Pro, there remains substantial room for improvement compared to WebArena, highlighting the increased challenges posed by WebChoreArena. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的网络浏览代理以类人的方式操作浏览器，并提供一条高度透明的途径以自动化广泛日常生活任务。随着网络代理的能力日益增强并在通用浏览任务上表现出色，一个关键问题 emerges: 它们能否超越通用浏览，稳健地处理那些繁琐且复杂的任务，或是人类通常避免亲自完成的杂务？在本文中，我们介绍了WebChoreArena，一个全新的完全可再现基准，包含532个精心挑选的任务，旨在将WebArena的范围扩展至更多的劳动密集型和繁琐任务。WebChoreArena系统地整合了三个关键挑战：（i）巨量记忆任务，要求准确检索大量信息；（ii）计算任务，需要精确的数学推理；（iii）长时记忆任务，需要跨多网页的长时记忆。WebChoreArena基于完全可再现且广泛采用的四个WebArena模拟环境构建，确保严格的可再现性，并能够公平直接地与现有的WebArena基准进行比较，提供关键的见解以衡量代理的进步。实验结果表明，随着LLM的演进，由GPT-4o、Claude 3.7 Sonnet和Gemini 2.5 Pro代表，WebChoreArena上的性能显著提高。这些发现表明，WebChoreArena适合更清晰地衡量最新的LLM的发展。然而，结果也表明，即使在Gemini 2.5 Pro的情况下，与WebArena相比仍存在显著改进空间，突显了WebChoreArena带来的增加挑战。 

---
# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning 

**Title (ZH)**: 超出80/20规则：高熵少数标记驱动的有效强化学习用于LLM推理 

**Authors**: Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01939)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）通过token熵模式提升大型语言模型的推理能力及其机制探究 

---
# Image Generation from Contextually-Contradictory Prompts 

**Title (ZH)**: 从上下文矛盾提示生成图像 

**Authors**: Saar Huberman, Or Patashnik, Omer Dahary, Ron Mokady, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2506.01929)  

**Abstract**: Text-to-image diffusion models excel at generating high-quality, diverse images from natural language prompts. However, they often fail to produce semantically accurate results when the prompt contains concept combinations that contradict their learned priors. We define this failure mode as contextual contradiction, where one concept implicitly negates another due to entangled associations learned during training. To address this, we propose a stage-aware prompt decomposition framework that guides the denoising process using a sequence of proxy prompts. Each proxy prompt is constructed to match the semantic content expected to emerge at a specific stage of denoising, while ensuring contextual coherence. To construct these proxy prompts, we leverage a large language model (LLM) to analyze the target prompt, identify contradictions, and generate alternative expressions that preserve the original intent while resolving contextual conflicts. By aligning prompt information with the denoising progression, our method enables fine-grained semantic control and accurate image generation in the presence of contextual contradictions. Experiments across a variety of challenging prompts show substantial improvements in alignment to the textual prompt. 

**Abstract (ZH)**: 基于文本到图像的扩散模型在从自然语言提示生成高质量、多样化图像方面表现出色。然而，当提示包含与模型学习先验矛盾的概念组合时，它们往往无法产生语义准确的结果。我们定义这种失败模式为上下文矛盾，其中一个概念由于训练期间学习到的纠缠关联而隐含地否定了另一个概念。为了解决这一问题，我们提出了一种阶段感知的提示分解框架，该框架使用一系列代理提示引导去噪过程。每个代理提示都构建为与特定去噪阶段预期生成的语义内容匹配，并确保上下文连贯性。为构建这些代理提示，我们利用大型语言模型（LLM）分析目标提示，识别矛盾，并生成能够保留原始意图同时解决上下文冲突的替代表达。通过将提示信息与去噪进程对齐，我们的方法在存在上下文矛盾的情况下能够实现细粒度的语义控制和准确的图像生成。在多种具有挑战性的提示下进行的实验表明，我们的方法在与文本提示对齐方面取得了显著改进。 

---
# MoDA: Modulation Adapter for Fine-Grained Visual Grounding in Instructional MLLMs 

**Title (ZH)**: MoDA: 调制适配器在指令式MLLLMs中细粒度视觉定位中的应用 

**Authors**: Wayner Barrios, Andrés Villa, Juan León Alcázar, SouYoung Jin, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2506.01850)  

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated impressive performance on instruction-following tasks by integrating pretrained visual encoders with large language models (LLMs). However, existing approaches often struggle to ground fine-grained visual concepts in complex scenes. In this paper, we propose MoDA (Modulation Adapter), a lightweight yet effective module designed to refine pre-aligned visual features through instruction-guided modulation. Our approach follows the standard LLaVA training protocol, consisting of a two-stage process: (1) aligning image features to the LLMs input space via a frozen vision encoder and adapter layers, and (2) refining those features using the MoDA adapter during the instructional tuning stage. MoDA employs a Transformer-based cross-attention mechanism to generate a modulation mask over the aligned visual tokens, thereby emphasizing semantically relevant embedding dimensions based on the language instruction. The modulated features are then passed to the LLM for autoregressive language generation. Our experimental evaluation shows that MoDA improves visual grounding and generates more contextually appropriate responses, demonstrating its effectiveness as a general-purpose enhancement for image-based MLLMs. 

**Abstract (ZH)**: Recently, 多模态大型语言模型（MLLMs）通过将预训练的视觉编码器与大型语言模型（LLMs）结合，在指令跟随任务中展现了 impressive 的性能。然而，现有方法往往难以在复杂场景中细粒度地ground视觉概念。本文中，我们提出 MoDA（调制适配器），这是一种轻量级但有效的模块，设计用于通过指令引导的调制细化预先对齐的视觉特征。我们的方法遵循标准的 LLaVA 训练协议，包括两阶段过程：（1）通过冻结的视觉编码器和适配器层将图像特征对齐到 LLMs 的输入空间，（2）在指令调整阶段使用 MoDA 适配器细化这些特征。MoDA 使用基于Transformer的跨注意力机制生成调制掩码，从而根据语言指令强调语义相关的嵌入维度。调制后的特征随后传递给 LLM 进行自回归语言生成。我们的实验评估表明，MoDA 改进了视觉grounding，产生了更上下文相关的目标响应，证明了其作为图像基础 MLLMs 通用增强手段的有效性。 

---
# iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering 

**Title (ZH)**: iQUEST：一种迭代问题导向的知识库问答框架 

**Authors**: Shuai Wang, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01784)  

**Abstract**: While Large Language Models (LLMs) excel at many natural language processing tasks, they often suffer from factual inaccuracies in knowledge-intensive scenarios. Integrating external knowledge resources, particularly knowledge graphs (KGs), provides a transparent and updatable foundation for more reliable reasoning. Knowledge Base Question Answering (KBQA), which queries and reasons over KGs, is central to this effort, especially for complex, multi-hop queries. However, multi-hop reasoning poses two key challenges: (1)~maintaining coherent reasoning paths, and (2)~avoiding prematurely discarding critical multi-hop connections. To address these issues, we introduce iQUEST, a question-guided KBQA framework that iteratively decomposes complex queries into simpler sub-questions, ensuring a structured and focused reasoning trajectory. Additionally, we integrate a Graph Neural Network (GNN) to look ahead and incorporate 2-hop neighbor information at each reasoning step. This dual approach strengthens the reasoning process, enabling the model to explore viable paths more effectively. Detailed experiments demonstrate the consistent improvement delivered by iQUEST across four benchmark datasets and four LLMs. 

**Abstract (ZH)**: 大型语言模型在许多自然语言处理任务中表现出色，但在知识密集型场景中往往会遇到事实准确性问题。通过集成外部知识资源，特别是知识图谱（KGs），可以为更加可靠的知识推理提供透明且可更新的基础。知识库问答（KBQA），即查询和在知识图谱上推理，是解决这一问题的关键，尤其是在处理复杂、多跳查询方面。然而，多跳推理带来两个关键挑战：（1）保持连贯的推理路径，（2）避免过早丢弃关键的多跳连接。为应对这些挑战，我们引入了iQUEST，这是一种基于问题的KBQA框架，通过迭代分解复杂查询为更简单的子问题，确保结构化和重点明确的推理轨迹。此外，我们集成了一个图神经网络（GNN）来提前查看并在每一个推理步骤中整合两个跳邻接信息。这种双重方法增强了推理过程，使模型能够更有效地探索可行路径。详细实验表明，iQUEST在四个基准数据集和四个大型语言模型上一致地提供了改进。 

---
# MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation 

**Title (ZH)**: MaXIFE：多语言和跨语言指令跟随评估 

**Authors**: Yile Liu, Ziwei Ma, Xiu Jiang, Jinglu Hu, Jing Chang, Liang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01776)  

**Abstract**: With the rapid adoption of large language models (LLMs) in natural language processing, the ability to follow instructions has emerged as a key metric for evaluating their practical utility. However, existing evaluation methods often focus on single-language scenarios, overlooking the challenges and differences present in multilingual and cross-lingual contexts. To address this gap, we introduce MaXIFE: a comprehensive evaluation benchmark designed to assess instruction-following capabilities across 23 languages with 1,667 verifiable instruction tasks. MaXIFE integrates both Rule-Based Evaluation and Model-Based Evaluation, ensuring a balance of efficiency and accuracy. We applied MaXIFE to evaluate several leading commercial and open-source LLMs, establishing baseline results for future comparisons. By providing a standardized tool for multilingual instruction-following evaluation, MaXIFE aims to advance research and development in natural language processing. 

**Abstract (ZH)**: 大规模语言模型在自然语言处理中的快速采用使得遵循指令的能力成为了评估其实用价值的关键指标。然而，现有的评估方法往往侧重于单一语言场景，忽视了多语言和跨语言环境中的挑战和差异。为解决这一问题，我们提出了MaXIFE：一个旨在评估23种语言中1,667项可验证指令任务的综合评估基准，以评估指令跟随能力。MaXIFE结合了基于规则的评估和基于模型的评估，确保效率与准确性的平衡。我们应用MaXIFE对几种领先的商业和开源大规模语言模型进行了评估，建立了未来比较的基础结果。通过提供一个多语言指令跟随评估的标准工具，MaXIFE旨在促进自然语言处理领域的研究与开发。 

---
# ReGA: Representation-Guided Abstraction for Model-based Safeguarding of LLMs 

**Title (ZH)**: ReGA: 基于表示的抽象方法用于LLM的模型导向安全保障 

**Authors**: Zeming Wei, Chengcan Wu, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01770)  

**Abstract**: Large Language Models (LLMs) have achieved significant success in various tasks, yet concerns about their safety and security have emerged. In particular, they pose risks in generating harmful content and vulnerability to jailbreaking attacks. To analyze and monitor machine learning models, model-based analysis has demonstrated notable potential in stateful deep neural networks, yet suffers from scalability issues when extending to LLMs due to their vast feature spaces. In this paper, we propose ReGA, a model-based analysis framework with representation-guided abstraction, to safeguard LLMs against harmful prompts and generations. By leveraging safety-critical representations, which are low-dimensional directions emerging in hidden states that indicate safety-related concepts, ReGA effectively addresses the scalability issue when constructing the abstract model for safety modeling. Our comprehensive evaluation shows that ReGA performs sufficiently well in distinguishing between safe and harmful inputs, achieving an AUROC of 0.975 at the prompt level and 0.985 at the conversation level. Additionally, ReGA exhibits robustness to real-world attacks and generalization across different safety perspectives, outperforming existing safeguard paradigms in terms of interpretability and scalability. Overall, ReGA serves as an efficient and scalable solution to enhance LLM safety by integrating representation engineering with model-based abstraction, paving the way for new paradigms to utilize software insights for AI safety. Our code is available at this https URL. 

**Abstract (ZH)**: 基于表示引导抽象的大语言模型安全性分析框架ReGA 

---
# Tug-of-war between idiom's figurative and literal meanings in LLMs 

**Title (ZH)**: LLMs中成语的比喻义与字面义之间的争夺战 

**Authors**: Soyoung Oh, Xinting Huang, Mathis Pink, Michael Hahn, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.01723)  

**Abstract**: Idioms present a unique challenge for language models due to their non-compositional figurative meanings, which often strongly diverge from the idiom's literal interpretation. This duality requires a model to learn representing and deciding between the two meanings to interpret an idiom in a figurative sense, or literally. In this paper, we employ tools from mechanistic interpretability to trace how a large pretrained causal transformer (LLama3.2-1B-base) deals with this ambiguity. We localize three steps of idiom processing: First, the idiom's figurative meaning is retrieved in early attention and MLP sublayers. We identify specific attention heads which boost the figurative meaning of the idiom while suppressing the idiom's literal interpretation. The model subsequently represents the figurative representation through an intermediate path. Meanwhile, a parallel bypass route forwards literal interpretation, ensuring that a both reading remain available. Overall, our findings provide a mechanistic evidence for idiom comprehension in an autoregressive transformer. 

**Abstract (ZH)**: IDIOMS 给语言模型带来独特的挑战，因为它们的比喻意义是非组合性的，往往与字面意义有显著差异。这种二元性要求模型学会表示和决定两种意义，以便在比喻意义上解释成语，或者按字面意义理解。在本文中，我们利用机制可解释性工具追踪大规模预训练因果变换器（LLama3.2-1B-base）如何处理这种歧义。我们定位了成语处理的三个步骤：首先，在早期注意力和MLP子层中检索成语的比喻意义。我们确定了特定的注意力头，它们增强了成语的比喻意义同时抑制了字面意义。随后，模型通过中间路径表示比喻表示，同时通过并行旁路路径前向传递字面意义，确保两种读法都可用。总体而言，我们的发现为自回归变换器中的成语理解提供了机制性的证据。 

---
# When LLMs Team Up: The Emergence of Collaborative Affective Computing 

**Title (ZH)**: 当大型语言模型联手：合作情感计算的兴起 

**Authors**: Wenna Lai, Haoran Xie, Guandong Xu, Qing Li, S. Joe Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01698)  

**Abstract**: Affective Computing (AC) is essential in bridging the gap between human emotional experiences and machine understanding. Traditionally, AC tasks in natural language processing (NLP) have been approached through pipeline architectures, which often suffer from structure rigidity that leads to inefficiencies and limited adaptability. The advent of Large Language Models (LLMs) has revolutionized this field by offering a unified approach to affective understanding and generation tasks, enhancing the potential for dynamic, real-time interactions. However, LLMs face cognitive limitations in affective reasoning, such as misinterpreting cultural nuances or contextual emotions, and hallucination problems in decision-making. To address these challenges, recent research advocates for LLM-based collaboration systems that emphasize interactions among specialized models and LLMs, mimicking human-like affective intelligence through the synergy of emotional and rational thinking that aligns with Dual Process Theory in psychology. This survey aims to provide a comprehensive overview of LLM-based collaboration systems in AC, exploring from structured collaborations to autonomous collaborations. Specifically, it includes: (1) A systematic review of existing methods, focusing on collaboration strategies, mechanisms, key functions, and applications; (2) Experimental comparisons of collaboration strategies across representative tasks in affective understanding and generation; (3) An analysis highlighting the potential of these systems to enhance robustness and adaptability in complex affective reasoning; (4) A discussion of key challenges and future research directions to further advance the field. This work is the first to systematically explore collaborative intelligence with LLMs in AC, paving the way for more powerful applications that approach human-like social intelligence. 

**Abstract (ZH)**: 情感计算中基于大型语言模型的合作系统：原理、实验与未来方向 

---
# ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge 

**Title (ZH)**: ESGenius：评估大语言模型在环境、社会与治理（ESG）及可持续性知识方面的表现 

**Authors**: Chaoyue He, Xin Zhou, Yi Wu, Xinjia Yu, Yan Zhang, Lei Zhang, Di Wang, Shengfei Lyu, Hong Xu, Xiaoqiao Wang, Wei Liu, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01646)  

**Abstract**: We introduce ESGenius, a comprehensive benchmark for evaluating and enhancing the proficiency of Large Language Models (LLMs) in Environmental, Social and Governance (ESG) and sustainability-focused question answering. ESGenius comprises two key components: (i) ESGenius-QA, a collection of 1 136 multiple-choice questions generated by LLMs and rigorously validated by domain experts, covering a broad range of ESG pillars and sustainability topics. Each question is systematically linked to its corresponding source text, enabling transparent evaluation and supporting retrieval-augmented generation (RAG) methods; and (ii) ESGenius-Corpus, a meticulously curated repository of 231 foundational frameworks, standards, reports and recommendation documents from seven authoritative sources. Moreover, to fully assess the capabilities and adaptation potential of the model, we implement a rigorous two-stage evaluation protocol -- Zero-Shot and RAG. Extensive experiments across 50 LLMs (ranging from 0.5 B to 671 B parameters) demonstrate that state-of-the-art models achieve only moderate performance in zero-shot settings, with accuracies typically around 55--70\%, highlighting ESGenius's challenging nature for LLMs in interdisciplinary contexts. However, models employing RAG show significant performance improvements, particularly for smaller models. For example, "DeepSeek-R1-Distill-Qwen-14B" improves from 63.82\% (zero-shot) to 80.46\% with RAG. These results underscore the necessity of grounding responses in authoritative sources for enhanced ESG understanding. To the best of our knowledge, ESGenius is the first benchmark curated for LLMs and the relevant enhancement technologies that focuses on ESG and sustainability topics. 

**Abstract (ZH)**: ESGenius：用于评估和提高大型语言模型在环境、社会与治理（ESG）和可持续性问题回答能力的全面基准 

---
# Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification 

**Title (ZH)**: 基于梯度的模型指纹识别用于大规模语言模型相似性检测和家族分类 

**Authors**: Zehao Wu, Yanjie Zhao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01631)  

**Abstract**: As Large Language Models (LLMs) become integral software components in modern applications, unauthorized model derivations through fine-tuning, merging, and redistribution have emerged as critical software engineering challenges. Unlike traditional software where clone detection and license compliance are well-established, the LLM ecosystem lacks effective mechanisms to detect model lineage and enforce licensing agreements. This gap is particularly problematic when open-source model creators, such as Meta's LLaMA, require derivative works to maintain naming conventions for attribution, yet no technical means exist to verify compliance.
To fill this gap, treating LLMs as software artifacts requiring provenance tracking, we present TensorGuard, a gradient-based fingerprinting framework for LLM similarity detection and family classification. Our approach extracts model-intrinsic behavioral signatures by analyzing gradient responses to random input perturbations across tensor layers, operating independently of training data, watermarks, or specific model formats. TensorGuard supports the widely-adopted safetensors format and constructs high-dimensional fingerprints through statistical analysis of gradient features. These fingerprints enable two complementary capabilities: direct pairwise similarity assessment between arbitrary models through distance computation, and systematic family classification of unknown models via the K-Means clustering algorithm with domain-informed centroid initialization using known base models. Experimental evaluation on 58 models comprising 8 base models and 50 derivatives across five model families (Llama, Qwen, Gemma, Phi, Mistral) demonstrates 94% classification accuracy under our centroid-initialized K-Means clustering. 

**Abstract (ZH)**: 大型语言模型（LLMs）成为现代应用核心软件组件后，通过微调、合并和重新分发进行的未经授权的模型衍生已成为关键的软件工程挑战。不同于传统的软件，其中克隆检测和许可证合规已有成熟机制，LLM生态系统缺乏有效的机制来追踪模型血统并执行许可协议。这一缺口在开源模型创作者如Meta的LLaMA要求衍生作品遵守命名惯例进行 attribution 时尤其突出，但目前没有技术手段来验证合规性。

为填补这一缺口，将LLMs视为需要溯源追踪的软件制品，我们提出了TensorGuard，一种基于梯度的指纹识别框架，用于检测LLM相似性和分类。我们的方法通过分析横跨张量层的随机输入扰动的梯度响应来提取模型固有的行为特征签名，与训练数据、水印或特定模型格式无关。TensorGuard支持广泛采用的safetensors格式，并通过统计分析梯度特征构建高维指纹。这些指纹使我们具备两种互补的能力：通过距离计算直接对任意模型进行成对相似性评估，以及通过基于领域知识的中心点初始化的K-Means聚类算法系统地对未知模型进行分类。在包含8个基模型和50个衍生模型的58个模型（包括Llama、Qwen、Gemma、Phi、Mistral）组合中的实验评估显示，在我们中心点初始化的K-Means聚类下分类准确率达到94%。 

---
# V-VAE: A Variational Auto Encoding Framework Towards Fine-Grained Control over Human-Like Chat 

**Title (ZH)**: V-VAE：一种细粒度控制类人类对话的变分自编码框架 

**Authors**: Qi Lin, Weikai Xu, Lisi Chen, Bin Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.01524)  

**Abstract**: With the continued proliferation of Large Language Model (LLM) based chatbots, there is a growing demand for generating responses that are not only linguistically fluent but also consistently aligned with persona-specific traits in conversations. However, existing role-play and persona-based chat approaches rely heavily on static role descriptions, coarse-grained signal space, and low-quality synthetic data, which fail to capture dynamic fine-grained details in human-like chat. Human-like chat requires modeling subtle latent traits, such as emotional tone, situational awareness, and evolving personality, which are difficult to predefine and cannot be easily learned from synthetic or distillation-based data. To address these limitations, we propose a Verbal Variational Auto-Encoding (V-VAE) framework, containing a variational auto-encoding module and fine-grained control space which dynamically adapts dialogue behaviour based on fine-grained, interpretable latent variables across talking style, interaction patterns, and personal attributes. We also construct a high-quality dataset, HumanChatData, and benchmark HumanChatBench to address the scarcity of high-quality data in the human-like domain. Experiments show that LLMs based on V-VAE consistently outperform standard baselines on HumanChatBench and DialogBench, which further demonstrates the effectiveness of V-VAE and HumanChatData. 

**Abstract (ZH)**: 基于Verbal变分自编码器的人类般对话生成框架 

---
# Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes 

**Title (ZH)**: 大型语言模型中事实、虚构与预测的表征：知识论与态度 

**Authors**: Meng Li, Michael Vrazitulis, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01512)  

**Abstract**: Rational speakers are supposed to know what they know and what they do not know, and to generate expressions matching the strength of evidence. In contrast, it is still a challenge for current large language models to generate corresponding utterances based on the assessment of facts and confidence in an uncertain real-world environment. While it has recently become popular to estimate and calibrate confidence of LLMs with verbalized uncertainty, what is lacking is a careful examination of the linguistic knowledge of uncertainty encoded in the latent space of LLMs. In this paper, we draw on typological frameworks of epistemic expressions to evaluate LLMs' knowledge of epistemic modality, using controlled stories. Our experiments show that the performance of LLMs in generating epistemic expressions is limited and not robust, and hence the expressions of uncertainty generated by LLMs are not always reliable. To build uncertainty-aware LLMs, it is necessary to enrich semantic knowledge of epistemic modality in LLMs. 

**Abstract (ZH)**: 理性讲话者应知道自己知道什么和不知道什么，并生成与证据强度相匹配的表达。相比之下，当前的大规模语言模型仍然难以在不确定的现实环境中根据事实评估和置信度生成相应的言语。虽然最近评估和校准大规模语言模型信心的程度变得流行，但缺乏对大规模语言模型潜在空间中编码的表征知识的研究。在这项研究中，我们利用表征知识表达的类型学框架来评估大规模语言模型的命题模态知识，使用控制故事进行实验。实验结果显示，大规模语言模型生成命题模态表达的能力有限且不稳健，因此由大规模语言模型生成的不确定表达并非总是可靠的。为了构建意识不确定性的大规模语言模型，有必要在大规模语言模型中丰富命题模态的语义知识。 

---
# Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models 

**Title (ZH)**: 激励大型语言模型进行高级指令跟随推理 

**Authors**: Yulei Qin, Gang Li, Zongyi Li, Zihan Xu, Yuchen Shi, Zhekai Lin, Xiao Cui, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01413)  

**Abstract**: Existing large language models (LLMs) face challenges of following complex instructions, especially when multiple constraints are present and organized in paralleling, chaining, and branching structures. One intuitive solution, namely chain-of-thought (CoT), is expected to universally improve capabilities of LLMs. However, we find that the vanilla CoT exerts a negative impact on performance due to its superficial reasoning pattern of simply paraphrasing the instructions. It fails to peel back the compositions of constraints for identifying their relationship across hierarchies of types and dimensions. To this end, we propose a systematic method to boost LLMs in dealing with complex instructions via incentivizing reasoning for test-time compute scaling. First, we stem from the decomposition of complex instructions under existing taxonomies and propose a reproducible data acquisition method. Second, we exploit reinforcement learning (RL) with verifiable rule-centric reward signals to cultivate reasoning specifically for instruction following. We address the shallow, non-essential nature of reasoning under complex instructions via sample-wise contrast for superior CoT enforcement. We also exploit behavior cloning of experts to facilitate steady distribution shift from fast-thinking LLMs to skillful reasoners. Extensive evaluations on seven comprehensive benchmarks confirm the validity of the proposed method, where a 1.5B LLM achieves 11.74% gains with performance comparable to a 8B LLM. Codes and data are available at this https URL. 

**Abstract (ZH)**: 现有大规模语言模型（LLMs）在遵循复杂指令方面面临挑战，尤其是当存在并行、串联和分支结构的多重约束时。一种直观的解决方案，即逐步思考（CoT），有望普遍提高LLMs的能力。然而，我们发现，传统的CoT由于其简单的重述指令的表面化推理模式，对性能产生了负面影响。它未能揭示不同层级和维度约束之间的关系。为了解决这一问题，我们提出了一种系统方法，通过激励推理以适应测试时计算扩展来促进LLMs处理复杂指令。首先，我们基于现有分类体系对复杂指令进行分解，并提出了一种可重复的数据采集方法。其次，我们利用验证性规则中心奖励信号的强化学习（RL）来培养专门针对指令遵循的推理能力。我们通过样本级对比强化CoT的有效性，解决了复杂指令下浅薄且非本质的推理问题。我们还利用专家行为克隆来促进从快速思考的LLMs到善于推理者的平稳分布转移。广泛的基准测试表明，所提出的方法具有有效性，一个1.5B的LLM在性能上与8B的LLM相当，获得了11.74%的提升。代码和数据可在以下链接获取。 

---
# Compiler Optimization via LLM Reasoning for Efficient Model Serving 

**Title (ZH)**: 基于LLM推理的编译优化以实现高效模型服务 

**Authors**: Sujun Tang, Christopher Priebe, Rohan Mahapatra, Lianhui Qin, Hadi Esmaeilzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.01374)  

**Abstract**: While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimization to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed REASONING COMPILER) that formulates optimization as a sequential, context-aware decision process, guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-aware transformations that reflect the current program state and accumulated performance feedback. Monte Carlo tree search (MCTS) incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization. 

**Abstract (ZH)**: 大型语言模型引导的编译优化推理：显著提高样本效率的研究 

---
# Incentivizing LLMs to Self-Verify Their Answers 

**Title (ZH)**: 激励大语言模型自我验证其答案 

**Authors**: Fuxiang Zhang, Jiacheng Xu, Chaojie Wang, Ce Cui, Yang Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2506.01369)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable progress in complex reasoning tasks through both post-training and test-time scaling laws. While prevalent test-time scaling approaches are often realized by using external reward models to guide the model generation process, we find only marginal gains can be acquired when scaling a model post-trained on specific reasoning tasks. We identify that the limited improvement stems from distribution discrepancies between the specific post-trained generator and the general reward model. To address this, we propose a framework that incentivizes LLMs to self-verify their own answers. By unifying answer generation and verification within a single reinforcement learning (RL) process, we train models that can effectively assess the correctness of their own solutions. The trained model can further scale its performance during inference time by verifying its generations, without the need for external verifiers. We train our self-verification models based on Qwen2.5-Math-7B and DeepSeek-R1-Distill-Qwen-1.5B, demonstrating its capabilities across varying reasoning context lengths. Experiments on multiple mathematical reasoning benchmarks show that our models can not only improve post-training performance but also enable effective test-time scaling. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过训练后和测试时的扩增规律在复杂推理任务中取得了显著进展。虽然常见的测试时扩增方法通常是通过使用外部奖励模型来指导模型生成过程，但我们发现，对于特定推理任务训练后的模型扩增仅能获得微小的增益。我们发现这种有限的改进源自特定后训练生成器与通用奖励模型之间的分布差异。为此，我们提出了一种框架，激励LLMs自我验证其答案。通过在单一强化学习（RL）过程中统一答案生成和验证，我们训练出能够在推理时有效评估自身解决方案正确性的模型。训练好的模型在验证其生成内容时，可以在不依赖外部验证者的情况下进一步提升性能。我们基于Qwen2.5-Math-7B和DeepSeek-R1-Distill-Qwen-1.5B训练自我验证模型，展示了其在不同推理情境长度下的能力。我们在多个数学推理基准上的实验表明，我们的模型不仅能提高后训练性能，还能实现有效的测试时扩增。我们的代码可通过以下链接访问。 

---
# KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors 

**Title (ZH)**: KokoroChat：由受训心理咨询师通过角色扮演收集的日语心理咨询服务对话数据集 

**Authors**: Zhiyang Qi, Takumasa Kaneko, Keiko Takamizo, Mariko Ukiyo, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2506.01357)  

**Abstract**: Generating psychological counseling responses with language models relies heavily on high-quality datasets. Crowdsourced data collection methods require strict worker training, and data from real-world counseling environments may raise privacy and ethical concerns. While recent studies have explored using large language models (LLMs) to augment psychological counseling dialogue datasets, the resulting data often suffers from limited diversity and authenticity. To address these limitations, this study adopts a role-playing approach where trained counselors simulate counselor-client interactions, ensuring high-quality dialogues while mitigating privacy risks. Using this method, we construct KokoroChat, a Japanese psychological counseling dialogue dataset comprising 6,589 long-form dialogues, each accompanied by comprehensive client feedback. Experimental results demonstrate that fine-tuning open-source LLMs with KokoroChat improves both the quality of generated counseling responses and the automatic evaluation of counseling dialogues. The KokoroChat dataset is available at this https URL. 

**Abstract (ZH)**: 使用语言模型生成心理咨询服务响应依赖于高質量数据集。 Crowd-sourced数据采集方法需要严格的工人培训，而来自真实心理咨询环境的数据可能引发隐私和伦理问题。尽管最近的研究探索了利用大规模语言模型（LLMs）扩充心理咨询服务对话数据集，但由此产生的数据往往缺乏多样性和真实性。为解决这些局限性，本研究采用角色扮演方法，由训练有素的咨询师模拟咨询师-来访者互动，确保高质量对话的同时减轻隐私风险。通过这种方法，我们构建了包含6,589条长对话的KokoroChat日语心理咨询服务对话数据集，每条对话均附有全面的来访者反馈。实验结果表明，使用KokoroChat微调开源语言模型可以提高生成的心理咨询响应质量和对话自动评估效果。KokoroChat数据集可从此链接获得：this https URL。 

---
# ETDI: Mitigating Tool Squatting and Rug Pull Attacks in Model Context Protocol (MCP) by using OAuth-Enhanced Tool Definitions and Policy-Based Access Control 

**Title (ZH)**: ETDI：通过使用OAuth增强的工具定义和基于策略的访问控制来缓解模型上下文协议（MCP）中的工具霸占和 rug pull 攻击 

**Authors**: Manish Bhatt, Vineeth Sai Narajala, Idan Habler  

**Link**: [PDF](https://arxiv.org/pdf/2506.01333)  

**Abstract**: The Model Context Protocol (MCP) plays a crucial role in extending the capabilities of Large Language Models (LLMs) by enabling integration with external tools and data sources. However, the standard MCP specification presents significant security vulnerabilities, notably Tool Poisoning and Rug Pull attacks. This paper introduces the Enhanced Tool Definition Interface (ETDI), a security extension designed to fortify MCP. ETDI incorporates cryptographic identity verification, immutable versioned tool definitions, and explicit permission management, often leveraging OAuth 2.0. We further propose extending MCP with fine-grained, policy-based access control, where tool capabilities are dynamically evaluated against explicit policies using a dedicated policy engine, considering runtime context beyond static OAuth scopes. This layered approach aims to establish a more secure, trustworthy, and controllable ecosystem for AI applications interacting with LLMs and external tools. 

**Abstract (ZH)**: Enhanced Tool Definition Interface (ETDI)：MCP的安全扩展 

---
# Evaluating Large Language Models in Crisis Detection: A Real-World Benchmark from Psychological Support Hotlines 

**Title (ZH)**: 危机检测中大型语言模型的评估：来自心理支持热线的现实世界基准 

**Authors**: Guifeng Deng, Shuyin Rao, Tianyu Lin, Anlu Dai, Pan Wang, Junyi Xie, Haidong Song, Ke Zhao, Dongwu Xu, Zhengdong Cheng, Tao Li, Haiteng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01329)  

**Abstract**: Psychological support hotlines are critical for crisis intervention but face significant challenges due to rising demand. Large language models (LLMs) could support crisis assessments, yet their capabilities in emotionally sensitive contexts remain unclear. We introduce PsyCrisisBench, a benchmark of 540 annotated transcripts from the Hangzhou Psychological Assistance Hotline, assessing four tasks: mood status recognition, suicidal ideation detection, suicide plan identification, and risk assessment. We evaluated 64 LLMs across 15 families (e.g., GPT, Claude, Gemini, Llama, Qwen, DeepSeek) using zero-shot, few-shot, and fine-tuning paradigms. Performance was measured by F1-score, with statistical comparisons via Welch's t-tests. LLMs performed strongly on suicidal ideation detection (F1=0.880), suicide plan identification (F1=0.779), and risk assessment (F1=0.907), improved with few-shot and fine-tuning. Mood status recognition was more challenging (max F1=0.709), likely due to lost vocal cues and ambiguity. A fine-tuned 1.5B-parameter model (Qwen2.5-1.5B) surpassed larger models on mood and suicidal ideation. Open-source models like QwQ-32B performed comparably to closed-source on most tasks (p>0.3), though closed models retained an edge in mood detection (p=0.007). Performance scaled with size up to a point; quantization (AWQ) reduced GPU memory by 70% with minimal F1 degradation. LLMs show substantial promise in structured psychological crisis assessments, especially with fine-tuning. Mood recognition remains limited due to contextual complexity. The narrowing gap between open- and closed-source models, combined with efficient quantization, suggests feasible integration. PsyCrisisBench offers a robust evaluation framework to guide model development and ethical deployment in mental health. 

**Abstract (ZH)**: 心理支持热线在危机干预中至关重要，但面对不断增长的需求面临重大挑战。大型语言模型（LLMs）可以支持危机评估，但在情绪敏感情境下的能力尚不清楚。我们引入了PsyCrisisBench基准数据集，包含来自杭州心理援助热线的540份标注转录文本，评估四项任务：情绪状态识别、自杀观念检测、自杀计划识别和风险评估。我们使用零样本、少样本和微调方法评估了64种LLM（如GPT、Claude、Gemini、Llama、Qwen、DeepSeek）的表现。性能通过F1分数衡量，并通过Welch’s t检验进行统计比较。LLMs在自杀观念检测（F1=0.880）、自杀计划识别（F1=0.779）和风险评估（F1=0.907）方面表现强劲，少样本和微调时表现更好。情绪状态识别更具挑战性（最大F1=0.709），可能由于失去语音线索和模糊性。微调的1.5B参数模型（Qwen2.5-1.5B）在情绪和自杀观念识别方面超越了更大模型。开源模型如QwQ-32B在大多数任务上表现与闭源模型相当（p>0.3），尽管闭源模型在情绪检测上仍有一定优势（p=0.007）。性能随着模型规模的增加而增强，但存在临界点；量化（AWQ）可将GPU内存减少70%，同时F1分数下降有限。LLMs在结构化的心理危机评估中表现出巨大潜力，特别是通过微调。情绪识别受限于上下文复杂性。开源和闭源模型之间的差距缩小，以及高效的量化，表明这些模型的集成是可行的。PsyCrisisBench提供了 robust的评估框架，以指导心理健康领域的模型开发和伦理部署。 

---
# T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning 

**Title (ZH)**: T-SHIRT: 颗粒度选择性的层次化数据选择用于指令调优 

**Authors**: Yanjun Fu, Faisal Hamman, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2506.01317)  

**Abstract**: Instruction tuning is essential for Large Language Models (LLMs) to effectively follow user instructions. To improve training efficiency and reduce data redundancy, recent works use LLM-based scoring functions, e.g., Instruction-Following Difficulty (IFD), to select high-quality instruction-tuning data with scores above a threshold. While these data selection methods often lead to models that can match or even exceed the performance of models trained on the full datasets, we identify two key limitations: (i) they assess quality at the sample level, ignoring token-level informativeness; and (ii) they overlook the robustness of the scoring method, often selecting a sample due to superficial lexical features instead of its true quality. In this work, we propose Token-Selective HIeRarchical Data Selection for Instruction Tuning (T-SHIRT), a novel data selection framework that introduces a new scoring method to include only informative tokens in quality evaluation and also promotes robust and reliable samples whose neighbors also show high quality with less local inconsistencies. We demonstrate that models instruction-tuned on a curated dataset (only 5% of the original size) using T-SHIRT can outperform those trained on the entire large-scale dataset by up to 5.48 points on average across eight benchmarks. Across various LLMs and training set scales, our method consistently surpasses existing state-of-the-art data selection techniques, while also remaining both cost-effective and highly efficient. For instance, by using GPT-2 for score computation, we are able to process a dataset of 52k samples using 40 minutes on a single GPU. 

**Abstract (ZH)**: Token-Selective HIeRarchical Data Selection for Instruction Tuning (T-SHIRT) 

---
# Align is not Enough: Multimodal Universal Jailbreak Attack against Multimodal Large Language Models 

**Title (ZH)**: 对齐不够：针对多模态大规模语言模型的多模态通用 Jailbreak 攻击 

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Jing Liu, Hanwang Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01307)  

**Abstract**: Large Language Models (LLMs) have evolved into Multimodal Large Language Models (MLLMs), significantly enhancing their capabilities by integrating visual information and other types, thus aligning more closely with the nature of human intelligence, which processes a variety of data forms beyond just text. Despite advancements, the undesirable generation of these models remains a critical concern, particularly due to vulnerabilities exposed by text-based jailbreak attacks, which have represented a significant threat by challenging existing safety protocols. Motivated by the unique security risks posed by the integration of new and old modalities for MLLMs, we propose a unified multimodal universal jailbreak attack framework that leverages iterative image-text interactions and transfer-based strategy to generate a universal adversarial suffix and image. Our work not only highlights the interaction of image-text modalities can be used as a critical vulnerability but also validates that multimodal universal jailbreak attacks can bring higher-quality undesirable generations across different MLLMs. We evaluate the undesirable context generation of MLLMs like LLaVA, Yi-VL, MiniGPT4, MiniGPT-v2, and InstructBLIP, and reveal significant multimodal safety alignment issues, highlighting the inadequacy of current safety mechanisms against sophisticated multimodal attacks. This study underscores the urgent need for robust safety measures in MLLMs, advocating for a comprehensive review and enhancement of security protocols to mitigate potential risks associated with multimodal capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已发展成为多模态大规模语言模型（MLLMs），通过整合视觉信息和其他类型的信息，显著提升了其能力，并更接近人类智能处理多种数据形式的特性。尽管取得了进步，但这些模型的不良生成仍然是一个关键问题，特别是由于基于文本的监狱打破攻击揭示的漏洞，这些攻击代表了对现有安全协议的重大威胁。受MLLMs整合新旧模态所带来的独特安全风险的启发，我们提出了一种统一的多模态通用监狱打破攻击框架，该框架利用迭代的图像-文本交互和基于迁移的策略生成通用对抗后缀和图像。我们的工作不仅表明图像-文本模态的交互可以作为关键漏洞使用，而且证明了多模态通用监狱打破攻击可以在不同MLLMs中带来更高质量的不良生成。我们评估了包括LLaVA、Yi-VL、MiniGPT4、MiniGPT-v2和InstructBLIP在内的多模态大规模语言模型的不良上下文生成，并揭示了重要的多模态安全对齐问题，突显了当前安全机制在应对复杂多模态攻击方面的不足。本研究强调了在MLLMs中迫切需要强大的安全措施，呼吁进行全面的安全协议审查和增强，以减轻与多模态能力相关的潜在风险。 

---
# Abstractive Visual Understanding of Multi-modal Structured Knowledge: A New Perspective for MLLM Evaluation 

**Title (ZH)**: 多模态结构化知识的抽象视觉理解：MLLM评估的新视角 

**Authors**: Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Min Zhang, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01293)  

**Abstract**: Multi-modal large language models (MLLMs) incorporate heterogeneous modalities into LLMs, enabling a comprehensive understanding of diverse scenarios and objects. Despite the proliferation of evaluation benchmarks and leaderboards for MLLMs, they predominantly overlook the critical capacity of MLLMs to comprehend world knowledge with structured abstractions that appear in visual form. To address this gap, we propose a novel evaluation paradigm and devise M3STR, an innovative benchmark grounded in the Multi-Modal Map for STRuctured understanding. This benchmark leverages multi-modal knowledge graphs to synthesize images encapsulating subgraph architectures enriched with multi-modal entities. M3STR necessitates that MLLMs not only recognize the multi-modal entities within the visual inputs but also decipher intricate relational topologies among them. We delineate the benchmark's statistical profiles and automated construction pipeline, accompanied by an extensive empirical analysis of 26 state-of-the-art MLLMs. Our findings reveal persistent deficiencies in processing abstractive visual information with structured knowledge, thereby charting a pivotal trajectory for advancing MLLMs' holistic reasoning capacities. Our code and data are released at this https URL 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）将异构模态整合到大规模语言模型中，使其能够全面理解多样化的场景和对象。尽管已经出现了多种评估基准和排行榜来评估MLLMs，但它们大多忽略了MLLMs理解以视觉形式出现的结构化知识的能力。为了填补这一空白，我们提出了一种新的评估范式，并设计了M3STR，一个基于多模态地图的结构理解基准。该基准利用多模态知识图谱合成了包含丰富多模态实体子图架构的图像。M3STR 要求 MLLOMs 不仅能够识别视觉输入中的多模态实体，还能够解析这些实体之间的复杂关系拓扑结构。我们详细阐述了基准的统计特征和自动化构建管道，并对26种最先进的MLLMs进行了详尽的实证分析。我们的发现揭示了在处理结构化知识的抽象视觉信息方面存在的持续性缺陷，从而指出了提高MLLMs整体推理能力的关键方向。我们的代码和数据在此处发布：这个httpsURL。 

---
# TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment 

**Title (ZH)**: TSRating: 通过元学习从大语言模型判断中评估多样化的时序数据质量 

**Authors**: Shunyu Wu, Dan Li, Haozheng Ye, Zhuomin Chen, Jiahui Zhou, Jian Lou, Zibin Zheng, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01290)  

**Abstract**: High-quality time series (TS) data are essential for ensuring TS model performance, rendering research on rating TS data quality indispensable. Existing methods have shown promising rating accuracy within individual domains, primarily by extending data quality rating techniques such as influence functions and Shapley values to account for temporal characteristics. However, they neglect the fact that real-world TS data can span vastly different domains and exhibit distinct properties, hampering the accurate and efficient rating of diverse TS data. In this paper, we propose TSRating, a novel and unified framework for rating the quality of time series data crawled from diverse domains. TSRating is built on the assumption that LLMs inherit ample knowledge, acquired during their extensive pretraining, enabling them to comprehend and discern quality differences in diverse TS data. We verify this assumption by devising a series of prompts to elicit quality comparisons from LLMs for pairs of TS samples. We then fit a dedicated rating model, termed TSRater, to convert the LLMs' judgments into efficient quality predictions via TSRater's inference on future TS samples. To ensure cross-domain adaptability, we develop a meta-learning scheme to train TSRater on quality comparisons collected from nine distinct domains. To improve training efficiency, we employ signSGD for inner-loop updates, thus circumventing the demanding computation of hypergradients. Extensive experimental results on eleven benchmark datasets across three time series tasks, each using both conventional TS models and TS foundation models, demonstrate that TSRating outperforms baselines in terms of estimation accuracy, efficiency, and domain adaptability. 

**Abstract (ZH)**: TS数据质量评估：一种跨领域的时间序列数据质量评估框架 

---
# Detoxification of Large Language Models through Output-layer Fusion with a Calibration Model 

**Title (ZH)**: 大型语言模型的去毒化通过与校准模型的输出层融合实现 

**Authors**: Yuanhe Tian, Mingjie Deng, Guoqing Jin, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.01266)  

**Abstract**: Existing approaches for Large language model (LLM) detoxification generally rely on training on large-scale non-toxic or human-annotated preference data, designing prompts to instruct the LLM to generate safe content, or modifying the model parameters to remove toxic information, which are computationally expensive, lack robustness, and often compromise LLMs' fluency and contextual understanding. In this paper, we propose a simple yet effective approach for LLM detoxification, which leverages a compact, pre-trained calibration model that guides the detoxification process of a target LLM via a lightweight intervention in its generation pipeline. By learning a detoxified embedding space from non-toxic data, the calibration model effectively steers the LLM away from generating harmful content. This approach only requires a one-time training of the calibration model that is able to be seamlessly applied to multiple LLMs without compromising fluency or contextual understanding. Experiment results on the benchmark dataset demonstrate that our approach reduces toxicity while maintaining reasonable content expression. 

**Abstract (ZH)**: 现有的大语言模型去毒化方法通常依赖于大规模无毒或人类标注的偏好数据训练、设计提示指令大语言模型生成安全内容，或是修改模型参数以移除有毒信息，这些方法计算成本高、 robustness 差，并且常会损害大语言模型的流畅性和语境理解能力。本文提出了一种简单而有效的去毒化方法，该方法利用一个紧凑的预训练校准模型，通过轻量级干预目标大语言模型的生成管道来引导去毒化过程。通过从无毒数据中学习去毒化嵌入空间，校准模型有效地引导大语言模型远离生成有害内容。该方法只需要对校准模型进行一次训练，即可无缝应用于多个大语言模型，而不影响其流畅性和语境理解能力。基准数据集上的实验结果表明，该方法在降低毒性的同时保持了合理的内容表达。 

---
# DeepSeek in Healthcare: A Survey of Capabilities, Risks, and Clinical Applications of Open-Source Large Language Models 

**Title (ZH)**: DeepSeek在医疗健康领域的调研：开源大规模语言模型的功能、风险及临床应用 

**Authors**: Jiancheng Ye, Sophie Bronstein, Jiarui Hai, Malak Abu Hashish  

**Link**: [PDF](https://arxiv.org/pdf/2506.01257)  

**Abstract**: DeepSeek-R1 is a cutting-edge open-source large language model (LLM) developed by DeepSeek, showcasing advanced reasoning capabilities through a hybrid architecture that integrates mixture of experts (MoE), chain of thought (CoT) reasoning, and reinforcement learning. Released under the permissive MIT license, DeepSeek-R1 offers a transparent and cost-effective alternative to proprietary models like GPT-4o and Claude-3 Opus; it excels in structured problem-solving domains such as mathematics, healthcare diagnostics, code generation, and pharmaceutical research. The model demonstrates competitive performance on benchmarks like the United States Medical Licensing Examination (USMLE) and American Invitational Mathematics Examination (AIME), with strong results in pediatric and ophthalmologic clinical decision support tasks. Its architecture enables efficient inference while preserving reasoning depth, making it suitable for deployment in resource-constrained settings. However, DeepSeek-R1 also exhibits increased vulnerability to bias, misinformation, adversarial manipulation, and safety failures - especially in multilingual and ethically sensitive contexts. This survey highlights the model's strengths, including interpretability, scalability, and adaptability, alongside its limitations in general language fluency and safety alignment. Future research priorities include improving bias mitigation, natural language comprehension, domain-specific validation, and regulatory compliance. Overall, DeepSeek-R1 represents a major advance in open, scalable AI, underscoring the need for collaborative governance to ensure responsible and equitable deployment. 

**Abstract (ZH)**: DeepSeek-R1是一种由DeepSeek开发的前沿开源大语言模型（LLM），通过结合专家混合架构（MoE）、思维链（CoT）推理和强化学习，展示了先进的推理能力。在MIT许可协议下发布，DeepSeek-R1为GPT-4o和Claude-3 Opus等专有模型提供了一个透明且成本效益高的替代方案；它在数学、医疗诊断、代码生成和制药研究等结构化问题解决领域表现出色。该模型在美国医学许可考试（USMLE）和美国邀请数学考试（AIME）等基准测试中显示出了竞争力，特别是在儿科和眼科临床决策支持任务中取得良好结果。其架构可在资源受限的环境中实现高效的推理的同时保持推理深度。然而，DeepSeek-R1也表现出对偏见、错误信息、对抗性操纵和安全失败的更大脆弱性，特别是在多语言和伦理敏感环境中。本文综述了模型的优势，包括可解释性、可扩展性和适应性，同时也指出了其在通用语言流畅性和安全对齐方面的局限性。未来的研究重点包括改进偏见缓解、自然语言理解、专业领域验证和合规性。总体而言，DeepSeek-R1代表了开放且可扩展AI的一项重大进步，强调了需要合作治理以确保负责任且公平的部署的重要性。 

---
# MTCMB: A Multi-Task Benchmark Framework for Evaluating LLMs on Knowledge, Reasoning, and Safety in Traditional Chinese Medicine 

**Title (ZH)**: MTCMB: 中医药领域知识、推理与安全评估的多任务基准框架 

**Authors**: Shufeng Kong, Xingru Yang, Yuanyuan Wei, Zijie Wang, Hao Tang, Jiuqi Qin, Shuting Lan, Yingheng Wang, Junwen Bai, Zhuangbin Chen, Zibin Zheng, Caihua Liu, Hao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01252)  

**Abstract**: Traditional Chinese Medicine (TCM) is a holistic medical system with millennia of accumulated clinical experience, playing a vital role in global healthcare-particularly across East Asia. However, the implicit reasoning, diverse textual forms, and lack of standardization in TCM pose major challenges for computational modeling and evaluation. Large Language Models (LLMs) have demonstrated remarkable potential in processing natural language across diverse domains, including general medicine. Yet, their systematic evaluation in the TCM domain remains underdeveloped. Existing benchmarks either focus narrowly on factual question answering or lack domain-specific tasks and clinical realism. To fill this gap, we introduce MTCMB-a Multi-Task Benchmark for Evaluating LLMs on TCM Knowledge, Reasoning, and Safety. Developed in collaboration with certified TCM experts, MTCMB comprises 12 sub-datasets spanning five major categories: knowledge QA, language understanding, diagnostic reasoning, prescription generation, and safety evaluation. The benchmark integrates real-world case records, national licensing exams, and classical texts, providing an authentic and comprehensive testbed for TCM-capable models. Preliminary results indicate that current LLMs perform well on foundational knowledge but fall short in clinical reasoning, prescription planning, and safety compliance. These findings highlight the urgent need for domain-aligned benchmarks like MTCMB to guide the development of more competent and trustworthy medical AI systems. All datasets, code, and evaluation tools are publicly available at: this https URL. 

**Abstract (ZH)**: 面向中医药知识、推理与安全的多任务基准MTCMB 

---
# Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean 

**Title (ZH)**: 优化GEM的每一个 facets：测试LLMs和人类在韩语中的语言能力 

**Authors**: SungHo Kim, Nayeon Kim, Taehee Jeon, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01237)  

**Abstract**: We introduce the $\underline{Ko}rean \underline{G}rammar \underline{E}valuation Bench\underline{M}ark (KoGEM)$, designed to assess the linguistic competence of LLMs and humans in Korean. KoGEM consists of 1.5k multiple-choice QA pairs covering five main categories and 16 subcategories. The zero-shot evaluation of 27 LLMs of various sizes and types reveals that while LLMs perform remarkably well on straightforward tasks requiring primarily definitional knowledge, they struggle with tasks that demand the integration of real-world experiential knowledge, such as phonological rules and pronunciation. Furthermore, our in-depth analysis suggests that incorporating such experiential knowledge could enhance the linguistic competence of LLMs. With KoGEM, we not only highlight the limitations of current LLMs in linguistic competence but also uncover hidden facets of LLMs in linguistic competence, paving the way for enhancing comprehensive language understanding. Our code and dataset are available at: this https URL. 

**Abstract (ZH)**: 韩语语法评估基准(KoGEM)：评估LLM和人类在韩语中的语言能力 

---
# Retrieval-Augmented Generation of Ontologies from Relational Databases 

**Title (ZH)**: 从关系数据库检索增强本体生成 

**Authors**: Mojtaba Nayyeri, Athish A Yogi, Nadeen Fathallah, Ratan Bahadur Thapa, Hans-Michael Tautenhahn, Anton Schnurpel, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2506.01232)  

**Abstract**: Transforming relational databases into knowledge graphs with enriched ontologies enhances semantic interoperability and unlocks advanced graph-based learning and reasoning over data. However, previous approaches either demand significant manual effort to derive an ontology from a database schema or produce only a basic ontology. We present RIGOR, Retrieval-augmented Iterative Generation of RDB Ontologies, an LLM-driven approach that turns relational schemas into rich OWL ontologies with minimal human effort. RIGOR combines three sources via RAG, the database schema and its documentation, a repository of domain ontologies, and a growing core ontology, to prompt a generative LLM for producing successive, provenance-tagged delta ontology fragments. Each fragment is refined by a judge-LLM before being merged into the core ontology, and the process iterates table-by-table following foreign key constraints until coverage is complete. Applied to real-world databases, our approach outputs ontologies that score highly on standard quality dimensions such as accuracy, completeness, conciseness, adaptability, clarity, and consistency, while substantially reducing manual effort. 

**Abstract (ZH)**: 将关系数据库转换为充实本体的知识图谱增强语义互操作性并解锁基于图的高级学习与推理。然而，先前的方法要么要求大量手动 effort 来从数据库模式推导本体，要么只能生成基础本体。我们提出 RIGOR：基于检索的迭代生成 RDB 本体方法，这是一种由大语言模型驱动的 approach，能够用最小的人工努力将关系模式转换为丰富的 OWL 本体。RIGOR 通过 RAG 结合数据库模式及其文档、领域本体仓库以及不断增长的核心本体，来提示生成型大语言模型以生成连续的、带有来源标注的 delta 本体片段。每个片段都由法官大语言模型 refining，然后合并到核心本体中，过程根据外键约束逐表迭代，直到完全覆盖。应用于实际数据库时，我们的方法在准确度、完整性、简洁性、可适应性、清晰性和一致性等标准质量维度上表现优异，同时大幅减少了人工努力。 

---
# Mamba Drafters for Speculative Decoding 

**Title (ZH)**: 猜想性解码的Mamba绘图工具 

**Authors**: Daewon Choi, Seunghyuk Oh, Saket Dingliwal, Jihoon Tack, Kyuyoung Kim, Woomin Song, Seojin Kim, Insu Han, Jinwoo Shin, Aram Galstyan, Shubham Katiyar, Sravan Babu Bodapati  

**Link**: [PDF](https://arxiv.org/pdf/2506.01206)  

**Abstract**: Speculative decoding has emerged as a promising approach to accelerating large language model (LLM) generation using a fast drafter while maintaining alignment with the target model's distribution. However, existing approaches face a trade-off: external drafters offer flexibility but can suffer from slower drafting, while self-speculation methods use drafters tailored to the target model but require re-training. In this paper, we introduce novel drafters based on Mamba, a state-of-the-art state space model (SSM), as a solution that combines the best aspects of both approaches. By leveraging the linear structure of SSMs, our approach avoids the quadratic complexity inherent in traditional Transformer-based methods, enabling faster drafting and lower memory usage while maintaining the flexibility to work across different target models. We further enhance efficiency with a novel test-time tree search algorithm for generating high-quality draft candidates. Our empirical evaluation demonstrates that Mamba-based drafters not only outperform existing external drafting methods but are also comparable to state-of-the-art self-speculation approaches while using less memory and maintaining their cross-model adaptability. 

**Abstract (ZH)**: 基于Mamba的状态空间模型的推测解码：结合外部草案器的灵活性和自我推测方法的适应性 

---
# Doubly Robust Alignment for Large Language Models 

**Title (ZH)**: 双重稳健对齐为大规模语言模型 

**Authors**: Erhan Xu, Kai Ye, Hongyi Zhou, Luhan Zhu, Francesco Quinzan, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01183)  

**Abstract**: This paper studies reinforcement learning from human feedback (RLHF) for aligning large language models with human preferences. While RLHF has demonstrated promising results, many algorithms are highly sensitive to misspecifications in the underlying preference model (e.g., the Bradley-Terry model), the reference policy, or the reward function, resulting in undesirable fine-tuning. To address model misspecification, we propose a doubly robust preference optimization algorithm that remains consistent when either the preference model or the reference policy is correctly specified (without requiring both). Our proposal demonstrates superior and more robust performance than state-of-the-art algorithms, both in theory and in practice. The code is available at this https URL 

**Abstract (ZH)**: 本文研究了人类反馈强化学习（RLHF）以使大规模语言模型与人类偏好一致。虽然RLHF展现出了令人鼓舞的结果，但许多算法对底层偏好模型（如Bradley-Terry模型）、参考策略或奖励函数中的错误规定非常敏感，导致不理想的模型微调。为了应对模型规定不准确的问题，我们提出了一种双重稳健的偏好优化算法，在偏好模型或参考策略中任一被正确指定的情况下（无需两者都正确）仍能保持一致性。我们的提议在理论和实践中都展现出了优于现有最佳算法的优越性和稳健性。代码可在以下链接获取。 

---
# Earley-Driven Dynamic Pruning for Efficient Structured Decoding 

**Title (ZH)**: Earley-驱动动态剪枝以实现高效的结构化解码 

**Authors**: Xintong Sun, Chi Wei, Minghao Tian, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.01151)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities, yet ensuring their outputs conform to strict structural or grammatical constraints remains challenging, which is critical in function calls and domain-specific language (DSL) generation. Constrained decoding with context-free grammar is a flexible approach to guarantee LLMs' adherence to a specific format by dynamically building a token logits mask. However, creating this mask requires checking the validity of all tokens in the LLM vocabulary at every decoding step, which often incurs significant overheads in existing constrained decoding engines. To address this challenge, we propose $\textbf{ZapFormat}$, a novel $\textbf{dynamic pruning}$ strategy based on the Earley algorithm that identifies and eliminates invalid or redundant Earley states in real-time, significantly reducing memory occupation of the Earley algorithm's states. This further enables us to use a state cache to speed up structured generations on a large number of queries. We implemented ZapFormat in a new constrained decoding engine called Formatron which also incorporates existing optimizations. Through comprehensive experiments on structured generation tasks, including JSON generation, JSON Schema validation, and semantic parsing, we demonstrate that Formatron not only $\textbf{consistently maintains}$ high-precision compliant outputs but also achieves $\textbf{significant improvements}$ in inference speed up to 2x compared to state-of-the-art implementations. More importantly, Formatron is generally applicable across various LLM architectures. We release Formatron as open source at this https URL. 

**Abstract (ZH)**: 基于Earley算法的动态剪枝策略ZapFormat在严格格式化生成中的应用 

---
# From Words to Waves: Analyzing Concept Formation in Speech and Text-Based Foundation Models 

**Title (ZH)**: 从词语到波浪：分析基于语言和文本的基础模型中的概念形成 

**Authors**: Asım Ersoy, Basel Mousi, Shammur Chowdhury, Firoj Alam, Fahim Dalvi, Nadir Durrani  

**Link**: [PDF](https://arxiv.org/pdf/2506.01133)  

**Abstract**: The emergence of large language models (LLMs) has demonstrated that systems trained solely on text can acquire extensive world knowledge, develop reasoning capabilities, and internalize abstract semantic concepts--showcasing properties that can be associated with general intelligence. This raises an intriguing question: Do such concepts emerge in models trained on other modalities, such as speech? Furthermore, when models are trained jointly on multiple modalities: Do they develop a richer, more structured semantic understanding? To explore this, we analyze the conceptual structures learned by speech and textual models both individually and jointly. We employ Latent Concept Analysis, an unsupervised method for uncovering and interpreting latent representations in neural networks, to examine how semantic abstractions form across modalities. For reproducibility we made scripts and other resources available to the community. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现表明，仅基于文本训练的系统可以获取广泛的世界知识、发展推理能力并内化抽象语义概念，这体现了与通用智能相关的特征。这引发了一个有趣的问题：在其他模态（如语音）上训练的模型是否会形成这样的概念？此外，当模型在多种模态上联合训练时，它们是否会发展出更丰富、更结构化的语义理解？为了探究这一点，我们分析了单个及联合训练的语音和文本模型所学到的概念结构。我们使用潜在概念分析（Latent Concept Analysis），这是一种用于发现和解释神经网络中潜在表示的无监督方法，来研究语义抽象如何跨模态形成。为了可重复性，我们向社区提供了相关脚本和其他资源。 

---
# Reconsidering LLM Uncertainty Estimation Methods in the Wild 

**Title (ZH)**: 重新审视实际应用中的大规模语言模型不确定性估计方法 

**Authors**: Yavuz Bakman, Duygu Nur Yaldiz, Sungmin Kang, Tuo Zhang, Baturalp Buyukates, Salman Avestimehr, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.01114)  

**Abstract**: Large Language Model (LLM) Uncertainty Estimation (UE) methods have become a crucial tool for detecting hallucinations in recent years. While numerous UE methods have been proposed, most existing studies evaluate them in isolated short-form QA settings using threshold-independent metrics such as AUROC or PRR. However, real-world deployment of UE methods introduces several challenges. In this work, we systematically examine four key aspects of deploying UE methods in practical settings. Specifically, we assess (1) the sensitivity of UE methods to decision threshold selection, (2) their robustness to query transformations such as typos, adversarial prompts, and prior chat history, (3) their applicability to long-form generation, and (4) strategies for handling multiple UE scores for a single query. Our evaluations on 19 UE methods reveal that most of them are highly sensitive to threshold selection when there is a distribution shift in the calibration dataset. While these methods generally exhibit robustness against previous chat history and typos, they are significantly vulnerable to adversarial prompts. Additionally, while existing UE methods can be adapted for long-form generation through various strategies, there remains considerable room for improvement. Lastly, ensembling multiple UE scores at test time provides a notable performance boost, which highlights its potential as a practical improvement strategy. Code is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型不确定性估计方法在实际部署中的关键考量 

---
# Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements 

**Title (ZH)**: 忽视语境信息：评估LLMs对指称元素的理解 

**Authors**: Metehan Oguz, Yavuz Bakman, Duygu Nur Yaldiz  

**Link**: [PDF](https://arxiv.org/pdf/2506.01089)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performances in tasks related to coreference resolution. However, previous studies mostly assessed LLM performance on coreference resolution with nouns and third person pronouns. This study evaluates LLM performance on coreference resolution with indexical like I, you, here and tomorrow, which come with unique challenges due to their linguistic properties. We present the first study examining how LLMs interpret indexicals in English, releasing the English Indexical Dataset with 1600 multiple-choice questions. We evaluate pioneering LLMs, including GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and DeepSeek V3. Our results reveal that LLMs exhibit an impressive performance with some indexicals (I), while struggling with others (you, here, tomorrow), and that syntactic cues (e.g. quotation) contribute to LLM performance with some indexicals, while they reduce performance with others. Code and data are available at: this https URL. 

**Abstract (ZH)**: 大型语言模型在处理代词理解任务中的表现：以“I”、“you”、“here”和“tomorrow”为例 

---
# Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs 

**Title (ZH)**: 以火攻火（F3）：一种无需训练且高效的视觉对抗样本净化方法在LVLMs中的应用 

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01064)  

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. Despite their potential impact, the development of effective methods for purifying such adversarial examples has received relatively limited attention. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive "fighting fire with fire" strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code will be made publicly available. 

**Abstract (ZH)**: 最近大型多模态视觉语言模型的进展展示了其在多种跨模态视觉语言任务中的出色能力。然而，这些模型仍然容易受到视觉对抗攻击的影响，这可能会显著损害其性能。尽管对抗攻击可能具有重大影响，但开发有效的对抗样本净化方法仍相对受到较少的关注。在本文中，我们介绍了F3，一种新颖的对抗样本净化框架，采用了反直觉的“以火攻火”策略：故意引入简单的扰动到对抗样本中以减轻其有害影响。具体而言，F3 利用来自随机扰动对手样本的跨模态注意作为参考目标。通过向这些对抗样本注入噪声，F3 有效改进了它们的注意机制，从而产生更清洁和更可靠的模型输出。令人惊讶的是，这种看似矛盾的方法——通过引入噪声来对抗对抗攻击——取得了令人印象深刻的净化效果。此外，F3 还提供了几个显著的优势：它无需训练且易于实现，并且在计算效率方面比现有净化方法表现出明显的改进。这些特性使得 F3 特别适用于大规模工业应用，其中稳健性能和操作效率是至关重要的优先事项。代码将公开发布。 

---
# Taming LLMs by Scaling Learning Rates with Gradient Grouping 

**Title (ZH)**: 通过梯度分组缩放学习率来驯服大规模语言模型 

**Authors**: Siyuan Li, Juanxi Tian, Zedong Wang, Xin Jin, Zicheng Liu, Wentao Zhang, Dan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01049)  

**Abstract**: Training large language models (LLMs) poses challenges due to their massive scale and heterogeneous architectures. While adaptive optimizers like AdamW help address gradient variations, they still struggle with efficient and effective parameter-wise learning rate estimation, resulting in training instability, slow convergence, and poor compatibility with parameter-efficient fine-tuning (PEFT) techniques. This work introduces Scaling with Gradient Grouping (SGG), an optimizer wrapper that improves adaptive learning rate estimation by dynamic grouping and group-specific scaling. SGG first groups gradient statistics in each layer into clusters and then applies cluster-specific scaling to calibrate learning rates for each parameter, thus imposing collective group-wise constraints while maintaining precise per-parameter adaptation. Experiments on diverse (M)LLM benchmarks show that SGG integrates seamlessly with existing optimizers, and offers consistent gains and faster convergence over baselines, with various model sizes. Its stability across varying batch sizes and learning rates establishes SGG as a robust choice for LLM optimization. 

**Abstract (ZH)**: 训练大规模语言模型（LLMs）由于其庞大的规模和异构架构而面临挑战。尽管自适应优化器如AdamW有助于处理梯度变化，但它们在参数级学习率估计的效率和有效性方面仍存在问题，导致训练不稳定、收敛缓慢以及与参数高效微调（PEFT）技术的兼容性差。本文介绍了梯度群组缩放（SGG），这是一种优化器包装器，通过动态分组和组别特定缩放来提高自适应学习率估计。SGG首先将每一层的梯度统计信息分组成簇，然后对每个簇应用特定缩放来校准每个参数的学习率，从而施加集体的群组约束，同时保留精细的参数级调整。在多种（M）LLM基准测试中，SGG与现有优化器无缝集成，并在各种模型规模上提供了持续的收益和更快的收敛速度。其在不同批量大小和学习率下的稳定性确立了SGG作为LLM优化稳健选择的地位。 

---
# Probing Neural Topology of Large Language Models 

**Title (ZH)**: 探查大型语言模型的神经拓扑结构 

**Authors**: Yu Zheng, Yuan Yuan, Yong Li, Paolo Santi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01042)  

**Abstract**: Probing large language models (LLMs) has yielded valuable insights into their internal mechanisms by linking neural representations to interpretable semantics. However, how neurons functionally co-activate with each other to give rise to emergent capabilities remains largely unknown, hindering a deeper understanding and safer development of LLMs. In this work, we introduce graph probing, a method for uncovering the functional connectivity topology of LLM neurons and relating it to language generation performance. By analyzing internal neural graphs across diverse LLM families and scales, we discover a universal predictability of next-token prediction performance using only neural topology. This predictability is robust even when retaining just 1% of neuron connections or probing models after only 8 pretraining steps, highlighting the sparsity and early emergence of topological patterns. Further graph matching analysis suggests that, despite significant distinctions in architectures, parameters, and training data, different LLMs develop intricate and consistent neural topological structures that may form the foundation for their language generation abilities. Codes and data for the graph probing toolbox are released at this https URL. 

**Abstract (ZH)**: 探查大规模语言模型的图探查方法揭示了其神经元功能连接拓扑与其语言生成性能之间的关系，尽管不同架构、参数和训练数据的LLMs在拓扑结构上表现出复杂的但一致的模式，这些模式可能构成了其语言生成能力的基础。该图探查工具箱的代码和数据可在以下链接获取：this https URL。 

---
# Less is More: Local Intrinsic Dimensions of Contextual Language Models 

**Title (ZH)**: 少即是多：情境语言模型的局部内在维度 

**Authors**: Benjamin Matthias Ruppik, Julius von Rohrscheidt, Carel van Niekerk, Michael Heck, Renato Vukovic, Shutong Feng, Hsien-chin Lin, Nurul Lubis, Bastian Rieck, Marcus Zibrowius, Milica Gašić  

**Link**: [PDF](https://arxiv.org/pdf/2506.01034)  

**Abstract**: Understanding the internal mechanisms of large language models (LLMs) remains a challenging and complex endeavor. Even fundamental questions, such as how fine-tuning affects model behavior, often require extensive empirical evaluation. In this paper, we introduce a novel perspective based on the geometric properties of contextual latent embeddings to study the effects of training and fine-tuning. To that end, we measure the local dimensions of a contextual language model's latent space and analyze their shifts during training and fine-tuning. We show that the local dimensions provide insights into the model's training dynamics and generalization ability. Specifically, the mean of the local dimensions predicts when the model's training capabilities are exhausted, as exemplified in a dialogue state tracking task, overfitting, as demonstrated in an emotion recognition task, and grokking, as illustrated with an arithmetic task. Furthermore, our experiments suggest a practical heuristic: reductions in the mean local dimension tend to accompany and predict subsequent performance gains. Through this exploration, we aim to provide practitioners with a deeper understanding of the implications of fine-tuning on embedding spaces, facilitating informed decisions when configuring models for specific applications. The results of this work contribute to the ongoing discourse on the interpretability, adaptability, and generalizability of LLMs by bridging the gap between intrinsic model mechanisms and geometric properties in the respective embeddings. 

**Abstract (ZH)**: 理解大型语言模型（LLMs）的内部机制仍然是一个具有挑战性和复杂性的任务。即使是最基本的问题，如微调如何影响模型行为，也往往需要大量的经验性评估。在本文中，我们提出了一种基于上下文latent嵌入的几何属性的新视角，以研究训练和微调的影响。为此，我们测量了上下文语言模型latent空间的局部维度，并分析了它们在训练和微调过程中的变化。我们展示了局部维度提供了关于模型训练动力学和泛化能力的见解。具体而言，局部维度的均值预测了模型训练能力耗尽的时期，如在对话状态跟踪任务中所示；局部维度的过拟合，在情感识别任务中得到证实；局部维度的变化与适应性学习过程，如在算术任务中所示。此外，我们的实验表明一个实用的方法：局部维度均值的减少往往伴随并预测后续性能的提高。通过这次探索，我们旨在为实践者提供对微调对嵌入空间影响的更深入理解，有助于在为特定应用配置模型时做出明智的决策。这些结果为大型语言模型（LLMs）的可解释性、适应性和泛化能力的持续讨论做出了贡献，填补了内在模型机制与相应嵌入的几何属性之间的空白。 

---
# NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction 

**Title (ZH)**: NTPP：基于下一个词对预测的双通道语音对话生成语言模型 

**Authors**: Qichao Wang, Ziqiao Meng, Wenqian Cui, Yifei Zhang, Pengcheng Wu, Bingzhe Wu, Irwin King, Liang Chen, Peilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00975)  

**Abstract**: Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications. 

**Abstract (ZH)**: 受到GPT-4o卓越能力的启发，人们对使语音语言模型（SLMs）能够与人类进行自然流畅的对话产生了 growing 兴趣。近期进展促使开发出了几种在这一领域表现令人鼓舞的 SLMs。然而，当前的方法尚未充分利用固有的双通道语音数据，这种数据能够捕捉人类对话的结构和动态。在此项工作中，我们系统地探讨了在现代大语言模型背景下使用双通道语音数据的方法，并引入了一种新颖的生成建模范式——下一令牌对预测（NTPP），首次使用解码器仅架构实现了独立说话人双通道语音对话学习。我们在标准基准上评估了我们的方法，实证结果表明，我们提出的方法 NTPP 显著提升了 SLMs 的对话能力，特别是在回合转换预测、响应连贯性和自然度方面。此外，与现有方法相比，NTPP 实现了显著更低的推理延迟，突显了其在实时应用中的实用性。 

---
# Legal Compliance Evaluation of Smart Contracts Generated By Large Language Models 

**Title (ZH)**: 由大型语言模型生成的智能合约的法律合规性评估 

**Authors**: Chanuka Wijayakoon, Hai Dong, H.M.N. Dilum Bandara, Zahir Tari, Anurag Soin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00943)  

**Abstract**: Smart contracts can implement and automate parts of legal contracts, but ensuring their legal compliance remains challenging. Existing approaches such as formal specification, verification, and model-based development require expertise in both legal and software development domains, as well as extensive manual effort. Given the recent advances of Large Language Models (LLMs) in code generation, we investigate their ability to generate legally compliant smart contracts directly from natural language legal contracts, addressing these challenges. We propose a novel suite of metrics to quantify legal compliance based on modeling both legal and smart contracts as processes and comparing their behaviors. We select four LLMs, generate 20 smart contracts based on five legal contracts, and analyze their legal compliance. We find that while all LLMs generate syntactically correct code, there is significant variance in their legal compliance with larger models generally showing higher levels of compliance. We also evaluate the proposed metrics against properties of software metrics, showing they provide fine-grained distinctions, enable nuanced comparisons, and are applicable across domains for code from any source, LLM or developer. Our results suggest that LLMs can assist in generating starter code for legally compliant smart contracts with strict reviews, and the proposed metrics provide a foundation for automated and self-improving development workflows. 

**Abstract (ZH)**: 基于大型语言模型生成合法合规智能合约的研究 

---
# Bridging Subjective and Objective QoE: Operator-Level Aggregation Using LLM-Based Comment Analysis and Network MOS Comparison 

**Title (ZH)**: 基于LLM的评论分析与网络MOS比较的主观与客观QoE桥梁构建：运营商级聚合 

**Authors**: Parsa Hassani Shariat Panahi, Amir Hossein Jalilvand, M. Hasan Najafi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00924)  

**Abstract**: This paper introduces a dual-layer framework for network operator-side quality of experience (QoE) assessment that integrates both objective network modeling and subjective user perception extracted from live-streaming platforms. On the objective side, we develop a machine learning model trained on mean opinion scores (MOS) computed via the ITU-T P.1203 reference implementation, allowing accurate prediction of user-perceived video quality using only network parameters such as packet loss, delay, jitter, and throughput without reliance on video content or client-side instrumentation. On the subjective side, we present a semantic filtering and scoring pipeline that processes user comments from live streams to extract performance-related feedback. A large language model is used to assign scalar MOS scores to filtered comments in a deterministic and reproducible manner. To support scalable and interpretable analysis, we con- struct a labeled dataset of 47,894 live-stream comments, of which about 34,000 are identified as QoE-relevant through multi-layer semantic filtering. Each comment is enriched with simulated Internet Service Provider attribution and temporally aligned using synthetic timestamps in 5-min intervals. The resulting dataset enables operator-level aggregation and time-series analysis of user-perceived quality. A delta MOS metric is proposed to measure each Internet service provider's deviation from platform-wide sentiment, allowing detection of localized degradations even in the absence of direct network telemetry. A controlled outage simulation confirms the framework's effectiveness in identifying service disruptions through comment-based trends alone. The system provides each operator with its own subjective MOS and the global platform average per interval, enabling real-time interpretation of performance deviations and comparison with objective network-based QoE estimates. 

**Abstract (ZH)**: 一种结合客观网络建模和主观用户感知的双层框架，用于网络运营商侧质量体验评估 

---
# How do Transformer Embeddings Represent Compositions? A Functional Analysis 

**Title (ZH)**: 变压器嵌入如何表示组成？一种功能分析 

**Authors**: Aishik Nagar, Ishaan Singh Rawal, Mansi Dhanania, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00914)  

**Abstract**: Compositionality is a key aspect of human intelligence, essential for reasoning and generalization. While transformer-based models have become the de facto standard for many language modeling tasks, little is known about how they represent compound words, and whether these representations are compositional. In this study, we test compositionality in Mistral, OpenAI Large, and Google embedding models, and compare them with BERT. First, we evaluate compositionality in the representations by examining six diverse models of compositionality (addition, multiplication, dilation, regression, etc.). We find that ridge regression, albeit linear, best accounts for compositionality. Surprisingly, we find that the classic vector addition model performs almost as well as any other model. Next, we verify that most embedding models are highly compositional, while BERT shows much poorer compositionality. We verify and visualize our findings with a synthetic dataset consisting of fully transparent adjective-noun compositions. Overall, we present a thorough investigation of compositionality. 

**Abstract (ZH)**: 组成性是人类智能的关键方面，对于推理和泛化至关重要。尽管基于变换器的模型已成为许多语言建模任务的事实标准，但对其如何表示复合词以及这些表示是否具有组成性的了解甚少。在本研究中，我们测试了 Mistral、OpenAI Large、Google 嵌入模型以及 BERT 在组成性方面的表现，并对其进行了比较。首先，我们通过检查六种不同的组成性模型（加法、乘法、扩张、回归等）来评估组成性在表示中的体现。我们发现，尽管岭回归是线性的，但它最好地解释了组成性。令人惊讶的是，我们发现经典的向量加法模型几乎与其他任何模型一样有效。接下来，我们验证大多数嵌入模型具有高度的组成性，而 BERT 的组成性却表现较差。我们通过一个由完全透明的形容词-名词组合构成的合成数据集验证并可视化了这些发现。总体而言，我们进行了全面的组成性研究。 

---
# CODEMENV: Benchmarking Large Language Models on Code Migration 

**Title (ZH)**: CODEMENV: 代码迁移任务上大语言模型的基准测试 

**Authors**: Keyuan Cheng, Xudong Shen, Yihao Yang, Tengyue Wang, Yang Cao, Muhammad Asif Ali, Hanbin Wang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00894)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities across various software engineering tasks; however, their effectiveness in code migration, adapting code to run in different environments, remains insufficiently studied. In this work, we introduce CODEMENV: Code Migration Across Environment, a new benchmark specifically designed to assess LLMs' abilities in code migration scenarios. CODEMENV consists of 922 examples spanning 19 Python and Java packages, and covers three core tasks: (1) identifying functions incompatible with specific versions, (2) detecting changes in function definitions, and (3) adapting code to target environments. Experimental evaluation with seven LLMs on CODEMENV yields an average pass@1 rate of 26.50%, with GPT-4O achieving the highest score at 43.84%. Key findings include: (i) LLMs tend to be more proficient with newer function versions, which aids in migrating legacy code, and (ii) LLMs sometimes exhibit logical inconsistencies by identifying function changes irrelevant to the intended migration environment. The datasets are available at this https URL. 

**Abstract (ZH)**: 大型语言模型在代码迁移任务中的能力已显示出显著潜力；然而，它们在适应不同环境运行代码方面的效果仍需进一步研究。在本工作中，我们引入了CODEMENV：跨环境代码迁移基准，专门用于评估大型语言模型在代码迁移场景中的能力。CODEMENV包含922个示例，涵盖了19个Python和Java包，并包含三个核心任务：(1)识别与特定版本不兼容的函数，(2)检测函数定义的变化，以及(3)将代码适应目标环境。在CODEMENV上使用七种大型语言模型进行实验评估，平均pass@1得分为26.50%，GPT-4O取得最高分43.84%。主要发现包括：(i) 大型语言模型在较新的函数版本上表现得更为熟练，这有助于迁移遗留代码；(ii) 大型语言模型有时会因识别与预期迁移环境无关的函数变化而表现出逻辑不一致。数据集可在此处获取。 

---
# Affordance Benchmark for MLLMs 

**Title (ZH)**: MLLMs的可用性基准 

**Authors**: Junying Wang, Wenzhe Li, Yalun Wu, Yingji Liang, Yijin Guo, Chunyi Li, Haodong Duan, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00893)  

**Abstract**: Affordance theory posits that environments inherently offer action possibilities that shape perception and behavior. While Multimodal Large Language Models (MLLMs) excel in vision-language tasks, their ability to perceive affordance, which is crucial for intuitive and safe interactions, remains underexplored. To address this, we introduce A4Bench, a novel benchmark designed to evaluate the affordance perception abilities of MLLMs across two dimensions: 1) Constitutive Affordance}, assessing understanding of inherent object properties through 1,282 question-answer pairs spanning nine sub-disciplines, and 2) Transformative Affordance, probing dynamic and contextual nuances (e.g., misleading, time-dependent, cultural, or individual-specific affordance) with 718 challenging question-answer pairs. Evaluating 17 MLLMs (nine proprietary and eight open-source) against human performance, we find that proprietary models generally outperform open-source counterparts, but all exhibit limited capabilities, particularly in transformative affordance perception. Furthermore, even top-performing models, such as Gemini-2.0-Pro (18.05% overall exact match accuracy), significantly lag behind human performance (best: 85.34%, worst: 81.25%). These findings highlight critical gaps in environmental understanding of MLLMs and provide a foundation for advancing AI systems toward more robust, context-aware interactions. The dataset is available in this https URL. 

**Abstract (ZH)**: 赋能理论认为，环境本身提供了影响知觉和行为的动作可能性。尽管多模态大型语言模型（MLLMs）在视觉语言任务上表现出色，但它们识别赋能的能力，这对于直观和安全的交互至关重要，仍待探索。为了解决这一问题，我们引入了A4Bench，这是一个旨在从两个维度评估MLLMs的赋能感知能力的新基准：1）构成性赋能，通过涵盖九个子学科的1,282个问题-答案对，评估对固有物体属性的理解；2）变换性赋能，通过718个具有挑战性的问题-答案对，探究动态和上下文细微差别（如误导性、时间依赖性、文化背景或个体特异性赋能）。在对17个MLLMs（九个专有和八个开源模型）进行与人类表现的比较评估中，我们发现专有模型通常优于开源模型，但所有模型在变换性赋能感知方面表现有限。此外，即使是表现最佳的模型，如Gemini-2.0-Pro（总体完全匹配准确率为18.05%），与人类表现相比（最佳为85.34%，最差为81.25%）仍有明显差距。这些发现突显了MLLMs在环境理解方面的关键差距，并为其向更具鲁棒性和上下文意识的交互发展的基础奠定基础。数据集可通过此链接获取。 

---
# ModuLM: Enabling Modular and Multimodal Molecular Relational Learning with Large Language Models 

**Title (ZH)**: ModuLM：通过大规模语言模型实现模块化和多模态分子关系学习 

**Authors**: Zhuo Chen, Yizhen Zheng, Huan Yee Koh, Hongxin Xiang, Linjiang Chen, Wenjie Du, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00880)  

**Abstract**: Molecular Relational Learning (MRL) aims to understand interactions between molecular pairs, playing a critical role in advancing biochemical research. With the recent development of large language models (LLMs), a growing number of studies have explored the integration of MRL with LLMs and achieved promising results. However, the increasing availability of diverse LLMs and molecular structure encoders has significantly expanded the model space, presenting major challenges for benchmarking. Currently, there is no LLM framework that supports both flexible molecular input formats and dynamic architectural switching. To address these challenges, reduce redundant coding, and ensure fair model comparison, we propose ModuLM, a framework designed to support flexible LLM-based model construction and diverse molecular representations. ModuLM provides a rich suite of modular components, including 8 types of 2D molecular graph encoders, 11 types of 3D molecular conformation encoders, 7 types of interaction layers, and 7 mainstream LLM backbones. Owing to its highly flexible model assembly mechanism, ModuLM enables the dynamic construction of over 50,000 distinct model configurations. In addition, we provide comprehensive results to demonstrate the effectiveness of ModuLM in supporting LLM-based MRL tasks. 

**Abstract (ZH)**: 分子关系学习(MRL)旨在理解分子对之间的相互作用，对推动生物化学研究具有重要作用。随着大型语言模型(LLMs)的 recent 发展，越来越多的研究探索了 MRL 与 LLMs 的集成并取得了有前景的结果。然而，多样化 LLMs 和分子结构编码器的不断增加使得模型空间大幅扩展，带来了重要挑战。目前尚无支持灵活的分子输入格式和动态架构切换的 LLM 框架。为应对这些挑战，减少冗余编码，并确保公平的模型比较，我们提出了 ModuLM，一种支持灵活的基于 LLM 的模型构建和多种分子表示的框架。ModuLM 提供了一整套模块化组件，包括 8 种 2D 分子图形编码器、11 种 3D 分子构象编码器、7 种相互作用层和 7 种主流 LLM 主干。由于其高度灵活的模型组装机制，ModuLM 能够动态构建超过 50,000 种不同的模型配置。此外，我们提供了全面的结果来证明 ModuLM 在支持基于 LLM 的 MRL 任务方面的有效性。 

---
# EEG2TEXT-CN: An Exploratory Study of Open-Vocabulary Chinese Text-EEG Alignment via Large Language Model and Contrastive Learning on ChineseEEG 

**Title (ZH)**: EEG2TEXT-CN：基于大规模语言模型和对比学习的开放词汇中文文本-EEG对齐探索性研究 

**Authors**: Jacky Tai-Yu Lu, Jung Chiang, Chi-Sheng Chen, Anna Nai-Yun Tung, Hsiang Wei Hu, Yuan Chiao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00854)  

**Abstract**: We propose EEG2TEXT-CN, which, to the best of our knowledge, represents one of the earliest open-vocabulary EEG-to-text generation frameworks tailored for Chinese. Built on a biologically grounded EEG encoder (NICE-EEG) and a compact pretrained language model (MiniLM), our architecture aligns multichannel brain signals with natural language representations via masked pretraining and contrastive learning. Using a subset of the ChineseEEG dataset, where each sentence contains approximately ten Chinese characters aligned with 128-channel EEG recorded at 256 Hz, we segment EEG into per-character embeddings and predict full sentences in a zero-shot setting. The decoder is trained with teacher forcing and padding masks to accommodate variable-length sequences. Evaluation on over 1,500 training-validation sentences and 300 held-out test samples shows promising lexical alignment, with a best BLEU-1 score of 6.38\%. While syntactic fluency remains a challenge, our findings demonstrate the feasibility of non-phonetic, cross-modal language decoding from EEG. This work opens a new direction in multilingual brain-to-text research and lays the foundation for future cognitive-language interfaces in Chinese. 

**Abstract (ZH)**: EEG2TEXT-CN：一种适用于中文的早期开放词汇EEG到文本生成框架 

---
# Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience 

**Title (ZH)**: 面向结构化知识推理：经验增强的对比检索生成 

**Authors**: Jiawei Gu, Ziting Xian, Yuanzhen Xie, Ye Liu, Enjie Liu, Ruichao Zhong, Mochi Gao, Yunzhi Tan, Bo Hu, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00842)  

**Abstract**: Large language models (LLMs) achieve strong performance on plain text tasks but underperform on structured data like tables and databases. Potential challenges arise from their underexposure during pre-training and rigid text-to-structure transfer mechanisms. Unlike humans who seamlessly apply learned patterns across data modalities, LLMs struggle to infer implicit relationships embedded in tabular formats, especially in the absence of explicit structural guidance. To bridge this cognitive gap, we introduce Contrastive Retrieval-Augmented Generation on Experience (CoRE), a framework that builds experience memory representations and enhances generalization through contrastive In-Context Learning (ICL) to simulate human-like knowledge transfer. Experiments on Text-to-SQL and TableQA show CoRE significantly improves performance, achieving average gains of 3.44% and 4.24%, with up to 17.2% on challenging tasks. Our Monte Carlo Tree Search (MCTS)-generated Experience Memory expands training data 8-9x, enhancing diversity and domain coverage. This training-free and continual method propels LLMs toward structured knowledge expertise. 

**Abstract (ZH)**: 大型语言模型在文本任务上表现强劲，但在处理表格和数据库等结构化数据时表现不佳。对比检索增强生成经验（CoRE）框架通过构建经验记忆表示并通过对比上下文学习（ICL）增强泛化能力，模拟人类的知识迁移，以弥补这一认知差距。在Text-to-SQL和TableQA任务上的实验表明，CoRE显著提高了性能，平均分别提升了3.44%和4.24%，在挑战性任务上的提升高达17.2%。通过蒙特卡洛树搜索（MCTS）生成的经验记忆将训练数据扩展8-9倍，增强了多样性和领域覆盖范围。这种无需训练且持续的方法促使大型语言模型向结构化知识专门性迈进。 

---
# A Large Language Model-Supported Threat Modeling Framework for Transportation Cyber-Physical Systems 

**Title (ZH)**: 一种由大型语言模型支持的交通 cyber-物理系统威胁建模框架 

**Authors**: M Sabbir Salek, Mashrur Chowdhury, Muhaimin Bin Munir, Yuchen Cai, Mohammad Imtiaz Hasan, Jean-Michel Tine, Latifur Khan, Mizanur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2506.00831)  

**Abstract**: Modern transportation systems rely on cyber-physical systems (CPS), where cyber systems interact seamlessly with physical systems like transportation-related sensors and actuators to enhance safety, mobility, and energy efficiency. However, growing automation and connectivity increase exposure to cyber vulnerabilities. Existing threat modeling frameworks for transportation CPS are often limited in scope, resource-intensive, and dependent on significant cybersecurity expertise. To address these gaps, we present TraCR-TMF (Transportation Cybersecurity and Resiliency Threat Modeling Framework), a large language model (LLM)-based framework that minimizes expert intervention. TraCR-TMF identifies threats, potential attack techniques, and corresponding countermeasures by leveraging the MITRE ATT&CK matrix through three LLM-based approaches: (i) a retrieval-augmented generation (RAG) method requiring no expert input, (ii) an in-context learning approach requiring low expert input, and (iii) a supervised fine-tuning method requiring moderate expert input. TraCR-TMF also maps attack paths to critical assets by analyzing vulnerabilities using a customized LLM. The framework was evaluated in two scenarios. First, it identified relevant attack techniques across transportation CPS applications, with 90% precision as validated by experts. Second, using a fine-tuned LLM, it successfully predicted multiple exploitations including lateral movement, data exfiltration, and ransomware-related encryption that occurred during a major real-world cyberattack incident. These results demonstrate TraCR-TMF's effectiveness in CPS threat modeling, its reduced reliance on cybersecurity expertise, and its adaptability across CPS domains. 

**Abstract (ZH)**: 基于大型语言模型的交通运输CPS威胁建模框架：TraCR-TMF 

---
# COMPKE: Complex Question Answering under Knowledge Editing 

**Title (ZH)**: COMPKE: 知识编辑下的复杂问题求解 

**Authors**: Keyuan Cheng, Zijian Kan, Zhixian He, Zhuoran Zhang, Muhammad Asif Ali, Ke Xu, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00829)  

**Abstract**: Knowledge Editing, which efficiently modifies the knowledge in large language models, has gathered great attention. Current benchmarks primarily use multi-hop question answering to assess and analyze newly injected or updated knowledge. However, we argue that these benchmarks fail to effectively evaluate how well the updated models apply this knowledge in real-life scenarios, particularly when questions require complex reasoning, involving one-to-many relationships or multi-step logical intersections. To fill in this gap, we introduce a new benchmark, COMPKE: Complex Question Answering under Knowledge Editing, which includes 11,924 complex questions that reflect real-life situations. We conduct an extensive evaluation of four knowledge editing methods on COMPKE, revealing that their effectiveness varies notably across different models. For instance, MeLLo attains an accuracy of 39.47 on GPT-4O-MINI, but this drops sharply to 3.83 on QWEN2.5-3B. We further investigate the underlying causes of these disparities from both methodological and model-specific perspectives. The datasets are available at this https URL. 

**Abstract (ZH)**: 知识编辑：复杂问答下的知识编辑基准（COMPKE） 

---
# Behavioral Augmentation of UML Class Diagrams: An Empirical Study of Large Language Models for Method Generation 

**Title (ZH)**: 基于大型语言模型的方法生成对UML类图的行为增强：一项实证研究 

**Authors**: Djaber Rouabhia, Ismail Hadjadj  

**Link**: [PDF](https://arxiv.org/pdf/2506.00788)  

**Abstract**: Automating the enrichment of UML class diagrams with behavioral methods from natural language use cases is a significant challenge. This study evaluates nine large language models (LLMs) in augmenting a methodless UML diagram (21 classes, 17 relationships) using 21 structured waste-management use cases. A total of 90 diagrams (3,373 methods) were assessed across six metrics: method quantity, signature richness (visibility, names, parameters, return types), annotation completeness (linking to use cases/actions), structural fidelity, syntactic correctness (PlantUML compilation), and naming convergence (across models). All LLMs produced valid PlantUML diagrams adhering to UML conventions. Some models excelled in method coverage and annotation accuracy, while others showed richer parameterization but weaker traceability. These results demonstrate that LLMs can generate well-structured methods with consistent naming, advancing automated behavioral modeling. However, inconsistencies in annotations and signatures highlight the need for improved prompt engineering and model selection. The rapid generation of these methods supports Agile practices by enabling faster design iterations. Despite their capabilities, human oversight is essential to ensure accuracy, appropriateness, and semantic alignment. This positions LLMs as collaborative partners in software design. All experimental artifacts (\texttt{.puml}, \texttt{.png}, \texttt{.csv}) are publicly available for reproducibility. 

**Abstract (ZH)**: 使用自然语言用例行为方法自动丰富UML类图是一个显著挑战。本研究评估了九个大型语言模型（LLMs）在使用21个结构化废弃物管理用案例信息增强一个无方法的UML图（21个类，17个关系）方面的效果。总共评估了90个图（3,373个方法），采用了六项指标：方法数量、签名丰富度（可见性、名称、参数、返回类型）、注释完整性（链接到用例/动作）、结构忠实度、句法正确性（PlantUML编译）以及命名一致性（跨模型）。所有LLM均生成了符合UML规范的有效PlantUML图。一些模型在方法覆盖率和注释准确性方面表现出色，而其他模型则在参数化方面更加丰富但跟踪能力较弱。这些结果表明，LLM能够生成结构良好且命名一致的方法，从而推进自动行为建模。然而，注释和签名的一致性问题凸显了改进提示工程和模型选择的必要性。这些方法的快速生成支持敏捷实践，通过加快设计迭代。尽管LLM具有这些能力，但人类监督仍然是确保准确性、适当性和语义一致性的关键。这将LLM定位为软件设计的合作伙伴。所有实验 artefact（.puml，.png，.csv）均公开可用，以确保可重复性。 

---
# KG-TRACES: Enhancing Large Language Models with Knowledge Graph-constrained Trajectory Reasoning and Attribution Supervision 

**Title (ZH)**: KG-TRACES：基于知识图谱约束轨迹推理和归因监督的大语言模型增强方法 

**Authors**: Rong Wu, Pinlong Cai, Jianbiao Mei, Licheng Wen, Tao Hu, Xuemeng Yang, Daocheng Fu, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00783)  

**Abstract**: Large language models (LLMs) have made remarkable strides in various natural language processing tasks, but their performance on complex reasoning problems remains hindered by a lack of explainability and trustworthiness. This issue, often manifesting as hallucinations or unattributable reasoning processes, limits their applicability in complex reasoning scenarios. To address this, we propose Knowledge Graph-constrained Trajectory Reasoning Attribution and Chain Explanation Supervision (KG-TRACES), a novel framework that enhances the reasoning ability of LLMs through explicit supervision over reasoning paths and processes. KG-TRACES jointly supervises the model to: (1) predict symbolic relation paths, (2) predict full triple-level reasoning paths, and (3) generate attribution-aware reasoning processes grounded in the reasoning paths. At inference phase, the model adapts to both KG-available and KG-unavailable scenarios, retrieving reasoning paths from a KG when possible or predicting plausible reasoning paths with only intrinsic knowledge when not. This design enables the model to reason in an explainable and source-attributable pattern. Through extensive experiments on complex reasoning tasks, we demonstrate that KG-TRACES significantly outperforms existing SOTA: it improves Hits@1 by 1.6% and F1 by 4.7% on WebQSP, and achieves improvements of 4.8% in Hits@1 and 2.1% in F1 on CWQ. Moreover, we show its transferability to specialized domains such as medicine. By visualizing the intermediate steps of reasoning processes, we further show that the explicit supervision introduced by KG-TRACES leads to more stable and goal-directed reasoning processes, aligning closely with correct answers. Code is available at this https URL. 

**Abstract (ZH)**: Knowledge Graph-constrained Trajectory Reasoning Attribution and Chain Explanation Supervision for Enhancing Explainability and Trustworthiness in Large Language Models 

---
# LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning 

**Title (ZH)**: 揭开真相的面纱：在基于推理的监督微调中，秩减少后主权重浮现 

**Authors**: Zihang Liu, Tianyu Pang, Oleg Balabanov, Chaoqun Yang, Tianjin Huang, Lu Yin, Yaoqing Yang, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00772)  

**Abstract**: Recent studies have shown that supervised fine-tuning of LLMs on a small number of high-quality datasets can yield strong reasoning capabilities. However, full fine-tuning (Full FT), while powerful, is computationally expensive and susceptible to overfitting and catastrophic forgetting, particularly when data is limited. Sparse fine-tuning, which previously achieved notable success by updating only a small subset of model parameters, offers a promising trade-off between efficiency and effectiveness. Yet, it has lagged behind in the LLM era due to the difficulty of identifying parameters truly critical for reasoning. In this work, we state that weights with the largest magnitude after low-rank approximation are critical weights for fine-tuning, which we call Principal Weights. Surprisingly, while magnitude-based sparse fine-tuning performs poorly as a baseline on LLM fine-tuning, it becomes highly effective after rank reduction. These insights motivate our method: Low-rank Informed Sparse Fine-Tuning (LIFT). LIFT only updates the top 5% Principal Weights throughout training and consistently achieves better performance on reasoning tasks than Full FT, while maintaining memory efficiency on par with popular parameter-efficient fine-tuning methods. In addition to strong performance on target domains such as arithmetic reasoning, LIFT also retains up to 20% more source-domain knowledge, compared to Full FT and LoRA. Our code is available at: this https URL. 

**Abstract (ZH)**: Recent Studies Show that Supervised Fine-Tuning of LLMs on Small High-Quality Datasets Can Yield Strong Reasoning Capabilities: Sparse Fine-Tuning with Principal Weights Offers a Promising Trade-Off Between Efficiency and Effectiveness 

---
# "Who experiences large model decay and why?" A Hierarchical Framework for Diagnosing Heterogeneous Performance Drift 

**Title (ZH)**: “谁会经历大规模模型衰减，以及为什么？”一种诊断异质性性能漂移的层级框架 

**Authors**: Harvineet Singh, Fan Xia, Alexej Gossmann, Andrew Chuang, Julian C. Hong, Jean Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00756)  

**Abstract**: Machine learning (ML) models frequently experience performance degradation when deployed in new contexts. Such degradation is rarely uniform: some subgroups may suffer large performance decay while others may not. Understanding where and how large differences in performance arise is critical for designing targeted corrective actions that mitigate decay for the most affected subgroups while minimizing any unintended effects. Current approaches do not provide such detailed insight, as they either (i) explain how average performance shifts arise or (ii) identify adversely affected subgroups without insight into how this occurred. To this end, we introduce a Subgroup-scanning Hierarchical Inference Framework for performance drifT (SHIFT). SHIFT first asks "Is there any subgroup with unacceptably large performance decay due to covariate/outcome shifts?" (Where?) and, if so, dives deeper to ask "Can we explain this using more detailed variable(subset)-specific shifts?" (How?). In real-world experiments, we find that SHIFT identifies interpretable subgroups affected by performance decay, and suggests targeted actions that effectively mitigate the decay. 

**Abstract (ZH)**: 一种用于性能漂移的亚群体层级推理框架（SHIFT） 

---
# CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning 

**Title (ZH)**: 代码感知：代码语义推理的现实世界基准和数据集 

**Authors**: Monoshi Kumar Roy, Simin Chen, Benjamin Steenhoek, Jinjun Peng, Gail Kaiser, Baishakhi Ray, Wei Le  

**Link**: [PDF](https://arxiv.org/pdf/2506.00750)  

**Abstract**: Understanding and reasoning about code semantics is essential for enhancing code LLMs' abilities to solve real-world software engineering (SE) tasks. Although several code reasoning benchmarks exist, most rely on synthetic datasets or educational coding problems and focus on coarse-grained reasoning tasks such as input/output prediction, limiting their effectiveness in evaluating LLMs in practical SE contexts. To bridge this gap, we propose CodeSense, the first benchmark that makes available a spectrum of fine-grained code reasoning tasks concerned with the software engineering of real-world code. We collected Python, C and Java software projects from real-world repositories. We executed tests from these repositories, collected their execution traces, and constructed a ground truth dataset for fine-grained semantic reasoning tasks. We then performed comprehensive evaluations on state-of-the-art LLMs. Our results show a clear performance gap for the models to handle fine-grained reasoning tasks. Although prompting techniques such as chain-of-thought and in-context learning helped, the lack of code semantics in LLMs fundamentally limit models' capabilities of code reasoning. Besides dataset, benchmark and evaluation, our work produced an execution tracing framework and tool set that make it easy to collect ground truth for fine-grained SE reasoning tasks, offering a strong basis for future benchmark construction and model post training. Our code and data are located at this https URL. 

**Abstract (ZH)**: 理解并推理代码语义对于提升代码LLMs解决实际软件工程任务的能力至关重要。虽然已存在多种代码推理基准，但大多数依赖合成数据集或教育编程问题，并主要关注输入/输出预测等粗粒度推理任务，限制了其在实际软件工程情境中评估LLMs的效果。为了弥合这一差距，我们提出了CodeSense，这是首个提供关注实际代码软件工程的一系列细粒度代码推理任务的基准。我们收集了真实的Python、C和Java软件项目。我们从这些仓库执行测试，收集其执行轨迹，并构建了用于细粒度语义推理任务的地面真实数据集。然后，我们对最先进的LLMs进行了全面评估。结果显示，模型在处理细粒度推理任务时存在明显的性能差距。尽管chain-of-thought和基于上下文的学习等提示技术有所帮助，但LLMs中缺乏代码语义从根本上限制了其代码推理能力。除了数据集、基准和评估，我们的工作还创建了一个执行跟踪框架和工具集，使其易于收集细粒度软件工程推理任务的地面真实值，为未来的基准构建和模型后续训练提供了坚实基础。我们的代码和数据位于此链接：https://github.com/alibaba/CodSense。 

---
# Assortment of Attention Heads: Accelerating Federated PEFT with Head Pruning and Strategic Client Selection 

**Title (ZH)**: 注意力头的优化组合：通过头剪枝和战略客户端选择加速联邦PEFT 

**Authors**: Yeshwanth Venkatesha, Souvik Kundu, Priyadarshini Panda  

**Link**: [PDF](https://arxiv.org/pdf/2506.00743)  

**Abstract**: Parameter Efficient Fine-Tuning (PEFT) has become the de-facto approach in adapting Large Language Models (LLMs) for downstream tasks in Natural Language Processing. However, its adoption in privacy-preserving distributed learning frameworks, such as Federated Learning (FL), remains relatively limited. This is mainly due to challenges specific to FL, such as resource-constrained devices and diverse data distributions among clients. In this paper, we propose an efficient method to perform PEFT within the FL framework for Multi-Head Attention (MHA) based language models. We address the challenges through head pruning, a novel head-specific weighted aggregation mechanism, and a client selection strategy. Head pruning minimizes training complexity within the clients, guided by the importance score computed based on the confidence of the attention head. Weighted aggregation of heads ensures the global model captures crucial updates from diverse clients complementing our client selection strategy. We show results on the MultiNLI benchmark along with 20 Newsgroups, XL-Sum, and E2E NLG datasets. We use the MultiNLI dataset and T5-small model with LoRA as our PEFT method, attaining sparsity levels of up to 90%, resulting in a communication advantage of up to 1.8x and a reduction in training OPs of 3.9x while maintaining the accuracy drop under 2%. 

**Abstract (ZH)**: Parameter Efficient Fine-Tuning within Federated Learning for Multi-Head Attention Based Language Models 

---
# Pitfalls in Evaluating Language Model Forecasters 

**Title (ZH)**: 语言模型预测器评估中的陷阱 

**Authors**: Daniel Paleka, Shashwat Goel, Jonas Geiping, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2506.00723)  

**Abstract**: Large language models (LLMs) have recently been applied to forecasting tasks, with some works claiming these systems match or exceed human performance. In this paper, we argue that, as a community, we should be careful about such conclusions as evaluating LLM forecasters presents unique challenges. We identify two broad categories of issues: (1) difficulty in trusting evaluation results due to many forms of temporal leakage, and (2) difficulty in extrapolating from evaluation performance to real-world forecasting. Through systematic analysis and concrete examples from prior work, we demonstrate how evaluation flaws can raise concerns about current and future performance claims. We argue that more rigorous evaluation methodologies are needed to confidently assess the forecasting abilities of LLMs. 

**Abstract (ZH)**: 大规模语言模型(LLMs) recently已被应用于预测任务，一些研究表明这些系统能够达到或超过人类的性能。在本文中，我们认为作为科研社区，我们应该谨慎对待这些结论，因为评估LLM预测器存在独特挑战。我们识别出两类主要问题：(1) 由于存在多种形式的时间泄漏，难以信任评估结果；(2) 从评估性能推断到实际预测的困难。通过系统分析和先前工作中的具体示例，我们展示了评估缺陷如何引发对未来和当前预测能力声明的担忧。我们认为，需要更严谨的评估方法来确信地评估LLM的预测能力。 

---
# An LLM Agent for Functional Bug Detection in Network Protocols 

**Title (ZH)**: 功能性 bug 检测在网络协议中的 LLAM 代理 

**Authors**: Mingwei Zheng, Chengpeng Wang, Xuwei Liu, Jinyao Guo, Shiwei Feng, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00714)  

**Abstract**: Functional correctness is critical for ensuring the reliability and security of network protocol implementations. Functional bugs, instances where implementations diverge from behaviors specified in RFC documents, can lead to severe consequences, including faulty routing, authentication bypasses, and service disruptions. Detecting these bugs requires deep semantic analysis across specification documents and source code, a task beyond the capabilities of traditional static analysis tools. This paper introduces RFCScan, an autonomous agent that leverages large language models (LLMs) to detect functional bugs by checking conformance between network protocol implementations and their RFC specifications. Inspired by the human auditing procedure, RFCScan comprises two key components: an indexing agent and a detection agent. The former hierarchically summarizes protocol code semantics, generating semantic indexes that enable the detection agent to narrow down the scanning scope. The latter employs demand-driven retrieval to iteratively collect additional relevant data structures and functions, eventually identifying potential inconsistencies with the RFC specifications effectively. We evaluate RFCScan across six real-world network protocol implementations. RFCScan identifies 47 functional bugs with 81.9% precision, of which 20 bugs have been confirmed or fixed by developers. 

**Abstract (ZH)**: 基于RFC规范的功能正确性对于确保网络协议实现的可靠性和安全性至关重要。功能漏洞会导致严重的后果，包括误路由、身份认证绕过和服务中断。检测这些漏洞需要对规范文档和源代码进行深入语义分析，这是传统静态分析工具无法完成的任务。本文介绍了RFCScan，这是一种自主代理，利用大规模语言模型（LLMs）通过检查网络协议实现与RFC规范的一致性来检测功能漏洞。RFCScan由两个关键组件组成：索引代理和检测代理。索引代理层次结构地总结协议代码语义，生成语义索引，使检测代理能够缩小扫描范围。检测代理采用需求驱动检索，迭代收集相关数据结构和函数，最终有效地识别潜在的与RFC规范的一致性问题。我们在六个实际网络协议实现中评估了RFCScan。RFCScan发现了47个功能漏洞，精确率为81.9%，其中20个漏洞已被开发者确认或修复。 

---
# Measuring Faithfulness and Abstention: An Automated Pipeline for Evaluating LLM-Generated 3-ply Case-Based Legal Arguments 

**Title (ZH)**: 测量忠实度与弃权：评估大语言模型生成的三元案例法法律论点的自动化管道 

**Authors**: Li Zhang, Morgan Gray, Jaromir Savelka, Kevin D. Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2506.00694)  

**Abstract**: Large Language Models (LLMs) demonstrate potential in complex legal tasks like argument generation, yet their reliability remains a concern. Building upon pilot work assessing LLM generation of 3-ply legal arguments using human evaluation, this paper introduces an automated pipeline to evaluate LLM performance on this task, specifically focusing on faithfulness (absence of hallucination), factor utilization, and appropriate abstention. We define hallucination as the generation of factors not present in the input case materials and abstention as the model's ability to refrain from generating arguments when instructed and no factual basis exists. Our automated method employs an external LLM to extract factors from generated arguments and compares them against the ground-truth factors provided in the input case triples (current case and two precedent cases). We evaluated eight distinct LLMs on three tests of increasing difficulty: 1) generating a standard 3-ply argument, 2) generating an argument with swapped precedent roles, and 3) recognizing the impossibility of argument generation due to lack of shared factors and abstaining. Our findings indicate that while current LLMs achieve high accuracy (over 90%) in avoiding hallucination on viable argument generation tests (Tests 1 & 2), they often fail to utilize the full set of relevant factors present in the cases. Critically, on the abstention test (Test 3), most models failed to follow instructions to stop, instead generating spurious arguments despite the lack of common factors. This automated pipeline provides a scalable method for assessing these crucial LLM behaviors, highlighting the need for improvements in factor utilization and robust abstention capabilities before reliable deployment in legal settings. Project page: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂法律任务如论点生成中展现出潜力，但其可靠性仍是一个关注点。基于初步研究评估LLM生成三段式法律论点的人类评估结果，本文提出了一种自动评估管道，用于评估LLM在该任务上的性能，特别是重点在于忠实性（无幻觉）、因素利用以及适当的回避。我们将幻觉定义为生成输入案例材料中未出现的因素，回避定义为在没有事实依据的情况下模型停止生成论点的能力。我们的自动化方法使用外部LLM从生成的论点中提取因素，并将其与输入案例三元组（当前案例和两个先例案例）中提供的真实因素进行比较。我们对八种不同的LLM进行了三项难度递增的测试评估：1）生成标准三段式论点，2）生成角色互换的论点，3）识别人因缺乏共享因素而不能生成论点并回避。我们的研究结果显示，当前的LLM在有效的论点生成测试（测试1和测试2）中避免幻觉的准确性超过90%，但在利用相关因素方面常常未能充分利用。特别是在回避测试（测试3）中，大多数模型未能遵循停止生成论点的指示，而是生成了无效的论点，尽管缺乏共同因素。该自动化管道提供了一种可扩展的方法来评估这些关键的LLM行为，突显了在法律环境中可靠部署之前需要改进因素利用能力和坚定的回避能力。项目页面：[this https URL]。 

---
# Existing Large Language Model Unlearning Evaluations Are Inconclusive 

**Title (ZH)**: 现有大规模语言模型遗忘评估结果不具决定性 

**Authors**: Zhili Feng, Yixuan Even Xu, Alexander Robey, Robert Kirk, Xander Davies, Yarin Gal, Avi Schwarzschild, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2506.00688)  

**Abstract**: Machine unlearning aims to remove sensitive or undesired data from large language models. However, recent studies suggest that unlearning is often shallow, claiming that removed knowledge can easily be recovered. In this work, we critically examine standard unlearning evaluation practices and uncover key limitations that shake our trust in those findings. First, we show that some evaluations introduce substantial new information into the model, potentially masking true unlearning performance by re-teaching the model during testing. Second, we demonstrate that evaluation outcomes vary significantly across tasks, undermining the generalizability of current evaluation routines. Finally, we find that many evaluations rely on spurious correlations, making their results difficult to trust and interpret. Taken together, these issues suggest that current evaluation protocols may both overstate and understate unlearning success. To address this, we propose two principles for future unlearning evaluations: minimal information injection and downstream task awareness. We validate these principles through a series of targeted experiments, showing how violations of each can lead to misleading conclusions. 

**Abstract (ZH)**: 机器卸载旨在从大型语言模型中移除敏感或不希望的数据。然而，近期的研究表明卸载往往是浅层的，声称删除的知识可以很容易被恢复。在本项研究中，我们对标准卸载评估实践进行了批判性审查，并揭示了一系列关键限制，这些限制动摇了我们对这些发现的信任。首先，我们展示了某些评估引入了大量新的信息，可能导致在测试过程中重新“教导”模型，从而掩盖真实的卸载性能。其次，我们证明了评估结果在不同任务上存在显著差异，削弱了当前评估流程的一般适用性。最后，我们发现许多评估依赖于虚假的相关性，使得其结果难以信任和解释。综上所述，这些问题表明当前的评估协议可能既高估又低估了卸载的成功。为解决这一问题，我们提出了两项原则，用于未来卸载评估：最小信息注入和下游任务意识。我们通过一系列针对性的实验验证了这些原则，展示了违反每项原则可能导致误导性结论。 

---
# SafeTuneBed: A Toolkit for Benchmarking LLM Safety Alignment in Fine-Tuning 

**Title (ZH)**: SafeTuneBed: 一种评估fine-tuning过程中LLM安全对齐基准的工具包 

**Authors**: Saad Hossain, Samanvay Vajpayee, Sirisha Rambhatla  

**Link**: [PDF](https://arxiv.org/pdf/2506.00676)  

**Abstract**: As large language models (LLMs) become ubiquitous, parameter-efficient fine-tuning methods and safety-first defenses have proliferated rapidly. However, the number of approaches and their recent increase have resulted in diverse evaluations-varied datasets, metrics, and inconsistent threat settings-making it difficult to fairly compare safety, utility, and robustness across methods. To address this, we introduce SafeTuneBed, a benchmark and toolkit unifying fine-tuning and defense evaluation. SafeTuneBed (i) curates a diverse repository of multiple fine-tuning datasets spanning sentiment analysis, question-answering, multi-step reasoning, and open-ended instruction tasks, and allows for the generation of harmful-variant splits; (ii) enables integration of state-of-the-art defenses, including alignment-stage immunization, in-training safeguards, and post-tuning repair; and (iii) provides evaluators for safety (attack success rate, refusal consistency) and utility. Built on Python-first, dataclass-driven configs and plugins, SafeTuneBed requires minimal additional code to specify any fine-tuning regime, defense method, and metric suite, while ensuring end-to-end reproducibility. We showcase its value by benchmarking representative defenses across varied poisoning scenarios and tasks. By standardizing data, code, and metrics, SafeTuneBed is the first focused toolkit of its kind to accelerate rigorous and comparable research in safe LLM fine-tuning. Code is available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）日益普及，参数高效微调方法和安全优先防护措施迅速增长。然而，方法的多样性和近期的增长导致了评估的差异性，包括不同的数据集、指标和不一致的威胁设定，使得公平比较安全、实用性和鲁棒性变得困难。为解决这一问题，我们引入了SafeTuneBed，一个统一微调和防护评估基准与工具包。SafeTuneBed (i) 精心整理了涵盖情感分析、问答、多步推理和开放式指令任务的多样化微调数据集，并允许生成有害变体分割；(ii) 支持集成最新的防护措施，包括对齐阶段免疫、训练中防护和微调后修复；(iii) 提供安全性和实用性评估工具（攻击成功率、拒绝一致性）。SafeTuneBed 基于 Python 编程，使用数据类驱动配置和插件，只需最少的额外代码即可指定任何微调方案、防护方法和指标套件，确保端到端可重复性。我们通过在各种中毒场景和任务中评估代表性防护措施展示了其价值。通过标准化数据、代码和指标，SafeTuneBed 成为了首个专注于加速安全的大规模语言模型微调严谨且可比研究的工具包。代码可在以下链接获取：this https URL。 

---
# Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models 

**Title (ZH)**: 线性表示迁移性假设：利用小型模型引导大型模型 

**Authors**: Femi Bello, Anubrata Das, Fanzhi Zeng, Fangcong Yin, Leqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00653)  

**Abstract**: It has been hypothesized that neural networks with similar architectures trained on similar data learn shared representations relevant to the learning task. We build on this idea by extending the conceptual framework where representations learned across models trained on the same data can be expressed as linear combinations of a \emph{universal} set of basis features. These basis features underlie the learning task itself and remain consistent across models, regardless of scale. From this framework, we propose the \textbf{Linear Representation Transferability (LRT)} Hypothesis -- that there exists an affine transformation between the representation spaces of different models. To test this hypothesis, we learn affine mappings between the hidden states of models of different sizes and evaluate whether steering vectors -- directions in hidden state space associated with specific model behaviors -- retain their semantic effect when transferred from small to large language models using the learned mappings. We find strong empirical evidence that such affine mappings can preserve steering behaviors. These findings suggest that representations learned by small models can be used to guide the behavior of large models, and that the LRT hypothesis may be a promising direction on understanding representation alignment across model scales. 

**Abstract (ZH)**: 具有相似架构的神经网络在相似数据上训练时学习到与学习任务相关的共享表示吗？——线性表示转移性（LRT）假设 

---
# SATA-BENCH: Select All That Apply Benchmark for Multiple Choice Questions 

**Title (ZH)**: SATA-BENCH: 全选适用基准测试用于多项选择题 

**Authors**: Weijie Xu, Shixian Cui, Xi Fang, Chi Xue, Stephanie Eckman, Chandan Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.00643)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on single-answer multiple-choice tasks, yet many real-world problems require identifying all correct answers from a set of options. This capability remains underexplored. We introduce SATA-BENCH, the first dedicated benchmark for evaluating LLMs on Select All That Apply (SATA) questions across diverse domains, including reading comprehension, law, and biomedicine. Our evaluation of 27 open-source and proprietary models reveals a significant gap: even the strongest model achieves only 41.8% exact match, exposing LLMs' inability to reliably identify all correct answers. We find that this weakness stems from two core challenges: selection bias - models favor certain choices regardless of content, and count bias - models fail to predict the correct number of answers. To address these issues, we propose Choice Funnel, a decoding strategy that combines token debiasing with adaptive thresholding to guide models toward complete and accurate selections. Choice Funnel achieves up to 29% higher exact match than competitive baselines while reducing inference cost by over 64%. Our findings expose fundamental limitations in current LLMs and introduce a new framework for diagnosing and improving multi-answer reasoning. We release SATA-BENCH and Choice Funnel to promote LLM development for robust decision-making in realistic, multi-answer applications. 

**Abstract (ZH)**: SATA-BENCH：面向选所有适用项问题的LLM评估基准与Choice Funnel解码策略 

---
# ORAN-GUIDE: RAG-Driven Prompt Learning for LLM-Augmented Reinforcement Learning in O-RAN Network Slicing 

**Title (ZH)**: ORAN-GUIDE：基于RAG的prompt学习在O-RAN网络切片中的LLM增强 reinforcement学习 

**Authors**: Fatemeh Lotfi, Hossein Rajoli, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2506.00576)  

**Abstract**: Advanced wireless networks must support highly dynamic and heterogeneous service demands. Open Radio Access Network (O-RAN) architecture enables this flexibility by adopting modular, disaggregated components, such as the RAN Intelligent Controller (RIC), Centralized Unit (CU), and Distributed Unit (DU), that can support intelligent control via machine learning (ML). While deep reinforcement learning (DRL) is a powerful tool for managing dynamic resource allocation and slicing, it often struggles to process raw, unstructured input like RF features, QoS metrics, and traffic trends. These limitations hinder policy generalization and decision efficiency in partially observable and evolving environments. To address this, we propose \textit{ORAN-GUIDE}, a dual-LLM framework that enhances multi-agent RL (MARL) with task-relevant, semantically enriched state representations. The architecture employs a domain-specific language model, ORANSight, pretrained on O-RAN control and configuration data, to generate structured, context-aware prompts. These prompts are fused with learnable tokens and passed to a frozen GPT-based encoder that outputs high-level semantic representations for DRL agents. This design adopts a retrieval-augmented generation (RAG) style pipeline tailored for technical decision-making in wireless systems. Experimental results show that ORAN-GUIDE improves sample efficiency, policy convergence, and performance generalization over standard MARL and single-LLM baselines. 

**Abstract (ZH)**: O-RAN-GUIDE：基于双LLM框架的多智能体强化学习增强技术 

---
# Prompt-Tuned LLM-Augmented DRL for Dynamic O-RAN Network Slicing 

**Title (ZH)**: 提示调优的大规模语言模型增强的分布式 reinforcement 学习在动态O-RAN网络切片中的应用 

**Authors**: Fatemeh Lotfi, Hossein Rajoli, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2506.00574)  

**Abstract**: Modern wireless networks must adapt to dynamic conditions while efficiently managing diverse service demands. Traditional deep reinforcement learning (DRL) struggles in these environments, as scattered and evolving feedback makes optimal decision-making challenging. Large Language Models (LLMs) offer a solution by structuring unorganized network feedback into meaningful latent representations, helping RL agents recognize patterns more effectively. For example, in O-RAN slicing, concepts like SNR, power levels and throughput are semantically related, and LLMs can naturally cluster them, providing a more interpretable state representation. To leverage this capability, we introduce a contextualization-based adaptation method that integrates learnable prompts into an LLM-augmented DRL framework. Instead of relying on full model fine-tuning, we refine state representations through task-specific prompts that dynamically adjust to network conditions. Utilizing ORANSight, an LLM trained on O-RAN knowledge, we develop Prompt-Augmented Multi agent RL (PA-MRL) framework. Learnable prompts optimize both semantic clustering and RL objectives, allowing RL agents to achieve higher rewards in fewer iterations and adapt more efficiently. By incorporating prompt-augmented learning, our approach enables faster, more scalable, and adaptive resource allocation in O-RAN slicing. Experimental results show that it accelerates convergence and outperforms other baselines. 

**Abstract (ZH)**: 现代无线网络需适应动态条件并高效管理多样化服务需求。大型语言模型通过将无组织的网络反馈结构化为有意义的潜在表示，帮助强化学习代理更有效地识别模式。为此，我们提出了一种基于上下文的适应方法，该方法将可学习的提示整合到大型语言模型增强的强化学习框架中。通过任务特定的提示动态调整状态表示，而非依赖于全面的模型微调。利用ORANSight，一个基于O-RAN知识训练的大型语言模型，我们开发了提示增强多代理强化学习（PA-MRL）框架。可学习的提示优化语义聚类和强化学习目标，使强化学习代理在更少的迭代中获得更高奖励并更具适应性。通过结合提示增强的学习，我们的方法在O-RAN切片中实现了更快、更可扩展和更适应的资源分配。实验结果表明，该方法加速了收敛并优于其他基线。 

---
# AnnaAgent: Dynamic Evolution Agent System with Multi-Session Memory for Realistic Seeker Simulation 

**Title (ZH)**: AnnaAgent: 具有多会话记忆的动态进化代理系统及其在现实seeker模拟中的应用 

**Authors**: Ming Wang, Peidong Wang, Lin Wu, Xiaocui Yang, Daling Wang, Shi Feng, Yuxin Chen, Bixuan Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00551)  

**Abstract**: Constrained by the cost and ethical concerns of involving real seekers in AI-driven mental health, researchers develop LLM-based conversational agents (CAs) with tailored configurations, such as profiles, symptoms, and scenarios, to simulate seekers. While these efforts advance AI in mental health, achieving more realistic seeker simulation remains hindered by two key challenges: dynamic evolution and multi-session memory. Seekers' mental states often fluctuate during counseling, which typically spans multiple sessions. To address this, we propose AnnaAgent, an emotional and cognitive dynamic agent system equipped with tertiary memory. AnnaAgent incorporates an emotion modulator and a complaint elicitor trained on real counseling dialogues, enabling dynamic control of the simulator's configurations. Additionally, its tertiary memory mechanism effectively integrates short-term and long-term memory across sessions. Evaluation results, both automated and manual, demonstrate that AnnaAgent achieves more realistic seeker simulation in psychological counseling compared to existing baselines. The ethically reviewed and screened code can be found on this https URL. 

**Abstract (ZH)**: 基于LLM的配置定制化情感与认知动态代理系统AnnaAgent及其在心理健康咨询中更真实的 Seeker 模拟研究 

---
# Towards Multi-dimensional Evaluation of LLM Summarization across Domains and Languages 

**Title (ZH)**: 跨领域多维度评估多语言大型语言模型总结能力 

**Authors**: Hyangsuk Min, Yuho Lee, Minjeong Ban, Jiaqi Deng, Nicole Hee-Yeon Kim, Taewon Yun, Hang Su, Jason Cai, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.00549)  

**Abstract**: Evaluation frameworks for text summarization have evolved in terms of both domain coverage and metrics. However, existing benchmarks still lack domain-specific assessment criteria, remain predominantly English-centric, and face challenges with human annotation due to the complexity of reasoning. To address these, we introduce MSumBench, which provides a multi-dimensional, multi-domain evaluation of summarization in English and Chinese. It also incorporates specialized assessment criteria for each domain and leverages a multi-agent debate system to enhance annotation quality. By evaluating eight modern summarization models, we discover distinct performance patterns across domains and languages. We further examine large language models as summary evaluators, analyzing the correlation between their evaluation and summarization capabilities, and uncovering systematic bias in their assessment of self-generated summaries. Our benchmark dataset is publicly available at this https URL. 

**Abstract (ZH)**: 文本摘要评价框架在领域覆盖和度量标准方面不断发展，但现有基准仍缺乏领域特定的评估标准，主要偏向英语，并且由于推理复杂性，在人工标注方面存在挑战。为解决这些问题，我们引入MSumBench，该框架提供了英语和中文的多维度、多领域摘要评价，并为每个领域纳入了专门的评估标准，同时利用多智能体辩论系统提升标注质量。通过评估八种现代摘要模型，我们发现不同领域和语言中的性能模式各异。进一步地，我们研究了大型语言模型作为摘要评价器的有效性，分析了其评价与摘要生成能力之间的关系，并揭示了其评估自我生成摘要中的系统性偏见。我们的基准数据集可从此链接访问：this https URL。 

---
# Decoupling Reasoning and Knowledge Injection for In-Context Knowledge Editing 

**Title (ZH)**: 解耦推理与知识注入以实现上下文内知识编辑 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yujia Zhou, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00536)  

**Abstract**: Knowledge editing aims to efficiently update Large Language Models (LLMs) by modifying specific knowledge without retraining the entire model. Among knowledge editing approaches, in-context editing (ICE) offers a lightweight solution by injecting new knowledge directly into the input context, leaving model parameters unchanged. However, existing ICE approaches do not explicitly separate the newly injected knowledge from the model's original reasoning process. This entanglement often results in conflicts between external updates and internal parametric knowledge, undermining the consistency and accuracy of the reasoning this http URL this work, we conduct preliminary experiments to examine how parametric knowledge influences reasoning path planning. We find that the model's reasoning is tightly coupled with its internal knowledge, and that naively injecting new information without adapting the reasoning path often leads to performance degradation, particularly in multi-hop tasks. To this end, we propose DecKER, a novel ICE framework that decouples reasoning from knowledge editing by generating a masked reasoning path and then resolving knowledge edits via hybrid retrieval and model-based validation. Experiments on multi-hop QA benchmarks show that DecKER significantly outperforms existing ICE methods by mitigating knowledge conflicts and preserving reasoning consistency. Our code is available at: this https URL . 

**Abstract (ZH)**: 知识编辑旨在通过修改特定知识来高效更新大型语言模型（LLMs），而无需重新训练整个模型。在知识编辑方法中，上下文内编辑（ICE）提供了一种轻量级的解决方案，通过直接将新知识注入输入上下文来修改模型，而无需改变模型参数。然而，现有的ICE方法没有明确地将新注入的知识与模型的原始推理过程分离。这种纠缠通常会导致外部更新与内部参数知识之间的冲突，损害推理的一致性和准确性。在本工作中，我们进行了初步实验以考察参数知识如何影响推理路径规划。我们发现，模型的推理与其内部知识紧密耦合，未经调整地注入新信息往往会导致性能下降，尤其是在多跳任务中。为此，我们提出了一种名为DecKER的新ICE框架，通过生成掩码推理路径并借助混合检索和模型验证来解耦推理与知识编辑。在多跳问答基准测试上的实验表明，DecKER通过缓解知识冲突并保持推理一致性显著优于现有ICE方法。我们的代码可在以下链接获取：this https URL。 

---
# The Security Threat of Compressed Projectors in Large Vision-Language Models 

**Title (ZH)**: 大型视觉-语言模型中压缩投影器的安全威胁 

**Authors**: Yudong Zhang, Ruobing Xie, Xingwu Sun, Jiansheng Chen, Zhanhui Kang, Di Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00534)  

**Abstract**: The choice of a suitable visual language projector (VLP) is critical to the successful training of large visual language models (LVLMs). Mainstream VLPs can be broadly categorized into compressed and uncompressed projectors, and each offering distinct advantages in performance and computational efficiency. However, their security implications have not been thoroughly examined. Our comprehensive evaluation reveals significant differences in their security profiles: compressed projectors exhibit substantial vulnerabilities, allowing adversaries to successfully compromise LVLMs even with minimal knowledge of structural information. In stark contrast, uncompressed projectors demonstrate robust security properties and do not introduce additional vulnerabilities. These findings provide critical guidance for researchers in selecting optimal VLPs that enhance the security and reliability of visual language models. The code will be released. 

**Abstract (ZH)**: 选择合适的视觉语言投影器（VLP）对于大型视觉语言模型（LVLM）的成功训练至关重要。主流的VLP可以大致分为压缩和未压缩的两类，每种类型在性能和计算效率上各有优势。然而，它们的安全性影响尚未得到充分研究。我们的全面评估揭示了它们在安全性方面的显著差异：压缩投影器存在重大漏洞，即使对手对结构信息知之甚少，也能成功攻破LVLMs。相比之下，未压缩投影器展现出 robust 的安全特性，并不引入额外的漏洞。这些发现为研究人员在选择能够增强视觉语言模型安全性和可靠性的最优VLP方面提供了宝贵的指导。代码将发布。 

---
# M2WLLM: Multi-Modal Multi-Task Ultra-Short-term Wind Power Prediction Algorithm Based on Large Language Model 

**Title (ZH)**: M2WLLM：基于大型语言模型的多模态多任务超短期风电预测算法 

**Authors**: Hang Fana, Mingxuan Lib, Zuhan Zhanga, Long Chengc, Yujian Ye, Dunnan Liua  

**Link**: [PDF](https://arxiv.org/pdf/2506.00531)  

**Abstract**: The integration of wind energy into power grids necessitates accurate ultra-short-term wind power forecasting to ensure grid stability and optimize resource allocation. This study introduces M2WLLM, an innovative model that leverages the capabilities of Large Language Models (LLMs) for predicting wind power output at granular time intervals. M2WLLM overcomes the limitations of traditional and deep learning methods by seamlessly integrating textual information and temporal numerical data, significantly improving wind power forecasting accuracy through multi-modal data. Its architecture features a Prompt Embedder and a Data Embedder, enabling an effective fusion of textual prompts and numerical inputs within the LLMs framework. The Semantic Augmenter within the Data Embedder translates temporal data into a format that the LLMs can comprehend, enabling it to extract latent features and improve prediction accuracy. The empirical evaluations conducted on wind farm data from three Chinese provinces demonstrate that M2WLLM consistently outperforms existing methods, such as GPT4TS, across various datasets and prediction horizons. The results highlight LLMs' ability to enhance accuracy and robustness in ultra-short-term forecasting and showcase their strong few-shot learning capabilities. 

**Abstract (ZH)**: 将风能融入电力 grid 需要精确的超短期风电功率预测以确保 grid 稳定性和优化资源配置。本研究介绍了 M2WLLM，这是一种创新模型，利用大型语言模型（LLMs）的能力，在粒度时间间隔内预测风电功率输出。M2WLLM 通过无缝集成文本信息和时间序列数值数据，克服了传统方法和深度学习方法的局限性，显著提高了风电功率预测准确性。其架构包括 Prompt Embedder 和 Data Embedder，能够在 LLMs 框架内有效融合文本提示和数值输入。Data Embedder 中的 Semantic Augmenter 将时间数据转换为 LLMs 可理解的格式，使其能够提取潜在特征并提高预测准确性。在来自中国三个省份风电场的数据上进行的经验评估表明，M2WLLM 在不同数据集和预测时间尺度上始终优于现有方法（如 GPT4TS），结果突显了 LLMs 在超短期预测中提高准确性和稳健性的能力，并展示了它们强大的少样本学习能力。 

---
# Retrieval-Augmented Generation Systems for Intellectual Property via Synthetic Multi-Angle Fine-tuning 

**Title (ZH)**: 基于合成多角度微调的知识产权检索增强生成系统 

**Authors**: Runtao Ren, Jian Ma, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00527)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems in the Intellectual Property (IP) field often struggle with diverse user queries, including colloquial expressions, spelling errors, and ambiguous terminology, leading to inaccurate retrieval and suboptimal responses. To address this challenge, we propose Multi-Angle Question Generation and Retrieval Fine-Tuning Method (MQG-RFM), a novel framework that leverages large language models (LLMs) to simulate varied user inquiries and fine-tunes retrieval models to align semantically equivalent but linguistically diverse questions. Unlike complex architectural modifications, MQG-RFM adopts a lightweight Data-to-Tune paradigm, combining prompt-engineered query generation with hard negative mining to enhance retrieval robustness without costly infrastructure changes. Experimental results on a Taiwan patent Q&A dataset show 185.62% improvement in retrieval accuracy on the Patent Consultation dataset and 262.26% improvement on the Novel Patent Technology Report dataset, with 14.22% and 53.58% improvements in generation quality over the baselines, respectively. By bridging the gap between user intent and system comprehension through semantic-aware retrieval optimization, MQG-RFM offers a practical, scalable approach for rapid, cost-effective deployment among small and medium-sized agencies seeking reliable patent intelligence solutions. Additionally, our proposed method has already been adopted by ScholarMate, the largest professional research social networking platform in China, to support real-world development and deployment. A demo version of the instantiated is available at this https URL. 

**Abstract (ZH)**: 知识产权领域中基于检索增强生成（RAG）系统在处理多样化用户查询时往往遇到挑战，包括非正式表达、拼写错误和含糊术语，导致检索不准确和响应不尽如人意。为此，我们提出了多角度问题生成和检索微调方法（MQG-RFM），一种利用大规模语言模型（LLMs）模拟多样化用户询问并微调检索模型以对齐语义上等价但语言上多样的问题的新型框架。MQG-RFM 采用轻量级的 Data-to-Tune 帕累托图，结合提示工程优化的查询生成和困难负样本挖掘，增强检索稳健性，而无需昂贵的基础设施变化。在台湾专利问答数据集上的实验结果表明，MQG-RFM 在专利咨询数据集上的检索准确性提高了 185.62%，在新型专利技术报告数据集上的检索准确性提高了 262.26%，相较于基线，在生成质量上分别提高了 14.22% 和 53.58%。通过通过语义感知的检索优化弥合用户意图与系统理解之间的差距，MQG-RFM 提供了一种实用、可扩展的方法，适用于寻求可靠专利情报解决方案的中小机构进行快速、低成本部署。此外，我们提出的方法已被中国最大的专业研究社交网络平台 ScholarMate 采纳，支持实际应用开发和部署。该实现的演示版本可访问此 [链接]。 

---
# CausalAbstain: Enhancing Multilingual LLMs with Causal Reasoning for Trustworthy Abstention 

**Title (ZH)**: 因果性规避：通过因果推理增强多语言LLMs以实现可靠的规避 

**Authors**: Yuxi Sun, Aoqi Zuo, Wei Gao, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00519)  

**Abstract**: Large Language Models (LLMs) often exhibit knowledge disparities across languages. Encouraging LLMs to \textit{abstain} when faced with knowledge gaps is a promising strategy to reduce hallucinations in multilingual settings. Current abstention strategies for multilingual scenarios primarily rely on generating feedback in various languages using LLMs and performing self-reflection. However, these methods can be adversely impacted by inaccuracies and biases in the generated feedback. To address this, from a causal perspective, we introduce \textit{CausalAbstain}, a method that helps LLMs determine whether to utilize multiple generated feedback responses and how to identify the most useful ones. Extensive experiments demonstrate that \textit{CausalAbstain} effectively selects helpful feedback and enhances abstention decisions with interpretability in both native language (\textsc{Casual-native}) and multilingual (\textsc{Causal-multi}) settings, outperforming strong baselines on two benchmark datasets covering encyclopedic and commonsense knowledge QA tasks. Our code and data are open-sourced at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在不同语言中往往表现出知识上的差异。当面对知识空白时，鼓励LLMs做出排除选择是减少多语言设置中幻觉的有效策略。目前的多语言排除策略主要依赖于使用LLMs生成多种语言的反馈并进行自我反思。然而，这些方法可能会受到生成反馈中的不准确性和偏向性的影响。为了解决这一问题，从因果角度出发，我们提出了一种名为CausalAbstain的方法，帮助LLMs判断是否以及如何利用多个生成的反馈响应，并识别最有用的反馈。广泛实验证明，CausalAbstain能够在母语（\textsc{Casual-native}）和多语言（\textsc{Causal-multi}）设置中有效地选择有用的反馈，并提高可解释性的排除决策，在两个基准数据集中优于强基线，这些基准数据集涵盖了百科知识和常识问答任务。我们的代码和数据已开源。 

---
# It Takes a Good Model to Train a Good Model: Generalized Gaussian Priors for Optimized LLMs 

**Title (ZH)**: 好的模型需要好的模型来训练：通用高斯先验优化大型语言模型 

**Authors**: Jun Wu, Yirong Xiong, Jiangtao Wen, Yuxing Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.00486)  

**Abstract**: Despite rapid advancements in the research and deployment of large language models (LLMs), the statistical distribution of model parameters, as well as their influence on initialization, training dynamics, and downstream efficiency, has received surprisingly little attention. A recent work introduced BackSlash, a training-time compression algorithm. It first demonstrated that pre-trained LLM parameters follow generalized Gaussian distributions (GGDs) better. By optimizing GG priors during training, BackSlash can reduce parameters by up to 90\% with minimal performance loss. Building on this foundational insight, we propose a unified, end-to-end framework for LLM optimization based on the GG model. Our contributions are threefold: (1) GG-based initialization scheme that aligns with the statistical structure of trained models, resulting in faster convergence and improved accuracy; (2) DeepShape, a post-training regularization method that reshapes weight distributions to match a GG profile, improving compressibility with minimized degradation in performance; and (3) RF8, a compact and hardware-efficient 8-bit floating-point format designed for GG-distributed-initialized BackSlash training, enabling low-cost inference without compromising accuracy. Experiments across diverse model architectures show that our framework consistently yields smaller and faster models that match or outperform standard training baselines. By grounding LLM development in principled statistical modeling, this work forges a new path toward efficient, scalable, and hardware-aware AI systems. The code is available on our project page: this https URL. 

**Abstract (ZH)**: 尽管在大型语言模型（LLMs）的研究和部署方面取得了 rapid advancements，模型参数的统计分布及其对初始化、训练动力学和下游效率的影响却收到了相对较少的关注。最近一项研究引入了 BackSlash，一种训练时压缩算法。研究表明，预训练的 LLM 参数更适合遵循广义高斯分布（GGDs）。通过在训练中优化 GG 先验，BackSlash 可以将参数量减少高达 90%，同时保持最小的性能损失。基于这一基础洞察，我们提出了一种统一的端到端框架，以 GG 模型为基础优化 LLM。我们的贡献包括三个方面：（1）基于 GG 的初始化方案，与训练模型的统计结构相契合，从而实现更快的收敛和更高的精度；（2）提出了一种后训练正则化方法 DeepShape，重塑权重分布以匹配 GG 轮廓，提高压缩性并减少性能退化；（3）设计了一种紧凑且硬件高效的 8 位浮点格式 RF8，专为 GG 分布初始化的 BackSlash 训练而设计，可以在不牺牲精度的情况下实现低成本推理。实验结果显示，我们的框架在多种模型架构下能够产生更小、更快且不劣于标准训练基准模型的模型。通过将 LLM 开发建立在原理性统计建模的基础上，本工作开辟了一条走向高效、可扩展且硬件感知的 AI 系统的新途径。该项目代码可在我们的项目页面获得：this https URL。 

---
# BenchHub: A Unified Benchmark Suite for Holistic and Customizable LLM Evaluation 

**Title (ZH)**: BenchHub: 综合可定制的大语言模型评估基准套件 

**Authors**: Eunsu Kim, Haneul Yoo, Guijin Son, Hitesh Patel, Amit Agarwal, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00482)  

**Abstract**: As large language models (LLMs) continue to advance, the need for up-to-date and well-organized benchmarks becomes increasingly critical. However, many existing datasets are scattered, difficult to manage, and make it challenging to perform evaluations tailored to specific needs or domains, despite the growing importance of domain-specific models in areas such as math or code. In this paper, we introduce BenchHub, a dynamic benchmark repository that empowers researchers and developers to evaluate LLMs more effectively. BenchHub aggregates and automatically classifies benchmark datasets from diverse domains, integrating 303K questions across 38 benchmarks. It is designed to support continuous updates and scalable data management, enabling flexible and customizable evaluation tailored to various domains or use cases. Through extensive experiments with various LLM families, we demonstrate that model performance varies significantly across domain-specific subsets, emphasizing the importance of domain-aware benchmarking. We believe BenchHub can encourage better dataset reuse, more transparent model comparisons, and easier identification of underrepresented areas in existing benchmarks, offering a critical infrastructure for advancing LLM evaluation research. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的不断进步，及时且井然有序的基准测试需求变得越来越关键。然而，许多现有的数据集分散、管理困难，使得针对特定需求或领域进行定制化评估变得具有挑战性，尽管在数学或代码等领域中特定领域模型的重要性日益增长。在本文中，我们介绍了一种动态基准存储库BenchHub，它能够帮助研究人员和开发者更有效地评估LLMs。BenchHub整合并自动分类来自不同领域的基准数据集，涵盖了38个基准中的303,000个问题，旨在支持持续更新和可扩展的数据管理，实现针对不同领域或应用场景的灵活和定制化评估。通过对各种LLM家族的广泛实验，我们展示了模型在特定领域子集中的性能差异显著，突显了领域意识基准测试的重要性。我们相信BenchHub能够促进数据集的更好重用、更透明的模型比较以及更容易识别现有基准中的不足领域，为推进LLM评估研究提供关键基础设施。 

---
# RLAE: Reinforcement Learning-Assisted Ensemble for LLMs 

**Title (ZH)**: RLAE: 基于强化学习的LLM集成方法 

**Authors**: Yuqian Fu, Yuanheng Zhu, Jiajun Chai, Guojun Yin, Wei Lin, Qichao Zhang, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00439)  

**Abstract**: Ensembling large language models (LLMs) can effectively combine diverse strengths of different models, offering a promising approach to enhance performance across various tasks. However, existing methods typically rely on fixed weighting strategies that fail to adapt to the dynamic, context-dependent characteristics of LLM capabilities. In this work, we propose Reinforcement Learning-Assisted Ensemble for LLMs (RLAE), a novel framework that reformulates LLM ensemble through the lens of a Markov Decision Process (MDP). Our approach introduces a RL agent that dynamically adjusts ensemble weights by considering both input context and intermediate generation states, with the agent being trained using rewards that directly correspond to the quality of final outputs. We implement RLAE using both single-agent and multi-agent reinforcement learning algorithms ($\text{RLAE}_\text{PPO}$ and $\text{RLAE}_\text{MAPPO}$ ), demonstrating substantial improvements over conventional ensemble methods. Extensive evaluations on a diverse set of tasks show that RLAE outperforms existing approaches by up to $3.3\%$ accuracy points, offering a more effective framework for LLM ensembling. Furthermore, our method exhibits superior generalization capabilities across different tasks without the need for retraining, while simultaneously achieving lower time latency. 

**Abstract (ZH)**: 增强学习辅助的大语言模型集成（RLAE）：一种基于马尔可夫决策过程的新框架 

---
# Dual Debiasing for Noisy In-Context Learning for Text Generation 

**Title (ZH)**: 基于文本生成的噪声在语境学习双重去偏差方法 

**Authors**: Siqi Liang, Sumyeong Ahn, Paramveer S. Dhillon, Jiayu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00418)  

**Abstract**: In context learning (ICL) relies heavily on high quality demonstrations drawn from large annotated corpora. Existing approaches detect noisy annotations by ranking local perplexities, presuming that noisy samples yield higher perplexities than their clean counterparts. However, this assumption breaks down when the noise ratio is high and many demonstrations are flawed. We reexamine the perplexity based paradigm for text generation under noisy annotations, highlighting two sources of bias in perplexity: the annotation itself and the domain specific knowledge inherent in large language models (LLMs). To overcome these biases, we introduce a dual debiasing framework that uses synthesized neighbors to explicitly correct perplexity estimates, yielding a robust Sample Cleanliness Score. This metric uncovers absolute sample cleanliness regardless of the overall corpus noise level. Extensive experiments demonstrate our method's superior noise detection capabilities and show that its final ICL performance is comparable to that of a fully clean demonstration corpus. Moreover, our approach remains robust even when noise ratios are extremely high. 

**Abstract (ZH)**: 基于上下文学习中的去偏见采样洁净度评分：在噪声标注下的文本生成 

---
# Wide Reflective Equilibrium in LLM Alignment: Bridging Moral Epistemology and AI Safety 

**Title (ZH)**: LLM对齐中的广泛反思平衡：道德Epistemology与AI安全的桥梁 

**Authors**: Matthew Brophy  

**Link**: [PDF](https://arxiv.org/pdf/2506.00415)  

**Abstract**: As large language models (LLMs) become more powerful and pervasive across society, ensuring these systems are beneficial, safe, and aligned with human values is crucial. Current alignment techniques, like Constitutional AI (CAI), involve complex iterative processes. This paper argues that the Method of Wide Reflective Equilibrium (MWRE) -- a well-established coherentist moral methodology -- offers a uniquely apt framework for understanding current LLM alignment efforts. Moreover, this methodology can substantively augment these processes by providing concrete pathways for improving their dynamic revisability, procedural legitimacy, and overall ethical grounding. Together, these enhancements can help produce more robust and ethically defensible outcomes. MWRE, emphasizing the achievement of coherence between our considered moral judgments, guiding moral principles, and relevant background theories, arguably better represents the intricate reality of LLM alignment and offers a more robust path to justification than prevailing foundationalist models or simplistic input-output evaluations. While current methods like CAI bear a structural resemblance to MWRE, they often lack its crucial emphasis on dynamic, bi-directional revision of principles and the procedural legitimacy derived from such a process. While acknowledging various disanalogies (e.g., consciousness, genuine understanding in LLMs), the paper demonstrates that MWRE serves as a valuable heuristic for critically analyzing current alignment efforts and for guiding the future development of more ethically sound and justifiably aligned AI systems. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在社会中的力量增强和普及，确保这些系统有益、安全，并与人类价值观保持一致至关重要。当前的对齐技术，如宪法AI（CAI），涉及到复杂的迭代过程。本文认为，广泛反思均衡的方法（MWRE）——一种成熟的共摄主义道德方法——为理解当前LLM对齐努力提供了独特而合适的框架。此外，这种方法可以通过提供改进当前对齐过程的动态可 revisability、程序合法性以及整体道德奠基的具体路径，实质性地增强这些过程。这些改进有助于产生更具可靠性和道德辩护性的结果。MWRE 强调实现我们考虑的道德判断、指导性道德原则和相关背景理论之间的一致性， arguably 更好地代表了LLM对齐的复杂现实，并提供了一条比当前占主导地位的基础主义模型或简单输入-输出评估更为坚实的论证路径。尽管当前方法如CAI在结构上与MWRE有相似之处，但它们往往缺乏MWRE 关注的动态、双向原则修订以及这样一种过程衍生的程序合法性等关键重点。尽管承认各种不相似之处（例如，意识、LLMs中的真正理解），本文证明MWRE 作为批判性分析当前对齐努力和指导未来开发更具道德依据和正当性的AI系统的有效启发式工具的价值。 

---
# Accelerating Diffusion LLMs via Adaptive Parallel Decoding 

**Title (ZH)**: 通过自适应并行解码加速扩散语言模型 

**Authors**: Daniel Israel, Guy Van den Broeck, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2506.00413)  

**Abstract**: The generation speed of LLMs are bottlenecked by autoregressive decoding, where tokens are predicted sequentially one by one. Alternatively, diffusion large language models (dLLMs) theoretically allow for parallel token generation, but in practice struggle to achieve the speed of autoregressive models without significantly sacrificing quality. We therefore introduce adaptive parallel decoding (APD), a novel method that dynamically adjusts the number of tokens sampled in parallel. We achieve this by defining a multiplicative mixture between the dLLM marginal probabilities and the joint probability of sequences under a small auxiliary autoregressive model. This inverts the standard setup of speculative decoding, where the goal is to sample from a large autoregressive verifier by drafting from a smaller model. We further optimize APD by enabling KV caching and limiting the size of the masked input. Altogether, our method puts forward three tunable parameters to flexibly tradeoff throughput and quality. We show that APD provides markedly higher throughput with minimal quality degradations on downstream benchmarks. 

**Abstract (ZH)**: LLMs的生成速度受限于自回归解码，其中 Tokens 逐个依次预测。 alternatively，扩散大语言模型（dLLMs）理论上允许并行 Token 生成，但在实践中难以在不显著牺牲质量的情况下达到自回归模型的生成速度。因此，我们引入了自适应并行解码（APD），这是一种新颖的方法，能够动态调整并行生成的 Tokens 数量。通过在 dLLM 的边际概率与一个小辅助自回归模型下的序列联合概率之间定义乘法混合，我们实现了这一点。这反转了投机性解码的标准设置，其中目标是从一个较小的模型中招募，以采样一个大型自回归验证器。我们进一步通过启用 KV 缓存并限制遮罩输入的大小优化了 APD。总体而言，我们的方法提出了三个可调参数，灵活权衡吞吐量和质量。我们展示了 APD 在下游基准测试中提供了显著更高的吞吐量，并且质量损失很少。 

---
# Scaling Textual Gradients via Sampling-Based Momentum 

**Title (ZH)**: 基于采样加速的文本梯度扩展 

**Authors**: Zixin Ding, Junyuan Hong, Jiachen T. Wang, Zinan Lin, Zhangyang Wang, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00400)  

**Abstract**: As prompts play an increasingly critical role in large language models (LLMs), optimizing textual prompts has become a crucial challenge. The Textual Gradient Descent (TGD) framework has emerged as a promising data-driven approach that iteratively refines textual prompts using LLM - suggested updates (or textual gradients) over minibatches of training samples. In this paper, we empirically demonstrate that scaling the number of training examples initially improves but later degrades TGD's performance across multiple downstream NLP tasks. However, while data scaling improves results for most tasks, it also significantly increases the computational cost when leveraging LLMs. To address this, we draw inspiration from numerical gradient descent and propose Textual Stochastic Gradient Descent with Momentum (TSGD-M) - a method that facilitates scalable in-context learning by reweighting prompt sampling based on past batch distributions. Across nine NLP tasks spanning three domains - including BIG-Bench Hard (BBH), natural language understanding tasks, and reasoning tasks - TSGD-M significantly outperforms TGD baselines that do not incorporate reweighted sampling, while also reducing variance in most tasks. 

**Abstract (ZH)**: 大型语言模型中文本提示的优化：文本随机梯度下降带动量（TSGD-M）方法的研究 

---
# Efficient Latent Semantic Clustering for Scaling Test-Time Computation of LLMs 

**Title (ZH)**: 高效的潜在语义聚类方法以扩展大型语言模型测试时的计算能力 

**Authors**: Sungjae Lee, Hoyoung Kim, Jeongyeon Hwang, Eunhyeok Park, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2506.00344)  

**Abstract**: Scaling test-time computation--generating and analyzing multiple or sequential outputs for a single input--has become a promising strategy for improving the reliability and quality of large language models (LLMs), as evidenced by advances in uncertainty quantification and multi-step reasoning. A key shared component is semantic clustering, which groups outputs that differ in form but convey the same meaning. Semantic clustering enables estimation of the distribution over the semantics of outputs and helps avoid redundant exploration of reasoning paths. However, existing approaches typically rely on external models, which introduce substantial computational overhead and often fail to capture context-aware semantics. We propose Latent Semantic Clustering (LSC), a lightweight and context-sensitive method that leverages the generator LLM's internal hidden states for clustering, eliminating the need for external models. Our extensive experiment across various LLMs and datasets shows that LSC significantly improves the computational efficiency of test-time scaling while maintaining or exceeding the performance of existing methods. 

**Abstract (ZH)**: 扩展测试时计算——为单个输入生成和分析多个或序列输出已成为提高大型语言模型（LLMs）可靠性和质量的有前途的策略，这得到了不确定性量化和多步推理进步的证明。一项关键共有组件是语义聚类，它将形式不同但意义相同的输出分组。语义聚类使输出语义分布的估计成为可能，并有助于避免对推理路径的冗余探索。然而，现有方法通常依赖于外部模型，这引入了大量计算开销，并且往往无法捕捉上下文感知的语义。我们提出了隐含语义聚类（LSC），这是一种轻量级且上下文感知的方法，利用生成器LLM的内部隐藏状态进行聚类，从而消除对外部模型的依赖。在各种LLM和数据集上的广泛实验表明，LSC在维护或超过现有方法性能的同时，显著提高了测试时扩展的计算效率。 

---
# An evaluation of LLMs for generating movie reviews: GPT-4o, Gemini-2.0 and DeepSeek-V3 

**Title (ZH)**: 对生成电影评论的LLMs进行评估：GPT-4o、Gemini-2.0和DeepSeek-V3 

**Authors**: Brendan Sands, Yining Wang, Chenhao Xu, Yuxuan Zhou, Lai Wei, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00312)  

**Abstract**: Large language models (LLMs) have been prominent in various tasks, including text generation and summarisation. The applicability of LLMs to the generation of product reviews is gaining momentum, paving the way for the generation of movie reviews. In this study, we propose a framework that generates movie reviews using three LLMs (GPT-4o, DeepSeek-V3, and Gemini-2.0), and evaluate their performance by comparing the generated outputs with IMDb user reviews. We use movie subtitles and screenplays as input to the LLMs and investigate how they affect the quality of reviews generated. We review the LLM-based movie reviews in terms of vocabulary, sentiment polarity, similarity, and thematic consistency in comparison to IMDB user reviews. The results demonstrate that LLMs are capable of generating syntactically fluent and structurally complete movie reviews. Nevertheless, there is still a noticeable gap in emotional richness and stylistic coherence between LLM-generated and IMDb reviews, suggesting that further refinement is needed to improve the overall quality of movie review generation. We provided a survey-based analysis where participants were told to distinguish between LLM and IMDb user reviews. The results show that LLM-generated reviews are difficult to distinguish from IMDB user reviews. We found that DeepSeek-V3 produced the most balanced reviews, closely matching IMDb reviews. GPT-4o overemphasised positive emotions, while Gemini-2.0 captured negative emotions better but showed excessive emotional intensity. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各类任务中占据重要地位，包括文本生成和摘要。LLMs在生成产品评论方面的应用逐渐增多，为电影评论的生成铺平了道路。本文提出一种框架，使用三种LLMs（GPT-4o、DeepSeek-V3和Gemini-2.0）生成电影评论，并通过将生成的输出与IMDb用户评论进行比较来评估其性能。我们使用电影字幕和剧本作为输入，探讨其对生成评论质量的影响。从词汇、情感极性、相似性和主题一致性方面，我们将LLM生成的电影评论与IMDb用户评论进行比较分析。结果表明，LLMs能够生成语法流畅且结构完整的电影评论。然而，LLM生成的评论在情感丰富性和风格连贯性方面仍与IMDb评论存在明显差距，提示需要进一步优化以提高电影评论生成的整体质量。我们还进行了基于调查的分析，让参与者辨别LLM生成的评论与IMDb用户评论。结果表明，LLM生成的评论难以区分。研究发现，DeepSeek-V3生成的评论最为平衡，接近IMDb评论。GPT-4o过分强调积极情绪，而Gemini-2.0较好地捕捉了负面情绪，但表现出过度的情感强度。 

---
# Lossless Token Sequence Compression via Meta-Tokens 

**Title (ZH)**: 基于元令牌的无损令牌序列压缩 

**Authors**: John Harvill, Ziwei Fan, Hao Wang, Yizhou Sun, Hao Ding, Luke Huan, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2506.00307)  

**Abstract**: Existing work on prompt compression for Large Language Models (LLM) focuses on lossy methods that try to maximize the retention of semantic information that is relevant to downstream tasks while significantly reducing the sequence length. In this paper, we introduce a task-agnostic lossless compression technique similar to LZ77 that makes it possible to reduce the input token sequence length on average by 27\% and 18\% for the two evaluation tasks explored here. Given that we use transformer-based LLMs, this equates to 47\% and 33\% less encoding computation, respectively, due to the quadratic nature of attention. The token sequence transformation is trivial to reverse and highlights that no semantic information is lost in the process. We evaluate our proposed approach on two tasks that require strict preservation of semantics/syntax and demonstrate that existing lossy compression methods perform poorly in this setting. We find that our lossless compression technique produces only a small gap in performance compared to using the uncompressed input and posit that larger models and an expanded computing budget would likely erase the gap entirely. 

**Abstract (ZH)**: 面向大型语言模型的提示压缩研究专注于在显著减少序列长度的同时，最大限度地保留与下游任务相关的语义信息。本文介绍了一种任务无关的无损压缩技术，类似于LZ77，能够平均减少两个评估任务输入标记序列长度的27%和18%。由于我们使用基于 Transformer 的大型语言模型，这意味着分别减少了47%和33%的编码计算量，这是由于注意力机制的二次特性。标记序列的转换易于逆向，表明在过程中没有丢失任何语义信息。我们评估了所提出的方法，并在需要严格保留语义/语法的两个任务中展示了现有无损压缩方法表现不佳的情况。我们发现，我们的无损压缩技术与使用未压缩输入相比，仅带来很小的性能差距，并推测更大的模型和更扩展的计算预算可能会完全消除这一差距。 

---
# Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation 

**Title (ZH)**: 大规模语言模型在持续预训练下的语言适应能力涌现 

**Authors**: Ahmed Elhady, Eneko Agirre, Mikel Artetxe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00288)  

**Abstract**: Continued pretraining (CPT) is a popular approach to adapt existing large language models (LLMs) to new languages. When doing so, it is common practice to include a portion of English data in the mixture, but its role has not been carefully studied to date. In this work, we show that including English does not impact validation perplexity, yet it is critical for the emergence of downstream capabilities in the target language. We introduce a language-agnostic benchmark for in-context learning (ICL), which reveals catastrophic forgetting early on CPT when English is not included. This in turn damages the ability of the model to generalize to downstream prompts in the target language as measured by perplexity, even if it does not manifest in terms of accuracy until later in training, and can be tied to a big shift in the model parameters. Based on these insights, we introduce curriculum learning and exponential moving average (EMA) of weights as effective alternatives to mitigate the need for English. All in all, our work sheds light into the dynamics by which emergent abilities arise when doing CPT for language adaptation, and can serve as a foundation to design more effective methods in the future. 

**Abstract (ZH)**: 持续预训练（CPT）是将现有大型语言模型（LLMs）适配到新语言的一种流行方法。尽管通常会在混合数据中包含一部分英语数据，但其作用尚未得到仔细研究。在本工作中，我们表明包含英语不会影响验证困惑度，但它对于目标语言下游能力的出现至关重要。我们引入了一种跨语言的在上下文学习基准测试，该基准测试揭示了在CPT过程中不包含英语时早期出现的灾难性遗忘。这反过来损害了模型在目标语言中对下游提示的泛化能力，即使困惑度在训练后期才表现出准确性下降，这与模型参数的大幅变化有关。基于这些洞见，我们引入了分阶段学习和权重的指数移动平均（EMA）作为减少对英语依赖的有效替代方法。总体而言，我们的工作揭示了在进行语言适配的持续预训练过程中涌现能力的动态机制，可以为未来设计更有效的方法奠定基础。 

---
# Adversarial Threat Vectors and Risk Mitigation for Retrieval-Augmented Generation Systems 

**Title (ZH)**: 检索增强生成系统中的对抗威胁向量与风险缓解 

**Authors**: Chris M. Ward, Josh Harguess  

**Link**: [PDF](https://arxiv.org/pdf/2506.00281)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems, which integrate Large Language Models (LLMs) with external knowledge sources, are vulnerable to a range of adversarial attack vectors. This paper examines the importance of RAG systems through recent industry adoption trends and identifies the prominent attack vectors for RAG: prompt injection, data poisoning, and adversarial query manipulation. We analyze these threats under risk management lens, and propose robust prioritized control list that includes risk-mitigating actions like input validation, adversarial training, and real-time monitoring. 

**Abstract (ZH)**: RAG系统中的 adversarial攻击向量及其风险管理策略：基于输入验证、对抗训练和实时监控的对策清单 

---
# Chances and Challenges of the Model Context Protocol in Digital Forensics and Incident Response 

**Title (ZH)**: 模型上下文协议在数字Forensics与事件响应中的机遇与挑战 

**Authors**: Jan-Niclas Hilgert, Carlo Jakobs, Michael Külper, Martin Lambertz, Axel Mahr, Elmar Padilla  

**Link**: [PDF](https://arxiv.org/pdf/2506.00274)  

**Abstract**: Large language models hold considerable promise for supporting forensic investigations, but their widespread adoption is hindered by a lack of transparency, explainability, and reproducibility. This paper explores how the emerging Model Context Protocol can address these challenges and support the meaningful use of LLMs in digital forensics. Through a theoretical analysis, we examine how MCP can be integrated across various forensic scenarios - ranging from artifact analysis to the generation of interpretable reports. We also outline both technical and conceptual considerations for deploying an MCP server in forensic environments. Our analysis reveals a wide range of use cases in which MCP not only strengthens existing forensic workflows but also facilitates the application of LLMs to areas of forensics where their use was previously limited. Furthermore, we introduce the concept of the inference constraint level - a way of characterizing how specific MCP design choices can deliberately constrain model behavior, thereby enhancing both auditability and traceability. Our insights demonstrate that MCP has significant potential as a foundational component for developing LLM-assisted forensic workflows that are not only more transparent, reproducible, and legally defensible, but also represent a step toward increased automation in digital forensic analysis. However, we also highlight potential challenges that the adoption of MCP may pose for digital forensics in the future. 

**Abstract (ZH)**: 大型语言模型在支持执法调查方面具有巨大潜力，但其广泛应用受到透明性、可解释性和可重复性不足的阻碍。本文探讨了新兴的模型上下文协议如何解决这些问题，并支持在数字取证中有效使用LLM。通过理论分析，我们研究了MCP如何跨各种取证场景集成——从 artifact 分析到生成可解释的报告。我们还概述了在取证环境中部署MCP服务器的技术和概念考虑。我们的分析表明，MCP不仅加强了现有的取证工作流程，还促进了LLM在之前受限的取证领域的应用。此外，我们提出了推断约束级别这一概念——一种描述特定MCP设计选择如何故意限制模型行为的方式，从而增强审计和追踪能力。我们的见解表明，MCP作为开发更透明、可重复且法律上可辩护的LLM辅助取证工作流程的基础组件具有巨大潜力，并代表了数字取证分析自动化的一个步骤。然而，我们还指出了MCP的采用可能在未来对数字取证提出的潜在挑战。 

---
# Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race 

**Title (ZH)**: 齊而不盲：對齊增強了內隱歧視By減少對种族的认知 

**Authors**: Lihao Sun, Chengzhi Mao, Valentin Hofmann, Xuechunzi Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00253)  

**Abstract**: Although value-aligned language models (LMs) appear unbiased in explicit bias evaluations, they often exhibit stereotypes in implicit word association tasks, raising concerns about their fair usage. We investigate the mechanisms behind this discrepancy and find that alignment surprisingly amplifies implicit bias in model outputs. Specifically, we show that aligned LMs, unlike their unaligned counterparts, overlook racial concepts in early internal representations when the context is ambiguous. Not representing race likely fails to activate safety guardrails, leading to unintended biases. Inspired by this insight, we propose a new bias mitigation strategy that works by incentivizing the representation of racial concepts in the early model layers. In contrast to conventional mitigation methods of machine unlearning, our interventions find that steering the model to be more aware of racial concepts effectively mitigates implicit bias. Similar to race blindness in humans, ignoring racial nuances can inadvertently perpetuate subtle biases in LMs. 

**Abstract (ZH)**: 尽管对齐的价值导向语言模型（LMs）在显性偏见评估中显得无偏，但在隐式单词联想任务中往往表现出刻板印象，这引起了对其公平使用的关注。我们探究了这种差异背后的机制并发现，令人惊讶的是，对齐实际上放大了模型输出中的隐式偏见。具体来说，我们展示了对齐的LM在上下文模糊时，不像其未对齐的对应物那样在早期内部表示中关注种族概念。不表示种族很可能未能激活安全防护，导致无意中的偏见。受到这一洞察的启发，我们提出了一种新的偏见缓解策略，该策略通过激励早期模型层中表示种族概念来发挥作用。与传统的机器遗忘缓解方法不同，我们的干预措施发现，将模型引导得更关注种族概念有效地缓解了隐式偏见。类似于人类的种族盲视，忽视种族细微差别可能会无意中在LM中延续微妙的偏见。 

---
# Structure-Aware Fill-in-the-Middle Pretraining for Code 

**Title (ZH)**: 结构感知填中间预训练用于代码生成 

**Authors**: Linyuan Gong, Alvin Cheung, Mostafa Elhoushi, Sida Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00204)  

**Abstract**: Fill-in-the-Middle (FIM) is a common pretraining method for code LLMs, where models complete code segments given surrounding context. However, existing LLMs treat code as plain text and mask random character spans. We propose and evaluate AST-FIM, a pretraining strategy that leverages Abstract Syntax Trees (ASTs) to mask complete syntactic structures at scale, ensuring coherent training examples better aligned with universal code structures and common code editing patterns such as blocks, expressions, or functions. To evaluate real-world fill-in-the-middle (FIM) programming tasks, we introduce Real-FIM-Eval, a benchmark derived from 30,000+ GitHub commits across 12 languages. On infilling tasks, experiments on 1B and 8B parameter models show that AST-FIM is particularly beneficial for real-world code editing as it outperforms standard random-character FIM by up to 5 pts on standard FIM benchmarks. Our code is publicly available at this https URL. 

**Abstract (ZH)**: Fill-in-the-Middle (FIM)是一种常见的代码LLM预训练方法，其中模型根据上下文完成代码片段。然而，现有的LLM将代码视为普通文本，并随机屏蔽字符跨度。我们提出并评估了AST-FIM，这是一种利用抽象语法树（AST）按规模屏蔽完整句法结构的预训练策略，确保更加连贯的训练示例，更好地与通用代码结构和常见的代码编辑模式（如代码块、表达式或函数）对齐。为了评估真实的填空（FIM）编程任务，我们引入了Real-FIM-Eval基准，该基准源于来自12种语言的30,000多个GitHub提交记录。在填充任务中，对于1B和8B参数模型的实验表明，AST-FIM特别适用于真实的代码编辑任务，在标准FIM基准上，AST-FIM相对于标准随机字符FIM最多可提高5个点。我们的代码已在此处公开：this https URL。 

---
# The World As Large Language Models See It: Exploring the reliability of LLMs in representing geographical features 

**Title (ZH)**: 大型语言模型眼中的世界：探索LLMs在表示地理特征方面的可靠性 

**Authors**: Omid Reza Abbasi, Franz Welscher, Georg Weinberger, Johannes Scholz  

**Link**: [PDF](https://arxiv.org/pdf/2506.00203)  

**Abstract**: As large language models (LLMs) continue to evolve, questions about their trustworthiness in delivering factual information have become increasingly important. This concern also applies to their ability to accurately represent the geographic world. With recent advancements in this field, it is relevant to consider whether and to what extent LLMs' representations of the geographical world can be trusted. This study evaluates the performance of GPT-4o and Gemini 2.0 Flash in three key geospatial tasks: geocoding, elevation estimation, and reverse geocoding. In the geocoding task, both models exhibited systematic and random errors in estimating the coordinates of St. Anne's Column in Innsbruck, Austria, with GPT-4o showing greater deviations and Gemini 2.0 Flash demonstrating more precision but a significant systematic offset. For elevation estimation, both models tended to underestimate elevations across Austria, though they captured overall topographical trends, and Gemini 2.0 Flash performed better in eastern regions. The reverse geocoding task, which involved identifying Austrian federal states from coordinates, revealed that Gemini 2.0 Flash outperformed GPT-4o in overall accuracy and F1-scores, demonstrating better consistency across regions. Despite these findings, neither model achieved an accurate reconstruction of Austria's federal states, highlighting persistent misclassifications. The study concludes that while LLMs can approximate geographic information, their accuracy and reliability are inconsistent, underscoring the need for fine-tuning with geographical information to enhance their utility in GIScience and Geoinformatics. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的不断进化，其在传递事实信息方面的可信度问题变得越来越重要。这一问题也涉及到它们准确表示地理世界的能力。随着该领域最近的发展，考虑LLMs在表示地理世界方面的可信度及其程度变得相关。本研究评估了GPT-4o和Gemini 2.0 Flash在三个关键的地理空间任务中的性能：地理编码、高程估计和逆地理编码。在地理编码任务中，两种模型在估算奥地利因斯布鲁克圣安妮柱的坐标时都出现了系统性和随机性误差，GPT-4o显示出更大的偏差，而Gemini 2.0 Flash则表现出更高的精度但存在显著的系统性偏差。对于高程估计，两种模型倾向于低估奥地利各地的高程，但它们捕捉了总体地形趋势，Gemini 2.0 Flash在东部地区表现更好。逆地理编码任务涉及从坐标识别奥地利联邦州，结果显示Gemini 2.0 Flash在总体准确性和F1分数上优于GPT-4o，显示出更好的区域一致性。尽管如此，这两种模型都无法准确重建奥地利的联邦州，突显出了持续存在的分类错误。研究结论认为，虽然LLMs可以近似地理信息，但其准确性和可靠性存在不一致性，强调了需要根据地理信息进行微调以增强其在地理信息系统（GIScience）和地理信息系统学（Geoinformatics）中的应用价值。 

---
# MOFGPT: Generative Design of Metal-Organic Frameworks using Language Models 

**Title (ZH)**: MOFGPT：使用语言模型设计金属有机框架 

**Authors**: Srivathsan Badrinarayanan, Rishikesh Magar, Akshay Antony, Radheesh Sharma Meda, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2506.00198)  

**Abstract**: The discovery of Metal-Organic Frameworks (MOFs) with application-specific properties remains a central challenge in materials chemistry, owing to the immense size and complexity of their structural design space. Conventional computational screening techniques such as molecular simulations and density functional theory (DFT), while accurate, are computationally prohibitive at scale. Machine learning offers an exciting alternative by leveraging data-driven approaches to accelerate materials discovery. The complexity of MOFs, with their extended periodic structures and diverse topologies, creates both opportunities and challenges for generative modeling approaches. To address these challenges, we present a reinforcement learning-enhanced, transformer-based framework for the de novo design of MOFs. Central to our approach is MOFid, a chemically-informed string representation encoding both connectivity and topology, enabling scalable generative modeling. Our pipeline comprises three components: (1) a generative GPT model trained on MOFid sequences, (2) MOFormer, a transformer-based property predictor, and (3) a reinforcement learning (RL) module that optimizes generated candidates via property-guided reward functions. By integrating property feedback into sequence generation, our method drives the model toward synthesizable, topologically valid MOFs with desired functional attributes. This work demonstrates the potential of large language models, when coupled with reinforcement learning, to accelerate inverse design in reticular chemistry and unlock new frontiers in computational MOF discovery. 

**Abstract (ZH)**: 具有应用特定性质的金属-有机框架（MOFs）的发现仍然是材料化学领域的核心挑战，由于其结构设计空间庞大且复杂。传统的计算筛选技术如分子模拟和密度泛函理论（DFT）虽然准确，但在大规模应用时计算成本高昂。机器学习通过利用数据驱动的方法加速材料发现提供了令人兴奋的替代方案。MOFs的复杂性，包括其拓展的周期结构和多样的拓扑，为生成模型方法带来了机会与挑战。为应对这些挑战，我们提出了一种增强学习增强的基于变换器的框架，用于从头设计MOFs。我们方法的核心是MOFid，一种化学启发的字符串表示，编码连接性和拓扑，实现可扩展的生成建模。我们的管道包括三个组件：（1）一个训练于MOFid序列的生成GPT模型，（2）MOFormer，一种基于变换器的性质预测器，以及（3）一个基于属性导向奖励函数的强化学习（RL）模块，优化生成的候选物。通过将属性反馈集成到序列生成中，我们的方法引导模型向合成可行且拓扑有效的MOFs及其所需功能属性的方向发展。本工作展示了大型语言模型与强化学习结合在晶格化学中的逆向设计加速方面以及在计算MOF发现中的潜在价值，开启了新的前沿领域。 

---
# Let Them Down Easy! Contextual Effects of LLM Guardrails on User Perceptions and Preferences 

**Title (ZH)**: 轻松地让他们失望！LLM 边界对用户感知和偏好的情境影响 

**Authors**: Mingqian Zheng, Wenjia Hu, Patrick Zhao, Motahhare Eslami, Jena D. Hwang, Faeze Brahman, Carolyn Rose, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2506.00195)  

**Abstract**: Current LLMs are trained to refuse potentially harmful input queries regardless of whether users actually had harmful intents, causing a tradeoff between safety and user experience. Through a study of 480 participants evaluating 3,840 query-response pairs, we examine how different refusal strategies affect user perceptions across varying motivations. Our findings reveal that response strategy largely shapes user experience, while actual user motivation has negligible impact. Partial compliance -- providing general information without actionable details -- emerges as the optimal strategy, reducing negative user perceptions by over 50% to flat-out refusals. Complementing this, we analyze response patterns of 9 state-of-the-art LLMs and evaluate how 6 reward models score different refusal strategies, demonstrating that models rarely deploy partial compliance naturally and reward models currently undervalue it. This work demonstrates that effective guardrails require focusing on crafting thoughtful refusals rather than detecting intent, offering a path toward AI safety mechanisms that ensure both safety and sustained user engagement. 

**Abstract (ZH)**: 当前的大型语言模型在训练时倾向于拒绝可能有害的输入查询，而不论用户实际是否有害意图，这导致了安全性和用户体验之间的权衡。通过对480名参与者评价3,840个查询-响应对的研究，我们探讨了不同拒绝策略如何影响不同动机下的用户感知。研究发现，响应策略极大地影响用户体验，而实际用户动机的影响甚微。部分合规——提供一般信息而不提供具体操作细节——被认为是最佳策略，可将负面用户感知降低超过50%，接近于完全拒绝。此外，我们还分析了9个最先进大型语言模型的响应模式，并评估了6种奖励模型对不同拒绝策略的评分，发现模型很少自然地采用部分合规，并且当前的奖励模型对其评价不足。本文表明，有效的护栏需要侧重于精心设计的拒绝策略而不是意图检测，为确保安全性的同时维持用户参与提供了一条途径。 

---
# Supporting architecture evaluation for ATAM scenarios with LLMs 

**Title (ZH)**: 使用大语言模型支持ATAM场景下的架构评估 

**Authors**: Rafael Capilla, J. Andrés Díaz-Pace, Yamid Ramírez, Jennifer Pérez, Vanessa Rodríguez-Horcajo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00150)  

**Abstract**: Architecture evaluation methods have long been used to evaluate software designs. Several evaluation methods have been proposed and used to analyze tradeoffs between different quality attributes. Having competing qualities leads to conflicts for selecting which quality-attribute scenarios are the most suitable ones that an architecture should tackle and for prioritizing the scenarios required by the stakeholders. In this context, architecture evaluation is carried out manually, often involving long brainstorming sessions to decide which are the most adequate quality scenarios. To reduce this effort and make the assessment and selection of scenarios more efficient, we suggest the usage of LLMs to partially automate evaluation activities. As a first step to validate this hypothesis, this work studies MS Copilot as an LLM tool to analyze quality scenarios suggested by students in a software architecture course and compares the students' results with the assessment provided by the LLM. Our initial study reveals that the LLM produces in most cases better and more accurate results regarding the risks, sensitivity points and tradeoff analysis of the quality scenarios. Overall, the use of generative AI has the potential to partially automate and support the architecture evaluation tasks, improving the human decision-making process. 

**Abstract (ZH)**: 使用生成式AI部分自动化软件架构评估任务的研究 

---
# Spurious Correlations and Beyond: Understanding and Mitigating Shortcut Learning in SDOH Extraction with Large Language Models 

**Title (ZH)**: 虚假相关性及其超越：理解并缓解大规模语言模型在社会决定因素提取中的捷径学习问题 

**Authors**: Fardin Ahsan Sakib, Ziwei Zhu, Karen Trister Grace, Meliha Yetisgen, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2506.00134)  

**Abstract**: Social determinants of health (SDOH) extraction from clinical text is critical for downstream healthcare analytics. Although large language models (LLMs) have shown promise, they may rely on superficial cues leading to spurious predictions. Using the MIMIC portion of the SHAC (Social History Annotation Corpus) dataset and focusing on drug status extraction as a case study, we demonstrate that mentions of alcohol or smoking can falsely induce models to predict current/past drug use where none is present, while also uncovering concerning gender disparities in model performance. We further evaluate mitigation strategies - such as prompt engineering and chain-of-thought reasoning - to reduce these false positives, providing insights into enhancing LLM reliability in health domains. 

**Abstract (ZH)**: 社会决定因素对健康的影响从临床文本中提取对于下游医疗数据分析至关重要。尽管大型语言模型显示出潜力，但它们可能会依赖于表面线索导致虚假预测。通过使用SHAC数据集中MIMIC部分和以药物状态提取为例进行研究，我们证明了提及酒精或吸烟可能会误导模型预测不存在的当前/过去用药情况，同时还揭示了模型性能中的性别差异问题。进一步评估了诸如提示工程和链式思考推理等缓解策略，以减少这些误报，为增强医疗领域大型语言模型的可靠性提供了见解。 

---
# ClinBench-HPB: A Clinical Benchmark for Evaluating LLMs in Hepato-Pancreato-Biliary Diseases 

**Title (ZH)**: ClinBench-HPB：肝胆胰疾病领域评估LLMs的临床基准 

**Authors**: Yuchong Li, Xiaojun Zeng, Chihua Fang, Jian Yang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00095)  

**Abstract**: Hepato-pancreato-biliary (HPB) disorders represent a global public health challenge due to their high morbidity and mortality. Although large language models (LLMs) have shown promising performance in general medical question-answering tasks, the current evaluation benchmarks are mostly derived from standardized examinations or manually designed questions, lacking HPB coverage and clinical cases. To address these issues, we systematically eatablish an HPB disease evaluation benchmark comprising 3,535 closed-ended multiple-choice questions and 337 open-ended real diagnosis cases, which encompasses all the 33 main categories and 465 subcategories of HPB diseases defined in the International Statistical Classification of Diseases, 10th Revision (ICD-10). The multiple-choice questions are curated from public datasets and synthesized data, and the clinical cases are collected from prestigious medical journals, case-sharing platforms, and collaborating hospitals. By evalauting commercial and open-source general and medical LLMs on our established benchmark, namely ClinBench-HBP, we find that while commercial LLMs perform competently on medical exam questions, they exhibit substantial performance degradation on HPB diagnosis tasks, especially on complex, inpatient clinical cases. Those medical LLMs also show limited generalizability to HPB diseases. Our results reveal the critical limitations of current LLMs in the domain of HPB diseases, underscoring the imperative need for future medical LLMs to handle real, complex clinical diagnostics rather than simple medical exam questions. The benchmark will be released at the homepage. 

**Abstract (ZH)**: 肝-胰-胆道（HPB）疾病评估基准：一种涵盖全球高发病率和死亡率疾病的综合性评估标准 

---
# TRAPDOC: Deceiving LLM Users by Injecting Imperceptible Phantom Tokens into Documents 

**Title (ZH)**: TRAPDOC: 通过向文档中注入不可感知的幽灵标记欺骗LLM用户 

**Authors**: Hyundong Jin, Sicheol Sung, Shinwoo Park, SeungYeop Baik, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.00089)  

**Abstract**: The reasoning, writing, text-editing, and retrieval capabilities of proprietary large language models (LLMs) have advanced rapidly, providing users with an ever-expanding set of functionalities. However, this growing utility has also led to a serious societal concern: the over-reliance on LLMs. In particular, users increasingly delegate tasks such as homework, assignments, or the processing of sensitive documents to LLMs without meaningful engagement. This form of over-reliance and misuse is emerging as a significant social issue. In order to mitigate these issues, we propose a method injecting imperceptible phantom tokens into documents, which causes LLMs to generate outputs that appear plausible to users but are in fact incorrect. Based on this technique, we introduce TRAPDOC, a framework designed to deceive over-reliant LLM users. Through empirical evaluation, we demonstrate the effectiveness of our framework on proprietary LLMs, comparing its impact against several baselines. TRAPDOC serves as a strong foundation for promoting more responsible and thoughtful engagement with language models. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理、写作、文本编辑和检索能力迅速提升，为用户提供了日益丰富的功能。然而，这种功能的增加也引发了一个严重的问题：对LLMs的过度依赖。特别是，用户越来越多地将任务如家庭作业、作业或敏感文档的处理交给LLMs，而缺乏有意义的参与。这种过度依赖和误用正逐渐成为一个重要的社会问题。为了缓解这些问题，我们提出一种方法，在文档中注入不可感知的幻影令牌，使LLMs生成看似合理的但实际上是错误的输出。基于这一技术，我们引入了TRAPDOC框架，旨在欺骗过度依赖的LLM用户。通过实证评估，我们展示了该框架在专用LLMs上的有效性，并将其影响与若干基线进行比较。TRAPDOC为促进更负责任和深思熟虑的语言模型使用提供了坚实的基础。我们的代码可在以下网址获得。 

---
# HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs 

**Title (ZH)**: HD-NDEs: 神经微分方程在大语言模型幻觉检测中的应用 

**Authors**: Qing Li, Jiahui Geng, Zongxiong Chen, Derui Zhu, Yuxia Wang, Congbo Ma, Chenyang Lyu, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2506.00088)  

**Abstract**: In recent years, large language models (LLMs) have made remarkable advancements, yet hallucination, where models produce inaccurate or non-factual statements, remains a significant challenge for real-world deployment. Although current classification-based methods, such as SAPLMA, are highly efficient in mitigating hallucinations, they struggle when non-factual information arises in the early or mid-sequence of outputs, reducing their reliability. To address these issues, we propose Hallucination Detection-Neural Differential Equations (HD-NDEs), a novel method that systematically assesses the truthfulness of statements by capturing the full dynamics of LLMs within their latent space. Our approaches apply neural differential equations (Neural DEs) to model the dynamic system in the latent space of LLMs. Then, the sequence in the latent space is mapped to the classification space for truth assessment. The extensive experiments across five datasets and six widely used LLMs demonstrate the effectiveness of HD-NDEs, especially, achieving over 14% improvement in AUC-ROC on the True-False dataset compared to state-of-the-art techniques. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）取得了显著进展，但 hallucination 问题——模型生成不准确或非事实的陈述——仍然是实际部署中的一个重要挑战。尽管目前基于分类的方法，如 SAPLMA，在减轻 hallucination 方面非常高效，但在输出序列的早期或中期出现非事实信息时，它们的可靠性会降低。为了解决这些问题，我们提出了一种新颖的方法 Hallucination Detection-Neural Differential Equations (HD-NDEs)，该方法通过捕捉 LLMs 在潜在空间中的完整动态来系统地评估陈述的真实性。我们的方法利用神经微分方程（Neural DEs）来建模 LLMs 潜在空间中的动态系统，然后将潜在空间中的序列映射到分类空间进行真实性评估。在五个数据集和六种广泛使用的 LLM 上进行的广泛实验表明，HD-NDEs 的有效性，特别是在 True-False 数据集上 AUC-ROC 指标上取得超过 14% 的改进，优于最先进的技术。 

---
# COSMIC: Generalized Refusal Direction Identification in LLM Activations 

**Title (ZH)**: COSMIC: LLM激活中泛化的拒绝方向识别 

**Authors**: Vincent Siu, Nicholas Crispino, Zihao Yu, Sam Pan, Zhun Wang, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00085)  

**Abstract**: Large Language Models (LLMs) encode behaviors such as refusal within their activation space, yet identifying these behaviors remains a significant challenge. Existing methods often rely on predefined refusal templates detectable in output tokens or require manual analysis. We introduce \textbf{COSMIC} (Cosine Similarity Metrics for Inversion of Concepts), an automated framework for direction selection that identifies viable steering directions and target layers using cosine similarity - entirely independent of model outputs. COSMIC achieves steering performance comparable to prior methods without requiring assumptions about a model's refusal behavior, such as the presence of specific refusal tokens. It reliably identifies refusal directions in adversarial settings and weakly aligned models, and is capable of steering such models toward safer behavior with minimal increase in false refusals, demonstrating robustness across a wide range of alignment conditions. 

**Abstract (ZH)**: Large Language Models (LLMs)中的拒绝行为编码在其激活空间内，但识别这些行为仍是重大挑战。现有方法通常依赖于预定义的拒绝模板来检测输出标记，或者需要人工分析。我们介绍了COSMIC（余弦相似度指标的概念反转），一种自动方向选择框架，该框架利用余弦相似度来识别可行的控制方向和目标层，完全不依赖于模型输出。COSMIC在不需要假设模型的拒绝行为（如特定拒绝标记的存在）的情况下，实现了与先前方法相当的控制性能。它能够在对抗性设置和弱对齐模型中可靠地识别拒绝方向，并能够以最小的误拒绝增加将此类模型引导至更安全的行为，展示了在广泛对齐条件下的稳健性。 

---
# Artificial Empathy: AI based Mental Health 

**Title (ZH)**: 人工共情：基于AI的心理健康 

**Authors**: Aditya Naik, Jovi Thomas, Teja Sree, Himavant Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.00081)  

**Abstract**: Many people suffer from mental health problems but not everyone seeks professional help or has access to mental health care. AI chatbots have increasingly become a go-to for individuals who either have mental disorders or simply want someone to talk to. This paper presents a study on participants who have previously used chatbots and a scenario-based testing of large language model (LLM) chatbots. Our findings indicate that AI chatbots were primarily utilized as a "Five minute therapist" or as a non-judgmental companion. Participants appreciated the anonymity and lack of judgment from chatbots. However, there were concerns about privacy and the security of sensitive information. The scenario-based testing of LLM chatbots highlighted additional issues. Some chatbots were consistently reassuring, used emojis and names to add a personal touch, and were quick to suggest seeking professional help. However, there were limitations such as inconsistent tone, occasional inappropriate responses (e.g., casual or romantic), and a lack of crisis sensitivity, particularly in recognizing red flag language and escalating responses appropriately. These findings can inform both the technology and mental health care industries on how to better utilize AI chatbots to support individuals during challenging emotional periods. 

**Abstract (ZH)**: AI聊天机器人在心理健康支持中的应用与挑战：基于场景的大型语言模型测试研究 

---
# Who Gets the Kidney? Human-AI Alignment, Indecision, and Moral Values 

**Title (ZH)**: 谁获得肾脏？人类与人工智能的aligniment、犹豫与道德价值观 

**Authors**: John P. Dickerson, Hadi Hosseini, Samarth Khanna, Leona Pierce  

**Link**: [PDF](https://arxiv.org/pdf/2506.00079)  

**Abstract**: The rapid integration of Large Language Models (LLMs) in high-stakes decision-making -- such as allocating scarce resources like donor organs -- raises critical questions about their alignment with human moral values. We systematically evaluate the behavior of several prominent LLMs against human preferences in kidney allocation scenarios and show that LLMs: i) exhibit stark deviations from human values in prioritizing various attributes, and ii) in contrast to humans, LLMs rarely express indecision, opting for deterministic decisions even when alternative indecision mechanisms (e.g., coin flipping) are provided. Nonetheless, we show that low-rank supervised fine-tuning with few samples is often effective in improving both decision consistency and calibrating indecision modeling. These findings illustrate the necessity of explicit alignment strategies for LLMs in moral/ethical domains. 

**Abstract (ZH)**: 大型语言模型在高stakes决策中的快速集成——如分配稀缺资源（如捐赠器官）——引起对其与人类道德价值一致性的关键问题。我们系统评估了几种 prominant LLMs 在肾脏分配情景中对人类偏好行为，并表明：i) LLMs 在优先处理各种属性方面表现出与人类价值观的明显偏离，ii) 与人类不同，LLMs 很少表现出犹豫不决，倾向于作出确定性决策，即使提供了替代的犹豫机制（如抛硬币）也是如此。然而，我们显示，使用少量样本的低秩监督微调通常能有效提高决策一致性并校准犹豫建模。这些发现表明，在道德/伦理领域中，需要明确的对齐策略以确保大型语言模型的一致性。 

---
# Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations 

**Title (ZH)**: whose Name comes up? 审核基于LLM的学者推荐 

**Authors**: Daniele Barolo, Chiara Valentin, Fariba Karimi, Luis Galárraga, Gonzalo G. Méndez, Lisette Espín-Noboa  

**Link**: [PDF](https://arxiv.org/pdf/2506.00074)  

**Abstract**: This paper evaluates the performance of six open-weight LLMs (llama3-8b, llama3.1-8b, gemma2-9b, mixtral-8x7b, llama3-70b, llama3.1-70b) in recommending experts in physics across five tasks: top-k experts by field, influential scientists by discipline, epoch, seniority, and scholar counterparts. The evaluation examines consistency, factuality, and biases related to gender, ethnicity, academic popularity, and scholar similarity. Using ground-truth data from the American Physical Society and OpenAlex, we establish scholarly benchmarks by comparing model outputs to real-world academic records. Our analysis reveals inconsistencies and biases across all models. mixtral-8x7b produces the most stable outputs, while llama3.1-70b shows the highest variability. Many models exhibit duplication, and some, particularly gemma2-9b and llama3.1-8b, struggle with formatting errors. LLMs generally recommend real scientists, but accuracy drops in field-, epoch-, and seniority-specific queries, consistently favoring senior scholars. Representation biases persist, replicating gender imbalances (reflecting male predominance), under-representing Asian scientists, and over-representing White scholars. Despite some diversity in institutional and collaboration networks, models favor highly cited and productive scholars, reinforcing the rich-getricher effect while offering limited geographical representation. These findings highlight the need to improve LLMs for more reliable and equitable scholarly recommendations. 

**Abstract (ZH)**: 本研究评估了六种开放权重的语言模型（llama3-8b、llama3.1-8b、gemma2-9b、mixtral-8x7b、llama3-70b、llama3.1-70b）在物理学领域推荐专家方面的性能，并在五个任务中进行了跨学科、影响力、时期、任期和同事方面的专家推荐：顶级专家推荐、学科有影响力的科学家推荐、时期推荐、任期推荐和学者同行推荐。评估检查了性别、种族、学术受欢迎程度和学者相似性相关的连贯性、事实性和偏见。使用美国物理学会和OpenAlex的真实数据，通过将模型输出与实际学术记录进行比较，建立了学术基准。分析结果表明，所有模型都存在不一致性和偏见。mixtral-8x7b产生最稳定的输出，而llama3.1-70b表现出最高的变异性。许多模型存在重复现象，特别是gemma2-9b和llama3.1-8b在格式错误方面存在问题。语言模型通常推荐真实的科学家，但在特定领域、时期和任期查询中的准确性下降，始终倾向于推荐资深学者。代表性偏见持续存在，复制性别失衡（反映男性主导现象）、低估亚裔科学家并高估白人学者。尽管机构和合作网络中存在一定程度的多样性，但模型倾向于推荐高被引和高产的学者，从而强化了“赢者通吃”效应，同时地理代表性有限。这些发现强调了改进语言模型以提供更多可靠和公平的学术推荐的必要性。 

---
# Evaluating Prompt Engineering Techniques for Accuracy and Confidence Elicitation in Medical LLMs 

**Title (ZH)**: 评估提示工程技术在医疗LLM中提高准确性和信心的效果 

**Authors**: Nariman Naderi, Zahra Atf, Peter R Lewis, Aref Mahjoub far, Seyed Amir Ahmad Safavi-Naini, Ali Soroush  

**Link**: [PDF](https://arxiv.org/pdf/2506.00072)  

**Abstract**: This paper investigates how prompt engineering techniques impact both accuracy and confidence elicitation in Large Language Models (LLMs) applied to medical contexts. Using a stratified dataset of Persian board exam questions across multiple specialties, we evaluated five LLMs - GPT-4o, o3-mini, Llama-3.3-70b, Llama-3.1-8b, and DeepSeek-v3 - across 156 configurations. These configurations varied in temperature settings (0.3, 0.7, 1.0), prompt styles (Chain-of-Thought, Few-Shot, Emotional, Expert Mimicry), and confidence scales (1-10, 1-100). We used AUC-ROC, Brier Score, and Expected Calibration Error (ECE) to evaluate alignment between confidence and actual performance. Chain-of-Thought prompts improved accuracy but also led to overconfidence, highlighting the need for calibration. Emotional prompting further inflated confidence, risking poor decisions. Smaller models like Llama-3.1-8b underperformed across all metrics, while proprietary models showed higher accuracy but still lacked calibrated confidence. These results suggest prompt engineering must address both accuracy and uncertainty to be effective in high-stakes medical tasks. 

**Abstract (ZH)**: 本文探讨了提示工程技术如何影响大规模语言模型（LLMs）在医疗情境下准确性和置信度的生成。利用跨多个专科的分层波斯语执业考试问题数据集，我们评估了五种LLM——GPT-4o、o3-mini、Llama-3.3-70b、Llama-3.1-8b和DeepSeek-v3，在156种不同配置下的表现。这些配置在温度设置（0.3、0.7、1.0）、提示风格（推理链、少样本、情感、专家模仿）和置信度尺度（1-10、1-100）方面有所不同。我们使用AUC-ROC、贝叶斯得分和预期校准误差（ECE）来评估信心和实际性能之间的对齐情况。推理链提示提高了准确性，但也导致了过度自信，突显了校准的需求。情感提示进一步夸大了信心，增加了做错决定的风险。小型模型如Llama-3.1-8b在所有指标上表现不佳，而专有模型则显示出更高的准确性，但在校准信心方面仍然不足。这些结果表明，在高风险医疗任务中，提示工程技术必须同时解决准确性和不确定性才能有效。 

---
# Evaluating the Sensitivity of LLMs to Prior Context 

**Title (ZH)**: 评估大语言模型对先验上下文的敏感性 

**Authors**: Robert Hankache, Kingsley Nketia Acheampong, Liang Song, Marek Brynda, Raad Khraishi, Greig A. Cowan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00069)  

**Abstract**: As large language models (LLMs) are increasingly deployed in multi-turn dialogue and other sustained interactive scenarios, it is essential to understand how extended context affects their performance. Popular benchmarks, focusing primarily on single-turn question answering (QA) tasks, fail to capture the effects of multi-turn exchanges. To address this gap, we introduce a novel set of benchmarks that systematically vary the volume and nature of prior context. We evaluate multiple conventional LLMs, including GPT, Claude, and Gemini, across these benchmarks to measure their sensitivity to contextual variations. Our findings reveal that LLM performance on multiple-choice questions can degrade dramatically in multi-turn interactions, with performance drops as large as 73% for certain models. Even highly capable models such as GPT-4o exhibit up to a 32% decrease in accuracy. Notably, the relative performance of larger versus smaller models is not always predictable. Moreover, the strategic placement of the task description within the context can substantially mitigate performance drops, improving the accuracy by as much as a factor of 3.5. These findings underscore the need for robust strategies to design, evaluate, and mitigate context-related sensitivity in LLMs. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在多轮对话和其他持续交互场景中的应用日益增多，理解扩展上下文对其性能的影响变得至关重要。主要集中在单轮问答任务的流行基准无法捕捉多轮交互的效果。为填补这一空白，我们引入了一组新的基准测试，系统地变化预设上下文的数量和性质。我们评估了多个传统的LLM模型，包括GPT、Claude和Gemini，以测量它们对上下文变化的敏感性。研究发现，某些模型在多选题上的表现可能会在多轮互动中显著下降，性能降幅高达73%。即使是像GPT-4o这样强大的模型，其准确性也会下降多达32%。值得注意的是，不同规模模型之间的相对性能并不总是可预测的。此外，任务描述在上下文中的战略位置可以显著减轻性能下降，提高准确性多达3.5倍。这些发现强调了制定、评估和减轻LLM上下文相关敏感性的稳健策略的必要性。 

---
# Probing Politico-Economic Bias in Multilingual Large Language Models: A Cultural Analysis of Low-Resource Pakistani Languages 

**Title (ZH)**: 探究多语言大型语言模型中的政治经济偏见：低资源旁遮普语言的文化分析 

**Authors**: Afrozah Nadeem, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.00068)  

**Abstract**: Large Language Models (LLMs) are increasingly shaping public discourse, yet their politico-economic biases remain underexamined in non-Western and low-resource multilingual contexts. This paper presents a systematic analysis of political bias in 13 state-of-the-art LLMs across five low-resource languages spoken in Pakistan: Urdu, Punjabi, Sindhi, Balochi, and Pashto. We propose a novel framework that integrates an adapted Political Compass Test (PCT) with a multi-level framing analysis. Our method combines quantitative assessment of political orientation across economic (left-right) and social (libertarian-authoritarian) axes with qualitative analysis of framing through content, style, and emphasis. We further contextualize this analysis by aligning prompts with 11 key socio-political themes relevant to Pakistani society. Our results reveal that LLMs predominantly align with liberal-left values, echoing Western training data influences, but exhibit notable shifts toward authoritarian framing in regional languages, suggesting strong cultural modulation effects. We also identify consistent model-specific bias signatures and language-conditioned variations in ideological expression. These findings show the urgent need for culturally grounded, multilingual bias auditing frameworks. 

**Abstract (ZH)**: 大型语言模型（LLMs） increasingly shaping public discourse: Underexamined politico-economic biases in non-Western and low-resource multilingual contexts - A systematic analysis of political bias in 13 state-of-the-art LLMs across five low-resource languages spoken in Pakistan: Urdu, Punjabi, Sindhi, Balochi, and Pashto 

---
# Literature Review Of Multi-Agent Debate For Problem-Solving 

**Title (ZH)**: 多Agent辩论解决问题领域的文献综述 

**Authors**: Arne Tillmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.00066)  

**Abstract**: Multi-agent large language models (MA-LLMs) are a rapidly growing research area that leverages multiple interacting language agents to tackle complex tasks, outperforming single-agent large language models. This literature review synthesizes the latest research on agent profiles, communication structures, and decision-making processes, drawing insights from both traditional multi-agent systems and state-of-the-art MA-LLM studies. In doing so, it aims to address the lack of direct comparisons in the field, illustrating how factors like scalability, communication structure, and decision-making processes influence MA-LLM performance. By examining frequent practices and outlining current challenges, the review reveals that multi-agent approaches can yield superior results but also face elevated computational costs and under-explored challenges unique to MA-LLM. Overall, these findings provide researchers and practitioners with a roadmap for developing robust and efficient multi-agent AI solutions. 

**Abstract (ZH)**: 多智能体大型语言模型（MA-LLMs）是一个快速增长的研究领域，利用多个互动的语言智能体解决复杂任务，并在性能上超过了单智能体大型语言模型。本文综述了最新的有关智能体特征、通信结构和决策过程的研究，借鉴了传统多智能体系统和最先进的MA-LLM研究的见解。通过这样做，本文旨在解决该领域中存在的直接比较不足的问题，说明诸如可扩展性、通信结构和决策过程等因素如何影响MA-LLM的表现。通过对常见实践的分析和当前挑战的概述，本文揭示了多智能体方法可以产生更优异的结果，但也面临着较高的计算成本和MA-LLM特有的未充分探索的挑战。总体而言，这些发现为研究人员和实践者提供了开发稳健且高效的多智能体AI解决方案的指南。 

---
# Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling 

**Title (ZH)**: 错觉提示：大型语言模型在先行错误处理方面的基准测试 

**Authors**: Jiayi Zeng, Yizhe Feng, Mengliang He, Wenhui Lei, Wei Zhang, Zeming Liu, Xiaoming Shi, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00064)  

**Abstract**: Large language models (LLMs) have demonstrated significant advancements in error handling. Current error-handling works are performed in a passive manner, with explicit error-handling instructions. However, in real-world scenarios, explicit error-handling instructions are usually unavailable. In this paper, our work identifies this challenge as how to conduct proactive error handling without explicit error handling instructions. To promote further research, this work introduces a new benchmark, termed Mis-prompt, consisting of four evaluation tasks, an error category taxonomy, and a new evaluation dataset. Furthermore, this work analyzes current LLMs' performance on the benchmark, and the experimental results reveal that current LLMs show poor performance on proactive error handling, and SFT on error handling instances improves LLMs' proactive error handling capabilities. The dataset will be publicly available. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在错误处理方面展示了显著的进步。当前的错误处理工作通常是被动的，并且依赖于明确的错误处理指令。然而，在实际场景中，明确的错误处理指令往往是不可用的。本文将这一挑战定义为如何在没有明确错误处理指令的情况下进行主动错误处理。为了促进进一步的研究，本文介绍了一个新的基准，称为Mis-prompt，其中包括四个评估任务、一个错误类别分类体系和一个新的评估数据集。此外，本文分析了当前LLMs在该基准上的表现，实验结果表明当前的LLMs在主动错误处理方面表现不佳，通过错误处理实例的精细调节（SFT）可以提高LLMs的主动错误处理能力。数据集将公开可用。 

---
# Unraveling SITT: Social Influence Technique Taxonomy and Detection with LLMs 

**Title (ZH)**: 揭示SITT：社会影响技术分类与检测——基于LLMs的方法 

**Authors**: Wiktoria Mieleszczenko-Kowszewicz, Beata Bajcar, Aleksander Szczęsny, Maciej Markiewicz, Jolanta Babiak, Berenika Dyczek, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2506.00061)  

**Abstract**: In this work we present the Social Influence Technique Taxonomy (SITT), a comprehensive framework of 58 empirically grounded techniques organized into nine categories, designed to detect subtle forms of social influence in textual content. We also investigate the LLMs ability to identify various forms of social influence. Building on interdisciplinary foundations, we construct the SITT dataset -- a 746-dialogue corpus annotated by 11 experts in Polish and translated into English -- to evaluate the ability of LLMs to identify these techniques. Using a hierarchical multi-label classification setup, we benchmark five LLMs, including GPT-4o, Claude 3.5, Llama-3.1, Mixtral, and PLLuM. Our results show that while some models, notably Claude 3.5, achieved moderate success (F1 score = 0.45 for categories), overall performance of models remains limited, particularly for context-sensitive techniques. The findings demonstrate key limitations in current LLMs' sensitivity to nuanced linguistic cues and underscore the importance of domain-specific fine-tuning. This work contributes a novel resource and evaluation example for understanding how LLMs detect, classify, and potentially replicate strategies of social influence in natural dialogues. 

**Abstract (ZH)**: 社会影响技术分类税onomies：一种全面的框架，用于检测文本内容中的微妙社会影响形式，并评估LLMs的能力 

---
# Comparative analysis of privacy-preserving open-source LLMs regarding extraction of diagnostic information from clinical CMR imaging reports 

**Title (ZH)**: 开放源代码诊断型语言模型在保留隐私方面的临床CMR影像报告诊断信息提取 Comparative Analysis 

**Authors**: Sina Amirrajab, Volker Vehof, Michael Bietenbeck, Ali Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.00060)  

**Abstract**: Purpose: We investigated the utilization of privacy-preserving, locally-deployed, open-source Large Language Models (LLMs) to extract diagnostic information from free-text cardiovascular magnetic resonance (CMR) reports. Materials and Methods: We evaluated nine open-source LLMs on their ability to identify diagnoses and classify patients into various cardiac diagnostic categories based on descriptive findings in 109 clinical CMR reports. Performance was quantified using standard classification metrics including accuracy, precision, recall, and F1 score. We also employed confusion matrices to examine patterns of misclassification across models. Results: Most open-source LLMs demonstrated exceptional performance in classifying reports into different diagnostic categories. Google's Gemma2 model achieved the highest average F1 score of 0.98, followed by Qwen2.5:32B and DeepseekR1-32B with F1 scores of 0.96 and 0.95, respectively. All other evaluated models attained average scores above 0.93, with Mistral and DeepseekR1-7B being the only exceptions. The top four LLMs outperformed our board-certified cardiologist (F1 score of 0.94) across all evaluation metrics in analyzing CMR reports. Conclusion: Our findings demonstrate the feasibility of implementing open-source, privacy-preserving LLMs in clinical settings for automated analysis of imaging reports, enabling accurate, fast and resource-efficient diagnostic categorization. 

**Abstract (ZH)**: 目的：我们调查了利用隐私保护、本地部署的开源大规模语言模型（LLMs）从自由文本心血管磁共振（CMR）报告中提取诊断信息的可能性。材料与方法：我们在109例临床CMR报告的描述性发现基础上，评估了九种开源LLM识别诊断和将患者分类到各种心脏诊断类别的能力。性能通过准确率、precision、召回率和F1分数等标准分类指标进行量化。我们还使用混淆矩阵来分析模型间的误分类模式。结果：大多数开源LLM在分类报告至不同诊断类别方面表现出色。Google的Gemma2模型获得了最高的平均F1分数0.98，其次是Qwen2.5:32B和DeepseekR1-32B，其F1分数分别为0.96和0.95。所有其他评估模型的平均分数均高于0.93，Mistral和DeepseekR1-7B是例外情况。排名前四的LLM在所有评估指标中均优于我们的执业心脏病学家（F1分数为0.94）分析CMR报告。结论：我们的研究结果显示，可以在临床环境中实施开源、隐私保护的LLM，用于影像报告的自动化分析，实现准确、快速且资源高效的诊断分类。 

---
# Prompt Engineer: Analyzing Skill Requirements in the AI Job Market 

**Title (ZH)**: 提示工程师：分析AI就业市场中的技能要求 

**Authors**: An Vu, Jonas Oppenlaender  

**Link**: [PDF](https://arxiv.org/pdf/2506.00058)  

**Abstract**: The rise of large language models (LLMs) has created a new job role: the Prompt Engineer. Despite growing interest in this position, we still do not fully understand what skills this new job role requires or how common these jobs are. We analyzed 20,662 job postings on LinkedIn, including 72 prompt engineer positions, to learn more about this emerging role. We found that prompt engineering is still rare (less than 0.5% of sampled job postings) but has a unique skill profile. Prompt engineers need AI knowledge (22.8%), prompt design skills (18.7%), good communication (21.9%), and creative problem-solving (15.8%) skills. These requirements significantly differ from those of established roles, such as data scientists and machine learning engineers, showing that prompt engineering is becoming its own profession. Our findings help job seekers, employers, and educational institutions in better understanding the emerging field of prompt engineering. 

**Abstract (ZH)**: 大型语言模型的兴起创造了新的职业角色：提示工程师。尽管对该职位的兴趣日益增长，我们仍不清楚这一新职业角色需要哪些技能，以及这些职位的普及程度。我们分析了LinkedIn上的20,662个招聘信息，其中包括72个提示工程师职位，以更深入了解这一新兴角色。我们发现，提示工程仍然罕见（少于0.5%的样本招聘信息），但具有独特的技能要求。提示工程师需要具备人工智能知识（22.8%）、提示设计技能（18.7%）、良好的沟通能力（21.9%）和创造性问题解决能力（15.8%）。这些要求与数据科学家和机器学习工程师等已有职位的要求有显著差异，表明提示工程正在形成自己的专业领域。我们的研究结果有助于求职者、雇主和教育机构更好地理解新兴的提示工程领域。 

---
# Using LLMs to Advance the Cognitive Science of Collectives 

**Title (ZH)**: 使用大型语言模型推进集体认知科学 

**Authors**: Ilia Sucholutsky, Katherine M. Collins, Nori Jacoby, Bill D. Thompson, Robert D. Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.00052)  

**Abstract**: LLMs are already transforming the study of individual cognition, but their application to studying collective cognition has been underexplored. We lay out how LLMs may be able to address the complexity that has hindered the study of collectives and raise possible risks that warrant new methods. 

**Abstract (ZH)**: 大规模语言模型已经在变革个体认知的研究，但其在研究集体认知方面的应用尚未充分探索。我们阐述了大规模语言模型如何能够应对阻碍集体研究的复杂性，并提出现有方法可能需要新方法来应对的风险。 

---
# Rethinking Hybrid Retrieval: When Small Embeddings and LLM Re-ranking Beat Bigger Models 

**Title (ZH)**: 重思混合检索：当小型嵌入和LLM重新排序击败更大模型 

**Authors**: Arjun Rao, Hanieh Alipour, Nick Pendar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00049)  

**Abstract**: This paper presents a comparison of embedding models in tri-modal hybrid retrieval for Retrieval-Augmented Generation (RAG) systems. We investigate the fusion of dense semantic, sparse lexical, and graph-based embeddings, focusing on the performance of the MiniLM-v6 and BGE-Large architectures. Contrary to conventional assumptions, our results show that the compact MiniLM-v6 outperforms the larger BGE-Large when integrated with LLM-based re-ranking within our tri-modal hybrid framework. Experiments conducted on the SciFact, FIQA, and NFCorpus datasets demonstrate significant improvements in retrieval quality with the MiniLM-v6 configuration. The performance difference is particularly pronounced in agentic re-ranking scenarios, indicating better alignment between MiniLM-v6's embedding space and LLM reasoning. Our findings suggest that embedding model selection for RAG systems should prioritize compatibility with multi-signal fusion and LLM alignment, rather than relying solely on larger models. This approach may reduce computational requirements while improving retrieval accuracy and efficiency. 

**Abstract (ZH)**: This paper compares embedding models in tri-modal hybrid retrieval for Retrieval-Augmented Generation (RAG) systems, focusing on the performance of MiniLM-v6 and BGE-Large architectures. 

---
# Amadeus-Verbo Technical Report: The powerful Qwen2.5 family models trained in Portuguese 

**Title (ZH)**: 阿玛德斯-韦博技术报告：葡萄牙语训练的Qwen2.5家族模型 

**Authors**: William Alberto Cruz-Castañeda, Marcellus Amadeus  

**Link**: [PDF](https://arxiv.org/pdf/2506.00019)  

**Abstract**: This report introduces the experience of developing Amadeus Verbo, a family of large language models for Brazilian Portuguese. To handle diverse use cases, Amadeus Verbo includes base-tuned, merged, and instruction-tuned models in sizes of 0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B parameters. Thus, the main objective is to show how easy it is to fine-tune foundation models to democratize the open-source development of Brazilian Portuguese LLMs when data and resources are available. Amadeus-Verbo family models are all available at HuggingFace at this https URL. 

**Abstract (ZH)**: 本报告介绍了Amadeus Verbo的发展经验，这是一个用于巴西葡萄牙语的大型语言模型家族。为了应对多样的应用场景，Amadeus Verbo包括了0.5B、1.5B、3B、7B、14B、32B和72B参数的基模型、合并模型和指令调优模型。因此，主要目标是展示在有足够的数据和资源的情况下，如何轻松地调优基础模型以促进巴西葡萄牙语开源大型语言模型的开发。Amadeus-Verbo家族模型均可在HuggingFace获取。 

---
# Advancing AI-assisted Hardware Design with Hierarchical Decentralized Training and Personalized Inference-Time Optimization 

**Title (ZH)**: 基于分层去中心化训练与个性化推理时优化的AI辅助硬件设计推进 

**Authors**: Hao Mark Chen, Zehuan Zhang, Wanru Zhao, Nicholas Lane, Hongxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00002)  

**Abstract**: Recent years have witnessed a significant increase in the adoption of AI techniques to enhance electronic design automation. In particular, the emergence of Large Language Models (LLMs) has sparked significant interest in LLM-assisted hardware design generation, spanning applications from classical digital circuits to quantum computing. Despite substantial progress in this direction, the quality of LLM-generated hardware design still cannot meet the requirements for practical deployment. In this work, we identify three critical challenges hindering the development of LLM-assisted hardware design generation: 1) limited data availability, 2) varied data quality, 3) inadequate inference-time efficiency. To address these fundamental challenges, this paper introduces a two-stage framework for AI-assisted hardware design by exploring decentralized training and personalized inference. In the first stage, we propose to harness private domain design sources through a hierarchical decentralized training mechanism that addresses data-sharing constraints. To mitigate the impact of low-quality data, we identify optimization opportunities in hardware generation tasks, using user-defined metrics for model aggregation. The second stage focuses on client personalization to enhance both speed and quality. We introduce a new metric, Trueput, to analyze LLM-assisted hardware generation efficiency. To optimize Trueput, we implement personalized inference-time acceleration and customized sampling strategies. Evaluating both classical and quantum benchmarks, our experimental results demonstrate that the proposed two-stage framework can significantly improve the model capability for hardware design generation. As orthogonal enhancements to existing methods, our framework can achieve $33\% \sim 50\%$ semantic accuracy improvement and $2.3$ times speedup, depending on the difficulty of the generation tasks. 

**Abstract (ZH)**: 近年来，人工智能技术在电子设计自动化中的应用显著增加。特别是大型语言模型（LLMs）的出现，引发了对LLM辅助硬件设计生成的兴趣，涵盖了从经典数字电路到量子计算的应用。尽管在此方向上取得了重大进展，但LLM生成的硬件设计质量仍无法满足实际部署的要求。本文识别了阻碍LLM辅助硬件设计生成发展的三大关键挑战：1）数据 availability有限，2）数据质量参差不齐，3）推理时效率不足。为了解决这些基本挑战，本文提出了一个两阶段的AI辅助硬件设计框架，通过探索去中心化训练和个人化推理。在第一阶段，我们提出通过分层去中心化训练机制利用私有的领域设计源，以解决数据共享约束问题。为了减轻低质量数据的影响，我们通过使用用户定义的指标在硬件生成任务中进行模型聚合来识别优化机会。第二阶段关注客户端个性化，以提高速度和质量。我们引入了一个新的度量标准Trueput来分析LLM辅助硬件生成的效率。为了优化Trueput，我们实施了个性化推理时间和定制采样策略。通过对经典和量子基准的评估，实验结果表明，所提出的两阶段框架可以显著提高硬件设计生成的模型能力。作为现有方法的补充改进，我们的框架在生成任务难度不同的情况下可以实现33%至50%的语义准确性提升和2.3倍的速度加速。 

---
