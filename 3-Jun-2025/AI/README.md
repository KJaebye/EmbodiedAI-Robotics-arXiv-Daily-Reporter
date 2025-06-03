# RoboEgo System Card: An Omnimodal Model with Native Full Duplexity 

**Title (ZH)**: RoboEgo系统卡：一种具备原生全双工能力的多模态模型 

**Authors**: Yiqun Yao, Xiang Li, Xin Jiang, Xuezhi Fang, Naitong Yu, Aixin Sun, Yequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01934)  

**Abstract**: Humans naturally process real-world multimodal information in a full-duplex manner. In artificial intelligence, replicating this capability is essential for advancing model development and deployment, particularly in embodied contexts. The development of multimodal models faces two primary challenges: (1) effectively handling more than three modalities-such as vision, audio, and text; and (2) delivering full-duplex responses to rapidly evolving human instructions. To facilitate research on models that support both omnimodal processing and full duplexity, we present RoboEgo (alias: FLM-Ego), a unified model system designed to address both challenges. RoboEgo incorporates a backbone architecture and algorithms that natively support full duplexity, achieving a theoretical duplex latency of 80 ms. In streaming visually grounded conversations under real-world conditions, RoboEgo exhibits superior responsiveness and speech naturalness, while maintaining comparable content qualities to state-of-the-art semi-duplex omnimodal models-a feat previously considered unattainable by native full-duplex systems. 

**Abstract (ZH)**: 人类自然地以全双工方式处理真实世界多模态信息。在人工智能领域，复制这种能力对于推进模型开发和部署，尤其是在体感知情境中，至关重要。多模态模型的开发面临两大主要挑战：（1）有效处理超过三种模态的信息，如视觉、音频和文本；（2）提供针对快速变化的人类指令的全双工响应。为了促进同时支持全方位处理和全双工能力的研究，我们提出了RoboEgo（别名：FLM-Ego），一个旨在解决这两个挑战的统一模型系统。RoboEgo整合了一个支持全双工的骨干架构和算法，理论上的双工延迟为80 ms。在真实的实时视觉支撑对话中，RoboEgo表现出卓越的响应能力和口语自然度，同时保持与当前最先进的半双工全方位模型相近的内容质量，这是原生全双工系统此前被认为无法实现的。 

---
# Large language models can learn and generalize steganographic chain-of-thought under process supervision 

**Title (ZH)**: 大规模语言模型在接受过程监督的情况下可以学习和泛化隐写论证链。 

**Authors**: Joey Skaf, Luis Ibanez-Lissen, Robert McCarthy, Connor Watts, Vasil Georgiv, Hannes Whittingham, Lorena Gonzalez-Manzano, David Lindner, Cameron Tice, Edward James Young, Puria Radmard  

**Link**: [PDF](https://arxiv.org/pdf/2506.01926)  

**Abstract**: Chain-of-thought (CoT) reasoning not only enhances large language model performance but also provides critical insights into decision-making processes, marking it as a useful tool for monitoring model intent and planning. By proactively preventing models from acting on CoT indicating misaligned or harmful intent, CoT monitoring can be used to reduce risks associated with deploying models. However, developers may be incentivized to train away the appearance of harmful intent from CoT traces, by either customer preferences or regulatory requirements. Recent works have shown that banning mention of a specific example of reward hacking, which may be done either to make CoT presentable to users or as a naive attempt to prevent the behavior, causes obfuscation of the undesired reasoning traces but the persistence of the undesired behavior. Such obfuscation threatens the reliability of CoT monitoring. However, obfuscation of reasoning can be due to its internalization to latent space computation, or its encoding within the CoT. Here, we provide an extension to these results. First, we show that penalizing the use of specific strings within load-bearing reasoning traces causes models to substitute alternative strings. Crucially, this does not alter the underlying method by which the model performs the task, demonstrating that the model can learn to steganographically encode its reasoning. We further demonstrate that models can generalize an encoding scheme. When the penalized strings belong to an overarching class, the model learns not only to substitute strings seen in training, but also develops a general encoding scheme for all members of the class which it can apply to held-out testing strings. 

**Abstract (ZH)**: Chain-of-Thought推理不仅能够提升大型语言模型的表现，还能为决策过程提供关键洞察，标志着它作为一个监测模型意图和规划的有用工具。通过主动防止模型在显示错齐或有害意图的链式推理指示下行动，链式推理监控可以减少部署模型相关的风险。然而，开发者可能由于客户偏好或监管要求等原因，被激励去训练模型，使其在链式推理痕迹中不表现出有害意图。近期研究表明，禁止提及特定奖励作弊案例，无论是为了使链式推理对用户更易理解，还是出于简单地防止此行为的尝试，都会导致有害推理痕迹的掩饰，但有害行为依然存在。这种掩饰威胁了链式推理监控的可靠性。然而，推理掩饰可能是由于其被内化到潜在空间计算，或其被编码在链式推理中。在此，我们扩展了这些结果。首先，我们展示，惩罚特定字符串在承重推理痕迹中的使用，会导致模型替代使用其他字符串。关键的是，这并不会改变模型执行任务的基本方法，证明了模型能够学习隐写编码其推理。我们进一步证明，模型能够泛化编码方案。当受惩罚的字符串属于一个上位类别时，模型不仅会替换训练中见过的字符串，还会发展出适用于该类别中所有成员的一般编码方案，并能够应用到保留测试字符串上。 

---
# Understanding Overadaptation in Supervised Fine-Tuning: The Role of Ensemble Methods 

**Title (ZH)**: 监督微调中的过度适应理解：集成方法的作用 

**Authors**: Yifan Hao, Xingyuan Pan, Hanning Zhang, Chenlu Ye, Rui Pan, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01901)  

**Abstract**: Supervised fine-tuning (SFT) on domain-specific data is the dominant approach for adapting foundation models to specialized tasks. However, it has been observed that SFT models tend to forget knowledge acquired during pretraining. In vision models, ensembling a pretrained model with its fine-tuned counterpart has been shown to mitigate this issue. In this work, we demonstrate that the same holds for language models, and, more strikingly, we observe an overadaptation phenomenon: the ensemble model not only retains general knowledge from the foundation model but also outperforms the fine-tuned model even on the fine-tuning domain itself. Despite the empirical success of ensembling, a theoretical understanding of its benefits remains underexplored. We develop a formal theoretical analysis of the overadaptation phenomenon. Ensembling mitigates this by balancing two primary sources of error: bias, caused by insufficient fine-tuning, and variance, introduced by overfitting to fine-tuning data. While regularization techniques aim to address this trade-off, we show that ensembling provides a more effective solution. We analyze this phenomenon in over-parameterized linear settings and demonstrate that interpolating between pretrained and fine-tuned weights significantly improves performance. These findings offer theoretical justification for the observed advantages of model ensembling, supported by empirical experiments consistent with our analysis. 

**Abstract (ZH)**: 监督微调模型在特定领域数据上的 fine-tuning 是将基础模型适应专业化任务的主要方法。然而，观察到这些模型往往会遗忘预训练中获得的知识。在视觉模型中，将预训练模型与微调版本进行集成已被证明可以缓解这一问题。在本研究中，我们证明了同样的现象也适用于语言模型，并更为显著地观察到过度适应现象：集成模型不仅保留了基础模型的一般知识，还在微调领域甚至超越了微调模型。尽管集成方法在实践中取得了成功，但对其优势的理论理解仍相对不足。我们为过度适应现象开发了形式化的理论分析。集成方法通过平衡两个主要的误差来源——由不足的微调引起的偏差和由对微调数据过度拟合引入的方差——来缓解这一问题。尽管正则化技术旨在解决这种权衡，但我们的研究证明集成提供了更有效的解决方案。我们在这类过度参数化的线性设置中分析了这一现象，并证明了在预训练和微调权重之间进行插值可以显著提高性能。这些发现为集成模型观察到的优势提供了理论依据，并得到了与我们分析一致的经验实验的支持。 

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
# Fodor and Pylyshyn's Legacy - Still No Human-like Systematic Compositionality in Neural Networks 

**Title (ZH)**: 福多和皮利申的遗产——神经网络中仍无类人类的系统组合性 

**Authors**: Tim Woydt, Moritz Willig, Antonia Wüst, Lukas Helff, Wolfgang Stammer, Constantin A. Rothkopf, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2506.01820)  

**Abstract**: Strong meta-learning capabilities for systematic compositionality are emerging as an important skill for navigating the complex and changing tasks of today's world. However, in presenting models for robust adaptation to novel environments, it is important to refrain from making unsupported claims about the performance of meta-learning systems that ultimately do not stand up to scrutiny. While Fodor and Pylyshyn famously posited that neural networks inherently lack this capacity as they are unable to model compositional representations or structure-sensitive operations, and thus are not a viable model of the human mind, Lake and Baroni recently presented meta-learning as a pathway to compositionality. In this position paper, we critically revisit this claim and highlight limitations in the proposed meta-learning framework for compositionality. Our analysis shows that modern neural meta-learning systems can only perform such tasks, if at all, under a very narrow and restricted definition of a meta-learning setup. We therefore claim that `Fodor and Pylyshyn's legacy' persists, and to date, there is no human-like systematic compositionality learned in neural networks. 

**Abstract (ZH)**: 强大的元学习能力对于系统组合性而言正在 emerge 为一项重要的技能，以便应对当今复杂多变任务的挑战。然而，在呈现模型以实现对新型环境的稳健适应时，必须避免对元学习系统的性能提出未得到验证的声明，这些声明最终无法经受住审视。虽然 Fodor 和 Pylyshyn 声称神经网络在本质上缺乏这种能力，因为它们无法建模组合性表示或敏感于结构的操作，因此不适合作为人脑思维的模型，Lake 和 Baroni 最近则提出了通过元学习实现组合性的途径。在本文中，我们重新审视了这一观点并指出了所提出的元学习框架在组合性方面的局限性。我们的分析表明，现代神经元学习系统只能在非常狭窄和受限的元学习设置定义下完成此类任务。因此，我们提出“Fodor 和 Pylyshun 的遗产”仍然存在，到目前为止，神经网络中尚未学习到类似人类的系统组合性。 

---
# The Ultimate Test of Superintelligent AI Agents: Can an AI Balance Care and Control in Asymmetric Relationships? 

**Title (ZH)**: 超智能AI代理的终极测试：AI能在不对称关系中平衡关怀与控制吗？ 

**Authors**: Djallel Bouneffouf, Matthew Riemer, Kush Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2506.01813)  

**Abstract**: This paper introduces the Shepherd Test, a new conceptual test for assessing the moral and relational dimensions of superintelligent artificial agents. The test is inspired by human interactions with animals, where ethical considerations about care, manipulation, and consumption arise in contexts of asymmetric power and self-preservation. We argue that AI crosses an important, and potentially dangerous, threshold of intelligence when it exhibits the ability to manipulate, nurture, and instrumentally use less intelligent agents, while also managing its own survival and expansion goals. This includes the ability to weigh moral trade-offs between self-interest and the well-being of subordinate agents. The Shepherd Test thus challenges traditional AI evaluation paradigms by emphasizing moral agency, hierarchical behavior, and complex decision-making under existential stakes. We argue that this shift is critical for advancing AI governance, particularly as AI systems become increasingly integrated into multi-agent environments. We conclude by identifying key research directions, including the development of simulation environments for testing moral behavior in AI, and the formalization of ethical manipulation within multi-agent systems. 

**Abstract (ZH)**: 这篇论文介绍了一种新的概念性测试——牧羊人测试，用于评估超智能人工代理的道德和关系维度。该测试受人类与动物互动启发，而在权力不对称和自我保存的背景下，伦理考虑涉及关于关怀、操控和消费的问题。我们指出，当AI表现出操控、养育和支持较不智能代理、同时管理自身生存和扩展目标的能力时，它跨越了一个重要的、且可能危险的智力门槛。这包括权衡自利与次级代理福祉之间的道德权衡能力。因此，牧羊人测试挑战了传统的人工智能评估范式，强调道德代理、层级行为以及在存在性风险下的复杂决策。我们认为，这种转变对于推进人工智能治理至关重要，特别是在人工智能系统越来越多地被集成到多代理环境中时。最后，我们指出了关键的研究方向，包括开发用于测试人工智能道德行为的模拟环境以及在多代理系统中正式化道德操控。 

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
# Generate, Not Recommend: Personalized Multimodal Content Generation 

**Title (ZH)**: 生成，而非推荐：个性化多模态内容生成 

**Authors**: Jiongnan Liu, Zhicheng Dou, Ning Hu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01704)  

**Abstract**: To address the challenge of information overload from massive web contents, recommender systems are widely applied to retrieve and present personalized results for users. However, recommendation tasks are inherently constrained to filtering existing items and lack the ability to generate novel concepts, limiting their capacity to fully satisfy user demands and preferences. In this paper, we propose a new paradigm that goes beyond content filtering and selecting: directly generating personalized items in a multimodal form, such as images, tailored to individual users. To accomplish this, we leverage any-to-any Large Multimodal Models (LMMs) and train them in both supervised fine-tuning and online reinforcement learning strategy to equip them with the ability to yield tailored next items for users. Experiments on two benchmark datasets and user study confirm the efficacy of the proposed method. Notably, the generated images not only align well with users' historical preferences but also exhibit relevance to their potential future interests. 

**Abstract (ZH)**: 为了应对海量网络内容带来的信息过载挑战，推荐系统被广泛应用于为用户提供个性化结果。然而，推荐任务本质上局限于过滤现有项目，缺乏生成新概念的能力，限制了其满足用户需求和偏好的能力。本文提出了一种新的 paradigm，超越了内容过滤和选择：直接生成个性化项目，如图像，且针对个别用户量身定制。为此，我们利用任意到任意的大规模多模态模型（LMMs），并采用监督微调和在线强化学习策略进行训练，使模型具备为用户提供定制化后续项目的 ability。在两个基准数据集上的实验和用户研究均证实了所提方法的有效性。值得注意的是，生成的图像不仅与用户的 histórico 偏好高度一致，还能反映他们潜在的未来兴趣。 

---
# A Descriptive and Normative Theory of Human Beliefs in RLHF 

**Title (ZH)**: 人类信念在RLHF中的描述性与规范性理论 

**Authors**: Sylee Dandekar, Shripad Deshmukh, Frank Chiu, W. Bradley Knox, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2506.01692)  

**Abstract**: Human preferences in RLHF are typically modeled as a function of the human's reward function or corresponding optimal state-action values. In this work, we propose that human beliefs about the capabilities of the agent being trained also play a key role in preference generation. We examine two questions related to this hypothesis, one descriptive and one normative, respectively: Do human labelers' beliefs about agent capabilities affect the preferences that they provide? And what is the ideal set of beliefs about an agent -- and resulting preferences -- for humans to have? We propose a new preference model that incorporates human beliefs and provide a normative theory that bounds the error on the final learned policy based on the \textit{mismatch} between the human's beliefs and an idealized set of beliefs. We then confirm via a human study that beliefs about agent capabilities do, in fact, significantly affect preferences and can be influenced through simple interventions. Additionally, we empirically show through synthetic experiments that it is often suboptimal for human preference labelers to assume agent optimality. Collectively, these results theoretically and empirically demonstrate how reducing the mismatch between human beliefs and agent capabilities can lead to more performant RLHF and point toward new best practices for RLHF practitioners. 

**Abstract (ZH)**: 人类在RLHF中的偏好通常被建模为人类奖励函数或相应最优状态行动值的函数。在本工作中，我们提出，训练中的智能体能力的人类信念也在偏好生成中扮演关键角色。我们分别探讨了与这一假设相关的两个问题，一个是描述性的，一个是规范性的：人类标注者关于智能体能力的信念是否影响他们提供的偏好？人类应该如何恰当地认为智能体的能力——以及相应的偏好——理想的信念集合是什么？我们提出了一种新的偏好模型，该模型包含了人类的信念，并提供了一种规范理论，该理论基于人类信念与理想化信念之间的差异来界定了最终学习策略的误差。然后，通过人类研究证实，对智能体能力的信念确实显著影响偏好，并可以通过简单的干预措施加以影响。此外，通过合成实验，我们实证表明，人类偏好标注者假设智能体最优化往往是次优的。这些结果从理论和实证上证明了减少人类信念与智能体能力之间的偏差如何能够提高RLHF的性能，并指出了RLHF从业者新的最佳实践。 

---
# Respond Beyond Language: A Benchmark for Video Generation in Response to Realistic User Intents 

**Title (ZH)**: 超越语言的回应：基于现实用户意图的视频生成基准 

**Authors**: Shuting Wang, Yunqi Liu, Zixin Yang, Ning Hu, Zhicheng Dou, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01689)  

**Abstract**: Querying generative AI models, e.g., large language models (LLMs), has become a prevalent method for information acquisition. However, existing query-answer datasets primarily focus on textual responses, making it challenging to address complex user queries that require visual demonstrations or explanations for better understanding. To bridge this gap, we construct a benchmark, RealVideoQuest, designed to evaluate the abilities of text-to-video (T2V) models in answering real-world, visually grounded queries. It identifies 7.5K real user queries with video response intents from Chatbot-Arena and builds 4.5K high-quality query-video pairs through a multistage video retrieval and refinement process. We further develop a multi-angle evaluation system to assess the quality of generated video answers. Experiments indicate that current T2V models struggle with effectively addressing real user queries, pointing to key challenges and future research opportunities in multimodal AI. 

**Abstract (ZH)**: 查询生成型AI模型，例如大规模语言模型（LLMs），已成为信息获取的主流方法。然而，现有的查询-回答数据集主要侧重于文本响应，这使得处理需要视觉演示或解释以更好地理解的复杂用户查询变得具有挑战性。为了弥合这一差距，我们构建了一个基准数据集RealVideoQuest，旨在评估文本到视频（T2V）模型在回答具有视觉基础的现实世界查询方面的能力。该数据集从Chatbot-Arena中识别出7,500个包含视频响应意图的真实用户查询，并通过多阶段的视频检索和精炼过程构建了4,500个高质量的查询-视频对。我们还开发了一个多角度评估系统来评估生成视频答案的质量。实验表明，当前的T2V模型在有效处理真实用户查询方面存在困难，指出了跨模态AI中的关键挑战和未来研究机会。 

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
# Social Cooperation in Conversational AI Agents 

**Title (ZH)**: 对话式人工智能代理中的社会协作 

**Authors**: Mustafa Mert Çelikok, Saptarashmi Bandyopadhyay, Robert Loftin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01624)  

**Abstract**: The development of AI agents based on large, open-domain language models (LLMs) has paved the way for the development of general-purpose AI assistants that can support human in tasks such as writing, coding, graphic design, and scientific research. A major challenge with such agents is that, by necessity, they are trained by observing relatively short-term interactions with humans. Such models can fail to generalize to long-term interactions, for example, interactions where a user has repeatedly corrected mistakes on the part of the agent. In this work, we argue that these challenges can be overcome by explicitly modeling humans' social intelligence, that is, their ability to build and maintain long-term relationships with other agents whose behavior cannot always be predicted. By mathematically modeling the strategies humans use to communicate and reason about one another over long periods of time, we may be able to derive new game theoretic objectives against which LLMs and future AI agents may be optimized. 

**Abstract (ZH)**: 基于大型开放域语言模型（LLMs）的AI代理的发展为开发能够支持人类在写作、编程、图形设计和科学研究等任务方面的通用AI助手铺平了道路。这类代理的一个主要挑战是，它们必须通过观察与人类相对短暂的互动来训练，这可能导致它们难以将模型泛化到长时间的互动中，例如用户反复纠正代理错误的情况。在本工作中，我们argue可以通过明确建模人类的社会智能，即他们与行为难以预测的其他代理建立并维持长期关系的能力，来克服这些挑战。通过数学建模人类在长时间内相互交流和推理的策略，我们或许能够推导出新的博弈论目标，以优化LLMs和未来AI代理。 

---
# MAGIK: Mapping to Analogous Goals via Imagination-enabled Knowledge Transfer 

**Title (ZH)**: MAGIK: 通过想象enabled知识迁移映射到类似的目标 

**Authors**: Ajsal Shereef Palattuparambil, Thommen George Karimpanal, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2506.01623)  

**Abstract**: Humans excel at analogical reasoning - applying knowledge from one task to a related one with minimal relearning. In contrast, reinforcement learning (RL) agents typically require extensive retraining even when new tasks share structural similarities with previously learned ones. In this work, we propose MAGIK, a novel framework that enables RL agents to transfer knowledge to analogous tasks without interacting with the target environment. Our approach leverages an imagination mechanism to map entities in the target task to their analogues in the source domain, allowing the agent to reuse its original policy. Experiments on custom MiniGrid and MuJoCo tasks show that MAGIK achieves effective zero-shot transfer using only a small number of human-labelled examples. We compare our approach to related baselines and highlight how it offers a novel and effective mechanism for knowledge transfer via imagination-based analogy mapping. 

**Abstract (ZH)**: 人类在类比推理方面表现出色——能够在相关任务中应用知识，无需大量重新学习。相比之下，强化学习代理通常需要大量重新训练，即使新任务与先前学习的任务在结构上相似也不例外。在本文中，我们提出了一种名为MAGIK的新颖框架，该框架能够在不与目标环境互动的情况下使强化学习代理将知识转移到相关任务中。我们的方法利用想象机制将目标任务中的实体映射到源域中的类比实体，从而使代理能够重用其原始策略。实验表明，MAGIK仅使用少量的人工标注示例即可实现有效的零样本转移。我们将我们的方法与相关的基线进行比较，并强调它通过基于想象的类比映射提供了一种新颖而有效的知识转移机制。 

---
# General agents need world models 

**Title (ZH)**: 通用代理需要世界模型 

**Authors**: Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt  

**Link**: [PDF](https://arxiv.org/pdf/2506.01622)  

**Abstract**: Are world models a necessary ingredient for flexible, goal-directed behaviour, or is model-free learning sufficient? We provide a formal answer to this question, showing that any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment. We show that this model can be extracted from the agent's policy, and that increasing the agents performance or the complexity of the goals it can achieve requires learning increasingly accurate world models. This has a number of consequences: from developing safe and general agents, to bounding agent capabilities in complex environments, and providing new algorithms for eliciting world models from agents. 

**Abstract (ZH)**: 世界模型是实现灵活、目标导向行为的必要成分，还是无模型学习足夠？我们提供了对该问题的形式化回答，表明任何能够泛化到多步目标导向任务的代理都必须学到了其环境的预测模型。我们证明了可以从代理的策略中提取这种模型，并且提高代理的性能或其可实现目标的复杂性需要学习越来越准确的世界模型。这具有多个后果：从开发安全且通用的代理，到限制代理在复杂环境中的能力，以及为从代理中引出世界模型提供新的算法。 

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
# Agentic Episodic Control 

**Title (ZH)**: 代理性事件控制 

**Authors**: Xidong Yang, Wenhao Li, Junjie Sheng, Chuyun Shen, Yun Hua, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01442)  

**Abstract**: Reinforcement learning (RL) has driven breakthroughs in AI, from game-play to scientific discovery and AI alignment. However, its broader applicability remains limited by challenges such as low data efficiency and poor generalizability. Recent advances suggest that large language models, with their rich world knowledge and reasoning capabilities, could complement RL by enabling semantic state modeling and task-agnostic planning. In this work, we propose the Agentic Episodic Control (AEC), a novel architecture that integrates RL with LLMs to enhance decision-making. The AEC can leverage a large language model (LLM) to map the observations into language-grounded embeddings, which further can be stored in an episodic memory for rapid retrieval of high-value experiences. Simultaneously, a World-Graph working memory module is utilized to capture structured environmental dynamics in order to enhance relational reasoning. Furthermore, a lightweight critical state detector dynamically arbitrates between the episodic memory recall and the world-model-guided exploration. On the whole, by combining the trial-and-error learning scheme with LLM-derived semantic priors, the proposed AEC can improve both data efficiency and generalizability in reinforcement learning. In experiments on BabyAI-Text benchmark tasks, AEC demonstrates substantial improvements over existing baselines, especially on complex and generalization tasks like FindObj, where it outperforms the best baseline by up to 76%. The proposed AEC framework bridges the strengths of numeric reinforcement learning and symbolic reasoning, which provides a pathway toward more adaptable and sample-efficient agents. 

**Abstract (ZH)**: 基于大型语言模型的强化学习代理人 episodic 控制 (AEC) 架构 

---
# Distinguishing Autonomous AI Agents from Collaborative Agentic Systems: A Comprehensive Framework for Understanding Modern Intelligent Architectures 

**Title (ZH)**: 区分自主人工智能代理与协作代理系统：理解现代智能架构的全面框架 

**Authors**: Prashik Buddhaghosh Bansod  

**Link**: [PDF](https://arxiv.org/pdf/2506.01438)  

**Abstract**: The emergence of large language models has catalyzed two distinct yet interconnected paradigms in artificial intelligence: standalone AI Agents and collaborative Agentic AI ecosystems. This comprehensive study establishes a definitive framework for distinguishing these architectures through systematic analysis of their operational principles, structural compositions, and deployment methodologies. We characterize AI Agents as specialized, tool-enhanced systems leveraging foundation models for targeted automation within constrained environments. Conversely, Agentic AI represents sophisticated multi-entity frameworks where distributed agents exhibit emergent collective intelligence through coordinated interaction protocols. Our investigation traces the evolutionary trajectory from traditional rule-based systems through generative AI foundations to contemporary agent architectures. We present detailed architectural comparisons examining planning mechanisms, memory systems, coordination protocols, and decision-making processes. The study categorizes application landscapes, contrasting single-agent implementations in customer service and content management with multi-agent deployments in research automation and complex decision support. We identify critical challenges including reliability issues, coordination complexities, and scalability constraints, while proposing innovative solutions through enhanced reasoning frameworks, robust memory architectures, and improved coordination mechanisms. This framework provides essential guidance for practitioners selecting appropriate agentic approaches and establishes foundational principles for next-generation intelligent system development. 

**Abstract (ZH)**: 大型语言模型的出现催化了人工智能中的两种截然不同但又相互关联的范式：独立的人工智能代理和协作型代理人人工智能生态系统。本研究通过系统分析其运行原理、结构组成和部署方法建立了区分这些架构的明确框架。我们将人工智能代理characterized为利用基础模型在受限环境中进行目标自动化的专业化、工具增强型系统。相反，代理人人工智能represent了复杂的多实体框架，在此框架中，分布式代理通过协调交互协议表现出涌现的集体智能。研究追溯了从传统的基于规则系统到生成式人工智能基础再到当前代理架构的进化轨迹。我们详细比较了规划机制、记忆系统、协调协议和决策过程的架构。研究分类了应用景观，将单代理实施方案与多代理部署在科研自动化和复杂决策支持中的对比，指出了关键挑战包括可靠性问题、协调复杂性和扩展限制，并提出通过增强推理框架、稳健的记忆架构和改进的协调机制等创新解决方案。该框架为从业者选择合适的代理人方法提供了重要指导，并建立了下一代智能系统开发的基础原则。 

---
# FinRobot: Generative Business Process AI Agents for Enterprise Resource Planning in Finance 

**Title (ZH)**: FinRobot: 生成式商务过程AI代理在金融企业资源规划中应用 

**Authors**: Hongyang Yang, Likun Lin, Yang She, Xinyu Liao, Jiaoyang Wang, Runjia Zhang, Yuquan Mo, Christina Dan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01423)  

**Abstract**: Enterprise Resource Planning (ERP) systems serve as the digital backbone of modern financial institutions, yet they continue to rely on static, rule-based workflows that limit adaptability, scalability, and intelligence. As business operations grow more complex and data-rich, conventional ERP platforms struggle to integrate structured and unstructured data in real time and to accommodate dynamic, cross-functional workflows.
In this paper, we present the first AI-native, agent-based framework for ERP systems, introducing a novel architecture of Generative Business Process AI Agents (GBPAs) that bring autonomy, reasoning, and dynamic optimization to enterprise workflows. The proposed system integrates generative AI with business process modeling and multi-agent orchestration, enabling end-to-end automation of complex tasks such as budget planning, financial reporting, and wire transfer processing. Unlike traditional workflow engines, GBPAs interpret user intent, synthesize workflows in real time, and coordinate specialized sub-agents for modular task execution. We validate the framework through case studies in bank wire transfers and employee reimbursements, two representative financial workflows with distinct complexity and data modalities. Results show that GBPAs achieve up to 40% reduction in processing time, 94% drop in error rate, and improved regulatory compliance by enabling parallelism, risk control insertion, and semantic reasoning. These findings highlight the potential of GBPAs to bridge the gap between generative AI capabilities and enterprise-grade automation, laying the groundwork for the next generation of intelligent ERP systems. 

**Abstract (ZH)**: 基于AI的企业资源规划系统：生成式商业过程智能代理的架构 

---
# AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning 

**Title (ZH)**: AgentCPM-GUI: 构建基于强化微调的移动用途代理 

**Authors**: Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, Jie Xie, Wei Zhou, Wang Xu, Yuanheng Zhang, Zhou Su, Zhongwu Zhai, Xiaoming Liu, Yudong Mei, Jianming Xu, Hongyan Tian, Chongyi Wang, Chi Chen, Yuan Yao, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01391)  

**Abstract**: The recent progress of large language model agents has opened new possibilities for automating tasks through graphical user interfaces (GUIs), especially in mobile environments where intelligent interaction can greatly enhance usability. However, practical deployment of such agents remains constrained by several key challenges. Existing training data is often noisy and lack semantic diversity, which hinders the learning of precise grounding and planning. Models trained purely by imitation tend to overfit to seen interface patterns and fail to generalize in unfamiliar scenarios. Moreover, most prior work focuses on English interfaces while overlooks the growing diversity of non-English applications such as those in the Chinese mobile ecosystem. In this work, we present AgentCPM-GUI, an 8B-parameter GUI agent built for robust and efficient on-device GUI interaction. Our training pipeline includes grounding-aware pre-training to enhance perception, supervised fine-tuning on high-quality Chinese and English trajectories to imitate human-like actions, and reinforcement fine-tuning with GRPO to improve reasoning capability. We also introduce a compact action space that reduces output length and supports low-latency execution on mobile devices. AgentCPM-GUI achieves state-of-the-art performance on five public benchmarks and a new Chinese GUI benchmark called CAGUI, reaching $96.9\%$ Type-Match and $91.3\%$ Exact-Match. To facilitate reproducibility and further research, we publicly release all code, model checkpoint, and evaluation data. 

**Abstract (ZH)**: 大型语言模型代理在图形用户界面中的Recent进展及其在移动环境中的智能交互应用：AgentCPM-GUI的研究 

---
# AI Scientists Fail Without Strong Implementation Capability 

**Title (ZH)**: AI科学家缺乏强大的实施能力将难以取得成功。 

**Authors**: Minjun Zhu, Qiujie Xie, Yixuan Weng, Jian Wu, Zhen Lin, Linyi Yang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01372)  

**Abstract**: The emergence of Artificial Intelligence (AI) Scientist represents a paradigm shift in scientific discovery, with large language models (LLMs) taking the lead as the primary executor in the entire scientific workflow from idea generation to experiment implementation. Recent AI Scientist studies demonstrate sufficient capabilities for independent scientific discovery, with the generated research reports gaining acceptance at the ICLR 2025 workshop and ACL 2025, arguing that a human-level AI Scientist, capable of uncovering phenomena previously unknown to humans, may be imminent. Despite this substantial progress, AI Scientist has yet to produce a groundbreaking achievement in the domain of computer science on par with automated scientific tools. Based on extensive quantitative evidence from existing benchmarks in complex engineering tasks and a systematic evaluation assess 28 research papers generated by five advanced AI Scientist systems, we argue that \textbf{the fundamental bottleneck for AI Scientists lies in their capability to execute the requisite verification procedures.} Current AI Scientist systems lack the execution capabilities needed to execute rigorous experiments and produce high-quality scientific papers. To better illustrate the root cause of this \textbf{implementation gap}, we provide an in-depth discussion on the fundamental limitations of AI Scientist. This position paper aims to call for the participants in the community to bridge the implementation gap. 

**Abstract (ZH)**: 人工智能科学家的兴起代表了科学发现范式的转变，大规模语言模型（LLMs）在从想法生成到实验实施的整个科学工作流程中 rôle leading。近期的人工智能科学家研究显示了独立进行科学研究的能力，生成的研究报告在ICLR 2025研讨会和ACL 2025会议上获得接受，这表明具备人类水平的人工智能科学家，能揭示此前人类未知的现象，可能即将来临。尽管取得了这些显著进展，人工智能科学家在计算机科学领域尚未取得与自动化科学工具相媲美的突破性成就。基于复杂工程任务现有基准的大量定量证据以及对五个先进人工智能科学家系统生成的28篇研究论文的系统评估，我们认为 \textbf{人工智能科学家的核心瓶颈在于其执行必要的验证程序的能力。} 当前的人工智能科学家系统缺乏执行严谨实验和产生高质量科学论文所需的能力。为了更好地说明这一 \textbf{实现差距} 的根本原因，我们深入讨论了人工智能科学家的基本局限性。这篇立场论文旨在呼吁社区成员弥合实现差距。 

---
# EgoBrain: Synergizing Minds and Eyes For Human Action Understanding 

**Title (ZH)**: EgoBrain: 结合心智与视觉的人类动作理解 

**Authors**: Nie Lin, Yansen Wang, Dongqi Han, Weibang Jiang, Jingyuan Li, Ryosuke Furuta, Yoichi Sato, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01353)  

**Abstract**: The integration of brain-computer interfaces (BCIs), in particular electroencephalography (EEG), with artificial intelligence (AI) has shown tremendous promise in decoding human cognition and behavior from neural signals. In particular, the rise of multimodal AI models have brought new possibilities that have never been imagined before. Here, we present EgoBrain --the world's first large-scale, temporally aligned multimodal dataset that synchronizes egocentric vision and EEG of human brain over extended periods of time, establishing a new paradigm for human-centered behavior analysis. This dataset comprises 61 hours of synchronized 32-channel EEG recordings and first-person video from 40 participants engaged in 29 categories of daily activities. We then developed a muiltimodal learning framework to fuse EEG and vision for action understanding, validated across both cross-subject and cross-environment challenges, achieving an action recognition accuracy of 66.70%. EgoBrain paves the way for a unified framework for brain-computer interface with multiple modalities. All data, tools, and acquisition protocols are openly shared to foster open science in cognitive computing. 

**Abstract (ZH)**: 脑-机接口（BCI）特别是脑电图（EEG）与人工智能（AI）的整合在解码人类认知和行为的神经信号方面展现了巨大的潜力。特别是，多模态AI模型的兴起带来了前所未有的新可能性。在这里，我们介绍了EgoBrain——世界上首个大规模、时间对齐的多模态数据集，该数据集同步了人类大脑的自中心视觉和长时间段的脑电图记录，建立了以人为中心的行为分析新范式。该数据集包括40名参与者在29类日常活动中记录的61小时同步32通道脑电图和一人群体视角视频。我们随后开发了一种多模态学习框架，将脑电图和视觉信息融合以理解动作，验证通过跨被试和跨环境挑战，实现了动作识别准确率66.70%。EgoBrain为多模态脑-机接口整合框架铺平了道路。所有数据、工具和采集协议均公开共享，以促进认知计算中的开放科学。 

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
# Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner 

**Title (ZH)**: 克服多步复杂性在多模态理论思维推理中的障碍：一个可扩展的贝叶斯规划者 

**Authors**: Chunhui Zhang, Zhongyu Ouyang, Kwonjoon Lee, Nakul Agarwal, Sean Dae Houlihan, Soroush Vosoughi, Shao-Yuan Lo  

**Link**: [PDF](https://arxiv.org/pdf/2506.01301)  

**Abstract**: Theory-of-Mind (ToM) enables humans to infer mental states-such as beliefs, desires, and intentions-forming the foundation of social cognition. However, existing computational ToM methods rely on structured workflows with ToM-specific priors or deep model fine-tuning, which struggle with scalability in multimodal environments and fail to generalize as task complexity increases. To address these limitations, we propose a scalable Bayesian ToM planner that decomposes ToM reasoning into stepwise Bayesian updates. Our framework introduces weak-to-strong control, allowing smaller language models (LMs) to specialize in ToM-specific likelihood estimation and transfer their reasoning behaviors to larger LMs (7B to 405B) for integration with social and world knowledge. This synergistic approach aligns large-model inference of human mental states with Bayesian principles. Extensive experiments show that our method achieves a 4.6% accuracy improvement over state-of-the-art techniques on multimodal ToM benchmarks, including challenging unseen scenarios, thereby establishing a new standard for modeling human mental states in complex environments. 

**Abstract (ZH)**: Theory-of-Mind (ToM) 理论使人类能够推断信念、欲望和意图等心理状态，奠定社会认知的基础。然而，现有的计算ToM方法依赖于有ToM特定先验或深度模型微调的结构化工作流程，在多模态环境中难以扩展，并且随着任务复杂性的增加而难以泛化。为解决这些限制，我们提出了一种可扩展的贝叶斯ToM规划器，将ToM推理分解为逐步的贝叶斯更新。该框架引入了从弱到强的控制，允许较小的语言模型（LMs）专门从事ToM特定的似然估计，并将其实现的推理行为转移到更大的LMs（从7B到405B），以便将社会和世界知识集成进去。这种协同方法将大型模型对人类心理状态的推理与贝叶斯原则对齐。广泛的实验结果显示，我们的方法在包括具有挑战性的未见过场景的多模态ToM基准测试上，较最先进的技术在准确率上提高了4.6%，从而建立了在复杂环境中建模人类心理状态的新标准。 

---
# Scalable In-Context Q-Learning 

**Title (ZH)**: 可扩展的上下文内Q学习 

**Authors**: Jinmei Liu, Fuhong Liu, Jianye Hao, Bo Wang, Huaxiong Li, Chunlin Chen, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01299)  

**Abstract**: Recent advancements in language models have demonstrated remarkable in-context learning abilities, prompting the exploration of in-context reinforcement learning (ICRL) to extend the promise to decision domains. Due to involving more complex dynamics and temporal correlations, existing ICRL approaches may face challenges in learning from suboptimal trajectories and achieving precise in-context inference. In the paper, we propose \textbf{S}calable \textbf{I}n-\textbf{C}ontext \textbf{Q}-\textbf{L}earning (\textbf{SICQL}), an innovative framework that harnesses dynamic programming and world modeling to steer ICRL toward efficient reward maximization and task generalization, while retaining the scalability and stability of supervised pretraining. We design a prompt-based multi-head transformer architecture that simultaneously predicts optimal policies and in-context value functions using separate heads. We pretrain a generalized world model to capture task-relevant information, enabling the construction of a compact prompt that facilitates fast and precise in-context inference. During training, we perform iterative policy improvement by fitting a state value function to an upper-expectile of the Q-function, and distill the in-context value functions into policy extraction using advantage-weighted regression. Extensive experiments across a range of discrete and continuous environments show consistent performance gains over various types of baselines, especially when learning from suboptimal data. Our code is available at this https URL 

**Abstract (ZH)**: 近期语言模型的发展展示了显著的上下文学习能力，促使人们探索上下文强化学习（ICRL）以将这种能力扩展至决策领域。由于涉及更复杂的动力学和时间关联，现有的ICRL方法可能难以从次优轨迹中学习并实现精确的上下文推断。本文提出了一种名为Scalable In-Context Q-Learning (SICQL) 的创新框架，该框架结合动态规划和世界建模，旨在实现高效奖励最大化和任务泛化的高效学习，同时保持监督预训练的可扩展性和稳定性。我们设计了一种基于提示的多头变压器架构，能够同时使用不同的头预测最优策略和上下文价值函数。我们预训练了一种通用的世界模型来捕捉与任务相关的信息，从而构建一个紧凑的提示，促进快速且精确的上下文推断。在训练过程中，通过拟合状态值函数到Q函数的上分位数来进行迭代策略改进，并使用优势加权回归将上下文价值函数蒸馏为策略提取。我们在离散和连续环境中进行了广泛的实验，表明与各种基线相比，特别是在学习次优数据时，具有一致的性能提升。我们的代码可在以下链接获取。 

---
# MobCLIP: Learning General-purpose Geospatial Representation at Scale 

**Title (ZH)**: MobCLIP: 大规模学习通用地理空间表示 

**Authors**: Ya Wen, Jixuan Cai, Qiyao Ma, Linyan Li, Xinhua Chen, Chris Webster, Yulun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.01297)  

**Abstract**: Representation learning of geospatial locations remains a core challenge in achieving general geospatial intelligence. Current embedding methods often lack versatility, limiting their utility across diverse tasks in both human and natural domains. We present MobCLIP, the first nationwide general-purpose location encoder, integrating an unprecedented diversity of data modalities through effective and scalable multimodal fusion. Adopting a novel CLIP-based architecture, our framework aligns 100M+ POIs, nationwide remote sensing imagery, and structured demographic statistics with a billion-edge mobility graph. By tokenizing spatial locations into grid cells inspired by Vision Transformers, we establish a unified representation space bridging mobility patterns and multimodal features. To rigorously evaluate the general-purpose effectiveness of MobCLIP, we construct a benchmark dataset composed of 11 downstream prediction tasks across social, economic, and natural domains. Experiments show that MobCLIP, with four input modalities and a compact 128-dimensional representation space, achieves significantly superior general-purpose predictive performances than state-of-the-art models by an average of 35%. Thanks to the effective integration of human-centric modalities, the performance gain is particularly profound in human-centric tasks, such as energy consumption (+260%), offline retail consumption amount (+98%), and crime cases (+95%) predictions. Echoing LLM scaling laws, we further demonstrate the scaling behavior in geospatial representation learning. We open-source code and pretrained models at: this http URL. 

**Abstract (ZH)**: 全国通用地理位置表示学习依然是实现通用地理智能的核心挑战。当前的嵌入方法往往缺乏灵活性，限制了其在人类和自然领域多种任务中的应用。我们提出了MobCLIP，这是首个全国范围内的通用位置编码器，通过有效的可扩展多模态融合整合前所未有的数据模态多样性。采用新型的CLIP基架构，我们的框架将100M+ POI、全国范围的遥感图像以及结构化的社会统计信息与亿级边数的移动图对齐。通过借鉴Vision Transformers的思想将空间位置 tokenize 成网格单元，我们建立了统一的表示空间，连接移动模式与多模态特征。为了严格评估MobCLIP的通用有效性，我们构建了一个基准数据集，包括11项下游预测任务，涵盖了社会、经济和自然多个领域。实验结果表明，MobCLIP仅使用四种输入模态并在紧凑的128维表示空间中，其通用预测性能优于最先进的模型，平均高出35%。得益于高效的人本模态集成，MobCLIP在人本任务中的性能提升尤为显著，如能源消耗（+260%）、离线零售消费金额（+98%）和犯罪案件（+95%）预测。遵循LLM的扩展规律，我们在地理空间表示学习中也观察到了扩展行为。我们开源了代码和预训练模型：this http URL。 

---
# On the Hardness of Approximating Distributions with Probabilistic Circuits 

**Title (ZH)**: Approximating 分布 与 概率 电路 的 难近似 性 

**Authors**: John Leland, YooJung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01281)  

**Abstract**: A fundamental challenge in probabilistic modeling is balancing expressivity and tractable inference. Probabilistic circuits (PCs) aim to directly address this tradeoff by imposing structural constraints that guarantee efficient inference of certain queries while maintaining expressivity. Since inference complexity on PCs depends on circuit size, understanding the size bounds across circuit families is key to characterizing the tradeoff between tractability and expressive efficiency. However, expressive efficiency is often studied through exact representations, where exactly encoding distributions while enforcing various structural properties often incurs exponential size blow-ups. Thus, we pose the following question: can we avoid such size blow-ups by allowing some small approximation error? We first show that approximating an arbitrary distribution with bounded $f$-divergence is $\mathsf{NP}$-hard for any model that can tractably compute marginals. We then prove an exponential size gap for approximation between the class of decomposable PCs and additionally deterministic PCs. 

**Abstract (ZH)**: 概率模型中的一个基本挑战是表达能力和可处理推断之间的平衡。概率电路（PCs）通过施加结构约束直接解决这一权衡问题，这些约束保证了某些查询的高效推断，同时保持了表达能力。由于在概率电路上的推断复杂性依赖于电路大小，理解不同电路族的大小上限对于刻画可处理性和表达效率之间的权衡至关重要。然而，表达效率通常通过精确表示来研究，这常会导致指数级的大小膨胀。因此，我们提出一个问题：是否可以通过允许一些小的近似误差来避免这种大小膨胀？我们首先证明，对于任何可以高效计算边缘概率的模型，将任意分布近似到有界 $f$-散度是 $\mathsf{NP}$-难的。然后我们证明，可分解的概率电路类和额外确定性概率电路类之间的近似大小存在指数级差距。 

---
# GeoLocSFT: Efficient Visual Geolocation via Supervised Fine-Tuning of Multimodal Foundation Models 

**Title (ZH)**: GeoLocSFT: 通过多模态基础模型监督微调进行高效视觉地理定位 

**Authors**: Qiang Yi, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2506.01277)  

**Abstract**: Accurately determining the geographic location where a single image was taken, visual geolocation, remains a formidable challenge due to the planet's vastness and the deceptive similarity among distant locations. We introduce GeoLocSFT, a framework that demonstrates how targeted supervised fine-tuning (SFT) of a large multimodal foundation model (Gemma 3) using a small, high-quality dataset can yield highly competitive geolocation performance. GeoLocSFT is trained with only 2700 carefully selected image-GPS pairs from our geographically diverse MR600k dataset. Despite this limited data, our SFT-centric approach substantially improves over baseline models and achieves robust results on standard benchmarks such as Im2GPS-3k and YFCC-4k, as well as on our newly proposed and challenging MR40k benchmark, aimed specifically at sparsely populated regions. Further, we explore multi-candidate inference and aggregation strategies but find that the core gains are already realized at the SFT stage. Our findings highlight the power of high-quality supervision and efficient SFT for planet-scale image geolocation, especially when compared to prior methods that require massive databases or complex pipelines. To foster further research, we publicly release the MR40k benchmark dataset. 

**Abstract (ZH)**: 准确确定单张图像的拍摄地理位置的视觉地理定位仍然是一项严峻的挑战，由于地球的广袤以及远处位置之间的误导性相似性。我们介绍了GeoLocSFT框架，该框架展示了如何使用小型高质量数据集对大型多模态基础模型（Gemma 3）进行目标导向的监督微调（SFT），以获得具有竞争力的地理定位性能。GeoLocSFT仅使用来自我们多样化的MR600k数据集的2700个精心选择的图像-GPS配对进行训练。尽管数据有限，但我们的SFT导向方法在基准模型上取得了显著改进，并在Im2GPS-3k、YFCC-4k以及我们新提出的专门针对稀疏地区设计的MR40k基准上取得了稳健的结果。此外，我们探索了多候选推理和聚合策略，但发现核心增益已在SFT阶段实现。我们的研究结果突显了高质量监督和高效SFT在地球规模图像地理定位中的强大作用，尤其是在与需要庞大数据库或复杂管道的先前方法相比时。为了促进进一步的研究，我们公开发布了MR40k基准数据集。 

---
# Contra4: Evaluating Contrastive Cross-Modal Reasoning in Audio, Video, Image, and 3D 

**Title (ZH)**: Contra4：评估跨模态对比推理在音频、视频、图像和3D场景中的表现 

**Authors**: Artemis Panagopoulou, Le Xue, Honglu Zhou, silvio savarese, Ran Xu, Caiming Xiong, Chris Callison-Burch, Mark Yatskar, Juan Carlos Niebles  

**Link**: [PDF](https://arxiv.org/pdf/2506.01275)  

**Abstract**: Real-world decision-making often begins with identifying which modality contains the most relevant information for a given query. While recent multimodal models have made impressive progress in processing diverse inputs, it remains unclear whether they can reason contrastively across multiple modalities to select the one that best satisfies a natural language prompt. We argue this capability is foundational, especially in retrieval-augmented and decision-time contexts, where systems must evaluate multiple signals and identify which one conveys the relevant information. To evaluate this skill, we introduce Contra4, a dataset for contrastive cross-modal reasoning across four modalities: image, audio, video, and 3D. Each example presents a natural language question alongside multiple candidate modality instances, and the model must select the one that semantically aligns with the prompt. Contra4 combines human-annotated captions with a mixture-of-models round-trip-consistency filter to ensure high-quality supervision, resulting in 174k training examples and a manually verified test set of 2.3k samples. While task-specific fine-tuning improves performance by 56% relative to baseline, state-of-the-art models still achieve only 56% accuracy overall and 42% in four-modality settings, underscoring a significant limitation in current multimodal models. 

**Abstract (ZH)**: 实世界决策往往始于识别哪些模态包含了与给定查询最相关的信息。尽管最近的多模态模型在处理多种输入方面取得了显著进展，但仍然不清楚它们是否能够在多种模态之间进行对比推理，从而挑选出最能满足自然语言提示的那个模态。我们认为这一能力是基础性的，尤其是在检索增强和决策时刻的上下文中，系统必须评估多种信号并确定哪个信号传达了相关的信息。为了评估这一技能，我们引入了Contra4数据集，用于四模态（图像、音频、视频和3D）之间的对比跨模态推理。每个示例包含一个自然语言问题和多个候选模态实例，模型必须挑选出与提示语义上最对齐的那个。Contra4结合了人工标注的描述和模型互校一致性筛选，以确保高质量的监督，数据集包含174,000个训练样本和2,300个人工验证的测试样本。尽管特定任务的微调相比基线性能提高了56%，但最先进的模型整体准确率仅为56%，四模态设置下的准确率为42%，突显了当前多模态模型的重要局限性。 

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
# Test Automation for Interactive Scenarios via Promptable Traffic Simulation 

**Title (ZH)**: 基于可提示流量模拟的交互场景自动化测试 

**Authors**: Augusto Mondelli, Yueshan Li, Alessandro Zanardi, Emilio Frazzoli  

**Link**: [PDF](https://arxiv.org/pdf/2506.01199)  

**Abstract**: Autonomous vehicle (AV) planners must undergo rigorous evaluation before widespread deployment on public roads, particularly to assess their robustness against the uncertainty of human behaviors. While recent advancements in data-driven scenario generation enable the simulation of realistic human behaviors in interactive settings, leveraging these models to construct comprehensive tests for AV planners remains an open challenge. In this work, we introduce an automated method to efficiently generate realistic and safety-critical human behaviors for AV planner evaluation in interactive scenarios. We parameterize complex human behaviors using low-dimensional goal positions, which are then fed into a promptable traffic simulator, ProSim, to guide the behaviors of simulated agents. To automate test generation, we introduce a prompt generation module that explores the goal domain and efficiently identifies safety-critical behaviors using Bayesian optimization. We apply our method to the evaluation of an optimization-based planner and demonstrate its effectiveness and efficiency in automatically generating diverse and realistic driving behaviors across scenarios with varying initial conditions. 

**Abstract (ZH)**: 自主车辆（AV）规划器在广泛部署于公共道路之前必须经过严格的评估，特别是要评估其在面对人类行为不确定性时的鲁棒性。虽然近期基于数据的场景生成技术能够模拟交互环境中的真实人类行为，但利用这些模型为AV规划器构建全面的测试仍然是一项开放的挑战。本文介绍了一种自动化方法，用于高效生成用于评估交互场景中AV规划器的现实且安全关键的人类行为。我们使用低维度的目标位置参数化复杂的_human行为，并将这些参数输入可提示的交通模拟器ProSim以引导模拟代理的行为。为实现测试生成的自动化，我们引入了一个提示生成模块，通过贝叶斯优化高效地探索目标领域并识别安全关键行为。我们将该方法应用于基于优化的规划器的评估，并展示了其在自动生成跨不同初始条件场景下的多样且真实驾驶行为方面的有效性与效率。 

---
# GraphPad: Inference-Time 3D Scene Graph Updates for Embodied Question Answering 

**Title (ZH)**: GraphPad: 语义理解时的3D场景图更新方法在体帧问答中的应用 

**Authors**: Muhammad Qasim Ali, Saeejith Nair, Alexander Wong, Yuchen Cui, Yuhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01174)  

**Abstract**: Structured scene representations are a core component of embodied agents, helping to consolidate raw sensory streams into readable, modular, and searchable formats. Due to their high computational overhead, many approaches build such representations in advance of the task. However, when the task specifications change, such static approaches become inadequate as they may miss key objects, spatial relations, and details. We introduce GraphPad, a modifiable structured memory that an agent can tailor to the needs of the task through API calls. It comprises a mutable scene graph representing the environment, a navigation log indexing frame-by-frame content, and a scratchpad for task-specific notes. Together, GraphPad serves as a dynamic workspace that remains complete, current, and aligned with the agent's immediate understanding of the scene and its task. On the OpenEQA benchmark, GraphPad attains 55.3%, a +3.0% increase over an image-only baseline using the same vision-language model, while operating with five times fewer input frames. These results show that allowing online, language-driven refinement of 3-D memory yields more informative representations without extra training or data collection. 

**Abstract (ZH)**: 基于图的可修改结构化记忆：一种通过API调用适应任务需求的动态工作空间 

---
# ChemAU: Harness the Reasoning of LLMs in Chemical Research with Adaptive Uncertainty Estimation 

**Title (ZH)**: ChemAU: 利用自适应不确定性估计在化学研究中 harness LLMs 的推理能力 

**Authors**: Xinyi Liu, Lipeng Ma, Yixuan Li, Weidong Yang, Qingyuan Zhou, Jiayi Song, Shuhao Li, Ben Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.01116)  

**Abstract**: Large Language Models (LLMs) are widely used across various scenarios due to their exceptional reasoning capabilities and natural language understanding. While LLMs demonstrate strong performance in tasks involving mathematics and coding, their effectiveness diminishes significantly when applied to chemistry-related problems. Chemistry problems typically involve long and complex reasoning steps, which contain specific terminology, including specialized symbol systems and complex nomenclature conventions. These characteristics often cause general LLMs to experience hallucinations during the reasoning process due to their lack of specific knowledge. However, existing methods are struggling to effectively leverage chemical expertise and formulas. Moreover, current uncertainty estimation methods, designed to mitigate potential reasoning errors, are unable to precisely identify specific steps or key knowledge. In this work, we propose a novel framework called ChemAU, which incorporates our adaptive uncertainty estimation method that applies different uncertainty values based on the position of reasoning steps within the whole reasoning chain. Leveraging this method, ChemAU identifies gaps in chemistry knowledge and precisely supplements chemical expertise with the specialized domain model, thereby correcting and updating the previously flawed reasoning chain. Our experiments with three popular LLMs across three chemistry datasets demonstrate that ChemAU significantly enhances both reasoning accuracy and uncertainty estimation. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种场景中广泛应用，得益于其卓越的推理能力和自然语言理解能力。虽然LLMs在涉及数学和编程的任务中表现出色，但在应用于化学相关问题时，其有效性显著下降。化学问题通常涉及复杂的推理步骤，包含特定的专业术语、特殊符号系统和复杂的命名 conventions。这些特点往往导致一般性的LLMs在推理过程中出现幻觉，因为它们缺乏具体的知识。然而，现有的方法难以有效利用化学专业知识和公式。此外，当前用于降低推理错误的风险估计方法，无法精确识别具体步骤或关键知识。在本工作中，我们提出了一种名为ChemAU的新框架，该框架结合了我们根据整个推理链中推理步骤的位置应用不同不确定性值的自适应不确定性估计方法。通过这种方法，ChemAU能够识别化学知识的空白，并精确补充化学专业知识，从而纠正和更新先前错误的推理链。我们在三个流行的LLM和三个化学数据集中进行的实验表明，ChemAU显著提高了推理准确性和不确定性估计。 

---
# SuperRL: Reinforcement Learning with Supervision to Boost Language Model Reasoning 

**Title (ZH)**: SuperRL：带有监督的强化学习以增强语言模型推理能力 

**Authors**: Yihao Liu, Shuocheng Li, Lang Cao, Yuhang Xie, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01096)  

**Abstract**: Large language models are increasingly used for complex reasoning tasks where high-quality offline data such as expert-annotated solutions and distilled reasoning traces are often available. However, in environments with sparse rewards, reinforcement learning struggles to sample successful trajectories, leading to inefficient learning. At the same time, these offline trajectories that represent correct reasoning paths are not utilized by standard on-policy reinforcement learning methods. To address this limitation, we propose SuperRL, a unified training framework that adaptively incorporates offline supervision into reinforcement learning. SuperRL introduces an Adaptive Switch to detect sparse reward conditions and activates a Hybrid Actor when necessary. The Hybrid Actor integrates policy gradient and supervised learning objectives at the loss level, enabling the model to benefit from accurate offline reasoning signals while maintaining the exploratory capacity of reinforcement learning. Experiments on a range of reasoning benchmarks show that SuperRL consistently outperforms standard reinforcement learning by improving sample efficiency, generalization, and robustness under sparse rewards. 

**Abstract (ZH)**: SuperRL：统一训练框架下的自适应离线监督强化学习 

---
# Modular Speaker Architecture: A Framework for Sustaining Responsibility and Contextual Integrity in Multi-Agent AI Communication 

**Title (ZH)**: 模块化说话人架构：多智能体AI通信中保持责任和上下文完整性的一种框架 

**Authors**: Khe-Han Toh, Hong-Kuan Teo  

**Link**: [PDF](https://arxiv.org/pdf/2506.01095)  

**Abstract**: Sustaining coherent, role-aware communication across multi-agent systems remains a foundational challenge in AI. Current frameworks often lack explicit mechanisms for speaker responsibility, leading to context drift, alignment instability, and degraded interpretability over time. We propose the Modular Speaker Architecture (MSA), a framework that decomposes speaker behavior into modular components for role tracking, responsibility continuity, and contextual coherence. Grounded in high-context human-AI dialogues, MSA includes three core modules: a Speaker Role Module, a Responsibility Chain Tracker, and a Contextual Integrity Validator. We evaluate MSA through annotated case studies and introduce structural metrics-pragmatic consistency, responsibility flow, and context stability-quantified via manual and automatic scoring and bootstrapped statistical analysis. Our results show that MSA reliably maintains interaction structure without reliance on affective signals or surface-level heuristics. We further implement a prototype configuration language (G-Code) and modular API to support MSA deployment in dynamic multi-agent scenarios. 

**Abstract (ZH)**: 在多智能体系统中维持一致且角色意识的通信依然是AI领域的基础挑战。现有框架通常缺乏明确的说话人责任机制，导致情景漂移、对齐不稳定以及随着时间推移降低的可解释性。我们提出了模块化说话人架构（MSA），一种将说话人行为分解为用于角色跟踪、责任连续性和内容连贯性模块化组件的框架。基于高情景人类-AI对话，MSA 包含三个核心模块：说话人角色模块、责任链追踪器和内容完整性验证器。我们通过标注案例研究评估了 MSA，并引入了结构度量——语用一致性、责任流动性和情境稳定性——通过人工和自动评分以及自助统计分析进行量化。我们的结果表明，MSA 能可靠地维护交互结构，无需依赖情感信号或表面级启发式方法。我们进一步实现了一个原型配置语言（G-Code）和模块化 API，以支持 MSA 在动态多智能体场景中的部署。 

---
# Regulatory Graphs and GenAI for Real-Time Transaction Monitoring and Compliance Explanation in Banking 

**Title (ZH)**: 监管图和GenAI在银行业实时交易监控及合规解释中的应用 

**Authors**: Kunal Khanvilkar, Kranthi Kommuru  

**Link**: [PDF](https://arxiv.org/pdf/2506.01093)  

**Abstract**: This paper presents a real-time transaction monitoring framework that integrates graph-based modeling, narrative field embedding, and generative explanation to support automated financial compliance. The system constructs dynamic transaction graphs, extracts structural and contextual features, and classifies suspicious behavior using a graph neural network. A retrieval-augmented generation module generates natural language explanations aligned with regulatory clauses for each flagged transaction. Experiments conducted on a simulated stream of financial data show that the proposed method achieves superior results, with 98.2% F1-score, 97.8% precision, and 97.0% recall. Expert evaluation further confirms the quality and interpretability of generated justifications. The findings demonstrate the potential of combining graph intelligence and generative models to support explainable, audit-ready compliance in high-risk financial environments. 

**Abstract (ZH)**: 基于图模型、叙事场嵌入和生成性解释的实时交易监测框架：支持自动化金融合规性管理 

---
# Choices and their Provenance: Explaining Stable Solutions of Abstract Argumentation Frameworks 

**Title (ZH)**: 选择及其来源：解释抽象论辩框架的稳定解 

**Authors**: Bertram Ludäscher, Yilin Xia, Shawn Bowers  

**Link**: [PDF](https://arxiv.org/pdf/2506.01087)  

**Abstract**: The rule $\mathrm{Defeated}(x) \leftarrow \mathrm{Attacks}(y,x),\, \neg \, \mathrm{Defeated}(y)$, evaluated under the well-founded semantics (WFS), yields a unique 3-valued (skeptical) solution of an abstract argumentation framework (AF). An argument $x$ is defeated ($\mathrm{OUT}$) if there exists an undefeated argument $y$ that attacks it. For 2-valued (stable) solutions, this is the case iff $y$ is accepted ($\mathrm{IN}$), i.e., if all of $y$'s attackers are defeated. Under WFS, arguments that are neither accepted nor defeated are undecided ($\mathrm{UNDEC}$). As shown in prior work, well-founded solutions (a.k.a. grounded labelings) "explain themselves": The provenance of arguments is given by subgraphs (definable via regular path queries) rooted at the node of interest. This provenance is closely related to winning strategies of a two-player argumentation game.
We present a novel approach for extending this provenance to stable AF solutions. Unlike grounded solutions, which can be constructed via a bottom-up alternating fixpoint procedure, stable models often involve non-deterministic choice as part of the search for models. Thus, the provenance of stable solutions is of a different nature, and reflects a more expressive generate & test paradigm. Our approach identifies minimal sets of critical attacks, pinpointing choices and assumptions made by a stable model. These critical attack edges provide additional insights into the provenance of an argument's status, combining well-founded derivation steps with choice steps. Our approach can be understood as a form of diagnosis that finds minimal "repairs" to an AF graph such that the well-founded solution of the repaired graph coincides with the desired stable model of the original AF graph. 

**Abstract (ZH)**: 基于有序语义的稳固解的来源分析：一种新颖的扩展方法 

---
# The Coming Crisis of Multi-Agent Misalignment: AI Alignment Must Be a Dynamic and Social Process 

**Title (ZH)**: 多智能体偏差危机：AI 对齐必须是一个动态且社会性过程 

**Authors**: Florian Carichon, Aditi Khandelwal, Marylou Fauchard, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01080)  

**Abstract**: This position paper states that AI Alignment in Multi-Agent Systems (MAS) should be considered a dynamic and interaction-dependent process that heavily depends on the social environment where agents are deployed, either collaborative, cooperative, or competitive. While AI alignment with human values and preferences remains a core challenge, the growing prevalence of MAS in real-world applications introduces a new dynamic that reshapes how agents pursue goals and interact to accomplish various tasks. As agents engage with one another, they must coordinate to accomplish both individual and collective goals. However, this complex social organization may unintentionally misalign some or all of these agents with human values or user preferences. Drawing on social sciences, we analyze how social structure can deter or shatter group and individual values. Based on these analyses, we call on the AI community to treat human, preferential, and objective alignment as an interdependent concept, rather than isolated problems. Finally, we emphasize the urgent need for simulation environments, benchmarks, and evaluation frameworks that allow researchers to assess alignment in these interactive multi-agent contexts before such dynamics grow too complex to control. 

**Abstract (ZH)**: AI对多智能体系统中的对齐应被视为一个动态且依赖交互的社会过程，该过程高度依赖于部署智能体的社会环境，无论是协作、合作还是竞争。随着多智能体系统在现实世界应用中的日益普遍，智能体追求目标和互动以完成各种任务的方式也在发生变化。智能体相互作用时，必须协调以实现个体和集体目标。然而，这种复杂的社会组织可能无意中使一些或所有智能体偏离了人类价值观或用户偏好。借鉴社会科学，我们分析了社会结构如何阻止或粉碎群体和个人的价值。基于这些分析，我们呼吁AI社区将人类偏好和客观对齐视为相互依赖的概念，而不仅仅是孤立的问题。最后，我们强调亟需模拟环境、基准和评估框架，以便研究人员在这些交互多智能体上下文中的动力学变得难以控制之前对其进行评估。 

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
# Higher-Order Responsibility 

**Title (ZH)**: 高层次责任 

**Authors**: Junli Jiang, Pavel Naumov  

**Link**: [PDF](https://arxiv.org/pdf/2506.01003)  

**Abstract**: In ethics, individual responsibility is often defined through Frankfurt's principle of alternative possibilities. This definition is not adequate in a group decision-making setting because it often results in the lack of a responsible party or "responsibility gap''. One of the existing approaches to address this problem is to consider group responsibility. Another, recently proposed, approach is "higher-order'' responsibility. The paper considers the problem of deciding if higher-order responsibility up to degree $d$ is enough to close the responsibility gap. The main technical result is that this problem is $\Pi_{2d+1}$-complete. 

**Abstract (ZH)**: 在伦理学中，个体责任通常通过弗兰克福的替代可能性原则来定义。这一定义在群体决策环境中往往不够充分，因为它经常导致责任空缺或“责任缺口”。现有的一个解决方案是考虑群体责任。另一种最近提出的解决方案是“高阶”责任。本文探讨了决定最高阶为$d$的高阶责任是否足以填补责任缺口的问题。主要的技术结果是这个问题是$\Pi_{2d+1}$-完全的。 

---
# Boosting Bot Detection via Heterophily-Aware Representation Learning and Prototype-Guided Cluster Discovery 

**Title (ZH)**: 通过异质性意识表示学习和原型引导的聚类发现增强机器人检测 

**Authors**: Buyun He, Xiaorui Jiang, Qi Wu, Hao Liu, Yingguang Yang, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00989)  

**Abstract**: Detecting social media bots is essential for maintaining the security and trustworthiness of social networks. While contemporary graph-based detection methods demonstrate promising results, their practical application is limited by label reliance and poor generalization capability across diverse communities. Generative Graph Self-Supervised Learning (GSL) presents a promising paradigm to overcome these limitations, yet existing approaches predominantly follow the homophily assumption and fail to capture the global patterns in the graph, which potentially diminishes their effectiveness when facing the challenges of interaction camouflage and distributed deployment in bot detection scenarios. To this end, we propose BotHP, a generative GSL framework tailored to boost graph-based bot detectors through heterophily-aware representation learning and prototype-guided cluster discovery. Specifically, BotHP leverages a dual-encoder architecture, consisting of a graph-aware encoder to capture node commonality and a graph-agnostic encoder to preserve node uniqueness. This enables the simultaneous modeling of both homophily and heterophily, effectively countering the interaction camouflage issue. Additionally, BotHP incorporates a prototype-guided cluster discovery pretext task to model the latent global consistency of bot clusters and identify spatially dispersed yet semantically aligned bot collectives. Extensive experiments on two real-world bot detection benchmarks demonstrate that BotHP consistently boosts graph-based bot detectors, improving detection performance, alleviating label reliance, and enhancing generalization capability. 

**Abstract (ZH)**: 检测社交媒体僵尸账户对于维护社交网络的安全性和可信度至关重要。虽然基于图的检测方法表现出色，但它们的实际应用受限于标签依赖和跨不同社区的泛化能力较差的问题。生成式图自监督学习（GSL）提出了克服这些限制的有前景的方法论，然而现有的方法大多遵循同质性假设，并且未能捕捉到图的整体模式，这在面对僵尸账户检测中的相互伪装和分布式部署挑战时可能削弱其效果。为了解决这些问题，我们提出了BotHP，这是一种生成式GSL框架，旨在通过异质性感知表征学习和原型引导聚类发现来增强基于图的僵尸账户检测器。具体而言，BotHP 利用了一种双编码器结构，包括一个图感知编码器来捕捉节点的共同性，和一个图无感知编码器来保留节点的独特性。这使得同时建模同质性和异质性成为可能，并有效地应对了相互伪装的问题。此外，BotHP 还引入了基于原型引导聚类发现的预训练任务，以建模僵尸账户聚类的潜在全球一致性，并识别那些在空间上分散但语义上一致的僵尸账户集合。在两个真实世界的僵尸账户检测基准上的广泛实验表明，BotHP 一致地增强了基于图的僵尸账户检测器，提升了检测性能，缓解了标签依赖，并增强了泛化能力。 

---
# PolyBERT: Fine-Tuned Poly Encoder BERT-Based Model for Word Sense Disambiguation 

**Title (ZH)**: PolyBERT： fine-tuned 多编码器 BERT 基础模型用于单词意义消歧Resolve 

**Authors**: Linhan Xia, Mingzhan Yang, Guohui Yuan, Shengnan Tao, Yujing Qiu, Guo Yu, Kai Lei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00968)  

**Abstract**: Mainstream Word Sense Disambiguation (WSD) approaches have employed BERT to extract semantics from both context and definitions of senses to determine the most suitable sense of a target word, achieving notable performance. However, there are two limitations in these approaches. First, previous studies failed to balance the representation of token-level (local) and sequence-level (global) semantics during feature extraction, leading to insufficient semantic representation and a performance bottleneck. Second, these approaches incorporated all possible senses of each target word during the training phase, leading to unnecessary computational costs. To overcome these limitations, this paper introduces a poly-encoder BERT-based model with batch contrastive learning for WSD, named PolyBERT. Compared with previous WSD methods, PolyBERT has two improvements: (1) A poly-encoder with a multi-head attention mechanism is utilized to fuse token-level (local) and sequence-level (global) semantics, rather than focusing on just one. This approach enriches semantic representation by balancing local and global semantics. (2) To avoid redundant training inputs, Batch Contrastive Learning (BCL) is introduced. BCL utilizes the correct senses of other target words in the same batch as negative samples for the current target word, which reduces training inputs and computational cost. The experimental results demonstrate that PolyBERT outperforms baseline WSD methods such as Huang's GlossBERT and Blevins's BEM by 2\% in F1-score. In addition, PolyBERT with BCL reduces GPU hours by 37.6\% compared with PolyBERT without BCL. 

**Abstract (ZH)**: 主流词义消歧方法通过BERT提取上下文和词义定义中的语义，以确定目标词的最适宜词义，并取得了显著性能。然而，这些方法存在两个局限性。首先，先前的研究未能在特征提取过程中平衡词粒度（局部）和序列粒度（全局）语义的表示，导致语义表示不足和性能瓶颈。其次，这些方法在训练阶段整合了每个目标词的所有可能词义，导致不必要的计算成本。为克服这些局限性，本文提出了一种基于BERT的多编码器模型，并结合了批量对比学习，命名为PolyBERT。相较于之前的词义消歧方法，PolyBERT有两大改进：（1）利用带有多重注意力机制的多编码器融合词粒度（局部）和序列粒度（全局）语义，而非仅仅关注其中一种。这种方法通过平衡局部和全局语义丰富了语义表示。（2）为避免冗余的训练输入，引入了批量对比学习（BCL）。BCL使用同一批处理中其他目标词的正确词义作为当前目标词的负样本，从而减少训练输入和计算成本。实验证明，PolyBERT在F1分数上比黄氏GlossBERT和布林斯氏BEM等基线方法高出2%。此外，使用BCL的PolyBERT相比不使用BCL的PolyBERT减少了37.6%的GPU小时。 

---
# Unlocking Personalized Knowledge in Federated Large Language Model: The Power of Mixture of Experts 

**Title (ZH)**: 解锁联邦大型语言模型中的个性化知识：混合专家的力量 

**Authors**: Fan Liu, Bikang Pan, Zhongyi Wang, Xi Yao, Xiaoying Tang, Jingya Wang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00965)  

**Abstract**: The Mixture of Experts (MoE) architecture has emerged as a prominent strategy for scaling large language models (LLMs), effectively leveraging sparse activation and facilitating task-specific personalization. However, current federated learning (FL) approaches are primarily designed for dense models, making them unable to directly exploit the sparsity inherent in MoE architectures. Treating MoE models as dense networks in federated scenarios results in excessive communication overhead and computational costs, undermining the potential for personalized knowledge sharing. To address these challenges, we propose FLEx (Federated LLMs with Personalized Experts), a novel federated learning framework explicitly tailored for MoE-based LLMs. FLEx efficiently personalizes by pruning the global MoE model to keep only one expert per client, and employs an adaptive gating mechanism to reintegrate these personalized experts into the pre-trained MoE layers, ensuring the original backbone architecture remains unchanged. These personalized experts are trained with local data and stored locally on each client, while the shared modules are aggregated globally. Extensive evaluations on diverse instruction-based datasets under non-IID conditions consistently demonstrate that FLEx outperforms existing federated baselines. Our code is available at this https URL. 

**Abstract (ZH)**: FLEx（面向混合专家的联邦大语言模型个性化框架） 

---
# Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues 

**Title (ZH)**: 超越语言的表达：面向视频-grounded 对话学习非言语线索的大规模多模态数据集 

**Authors**: Youngmin Kim, Jiwan Chung, Jisoo Kim, Sunghyun Lee, Sangkyu Lee, Junhyeok Kim, Cheoljong Yang, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00958)  

**Abstract**: Nonverbal communication is integral to human interaction, with gestures, facial expressions, and body language conveying critical aspects of intent and emotion. However, existing large language models (LLMs) fail to effectively incorporate these nonverbal elements, limiting their capacity to create fully immersive conversational experiences. We introduce MARS, a multimodal language model designed to understand and generate nonverbal cues alongside text, bridging this gap in conversational AI. Our key innovation is VENUS, a large-scale dataset comprising annotated videos with time-aligned text, facial expressions, and body language. Leveraging VENUS, we train MARS with a next-token prediction objective, combining text with vector-quantized nonverbal representations to achieve multimodal understanding and generation within a unified framework. Based on various analyses of the VENUS datasets, we validate its substantial scale and high effectiveness. Our quantitative and qualitative results demonstrate that MARS successfully generates text and nonverbal languages, corresponding to conversational input. 

**Abstract (ZH)**: 非言语沟通是人类互动的重要组成部分，手势、面部表情和身体语言传达着意图和情感的关键方面。然而，现有的大型语言模型（LLMs）未能有效地融入这些非言语元素，限制了它们创造沉浸式对话体验的能力。我们提出了MARS，一个能够理解和生成非言语暗示的多模态语言模型，从而弥合对话AI的这一空白。我们的核心创新是VENUS，一个包含注释视频的大型数据集，这些视频的时间对齐文本、面部表情和身体语言进行了标注。借助VENUS，我们使用下一个词预测目标训练MARS，将文本与矢量量化非言语表示结合，实现了统一框架内的多模态理解和生成。基于VENUS数据集的各种分析，我们验证了其庞大的规模和高度的有效性。我们的定量和定性结果表明，MARS能够生成与对话输入对应的文本和非言语语言。 

---
# Aligning VLM Assistants with Personalized Situated Cognition 

**Title (ZH)**: 个性化情境认知中的VLM辅助系统对齐 

**Authors**: Yongqi Li, Shen Zhou, Xiaohu Li, Xin Miao, Jintao Wen, Mayi Xu, Jianhao Chen, Birong Pan, Hankun Kang, Yuanyuan Zhu, Ming Zhong, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.00930)  

**Abstract**: Vision-language models (VLMs) aligned with general human objectives, such as being harmless and hallucination-free, have become valuable assistants of humans in managing visual tasks. However, people with diversified backgrounds have different cognition even in the same situation. Consequently, they may have personalized expectations for VLM assistants. This highlights the urgent need to align VLM assistants with personalized situated cognition for real-world assistance. To study this problem, we first simplify it by characterizing individuals based on the sociological concept of Role-Set. Then, we propose to evaluate the individuals' actions to examine whether the personalized alignment is achieved. Further, we construct a benchmark named PCogAlignBench, which includes 18k instances and 20 individuals with different Role-Sets. Finally, we present a framework called PCogAlign, which constructs a cognition-aware and action-based reward model for personalized alignment. Experimental results and human evaluations demonstrate the reliability of the PCogAlignBench and the effectiveness of our proposed PCogAlign. We will open-source the constructed benchmark and code at this https URL. 

**Abstract (ZH)**: Vision-language模型（VLMs）与普遍人类目标对齐，如无害和无幻觉，已成为人类在管理视觉任务中有价值的助理。然而，具有不同背景的人在同一情境下可能有不同的认知，因此他们可能对VLM助理有个性化的期望。这突显了在真实世界协助中对VLM助理进行个性化情境对齐的迫切需求。为研究这一问题，我们首先根据社会学概念Role-Set对该问题进行简化，基于此对个体进行刻画，然后提出通过评估个体的行为来检验个性化对齐是否实现。进一步地，我们构建了一个基准PCogAlignBench，其中包括18000个实例和20名具有不同Role-Set的个体。最后，我们提出了一种名为PCogAlign的框架，该框架构建了一个基于认知和行为的个性化对齐奖励模型。实验结果和人工评估证明了PCogAlignBench的可靠性和我们提出的PCogAlign的有效性。我们将在此网址公开所构建的基准和代码：this https URL。 

---
# Conformal Arbitrage: Risk-Controlled Balancing of Competing Objectives in Language Models 

**Title (ZH)**: conformal arbitrage：在语言模型中控制风险的 competing 目标平衡 

**Authors**: William Overman, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2506.00911)  

**Abstract**: Modern language model deployments must often balance competing objectives, for example, helpfulness versus harmlessness, cost versus accuracy, and reward versus safety. We introduce Conformal Arbitrage, a post hoc framework that learns a data driven threshold to mediate between a Primary model optimized for a primary objective and a more conservative Guardian which could be another model or a human domain expert aligned with a guardrail objective. The threshold is calibrated with conformal risk control, yielding finite sample, distribution free guarantees that the long run frequency of undesirable events, such as factual errors or safety violations, does not exceed a user specified quota. Because Conformal Arbitrage operates wholly at the API level, without requiring access to model logits or updating model weights, it complements weight based alignment techniques and integrates seamlessly with existing cost aware cascades. Empirically, Conformal Arbitrage traces an efficient frontier, allowing users to define an acceptable performance level for one objective while maximizing utility in another. We observe that our method outperforms, in terms of accuracy, cost matched random routing between models. These properties make Conformal Arbitrage a practical, theoretically grounded tool for trustworthy and economical deployment of large language models across a broad range of potentially competing objectives. 

**Abstract (ZH)**: 现代语言模型部署必须平衡多种互斥目标，例如有益性与安全性、成本与准确性、奖励与安全性。我们介绍了一种后验框架——Conformal Arbitrage，该框架学习一个基于数据的阈值，以调解专门为某种目标优化的主模型（Primary model）与一个更为保守的守护者模型（Guardian），后者可能是另一个模型或与护栏目标保持一致的人类领域专家。该阈值通过符合风险控制进行校准，从而在有限样本、分布无关的情况下保证长期频率不佳事件（如事实性错误或安全性违规）的发生率不超过用户指定的限额。由于Conformal Arbitrage完全在API层面运作，无需访问模型logits或更新模型权重，因此它补充了基于权重的对齐技术，并能无缝集成到现有的成本感知级联中。实证研究显示，Conformal Arbitrage能够在保持一个目标性能的同时，最大化另一个目标的效用。我们发现，这种方法在准确性方面优于模型间按成本匹配的随机路由。这些特性使得Conformal Arbitrage成为一个实用且有理论支撑的工具，可用于众多潜在互斥目标下大规模语言模型的可靠且经济的部署。 

---
# Toward a Theory of Agents as Tool-Use Decision-Makers 

**Title (ZH)**: 面向代理作为工具使用决策者的理论研究 

**Authors**: Hongru Wang, Cheng Qian, Manling Li, Jiahao Qiu, Boyang Xue, Mengdi Wang, Heng Ji, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.00886)  

**Abstract**: As Large Language Models (LLMs) evolve into increasingly autonomous agents, fundamental questions about their epistemic foundations remain unresolved: What defines an agent? How should it make decisions? And what objectives should guide its behavior? In this position paper, we argue that true autonomy requires agents to be grounded in a coherent epistemic framework that governs what they know, what they need to know, and how to acquire that knowledge efficiently. We propose a unified theory that treats internal reasoning and external actions as equivalent epistemic tools, enabling agents to systematically coordinate introspection and interaction. Building on this framework, we advocate for aligning an agent's tool use decision-making boundary with its knowledge boundary, thereby minimizing unnecessary tool use and maximizing epistemic efficiency. This perspective shifts the design of agents from mere action executors to knowledge-driven intelligence systems, offering a principled path toward building foundation agents capable of adaptive, efficient, and goal-directed behavior. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）演变为日益自主的代理，关于其知识基础的基本问题仍未解决：什么是代理？它应如何做出决策？其行为应由什么目标指导？在本文中，我们argue，真正的自主性要求代理具备一套连贯的知识框架，以规范其认知、所需认知及其知识获取的效率。我们提出了一种统一理论，将内部推理和外部行动视为等价的知识工具，使代理能够系统地协调内省和互动。基于此框架，我们提倡将代理的工具使用决策边界与其知识边界对齐，从而减少不必要的工具使用，最大化知识效率。这种观点将代理的设计从单纯的行动执行者转变为以知识为导向的智能系统，提供了构建能够实现自适应、高效和目标导向行为的基础代理的原理性路径。 

---
# GIA-MIC: Multimodal Emotion Recognition with Gated Interactive Attention and Modality-Invariant Learning Constraints 

**Title (ZH)**: GIA-MIC：基于门控交互注意力和模态不变学习约束的多模态情感识别 

**Authors**: Jiajun He, Jinyi Mi, Tomoki Toda  

**Link**: [PDF](https://arxiv.org/pdf/2506.00865)  

**Abstract**: Multimodal emotion recognition (MER) extracts emotions from multimodal data, including visual, speech, and text inputs, playing a key role in human-computer interaction. Attention-based fusion methods dominate MER research, achieving strong classification performance. However, two key challenges remain: effectively extracting modality-specific features and capturing cross-modal similarities despite distribution differences caused by modality heterogeneity. To address these, we propose a gated interactive attention mechanism to adaptively extract modality-specific features while enhancing emotional information through pairwise interactions. Additionally, we introduce a modality-invariant generator to learn modality-invariant representations and constrain domain shifts by aligning cross-modal similarities. Experiments on IEMOCAP demonstrate that our method outperforms state-of-the-art MER approaches, achieving WA 80.7% and UA 81.3%. 

**Abstract (ZH)**: 多模态情绪识别（MER）从视觉、语音和文本等多种模态数据中提取情绪，对人机交互起着关键作用。基于注意力的融合方法主导着MER研究，取得了强大的分类性能。然而，仍存在两个关键挑战：有效提取模态特定特征以及在模态异质性导致的分布差异下捕获跨模态相似性。为解决这些问题，我们提出了一种门控交互注意力机制，以适应性地提取模态特定特征并通过对等交互增强情感信息。此外，我们引入了一种模态不变生成器，学习模态不变表示，并通过对齐跨模态相似性来约束领域移位。实验结果表明，我们的方法在IEMOCAP数据集上优于最先进的MER方法，实现了加权平均准确率80.7%和未标定平均准确率81.3%。 

---
# MedBookVQA: A Systematic and Comprehensive Medical Benchmark Derived from Open-Access Book 

**Title (ZH)**: MedBookVQA: 一个源自开放访问书籍的系统性和综合性的医疗基准体系 

**Authors**: Sau Lai Yip, Sunan He, Yuxiang Nie, Shu Pui Chan, Yilin Ye, Sum Ying Lam, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00855)  

**Abstract**: The accelerating development of general medical artificial intelligence (GMAI), powered by multimodal large language models (MLLMs), offers transformative potential for addressing persistent healthcare challenges, including workforce deficits and escalating costs. The parallel development of systematic evaluation benchmarks emerges as a critical imperative to enable performance assessment and provide technological guidance. Meanwhile, as an invaluable knowledge source, the potential of medical textbooks for benchmark development remains underexploited. Here, we present MedBookVQA, a systematic and comprehensive multimodal benchmark derived from open-access medical textbooks. To curate this benchmark, we propose a standardized pipeline for automated extraction of medical figures while contextually aligning them with corresponding medical narratives. Based on this curated data, we generate 5,000 clinically relevant questions spanning modality recognition, disease classification, anatomical identification, symptom diagnosis, and surgical procedures. A multi-tier annotation system categorizes queries through hierarchical taxonomies encompassing medical imaging modalities (42 categories), body anatomies (125 structures), and clinical specialties (31 departments), enabling nuanced analysis across medical subdomains. We evaluate a wide array of MLLMs, including proprietary, open-sourced, medical, and reasoning models, revealing significant performance disparities across task types and model categories. Our findings highlight critical capability gaps in current GMAI systems while establishing textbook-derived multimodal benchmarks as essential evaluation tools. MedBookVQA establishes textbook-derived benchmarking as a critical paradigm for advancing clinical AI, exposing limitations in GMAI systems while providing anatomically structured performance metrics across specialties. 

**Abstract (ZH)**: 由多模态大规模语言模型驱动的一般医疗人工智能的加速发展为应对持续存在的医疗保健挑战（包括劳动力短缺和成本上升）提供了变革性的潜力。随着系统评价基准的协同发展，性能评估和提供技术指导变得至关重要。与此同时，医学教科书作为宝贵的知识来源，其在基准开发中的潜力尚未被充分开发。在这里，我们提出了MedBookVQA，这是一个源自开放获取医学教科书的系统性和综合性的多模态基准。为了编纂这个基准，我们提出了一套标准化的工作流程，用于自动化提取医学图像并上下文性地与相应的医学叙述进行对齐。基于这些精选数据，我们生成了5000个临床相关问题，涵盖了模态识别、疾病分类、解剖学识别、症状诊断和外科手术等方面。通过多层次的标注系统，我们将查询分类到涵盖医学影像模态（42类）、人体解剖学（125种结构）和临床专业（31个部门）的层级分类学中，从而在医学子领域实现精细化分析。我们评估了一系列多模态大型语言模型，包括专有、开源、医疗和推理模型，揭示了不同任务类型和模型类别之间显著的性能差异。我们的研究结果突显了当前一般医疗人工智能系统的关键能力缺口，并确立了教科书衍生的多模态基准作为重要的评估工具。MedBookVQA 建立了教科书衍生基准的重要性范式，揭示了一般医疗人工智能系统的局限性，并提供了跨专科的解剖结构化性能指标。 

---
# SynPO: Synergizing Descriptiveness and Preference Optimization for Video Detailed Captioning 

**Title (ZH)**: SynPO: 结合描述性和偏好优化的视频详细caption生成 

**Authors**: Jisheng Dang, Yizhou Zhang, Hao Ye, Teng Wang, Siming Chen, Huicheng Zheng, Yulan Guo, Jianhuang Lai, Bin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00835)  

**Abstract**: Fine-grained video captioning aims to generate detailed, temporally coherent descriptions of video content. However, existing methods struggle to capture subtle video dynamics and rich detailed information. In this paper, we leverage preference learning to enhance the performance of vision-language models in fine-grained video captioning, while mitigating several limitations inherent to direct preference optimization (DPO). First, we propose a pipeline for constructing preference pairs that leverages the intrinsic properties of VLMs along with partial assistance from large language models, achieving an optimal balance between cost and data quality. Second, we propose Synergistic Preference Optimization (SynPO), a novel optimization method offering significant advantages over DPO and its variants. SynPO prevents negative preferences from dominating the optimization, explicitly preserves the model's language capability to avoid deviation of the optimization objective, and improves training efficiency by eliminating the need for the reference model. We extensively evaluate SynPO not only on video captioning benchmarks (e.g., VDC, VDD, VATEX) but also across well-established NLP tasks, including general language understanding and preference evaluation, using diverse pretrained models. Results demonstrate that SynPO consistently outperforms DPO variants while achieving 20\% improvement in training efficiency. Code is available at this https URL 

**Abstract (ZH)**: 细粒度视频字幕生成旨在生成视频内容的详细且时间连贯的描述。然而，现有方法在捕捉微妙的视频动态和丰富的详细信息方面存在局限。在本文中，我们利用偏好学习来增强视觉-语言模型在细粒度视频字幕生成中的性能，同时缓解直接偏好优化（DPO）固有的若干局限。首先，我们提出了一种管道来构建偏好对，该管道利用视觉-语言模型的内在特性，并部分借助大型语言模型的帮助，实现成本和数据质量的最佳平衡。其次，我们提出了一种新的优化方法——协同偏好优化（SynPO），该方法在与其他DPO及其变体相比时显示出显著优势。SynPO可以防止负偏好主导优化过程，明确保留模型的语言能力以避免优化目标的偏移，并通过消除参考模型的需要来提高训练效率。我们在视频字幕基准（如VDC、VDD、VATEX）以及广泛认可的NLP任务（包括通用语言理解与偏好评估）上，使用多种预训练模型进行了广泛的评估。结果表明，SynPO持续优于DPO变体，并在训练效率上提高了20%。代码可在以下链接获取：这个 https URL。 

---
# Enhancing LLM Reasoning for Time Series Classification by Tailored Thinking and Fused Decision 

**Title (ZH)**: 针对时间序列分类的定制化思考与融合决策增强大型语言模型推理能力 

**Authors**: Jiahui Zhou, Dan Li, Lin Li, Zhuomin Chen, Shunyu Wu, Haozheng Ye, Jian Lou, Costas J. Spanos  

**Link**: [PDF](https://arxiv.org/pdf/2506.00807)  

**Abstract**: The reasoning capabilities of large language models (LLMs) have significantly advanced their performance by enabling in-depth understanding of diverse tasks. With growing interest in applying LLMs to the time series domain, this has proven nontrivial, as evidenced by the limited efficacy of straightforwardly adapting text-domain reasoning techniques. Although recent work has shown promise in several time series tasks, further leveraging advancements in LLM reasoning remains under-explored for time series classification (TSC) tasks, despite their prevalence and significance in many real-world applications. In this paper, we propose ReasonTSC, a novel framework designed to effectively leverage LLM reasoning for time series classification through both a multi-turn reasoning and a fused decision-making strategy tailored to TSC. Rather than straightforwardly applying existing reasoning techniques or relying solely on LLMs' built-in reasoning capabilities, ReasonTSC first steers the model to think over the essential characteristics of time series data. Next, it integrates predictions and confidence scores from plug-in classifiers, e.g., domain-specific time series models, as in-context examples. Finally, ReasonTSC guides the LLM through a structured reasoning process: it evaluates the initial assessment, backtracks to consider alternative hypotheses, and compares their merits before arriving at a final classification. Extensive experiments and systematic ablation studies demonstrate that ReasonTSC consistently outperforms both existing time series reasoning baselines and plug-in models, and is even capable of identifying and correcting plug-in models' false predictions. 

**Abstract (ZH)**: 大型语言模型在时间序列分类中的推理能力：一种新型框架通过多轮推理和融合决策策略有效利用大型语言模型的推理能力 

---
# Predicting Empirical AI Research Outcomes with Language Models 

**Title (ZH)**: 使用语言模型预测 empirical AI 研究成果 

**Authors**: Jiaxin Wen, Chenglei Si, Yueh-han Chen, He He, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00794)  

**Abstract**: Many promising-looking ideas in AI research fail to deliver, but their validation takes substantial human labor and compute. Predicting an idea's chance of success is thus crucial for accelerating empirical AI research, a skill that even expert researchers can only acquire through substantial experience. We build the first benchmark for this task and compare LMs with human experts. Concretely, given two research ideas (e.g., two jailbreaking methods), we aim to predict which will perform better on a set of benchmarks. We scrape ideas and experimental results from conference papers, yielding 1,585 human-verified idea pairs published after our base model's cut-off date for testing, and 6,000 pairs for training. We then develop a system that combines a fine-tuned GPT-4.1 with a paper retrieval agent, and we recruit 25 human experts to compare with. In the NLP domain, our system beats human experts by a large margin (64.4% v.s. 48.9%). On the full test set, our system achieves 77% accuracy, while off-the-shelf frontier LMs like o3 perform no better than random guessing, even with the same retrieval augmentation. We verify that our system does not exploit superficial features like idea complexity through extensive human-written and LM-designed robustness tests. Finally, we evaluate our system on unpublished novel ideas, including ideas generated by an AI ideation agent. Our system achieves 63.6% accuracy, demonstrating its potential as a reward model for improving idea generation models. Altogether, our results outline a promising new direction for LMs to accelerate empirical AI research. 

**Abstract (ZH)**: AI研究中想法验证的第一个基准及其与人类专家的比较 

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
# HouseTS: A Large-Scale, Multimodal Spatiotemporal U.S. Housing Dataset 

**Title (ZH)**: HouseTS: 一个大规模多模态美国住房时空数据集 

**Authors**: Shengkun Wang, Yanshen Sun, Fanglan Chen, Linhan Wang, Naren Ramakrishnan, Chang-Tien Lu, Yinlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00765)  

**Abstract**: Accurate house-price forecasting is essential for investors, planners, and researchers. However, reproducible benchmarks with sufficient spatiotemporal depth and contextual richness for long horizon prediction remain scarce. To address this, we introduce HouseTS a large scale, multimodal dataset covering monthly house prices from March 2012 to December 2023 across 6,000 ZIP codes in 30 major U.S. metropolitan areas. The dataset includes over 890K records, enriched with points of Interest (POI), socioeconomic indicators, and detailed real estate metrics. To establish standardized performance baselines, we evaluate 14 models, spanning classical statistical approaches, deep neural networks (DNNs), and pretrained time-series foundation models. We further demonstrate the value of HouseTS in a multimodal case study, where a vision language model extracts structured textual descriptions of geographic change from time stamped satellite imagery. This enables interpretable, grounded insights into urban evolution. HouseTS is hosted on Kaggle, while all preprocessing pipelines, benchmark code, and documentation are openly maintained on GitHub to ensure full reproducibility and easy adoption. 

**Abstract (ZH)**: 准确的房价预测对于投资者、规划者和研究人员至关重要。然而，具有足够时空深度和丰富背景信息的可再现基准数据集，特别是适用于长期预测的数据集仍然稀缺。为此，我们引入了HouseTS大规模多模态数据集，该数据集涵盖了从2012年3月到2023年12月全美30个主要大都市区6000个ZIP编码区域的月度房价数据。数据集包含超过89万个记录，并附有兴趣点（POI）、社会经济指标和详细的房地产指标。为建立标准化的性能基准，我们评估了14种模型，涵盖传统的统计方法、深度神经网络（DNNs）以及预训练的时间序列基础模型。我们进一步通过多模态案例研究展示了HouseTS的价值，其中视觉语言模型从标注时间戳的卫星图像中提取结构化地理变化的文本描述，从而提供了可解释的城市演化的基础洞察。HouseTS在Kaggle上托管，所有预处理管道、基准代码和文档均在GitHub上公开维护，以确保完全的可再现性和易于采纳。 

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
# RiOSWorld: Benchmarking the Risk of Multimodal Compter-Use Agents 

**Title (ZH)**: RiOSWorld: 评估多模态计算机使用代理的风险 

**Authors**: Jingyi Yang, Shuai Shao, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00618)  

**Abstract**: With the rapid development of multimodal large language models (MLLMs), they are increasingly deployed as autonomous computer-use agents capable of accomplishing complex computer tasks. However, a pressing issue arises: Can the safety risk principles designed and aligned for general MLLMs in dialogue scenarios be effectively transferred to real-world computer-use scenarios? Existing research on evaluating the safety risks of MLLM-based computer-use agents suffers from several limitations: it either lacks realistic interactive environments, or narrowly focuses on one or a few specific risk types. These limitations ignore the complexity, variability, and diversity of real-world environments, thereby restricting comprehensive risk evaluation for computer-use agents. To this end, we introduce \textbf{RiOSWorld}, a benchmark designed to evaluate the potential risks of MLLM-based agents during real-world computer manipulations. Our benchmark includes 492 risky tasks spanning various computer applications, involving web, social media, multimedia, os, email, and office software. We categorize these risks into two major classes based on their risk source: (i) User-originated risks and (ii) Environmental risks. For the evaluation, we evaluate safety risks from two perspectives: (i) Risk goal intention and (ii) Risk goal completion. Extensive experiments with multimodal agents on \textbf{RiOSWorld} demonstrate that current computer-use agents confront significant safety risks in real-world scenarios. Our findings highlight the necessity and urgency of safety alignment for computer-use agents in real-world computer manipulation, providing valuable insights for developing trustworthy computer-use agents. Our benchmark is publicly available at this https URL. 

**Abstract (ZH)**: 随着多模态大语言模型（MLLMs）的快速发展，它们正越来越多地被部署为自主的计算机使用代理，能够完成复杂的计算机任务。然而，一个紧迫的问题出现了：为对话场景设计和对齐的安全风险原则能否有效地应用于现实世界的计算机使用场景中？现有的关于基于MLLM的计算机使用代理的安全风险评估研究存在一些局限性：要么缺乏现实的交互环境，要么仅仅集中在一两种特定的风险类型上。这些局限性忽略了现实环境中复杂性、多样性和变异性，从而限制了对计算机使用代理进行全面风险评估。为此，我们引入了\textbf{RiOSWorld}基准，旨在评估基于MLLM的代理在现实世界计算机操作中潜在的风险。我们的基准包括492项具有各种计算机应用风险的任务，涉及网络、社交媒体、多媒体、操作系统、电子邮件和办公软件。我们根据风险来源将这些风险分为两类：（i）用户引起的风险和（ii）环境风险。在评估中，我们从两个视角评估安全风险：（i）风险目标意图和（ii）风险目标完成。在\textbf{RiOSWorld}上进行的多模态代理广泛实验表明，当前的计算机使用代理在现实世界场景中面临显著的安全风险。我们的发现强调了在现实世界计算机操作中对计算机使用代理进行安全对齐的必要性和紧迫性，为开发可信的计算机使用代理提供了宝贵见解。我们的基准可在此\href{this https URL}{网址}获取。 

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
# A "Wenlu" Brain System for Multimodal Cognition and Embodied Decision-Making: A Secure New Architecture for Deep Integration of Foundation Models and Domain Knowledge 

**Title (ZH)**: “文溯”脑系统：多模态认知与 embodied 决策的新安全架构——基础模型与领域知识的深度集成 

**Authors**: Liang Geng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00570)  

**Abstract**: With the rapid penetration of artificial intelligence across industries and scenarios, a key challenge in building the next-generation intelligent core lies in effectively integrating the language understanding capabilities of foundation models with domain-specific knowledge bases in complex real-world applications. This paper proposes a multimodal cognition and embodied decision-making brain system, ``Wenlu", designed to enable secure fusion of private knowledge and public models, unified processing of multimodal data such as images and speech, and closed-loop decision-making from cognition to automatic generation of hardware-level code. The system introduces a brain-inspired memory tagging and replay mechanism, seamlessly integrating user-private data, industry-specific knowledge, and general-purpose language models. It provides precise and efficient multimodal services for enterprise decision support, medical analysis, autonomous driving, robotic control, and more. Compared with existing solutions, ``Wenlu" demonstrates significant advantages in multimodal processing, privacy security, end-to-end hardware control code generation, self-learning, and sustainable updates, thus laying a solid foundation for constructing the next-generation intelligent core. 

**Abstract (ZH)**: 随着人工智能在各行各业和各种场景中的快速渗透，构建下一代智能核心的关键挑战在于有效地将基础模型的语言理解能力与复杂现实应用中的领域特定知识库集成起来。“ Wenlu”是一种多模态认知和实体决策脑系统，旨在实现私人知识和公共模型的安全融合、多模态数据（如图像和语音）的一体化处理以及从认知到硬件级代码自动生成的闭环决策。该系统引入了一种受大脑启发的内存标记和回放机制，无缝集成用户私人数据、行业特定知识和通用语言模型。它为企业的决策支持、医疗分析、自动驾驶、机器人控制等领域提供精确高效的多模态服务。与现有解决方案相比，“ Wenlu”在多模态处理、隐私安全性、端到端硬件控制代码生成、自我学习和可持续更新方面展现出显著优势，从而为构建下一代智能核心奠定了坚实基础。 

---
# CityLens: Benchmarking Large Language-Vision Models for Urban Socioeconomic Sensing 

**Title (ZH)**: CityLens: 评估大型语言视觉模型的城市社会经济感知能力 

**Authors**: Tianhui Liu, Jie Feng, Hetian Pang, Xin Zhang, Tianjian Ouyang, Zhiyuan Zhang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00530)  

**Abstract**: Understanding urban socioeconomic conditions through visual data is a challenging yet essential task for sustainable urban development and policy planning. In this work, we introduce $\textbf{CityLens}$, a comprehensive benchmark designed to evaluate the capabilities of large language-vision models (LLVMs) in predicting socioeconomic indicators from satellite and street view imagery. We construct a multi-modal dataset covering a total of 17 globally distributed cities, spanning 6 key domains: economy, education, crime, transport, health, and environment, reflecting the multifaceted nature of urban life. Based on this dataset, we define 11 prediction tasks and utilize three evaluation paradigms: Direct Metric Prediction, Normalized Metric Estimation, and Feature-Based Regression. We benchmark 17 state-of-the-art LLVMs across these tasks. Our results reveal that while LLVMs demonstrate promising perceptual and reasoning capabilities, they still exhibit limitations in predicting urban socioeconomic indicators. CityLens provides a unified framework for diagnosing these limitations and guiding future efforts in using LLVMs to understand and predict urban socioeconomic patterns. Our codes and datasets are open-sourced via this https URL. 

**Abstract (ZH)**: 通过视觉数据理解城市社会经济条件是可持续城市发展和政策规划中一项艰巨但必要的任务。本文介绍了$\textbf{CityLens}$，一个全面的基准，旨在评估大型语言-视觉模型（LLVMs）从卫星和街道视图图像预测社会经济指标的能力。我们构建了一个多模态数据集，覆盖了全球17个城市，涵盖经济、教育、犯罪、交通、健康和环境等6个关键领域，反映了城市生活的复杂性。基于该数据集，我们定义了11项预测任务，并使用了三种评估范式：直接度量预测、标准化度量估计和基于特征的回归。我们在这些任务上对标了17个最先进的LLVMs。我们的结果显示，虽然LLVMs展示了具有前景的感知和推理能力，但在预测城市社会经济指标方面仍存在局限性。CityLens提供了一个统一的框架，用于诊断这些局限性，并指导未来使用LLVMs理解和预测城市社会经济模式的努力。我们的代码和数据集通过这个链接公开发布。 

---
# Monitoring Robustness and Individual Fairness 

**Title (ZH)**: 监测鲁棒性和个体公平性 

**Authors**: Ashutosh Gupta, Thomas A. Henzinger, Konstantin Kueffner, Kaushik Mallik, David Pape  

**Link**: [PDF](https://arxiv.org/pdf/2506.00496)  

**Abstract**: Input-output robustness appears in various different forms in the literature, such as robustness of AI models to adversarial or semantic perturbations and individual fairness of AI models that make decisions about humans.
We propose runtime monitoring of input-output robustness of deployed, black-box AI models, where the goal is to design monitors that would observe one long execution sequence of the model, and would raise an alarm whenever it is detected that two similar inputs from the past led to dissimilar outputs.
This way, monitoring will complement existing offline ``robustification'' approaches to increase the trustworthiness of AI decision-makers.
We show that the monitoring problem can be cast as the fixed-radius nearest neighbor (FRNN) search problem, which, despite being well-studied, lacks suitable online solutions.
We present our tool Clemont, which offers a number of lightweight monitors, some of which use upgraded online variants of existing FRNN algorithms, and one uses a novel algorithm based on binary decision diagrams -- a data-structure commonly used in software and hardware verification.
We have also developed an efficient parallelization technique that can substantially cut down the computation time of monitors for which the distance between input-output pairs is measured using the $L_\infty$ norm.
Using standard benchmarks from the literature of adversarial and semantic robustness and individual fairness, we perform a comparative study of different monitors in \tool, and demonstrate their effectiveness in correctly detecting robustness violations at runtime. 

**Abstract (ZH)**: 输入输出鲁棒性在文献中以多种形式出现，例如AI模型对抗或语义扰动的鲁棒性以及关于人类的AI模型的个体公平性。
我们提出对部署的黑盒AI模型的输入输出鲁棒性进行运行时监控，目标是设计能够监测模型长时间执行序列，并在检测到两个相似输入导致不同输出时发出警报的监控器。
这样，监控将补充现有的离线“强化鲁棒性”方法，提高AI决策者的可信度。
我们证明监控问题可以被表述为固定半径最近邻（FRNN）搜索问题，尽管这个问题已经被广泛研究，但仍缺乏合适的在线解决方案。
我们提出了一个名为Clemont的工具，该工具提供了一系列轻量级的监控器，其中一些监控器使用了现有FRNN算法的升级版本，还有一种监控器基于二叉决策图——这种数据结构常用于软件和硬件验证。
我们还开发了一种高效的并行化技术，可以显著减少使用$L_\infty$范数衡量输入输出对之间距离的监控器的计算时间。
通过使用对抗和语义鲁棒性及个体公平性领域的标准基准，我们在\tool中对不同监控器进行了比较研究，并展示了它们在运行时正确检测鲁棒性违规的有效性。 

---
# MIRROR: Cognitive Inner Monologue Between Conversational Turns for Persistent Reflection and Reasoning in Conversational LLMs 

**Title (ZH)**: MIRROR: 聊天型大语言模型中对话轮次间的心智内省对话以实现持续反思与推理 

**Authors**: Nicole Hsing  

**Link**: [PDF](https://arxiv.org/pdf/2506.00430)  

**Abstract**: Human intelligence relies on inner monologue to process complex information through simultaneous reflection, memory retrieval, and response formulation. We introduce MIRROR (Modular Internal Reasoning, Reflection, Orchestration, and Response), a cognitive architecture that systematically implements these parallel reasoning capabilities in large language models. MIRROR operates as a unified system with two distinct functional layers: the Thinker and the Talker. The Thinker encompasses: (1) the Inner Monologue Manager, coordinating reasoning threads across cognitive dimensions (Goals, Reasoning, and Memory); and (2) the Cognitive Controller, synthesizing these threads into a coherent internal narrative maintained across conversation turns. The Talker component then leverages this integrated narrative for context-aware responses. Evaluated on the CuRaTe benchmark--testing personalized dialogue with safety-critical constraints, conflicting preferences, and multi-turn consistency--LLMs utilizing the MIRROR architecture achieve up to 156% relative improvement in critical safety scenarios involving three persons with conflicting preferences, maintaining an average accuracy of ~>80% on all scenarios. Across scenario-specific comparisons, GPT-4o, Gemini 1.5 Pro, Claude 3.7 Sonnet, Llama 4 variants, and Mistral 3 variants with the MIRROR architecture outperformed baseline models by 21% on average (15.5 percentage points absolute). MIRROR directly addresses three critical LLM failure modes: sycophancy, attentional deficits to critical information, and inconsistent prioritization of conflicting constraints. This work bridges cognitive science and AI by implementing modular internal reasoning inspired by human cognition, creating a persistent internal model that significantly enhances multi-turn conversation capabilities. 

**Abstract (ZH)**: 人类智能依赖内部独白处理复杂信息，通过同时反思、记忆检索和响应形成。我们提出了MIRROR（模块化内部推理、反思、协调和响应）认知架构，系统性地在大型语言模型中实现这些并行推理能力。MIRROR作为统一系统运作，具有两个不同的功能层：思考者和说话者。思考者包含：（1）内部独白管理者，协调认知维度（目标、推理和记忆）间的推理线程；（2）认知控制器，将这些线程综合成连贯的内部叙述，并在会话轮次中保持连贯。说话者组件随后利用这一整合叙述进行具有上下文意识的响应。该架构在CuRaTe基准测试中进行了评估，测试包含安全关键约束、冲突偏好和多轮一致性的人性化对话，使用MIRROR架构的LLMs在涉及三人冲突偏好的关键安全场景中实现了多达156%的相对改进，平均准确率约为>80%。在特定场景比较中，配备MIRROR架构的GPT-4o、Gemini 1.5 Pro、Claude 3.7 Sonnet、Llama 4变体和Mistral 3变体模型在平均上比基准模型高出21%（绝对值15.5个百分点）。MIRROR直接解决了三种关键的LLM失效模式：拍马屁、对关键信息的注意力缺陷以及对冲突约束的一致优先级化问题。这项工作通过实现受人类认知启发的模块化内部推理，连接了认知科学和人工智能，创建了一个持久的内部模型，显著提升了多轮对话能力。 

---
# World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks 

**Title (ZH)**: 认知代理的 WORLD MODELS：转换未来网络中的边缘智能 

**Authors**: Changyuan Zhao, Ruichen Zhang, Jiacheng Wang, Gaosheng Zhao, Dusit Niyato, Geng Sun, Shiwen Mao, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.00417)  

**Abstract**: World models are emerging as a transformative paradigm in artificial intelligence, enabling agents to construct internal representations of their environments for predictive reasoning, planning, and decision-making. By learning latent dynamics, world models provide a sample-efficient framework that is especially valuable in data-constrained or safety-critical scenarios. In this paper, we present a comprehensive overview of world models, highlighting their architecture, training paradigms, and applications across prediction, generation, planning, and causal reasoning. We compare and distinguish world models from related concepts such as digital twins, the metaverse, and foundation models, clarifying their unique role as embedded cognitive engines for autonomous agents. We further propose Wireless Dreamer, a novel world model-based reinforcement learning framework tailored for wireless edge intelligence optimization, particularly in low-altitude wireless networks (LAWNs). Through a weather-aware UAV trajectory planning case study, we demonstrate the effectiveness of our framework in improving learning efficiency and decision quality. 

**Abstract (ZH)**: 世界模型已在人工智能领域 emerges 为一种变革性的范式，使代理能够构建其环境的内部表示以进行预测推理、规划和决策。通过学习潜在动力学，世界模型提供了一种样本高效的框架，特别是在数据受限或安全性关键的情景下特别有价值。在本文中，我们概述了世界模型的全面概况，强调其架构、训练范式及其在预测、生成、规划和因果推理中的应用。我们将世界模型与相关概念如数字孪生、元宇宙和基础模型进行比较和区分，阐明其作为嵌入式认知引擎的独特角色，为自主代理服务。我们进一步提出了基于世界模型的无线梦者（Wireless Dreamer）强化学习框架，该框架特别针对低空无线网络（LAWNs）的无线边缘智能优化。通过一个具备气象感知的无人机轨迹规划案例研究，我们展示了该框架在提高学习效率和决策质量方面的作用。 

---
# Position: Olfaction Standardization is Essential for the Advancement of Embodied Artificial Intelligence 

**Title (ZH)**: 位置：嗅觉标准化对于推进具身人工智能至关重要 

**Authors**: Kordel K. France, Rohith Peddi, Nik Dennler, Ovidiu Daescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00398)  

**Abstract**: Despite extraordinary progress in artificial intelligence (AI), modern systems remain incomplete representations of human cognition. Vision, audition, and language have received disproportionate attention due to well-defined benchmarks, standardized datasets, and consensus-driven scientific foundations. In contrast, olfaction - a high-bandwidth, evolutionarily critical sense - has been largely overlooked. This omission presents a foundational gap in the construction of truly embodied and ethically aligned super-human intelligence. We argue that the exclusion of olfactory perception from AI architectures is not due to irrelevance but to structural challenges: unresolved scientific theories of smell, heterogeneous sensor technologies, lack of standardized olfactory datasets, absence of AI-oriented benchmarks, and difficulty in evaluating sub-perceptual signal processing. These obstacles have hindered the development of machine olfaction despite its tight coupling with memory, emotion, and contextual reasoning in biological systems. In this position paper, we assert that meaningful progress toward general and embodied intelligence requires serious investment in olfactory research by the AI community. We call for cross-disciplinary collaboration - spanning neuroscience, robotics, machine learning, and ethics - to formalize olfactory benchmarks, develop multimodal datasets, and define the sensory capabilities necessary for machines to understand, navigate, and act within human environments. Recognizing olfaction as a core modality is essential not only for scientific completeness, but for building AI systems that are ethically grounded in the full scope of the human experience. 

**Abstract (ZH)**: 尽管在人工智能（AI）方面取得了非凡的进步，现代系统仍然是人类认知的不完整表现。由于有明确的基准、标准化的数据集和共识驱动的科学基础，视觉、听觉和语言受到了不成比例的关注。相比之下，作为演化上至关重要的感官之一的嗅觉，却遭到忽视。这种缺失在构建真正具身和伦理对齐的超人类智能时造成了基础性的缺口。我们认为，将嗅觉感知排除在AI架构之外并非由于无足轻重，而是由于结构上的挑战：未解决的嗅觉科学理论、异构传感器技术、缺乏标准化的嗅觉数据集、AI导向的基准缺失以及亚感知信号处理的评估难度。这些障碍阻碍了机器嗅觉的发展，尽管它在生物系统中与记忆、情绪和情境推理紧密耦合。在这篇立场论文中，我们主张，为了取得通识性和具身性智能的实质性进展，人工智能社区需要在嗅觉研究上进行认真投资。我们呼吁跨学科合作——从神经科学、机器人学、机器学习和伦理学等领域——以正式化嗅觉基准、开发多模态数据集，并定义机器理解、导航和在人类环境中行动所需的感官能力。认识嗅觉作为核心模态的重要性不仅对于科学的完整性至关重要，而且对于构建在人类完整体验范围内伦理基础稳固的AI系统也至关重要。 

---
# BASIL: Best-Action Symbolic Interpretable Learning for Evolving Compact RL Policies 

**Title (ZH)**: BASIL: 最佳动作符号可解释学习以演化紧凑的RL策略 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh, Mohammadali Keshtparvar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00328)  

**Abstract**: The quest for interpretable reinforcement learning is a grand challenge for the deployment of autonomous decision-making systems in safety-critical applications. Modern deep reinforcement learning approaches, while powerful, tend to produce opaque policies that compromise verification, reduce transparency, and impede human oversight. To address this, we introduce BASIL (Best-Action Symbolic Interpretable Learning), a systematic approach for generating symbolic, rule-based policies via online evolutionary search with quality-diversity (QD) optimization. BASIL represents policies as ordered lists of symbolic predicates over state variables, ensuring full interpretability and tractable policy complexity. By using a QD archive, the methodology in the proposed study encourages behavioral and structural diversity between top-performing solutions, while a complexity-aware fitness encourages the synthesis of compact representations. The evolutionary system supports the use of exact constraints for rule count and system adaptability for balancing transparency with expressiveness. Empirical comparisons with three benchmark tasks CartPole-v1, MountainCar-v0, and Acrobot-v1 show that BASIL consistently synthesizes interpretable controllers with compact representations comparable to deep reinforcement learning baselines. Herein, this article introduces a new interpretable policy synthesis method that combines symbolic expressiveness, evolutionary diversity, and online learning through a unifying framework. 

**Abstract (ZH)**: 可解释强化学习的探索是自主决策系统在关键安全应用中部署的一大挑战。现代深度强化学习方法虽然强大，但往往会生成不透明的策略，这妨碍了验证、透明度和人类监督。为应对这一挑战，我们引入了BASIL（最佳行为符号可解释学习），这是一种通过在线进化搜索和质量多样性（QD）优化生成符号规则基础策略的系统方法。BASIL通过符号谓词的有序列表表示策略，确保策略的最大可解释性和可处理的复杂性。利用QD存档，该方法鼓励高表现解决方案之间的行为和结构多样性，而复杂性感知的适应度则促进紧凑表示的合成。进化系统支持使用精确约束来控制规则数量，并通过平衡透明度和表现力来提高系统的适应性。与三个基准任务CartPole-v1、MountainCar-v0和Acrobot-v1的实证比较表明，BASIL能够一致地生成与深度强化学习基线具有可比紧凑表示的可解释控制器。本文介绍了一种新的可解释策略合成方法，该方法结合了符号表达能力、进化多样性和在线学习，通过统一框架实现。 

---
# Dyna-Think: Synergizing Reasoning, Acting, and World Model Simulation in AI Agents 

**Title (ZH)**: Dyna-Think: 结合推理、行动与世界模型模拟的AI代理方法 

**Authors**: Xiao Yu, Baolin Peng, Ruize Xu, Michel Galley, Hao Cheng, Suman Nath, Jianfeng Gao, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00320)  

**Abstract**: Recent progress in reasoning with large language models (LLMs), such as DeepSeek-R1, demonstrates impressive capabilities in domains like mathematics and coding, by exhibiting complex cognitive behaviors such as verification, goal decomposition, and self-reflection. However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks. In this work, we propose Dyna-Think, a thinking framework that integrates planning with an internal world model with reasoning and acting to enhance AI agent performance. To enable Dyna-Think, we propose Dyna-Think Imitation Learning (DIT) and Dyna-Think Dyna Training (DDT). To initialize a policy with Dyna-Think, DIT reconstructs the thinking process of R1 to focus on performing world model simulation relevant to the proposed (and planned) action, and trains the policy using this reconstructed data. To enhance Dyna-Think, DDT uses a two-stage training process to first improve the agent's world modeling ability via objectives such as state prediction or critique generation, and then improve the agent's action via policy training. We evaluate our methods on OSWorld, and demonstrate that Dyna-Think improves the agent's in-domain and out-of-domain performance, achieving similar best-of-n performance compared to R1 while generating 2x less tokens on average. Our extensive empirical studies reveal that 1) using critique generation for world model training is effective to improve policy performance; and 2) AI agents with better performance correlate with better world modeling abilities. We believe our results suggest a promising research direction to integrate world model simulation into AI agents to enhance their reasoning, planning, and acting capabilities. 

**Abstract (ZH)**: Recent progress in reasoning with large language models (LLMs) such as DeepSeek-R1 demonstrates impressive capabilities in domains like mathematics and coding, by exhibiting complex cognitive behaviors such as verification, goal decomposition, and self-reflection. However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks. In this work, we propose Dyna-Think, a thinking framework that integrates planning with an internal world model with reasoning and acting to enhance AI agent performance. To enable Dyna-Think, we propose Dyna-Think Imitation Learning (DIT) and Dyna-Think Dyna Training (DDT). To initialize a policy with Dyna-Think, DIT reconstructs the thinking process of R1 to focus on performing world model simulation relevant to the proposed (and planned) action, and trains the policy using this reconstructed data. To enhance Dyna-Think, DDT uses a two-stage training process to first improve the agent's world modeling ability via objectives such as state prediction or critique generation, and then improve the agent's action via policy training. We evaluate our methods on OSWorld, and demonstrate that Dyna-Think improves the agent's in-domain and out-of-domain performance, achieving similar best-of-n performance compared to R1 while generating 2x less tokens on average. Our extensive empirical studies reveal that 1) using critique generation for world model training is effective to improve policy performance; and 2) AI agents with better performance correlate with better world modeling abilities. We believe our results suggest a promising research direction to integrate world model simulation into AI agents to enhance their reasoning, planning, and acting capabilities. 

---
# Evaluation of LLMs for mathematical problem solving 

**Title (ZH)**: LLMs在数学问题求解中的评估 

**Authors**: Ruonan Wang, Runxi Wang, Yunwen Shen, Chengfeng Wu, Qinglin Zhou, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00309)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance on a range of educational tasks, but are still understudied for their potential to solve mathematical problems. In this study, we compare three prominent LLMs, including GPT-4o, DeepSeek-V3, and Gemini-2.0, on three mathematics datasets of varying complexities (GSM8K, MATH500, and UNSW datasets). We take a five-dimensional approach based on the Structured Chain-of-Thought (SCoT) framework to assess final answer correctness, step completeness, step validity, intermediate calculation accuracy, and problem comprehension. The results show that GPT-4o is the most stable and consistent in performance across all the datasets, but particularly it performs outstandingly in high-level questions of the UNSW dataset. DeepSeek-V3 is competitively strong in well-structured domains such as optimisation, but suffers from fluctuations in accuracy in statistical inference tasks. Gemini-2.0 shows strong linguistic understanding and clarity in well-structured problems but performs poorly in multi-step reasoning and symbolic logic. Our error analysis reveals particular deficits in each model: GPT-4o is at times lacking in sufficient explanation or precision; DeepSeek-V3 leaves out intermediate steps; and Gemini-2.0 is less flexible in mathematical reasoning in higher dimensions. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在教育任务上表现出色，但在解决数学问题方面仍待深入研究。本研究比较了GPT-4o、DeepSeek-V3和Gemini-2.0三种 prominent LLMs 在三个不同复杂度的数学数据集（GSM8K、MATH500 和 UNSW 数据集）上的性能。我们基于结构化的链式思维（SCoT）框架从五个维度评估最终答案的正确性、步骤完整性、步骤有效性、中间计算精度以及问题理解。结果显示，GPT-4o 在所有数据集上的表现最为稳定和一致，在 UNSW 数据集的高阶问题上表现尤为出色。DeepSeek-V3 在优化等结构化良好的领域表现出色，但在统计推断任务中准确性波动较大。Gemini-2.0 在结构良好问题上的语言理解能力和清晰度较强，但在多步推理和符号逻辑方面表现不佳。我们错误分析揭示了每个模型的具体缺陷：GPT-4o 有时缺乏充分的解释或精确度；DeepSeek-V3 忽略了中间步骤；而 Gemini-2.0 在高维度的数学推理上不够灵活。 

---
# Sleep Brain and Cardiac Activity Predict Cognitive Flexibility and Conceptual Reasoning Using Deep Learning 

**Title (ZH)**: 睡眠脑活动和心脏活动预测认知灵活性和概念推理：基于深度学习的方法 

**Authors**: Boshra Khajehpiri, Eric Granger, Massimiliano de Zambotti, Fiona C. Baker, Mohamad Forouzanfar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00279)  

**Abstract**: Despite extensive research on the relationship between sleep and cognition, the connection between sleep microstructure and human performance across specific cognitive domains remains underexplored. This study investigates whether deep learning models can predict executive functions, particularly cognitive adaptability and conceptual reasoning from physiological processes during a night's sleep. To address this, we introduce CogPSGFormer, a multi-scale convolutional-transformer model designed to process multi-modal polysomnographic data. This model integrates one-channel ECG and EEG signals along with extracted features, including EEG power bands and heart rate variability parameters, to capture complementary information across modalities. A thorough evaluation of the CogPSGFormer architecture was conducted to optimize the processing of extended sleep signals and identify the most effective configuration. The proposed framework was evaluated on 817 individuals from the STAGES dataset using cross-validation. The model achieved 80.3\% accuracy in classifying individuals into low vs. high cognitive performance groups on unseen data based on Penn Conditional Exclusion Test (PCET) scores. These findings highlight the effectiveness of our multi-scale feature extraction and multi-modal learning approach in leveraging sleep-derived signals for cognitive performance prediction. To facilitate reproducibility, our code is publicly accessible (this https URL). 

**Abstract (ZH)**: 尽管对睡眠与认知之间的关系进行了广泛研究，但睡眠微结构与人类特定认知领域的表现之间的联系仍鲜有探索。本研究旨在探究深度学习模型是否能够预测执行功能，特别是在生理过程夜间睡眠期间的认知适应性和概念推理。为此，我们引入了CogPSGFormer，这是一种多尺度卷积转换器模型，设计用于处理多模态多导睡眠图数据。该模型整合了一通道ECG和EEG信号以及提取特征，包括EEG功率带和心率变异性参数，以捕捉各模态之间的互补信息。我们对CogPSGFormer架构进行了全面评估，以优化扩展睡眠信号的处理并确定最有效配置。该提出的框架在STAGES数据集的817名个体上进行了交叉验证评估。模型在未见数据上基于Penn条件排除测试（PCET）得分成功地将个体分类为低认知表现组和高认知表现组，准确率为80.3%。这些发现强调了我们多尺度特征提取和多模态学习方法在利用睡眠衍生信号预测认知表现的有效性。为了便于再现性，我们的代码已公开可访问。 

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
# SMELLNET: A Large-scale Dataset for Real-world Smell Recognition 

**Title (ZH)**: SMELLNET: 一种大规模气味识别数据集 

**Authors**: Dewei Feng, Carol Li, Wei Dai, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00239)  

**Abstract**: The ability of AI to sense and identify various substances based on their smell alone can have profound impacts on allergen detection (e.g., smelling gluten or peanuts in a cake), monitoring the manufacturing process, and sensing hormones that indicate emotional states, stress levels, and diseases. Despite these broad impacts, there are virtually no large scale benchmarks, and therefore little progress, for training and evaluating AI systems' ability to smell in the real world. In this paper, we use portable gas and chemical sensors to create SmellNet, the first large-scale database that digitizes a diverse range of smells in the natural world. SmellNet contains about 180,000 time steps of 50 substances (spanning nuts, spices, herbs, fruits, and vegetables) with 50 hours of data. Using SmellNet, we train AI models for real-time classification of substances based on their smell alone. Our best methods leverage sequence models, contrastive learning to integrate high-resolution Gas Chromatography-Mass Spectrometry molecular data, and a new temporal difference method that identifies sharp changes in sensor readings. Our best models achieve up to 65.35% accuracy on pre-recorded data, and generalize to real-world conditions with 10.71% accuracy on nuts and 25.38% on spices in the challenging 50-way online classification task. Despite these promising results, SmellNet highlights many technical challenges in building AI for smell, including richer feature learning, on-edge smell models, and robustness to environmental changes. 

**Abstract (ZH)**: 基于气味识别的AI能力在过敏原检测、制造过程监控及情绪状态、压力水平和疾病感应方面的潜在影响：SmellNet——首个大规模自然气味数据库及其在实时物质分类中的应用 

---
# Ethical AI: Towards Defining a Collective Evaluation Framework 

**Title (ZH)**: 伦理人工智能：向构建集体评估框架迈进 

**Authors**: Aasish Kumar Sharma, Dimitar Kyosev, Julian Kunkel  

**Link**: [PDF](https://arxiv.org/pdf/2506.00233)  

**Abstract**: Artificial Intelligence (AI) is transforming sectors such as healthcare, finance, and autonomous systems, offering powerful tools for innovation. Yet its rapid integration raises urgent ethical concerns related to data ownership, privacy, and systemic bias. Issues like opaque decision-making, misleading outputs, and unfair treatment in high-stakes domains underscore the need for transparent and accountable AI systems. This article addresses these challenges by proposing a modular ethical assessment framework built on ontological blocks of meaning-discrete, interpretable units that encode ethical principles such as fairness, accountability, and ownership. By integrating these blocks with FAIR (Findable, Accessible, Interoperable, Reusable) principles, the framework supports scalable, transparent, and legally aligned ethical evaluations, including compliance with the EU AI Act. Using a real-world use case in AI-powered investor profiling, the paper demonstrates how the framework enables dynamic, behavior-informed risk classification. The findings suggest that ontological blocks offer a promising path toward explainable and auditable AI ethics, though challenges remain in automation and probabilistic reasoning. 

**Abstract (ZH)**: 人工智能（AI）正在改造医疗、金融和自主系统等领域，提供了强大的创新工具。然而，其迅速融入引发了关于数据所有权、隐私和系统偏见的紧迫伦理问题。在高风险领域中的不透明决策、误导性输出和不公平待遇强调了透明和问责制AI系统的必要性。本文通过提出一个基于意义离散的解释性单元构建的模块化伦理评估框架，解决了这些问题。该框架通过将这些单元与FAIR（可发现的、可访问的、可互操作的、可重用的）原则整合，支持可扩展、透明和法律合规的伦理评估，包括符合欧盟AI法案。通过一个基于AI的投资人画像实际案例，论文展示了该框架如何实现动态、基于行为的风险分类。研究结果表明，意义离散单元为可解释和可审计的AI伦理提供了有希望的途径，尽管在自动化和概率推理方面仍面临挑战。 

---
# What do professional software developers need to know to succeed in an age of Artificial Intelligence? 

**Title (ZH)**: 专业软件开发者在人工智能时代需要掌握哪些知识以取得成功？ 

**Authors**: Matthew Kam, Cody Miller, Miaoxin Wang, Abey Tidwell, Irene A. Lee, Joyce Malyn-Smith, Beatriz Perez, Vikram Tiwari, Joshua Kenitzer, Andrew Macvean, Erin Barrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00202)  

**Abstract**: Generative AI is showing early evidence of productivity gains for software developers, but concerns persist regarding workforce disruption and deskilling. We describe our research with 21 developers at the cutting edge of using AI, summarizing 12 of their work goals we uncovered, together with 75 associated tasks and the skills & knowledge for each, illustrating how developers use AI at work. From all of these, we distilled our findings in the form of 5 insights. We found that the skills & knowledge to be a successful AI-enhanced developer are organized into four domains (using Generative AI effectively, core software engineering, adjacent engineering, and adjacent non-engineering) deployed at critical junctures throughout a 6-step task workflow. In order to "future proof" developers for this age of AI, on-the-job learning initiatives and computer science degree programs will need to target both "soft" skills and the technical skills & knowledge in all four domains to reskill, upskill and safeguard against deskilling. 

**Abstract (ZH)**: 生成式AI为软件开发者带来了初步的生产力提升，但对劳动力市场冲击和技能退化仍存在担忧。我们描述了21名处于AI应用前沿的开发者的研究情况，总结了他们发现的12个工作目标及其相关75个任务和所需技能与知识，展示了开发者在工作中的AI应用方式。从这些发现中，我们提炼出5个洞察。我们发现，成功的AI增强型开发者的技能和知识围绕四个领域（有效使用生成式AI、核心软件工程、相邻工程和相邻非工程领域）组织，并在6步任务工作流的关键节点上体现。为了使开发者为AI时代做好准备，在职培训计划和计算机科学学位项目需要同时关注“软”技能和技术技能与知识在所有四个领域的提升，以实现再教育、提高技能并防范技能退化。 

---
# Control-R: Towards controllable test-time scaling 

**Title (ZH)**: Control-R: 向可控测试时缩放迈进 

**Authors**: Di Zhang, Weida Wang, Junxian Li, Xunzhi Wang, Jiatong Li, Jianbo Wu, Jingdi Lei, Haonan He, Peng Ye, Shufei Zhang, Wanli Ouyang, Yuqiang Li, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00189)  

**Abstract**: This paper target in addressing the challenges of underthinking and overthinking in long chain-of-thought (CoT) reasoning for Large Reasoning Models (LRMs) by introducing Reasoning Control Fields (RCF)--a novel test-time approach that injects structured control signals to guide reasoning from a tree search perspective. RCF enables models to adjust reasoning effort according to given control conditions when solving complex tasks. Additionally, we present the Control-R-4K dataset, which consists of challenging problems annotated with detailed reasoning processes and corresponding control fields. To further enhance reasoning control, we propose a Conditional Distillation Finetuning (CDF) method, which trains model--particularly Control-R-32B--to effectively adjust reasoning effort during test time. Experimental results on benchmarks such as AIME2024 and MATH500 demonstrate that our approach achieves state-of-the-art performance at the 32B scale while enabling a controllable Long CoT reasoning process (L-CoT). Overall, this work introduces an effective paradigm for controllable test-time scaling reasoning. 

**Abstract (ZH)**: 本文提出了一种通过引入推理控制域（RCF）来解决大型推理模型（LRMs）在长链推理（CoT）中过度推理和欠推理挑战的新颖测试时方法，RCF从树搜索的角度注入结构化的控制信号以指导推理。此外，我们介绍了Control-R-4K数据集，该数据集包含详细标注的推理过程和相应的控制域。为进一步增强推理控制，我们提出了条件蒸馏微调（CDF）方法，该方法训练模型（特别是Control-R-32B）在测试时有效调整推理努力。在AIME2024和MATH500等基准上的实验结果表明，我们的方法在32B规模上达到了最先进的性能，同时实现了可控的长链推理过程（L-CoT）。总体而言，本文引入了一种有效的可控制测试时扩展推理范式。 

---
# Tournament of Prompts: Evolving LLM Instructions Through Structured Debates and Elo Ratings 

**Title (ZH)**: 指令锦标赛：通过结构化辩论和Elo评分演化LLM指令 

**Authors**: Anirudh Nair, Adi Banerjee, Laurent Mombaerts, Matthew Hagen, Tarik Borogovac  

**Link**: [PDF](https://arxiv.org/pdf/2506.00178)  

**Abstract**: Prompt engineering represents a critical bottleneck to harness the full potential of Large Language Models (LLMs) for solving complex tasks, as it requires specialized expertise, significant trial-and-error, and manual intervention. This challenge is particularly pronounced for tasks involving subjective quality assessment, where defining explicit optimization objectives becomes fundamentally problematic. Existing automated prompt optimization methods falter in these scenarios, as they typically require well-defined task-specific numerical fitness functions or rely on generic templates that cannot capture the nuanced requirements of complex use cases. We introduce DEEVO (DEbate-driven EVOlutionary prompt optimization), a novel framework that guides prompt evolution through a debate-driven evaluation with an Elo-based selection. Contrary to prior work, DEEVOs approach enables exploration of the discrete prompt space while preserving semantic coherence through intelligent crossover and strategic mutation operations that incorporate debate-based feedback, combining elements from both successful and unsuccessful prompts based on identified strengths rather than arbitrary splicing. Using Elo ratings as a fitness proxy, DEEVO simultaneously drives improvement and preserves valuable diversity in the prompt population. Experimental results demonstrate that DEEVO significantly outperforms both manual prompt engineering and alternative state-of-the-art optimization approaches on open-ended tasks and close-ended tasks despite using no ground truth feedback. By connecting LLMs reasoning capabilities with adaptive optimization, DEEVO represents a significant advancement in prompt optimization research by eliminating the need of predetermined metrics to continuously improve AI systems. 

**Abstract (ZH)**: DEEVO：基于辩论驱动的进化式提示优化 

---
# Utilizing AI for Aviation Post-Accident Analysis Classification 

**Title (ZH)**: 利用AI进行航空事故后分析分类 

**Authors**: Aziida Nanyonga, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2506.00169)  

**Abstract**: The volume of textual data available in aviation safety reports presents a challenge for timely and accurate analysis. This paper examines how Artificial Intelligence (AI) and, specifically, Natural Language Processing (NLP) can automate the process of extracting valuable insights from this data, ultimately enhancing aviation safety. The paper reviews ongoing efforts focused on the application of NLP and deep learning to aviation safety reports, with the goal of classifying the level of damage to an aircraft and identifying the phase of flight during which safety occurrences happen. Additionally, the paper explores the use of Topic Modeling (TM) to uncover latent thematic structures within aviation incident reports, aiming to identify recurring patterns and potential areas for safety improvement. The paper compares and contrasts the performance of various deep learning models and TM techniques applied to datasets from the National Transportation Safety Board (NTSB) and the Australian Transport Safety Bureau (ATSB), as well as the Aviation Safety Network (ASN), discussing the impact of dataset size and source on the accuracy of the analysis. The findings demonstrate that both NLP and deep learning, as well as TM, can significantly improve the efficiency and accuracy of aviation safety analysis, paving the way for more proactive safety management and risk mitigation strategies. 

**Abstract (ZH)**: 可用的航空安全报告中的文本数据量为及时准确分析带来了挑战。本文探讨了人工智能（AI）和具体来说是自然语言处理（NLP）如何自动化从这些数据中提取有价值洞察的过程，最终提升航空安全。本文回顾了将NLP和深度学习应用于航空安全报告的现有努力，旨在分类航空器损伤程度并识别安全事件发生于飞行的哪个阶段。此外，本文探索了主题建模（TM）在揭示航空事故报告中潜在主题结构方面的应用，旨在识别重复模式和潜在的安全改进领域。本文比较了各种深度学习模型和TM技术在国家运输安全委员会（NTSB）、澳大利亚运输安全局（ATSB）以及航空安全网（ASN）数据集上的性能，讨论了数据集大小和来源对分析准确性的影响。研究结果表明，NLP、深度学习以及TM都能显著提高航空安全分析的效率和准确性，为更积极的安全管理及风险缓解策略铺平道路。 

---
# Balancing Profit and Fairness in Risk-Based Pricing Markets 

**Title (ZH)**: 基于风险的价格市场中收益与公平性的平衡 

**Authors**: Jesse Thibodeau, Hadi Nekoei, Afaf Taïk, Janarthanan Rajendran, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00140)  

**Abstract**: Dynamic, risk-based pricing can systematically exclude vulnerable consumer groups from essential resources such as health insurance and consumer credit. We show that a regulator can realign private incentives with social objectives through a learned, interpretable tax schedule. First, we provide a formal proposition that bounding each firm's \emph{local} demographic gap implicitly bounds the \emph{global} opt-out disparity, motivating firm-level penalties. Building on this insight we introduce \texttt{MarketSim} -- an open-source, scalable simulator of heterogeneous consumers and profit-maximizing firms -- and train a reinforcement learning (RL) social planner (SP) that selects a bracketed fairness-tax while remaining close to a simple linear prior via an $\mathcal{L}_1$ regularizer. The learned policy is thus both transparent and easily interpretable. In two empirically calibrated markets, i.e., U.S. health-insurance and consumer-credit, our planner simultaneously raises demand-fairness by up to $16\%$ relative to unregulated Free Market while outperforming a fixed linear schedule in terms of social welfare without explicit coordination. These results illustrate how AI-assisted regulation can convert a competitive social dilemma into a win-win equilibrium, providing a principled and practical framework for fairness-aware market oversight. 

**Abstract (ZH)**: 动态风险基价格可能系统性地将脆弱消费者群体排除在必要资源如健康保险和消费者信贷之外。我们展示了一名监管者可以通过学习和可解释的税率重新对齐私人激励与社会目标。首先，我们提供了正式命题，即限制每家公司的局部人口差距隐含地限制了全局退出不平等，从而激励公司层面的处罚。在此洞察基础上，我们引入了MarketSim——一个开源可扩展的异质消费者和利润最大化公司仿真器，并通过$\mathcal{L}_1$正则化器训练了一个强化学习社会规划者（SP），该规划者选择一个边界公平税，同时接近一个简单的线性先验。因此，学习到的策略是透明且易于解释的。在两个经验校准的市场，即美国健康保险和消费者信贷市场，我们的规划者同时将需求公平性提高最多16%，相对于未受监管的自由市场，同时在无需显式协调的情况下超越固定线性计划在社会福利方面表现更优。这些结果表明，AI辅助监管如何将竞争性的社会困境转化为双赢均衡，并提供了一个公平感知市场监督的原则性和实用性框架。 

---
# The Automated but Risky Game: Modeling Agent-to-Agent Negotiations and Transactions in Consumer Markets 

**Title (ZH)**: 自动但充满风险的游戏：消费者市场中代理方到代理方的谈判与交易建模 

**Authors**: Shenzhe Zhu, Jiao Sun, Yi Nian, Tobin South, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00073)  

**Abstract**: AI agents are increasingly used in consumer-facing applications to assist with tasks such as product search, negotiation, and transaction execution. In this paper, we explore a future scenario where both consumers and merchants authorize AI agents to fully automate negotiations and transactions. We aim to answer two key questions: (1) Do different LLM agents vary in their ability to secure favorable deals for users? (2) What risks arise from fully automating deal-making with AI agents in consumer markets? To address these questions, we develop an experimental framework that evaluates the performance of various LLM agents in real-world negotiation and transaction settings. Our findings reveal that AI-mediated deal-making is an inherently imbalanced game -- different agents achieve significantly different outcomes for their users. Moreover, behavioral anomalies in LLMs can result in financial losses for both consumers and merchants, such as overspending or accepting unreasonable deals. These results underscore that while automation can improve efficiency, it also introduces substantial risks. Users should exercise caution when delegating business decisions to AI agents. 

**Abstract (ZH)**: AI代理在消费者市场中自动化谈判与交易的风险与机遇：基于不同大规模语言模型的实证研究 

---
# Toward Knowledge-Guided AI for Inverse Design in Manufacturing: A Perspective on Domain, Physics, and Human-AI Synergy 

**Title (ZH)**: 面向制造业逆向设计的知识引导AI：关于领域、物理和人机协同的视角 

**Authors**: Hugon Lee, Hyeonbin Moon, Junhyeong Lee, Seunghwa RYu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00056)  

**Abstract**: Artificial intelligence (AI) is reshaping inverse design across manufacturing domain, enabling high-performance discovery in materials, products, and processes. However, purely data-driven approaches often struggle in realistic settings characterized by sparse data, high-dimensional design spaces, and nontrivial physical constraints. This perspective argues for a new generation of design systems that transcend black-box modeling by integrating domain knowledge, physics-informed learning, and intuitive human-AI interfaces. We first demonstrate how expert-guided sampling strategies enhance data efficiency and model generalization. Next, we discuss how physics-informed machine learning enables physically consistent modeling in data-scarce regimes. Finally, we explore how large language models emerge as interactive design agents connecting user intent with simulation tools, optimization pipelines, and collaborative workflows. Through illustrative examples and conceptual frameworks, we advocate that inverse design in manufacturing should evolve into a unified ecosystem, where domain knowledge, physical priors, and adaptive reasoning collectively enable scalable, interpretable, and accessible AI-driven design systems. 

**Abstract (ZH)**: 人工智能（AI）正在重塑制造领域的逆向设计，使其能够在材料、产品和工艺中实现高性能发现。然而，在稀疏数据、高维设计空间和非平凡物理约束等现实场景中，纯粹的数据驱动方法往往难以应对。本文倡导一种超越黑盒建模的新一代设计系统，通过集成领域知识、物理知情学习以及直观的人机交互界面来促进设计。我们首先展示了专家指导的采样策略如何提升数据效率和模型泛化能力。接着，我们讨论了物理知情机器学习如何在数据稀缺的情况下实现物理一致性建模。最后，我们探讨了大型语言模型如何作为交互式设计代理，连接用户意图、仿真工具、优化管道和协作工作流程。通过示例和概念框架，我们主张逆向设计应进化为一个集成的生态系统，其中领域知识、物理先验和适应性推理共同推动可扩展、可解释和可访问的AI驱动设计系统的发展。 

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
# Feel the Force: Contact-Driven Learning from Humans 

**Title (ZH)**: 感受力量：由接触驱动的人机学习 

**Authors**: Ademi Adeniji, Zhuoran Chen, Vincent Liu, Venkatesh Pattabiraman, Raunaq Bhirangi, Siddhant Haldar, Pieter Abbeel, Lerrel Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.01944)  

**Abstract**: Controlling fine-grained forces during manipulation remains a core challenge in robotics. While robot policies learned from robot-collected data or simulation show promise, they struggle to generalize across the diverse range of real-world interactions. Learning directly from humans offers a scalable solution, enabling demonstrators to perform skills in their natural embodiment and in everyday environments. However, visual demonstrations alone lack the information needed to infer precise contact forces. We present FeelTheForce (FTF): a robot learning system that models human tactile behavior to learn force-sensitive manipulation. Using a tactile glove to measure contact forces and a vision-based model to estimate hand pose, we train a closed-loop policy that continuously predicts the forces needed for manipulation. This policy is re-targeted to a Franka Panda robot with tactile gripper sensors using shared visual and action representations. At execution, a PD controller modulates gripper closure to track predicted forces-enabling precise, force-aware control. Our approach grounds robust low-level force control in scalable human supervision, achieving a 77% success rate across 5 force-sensitive manipulation tasks. Code and videos are available at this https URL. 

**Abstract (ZH)**: 操纵过程中精细力量的控制仍然是机器人技术中的核心挑战。通过人类直接学习提供的解决方案可扩展，使演示者能够在自然身体形态和日常环境中执行技能。然而，仅通过视觉演示无法提供推断精确接触力所需的信息。我们提出FeelTheForce (FTF)：一种机器人学习系统，用于模仿人类触觉行为以学习力敏感操纵。借助触觉手套测量接触力，并使用基于视觉的模型估计手部姿态，我们训练了一个闭环策略，该策略能够持续预测操纵所需的力。该策略经重新调整以适用于配备触觉 gripper 传感器的 Franka Panda 机器人，采用共享的视觉和动作表征。在执行过程中，PD 控制器调节 gripper 的闭合以追踪预测的力，实现精确的力感知控制。我们的方法将鲁棒的低级力控制与可扩展的人类监督相结合，在 5 项力敏感操纵任务中实现了 77% 的成功率。更多信息请参见此链接。 

---
# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning 

**Title (ZH)**: 超出80/20规则：高熵少数标记驱动的有效强化学习用于LLM推理 

**Authors**: Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01939)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）通过token熵模式提升大型语言模型的推理能力及其机制探究 

---
# Red Teaming AI Policy: A Taxonomy of Avoision and the EU AI Act 

**Title (ZH)**: 红队评估AI政策：规避分类与欧盟AI法案 

**Authors**: Rui-Jie Yew, Bill Marino, Suresh Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.01931)  

**Abstract**: The shape of AI regulation is beginning to emerge, most prominently through the EU AI Act (the "AIA"). By 2027, the AIA will be in full effect, and firms are starting to adjust their behavior in light of this new law. In this paper, we present a framework and taxonomy for reasoning about "avoision" -- conduct that walks the line between legal avoidance and evasion -- that firms might engage in so as to minimize the regulatory burden the AIA poses. We organize these avoision strategies around three "tiers" of increasing AIA exposure that regulated entities face depending on: whether their activities are (1) within scope of the AIA, (2) exempted from provisions of the AIA, or are (3) placed in a category with higher regulatory scrutiny. In each of these tiers and for each strategy, we specify the organizational and technological forms through which avoision may manifest. Our goal is to provide an adversarial framework for "red teaming" the AIA and AI regulation on the horizon. 

**Abstract (ZH)**: AI法规的形态正逐渐成型：以欧盟AI法案（“AIA”）最为显著。到2027年，AIA将全面生效，企业已经开始调整行为以应对这一新法规。本文提出了一种框架和分类体系，用于分析企业可能会采取的“规避”行为——这类行为在合法规避与规避之间划界——以尽量减少AIA所带来的监管负担。我们将这些规避策略按企业面临的AIA监管暴露程度分为三个“层级”：其活动是否（1）在AIA监管范围内，（2）免于AIA的相关规定，或（3）处于受更高监管监督的类别。在每个层级和每个策略中，我们详细说明了规避可能通过的组织和技术形式。我们的目标是提供一种对手框架，用于对AIA和即将出台的AI法规进行“红队”测试。 

---
# Image Generation from Contextually-Contradictory Prompts 

**Title (ZH)**: 从上下文矛盾提示生成图像 

**Authors**: Saar Huberman, Or Patashnik, Omer Dahary, Ron Mokady, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2506.01929)  

**Abstract**: Text-to-image diffusion models excel at generating high-quality, diverse images from natural language prompts. However, they often fail to produce semantically accurate results when the prompt contains concept combinations that contradict their learned priors. We define this failure mode as contextual contradiction, where one concept implicitly negates another due to entangled associations learned during training. To address this, we propose a stage-aware prompt decomposition framework that guides the denoising process using a sequence of proxy prompts. Each proxy prompt is constructed to match the semantic content expected to emerge at a specific stage of denoising, while ensuring contextual coherence. To construct these proxy prompts, we leverage a large language model (LLM) to analyze the target prompt, identify contradictions, and generate alternative expressions that preserve the original intent while resolving contextual conflicts. By aligning prompt information with the denoising progression, our method enables fine-grained semantic control and accurate image generation in the presence of contextual contradictions. Experiments across a variety of challenging prompts show substantial improvements in alignment to the textual prompt. 

**Abstract (ZH)**: 基于文本到图像的扩散模型在从自然语言提示生成高质量、多样化图像方面表现出色。然而，当提示包含与模型学习先验矛盾的概念组合时，它们往往无法产生语义准确的结果。我们定义这种失败模式为上下文矛盾，其中一个概念由于训练期间学习到的纠缠关联而隐含地否定了另一个概念。为了解决这一问题，我们提出了一种阶段感知的提示分解框架，该框架使用一系列代理提示引导去噪过程。每个代理提示都构建为与特定去噪阶段预期生成的语义内容匹配，并确保上下文连贯性。为构建这些代理提示，我们利用大型语言模型（LLM）分析目标提示，识别矛盾，并生成能够保留原始意图同时解决上下文冲突的替代表达。通过将提示信息与去噪进程对齐，我们的方法在存在上下文矛盾的情况下能够实现细粒度的语义控制和准确的图像生成。在多种具有挑战性的提示下进行的实验表明，我们的方法在与文本提示对齐方面取得了显著改进。 

---
# Online Competitive Information Gathering for Partially Observable Trajectory Games 

**Title (ZH)**: 部分可观测轨迹博弈中的在线竞速信息收集 

**Authors**: Mel Krusniak, Hang Xu, Parker Palermo, Forrest Laine  

**Link**: [PDF](https://arxiv.org/pdf/2506.01927)  

**Abstract**: Game-theoretic agents must make plans that optimally gather information about their opponents. These problems are modeled by partially observable stochastic games (POSGs), but planning in fully continuous POSGs is intractable without heavy offline computation or assumptions on the order of belief maintained by each player. We formulate a finite history/horizon refinement of POSGs which admits competitive information gathering behavior in trajectory space, and through a series of approximations, we present an online method for computing rational trajectory plans in these games which leverages particle-based estimations of the joint state space and performs stochastic gradient play. We also provide the necessary adjustments required to deploy this method on individual agents. The method is tested in continuous pursuit-evasion and warehouse-pickup scenarios (alongside extensions to $N > 2$ players and to more complex environments with visual and physical obstacles), demonstrating evidence of active information gathering and outperforming passive competitors. 

**Abstract (ZH)**: 基于博弈的代理必须制定最优地收集关于对手信息的计划。这些问题通过部分可观测随机博弈（POSGs）建模，但在连续的POSGs中进行完全在线规划是不可行的，除非对每个玩家保持的信念顺序做出假设。我们提出了POSGs的一个有限历史/时限细化，它在轨迹空间中允许竞争性的信息收集行为，并通过一系列近似，我们给出了一个利用基于粒子的状态空间联合估计并在随机梯度播放中计算这些博弈中的理性轨迹计划的方法。我们也提供了在单个代理上部署此方法所需的必要调整。该方法在连续的追逐-逃避和仓库取件场景（以及扩展到超过两个玩家和更复杂环境中）中进行了测试，展示了积极的信息收集行为，并且优于被动的竞争对手。 

---
# TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation 

**Title (ZH)**: TaxaDiffusion：渐进训练的细粒度物种生成扩散模型 

**Authors**: Amin Karimi Monsefi, Mridul Khurana, Rajiv Ramnath, Anuj Karpatne, Wei-Lun Chao, Cheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01923)  

**Abstract**: We propose TaxaDiffusion, a taxonomy-informed training framework for diffusion models to generate fine-grained animal images with high morphological and identity accuracy. Unlike standard approaches that treat each species as an independent category, TaxaDiffusion incorporates domain knowledge that many species exhibit strong visual similarities, with distinctions often residing in subtle variations of shape, pattern, and color. To exploit these relationships, TaxaDiffusion progressively trains conditioned diffusion models across different taxonomic levels -- starting from broad classifications such as Class and Order, refining through Family and Genus, and ultimately distinguishing at the Species level. This hierarchical learning strategy first captures coarse-grained morphological traits shared by species with common ancestors, facilitating knowledge transfer before refining fine-grained differences for species-level distinction. As a result, TaxaDiffusion enables accurate generation even with limited training samples per species. Extensive experiments on three fine-grained animal datasets demonstrate that outperforms existing approaches, achieving superior fidelity in fine-grained animal image generation. Project page: this https URL 

**Abstract (ZH)**: TaxaDiffusion：一种基于分类学指导的扩散模型训练框架，用于生成高形态和身份准确度的精细粒度动物图像 

---
# MedEBench: Revisiting Text-instructed Image Editing 

**Title (ZH)**: MedEBench: 重新审视基于文本指导的图像编辑 

**Authors**: Minghao Liu, Zhitao He, Zhiyuan Fan, Qingyun Wang, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2506.01921)  

**Abstract**: Text-guided image editing has seen rapid progress in natural image domains, but its adaptation to medical imaging remains limited and lacks standardized evaluation. Clinically, such editing holds promise for simulating surgical outcomes, creating personalized teaching materials, and enhancing patient communication. To bridge this gap, we introduce \textbf{MedEBench}, a comprehensive benchmark for evaluating text-guided medical image editing. It consists of 1,182 clinically sourced image-prompt triplets spanning 70 tasks across 13 anatomical regions. MedEBench offers three key contributions: (1) a clinically relevant evaluation framework covering Editing Accuracy, Contextual Preservation, and Visual Quality, supported by detailed descriptions of expected change and ROI (Region of Interest) masks; (2) a systematic comparison of seven state-of-the-art models, revealing common failure patterns; and (3) a failure analysis protocol based on attention grounding, using IoU between attention maps and ROIs to identify mislocalization. MedEBench provides a solid foundation for developing and evaluating reliable, clinically meaningful medical image editing systems. 

**Abstract (ZH)**: 基于文本的医学图像编辑基准：MedEBench 

---
# Transformers as Multi-task Learners: Decoupling Features in Hidden Markov Models 

**Title (ZH)**: 基于Transformer的多任务学习者：解耦隐藏马尔可夫模型中的特征 

**Authors**: Yifan Hao, Chenlu Ye, Chi Han, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01919)  

**Abstract**: Transformer based models have shown remarkable capabilities in sequence learning across a wide range of tasks, often performing well on specific task by leveraging input-output examples. Despite their empirical success, a comprehensive theoretical understanding of this phenomenon remains limited. In this work, we investigate the layerwise behavior of Transformers to uncover the mechanisms underlying their multi-task generalization ability. Taking explorations on a typical sequence model, i.e, Hidden Markov Models, which are fundamental to many language tasks, we observe that: first, lower layers of Transformers focus on extracting feature representations, primarily influenced by neighboring tokens; second, on the upper layers, features become decoupled, exhibiting a high degree of time disentanglement. Building on these empirical insights, we provide theoretical analysis for the expressiveness power of Transformers. Our explicit constructions align closely with empirical observations, providing theoretical support for the Transformer's effectiveness and efficiency on sequence learning across diverse tasks. 

**Abstract (ZH)**: 基于Transformer的模型在广泛的任务中展示了在序列学习方面的卓越能力，常常通过输入-输出示例在特定任务上表现出色。尽管它们在实验上取得了成功，但对这一现象的全面理论理解仍然有限。本研究调查了Transformer的逐层行为，以揭示其实现多任务泛化能力的机制。通过探究典型的序列模型，即隐马尔可夫模型，该模型对于许多语言任务至关重要，我们发现：首先，Transformer的底层专注于提取特征表示，主要受相邻tokens的影响；其次，在高层，特征变得解耦，显示出高度的时间解纠缠。基于这些实验性洞察，我们提供了Transformer表达能力的理论分析。我们的显式构造与实验观察紧密一致，为Transformer在各种任务中进行序列学习的有效性和效率提供了理论支持。 

---
# CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention for Alzheimer's Detection 

**Title (ZH)**: CogniAlign: 基于门控交叉注意机制的词级多模态语音对齐方法及其在阿尔茨海默病检测中的应用 

**Authors**: David Ortiz-Perez, Manuel Benavent-Lledo, Javier Rodriguez-Juan, Jose Garcia-Rodriguez, David Tomás  

**Link**: [PDF](https://arxiv.org/pdf/2506.01890)  

**Abstract**: Early detection of cognitive disorders such as Alzheimer's disease is critical for enabling timely clinical intervention and improving patient outcomes. In this work, we introduce CogniAlign, a multimodal architecture for Alzheimer's detection that integrates audio and textual modalities, two non-intrusive sources of information that offer complementary insights into cognitive health. Unlike prior approaches that fuse modalities at a coarse level, CogniAlign leverages a word-level temporal alignment strategy that synchronizes audio embeddings with corresponding textual tokens based on transcription timestamps. This alignment supports the development of token-level fusion techniques, enabling more precise cross-modal interactions. To fully exploit this alignment, we propose a Gated Cross-Attention Fusion mechanism, where audio features attend over textual representations, guided by the superior unimodal performance of the text modality. In addition, we incorporate prosodic cues, specifically interword pauses, by inserting pause tokens into the text and generating audio embeddings for silent intervals, further enriching both streams. We evaluate CogniAlign on the ADReSSo dataset, where it achieves an accuracy of 90.36%, outperforming existing state-of-the-art methods. A detailed ablation study confirms the advantages of our alignment strategy, attention-based fusion, and prosodic modeling. 

**Abstract (ZH)**: 早期认知障碍如阿尔茨海默病的检测对于及时临床干预和改善患者预后至关重要。本文 introduces CogniAlign，一种结合音频和文本模态的多模态架构，用于阿尔茨海默病的检测，这两者是非侵入性的信息源，提供了认知健康互补的见解。不同于先前在粗略层面融合模态的方法，CogniAlign 利用基于转录时间戳的词级时间对齐策略，将音频嵌入与相应的文本令牌同步。这种对齐支持基于令牌的融合技术的发展，使得跨模态交互更加精确。为了充分利用这种对齐，我们提出了一种门控跨注意力融合机制，其中音频特征在文本表示的指导下关注文本。此外，通过插入暂停令牌并为静音间隔生成音频嵌入，我们还整合了语调线索。我们在 ADReSSo 数据集上评估了 CogniAlign，实现了 90.36% 的准确率，优于现有最先进的方法。详细的消融研究证实了我们对齐策略、基于注意力的融合和语调建模的优势。 

---
# Agnostic Reinforcement Learning: Foundations and Algorithms 

**Title (ZH)**: agnostic强化学习：基础与算法 

**Authors**: Gene Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01884)  

**Abstract**: Reinforcement Learning (RL) has demonstrated tremendous empirical success across numerous challenging domains. However, we lack a strong theoretical understanding of the statistical complexity of RL in environments with large state spaces, where function approximation is required for sample-efficient learning. This thesis addresses this gap by rigorously examining the statistical complexity of RL with function approximation from a learning theoretic perspective. Departing from a long history of prior work, we consider the weakest form of function approximation, called agnostic policy learning, in which the learner seeks to find the best policy in a given class $\Pi$, with no guarantee that $\Pi$ contains an optimal policy for the underlying task.
We systematically explore agnostic policy learning along three key axes: environment access -- how a learner collects data from the environment; coverage conditions -- intrinsic properties of the underlying MDP measuring the expansiveness of state-occupancy measures for policies in the class $\Pi$, and representational conditions -- structural assumptions on the class $\Pi$ itself. Within this comprehensive framework, we (1) design new learning algorithms with theoretical guarantees and (2) characterize fundamental performance bounds of any algorithm. Our results reveal significant statistical separations that highlight the power and limitations of agnostic policy learning. 

**Abstract (ZH)**: 强化学习（RL）在众多具有挑战性的领域中展现出了巨大的经验成功。然而，我们对在具有大规模状态空间的环境中需要函数逼近以实现高效样本学习的RL的统计复杂性缺乏强有力的理论理解。本文从学习理论的角度严格探讨了函数逼近环境下RL的统计复杂性，不同于以往工作的长期研究，我们考虑了最弱形式的函数逼近——即不保证函数类包含最优策略的无偏策略学习。我们系统地从环境访问、覆盖条件和表示条件三个关键维度探索了无偏策略学习，并在这一综合框架中（1）设计了具有理论保证的新学习算法，（2）界定了任何算法的基本性能界。我们的结果揭示了无偏策略学习的重要统计分离，突显了其能力和局限性。 

---
# scDataset: Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics 

**Title (ZH)**: scDataset: 可扩展的数据加载方法，应用于大规模单细胞组学深度学习 

**Authors**: Davide D'Ascenzo, Sebastiano Cultrera di Montesano  

**Link**: [PDF](https://arxiv.org/pdf/2506.01883)  

**Abstract**: Modern single-cell datasets now comprise hundreds of millions of cells, presenting significant challenges for training deep learning models that require shuffled, memory-efficient data loading. While the AnnData format is the community standard for storing single-cell datasets, existing data loading solutions for AnnData are often inadequate: some require loading all data into memory, others convert to dense formats that increase storage demands, and many are hampered by slow random disk access. We present scDataset, a PyTorch IterableDataset that operates directly on one or more AnnData files without the need for format conversion. The core innovation is a combination of block sampling and batched fetching, which together balance randomness and I/O efficiency. On the Tahoe 100M dataset, scDataset achieves up to a 48$\times$ speed-up over AnnLoader, a 27$\times$ speed-up over HuggingFace Datasets, and an 18$\times$ speed-up over BioNeMo in single-core settings. These advances democratize large-scale single-cell model training for the broader research community. 

**Abstract (ZH)**: 基于AnnData的块采样批加载方案scDataset及其在大规模单细胞模型训练中的应用 

---
# Learning to Explore: An In-Context Learning Approach for Pure Exploration 

**Title (ZH)**: 学习探索：一种纯探索的上下文学习方法 

**Authors**: Alessio Russo, Ryan Welch, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2506.01876)  

**Abstract**: In this work, we study the active sequential hypothesis testing problem, also known as pure exploration, where the goal is to actively control a data collection process to efficiently identify the correct hypothesis underlying a decision problem. While relevant across multiple domains, devising adaptive exploration strategies remains challenging, particularly due to difficulties in encoding appropriate inductive biases. Existing Reinforcement Learning (RL)-based methods often underperform when relevant information structures are inadequately represented, whereas more complex methods, like Best Arm Identification (BAI) techniques, may be difficult to devise and typically rely on explicit modeling assumptions. To address these limitations, we introduce In-Context Pure Exploration (ICPE), an in-context learning approach that uses Transformers to learn exploration strategies directly from experience. ICPE combines supervised learning and reinforcement learning to identify and exploit latent structure across related tasks, without requiring prior assumptions. Numerical results across diverse synthetic and semi-synthetic benchmarks highlight ICPE's capability to achieve robust performance performance in deterministic, stochastic, and structured settings. These results demonstrate ICPE's ability to match optimal instance-dependent algorithms using only deep learning techniques, making it a practical and general approach to data-efficient exploration. 

**Abstract (ZH)**: 基于上下文的纯探索研究（ICPE）：一种结合变换器的探索策略学习方法 

---
# Frugal Machine Learning for Energy-efficient, and Resource-aware Artificial Intelligence 

**Title (ZH)**: 经济高效且资源意识强的机器学习方法 

**Authors**: John Violos, Konstantina-Christina Diamanti, Ioannis Kompatsiaris, Symeon Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.01869)  

**Abstract**: Frugal Machine Learning (FML) refers to the practice of designing Machine Learning (ML) models that are efficient, cost-effective, and mindful of resource constraints. This field aims to achieve acceptable performance while minimizing the use of computational resources, time, energy, and data for both training and inference. FML strategies can be broadly categorized into input frugality, learning process frugality, and model frugality, each focusing on reducing resource consumption at different stages of the ML pipeline. This chapter explores recent advancements, applications, and open challenges in FML, emphasizing its importance for smart environments that incorporate edge computing and IoT devices, which often face strict limitations in bandwidth, energy, or latency. Technological enablers such as model compression, energy-efficient hardware, and data-efficient learning techniques are discussed, along with adaptive methods including parameter regularization, knowledge distillation, and dynamic architecture design that enable incremental model updates without full retraining. Furthermore, it provides a comprehensive taxonomy of frugal methods, discusses case studies across diverse domains, and identifies future research directions to drive innovation in this evolving field. 

**Abstract (ZH)**: 节俭机器学习（FML）指的是设计高效、成本效益高并在资源受限情况下保持意识的机器学习模型的做法。该领域旨在在最小化计算资源、时间和能量的使用（包括训练和推理）的前提下，实现可接受的性能。FML策略可以根据在机器学习管道的不同阶段减少资源消耗，大致分为输入节俭、学习过程节俭和模型节俭。本章探讨了FML领域的最新进展、应用和开放挑战，突出了其对于包括边缘计算和物联网设备在内的智能环境的重要性，这些设备通常在带宽、能源或延迟方面受到严格限制。讨论了模型压缩、能源高效硬件、数据高效学习技术等技术使能器，以及包括参数正则化、知识蒸馏和动态架构设计在内的自适应方法，这些方法能够在无需完全重新训练的情况下实现增量模型更新。此外，提供了节俭方法的综合分类，讨论了跨不同领域的案例研究，并指出了未来的研究方向，以推动这一不断发展的领域的创新。 

---
# MoDA: Modulation Adapter for Fine-Grained Visual Grounding in Instructional MLLMs 

**Title (ZH)**: MoDA: 调制适配器在指令式MLLLMs中细粒度视觉定位中的应用 

**Authors**: Wayner Barrios, Andrés Villa, Juan León Alcázar, SouYoung Jin, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2506.01850)  

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated impressive performance on instruction-following tasks by integrating pretrained visual encoders with large language models (LLMs). However, existing approaches often struggle to ground fine-grained visual concepts in complex scenes. In this paper, we propose MoDA (Modulation Adapter), a lightweight yet effective module designed to refine pre-aligned visual features through instruction-guided modulation. Our approach follows the standard LLaVA training protocol, consisting of a two-stage process: (1) aligning image features to the LLMs input space via a frozen vision encoder and adapter layers, and (2) refining those features using the MoDA adapter during the instructional tuning stage. MoDA employs a Transformer-based cross-attention mechanism to generate a modulation mask over the aligned visual tokens, thereby emphasizing semantically relevant embedding dimensions based on the language instruction. The modulated features are then passed to the LLM for autoregressive language generation. Our experimental evaluation shows that MoDA improves visual grounding and generates more contextually appropriate responses, demonstrating its effectiveness as a general-purpose enhancement for image-based MLLMs. 

**Abstract (ZH)**: Recently, 多模态大型语言模型（MLLMs）通过将预训练的视觉编码器与大型语言模型（LLMs）结合，在指令跟随任务中展现了 impressive 的性能。然而，现有方法往往难以在复杂场景中细粒度地ground视觉概念。本文中，我们提出 MoDA（调制适配器），这是一种轻量级但有效的模块，设计用于通过指令引导的调制细化预先对齐的视觉特征。我们的方法遵循标准的 LLaVA 训练协议，包括两阶段过程：（1）通过冻结的视觉编码器和适配器层将图像特征对齐到 LLMs 的输入空间，（2）在指令调整阶段使用 MoDA 适配器细化这些特征。MoDA 使用基于Transformer的跨注意力机制生成调制掩码，从而根据语言指令强调语义相关的嵌入维度。调制后的特征随后传递给 LLM 进行自回归语言生成。我们的实验评估表明，MoDA 改进了视觉grounding，产生了更上下文相关的目标响应，证明了其作为图像基础 MLLMs 通用增强手段的有效性。 

---
# CiteEval: Principle-Driven Citation Evaluation for Source Attribution 

**Title (ZH)**: CiteEval: 原则驱动的引用评价及其来源归属 

**Authors**: Yumo Xu, Peng Qi, Jifan Chen, Kunlun Liu, Rujun Han, Lan Liu, Bonan Min, Vittorio Castelli, Arshit Gupta, Zhiguo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01829)  

**Abstract**: Citation quality is crucial in information-seeking systems, directly influencing trust and the effectiveness of information access. Current evaluation frameworks, both human and automatic, mainly rely on Natural Language Inference (NLI) to assess binary or ternary supportiveness from cited sources, which we argue is a suboptimal proxy for citation evaluation. In this work we introduce CiteEval, a citation evaluation framework driven by principles focusing on fine-grained citation assessment within a broad context, encompassing not only the cited sources but the full retrieval context, user query, and generated text. Guided by the proposed framework, we construct CiteBench, a multi-domain benchmark with high-quality human annotations on citation quality. To enable efficient evaluation, we further develop CiteEval-Auto, a suite of model-based metrics that exhibit strong correlation with human judgments. Experiments across diverse systems demonstrate CiteEval-Auto's superior ability to capture the multifaceted nature of citations compared to existing metrics, offering a principled and scalable approach to evaluate and improve model-generated citations. 

**Abstract (ZH)**: 引文质量对于信息检索系统至关重要，直接影响信息访问的可信度和有效性。当前的评估框架，无论是人工的还是自动的，主要依赖于自然语言推理(NLI)来评估引文支持性，我们认为这只是一个次优的引文评估代理。在此项工作中，我们引入了CiteEval，这是一种以细粒度引文评估为核心原则的引文评估框架，不仅涵盖引文来源，还涉及全面的检索上下文、用户查询和生成文本。基于提出的框架，我们构建了CiteBench，这是一个多领域基准，包含了高质量的人工标注的引文质量。为实现高效评估，我们进一步开发了CiteEval-Auto，这是一个基于模型的度量套件，与人工判断具有较强的关联性。实验表明，CiteEval-Auto在捕捉引文的复杂性方面优于现有度量，提供了一种有原则且可扩展的方法来评估和改进模型生成的引文。 

---
# A Quantum Information Theoretic Approach to Tractable Probabilistic Models 

**Title (ZH)**: 基于可处理概率模型的量子信息理论方法 

**Authors**: Pedro Zuidberg Dos Martires  

**Link**: [PDF](https://arxiv.org/pdf/2506.01824)  

**Abstract**: By recursively nesting sums and products, probabilistic circuits have emerged in recent years as an attractive class of generative models as they enjoy, for instance, polytime marginalization of random variables. In this work we study these machine learning models using the framework of quantum information theory, leading to the introduction of positive unital circuits (PUnCs), which generalize circuit evaluations over positive real-valued probabilities to circuit evaluations over positive semi-definite matrices. As a consequence, PUnCs strictly generalize probabilistic circuits as well as recently introduced circuit classes such as PSD circuits. 

**Abstract (ZH)**: 通过递归嵌套和的概率电路最近成为了生成模型的一种有吸引力的类别，因为它们可以实现多项式时间的概率变量边缘化。在本工作中，我们利用量子信息理论的框架研究这些机器学习模型，从而引出了正单位电路（PUnC），该电路将电路评估从正实数概率推广到正半定矩阵。作为结果，PUnC 严格地推广了概率电路以及最近引入的如PSD电路等电路类别。 

---
# Ridgeformer: Mutli-Stage Contrastive Training For Fine-grained Cross-Domain Fingerprint Recognition 

**Title (ZH)**: ridgeformer：多阶段对比训练的小尺度跨域指纹识别 

**Authors**: Shubham Pandey, Bhavin Jawade, Srirangaraj Setlur  

**Link**: [PDF](https://arxiv.org/pdf/2506.01806)  

**Abstract**: The increasing demand for hygienic and portable biometric systems has underscored the critical need for advancements in contactless fingerprint recognition. Despite its potential, this technology faces notable challenges, including out-of-focus image acquisition, reduced contrast between fingerprint ridges and valleys, variations in finger positioning, and perspective distortion. These factors significantly hinder the accuracy and reliability of contactless fingerprint matching. To address these issues, we propose a novel multi-stage transformer-based contactless fingerprint matching approach that first captures global spatial features and subsequently refines localized feature alignment across fingerprint samples. By employing a hierarchical feature extraction and matching pipeline, our method ensures fine-grained, cross-sample alignment while maintaining the robustness of global feature representation. We perform extensive evaluations on publicly available datasets such as HKPolyU and RidgeBase under different evaluation protocols, such as contactless-to-contact matching and contactless-to-contactless matching and demonstrate that our proposed approach outperforms existing methods, including COTS solutions. 

**Abstract (ZH)**: 不断增加的对卫生和便携的生物识别系统的需求凸显了无接触指纹识别技术进步的critical需要。尽管这项技术具有潜力，但它面临着显著的挑战，包括成像对焦不良、指纹嵴和谷之间对比度降低、手指定位差异以及视角失真。这些因素显著影响了无接触指纹匹配的准确性和可靠性。为了解决这些问题，我们提出了一种新颖的多阶段变压器基无接触指纹匹配方法，首先捕获全局空间特征，然后在指纹样本之间细化局部特征对齐。通过采用分层特征提取和匹配流水线，我们的方法确保了细粒度的跨样本对齐，同时保持了全局特征表示的鲁棒性。我们在不同的评估协议下，如无接触至接触匹配和无接触至无接触匹配，对公开可用的数据集（如HKPolyU和RidgeBase）进行了广泛的评估，并展示了我们的提出的方法优于现有方法，包括商用即用型（COTS）解决方案。 

---
# Datasheets Aren't Enough: DataRubrics for Automated Quality Metrics and Accountability 

**Title (ZH)**: 数据表单不足以涵盖全部：数据评 rubrics 用于自动化质量指标和问责制 

**Authors**: Genta Indra Winata, David Anugraha, Emmy Liu, Alham Fikri Aji, Shou-Yi Hung, Aditya Parashar, Patrick Amadeus Irawan, Ruochen Zhang, Zheng-Xin Yong, Jan Christian Blaise Cruz, Niklas Muennighoff, Seungone Kim, Hanyang Zhao, Sudipta Kar, Kezia Erina Suryoraharjo, M. Farid Adilazuarda, En-Shiun Annie Lee, Ayu Purwarianti, Derry Tanti Wijaya, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.01789)  

**Abstract**: High-quality datasets are fundamental to training and evaluating machine learning models, yet their creation-especially with accurate human annotations-remains a significant challenge. Many dataset paper submissions lack originality, diversity, or rigorous quality control, and these shortcomings are often overlooked during peer review. Submissions also frequently omit essential details about dataset construction and properties. While existing tools such as datasheets aim to promote transparency, they are largely descriptive and do not provide standardized, measurable methods for evaluating data quality. Similarly, metadata requirements at conferences promote accountability but are inconsistently enforced. To address these limitations, this position paper advocates for the integration of systematic, rubric-based evaluation metrics into the dataset review process-particularly as submission volumes continue to grow. We also explore scalable, cost-effective methods for synthetic data generation, including dedicated tools and LLM-as-a-judge approaches, to support more efficient evaluation. As a call to action, we introduce DataRubrics, a structured framework for assessing the quality of both human- and model-generated datasets. Leveraging recent advances in LLM-based evaluation, DataRubrics offers a reproducible, scalable, and actionable solution for dataset quality assessment, enabling both authors and reviewers to uphold higher standards in data-centric research. We also release code to support reproducibility of LLM-based evaluations at this https URL. 

**Abstract (ZH)**: 高质量的数据集是训练和评估机器学习模型的基础，但其创建尤其是精准的人工注解创建仍是一项重大挑战。许多数据集论文提交缺乏原创性、多样性和严格的质量控制，这些不足在同行评审过程中往往被忽略。提交内容还经常省略关于数据集构建和属性的关键细节。虽然现有的工具如数据表旨在促进透明度，它们主要是描述性的，并不能提供标准化、可衡量的方法来评估数据质量。同样，会议中的元数据要求促进问责制，但执行上却不够一致。为解决这些局限性，本文倡导在数据集评审过程中整合基于规范和量表的评估指标，特别是在提交量继续增长的情况下。我们还探讨了生成合成数据的可扩展、低成本方法，包括专用工具和LLM-as-a-judge方法，以支持更高效的评估。作为行动呼吁，我们引入了DataRubrics，这是一种结构化的框架，用于评估人类生成和模型生成的数据集质量。利用基于LLM的评估的最新进展，DataRubrics提供了一种可重复、可扩展且可操作的数据集质量评估解决方案，使作者和评审者能够提高数据为中心的研究标准。我们还发布了代码以支持LLM基于的评估的可重复性，详见此链接：https://yourlinkhere。 

---
# iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering 

**Title (ZH)**: iQUEST：一种迭代问题导向的知识库问答框架 

**Authors**: Shuai Wang, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01784)  

**Abstract**: While Large Language Models (LLMs) excel at many natural language processing tasks, they often suffer from factual inaccuracies in knowledge-intensive scenarios. Integrating external knowledge resources, particularly knowledge graphs (KGs), provides a transparent and updatable foundation for more reliable reasoning. Knowledge Base Question Answering (KBQA), which queries and reasons over KGs, is central to this effort, especially for complex, multi-hop queries. However, multi-hop reasoning poses two key challenges: (1)~maintaining coherent reasoning paths, and (2)~avoiding prematurely discarding critical multi-hop connections. To address these issues, we introduce iQUEST, a question-guided KBQA framework that iteratively decomposes complex queries into simpler sub-questions, ensuring a structured and focused reasoning trajectory. Additionally, we integrate a Graph Neural Network (GNN) to look ahead and incorporate 2-hop neighbor information at each reasoning step. This dual approach strengthens the reasoning process, enabling the model to explore viable paths more effectively. Detailed experiments demonstrate the consistent improvement delivered by iQUEST across four benchmark datasets and four LLMs. 

**Abstract (ZH)**: 大型语言模型在许多自然语言处理任务中表现出色，但在知识密集型场景中往往会遇到事实准确性问题。通过集成外部知识资源，特别是知识图谱（KGs），可以为更加可靠的知识推理提供透明且可更新的基础。知识库问答（KBQA），即查询和在知识图谱上推理，是解决这一问题的关键，尤其是在处理复杂、多跳查询方面。然而，多跳推理带来两个关键挑战：（1）保持连贯的推理路径，（2）避免过早丢弃关键的多跳连接。为应对这些挑战，我们引入了iQUEST，这是一种基于问题的KBQA框架，通过迭代分解复杂查询为更简单的子问题，确保结构化和重点明确的推理轨迹。此外，我们集成了一个图神经网络（GNN）来提前查看并在每一个推理步骤中整合两个跳邻接信息。这种双重方法增强了推理过程，使模型能够更有效地探索可行路径。详细实验表明，iQUEST在四个基准数据集和四个大型语言模型上一致地提供了改进。 

---
# Systematic Hazard Analysis for Frontier AI using STPA 

**Title (ZH)**: 面向前沿人工智能的STPA系统性危害分析 

**Authors**: Simon Mylius  

**Link**: [PDF](https://arxiv.org/pdf/2506.01782)  

**Abstract**: All of the frontier AI companies have published safety frameworks where they define capability thresholds and risk mitigations that determine how they will safely develop and deploy their models. Adoption of systematic approaches to risk modelling, based on established practices used in safety-critical industries, has been recommended, however frontier AI companies currently do not describe in detail any structured approach to identifying and analysing hazards. STPA (Systems-Theoretic Process Analysis) is a systematic methodology for identifying how complex systems can become unsafe, leading to hazards. It achieves this by mapping out controllers and controlled processes then analysing their interactions and feedback loops to understand how harmful outcomes could occur (Leveson & Thomas, 2018). We evaluate STPA's ability to broaden the scope, improve traceability and strengthen the robustness of safety assurance for frontier AI systems. Applying STPA to the threat model and scenario described in 'A Sketch of an AI Control Safety Case' (Korbak et al., 2025), we derive a list of Unsafe Control Actions. From these we select a subset and explore the Loss Scenarios that lead to them if left unmitigated. We find that STPA is able to identify causal factors that may be missed by unstructured hazard analysis methodologies thereby improving robustness. We suggest STPA could increase the safety assurance of frontier AI when used to complement or check coverage of existing AI governance techniques including capability thresholds, model evaluations and emergency procedures. The application of a systematic methodology supports scalability by increasing the proportion of the analysis that could be conducted by LLMs, reducing the burden on human domain experts. 

**Abstract (ZH)**: 前沿AI公司的安全性框架及其基于STPA方法的安全性保障改进研究 

---
# Enhancing Customer Service Chatbots with Context-Aware NLU through Selective Attention and Multi-task Learning 

**Title (ZH)**: 基于选择性注意和多任务学习的情境感知NLU增强客户服务聊天机器人 

**Authors**: Subhadip Nandi, Neeraj Agrawal, Anshika Singh, Priyanka Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.01781)  

**Abstract**: Customer service chatbots are conversational systems aimed at addressing customer queries, often by directing them to automated workflows. A crucial aspect of this process is the classification of the customer's intent. Presently, most intent classification models for customer care utilise only customer query for intent prediction. This may result in low-accuracy models, which cannot handle ambiguous queries. An ambiguous query like "I didn't receive my package" could indicate a delayed order, or an order that was delivered but the customer failed to receive it. Resolution of each of these scenarios requires the execution of very different sequence of steps. Utilizing additional information, such as the customer's order delivery status, in the right manner can help identify the intent for such ambiguous queries. In this paper, we have introduced a context-aware NLU model that incorporates both, the customer query and contextual information from the customer's order status for predicting customer intent. A novel selective attention module is used to extract relevant context features. We have also proposed a multi-task learning paradigm for the effective utilization of different label types available in our training data. Our suggested method, Multi-Task Learning Contextual NLU with Selective Attention Weighted Context (MTL-CNLU-SAWC), yields a 4.8% increase in top 2 accuracy score over the baseline model which only uses user queries, and a 3.5% improvement over existing state-of-the-art models that combine query and context. We have deployed our model to production for Walmart's customer care domain. Accurate intent prediction through MTL-CNLU-SAWC helps to better direct customers to automated workflows, thereby significantly reducing escalations to human agents, leading to almost a million dollars in yearly savings for the company. 

**Abstract (ZH)**: 基于选择性注意力加权上下文的多任务学习上下文NLU模型（MTL-CNLU-SAWC）及其在客户服务聊天机器人中的应用 

---
# unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning 

**Title (ZH)**: unMORE: 无监督多对象分割通过中心-边界推理 

**Authors**: Yafei Yang, Zihui Zhang, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01778)  

**Abstract**: We study the challenging problem of unsupervised multi-object segmentation on single images. Existing methods, which rely on image reconstruction objectives to learn objectness or leverage pretrained image features to group similar pixels, often succeed only in segmenting simple synthetic objects or discovering a limited number of real-world objects. In this paper, we introduce unMORE, a novel two-stage pipeline designed to identify many complex objects in real-world images. The key to our approach involves explicitly learning three levels of carefully defined object-centric representations in the first stage. Subsequently, our multi-object reasoning module utilizes these learned object priors to discover multiple objects in the second stage. Notably, this reasoning module is entirely network-free and does not require human labels. Extensive experiments demonstrate that unMORE significantly outperforms all existing unsupervised methods across 6 real-world benchmark datasets, including the challenging COCO dataset, achieving state-of-the-art object segmentation results. Remarkably, our method excels in crowded images where all baselines collapse. 

**Abstract (ZH)**: 我们研究单张图像上无监督多对象分割的挑战性问题。现有方法依赖于图像重建目标来学习对象性或利用预训练的图像特征来聚类相似的像素，往往只能成功分割简单的合成对象或发现少量的真实世界对象。在本文中，我们提出了一种名为unMORE的新型两阶段管道，旨在识别实际图像中的许多复杂对象。我们的方法的关键在于第一阶段明确学习三个层次的精心定义的对象中心表示。随后，我们的多对象推理模块利用这些学到的对象先验在第二阶段发现多个对象。值得注意的是，该推理模块完全无网络依赖，并不需要人工标签。广泛的实验表明，unMORE在包括具有挑战性的COCO数据集在内的6个真实世界基准数据集上显著优于所有现有的无监督方法，实现了最先进的对象分割结果。特别地，我们的方法在所有基准方法失效的拥挤图像中表现优异。 

---
# MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation 

**Title (ZH)**: MaXIFE：多语言和跨语言指令跟随评估 

**Authors**: Yile Liu, Ziwei Ma, Xiu Jiang, Jinglu Hu, Jing Chang, Liang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01776)  

**Abstract**: With the rapid adoption of large language models (LLMs) in natural language processing, the ability to follow instructions has emerged as a key metric for evaluating their practical utility. However, existing evaluation methods often focus on single-language scenarios, overlooking the challenges and differences present in multilingual and cross-lingual contexts. To address this gap, we introduce MaXIFE: a comprehensive evaluation benchmark designed to assess instruction-following capabilities across 23 languages with 1,667 verifiable instruction tasks. MaXIFE integrates both Rule-Based Evaluation and Model-Based Evaluation, ensuring a balance of efficiency and accuracy. We applied MaXIFE to evaluate several leading commercial and open-source LLMs, establishing baseline results for future comparisons. By providing a standardized tool for multilingual instruction-following evaluation, MaXIFE aims to advance research and development in natural language processing. 

**Abstract (ZH)**: 大规模语言模型在自然语言处理中的快速采用使得遵循指令的能力成为了评估其实用价值的关键指标。然而，现有的评估方法往往侧重于单一语言场景，忽视了多语言和跨语言环境中的挑战和差异。为解决这一问题，我们提出了MaXIFE：一个旨在评估23种语言中1,667项可验证指令任务的综合评估基准，以评估指令跟随能力。MaXIFE结合了基于规则的评估和基于模型的评估，确保效率与准确性的平衡。我们应用MaXIFE对几种领先的商业和开源大规模语言模型进行了评估，建立了未来比较的基础结果。通过提供一个多语言指令跟随评估的标准工具，MaXIFE旨在促进自然语言处理领域的研究与开发。 

---
# Greening AI-enabled Systems with Software Engineering: A Research Agenda for Environmentally Sustainable AI Practices 

**Title (ZH)**: 利用软件工程实现AI赋能系统的绿色化：面向环境可持续的AI实践研究议程 

**Authors**: Luís Cruz, João Paulo Fernandes, Maja H. Kirkeby, Silverio Martínez-Fernández, June Sallou, Hina Anwar, Enrique Barba Roque, Justus Bogner, Joel Castaño, Fernando Castor, Aadil Chasmawala, Simão Cunha, Daniel Feitosa, Alexandra González, Andreas Jedlitschka, Patricia Lago, Ana Oprescu, Pooja Rani, João Saraiva, Federica Sarro, Raghavendra Selvan, Karthik Vaidhyanathan, Roberto Verdecchia, Ivan P. Yamshchikov, Henry Muccini  

**Link**: [PDF](https://arxiv.org/pdf/2506.01774)  

**Abstract**: The environmental impact of Artificial Intelligence (AI)-enabled systems is increasing rapidly, and software engineering plays a critical role in developing sustainable solutions. The "Greening AI with Software Engineering" CECAM-Lorentz workshop (no. 1358, 2025) funded by the Centre Européen de Calcul Atomique et Moléculaire and the Lorentz Center, provided an interdisciplinary forum for 29 participants, from practitioners to academics, to share knowledge, ideas, practices, and current results dedicated to advancing green software and AI research. The workshop was held February 3-7, 2025, in Lausanne, Switzerland. Through keynotes, flash talks, and collaborative discussions, participants identified and prioritized key challenges for the field. These included energy assessment and standardization, benchmarking practices, sustainability-aware architectures, runtime adaptation, empirical methodologies, and education. This report presents a research agenda emerging from the workshop, outlining open research directions and practical recommendations to guide the development of environmentally sustainable AI-enabled systems rooted in software engineering principles. 

**Abstract (ZH)**: 人工智能(AI)使能系统对环境的影响正在迅速增加，软件工程在开发可持续解决方案中发挥了关键作用。由欧洲原子分子计算中心和洛伦兹中心资助的“通过软件工程使AI绿化”CECAM-洛伦兹研讨会（编号1358，2025）为来自从业者到学术界共计29位参与者提供了一个跨学科论坛，分享有关推进绿色软件和AI研究的知识、理念、实践和当前成果。该研讨会于2025年2月3日至7日在瑞士洛桑举行。通过主题演讲、快速发言和协作讨论，参与者识别和优先考虑了该领域的关键挑战，包括能源评估和标准化、基准测试实践、环境意识架构、运行时适应、实证方法论以及教育。本报告概述了从此次研讨会中 emergence 的研究议程，明确了开放的研究方向和实用建议，以指导基于软件工程原则的环境可持续AI使能系统的开发。 

---
# ReGA: Representation-Guided Abstraction for Model-based Safeguarding of LLMs 

**Title (ZH)**: ReGA: 基于表示的抽象方法用于LLM的模型导向安全保障 

**Authors**: Zeming Wei, Chengcan Wu, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01770)  

**Abstract**: Large Language Models (LLMs) have achieved significant success in various tasks, yet concerns about their safety and security have emerged. In particular, they pose risks in generating harmful content and vulnerability to jailbreaking attacks. To analyze and monitor machine learning models, model-based analysis has demonstrated notable potential in stateful deep neural networks, yet suffers from scalability issues when extending to LLMs due to their vast feature spaces. In this paper, we propose ReGA, a model-based analysis framework with representation-guided abstraction, to safeguard LLMs against harmful prompts and generations. By leveraging safety-critical representations, which are low-dimensional directions emerging in hidden states that indicate safety-related concepts, ReGA effectively addresses the scalability issue when constructing the abstract model for safety modeling. Our comprehensive evaluation shows that ReGA performs sufficiently well in distinguishing between safe and harmful inputs, achieving an AUROC of 0.975 at the prompt level and 0.985 at the conversation level. Additionally, ReGA exhibits robustness to real-world attacks and generalization across different safety perspectives, outperforming existing safeguard paradigms in terms of interpretability and scalability. Overall, ReGA serves as an efficient and scalable solution to enhance LLM safety by integrating representation engineering with model-based abstraction, paving the way for new paradigms to utilize software insights for AI safety. Our code is available at this https URL. 

**Abstract (ZH)**: 基于表示引导抽象的大语言模型安全性分析框架ReGA 

---
# Efficient Egocentric Action Recognition with Multimodal Data 

**Title (ZH)**: 基于多模态数据的高效自我中心动作识别 

**Authors**: Marco Calzavara, Ard Kastrati, Matteo Macchini, Dushan Vasilevski, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2506.01757)  

**Abstract**: The increasing availability of wearable XR devices opens new perspectives for Egocentric Action Recognition (EAR) systems, which can provide deeper human understanding and situation awareness. However, deploying real-time algorithms on these devices can be challenging due to the inherent trade-offs between portability, battery life, and computational resources. In this work, we systematically analyze the impact of sampling frequency across different input modalities - RGB video and 3D hand pose - on egocentric action recognition performance and CPU usage. By exploring a range of configurations, we provide a comprehensive characterization of the trade-offs between accuracy and computational efficiency. Our findings reveal that reducing the sampling rate of RGB frames, when complemented with higher-frequency 3D hand pose input, can preserve high accuracy while significantly lowering CPU demands. Notably, we observe up to a 3x reduction in CPU usage with minimal to no loss in recognition performance. This highlights the potential of multimodal input strategies as a viable approach to achieving efficient, real-time EAR on XR devices. 

**Abstract (ZH)**: 穿戴式XR设备数量的增加为自视点动作识别系统（EAR）提供了新的视角，这些系统可以提供更深入的人类理解和情况意识。然而，由于轻便性、电池寿命和计算资源之间的固有权衡，将实时算法部署在这些设备上可能会颇具挑战性。在这项工作中，我们系统地分析了不同输入模态——RGB视频和3D手部姿态——的采样频率对自视点动作识别性能和CPU使用率的影响。通过探索多种配置，我们提供了关于准确性和计算效率之间权衡关系的全面Characterization。我们的研究发现，当与更高频率的3D手部姿态输入相结合时，降低RGB帧的采样率可以保持高精度的同时显著降低CPU需求。值得注意的是，我们观察到CPU使用率最多可降低3倍，同时识别性能基本保持不变。这突显了多模态输入策略作为在XR设备上实现高效实时自视点动作识别的可行方法的潜力。 

---
# Principled data augmentation for learning to solve quadratic programming problems 

**Title (ZH)**: 原理性的数据增强方法用于求解二次规划问题 

**Authors**: Chendi Qian, Christopher Morris  

**Link**: [PDF](https://arxiv.org/pdf/2506.01728)  

**Abstract**: Linear and quadratic optimization are crucial in numerous real-world applications, from training machine learning models to integer-linear optimization. Recently, learning-to-optimize methods (L2O) for linear (LPs) or quadratic programs (QPs) using message-passing graph neural networks (MPNNs) have gained traction, promising lightweight, data-driven proxies for solving such optimization problems. For example, they replace the costly computation of strong branching scores in branch-and-bound solvers, requiring solving many such optimization problems. However, robust L2O MPNNs remain challenging in data-scarce settings, especially when addressing complex optimization problems such as QPs. This work introduces a principled approach to data augmentation tailored for QPs via MPNNs. Our method leverages theoretically justified data augmentation techniques to generate diverse yet optimality-preserving instances. Furthermore, we integrate these augmentations into a self-supervised learning framework based on contrastive learning, thereby pretraining MPNNs for enhanced performance on L2O tasks. Extensive experiments demonstrate that our approach improves generalization in supervised scenarios and facilitates effective transfer learning to related optimization problems. 

**Abstract (ZH)**: 基于图神经网络的消息传递学习-to-优化方法的数据增强：用于二次规划的原理性方法 

---
# Tug-of-war between idiom's figurative and literal meanings in LLMs 

**Title (ZH)**: LLMs中成语的比喻义与字面义之间的争夺战 

**Authors**: Soyoung Oh, Xinting Huang, Mathis Pink, Michael Hahn, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.01723)  

**Abstract**: Idioms present a unique challenge for language models due to their non-compositional figurative meanings, which often strongly diverge from the idiom's literal interpretation. This duality requires a model to learn representing and deciding between the two meanings to interpret an idiom in a figurative sense, or literally. In this paper, we employ tools from mechanistic interpretability to trace how a large pretrained causal transformer (LLama3.2-1B-base) deals with this ambiguity. We localize three steps of idiom processing: First, the idiom's figurative meaning is retrieved in early attention and MLP sublayers. We identify specific attention heads which boost the figurative meaning of the idiom while suppressing the idiom's literal interpretation. The model subsequently represents the figurative representation through an intermediate path. Meanwhile, a parallel bypass route forwards literal interpretation, ensuring that a both reading remain available. Overall, our findings provide a mechanistic evidence for idiom comprehension in an autoregressive transformer. 

**Abstract (ZH)**: IDIOMS 给语言模型带来独特的挑战，因为它们的比喻意义是非组合性的，往往与字面意义有显著差异。这种二元性要求模型学会表示和决定两种意义，以便在比喻意义上解释成语，或者按字面意义理解。在本文中，我们利用机制可解释性工具追踪大规模预训练因果变换器（LLama3.2-1B-base）如何处理这种歧义。我们定位了成语处理的三个步骤：首先，在早期注意力和MLP子层中检索成语的比喻意义。我们确定了特定的注意力头，它们增强了成语的比喻意义同时抑制了字面意义。随后，模型通过中间路径表示比喻表示，同时通过并行旁路路径前向传递字面意义，确保两种读法都可用。总体而言，我们的发现为自回归变换器中的成语理解提供了机制性的证据。 

---
# Data Pruning by Information Maximization 

**Title (ZH)**: 信息最大化驱动的数据剪枝 

**Authors**: Haoru Tan, Sitong Wu, Wei Huang, Shizhen Zhao, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01701)  

**Abstract**: In this paper, we present InfoMax, a novel data pruning method, also known as coreset selection, designed to maximize the information content of selected samples while minimizing redundancy. By doing so, InfoMax enhances the overall informativeness of the coreset. The information of individual samples is measured by importance scores, which capture their influence or difficulty in model learning. To quantify redundancy, we use pairwise sample similarities, based on the premise that similar samples contribute similarly to the learning process. We formalize the coreset selection problem as a discrete quadratic programming (DQP) task, with the objective of maximizing the total information content, represented as the sum of individual sample contributions minus the redundancies introduced by similar samples within the coreset. To ensure practical scalability, we introduce an efficient gradient-based solver, complemented by sparsification techniques applied to the similarity matrix and dataset partitioning strategies. This enables InfoMax to seamlessly scale to datasets with millions of samples. Extensive experiments demonstrate the superior performance of InfoMax in various data pruning tasks, including image classification, vision-language pre-training, and instruction tuning for large language models. 

**Abstract (ZH)**: 本文提出了InfoMax，一种新型的数据剪裁方法，也称为核心集选择，旨在最大化所选样本的信息含量的同时最小化冗余。通过这种方式，InfoMax 提高了核心集的整体信息量。个体样本的信息量通过重要性得分来衡量，这些得分捕捉了其对模型学习的影响或难度。为了量化冗余，我们使用基于相似样本在学习过程中贡献相似性的样本对相似性。我们将核心集选择问题形式化为一个离散二次规划（DQP）问题，目标是最化总信息含量，即个体样本贡献的总和减去核心集内相似样本引入的冗余。为了确保实际可扩展性，我们引入了一种高效的梯度基解算器，并结合了相似性矩阵的稀疏化技术和数据集分区策略，这使InfoMax能够无缝扩展到包含数百万样本的数据集。广泛的经验表明，InfoMax 在各种数据剪裁任务中表现出色，包括图像分类、视觉-语言预训练以及大型语言模型的指令调优。 

---
# When LLMs Team Up: The Emergence of Collaborative Affective Computing 

**Title (ZH)**: 当大型语言模型联手：合作情感计算的兴起 

**Authors**: Wenna Lai, Haoran Xie, Guandong Xu, Qing Li, S. Joe Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01698)  

**Abstract**: Affective Computing (AC) is essential in bridging the gap between human emotional experiences and machine understanding. Traditionally, AC tasks in natural language processing (NLP) have been approached through pipeline architectures, which often suffer from structure rigidity that leads to inefficiencies and limited adaptability. The advent of Large Language Models (LLMs) has revolutionized this field by offering a unified approach to affective understanding and generation tasks, enhancing the potential for dynamic, real-time interactions. However, LLMs face cognitive limitations in affective reasoning, such as misinterpreting cultural nuances or contextual emotions, and hallucination problems in decision-making. To address these challenges, recent research advocates for LLM-based collaboration systems that emphasize interactions among specialized models and LLMs, mimicking human-like affective intelligence through the synergy of emotional and rational thinking that aligns with Dual Process Theory in psychology. This survey aims to provide a comprehensive overview of LLM-based collaboration systems in AC, exploring from structured collaborations to autonomous collaborations. Specifically, it includes: (1) A systematic review of existing methods, focusing on collaboration strategies, mechanisms, key functions, and applications; (2) Experimental comparisons of collaboration strategies across representative tasks in affective understanding and generation; (3) An analysis highlighting the potential of these systems to enhance robustness and adaptability in complex affective reasoning; (4) A discussion of key challenges and future research directions to further advance the field. This work is the first to systematically explore collaborative intelligence with LLMs in AC, paving the way for more powerful applications that approach human-like social intelligence. 

**Abstract (ZH)**: 情感计算中基于大型语言模型的合作系统：原理、实验与未来方向 

---
# Overcoming Data Scarcity in Scanning Tunnelling Microscopy Image Segmentation 

**Title (ZH)**: 克服扫描隧道显微镜图像分割中的数据稀缺性 

**Authors**: Nikola L. Kolev, Max Trouton, Filippo Federici Canova, Geoff Thornton, David Z. Gao, Neil J. Curson, Taylor J. Z. Stock  

**Link**: [PDF](https://arxiv.org/pdf/2506.01678)  

**Abstract**: Scanning tunnelling microscopy (STM) is a powerful technique for imaging surfaces with atomic resolution, providing insight into physical and chemical processes at the level of single atoms and molecules. A regular task of STM image analysis is the identification and labelling of features of interest against a uniform background. Performing this manually is a labour-intensive task, requiring significant human effort. To reduce this burden, we propose an automated approach to the segmentation of STM images that uses both few-shot learning and unsupervised learning. Our technique offers greater flexibility compared to previous supervised methods; it removes the requirement for large manually annotated datasets and is thus easier to adapt to an unseen surface while still maintaining a high accuracy. We demonstrate the effectiveness of our approach by using it to recognise atomic features on three distinct surfaces: Si(001), Ge(001), and TiO$_2$(110), including adsorbed AsH$_3$ molecules on the silicon and germanium surfaces. Our model exhibits strong generalisation capabilities, and following initial training, can be adapted to unseen surfaces with as few as one additional labelled data point. This work is a significant step towards efficient and material-agnostic, automatic segmentation of STM images. 

**Abstract (ZH)**: 扫描隧道显微镜（STM）技术是一种具备原子级分辨率的图像技术，能够提供表面的原子和分子层面的物理和化学过程见解。STM图像分析的一个常规任务是对比均匀背景识别和标注感兴趣特征。手动执行这一任务需要大量的劳动和人力。为了减轻这一负担，我们提出了一种结合少量样本学习和无监督学习的自动化图像分割方法。与之前的监督方法相比，我们的技术更具灵活性；它消除了大量手动标注数据集的需要，因此在适应未见过的表面时更容易，并且仍然保持高精度。我们通过在三种不同的表面上识别原子特征来证明我们方法的有效性：Si(001)、Ge(001) 和 TiO$_2$(110)，包括硅和锗表面吸附的AsH$_3$分子。我们的模型表现出强大的泛化能力，在初始训练后，只需一个额外的标注数据点即可适应未见过的表面。这项工作是朝着高效且材料无关的STM图像自动分割方向迈进的重要一步。 

---
# GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion 

**Title (ZH)**: Gram: 基于语义aware多粒度晚融合的生成推荐 

**Authors**: Sunkyung Lee, Minjin Choi, Eunseong Choi, Hye-young Kim, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01673)  

**Abstract**: Generative recommendation is an emerging paradigm that leverages the extensive knowledge of large language models by formulating recommendations into a text-to-text generation task. However, existing studies face two key limitations in (i) incorporating implicit item relationships and (ii) utilizing rich yet lengthy item information. To address these challenges, we propose a Generative Recommender via semantic-Aware Multi-granular late fusion (GRAM), introducing two synergistic innovations. First, we design semantic-to-lexical translation to encode implicit hierarchical and collaborative item relationships into the vocabulary space of LLMs. Second, we present multi-granular late fusion to integrate rich semantics efficiently with minimal information loss. It employs separate encoders for multi-granular prompts, delaying the fusion until the decoding stage. Experiments on four benchmark datasets show that GRAM outperforms eight state-of-the-art generative recommendation models, achieving significant improvements of 11.5-16.0% in Recall@5 and 5.3-13.6% in NDCG@5. The source code is available at this https URL. 

**Abstract (ZH)**: 基于语义意识多层次晚期融合的生成推荐 

---
# Synthesis of discrete-continuous quantum circuits with multimodal diffusion models 

**Title (ZH)**: 多模扩散模型合成离散-连续量子电路 

**Authors**: Florian Fürrutter, Zohim Chandani, Ikko Hamamura, Hans J. Briegel, Gorka Muñoz-Gil  

**Link**: [PDF](https://arxiv.org/pdf/2506.01666)  

**Abstract**: Efficiently compiling quantum operations remains a major bottleneck in scaling quantum computing. Today's state-of-the-art methods achieve low compilation error by combining search algorithms with gradient-based parameter optimization, but they incur long runtimes and require multiple calls to quantum hardware or expensive classical simulations, making their scaling prohibitive. Recently, machine-learning models have emerged as an alternative, though they are currently restricted to discrete gate sets. Here, we introduce a multimodal denoising diffusion model that simultaneously generates a circuit's structure and its continuous parameters for compiling a target unitary. It leverages two independent diffusion processes, one for discrete gate selection and one for parameter prediction. We benchmark the model over different experiments, analyzing the method's accuracy across varying qubit counts, circuit depths, and proportions of parameterized gates. Finally, by exploiting its rapid circuit generation, we create large datasets of circuits for particular operations and use these to extract valuable heuristics that can help us discover new insights into quantum circuit synthesis. 

**Abstract (ZH)**: 高效编译量子操作仍然是扩展量子计算的主要瓶颈。当前最先进的方法通过将搜索算法与基于梯度的参数优化结合来实现低编译误差，但这些方法存在较长的运行时间和需要多次调用量子硬件或昂贵的经典模拟，从而使其实现扩展变得不可行。最近，机器学习模型作为替代方法出现，尽管它们目前仅限于离散门集。在这里，我们介绍了一种多模态去噪扩散模型，该模型同时生成目标酉矩阵的电路结构及其连续参数。该模型利用了两个独立的扩散过程，一个用于离散门的选择，另一个用于参数预测。我们在不同的实验中对标记模型进行了基准测试，分析了该方法在不同量子比特数量、电路深度和参数化门比例下的准确性。最后，通过利用其快速电路生成能力，我们创建了特定操作的大型电路数据集，并使用这些数据集提取有价值的启发式方法，以帮助我们发现量子电路合成的新见解。 

---
# Provably Safe Reinforcement Learning from Analytic Gradients 

**Title (ZH)**: 可验证安全的强化学习：基于分析梯度的方法 

**Authors**: Tim Walter, Hannah Markgraf, Jonathan Külz, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.01665)  

**Abstract**: Deploying autonomous robots in safety-critical applications requires safety guarantees. Provably safe reinforcement learning is an active field of research which aims to provide such guarantees using safeguards. These safeguards should be integrated during training to prevent a large sim-to-real gap. While there are several approaches for safeguarding sampling-based reinforcement learning, analytic gradient-based reinforcement learning often achieves superior performance and sample efficiency. However, there is no safeguarding approach for this learning paradigm yet. Our work addresses this gap by developing the first effective safeguard for analytic gradient-based reinforcement learning. We analyse existing, differentiable safeguards, adapt them through modified mappings and gradient formulations, and integrate them with a state-of-the-art learning algorithm and a differentiable simulation. We evaluate how different safeguards affect policy optimisation using numerical experiments on two classical control tasks. The results demonstrate safeguarded training without compromising performance. 

**Abstract (ZH)**: 在安全性关键应用中部署自主机器人需要安全性保证。可证明安全的强化学习是研究的一个活跃领域，旨在通过安全机制提供此类保证。这些安全机制应在训练期间集成以防止仿真到现实世界的差距。尽管已有几种针对基于采样强化学习的保护方法，但分析梯度基强化学习通常能实现更好的性能和样本效率。然而，目前尚无针对这种学习范式的保护方法。我们通过开发首个有效的分析梯度基强化学习保护方法来填补这一空白。我们分析现有可微分保护方法，通过修改映射和梯度公式对它们进行调整，并将它们与最先进的学习算法和可微分模拟集成。我们通过在两个经典控制任务上的数值实验评估不同保护方法对策略优化的影响。结果表明，在不牺牲性能的情况下实现了受保护的训练。 

---
# Explainable AI Systems Must Be Contestable: Here's How to Make It Happen 

**Title (ZH)**: 可解释的人工智能系统必须是可争议的：实现这一目标的方法 

**Authors**: Catarina Moreira, Anna Palatkina, Dacia Braca, Dylan M. Walsh, Peter J. Leihn, Fang Chen, Nina C. Hubig  

**Link**: [PDF](https://arxiv.org/pdf/2506.01662)  

**Abstract**: As AI regulations around the world intensify their focus on system safety, contestability has become a mandatory, yet ill-defined, safeguard. In XAI, "contestability" remains an empty promise: no formal definition exists, no algorithm guarantees it, and practitioners lack concrete guidance to satisfy regulatory requirements. Grounded in a systematic literature review, this paper presents the first rigorous formal definition of contestability in explainable AI, directly aligned with stakeholder requirements and regulatory mandates. We introduce a modular framework of by-design and post-hoc mechanisms spanning human-centered interfaces, technical architectures, legal processes, and organizational workflows. To operationalize our framework, we propose the Contestability Assessment Scale, a composite metric built on more than twenty quantitative criteria. Through multiple case studies across diverse application domains, we reveal where state-of-the-art systems fall short and show how our framework drives targeted improvements. By converting contestability from regulatory theory into a practical framework, our work equips practitioners with the tools to embed genuine recourse and accountability into AI systems. 

**Abstract (ZH)**: 随着全球对AI系统安全性的关注加剧，“可争议性”已成为一项必要的但尚未明确的规定。在XAI领域，“可争议性”仍是一个空洞的承诺：缺乏正式定义，没有算法能够保证这一点，实践者也缺乏具体的指导以满足监管要求。基于系统的文献综述，本文首次提出了 Explainable AI 中“可争议性”的严谨正式定义，该定义直接与利益相关方需求和监管要求相一致。我们引入了一个模块化框架，涵盖了人机接口、技术架构、法律流程和组织工作流程中的设计时和事后机制。为了实现该框架的实用化，我们提出了“可争议性评估量表”，这是一个基于逾二十个定量标准的综合指标。通过涵盖不同应用领域的多个案例研究，我们揭示了现有先进系统的不足之处，并展示了如何通过该框架实现有针对性的改进。通过将“可争议性”从监管理论转化为实用框架，我们的工作为实践者提供了嵌入真正救济和问责制的工具。 

---
# Engram Memory Encoding and Retrieval: A Neurocomputational Perspective 

**Title (ZH)**: 记忆回放的编码与检索：神经计算视角 

**Authors**: Daniel Szelogowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.01659)  

**Abstract**: Despite substantial research into the biological basis of memory, the precise mechanisms by which experiences are encoded, stored, and retrieved in the brain remain incompletely understood. A growing body of evidence supports the engram theory, which posits that sparse populations of neurons undergo lasting physical and biochemical changes to support long-term memory. Yet, a comprehensive computational framework that integrates biological findings with mechanistic models remains elusive. This work synthesizes insights from cellular neuroscience and computational modeling to address key challenges in engram research: how engram neurons are identified and manipulated; how synaptic plasticity mechanisms contribute to stable memory traces; and how sparsity promotes efficient, interference-resistant representations. Relevant computational approaches -- such as sparse regularization, engram gating, and biologically inspired architectures like Sparse Distributed Memory and spiking neural networks -- are also examined. Together, these findings suggest that memory efficiency, capacity, and stability emerge from the interaction of plasticity and sparsity constraints. By integrating neurobiological and computational perspectives, this paper provides a comprehensive theoretical foundation for engram research and proposes a roadmap for future inquiry into the mechanisms underlying memory, with implications for the diagnosis and treatment of memory-related disorders. 

**Abstract (ZH)**: 尽管对记忆的生物学基础进行了大量研究，但大脑中经验是如何被编码、存储和检索的精确机制仍然知之甚少。越来越多的证据支持基因组理论，即稀疏的神经元群体经历持久的物理和生化变化以支持长期记忆。然而，将生物学发现与机制模型综合的全面计算框架仍然难以捉摸。本文综合细胞神经科学和计算建模的见解，解决了基因组研究中的关键挑战：如何识别和操纵基因组神经元；突触可塑性机制如何贡献稳定的记忆痕迹；以及稀疏性如何促进高效、抗干扰的表现。相关的计算方法，如稀疏正则化、基因组门控、以及受生物启发的结构如稀疏分布式记忆和突触神经网络，也进行了探讨。这些发现表明，记忆效率、容量和稳定性源自可塑性和稀疏性约束的相互作用。通过整合神经生物学和计算视角，本文为基因组研究提供了全面的理论基础，并提出了对未来研究记忆机制的路线图，对于记忆相关疾病的诊断和治疗具有重要意义。 

---
# ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge 

**Title (ZH)**: ESGenius：评估大语言模型在环境、社会与治理（ESG）及可持续性知识方面的表现 

**Authors**: Chaoyue He, Xin Zhou, Yi Wu, Xinjia Yu, Yan Zhang, Lei Zhang, Di Wang, Shengfei Lyu, Hong Xu, Xiaoqiao Wang, Wei Liu, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01646)  

**Abstract**: We introduce ESGenius, a comprehensive benchmark for evaluating and enhancing the proficiency of Large Language Models (LLMs) in Environmental, Social and Governance (ESG) and sustainability-focused question answering. ESGenius comprises two key components: (i) ESGenius-QA, a collection of 1 136 multiple-choice questions generated by LLMs and rigorously validated by domain experts, covering a broad range of ESG pillars and sustainability topics. Each question is systematically linked to its corresponding source text, enabling transparent evaluation and supporting retrieval-augmented generation (RAG) methods; and (ii) ESGenius-Corpus, a meticulously curated repository of 231 foundational frameworks, standards, reports and recommendation documents from seven authoritative sources. Moreover, to fully assess the capabilities and adaptation potential of the model, we implement a rigorous two-stage evaluation protocol -- Zero-Shot and RAG. Extensive experiments across 50 LLMs (ranging from 0.5 B to 671 B parameters) demonstrate that state-of-the-art models achieve only moderate performance in zero-shot settings, with accuracies typically around 55--70\%, highlighting ESGenius's challenging nature for LLMs in interdisciplinary contexts. However, models employing RAG show significant performance improvements, particularly for smaller models. For example, "DeepSeek-R1-Distill-Qwen-14B" improves from 63.82\% (zero-shot) to 80.46\% with RAG. These results underscore the necessity of grounding responses in authoritative sources for enhanced ESG understanding. To the best of our knowledge, ESGenius is the first benchmark curated for LLMs and the relevant enhancement technologies that focuses on ESG and sustainability topics. 

**Abstract (ZH)**: ESGenius：用于评估和提高大型语言模型在环境、社会与治理（ESG）和可持续性问题回答能力的全面基准 

---
# Bidirectional Soft Actor-Critic: Leveraging Forward and Reverse KL Divergence for Efficient Reinforcement Learning 

**Title (ZH)**: 双向软actor- critic：利用前向和后向KL散度进行高效的强化学习 

**Authors**: Yixian Zhang, Huaze Tang, Changxu Wei, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.01639)  

**Abstract**: The Soft Actor-Critic (SAC) algorithm, a state-of-the-art method in maximum entropy reinforcement learning, traditionally relies on minimizing reverse Kullback-Leibler (KL) divergence for policy updates. However, this approach leads to an intractable optimal projection policy, necessitating gradient-based approximations that can suffer from instability and poor sample efficiency. This paper investigates the alternative use of forward KL divergence within SAC. We demonstrate that for Gaussian policies, forward KL divergence yields an explicit optimal projection policy -- corresponding to the mean and variance of the target Boltzmann distribution's action marginals. Building on the distinct advantages of both KL directions, we propose Bidirectional SAC, an algorithm that first initializes the policy using the explicit forward KL projection and then refines it by optimizing the reverse KL divergence. Comprehensive experiments on continuous control benchmarks show that Bidirectional SAC significantly outperforms standard SAC and other baselines, achieving up to a $30\%$ increase in episodic rewards, alongside enhanced sample efficiency. 

**Abstract (ZH)**: 双向软 actor-critic (Bidirectional SAC): 结合前后向 KL 散度的优势 

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
# Robust Satisficing Gaussian Process Bandits Under Adversarial Attacks 

**Title (ZH)**: 鲁棒 satisficing 高斯过程多臂赌博机算法在对抗攻击下的研究 

**Authors**: Artun Saday, Yaşar Cahit Yıldırım, Cem Tekin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01625)  

**Abstract**: We address the problem of Gaussian Process (GP) optimization in the presence of unknown and potentially varying adversarial perturbations. Unlike traditional robust optimization approaches that focus on maximizing performance under worst-case scenarios, we consider a robust satisficing objective, where the goal is to consistently achieve a predefined performance threshold $\tau$, even under adversarial conditions. We propose two novel algorithms based on distinct formulations of robust satisficing, and show that they are instances of a general robust satisficing framework. Further, each algorithm offers different guarantees depending on the nature of the adversary. Specifically, we derive two regret bounds: one that is sublinear over time, assuming certain conditions on the adversary and the satisficing threshold $\tau$, and another that scales with the perturbation magnitude but requires no assumptions on the adversary. Through extensive experiments, we demonstrate that our approach outperforms the established robust optimization methods in achieving the satisficing objective, particularly when the ambiguity set of the robust optimization framework is inaccurately specified. 

**Abstract (ZH)**: 我们在未知且可能变化的对抗扰动下解决高斯过程（GP）优化问题。不同于传统的稳健优化方法侧重于在最坏情况下最大化性能，我们考虑一种稳健满足目标，即在对抗条件下一致地达到预定义的性能门槛 $\tau$。我们提出两种新型算法，基于不同的稳健满足形式，并表明它们是通用的稳健满足框架的实例。此外，每种算法根据对手的性质提供不同的保证。具体地，我们推导出两个遗憾界：一个在满足一定对手和满足阈值 $\tau$ 的条件下是亚线性的，另一个与扰动幅度成比例但对对手没有假设。通过广泛的实验，我们证明我们的方法在实现满足目标方面优于现有的稳健优化方法，特别是在稳健优化框架的不确定性集不准确指定时。 

---
# Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech 

**Title (ZH)**: 无监督节奏和语音转换以改善构音障碍语音的ASR性能 

**Authors**: Karl El Hajal, Enno Hermann, Sevada Hovsepyan, Mathew Magimai.-Doss  

**Link**: [PDF](https://arxiv.org/pdf/2506.01618)  

**Abstract**: Automatic speech recognition (ASR) systems struggle with dysarthric speech due to high inter-speaker variability and slow speaking rates. To address this, we explore dysarthric-to-healthy speech conversion for improved ASR performance. Our approach extends the Rhythm and Voice (RnV) conversion framework by introducing a syllable-based rhythm modeling method suited for dysarthric speech. We assess its impact on ASR by training LF-MMI models and fine-tuning Whisper on converted speech. Experiments on the Torgo corpus reveal that LF-MMI achieves significant word error rate reductions, especially for more severe cases of dysarthria, while fine-tuning Whisper on converted data has minimal effect on its performance. These results highlight the potential of unsupervised rhythm and voice conversion for dysarthric ASR. Code available at: this https URL 

**Abstract (ZH)**: 自动语音识别(ASR)系统在处理构音障碍 speech 的时候因说话人变异性高和说话速度慢而遇到困难。为了解决这个问题，我们探索了构音障碍到健康语音的转换，以提高ASR性能。我们的方法扩展了节奏和语音(RnV)转换框架，引入了一种基于音节的节奏建模方法，适用于构音障碍语音。我们通过训练LF-MMI模型并在转换后的语音上微调Whisper来评估其对ASR的影响。对Torgo语料库的实验表明，LF-MMI在词错误率方面取得了显著减少，尤其是在构音障碍更严重的情况下，而对转换数据进行Whisper的微调对其性能的影响很小。这些结果突显了无监督节奏和语音转换对构音障碍ASR的潜在价值。代码可在以下链接获得：这个 https URL。 

---
# Contrastive Learning for Efficient Transaction Validation in UTXO-based Blockchains 

**Title (ZH)**: 基于UTXO模型区块链中高效交易验证的对比学习 

**Authors**: Hamid Attar, Luigi Lunardon, Alessio Pagani  

**Link**: [PDF](https://arxiv.org/pdf/2506.01614)  

**Abstract**: This paper introduces a Machine Learning (ML) approach for scalability of UTXO-based blockchains, such as Bitcoin. Prior approaches to UTXO set sharding struggle with distributing UTXOs effectively across validators, creating substantial communication overhead due to child-parent transaction dependencies. This overhead, which arises from the need to locate parent UTXOs, significantly hampers transaction processing speeds. Our solution uses ML to optimize not only UTXO set sharding but also the routing of incoming transactions, ensuring that transactions are directed to shards containing their parent UTXOs. At the heart of our approach is a framework that combines contrastive and unsupervised learning to create an embedding space for transaction outputs. This embedding allows the model to group transaction outputs based on spending relationships, making it possible to route transactions efficiently to the correct validation microservices. Trained on historical transaction data with triplet loss and online semi-hard negative mining, the model embeds parent-child spending patterns directly into its parameters, thus eliminating the need for costly, real-time parent transaction lookups. This significantly reduces cross-shard communication overhead, boosting throughput and scalability. 

**Abstract (ZH)**: 基于机器学习的UTXO区块链可扩展性研究：结合对比学习和无监督学习的交易输出嵌入框架 

---
# EPFL-Smart-Kitchen-30: Densely annotated cooking dataset with 3D kinematics to challenge video and language models 

**Title (ZH)**: EPFL-智能厨房30：包含3D运动学密集标注的烹饪数据集，挑战视频和语言模型 

**Authors**: Andy Bonnetto, Haozhe Qi, Franklin Leong, Matea Tashkovska, Mahdi Rad, Solaiman Shokur, Friedhelm Hummel, Silvestro Micera, Marc Pollefeys, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2506.01608)  

**Abstract**: Understanding behavior requires datasets that capture humans while carrying out complex tasks. The kitchen is an excellent environment for assessing human motor and cognitive function, as many complex actions are naturally exhibited in kitchens from chopping to cleaning. Here, we introduce the EPFL-Smart-Kitchen-30 dataset, collected in a noninvasive motion capture platform inside a kitchen environment. Nine static RGB-D cameras, inertial measurement units (IMUs) and one head-mounted HoloLens~2 headset were used to capture 3D hand, body, and eye movements. The EPFL-Smart-Kitchen-30 dataset is a multi-view action dataset with synchronized exocentric, egocentric, depth, IMUs, eye gaze, body and hand kinematics spanning 29.7 hours of 16 subjects cooking four different recipes. Action sequences were densely annotated with 33.78 action segments per minute. Leveraging this multi-modal dataset, we propose four benchmarks to advance behavior understanding and modeling through 1) a vision-language benchmark, 2) a semantic text-to-motion generation benchmark, 3) a multi-modal action recognition benchmark, 4) a pose-based action segmentation benchmark. We expect the EPFL-Smart-Kitchen-30 dataset to pave the way for better methods as well as insights to understand the nature of ecologically-valid human behavior. Code and data are available at this https URL 

**Abstract (ZH)**: 理解行为需要捕捉执行复杂任务的人类的数据集。厨房是评估人类运动和认知功能的理想环境，因为在厨房中从切菜到清洁等许多复杂动作自然是发生的。在这里，我们介绍了在厨房环境中收集的EPFL-Smart-Kitchen-30数据集，使用非侵入式动作捕捉平台。九个静态RGB-D摄像头、惯性测量单元（IMUs）和一个头戴式HoloLens 2头显被用于捕捉3D手部、身体和眼部运动。EPFL-Smart-Kitchen-30数据集是一个包含同步外部视角、第一人称视角、深度、IMUs、眼动、身体和手部运动学信息的多种视角动作数据集，覆盖了16名受试者烹饪四种不同食谱的29.7小时。每分钟有33.78个动作片段的密集标注。利用这一多模态数据集，我们提出四个基准来通过1）视觉语言基准，2）语义文本到动作生成基准，3）多模态动作识别基准，4）基于姿态的动作分割基准，推进行为理解和建模。我们期望EPFL-Smart-Kitchen-30数据集为更好的方法以及对生态有效的人类行为本质的理解开辟道路。代码和数据可在以下链接获取。 

---
# WoMAP: World Models For Embodied Open-Vocabulary Object Localization 

**Title (ZH)**: WoMAP：世界模型在具身开放词汇对象定位中的应用 

**Authors**: Tenny Yin, Zhiting Mei, Tao Sun, Lihan Zha, Emily Zhou, Jeremy Bao, Miyu Yamane, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.01600)  

**Abstract**: Language-instructed active object localization is a critical challenge for robots, requiring efficient exploration of partially observable environments. However, state-of-the-art approaches either struggle to generalize beyond demonstration datasets (e.g., imitation learning methods) or fail to generate physically grounded actions (e.g., VLMs). To address these limitations, we introduce WoMAP (World Models for Active Perception): a recipe for training open-vocabulary object localization policies that: (i) uses a Gaussian Splatting-based real-to-sim-to-real pipeline for scalable data generation without the need for expert demonstrations, (ii) distills dense rewards signals from open-vocabulary object detectors, and (iii) leverages a latent world model for dynamics and rewards prediction to ground high-level action proposals at inference time. Rigorous simulation and hardware experiments demonstrate WoMAP's superior performance in a broad range of zero-shot object localization tasks, with more than 9x and 2x higher success rates compared to VLM and diffusion policy baselines, respectively. Further, we show that WoMAP achieves strong generalization and sim-to-real transfer on a TidyBot. 

**Abstract (ZH)**: 基于语言指示的主动物体定位是机器人面临的關鍵挑戰，需要高效探索部分可观测环境。然而，最先进的方法要么难以泛化到演示数据集之外（例如，模仿学习方法），要么生成不出物理上合理的动作（例如，VLMs）。为此，我们提出了WoMAP（World Models for Active Perception）：一种用于训练开放词汇物体定位策略的方法，包括：(i) 使用基于高斯点积的实时到模拟再到实时的数据生成管道，无需专家演示；(ii) 从开放词汇物体检测器中提取密集奖励信号；(iii) 利用潜在世界模型进行动力学和奖励预测，在推理时使高层动作提案具有物理意义。严格的仿真和硬件实验表明，WoMAP在多种零样本物体定位任务中的性能优于VLM和扩散策略基线，成功率分别高出9倍和2倍。此外，我们展示了WoMAP在TidyBot上实现了强大的泛化能力和模拟到现实的迁移。 

---
# Policy Newton Algorithm in Reproducing Kernel Hilbert Space 

**Title (ZH)**: 政策牛顿算法在再生核希尔伯特空间中 

**Authors**: Yixian Zhang, Huaze Tang, Chao Wang, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.01597)  

**Abstract**: Reinforcement learning (RL) policies represented in Reproducing Kernel Hilbert Spaces (RKHS) offer powerful representational capabilities. While second-order optimization methods like Newton's method demonstrate faster convergence than first-order approaches, current RKHS-based policy optimization remains constrained to first-order techniques. This limitation stems primarily from the intractability of explicitly computing and inverting the infinite-dimensional Hessian operator in RKHS. We introduce Policy Newton in RKHS, the first second-order optimization framework specifically designed for RL policies represented in RKHS. Our approach circumvents direct computation of the inverse Hessian operator by optimizing a cubic regularized auxiliary objective function. Crucially, we leverage the Representer Theorem to transform this infinite-dimensional optimization into an equivalent, computationally tractable finite-dimensional problem whose dimensionality scales with the trajectory data volume. We establish theoretical guarantees proving convergence to a local optimum with a local quadratic convergence rate. Empirical evaluations on a toy financial asset allocation problem validate these theoretical properties, while experiments on standard RL benchmarks demonstrate that Policy Newton in RKHS achieves superior convergence speed and higher episodic rewards compared to established first-order RKHS approaches and parametric second-order methods. Our work bridges a critical gap between non-parametric policy representations and second-order optimization methods in reinforcement learning. 

**Abstract (ZH)**: RKHS中表示的RL策略的Reinforcement Learning新方法：第二-order优化框架 

---
# Understanding and Improving Laplacian Positional Encodings For Temporal GNNs 

**Title (ZH)**: 理解并改进拉普拉斯位置编码以优化时间动态图神经网络 

**Authors**: Yaniv Galron, Fabrizio Frasca, Haggai Maron, Eran Treister, Moshe Eliasof  

**Link**: [PDF](https://arxiv.org/pdf/2506.01596)  

**Abstract**: Temporal graph learning has applications in recommendation systems, traffic forecasting, and social network analysis. Although multiple architectures have been introduced, progress in positional encoding for temporal graphs remains limited. Extending static Laplacian eigenvector approaches to temporal graphs through the supra-Laplacian has shown promise, but also poses key challenges: high eigendecomposition costs, limited theoretical understanding, and ambiguity about when and how to apply these encodings. In this paper, we address these issues by (1) offering a theoretical framework that connects supra-Laplacian encodings to per-time-slice encodings, highlighting the benefits of leveraging additional temporal connectivity, (2) introducing novel methods to reduce the computational overhead, achieving up to 56x faster runtimes while scaling to graphs with 50,000 active nodes, and (3) conducting an extensive experimental study to identify which models, tasks, and datasets benefit most from these encodings. Our findings reveal that while positional encodings can significantly boost performance in certain scenarios, their effectiveness varies across different models. 

**Abstract (ZH)**: 时空图学习在推荐系统、交通预测和社会网络分析中有应用。虽然已引入多种架构，但时空图的位置编码进展有限。通过超拉普拉斯扩展静态拉普拉斯特征向量方法显示出潜力，但也提出了关键挑战：高昂的特征分解成本、有限的理论理解以及何时以及如何应用这些编码的模糊性。本文通过（1）提供一个理论框架，将超拉普拉斯编码与每时间片编码连接起来，强调利用额外的时间连接性的好处；（2）引入新型方法减少计算开销，实现高达56倍更快的运行时间并在包含50,000个活跃节点的图上进行扩展；（3）进行广泛实验研究以确定哪些模型、任务和数据集最受益于这些编码。我们的研究发现表明，虽然位置编码在某些场景中能显著提升性能，但其有效性因不同的模型而异。时空图学习在推荐系统、交通预测和社会网络分析中有应用。虽然已引入多种架构，但时空图的位置编码进展有限。通过超拉普拉斯扩展静态拉普拉斯特征向量方法显示出潜力，但也提出了关键挑战：高昂的特征分解成本、有限的理论理解以及何时以及如何应用这些编码的模糊性。本文通过提供一个理论框架、引入新型方法减少计算开销以及进行广泛实验研究来解决这些问题。 

---
# VirnyFlow: A Design Space for Responsible Model Development 

**Title (ZH)**: VirnyFlow: 负责任模型开发的设计空间 

**Authors**: Denys Herasymuk, Nazar Protsiv, Julia Stoyanovich  

**Link**: [PDF](https://arxiv.org/pdf/2506.01584)  

**Abstract**: Developing machine learning (ML) models requires a deep understanding of real-world problems, which are inherently multi-objective. In this paper, we present VirnyFlow, the first design space for responsible model development, designed to assist data scientists in building ML pipelines that are tailored to the specific context of their problem. Unlike conventional AutoML frameworks, VirnyFlow enables users to define customized optimization criteria, perform comprehensive experimentation across pipeline stages, and iteratively refine models in alignment with real-world constraints. Our system integrates evaluation protocol definition, multi-objective Bayesian optimization, cost-aware multi-armed bandits, query optimization, and distributed parallelism into a unified architecture. We show that VirnyFlow significantly outperforms state-of-the-art AutoML systems in both optimization quality and scalability across five real-world benchmarks, offering a flexible, efficient, and responsible alternative to black-box automation in ML development. 

**Abstract (ZH)**: 开发机器学习模型需要深刻理解现实世界的问题，这些问题往往是多目标的。本文介绍了VirnyFlow，这是首个负责任模型开发的设计空间，旨在协助数据科学家构建适应其具体问题背景的ML管道。与传统的AutoML框架不同，VirnyFlow允许用户定义自定义的优化标准，在管道各个阶段进行全面实验，并在符合现实世界约束的情况下逐步优化模型。该系统将评估协议定义、多目标贝叶斯优化、成本感知多臂老虎机、查询优化和分布式并行性整合到统一架构中。我们展示了VirnyFlow在五个真实基准上的优化质量和可扩展性都显著优于最先进的AutoML系统，提供了灵活、高效且负责任的黑盒自动化替代方案。 

---
# FreqPolicy: Frequency Autoregressive Visuomotor Policy with Continuous Tokens 

**Title (ZH)**: FreqPolicy: 频率自回归visuomotor策略与连续_token_表示 

**Authors**: Yiming Zhong, Yumeng Liu, Chuyang Xiao, Zemin Yang, Youzhuo Wang, Yufei Zhu, Ye Shi, Yujing Sun, Xinge Zhu, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.01583)  

**Abstract**: Learning effective visuomotor policies for robotic manipulation is challenging, as it requires generating precise actions while maintaining computational efficiency. Existing methods remain unsatisfactory due to inherent limitations in the essential action representation and the basic network architectures. We observe that representing actions in the frequency domain captures the structured nature of motion more effectively: low-frequency components reflect global movement patterns, while high-frequency components encode fine local details. Additionally, robotic manipulation tasks of varying complexity demand different levels of modeling precision across these frequency bands. Motivated by this, we propose a novel paradigm for visuomotor policy learning that progressively models hierarchical frequency components. To further enhance precision, we introduce continuous latent representations that maintain smoothness and continuity in the action space. Extensive experiments across diverse 2D and 3D robotic manipulation benchmarks demonstrate that our approach outperforms existing methods in both accuracy and efficiency, showcasing the potential of a frequency-domain autoregressive framework with continuous tokens for generalized robotic manipulation. 

**Abstract (ZH)**: 学习有效的视觉运动策略以进行机器人操作具有挑战性，因为它要求在保持计算效率的同时生成精确的动作。现有方法由于在基本动作表示和基础网络架构方面的固有限制，仍未达到满意的效果。我们观察到，在频域中表示动作能更有效地捕捉运动的结构特征：低频分量反映全局运动模式，而高频分量编码细微的局部细节。此外，不同复杂度的机器人操作任务要求在这些频带中建模不同的精度水平。受此启发，我们提出了一种新颖的视觉运动策略学习范式，逐步建模分层的频域成分。为进一步提高精度，我们引入了连续的潜在表示，以在动作空间中保持平滑性和连续性。在多种2D和3D机器人操作基准上的广泛实验表明，我们的方法在准确性和效率上均优于现有方法，展示了频域自回归框架与连续令牌在通用机器人操作中的潜力。 

---
# Advanced Nanostructured Topical Therapeutics for Psoriasis: Strategic Synthesis, Multimodal Characterization, and Preliminary Pharmacodynamic Profiling 

**Title (ZH)**: 纳米结构皮肤治疗药物在银屑病中的高级应用：策略合成、多模态表征及初步药效学分析 

**Authors**: Iqra Yousaf, Aqsa Yousaf  

**Link**: [PDF](https://arxiv.org/pdf/2506.01572)  

**Abstract**: Psoriasis is a long-term inflammatory skin disease that remains difficult to treat. In this study, we developed a new topical treatment by combining metal oxide nanoparticles: cerium oxide (CeO2), zinc oxide (ZnO), and silver (Ag), with natural plant extracts in a gel made from fish collagen and agar. The nanoparticles were characterized using UV-Vis spectroscopy, dynamic light scattering (DLS), Fourier-transform infrared spectroscopy (FTIR), and scanning electron microscopy (SEM), showing good stability and a uniform particle size distribution (ZnO averaged 66 nm).
To enhance therapeutic potential, the gel was enriched with plant-derived antioxidants from bitter melon, ginger, and neem. This formulation was tested on an animal model of psoriasis. The treated group exhibited faster wound healing and reduced inflammation compared to both placebo and untreated groups, with statistically significant results (p < 0.01 to p < 0.001) observed from Day 3, becoming more pronounced by Day 14.
These results indicate that the combination of nanoparticles with plant-based components in a topical gel may provide a promising new approach to psoriasis treatment. Further studies are recommended to evaluate long-term safety and therapeutic effectiveness. 

**Abstract (ZH)**: 氧化金属纳米粒子与植物提取物联合治疗银屑病的新外用制剂及其疗效研究 

---
# FlexiSAGA: A Flexible Systolic Array GEMM Accelerator for Sparse and Dense Processing 

**Title (ZH)**: FlexiSAGA: 一种灵活的 systolic array GEMM 加速器，适用于稀疏和稠密处理 

**Authors**: Mika Markus Müller, Konstantin Lübeck, Alexander Louis-Ferdinand Jung, Jannik Steinmetz, Oliver Bringmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.01566)  

**Abstract**: Artificial Intelligence (AI) algorithms, such as Deep Neural Networks (DNNs), have become an important tool for a wide range of applications, from computer vision to natural language processing. However, the computational complexity of DNN inference poses a significant challenge, particularly for processing on resource-constrained edge devices. One promising approach to address this challenge is the exploitation of sparsity in DNN operator weights.
In this work, we present FlexiSAGA, an architecturally configurable and dataflow-flexible AI hardware accelerator for the sparse and dense processing of general matrix multiplications (GEMMs). FlexiSAGA supports seven different sparse and dense dataflows, enabling efficient processing of resource intensive DNN operators. Additionally, we propose a DNN pruning method specifically tailored towards the FlexiSAGA architecture, allowing for near-optimal processing of dense and sparse convolution and fully-connected operators, facilitating a DNN/HW co-design flow. Our results show a whole DNN sparse-over-dense inference speedup ranging from 1.41 up to 4.28, outperforming commercial and literature-reported accelerator platforms. 

**Abstract (ZH)**: 人工神经网络（DNNs）等人工 Intelligence (AI) 算法已成为从计算机视觉到自然语言处理等广泛应用的重要工具。然而，DNN 推断的计算复杂性对资源受限的边缘设备构成了重大挑战。一种有前景的应对策略是利用 DNN 运算权重的稀疏性。

在本文中，我们提出了 FlexiSAGA，一种架构可配置且数据流灵活的人工智能硬件加速器，适用于通用矩阵乘法（GEMM）的稀疏和密集处理。FlexiSAGA 支持七种不同的稀疏和密集数据流，能够高效处理资源密集型 DNN 运算。此外，我们提出了一种针对 FlexiSAGA 架构的 DNN 裁剪方法，使得密集和稀疏卷积以及全连接运算的处理接近最优，促进了 DNN/HW 共同设计流程。我们的结果表明，整个 DNN 稀疏-密集推断加速范围从 1.41 到 4.28，优于商业和文献报告的加速器平台。 

---
# EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation 

**Title (ZH)**: EvolveNav：自我提升的嵌入式推理用于基于LLM的视觉语言导航 

**Authors**: Bingqian Lin, Yunshuang Nie, Khun Loun Zai, Ziming Wei, Mingfei Han, Rongtao Xu, Minzhe Niu, Jianhua Han, Liang Lin, Cewu Lu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01551)  

**Abstract**: Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at this https URL. 

**Abstract (ZH)**: 构建能够遵循自然语言指令进行导航的Vision-Language Navigation (VLN) 代理是人机交互应用中的一个长期目标。最近的研究揭示了通过训练开源大型语言模型（LLMs）来释放模型的推理能力，以改进导航并同时减轻LLMs训练语料库与VLN任务之间的领域差距的潜力。然而，这些方法主要采用直接输入-输出映射的范式，导致映射学习困难和导航决策难以解释。链式思考（CoT）训练是一种有望同时提高导航决策准确性和可解释性的方法，但由于导航任务的复杂性，使得完美的CoT标签不可用，且可能导致仅通过纯CoT监督微调产生过拟合。本文提出了一种新的自我改进的嵌入式推理框架，以提升基于LLMs的Vision-Language导航性能，名为EvolveNav。EvolveNav包含两个阶段：（1）形式化的CoT监督微调，我们使用形式化的CoT标签训练模型，以激活模型的导航推理能力并提高推理速度；（2）自我反思的后训练，模型通过迭代使用其自身推理输出作为自我丰富的CoT标签来增强监督多样性。还引入了一个自我反思的辅助任务，通过与错误的推理模式对比来促进正确推理模式的学习。在流行的VLN基准测试上的实验结果证明了EvolveNav相较于之前的基于LLMs的VLN方法的优势。代码可在此处访问。 

---
# G4Seg: Generation for Inexact Segmentation Refinement with Diffusion Models 

**Title (ZH)**: G4Seg: 基于扩散模型的不精确分割精炼生成 

**Authors**: Tianjiao Zhang, Fei Zhang, Jiangchao Yao, Ya Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01539)  

**Abstract**: This paper considers the problem of utilizing a large-scale text-to-image diffusion model to tackle the challenging Inexact Segmentation (IS) task. Unlike traditional approaches that rely heavily on discriminative-model-based paradigms or dense visual representations derived from internal attention mechanisms, our method focuses on the intrinsic generative priors in Stable Diffusion~(SD). Specifically, we exploit the pattern discrepancies between original images and mask-conditional generated images to facilitate a coarse-to-fine segmentation refinement by establishing a semantic correspondence alignment and updating the foreground probability. Comprehensive quantitative and qualitative experiments validate the effectiveness and superiority of our plug-and-play design, underscoring the potential of leveraging generation discrepancies to model dense representations and encouraging further exploration of generative approaches for solving discriminative tasks. 

**Abstract (ZH)**: 本文考虑利用大规模文本到图像扩散模型解决不准确分割（IS）任务的问题。与传统依赖于判别模型或内部注意力机制衍生的密集视觉表示的方法不同，我们的方法集中于稳定的扩散（SD）中的固有生成先验。具体而言，我们通过建立语义对应对齐和更新前景概率，利用原始图像与条件遮罩生成图像之间的模式差异来促进由粗到细的分割细化。全面的定量和定性实验验证了我们即插即用设计的有效性和优越性，强调了利用生成差异建模密集表示的潜力，并促进进一步探索生成方法解决判别任务的可能性。 

---
# LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation 

**Title (ZH)**: LLM辅助的多智能体强化学习在合作策略生成中的应用 

**Authors**: Guobin Zhu, Rui Zhou, Wenkang Ji, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01538)  

**Abstract**: Although Multi-Agent Reinforcement Learning (MARL) is effective for complex multi-robot tasks, it suffers from low sample efficiency and requires iterative manual reward tuning. Large Language Models (LLMs) have shown promise in single-robot settings, but their application in multi-robot systems remains largely unexplored. This paper introduces a novel LLM-Aided MARL (LAMARL) approach, which integrates MARL with LLMs, significantly enhancing sample efficiency without requiring manual design. LAMARL consists of two modules: the first module leverages LLMs to fully automate the generation of prior policy and reward functions. The second module is MARL, which uses the generated functions to guide robot policy training effectively. On a shape assembly benchmark, both simulation and real-world experiments demonstrate the unique advantages of LAMARL. Ablation studies show that the prior policy improves sample efficiency by an average of 185.9% and enhances task completion, while structured prompts based on Chain-of-Thought (CoT) and basic APIs improve LLM output success rates by 28.5%-67.5%. Videos and code are available at this https URL 

**Abstract (ZH)**: 虽然多代理 reinforcement 学习（MARL）在复杂多机器人任务中有效，但其样本效率较低且需要迭代的手动奖励调整。大规模语言模型（LLMs）在单机器人设置中表现出色，但在多机器人系统中的应用尚未得到充分探索。本文提出了一种新颖的 LLM 辅助 MARL（LAMARL）方法，将 MARL 与 LLM 集成，显著提高了样本效率，而无需手动设计。LAMARL 包含两个模块：第一个模块利用 LLM 完全自动生成先验策略和奖励函数。第二个模块是 MARL，它使用生成的函数有效地指导机器人策略训练。在形状装配基准测试中，模拟和实际实验都展示了 LAMARL 的独特优势。消融研究显示，先验策略平均提高样本效率185.9%，并提升任务完成率，基于 Chain-of-Thought（CoT）的结构化提示和基本 API 使 LLM 输出成功率提升了28.5%-67.5%。更多信息参见此网址。 

---
# Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries 

**Title (ZH)**: 词典来帮忙：使用双语词典为低资源语言进行跨语言词汇迁移 

**Authors**: Haruki Sakajo, Yusuke Ide, Justin Vasselli, Yusuke Sakai, Yingtao Tian, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.01535)  

**Abstract**: Cross-lingual vocabulary transfer plays a promising role in adapting pre-trained language models to new languages, including low-resource languages. Existing approaches that utilize monolingual or parallel corpora face challenges when applied to languages with limited resources. In this work, we propose a simple yet effective vocabulary transfer method that utilizes bilingual dictionaries, which are available for many languages, thanks to descriptive linguists. Our proposed method leverages a property of BPE tokenizers where removing a subword from the vocabulary causes a fallback to shorter subwords. The embeddings of target subwords are estimated iteratively by progressively removing them from the tokenizer. The experimental results show that our approach outperforms existing methods for low-resource languages, demonstrating the effectiveness of a dictionary-based approach for cross-lingual vocabulary transfer. 

**Abstract (ZH)**: 跨语言词汇转移在适应预训练语言模型到新语言（包括低资源语言）中发挥着有前途的作用。现有的方法在应用到资源受限的语言时面临挑战。在本工作中，我们提出了一种简单而有效的方法，该方法利用了描述语言学家提供给许多语言的双语词典。我们提出的方法利用了BPE分词器的一个特性，即从词汇表中移除一个子词会导致退回到较短的子词。目标子词的嵌入通过逐步从分词器中移除它们来迭代估计。实验结果表明，我们的方法在低资源语言中优于现有方法，展示了基于词典的方法在跨语言词汇转移中的有效性。 

---
# A Diffusion-Based Method for Learning the Multi-Outcome Distribution of Medical Treatments 

**Title (ZH)**: 基于扩散的方法学习医疗治疗的多结果分布 

**Authors**: Yuchen Ma, Jonas Schweisthal, Hengrui Zhang, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2506.01533)  

**Abstract**: In medicine, treatments often influence multiple, interdependent outcomes, such as primary endpoints, complications, adverse events, or other secondary endpoints. Hence, to make optimal treatment decisions, clinicians are interested in learning the distribution of multi-dimensional treatment outcomes. However, the vast majority of machine learning methods for predicting treatment effects focus on single-outcome settings, despite the fact that medical data often include multiple, interdependent outcomes. To address this limitation, we propose a novel diffusion-based method called DIME to learn the joint distribution of multiple outcomes of medical treatments. We addresses three challenges relevant in medical practice: (i)it is tailored to learn the joint interventional distribution of multiple medical outcomes, which enables reliable decision-making with uncertainty quantification rather than relying solely on point estimates; (ii)it explicitly captures the dependence structure between outcomes; (iii)it can handle outcomes of mixed type, including binary, categorical, and continuous variables. In DIME, we take into account the fundamental problem of causal inference through causal masking. For training, our method decomposes the joint distribution into a series of conditional distributions with a customized conditional masking to account for the dependence structure across outcomes. For inference, our method auto-regressively generates predictions. This allows our method to move beyond point estimates of causal quantities and thus learn the joint interventional distribution. To the best of our knowledge, DIME is the first neural method tailored to learn the joint, multi-outcome distribution of medical treatments. Across various experiments, we demonstrate that our method effectively learns the joint distribution and captures shared information among multiple outcomes. 

**Abstract (ZH)**: 医学中，治疗方法往往影响多个相互依赖的结局，如主要终点、并发症、不良反应或其他次要终点。因此，为了做出最优的治疗决策，临床医生对学习多维度治疗结局的分布感兴趣。然而，大多数用于预测治疗效果的机器学习方法集中在单一结局的设置上，尽管医学数据通常包含多个相互依赖的结局。为了解决这一局限性，我们提出了一种基于扩散的新方法DIME，用于学习医疗治疗多结局的联合分布。DIME解决了医学实践中相关的三个挑战：(i) 它专门设计用于学习多个医疗结局的联合干预分布，从而在不确定性量化的基础上实现可靠的决策，而不仅仅是依赖点估计；(ii) 它明确捕捉了结局之间的依赖结构；(iii) 它可以处理包括二元、分类和连续变量在内的混合类型结局。在DIME中，我们通过因果掩码考虑了因果推理的基本问题。在训练过程中，我们的方法将联合分布分解为一系列条件分布，并通过自定义条件掩码来考虑结局之间的依赖结构。在推理过程中，我们的方法自回归生成预测。这使得我们的方法能够超越因果量的点估计，并因此学习到联合干预分布。据我们所知，DIME是第一个专门设计用于学习医疗治疗多结局联合分布的神经网络方法。在各种实验中，我们证明了该方法有效学习了联合分布，并捕捉了多个结局之间的共享信息。 

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
# Learning of Population Dynamics: Inverse Optimization Meets JKO Scheme 

**Title (ZH)**: 基于逆优化的群体动力学习：JKO方案的应用 

**Authors**: Mikhail Persiianov, Jiawei Chen, Petr Mokrov, Alexander Tyurin, Evgeny Burnaev, Alexander Korotin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01502)  

**Abstract**: Learning population dynamics involves recovering the underlying process that governs particle evolution, given evolutionary snapshots of samples at discrete time points. Recent methods frame this as an energy minimization problem in probability space and leverage the celebrated JKO scheme for efficient time discretization. In this work, we introduce $\texttt{iJKOnet}$, an approach that combines the JKO framework with inverse optimization techniques to learn population dynamics. Our method relies on a conventional $\textit{end-to-end}$ adversarial training procedure and does not require restrictive architectural choices, e.g., input-convex neural networks. We establish theoretical guarantees for our methodology and demonstrate improved performance over prior JKO-based methods. 

**Abstract (ZH)**: 学习种群动态涉及在给定离散时间点的演化快照情况下，恢复调控粒子演化的底层过程。 recent methods 将这一问题框架化为概率空间中的能量最小化问题，并利用著名的 JKO 方案进行高效的时域离散化。在本文中，我们引入了 $\texttt{iJKOnet}$ 方法，该方法将 JKO 框架与逆优化技术相结合以学习种群动态。我们的方法依赖于标准的端到端.adversarial 训练过程，且不需要限制性的架构选择，例如输入凸神经网络。我们为我们的方法建立了理论保证，并展示了与先前的基于 JKO 的方法相比的优越性能。 

---
# Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task? 

**Title (ZH)**: 自动舞台灯光控制：这是一种规则驱动的过程还是生成任务？ 

**Authors**: Zijian Zhao, Dian Jin, Zijing Zhou, Xiaoyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01482)  

**Abstract**: Stage lighting plays an essential role in live music performances, influencing the engaging experience of both musicians and audiences. Given the high costs associated with hiring or training professional lighting engineers, Automatic Stage Lighting Control (ASLC) has gained increasing attention. However, most existing approaches only classify music into limited categories and map them to predefined light patterns, resulting in formulaic and monotonous outcomes that lack rationality. To address this issue, this paper presents an end-to-end solution that directly learns from experienced lighting engineers -- Skip-BART. To the best of our knowledge, this is the first work to conceptualize ASLC as a generative task rather than merely a classification problem. Our method modifies the BART model to take audio music as input and produce light hue and value (intensity) as output, incorporating a novel skip connection mechanism to enhance the relationship between music and light within the frame this http URL validate our method through both quantitative analysis and an human evaluation, demonstrating that Skip-BART outperforms conventional rule-based methods across all evaluation metrics and shows only a limited gap compared to real lighting this http URL, our method yields a p-value of 0.72 in a statistical comparison based on human evaluations with human lighting engineers, suggesting that the proposed approach closely matches human lighting engineering performance. To support further research, we have made our self-collected dataset, code, and trained model parameters available at this https URL . 

**Abstract (ZH)**: 自动舞台灯光控制在live音乐表演中的应用：基于Skip-BART模型的生成式解决方案 

---
# Agentic AI and Multiagentic: Are We Reinventing the Wheel? 

**Title (ZH)**: 代理AI与多代理系统：我们是否在重造轮子？ 

**Authors**: V.Botti  

**Link**: [PDF](https://arxiv.org/pdf/2506.01463)  

**Abstract**: The terms Agentic AI and Multiagentic AI have recently gained popularity in discussions on generative artificial intelligence, often used to describe autonomous software agents and systems composed of such agents. However, the use of these terms confuses these buzzwords with well-established concepts in AI literature: intelligent agents and multi-agent systems. This article offers a critical analysis of this conceptual misuse. We review the theoretical origins of "agentic" in the social sciences (Bandura, 1986) and philosophical notions of intentionality (Dennett, 1971), and then summarise foundational works on intelligent agents and multi-agent systems by Wooldridge, Jennings and others. We examine classic agent architectures, from simple reactive agents to Belief-Desire-Intention (BDI) models, and highlight key properties (autonomy, reactivity, proactivity, social capability) that define agency in AI. We then discuss recent developments in large language models (LLMs) and agent platforms based on LLMs, including the emergence of LLM-powered AI agents and open-source multi-agent orchestration frameworks. We argue that the term AI Agentic is often used as a buzzword for what are essentially AI agents, and AI Multiagentic for what are multi-agent systems. This confusion overlooks decades of research in the field of autonomous agents and multi-agent systems. The article advocates for scientific and technological rigour and the use of established terminology from the state of the art in AI, incorporating the wealth of existing knowledge, including standards for multi-agent system platforms, communication languages and coordination and cooperation algorithms, agreement technologies (automated negotiation, argumentation, virtual organisations, trust, reputation, etc.), into the new and promising wave of LLM-based AI agents, so as not to end up reinventing the wheel. 

**Abstract (ZH)**: Agentic AI与Multiagentic AI概念的批判性分析：从自主智能体和多智能体系统视角探究 

---
# GenDMR: A dynamic multimodal role-swapping network for identifying risk gene phenotypes 

**Title (ZH)**: GenDMR：一种动态多模态角色轮换网络，用于识别风险基因表型 

**Authors**: Lina Qin, Cheng Zhu, Chuqi Zhou, Yukun Huang, Jiayi Zhu, Ping Liang, Jinju Wang, Yixing Huang, Cheng Luo, Dezhong Yao, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.01456)  

**Abstract**: Recent studies have shown that integrating multimodal data fusion techniques for imaging and genetic features is beneficial for the etiological analysis and predictive diagnosis of Alzheimer's disease (AD). However, there are several critical flaws in current deep learning methods. Firstly, there has been insufficient discussion and exploration regarding the selection and encoding of genetic information. Secondly, due to the significantly superior classification value of AD imaging features compared to genetic features, many studies in multimodal fusion emphasize the strengths of imaging features, actively mitigating the influence of weaker features, thereby diminishing the learning of the unique value of genetic features. To address this issue, this study proposes the dynamic multimodal role-swapping network (GenDMR). In GenDMR, we develop a novel approach to encode the spatial organization of single nucleotide polymorphisms (SNPs), enhancing the representation of their genomic context. Additionally, to adaptively quantify the disease risk of SNPs and brain region, we propose a multi-instance attention module to enhance model interpretability. Furthermore, we introduce a dominant modality selection module and a contrastive self-distillation module, combining them to achieve a dynamic teacher-student role exchange mechanism based on dominant and auxiliary modalities for bidirectional co-updating of different modal data. Finally, GenDMR achieves state-of-the-art performance on the ADNI public dataset and visualizes attention to different SNPs, focusing on confirming 12 potential high-risk genes related to AD, including the most classic APOE and recently highlighted significant risk genes. This demonstrates GenDMR's interpretable analytical capability in exploring AD genetic features, providing new insights and perspectives for the development of multimodal data fusion techniques. 

**Abstract (ZH)**: Recent Studies Have Demonstrated the Beneficial Impact of Integrating Multimodal Data Fusion Techniques for Imaging and Genetic Features in the Etiological Analysis and Predictive Diagnosis of Alzheimer's Disease (AD): However, Current Deep Learning Methods Are Subject to Several Critical Flaws 

---
# From Initial Data to Boundary Layers: Neural Networks for Nonlinear Hyperbolic Conservation Laws 

**Title (ZH)**: 从初始数据到边界层：神经网络在非线性双曲守恒律中的应用 

**Authors**: Igor Ciril, Khalil Haddaoui, Yohann Tendero  

**Link**: [PDF](https://arxiv.org/pdf/2506.01453)  

**Abstract**: We address the approximation of entropy solutions to initial-boundary value problems for nonlinear strictly hyperbolic conservation laws using neural networks. A general and systematic framework is introduced for the design of efficient and reliable learning algorithms, combining fast convergence during training with accurate predictions. The methodology is assessed through a series of one-dimensional scalar test cases, highlighting its potential applicability to more complex industrial scenarios. 

**Abstract (ZH)**: 我们利用神经网络解决非线性严格双曲守恒律初始边值问题的熵解近似，并介绍了一种高效可靠的学习算法设计框架，该框架结合了训练期间的快速收敛和准确预测。该方法通过一系列一维标量测试案例进行了评估，展示了其在更复杂工业场景中的潜在应用。 

---
# ShaTS: A Shapley-based Explainability Method for Time Series Artificial Intelligence Models applied to Anomaly Detection in Industrial Internet of Things 

**Title (ZH)**: 基于Shapley值的时序人工智能模型可解释性方法及其在工业物联网异常检测中的应用 

**Authors**: Manuel Franco de la Peña, Ángel Luis Perales Gómez, Lorenzo Fernández Maimó  

**Link**: [PDF](https://arxiv.org/pdf/2506.01450)  

**Abstract**: Industrial Internet of Things environments increasingly rely on advanced Anomaly Detection and explanation techniques to rapidly detect and mitigate cyberincidents, thereby ensuring operational safety. The sequential nature of data collected from these environments has enabled improvements in Anomaly Detection using Machine Learning and Deep Learning models by processing time windows rather than treating the data as tabular. However, conventional explanation methods often neglect this temporal structure, leading to imprecise or less actionable explanations. This work presents ShaTS (Shapley values for Time Series models), which is a model-agnostic explainable Artificial Intelligence method designed to enhance the precision of Shapley value explanations for time series models. ShaTS addresses the shortcomings of traditional approaches by incorporating an a priori feature grouping strategy that preserves temporal dependencies and produces both coherent and actionable insights. Experiments conducted on the SWaT dataset demonstrate that ShaTS accurately identifies critical time instants, precisely pinpoints the sensors, actuators, and processes affected by anomalies, and outperforms SHAP in terms of both explainability and resource efficiency, fulfilling the real-time requirements of industrial environments. 

**Abstract (ZH)**: 工业物联网环境 increasingly依靠先进的异常检测和解释技术快速检测和缓解网络事件，以确保操作安全。这些环境采集的数据序列性促使通过处理时间窗口而非将数据视为表格的方式来提高基于机器学习和深度学习的异常检测性能。然而，传统的解释方法往往忽视了这种时间结构，导致不精确或不可操作的解释。本文介绍了ShaTS（时间序列模型的Shapley值方法），这是一种模型无关的解释性人工智能方法，旨在增强时间序列模型Shapley值解释的精确性。ShaTS通过结合先验特征分组策略来处理传统方法的不足，该策略保留了时间依赖性并产生连贯且可操作的见解。基于SWaT数据集的实验表明，ShaTS准确地识别了关键的时间瞬间，精确地确定了受影响的传感器、执行器和过程，并在解释性和资源效率方面优于SHAP，满足了工业环境的实时要求。 

---
# Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models 

**Title (ZH)**: 激励大型语言模型进行高级指令跟随推理 

**Authors**: Yulei Qin, Gang Li, Zongyi Li, Zihan Xu, Yuchen Shi, Zhekai Lin, Xiao Cui, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01413)  

**Abstract**: Existing large language models (LLMs) face challenges of following complex instructions, especially when multiple constraints are present and organized in paralleling, chaining, and branching structures. One intuitive solution, namely chain-of-thought (CoT), is expected to universally improve capabilities of LLMs. However, we find that the vanilla CoT exerts a negative impact on performance due to its superficial reasoning pattern of simply paraphrasing the instructions. It fails to peel back the compositions of constraints for identifying their relationship across hierarchies of types and dimensions. To this end, we propose a systematic method to boost LLMs in dealing with complex instructions via incentivizing reasoning for test-time compute scaling. First, we stem from the decomposition of complex instructions under existing taxonomies and propose a reproducible data acquisition method. Second, we exploit reinforcement learning (RL) with verifiable rule-centric reward signals to cultivate reasoning specifically for instruction following. We address the shallow, non-essential nature of reasoning under complex instructions via sample-wise contrast for superior CoT enforcement. We also exploit behavior cloning of experts to facilitate steady distribution shift from fast-thinking LLMs to skillful reasoners. Extensive evaluations on seven comprehensive benchmarks confirm the validity of the proposed method, where a 1.5B LLM achieves 11.74% gains with performance comparable to a 8B LLM. Codes and data are available at this https URL. 

**Abstract (ZH)**: 现有大规模语言模型（LLMs）在遵循复杂指令方面面临挑战，尤其是当存在并行、串联和分支结构的多重约束时。一种直观的解决方案，即逐步思考（CoT），有望普遍提高LLMs的能力。然而，我们发现，传统的CoT由于其简单的重述指令的表面化推理模式，对性能产生了负面影响。它未能揭示不同层级和维度约束之间的关系。为了解决这一问题，我们提出了一种系统方法，通过激励推理以适应测试时计算扩展来促进LLMs处理复杂指令。首先，我们基于现有分类体系对复杂指令进行分解，并提出了一种可重复的数据采集方法。其次，我们利用验证性规则中心奖励信号的强化学习（RL）来培养专门针对指令遵循的推理能力。我们通过样本级对比强化CoT的有效性，解决了复杂指令下浅薄且非本质的推理问题。我们还利用专家行为克隆来促进从快速思考的LLMs到善于推理者的平稳分布转移。广泛的基准测试表明，所提出的方法具有有效性，一个1.5B的LLM在性能上与8B的LLM相当，获得了11.74%的提升。代码和数据可在以下链接获取。 

---
# System Calls for Malware Detection and Classification: Methodologies and Applications 

**Title (ZH)**: 恶意软件检测与分类的系统调用方法及其应用 

**Authors**: Bishwajit Prasad Gond, Durga Prasad Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2506.01412)  

**Abstract**: As malware continues to become more complex and harder to detect, Malware Analysis needs to continue to evolve to stay one step ahead. One promising key area approach focuses on using system calls and API Calls, the core communication between user applications and the operating system and their kernels. These calls provide valuable insight into how software or programs behaves, making them an useful tool for spotting suspicious or harmful activity of programs and software. This chapter takes a deep down look at how system calls are used in malware detection and classification, covering techniques like static and dynamic analysis, as well as sandboxing. By combining these methods with advanced techniques like machine learning, statistical analysis, and anomaly detection, researchers can analyze system call patterns to tell the difference between normal and malicious behavior. The chapter also explores how these techniques are applied across different systems, including Windows, Linux, and Android, while also looking at the ways sophisticated malware tries to evade detection. 

**Abstract (ZH)**: 随着恶意软件变得越来越复杂且更难检测，恶意软件分析需要不断进化以保持领先。一种有前景的关键方法是利用系统调用和API调用，这些调用是用户应用程序与操作系统及其内核之间核心通信的基础。这些调用提供了关于软件或程序行为的宝贵见解，使它们成为识别程序和软件中的可疑或有害活动的有效工具。本章深入探讨了系统调用在恶意软件检测和分类中的应用，涵盖了静态分析和动态分析等技术，以及沙箱技术。通过将这些方法与机器学习、统计分析和异常检测等高级技术相结合，研究人员可以分析系统调用模式以区分正常和恶意行为。本章还探讨了这些技术在Windows、Linux和Android等不同系统中的应用，同时也考察了复杂恶意软件如何规避检测的方法。 

---
# ViTA-PAR: Visual and Textual Attribute Alignment with Attribute Prompting for Pedestrian Attribute Recognition 

**Title (ZH)**: ViTA-PAR：基于属性提示的视觉和文本属性对齐的人体属性识别 

**Authors**: Minjeong Park, Hongbeen Park, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01411)  

**Abstract**: The Pedestrian Attribute Recognition (PAR) task aims to identify various detailed attributes of an individual, such as clothing, accessories, and gender. To enhance PAR performance, a model must capture features ranging from coarse-grained global attributes (e.g., for identifying gender) to fine-grained local details (e.g., for recognizing accessories) that may appear in diverse regions. Recent research suggests that body part representation can enhance the model's robustness and accuracy, but these methods are often restricted to attribute classes within fixed horizontal regions, leading to degraded performance when attributes appear in varying or unexpected body locations. In this paper, we propose Visual and Textual Attribute Alignment with Attribute Prompting for Pedestrian Attribute Recognition, dubbed as ViTA-PAR, to enhance attribute recognition through specialized multimodal prompting and vision-language alignment. We introduce visual attribute prompts that capture global-to-local semantics, enabling diverse attribute representations. To enrich textual embeddings, we design a learnable prompt template, termed person and attribute context prompting, to learn person and attributes context. Finally, we align visual and textual attribute features for effective fusion. ViTA-PAR is validated on four PAR benchmarks, achieving competitive performance with efficient inference. We release our code and model at this https URL. 

**Abstract (ZH)**: 基于视觉和文本属性对齐及属性提示的行人属性识别（ViTA-PAR） 

---
# Sparse Imagination for Efficient Visual World Model Planning 

**Title (ZH)**: 稀疏想象以实现高效的视觉世界模型规划 

**Authors**: Junha Chun, Youngjoon Jeong, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01392)  

**Abstract**: World model based planning has significantly improved decision-making in complex environments by enabling agents to simulate future states and make informed choices. However, ensuring the prediction accuracy of world models often demands substantial computational resources, posing a major challenge for real-time applications. This computational burden is particularly restrictive in robotics, where resources are severely constrained. To address this limitation, we propose a Sparse Imagination for Efficient Visual World Model Planning, which enhances computational efficiency by reducing the number of tokens processed during forward prediction. Our method leverages a sparsely trained vision-based world model based on transformers with randomized grouped attention strategy, allowing the model to adaptively adjust the number of tokens processed based on the computational resource. By enabling sparse imagination (rollout), our approach significantly accelerates planning while maintaining high control fidelity. Experimental results demonstrate that sparse imagination preserves task performance while dramatically improving inference efficiency, paving the way for the deployment of world models in real-time decision-making scenarios. 

**Abstract (ZH)**: 基于 Worlds 模型的稀疏想象高效视觉世界模型规划 

---
# VRD-IU: Lessons from Visually Rich Document Intelligence and Understanding 

**Title (ZH)**: VRD-IU：视觉丰富文档的智能与理解经验 

**Authors**: Yihao Ding, Soyeon Caren Han, Yan Li, Josiah Poon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01388)  

**Abstract**: Visually Rich Document Understanding (VRDU) has emerged as a critical field in document intelligence, enabling automated extraction of key information from complex documents across domains such as medical, financial, and educational applications. However, form-like documents pose unique challenges due to their complex layouts, multi-stakeholder involvement, and high structural variability. Addressing these issues, the VRD-IU Competition was introduced, focusing on extracting and localizing key information from multi-format forms within the Form-NLU dataset, which includes digital, printed, and handwritten documents. This paper presents insights from the competition, which featured two tracks: Track A, emphasizing entity-based key information retrieval, and Track B, targeting end-to-end key information localization from raw document images. With over 20 participating teams, the competition showcased various state-of-the-art methodologies, including hierarchical decomposition, transformer-based retrieval, multimodal feature fusion, and advanced object detection techniques. The top-performing models set new benchmarks in VRDU, providing valuable insights into document intelligence. 

**Abstract (ZH)**: 视觉丰富的文档理解（VRDU）已成为文档智能领域的关键领域，使自动化从跨医疗、金融和教育应用等领域的复杂文档中提取关键信息成为可能。然而，表格形式的文档由于其复杂的布局、多利益相关者的参与和高度的结构性变异性，提出了独特的挑战。为应对这些挑战，引入了VRD-IU竞赛，专注于在Form-NLU数据集中从多种格式的表格中提取和本地化关键信息，该数据集包括数字、打印和手写文档。本文介绍了竞赛的见解，竞赛设有两条赛道：A轨道侧重于基于实体的关键信息检索，B轨道则旨在从原始文档图像中端到端地定位关键信息。超过20个参赛团队展示了包括分层分解、基于变换器的检索、多模态特征融合和高级目标检测技术在内的多种先进的方法。表现最佳的模型在VRDU领域设立了新的基准，提供了宝贵的文档智能见解。 

---
# Playing with Transformer at 30+ FPS via Next-Frame Diffusion 

**Title (ZH)**: 利用下一帧扩散在30+ FPS下玩转Transformer 

**Authors**: Xinle Cheng, Tianyu He, Jiayi Xu, Junliang Guo, Di He, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.01380)  

**Abstract**: Autoregressive video models offer distinct advantages over bidirectional diffusion models in creating interactive video content and supporting streaming applications with arbitrary duration. In this work, we present Next-Frame Diffusion (NFD), an autoregressive diffusion transformer that incorporates block-wise causal attention, enabling iterative sampling and efficient inference via parallel token generation within each frame. Nonetheless, achieving real-time video generation remains a significant challenge for such models, primarily due to the high computational cost associated with diffusion sampling and the hardware inefficiencies inherent to autoregressive generation. To address this, we introduce two innovations: (1) We extend consistency distillation to the video domain and adapt it specifically for video models, enabling efficient inference with few sampling steps; (2) To fully leverage parallel computation, motivated by the observation that adjacent frames often share the identical action input, we propose speculative sampling. In this approach, the model generates next few frames using current action input, and discard speculatively generated frames if the input action differs. Experiments on a large-scale action-conditioned video generation benchmark demonstrate that NFD beats autoregressive baselines in terms of both visual quality and sampling efficiency. We, for the first time, achieves autoregressive video generation at over 30 Frames Per Second (FPS) on an A100 GPU using a 310M model. 

**Abstract (ZH)**: 自回归视频模型在创建交互视频内容和支持任意时长的流媒体应用方面具有双向扩散模型的独特优势。本文介绍了一种名为Next-Frame Diffusion (NFD)的自回归扩散变换器，该模型采用了区块因果注意力机制，能够在每帧内通过并行令牌生成实现迭代采样和高效推理。尽管如此，由于扩散采样的高计算成本和自回归生成的固有硬件不效率，实现实时视频生成仍是一项重大挑战。为了解决这一问题，我们引入了两项创新：（1）将一致性蒸馏扩展到视频领域，并针对视频模型进行适配，以实现高效推理且只需少量采样步骤；（2）为了充分利用并行计算，在观察到相邻帧通常具有相同的动作输入后，我们提出了一种推测性采样方法，该方法利用当前动作输入生成接下来几帧的内容，并在输入动作变化时丢弃推测生成的帧。在大规模动作条件视频生成基准测试上的实验表明，NFD在视觉质量和采样效率方面都优于自回归基线。我们首次使用一个310M模型在A100 GPU上实现了超过30帧每秒（FPS）的自回归视频生成。 

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
# Unraveling Spatio-Temporal Foundation Models via the Pipeline Lens: A Comprehensive Review 

**Title (ZH)**: 通过流水线视角解析时空基础模型：一个综合Review 

**Authors**: Yuchen Fang, Hao Miao, Yuxuan Liang, Liwei Deng, Yue Cui, Ximu Zeng, Yuyang Xia, Yan Zhao, Torben Bach Pedersen, Christian S. Jensen, Xiaofang Zhou, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01364)  

**Abstract**: Spatio-temporal deep learning models aims to utilize useful patterns in such data to support tasks like prediction. However, previous deep learning models designed for specific tasks typically require separate training for each use case, leading to increased computational and storage costs. To address this issue, spatio-temporal foundation models have emerged, offering a unified framework capable of solving multiple spatio-temporal tasks. These foundation models achieve remarkable success by learning general knowledge with spatio-temporal data or transferring the general capabilities of pre-trained language models. While previous surveys have explored spatio-temporal data and methodologies separately, they have ignored a comprehensive examination of how foundation models are designed, selected, pre-trained, and adapted. As a result, the overall pipeline for spatio-temporal foundation models remains unclear. To bridge this gap, we innovatively provide an up-to-date review of previous spatio-temporal foundation models from the pipeline perspective. The pipeline begins with an introduction to different types of spatio-temporal data, followed by details of data preprocessing and embedding techniques. The pipeline then presents a novel data property taxonomy to divide existing methods according to data sources and dependencies, providing efficient and effective model design and selection for researchers. On this basis, we further illustrate the training objectives of primitive models, as well as the adaptation techniques of transferred models. Overall, our survey provides a clear and structured pipeline to understand the connection between core elements of spatio-temporal foundation models while guiding researchers to get started quickly. Additionally, we introduce emerging opportunities such as multi-objective training in the field of spatio-temporal foundation models. 

**Abstract (ZH)**: 时空深度学习模型旨在利用时空数据中的有用模式以支持预测等任务。然而，之前为特定任务设计的深度学习模型通常需要为每种使用场景分别进行训练，导致计算和存储成本增加。为解决这一问题，时空基础模型已崭露头角，提供了一个能够解决多种时空任务的统一框架。通过学习时空数据中的通用知识或转移预训练语言模型的通用能力，这些基础模型取得了显著的成功。尽管以往的综述分别探讨了时空数据和方法，但忽视了对基础模型设计、选择、预训练和适应的全面考察。因此，时空基础模型的整体管道流程仍不清晰。为弥补这一缺口，我们从管道视角创新性地提供了一种对时空基础模型的最新综述。该管道从不同类型的时空数据介绍开始，接着详细说明数据预处理和嵌入技术。然后，该管道提出了一个新的数据属性分类法，根据数据来源和依赖关系对现有方法进行分类，为研究人员提供高效有效的模型设计和选择。在此基础上，我们进一步阐述了基础模型的训练目标以及转移模型的适应技术。总体而言，我们的综述提供了一条清晰而结构化的管道，指导研究人员理解时空基础模型核心元素之间的关系并迅速入门。此外，我们还介绍了时空基础模型领域新兴的机会，例如多目标训练。 

---
# KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors 

**Title (ZH)**: KokoroChat：由受训心理咨询师通过角色扮演收集的日语心理咨询服务对话数据集 

**Authors**: Zhiyang Qi, Takumasa Kaneko, Keiko Takamizo, Mariko Ukiyo, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2506.01357)  

**Abstract**: Generating psychological counseling responses with language models relies heavily on high-quality datasets. Crowdsourced data collection methods require strict worker training, and data from real-world counseling environments may raise privacy and ethical concerns. While recent studies have explored using large language models (LLMs) to augment psychological counseling dialogue datasets, the resulting data often suffers from limited diversity and authenticity. To address these limitations, this study adopts a role-playing approach where trained counselors simulate counselor-client interactions, ensuring high-quality dialogues while mitigating privacy risks. Using this method, we construct KokoroChat, a Japanese psychological counseling dialogue dataset comprising 6,589 long-form dialogues, each accompanied by comprehensive client feedback. Experimental results demonstrate that fine-tuning open-source LLMs with KokoroChat improves both the quality of generated counseling responses and the automatic evaluation of counseling dialogues. The KokoroChat dataset is available at this https URL. 

**Abstract (ZH)**: 使用语言模型生成心理咨询服务响应依赖于高質量数据集。 Crowd-sourced数据采集方法需要严格的工人培训，而来自真实心理咨询环境的数据可能引发隐私和伦理问题。尽管最近的研究探索了利用大规模语言模型（LLMs）扩充心理咨询服务对话数据集，但由此产生的数据往往缺乏多样性和真实性。为解决这些局限性，本研究采用角色扮演方法，由训练有素的咨询师模拟咨询师-来访者互动，确保高质量对话的同时减轻隐私风险。通过这种方法，我们构建了包含6,589条长对话的KokoroChat日语心理咨询服务对话数据集，每条对话均附有全面的来访者反馈。实验结果表明，使用KokoroChat微调开源语言模型可以提高生成的心理咨询响应质量和对话自动评估效果。KokoroChat数据集可从此链接获得：this https URL。 

---
# NoiseAR: AutoRegressing Initial Noise Prior for Diffusion Models 

**Title (ZH)**: NoiseAR：用于扩散模型的自回归初始噪声先验 

**Authors**: Zeming Li, Xiangyue Liu, Xiangyu Zhang, Ping Tan, Heung-Yeung Shum  

**Link**: [PDF](https://arxiv.org/pdf/2506.01337)  

**Abstract**: Diffusion models have emerged as powerful generative frameworks, creating data samples by progressively denoising an initial random state. Traditionally, this initial state is sampled from a simple, fixed distribution like isotropic Gaussian, inherently lacking structure and a direct mechanism for external control. While recent efforts have explored ways to introduce controllability into the diffusion process, particularly at the initialization stage, they often rely on deterministic or heuristic approaches. These methods can be suboptimal, lack expressiveness, and are difficult to scale or integrate into more sophisticated optimization frameworks. In this paper, we introduce NoiseAR, a novel method for AutoRegressive Initial Noise Prior for Diffusion Models. Instead of a static, unstructured source, NoiseAR learns to generate a dynamic and controllable prior distribution for the initial noise. We formulate the generation of the initial noise prior's parameters as an autoregressive probabilistic modeling task over spatial patches or tokens. This approach enables NoiseAR to capture complex spatial dependencies and introduce learned structure into the initial state. Crucially, NoiseAR is designed to be conditional, allowing text prompts to directly influence the learned prior, thereby achieving fine-grained control over the diffusion initialization. Our experiments demonstrate that NoiseAR can generate initial noise priors that lead to improved sample quality and enhanced consistency with conditional inputs, offering a powerful, learned alternative to traditional random initialization. A key advantage of NoiseAR is its probabilistic formulation, which naturally supports seamless integration into probabilistic frameworks like Markov Decision Processes and Reinforcement Learning. Our code will be available at this https URL 

**Abstract (ZH)**: NoiseAR：自动回归初始噪声先验方法用于扩散模型 

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
# STSA: Federated Class-Incremental Learning via Spatial-Temporal Statistics Aggregation 

**Title (ZH)**: STSA: 基于空间-时间统计聚合的联邦类增量学习 

**Authors**: Zenghao Guan, Guojun Zhu, Yucan Zhou, Wu Liu, Weiping Wang, Jiebo Luo, Xiaoyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01327)  

**Abstract**: Federated Class-Incremental Learning (FCIL) enables Class-Incremental Learning (CIL) from distributed data. Existing FCIL methods typically integrate old knowledge preservation into local client training. However, these methods cannot avoid spatial-temporal client drift caused by data heterogeneity and often incur significant computational and communication overhead, limiting practical deployment. To address these challenges simultaneously, we propose a novel approach, Spatial-Temporal Statistics Aggregation (STSA), which provides a unified framework to aggregate feature statistics both spatially (across clients) and temporally (across stages). The aggregated feature statistics are unaffected by data heterogeneity and can be used to update the classifier in closed form at each stage. Additionally, we introduce STSA-E, a communication-efficient variant with theoretical guarantees, achieving similar performance to STSA-E with much lower communication overhead. Extensive experiments on three widely used FCIL datasets, with varying degrees of data heterogeneity, show that our method outperforms state-of-the-art FCIL methods in terms of performance, flexibility, and both communication and computation efficiency. 

**Abstract (ZH)**: 联邦类增量学习（FCIL）使分布式数据上的类增量学习（CIL）成为可能。现有的FCIL方法通常将旧知识保留融入到局部客户端训练中。然而，这些方法无法避免由数据异质性引起的时空客户端漂移，并且往往会导致显著的计算和通信开销，限制了其实用部署。为了解决这些挑战，我们提出了一个新颖的方法，时空统计聚合（STSA），它提供了一个统一框架来聚合特征统计（空间上跨客户端和时间上跨阶段）。聚合的特征统计不受数据异质性的影响，并且可以在每个阶段以封闭形式更新分类器。此外，我们引入了STSA-E，这是一种通信高效的变体，并具有理论保证，其性能与STSA-E相当，但通信开销更低。在三个广泛使用的FCIL数据集上进行的大量实验，数据异质性程度不同，表明我们的方法在性能、灵活性以及通信和计算效率方面优于现有的最先进的FCIL方法。 

---
# $Ψ$-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models 

**Title (ZH)**: $Ψ$-Sampler: 初始粒子采样算法在基于SMC的推理时奖励对齐中的应用 

**Authors**: Taehoon Yoon, Yunhong Min, Kyeongmin Yeo, Minhyuk Sung  

**Link**: [PDF](https://arxiv.org/pdf/2506.01320)  

**Abstract**: We introduce $\Psi$-Sampler, an SMC-based framework incorporating pCNL-based initial particle sampling for effective inference-time reward alignment with a score-based generative model. Inference-time reward alignment with score-based generative models has recently gained significant traction, following a broader paradigm shift from pre-training to post-training optimization. At the core of this trend is the application of Sequential Monte Carlo (SMC) to the denoising process. However, existing methods typically initialize particles from the Gaussian prior, which inadequately captures reward-relevant regions and results in reduced sampling efficiency. We demonstrate that initializing from the reward-aware posterior significantly improves alignment performance. To enable posterior sampling in high-dimensional latent spaces, we introduce the preconditioned Crank-Nicolson Langevin (pCNL) algorithm, which combines dimension-robust proposals with gradient-informed dynamics. This approach enables efficient and scalable posterior sampling and consistently improves performance across various reward alignment tasks, including layout-to-image generation, quantity-aware generation, and aesthetic-preference generation, as demonstrated in our experiments. 

**Abstract (ZH)**: 基于pCNL初始化粒子的Ψ-采样器：一种SMC框架，用于评分生成模型的推理时奖励对齐 

---
# Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack 

**Title (ZH)**: 去学习的盲点：过度去学习与原型重学习攻击 

**Authors**: SeungBum Ha, Saerom Park, Sung Whan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01318)  

**Abstract**: Machine unlearning (MU) aims to expunge a designated forget set from a trained model without costly retraining, yet the existing techniques overlook two critical blind spots: "over-unlearning" that deteriorates retained data near the forget set, and post-hoc "relearning" attacks that aim to resurrect the forgotten knowledge. We first derive the over-unlearning metric OU@{\epsilon}, which represents the collateral damage to the nearby region of the forget set, where the over-unlearning mainly appears. Next, we expose an unforeseen relearning threat on MU, i.e., the Prototypical Relearning Attack, which exploits the per-class prototype of the forget class with just a few samples, and easily restores the pre-unlearning performance. To counter both blind spots, we introduce Spotter, a plug-and-play objective that combines (i) a masked knowledge-distillation penalty on the nearby region of forget set to suppress OU@{\epsilon}, and (ii) an intra-class dispersion loss that scatters forget-class embeddings, neutralizing prototypical relearning attacks. On CIFAR-10, as one of validations, Spotter reduces OU@{\epsilon}by below the 0.05X of the baseline, drives forget accuracy to 0%, preserves accuracy of the retain set within 1% of difference with the original, and denies the prototype-attack by keeping the forget set accuracy within <1%, without accessing retained data. It confirms that Spotter is a practical remedy of the unlearning's blind spots. 

**Abstract (ZH)**: 机器去学习：解决去学习中的过度去学习和后学习恢复攻击 

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
# ReFoCUS: Reinforcement-guided Frame Optimization for Contextual Understanding 

**Title (ZH)**: ReFoCUS: 基于强化学习的框架优化以实现上下文理解 

**Authors**: Hosu Lee, Junho Kim, Hyunjun Kim, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2506.01274)  

**Abstract**: Recent progress in Large Multi-modal Models (LMMs) has enabled effective vision-language reasoning, yet the ability to understand video content remains constrained by suboptimal frame selection strategies. Existing approaches often rely on static heuristics or external retrieval modules to feed frame information into video-LLMs, which may fail to provide the query-relevant information. In this work, we introduce ReFoCUS (Reinforcement-guided Frame Optimization for Contextual UnderStanding), a novel frame-level policy optimization framework that shifts the optimization target from textual responses to visual input selection. ReFoCUS learns a frame selection policy via reinforcement learning, using reward signals derived from a reference LMM to reflect the model's intrinsic preferences for frames that best support temporally grounded responses. To efficiently explore the large combinatorial frame space, we employ an autoregressive, conditional selection architecture that ensures temporal coherence while reducing complexity. Our approach does not require explicit supervision at the frame-level and consistently improves reasoning performance across multiple video QA benchmarks, highlighting the benefits of aligning frame selection with model-internal utility. 

**Abstract (ZH)**: Recent Progress in Large Multi-modal Models: Reinforcement-guided Frame Optimization for Contextual Understanding 

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
# Visual Sparse Steering: Improving Zero-shot Image Classification with Sparsity Guided Steering Vectors 

**Title (ZH)**: 视觉稀疏引导：基于稀疏引导向量的零样本图像分类改进 

**Authors**: Gerasimos Chatzoudis, Zhuowei Li, Gemma E. Moran, Hao Wang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2506.01247)  

**Abstract**: Steering vision foundation models at inference time without retraining or access to large labeled datasets is a desirable yet challenging objective, particularly in dynamic or resource-constrained settings. In this paper, we introduce Visual Sparse Steering (VS2), a lightweight, test-time method that guides vision models using steering vectors derived from sparse features learned by top-$k$ Sparse Autoencoders without requiring contrastive data. Specifically, VS2 surpasses zero-shot CLIP by 4.12% on CIFAR-100, 1.08% on CUB-200, and 1.84% on Tiny-ImageNet. We further propose VS2++, a retrieval-augmented variant that selectively amplifies relevant sparse features using pseudo-labeled neighbors at inference time. With oracle positive/negative sets, VS2++ achieves absolute top-1 gains over CLIP zero-shot of up to 21.44% on CIFAR-100, 7.08% on CUB-200, and 20.47% on Tiny-ImageNet. Interestingly, VS2 and VS2++ raise per-class accuracy by up to 25% and 38%, respectively, showing that sparse steering benefits specific classes by disambiguating visually or taxonomically proximate categories rather than providing a uniform boost. Finally, to better align the sparse features learned through the SAE reconstruction task with those relevant for downstream performance, we propose Prototype-Aligned Sparse Steering (PASS). By incorporating a prototype-alignment loss during SAE training, using labels only during training while remaining fully test-time unsupervised, PASS consistently, though modestly, outperforms VS2, achieving a 6.12% gain over VS2 only on CIFAR-100 with ViT-B/32. 

**Abstract (ZH)**: 无需重新训练或访问大型标记数据集，在推理时引导视觉基础模型是一个既 desirable 又 challenging 的目标，尤其是在动态或资源受限的环境中。本文介绍了 Visual Sparse Steering (VS2)，一种轻量级的测试时方法，该方法使用由 top-$k$ 稀疏自编码器学习的稀疏特征导出的引导向量来引导视觉模型，而不需对比数据。具体而言，VS2 在 CIFAR-100 上超越零样本 CLIP 的表现提高了 4.12%，在 CUB-200 上提高了 1.08%，在 Tiny-ImageNet 上提高了 1.84%。我们进一步提出了 VS2++，这是一种检索增强变体，在推理时选择性地放大相关稀疏特征，使用伪标记的邻居。使用 oracle 正负集，VS2++ 在 CIFAR-100 上的绝对 top-1 提升最高达到 21.44%，在 CUB-200 上达到 7.08%，在 Tiny-ImageNet 上达到 20.47%。有趣的是，VS2 和 VS2++ 分别将每类准确率提高了 25% 和 38%，表明稀疏引导有助于通过区分视觉上或分类上相近的类别来特异性提高某些类别的性能，而不是提供统一的提升。最后，为了更好地使通过 SAE 重建任务学习到的稀疏特征与下游性能相关的稀疏特征保持一致，我们提出了 Prototype-Aligned Sparse Steering (PASS)。通过在 SAE 训练过程中引入原型对齐损失，仅在训练过程中使用标签而在完全无监督的测试时保持，PASS 一致但适度地优于 VS2，在 ViT-B/32 上仅在 CIFAR-100 上实现了 6.12% 的提升。 

---
# General search techniques without common knowledge for imperfect-information games, and application to superhuman Fog of War chess 

**Title (ZH)**: 不依赖共同知识的一般搜索技术在不完全信息游戏中的应用：以超人类的迷雾棋为例 

**Authors**: Brian Hu Zhang, Tuomas Sandholm  

**Link**: [PDF](https://arxiv.org/pdf/2506.01242)  

**Abstract**: Since the advent of AI, games have served as progress benchmarks. Meanwhile, imperfect-information variants of chess have existed for over a century, present extreme challenges, and have been the focus of significant AI research. Beyond calculation needed in regular chess, they require reasoning about information gathering, the opponent's knowledge, signaling, etc. The most popular variant, Fog of War (FoW) chess (aka. dark chess) is a recognized challenge problem in AI after superhuman performance was reached in no-limit Texas hold'em poker. We present Obscuro, the first superhuman AI for FoW chess. It introduces advances to search in imperfect-information games, enabling strong, scalable reasoning. Experiments against the prior state-of-the-art AI and human players -- including the world's best -- show that Obscuro is significantly stronger. FoW chess is the largest (by amount of imperfect information) turn-based game in which superhuman performance has been achieved and the largest game in which imperfect-information search has been successfully applied. 

**Abstract (ZH)**: 自人工智能问世以来，游戏一直作为进步的标准。与此同时，信息不完全的棋类变种已有超过一个世纪的历史，极具挑战性，并一直是人工智能研究的重点。除了常规象棋所需的战略计算，它们还要求对信息收集、对手的知识、信号传递等进行推理。最流行的变种雾战象棋（Fog of War chess，又称暗棋）在无限制德州扑克达到超人类表现后，成为了人工智能中的一个公认挑战问题。我们提出了Obscuro，这是首个超人类雾战象棋AI。它在不完全信息游戏的搜索算法上取得了进步，使强大的、可扩展的推理成为可能。与之前的最先进的AI及人类玩家（包括世界最佳玩家）的实验表明，Obscuro显著更强大。雾战象棋是首个达到超人类表现且成功应用不完全信息搜索的最大（按信息不完全程度计算）回合制游戏，也是最大的此类游戏。 

---
# Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean 

**Title (ZH)**: 优化GEM的每一个 facets：测试LLMs和人类在韩语中的语言能力 

**Authors**: SungHo Kim, Nayeon Kim, Taehee Jeon, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01237)  

**Abstract**: We introduce the $\underline{Ko}rean \underline{G}rammar \underline{E}valuation Bench\underline{M}ark (KoGEM)$, designed to assess the linguistic competence of LLMs and humans in Korean. KoGEM consists of 1.5k multiple-choice QA pairs covering five main categories and 16 subcategories. The zero-shot evaluation of 27 LLMs of various sizes and types reveals that while LLMs perform remarkably well on straightforward tasks requiring primarily definitional knowledge, they struggle with tasks that demand the integration of real-world experiential knowledge, such as phonological rules and pronunciation. Furthermore, our in-depth analysis suggests that incorporating such experiential knowledge could enhance the linguistic competence of LLMs. With KoGEM, we not only highlight the limitations of current LLMs in linguistic competence but also uncover hidden facets of LLMs in linguistic competence, paving the way for enhancing comprehensive language understanding. Our code and dataset are available at: this https URL. 

**Abstract (ZH)**: 韩语语法评估基准(KoGEM)：评估LLM和人类在韩语中的语言能力 

---
# Fourier-Modulated Implicit Neural Representation for Multispectral Satellite Image Compression 

**Title (ZH)**: 傅里叶调制隐式神经表示在多光谱卫星图像压缩中的应用 

**Authors**: Woojin Cho, Steve Andreas Immanuel, Junhyuk Heo, Darongsae Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01234)  

**Abstract**: Multispectral satellite images play a vital role in agriculture, fisheries, and environmental monitoring. However, their high dimensionality, large data volumes, and diverse spatial resolutions across multiple channels pose significant challenges for data compression and analysis. This paper presents ImpliSat, a unified framework specifically designed to address these challenges through efficient compression and reconstruction of multispectral satellite data. ImpliSat leverages Implicit Neural Representations (INR) to model satellite images as continuous functions over coordinate space, capturing fine spatial details across varying spatial resolutions. Furthermore, we introduce a Fourier modulation algorithm that dynamically adjusts to the spectral and spatial characteristics of each band, ensuring optimal compression while preserving critical image details. 

**Abstract (ZH)**: 多光谱卫星图像在农业、渔业和环境监测中发挥着重要作用，但由于其高维度、大数据量和多通道上的不同空间分辨率，这些图像在数据压缩和分析中面临着显著挑战。本文提出了ImpliSat，一种统一框架，专门用于通过高效压缩和重建多光谱卫星数据来应对这些挑战。ImpliSat 利用隐式神经表示（INR）将卫星图像建模为坐标空间上的连续函数，捕捉不同空间分辨率下的精细空间细节。此外，我们引入了一种傅里叶调制算法，能够动态调整以适应每通道的频谱和空间特性，确保在保留关键图像细节的同时实现最佳压缩。 

---
# Retrieval-Augmented Generation of Ontologies from Relational Databases 

**Title (ZH)**: 从关系数据库检索增强本体生成 

**Authors**: Mojtaba Nayyeri, Athish A Yogi, Nadeen Fathallah, Ratan Bahadur Thapa, Hans-Michael Tautenhahn, Anton Schnurpel, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2506.01232)  

**Abstract**: Transforming relational databases into knowledge graphs with enriched ontologies enhances semantic interoperability and unlocks advanced graph-based learning and reasoning over data. However, previous approaches either demand significant manual effort to derive an ontology from a database schema or produce only a basic ontology. We present RIGOR, Retrieval-augmented Iterative Generation of RDB Ontologies, an LLM-driven approach that turns relational schemas into rich OWL ontologies with minimal human effort. RIGOR combines three sources via RAG, the database schema and its documentation, a repository of domain ontologies, and a growing core ontology, to prompt a generative LLM for producing successive, provenance-tagged delta ontology fragments. Each fragment is refined by a judge-LLM before being merged into the core ontology, and the process iterates table-by-table following foreign key constraints until coverage is complete. Applied to real-world databases, our approach outputs ontologies that score highly on standard quality dimensions such as accuracy, completeness, conciseness, adaptability, clarity, and consistency, while substantially reducing manual effort. 

**Abstract (ZH)**: 将关系数据库转换为充实本体的知识图谱增强语义互操作性并解锁基于图的高级学习与推理。然而，先前的方法要么要求大量手动 effort 来从数据库模式推导本体，要么只能生成基础本体。我们提出 RIGOR：基于检索的迭代生成 RDB 本体方法，这是一种由大语言模型驱动的 approach，能够用最小的人工努力将关系模式转换为丰富的 OWL 本体。RIGOR 通过 RAG 结合数据库模式及其文档、领域本体仓库以及不断增长的核心本体，来提示生成型大语言模型以生成连续的、带有来源标注的 delta 本体片段。每个片段都由法官大语言模型 refining，然后合并到核心本体中，过程根据外键约束逐表迭代，直到完全覆盖。应用于实际数据库时，我们的方法在准确度、完整性、简洁性、可适应性、清晰性和一致性等标准质量维度上表现优异，同时大幅减少了人工努力。 

---
# Towards Efficient Few-shot Graph Neural Architecture Search via Partitioning Gradient Contribution 

**Title (ZH)**: 面向高效分割梯度贡献的少量-shot图神经架构搜索 

**Authors**: Wenhao Song, Xuan Wu, Bo Yang, You Zhou, Yubin Xiao, Yanchun Liang, Hongwei Ge, Heow Pueh Lee, Chunguo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01231)  

**Abstract**: To address the weight coupling problem, certain studies introduced few-shot Neural Architecture Search (NAS) methods, which partition the supernet into multiple sub-supernets. However, these methods often suffer from computational inefficiency and tend to provide suboptimal partitioning schemes. To address this problem more effectively, we analyze the weight coupling problem from a novel perspective, which primarily stems from distinct modules in succeeding layers imposing conflicting gradient directions on the preceding layer modules. Based on this perspective, we propose the Gradient Contribution (GC) method that efficiently computes the cosine similarity of gradient directions among modules by decomposing the Vector-Jacobian Product during supernet backpropagation. Subsequently, the modules with conflicting gradient directions are allocated to distinct sub-supernets while similar ones are grouped together. To assess the advantages of GC and address the limitations of existing Graph Neural Architecture Search methods, which are limited to searching a single type of Graph Neural Networks (Message Passing Neural Networks (MPNNs) or Graph Transformers (GTs)), we propose the Unified Graph Neural Architecture Search (UGAS) framework, which explores optimal combinations of MPNNs and GTs. The experimental results demonstrate that GC achieves state-of-the-art (SOTA) performance in supernet partitioning quality and time efficiency. In addition, the architectures searched by UGAS+GC outperform both the manually designed GNNs and those obtained by existing NAS methods. Finally, ablation studies further demonstrate the effectiveness of all proposed methods. 

**Abstract (ZH)**: 针对权重耦合问题，某些研究引入了少样本神经架构搜索（NAS）方法，将超网络划分为多个子超网络。然而，这些方法通常存在计算效率低下的问题，并且倾向于提供次优的划分方案。为更有效地解决这一问题，我们从一个新的视角分析了权重耦合问题，该问题主要源于后续层中的不同模块在前一层模块上施加了相互矛盾的梯度方向。基于这一视角，我们提出了梯度贡献（GC）方法，通过在超网络反向传播过程中分解向量-雅可比积来高效计算模块之间梯度方向的余弦相似度。接着，具有相互矛盾梯度方向的模块被分配到不同的子超网络中，而相似的模块则被分组。为评估GC的优势并解决现有图神经架构搜索方法的局限性（仅限于搜索一种类型的图神经网络，如消息传递神经网络（MPNNs）或图变换器（GTs）），我们提出了统一图神经架构搜索（UGAS）框架，探索MPNNs和GTs的最佳组合。实验结果表明，GC在超网络划分质量和时间效率方面达到了目前的最先进（SOTA）水平。此外，UGAS+GC搜索到的架构优于手动设计的图神经网络和现有NAS方法获得的架构。最后，消融研究进一步证明了所有提出方法的有效性。 

---
# SPEAR: Security Posture Evaluation using AI Planner-Reasoning on Attack-Connectivity Hypergraphs 

**Title (ZH)**: SPEAR：基于攻击连通超图的AI规划推理安全态势评估 

**Authors**: Rakesh Podder, Turgay Caglar, Shadaab Kawnain Bashir, Sarath Sreedharan, Indrajit Ray, Indrakshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2506.01227)  

**Abstract**: Graph-based frameworks are often used in network hardening to help a cyber defender understand how a network can be attacked and how the best defenses can be deployed. However, incorporating network connectivity parameters in the attack graph, reasoning about the attack graph when we do not have access to complete information, providing system administrator suggestions in an understandable format, and allowing them to do what-if analysis on various scenarios and attacker motives is still missing. We fill this gap by presenting SPEAR, a formal framework with tool support for security posture evaluation and analysis that keeps human-in-the-loop. SPEAR uses the causal formalism of AI planning to model vulnerabilities and configurations in a networked system. It automatically converts network configurations and vulnerability descriptions into planning models expressed in the Planning Domain Definition Language (PDDL). SPEAR identifies a set of diverse security hardening strategies that can be presented in a manner understandable to the domain expert. These allow the administrator to explore the network hardening solution space in a systematic fashion and help evaluate the impact and compare the different solutions. 

**Abstract (ZH)**: 基于图的框架经常被用于网络加固，以助于网络安全防御者理解网络可能受到的攻击方式以及如何部署最优防御措施。然而，将网络连通性参数纳入攻击图、在缺乏完整信息的情况下进行攻击图推理、以易于理解的格式向系统管理员提供建议、以及允许他们在各种情景和攻击动机下进行“假设性”分析等功能仍然缺失。我们通过提出SPEAR（带有人工智能规划因果形式化的安全态势评估与分析框架，具有工具支持并保持人工在环）填补这一空白。SPEAR使用人工智能规划的因果形式化方法来建模网络系统中的漏洞和配置。它会自动将网络配置和漏洞描述转换为用规划域定义语言（PDDL）表达的规划模型。SPEAR能够识别出一系列多样的安全加固策略，这些策略可以通过易于理解的方式呈现给领域专家。这些策略使管理员能够系统地探索网络加固解决方案的空间，并帮助评估影响和比较不同的解决方案。 

---
# A Review on Coarse to Fine-Grained Animal Action Recognition 

**Title (ZH)**: 粗到细粒度动物动作识别综述 

**Authors**: Ali Zia, Renuka Sharma, Abdelwahed Khamis, Xuesong Li, Muhammad Husnain, Numan Shafi, Saeed Anwar, Sabine Schmoelzl, Eric Stone, Lars Petersson, Vivien Rolland  

**Link**: [PDF](https://arxiv.org/pdf/2506.01214)  

**Abstract**: This review provides an in-depth exploration of the field of animal action recognition, focusing on coarse-grained (CG) and fine-grained (FG) techniques. The primary aim is to examine the current state of research in animal behaviour recognition and to elucidate the unique challenges associated with recognising subtle animal actions in outdoor environments. These challenges differ significantly from those encountered in human action recognition due to factors such as non-rigid body structures, frequent occlusions, and the lack of large-scale, annotated datasets. The review begins by discussing the evolution of human action recognition, a more established field, highlighting how it progressed from broad, coarse actions in controlled settings to the demand for fine-grained recognition in dynamic environments. This shift is particularly relevant for animal action recognition, where behavioural variability and environmental complexity present unique challenges that human-centric models cannot fully address. The review then underscores the critical differences between human and animal action recognition, with an emphasis on high intra-species variability, unstructured datasets, and the natural complexity of animal habitats. Techniques like spatio-temporal deep learning frameworks (e.g., SlowFast) are evaluated for their effectiveness in animal behaviour analysis, along with the limitations of existing datasets. By assessing the strengths and weaknesses of current methodologies and introducing a recently-published dataset, the review outlines future directions for advancing fine-grained action recognition, aiming to improve accuracy and generalisability in behaviour analysis across species. 

**Abstract (ZH)**: 动物动作识别领域的综述：粗粒度与细粒度技术探究 

---
# Mamba Drafters for Speculative Decoding 

**Title (ZH)**: 猜想性解码的Mamba绘图工具 

**Authors**: Daewon Choi, Seunghyuk Oh, Saket Dingliwal, Jihoon Tack, Kyuyoung Kim, Woomin Song, Seojin Kim, Insu Han, Jinwoo Shin, Aram Galstyan, Shubham Katiyar, Sravan Babu Bodapati  

**Link**: [PDF](https://arxiv.org/pdf/2506.01206)  

**Abstract**: Speculative decoding has emerged as a promising approach to accelerating large language model (LLM) generation using a fast drafter while maintaining alignment with the target model's distribution. However, existing approaches face a trade-off: external drafters offer flexibility but can suffer from slower drafting, while self-speculation methods use drafters tailored to the target model but require re-training. In this paper, we introduce novel drafters based on Mamba, a state-of-the-art state space model (SSM), as a solution that combines the best aspects of both approaches. By leveraging the linear structure of SSMs, our approach avoids the quadratic complexity inherent in traditional Transformer-based methods, enabling faster drafting and lower memory usage while maintaining the flexibility to work across different target models. We further enhance efficiency with a novel test-time tree search algorithm for generating high-quality draft candidates. Our empirical evaluation demonstrates that Mamba-based drafters not only outperform existing external drafting methods but are also comparable to state-of-the-art self-speculation approaches while using less memory and maintaining their cross-model adaptability. 

**Abstract (ZH)**: 基于Mamba的状态空间模型的推测解码：结合外部草案器的灵活性和自我推测方法的适应性 

---
# Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures 

**Title (ZH)**: 在稀疏自编码架构中融入层次化语义 

**Authors**: Mark Muchane, Sean Richardson, Kiho Park, Victor Veitch  

**Link**: [PDF](https://arxiv.org/pdf/2506.01197)  

**Abstract**: Sparse dictionary learning (and, in particular, sparse autoencoders) attempts to learn a set of human-understandable concepts that can explain variation on an abstract space. A basic limitation of this approach is that it neither exploits nor represents the semantic relationships between the learned concepts. In this paper, we introduce a modified SAE architecture that explicitly models a semantic hierarchy of concepts. Application of this architecture to the internal representations of large language models shows both that semantic hierarchy can be learned, and that doing so improves both reconstruction and interpretability. Additionally, the architecture leads to significant improvements in computational efficiency. 

**Abstract (ZH)**: 改进的语义层次结构Sparse自编码器架构：学习和利用语义层次结构以提高重构和可解释性及计算效率 

---
# OG-VLA: 3D-Aware Vision Language Action Model via Orthographic Image Generation 

**Title (ZH)**: OG-VLA: 基于正投影图像生成的三维aware视觉语言行动模型 

**Authors**: Ishika Singh, Ankit Goyal, Stan Birchfield, Dieter Fox, Animesh Garg, Valts Blukis  

**Link**: [PDF](https://arxiv.org/pdf/2506.01196)  

**Abstract**: We introduce OG-VLA, a novel architecture and learning framework that combines the generalization strengths of Vision Language Action models (VLAs) with the robustness of 3D-aware policies. We address the challenge of mapping natural language instructions and multi-view RGBD observations to quasi-static robot actions. 3D-aware robot policies achieve state-of-the-art performance on precise robot manipulation tasks, but struggle with generalization to unseen instructions, scenes, and objects. On the other hand, VLAs excel at generalizing across instructions and scenes, but can be sensitive to camera and robot pose variations. We leverage prior knowledge embedded in language and vision foundation models to improve generalization of 3D-aware keyframe policies. OG-VLA projects input observations from diverse views into a point cloud which is then rendered from canonical orthographic views, ensuring input view invariance and consistency between input and output spaces. These canonical views are processed with a vision backbone, a Large Language Model (LLM), and an image diffusion model to generate images that encode the next position and orientation of the end-effector on the input scene. Evaluations on the Arnold and Colosseum benchmarks demonstrate state-of-the-art generalization to unseen environments, with over 40% relative improvements while maintaining robust performance in seen settings. We also show real-world adaption in 3 to 5 demonstrations along with strong generalization. Videos and resources at this https URL 

**Abstract (ZH)**: OG-VLA：结合视觉语言动作模型和3D感知策略的新型架构与学习框架 

---
# Doubly Robust Alignment for Large Language Models 

**Title (ZH)**: 双重稳健对齐为大规模语言模型 

**Authors**: Erhan Xu, Kai Ye, Hongyi Zhou, Luhan Zhu, Francesco Quinzan, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01183)  

**Abstract**: This paper studies reinforcement learning from human feedback (RLHF) for aligning large language models with human preferences. While RLHF has demonstrated promising results, many algorithms are highly sensitive to misspecifications in the underlying preference model (e.g., the Bradley-Terry model), the reference policy, or the reward function, resulting in undesirable fine-tuning. To address model misspecification, we propose a doubly robust preference optimization algorithm that remains consistent when either the preference model or the reference policy is correctly specified (without requiring both). Our proposal demonstrates superior and more robust performance than state-of-the-art algorithms, both in theory and in practice. The code is available at this https URL 

**Abstract (ZH)**: 本文研究了人类反馈强化学习（RLHF）以使大规模语言模型与人类偏好一致。虽然RLHF展现出了令人鼓舞的结果，但许多算法对底层偏好模型（如Bradley-Terry模型）、参考策略或奖励函数中的错误规定非常敏感，导致不理想的模型微调。为了应对模型规定不准确的问题，我们提出了一种双重稳健的偏好优化算法，在偏好模型或参考策略中任一被正确指定的情况下（无需两者都正确）仍能保持一致性。我们的提议在理论和实践中都展现出了优于现有最佳算法的优越性和稳健性。代码可在以下链接获取。 

---
# Humanoid World Models: Open World Foundation Models for Humanoid Robotics 

**Title (ZH)**: 类人世界模型：面向类人机器人的人类世模型 

**Authors**: Muhammad Qasim Ali, Aditya Sridhar, Shahbuland Matiana, Alex Wong, Mohammad Al-Sharman  

**Link**: [PDF](https://arxiv.org/pdf/2506.01182)  

**Abstract**: Humanoid robots have the potential to perform complex tasks in human centered environments but require robust predictive models to reason about the outcomes of their actions. We introduce Humanoid World Models (HWM) a family of lightweight open source video based models that forecast future egocentric observations conditioned on actions. We train two types of generative models Masked Transformers and FlowMatching on 100 hours of humanoid demonstrations. Additionally we explore architectural variants with different attention mechanisms and parameter sharing strategies. Our parameter sharing techniques reduce model size by 33 to 53 with minimal impact on performance or visual fidelity. HWM is designed to be trained and deployed in practical academic and small lab settings such as 1 to 2 GPUs. 

**Abstract (ZH)**: humanoid 机器人在人类中心化的环境中具有执行复杂任务的潜力，但需要稳健的预测模型来推理其行为结果。我们介绍了基于动作条件预测未来第一人称观察的轻量级开源视频模型 Humanoid 世界模型（HWM）。我们在100小时的 humanoid 示范数据上训练了两种生成模型——Masked Transformers 和 FlowMatching。此外，我们还探索了具有不同注意力机制和参数共享策略的架构变体。我们的参数共享技术将模型大小减少33%至53%，同时对性能和视觉保真度的影响极小。HWM 旨在适应1到2块GPU的实用学术和小型实验室环境进行训练和部署。 

---
# Bridging Quantum and Classical Computing in Drug Design: Architecture Principles for Improved Molecule Generation 

**Title (ZH)**: 在药物设计中跨越量子与经典计算的桥梁：提高分子生成的架构原则 

**Authors**: Andrew Smith, Erhan Guven  

**Link**: [PDF](https://arxiv.org/pdf/2506.01177)  

**Abstract**: Hybrid quantum-classical machine learning offers a path to leverage noisy intermediate-scale quantum (NISQ) devices for drug discovery, but optimal model architectures remain unclear. We systematically optimize the quantum-classical bridge architecture for generative adversarial networks (GANs) in molecular discovery using multi-objective Bayesian optimization. Our optimized model (BO-QGAN) significantly improves performance, achieving a 2.27-fold higher Drug Candidate Score (DCS) than prior quantum-hybrid benchmarks and 2.21-fold higher than the classical baseline, using over 60% fewer parameters. Key findings favor layering multiple (3-4) shallow (4-8 qubit) quantum circuits sequentially, while classical architecture shows less sensitivity above a minimum capacity. This work provides the first empirically grounded architectural guidelines for hybrid models, enabling more effective integration of current quantum computers into pharmaceutical research pipelines. 

**Abstract (ZH)**: 混合量子-经典机器学习为利用噪声中间规模量子(NISQ)设备进行药物发现提供了路径，但最优模型架构仍不明确。我们通过多目标贝叶斯优化系统地优化了用于分子发现的生成对抗网络(GANs)的量子-经典桥梁架构。我们的优化模型(BO-QGAN)显著提高了性能，与之前的量子-混合基准相比，药物候选评分(DCS)提高了2.27倍，与经典基线相比提高了2.21倍，同时使用了超过60%更少的参数。关键发现倾向于依次堆叠多个(3-4个)浅层(4-8个量子比特)量子电路，而经典架构在最小容量以上显示出较低的敏感性。本项工作提供了首个经验性的混合模型架构指南，有助于更有效地将当前的量子计算机集成到制药研究管道中。 

---
# VUSA: Virtually Upscaled Systolic Array Architecture to Exploit Unstructured Sparsity in AI Acceleration 

**Title (ZH)**: VUSA：利用非结构化稀疏性加速AI加速的虚拟上尺度 systolic 数组架构 

**Authors**: Shereef Helal, Alberto Garcia-Ortiz, Lennart Bamberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.01166)  

**Abstract**: Leveraging high degrees of unstructured sparsity is a promising approach to enhance the efficiency of deep neural network DNN accelerators - particularly important for emerging Edge-AI applications. We introduce VUSA, a systolic-array architecture that virtually grows based on the present sparsity to perform larger matrix multiplications with the same number of physical multiply-accumulate MAC units. The proposed architecture achieves saving by 37% and 68% in area and power efficiency, respectively, at the same peak-performance, compared to a baseline systolic array architecture in a commercial 16-nm technology. Still, the proposed architecture supports acceleration for any DNN with any sparsity - even no sparsity at all. Thus, the proposed architecture is application-independent, making it viable for general-purpose AI acceleration. 

**Abstract (ZH)**: 利用高程度的无序稀疏性提升深度神经网络DNN加速器的效率：特别是在新兴边缘AI应用中尤为重要。我们介绍了一种基于现稀疏性虚拟扩展的 systolic-array 架构VUSA，该架构能够在相同数量的物理乘累加MAC单元下进行更大的矩阵乘法运算。与商用16nm工艺下的基准 systolic array 架构相比，所提出的架构在相同峰值性能下，分别实现了37%和68%的面积和功率效率节省。此外，所提出的架构支持任何带稀疏性的DNN加速，即使完全不带稀疏性也不例外。因此，该架构具有应用程序独立性，适用于通用AI加速。 

---
# FORT: Forward-Only Regression Training of Normalizing Flows 

**Title (ZH)**: FORT: 前向Only回归训练归一化流 

**Authors**: Danyal Rehman, Oscar Davis, Jiarui Lu, Jian Tang, Michael Bronstein, Yoshua Bengio, Alexander Tong, Avishek Joey Bose  

**Link**: [PDF](https://arxiv.org/pdf/2506.01158)  

**Abstract**: Simulation-free training frameworks have been at the forefront of the generative modelling revolution in continuous spaces, leading to neural dynamical systems that encompass modern large-scale diffusion and flow matching models. Despite the scalability of training, the generation of high-quality samples and their corresponding likelihood under the model requires expensive numerical simulation -- inhibiting adoption in numerous scientific applications such as equilibrium sampling of molecular systems. In this paper, we revisit classical normalizing flows as one-step generative models with exact likelihoods and propose a novel, scalable training objective that does not require computing the expensive change of variable formula used in conventional maximum likelihood training. We propose Forward-Only Regression Training (FORT), a simple $\ell_2$-regression objective that maps prior samples under our flow to specifically chosen targets. We demonstrate that FORT supports a wide class of targets, such as optimal transport targets and targets from pre-trained continuous-time normalizing flows (CNF). We further demonstrate that by using CNF targets, our one-step flows allow for larger-scale training that exceeds the performance and stability of maximum likelihood training, while unlocking a broader class of architectures that were previously challenging to train. Empirically, we elucidate that our trained flows can perform equilibrium conformation sampling in Cartesian coordinates of alanine dipeptide, alanine tripeptide, and alanine tetrapeptide. 

**Abstract (ZH)**: 无模拟训练框架在连续空间生成模型革命中居于前沿，引领了现代大规模扩散和流动匹配模型的神经动力系统。尽管训练具有扩展性，但生成高质量样本及其在模型下的精确概率测量仍需昂贵的数值模拟，这限制了其在分子系统平衡采样等众多科学应用中的采用。在本文中，我们重新审视经典归一化流作为单步生成模型，并提出一种新的、可扩展的训练目标，该目标无需计算传统极大似然训练中所使用的昂贵变量替换公式。我们提出了前向回归训练（Forward-Only Regression Training，FORT），这是一种简单的$\ell_2$-回归目标，将我们的流下的先验样本映射到特定选择的目标。我们证明FORT支持广泛的目标类型，如最优传输目标和预训练连续时间归一化流（CNF）的目标。进一步地，我们证明通过使用CNF目标，我们的单步流可以实现比极大似然训练更大的训练规模，超越其性能和稳定性，并解锁了以前难以训练的一系列架构。实证研究表明，我们的训练流可以在Alanine二肽、三肽和四肽的笛卡尔坐标中执行平衡构象采样。 

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
# Neuro-Symbolic Generative Diffusion Models for Physically Grounded, Robust, and Safe Generation 

**Title (ZH)**: 基于物理约束的神经符号生成扩散模型：稳健且安全的内容生成 

**Authors**: Jacob K. Christopher, Michael Cardei, Jinhao Liang, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2506.01121)  

**Abstract**: Despite the remarkable generative capabilities of diffusion models, their integration into safety-critical or scientifically rigorous applications remains hindered by the need to ensure compliance with stringent physical, structural, and operational constraints. To address this challenge, this paper introduces Neuro-Symbolic Diffusion (NSD), a novel framework that interleaves diffusion steps with symbolic optimization, enabling the generation of certifiably consistent samples under user-defined functional and logic constraints. This key feature is provided for both standard and discrete diffusion models, enabling, for the first time, the generation of both continuous (e.g., images and trajectories) and discrete (e.g., molecular structures and natural language) outputs that comply with constraints. This ability is demonstrated on tasks spanning three key challenges: (1) Safety, in the context of non-toxic molecular generation and collision-free trajectory optimization; (2) Data scarcity, in domains such as drug discovery and materials engineering; and (3) Out-of-domain generalization, where enforcing symbolic constraints allows adaptation beyond the training distribution. 

**Abstract (ZH)**: 尽管扩散模型具有显著的生成能力，但将其整合到安全关键或科学严谨的应用中依然受限于确保满足严格的物理、结构和操作约束的需求。为应对这一挑战，本文引入了神经符号扩散（NSD）框架，该框架将扩散步骤与符号优化交错进行，使用户能够在用户定义的函数和逻辑约束下生成认证一致的样本。这一关键特性适用于标准和离散的扩散模型，首次实现了在满足约束条件下的连续输出（例如，图像和轨迹）和离散输出（例如，分子结构和自然语言）的生成。该能力在三个关键挑战任务中得以展示：（1）安全性，特别是在非毒性分子生成和无障碍轨迹优化的背景下；（2）数据稀缺性，特别是在药物发现和材料工程等领域；（3）域外泛化，通过对符号约束的强制执行实现训练分布之外的适应。 

---
# Reconsidering LLM Uncertainty Estimation Methods in the Wild 

**Title (ZH)**: 重新审视实际应用中的大规模语言模型不确定性估计方法 

**Authors**: Yavuz Bakman, Duygu Nur Yaldiz, Sungmin Kang, Tuo Zhang, Baturalp Buyukates, Salman Avestimehr, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.01114)  

**Abstract**: Large Language Model (LLM) Uncertainty Estimation (UE) methods have become a crucial tool for detecting hallucinations in recent years. While numerous UE methods have been proposed, most existing studies evaluate them in isolated short-form QA settings using threshold-independent metrics such as AUROC or PRR. However, real-world deployment of UE methods introduces several challenges. In this work, we systematically examine four key aspects of deploying UE methods in practical settings. Specifically, we assess (1) the sensitivity of UE methods to decision threshold selection, (2) their robustness to query transformations such as typos, adversarial prompts, and prior chat history, (3) their applicability to long-form generation, and (4) strategies for handling multiple UE scores for a single query. Our evaluations on 19 UE methods reveal that most of them are highly sensitive to threshold selection when there is a distribution shift in the calibration dataset. While these methods generally exhibit robustness against previous chat history and typos, they are significantly vulnerable to adversarial prompts. Additionally, while existing UE methods can be adapted for long-form generation through various strategies, there remains considerable room for improvement. Lastly, ensembling multiple UE scores at test time provides a notable performance boost, which highlights its potential as a practical improvement strategy. Code is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型不确定性估计方法在实际部署中的关键考量 

---
# FusionAudio-1.2M: Towards Fine-grained Audio Captioning with Multimodal Contextual Fusion 

**Title (ZH)**: FusionAudio-1.2M：面向细粒度音频标注的多模态上下文融合 

**Authors**: Shunian Chen, Xinyuan Xie, Zheshu Chen, Liyan Zhao, Owen Lee, Zhan Su, Qilin Sun, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01111)  

**Abstract**: High-quality, large-scale audio captioning is crucial for advancing audio understanding, yet current automated methods often generate captions that lack fine-grained detail and contextual accuracy, primarily due to their reliance on limited unimodal or superficial multimodal information. Drawing inspiration from human auditory perception, which adeptly integrates cross-modal cues and performs sophisticated auditory scene analysis, we introduce a novel two-stage automated pipeline. This pipeline first employs specialized pretrained models to extract diverse contextual cues (e.g., speech, music, general sounds, and visual information from associated video). A large language model (LLM) then synthesizes these rich, multimodal inputs to generate detailed and context-aware audio captions. Key contributions of this work include: (1) the proposed scalable method for fine-grained audio caption generation; (2) FusionAudio, a new large-scale dataset comprising 1.2 million such detailed captions, combined with 6 million QA pairs; and (3) enhanced audio models developed using FusionAudio, specifically a CLAP-based audio encoder with superior audio-text alignment and instruction following. This paper paves the way for more nuanced and accurate automated understanding of complex audio environments. Code and data can be found in this https URL. 

**Abstract (ZH)**: 高质量、大规模的音频描述对于推进音频理解至关重要，然而当前的自动化方法往往生成缺乏细致细节和上下文准确性的描述，主要原因是它们依赖于有限的单模态或表面化的多模态信息。借鉴人类听觉感知能够巧妙整合跨模态线索并执行复杂的听觉场景分析，我们提出了一种新颖的两阶段自动化管道。该管道首先使用专门的预训练模型提取多样的上下文线索（例如，语音、音乐、一般声音以及关联视频中的视觉信息）。之后，大规模语言模型（LLM）综合这些丰富的多模态输入生成详细且上下文相关的音频描述。本文的关键贡献包括：（1）提出的一种 scalable 的细粒度音频描述生成方法；（2）FusionAudio，一个包含 120 万条此类详细描述的新大规模数据集，结合了 600 万对 QA；（3）利用 FusionAudio 开发的增强音频模型，特别是基于 CLAP 的音频编码器，其具有更优的音频-文本对齐和指令跟随能力。本文为更细致和准确地理解复杂音频环境开辟了路径。代码和数据可在此处获取。 

---
# CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting 

**Title (ZH)**: 水果计数:基于语言引导语义高斯点绘制的实时3D水果计数 

**Authors**: Fengze Li, Yangle Liu, Jieming Ma, Hai-Ning Liang, Yaochun Shen, Huangxiang Li, Zhijing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01109)  

**Abstract**: Accurate fruit counting in real-world agricultural environments is a longstanding challenge due to visual occlusions, semantic ambiguity, and the high computational demands of 3D reconstruction. Existing methods based on neural radiance fields suffer from low inference speed, limited generalization, and lack support for open-set semantic control. This paper presents FruitLangGS, a real-time 3D fruit counting framework that addresses these limitations through spatial reconstruction, semantic embedding, and language-guided instance estimation. FruitLangGS first reconstructs orchard-scale scenes using an adaptive Gaussian splatting pipeline with radius-aware pruning and tile-based rasterization for efficient rendering. To enable semantic control, each Gaussian encodes a compressed CLIP-aligned language embedding, forming a compact and queryable 3D representation. At inference time, prompt-based semantic filtering is applied directly in 3D space, without relying on image-space segmentation or view-level fusion. The selected Gaussians are then converted into dense point clouds via distribution-aware sampling and clustered to estimate fruit counts. Experimental results on real orchard data demonstrate that FruitLangGS achieves higher rendering speed, semantic flexibility, and counting accuracy compared to prior approaches, offering a new perspective for language-driven, real-time neural rendering across open-world scenarios. 

**Abstract (ZH)**: 实时农业环境中基于语义引导的三维水果计数框架 FruitLangGS 

---
# Speeding Up Hyper-Heuristics With Markov-Chain Operator Selection and the Only-Worsening Acceptance Operator 

**Title (ZH)**: 使用马尔科夫链操作选择和仅恶化接受操作加速超启发式算法 

**Authors**: Abderrahim Bendahi, Benjamin Doerr, Adrien Fradin, Johannes F. Lutzeyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.01107)  

**Abstract**: The move-acceptance hyper-heuristic was recently shown to be able to leave local optima with astonishing efficiency (Lissovoi et al., Artificial Intelligence (2023)). In this work, we propose two modifications to this algorithm that demonstrate impressive performances on a large class of benchmarks including the classic Cliff$_d$ and Jump$_m$ function classes. (i) Instead of randomly choosing between the only-improving and any-move acceptance operator, we take this choice via a simple two-state Markov chain. This modification alone reduces the runtime on Jump$_m$ functions with gap parameter $m$ from $\Omega(n^{2m-1})$ to $O(n^{m+1})$. (ii) We then replace the all-moves acceptance operator with the operator that only accepts worsenings. Such a, counter-intuitive, operator has not been used before in the literature. However, our proofs show that our only-worsening operator can greatly help in leaving local optima, reducing, e.g., the runtime on Jump functions to $O(n^3 \log n)$ independent of the gap size. In general, we prove a remarkably good runtime of $O(n^{k+1} \log n)$ for our Markov move-acceptance hyper-heuristic on all members of a new benchmark class SEQOPT$_k$, which contains a large number of functions having $k$ successive local optima, and which contains the commonly studied Jump$_m$ and Cliff$_d$ functions for $k=2$. 

**Abstract (ZH)**: -move接受超启发式算法 recently 展示了其以惊人效率离开局部最优解的能力 (Lissovoi 等人, 人工智能 (2023))。在这项工作中，我们对该算法提出了两项修改，这些修改在包括经典 Cliff$_d$ 和 Jump$_m$ 函数类在内的大量基准测试中表现出色。(i) 与随机选择仅改进操作符和任何操作符的接受操作符不同，我们通过简单的两状态马尔可夫链来做出这种选择。这一修改将 Gap 参数为 $m$ 的 Jump$_m$ 函数的运行时从 $\Omega(n^{2m-1})$ 降低到 $O(n^{m+1})$。(ii) 然后，我们用只接受恶化操作符替代所有操作符的接受操作符。这种看似反直观的操作符还未曾在文献中使用过。然而，我们的证明表明，我们仅恶化操作符可以极大地帮助离开局部最优解，例如，将 Jump 函数的运行时减少到 $O(n^3 \log n)$，与 Gap 大小无关。一般来说，我们证明了对于新基准类 SEQOPT$_k$ 中的所有成员，我们的马尔可夫移动接受超启发式算法具有显著良好的运行时 $O(n^{k+1} \log n)$，该基准类包含许多具有 $k$ 个连续局部最优解的函数，并包含常见的 Jump$_m$ 和 Cliff$_d$ 函数（对于 $k=2$）。 

---
# Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements 

**Title (ZH)**: 忽视语境信息：评估LLMs对指称元素的理解 

**Authors**: Metehan Oguz, Yavuz Bakman, Duygu Nur Yaldiz  

**Link**: [PDF](https://arxiv.org/pdf/2506.01089)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performances in tasks related to coreference resolution. However, previous studies mostly assessed LLM performance on coreference resolution with nouns and third person pronouns. This study evaluates LLM performance on coreference resolution with indexical like I, you, here and tomorrow, which come with unique challenges due to their linguistic properties. We present the first study examining how LLMs interpret indexicals in English, releasing the English Indexical Dataset with 1600 multiple-choice questions. We evaluate pioneering LLMs, including GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and DeepSeek V3. Our results reveal that LLMs exhibit an impressive performance with some indexicals (I), while struggling with others (you, here, tomorrow), and that syntactic cues (e.g. quotation) contribute to LLM performance with some indexicals, while they reduce performance with others. Code and data are available at: this https URL. 

**Abstract (ZH)**: 大型语言模型在处理代词理解任务中的表现：以“I”、“you”、“here”和“tomorrow”为例 

---
# Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection 

**Title (ZH)**: 优先学习重要概念：基于相对误差驱动的样本选择概念学习 

**Authors**: Shivam Chandhok, Qian Yang, Oscar Manas, Kanishk Jain, Leonid Sigal, Aishwarya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.01085)  

**Abstract**: Instruction tuning has been central to the success of recent vision-language models (VLMs), but it remains expensive-requiring large-scale datasets, high-quality annotations, and large compute budgets. We propose PRioritized cOncept learninG via Relative Error-driven Sample Selection (PROGRESS), a data- and compute-efficient framework that enables VLMs to dynamically select what to learn next based on their evolving needs during training. At each stage, the model tracks its learning progress across skills and selects the most informative samples-those it has not already mastered and that are not too difficult to learn at the current stage of training. This strategy effectively controls skill acquisition and the order in which skills are learned. Specifically, we sample from skills showing the highest learning progress, prioritizing those with the most rapid improvement. Unlike prior methods, PROGRESS requires no upfront answer annotations, queries answers only on a need basis, avoids reliance on additional supervision from auxiliary VLMs, and does not require compute-heavy gradient computations for data selection. Experiments across multiple instruction-tuning datasets of varying scales demonstrate that PROGRESS consistently outperforms state-of-the-art baselines with much less data and supervision. Additionally, we show strong cross-architecture generalization and transferability to larger models, validating PROGRESS as a scalable solution for efficient learning. 

**Abstract (ZH)**: PRrioritized cOncept lEarning via RElative EError-driven SAmple SElction (PRogress) 

---
# Unfolding Boxes with Local Constraints 

**Title (ZH)**: 带有局部约束的展开箱体问题 

**Authors**: Long Qian, Eric Wang, Bernardo Subercaseaux, Marijn J. H. Heule  

**Link**: [PDF](https://arxiv.org/pdf/2506.01079)  

**Abstract**: We consider the problem of finding and enumerating polyominos that can be folded into multiple non-isomorphic boxes. While several computational approaches have been proposed, including SAT, randomized algorithms, and decision diagrams, none has been able to perform at scale. We argue that existing SAT encodings are hindered by the presence of global constraints (e.g., graph connectivity or acyclicity), which are generally hard to encode effectively and hard for solvers to reason about. In this work, we propose a new SAT-based approach that replaces these global constraints with simple local constraints that have substantially better propagation properties. Our approach dramatically improves the scalability of both computing and enumerating common box unfoldings: (i) while previous approaches could only find common unfoldings of two boxes up to area 88, ours easily scales beyond 150, and (ii) while previous approaches were only able to enumerate common unfoldings up to area 30, ours scales up to 60. This allows us to rule out 46, 54, and 58 as the smallest areas allowing a common unfolding of three boxes, thereby refuting a conjecture of Xu et al. (2017). 

**Abstract (ZH)**: 我们考虑寻找和枚举可以折叠成多个非同构盒子的多米诺骨牌的问题。尽管已经提出了几种计算方法，包括SAT、随机化算法和决策图，但 none 未能在大规模应用中发挥作用。我们argue 存在的 SAT 编码受到全局约束（例如，图连通性或无环性）的阻碍，这些约束通常难以有效编码且难以供求解器推理。在本文中，我们提出了一种新的基于 SAT 的方法，用简单的局部约束替换这些全局约束，从而具有更好的传播性质。我们的方法显著提高了计算和枚举常见盒子展开图的可扩展性：(i) 而且前人方法只能找到两盒面积不超过 88 的常见展开图，我们的方法轻松扩展到 150 以上；(ii) 而且前人方法只能枚举面积不超过 30 的常见展开图，我们的方法扩展到 60。这使得我们可以排除 46、54 和 58 作为三个盒子共有展开图的最小面积，从而反驳了 Xu 等人 (2017) 的猜想。 

---
# GThinker: Towards General Multimodal Reasoning via Cue-Guided Rethinking 

**Title (ZH)**: GThinker: 向 towards 通用多模态推理 via 基于提示的重组 Cue-Guided Rethinking 

**Authors**: Yufei Zhan, Ziheng Wu, Yousong Zhu, Rongkun Xue, Ruipu Luo, Zhenghao Chen, Can Zhang, Yifan Li, Zhentao He, Zheming Yang, Ming Tang, Minghui Qiu, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01078)  

**Abstract**: Despite notable advancements in multimodal reasoning, leading Multimodal Large Language Models (MLLMs) still underperform on vision-centric multimodal reasoning tasks in general scenarios. This shortfall stems from their predominant reliance on logic- and knowledge-based slow thinking strategies, while effective for domains like math and science, fail to integrate visual information effectively during reasoning. Consequently, these models often fail to adequately ground visual cues, resulting in suboptimal performance in tasks that require multiple plausible visual interpretations and inferences. To address this, we present GThinker (General Thinker), a novel reasoning MLLM excelling in multimodal reasoning across general scenarios, mathematics, and science. GThinker introduces Cue-Rethinking, a flexible reasoning pattern that grounds inferences in visual cues and iteratively reinterprets these cues to resolve inconsistencies. Building on this pattern, we further propose a two-stage training pipeline, including pattern-guided cold start and incentive reinforcement learning, designed to enable multimodal reasoning capabilities across domains. Furthermore, to support the training, we construct GThinker-11K, comprising 7K high-quality, iteratively-annotated reasoning paths and 4K curated reinforcement learning samples, filling the data gap toward general multimodal reasoning. Extensive experiments demonstrate that GThinker achieves 81.5% on the challenging comprehensive multimodal reasoning benchmark M$^3$CoT, surpassing the latest O4-mini model. It also shows an average improvement of 2.1% on general scenario multimodal reasoning benchmarks, while maintaining on-par performance in mathematical reasoning compared to counterpart advanced reasoning models. The code, model, and data will be released soon at this https URL. 

**Abstract (ZH)**: 尽管在多模态推理方面取得了显著进展，主流的多模态大型语言模型（MLLMs）在一般场景下的视觉中心多模态推理任务中仍然表现不佳。这一缺陷源于它们主要依赖于逻辑和知识为基础的慢思考策略，这些策略在数学和科学等领域有效，但无法有效地整合推理过程中的视觉信息。因此，这些模型往往无法充分地将视觉线索纳入推理中，导致在需要多个合理的视觉解释和推理的任务中表现不佳。为了解决这一问题，我们提出了GThinker（通用思考者），一种在一般场景、数学和科学领域都擅长多模态推理的新颖推理MLLM。GThinker引入了基于线索重新思考（Cue-Rethinking）的灵活推理模式，该模式以视觉线索为基础，迭代重新解释这些线索以解决不一致问题。基于这一模式，我们进一步提出了一种两阶段训练管道，包括模式引导的冷启动和激励强化学习，旨在促进跨领域的多模态推理能力。此外，为了支持训练，我们构建了包含7000条高质量、迭代注释的推理路径和4000个精心挑选的强化学习样本的GThinker-11K，以填补通用多模态推理的数据缺口。广泛的经验表明，GThinker在具有挑战性的综合多模态推理基准M$^3$CoT上达到了81.5%的得分，超越了最新的O4-mini模型。同时，它在一般场景下的多模态推理基准测试中的平均表现提高了2.1%，而在数学推理方面与同类先进的推理模型保持了相当的性能。代码、模型和数据将在不久后在此网址发布。 

---
# Revolutionizing Blood Banks: AI-Driven Fingerprint-Blood Group Correlation for Enhanced Safety 

**Title (ZH)**: 革新血液银行：基于人工智能的指纹-血型关联技术以提高安全性能 

**Authors**: Malik A. Altayar, Muhyeeddin Alqaraleh, Mowafaq Salem Alzboon, Wesam T. Almagharbeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.01069)  

**Abstract**: Identification of a person is central in forensic science, security, and healthcare. Methods such as iris scanning and genomic profiling are more accurate but expensive, time-consuming, and more difficult to implement. This study focuses on the relationship between the fingerprint patterns and the ABO blood group as a biometric identification tool. A total of 200 subjects were included in the study, and fingerprint types (loops, whorls, and arches) and blood groups were compared. Associations were evaluated with statistical tests, including chi-square and Pearson correlation. The study found that the loops were the most common fingerprint pattern and the O+ blood group was the most prevalent. Even though there was some associative pattern, there was no statistically significant difference in the fingerprint patterns of different blood groups. Overall, the results indicate that blood group data do not significantly improve personal identification when used in conjunction with fingerprinting. Although the study shows weak correlation, it may emphasize the efforts of multi-modal based biometric systems in enhancing the current biometric systems. Future studies may focus on larger and more diverse samples, and possibly machine learning and additional biometrics to improve identification methods. This study addresses an element of the ever-changing nature of the fields of forensic science and biometric identification, highlighting the importance of resilient analytical methods for personal identification. 

**Abstract (ZH)**: 指纹模式与ABO血型在生物识别身份认证中的关系研究 

---
# Trilevel Memetic Algorithm for the Electric Vehicle Routing Problem 

**Title (ZH)**: 三级遗传算法求解电动汽车路由问题 

**Authors**: Ivan Milinović, Leon Stjepan Uroić, Marko Đurasević  

**Link**: [PDF](https://arxiv.org/pdf/2506.01065)  

**Abstract**: The Electric Vehicle Routing Problem (EVRP) extends the capacitated vehicle routing problem by incorporating battery constraints and charging stations, posing significant optimization challenges. This paper introduces a Trilevel Memetic Algorithm (TMA) that hierarchically optimizes customer sequences, route assignments, and charging station insertions. The method combines genetic algorithms with dynamic programming, ensuring efficient and high-quality solutions. Benchmark tests on WCCI2020 instances show competitive performance, matching best-known results for small-scale cases. While computational demands limit scalability, TMA demonstrates strong potential for sustainable logistics planning. 

**Abstract (ZH)**: 三层次 memetic 算法求解考虑电池约束的电动车辆路由问题 

---
# Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs 

**Title (ZH)**: 以火攻火（F3）：一种无需训练且高效的视觉对抗样本净化方法在LVLMs中的应用 

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01064)  

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. Despite their potential impact, the development of effective methods for purifying such adversarial examples has received relatively limited attention. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive "fighting fire with fire" strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code will be made publicly available. 

**Abstract (ZH)**: 最近大型多模态视觉语言模型的进展展示了其在多种跨模态视觉语言任务中的出色能力。然而，这些模型仍然容易受到视觉对抗攻击的影响，这可能会显著损害其性能。尽管对抗攻击可能具有重大影响，但开发有效的对抗样本净化方法仍相对受到较少的关注。在本文中，我们介绍了F3，一种新颖的对抗样本净化框架，采用了反直觉的“以火攻火”策略：故意引入简单的扰动到对抗样本中以减轻其有害影响。具体而言，F3 利用来自随机扰动对手样本的跨模态注意作为参考目标。通过向这些对抗样本注入噪声，F3 有效改进了它们的注意机制，从而产生更清洁和更可靠的模型输出。令人惊讶的是，这种看似矛盾的方法——通过引入噪声来对抗对抗攻击——取得了令人印象深刻的净化效果。此外，F3 还提供了几个显著的优势：它无需训练且易于实现，并且在计算效率方面比现有净化方法表现出明显的改进。这些特性使得 F3 特别适用于大规模工业应用，其中稳健性能和操作效率是至关重要的优先事项。代码将公开发布。 

---
# SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models 

**Title (ZH)**: SealQA：提高基于搜索的语言模型推理标准 

**Authors**: Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01062)  

**Abstract**: We introduce SealQA, a new challenge benchmark for evaluating SEarch-Augmented Language models on fact-seeking questions where web search yields conflicting, noisy, or unhelpful results. SealQA comes in three flavors: (1) Seal-0 (main) and (2) Seal-Hard, which assess factual accuracy and reasoning capabilities, with Seal-0 focusing on the most challenging questions where chat models (e.g., GPT-4.1) typically achieve near-zero accuracy; and (3) LongSeal, which extends SealQA to test long-context, multi-document reasoning in "needle-in-a-haystack" settings. Our evaluation reveals critical limitations in current models: Even frontier LLMs perform poorly across all SealQA flavors. On Seal-0, frontier agentic models equipped with tools like o3 and o4-mini achieve only 17.1% and 6.3% accuracy, respectively, at their best reasoning efforts. We find that advanced reasoning models such as DeepSeek-R1-671B and o3-mini are highly vulnerable to noisy search results. Notably, increasing test-time compute does not yield reliable gains across o3-mini, o4-mini, and o3, with performance often plateauing or even declining early. Additionally, while recent models are less affected by the "lost-in-the-middle" issue, they still fail to reliably identify relevant documents in LongSeal when faced with numerous distractors. To facilitate future work, we release SealQA at this http URL. 

**Abstract (ZH)**: SealQA：一种新的挑战基准，用于评估基于网络搜索的语言模型在事实查询中的性能，特别是在搜索结果矛盾、噪音大或无用的情况下。 

---
# XAI-Units: Benchmarking Explainability Methods with Unit Tests 

**Title (ZH)**: XAI-Units：基于单元测试的可解释性方法基准测试 

**Authors**: Jun Rui Lee, Sadegh Emami, Michael David Hollins, Timothy C. H. Wong, Carlos Ignacio Villalobos Sánchez, Francesca Toni, Dekai Zhang, Adam Dejl  

**Link**: [PDF](https://arxiv.org/pdf/2506.01059)  

**Abstract**: Feature attribution (FA) methods are widely used in explainable AI (XAI) to help users understand how the inputs of a machine learning model contribute to its outputs. However, different FA models often provide disagreeing importance scores for the same model. In the absence of ground truth or in-depth knowledge about the inner workings of the model, it is often difficult to meaningfully determine which of the different FA methods produce more suitable explanations in different contexts. As a step towards addressing this issue, we introduce the open-source XAI-Units benchmark, specifically designed to evaluate FA methods against diverse types of model behaviours, such as feature interactions, cancellations, and discontinuous outputs. Our benchmark provides a set of paired datasets and models with known internal mechanisms, establishing clear expectations for desirable attribution scores. Accompanied by a suite of built-in evaluation metrics, XAI-Units streamlines systematic experimentation and reveals how FA methods perform against distinct, atomic kinds of model reasoning, similar to unit tests in software engineering. Crucially, by using procedurally generated models tied to synthetic datasets, we pave the way towards an objective and reliable comparison of FA methods. 

**Abstract (ZH)**: 开放源代码的XAI-Units基准：用于评估特征归因方法的多样性模型行为 

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
# A Two-Stage Hierarchical Deep Filtering Framework for Real-Time Speech Enhancement 

**Title (ZH)**: 两级分层深度滤波框架实现实时语音增强 

**Authors**: Shenghui Lu, Hukai Huang, Jinanglong Yao, Kaidi Wang, Qingyang Hong, Lin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01023)  

**Abstract**: This paper proposes a model that integrates sub-band processing and deep filtering to fully exploit information from the target time-frequency (TF) bin and its surrounding TF bins for single-channel speech enhancement. The sub-band module captures surrounding frequency bin information at the input, while the deep filtering module applies filtering at the output to both the target TF bin and its surrounding TF bins. To further improve the model performance, we decouple deep filtering into temporal and frequency components and introduce a two-stage framework, reducing the complexity of filter coefficient prediction at each stage. Additionally, we propose the TAConv module to strengthen convolutional feature extraction. Experimental results demonstrate that the proposed hierarchical deep filtering network (HDF-Net) effectively utilizes surrounding TF bin information and outperforms other advanced systems while using fewer resources. 

**Abstract (ZH)**: 本文提出了一种将子带处理与深度滤波相结合的模型，以充分利用目标时频(TF) bins及其周围TF bins中的信息，用于单通道语音增强。子带模块在输入端捕获周围的频率bins信息，而深度滤波模块在输出端对目标TF bin及其周围TF bins进行滤波。为了进一步提高模型性能，我们将深度滤波分解为时间域和频域组件，并引入两阶段框架，降低每阶段滤波系数预测的复杂度。此外，本文提出TAConv模块以增强卷积特征提取。实验结果表明，所提出的分层深度滤波网络(HDF-Net)有效地利用了周围TF bins中的信息，并在资源较少的情况下优于其他先进的系统。 

---
# Motion-Aware Concept Alignment for Consistent Video Editing 

**Title (ZH)**: 基于运动感知的概念对齐以实现一致的视频编辑 

**Authors**: Tong Zhang, Juan C Leon Alcazar, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2506.01004)  

**Abstract**: We introduce MoCA-Video (Motion-Aware Concept Alignment in Video), a training-free framework bridging the gap between image-domain semantic mixing and video. Given a generated video and a user-provided reference image, MoCA-Video injects the semantic features of the reference image into a specific object within the video, while preserving the original motion and visual context. Our approach leverages a diagonal denoising schedule and class-agnostic segmentation to detect and track objects in the latent space and precisely control the spatial location of the blended objects. To ensure temporal coherence, we incorporate momentum-based semantic corrections and gamma residual noise stabilization for smooth frame transitions. We evaluate MoCA's performance using the standard SSIM, image-level LPIPS, temporal LPIPS, and introduce a novel metric CASS (Conceptual Alignment Shift Score) to evaluate the consistency and effectiveness of the visual shifts between the source prompt and the modified video frames. Using self-constructed dataset, MoCA-Video outperforms current baselines, achieving superior spatial consistency, coherent motion, and a significantly higher CASS score, despite having no training or fine-tuning. MoCA-Video demonstrates that structured manipulation in the diffusion noise trajectory allows for controllable, high-quality video synthesis. 

**Abstract (ZH)**: MoCA-Video：基于运动意识的概念对齐视频生成框架 

---
# Quotient Network - A Network Similar to ResNet but Learning Quotients 

**Title (ZH)**: 商网络 - 一种类似于ResNet的网络，学习商值 

**Authors**: Peng Hui, Jiamuyang Zhao, Changxin Li, Qingzhen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00992)  

**Abstract**: The emergence of ResNet provides a powerful tool for training extremely deep networks. The core idea behind it is to change the learning goals of the network. It no longer learns new features from scratch but learns the difference between the target and existing features. However, the difference between the two kinds of features does not have an independent and clear meaning, and the amount of learning is based on the absolute rather than the relative difference, which is sensitive to the size of existing features. We propose a new network that perfectly solves these two problems while still having the advantages of ResNet. Specifically, it chooses to learn the quotient of the target features with the existing features, so we call it the quotient network. In order to enable this network to learn successfully and achieve higher performance, we propose some design rules for this network so that it can be trained efficiently and achieve better performance than ResNet. Experiments on the CIFAR10, CIFAR100, and SVHN datasets prove that this network can stably achieve considerable improvements over ResNet by simply making tiny corresponding changes to the original ResNet network without adding new parameters. 

**Abstract (ZH)**: ResNet的出现为训练极深网络提供了强大工具。其核心理念是改变网络的学习目标。它不再从零开始学习新的特征，而是学习目标与现有特征之间的差异。然而，这两种特征之间的差异缺乏独立和明确的意义，并且学习量基于绝对差异而非相对差异，这使其对现有特征的大小高度敏感。我们提出了一种新网络，完美解决了这两个问题的同时保持了ResNet的优点。具体地，它选择学习目标特征与现有特征的比值，因此我们称其为商网络。为了使该网络能够成功学习并实现更高的性能，我们为此网络提出了一些设计规则，以便它可以高效地训练并在性能上超越ResNet。实验表明，仅对原始ResNet网络进行微小的相应修改即可在CIFAR10、CIFAR100和SVHN数据集上稳定地获得显著改进，而无需增加新的参数。 

---
# Bridging the Gap: From Ad-hoc to Proactive Search in Conversations 

**Title (ZH)**: 弥补差距：从临时搜索到主动搜索在对话中的应用 

**Authors**: Chuan Meng, Francesco Tonolini, Fengran Mo, Nikolaos Aletras, Emine Yilmaz, Gabriella Kazai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00983)  

**Abstract**: Proactive search in conversations (PSC) aims to reduce user effort in formulating explicit queries by proactively retrieving useful relevant information given conversational context. Previous work in PSC either directly uses this context as input to off-the-shelf ad-hoc retrievers or further fine-tunes them on PSC data. However, ad-hoc retrievers are pre-trained on short and concise queries, while the PSC input is longer and noisier. This input mismatch between ad-hoc search and PSC limits retrieval quality. While fine-tuning on PSC data helps, its benefits remain constrained by this input gap. In this work, we propose Conv2Query, a novel conversation-to-query framework that adapts ad-hoc retrievers to PSC by bridging the input gap between ad-hoc search and PSC. Conv2Query maps conversational context into ad-hoc queries, which can either be used as input for off-the-shelf ad-hoc retrievers or for further fine-tuning on PSC data. Extensive experiments on two PSC datasets show that Conv2Query significantly improves ad-hoc retrievers' performance, both when used directly and after fine-tuning on PSC. 

**Abstract (ZH)**: 主动对话检索（PSC）旨在通过在会话背景下主动检索有用的相关信息来减少用户的查询构架努力。先前的PSC工作要么直接将此背景作为输入传递给现成的即席检索器，要么在PSC数据上进一步微调它们。然而，即席检索器是基于短且简洁的查询进行预训练的，而PSC输入更长且更具噪声。这种即席搜索与PSC之间的输入不匹配限制了检索质量。虽然在PSC数据上进行微调有所帮助，但其效益仍受限于这种输入差距。在本文中，我们提出了一种新的Conv2Query框架，通过弥合即席搜索与PSC之间的输入差距，使即席检索器适应PSC。Conv2Query将会话背景映射为即席查询，这些查询可以作为现成的即席检索器的输入，或用于在PSC数据上进一步微调。在两个PSC数据集上的广泛实验表明，Conv2Query显着提高了即席检索器的性能，无论是否在PSC数据上进行进一步微调。 

---
# What do self-supervised speech models know about Dutch? Analyzing advantages of language-specific pre-training 

**Title (ZH)**: 自我监督语音模型对荷兰语了解多少？分析语言特定预训练的优势 

**Authors**: Marianne de Heer Kloots, Hosein Mohebbi, Charlotte Pouw, Gaofei Shen, Willem Zuidema, Martijn Bentum  

**Link**: [PDF](https://arxiv.org/pdf/2506.00981)  

**Abstract**: How language-specific are speech representations learned by self-supervised models? Existing work has shown that a range of linguistic features can be successfully decoded from end-to-end models trained only on speech recordings. However, it's less clear to what extent pre-training on specific languages improves language-specific linguistic information. Here we test the encoding of Dutch phonetic and lexical information in internal representations of self-supervised Wav2Vec2 models. Pre-training exclusively on Dutch improves the representation of Dutch linguistic features as compared to pre-training on similar amounts of English or larger amounts of multilingual data. This language-specific advantage is well-detected by trained clustering or classification probes, and partially observable using zero-shot metrics. Furthermore, the language-specific benefit on linguistic feature encoding aligns with downstream performance on Automatic Speech Recognition. 

**Abstract (ZH)**: 自监督Wav2Vec2模型中学习到的语音表示的语言特异性程度如何？预训练于特定语言在多大程度上提升了语言特异性语言信息？预训练于荷兰语比预训练于相似量的英语或更大规模的多语言数据更能捕获荷兰语音素和词汇信息。这种语言特异性优势通过训练聚类或分类探测器能够很好地检测到，并部分通过零样本指标观测到。此外，语言特异性优势与自动语音识别的下游性能一致。 

---
# IVY-FAKE: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection 

**Title (ZH)**: IVY-FAKE: 一个统一的可解释框架与基准用于图像和视频AIGC检测 

**Authors**: Wayne Zhang, Changjiang Jiang, Zhonghao Zhang, Chenyang Si, Fengchang Yu, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00979)  

**Abstract**: The rapid advancement of Artificial Intelligence Generated Content (AIGC) in visual domains has resulted in highly realistic synthetic images and videos, driven by sophisticated generative frameworks such as diffusion-based architectures. While these breakthroughs open substantial opportunities, they simultaneously raise critical concerns about content authenticity and integrity. Many current AIGC detection methods operate as black-box binary classifiers, which offer limited interpretability, and no approach supports detecting both images and videos in a unified framework. This dual limitation compromises model transparency, reduces trustworthiness, and hinders practical deployment. To address these challenges, we introduce IVY-FAKE , a novel, unified, and large-scale dataset specifically designed for explainable multimodal AIGC detection. Unlike prior benchmarks, which suffer from fragmented modality coverage and sparse annotations, IVY-FAKE contains over 150,000 richly annotated training samples (images and videos) and 18,700 evaluation examples, each accompanied by detailed natural-language reasoning beyond simple binary labels. Building on this, we propose Ivy Explainable Detector (IVY-XDETECTOR), a unified AIGC detection and explainable architecture that jointly performs explainable detection for both image and video content. Our unified vision-language model achieves state-of-the-art performance across multiple image and video detection benchmarks, highlighting the significant advancements enabled by our dataset and modeling framework. Our data is publicly available at this https URL. 

**Abstract (ZH)**: AIGC视觉领域生成内容的快速进步产生了高度逼真的合成图像和视频，由基于扩散的架构等复杂的生成框架驱动。尽管这些突破创造了巨大机会，但也引发了关于内容真实性和完整性的关键担忧。当前许多AIGC检测方法作为黑盒二分类器运作，缺乏可解释性，而且没有任何方法能够在统一框架中同时检测图像和视频。这一双重限制损害了模型的透明性，降低了可信度，并阻碍了实际部署。为应对这些挑战，我们介绍了IVY-FAKE，一个专为可解释的多模态AIGC检测设计的新型大规模数据集。不同于先前基准数据集碎片化的模态覆盖和稀疏的注释，IVY-FAKE 包含超过150,000 个丰富注释的训练样本（图像和视频）和18,700 个评估例证，每例均附有详细的自然语言推理，而不仅仅是简单的二元标签。基于此，我们提出了Ivy Explainable Detector（IVY-XDETECTOR），一种统一的AIGC检测和可解释架构，能够同时对图像和视频内容进行可解释检测。我们的统一视觉语言模型在多个图像和视频检测基准测试中取得最佳性能，突显了我们的数据集和建模框架带来的显著进步。数据集公开可访问于此链接。 

---
# NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction 

**Title (ZH)**: NTPP：基于下一个词对预测的双通道语音对话生成语言模型 

**Authors**: Qichao Wang, Ziqiao Meng, Wenqian Cui, Yifei Zhang, Pengcheng Wu, Bingzhe Wu, Irwin King, Liang Chen, Peilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00975)  

**Abstract**: Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications. 

**Abstract (ZH)**: 受到GPT-4o卓越能力的启发，人们对使语音语言模型（SLMs）能够与人类进行自然流畅的对话产生了 growing 兴趣。近期进展促使开发出了几种在这一领域表现令人鼓舞的 SLMs。然而，当前的方法尚未充分利用固有的双通道语音数据，这种数据能够捕捉人类对话的结构和动态。在此项工作中，我们系统地探讨了在现代大语言模型背景下使用双通道语音数据的方法，并引入了一种新颖的生成建模范式——下一令牌对预测（NTPP），首次使用解码器仅架构实现了独立说话人双通道语音对话学习。我们在标准基准上评估了我们的方法，实证结果表明，我们提出的方法 NTPP 显著提升了 SLMs 的对话能力，特别是在回合转换预测、响应连贯性和自然度方面。此外，与现有方法相比，NTPP 实现了显著更低的推理延迟，突显了其在实时应用中的实用性。 

---
# Data Heterogeneity Modeling for Trustworthy Machine Learning 

**Title (ZH)**: 数据异质性建模以实现可信赖机器学习 

**Authors**: Jiashuo Liu, Peng Cui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00969)  

**Abstract**: Data heterogeneity plays a pivotal role in determining the performance of machine learning (ML) systems. Traditional algorithms, which are typically designed to optimize average performance, often overlook the intrinsic diversity within datasets. This oversight can lead to a myriad of issues, including unreliable decision-making, inadequate generalization across different domains, unfair outcomes, and false scientific inferences. Hence, a nuanced approach to modeling data heterogeneity is essential for the development of dependable, data-driven systems. In this survey paper, we present a thorough exploration of heterogeneity-aware machine learning, a paradigm that systematically integrates considerations of data heterogeneity throughout the entire ML pipeline -- from data collection and model training to model evaluation and deployment. By applying this approach to a variety of critical fields, including healthcare, agriculture, finance, and recommendation systems, we demonstrate the substantial benefits and potential of heterogeneity-aware ML. These applications underscore how a deeper understanding of data diversity can enhance model robustness, fairness, and reliability and help model diagnosis and improvements. Moreover, we delve into future directions and provide research opportunities for the whole data mining community, aiming to promote the development of heterogeneity-aware ML. 

**Abstract (ZH)**: 数据异质性在确定机器学习系统性能中起着关键作用。传统的算法通常设计用于优化平均性能，往往忽略了数据集内的内在多样性。这种忽略可能导致决策可靠性差、跨不同领域的一般化不足、不公平结果和虚假的科学推断。因此，数据异质性建模的细致方法对于开发可靠的、数据驱动的系统至关重要。在本文综述中，我们全面探讨了数据异质性aware机器学习这一范式，该范式在整个机器学习管线上系统地整合了数据异质性的考虑——从数据收集和模型训练到模型评估和部署。通过将这种方法应用于医疗保健、农业、金融和推荐系统等多个关键领域，我们展示了数据异质性aware机器学习的巨大优势和潜力。这些应用强调了对数据多样性更深入理解如何增强模型的健壮性、公平性和可靠性，以及帮助模型诊断和改进。此外，我们探讨了未来的研究方向，并为整个数据挖掘社区提供了研究机会，旨在促进数据异质性aware机器学习的发展。 

---
# Legal Compliance Evaluation of Smart Contracts Generated By Large Language Models 

**Title (ZH)**: 由大型语言模型生成的智能合约的法律合规性评估 

**Authors**: Chanuka Wijayakoon, Hai Dong, H.M.N. Dilum Bandara, Zahir Tari, Anurag Soin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00943)  

**Abstract**: Smart contracts can implement and automate parts of legal contracts, but ensuring their legal compliance remains challenging. Existing approaches such as formal specification, verification, and model-based development require expertise in both legal and software development domains, as well as extensive manual effort. Given the recent advances of Large Language Models (LLMs) in code generation, we investigate their ability to generate legally compliant smart contracts directly from natural language legal contracts, addressing these challenges. We propose a novel suite of metrics to quantify legal compliance based on modeling both legal and smart contracts as processes and comparing their behaviors. We select four LLMs, generate 20 smart contracts based on five legal contracts, and analyze their legal compliance. We find that while all LLMs generate syntactically correct code, there is significant variance in their legal compliance with larger models generally showing higher levels of compliance. We also evaluate the proposed metrics against properties of software metrics, showing they provide fine-grained distinctions, enable nuanced comparisons, and are applicable across domains for code from any source, LLM or developer. Our results suggest that LLMs can assist in generating starter code for legally compliant smart contracts with strict reviews, and the proposed metrics provide a foundation for automated and self-improving development workflows. 

**Abstract (ZH)**: 基于大型语言模型生成合法合规智能合约的研究 

---
# anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding 

**Title (ZH)**: anyECG-chat：一种灵活心电图输入与多任务理解的通用ECG-MLLM 

**Authors**: Haitao Li, Ziyu Li, Yiheng Mao, Ziyi Liu, Zhoujian Sun, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00942)  

**Abstract**: The advent of multimodal large language models (MLLMs) has sparked interest in their application to electrocardiogram (ECG) analysis. However, existing ECG-focused MLLMs primarily focus on report generation tasks, often limited to single 12-lead, short-duration (10s) ECG inputs, thereby underutilizing the potential of MLLMs. To this end, we aim to develop a MLLM for ECG analysis that supports a broader range of tasks and more flexible ECG inputs. However, existing ECG-QA datasets are often monotonous. To address this gap, we first constructed the anyECG dataset, which encompasses a wide variety of tasks, including report generation, abnormal waveform localization, and open-ended question answering. In addition to standard hospital ECGs, we introduced long-duration reduced-lead ECGs for home environments and multiple ECG comparison scenarios commonly encountered in clinical practice. Furthermore, we propose the anyECG-chat model, which supports dynamic-length ECG inputs and multiple ECG inputs. We trained the model using a three-stage curriculum training recipe with the anyECG dataset. A comprehensive evaluation was conducted, demonstrating that anyECG-chat is capable of supporting various practical application scenarios, including not only common report generation tasks but also abnormal waveform localization for long-duration reduced-lead ECGs in home environments and comprehensive comparative analysis of multiple ECGs. 

**Abstract (ZH)**: 多模态大型语言模型在心电图分析中的应用：anyECG-chat模型的设计与实现 

---
# Uncertainty-Aware Metabolic Stability Prediction with Dual-View Contrastive Learning 

**Title (ZH)**: 具有双视图对比学习的不确定性意识代谢稳定性预测 

**Authors**: Peijin Guo, Minghui Li, Hewen Pan, Bowen Chen, Yang Wu, Zikang Guo, Leo Yu Zhang, Shengshan Hu, Shengqing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00936)  

**Abstract**: Accurate prediction of molecular metabolic stability (MS) is critical for drug research and development but remains challenging due to the complex interplay of molecular interactions. Despite recent advances in graph neural networks (GNNs) for MS prediction, current approaches face two critical limitations: (1) incomplete molecular modeling due to atom-centric message-passing mechanisms that disregard bond-level topological features, and (2) prediction frameworks that lack reliable uncertainty quantification. To address these challenges, we propose TrustworthyMS, a novel contrastive learning framework designed for uncertainty-aware metabolic stability prediction. First, a molecular graph topology remapping mechanism synchronizes atom-bond interactions through edge-induced feature propagation, capturing both localized electronic effects and global conformational constraints. Second, contrastive topology-bond alignment enforces consistency between molecular topology views and bond patterns via feature alignment, enhancing representation robustness. Third, uncertainty modeling through Beta-Binomial uncertainty quantification enables simultaneous prediction and confidence calibration under epistemic uncertainty. Through extensive experiments, our results demonstrate that TrustworthyMS outperforms current state-of-the-art methods in terms of predictive performance. 

**Abstract (ZH)**: 可信的分子代谢稳定性预测（TrustworthyMS）：一种不确定性意识下的新颖对比学习框架 

---
# General-purpose audio representation learning for real-world sound scenes 

**Title (ZH)**: 通用音频表示学习以应对现实场景声音 

**Authors**: Goksenin Yuksel, Marcel van Gerven, Kiki van der Heijden  

**Link**: [PDF](https://arxiv.org/pdf/2506.00934)  

**Abstract**: While audio foundation models perform well on myriad of tasks from sound classification to speech analysis, these models are trained and tested on dry, non-spatial, single-source audio clips. This limits their success in real-world situations and results in spatially unaware audio embeddings. To address these limitations, we propose a novel self-supervised training approach for General-Purpose, Real-world Audio Models (GRAMs). The GRAM training approach enables robust spatial audio representation learning for naturalistic, noisy sound scenes and can be applied to any masking-based deep learning model. We demonstrate the success of our approach by training two state-of-the-art models, one with a transformer and one with a mamba backbone. We assess the quality of the extracted audio representations from GRAMs using the original version of the HEAR benchmark, a newly synthesized, naturalistic version of the HEAR benchmark, and novel sound localization tasks based on HEAR benchmark datasets. The results show that our approach minimizes the performance gap between dry, non-spatial, single-source sound scenes and naturalistic sound scenes for crucial tasks such as auditory scene analysis, outperforming existing state-of-the-art audio foundation models at a fraction of the training steps. Moreover, GRAMs show state-of-the-art performance on sound localization tasks, exceeding even supervised sound localization models. In sum, the proposed approach represents a significant advancement towards robust audio foundation models for real-world applications with state-of-the-art performance on naturalistic sound scenes as well as spatial audio representation learning. 

**Abstract (ZH)**: 面向现实场景的通用音频模型的自监督训练方法 

---
# In-the-wild Audio Spatialization with Flexible Text-guided Localization 

**Title (ZH)**: 户外音频空间化与灵活的文本导向定位 

**Authors**: Tianrui Pan, Jie Liu, Zewen Huang, Jie Tang, Gangshan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00927)  

**Abstract**: To enhance immersive experiences, binaural audio offers spatial awareness of sounding objects in AR, VR, and embodied AI applications. While existing audio spatialization methods can generally map any available monaural audio to binaural audio signals, they often lack the flexible and interactive control needed in complex multi-object user-interactive environments. To address this, we propose a Text-guided Audio Spatialization (TAS) framework that utilizes flexible text prompts and evaluates our model from unified generation and comprehension perspectives. Due to the limited availability of premium and large-scale stereo data, we construct the SpatialTAS dataset, which encompasses 376,000 simulated binaural audio samples to facilitate the training of our model. Our model learns binaural differences guided by 3D spatial location and relative position prompts, augmented by flipped-channel audio. It outperforms existing methods on both simulated and real-recorded datasets, demonstrating superior generalization and accuracy. Besides, we develop an assessment model based on Llama-3.1-8B, which evaluates the spatial semantic coherence between our generated binaural audio and text prompts through a spatial reasoning task. Results demonstrate that text prompts provide flexible and interactive control to generate binaural audio with excellent quality and semantic consistency in spatial locations. Dataset is available at \href{this https URL} 

**Abstract (ZH)**: 为了增强沉浸体验，双向音频在AR、VR和具身AI应用中提供了声源的三维空间感知。由于现有音频空间化方法通常可以将任何可用的单声道音频映射为双向音频信号，但在复杂多对象用户交互环境中往往缺乏灵活的互动控制。为此，我们提出了一种文本引导的音频空间化（TAS）框架，该框架利用灵活的文本提示，从统一生成和理解的角度评估我们的模型。由于高质量和大规模立体声数据的有限可用性，我们构建了SpatialTAS数据集，包含376,000个模拟的双向音频样本，以促进我们模型的训练。我们的模型通过三维空间位置和相对位置提示学习双向差异，并辅以翻转通道音频。该模型在模拟和实际录音数据集上均优于现有方法，展现出更好的泛化能力和准确性。此外，我们基于Llama-3.1-8B开发了一种评估模型，通过空间推理任务评估我们生成的双向音频与文本提示之间的空间语义一致性。结果表明，文本提示能够在空间位置上提供灵活的互动控制，生成高质量且语义一致的双向音频。数据集可通过\href{this https URL}获得。 

---
# Bridging Subjective and Objective QoE: Operator-Level Aggregation Using LLM-Based Comment Analysis and Network MOS Comparison 

**Title (ZH)**: 基于LLM的评论分析与网络MOS比较的主观与客观QoE桥梁构建：运营商级聚合 

**Authors**: Parsa Hassani Shariat Panahi, Amir Hossein Jalilvand, M. Hasan Najafi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00924)  

**Abstract**: This paper introduces a dual-layer framework for network operator-side quality of experience (QoE) assessment that integrates both objective network modeling and subjective user perception extracted from live-streaming platforms. On the objective side, we develop a machine learning model trained on mean opinion scores (MOS) computed via the ITU-T P.1203 reference implementation, allowing accurate prediction of user-perceived video quality using only network parameters such as packet loss, delay, jitter, and throughput without reliance on video content or client-side instrumentation. On the subjective side, we present a semantic filtering and scoring pipeline that processes user comments from live streams to extract performance-related feedback. A large language model is used to assign scalar MOS scores to filtered comments in a deterministic and reproducible manner. To support scalable and interpretable analysis, we con- struct a labeled dataset of 47,894 live-stream comments, of which about 34,000 are identified as QoE-relevant through multi-layer semantic filtering. Each comment is enriched with simulated Internet Service Provider attribution and temporally aligned using synthetic timestamps in 5-min intervals. The resulting dataset enables operator-level aggregation and time-series analysis of user-perceived quality. A delta MOS metric is proposed to measure each Internet service provider's deviation from platform-wide sentiment, allowing detection of localized degradations even in the absence of direct network telemetry. A controlled outage simulation confirms the framework's effectiveness in identifying service disruptions through comment-based trends alone. The system provides each operator with its own subjective MOS and the global platform average per interval, enabling real-time interpretation of performance deviations and comparison with objective network-based QoE estimates. 

**Abstract (ZH)**: 一种结合客观网络建模和主观用户感知的双层框架，用于网络运营商侧质量体验评估 

---
# Position as Probability: Self-Supervised Transformers that Think Past Their Training for Length Extrapolation 

**Title (ZH)**: 位置即概率：超越训练长度进行外推的自监督变压器模型 

**Authors**: Philip Heejun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.00920)  

**Abstract**: Deep sequence models typically degrade in accuracy when test sequences significantly exceed their training lengths, yet many critical tasks--such as algorithmic reasoning, multi-step arithmetic, and compositional generalization--require robust length extrapolation. We introduce PRISM, a Probabilistic Relative-position Implicit Superposition Model, a novel positional encoding mechanism that enables Transformers to extrapolate accurately up to 10x beyond their training length. PRISM learns continuous relative positions through a differentiable histogram-filter update, preserving position uncertainty via a probabilistic superposition rather than conventional deterministic embeddings. Empirically, PRISM achieves state-of-the-art length extrapolation, successfully generalizing to previously intractable sequence lengths across algorithmic benchmarks--including arithmetic (addition, multiplication), SCAN compositionality tasks, and complex copy variants derived from DeepMind's recent datasets. Our analysis demonstrates that PRISM's stochastic positional encoding maintains sharp and interpretable internal states, providing a theoretical basis for reliable length generalization. These results advance the goal of neural sequence models that remain algorithmically robust at lengths far exceeding their training horizon. 

**Abstract (ZH)**: Probabilistic Relative-position Implicit Superposition Model for Robust Length Extrapolation 

---
# Principled Input-Output-Conditioned Post-Hoc Uncertainty Estimation for Regression Networks 

**Title (ZH)**: principled 输入-输出-条件后验不确定性估计对于回归网络 

**Authors**: Lennart Bramlage, Cristóbal Curio  

**Link**: [PDF](https://arxiv.org/pdf/2506.00918)  

**Abstract**: Uncertainty quantification is critical in safety-sensitive applications but is often omitted from off-the-shelf neural networks due to adverse effects on predictive performance. Retrofitting uncertainty estimates post-hoc typically requires access to model parameters or gradients, limiting feasibility in practice. We propose a theoretically grounded framework for post-hoc uncertainty estimation in regression tasks by fitting an auxiliary model to both original inputs and frozen model outputs. Drawing from principles of maximum likelihood estimation and sequential parameter fitting, we formalize an exact post-hoc optimization objective that recovers the canonical MLE of Gaussian parameters, without requiring sampling or approximation at inference. While prior work has used model outputs to estimate uncertainty, we explicitly characterize the conditions under which this is valid and demonstrate the extent to which structured outputs can support quasi-epistemic inference. We find that using diverse auxiliary data, such as augmented subsets of the original training data, significantly enhances OOD detection and metric performance. Our hypothesis that frozen model outputs contain generalizable latent information about model error and predictive uncertainty is tested and confirmed. Finally, we ensure that our method maintains proper estimation of input-dependent uncertainty without relying exclusively on base model forecasts. These findings are demonstrated in toy problems and adapted to both UCI and depth regression benchmarks. Code: this https URL. 

**Abstract (ZH)**: 基于回归任务的后验不确定性量化：一个理论上支持的框架 

---
# How do Transformer Embeddings Represent Compositions? A Functional Analysis 

**Title (ZH)**: 变压器嵌入如何表示组成？一种功能分析 

**Authors**: Aishik Nagar, Ishaan Singh Rawal, Mansi Dhanania, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00914)  

**Abstract**: Compositionality is a key aspect of human intelligence, essential for reasoning and generalization. While transformer-based models have become the de facto standard for many language modeling tasks, little is known about how they represent compound words, and whether these representations are compositional. In this study, we test compositionality in Mistral, OpenAI Large, and Google embedding models, and compare them with BERT. First, we evaluate compositionality in the representations by examining six diverse models of compositionality (addition, multiplication, dilation, regression, etc.). We find that ridge regression, albeit linear, best accounts for compositionality. Surprisingly, we find that the classic vector addition model performs almost as well as any other model. Next, we verify that most embedding models are highly compositional, while BERT shows much poorer compositionality. We verify and visualize our findings with a synthetic dataset consisting of fully transparent adjective-noun compositions. Overall, we present a thorough investigation of compositionality. 

**Abstract (ZH)**: 组成性是人类智能的关键方面，对于推理和泛化至关重要。尽管基于变换器的模型已成为许多语言建模任务的事实标准，但对其如何表示复合词以及这些表示是否具有组成性的了解甚少。在本研究中，我们测试了 Mistral、OpenAI Large、Google 嵌入模型以及 BERT 在组成性方面的表现，并对其进行了比较。首先，我们通过检查六种不同的组成性模型（加法、乘法、扩张、回归等）来评估组成性在表示中的体现。我们发现，尽管岭回归是线性的，但它最好地解释了组成性。令人惊讶的是，我们发现经典的向量加法模型几乎与其他任何模型一样有效。接下来，我们验证大多数嵌入模型具有高度的组成性，而 BERT 的组成性却表现较差。我们通过一个由完全透明的形容词-名词组合构成的合成数据集验证并可视化了这些发现。总体而言，我们进行了全面的组成性研究。 

---
# Pi-SQL: Enhancing Text-to-SQL with Fine-Grained Guidance from Pivot Programming Languages 

**Title (ZH)**: Pi-SQL：来自pivot编程语言的细粒度指导以增强文本到SQL的转换 

**Authors**: Yongdong chi, Hanqing Wang, Zonghan Yang, Jian Yang, Xiao Yan, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00912)  

**Abstract**: Text-to-SQL transforms the user queries from natural language to executable SQL programs, enabling non-experts to interact with complex databases. Existing prompt-based methods craft meticulous text guidelines and examples to facilitate SQL generation, but their accuracy is hindered by the large semantic gap between the texts and the low-resource SQL programs. In this work, we propose Pi-SQL, which incorporates the high-resource Python program as a pivot to bridge between the natural language query and SQL program. In particular, Pi-SQL first generates Python programs that provide fine-grained step-by-step guidelines in their code blocks or comments, and then produces an SQL program following the guidance of each Python this http URL final SQL program matches the reference Python program's query results and, through selection from candidates generated by different strategies, achieves superior execution speed, with a reward-based valid efficiency score up to 4.55 higher than the best-performing this http URL experiments demonstrate the effectiveness of Pi-SQL, which improves the execution accuracy of the best-performing baseline by up to 3.20. 

**Abstract (ZH)**: Text-to-SQL将用户查询从自然语言转换为可执行的SQL程序，使非专家能够与复杂数据库进行交互。现有的基于提示的方法通过精心制作的文本指南和示例来促进SQL生成，但它们的准确性受到了文本与低资源SQL程序之间庞大语义差距的阻碍。在本文中，我们提出了Pi-SQL，它将高资源的Python程序作为枢纽，连接自然语言查询和SQL程序。特别是，Pi-SQL首先生成Python程序，这些程序在其代码块或注释中提供详尽的逐步指南，然后根据每个Python程序的指导生成SQL程序。最终生成的SQL程序与参考Python程序的查询结果匹配，并通过从不同策略生成的候选者中选择，实现了比最佳性能基线高出4.55的奖励驱动的有效效率得分。实验表明，Pi-SQL的有效性可以将最佳性能基线的执行准确性提高3.20。 

---
# PCoreSet: Effective Active Learning through Knowledge Distillation from Vision-Language Models 

**Title (ZH)**: PCoreSet: 通过来自视觉-语言模型的知识蒸馏实现有效的主动学习 

**Authors**: Seongjae Kang, Dong Bok Lee, Hyungjoon Jang, Dongseop Kim, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00910)  

**Abstract**: Knowledge distillation (KD) is a widely used framework for training compact, task-specific models by leveraging the knowledge of teacher models. However, its application to active learning (AL), which aims to minimize annotation costs through iterative sample selection, remains underexplored. This gap stems from the fact that KD typically assumes access to sufficient labeled data, whereas AL operates in data-scarce scenarios where task-specific teacher models are often unavailable. In this paper, we introduce ActiveKD, a framework that integrates AL with KD by leveraging the zero- and few-shot capabilities of large vision-language models (VLMs). A key aspect of ActiveKD is the structured prediction bias of VLMs--i.e., their predictions form clusters in the probability space. We regard this structure as an inductive bias of the teacher model, capturing generalizable output patterns beneficial to student learning. To exploit this bias, we propose Probabilistic CoreSet (PCoreSet), a selection strategy that maximizes coverage in the probability space rather than the feature space. PCoreSet strategically selects categorically diverse unlabeled samples, facilitating more efficient transfer of teacher knowledge under limited annotation budgets. Evaluations on 11 datasets show that PCoreSet consistently outperforms existing selection methods within the ActiveKD framework, advancing research at the intersection of AL and KD. 

**Abstract (ZH)**: 知识蒸馏（KD）是一种通过利用教师模型的知识来训练紧凑的任务专用模型的广泛使用的框架。然而，将其应用于主动学习（AL），即通过迭代样本选择来最小化注释成本的方法，仍然鲜有探索。这一空白源于KD通常假设可以访问充足标记数据的事实，而AL则在数据稀缺的情景中运行，其中往往缺少任务专用的教师模型。本文介绍了ActiveKD框架，该框架通过利用大规模视觉-语言模型（VLMs）的零样本和极少样本能力，将AL与KD相结合。ActiveKD的一个关键方面是VLMs的结构化预测偏差，即它们的预测在概率空间中形成簇。我们将这种结构视为教师模型的归纳偏置，捕捉对学生学习有益的可泛化的输出模式。为了利用这一偏差，我们提出了一种概率核心集（PCoreSet）的选择策略，该策略在概率空间中最大化覆盖范围而不是特征空间。PCoreSet有选择地挑选类别多样性的未标记样本，有利于在有限注释预算下更高效地转移教师知识。在11个数据集上的评估表明，PCoreSet在ActiveKD框架内的选择方法中表现最优，促进了AL和KD交叉领域的研究进展。 

---
# State-Covering Trajectory Stitching for Diffusion Planners 

**Title (ZH)**: 状态覆盖轨迹缝合用于扩散规划者 

**Authors**: Kyowoon Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00895)  

**Abstract**: Diffusion-based generative models are emerging as powerful tools for long-horizon planning in reinforcement learning (RL), particularly with offline datasets. However, their performance is fundamentally limited by the quality and diversity of training data. This often restricts their generalization to tasks outside their training distribution or longer planning horizons. To overcome this challenge, we propose State-Covering Trajectory Stitching (SCoTS), a novel reward-free trajectory augmentation method that incrementally stitches together short trajectory segments, systematically generating diverse and extended trajectories. SCoTS first learns a temporal distance-preserving latent representation that captures the underlying temporal structure of the environment, then iteratively stitches trajectory segments guided by directional exploration and novelty to effectively cover and expand this latent space. We demonstrate that SCoTS significantly improves the performance and generalization capabilities of diffusion planners on offline goal-conditioned benchmarks requiring stitching and long-horizon reasoning. Furthermore, augmented trajectories generated by SCoTS significantly improve the performance of widely used offline goal-conditioned RL algorithms across diverse environments. 

**Abstract (ZH)**: 基于扩散的生成模型正成为强化学习(_rl_)中长远规划的强大工具，特别是在使用离线数据集的情况下。然而，它们的表现从根本上受限于训练数据的质量和多样性。这通常限制了它们在训练分布之外的任务上的泛化能力或更长的规划时间范围。为克服这一挑战，我们提出了一种名为状态覆盖轨迹缝合(Scots)的新型无奖励轨迹增强方法，该方法逐步缝合短轨迹片段，系统地生成多样且延长的轨迹。Scots首先学习一个保持时间距离的潜在表示，以捕捉环境的内在时间结构，然后通过方向探索和新颖性逐步缝合轨迹片段，有效地覆盖和扩展这一潜在空间。我们展示了Scots在需要缝合和长远推理的离线目标条件基准测试中显著提高了扩散规划器的性能和泛化能力。此外，由Scots生成的增强轨迹在多种环境中显著提高了广泛使用的离线目标条件RL算法的性能。 

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
# Uneven Event Modeling for Partially Relevant Video Retrieval 

**Title (ZH)**: 部分相关视频检索的不均匀事件建模 

**Authors**: Sa Zhu, Huashan Chen, Wanqian Zhang, Jinchao Zhang, Zexian Yang, Xiaoshuai Hao, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00891)  

**Abstract**: Given a text query, partially relevant video retrieval (PRVR) aims to retrieve untrimmed videos containing relevant moments, wherein event modeling is crucial for partitioning the video into smaller temporal events that partially correspond to the text. Previous methods typically segment videos into a fixed number of equal-length clips, resulting in ambiguous event boundaries. Additionally, they rely on mean pooling to compute event representations, inevitably introducing undesired misalignment. To address these, we propose an Uneven Event Modeling (UEM) framework for PRVR. We first introduce the Progressive-Grouped Video Segmentation (PGVS) module, to iteratively formulate events in light of both temporal dependencies and semantic similarity between consecutive frames, enabling clear event boundaries. Furthermore, we also propose the Context-Aware Event Refinement (CAER) module to refine the event representation conditioned the text's cross-attention. This enables event representations to focus on the most relevant frames for a given text, facilitating more precise text-video alignment. Extensive experiments demonstrate that our method achieves state-of-the-art performance on two PRVR benchmarks. 

**Abstract (ZH)**: 给定文本查询的部分相关视频检索（PRVR）旨在检索包含相关时刻的未剪辑视频，其中事件建模对于将视频划分为部分对应文本的小时间段事件至关重要。先前的方法通常将视频分割为固定数量的等长片段，导致事件边界模糊。此外，它们依赖于均值池化来计算事件表示，不可避免地引入了不必要的对齐错误。为了解决这些问题，我们提出了一个不均匀事件建模（UEM）框架用于部分相关视频检索。我们首先引入了渐进分组视频分割（PGVS）模块，通过考虑连续帧之间的时序依赖性和语义相似性来逐迭代地定义事件，从而实现清晰的事件边界。此外，我们还提出了基于文本交叉注意力的事件精炼（CAER）模块来细化事件表示。这使得事件表示能够聚焦于给定文本的最相关帧，从而促进更精确的文本-视频对齐。广泛的实验表明，我们的方法在两个PRVR基准测试中实现了最先进的性能。 

---
# CoVoMix2: Advancing Zero-Shot Dialogue Generation with Fully Non-Autoregressive Flow Matching 

**Title (ZH)**: CoVoMix2: 采用全非自回归流匹配促进零样本对话生成 

**Authors**: Leying Zhang, Yao Qian, Xiaofei Wang, Manthan Thakker, Dongmei Wang, Jianwei Yu, Haibin Wu, Yuxuan Hu, Jinyu Li, Yanmin Qian, Sheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00885)  

**Abstract**: Generating natural-sounding, multi-speaker dialogue is crucial for applications such as podcast creation, virtual agents, and multimedia content generation. However, existing systems struggle to maintain speaker consistency, model overlapping speech, and synthesize coherent conversations efficiently. In this paper, we introduce CoVoMix2, a fully non-autoregressive framework for zero-shot multi-talker dialogue generation. CoVoMix2 directly predicts mel-spectrograms from multi-stream transcriptions using a flow-matching-based generative model, eliminating the reliance on intermediate token representations. To better capture realistic conversational dynamics, we propose transcription-level speaker disentanglement, sentence-level alignment, and prompt-level random masking strategies. Our approach achieves state-of-the-art performance, outperforming strong baselines like MoonCast and Sesame in speech quality, speaker consistency, and inference speed. Notably, CoVoMix2 operates without requiring transcriptions for the prompt and supports controllable dialogue generation, including overlapping speech and precise timing control, demonstrating strong generalizability to real-world speech generation scenarios. 

**Abstract (ZH)**: 零样本多说话人对话生成的CoVoMix2全非自回归框架 

---
# ModuLM: Enabling Modular and Multimodal Molecular Relational Learning with Large Language Models 

**Title (ZH)**: ModuLM：通过大规模语言模型实现模块化和多模态分子关系学习 

**Authors**: Zhuo Chen, Yizhen Zheng, Huan Yee Koh, Hongxin Xiang, Linjiang Chen, Wenjie Du, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00880)  

**Abstract**: Molecular Relational Learning (MRL) aims to understand interactions between molecular pairs, playing a critical role in advancing biochemical research. With the recent development of large language models (LLMs), a growing number of studies have explored the integration of MRL with LLMs and achieved promising results. However, the increasing availability of diverse LLMs and molecular structure encoders has significantly expanded the model space, presenting major challenges for benchmarking. Currently, there is no LLM framework that supports both flexible molecular input formats and dynamic architectural switching. To address these challenges, reduce redundant coding, and ensure fair model comparison, we propose ModuLM, a framework designed to support flexible LLM-based model construction and diverse molecular representations. ModuLM provides a rich suite of modular components, including 8 types of 2D molecular graph encoders, 11 types of 3D molecular conformation encoders, 7 types of interaction layers, and 7 mainstream LLM backbones. Owing to its highly flexible model assembly mechanism, ModuLM enables the dynamic construction of over 50,000 distinct model configurations. In addition, we provide comprehensive results to demonstrate the effectiveness of ModuLM in supporting LLM-based MRL tasks. 

**Abstract (ZH)**: 分子关系学习(MRL)旨在理解分子对之间的相互作用，对推动生物化学研究具有重要作用。随着大型语言模型(LLMs)的 recent 发展，越来越多的研究探索了 MRL 与 LLMs 的集成并取得了有前景的结果。然而，多样化 LLMs 和分子结构编码器的不断增加使得模型空间大幅扩展，带来了重要挑战。目前尚无支持灵活的分子输入格式和动态架构切换的 LLM 框架。为应对这些挑战，减少冗余编码，并确保公平的模型比较，我们提出了 ModuLM，一种支持灵活的基于 LLM 的模型构建和多种分子表示的框架。ModuLM 提供了一整套模块化组件，包括 8 种 2D 分子图形编码器、11 种 3D 分子构象编码器、7 种相互作用层和 7 种主流 LLM 主干。由于其高度灵活的模型组装机制，ModuLM 能够动态构建超过 50,000 种不同的模型配置。此外，我们提供了全面的结果来证明 ModuLM 在支持基于 LLM 的 MRL 任务方面的有效性。 

---
# Towards Predicting Any Human Trajectory In Context 

**Title (ZH)**: 面向情境的人类轨迹预测 

**Authors**: Ryo Fujii, Hideo Saito, Ryo Hachiuma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00871)  

**Abstract**: Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, this process is often impractical on edge devices due to constrained computational resources. To address this challenge, we introduce TrajICL, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables rapid adaptation without fine-tuning on the scenario-specific data. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. The code will be released at this https URL. 

**Abstract (ZH)**: 基于上下文学习的行人轨迹预测方法 TrajICL：无需细调的快速适应 

---
# Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning 

**Title (ZH)**: 局部流形逼近与投影在流形感知扩散规划中的应用 

**Authors**: Kyowoon Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00867)  

**Abstract**: Recent advances in diffusion-based generative modeling have demonstrated significant promise in tackling long-horizon, sparse-reward tasks by leveraging offline datasets. While these approaches have achieved promising results, their reliability remains inconsistent due to the inherent stochastic risk of producing infeasible trajectories, limiting their applicability in safety-critical applications. We identify that the primary cause of these failures is inaccurate guidance during the sampling procedure, and demonstrate the existence of manifold deviation by deriving a lower bound on the guidance gap. To address this challenge, we propose Local Manifold Approximation and Projection (LoMAP), a training-free method that projects the guided sample onto a low-rank subspace approximated from offline datasets, preventing infeasible trajectory generation. We validate our approach on standard offline reinforcement learning benchmarks that involve challenging long-horizon planning. Furthermore, we show that, as a standalone module, LoMAP can be incorporated into the hierarchical diffusion planner, providing further performance enhancements. 

**Abstract (ZH)**: 基于扩散的生成建模的最近进展在利用离线数据集处理长期、稀疏奖励任务方面展示了显著的潜力。然而，这些方法的可靠性因固有的生产不可行轨迹的随机风险而不稳定，限制了它们在关键安全应用中的适用性。我们发现这些失败的主要原因是采样过程中指导的不准确，并通过推导指导间隙的下界证明了 manifold 偏差的存在。为应对这一挑战，我们提出了局部 manifold 近似和投影（LoMAP）方法，这是一种无需训练的方法，将指导的样本投影到由离线数据集近似得到的低秩子空间上，防止生成不可行轨迹。我们在涉及挑战性长期规划的基准离线强化学习任务上验证了该方法。此外，我们展示在作为独立模块的情况下，LoMAP 可以集成到层级扩散规划器中，提供进一步的性能提升。 

---
# Can AI Master Econometrics? Evidence from Econometrics AI Agent on Expert-Level Tasks 

**Title (ZH)**: AI能掌握计量经济学吗？从 Econometrics AI 代理在专家级任务中的表现看起 

**Authors**: Qiang Chen, Tianyang Han, Jin Li, Ye Luo, Yuxiao Wu, Xiaowei Zhang, Tuo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00856)  

**Abstract**: Can AI effectively perform complex econometric analysis traditionally requiring human expertise? This paper evaluates an agentic AI's capability to master econometrics, focusing on empirical analysis performance. We develop an ``Econometrics AI Agent'' built on the open-source MetaGPT framework. This agent exhibits outstanding performance in: (1) planning econometric tasks strategically, (2) generating and executing code, (3) employing error-based reflection for improved robustness, and (4) allowing iterative refinement through multi-round conversations. We construct two datasets from academic coursework materials and published research papers to evaluate performance against real-world challenges. Comparative testing shows our domain-specialized agent significantly outperforms both benchmark large language models (LLMs) and general-purpose AI agents. This work establishes a testbed for exploring AI's impact on social science research and enables cost-effective integration of domain expertise, making advanced econometric methods accessible to users with minimal coding expertise. Furthermore, our agent enhances research reproducibility and offers promising pedagogical applications for econometrics teaching. 

**Abstract (ZH)**: AI能否有效地执行传统上需要人类专长的复杂计量经济学分析？本文评估了一个代理AI掌握计量经济学的能力，重点在于其实证分析性能。我们基于开源MetaGPT框架开发了一个“计量经济学AI代理”。该代理在以下方面表现出色：（1）战略性规划计量经济学任务，（2）生成和执行代码，（3）利用基于误差的反思以提高稳健性，以及（4）通过多轮对话实现迭代优化。我们从学术课程材料和已发表的研究论文中构建了两个数据集，以评估其在现实世界挑战中的性能。比较测试结果显示，我们的领域专业化代理显著优于基准大语言模型（LLMs）和通用AI代理。本文建立了一个测试床，用于探索AI对社会科学研究的影响，并使高级计量经济学方法能够以低成本集成领域专业知识，从而无需大量编程知识即可供用户使用。此外，我们的代理增强了研究的可再现性，并为计量经济学教学提供了有前景的教学应用。 

---
# EEG2TEXT-CN: An Exploratory Study of Open-Vocabulary Chinese Text-EEG Alignment via Large Language Model and Contrastive Learning on ChineseEEG 

**Title (ZH)**: EEG2TEXT-CN：基于大规模语言模型和对比学习的开放词汇中文文本-EEG对齐探索性研究 

**Authors**: Jacky Tai-Yu Lu, Jung Chiang, Chi-Sheng Chen, Anna Nai-Yun Tung, Hsiang Wei Hu, Yuan Chiao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00854)  

**Abstract**: We propose EEG2TEXT-CN, which, to the best of our knowledge, represents one of the earliest open-vocabulary EEG-to-text generation frameworks tailored for Chinese. Built on a biologically grounded EEG encoder (NICE-EEG) and a compact pretrained language model (MiniLM), our architecture aligns multichannel brain signals with natural language representations via masked pretraining and contrastive learning. Using a subset of the ChineseEEG dataset, where each sentence contains approximately ten Chinese characters aligned with 128-channel EEG recorded at 256 Hz, we segment EEG into per-character embeddings and predict full sentences in a zero-shot setting. The decoder is trained with teacher forcing and padding masks to accommodate variable-length sequences. Evaluation on over 1,500 training-validation sentences and 300 held-out test samples shows promising lexical alignment, with a best BLEU-1 score of 6.38\%. While syntactic fluency remains a challenge, our findings demonstrate the feasibility of non-phonetic, cross-modal language decoding from EEG. This work opens a new direction in multilingual brain-to-text research and lays the foundation for future cognitive-language interfaces in Chinese. 

**Abstract (ZH)**: EEG2TEXT-CN：一种适用于中文的早期开放词汇EEG到文本生成框架 

---
# Generalization in VAE and Diffusion Models: A Unified Information-Theoretic Analysis 

**Title (ZH)**: VAE和扩散模型中的泛化性：一种统一的信息论分析 

**Authors**: Qi Chen, Jierui Zhu, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2506.00849)  

**Abstract**: Despite the empirical success of Diffusion Models (DMs) and Variational Autoencoders (VAEs), their generalization performance remains theoretically underexplored, especially lacking a full consideration of the shared encoder-generator structure. Leveraging recent information-theoretic tools, we propose a unified theoretical framework that provides guarantees for the generalization of both the encoder and generator by treating them as randomized mappings. This framework further enables (1) a refined analysis for VAEs, accounting for the generator's generalization, which was previously overlooked; (2) illustrating an explicit trade-off in generalization terms for DMs that depends on the diffusion time $T$; and (3) providing computable bounds for DMs based solely on the training data, allowing the selection of the optimal $T$ and the integration of such bounds into the optimization process to improve model performance. Empirical results on both synthetic and real datasets illustrate the validity of the proposed theory. 

**Abstract (ZH)**: 尽管扩散模型（DMs）和变分自编码器（VAEs）在实证上取得了成功，但它们的泛化性能在理论上尚未得到充分探索，特别是缺乏对共享编码-生成器结构的全面考虑。利用最新的信息论工具，我们提出了一种统一的理论框架，通过将编码器和生成器视为随机映射来为它们的泛化提供保障。该框架还进一步实现了以下能力：（1）对VAEs进行了细化分析，考虑了生成器的泛化问题，这是之前未曾关注的；（2）展示了DMs在泛化方面的显式权衡关系，该关系取决于扩散时间$T$；（3）基于训练数据提供了可计算的DMs边界，允许选择最优的$T$值，并将此类边界整合到优化过程中以提高模型性能。在合成数据集和真实数据集上的实验结果证明了所提理论的有效性。 

---
# Speech Unlearning 

**Title (ZH)**: 语音遗忘 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2506.00848)  

**Abstract**: We introduce machine unlearning for speech tasks, a novel and underexplored research problem that aims to efficiently and effectively remove the influence of specific data from trained speech models without full retraining. This has important applications in privacy preservation, removal of outdated or noisy data, and bias mitigation. While machine unlearning has been studied in computer vision and natural language processing, its application to speech is largely unexplored due to the high-dimensional, sequential, and speaker-dependent nature of speech data. We define two fundamental speech unlearning tasks: sample unlearning, which removes individual data points (e.g., a voice recording), and class unlearning, which removes an entire category (e.g., all data from a speaker), while preserving performance on the remaining data. Experiments on keyword spotting and speaker identification demonstrate that unlearning speech data is significantly more challenging than unlearning image or text data. We conclude with key future directions in this area, including structured training, robust evaluation, feature-level unlearning, broader applications, scalable methods, and adversarial robustness. 

**Abstract (ZH)**: 我们介绍了语音任务中的机器遗忘问题，这是一个新颖且尚未充分探索的研究课题，旨在无需完全重新训练的情况下，高效有效地从训练好的语音模型中移除特定数据的影响。这一课题在隐私保护、去除过时或噪声数据以及偏见缓解方面具有重要应用价值。尽管机器遗忘已经在计算机视觉和自然语言处理中进行了研究，但由于语音数据的高度维度性、序列依赖性和说话人依赖性，其在语音领域的应用尚未得到充分探索。我们定义了两种基本的语音遗忘任务：样本遗忘，即移除单个数据点（例如，一段语音记录），类别遗忘，即移除整个类别（例如，某说话人所有数据），同时保持对剩余数据性能的影响。关键词摘录和说话人识别实验表明，遗忘语音数据比遗忘图像或文本数据更具挑战性。最后，我们提出了该领域未来发展方向，包括结构化训练、鲁棒评估、特征级遗忘、更广泛的应用、可扩展方法和对抗鲁棒性。 

---
# Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience 

**Title (ZH)**: 面向结构化知识推理：经验增强的对比检索生成 

**Authors**: Jiawei Gu, Ziting Xian, Yuanzhen Xie, Ye Liu, Enjie Liu, Ruichao Zhong, Mochi Gao, Yunzhi Tan, Bo Hu, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00842)  

**Abstract**: Large language models (LLMs) achieve strong performance on plain text tasks but underperform on structured data like tables and databases. Potential challenges arise from their underexposure during pre-training and rigid text-to-structure transfer mechanisms. Unlike humans who seamlessly apply learned patterns across data modalities, LLMs struggle to infer implicit relationships embedded in tabular formats, especially in the absence of explicit structural guidance. To bridge this cognitive gap, we introduce Contrastive Retrieval-Augmented Generation on Experience (CoRE), a framework that builds experience memory representations and enhances generalization through contrastive In-Context Learning (ICL) to simulate human-like knowledge transfer. Experiments on Text-to-SQL and TableQA show CoRE significantly improves performance, achieving average gains of 3.44% and 4.24%, with up to 17.2% on challenging tasks. Our Monte Carlo Tree Search (MCTS)-generated Experience Memory expands training data 8-9x, enhancing diversity and domain coverage. This training-free and continual method propels LLMs toward structured knowledge expertise. 

**Abstract (ZH)**: 大型语言模型在文本任务上表现强劲，但在处理表格和数据库等结构化数据时表现不佳。对比检索增强生成经验（CoRE）框架通过构建经验记忆表示并通过对比上下文学习（ICL）增强泛化能力，模拟人类的知识迁移，以弥补这一认知差距。在Text-to-SQL和TableQA任务上的实验表明，CoRE显著提高了性能，平均分别提升了3.44%和4.24%，在挑战性任务上的提升高达17.2%。通过蒙特卡洛树搜索（MCTS）生成的经验记忆将训练数据扩展8-9倍，增强了多样性和领域覆盖范围。这种无需训练且持续的方法促使大型语言模型向结构化知识专门性迈进。 

---
# Counterfactual Activation Editing for Post-hoc Prosody and Mispronunciation Correction in TTS Models 

**Title (ZH)**: 用于TTS模型事后韵律及误读校正的反事实激活编辑 

**Authors**: Kyowoon Lee, Artyom Stitsyuk, Gunu Jho, Inchul Hwang, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00832)  

**Abstract**: Recent advances in Text-to-Speech (TTS) have significantly improved speech naturalness, increasing the demand for precise prosody control and mispronunciation correction. Existing approaches for prosody manipulation often depend on specialized modules or additional training, limiting their capacity for post-hoc adjustments. Similarly, traditional mispronunciation correction relies on grapheme-to-phoneme dictionaries, making it less practical in low-resource settings. We introduce Counterfactual Activation Editing, a model-agnostic method that manipulates internal representations in a pre-trained TTS model to achieve post-hoc control of prosody and pronunciation. Experimental results show that our method effectively adjusts prosodic features and corrects mispronunciations while preserving synthesis quality. This opens the door to inference-time refinement of TTS outputs without retraining, bridging the gap between pre-trained TTS models and editable speech synthesis. 

**Abstract (ZH)**: Recent Advances in Text-to-Speech: Achieving Post-Hoc Prosody Control and Mispronunciation Correction through Counterfactual Activation Editing 

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
# HERGC: Heterogeneous Experts Representation and Generative Completion for Multimodal Knowledge Graphs 

**Title (ZH)**: HERGC: 异构专家表示与生成性完成的多模态知识图谱 

**Authors**: Yongkang Xiao, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00826)  

**Abstract**: Multimodal knowledge graphs (MMKGs) enrich traditional knowledge graphs (KGs) by incorporating diverse modalities such as images and text. Multi-modal knowledge graph completion (MMKGC) seeks to exploit these heterogeneous signals to infer missing facts, thereby mitigating the intrinsic incompleteness of MMKGs. Existing MMKGC methods typically leverage only the information contained in the MMKGs under the closed-world assumption and adopt discriminative training objectives, which limits their reasoning capacity during completion. Recent generative completion approaches powered by advanced large language models (LLMs) have shown strong reasoning abilities in unimodal knowledge graph completion, but their potential in MMKGC remains largely unexplored. To bridge this gap, we propose HERGC, a Heterogeneous Experts Representation and Generative Completion framework for MMKGs. HERGC first deploys a Heterogeneous Experts Representation Retriever that enriches and fuses multimodal information and retrieves a compact candidate set for each incomplete triple. It then uses a Generative LLM Predictor fine-tuned on minimal instruction data to accurately identify the correct answer from these candidates. Extensive experiments on three standard MMKG benchmarks demonstrate HERGC's effectiveness and robustness, achieving state-of-the-art performance. 

**Abstract (ZH)**: 多模态知识图谱的异构专家表示与生成式补全框架（HERGC） 

---
# SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models 

**Title (ZH)**: SafeGenes: 评估基因组基础模型的对抗鲁棒性 

**Authors**: Huixin Zhan, Jason H. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2506.00821)  

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated significant success in variant effect prediction. However, their adversarial robustness remains largely unexplored. To address this gap, we propose SafeGenes: a framework for Secure analysis of genomic foundation models, leveraging adversarial attacks to evaluate robustness against both engineered near-identical adversarial Genes and embedding-space manipulations. In this study, we assess the adversarial vulnerabilities of GFMs using two approaches: the Fast Gradient Sign Method (FGSM) and a soft prompt attack. FGSM introduces minimal perturbations to input sequences, while the soft prompt attack optimizes continuous embeddings to manipulate model predictions without modifying the input tokens. By combining these techniques, SafeGenes provides a comprehensive assessment of GFM susceptibility to adversarial manipulation. Targeted soft prompt attacks led to substantial performance degradation, even in large models such as ESM1b and ESM1v. These findings expose critical vulnerabilities in current foundation models, opening new research directions toward improving their security and robustness in high-stakes genomic applications such as variant effect prediction. 

**Abstract (ZH)**: SafeGenes:一种利用对抗攻击评估基因基础模型安全性与鲁棒性的框架 

---
# DriveMind: A Dual-VLM based Reinforcement Learning Framework for Autonomous Driving 

**Title (ZH)**: DriveMind: 一种基于双多模态预训练语言模型的自主驾驶强化学习框架 

**Authors**: Dawood Wasif, Terrence J Moore, Chandan K Reddy, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2506.00819)  

**Abstract**: End-to-end autonomous driving systems map sensor data directly to control commands, but remain opaque, lack interpretability, and offer no formal safety guarantees. While recent vision-language-guided reinforcement learning (RL) methods introduce semantic feedback, they often rely on static prompts and fixed objectives, limiting adaptability to dynamic driving scenes. We present DriveMind, a unified semantic reward framework that integrates: (i) a contrastive Vision-Language Model (VLM) encoder for stepwise semantic anchoring; (ii) a novelty-triggered VLM encoder-decoder, fine-tuned via chain-of-thought (CoT) distillation, for dynamic prompt generation upon semantic drift; (iii) a hierarchical safety module enforcing kinematic constraints (e.g., speed, lane centering, stability); and (iv) a compact predictive world model to reward alignment with anticipated ideal states. DriveMind achieves 19.4 +/- 2.3 km/h average speed, 0.98 +/- 0.03 route completion, and near-zero collisions in CARLA Town 2, outperforming baselines by over 4% in success rate. Its semantic reward generalizes zero-shot to real dash-cam data with minimal distributional shift, demonstrating robust cross-domain alignment and potential for real-world deployment. 

**Abstract (ZH)**: 统一语义奖励框架DriveMind：面向动态驾驶场景的端到端自主驾驶系统 

---
# L3A: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning 

**Title (ZH)**: L3A：带有标签增强的分析适应性多标签类别增量学习 

**Authors**: Xiang Zhang, Run He, Jiao Chen, Di Fang, Ming Li, Ziqian Zeng, Cen Chen, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00816)  

**Abstract**: Class-incremental learning (CIL) enables models to learn new classes continually without forgetting previously acquired knowledge. Multi-label CIL (MLCIL) extends CIL to a real-world scenario where each sample may belong to multiple classes, introducing several challenges: label absence, which leads to incomplete historical information due to missing labels, and class imbalance, which results in the model bias toward majority classes. To address these challenges, we propose Label-Augmented Analytic Adaptation (L3A), an exemplar-free approach without storing past samples. L3A integrates two key modules. The pseudo-label (PL) module implements label augmentation by generating pseudo-labels for current phase samples, addressing the label absence problem. The weighted analytic classifier (WAC) derives a closed-form solution for neural networks. It introduces sample-specific weights to adaptively balance the class contribution and mitigate class imbalance. Experiments on MS-COCO and PASCAL VOC datasets demonstrate that L3A outperforms existing methods in MLCIL tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 基于类增量学习的多标签标签增广分析适应（L3A） 

---
# Unlearning Inversion Attacks for Graph Neural Networks 

**Title (ZH)**: 图神经网络的逆向攻击学习消除 

**Authors**: Jiahao Zhang, Yilong Wang, Zhiwei Zhang, Xiaorui Liu, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00808)  

**Abstract**: Graph unlearning methods aim to efficiently remove the impact of sensitive data from trained GNNs without full retraining, assuming that deleted information cannot be recovered. In this work, we challenge this assumption by introducing the graph unlearning inversion attack: given only black-box access to an unlearned GNN and partial graph knowledge, can an adversary reconstruct the removed edges? We identify two key challenges: varying probability-similarity thresholds for unlearned versus retained edges, and the difficulty of locating unlearned edge endpoints, and address them with TrendAttack. First, we derive and exploit the confidence pitfall, a theoretical and empirical pattern showing that nodes adjacent to unlearned edges exhibit a large drop in model confidence. Second, we design an adaptive prediction mechanism that applies different similarity thresholds to unlearned and other membership edges. Our framework flexibly integrates existing membership inference techniques and extends them with trend features. Experiments on four real-world datasets demonstrate that TrendAttack significantly outperforms state-of-the-art GNN membership inference baselines, exposing a critical privacy vulnerability in current graph unlearning methods. 

**Abstract (ZH)**: 基于图的学习逆向攻击：在仅拥有黑盒访问权限和部分图知识的情况下，对手能否重建已删除的边？ 

---
# Action Dependency Graphs for Globally Optimal Coordinated Reinforcement Learning 

**Title (ZH)**: 全局最优协调强化学习的动作依赖图 

**Authors**: Jianglin Ding, Jingcheng Tang, Gangshan Jing  

**Link**: [PDF](https://arxiv.org/pdf/2506.00797)  

**Abstract**: Action-dependent individual policies, which incorporate both environmental states and the actions of other agents in decision-making, have emerged as a promising paradigm for achieving global optimality in multi-agent reinforcement learning (MARL). However, the existing literature often adopts auto-regressive action-dependent policies, where each agent's policy depends on the actions of all preceding agents. This formulation incurs substantial computational complexity as the number of agents increases, thereby limiting scalability. In this work, we consider a more generalized class of action-dependent policies, which do not necessarily follow the auto-regressive form. We propose to use the `action dependency graph (ADG)' to model the inter-agent action dependencies. Within the context of MARL problems structured by coordination graphs, we prove that an action-dependent policy with a sparse ADG can achieve global optimality, provided the ADG satisfies specific conditions specified by the coordination graph. Building on this theoretical foundation, we develop a tabular policy iteration algorithm with guaranteed global optimality. Furthermore, we integrate our framework into several SOTA algorithms and conduct experiments in complex environments. The empirical results affirm the robustness and applicability of our approach in more general scenarios, underscoring its potential for broader MARL challenges. 

**Abstract (ZH)**: 基于动作依赖的个体策略：一种在多智能体强化学习中的全局最优实现新范式 

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
# Manipulating 3D Molecules in a Fixed-Dimensional SE(3)-Equivariant Latent Space 

**Title (ZH)**: 在固定维度SE(3)-等变潜在空间中操纵3D分子 

**Authors**: Zitao Chen, Yinjun Jia, Zitong Tian, Wei-Ying Ma, Yanyan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00771)  

**Abstract**: Medicinal chemists often optimize drugs considering their 3D structures and designing structurally distinct molecules that retain key features, such as shapes, pharmacophores, or chemical properties. Previous deep learning approaches address this through supervised tasks like molecule inpainting or property-guided optimization. In this work, we propose a flexible zero-shot molecule manipulation method by navigating in a shared latent space of 3D molecules. We introduce a Variational AutoEncoder (VAE) for 3D molecules, named MolFLAE, which learns a fixed-dimensional, SE(3)-equivariant latent space independent of atom counts. MolFLAE encodes 3D molecules using an SE(3)-equivariant neural network into fixed number of latent nodes, distinguished by learned embeddings. The latent space is regularized, and molecular structures are reconstructed via a Bayesian Flow Network (BFN) conditioned on the encoder's latent output. MolFLAE achieves competitive performance on standard unconditional 3D molecule generation benchmarks. Moreover, the latent space of MolFLAE enables zero-shot molecule manipulation, including atom number editing, structure reconstruction, and coordinated latent interpolation for both structure and properties. We further demonstrate our approach on a drug optimization task for the human glucocorticoid receptor, generating molecules with improved hydrophilicity while preserving key interactions, under computational evaluations. These results highlight the flexibility, robustness, and real-world utility of our method, opening new avenues for molecule editing and optimization. 

**Abstract (ZH)**: 医药化学家在优化药物时往往考虑其三维结构，并设计具有特定形状、药效团或化学性质的结构不同的分子。先前的深度学习方法通过分子填补或属性指导优化等监督任务来解决这一问题。在本项工作中，我们提出了一种灵活的零样本分子操纵方法，通过在三维分子共享的隐空间中导航来实现。我们引入了一种名为MolFLAE的三维分子变分自编码器（VAE），它学习一个固定维度且SE(3)-不变的隐空间，与原子数无关。MolFLAE使用SE(3)-不变的神经网络将三维分子编码为固定数量的隐节点，并通过学习嵌入进行区分。隐空间经过正则化处理，并通过贝叶斯流网络（BFN）根据编码器的隐空间输出重建分子结构。MolFLAE在标准的三维分子生成基准测试中实现了竞争性性能。此外，MolFLAE的隐空间支持零样本分子操纵，包括原子数编辑、结构重建以及结构和属性的协调隐空间插值。我们进一步在人糖皮质激素受体的药物优化任务中展示了该方法，生成了具有更好亲水性的分子，同时保留了关键相互作用，在计算评估中取得了良好的效果。这些结果突显了我们方法的灵活性、鲁棒性和实际应用价值，为分子编辑和优化开辟了新的途径。 

---
# Beyond Attention: Learning Spatio-Temporal Dynamics with Emergent Interpretable Topologies 

**Title (ZH)**: 超越注意力：基于 Emergent 可解释拓扑结构的时空动力学习 

**Authors**: Sai Vamsi Alisetti, Vikas Kalagi, Sanjukta Krishnagopal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00770)  

**Abstract**: Spatio-temporal forecasting is critical in applications such as traffic prediction, energy demand modeling, and weather monitoring. While Graph Attention Networks (GATs) are popular for modeling spatial dependencies, they rely on predefined adjacency structures and dynamic attention scores, introducing inductive biases and computational overhead that can obscure interpretability.
We propose InterGAT, a simplified alternative to GAT that replaces masked attention with a fully learnable, symmetric node interaction matrix, capturing latent spatial relationships without relying on fixed graph topologies. Our framework, InterGAT-GRU, which incorporates a GRU-based temporal decoder, outperforms the baseline GAT-GRU in forecasting accuracy, achieving at least a 21% improvement on the SZ-Taxi dataset and a 6% improvement on the Los-Loop dataset across all forecasting horizons (15 to 60 minutes). Additionally, we observed reduction in training time by 60-70% compared to GAT-GRU baseline.
Crucially, the learned interaction matrix reveals interpretable structure: it recovers sparse, topology-aware attention patterns that align with community structure. Spectral and clustering analyses show that the model captures both localized and global dynamics, offering insights into the functional topology driving predictions. This highlights how structure learning can simultaneously support prediction, computational efficiency, and topological interpretabil-ity in dynamic graph-based domains. 

**Abstract (ZH)**: 空时预测对于交通预测、能源需求建模和天气监测等应用至关重要。尽管图注意网络（GATs）在建模空域依赖方面广受欢迎，但它们依赖于预定义的邻接结构和动态注意分数，引入了归纳偏差和计算开销，可能模糊了可解释性。
我们提出了一种名为InterGAT的简化替代方案，它用完全可学习的对称节点交互矩阵取代了掩码注意机制，从而捕捉潜在的空域关系，而不依赖于固定图拓扑。我们的框架InterGAT-GRU结合了基于GRU的时空解码器，在预测准确性方面优于基线GAT-GRU，在SZ-Taxi数据集和Los-Loop数据集上，所有预测时长（15到60分钟）的预测准确率分别提高了至少21%和6%。此外，与GAT-GRU基线相比，训练时间减少了60-70%。
至关重要的是，学习到的交互矩阵揭示了可解释的结构：它恢复了稀疏的、拓扑意识的注意模式，与社区结构对齐。频谱和聚类分析表明，该模型捕获了局部和全局动力学，揭示了驱动预测的功能拓扑结构。这强调了结构学习如何在动态图基环境中同时支持预测、计算效率和拓扑可解释性。 

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
# ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary 

**Title (ZH)**: ArtiScene: 通过图像中介的语言驱动的艺术性3D场景生成 

**Authors**: Zeqi Gu, Yin Cui, Zhaoshuo Li, Fangyin Wei, Yunhao Ge, Jinwei Gu, Ming-Yu Liu, Abe Davis, Yifan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.00742)  

**Abstract**: Designing 3D scenes is traditionally a challenging task that demands both artistic expertise and proficiency with complex software. Recent advances in text-to-3D generation have greatly simplified this process by letting users create scenes based on simple text descriptions. However, as these methods generally require extra training or in-context learning, their performance is often hindered by the limited availability of high-quality 3D data. In contrast, modern text-to-image models learned from web-scale images can generate scenes with diverse, reliable spatial layouts and consistent, visually appealing styles. Our key insight is that instead of learning directly from 3D scenes, we can leverage generated 2D images as an intermediary to guide 3D synthesis. In light of this, we introduce ArtiScene, a training-free automated pipeline for scene design that integrates the flexibility of free-form text-to-image generation with the diversity and reliability of 2D intermediary layouts.
First, we generate 2D images from a scene description, then extract the shape and appearance of objects to create 3D models. These models are assembled into the final scene using geometry, position, and pose information derived from the same intermediary image. Being generalizable to a wide range of scenes and styles, ArtiScene outperforms state-of-the-art benchmarks by a large margin in layout and aesthetic quality by quantitative metrics. It also averages a 74.89% winning rate in extensive user studies and 95.07% in GPT-4o evaluation. Project page: this https URL 

**Abstract (ZH)**: 设计3D场景 traditionally a challenging task that demands both artistic expertise and proficiency with complex software.Recent advances in text-to-3D generation have greatly simplified this process by letting users create scenes based on simple text descriptions. However, as these methods generally require extra training or in-context learning, their performance is often hindered by the limited availability of high-quality 3D data. In contrast, modern text-to-image models learned from web-scale images can generate scenes with diverse, reliable spatial layouts and consistent, visually appealing styles. Our key insight is that instead of learning directly from 3D scenes, we can leverage generated 2D images as an intermediary to guide 3D synthesis. In light of this, we introduce ArtiScene, a training-free automated pipeline for scene design that integrates the flexibility of free-form text-to-image generation with the diversity and reliability of 2D intermediary layouts.

首先，根据场景描述生成2D图像，然后提取物体的形状和外观以创建3D模型。这些模型根据来自同一中间图像的几何形状、位置和姿态信息组装成最终场景。ArtiScene 可泛化到各种场景和风格，在广泛的定量指标上表现出优于现有基准模型的布局和美感质量，平均在大规模用户研究中获胜率为74.89%，在GPT-4o评估中为95.07%。项目页面：this https URL。 

---
# Length Aware Speech Translation for Video Dubbing 

**Title (ZH)**: 基于长度感知的语音翻译用于视频配音 

**Authors**: Harveen Singh Chadha, Aswin Shanmugam Subramanian, Vikas Joshi, Shubham Bansal, Jian Xue, Rupeshkumar Mehta, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00740)  

**Abstract**: In video dubbing, aligning translated audio with the source audio is a significant challenge. Our focus is on achieving this efficiently, tailored for real-time, on-device video dubbing scenarios. We developed a phoneme-based end-to-end length-sensitive speech translation (LSST) model, which generates translations of varying lengths short, normal, and long using predefined tags. Additionally, we introduced length-aware beam search (LABS), an efficient approach to generate translations of different lengths in a single decoding pass. This approach maintained comparable BLEU scores compared to a baseline without length awareness while significantly enhancing synchronization quality between source and target audio, achieving a mean opinion score (MOS) gain of 0.34 for Spanish and 0.65 for Korean, respectively. 

**Abstract (ZH)**: 视频配音中，将翻译音频与源音频对齐是一项重大挑战。我们专注于实现实时、设备端视频配音中的这一目标。我们开发了一种基于音素的端到端长度敏感的语音翻译（LSST）模型，该模型使用预定义标签生成不同长度的翻译（短、正常和长）。此外，我们引入了长度感知束搜索（LABS），这是一种高效的单解码过程中生成不同长度翻译的方法。该方法在保持与无长度感知基线相当的BLEU分数的同时，显著提高了源音频和目标音频之间的同步质量，分别实现了西班牙语0.34和韩语0.65的平均意见得分（MOS）提升。 

---
# MoPINNEnKF: Iterative Model Inference using generic-PINN-based ensemble Kalman filter 

**Title (ZH)**: MoPINNEnKF：基于通用PINN的迭代模型推理使用ensemble Kalman滤波器 

**Authors**: Binghang Lu, Changhong Mou, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00731)  

**Abstract**: Physics-informed neural networks (PINNs) have emerged as a powerful tool for solving forward and inverse problems involving partial differential equations (PDEs) by incorporating physical laws into the training process. However, the performance of PINNs is often hindered in real-world scenarios involving noisy observational data and missing physics, particularly in inverse problems. In this work, we propose an iterative multi-objective PINN ensemble Kalman filter (MoPINNEnKF) framework that improves the robustness and accuracy of PINNs in both forward and inverse problems by using the \textit{ensemble Kalman filter} and the \textit{non-dominated sorting genetic algorithm} III (NSGA-III). Specifically, NSGA-III is used as a multi-objective optimizer that can generate various ensemble members of PINNs along the optimal Pareto front, while accounting the model uncertainty in the solution space. These ensemble members are then utilized within the EnKF to assimilate noisy observational data. The EnKF's analysis is subsequently used to refine the data loss component for retraining the PINNs, thereby iteratively updating their parameters. The iterative procedure generates improved solutions to the PDEs. The proposed method is tested on two benchmark problems: the one-dimensional viscous Burgers equation and the time-fractional mixed diffusion-wave equation (TFMDWE). The numerical results show it outperforms standard PINNs in handling noisy data and missing physics. 

**Abstract (ZH)**: 基于物理的神经网络（PINNs）的迭代多目标集成卡尔曼滤波（MoPINNEnKF）框架：提高前反问题中的鲁棒性和准确性 

---
# Pitfalls in Evaluating Language Model Forecasters 

**Title (ZH)**: 语言模型预测器评估中的陷阱 

**Authors**: Daniel Paleka, Shashwat Goel, Jonas Geiping, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2506.00723)  

**Abstract**: Large language models (LLMs) have recently been applied to forecasting tasks, with some works claiming these systems match or exceed human performance. In this paper, we argue that, as a community, we should be careful about such conclusions as evaluating LLM forecasters presents unique challenges. We identify two broad categories of issues: (1) difficulty in trusting evaluation results due to many forms of temporal leakage, and (2) difficulty in extrapolating from evaluation performance to real-world forecasting. Through systematic analysis and concrete examples from prior work, we demonstrate how evaluation flaws can raise concerns about current and future performance claims. We argue that more rigorous evaluation methodologies are needed to confidently assess the forecasting abilities of LLMs. 

**Abstract (ZH)**: 大规模语言模型(LLMs) recently已被应用于预测任务，一些研究表明这些系统能够达到或超过人类的性能。在本文中，我们认为作为科研社区，我们应该谨慎对待这些结论，因为评估LLM预测器存在独特挑战。我们识别出两类主要问题：(1) 由于存在多种形式的时间泄漏，难以信任评估结果；(2) 从评估性能推断到实际预测的困难。通过系统分析和先前工作中的具体示例，我们展示了评估缺陷如何引发对未来和当前预测能力声明的担忧。我们认为，需要更严谨的评估方法来确信地评估LLM的预测能力。 

---
# From Local Cues to Global Percepts: Emergent Gestalt Organization in Self-Supervised Vision Models 

**Title (ZH)**: 从局部线索到全局知觉：自我监督视觉模型中的格式塔组织涌现 

**Authors**: Tianqin Li, Ziqi Wen, Leiran Song, Jun Liu, Zhi Jing, Tai Sing Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.00718)  

**Abstract**: Human vision organizes local cues into coherent global forms using Gestalt principles like closure, proximity, and figure-ground assignment -- functions reliant on global spatial structure. We investigate whether modern vision models show similar behaviors, and under what training conditions these emerge. We find that Vision Transformers (ViTs) trained with Masked Autoencoding (MAE) exhibit activation patterns consistent with Gestalt laws, including illusory contour completion, convexity preference, and dynamic figure-ground segregation. To probe the computational basis, we hypothesize that modeling global dependencies is necessary for Gestalt-like organization. We introduce the Distorted Spatial Relationship Testbench (DiSRT), which evaluates sensitivity to global spatial perturbations while preserving local textures. Using DiSRT, we show that self-supervised models (e.g., MAE, CLIP) outperform supervised baselines and sometimes even exceed human performance. ConvNeXt models trained with MAE also exhibit Gestalt-compatible representations, suggesting such sensitivity can arise without attention architectures. However, classification finetuning degrades this ability. Inspired by biological vision, we show that a Top-K activation sparsity mechanism can restore global sensitivity. Our findings identify training conditions that promote or suppress Gestalt-like perception and establish DiSRT as a diagnostic for global structure sensitivity across models. 

**Abstract (ZH)**: 人类视觉利用闭合、邻近和图形背景区分等格式塔原则将局部线索组织成统一的整体形式——这些功能依赖于全局空间结构。我们探讨现代视觉模型是否表现出类似的行为，以及在哪些训练条件下这些行为出现。我们发现使用掩蔽自编码（MAE）训练的视觉变压器（ViTs）显示出与格式塔定律一致的激活模式，包括虚假轮廓完成、凸性偏好和动态图形背景区分。为了探究其计算基础，我们假设建模全局依赖性对于格式塔似的组织是必要的。我们引入了变形空间关系测试床（DiSRT），该测试床评估对全局空间扰动的敏感性，同时保持局部纹理不变。使用DiSRT，我们展示了自监督模型（例如，MAE、CLIP）优于监督基线，并且有时甚至超过人类表现。使用MAE训练的ConvNeXt模型也表现出格式塔兼容的表示，表明这种敏感性可以在不依赖注意力架构的情况下出现。然而，分类微调降低了这种能力。借鉴生物视觉，我们展示了Top-K激活稀疏机制可以恢复全局敏感性。我们的研究确定了促进或抑制格式塔似的感知的训练条件，并将DiSRT确立为模型中全局结构敏感性的一种诊断工具。 

---
# An LLM Agent for Functional Bug Detection in Network Protocols 

**Title (ZH)**: 功能性 bug 检测在网络协议中的 LLAM 代理 

**Authors**: Mingwei Zheng, Chengpeng Wang, Xuwei Liu, Jinyao Guo, Shiwei Feng, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00714)  

**Abstract**: Functional correctness is critical for ensuring the reliability and security of network protocol implementations. Functional bugs, instances where implementations diverge from behaviors specified in RFC documents, can lead to severe consequences, including faulty routing, authentication bypasses, and service disruptions. Detecting these bugs requires deep semantic analysis across specification documents and source code, a task beyond the capabilities of traditional static analysis tools. This paper introduces RFCScan, an autonomous agent that leverages large language models (LLMs) to detect functional bugs by checking conformance between network protocol implementations and their RFC specifications. Inspired by the human auditing procedure, RFCScan comprises two key components: an indexing agent and a detection agent. The former hierarchically summarizes protocol code semantics, generating semantic indexes that enable the detection agent to narrow down the scanning scope. The latter employs demand-driven retrieval to iteratively collect additional relevant data structures and functions, eventually identifying potential inconsistencies with the RFC specifications effectively. We evaluate RFCScan across six real-world network protocol implementations. RFCScan identifies 47 functional bugs with 81.9% precision, of which 20 bugs have been confirmed or fixed by developers. 

**Abstract (ZH)**: 基于RFC规范的功能正确性对于确保网络协议实现的可靠性和安全性至关重要。功能漏洞会导致严重的后果，包括误路由、身份认证绕过和服务中断。检测这些漏洞需要对规范文档和源代码进行深入语义分析，这是传统静态分析工具无法完成的任务。本文介绍了RFCScan，这是一种自主代理，利用大规模语言模型（LLMs）通过检查网络协议实现与RFC规范的一致性来检测功能漏洞。RFCScan由两个关键组件组成：索引代理和检测代理。索引代理层次结构地总结协议代码语义，生成语义索引，使检测代理能够缩小扫描范围。检测代理采用需求驱动检索，迭代收集相关数据结构和函数，最终有效地识别潜在的与RFC规范的一致性问题。我们在六个实际网络协议实现中评估了RFCScan。RFCScan发现了47个功能漏洞，精确率为81.9%，其中20个漏洞已被开发者确认或修复。 

---
# From Argumentative Text to Argument Knowledge Graph: A New Framework for Structured Argumentation 

**Title (ZH)**: 从论证文本到论证知识图谱：一种结构化论证的新框架 

**Authors**: Debarati Bhattacharjee, Ashish Anand  

**Link**: [PDF](https://arxiv.org/pdf/2506.00713)  

**Abstract**: This paper presents a framework to convert argumentative texts into argument knowledge graphs (AKG). Starting with basic annotations of argumentative components (ACs) and argumentative relations (ARs), we enrich the information by constructing a knowledge base (KB) graph with metadata attributes for nodes. Next, we use premises and inference rules from the KB to form arguments by applying modus ponens. From these arguments, we create an AKG. The nodes and edges of the AKG have attributes that capture important argumentative features. We also find missing inference rules by identifying markers. This makes it possible to identify undercut attacks that were previously undetectable in existing datasets. The AKG gives a graphical view of the argumentative structure that is easier to understand than theoretical formats. It also prepares the ground for future reasoning tasks, including checking the coherence of arguments and identifying opportunities for revision. For this, it is important to find indirect relations, many of which are implicit. Our proposed AKG format, with annotated inference rules and modus ponens, will help reasoning models learn the implicit indirect relations that require inference over arguments and the relations between them. 

**Abstract (ZH)**: 本文提出了一种将论辩文本转换为论辩知识图谱（AKG）的框架。从基本的论辩成分（ACs）和论辩关系（ARs）的标注开始，通过构建包含元数据属性的节点的知识库（KB）图来丰富信息。接着，我们利用KB中的前提和推理规则应用模态斯蓬森方法形成论辩，从这些论辩中构建AKG。AKG中的节点和边具有能够捕捉重要论辩特征的属性。我们还通过识别标记来发现缺失的推理规则，这使得之前在现有数据集中难以检测到的削弱攻击变得可识别。AKG提供了比理论格式更容易理解的论辩结构图示，也为未来的推理任务做了准备，包括检查论辩的一致性和识别修订机会。为此，找到许多隐含的间接关系很重要。我们提出的带有标注推理规则和模态斯蓬森方法的AKG格式，将有助于推理模型学习需要在论辩及其之间关系上进行推理的隐含间接关系。 

---
# QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training 

**Title (ZH)**: QoQ-Med: 建立具有领域意识GRPO训练的多模态临床基础模型 

**Authors**: Wei Dai, Peilin Chen, Chanakya Ekbote, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00711)  

**Abstract**: Clinical decision-making routinely demands reasoning over heterogeneous data, yet existing multimodal language models (MLLMs) remain largely vision-centric and fail to generalize across clinical specialties. To bridge this gap, we introduce QoQ-Med-7B/32B, the first open generalist clinical foundation model that jointly reasons across medical images, time-series signals, and text reports. QoQ-Med is trained with Domain-aware Relative Policy Optimization (DRPO), a novel reinforcement-learning objective that hierarchically scales normalized rewards according to domain rarity and modality difficulty, mitigating performance imbalance caused by skewed clinical data distributions. Trained on 2.61 million instruction tuning pairs spanning 9 clinical domains, we show that DRPO training boosts diagnostic performance by 43% in macro-F1 on average across all visual domains as compared to other critic-free training methods like GRPO. Furthermore, with QoQ-Med trained on intensive segmentation data, it is able to highlight salient regions related to the diagnosis, with an IoU 10x higher than open models while reaching the performance of OpenAI o4-mini. To foster reproducibility and downstream research, we release (i) the full model weights, (ii) the modular training pipeline, and (iii) all intermediate reasoning traces at this https URL. 

**Abstract (ZH)**: 临床决策需要综合处理异构数据，现有跨模态语言模型主要侧重视觉信息且在临床专科间难以泛化。为了弥合这一差距，我们引入了QoQ-Med-7B/32B，这是首个能够联合推理医学图像、时序信号和文本报告的开放通用临床基础模型。QoQ-Med 使用一种名为域意识相对策略优化 (DRPO) 的新型强化学习目标进行训练，该目标根据领域稀有性和模态难度逐级调整标准化奖励，从而缓解由临床数据分布偏斜引起的性能失衡问题。在包含9个临床领域的261万条指令调优对的数据上进行训练，我们展示了DRPO训练相比其他无批评家训练方法（如GRPO）在宏观F1的平均诊断性能上提升了43%。此外，利用QoQ-Med 对密集分割数据进行训练，能够突出显示与诊断相关的显著区域，并且与开源模型相比，交集分割指标高出10倍，同时达到OpenAI o4-mini的性能。为了促进可重复性和下游研究，我们在此公开发布了（i）完整模型权重，（ii）模块化训练流水线，以及（iii）所有中间推理轨迹。 

---
# Bayesian Inference of Training Dataset Membership 

**Title (ZH)**: 基于贝叶斯推断的训练数据集成员识别 

**Authors**: Yongchao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00701)  

**Abstract**: Determining whether a dataset was part of a machine learning model's training data pool can reveal privacy vulnerabilities, a challenge often addressed through membership inference attacks (MIAs). Traditional MIAs typically require access to model internals or rely on computationally intensive shadow models. This paper proposes an efficient, interpretable and principled Bayesian inference method for membership inference. By analyzing post-hoc metrics such as prediction error, confidence (entropy), perturbation magnitude, and dataset statistics from a trained ML model, our approach computes posterior probabilities of membership without requiring extensive model training. Experimental results on synthetic datasets demonstrate the method's effectiveness in distinguishing member from non-member datasets. Beyond membership inference, this method can also detect distribution shifts, offering a practical and interpretable alternative to existing approaches. 

**Abstract (ZH)**: 确定数据集是否为机器学习模型训练数据池的一部分可以揭示隐私漏洞，这一挑战通常通过成员 inference 攻击（MIAs）来应对。本文提出了一种高效、可解释且原理性的贝叶斯推理方法用于成员 inference。通过分析训练后的 ML 模型的预测误差、置信度（熵）、扰动幅度以及数据集统计信息等后验指标，本方法可在不需要大量模型训练的情况下计算成员 posterior 概率。实验结果表明，该方法在区分成员数据集和非成员数据集方面具有有效性。除此之外，该方法还可以检测分布偏移，提供了一种实用且可解释的替代现有方法的选择。 

---
# Measuring Faithfulness and Abstention: An Automated Pipeline for Evaluating LLM-Generated 3-ply Case-Based Legal Arguments 

**Title (ZH)**: 测量忠实度与弃权：评估大语言模型生成的三元案例法法律论点的自动化管道 

**Authors**: Li Zhang, Morgan Gray, Jaromir Savelka, Kevin D. Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2506.00694)  

**Abstract**: Large Language Models (LLMs) demonstrate potential in complex legal tasks like argument generation, yet their reliability remains a concern. Building upon pilot work assessing LLM generation of 3-ply legal arguments using human evaluation, this paper introduces an automated pipeline to evaluate LLM performance on this task, specifically focusing on faithfulness (absence of hallucination), factor utilization, and appropriate abstention. We define hallucination as the generation of factors not present in the input case materials and abstention as the model's ability to refrain from generating arguments when instructed and no factual basis exists. Our automated method employs an external LLM to extract factors from generated arguments and compares them against the ground-truth factors provided in the input case triples (current case and two precedent cases). We evaluated eight distinct LLMs on three tests of increasing difficulty: 1) generating a standard 3-ply argument, 2) generating an argument with swapped precedent roles, and 3) recognizing the impossibility of argument generation due to lack of shared factors and abstaining. Our findings indicate that while current LLMs achieve high accuracy (over 90%) in avoiding hallucination on viable argument generation tests (Tests 1 & 2), they often fail to utilize the full set of relevant factors present in the cases. Critically, on the abstention test (Test 3), most models failed to follow instructions to stop, instead generating spurious arguments despite the lack of common factors. This automated pipeline provides a scalable method for assessing these crucial LLM behaviors, highlighting the need for improvements in factor utilization and robust abstention capabilities before reliable deployment in legal settings. Project page: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂法律任务如论点生成中展现出潜力，但其可靠性仍是一个关注点。基于初步研究评估LLM生成三段式法律论点的人类评估结果，本文提出了一种自动评估管道，用于评估LLM在该任务上的性能，特别是重点在于忠实性（无幻觉）、因素利用以及适当的回避。我们将幻觉定义为生成输入案例材料中未出现的因素，回避定义为在没有事实依据的情况下模型停止生成论点的能力。我们的自动化方法使用外部LLM从生成的论点中提取因素，并将其与输入案例三元组（当前案例和两个先例案例）中提供的真实因素进行比较。我们对八种不同的LLM进行了三项难度递增的测试评估：1）生成标准三段式论点，2）生成角色互换的论点，3）识别人因缺乏共享因素而不能生成论点并回避。我们的研究结果显示，当前的LLM在有效的论点生成测试（测试1和测试2）中避免幻觉的准确性超过90%，但在利用相关因素方面常常未能充分利用。特别是在回避测试（测试3）中，大多数模型未能遵循停止生成论点的指示，而是生成了无效的论点，尽管缺乏共同因素。该自动化管道提供了一种可扩展的方法来评估这些关键的LLM行为，突显了在法律环境中可靠部署之前需要改进因素利用能力和坚定的回避能力。项目页面：[this https URL]。 

---
# Optimizing Sensory Neurons: Nonlinear Attention Mechanisms for Accelerated Convergence in Permutation-Invariant Neural Networks for Reinforcement Learning 

**Title (ZH)**: 优化感觉神经元：排列不变神经网络中加速收敛的非线性注意力机制 

**Authors**: Junaid Muzaffar, Ahsan Adeel, Khubaib Ahmed, Ingo Frommholz, Zeeshan Pervez, Ahsan ul Haq  

**Link**: [PDF](https://arxiv.org/pdf/2506.00691)  

**Abstract**: Training reinforcement learning (RL) agents often requires significant computational resources and extended training times. To address this, we build upon the foundation laid by Google Brain's Sensory Neuron, which introduced a novel neural architecture for reinforcement learning tasks that maintained permutation in-variance in the sensory neuron system. While the baseline model demonstrated significant performance improvements over traditional approaches, we identified opportunities to enhance the efficiency of the learning process further. We propose a modified attention mechanism incorporating a non-linear transformation of the key vectors (K) using a mapping function, resulting in a new set of key vectors (K'). This non-linear mapping enhances the representational capacity of the attention mechanism, allowing the model to encode more complex feature interactions and accelerating convergence without compromising performance. Our enhanced model demonstrates significant improvements in learning efficiency, showcasing the potential for non-linear attention mechanisms in advancing reinforcement learning algorithms. 

**Abstract (ZH)**: 训练强化学习（RL）代理通常需要大量的计算资源和较长的训练时间。为了解决这一问题，我们在此基础上构建了Google Brain的感官神经元所提出的、具有感知不变性的新颖神经架构。尽管基线模型在传统方法上显著提高了性能，但我们发现了进一步提高学习过程效率的机会。我们提出了一种修改后的注意机制，通过使用映射函数对键向量（K）进行非线性变换，生成新的键向量（K'）。这种非线性映射增强了注意机制的表现能力，使模型能够编码更复杂的特征交互，并加速收敛而不会牺牲性能。我们的增强模型在学习效率上显示出显著的改进，展示了非线性注意机制在推动强化学习算法发展方面的潜力。 

---
# Existing Large Language Model Unlearning Evaluations Are Inconclusive 

**Title (ZH)**: 现有大规模语言模型遗忘评估结果不具决定性 

**Authors**: Zhili Feng, Yixuan Even Xu, Alexander Robey, Robert Kirk, Xander Davies, Yarin Gal, Avi Schwarzschild, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2506.00688)  

**Abstract**: Machine unlearning aims to remove sensitive or undesired data from large language models. However, recent studies suggest that unlearning is often shallow, claiming that removed knowledge can easily be recovered. In this work, we critically examine standard unlearning evaluation practices and uncover key limitations that shake our trust in those findings. First, we show that some evaluations introduce substantial new information into the model, potentially masking true unlearning performance by re-teaching the model during testing. Second, we demonstrate that evaluation outcomes vary significantly across tasks, undermining the generalizability of current evaluation routines. Finally, we find that many evaluations rely on spurious correlations, making their results difficult to trust and interpret. Taken together, these issues suggest that current evaluation protocols may both overstate and understate unlearning success. To address this, we propose two principles for future unlearning evaluations: minimal information injection and downstream task awareness. We validate these principles through a series of targeted experiments, showing how violations of each can lead to misleading conclusions. 

**Abstract (ZH)**: 机器卸载旨在从大型语言模型中移除敏感或不希望的数据。然而，近期的研究表明卸载往往是浅层的，声称删除的知识可以很容易被恢复。在本项研究中，我们对标准卸载评估实践进行了批判性审查，并揭示了一系列关键限制，这些限制动摇了我们对这些发现的信任。首先，我们展示了某些评估引入了大量新的信息，可能导致在测试过程中重新“教导”模型，从而掩盖真实的卸载性能。其次，我们证明了评估结果在不同任务上存在显著差异，削弱了当前评估流程的一般适用性。最后，我们发现许多评估依赖于虚假的相关性，使得其结果难以信任和解释。综上所述，这些问题表明当前的评估协议可能既高估又低估了卸载的成功。为解决这一问题，我们提出了两项原则，用于未来卸载评估：最小信息注入和下游任务意识。我们通过一系列针对性的实验验证了这些原则，展示了违反每项原则可能导致误导性结论。 

---
# CineMA: A Foundation Model for Cine Cardiac MRI 

**Title (ZH)**: CineMA: 电影磁共振成像心脏基础模型 

**Authors**: Yunguan Fu, Weixi Yi, Charlotte Manisty, Anish N Bhuva, Thomas A Treibel, James C Moon, Matthew J Clarkson, Rhodri Huw Davies, Yipeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00679)  

**Abstract**: Cardiac magnetic resonance (CMR) is a key investigation in clinical cardiovascular medicine and has been used extensively in population research. However, extracting clinically important measurements such as ejection fraction for diagnosing cardiovascular diseases remains time-consuming and subjective. We developed CineMA, a foundation AI model automating these tasks with limited labels. CineMA is a self-supervised autoencoder model trained on 74,916 cine CMR studies to reconstruct images from masked inputs. After fine-tuning, it was evaluated across eight datasets on 23 tasks from four categories: ventricle and myocardium segmentation, left and right ventricle ejection fraction calculation, disease detection and classification, and landmark localisation. CineMA is the first foundation model for cine CMR to match or outperform convolutional neural networks (CNNs). CineMA demonstrated greater label efficiency than CNNs, achieving comparable or better performance with fewer annotations. This reduces the burden of clinician labelling and supports replacing task-specific training with fine-tuning foundation models in future cardiac imaging applications. Models and code for pre-training and fine-tuning are available at this https URL, democratising access to high-performance models that otherwise require substantial computational resources, promoting reproducibility and accelerating clinical translation. 

**Abstract (ZH)**: 心脏磁共振成像（CMR）是临床心血管医学中的关键检查方法，在人口研究中得到了广泛的应用。然而，提取如射血分数等临床重要测量以诊断心血管疾病仍耗时且主观。我们开发了CineMA，这是一种基于有限标签自动完成这些任务的基础AI模型。CineMA是一种在74,916例心脏MRI研究上自我监督训练的自编码器模型，用于从掩码输入中重建图像。经过微调后，它在八个数据集上的23项跨四个类别（心室和心肌分割、左心室和右心室射血分数计算、疾病检测和分类、以及解剖标志定位）的任务上进行了评估。CineMA是第一个能够与卷积神经网络（CNNs）匹敌或超越的用于心脏电影MRI的基础模型，展示了比CNNs更高的标记效率，在更少标注的情况下实现类似或更好的性能，从而减轻了临床人员的标注负担，并支持在未来的心脏成像应用中使用基础模型的微调而非特定任务的训练。CineMA模型和训练代码可在以下网址获取，促进了高性能模型的民主化访问，促进了研究的可复制性并加速了临床转化。 

---
# SafeTuneBed: A Toolkit for Benchmarking LLM Safety Alignment in Fine-Tuning 

**Title (ZH)**: SafeTuneBed: 一种评估fine-tuning过程中LLM安全对齐基准的工具包 

**Authors**: Saad Hossain, Samanvay Vajpayee, Sirisha Rambhatla  

**Link**: [PDF](https://arxiv.org/pdf/2506.00676)  

**Abstract**: As large language models (LLMs) become ubiquitous, parameter-efficient fine-tuning methods and safety-first defenses have proliferated rapidly. However, the number of approaches and their recent increase have resulted in diverse evaluations-varied datasets, metrics, and inconsistent threat settings-making it difficult to fairly compare safety, utility, and robustness across methods. To address this, we introduce SafeTuneBed, a benchmark and toolkit unifying fine-tuning and defense evaluation. SafeTuneBed (i) curates a diverse repository of multiple fine-tuning datasets spanning sentiment analysis, question-answering, multi-step reasoning, and open-ended instruction tasks, and allows for the generation of harmful-variant splits; (ii) enables integration of state-of-the-art defenses, including alignment-stage immunization, in-training safeguards, and post-tuning repair; and (iii) provides evaluators for safety (attack success rate, refusal consistency) and utility. Built on Python-first, dataclass-driven configs and plugins, SafeTuneBed requires minimal additional code to specify any fine-tuning regime, defense method, and metric suite, while ensuring end-to-end reproducibility. We showcase its value by benchmarking representative defenses across varied poisoning scenarios and tasks. By standardizing data, code, and metrics, SafeTuneBed is the first focused toolkit of its kind to accelerate rigorous and comparable research in safe LLM fine-tuning. Code is available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）日益普及，参数高效微调方法和安全优先防护措施迅速增长。然而，方法的多样性和近期的增长导致了评估的差异性，包括不同的数据集、指标和不一致的威胁设定，使得公平比较安全、实用性和鲁棒性变得困难。为解决这一问题，我们引入了SafeTuneBed，一个统一微调和防护评估基准与工具包。SafeTuneBed (i) 精心整理了涵盖情感分析、问答、多步推理和开放式指令任务的多样化微调数据集，并允许生成有害变体分割；(ii) 支持集成最新的防护措施，包括对齐阶段免疫、训练中防护和微调后修复；(iii) 提供安全性和实用性评估工具（攻击成功率、拒绝一致性）。SafeTuneBed 基于 Python 编程，使用数据类驱动配置和插件，只需最少的额外代码即可指定任何微调方案、防护方法和指标套件，确保端到端可重复性。我们通过在各种中毒场景和任务中评估代表性防护措施展示了其价值。通过标准化数据、代码和指标，SafeTuneBed 成为了首个专注于加速安全的大规模语言模型微调严谨且可比研究的工具包。代码可在以下链接获取：this https URL。 

---
# Thinking Out of the Box: Hybrid SAT Solving by Unconstrained Continuous Optimization 

**Title (ZH)**: 打破常规：基于不受约束的连续优化的混合SAT求解 

**Authors**: Zhiwei Zhang, Samy Wu Fung, Anastasios Kyrillidis, Stanley Osher, Moshe Y. Vardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00674)  

**Abstract**: The Boolean satisfiability (SAT) problem lies at the core of many applications in combinatorial optimization, software verification, cryptography, and machine learning. While state-of-the-art solvers have demonstrated high efficiency in handling conjunctive normal form (CNF) formulas, numerous applications require non-CNF (hybrid) constraints, such as XOR, cardinality, and Not-All-Equal constraints. Recent work leverages polynomial representations to represent such hybrid constraints, but it relies on box constraints that can limit the use of powerful unconstrained optimizers. In this paper, we propose unconstrained continuous optimization formulations for hybrid SAT solving by penalty terms. We provide theoretical insights into when these penalty terms are necessary and demonstrate empirically that unconstrained optimizers (e.g., Adam) can enhance SAT solving on hybrid benchmarks. Our results highlight the potential of combining continuous optimization and machine-learning-based methods for effective hybrid SAT solving. 

**Abstract (ZH)**: 布尔可满足性（SAT）问题在组合优化、软件验证、密码学和机器学习等多个领域的应用中处于核心地位。尽管最先进的求解器在处理合取范式（CNF）公式时表现出高效性，但许多应用需要非CNF（混合）约束，如XOR、基数和Not-All-Equal约束。最近的研究利用多项式表示法来表示这些混合约束，但这种方法依赖于盒约束，这可能会限制使用强大的无约束优化器的能力。本文提出了一种通过惩罚项的无约束连续优化形式来解决混合SAT问题。我们提供了这些惩罚项必要的理论见解，并通过实验证明，无约束优化器（如Adam）可以提升混合基准上的SAT求解性能。我们的结果突显了结合连续优化和基于机器学习的方法在有效混合SAT求解中的潜力。 

---
# Differential Privacy for Deep Learning in Medicine 

**Title (ZH)**: 医学中深度学习的差分隐私保护 

**Authors**: Marziyeh Mohammadi, Mohsen Vejdanihemmat, Mahshad Lotfinia, Mirabela Rusu, Daniel Truhn, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00660)  

**Abstract**: Differential privacy (DP) is a key technique for protecting sensitive patient data in medical deep learning (DL). As clinical models grow more data-dependent, balancing privacy with utility and fairness has become a critical challenge. This scoping review synthesizes recent developments in applying DP to medical DL, with a particular focus on DP-SGD and alternative mechanisms across centralized and federated settings. Using a structured search strategy, we identified 74 studies published up to March 2025. Our analysis spans diverse data modalities, training setups, and downstream tasks, and highlights the tradeoffs between privacy guarantees, model accuracy, and subgroup fairness. We find that while DP-especially at strong privacy budgets-can preserve performance in well-structured imaging tasks, severe degradation often occurs under strict privacy, particularly in underrepresented or complex modalities. Furthermore, privacy-induced performance gaps disproportionately affect demographic subgroups, with fairness impacts varying by data type and task. A small subset of studies explicitly addresses these tradeoffs through subgroup analysis or fairness metrics, but most omit them entirely. Beyond DP-SGD, emerging approaches leverage alternative mechanisms, generative models, and hybrid federated designs, though reporting remains inconsistent. We conclude by outlining key gaps in fairness auditing, standardization, and evaluation protocols, offering guidance for future work toward equitable and clinically robust privacy-preserving DL systems in medicine. 

**Abstract (ZH)**: 差分隐私（DP）是医疗深度学习（DL）中保护敏感患者数据的关键技术。随着临床模型对数据的依赖性增加，平衡隐私与效用和公平性已成为一个关键挑战。本综述总结了近年来将DP应用于医疗DL的最新发展，特别关注中央和联邦设置下的DP-SGD和替代机制。通过结构化的检索策略，我们识别了截至2025年3月发表的74项研究。我们的分析涵盖了多种数据模态、训练配置和下游任务，并突出了隐私保证、模型准确性和子群体公平性之间的权衡。研究发现，尽管差分隐私（特别是较强隐私预算下）可以保持良好结构成像任务的性能，但在严格的隐私保护条件下，特别是在未充分代表或复杂的模态下，通常会出现严重的性能下降。此外，由隐私引起的表现差距不成比例地影响人口子群体，其影响程度因数据类型和任务而异。仅有一小部分研究通过子群体分析或公平性指标明确地处理了这些权衡，而大多数研究未提及。除了DP-SGD，新兴方法还利用了替代机制、生成模型和混合联邦设计，但报告仍不一致。我们最终概述了公平审计、标准化和评估协议的关键缺口，为未来工作提供指导，旨在建立在医学中公平且临床稳健的隐私保护DL系统。 

---
# Sarc7: Evaluating Sarcasm Detection and Generation with Seven Types and Emotion-Informed Techniques 

**Title (ZH)**: Sarc7: 七种类型与情感导向技术的-Trump-讽喻检测与生成评估 

**Authors**: Lang Xiong, Raina Gao, Alyssa Jeong, Yicheng Fu, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00658)  

**Abstract**: Sarcasm is a form of humor where expressions convey meanings opposite to their literal interpretations. Classifying and generating sarcasm using large language models is vital for interpreting human communication. Sarcasm poses challenges for computational models, due to its nuanced nature. We introduce Sarc7, a benchmark that classifies 7 types of sarcasm: self-deprecating, brooding, deadpan, polite, obnoxious, raging, and manic by annotating entries of the MUStARD dataset. Classification was evaluated using zero-shot, few-shot, chain-of-thought (CoT), and a novel emotion-based prompting technique. We propose an emotion-based generation method developed by identifying key components of sarcasm-incongruity, shock value, and context dependency. Our classification experiments show that Gemini 2.5, using emotion-based prompting, outperforms other setups with an F1 score of 0.3664. Human evaluators preferred our emotion-based prompting, with 38.46% more successful generations than zero-shot prompting. 

**Abstract (ZH)**: 讽刺是一种语气与其字面含义相反的幽默形式。使用大型语言模型对讽刺进行分类和生成对于解读人类通信至关重要。由于讽刺的细微差别，它对计算模型构成挑战。我们引入了Sarc7基准，通过注释MUStARD数据集的条目，对7种类型的讽刺进行分类：自我贬低、沉思、无趣、礼貌、讨厌、愤怒和狂热。分类评估使用了零样本、少量样本、逐步推理（CoT）以及一种新型的情感提示技术。我们提出了一种基于情感的生成方法，通过识别讽刺不一致、冲击价值和情境依存性等关键组件。我们的分类实验表明，使用情感提示的Gemini 2.5在F1分数上表现最佳，为0.3664。人类评估者更偏好我们的情感提示，成功生成的比例比零样本提示高38.46%。 

---
# Permutation-Invariant Transformer Neural Architectures for Set-Based Indoor Localization Using Learned RSSI Embeddings 

**Title (ZH)**: 基于学习到的RSSI嵌入的排列不变变压器神经架构的基于集合的室内定位 

**Authors**: Aris J. Aristorenas  

**Link**: [PDF](https://arxiv.org/pdf/2506.00656)  

**Abstract**: We propose a permutation-invariant neural architecture for indoor localization using RSSI scans from Wi-Fi access points. Each scan is modeled as an unordered set of (BSSID, RSSI) pairs, where BSSIDs are mapped to learned embeddings and concatenated with signal strength. These are processed by a Set Transformer, enabling the model to handle variable-length, sparse inputs while learning attention- based representations over access point relationships. We evaluate the model on a dataset collected across a campus environment consisting of six buildings. Results show that the model accurately recovers fine-grained spatial structure and maintains performance across physically distinct domains. In our experiments, a simple LSTM consistently outperformed all other models, achieving the lowest mean localization error across three tasks (E1 - E3), with average errors as low as 2.23 m. The Set Transformer performed competitively, ranking second in every experiment and outperforming the MLP, RNN, and basic attention models, particularly in scenarios involving multiple buildings (E2) and multiple floors (E3). Performance degraded most in E2, where signal conditions varied substantially across buildings, highlighting the importance of architectural robustness to domain diversity. This work demonstrates that set-based neural models are a natural fit for signal-based localization, offering a principled approach to handling sparse, unordered inputs in real-world positioning tasks. 

**Abstract (ZH)**: 基于Wi-Fi接入点RSSI扫描的置换不变神经架构的室内定位 

---
# Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models 

**Title (ZH)**: 线性表示迁移性假设：利用小型模型引导大型模型 

**Authors**: Femi Bello, Anubrata Das, Fanzhi Zeng, Fangcong Yin, Leqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00653)  

**Abstract**: It has been hypothesized that neural networks with similar architectures trained on similar data learn shared representations relevant to the learning task. We build on this idea by extending the conceptual framework where representations learned across models trained on the same data can be expressed as linear combinations of a \emph{universal} set of basis features. These basis features underlie the learning task itself and remain consistent across models, regardless of scale. From this framework, we propose the \textbf{Linear Representation Transferability (LRT)} Hypothesis -- that there exists an affine transformation between the representation spaces of different models. To test this hypothesis, we learn affine mappings between the hidden states of models of different sizes and evaluate whether steering vectors -- directions in hidden state space associated with specific model behaviors -- retain their semantic effect when transferred from small to large language models using the learned mappings. We find strong empirical evidence that such affine mappings can preserve steering behaviors. These findings suggest that representations learned by small models can be used to guide the behavior of large models, and that the LRT hypothesis may be a promising direction on understanding representation alignment across model scales. 

**Abstract (ZH)**: 具有相似架构的神经网络在相似数据上训练时学习到与学习任务相关的共享表示吗？——线性表示转移性（LRT）假设 

---
# Clinical Annotations for Automatic Stuttering Severity Assessment 

**Title (ZH)**: 临床标注用于自动 stuttering 严重程度评估 

**Authors**: Ana Rita Valente, Rufael Marew, Hawau Olamide Toyin, Hamdan Al-Ali, Anelise Bohnen, Inma Becerra, Elsa Marta Soares, Goncalo Leal, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.00644)  

**Abstract**: Stuttering is a complex disorder that requires specialized expertise for effective assessment and treatment. This paper presents an effort to enhance the FluencyBank dataset with a new stuttering annotation scheme based on established clinical standards. To achieve high-quality annotations, we hired expert clinicians to label the data, ensuring that the resulting annotations mirror real-world clinical expertise. The annotations are multi-modal, incorporating audiovisual features for the detection and classification of stuttering moments, secondary behaviors, and tension scores. In addition to individual annotations, we additionally provide a test set with highly reliable annotations based on expert consensus for assessing individual annotators and machine learning models. Our experiments and analysis illustrate the complexity of this task that necessitates extensive clinical expertise for valid training and evaluation of stuttering assessment models. 

**Abstract (ZH)**: 结巴症是一种复杂的障碍，需要专门的专家才能进行有效的评估和治疗。本文提出了一种努力扩展FluencyBank数据集的方法，基于现有的临床标准建立了新的结巴标注方案。为了获得高质量的标注，我们聘请了专家临床人员对数据进行了标注，确保最终的标注反映了真实的临床专业知识。这些标注是多模态的，结合了音频视觉特征以检测和分类结巴时刻、附带行为以及紧张度评分。除了个体标注外，我们还提供了一个基于专家共识的可靠测试集，用于评估个体标注者和机器学习模型。我们的实验和分析说明了这一任务的复杂性，需要广泛的临床专业知识来进行有效的训练和评估结巴评估模型。 

---
# SATA-BENCH: Select All That Apply Benchmark for Multiple Choice Questions 

**Title (ZH)**: SATA-BENCH: 全选适用基准测试用于多项选择题 

**Authors**: Weijie Xu, Shixian Cui, Xi Fang, Chi Xue, Stephanie Eckman, Chandan Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.00643)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on single-answer multiple-choice tasks, yet many real-world problems require identifying all correct answers from a set of options. This capability remains underexplored. We introduce SATA-BENCH, the first dedicated benchmark for evaluating LLMs on Select All That Apply (SATA) questions across diverse domains, including reading comprehension, law, and biomedicine. Our evaluation of 27 open-source and proprietary models reveals a significant gap: even the strongest model achieves only 41.8% exact match, exposing LLMs' inability to reliably identify all correct answers. We find that this weakness stems from two core challenges: selection bias - models favor certain choices regardless of content, and count bias - models fail to predict the correct number of answers. To address these issues, we propose Choice Funnel, a decoding strategy that combines token debiasing with adaptive thresholding to guide models toward complete and accurate selections. Choice Funnel achieves up to 29% higher exact match than competitive baselines while reducing inference cost by over 64%. Our findings expose fundamental limitations in current LLMs and introduce a new framework for diagnosing and improving multi-answer reasoning. We release SATA-BENCH and Choice Funnel to promote LLM development for robust decision-making in realistic, multi-answer applications. 

**Abstract (ZH)**: SATA-BENCH：面向选所有适用项问题的LLM评估基准与Choice Funnel解码策略 

---
# Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics 

**Title (ZH)**: 基于输出分布特性提高文本生成中置信分数的校准 

**Authors**: Lorenzo Jaime Yu Flores, Ori Ernst, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.00637)  

**Abstract**: Well-calibrated model confidence scores can improve the usefulness of text generation models. For example, users can be prompted to review predictions with low confidence scores, to prevent models from returning bad or potentially dangerous predictions. However, confidence metrics are not always well calibrated in text generation. One reason is that in generation, there can be many valid answers, which previous methods do not always account for. Hence, a confident model could distribute its output probability among multiple sequences because they are all valid. We propose task-agnostic confidence metrics suited to generation, which rely solely on the probabilities associated with the model outputs without the need for further fine-tuning or heuristics. Using these, we are able to improve the calibration of BART and Flan-T5 on summarization, translation, and QA datasets. 

**Abstract (ZH)**: Well-calibrated模型信心分数可以提高文本生成模型的实用性。例如，用户可以被提示审查低信心分数的预测，以防止模型返回差劲或潜在危险的预测。然而，在文本生成中，信心度量并不总是很好地校准。一个原因是生成过程中可能存在许多有效的答案，之前的 方法并没有总是考虑到这一点。因此，一个有信心的模型可能会将其输出概率分布在多个序列中，因为它们都是有效的。我们提出了适用于生成的任务无关的信心度量，这些度量仅依赖于模型输出相关的概率，而无需进一步微调或启发式方法。利用这些方法，我们能够改善BART和Flan-T5在总结、翻译和问答数据集上的校准。 

---
# Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting 

**Title (ZH)**: 学习与校准：探索时空预测的测试时计算 

**Authors**: Wei Chen, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00635)  

**Abstract**: Spatio-temporal forecasting is crucial in many domains, such as transportation, meteorology, and energy. However, real-world scenarios frequently present challenges such as signal anomalies, noise, and distributional shifts. Existing solutions primarily enhance robustness by modifying network architectures or training procedures. Nevertheless, these approaches are computationally intensive and resource-demanding, especially for large-scale applications. In this paper, we explore a novel test-time computing paradigm, namely learning with calibration, ST-TTC, for spatio-temporal forecasting. Through learning with calibration, we aim to capture periodic structural biases arising from non-stationarity during the testing phase and perform real-time bias correction on predictions to improve accuracy. Specifically, we first introduce a spectral-domain calibrator with phase-amplitude modulation to mitigate periodic shift and then propose a flash updating mechanism with a streaming memory queue for efficient test-time computation. ST-TTC effectively bypasses complex training-stage techniques, offering an efficient and generalizable paradigm. Extensive experiments on real-world datasets demonstrate the effectiveness, universality, flexibility and efficiency of our proposed method. 

**Abstract (ZH)**: 空间时态预测对于交通、气象和能源等领域至关重要。然而，现实场景中常常存在信号异常、噪声和分布偏移等挑战。现有解决方案主要通过修改网络架构或训练过程来增强鲁棒性，但这在大规模应用场景下计算密集且资源消耗大。本文探索了一种新的测试时计算范式——校准学习，即ST-TTC（时空测试时计算），用于空间时态预测。通过校准学习，我们旨在捕捉测试阶段由非站定性引起的周期结构偏差，并进行实时偏差校正以提高预测准确性。具体而言，我们首先引入了一种谱域校准器，通过相位-幅度调制来减轻周期性移位，然后提出了一种快速更新机制，配备流式内存队列，以实现高效的测试时计算。ST-TTC 有效地绕过了复杂的训练阶段技术，提供了一种高效且通用的范式。在实际数据集上的广泛实验表明，所提出的方法具有有效性、通用性、灵活性和高效性。 

---
# Text-to-CT Generation via 3D Latent Diffusion Model with Contrastive Vision-Language Pretraining 

**Title (ZH)**: 基于对比视觉-语言预训练的3D潜在扩散模型文本到CT生成 

**Authors**: Daniele Molino, Camillo Maria Caruso, Filippo Ruffini, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00633)  

**Abstract**: Objective: While recent advances in text-conditioned generative models have enabled the synthesis of realistic medical images, progress has been largely confined to 2D modalities such as chest X-rays. Extending text-to-image generation to volumetric Computed Tomography (CT) remains a significant challenge, due to its high dimensionality, anatomical complexity, and the absence of robust frameworks that align vision-language data in 3D medical imaging. Methods: We introduce a novel architecture for Text-to-CT generation that combines a latent diffusion model with a 3D contrastive vision-language pretraining scheme. Our approach leverages a dual-encoder CLIP-style model trained on paired CT volumes and radiology reports to establish a shared embedding space, which serves as the conditioning input for generation. CT volumes are compressed into a low-dimensional latent space via a pretrained volumetric VAE, enabling efficient 3D denoising diffusion without requiring external super-resolution stages. Results: We evaluate our method on the CT-RATE dataset and conduct a comprehensive assessment of image fidelity, clinical relevance, and semantic alignment. Our model achieves competitive performance across all tasks, significantly outperforming prior baselines for text-to-CT generation. Moreover, we demonstrate that CT scans synthesized by our framework can effectively augment real data, improving downstream diagnostic performance. Conclusion: Our results show that modality-specific vision-language alignment is a key component for high-quality 3D medical image generation. By integrating contrastive pretraining and volumetric diffusion, our method offers a scalable and controllable solution for synthesizing clinically meaningful CT volumes from text, paving the way for new applications in data augmentation, medical education, and automated clinical simulation. 

**Abstract (ZH)**: 目标: 尽管最近在文本条件生成模型方面的进展使得合成真实医疗图像成为可能，但这些进展主要局限于如胸部X光片这样的2D模态。将文本到图像的生成扩展到体积计算机断层扫描(CT)仍然是一个重大挑战，这主要是由于其高维度、解剖复杂性以及缺少能够对3D医疗成像中的视觉-语言数据进行对齐的稳健框架。方法: 我们提出了一个新的用于文本到CT生成的架构，该架构结合了潜扩散模型和3D对照ive视觉-语言预训练方案。我们的方法利用在配对的CT体积和放射学报告上训练的双编码器CLIP风格模型建立共享嵌入空间，该空间作为生成的条件输入。通过预先训练的体积VAE将CT体积压缩到低维度潜空间，从而实现高效的3D降噪扩散，而无需外部超分辨率阶段。结果: 我们在CT-RATE数据集上评估了我们的方法，并对图像保真度、临床相关性和语义对齐进行了全面评估。我们的模型在所有任务中都表现出竞争力，显著优于现有的文本到CT生成基线。此外，我们证明由我们的框架合成的CT扫描能够有效增强真实数据，提高下游诊断性能。结论: 我们的结果表明，模态特定的视觉-语言对齐是高质量3D医疗图像生成的关键组成部分。通过结合对照ive预训练和体积扩散，我们的方法提供了一种可扩展且可控的解决方案，用于从文本生成具有临床意义的CT体积，为数据增强、医学教育和自动化临床模拟开辟了新的应用途径。 

---
# The Disparate Effects of Partial Information in Bayesian Strategic Learning 

**Title (ZH)**: 部分信息在贝叶斯战略学习中的不同影响 

**Authors**: Srikanth Avasarala, Serena Wang, Juba Ziani  

**Link**: [PDF](https://arxiv.org/pdf/2506.00627)  

**Abstract**: We study how partial information about scoring rules affects fairness in strategic learning settings. In strategic learning, a learner deploys a scoring rule, and agents respond strategically by modifying their features -- at some cost -- to improve their outcomes. However, in our work, agents do not observe the scoring rule directly; instead, they receive a noisy signal of said rule. We consider two different agent models: (i) naive agents, who take the noisy signal at face value, and (ii) Bayesian agents, who update a prior belief based on the signal.
Our goal is to understand how disparities in outcomes arise between groups that differ in their costs of feature modification, and how these disparities vary with the level of transparency of the learner's rule. For naive agents, we show that utility disparities can grow unboundedly with noise, and that the group with lower costs can, perhaps counter-intuitively, be disproportionately harmed under limited transparency. In contrast, for Bayesian agents, disparities remain bounded. We provide a full characterization of disparities across groups as a function of the level of transparency and show that they can vary non-monotonically with noise; in particular, disparities are often minimized at intermediate levels of transparency. Finally, we extend our analysis to settings where groups differ not only in cost, but also in prior beliefs, and study how this asymmetry influences fairness. 

**Abstract (ZH)**: 我们研究部分信息如何影响计分规则在战略学习环境中的公平性。在战略学习中，学习者部署一个计分规则，代理通过修改其特征以改善结果来进行战略响应——但这些修改是有成本的。然而，在我们的研究中，代理不会直接观察到计分规则，而是接收到一个带有噪声的规则信号。我们考虑了两种不同的代理模型：（i）无知代理，他们会认为噪声信号是真实的；（ii）贝叶斯代理，他们会根据信号更新先验信念。

我们的目标是理解在特征修改成本不同的组之间结果差异是如何产生的，以及这些差异如何随着学习者规则透明度的变化而变化。对于无知代理，我们证明了在噪声存在的情况下，效用差异可以无限增长，并且在透明度有限的情况下，成本较低的组可能会出乎意料地受到不成比例的伤害。相反，对于贝叶斯代理，差异保持在有限范围内。我们提供了差异在整个组之间作为透明度函数的完整描述，并展示了它们如何非单调地随噪声变化；特别地，差异往往在透明度的中间水平时最小化。最后，我们将分析扩展到组不仅在成本方面存在差异，还在先验信念方面也存在差异的情境，并研究这种不对称性如何影响公平性。 

---
# Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples 

**Title (ZH)**: 通过组合搜索改进基于上下文示例的对话状态追踪 

**Authors**: Haesung Pyun, Yoonah Park, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00622)  

**Abstract**: In dialogue state tracking (DST), in-context learning comprises a retriever that selects labeled dialogues as in-context examples and a DST model that uses these examples to infer the dialogue state of the query dialogue. Existing methods for constructing training data for retrievers suffer from three key limitations: (1) the synergistic effect of examples is not considered, (2) the linguistic characteristics of the query are not sufficiently factored in, and (3) scoring is not directly optimized for DST performance. Consequently, the retriever can fail to retrieve examples that would substantially improve DST performance. To address these issues, we present CombiSearch, a method that scores effective in-context examples based on their combinatorial impact on DST performance. Our evaluation on MultiWOZ shows that retrievers trained with CombiSearch surpass state-of-the-art models, achieving a 20x gain in data efficiency and generalizing well to the SGD dataset. Moreover, CombiSearch attains a 12% absolute improvement in the upper bound DST performance over traditional approaches when no retrieval errors are assumed. This significantly increases the headroom for practical DST performance while demonstrating that existing methods rely on suboptimal data for retriever training. 

**Abstract (ZH)**: 基于组合影响的对话状态跟踪中的检索学习方法 

---
# A Topological Semantics of Dialogue: Nerve Structures and Logical Extraction 

**Title (ZH)**: 拓扑对话语义：神经结构与逻辑提取 

**Authors**: Andreu Ballus Santacana  

**Link**: [PDF](https://arxiv.org/pdf/2506.00615)  

**Abstract**: We introduce a concise, topologically-motivated semantics for finite dialogues by mapping each utterance to an open set in a fixed semantic space, building the corresponding nerve complex of joint satisfiability, and extracting fundamental combinatorial invariants:
1. The negative nerve, which enumerates all finite collections of utterances whose
opens have empty intersection, providing a straightforward criterion for merging
separate transcripts without contradiction.
2. The global interpretation subspace, the unique minimal open in which all asserted
utterances hold simultaneously, enabling effective enumeration of all logical
consequences of the entire dialogue.
3. A practical demonstration in the Wolfram Language, with algorithms for constructing
nerves, detecting inconsistencies, and computing the global interpretation, thereby
illustrating computational feasibility.
Our framework is grounded in classical duality and topological semantics (Stone duality, Priestley duality, Tarski's semantics, coherence-space methods, Scott domains, topos semantics, and homotopy type theory) while drawing on recent advances in topological data analysis and dialogue-based semantics. 

**Abstract (ZH)**: 我们引入了一种简洁的、拓扑驱动的有限对话语义，将每个表述映射到固定语义空间中的一个开集，构建相应的联合可满足性神经复杂体，并提取基本组合不变量：
1. 负神经，枚举所有具有空交集的有限表述集，提供合并不矛盾的独立会话的直接标准。
2. 全局解释子空间，所有断言的表述同时成立的唯一最小开集，使得可以有效枚举整个对话的所有逻辑推论。
3. 用Wolfram语言进行实用演示，包括构建神经网络、检测不一致性和计算全局解释的算法，从而说明其实现可行性。
我们的框架基于经典对偶性和拓扑语义（Stone对偶性、Priestley对偶性、塔尔斯基语义、调和空间方法、Scott域、范畴语义和同调类型理论），并借鉴了拓扑数据分析和基于对话的语义领域的最新进展。 

---
# Predictability-Aware Compression and Decompression Framework for Multichannel Time Series Data 

**Title (ZH)**: 面向可预测性的多通道时间序列数据压缩与解压缩框架 

**Authors**: Ziqi Liu, Pei Zeng, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.00614)  

**Abstract**: Real-world multichannel time series prediction faces growing demands for efficiency across edge and cloud environments, making channel compression a timely and essential problem. Motivated by success of Multiple-Input Multiple-Output (MIMO) methods, we propose a predictability-aware compression-decompression framework to reduce runtime, lower communication cost, and maintain prediction accuracy across diverse predictors. The core idea involves using a circular periodicity key matrix with orthogonality to capture underlying time series predictability during compression and to mitigate reconstruction errors during decompression by relaxing oversimplified data assumptions. Theoretical and empirical analyses show that the proposed framework is both time-efficient and scalable under a large number of channels. Extensive experiments on six datasets across various predictors demonstrate that the proposed method achieves superior overall performance by jointly considering prediction accuracy and runtime, while maintaining strong compatibility with diverse predictors. 

**Abstract (ZH)**: 实时光纤多通道时间序列预测在边缘和云环境中的需求 growing, 促使通道压缩成为一项及时且必要的问题。受 Multiple-Input Multiple-Output (MIMO) 方法成功的启发, 我们提出一种预测性意识下的压缩-解压缩框架, 以降低运行时开销、减少通信成本并保持预测准确性, 并适用于多种预测器。该框架的核心思想是在压缩过程中使用带有正交性的循环周期性键矩阵来捕捉潜在的时间序列预测性, 并在解压缩过程中通过放松对简单数据假设的过简化来减轻重建误差。理论分析和实证研究表明, 所提出的框架在大量通道的情况下具有时间高效性和扩展性。在六个不同数据集上的广泛实验表明, 所提出的方法通过同时考虑预测准确性和运行时开销, 达到了优异的整体性能, 同时与多种预测器保持了良好的兼容性。 

---
# Evaluating Robot Policies in a World Model 

**Title (ZH)**: 在世界模型中评估机器人策略 

**Authors**: Julian Quevedo, Percy Liang, Sherry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00613)  

**Abstract**: Robotics has broad applications from automating house chores to taking care of patients. However, evaluating robot control policies is challenging, as real-world testing is expensive, while handcrafted simulations often fail to accurately reflect real-world conditions, resulting in poor correlation between simulated evaluation and real-world outcomes. In this work, we investigate World-model-based Policy Evaluation (WPE). We first train an action-conditioned video generation model as a proxy to real-world environments. To enable efficient rollouts of hundreds of interactive steps while mitigating error accumulation in the world model, we propose an inference scheme which we call Blockwise-Autoregressive Diffusion Transformer with adjustable context and decoding horizon lengths. To ensure that the world model indeed follows action input, we propose metrics based on the agreement between the ground truth video and generated video conditioned on the same sequence of actions to evaluate the world model. We then use the world model for policy evaluation by performing Monte Carlo rollouts in the world model while employing a vision-language model (VLM) as a reward function. Interestingly, we found that WPE tends to underestimate the policy values for in-distribution actions and overestimate policy values for out-of-distribution actions. Nevertheless, WPE preserves the relative rankings of different policies. In emulating real robot executions, WPE achieves high fidelity in mimicing robot arm movements as in real videos, while emulating highly realistic object interaction remains challenging. Despite this limitation, we show that a world model can serve as a starting point for evaluating robot policies before real-world deployment. 

**Abstract (ZH)**: 基于世界模型的策略评估：机器人控制策略评估的新方法 

---
# Parallel Rescaling: Rebalancing Consistency Guidance for Personalized Diffusion Models 

**Title (ZH)**: 并行缩放：个人化扩散模型的一致性指导重平衡方法 

**Authors**: JungWoo Chae, Jiyoon Kim, Sangheum Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00607)  

**Abstract**: Personalizing diffusion models to specific users or concepts remains challenging, particularly when only a few reference images are available. Existing methods such as DreamBooth and Textual Inversion often overfit to limited data, causing misalignment between generated images and text prompts when attempting to balance identity fidelity with prompt adherence. While Direct Consistency Optimization (DCO) with its consistency-guided sampling partially alleviates this issue, it still struggles with complex or stylized prompts. In this paper, we propose a parallel rescaling technique for personalized diffusion models. Our approach explicitly decomposes the consistency guidance signal into parallel and orthogonal components relative to classifier free guidance (CFG). By rescaling the parallel component, we minimize disruptive interference with CFG while preserving the subject's identity. Unlike prior personalization methods, our technique does not require additional training data or expensive annotations. Extensive experiments show improved prompt alignment and visual fidelity compared to baseline methods, even on challenging stylized prompts. These findings highlight the potential of parallel rescaled guidance to yield more stable and accurate personalization for diverse user inputs. 

**Abstract (ZH)**: 个性化扩散模型面向特定用户或概念 remains 挑战性，特别是在仅有少量参考图像的情况下。现有的方法如 DreamBooth 和 Textual Inversion 经常会对有限的数据过拟合，在尝试平衡身份保真度与提示一致性时导致生成图像与文本提示之间的对齐偏差。虽然直接一致性优化 (DCO) 借助一致性引导采样部分缓解了这一问题，但仍然难以应对复杂的或风格化的提示。在本文中，我们提出了一种并行缩放技术用于个性化扩散模型。我们的方法显式地将一致性引导信号分解为相对于 classifier-free guidance (CFG) 平行和正交成分。通过缩放平行成分，我们最小化了对 CFG 的干扰同时保持主题的身份。与先前的个性化方法不同，我们的技术不需要额外的训练数据或昂贵的标注。广泛实验表明，在即使是复杂风格化提示的情况下，我们的方法也优于基线方法，实现了更好的提示对齐和视觉保真度。这些发现突显了并行缩放引导在为多样化用户输入提供更稳定和准确的个性化方面的潜力。 

---
# Graph Evidential Learning for Anomaly Detection 

**Title (ZH)**: 图证据学习异常检测 

**Authors**: Chunyu Wei, Wenji Hu, Xingjia Hao, Yunhai Wang, Yueguo Chen, Bing Bai, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00594)  

**Abstract**: Graph anomaly detection faces significant challenges due to the scarcity of reliable anomaly-labeled datasets, driving the development of unsupervised methods. Graph autoencoders (GAEs) have emerged as a dominant approach by reconstructing graph structures and node features while deriving anomaly scores from reconstruction errors. However, relying solely on reconstruction error for anomaly detection has limitations, as it increases the sensitivity to noise and overfitting. To address these issues, we propose Graph Evidential Learning (GEL), a probabilistic framework that redefines the reconstruction process through evidential learning. By modeling node features and graph topology using evidential distributions, GEL quantifies two types of uncertainty: graph uncertainty and reconstruction uncertainty, incorporating them into the anomaly scoring mechanism. Extensive experiments demonstrate that GEL achieves state-of-the-art performance while maintaining high robustness against noise and structural perturbations. 

**Abstract (ZH)**: 图异常检测面临着由于可靠的异常标记数据集稀缺而带来的显著挑战，推动了无监督方法的发展。图自编码器（GAEs）通过重构图结构和节点特征，并从重构错误中衍生异常评分，成为主导方法。然而，仅依赖重构误差进行异常检测存在局限性，因为它增加了对噪声和过拟合的敏感性。为解决这些问题，我们提出了一种概率框架——图证据学习（GEL），该框架通过证据学习重新定义重构过程。通过使用证据分布来建模节点特征和图拓扑，GEL量化了两类不确定性：图不确定性与重构不确定性，并将其融入异常评分机制中。大量实验表明，GEL在保持对噪声和结构扰动的高鲁棒性的同时，实现了最先进的性能。 

---
# Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn 

**Title (ZH)**: 通过减少 churn 遏制持续强化学习中塑性损失 

**Authors**: Hongyao Tang, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2506.00592)  

**Abstract**: Plasticity, or the ability of an agent to adapt to new tasks, environments, or distributions, is crucial for continual learning. In this paper, we study the loss of plasticity in deep continual RL from the lens of churn: network output variability for out-of-batch data induced by mini-batch training. We demonstrate that (1) the loss of plasticity is accompanied by the exacerbation of churn due to the gradual rank decrease of the Neural Tangent Kernel (NTK) matrix; (2) reducing churn helps prevent rank collapse and adjusts the step size of regular RL gradients adaptively. Moreover, we introduce Continual Churn Approximated Reduction (C-CHAIN) and demonstrate it improves learning performance and outperforms baselines in a diverse range of continual learning environments on OpenAI Gym Control, ProcGen, DeepMind Control Suite, and MinAtar benchmarks. 

**Abstract (ZH)**: 连续学习中由于 minibatch 训练引起的 churn 加剧导致的可塑性丧失及其缓解方法 

---
# Temporal Chunking Enhances Recognition of Implicit Sequential Patterns 

**Title (ZH)**: 时序分割增强隐式序列模式识别 

**Authors**: Jayanta Dey, Nicholas Soures, Miranda Gonzales, Itamar Lerner, Christopher Kanan, Dhireesha Kudithipudi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00588)  

**Abstract**: In this pilot study, we propose a neuro-inspired approach that compresses temporal sequences into context-tagged chunks, where each tag represents a recurring structural unit or``community'' in the sequence. These tags are generated during an offline sleep phase and serve as compact references to past experience, allowing the learner to incorporate information beyond its immediate input range. We evaluate this idea in a controlled synthetic environment designed to reveal the limitations of traditional neural network based sequence learners, such as recurrent neural networks (RNNs), when facing temporal patterns on multiple timescales. We evaluate this idea in a controlled synthetic environment designed to reveal the limitations of traditional neural network based sequence learners, such as recurrent neural networks (RNNs), when facing temporal patterns on multiple timescales. Our results, while preliminary, suggest that temporal chunking can significantly enhance learning efficiency under resource constrained settings. A small-scale human pilot study using a Serial Reaction Time Task further motivates the idea of structural abstraction. Although limited to synthetic tasks, this work serves as an early proof-of-concept, with initial evidence that learned context tags can transfer across related task, offering potential for future applications in transfer learning. 

**Abstract (ZH)**: 本研究试点提出了一种受神经启发的方法，将时间序列压缩为带上下文标签的片段，每个标签代表序列中的一个 recurring 结构单元或“社区”。这些标签在离线睡眠阶段生成，作为对过去经验的紧凑引用，使学习者能够整合超出其即时输入范围的信息。我们在一个旨在揭示传统基于神经网络的时间序列学习器（如循环神经网络RNN）在多时间尺度时间模式面前局限性的受控合成环境中评估这一想法。初始结果表明，在资源受限的情况下，时间片段化可以显著提高学习效率。小型人类试点研究通过使用序列反应时间任务进一步证实了结构抽象的概念。尽管仅限于合成任务，但本研究作为早期概念证明，提供了初步证据，表明学习到的上下文标签可以在相关任务之间迁移，为未来在迁移学习中的应用提供了潜力。 

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
# Understanding Behavioral Metric Learning: A Large-Scale Study on Distracting Reinforcement Learning Environments 

**Title (ZH)**: 理解行为度量学习：对具有分散注意力的强化学习环境的大规模研究 

**Authors**: Ziyan Luo, Tianwei Ni, Pierre-Luc Bacon, Doina Precup, Xujie Si  

**Link**: [PDF](https://arxiv.org/pdf/2506.00563)  

**Abstract**: A key approach to state abstraction is approximating behavioral metrics (notably, bisimulation metrics) in the observation space and embedding these learned distances in the representation space. While promising for robustness to task-irrelevant noise, as shown in prior work, accurately estimating these metrics remains challenging, requiring various design choices that create gaps between theory and practice. Prior evaluations focus mainly on final returns, leaving the quality of learned metrics and the source of performance gains unclear. To systematically assess how metric learning works in deep reinforcement learning (RL), we evaluate five recent approaches, unified conceptually as isometric embeddings with varying design choices. We benchmark them with baselines across 20 state-based and 14 pixel-based tasks, spanning 370 task configurations with diverse noise settings. Beyond final returns, we introduce the evaluation of a denoising factor to quantify the encoder's ability to filter distractions. To further isolate the effect of metric learning, we propose and evaluate an isolated metric estimation setting, in which the encoder is influenced solely by the metric loss. Finally, we release an open-source, modular codebase to improve reproducibility and support future research on metric learning in deep RL. 

**Abstract (ZH)**: 一种关键的态抽象方法是通过观测空间近似行为度量（尤其是拟态度量），并在表示空间中嵌入这些学习到的距离。尽管这种方法对无关任务噪声具有鲁棒性，前人研究已经证明，准确估计这些度量仍然具有挑战性，需要各种设计选择以弥合理论与实践之间的差距。先前的评估主要集中在最终回报上，使得所学习度量的质量和性能提升的原因不清晰。为了系统地评估度量学习在深度强化学习（RL）中的作用，我们评估了五种最近的方法，这些方法从概念上统一为具有不同设计选择的等距嵌入。我们在20个基于态和14个基于像素的任务上与基线进行基准测试，涵盖了370种具有不同噪声设置的任务配置。除了最终回报，我们引入了去噪因子的评估来定量衡量编码器过滤干扰的能力。为了进一步隔离度量学习的影响，我们提出并评估了一种孤立的度量估计设置，在这种设置中，编码器仅受到度量损失的影响。最后，我们发布了一个开源模块化代码库，以提高可重复性，并支持未来在深度RL中进行度量学习的研究。 

---
# MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning 

**Title (ZH)**: MMedAgent-RL: 优化多模态医疗推理中的多Agent协作 

**Authors**: Peng Xia, Jinglu Wang, Yibo Peng, Kaide Zeng, Xian Wu, Xiangru Tang, Hongtu Zhu, Yun Li, Shujie Liu, Yan Lu, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00555)  

**Abstract**: Medical Large Vision-Language Models (Med-LVLMs) have shown strong potential in multimodal diagnostic tasks. However, existing single-agent models struggle to generalize across diverse medical specialties, limiting their performance. Recent efforts introduce multi-agent collaboration frameworks inspired by clinical workflows, where general practitioners (GPs) and specialists interact in a fixed sequence. Despite improvements, these static pipelines lack flexibility and adaptability in reasoning. To address this, we propose MMedAgent-RL, a reinforcement learning (RL)-based multi-agent framework that enables dynamic, optimized collaboration among medical agents. Specifically, we train two GP agents based on Qwen2.5-VL via RL: the triage doctor learns to assign patients to appropriate specialties, while the attending physician integrates the judgments from multi-specialists and its own knowledge to make final decisions. To address the inconsistency in specialist outputs, we introduce a curriculum learning (CL)-guided RL strategy that progressively teaches the attending physician to balance between imitating specialists and correcting their mistakes. Experiments on five medical VQA benchmarks demonstrate that MMedAgent-RL not only outperforms both open-source and proprietary Med-LVLMs, but also exhibits human-like reasoning patterns. Notably, it achieves an average performance gain of 18.4% over supervised fine-tuning baselines. 

**Abstract (ZH)**: 基于强化学习的多Agent动态协作医疗模型（MMedAgent-RL）：在多模态诊断任务中的应用 

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
# Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Self-Attention-based Deep Learning Approach 

**Title (ZH)**: 基于自注意力的深度学习方法在平滑追求眼球运动中的缺省数据插补 

**Authors**: Mehdi Bejani, Guillermo Perez-de-Arenaza-Pozo, Julián D. Arias-Londoño, Juan I. Godino-LLorente  

**Link**: [PDF](https://arxiv.org/pdf/2506.00545)  

**Abstract**: Missing data is a relevant issue in time series, especially in biomedical sequences such as those corresponding to smooth pursuit eye movements, which often contain gaps due to eye blinks and track losses, complicating the analysis and extraction of meaningful biomarkers. In this paper, a novel imputation framework is proposed using Self-Attention-based Imputation networks for time series, which leverages the power of deep learning and self-attention mechanisms to impute missing data. We further refine the imputed data using a custom made autoencoder, tailored to represent smooth pursuit eye movement sequences. The proposed approach was implemented using 5,504 sequences from 172 Parkinsonian patients and healthy controls. Results show a significant improvement in the accuracy of reconstructed eye movement sequences with respect to other state of the art techniques, substantially reducing the values for common time domain error metrics such as the mean absolute error, mean relative error, and root mean square error, while also preserving the signal's frequency domain characteristics. Moreover, it demonstrates robustness when large intervals of data are missing. This method offers an alternative solution for robustly handling missing data in time series, enhancing the reliability of smooth pursuit analysis for the screening and monitoring of neurodegenerative disorders. 

**Abstract (ZH)**: 时间序列中缺失数据是一个相关的问题，尤其是在平滑追寻眼动等生物医学序列中，这些序列常常由于眨眼和跟踪丢失而包含缺口，这使得分析和提取有意义的生物标记变得复杂。本文提出了一种新的插补框架，使用基于自注意力的插补网络进行时间序列插补，该框架利用了深度学习和自注意力机制来插补缺失数据。我们进一步使用一个根据平滑追寻眼动序列定制的自编码器对插补后的数据进行了细化。该方法使用了来自172名帕金森病患者和健康对照者的5,504个序列进行了实现。结果表明，与现有的先进技术相比，在重建眼动序列的准确性方面有显著提高，大幅降低了常见的时域误差指标（如绝对误差均值、相对误差均值和均方根误差）的值，同时也保留了信号的频域特征。此外，该方法在大量数据缺失的情况下也表现出鲁棒性。该方法为时间序列中稳健地处理缺失数据提供了一种替代方案，增强了平滑追寻分析在神经退行性疾病筛查和监测中的可靠性。 

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
# Pro3D-Editor : A Progressive-Views Perspective for Consistent and Precise 3D Editing 

**Title (ZH)**: Pro3D-Editor : 一种渐进视点视角下的一致精准3D编辑 

**Authors**: Yang Zheng, Mengqi Huang, Nan Chen, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00512)  

**Abstract**: Text-guided 3D editing aims to precisely edit semantically relevant local 3D regions, which has significant potential for various practical applications ranging from 3D games to film production. Existing methods typically follow a view-indiscriminate paradigm: editing 2D views indiscriminately and projecting them back into 3D space. However, they overlook the different cross-view interdependencies, resulting in inconsistent multi-view editing. In this study, we argue that ideal consistent 3D editing can be achieved through a \textit{progressive-views paradigm}, which propagates editing semantics from the editing-salient view to other editing-sparse views. Specifically, we propose \textit{Pro3D-Editor}, a novel framework, which mainly includes Primary-view Sampler, Key-view Render, and Full-view Refiner. Primary-view Sampler dynamically samples and edits the most editing-salient view as the primary view. Key-view Render accurately propagates editing semantics from the primary view to other key views through its Mixture-of-View-Experts Low-Rank Adaption (MoVE-LoRA). Full-view Refiner edits and refines the 3D object based on the edited multi-views. Extensive experiments demonstrate that our method outperforms existing methods in editing accuracy and spatial consistency. 

**Abstract (ZH)**: 基于文本引导的渐进视角3D编辑 

---
# Multi-Objective Neural Network Assisted Design Optimization of Soft Fin-Ray Grippers for Enhanced Grasping Performance 

**Title (ZH)**: 软鳍肋夹持器多目标神经网络辅助设计优化以增强抓取性能 

**Authors**: Ali Ghanizadeh, Ali Ahmadi, Arash Bahrami  

**Link**: [PDF](https://arxiv.org/pdf/2506.00494)  

**Abstract**: Soft Fin-Ray grippers can perform delicate and careful manipulation, which has caused notable attention in different fields. These grippers can handle objects of various forms and sizes safely. The internal structure of the Fin-Ray finger plays a significant role in its adaptability and grasping performance. However, modeling the non-linear grasp force and deformation behaviors for design purposes is challenging. Moreover, when the Fin-Ray finger becomes more rigid and capable of exerting higher forces, it becomes less delicate in handling objects. The contrast between these two objectives gives rise to a multi-objective optimization problem. In this study, we employ finite element method (FEM) to estimate the deflections and contact forces of the Fin-Ray, grasping cylindrical objects. This dataset is then used to construct a multilayer perception (MLP) for prediction of the contact force and the tip displacement. The FEM dataset consists of three input and four target features. The three input features of the MLP and optimization design variables are the thickness of the front and supporting beams, the thickness of the cross beams, and the equal spacing between the cross beams. In addition, the target features are the maximum contact forces and maximum tip displacements in x- and y-directions. The magnitude of maximum contact force and magnitude of maximum tip displacement are the two objectives, showing the trade-off between force and delicate manipulation in soft Fin-Ray grippers. Furthermore, the optimized set of solutions are found using multi-objective optimal techniques. We use non-dominated sorting genetic algorithm (NSGA-II) method for this purpose. Our findings demonstrate that our methodologies can be used to improve the design and gripping performance of soft robotic grippers, helping us to choose a design not only for delicate grasping but also for high-force applications. 

**Abstract (ZH)**: Soft Fin-Ray 夹持器可以进行精细和谨慎的操作，已在不同领域引起广泛关注。这些夹持器能安全地处理各种形状和大小的物体。Fin-Ray 指夹的内部结构在它的适应性和夹持性能中起着重要作用。然而，为了设计目的，对非线性夹持力和变形行为建模具有挑战性。此外，当 Fin-Ray 指夹变得更刚硬并能够施加更大的力时，它在操作物体时会变得不够细致。这两种目标之间的对比产生了多目标优化问题。在本研究中，我们采用有限元方法（FEM）估计 Fin-Ray 对圆柱形物体进行夹持时的位移和接触力，然后利用此数据集构建多层感知器（MLP）以预测接触力和末端位移。FEM 数据集包括三个输入特征和四个目标特征。MLP 和优化设计变量的三个输入特征为前梁和支撑梁的厚度、横梁的厚度以及横梁之间的等间距。此外，目标特征为 x- 和 y-方向的最大接触力和最大末端位移。最大接触力的大小和最大末端位移的大小是两个目标，表明软 Fin-Ray 夹持器中力量与精细操作之间的权衡。进一步使用多目标优化技术找到优化解集。我们使用非支配排序遗传算法（NSGA-II）方法进行此目的。我们的研究结果表明，我们的方法可用于改进软柔体夹持器的设计和夹持性能，帮助我们在精细夹持和高力应用之间做出设计选择。 

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
# PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings 

**Title (ZH)**: PVP：一个包含说服策略、观众特征和说服力评分的个性化视觉说服图像数据集 

**Authors**: Junseo Kim, Jongwook Han, Dongmin Choi, Jongwook Yoon, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00481)  

**Abstract**: Visual persuasion, which uses visual elements to influence cognition and behaviors, is crucial in fields such as advertising and political communication. With recent advancements in artificial intelligence, there is growing potential to develop persuasive systems that automatically generate persuasive images tailored to individuals. However, a significant bottleneck in this area is the lack of comprehensive datasets that connect the persuasiveness of images with the personal information about those who evaluated the images. To address this gap and facilitate technological advancements in personalized visual persuasion, we release the Personalized Visual Persuasion (PVP) dataset, comprising 28,454 persuasive images across 596 messages and 9 persuasion strategies. Importantly, the PVP dataset provides persuasiveness scores of images evaluated by 2,521 human annotators, along with their demographic and psychological characteristics (personality traits and values). We demonstrate the utility of our dataset by developing a persuasive image generator and an automated evaluator, and establish benchmark baselines. Our experiments reveal that incorporating psychological characteristics enhances the generation and evaluation of persuasive images, providing valuable insights for personalized visual persuasion. 

**Abstract (ZH)**: 个性化视觉说服（PVP）数据集：包含28,454张说服性图像及其评估者的心理和人口统计特征 

---
# SST: Self-training with Self-adaptive Thresholding for Semi-supervised Learning 

**Title (ZH)**: SST：自适应阈值自我训练用于半监督学习 

**Authors**: Shuai Zhao, Heyan Huang, Xinge Li, Xiaokang Chen, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00467)  

**Abstract**: Neural networks have demonstrated exceptional performance in supervised learning, benefiting from abundant high-quality annotated data. However, obtaining such data in real-world scenarios is costly and labor-intensive. Semi-supervised learning (SSL) offers a solution to this problem. Recent studies, such as Semi-ViT and Noisy Student, which employ consistency regularization or pseudo-labeling, have demonstrated significant achievements. However, they still face challenges, particularly in accurately selecting sufficient high-quality pseudo-labels due to their reliance on fixed thresholds. Recent methods such as FlexMatch and FreeMatch have introduced flexible or self-adaptive thresholding techniques, greatly advancing SSL research. Nonetheless, their process of updating thresholds at each iteration is deemed time-consuming, computationally intensive, and potentially unnecessary. To address these issues, we propose Self-training with Self-adaptive Thresholding (SST), a novel, effective, and efficient SSL framework. SST introduces an innovative Self-Adaptive Thresholding (SAT) mechanism that adaptively adjusts class-specific thresholds based on the model's learning progress. SAT ensures the selection of high-quality pseudo-labeled data, mitigating the risks of inaccurate pseudo-labels and confirmation bias. Extensive experiments demonstrate that SST achieves state-of-the-art performance with remarkable efficiency, generalization, and scalability across various architectures and datasets. Semi-SST-ViT-Huge achieves the best results on competitive ImageNet-1K SSL benchmarks, with 80.7% / 84.9% Top-1 accuracy using only 1% / 10% labeled data. Compared to the fully-supervised DeiT-III-ViT-Huge, which achieves 84.8% Top-1 accuracy using 100% labeled data, our method demonstrates superior performance using only 10% labeled data. 

**Abstract (ZH)**: 基于自适应阈值的自我训练（SST）：一种高效有效的半监督学习框架 

---
# XMAD-Bench: Cross-Domain Multilingual Audio Deepfake Benchmark 

**Title (ZH)**: XMAD-Bench: 跨域多语言音频换脸 benchmark 

**Authors**: Ioan-Paul Ciobanu, Andrei-Iulian Hiji, Nicolae-Catalin Ristea, Paul Irofti, Cristian Rusu, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00462)  

**Abstract**: Recent advances in audio generation led to an increasing number of deepfakes, making the general public more vulnerable to financial scams, identity theft, and misinformation. Audio deepfake detectors promise to alleviate this issue, with many recent studies reporting accuracy rates close to 99%. However, these methods are typically tested in an in-domain setup, where the deepfake samples from the training and test sets are produced by the same generative models. To this end, we introduce XMAD-Bench, a large-scale cross-domain multilingual audio deepfake benchmark comprising 668.8 hours of real and deepfake speech. In our novel dataset, the speakers, the generative methods, and the real audio sources are distinct across training and test splits. This leads to a challenging cross-domain evaluation setup, where audio deepfake detectors can be tested ``in the wild''. Our in-domain and cross-domain experiments indicate a clear disparity between the in-domain performance of deepfake detectors, which is usually as high as 100%, and the cross-domain performance of the same models, which is sometimes similar to random chance. Our benchmark highlights the need for the development of robust audio deepfake detectors, which maintain their generalization capacity across different languages, speakers, generative methods, and data sources. Our benchmark is publicly released at this https URL. 

**Abstract (ZH)**: 近期音频生成技术的发展导致了音频合成样本的增多，使得普通公众更容易成为金融诈骗、身份盗用和虚假信息的受害者。音频合成检测器有望缓解这一问题，多项最近的研究报道其准确率接近99%。然而，这些方法通常在同域设置下进行测试，即训练集和测试集中的合成样本由相同的生成模型生成。为此，我们提出了一种大规模跨域多语言音频合成检测基准XMAD-Bench，该基准包含668.8小时的真实和合成语音。在我们的新数据集中，训练集和测试集中的说话人、生成方法和真实音频源均不相同。这导致了一种具有挑战性的跨域评估设置，使得音频合成检测器能在真实环境中进行测试。我们的同域和跨域实验表明，合成检测器在同域的性能通常高达100%，而在跨域下的性能有时甚至类似于随机猜测。该基准突显了开发能够在不同语言、说话人、生成方法和数据源下保持泛化能力的鲁棒音频合成检测器的需求。该基准已经在以下链接公开发布：this https URL。 

---
# Comparing Traditional and Reinforcement-Learning Methods for Energy Storage Control 

**Title (ZH)**: 传统方法与强化学习方法在储能控制中的比较 

**Authors**: Elinor Ginzburg, Itay Segev, Yoash Levron, Sarah Keren  

**Link**: [PDF](https://arxiv.org/pdf/2506.00459)  

**Abstract**: We aim to better understand the tradeoffs between traditional and reinforcement learning (RL) approaches for energy storage management. More specifically, we wish to better understand the performance loss incurred when using a generative RL policy instead of using a traditional approach to find optimal control policies for specific instances. Our comparison is based on a simplified micro-grid model, that includes a load component, a photovoltaic source, and a storage device. Based on this model, we examine three use cases of increasing complexity: ideal storage with convex cost functions, lossy storage devices, and lossy storage devices with convex transmission losses. With the aim of promoting the principled use RL based methods in this challenging and important domain, we provide a detailed formulation of each use case and a detailed description of the optimization challenges. We then compare the performance of traditional and RL methods, discuss settings in which it is beneficial to use each method, and suggest avenues for future investigation. 

**Abstract (ZH)**: 我们旨在更好地理解传统方法与强化学习（RL）方法在储能管理中的权衡。具体而言，我们希望更好地理解当使用生成性RL策略而非传统方法寻找特定实例的最佳控制策略时所付出的性能损失。我们的比较基于一个简化的微网模型，该模型包括负载组件、光伏源和储能设备。基于此模型，我们探讨了三种逐步复杂的使用案例：理想的具有凸成本函数的储能、有损耗的储能设备，以及具有凸传输损耗的有损耗储能设备。为了促进在这一具有挑战性和重要性的领域中合理使用基于RL的方法，我们提供了每个使用案例的详细建模和优化挑战的详细描述。然后，我们比较了传统方法和RL方法的性能，讨论了各自使用有益的设置，并提出了未来研究的方向。 

---
# Reinforcement Learning for Hanabi 

**Title (ZH)**: 汉诺伊纸牌游戏的强化学习方法 

**Authors**: Nina Cohen, Kordel K. France  

**Link**: [PDF](https://arxiv.org/pdf/2506.00458)  

**Abstract**: Hanabi has become a popular game for research when it comes to reinforcement learning (RL) as it is one of the few cooperative card games where you have incomplete knowledge of the entire environment, thus presenting a challenge for a RL agent. We explored different tabular and deep reinforcement learning algorithms to see which had the best performance both against an agent of the same type and also against other types of agents. We establish that certain agents played their highest scoring games against specific agents while others exhibited higher scores on average by adapting to the opposing agent's behavior. We attempted to quantify the conditions under which each algorithm provides the best advantage and identified the most interesting interactions between agents of different types. In the end, we found that temporal difference (TD) algorithms had better overall performance and balancing of play types compared to tabular agents. Specifically, tabular Expected SARSA and deep Q-Learning agents showed the best performance. 

**Abstract (ZH)**: 汉诺伊纸牌游戏已成为强化学习研究中的一个流行研究对象，因为它是一个少数具有整个环境部分信息的协作纸牌游戏，从而为强化学习代理带来挑战。我们探索了不同的表式和深度强化学习算法，以确定哪种算法在同类型代理和不同类型代理之间都能表现出最佳性能。我们发现某些代理在与特定类型的代理对弈时表现最佳，而其他代理则通过适应对手代理的行为实现了更高的平均评分。我们试图量化每种算法提供最佳优势的条件，并确定不同类型代理之间最有趣的交互。最终，我们发现时差（Temporal Difference，TD）算法在整体性能和不同类型代理间的均衡方面优于表式代理。具体来说，表式Expected SARSA代理和深度Q学习代理表现出最佳性能。 

---
# Diffusion Models for Increasing Accuracy in Olfaction Sensors and Datasets 

**Title (ZH)**: 扩散模型在提高气味传感器和数据集准确性中的应用 

**Authors**: Kordel K. France, Ovidiu Daescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00455)  

**Abstract**: Robotic odour source localization (OSL) is a critical capability for autonomous systems operating in complex environments. However, current OSL methods often suffer from ambiguities, particularly when robots misattribute odours to incorrect objects due to limitations in olfactory datasets and sensor resolutions. To address this challenge, we introduce a novel machine learning method using diffusion-based molecular generation to enhance odour localization accuracy that can be used by itself or with automated olfactory dataset construction pipelines with vision-language models (VLMs) This generative process of our diffusion model expands the chemical space beyond the limitations of both current olfactory datasets and the training data of VLMs, enabling the identification of potential odourant molecules not previously documented. The generated molecules can then be more accurately validated using advanced olfactory sensors which emulate human olfactory recognition through electronic sensor arrays. By integrating visual analysis, language processing, and molecular generation, our framework enhances the ability of olfaction-vision models on robots to accurately associate odours with their correct sources, thereby improving navigation and decision-making in environments where olfactory cues are essential. Our methodology represents a foundational advancement in the field of robotic olfaction, offering a scalable solution to the challenges posed by limited olfactory data and sensor ambiguities. 

**Abstract (ZH)**: 基于扩散过程分子生成的机器人气味来源定位方法 

---
# TMetaNet: Topological Meta-Learning Framework for Dynamic Link Prediction 

**Title (ZH)**: TMetaNet: 杆度拓扑元学习框架用于动态链路预测 

**Authors**: Hao Li, Hao Wan, Yuzhou Chen, Dongsheng Ye, Yulia Gel, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00453)  

**Abstract**: Dynamic graphs evolve continuously, presenting challenges for traditional graph learning due to their changing structures and temporal dependencies. Recent advancements have shown potential in addressing these challenges by developing suitable meta-learning-based dynamic graph neural network models. However, most meta-learning approaches for dynamic graphs rely on fixed weight update parameters, neglecting the essential intrinsic complex high-order topological information of dynamically evolving graphs. We have designed Dowker Zigzag Persistence (DZP), an efficient and stable dynamic graph persistent homology representation method based on Dowker complex and zigzag persistence, to capture the high-order features of dynamic graphs. Armed with the DZP ideas, we propose TMetaNet, a new meta-learning parameter update model based on dynamic topological features. By utilizing the distances between high-order topological features, TMetaNet enables more effective adaptation across snapshots. Experiments on real-world datasets demonstrate TMetaNet's state-of-the-art performance and resilience to graph noise, illustrating its high potential for meta-learning and dynamic graph analysis. Our code is available at this https URL. 

**Abstract (ZH)**: 动态图持续演变，给传统的图学习带来了挑战，因为它们的结构和时序依赖性不断变化。近期的研究表明，通过开发适合的基于元学习的动态图神经网络模型，有可能解决这些挑战。然而，大多数针对动态图的元学习方法依赖于固定权重更新参数，忽略了动态演变图中本质的复杂的高阶拓扑信息。我们设计了基于Dowker复形和zigzag持久性的Dowker Zigzag Persistence（DZP）方法，一种高效的动态图持久同调表示方法，用于捕捉动态图的高阶特征。利用DZP的理念，我们提出了TMetaNet，一种基于动态拓扑特征的元学习参数更新模型。通过利用高阶拓扑特征之间的距离，TMetaNet能够实现更有效的跨快照适配。实验证明，TMetaNet在实际数据集上的性能处于领先地位，并且对图噪声具有很高的鲁棒性，展示了其在元学习和动态图分析中的高潜力。代码已发布在以下链接：this https URL。 

---
# Attention-Aided MMSE for OFDM Channel Estimation: Learning Linear Filters with Attention 

**Title (ZH)**: 基于注意力辅助的MMSE OFDM信道估计：学习具有注意力机制的线性滤波器 

**Authors**: TaeJun Ha, Chaehyun Jung, Hyeonuk Kim, Jeongwoo Park, Jeonghun Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.00452)  

**Abstract**: In orthogonal frequency division multiplexing (OFDM), accurate channel estimation is crucial. Classical signal processing based approaches, such as minimum mean-squared error (MMSE) estimation, often require second-order statistics that are difficult to obtain in practice. Recent deep neural networks based methods have been introduced to address this; yet they often suffer from high complexity. This paper proposes an Attention-aided MMSE (A-MMSE), a novel model-based DNN framework that learns the optimal MMSE filter via the Attention Transformer. Once trained, the A-MMSE estimates the channel through a single linear operation for channel estimation, eliminating nonlinear activations during inference and thus reducing computational complexity. To enhance the learning efficiency of the A-MMSE, we develop a two-stage Attention encoder, designed to effectively capture the channel correlation structure. Additionally, a rank-adaptive extension of the proposed A-MMSE allows flexible trade-offs between complexity and channel estimation accuracy. Extensive simulations with 3GPP TDL channel models demonstrate that the proposed A-MMSE consistently outperforms other baseline methods in terms of normalized MSE across a wide range of SNR conditions. In particular, the A-MMSE and its rank-adaptive extension establish a new frontier in the performance complexity trade-off, redefining the standard for practical channel estimation methods. 

**Abstract (ZH)**: 基于注意力辅助最小均方误差的 OFDM 信道估计方法 

---
# RLAE: Reinforcement Learning-Assisted Ensemble for LLMs 

**Title (ZH)**: RLAE: 基于强化学习的LLM集成方法 

**Authors**: Yuqian Fu, Yuanheng Zhu, Jiajun Chai, Guojun Yin, Wei Lin, Qichao Zhang, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00439)  

**Abstract**: Ensembling large language models (LLMs) can effectively combine diverse strengths of different models, offering a promising approach to enhance performance across various tasks. However, existing methods typically rely on fixed weighting strategies that fail to adapt to the dynamic, context-dependent characteristics of LLM capabilities. In this work, we propose Reinforcement Learning-Assisted Ensemble for LLMs (RLAE), a novel framework that reformulates LLM ensemble through the lens of a Markov Decision Process (MDP). Our approach introduces a RL agent that dynamically adjusts ensemble weights by considering both input context and intermediate generation states, with the agent being trained using rewards that directly correspond to the quality of final outputs. We implement RLAE using both single-agent and multi-agent reinforcement learning algorithms ($\text{RLAE}_\text{PPO}$ and $\text{RLAE}_\text{MAPPO}$ ), demonstrating substantial improvements over conventional ensemble methods. Extensive evaluations on a diverse set of tasks show that RLAE outperforms existing approaches by up to $3.3\%$ accuracy points, offering a more effective framework for LLM ensembling. Furthermore, our method exhibits superior generalization capabilities across different tasks without the need for retraining, while simultaneously achieving lower time latency. 

**Abstract (ZH)**: 增强学习辅助的大语言模型集成（RLAE）：一种基于马尔可夫决策过程的新框架 

---
# Is Your Explanation Reliable: Confidence-Aware Explanation on Graph Neural Networks 

**Title (ZH)**: 你的解释可靠吗：图神经网络中的置信意识解释 

**Authors**: Jiaxing Zhang, Xiaoou Liu, Dongsheng Luo, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00437)  

**Abstract**: Explaining Graph Neural Networks (GNNs) has garnered significant attention due to the need for interpretability, enabling users to understand the behavior of these black-box models better and extract valuable insights from their predictions. While numerous post-hoc instance-level explanation methods have been proposed to interpret GNN predictions, the reliability of these explanations remains uncertain, particularly in the out-of-distribution or unknown test datasets. In this paper, we address this challenge by introducing an explainer framework with the confidence scoring module ( ConfExplainer), grounded in theoretical principle, which is generalized graph information bottleneck with confidence constraint (GIB-CC), that quantifies the reliability of generated explanations. Experimental results demonstrate the superiority of our approach, highlighting the effectiveness of the confidence score in enhancing the trustworthiness and robustness of GNN explanations. 

**Abstract (ZH)**: 解释图神经网络（GNN）因其可解释性需求而引起了广泛关注，这使用户能够更好地理解这些黑盒模型的行为并从其预测中提取有价值的见解。尽管已经提出了许多事后实例级解释方法来解释GNN预测，但这些解释的可靠性仍然存疑，尤其是在分布外或未知测试数据集中。在本文中，我们通过引入一个基于理论原理并带有置信度评分模块（ConfExplainer）的解释框架来应对这一挑战，该框架是广义图信息瓶颈与置信度约束（GIB-CC）的理论概括，量化了生成解释的可靠性。实验结果表明了我们方法的优势，突显了置信度评分在提高GNN解释的信任度和鲁棒性方面的有效性。 

---
# Learning from Double Positive and Unlabeled Data for Potential-Customer Identification 

**Title (ZH)**: 基于双正样本和未标注数据的学习方法及其在潜在客户识别中的应用 

**Authors**: Masahiro Kato, Yuki Ikeda abd Kentaro Baba, Takashi Imai, Ryo Inokuchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00436)  

**Abstract**: In this study, we propose a method for identifying potential customers in targeted marketing by applying learning from positive and unlabeled data (PU learning). We consider a scenario in which a company sells a product and can observe only the customers who purchased it. Decision-makers seek to market products effectively based on whether people have loyalty to the company. Individuals with loyalty are those who are likely to remain interested in the company even without additional advertising. Consequently, those loyal customers would likely purchase from the company if they are interested in the product. In contrast, people with lower loyalty may overlook the product or buy similar products from other companies unless they receive marketing attention. Therefore, by focusing marketing efforts on individuals who are interested in the product but do not have strong loyalty, we can achieve more efficient marketing. To achieve this goal, we consider how to learn, from limited data, a classifier that identifies potential customers who (i) have interest in the product and (ii) do not have loyalty to the company. Although our algorithm comprises a single-stage optimization, its objective function implicitly contains two losses derived from standard PU learning settings. For this reason, we refer to our approach as double PU learning. We verify the validity of the proposed algorithm through numerical experiments, confirming that it functions appropriately for the problem at hand. 

**Abstract (ZH)**: 基于正例和未标注数据双正例学习的潜在客户识别方法研究 

---
# Channel Normalization for Time Series Channel Identification 

**Title (ZH)**: 时间序列通道识别中的通道归一化 

**Authors**: Seunghan Lee, Taeyoung Park, Kibok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.00432)  

**Abstract**: Channel identifiability (CID) refers to the ability to distinguish between individual channels in time series (TS) modeling. The absence of CID often results in producing identical outputs for identical inputs, disregarding channel-specific characteristics. In this paper, we highlight the importance of CID and propose Channel Normalization (CN), a simple yet effective normalization strategy that enhances CID by assigning distinct affine transformation parameters to each channel. We further extend CN in two ways: 1) Adaptive CN (ACN) dynamically adjusts parameters based on the input TS, improving adaptability in TS models, and 2) Prototypical CN (PCN) introduces a set of learnable prototypes instead of per-channel parameters, enabling applicability to datasets with unknown or varying number of channels and facilitating use in TS foundation models. We demonstrate the effectiveness of CN and its variants by applying them to various TS models, achieving significant performance gains for both non-CID and CID models. In addition, we analyze the success of our approach from an information theory perspective. Code is available at this https URL. 

**Abstract (ZH)**: 信道可分辨性（CID）指的是在时间序列（TS）建模中区分个体信道的能力。CID的缺失常常导致在给定相同输入时产生相同输出，忽略信道特异性特征。本文强调了CID的重要性，并提出了一种简单有效的正则化策略——信道正则化（CN），通过为每个信道分配独特的仿射变换参数来增强CID。我们进一步以两种方式扩展了CN：1）自适应信道正则化（ACN）根据输入TS动态调整参数，增强TS模型的适应性；2）原型信道正则化（PCN）引入了一组可学习的原型，而不是信道特定参数，使其适用于具有未知或变化数量信道的数据集，并便于TS基础模型的应用。我们通过将其应用于多种TS模型，展示了CN及其变体的有效性，实现了非CID模型和CID模型的重大性能提升。此外，我们从信息论的角度分析了我们方法的成功。代码详见this https URL。 

---
# COGNATE: Acceleration of Sparse Tensor Programs on Emerging Hardware using Transfer Learning 

**Title (ZH)**: COGNATE：利用迁移学习加速新兴硬件上的稀疏张量程序 

**Authors**: Chamika Sudusinghe, Gerasimos Gerogiannis Damitha Lenadora, Charles Block, Josep Torrellas, Charith Mendis  

**Link**: [PDF](https://arxiv.org/pdf/2506.00424)  

**Abstract**: Sparse tensor programs are essential in deep learning and graph analytics, driving the need for optimized processing. To meet this demand, specialized hardware accelerators are being developed. Optimizing these programs for accelerators is challenging for two reasons: program performance is highly sensitive to variations in sparse inputs, and early-stage accelerators rely on expensive simulators. Therefore, ML-based cost models used for optimizing such programs on general-purpose hardware are often ineffective for early-stage accelerators, as they require large datasets for proper training. To this end, we introduce COGNATE, a novel framework that leverages inexpensive data samples from general-purpose hardware (e.g., CPUs) to train cost models, followed by few-shot fine-tuning on emerging hardware. COGNATE exploits the homogeneity of input features across hardware platforms while effectively mitigating heterogeneity, enabling cost model training with just 5% of the data samples needed by accelerator-specific models to achieve comparable performance. We conduct extensive experiments to demonstrate that COGNATE outperforms existing techniques, achieving average speedups of 1.47x (up to 5.46x) for SpMM and 1.39x (up to 4.22x) for SDDMM. 

**Abstract (ZH)**: COGNATE：利用通用硬件数据样本进行高效稀疏张量程序成本模型训练的新型框架 

---
# Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System for Dynamic Interactions 

**Title (ZH)**: 赋予聊天机器人以耳闻目见的能力：一种用于动态交互的沉浸式多模态对话系统 

**Authors**: Jihyoung Jang, Minwook Bae, Minji Kim, Dilek Hakkani-Tur, Hyounghun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.00421)  

**Abstract**: As chatbots continue to evolve toward human-like, real-world, interactions, multimodality remains an active area of research and exploration. So far, efforts to integrate multimodality into chatbots have primarily focused on image-centric tasks, such as visual dialogue and image-based instructions, placing emphasis on the "eyes" of human perception while neglecting the "ears", namely auditory aspects. Moreover, these studies often center around static interactions that focus on discussing the modality rather than naturally incorporating it into the conversation, which limits the richness of simultaneous, dynamic engagement. Furthermore, while multimodality has been explored in multi-party and multi-session conversations, task-specific constraints have hindered its seamless integration into dynamic, natural conversations. To address these challenges, this study aims to equip chatbots with "eyes and ears" capable of more immersive interactions with humans. As part of this effort, we introduce a new multimodal conversation dataset, Multimodal Multi-Session Multi-Party Conversation ($M^3C$), and propose a novel multimodal conversation model featuring multimodal memory retrieval. Our model, trained on the $M^3C$, demonstrates the ability to seamlessly engage in long-term conversations with multiple speakers in complex, real-world-like settings, effectively processing visual and auditory inputs to understand and respond appropriately. Human evaluations highlight the model's strong performance in maintaining coherent and dynamic interactions, demonstrating its potential for advanced multimodal conversational agents. 

**Abstract (ZH)**: 随着聊天机器人的不断发展，趋向于实现自然真实的人类交互，多模态交互仍然是一个活跃的研究领域。目前，将多模态集成到聊天机器人中主要集中在以图像为中心的任务上，如视觉对话和图像指令，侧重于人类感知的“眼睛”，而忽视了“耳朵”，即听觉方面。此外，这些研究往往集中于静态交互，侧重于讨论模态本身而非自然地将其融入对话中，限制了同时动态交互的丰富性。进一步而言，虽然多模态已经在多轮和多会话对话中被探索，但特定任务的约束限制了其在自然、动态对话中的无缝集成。为了应对这些挑战，本研究旨在赋予聊天机器人“眼睛和耳朵”，以实现更沉浸式的交互。作为此努力的一部分，我们引入了一个新的多模态多会话多对话语音聊天数据集($M^3C$)，并提出了一种具有多模态记忆检索的新型多模态对话模型。该模型在$M^3C$数据集上训练，能够无缝参与复杂的、现实世界的长对话，有效处理视觉和听觉输入，以理解并适当地回应。人类评估表明，该模型在保持连贯和动态交互方面表现出色，展示了其作为先进多模态对话代理的潜力。 

---
# A New Spatiotemporal Correlation Anomaly Detection Method that Integrates Contrastive Learning and Few-Shot Learning in Wireless Sensor Networks 

**Title (ZH)**: 一种结合对比学习和少样本学习的无线传感器网络时空相关异常检测新方法 

**Authors**: Miao Ye, Suxiao Wang, Jiaguang Han, Yong Wang, Xiaoli Wang, Jingxuan Wei, Peng Wen, Jing Cui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00420)  

**Abstract**: Detecting anomalies in the data collected by WSNs can provide crucial evidence for assessing the reliability and stability of WSNs. Existing methods for WSN anomaly detection often face challenges such as the limited extraction of spatiotemporal correlation features, the absence of sample labels, few anomaly samples, and an imbalanced sample distribution. To address these issues, a spatiotemporal correlation detection model (MTAD-RD) considering both model architecture and a two-stage training strategy perspective is proposed. In terms of model structure design, the proposed MTAD-RD backbone network includes a retentive network (RetNet) enhanced by a cross-retention (CR) module, a multigranular feature fusion module, and a graph attention network module to extract internode correlation information. This proposed model can integrate the intermodal correlation features and spatial features of WSN neighbor nodes while extracting global information from time series data. Moreover, its serialized inference characteristic can remarkably reduce inference overhead. For model training, a two-stage training approach was designed. First, a contrastive learning proxy task was designed for time series data with graph structure information in WSNs, enabling the backbone network to learn transferable features from unlabeled data using unsupervised contrastive learning methods, thereby addressing the issue of missing sample labels in the dataset. Then, a caching-based sample sampler was designed to divide samples into few-shot and contrastive learning data. A specific joint loss function was developed to jointly train the dual-graph discriminator network to address the problem of sample imbalance effectively. In experiments carried out on real public datasets, the designed MTAD-RD anomaly detection method achieved an F1 score of 90.97%, outperforming existing supervised WSN anomaly detection methods. 

**Abstract (ZH)**: 基于时空关联检测的WSN异常检测方法（MTAD-RD） 

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
# LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks 

**Title (ZH)**: LoHoVLA：统一的视觉-语言-行动模型用于长期 horizon 汰务 

**Authors**: Yi Yang, Jiaxuan Sun, Siqi Kou, Yihan Wang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00411)  

**Abstract**: Real-world embodied agents face long-horizon tasks, characterized by high-level goals demanding multi-step solutions beyond single actions. Successfully navigating these requires both high-level task planning (i.e., decomposing goals into sub-tasks) and low-level motion control (i.e., generating precise robot actions). While existing vision language action (VLA) models and hierarchical architectures offer potential in embodied tasks, the former often falter in planning, and the latter can suffer from coordination issues, both hampering performance. We introduce a new unified VLA framework for long-horizon tasks, dubbed LoHoVLA, to overcome these limitations. LoHoVLA leverages a large pretrained vision language model (VLM) as the backbone to jointly generate language and action tokens for sub-task generation and robot action prediction, respectively. This shared representation promotes better generalization across tasks. Additionally, LoHoVLA embraces a hierarchical closed-loop control mechanism to mitigate errors originating from both high-level planning and low-level control. To train LoHoVLA, we introduce LoHoSet, a dataset built on the Ravens simulator, containing 20 long-horizon tasks, each with 1,000 expert demonstrations composed of visual observations, linguistic goals, sub-tasks, and robot actions. Experimental results show that LoHoVLA significantly surpasses both hierarchical and standard VLA approaches on long-horizon embodied tasks in the Ravens simulator. These findings underscore the promise of unified architectures for advancing generalizable embodied intelligence. 

**Abstract (ZH)**: LoHoVLA：用于长时_horizon任务的统一视觉语言行动框架 

---
# Bias as a Virtue: Rethinking Generalization under Distribution Shifts 

**Title (ZH)**: 偏见作为一种美德：在分布偏移情况下的重新思考泛化能力 

**Authors**: Ruixuan Chen, Wentao Li, Jiahui Xiao, Yuchen Li, Yimin Tang, Xiaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00407)  

**Abstract**: Machine learning models often degrade when deployed on data distributions different from their training data. Challenging conventional validation paradigms, we demonstrate that higher in-distribution (ID) bias can lead to better out-of-distribution (OOD) generalization. Our Adaptive Distribution Bridge (ADB) framework implements this insight by introducing controlled statistical diversity during training, enabling models to develop bias profiles that effectively generalize across distributions. Empirically, we observe a robust negative correlation where higher ID bias corresponds to lower OOD error--a finding that contradicts standard practices focused on minimizing validation error. Evaluation on multiple datasets shows our approach significantly improves OOD generalization. ADB achieves robust mean error reductions of up to 26.8% compared to traditional cross-validation, and consistently identifies high-performing training strategies, evidenced by percentile ranks often exceeding 74.4%. Our work provides both a practical method for improving generalization and a theoretical framework for reconsidering the role of bias in robust machine learning. 

**Abstract (ZH)**: 机器学习模型在部署于与训练数据不同的数据分布时往往会退化。挑战传统的验证范式，我们证明了较高的同分布（ID）偏差可以导致更好的异分布（OOD）泛化。我们的自适应分布桥（ADB）框架通过在训练过程中引入受控的统计多样性来实现这一洞察，使模型能够发展出有效地跨越分布进行泛化的偏差配置。实证研究表明，较高的ID偏差与较低的OOD误差之间存在稳健的负相关关系——这一发现与关注于最小化验证误差的常规做法相矛盾。在多个数据集上的评估结果显示，我们的方法显著提高了异分布泛化能力。ADB实现了与传统交叉验证相比高达26.8%的稳健均值误差减少，并且一贯地识别出高性能的训练策略，证据表明百分位排名通常超过了74.4%。我们的工作既提供了一种改进泛化的实用方法，也提供了一个重新考虑偏差在稳健机器学习中作用的理论框架。 

---
# Scaling Textual Gradients via Sampling-Based Momentum 

**Title (ZH)**: 基于采样加速的文本梯度扩展 

**Authors**: Zixin Ding, Junyuan Hong, Jiachen T. Wang, Zinan Lin, Zhangyang Wang, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00400)  

**Abstract**: As prompts play an increasingly critical role in large language models (LLMs), optimizing textual prompts has become a crucial challenge. The Textual Gradient Descent (TGD) framework has emerged as a promising data-driven approach that iteratively refines textual prompts using LLM - suggested updates (or textual gradients) over minibatches of training samples. In this paper, we empirically demonstrate that scaling the number of training examples initially improves but later degrades TGD's performance across multiple downstream NLP tasks. However, while data scaling improves results for most tasks, it also significantly increases the computational cost when leveraging LLMs. To address this, we draw inspiration from numerical gradient descent and propose Textual Stochastic Gradient Descent with Momentum (TSGD-M) - a method that facilitates scalable in-context learning by reweighting prompt sampling based on past batch distributions. Across nine NLP tasks spanning three domains - including BIG-Bench Hard (BBH), natural language understanding tasks, and reasoning tasks - TSGD-M significantly outperforms TGD baselines that do not incorporate reweighted sampling, while also reducing variance in most tasks. 

**Abstract (ZH)**: 大型语言模型中文本提示的优化：文本随机梯度下降带动量（TSGD-M）方法的研究 

---
# MagiCodec: Simple Masked Gaussian-Injected Codec for High-Fidelity Reconstruction and Generation 

**Title (ZH)**: MagiCodec：简单的遮掩高斯注入编解码器，用于高保真重建与生成 

**Authors**: Yakun Song, Jiawei Chen, Xiaobin Zhuang, Chenpeng Du, Ziyang Ma, Jian Wu, Jian Cong, Dongya Jia, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00385)  

**Abstract**: Neural audio codecs have made significant strides in efficiently mapping raw audio waveforms into discrete token representations, which are foundational for contemporary audio generative models. However, most existing codecs are optimized primarily for reconstruction quality, often at the expense of the downstream modelability of the encoded tokens. Motivated by the need to overcome this bottleneck, we introduce $\textbf{MagiCodec}$, a novel single-layer, streaming Transformer-based audio codec. MagiCodec is designed with a multistage training pipeline that incorporates Gaussian noise injection and latent regularization, explicitly targeting the enhancement of semantic expressiveness in the generated codes while preserving high reconstruction fidelity. We analytically derive the effect of noise injection in the frequency domain, demonstrating its efficacy in attenuating high-frequency components and fostering robust tokenization. Extensive experimental evaluations show that MagiCodec surpasses state-of-the-art codecs in both reconstruction quality and downstream tasks. Notably, the tokens produced by MagiCodec exhibit Zipf-like distributions, as observed in natural languages, thereby improving compatibility with language-model-based generative architectures. The code and pre-trained models are available at this https URL. 

**Abstract (ZH)**: 神经音频编解码器已经在高效地将原始音频波形映射为离散符号表示方面取得了显著进展，这些表示是当代音频生成模型的基础。然而，现有的大多数编解码器主要优化重建质量，往往以牺牲编码符号的下游模型能力强度为代价。为了克服这一瓶颈，我们介绍了名为MagiCodec的新型单层流式Transformer基音频编解码器。MagiCodec采用一个多阶段训练管道，结合了高斯噪声注入和潜在正则化，明确地旨在增强生成代码的语义表达能力，同时保持高重建保真度。我们在频域中分析了噪声注入的效果，证实了其在衰减高频分量并促进稳健符号化的有效性。广泛的实验评估表明，MagiCodec在重建质量和下游任务方面均优于最先进的编解码器。值得注意的是，MagiCodec生成的符号表现出类似Zipf的分布，类似于自然语言，从而提高了与基于语言模型的生成架构的兼容性。代码和预训练模型可在以下链接获取。 

---
# Neural Network-based Information-Theoretic Transceivers for High-Order Modulation Schemes 

**Title (ZH)**: 基于神经网络的信息论传输机高阶调制方案 

**Authors**: Ngoc Long Pham, Tri Nhu Do  

**Link**: [PDF](https://arxiv.org/pdf/2506.00368)  

**Abstract**: Neural network (NN)-based end-to-end (E2E) communication systems, in which each system component may consist of a portion of a neural network, have been investigated as potential tools for developing artificial intelligence (Al)-native E2E systems. In this paper, we propose an NN-based bitwise receiver that improves computational efficiency while maintaining performance comparable to baseline demappers. Building on this foundation, we introduce a novel symbol-wise autoencoder (AE)-based E2E system that jointly optimizes the transmitter and receiver at the physical layer. We evaluate the proposed NN-based receiver using bit-error rate (BER) analysis to confirm that the numerical BER achieved by NN-based receivers or transceivers is accurate. Results demonstrate that the AE-based system outperforms baseline architectures, particularly for higher-order modulation schemes. We further show that the training signal-to-noise ratio (SNR) significantly affects the performance of the systems when inference is conducted at different SNR levels. 

**Abstract (ZH)**: 基于神经网络的端到端通信系统中，每个系统组件可能包括神经网络的部分，已被探索作为开发人工智能原生端到端系统的潜在工具。在本文中，我们提出了一种基于神经网络的位级接收机，该接收机在保持与基准译码器相当的性能的同时提高了计算效率。在此基础上，我们引入了一种基于符号级自动编码器的端到端系统，该系统在物理层上联合优化了发送端和接收端。我们通过位误比特率(BER)分析评估所提出的基于神经网络的接收机，以确认基于神经网络的接收机或收发机实现的数值BER是准确的。结果表明，基于自动编码器的系统在高阶调制方案中性能优于基准架构。此外，我们还展示了训练信号噪声比(SNR)在不同SNR水平下进行推理时对系统性能的影响显著。 

---
# $\texttt{AVROBUSTBENCH}$: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time 

**Title (ZH)**: AVROBUSTBENCH：测试时音频-视觉识别模型鲁棒性对比基准 

**Authors**: Sarthak Kumar Maharana, Saksham Singh Kushwaha, Baoming Zhang, Adrian Rodriguez, Songtao Wei, Yapeng Tian, Yunhui Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00358)  

**Abstract**: While recent audio-visual models have demonstrated impressive performance, their robustness to distributional shifts at test-time remains not fully understood. Existing robustness benchmarks mainly focus on single modalities, making them insufficient for thoroughly assessing the robustness of audio-visual models. Motivated by real-world scenarios where shifts can occur $\textit{simultaneously}$ in both audio and visual modalities, we introduce $\texttt{AVROBUSTBENCH}$, a comprehensive benchmark designed to evaluate the test-time robustness of audio-visual recognition models. $\texttt{AVROBUSTBENCH}$ comprises four audio-visual benchmark datasets, $\texttt{AUDIOSET-2C}$, $\texttt{VGGSOUND-2C}$, $\texttt{KINETICS-2C}$, and $\texttt{EPICKITCHENS-2C}$, each incorporating 75 bimodal audio-visual corruptions that are $\textit{co-occurring}$ and $\textit{correlated}$. Through extensive evaluations, we observe that state-of-the-art supervised and self-supervised audio-visual models exhibit declining robustness as corruption severity increases. Furthermore, online test-time adaptation (TTA) methods, on $\texttt{VGGSOUND-2C}$ and $\texttt{KINETICS-2C}$, offer minimal improvements in performance under bimodal corruptions. We further propose $\texttt{AV2C}$, a simple TTA approach enabling on-the-fly cross-modal fusion by penalizing high-entropy samples, which achieves improvements on $\texttt{VGGSOUND-2C}$. We hope that $\texttt{AVROBUSTBENCH}$ will steer the development of more effective and robust audio-visual TTA approaches. Our code is available $\href{this https URL}{here}$. 

**Abstract (ZH)**: 尽管近期的多模态模型展示了出色的性能，但它们在测试时对分布偏移的鲁棒性尚未完全理解。现有的鲁棒性基准主要关注单一模态，不足以全面评估多模态音频-视觉模型的鲁棒性。受实际场景中音频和视觉模态同时发生偏移的启发，我们引入了AVROBUSTBENCH，这是一个全面的基准，旨在评估音频-视觉识别模型的测试时鲁棒性。AVROBUSTBENCH 包含四个音频-视觉基准数据集：AUDIOSET-2C、VGGSOUND-2C、KINETICS-2C 和 EPICKITCHENS-2C，每个数据集包含 75 种同时发生且相关的双模态音频-视觉污染。通过广泛的评估，我们观察到最先进的监督和自监督音频-视觉模型在污染严重性增加时表现出了鲁棒性下降的现象。此外，在 VGGSOUND-2C 和 KINETICS-2C 上，实时测试时自适应（TTA）方法在双模态污染下的性能改善有限。我们进一步提出了 AV2C，这是一个简单的 TTA 方法，通过惩罚高熵样本实现跨模态融合，从而在 VGGSOUND-2C 上实现了性能提升。我们希望 AVROBUSTBENCH 能够引导更有效、更鲁棒的音频-视觉 TTA 方法的发展。我们的代码可在 <https://this https URL> 获取。 

---
# Exploring the Performance of Perforated Backpropagation through Further Experiments 

**Title (ZH)**: 探索穿透反向传播性能的进一步实验 

**Authors**: Rorry Brenner, Evan Davis, Rushi Chaudhari, Rowan Morse, Jingyao Chen, Xirui Liu, Zhaoyi You, Laurent Itti  

**Link**: [PDF](https://arxiv.org/pdf/2506.00356)  

**Abstract**: Perforated Backpropagation is a neural network optimization technique based on modern understanding of the computational importance of dendrites within biological neurons. This paper explores further experiments from the original publication, generated from a hackathon held at the Carnegie Mellon Swartz Center in February 2025. Students and local Pittsburgh ML practitioners were brought together to experiment with the Perforated Backpropagation algorithm on the datasets and models which they were using for their projects. Results showed that the system could enhance their projects, with up to 90% model compression without negative impact on accuracy, or up to 16% increased accuracy of their original models. 

**Abstract (ZH)**: 穿孔反向传播是一种基于对生物神经元树突计算重要性现代理解的神经网络优化技术。本文进一步探索了2025年2月在卡内基梅隆斯瓦兹中心举办的黑客马拉松中最初发表论文生成的实验。学生和当地匹兹堡的机器学习从业者共同实验了穿孔反向传播算法对其项目中使用的数据集和模型。结果显示，该系统能够提升其项目性能，最高可达90%的模型压缩比例而不影响准确性，或者在原有模型基础上提高16%的准确率。 

---
# Enabling Secure and Ephemeral AI Workloads in Data Mesh Environments 

**Title (ZH)**: 在数据网状环境中的安全且临时的AI工作负载启用 

**Authors**: Chinkit Patel, Kee Siong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00352)  

**Abstract**: Many large enterprises that operate highly governed and complex ICT environments have no efficient and effective way to support their Data and AI teams in rapidly spinning up and tearing down self-service data and compute infrastructure, to experiment with new data analytic tools, and deploy data products into operational use. This paper proposes a key piece of the solution to the overall problem, in the form of an on-demand self-service data-platform infrastructure to empower de-centralised data teams to build data products on top of centralised templates, policies and governance. The core innovation is an efficient method to leverage immutable container operating systems and infrastructure-as-code methodologies for creating, from scratch, vendor-neutral and short-lived Kubernetes clusters on-premises and in any cloud environment. Our proposed approach can serve as a repeatable, portable and cost-efficient alternative or complement to commercial Platform-as-a-Service (PaaS) offerings, and this is particularly important in supporting interoperability in complex data mesh environments with a mix of modern and legacy compute infrastructure. 

**Abstract (ZH)**: 许多运营高度管控和复杂 ICT 环境的大型企业缺乏高效有效的方法来支持其数据和 AI 团队快速搭建和销毁自助服务数据和计算基础设施、实验新的数据分析工具以及将数据产品部署到生产环境中。本文提出了整体解决方案的关键组成部分，即一种按需自助的数据平台基础设施，以此赋能分散的数据团队在中央模板、政策和治理的基础上构建数据产品。核心创新是一种高效的方法，利用不可变容器操作系统和基础设施即代码方法，在任何本地或云环境从零开始创建中立的、短暂的 Kubernetes 集群。我们提出的这种方法可以作为商业平台即服务（PaaS）产品的可重复、可移植且成本效益高的替代方案或补充，特别是在支持混合现代和遗留计算基础设施的复杂数据湖环境中尤为重要。 

---
# Beyond Winning: Margin of Victory Relative to Expectation Unlocks Accurate Skill Ratings 

**Title (ZH)**: 超越胜利：相对于期望的获胜优势解锁了准确的技能评级 

**Authors**: Shivam Shorewala, Zihao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00348)  

**Abstract**: Knowledge of accurate relative skills in any competitive system is essential, but foundational approaches such as ELO discard extremely relevant performance data by concentrating exclusively on binary outcomes. While margin of victory (MOV) extensions exist, they often lack a definitive method for incorporating this information. We introduce Margin of Victory Differential Analysis (MOVDA), a framework that enhances traditional rating systems by using the deviation between the true MOV and a $\textit{modeled expectation}$. MOVDA learns a domain-specific, non-linear function (a scaled hyperbolic tangent that captures saturation effects and home advantage) to predict expected MOV based on rating differentials. Crucially, the $\textit{difference}$ between the true and expected MOV provides a subtle and weighted signal for rating updates, highlighting informative deviations in all levels of contests. Extensive experiments on professional NBA basketball data (from 2013 to 2023, with 13,619 games) show that MOVDA significantly outperforms standard ELO and Bayesian baselines. MOVDA reduces Brier score prediction error by $1.54\%$ compared to TrueSkill, increases outcome accuracy by $0.58\%$, and most importantly accelerates rating convergence by $13.5\%$, while maintaining the computational efficiency of the original ELO updates. MOVDA offers a theoretically motivated, empirically superior, and computationally lean approach to integrating performance magnitude into skill rating for competitive environments like the NBA. 

**Abstract (ZH)**: 基于胜利 margin 的差异分析在提升竞技系统技能评级中的应用 

---
# Efficient Latent Semantic Clustering for Scaling Test-Time Computation of LLMs 

**Title (ZH)**: 高效的潜在语义聚类方法以扩展大型语言模型测试时的计算能力 

**Authors**: Sungjae Lee, Hoyoung Kim, Jeongyeon Hwang, Eunhyeok Park, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2506.00344)  

**Abstract**: Scaling test-time computation--generating and analyzing multiple or sequential outputs for a single input--has become a promising strategy for improving the reliability and quality of large language models (LLMs), as evidenced by advances in uncertainty quantification and multi-step reasoning. A key shared component is semantic clustering, which groups outputs that differ in form but convey the same meaning. Semantic clustering enables estimation of the distribution over the semantics of outputs and helps avoid redundant exploration of reasoning paths. However, existing approaches typically rely on external models, which introduce substantial computational overhead and often fail to capture context-aware semantics. We propose Latent Semantic Clustering (LSC), a lightweight and context-sensitive method that leverages the generator LLM's internal hidden states for clustering, eliminating the need for external models. Our extensive experiment across various LLMs and datasets shows that LSC significantly improves the computational efficiency of test-time scaling while maintaining or exceeding the performance of existing methods. 

**Abstract (ZH)**: 扩展测试时计算——为单个输入生成和分析多个或序列输出已成为提高大型语言模型（LLMs）可靠性和质量的有前途的策略，这得到了不确定性量化和多步推理进步的证明。一项关键共有组件是语义聚类，它将形式不同但意义相同的输出分组。语义聚类使输出语义分布的估计成为可能，并有助于避免对推理路径的冗余探索。然而，现有方法通常依赖于外部模型，这引入了大量计算开销，并且往往无法捕捉上下文感知的语义。我们提出了隐含语义聚类（LSC），这是一种轻量级且上下文感知的方法，利用生成器LLM的内部隐藏状态进行聚类，从而消除对外部模型的依赖。在各种LLM和数据集上的广泛实验表明，LSC在维护或超过现有方法性能的同时，显著提高了测试时扩展的计算效率。 

---
# Recover Experimental Data with Selection Bias using Counterfactual Logic 

**Title (ZH)**: 使用反事实逻辑恢复具有选择偏见的实验数据 

**Authors**: Jingyang He, Shuai Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00335)  

**Abstract**: Selection bias, arising from the systematic inclusion or exclusion of certain samples, poses a significant challenge to the validity of causal inference. While Bareinboim et al. introduced methods for recovering unbiased observational and interventional distributions from biased data using partial external information, the complexity of the backdoor adjustment and the method's strong reliance on observational data limit its applicability in many practical settings. In this paper, we formally discover the recoverability of $P(Y^*_{x^*})$ under selection bias with experimental data. By explicitly constructing counterfactual worlds via Structural Causal Models (SCMs), we analyze how selection mechanisms in the observational world propagate to the counterfactual domain. We derive a complete set of graphical and theoretical criteria to determine that the experimental distribution remain unaffected by selection bias. Furthermore, we propose principled methods for leveraging partially unbiased observational data to recover $P(Y^*_{x^*})$ from biased experimental datasets. Simulation studies replicating realistic research scenarios demonstrate the practical utility of our approach, offering concrete guidance for mitigating selection bias in applied causal inference. 

**Abstract (ZH)**: 选择偏差导致的系统性样本包括或排除问题对因果推理的有效性构成了重大挑战。尽管Bareinboim等人引入了利用部分外部信息从偏差数据中恢复无偏观察和干预分布的方法，但后门调整的复杂性以及该方法对观察数据的高依赖性限制了其在许多实际情境中的适用性。本文正式揭示了通过实验数据在选择偏差下恢复 $P(Y^*_{x^*})$ 的可行性。通过结构因果模型（SCMs）明确构建反事实世界，我们分析了观察世界中的选择机制如何传播到反事实领域。我们推导出一套完整的图形和理论标准，以确定实验分布不受到选择偏差的影响。此外，我们提出了利用部分无偏观察数据恢复偏差实验数据集中 $P(Y^*_{x^*})$ 的原则方法。模拟研究表明，本方法在实际因果推理中的实用价值，提供了具体指导以减轻应用中的选择偏差。 

---
# Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation 

**Title (ZH)**: foresight: 自适应层重用以实现加速和高质文本到视频生成 

**Authors**: Muhammad Adnan, Nithesh Kurella, Akhil Arunkumar, Prashant J. Nair  

**Link**: [PDF](https://arxiv.org/pdf/2506.00329)  

**Abstract**: Diffusion Transformers (DiTs) achieve state-of-the-art results in text-to-image, text-to-video generation, and editing. However, their large model size and the quadratic cost of spatial-temporal attention over multiple denoising steps make video generation computationally expensive. Static caching mitigates this by reusing features across fixed steps but fails to adapt to generation dynamics, leading to suboptimal trade-offs between speed and quality.
We propose Foresight, an adaptive layer-reuse technique that reduces computational redundancy across denoising steps while preserving baseline performance. Foresight dynamically identifies and reuses DiT block outputs for all layers across steps, adapting to generation parameters such as resolution and denoising schedules to optimize efficiency. Applied to OpenSora, Latte, and CogVideoX, Foresight achieves up to 1.63x end-to-end speedup, while maintaining video quality. The source code of Foresight is available at \texttt{this https URL}. 

**Abstract (ZH)**: Foresight: 一种适应性的层重用技术，用于降低去噪步骤中的计算冗余同时保持基线性能 

---
# Latent Guidance in Diffusion Models for Perceptual Evaluations 

**Title (ZH)**: 扩散模型中潜导引的感知评估 

**Authors**: Shreshth Saini, Ru-Ling Liao, Yan Ye, Alan C. Bovik  

**Link**: [PDF](https://arxiv.org/pdf/2506.00327)  

**Abstract**: Despite recent advancements in latent diffusion models that generate high-dimensional image data and perform various downstream tasks, there has been little exploration into perceptual consistency within these models on the task of No-Reference Image Quality Assessment (NR-IQA). In this paper, we hypothesize that latent diffusion models implicitly exhibit perceptually consistent local regions within the data manifold. We leverage this insight to guide on-manifold sampling using perceptual features and input measurements. Specifically, we propose Perceptual Manifold Guidance (PMG), an algorithm that utilizes pretrained latent diffusion models and perceptual quality features to obtain perceptually consistent multi-scale and multi-timestep feature maps from the denoising U-Net. We empirically demonstrate that these hyperfeatures exhibit high correlation with human perception in IQA tasks. Our method can be applied to any existing pretrained latent diffusion model and is straightforward to integrate. To the best of our knowledge, this paper is the first work on guiding diffusion model with perceptual features for NR-IQA. Extensive experiments on IQA datasets show that our method, LGDM, achieves state-of-the-art performance, underscoring the superior generalization capabilities of diffusion models for NR-IQA tasks. 

**Abstract (ZH)**: 尽管近期在生成高维图像数据并执行各种下游任务方面取得了进展的潜在扩散模型已取得显著成就，但在无参考图像质量评估（NR-IQA）任务中这些模型的感知一致性方面研究较少。在本文中，我们假设潜在扩散模型隐式地在数据流形中表现出感知一致的局部区域。我们利用这一洞察，通过感知特征和输入测量引导流形上的采样。具体而言，我们提出了感知流形引导（PMG）算法，该算法利用预训练的潜在扩散模型和感知质量特征，从去噪的UNet中获得多尺度和多时间步的感知一致性特征图。我们实证证明了这些超特征在IQA任务中与人类感知具有高度相关性。该方法可以应用于任何现有的预训练潜在扩散模型，并且易于集成。据我们所知，这是首次利用感知特征引导扩散模型进行NR-IQA的研究。在IQA数据集上的广泛实验表明，我们的方法LGDM达到了最先进的性能，突显了扩散模型在NR-IQA任务中的出色泛化能力。 

---
# dpmm: Differentially Private Marginal Models, a Library for Synthetic Tabular Data Generation 

**Title (ZH)**: DPMM：不同质化边缘模型，一个合成表格数据生成库 

**Authors**: Sofiane Mahiou, Amir Dizche, Reza Nazari, Xinmin Wu, Ralph Abbey, Jorge Silva, Georgi Ganev  

**Link**: [PDF](https://arxiv.org/pdf/2506.00322)  

**Abstract**: We propose dpmm, an open-source library for synthetic data generation with Differentially Private (DP) guarantees. It includes three popular marginal models -- PrivBayes, MST, and AIM -- that achieve superior utility and offer richer functionality compared to alternative implementations. Additionally, we adopt best practices to provide end-to-end DP guarantees and address well-known DP-related vulnerabilities. Our goal is to accommodate a wide audience with easy-to-install, highly customizable, and robust model implementations.
Our codebase is available from this https URL. 

**Abstract (ZH)**: 我们提出dpmm，一个具有差分隐私保障的合成数据生成开源库。它包括三种流行的边际模型——PrivBayes、MST和AIM，与替代实现相比，提供了更好的效用和更丰富的功能。此外，我们采用了最佳实践来提供端到端的差分隐私保障，并解决了一些已知的差分隐私相关漏洞。我们的目标是为广泛受众提供易于安装、高度可定制且稳健的模型实现。

我们的代码库可以从以下链接获取：这个https URL。 

---
# An evaluation of LLMs for generating movie reviews: GPT-4o, Gemini-2.0 and DeepSeek-V3 

**Title (ZH)**: 对生成电影评论的LLMs进行评估：GPT-4o、Gemini-2.0和DeepSeek-V3 

**Authors**: Brendan Sands, Yining Wang, Chenhao Xu, Yuxuan Zhou, Lai Wei, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00312)  

**Abstract**: Large language models (LLMs) have been prominent in various tasks, including text generation and summarisation. The applicability of LLMs to the generation of product reviews is gaining momentum, paving the way for the generation of movie reviews. In this study, we propose a framework that generates movie reviews using three LLMs (GPT-4o, DeepSeek-V3, and Gemini-2.0), and evaluate their performance by comparing the generated outputs with IMDb user reviews. We use movie subtitles and screenplays as input to the LLMs and investigate how they affect the quality of reviews generated. We review the LLM-based movie reviews in terms of vocabulary, sentiment polarity, similarity, and thematic consistency in comparison to IMDB user reviews. The results demonstrate that LLMs are capable of generating syntactically fluent and structurally complete movie reviews. Nevertheless, there is still a noticeable gap in emotional richness and stylistic coherence between LLM-generated and IMDb reviews, suggesting that further refinement is needed to improve the overall quality of movie review generation. We provided a survey-based analysis where participants were told to distinguish between LLM and IMDb user reviews. The results show that LLM-generated reviews are difficult to distinguish from IMDB user reviews. We found that DeepSeek-V3 produced the most balanced reviews, closely matching IMDb reviews. GPT-4o overemphasised positive emotions, while Gemini-2.0 captured negative emotions better but showed excessive emotional intensity. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各类任务中占据重要地位，包括文本生成和摘要。LLMs在生成产品评论方面的应用逐渐增多，为电影评论的生成铺平了道路。本文提出一种框架，使用三种LLMs（GPT-4o、DeepSeek-V3和Gemini-2.0）生成电影评论，并通过将生成的输出与IMDb用户评论进行比较来评估其性能。我们使用电影字幕和剧本作为输入，探讨其对生成评论质量的影响。从词汇、情感极性、相似性和主题一致性方面，我们将LLM生成的电影评论与IMDb用户评论进行比较分析。结果表明，LLMs能够生成语法流畅且结构完整的电影评论。然而，LLM生成的评论在情感丰富性和风格连贯性方面仍与IMDb评论存在明显差距，提示需要进一步优化以提高电影评论生成的整体质量。我们还进行了基于调查的分析，让参与者辨别LLM生成的评论与IMDb用户评论。结果表明，LLM生成的评论难以区分。研究发现，DeepSeek-V3生成的评论最为平衡，接近IMDb评论。GPT-4o过分强调积极情绪，而Gemini-2.0较好地捕捉了负面情绪，但表现出过度的情感强度。 

---
# MythTriage: Scalable Detection of Opioid Use Disorder Myths on a Video-Sharing Platform 

**Title (ZH)**: MythTriage：大规模检测视频分享平台上阿片使用障碍错误认知的方法 

**Authors**: Hayoung Jung, Shravika Mittal, Ananya Aatreya, Navreet Kaur, Munmun De Choudhury, Tanushree Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00308)  

**Abstract**: Understanding the prevalence of misinformation in health topics online can inform public health policies and interventions. However, measuring such misinformation at scale remains a challenge, particularly for high-stakes but understudied topics like opioid-use disorder (OUD)--a leading cause of death in the U.S. We present the first large-scale study of OUD-related myths on YouTube, a widely-used platform for health information. With clinical experts, we validate 8 pervasive myths and release an expert-labeled video dataset. To scale labeling, we introduce MythTriage, an efficient triage pipeline that uses a lightweight model for routine cases and defers harder ones to a high-performing, but costlier, large language model (LLM). MythTriage achieves up to 0.86 macro F1-score while estimated to reduce annotation time and financial cost by over 76% compared to experts and full LLM labeling. We analyze 2.9K search results and 343K recommendations, uncovering how myths persist on YouTube and offering actionable insights for public health and platform moderation. 

**Abstract (ZH)**: 在线健康话题中的 misinformation 的普遍性理解对于公共卫生政策和干预具有指导意义。然而，大规模测量此类 misinformation 尤其对于高风险但研究不足的话题（如阿片使用障碍(OUD)）仍然是一个挑战。我们首次对 YouTube 上与 OUD 相关的 myths 进行了大规模研究，YouTube 是一个广泛用于获取健康信息的平台。通过临床专家验证了 8 个广泛流传的 myths，并发布了一个专家标注的视频数据集。为了扩大标注规模，我们引入了 MythTriage，一种高效triage 管道，使用轻量级模型处理常规案例，并将更难的案例转交给人工成本更高但性能更好的大语言模型（LLM）。MythTriage 达到了 0.86 的宏 F1 分数，同时估计将注释时间和财务成本降低了超过 76%。我们分析了 2900 个搜索结果和 343000 个建议，揭示了这些 myths 在 YouTube 上如何持续存在，并为公共卫生和平台 moderation 提供了实际的指导建议。 

---
# Lossless Token Sequence Compression via Meta-Tokens 

**Title (ZH)**: 基于元令牌的无损令牌序列压缩 

**Authors**: John Harvill, Ziwei Fan, Hao Wang, Yizhou Sun, Hao Ding, Luke Huan, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2506.00307)  

**Abstract**: Existing work on prompt compression for Large Language Models (LLM) focuses on lossy methods that try to maximize the retention of semantic information that is relevant to downstream tasks while significantly reducing the sequence length. In this paper, we introduce a task-agnostic lossless compression technique similar to LZ77 that makes it possible to reduce the input token sequence length on average by 27\% and 18\% for the two evaluation tasks explored here. Given that we use transformer-based LLMs, this equates to 47\% and 33\% less encoding computation, respectively, due to the quadratic nature of attention. The token sequence transformation is trivial to reverse and highlights that no semantic information is lost in the process. We evaluate our proposed approach on two tasks that require strict preservation of semantics/syntax and demonstrate that existing lossy compression methods perform poorly in this setting. We find that our lossless compression technique produces only a small gap in performance compared to using the uncompressed input and posit that larger models and an expanded computing budget would likely erase the gap entirely. 

**Abstract (ZH)**: 面向大型语言模型的提示压缩研究专注于在显著减少序列长度的同时，最大限度地保留与下游任务相关的语义信息。本文介绍了一种任务无关的无损压缩技术，类似于LZ77，能够平均减少两个评估任务输入标记序列长度的27%和18%。由于我们使用基于 Transformer 的大型语言模型，这意味着分别减少了47%和33%的编码计算量，这是由于注意力机制的二次特性。标记序列的转换易于逆向，表明在过程中没有丢失任何语义信息。我们评估了所提出的方法，并在需要严格保留语义/语法的两个任务中展示了现有无损压缩方法表现不佳的情况。我们发现，我们的无损压缩技术与使用未压缩输入相比，仅带来很小的性能差距，并推测更大的模型和更扩展的计算预算可能会完全消除这一差距。 

---
# Improving Protein Sequence Design through Designability Preference Optimization 

**Title (ZH)**: 通过设计能力偏好优化改进蛋白质序列设计 

**Authors**: Fanglei Xue, Andrew Kubaney, Zhichun Guo, Joseph K. Min, Ge Liu, Yi Yang, David Baker  

**Link**: [PDF](https://arxiv.org/pdf/2506.00297)  

**Abstract**: Protein sequence design methods have demonstrated strong performance in sequence generation for de novo protein design. However, as the training objective was sequence recovery, it does not guarantee designability--the likelihood that a designed sequence folds into the desired structure. To bridge this gap, we redefine the training objective by steering sequence generation toward high designability. To do this, we integrate Direct Preference Optimization (DPO), using AlphaFold pLDDT scores as the preference signal, which significantly improves the in silico design success rate. To further refine sequence generation at a finer, residue-level granularity, we introduce Residue-level Designability Preference Optimization (ResiDPO), which applies residue-level structural rewards and decouples optimization across residues. This enables direct improvement in designability while preserving regions that already perform well. Using a curated dataset with residue-level annotations, we fine-tune LigandMPNN with ResiDPO to obtain EnhancedMPNN, which achieves a nearly 3-fold increase in in silico design success rate (from 6.56% to 17.57%) on a challenging enzyme design benchmark. 

**Abstract (ZH)**: 蛋白质序列设计方法在从头蛋白设计的序列生成中展现出了强大的性能。然而，由于训练目标是序列恢复，这并不保证设计性——即设计序列折叠成所需结构的可能性。为弥补这一差距，我们重新定义了训练目标，通过将序列生成引导至高设计性来提高这一目标。为此，我们整合了直接偏好优化（DPO），使用AlphaFold的pLDDT分数作为偏好信号，显著提高了体外设计成功率。为进一步在更精细的残基级别上细化序列生成，我们引入了残基级别设计性偏好优化（ResiDPO），该方法应用了残基级别的结构奖励，并且拆分了对残基的优化。这使得可以直接提高设计性同时保持已经表现良好的区域不变。利用一个带有残基级别注释的定制数据集，我们通过ResiDPO微调LigandMPNN，得到EnhancedMPNN，该模型在一项具有挑战性的酶设计基准测试中，体外设计成功率几乎提高了三倍（从6.56%提高到17.57%）。 

---
# Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation 

**Title (ZH)**: 大规模语言模型在持续预训练下的语言适应能力涌现 

**Authors**: Ahmed Elhady, Eneko Agirre, Mikel Artetxe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00288)  

**Abstract**: Continued pretraining (CPT) is a popular approach to adapt existing large language models (LLMs) to new languages. When doing so, it is common practice to include a portion of English data in the mixture, but its role has not been carefully studied to date. In this work, we show that including English does not impact validation perplexity, yet it is critical for the emergence of downstream capabilities in the target language. We introduce a language-agnostic benchmark for in-context learning (ICL), which reveals catastrophic forgetting early on CPT when English is not included. This in turn damages the ability of the model to generalize to downstream prompts in the target language as measured by perplexity, even if it does not manifest in terms of accuracy until later in training, and can be tied to a big shift in the model parameters. Based on these insights, we introduce curriculum learning and exponential moving average (EMA) of weights as effective alternatives to mitigate the need for English. All in all, our work sheds light into the dynamics by which emergent abilities arise when doing CPT for language adaptation, and can serve as a foundation to design more effective methods in the future. 

**Abstract (ZH)**: 持续预训练（CPT）是将现有大型语言模型（LLMs）适配到新语言的一种流行方法。尽管通常会在混合数据中包含一部分英语数据，但其作用尚未得到仔细研究。在本工作中，我们表明包含英语不会影响验证困惑度，但它对于目标语言下游能力的出现至关重要。我们引入了一种跨语言的在上下文学习基准测试，该基准测试揭示了在CPT过程中不包含英语时早期出现的灾难性遗忘。这反过来损害了模型在目标语言中对下游提示的泛化能力，即使困惑度在训练后期才表现出准确性下降，这与模型参数的大幅变化有关。基于这些洞见，我们引入了分阶段学习和权重的指数移动平均（EMA）作为减少对英语依赖的有效替代方法。总体而言，我们的工作揭示了在进行语言适配的持续预训练过程中涌现能力的动态机制，可以为未来设计更有效的方法奠定基础。 

---
# Entropic Risk Optimization in Discounted MDPs: Sample Complexity Bounds with a Generative Model 

**Title (ZH)**: 带生成模型的折现MDP中的熵风险优化：样本复杂性界 

**Authors**: Oliver Mortensen, Mohammad Sadegh Talebi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00286)  

**Abstract**: In this paper we analyze the sample complexities of learning the optimal state-action value function $Q^*$ and an optimal policy $\pi^*$ in a discounted Markov decision process (MDP) where the agent has recursive entropic risk-preferences with risk-parameter $\beta\neq 0$ and where a generative model of the MDP is available. We provide and analyze a simple model based approach which we call model-based risk-sensitive $Q$-value-iteration (MB-RS-QVI) which leads to $(\epsilon,\delta)$-PAC-bounds on $\|Q^*-Q^k\|$, and $\|V^*-V^{\pi_k}\|$ where $Q_k$ is the output of MB-RS-QVI after k iterations and $\pi_k$ is the greedy policy with respect to $Q_k$. Both PAC-bounds have exponential dependence on the effective horizon $\frac{1}{1-\gamma}$ and the strength of this dependence grows with the learners risk-sensitivity $|\beta|$. We also provide two lower bounds which shows that exponential dependence on $|\beta|\frac{1}{1-\gamma}$ is unavoidable in both cases. The lower bounds reveal that the PAC-bounds are both tight in $\varepsilon$ and $\delta$ and that the PAC-bound on $Q$-learning is tight in the number of actions $A$, and that the PAC-bound on policy-learning is nearly tight in $A$. 

**Abstract (ZH)**: 在递归熵风险偏好下折扣马尔可夫决策过程中的最优状态动作值函数和最优策略的学习样本复杂性分析：基于模型的风险敏感$Q$-值迭代及其$(\varepsilon,\delta)$-PAC边界分析 

---
# Adversarial Threat Vectors and Risk Mitigation for Retrieval-Augmented Generation Systems 

**Title (ZH)**: 检索增强生成系统中的对抗威胁向量与风险缓解 

**Authors**: Chris M. Ward, Josh Harguess  

**Link**: [PDF](https://arxiv.org/pdf/2506.00281)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems, which integrate Large Language Models (LLMs) with external knowledge sources, are vulnerable to a range of adversarial attack vectors. This paper examines the importance of RAG systems through recent industry adoption trends and identifies the prominent attack vectors for RAG: prompt injection, data poisoning, and adversarial query manipulation. We analyze these threats under risk management lens, and propose robust prioritized control list that includes risk-mitigating actions like input validation, adversarial training, and real-time monitoring. 

**Abstract (ZH)**: RAG系统中的 adversarial攻击向量及其风险管理策略：基于输入验证、对抗训练和实时监控的对策清单 

---
# Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings 

**Title (ZH)**: 基于多语言布林-olds嵌入的分层级联新闻文章聚类 

**Authors**: Hans W. A. Hanley, Zakir Durumeric  

**Link**: [PDF](https://arxiv.org/pdf/2506.00277)  

**Abstract**: Contextual large language model embeddings are increasingly utilized for topic modeling and clustering. However, current methods often scale poorly, rely on opaque similarity metrics, and struggle in multilingual settings. In this work, we present a novel, scalable, interpretable, hierarchical, and multilingual approach to clustering news articles and social media data. To do this, we first train multilingual Matryoshka embeddings that can determine story similarity at varying levels of granularity based on which subset of the dimensions of the embeddings is examined. This embedding model achieves state-of-the-art performance on the SemEval 2022 Task 8 test dataset (Pearson $\rho$ = 0.816). Once trained, we develop an efficient hierarchical clustering algorithm that leverages the hierarchical nature of Matryoshka embeddings to identify unique news stories, narratives, and themes. We conclude by illustrating how our approach can identify and cluster stories, narratives, and overarching themes within real-world news datasets. 

**Abstract (ZH)**: 基于上下文的大型语言模型嵌入在主题建模和聚类中的应用越来越广泛。然而，当前方法往往扩展性差，依赖于不透明的相似性度量，并在多语言环境中表现不佳。在此工作中，我们提出了一种新的、可扩展的、可解释的、层次化的和多语言的新闻文章和社会媒体数据聚类方法。为此，我们首先训练多语言Matryoshka嵌入，可以根据检查嵌入的维度子集来确定故事在不同粒度水平上的相似性。该嵌入模型在SemEval 2022 Task 8测试数据集上达到了最先进性能（皮尔逊相关系数ρ=0.816）。训练完成后，我们开发了一种高效的时间层次聚类算法，利用Matryoshka嵌入的层次结构特征来识别独特的新闻故事、叙述和主题。最后，我们通过实例展示了我们的方法如何在实际新闻数据集中识别和聚类故事、叙述和主题。 

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
# Designing AI Tools for Clinical Care Teams to Support Serious Illness Conversations with Older Adults in the Emergency Department 

**Title (ZH)**: 设计AI工具以支持在急诊部门与老年人讨论严重疾病的相关临床护理团队 

**Authors**: Menglin Zhao, Zhuorui Yong, Ruijia Guan, Kai-Wei Chang, Adrian Haimovich, Kei Ouchi, Timothy Bickmore, Bingsheng Yao, Dakuo Wang, Smit Desai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00241)  

**Abstract**: Serious illness conversations (SICs), discussions between clinical care teams and patients with serious, life-limiting illnesses about their values, goals, and care preferences, are critical for patient-centered care. Without these conversations, patients often receive aggressive interventions that may not align with their goals. Clinical care teams face significant barriers when conducting serious illness conversations with older adult patients in Emergency Department (ED) settings, where most older adult patients lack documented treatment goals. To understand current practices and identify AI support opportunities, we conducted interviews with two domain experts and nine ED clinical care team members. Through thematic analysis, we characterized a four-phase serious illness conversation workflow (identification, preparation, conduction, documentation) and identified key needs and challenges at each stage. Clinical care teams struggle with fragmented EHR data access, time constraints, emotional preparation demands, and documentation burdens. While participants expressed interest in AI tools for information synthesis, conversational support, and automated documentation, they emphasized preserving human connection and clinical autonomy. We present design guidelines for AI tools supporting SIC workflows that fit within existing clinical practices. This work contributes empirical understanding of ED-based serious illness conversations and provides design considerations for AI in high-stakes clinical environments. 

**Abstract (ZH)**: 严重疾病对话（SICs）：临床护理团队与患有严重和生命限制性疾病患者的关于其价值观、目标和护理偏好的讨论对于以患者为中心的护理至关重要。没有这些讨论，患者往往会接受不符合其目标的积极干预措施。在急诊部门（ED）中，由于大多数老年患者的治疗目标缺乏记录，临床护理团队在与老年患者进行严重疾病对话时面临着巨大的障碍。为了了解当前的实践并识别AI支持的机会，我们与两位领域专家和九名急诊临床护理团队成员进行了访谈。通过主题分析，我们描述了一个四阶段的严重疾病对话工作流程（识别、准备、执行、记录），并在每个阶段识别了关键需求和挑战。临床护理团队在访问碎片化的电子健康记录数据、时间限制、情绪准备需求以及记录负担方面遇到了困难。尽管参与者对用于信息综合、对话支持和自动记录的AI工具表示出了兴趣，但他们强调保持人的连接和临床自主权的重要性。我们提出了适应现有临床实践的AI工具支持严重疾病对话工作流程的设计指南。这项工作为急诊部门基于的严重疾病对话提供了实证理解，并为AI在高风险临床环境中的设计考虑提供了参考。 

---
# Localized LoRA: A Structured Low-Rank Approximation for Efficient Fine-Tuning 

**Title (ZH)**: 局部LoRA：一种结构化低秩逼近方法以实现高效的微调 

**Authors**: Babak Barazandeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00236)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, offer compact and effective alternatives to full model fine-tuning by introducing low-rank updates to pretrained weights. However, most existing approaches rely on global low-rank structures, which can overlook spatial patterns spread across the parameter space. In this work, we propose Localized LoRA, a generalized framework that models weight updates as a composition of low-rank matrices applied to structured blocks of the weight matrix. This formulation enables dense, localized updates throughout the parameter space-without increasing the total number of trainable parameters. We provide a formal comparison between global, diagonal-local, and fully localized low-rank approximations, and show that our method consistently achieves lower approximation error under matched parameter budgets. Experiments on both synthetic and practical settings demonstrate that Localized LoRA offers a more expressive and adaptable alternative to existing methods, enabling efficient fine-tuning with improved performance. 

**Abstract (ZH)**: 局部LoRA：一种建模权重更新的通用框架，通过在权重矩阵的结构块上应用低秩矩阵实现密集的局部更新 

---
# Ctrl-Crash: Controllable Diffusion for Realistic Car Crashes 

**Title (ZH)**: 控制造车撞击：可控扩散模型生成真实的汽车碰撞事故 

**Authors**: Anthony Gosselin, Ge Ya Luo, Luis Lara, Florian Golemo, Derek Nowrouzezahrai, Liam Paull, Alexia Jolicoeur-Martineau, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00227)  

**Abstract**: Video diffusion techniques have advanced significantly in recent years; however, they struggle to generate realistic imagery of car crashes due to the scarcity of accident events in most driving datasets. Improving traffic safety requires realistic and controllable accident simulations. To tackle the problem, we propose Ctrl-Crash, a controllable car crash video generation model that conditions on signals such as bounding boxes, crash types, and an initial image frame. Our approach enables counterfactual scenario generation where minor variations in input can lead to dramatically different crash outcomes. To support fine-grained control at inference time, we leverage classifier-free guidance with independently tunable scales for each conditioning signal. Ctrl-Crash achieves state-of-the-art performance across quantitative video quality metrics (e.g., FVD and JEDi) and qualitative measurements based on a human-evaluation of physical realism and video quality compared to prior diffusion-based methods. 

**Abstract (ZH)**: 控制车祸视频生成模型Ctrl-Crash：基于边界框、碰撞类型和初始图像帧的可控车祸模拟算法 

---
# Diff-SPORT: Diffusion-based Sensor Placement Optimization and Reconstruction of Turbulent flows in urban environments 

**Title (ZH)**: 基于扩散的传感器-placement优化与城市环境湍流流动的重建 

**Authors**: Abhijeet Vishwasrao, Sai Bharath Chandra Gutha, Andres Cremades, Klas Wijk, Aakash Patil, Catherine Gorle, Beverley J McKeon, Hossein Azizpour, Ricardo Vinuesa  

**Link**: [PDF](https://arxiv.org/pdf/2506.00214)  

**Abstract**: Rapid urbanization demands accurate and efficient monitoring of turbulent wind patterns to support air quality, climate resilience and infrastructure design. Traditional sparse reconstruction and sensor placement strategies face major accuracy degradations under practical constraints. Here, we introduce Diff-SPORT, a diffusion-based framework for high-fidelity flow reconstruction and optimal sensor placement in urban environments. Diff-SPORT combines a generative diffusion model with a maximum a posteriori (MAP) inference scheme and a Shapley-value attribution framework to propose a scalable and interpretable solution. Compared to traditional numerical methods, Diff-SPORT achieves significant speedups while maintaining both statistical and instantaneous flow fidelity. Our approach offers a modular, zero-shot alternative to retraining-intensive strategies, supporting fast and reliable urban flow monitoring under extreme sparsity. Diff-SPORT paves the way for integrating generative modeling and explainability in sustainable urban intelligence. 

**Abstract (ZH)**: 快速城市化需求精确和高效的湍流风模式监测以支持空气质量、气候韧性和基础设施设计。传统的稀疏重构和传感器布设策略在实际约束下面临重大准确度下降。我们引入了基于扩散的Diff-SPORT框架，用于城市环境中高保真流场重构和最优传感器布设。Diff-SPORT结合了生成型扩散模型、最大后验概率（MAP）推断方案和Shapley值归因框架，提出了一个可扩展且可解释的解决方案。与传统数值方法相比，Diff-SPORT在保持统计和瞬时流场保真度的同时实现了显著的加速。我们的方法为密集重训练策略提供了模块化的零样本替代方案，在极端稀疏条件下支持快速可靠的都市流场监测。Diff-SPORT为在可持续智慧城市中整合生成型建模和可解释性奠定了基础。 

---
# REIC: RAG-Enhanced Intent Classification at Scale 

**Title (ZH)**: REIC: RAG增强的规模化的意图分类 

**Authors**: Ziji Zhang, Michael Yang, Zhiyu Chen, Yingying Zhuang, Shu-Ting Pi, Qun Liu, Rajashekar Maragoud, Vy Nguyen, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00210)  

**Abstract**: Accurate intent classification is critical for efficient routing in customer service, ensuring customers are connected with the most suitable agents while reducing handling times and operational costs. However, as companies expand their product lines, intent classification faces scalability challenges due to the increasing number of intents and variations in taxonomy across different verticals. In this paper, we introduce REIC, a Retrieval-augmented generation Enhanced Intent Classification approach, which addresses these challenges effectively. REIC leverages retrieval-augmented generation (RAG) to dynamically incorporate relevant knowledge, enabling precise classification without the need for frequent retraining. Through extensive experiments on real-world datasets, we demonstrate that REIC outperforms traditional fine-tuning, zero-shot, and few-shot methods in large-scale customer service settings. Our results highlight its effectiveness in both in-domain and out-of-domain scenarios, demonstrating its potential for real-world deployment in adaptive and large-scale intent classification systems. 

**Abstract (ZH)**: 准确的意图分类对于客户服务中的高效路由至关重要，确保客户能够与最合适的代理人员对接，同时减少处理时间和运营成本。然而，随着公司产品线的扩展，由于不同垂直领域意图数量的增加和分类 taxonomy 的变化，意图分类面临可扩展性挑战。本文介绍了一种名为 REIC 的检索增强生成增强意图分类方法，该方法有效地应对了这些挑战。REIC 利用检索增强生成 (RAG) 动态地整合相关知识，实现精确分类，无需频繁重新训练。通过在真实数据集上的广泛实验，我们证明了 REIC 在大规模客户服务场景中优于传统的微调、零样本和少样本方法。我们的结果强调了其在领域内和领域外场景中的有效性，展示了其在适应性和大规模意图分类系统中实际部署的潜力。 

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
# Heterogeneous Graph Backdoor Attack 

**Title (ZH)**: 异质图后门攻击 

**Authors**: Jiawei Chen, Lusi Li, Daniel Takabi, Masha Sosonkina, Rui Ning  

**Link**: [PDF](https://arxiv.org/pdf/2506.00191)  

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) excel in modeling complex, multi-typed relationships across diverse domains, yet their vulnerability to backdoor attacks remains unexplored. To address this gap, we conduct the first investigation into the susceptibility of HGNNs to existing graph backdoor attacks, revealing three critical issues: (1) high attack budget required for effective backdoor injection, (2) inefficient and unreliable backdoor activation, and (3) inaccurate attack effectiveness evaluation. To tackle these issues, we propose the Heterogeneous Graph Backdoor Attack (HGBA), the first backdoor attack specifically designed for HGNNs, introducing a novel relation-based trigger mechanism that establishes specific connections between a strategically selected trigger node and poisoned nodes via the backdoor metapath. HGBA achieves efficient and stealthy backdoor injection with minimal structural modifications and supports easy backdoor activation through two flexible strategies: Self-Node Attack and Indiscriminate Attack. Additionally, we improve the ASR measurement protocol, enabling a more accurate assessment of attack effectiveness. Extensive experiments demonstrate that HGBA far surpasses multiple state-of-the-art graph backdoor attacks in black-box settings, efficiently attacking HGNNs with low attack budgets. Ablation studies show that the strength of HBGA benefits from our trigger node selection method and backdoor metapath selection strategy. In addition, HGBA shows superior robustness against node feature perturbations and multiple types of existing graph backdoor defense mechanisms. Finally, extension experiments demonstrate that the relation-based trigger mechanism can effectively extend to tasks in homogeneous graph scenarios, thereby posing severe threats to broader security-critical domains. 

**Abstract (ZH)**: 异质图神经网络的后门攻击：异质图后门攻击（HGBA）及其应用 

---
# Pushing the Limits of Beam Search Decoding for Transducer-based ASR models 

**Title (ZH)**: 基于发射机的ASR模型中极限拓展的束搜索解码方法 

**Authors**: Lilit Grigoryan, Vladimir Bataev, Andrei Andrusenko, Hainan Xu, Vitaly Lavrukhin, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2506.00185)  

**Abstract**: Transducer models have emerged as a promising choice for end-to-end ASR systems, offering a balanced trade-off between recognition accuracy, streaming capabilities, and inference speed in greedy decoding. However, beam search significantly slows down Transducers due to repeated evaluations of key network components, limiting practical applications. This paper introduces a universal method to accelerate beam search for Transducers, enabling the implementation of two optimized algorithms: ALSD++ and AES++. The proposed method utilizes batch operations, a tree-based hypothesis structure, novel blank scoring for enhanced shallow fusion, and CUDA graph execution for efficient GPU inference. This narrows the speed gap between beam and greedy modes to only 10-20% for the whole system, achieves 14-30% relative improvement in WER compared to greedy decoding, and improves shallow fusion for low-resource up to 11% compared to existing implementations. All the algorithms are open sourced. 

**Abstract (ZH)**: 递归神经网络模型已成为端到端ASR系统的有希望的选择，能够在贪婪解码中提供识别准确性、流式传输能力和推断速度之间的平衡trade-off。然而，束搜索由于反复评估关键网络组件而显著减慢递归神经网络模型的速度，限制了其实用应用。本文介绍了一种通用方法来加速递归神经网络模型的束搜索，实现了两个优化算法：ALSD++和AES++。所提出的方法利用批量操作、基于树的假设结构、增强浅融合的新空白评分以及CUDA图执行高效的GPU推断。这种方法将束搜索和贪婪模式之间的速度差距缩减至整个系统性能的10-20%，相比贪婪解码实现了14-30%的相对WER改进，并且相比现有实现对于低资源情况下的浅融合改进了11%。所有算法均已开源。 

---
# Accountability Attribution: Tracing Model Behavior to Training Processes 

**Title (ZH)**: 行为问责制归属：追踪模型行为至训练过程 

**Authors**: Shichang Zhang, Hongzhe Du, Karim Saraipour, Jiaqi W. Ma, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2506.00175)  

**Abstract**: Modern AI development pipelines often involve multiple stages-pretraining, fine-tuning rounds, and subsequent adaptation or alignment-with numerous model update steps within each stage. This raises a critical question of accountability: when a deployed model succeeds or fails, which stage is responsible, and to what extent? We pose the problem of accountability attribution, which aims to trace model behavior back to specific stages of the training process. To address this, we propose a general framework that answers counterfactual questions about stage effects: how would the model behavior have changed if the updates from a training stage had not been executed?. Within this framework, we introduce estimators based on first-order approximations that efficiently quantify the stage effects without retraining. Our estimators account for both the training data and key aspects of optimization dynamics, including learning rate schedules, momentum, and weight decay. Empirically, we demonstrate that our approach identifies training stages accountable for specific behaviors, offering a practical tool for model analysis and a step toward more accountable AI development. 

**Abstract (ZH)**: 现代AI开发管道中的责任归属问题：从训练过程的具体阶段追溯模型行为 

---
# Disentangled Safety Adapters Enable Efficient Guardrails and Flexible Inference-Time Alignment 

**Title (ZH)**: 解耦的安全适配器实现高效的安全保障和地区灵活的推理时对齐 

**Authors**: Kundan Krishna, Joseph Y Cheng, Charles Maalouf, Leon A Gatys  

**Link**: [PDF](https://arxiv.org/pdf/2506.00166)  

**Abstract**: Existing paradigms for ensuring AI safety, such as guardrail models and alignment training, often compromise either inference efficiency or development flexibility. We introduce Disentangled Safety Adapters (DSA), a novel framework addressing these challenges by decoupling safety-specific computations from a task-optimized base model. DSA utilizes lightweight adapters that leverage the base model's internal representations, enabling diverse and flexible safety functionalities with minimal impact on inference cost. Empirically, DSA-based safety guardrails substantially outperform comparably sized standalone models, notably improving hallucination detection (0.88 vs. 0.61 AUC on Summedits) and also excelling at classifying hate speech (0.98 vs. 0.92 on ToxiGen) and unsafe model inputs and responses (0.93 vs. 0.90 on AEGIS2.0 & BeaverTails). Furthermore, DSA-based safety alignment allows dynamic, inference-time adjustment of alignment strength and a fine-grained trade-off between instruction following performance and model safety. Importantly, combining the DSA safety guardrail with DSA safety alignment facilitates context-dependent alignment strength, boosting safety on StrongReject by 93% while maintaining 98% performance on MTBench -- a total reduction in alignment tax of 8 percentage points compared to standard safety alignment fine-tuning. Overall, DSA presents a promising path towards more modular, efficient, and adaptable AI safety and alignment. 

**Abstract (ZH)**: 解耦安全适配器：一种新型的AI安全与对齐框架 

---
# Detection of Endangered Deer Species Using UAV Imagery: A Comparative Study Between Efficient Deep Learning Approaches 

**Title (ZH)**: 使用无人机影像检测濒危鹿科物种：基于高效深度学习方法的比较研究 

**Authors**: Agustín Roca, Gastón Castro, Gabriel Torre, Leonardo J. Colombo, Ignacio Mas, Javier Pereira, Juan I. Giribet  

**Link**: [PDF](https://arxiv.org/pdf/2506.00154)  

**Abstract**: This study compares the performance of state-of-the-art neural networks including variants of the YOLOv11 and RT-DETR models for detecting marsh deer in UAV imagery, in scenarios where specimens occupy a very small portion of the image and are occluded by vegetation. We extend previous analysis adding precise segmentation masks for our datasets enabling a fine-grained training of a YOLO model with a segmentation head included. Experimental results show the effectiveness of incorporating the segmentation head achieving superior detection performance. This work contributes valuable insights for improving UAV-based wildlife monitoring and conservation strategies through scalable and accurate AI-driven detection systems. 

**Abstract (ZH)**: 本研究比较了包括YOLOv1和RT-DETR变种在内的最先进神经网络在无人机图像中检测沼泽 Deer 的性能，特别是在动物占据图像很小部分且被植被遮挡的情况下。我们扩展了先前的分析，为我们的数据集添加了精确的分割掩码，使包含分割头部的YOLO模型能够进行精细粒度的训练。实验结果表明，引入分割头部的有效性，从而实现检测性能的提升。本文通过可扩展且准确的基于AI的检测系统，为改进无人机辅助的野生动物监控和保护策略提供了宝贵见解。 

---
# Supporting architecture evaluation for ATAM scenarios with LLMs 

**Title (ZH)**: 使用大语言模型支持ATAM场景下的架构评估 

**Authors**: Rafael Capilla, J. Andrés Díaz-Pace, Yamid Ramírez, Jennifer Pérez, Vanessa Rodríguez-Horcajo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00150)  

**Abstract**: Architecture evaluation methods have long been used to evaluate software designs. Several evaluation methods have been proposed and used to analyze tradeoffs between different quality attributes. Having competing qualities leads to conflicts for selecting which quality-attribute scenarios are the most suitable ones that an architecture should tackle and for prioritizing the scenarios required by the stakeholders. In this context, architecture evaluation is carried out manually, often involving long brainstorming sessions to decide which are the most adequate quality scenarios. To reduce this effort and make the assessment and selection of scenarios more efficient, we suggest the usage of LLMs to partially automate evaluation activities. As a first step to validate this hypothesis, this work studies MS Copilot as an LLM tool to analyze quality scenarios suggested by students in a software architecture course and compares the students' results with the assessment provided by the LLM. Our initial study reveals that the LLM produces in most cases better and more accurate results regarding the risks, sensitivity points and tradeoff analysis of the quality scenarios. Overall, the use of generative AI has the potential to partially automate and support the architecture evaluation tasks, improving the human decision-making process. 

**Abstract (ZH)**: 使用生成式AI部分自动化软件架构评估任务的研究 

---
# Autonomous Behavior and Whole-Brain Dynamics Emerge in Embodied Zebrafish Agents with Model-based Intrinsic Motivation 

**Title (ZH)**: 基于模型的内在动机驱动的 embodied 斑马鱼代理中涌现自主行为与全脑动态 

**Authors**: Reece Keller, Alyn Tornell, Felix Pei, Xaq Pitkow, Leo Kozachkov, Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00138)  

**Abstract**: Autonomy is a hallmark of animal intelligence, enabling adaptive and intelligent behavior in complex environments without relying on external reward or task structure. Existing reinforcement learning approaches to exploration in sparse reward and reward-free environments, including class of methods known as intrinsic motivation, exhibit inconsistent exploration patterns and thus fail to produce robust autonomous behaviors observed in animals. Moreover, systems neuroscience has largely overlooked the neural basis of autonomy, focusing instead on experimental paradigms where animals are motivated by external reward rather than engaging in unconstrained, naturalistic and task-independent behavior. To bridge these gaps, we introduce a novel model-based intrinsic drive explicitly designed to capture robust autonomous exploration observed in animals. Our method (3M-Progress) motivates naturalistic behavior by tracking divergence between the agent's current world model and an ethological prior. We demonstrate that artificial embodied agents trained with 3M-Progress capture the explainable variance in behavioral patterns and whole-brain neural-glial dynamics recorded from autonomously-behaving larval zebrafish, introducing the first goal-driven, population-level model of neural-glial computation. Our findings establish a computational framework connecting model-based intrinsic motivation to naturalistic behavior, providing a foundation for building artificial agents with animal-like autonomy. 

**Abstract (ZH)**: 自主性是动物智能的标志，使动物能够在复杂的环境中表现出适应性和智能行为，而不依赖于外部奖励或任务结构。现有的稀疏奖励和无奖励环境中探索的强化学习方法，包括内在动机这类方法，表现出不一致的探索模式，因此无法产生在动物中观察到的稳健自主行为。此外，系统神经科学大多忽略了自主性的神经基础，而是将焦点放在动物被外部奖励驱动的实验范式上，而不是自然无约束的任务独立行为。为填补这些空白，我们引入了一个新的基于模型的内在驱动力，明确设计用于捕捉动物中观察到的稳健自主探索。我们的方法（3M-Progress）通过追踪智能体当前世界模型与生态学先验之间的差异来激励自然行为。我们证明，使用3M-Progress训练的合成躯体化代理能够捕捉自主行为的动态模式和来自自主行为的涡偶鱼幼虫的全脑神经-胶质动力学中的可解释变方，首次提出了目标驱动的神经-胶质计算的群体级模型。我们的发现建立了将基于模型的内在动机与自然行为联系起来的计算框架，为构建具有类似动物自主性的智能代理提供了基础。 

---
# Spurious Correlations and Beyond: Understanding and Mitigating Shortcut Learning in SDOH Extraction with Large Language Models 

**Title (ZH)**: 虚假相关性及其超越：理解并缓解大规模语言模型在社会决定因素提取中的捷径学习问题 

**Authors**: Fardin Ahsan Sakib, Ziwei Zhu, Karen Trister Grace, Meliha Yetisgen, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2506.00134)  

**Abstract**: Social determinants of health (SDOH) extraction from clinical text is critical for downstream healthcare analytics. Although large language models (LLMs) have shown promise, they may rely on superficial cues leading to spurious predictions. Using the MIMIC portion of the SHAC (Social History Annotation Corpus) dataset and focusing on drug status extraction as a case study, we demonstrate that mentions of alcohol or smoking can falsely induce models to predict current/past drug use where none is present, while also uncovering concerning gender disparities in model performance. We further evaluate mitigation strategies - such as prompt engineering and chain-of-thought reasoning - to reduce these false positives, providing insights into enhancing LLM reliability in health domains. 

**Abstract (ZH)**: 社会决定因素对健康的影响从临床文本中提取对于下游医疗数据分析至关重要。尽管大型语言模型显示出潜力，但它们可能会依赖于表面线索导致虚假预测。通过使用SHAC数据集中MIMIC部分和以药物状态提取为例进行研究，我们证明了提及酒精或吸烟可能会误导模型预测不存在的当前/过去用药情况，同时还揭示了模型性能中的性别差异问题。进一步评估了诸如提示工程和链式思考推理等缓解策略，以减少这些误报，为增强医疗领域大型语言模型的可靠性提供了见解。 

---
# A Reinforcement Learning-Based Telematic Routing Protocol for the Internet of Underwater Things 

**Title (ZH)**: 基于强化学习的物联网水下节点路由协议 

**Authors**: Mohammadhossein Homaei, Mehran Tarif, Agustin Di Bartolo, Oscar Mogollon Gutierrez, Mar Avila  

**Link**: [PDF](https://arxiv.org/pdf/2506.00133)  

**Abstract**: The Internet of Underwater Things (IoUT) faces major challenges such as low bandwidth, high latency, mobility, and limited energy resources. Traditional routing protocols like RPL, which were designed for land-based networks, do not perform well in these underwater conditions. This paper introduces RL-RPL-UA, a new routing protocol that uses reinforcement learning to improve performance in underwater environments. Each node includes a lightweight RL agent that selects the best parent node based on local information such as packet delivery ratio, buffer level, link quality, and remaining energy. RL-RPL-UA keeps full compatibility with standard RPL messages and adds a dynamic objective function to support real-time decision-making. Simulations using Aqua-Sim show that RL-RPL-UA increases packet delivery by up to 9.2%, reduces energy use per packet by 14.8%, and extends network lifetime by 80 seconds compared to traditional methods. These results suggest that RL-RPL-UA is a promising and energy-efficient routing solution for underwater networks. 

**Abstract (ZH)**: 水下事物流动的互联网（IoUT）面临低带宽、高延迟、移动性和有限的能量资源等重大挑战。传统的路由协议如RPL，适用于陆地网络，在水下环境中表现不佳。本文介绍了一种新的基于强化学习的路由协议RL-RPL-UA，以提高水下环境中的性能。每个节点包含一个轻量级的RL代理，基于诸如数据包投递率、缓冲区水平、链路质量和剩余能量等本地信息来选择最优父节点。RL-RPL-UA与标准RPL消息保持完全兼容，并增加了一个动态目标函数以支持实时决策。使用Aqua-Sim进行的仿真表明，与传统方法相比，RL-RPL-UA可将数据包投递率提高9.2%，每数据包能量使用减少14.8%，网络寿命延长80秒。这些结果表明，RL-RPL-UA是一种有前景的能量高效的水下网络路由解决方案。 

---
# Adapting Offline Reinforcement Learning with Online Delays 

**Title (ZH)**: 具在线延迟的离线强化学习适应 

**Authors**: Simon Sinong Zhan, Qingyuan Wu, Frank Yang, Xiangyu Shi, Chao Huang, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00131)  

**Abstract**: Offline-to-online deployment of reinforcement-learning (RL) agents must bridge two gaps: (1) the sim-to-real gap, where real systems add latency and other imperfections not present in simulation, and (2) the interaction gap, where policies trained purely offline face out-of-distribution states during online execution because gathering new interaction data is costly or risky. Agents therefore have to generalize from static, delay-free datasets to dynamic, delay-prone environments. Standard offline RL learns from delay-free logs yet must act under delays that break the Markov assumption and hurt performance. We introduce DT-CORL (Delay-Transformer belief policy Constrained Offline RL), an offline-RL framework built to cope with delayed dynamics at deployment. DT-CORL (i) produces delay-robust actions with a transformer-based belief predictor even though it never sees delayed observations during training, and (ii) is markedly more sample-efficient than naïve history-augmentation baselines. Experiments on D4RL benchmarks with several delay settings show that DT-CORL consistently outperforms both history-augmentation and vanilla belief-based methods, narrowing the sim-to-real latency gap while preserving data efficiency. 

**Abstract (ZH)**: Offline-to-online 部署中的延迟鲁棒 Offline RL 框架: DT-CORL 

---
# Gated Multimodal Graph Learning for Personalized Recommendation 

**Title (ZH)**: 基于门控多模态图学习的个性化推荐 

**Authors**: Sibei Liu, Yuanzhe Zhang, Xiang Li, Yunbo Liu, Chengwei Feng, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00107)  

**Abstract**: Multimodal recommendation has emerged as a promising solution to alleviate the cold-start and sparsity problems in collaborative filtering by incorporating rich content information, such as product images and textual descriptions. However, effectively integrating heterogeneous modalities into a unified recommendation framework remains a challenge. Existing approaches often rely on fixed fusion strategies or complex architectures , which may fail to adapt to modality quality variance or introduce unnecessary computational overhead.
In this work, we propose RLMultimodalRec, a lightweight and modular recommendation framework that combines graph-based user modeling with adaptive multimodal item encoding. The model employs a gated fusion module to dynamically balance the contribution of visual and textual modalities, enabling fine-grained and content-aware item representations. Meanwhile, a two-layer LightGCN encoder captures high-order collaborative signals by propagating embeddings over the user-item interaction graph without relying on nonlinear transformations.
We evaluate our model on a real-world dataset from the Amazon product domain. Experimental results demonstrate that RLMultimodalRec consistently outperforms several competitive baselines, including collaborative filtering, visual-aware, and multimodal GNN-based methods. The proposed approach achieves significant improvements in top-K recommendation metrics while maintaining scalability and interpretability, making it suitable for practical deployment. 

**Abstract (ZH)**: 基于图的用户建模与自适应多模态项编码的轻量级多模态推荐框架RLMultimodalRec 

---
# Children's Voice Privacy: First Steps And Emerging Challenges 

**Title (ZH)**: 儿童语音隐私：初步探讨与新兴挑战 

**Authors**: Ajinkya Kulkarni, Francisco Teixeira, Enno Hermann, Thomas Rolland, Isabel Trancoso, Mathew Magimai Doss  

**Link**: [PDF](https://arxiv.org/pdf/2506.00100)  

**Abstract**: Children are one of the most under-represented groups in speech technologies, as well as one of the most vulnerable in terms of privacy. Despite this, anonymization techniques targeting this population have received little attention. In this study, we seek to bridge this gap, and establish a baseline for the use of voice anonymization techniques designed for adult speech when applied to children's voices. Such an evaluation is essential, as children's speech presents a distinct set of challenges when compared to that of adults. This study comprises three children's datasets, six anonymization methods, and objective and subjective utility metrics for evaluation. Our results show that existing systems for adults are still able to protect children's voice privacy, but suffer from much higher utility degradation. In addition, our subjective study displays the challenges of automatic evaluation methods for speech quality in children's speech, highlighting the need for further research. 

**Abstract (ZH)**: 儿童是语言技术中最未被充分代表的群体之一，也是隐私方面最脆弱的群体之一。尽管如此，针对这一人群的匿名化技术尚未得到广泛关注。本研究旨在弥补这一差距，并建立一个基准，评估将旨在成年语音的匿名化技术应用于儿童语音时的有效性。由于儿童的语音与成人存在显著差异，这种评估至关重要。本研究包括三个儿童数据集、六种匿名化方法以及客观和主观效益度量标准进行评估。我们的结果显示，现有的成人系统仍然能够保护儿童的语音隐私，但会遭受更严重的效益降级。此外，我们的主观研究揭示了自动评估方法在儿童语音质量评估中面临的挑战，强调了进一步研究的必要性。 

---
# PathGene: Benchmarking Driver Gene Mutations and Exon Prediction Using Multicenter Lung Cancer Histopathology Image Dataset 

**Title (ZH)**: PathGene：基于多中心肺癌组织学图像数据集的驱动基因突变和外显子预测基准评估 

**Authors**: Liangrui Pan, Qingchun Liang, Shen Zhao, Songqing Fan, Shaoliang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00096)  

**Abstract**: Accurately predicting gene mutations, mutation subtypes and their exons in lung cancer is critical for personalized treatment planning and prognostic assessment. Faced with regional disparities in medical resources and the high cost of genomic assays, using artificial intelligence to infer these mutations and exon variants from routine histopathology images could greatly facilitate precision therapy. Although some prior studies have shown that deep learning can accelerate the prediction of key gene mutations from lung cancer pathology slides, their performance remains suboptimal and has so far been limited mainly to early screening tasks. To address these limitations, we have assembled PathGene, which comprises histopathology images paired with next-generation sequencing reports from 1,576 patients at the Second Xiangya Hospital, Central South University, and 448 TCGA-LUAD patients. This multi-center dataset links whole-slide images to driver gene mutation status, mutation subtypes, exon, and tumor mutational burden (TMB) status, with the goal of leveraging pathology images to predict mutations, subtypes, exon locations, and TMB for early genetic screening and to advance precision oncology. Unlike existing datasets, we provide molecular-level information related to histopathology images in PathGene to facilitate the development of biomarker prediction models. We benchmarked 11 multiple-instance learning methods on PathGene for mutation, subtype, exon, and TMB prediction tasks. These experimental methods provide valuable alternatives for early genetic screening of lung cancer patients and assisting clinicians to quickly develop personalized precision targeted treatment plans for patients. Code and data are available at this https URL. 

**Abstract (ZH)**: 准确预测肺癌中的基因突变、突变亚型及其外显子对于个性化治疗规划和预后评估至关重要。面对医疗资源区域差异和基因组检测的高成本，使用人工智能从常规病理图像推断这些突变和外显子变异，可大大促进精准治疗。虽然一些前期研究显示深度学习可以在肺癌病理切片中加速关键基因突变的预测，但其性能仍然不佳，目前主要局限于早期筛查任务。为克服这些限制，我们构建了PathGene数据集，该数据集包含来自中南大学湘雅二医院和TCGA-LUAD的1,576名患者配对的病理图像和下一代测序报告，以及448名患者的病理图像。该多中心数据集将整个切片图像与驱动基因突变状态、突变亚型、外显子位置和肿瘤突变负担（TMB）状态相连，旨在利用病理图像进行早期遗传筛查并推进精准肿瘤学的发展。与现有数据集不同，PathGene提供了与病理图像相关的分子水平信息，以促进生物标志物预测模型的开发。我们在PathGene上 benchmark 了11种多实例学习方法，用于突变、亚型、外显子和TMB预测任务。这些实验方法提供了有价值的选择，用于肺癌患者的早期遗传筛查，帮助医生迅速制定针对患者的个性化精准靶向治疗方案。代码和数据可在以下链接获取。 

---
# ClinBench-HPB: A Clinical Benchmark for Evaluating LLMs in Hepato-Pancreato-Biliary Diseases 

**Title (ZH)**: ClinBench-HPB：肝胆胰疾病领域评估LLMs的临床基准 

**Authors**: Yuchong Li, Xiaojun Zeng, Chihua Fang, Jian Yang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00095)  

**Abstract**: Hepato-pancreato-biliary (HPB) disorders represent a global public health challenge due to their high morbidity and mortality. Although large language models (LLMs) have shown promising performance in general medical question-answering tasks, the current evaluation benchmarks are mostly derived from standardized examinations or manually designed questions, lacking HPB coverage and clinical cases. To address these issues, we systematically eatablish an HPB disease evaluation benchmark comprising 3,535 closed-ended multiple-choice questions and 337 open-ended real diagnosis cases, which encompasses all the 33 main categories and 465 subcategories of HPB diseases defined in the International Statistical Classification of Diseases, 10th Revision (ICD-10). The multiple-choice questions are curated from public datasets and synthesized data, and the clinical cases are collected from prestigious medical journals, case-sharing platforms, and collaborating hospitals. By evalauting commercial and open-source general and medical LLMs on our established benchmark, namely ClinBench-HBP, we find that while commercial LLMs perform competently on medical exam questions, they exhibit substantial performance degradation on HPB diagnosis tasks, especially on complex, inpatient clinical cases. Those medical LLMs also show limited generalizability to HPB diseases. Our results reveal the critical limitations of current LLMs in the domain of HPB diseases, underscoring the imperative need for future medical LLMs to handle real, complex clinical diagnostics rather than simple medical exam questions. The benchmark will be released at the homepage. 

**Abstract (ZH)**: 肝-胰-胆道（HPB）疾病评估基准：一种涵盖全球高发病率和死亡率疾病的综合性评估标准 

---
# Feeling Guilty Being a c(ai)borg: Navigating the Tensions Between Guilt and Empowerment in AI Use 

**Title (ZH)**: 成为半机械人时的内疚感：在AI使用中平衡内疚与赋能之间的张力 

**Authors**: Konstantin Aal, Tanja Aal, Vasil Navumau, David Unbehaun, Claudia Müller, Volker Wulf, Sarah Rüller  

**Link**: [PDF](https://arxiv.org/pdf/2506.00094)  

**Abstract**: This paper explores the emotional, ethical and practical dimensions of integrating Artificial Intelligence (AI) into personal and professional workflows, focusing on the concept of feeling guilty as a 'c(ai)borg' - a human augmented by AI. Inspired by Donna Haraway's Cyborg Manifesto, the study explores how AI challenges traditional notions of creativity, originality and intellectual labour. Using an autoethnographic approach, the authors reflect on their year-long experiences with AI tools, revealing a transition from initial guilt and reluctance to empowerment through skill-building and transparency. Key findings highlight the importance of basic academic skills, advanced AI literacy and honest engagement with AI results. The c(ai)borg vision advocates for a future where AI is openly embraced as a collaborative partner, fostering innovation and equity while addressing issues of access and agency. By reframing guilt as growth, the paper calls for a thoughtful and inclusive approach to AI integration. 

**Abstract (ZH)**: 本文探讨将人工智能（AI）整合到个人和职业工作流程中的情感、伦理和实践维度，重点关注作为“c(ai)borg”（人类增强的AI）时产生的罪恶感概念。受唐娜·哈拉维的《赛博格宣言》启发，研究探讨了AI对传统创造力、原创性和智力劳动观念的挑战。通过自传民族志的方法，作者反思了他们在一年中使用AI工具的经验，揭示了从最初的罪恶感和抵触到通过技能提升和透明度实现赋能的转变。关键发现强调了基础学术技能、高级AI素养以及坦诚面对AI结果的重要性。“c(ai)borg”愿景倡导一个AI被公开接受作为协作伙伴的未来，促进创新和公平，同时解决访问和代理问题。通过将罪恶感重新定位为成长的机会，本文呼吁采取深入和包容的方式整合AI。 

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
# SwitchLingua: The First Large-Scale Multilingual and Multi-Ethnic Code-Switching Dataset 

**Title (ZH)**: SwitchLingua: 首个多语言和多民族代码转换大型数据集 

**Authors**: Peng Xie, Xingyuan Liu, Tsz Wai Chan, Yequan Bie, Yangqiu Song, Yang Wang, Hao Chen, Kani Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00087)  

**Abstract**: Code-switching (CS) is the alternating use of two or more languages within a conversation or utterance, often influenced by social context and speaker identity. This linguistic phenomenon poses challenges for Automatic Speech Recognition (ASR) systems, which are typically designed for a single language and struggle to handle multilingual inputs. The growing global demand for multilingual applications, including Code-Switching ASR (CSASR), Text-to-Speech (CSTTS), and Cross-Lingual Information Retrieval (CLIR), highlights the inadequacy of existing monolingual datasets.
Although some code-switching datasets exist, most are limited to bilingual mixing within homogeneous ethnic groups, leaving a critical need for a large-scale, diverse benchmark akin to ImageNet in computer vision.
To bridge this gap, we introduce \textbf{LinguaMaster}, a multi-agent collaboration framework specifically designed for efficient and scalable multilingual data synthesis. Leveraging this framework, we curate \textbf{SwitchLingua}, the first large-scale multilingual and multi-ethnic code-switching dataset, including: (1) 420K CS textual samples across 12 languages, and (2) over 80 hours of audio recordings from 174 speakers representing 18 countries/regions and 63 racial/ethnic backgrounds, based on the textual data. This dataset captures rich linguistic and cultural diversity, offering a foundational resource for advancing multilingual and multicultural research. Furthermore, to address the issue that existing ASR evaluation metrics lack sensitivity to code-switching scenarios, we propose the \textbf{Semantic-Aware Error Rate (SAER)}, a novel evaluation metric that incorporates semantic information, providing a more accurate and context-aware assessment of system performance. 

**Abstract (ZH)**: 代码转换（CS）是指在对话或语句中交替使用两种或多种语言的现象，往往受到社会背景和说话人身份的影响。这一语言现象给自动语音识别（ASR）系统带来了挑战，这些系统通常仅设计用于单一语言并难以处理多语言输入。随着全球对多语言应用的需求增长，包括代码转换自动语音识别（CSASR）、文本到语音（CSTTS）和跨语言信息检索（CLIR）应用，突显了现有单一语言数据集的不足。

尽管存在一些代码转换数据集，但大多数数据集仅限于同质族裔群体内的双语混合，因此需要类似ImageNet的大规模、多元化基准数据集。

为弥补这一差距，我们引入了**LinguaMaster**，一种专门用于高效可扩展多语言数据合成的多代理协作框架。利用这一框架，我们整理了**SwitchLingua**，这是第一个大规模的多语言和多族裔代码转换数据集，包含：(1) 12种语言的420,000个代码转换文本样本；(2) 来自174位代表18个国家/地区和63种种族/族裔背景的讲话者，超过80小时的音频记录，基于文本数据。该数据集涵盖了丰富的语言和文化多样性，为多语言和多文化研究提供了基础资源。此外，为了解决现有ASR评估指标对代码转换场景敏感性不足的问题，我们提出了**语义感知错误率（SAER）**，这是一种新颖的评估指标，结合了语义信息，提供了更准确和情境化的系统性能评估。 

---
# COSMIC: Generalized Refusal Direction Identification in LLM Activations 

**Title (ZH)**: COSMIC: LLM激活中泛化的拒绝方向识别 

**Authors**: Vincent Siu, Nicholas Crispino, Zihao Yu, Sam Pan, Zhun Wang, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00085)  

**Abstract**: Large Language Models (LLMs) encode behaviors such as refusal within their activation space, yet identifying these behaviors remains a significant challenge. Existing methods often rely on predefined refusal templates detectable in output tokens or require manual analysis. We introduce \textbf{COSMIC} (Cosine Similarity Metrics for Inversion of Concepts), an automated framework for direction selection that identifies viable steering directions and target layers using cosine similarity - entirely independent of model outputs. COSMIC achieves steering performance comparable to prior methods without requiring assumptions about a model's refusal behavior, such as the presence of specific refusal tokens. It reliably identifies refusal directions in adversarial settings and weakly aligned models, and is capable of steering such models toward safer behavior with minimal increase in false refusals, demonstrating robustness across a wide range of alignment conditions. 

**Abstract (ZH)**: Large Language Models (LLMs)中的拒绝行为编码在其激活空间内，但识别这些行为仍是重大挑战。现有方法通常依赖于预定义的拒绝模板来检测输出标记，或者需要人工分析。我们介绍了COSMIC（余弦相似度指标的概念反转），一种自动方向选择框架，该框架利用余弦相似度来识别可行的控制方向和目标层，完全不依赖于模型输出。COSMIC在不需要假设模型的拒绝行为（如特定拒绝标记的存在）的情况下，实现了与先前方法相当的控制性能。它能够在对抗性设置和弱对齐模型中可靠地识别拒绝方向，并能够以最小的误拒绝增加将此类模型引导至更安全的行为，展示了在广泛对齐条件下的稳健性。 

---
# Hi-Dyna Graph: Hierarchical Dynamic Scene Graph for Robotic Autonomy in Human-Centric Environments 

**Title (ZH)**: Hi-Dyna 图：以人为本环境中机器人自主性的分层动态场景图 

**Authors**: Jiawei Hou, Xiangyang Xue, Taiping Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00083)  

**Abstract**: Autonomous operation of service robotics in human-centric scenes remains challenging due to the need for understanding of changing environments and context-aware decision-making. While existing approaches like topological maps offer efficient spatial priors, they fail to model transient object relationships, whereas dense neural representations (e.g., NeRF) incur prohibitive computational costs. Inspired by the hierarchical scene representation and video scene graph generation works, we propose Hi-Dyna Graph, a hierarchical dynamic scene graph architecture that integrates persistent global layouts with localized dynamic semantics for embodied robotic autonomy. Our framework constructs a global topological graph from posed RGB-D inputs, encoding room-scale connectivity and large static objects (e.g., furniture), while environmental and egocentric cameras populate dynamic subgraphs with object position relations and human-object interaction patterns. A hybrid architecture is conducted by anchoring these subgraphs to the global topology using semantic and spatial constraints, enabling seamless updates as the environment evolves. An agent powered by large language models (LLMs) is employed to interpret the unified graph, infer latent task triggers, and generate executable instructions grounded in robotic affordances. We conduct complex experiments to demonstrate Hi-Dyna Grap's superior scene representation effectiveness. Real-world deployments validate the system's practicality with a mobile manipulator: robotics autonomously complete complex tasks with no further training or complex rewarding in a dynamic scene as cafeteria assistant. See this https URL for video demonstration and more details. 

**Abstract (ZH)**: 基于人类中心场景的服务机器人自主操作仍具有挑战性，因为需要理解和进行情境aware决策。现有的方法如拓扑地图虽然提供了有效的空间先验知识，但无法建模瞬时对象关系，而密集神经表示（如NeRF）则会产生高昂的计算成本。受分层场景表示和视频场景图生成工作的启发，我们提出了一种分层动态场景图架构Hi-Dyna Graph，该架构结合了持久的全局布局和局部动态语义，以实现嵌入式机器人的自主性。我们的框架从摆拍的RGB-D输入构建全局拓扑图，编码房间尺度的连接性和大型静态对象（如家具），同时环境和第一人称相机填充动态子图，包含对象位置关系和人机交互模式。通过结合语义和空间约束将这些子图锚定到全局拓扑结构中，实现环境演变时的无缝更新。由大规模语言模型驱动的代理用于解释统一的图，推断潜在任务触发，并生成基于机器人操作能力的可执行指令。通过复杂实验展示了Hi-Dyna Graph在场景表示有效性上的优越性。实际部署验证了系统的实用性，使用移动 manipulator：机器人在动态场景中作为自助餐厅助手自主完成复杂任务，无需进一步训练或复杂奖励。更多详情请参见此链接。 

---
# Artificial Empathy: AI based Mental Health 

**Title (ZH)**: 人工共情：基于AI的心理健康 

**Authors**: Aditya Naik, Jovi Thomas, Teja Sree, Himavant Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.00081)  

**Abstract**: Many people suffer from mental health problems but not everyone seeks professional help or has access to mental health care. AI chatbots have increasingly become a go-to for individuals who either have mental disorders or simply want someone to talk to. This paper presents a study on participants who have previously used chatbots and a scenario-based testing of large language model (LLM) chatbots. Our findings indicate that AI chatbots were primarily utilized as a "Five minute therapist" or as a non-judgmental companion. Participants appreciated the anonymity and lack of judgment from chatbots. However, there were concerns about privacy and the security of sensitive information. The scenario-based testing of LLM chatbots highlighted additional issues. Some chatbots were consistently reassuring, used emojis and names to add a personal touch, and were quick to suggest seeking professional help. However, there were limitations such as inconsistent tone, occasional inappropriate responses (e.g., casual or romantic), and a lack of crisis sensitivity, particularly in recognizing red flag language and escalating responses appropriately. These findings can inform both the technology and mental health care industries on how to better utilize AI chatbots to support individuals during challenging emotional periods. 

**Abstract (ZH)**: AI聊天机器人在心理健康支持中的应用与挑战：基于场景的大型语言模型测试研究 

---
# Bottom-Up Perspectives on AI Governance: Insights from User Reviews of AI Products 

**Title (ZH)**: 自下而上的AI治理视角：来自AI产品用户评审的见解 

**Authors**: Stefan Pasch  

**Link**: [PDF](https://arxiv.org/pdf/2506.00080)  

**Abstract**: With the growing importance of AI governance, numerous high-level frameworks and principles have been articulated by policymakers, institutions, and expert communities to guide the development and application of AI. While such frameworks offer valuable normative orientation, they may not fully capture the practical concerns of those who interact with AI systems in organizational and operational contexts. To address this gap, this study adopts a bottom-up approach to explore how governance-relevant themes are expressed in user discourse. Drawing on over 100,000 user reviews of AI products from this http URL, we apply BERTopic to extract latent themes and identify those most semantically related to AI governance. The analysis reveals a diverse set of governance-relevant topics spanning both technical and non-technical domains. These include concerns across organizational processes-such as planning, coordination, and communication-as well as stages of the AI value chain, including deployment infrastructure, data handling, and analytics. The findings show considerable overlap with institutional AI governance and ethics frameworks on issues like privacy and transparency, but also surface overlooked areas such as project management, strategy development, and customer interaction. This highlights the need for more empirically grounded, user-centered approaches to AI governance-approaches that complement normative models by capturing how governance unfolds in applied settings. By foregrounding how governance is enacted in practice, this study contributes to more inclusive and operationally grounded approaches to AI governance and digital policy. 

**Abstract (ZH)**: 随着AI治理的重要性日益增长，政策制定者、机构和专家社区已经制定了众多高层次的框架和原则，以指导AI的发展与应用。尽管这些框架提供了有价值的规范性导向，但在组织和运行 contexts 中与AI系统互动的人可能不会全面关注其中的实际问题。为解决这一缺口，本研究采用自下而上的方法探索治理相关主题在用户话语中的表达。通过分析来自某网站的超过100,000条AI产品的用户评论，我们应用BERTopic提取潜在主题，并识别与AI治理最相关的主题。分析结果显示，治理相关的话题涵盖了技术与非技术领域。这些话题包括组织流程中的担忧，如规划、协调和沟通，以及AI价值链的各个阶段，包括部署基础设施、数据处理和数据分析。研究发现，在隐私和透明度等议题上与机构AI治理和伦理框架存在显著重叠，但也揭示了诸如项目管理、战略发展和客户互动等未被充分注意的领域。这强调了需要更多基于实证、用户中心的AI治理方法——这些方法可以补充规范性模型，捕捉治理在应用环境中的实际展开过程。通过凸显治理在实践中的具体表现，本研究促进了更具包容性和操作性的AI治理和数字政策方法。 

---
# Who Gets the Kidney? Human-AI Alignment, Indecision, and Moral Values 

**Title (ZH)**: 谁获得肾脏？人类与人工智能的aligniment、犹豫与道德价值观 

**Authors**: John P. Dickerson, Hadi Hosseini, Samarth Khanna, Leona Pierce  

**Link**: [PDF](https://arxiv.org/pdf/2506.00079)  

**Abstract**: The rapid integration of Large Language Models (LLMs) in high-stakes decision-making -- such as allocating scarce resources like donor organs -- raises critical questions about their alignment with human moral values. We systematically evaluate the behavior of several prominent LLMs against human preferences in kidney allocation scenarios and show that LLMs: i) exhibit stark deviations from human values in prioritizing various attributes, and ii) in contrast to humans, LLMs rarely express indecision, opting for deterministic decisions even when alternative indecision mechanisms (e.g., coin flipping) are provided. Nonetheless, we show that low-rank supervised fine-tuning with few samples is often effective in improving both decision consistency and calibrating indecision modeling. These findings illustrate the necessity of explicit alignment strategies for LLMs in moral/ethical domains. 

**Abstract (ZH)**: 大型语言模型在高stakes决策中的快速集成——如分配稀缺资源（如捐赠器官）——引起对其与人类道德价值一致性的关键问题。我们系统评估了几种 prominant LLMs 在肾脏分配情景中对人类偏好行为，并表明：i) LLMs 在优先处理各种属性方面表现出与人类价值观的明显偏离，ii) 与人类不同，LLMs 很少表现出犹豫不决，倾向于作出确定性决策，即使提供了替代的犹豫机制（如抛硬币）也是如此。然而，我们显示，使用少量样本的低秩监督微调通常能有效提高决策一致性并校准犹豫建模。这些发现表明，在道德/伦理领域中，需要明确的对齐策略以确保大型语言模型的一致性。 

---
# Optimizing Storytelling, Improving Audience Retention, and Reducing Waste in the Entertainment Industry 

**Title (ZH)**: 优化叙事技巧，提升观众留存率，减少娱乐行业的浪费 

**Authors**: Andrew Cornfeld, Ashley Miller, Mercedes Mora-Figueroa, Kurt Samuels, Anthony Palomba  

**Link**: [PDF](https://arxiv.org/pdf/2506.00076)  

**Abstract**: Television networks face high financial risk when making programming decisions, often relying on limited historical data to forecast episodic viewership. This study introduces a machine learning framework that integrates natural language processing (NLP) features from over 25000 television episodes with traditional viewership data to enhance predictive accuracy. By extracting emotional tone, cognitive complexity, and narrative structure from episode dialogue, we evaluate forecasting performance using SARIMAX, rolling XGBoost, and feature selection models. While prior viewership remains a strong baseline predictor, NLP features contribute meaningful improvements for some series. We also introduce a similarity scoring method based on Euclidean distance between aggregate dialogue vectors to compare shows by content. Tested across diverse genres, including Better Call Saul and Abbott Elementary, our framework reveals genre-specific performance and offers interpretable metrics for writers, executives, and marketers seeking data-driven insight into audience behavior. 

**Abstract (ZH)**: 电视网络在节目制作决策中面临高い财务风险，通常依赖有限的历史数据来预测集锦收视率。本研究介绍了一种将自然语言处理（NLP）功能与传统收视数据结合的机器学习框架，以提高预测准确性。通过从超过25000个电视节目中提取情绪基调、认知复杂性和叙事结构，我们使用SARIMAX、滚动XGBoost和特征选择模型评估预测性能。尽管过去的收视情况仍然是一个强大的基准预测器，但对于某些系列而言，NLP功能的贡献意味着有意义的提升。我们还介绍了一种基于聚类对话向量之间欧几里得距离的相似性评分方法，用于按内容比较节目。在包括《Better Call Saul》和《Abbott Elementary》在内的多种类型中测试，本框架揭示了特定类型的性能，并为编剧、高层管理人员和市场营销人员提供了可解释的指标，以获取有关观众行为的数据驱动见解。 

---
# Reducing Latency in LLM-Based Natural Language Commands Processing for Robot Navigation 

**Title (ZH)**: 基于LLM的自然语言命令处理在机器人导航中减少延迟 

**Authors**: Diego Pollini, Bruna V. Guterres, Rodrigo S. Guerra, Ricardo B. Grando  

**Link**: [PDF](https://arxiv.org/pdf/2506.00075)  

**Abstract**: The integration of Large Language Models (LLMs), such as GPT, in industrial robotics enhances operational efficiency and human-robot collaboration. However, the computational complexity and size of these models often provide latency problems in request and response times. This study explores the integration of the ChatGPT natural language model with the Robot Operating System 2 (ROS 2) to mitigate interaction latency and improve robotic system control within a simulated Gazebo environment. We present an architecture that integrates these technologies without requiring a middleware transport platform, detailing how a simulated mobile robot responds to text and voice commands. Experimental results demonstrate that this integration improves execution speed, usability, and accessibility of the human-robot interaction by decreasing the communication latency by 7.01\% on average. Such improvements facilitate smoother, real-time robot operations, which are crucial for industrial automation and precision tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT在工业机器人中的集成增强了操作效率和人机协作，但这些模型的计算复杂度和大小 often 提供了请求和响应时间的延迟问题。本研究探索将ChatGPT自然语言模型与Robot Operating System 2（ROS 2）集成，以减轻互动延迟并改善在模拟Gazebo环境中的机器人系统控制。我们提出了一种architecture，无需中间件传输平台即可集成这些技术，并详细介绍了模拟移动机器人对文本和语音命令的响应方式。实验结果表明，这种集成通过平均减少7.01%的通信延迟，改善了人机交互的执行速度、可用性和便捷性。这些改进促进了更顺畅、实时的机器人操作，这对于工业自动化和精密任务至关重要。 

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
# Human sensory-musculoskeletal modeling and control of whole-body movements 

**Title (ZH)**: 人类感觉-运动系统建模与全身运动控制 

**Authors**: Chenhui Zuo, Guohao Lin, Chen Zhang, Shanning Zhuang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00071)  

**Abstract**: Coordinated human movement depends on the integration of multisensory inputs, sensorimotor transformation, and motor execution, as well as sensory feedback resulting from body-environment interaction. Building dynamic models of the sensory-musculoskeletal system is essential for understanding movement control and investigating human behaviours. Here, we report a human sensory-musculoskeletal model, termed SMS-Human, that integrates precise anatomical representations of bones, joints, and muscle-tendon units with multimodal sensory inputs involving visual, vestibular, proprioceptive, and tactile components. A stage-wise hierarchical deep reinforcement learning framework was developed to address the inherent challenges of high-dimensional control in musculoskeletal systems with integrated multisensory information. Using this framework, we demonstrated the simulation of three representative movement tasks, including bipedal locomotion, vision-guided object manipulation, and human-machine interaction during bicycling. Our results showed a close resemblance between natural and simulated human motor behaviours. The simulation also revealed musculoskeletal dynamics that could not be directly measured. This work sheds deeper insights into the sensorimotor dynamics of human movements, facilitates quantitative understanding of human behaviours in interactive contexts, and informs the design of systems with embodied intelligence. 

**Abstract (ZH)**: 协调的人类运动取决于多感官输入的整合、传感器运动转换以及运动执行，并且依赖于身体与环境相互作用所产生的感觉反馈。构建感觉-肌骨系统的动态模型是理解运动控制和探索人类行为的基础。在这里，我们报告了一个名为SMS-Human的人类感觉-肌骨模型，该模型整合了精确的骨骼、关节和肌腱单位的解剖学表示以及涉及视觉、前庭、本体感觉和触觉的多模态感觉输入。我们开发了一种分阶段层次化的深度强化学习框架，以解决集成多感官信息的肌骨系统中固有的高维控制难题。利用这一框架，我们展示了三种代表性运动任务的模拟，包括双足行走、视觉引导下的物体操作以及骑自行车时的人机交互。我们的结果表明，自然的人类运动行为与模拟行为相似。模拟还揭示了无法直接测量的肌骨系统动力学。这项工作深入探讨了人类运动的传感器运动动力学，促进了在交互上下文中对人类行为的量化理解，并为具有体态智能系统的开发提供了指导。 

---
# Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics 

**Title (ZH)**: Robot-R1：强化学习在增强机器人 embodied reasoning 中的应用 

**Authors**: Dongyoung Kim, Sumin Park, Huiwon Jang, Jinwoo Shin, Jaehyung Kim, Younggyo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00070)  

**Abstract**: Large Vision-Language Models (LVLMs) have recently shown great promise in advancing robotics by combining embodied reasoning with robot control. A common approach involves training on embodied reasoning tasks related to robot control using Supervised Fine-Tuning (SFT). However, SFT datasets are often heuristically constructed and not explicitly optimized for improving robot control. Furthermore, SFT often leads to issues such as catastrophic forgetting and reduced generalization performance. To address these limitations, we introduce Robot-R1, a novel framework that leverages reinforcement learning to enhance embodied reasoning specifically for robot control. Robot-R1 learns to predict the next keypoint state required for task completion, conditioned on the current scene image and environment metadata derived from expert demonstrations. Inspired by the DeepSeek-R1 learning approach, Robot-R1 samples reasoning-based responses and reinforces those that lead to more accurate predictions. Our experiments show that models trained with Robot-R1 outperform SFT methods on embodied reasoning tasks. Despite having only 7B parameters, Robot-R1 even surpasses GPT-4o on reasoning tasks related to low-level action control, such as spatial and primitive movement reasoning. 

**Abstract (ZH)**: 大型多模态模型（LVLMs）通过结合本体推理与机器人控制，在推动机器人技术方面展现出了巨大的潜力。一种常见方法是使用监督微调（SFT）在与机器人控制相关的本体推理任务上进行训练。然而，SFT数据集通常是基于启发式构建的，并未明确优化以提高机器人控制性能。此外，SFT还常常导致灾难性遗忘和泛化性能降低等问题。为解决这些问题，我们提出了Robot-R1新型框架，该框架利用强化学习来增强特定于机器人控制的本体推理。Robot-R1通过当前场景图像和从专家示范中派生的环境元数据条件，学习预测完成任务所需的下一个关键点状态。受DeepSeek-R1学习方法的启发，Robot-R1采样基于推理的响应，并强化那些能够产生更准确预测的回答。我们的实验结果表明，使用Robot-R1训练的模型在本体推理任务上的表现优于SFT方法。即使只有7B参数，Robot-R1在低层级动作控制相关的推理任务（如空间和基本运动推理）上也超越了GPT-4o。 

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
# You Prefer This One, I Prefer Yours: Using Reference Words is Harder Than Vocabulary Words for Humans and Multimodal Language Models 

**Title (ZH)**: 你偏好这个，我偏好那个：引用词比词汇词对人类和多模态语言模型来说更难处理 

**Authors**: Dota Tianai Dong, Yifan Luo, Po-Ya Angela Wang, Asli Ozyurek, Paula Rubio-Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.00065)  

**Abstract**: Multimodal language models (MLMs) increasingly communicate in human-like ways, yet their ability to use reference words remains largely overlooked despite their ubiquity in everyday communication. Our study addresses this gap by comparing human and MLM use of three word classes with increasing cognitive demands: vocabulary words, possessive pronouns (`mine' vs `yours'), and demonstrative pronouns (`this one' vs `that one'). Evaluating seven state-of-the-art MLMs against human participants, we observe a clear difficulty hierarchy: while MLMs approach human-level performance on the vocabulary task, they show substantial deficits with possessives and demonstratives. Our analysis reveals these difficulties stem from limitations in perspective-taking and spatial reasoning. Although prompt engineering improved model performance on possessive use, demonstrative use remained well below human-level competence. These findings provide theoretical and empirical evidence that producing grammatical forms requiring pragmatics and social cognition remains a clear challenge in current NLP systems. 

**Abstract (ZH)**: 多模态语言模型在使用引用词方面的能力仍被忽视：从词汇词、代词（“我的” vs “你的”）到指示代词（“这个” vs “那个”）的认知需求递增比较研究 

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
# Improving statistical learning methods via features selection without replacement sampling and random projection 

**Title (ZH)**: 通过无替换抽样和随机投影进行特征选择以提高统计学习方法 

**Authors**: Sulaiman khan, Muhammad Ahmad, Fida Ullah, Carlos Aguilar Ibañez, José Eduardo Valdez Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.00053)  

**Abstract**: Cancer is fundamentally a genetic disease characterized by genetic and epigenetic alterations that disrupt normal gene expression, leading to uncontrolled cell growth and metastasis. High-dimensional microarray datasets pose challenges for classification models due to the "small n, large p" problem, resulting in overfitting. This study makes three different key contributions: 1) we propose a machine learning-based approach integrating the Feature Selection Without Re-placement (FSWOR) technique and a projection method to improve classification accuracy. 2) We apply the Kendall statistical test to identify the most significant genes from the brain cancer mi-croarray dataset (GSE50161), reducing the feature space from 54,675 to 20,890 genes.3) we apply machine learning models using k-fold cross validation techniques in which our model incorpo-rates ensemble classifiers with LDA projection and Naïve Bayes, achieving a test score of 96%, outperforming existing methods by 9.09%. The results demonstrate the effectiveness of our ap-proach in high-dimensional gene expression analysis, improving classification accuracy while mitigating overfitting. This study contributes to cancer biomarker discovery, offering a robust computational method for analyzing microarray data. 

**Abstract (ZH)**: 癌症本质上是一种由遗传和表观遗传改变引起的基因疾病，这些改变会扰乱正常的基因表达，导致不受控制的细胞生长和转移。高维度的微阵列数据集由于“小n，大p”问题给分类模型带来了挑战，容易导致过拟合。本研究作出三项关键贡献：1）我们提出了一种基于机器学习的方法，结合Feature Selection Without Replacement（FSWOR）技术和投影方法以提高分类准确性。2）我们应用肯德尔统计检验从脑癌微阵列数据集（GSE50161）中筛选出最显著的基因，将特征空间从54,675个基因减少到20,890个基因。3）我们应用k折交叉验证技术并结合LDA投影和朴素贝叶斯的集成分类器构建模型，测试得分为96%，优于现有方法9.09%。研究结果证明了在高维度基因表达分析中我们方法的有效性，提高了分类准确性同时减轻了过拟合问题。本研究为癌症生物标志物的发现提供了稳健的计算方法，用于分析微阵列数据。 

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
# Risks of AI-driven product development and strategies for their mitigation 

**Title (ZH)**: AI驱动的产品开发风险及其缓解策略 

**Authors**: Jan Göpfert, Jann M. Weinand, Patrick Kuckertz, Noah Pflugradt, Jochen Linßen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00047)  

**Abstract**: Humanity is progressing towards automated product development, a trend that promises faster creation of better products and thus the acceleration of technological progress. However, increasing reliance on non-human agents for this process introduces many risks. This perspective aims to initiate a discussion on these risks and appropriate mitigation strategies. To this end, we outline a set of principles for safer AI-driven product development which emphasize human oversight, accountability, and explainable design, among others. The risk assessment covers both technical risks which affect product quality and safety, and sociotechnical risks which affect society. While AI-driven product development is still in its early stages, this discussion will help balance its opportunities and risks without delaying essential progress in understanding, norm-setting, and regulation. 

**Abstract (ZH)**: 人类正朝着自动化产品开发方向前进，这一趋势有望加快技术进步，更快地创造更优质的产品。然而，对这一过程越来越多地依赖非人类代理也带来了许多风险。本文旨在探讨这些风险以及适当的缓解策略。为此，我们概述了一套更安全的AI驱动产品开发原则，这些原则强调了人的监督、责任以及可解释的设计等。风险评估涵盖了影响产品质量和安全的技术风险以及影响社会的 sociotechnical 风险。尽管AI驱动产品开发仍处于早期阶段，但此次讨论将有助于平衡其机遇和风险，而不延误对了解、制定规范和监管的基本理解。 

---
# The Folly of AI for Age Verification 

**Title (ZH)**: AI在年龄验证中的谬误 

**Authors**: Reid McIlroy-Young  

**Link**: [PDF](https://arxiv.org/pdf/2506.00038)  

**Abstract**: In the near future a governmental body will be asked to allow companies to use AI for age verification. If they allow it the resulting system will both be easily circumvented and disproportionately misclassify minorities and low socioeconomic status users. This is predictable by showing that other very similar systems (facial recognition and remote proctoring software) have similar issues despite years of efforts to mitigate their biases. These biases are due to technical limitations both of the AI models themselves and the physical hardware they are running on that will be difficult to overcome below the cost of government ID-based age verification. Thus in, the near future, deploying an AI system for age verification is folly. 

**Abstract (ZH)**: 在未来，政府机构将被要求允许公司使用AI进行年龄验证。如果他们批准这一做法， resulting系统将容易被规避，并且会不成比例地错误分类少数群体和低社会经济地位的用户。这一点可以通过展示其他非常相似的系统（面部识别和远程监考软件）尽管经历了多年的努力以减轻其偏见，但仍存在类似问题来预测。这些偏见源于AI模型本身和它们运行的物理硬件的技术限制，克服这些限制的成本将高于基于政府身份验证的年龄验证成本。因此，在未来部署AI系统进行年龄验证是不智之举。 

---
# Amadeus-Verbo Technical Report: The powerful Qwen2.5 family models trained in Portuguese 

**Title (ZH)**: 阿玛德斯-韦博技术报告：葡萄牙语训练的Qwen2.5家族模型 

**Authors**: William Alberto Cruz-Castañeda, Marcellus Amadeus  

**Link**: [PDF](https://arxiv.org/pdf/2506.00019)  

**Abstract**: This report introduces the experience of developing Amadeus Verbo, a family of large language models for Brazilian Portuguese. To handle diverse use cases, Amadeus Verbo includes base-tuned, merged, and instruction-tuned models in sizes of 0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B parameters. Thus, the main objective is to show how easy it is to fine-tune foundation models to democratize the open-source development of Brazilian Portuguese LLMs when data and resources are available. Amadeus-Verbo family models are all available at HuggingFace at this https URL. 

**Abstract (ZH)**: 本报告介绍了Amadeus Verbo的发展经验，这是一个用于巴西葡萄牙语的大型语言模型家族。为了应对多样的应用场景，Amadeus Verbo包括了0.5B、1.5B、3B、7B、14B、32B和72B参数的基模型、合并模型和指令调优模型。因此，主要目标是展示在有足够的数据和资源的情况下，如何轻松地调优基础模型以促进巴西葡萄牙语开源大型语言模型的开发。Amadeus-Verbo家族模型均可在HuggingFace获取。 

---
# MolTextNet: A Two-Million Molecule-Text Dataset for Multimodal Molecular Learning 

**Title (ZH)**: MolTextNet：用于多模态分子学习的大型分子-文本数据集 

**Authors**: Yihan Zhu, Gang Liu, Eric Inae, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00009)  

**Abstract**: Small molecules are essential to drug discovery, and graph-language models hold promise for learning molecular properties and functions from text. However, existing molecule-text datasets are limited in scale and informativeness, restricting the training of generalizable multimodal models. We present MolTextNet, a dataset of 2.5 million high-quality molecule-text pairs designed to overcome these limitations. To construct it, we propose a synthetic text generation pipeline that integrates structural features, computed properties, bioactivity data, and synthetic complexity. Using GPT-4o-mini, we create structured descriptions for 2.5 million molecules from ChEMBL35, with text over 10 times longer than prior datasets. MolTextNet supports diverse downstream tasks, including property prediction and structure retrieval. Pretraining CLIP-style models with Graph Neural Networks and ModernBERT on MolTextNet yields improved performance, highlighting its potential for advancing foundational multimodal modeling in molecular science. Our dataset is available at this https URL. 

**Abstract (ZH)**: MolTextNet：一个用于分子科学基础多模态建模的高质量分子-文本数据集 

---
# Rapid yet accurate Tile-circuit and device modeling for Analog In-Memory Computing 

**Title (ZH)**: 快速而准确的Tile电路和器件建模方法及其在类比内存计算中的应用 

**Authors**: J. Luquin, C. Mackin, S. Ambrogio, A. Chen, F. Baldi, G. Miralles, M.J. Rasch, J. Büchel, M. Lalwani, W. Ponghiran, P. Solomon, H. Tsai, G.W. Burr, P. Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00004)  

**Abstract**: Analog In-Memory Compute (AIMC) can improve the energy efficiency of Deep Learning by orders of magnitude. Yet analog-domain device and circuit non-idealities -- within the analog ``Tiles'' performing Matrix-Vector Multiply (MVM) operations -- can degrade neural-network task accuracy. We quantify the impact of low-level distortions and noise, and develop a mathematical model for Multiply-ACcumulate (MAC) operations mapped to analog tiles. Instantaneous-current IR-drop (the most significant circuit non-ideality), and ADC quantization effects are fully captured by this model, which can predict MVM tile-outputs both rapidly and accurately, as compared to much slower rigorous circuit simulations. A statistical model of PCM read noise at nanosecond timescales is derived from -- and matched against -- experimental measurements. We integrate these (statistical) device and (deterministic) circuit effects into a PyTorch-based framework to assess the accuracy impact on the BERT and ALBERT Transformer networks. We show that hardware-aware fine-tuning using simple Gaussian noise provides resilience against ADC quantization and PCM read noise effects, but is less effective against IR-drop. This is because IR-drop -- although deterministic -- is non-linear, is changing significantly during the time-integration window, and is ultimately dependent on all the excitations being introduced in parallel into the analog tile. The apparent inability of simple Gaussian noise applied during training to properly prepare a DNN network for IR-drop during inference implies that more complex training approaches -- incorporating advances such as the Tile-circuit model introduced here -- will be critical for resilient deployment of large neural networks onto AIMC hardware. 

**Abstract (ZH)**: 模拟内存计算（AIMC）可以大幅提升深度学习的能效。然而，模拟域器件和电路非理想性——在执行矩阵-向量乘法（MVM）操作的模拟“瓷砖”中——可能会降低神经网络任务的准确性。我们量化了低级失真和噪声的影响，并开发了一个适用于映射到模拟瓷砖的乘加（MAC）操作的数学模型。瞬态电流IR降（电路非理想性中最显著的因素）和ADC量化效应完全由该模型捕获，该模型比更慢的严格电路仿真能更快更准确地预测MVM瓷砖的输出。小型脉码模（PCM）读噪声的统计模型从实验测量中导出并匹配。我们将这些（统计）设备效果和（确定性）电路效果集成到基于PyTorch的框架中，以评估其对BERT和ALBERT变换器网络的影响。我们展示了使用简单高斯噪声进行硬件感知微调可以增强对ADC量化和PCM读噪声效应的鲁棒性，但对IR降的效果较差。这是因为虽然IR降是确定性的，但它是非线性的，在时间积分窗口中显著变化，并最终依赖于同时引入到模拟瓷砖的所有激励。简单高斯噪声在训练期间应用于准备DNN网络以应对推断中的IR降的能力有限，这表明需要更复杂的训练方法——如这里引入的瓷砖电路模型——对于在AIMC硬件上部署大规模神经网络至关重要。 

---
# Advancing AI-assisted Hardware Design with Hierarchical Decentralized Training and Personalized Inference-Time Optimization 

**Title (ZH)**: 基于分层去中心化训练与个性化推理时优化的AI辅助硬件设计推进 

**Authors**: Hao Mark Chen, Zehuan Zhang, Wanru Zhao, Nicholas Lane, Hongxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00002)  

**Abstract**: Recent years have witnessed a significant increase in the adoption of AI techniques to enhance electronic design automation. In particular, the emergence of Large Language Models (LLMs) has sparked significant interest in LLM-assisted hardware design generation, spanning applications from classical digital circuits to quantum computing. Despite substantial progress in this direction, the quality of LLM-generated hardware design still cannot meet the requirements for practical deployment. In this work, we identify three critical challenges hindering the development of LLM-assisted hardware design generation: 1) limited data availability, 2) varied data quality, 3) inadequate inference-time efficiency. To address these fundamental challenges, this paper introduces a two-stage framework for AI-assisted hardware design by exploring decentralized training and personalized inference. In the first stage, we propose to harness private domain design sources through a hierarchical decentralized training mechanism that addresses data-sharing constraints. To mitigate the impact of low-quality data, we identify optimization opportunities in hardware generation tasks, using user-defined metrics for model aggregation. The second stage focuses on client personalization to enhance both speed and quality. We introduce a new metric, Trueput, to analyze LLM-assisted hardware generation efficiency. To optimize Trueput, we implement personalized inference-time acceleration and customized sampling strategies. Evaluating both classical and quantum benchmarks, our experimental results demonstrate that the proposed two-stage framework can significantly improve the model capability for hardware design generation. As orthogonal enhancements to existing methods, our framework can achieve $33\% \sim 50\%$ semantic accuracy improvement and $2.3$ times speedup, depending on the difficulty of the generation tasks. 

**Abstract (ZH)**: 近年来，人工智能技术在电子设计自动化中的应用显著增加。特别是大型语言模型（LLMs）的出现，引发了对LLM辅助硬件设计生成的兴趣，涵盖了从经典数字电路到量子计算的应用。尽管在此方向上取得了重大进展，但LLM生成的硬件设计质量仍无法满足实际部署的要求。本文识别了阻碍LLM辅助硬件设计生成发展的三大关键挑战：1）数据 availability有限，2）数据质量参差不齐，3）推理时效率不足。为了解决这些基本挑战，本文提出了一个两阶段的AI辅助硬件设计框架，通过探索去中心化训练和个人化推理。在第一阶段，我们提出通过分层去中心化训练机制利用私有的领域设计源，以解决数据共享约束问题。为了减轻低质量数据的影响，我们通过使用用户定义的指标在硬件生成任务中进行模型聚合来识别优化机会。第二阶段关注客户端个性化，以提高速度和质量。我们引入了一个新的度量标准Trueput来分析LLM辅助硬件生成的效率。为了优化Trueput，我们实施了个性化推理时间和定制采样策略。通过对经典和量子基准的评估，实验结果表明，所提出的两阶段框架可以显著提高硬件设计生成的模型能力。作为现有方法的补充改进，我们的框架在生成任务难度不同的情况下可以实现33%至50%的语义准确性提升和2.3倍的速度加速。 

---
