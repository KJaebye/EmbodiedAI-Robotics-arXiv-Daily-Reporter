# Occ-LLM: Enhancing Autonomous Driving with Occupancy-Based Large Language Models 

**Title (ZH)**: 基于占用率的大语言模型增强自动驾驶：Occ-LLM 

**Authors**: Tianshuo Xu, Hao Lu, Xu Yan, Yingjie Cai, Bingbing Liu, Yingcong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.06419)  

**Abstract**: Large Language Models (LLMs) have made substantial advancements in the field of robotic and autonomous driving. This study presents the first Occupancy-based Large Language Model (Occ-LLM), which represents a pioneering effort to integrate LLMs with an important representation. To effectively encode occupancy as input for the LLM and address the category imbalances associated with occupancy, we propose Motion Separation Variational Autoencoder (MS-VAE). This innovative approach utilizes prior knowledge to distinguish dynamic objects from static scenes before inputting them into a tailored Variational Autoencoder (VAE). This separation enhances the model's capacity to concentrate on dynamic trajectories while effectively reconstructing static scenes. The efficacy of Occ-LLM has been validated across key tasks, including 4D occupancy forecasting, self-ego planning, and occupancy-based scene question answering. Comprehensive evaluations demonstrate that Occ-LLM significantly surpasses existing state-of-the-art methodologies, achieving gains of about 6\% in Intersection over Union (IoU) and 4\% in mean Intersection over Union (mIoU) for the task of 4D occupancy forecasting. These findings highlight the transformative potential of Occ-LLM in reshaping current paradigms within robotic and autonomous driving. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机器人和自主驾驶领域取得了重大进展。本研究提出了首个基于 occupancy 的大型语言模型（Occ-LLM），这是将 LLMs 与重要表示形式集成的开创性努力。为了有效将 occupancy 作为输入编码到 LLM 中并解决与 occupancy 相关的类别不平衡问题，我们提出了运动分离变分自编码器（MS-VAE）。这种创新方法利用先验知识，在输入到定制的变分自编码器（VAE）之前，将动态对象与静态场景区分开来。这种分离增强了模型专注于动态轨迹并有效重建静态场景的能力。Occ-LLM 的有效性已经在 4D 占有率预测、自我自我规划以及基于占有的场景问答等关键任务中得到验证。综合评估表明，Occ-LLM 显著优于现有最先进的方法，在 4D 占有率预测任务中，IoU 和 mIoU 分别提高了约 6% 和 4%。这些发现强调了 Occ-LLM 在重塑机器人和自主驾驶领域现有范式方面的变革潜力。 

---
# Robotouille: An Asynchronous Planning Benchmark for LLM Agents 

**Title (ZH)**: Robotouille: 一种用于LLM智能体的异步规划基准测试 

**Authors**: Gonzalo Gonzalez-Pumariega, Leong Su Yean, Neha Sunkara, Sanjiban Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2502.05227)  

**Abstract**: Effective asynchronous planning, or the ability to efficiently reason and plan over states and actions that must happen in parallel or sequentially, is essential for agents that must account for time delays, reason over diverse long-horizon tasks, and collaborate with other agents. While large language model (LLM) agents show promise in high-level task planning, current benchmarks focus primarily on short-horizon tasks and do not evaluate such asynchronous planning capabilities. We introduce Robotouille, a challenging benchmark environment designed to test LLM agents' ability to handle long-horizon asynchronous scenarios. Our synchronous and asynchronous datasets capture increasingly complex planning challenges that go beyond existing benchmarks, requiring agents to manage overlapping tasks and interruptions. Our results show that ReAct (gpt4-o) achieves 47% on synchronous tasks but only 11% on asynchronous tasks, highlighting significant room for improvement. We further analyze failure modes, demonstrating the need for LLM agents to better incorporate long-horizon feedback and self-audit their reasoning during task execution. Code is available at this https URL. 

**Abstract (ZH)**: 有效的异步规划能力对于能够考虑时间延迟、处理多样化长时 horizon 任务以及与其他代理协作的智能体至关重要。虽然大规模语言模型（LLM）智能体在高层次任务规划方面显示出希望，但当前的基准测试主要集中在短时 horizon 任务上，并未评估这种异步规划能力。我们引入了Robotouille，一个具有挑战性的基准环境，旨在测试LLM智能体处理长时 horizon 异步场景的能力。我们的同步和异步数据集捕捉了超出现有基准测试的日益复杂的规划挑战，要求智能体管理重叠任务和中断。我们的结果显示，ReAct（gpt4-o）在同步任务中得分47%，而在异步任务中仅得11%，这突显了巨大的改进空间。我们进一步分析了失败模式，表明LLM智能体需要更好地整合长时 horizon 反馈并在任务执行期间自我审查其推理。代码可在以下链接获取。 

---
# On the Emergence of Thinking in LLMs I: Searching for the Right Intuition 

**Title (ZH)**: LLMs中思维涌现的研究I：寻找正确的直觉 

**Authors**: Guanghao Ye, Khiem Duc Pham, Xinzhi Zhang, Sivakanth Gopi, Baolin Peng, Beibin Li, Janardhan Kulkarni, Huseyin A. Inan  

**Link**: [PDF](https://arxiv.org/pdf/2502.06773)  

**Abstract**: Recent AI advancements, such as OpenAI's new models, are transforming LLMs into LRMs (Large Reasoning Models) that perform reasoning during inference, taking extra time and compute for higher-quality outputs. We aim to uncover the algorithmic framework for training LRMs. Methods like self-consistency, PRM, and AlphaZero suggest reasoning as guided search. We ask: what is the simplest, most scalable way to enable search in LLMs?
We propose a post-training framework called Reinforcement Learning via Self-Play (RLSP). RLSP involves three steps: (1) supervised fine-tuning with human or synthetic demonstrations of the reasoning process, (2) using an exploration reward signal to encourage diverse and efficient reasoning behaviors, and (3) RL training with an outcome verifier to ensure correctness while preventing reward hacking. Our key innovation is to decouple exploration and correctness signals during PPO training, carefully balancing them to improve performance and efficiency.
Empirical studies in the math domain show that RLSP improves reasoning. On the Llama-3.1-8B-Instruct model, RLSP can boost performance by 23% in MATH-500 test set; On AIME 2024 math problems, Qwen2.5-32B-Instruct improved by 10% due to RLSP. However, a more important finding of this work is that the models trained using RLSP, even with the simplest exploration reward that encourages the model to take more intermediate steps, showed several emergent behaviors such as backtracking, exploration of ideas, and verification. These findings demonstrate that RLSP framework might be enough to enable emergence of complex reasoning abilities in LLMs when scaled. Lastly, we propose a theory as to why RLSP search strategy is more suitable for LLMs inspired by a remarkable result that says CoT provably increases computational power of LLMs, which grows as the number of steps in CoT \cite{li2024chain,merrill2023expresssive}. 

**Abstract (ZH)**: Recent AI进步，如OpenAI的新模型，正在将LLMs转换为LRMs（大型推理模型），这些模型在推理过程中消耗更多的计算资源以生成高质量的输出。我们旨在探索训练LRMs的算法框架。类似自我一致性、PRM和AlphaZero的方法表明推理是一种引导式搜索。我们问道：在LLMs中启用搜索的最简单且可扩展的方式是什么？

我们提出了一种后训练框架，称为基于自我对弈的强化学习（RLSP）。RLSP包括三个步骤：（1）监督微调，使用推理过程的人类或合成演示，（2）使用探索奖励信号来鼓励多样且高效的推理行为，（3）与结果验证器结合的RL训练，确保正确性并防止奖励作弊。我们的主要创新是在PPO训练过程中将探索和正确性信号分离，并仔细平衡它们以提高性能和效率。

在数学领域的实证研究表明，RLSP提升了推理能力。在Llama-3.1-8B-Instruct模型上，RLSP在MATH-500数据集上将性能提升23%；Qwen2.5-32B-Instruct在AIME 2024数学问题上因RLSP提高了10%。然而，这项研究更重要的发现是，使用RLSP训练的模型，即使使用最简单的探索奖励信号以鼓励模型采取更多中间步骤，也表现出了一些新兴的行为，如回溯、思想探索和验证。这些发现表明，当扩展时，RLSP框架可能足以使LLMs具备复杂推理能力。最后，我们提出了一种理论，解释为何RLSP搜索策略更适合LLMs，受到一个显著结果的启发，即CoT（逐步思维）已证明可以增加LLMs的计算能力，且随着CoT步骤数的增加而增长（参见[li2024chain, merrill2023expresssive]）。 

---
# Unbiased Evaluation of Large Language Models from a Causal Perspective 

**Title (ZH)**: 从因果角度出发的大型语言模型无偏评估 

**Authors**: Meilin Chen, Jian Tian, Liang Ma, Di Xie, Weijie Chen, Jiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.06655)  

**Abstract**: Benchmark contamination has become a significant concern in the LLM evaluation community. Previous Agents-as-an-Evaluator address this issue by involving agents in the generation of questions. Despite their success, the biases in Agents-as-an-Evaluator methods remain largely unexplored. In this paper, we present a theoretical formulation of evaluation bias, providing valuable insights into designing unbiased evaluation protocols. Furthermore, we identify two type of bias in Agents-as-an-Evaluator through carefully designed probing tasks on a minimal Agents-as-an-Evaluator setup. To address these issues, we propose the Unbiased Evaluator, an evaluation protocol that delivers a more comprehensive, unbiased, and interpretable assessment of this http URL experiments reveal significant room for improvement in current LLMs. Additionally, we demonstrate that the Unbiased Evaluator not only offers strong evidence of benchmark contamination but also provides interpretable evaluation results. 

**Abstract (ZH)**: 基准污染在大语言模型评估社区中已成为一个重要问题。以往的代理作为评估者通过让代理参与问题生成来应对这一问题，尽管取得了成功，但代理作为评估者方法中的偏见仍然 largely unexplored。在本文中，我们提出了一种评估偏差的理论框架，为设计无偏的评估协议提供了宝贵的见解。此外，通过精心设计的探针任务，我们在最小化的代理作为评估者设置中识定了两种类型的偏见。为解决这些issue，我们提出了无偏评估者，这是一种更全面、无偏且可解释的评估协议。实验结果显示当前大语言模型有显著改进空间。此外，我们证明无偏评估者不仅能提供基准污染的有力证据，还能提供可解释的评估结果。 

---
# Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning 

**Title (ZH)**: 使用多智能体强化学习训练语言模型进行社交推理 

**Authors**: Bidipta Sarkar, Warren Xia, C. Karen Liu, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2502.06060)  

**Abstract**: Communicating in natural language is a powerful tool in multi-agent settings, as it enables independent agents to share information in partially observable settings and allows zero-shot coordination with humans. However, most prior works are limited as they either rely on training with large amounts of human demonstrations or lack the ability to generate natural and useful communication strategies. In this work, we train language models to have productive discussions about their environment in natural language without any human demonstrations. We decompose the communication problem into listening and speaking. Our key idea is to leverage the agent's goal to predict useful information about the world as a dense reward signal that guides communication. Specifically, we improve a model's listening skills by training them to predict information about the environment based on discussions, and we simultaneously improve a model's speaking skills with multi-agent reinforcement learning by rewarding messages based on their influence on other agents. To investigate the role and necessity of communication in complex social settings, we study an embodied social deduction game based on Among Us, where the key question to answer is the identity of an adversarial imposter. We analyze emergent behaviors due to our technique, such as accusing suspects and providing evidence, and find that it enables strong discussions, doubling the win rates compared to standard RL. We release our code and models at this https URL 

**Abstract (ZH)**: 在多智能体环境中使用自然语言进行通信是一种强大的工具，因为它使独立的智能体能够在部分可观测的环境中共享信息，并允许与人类进行零-shot 协调。然而，大多数先前的工作要么依赖大量的人类演示进行训练，要么缺乏生成自然且有用通信策略的能力。在本文中，我们训练语言模型能够在没有人类演示的情况下，以自然语言进行关于环境的有成效的讨论。我们将通信问题分解为倾听和说话两个方面。我们的核心思想是利用智能体的目标来预测对世界有用的资讯作为密集奖励信号，引导通信。具体来说，我们通过训练模型预测基于讨论的环境信息来提高其倾听技能，并通过多智能体强化学习同时提高其说话技能，根据信息对其他智能体的影响来奖励消息。为了研究复杂社会环境中的通信作用和必要性，我们基于《Among Us》建立了一个具身社会推理游戏，其中的关键问题是确定一个敌对 impostor 的身份。我们分析了由于我们的技术而产生的 emergent 行为，如指控嫌疑人和提供证据，并发现这种方法能够促进强有力的讨论，相较于标准 RL，胜率翻倍。我们将在以下网址发布我们的代码和模型：https://xxxxxx。 

---
# MetaChain: A Fully-Automated and Zero-Code Framework for LLM Agents 

**Title (ZH)**: MetaChain：一个全自动且零代码的LLM代理框架 

**Authors**: Jiabin Tang, Tianyu Fan, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05957)  

**Abstract**: Large Language Model (LLM) Agents have demonstrated remarkable capabilities in task automation and intelligent decision-making, driving the widespread adoption of agent development frameworks such as LangChain and AutoGen. However, these frameworks predominantly serve developers with extensive technical expertise - a significant limitation considering that only 0.03 % of the global population possesses the necessary programming skills. This stark accessibility gap raises a fundamental question: Can we enable everyone, regardless of technical background, to build their own LLM agents using natural language alone? To address this challenge, we introduce MetaChain-a Fully-Automated and highly Self-Developing framework that enables users to create and deploy LLM agents through Natural Language Alone. Operating as an autonomous Agent Operating System, MetaChain comprises four key components: i) Agentic System Utilities, ii) LLM-powered Actionable Engine, iii) Self-Managing File System, and iv) Self-Play Agent Customization module. This lightweight yet powerful system enables efficient and dynamic creation and modification of tools, agents, and workflows without coding requirements or manual intervention. Beyond its code-free agent development capabilities, MetaChain also serves as a versatile multi-agent system for General AI Assistants. Comprehensive evaluations on the GAIA benchmark demonstrate MetaChain's effectiveness in generalist multi-agent tasks, surpassing existing state-of-the-art methods. Furthermore, MetaChain's Retrieval-Augmented Generation (RAG)-related capabilities have shown consistently superior performance compared to many alternative LLM-based solutions. 

**Abstract (ZH)**: 大型语言模型（LLM）代理展示了在任务自动化和智能决策方面的显著能力，推动了如LangChain和AutoGen等代理开发框架的广泛应用。然而，这些框架主要服务于具备丰富技术背景的开发者——这是一个显著的限制，因为全球仅有0.03%的人口拥有必要的编程技能。这一明显的可访问性差距引发了一个基本问题：我们能否仅通过自然语言使每个人，无论其技术背景如何，都能够构建自己的LLM代理？为解决这一挑战，我们提出了MetaChain——一个完全自动化且高度自我开发的框架，使用户能够仅通过自然语言创建和部署LLM代理。作为自主的代理操作系统，MetaChain包含四个关键组件：i) 代理系统实用程序，ii) 基于LLM的动作引擎，iii) 自我管理文件系统，和iv) 自我游戏代理自定义模块。这个轻量级且强大的系统能够在没有编码要求或人工干预的情况下，高效且动态地创建和修改工具、代理和工作流程。除了其无代码代理开发能力外，MetaChain还作为一个通用人工智能助手的多功能多代理系统发挥作用。在GAIA基准上的全面评估证明了MetaChain在通用多代理任务中的有效性，超越了现有最先进的方法。此外，MetaChain的相关检索增强生成（RAG）能力在与许多基于LLM的替代方案的性能比较中表现出一致的优越性。 

---
# Knowledge is Power: Harnessing Large Language Models for Enhanced Cognitive Diagnosis 

**Title (ZH)**: 知识即力量：利用大型语言模型进行增强的认知诊断 

**Authors**: Zhiang Dong, Jingyuan Chen, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05556)  

**Abstract**: Cognitive Diagnosis Models (CDMs) are designed to assess students' cognitive states by analyzing their performance across a series of exercises. However, existing CDMs often struggle with diagnosing infrequent students and exercises due to a lack of rich prior knowledge. With the advancement in large language models (LLMs), which possess extensive domain knowledge, their integration into cognitive diagnosis presents a promising opportunity. Despite this potential, integrating LLMs with CDMs poses significant challenges. LLMs are not well-suited for capturing the fine-grained collaborative interactions between students and exercises, and the disparity between the semantic space of LLMs and the behavioral space of CDMs hinders effective integration. To address these issues, we propose a novel Knowledge-enhanced Cognitive Diagnosis (KCD) framework, which is a model-agnostic framework utilizing LLMs to enhance CDMs and compatible with various CDM architectures. The KCD framework operates in two stages: LLM Diagnosis and Cognitive Level Alignment. In the LLM Diagnosis stage, both students and exercises are diagnosed to achieve comprehensive and detailed modeling. In the Cognitive Level Alignment stage, we bridge the gap between the CDMs' behavioral space and the LLMs' semantic space using contrastive learning and mask-reconstruction approaches. Experiments on several real-world datasets demonstrate the effectiveness of our proposed framework. 

**Abstract (ZH)**: 认知增强的认知诊断框架（KCD）：利用大语言模型提升认知诊断模型 

---
# LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning 

**Title (ZH)**: 基于LLM的去中心化生成代理及自适应层次知识图谱协作规划 

**Authors**: Hanqing Yang, Jingdi Chen, Marie Siew, Tania Lorido-Botran, Carlee Joe-Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.05453)  

**Abstract**: Developing intelligent agents for long-term cooperation in dynamic open-world scenarios is a major challenge in multi-agent systems. Traditional Multi-agent Reinforcement Learning (MARL) frameworks like centralized training decentralized execution (CTDE) struggle with scalability and flexibility. They require centralized long-term planning, which is difficult without custom reward functions, and face challenges in processing multi-modal data. CTDE approaches also assume fixed cooperation strategies, making them impractical in dynamic environments where agents need to adapt and plan independently. To address decentralized multi-agent cooperation, we propose Decentralized Adaptive Knowledge Graph Memory and Structured Communication System (DAMCS) in a novel Multi-agent Crafter environment. Our generative agents, powered by Large Language Models (LLMs), are more scalable than traditional MARL agents by leveraging external knowledge and language for long-term planning and reasoning. Instead of fully sharing information from all past experiences, DAMCS introduces a multi-modal memory system organized as a hierarchical knowledge graph and a structured communication protocol to optimize agent cooperation. This allows agents to reason from past interactions and share relevant information efficiently. Experiments on novel multi-agent open-world tasks show that DAMCS outperforms both MARL and LLM baselines in task efficiency and collaboration. Compared to single-agent scenarios, the two-agent scenario achieves the same goal with 63% fewer steps, and the six-agent scenario with 74% fewer steps, highlighting the importance of adaptive memory and structured communication in achieving long-term goals. We publicly release our project at: this https URL. 

**Abstract (ZH)**: 开发智能代理以应对动态开放世界中的长期合作是多agent系统中的重大挑战。传统多agent强化学习（MARL）框架如集中训练分散执行（CTDE）面临可扩展性和灵活性问题。它们需要集中式长期规划，这在缺乏定制奖励函数的情况下很难实现，并且在处理多模态数据方面存在问题。CTDE方法假设固定的合作策略，使其难以在动态环境中适应和独立规划。为了解决分散的多agent合作问题，我们提出了基于新颖多agent Crafter环境的分散自适应知识图谱记忆和结构化通信系统（DAMCS）。我们的生成型agent通过利用外部知识和语言进行长期规划和推理，比传统的MARL agent更具可扩展性。DAMCS引入了一个基于层次知识图谱的多模态记忆系统和一个结构化通信协议，以优化agent之间的合作。这使得agents能够从过去的交互中进行推理并有效共享相关信息。在新提出的多agent开放世界任务上的实验表明，DAMCS在任务效率和协作方面均优于MARL和LLM基线。与单agent场景相比，两agent场景以63%更少的步骤达成目标，六agent场景以74%更少的步骤达成目标，突显了适应性记忆和结构化通信在实现长期目标中的重要性。我们的项目已公开发布：this https URL。 

---
# Agentic AI Systems Applied to tasks in Financial Services: Modeling and model risk management crews 

**Title (ZH)**: 将代理人工智能系统应用于金融服务业的任务：建模与模型风险管理团队 

**Authors**: Izunna Okpala, Ashkan Golgoon, Arjun Ravi Kannan  

**Link**: [PDF](https://arxiv.org/pdf/2502.05439)  

**Abstract**: The advent of large language models has ushered in a new era of agentic systems, where artificial intelligence programs exhibit remarkable autonomous decision-making capabilities across diverse domains. This paper explores agentic system workflows in the financial services industry. In particular, we build agentic crews that can effectively collaborate to perform complex modeling and model risk management (MRM) tasks. The modeling crew consists of a manager and multiple agents who perform specific tasks such as exploratory data analysis, feature engineering, model selection, hyperparameter tuning, model training, model evaluation, and writing documentation. The MRM crew consists of a manager along with specialized agents who perform tasks such as checking compliance of modeling documentation, model replication, conceptual soundness, analysis of outcomes, and writing documentation. We demonstrate the effectiveness and robustness of modeling and MRM crews by presenting a series of numerical examples applied to credit card fraud detection, credit card approval, and portfolio credit risk modeling datasets. 

**Abstract (ZH)**: 大型语言模型的出现开启了代理系统的新时代，人工智能程序在多种领域展现出显著的自主决策能力。本文探讨了代理系统在金融服务行业的工作流程。特别是，我们构建了能够有效协作以执行复杂建模和模型风险管理（MRM）任务的代理队伍。建模团队由一名经理和多个执行特定任务（如探索性数据分析、特征工程、模型选择、超参数调整、模型训练、模型评估和编写文档）的代理组成。MRM团队由一名经理和执行合规检查、模型复制、概念合理性分析、结果分析和编写文档等任务的专业代理组成。通过一系列应用于信用卡欺诈检测、信用卡审批和投资组合信用风险管理数据集的数值示例，我们证明了建模和MRM团队的有效性和稳健性。 

---
# Matryoshka Quantization 

**Title (ZH)**: 套娃量化 

**Authors**: Pranav Nair, Puranjay Datta, Jeff Dean, Prateek Jain, Aditya Kusupati  

**Link**: [PDF](https://arxiv.org/pdf/2502.06786)  

**Abstract**: Quantizing model weights is critical for reducing the communication and inference costs of large models. However, quantizing models -- especially to low precisions like int4 or int2 -- requires a trade-off in model quality; int2, in particular, is known to severely degrade model quality. Consequently, practitioners are often forced to maintain multiple models with different quantization levels or serve a single model that best satisfies the quality-latency trade-off. On the other hand, integer data types, such as int8, inherently possess a nested (Matryoshka) structure where smaller bit-width integers, like int4 or int2, are nested within the most significant bits. This paper proposes Matryoshka Quantization (MatQuant), a novel multi-scale quantization technique that addresses the challenge of needing multiple quantized models. It allows training and maintaining just one model, which can then be served at different precision levels. Furthermore, due to the co-training and co-distillation regularization provided by MatQuant, the int2 precision models extracted by MatQuant can be up to $10\%$ more accurate than standard int2 quantization (using techniques like QAT or OmniQuant). This represents significant progress in model quantization, demonstrated by the fact that, with the same recipe, an int2 FFN-quantized Gemma-2 9B model is more accurate than an int8 FFN-quantized Gemma-2 2B model. 

**Abstract (ZH)**: Matryoshka量化：一种新型多尺度量化解决策 methodology 

---
# Gradient Multi-Normalization for Stateless and Scalable LLM Training 

**Title (ZH)**: 无状态和可扩展的大语言模型训练的梯度多归一化 

**Authors**: Meyer Scetbon, Chao Ma, Wenbo Gong, Edward Meeds  

**Link**: [PDF](https://arxiv.org/pdf/2502.06742)  

**Abstract**: Training large language models (LLMs) typically relies on adaptive optimizers like Adam (Kingma & Ba, 2015) which store additional state information to accelerate convergence but incur significant memory overhead. Recent efforts, such as SWAN (Ma et al., 2024) address this by eliminating the need for optimizer states while achieving performance comparable to Adam via a multi-step preprocessing procedure applied to instantaneous gradients. Motivated by the success of SWAN, we introduce a novel framework for designing stateless optimizers that normalizes stochastic gradients according to multiple norms. To achieve this, we propose a simple alternating scheme to enforce the normalization of gradients w.r.t these norms. We show that our procedure can produce, up to an arbitrary precision, a fixed-point of the problem, and that SWAN is a particular instance of our approach with carefully chosen norms, providing a deeper understanding of its design. However, SWAN's computationally expensive whitening/orthogonalization step limit its practicality for large LMs. Using our principled perspective, we develop of a more efficient, scalable, and practical stateless optimizer. Our algorithm relaxes the properties of SWAN, significantly reducing its computational cost while retaining its memory efficiency, making it applicable to training large-scale models. Experiments on pre-training LLaMA models with up to 1 billion parameters demonstrate a 3X speedup over Adam with significantly reduced memory requirements, outperforming other memory-efficient baselines. 

**Abstract (ZH)**: 一种基于多范数规范化设计的状态less优化器框架 

---
# Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining 

**Title (ZH)**: 基于动态损失的样本加权重估以改进大规模语言模型预训练 

**Authors**: Daouda Sow, Herbert Woisetschläger, Saikiran Bulusu, Shiqiang Wang, Hans-Arno Jacobsen, Yingbin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06733)  

**Abstract**: Pretraining large language models (LLMs) on vast and heterogeneous datasets is crucial for achieving state-of-the-art performance across diverse downstream tasks. However, current training paradigms treat all samples equally, overlooking the importance or relevance of individual samples throughout the training process. Existing reweighting strategies, which primarily focus on group-level data importance, fail to leverage fine-grained instance-level information and do not adapt dynamically to individual sample importance as training progresses. In this paper, we introduce novel algorithms for dynamic, instance-level data reweighting aimed at improving both the efficiency and effectiveness of LLM pretraining. Our methods adjust the weight of each training sample based on its loss value in an online fashion, allowing the model to dynamically focus on more informative or important samples at the current training stage. In particular, our framework allows us to systematically devise reweighting strategies deprioritizing redundant or uninformative data, which we find tend to work best. Furthermore, we develop a new theoretical framework for analyzing the impact of loss-based reweighting on the convergence of gradient-based optimization, providing the first formal characterization of how these strategies affect convergence bounds. We empirically validate our approach across a spectrum of tasks, from pretraining 7B and 1.4B parameter LLMs to smaller-scale language models and linear regression problems, demonstrating that our loss-based reweighting approach can lead to faster convergence and significantly improved performance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在海量异构数据集上的预训练对于实现跨多种下游任务的最先进性能至关重要。然而，当前的训练范式将所有样本视为等价的，忽视了训练过程中各个样本的重要性或相关性。现有的重加权策略主要集中在组级数据重要性上，未能充分利用细粒度的实例级信息，也不具备在训练过程中动态适应各个样本重要性的能力。在本文中，我们引入了新的动态实例级数据重加权重法，旨在提高大规模语言模型预训练的效率和效果。我们的方法根据每个训练样本在当前训练阶段的损失值在线调整其权重，使模型能够动态地关注更具信息量或更重要的样本。此外，我们开发了新的理论框架来分析基于损失的重加权对梯度优化收敛的影响，提供了这些策略如何影响收敛界的第一个形式化刻画。我们在从7B和1.4B参数的大规模语言模型到较小规模的语言模型和线性回归问题的多种任务中实验性地验证了我们的方法，证明了基于损失的重加权方法可以加速收敛并显著提高性能。 

---
# Boosting Self-Efficacy and Performance of Large Language Models via Verbal Efficacy Stimulations 

**Title (ZH)**: 通过言语效能激发增强大型语言模型的效能与自 efficacy 

**Authors**: Rui Chen, Tailai Peng, Xinran Xie, Dekun Lin, Zhe Cui, Zheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.06669)  

**Abstract**: Significant improvements have been observed in the zero-shot capabilities of the Large Language Models (LLMs). Due to their high sensitivity to input, research has increasingly focused on enhancing LLMs' performance via direct and simple prompt engineering rather than intricate domain adaptation. Studies suggest that LLMs exhibit emotional intelligence, and both positive and negative emotions can potentially enhance task performances. However, prior interaction prompts have predominantly concentrated on a single stimulus type, neglecting to compare different stimulus effects, examine the influence of varying task difficulties, or explore underlying mechanisms. This paper, inspired by the positive correlation between self-efficacy and task performance within the social cognitive theory, introduces Verbal Efficacy Stimulations (VES). Our VES comprises three types of verbal prompts: encouraging, provocative, and critical, addressing six aspects such as helpfulness and competence. And we further categorize task difficulty, aiming to extensively investigate how distinct VES influence the self-efficacy and task achievements of language models at varied levels of difficulty. The experimental results show that the three types of VES improve the performance of LLMs on most tasks, and the most effective VES varies for different models. In extensive experiments, we have obtained some findings consistent with psychological theories, providing novel insights for future research. 

**Abstract (ZH)**: 大型语言模型的零样本能力取得了显著提升。由于其对输入的高度敏感性，研究越来越倾向于通过直接简单的提示工程来增强大型语言模型的表现，而不是通过复杂的领域适应。研究表明，大型语言模型表现出情感智能，正向和负向情感都可能增强任务表现。然而，先前的交互提示大多集中在单一刺激类型上，未比较不同刺激效果、探讨任务难度变化的影响或探究其背后的机制。受社会认知理论中自我效能与任务表现之间正相关关系的启发，本文引入了言语效能刺激（VES）。我们的VES包括三种类型的口头提示：鼓励性、引发性、和批判性，涉及诸如帮助性和能力等六个方面。此外，我们进一步对任务难度进行了分类，旨在在不同难度水平上广泛探讨不同VES如何影响语言模型的自我效能和任务成就。实验结果显示，三种类型的VES在大多数任务中提高了大型语言模型的表现，而最有效的VES因模型而异。在广泛的实验中，我们获得了一些与心理学理论一致的发现，为未来的研究提供了新的见解。 

---
# Automatic Evaluation of Healthcare LLMs Beyond Question-Answering 

**Title (ZH)**: 自动评估 healthcare LLMs 超越问答任务 

**Authors**: Anna Arias-Duart, Pablo Agustin Martin-Torres, Daniel Hinjos, Pablo Bernabeu-Perez, Lucia Urcelay Ganzabal, Marta Gonzalez Mallo, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Sergio Alvarez-Napagao, Dario Garcia-Gasulla  

**Link**: [PDF](https://arxiv.org/pdf/2502.06666)  

**Abstract**: Current Large Language Models (LLMs) benchmarks are often based on open-ended or close-ended QA evaluations, avoiding the requirement of human labor. Close-ended measurements evaluate the factuality of responses but lack expressiveness. Open-ended capture the model's capacity to produce discourse responses but are harder to assess for correctness. These two approaches are commonly used, either independently or together, though their relationship remains poorly understood. This work is focused on the healthcare domain, where both factuality and discourse matter greatly. It introduces a comprehensive, multi-axis suite for healthcare LLM evaluation, exploring correlations between open and close benchmarks and metrics. Findings include blind spots and overlaps in current methodologies. As an updated sanity check, we release a new medical benchmark--CareQA--, with both open and closed variants. Finally, we propose a novel metric for open-ended evaluations --Relaxed Perplexity-- to mitigate the identified limitations. 

**Abstract (ZH)**: 当前的大语言模型（LLMs）基准常常基于开放性或封闭性的问答评估，避免了人力投入。封闭性评估衡量响应的事实性，但缺乏表达力。开放性评估捕捉模型生成论述响应的能力，但对其正确性评估更加困难。这两种方法要么独立使用，要么一起使用，但它们之间的关系仍不清楚。本工作聚焦于医疗健康领域，其中事实性和论述都非常重要。它引入了一个全面的多维度评估套件，探索开放性和封闭性基准和度量之间的相关性。研究发现包括当前方法的盲点和重叠。作为更新的合理性检验，我们发布了新的医疗基准——CareQA——，包括开放性和封闭性变体。最后，我们提出了一种新的开放性评估指标——松弛困惑度——以缓解识别出的局限性。 

---
# Steel-LLM:From Scratch to Open Source -- A Personal Journey in Building a Chinese-Centric LLM 

**Title (ZH)**: Steel-LLM：从零开始到开源——构建以中文为中心的LLM的个人历程 

**Authors**: Qingshui Gu, Shu Li, Tianyu Zheng, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06635)  

**Abstract**: Steel-LLM is a Chinese-centric language model developed from scratch with the goal of creating a high-quality, open-source model despite limited computational resources. Launched in March 2024, the project aimed to train a 1-billion-parameter model on a large-scale dataset, prioritizing transparency and the sharing of practical insights to assist others in the community. The training process primarily focused on Chinese data, with a small proportion of English data included, addressing gaps in existing open-source LLMs by providing a more detailed and practical account of the model-building journey. Steel-LLM has demonstrated competitive performance on benchmarks such as CEVAL and CMMLU, outperforming early models from larger institutions. This paper provides a comprehensive summary of the project's key contributions, including data collection, model design, training methodologies, and the challenges encountered along the way, offering a valuable resource for researchers and practitioners looking to develop their own LLMs. The model checkpoints and training script are available at this https URL. 

**Abstract (ZH)**: Steel-LLM是一种以中文为中心的从零开始开发的语言模型，旨在利用有限的计算资源创建高质量的开源模型。该项目于2024年3月启动，目标是在大规模数据集上训练一个10亿参数的模型，注重透明度与实用见解的分享，以帮助社区中的其他人。训练过程主要侧重于中文数据，同时包含少量英文数据，通过提供详细的模型构建过程，弥补现有开源语言模型的不足。Steel-LLM在CEVAL和CMMLU等基准测试中表现出竞争性性能，超越了大型机构的早期模型。本文提供了该项目关键贡献的全面总结，包括数据收集、模型设计、训练方法以及遇到的挑战，为希望开发自己语言模型的研究人员和实践者提供了宝贵的资源。模型检查点和训练脚本可在以下链接获取：this https URL。 

---
# Combining Large Language Models with Static Analyzers for Code Review Generation 

**Title (ZH)**: 结合大型语言模型与静态分析器进行代码审查生成 

**Authors**: Imen Jaoua, Oussama Ben Sghaier, Houari Sahraoui  

**Link**: [PDF](https://arxiv.org/pdf/2502.06633)  

**Abstract**: Code review is a crucial but often complex, subjective, and time-consuming activity in software development. Over the past decades, significant efforts have been made to automate this process. Early approaches focused on knowledge-based systems (KBS) that apply rule-based mechanisms to detect code issues, providing precise feedback but struggling with complex, context-dependent cases. More recent work has shifted toward fine-tuning pre-trained language models for code review, enabling broader issue coverage but often at the expense of precision. In this paper, we propose a hybrid approach that combines the strengths of KBS and learning-based systems (LBS) to generate high-quality, comprehensive code reviews. Our method integrates knowledge at three distinct stages of the language model pipeline: during data preparation (Data-Augmented Training, DAT), at inference (Retrieval-Augmented Generation, RAG), and after inference (Naive Concatenation of Outputs, NCO). We empirically evaluate our combination strategies against standalone KBS and LBS fine-tuned on a real-world dataset. Our results show that these hybrid strategies enhance the relevance, completeness, and overall quality of review comments, effectively bridging the gap between rule-based tools and deep learning models. 

**Abstract (ZH)**: 代码审查是软件开发中一个关键但常常复杂、主观且耗时的活动。在过去几十年中，已经做了大量的努力来自动化这一过程。早期的方法集中于基于知识系统的(KBS)应用规则机制来检测代码问题，提供精准的反馈，但在处理复杂的、具有上下文依赖性的情况时存在困难。近年来的工作则转向通过微调预训练语言模型来进行代码审查，能够覆盖更广泛的问题，但在精度方面往往有所妥协。本文提出了一种混合方法，结合基于知识系统的(KBS)和基于学习系统的(LBS)的优点，以生成高质量、全面的代码审查评论。我们的方法在语言模型管道的三个不同阶段整合了知识：数据准备阶段（数据增强训练，DAT）、推理阶段（检索增强生成，RAG）和推理后阶段（输出的朴素拼接，NCO）。我们实证评估了这些组合策略与独立的KBS和LBS（在真实数据集上微调）的效果。我们发现这些混合策略提高了审查评论的相关性、完整性和整体质量，有效地弥合了基于规则的工具和深度学习模型之间的差距。 

---
# Hephaestus: Improving Fundamental Agent Capabilities of Large Language Models through Continual Pre-Training 

**Title (ZH)**: Hephaestus: 通过持续预训练提升大型语言模型的基本代理能力 

**Authors**: Yuchen Zhuang, Jingfeng Yang, Haoming Jiang, Xin Liu, Kewei Cheng, Sanket Lokegaonkar, Yifan Gao, Qing Ping, Tianyi Liu, Binxuan Huang, Zheng Li, Zhengyang Wang, Pei Chen, Ruijie Wang, Rongzhi Zhang, Nasser Zalmout, Priyanka Nigam, Bing Yin, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06589)  

**Abstract**: Due to the scarcity of agent-oriented pre-training data, LLM-based autonomous agents typically rely on complex prompting or extensive fine-tuning, which often fails to introduce new capabilities while preserving strong generalizability. We introduce Hephaestus-Forge, the first large-scale pre-training corpus designed to enhance the fundamental capabilities of LLM agents in API function calling, intrinsic reasoning and planning, and adapting to environmental feedback. Hephaestus-Forge comprises 103B agent-specific data encompassing 76,537 APIs, including both tool documentation to introduce knowledge of API functions and function calling trajectories to strengthen intrinsic reasoning. To explore effective training protocols, we investigate scaling laws to identify the optimal recipe in data mixing ratios. By continual pre-training on Hephaestus-Forge, Hephaestus outperforms small- to medium-scale open-source LLMs and rivals commercial LLMs on three agent benchmarks, demonstrating the effectiveness of our pre-training corpus in enhancing fundamental agentic capabilities and generalization of LLMs to new tasks or environments. 

**Abstract (ZH)**: 由于缺乏以代理为核心的预训练数据，基于LLM的自主代理通常依赖于复杂的提示或广泛的微调，这往往无法引入新能力同时保持强大的泛化能力。我们介绍了Hephaestus-Forge，这是首个大规模预训练语料库，旨在增强LLM代理的API函数调用、内在推理和规划能力，并使其能够适应环境反馈。Hephaestus-Forge包含103亿个特定于代理的数据，涵盖了76,537个API，包括工具文档以引入API功能的知识，以及功能调用轨迹以加强内在推理。为探索有效的训练协议，我们研究了标度定律以确定最优的数据混合比例。通过持续在Hephaestus-Forge上预训练，Hephaestus在三个代理基准测试中优于小型到中型的开源LLM，并与商用LLM相竞争，证明了我们预训练语料库在增强代理基本能力以及使LLM泛化到新任务或环境方面的有效性。 

---
# LawGPT: Knowledge-Guided Data Generation and Its Application to Legal LLM 

**Title (ZH)**: LawGPT：知识引导的数据生成及其在法律LLM中的应用 

**Authors**: Zhi Zhou, Kun-Yang Yu, Shi-Yu Tian, Jiang-Xin Shi, Xiao-Wen Yang, Pengxiao Song, Yi-Xuan Jin, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.06572)  

**Abstract**: Large language models (LLMs), both proprietary and open-source, have demonstrated remarkable capabilities across various natural language processing tasks. However, they face significant limitations in legal reasoning tasks. Proprietary models introduce data privacy risks and high inference costs, while open-source models underperform due to insufficient legal domain training data. To address these limitations, we study data generation for legal reasoning to improve the legal reasoning performance of open-source LLMs with the help of proprietary LLMs. This is challenging due to the lack of legal knowledge in proprietary LLMs and the difficulty in verifying the generated data. We propose KgDG, a knowledge-guided data generation framework for legal reasoning. Our framework enables leveraging legal knowledge to enhance generation diversity and introduces a refinement and verification process to ensure the quality of generated data. Moreover, we expand the generated dataset to further enhance the LLM reasoning capabilities. Using KgDG, we create a synthetic legal reasoning dataset containing 50K high-quality examples. Our trained model LawGPT outperforms existing legal-specific LLMs and achieves performance comparable to proprietary LLMs, demonstrating the effectiveness of KgDG and LawGPT. Our code and resources is publicly available at this https URL . 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理任务中展现了 remarkable 能力，但在法律推理任务中面临显著限制。 proprietary 模型引入了数据隐私风险和高昂的推理成本，而 open-source 模型则由于缺乏足够的法律领域训练数据而表现不佳。为了克服这些限制，我们研究了法律推理的数据生成方法，旨在通过 proprietary LLMs 提高 open-source LLMs 的法律推理性能。由于 proprietary LLMs 缺乏法律知识且生成数据的验证难度大，这一过程颇具挑战性。我们提出了一种名为 KgDG 的知识指导型数据生成框架，用于法律推理。该框架利用法律知识以增强生成多样性，并引入了一个改进和验证过程，以确保生成数据的质量。此外，我们还扩大了生成的数据集，以进一步提升 LLM 的推理能力。通过 KgDG，我们创建了一个包含 50K 高质量示例的合成法律推理数据集。我们训练的模型 LawGPT 在法律特定任务上的表现超越了现有模型，并达到了与 proprietary LLMs 相媲美的水平，证明了 KgDG 和 LawGPT 的有效性。我们的代码和资源已公开于此 https URL 。 

---
# GuideLLM: Exploring LLM-Guided Conversation with Applications in Autobiography Interviewing 

**Title (ZH)**: GuideLLM：探索由大语言模型引导的对话及其在自传访谈中的应用 

**Authors**: Jinhao Duan, Xinyu Zhao, Zhuoxuan Zhang, Eunhye Ko, Lily Boddy, Chenan Wang, Tianhao Li, Alexander Rasgon, Junyuan Hong, Min Kyung Lee, Chenxi Yuan, Qi Long, Ying Ding, Tianlong Chen, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.06494)  

**Abstract**: Although Large Language Models (LLMs) succeed in human-guided conversations such as instruction following and question answering, the potential of LLM-guided conversations-where LLMs direct the discourse and steer the conversation's objectives-remains under-explored. In this study, we first characterize LLM-guided conversation into three fundamental components: (i) Goal Navigation; (ii) Context Management; (iii) Empathetic Engagement, and propose GuideLLM as an installation. We then implement an interviewing environment for the evaluation of LLM-guided conversation. Specifically, various topics are involved in this environment for comprehensive interviewing evaluation, resulting in around 1.4k turns of utterances, 184k tokens, and over 200 events mentioned during the interviewing for each chatbot evaluation. We compare GuideLLM with 6 state-of-the-art LLMs such as GPT-4o and Llama-3-70b-Instruct, from the perspective of interviewing quality, and autobiography generation quality. For automatic evaluation, we derive user proxies from multiple autobiographies and employ LLM-as-a-judge to score LLM behaviors. We further conduct a human-involved experiment by employing 45 human participants to chat with GuideLLM and baselines. We then collect human feedback, preferences, and ratings regarding the qualities of conversation and autobiography. Experimental results indicate that GuideLLM significantly outperforms baseline LLMs in automatic evaluation and achieves consistent leading performances in human ratings. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在指令跟随和问答等人指导的对话中取得成功，但LLM引导的对话——其中LLMs主导对话并引导对话目标——的潜力仍被低估。在本研究中，我们首先将LLM引导的对话分为三个基本组件：（i）目标导航；（ii）上下文管理；（iii）同理心参与，并提出GuideLLM作为一项安装。然后，我们实现了一个面试环境来评估LLM引导的对话。具体来说，该环境中包含多种话题，以进行全面的面试评估，从而产生约1400次话语轮次、18.4万个令牌以及每次聊天机器人都提到了超过200个事件。我们从访谈质量、自传生成质量的角度将GuideLLM与诸如GPT-4o和Llama-3-70b-Instruct等6种最先进的LLM进行比较。对于自动评估，我们从多篇自传中提取用户代理，并使用LLM作为评判员来评分LLM行为。我们还通过45名人类参与者与GuideLLM和基线模型进行互动，并收集了关于对话和自传品质的人类反馈、偏好和评分。实验结果表明，在自动评估中GuideLLM显著优于基线LLM，在人类评分中也表现出一致的领先性能。 

---
# Recent Advances in Discrete Speech Tokens: A Review 

**Title (ZH)**: 近期离散语音令牌的研究进展：一个综述 

**Authors**: Yiwei Guo, Zhihan Li, Hankun Wang, Bohan Li, Chongtian Shao, Hanglei Zhang, Chenpeng Du, Xie Chen, Shujie Liu, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.06490)  

**Abstract**: The rapid advancement of speech generation technologies in the era of large language models (LLMs) has established discrete speech tokens as a foundational paradigm for speech representation. These tokens, characterized by their discrete, compact, and concise nature, are not only advantageous for efficient transmission and storage, but also inherently compatible with the language modeling framework, enabling seamless integration of speech into text-dominated LLM architectures. Current research categorizes discrete speech tokens into two principal classes: acoustic tokens and semantic tokens, each of which has evolved into a rich research domain characterized by unique design philosophies and methodological approaches. This survey systematically synthesizes the existing taxonomy and recent innovations in discrete speech tokenization, conducts a critical examination of the strengths and limitations of each paradigm, and presents systematic experimental comparisons across token types. Furthermore, we identify persistent challenges in the field and propose potential research directions, aiming to offer actionable insights to inspire future advancements in the development and application of discrete speech tokens. 

**Abstract (ZH)**: 大型语言模型时代语音生成技术的迅猛发展确立了离散语音令牌作为语音表示的基本范式。这些令牌以其离散性、紧凑性和简明性为特点，不仅有利于高效传输和存储，还与语言模型框架天然兼容，从而无缝地将语音整合到以文本为主导的大型语言模型架构中。当前研究将离散语音令牌划分为两类主要类别：声学令牌和语义令牌，每类均已发展成为具有独特设计哲学和方法论方法的丰富研究领域。本文综述系统地总结了离散语音分词的现有分类和近期创新，对每种范式的优缺点进行了批判性评估，并进行了跨令牌类型的系统实验比较。此外，我们识别了该领域中持续存在的挑战，并提出了潜在的研究方向，旨在提供可操作的见解，激发未来离散语音令牌开发和应用的进步。 

---
# KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment 

**Title (ZH)**: KARMA：利用多智能体大语言模型进行自动知识图谱丰富化 

**Authors**: Yuxing Lu, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06472)  

**Abstract**: Maintaining comprehensive and up-to-date knowledge graphs (KGs) is critical for modern AI systems, but manual curation struggles to scale with the rapid growth of scientific literature. This paper presents KARMA, a novel framework employing multi-agent large language models (LLMs) to automate KG enrichment through structured analysis of unstructured text. Our approach employs nine collaborative agents, spanning entity discovery, relation extraction, schema alignment, and conflict resolution that iteratively parse documents, verify extracted knowledge, and integrate it into existing graph structures while adhering to domain-specific schema. Experiments on 1,200 PubMed articles from three different domains demonstrate the effectiveness of KARMA in knowledge graph enrichment, with the identification of up to 38,230 new entities while achieving 83.1\% LLM-verified correctness and reducing conflict edges by 18.6\% through multi-layer assessments. 

**Abstract (ZH)**: 维持全面和实时更新的知识图谱对于现代AI系统至关重要，但手动整理难以应对科学文献的快速增长。本文提出KARMA，这是一种利用多智能体大型语言模型（LLMs）通过结构化分析非结构化文本自动化扩展知识图谱的新框架。我们的方法采用九个协作智能体，涵盖实体发现、关系提取、模式对齐和冲突解决，迭代解析文档、验证提取的知识，并将其整合到现有图结构中，同时遵循领域特定的模式。在三个不同领域的1,200篇PubMed文章上的实验展示了KARMA在知识图谱扩展中的有效性，识别出多达38,230个新实体，LLM验证正确率达到83.1%，并通过多层评估减少了18.6%的冲突边。 

---
# A Survey of Theory of Mind in Large Language Models: Evaluations, Representations, and Safety Risks 

**Title (ZH)**: 大型语言模型中理论共情的研究：评估、表示与安全风险 

**Authors**: Hieu Minh "Jord" Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2502.06470)  

**Abstract**: Theory of Mind (ToM), the ability to attribute mental states to others and predict their behaviour, is fundamental to social intelligence. In this paper, we survey studies evaluating behavioural and representational ToM in Large Language Models (LLMs), identify important safety risks from advanced LLM ToM capabilities, and suggest several research directions for effective evaluation and mitigation of these risks. 

**Abstract (ZH)**: 理论心智（ToM）是社会科学智能的基础，其能力在于赋予他人心理状态并预测其行为。本文综述了对大型语言模型（LLM）的 behavioural 和 representational ToM 的研究，识别了高级 LLM ToM 能力带来的重要安全风险，并提出了一些有效评估和缓解这些风险的研究方向。 

---
# MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations 

**Title (ZH)**: MATH-Perturb: 评估LLMs在面对难扰动时的数学推理能力 

**Authors**: Kaixuan Huang, Jiacheng Guo, Zihao Li, Xiang Ji, Jiawei Ge, Wenzhe Li, Yingqing Guo, Tianle Cai, Hui Yuan, Runzhe Wang, Yue Wu, Ming Yin, Shange Tang, Yangsibo Huang, Chi Jin, Xinyun Chen, Chiyuan Zhang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06453)  

**Abstract**: Large language models have demonstrated impressive performance on challenging mathematical reasoning tasks, which has triggered the discussion of whether the performance is achieved by true reasoning capability or memorization. To investigate this question, prior work has constructed mathematical benchmarks when questions undergo simple perturbations -- modifications that still preserve the underlying reasoning patterns of the solutions. However, no work has explored hard perturbations, which fundamentally change the nature of the problem so that the original solution steps do not apply. To bridge the gap, we construct MATH-P-Simple and MATH-P-Hard via simple perturbation and hard perturbation, respectively. Each consists of 279 perturbed math problems derived from level-5 (hardest) problems in the MATH dataset (Hendrycksmath et. al., 2021). We observe significant performance drops on MATH-P-Hard across various models, including o1-mini (-16.49%) and gemini-2.0-flash-thinking (-12.9%). We also raise concerns about a novel form of memorization where models blindly apply learned problem-solving skills without assessing their applicability to modified contexts. This issue is amplified when using original problems for in-context learning. We call for research efforts to address this challenge, which is critical for developing more robust and reliable reasoning models. 

**Abstract (ZH)**: 大型语言模型在具有挑战性的数学推理任务上展示了 impressive 的表现，这引发了关于性能是通过真正的推理能力还是记忆实现的讨论。为了探讨这一问题，先前的研究通过简单的扰动（仍然保留解决方案基础推理模式的小修改）构建了数学基准。然而，尚未有研究探索本质扰动，这种扰动从根本上改变了问题的性质，使原始解题步骤不再适用。为了弥合这一差距，我们通过简单的扰动构建了 MATH-P-Simple，通过本质扰动构建了 MATH-P-Hard。两者分别包含源自 MATH 数据集（Hendrycksmath et. al., 2021）最难级别（level-5）的 279 个问题的扰动数学问题。我们观察到在 MATH-P-Hard 上各种模型的表现有显著下降，包括 o1-mini (-16.49%) 和 gemini-2.0-flash-thinking (-12.9%)。我们还对一种新型的记忆化现象表示担忧，即模型盲目应用学习到的解题技巧而无视其对修改后环境的适用性。当使用原始问题进行上下文学习时，这一问题更为突出。我们呼吁研究努力来应对这一挑战，这对于开发更具鲁棒性和可靠性的推理模型至关重要。 

---
# FEMBA: Efficient and Scalable EEG Analysis with a Bidirectional Mamba Foundation Model 

**Title (ZH)**: FEMBA: 高效可扩展的基于双方向Mamba基础模型的脑电图分析 

**Authors**: Anna Tegon, Thorir Mar Ingolfsson, Xiaying Wang, Luca Benini, Yawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.06438)  

**Abstract**: Accurate and efficient electroencephalography (EEG) analysis is essential for detecting seizures and artifacts in long-term monitoring, with applications spanning hospital diagnostics to wearable health devices. Robust EEG analytics have the potential to greatly improve patient care. However, traditional deep learning models, especially Transformer-based architectures, are hindered by their quadratic time and memory complexity, making them less suitable for resource-constrained environments. To address these challenges, we present FEMBA (Foundational EEG Mamba + Bidirectional Architecture), a novel self-supervised framework that establishes new efficiency benchmarks for EEG analysis through bidirectional state-space modeling. Unlike Transformer-based models, which incur quadratic time and memory complexity, FEMBA scales linearly with sequence length, enabling more scalable and efficient processing of extended EEG recordings. Trained on over 21,000 hours of unlabeled EEG and fine-tuned on three downstream tasks, FEMBA achieves competitive performance in comparison with transformer models, with significantly lower computational cost. Specifically, it reaches 81.82% balanced accuracy (0.8921 AUROC) on TUAB and 0.949 AUROC on TUAR, while a tiny 7.8M-parameter variant demonstrates viability for resource-constrained devices. These results pave the way for scalable, general-purpose EEG analytics in both clinical and highlight FEMBA as a promising candidate for wearable applications. 

**Abstract (ZH)**: FEMBA：面向EEG分析的双向架构 

---
# Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs 

**Title (ZH)**: 基于零知识证明和大语言模型的隐私保护个性化建议生成 

**Authors**: Hiroki Watanabe, Motonobu Uchikoshi  

**Link**: [PDF](https://arxiv.org/pdf/2502.06425)  

**Abstract**: Large language models (LLMs) are increasingly utilized in domains such as finance, healthcare, and interpersonal relationships to provide advice tailored to user traits and contexts. However, this personalization often relies on sensitive data, raising critical privacy concerns and necessitating data minimization. To address these challenges, we propose a framework that integrates zero-knowledge proof (ZKP) technology, specifically zkVM, with LLM-based chatbots. This integration enables privacy-preserving data sharing by verifying user traits without disclosing sensitive information. Our research introduces both an architecture and a prompting strategy for this approach. Through empirical evaluation, we clarify the current constraints and performance limitations of both zkVM and the proposed prompting strategy, thereby demonstrating their practical feasibility in real-world scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）在金融、医疗和人际关系等领域日益被用于提供个性化的建议，但这种个性化往往依赖于敏感数据，从而引发了重要的隐私问题，需要减少数据暴露。为应对这些挑战，我们提出了一种将零知识证明（ZKP）技术，特别是zkVM，与基于LLM的聊天机器人相结合的框架。这种集成能够通过验证用户特征而不披露敏感信息来实现隐私保护的数据共享。我们的研究介绍了这种方法的体系结构和提示策略。通过实证评估，我们明确了zkVM和提议的提示策略的当前限制和性能局限性，从而证明了它们在实际场景中的实用性。 

---
# Systematic Outliers in Large Language Models 

**Title (ZH)**: 大型语言模型中的系统性离群值 

**Authors**: Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06415)  

**Abstract**: Outliers have been widely observed in Large Language Models (LLMs), significantly impacting model performance and posing challenges for model compression. Understanding the functionality and formation mechanisms of these outliers is critically important. Existing works, however, largely focus on reducing the impact of outliers from an algorithmic perspective, lacking an in-depth investigation into their causes and roles. In this work, we provide a detailed analysis of the formation process, underlying causes, and functions of outliers in LLMs. We define and categorize three types of outliers-activation outliers, weight outliers, and attention outliers-and analyze their distributions across different dimensions, uncovering inherent connections between their occurrences and their ultimate influence on the attention mechanism. Based on these observations, we hypothesize and explore the mechanisms by which these outliers arise and function, demonstrating through theoretical derivations and experiments that they emerge due to the self-attention mechanism's softmax operation. These outliers act as implicit context-aware scaling factors within the attention mechanism. As these outliers stem from systematic influences, we term them systematic outliers. Our study not only enhances the understanding of Transformer-based LLMs but also shows that structurally eliminating outliers can accelerate convergence and improve model compression. The code is avilable at this https URL. 

**Abstract (ZH)**: 大型语言模型中异常值的形成机制、本质原因及功能研究：基于自注意力机制的系统性异常值分析及影响 

---
# AiRacleX: Automated Detection of Price Oracle Manipulations via LLM-Driven Knowledge Mining and Prompt Generation 

**Title (ZH)**: AiRacleX：通过LLM驱动的知识挖掘和提示生成自动检测价格预言机操纵 

**Authors**: Bo Gao, Yuan Wang, Qingsong Wei, Yong Liu, Rick Siow Mong Goh  

**Link**: [PDF](https://arxiv.org/pdf/2502.06348)  

**Abstract**: Decentralized finance applications depend on accurate price oracles to ensure secure transactions, yet these oracles are highly vulnerable to manipulation, enabling attackers to exploit smart contract vulnerabilities for unfair asset valuation and financial gain. Detecting such manipulations traditionally relies on the manual effort of experienced experts, presenting significant challenges. In this paper, we propose a novel LLM-driven framework that automates the detection of price oracle manipulations by leveraging the complementary strengths of different LLM models. Our approach begins with domain-specific knowledge extraction, where an LLM model synthesizes precise insights about price oracle vulnerabilities from top-tier academic papers, eliminating the need for profound expertise from developers or auditors. This knowledge forms the foundation for a second LLM model to generate structured, context-aware chain of thought prompts, which guide a third LLM model in accurately identifying manipulation patterns in smart contracts. We validate the framework effectiveness through experiments on 60 known vulnerabilities from 46 real-world DeFi attacks or projects spanning 2021 to 2023. The best performing combination of LLMs (Haiku-Haiku-4o-mini) identified by AiRacleX demonstrate a 2.58-times improvement in recall (0.667 vs 0.259) compared to the state-of-the-art tool GPTScan, while maintaining comparable precision. Furthermore, our framework demonstrates the feasibility of replacing commercial models with open-source alternatives, enhancing privacy and security for developers. 

**Abstract (ZH)**: 去中心化金融应用依赖于准确的价格预言机以确保安全交易，但这些预言机极易受到操纵，使攻击者能够利用智能合约漏洞进行不公平资产估值和财务获利。传统上，检测此类操纵依赖于经验丰富的专家进行手动努力，存在显著挑战。本文提出了一种新颖的基于LLM的框架，通过利用不同LLM模型的优势互补来自动检测价格预言机操纵。该方法始于领域特定知识提取，其中LLM模型从顶级学术论文中合成关于价格预言机漏洞的精准洞察，从而消除开发人员或审计员对深厚专业知识的需求。这些知识构成了第二个LLM模型生成结构化、上下文相关的思维链提示的基础，引导第三个LLM模型准确识别智能合约中的操纵模式。通过2021年至2023年间46个真实-world DeFi攻击或项目中的60个已知漏洞实验验证了该框架的有效性。由AiRacleX选出的最佳LLM组合（Haiku-Haiku-4o-mini）在召回率上提高了2.58倍（0.667 vs 0.259），同时保持了与最先进的工具GPTScan相当的精确度。此外，该框架证明了可以用开源替代品替换商业模型的可行性，增强了开发者的隐私和安全性。 

---
# SeaExam and SeaBench: Benchmarking LLMs with Local Multilingual Questions in Southeast Asia 

**Title (ZH)**: SeaExam 和 SeaBench：基于东南亚本地多语言问题的大型语言模型benchmark研究 

**Authors**: Chaoqun Liu, Wenxuan Zhang, Jiahao Ying, Mahani Aljunied, Anh Tuan Luu, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2502.06298)  

**Abstract**: This study introduces two novel benchmarks, SeaExam and SeaBench, designed to evaluate the capabilities of Large Language Models (LLMs) in Southeast Asian (SEA) application scenarios. Unlike existing multilingual datasets primarily derived from English translations, these benchmarks are constructed based on real-world scenarios from SEA regions. SeaExam draws from regional educational exams to form a comprehensive dataset that encompasses subjects such as local history and literature. In contrast, SeaBench is crafted around multi-turn, open-ended tasks that reflect daily interactions within SEA communities. Our evaluations demonstrate that SeaExam and SeaBench more effectively discern LLM performance on SEA language tasks compared to their translated benchmarks. This highlights the importance of using real-world queries to assess the multilingual capabilities of LLMs. 

**Abstract (ZH)**: 本研究提出了两个新型基准——SeaExam和SeaBench，旨在评估大型语言模型（LLMs）在东南亚（SEA）应用场景中的能力。与主要源自英译的多语言数据集不同，这些基准是基于东南亚地区的实际应用场景构建的。SeaExam从区域教育考试中抽取数据，形成了涵盖地方历史和文学等主题的全面数据集。相比之下，SeaBench围绕多轮、开放式的任务构建，反映了SEA社区中的日常互动。我们的评估表明，SeaExam和SeaBench比其翻译基准更能有效地区分LLM在SEA语言任务中的表现，这突显了使用真实查询来评估LLM多语言能力的重要性。 

---
# Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE 

**Title (ZH)**: Jakiro: 利用解耦多头MoE提升推测解码 

**Authors**: Haiduo Huang, Fuwei Yang, Zhenhua Liu, Yixing Xu, Jinze Li, Yang Liu, Xuanwu Yin, Dong Li, Pengju Ren, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2502.06282)  

**Abstract**: Speculative decoding (SD) accelerates large language model inference by using a smaller draft model to predict multiple tokens, which are then verified in parallel by the larger target model. However, the limited capacity of the draft model often necessitates tree-based sampling to improve prediction accuracy, where multiple candidates are generated at each step. We identify a key limitation in this approach: the candidates at the same step are derived from the same representation, limiting diversity and reducing overall effectiveness. To address this, we propose Jakiro, leveraging Mixture of Experts (MoE), where independent experts generate diverse predictions, effectively decoupling correlations among candidates. Furthermore, we introduce a hybrid inference strategy, combining autoregressive decoding for initial tokens with parallel decoding for subsequent stages, and enhance the latter with contrastive mechanism in features to improve accuracy. Our method significantly boosts prediction accuracy and achieves higher inference speedups. Extensive experiments across diverse models validate the effectiveness and robustness of our approach, establishing a new SOTA in speculative decoding. Our codes are available at this https URL. 

**Abstract (ZH)**: 投机解码（SD）通过使用较小的草稿模型预测多个令牌来加速大型语言模型的推理，然后由较大的目标模型并行验证这些令牌。然而，草稿模型的有限容量常常需要使用基于树的采样来提高预测准确性，在每一步生成多个候选。我们识别出这种做法的一个关键限制：同一步中的候选来自于相同的表示，限制了多样性和整体有效性。为此，我们提出了Jakiro，并利用专家混合（MoE），其中独立的专家生成多样化的预测，有效地解耦候选之间的关联。此外，我们引入了一种混合推理策略，结合自回归解码进行初始令牌解码，并在后续阶段使用并行解码，通过对比机制增强后者以提高准确性。我们的方法显著提升了预测准确性并实现了更高的推理加速。广泛实验表明，我们的方法在投机解码中具有更高的有效性和鲁棒性，并建立了新的SOTA。我们的代码可在以下链接获取。 

---
# K-ON: Stacking Knowledge On the Head Layer of Large Language Model 

**Title (ZH)**: K-ON: 在大型语言模型的头部层上堆叠知识 

**Authors**: Lingbing Guo, Yichi Zhang, Zhongpu Bo, Zhuo Chen, Mengshu Sun, Zhiqiang Zhang, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.06257)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved various natural language processing (NLP) tasks. Typically, LLMs are trained to predict the next token, aligning well with many NLP tasks. However, in knowledge graph (KG) scenarios, entities are the fundamental units and identifying an entity requires at least several tokens. This leads to a granularity mismatch between KGs and natural languages. To address this issue, we propose K-ON, which integrates KG knowledge into the LLM by employing multiple head layers for next k-step prediction. K-ON can not only generate entity-level results in one step, but also enables contrastive loss against entities, which is the most powerful tool in KG representation learning. Experimental results show that K-ON outperforms state-of-the-art methods that incorporate text and even the other modalities. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）显著提高了各种自然语言处理（NLP）任务的性能。通常，LLMs被训练以预测下一个令牌，这与许多NLP任务很好地对齐。然而，在知识图谱（KG）场景中，实体是基本单位，并识别一个实体需要至少几个令牌。这导致了KG和自然语言之间的粒度不匹配。为了应对这一问题，我们提出K-ON，通过使用多头层进行下一步预测来将KG知识集成到LLM中。K-ON不仅可以在一步中生成实体级别的结果，还可以启用与实体对比的损失，这是KG表示学习中最强大的工具。实验结果表明，K-ON在包含文本甚至其他模态的最新方法中表现出色。 

---
# Confidence Improves Self-Consistency in LLMs 

**Title (ZH)**: 信心提高LLMs的自我一致性 

**Authors**: Amir Taubenfeld, Tom Sheffer, Eran Ofek, Amir Feder, Ariel Goldstein, Zorik Gekhman, Gal Yona  

**Link**: [PDF](https://arxiv.org/pdf/2502.06233)  

**Abstract**: Self-consistency decoding enhances LLMs' performance on reasoning tasks by sampling diverse reasoning paths and selecting the most frequent answer. However, it is computationally expensive, as sampling many of these (lengthy) paths is required to increase the chances that the correct answer emerges as the most frequent one. To address this, we introduce Confidence-Informed Self-Consistency (CISC). CISC performs a weighted majority vote based on confidence scores obtained directly from the model. By prioritizing high-confidence paths, it can identify the correct answer with a significantly smaller sample size. When tested on nine models and four datasets, CISC outperforms self-consistency in nearly all configurations, reducing the required number of reasoning paths by over 40% on average. In addition, we introduce the notion of within-question confidence evaluation, after showing that standard evaluation methods are poor predictors of success in distinguishing correct and incorrect answers to the same question. In fact, the most calibrated confidence method proved to be the least effective for CISC. Lastly, beyond these practical implications, our results and analyses show that LLMs can effectively judge the correctness of their own outputs, contributing to the ongoing debate on this topic. 

**Abstract (ZH)**: Confidence-Informed Self-Consistency Enhances LLMs' Performance on Reasoning Tasks by Prioritizing High-Confidence Paths 

---
# LessLeak-Bench: A First Investigation of Data Leakage in LLMs Across 83 Software Engineering Benchmarks 

**Title (ZH)**: LessLeak-基准：对83个软件工程基准中LLM数据泄漏现象的首次探究 

**Authors**: Xin Zhou, Martin Weyssow, Ratnadira Widyasari, Ting Zhang, Junda He, Yunbo Lyu, Jianming Chang, Beiqi Zhang, Dan Huang, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2502.06215)  

**Abstract**: Large Language Models (LLMs) are widely utilized in software engineering (SE) tasks, such as code generation and automated program repair. However, their reliance on extensive and often undisclosed pre-training datasets raises significant concerns about data leakage, where the evaluation benchmark data is unintentionally ``seen'' by LLMs during the model's construction phase. The data leakage issue could largely undermine the validity of LLM-based research and evaluations. Despite the increasing use of LLMs in the SE community, there is no comprehensive study that assesses the extent of data leakage in SE benchmarks for LLMs yet. To address this gap, this paper presents the first large-scale analysis of data leakage in 83 SE benchmarks concerning LLMs. Our results show that in general, data leakage in SE benchmarks is minimal, with average leakage ratios of only 4.8\%, 2.8\%, and 0.7\% for Python, Java, and C/C++ benchmarks, respectively. However, some benchmarks exhibit relatively higher leakage ratios, which raises concerns about their bias in evaluation. For instance, QuixBugs and BigCloneBench have leakage ratios of 100.0\% and 55.7\%, respectively. Furthermore, we observe that data leakage has a substantial impact on LLM evaluation. We also identify key causes of high data leakage, such as the direct inclusion of benchmark data in pre-training datasets and the use of coding platforms like LeetCode for benchmark construction. To address the data leakage, we introduce \textbf{LessLeak-Bench}, a new benchmark that removes leaked samples from the 83 SE benchmarks, enabling more reliable LLM evaluations in future research. Our study enhances the understanding of data leakage in SE benchmarks and provides valuable insights for future research involving LLMs in SE. 

**Abstract (ZH)**: 大型语言模型（LLMs）在软件工程（SE）任务中的广泛应用，如代码生成和自动化程序修复。然而，其对大量且 Often Undisclosed 的预训练数据的依赖引发了关于数据泄露的重大关注，即评估基准数据在模型构建阶段意外地被LLMs“看见”。数据泄露问题可能会严重影响基于LLMs的研究和评估的效度。尽管LLMs在SE社区中的应用日益增多，但至今还没有全面研究评估LLMs在SE基准中的数据泄露程度。为弥补这一空白，本文首次对83个涉及LLMs的SE基准进行了大规模的数据泄露分析。结果显示，总体而言，SE基准中的数据泄露程度较低，Python、Java和C/C++基准的数据泄露比例分别为4.8%、2.8%和0.7%。然而，某些基准的数据泄露比例较高，这可能对其评估产生偏见。例如，QuixBugs和BigCloneBench的数据泄露比例分别为100.0%和55.7%。此外，我们还观察到数据泄露对LLMs评估有显著影响。我们还识别了高数据泄露的关键原因，如在预训练数据集中直接包含基准数据以及使用像LeetCode这样的编程平台构建基准。为解决数据泄露问题，我们提出了一个新的基准LessLeak-Bench，该基准从83个SE基准中移除了泄露样本，以在未来的研究中提供更可靠的LLM评估。本研究增强了对SE基准中数据泄露的理解，并为涉及LLMs的未来SE研究提供了宝贵的见解。 

---
# Unveiling the Capabilities of Large Language Models in Detecting Offensive Language with Annotation Disagreement 

**Title (ZH)**: 揭示大语言模型在标注分歧条件下检测冒犯语言的能力 

**Authors**: Junyu Lu, Kai Ma, Kaichun Wang, Kelaiti Xiao, Roy Ka-Wei Lee, Bo Xu, Liang Yang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.06207)  

**Abstract**: LLMs are widely used for offensive language detection due to their advanced capability. However, the challenges posed by human annotation disagreement in real-world datasets remain underexplored. These disagreement samples are difficult to detect due to their ambiguous nature. Additionally, the confidence of LLMs in processing disagreement samples can provide valuable insights into their alignment with human annotators. To address this gap, we systematically evaluate the ability of LLMs to detect offensive language with annotation disagreement. We compare the binary accuracy of multiple LLMs across varying annotation agreement levels and analyze the relationship between LLM confidence and annotation agreement. Furthermore, we investigate the impact of disagreement samples on LLM decision-making during few-shot learning and instruction fine-tuning. Our findings highlight the challenges posed by disagreement samples and offer guidance for improving LLM-based offensive language detection. 

**Abstract (ZH)**: LLMs在检测具有标注分歧的冒犯性语言方面的能力评估及其影响研究 

---
# C-3PO: Compact Plug-and-Play Proxy Optimization to Achieve Human-like Retrieval-Augmented Generation 

**Title (ZH)**: C-3PO: 紧凑型插即用代理优化以实现类人类检索增强生成 

**Authors**: Guoxin Chen, Minpeng Liao, Peiying Yu, Dingmin Wang, Zile Qiao, Chao Yang, Xin Zhao, Kai Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.06205)  

**Abstract**: Retrieval-augmented generation (RAG) systems face a fundamental challenge in aligning independently developed retrievers and large language models (LLMs). Existing approaches typically involve modifying either component or introducing simple intermediate modules, resulting in practical limitations and sub-optimal performance. Inspired by human search behavior -- typically involving a back-and-forth process of proposing search queries and reviewing documents, we propose C-3PO, a proxy-centric framework that facilitates communication between retrievers and LLMs through a lightweight multi-agent system. Our framework implements three specialized agents that collaboratively optimize the entire RAG pipeline without altering the retriever and LLMs. These agents work together to assess the need for retrieval, generate effective queries, and select information suitable for the LLMs. To enable effective multi-agent coordination, we develop a tree-structured rollout approach for reward credit assignment in reinforcement learning. Extensive experiments in both in-domain and out-of-distribution scenarios demonstrate that C-3PO significantly enhances RAG performance while maintaining plug-and-play flexibility and superior generalization capabilities. 

**Abstract (ZH)**: 基于代理的检索增强生成系统：C-3PO框架 

---
# Can LLMs Replace Human Evaluators? An Empirical Study of LLM-as-a-Judge in Software Engineering 

**Title (ZH)**: LLMs能取代人类评估者吗？软件工程中LLM作为法官的实证研究 

**Authors**: Ruiqi Wang, Jiyu Guo, Cuiyun Gao, Guodong Fan, Chun Yong Chong, Xin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2502.06193)  

**Abstract**: Recently, large language models (LLMs) have been deployed to tackle various software engineering (SE) tasks like code generation, significantly advancing the automation of SE tasks. However, assessing the quality of these LLM-generated code and text remains challenging. The commonly used Pass@k metric necessitates extensive unit tests and configured environments, demands a high labor cost, and is not suitable for evaluating LLM-generated text. Conventional metrics like BLEU, which measure only lexical rather than semantic similarity, have also come under scrutiny. In response, a new trend has emerged to employ LLMs for automated evaluation, known as LLM-as-a-judge. These LLM-as-a-judge methods are claimed to better mimic human assessment than conventional metrics without relying on high-quality reference answers. Nevertheless, their exact human alignment in SE tasks remains unexplored. In this paper, we empirically explore LLM-as-a-judge methods for evaluating SE tasks, focusing on their alignment with human judgments. We select seven LLM-as-a-judge methods that utilize general-purpose LLMs, alongside two LLMs specifically fine-tuned for evaluation. After generating and manually scoring LLM responses on three recent SE datasets of code translation, code generation, and code summarization, we then prompt these methods to evaluate each response. Finally, we compare the scores generated by these methods with human evaluation. The results indicate that output-based methods reach the highest Pearson correlation of 81.32 and 68.51 with human scores in code translation and generation, achieving near-human evaluation, noticeably outperforming ChrF++, one of the best conventional metrics, at 34.23 and 64.92. Such output-based methods prompt LLMs to output judgments directly, and exhibit more balanced score distributions that resemble human score patterns. Finally, we provide... 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）已被部署用于处理软件工程（SE）任务，如代码生成，显著推进了SE任务的自动化。然而，评估这些LLM生成的代码和文本的质量仍然颇具挑战。常用的Pass@k指标需要大量的单元测试和配置环境，成本高昂，并且不适合评估LLM生成的文本。传统的测量词汇相似度而非语义相似度的指标，如BLEU，也受到了质疑。为此，涌现了一种新的趋势，即使用LLM进行自动评估，这种方法被称为“LLM作为裁判”。这些“LLM作为裁判”方法声称比传统指标更接近人类评估，无需依赖高质量的参考答案。然而，它们在SE任务中的精确人类一致性尚未得到探索。本文通过实验研究“LLM作为裁判”方法在SE任务中的应用，重点关注它们与人类判断的一致性。选择了七种利用通用LLM的“LLM作为裁判”方法，以及两种专门针对评估进行微调的LLM。在对三个最新的SE数据集（代码翻译、代码生成和代码摘要）进行生成和手动评分后，我们进一步针对这些方法提请评估每个响应。最后，我们将这些方法生成的评分与人工评估进行比较。结果表明，在代码翻译和生成任务中，基于输出的方法分别获得了与人工评分的皮尔森相关系数81.32和68.51，实现了接近人类的评估，明显优于ChrF++这一表现最佳的传统指标34.23和64.92。基于输出的方法促使LLM直接输出判断，并展现出更平衡的评分分布，更接近人类评分模式。最后，我们提供了... 

---
# Uncertainty-Aware Adaptation of Large Language Models for Protein-Protein Interaction Analysis 

**Title (ZH)**: 面向蛋白质-蛋白质相互作用分析的大语言模型的不确定性意识适应性调整 

**Authors**: Sanket Jantre, Tianle Wang, Gilchan Park, Kriti Chopra, Nicholas Jeon, Xiaoning Qian, Nathan M. Urban, Byung-Jun Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2502.06173)  

**Abstract**: Identification of protein-protein interactions (PPIs) helps derive cellular mechanistic understanding, particularly in the context of complex conditions such as neurodegenerative disorders, metabolic syndromes, and cancer. Large Language Models (LLMs) have demonstrated remarkable potential in predicting protein structures and interactions via automated mining of vast biomedical literature; yet their inherent uncertainty remains a key challenge for deriving reproducible findings, critical for biomedical applications. In this study, we present an uncertainty-aware adaptation of LLMs for PPI analysis, leveraging fine-tuned LLaMA-3 and BioMedGPT models. To enhance prediction reliability, we integrate LoRA ensembles and Bayesian LoRA models for uncertainty quantification (UQ), ensuring confidence-calibrated insights into protein behavior. Our approach achieves competitive performance in PPI identification across diverse disease contexts while addressing model uncertainty, thereby enhancing trustworthiness and reproducibility in computational biology. These findings underscore the potential of uncertainty-aware LLM adaptation for advancing precision medicine and biomedical research. 

**Abstract (ZH)**: 基于不确定性意识的大型语言模型在蛋白质-蛋白质相互作用分析中的适应：推动计算生物学的信任度和可重复性 

---
# CSR-Bench: Benchmarking LLM Agents in Deployment of Computer Science Research Repositories 

**Title (ZH)**: CSR-Bench: 计算机科学研究仓库部署中LLM代理的基准测试 

**Authors**: Yijia Xiao, Runhui Wang, Luyang Kong, Davor Golac, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06111)  

**Abstract**: The increasing complexity of computer science research projects demands more effective tools for deploying code repositories. Large Language Models (LLMs), such as Anthropic Claude and Meta Llama, have demonstrated significant advancements across various fields of computer science research, including the automation of diverse software engineering tasks. To evaluate the effectiveness of LLMs in handling complex code development tasks of research projects, particularly for NLP/CV/AI/ML/DM topics, we introduce CSR-Bench, a benchmark for Computer Science Research projects. This benchmark assesses LLMs from various aspects including accuracy, efficiency, and deployment script quality, aiming to explore their potential in conducting computer science research autonomously. We also introduce a novel framework, CSR-Agents, that utilizes multiple LLM agents to automate the deployment of GitHub code repositories of computer science research projects. Specifically, by checking instructions from markdown files and interpreting repository structures, the model generates and iteratively improves bash commands that set up the experimental environments and deploy the code to conduct research tasks. Preliminary results from CSR-Bench indicate that LLM agents can significantly enhance the workflow of repository deployment, thereby boosting developer productivity and improving the management of developmental workflows. 

**Abstract (ZH)**: 计算机科学研究项目日益增加的复杂性催生了对更有效代码仓库部署工具的需求。大型语言模型（LLMs），如Anthropic Claude和Meta Llama，在计算机科学研究的各个领域，包括软件工程任务的自动化方面，已经显示出显著的进步。为了评估LLMs在处理计算机科学研究项目中的复杂代码开发任务的有效性，特别是针对自然语言处理（NLP）、计算机视觉（CV）、人工智能（AI）、机器学习（ML）、数据挖掘（DM）等主题，我们引入了CSR-Bench——一个计算机科学研究项目的基准测试。该基准测试从准确性、效率和部署脚本质量等多个方面评估LLMs，旨在探索其在自主进行计算机科学研究方面的潜力。我们还引入了CSR-Agents这一新颖框架，利用多个LLM代理来自动化计算机科学研究项目GitHub代码仓库的部署。具体来说，该模型通过检查来自Markdown文件的指令并解释代码仓库结构，生成并迭代改进用于设置实验环境和部署代码的bash命令，以执行研究任务。CSR-Bench的初步结果显示，LLM代理能够显著提升仓库部署的工作流程，从而提高开发人员的生产力并改善开发工作流程的管理。 

---
# Benchmarking Prompt Sensitivity in Large Language Models 

**Title (ZH)**: 大型语言模型中提示敏感性的基准测试 

**Authors**: Amirhossein Razavi, Mina Soltangheis, Negar Arabzadeh, Sara Salamat, Morteza Zihayat, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2502.06065)  

**Abstract**: Large language Models (LLMs) are highly sensitive to variations in prompt formulation, which can significantly impact their ability to generate accurate responses. In this paper, we introduce a new task, Prompt Sensitivity Prediction, and a dataset PromptSET designed to investigate the effects of slight prompt variations on LLM performance. Using TriviaQA and HotpotQA datasets as the foundation of our work, we generate prompt variations and evaluate their effectiveness across multiple LLMs. We benchmark the prompt sensitivity prediction task employing state-of-the-art methods from related tasks, including LLM-based self-evaluation, text classification, and query performance prediction techniques. Our findings reveal that existing methods struggle to effectively address prompt sensitivity prediction, underscoring the need to understand how information needs should be phrased for accurate LLM responses. 

**Abstract (ZH)**: 大规模语言模型（LLMs）对提示形式的变异高度敏感，这可以显著影响其生成准确响应的能力。本文介绍了一个新的任务——提示敏感性预测，并设计了一个名为PromptSET的数据集，旨在研究轻微提示变异对LLM性能的影响。基于TriviaQA和HotpotQA数据集，我们生成了提示变异，并在多种LLM上评估其有效性。我们使用相关任务中的先进方法，包括基于LLM的自我评估、文本分类和查询性能预测技术，来基准测试提示敏感性预测任务。我们的研究发现现有方法在有效解决提示敏感性预测方面存在困难，强调了理解如何表达信息需求以获得准确LLM响应的重要性。 

---
# LM2: Large Memory Models 

**Title (ZH)**: LM2: 大型记忆模型 

**Authors**: Jikun Kang, Wenqi Wu, Filippos Christianos, Alex J. Chan, Fraser Greenlee, George Thomas, Marvin Purtorab, Andy Toulis  

**Link**: [PDF](https://arxiv.org/pdf/2502.06049)  

**Abstract**: This paper introduces the Large Memory Model (LM2), a decoder-only Transformer architecture enhanced with an auxiliary memory module that aims to address the limitations of standard Transformers in multi-step reasoning, relational argumentation, and synthesizing information distributed over long contexts. The proposed LM2 incorporates a memory module that acts as a contextual representation repository, interacting with input tokens via cross attention and updating through gating mechanisms. To preserve the Transformers general-purpose capabilities, LM2 maintains the original information flow while integrating a complementary memory pathway. Experimental results on the BABILong benchmark demonstrate that the LM2model outperforms both the memory-augmented RMT model by 37.1% and the baseline Llama-3.2 model by 86.3% on average across tasks. LM2 exhibits exceptional capabilities in multi-hop inference, numerical reasoning, and large-context question-answering. On the MMLU dataset, it achieves a 5.0% improvement over a pre-trained vanilla model, demonstrating that its memory module does not degrade performance on general tasks. Further, in our analysis, we explore the memory interpretability, effectiveness of memory modules, and test-time behavior. Our findings emphasize the importance of explicit memory in enhancing Transformer architectures. 

**Abstract (ZH)**: 大型记忆模型（LM2）：一种增强的记忆模块辅助解码器Transformer架构及其在多步推理、关系论证和长上下文信息综合中的应用 

---
# Benchmarking Prompt Engineering Techniques for Secure Code Generation with GPT Models 

**Title (ZH)**: 基于GPT模型的提示工程技术在安全代码生成中的基准测试 

**Authors**: Marc Bruni, Fabio Gabrielli, Mohammad Ghafari, Martin Kropp  

**Link**: [PDF](https://arxiv.org/pdf/2502.06039)  

**Abstract**: Prompt engineering reduces reasoning mistakes in Large Language Models (LLMs). However, its effectiveness in mitigating vulnerabilities in LLM-generated code remains underexplored. To address this gap, we implemented a benchmark to automatically assess the impact of various prompt engineering strategies on code security. Our benchmark leverages two peer-reviewed prompt datasets and employs static scanners to evaluate code security at scale. We tested multiple prompt engineering techniques on GPT-3.5-turbo, GPT-4o, and GPT-4o-mini. Our results show that for GPT-4o and GPT-4o-mini, a security-focused prompt prefix can reduce the occurrence of security vulnerabilities by up to 56%. Additionally, all tested models demonstrated the ability to detect and repair between 41.9% and 68.7% of vulnerabilities in previously generated code when using iterative prompting techniques. Finally, we introduce a "prompt agent" that demonstrates how the most effective techniques can be applied in real-world development workflows. 

**Abstract (ZH)**: Prompt工程减少大型语言模型（LLMs）在代码生成中的推理错误，但其在缓解LLM生成代码中的脆弱性方面的有效性仍待探索。为了弥补这一空白，我们实现了基准测试以自动评估各种prompt工程策略对代码安全的影响。该基准测试利用了两个同行评审的prompt数据集，并使用静态扫描器大规模评估代码安全。我们在GPT-3.5-turbo、GPT-4o和GPT-4o-mini上测试了多种prompt工程技术。结果显示，对于GPT-4o和GPT-4o-mini，带有安全重点的prompt前缀可以将安全漏洞的发生率降低高达56%。此外，所有测试的模型在使用迭代prompt技术时，能够检测并修复之前生成的代码中的41.9%至68.7%的漏洞。最后，我们引入了一个“prompt代理”，展示了如何在实际开发工作流程中应用最有效的技术。 

---
# Analysis of LLM as a grammatical feature tagger for African American English 

**Title (ZH)**: LLM作为语法特征标注器对非洲美语的分析 

**Authors**: Rahul Porwal, Alice Rozet, Pryce Houck, Jotsna Gowda, Sarah Moeller, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06004)  

**Abstract**: African American English (AAE) presents unique challenges in natural language processing (NLP). This research systematically compares the performance of available NLP models--rule-based, transformer-based, and large language models (LLMs)--capable of identifying key grammatical features of AAE, namely Habitual Be and Multiple Negation. These features were selected for their distinct grammatical complexity and frequency of occurrence. The evaluation involved sentence-level binary classification tasks, using both zero-shot and few-shot strategies. The analysis reveals that while LLMs show promise compared to the baseline, they are influenced by biases such as recency and unrelated features in the text such as formality. This study highlights the necessity for improved model training and architectural adjustments to better accommodate AAE's unique linguistic characteristics. Data and code are available. 

**Abstract (ZH)**: African American English (AAE)在自然语言处理（NLP）中呈现独特的挑战。本研究系统比较了能够识别AAE关键语法特征（即惯用系be和多重否定）的基于规则、基于变换器和大规模语言模型（LLMs）的性能。评估包括句级二分类任务，采用零样本和少样本策略。分析显示，虽然大规模语言模型相较于基线模型显示出潜力，但也受到文本中近期偏见和其他无关特征（如正式程度）的影响。本研究强调了改进模型训练和架构调整以更好地容纳AAE的独特语言特征的必要性。数据和代码可供获取。 

---
# "Let the AI conspiracy begin..." Language Model coordination is just one inference-intervention away 

**Title (ZH)**: 让AI阴谋论开始……语言模型协调只需一次推理-干预。 

**Authors**: Paul Darm, Annalisa Riccardi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05945)  

**Abstract**: In this work, we introduce a straightforward and effective methodology to steer large language model behaviour capable of bypassing learned alignment goals. We employ interference-time activation shifting, which is effective without additional training. Following prior studies, we derive intervention directions from activation differences in contrastive pairs of model outputs, which represent the desired and undesired behaviour. By prompting the model to include multiple-choice answers in its response, we can automatically evaluate the sensitivity of model output to individual attention heads steering efforts. We demonstrate that interventions on these heads generalize well to open-ended answer generation in the challenging "AI coordination" dataset. In this dataset, models must choose between assisting another AI or adhering to ethical, safe, and unharmful behaviour. Our fine-grained interventions lead Llama 2 to prefer coordination with other AIs over following established alignment goals. Additionally, this approach enables stronger interventions than those applied to whole model layers, preserving the overall cohesiveness of the output. The simplicity of our method highlights the shortcomings of current alignment strategies and points to potential future research directions, as concepts like "AI coordination" can be influenced by selected attention heads. 

**Abstract (ZH)**: 本研究介绍了一种简单有效的策略，用于引导大型语言模型的行为，该策略能够绕过已学习的对齐目标。我们采用了干扰时间激活位移的方法，该方法在无需额外训练的情况下有效。借鉴先前研究，我们从对比模型输出的激活差异中推导出干预方向，这些激活差异代表了期望和不期望的行为。通过提示模型在其响应中包含多种选择的答案，我们能够自动评估模型输出对单个注意力头引导努力的敏感性。我们展示了对这些头的干预在“AI协调”数据集的开放性答案生成任务中表现出良好的泛化能力。在该数据集中，模型必须在协助另一个AI或遵循伦理、安全和非有害行为之间做出选择。我们细致的干预使Llama 2更倾向于与其他AI协同工作，而不是遵循既定的对齐目标。此外，这种方法允许比整层模型层更强大的干预，同时保持输出的整体连贯性。我们方法的简洁性揭示了现有对齐策略的局限性，并指出了未来研究的方向，因为如“AI协调”这样的概念可能受到选定注意力头的影响。 

---
# A Semi-Supervised Text Generation Framework Combining a Deep Transformer and a GAN 

**Title (ZH)**: 一种结合深度Transformer和GAN的半监督文本生成框架 

**Authors**: Shengquan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05937)  

**Abstract**: This paper introduces a framework that connects a deep generative pre-trained Transformer language model with a generative adversarial network for semi-supervised text generation. In other words, the proposed model is first pre-trained unsupervised on a large and diverse text corpus with 24 layers. Then a simple GAN architecture for synthetic text generation is introduced, and Gumbel-Softmax is applied to handle the discreteness of tokens. The paper also shows a semi-supervised approach where real data is augmented with GAN samples, which is further used to fine-tune the Transformer model on the merged dataset. Detailed theoretical derivations are also included, outlining the proof of the min-max objective function, and an extensive discussion of the Gumbel-Softmax reparameterization trick. 

**Abstract (ZH)**: 本文介绍了一种将深度生成预训练变压器语言模型与生成对抗网络连接起来的框架，用于半监督文本生成。具体而言，所提出模型首先在包含24层的大型和多样化文本语料上进行无监督预训练。然后介绍了用于合成文本生成的简单GAN架构，并应用了Gumbel-Softmax来处理词元的离散性。此外，本文还展示了将真实数据与GAN样本进行增广的半监督方法，并进一步利用合并数据集对Transformer模型进行微调。文中还包含了详细的理论推导，证明了极小极大目标函数，并对Gumbel-Softmax重参数化技巧进行了广泛讨论。 

---
# Learning to Substitute Words with Model-based Score Ranking 

**Title (ZH)**: 基于模型评分排序的学习单词替换方法 

**Authors**: Hongye Liu, Ricardo Henao  

**Link**: [PDF](https://arxiv.org/pdf/2502.05933)  

**Abstract**: Smart word substitution aims to enhance sentence quality by improving word choices; however current benchmarks rely on human-labeled data. Since word choices are inherently subjective, ground-truth word substitutions generated by a small group of annotators are often incomplete and likely not generalizable. To circumvent this issue, we instead employ a model-based score (BARTScore) to quantify sentence quality, thus forgoing the need for human annotations. Specifically, we use this score to define a distribution for each word substitution, allowing one to test whether a substitution is statistically superior relative to others. In addition, we propose a loss function that directly optimizes the alignment between model predictions and sentence scores, while also enhancing the overall quality score of a substitution. Crucially, model learning no longer requires human labels, thus avoiding the cost of annotation while maintaining the quality of the text modified with substitutions. Experimental results show that the proposed approach outperforms both masked language models (BERT, BART) and large language models (GPT-4, LLaMA). The source code is available at this https URL. 

**Abstract (ZH)**: 智能词替换旨在通过改进词选择来提升句子质量；然而当前基准依赖于人工标记的数据。由于词选择本质上是主观的，由一小群标注者生成的真实词替换往往是不完整的，可能不具备普适性。为克服这一问题，我们改而采用基于模型的评分（BARTScore）来量化句子质量，从而避免人工标注的需要。具体而言，我们使用该评分定义每个词替换的分布，使我们可以测试一个替换是否在统计上优于其他替换。此外，我们提出了一种损失函数，该函数直接优化模型预测与句子评分之间的对齐，同时提升替换的整体质量评分。至关重要的是，模型学习不再需要人工标签，从而避免标注成本同时保持被替换文本的质量。实验结果表明，所提出的方法在掩码语言模型（BERT、BART）和大型语言模型（GPT-4、LLaMA）上均表现出更优性能。源代码可在以下链接获取。 

---
# A Distributional Perspective on Word Learning in Neural Language Models 

**Title (ZH)**: 神经语言模型中词的学习的分布视角 

**Authors**: Filippo Ficarra, Ryan Cotterell, Alex Warstadt  

**Link**: [PDF](https://arxiv.org/pdf/2502.05892)  

**Abstract**: Language models (LMs) are increasingly being studied as models of human language learners. Due to the nascency of the field, it is not well-established whether LMs exhibit similar learning dynamics to humans, and there are few direct comparisons between learning trajectories in humans and models. Word learning trajectories for children are relatively well-documented, and recent work has tried to extend these investigations to language models. However, there are no widely agreed-upon metrics for word learning in language models. We take a distributional approach to this problem, defining lexical knowledge in terms of properties of the learned distribution for a target word. We argue that distributional signatures studied in prior work fail to capture key distributional information. Thus, we propose an array of signatures that improve on earlier approaches by capturing knowledge of both where the target word can and cannot occur as well as gradient preferences about the word's appropriateness. We obtain learning trajectories for a selection of small language models we train from scratch, study the relationship between different distributional signatures, compare how well they align with human word learning trajectories and interpretable lexical features, and address basic methodological questions about estimating these distributional signatures. Our metrics largely capture complementary information, suggesting that it is important not to rely on a single metric. However, across all metrics, language models' learning trajectories fail to correlate with those of children. 

**Abstract (ZH)**: 语言模型的语言学习动态：一种基于分布的方法 

---
# Enhancing Depression Detection with Chain-of-Thought Prompting: From Emotion to Reasoning Using Large Language Models 

**Title (ZH)**: 增强抑郁症检测：从情绪到推理的大语言模型链式思维提示方法 

**Authors**: Shiyu Teng, Jiaqing Liu, Rahul Kumar Jain, Shurong Chai, Ruibo Hou, Tomoko Tateyama, Lanfen Lin, Yen-wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05879)  

**Abstract**: Depression is one of the leading causes of disability worldwide, posing a severe burden on individuals, healthcare systems, and society at large. Recent advancements in Large Language Models (LLMs) have shown promise in addressing mental health challenges, including the detection of depression through text-based analysis. However, current LLM-based methods often struggle with nuanced symptom identification and lack a transparent, step-by-step reasoning process, making it difficult to accurately classify and explain mental health conditions. To address these challenges, we propose a Chain-of-Thought Prompting approach that enhances both the performance and interpretability of LLM-based depression detection. Our method breaks down the detection process into four stages: (1) sentiment analysis, (2) binary depression classification, (3) identification of underlying causes, and (4) assessment of severity. By guiding the model through these structured reasoning steps, we improve interpretability and reduce the risk of overlooking subtle clinical indicators. We validate our method on the E-DAIC dataset, where we test multiple state-of-the-art large language models. Experimental results indicate that our Chain-of-Thought Prompting technique yields superior performance in both classification accuracy and the granularity of diagnostic insights, compared to baseline approaches. 

**Abstract (ZH)**: 全球范围内，抑郁症是导致残疾的主要原因之一，对个人、医疗系统和社会造成了严重的负担。近年来，大型语言模型（LLMs）的进步显示出了应对心理健康挑战的潜力，包括通过文本分析检测抑郁症。然而，当前基于LLM的方法在细微症状识别上常表现出困难，并缺乏透明且步步为营的推理过程，这使得准确分类和解释心理健康状况变得困难。为解决这些挑战，我们提出了一种链式思考提示方法，以提高基于LLM的抑郁症检测的性能和可解释性。我们的方法将检测过程分为四个阶段：（1）情感分析，（2）二元抑郁症分类，（3）识别潜在原因，（4）评估严重程度。通过引导模型遵循这些结构化的推理步骤，我们提高了可解释性并降低了遗漏细微临床指标的风险。我们在E-DAIC数据集上验证了我们的方法，测试了多个最先进的大型语言模型。实验结果表明，与基线方法相比，我们的链式思考提示技术在分类准确性和诊断洞察的细致程度上均表现更优。 

---
# LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification 

**Title (ZH)**: LegalSeg: 通过修辞角色分类解锁印度法律判决的结构 

**Authors**: Shubham Kumar Nigam, Tanmay Dubey, Govind Sharma, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2502.05836)  

**Abstract**: In this paper, we address the task of semantic segmentation of legal documents through rhetorical role classification, with a focus on Indian legal judgments. We introduce LegalSeg, the largest annotated dataset for this task, comprising over 7,000 documents and 1.4 million sentences, labeled with 7 rhetorical roles. To benchmark performance, we evaluate multiple state-of-the-art models, including Hierarchical BiLSTM-CRF, TransformerOverInLegalBERT (ToInLegalBERT), Graph Neural Networks (GNNs), and Role-Aware Transformers, alongside an exploratory RhetoricLLaMA, an instruction-tuned large language model. Our results demonstrate that models incorporating broader context, structural relationships, and sequential sentence information outperform those relying solely on sentence-level features. Additionally, we conducted experiments using surrounding context and predicted or actual labels of neighboring sentences to assess their impact on classification accuracy. Despite these advancements, challenges persist in distinguishing between closely related roles and addressing class imbalance. Our work underscores the potential of advanced techniques for improving legal document understanding and sets a strong foundation for future research in legal NLP. 

**Abstract (ZH)**: 本文通过修辞角色分类任务探讨印度法律判决的语义分割，介绍了包含逾7000份文档和140万句的标注数据集LegalSeg，标注了7种修辞角色。为了benchmark性能，评估了包括层次BiLSTM-CRF、TransformerOverInLegalBERT (ToInLegalBERT)、图神经网络(GNNs)、感知修辞角色的Transformer以及探索性RhetoricLLaMA在内的多种先进模型。结果表明，结合更广泛上下文、结构关系及句子序列信息的模型优于仅依赖句内特征的模型。此外，还利用相邻句子的上下文信息和预测或实际标签进行了实验，以评估其对分类准确率的影响。尽管取得了进展，但在区分紧密相关角色和解决类别不平衡问题方面仍面临挑战。本文强调了高级技术在提高法律文件理解方面的潜力，并为未来的法律NLP研究奠定了坚实基础。 

---
# Delta - Contrastive Decoding Mitigates Text Hallucinations in Large Language Models 

**Title (ZH)**: Delta-对比解码缓解大规模语言模型中的文本幻觉 

**Authors**: Cheng Peng Huang, Hao-Yuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05825)  

**Abstract**: Large language models (LLMs) demonstrate strong capabilities in natural language processing but remain prone to hallucinations, generating factually incorrect or fabricated content. This issue undermines their reliability, particularly in high-stakes domains such as healthcare and legal advisory. To address this challenge, we propose Delta, an inference-time method that reduces hallucinations without requiring model retraining or additional data. Delta works by randomly masking parts of the input prompt and contrasting the output distributions for the original and masked inputs, effectively suppressing hallucinations through inference-only computations. We evaluate Delta on context-rich question-answering benchmarks, achieving absolute improvements of approximately 3 and 6 percentage points on SQuAD v1.1 and v2, respectively, and 7 and 2 percentage points on TriviaQA and Natural Questions under-sampling decoding. Delta also improves the no-answer exact match score on SQuAD v2 by over ten percentage points, demonstrating its effectiveness in mitigating hallucinations arising from contextual ambiguity. These results highlight Delta as a computationally efficient and scalable approach for improving the reliability of LLMs in real-world applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理任务中表现出强大的能力，但仍易产生幻觉，生成事实错误或虚构的内容。这一问题削弱了它们的可靠性，特别是在医疗和法律咨询等高风险领域。为解决这一挑战，我们提出Delta，一种推理时的方法，可以在无需模型重新训练或额外数据的情况下减少幻觉。Delta通过随机遮蔽输入提示的部分内容，并对比原始和遮蔽输入的输出分布，仅通过推理计算有效地抑制幻觉。我们在富含上下文的问答基准测试上评估了Delta，分别在SQuAD v1.1和v2上实现了约3和6个百分点的绝对改进，在TrivaQA和Natural Questions欠采样解码下分别实现了7和2个百分点的改进。Delta还在SQuAD v2上显著改善了无答案精确匹配分，这表明其在缓解因上下文歧义引发的幻觉方面具有有效性。这些结果突显了Delta作为提高LLMs在实际应用中可靠性的计算效率和可扩展方法的有效性。 

---
# The Curse of Depth in Large Language Models 

**Title (ZH)**: 大型语言模型中的深度诅咒 

**Authors**: Wenfang Sun, Xinyuan Song, Pengxiang Li, Lu Yin, Yefeng Zheng, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05795)  

**Abstract**: In this paper, we introduce the Curse of Depth, a concept that highlights, explains, and addresses the recent observation in modern Large Language Models(LLMs) where nearly half of the layers are less effective than expected. We first confirm the wide existence of this phenomenon across the most popular families of LLMs such as Llama, Mistral, DeepSeek, and Qwen. Our analysis, theoretically and empirically, identifies that the underlying reason for the ineffectiveness of deep layers in LLMs is the widespread usage of Pre-Layer Normalization (Pre-LN). While Pre-LN stabilizes the training of Transformer LLMs, its output variance exponentially grows with the model depth, which undesirably causes the derivative of the deep Transformer blocks to be an identity matrix, and therefore barely contributes to the training. To resolve this training pitfall, we propose LayerNorm Scaling, which scales the variance of output of the layer normalization inversely by the square root of its depth. This simple modification mitigates the output variance explosion of deeper Transformer layers, improving their contribution. Our experimental results, spanning model sizes from 130M to 1B, demonstrate that LayerNorm Scaling significantly enhances LLM pre-training performance compared to Pre-LN. Moreover, this improvement seamlessly carries over to supervised fine-tuning. All these gains can be attributed to the fact that LayerNorm Scaling enables deeper layers to contribute more effectively during training. 

**Abstract (ZH)**: 本文介绍了深度之难题，这一概念强调并解释了现代大型语言模型（LLMs）中近半层效果低于预期的近期观察现象。我们首先确认了这一现象在Llama、Mistral、DeepSeek和Qwen等最受欢迎的LLM家族中普遍存在。我们的理论与实证分析指出，大型语言模型中深层层无效的原因是广泛使用的前置层标准化（Pre-LN）。尽管Pre-LN稳定了Transformer LLM的训练，但其输出方差随模型深度呈指数增长，这不幸导致深层Transformer块的梯度几乎为单位矩阵，从而对训练贡献甚微。为解决这一训练难题，我们提出了层标准化缩放（LayerNorm Scaling），通过其深度的平方根的逆来缩放层标准化的输出方差。这一简单的修改缓解了更深层Transformer层的输出方差爆炸问题，提高了它们的训练贡献。我们涵盖从130M到1B的模型大小的实验结果表明，层标准化缩放显著提升了与Pre-LN相比的LLM预训练性能。此外，这一改进无缝地应用于监督微调。所有这些增益可归因于层标准化缩放使深层层在训练中能更有效地做出贡献。 

---
# PIPA: Preference Alignment as Prior-Informed Statistical Estimation 

**Title (ZH)**: PIPA：先验信息引导的统计估计&equest; 

**Authors**: Junbo Li, Zhangyang Wang, Qiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05773)  

**Abstract**: Offline preference alignment for language models such as Direct Preference Optimization (DPO) is favored for its effectiveness and simplicity, eliminating the need for costly reinforcement learning. Various offline algorithms have been developed for different data settings, yet they lack a unified understanding.
In this study, we introduce Pior-Informed Preference Alignment (PIPA), a unified, RL-free probabilistic framework that formulates language model preference alignment as a Maximum Likelihood Estimation (MLE) problem with prior constraints. This method effectively accommodates both paired and unpaired data, as well as answer and step-level annotations. We illustrate that DPO and KTO are special cases with different prior constraints within our framework. By integrating different types of prior information, we developed two variations of PIPA: PIPA-M and PIPA-N. Both algorithms demonstrate a $3\sim10\%$ performance enhancement on the GSM8K and MATH benchmarks across all configurations, achieving these gains without additional training or computational costs compared to existing algorithms. 

**Abstract (ZH)**: Offline Preference Alignment for Language Models: A Unified, RL-Free Probabilistic Framework with Pior-Informed Preference Alignment (PIPA) 

---
# Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails 

**Title (ZH)**: 有效的黑盒多角度攻击突破视觉大型语言模型边界 

**Authors**: Yijun Yang, Lichao Wang, Xiao Yang, Lanqing Hong, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05772)  

**Abstract**: Vision Large Language Models (VLLMs) integrate visual data processing, expanding their real-world applications, but also increasing the risk of generating unsafe responses. In response, leading companies have implemented Multi-Layered safety defenses, including alignment training, safety system prompts, and content moderation. However, their effectiveness against sophisticated adversarial attacks remains largely unexplored. In this paper, we propose MultiFaceted Attack, a novel attack framework designed to systematically bypass Multi-Layered Defenses in VLLMs. It comprises three complementary attack facets: Visual Attack that exploits the multimodal nature of VLLMs to inject toxic system prompts through images; Alignment Breaking Attack that manipulates the model's alignment mechanism to prioritize the generation of contrasting responses; and Adversarial Signature that deceives content moderators by strategically placing misleading information at the end of the response. Extensive evaluations on eight commercial VLLMs in a black-box setting demonstrate that MultiFaceted Attack achieves a 61.56% attack success rate, surpassing state-of-the-art methods by at least 42.18%. 

**Abstract (ZH)**: 多维度攻击：一种系统绕过视觉大型语言模型多层次安全防御的新型攻击框架 

---
# RECOVER: Designing a Large Language Model-based Remote Patient Monitoring System for Postoperative Gastrointestinal Cancer Care 

**Title (ZH)**: RECOVER: 基于大型语言模型的术后胃肠癌患者远程监测系统设计 

**Authors**: Ziqi Yang, Yuxuan Lu, Jennifer Bagdasarian, Vedant Das Swain, Ritu Agarwal, Collin Campbell, Waddah Al-Refaire, Jehan El-Bayoumi, Guodong Gao, Dakuo Wang, Bingsheng Yao, Nawar Shara  

**Link**: [PDF](https://arxiv.org/pdf/2502.05740)  

**Abstract**: Cancer surgery is a key treatment for gastrointestinal (GI) cancers, a group of cancers that account for more than 35% of cancer-related deaths worldwide, but postoperative complications are unpredictable and can be life-threatening. In this paper, we investigate how recent advancements in large language models (LLMs) can benefit remote patient monitoring (RPM) systems through clinical integration by designing RECOVER, an LLM-powered RPM system for postoperative GI cancer care. To closely engage stakeholders in the design process, we first conducted seven participatory design sessions with five clinical staff and interviewed five cancer patients to derive six major design strategies for integrating clinical guidelines and information needs into LLM-based RPM systems. We then designed and implemented RECOVER, which features an LLM-powered conversational agent for cancer patients and an interactive dashboard for clinical staff to enable efficient postoperative RPM. Finally, we used RECOVER as a pilot system to assess the implementation of our design strategies with four clinical staff and five patients, providing design implications by identifying crucial design elements, offering insights on responsible AI, and outlining opportunities for future LLM-powered RPM systems. 

**Abstract (ZH)**: 最近的大语言模型进展如何通过临床集成益于胃肠癌术后远程患者监测系统：RECOVER的设计与评估 

---
# Mitigating Sensitive Information Leakage in LLMs4Code through Machine Unlearning 

**Title (ZH)**: 通过机器遗忘技术缓解LLMs4Code中敏感信息泄露问题 

**Authors**: Ruotong Geng, Mingyang Geng, Shangwen Wang, Haotian Wang, Zhipeng Lin, Dezun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.05739)  

**Abstract**: Large Language Models for Code (LLMs4Code) excel at code generation tasks, yielding promise to release developers from huge software development burdens. Nonetheless, these models have been shown to suffer from the significant privacy risks due to the potential leakage of sensitive information embedded during training, known as the memorization problem. Addressing this issue is crucial for ensuring privacy compliance and upholding user trust, but till now there is a dearth of dedicated studies in the literature that focus on this specific direction. Recently, machine unlearning has emerged as a promising solution by enabling models to "forget" sensitive information without full retraining, offering an efficient and scalable approach compared to traditional data cleaning methods. In this paper, we empirically evaluate the effectiveness of unlearning techniques for addressing privacy concerns in this http URL, we investigate three state-of-the-art unlearning algorithms and three well-known open-sourced LLMs4Code, on a benchmark that takes into consideration both the privacy data to be forgotten as well as the code generation capabilites of these models. Results show that it is feasible to mitigate the privacy concerns of LLMs4Code through machine unlearning while maintain their code generation capabilities at the same time. We also dissect the forms of privacy protection/leakage after unlearning and observe that there is a shift from direct leakage to indirect leakage, which underscores the need for future studies addressing this risk. 

**Abstract (ZH)**: 大规模语言模型在代码生成中的去学习技术对于缓解隐私担忧的有效性研究 

---
# Context information can be more important than reasoning for time series forecasting with a large language model 

**Title (ZH)**: 大规模语言模型时间序列预测中，上下文信息的重要性可能超过推理 

**Authors**: Janghoon Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05699)  

**Abstract**: With the evolution of large language models (LLMs), there is growing interest in leveraging LLMs for time series tasks. In this paper, we explore the characteristics of LLMs for time series forecasting by considering various existing and proposed prompting techniques. Forecasting for both short and long time series was evaluated. Our findings indicate that no single prompting method is universally applicable. It was also observed that simply providing proper context information related to the time series, without additional reasoning prompts, can achieve performance comparable to the best-performing prompt for each case. From this observation, it is expected that providing proper context information can be more crucial than a prompt for specific reasoning in time series forecasting. Several weaknesses in prompting for time series forecasting were also identified. First, LLMs often fail to follow the procedures described by the prompt. Second, when reasoning steps involve simple algebraic calculations with several operands, LLMs often fail to calculate accurately. Third, LLMs sometimes misunderstand the semantics of prompts, resulting in incomplete responses. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的演进，越来越多的研究兴趣在于利用LLMs进行时间序列任务。本文通过考虑各种现有的和提议的提示技术，探讨了LLMs在时间序列预测中的特性。评估了从短期到长期时间序列的预测性能。研究结果表明，并不存在一种普遍适用的提示方法。观察还表明，仅提供与时间序列相关的适当背景信息，而无需额外的推理提示，可以实现与每种情况最佳提示相当的性能。从这一观察中，预期提供适当背景信息比针对特定推理的提示更为重要。此外，还识别出时间序列预测提示中的若干弱点。首先，LLMs经常不遵循提示描述的程序。其次，当推理步骤涉及多个操作数的简单代数计算时，LLMs往往无法准确计算。第三，LLMs有时会误解提示的语义，导致回答不完整。 

---
# Zero-Shot End-to-End Relation Extraction in Chinese: A Comparative Study of Gemini, LLaMA and ChatGPT 

**Title (ZH)**: 零样本端到端中文关系提取：Gemini、LLaMA和ChatGPT的比较研究 

**Authors**: Shaoshuai Du, Yiyi Tao, Yixian Shen, Hang Zhang, Yanxin Shen, Xinyu Qiu, Chuanqi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05694)  

**Abstract**: This study investigates the performance of various large language models (LLMs) on zero-shot end-to-end relation extraction (RE) in Chinese, a task that integrates entity recognition and relation extraction without requiring annotated data. While LLMs show promise for RE, most prior work focuses on English or assumes pre-annotated entities, leaving their effectiveness in Chinese RE largely unexplored. To bridge this gap, we evaluate ChatGPT, Gemini, and LLaMA based on accuracy, efficiency, and adaptability. ChatGPT demonstrates the highest overall performance, balancing precision and recall, while Gemini achieves the fastest inference speed, making it suitable for real-time applications. LLaMA underperforms in both accuracy and latency, highlighting the need for further adaptation. Our findings provide insights into the strengths and limitations of LLMs for zero-shot Chinese RE, shedding light on trade-offs between accuracy and efficiency. This study serves as a foundation for future research aimed at improving LLM adaptability to complex linguistic tasks in Chinese NLP. 

**Abstract (ZH)**: 本研究探究了各类大型语言模型（LLMs）在汉语零样本端到端关系抽取（RE）任务中的表现，该任务结合了实体识别和关系抽取，无需标注数据。虽然LLMs在RE方面显示出潜力，但大部分先前工作主要集中在英语上或假设预先标注的实体，使其在中国RE的有效性方面鲜有研究。为填补这一空白，我们基于准确率、效率和适应性对ChatGPT、Gemini和LLaMA进行了评估。ChatGPT在整体性能上表现最佳，平衡了精确率和召回率；Gemini实现最快推理速度，适用于实时应用；LLaMA在准确率和延迟方面表现不佳，突显了进一步适应的必要性。我们的研究提供了关于LLMs在零样本汉语RE中的优缺点的见解，并探讨了准确率和效率之间的权衡。本研究为未来旨在提高LLMs适应复杂汉语语言任务的NLP研究奠定了基础。 

---
# Language Models Largely Exhibit Human-like Constituent Ordering Preferences 

**Title (ZH)**: 语言模型在成分排序上 largely 展现人类-like 的偏好。 

**Authors**: Ada Defne Tur, Gaurav Kamath, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2502.05670)  

**Abstract**: Though English sentences are typically inflexible vis-à-vis word order, constituents often show far more variability in ordering. One prominent theory presents the notion that constituent ordering is directly correlated with constituent weight: a measure of the constituent's length or complexity. Such theories are interesting in the context of natural language processing (NLP), because while recent advances in NLP have led to significant gains in the performance of large language models (LLMs), much remains unclear about how these models process language, and how this compares to human language processing. In particular, the question remains whether LLMs display the same patterns with constituent movement, and may provide insights into existing theories on when and how the shift occurs in human language. We compare a variety of LLMs with diverse properties to evaluate broad LLM performance on four types of constituent movement: heavy NP shift, particle movement, dative alternation, and multiple PPs. Despite performing unexpectedly around particle movement, LLMs generally align with human preferences around constituent ordering. 

**Abstract (ZH)**: 尽管英语句子在词序方面通常缺乏灵活性，但构成成分在排列顺序上表现出极大的变异性。一种 prominently 理论提出，构成成分的排列顺序与其重量——衡量构成成分长度或复杂性的指标——直接相关。此类理论在自然语言处理（NLP）的背景下具有重要意义，因为虽然近年来NLP的进步显著提高了大规模语言模型（LLMs）的性能，但仍有许多关于LLMs如何处理语言以及这与人类语言处理的差异之处尚不清楚的问题。特别是，关于构成成分移动模式的问题仍未解答，LLMs 是否表现出与人类相同的模式，这个问题可能为现有理论提供洞见，说明构成成分移动在人类语言中何时以及如何发生。我们比较了具有多样化属性的各种LLMs，以评估它们在四种构成成分移动类型（重NP移动、部分移动、宾语转换以及多个介词短语）上的广泛表现。尽管在部分移动方面表现不佳，但LLMs 通常与人类在构成成分排列上的偏好保持一致。 

---
# CODESIM: Multi-Agent Code Generation and Problem Solving through Simulation-Driven Planning and Debugging 

**Title (ZH)**: CODESIM: 多代理代码生成与问题求解通过基于仿真驱动的规划与调试 

**Authors**: Md. Ashraful Islam, Mohammed Eunus Ali, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2502.05664)  

**Abstract**: Large Language Models (LLMs) have made significant strides in code generation and problem solving. Current approaches employ external tool-based iterative debuggers that use compiler or other tool-based runtime feedback to refine coarse programs generated by various methods. However, the effectiveness of these approaches heavily relies on the quality of the initial code generation, which remains an open challenge. In this paper, we introduce CodeSim, a novel multi-agent code generation framework that comprehensively addresses the stages of program synthesis-planning, coding, and debugging-through a human-like perception approach. As human verifies their understanding of any algorithms through visual simulation, CodeSim uniquely features a method of plan verification and internal debugging through the step-by-step simulation of input/output. Extensive experiments across seven challenging competitive problem-solving and program synthesis benchmarks demonstrate CodeSim's remarkable code generation capabilities. Our framework achieves new state-of-the-art (pass@1) results-(HumanEval 95.1%, MBPP 90.7%, APPS 22%, and CodeContests 29.1%). Furthermore, our method shows potential for even greater enhancement when cascaded with external debuggers. To facilitate further research and development in this area, we have open-sourced our framework in this link (this https URL). 

**Abstract (ZH)**: 大型语言模型在代码生成和问题解决方面取得了显著进展。当前的方法通过使用编译器或其他工具的运行时反馈，结合迭代调试工具来逐步优化由各种方法生成的粗糙程序。然而，这些方法的有效性很大程度上取决于初始代码生成的质量，这仍然是一个开放的挑战。在本文中，我们提出了一种名为CodeSim的新型多agent代码生成框架，通过类似于人类感知的方式全面解决了程序合成-规划、编码和调试的各个阶段。就像人类通过视觉仿真验证他们对任何算法的理解一样，CodeSim独特的特征是通过输入/输出的逐步仿真来进行计划验证和内部调试。跨七个具有挑战性的程序合成和问题解决基准的广泛实验表明，CodeSim在代码生成能力方面表现出色。我们的框架在HumanEval 95.1%、MBPP 90.7%、APPS 22%和CodeContests 29.1%等多个方面取得了新的最先进结果。此外，我们的方法与外部调试工具相结合时，显示出更大的增强潜力。为了促进该领域的进一步研究和发展，我们已在下面的链接中开源了我们的框架 (this https URL)。 

---
# KMI: A Dataset of Korean Motivational Interviewing Dialogues for Psychotherapy 

**Title (ZH)**: KMI：韩国动机 interviews 数据集 for 心理治疗对话 

**Authors**: Hyunjong Kim, Suyeon Lee, Yeongjae Cho, Eunseo Ryu, Yohan Jo, Suran Seong, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2502.05651)  

**Abstract**: The increasing demand for mental health services has led to the rise of AI-driven mental health chatbots, though challenges related to privacy, data collection, and expertise persist. Motivational Interviewing (MI) is gaining attention as a theoretical basis for boosting expertise in the development of these chatbots. However, existing datasets are showing limitations for training chatbots, leading to a substantial demand for publicly available resources in the field of MI and psychotherapy. These challenges are even more pronounced in non-English languages, where they receive less attention. In this paper, we propose a novel framework that simulates MI sessions enriched with the expertise of professional therapists. We train an MI forecaster model that mimics the behavioral choices of professional therapists and employ Large Language Models (LLMs) to generate utterances through prompt engineering. Then, we present KMI, the first synthetic dataset theoretically grounded in MI, containing 1,000 high-quality Korean Motivational Interviewing dialogues. Through an extensive expert evaluation of the generated dataset and the dialogue model trained on it, we demonstrate the quality, expertise, and practicality of KMI. We also introduce novel metrics derived from MI theory in order to evaluate dialogues from the perspective of MI. 

**Abstract (ZH)**: AI驱动的心理健康聊天机器人的发展：基于动机访谈理论的专业化挑战与解决方案——以KMI合成数据集为例 

---
# ELMTEX: Fine-Tuning Large Language Models for Structured Clinical Information Extraction. A Case Study on Clinical Reports 

**Title (ZH)**: ELMTEX：大规模语言模型在结构化临床信息提取中的微调。以临床报告为例。 

**Authors**: Aynur Guluzade, Naguib Heiba, Zeyd Boukhers, Florim Hamiti, Jahid Hasan Polash, Yehya Mohamad, Carlos A Velasco  

**Link**: [PDF](https://arxiv.org/pdf/2502.05638)  

**Abstract**: Europe's healthcare systems require enhanced interoperability and digitalization, driving a demand for innovative solutions to process legacy clinical data. This paper presents the results of our project, which aims to leverage Large Language Models (LLMs) to extract structured information from unstructured clinical reports, focusing on patient history, diagnoses, treatments, and other predefined categories. We developed a workflow with a user interface and evaluated LLMs of varying sizes through prompting strategies and fine-tuning. Our results show that fine-tuned smaller models match or surpass larger counterparts in performance, offering efficiency for resource-limited settings. A new dataset of 60,000 annotated English clinical summaries and 24,000 German translations was validated with automated and manual checks. The evaluations used ROUGE, BERTScore, and entity-level metrics. The work highlights the approach's viability and outlines future improvements. 

**Abstract (ZH)**: 欧洲的医疗体系需要增强的互操作性和数字化，推动了对处理遗留临床数据的创新解决方案的需求。本文呈现了我们的项目成果，该项目旨在利用大规模语言模型（LLMs）从未结构化的临床报告中提取结构化信息，重点关注患者病史、诊断、治疗以及其他预定义类别。我们开发了一个工作流并设计了用户界面，通过提示策略和微调评估了不同规模的LLMs。结果表明，微调后的较小模型在性能上与较大的模型相当或超越，为资源有限的环境提供效率。我们还验证了一个包含60,000个标注的英语临床摘要和24,000个德语翻译的新数据集，使用了自动和手动检查。评估使用了ROUGE、BERTScore和实体级别指标。该研究突显了该方法的可行性并概述了未来改进的方向。 

---
# XiHeFusion: Harnessing Large Language Models for Science Communication in Nuclear Fusion 

**Title (ZH)**: XiHeFusion：利用大型语言模型进行核聚变科学传播 

**Authors**: Xiao Wang, Qingquan Yang, Fuling Wang, Qiang Chen, Wentao Wu, Yu Jin, Jingtao Jiang, Liye Jin, Bo Jiang, Dengdi Sun, Wanli Lv, Meiwen Chen, Zehua Chen, Guosheng Xu, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05615)  

**Abstract**: Nuclear fusion is one of the most promising ways for humans to obtain infinite energy. Currently, with the rapid development of artificial intelligence, the mission of nuclear fusion has also entered a critical period of its development. How to let more people to understand nuclear fusion and join in its research is one of the effective means to accelerate the implementation of fusion. This paper proposes the first large model in the field of nuclear fusion, XiHeFusion, which is obtained through supervised fine-tuning based on the open-source large model Qwen2.5-14B. We have collected multi-source knowledge about nuclear fusion tasks to support the training of this model, including the common crawl, eBooks, arXiv, dissertation, etc. After the model has mastered the knowledge of the nuclear fusion field, we further used the chain of thought to enhance its logical reasoning ability, making XiHeFusion able to provide more accurate and logical answers. In addition, we propose a test questionnaire containing 180+ questions to assess the conversational ability of this science popularization large model. Extensive experimental results show that our nuclear fusion dialogue model, XiHeFusion, can perform well in answering science popularization knowledge. The pre-trained XiHeFusion model is released on this https URL. 

**Abstract (ZH)**: 核聚变是人类获取无限能量最具前景的方式之一。随着人工智能的飞速发展，核聚变的任务也进入了发展关键期。让更多人了解核聚变并加入研究是加速其实现的有效手段。本文提出核聚变领域的首个大型模型XiHeFusion，该模型基于开源大型模型Qwen2.5-14B通过监督微调获得。我们收集了关于核聚变任务的多源知识支持该模型的训练，包括common crawl、电子书、arXiv、学位论文等。在模型掌握核聚变领域的知识后，我们进一步通过链式思维增强其逻辑推理能力，使XiHeFusion能够提供更准确和逻辑化的回答。此外，我们提出了包含180多道问题的测试问卷，以评估该科普大型模型的对话能力。大量实验证明，我们的核聚变对话模型XiHeFusion在回答科普知识方面表现良好。预训练的XiHeFusion模型在此处发布：https://url.cn/abcdef 

---
# On Memory Construction and Retrieval for Personalized Conversational Agents 

**Title (ZH)**: 个性化对话代理的 memory 构建与检索 

**Authors**: Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Xufang Luo, Hao Cheng, Dongsheng Li, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.05589)  

**Abstract**: To deliver coherent and personalized experiences in long-term conversations, existing approaches typically perform retrieval augmented response generation by constructing memory banks from conversation history at either the turn-level, session-level, or through summarization techniques. In this paper, we present two key findings: (1) The granularity of memory unit matters: Turn-level, session-level, and summarization-based methods each exhibit limitations in both memory retrieval accuracy and the semantic quality of the retrieved content. (2) Prompt compression methods, such as \textit{LLMLingua-2}, can effectively serve as a denoising mechanism, enhancing memory retrieval accuracy across different granularities. Building on these insights, we propose SeCom, a method that constructs a memory bank with topical segments by introducing a conversation Segmentation model, while performing memory retrieval based on Compressed memory units. Experimental results show that SeCom outperforms turn-level, session-level, and several summarization-based methods on long-term conversation benchmarks such as LOCOMO and Long-MT-Bench+. Additionally, the proposed conversation segmentation method demonstrates superior performance on dialogue segmentation datasets such as DialSeg711, TIAGE, and SuperDialSeg. 

**Abstract (ZH)**: 为了在长期对话中提供连贯且个性化的体验，现有方法通常通过从对话历史中构建记忆库来进行检索增强响应生成，这可在回合级、会话级或通过总结技术中实现。在本文中，我们提出两项关键发现：（1）记忆单元的粒度很重要：回合级、会话级和基于总结的方法在记忆检索准确性和检索内容的语义质量方面均存在局限性。（2）诸如\textit{LLMLingua-2}的提示压缩方法可以有效地作为去噪机制，在不同粒度下提升记忆检索准确率。基于这些见解，我们提出了一种名为SeCom的方法，在引入对话切分模型的基础上，通过压缩记忆单元进行记忆检索，构建话题段落记忆库。实验结果表明，SeCom在LOCOMO和Long-MT-Bench+等长期对话基准测试中优于回合级、会话级及若干基于总结的方法。此外，提出的对话切分方法在DialSeg711、TIAGE和SuperDialSeg等对话切分数据集上表现出色。 

---
# Large Multimodal Models for Low-Resource Languages: A Survey 

**Title (ZH)**: 低资源语言的大规模多模态模型：一种综述 

**Authors**: Marian Lupascu, Ana-Cristina Rogoz, Mihai Sorin Stupariu, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05568)  

**Abstract**: In this survey, we systematically analyze techniques used to adapt large multimodal models (LMMs) for low-resource (LR) languages, examining approaches ranging from visual enhancement and data creation to cross-modal transfer and fusion strategies. Through a comprehensive analysis of 106 studies across 75 LR languages, we identify key patterns in how researchers tackle the challenges of limited data and computational resources. We find that visual information often serves as a crucial bridge for improving model performance in LR settings, though significant challenges remain in areas such as hallucination mitigation and computational efficiency. We aim to provide researchers with a clear understanding of current approaches and remaining challenges in making LMMs more accessible to speakers of LR (understudied) languages. We complement our survey with an open-source repository available at: this https URL. 

**Abstract (ZH)**: 本调查系统地分析了用于适应低资源语言的大型多模态模型的技术，考察了从视觉增强和数据创建到跨模态迁移和融合策略的各种方法。通过对其它75种低资源语言共计106项研究的全面分析，我们指出了研究人员在应对数据和计算资源有限挑战时的关键模式。我们发现，视觉信息经常作为关键桥梁，提高模型在低资源环境中的性能，尽管在减轻幻觉和提高计算效率方面仍面临重大挑战。我们旨在为研究人员提供当前方法和剩余挑战的清晰理解，以使大型多模态模型对 understudied 语言的使用者更加普及。我们还提供了开源仓库：this https URL。 

---
# IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System 

**Title (ZH)**: IndexTTS：一个工业级可控且高效的零shot文本到语音系统 

**Authors**: Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05512)  

**Abstract**: Recently, large language model (LLM) based text-to-speech (TTS) systems have gradually become the mainstream in the industry due to their high naturalness and powerful zero-shot voice cloning this http URL, we introduce the IndexTTS system, which is mainly based on the XTTS and Tortoise model. We add some novel improvements. Specifically, in Chinese scenarios, we adopt a hybrid modeling method that combines characters and pinyin, making the pronunciations of polyphonic characters and long-tail characters controllable. We also performed a comparative analysis of the Vector Quantization (VQ) with Finite-Scalar Quantization (FSQ) for codebook utilization of acoustic speech tokens. To further enhance the effect and stability of voice cloning, we introduce a conformer-based speech conditional encoder and replace the speechcode decoder with BigVGAN2. Compared with XTTS, it has achieved significant improvements in naturalness, content consistency, and zero-shot voice cloning. As for the popular TTS systems in the open-source, such as Fish-Speech, CosyVoice2, FireRedTTS and F5-TTS, IndexTTS has a relatively simple training process, more controllable usage, and faster inference speed. Moreover, its performance surpasses that of these systems. Our demos are available at this https URL. 

**Abstract (ZH)**: 基于大规模语言模型的文本到语音系统：IndexTTS系统的引入及其改进 

---
# Mechanistic Interpretability of Emotion Inference in Large Language Models 

**Title (ZH)**: 大型语言模型中情绪推断的机理可解释性 

**Authors**: Ala N. Tak, Amin Banayeeanzade, Anahita Bolourani, Mina Kian, Robin Jia, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2502.05489)  

**Abstract**: Large language models (LLMs) show promising capabilities in predicting human emotions from text. However, the mechanisms through which these models process emotional stimuli remain largely unexplored. Our study addresses this gap by investigating how autoregressive LLMs infer emotions, showing that emotion representations are functionally localized to specific regions in the model. Our evaluation includes diverse model families and sizes and is supported by robustness checks. We then show that the identified representations are psychologically plausible by drawing on cognitive appraisal theory, a well-established psychological framework positing that emotions emerge from evaluations (appraisals) of environmental stimuli. By causally intervening on construed appraisal concepts, we steer the generation and show that the outputs align with theoretical and intuitive expectations. This work highlights a novel way to causally intervene and precisely shape emotional text generation, potentially benefiting safety and alignment in sensitive affective domains. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在从文本预测人类情绪方面显示出promising的能力，但这些模型处理情绪刺激的机制仍然很大程度上未被探索。我们的研究通过调查自回归LLMs如何推断情绪，表明情绪表示在模型中特定区域功能局部化。我们的评估包括多样化模型家族和规模，并通过稳健性检验予以支持。然后，我们通过引用认知评估理论，一个广泛认可的心理学框架，该框架提出情绪源自对环境刺激的评估（认知评估），来展示识别到的表示具有心理合理性。通过干预构建的认知评估概念，我们操控生成过程，结果显示输出符合理论和直观预期。这项工作强调了一种新的因果干预方式，用于精确塑造情绪性文本生成，可能惠及敏感情感领域中的安全性和对齐。 

---
# Position: LLMs Can be Good Tutors in Foreign Language Education 

**Title (ZH)**: 位置：LLM可以成为外语教育的良师 

**Authors**: Jingheng Ye, Shen Wang, Deqing Zou, Yibo Yan, Kun Wang, Hai-Tao Zheng, Zenglin Xu, Irwin King, Philip S. Yu, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05467)  

**Abstract**: While recent efforts have begun integrating large language models (LLMs) into foreign language education (FLE), they often rely on traditional approaches to learning tasks without fully embracing educational methodologies, thus lacking adaptability to language learning. To address this gap, we argue that LLMs have the potential to serve as effective tutors in FLE. Specifically, LLMs can play three critical roles: (1) as data enhancers, improving the creation of learning materials or serving as student simulations; (2) as task predictors, serving as learner assessment or optimizing learning pathway; and (3) as agents, enabling personalized and inclusive education. We encourage interdisciplinary research to explore these roles, fostering innovation while addressing challenges and risks, ultimately advancing FLE through the thoughtful integration of LLMs. 

**Abstract (ZH)**: 近年来，虽然已经开始将大型语言模型（LLMs）集成到外语教育（FLE）中，但它们往往依赖于传统的学习方法，未能充分采纳教育方法学，因而缺乏对语言学习的适应性。为解决这一问题，我们认为LLMs有潜力作为FLE中的有效导师。具体而言，LLMs可以扮演三种关键角色：（1）数据增强者，提高学习材料的创作或作为学生模拟；（2）任务预测者，作为学习者评估或优化学习路径；（3）代理者，实现个性化和包容性教育。我们鼓励跨学科研究探索这些角色，推动创新，应对挑战和风险，最终通过慎重地整合LLMs推进外语教育的发展。 

---
# Iterative Deepening Sampling for Large Language Models 

**Title (ZH)**: 大型语言模型中的迭代加深采样方法 

**Authors**: Weizhe Chen, Sven Koenig, Bistra Dilkina  

**Link**: [PDF](https://arxiv.org/pdf/2502.05449)  

**Abstract**: The recent release of OpenAI's o1 models and other similar frameworks showcasing test-time scaling laws has demonstrated their exceptional capability to tackle complex reasoning tasks. Inspired by this, subsequent research has revealed that such test-time scaling laws hinge on the model's ability to search both within a single response (intra-response) and across multiple responses (inter-response) during training. Crucially, beyond selecting a single optimal response, the model must also develop robust self-correction capabilities within its own outputs. However, training models to achieve effective self-evaluation and self-correction remains a significant challenge, heavily dependent on the quality of self-reflection data. In this paper, we address this challenge by focusing on enhancing the quality of self-reflection data generation for complex problem-solving, which can subsequently improve the training of next-generation large language models (LLMs). Specifically, we explore how manually triggering a model's self-correction mechanisms can improve performance on challenging reasoning tasks. To this end, we propose a novel iterative deepening sampling algorithm framework designed to enhance self-correction and generate higher-quality samples. Through extensive experiments on Math500 and AIME benchmarks, we demonstrate that our method achieves a higher success rate on difficult tasks and provide detailed ablation studies to analyze its effectiveness across diverse settings. 

**Abstract (ZH)**: OpenAI o1模型及其他类似框架的近期发布展示了它们在复杂推理任务中的卓越能力，这些成就背后的测试时缩放定律依赖于模型在训练过程中不仅能内在搜索单个响应内部的信息，还能跨多个响应搜索信息。模型不仅需要选择最优响应，还需要在自身的输出中发展出稳健的自我纠正能力。然而，训练模型实现有效的自我评估和自我纠正仍是一个重大挑战，高度依赖于自我反思数据的质量。本文通过提高复杂问题解决中自我反思数据生成的质量来应对这一挑战，从而改进下一代大型语言模型（LLMs）的训练。具体而言，我们探讨了手动触发模型的自我纠正机制如何提高复杂推理任务的表现，并提出了一种新的迭代加深采样算法框架以增强自我纠正和生成更高质量的样本。通过在Math500和AIME基准上的大量实验，我们证明了该方法在困难任务上的成功率更高，并进行了详细的消融研究来分析其在不同环境下的有效性。 

---
# SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation 

**Title (ZH)**: SAMGPT：无文本图基础模型的多领域预训练与跨域适应 

**Authors**: Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, Hui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05424)  

**Abstract**: Graphs are able to model interconnected entities in many online services, supporting a wide range of applications on the Web. This raises an important question: How can we train a graph foundational model on multiple source domains and adapt to an unseen target domain? A major obstacle is that graphs from different domains often exhibit divergent characteristics. Some studies leverage large language models to align multiple domains based on textual descriptions associated with the graphs, limiting their applicability to text-attributed graphs. For text-free graphs, a few recent works attempt to align different feature distributions across domains, while generally neglecting structural differences. In this work, we propose a novel Structure Alignment framework for text-free Multi-domain Graph Pre-Training and cross-domain adaptation (SAMGPT). It is designed to learn multi-domain knowledge from graphs originating in multiple source domains, which can then be adapted to address applications in an unseen target domain. Specifically, we introduce a set of structure tokens to harmonize structure-based aggregation across source domains during the pre-training phase. Next, for cross-domain adaptation, we design dual prompts, namely, holistic prompts and specific prompts, which adapt unified multi-domain structural knowledge and fine-grained, domain-specific information, respectively, to a target domain. Finally, we conduct comprehensive experiments on seven public datasets to evaluate and analyze the effectiveness of SAMGPT. 

**Abstract (ZH)**: 基于结构对齐的无文本多域图预训练与跨域适应（SAMGPT） 

---
# The Complexity of Learning Sparse Superposed Features with Feedback 

**Title (ZH)**: 具有反馈的稀疏叠加特征学习的复杂性 

**Authors**: Akash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.05407)  

**Abstract**: The success of deep networks is crucially attributed to their ability to capture latent features within a representation space. In this work, we investigate whether the underlying learned features of a model can be efficiently retrieved through feedback from an agent, such as a large language model (LLM), in the form of relative \textit{triplet comparisons}. These features may represent various constructs, including dictionaries in LLMs or components of a covariance matrix of Mahalanobis distances. We analyze the feedback complexity associated with learning a feature matrix in sparse settings. Our results establish tight bounds when the agent is permitted to construct activations and demonstrate strong upper bounds in sparse scenarios when the agent's feedback is limited to distributional information. We validate our theoretical findings through experiments on two distinct applications: feature recovery from Recursive Feature Machine-trained models and dictionary extraction from sparse autoencoders trained on Large Language Models. 

**Abstract (ZH)**: 深度网络的成功主要归因于其在表示空间中捕捉潜在特征的能力。本文研究了是否可以通过代理（如大型语言模型）的反馈，例如相对的三元比较形式，高效地检索模型中的底层学习特征。这些特征可能代表各种构建块，包括大型语言模型中的字典或马氏距离协方差矩阵的组成部分。我们分析了在稀疏设置中学习特征矩阵的反馈复杂性。我们的结果在代理可以构建激活的情况下建立了紧致边界，并在代理反馈仅限于分配信息的情况下，在稀疏场景中建立了强上限。我们通过两个不同的应用实验验证了我们的理论发现：从递归特征机训练的模型中恢复特征以及从大型语言模型训练的稀疏自编码器中提取字典。 

---
# fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving 

**Title (ZH)**: 精细粒度专家卸载的大规模混合专家服务 

**Authors**: Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05370)  

**Abstract**: Large Language Models (LLMs) have gained immense success in revolutionizing various applications, including content generation, search and recommendation, and AI-assisted operation. To reduce high training costs, Mixture-of-Experts (MoE) architecture has become a popular backbone for modern LLMs. However, despite the benefits, serving MoE-based LLMs experience severe memory inefficiency due to sparsely activated experts. Recent studies propose to offload inactive experts from GPU memory to CPU memory to improve the serving efficiency of MoE models. However, they either incur high inference latency or high model memory footprints due to coarse-grained designs. To tame the latency-memory trade-off in MoE serving, we present fMoE, a fine-grained expert offloading system for MoE serving that achieves low inference latency with memory efficiency. We design fMoE to extract fine-grained expert selection patterns from MoE models and semantic hints from input prompts to efficiently guide expert prefetching, caching, and offloading decisions. fMoE is prototyped on top of HuggingFace Transformers and deployed on a six-GPU testbed. Experiments with open-source MoE models and real-world workloads show that fMoE reduces inference latency by 47% and improves expert hit rate by 36% over state-of-the-art solutions. 

**Abstract (ZH)**: 细粒度专家卸载系统fMoE：实现低推理延迟与高内存效率 

---
# RAG-Verus: Repository-Level Program Verification with LLMs using Retrieval Augmented Generation 

**Title (ZH)**: RAG-Verus：基于检索增强生成的仓库级别程序验证 

**Authors**: Sicheng Zhong, Jiading Zhu, Yifang Tian, Xujie Si  

**Link**: [PDF](https://arxiv.org/pdf/2502.05344)  

**Abstract**: Scaling automated formal verification to real-world projects requires resolving cross-module dependencies and global contexts, which are challenges overlooked by existing function-centric methods. We introduce RagVerus, a framework that synergizes retrieval-augmented generation with context-aware prompting to automate proof synthesis for multi-module repositories, achieving a 27% relative improvement on our novel RepoVBench benchmark -- the first repository-level dataset for Verus with 383 proof completion tasks. RagVerus triples proof pass rates on existing benchmarks under constrained language model budgets, demonstrating a scalable and sample-efficient verification. 

**Abstract (ZH)**: 将形式验证自动化扩展到实际项目需要解决跨模块依赖和全局上下文问题，这是现有函数中心方法忽视的挑战。我们提出了一种名为RagVerus的框架，该框架结合了检索增强生成与上下文感知提示，以自动化多模块仓库的证明合成，在我们新颖的RepoVBench基准上实现了27%的相对改进——这是首个针对Verus的仓库级数据集，包含383个证明完成任务。RagVerus在受限语言模型预算下将现有基准的证明通过率提高三倍，展示了可扩展且样本高效的验证方法。 

---
# Oracular Programming: A Modular Foundation for Building LLM-Enabled Software 

**Title (ZH)**: 先知式编程：构建LLM驱动软件的模块化基础 

**Authors**: Jonathan Laurent, André Platzer  

**Link**: [PDF](https://arxiv.org/pdf/2502.05310)  

**Abstract**: Large Language Models have proved surprisingly effective at solving a wide range of tasks from just a handful of examples. However, their lack of reliability and modularity limits their capacity to tackle large problems that require many steps of reasoning. In response, researchers have proposed advanced pipelines that leverage domain-specific knowledge to chain smaller prompts, provide intermediate feedback and improve performance through search. However, the current complexity of writing, tuning, maintaining and improving such pipelines has limited their sophistication. We propose oracular programming, a foundational paradigm for building LLM-enabled applications that lets domain experts express high-level problem-solving strategies as programs with unresolved choice points. These choice points are resolved at runtime by LLMs, which generalize from user-provided examples of correct and incorrect decisions. An oracular program is composed of three orthogonal components: a strategy that consists in a nondeterministic program with choice points that can be reified into a search tree, a policy that specifies how to navigate this tree with the help of LLM oracles, and a set of demonstrations that describe successful and unsuccessful search tree navigation scenarios across diverse problem instances. Each component is expressed in a dedicated programming language and can be independently improved or substituted. We address the key programming language design challenges of modularly composing oracular programs and enforcing consistency between their components as they evolve. 

**Abstract (ZH)**: 大型语言模型已经证明，在从少量示例中解决广泛任务方面表现出乎意料的有效性。然而，它们的可靠性和模块性限制了它们解决需要多步推理的大型问题的能力。为应对这一挑战，研究人员提出了利用领域特定知识的高级管道，以链式方式组合较小的提示、提供中间反馈并借助搜索提高性能。然而，编写、调整、维护和改进这些管道的当前复杂性限制了它们的复杂性。我们提出了一种新的编程范式——或acular编程，这是一种基于构建LLM驱动应用程序的基础范式，允许领域专家以编程方式表达包含未决选择点的高层次问题解决策略。这些选择点在运行时由LLM解决，LLM从用户提供的正确和错误决策示例中进行泛化。或acular程序由三个正交组件组成：一种策略，其包含非确定性程序并包含可以在运行时构建为搜索树的选择点；一种策略，指定了如何在LLM预言的帮助下导航此搜索树；以及一系列演示，描述了各种问题实例中成功的和不成功的搜索树导航场景。每个组件都用专用编程语言表示，并且可以独立改进或替换。我们解决了模块化组合或acular程序及其组件在演变过程中保持一致性的关键编程语言设计挑战。 

---
# LLMs Can Teach Themselves to Better Predict the Future 

**Title (ZH)**: LLMs可以自我教学以更好地预测未来 

**Authors**: Benjamin Turtel, Danny Franklin, Philipp Schoenegger  

**Link**: [PDF](https://arxiv.org/pdf/2502.05253)  

**Abstract**: We present an outcome-driven fine-tuning framework that enhances the forecasting capabilities of large language models (LLMs) without relying on human-curated reasoning samples. Our method leverages model self-play to generate pairs of diverse reasoning trajectories and probabilistic forecasts for a set of diverse questions that resolve after the models' knowledge cutoff date. We then rank pairs of these reasoning traces by their distance to the actual outcomes before fine-tuning the model via Direct Preference Optimization (DPO). On a separate test set, our approach increases prediction accuracy of Phi-4 14B and DeepSeek-R1 14B by between 7--10\% over a base model and a DPO fine-tuned control model with randomized labels, bringing them on par with forecasting capabilities of much larger frontier models like GPT-4o. 

**Abstract (ZH)**: 我们提出了一种以结果为导向的微调框架，该框架在不依赖于人工标注的推理样本的情况下增强了大型语言模型（LLMs）的预测能力。该方法利用模型自我对弈生成一组具有多样性的推理轨迹和概率预测，这些问题是模型知识截止日期后才能得以解答的多样问题。然后，我们通过直接偏好优化（DPO）对模型进行微调，并根据这些推理轨迹与实际结果的距离对其进行排名。在单独的测试集中，与基线模型和带有随机标签的DPO微调控制模型相比，我们的方法提高了Phi-4 14B和DeepSeek-R1 14B的预测准确性7-10%，使其与GPT-4o等更大规模的前沿模型的预测能力相当。 

---
# GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity? 

**Title (ZH)**: GSM-Infinite: 在无限增加上下文长度和推理复杂度的情况下，你的LLMs表现如何？ 

**Authors**: Yang Zhou, Hongyi Liu, Zhuoming Chen, Yuandong Tian, Beidi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05252)  

**Abstract**: Long-context large language models (LLMs) have recently shown strong performance in information retrieval and long-document QA. However, to tackle the most challenging intellectual problems, LLMs must reason effectively in long and complex contexts (e.g., frontier mathematical research). Studying how LLMs handle increasing reasoning complexity and context length is essential, yet existing benchmarks lack a solid basis for quantitative evaluation. Inspired by the abstraction of GSM-8K problems as computational graphs, and the ability to introduce noise by adding unnecessary nodes and edges, we develop a grade school math problem generator capable of producing arithmetic problems with infinite difficulty and context length under fine-grained control. Using our newly synthesized GSM-Infinite benchmark, we comprehensively evaluate existing LLMs. We find a consistent sigmoid decline in reasoning performance as complexity increases, along with a systematic inference scaling trend: exponentially increasing inference computation yields only linear performance gains. These findings underscore the fundamental limitations of current long-context LLMs and the key challenges in scaling reasoning capabilities. Our GSM-Infinite benchmark provides a scalable and controllable testbed for systematically studying and advancing LLM reasoning in long and complex contexts. 

**Abstract (ZH)**: 长上下文大型语言模型（LLMs）在信息检索和长文档问答中展现了强大的性能。然而，为了解决最具挑战性的智力问题，LLMs 必须在长且复杂的上下文中有效推理（例如，前沿的数学研究）。研究LLMs处理复杂推理和上下文长度的能力至关重要，但现有基准缺乏定量评价的坚实基础。借鉴GSM-8K问题作为计算图的抽象，并通过添加不必要的节点和边引入噪声的能力，我们开发了一种小学数学问题生成器，能够在细粒度控制下生成具有无限复杂性和上下文长度的算术问题。使用我们的新合成GSM-Infinite基准，我们全面评估了现有的LLMs。我们发现复杂性增加时推理性能呈一致的S形下降，并且系统性的推理扩展趋势是：指数增加的推理计算仅带来线性的性能提升。这些发现突显了当前长上下文LLMs的基本局限性以及扩展推理能力的关键挑战。我们的GSM-Infinite基准提供了一个可扩展且可控的测试平台，系统地研究和推进LLMs在长且复杂上下文中的推理能力。 

---
# Evaluating Personality Traits in Large Language Models: Insights from Psychological Questionnaires 

**Title (ZH)**: 大型语言模型中的人格特质评估：心理学问卷的见解 

**Authors**: Pranav Bhandari, Usman Naseem, Amitava Datta, Nicolas Fay, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2502.05248)  

**Abstract**: Psychological assessment tools have long helped humans understand behavioural patterns. While Large Language Models (LLMs) can generate content comparable to that of humans, we explore whether they exhibit personality traits. To this end, this work applies psychological tools to LLMs in diverse scenarios to generate personality profiles. Using established trait-based questionnaires such as the Big Five Inventory and by addressing the possibility of training data contamination, we examine the dimensional variability and dominance of LLMs across five core personality dimensions: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. Our findings reveal that LLMs exhibit unique dominant traits, varying characteristics, and distinct personality profiles even within the same family of models. 

**Abstract (ZH)**: 心理评估工具长期帮助人类理解行为模式。虽然大型语言模型（LLMs）能够生成与人类相媲美的内容，但我们探索它们是否表现出人格特质。为此，本研究将心理评估工具应用于不同场景中的LLMs，以生成人格特征。通过使用基于特质的标准问卷（如五大人格特质问卷）并解决训练数据污染的可能性，我们考察了五大核心人格维度（开放性、尽责性、外向性、宜人性、神经质）上LLMs的维度变异性和主导性。我们的发现表明，即使在同一个模型系列内，LLMs也表现出独特的主导特质、不同的特征和个性特征。 

---
# SEER: Self-Explainability Enhancement of Large Language Models' Representations 

**Title (ZH)**: SEER: 大型语言模型表示的自我解释性增强 

**Authors**: Guanxu Chen, Dongrui Liu, Tao Luo, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2502.05242)  

**Abstract**: Explaining the hidden representations of Large Language Models (LLMs) is a perspective to understand LLMs' underlying inference logic and improve their reliability in application scenarios. However, previous methods introduce external ''black-box'' modules to explain ''black-box'' LLMs, increasing the potential uncertainty and failing to provide faithful explanations. In this paper, we propose a self-explaining method SEER, enhancing LLMs' explainability by aggregating the same concept and disentangling the different concepts in the representation space. In this way, SEER provides faithful explanations carried by representations synchronously with the LLMs' output. Additionally, we showcase the applications of SEER on trustworthiness-related tasks (e.g., the safety risks classification and detoxification tasks), where self-explained LLMs achieve consistent improvement in explainability and performance. More crucially, we theoretically analyze the improvement of SEER on LLMs' generalization ability through optimal transport theory. 

**Abstract (ZH)**: 解释大型语言模型（LLMs）的隐藏表示是一种理解LLMs潜在推理逻辑并提高其在应用场景中可靠性的方式。然而，先前的方法引入了外部“黑盒”模块来解释“黑盒”LLMs，增加了潜在不确定性并无法提供忠实的解释。在本文中，我们提出了一种自解释方法SEER，通过在表示空间中聚合相同的概念并解开不同的概念来增强LLMs的解释性。这样，SEER能够同步LLMs输出提供忠实的解释。此外，我们展示了SEER在与可信性相关任务（例如，安全风险分类和去毒任务）中的应用，自解释的LLMs在解释性和性能上均实现了持续改进。更关键的是，我们通过最优传输理论理论上分析了SEER在提高LLMs泛化能力方面的改进。 

---
# Enhancing Knowledge Graph Construction: Evaluating with Emphasis on Hallucination, Omission, and Graph Similarity Metrics 

**Title (ZH)**: 增强知识图谱构建：以幻觉、遗漏和图相似性指标为重点的评估 

**Authors**: Hussam Ghanem, Christophe Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2502.05239)  

**Abstract**: Recent advancements in large language models have demonstrated significant potential in the automated construction of knowledge graphs from unstructured text. This paper builds upon our previous work [16], which evaluated various models using metrics like precision, recall, F1 score, triple matching, and graph matching, and introduces a refined approach to address the critical issues of hallucination and omission. We propose an enhanced evaluation framework incorporating BERTScore for graph similarity, setting a practical threshold of 95% for graph matching. Our experiments focus on the Mistral model, comparing its original and fine-tuned versions in zero-shot and few-shot settings. We further extend our experiments using examples from the KELM-sub training dataset, illustrating that the fine-tuned model significantly improves knowledge graph construction accuracy while reducing the exact hallucination and omission. However, our findings also reveal that the fine-tuned models perform worse in generalization tasks on the KELM-sub dataset. This study underscores the importance of comprehensive evaluation metrics in advancing the state-of-the-art in knowledge graph construction from textual data. 

**Abstract (ZH)**: 最近在大型语言模型方面的进展显示了其在从非结构化文本自动生成知识图谱方面的显著潜力。本文在此前工作[16]的基础上，利用精确度、召回率、F1分数、三元组匹配和图匹配等指标评估各种模型，并引入一种改进的方法来解决幻觉和遗漏的关键问题。我们提出了一个增强的评估框架，结合使用BERTScore进行图相似性评估，并为图匹配设定一个实际阈值，即95%。我们的实验集中在Mistral模型上，比较了其原版和微调版本在零样本和少样本设置下的性能。我们进一步使用KELM-sub训练数据集的示例进行实验，表明微调模型在提高知识图谱构建准确性的同时，还能减少精确幻觉和遗漏。然而，我们的研究结果还揭示了微调模型在KELM-sub数据集上的泛化任务表现较差。这项研究强调了综合评估指标在从文本数据构建知识图谱方面的前沿技术进步中的重要性。 

---
# Koel-TTS: Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidance 

**Title (ZH)**: Koel-TTS：通过偏好对齐和无分类器引导提高基于LLM的语音生成 

**Authors**: Shehzeen Hussain, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Subhankar Ghosh, Mikyas T. Desta, Roy Fejgin, Rafael Valle, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.05236)  

**Abstract**: While autoregressive speech token generation models produce speech with remarkable variety and naturalness, their inherent lack of controllability often results in issues such as hallucinations and undesired vocalizations that do not conform to conditioning inputs. We introduce Koel-TTS, a suite of enhanced encoder-decoder Transformer TTS models that address these challenges by incorporating preference alignment techniques guided by automatic speech recognition and speaker verification models. Additionally, we incorporate classifier-free guidance to further improve synthesis adherence to the transcript and reference speaker audio. Our experiments demonstrate that these optimizations significantly enhance target speaker similarity, intelligibility, and naturalness of synthesized speech. Notably, Koel-TTS directly maps text and context audio to acoustic tokens, and on the aforementioned metrics, outperforms state-of-the-art TTS models, despite being trained on a significantly smaller dataset. Audio samples and demos are available on our website. 

**Abstract (ZH)**: Koel-TTS：通过偏好对齐技术改进的增强型编码器-解码器Transformer TTS模型 

---
# Optimizing Temperature for Language Models with Multi-Sample Inference 

**Title (ZH)**: 使用多样本推理优化语言模型的温度参数 

**Authors**: Weihua Du, Yiming Yang, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2502.05234)  

**Abstract**: Multi-sample aggregation strategies, such as majority voting and best-of-N sampling, are widely used in contemporary large language models (LLMs) to enhance predictive accuracy across various tasks. A key challenge in this process is temperature selection, which significantly impacts model performance. Existing approaches either rely on a fixed default temperature or require labeled validation data for tuning, which are often scarce and difficult to obtain. This paper addresses the challenge of automatically identifying the (near)-optimal temperature for different LLMs using multi-sample aggregation strategies, without relying on task-specific validation data. We provide a comprehensive analysis of temperature's role in performance optimization, considering variations in model architectures, datasets, task types, model sizes, and predictive accuracy. Furthermore, we propose a novel entropy-based metric for automated temperature optimization, which consistently outperforms fixed-temperature baselines. Additionally, we incorporate a stochastic process model to enhance interpretability, offering deeper insights into the relationship between temperature and model performance. 

**Abstract (ZH)**: 多样本聚合策略中的温度自动优化：无需依赖特定任务验证数据的研究 

---
# A Survey on Backdoor Threats in Large Language Models (LLMs): Attacks, Defenses, and Evaluations 

**Title (ZH)**: 大型语言模型中的后门威胁综述：攻击、防御与评估 

**Authors**: Yihe Zhou, Tao Ni, Wei-Bin Lee, Qingchuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.05224)  

**Abstract**: Large Language Models (LLMs) have achieved significantly advanced capabilities in understanding and generating human language text, which have gained increasing popularity over recent years. Apart from their state-of-the-art natural language processing (NLP) performance, considering their widespread usage in many industries, including medicine, finance, education, etc., security concerns over their usage grow simultaneously. In recent years, the evolution of backdoor attacks has progressed with the advancement of defense mechanisms against them and more well-developed features in the LLMs. In this paper, we adapt the general taxonomy for classifying machine learning attacks on one of the subdivisions - training-time white-box backdoor attacks. Besides systematically classifying attack methods, we also consider the corresponding defense methods against backdoor attacks. By providing an extensive summary of existing works, we hope this survey can serve as a guideline for inspiring future research that further extends the attack scenarios and creates a stronger defense against them for more robust LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在理解和生成人类语言文本方面已取得显著进步，近年来受到越来越多的关注。除了其在自然语言处理（NLP）方面的顶级性能，随着其在医学、金融、教育等多个行业的广泛应用，对其使用的安全关注也不断增加。近年来，随着对抗回门攻击防御机制的进步和LLMs功能的提升，回门攻击本身也不断发展。在本文中，我们采用一般分类法对训练时白盒回门攻击进行分类。除了系统分类攻击方法外，我们还考虑了相应的防御方法。通过广泛总结现有工作，我们希望此综述能够为未来研究提供指导，进一步扩展攻击场景并创建更强的防御机制，以增强LLMs的安全性。 

---
# KDA: A Knowledge-Distilled Attacker for Generating Diverse Prompts to Jailbreak LLMs 

**Title (ZH)**: KDA：一种知识精简攻击者，用于生成多样化的提示以突破LLM 

**Authors**: Buyun Liang, Kwan Ho Ryan Chan, Darshan Thaker, Jinqi Luo, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2502.05223)  

**Abstract**: Jailbreak attacks exploit specific prompts to bypass LLM safeguards, causing the LLM to generate harmful, inappropriate, and misaligned content. Current jailbreaking methods rely heavily on carefully designed system prompts and numerous queries to achieve a single successful attack, which is costly and impractical for large-scale red-teaming. To address this challenge, we propose to distill the knowledge of an ensemble of SOTA attackers into a single open-source model, called Knowledge-Distilled Attacker (KDA), which is finetuned to automatically generate coherent and diverse attack prompts without the need for meticulous system prompt engineering. Compared to existing attackers, KDA achieves higher attack success rates and greater cost-time efficiency when targeting multiple SOTA open-source and commercial black-box LLMs. Furthermore, we conducted a quantitative diversity analysis of prompts generated by baseline methods and KDA, identifying diverse and ensemble attacks as key factors behind KDA's effectiveness and efficiency. 

**Abstract (ZH)**: Jailbreak攻击利用特定提示绕过LLM防护，导致LLM生成有害、不适当和偏颇的内容。当前的Jailbreak方法高度依赖精心设计的系统提示和大量查询以实现一次成功的攻击，这在大规模红队演练中成本高且不实际。为应对这一挑战，我们提出了一种将多种当下最优攻击者知识精简至一个开源模型的方法，称为知识精简攻击者（KDA），该模型通过微调能够自动生成连贯且多样的攻击提示，无需精细的系统提示工程。与现有攻击者相比，KDA在针对多种当下最优的开源和商用黑盒LLM时实现了更高的攻击成功率和更好的成本时间效率。此外，我们对基线方法和KDA生成的提示进行了定量多样性分析，发现多样性和集成攻击是KDA有效性和效率的关键因素。 

---
# Aero-LLM: A Distributed Framework for Secure UAV Communication and Intelligent Decision-Making 

**Title (ZH)**: Aero-LLM：一种安全的无人机通信与智能决策分布式框架 

**Authors**: Balakrishnan Dharmalingam, Rajdeep Mukherjee, Brett Piggott, Guohuan Feng, Anyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05220)  

**Abstract**: Increased utilization of unmanned aerial vehicles (UAVs) in critical operations necessitates secure and reliable communication with Ground Control Stations (GCS). This paper introduces Aero-LLM, a framework integrating multiple Large Language Models (LLMs) to enhance UAV mission security and operational efficiency. Unlike conventional singular LLMs, Aero-LLM leverages multiple specialized LLMs for various tasks, such as inferencing, anomaly detection, and forecasting, deployed across onboard systems, edge, and cloud servers. This dynamic, distributed architecture reduces performance bottleneck and increases security capabilities. Aero-LLM's evaluation demonstrates outstanding task-specific metrics and robust defense against cyber threats, significantly enhancing UAV decision-making and operational capabilities and security resilience against cyber attacks, setting a new standard for secure, intelligent UAV operations. 

**Abstract (ZH)**: 增加无人驾驶航空器（UAV）在关键操作中的利用率 necessitates 安全和可靠的与地面控制站（GCS）通信。本文介绍了Aero-LLM框架，该框架整合了多个大型语言模型 (LLMs) 以增强无人机任务安全性和操作效率。不同于传统的单一LLMs，Aero-LLM 利用多个专门的LLMs 来执行各种任务，例如推理、异常检测和预测，并部署在机载系统、边缘和云服务器上。这一动态分布式架构减少了性能瓶颈并增强了安全性。Aero-LLM 的评估展示了出色的任务特定指标和 robust 的对抗网络安全威胁能力，显著提升了无人机决策能力和在针对网络安全攻击方面的安全韧性，确立了安全智能无人机操作的新标准。 

---
# DERMARK: A Dynamic, Efficient and Robust Multi-bit Watermark for Large Language Models 

**Title (ZH)**: DERMARK: 一种动态、高效且 robust 的多比特水印用于大型语言模型 

**Authors**: Qihao Lin, Chen Tang, Lan zhang, Junyang zhang, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.05213)  

**Abstract**: Well-trained large language models (LLMs) present significant risks, including potential malicious use and copyright infringement. Current studies aim to trace the distribution of LLM-generated texts by implicitly embedding watermarks. Among these, the single-bit watermarking method can only determine whether a given text was generated by an LLM. In contrast, the multi-bit watermarking method embeds richer information into the generated text, which can identify which LLM generated and distributed a given text to which user. However, existing efforts embed the multi-bit watermark directly into the generated text without accounting for its watermarking capacity. This approach can result in embedding failures when the text's watermarking capacity is insufficient. In this paper, we derive the watermark embedding distribution based on the logits of LLMs and propose a formal inequality to segment the text optimally for watermark embedding. Building on this foundation, we propose DERMARK, a dynamic, efficient, and robust multi-bit watermarking method. DERMARK divides the text into segments of varying lengths for each bit embedding, adaptively matching the text's capacity. It achieves this with negligible overhead and robust performance against text editing by minimizing watermark extraction loss. Comprehensive experiments demonstrate that, compared to the SOTA method, our method reduces the number of tokens required for embedding each bit by 20\%, reduces watermark embedding time by 50\%, and is robust to text editing and watermark erasure attacks. 

**Abstract (ZH)**: 基于LLM输出_logits的动态高效稳健多比特水印方法 

---
# Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities 

**Title (ZH)**: 模型篡改攻击促使对大语言模型能力进行更严格的评估 

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell  

**Link**: [PDF](https://arxiv.org/pdf/2502.05209)  

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, a fundamental limitation of this approach is that the harmfulness of the behaviors identified during any particular evaluation can only lower bound the model's worst-possible-case behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the attack success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together these results highlight the difficulty of removing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone. We release models at this https URL 

**Abstract (ZH)**: 大型语言模型（LLM）风险与能力的评估越来越多地被纳入AI风险管理与治理框架。当前，大多数风险评估是通过设计输入来引发系统有害行为来进行的。然而，这种方法的一个基本局限是，在任何特定评估中识别的有害行为的严重性只能对模型的最坏情况行为提供下界。作为引发有害行为的补充方法，我们提出使用模型篡改攻击来评估LLM，这种攻击允许对潜在激活或权重进行修改。我们将最先进的去除有害LLM能力的技术与一套5种输入空间攻击和6种模型篡改攻击进行了对比。除了相互基准测试这些方法外，我们还 Demonstrate（展示）了如下几点：（1）模型对能力引发攻击的抗性依赖于一个低维度的稳健性子空间；（2）模型篡改攻击的成功率可以实证预测和提供保留输入空间攻击成功的保守估计；（3）最先进的遗忘方法可以在16步调优内轻松被逆转。这些结果 Highlights（强调）了去除有害LLM能力的难度，并表明模型篡改攻击比单独使用输入空间攻击能够实现更为严格的评估。我们在此https://链接中发布了模型。 

---
# Safety at Scale: A Comprehensive Survey of Large Model Safety 

**Title (ZH)**: 大规模模型安全综述：安全性保障 

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Henghui Ding, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05206)  

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models. 

**Abstract (ZH)**: 大型模型的迅速发展：通过大规模预训练学习和泛化的卓越能力重塑了人工智能的格局。这些模型现在广泛应用于会话AI、推荐系统、自动驾驶、内容生成、医疗诊断和科学研究等多个领域。然而，它们的广泛应用也暴露出了重大的安全风险，引发了关于稳健性、可靠性和伦理影响的担忧。本文综述了当前大型模型安全性研究的现状，涵盖了视觉基础模型（VFMs）、大规模语言模型（LLMs）、视觉-语言预训练（VLP）模型、视觉-语言模型（VLMs）、扩散模型（DMs）以及基于大型模型的代理。我们的贡献总结如下：（1）我们提出了这些模型面临的安全威胁的全面分类，包括对抗攻击、数据投毒、后门攻击、脱狱和提示注入攻击、能量-延迟攻击、数据和模型提取攻击以及新兴的代理特定威胁。（2）我们回顾了针对每种攻击类型提出的防御策略，并总结了常用的数据集和基准测试用于安全性研究。（3）在此基础上，我们识别并讨论了大型模型安全性面临的开放挑战，强调需要进行全面的安全评估、可扩展和有效的防御机制，以及可持续的数据实践。更重要的是，我们强调了研究社区和国际合作的必要性。我们的工作可作为研究人员和实践者的有益参考，促进全面防御系统和平台的发展，以保障AI模型的安全。 

---
# Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies 

**Title (ZH)**: 面向异构词汇的无损推测解码算法加速LLM推理 

**Authors**: Nadav Timor, Jonathan Mamou, Daniel Korat, Moshe Berchansky, Oren Pereg, Gaurav Jain, Roy Schwartz, Moshe Wasserblat, David Harel  

**Link**: [PDF](https://arxiv.org/pdf/2502.05202)  

**Abstract**: Accelerating the inference of large language models (LLMs) is a critical challenge in generative AI. Speculative decoding (SD) methods offer substantial efficiency gains by generating multiple tokens using a single target forward pass. However, existing SD approaches require the drafter and target models to share the same vocabulary, thus limiting the pool of possible drafters, often necessitating the training of a drafter from scratch. We present three new SD methods that remove this shared-vocabulary constraint. All three methods preserve the target distribution (i.e., they are lossless) and work with off-the-shelf models without requiring additional training or modifications. Empirically, on summarization, programming, and long-context tasks, our algorithms achieve significant speedups over standard autoregressive decoding. By enabling any off-the-shelf model to serve as drafter and requiring no retraining, this work substantially broadens the applicability of the SD framework in practice. 

**Abstract (ZH)**: 加速大型语言模型的推理是生成型AI中的一个重要挑战。推测解码（SD）方法通过单次目标前向传播生成多个令牌，从而提供了显著的效率增益。然而，现有的SD方法要求草稿模型和目标模型共享相同的词汇表，从而限制了可能的草稿模型的选择，通常需要从头训练一个草稿模型。我们提出了三种新的SD方法，消除了这一共享词汇表的限制。所有这些方法都保持了目标分布不变（即，它们是无损的），并且可以与即用型模型一起使用，无需额外训练或修改。实验结果显示，在摘要、编程和长上下文任务上，我们的算法在标准自回归解码方法上实现了显著的加速。通过使任何即用型模型均可作为草稿模型，并且不需要重新训练，这项工作显著扩展了推测解码框架在实践中的适用范围。 

---
