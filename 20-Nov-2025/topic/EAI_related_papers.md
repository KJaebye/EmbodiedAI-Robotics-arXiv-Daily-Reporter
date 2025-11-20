# Know Your Intent: An Autonomous Multi-Perspective LLM Agent Framework for DeFi User Transaction Intent Mining 

**Title (ZH)**: 了解你的意图：一种自主多视角LLM代理框架，用于DeFi用户交易意图挖掘 

**Authors**: Qian'ang Mao, Yuxuan Zhang, Jiaman Chen, Wenjun Zhou, Jiaqi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.15456)  

**Abstract**: As Decentralized Finance (DeFi) develops, understanding user intent behind DeFi transactions is crucial yet challenging due to complex smart contract interactions, multifaceted on-/off-chain factors, and opaque hex logs. Existing methods lack deep semantic insight. To address this, we propose the Transaction Intent Mining (TIM) framework. TIM leverages a DeFi intent taxonomy built on grounded theory and a multi-agent Large Language Model (LLM) system to robustly infer user intents. A Meta-Level Planner dynamically coordinates domain experts to decompose multiple perspective-specific intent analyses into solvable subtasks. Question Solvers handle the tasks with multi-modal on/off-chain data. While a Cognitive Evaluator mitigates LLM hallucinations and ensures verifiability. Experiments show that TIM significantly outperforms machine learning models, single LLMs, and single Agent baselines. We also analyze core challenges in intent inference. This work helps provide a more reliable understanding of user motivations in DeFi, offering context-aware explanations for complex blockchain activity. 

**Abstract (ZH)**: 随着去中心化金融（DeFi）的发展，理解DeFi交易背后的用户意图由于复杂的智能合约交互、多方面的链上/链下因素以及不透明的十六进制日志而变得至关重要且具有挑战性。现有方法缺乏深入的语义洞察。为此，我们提出了交易意图挖掘（TIM）框架。TIM利用基于扎根理论构建的DeFi意图分类框架和多agent大型语言模型系统，以 robust地推断用户意图。元级规划师动态协调领域专家，将多视角的具体意图分析分解为可解决的子任务。问题求解器使用多模态链上/链下数据处理任务。认知评估器则减轻了大型语言模型的幻觉，确保了可验证性。实验表明，TIM显著优于机器学习模型、单一的大规模语言模型以及单一智能体基线。我们还分析了意图推断的核心挑战。这项工作有助于提供对DeFi中用户动机更可靠的理解，提供对复杂区块链活动的上下文感知解释。 

---
# IPR-1: Interactive Physical Reasoner 

**Title (ZH)**: IPR-1: 交互物理推理器 

**Authors**: Mingyu Zhang, Lifeng Zhuo, Tianxi Tan, Guocan Xie, Xian Nie, Yan Li, Renjie Zhao, Zizhu He, Ziyu Wang, Jiting Cai, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.15407)  

**Abstract**: Humans learn by observing, interacting with environments, and internalizing physics and causality. Here, we aim to ask whether an agent can similarly acquire human-like reasoning from interaction and keep improving with more experience. We study this in a Game-to-Unseen (G2U) setting, curating 1,000+ heterogeneous games with diverse physical and causal mechanisms, and evaluate at three human-like levels: Survival, Curiosity, Utility, from primitive intuition to goal-driven reasoning. Our analysis reveals complementary failures: VLM/VLA agents reason but lack look-ahead in interactive settings, while world models imagine but imitate visual patterns rather than analyze physics and causality. We therefore propose IPR (Interactive Physical Reasoner), using world-model rollouts to score and reinforce a VLM's policy, and introduce PhysCode, a physics-centric action code aligning semantic intent with dynamics to provide a shared action space for prediction and reasoning. Pretrained on 1,000+ games, our IPR performs robustly on three levels, matches GPT-5 overall, and surpasses it on Curiosity. We find that performance improves with more training games and interaction steps, and that the model also zero-shot transfers to unseen games. These results support physics-centric interaction as a path to steadily improving physical reasoning. 

**Abstract (ZH)**: 人类通过观察、与环境互动并内化物理和因果关系来学习。在这里，我们旨在探索代理是否可以通过互动和积累经验来获得类似人类的推理能力并持续改进。我们研究这一问题是在Game-to-Unseen (G2U) 设置下，收集了1,000多款异质游戏，涵盖多种物理和因果机制，并从基础直觉到目标驱动的推理评估了三个类似人类的层次：生存、好奇心、实用性。我们的分析揭示了互为补充的失败：VLM/VLA代理可以推理但缺乏前瞻规划能力，在互动环境中；世界模型可以想象但模仿视觉模式而非分析物理和因果关系。因此，我们提出了一种交互物理推理器（IPR），使用世界模型的展开来评估并强化VLM的策略，并引入了以物理为中心的动作编码（PhysCode），使语义意图与动力学对齐，为预测和推理提供共享的动作空间。在1,000多款游戏中预训练，我们的IPR在三个层次上表现稳健，并在总体上匹配GPT-5，且在好奇心方面超越了它。我们发现性能随更多训练游戏和交互步骤的增加而提升，并且该模型也能零样本迁移至未见过的游戏。这些结果支持以物理为中心的交互作为持续提升物理推理表现的途径。 

---
# Terra Nova: A Comprehensive Challenge Environment for Intelligent Agents 

**Title (ZH)**: Terra Nova: 一个全面的智能代理挑战环境 

**Authors**: Trevor McInroe  

**Link**: [PDF](https://arxiv.org/pdf/2511.15378)  

**Abstract**: We introduce Terra Nova, a new comprehensive challenge environment (CCE) for reinforcement learning (RL) research inspired by Civilization V. A CCE is a single environment in which multiple canonical RL challenges (e.g., partial observability, credit assignment, representation learning, enormous action spaces, etc.) arise simultaneously. Mastery therefore demands integrated, long-horizon understanding across many interacting variables. We emphasize that this definition excludes challenges that only aggregate unrelated tasks in independent, parallel streams (e.g., learning to play all Atari games at once). These aggregated multitask benchmarks primarily asses whether an agent can catalog and switch among unrelated policies rather than test an agent's ability to perform deep reasoning across many interacting challenges. 

**Abstract (ZH)**: 我们介绍Terra Nova：一种受文明V启发的新型综合挑战环境（CCE）以促进强化学习（RL）研究 

---
# Learning Human-Like RL Agents Through Trajectory Optimization With Action Quantization 

**Title (ZH)**: 通过动作量化进行轨迹优化学习类人类的RL代理 

**Authors**: Jian-Ting Guo, Yu-Cheng Chen, Ping-Chun Hsieh, Kuo-Hao Ho, Po-Wei Huang, Ti-Rong Wu, I-Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15055)  

**Abstract**: Human-like agents have long been one of the goals in pursuing artificial intelligence. Although reinforcement learning (RL) has achieved superhuman performance in many domains, relatively little attention has been focused on designing human-like RL agents. As a result, many reward-driven RL agents often exhibit unnatural behaviors compared to humans, raising concerns for both interpretability and trustworthiness. To achieve human-like behavior in RL, this paper first formulates human-likeness as trajectory optimization, where the objective is to find an action sequence that closely aligns with human behavior while also maximizing rewards, and adapts the classic receding-horizon control to human-like learning as a tractable and efficient implementation. To achieve this, we introduce Macro Action Quantization (MAQ), a human-like RL framework that distills human demonstrations into macro actions via Vector-Quantized VAE. Experiments on D4RL Adroit benchmarks show that MAQ significantly improves human-likeness, increasing trajectory similarity scores, and achieving the highest human-likeness rankings among all RL agents in the human evaluation study. Our results also demonstrate that MAQ can be easily integrated into various off-the-shelf RL algorithms, opening a promising direction for learning human-like RL agents. Our code is available at this https URL. 

**Abstract (ZH)**: 类人代理在追求人工智能中的长期目标之一。尽管强化学习（RL）已在许多领域实现了超人性能，但相对较少关注设计类人的RL代理。因此，许多奖励驱动的RL代理经常表现出与人类不自然的行为，这在可解释性和可信度方面引起了担忧。为了在RL中实现类人行为，本文首先将类人性形式化为轨迹优化，其中目标是找到一个与人类行为紧密对齐且同时最大化奖励的动作序列，并将经典的局部反馈控制适应为类人的学习，从而实现可操作且高效的实施方案。为此，我们引入了宏动作量化（MAQ），这是一种类人RL框架，通过向量量化VAE将人类演示提炼为宏动作。在D4RL Adroit基准上的实验结果显示，MAQ显著提高了类人性，增加了轨迹相似性得分，并在人类评估研究中实现了所有RL代理中最高的类人性排名。我们的结果还表明，MAQ可以轻松整合到各种即用型RL算法中，为学习类人RL代理开启了有希望的方向。我们的代码可在以下链接获取：this https URL。 

---
# Task Specific Sharpness Aware O-RAN Resource Management using Multi Agent Reinforcement Learning 

**Title (ZH)**: 任务特定的敏锐度意识O-RAN资源管理多代理强化学习 

**Authors**: Fatemeh Lotfi, Hossein Rajoli, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2511.15002)  

**Abstract**: Next-generation networks utilize the Open Radio Access Network (O-RAN) architecture to enable dynamic resource management, facilitated by the RAN Intelligent Controller (RIC). While deep reinforcement learning (DRL) models show promise in optimizing network resources, they often struggle with robustness and generalizability in dynamic environments. This paper introduces a novel resource management approach that enhances the Soft Actor Critic (SAC) algorithm with Sharpness-Aware Minimization (SAM) in a distributed Multi-Agent RL (MARL) framework. Our method introduces an adaptive and selective SAM mechanism, where regularization is explicitly driven by temporal-difference (TD)-error variance, ensuring that only agents facing high environmental complexity are regularized. This targeted strategy reduces unnecessary overhead, improves training stability, and enhances generalization without sacrificing learning efficiency. We further incorporate a dynamic $\rho$ scheduling scheme to refine the exploration-exploitation trade-off across agents. Experimental results show our method significantly outperforms conventional DRL approaches, yielding up to a $22\%$ improvement in resource allocation efficiency and ensuring superior QoS satisfaction across diverse O-RAN slices. 

**Abstract (ZH)**: 下一代网络利用开放无线接入网（O-RAN）架构并通过无线接入网智能控制器（RIC）实现动态资源管理。虽然深度强化学习（DRL）模型在优化网络资源方面显示出潜力，但它们在动态环境中的健壮性和泛化能力通常较差。本文介绍了一种新颖的资源管理方法，该方法在分布式多代理强化学习（MARL）框架中增强了Soft Actor Critic（SAC）算法，并结合了Sharpness-Aware Minimization（SAM）机制。该方法引入了一种自适应和选择性的SAM机制，通过时间差分（TD）误差方差显式驱动正则化，确保仅在面对高环境复杂性时才进行正则化。这一目标策略减少了不必要的开销，提高了训练稳定性，并在不牺牲学习效率的情况下增强泛化能力。此外，本文还引入了一种动态$\rho$调度方案，以跨代理优化探索与利用的权衡。实验结果表明，该方法显著优于传统DRL方法，在资源分配效率上提升了高达22%，并在各种O-RAN切片中确保了更优的QoS满意度。 

---
# Learning Interestingness in Automated Mathematical Theory Formation 

**Title (ZH)**: 自动数学理论形成中的兴趣性学习 

**Authors**: George Tsoukalas, Rahul Saha, Amitayush Thakur, Sabrina Reguyal, Swarat Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2511.14778)  

**Abstract**: We take two key steps in automating the open-ended discovery of new mathematical theories, a grand challenge in artificial intelligence. First, we introduce $\emph{FERMAT}$, a reinforcement learning (RL) environment that models concept discovery and theorem-proving using a set of symbolic actions, opening up a range of RL problems relevant to theory discovery. Second, we explore a specific problem through $\emph{FERMAT}$: automatically scoring the $\emph{interestingness}$ of mathematical objects. We investigate evolutionary algorithms for synthesizing nontrivial interestingness measures. In particular, we introduce an LLM-based evolutionary algorithm that features function abstraction, leading to notable improvements in discovering elementary number theory and finite fields over hard-coded baselines. We open-source the $\emph{FERMAT}$ environment at this URL(this https URL). 

**Abstract (ZH)**: 我们在这项工作中共实现了自动化发现新数学理论的两个关键步骤，这是人工智能领域的一项宏伟挑战。首先，我们引入了$\emph{FERMAT}$，一个基于符号操作的强化学习环境，用于建模概念发现和定理证明，从而开启了与理论发现相关的广泛RL问题空间。其次，我们通过$\emph{FERMAT}$探讨了一个具体问题：自动评估数学对象的“有趣性”。我们研究了进化算法以合成非平凡的兴趣度量，并引入了一种基于大语言模型的进化算法，该算法具备函数抽象特性，并在发现初等数论和有限域方面取得了显著进步。我们开源了$\emph{FERMAT}$环境，网址见下：(这个 https URL)。 

---
# In-N-On: Scaling Egocentric Manipulation with in-the-wild and on-task Data 

**Title (ZH)**: In-N-On: 用野生数据和任务中数据扩展第一人称操控规模 

**Authors**: Xiongyi Cai, Ri-Zhao Qiu, Geng Chen, Lai Wei, Isabella Liu, Tianshu Huang, Xuxin Cheng, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15704)  

**Abstract**: Egocentric videos are a valuable and scalable data source to learn manipulation policies. However, due to significant data heterogeneity, most existing approaches utilize human data for simple pre-training, which does not unlock its full potential. This paper first provides a scalable recipe for collecting and using egocentric data by categorizing human data into two categories: in-the-wild and on-task alongside with systematic analysis on how to use the data. We first curate a dataset, PHSD, which contains over 1,000 hours of diverse in-the-wild egocentric data and over 20 hours of on-task data directly aligned to the target manipulation tasks. This enables learning a large egocentric language-conditioned flow matching policy, Human0. With domain adaptation techniques, Human0 minimizes the gap between humans and humanoids. Empirically, we show Human0 achieves several novel properties from scaling human data, including language following of instructions from only human data, few-shot learning, and improved robustness using on-task data. Project website: this https URL 

**Abstract (ZH)**: 以自我为中心的视频是学习操作策略的一种宝贵且可扩展的数据源。然而，由于数据的高度异质性，大多数现有方法仅利用人类数据进行简单的预训练，未能充分发挥其潜力。本文首先提供了一个可扩展的方案，用于收集和利用以自我为中心的数据，通过将人类数据分类为两类：野外数据和任务中数据，并系统分析了如何使用这些数据。我们首先构建了一个数据集PHSD，包含超过1000小时的多样化的野外以自我为中心数据和超过20小时的任务中数据，直接与目标操作任务对齐。这使得能够学习一个大规模的以自我为中心的语言条件流动匹配策略Human0。通过领域适应技术，Human0最小化了人类与类人机器人之间的差距。实验上，我们展示了Human0通过放大人类数据获得了几个新颖的特性，包括仅通过人类数据进行指令跟随、少样本学习以及通过任务中数据提高鲁棒性。项目网站：这个https URL。 

---
# DeepThinkVLA: Enhancing Reasoning Capability of Vision-Language-Action Models 

**Title (ZH)**: DeepThinkVLA: 提升视觉-语言-行动模型的推理能力 

**Authors**: Cheng Yin, Yankai Lin, Wang Xu, Sikyuen Tam, Xiangrui Zeng, Zhiyuan Liu, Zhouping Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.15669)  

**Abstract**: Enabling Vision-Language-Action (VLA) models to "think before acting" via Chain-of-Thought (CoT) is a promising path to overcoming the data-hungry nature of end-to-end robot policies. However, progress is stalled by a fundamental conflict: existing models use a single autoregressive decoder for both sequential CoT reasoning and high-dimensional, parallelizable robot actions. This architectural mismatch degrades motor control and fails to forge a strong causal link between thought and action. We introduce DeepThinkVLA, which resolves this conflict through a tightly integrated architecture and training strategy. Architecturally, our hybrid-attention decoder generates sequential CoT with causal attention and then switches to bidirectional attention for fast, parallel decoding of action vectors. This design is complemented by a two-stage training pipeline: we first use Supervised Fine-Tuning (SFT) to teach the model foundational reasoning, then apply Reinforcement Learning (RL) with task-success rewards to causally align the full reasoning-action sequence with desired outcomes. This synergy leads to state-of-the-art performance, achieving a 97.0% success rate on the LIBERO benchmark. Our ablations confirm the design's effectiveness: the hybrid architecture alone outperforms standard decoders by 15.5%, and the final RL stage provides a crucial 2% boost to secure top performance. 

**Abstract (ZH)**: 通过链式思考（Chain-of-Thought）使视觉-语言-动作（VLA）模型在“思考后再行动”：一种克服端到端机器人策略数据饥饿性质的有希望的方法，但进展受制于根本冲突的解决之路 

---
# VisPlay: Self-Evolving Vision-Language Models from Images 

**Title (ZH)**: VisPlay: 自适应进化视觉-语言模型 

**Authors**: Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, Yonghui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15661)  

**Abstract**: Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained with Group Relative Policy Optimization (GRPO), which incorporates diversity and difficulty rewards to balance the complexity of generated questions with the quality of the silver answers. VisPlay scales efficiently across two model families. When trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks, including MM-Vet and MMMU, demonstrating a scalable path toward self-evolving multimodal intelligence. The project page is available at this https URL 

**Abstract (ZH)**: 强化学习（RL）为通过复杂推理任务提高视觉语言模型（VLMs）提供了原则性的框架。然而，现有的RL方法通常依赖于人工标注的标签或特定任务的启发式方法来定义可验证的奖励，这两种方法都成本高昂且难以扩展。我们提出了一种自进化的RL框架VisPlay，该框架使VLMs能够自主利用大量未标注的图像数据提高其推理能力。VisPlay从一个基础的VLM开始，将其分配为两种交互的角色：一种是基于图像的问题提出者，提出具有挑战性但可回答的视觉问题；另一种是跨模态推理器，生成银级回答。这些角色通过联合训练和组相对策略优化（GRPO）进行训练，这种方法整合了多样性和难度奖励，以平衡生成问题的复杂性和银级回答的质量。VisPlay在两种模型家族中高效扩展。当在Qwen2.5-VL和MiMo-VL上训练时，VisPlay在八个基准测试中，包括MM-Vet和MMMU，在视觉推理、组合泛化和幻觉减少方面取得一致改进，展示了自进化的跨模态智能的可扩展路径。项目的页面可通过该链接访问。 

---
# Continual Reinforcement Learning for Cyber-Physical Systems: Lessons Learned and Open Challenges 

**Title (ZH)**: 持续强化学习在网络物理系统中的应用：经验教训与开放挑战 

**Authors**: Kim N. Nolle, Ivana Dusparic, Rhodri Cusack, Vinny Cahill  

**Link**: [PDF](https://arxiv.org/pdf/2511.15652)  

**Abstract**: Continual learning (CL) is a branch of machine learning that aims to enable agents to adapt and generalise previously learned abilities so that these can be reapplied to new tasks or environments. This is particularly useful in multi-task settings or in non-stationary environments, where the dynamics can change over time. This is particularly relevant in cyber-physical systems such as autonomous driving. However, despite recent advances in CL, successfully applying it to reinforcement learning (RL) is still an open problem.
This paper highlights open challenges in continual RL (CRL) based on experiments in an autonomous driving environment. In this environment, the agent must learn to successfully park in four different scenarios corresponding to parking spaces oriented at varying angles. The agent is successively trained in these four scenarios one after another, representing a CL environment, using Proximal Policy Optimisation (PPO). These experiments exposed a number of open challenges in CRL: finding suitable abstractions of the environment, oversensitivity to hyperparameters, catastrophic forgetting, and efficient use of neural network capacity.
Based on these identified challenges, we present open research questions that are important to be addressed for creating robust CRL systems. In addition, the identified challenges call into question the suitability of neural networks for CL. We also identify the need for interdisciplinary research, in particular between computer science and neuroscience. 

**Abstract (ZH)**: 持续学习在自主驾驶环境中的挑战与研究问题探讨 

---
# Optimus-Q: Utilizing Federated Learning in Adaptive Robots for Intelligent Nuclear Power Plant Operations through Quantum Cryptography 

**Title (ZH)**: Optimus-Q：通过量子密码学在适应性机器人中利用联邦学习以实现智能核电站运营 

**Authors**: Sai Puppala, Ismail Hossain, Jahangir Alam, Sajedul Talukder  

**Link**: [PDF](https://arxiv.org/pdf/2511.15614)  

**Abstract**: The integration of advanced robotics in nuclear power plants (NPPs) presents a transformative opportunity to enhance safety, efficiency, and environmental monitoring in high-stakes environments. Our paper introduces the Optimus-Q robot, a sophisticated system designed to autonomously monitor air quality and detect contamination while leveraging adaptive learning techniques and secure quantum communication. Equipped with advanced infrared sensors, the Optimus-Q robot continuously streams real-time environmental data to predict hazardous gas emissions, including carbon dioxide (CO$_2$), carbon monoxide (CO), and methane (CH$_4$). Utilizing a federated learning approach, the robot collaborates with other systems across various NPPs to improve its predictive capabilities without compromising data privacy. Additionally, the implementation of Quantum Key Distribution (QKD) ensures secure data transmission, safeguarding sensitive operational information. Our methodology combines systematic navigation patterns with machine learning algorithms to facilitate efficient coverage of designated areas, thereby optimizing contamination monitoring processes. Through simulations and real-world experiments, we demonstrate the effectiveness of the Optimus-Q robot in enhancing operational safety and responsiveness in nuclear facilities. This research underscores the potential of integrating robotics, machine learning, and quantum technologies to revolutionize monitoring systems in hazardous environments. 

**Abstract (ZH)**: 先进机器人在核电力植物中的集成：提高高风险环境下的安全、效率和环境监测的变革性机会 

---
# Multimodal Evaluation of Russian-language Architectures 

**Title (ZH)**: 俄语架构的多模态评估 

**Authors**: Artem Chervyakov, Ulyana Isaeva, Anton Emelyanov, Artem Safin, Maria Tikhonova, Alexander Kharitonov, Yulia Lyakh, Petr Surovtsev, Denis Shevelev Vildan Saburov, Vasily Konovalov, Elisei Rykov, Ivan Sviridov, Amina Miftakhova, Ilseyar Alimova, Alexander Panchenko, Alexander Kapitanov, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2511.15552)  

**Abstract**: Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce Mera Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (image-to-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking and licenses for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）目前是研究的中心，展示了在规模和能力上的快速进步，但它们的智能、局限性和风险仍不够了解。为此，特别是在目前俄语缺乏多模态基准的情况下，我们引入了Mera Multi，一个用于俄语架构的开放多模态评估框架。该基准以指令为基础，涵盖了默认的文字、图像、音频和视频模态，包括18项新的评估任务，适用于通用模型和模态特定架构（图像到文本、视频到文本和音频到文本）。我们的贡献包括：(i) 多模态能力的通用分类；(ii) 18个全新的数据集，确保考虑了俄语文化与语言的特定性、统一的提示和指标；(iii) 闭源和开源模型的基线结果；(iv) 防止基准泄露的方法论，包括水印和私有集合的许可证。尽管我们目前的关注点是俄语，但提出的基准提供了一种可复制的方法论，用于构建类型多样的语言的多模态基准，特别是在斯拉夫语族中。 

---
# Path Planning through Multi-Agent Reinforcement Learning in Dynamic Environments 

**Title (ZH)**: 动态环境中的多智能体强化学习路径规划 

**Authors**: Jonas De Maeyer, Hossein Yarahmadi, Moharram Challenger  

**Link**: [PDF](https://arxiv.org/pdf/2511.15284)  

**Abstract**: Path planning in dynamic environments is a fundamental challenge in intelligent transportation and robotics, where obstacles and conditions change over time, introducing uncertainty and requiring continuous adaptation. While existing approaches often assume complete environmental unpredictability or rely on global planners, these assumptions limit scalability and practical deployment in real-world settings. In this paper, we propose a scalable, region-aware reinforcement learning (RL) framework for path planning in dynamic environments. Our method builds on the observation that environmental changes, although dynamic, are often localized within bounded regions. To exploit this, we introduce a hierarchical decomposition of the environment and deploy distributed RL agents that adapt to changes locally. We further propose a retraining mechanism based on sub-environment success rates to determine when policy updates are necessary. Two training paradigms are explored: single-agent Q-learning and multi-agent federated Q-learning, where local Q-tables are aggregated periodically to accelerate the learning process. Unlike prior work, we evaluate our methods in more realistic settings, where multiple simultaneous obstacle changes and increasing difficulty levels are present. Results show that the federated variants consistently outperform their single-agent counterparts and closely approach the performance of A* Oracle while maintaining shorter adaptation times and robust scalability. Although initial training remains time-consuming in large environments, our decentralized framework eliminates the need for a global planner and lays the groundwork for future improvements using deep RL and flexible environment decomposition. 

**Abstract (ZH)**: 在动态环境中的路径规划是智能交通和机器人领域的基本挑战，其中障碍物和条件会随时间变化，引入不确定性并需要持续适应。现有方法通常假设环境完全不可预测或依赖全局规划器，这些假设限制了在实际环境中的可扩展性和部署。本文提出了一种适用于动态环境的可扩展、区域aware的强化学习（RL）路径规划框架。我们的方法基于对环境变化虽然动态但通常局限于限定区域的观察。为此，我们引入了环境的分层分解，并部署了分布式RL代理以当地适应变化。进一步提出了一种基于子环境成功率的重新训练机制，用于确定策略更新的必要性。探索了两种训练范式：单代理Q学习和多代理联邦Q学习，其中局部Q表定期聚合以加速学习过程。与现有工作不同，我们将在包含多个同时障碍变化和增加难度级别的更现实环境中评估我们的方法。结果表明，联邦变体始终优于单代理变体，并且在维护更短的适应时间和稳健的可扩展性的同时接近A* Oracle的性能。尽管在大环境中初始训练仍然耗时，但我们的去中心化框架消除了需要全局规划器的需求，并为未来使用深度RL和灵活的环境分解进行了改进打下了基础。 

---
# Behavior Trees vs Executable Ontologies: a Comparative Analysis of Robot Control Paradigms 

**Title (ZH)**: 行为树 vs 可执行本体：机器人控制范式的比较分析 

**Authors**: Alexander Boldachev  

**Link**: [PDF](https://arxiv.org/pdf/2511.15274)  

**Abstract**: This paper compares two distinct approaches to modeling robotic behavior: imperative Behavior Trees (BTs) and declarative Executable Ontologies (EO), implemented through the boldsea framework. BTs structure behavior hierarchically using control-flow, whereas EO represents the domain as a temporal, event-based semantic graph driven by dataflow rules. We demonstrate that EO achieves comparable reactivity and modularity to BTs through a fundamentally different architecture: replacing polling-based tick execution with event-driven state propagation. We propose that EO offers an alternative framework, moving from procedural programming to semantic domain modeling, to address the semantic-process gap in traditional robotic control. EO supports runtime model modification, full temporal traceability, and a unified representation of data, logic, and interface - features that are difficult or sometimes impossible to achieve with BTs, although BTs excel in established, predictable scenarios. The comparison is grounded in a practical mobile manipulation task. This comparison highlights the respective operational strengths of each approach in dynamic, evolving robotic systems. 

**Abstract (ZH)**: 本文比较了两种不同的机器人行为建模方法：命令式行为树（BTs）和声明式执行本体论（EO），并通过boldsea框架实现。行为树采用控制流将行为分层结构化，而执行本体论以数据流规则驱动的时间事件基语义图来表示领域。我们通过一种从根本上不同的架构，展示了EO在反应性和模块性方面能达到与BTs相当的水平：使用事件驱动的状态传播替代基于轮询的周期执行。我们提出，执行本体论提供了一种替代框架，从过程化编程转向语义领域建模，以弥补传统机器人控制中的语义-过程差距。执行本体论支持运行时模型修改、完整的时序可追溯性以及数据、逻辑和接口的一体化表示 - 虽然行为树在传统上在已建立且可预测的场景中表现出色，但这些功能在行为树中难以实现或有时是不可能实现的。本文的比较基于一个实际的移动操作任务。这种比较突显了每种方法在动态演化的机器人系统中的各自操作优势。 

---
# PresentCoach: Dual-Agent Presentation Coaching through Exemplars and Interactive Feedback 

**Title (ZH)**: 现时教练：通过范例与互动反馈的双重代理呈现指导 

**Authors**: Sirui Chen, Jinsong Zhou, Xinli Xu, Xiaoyu Yang, Litao Guo, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15253)  

**Abstract**: Effective presentation skills are essential in education, professional communication, and public speaking, yet learners often lack access to high-quality exemplars or personalized coaching. Existing AI tools typically provide isolated functionalities such as speech scoring or script generation without integrating reference modeling and interactive feedback into a cohesive learning experience. We introduce a dual-agent system that supports presentation practice through two complementary roles: the Ideal Presentation Agent and the Coach Agent. The Ideal Presentation Agent converts user-provided slides into model presentation videos by combining slide processing, visual-language analysis, narration script generation, personalized voice synthesis, and synchronized video assembly. The Coach Agent then evaluates user-recorded presentations against these exemplars, conducting multimodal speech analysis and delivering structured feedback in an Observation-Impact-Suggestion (OIS) format. To enhance the authenticity of the learning experience, the Coach Agent incorporates an Audience Agent, which simulates the perspective of a human listener and provides humanized feedback reflecting audience reactions and engagement. Together, these agents form a closed loop of observation, practice, and feedback. Implemented on a robust backend with multi-model integration, voice cloning, and error handling mechanisms, the system demonstrates how AI-driven agents can provide engaging, human-centered, and scalable support for presentation skill development in both educational and professional contexts. 

**Abstract (ZH)**: 有效的プレゼンテーションスキルは教育、専門的なコミュニケーション、および公開スピーチにおいて重要であり、しかし多くの学習者は高品質のモデルや個別的な指導にアクセスできません。既存のAIツールは通常、スピーチ検証やスクリプト生成など個别的機能を提供し、引用モデルや互动反馈的整合性学习体验中缺少集成。我们介绍了一个双代理系统，通过互补的角色支持演示实践：理想演示代理和教练代理。理想演示代理通过结合幻灯片处理、视觉语言分析、叙述脚本生成、个性化语音合成以及同步视频编排，将用户提供的幻灯片转换为模型演示视频。接着，教练代理根据这些模型评估用户录制的演示，进行多模态语音分析，并以观察-影响-建议(OIS)格式提供结构化反馈。为了增强学习体验的真实性，教练代理引入了观众代理，模拟人类听众的视角并向用户提供反映观众反应和参与度的人性化反馈。这些代理共同形成了观察、练习和反馈的闭环。在具有多模型集成、语音克隆和错误处理机制的坚固后端实现下，该系统展示了如何通过AI驱动的代理为教育和专业背景下演示技能的发展提供富有吸引力、以人文为中心且可扩展的支持。 

---
# Eq.Bot: Enhance Robotic Manipulation Learning via Group Equivariant Canonicalization 

**Title (ZH)**: Eq.Bot: 通过群共变规范化解增强机器人操作学习 

**Authors**: Jian Deng, Yuandong Wang, Yangfu Zhu, Tao Feng, Tianyu Wo, Zhenzhou Shao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15194)  

**Abstract**: Robotic manipulation systems are increasingly deployed across diverse domains. Yet existing multi-modal learning frameworks lack inherent guarantees of geometric consistency, struggling to handle spatial transformations such as rotations and translations. While recent works attempt to introduce equivariance through bespoke architectural modifications, these methods suffer from high implementation complexity, computational cost, and poor portability. Inspired by human cognitive processes in spatial reasoning, we propose this http URL, a universal canonicalization framework grounded in SE(2) group equivariant theory for robotic manipulation learning. Our framework transforms observations into a canonical space, applies an existing policy, and maps the resulting actions back to the original space. As a model-agnostic solution, this http URL aims to endow models with spatial equivariance without requiring architectural modifications. Extensive experiments demonstrate the superiority of this http URL under both CNN-based (e.g., CLIPort) and Transformer-based (e.g., OpenVLA-OFT) architectures over existing methods on various robotic manipulation tasks, where the most significant improvement can reach 50.0%. 

**Abstract (ZH)**: 机器人操作系统在多个领域中的应用日益增多。然而，现有的多模态学习框架缺乏内在的几何一致性保证，难以处理如旋转和平移等空间变换。尽管近期的研究通过定制的架构修改尝试引入等变性，但这些方法面临着实现复杂性高、计算成本大和移植性差的问题。受人类空间推理认知过程的启发，我们提出了基于SE(2)群等变理论的通用规范框架——this http URL，旨在为机器人操作学习赋予空间等变性而无需进行架构修改。广泛实验证明，与基于CNN（例如CLIPort）和基于Transformer（例如OpenVLA-OFT）的现有方法相比，该框架在各种机器人操作任务中表现出显著优势，最高可提高50.0%。 

---
# Can MLLMs Detect Phishing? A Comprehensive Security Benchmark Suite Focusing on Dynamic Threats and Multimodal Evaluation in Academic Environments 

**Title (ZH)**: MLLMs在学术环境中的动态威胁检测及多模态评估综合安全基准套件：针对钓鱼攻击的能力探究 

**Authors**: Jingzhuo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.15165)  

**Abstract**: The rapid proliferation of Multimodal Large Language Models (MLLMs) has introduced unprecedented security challenges, particularly in phishing detection within academic environments. Academic institutions and researchers are high-value targets, facing dynamic, multilingual, and context-dependent threats that leverage research backgrounds, academic collaborations, and personal information to craft highly tailored attacks. Existing security benchmarks largely rely on datasets that do not incorporate specific academic background information, making them inadequate for capturing the evolving attack patterns and human-centric vulnerability factors specific to academia. To address this gap, we present AdapT-Bench, a unified methodological framework and benchmark suite for systematically evaluating MLLM defense capabilities against dynamic phishing attacks in academic settings. 

**Abstract (ZH)**: Multimodal Large Language Models安全性挑战及其在学术环境中的鱼叉攻击检测：AdapT-Bench统一方法框架与基准测试套件 

---
# Learning Depth from Past Selves: Self-Evolution Contrast for Robust Depth Estimation 

**Title (ZH)**: 从过往自我中学习深度：稳健深度估计的自我进化对比方法 

**Authors**: Jing Cao, Kui Jiang, Shenyi Li, Xiaocheng Feng, Yong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15167)  

**Abstract**: Self-supervised depth estimation has gained significant attention in autonomous driving and robotics. However, existing methods exhibit substantial performance degradation under adverse weather conditions such as rain and fog, where reduced visibility critically impairs depth prediction. To address this issue, we propose a novel self-evolution contrastive learning framework called SEC-Depth for self-supervised robust depth estimation tasks. Our approach leverages intermediate parameters generated during training to construct temporally evolving latency models. Using these, we design a self-evolution contrastive scheme to mitigate performance loss under challenging conditions. Concretely, we first design a dynamic update strategy of latency models for the depth estimation task to capture optimization states across training stages. To effectively leverage latency models, we introduce a self-evolution contrastive Loss (SECL) that treats outputs from historical latency models as negative samples. This mechanism adaptively adjusts learning objectives while implicitly sensing weather degradation severity, reducing the needs for manual intervention. Experiments show that our method integrates seamlessly into diverse baseline models and significantly enhances robustness in zero-shot evaluations. 

**Abstract (ZH)**: 自监督深度估计在自主驾驶和机器人技术中引起了显著关注。然而，现有方法在雨、雾等恶劣天气条件下表现出显著的性能下降，其中降低的能见度严重妨碍了深度预测。为解决这一问题，我们提出了一种新型的自进化对比学习框架SEC-Depth，用于自监督鲁棒深度估计任务。我们的方法利用训练过程中生成的中间参数构建时间演变的延迟模型，并设计了一种自进化对比方案来缓解在恶劣条件下的性能损失。具体而言，我们首先设计了一种深度估计任务的动态更新策略，以捕获训练阶段的优化状态。为了有效利用延迟模型，我们引入了一种自进化对比损失（SECL），将历史延迟模型的输出作为负样本。该机制能够自适应地调整学习目标，隐式感知天气降级严重程度，减少手动干预的需求。实验结果表明，我们的方法能够无缝集成到多种基线模型中，并在零样本评估中显著提高鲁棒性。 

---
# Generating Natural-Language Surgical Feedback: From Structured Representation to Domain-Grounded Evaluation 

**Title (ZH)**: 生成自然语言手术反馈：从结构化表示到领域导向评估 

**Authors**: Firdavs Nasriddinov, Rafal Kocielnik, Anima Anandkumar, Andrew J. Hung  

**Link**: [PDF](https://arxiv.org/pdf/2511.15159)  

**Abstract**: High-quality intraoperative feedback from a surgical trainer is pivotal for improving trainee performance and long-term skill acquisition. Automating natural, trainer-style feedback promises timely, accessible, and consistent guidance at scale but requires models that understand clinically relevant representations. We present a structure-aware pipeline that learns a surgical action ontology from real trainer-to-trainee transcripts (33 surgeries) and uses it to condition feedback generation. We contribute by (1) mining Instrument-Action-Target (IAT) triplets from real-world feedback text and clustering surface forms into normalized categories, (2) fine-tuning a video-to-IAT model that leverages the surgical procedure and task contexts as well as fine-grained temporal instrument motion, and (3) demonstrating how to effectively use IAT triplet representations to guide GPT-4o in generating clinically grounded, trainer-style feedback. We show that, on Task 1: Video-to-IAT recognition, our context injection and temporal tracking deliver consistent AUC gains (Instrument: 0.67 to 0.74; Action: 0.60 to 0.63; Tissue: 0.74 to 0.79). For Task 2: feedback text generation (rated on a 1-5 fidelity rubric where 1 = opposite/unsafe, 3 = admissible, and 5 = perfect match to a human trainer), GPT-4o from video alone scores 2.17, while IAT conditioning reaches 2.44 (+12.4%), doubling the share of admissible generations with score >= 3 from 21% to 42%. Traditional text-similarity metrics also improve: word error rate decreases by 15-31% and ROUGE (phrase/substring overlap) increases by 9-64%. Grounding generation in explicit IAT structure improves fidelity and yields clinician-verifiable rationales, supporting auditable use in surgical training. 

**Abstract (ZH)**: 高质量的手术培训师 intraoperative 反馈对于提高学生成绩和长期技能获取至关重要。自动化的自然式培训师反馈有望提供及时、便捷且一致的指导，但需要理解临床相关表示的模型。我们提出了一种结构感知的管道，从中学习手术动作本体，并利用其来条件化反馈生成。我们通过以下贡献：(1) 从真实世界反馈文本中挖掘器械-动作-目标（IAT）三元组，并将表面形式聚类为标准化类别；(2) 对利用手术程序和任务上下文以及精细时间标度器械运动的视频到IAT模型进行微调；(3) 展示如何有效利用IAT三元组表示来指导GPT-4o生成基于临床的培训师式反馈。结果显示，在任务1：视频到IAT识别中，我们的上下文注入和时间跟踪带来了一致的AUC提升（器械：0.67到0.74；动作：0.60到0.63；组织：0.74到0.79）。在任务2：反馈文本生成（根据1-5保真度量表评分，1=完全相反/不安全，3=可接受，5=完全匹配人类培训师），仅从视频生成的GPT-4o得分为2.17，而IAT条件化得分为2.44 (+12.4%)，生成得分>=3的比例从21%提高到42%。传统的文本相似性指标也有所提高：词错误率降低15-31%，ROUGE（短语/子字符串重叠）增加9-64%。基于明确的IAT结构进行生成提高了保真度并提供了可临床验证的理由，支持在手术培训中的可审计使用。 

---
# Eye Care You: Voice Guidance Application Using Social Robot for Visually Impaired People 

**Title (ZH)**: 你的眼眸：用于视觉障碍人士的社交机器人语音引导应用 

**Authors**: Ting-An Lin, Pei-Lin Tsai, Yi-An Chen, Feng-Yu Chen, Lyn Chao-ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15110)  

**Abstract**: In the study, the device of social robot was designed for visually impaired users, and along with a mobile application for provide functions to assist their lives. Both physical and mental conditions of visually impaired users are considered, and the mobile application provides functions: photo record, mood lift, greeting guest and today highlight. The application was designed for visually impaired users, and uses voice control to provide a friendly interface. Photo record function allows visually impaired users to capture image immediately when they encounter danger situations. Mood lift function accompanies visually impaired users by asking questions, playing music and reading articles. Greeting guest function answers to the visitors for the inconvenient physical condition of visually impaired users. In addition, today highlight function read news including weather forecast, daily horoscopes and daily reminder for visually impaired users. Multiple tools were adopted for developing the mobile application, and a website was developed for caregivers to check statues of visually impaired users and for marketing of the application. 

**Abstract (ZH)**: 基于社交机器人的辅助视障用户移动应用设计与实现 

---
# Aligning Generative Music AI with Human Preferences: Methods and Challenges 

**Title (ZH)**: 面向人类偏好的生成音乐AI：方法与挑战 

**Authors**: Dorien Herremans, Abhinaba Roy  

**Link**: [PDF](https://arxiv.org/pdf/2511.15038)  

**Abstract**: Recent advances in generative AI for music have achieved remarkable fidelity and stylistic diversity, yet these systems often fail to align with nuanced human preferences due to the specific loss functions they use. This paper advocates for the systematic application of preference alignment techniques to music generation, addressing the fundamental gap between computational optimization and human musical appreciation. Drawing on recent breakthroughs including MusicRL's large-scale preference learning, multi-preference alignment frameworks like diffusion-based preference optimization in DiffRhythm+, and inference-time optimization techniques like Text2midi-InferAlign, we discuss how these techniques can address music's unique challenges: temporal coherence, harmonic consistency, and subjective quality assessment. We identify key research challenges including scalability to long-form compositions, reliability amongst others in preference modelling. Looking forward, we envision preference-aligned music generation enabling transformative applications in interactive composition tools and personalized music services. This work calls for sustained interdisciplinary research combining advances in machine learning, music-theory to create music AI systems that truly serve human creative and experiential needs. 

**Abstract (ZH)**: Recent Advances in Preference-Aligned Generative AI for Music 

---
# Simulated Human Learning in a Dynamic, Partially-Observed, Time-Series Environment 

**Title (ZH)**: 模拟在动态、部分可观测的时间序列环境中的人类学习 

**Authors**: Jeffrey Jiang, Kevin Hong, Emily Kuczynski, Gregory Pottie  

**Link**: [PDF](https://arxiv.org/pdf/2511.15032)  

**Abstract**: While intelligent tutoring systems (ITSs) can use information from past students to personalize instruction, each new student is unique. Moreover, the education problem is inherently difficult because the learning process is only partially observable. We therefore develop a dynamic, time-series environment to simulate a classroom setting, with student-teacher interventions - including tutoring sessions, lectures, and exams. In particular, we design the simulated environment to allow for varying levels of probing interventions that can gather more information. Then, we develop reinforcement learning ITSs that combine learning the individual state of students while pulling from population information through the use of probing interventions. These interventions can reduce the difficulty of student estimation, but also introduce a cost-benefit decision to find a balance between probing enough to get accurate estimates and probing so often that it becomes disruptive to the student. We compare the efficacy of standard RL algorithms with several greedy rules-based heuristic approaches to find that they provide different solutions, but with similar results. We also highlight the difficulty of the problem with increasing levels of hidden information, and the boost that we get if we allow for probing interventions. We show the flexibility of both heuristic and RL policies with regards to changing student population distributions, finding that both are flexible, but RL policies struggle to help harder classes. Finally, we test different course structures with non-probing policies and we find that our policies are able to boost the performance of quiz and midterm structures more than we can in a finals-only structure, highlighting the benefit of having additional information. 

**Abstract (ZH)**: 基于探查干预的强化学习智能辅导系统研究 

---
# Skin-R1: Toward Trustworthy Clinical Reasoning for Dermatological Diagnosis 

**Title (ZH)**: Skin-R1: 向可靠的皮肤科诊断临床推理迈进 

**Authors**: Zehao Liu, Wejieying Ren, Jipeng Zhang, Tianxiang Zhao, Jingxi Zhu, Xiaoting Li, Vasant G. Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2511.14900)  

**Abstract**: The emergence of vision-language models (VLMs) has opened new possibilities for clinical reasoning and has shown promising performance in dermatological diagnosis. However, their trustworthiness and clinical utility are often limited by three major factors: (1) Data heterogeneity, where diverse datasets lack consistent diagnostic labels and clinical concept annotations; (2) Absence of grounded diagnostic rationales, leading to a scarcity of reliable reasoning supervision; and (3) Limited scalability and generalization, as models trained on small, densely annotated datasets struggle to transfer nuanced reasoning to large, sparsely-annotated ones.
To address these limitations, we propose SkinR1, a novel dermatological VLM that combines deep, textbook-based reasoning with the broad generalization capabilities of reinforcement learning (RL). SkinR1 systematically resolves the key challenges through a unified, end-to-end framework. First, we design a textbook-based reasoning generator that synthesizes high-fidelity, hierarchy-aware, and differential-diagnosis (DDx)-informed trajectories, providing reliable expert-level supervision. Second, we leverage the constructed trajectories for supervised fine-tuning (SFT) empowering the model with grounded reasoning ability. Third, we develop a novel RL paradigm that, by incorporating the hierarchical structure of diseases, effectively transfers these grounded reasoning patterns to large-scale, sparse data. Extensive experiments on multiple dermatology datasets demonstrate that SkinR1 achieves superior diagnostic accuracy. The ablation study demonstrates the importance of the reasoning foundation instilled by SFT. 

**Abstract (ZH)**: 视觉语言模型的出现为临床推理开启了新的可能性，并已在皮肤科诊断中展现出 promising 的性能。然而，它们的信任度和临床应用往往受限于三个主要因素：(1) 数据异质性，多种多样的数据集缺乏一致的诊断标签和临床概念注释；(2) 缺乏基于现实的诊断推理理由，导致可靠的推理监督稀缺；(3) 缺乏可扩展性和泛化能力，训练于小规模密集标注数据集的模型难以将细腻的推理迁移到大规模稀疏标注数据集上。

为解决这些局限性，我们提出了 SkinR1，一种结合深度教科书推理与强化学习广泛泛化能力的新型皮肤科视觉语言模型。SkinR1 通过一个统一的端到端框架系统性地解决了关键挑战。首先，我们设计了一个基于教科书的推理生成器，合成高保真、层次意识和鉴别诊断（DDx）知情的轨迹，提供了可靠的专家级监督。其次，我们利用合成的轨迹进行监督微调（SFT），赋予模型基于现实的推理能力。最后，我们开发了一种新的强化学习范式，通过引入疾病的层次结构，有效地将这些基于现实的推理模式转移到大规模稀疏数据上。在多个皮肤科数据集上的大量实验表明，SkinR1 在诊断准确性方面表现出色。消融研究证明了 SFT 培养的推理基础的重要性。 

---
# ExplainRec: Towards Explainable Multi-Modal Zero-Shot Recommendation with Preference Attribution and Large Language Models 

**Title (ZH)**: ExplainRec: 基于偏好归因和大规模语言模型的可解释多模态零样本推荐 

**Authors**: Bo Ma, LuYao Liu, ZeHua Hu, Simon Lau  

**Link**: [PDF](https://arxiv.org/pdf/2511.14770)  

**Abstract**: Recent advances in Large Language Models (LLMs) have opened new possibilities for recommendation systems, though current approaches such as TALLRec face challenges in explainability and cold-start scenarios. We present ExplainRec, a framework that extends LLM-based recommendation capabilities through preference attribution, multi-modal fusion, and zero-shot transfer learning. The framework incorporates four technical contributions: preference attribution tuning for explainable recommendations, zero-shot preference transfer for cold-start users and items, multi-modal enhancement leveraging visual and textual content, and multi-task collaborative optimization. Experimental evaluation on MovieLens-25M and Amazon datasets shows that ExplainRec outperforms existing methods, achieving AUC improvements of 0.7\% on movie recommendation and 0.9\% on cross-domain tasks, while generating interpretable explanations and handling cold-start scenarios effectively. 

**Abstract (ZH)**: Recent Advances in Large Language Models for Recommendation Systems: ExplainRec Framework Through Preference Attribution, Multi-Modal Fusion, and Zero-Shot Transfer Learning 

---
# Causally-Informed Reinforcement Learning for Adaptive Emotion-Aware Social Media Recommendation 

**Title (ZH)**: 因果驱动的强化学习在自适应情绪感知社交媒体推荐中应用 

**Authors**: Bhavika Jain, Robert Pitsko, Ananya Drishti, Mahfuza Farooque  

**Link**: [PDF](https://arxiv.org/pdf/2511.14768)  

**Abstract**: Social media recommendation systems play a central role in shaping users' emotional experiences. However, most systems are optimized solely for engagement metrics, such as click rate, viewing time, or scrolling, without accounting for users' emotional states. Repeated exposure to emotionally charged content has been shown to negatively affect users' emotional well-being over time. We propose an Emotion-aware Social Media Recommendation (ESMR) framework that personalizes content based on users' evolving emotional trajectories. ESMR integrates a Transformer-based emotion predictor with a hybrid recommendation policy: a LightGBM model for engagement during stable periods and a reinforcement learning agent with causally informed rewards when negative emotional states persist. Through behaviorally grounded evaluation over 30-day interaction traces, ESMR demonstrates improved emotional recovery, reduced volatility, and strong engagement retention. ESMR offers a path toward emotionally aware recommendations without compromising engagement performance. 

**Abstract (ZH)**: 基于情感意识的社会媒体推荐框架（ESMR）在塑造用户情感体验中的作用与其优化研究 

---
