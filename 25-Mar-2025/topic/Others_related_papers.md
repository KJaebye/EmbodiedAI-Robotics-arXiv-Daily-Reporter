# Learning Multi-Robot Coordination through Locality-Based Factorized Multi-Agent Actor-Critic Algorithm 

**Title (ZH)**: 基于局部性因素化的多智能体演员-评论家算法学习多机器人协调 

**Authors**: Chak Lam Shek, Amrit Singh Bedi, Anjon Basak, Ellen Novoseller, Nick Waytowich, Priya Narayanan, Dinesh Manocha, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2503.18816)  

**Abstract**: In this work, we present a novel cooperative multi-agent reinforcement learning method called \textbf{Loc}ality based \textbf{Fac}torized \textbf{M}ulti-Agent \textbf{A}ctor-\textbf{C}ritic (Loc-FACMAC). Existing state-of-the-art algorithms, such as FACMAC, rely on global reward information, which may not accurately reflect the quality of individual robots' actions in decentralized systems. We integrate the concept of locality into critic learning, where strongly related robots form partitions during training. Robots within the same partition have a greater impact on each other, leading to more precise policy evaluation. Additionally, we construct a dependency graph to capture the relationships between robots, facilitating the partitioning process. This approach mitigates the curse of dimensionality and prevents robots from using irrelevant information. Our method improves existing algorithms by focusing on local rewards and leveraging partition-based learning to enhance training efficiency and performance. We evaluate the performance of Loc-FACMAC in three environments: Hallway, Multi-cartpole, and Bounded-Cooperative-Navigation. We explore the impact of partition sizes on the performance and compare the result with baseline MARL algorithms such as LOMAQ, FACMAC, and QMIX. The experiments reveal that, if the locality structure is defined properly, Loc-FACMAC outperforms these baseline algorithms up to 108\%, indicating that exploiting the locality structure in the actor-critic framework improves the MARL performance. 

**Abstract (ZH)**: 基于局部性的因子化多智能体actor-critic方法(Loc-FACMAC) 

---
# Multi-agent coordination for data gathering with periodic requests and deliveries 

**Title (ZH)**: 周期性请求与交付的数据收集多agent协调 

**Authors**: Yaroslav Marchukov, Luis Montano  

**Link**: [PDF](https://arxiv.org/pdf/2503.18546)  

**Abstract**: In this demo work we develop a method to plan and coordinate a multi-agent team to gather information on demand. The data is periodically requested by a static Operation Center (OC) from changeable goals locations. The mission of the team is to reach these locations, taking measurements and delivering the data to the OC. Due to the limited communication range as well as signal attenuation because of the obstacles, the agents must travel to the OC, to upload the data. The agents can play two roles: ones as workers gathering data, the others as collectors traveling invariant paths for collecting the data of the workers to re-transmit it to the OC. The refreshing time of the delivered information depends on the number of available agents as well as of the scenario. The proposed algorithm finds out the best balance between the number of collectors-workers and the partition of the scenario into working areas in the planning phase, which provides the minimum refreshing time and will be the one executed by the agents. 

**Abstract (ZH)**: 本演示工作开发了一种方法，计划和协调多代理团队以按需收集信息。数据由静态的操作中心（OC）从可变目标位置处周期性请求。团队的任务是到达这些位置，进行测量并将数据传达给OC。由于存在有限的通信范围以及由于障碍物引起的信号衰减，代理必须前往OC上传数据。代理可以扮演两种角色：一部分作为采集数据的工人，另一部分作为收集者，沿不变路径采集工人的数据并重新传输给OC。所提供信息的刷新时间取决于可用代理的数量以及场景本身。所提出的算法在规划阶段确定采集者-工人数目的最佳平衡以及将场景划分为工作区域的方式，以实现最小的刷新时间并由代理执行。 

---
# Aportes para el cumplimiento del Reglamento (UE) 2024/1689 en robótica y sistemas autónomos 

**Title (ZH)**: 关于执行欧盟条例（UE）2024/1689在机器人和自主系统领域的贡献 

**Authors**: Francisco J. Rodríguez Lera, Yoana Pita Lorenzo, David Sobrín Hidalgo, Laura Fernández Becerra, Irene González Fernández, Jose Miguel Guerrero Hernández  

**Link**: [PDF](https://arxiv.org/pdf/2503.17730)  

**Abstract**: Cybersecurity in robotics stands out as a key aspect within Regulation (EU) 2024/1689, also known as the Artificial Intelligence Act, which establishes specific guidelines for intelligent and automated systems. A fundamental distinction in this regulatory framework is the difference between robots with Artificial Intelligence (AI) and those that operate through automation systems without AI, since the former are subject to stricter security requirements due to their learning and autonomy capabilities. This work analyzes cybersecurity tools applicable to advanced robotic systems, with special emphasis on the protection of knowledge bases in cognitive architectures. Furthermore, a list of basic tools is proposed to guarantee the security, integrity, and resilience of these systems, and a practical case is presented, focused on the analysis of robot knowledge management, where ten evaluation criteria are defined to ensure compliance with the regulation and reduce risks in human-robot interaction (HRI) environments. 

**Abstract (ZH)**: 机器人领域的网络安全在欧盟《人工智能法案》（Regulation (EU) 2024/1689）中崭露头角，该法案为智能和自动化系统制定了具体指导原则。这一监管框架中的一个关键区别在于，具有人工智能（AI）的机器人与仅通过自动化系统运行且不包含AI的机器人之间的区别，因为前者的安全要求更为严格，这是由于它们具备学习和自主能力。本文分析适用于高级机器人系统的网络安全工具，特别强调认知架构中知识库的保护。此外，提出了一套基本工具，以确保这些系统的安全、完整性和韧性，并呈现了一个实用案例，专注于机器人知识管理分析，定义了十个评估标准以确保遵守法规并降低人类-机器人交互（HRI）环境中的风险。 

---
# Sense4FL: Vehicular Crowdsensing Enhanced Federated Learning for Autonomous Driving 

**Title (ZH)**: Sense4FL: 车载众包增强的联邦学习方法用于自动驾驶 

**Authors**: Yanan Ma, Senkang Hu, Zhengru Fang, Yun Ji, Yiqin Deng, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17697)  

**Abstract**: To accommodate constantly changing road conditions, real-time model training is essential for autonomous driving (AD). Federated learning (FL) serves as a promising paradigm to enable autonomous vehicles to train models collaboratively with their onboard computing resources. However, existing vehicle selection schemes for FL all assume predetermined and location-independent vehicles' datasets, neglecting the fact that vehicles collect training data along their routes, thereby resulting in suboptimal vehicle selection. To improve the perception quality in AD for a region, we propose Sense4FL, a vehicular crowdsensing-enhanced FL framework featuring trajectory-dependent vehicular training data collection. To this end, we first derive the convergence bound of FL by considering the impact of both vehicles' uncertain trajectories and uploading probabilities, from which we discover that minimizing the training loss is equivalent to minimizing a weighted sum of local and global earth mover's distance (EMD) between vehicles' collected data distribution and global data distribution. Based on this observation, we formulate the trajectory-dependent vehicle selection and data collection problem for FL in AD. Given that the problem is NP-hard, we develop an efficient algorithm to find the solution with an approximation guarantee. Extensive simulation results have demonstrated the effectiveness of our approach in improving object detection performance compared with existing benchmarks. 

**Abstract (ZH)**: 基于轨迹感知的增强联邦学习框架Sense4FL：面向自动驾驶的自 Crowdsensing-Enhanced Federated Learning Framework for Autonomous Driving: Trajectory-Aware Vehicle Selection and Data Collection 

---
# Latent Embedding Adaptation for Human Preference Alignment in Diffusion Planners 

**Title (ZH)**: 潜空间嵌入适应以实现扩散计划中的人类偏好对齐 

**Authors**: Wen Zheng Terence Ng, Jianda Chen, Yuan Xu, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18347)  

**Abstract**: This work addresses the challenge of personalizing trajectories generated in automated decision-making systems by introducing a resource-efficient approach that enables rapid adaptation to individual users' preferences. Our method leverages a pretrained conditional diffusion model with Preference Latent Embeddings (PLE), trained on a large, reward-free offline dataset. The PLE serves as a compact representation for capturing specific user preferences. By adapting the pretrained model using our proposed preference inversion method, which directly optimizes the learnable PLE, we achieve superior alignment with human preferences compared to existing solutions like Reinforcement Learning from Human Feedback (RLHF) and Low-Rank Adaptation (LoRA). To better reflect practical applications, we create a benchmark experiment using real human preferences on diverse, high-reward trajectories. 

**Abstract (ZH)**: 本工作通过引入一种资源高效的方法来解决自动决策系统生成轨迹个性化的问题，该方法能够快速适应个体用户的偏好。我们的方法利用一个在大规模无奖励离线数据集上预训练的条件扩散模型，并结合偏好潜在嵌入（PLE）。PLE 作为紧凑表示，用于捕获特定用户偏好。通过使用我们提出的偏好反转方法对预训练模型进行适应，直接优化可学习的 PLE，我们实现了与现有解决方案（如基于人类反馈的强化学习 RLHF 和低秩适应 LoRA）相比更好的人类偏好对齐。为了更好地反映实际应用，我们使用多样且高奖励的轨迹上真实人类偏好的基准实验进行评估。 

---
# Likelihood Reward Redistribution 

**Title (ZH)**: 奖励 likelihood 重分布 

**Authors**: Minheng Xiao, Zhenbang Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.17409)  

**Abstract**: In many practical reinforcement learning scenarios, feedback is provided only at the end of a long horizon, leading to sparse and delayed rewards. Existing reward redistribution methods typically assume that per-step rewards are independent, thus overlooking interdependencies among state--action pairs. In this paper, we propose a \emph{Likelihood Reward Redistribution} (LRR) framework that addresses this issue by modeling each per-step reward with a parametric probability distribution whose parameters depend on the state--action pair. By maximizing the likelihood of the observed episodic return via a leave-one-out (LOO) strategy that leverages the entire trajectory, our framework inherently introduces an uncertainty regularization term into the surrogate objective. Moreover, we show that the conventional mean squared error (MSE) loss for reward redistribution emerges as a special case of our likelihood framework when the uncertainty is fixed under the Gaussian distribution. When integrated with an off-policy algorithm such as Soft Actor-Critic, LRR yields dense and informative reward signals, resulting in superior sample efficiency and policy performance on Box-2d and MuJoCo benchmarks. 

**Abstract (ZH)**: 在长时间_horizon_末提供反馈的强化学习场景中，奖励反馈稀疏且延迟。现有的奖励再分配方法通常假设每步奖励是独立的，从而忽略了状态-动作对之间的依赖性。本文提出了一种基于似然奖励再分配（LRR）框架，通过使用参数概率分布来建模每步奖励，该分布的参数依赖于状态-动作对，从而解决了这一问题。通过利用完整轨迹实现逐点剔除（LOO）策略来最大化观察到的 episodic 返回值，我们的框架内生地引入了一个不确定性正则化项到代理目标函数中。此外，我们证明，在高斯分布的不确定性固定时，我们的似然框架退化为传统的均方误差（MSE）损失函数。将LRR与Soft Actor-Critic等离策略算法结合使用，可以在Box-2d和MuJoCo基准测试中获得稠密且信息丰富的奖励信号，从而显著提高样本效率和策略性能。 

---
# IRef-VLA: A Benchmark for Interactive Referential Grounding with Imperfect Language in 3D Scenes 

**Title (ZH)**: IRef-VLA: 一种针对3D场景中不完美语言的交互式参照定位基准 

**Authors**: Haochen Zhang, Nader Zantout, Pujith Kachana, Ji Zhang, Wenshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17406)  

**Abstract**: With the recent rise of large language models, vision-language models, and other general foundation models, there is growing potential for multimodal, multi-task robotics that can operate in diverse environments given natural language input. One such application is indoor navigation using natural language instructions. However, despite recent progress, this problem remains challenging due to the 3D spatial reasoning and semantic understanding required. Additionally, the language used may be imperfect or misaligned with the scene, further complicating the task. To address this challenge, we curate a benchmark dataset, IRef-VLA, for Interactive Referential Vision and Language-guided Action in 3D Scenes with imperfect references. IRef-VLA is the largest real-world dataset for the referential grounding task, consisting of over 11.5K scanned 3D rooms from existing datasets, 7.6M heuristically generated semantic relations, and 4.7M referential statements. Our dataset also contains semantic object and room annotations, scene graphs, navigable free space annotations, and is augmented with statements where the language has imperfections or ambiguities. We verify the generalizability of our dataset by evaluating with state-of-the-art models to obtain a performance baseline and also develop a graph-search baseline to demonstrate the performance bound and generation of alternatives using scene-graph knowledge. With this benchmark, we aim to provide a resource for 3D scene understanding that aids the development of robust, interactive navigation systems. The dataset and all source code is publicly released at this https URL. 

**Abstract (ZH)**: 基于有缺陷参考的3D场景交互式参考视觉与语言引导行动基准数据集IRef-VLA 

---
# Reachable Sets-based Trajectory Planning Combining Reinforcement Learning and iLQR 

**Title (ZH)**: 基于可达集的轨迹规划：结合强化学习和iLQR方法 

**Authors**: Wenjie Huang, Yang Li, Shijie Yuan, Jingjia Teng, Hongmao Qin, Yougang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2503.17398)  

**Abstract**: The driving risk field is applicable to more complex driving scenarios, providing new approaches for safety decision-making and active vehicle control in intricate environments. However, existing research often overlooks the driving risk field and fails to consider the impact of risk distribution within drivable areas on trajectory planning, which poses challenges for enhancing safety. This paper proposes a trajectory planning method for intelligent vehicles based on the risk reachable set to further improve the safety of trajectory planning. First, we construct the reachable set incorporating the driving risk field to more accurately assess and avoid potential risks in drivable areas. Then, the initial trajectory is generated based on safe reinforcement learning and projected onto the reachable set. Finally, we introduce a trajectory planning method based on a constrained iterative quadratic regulator to optimize the initial solution, ensuring that the planned trajectory achieves optimal comfort, safety, and efficiency. We conduct simulation tests of trajectory planning in high-speed lane-changing scenarios. The results indicate that the proposed method can guarantee trajectory comfort and driving efficiency, with the generated trajectory situated outside high-risk boundaries, thereby ensuring vehicle safety during operation. 

**Abstract (ZH)**: 基于风险可达集的智能车辆轨迹规划方法研究 

---
# CP-NCBF: A Conformal Prediction-based Approach to Synthesize Verified Neural Control Barrier Functions 

**Title (ZH)**: CP-NCBF：一种基于齐性预测的方法合成验证神经控制壁垒函数 

**Authors**: Manan Tayal, Aditya Singh, Pushpak Jagtap, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2503.17395)  

**Abstract**: Control Barrier Functions (CBFs) are a practical approach for designing safety-critical controllers, but constructing them for arbitrary nonlinear dynamical systems remains a challenge. Recent efforts have explored learning-based methods, such as neural CBFs (NCBFs), to address this issue. However, ensuring the validity of NCBFs is difficult due to potential learning errors. In this letter, we propose a novel framework that leverages split-conformal prediction to generate formally verified neural CBFs with probabilistic guarantees based on a user-defined error rate, referred to as CP-NCBF. Unlike existing methods that impose Lipschitz constraints on neural CBF-leading to scalability limitations and overly conservative safe sets--our approach is sample-efficient, scalable, and results in less restrictive safety regions. We validate our framework through case studies on obstacle avoidance in autonomous driving and geo-fencing of aerial vehicles, demonstrating its ability to generate larger and less conservative safe sets compared to conventional techniques. 

**Abstract (ZH)**: 基于split-conformal预测的正式验证神经控制 Barrier 函数（CP-NCBF）框架：应用于自主驾驶中的障碍物避障和航空器的地理围栏 

---
# Statistical Proof of Execution (SPEX) 

**Title (ZH)**: 执行的统计证明（SPEX） 

**Authors**: Michele Dallachiesa, Antonio Pitasi, David Pinger, Josh Goodbody, Luis Vaello  

**Link**: [PDF](https://arxiv.org/pdf/2503.18899)  

**Abstract**: Many real-world applications are increasingly incorporating automated decision-making, driven by the widespread adoption of ML/AI inference for planning and guidance. This study examines the growing need for verifiable computing in autonomous decision-making. We formalize the problem of verifiable computing and introduce a sampling-based protocol that is significantly faster, more cost-effective, and simpler than existing methods. Furthermore, we tackle the challenges posed by non-determinism, proposing a set of strategies to effectively manage common scenarios. 

**Abstract (ZH)**: 越来越多的实际应用通过普及使用机器学习/人工智能推断来进行规划和指导，逐步纳入了自动决策机制。本研究探讨了在自主决策中对可验证计算 growing need for verifiable computing 的日益增长需求。我们形式化了可验证计算的问题，并引入了一种采样为基础的协议，该协议比现有方法更快、更经济且更简单。此外，我们应对非确定性带来的挑战，提出了一套有效的策略来有效管理常见场景。 

---
# Structuring Scientific Innovation: A Framework for Modeling and Discovering Impactful Knowledge Combinations 

**Title (ZH)**: 结构化科学创新：发现影响性知识组合的模型与框架 

**Authors**: Junlan Chen, Kexin Zhang, Daifeng Li, Yangyang Feng, Yuxuan Zhang, Bowen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2503.18865)  

**Abstract**: The emergence of large language models offers new possibilities for structured exploration of scientific knowledge. Rather than viewing scientific discovery as isolated ideas or content, we propose a structured approach that emphasizes the role of method combinations in shaping disruptive insights. Specifically, we investigate how knowledge unit--especially those tied to methodological design--can be modeled and recombined to yield research this http URL proposed framework addresses two key challenges. First, we introduce a contrastive learning-based mechanism to identify distinguishing features of historically disruptive method combinations within problem-driven this http URL, we propose a reasoning-guided Monte Carlo search algorithm that leverages the chain-of-thought capability of LLMs to identify promising knowledge recombinations for new problem statements.Empirical studies across multiple domains show that the framework is capable of modeling the structural dynamics of innovation and successfully highlights combinations with high disruptive this http URL research provides a new path for computationally guided scientific ideation grounded in structured reasoning and historical data modeling. 

**Abstract (ZH)**: 大型语言模型的出现为结构化探索科学知识提供了新可能性。不同于将科学发现视为孤立的思想或内容，我们提出了一种强调方法组合在形成颠覆性见解中的作用的结构化方法。具体而言，我们探讨了如何建模和重组知识单元（尤其是与方法设计相关联的知识单元）以产生创新性的研究成果。该框架解决了两个关键挑战。首先，我们引入了一种对比学习机制，以识别问题导向型 históricamente颠覆性方法组合的特征。其次，我们提出了一种基于推理指导的蒙特卡洛搜索算法，利用大语言模型的链式思考能力来识别新问题陈述中具有潜力的知识重组组合。跨多个领域的实证研究表明，该框架能够建模创新的结构动态，并成功地突出显示高颠覆性知识组合。该研究为基于结构化推理和历史数据建模的计算导向型科学构想提供了新路径。 

---
# Self-Organizing Graph Reasoning Evolves into a Critical State for Continuous Discovery Through Structural-Semantic Dynamics 

**Title (ZH)**: 自我组织图推理演化为通过结构语义动态持续发现的关键状态 

**Authors**: Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2503.18852)  

**Abstract**: We report fundamental insights into how agentic graph reasoning systems spontaneously evolve toward a critical state that sustains continuous semantic discovery. By rigorously analyzing structural (Von Neumann graph entropy) and semantic (embedding) entropy, we identify a subtle yet robust regime in which semantic entropy persistently dominates over structural entropy. This interplay is quantified by a dimensionless Critical Discovery Parameter that stabilizes at a small negative value, indicating a consistent excess of semantic entropy. Empirically, we observe a stable fraction (12%) of "surprising" edges, links between semantically distant concepts, providing evidence of long-range or cross-domain connections that drive continuous innovation. Concomitantly, the system exhibits scale-free and small-world topological features, alongside a negative cross-correlation between structural and semantic measures, reinforcing the analogy to self-organized criticality. These results establish clear parallels with critical phenomena in physical, biological, and cognitive complex systems, revealing an entropy-based principle governing adaptability and continuous innovation. Crucially, semantic richness emerges as the underlying driver of sustained exploration, despite not being explicitly used by the reasoning process. Our findings provide interdisciplinary insights and practical strategies for engineering intelligent systems with intrinsic capacities for long-term discovery and adaptation, and offer insights into how model training strategies can be developed that reinforce critical discovery. 

**Abstract (ZH)**: 我们报告了关于如何agency图推理系统自发进化至维持持续语义发现的临界状态的基本见解。通过对结构（冯·诺伊曼图熵）和语义（嵌入）熵进行严格分析，我们识别出一种微妙但稳健的区域，在该区域中持续的语义熵始终占主导地位。这种相互作用通过无量纲的关键发现参数进行量化，该参数稳定在一个略小于零的值，表明持续存在的语义熵过剩。实证上，我们观察到稳定的“惊讶”边的比例（12%），即语义上相距甚远的概念之间的连接，提供了长程或跨域连接的证据，这些连接推动了持续的创新。同时，系统表现出无标度和小世界拓扑特征，且结构和语义测量之间的负交叉相关性，强化了自我组织临界性的类比。这些结果建立了与物理、生物和认知复杂系统中临界现象的明确类比，揭示了指导适应性和持续创新的熵基原理。关键的是，语义丰富性成为持续探索的潜在驱动因素，即使推理过程并未显式使用这一信息。我们的发现为工程具有长期发现和适应能力的智能系统提供了跨学科的见解和实用策略，并为开发强化关键发现的模型训练策略提供了启示。 

---
# Towards Responsible AI Music: an Investigation of Trustworthy Features for Creative Systems 

**Title (ZH)**: 负责任的AI音乐：创意系统中可信赖特征的研究 

**Authors**: Jacopo de Berardinis, Lorenzo Porcaro, Albert Meroño-Peñuela, Angelo Cangelosi, Tess Buckley  

**Link**: [PDF](https://arxiv.org/pdf/2503.18814)  

**Abstract**: Generative AI is radically changing the creative arts, by fundamentally transforming the way we create and interact with cultural artefacts. While offering unprecedented opportunities for artistic expression and commercialisation, this technology also raises ethical, societal, and legal concerns. Key among these are the potential displacement of human creativity, copyright infringement stemming from vast training datasets, and the lack of transparency, explainability, and fairness mechanisms. As generative systems become pervasive in this domain, responsible design is crucial. Whilst previous work has tackled isolated aspects of generative systems (e.g., transparency, evaluation, data), we take a comprehensive approach, grounding these efforts within the Ethics Guidelines for Trustworthy Artificial Intelligence produced by the High-Level Expert Group on AI appointed by the European Commission - a framework for designing responsible AI systems across seven macro requirements. Focusing on generative music AI, we illustrate how these requirements can be contextualised for the field, addressing trustworthiness across multiple dimensions and integrating insights from the existing literature. We further propose a roadmap for operationalising these contextualised requirements, emphasising interdisciplinary collaboration and stakeholder engagement. Our work provides a foundation for designing and evaluating responsible music generation systems, calling for collaboration among AI experts, ethicists, legal scholars, and artists. This manuscript is accompanied by a website: this https URL. 

**Abstract (ZH)**: 生成式AI正从根本上改变创意艺术，通过根本性地改变我们创造和互动的文化 artefacts 方式。虽然这项技术为艺术表达和商业化提供了前所未有的机会，但也引发了伦理、社会和法律方面的担忧。这些担忧的关键包括人类创造力可能被取代、源于大规模训练数据的版权侵权以及透明度、可解释性和公平机制的缺失。随着生成系统在这一领域变得普遍，负责任的设计至关重要。尽管以往的工作集中在生成系统孤立的方面（如透明度、评估、数据），我们采取了全面的方法，将这些努力建立在欧洲委员会任命的高级专家小组关于AI的可信赖AI伦理指南之上——这是一个涵盖七大宏观要求的设计负责任AI系统的框架。针对生成音乐AI，我们展示了如何将这些要求具体化，并从现有文献中整合跨多个维度的信任要素，进一步提出实施这些具体化要求的路线图，强调跨学科合作和利益相关者参与。我们的研究为设计和评估负责任的音乐生成系统奠定了基础，呼吁人工智能专家、伦理学家、法学学者和艺术家之间的合作。本文附带一个网站：该链接。 

---
# From Fragment to One Piece: A Survey on AI-Driven Graphic Design 

**Title (ZH)**: 从碎片到完整：基于AI的图形设计综述 

**Authors**: Xingxing Zou, Wen Zhang, Nanxuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.18641)  

**Abstract**: This survey provides a comprehensive overview of the advancements in Artificial Intelligence in Graphic Design (AIGD), focusing on integrating AI techniques to support design interpretation and enhance the creative process. We categorize the field into two primary directions: perception tasks, which involve understanding and analyzing design elements, and generation tasks, which focus on creating new design elements and layouts. The survey covers various subtasks, including visual element perception and generation, aesthetic and semantic understanding, layout analysis, and generation. We highlight the role of large language models and multimodal approaches in bridging the gap between localized visual features and global design intent. Despite significant progress, challenges remain to understanding human intent, ensuring interpretability, and maintaining control over multilayered compositions. This survey serves as a guide for researchers, providing information on the current state of AIGD and potential future directions\footnote{this https URL\_Intelligent\_graphic\_design}. 

**Abstract (ZH)**: 这项调查提供了人工智能在图形设计（AIGD）领域的全面综述，重点在于集成AI技术以支持设计解释并增强创造性过程。我们将该领域分为两个主要方向：感知任务，涉及理解和分析设计元素；以及生成任务，专注于创建新设计元素和布局。调查涵盖了各种子任务，包括视觉元素的感知与生成、审美与语义理解、布局分析与生成。我们强调了大型语言模型和多模态方法在链接局部视觉特征与全局设计意图方面的作用。尽管取得了显著进展，但理解人类意图、确保可解释性以及控制多层组合仍面临挑战。该调查为研究人员提供了一份指南，提供了AIGD当前状态以及潜在未来方向的信息。 

---
# The Role of Artificial Intelligence in Enhancing Insulin Recommendations and Therapy Outcomes 

**Title (ZH)**: 人工智能在增强胰岛素建议和治疗效果中的作用 

**Authors**: Maria Panagiotou, Knut Stroemmen, Lorenzo Brigato, Bastiaan E. de Galan, Stavroula Mougiakakou  

**Link**: [PDF](https://arxiv.org/pdf/2503.18592)  

**Abstract**: The growing worldwide incidence of diabetes requires more effective approaches for managing blood glucose levels. Insulin delivery systems have advanced significantly, with artificial intelligence (AI) playing a key role in improving their precision and adaptability. AI algorithms, particularly those based on reinforcement learning, allow for personalised insulin dosing by continuously adapting to an individual's responses. Despite these advancements, challenges such as data privacy, algorithm transparency, and accessibility still need to be addressed. Continued progress and validation in AI-driven insulin delivery systems promise to improve therapy outcomes further, offering people more effective and individualised management of their diabetes. This paper presents an overview of current strategies, key challenges, and future directions. 

**Abstract (ZH)**: 全球糖尿病发病率的日益增长需要更有效的血糖管理方法。胰岛素输送系统有了显著进步，其中人工智能（AI）在提高其精准度和适应性方面扮演了关键角色。基于强化学习的AI算法特别允许个性化胰岛素剂量，通过不断适应个体的反应。尽管取得了这些进展，数据隐私、算法透明度和可访问性等挑战仍需解决。基于人工智能的胰岛素输送系统的持续进步和验证有望进一步改善治疗效果，为糖尿病患者提供更有效和个性化的管理方案。本文综述了当前策略、关键挑战和未来方向。 

---
# Neuro-symbolic Weak Supervision: Theory and Semantics 

**Title (ZH)**: 神经符号弱监督：理论与语义 

**Authors**: Nijesh Upreti, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2503.18509)  

**Abstract**: Weak supervision allows machine learning models to learn from limited or noisy labels, but it introduces challenges in interpretability and reliability - particularly in multi-instance partial label learning (MI-PLL), where models must resolve both ambiguous labels and uncertain instance-label mappings. We propose a semantics for neuro-symbolic framework that integrates Inductive Logic Programming (ILP) to improve MI-PLL by providing structured relational constraints that guide learning. Within our semantic characterization, ILP defines a logical hypothesis space for label transitions, clarifies classifier semantics, and establishes interpretable performance standards. This hybrid approach improves robustness, transparency, and accountability in weakly supervised settings, ensuring neural predictions align with domain knowledge. By embedding weak supervision into a logical framework, we enhance both interpretability and learning, making weak supervision more suitable for real-world, high-stakes applications. 

**Abstract (ZH)**: 弱监督允许机器学习模型从有限或噪声标签中学习，但在多实例部分标签学习（MI-PLL）中引入了可解释性和可靠性方面的挑战——其中模型必须解决既含模糊标签又含不确定实例-标签映射的问题。我们提出了一种神经符号框架的语义，通过整合归纳逻辑编程（ILP）来改进MI-PLL，提供结构化的关系约束以指导学习。在我们的语义表征中，ILP定义了标签转换的逻辑假设空间，澄清了分类器语义，并建立了可解释的性能标准。这种混合方法提高了弱监督设置中的鲁棒性、透明性和问责性，确保神经预测与领域知识一致。通过将弱监督嵌入到逻辑框架中，我们增强了可解释性和学习能力，使弱监督更适用于现实世界的高风险应用场景。 

---
# DiffMove: Group Mobility Tendency Enhanced Trajectory Recovery via Diffusion Model 

**Title (ZH)**: DiffMove：通过扩散模型增强群体移动倾向的轨迹恢复 

**Authors**: Qingyue Long, Can Rong, Huandong Wang, Shaw Rajib, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.18302)  

**Abstract**: In the real world, trajectory data is often sparse and incomplete due to low collection frequencies or limited device coverage. Trajectory recovery aims to recover these missing trajectory points, making the trajectories denser and more complete. However, this task faces two key challenges: 1) The excessive sparsity of individual trajectories makes it difficult to effectively leverage historical information for recovery; 2) Sparse trajectories make it harder to capture complex individual mobility preferences. To address these challenges, we propose a novel method called DiffMove. Firstly, we harness crowd wisdom for trajectory recovery. Specifically, we construct a group tendency graph using the collective trajectories of all users and then integrate the group mobility trends into the location representations via graph embedding. This solves the challenge of sparse trajectories being unable to rely on individual historical trajectories for recovery. Secondly, we capture individual mobility preferences from both historical and current perspectives. Finally, we integrate group mobility tendencies and individual preferences into the spatiotemporal distribution of the trajectory to recover high-quality trajectories. Extensive experiments on two real-world datasets demonstrate that DiffMove outperforms existing state-of-the-art methods. Further analysis validates the robustness of our method. 

**Abstract (ZH)**: 基于众包智慧的轨迹恢复方法DiffMove 

---
# A Study on Neuro-Symbolic Artificial Intelligence: Healthcare Perspectives 

**Title (ZH)**: 神经符号人工智能的研究：医疗健康视角 

**Authors**: Delower Hossain, Jake Y Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18213)  

**Abstract**: Over the last few decades, Artificial Intelligence (AI) scientists have been conducting investigations to attain human-level performance by a machine in accomplishing a cognitive task. Within machine learning, the ultimate aspiration is to attain Artificial General Intelligence (AGI) through a machine. This pursuit has led to the exploration of two distinct AI paradigms. Symbolic AI, also known as classical or GOFAI (Good Old-Fashioned AI) and Connectionist (Sub-symbolic) AI, represented by Neural Systems, are two mutually exclusive paradigms. Symbolic AI excels in reasoning, explainability, and knowledge representation but faces challenges in processing complex real-world data with noise. Conversely, deep learning (Black-Box systems) research breakthroughs in neural networks are notable, yet they lack reasoning and interpretability. Neuro-symbolic AI (NeSy), an emerging area of AI research, attempts to bridge this gap by integrating logical reasoning into neural networks, enabling them to learn and reason with symbolic representations. While a long path, this strategy has made significant progress towards achieving common sense reasoning by systems. This article conducts an extensive review of over 977 studies from prominent scientific databases (DBLP, ACL, IEEExplore, Scopus, PubMed, ICML, ICLR), thoroughly examining the multifaceted capabilities of Neuro-Symbolic AI, with a particular focus on its healthcare applications, particularly in drug discovery, and Protein engineering research. The survey addresses vital themes, including reasoning, explainability, integration strategies, 41 healthcare-related use cases, benchmarking, datasets, current approach limitations from both healthcare and broader perspectives, and proposed novel approaches for future experiments. 

**Abstract (ZH)**: 近年来，人工智能科学家们一直在进行研究，旨在通过机器在完成认知任务时达到人类水平的表现。在机器学习领域，最终目标是通过机器达到人工通用智能（AGI）。这场追求促使人们探索了两种不同的AI范式。符号AI，也称为经典AI或GOFAI（Good Old-Fashioned AI），和以神经网络为代表的连接主义（次符号）AI是两种互斥的范式。符号AI在推理、可解释性和知识表示方面表现出色，但在处理嘈杂的复杂真实世界数据时面临挑战。相反，虽然深度学习（黑盒系统）在神经网络领域的突破令人瞩目，但它们缺乏推理和可解释性。神经符号AI（NeSy）作为一个新兴的AI研究领域，试图通过将逻辑推理整合到神经网络中，使它们能够学习和使用符号表示进行推理。尽管还有一段很长的路要走，但这种策略在实现系统常识推理方面取得了显著进展。本文对手风琴数据库（DBLP）、ACL、IEEExplore、Scopus、PubMed、ICML、ICLR等主要科学数据库中的超过977篇研究进行了广泛的综述，详细探讨了神经符号AI的多方面能力，特别是其在药物发现和蛋白质工程研究中的医疗应用。综述涵盖了推理、可解释性、整合策略、41个医疗相关用例、基准测试、数据集、从医疗到更广泛视角的当前方法局限性以及未来实验的新型方法等重要主题。 

---
# Exploring Energy Landscapes for Minimal Counterfactual Explanations: Applications in Cybersecurity and Beyond 

**Title (ZH)**: 探索能量景观以寻找最小化反事实解释：在网络安全及其他领域的应用 

**Authors**: Spyridon Evangelatos, Eleni Veroni, Vasilis Efthymiou, Christos Nikolopoulos, Georgios Th. Papadopoulos, Panagiotis Sarigiannidis  

**Link**: [PDF](https://arxiv.org/pdf/2503.18185)  

**Abstract**: Counterfactual explanations have emerged as a prominent method in Explainable Artificial Intelligence (XAI), providing intuitive and actionable insights into Machine Learning model decisions. In contrast to other traditional feature attribution methods that assess the importance of input variables, counterfactual explanations focus on identifying the minimal changes required to alter a model's prediction, offering a ``what-if'' analysis that is close to human reasoning. In the context of XAI, counterfactuals enhance transparency, trustworthiness and fairness, offering explanations that are not just interpretable but directly applicable in the decision-making processes.
In this paper, we present a novel framework that integrates perturbation theory and statistical mechanics to generate minimal counterfactual explanations in explainable AI. We employ a local Taylor expansion of a Machine Learning model's predictive function and reformulate the counterfactual search as an energy minimization problem over a complex landscape. In sequence, we model the probability of candidate perturbations leveraging the Boltzmann distribution and use simulated annealing for iterative refinement. Our approach systematically identifies the smallest modifications required to change a model's prediction while maintaining plausibility. Experimental results on benchmark datasets for cybersecurity in Internet of Things environments, demonstrate that our method provides actionable, interpretable counterfactuals and offers deeper insights into model sensitivity and decision boundaries in high-dimensional spaces. 

**Abstract (ZH)**: 一种结合扰动理论和统计力学的新型解释型人工智能最小事实推理框架 

---
# Strategic Prompt Pricing for AIGC Services: A User-Centric Approach 

**Title (ZH)**: 面向用户的AIGC服务战略提示定价方法 

**Authors**: Xiang Li, Bing Luo, Jianwei Huang, Yuan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.18168)  

**Abstract**: The rapid growth of AI-generated content (AIGC) services has created an urgent need for effective prompt pricing strategies, yet current approaches overlook users' strategic two-step decision-making process in selecting and utilizing generative AI models. This oversight creates two key technical challenges: quantifying the relationship between user prompt capabilities and generation outcomes, and optimizing platform payoff while accounting for heterogeneous user behaviors. We address these challenges by introducing prompt ambiguity, a theoretical framework that captures users' varying abilities in prompt engineering, and developing an Optimal Prompt Pricing (OPP) algorithm. Our analysis reveals a counterintuitive insight: users with higher prompt ambiguity (i.e., lower capability) exhibit non-monotonic prompt usage patterns, first increasing then decreasing with ambiguity levels, reflecting complex changes in marginal utility. Experimental evaluation using a character-level GPT-like model demonstrates that our OPP algorithm achieves up to 31.72% improvement in platform payoff compared to existing pricing mechanisms, validating the importance of user-centric prompt pricing in AIGC services. 

**Abstract (ZH)**: AI生成内容服务的迅速增长迫切需要有效的提示定价策略，但当前方法忽视了用户在选择和利用生成式AI模型时的战略两步决策过程。这一忽视造成了两个关键技术挑战：量化用户提示能力与生成结果之间的关系，以及在考虑用户行为异质性的情况下优化平台收益。我们通过引入提示含糊性这一理论框架来应对这些挑战，该框架捕捉了用户在提示工程方面的不同能力，并开发了最优提示定价（OPP）算法。我们的分析揭示了一个出人意料的见解：提示含糊性较高（即能力较低）的用户表现出非单调的提示使用模式，含糊性水平先增加后减少，反映了边际效用的复杂变化。使用字符级GPT-like模型的实验评估表明，我们的OPP算法在现有定价机制的基础上，平台收益提高了最高31.72%，验证了用户为中心的提示定价在AI生成内容服务中的重要性。 

---
# OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination 

**Title (ZH)**: OvercookedV2: 重新思考Overcooked中的零样本协作 

**Authors**: Tobias Gessler, Tin Dizdarevic, Ani Calinescu, Benjamin Ellis, Andrei Lupu, Jakob Nicolaus Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2503.17821)  

**Abstract**: AI agents hold the potential to transform everyday life by helping humans achieve their goals. To do this successfully, agents need to be able to coordinate with novel partners without prior interaction, a setting known as zero-shot coordination (ZSC). Overcooked has become one of the most popular benchmarks for evaluating coordination capabilities of AI agents and learning algorithms. In this work, we investigate the origins of ZSC challenges in Overcooked. We introduce a state augmentation mechanism which mixes states that might be encountered when paired with unknown partners into the training distribution, reducing the out-of-distribution challenge associated with ZSC. We show that independently trained agents under this algorithm coordinate successfully in Overcooked. Our results suggest that ZSC failure can largely be attributed to poor state coverage under self-play rather than more sophisticated coordination challenges. The Overcooked environment is therefore not suitable as a ZSC benchmark. To address these shortcomings, we introduce OvercookedV2, a new version of the benchmark, which includes asymmetric information and stochasticity, facilitating the creation of interesting ZSC scenarios. To validate OvercookedV2, we conduct experiments demonstrating that mere exhaustive state coverage is insufficient to coordinate well. Finally, we use OvercookedV2 to build a new range of coordination challenges, including ones that require test time protocol formation, and we demonstrate the need for new coordination algorithms that can adapt online. We hope that OvercookedV2 will help benchmark the next generation of ZSC algorithms and advance collaboration between AI agents and humans. 

**Abstract (ZH)**: AI代理有潜力通过帮助人类实现目标来转变日常生活。为了成功做到这一点，代理需要能够在没有先前交互的情况下与新颖的合作伙伴协调，这种情境被称为零样本协调（ZSC）。Overcooked已成为评估AI代理和学习算法协调能力的最流行基准之一。在本工作中，我们探讨了Overcooked中ZSC挑战的根源。我们提出了一种状态扩充机制，将与未知合作伙伴可能遇到的状态混合到训练分布中，从而减少与ZSC相关的出分布挑战。我们表明，在该算法下独立训练的代理能够在Overcooked中成功协调。我们的结果显示，ZSC失败主要归因于自我对弈下状态覆盖不足，而不是更复杂的协调挑战。因此，Overcooked环境不适合用作ZSC基准。为了弥补这些不足，我们引入了OvercookedV2，这是基准的一个新版本，包括非对称信息和随机性，有助于创建有趣的ZSC情境。为了验证OvercookedV2，我们进行了实验，证明仅仅状态覆盖是不充分的。最后，我们使用OvercookedV2构建了一系列新的协调挑战，包括那些需要在测试时形成协议的任务，并展示了需要新的在线适应的协调算法。我们希望OvercookedV2能够帮助基准测试下一代ZSC算法，并促进AI代理与人类的合作。 

---
# Intelligence Sequencing and the Path-Dependence of Intelligence Evolution: AGI-First vs. DCI-First as Irreversible Attractors 

**Title (ZH)**: 智能排序与智能进化路径依赖性：AGI优先 vs. DCI优先作为不可逆的吸引子 

**Authors**: Andy E. Williams  

**Link**: [PDF](https://arxiv.org/pdf/2503.17688)  

**Abstract**: The trajectory of intelligence evolution is often framed around the emergence of artificial general intelligence (AGI) and its alignment with human values. This paper challenges that framing by introducing the concept of intelligence sequencing: the idea that the order in which AGI and decentralized collective intelligence (DCI) emerge determines the long-term attractor basin of intelligence. Using insights from dynamical systems, evolutionary game theory, and network models, it argues that intelligence follows a path-dependent, irreversible trajectory. Once development enters a centralized (AGI-first) or decentralized (DCI-first) regime, transitions become structurally infeasible due to feedback loops and resource lock-in. Intelligence attractors are modeled in functional state space as the co-navigation of conceptual and adaptive fitness spaces. Early-phase structuring constrains later dynamics, much like renormalization in physics. This has major implications for AI safety: traditional alignment assumes AGI will emerge and must be controlled after the fact, but this paper argues that intelligence sequencing is more foundational. If AGI-first architectures dominate before DCI reaches critical mass, hierarchical monopolization and existential risk become locked in. If DCI-first emerges, intelligence stabilizes around decentralized cooperative equilibrium. The paper further explores whether intelligence structurally biases itself toward an attractor based on its self-modeling method -- externally imposed axioms (favoring AGI) vs. recursive internal visualization (favoring DCI). Finally, it proposes methods to test this theory via simulations, historical lock-in case studies, and intelligence network analysis. The findings suggest that intelligence sequencing is a civilizational tipping point: determining whether the future is shaped by unbounded competition or unbounded cooperation. 

**Abstract (ZH)**: 智能演化的轨迹往往围绕人工通用智能（AGI）的出现及其与人类价值的契合展开。本文通过引入智能序列化的概念对此框架提出了挑战：即AGI和分布式集体智能（DCI）出现的顺序决定了智能的长期吸引子盆地。本文利用动力系统理论、进化博弈论和网络模型的见解，论证了智能遵循一条路径依赖且不可逆的轨迹。一旦发展进入集中的（AGI优先）或去中心化的（DCI优先）模式，由于反馈循环和资源锁定，转变成为结构上不可行。智能吸引子在功能性状态空间中模型化为概念性和适应性fitness空间的协调导航。早期阶段的结构化限制了后来的动态，类似于物理学中的重整化过程。这在人工智能安全方面具有重大意义：传统的对齐假设AGI将会出现并应在事后进行控制，但本文认为智能序列化更根本。如果在DCI达到临界规模之前AGI架构占据主导地位，层级垄断和 existential 风险将被锁定。如果DCI优先出现，智能将稳定在去中心化的合作均衡周围。本文进一步探讨智能根据其自模型化方法是否结构上偏向于特定的吸引子——外在强加的公理（倾向AGI）与递归内部可视化（倾向DCI）。最后，本文提出了通过模拟、历史锁定案例研究和智能网络分析来测试这一理论的方法。研究发现表明，智能序列化是文明转折点：决定未来是被无约束竞争还是无约束合作塑造的关键。 

---
# Exploring the Integration of Key-Value Attention Into Pure and Hybrid Transformers for Semantic Segmentation 

**Title (ZH)**: 探索将键值注意机制集成到纯Transformer和混合Transformer中以进行语义分割 

**Authors**: DeShin Hwa, Tobias Holmes, Klaus Drechsler  

**Link**: [PDF](https://arxiv.org/pdf/2503.18862)  

**Abstract**: While CNNs were long considered state of the art for image processing, the introduction of Transformer architectures has challenged this position. While achieving excellent results in image classification and segmentation, Transformers remain inherently reliant on large training datasets and remain computationally expensive. A newly introduced Transformer derivative named KV Transformer shows promising results in synthetic, NLP, and image classification tasks, while reducing complexity and memory usage. This is especially conducive to use cases where local inference is required, such as medical screening applications. We endeavoured to further evaluate the merit of KV Transformers on semantic segmentation tasks, specifically in the domain of medical imaging. By directly comparing traditional and KV variants of the same base architectures, we provide further insight into the practical tradeoffs of reduced model complexity. We observe a notable reduction in parameter count and multiply accumulate operations, while achieving similar performance from most of the KV variant models when directly compared to their QKV implementation. 

**Abstract (ZH)**: 虽然CNN曾长期被认为是图像处理的前沿技术，但Transformer架构的 introduction 已对其地位提出了挑战。尽管在图像分类和分割任务中取得了卓越成果，Transformer依然依赖大规模训练数据集，并且计算成本较高。一种新引入的Transformer变体——KV Transformer，在合成数据、自然语言处理和图像分类任务中表现出令人鼓舞的结果，同时降低了复杂度和内存使用量。这特别适合需要局部推理的应用场景，如医疗筛查。我们对KV Transformer在语义分割任务中的表现进行了进一步评估，特别是在医疗成像领域。通过直接比较传统架构和KV架构的变体，我们进一步探讨了模型复杂度降低的实际权衡。我们观察到，在直接与QKV实现形式对比时，大多数KV变体模型的参数数量和乘积累加操作有了显著减少，同时仍能实现相当相当的性能。 

---
# Three Kinds of AI Ethics 

**Title (ZH)**: 三种人工智能伦理类型 

**Authors**: Emanuele Ratti  

**Link**: [PDF](https://arxiv.org/pdf/2503.18842)  

**Abstract**: There is an overwhelmingly abundance of works in AI Ethics. This growth is chaotic because of how sudden it is, its volume, and its multidisciplinary nature. This makes difficult to keep track of debates, and to systematically characterize goals, research questions, methods, and expertise required by AI ethicists. In this article, I show that the relation between AI and ethics can be characterized in at least three ways, which correspond to three well-represented kinds of AI ethics: ethics and AI; ethics in AI; ethics of AI. I elucidate the features of these three kinds of AI Ethics, characterize their research questions, and identify the kind of expertise that each kind needs. I also show how certain criticisms to AI ethics are misplaced, as being done from the point of view of one kind of AI ethics, to another kind with different goals. All in all, this work sheds light on the nature of AI ethics, and set the grounds for more informed discussions about scope, methods, and trainings of AI ethicists. 

**Abstract (ZH)**: 人工智能伦理学中存在着大量的研究工作。由于其突然性、体量以及跨学科的特性，这种增长显得杂乱无章，使得跟踪辩论、系统化地描述人工智能伦理学家的目标、研究问题、方法以及所需的专业知识变得困难。在本文中，我展示了一种方法，通过该方法可以将人工智能与伦理的关系至少从三个方面加以描述，这对应于三种高度代表性的AI伦理学类型：人工智能与伦理；伦理与人工智能；人工智能的伦理。我阐述了这三种类型AI伦理学的特点，描述了它们的研究问题，并指出了每种类型所需的专业知识。我还表明，某些对人工智能伦理学的批评可能是基于一种类型的AI伦理学观点，而忽略了具有不同目标的另一种类型，因此存在偏差。总体而言，这项工作揭示了人工智能伦理学的本质，并为更明晰地讨论范围、方法和人工智能伦理学家的培训奠定了基础。 

---
# Interpretable and Fair Mechanisms for Abstaining Classifiers 

**Title (ZH)**: 可解释且公平的弃权分类机制 

**Authors**: Daphne Lenders, Andrea Pugnana, Roberto Pellungrini, Toon Calders, Dino Pedreschi, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.18826)  

**Abstract**: Abstaining classifiers have the option to refrain from providing a prediction for instances that are difficult to classify. The abstention mechanism is designed to trade off the classifier's performance on the accepted data while ensuring a minimum number of predictions. In this setting, often fairness concerns arise when the abstention mechanism solely reduces errors for the majority groups of the data, resulting in increased performance differences across demographic groups. While there exist a bunch of methods that aim to reduce discrimination when abstaining, there is no mechanism that can do so in an explainable way. In this paper, we fill this gap by introducing Interpretable and Fair Abstaining Classifier IFAC, an algorithm that can reject predictions both based on their uncertainty and their unfairness. By rejecting possibly unfair predictions, our method reduces error and positive decision rate differences across demographic groups of the non-rejected data. Since the unfairness-based rejections are based on an interpretable-by-design method, i.e., rule-based fairness checks and situation testing, we create a transparent process that can empower human decision-makers to review the unfair predictions and make more just decisions for them. This explainable aspect is especially important in light of recent AI regulations, mandating that any high-risk decision task should be overseen by human experts to reduce discrimination risks. 

**Abstract (ZH)**: 可解释且公平的弃权分类器：IFAC 

---
# Construction Identification and Disambiguation Using BERT: A Case Study of NPN 

**Title (ZH)**: 基于BERT的NPN的构造识别与消歧：一个案例研究 

**Authors**: Wesley Scivetti, Nathan Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2503.18751)  

**Abstract**: Construction Grammar hypothesizes that knowledge of a language consists chiefly of knowledge of form-meaning pairs (''constructions'') that include vocabulary, general grammar rules, and even idiosyncratic patterns. Recent work has shown that transformer language models represent at least some constructional patterns, including ones where the construction is rare overall. In this work, we probe BERT's representation of the form and meaning of a minor construction of English, the NPN (noun-preposition-noun) construction -- exhibited in such expressions as face to face and day to day -- which is known to be polysemous. We construct a benchmark dataset of semantically annotated corpus instances (including distractors that superficially resemble the construction). With this dataset, we train and evaluate probing classifiers. They achieve decent discrimination of the construction from distractors, as well as sense disambiguation among true instances of the construction, revealing that BERT embeddings carry indications of the construction's semantics. Moreover, artificially permuting the word order of true construction instances causes them to be rejected, indicating sensitivity to matters of form. We conclude that BERT does latently encode at least some knowledge of the NPN construction going beyond a surface syntactic pattern and lexical cues. 

**Abstract (ZH)**: 构造语法假定语言知识主要由形式意义对（“构造”）组成，这些构造包括词汇、一般的语法规则，甚至包括特有的模式。近期研究表明，转换器语言模型至少表示了一些构造模式，即使这些构造在整体上较为罕见。在此项工作中，我们探究了BERT对英语小量使用的NPN（名词-介词-名词）构造的形式和意义的表示——如“面对面”、“日复一日”等表达式中所体现的——该构造已知具有多义性。我们构建了一个语义标注的基准数据集（包括外观上类似但语义不同的干扰项），基于此数据集训练并评估了探针分类器。分类器能够较好地区分构造与其干扰项，并能够区分构造的真实实例中的意义歧义，揭示BERT嵌入能够携带构造语义的线索。此外，人为改变构造实例中的词序会使其被拒绝，表明其对形式问题具有敏感性。我们得出结论，BERT在表面句法模式和词汇线索之外，隐含地编码了至少部分关于NPN构造的知识。 

---
# Energy-Efficient Dynamic Training and Inference for GNN-Based Network Modeling 

**Title (ZH)**: 基于GNN的网络建模的能效动态训练与推理 

**Authors**: Chetna Singhal, Yassine Hadjadj-Aoul  

**Link**: [PDF](https://arxiv.org/pdf/2503.18706)  

**Abstract**: Efficient network modeling is essential for resource optimization and network planning in next-generation large-scale complex networks. Traditional approaches, such as queuing theory-based modeling and packet-based simulators, can be inefficient due to the assumption made and the computational expense, respectively. To address these challenges, we propose an innovative energy-efficient dynamic orchestration of Graph Neural Networks (GNN) based model training and inference framework for context-aware network modeling and predictions. We have developed a low-complexity solution framework, QAG, that is a Quantum approximation optimization (QAO) algorithm for Adaptive orchestration of GNN-based network modeling. We leverage the tripartite graph model to represent a multi-application system with many compute nodes. Thereafter, we apply the constrained graph-cutting using QAO to find the feasible energy-efficient configurations of the GNN-based model and deploying them on the available compute nodes to meet the network modeling application requirements. The proposed QAG scheme closely matches the optimum and offers atleast a 50% energy saving while meeting the application requirements with 60% lower churn-rate. 

**Abstract (ZH)**: 高效的网络建模对于下一代大规模复杂网络的资源优化和网络规划至关重要。传统的基于排队理论的建模方法和基于包的仿真器由于假设条件和计算成本的原因可能存在低效问题。为解决这些问题，我们提出了一种基于图神经网络（GNN）动态 orchestration 的创新性能量高效模型训练与推理框架，用于上下文感知的网络建模与预测。我们开发了一个低复杂度的解决方案框架，QAG，这是一种量子近似优化（QAO）算法，用于自适应 orchestration 的 GNN 基础网络建模。利用 tripartite 图模型表示包含多个计算节点的多应用系统。然后，我们通过 QAO 进行约束图划分，寻找 GNN 基础模型的能量高效配置，并将其部署到可用的计算节点上，以满足网络建模应用需求。所提出的 QAG 方案接近最优方案，至少可以节省50%的能量，在满足应用需求的同时将更换率降低60%。 

---
# Towards Human-Understandable Multi-Dimensional Concept Discovery 

**Title (ZH)**: 面向人类可理解的多维度概念发现 

**Authors**: Arne Grobrügge, Niklas Kühl, Gerhard Satzger, Philipp Spitzer  

**Link**: [PDF](https://arxiv.org/pdf/2503.18629)  

**Abstract**: Concept-based eXplainable AI (C-XAI) aims to overcome the limitations of traditional saliency maps by converting pixels into human-understandable concepts that are consistent across an entire dataset. A crucial aspect of C-XAI is completeness, which measures how well a set of concepts explains a model's decisions. Among C-XAI methods, Multi-Dimensional Concept Discovery (MCD) effectively improves completeness by breaking down the CNN latent space into distinct and interpretable concept subspaces. However, MCD's explanations can be difficult for humans to understand, raising concerns about their practical utility. To address this, we propose Human-Understandable Multi-dimensional Concept Discovery (HU-MCD). HU-MCD uses the Segment Anything Model for concept identification and implements a CNN-specific input masking technique to reduce noise introduced by traditional masking methods. These changes to MCD, paired with the completeness relation, enable HU-MCD to enhance concept understandability while maintaining explanation faithfulness. Our experiments, including human subject studies, show that HU-MCD provides more precise and reliable explanations than existing C-XAI methods. The code is available at this https URL. 

**Abstract (ZH)**: 基于概念的可解释人工智能（C-XAI）旨在通过将像素转换为整个数据集中一致的人类可理解概念来克服传统显著性图的局限性。C-XAI的关键方面是完备性，它衡量一组概念解释模型决策的能力。在C-XAI方法中，多维概念发现（MCD）通过将CNN潜在空间分解为独立且可解释的概念子空间，有效地提高了完备性。然而，MCD的解释可能难以为人理解，对其实际应用性提出质疑。为了解决这一问题，我们提出了人类可理解的多维概念发现（HU-MCD）。HU-MCD利用Segment Anything模型进行概念识别，并采用特定于CNN的输入蒙版技术以减少传统蒙版方法引入的噪声。这些对MCD的改进，加上完备性关系的结合，使HU-MCD能够在提高概念可理解性的同时保持解释的可信性。我们的实验，包括人类受控实验，显示HU-MCD提供了比现有C-XAI方法更精确和可靠的解释。代码可在以下链接获取。 

---
# Reinforcement Learning in Switching Non-Stationary Markov Decision Processes: Algorithms and Convergence Analysis 

**Title (ZH)**: 切换非稳态马尔可夫决策过程的强化学习：算法与收敛性分析 

**Authors**: Mohsen Amiri, Sindri Magnússon  

**Link**: [PDF](https://arxiv.org/pdf/2503.18607)  

**Abstract**: Reinforcement learning in non-stationary environments is challenging due to abrupt and unpredictable changes in dynamics, often causing traditional algorithms to fail to converge. However, in many real-world cases, non-stationarity has some structure that can be exploited to develop algorithms and facilitate theoretical analysis. We introduce one such structure, Switching Non-Stationary Markov Decision Processes (SNS-MDP), where environments switch over time based on an underlying Markov chain. Under a fixed policy, the value function of an SNS-MDP admits a closed-form solution determined by the Markov chain's statistical properties, and despite the inherent non-stationarity, Temporal Difference (TD) learning methods still converge to the correct value function. Furthermore, policy improvement can be performed, and it is shown that policy iteration converges to the optimal policy. Moreover, since Q-learning converges to the optimal Q-function, it likewise yields the corresponding optimal policy. To illustrate the practical advantages of SNS-MDPs, we present an example in communication networks where channel noise follows a Markovian pattern, demonstrating how this framework can effectively guide decision-making in complex, time-varying contexts. 

**Abstract (ZH)**: 在非平稳环境中的强化学习因动力学的突然和不可预测的变化而具有挑战性，通常导致传统算法无法收敛。然而，在许多实际情况下，非平稳性具有某种可利用的结构，可以开发算法并促进理论分析。我们介绍了一种这样的结构——切换非平稳马尔可夫决策过程（SNS-MDP），其中环境根据潜在的马尔可夫链在时间上切换。在固定策略下，SNS-MDP的价值函数可以通过马尔可夫链的统计属性获得闭式解，尽管存在固有的非平稳性，时差学习方法仍能收敛到正确的价值函数。此外，可以通过执行策略改进，证明策略迭代能收敛到最优策略。由于Q学习收敛于最优Q函数，它同样会产生相应的最优策略。为了说明SNS-MDP的实际优势，我们给出一个通信网络中的示例，其中信道噪声遵循马尔可夫模式，展示该框架如何有效指导复杂、时变环境中的决策。 

---
# Identifying and Characterising Higher Order Interactions in Mobility Networks Using Hypergraphs 

**Title (ZH)**: 使用超图识别和表征移动网络中的高阶交互 

**Authors**: Prathyush Sambaturu, Bernardo Gutierrez, Moritz U.G. Kraemer  

**Link**: [PDF](https://arxiv.org/pdf/2503.18572)  

**Abstract**: Understanding human mobility is essential for applications ranging from urban planning to public health. Traditional mobility models such as flow networks and colocation matrices capture only pairwise interactions between discrete locations, overlooking higher-order relationships among locations (i.e., mobility flow among two or more locations). To address this, we propose co-visitation hypergraphs, a model that leverages temporal observation windows to extract group interactions between locations from individual mobility trajectory data. Using frequent pattern mining, our approach constructs hypergraphs that capture dynamic mobility behaviors across different spatial and temporal scales. We validate our method on a publicly available mobility dataset and demonstrate its effectiveness in analyzing city-scale mobility patterns, detecting shifts during external disruptions such as extreme weather events, and examining how a location's connectivity (degree) relates to the number of points of interest (POIs) within it. Our results demonstrate that our hypergraph-based mobility analysis framework is a valuable tool with potential applications in diverse fields such as public health, disaster resilience, and urban planning. 

**Abstract (ZH)**: 理解人类移动性对于从城市规划到公共卫生等应用至关重要。传统的移动性模型如流网络和共存矩阵仅捕捉离散地点之间的成对交互，忽视了地点间的高阶关系（即两个或多个地点之间的移动流）。为解决这一问题，我们提出了共访问超图模型，该模型利用时间观察窗口从个体移动轨迹数据中提取地点之间的群组交互。通过频繁模式挖掘，我们的方法构建了能够捕捉不同空间和时间尺度下动态移动行为的超图。我们在一个公开的移动性数据集上验证了该方法，并展示了其在分析城市规模的移动模式、检测外部干扰（如极端天气事件）期间的变化以及研究地点连通性（度）与其内部兴趣点（POI）数量的关系方面的有效性。我们的结果表明，基于超图的移动性分析框架是一个有价值的工具，具有在公共卫生、灾害抵御和城市规划等领域应用的潜力。 

---
# Anchor-based oversampling for imbalanced tabular data via contrastive and adversarial learning 

**Title (ZH)**: 基于锚点的过采样方法：通过对比学习和对抗学习处理不平衡表格数据 

**Authors**: Hadi Mohammadi, Ehsan Nazerfard, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2503.18569)  

**Abstract**: Imbalanced data represent a distribution with more frequencies of one class (majority) than the other (minority). This phenomenon occurs across various domains, such as security, medical care and human activity. In imbalanced learning, classification algorithms are typically inclined to classify the majority class accurately, resulting in artificially high accuracy rates. As a result, many minority samples are mistakenly labelled as majority-class instances, resulting in a bias that benefits the majority class. This study presents a framework based on boundary anchor samples to tackle the imbalance learning challenge. First, we select and use anchor samples to train a multilayer perceptron (MLP) classifier, which acts as a prior knowledge model and aids the adversarial and contrastive learning procedures. Then, we designed a novel deep generative model called Anchor Stabilized Conditional Generative Adversarial Network or Anch-SCGAN in short. Anch-SCGAN is supported with two generators for the minority and majority classes and a discriminator incorporating additional class-specific information from the pre-trained feature extractor MLP. In addition, we facilitate the generator's training procedure in two ways. First, we define a new generator loss function based on reprocessed anchor samples and contrastive learning. Second, we apply a scoring strategy to stabilize the adversarial training part in generators. We train Anch-SCGAN and further finetune it with anchor samples to improve the precision of the generated samples. Our experiments on 16 real-world imbalanced datasets illustrate that Anch-SCGAN outperforms the renowned methods in imbalanced learning. 

**Abstract (ZH)**: 不平衡数据表示一种分布，其中一类（多数类）的频率远高于另一类（少数类）。这种现象在安全、医疗护理和人类活动等领域普遍存在。在不平衡学习中，分类算法通常倾向于准确分类多数类，导致人为提高了准确率。结果，许多少数类样本被错误地标记为多数类实例，造成了有利于多数类的偏差。本研究提出了一种基于边界锚样本的框架，以应对不平衡学习的挑战。首先，我们选择并使用锚样本训练一个多层感知机（MLP）分类器，该分类器作为先验知识模型，辅助对抗学习和对比学习过程。然后，我们设计了一个名为锚稳定条件生成器对抗网络或简称Anch-SCGAN的新颖深层生成模型。Anch-SCGAN配备有为少数类和多数类提供支持的两个生成器，并整合了预训练特征提取器MLP提供的额外类特定信息的判别器。此外，我们通过两种方式促进了生成器的训练过程。首先，我们定义了一个基于重新处理锚样本和对比学习的新生成器损失函数。其次，我们应用了一种评分策略来稳定生成器中的对抗训练部分。我们通过锚样本进一步微调Anch-SCGAN，并提高生成样本的精度。我们在16个实际的不平衡数据集上的实验显示，Anch-SCGAN优于现有的知名方法。 

---
# Discriminative protein sequence modelling with Latent Space Diffusion 

**Title (ZH)**: 基于潜在空间扩散的辨别性蛋白质序列建模 

**Authors**: Eoin Quinn, Ghassene Jebali, Maxime Seince, Oliver Bent  

**Link**: [PDF](https://arxiv.org/pdf/2503.18551)  

**Abstract**: We explore a framework for protein sequence representation learning that decomposes the task between manifold learning and distributional modelling. Specifically we present a Latent Space Diffusion architecture which combines a protein sequence autoencoder with a denoising diffusion model operating on its latent space. We obtain a one-parameter family of learned representations from the diffusion model, along with the autoencoder's latent representation. We propose and evaluate two autoencoder architectures: a homogeneous model forcing amino acids of the same type to be identically distributed in the latent space, and an inhomogeneous model employing a noise-based variant of masking. As a baseline we take a latent space learned by masked language modelling, and evaluate discriminative capability on a range of protein property prediction tasks. Our finding is twofold: the diffusion models trained on both our proposed variants display higher discriminative power than the one trained on the masked language model baseline, none of the diffusion representations achieve the performance of the masked language model embeddings themselves. 

**Abstract (ZH)**: 我们探索了一种将蛋白质序列表示学习任务分解到流形学习和分布建模之间的框架。具体而言，我们提出了一种潜空间扩散架构，该架构结合了蛋白质序列自编码器和在其潜空间上操作的去噪扩散模型。我们从扩散模型中获得了具有一个参数的可学习表示族，以及自编码器的潜空间表示。我们提出了两种自编码器架构：一种同质模型，强制相同类型的氨基酸在潜空间中服从同一分布；以及一种异质模型，采用基于噪声的掩码变体。作为基准，我们采用了由掩码语言模型学习的潜空间，并在一系列蛋白质性质预测任务上评估其区分能力。我们的发现有两个方面：在我们提出的两种变体上训练的扩散模型表现出比基于掩码语言模型基准更高的区分能力，但没有一种扩散表示能够达到掩码语言模型嵌入本身的表现。 

---
# RLCAD: Reinforcement Learning Training Gym for Revolution Involved CAD Command Sequence Generation 

**Title (ZH)**: RLCAD: 用于涉及革命的CAD命令序列生成的强化学习训练 Gym 

**Authors**: Xiaolong Yin, Xingyu Lu, Jiahang Shen, Jingzhe Ni, Hailong Li, Ruofeng Tong, Min Tang, Peng Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.18549)  

**Abstract**: A CAD command sequence is a typical parametric design paradigm in 3D CAD systems where a model is constructed by overlaying 2D sketches with operations such as extrusion, revolution, and Boolean operations. Although there is growing academic interest in the automatic generation of command sequences, existing methods and datasets only support operations such as 2D sketching, extrusion,and Boolean operations. This limitation makes it challenging to represent more complex geometries. In this paper, we present a reinforcement learning (RL) training environment (gym) built on a CAD geometric engine. Given an input boundary representation (B-Rep) geometry, the policy network in the RL algorithm generates an action. This action, along with previously generated actions, is processed within the gym to produce the corresponding CAD geometry, which is then fed back into the policy network. The rewards, determined by the difference between the generated and target geometries within the gym, are used to update the RL network. Our method supports operations beyond sketches, Boolean, and extrusion, including revolution operations. With this training gym, we achieve state-of-the-art (SOTA) quality in generating command sequences from B-Rep geometries. In addition, our method can significantly improve the efficiency of command sequence generation by a factor of 39X compared with the previous training gym. 

**Abstract (ZH)**: 一种CAD命令序列的强化学习训练环境：超越传统操作的复杂几何体生成 

---
# An Identity and Interaction Based Network Forensic Analysis 

**Title (ZH)**: 基于身份与交互的网络取证分析 

**Authors**: Nathan Clarke, Gaseb Alotibi, Dany Joy, Fudong Li, Steven Furnell, Ali Alshumrani, Hussan Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2503.18542)  

**Abstract**: In todays landscape of increasing electronic crime, network forensics plays a pivotal role in digital investigations. It aids in understanding which systems to analyse and as a supplement to support evidence found through more traditional computer based investigations. However, the nature and functionality of the existing Network Forensic Analysis Tools (NFATs) fall short compared to File System Forensic Analysis Tools (FS FATs) in providing usable data. The analysis tends to focus upon IP addresses, which are not synonymous with user identities, a point of significant interest to investigators. This paper presents several experiments designed to create a novel NFAT approach that can identify users and understand how they are using network based applications whilst the traffic remains encrypted. The experiments build upon the prior art and investigate how effective this approach is in classifying users and their actions. Utilising an in-house dataset composed of 50 million packers, the experiments are formed of three incremental developments that assist in improving performance. Building upon the successful experiments, a proposed NFAT interface is presented to illustrate the ease at which investigators would be able to ask relevant questions of user interactions. The experiments profiled across 27 users, has yielded an average 93.3% True Positive Identification Rate (TPIR), with 41% of users experiencing 100% TPIR. Skype, Wikipedia and Hotmail services achieved a notably high level of recognition performance. The study has developed and evaluated an approach to analyse encrypted network traffic more effectively through the modelling of network traffic and to subsequently visualise these interactions through a novel network forensic analysis tool. 

**Abstract (ZH)**: 现今电子犯罪日益增多的背景下，网络取证在数字调查中扮演着重要角色。它有助于明确需要分析的系统，并作为补充，支持通过传统计算机调查发现的证据。然而，现有的网络取证分析工具（NFATs）在提供可用数据方面不如文件系统取证分析工具（FS FATs）有效。分析倾向于关注IP地址，而这些IP地址并不等同于用户身份，这是调查人员特别关注的点。本文提出了若干实验，旨在创建一种新型的NFAT方法，能够在流量仍处于加密状态时识别用户并理解他们如何使用基于网络的应用程序。这些实验基于先前的研究，并探讨了该方法在分类用户及其行为方面的有效性。利用包含5000万个数据包的内部数据集，实验分为三个逐步发展的阶段，以提高性能。基于成功的实验结果，提出了一个建议的NFAT界面，展示了调查人员如何轻松提出与用户交互相关的问题。在27名用户的研究中，平均实现了93.3%的真实阳性识别率（TPIR），其中41%的用户实现了100%的TPIR。Skype、Wikipedia和Hotmail服务实现了显著高的识别性能。本研究开发并评估了一种通过建模网络流量并随后通过新型网络取证分析工具可视化这些交互，来更有效地分析加密网络流量的方法。 

---
# UniPCGC: Towards Practical Point Cloud Geometry Compression via an Efficient Unified Approach 

**Title (ZH)**: UniPCGC：通过一种高效的统一方法面向实用的点云几何压缩 

**Authors**: Kangli Wang, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.18541)  

**Abstract**: Learning-based point cloud compression methods have made significant progress in terms of performance. However, these methods still encounter challenges including high complexity, limited compression modes, and a lack of support for variable rate, which restrict the practical application of these methods. In order to promote the development of practical point cloud compression, we propose an efficient unified point cloud geometry compression framework, dubbed as UniPCGC. It is a lightweight framework that supports lossy compression, lossless compression, variable rate and variable complexity. First, we introduce the Uneven 8-Stage Lossless Coder (UELC) in the lossless mode, which allocates more computational complexity to groups with higher coding difficulty, and merges groups with lower coding difficulty. Second, Variable Rate and Complexity Module (VRCM) is achieved in the lossy mode through joint adoption of a rate modulation module and dynamic sparse convolution. Finally, through the dynamic combination of UELC and VRCM, we achieve lossy compression, lossless compression, variable rate and complexity within a unified framework. Compared to the previous state-of-the-art method, our method achieves a compression ratio (CR) gain of 8.1\% on lossless compression, and a Bjontegaard Delta Rate (BD-Rate) gain of 14.02\% on lossy compression, while also supporting variable rate and variable complexity. 

**Abstract (ZH)**: 基于学习的点云压缩方法在性能上取得了显著进步，但仍面临高复杂度、压缩模式有限和不支持可变比特率等挑战，限制了这些方法的实用应用。为促进实用点云压缩的发展，我们提出了一种高效的统一点云几何压缩框架，命名为UniPCGC。该框架支持有损压缩、无损压缩、可变比特率和可变复杂度。首先，在无损模式下引入了非均匀8级无损编码器(UELC)，将更多的计算复杂性分配给编码难度较高的组，并合并编码难度较低的组。其次，无损模式和有损模式通过联合采用速率调制模块和动态稀疏卷积实现了可变比特率和复杂度模块(VRCM)。最后，通过UELC和VRCM的动态组合，我们在统一框架中实现了有损压缩、无损压缩、可变比特率和可变复杂度。与先前的最先进方法相比，在无损压缩中我们的方法获得了8.1%的压缩比(CR)增益，在有损压缩中获得了14.02%的Bjontegaard Delta Rate(BD-Rate)增益，同时支持可变比特率和可变复杂度。 

---
# Natural Language Processing for Electronic Health Records in Scandinavian Languages: Norwegian, Swedish, and Danish 

**Title (ZH)**: Scandinavian语言的电子健康记录自然语言处理：挪威语、瑞典语和丹麦语 

**Authors**: Ashenafi Zebene Woldaregay, Jørgen Aarmo Lund, Phuong Dinh Ngo, Mariyam Tayefi, Joel Burman, Stine Hansen, Martin Hylleholt Sillesen, Hercules Dalianis, Robert Jenssen, Lindsetmo Rolf Ole, Karl Øyvind Mikalsen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18539)  

**Abstract**: Background: Clinical natural language processing (NLP) refers to the use of computational methods for extracting, processing, and analyzing unstructured clinical text data, and holds a huge potential to transform healthcare in various clinical tasks. Objective: The study aims to perform a systematic review to comprehensively assess and analyze the state-of-the-art NLP methods for the mainland Scandinavian clinical text. Method: A literature search was conducted in various online databases including PubMed, ScienceDirect, Google Scholar, ACM digital library, and IEEE Xplore between December 2022 and February 2024. Further, relevant references to the included articles were also used to solidify our search. The final pool includes articles that conducted clinical NLP in the mainland Scandinavian languages and were published in English between 2010 and 2024. Results: Out of the 113 articles, 18% (n=21) focus on Norwegian clinical text, 64% (n=72) on Swedish, 10% (n=11) on Danish, and 8% (n=9) focus on more than one language. Generally, the review identified positive developments across the region despite some observable gaps and disparities between the languages. There are substantial disparities in the level of adoption of transformer-based models. In essential tasks such as de-identification, there is significantly less research activity focusing on Norwegian and Danish compared to Swedish text. Further, the review identified a low level of sharing resources such as data, experimentation code, pre-trained models, and rate of adaptation and transfer learning in the region. Conclusion: The review presented a comprehensive assessment of the state-of-the-art Clinical NLP for electronic health records (EHR) text in mainland Scandinavian languages and, highlighted the potential barriers and challenges that hinder the rapid advancement of the field in the region. 

**Abstract (ZH)**: 背景：临床自然语言处理（NLP）指的是通过计算方法提取、处理和分析未结构化临床文本数据，并在各种临床任务中具有巨大的潜力，以变革医疗保健。目标：本研究旨在进行系统评价，全面评估和分析适用于中国大陆斯堪的纳维亚临床文本的最新NLP方法。方法：于2022年12月至2024年2月期间，在PubMed、ScienceDirect、Google Scholar、ACM数字图书馆和IEEE Xplore等多个在线数据库中进行文献检索，并利用相关参考文献以加强搜索。最终池包括2010年至2024年间以英语发表的适用于中国大陆斯堪的纳维亚语言的临床NLP文章。结果：在113篇文章中，21%（n=21）专注于挪威临床文本，72%（n=72）专注于瑞典，11%（n=11）专注于丹麦，9%（n=9）同时涉及多种语言。总体而言，地区内发现了一些积极的发展趋势，尽管不同语言之间存在可观察到的差距。在采用基于变换器的模型方面，存在显著的差异。在去识别等关键任务中，对挪威和丹麦文本的研究活动显著少于瑞典文本。此外，地区内分享数据、实验代码、预训练模型以及适应和迁移学习的速度较低。结论：本研究全面评估了中国大陆斯堪的纳维亚语言电子健康记录（EHR）文本的最新临床NLP状况，并强调了阻碍该地区领域快速发展的潜在障碍和挑战。 

---
# Statistically Testing Training Data for Unwanted Error Patterns using Rule-Oriented Regression 

**Title (ZH)**: 基于规则定向回归的训练数据统计测试以检测不需要的错误模式 

**Authors**: Stefan Rass, Martin Dallinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.18497)  

**Abstract**: Artificial intelligence models trained from data can only be as good as the underlying data is. Biases in training data propagating through to the output of a machine learning model are a well-documented and well-understood phenomenon, but the machinery to prevent these undesired effects is much less developed. Efforts to ensure data is clean during collection, such as using bias-aware sampling, are most effective when the entity controlling data collection also trains the AI. In cases where the data is already available, how do we find out if the data was already manipulated, i.e., ``poisoned'', so that an undesired behavior would be trained into a machine learning model? This is a challenge fundamentally different to (just) improving approximation accuracy or efficiency, and we provide a method to test training data for flaws, to establish a trustworthy ground-truth for a subsequent training of machine learning models (of any kind). Unlike the well-studied problem of approximating data using fuzzy rules that are generated from the data, our method hinges on a prior definition of rules to happen before seeing the data to be tested. Therefore, the proposed method can also discover hidden error patterns, which may also have substantial influence. Our approach extends the abilities of conventional statistical testing by letting the ``test-condition'' be any Boolean condition to describe a pattern in the data, whose presence we wish to determine. The method puts fuzzy inference into a regression model, to get the best of the two: explainability from fuzzy logic with statistical properties and diagnostics from the regression, and finally also being applicable to ``small data'', hence not requiring large datasets as deep learning methods do. We provide an open source implementation for demonstration and experiments. 

**Abstract (ZH)**: 人工训练的智能模型的质量取决于底层数据的质量。训练数据中的偏差通过机器学习模型输出传递是一种已文档化且被广泛理解的现象，但预防这些不良影响的机制还未充分发展。在数据收集过程中采用偏差感知采样的努力在数据控制实体也训练AI的情况下最为有效。当数据已经准备好时，如何检测数据是否已被操纵，即“污染”，以确定是否会将不良行为训练进机器学习模型中？这是一个与仅仅提高逼近精度或效率不同的根本性挑战。我们提供了一种方法来测试训练数据是否存在缺陷，以建立一个可信赖的地面真实值，用于随后的机器学习模型训练（任何类型）。与从数据生成模糊规则进行数据逼近的已研究问题不同，我们方法依赖于在看到测试数据之前先定义规则。因此，提出的方法还可以发现隐藏的错误模式，这些模式也可能具有重要影响。我们的方法扩展了传统统计测试的能力，使“测试条件”可以是任何布尔条件来描述数据中的模式，我们希望确定其存在性。该方法将模糊推理与回归模型结合，结合了模糊逻辑的解释性与统计特性和诊断功能，并且适用于“小数据”，因此无需像深度学习方法那样依赖大量数据集。我们提供了一个开源实现用于演示和实验。 

---
# Words as Bridges: Exploring Computational Support for Cross-Disciplinary Translation Work 

**Title (ZH)**: 词语作为桥梁：探索跨学科翻译工作中的计算支持 

**Authors**: Calvin Bao, Yow-Ting Shiue, Marine Carpuat, Joel Chan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18471)  

**Abstract**: Scholars often explore literature outside of their home community of study. This exploration process is frequently hampered by field-specific jargon. Past computational work often focuses on supporting translation work by removing jargon through simplification and summarization; here, we explore a different approach that preserves jargon as useful bridges to new conceptual spaces. Specifically, we cast different scholarly domains as different language-using communities, and explore how to adapt techniques from unsupervised cross-lingual alignment of word embeddings to explore conceptual alignments between domain-specific word embedding this http URL developed a prototype cross-domain search engine that uses aligned domain-specific embeddings to support conceptual exploration, and tested this prototype in two case studies. We discuss qualitative insights into the promises and pitfalls of this approach to translation work, and suggest design insights for future interfaces that provide computational support for cross-domain information seeking. 

**Abstract (ZH)**: 学者们常探索其研究社群之外的文献。这一探索过程经常受到领域特定术语的阻碍。以往的计算工作往往侧重于通过简化和总结去除术语以支持翻译工作；相比之下，我们探索了一种不同的方法，该方法保留术语作为连接新概念空间的有用桥梁。具体而言，我们将不同的学术领域视为不同的语言使用社区，并探索如何将无监督跨境词嵌入对齐的技术应用于领域特定词嵌入的概念对齐。我们开发了一个原型跨域搜索引擎，该引擎使用对齐的领域特定嵌入来支持概念探索，并在两个案例研究中测试了该原型。我们讨论了这种方法在翻译工作中的潜力与局限性的定性见解，并提出了为跨域信息检索提供计算支持的未来界面的设计建议。 

---
# PALATE: Peculiar Application of the Law of Total Expectation to Enhance the Evaluation of Deep Generative Models 

**Title (ZH)**: PALATE：独特应用全面期望定律以增强深度生成模型的评估 

**Authors**: Tadeusz Dziarmaga, Marcin Kądziołka, Artur Kasymov, Marcin Mazur  

**Link**: [PDF](https://arxiv.org/pdf/2503.18462)  

**Abstract**: Deep generative models (DGMs) have caused a paradigm shift in the field of machine learning, yielding noteworthy advancements in domains such as image synthesis, natural language processing, and other related areas. However, a comprehensive evaluation of these models that accounts for the trichotomy between fidelity, diversity, and novelty in generated samples remains a formidable challenge. A recently introduced solution that has emerged as a promising approach in this regard is the Feature Likelihood Divergence (FLD), a method that offers a theoretically motivated practical tool, yet also exhibits some computational challenges. In this paper, we propose PALATE, a novel enhancement to the evaluation of DGMs that addresses limitations of existing metrics. Our approach is based on a peculiar application of the law of total expectation to random variables representing accessible real data. When combined with the MMD baseline metric and DINOv2 feature extractor, PALATE offers a holistic evaluation framework that matches or surpasses state-of-the-art solutions while providing superior computational efficiency and scalability to large-scale datasets. Through a series of experiments, we demonstrate the effectiveness of the PALATE enhancement, contributing a computationally efficient, holistic evaluation approach that advances the field of DGMs assessment, especially in detecting sample memorization and evaluating generalization capabilities. 

**Abstract (ZH)**: 深度生成模型（DGMs）在机器学习领域引发了范式转变，取得了在图像合成、自然语言处理及其它相关领域的显著进展。然而，全面评估这些模型，考虑到生成样本在忠实性、多样性和新颖性之间的平衡，仍是一项艰巨的挑战。最近提出的一种潜在有前景的解决方案是特征似然散度（FLD），它提供了一种理论依据的实际工具，但也存在一定的计算挑战。在本文中，我们提出了PALATE，一种针对DGMs评估的新颖增强方法，以解决现有评估指标的局限性。我们的方法基于对随机变量表示可访问真实数据的总期望定律的特殊应用。结合MMD基准度量和DINOv2特征提取器，PALATE提供了一种综合评估框架，匹配甚至超越现有先进解决方案，同时具备更高的计算效率和面向大规模数据集的可扩展性。通过一系列实验，我们展示了PALATE增强的有效性，提出了一种计算效率高、综合的评估方法，推动了DGMs评估领域的发展，特别是在检测样本记忆和评估泛化能力方面。 

---
# Generative AI in Knowledge Work: Design Implications for Data Navigation and Decision-Making 

**Title (ZH)**: 生成式AI在知识工作中的设计影响：数据导航与决策制定 

**Authors**: Bhada Yun, Dana Feng, Ace S. Chen, Afshin Nikzad, Niloufar Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2503.18419)  

**Abstract**: Our study of 20 knowledge workers revealed a common challenge: the difficulty of synthesizing unstructured information scattered across multiple platforms to make informed decisions. Drawing on their vision of an ideal knowledge synthesis tool, we developed Yodeai, an AI-enabled system, to explore both the opportunities and limitations of AI in knowledge work. Through a user study with 16 product managers, we identified three key requirements for Generative AI in knowledge work: adaptable user control, transparent collaboration mechanisms, and the ability to integrate background knowledge with external information. However, we also found significant limitations, including overreliance on AI, user isolation, and contextual factors outside the AI's reach. As AI tools become increasingly prevalent in professional settings, we propose design principles that emphasize adaptability to diverse workflows, accountability in personal and collaborative contexts, and context-aware interoperability to guide the development of human-centered AI systems for product managers and knowledge workers. 

**Abstract (ZH)**: 我们的研究发现，20名知识工作者面临一个共同挑战：在多个平台上将分散的非结构化信息综合起来以做出知情决策的难度。基于他们对理想的知识综合工具的构想，我们开发了Yodeai这一AI赋能系统，以探索AI在知识工作中的机遇与局限。通过16名产品管理者的用户研究，我们确定了生成式AI在知识工作中的三项关键要求：可适应的用户控制、透明的协作机制以及将背景知识与外部信息整合的能力。然而，我们也发现了显著的局限性，包括对AI的过度依赖、用户孤立以及AI无法触及的上下文因素。随着AI工具在专业环境中的普及，我们提出了侧重于适应多样工作流程、个人和协作情境中的问责制以及上下文感知的互操作性的设计原则，以指导面向产品管理人员和知识工作者的人本AI系统开发。 

---
# PRECTR: A Synergistic Framework for Integrating Personalized Search Relevance Matching and CTR Prediction 

**Title (ZH)**: PRECTR：一种结合个性化搜索相关性匹配和点击率预测的协同框架 

**Authors**: Rong Chen, Shuzhi Cao, Ailong He, Shuguang Han, Jufeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18395)  

**Abstract**: The two primary tasks in the search recommendation system are search relevance matching and click-through rate (CTR) prediction -- the former focuses on seeking relevant items for user queries whereas the latter forecasts which item may better match user interest. Prior research typically develops two models to predict the CTR and search relevance separately, then ranking candidate items based on the fusion of the two outputs. However, such a divide-and-conquer paradigm creates the inconsistency between different models. Meanwhile, the search relevance model mainly concentrates on the degree of objective text matching while neglecting personalized differences among different users, leading to restricted model performance. To tackle these issues, we propose a unified \textbf{P}ersonalized Search RElevance Matching and CTR Prediction Fusion Model(PRECTR). Specifically, based on the conditional probability fusion mechanism, PRECTR integrates the CTR prediction and search relevance matching into one framework to enhance the interaction and consistency of the two modules. However, directly optimizing CTR binary classification loss may bring challenges to the fusion model's convergence and indefinitely promote the exposure of items with high CTR, regardless of their search relevance. Hence, we further introduce two-stage training and semantic consistency regularization to accelerate the model's convergence and restrain the recommendation of irrelevant items. Finally, acknowledging that different users may have varied relevance preferences, we assessed current users' relevance preferences by analyzing past users' preferences for similar queries and tailored incentives for different candidate items accordingly. Extensive experimental results on our production dataset and online A/B testing demonstrate the effectiveness and superiority of our proposed PRECTR method. 

**Abstract (ZH)**: 一种统一的个性化搜索相关性匹配与点击率预测融合模型(PRECTR) 

---
# RoCA: Robust Contrastive One-class Time Series Anomaly Detection with Contaminated Data 

**Title (ZH)**: RoCA: 健 robust 对比单一类时间序列异常检测方法在污染数据中的应用 

**Authors**: Xudong Mou, Rui Wang, Bo Li, Tianyu Wo, Jie Sun, Hui Wang, Xudong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18385)  

**Abstract**: The accumulation of time-series signals and the absence of labels make time-series Anomaly Detection (AD) a self-supervised task of deep learning. Methods based on normality assumptions face the following three limitations: (1) A single assumption could hardly characterize the whole normality or lead to some deviation. (2) Some assumptions may go against the principle of AD. (3) Their basic assumption is that the training data is uncontaminated (free of anomalies), which is unrealistic in practice, leading to a decline in robustness. This paper proposes a novel robust approach, RoCA, which is the first to address all of the above three challenges, as far as we are aware. It fuses the separated assumptions of one-class classification and contrastive learning in a single training process to characterize a more complete so-called normality. Additionally, it monitors the training data and computes a carefully designed anomaly score throughout the training process. This score helps identify latent anomalies, which are then used to define the classification boundary, inspired by the concept of outlier exposure. The performance on AIOps datasets improved by 6% compared to when contamination was not considered (COCA). On two large and high-dimensional multivariate datasets, the performance increased by 5% to 10%. RoCA achieves the highest average performance on both univariate and multivariate datasets. The source code is available at this https URL. 

**Abstract (ZH)**: 时间序列异序检测中的时间序列信号积累和无标签问题使得时间序列异序检测（AD）成为一个基于深度学习的自监督任务。基于正态性假设的方法面临以下三个局限：（1）单一假设难以全面描述正态性或导致偏差。（2）某些假设可能违反异序检测的基本原则。（3）其基本假设是训练数据未受污染（不含异常值），这在实际应用中是不现实的，导致鲁棒性下降。本文提出了一种新型鲁棒方法RoCA，据我们所知，这是首款能够同时解决上述所有三个挑战的方法。它在单一训练过程中融合了一类分类和对比学习的分离假设，以更全面地描述所谓的正态性。此外，该方法在整个训练过程中监控训练数据并计算精心设计的异常分数，该分数有助于识别潜在异常值，进而定义分类边界，这一过程借鉴了离群值暴露的概念。在AIOps数据集中，与未考虑污染的情况相比（COCA），性能提高了6%。在两个大型高维多变量数据集中，性能提高了5%到10%。RoCA在单变量和多变量数据集中均实现了最高平均性能。源代码可在以下网址获取。 

---
# PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition 

**Title (ZH)**: PP-FormulaNet：在高级公式识别中平衡准确性和效率 

**Authors**: Hongen Liu, Cheng Cui, Yuning Du, Yi Liu, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18382)  

**Abstract**: Formula recognition is an important task in document intelligence. It involves converting mathematical expressions from document images into structured symbolic formats that computers can easily work with. LaTeX is the most common format used for this purpose. In this work, we present PP-FormulaNet, a state-of-the-art formula recognition model that excels in both accuracy and efficiency. To meet the diverse needs of applications, we have developed two specialized models: PP-FormulaNet-L, tailored for high-accuracy scenarios, and PP-FormulaNet-S, optimized for high-efficiency contexts. Our extensive evaluations reveal that PP-FormulaNet-L attains accuracy levels that surpass those of prominent models such as UniMERNet by a significant 6%. Conversely, PP-FormulaNet-S operates at speeds that are over 16 times faster. These advancements facilitate seamless integration of PP-FormulaNet into a broad spectrum of document processing environments that involve intricate mathematical formulas. Furthermore, we introduce a Formula Mining System, which is capable of extracting a vast amount of high-quality formula data. This system further enhances the robustness and applicability of our formula recognition model. Code and models are publicly available at PaddleOCR(this https URL) and PaddleX(this https URL). 

**Abstract (ZH)**: 公式识别是文档智能中的一个重要任务。它涉及将文档图像中的数学表达式转换为计算机可以轻松处理的结构化符号格式。LaTeX是用于此目的最常见的格式。在此工作中，我们提出了一种先进公式识别模型PP-FormulaNet，既准确又高效。为满足各种应用场景的需求，我们开发了两种专门模型：PP-FormulaNet-L，适用于高精度场景；以及PP-FormulaNet-S，优化了高效率场景。广泛评估表明，PP-FormulaNet-L在准确率上比UniMERNet等知名模型高出显著的6%；而PP-FormulaNet-S的运行速度快了逾16倍。这些进步使得PP-FormulaNet能够无缝集成到涉及复杂数学公式的广泛文档处理环境中。此外，我们引入了公式挖掘系统，可以提取大量高质量的公式数据，进一步增强了我们的公式识别模型的 robustness 和适用性。相关代码和模型可在PaddleOCR和PaddleX公开访问。 

---
# Optimizing Influence Campaigns: Nudging under Bounded Confidence 

**Title (ZH)**: 优化影响竞选活动：在有限信任下引导 

**Authors**: Yen-Shao Chen, Tauhid Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2503.18331)  

**Abstract**: Influence campaigns in online social networks are often run by organizations, political parties, and nation states to influence large audiences. These campaigns are employed through the use of agents in the network that share persuasive content. Yet, their impact might be minimal if the audiences remain unswayed, often due to the bounded confidence phenomenon, where only a narrow spectrum of viewpoints can influence them. Here we show that to persuade under bounded confidence, an agent must nudge its targets to gradually shift their opinions. Using a control theory approach, we show how to construct an agent's nudging policy under the bounded confidence opinion dynamics model and also how to select targets for multiple agents in an influence campaign on a social network. Simulations on real Twitter networks show that a multi-agent nudging policy can shift the mean opinion, decrease opinion polarization, or even increase it. We find that our nudging based policies outperform other common techniques that do not consider the bounded confidence effect. Finally, we show how to craft prompts for large language models, such as ChatGPT, to generate text-based content for real nudging policies. This illustrates the practical feasibility of our approach, allowing one to go from mathematical nudging policies to real social media content. 

**Abstract (ZH)**: 有界信心下的在线社交网络影响campaign通过网络代理传播说服性内容，但其影响可能因观众固守观点而有限。我们展示了在有界信心条件下，代理需要逐渐引导其目标改变观点以实现说服。利用控制理论方法，我们展示了如何在有界信心意见动力学模型下构造代理的引导策略，并在社交网络影响campaign中选择多个代理的目标。实测Twitter网络的模拟结果显示，多代理引导策略可以改变平均观点，降低观点极化，甚至增加极化。我们发现基于引导的策略优于未考虑有界信心效应的其他常用技术。最后，我们展示了如何为大型语言模型，如ChatGPT，制定提示生成基于文本的引导策略，这表明我们方法的实际可行性，可以从数学上的引导策略过渡到实际的社交媒体内容。 

---
# LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty 

**Title (ZH)**: LoTUS: 大规模机器遗忘与不确定性口味 

**Authors**: Christoforos N. Spartalis, Theodoros Semertzidis, Stratis Gavves, Petros Daras  

**Link**: [PDF](https://arxiv.org/pdf/2503.18314)  

**Abstract**: We present LoTUS, a novel Machine Unlearning (MU) method that eliminates the influence of training samples from pre-trained models, avoiding retraining from scratch. LoTUS smooths the prediction probabilities of the model -- up to an information theoretic bound -- mitigating its over-confidence that stems from data memorization. We evaluate LoTUS on the Transformer and ResNet18 models, against eight baseline methods, on five public datasets. Beyond established MU benchmarks, we evaluate unlearning on a large-scale dataset (ImageNet1k) which deters retraining, simulating real-world conditions. Moreover, we introduce the novel Retrain-Free Jensen-Shannon Divergence (RF-JSD) metric to enable evaluation under real-world conditions. Experimental results show that LoTUS outperforms state-of-the-art methods in terms of both efficiency and effectiveness. Code: this https URL. 

**Abstract (ZH)**: 我们提出LoTUS，这是一种新颖的机器遗忘方法，能够从预训练模型中消除训练样本的影响，避免从头开始重新训练。LoTUS通过信息论边界平滑模型的预测概率，缓解其由于数据记忆导致的过度自信。我们在Transformer和ResNet18模型上，针对八种基线方法，在五个公开数据集上评估LoTUS。除了现有的机器遗忘基准之外，我们还在一个大规模数据集（ImageNet1k）上评估遗忘效果，该数据集难以重新训练，模拟了实际条件。此外，我们引入了一种新的无重新训练的杰况-舍恩贝格尔散度（RF-JSD）度量标准，以在实际条件下进行评估。实验结果表明，LoTUS在效率和效果上都优于现有方法。代码：这个 https URL。 

---
# When is dataset cartography ineffective? Using training dynamics does not improve robustness against Adversarial SQuAD 

**Title (ZH)**: 数据集地理测绘在何种情况下无效？使用训练动力学不能提高对抗SQuAD的鲁棒性 

**Authors**: Paul K. Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2503.18290)  

**Abstract**: In this paper, I investigate the effectiveness of dataset cartography for extractive question answering on the SQuAD dataset. I begin by analyzing annotation artifacts in SQuAD and evaluate the impact of two adversarial datasets, AddSent and AddOneSent, on an ELECTRA-small model. Using training dynamics, I partition SQuAD into easy-to-learn, ambiguous, and hard-to-learn subsets. I then compare the performance of models trained on these subsets to those trained on randomly selected samples of equal size. Results show that training on cartography-based subsets does not improve generalization to the SQuAD validation set or the AddSent adversarial set. While the hard-to-learn subset yields a slightly higher F1 score on the AddOneSent dataset, the overall gains are limited. These findings suggest that dataset cartography provides little benefit for adversarial robustness in SQuAD-style QA tasks. I conclude by comparing these results to prior findings on SNLI and discuss possible reasons for the observed differences. 

**Abstract (ZH)**: 本文探讨了数据集测绘在SQuAD数据集上提取式问答任务中的有效性。通过分析SQuAD的注释特征，并评估AddSent和AddOneSent两个对抗数据集对ELECTRA-small模型的影响，利用训练动力学将SQuAD划分为易学、模糊和难学子集，对比这些子集训练的模型与随机等量样本训练的模型的性能。结果表明，基于数据集测绘划分的子集训练并不能提高对SQuAD验证集或AddSent对抗集的泛化能力。虽然难学子集在AddOneSent数据集上的F1分数略有提高，但总体增益有限。这些发现表明，数据集测绘在SQuAD风格的问答任务中对对抗鲁棒性的提升作用有限。最后，将这些结果与SNLI之前的发现进行了比较，并讨论了观察到差异的可能原因。 

---
# Risk Management for Distributed Arbitrage Systems: Integrating Artificial Intelligence 

**Title (ZH)**: 分布式套利系统中的风险管理：整合人工智能 

**Authors**: Akaash Vishal Hazarika, Mahak Shah, Swapnil Patil, Pradyumna Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2503.18265)  

**Abstract**: Effective risk management solutions become absolutely crucial when financial markets embrace distributed technology and decentralized financing (DeFi). This study offers a thorough survey and comparative analysis of the integration of artificial intelligence (AI) in risk management for distributed arbitrage systems. We examine several modern caching techniques namely in memory caching, distributed caching, and proxy caching and their functions in enhancing performance in decentralized settings. Through literature review we examine the utilization of AI techniques for alleviating risks related to market volatility, liquidity challenges, operational failures, regulatory compliance, and security threats. This comparison research evaluates various case studies from prominent DeFi technologies, emphasizing critical performance metrics like latency reduction, load balancing, and system resilience. Additionally, we examine the problems and trade offs associated with these technologies, emphasizing their effects on consistency, scalability, and fault tolerance. By meticulously analyzing real world applications, specifically centering on the Aave platform as our principal case study, we illustrate how the purposeful amalgamation of AI with contemporary caching methodologies has revolutionized risk management in distributed arbitrage systems. 

**Abstract (ZH)**: 有效风险管理解决方案在金融 markets拥抱分布式技术与去中心化融资（DeFi）时变得至关重要。本研究提供了人工智能（AI）在分布式对冲系统风险管理集成中的全面综述与比较分析。我们探讨了几种现代缓存技术，包括内存缓存、分布式缓存和代理缓存，并分析了它们在去中心化环境中提升性能的功能。通过文献综述，我们研究了人工智能技术在缓解与市场波动、流动性挑战、运营失败、监管合规和安全威胁相关风险方面的应用。本比较研究评估了来自知名DeFi技术的多种案例研究，强调了关键性能指标，如延迟降低、负载均衡和系统韧性。此外，我们还分析了这些技术的问题和权衡，强调了它们对一致性和可扩展性以及容错性的影响。通过细致分析实际应用，特别是重点以Aave平台作为主要案例研究，我们展示了有目的地将AI与现代缓存方法相结合如何在分布式对冲系统中革新风险管理。 

---
# Severing Spurious Correlations with Data Pruning 

**Title (ZH)**: 切断虚假相关性的数据修剪方法 

**Authors**: Varun Mulchandani, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.18258)  

**Abstract**: Deep neural networks have been shown to learn and rely on spurious correlations present in the data that they are trained on. Reliance on such correlations can cause these networks to malfunction when deployed in the real world, where these correlations may no longer hold. To overcome the learning of and reliance on such correlations, recent studies propose approaches that yield promising results. These works, however, study settings where the strength of the spurious signal is significantly greater than that of the core, invariant signal, making it easier to detect the presence of spurious features in individual training samples and allow for further processing. In this paper, we identify new settings where the strength of the spurious signal is relatively weaker, making it difficult to detect any spurious information while continuing to have catastrophic consequences. We also discover that spurious correlations are learned primarily due to only a handful of all the samples containing the spurious feature and develop a novel data pruning technique that identifies and prunes small subsets of the training data that contain these samples. Our proposed technique does not require inferred domain knowledge, information regarding the sample-wise presence or nature of spurious information, or human intervention. Finally, we show that such data pruning attains state-of-the-art performance on previously studied settings where spurious information is identifiable. 

**Abstract (ZH)**: 深神经网络已被证明会学习并依赖于训练数据中存在的虚假相关性。在这些相关性在实际应用中不再成立的情况下，这些网络可能会失效。为了克服学习和依赖这些相关性的问题，最近的研究提出了一些取得良好效果的方法。然而，这些研究主要关注虚假信号强度显著大于核心不变信号强度的情景，使得在单个训练样本中检测到虚假特征并进行进一步处理更加容易。在本文中，我们识别了一种新的情景，其中虚假信号的强度相对较弱，这使得检测任何虚假信息变得更加困难，但仍会导致灾难性的后果。我们还发现，虚假相关性主要是由于少量包含虚假特征的样本而被学习，因此开发了一种新颖的数据修剪技术来识别和修剪训练数据中包含这些样本的小子集。我们提出的技术不需要推断领域的知识、样本级别虚假信息的存在与否或性质，也不需要人工干预。最后，我们表明这种数据修剪在先前研究中虚假信息可识别的情景中达到了最先进的性能。 

---
# The Human-Machine Identity Blur: A Unified Framework for Cybersecurity Risk Management in 2025 

**Title (ZH)**: 人机身份模糊：2025年网络安全风险统一管理框架 

**Authors**: Kush Janani  

**Link**: [PDF](https://arxiv.org/pdf/2503.18255)  

**Abstract**: The modern enterprise is facing an unprecedented surge in digital identities, with machine identities now significantly outnumbering human identities. This paper examines the cybersecurity risks emerging from what we define as the "human-machine identity blur" - the point at which human and machine identities intersect, delegate authority, and create new attack surfaces. Drawing from industry data, expert insights, and real-world incident analysis, we identify key governance gaps in current identity management models that treat human and machine entities as separate domains. To address these challenges, we propose a Unified Identity Governance Framework based on four core principles: treating identity as a continuum rather than a binary distinction, applying consistent risk evaluation across all identity types, implementing continuous verification guided by zero trust principles, and maintaining governance throughout the entire identity lifecycle. Our research shows that organizations adopting this unified approach experience a 47 percent reduction in identity-related security incidents and a 62 percent improvement in incident response time. We conclude by offering a practical implementation roadmap and outlining future research directions as AI-driven systems become increasingly autonomous. 

**Abstract (ZH)**: 现代企业面临的空前数字身份激增，机器身份现已显著超过人类身份。本文探讨了我们定义的“人机身份模糊”所带来的网络安全风险——即人类身份与机器身份交汇、权力委托并创建新的攻击面的点。基于行业数据、专家见解和实际案例分析，我们识别了当前身份管理模型中处理人类和机器实体作为分离领域存在的关键治理缺口。为应对这些挑战，我们提出了一种基于四大原则的统一体身份治理框架：将身份视为连续体而非二元区分，对所有身份类型应用一致的风险评估，根据零信任原则实施持续验证，并在整个身份生命周期中保持治理。我们的研究表明，采用这种统一方法的企业在身份相关安全事件中减少了47%，并在事件响应时间上提高了62%。我们以提供实用的实施路线图为结尾，并概述了随着AI驱动系统的日益自主，未来的研究方向。 

---
# Collaborating with AI Agents: Field Experiments on Teamwork, Productivity, and Performance 

**Title (ZH)**: 与AI代理合作：团队合作、生产力和表现的实地实验 

**Authors**: Harang Ju, Sinan Aral  

**Link**: [PDF](https://arxiv.org/pdf/2503.18238)  

**Abstract**: To uncover how AI agents change productivity, performance, and work processes, we introduce MindMeld: an experimentation platform enabling humans and AI agents to collaborate in integrative workspaces. In a large-scale marketing experiment on the platform, 2310 participants were randomly assigned to human-human and human-AI teams, with randomized AI personality traits. The teams exchanged 183,691 messages, and created 63,656 image edits, 1,960,095 ad copy edits, and 10,375 AI-generated images while producing 11,138 ads for a large think tank. Analysis of fine-grained communication, collaboration, and workflow logs revealed that collaborating with AI agents increased communication by 137% and allowed humans to focus 23% more on text and image content generation messaging and 20% less on direct text editing. Humans on Human-AI teams sent 23% fewer social messages, creating 60% greater productivity per worker and higher-quality ad copy. In contrast, human-human teams produced higher-quality images, suggesting that AI agents require fine-tuning for multimodal workflows. AI personality prompt randomization revealed that AI traits can complement human personalities to enhance collaboration. For example, conscientious humans paired with open AI agents improved image quality, while extroverted humans paired with conscientious AI agents reduced the quality of text, images, and clicks. In field tests of ad campaigns with ~5M impressions, ads with higher image quality produced by human collaborations and higher text quality produced by AI collaborations performed significantly better on click-through rate and cost per click metrics. Overall, ads created by human-AI teams performed similarly to those created by human-human teams. Together, these results suggest AI agents can improve teamwork and productivity, especially when tuned to complement human traits. 

**Abstract (ZH)**: 探索AI代理如何改变生产力、绩效和工作流程：MindMeld：一个让人类与AI代理在整合工作空间中协作的实验平台 

---
# ViVa: Video-Trained Value Functions for Guiding Online RL from Diverse Data 

**Title (ZH)**: ViVa: 基于视频训练的价值函数及其在多元数据引导在线强化学习中的应用 

**Authors**: Nitish Dashora, Dibya Ghosh, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2503.18210)  

**Abstract**: Online reinforcement learning (RL) with sparse rewards poses a challenge partly because of the lack of feedback on states leading to the goal. Furthermore, expert offline data with reward signal is rarely available to provide this feedback and bootstrap online learning. How can we guide online agents to the right solution without this on-task data? Reward shaping offers a solution by providing fine-grained signal to nudge the policy towards the optimal solution. However, reward shaping often requires domain knowledge to hand-engineer heuristics for a specific goal. To enable more general and inexpensive guidance, we propose and analyze a data-driven methodology that automatically guides RL by learning from widely available video data such as Internet recordings, off-task demonstrations, task failures, and undirected environment interaction. By learning a model of optimal goal-conditioned value from diverse passive data, we open the floor to scaling up and using various data sources to model general goal-reaching behaviors relevant to guiding online RL. Specifically, we use intent-conditioned value functions to learn from diverse videos and incorporate these goal-conditioned values into the reward. Our experiments show that video-trained value functions work well with a variety of data sources, exhibit positive transfer from human video pre-training, can generalize to unseen goals, and scale with dataset size. 

**Abstract (ZH)**: 基于视频数据的在线强化学习引导方法：利用稀疏奖励信号实现通用目标导向行为建模 

---
# FROG: Fair Removal on Graphs 

**Title (ZH)**: FROG: 图上的公平删除 

**Authors**: Ziheng Chen, Jiali Cheng, Gabriele Tolomei, Sijia Liu, Hadi Amiri, Yu Wang, Kaushiki Nag, Lu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.18197)  

**Abstract**: As compliance with privacy regulations becomes increasingly critical, the growing demand for data privacy has highlighted the significance of machine unlearning in many real world applications, such as social network and recommender systems, many of which can be represented as graph-structured data. However, existing graph unlearning algorithms indiscriminately modify edges or nodes from well-trained models without considering the potential impact of such structural modifications on fairness. For example, forgetting links between nodes with different genders in a social network may exacerbate group disparities, leading to significant fairness concerns. To address these challenges, we propose a novel approach that jointly optimizes the graph structure and the corresponding model for fair unlearning tasks. Specifically,our approach rewires the graph to enhance unlearning efficiency by removing redundant edges that hinder forgetting while preserving fairness through targeted edge augmentation. Additionally, we introduce a worst-case evaluation mechanism to assess the reliability of fair unlearning performance. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed approach in achieving superior unlearning outcomes. 

**Abstract (ZH)**: 随着遵守隐私法规变得越来越关键，数据隐私需求的增长凸显了在社交网络和推荐系统等许多实际应用中实现机器遗忘的重要性，这些应用可以表示为图结构数据。然而，现有的图遗忘算法在修改从训练好的模型中选择的边或节点时，并未考虑到这种结构修改可能对公平性产生的潜在影响。例如，在社交网络中遗忘不同性别节点之间的链接可能会加剧群体差异，从而引起显著的公平性问题。为应对这些挑战，我们提出了一种新的方法，该方法同时优化图结构和相应的模型以实现公正的遗忘任务。具体来说，我们的方法通过去除阻碍遗忘的冗余边来重连图，同时通过针对性地增加边来保持公平性。此外，我们引入了一种最坏情况评估机制来评估公正遗忘性能的可靠性。在真实数据集上的广泛实验表明，所提出的方法在实现优越的遗忘结果方面具有有效性。 

---
# Adaptive Physics-informed Neural Networks: A Survey 

**Title (ZH)**: 自适应物理约束神经网络：一个综述 

**Authors**: Edgar Torres, Jonathan Schiefer, Mathias Niepert  

**Link**: [PDF](https://arxiv.org/pdf/2503.18181)  

**Abstract**: Physics-informed neural networks (PINNs) have emerged as a promising approach to solving partial differential equations (PDEs) using neural networks, particularly in data-scarce scenarios, due to their unsupervised training capability. However, limitations related to convergence and the need for re-optimization with each change in PDE parameters hinder their widespread adoption across scientific and engineering applications. This survey reviews existing research that addresses these limitations through transfer learning and meta-learning. The covered methods improve the training efficiency, allowing faster adaptation to new PDEs with fewer data and computational resources. While traditional numerical methods solve systems of differential equations directly, neural networks learn solutions implicitly by adjusting their parameters. One notable advantage of neural networks is their ability to abstract away from specific problem domains, allowing them to retain, discard, or adapt learned representations to efficiently address similar problems. By exploring the application of these techniques to PINNs, this survey identifies promising directions for future research to facilitate the broader adoption of PINNs in a wide range of scientific and engineering applications. 

**Abstract (ZH)**: 物理约束神经网络（PINNs）通过神经网络求解偏微分方程（PDEs）的一种有前景的方法，特别是在数据稀少的情况下，由于其无监督训练能力而受到关注。然而，收敛性问题以及每更改一次PDE参数就需要重新优化的限制阻碍了其在科学和工程应用中的广泛应用。本文综述了通过迁移学习和元学习解决这些限制的研究进展。涵盖了的方法提高了训练效率，允许在更少的数据和计算资源下更快地适应新的PDEs。虽然传统的数值方法直接求解微分方程系统，神经网络通过调整参数隐式地学习解。神经网络的一个显著优势是能够脱离特定问题领域，允许它们保留、丢弃或适应学到的表示，以高效地解决类似问题。通过探索这些技术在PINNs中的应用，本文指出了未来研究的有前景的方向，以促进PINNs在广泛的科学和工程应用中的更广泛应用。 

---
# Evaluating Negative Sampling Approaches for Neural Topic Models 

**Title (ZH)**: 评估负采样方法在神经主题模型中的有效性 

**Authors**: Suman Adhya, Avishek Lahiri, Debarshi Kumar Sanyal, Partha Pratim Das  

**Link**: [PDF](https://arxiv.org/pdf/2503.18167)  

**Abstract**: Negative sampling has emerged as an effective technique that enables deep learning models to learn better representations by introducing the paradigm of learn-to-compare. The goal of this approach is to add robustness to deep learning models to learn better representation by comparing the positive samples against the negative ones. Despite its numerous demonstrations in various areas of computer vision and natural language processing, a comprehensive study of the effect of negative sampling in an unsupervised domain like topic modeling has not been well explored. In this paper, we present a comprehensive analysis of the impact of different negative sampling strategies on neural topic models. We compare the performance of several popular neural topic models by incorporating a negative sampling technique in the decoder of variational autoencoder-based neural topic models. Experiments on four publicly available datasets demonstrate that integrating negative sampling into topic models results in significant enhancements across multiple aspects, including improved topic coherence, richer topic diversity, and more accurate document classification. Manual evaluations also indicate that the inclusion of negative sampling into neural topic models enhances the quality of the generated topics. These findings highlight the potential of negative sampling as a valuable tool for advancing the effectiveness of neural topic models. 

**Abstract (ZH)**: 负采样已成为一种有效技术，通过引入学习比较的 paradigm，使深度学习模型能够学习更好的表示。该方法的目标是通过将正样本与负样本进行比较，增强深度学习模型学习更好表示的能力。尽管在计算机视觉和自然语言处理的各个领域中，负采样的效果得到了广泛证实，但对其在无监督领域如主题建模中的影响的研究尚不全面。在本文中，我们对不同负采样策略对神经主题模型的影响进行了全面分析。通过在基于变分自编码器的神经主题模型的解码器中引入负采样技术，我们将几种流行的神经主题模型的性能进行了比较。在四个公开数据集上的实验表明，将负采样整合到主题模型中，在多个方面均带来了显著的提升，包括提高了主题的一致性、增加了主题的多样性以及提高了文档分类的准确性。 manual评估也表明，将负采样纳入神经主题模型可以提高生成主题的质量。这些发现突显了负采样作为增强神经主题模型效果的有价值工具的潜力。 

---
# SNRAware: Improved Deep Learning MRI Denoising with SNR Unit Training and G-factor Map Augmentation 

**Title (ZH)**: SNRAware: 通过SNR单元训练和G因子图增强改进的深度学习MRI降噪方法 

**Authors**: Hui Xue, Sarah M. Hooper, Iain Pierce, Rhodri H. Davies, John Stairs, Joseph Naegele, Adrienne E. Campbell-Washburn, Charlotte Manisty, James C. Moon, Thomas A. Treibel, Peter Kellman, Michael S. Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18162)  

**Abstract**: To develop and evaluate a new deep learning MR denoising method that leverages quantitative noise distribution information from the reconstruction process to improve denoising performance and generalization.
This retrospective study trained 14 different transformer and convolutional models with two backbone architectures on a large dataset of 2,885,236 images from 96,605 cardiac retro-gated cine complex series acquired at 3T. The proposed training scheme, termed SNRAware, leverages knowledge of the MRI reconstruction process to improve denoising performance by simulating large, high quality, and diverse synthetic datasets, and providing quantitative information about the noise distribution to the model. In-distribution testing was performed on a hold-out dataset of 3000 samples with performance measured using PSNR and SSIM, with ablation comparison without the noise augmentation. Out-of-distribution tests were conducted on cardiac real-time cine, first-pass cardiac perfusion, and neuro and spine MRI, all acquired at 1.5T, to test model generalization across imaging sequences, dynamically changing contrast, different anatomies, and field strengths. The best model found in the in-distribution test generalized well to out-of-distribution samples, delivering 6.5x and 2.9x CNR improvement for real-time cine and perfusion imaging, respectively. Further, a model trained with 100% cardiac cine data generalized well to a T1 MPRAGE neuro 3D scan and T2 TSE spine MRI. 

**Abstract (ZH)**: 开发并评估一种新的深度学习MRI降噪方法，该方法利用重建过程中定量的噪声分布信息以提高降噪性能和泛化能力 

---
# Active Inference for Energy Control and Planning in Smart Buildings and Communities 

**Title (ZH)**: 智能建筑与社区中的能量控制与规划的主动推理方法 

**Authors**: Seyyed Danial Nazemi, Mohsen A. Jafari, Andrea Matta  

**Link**: [PDF](https://arxiv.org/pdf/2503.18161)  

**Abstract**: Active Inference (AIF) is emerging as a powerful framework for decision-making under uncertainty, yet its potential in engineering applications remains largely unexplored. In this work, we propose a novel dual-layer AIF architecture that addresses both building-level and community-level energy management. By leveraging the free energy principle, each layer adapts to evolving conditions and handles partial observability without extensive sensor information and respecting data privacy. We validate the continuous AIF model against both a perfect optimization baseline and a reinforcement learning-based approach. We also test the community AIF framework under extreme pricing scenarios. The results highlight the model's robustness in handling abrupt changes. This study is the first to show how a distributed AIF works in engineering. It also highlights new opportunities for privacy-preserving and uncertainty-aware control strategies in engineering applications. 

**Abstract (ZH)**: 基于活跃推断的分布式能源管理架构及其在工程应用中的探索：隐私保护与不确定性感知控制的新机遇 

---
# Adoption of Watermarking for Generative AI Systems in Practice and Implications under the new EU AI Act 

**Title (ZH)**: 实践中的生成式人工智能系统水印采用及在新欧盟人工智能法案下的 implications 

**Authors**: Bram Rijsbosch, Gijs van Dijck, Konrad Kollnig  

**Link**: [PDF](https://arxiv.org/pdf/2503.18156)  

**Abstract**: AI-generated images have become so good in recent years that individuals cannot distinguish them any more from "real" images. This development creates a series of societal risks, and challenges our perception of what is true and what is not, particularly with the emergence of "deep fakes" that impersonate real individuals. Watermarking, a technique that involves embedding identifying information within images to indicate their AI-generated nature, has emerged as a primary mechanism to address the risks posed by AI-generated images. The implementation of watermarking techniques is now becoming a legal requirement in many jurisdictions, including under the new 2024 EU AI Act. Despite the widespread use of AI image generation systems, the current status of watermarking implementation remains largely unexamined. Moreover, the practical implications of the AI Act's watermarking requirements have not previously been studied. The present paper therefore both provides an empirical analysis of 50 of the most widely used AI systems for image generation, and embeds this empirical analysis into a legal analysis of the AI Act. We identify four categories of generative AI image systems relevant under the AI Act, outline the legal obligations for each category, and find that only a minority number of providers currently implement adequate watermarking practices. 

**Abstract (ZH)**: AI生成图像的水印标识：现状、法律要求及实证分析 

---
# LocDiffusion: Identifying Locations on Earth by Diffusing in the Hilbert Space 

**Title (ZH)**: LocDiffusion：在希尔伯特空间中扩散以识别地球上的位置 

**Authors**: Zhangyu Wang, Jielu Zhang, Zhongliang Zhou, Qian Cao, Nemin Wu, Zeping Liu, Lan Mu, Yang Song, Yiqun Xie, Ni Lao, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2503.18142)  

**Abstract**: Image geolocalization is a fundamental yet challenging task, aiming at inferring the geolocation on Earth where an image is taken. Existing methods approach it either via grid-based classification or via image retrieval. Their performance significantly suffers when the spatial distribution of test images does not align with such choices. To address these limitations, we propose to leverage diffusion as a mechanism for image geolocalization. To avoid the problematic manifold reprojection step in diffusion, we developed a novel spherical positional encoding-decoding framework, which encodes points on a spherical surface (e.g., geolocations on Earth) into a Hilbert space of Spherical Harmonics coefficients and decodes points (geolocations) by mode-seeking. We call this type of position encoding Spherical Harmonics Dirac Delta (SHDD) Representation. We also propose a novel SirenNet-based architecture called CS-UNet to learn the conditional backward process in the latent SHDD space by minimizing a latent KL-divergence loss. We train a conditional latent diffusion model called LocDiffusion that generates geolocations under the guidance of images -- to the best of our knowledge, the first generative model for image geolocalization by diffusing geolocation information in a hidden location embedding space. We evaluate our method against SOTA image geolocalization baselines. LocDiffusion achieves competitive geolocalization performance and demonstrates significantly stronger generalizability to unseen geolocations. 

**Abstract (ZH)**: 图像地理定位是基础而又具有挑战性的任务，旨在推断图像拍摄地的地理位置。现有方法要么通过基于网格的分类，要么通过图像检索来实现这一目标。当测试图像的空间分布与这些选择不一致时，其性能会显著下降。为解决这些问题，我们提出利用扩散机制进行图像地理定位。为避免扩散过程中的有争议的流形重建步骤，我们开发了一种新颖的球面位置编码-解码框架，将球面上的点（例如，地球上的地理坐标）编码为球谐系数的希尔伯特空间，并通过模式寻找进行解码。我们将这种类型的位置编码称为球谐狄拉克δ表示（SHDD表示）。我们还提出了一种基于SirenNet的新型架构CS-UNet，通过最小化潜在的KL散度损失来学习潜在SHDD空间中的条件逆过程。我们训练了一个条件潜在扩散模型LocDiffusion，在图像的指导下生成地理坐标——据我们所知，这是首个通过在隐藏位置嵌入空间中扩散地理坐标信息来进行图像地理定位的生成模型。我们将我们的方法与当前最先进的图像地理定位基线进行评估。LocDiffusion在地理定位性能上取得了竞争力，并且在未知地理坐标的泛化能力上展现出显著的优势。 

---
# Temporal Relation Extraction in Clinical Texts: A Span-based Graph Transformer Approach 

**Title (ZH)**: 临床文本中的时间关系提取：一种基于跨度的图变换器方法 

**Authors**: Rochana Chaturvedi, Peyman Baghershahi, Sourav Medya, Barbara Di Eugenio  

**Link**: [PDF](https://arxiv.org/pdf/2503.18085)  

**Abstract**: Temporal information extraction from unstructured text is essential for contextualizing events and deriving actionable insights, particularly in the medical domain. We address the task of extracting clinical events and their temporal relations using the well-studied I2B2 2012 Temporal Relations Challenge corpus. This task is inherently challenging due to complex clinical language, long documents, and sparse annotations. We introduce GRAPHTREX, a novel method integrating span-based entity-relation extraction, clinical large pre-trained language models (LPLMs), and Heterogeneous Graph Transformers (HGT) to capture local and global dependencies. Our HGT component facilitates information propagation across the document through innovative global landmarks that bridge distant entities. Our method improves the state-of-the-art with 5.5% improvement in the tempeval $F_1$ score over the previous best and up to 8.9% improvement on long-range relations, which presents a formidable challenge. This work not only advances temporal information extraction but also lays the groundwork for improved diagnostic and prognostic models through enhanced temporal reasoning. 

**Abstract (ZH)**: 从无结构文本中提取时间信息对于事件的语境化和衍生行动洞见至关重要，特别是在医疗领域。我们使用广泛研究的I2B2 2012时间关系挑战语料库来解决提取临床事件及其时间关系的任务。该任务由于复杂的临床语言、长文档和稀疏注解而具有固有的挑战性。我们提出了GRAPHTREX，一种结合基于跨度的实体-关系提取、临床大型预训练语言模型（LPLMs）和异构图变换器（HGT）的新方法，以捕捉局部和全局依赖关系。我们的HGT组件通过创新的全局地标来促进在文档中远程实体之间的信息传播。我们的方法在tempeval $F_1$评分上相较于之前最佳方法提高了5.5%，在长距离关系上最多提高了8.9%，这构成了一个严峻的挑战。这项工作不仅促进了时间信息的提取，也为通过增强时间推理改进诊断和预后模型奠定了基础。 

---
# Dynamic Task Vector Grouping for Efficient Multi-Task Prompt Tuning 

**Title (ZH)**: 动态任务向量分组以实现高效的多任务提示调优 

**Authors**: Pieyi Zhang, Richong Zhang, Zhijie Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.18063)  

**Abstract**: Multi-task prompt tuning utilizes multiple high-resource source tasks to improve performance on low-source target tasks. Existing approaches transfer the soft prompt trained by combining all source tasks or a single ``high-similar'' source task one-time-only. However, we find that the optimal transfer performance often comes from a combination of source tasks, which is neither one nor all. Further, we find that the similarity between source and target tasks also changes dynamically during fine-tuning after transfering, making similarity calculation in the initiation stage inadequate. To address these issues, we propose a method called Dynamic Task Vector Grouping (DTVG), whose core ideas contain (1) measuring the task similarity with task vectors instead of soft prompt, (2) grouping the optimal source task combination based on two metrics: {\it target similarity} and {\it knowledge consistency}; (3) dynamically updating the combination in each iteration step. Extensive experiments on the 26 NLP datasets under different settings demonstrate that DTVG effectively groups similar source tasks while reducing negative transfer, achieving the start-of-art performance. 

**Abstract (ZH)**: 动态任务向量分组方法（DTVG）在不同设置下的26个NLP数据集上的广泛实验表明，DTVG能够有效分组相似源任务的同时减少负迁移，取得业界最佳性能。 

---
# Decision from Suboptimal Classifiers: Excess Risk Pre- and Post-Calibration 

**Title (ZH)**: 从次优分类器的决策：校准前后超过风险解析 

**Authors**: Alexandre Perez-Lebel, Gael Varoquaux, Sanmi Koyejo, Matthieu Doutreligne, Marine Le Morvan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18025)  

**Abstract**: Probabilistic classifiers are central for making informed decisions under uncertainty. Based on the maximum expected utility principle, optimal decision rules can be derived using the posterior class probabilities and misclassification costs. Yet, in practice only learned approximations of the oracle posterior probabilities are available. In this work, we quantify the excess risk (a.k.a. regret) incurred using approximate posterior probabilities in batch binary decision-making. We provide analytical expressions for miscalibration-induced regret ($R^{\mathrm{CL}}$), as well as tight and informative upper and lower bounds on the regret of calibrated classifiers ($R^{\mathrm{GL}}$). These expressions allow us to identify regimes where recalibration alone addresses most of the regret, and regimes where the regret is dominated by the grouping loss, which calls for post-training beyond recalibration. Crucially, both $R^{\mathrm{CL}}$ and $R^{\mathrm{GL}}$ can be estimated in practice using a calibration curve and a recent grouping loss estimator. On NLP experiments, we show that these quantities identify when the expected gain of more advanced post-training is worth the operational cost. Finally, we highlight the potential of multicalibration approaches as efficient alternatives to costlier fine-tuning approaches. 

**Abstract (ZH)**: 概率分类器在不确定性下进行 informed 决策中至关重要。基于最大期望效用原则，可以利用后验类概率和误分类成本推导出最优决策规则。然而，在实践中只能获得后验概率的近似值。在本文中，我们量化了使用近似后验概率进行批处理二元决策时产生的 excess risk（亦称为 regret）。我们提供了 miscalibration 引发的 regret ($R^{\mathrm{CL}}$) 的解析表达式，以及校准分类器 ($R^{\mathrm{GL}}$) 的 regret 的紧致且信息丰富的上界和下界。这些表达式使我们能够识别出仅通过重新校准可以解决大部分 regret 的情况，以及 regret 主要由 grouping loss 控制的情况，后者需要超出重新校准的后续训练。关键的是，$R^{\mathrm{CL}}$ 和 $R^{\mathrm{GL}}$ 在实践中都可以通过校准曲线和最近的 grouping loss 估计器进行估计。在 NLP 实验中，我们展示了这些量度可以识别出更高级的后续训练带来的期望收益是否值得运营成本。最后，我们强调了多校准方法作为成本更低的微调方法替代方案的潜力。 

---
# Predicting Multitasking in Manual and Automated Driving with Optimal Supervisory Control 

**Title (ZH)**: 基于最优监督控制的 Manual 和 Automated 驾驶中的 multitasking 预测 

**Authors**: Jussi Jokinen, Patrick Ebel, Tuomo Kujala  

**Link**: [PDF](https://arxiv.org/pdf/2503.17993)  

**Abstract**: Modern driving involves interactive technologies that can divert attention, increasing the risk of accidents. This paper presents a computational cognitive model that simulates human multitasking while driving. Based on optimal supervisory control theory, the model predicts how multitasking adapts to variations in driving demands, interactive tasks, and automation levels. Unlike previous models, it accounts for context-dependent multitasking across different degrees of driving automation. The model predicts longer in-car glances on straight roads and shorter glances during curves. It also anticipates increased glance durations with driver aids such as lane-centering assistance and their interaction with environmental demands. Validated against two empirical datasets, the model offers insights into driver multitasking amid evolving in-car technologies and automation. 

**Abstract (ZH)**: 现代驾驶涉及可以分散注意力的交互技术，增加了事故风险。本文提出了一种计算认知模型，模拟驾驶中的多任务处理。该模型基于最优监督控制理论，预测多任务处理如何适应驾驶需求、交互任务以及自动化水平的变化。与之前模型不同，它考虑了不同驾驶自动化程度下的情境依赖性多任务处理。模型预测直路上更长的车内注视时间和弯道中更短的注视时间。它还预测了车道保持辅助等驾驶辅助工具及其与环境需求交互时注视时间增加的情况。该模型在两个实验数据集上得到了验证，提供了关于不断演变的车内技术和自动化背景下驾驶员多任务处理的见解。 

---
# Taste More, Taste Better: Diverse Data and Strong Model Boost Semi-Supervised Crowd Counting 

**Title (ZH)**: 味道更丰富，精度更高：多样数据与强模型助推半监督人群计数 

**Authors**: Maochen Yang, Zekun Li, Jian Zhang, Lei Qi, Yinghuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17984)  

**Abstract**: Semi-supervised crowd counting is crucial for addressing the high annotation costs of densely populated scenes. Although several methods based on pseudo-labeling have been proposed, it remains challenging to effectively and accurately utilize unlabeled data. In this paper, we propose a novel framework called Taste More Taste Better (TMTB), which emphasizes both data and model aspects. Firstly, we explore a data augmentation technique well-suited for the crowd counting task. By inpainting the background regions, this technique can effectively enhance data diversity while preserving the fidelity of the entire scenes. Secondly, we introduce the Visual State Space Model as backbone to capture the global context information from crowd scenes, which is crucial for extremely crowded, low-light, and adverse weather scenarios. In addition to the traditional regression head for exact prediction, we employ an Anti-Noise classification head to provide less exact but more accurate supervision, since the regression head is sensitive to noise in manual annotations. We conduct extensive experiments on four benchmark datasets and show that our method outperforms state-of-the-art methods by a large margin. Code is publicly available on this https URL. 

**Abstract (ZH)**: 半监督人群计数对于解决密集人群场景的高标注成本至关重要。尽管已经提出了一些基于伪标签的方法，但仍难以有效准确地利用未标注数据。在本文中，我们提出了一种名为Taste More Taste Better (TMTB)的新框架，强调数据和模型两个方面。首先，我们探索了一种适用于人群计数任务的数据增强技术。通过修复背景区域，该技术可以有效增强数据多样性的同时保持整个场景的保真度。其次，我们引入了Visual State Space Model作为骨干网络，以从人群场景中捕获全局上下文信息，这对于极 crowded、低光和恶劣天气场景至关重要。除了传统的回归头进行精确预测外，我们还采用了Anti-Noise分类头提供较少精确但更准确的监督，因为回归头对手动标注中的噪声敏感。我们在四个基准数据集上进行了广泛的实验，并显示我们的方法在多个指标上显著优于现有最先进的方法。代码已在以下网址公开：此https URL。 

---
# PIM: Physics-Informed Multi-task Pre-training for Improving Inertial Sensor-Based Human Activity Recognition 

**Title (ZH)**: 基于物理约束的多任务预训练以改进惯性传感器人体活动识别 

**Authors**: Dominique Nshimyimana, Vitor Fortes Rey, Sungho Suh, Bo Zhou, Paul Lukowicz  

**Link**: [PDF](https://arxiv.org/pdf/2503.17978)  

**Abstract**: Human activity recognition (HAR) with deep learning models relies on large amounts of labeled data, often challenging to obtain due to associated cost, time, and labor. Self-supervised learning (SSL) has emerged as an effective approach to leverage unlabeled data through pretext tasks, such as masked reconstruction and multitask learning with signal processing-based data augmentations, to pre-train encoder models. However, such methods are often derived from computer vision approaches that disregard physical mechanisms and constraints that govern wearable sensor data and the phenomena they reflect. In this paper, we propose a physics-informed multi-task pre-training (PIM) framework for IMU-based HAR. PIM generates pre-text tasks based on the understanding of basic physical aspects of human motion: including movement speed, angles of movement, and symmetry between sensor placements. Given a sensor signal, we calculate corresponding features using physics-based equations and use them as pretext tasks for SSL. This enables the model to capture fundamental physical characteristics of human activities, which is especially relevant for multi-sensor systems. Experimental evaluations on four HAR benchmark datasets demonstrate that the proposed method outperforms existing state-of-the-art methods, including data augmentation and masked reconstruction, in terms of accuracy and F1 score. We have observed gains of almost 10\% in macro f1 score and accuracy with only 2 to 8 labeled examples per class and up to 3% when there is no reduction in the amount of training data. 

**Abstract (ZH)**: 基于物理约束的多任务预训练框架（PIM）用于惯性传感器的人体活动识别 

---
# Dynamic Gradient Sparse Update for Edge Training 

**Title (ZH)**: 边缘训练中的动态梯度稀疏更新 

**Authors**: I-Hsuan Li, Tian-Sheuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17959)  

**Abstract**: Training on edge devices enables personalized model fine-tuning to enhance real-world performance and maintain data privacy. However, the gradient computation for backpropagation in the training requires significant memory buffers to store intermediate features and compute losses. This is unacceptable for memory-constrained edge devices such as microcontrollers. To tackle this issue, we propose a training acceleration method using dynamic gradient sparse updates. This method updates the important channels and layers only and skips gradient computation for the less important channels and layers to reduce memory usage for each update iteration. In addition, the channel selection is dynamic for different iterations to traverse most of the parameters in the update layers along the time dimension for better performance. The experimental result shows that the proposed method enables an ImageNet pre-trained MobileNetV2 trained on CIFAR-10 to achieve an accuracy of 85.77\% while updating only 2\% of convolution weights within 256KB on-chip memory. This results in a remarkable 98\% reduction in feature memory usage compared to dense model training. 

**Abstract (ZH)**: 在边缘设备上训练通过动态梯度稀疏更新实现个性化的模型微调，以增强实际性能并保持数据隐私。然而，训练中的反向传播梯度计算需要大量内存缓冲区来存储中间特征和计算损失函数。对于内存受限的边缘设备如微控制器来说，这是无法接受的。为了解决这个问题，我们提出了一种使用动态梯度稀疏更新的训练加速方法。该方法仅更新重要的通道和层，并跳过不重要的通道和层的梯度计算，以减少每次更新迭代的内存使用量。此外，通道选择在不同的迭代中是动态的，沿着时间维度遍历更新层中的大多数参数以获得更好的性能。实验结果表明，所提出的方法使在CIFAR-10上预训练的ImageNet先验MobileNetV2在256KB片内内存中仅更新2%的卷积权重时实现了85.77%的精度。与密集模型训练相比，这导致了特征内存使用量减少了98%。 

---
# Human-AI Interaction and User Satisfaction: Empirical Evidence from Online Reviews of AI Products 

**Title (ZH)**: 人类-人工智能交互与用户满意度：来自AI产品在线评价的实证证据 

**Authors**: Stefan Pasch, Sun-Young Ha  

**Link**: [PDF](https://arxiv.org/pdf/2503.17955)  

**Abstract**: Human-AI Interaction (HAI) guidelines and design principles have become increasingly important in both industry and academia to guide the development of AI systems that align with user needs and expectations. However, large-scale empirical evidence on how HAI principles shape user satisfaction in practice remains limited. This study addresses that gap by analyzing over 100,000 user reviews of AI-related products from this http URL, a leading review platform for business software and services. Based on widely adopted industry guidelines, we identify seven core HAI dimensions and examine their coverage and sentiment within the reviews. We find that the sentiment on four HAI dimensions-adaptability, customization, error recovery, and security-is positively associated with overall user satisfaction. Moreover, we show that engagement with HAI dimensions varies by professional background: Users with technical job roles are more likely to discuss system-focused aspects, such as reliability, while non-technical users emphasize interaction-focused features like customization and feedback. Interestingly, the relationship between HAI sentiment and overall satisfaction is not moderated by job role, suggesting that once an HAI dimension has been identified by users, its effect on satisfaction is consistent across job roles. 

**Abstract (ZH)**: 基于用户评价的大规模实证分析：理解人机交互原则对用户满意度的影响 

---
# Physics-Guided Multi-Fidelity DeepONet for Data-Efficient Flow Field Prediction 

**Title (ZH)**: 基于物理引导的多保真度DeepONet在高效流场预测中的应用 

**Authors**: Sunwoong Yang, Youngkyu Lee, Namwoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17941)  

**Abstract**: This study presents an enhanced multi-fidelity deep operator network (DeepONet) framework for efficient spatio-temporal flow field prediction, with particular emphasis on practical scenarios where high-fidelity data is scarce. We introduce several key innovations to improve the framework's efficiency and accuracy. First, we enhance the DeepONet architecture by incorporating a merge network that enables more complex feature interactions between operator and coordinate spaces, achieving a 50.4% reduction in prediction error compared to traditional dot-product operations. We further optimize the architecture through temporal positional encoding and point-based sampling strategies, achieving a 7.57% improvement in prediction accuracy while reducing training time by 96% through efficient sampling and automatic mixed precision training. Building upon this foundation, we develop a transfer learning-based multi-fidelity framework that leverages knowledge from pre-trained low-fidelity models to guide high-fidelity predictions. Our approach freezes the pre-trained branch and trunk networks while making only the merge network trainable during high-fidelity training, preserving valuable low-fidelity representations while efficiently adapting to high-fidelity features. Through systematic investigation, we demonstrate that this fine-tuning strategy not only significantly outperforms linear probing and full-tuning alternatives but also surpasses conventional multi-fidelity frameworks by up to 76%, while achieving up to 43.7% improvement in prediction accuracy compared to single-fidelity training. The core contribution lies in our novel time-derivative guided sampling approach: it maintains prediction accuracy equivalent to models trained with the full dataset while requiring only 60% of the original high-fidelity samples. 

**Abstract (ZH)**: 本研究提出了一种增强的多保真度深操作网络（DeepONet）框架，用于高效的时空流场预测，特别是在高保真数据稀缺的实际场景中。我们引入了几项关键创新来提高框架的效率和精度。首先，通过引入合并网络，增强了DeepONet架构，使其能够在操作空间和坐标空间之间实现更复杂的特征交互，相比传统点积操作，预测误差降低了50.4%。我们进一步通过时间位置编码和基于点的采样策略优化了架构，在保持84%的训练时间同时，通过高效的采样和自动混合精度训练，预测精度提高了7.57%。在此基础上，我们开发了一种基于迁移学习的多保真度框架，利用预训练的低保真模型知识指导高保真预测。我们的方法在高保真训练时仅使合并网络可训练，冻结预训练的分支网络和主干网络，从而保留有价值的低保真表示并高效适应高保真特征。通过系统研究，我们证明了这种微调策略不仅显著优于线性探针和全微调的替代方案，也超越了传统的多保真度框架，最多提高了76%，同时在预测精度上相比单保真训练提高了43.7%。核心贡献在于我们提出的一种新型时间导数引导的采样方法：它在保留与全数据集训练模型相当的预测精度的同时，仅需要原始高保真样本的60%。 

---
# GLADMamba: Unsupervised Graph-Level Anomaly Detection Powered by Selective State Space Model 

**Title (ZH)**: GLADMamba：由选择性状态空间模型驱动的无监督图级异常检测 

**Authors**: Yali Fu, Jindong Li, Qi Wang, Qianli Xing  

**Link**: [PDF](https://arxiv.org/pdf/2503.17903)  

**Abstract**: Unsupervised graph-level anomaly detection (UGLAD) is a critical and challenging task across various domains, such as social network analysis, anti-cancer drug discovery, and toxic molecule identification. However, existing methods often struggle to capture the long-range dependencies efficiently and neglect the spectral information. Recently, selective State Space Models (SSMs), particularly Mamba, have demonstrated remarkable advantages in capturing long-range dependencies with linear complexity and a selection mechanism. Motivated by their success across various domains, we propose GLADMamba, a novel framework that adapts the selective state space model into UGLAD field. We design View-Fused Mamba (VFM) with a Mamba-Transformer-style architecture to efficiently fuse information from different views with a selective state mechanism. We also design Spectrum-Guided Mamba (SGM) with a Mamba-Transformer-style architecture to leverage the Rayleigh quotient to guide the embedding refining process. GLADMamba can dynamically focus on anomaly-related information while discarding irrelevant information for anomaly detection. To the best of our knowledge, this is the first work to introduce Mamba and explicit spectral information to UGLAD. Extensive experiments on 12 real-world datasets demonstrate that GLADMamba outperforms existing state-of-the-art methods, achieving superior performance in UGLAD. The code is available at this https URL. 

**Abstract (ZH)**: 无监督图级异常检测（UGLAD）是社交网络分析、抗癌药物发现和毒物分子识别等诸多领域的一个关键且具有挑战性的任务。然而，现有方法往往难以高效捕捉长期依赖关系，并忽视了光谱信息。近期，选择性状态空间模型（SSMs），特别是Mamba，已经在线性复杂度和选择机制下展示了捕获长期依赖关系的显著优势。受其在各领域成功应用的启发，我们提出GLADMamba框架，将其适应到UGLAD领域。我们设计了具有Mamba-Transformer风格架构的视图融合Mamba（VFM），以选择机制高效融合不同视图的信息。同时，我们设计了具有Mamba-Transformer风格架构的光谱引导Mamba（SGM），利用瑞利商指导嵌入精炼过程。GLADMamba能够动态关注与异常相关的信息，同时剔除无关信息进行异常检测。据我们所知，这是首次将Mamba和显式光谱信息引入UGLAD的研究。广泛实验在12个真实世界数据集上的结果显示，GLADMamba优于现有最先进的方法，在UGLAD中取得了 superior 的性能。代码可在以下链接获取。 

---
# Generative AI for Validating Physics Laws 

**Title (ZH)**: 生成式AI在验证物理定律中的应用 

**Authors**: Maria Nareklishvili, Nicholas Polson, Vadim Sokolov  

**Link**: [PDF](https://arxiv.org/pdf/2503.17894)  

**Abstract**: We present generative artificial intelligence (AI) to empirically validate fundamental laws of physics, focusing on the Stefan-Boltzmann law linking stellar temperature and luminosity. Our approach simulates counterfactual luminosities under hypothetical temperature regimes for each individual star and iteratively refines the temperature-luminosity relationship in a deep learning architecture. We use Gaia DR3 data and find that, on average, temperature's effect on luminosity increases with stellar radius and decreases with absolute magnitude, consistent with theoretical predictions. By framing physics laws as causal problems, our method offers a novel, data-driven approach to refine theoretical understanding and inform evidence-based policy and practice. 

**Abstract (ZH)**: 我们展示了生成式人工智能（AI）在经验验证物理学基本定律方面的应用，聚焦于将恒星温度与亮度关联起来的斯蒂芬-玻尔兹曼定律。我们的方法模拟了在每颗恒星假定温度条件下对应的非现实亮度，并在深度学习架构中逐步 refine 温度-亮度关系。我们使用盖亚 DR3 数据发现，平均而言，温度对亮度的影响随恒星半径增加而增强，随绝对星等增加而减弱，这与理论预测一致。通过将物理定律表述为因果问题，我们的方法提供了一种新的数据驱动方法，以 refinement 理论理解并指导基于证据的政策和实践。 

---
# Detecting and Mitigating DDoS Attacks with AI: A Survey 

**Title (ZH)**: 使用AI检测和缓解DDoS攻击：一个综述 

**Authors**: Alexandru Apostu, Silviu Gheorghe, Andrei Hîji, Nicolae Cleju, Andrei Pătraşcu, Cristian Rusu, Radu Ionescu, Paul Irofti  

**Link**: [PDF](https://arxiv.org/pdf/2503.17867)  

**Abstract**: Distributed Denial of Service attacks represent an active cybersecurity research problem. Recent research shifted from static rule-based defenses towards AI-based detection and mitigation. This comprehensive survey covers several key topics. Preeminently, state-of-the-art AI detection methods are discussed. An in-depth taxonomy based on manual expert hierarchies and an AI-generated dendrogram are provided, thus settling DDoS categorization ambiguities. An important discussion on available datasets follows, covering data format options and their role in training AI detection methods together with adversarial training and examples augmentation. Beyond detection, AI based mitigation techniques are surveyed as well. Finally, multiple open research directions are proposed. 

**Abstract (ZH)**: 分布式拒绝服务攻击代表了活跃的网络安全研究问题。近期研究从静态规则防御转向基于AI的检测与缓解。本文综述涵盖了多个关键主题。首先，讨论了最先进的AI检测方法。提供了基于手动专家层次结构和AI生成的谱系图的详细分类学，从而解决了DDoS分类的不确定性。接着讨论了可用的数据集，涵盖数据格式选项及其在培训AI检测方法中的作用，包括对抗训练和数据增强。除了检测之外，还综述了基于AI的缓解技术。最后，提出了多个开放的研究方向。 

---
# A Causal Adjustment Module for Debiasing Scene Graph Generation 

**Title (ZH)**: 因果调整模块用于去偏场景图生成 

**Authors**: Li Liu, Shuzhou Sun, Shuaifeng Zhi, Fan Shi, Zhen Liu, Janne Heikkilä, Yongxiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17862)  

**Abstract**: While recent debiasing methods for Scene Graph Generation (SGG) have shown impressive performance, these efforts often attribute model bias solely to the long-tail distribution of relationships, overlooking the more profound causes stemming from skewed object and object pair distributions. In this paper, we employ causal inference techniques to model the causality among these observed skewed distributions. Our insight lies in the ability of causal inference to capture the unobservable causal effects between complex distributions, which is crucial for tracing the roots of model bias. Specifically, we introduce the Mediator-based Causal Chain Model (MCCM), which, in addition to modeling causality among objects, object pairs, and relationships, incorporates mediator variables, i.e., cooccurrence distribution, for complementing the causality. Following this, we propose the Causal Adjustment Module (CAModule) to estimate the modeled causal structure, using variables from MCCM as inputs to produce a set of adjustment factors aimed at correcting biased model predictions. Moreover, our method enables the composition of zero-shot relationships, thereby enhancing the model's ability to recognize such relationships. Experiments conducted across various SGG backbones and popular benchmarks demonstrate that CAModule achieves state-of-the-art mean recall rates, with significant improvements also observed on the challenging zero-shot recall rate metric. 

**Abstract (ZH)**: 尽管最近的场景图生成（SGG）去偏见方法展现了令人印象深刻的性能，但这些努力往往将模型偏见仅归因于关系分布的长尾效应，而忽视了来自对象和对象对分布偏差的更深层次原因。本文采用因果推理技术建模这些观察到的偏斜分布之间的因果关系。我们的见解在于因果推理能够捕捉复杂分布之间的不可观察因果效应，这对于追踪模型偏见的根源至关重要。具体而言，我们引入了中介因子基础因果链模型（MCCM），该模型除了建模对象、对象对和关系之间的因果关系外，还引入了中介变量，即共现分布，以补充因果关系。随后，我们提出因果调整模块（CAModule），使用MCCM中的变量作为输入，生成一组调整因子，旨在纠正偏差的模型预测。此外，我们的方法能够合成零样本关系，从而增强模型识别此类关系的能力。在多种SGG基础架构和流行基准上的实验表明，CAModule实现了最先进的平均召回率，同时在具有挑战性的零样本召回率指标上也观察到显著提高。 

---
# Adapt, Agree, Aggregate: Semi-Supervised Ensemble Labeling for Graph Convolutional Networks 

**Title (ZH)**: 适应、共识、聚合：图卷积网络的半监督集成标签标注 

**Authors**: Maryam Abdolali, Romina Zakerian, Behnam Roshanfekr, Fardin Ayar, Mohammad Rahmati  

**Link**: [PDF](https://arxiv.org/pdf/2503.17842)  

**Abstract**: In this paper, we propose a novel framework that combines ensemble learning with augmented graph structures to improve the performance and robustness of semi-supervised node classification in graphs. By creating multiple augmented views of the same graph, our approach harnesses the "wisdom of a diverse crowd", mitigating the challenges posed by noisy graph structures. Leveraging ensemble learning allows us to simultaneously achieve three key goals: adaptive confidence threshold selection based on model agreement, dynamic determination of the number of high-confidence samples for training, and robust extraction of pseudo-labels to mitigate confirmation bias. Our approach uniquely integrates adaptive ensemble consensus to flexibly guide pseudo-label extraction and sample selection, reducing the risks of error accumulation and improving robustness. Furthermore, the use of ensemble-driven consensus for pseudo-labeling captures subtle patterns that individual models often overlook, enabling the model to generalize better. Experiments on several real-world datasets demonstrate the effectiveness of our proposed method. 

**Abstract (ZH)**: 本文提出了一种结合集成学习和增强图结构的新框架，以提高图上半监督节点分类性能和鲁棒性。通过创建同一个图的多个增强视图，我们的方法利用了“多样群体的智慧”，缓解了嘈杂图结构带来的挑战。结合集成学习使我们能够同时实现三个关键目标：基于模型一致性自适应选择置信阈值、动态确定用于训练的高置信度样本数量以及稳健地提取伪标签以减轻确认偏见。我们的方法独特地将自适应集成共识灵活地引导伪标签提取和样本选择，减少了错误累积的风险并提高了鲁棒性。此外，集成驱动的共识用于伪标签化捕获了单个模型往往忽视的微妙模式，使模型能够更好地泛化。实验结果表明，所提出的方法在多个真实世界数据集上是有效的。 

---
# A Roadmap Towards Improving Multi-Agent Reinforcement Learning With Causal Discovery And Inference 

**Title (ZH)**: 面向因果发现与推理的多智能体强化学习改进之路 

**Authors**: Giovanni Briglia, Stefano Mariani, Franco Zambonelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.17803)  

**Abstract**: Causal reasoning is increasingly used in Reinforcement Learning (RL) to improve the learning process in several dimensions: efficacy of learned policies, efficiency of convergence, generalisation capabilities, safety and interpretability of behaviour. However, applications of causal reasoning to Multi-Agent RL (MARL) are still mostly unexplored. In this paper, we take the first step in investigating the opportunities and challenges of applying causal reasoning in MARL. We measure the impact of a simple form of causal augmentation in state-of-the-art MARL scenarios increasingly requiring cooperation, and with state-of-the-art MARL algorithms exploiting various degrees of collaboration between agents. Then, we discuss the positive as well as negative results achieved, giving us the chance to outline the areas where further research may help to successfully transfer causal RL to the multi-agent setting. 

**Abstract (ZH)**: 因果推理在多Agent强化学习中的应用：机遇与挑战 

---
# CODA: Repurposing Continuous VAEs for Discrete Tokenization 

**Title (ZH)**: CODA: 重新利用连续VAEs进行离散标记化 

**Authors**: Zeyu Liu, Zanlin Ni, Yeguo Hua, Xin Deng, Xiao Ma, Cheng Zhong, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17760)  

**Abstract**: Discrete visual tokenizers transform images into a sequence of tokens, enabling token-based visual generation akin to language models. However, this process is inherently challenging, as it requires both compressing visual signals into a compact representation and discretizing them into a fixed set of codes. Traditional discrete tokenizers typically learn the two tasks jointly, often leading to unstable training, low codebook utilization, and limited reconstruction quality. In this paper, we introduce \textbf{CODA}(\textbf{CO}ntinuous-to-\textbf{D}iscrete \textbf{A}daptation), a framework that decouples compression and discretization. Instead of training discrete tokenizers from scratch, CODA adapts off-the-shelf continuous VAEs -- already optimized for perceptual compression -- into discrete tokenizers via a carefully designed discretization process. By primarily focusing on discretization, CODA ensures stable and efficient training while retaining the strong visual fidelity of continuous VAEs. Empirically, with $\mathbf{6 \times}$ less training budget than standard VQGAN, our approach achieves a remarkable codebook utilization of 100% and notable reconstruction FID (rFID) of $\mathbf{0.43}$ and $\mathbf{1.34}$ for $8 \times$ and $16 \times$ compression on ImageNet 256$\times$ 256 benchmark. 

**Abstract (ZH)**: 连续到离散适应（CODA：Continuous-to-Discrete Adaptation） 

---
# Bandwidth Reservation for Time-Critical Vehicular Applications: A Multi-Operator Environment 

**Title (ZH)**: 时间关键型车载应用的带宽预留：多运营商环境 

**Authors**: Abdullah Al-Khatib, Abdullah Ahmed, Klaus Moessner, Holger Timinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.17756)  

**Abstract**: Onsite bandwidth reservation requests often face challenges such as price fluctuations and fairness issues due to unpredictable bandwidth availability and stringent latency requirements. Requesting bandwidth in advance can mitigate the impact of these fluctuations and ensure timely access to critical resources. In a multi-Mobile Network Operator (MNO) environment, vehicles need to select cost-effective and reliable resources for their safety-critical applications. This research aims to minimize resource costs by finding the best price among multiple MNOs. It formulates multi-operator scenarios as a Markov Decision Process (MDP), utilizing a Deep Reinforcement Learning (DRL) algorithm, specifically Dueling Deep Q-Learning. For efficient and stable learning, we propose a novel area-wise approach and an adaptive MDP synthetic close to the real environment. The Temporal Fusion Transformer (TFT) is used to handle time-dependent data and model training. Furthermore, the research leverages Amazon spot price data and adopts a multi-phase training approach, involving initial training on synthetic data, followed by real-world data. These phases enable the DRL agent to make informed decisions using insights from historical data and real-time observations. The results show that our model leads to significant cost reductions, up to 40%, compared to scenarios without a policy model in such a complex environment. 

**Abstract (ZH)**: 基于现场的带宽预留请求常常面临价格波动和公平性问题，这源于带宽可用性的不可预测性和严格的延迟要求。提前预留带宽可以缓解这些波动的影响，确保及时访问关键资源。在多移动网络运营商(MNO)环境中，车辆需要选择成本效益高且可靠的资源以支持其安全关键应用。本研究旨在通过在多家MNO中寻找最优价格来最小化资源成本。该研究将多运营商场景建模为马尔可夫决策过程(MDP)，并利用深度强化学习(DRL)算法，具体采用对偶深度Q学习(Dueling Deep Q-Learning)。为了实现高效的稳定学习，我们提出了一种新颖的区域化方法和一种接近真实环境的自适应MDP。使用时间融合变换器(TFT)处理时间相关数据并进行模型训练。此外，本研究利用亚马逊的即时价格数据，并采用多阶段训练方法，首先是使用合成数据进行初始训练，随后使用真实数据进行训练。这些阶段使DRL代理能够利用历史数据洞察和实时观察做出明智的决策。结果显示，与没有政策模型的场景相比，我们的模型在这样复杂的环境中可实现高达40%的成本节约。 

---
# Towards Invisible Backdoor Attack on Text-to-Image Diffusion Model 

**Title (ZH)**: 面向文本到图像扩散模型的隐形后门攻击 

**Authors**: Jie Zhang, Zhongqi Wang, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17724)  

**Abstract**: Backdoor attacks targeting text-to-image diffusion models have advanced rapidly, enabling attackers to implant malicious triggers into these models to manipulate their outputs. However, current backdoor samples often exhibit two key abnormalities compared to benign samples: 1) Semantic Consistency, where backdoor prompts tend to generate images with similar semantic content even with significant textual variations to the prompts; 2) Attention Consistency, where the trigger induces consistent structural responses in the cross-attention maps. These consistencies leave detectable traces for defenders, making backdoors easier to identify. To enhance the stealthiness of backdoor samples, we propose a novel Invisible Backdoor Attack (IBA) by explicitly mitigating these consistencies. Specifically, our approach leverages syntactic structures as backdoor triggers to amplify the sensitivity to textual variations, effectively breaking down the semantic consistency. Besides, a regularization method based on Kernel Maximum Mean Discrepancy (KMMD) is proposed to align the distribution of cross-attention responses between backdoor and benign samples, thereby disrupting attention consistency. Extensive experiments demonstrate that our IBA achieves a 97.5% attack success rate while exhibiting stronger resistance to defenses, with an average of over 98% backdoor samples bypassing three state-of-the-art detection mechanisms. The code is available at this https URL. 

**Abstract (ZH)**: 针对文本到图像扩散模型的后门攻击已快速发展，使攻击者能够将恶意触发器植入这些模型以操控其输出。然而，当前的后门样本与良性样本相比通常表现出两种关键异常：1）语义一致性，后门提示即使在文本提示有显著变化的情况下，仍倾向于生成具有相似语义内容的图像；2）注意力一致性，触发器在交叉注意力图中诱导一致的结构响应。这些一致性为防御者留下了可检测的痕迹，使后门更容易被识别。为了增强后门样本的隐匿性，我们提出了一种新型隐形后门攻击（IBA），通过显式地减轻这些一致性。具体来说，我们的方法利用句法结构作为后门触发器，增强对文本变化的敏感性，从而打破语义一致性。此外，我们还提出了一种基于核最大均值差异（KMMD）的正则化方法，以使后门样本和良性样本的交叉注意力响应分布保持一致，从而破坏注意力一致性。广泛实验表明，我们的IBA在攻击成功率上达到97.5%，同时表现出更强的防御抵抗性，平均超过98%的后门样本能够绕过三种最先进的检测机制。代码可在以下链接获取。 

---
# GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration 

**Title (ZH)**: GUI-Xplore：通过一次探索赋予通用GUI代理泛化能力 

**Authors**: Yuchen Sun, Shanhui Zhao, Tao Yu, Hao Wen, Samith Va, Mengwei Xu, Yuanchun Li, Chongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17709)  

**Abstract**: GUI agents hold significant potential to enhance the experience and efficiency of human-device interaction. However, current methods face challenges in generalizing across applications (apps) and tasks, primarily due to two fundamental limitations in existing datasets. First, these datasets overlook developer-induced structural variations among apps, limiting the transferability of knowledge across diverse software environments. Second, many of them focus solely on navigation tasks, which restricts their capacity to represent comprehensive software architectures and complex user interactions. To address these challenges, we introduce GUI-Xplore, a dataset meticulously designed to enhance cross-application and cross-task generalization via an exploration-and-reasoning framework. GUI-Xplore integrates pre-recorded exploration videos providing contextual insights, alongside five hierarchically structured downstream tasks designed to comprehensively evaluate GUI agent capabilities. To fully exploit GUI-Xplore's unique features, we propose Xplore-Agent, a GUI agent framework that combines Action-aware GUI Modeling with Graph-Guided Environment Reasoning. Further experiments indicate that Xplore-Agent achieves a 10% improvement over existing methods in unfamiliar environments, yet there remains significant potential for further enhancement towards truly generalizable GUI agents. 

**Abstract (ZH)**: GUI代理在提升人机交互体验和效率方面拥有巨大潜力，但当前方法在跨应用和任务的一般化方面面临挑战，主要是由于现有数据集的两个基本限制。首先，这些数据集忽视了开发者引入的应用结构差异，限制了在不同软件环境中的知识迁移。其次，许多数据集仅专注于导航任务，限制了其代表全面软件架构和复杂用户交互的能力。为应对这些挑战，我们引入了GUI-Xplore数据集，该数据集通过探索与推理框架精心设计，旨在增强跨应用和跨任务的一般化能力。GUI-Xplore集成了预先录制的探索视频和五个层次化的下游任务，以全面评估GUI代理的能力。为进一步充分利用GUI-Xplore的独特功能，我们提出了一种结合行动意识GUI建模与图引导环境推理的GUI代理框架Xplore-Agent。进一步的实验表明，Xplore-Agent在陌生环境中比现有方法提高了10%的表现，但仍存在向真正可泛化的GUI代理进一步提升的巨大潜力。 

---
# PT-PINNs: A Parametric Engineering Turbulence Solver based on Physics-Informed Neural Networks 

**Title (ZH)**: 基于物理知情神经网络的参数化工程湍流求解器 PT-PINNs 

**Authors**: Liang Jiang, Yuzhou Cheng, Kun Luo, Jianren Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17704)  

**Abstract**: Physics-informed neural networks (PINNs) demonstrate promising potential in parameterized engineering turbulence optimization problems but face challenges, such as high data requirements and low computational accuracy when applied to engineering turbulence problems. This study proposes a framework that enhances the ability of PINNs to solve parametric turbulence problems without training datasets from experiments or CFD-Parametric Turbulence PINNs (PT-PINNs)). Two key methods are introduced to improve the accuracy and robustness of this framework. The first is a soft constraint method for turbulent viscosity calculation. The second is a pre-training method based on the conservation of flow rate in the flow field. The effectiveness of PT-PINNs is validated using a three-dimensional backward-facing step (BFS) turbulence problem with two varying parameters (Re = 3000-200000, ER = 1.1-1.5). PT-PINNs produce predictions that closely match experimental data and computational fluid dynamics (CFD) results across various conditions. Moreover, PT-PINNs offer a computational efficiency advantage over traditional CFD methods. The total time required to construct the parametric BFS turbulence model is 39 hours, one-sixteenth of the time required by traditional numerical methods. The inference time for a single-condition prediction is just 40 seconds-only 0.5% of a single CFD computation. These findings highlight the potential of PT-PINNs for future applications in engineering turbulence optimization problems. 

**Abstract (ZH)**: 基于物理的神经网络（PT-PINNs）在工程湍流优化问题中的参数化建模与应用 

---
# On the (im)possibility of sustainable artificial intelligence. Why it does not make sense to move faster when heading the wrong way 

**Title (ZH)**: 关于可持续人工智能的可能性（或不可能性）。为什么朝错误的方向前进时加快速度没有意义 

**Authors**: Rainer Rehak  

**Link**: [PDF](https://arxiv.org/pdf/2503.17702)  

**Abstract**: Artificial intelligence (AI) is currently considered a sustainability "game-changer" within and outside of academia. In order to discuss sustainable AI this article draws from insights by critical data and algorithm studies, STS, transformative sustainability science, critical computer science, and public interest theory. I argue that while there are indeed many sustainability-related use cases for AI, they are likely to have more overall drawbacks than benefits. To substantiate this claim, I differentiate three 'AI materialities' of the AI supply chain: first the literal materiality (e.g. water, cobalt, lithium, energy consumption etc.), second, the informational materiality (e.g. lots of data and centralised control necessary), and third, the social materiality (e.g. exploitative data work, communities harm by waste and pollution). In all materialities, effects are especially devastating for the global south while benefiting the global north. A second strong claim regarding sustainable AI circles around so called apolitical optimisation (e.g. regarding city traffic), however the optimisation criteria (e.g. cars, bikes, emissions, commute time, health) are purely political and have to be collectively negotiated before applying AI optimisation. Hence, sustainable AI, in principle, cannot break the glass ceiling of transformation and might even distract from necessary societal change. To address that I propose to stop 'unformation gathering' and to apply the 'small is beautiful' principle. This aims to contribute to an informed academic and collective negotiation on how to (not) integrate AI into the sustainability project while avoiding to reproduce the status quo by serving hegemonic interests between useful AI use cases, techno-utopian salvation narratives, technology-centred efficiency paradigms, the exploitative and extractivist character of AI and concepts of digital degrowth. 

**Abstract (ZH)**: 人工智能（AI）currently considered a sustainability “游戏规则改变者” 在学术界内外目前被认为是对可持续性产生“游戏规则改变者”影响的技术。为了讨论可持续AI，本文借鉴了批判性数据与算法研究、STS、转变性可持续科学、批判性计算机科学以及公共利益理论中的洞见。我主张，尽管人工智能确实有许多与可持续性相关的应用场景，但它们的整体弊端可能多于益处。为证明这一论点，我区分了AI供应链中的三种“AI物质性”：首先是字面意义上的物质性（例如水资源、钴、锂、能源消耗等），其次是信息性的物质性（例如大量数据和集中控制的必要性），再次是社会性的物质性（例如剥削性的数据劳动、社区因废物和污染而受损）。在所有物质性方面，其影响对全球南半球尤其具有毁灭性，而对全球北半球则有所益处。关于可持续AI的第二个重要论点围绕所谓的无政治色彩优化（例如城市交通优化），然而，优化标准（例如汽车、自行车、排放、通勤时间、健康等）都是纯粹的政治性问题，必须在应用AI优化之前由集体协商达成一致。因此，原则上，可持续AI无法突破转型的天花板，甚至可能分散对必要社会变革的注意力。为应对这一挑战，我建议停止“无结构信息收集”并应用“小即是美”的原则。这旨在促成对如何（不）将人工智能融入可持续性项目进行知情的学术界和集体协商，以避免通过有用的人工智能应用场景、 techno-乌托邦救赎叙事、技术为中心的效率观念、人工智能的剥削性与提取主义特征及数字减缩概念来复制现状。 

---
# ComfyGPT: A Self-Optimizing Multi-Agent System for Comprehensive ComfyUI Workflow Generation 

**Title (ZH)**: ComfyGPT：一个自我优化的多代理系统，用于全面的ComfyUI工作流生成 

**Authors**: Oucheng Huang, Yuhang Ma, Zeng Zhao, Mingrui Wu, Jiayi Ji, Rongsheng Zhang, Zhipeng Hu, Xiaoshuai Sun, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.17671)  

**Abstract**: ComfyUI provides a widely-adopted, workflow-based interface that enables users to customize various image generation tasks through an intuitive node-based architecture. However, the intricate connections between nodes and diverse modules often present a steep learning curve for users. In this paper, we introduce ComfyGPT, the first self-optimizing multi-agent system designed to generate ComfyUI workflows based on task descriptions automatically. ComfyGPT comprises four specialized agents: ReformatAgent, FlowAgent, RefineAgent, and ExecuteAgent. The core innovation of ComfyGPT lies in two key aspects. First, it focuses on generating individual node links rather than entire workflows, significantly improving generation precision. Second, we proposed FlowAgent, a LLM-based workflow generation agent that uses both supervised fine-tuning (SFT) and reinforcement learning (RL) to improve workflow generation accuracy. Moreover, we introduce FlowDataset, a large-scale dataset containing 13,571 workflow-description pairs, and FlowBench, a comprehensive benchmark for evaluating workflow generation systems. We also propose four novel evaluation metrics: Format Validation (FV), Pass Accuracy (PA), Pass Instruct Alignment (PIA), and Pass Node Diversity (PND). Experimental results demonstrate that ComfyGPT significantly outperforms existing LLM-based methods in workflow generation. 

**Abstract (ZH)**: ComfyGPT：一种基于任务描述自优化的多智能体系统，用于生成ComfyUI工作流 

---
# A Qualitative Study of User Perception of M365 AI Copilot 

**Title (ZH)**: M365 AI 导航员的用户感知 qualitative 研究 

**Authors**: Muneera Bano, Didar Zowghi, Jon Whittle, Liming Zhu, Andrew Reeson, Rob Martin, Jen Parson  

**Link**: [PDF](https://arxiv.org/pdf/2503.17661)  

**Abstract**: Adopting AI copilots in professional workflows presents opportunities for enhanced productivity, efficiency, and decision making. In this paper, we present results from a six month trial of M365 Copilot conducted at our organisation in 2024. A qualitative interview study was carried out with 27 participants. The study explored user perceptions of M365 Copilot's effectiveness, productivity impact, evolving expectations, ethical concerns, and overall satisfaction. Initial enthusiasm for the tool was met with mixed post trial experiences. While some users found M365 Copilot beneficial for tasks such as email coaching, meeting summaries, and content retrieval, others reported unmet expectations in areas requiring deeper contextual understanding, reasoning, and integration with existing workflows. Ethical concerns were a recurring theme, with users highlighting issues related to data privacy, transparency, and AI bias. While M365 Copilot demonstrated value in specific operational areas, its broader impact remained constrained by usability limitations and the need for human oversight to validate AI generated outputs. 

**Abstract (ZH)**: 采用AI副驾在专业工作流程中的应用提供了增强生产力、效率和决策的机会。本文报告了我们在2024年进行的为期六个月的M365 Copilot试用研究结果。研究通过定量访谈对27名参与者进行了调查，探讨了用户对M365 Copilot有效性的看法、生产力影响、期望变化、伦理问题以及总体满意度。初步对工具的热情在试用后体验中变得复杂多变。一些用户发现M365 Copilot在邮件指导、会议摘要和内容检索等方面有益，而另一些用户则在需要更深层次上下文理解、推理和与现有工作流程集成的领域报告了期望未得到满足的情况。伦理问题是一个反复出现的主题，用户指出了与数据隐私、透明度和AI偏见相关的问题。尽管M365 Copilot在特定运营领域显示出价值，但其更广泛的影响仍受到易用性限制和需要人类监督验证AI生成输出的需求的制约。 

---
# NaFM: Pre-training a Foundation Model for Small-Molecule Natural Products 

**Title (ZH)**: NaFM：预训练一个小分子天然产物基础模型 

**Authors**: Yuheng Ding, Yusong Wang, Bo Qiang, Jie Yu, Qi Li, Yiran Zhou, Zhenmin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17656)  

**Abstract**: Natural products, as metabolites from microorganisms, animals, or plants, exhibit diverse biological activities, making them crucial for drug discovery. Nowadays, existing deep learning methods for natural products research primarily rely on supervised learning approaches designed for specific downstream tasks. However, such one-model-for-a-task paradigm often lacks generalizability and leaves significant room for performance improvement. Additionally, existing molecular characterization methods are not well-suited for the unique tasks associated with natural products. To address these limitations, we have pre-trained a foundation model for natural products based on their unique properties. Our approach employs a novel pretraining strategy that is especially tailored to natural products. By incorporating contrastive learning and masked graph learning objectives, we emphasize evolutional information from molecular scaffolds while capturing side-chain information. Our framework achieves state-of-the-art (SOTA) results in various downstream tasks related to natural product mining and drug discovery. We first compare taxonomy classification with synthesized molecule-focused baselines to demonstrate that current models are inadequate for understanding natural synthesis. Furthermore, by diving into a fine-grained analysis at both the gene and microbial levels, NaFM demonstrates the ability to capture evolutionary information. Eventually, our method is experimented with virtual screening, illustrating informative natural product representations that can lead to more effective identification of potential drug candidates. 

**Abstract (ZH)**: 基于自然产物的独特性质的预训练模型研究：结合对比学习和掩蔽图学习以实现药物发现下游任务的最先进的性能 

---
# FairFlow: Mitigating Dataset Biases through Undecided Learning 

**Title (ZH)**: FairFlow：通过未决学习减轻数据集偏见 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2503.17632)  

**Abstract**: Language models are prone to dataset biases, known as shortcuts and spurious correlations in data, which often result in performance drop on new data. We present a new debiasing framework called ``FairFlow'' that mitigates dataset biases by learning to be undecided in its predictions for data samples or representations associated with known or unknown biases. The framework introduces two key components: a suite of data and model perturbation operations that generate different biased views of input samples, and a contrastive objective that learns debiased and robust representations from the resulting biased views of samples. Experiments show that FairFlow outperforms existing debiasing methods, particularly against out-of-domain and hard test samples without compromising the in-domain performance 

**Abstract (ZH)**: FairFlow：通过学习在包含已知或未知偏差的数据样本或表示上保持犹豫来缓解数据集偏差的新框架 

---
# AI-Based Screening for Depression and Social Anxiety Through Eye Tracking: An Exploratory Study 

**Title (ZH)**: 基于眼动追踪的AI辅助抑郁和社交焦虑筛查：一项探索性研究 

**Authors**: Karol Chlasta, Katarzyna Wisiecka, Krzysztof Krejtz, Izabela Krejtz  

**Link**: [PDF](https://arxiv.org/pdf/2503.17625)  

**Abstract**: Well-being is a dynamic construct that evolves over time and fluctuates within individuals, presenting challenges for accurate quantification. Reduced well-being is often linked to depression or anxiety disorders, which are characterised by biases in visual attention towards specific stimuli, such as human faces. This paper introduces a novel approach to AI-assisted screening of affective disorders by analysing visual attention scan paths using convolutional neural networks (CNNs). Data were collected from two studies examining (1) attentional tendencies in individuals diagnosed with major depression and (2) social anxiety. These data were processed using residual CNNs through images generated from eye-gaze patterns. Experimental results, obtained with ResNet architectures, demonstrated an average accuracy of 48% for a three-class system and 62% for a two-class system. Based on these exploratory findings, we propose that this method could be employed in rapid, ecological, and effective mental health screening systems to assess well-being through eye-tracking. 

**Abstract (ZH)**: 福祉是一个动态的构建体，随着时间的推移而演变并在个体间波动，这为准确量化带来了挑战。降低的福祉状态通常与抑郁或焦虑障碍相关，这些障碍的特征是对特定刺激，如人脸，存在视觉注意力偏差。本文介绍了一种通过卷积神经网络（CNNs）分析视觉注意力扫描路径来辅助筛查情感障碍的新方法。研究数据来自两个研究项目，分别探讨（1）重度抑郁诊断个体的注意力倾向以及（2）社交焦虑。这些数据通过眼球注视模式生成的图像使用残差CNN进行处理。使用ResNet架构的实验结果表明，对于三分类系统，平均准确率为48%，而对于二分类系统，平均准确率为62%。基于这些探索性发现，我们提出，这种方法可以应用于快速、生态有效的精神健康筛查系统，并通过眼动追踪评估福祉。 

---
# Unraveling Pedestrian Fatality Patterns: A Comparative Study with Explainable AI 

**Title (ZH)**: 解析行人死亡模式：可解释AI的对比研究 

**Authors**: Methusela Sulle, Judith Mwakalonge, Gurcan Comert, Saidi Siuhi, Nana Kankam Gyimah  

**Link**: [PDF](https://arxiv.org/pdf/2503.17623)  

**Abstract**: Road fatalities pose significant public safety and health challenges worldwide, with pedestrians being particularly vulnerable in vehicle-pedestrian crashes due to disparities in physical and performance characteristics. This study employs explainable artificial intelligence (XAI) to identify key factors contributing to pedestrian fatalities across the five U.S. states with the highest crash rates (2018-2022). It compares them to the five states with the lowest fatality rates. Using data from the Fatality Analysis Reporting System (FARS), the study applies machine learning techniques-including Decision Trees, Gradient Boosting Trees, Random Forests, and XGBoost-to predict contributing factors to pedestrian fatalities. To address data imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) is utilized, while SHapley Additive Explanations (SHAP) values enhance model interpretability. The results indicate that age, alcohol and drug use, location, and environmental conditions are significant predictors of pedestrian fatalities. The XGBoost model outperformed others, achieving a balanced accuracy of 98 %, accuracy of 90 %, precision of 92 %, recall of 90 %, and an F1 score of 91 %. Findings reveal that pedestrian fatalities are more common in mid-block locations and areas with poor visibility, with older adults and substance-impaired individuals at higher risk. These insights can inform policymakers and urban planners in implementing targeted safety measures, such as improved lighting, enhanced pedestrian infrastructure, and stricter traffic law enforcement, to reduce fatalities and improve public safety. 

**Abstract (ZH)**: 车辆与行人碰撞导致的行人死亡事件在全球范围内对公共安全和健康构成了重大挑战，其中行人因身体和性能特征的差异，在此类碰撞中尤其脆弱。本研究采用解释性人工智能（XAI）来识别五个车辆行人事故率最高（2018-2022年）的美国州（五个州）中行人死亡的关键因素，并将其与五个行人死亡率最低的州进行对比。利用致命事故报告系统（FARS）的数据，并采用包括决策树、梯度提升树、随机森林和XGBoost在内的机器学习技术来预测行人死亡的主要因素。通过使用合成少数类过采样技术（SMOTE）来解决数据不平衡问题，同时利用SHapley加性和解释值（SHAP）提高模型的可解释性。研究结果表明，年龄、酒精和药物使用、位置和环境条件是行人死亡的重要预测因子。XGBoost模型表现最佳，实现了均衡准确率98%、准确率90%、精确率92%、召回率90%和F1分数91%。研究发现，行人死亡事件在中段位置和能见度差的地区更为常见，老年人和物质摄入影响的个体面临更高的风险。这些洞见可以指导政策制定者和城市规划者采取针对性的安全措施，如改进照明、增强行人基础设施和加强交通执法，以减少死亡人数并提高公共安全。 

---
# Measuring the Robustness of Audio Deepfake Detectors 

**Title (ZH)**: 测量音频换音鉴定器的鲁棒性 

**Authors**: Xiang Li, Pin-Yu Chen, Wenqi Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.17577)  

**Abstract**: Deepfakes have become a universal and rapidly intensifying concern of generative AI across various media types such as images, audio, and videos. Among these, audio deepfakes have been of particular concern due to the ease of high-quality voice synthesis and distribution via platforms such as social media and robocalls. Consequently, detecting audio deepfakes plays a critical role in combating the growing misuse of AI-synthesized speech. However, real-world scenarios often introduce various audio corruptions, such as noise, modification, and compression, that may significantly impact detection performance. This work systematically evaluates the robustness of 10 audio deepfake detection models against 16 common corruptions, categorized into noise perturbation, audio modification, and compression. Using both traditional deep learning models and state-of-the-art foundation models, we make four unique observations. First, our findings show that while most models demonstrate strong robustness to noise, they are notably more vulnerable to modifications and compression, especially when neural codecs are applied. Second, speech foundation models generally outperform traditional models across most scenarios, likely due to their self-supervised learning paradigm and large-scale pre-training. Third, our results show that increasing model size improves robustness, albeit with diminishing returns. Fourth, we demonstrate how targeted data augmentation during training can enhance model resilience to unseen perturbations. A case study on political speech deepfakes highlights the effectiveness of foundation models in achieving high accuracy under real-world conditions. These findings emphasize the importance of developing more robust detection frameworks to ensure reliability in practical deployment settings. 

**Abstract (ZH)**: 深度伪造已成为各种媒体类型（如图像、音频和视频）生成AI面临的普遍且迅速加剧的关切。其中，音频深度伪造特别受到关注，因为通过社交媒体和自动电话等方式可以轻松实现高质量语音的合成与传播。因此，检测音频深度伪造在应对AI合成语音的日益滥用方面起着关键作用。然而，现实场景常引入各种音频损坏，如噪声、修改和压缩，这些都可能显著影响检测性能。本工作系统地评估了10种音频深度伪造检测模型在16种常见损坏（分为噪声扰动、音频修改和压缩）下的鲁棒性。使用传统的深度学习模型和最先进的基础模型，我们做出了四个独特的观察。首先，我们的发现表明，虽然大多数模型在噪声方面表现出较强的鲁棒性，但它们对修改和压缩更为敏感，尤其是在应用神经编解码器时。其次，基础模型在大多数场景中通常优于传统模型，这可能是由于它们的自监督学习范式和大规模预训练。第三，我们的结果显示，增加模型规模可以提高鲁棒性，尽管边际效益递减。第四，我们展示了如何在训练过程中进行有针对性的数据增强以提高模型对未见过扰动的鲁棒性。针对政治演说深度伪造的案例研究突显了基础模型在实际条件下的高精度。这些发现强调了在实际部署环境中开发更稳健检测框架的重要性。 

---
# Learning Multi-Level Features with Matryoshka Sparse Autoencoders 

**Title (ZH)**: 学习多层特征的套娃稀疏自编码器 

**Authors**: Bart Bussmann, Noa Nabeshima, Adam Karvonen, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2503.17547)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as a powerful tool for interpreting neural networks by extracting the concepts represented in their activations. However, choosing the size of the SAE dictionary (i.e. number of learned concepts) creates a tension: as dictionary size increases to capture more relevant concepts, sparsity incentivizes features to be split or absorbed into more specific features, leaving high-level features missing or warped. We introduce Matryoshka SAEs, a novel variant that addresses these issues by simultaneously training multiple nested dictionaries of increasing size, forcing the smaller dictionaries to independently reconstruct the inputs without using the larger dictionaries. This organizes features hierarchically - the smaller dictionaries learn general concepts, while the larger dictionaries learn more specific concepts, without incentive to absorb the high-level features. We train Matryoshka SAEs on Gemma-2-2B and TinyStories and find superior performance on sparse probing and targeted concept erasure tasks, more disentangled concept representations, and reduced feature absorption. While there is a minor tradeoff with reconstruction performance, we believe Matryoshka SAEs are a superior alternative for practical tasks, as they enable training arbitrarily large SAEs while retaining interpretable features at different levels of abstraction. 

**Abstract (ZH)**: 嵌套稀疏自编码器：一种解决稀疏自编码器问题的新变体 

---
# A Predictive Services Architecture for Efficient Airspace Operations 

**Title (ZH)**: 一种用于高效 airspace 运行的预测性服务架构 

**Authors**: Ítalo Romani de Oliveira, Samet Ayhan, Glaucia Balvedi, Michael Biglin, Pablo Costas, Euclides C. Pinto Neto, Alexandre Leite, Felipe C. F. de Azevedo  

**Link**: [PDF](https://arxiv.org/pdf/2503.17515)  

**Abstract**: Predicting air traffic congestion and flow management is essential for airlines and Air Navigation Service Providers (ANSP) to enhance operational efficiency. Accurate estimates of future airport capacity and airspace density are vital for better airspace management, reducing air traffic controller workload and fuel consumption, ultimately promoting sustainable aviation. While existing literature has addressed these challenges, data management and query processing remain complex due to the vast volume of high-rate air traffic data. Many analytics use cases require a common pre-processing infrastructure, as ad-hoc approaches are insufficient. Additionally, linear prediction models often fall short, necessitating more advanced techniques.
This paper presents a data processing and predictive services architecture that ingests large, uncorrelated, and noisy streaming data to forecast future airspace system states. The system continuously collects raw data, periodically compresses it, and stores it in NoSQL databases for efficient query processing. For prediction, the system learns from historical traffic by extracting key features such as airport arrival and departure events, sector boundary crossings, weather parameters, and other air traffic data. These features are input into various regression models, including linear, non-linear, and ensemble models, with the best-performing model selected for predictions. We evaluate this infrastructure across three prediction use cases in the US National Airspace System (NAS) and a segment of European airspace, using extensive real operations data, confirming that our system can predict future system states efficiently and accurately. 

**Abstract (ZH)**: 预测空中交通拥堵和流量管理对于航空公司和空中导航服务提供商（ANSP）提升运营效率至关重要。准确估计未来的机场容量和 airspace 密度对于更好的 airspace 管理、减少空中交通管制员的工作负荷和燃油消耗、最终促进可持续航空具有重要作用。尽管现有文献已解决了这些挑战，但数据管理和查询处理由于高流量空中交通数据量巨大仍非常复杂。许多分析用例需要一种通用的预处理基础设施，临时解决方案远远不够。此外，线性预测模型经常不足，需要更高级的技术。本文提出了一种数据处理和预测服务架构，该架构能够处理大规模、不相关和嘈杂的流数据，以预测未来的 airspace 系统状态。该系统持续收集原始数据，定期对其进行压缩，并将其存储在 NoSQL 数据库中，以实现高效查询处理。对于预测，该系统通过提取历史交通的关键特征（如机场到达和离场事件、管制区边界穿越、天气参数和其他空中交通数据）来学习历史交通。这些特征被输入到各种回归模型中，包括线性模型、非线性模型和集成模型，选择表现最佳的模型进行预测。我们通过在美国国家 airspace 系统（NAS）和欧洲 airspace 的一个区域使用广泛的真实操作数据，跨三个预测用例评估了此基础设施，证实了我们的系统能够高效准确地预测未来系统状态。 

---
# Follow-up Question Generation For Enhanced Patient-Provider Conversations 

**Title (ZH)**: 增强患者-提供者对话的跟进问题生成 

**Authors**: Joseph Gatto, Parker Seegmiller, Timothy Burdick, Inas S. Khayal, Sarah DeLozier, Sarah M. Preum  

**Link**: [PDF](https://arxiv.org/pdf/2503.17509)  

**Abstract**: Follow-up question generation is an essential feature of dialogue systems as it can reduce conversational ambiguity and enhance modeling complex interactions. Conversational contexts often pose core NLP challenges such as (i) extracting relevant information buried in fragmented data sources, and (ii) modeling parallel thought processes. These two challenges occur frequently in medical dialogue as a doctor asks questions based not only on patient utterances but also their prior EHR data and current diagnostic hypotheses. Asking medical questions in asynchronous conversations compounds these issues as doctors can only rely on static EHR information to motivate follow-up questions.
To address these challenges, we introduce FollowupQ, a novel framework for enhancing asynchronous medical conversation. FollowupQ is a multi-agent framework that processes patient messages and EHR data to generate personalized follow-up questions, clarifying patient-reported medical conditions. FollowupQ reduces requisite provider follow-up communications by 34%. It also improves performance by 17% and 5% on real and synthetic data, respectively. We also release the first public dataset of asynchronous medical messages with linked EHR data alongside 2,300 follow-up questions written by clinical experts for the wider NLP research community. 

**Abstract (ZH)**: FollowupQ：一种增强异步医疗对话的新型框架 

---
# Efficient Knowledge Distillation via Curriculum Extraction 

**Title (ZH)**: 通过课程提取实现高效知识蒸馏 

**Authors**: Shivam Gupta, Sushrut Karmalkar  

**Link**: [PDF](https://arxiv.org/pdf/2503.17494)  

**Abstract**: Knowledge distillation is a technique used to train a small student network using the output generated by a large teacher network, and has many empirical advantages~\citep{Hinton2015DistillingTK}. While the standard one-shot approach to distillation only uses the output of the final teacher network, recent work~\citep{panigrahi2024progressive} has shown that using intermediate checkpoints from the teacher's training process as an implicit ``curriculum'' for progressive distillation can significantly speed up training. However, such schemes require storing these checkpoints, and often require careful selection of the intermediate checkpoints to train on, which can be impractical for large-scale training.
In this paper, we show that a curriculum can be \emph{extracted} from just the fully trained teacher network, and that this extracted curriculum can give similar efficiency benefits to those of progressive distillation. Our extraction scheme is natural; we use a random projection of the hidden representations of the teacher network to progressively train the student network, before training using the output of the full network. We show that our scheme significantly outperforms one-shot distillation and achieves a performance similar to that of progressive distillation for learning sparse parities with two-layer networks, and provide theoretical guarantees for this setting. Additionally, we show that our method outperforms one-shot distillation even when using transformer-based architectures, both for sparse-parity learning, and language modeling tasks. 

**Abstract (ZH)**: 知识蒸馏是一种使用大型教师网络生成的输出来训练小型学生网络的技术，它具有许多实证优势~\citep{Hinton2015DistillingTK}。虽然标准的一次性蒸馏方法仅使用教师网络最终的输出，但最近的研究~\citep{panigrahi2024progressive}表明，将教师网络训练过程中的中间检查点作为渐进蒸馏的隐式“课程”可以显著加快训练速度。然而，这样的方案需要存储这些检查点，并且往往需要精心选择用于训练的中间检查点，这在大规模训练中可能是不实际的。
在本文中，我们展示了可以从仅有的完全训练好的教师网络中提取出一个课程，并且提取出的课程可以提供类似于渐进蒸馏的效率优势。我们的提取方案自然；我们通过将教师网络的隐藏表示进行随机投影来逐步训练学生网络，在使用整个网络的输出进行训练之前。我们证明了我们的方案在稀疏相学习中显著优于一次性蒸馏，并且在两层网络中达到与渐进蒸馏相似的性能，并为此场景提供了理论保证。此外，我们展示了即使在使用基于Transformer的架构时，我们的方法在稀疏相学习和语言建模任务中也优于一次性蒸馏。 

---
# What's Producible May Not Be Reachable: Measuring the Steerability of Generative Models 

**Title (ZH)**: 可生成的未必可达：生成模型的可控性度量 

**Authors**: Keyon Vafa, Sarah Bentley, Jon Kleinberg, Sendhil Mullainathan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17482)  

**Abstract**: How should we evaluate the quality of generative models? Many existing metrics focus on a model's producibility, i.e. the quality and breadth of outputs it can generate. However, the actual value from using a generative model stems not just from what it can produce but whether a user with a specific goal can produce an output that satisfies that goal. We refer to this property as steerability. In this paper, we first introduce a mathematical framework for evaluating steerability independently from producibility. Steerability is more challenging to evaluate than producibility because it requires knowing a user's goals. We address this issue by creating a benchmark task that relies on one key idea: sample an output from a generative model and ask users to reproduce it. We implement this benchmark in a large-scale user study of text-to-image models and large language models. Despite the ability of these models to produce high-quality outputs, they all perform poorly on steerabilty. This suggests that we need to focus on improving the steerability of generative models. We show such improvements are indeed possible: through reinforcement learning techniques, we create an alternative steering mechanism for image models that achieves more than 2x improvement on this benchmark. 

**Abstract (ZH)**: 如何评估生成模型的质量？：独立于生成性之外评估可控性 

---
# CausalRivers -- Scaling up benchmarking of causal discovery for real-world time-series 

**Title (ZH)**: 因果河流——因果发现基准测试的扩展与应用 

**Authors**: Gideon Stein, Maha Shadaydeh, Jan Blunk, Niklas Penzel, Joachim Denzler  

**Link**: [PDF](https://arxiv.org/pdf/2503.17452)  

**Abstract**: Causal discovery, or identifying causal relationships from observational data, is a notoriously challenging task, with numerous methods proposed to tackle it. Despite this, in-the-wild evaluation of these methods is still lacking, as works frequently rely on synthetic data evaluation and sparse real-world examples under critical theoretical assumptions. Real-world causal structures, however, are often complex, making it hard to decide on a proper causal discovery strategy. To bridge this gap, we introduce CausalRivers, the largest in-the-wild causal discovery benchmarking kit for time-series data to date. CausalRivers features an extensive dataset on river discharge that covers the eastern German territory (666 measurement stations) and the state of Bavaria (494 measurement stations). It spans the years 2019 to 2023 with a 15-minute temporal resolution. Further, we provide additional data from a flood around the Elbe River, as an event with a pronounced distributional shift. Leveraging multiple sources of information and time-series meta-data, we constructed two distinct causal ground truth graphs (Bavaria and eastern Germany). These graphs can be sampled to generate thousands of subgraphs to benchmark causal discovery across diverse and challenging settings. To demonstrate the utility of CausalRivers, we evaluate several causal discovery approaches through a set of experiments to identify areas for improvement. CausalRivers has the potential to facilitate robust evaluations and comparisons of causal discovery methods. Besides this primary purpose, we also expect that this dataset will be relevant for connected areas of research, such as time-series forecasting and anomaly detection. Based on this, we hope to push benchmark-driven method development that fosters advanced techniques for causal discovery, as is the case for many other areas of machine learning. 

**Abstract (ZH)**: 因果发现，即从观察数据中识别因果关系，是一个 notoriously 挑战性的任务，尽管已经提出了众多方法来解决这一问题，但在实际应用中的评估仍然不足，研究工作经常依赖合成数据评估和在关键理论假设下的稀疏真实世界示例。然而，真实世界中的因果结构往往极为复杂，这使得选择合适的因果发现策略变得困难。为了弥合这一差距，我们介绍了CausalRivers，这是迄今为止规模最大的实际应用因果发现基准数据集，专用于时间序列数据。CausalRivers 包含覆盖德国东部（666个测站）和巴伐利亚州（494个测站）的河流流量数据集。该数据集的时间跨度为2019年至2023年，时间分辨率为15分钟。此外，我们还提供了关于易北河洪灾的附加数据，这是一个具有明显分布变化的事件。利用多种信息来源和时间序列元数据，我们构建了两个独立的因果真实图形（巴伐利亚和德国东部）。这些图形可以通过采样生成数千个子图，以在不同的复杂场景中测试因果发现方法。为了展示CausalRivers的实用性，我们通过一系列实验评估了几种因果发现方法，以识别改进领域。CausalRivers有可能促进因果发现方法的稳健评估和比较。除了这一主要目的外，我们还期望该数据集在时间序列预测和异常检测等关联研究领域中具有相关性。基于此，我们希望推动以基准驱动的方法研发，促进因果发现技术的发展，这在许多其他机器学习领域中已有先例。 

---
# Enhanced Smart Contract Reputability Analysis using Multimodal Data Fusion on Ethereum 

**Title (ZH)**: 基于多模态数据融合的增强 Ethereum 智能合约声誉分析 

**Authors**: Cyrus Malik, Josef Bajada, Joshua Ellul  

**Link**: [PDF](https://arxiv.org/pdf/2503.17426)  

**Abstract**: The evaluation of smart contract reputability is essential to foster trust in decentralized ecosystems. However, existing methods that rely solely on static code analysis or transactional data, offer limited insight into evolving trustworthiness. We propose a multimodal data fusion framework that integrates static code features with transactional data to enhance reputability prediction. Our framework initially focuses on static code analysis, utilizing GAN-augmented opcode embeddings to address class imbalance, achieving 97.67% accuracy and a recall of 0.942 in detecting illicit contracts, surpassing traditional oversampling methods. This forms the crux of a reputability-centric fusion strategy, where combining static and transactional data improves recall by 7.25% over single-source models, demonstrating robust performance across validation sets. By providing a holistic view of smart contract behaviour, our approach enhances the model's ability to assess reputability, identify fraudulent activities, and predict anomalous patterns. These capabilities contribute to more accurate reputability assessments, proactive risk mitigation, and enhanced blockchain security. 

**Abstract (ZH)**: 智能合约声誉性的多模态数据融合评估对于促进去中心化生态系统中的信任至关重要。现有的仅依赖静态代码分析或交易数据的方法提供有限的关于信任演化程度的见解。我们提出了一种多模态数据融合框架，将静态代码特征与交易数据结合起来以增强声誉性预测。该框架首先专注于静态代码分析，利用GAN增强的opcode嵌入来解决类别不平衡问题，实现了97.67%的准确率和0.942的召回率，超越了传统的过采样方法。这构成了以声誉性为中心的融合策略的核心，结合静态和交易数据比单一数据源模型召回率提升7.25%，展示了在验证集上的稳健性能。通过提供智能合约行为的全面视角，我们的方法增强了模型评估声誉性、识别欺诈活动和预测异常模式的能力，从而促进了更准确的声誉性评估、主动风险管理以及区块链安全的提升。 

---
# Data to Decisions: A Computational Framework to Identify skill requirements from Advertorial Data 

**Title (ZH)**: 从广告数据到决策：一种识别技能需求的计算框架 

**Authors**: Aakash Singh, Anurag Kanaujia, Vivek Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.17424)  

**Abstract**: Among the factors of production, human capital or skilled manpower is the one that keeps evolving and adapts to changing conditions and resources. This adaptability makes human capital the most crucial factor in ensuring a sustainable growth of industry/sector. As new technologies are developed and adopted, the new generations are required to acquire skills in newer technologies in order to be employable. At the same time professionals are required to upskill and reskill themselves to remain relevant in the industry. There is however no straightforward method to identify the skill needs of the industry at a given point of time. Therefore, this paper proposes a data to decision framework that can successfully identify the desired skill set in a given area by analysing the advertorial data collected from popular online job portals and supplied as input to the framework. The proposed framework uses techniques of statistical analysis, data mining and natural language processing for the purpose. The applicability of the framework is demonstrated on CS&IT job advertisement data from India. The analytical results not only provide useful insights about current state of skill needs in CS&IT industry but also provide practical implications to prospective job applicants, training agencies, and institutions of higher education & professional training. 

**Abstract (ZH)**: 生产要素中，人力资本或技术劳动力是最具适应性的因素，能够适应不断变化的条件和资源。这种适应性使得人力资本成为确保产业可持续增长的关键因素。随着新技术的开发和采纳，新一代人需要掌握新的技术以保持就业能力。与此同时，专业人士也需要不断自我提升和重新技能培训以保持在行业的相关性。然而，并没有直接的方法来确定某一时间点行业的技能需求。因此，本文提出了一种数据到决策的框架，通过分析来自流行的在线招聘网站的广告数据并将其作为输入提供给框架，以成功识别给定区域所需的技能集。该框架利用统计分析、数据挖掘和自然语言处理技术。该框架的应用性在印度CS&IT职位广告数据上进行了验证。分析结果不仅提供了有关CS&IT行业当前技能需求状态的有用见解，还为求职者、培训机构以及高等教育和专业培训机构提供了实际意义。 

---
# Opportunities and Challenges of Frontier Data Governance With Synthetic Data 

**Title (ZH)**: 前沿数据治理中的合成数据机遇与挑战 

**Authors**: Madhavendra Thakur, Jason Hausenloy  

**Link**: [PDF](https://arxiv.org/pdf/2503.17414)  

**Abstract**: Synthetic data, or data generated by machine learning models, is increasingly emerging as a solution to the data access problem. However, its use introduces significant governance and accountability challenges, and potentially debases existing governance paradigms, such as compute and data governance. In this paper, we identify 3 key governance and accountability challenges that synthetic data poses - it can enable the increased emergence of malicious actors, spontaneous biases and value drift. We thus craft 3 technical mechanisms to address these specific challenges, finding applications for synthetic data towards adversarial training, bias mitigation and value reinforcement. These could not only counteract the risks of synthetic data, but serve as critical levers for governance of the frontier in the future. 

**Abstract (ZH)**: 合成数据作为一种由机器学习模型生成的数据，在解决数据访问问题方面越来越受到关注，但其使用引入了重大的治理和问责挑战，并可能 debases 现有的治理范式，如计算和数据治理。本文识别了合成数据带来的 3 个关键治理和问责挑战：它可能促进恶意行为者增多、自发偏见的出现以及价值漂移。为此，我们提出了 3 种技术机制来应对这些特定挑战，并探讨了合成数据在对抗性训练、偏见缓解和价值强化方面的应用。这些机制不仅能够抵消合成数据的风险，还可能成为未来治理框架中的关键杠杆。 

---
# Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting 

**Title (ZH)**: 深度学习模型在实际ISP网络流量预测中的 comparative analysis 

**Authors**: Josef Koumar, Timotej Smoleň, Kamil Jeřábek, Tomáš Čejka  

**Link**: [PDF](https://arxiv.org/pdf/2503.17410)  

**Abstract**: Accurate network traffic forecasting is essential for Internet Service Providers (ISP) to optimize resources, enhance user experience, and mitigate anomalies. This study evaluates state-of-the-art deep learning models on CESNET-TimeSeries24, a recently published, comprehensive real-world network traffic dataset from the ISP network CESNET3 spanning multivariate time series over 40 weeks. Our findings highlight the balance between prediction accuracy and computational efficiency across different levels of network granularity. Additionally, this work establishes a reproducible methodology that facilitates direct comparison of existing approaches, explores their strengths and weaknesses, and provides a benchmark for future studies using this dataset. 

**Abstract (ZH)**: 准确的网络流量预测对于Internet服务提供商优化资源、提升用户体验和缓解异常至关重要。本研究评估了最新的深度学习模型在CESNET-TimeSeries24数据集上的性能，该数据集是来自ISP网络CESNET3的综合实际网络流量数据，涵盖40周多变量时间序列。研究发现突显了不同网络粒度级别下预测准确性和计算效率之间的平衡关系。此外，本工作还建立了一种可重复的方法，便于直接比较现有方法，探索其优缺点，并为未来使用此数据集的研究提供基准。 

---
# AEJIM: A Real-Time AI Framework for Crowdsourced, Transparent, and Ethical Environmental Hazard Detection and Reporting 

**Title (ZH)**: AEJIM：一种实时人工智能框架，用于 crowdsourced、透明和伦理的环境危害检测与报告 

**Authors**: Torsten Tiltack  

**Link**: [PDF](https://arxiv.org/pdf/2503.17401)  

**Abstract**: Environmental journalism is vital for raising awareness of ecological crises and driving evidence-based policy, yet traditional methods falter under delays, inaccuracies, and scalability limits, especially in under-monitored regions critical to the United Nations Sustainable Development Goals. To bridge these gaps, this paper introduces the AI-Environmental Journalism Integration Model (AEJIM), an innovative framework combining real-time hazard detection, crowdsourced validation, and AI-driven reporting.
Validated through a pilot study, AEJIM significantly improved the speed and accuracy of environmental hazard reporting, outperforming traditional methods. Furthermore, the model directly addresses key ethical, regulatory, and scalability challenges, ensuring AI accountability through Explainable AI (XAI), GDPR-compliant data governance, and active public participation. AEJIM provides a transparent and adaptable solution, setting a new benchmark for AI-enhanced environmental journalism and supporting informed global decision-making across diverse socio-political landscapes. 

**Abstract (ZH)**: 环保 journalism 对提高生态危机意识和推动基于证据的政策至关重要，但传统方法在延时、不准确和可扩展性方面存在局限，尤其是在对于联合国可持续发展目标至关重要的未监测地区。为此，本文引入了AI-环保 Journalism 整合模型（AEJIM），这是一种将实时风险检测、群众验证和AI驱动报道结合在一起的创新框架。经过试点研究验证，AEJIM 显著提高了环境风险报告的速度和准确性，超越了传统方法。此外，该模型直接解决了关键的伦理、监管和可扩展性挑战，通过可解释的AI（XAI）、GDPR 合规的数据治理和积极的公众参与确保AI问责。AEJIM 提供了一个透明且可适应的解决方案，为AI增强的环保 journalism 设立了新标准，并支持跨多样社会政治景观的知情全球决策。 

---
# Temporal Flexibility in Spiking Neural Networks: Towards Generalization Across Time Steps and Deployment Friendliness 

**Title (ZH)**: 时空灵活的脉冲神经网络：面向时间步长泛化与部署友好性 

**Authors**: Kangrui Du, Yuhang Wu, Shikuang Deng, Shi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17394)  

**Abstract**: Spiking Neural Networks (SNNs), models inspired by neural mechanisms in the brain, allow for energy-efficient implementation on neuromorphic hardware. However, SNNs trained with current direct training approaches are constrained to a specific time step. This "temporal inflexibility" 1) hinders SNNs' deployment on time-step-free fully event-driven chips and 2) prevents energy-performance balance based on dynamic inference time steps. In this study, we first explore the feasibility of training SNNs that generalize across different time steps. We then introduce Mixed Time-step Training (MTT), a novel method that improves the temporal flexibility of SNNs, making SNNs adaptive to diverse temporal structures. During each iteration of MTT, random time steps are assigned to different SNN stages, with spikes transmitted between stages via communication modules. After training, the weights are deployed and evaluated on both time-stepped and fully event-driven platforms. Experimental results show that models trained by MTT gain remarkable temporal flexibility, friendliness for both event-driven and clock-driven deployment (nearly lossless on N-MNIST and 10.1% higher than standard methods on CIFAR10-DVS), enhanced network generalization, and near SOTA performance. To the best of our knowledge, this is the first work to report the results of large-scale SNN deployment on fully event-driven scenarios. 

**Abstract (ZH)**: 基于混合时间步训练的Spiking神经网络的时间灵活性提升研究 

---
# AI-driven Automation of End-to-end Assessment of Suturing Expertise 

**Title (ZH)**: 基于AI的缝合技能端到端评估自动化 

**Authors**: Atharva Deo, Nicholas Matsumoto, Sun Kim, Peter Wager, Randy G. Tsai, Aaron Denmark, Cherine Yang, Xi Li, Jay Moran, Miguel Hernandez, Andrew J. Hung  

**Link**: [PDF](https://arxiv.org/pdf/2503.17391)  

**Abstract**: We present an AI based approach to automate the End-to-end Assessment of Suturing Expertise (EASE), a suturing skills assessment tool that comprehensively defines criteria around relevant sub-skills.1 While EASE provides granular skills assessment related to suturing to provide trainees with an objective evaluation of their aptitude along with actionable insights, the scoring process is currently performed by human evaluators, which is time and resource consuming. The AI based approach solves this by enabling real-time score prediction with minimal resources during model inference. This enables the possibility of real-time feedback to the surgeons/trainees, potentially accelerating the learning process for the suturing task and mitigating critical errors during the surgery, improving patient outcomes. In this study, we focus on the following 7 EASE domains that come under 3 suturing phases: 1) Needle Handling: Number of Repositions, Needle Hold Depth, Needle Hold Ratio, and Needle Hold Angle; 2) Needle Driving: Driving Smoothness, and Wrist Rotation; 3) Needle Withdrawal: Wrist Rotation. 

**Abstract (ZH)**: 基于AI的全自动端到端缝合技巧评估（EASE）方法：一种全面定义相关子技能标准的缝合技能评估工具 

---
# AI Companies Should Report Pre- and Post-Mitigation Safety Evaluations 

**Title (ZH)**: AI公司应报告预处理和后处理安全性评估结果 

**Authors**: Dillon Bowen, Ann-Kathrin Dombrowski, Adam Gleave, Chris Cundy  

**Link**: [PDF](https://arxiv.org/pdf/2503.17388)  

**Abstract**: The rapid advancement of AI systems has raised widespread concerns about potential harms of frontier AI systems and the need for responsible evaluation and oversight. In this position paper, we argue that frontier AI companies should report both pre- and post-mitigation safety evaluations to enable informed policy decisions. Evaluating models at both stages provides policymakers with essential evidence to regulate deployment, access, and safety standards. We show that relying on either in isolation can create a misleading picture of model safety. Our analysis of AI safety disclosures from leading frontier labs identifies three critical gaps: (1) companies rarely evaluate both pre- and post-mitigation versions, (2) evaluation methods lack standardization, and (3) reported results are often too vague to inform policy. To address these issues, we recommend mandatory disclosure of pre- and post-mitigation capabilities to approved government bodies, standardized evaluation methods, and minimum transparency requirements for public safety reporting. These ensure that policymakers and regulators can craft targeted safety measures, assess deployment risks, and scrutinize companies' safety claims effectively. 

**Abstract (ZH)**: 人工智能系统的迅速发展引发了对前沿人工智能系统潜在危害的广泛担忧，以及对其负责任评估和监管的需要。在本文中，我们主张前沿人工智能公司应报告预干预和后干预的安全评估，以促进知情政策决策。在两个阶段评估模型为政策制定者提供了至关重要的证据，以规范部署、访问和安全标准。我们表明，仅依赖其中任何一个都会导致对模型安全性的误导性描述。通过对领先前沿实验室的人工智能安全披露进行分析，我们发现了三个关键缺口：（1）公司通常不评估预干预和后干预的版本，（2）评估方法缺乏标准化，（3）报告的结果往往不够具体，无法为政策制定提供信息。为解决这些问题，我们建议强制要求在获批的政府机构披露预干预和后干预的能力，采用标准化的评估方法，并对公众安全报告设定最低透明度要求。这些措施确保政策制定者和监管机构能够制定有针对性的安全措施，评估部署风险，并有效地审查公司的安全声明。 

---
# Non-Canonical Crosslinks Confound Evolutionary Protein Structure Models 

**Title (ZH)**: 非范式交叉链接困扰着进化蛋白质结构模型 

**Authors**: Romain Lacombe  

**Link**: [PDF](https://arxiv.org/pdf/2503.17368)  

**Abstract**: Evolution-based protein structure prediction models have achieved breakthrough success in recent years. However, they struggle to generalize beyond evolutionary priors and on sequences lacking rich homologous data. Here we present a novel, out-of-domain benchmark based on sactipeptides, a rare class of ribosomally synthesized and post-translationally modified peptides (RiPPs) characterized by sulfur-to-$\alpha$-carbon thioether bridges creating cross-links between cysteine residues and backbone. We evaluate recent models on predicting conformations compatible with these cross-links bridges for the 10 known sactipeptides with elucidated post-translational modifications. Crucially, the structures of 5 of them have not yet been experimentally resolved. This makes the task a challenging problem for evolution-based models, which we find exhibit limited performance (0.0% to 19.2% GDT-TS on sulfur-to-$\alpha$-carbon distance). Our results point at the need for physics-informed models to sustain progress in biomolecular structure prediction. 

**Abstract (ZH)**: 基于进化的新颖领域外基准：硫至α碳硫醚桥连接的桑蒂肽结构预测 

---
# Big Help or Big Brother? Auditing Tracking, Profiling, and Personalization in Generative AI Assistants 

**Title (ZH)**: 大助手中还是大弟弟？生成式人工智能助手的审计、建模与个性化探究 

**Authors**: Yash Vekaria, Aurelio Loris Canino, Jonathan Levitsky, Alex Ciechonski, Patricia Callejo, Anna Maria Mandalari, Zubair Shafiq  

**Link**: [PDF](https://arxiv.org/pdf/2503.16586)  

**Abstract**: Generative AI (GenAI) browser assistants integrate powerful capabilities of GenAI in web browsers to provide rich experiences such as question answering, content summarization, and agentic navigation. These assistants, available today as browser extensions, can not only track detailed browsing activity such as search and click data, but can also autonomously perform tasks such as filling forms, raising significant privacy concerns. It is crucial to understand the design and operation of GenAI browser extensions, including how they collect, store, process, and share user data. To this end, we study their ability to profile users and personalize their responses based on explicit or inferred demographic attributes and interests of users. We perform network traffic analysis and use a novel prompting framework to audit tracking, profiling, and personalization by the ten most popular GenAI browser assistant extensions. We find that instead of relying on local in-browser models, these assistants largely depend on server-side APIs, which can be auto-invoked without explicit user interaction. When invoked, they collect and share webpage content, often the full HTML DOM and sometimes even the user's form inputs, with their first-party servers. Some assistants also share identifiers and user prompts with third-party trackers such as Google Analytics. The collection and sharing continues even if a webpage contains sensitive information such as health or personal information such as name or SSN entered in a web form. We find that several GenAI browser assistants infer demographic attributes such as age, gender, income, and interests and use this profile--which carries across browsing contexts--to personalize responses. In summary, our work shows that GenAI browser assistants can and do collect personal and sensitive information for profiling and personalization with little to no safeguards. 

**Abstract (ZH)**: 基于生成式AI的浏览器助手整合了强大生成式AI能力以提供丰富的浏览体验，如问答、内容总结和智能导航。这些助手作为浏览器插件可用，不仅能追踪详细的浏览活动，如搜索和点击数据，还能自主完成填写表单等任务，引发重大隐私担忧。理解生成式AI浏览器插件的设计和运行机制至关重要，包括它们如何收集、存储、处理和共享用户数据。为此，我们研究了它们基于用户显性和隐性的人口统计属性和兴趣进行用户画像和个人化响应的能力。我们进行了网络流量分析，并使用新颖的提示框架对十款最流行的生成式AI浏览器助手插件的跟踪、画像和个人化行为进行审计。我们发现，这些助手主要依赖服务器端API，这些API可以在用户未进行显式交互的情况下自动调用。调用时，它们会收集并与其他第一方服务器共享网页内容，包括完整HTML DOM，有时还会共享用户的表单输入。一些助手还会与第三方跟踪器（如Google Analytics）共享标识符和用户提示。即使网页包含敏感信息，如健康或个人信息（如姓名或社保号码），收集和共享过程也继续进行。我们发现，多个生成式AI浏览器助手能够推断出年龄、性别、收入和兴趣等人口统计属性，并使用这些画像（这些画像跨越不同的浏览上下文）来个性化响应。总之，我们的研究显示，生成式AI浏览器助手在几乎没有安全措施的情况下收集并使用个人和敏感信息进行画像和个性化处理。 

---
