# EMMA: Scaling Mobile Manipulation via Egocentric Human Data 

**Title (ZH)**: EMMA: 通过第一人称人体数据扩展移动 manipulation 技术 

**Authors**: Lawrence Y. Zhu, Pranav Kuppili, Ryan Punamiya, Patcharapong Aphiwetsa, Dhruv Patel, Simar Kareer, Sehoon Ha, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04443)  

**Abstract**: Scaling mobile manipulation imitation learning is bottlenecked by expensive mobile robot teleoperation. We present Egocentric Mobile MAnipulation (EMMA), an end-to-end framework training mobile manipulation policies from human mobile manipulation data with static robot data, sidestepping mobile teleoperation. To accomplish this, we co-train human full-body motion data with static robot data. In our experiments across three real-world tasks, EMMA demonstrates comparable performance to baselines trained on teleoperated mobile robot data (Mobile ALOHA), achieving higher or equivalent task performance in full task success. We find that EMMA is able to generalize to new spatial configurations and scenes, and we observe positive performance scaling as we increase the hours of human data, opening new avenues for scalable robotic learning in real-world environments. Details of this project can be found at this https URL. 

**Abstract (ZH)**: 移动 manipulation 模仿学习的扩展受制于昂贵的移动机器人遥操作。我们提出了第一人称移动 manipulation (EMMA) 框架，该框架从人类移动 manipulation 数据和静态机器人数据中端到端训练移动 manipulation 策略，避开移动遥操作。为了实现这一目标，我们同时训练人类全身运动数据和静态机器人数据。在我们的三次实际任务实验中，EMMA 在全任务成功率方面展示了与基于遥操作移动机器人数据（Mobile ALOHA）训练的基础模型相当或更好的性能。我们发现 EMMA 能够泛化到新的空间配置和场景，并观察到随着人类数据小时数的增加，性能呈现积极的扩展趋势，为实际环境中的可扩展机器人学习开辟了新途径。详细内容请参见 <https://www.example.com>。 

---
# Cloud-Assisted Remote Control for Aerial Robots: From Theory to Proof-of-Concept Implementation 

**Title (ZH)**: 云辅助远程控制的空中机器人技术：从理论到概念验证实现 

**Authors**: Achilleas Santi Seisa, Viswa Narayanan Sankaranarayanan, Gerasimos Damigos, Sumeet Gajanan Satpute, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.04095)  

**Abstract**: Cloud robotics has emerged as a promising technology for robotics applications due to its advantages of offloading computationally intensive tasks, facilitating data sharing, and enhancing robot coordination. However, integrating cloud computing with robotics remains a complex challenge due to network latency, security concerns, and the need for efficient resource management. In this work, we present a scalable and intuitive framework for testing cloud and edge robotic systems. The framework consists of two main components enabled by containerized technology: (a) a containerized cloud cluster and (b) the containerized robot simulation environment. The system incorporates two endpoints of a User Datagram Protocol (UDP) tunnel, enabling bidirectional communication between the cloud cluster container and the robot simulation environment, while simulating realistic network conditions. To achieve this, we consider the use case of cloud-assisted remote control for aerial robots, while utilizing Linux-based traffic control to introduce artificial delay and jitter, replicating variable network conditions encountered in practical cloud-robot deployments. 

**Abstract (ZH)**: 云 robotics 作为一种具有卸载计算密集型任务、促进数据共享以及增强机器人协调优势的技术，在机器人应用中展现出令人期待的前景。然而，将云计算与机器人技术整合仍是一项复杂的挑战，主要由于网络延迟、安全问题以及高效资源管理的需求。在本文中，我们提出了一种可扩展且直观的框架，用于测试云和边缘机器人系统。该框架由两种主要组件构成，基于容器化技术实现：（a）容器化云集群和（b）容器化机器人仿真环境。该系统包括用户数据报协议（UDP）隧道的两个端点，允许云集群容器与机器人仿真环境之间的双向通信，并模拟实际网络条件。为了实现这一目标，我们以云辅助远程控制固定翼机器人作为应用场景，利用基于 Linux 的流量控制引入人工延迟和抖动，以模拟实际云-机器人部署中遇到的可变网络条件。 

---
# Balancing Signal and Variance: Adaptive Offline RL Post-Training for VLA Flow Models 

**Title (ZH)**: 信号与方差的平衡：适用于VLA流模型的自适应离线RL后训练 

**Authors**: Hongyin Zhang, Shiyuan Zhang, Junxi Jin, Qixin Zeng, Yifan Qiao, Hongchao Lu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04063)  

**Abstract**: Vision-Language-Action (VLA) models based on flow matching have shown excellent performance in general-purpose robotic manipulation tasks. However, the action accuracy of these models on complex downstream tasks is unsatisfactory. One important reason is that these models rely solely on the post-training paradigm of imitation learning, which makes it difficult to have a deeper understanding of the distribution properties of data quality, which is exactly what Reinforcement Learning (RL) excels at. In this paper, we theoretically propose an offline RL post-training objective for VLA flow models and induce an efficient and feasible offline RL fine-tuning algorithm -- Adaptive Reinforced Flow Matching (ARFM). By introducing an adaptively adjusted scaling factor in the VLA flow model loss, we construct a principled bias-variance trade-off objective function to optimally control the impact of RL signal on flow loss. ARFM adaptively balances RL advantage preservation and flow loss gradient variance control, resulting in a more stable and efficient fine-tuning process. Extensive simulation and real-world experimental results show that ARFM exhibits excellent generalization, robustness, few-shot learning, and continuous learning performance. 

**Abstract (ZH)**: 基于流匹配的Vision-Language-Action (VLA) 模型在通用机器人操作任务中表现出色，但在复杂下游任务中的动作精度不令人满意。一个重要的原因是这些模型仅依赖于模仿学习的后训练范式，这使得它们难以深入理解数据质量分布特性，而这正是强化学习（RL）的优势所在。本文从理论上为VLA流模型提出了一个离线RL后训练目标，并设计了一个高效可行的离线RL微调算法——自适应强化流匹配（ARFM）。通过在VLA流模型损失中引入自适应调整的缩放因子，我们构建了一个原理上的偏置-方差权衡目标函数，以最优控制RL信号对流损失的影响。ARFM自适应平衡了RL优势保留和流损失梯度方差控制，从而实现更稳定和高效的微调过程。广泛的仿真和真实世界实验结果表明，ARFM表现出出色的泛化能力、鲁棒性、少样本学习能力和持续学习性能。 

---
# FPC-VLA: A Vision-Language-Action Framework with a Supervisor for Failure Prediction and Correction 

**Title (ZH)**: FPC-VLA：带监督的视觉-语言-动作框架，用于故障预测与纠正 

**Authors**: Yifan Yang, Zhixiang Duan, Tianshi Xie, Fuyu Cao, Pinxi Shen, Peili Song, Piaopiao Jin, Guokang Sun, Shaoqing Xu, Yangwei You, Jingtai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04018)  

**Abstract**: Robotic manipulation is a fundamental component of automation. However, traditional perception-planning pipelines often fall short in open-ended tasks due to limited flexibility, while the architecture of a single end-to-end Vision-Language-Action (VLA) offers promising capabilities but lacks crucial mechanisms for anticipating and recovering from failure. To address these challenges, we propose FPC-VLA, a dual-model framework that integrates VLA with a supervisor for failure prediction and correction. The supervisor evaluates action viability through vision-language queries and generates corrective strategies when risks arise, trained efficiently without manual labeling. A similarity-guided fusion module further refines actions by leveraging past predictions. Evaluation results on multiple simulation platforms (SIMPLER and LIBERO) and robot embodiments (WidowX, Google Robot, Franka) show that FPC-VLA outperforms state-of-the-art models in both zero-shot and fine-tuned settings. By activating the supervisor only at keyframes, our approach significantly increases task success rates with minimal impact on execution time. Successful real-world deployments on diverse, long-horizon tasks confirm FPC-VLA's strong generalization and practical utility for building more reliable autonomous systems. 

**Abstract (ZH)**: 机器人 manipulation 是自动化的基础组成部分。然而，传统的感知-规划管道在开放任务中常常由于缺乏灵活性而表现不佳，而单一的端到端视觉-语言-行动 (VLA) 架构虽然提供了有前景的能力，但缺乏预见和从失败中恢复的关键机制。为了解决这些挑战，我们提出了一种名为 FPC-VLA 的双模型框架，该框架将 VLA 与故障预测和纠正的监督机制相结合。监督机制通过视觉-语言查询评估行动的有效性，并在风险出现时生成纠正策略，通过高效训练且无需手动标注。一个基于相似性引导的融合模块进一步通过利用过往预测来细化行动。在多个仿真平台 (SIMPLER 和 LIBERO) 和机器人硬件 (WidowX、Google Robot、Franka) 上的评估结果显示，FPC-VLA 在零样本和微调设置中均优于现有最先进的模型。只有在关键帧上激活监督机制，我们的方法显著提高了任务成功率，且 minimal 影响执行时间。在多样化的长期任务上的成功实际部署证实了 FPC-VLA 强大的泛化能力和实际应用价值，对于构建更可靠的自主系统具有重要意义。 

---
# Learning Multi-Stage Pick-and-Place with a Legged Mobile Manipulator 

**Title (ZH)**: 基于腿式移动 manipulator 的多阶段抓取放置学习 

**Authors**: Haichao Zhang, Haonan Yu, Le Zhao, Andrew Choi, Qinxun Bai, Yiqing Yang, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03859)  

**Abstract**: Quadruped-based mobile manipulation presents significant challenges in robotics due to the diversity of required skills, the extended task horizon, and partial observability. After presenting a multi-stage pick-and-place task as a succinct yet sufficiently rich setup that captures key desiderata for quadruped-based mobile manipulation, we propose an approach that can train a visuo-motor policy entirely in simulation, and achieve nearly 80\% success in the real world. The policy efficiently performs search, approach, grasp, transport, and drop into actions, with emerged behaviors such as re-grasping and task chaining. We conduct an extensive set of real-world experiments with ablation studies highlighting key techniques for efficient training and effective sim-to-real transfer. Additional experiments demonstrate deployment across a variety of indoor and outdoor environments. Demo videos and additional resources are available on the project page: this https URL. 

**Abstract (ZH)**: 基于四足机器人移动操作的抓取与放置任务展示了显著的挑战，由于所需技能的多样性、延展的任务时间范围以及部分可观测性。在提出了一种简洁但足够丰富的多阶段抓取与放置任务设定以捕捉四足机器人移动操作的关键需求后，我们提出了一种完全在仿真环境中训练视觉-运动策略的方法，并在现实世界中实现了近80%的成功率。该策略高效地执行搜索、接近、抓取、运输及释放动作，展现出重抓取和任务链等行为。我们进行了广泛的现实世界实验，并通过消融研究强调了高效训练和有效仿真到现实世界迁移的关键技术。此外，实验还展示了在多种室内外环境中的部署能力。更多实验视频和资源可在项目页面获取：this https URL。 

---
# INGRID: Intelligent Generative Robotic Design Using Large Language Models 

**Title (ZH)**: INGRID：使用大型语言模型的智能生成性机器人设计 

**Authors**: Guanglu Jia, Ceng Zhang, Gregory S. Chirikjian  

**Link**: [PDF](https://arxiv.org/pdf/2509.03842)  

**Abstract**: The integration of large language models (LLMs) into robotic systems has accelerated progress in embodied artificial intelligence, yet current approaches remain constrained by existing robotic architectures, particularly serial mechanisms. This hardware dependency fundamentally limits the scope of robotic intelligence. Here, we present INGRID (Intelligent Generative Robotic Design), a framework that enables the automated design of parallel robotic mechanisms through deep integration with reciprocal screw theory and kinematic synthesis methods. We decompose the design challenge into four progressive tasks: constraint analysis, kinematic joint generation, chain construction, and complete mechanism design. INGRID demonstrates the ability to generate novel parallel mechanisms with both fixed and variable mobility, discovering kinematic configurations not previously documented in the literature. We validate our approach through three case studies demonstrating how INGRID assists users in designing task-specific parallel robots based on desired mobility requirements. By bridging the gap between mechanism theory and machine learning, INGRID enables researchers without specialized robotics training to create custom parallel mechanisms, thereby decoupling advances in robotic intelligence from hardware constraints. This work establishes a foundation for mechanism intelligence, where AI systems actively design robotic hardware, potentially transforming the development of embodied AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机器人系统中的集成加速了具身人工智能的进步，但当前的方法仍受现有机器人架构的限制，尤其是序列机制。这种硬件依赖性从根本上限制了机器人智能的范围。为此，我们提出了INGRID（Intelligent Generative Robotic Design）框架，通过深度整合互反螺旋理论和运动合成方法，实现并行机器人机制的自动化设计。我们将设计挑战分解为四个渐进任务：约束分析、运动副生成、链路构造和完整机制设计。INGRID展示了生成既有固定移动性又有可变移动性的新型并行机制的能力，发现文献中未曾记载的运动学配置。我们通过三个案例研究验证了我们的方法，展示了INGRID如何帮助用户根据所需的移动性要求设计特定任务的并行机器人。通过弥合机制理论与机器学习之间的差距，INGRID使没有专门机器人训练的研究人员能够创建自定义的并行机制，从而解开机器人智能进步与硬件限制的关系。这项工作为机制智能奠定了基础，其中AI系统积极设计机器人硬件，有可能变革具身AI系统的开发。 

---
# Cooperative Grasping for Collective Object Transport in Constrained Environments 

**Title (ZH)**: 约束环境下协作抓取与集体对象运输 

**Authors**: David Alvear, George Turkiyyah, Shinkyu Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.03638)  

**Abstract**: We propose a novel framework for decision-making in cooperative grasping for two-robot object transport in constrained environments. The core of the framework is a Conditional Embedding (CE) model consisting of two neural networks that map grasp configuration information into an embedding space. The resulting embedding vectors are then used to identify feasible grasp configurations that allow two robots to collaboratively transport an object. To ensure generalizability across diverse environments and object geometries, the neural networks are trained on a dataset comprising a range of environment maps and object shapes. We employ a supervised learning approach with negative sampling to ensure that the learned embeddings effectively distinguish between feasible and infeasible grasp configurations. Evaluation results across a wide range of environments and objects in simulations demonstrate the model's ability to reliably identify feasible grasp configurations. We further validate the framework through experiments on a physical robotic platform, confirming its practical applicability. 

**Abstract (ZH)**: 一种用于受限环境下双机器人物体搬运协同抓取决策的新颖框架 

---
# SRWToolkit: An Open Source Wizard of Oz Toolkit to Create Social Robotic Avatars 

**Title (ZH)**: SRWToolkit: 一个开源的社会机器人avatar创建Wizard of Oz工具-kit 

**Authors**: Atikkhan Faridkhan Nilgar, Kristof Van Laerhoven, Ayub Kinoti  

**Link**: [PDF](https://arxiv.org/pdf/2509.04356)  

**Abstract**: We present SRWToolkit, an open-source Wizard of Oz toolkit designed to facilitate the rapid prototyping of social robotic avatars powered by local large language models (LLMs). Our web-based toolkit enables multimodal interaction through text input, button-activated speech, and wake-word command. The toolkit offers real-time configuration of avatar appearance, behavior, language, and voice via an intuitive control panel. In contrast to prior works that rely on cloud-based LLM services, SRWToolkit emphasizes modularity and ensures on-device functionality through local LLM inference. In our small-scale user study ($n=11$), participants created and interacted with diverse robotic roles (hospital receptionist, mathematics teacher, and driving assistant), which demonstrated positive outcomes in the toolkit's usability, trust, and user experience. The toolkit enables rapid and efficient development of robot characters customized to researchers' needs, supporting scalable research in human-robot interaction. 

**Abstract (ZH)**: SRWToolkit：一个用于基于本地大型语言模型的社会机器人avatar快速原型设计的开源Wizard of Oz工具包 

---
# Psychologically Enhanced AI Agents 

**Title (ZH)**: 心理增强型人工智能代理 

**Authors**: Maciej Besta, Shriram Chandran, Robert Gerstenberger, Mathis Lindner, Marcin Chrapek, Sebastian Hermann Martschat, Taraneh Ghandi, Patrick Iff, Hubert Niewiadomski, Piotr Nyczyk, Jürgen Müller, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2509.04343)  

**Abstract**: We introduce MBTI-in-Thoughts, a framework for enhancing the effectiveness of Large Language Model (LLM) agents through psychologically grounded personality conditioning. Drawing on the Myers-Briggs Type Indicator (MBTI), our method primes agents with distinct personality archetypes via prompt engineering, enabling control over behavior along two foundational axes of human psychology, cognition and affect. We show that such personality priming yields consistent, interpretable behavioral biases across diverse tasks: emotionally expressive agents excel in narrative generation, while analytically primed agents adopt more stable strategies in game-theoretic settings. Our framework supports experimenting with structured multi-agent communication protocols and reveals that self-reflection prior to interaction improves cooperation and reasoning quality. To ensure trait persistence, we integrate the official 16Personalities test for automated verification. While our focus is on MBTI, we show that our approach generalizes seamlessly to other psychological frameworks such as Big Five, HEXACO, or Enneagram. By bridging psychological theory and LLM behavior design, we establish a foundation for psychologically enhanced AI agents without any fine-tuning. 

**Abstract (ZH)**: 基于MBTI的心理定向框架：通过 personality conditioning 提升大型语言模型代理的有效性 

---
# EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation 

**Title (ZH)**: EvoEmo: 向往用于多轮谈判的LLM代理的情感策略演化 

**Authors**: Yunbo Long, Liming Xu, Lukas Beckenbauer, Yuhan Liu, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2509.04310)  

**Abstract**: Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation. 

**Abstract (ZH)**: Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in 复杂的、多轮的协商，开辟了代理型人工智能的新途径。然而，现有的LLM代理在这样的协商中很大程度上忽视了情绪的功能性作用，反而生成了被动的、基于偏好的情绪反应，使其容易受到对手方操纵和战略性利用。为解决这一问题，我们提出了EvoEmo，一种进化的强化学习框架，以优化协商中的动态情绪表达。EvoEmo将情绪状态转换建模为马尔可夫决策过程，并采用基于群体的遗传优化来进化出适用于多种协商场景的高奖励情绪策略。我们进一步提出了一种包含两种基线——常规策略和固定情绪策略——的评估框架，用于情绪意识协商的基准测试。广泛的实验和消融研究显示，EvoEmo在成功率、效率和买家节省方面均优于基线策略。这些发现突显了适应性情绪表达在使多轮协商中的人工智能代理更具效用方面的重要性。 

---
# World Model Implanting for Test-time Adaptation of Embodied Agents 

**Title (ZH)**: 基于世界模型的体 Agent 测试时适应植入 

**Authors**: Minjong Yoo, Jinwoo Jang, Sihyung Yoon, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2509.03956)  

**Abstract**: In embodied AI, a persistent challenge is enabling agents to robustly adapt to novel domains without requiring extensive data collection or retraining. To address this, we present a world model implanting framework (WorMI) that combines the reasoning capabilities of large language models (LLMs) with independently learned, domain-specific world models through test-time composition. By allowing seamless implantation and removal of the world models, the embodied agent's policy achieves and maintains cross-domain adaptability. In the WorMI framework, we employ a prototype-based world model retrieval approach, utilizing efficient trajectory-based abstract representation matching, to incorporate relevant models into test-time composition. We also develop a world-wise compound attention method that not only integrates the knowledge from the retrieved world models but also aligns their intermediate representations with the reasoning model's representation within the agent's policy. This framework design effectively fuses domain-specific knowledge from multiple world models, ensuring robust adaptation to unseen domains. We evaluate our WorMI on the VirtualHome and ALFWorld benchmarks, demonstrating superior zero-shot and few-shot performance compared to several LLM-based approaches across a range of unseen domains. These results highlight the frameworks potential for scalable, real-world deployment in embodied agent scenarios where adaptability and data efficiency are essential. 

**Abstract (ZH)**: 在具身AI中，一个持续的挑战是使代理能够 robustly 而不过分依赖大规模数据收集或重新训练的情况下适应新的领域。为了应对这一挑战，我们提出了一种世界模型植入框架（WorMI），该框架结合了大型语言模型的推理能力与独立学习的领域特定世界模型，通过测试时的组合实现。通过允许世界模型的无缝植入和移除，具身代理的策略实现了并保持了跨领域的适应性。在WorMI框架中，我们采用基于原型的世界模型检索方法，利用高效的基于轨迹的抽象表示匹配，将相关模型纳入测试时的组合。我们还开发了一种世界导向的复合注意力方法，不仅整合了检索到的世界模型的知识，还使它们的中间表示与代理策略中的推理模型表示进行对齐。该框架设计有效地融合了多个世界模型的领域特定知识，确保在未见过的领域中的稳健适应。我们在VirtualHome和ALFWorld基准上评估了我们的WorMI，展示了优于多种基于大型语言模型的方法的零样本和少样本性能，适用于一系列未见过的领域。这些结果突显了该框架在需要适应性与数据效率的具身代理场景中的潜在 scalability 和实际部署能力。 

---
# A Foundation Model for Chest X-ray Interpretation with Grounded Reasoning via Online Reinforcement Learning 

**Title (ZH)**: 基于 grounded reasoning 通过在线强化学习进行胸片解释的foundation模型 

**Authors**: Qika Lin, Yifan Zhu, Bin Pu, Ling Huang, Haoran Luo, Jingying Ma, Zhen Peng, Tianzhe Zhao, Fangzhi Xu, Jian Zhang, Kai He, Zhonghong Ou, Swapnil Mishra, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.03906)  

**Abstract**: Medical foundation models (FMs) have shown tremendous promise amid the rapid advancements in artificial intelligence (AI) technologies. However, current medical FMs typically generate answers in a black-box manner, lacking transparent reasoning processes and locally grounded interpretability, which hinders their practical clinical deployments. To this end, we introduce DeepMedix-R1, a holistic medical FM for chest X-ray (CXR) interpretation. It leverages a sequential training pipeline: initially fine-tuned on curated CXR instruction data to equip with fundamental CXR interpretation capabilities, then exposed to high-quality synthetic reasoning samples to enable cold-start reasoning, and finally refined via online reinforcement learning to enhance both grounded reasoning quality and generation performance. Thus, the model produces both an answer and reasoning steps tied to the image's local regions for each query. Quantitative evaluation demonstrates substantial improvements in report generation (e.g., 14.54% and 31.32% over LLaVA-Rad and MedGemma) and visual question answering (e.g., 57.75% and 23.06% over MedGemma and CheXagent) tasks. To facilitate robust assessment, we propose Report Arena, a benchmarking framework using advanced language models to evaluate answer quality, further highlighting the superiority of DeepMedix-R1. Expert review of generated reasoning steps reveals greater interpretability and clinical plausibility compared to the established Qwen2.5-VL-7B model (0.7416 vs. 0.2584 overall preference). Collectively, our work advances medical FM development toward holistic, transparent, and clinically actionable modeling for CXR interpretation. 

**Abstract (ZH)**: 基于深度 Medix-R1 在胸部 X 光片解释中的整体医疗基础模型：面向透明和临床实用的解释推理 

---
# Towards a Neurosymbolic Reasoning System Grounded in Schematic Representations 

**Title (ZH)**: 基于方案表示的神经符号推理系统研究 

**Authors**: François Olivier, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2509.03644)  

**Abstract**: Despite significant progress in natural language understanding, Large Language Models (LLMs) remain error-prone when performing logical reasoning, often lacking the robust mental representations that enable human-like comprehension. We introduce a prototype neurosymbolic system, Embodied-LM, that grounds understanding and logical reasoning in schematic representations based on image schemas-recurring patterns derived from sensorimotor experience that structure human cognition. Our system operationalizes the spatial foundations of these cognitive structures using declarative spatial reasoning within Answer Set Programming. Through evaluation on logical deduction problems, we demonstrate that LLMs can be guided to interpret scenarios through embodied cognitive structures, that these structures can be formalized as executable programs, and that the resulting representations support effective logical reasoning with enhanced interpretability. While our current implementation focuses on spatial primitives, it establishes the computational foundation for incorporating more complex and dynamic representations. 

**Abstract (ZH)**: 尽管在自然语言理解方面取得了显著进展，大型语言模型（LLMs）在执行逻辑推理时仍易出错，往往缺乏支撑人类类似理解的稳健的心理表征。我们提出了一种原型神经符号系统——Embodied-LM，该系统基于图式表示进行理解与逻辑推理，这些图式是源自感官运动体验中的反复出现的模式，结构化人类认知。我们的系统通过答案集程序中的声明性空间推理对这些认知结构的空间基础进行了实现。通过在逻辑推理问题上的评估，我们证明LLMs可以通过基于 embodied 认知结构来解释场景，这些结构可以被形式化为可执行程序，并且由此产生的表示支持具有增强解释性的有效逻辑推理。尽管我们当前的实现专注于空间原语，但它为整合更复杂和动态的表示奠定了计算基础。 

---
# CausalARC: Abstract Reasoning with Causal World Models 

**Title (ZH)**: 因果ARC：基于因果世界模型的抽象推理 

**Authors**: Jacqueline Maasch, John Kalantari, Kia Khezeli  

**Link**: [PDF](https://arxiv.org/pdf/2509.03636)  

**Abstract**: Reasoning requires adaptation to novel problem settings under limited data and distribution shift. This work introduces CausalARC: an experimental testbed for AI reasoning in low-data and out-of-distribution regimes, modeled after the Abstraction and Reasoning Corpus (ARC). Each CausalARC reasoning task is sampled from a fully specified causal world model, formally expressed as a structural causal model. Principled data augmentations provide observational, interventional, and counterfactual feedback about the world model in the form of few-shot, in-context learning demonstrations. As a proof-of-concept, we illustrate the use of CausalARC for four language model evaluation settings: (1) abstract reasoning with test-time training, (2) counterfactual reasoning with in-context learning, (3) program synthesis, and (4) causal discovery with logical reasoning. 

**Abstract (ZH)**: 因果推理要求在有限数据和分布偏移的情况下适应新的问题设置。本文介绍了CausalARC：一种针对低数据和分布外域的AI推理实验测试床，其设计灵感来源于抽象和推理语料库（ARC）。每个CausalARC推理任务都源自一个完全指定的因果世界模型，并以结构因果模型的形式正式表达。基于原则的数据增强提供了关于世界模型的观察性、干预性和反事实反馈，以少样本、上下文相关的学习演示形式呈现。作为概念验证，我们展示了CausalARC在四种语言模型评估场景中的应用：(1) 测试时训练的抽象推理，(2) 上下文学习的反事实推理，(3) 程序合成，(4) 逻辑推理驱动的因果发现。 

---
# Diffusion-RL Based Air Traffic Conflict Detection and Resolution Method 

**Title (ZH)**: 基于扩散-强化学习的空中交通冲突检测与化解方法 

**Authors**: Tonghe Li, Jixin Liu, Weili Zeng, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03550)  

**Abstract**: In the context of continuously rising global air traffic, efficient and safe Conflict Detection and Resolution (CD&R) is paramount for air traffic management. Although Deep Reinforcement Learning (DRL) offers a promising pathway for CD&R automation, existing approaches commonly suffer from a "unimodal bias" in their policies. This leads to a critical lack of decision-making flexibility when confronted with complex and dynamic constraints, often resulting in "decision deadlocks." To overcome this limitation, this paper pioneers the integration of diffusion probabilistic models into the safety-critical task of CD&R, proposing a novel autonomous conflict resolution framework named Diffusion-AC. Diverging from conventional methods that converge to a single optimal solution, our framework models its policy as a reverse denoising process guided by a value function, enabling it to generate a rich, high-quality, and multimodal action distribution. This core architecture is complemented by a Density-Progressive Safety Curriculum (DPSC), a training mechanism that ensures stable and efficient learning as the agent progresses from sparse to high-density traffic environments. Extensive simulation experiments demonstrate that the proposed method significantly outperforms a suite of state-of-the-art DRL benchmarks. Most critically, in the most challenging high-density scenarios, Diffusion-AC not only maintains a high success rate of 94.1% but also reduces the incidence of Near Mid-Air Collisions (NMACs) by approximately 59% compared to the next-best-performing baseline, significantly enhancing the system's safety margin. This performance leap stems from its unique multimodal decision-making capability, which allows the agent to flexibly switch to effective alternative maneuvers. 

**Abstract (ZH)**: 在全球 aviation 交通持续增长的背景下，高效的且安全的冲突检测与解决（CD&R）对于 aviation 交通管理至关重要。尽管深度强化学习（DRL）为 CD&R 的自动化提供了有前景的道路，但现有方法通常在其策略中存在“单模偏差”。这导致在面对复杂和动态的约束时缺乏决策灵活性，常导致“决策僵局”。为克服这一局限，本文首次将扩散概率模型集成到安全关键的任务——CD&R 中，提出了一种名为 Diffusion-AC 的新型自主冲突解决框架。不同于传统方法收敛于单一最优解，我们的框架将策略建模为由价值函数引导的逆去噪过程，使其能够生成丰富、高质且多模态的动作分布。该核心架构由密度进展安全课程（DPSC）训练机制加以补充，确保代理在从稀疏环境过渡到高密度环境时能实现稳定且高效的训练。大量仿真实验表明，所提出的方法显著优于一系列现有的 DRL 参考基准。尤其在最具有挑战性的高密度场景中，Diffusion-AC 保持了高达 94.1% 的高成功率，并且与性能次之的竞争者相比，将近失接近空中碰撞（NMAC）的频率降低了约 59%，显著提升了系统的安全余度。这种性能跃升源于其独特的多模态决策能力，使代理能够灵活切换至有效的替代机动。 

---
# Reinforcement Learning for Robust Ageing-Aware Control of Li-ion Battery Systems with Data-Driven Formal Verification 

**Title (ZH)**: 基于数据驱动形式验证的鲁棒老化感知Li-ion电池系统强化学习控制 

**Authors**: Rudi Coppola, Hovsep Touloujian, Pierfrancesco Ombrini, Manuel Mazo Jr  

**Link**: [PDF](https://arxiv.org/pdf/2509.04288)  

**Abstract**: Rechargeable lithium-ion (Li-ion) batteries are a ubiquitous element of modern technology. In the last decades, the production and design of such batteries and their adjacent embedded charging and safety protocols, denoted by Battery Management Systems (BMS), has taken central stage. A fundamental challenge to be addressed is the trade-off between the speed of charging and the ageing behavior, resulting in the loss of capacity in the battery cell. We rely on a high-fidelity physics-based battery model and propose an approach to data-driven charging and safety protocol design. Following a Counterexample-Guided Inductive Synthesis scheme, we combine Reinforcement Learning (RL) with recent developments in data-driven formal methods to obtain a hybrid control strategy: RL is used to synthesise the individual controllers, and a data-driven abstraction guides their partitioning into a switched structure, depending on the initial output measurements of the battery. The resulting discrete selection among RL-based controllers, coupled with the continuous battery dynamics, realises a hybrid system. When a design meets the desired criteria, the abstraction provides probabilistic guarantees on the closed-loop performance of the cell. 

**Abstract (ZH)**: 可充放锂离子（Li-ion）电池是现代技术中的一个普遍元件。在过去的几十年里，这类电池的生产、设计及其相关的嵌入式充电和安全协议，即电池管理系统（BMS），占据了核心位置。解决的基本挑战之一是在充电速度与电池老化行为之间权衡，导致电池容量损失。我们依赖于高保真物理电池模型，并提出了一种数据驱动的充电和安全协议设计方法。采用反例引导归纳综合方案，我们将强化学习（RL）与最近的数据驱动形式化方法进展相结合，获得一种混合控制策略：使用RL合成分离的控制器，并通过电池初始输出测量的数据驱动抽象将其划分为切换结构。由此产生的基于RL的控制器的离散选择，与连续的电池动力学相结合，实现了一种混合系统。当设计方案满足预期标准时，该抽象可提供闭环性能的概率保证。 

---
# Keypoint-based Diffusion for Robotic Motion Planning on the NICOL Robot 

**Title (ZH)**: 基于关键点的扩散模型在NICOL机器人运动规划中的应用 

**Authors**: Lennart Clasmeier, Jan-Gerrit Habekost, Connor Gäde, Philipp Allgeuer, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2509.04076)  

**Abstract**: We propose a novel diffusion-based action model for robotic motion planning. Commonly, established numerical planning approaches are used to solve general motion planning problems, but have significant runtime requirements. By leveraging the power of deep learning, we are able to achieve good results in a much smaller runtime by learning from a dataset generated by these planners. While our initial model uses point cloud embeddings in the input to predict keypoint-based joint sequences in its output, we observed in our ablation study that it remained challenging to condition the network on the point cloud embeddings. We identified some biases in our dataset and refined it, which improved the model's performance. Our model, even without the use of the point cloud encodings, outperforms numerical models by an order of magnitude regarding the runtime, while reaching a success rate of up to 90% of collision free solutions on the test set. 

**Abstract (ZH)**: 我们提出了一种基于扩散的动作模型用于机器人运动规划。通常，现有的数值规划方法被用于解决一般的运动规划问题，但这些方法需要显著的运行时间。通过利用深度学习的能力，我们能够在学习来自这些规划器生成的数据集之后，在更小的运行时间内取得良好的结果。虽然最初模型在输入中使用点云嵌入来预测关键点基于的关节序列，但在去噪研究中我们发现难以条件化网络依赖于点云嵌入。我们识别并修正了数据集中的某些偏差，从而提高了模型的性能。即使不使用点云编码，我们的模型在运行时间上比数值模型快一个数量级，在测试集上无碰撞解决方案的成功率可达90%。 

---
# SAMVAD: A Multi-Agent System for Simulating Judicial Deliberation Dynamics in India 

**Title (ZH)**: SAMVAD：一种模拟印度司法审议动力学的多agent系统 

**Authors**: Prathamesh Devadiga, Omkaar Jayadev Shetty, Pooja Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.03793)  

**Abstract**: Understanding the complexities of judicial deliberation is crucial for assessing the efficacy and fairness of a justice system. However, empirical studies of judicial panels are constrained by significant ethical and practical barriers. This paper introduces SAMVAD, an innovative Multi-Agent System (MAS) designed to simulate the deliberation process within the framework of the Indian justice system.
Our system comprises agents representing key judicial roles: a Judge, a Prosecution Counsel, a Defense Counsel, and multiple Adjudicators (simulating a judicial bench), all powered by large language models (LLMs). A primary contribution of this work is the integration of Retrieval-Augmented Generation (RAG), grounded in a domain-specific knowledge base of landmark Indian legal documents, including the Indian Penal Code and the Constitution of India. This RAG functionality enables the Judge and Counsel agents to generate legally sound instructions and arguments, complete with source citations, thereby enhancing both the fidelity and transparency of the simulation.
The Adjudicator agents engage in iterative deliberation rounds, processing case facts, legal instructions, and arguments to reach a consensus-based verdict. We detail the system architecture, agent communication protocols, the RAG pipeline, the simulation workflow, and a comprehensive evaluation plan designed to assess performance, deliberation quality, and outcome consistency.
This work provides a configurable and explainable MAS platform for exploring legal reasoning and group decision-making dynamics in judicial simulations, specifically tailored to the Indian legal context and augmented with verifiable legal grounding via RAG. 

**Abstract (ZH)**: 理解司法审议的复杂性对于评估司法系统的有效性与公正性至关重要。然而，对司法合议庭的实证研究受到重大伦理和实践障碍的限制。本文介绍了一种创新性的多智能体系统（MAS）——SAMVAD，用于模拟印度司法系统框架下的审议过程。 

---
# Learning an Adversarial World Model for Automated Curriculum Generation in MARL 

**Title (ZH)**: 学习对抗世界模型以实现自动课程生成在多智能体 reinforcement 学习中的应用 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.03771)  

**Abstract**: World models that infer and predict environmental dynamics are foundational to embodied intelligence. However, their potential is often limited by the finite complexity and implicit biases of hand-crafted training environments. To develop truly generalizable and robust agents, we need environments that scale in complexity alongside the agents learning within them. In this work, we reframe the challenge of environment generation as the problem of learning a goal-conditioned, generative world model. We propose a system where a generative **Attacker** agent learns an implicit world model to synthesize increasingly difficult challenges for a team of cooperative **Defender** agents. The Attacker's objective is not passive prediction, but active, goal-driven interaction: it models and generates world states (i.e., configurations of enemy units) specifically to exploit the Defenders' weaknesses. Concurrently, the embodied Defender team learns a cooperative policy to overcome these generated worlds. This co-evolutionary dynamic creates a self-scaling curriculum where the world model continuously adapts to challenge the decision-making policy of the agents, providing an effectively infinite stream of novel and relevant training scenarios. We demonstrate that this framework leads to the emergence of complex behaviors, such as the world model learning to generate flanking and shielding formations, and the defenders learning coordinated focus-fire and spreading tactics. Our findings position adversarial co-evolution as a powerful method for learning instrumental world models that drive agents toward greater strategic depth and robustness. 

**Abstract (ZH)**: 基于目标的生成式世界模型在生成性强健代理中的应用 

---
# Designing Gaze Analytics for ELA Instruction: A User-Centered Dashboard with Conversational AI Support 

**Title (ZH)**: 基于用户中心设计与对话式AI支持的ELA教学注视分析仪表板设计 

**Authors**: Eduardo Davalos, Yike Zhang, Shruti Jain, Namrata Srivastava, Trieu Truong, Nafees-ul Haque, Tristan Van, Jorge Salas, Sara McFadden, Sun-Joo Cho, Gautam Biswas, Amanda Goodwin  

**Link**: [PDF](https://arxiv.org/pdf/2509.03741)  

**Abstract**: Eye-tracking offers rich insights into student cognition and engagement, but remains underutilized in classroom-facing educational technology due to challenges in data interpretation and accessibility. In this paper, we present the iterative design and evaluation of a gaze-based learning analytics dashboard for English Language Arts (ELA), developed through five studies involving teachers and students. Guided by user-centered design and data storytelling principles, we explored how gaze data can support reflection, formative assessment, and instructional decision-making. Our findings demonstrate that gaze analytics can be approachable and pedagogically valuable when supported by familiar visualizations, layered explanations, and narrative scaffolds. We further show how a conversational agent, powered by a large language model (LLM), can lower cognitive barriers to interpreting gaze data by enabling natural language interactions with multimodal learning analytics. We conclude with design implications for future EdTech systems that aim to integrate novel data modalities in classroom contexts. 

**Abstract (ZH)**: 眼动追踪为洞察学生认知和参与提供了丰富的见解，但在面向课堂的教育技术中由于数据解释和获取的挑战仍被严重低估。本文介绍了通过五项涉及教师和学生的研究，迭代设计和评估的一种基于凝视的学习分析仪表板，用于英语语言艺术（ELA）的教学应用。受用户中心设计和数据叙事原则的指导，我们探讨了凝视数据如何支持反思、形成性评估和教学决策。研究结果表明，当结合熟悉的可视化、分层解释和叙事支架时，凝视分析可以变得易于理解和具有教育价值。我们进一步展示了由大型语言模型（LLM）驱动的对话代理如何通过使多模态学习分析中的自然语言交互成为可能，降低解析凝视数据的认知障碍。最后，我们提出了未来旨在课堂环境中整合新型数据模态的教育技术系统的设计启示。 

---
