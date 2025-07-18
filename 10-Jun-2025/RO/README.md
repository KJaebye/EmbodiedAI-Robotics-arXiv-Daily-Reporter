# BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models 

**Title (ZH)**: BridgeVLA：高效三维 manipulation 学习中的输入-输出对齐 

**Authors**: Peiyan Li, Yixiang Chen, Hongtao Wu, Xiao Ma, Xiangnan Wu, Yan Huang, Liang Wang, Tao Kong, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07961)  

**Abstract**: Recently, leveraging pre-trained vision-language models (VLMs) for building vision-language-action (VLA) models has emerged as a promising approach to effective robot manipulation learning. However, only few methods incorporate 3D signals into VLMs for action prediction, and they do not fully leverage the spatial structure inherent in 3D data, leading to low sample efficiency. In this paper, we introduce BridgeVLA, a novel 3D VLA model that (1) projects 3D inputs to multiple 2D images, ensuring input alignment with the VLM backbone, and (2) utilizes 2D heatmaps for action prediction, unifying the input and output spaces within a consistent 2D image space. In addition, we propose a scalable pre-training method that equips the VLM backbone with the capability to predict 2D heatmaps before downstream policy learning. Extensive experiments show the proposed method is able to learn 3D manipulation efficiently and effectively. BridgeVLA outperforms state-of-the-art baseline methods across three simulation benchmarks. In RLBench, it improves the average success rate from 81.4% to 88.2%. In COLOSSEUM, it demonstrates significantly better performance in challenging generalization settings, boosting the average success rate from 56.7% to 64.0%. In GemBench, it surpasses all the comparing baseline methods in terms of average success rate. In real-robot experiments, BridgeVLA outperforms a state-of-the-art baseline method by 32% on average. It generalizes robustly in multiple out-of-distribution settings, including visual disturbances and unseen instructions. Remarkably, it is able to achieve a success rate of 96.8% on 10+ tasks with only 3 trajectories per task, highlighting its extraordinary sample efficiency. Project Website:this https URL 

**Abstract (ZH)**: Recently, 利用预训练的视觉-语言模型构建视觉-语言-动作模型已成为有效的机器人操作学习的一种有前景的方法。然而，只有少数方法将3D信号整合到视觉-语言模型中进行动作预测，且未能充分利用3D数据中固有的空间结构，导致样本效率低。本文介绍了BridgeVLA，一种新颖的3D视觉-语言-动作模型，该模型通过（1）将3D输入投影到多个2D图像中，确保输入与视觉-语言模型主干对齐，以及（2）利用2D热图进行动作预测，在一致的2D图像空间内统一输入和输出空间。此外，本文提出了一种可扩展的预训练方法，使视觉-语言模型主干能够在下游策略学习之前预测2D热图。大量实验表明，所提出的方法能够高效有效地学习3D操作。BridgeVLA在三个模拟基准测试中均优于最新基线方法。在RLBenchmark中，它将平均成功率从81.4%提高到88.2%。在COLOSSEUM中，它在具有挑战性的泛化设置中表现出更出色的表现，将平均成功率从56.7%提高到64.0%。在GemBench中，它在平均成功率方面超过了所有比较基线方法。在真实机器人实验中，BridgeVLA在平均成功率方面比最新基线方法高出32%。它在包括视觉干扰和未见过的指令在内的多种分布外设置中表现出鲁棒的泛化能力。值得注意的是，它能够仅通过每任务3条轨迹实现96.8%的成功率，突显了其极高的样本效率。项目网站：this https URL。 

---
# Design and Implementation of a Peer-to-Peer Communication, Modular and Decentral YellowCube UUV 

**Title (ZH)**: 面向模块化与去中心化的Peer-to-Peer通信黄立方无人潜水器设计与实现 

**Authors**: Zhizun Xu, Baozhu Jia, Weichao Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07924)  

**Abstract**: The underwater Unmanned Vehicles(UUVs) are pivot tools for offshore engineering and oceanographic research. Most existing UUVs do not facilitate easy integration of new or upgraded sensors. A solution to this problem is to have a modular UUV system with changeable payload sections capable of carrying different sensor to suite different missions. The design and implementation of a modular and decentral UUV named YellowCube is presented in the paper. Instead a centralised software architecture which is adopted by the other modular underwater vehicles designs, a Peer-To-Peer(P2P) communication mechanism is implemented among the UUV's modules. The experiments in the laboratory and sea trials have been executed to verify the performances of the UUV. 

**Abstract (ZH)**: 模块化和去中心化的水下无人车辆YellowCube的设计与实现 

---
# Versatile Loco-Manipulation through Flexible Interlimb Coordination 

**Title (ZH)**: 灵活的肢体协作实现多功能移动操作 

**Authors**: Xinghao Zhu, Yuxin Chen, Lingfeng Sun, Farzad Niroui, Simon Le CleacH, Jiuguang Wang, Kuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07876)  

**Abstract**: The ability to flexibly leverage limbs for loco-manipulation is essential for enabling autonomous robots to operate in unstructured environments. Yet, prior work on loco-manipulation is often constrained to specific tasks or predetermined limb configurations. In this work, we present Reinforcement Learning for Interlimb Coordination (ReLIC), an approach that enables versatile loco-manipulation through flexible interlimb coordination. The key to our approach is an adaptive controller that seamlessly bridges the execution of manipulation motions and the generation of stable gaits based on task demands. Through the interplay between two controller modules, ReLIC dynamically assigns each limb for manipulation or locomotion and robustly coordinates them to achieve the task success. Using efficient reinforcement learning in simulation, ReLIC learns to perform stable gaits in accordance with the manipulation goals in the real world. To solve diverse and complex tasks, we further propose to interface the learned controller with different types of task specifications, including target trajectories, contact points, and natural language instructions. Evaluated on 12 real-world tasks that require diverse and complex coordination patterns, ReLIC demonstrates its versatility and robustness by achieving a success rate of 78.9% on average. Videos and code can be found at this https URL. 

**Abstract (ZH)**: 灵活利用肢体进行运动操作的能力是使自主机器人能够在未结构化环境中操作的关键。然而，现有的运动操作研究往往局限于特定任务或预设的肢体配置。在此工作中，我们提出了一种通过灵活的肢体协调实现多样化运动操作的方法——基于强化学习的肢体间协调（ReLIC）。我们方法的关键在于一个自适应控制器，该控制器能够无缝地将操作动作的执行与基于任务需求的稳定运动模式生成相结合。通过两个控制器模块之间的相互作用，ReLIC 动态地分配每个肢体用于操作或运动，并稳健地协调它们以实现任务的成功。通过在仿真中高效地使用强化学习，ReLIC 学习在实际任务中根据操作目标生成稳定的运动模式。为了解决各种复杂任务，我们进一步提出将所学习的控制器与不同类型的任务规范（包括目标轨迹、接触点和自然语言指令）进行接口连接。在对12个需要多样化和复杂协调模式的现实任务进行评估后，ReLIC展示了其多样性和鲁棒性，平均成功率达到了78.9%。更多视频和代码请访问此链接。 

---
# Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots 

**Title (ZH)**: 基于原始对偶iLQR的GPU加速腿足机器人学习与控制 

**Authors**: Lorenzo Amatucci, João Sousa-Pinto, Giulio Turrisi, Dominique Orban, Victor Barasuol, Claudio Semini  

**Link**: [PDF](https://arxiv.org/pdf/2506.07823)  

**Abstract**: This paper introduces a novel Model Predictive Control (MPC) implementation for legged robot locomotion that leverages GPU parallelization. Our approach enables both temporal and state-space parallelization by incorporating a parallel associative scan to solve the primal-dual Karush-Kuhn-Tucker (KKT) system. In this way, the optimal control problem is solved in $\mathcal{O}(n\log{N} + m)$ complexity, instead of $\mathcal{O}(N(n + m)^3)$, where $n$, $m$, and $N$ are the dimension of the system state, control vector, and the length of the prediction horizon. We demonstrate the advantages of this implementation over two state-of-the-art solvers (acados and crocoddyl), achieving up to a 60\% improvement in runtime for Whole Body Dynamics (WB)-MPC and a 700\% improvement for Single Rigid Body Dynamics (SRBD)-MPC when varying the prediction horizon length. The presented formulation scales efficiently with the problem state dimensions as well, enabling the definition of a centralized controller for up to 16 legged robots that can be computed in less than 25 ms. Furthermore, thanks to the JAX implementation, the solver supports large-scale parallelization across multiple environments, allowing the possibility of performing learning with the MPC in the loop directly in GPU. 

**Abstract (ZH)**: 一种基于GPU并行化的腿式机器人运动新型模型预测控制实现 

---
# SMaRCSim: Maritime Robotics Simulation Modules 

**Title (ZH)**: SMaRCSim: 海洋机器人仿真模块 

**Authors**: Mart Kartašev, David Dörner, Özer Özkahraman, Petter Ögren, Ivan Stenius, John Folkesson  

**Link**: [PDF](https://arxiv.org/pdf/2506.07781)  

**Abstract**: Developing new functionality for underwater robots and testing them in the real world is time-consuming and resource-intensive. Simulation environments allow for rapid testing before field deployment. However, existing tools lack certain functionality for use cases in our project: i) developing learning-based methods for underwater vehicles; ii) creating teams of autonomous underwater, surface, and aerial vehicles; iii) integrating the simulation with mission planning for field experiments. A holistic solution to these problems presents great potential for bringing novel functionality into the underwater domain. In this paper we present SMaRCSim, a set of simulation packages that we have developed to help us address these issues. 

**Abstract (ZH)**: 开发水下机器人的新功能并在实际环境中测试是耗时且资源密集的。仿真环境允许在实地部署前快速测试。然而，现有工具缺乏我们项目中某些使用案例所需的特定功能：i) 开发基于学习的方法用于水下车辆；ii) 创建自主水下、水面和空中车辆的团队；iii) 将仿真与实地试验的使命规划集成。为这些难题提供一个整体解决方案具有将新颖功能引入水下领域的巨大潜力。在本文中，我们介绍了我们开发的一套仿真包SMaRCSim，以帮助我们解决这些问题。 

---
# A Communication-Latency-Aware Co-Simulation Platform for Safety and Comfort Evaluation of Cloud-Controlled ICVs 

**Title (ZH)**: 一种通信时延感知的联合仿真平台：用于云控制的ICVs安全与舒适性评估 

**Authors**: Yongqi Zhao, Xinrui Zhang, Tomislav Mihalj, Martin Schabauer, Luis Putzer, Erik Reichmann-Blaga, Ádám Boronyák, András Rövid, Gábor Soós, Peizhi Zhang, Lu Xiong, Jia Hu, Arno Eichberger  

**Link**: [PDF](https://arxiv.org/pdf/2506.07696)  

**Abstract**: Testing cloud-controlled intelligent connected vehicles (ICVs) requires simulation environments that faithfully emulate both vehicle behavior and realistic communication latencies. This paper proposes a latency-aware co-simulation platform integrating CarMaker and Vissim to evaluate safety and comfort under real-world vehicle-to-cloud (V2C) latency conditions. Two communication latency models, derived from empirical 5G measurements in China and Hungary, are incorporated and statistically modeled using Gamma distributions. A proactive conflict module (PCM) is proposed to dynamically control background vehicles and generate safety-critical scenarios. The platform is validated through experiments involving an exemplary system under test (SUT) across six testing conditions combining two PCM modes (enabled/disabled) and three latency conditions (none, China, Hungary). Safety and comfort are assessed using metrics including collision rate, distance headway, post-encroachment time, and the spectral characteristics of longitudinal acceleration. Results show that the PCM effectively increases driving environment criticality, while V2C latency primarily affects ride comfort. These findings confirm the platform's effectiveness in systematically evaluating cloud-controlled ICVs under diverse testing conditions. 

**Abstract (ZH)**: 基于延迟感知的集成CarMaker和Vissim协同仿真平台：评估实际车辆到云端（V2C）延迟条件下的安全性和舒适性 

---
# Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse 

**Title (ZH)**: Fast ECoT: 通过思考重用的高效体态链式思维 

**Authors**: Zhekai Duan, Yuan Zhang, Shikai Geng, Gaowen Liu, Joschka Boedecker, Chris Xiaoxuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07639)  

**Abstract**: Embodied Chain-of-Thought (ECoT) reasoning enhances vision-language-action (VLA) models by improving performance and interpretability through intermediate reasoning steps. However, its sequential autoregressive token generation introduces significant inference latency, limiting real-time deployment. We propose Fast ECoT, an inference-time acceleration method that exploits the structured and repetitive nature of ECoT to (1) cache and reuse high-level reasoning across timesteps and (2) parallelise the generation of modular reasoning steps. Additionally, we introduce an asynchronous scheduler that decouples reasoning from action decoding, further boosting responsiveness. Fast ECoT requires no model changes or additional training and integrates easily into existing VLA pipelines. Experiments in both simulation (LIBERO) and real-world robot tasks show up to a 7.5% reduction in latency with comparable or improved task success rate and reasoning faithfulness, bringing ECoT policies closer to practical real-time deployment. 

**Abstract (ZH)**: 富含实体的链式思维(Fast ECoT)推理通过中间推理步骤提高视觉-语言-行动(VLA)模型的性能和可解释性，但其 sequential 自回归标记生成引入了显著的推理延迟，限制了实时部署。我们提出了一种在推理时加速的方法 Fast ECoT，该方法利用了 ECoT 的结构化和重复性特征，(1) 缓存并跨时间步重用高层次推理，(2) 并行生成模块化推理步骤。此外，我们引入了一个异步调度器，将推理与动作解码解耦，进一步提升了响应性。Fast ECoT 不需要对模型进行更改或额外训练，并且可以轻松集成到现有的 VLA 管道中。在模拟环境(LIBERO)和真实世界机器人任务中的实验显示，与可比或改进的任务成功率和推理正确性相比，延迟最多可减少 7.5%，使 ECoT 策略更接近实际的实时部署。 

---
# Blending Participatory Design and Artificial Awareness for Trustworthy Autonomous Vehicles 

**Title (ZH)**: 融合参与设计与人工意识以提高自主车辆的可信度 

**Authors**: Ana Tanevska, Ananthapathmanabhan Ratheesh Kumar, Arabinda Ghosh, Ernesto Casablanca, Ginevra Castellano, Sadegh Soudjani  

**Link**: [PDF](https://arxiv.org/pdf/2506.07633)  

**Abstract**: Current robotic agents, such as autonomous vehicles (AVs) and drones, need to deal with uncertain real-world environments with appropriate situational awareness (SA), risk awareness, coordination, and decision-making. The SymAware project strives to address this issue by designing an architecture for artificial awareness in multi-agent systems, enabling safe collaboration of autonomous vehicles and drones. However, these agents will also need to interact with human users (drivers, pedestrians, drone operators), which in turn requires an understanding of how to model the human in the interaction scenario, and how to foster trust and transparency between the agent and the human.
In this work, we aim to create a data-driven model of a human driver to be integrated into our SA architecture, grounding our research in the principles of trustworthy human-agent interaction. To collect the data necessary for creating the model, we conducted a large-scale user-centered study on human-AV interaction, in which we investigate the interaction between the AV's transparency and the users' behavior.
The contributions of this paper are twofold: First, we illustrate in detail our human-AV study and its findings, and second we present the resulting Markov chain models of the human driver computed from the study's data. Our results show that depending on the AV's transparency, the scenario's environment, and the users' demographics, we can obtain significant differences in the model's transitions. 

**Abstract (ZH)**: 当前的机器人代理，如自动驾驶车辆（AVs）和无人机，需要在不确定的现实环境中具备适当的情境意识（SA）、风险意识、协调和决策能力。SymAware项目致力于通过为多智能体系统设计人工意识架构，促进自动驾驶车辆和无人机的安全协作。然而，这些代理还将需要与人类用户（驾驶员、行人、无人机操作员）互动，这就要求我们理解如何在交互场景中建模人类行为，并促进代理与人类之间的信任和透明度。

在本文中，我们旨在创建一个基于数据的人类驾驶员模型，将其集成到我们的SA架构中，以实现可信赖的人机交互。为了收集创建模型所必需的数据，我们进行了大规模的以用户为中心的研究，探讨了AV透明度与用户行为之间的交互。

本文的贡献主要有两点：首先，我们详细介绍了人类-AV研究及其发现；其次，我们展示了从研究数据中计算得到的人类驾驶员马尔可夫链模型。我们的结果显示，根据不同AV的透明度、场景的环境以及用户的人口统计特征，模型的状态转换会存在显著差异。 

---
# Fractional Collisions: A Framework for Risk Estimation of Counterfactual Conflicts using Autonomous Driving Behavior Simulations 

**Title (ZH)**: 分数碰撞：一种基于自动驾驶行为仿真对抗事实冲突风险估计的框架 

**Authors**: Sreeja Roy-Singh, Sarvesh Kolekar, Daniel P. Bonny, Kyle Foss  

**Link**: [PDF](https://arxiv.org/pdf/2506.07540)  

**Abstract**: We present a methodology for estimating collision risk from counterfactual simulated scenarios built on sensor data from automated driving systems (ADS) or naturalistic driving databases. Two-agent conflicts are assessed by detecting and classifying conflict type, identifying the agents' roles (initiator or responder), identifying the point of reaction of the responder, and modeling their human behavioral expectations as probabilistic counterfactual trajectories. The states are used to compute velocity differentials at collision, which when combined with crash models, estimates severity of loss in terms of probabilistic injury or property damage, henceforth called fractional collisions. The probabilistic models may also be extended to include other uncertainties associated with the simulation, features, and agents. We verify the effectiveness of the methodology in a synthetic simulation environment using reconstructed trajectories from 300+ collision and near-collision scenes sourced from VTTI's SHRP2 database and Nexar dashboard camera data. Our methodology predicted fractional collisions within 1% of ground truth collisions. We then evaluate agent-initiated collision risk of an arbitrary ADS software release by replacing the naturalistic responder in these synthetic reconstructions with an ADS simulator and comparing the outcome to human-response outcomes. Our ADS reduced naturalistic collisions by 4x and fractional collision risk by ~62%. The framework's utility is also demonstrated on 250k miles of proprietary, open-loop sensor data collected on ADS test vehicles, re-simulated with an arbitrary ADS software release. The ADS initiated conflicts that caused 0.4 injury-causing and 1.7 property-damaging fractional collisions, and the ADS improved collision risk in 96% of the agent-initiated conflicts. 

**Abstract (ZH)**: 基于传感器数据的自动驾驶系统中碰撞风险估计方法 

---
# BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation 

**Title (ZH)**: BitVLA: 1-bit Vision-Language-Action模型用于机器人 manipulation 

**Authors**: Hongyu Wang, Chuyan Xiong, Ruiping Wang, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07530)  

**Abstract**: Vision-Language-Action (VLA) models have shown impressive capabilities across a wide range of robotics manipulation tasks. However, their growing model size poses significant challenges for deployment on resource-constrained robotic systems. While 1-bit pretraining has proven effective for enhancing the inference efficiency of large language models with minimal performance loss, its application to VLA models remains underexplored. In this work, we present BitVLA, the first 1-bit VLA model for robotics manipulation, in which every parameter is ternary, i.e., {-1, 0, 1}. To further reduce the memory footprint of the vision encoder, we propose the distillation-aware training strategy that compresses the full-precision encoder to 1.58-bit weights. During this process, a full-precision encoder serves as a teacher model to better align latent representations. Despite the lack of large-scale robotics pretraining, BitVLA achieves performance comparable to the state-of-the-art model OpenVLA-OFT with 4-bit post-training quantization on the LIBERO benchmark, while consuming only 29.8% of the memory. These results highlight BitVLA's promise for deployment on memory-constrained edge devices. We release the code and model weights in this https URL. 

**Abstract (ZH)**: 面向机器人操作的1比特Vision-Language-Action (VLA) 模型 

---
# Taking Flight with Dialogue: Enabling Natural Language Control for PX4-based Drone Agent 

**Title (ZH)**: 凭借对话起飞：为基于PX4的无人机代理启用自然语言控制 

**Authors**: Shoon Kit Lim, Melissa Jia Ying Chong, Jing Huey Khor, Ting Yang Ling  

**Link**: [PDF](https://arxiv.org/pdf/2506.07509)  

**Abstract**: Recent advances in agentic and physical artificial intelligence (AI) have largely focused on ground-based platforms such as humanoid and wheeled robots, leaving aerial robots relatively underexplored. Meanwhile, state-of-the-art unmanned aerial vehicle (UAV) multimodal vision-language systems typically rely on closed-source models accessible only to well-resourced organizations. To democratize natural language control of autonomous drones, we present an open-source agentic framework that integrates PX4-based flight control, Robot Operating System 2 (ROS 2) middleware, and locally hosted models using Ollama. We evaluate performance both in simulation and on a custom quadcopter platform, benchmarking four large language model (LLM) families for command generation and three vision-language model (VLM) families for scene understanding. 

**Abstract (ZH)**: 近期在代理人和物理人工智能领域的进展主要集中在地面平台如人形机器人和轮式机器人上，而对飞行机器人则相对探索较少。与此同时，最先进的无人驾驶飞行器（UAV）多模态视觉-语言系统通常依赖于仅对资源充足的组织开放的闭源模型。为了使自然语言控制自主无人机的应用更加普及，我们提出了一种开源代理人框架，该框架结合了基于PX4的飞行控制、Robot Operating System 2（ROS 2）中间件以及使用Ollama托管的本地模型。我们在模拟和自定义四旋翼飞行器平台上评估了性能，并对四种大型语言模型（LLM）家族的命令生成和三种视觉-语言模型（VLM）家族的场景理解进行了基准测试。 

---
# RAPID Hand: A Robust, Affordable, Perception-Integrated, Dexterous Manipulation Platform for Generalist Robot Autonomy 

**Title (ZH)**: RAPID 手：一种 robust、经济、感知集成、灵巧的操作平台，适用于通用机器人自主性 

**Authors**: Zhaoliang Wan, Zetong Bi, Zida Zhou, Hao Ren, Yiming Zeng, Yihan Li, Lu Qi, Xu Yang, Ming-Hsuan Yang, Hui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07490)  

**Abstract**: This paper addresses the scarcity of low-cost but high-dexterity platforms for collecting real-world multi-fingered robot manipulation data towards generalist robot autonomy. To achieve it, we propose the RAPID Hand, a co-optimized hardware and software platform where the compact 20-DoF hand, robust whole-hand perception, and high-DoF teleoperation interface are jointly designed. Specifically, RAPID Hand adopts a compact and practical hand ontology and a hardware-level perception framework that stably integrates wrist-mounted vision, fingertip tactile sensing, and proprioception with sub-7 ms latency and spatial alignment. Collecting high-quality demonstrations on high-DoF hands is challenging, as existing teleoperation methods struggle with precision and stability on complex multi-fingered systems. We address this by co-optimizing hand design, perception integration, and teleoperation interface through a universal actuation scheme, custom perception electronics, and two retargeting constraints. We evaluate the platform's hardware, perception, and teleoperation interface. Training a diffusion policy on collected data shows superior performance over prior works, validating the system's capability for reliable, high-quality data collection. The platform is constructed from low-cost and off-the-shelf components and will be made public to ensure reproducibility and ease of adoption. 

**Abstract (ZH)**: 本文解决了低本钱但高灵巧度平台收集现实世界多指机器人操作数据以实现通用机器人自主性的稀缺问题。为此，我们提出了一种协同优化硬件和软件平台——RAPID手，该平台集成了紧凑的20自由度手、稳健的整体手部感知以及高自由度远程操作界面。具体而言，RAPID手采用了紧凑且实用的手部本体论和硬件级感知框架，该框架能够稳定地整合腕部摄像头、指尖触觉感知和 proprioception，且具有亚7毫秒的延迟和空间对齐。在高自由度手上收集高质量演示动作具有挑战性，现有远程操作方法在复杂多指系统上难以实现精确性和稳定性。我们通过通用驱动方案、定制感知电子学以及两套适配约束，协同优化了手部设计、感知集成和远程操作界面。我们评估了该平台的硬件、感知能力和远程操作界面。基于收集的数据训练扩散策略显示了优于现有工作的性能，验证了该系统可靠地收集高质量数据的能力。该平台由低成本和现成组件构建，并将公开发布以确保可重复性和易于采用。 

---
# Language-Grounded Hierarchical Planning and Execution with Multi-Robot 3D Scene Graphs 

**Title (ZH)**: 基于语言引导的多机器人多层次规划与执行及多维场景图 

**Authors**: Jared Strader, Aaron Ray, Jacob Arkin, Mason B. Peterson, Yun Chang, Nathan Hughes, Christopher Bradley, Yi Xuan Jia, Carlos Nieto-Granda, Rajat Talak, Chuchu Fan, Luca Carlone, Jonathan P. How, Nicholas Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.07454)  

**Abstract**: In this paper, we introduce a multi-robot system that integrates mapping, localization, and task and motion planning (TAMP) enabled by 3D scene graphs to execute complex instructions expressed in natural language. Our system builds a shared 3D scene graph incorporating an open-set object-based map, which is leveraged for multi-robot 3D scene graph fusion. This representation supports real-time, view-invariant relocalization (via the object-based map) and planning (via the 3D scene graph), allowing a team of robots to reason about their surroundings and execute complex tasks. Additionally, we introduce a planning approach that translates operator intent into Planning Domain Definition Language (PDDL) goals using a Large Language Model (LLM) by leveraging context from the shared 3D scene graph and robot capabilities. We provide an experimental assessment of the performance of our system on real-world tasks in large-scale, outdoor environments. 

**Abstract (ZH)**: 本文介绍了集成基于3D场景图的地图构建、定位及任务与运动规划（TAMP）功能的多机器人系统，以执行自然语言表达的复杂指令。我们的系统构建了一个共享的3D场景图，包含开放式对象地图，用于多机器人3D场景图融合。该表示支持通过对象地图实现的实时、视角不变的再定位和基于3D场景图的规划，使得机器人团队能够理解和执行复杂任务。此外，我们还提出了一种规划方法，利用大型语言模型（LLM）和共享的3D场景图及机器人能力上下文，将操作员的意图转换为规划领域定义语言（PDDL）的目标。我们在大规模室外环境中的实际任务中对系统性能进行了实验评估。 

---
# MapBERT: Bitwise Masked Modeling for Real-Time Semantic Mapping Generation 

**Title (ZH)**: MapBERT：位级掩码建模以实现实时语义地图生成 

**Authors**: Yijie Deng, Shuaihang Yuan, Congcong Wen, Hao Huang, Anthony Tzes, Geeta Chandra Raju Bethala, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07350)  

**Abstract**: Spatial awareness is a critical capability for embodied agents, as it enables them to anticipate and reason about unobserved regions. The primary challenge arises from learning the distribution of indoor semantics, complicated by sparse, imbalanced object categories and diverse spatial scales. Existing methods struggle to robustly generate unobserved areas in real time and do not generalize well to new environments. To this end, we propose \textbf{MapBERT}, a novel framework designed to effectively model the distribution of unseen spaces. Motivated by the observation that the one-hot encoding of semantic maps aligns naturally with the binary structure of bit encoding, we, for the first time, leverage a lookup-free BitVAE to encode semantic maps into compact bitwise tokens. Building on this, a masked transformer is employed to infer missing regions and generate complete semantic maps from limited observations. To enhance object-centric reasoning, we propose an object-aware masking strategy that masks entire object categories concurrently and pairs them with learnable embeddings, capturing implicit relationships between object embeddings and spatial tokens. By learning these relationships, the model more effectively captures indoor semantic distributions crucial for practical robotic tasks. Experiments on Gibson benchmarks show that MapBERT achieves state-of-the-art semantic map generation, balancing computational efficiency with accurate reconstruction of unobserved regions. 

**Abstract (ZH)**: 基于空间意识的MapBERT：一种用于有效建模未见空间分布的新框架 

---
# UruBots Autonomous Cars Challenge Pro Team Description Paper for FIRA 2025 

**Title (ZH)**: UruBots自主汽车挑战专业团队描述论文：FIRA 2025 

**Authors**: Pablo Moraes, Mónica Rodríguez, Sebastian Barcelona, Angel Da Silva, Santiago Fernandez, Hiago Sodre, Igor Nunes, Bruna Guterres, Ricardo Grando  

**Link**: [PDF](https://arxiv.org/pdf/2506.07348)  

**Abstract**: This paper describes the development of an autonomous car by the UruBots team for the 2025 FIRA Autonomous Cars Challenge (Pro). The project involves constructing a compact electric vehicle, approximately the size of an RC car, capable of autonomous navigation through different tracks. The design incorporates mechanical and electronic components and machine learning algorithms that enable the vehicle to make real-time navigation decisions based on visual input from a camera. We use deep learning models to process camera images and control vehicle movements. Using a dataset of over ten thousand images, we trained a Convolutional Neural Network (CNN) to drive the vehicle effectively, through two outputs, steering and throttle. The car completed the track in under 30 seconds, achieving a pace of approximately 0.4 meters per second while avoiding obstacles. 

**Abstract (ZH)**: 乌鲁机器人团队2025年FIRA自主汽车挑战赛（专业组）的自主汽车开发研究 

---
# Reproducibility in the Control of Autonomous Mobility-on-Demand Systems 

**Title (ZH)**: 自主出行系统控制中的可重复性研究 

**Authors**: Xinling Li, Meshal Alharbi, Daniele Gammelli, James Harrison, Filipe Rodrigues, Maximilian Schiffer, Marco Pavone, Emilio Frazzoli, Jinhua Zhao, Gioele Zardini  

**Link**: [PDF](https://arxiv.org/pdf/2506.07345)  

**Abstract**: Autonomous Mobility-on-Demand (AMoD) systems, powered by advances in robotics, control, and Machine Learning (ML), offer a promising paradigm for future urban transportation. AMoD offers fast and personalized travel services by leveraging centralized control of autonomous vehicle fleets to optimize operations and enhance service performance. However, the rapid growth of this field has outpaced the development of standardized practices for evaluating and reporting results, leading to significant challenges in reproducibility. As AMoD control algorithms become increasingly complex and data-driven, a lack of transparency in modeling assumptions, experimental setups, and algorithmic implementation hinders scientific progress and undermines confidence in the results. This paper presents a systematic study of reproducibility in AMoD research. We identify key components across the research pipeline, spanning system modeling, control problems, simulation design, algorithm specification, and evaluation, and analyze common sources of irreproducibility. We survey prevalent practices in the literature, highlight gaps, and propose a structured framework to assess and improve reproducibility. Specifically, concrete guidelines are offered, along with a "reproducibility checklist", to support future work in achieving replicable, comparable, and extensible results. While focused on AMoD, the principles and practices we advocate generalize to a broader class of cyber-physical systems that rely on networked autonomy and data-driven control. This work aims to lay the foundation for a more transparent and reproducible research culture in the design and deployment of intelligent mobility systems. 

**Abstract (ZH)**: 基于自主机器人、控制和机器学习技术的按需自主移动（AMoD）系统为未来城市交通提供了有希望的范式。本文系统研究了AMoD研究中的可重复性问题，识别了研究管道中的关键组件，包括系统建模、控制问题、仿真设计、算法规范和评估，并分析了不可重复性的常见来源。本文还概述了具体的指导意见，并提出了“可重复性检查表”来支持未来工作中实现可重复、可比较和可扩展的结果。虽然重点是AMoD系统，但所倡导的原则和实践适用于更广泛依赖网络自主性和数据驱动控制的网络物理系统。这项工作旨在为智能移动系统的设计和部署建立一个更透明和可重复的研究文化奠定基础。 

---
# Real-Time Execution of Action Chunking Flow Policies 

**Title (ZH)**: 实时执行行动Chunking流程策略 

**Authors**: Kevin Black, Manuel Y. Galliker, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2506.07339)  

**Abstract**: Modern AI systems, especially those interacting with the physical world, increasingly require real-time performance. However, the high latency of state-of-the-art generalist models, including recent vision-language action models (VLAs), poses a significant challenge. While action chunking has enabled temporal consistency in high-frequency control tasks, it does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries. This paper presents a novel inference-time algorithm that enables smooth asynchronous execution of action chunking policies. Our method, real-time chunking (RTC), is applicable to any diffusion- or flow-based VLA out of the box with no re-training. It generates the next action chunk while executing the current one, "freezing" actions guaranteed to execute and "inpainting" the rest. To test RTC, we introduce a new benchmark of 12 highly dynamic tasks in the Kinetix simulator, as well as evaluate 6 challenging real-world bimanual manipulation tasks. Results demonstrate that RTC is fast, performant, and uniquely robust to inference delay, significantly improving task throughput and enabling high success rates in precise tasks $\unicode{x2013}$ such as lighting a match $\unicode{x2013}$ even in the presence of significant latency. See this https URL for videos. 

**Abstract (ZH)**: 现代AI系统，尤其是那些与物理世界交互的系统，越来越需要实时性能。然而，最先进的通用于物理世界的模型，包括近期的视觉-语言动作模型（VLAs），所面临的高延迟问题构成了一个重大挑战。尽管动作切片化能够确保高频率控制任务中的时间一致性，但它并不能完全解决延迟问题，在动作切片边界处会导致暂停或分布外的不流畅运动。本文提出了一种新颖的推理时算法，能够实现动作切片策略的平滑异步执行。我们的方法实时切片（RTC）可以直接应用于任何基于扩散或流的VLAs，无需重新训练。在执行当前动作切片的同时生成下一个动作切片，其方法是冻结确保执行的动作，并对剩余部分进行“填充”。为了测试RTC，我们引入了Kinetix模拟器中的一项新的包含12项高度动态任务的基准，并评估了6项具挑战性的实时双臂操纵任务。结果表明，RTC快速、高效，并且对推理延迟具有独特鲁棒性，显著提高了任务吞吐量，并即使在存在显著延迟的情况下也能在精确的任务中（如点火柴）实现高成功率。请参见此链接获取视频：https://xxxxxx 

---
# BR-MPPI: Barrier Rate guided MPPI for Enforcing Multiple Inequality Constraints with Learned Signed Distance Field 

**Title (ZH)**: BR-MPPI: 障碍率引导的MPPI方法，用于结合学习到的符号距离场强制执行多个不等式约束 

**Authors**: Hardik Parwana, Taekyung Kim, Kehan Long, Bardh Hoxha, Hideki Okamoto, Georgios Fainekos, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07325)  

**Abstract**: Model Predictive Path Integral (MPPI) controller is used to solve unconstrained optimal control problems and Control Barrier Function (CBF) is a tool to impose strict inequality constraints, a.k.a, barrier constraints. In this work, we propose an integration of these two methods that employ CBF-like conditions to guide the control sampling procedure of MPPI. CBFs provide an inequality constraint restricting the rate of change of barrier functions by a classK function of the barrier itself. We instead impose the CBF condition as an equality constraint by choosing a parametric linear classK function and treating this parameter as a state in an augmented system. The time derivative of this parameter acts as an additional control input that is designed by MPPI. A cost function is further designed to reignite Nagumo's theorem at the boundary of the safe set by promoting specific values of classK parameter to enforce safety. Our problem formulation results in an MPPI subject to multiple state and control-dependent equality constraints which are non-trivial to satisfy with randomly sampled control inputs. We therefore also introduce state transformations and control projection operations, inspired by the literature on path planning for manifolds, to resolve the aforementioned issue. We show empirically through simulations and experiments on quadrotor that our proposed algorithm exhibits better sampled efficiency and enhanced capability to operate closer to the safe set boundary over vanilla MPPI. 

**Abstract (ZH)**: 基于MPPI的MPFI控制器结合控制屏障函数的方法 

---
# Very Large-scale Multi-Robot Task Allocation in Challenging Environments via Robot Redistribution 

**Title (ZH)**: 挑战环境下的大规模多机器人任务分配通过机器人重新分配 

**Authors**: Seabin Lee, Joonyeol Sim, Changjoo Nam  

**Link**: [PDF](https://arxiv.org/pdf/2506.07293)  

**Abstract**: We consider the Multi-Robot Task Allocation (MRTA) problem that aims to optimize an assignment of multiple robots to multiple tasks in challenging environments which are with densely populated obstacles and narrow passages. In such environments, conventional methods optimizing the sum-of-cost are often ineffective because the conflicts between robots incur additional costs (e.g., collision avoidance, waiting). Also, an allocation that does not incorporate the actual robot paths could cause deadlocks, which significantly degrade the collective performance of the robots.
We propose a scalable MRTA method that considers the paths of the robots to avoid collisions and deadlocks which result in a fast completion of all tasks (i.e., minimizing the \textit{makespan}). To incorporate robot paths into task allocation, the proposed method constructs a roadmap using a Generalized Voronoi Diagram. The method partitions the roadmap into several components to know how to redistribute robots to achieve all tasks with less conflicts between the robots. In the redistribution process, robots are transferred to their final destinations according to a push-pop mechanism with the first-in first-out principle. From the extensive experiments, we show that our method can handle instances with hundreds of robots in dense clutter while competitors are unable to compute a solution within a time limit. 

**Abstract (ZH)**: 多机器人任务分配中的路径考虑方法：避免碰撞与死锁以优化完成任务时间 

---
# Model Analysis And Design Of Ellipse Based Segmented Varying Curved Foot For Biped Robot Walking 

**Title (ZH)**: 基于椭圆分段变曲率足的 biped 机器人行走模型分析与设计 

**Authors**: Boyang Chen, Xizhe Zang, Chao Song, Yue Zhang, Jie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07283)  

**Abstract**: This paper presents the modeling, design, and experimental validation of an Ellipse-based Segmented Varying Curvature (ESVC) foot for bipedal robots. Inspired by the segmented curvature rollover shape of human feet, the ESVC foot aims to enhance gait energy efficiency while maintaining analytical tractability for foot location based controller. First, we derive a complete analytical contact model for the ESVC foot by formulating spatial transformations of elliptical segments only using elementary functions. Then a nonlinear programming approach is engaged to determine optimal elliptical parameters of hind foot and fore foot based on a known mid-foot. An error compensation method is introduced to address approximation inaccuracies in rollover length calculation. The proposed ESVC foot is then integrated with a Hybrid Linear Inverted Pendulum model-based walking controller and validated through both simulation and physical experiments on the TT II biped robot. Experimental results across marking time, sagittal, and lateral walking tasks show that the ESVC foot consistently reduces energy consumption compared to line, and flat feet, with up to 18.52\% improvement in lateral walking. These findings demonstrate that the ESVC foot provides a practical and energy-efficient alternative for real-world bipedal locomotion. The proposed design methodology also lays a foundation for data-driven foot shape optimization in future research. 

**Abstract (ZH)**: 基于椭圆分段变曲率foot的设计与实验验证：提升双足机器人步态能量效率的新方法 

---
# Machine Learning-Based Self-Localization Using Internal Sensors for Automating Bulldozers 

**Title (ZH)**: 基于机器学习的内部传感器自我定位方法及其在自动化推土机中的应用 

**Authors**: Hikaru Sawafuji, Ryota Ozaki, Takuto Motomura, Toyohisa Matsuda, Masanori Tojima, Kento Uchida, Shinichi Shirakawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.07271)  

**Abstract**: Self-localization is an important technology for automating bulldozers. Conventional bulldozer self-localization systems rely on RTK-GNSS (Real Time Kinematic-Global Navigation Satellite Systems). However, RTK-GNSS signals are sometimes lost in certain mining conditions. Therefore, self-localization methods that do not depend on RTK-GNSS are required. In this paper, we propose a machine learning-based self-localization method for bulldozers. The proposed method consists of two steps: estimating local velocities using a machine learning model from internal sensors, and incorporating these estimates into an Extended Kalman Filter (EKF) for global localization. We also created a novel dataset for bulldozer odometry and conducted experiments across various driving scenarios, including slalom, excavation, and driving on slopes. The result demonstrated that the proposed self-localization method suppressed the accumulation of position errors compared to kinematics-based methods, especially when slip occurred. Furthermore, this study showed that bulldozer-specific sensors, such as blade position sensors and hydraulic pressure sensors, contributed to improving self-localization accuracy. 

**Abstract (ZH)**: 基于机器学习的推土机自定位方法 

---
# MorphoCopter: Design, Modeling, and Control of a New Transformable Quad-Bi Copter 

**Title (ZH)**: MorphoCopter：新型变形四轴双旋翼飞行器的设计、建模与控制 

**Authors**: Harsh Modi, Hao Su, Xiao Liang, Minghui Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07204)  

**Abstract**: This paper presents a novel morphing quadrotor, named MorphoCopter, covering its design, modeling, control, and experimental tests. It features a unique single rotary joint that enables rapid transformation into an ultra-narrow profile. Although quadrotors have seen widespread adoption in applications such as cinematography, agriculture, and disaster management with increasingly sophisticated control systems, their hardware configurations have remained largely unchanged, limiting their capabilities in certain environments. Our design addresses this by enabling the hardware configuration to change on the fly when required. In standard flight mode, the MorphoCopter adopts an X configuration, functioning as a traditional quadcopter, but can quickly fold into a stacked bicopters arrangement or any configuration in between. Existing morphing designs often sacrifice controllability in compact configurations or rely on complex multi-joint systems. Moreover, our design achieves a greater width reduction than any existing solution. We develop a new inertia and control-action aware adaptive control system that maintains robust performance across all rotary-joint configurations. The prototype can reduce its width from 447 mm to 138 mm (nearly 70\% reduction) in just a few seconds. We validated the MorphoCopter through rigorous simulations and a comprehensive series of flight experiments, including robustness tests, trajectory tracking, and narrow-gap passing tests. 

**Abstract (ZH)**: 一种新型变形四旋翼机MorphoCopter的设计、建模、控制及实验研究 

---
# Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset 

**Title (ZH)**: 改善Waymo开放运动数据集中的交通信号数据质量 

**Authors**: Xintao Yan, Erdao Liang, Jiawei Wang, Haojie Zhu, Henry X. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07150)  

**Abstract**: Datasets pertaining to autonomous vehicles (AVs) hold significant promise for a range of research fields, including artificial intelligence (AI), autonomous driving, and transportation engineering. Nonetheless, these datasets often encounter challenges related to the states of traffic signals, such as missing or inaccurate data. Such issues can compromise the reliability of the datasets and adversely affect the performance of models developed using them. This research introduces a fully automated approach designed to tackle these issues by utilizing available vehicle trajectory data alongside knowledge from the transportation domain to effectively impute and rectify traffic signal information within the Waymo Open Motion Dataset (WOMD). The proposed method is robust and flexible, capable of handling diverse intersection geometries and traffic signal configurations in real-world scenarios. Comprehensive validations have been conducted on the entire WOMD, focusing on over 360,000 relevant scenarios involving traffic signals, out of a total of 530,000 real-world driving scenarios. In the original dataset, 71.7% of traffic signal states are either missing or unknown, all of which were successfully imputed by our proposed method. Furthermore, in the absence of ground-truth signal states, the accuracy of our approach is evaluated based on the rate of red-light violations among vehicle trajectories. Results show that our method reduces the estimated red-light running rate from 15.7% in the original data to 2.9%, thereby demonstrating its efficacy in rectifying data inaccuracies. This paper significantly enhances the quality of AV datasets, contributing to the wider AI and AV research communities and benefiting various downstream applications. The code and improved traffic signal data are open-sourced at this https URL 

**Abstract (ZH)**: 自主驾驶车辆数据集在人工智能、自动驾驶和交通工程领域的潜在应用及其改进方法 

---
# Robotic Policy Learning via Human-assisted Action Preference Optimization 

**Title (ZH)**: 基于人类辅助动作偏好优化的机器人策略学习 

**Authors**: Wenke xia, Yichu Yang, Hongtao Wu, Xiao Ma, Tao Kong, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07127)  

**Abstract**: Establishing a reliable and iteratively refined robotic system is essential for deploying real-world applications. While Vision-Language-Action (VLA) models are widely recognized as the foundation model for such robotic deployment, their dependence on expert demonstrations hinders the crucial capabilities of correction and learning from failures. To mitigate this limitation, we introduce a Human-assisted Action Preference Optimization method named HAPO, designed to correct deployment failures and foster effective adaptation through preference alignment for VLA models. This method begins with a human-robot collaboration framework for reliable failure correction and interaction trajectory collection through human intervention. These human-intervention trajectories are further employed within the action preference optimization process, facilitating VLA models to mitigate failure action occurrences while enhancing corrective action adaptation. Specifically, we propose an adaptive reweighting algorithm to address the issues of irreversible interactions and token probability mismatch when introducing preference optimization into VLA models, facilitating model learning from binary desirability signals derived from interactions. Through combining these modules, our human-assisted action preference optimization method ensures reliable deployment and effective learning from failure for VLA models. The experiments conducted in simulation and real-world scenarios prove superior generalization and robustness of our framework across a variety of manipulation tasks. 

**Abstract (ZH)**: 建立可靠且不断完善的机器人系统对于部署实际应用至关重要。尽管视觉-语言-动作（VLA）模型被广泛认为是此类机器人部署的基础模型，但它们对专家演示的依赖阻碍了其纠正错误和从失败中学习的关键能力。为缓解这一限制，我们提出了一种名为HAPO的人工辅助动作偏好优化方法，旨在通过偏好对齐来纠正部署错误并促进VLA模型的有效适应。该方法从可靠的人机协作框架开始，通过人的干预收集可靠的失败修正和交互轨迹。这些干预产生的轨迹进一步用于动作偏好优化过程，使VLA模型能够减少失败动作的发生并增强纠正动作的适应性。具体地，我们提出了一种自适应加权算法，以解决将偏好优化引入VLA模型时不可逆交互和标记概率不匹配的问题，使模型能够从交互中获得二元喜好信号进行学习。通过结合这些模块，我们的人工辅助动作偏好优化方法确保了VLA模型在部署过程中的可靠性，并有效地从失败中学习。在模拟和真实场景下的实验表明，我们的框架在各种操作任务中表现出色，具有出色的泛化能力和鲁棒性。 

---
# Prime the search: Using large language models for guiding geometric task and motion planning by warm-starting tree search 

**Title (ZH)**: 先验引导：使用大规模语言模型进行几何任务和运动规划的树搜索温启动 

**Authors**: Dongryung Lee, Sejune Joo, Kimin Lee, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07062)  

**Abstract**: The problem of relocating a set of objects to designated areas amidst movable obstacles can be framed as a Geometric Task and Motion Planning (G-TAMP) problem, a subclass of task and motion planning (TAMP). Traditional approaches to G-TAMP have relied either on domain-independent heuristics or on learning from planning experience to guide the search, both of which typically demand significant computational resources or data. In contrast, humans often use common sense to intuitively decide which objects to manipulate in G-TAMP problems. Inspired by this, we propose leveraging Large Language Models (LLMs), which have common sense knowledge acquired from internet-scale data, to guide task planning in G-TAMP problems. To enable LLMs to perform geometric reasoning, we design a predicate-based prompt that encodes geometric information derived from a motion planning algorithm. We then query the LLM to generate a task plan, which is then used to search for a feasible set of continuous parameters. Since LLMs are prone to mistakes, instead of committing to LLM's outputs, we extend Monte Carlo Tree Search (MCTS) to a hybrid action space and use the LLM to guide the search. Unlike the previous approach that calls an LLM at every node and incurs high computational costs, we use it to warm-start the MCTS with the nodes explored in completing the LLM's task plan. On six different G-TAMP problems, we show our method outperforms previous LLM planners and pure search algorithms. Code can be found at: this https URL 

**Abstract (ZH)**: 基于几何任务和运动规划的物体重新定位问题中的几何常识引导方法 

---
# CARoL: Context-aware Adaptation for Robot Learning 

**Title (ZH)**: CARoL: 基于上下文的机器人学习适应机制 

**Authors**: Zechen Hu, Tong Xu, Xuesu Xiao, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07006)  

**Abstract**: Using Reinforcement Learning (RL) to learn new robotic tasks from scratch is often inefficient. Leveraging prior knowledge has the potential to significantly enhance learning efficiency, which, however, raises two critical challenges: how to determine the relevancy of existing knowledge and how to adaptively integrate them into learning a new task. In this paper, we propose Context-aware Adaptation for Robot Learning (CARoL), a novel framework to efficiently learn a similar but distinct new task from prior knowledge. CARoL incorporates context awareness by analyzing state transitions in system dynamics to identify similarities between the new task and prior knowledge. It then utilizes these identified similarities to prioritize and adapt specific knowledge pieces for the new task. Additionally, CARoL has a broad applicability spanning policy-based, value-based, and actor-critic RL algorithms. We validate the efficiency and generalizability of CARoL on both simulated robotic platforms and physical ground vehicles. The simulations include CarRacing and LunarLander environments, where CARoL demonstrates faster convergence and higher rewards when learning policies for new tasks. In real-world experiments, we show that CARoL enables a ground vehicle to quickly and efficiently adapt policies learned in simulation to smoothly traverse real-world off-road terrain. 

**Abstract (ZH)**: 基于上下文适配的机器人学习（Context-aware Adaptation for Robot Learning, CARoL） 

---
# Hierarchical Intention Tracking with Switching Trees for Real-Time Adaptation to Dynamic Human Intentions during Collaboration 

**Title (ZH)**: 基于切换树的分层意图跟踪：实现协作中动态人类意图的实时适应 

**Authors**: Zhe Huang, Ye-Ji Mun, Fatemeh Cheraghi Pouria, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.07004)  

**Abstract**: During collaborative tasks, human behavior is guided by multiple levels of intentions that evolve over time, such as task sequence preferences and interaction strategies. To adapt to these changing preferences and promptly correct any inaccurate estimations, collaborative robots must accurately track these dynamic human intentions in real time. We propose a Hierarchical Intention Tracking (HIT) algorithm for collaborative robots to track dynamic and hierarchical human intentions effectively in real time. HIT represents human intentions as intention trees with arbitrary depth, and probabilistically tracks human intentions by Bayesian filtering, upward measurement propagation, and downward posterior propagation across all levels. We develop a HIT-based robotic system that dynamically switches between Interaction-Task and Verification-Task trees for a collaborative assembly task, allowing the robot to effectively coordinate human intentions at three levels: task-level (subtask goal locations), interaction-level (mode of engagement with the robot), and verification-level (confirming or correcting intention recognition). Our user study shows that our HIT-based collaborative robot system surpasses existing collaborative robot solutions by achieving a balance between efficiency, physical workload, and user comfort while ensuring safety and task completion. Post-experiment surveys further reveal that the HIT-based system enhances the user trust and minimizes interruptions to user's task flow through its effective understanding of human intentions across multiple levels. 

**Abstract (ZH)**: 基于层次意图跟踪的协作机器人系统 

---
# Multimodal Spatial Language Maps for Robot Navigation and Manipulation 

**Title (ZH)**: 多模态空间语言地图在机器人导航与操作中的应用 

**Authors**: Chenguang Huang, Oier Mees, Andy Zeng, Wolfram Burgard  

**Link**: [PDF](https://arxiv.org/pdf/2506.06862)  

**Abstract**: Grounding language to a navigating agent's observations can leverage pretrained multimodal foundation models to match perceptions to object or event descriptions. However, previous approaches remain disconnected from environment mapping, lack the spatial precision of geometric maps, or neglect additional modality information beyond vision. To address this, we propose multimodal spatial language maps as a spatial map representation that fuses pretrained multimodal features with a 3D reconstruction of the environment. We build these maps autonomously using standard exploration. We present two instances of our maps, which are visual-language maps (VLMaps) and their extension to audio-visual-language maps (AVLMaps) obtained by adding audio information. When combined with large language models (LLMs), VLMaps can (i) translate natural language commands into open-vocabulary spatial goals (e.g., "in between the sofa and TV") directly localized in the map, and (ii) be shared across different robot embodiments to generate tailored obstacle maps on demand. Building upon the capabilities above, AVLMaps extend VLMaps by introducing a unified 3D spatial representation integrating audio, visual, and language cues through the fusion of features from pretrained multimodal foundation models. This enables robots to ground multimodal goal queries (e.g., text, images, or audio snippets) to spatial locations for navigation. Additionally, the incorporation of diverse sensory inputs significantly enhances goal disambiguation in ambiguous environments. Experiments in simulation and real-world settings demonstrate that our multimodal spatial language maps enable zero-shot spatial and multimodal goal navigation and improve recall by 50% in ambiguous scenarios. These capabilities extend to mobile robots and tabletop manipulators, supporting navigation and interaction guided by visual, audio, and spatial cues. 

**Abstract (ZH)**: 将语言 Grounding 到导航代理的观察中可以利用预训练的多模态基础模型将感知与物体或事件描述匹配起来。然而，之前的approaches 仍与环境映射脱节，缺乏几何地图的空间精度，或忽略了视觉之外的其他模态信息。为了解决这一问题，我们提出了多模态空间语言地图作为空间地图表示，将预训练的多模态特征与环境的3D重建融合。我们利用标准探索自主构建这些地图。我们提出了两个多模态空间语言地图实例，视觉-语言地图（VLMaps）及其通过添加音频信息扩展的音频-视觉-语言地图（AVLMaps）。当与大规模语言模型（LLMs）结合使用时，VLMaps 可以（i）将自然语言命令直接翻译成开放词汇空间目标（例如，“沙发和电视之间”）并定位到地图中；（ii）在不同机器人主体间共享以按需生成定制化的障碍地图。在这些能力的基础上，AVLMaps 扩展了VLMaps，通过预训练多模态基础模型的特征融合引入了一个统一的3D空间表示，联合整合了音频、视觉和语言提示，使机器人能够将多模态目标查询（例如，文本、图像或音频片段）定位到空间位置进行导航。此外，多样化的感官输入显著增强了在模棱两可环境中目标的消歧义。在仿真和真实世界场景中的实验表明，我们的多模态空间语言地图使零样本空间和多模态目标导航成为可能，并在模棱两可的情况下将召回率提高了50%。这些能力还扩展到移动机器人和桌面操作臂，支持由视觉、音频和空间提示引导的导航和交互。 

---
# RF-Source Seeking with Obstacle Avoidance using Real-time Modified Artificial Potential Fields in Unknown Environments 

**Title (ZH)**: 基于实时修改人工势场的未知环境中的RF-源搜索与避障 

**Authors**: Shahid Mohammad Mulla, Aryan Kanakapudi, Lakshmi Narasimhan, Anuj Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2506.06811)  

**Abstract**: Navigation of UAVs in unknown environments with obstacles is essential for applications in disaster response and infrastructure monitoring. However, existing obstacle avoidance algorithms, such as Artificial Potential Field (APF) are unable to generalize across environments with different obstacle configurations. Furthermore, the precise location of the final target may not be available in applications such as search and rescue, in which case approaches such as RF source seeking can be used to align towards the target location. This paper proposes a real-time trajectory planning method, which involves real-time adaptation of APF through a sampling-based approach. The proposed approach utilizes only the bearing angle of the target without its precise location, and adjusts the potential field parameters according to the environment with new obstacle configurations in real time. The main contributions of the article are i) an RF source seeking algorithm to provide a bearing angle estimate using RF signal calculations based on antenna placement, and ii) a modified APF for adaptable collision avoidance in changing environments, which are evaluated separately in the simulation software Gazebo, using ROS2 for communication. Simulation results show that the RF source-seeking algorithm achieves high accuracy, with an average angular error of just 1.48 degrees, and with this estimate, the proposed navigation algorithm improves the success rate of reaching the target by 46% and reduces the trajectory length by 1.2% compared to standard potential fields. 

**Abstract (ZH)**: 无人机在未知环境中的障碍物导航对于灾难响应和基础设施监测应用至关重要。然而，现有的障碍物规避算法，如人工势场法（APF），在不同障碍配置的环境中无法泛化。此外，在搜索与救援等应用场景中，目标的精确位置可能不可得，此时可以使用RF信号追踪方法对准目标位置。本文提出了一种实时轨迹规划方法，该方法通过采样方法实时调整APF。所提出的方法仅使用目标的方向角而无需其精确位置，并根据新障碍配置的环境实时调整势场参数。文章的主要贡献包括：i) 一种RF信号追踪算法，基于天线布局进行RF信号计算以提供方向角估计；ii) 一种修改的APF方法，用于适应性碰撞规避以应对变化的环境。这些方法在仿真软件Gazebo中分别进行评估，并使用ROS2进行通信。仿真结果显示，RF信号追踪算法具有高精度，平均角误差仅为1.48度，使用该估计值，所提出的导航算法将到达目标的成功率提高了46%，并使轨迹长度减少了1.2%。 

---
# IRS: Instance-Level 3D Scene Graphs via Room Prior Guided LiDAR-Camera Fusion 

**Title (ZH)**: IRS: Instance级3D场景图通过房间先验引导的LiDAR-相机融合 

**Authors**: Hongming Chen, Yiyang Lin, Ziliang Li, Biyu Ye, Yuying Zhang, Ximin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06804)  

**Abstract**: Indoor scene understanding remains a fundamental challenge in robotics, with direct implications for downstream tasks such as navigation and manipulation. Traditional approaches often rely on closed-set recognition or loop closure, limiting their adaptability in open-world environments. With the advent of visual foundation models (VFMs), open-vocabulary recognition and natural language querying have become feasible, unlocking new possibilities for 3D scene graph construction.
In this paper, we propose a robust and efficient framework for instance-level 3D scene graph construction via LiDAR-camera fusion. Leveraging LiDAR's wide field of view (FOV) and long-range sensing capabilities, we rapidly acquire room-level geometric priors. Multi-level VFMs are employed to improve the accuracy and consistency of semantic extraction. During instance fusion, room-based segmentation enables parallel processing, while the integration of geometric and semantic cues significantly enhances fusion accuracy and robustness. Compared to state-of-the-art methods, our approach achieves up to an order-of-magnitude improvement in construction speed while maintaining high semantic precision.
Extensive experiments in both simulated and real-world environments validate the effectiveness of our approach. We further demonstrate its practical value through a language-guided semantic navigation task, highlighting its potential for real-world robotic applications. 

**Abstract (ZH)**: 室内场景理解仍然是机器人技术中的一个基本挑战，直接影响到诸如导航和操作等下游任务。传统方法往往依靠封闭集识别或环路闭合，限制了其在开放世界环境中的适应性。随着视觉基础模型（VFMs）的出现，开放式词汇识别和自然语言查询变得可行，为3D场景图构建解锁了新的可能性。

在本文中，我们提出了一种基于LiDAR-相机融合的稳健且高效的实例级3D场景图构建框架。利用LiDAR的宽视角（FOV）和长距离感知能力，我们快速获取室级别几何先验。多级视觉基础模型被用于提高语义提取的准确性和一致性。在实例融合过程中，基于房间的分割实现并行处理，而几何和语义线索的结合大幅提升了融合的准确性和鲁棒性。与现有最佳方法相比，我们的方法在构建速度上提高了数个数量级，同时保持了较高的语义精度。

在仿真和真实环境中的广泛实验验证了我们方法的有效性。我们进一步通过语言引导的语义导航任务展示了其实用价值，突显了其在实际机器人应用中的潜力。 

---
# SARAL-Bot: Autonomous Robot for Strawberry Plant Care 

**Title (ZH)**: SARAL-Bot: 自主草莓植物护理机器人 

**Authors**: Arif Ahmed, Ritvik Agarwal, Gaurav Srikar, Nathaniel Rose, Parikshit Maini  

**Link**: [PDF](https://arxiv.org/pdf/2506.06798)  

**Abstract**: Strawberry farming demands intensive labor for monitoring and maintaining plant health. To address this, Team SARAL develops an autonomous robot for the 2024 ASABE Student Robotics Challenge, capable of navigation, unhealthy leaf detection, and removal. The system addresses labor shortages, reduces costs, and supports sustainable farming through vision-based plant assessment. This work demonstrates the potential of robotics to modernize strawberry cultivation and enable scalable, intelligent agricultural solutions. 

**Abstract (ZH)**: 草莓种植需要密集的人力来监测和维持植物健康。为了解决这一问题，SARAL团队开发了一款 autonomous 机器人参加2024年ASABE Student Robotics Challenge，该机器人具备导航、不健康叶片检测和移除功能。该系统解决了劳动力短缺问题，降低了成本，并通过基于视觉的植物评估支持可持续农业。这项工作展示了机器人技术在现代草莓种植中的潜力及其在可扩展、智能农业解决方案中的应用。 

---
# SpikePingpong: High-Frequency Spike Vision-based Robot Learning for Precise Striking in Table Tennis Game 

**Title (ZH)**: SpikePingpong：基于尖峰视觉的高频率击球机器人学习方法以实现乒乓球精确打击 

**Authors**: Hao Wang, Chengkai Hou, Xianglong Li, Yankai Fu, Chenxuan Li, Ning Chen, Gaole Dai, Jiaming Liu, Tiejun Huang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06690)  

**Abstract**: Learning to control high-speed objects in the real world remains a challenging frontier in robotics. Table tennis serves as an ideal testbed for this problem, demanding both rapid interception of fast-moving balls and precise adjustment of their trajectories. This task presents two fundamental challenges: it requires a high-precision vision system capable of accurately predicting ball trajectories, and it necessitates intelligent strategic planning to ensure precise ball placement to target regions. The dynamic nature of table tennis, coupled with its real-time response requirements, makes it particularly well-suited for advancing robotic control capabilities in fast-paced, precision-critical domains. In this paper, we present SpikePingpong, a novel system that integrates spike-based vision with imitation learning for high-precision robotic table tennis. Our approach introduces two key attempts that directly address the aforementioned challenges: SONIC, a spike camera-based module that achieves millimeter-level precision in ball-racket contact prediction by compensating for real-world uncertainties such as air resistance and friction; and IMPACT, a strategic planning module that enables accurate ball placement to targeted table regions. The system harnesses a 20 kHz spike camera for high-temporal resolution ball tracking, combined with efficient neural network models for real-time trajectory correction and stroke planning. Experimental results demonstrate that SpikePingpong achieves a remarkable 91% success rate for 30 cm accuracy target area and 71% in the more challenging 20 cm accuracy task, surpassing previous state-of-the-art approaches by 38% and 37% respectively. These significant performance improvements enable the robust implementation of sophisticated tactical gameplay strategies, providing a new research perspective for robotic control in high-speed dynamic tasks. 

**Abstract (ZH)**: 在现实世界中控制高速物体仍然是机器人技术的一个挑战性前沿问题。乒乓球为这一问题提供了一个理想的测试平台，既要求快速拦截快速移动的球，又要求精确调整球的轨迹。该任务提出了两个基本挑战：它需要一个高精度的视觉系统来准确预测球的轨迹，并需要智能化的战略规划以确保将球精确放置到目标区域。由于乒乓球的动态特性和实时响应要求，它特别适合于推进机器人在快节奏、高精度关键领域的控制能力。在本文中，我们提出了一种名为SpikePingpong的新系统，该系统结合了基于尖峰的视觉与模仿学习，以实现高精度的机器人乒乓球。我们的方法引入了两个关键尝试，直接应对上述挑战：SONIC，一种基于尖峰摄像头的模块，通过补偿诸如空气阻力和摩擦等现实世界不确定性，实现了毫米级精度的球拍接触预测；和IMPACT，一种战略规划模块，能够实现将球精确放置到指定球台区域的准确性。该系统利用20 kHz的尖峰摄像头进行高时间分辨率的球跟踪，结合高效的神经网络模型进行实时轨迹校正和击球规划。实验结果表明，SpikePingpong在30 cm精度目标区域的击球成功率达到了91%，在更具挑战性的20 cm精度任务中为71%，分别超越了之前最先进的方法38%和37%。这些显著的性能提升使得复杂的战术游戏策略得以稳健实施，为在高速动态任务中的机器人控制提供了新的研究视角。 

---
# RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks 

**Title (ZH)**: RoboPARA：跨任务的并行分配与重组的双臂机器人规划 

**Authors**: Shiying Duan, Pei Ren, Nanxiang Jiang, Zhengping Che, Jian Tang, Yifan Sun, Zhaoxin Fan, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06683)  

**Abstract**: Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking scenarios. While existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm collaboration. To address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism planning. RoboPARA employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task coherence. In addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty levels. Extensive experiments on the X-DAPT dataset demonstrate that RoboPARA significantly outperforms existing methods, achieving higher efficiency and reliability, particularly in complex task combinations. The code and dataset will be released upon acceptance. 

**Abstract (ZH)**: 双臂机器人在复杂多任务场景中提高效率和灵活性方面发挥着关键作用。尽管现有方法在任务规划方面取得了令人鼓舞的结果，但它们往往未能充分优化任务并行性，限制了双臂协作的潜力。为解决这一问题，我们提出了RoboPARA，这是一种新颖的基于大规模语言模型的双臂任务并行规划框架。RoboPARA采用两阶段过程：（1）依赖图为基础的规划候选生成，构建有向无环图（DAG）来建模任务依赖关系并消除冗余；（2）图再遍历为基础的双臂并行规划，优化DAG遍历以最大限度地提高并行性同时保持任务的一致性。此外，我们引入了Cross-Scenario Dual-Arm Parallel Task数据集（X-DAPT数据集），这是第一个专门用于评估不同场景和难度级别下双臂任务并行性的数据集。在X-DAPT数据集上的广泛实验表明，RoboPARA显著优于现有方法，特别是在复杂任务组合中实现了更高的效率和可靠性。代码和数据集将在接受后发布。 

---
# RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation 

**Title (ZH)**: RoboCerebra：大规模长期 horizon 机器人操控评估基准 

**Authors**: Songhao Han, Boxiang Qiu, Yue Liao, Siyuan Huang, Chen Gao, Shuicheng Yan, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06677)  

**Abstract**: Recent advances in vision-language models (VLMs) have enabled instruction-conditioned robotic systems with improved generalization. However, most existing work focuses on reactive System 1 policies, underutilizing VLMs' strengths in semantic reasoning and long-horizon planning. These System 2 capabilities-characterized by deliberative, goal-directed thinking-remain under explored due to the limited temporal scale and structural complexity of current benchmarks. To address this gap, we introduce RoboCerebra, a benchmark for evaluating high-level reasoning in long-horizon robotic manipulation. RoboCerebra includes: (1) a large-scale simulation dataset with extended task horizons and diverse subtask sequences in household environments; (2) a hierarchical framework combining a high-level VLM planner with a low-level vision-language-action (VLA) controller; and (3) an evaluation protocol targeting planning, reflection, and memory through structured System 1-System 2 interaction. The dataset is constructed via a top-down pipeline, where GPT generates task instructions and decomposes them into subtask sequences. Human operators execute the subtasks in simulation, yielding high-quality trajectories with dynamic object variations. Compared to prior benchmarks, RoboCerebra features significantly longer action sequences and denser annotations. We further benchmark state-of-the-art VLMs as System 2 modules and analyze their performance across key cognitive dimensions, advancing the development of more capable and generalizable robotic planners. 

**Abstract (ZH)**: 近期在视觉-语言模型（VLMs）方面的进展使得基于指令的机器人系统具备了更强的泛化能力。然而，大多数现有工作集中在反应性System 1策略上，未能充分利用VLMs在语义推理和长远规划方面的优势。这些体现为深思熟虑、目标导向思考的System 2能力由于当前基准在时间尺度和结构复杂性上的限制而未得到充分探索。为解决这一问题，我们引入了RoboCerebra，一个评估长时机器人操作高级推理能力的标准。RoboCerebra包括：（1）一个大规模模拟数据集，包含扩展的任务时间轴和多样的子任务序列在家用环境中；（2）一个层次框架，结合高层VLM规划器和低层视觉-语言-行动（VLA）控制器；（3）一种评估计划、反思和记忆的评价协议，通过结构化的System 1-System 2交互实现。该数据集通过自上而下的管道构建，其中GPT生成任务指令并分解为子任务序列。人类操作员在模拟中执行子任务，生成具有动态物体变化的高质量轨迹。与以前的基准相比，RoboCerebra的特点是显著较长的动作序列和更密集的标注。我们进一步以System 2模块形式评估最先进的VLMs，并在关键认知维度上分析其性能，促进更强大和通用的机器人规划器的发展。 

---
# Generalized Trajectory Scoring for End-to-end Multimodal Planning 

**Title (ZH)**: 端到端多模态规划的广义轨迹评分 

**Authors**: Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Joshua Chen, Nadine Chang, Maying Shen, Zuxuan Wu, Shiyi Lan, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2506.06664)  

**Abstract**: End-to-end multi-modal planning is a promising paradigm in autonomous driving, enabling decision-making with diverse trajectory candidates. A key component is a robust trajectory scorer capable of selecting the optimal trajectory from these candidates. While recent trajectory scorers focus on scoring either large sets of static trajectories or small sets of dynamically generated ones, both approaches face significant limitations in generalization. Static vocabularies provide effective coarse discretization but struggle to make fine-grained adaptation, while dynamic proposals offer detailed precision but fail to capture broader trajectory distributions. To overcome these challenges, we propose GTRS (Generalized Trajectory Scoring), a unified framework for end-to-end multi-modal planning that combines coarse and fine-grained trajectory evaluation. GTRS consists of three complementary innovations: (1) a diffusion-based trajectory generator that produces diverse fine-grained proposals; (2) a vocabulary generalization technique that trains a scorer on super-dense trajectory sets with dropout regularization, enabling its robust inference on smaller subsets; and (3) a sensor augmentation strategy that enhances out-of-domain generalization while incorporating refinement training for critical trajectory discrimination. As the winning solution of the Navsim v2 Challenge, GTRS demonstrates superior performance even with sub-optimal sensor inputs, approaching privileged methods that rely on ground-truth perception. Code will be available at this https URL. 

**Abstract (ZH)**: 端到端多模态规划是自动驾驶的一个有前途的范式，能够实现具有多样化轨迹候选者的决策制定。关键组件是一个 robust 轨迹评分器，能够从中选择最优轨迹。虽然最近的轨迹评分器专注于评分大量静态轨迹或少量动态生成的轨迹，但两种方法都面临着泛化能力的显著限制。静态词汇表提供有效的粗粒度离散化，但在细粒度适应方面存在困难，而动态提案虽然提供了详细精度，但无法捕捉更广泛的轨迹分布。为克服这些挑战，我们提出了 GTRS（通用轨迹评分），这是一种结合粗粒度和细粒度轨迹评估的端到端多模态规划统一框架。GTRS 包含三个互补创新：（1）基于扩散的轨迹生成器，产生多样化的细粒度提案；（2）词汇表泛化技术，通过 Dropout 正则化在超密轨迹集中训练评分器，使其能够在较小的子集上进行稳健推理；以及（3）传感器增强策略，在增强域外泛化的同时结合关键轨迹区分的细化训练。作为 Navsim v2 挑战赛的获胜解决方案，GTRS 即使在传感器输入不理想的条件下也能表现出优越的性能，接近依赖地面真实感知的特权方法。代码将在此链接获得。 

---
# DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning 

**Title (ZH)**: DriveSuprim: 向着端到端规划中精确轨迹选择的方向 

**Authors**: Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M. Alvarez, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06659)  

**Abstract**: In complex driving environments, autonomous vehicles must navigate safely. Relying on a single predicted path, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each, but face optimization challenges in precisely selecting the best option from thousands of possibilities and distinguishing subtle but safety-critical differences, especially in rare or underrepresented scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, demonstrating superior safetycritical capabilities, including collision avoidance and compliance with rules, while maintaining high trajectory quality in various driving scenarios. 

**Abstract (ZH)**: 在复杂驾驶环境中，自动驾驶车辆必须安全导航。单纯依赖基于回归的方法预测单一路径通常不会明示评估预测轨迹的安全性。选择性方法通过生成和评估多个轨迹候选并为每个候选预测安全评分来解决这一问题，但面临在成千上万种可能性中精确选择最佳选项及区分细微但关键的安全差异的优化挑战，尤其是在罕见或代表性不足的场景中。我们提出了DriveSuprim以克服这些挑战，并通过粗细结合的渐进式候选过滤、基于旋转的增强方法提高在分布外场景中的鲁棒性以及自我蒸馏框架稳定训练来推进选择性方法的范式。DriveSuprim在不使用额外数据的情况下实现了最先进的性能，在NAVSIM v1中达到93.5%的PDMS，在NAVSIM v2中达到87.1%的EPDMS，展示了卓越的安全关键能力，包括碰撞避免和遵守规则，同时在各种驾驶场景中保持高质量的轨迹。 

---
# Self-Adapting Improvement Loops for Robotic Learning 

**Title (ZH)**: 自适应改进循环在机器人学习中的应用 

**Authors**: Calvin Luo, Zilai Zeng, Mingxi Jia, Yilun Du, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.06658)  

**Abstract**: Video generative models trained on expert demonstrations have been utilized as performant text-conditioned visual planners for solving robotic tasks. However, generalization to unseen tasks remains a challenge. Whereas improved generalization may be facilitated by leveraging learned prior knowledge from additional pre-collected offline data sources, such as web-scale video datasets, in the era of experience we aim to design agents that can continuously improve in an online manner from self-collected behaviors. In this work we thus propose the Self-Adapting Improvement Loop (SAIL), where an in-domain video model iteratively updates itself on self-produced trajectories, collected through adaptation with an internet-scale pretrained video model, and steadily improves its performance for a specified task of interest. We apply SAIL to a diverse suite of MetaWorld tasks, as well as two manipulation tasks on a real robot arm, and find that performance improvements continuously emerge over multiple iterations for novel tasks initially unseen during original in-domain video model training. Furthermore, we discover that SAIL is surprisingly robust regarding if and how the self-collected experience is filtered, and the quality of the initial in-domain demonstrations. Through adaptation with summarized internet-scale data, and learning through online experience, we thus demonstrate a way to iteratively bootstrap a high-performance video model for solving novel robotic tasks through self-improvement. 

**Abstract (ZH)**: 自我适应改进循环（SAIL）：通过自我收集的行为持续改进视觉规划模型以解决新型机器人任务 

---
# Active Test-time Vision-Language Navigation 

**Title (ZH)**: 运行时视觉-语言导航actively test-time vision-language navigation 

**Authors**: Heeju Ko, Sungjune Kim, Gyeongrok Oh, Jeongyoon Yoon, Honglak Lee, Sujin Jang, Seungryong Kim, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.06630)  

**Abstract**: Vision-Language Navigation (VLN) policies trained on offline datasets often exhibit degraded task performance when deployed in unfamiliar navigation environments at test time, where agents are typically evaluated without access to external interaction or feedback. Entropy minimization has emerged as a practical solution for reducing prediction uncertainty at test time; however, it can suffer from accumulated errors, as agents may become overconfident in incorrect actions without sufficient contextual grounding. To tackle these challenges, we introduce ATENA (Active TEst-time Navigation Agent), a test-time active learning framework that enables a practical human-robot interaction via episodic feedback on uncertain navigation outcomes. In particular, ATENA learns to increase certainty in successful episodes and decrease it in failed ones, improving uncertainty calibration. Here, we propose mixture entropy optimization, where entropy is obtained from a combination of the action and pseudo-expert distributions-a hypothetical action distribution assuming the agent's selected action to be optimal-controlling both prediction confidence and action preference. In addition, we propose a self-active learning strategy that enables an agent to evaluate its navigation outcomes based on confident predictions. As a result, the agent stays actively engaged throughout all iterations, leading to well-grounded and adaptive decision-making. Extensive evaluations on challenging VLN benchmarks-REVERIE, R2R, and R2R-CE-demonstrate that ATENA successfully overcomes distributional shifts at test time, outperforming the compared baseline methods across various settings. 

**Abstract (ZH)**: 基于测试时主动学习的导航剂ATENA：应对视觉-语言导航中的不确定性调整与适应决策 

---
# Attention-Based Convolutional Neural Network Model for Human Lower Limb Activity Recognition using sEMG 

**Title (ZH)**: 基于注意力机制的卷积神经网络模型用于sEMG的人类下肢活动识别 

**Authors**: Mojtaba Mollahossein, Farshad Haghgoo Daryakenari, Mohammad Hossein Rohban, Gholamreza Vossoughi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06624)  

**Abstract**: Accurate classification of lower limb movements using surface electromyography (sEMG) signals plays a crucial role in assistive robotics and rehabilitation systems. In this study, we present a lightweight attention-based deep neural network (DNN) for real-time movement classification using multi-channel sEMG data from the publicly available BASAN dataset. The proposed model consists of only 62,876 parameters and is designed without the need for computationally expensive preprocessing, making it suitable for real-time deployment. We employed a leave-oneout validation strategy to ensure generalizability across subjects, and evaluated the model on three movement classes: walking, standing with knee flexion, and sitting with knee extension. The network achieved 86.74% accuracy on the validation set and 85.38% on the test set, demonstrating strong classification performance under realistic conditions. Comparative analysis with existing models in the literature highlights the efficiency and effectiveness of our approach, especially in scenarios where computational cost and real-time response are critical. The results indicate that the proposed model is a promising candidate for integration into upper-level controllers in human-robot interaction systems. 

**Abstract (ZH)**: 使用表面肌电信号（sEMG）进行下肢运动准确分类在辅助机器人和康复系统中起着重要作用。本研究提出了一种轻量级的基于注意力的深度神经网络（DNN），用于实时分类公开可用BASAN数据集中多通道sEMG数据的运动。所提出模型仅包含62,876个参数，并设计无需复杂的预处理，适合实时部署。我们采用留一出验证策略以确保模型在不同被试者间的一般性，并在行走、膝屈曲站立和膝伸展坐下三种运动类别上评估了该模型。网络在验证集上的准确率为86.74%，测试集上的准确率为85.38%，在实际场景下展示了强大的分类性能。与文献中现有模型的对比分析突显了我们方法的效率和有效性，特别是在对计算成本和实时响应有严格要求的场景中。结果表明，所提出的模型是集成到人机交互系统高层控制器中的有前途候选之一。 

---
# Underwater Multi-Robot Simulation and Motion Planning in Angler 

**Title (ZH)**: Angler中的水下多机器人仿真与运动规划 

**Authors**: Akshaya Agrawal, Evan Palmer, Zachary Kingston, Geoffrey A. Hollinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.06612)  

**Abstract**: Deploying multi-robot systems in underwater environments is expensive and lengthy; testing algorithms and software in simulation improves development by decoupling software and hardware. However, this requires a simulation framework that closely resembles the real-world. Angler is an open-source framework that simulates low-level communication protocols for an onboard autopilot, such as ArduSub, providing a framework that is close to reality, but unfortunately lacking support for simulating multiple robots. We present an extension to Angler that supports multi-robot simulation and motion planning. Our extension has a modular architecture that creates non-conflicting communication channels between Gazebo, ArduSub Software-in-the-Loop (SITL), and MAVROS to operate multiple robots simultaneously in the same environment. Our multi-robot motion planning module interfaces with cascaded controllers via a JointTrajectory controller in ROS~2. We also provide an integration with the Open Motion Planning Library (OMPL), a collision avoidance module, and tools for procedural environment generation. Our work enables the development and benchmarking of underwater multi-robot motion planning in dynamic environments. 

**Abstract (ZH)**: 多机器人系统在水下环境中的部署成本高且耗时；通过在仿真中测试算法和软件可以解耦软件和硬件从而提高开发效率。然而，这需要一个与真实世界高度相似的仿真框架。Angler是一个开源框架，用于模拟岸上自主导航系统的低层通信协议，如ArduSub，提供了一个接近现实的框架，但不幸的是缺乏多个机器人仿真的支持。我们提出了一种扩展Angler，以支持多机器人仿真和运动规划。我们的扩展具有模块化架构，通过ROS~2中的JointTrajectory控制器接口与嵌套控制器进行交互，在Gazebo、ArduSub Software-in-the-Loop (SITL)和MAVROS之间创建非冲突的通信通道，以便在同一环境中同时操作多个机器人。我们的多机器人运动规划模块与OMPL、碰撞避免模块以及用于生成程序化环境的工具进行集成。我们的工作使在动态环境中开发和基准测试水下多机器人运动规划成为可能。 

---
# Enhancing Robot Safety via MLLM-Based Semantic Interpretation of Failure Data 

**Title (ZH)**: 基于MLLM的故障数据语义解释增强机器人安全性 

**Authors**: Aryaman Gupta, Yusuf Umut Ciftci, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.06570)  

**Abstract**: As robotic systems become increasingly integrated into real-world environments, ranging from autonomous vehicles to household assistants, they inevitably encounter diverse and unstructured scenarios that lead to failures. While such failures pose safety and reliability challenges, they also provide rich perceptual data for improving future performance. However, manually analyzing large-scale failure datasets is impractical. In this work, we present a method for automatically organizing large-scale robotic failure data into semantically meaningful clusters, enabling scalable learning from failure without human supervision. Our approach leverages the reasoning capabilities of Multimodal Large Language Models (MLLMs), trained on internet-scale data, to infer high-level failure causes from raw perceptual trajectories and discover interpretable structure within uncurated failure logs. These semantic clusters reveal latent patterns and hypothesized causes of failure, enabling scalable learning from experience. We demonstrate that the discovered failure modes can guide targeted data collection for policy refinement, accelerating iterative improvement in agent policies and overall safety. Additionally, we show that these semantic clusters can be employed for online failure detection, offering a lightweight yet powerful safeguard for real-time adaptation. We demonstrate that this framework enhances robot learning and robustness by transforming real-world failures into actionable and interpretable signals for adaptation. 

**Abstract (ZH)**: 随着机器人系统越来越广泛地集成到现实环境中，从自动驾驶车辆到家庭助手，它们不可避免地会遇到各种未结构化的场景，导致失败。虽然这些失败带来了安全性和可靠性的挑战，但也提供了丰富的感知数据，有助于改善未来的性能。然而，手动分析大规模失败数据集是不实际的。本文提出了一种方法，可以在无需人工监督的情况下，自动将大规模机器人失败数据组织成语义上有意义的簇，从而使能够在失败中进行可扩展的学习。我们的方法利用了预训练于互联网规模数据的多模态大型语言模型（MLLMs）的推理能力，从原始感知轨迹中推断出高层级的失败原因，并在未整理的失败日志中发现可解释的结构。这些语义簇揭示了隐藏的模式和失败的假设原因，有助于在失败中进行可扩展的学习。我们证明，发现的失败模式可以引导针对策略细化的目标数据采集，加速智能体策略的迭代改进和整体安全性。此外，我们展示了这些语义簇可以用于在线故障检测，提供了轻量级但强大的实时适应保护措施。我们证明，该框架通过将现实世界的故障转化为可操作和可解释的适应信号，增强了机器人的学习和鲁棒性。 

---
# NeSyPack: A Neuro-Symbolic Framework for Bimanual Logistics Packing 

**Title (ZH)**: NeSyPack: 一种双臂物流包装的神经符号框架 

**Authors**: Bowei Li, Peiqi Yu, Zhenran Tang, Han Zhou, Yifan Sun, Ruixuan Liu, Changliu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06567)  

**Abstract**: This paper presents NeSyPack, a neuro-symbolic framework for bimanual logistics packing. NeSyPack combines data-driven models and symbolic reasoning to build an explainable hierarchical system that is generalizable, data-efficient, and reliable. It decomposes a task into subtasks via hierarchical reasoning, and further into atomic skills managed by a symbolic skill graph. The graph selects skill parameters, robot configurations, and task-specific control strategies for execution. This modular design enables robustness, adaptability, and efficient reuse - outperforming end-to-end models that require large-scale retraining. Using NeSyPack, our team won the First Prize in the What Bimanuals Can Do (WBCD) competition at the 2025 IEEE International Conference on Robotics and Automation. 

**Abstract (ZH)**: 本论文介绍了NeSyPack，一种用于双臂物流包装的神经符号框架。NeSyPack 结合数据驱动模型和符号推理，构建了一个可解释的分层系统，该系统具有通用性、数据高效性和可靠性。该框架通过分层推理将任务分解为子任务，进一步分解为由符号技能图管理的基本技能。该图选择技能参数、机器人配置和任务特定的控制策略以供执行。这种模块化设计增强了系统的鲁棒性、适应性和高效复用性，超越了需要大规模重新训练的端到端模型。使用NeSyPack，我们的团队在2025年IEEE国际机器人与自动化会议上举办的What Bimanuals Can Do (WBCD) 竞赛中获得了第一名。 

---
# Towards Terrain-Aware Task-Driven 3D Scene Graph Generation in Outdoor Environments 

**Title (ZH)**: 面向地形感知的任务驱动户外环境3D场景图生成 

**Authors**: Chad R Samuelson, Timothy W McLain, Joshua G Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2506.06562)  

**Abstract**: High-level autonomous operations depend on a robot's ability to construct a sufficiently expressive model of its environment. Traditional three-dimensional (3D) scene representations, such as point clouds and occupancy grids, provide detailed geometric information but lack the structured, semantic organization needed for high-level reasoning. 3D scene graphs (3DSGs) address this limitation by integrating geometric, topological, and semantic relationships into a multi-level graph-based representation. By capturing hierarchical abstractions of objects and spatial layouts, 3DSGs enable robots to reason about environments in a structured manner, improving context-aware decision-making and adaptive planning. Although most recent work has focused on indoor 3DSGs, this paper investigates their construction and utility in outdoor environments. We present a method for generating a task-agnostic metric-semantic point cloud for large outdoor settings and propose modifications to existing indoor 3DSG generation techniques for outdoor applicability. Our preliminary qualitative results demonstrate the feasibility of outdoor 3DSGs and highlight their potential for future deployment in real-world field robotic applications. 

**Abstract (ZH)**: 高阶自主操作依赖于机器人构建其环境的充分表达模型的能力。传统的三维（3D）场景表示，如点云和占用网格，提供了详细的几何信息，但缺乏进行高阶推理所需的结构化、语义组织。3D场景图（3DSGs）通过将几何、拓扑和语义关系整合到多级图表示中，解决了这一限制。通过捕捉对象和空间布局的层级抽象，3DSGs使机器人能够以结构化的方式推理环境，从而改善上下文相关的决策制定和适应性规划。尽管最近大多数工作都集中在室内3D场景图上，本文则探讨了它们在户外环境中的构造和实用性。我们提出了一种方法，用于生成适用于大型户外环境的无任务特定的度量语义点云，并对现有的室内3D场景图生成技术进行了修改，以使其适用于户外应用。初步的定性结果表明，户外3D场景图的可行性，并强调了其在未来在实际野外机器人应用中的潜在作用。 

---
# Semantics-aware Predictive Inspection Path Planning 

**Title (ZH)**: 基于语义的预测性检验路径规划 

**Authors**: Mihir Dharmadhikari, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2506.06560)  

**Abstract**: This paper presents a novel semantics-aware inspection path planning paradigm called "Semantics-aware Predictive Planning" (SPP). Industrial environments that require the inspection of specific objects or structures (called "semantics"), such as ballast water tanks inside ships, often present structured and repetitive spatial arrangements of the semantics of interest. Motivated by this, we first contribute an algorithm that identifies spatially repeating patterns of semantics - exact or inexact - in a semantic scene graph representation and makes predictions about the evolution of the graph in the unseen parts of the environment using these patterns. Furthermore, two inspection path planning strategies, tailored to ballast water tank inspection, that exploit these predictions are proposed. To assess the performance of the novel predictive planning paradigm, both simulation and experimental evaluations are performed. First, we conduct a simulation study comparing the method against relevant state-of-the-art techniques and further present tests showing its ability to handle imperfect patterns. Second, we deploy our method onboard a collision-tolerant aerial robot operating inside the ballast tanks of two real ships. The results, both in simulation and field experiments, demonstrate significant improvement over the state-of-the-art in terms of inspection time while maintaining equal or better semantic surface coverage. A set of videos describing the different parts of the method and the field deployments is available at this https URL. The code for this work is made available at this https URL. 

**Abstract (ZH)**: 基于语义预测规划的新型检测路径规划范式 

---
# MapleGrasp: Mask-guided Feature Pooling for Language-driven Efficient Robotic Grasping 

**Title (ZH)**: MapleGrasp: 基于掩码引导的特征池化用于语言驱动的高效机器人抓取 

**Authors**: Vineet Bhat, Naman Patel, Prashanth Krishnamurthy, Ramesh Karri, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2506.06535)  

**Abstract**: Robotic manipulation of unseen objects via natural language commands remains challenging. Language driven robotic grasping (LDRG) predicts stable grasp poses from natural language queries and RGB-D images. Here we introduce Mask-guided feature pooling, a lightweight enhancement to existing LDRG methods. Our approach employs a two-stage training strategy: first, a vision-language model generates feature maps from CLIP-fused embeddings, which are upsampled and weighted by text embeddings to produce segmentation masks. Next, the decoder generates separate feature maps for grasp prediction, pooling only token features within these masked regions to efficiently predict grasp poses. This targeted pooling approach reduces computational complexity, accelerating both training and inference. Incorporating mask pooling results in a 12% improvement over prior approaches on the OCID-VLG benchmark. Furthermore, we introduce RefGraspNet, an open-source dataset eight times larger than existing alternatives, significantly enhancing model generalization for open-vocabulary grasping. By extending 2D grasp predictions to 3D via depth mapping and inverse kinematics, our modular method achieves performance comparable to recent Vision-Language-Action (VLA) models on the LIBERO simulation benchmark, with improved generalization across different task suites. Real-world experiments on a 7 DoF Franka robotic arm demonstrate a 57% success rate with unseen objects, surpassing competitive baselines by 7%. Code will be released post publication. 

**Abstract (ZH)**: 通过自然语言指令操纵未见物体的机器人操作仍具有挑战性。基于语言的机器人抓取（LDRG）方法从自然语言查询和RGB-D图像中预测稳定的手 grasp 姿态。我们介绍了掩码引导特征池化，这是一种对现有 LDRG 方法的轻量级增强。我们的方法采用两阶段训练策略：首先，视觉-语言模型从 CLIP 融合嵌入中生成特征图，并通过文本嵌入上采样和加权生成分割掩码。然后，解码器为抓取预测生成单独的特征图，仅在这些掩码区域内池化标记特征，从而高效地预测抓取姿态。这种目标导向的池化方法减少了计算复杂性，加速了训练和推理过程。引入掩码池化后，在 OCID-VLG 基准上的性能提高了 12%。此外，我们引入了 RefGraspNet 数据集，其大小是现有替代数据集的八倍，显著增强了开放词汇抓取的模型泛化能力。通过深度映射和逆运动学将 2D 抓取预测扩展到 3D，我们的模块化方法在 LIBERO 模拟基准上的性能与最近的视觉-语言-动作（VLA）模型相当，并且在不同任务套件上的泛化能力得到提高。在具有 7 自由度的 Franka 机器人手臂上的真实世界实验中，对于未见物体的成功率为 57%，超过了竞争性基线 7%。代码将在发表后发布。 

---
# BeliefMapNav: 3D Voxel-Based Belief Map for Zero-Shot Object Navigation 

**Title (ZH)**: BeliefMapNav: 基于3D体素的信念地图零样本对象导航 

**Authors**: Zibo Zhou, Yue Hu, Lingkai Zhang, Zonglin Li, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06487)  

**Abstract**: Zero-shot object navigation (ZSON) allows robots to find target objects in unfamiliar environments using natural language instructions, without relying on pre-built maps or task-specific training. Recent general-purpose models, such as large language models (LLMs) and vision-language models (VLMs), equip agents with semantic reasoning abilities to estimate target object locations in a zero-shot manner. However, these models often greedily select the next goal without maintaining a global understanding of the environment and are fundamentally limited in the spatial reasoning necessary for effective navigation. To overcome these limitations, we propose a novel 3D voxel-based belief map that estimates the target's prior presence distribution within a voxelized 3D space. This approach enables agents to integrate semantic priors from LLMs and visual embeddings with hierarchical spatial structure, alongside real-time observations, to build a comprehensive 3D global posterior belief of the target's location. Building on this 3D voxel map, we introduce BeliefMapNav, an efficient navigation system with two key advantages: i) grounding LLM semantic reasoning within the 3D hierarchical semantics voxel space for precise target position estimation, and ii) integrating sequential path planning to enable efficient global navigation decisions. Experiments on HM3D, MP3D, and HSSD benchmarks show that BeliefMapNav achieves state-of-the-art (SOTA) Success Rate (SR) and Success weighted by Path Length (SPL), with a notable 46.4% SPL improvement over the previous best SR method, validating its effectiveness and efficiency. 

**Abstract (ZH)**: 基于3D体素的信念图零样本物体导航 

---
# Enhancing Situational Awareness in Underwater Robotics with Multi-modal Spatial Perception 

**Title (ZH)**: 基于多模态空间感知增强水下机器人的情境感知 

**Authors**: Pushyami Kaveti, Ambjorn Grimsrud Waldum, Hanumant Singh, Martin Ludvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06476)  

**Abstract**: Autonomous Underwater Vehicles (AUVs) and Remotely Operated Vehicles (ROVs) demand robust spatial perception capabilities, including Simultaneous Localization and Mapping (SLAM), to support both remote and autonomous tasks. Vision-based systems have been integral to these advancements, capturing rich color and texture at low cost while enabling semantic scene understanding. However, underwater conditions -- such as light attenuation, backscatter, and low contrast -- often degrade image quality to the point where traditional vision-based SLAM pipelines fail. Moreover, these pipelines typically rely on monocular or stereo inputs, limiting their scalability to the multi-camera configurations common on many vehicles. To address these issues, we propose to leverage multi-modal sensing that fuses data from multiple sensors-including cameras, inertial measurement units (IMUs), and acoustic devices-to enhance situational awareness and enable robust, real-time SLAM. We explore both geometric and learning-based techniques along with semantic analysis, and conduct experiments on the data collected from a work-class ROV during several field deployments in the Trondheim Fjord. Through our experimental results, we demonstrate the feasibility of real-time reliable state estimation and high-quality 3D reconstructions in visually challenging underwater conditions. We also discuss system constraints and identify open research questions, such as sensor calibration, limitations with learning-based methods, that merit further exploration to advance large-scale underwater operations. 

**Abstract (ZH)**: 自主水下车辆(AUVs)和遥控水下车辆(ROVs)需要 robust 空间感知能力，包括同时定位与映射(SLAM)，以支持远程和自主任务。视觉系统一直是这些进步的核心，能够以低成本捕捉丰富的颜色和纹理，同时实现语义场景理解。然而，水下条件，如光线衰减、后向散射和低对比度，往往严重恶化图像质量，使传统基于视觉的SLAM流水线失效。此外，这些流水线通常依赖于单目或双目输入，限制了其在许多水下车辆上常见的多相机配置中的可扩展性。为了解决这些问题，我们提出利用多模态传感，融合来自多个传感器（包括摄像头、惯性测量单元(IMUs)和声学设备）的数据，以增强态势感知并实现稳健的实时SLAM。我们探讨了几何技术和基于学习的方法以及语义分析，并在Trondheim峡湾进行的几次现场部署中收集的工作类ROV数据上进行了实验。通过实验结果，我们证明了在视觉挑战的水下条件下进行实时可靠的状态估计和高质量3D重建的可行性。我们还讨论了系统约束并指出了需要进一步探索的研究问题，例如传感器校准，基于学习的方法的限制，以促进大规模水下操作的发展。 

---
# Edge-Enabled Collaborative Object Detection for Real-Time Multi-Vehicle Perception 

**Title (ZH)**: 基于边缘的协作目标检测：实时多车辆感知 

**Authors**: Everett Richards, Bipul Thapa, Lena Mashayekhy  

**Link**: [PDF](https://arxiv.org/pdf/2506.06474)  

**Abstract**: Accurate and reliable object detection is critical for ensuring the safety and efficiency of Connected Autonomous Vehicles (CAVs). Traditional on-board perception systems have limited accuracy due to occlusions and blind spots, while cloud-based solutions introduce significant latency, making them unsuitable for real-time processing demands required for autonomous driving in dynamic environments. To address these challenges, we introduce an innovative framework, Edge-Enabled Collaborative Object Detection (ECOD) for CAVs, that leverages edge computing and multi-CAV collaboration for real-time, multi-perspective object detection. Our ECOD framework integrates two key algorithms: Perceptive Aggregation and Collaborative Estimation (PACE) and Variable Object Tally and Evaluation (VOTE). PACE aggregates detection data from multiple CAVs on an edge server to enhance perception in scenarios where individual CAVs have limited visibility. VOTE utilizes a consensus-based voting mechanism to improve the accuracy of object classification by integrating data from multiple CAVs. Both algorithms are designed at the edge to operate in real-time, ensuring low-latency and reliable decision-making for CAVs. We develop a hardware-based controlled testbed consisting of camera-equipped robotic CAVs and an edge server to evaluate the efficacy of our framework. Our experimental results demonstrate the significant benefits of ECOD in terms of improved object classification accuracy, outperforming traditional single-perspective onboard approaches by up to 75%, while ensuring low-latency, edge-driven real-time processing. This research highlights the potential of edge computing to enhance collaborative perception for latency-sensitive autonomous systems. 

**Abstract (ZH)**: 基于边缘计算的协作式车联网对象检测框架（ECOD）：提高实时多视角对象检测的准确性和可靠性 

---
# Active Illumination Control in Low-Light Environments using NightHawk 

**Title (ZH)**: 低光环境下使用NightHawk的主动照明控制 

**Authors**: Yash Turkar, Youngjin Kim, Karthik Dantu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06394)  

**Abstract**: Subterranean environments such as culverts present significant challenges to robot vision due to dim lighting and lack of distinctive features. Although onboard illumination can help, it introduces issues such as specular reflections, overexposure, and increased power consumption. We propose NightHawk, a framework that combines active illumination with exposure control to optimize image quality in these settings. NightHawk formulates an online Bayesian optimization problem to determine the best light intensity and exposure-time for a given scene. We propose a novel feature detector-based metric to quantify image utility and use it as the cost function for the optimizer. We built NightHawk as an event-triggered recursive optimization pipeline and deployed it on a legged robot navigating a culvert beneath the Erie Canal. Results from field experiments demonstrate improvements in feature detection and matching by 47-197% enabling more reliable visual estimation in challenging lighting conditions. 

**Abstract (ZH)**: 地下环境如涵洞对机器人视觉构成了显著挑战，由于光线昏暗和缺乏 distinctive 特征。尽管可以采用机载照明，但这种方法会导致镜面反射、过度曝光和增加能耗等问题。我们提出 NightHawk，一种结合主动照明与曝光控制的框架，以优化这些环境中的图像质量。NightHawk 构建了一个在线贝叶斯优化问题，以确定给定场景的最佳光照强度和曝光时间。我们提出了一种基于特征检测的新颖度量标准来量化图像的实用性，并将其用作优化器的成本函数。我们构建了 NightHawk 作为一个事件触发的递归优化管道，并将其部署在伊利运河下方的腿式机器人上。实地实验结果表明，特征检测和匹配性能提高了 47-197%，从而在恶劣光照条件下实现了更可靠的视觉估计。 

---
# CPS-Guard: Framework for Dependability Assurance of AI- and LLM-Based Cyber-Physical Systems 

**Title (ZH)**: CPS-Guard: 嵌入式人工智能和大语言模型的赛博物理系统可靠xing保障框架 

**Authors**: Trisanth Srinivasan, Santosh Patapati, Himani Musku, Idhant Gode, Aditya Arora, Samvit Bhattacharya, Abubakr Nazriev, Sanika Hirave, Zaryab Kanjiani, Srinjoy Ghose, Srinidhi Shetty  

**Link**: [PDF](https://arxiv.org/pdf/2506.06381)  

**Abstract**: Cyber-Physical Systems (CPS) increasingly depend on advanced AI techniques to operate in critical applications. However, traditional verification and validation methods often struggle to handle the unpredictable and dynamic nature of AI components. In this paper, we introduce CPS-Guard, a novel framework that employs multi-role orchestration to automate the iterative assurance process for AI-powered CPS. By assigning specialized roles (e.g., safety monitoring, security assessment, fault injection, and recovery planning) to dedicated agents within a simulated environment, CPS-Guard continuously evaluates and refines AI behavior against a range of dependability requirements. We demonstrate the framework through a case study involving an autonomous vehicle navigating an intersection with an AI-based planner. Our results show that CPS-Guard effectively detects vulnerabilities, manages performance impacts, and supports adaptive recovery strategies, thereby offering a structured and extensible solution for rigorous V&V in safety- and security-critical systems. 

**Abstract (ZH)**: 基于多角色 orchestration 的 CPS-Guard 框架：面向 AI 驱动的 CPS 的迭代保障自动化方法 

---
# Tactile MNIST: Benchmarking Active Tactile Perception 

**Title (ZH)**: 触觉MNIST：活性触觉感知基准测试 

**Authors**: Tim Schneider, Guillaume Duret, Cristiana de Farias, Roberto Calandra, Liming Chen, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2506.06361)  

**Abstract**: Tactile perception has the potential to significantly enhance dexterous robotic manipulation by providing rich local information that can complement or substitute for other sensory modalities such as vision. However, because tactile sensing is inherently local, it is not well-suited for tasks that require broad spatial awareness or global scene understanding on its own. A human-inspired strategy to address this issue is to consider active perception techniques instead. That is, to actively guide sensors toward regions with more informative or significant features and integrate such information over time in order to understand a scene or complete a task. Both active perception and different methods for tactile sensing have received significant attention recently. Yet, despite advancements, both fields lack standardized benchmarks. To bridge this gap, we introduce the Tactile MNIST Benchmark Suite, an open-source, Gymnasium-compatible benchmark specifically designed for active tactile perception tasks, including localization, classification, and volume estimation. Our benchmark suite offers diverse simulation scenarios, from simple toy environments all the way to complex tactile perception tasks using vision-based tactile sensors. Furthermore, we also offer a comprehensive dataset comprising 13,500 synthetic 3D MNIST digit models and 153,600 real-world tactile samples collected from 600 3D printed digits. Using this dataset, we train a CycleGAN for realistic tactile simulation rendering. By providing standardized protocols and reproducible evaluation frameworks, our benchmark suite facilitates systematic progress in the fields of tactile sensing and active perception. 

**Abstract (ZH)**: 触觉感知有潜力通过提供丰富的本地信息来显著增强灵巧的机器人操作，这些信息可以补充或替代其他感官模态（如视觉）。然而，由于触觉感知本质上是局部的，它单独使用时并不适合需要广泛空间意识或全局场景理解的任务。受人类策略启发的解决思路是考虑主动感知技术，即主动引导传感器朝向更有信息量或更显著特征的区域，并通过时间上的信息整合来理解和完成任务。主动感知和不同类型的触觉传感方法近年来受到了广泛关注。尽管取得了进展，但两个领域仍缺乏标准基准。为填补这一空白，我们引入了触觉MNIST基准套件，这是一个开源的、兼容Gymnasium的基准，专门针对主动触觉感知任务，包括定位、分类和体积估计。我们的基准套件提供了多样化的模拟场景，从简单的玩具环境到基于视觉的触觉传感器进行的复杂触觉感知任务。此外，我们还提供了一个详尽的数据集，包含13,500个3D合成MNIST数字模型和153,600个来自600个3D打印数字的真实世界的触觉样本。利用该数据集，我们训练了一个CycleGAN进行逼真的触觉仿真渲染。通过提供标准化协议和可重复的评估框架，我们的基准套件促进了触觉传感和主动感知领域的系统性进步。 

---
# UA-Pose: Uncertainty-Aware 6D Object Pose Estimation and Online Object Completion with Partial References 

**Title (ZH)**: UA-Pose：基于不确定性意识的6D物体姿态估计及部分参考下的在线物体完成 

**Authors**: Ming-Feng Li, Xin Yang, Fu-En Wang, Hritam Basak, Yuyin Sun, Shreekant Gayaka, Min Sun, Cheng-Hao Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07996)  

**Abstract**: 6D object pose estimation has shown strong generalizability to novel objects. However, existing methods often require either a complete, well-reconstructed 3D model or numerous reference images that fully cover the object. Estimating 6D poses from partial references, which capture only fragments of an object's appearance and geometry, remains challenging. To address this, we propose UA-Pose, an uncertainty-aware approach for 6D object pose estimation and online object completion specifically designed for partial references. We assume access to either (1) a limited set of RGBD images with known poses or (2) a single 2D image. For the first case, we initialize a partial object 3D model based on the provided images and poses, while for the second, we use image-to-3D techniques to generate an initial object 3D model. Our method integrates uncertainty into the incomplete 3D model, distinguishing between seen and unseen regions. This uncertainty enables confidence assessment in pose estimation and guides an uncertainty-aware sampling strategy for online object completion, enhancing robustness in pose estimation accuracy and improving object completeness. We evaluate our method on the YCB-Video, YCBInEOAT, and HO3D datasets, including RGBD sequences of YCB objects manipulated by robots and human hands. Experimental results demonstrate significant performance improvements over existing methods, particularly when object observations are incomplete or partially captured. Project page: this https URL 

**Abstract (ZH)**: 基于不确定性意识的6D物体姿态估计与在线物体完成方法：针对部分参考图像的方案 

---
# FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity 

**Title (ZH)**: FreeGave: 由动态视频中的高斯速度学习3D物理 

**Authors**: Jinxi Li, Ziyang Song, Siyuan Zhou, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07865)  

**Abstract**: In this paper, we aim to model 3D scene geometry, appearance, and the underlying physics purely from multi-view videos. By applying various governing PDEs as PINN losses or incorporating physics simulation into neural networks, existing works often fail to learn complex physical motions at boundaries or require object priors such as masks or types. In this paper, we propose FreeGave to learn the physics of complex dynamic 3D scenes without needing any object priors. The key to our approach is to introduce a physics code followed by a carefully designed divergence-free module for estimating a per-Gaussian velocity field, without relying on the inefficient PINN losses. Extensive experiments on three public datasets and a newly collected challenging real-world dataset demonstrate the superior performance of our method for future frame extrapolation and motion segmentation. Most notably, our investigation into the learned physics codes reveals that they truly learn meaningful 3D physical motion patterns in the absence of any human labels in training. 

**Abstract (ZH)**: 本文旨在仅通过多视角视频，建模3D场景的几何结构、外观及其底层物理现象。现有工作往往难以学习复杂边界上的物理运动，或是需要对象先验如掩模或类型。本文提出FreeGave，无需任何对象先验即可学习复杂动态3D场景的物理原理。我们的方法关键在于引入物理代码，并通过精心设计的无散模块估计每个高斯速度场，不依赖于效率低下的PINN损失。在三个公开数据集和一个新收集的具有挑战性的现实世界数据集上的广泛实验表明，我们的方法在后续帧外推和运动分割方面表现出更优性能。尤为重要的是，我们对学习到的物理代码的研究表明，在无任何人工标签的情况下，它们确实学习到了有意义的3D物理运动模式。 

---
# LogoSP: Local-global Grouping of Superpoints for Unsupervised Semantic Segmentation of 3D Point Clouds 

**Title (ZH)**: LogoSP：局部-全局超点分组算法及其在无监督3D点云语义分割中的应用 

**Authors**: Zihui Zhang, Weisheng Dai, Hongtao Wen, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07857)  

**Abstract**: We study the problem of unsupervised 3D semantic segmentation on raw point clouds without needing human labels in training. Existing methods usually formulate this problem into learning per-point local features followed by a simple grouping strategy, lacking the ability to discover additional and possibly richer semantic priors beyond local features. In this paper, we introduce LogoSP to learn 3D semantics from both local and global point features. The key to our approach is to discover 3D semantic information by grouping superpoints according to their global patterns in the frequency domain, thus generating highly accurate semantic pseudo-labels for training a segmentation network. Extensive experiments on two indoor and an outdoor datasets show that our LogoSP surpasses all existing unsupervised methods by large margins, achieving the state-of-the-art performance for unsupervised 3D semantic segmentation. Notably, our investigation into the learned global patterns reveals that they truly represent meaningful 3D semantics in the absence of human labels during training. 

**Abstract (ZH)**: 我们研究无需人工标签的原始点云无监督3D语义分割问题。现有方法通常将此问题表述为学习每个点的局部特征，然后采用简单的分组策略，缺乏发现超出局部特征的附加且可能更丰富的语义先验的能力。在本文中，我们引入了LogoSP，以点的局部和全局特征学习3D语义。我们方法的关键在于通过在频域中根据超点的全局模式进行分组来发现3D语义信息，从而为训练分割网络生成高精度的语义伪标签。在两个室内和一个室外数据集上的大量实验表明，我们的LogoSP在无监督3D语义分割方面大幅超越了所有现有方法，达到了当前最佳性能。值得注意的是，我们对学习到的全局模式的研究表明，它们在训练过程中没有人工标签的情况下真正代表了有意义的3D语义。 

---
# R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation 

**Title (ZH)**: R3D2：基于扩散模型的自主驾驶仿真中真实istic的3D资产插入 

**Authors**: William Ljungbergh, Bernardo Taveira, Wenzhao Zheng, Adam Tonderski, Chensheng Peng, Fredrik Kahl, Christoffer Petersson, Michael Felsberg, Kurt Keutzer, Masayoshi Tomizuka, Wei Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07826)  

**Abstract**: Validating autonomous driving (AD) systems requires diverse and safety-critical testing, making photorealistic virtual environments essential. Traditional simulation platforms, while controllable, are resource-intensive to scale and often suffer from a domain gap with real-world data. In contrast, neural reconstruction methods like 3D Gaussian Splatting (3DGS) offer a scalable solution for creating photorealistic digital twins of real-world driving scenes. However, they struggle with dynamic object manipulation and reusability as their per-scene optimization-based methodology tends to result in incomplete object models with integrated illumination effects. This paper introduces R3D2, a lightweight, one-step diffusion model designed to overcome these limitations and enable realistic insertion of complete 3D assets into existing scenes by generating plausible rendering effects-such as shadows and consistent lighting-in real time. This is achieved by training R3D2 on a novel dataset: 3DGS object assets are generated from in-the-wild AD data using an image-conditioned 3D generative model, and then synthetically placed into neural rendering-based virtual environments, allowing R3D2 to learn realistic integration. Quantitative and qualitative evaluations demonstrate that R3D2 significantly enhances the realism of inserted assets, enabling use-cases like text-to-3D asset insertion and cross-scene/dataset object transfer, allowing for true scalability in AD validation. To promote further research in scalable and realistic AD simulation, we will release our dataset and code, see this https URL. 

**Abstract (ZH)**: 基于3D高斯点云的轻量级一步扩散模型R3D2在自动驾驶验证中的应用 

---
# Deep Equivariant Multi-Agent Control Barrier Functions 

**Title (ZH)**: 深度同变多代理控制 barrier 函数 

**Authors**: Nikolaos Bousias, Lars Lindemann, George Pappas  

**Link**: [PDF](https://arxiv.org/pdf/2506.07755)  

**Abstract**: With multi-agent systems increasingly deployed autonomously at scale in complex environments, ensuring safety of the data-driven policies is critical. Control Barrier Functions have emerged as an effective tool for enforcing safety constraints, yet existing learning-based methods often lack in scalability, generalization and sampling efficiency as they overlook inherent geometric structures of the system. To address this gap, we introduce symmetries-infused distributed Control Barrier Functions, enforcing the satisfaction of intrinsic symmetries on learnable graph-based safety certificates. We theoretically motivate the need for equivariant parametrization of CBFs and policies, and propose a simple, yet efficient and adaptable methodology for constructing such equivariant group-modular networks via the compatible group actions. This approach encodes safety constraints in a distributed data-efficient manner, enabling zero-shot generalization to larger and denser swarms. Through extensive simulations on multi-robot navigation tasks, we demonstrate that our method outperforms state-of-the-art baselines in terms of safety, scalability, and task success rates, highlighting the importance of embedding symmetries in safe distributed neural policies. 

**Abstract (ZH)**: 多智能体系统在复杂环境中大规模自治部署时，确保数据驱动策略的安全性至关重要。为此，我们引入了融合对称性的分布式控制屏障函数，通过对可学习的图基安全证书施加内在对称性来确保其满足条件。我们从理论上阐述了对称性在控制屏障函数和策略参数化中的必要性，并提出了一种简单但高效且可适应的方法，通过兼容的群作用构建此类对称性群模网络。该方法以分布式数据高效的方式编码安全约束，使算法能够零样本泛化到更大、更密集的群体中。通过在多机器人导航任务中的广泛仿真实验，我们证明了该方法在安全性、可扩展性和任务成功率方面优于现有先进基线，强调了在安全分布式神经策略中嵌入对称性的的重要性。 

---
# Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning 

**Title (ZH)**: 基于图辅助拼接的离线分层强化学习 

**Authors**: Seungho Baek, Taegeon Park, Jongchan Park, Seungjun Oh, Yusung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07744)  

**Abstract**: Existing offline hierarchical reinforcement learning methods rely on high-level policy learning to generate subgoal sequences. However, their efficiency degrades as task horizons increase, and they lack effective strategies for stitching useful state transitions across different trajectories. We propose Graph-Assisted Stitching (GAS), a novel framework that formulates subgoal selection as a graph search problem rather than learning an explicit high-level policy. By embedding states into a Temporal Distance Representation (TDR) space, GAS clusters semantically similar states from different trajectories into unified graph nodes, enabling efficient transition stitching. A shortest-path algorithm is then applied to select subgoal sequences within the graph, while a low-level policy learns to reach the subgoals. To improve graph quality, we introduce the Temporal Efficiency (TE) metric, which filters out noisy or inefficient transition states, significantly enhancing task performance. GAS outperforms prior offline HRL methods across locomotion, navigation, and manipulation tasks. Notably, in the most stitching-critical task, it achieves a score of 88.3, dramatically surpassing the previous state-of-the-art score of 1.0. Our source code is available at: this https URL. 

**Abstract (ZH)**: 基于图辅助缝合的离线分层强化学习方法 

---
# Curriculum Learning With Counterfactual Group Relative Policy Advantage For Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于因果替代群体相对策略优势的多智能体强化学习 curriculum 学习 

**Authors**: Weiqiang Jin, Hongyang Du, Guizhong Liu, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07548)  

**Abstract**: Multi-agent reinforcement learning (MARL) has achieved strong performance in cooperative adversarial tasks. However, most existing methods typically train agents against fixed opponent strategies and rely on such meta-static difficulty conditions, which limits their adaptability to changing environments and often leads to suboptimal policies. Inspired by the success of curriculum learning (CL) in supervised tasks, we propose a dynamic CL framework for MARL that employs an self-adaptive difficulty adjustment mechanism. This mechanism continuously modulates opponent strength based on real-time agent training performance, allowing agents to progressively learn from easier to more challenging scenarios. However, the dynamic nature of CL introduces instability due to nonstationary environments and sparse global rewards. To address this challenge, we develop a Counterfactual Group Relative Policy Advantage (CGRPA), which is tightly coupled with the curriculum by providing intrinsic credit signals that reflect each agent's impact under evolving task demands. CGRPA constructs a counterfactual advantage function that isolates individual contributions within group behavior, facilitating more reliable policy updates throughout the curriculum. CGRPA evaluates each agent's contribution through constructing counterfactual action advantage function, providing intrinsic rewards that enhance credit assignment and stabilize learning under non-stationary conditions. Extensive experiments demonstrate that our method improves both training stability and final performance, achieving competitive results against state-of-the-art methods. The code is available at this https URL. 

**Abstract (ZH)**: 多智能体强化学习中的自适应难度调节框架 

---
# Hierarchical Scoring with 3D Gaussian Splatting for Instance Image-Goal Navigation 

**Title (ZH)**: 基于3D高斯散列的分层评分实例图像-目标导航 

**Authors**: Yijie Deng, Shuaihang Yuan, Geeta Chandra Raju Bethala, Anthony Tzes, Yu-Shen Liu, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07338)  

**Abstract**: Instance Image-Goal Navigation (IIN) requires autonomous agents to identify and navigate to a target object or location depicted in a reference image captured from any viewpoint. While recent methods leverage powerful novel view synthesis (NVS) techniques, such as three-dimensional Gaussian splatting (3DGS), they typically rely on randomly sampling multiple viewpoints or trajectories to ensure comprehensive coverage of discriminative visual cues. This approach, however, creates significant redundancy through overlapping image samples and lacks principled view selection, substantially increasing both rendering and comparison overhead. In this paper, we introduce a novel IIN framework with a hierarchical scoring paradigm that estimates optimal viewpoints for target matching. Our approach integrates cross-level semantic scoring, utilizing CLIP-derived relevancy fields to identify regions with high semantic similarity to the target object class, with fine-grained local geometric scoring that performs precise pose estimation within promising regions. Extensive evaluations demonstrate that our method achieves state-of-the-art performance on simulated IIN benchmarks and real-world applicability. 

**Abstract (ZH)**: 实例图像-目标导航（IIN）要求自主代理识别并导航至参考图像中任意视角捕获的目标物体或位置。虽然近期方法利用了强大的新型视图合成（NVS）技术，如三维正态分布点绘制（3DGS），但它们通常依赖于随机采样多个视角或轨迹来确保具有 discriminative 视觉线索的全面覆盖。这种方法通过重叠图像样本产生了显著的冗余，并且缺乏原理上的视图选择，显著增加了渲染和对比的开销。在本文中，我们提出了一种新型的IIN框架，该框架采用分层评分范式来估计目标匹配的最佳视角。我们的方法结合了跨级别的语义评分，利用CLIP导出的相关性领域来识别与目标物体类别具有高语义相似性的区域，以及细粒度的局部几何评分来精确估计有希望区域内的姿态。广泛的评估表明，我们的方法在模拟的IIN基准测试和实际应用中达到了最先进的性能。 

---
# Multi-Step Guided Diffusion for Image Restoration on Edge Devices: Toward Lightweight Perception in Embodied AI 

**Title (ZH)**: 基于边缘设备的多步引导扩散影像恢复：朝向具身AI的轻量级感知 

**Authors**: Aditya Chakravarty  

**Link**: [PDF](https://arxiv.org/pdf/2506.07286)  

**Abstract**: Diffusion models have shown remarkable flexibility for solving inverse problems without task-specific retraining. However, existing approaches such as Manifold Preserving Guided Diffusion (MPGD) apply only a single gradient update per denoising step, limiting restoration fidelity and robustness, especially in embedded or out-of-distribution settings. In this work, we introduce a multistep optimization strategy within each denoising timestep, significantly enhancing image quality, perceptual accuracy, and generalization. Our experiments on super-resolution and Gaussian deblurring demonstrate that increasing the number of gradient updates per step improves LPIPS and PSNR with minimal latency overhead. Notably, we validate this approach on a Jetson Orin Nano using degraded ImageNet and a UAV dataset, showing that MPGD, originally trained on face datasets, generalizes effectively to natural and aerial scenes. Our findings highlight MPGD's potential as a lightweight, plug-and-play restoration module for real-time visual perception in embodied AI agents such as drones and mobile robots. 

**Abstract (ZH)**: 多步优化策略在去噪每个时间步的应用：显著提高图像质量、感知准确性和通用性 

---
# Active Lubrication of Transluminal Medical Instruments 

**Title (ZH)**: 经腔内医疗器械的主动润滑 

**Authors**: Mostafa A. Atalla, Jelte Nieuwenhuis, Alan Martin, Xuan Wang, Ahranee Canden, Matt J. Carré, Roger Lewis, Aimée Sakes, Michaël Wiertlewski  

**Link**: [PDF](https://arxiv.org/pdf/2506.07225)  

**Abstract**: Transluminal minimally invasive surgery uses natural orifices and small incisions to access internal anatomical structures, promoting quicker recovery and reduced morbidity. However, navigating instruments--catheters and endoscopes--through anatomical pathways creates frictional interactions with luminal walls, risking complications such as perforation, poor haptic feedback, and instrument buckling. In this paper, we present a new approach to actively lubricate transluminal instruments and dynamically reduce friction with surrounding tissues. This approach employs ultrasonic vibrations, at the instrument surface, to generate a pressurized fluid layer at the contact interface, lubricating the interface and thereby reducing friction. We implemented this approach in a prototype catheter, which we validated under dry and liquid-lubricated conditions, across rigid and soft interfaces, and along varied anatomical curvatures. In a cardiac catheter use case, active lubrication reduced friction by up to 42% on ex-vivo porcine aorta tissue and 82% on rigid substrates, denoting its potential performance on healthy and calcified tissue, respectively. Thermal imaging confirmed that temperature at the tissue-catheter interface remained within safe limits. Additionally, the system effectively prevented buckling during catheter insertion experiment, further showcasing its potential. By minimizing injury risk and enhancing procedural stability, active lubrication can drastically enhance the safety and efficacy of transluminal interventions. 

**Abstract (ZH)**: 经自然腔道和小切口的最小侵入手术通过自然开口和小切口访问内部解剖结构，促进更快恢复和减少并发症。然而，通过解剖路径导航器械（导管和内窥镜）会产生与腔壁的摩擦交互，从而增加穿孔、触觉反馈差和器械弯曲等并发症的风险。本文提出了一种新的方法，通过在器械表面产生超声振动来主动润滑器械并动态减少与周围组织的摩擦。该方法通过在接触界面产生压力流体层来润滑界面，从而减少摩擦。我们在此方法上实现了一个原型导管，分别在干式和液体润滑条件下，以及在刚性和软性界面和不同解剖曲率上进行了验证。在心脏导管使用案例中，主动润滑在离体猪主动脉组织和刚性基底上分别减少了高达42%和82%的摩擦，表明其在健康和钙化组织上的潜在性能。热成像证实，组织-导管界面的温度保持在安全范围内。此外，该系统有效防止了导管插入期间的弯曲，进一步展示了其潜力。通过减少损伤风险并增强操作稳定性，主动润滑可以大幅提高经自然腔道干预的安全性和有效性。 

---
# QForce-RL: Quantized FPGA-Optimized Reinforcement Learning Compute Engine 

**Title (ZH)**: QForce-RL: 量化FPGA优化的强化学习计算引擎 

**Authors**: Anushka Jha, Tanushree Dewangan, Mukul Lokhande, Santosh Kumar Vishvakarma  

**Link**: [PDF](https://arxiv.org/pdf/2506.07046)  

**Abstract**: Reinforcement Learning (RL) has outperformed other counterparts in sequential decision-making and dynamic environment control. However, FPGA deployment is significantly resource-expensive, as associated with large number of computations in training agents with high-quality images and possess new challenges. In this work, we propose QForce-RL takes benefits of quantization to enhance throughput and reduce energy footprint with light-weight RL architecture, without significant performance degradation. QForce-RL takes advantages from E2HRL to reduce overall RL actions to learn desired policy and QuaRL for quantization based SIMD for hardware acceleration. We have also provided detailed analysis for different RL environments, with emphasis on model size, parameters, and accelerated compute ops. The architecture is scalable for resource-constrained devices and provide parametrized efficient deployment with flexibility in latency, throughput, power, and energy efficiency. The proposed QForce-RL provides performance enhancement up to 2.3x and better FPS - 2.6x compared to SoTA works. 

**Abstract (ZH)**: QForce-RL：利用量化提高 reinforcement learning  throughput 和降低能量足迹的轻量级架构 

---
# Safety-Aware Reinforcement Learning for Control via Risk-Sensitive Action-Value Iteration and Quantile Regression 

**Title (ZH)**: 基于风险敏感动作价值迭代和分位数回归的 Awareness 安全强化学习控制 

**Authors**: Clinton Enwerem, Aniruddh G. Puranic, John S. Baras, Calin Belta  

**Link**: [PDF](https://arxiv.org/pdf/2506.06954)  

**Abstract**: Mainstream approximate action-value iteration reinforcement learning (RL) algorithms suffer from overestimation bias, leading to suboptimal policies in high-variance stochastic environments. Quantile-based action-value iteration methods reduce this bias by learning a distribution of the expected cost-to-go using quantile regression. However, ensuring that the learned policy satisfies safety constraints remains a challenge when these constraints are not explicitly integrated into the RL framework. Existing methods often require complex neural architectures or manual tradeoffs due to combined cost functions. To address this, we propose a risk-regularized quantile-based algorithm integrating Conditional Value-at-Risk (CVaR) to enforce safety without complex architectures. We also provide theoretical guarantees on the contraction properties of the risk-sensitive distributional Bellman operator in Wasserstein space, ensuring convergence to a unique cost distribution. Simulations of a mobile robot in a dynamic reach-avoid task show that our approach leads to more goal successes, fewer collisions, and better safety-performance trade-offs compared to risk-neutral methods. 

**Abstract (ZH)**: 基于分位数的风险正则化行动价值迭代强化学习算法：通过条件值风险（CVaR）确保安全而不依赖复杂架构 

---
# Towards Data-Driven Model-Free Safety-Critical Control 

**Title (ZH)**: 基于数据驱动的模型自由的安全关键控制 

**Authors**: Zhe Shen, Yitaek Kim, Christoffer Sloth  

**Link**: [PDF](https://arxiv.org/pdf/2506.06931)  

**Abstract**: This paper presents a framework for enabling safe velocity control of general robotic systems using data-driven model-free Control Barrier Functions (CBFs). Model-free CBFs rely on an exponentially stable velocity controller and a design parameter (e.g. alpha in CBFs); this design parameter depends on the exponential decay rate of the controller. However, in practice, the decay rate is often unavailable, making it non-trivial to use model-free CBFs, as it requires manual tuning for alpha. To address this, a Neural Network is used to learn the Lyapunov function from data, and the maximum decay rate of the systems built-in velocity controller is subsequently estimated. Furthermore, to integrate the estimated decay rate with model-free CBFs, we derive a probabilistic safety condition that incorporates a confidence bound on the violation rate of the exponential stability condition, using Chernoff bound. This enhances robustness against uncertainties in stability violations. The proposed framework has been tested on a UR5e robot in multiple experimental settings, and its effectiveness in ensuring safe velocity control with model-free CBFs has been demonstrated. 

**Abstract (ZH)**: 基于数据驱动的控Barrier函数框架：实现通用机器人系统的安全速度控制 

---
# Reading in the Dark with Foveated Event Vision 

**Title (ZH)**: 在暗光环境下使用聚焦事件视觉阅读 

**Authors**: Carl Brander, Giovanni Cioffi, Nico Messikommer, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2506.06918)  

**Abstract**: Current smart glasses equipped with RGB cameras struggle to perceive the environment in low-light and high-speed motion scenarios due to motion blur and the limited dynamic range of frame cameras. Additionally, capturing dense images with a frame camera requires large bandwidth and power consumption, consequently draining the battery faster. These challenges are especially relevant for developing algorithms that can read text from images. In this work, we propose a novel event-based Optical Character Recognition (OCR) approach for smart glasses. By using the eye gaze of the user, we foveate the event stream to significantly reduce bandwidth by around 98% while exploiting the benefits of event cameras in high-dynamic and fast scenes. Our proposed method performs deep binary reconstruction trained on synthetic data and leverages multimodal LLMs for OCR, outperforming traditional OCR solutions. Our results demonstrate the ability to read text in low light environments where RGB cameras struggle while using up to 2400 times less bandwidth than a wearable RGB camera. 

**Abstract (ZH)**: 基于事件的双眼佩戴式光学字符识别方法 

---
# LoopDB: A Loop Closure Dataset for Large Scale Simultaneous Localization and Mapping 

**Title (ZH)**: LoopDB：一种用于大规模同步定位与建图的闭环匹配数据集 

**Authors**: Mohammad-Maher Nakshbandi, Ziad Sharawy, Dorian Cojocaru, Sorin Grigorescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06771)  

**Abstract**: In this study, we introduce LoopDB, which is a challenging loop closure dataset comprising over 1000 images captured across diverse environments, including parks, indoor scenes, parking spaces, as well as centered around individual objects. Each scene is represented by a sequence of five consecutive images. The dataset was collected using a high resolution camera, providing suitable imagery for benchmarking the accuracy of loop closure algorithms, typically used in simultaneous localization and mapping. As ground truth information, we provide computed rotations and translations between each consecutive images. Additional to its benchmarking goal, the dataset can be used to train and fine-tune loop closure methods based on deep neural networks. LoopDB is publicly available at this https URL. 

**Abstract (ZH)**: LoopDB：一种包含超过1000张图像的具有挑战性的环回闭合数据集，这些图像捕捉了包括公园、室内场景、停车位以及个体物体在内的多种环境。每个场景由五个连续图像组成。该数据集使用高分辨率相机收集，适合用于环回闭合算法（常用于即时定位与建图）的准确性benchmark。此外，我们提供了每对连续图像之间的旋转和平移的真实地面truth信息。除了benchmark目标外，该数据集还可用于训练和微调基于深度神经网络的环回闭合方法。LoopDB已公开，可通过以下链接获取：this https URL。 

---
# Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks 

**Title (ZH)**: 基于多层级协作的LLM控制在集成陆地及非陆地网络中的多无人机运动和通信控制 

**Authors**: Zijiang Yan, Hao Zhou, Jianhua Pei, Hina Tabassum  

**Link**: [PDF](https://arxiv.org/pdf/2506.06532)  

**Abstract**: Unmanned aerial vehicles (UAVs) have been widely adopted in various real-world applications. However, the control and optimization of multi-UAV systems remain a significant challenge, particularly in dynamic and constrained environments. This work explores the joint motion and communication control of multiple UAVs operating within integrated terrestrial and non-terrestrial networks that include high-altitude platform stations (HAPS). Specifically, we consider an aerial highway scenario in which UAVs must accelerate, decelerate, and change lanes to avoid collisions and maintain overall traffic flow. Different from existing studies, we propose a novel hierarchical and collaborative method based on large language models (LLMs). In our approach, an LLM deployed on the HAPS performs UAV access control, while another LLM onboard each UAV handles motion planning and control. This LLM-based framework leverages the rich knowledge embedded in pre-trained models to enable both high-level strategic planning and low-level tactical decisions. This knowledge-driven paradigm holds great potential for the development of next-generation 3D aerial highway systems. Experimental results demonstrate that our proposed collaborative LLM-based method achieves higher system rewards, lower operational costs, and significantly reduced UAV collision rates compared to baseline approaches. 

**Abstract (ZH)**: 无人驾驶航空vehicles (UAVs)已在多种实际应用中得到广泛应用。然而，在动态和受限环境中，多UAV系统的控制与优化仍是一项重大挑战。本研究探讨了高海拔平台站(HAPS)等综合地面和非地面网络中多个UAV的联合运动和通信控制。具体而言，我们考虑了一个空中高速公路场景，其中UAVs必须加速、减速并换道以避免碰撞并保持整体交通流量。与现有研究不同，我们提出了一种基于大型语言模型（LLMs）的新型分层和协作方法。在我们的方法中，部署在HAPS上的LLM执行UAV接入控制，而每架UAV上的LLM处理运动规划与控制。该基于LLM的框架利用预训练模型中嵌入的丰富知识，实现高层次的战略规划和低层次的战术决策。这种知识驱动的范式在下一代3D空中高速公路系统的开发中具有巨大的潜力。实验结果表明，我们提出的合作LLM方法在系统奖励、运营成本和无人机碰撞率方面均优于基线方法。 

---
# Towards Infant Sleep-Optimized Driving: Synergizing Wearable and Vehicle Sensing in Intelligent Cruise Control 

**Title (ZH)**: 面向婴儿睡眠优化的驾驶：融合可穿戴设备与车辆传感的智能巡航控制 

**Authors**: Ruitao Chen, Mozhang Guo, Jinge Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06459)  

**Abstract**: Automated driving (AD) has substantially improved vehicle safety and driving comfort, but their impact on passenger well-being, particularly infant sleep, is not sufficiently studied. Sudden acceleration, abrupt braking, and sharp maneuvers can disrupt infant sleep, compromising both passenger comfort and parental convenience. To solve this problem, this paper explores the integration of reinforcement learning (RL) within AD to personalize driving behavior and optimally balance occupant comfort and travel efficiency. In particular, we propose an intelligent cruise control framework that adapts to varying driving conditions to enhance infant sleep quality by effectively synergizing wearable sensing and vehicle data. Long short-term memory (LSTM) and transformer-based neural networks are integrated with RL to model the relationship between driving behavior and infant sleep quality under diverse traffic and road conditions. Based on the sleep quality indicators from the wearable sensors, driving action data from vehicle controllers, and map data from map applications, the model dynamically computes the optimal driving aggressiveness level, which is subsequently translated into specific AD control strategies, e.g., the magnitude and frequency of acceleration, lane change, and overtaking. Simulation results demonstrate that the proposed solution significantly improves infant sleep quality compared to baseline methods, while preserving desirable travel efficiency. 

**Abstract (ZH)**: 自动化驾驶对乘客福祉特别是婴儿睡眠的影响研究：结合强化学习的个性化驾驶行为优化 

---
