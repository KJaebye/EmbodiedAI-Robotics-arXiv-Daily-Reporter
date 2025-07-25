# Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning 

**Title (ZH)**: 快入慢思：一种结合快速操作与缓慢推理的双系统基础模型 

**Authors**: Hao Chen, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Renrui Zhang, Xiaoqi Li, Xiao He, Yandong Guo, Chi-Wing Fu, Shanghang Zhang, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01953)  

**Abstract**: Generalized policy and execution efficiency constitute the two critical challenges in robotic manipulation. While recent foundation policies benefit from the common-sense reasoning capabilities of internet-scale pretrained vision-language models (VLMs), they often suffer from low execution frequency. To mitigate this dilemma, dual-system approaches, inspired by Kahneman's theory, have been proposed to leverage a VLM-based System 2 model handling high-level reasoning and a separate System 1 action model ensuring real-time control. However, existing designs maintain both systems as separate models, limiting System 1 from fully leveraging the rich pretrained knowledge from the VLM-based System 2. In this work, we propose Fast-in-Slow (FiS), a unified dual-system vision-language-action (VLA) model that embeds the System 1 execution module within the VLM-based System 2 by partially sharing parameters. This innovative paradigm not only enables high-frequency execution in System 1 but also facilitates coordination between the reasoning and execution components within a single foundation model of System 2. Given their fundamentally distinct roles within FiS-VLA, we design the two systems to incorporate heterogeneous modality inputs alongside asynchronous operating frequencies, enabling both fast and precise manipulation. To enable coordination between the two systems, a dual-aware co-training strategy is proposed that equips System 1 with action generation capabilities while preserving System 2's contextual reasoning representation. For evaluation, FiS-VLA outperforms previous state-of-the-art methods by 8% in simulation and 11% in real-world tasks in terms of average success rate, while achieving a 117.7 Hz control frequency with action chunk set to eight. Project web page: this http URL. 

**Abstract (ZH)**: 广义的政策和执行效率是机器人操作面临的两大关键挑战。尽管近期的基础政策得益于互联网规模预训练视觉-语言模型（VLMs）的常识推理能力，但通常执行频率较低。为缓解这一问题，受Kahneman理论启发的双系统方法提出了利用VLM为基础的System 2模型处理高层推理，以及独立的System 1动作模型确保实时控制。然而，现有设计将两个系统保持为独立模型，限制了System 1从VLM为基础的System 2的丰富预训练知识中充分利用。在本工作中，我们提出了Fast-in-Slow（FiS），一种统一的双系统视觉-语言-动作（VLA）模型，通过部分共享参数将System 1执行模块嵌入到VLM为基础的System 2中。这一创新范式不仅使System 1能够实现高频执行，还促进了System 2内推理和执行组件之间的协调。鉴于FiS-VLA中的两个系统具有根本不同的角色，我们设计两个系统集成异质模态输入和异步操作频率，实现即快又准的操纵。为了在两个系统之间实现协调，提出了一种双系统意识的协同训练策略，使System 1具备动作生成能力，同时保留System 2的上下文推理表示。在评估中，FiS-VLA在仿真任务中平均成功率上比之前的方法高出8%，在真实世界任务中高出11%，且动作块集设置为八个时，实现了117.7 Hz的控制频率。项目网页：this http URL。 

---
# DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes 

**Title (ZH)**: DualMap: 在动态变化场景中进行自然语言导航的在线开放词汇语义映射 

**Authors**: Jiajun Jiang, Yiming Zhu, Zirui Wu, Jie Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.01950)  

**Abstract**: We introduce DualMap, an online open-vocabulary mapping system that enables robots to understand and navigate dynamically changing environments through natural language queries. Designed for efficient semantic mapping and adaptability to changing environments, DualMap meets the essential requirements for real-world robot navigation applications. Our proposed hybrid segmentation frontend and object-level status check eliminate the costly 3D object merging required by prior methods, enabling efficient online scene mapping. The dual-map representation combines a global abstract map for high-level candidate selection with a local concrete map for precise goal-reaching, effectively managing and updating dynamic changes in the environment. Through extensive experiments in both simulation and real-world scenarios, we demonstrate state-of-the-art performance in 3D open-vocabulary segmentation, efficient scene mapping, and online language-guided navigation. 

**Abstract (ZH)**: 我们介绍了DualMap，一个在线开放词汇映射系统，通过自然语言查询使机器人能够理解和导航动态变化的环境。设计用于高效语义映射和适应变化的环境，DualMap 满足了现实世界机器人导航应用的基本要求。我们提出的混合分割前端和对象级状态检查消除了先前方法所需的昂贵的3D物体合并，从而实现高效的在线场景映射。双地图表示结合全局抽象地图用于高层候选选择以及局部具体地图用于精确目标到达，有效地管理和更新环境中的动态变化。通过在仿真和实际场景中的广泛实验，我们展示了在3D开放式词汇分割、高效场景映射和在线语言引导导航方面的前沿性能。 

---
# Feel the Force: Contact-Driven Learning from Humans 

**Title (ZH)**: 感知力: 由接触驱动的人类知识学习 

**Authors**: Ademi Adeniji, Zhuoran Chen, Vincent Liu, Venkatesh Pattabiraman, Raunaq Bhirangi, Siddhant Haldar, Pieter Abbeel, Lerrel Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.01944)  

**Abstract**: Controlling fine-grained forces during manipulation remains a core challenge in robotics. While robot policies learned from robot-collected data or simulation show promise, they struggle to generalize across the diverse range of real-world interactions. Learning directly from humans offers a scalable solution, enabling demonstrators to perform skills in their natural embodiment and in everyday environments. However, visual demonstrations alone lack the information needed to infer precise contact forces. We present FeelTheForce (FTF): a robot learning system that models human tactile behavior to learn force-sensitive manipulation. Using a tactile glove to measure contact forces and a vision-based model to estimate hand pose, we train a closed-loop policy that continuously predicts the forces needed for manipulation. This policy is re-targeted to a Franka Panda robot with tactile gripper sensors using shared visual and action representations. At execution, a PD controller modulates gripper closure to track predicted forces-enabling precise, force-aware control. Our approach grounds robust low-level force control in scalable human supervision, achieving a 77% success rate across 5 force-sensitive manipulation tasks. Code and videos are available at this https URL. 

**Abstract (ZH)**: 控制操作中的精细力控制仍然是机器人技术中的一个核心挑战。从机器人收集的数据或模拟中学习的机器人策略虽然具有潜力，但在泛化到各种真实世界的交互中仍存在问题。直接从人类学习提供了一种可扩展的解决方案，使演示者能够在自然身体和日常生活环境中执行技能。然而，仅通过视觉演示无法提供推断精确接触力所需的信息。我们提出了FeelTheForce (FTF)：一种机器人学习系统，用于建模人类的触觉行为以学习力敏感的操作。通过触觉手套测量接触力并通过基于视觉的模型估计手部姿势，我们训练了一个闭环策略，该策略能持续预测进行操作所需的力。该策略通过共享的视觉和动作表示重新针对配备触觉夹爪传感器的Franka Panda机器人。执行时，使用PD控制器调节夹爪的闭合以追踪预测的力，实现精确的、力感知的控制。我们的方法将稳健的低级力控制扎根于可扩展的人类监督中，在5种力敏感操作任务中达到了77%的成功率。代码和视频可在以下链接访问。 

---
# FreeTacMan: Robot-free Visuo-Tactile Data Collection System for Contact-rich Manipulation 

**Title (ZH)**: FreeTacMan: 无需机器人的一种触觉-视觉数据采集系统用于接触丰富的操作任务 

**Authors**: Longyan Wu, Checheng Yu, Jieji Ren, Li Chen, Ran Huang, Guoying Gu, Hongyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01941)  

**Abstract**: Enabling robots with contact-rich manipulation remains a pivotal challenge in robot learning, which is substantially hindered by the data collection gap, including its inefficiency and limited sensor setup. While prior work has explored handheld paradigms, their rod-based mechanical structures remain rigid and unintuitive, providing limited tactile feedback and posing challenges for human operators. Motivated by the dexterity and force feedback of human motion, we propose FreeTacMan, a human-centric and robot-free data collection system for accurate and efficient robot manipulation. Concretely, we design a wearable data collection device with dual visuo-tactile grippers, which can be worn by human fingers for intuitive and natural control. A high-precision optical tracking system is introduced to capture end-effector poses, while synchronizing visual and tactile feedback simultaneously. FreeTacMan achieves multiple improvements in data collection performance compared to prior works, and enables effective policy learning for contact-rich manipulation tasks with the help of the visuo-tactile information. We will release the work to facilitate reproducibility and accelerate research in visuo-tactile manipulation. 

**Abstract (ZH)**: 基于接触丰富的 mão 动作的自适应机器人数据收集系统：FreeTacMan 

---
# ADEPT: Adaptive Diffusion Environment for Policy Transfer Sim-to-Real 

**Title (ZH)**: ADEPT: 自适应扩散环境用于政策传输的模拟到现实转化 

**Authors**: Youwei Yu, Junhong Xu, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01759)  

**Abstract**: Model-free reinforcement learning has emerged as a powerful method for developing robust robot control policies capable of navigating through complex and unstructured environments. The effectiveness of these methods hinges on two essential elements: (1) the use of massively parallel physics simulations to expedite policy training, and (2) an environment generator tasked with crafting sufficiently challenging yet attainable environments to facilitate continuous policy improvement. Existing methods of outdoor environment generation often rely on heuristics constrained by a set of parameters, limiting the diversity and realism. In this work, we introduce ADEPT, a novel \textbf{A}daptive \textbf{D}iffusion \textbf{E}nvironment for \textbf{P}olicy \textbf{T}ransfer in the zero-shot sim-to-real fashion that leverages Denoising Diffusion Probabilistic Models to dynamically expand existing training environments by adding more diverse and complex environments adaptive to the current policy. ADEPT guides the diffusion model's generation process through initial noise optimization, blending noise-corrupted environments from existing training environments weighted by the policy's performance in each corresponding environment. By manipulating the noise corruption level, ADEPT seamlessly transitions between generating similar environments for policy fine-tuning and novel ones to expand training diversity. To benchmark ADEPT in off-road navigation, we propose a fast and effective multi-layer map representation for wild environment generation. Our experiments show that the policy trained by ADEPT outperforms both procedural generated and natural environments, along with popular navigation methods. 

**Abstract (ZH)**: 无需生成标题，以下是翻译内容：

无模型强化学习已成为开发能够在复杂和未结构化环境中导航的稳健机器人控制策略的强大方法。这些方法的有效性依赖于两个关键要素：（1）使用大规模并行物理模拟加速策略训练；（2）负责生成足够具有挑战性但又可行的环境以促进连续策略改进的环境生成器。现有的户外环境生成方法通常依赖受参数集约束的经验规则，这限制了环境的多样性和逼真度。在此工作中，我们提出了ADEPT，一种新颖的自适应扩散环境，用于零样本模拟到现实的策略转移，它利用去噪扩散概率模型动态扩展现有训练环境，添加更多多样和复杂、适应当前策略的环境。ADEPT通过初始噪声优化引导扩散模型的生成过程，通过根据每个环境中的政策表现加权融合现有训练环境中的噪声污染环境。通过调整噪声污染水平，ADEPT可以无缝地在为策略微调生成相似环境和为扩展训练多样性生成新颖环境之间过渡。为了在无路导航中测试ADEPT，我们提出了一种快速有效的多层地图表示方法，用于自然环境生成。实验结果显示，由ADEPT训练的策略在性能上优于程序生成和自然环境，以及流行的导航方法。 

---
# Learning with pyCub: A New Simulation and Exercise Framework for Humanoid Robotics 

**Title (ZH)**: 基于 pyCub 的新型人形机器人模拟与练习框架 

**Authors**: Lukas Rustler, Matej Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.01756)  

**Abstract**: We present pyCub, an open-source physics-based simulation of the humanoid robot iCub, along with exercises to teach students the basics of humanoid robotics. Compared to existing iCub similators (iCub SIM, iCub Gazebo), which require C++ code and YARP as middleware, pyCub works without YARP and with Python code. The complete robot with all articulations has been simulated, with two cameras in the eyes and the unique sensitive skin of the iCub comprising 4000 receptors on its body surface. The exercises range from basic control of the robot in velocity, joint, and Cartesian space to more complex tasks like gazing, grasping, or reactive control. The whole framework is written and controlled with Python, thus allowing to be used even by people with small or almost no programming practice. The exercises can be scaled to different difficulty levels. We tested the framework in two runs of a course on humanoid robotics. The simulation, exercises, documentation, Docker images, and example videos are publicly available at this https URL. 

**Abstract (ZH)**: 我们介绍了基于物理的iCub人形机器人开源模拟器pyCub及其相关练习，用以教授人形机器人基础。相比现有的iCub模拟器（iCub SIM, iCub Gazebo），pyCub无需YARP中间件且使用Python代码。整个机器人包括所有关节以及iCub特有的眼球中的两个相机和身体表面的4000个敏感受体。练习内容从基本的速度控制、关节控制以及笛卡尔空间控制，到复杂的任务如注视、抓取或反应控制。整个框架使用Python编写和控制，因此即便是编程经验较少的人也可以使用。练习可根据难度级别进行调整。我们在人形机器人课程的两次运行中测试了该框架。模拟器、练习、文档、Docker镜像和示例视频均可在该网址公开获取。 

---
# Riemannian Time Warping: Multiple Sequence Alignment in Curved Spaces 

**Title (ZH)**: 黎曼流形时间扭曲：弯曲空间中的多重序列对齐 

**Authors**: Julian Richter, Christopher Erdös, Christian Scheurer, Jochen J. Steil, Niels Dehio  

**Link**: [PDF](https://arxiv.org/pdf/2506.01635)  

**Abstract**: Temporal alignment of multiple signals through time warping is crucial in many fields, such as classification within speech recognition or robot motion learning. Almost all related works are limited to data in Euclidean space. Although an attempt was made in 2011 to adapt this concept to unit quaternions, a general extension to Riemannian manifolds remains absent. Given its importance for numerous applications in robotics and beyond, we introduce Riemannian Time Warping~(RTW). This novel approach efficiently aligns multiple signals by considering the geometric structure of the Riemannian manifold in which the data is embedded. Extensive experiments on synthetic and real-world data, including tests with an LBR iiwa robot, demonstrate that RTW consistently outperforms state-of-the-art baselines in both averaging and classification tasks. 

**Abstract (ZH)**: 基于黎曼几何的时间扭曲多信号对齐方法 

---
# A Hierarchical Bin Packing Framework with Dual Manipulators via Heuristic Search and Deep Reinforcement Learning 

**Title (ZH)**: 基于启发式搜索和深度强化学习的双 manipulator 分级集装箱打包框架 

**Authors**: Beomjoon Lee, Changjoo Nam  

**Link**: [PDF](https://arxiv.org/pdf/2506.01628)  

**Abstract**: We address the bin packing problem (BPP), which aims to maximize bin utilization when packing a variety of items. The offline problem, where the complete information about the item set and their sizes is known in advance, is proven to be NP-hard. The semi-online and online variants are even more challenging, as full information about incoming items is unavailable. While existing methods have tackled both 2D and 3D BPPs, the 2D BPP remains underexplored in terms of fully maximizing utilization. We propose a hierarchical approach for solving the 2D online and semi-online BPP by combining deep reinforcement learning (RL) with heuristic search. The heuristic search selects which item to pack or unpack, determines the packing order, and chooses the orientation of each item, while the RL agent decides the precise position within the bin. Our method is capable of handling diverse scenarios, including repacking, varying levels of item information, differing numbers of accessible items, and coordination of dual manipulators. Experimental results demonstrate that our approach achieves near-optimal utilization across various practical scenarios, largely due to its repacking capability. In addition, the algorithm is evaluated in a physics-based simulation environment, where execution time is measured to assess its real-world performance. 

**Abstract (ZH)**: 我们提出了一种层次化方法，通过结合深度强化学习（RL）和启发式搜索来解决2D在线和半在线填箱问题（BPP），以最大化箱利用率。 

---
# WoMAP: World Models For Embodied Open-Vocabulary Object Localization 

**Title (ZH)**: WoMAP: 世界模型驱动的具身开放词汇对象定位 

**Authors**: Tenny Yin, Zhiting Mei, Tao Sun, Lihan Zha, Emily Zhou, Jeremy Bao, Miyu Yamane, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.01600)  

**Abstract**: Language-instructed active object localization is a critical challenge for robots, requiring efficient exploration of partially observable environments. However, state-of-the-art approaches either struggle to generalize beyond demonstration datasets (e.g., imitation learning methods) or fail to generate physically grounded actions (e.g., VLMs). To address these limitations, we introduce WoMAP (World Models for Active Perception): a recipe for training open-vocabulary object localization policies that: (i) uses a Gaussian Splatting-based real-to-sim-to-real pipeline for scalable data generation without the need for expert demonstrations, (ii) distills dense rewards signals from open-vocabulary object detectors, and (iii) leverages a latent world model for dynamics and rewards prediction to ground high-level action proposals at inference time. Rigorous simulation and hardware experiments demonstrate WoMAP's superior performance in a broad range of zero-shot object localization tasks, with more than 9x and 2x higher success rates compared to VLM and diffusion policy baselines, respectively. Further, we show that WoMAP achieves strong generalization and sim-to-real transfer on a TidyBot. 

**Abstract (ZH)**: 基于语言指导的主动物体定位是机器人面临的᯿大挑战，要求其能够有效地探索部分可观测环境。为了解决这一限制，我们提出了WoMAP（世界模型用于主动感知）方法：一种用于训练开放词汇物体定位策略的方案，包括：（i）使用高斯斑点法实现可扩展的数据生成，无需专家演示，（ii）从开放词汇物体检测器中提炼密集奖励信号，（iii）利用潜在世界模型进行动力学和奖励预测，以在推理时接地高层动作建议。严格的仿真和硬件实验表明，WoMAP在多种零样本物体定位任务中表现出 superiority，与 VLM 和扩散策略基线相比，成功率分别提高了 9 倍和 2 倍。此外，我们展示了 WoMAP 在 TidyBot 上实现了强大的泛化能力和仿真到现实世界的迁移。 

---
# FreqPolicy: Frequency Autoregressive Visuomotor Policy with Continuous Tokens 

**Title (ZH)**: FreqPolicy: 频率自回归视听运动策略与连续令牌 

**Authors**: Yiming Zhong, Yumeng Liu, Chuyang Xiao, Zemin Yang, Youzhuo Wang, Yufei Zhu, Ye Shi, Yujing Sun, Xinge Zhu, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.01583)  

**Abstract**: Learning effective visuomotor policies for robotic manipulation is challenging, as it requires generating precise actions while maintaining computational efficiency. Existing methods remain unsatisfactory due to inherent limitations in the essential action representation and the basic network architectures. We observe that representing actions in the frequency domain captures the structured nature of motion more effectively: low-frequency components reflect global movement patterns, while high-frequency components encode fine local details. Additionally, robotic manipulation tasks of varying complexity demand different levels of modeling precision across these frequency bands. Motivated by this, we propose a novel paradigm for visuomotor policy learning that progressively models hierarchical frequency components. To further enhance precision, we introduce continuous latent representations that maintain smoothness and continuity in the action space. Extensive experiments across diverse 2D and 3D robotic manipulation benchmarks demonstrate that our approach outperforms existing methods in both accuracy and efficiency, showcasing the potential of a frequency-domain autoregressive framework with continuous tokens for generalized robotic manipulation. 

**Abstract (ZH)**: 学习有效的视觉-运动策略以实现机器人操作具有挑战性，因为它要求在保持计算效率的同时生成精确的动作。现有方法由于固有的动作表示和基本网络架构的局限性而不尽如人意。我们观察到，在频域中表示动作更有效地捕捉了运动的结构化特性：低频分量反映了全局运动模式，而高频分量则编码了细微的局部细节。此外，不同复杂度的机器人操作任务对这些频率带宽中的建模精度有着不同的需求。基于此，我们提出了一种新的视觉-运动策略学习范式，逐步建模层次化的频域分量。为了进一步提高精度，我们引入了连续的潜在表示，以保持动作空间中的平滑性和连续性。在多种2D和3D机器人操作基准测试中的广泛实验表明，我们的方法在准确性和效率上都优于现有方法，展示了频域自回归框架与连续标记token在通用机器人操作中的潜力。 

---
# Hierarchical Intention-Aware Expressive Motion Generation for Humanoid Robots 

**Title (ZH)**: humanoid机器人分层意图感知表达性运动生成 

**Authors**: Lingfan Bao, Yan Pan, Tianhu Peng, Chengxu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.01563)  

**Abstract**: Effective human-robot interaction requires robots to identify human intentions and generate expressive, socially appropriate motions in real-time. Existing approaches often rely on fixed motion libraries or computationally expensive generative models. We propose a hierarchical framework that combines intention-aware reasoning via in-context learning (ICL) with real-time motion generation using diffusion models. Our system introduces structured prompting with confidence scoring, fallback behaviors, and social context awareness to enable intention refinement and adaptive response. Leveraging large-scale motion datasets and efficient latent-space denoising, the framework generates diverse, physically plausible gestures suitable for dynamic humanoid interactions. Experimental validation on a physical platform demonstrates the robustness and social alignment of our method in realistic scenarios. 

**Abstract (ZH)**: 有效的人机交互要求机器人识别人类意图并在实时生成表达性、社会适当的运动。现有方法通常依赖于固定的运动库或计算成本高的生成模型。我们提出了一种分层框架，该框架结合了上下文学习（ICL）意图感知推理与基于扩散模型的实时运动生成。我们的系统引入了具有置信评分、备用行为和社会上下文意识的结构化提示，以实现意图细化和适应性响应。利用大规模运动数据集和高效的潜在空间去噪，该框架生成适用于动态类人交互的多样化且物理合理的手势。在物理平台上的实验验证表明，该方法在现实场景中的稳健性和社会一致性。 

---
# LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation 

**Title (ZH)**: LLM辅助的多智能体强化学习在协同策略生成中的应用 

**Authors**: Guobin Zhu, Rui Zhou, Wenkang Ji, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01538)  

**Abstract**: Although Multi-Agent Reinforcement Learning (MARL) is effective for complex multi-robot tasks, it suffers from low sample efficiency and requires iterative manual reward tuning. Large Language Models (LLMs) have shown promise in single-robot settings, but their application in multi-robot systems remains largely unexplored. This paper introduces a novel LLM-Aided MARL (LAMARL) approach, which integrates MARL with LLMs, significantly enhancing sample efficiency without requiring manual design. LAMARL consists of two modules: the first module leverages LLMs to fully automate the generation of prior policy and reward functions. The second module is MARL, which uses the generated functions to guide robot policy training effectively. On a shape assembly benchmark, both simulation and real-world experiments demonstrate the unique advantages of LAMARL. Ablation studies show that the prior policy improves sample efficiency by an average of 185.9% and enhances task completion, while structured prompts based on Chain-of-Thought (CoT) and basic APIs improve LLM output success rates by 28.5%-67.5%. Videos and code are available at this https URL 

**Abstract (ZH)**: 尽管多代理强化学习（MARL）适用于复杂的多机器人任务，但其样本效率较低且需要迭代的手动奖励调优。大型语言模型（LLMs）在单机器人设置中显示出潜力，但在多机器人系统中的应用尚未得到充分探索。本文提出了一种新的LLM辅助MARL（LAMARL）方法，将MARL与LLMs结合，显著提高了样本效率，且无需手动设计。LAMARL包括两个模块：第一个模块利用LLMs完全自动化先验策略和奖励函数的生成。第二个模块是MARL，它使用生成的函数有效地指导机器人策略的训练。在形状组装基准测试中，模拟和现实世界的实验均展示了LAMARL的独特优势。消融研究显示，先验策略平均提高样本效率185.9%，并提高任务完成率；基于Chain-of-Thought（CoT）的结构化提示和基本API提高了LLMs输出成功率28.5%-67.5%。更多视频和代码请点击此处：this <https://> URL。 

---
# SEMNAV: A Semantic Segmentation-Driven Approach to Visual Semantic Navigation 

**Title (ZH)**: SEMNAV：基于语义分割的视觉语义导航方法 

**Authors**: Rafael Flor-Rodríguez, Carlos Gutiérrez-Álvarez, Francisco Javier Acevedo-Rodríguez, Sergio Lafuente-Arroyo, Roberto J. López-Sastre  

**Link**: [PDF](https://arxiv.org/pdf/2506.01418)  

**Abstract**: Visual Semantic Navigation (VSN) is a fundamental problem in robotics, where an agent must navigate toward a target object in an unknown environment, mainly using visual information. Most state-of-the-art VSN models are trained in simulation environments, where rendered scenes of the real world are used, at best. These approaches typically rely on raw RGB data from the virtual scenes, which limits their ability to generalize to real-world environments due to domain adaptation issues. To tackle this problem, in this work, we propose SEMNAV, a novel approach that leverages semantic segmentation as the main visual input representation of the environment to enhance the agent's perception and decision-making capabilities. By explicitly incorporating high-level semantic information, our model learns robust navigation policies that improve generalization across unseen environments, both in simulated and real world settings. We also introduce a newly curated dataset, i.e. the SEMNAV dataset, designed for training semantic segmentation-aware navigation models like SEMNAV. Our approach is evaluated extensively in both simulated environments and with real-world robotic platforms. Experimental results demonstrate that SEMNAV outperforms existing state-of-the-art VSN models, achieving higher success rates in the Habitat 2.0 simulation environment, using the HM3D dataset. Furthermore, our real-world experiments highlight the effectiveness of semantic segmentation in mitigating the sim-to-real gap, making our model a promising solution for practical VSN-based robotic applications. We release SEMNAV dataset, code and trained models at this https URL 

**Abstract (ZH)**: 视觉语义导航（VSN）是机器人学中的一个基本问题，其中智能体必须使用视觉信息在未知环境中导航至目标对象。大多数最先进的VSN模型在模拟环境中训练，这些环境使用虚拟世界渲染场景，最好情况下。这些方法通常依赖于虚拟场景的原始RGB数据，这限制了它们在现实世界环境中的泛化能力，由于领域适应问题。为了解决这个问题，在本工作中，我们提出了一种新颖的方法SEMNAV，该方法利用语义分割作为环境的主要视觉输入表示，以增强智能体的感知和决策能力。通过显式地融入高级语义信息，我们的模型学习到鲁棒的导航策略，从而在模拟和真实环境中提高泛化能力。我们还引入了一个新构建的数据集，即SEMNAV数据集，适用于训练感知语义分割的导航模型，如SEMNAV。我们的方法在模拟环境和真实世界机器人平台上进行了广泛评估。实验结果表明，SEMNAV在Habitat 2.0模拟环境中使用HM3D数据集在成功率方面优于现有最先进的VSN模型。此外，我们的实际实验强调了语义分割在减小模拟到现实差距方面的有效性，使我们的模型成为实际VSN基于机器人应用的有前途的解决方案。我们在此提供的SEMNAV数据集、代码和训练模型。 

---
# Sparse Imagination for Efficient Visual World Model Planning 

**Title (ZH)**: 稀疏想象以实现高效的视觉世界模型规划 

**Authors**: Junha Chun, Youngjoon Jeong, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01392)  

**Abstract**: World model based planning has significantly improved decision-making in complex environments by enabling agents to simulate future states and make informed choices. However, ensuring the prediction accuracy of world models often demands substantial computational resources, posing a major challenge for real-time applications. This computational burden is particularly restrictive in robotics, where resources are severely constrained. To address this limitation, we propose a Sparse Imagination for Efficient Visual World Model Planning, which enhances computational efficiency by reducing the number of tokens processed during forward prediction. Our method leverages a sparsely trained vision-based world model based on transformers with randomized grouped attention strategy, allowing the model to adaptively adjust the number of tokens processed based on the computational resource. By enabling sparse imagination (rollout), our approach significantly accelerates planning while maintaining high control fidelity. Experimental results demonstrate that sparse imagination preserves task performance while dramatically improving inference efficiency, paving the way for the deployment of world models in real-time decision-making scenarios. 

**Abstract (ZH)**: 基于世界模型的规划通过使代理模拟未来状态并作出知情选择，在复杂环境中显著提高了决策能力。然而，确保世界模型预测准确度通常需要大量计算资源，这对实时应用构成了重大挑战。在资源严重受限的机器人领域，这一计算负担尤为限制性。为解决这一局限性，我们提出了一种高效的视觉世界模型规划方法——稀疏想象，通过减少前向预测过程中处理的令牌数量来提高计算效率。我们利用基于变压器的稀疏训练视觉世界模型，并采用随机分组注意力策略，使模型能够根据计算资源自适应调整处理的令牌数量。通过启用稀疏想象（ rollout），我们的方法在保持高控制精度的同时显著加速了规划过程。实验结果表明，稀疏想象在保持任务性能的同时大大提升了推理效率，为世界模型在实时决策场景中的部署开辟了道路。 

---
# Generating Diverse Challenging Terrains for Legged Robots Using Quality-Diversity Algorithm 

**Title (ZH)**: 使用质量多样性算法生成多样且具有挑战性的地形用于腿式机器人 

**Authors**: Arthur Esquerre-Pourtère, Minsoo Kim, Jaeheung Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.01362)  

**Abstract**: While legged robots have achieved significant advancements in recent years, ensuring the robustness of their controllers on unstructured terrains remains challenging. It requires generating diverse and challenging unstructured terrains to test the robot and discover its vulnerabilities. This topic remains underexplored in the literature. This paper presents a Quality-Diversity framework to generate diverse and challenging terrains that uncover weaknesses in legged robot controllers. Our method, applied to both simulated bipedal and quadruped robots, produces an archive of terrains optimized to challenge the controller in different ways. Quantitative and qualitative analyses show that the generated archive effectively contains terrains that the robots struggled to traverse, presenting different failure modes. Interesting results were observed, including failure cases that were not necessarily expected. Experiments show that the generated terrains can also be used to improve RL-based controllers. 

**Abstract (ZH)**: 基于质量多样性框架生成挑战腿式机器人控制器的多样化地形 

---
# OG-VLA: 3D-Aware Vision Language Action Model via Orthographic Image Generation 

**Title (ZH)**: OG-VLA：基于正投影图像生成的三维意识视觉语言行动模型 

**Authors**: Ishika Singh, Ankit Goyal, Stan Birchfield, Dieter Fox, Animesh Garg, Valts Blukis  

**Link**: [PDF](https://arxiv.org/pdf/2506.01196)  

**Abstract**: We introduce OG-VLA, a novel architecture and learning framework that combines the generalization strengths of Vision Language Action models (VLAs) with the robustness of 3D-aware policies. We address the challenge of mapping natural language instructions and multi-view RGBD observations to quasi-static robot actions. 3D-aware robot policies achieve state-of-the-art performance on precise robot manipulation tasks, but struggle with generalization to unseen instructions, scenes, and objects. On the other hand, VLAs excel at generalizing across instructions and scenes, but can be sensitive to camera and robot pose variations. We leverage prior knowledge embedded in language and vision foundation models to improve generalization of 3D-aware keyframe policies. OG-VLA projects input observations from diverse views into a point cloud which is then rendered from canonical orthographic views, ensuring input view invariance and consistency between input and output spaces. These canonical views are processed with a vision backbone, a Large Language Model (LLM), and an image diffusion model to generate images that encode the next position and orientation of the end-effector on the input scene. Evaluations on the Arnold and Colosseum benchmarks demonstrate state-of-the-art generalization to unseen environments, with over 40% relative improvements while maintaining robust performance in seen settings. We also show real-world adaption in 3 to 5 demonstrations along with strong generalization. Videos and resources at this https URL 

**Abstract (ZH)**: OG-VLA：结合视觉语言动作模型泛化优势和三维感知鲁棒性的新型架构与学习框架 

---
# HoMeR: Learning In-the-Wild Mobile Manipulation via Hybrid Imitation and Whole-Body Control 

**Title (ZH)**: HoMeR: 结合部分示教和全身控制的野生环境下移动操作学习 

**Authors**: Priya Sundaresan, Rhea Malhotra, Phillip Miao, Jingyun Yang, Jimmy Wu, Hengyuan Hu, Rika Antonova, Francis Engelmann, Dorsa Sadigh, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2506.01185)  

**Abstract**: We introduce HoMeR, an imitation learning framework for mobile manipulation that combines whole-body control with hybrid action modes that handle both long-range and fine-grained motion, enabling effective performance on realistic in-the-wild tasks. At its core is a fast, kinematics-based whole-body controller that maps desired end-effector poses to coordinated motion across the mobile base and arm. Within this reduced end-effector action space, HoMeR learns to switch between absolute pose predictions for long-range movement and relative pose predictions for fine-grained manipulation, offloading low-level coordination to the controller and focusing learning on task-level decisions. We deploy HoMeR on a holonomic mobile manipulator with a 7-DoF arm in a real home. We compare HoMeR to baselines without hybrid actions or whole-body control across 3 simulated and 3 real household tasks such as opening cabinets, sweeping trash, and rearranging pillows. Across tasks, HoMeR achieves an overall success rate of 79.17% using just 20 demonstrations per task, outperforming the next best baseline by 29.17 on average. HoMeR is also compatible with vision-language models and can leverage their internet-scale priors to better generalize to novel object appearances, layouts, and cluttered scenes. In summary, HoMeR moves beyond tabletop settings and demonstrates a scalable path toward sample-efficient, generalizable manipulation in everyday indoor spaces. Code, videos, and supplementary material are available at: this http URL 

**Abstract (ZH)**: HoMeR：一种结合全身控制和混合动作模式的移动 manipulation 模拟学习框架 

---
# Humanoid World Models: Open World Foundation Models for Humanoid Robotics 

**Title (ZH)**: 类人机器人世界模型：面向类人机器人的人类中心基础模型 

**Authors**: Muhammad Qasim Ali, Aditya Sridhar, Shahbuland Matiana, Alex Wong, Mohammad Al-Sharman  

**Link**: [PDF](https://arxiv.org/pdf/2506.01182)  

**Abstract**: Humanoid robots have the potential to perform complex tasks in human centered environments but require robust predictive models to reason about the outcomes of their actions. We introduce Humanoid World Models (HWM) a family of lightweight open source video based models that forecast future egocentric observations conditioned on actions. We train two types of generative models Masked Transformers and FlowMatching on 100 hours of humanoid demonstrations. Additionally we explore architectural variants with different attention mechanisms and parameter sharing strategies. Our parameter sharing techniques reduce model size by 33 to 53 with minimal impact on performance or visual fidelity. HWM is designed to be trained and deployed in practical academic and small lab settings such as 1 to 2 GPUs. 

**Abstract (ZH)**: 类人机器人具有在以人为中心的环境中执行复杂任务的潜力，但需要 robust 预测模型来推理其行动结果。我们引入了类人世界模型（HWM），这是一种基于视频的轻量级开源模型家族，能够在执行动作条件下预测未来的第一人称观测结果。我们使用 100 小时的类人展示数据训练两种生成模型——遮蔽 Transformer 和 FlowMatching，并探索了具有不同注意力机制和参数共享策略的架构变体。我们的参数共享技术将模型大小减少 33% 至 53%，同时对性能或视觉保真度的影响最小。HWM 设计用于在实际的学术和小型实验室环境中进行训练和部署，例如使用 1 到 2 块 GPU。 

---
# Standing Tall: Robust Fall Prediction for Bipedal Robots 

**Title (ZH)**: 挺立前行：双足机器人稳健跌倒预测 

**Authors**: Gokul Prabhakaran, Jessy W. Grizzle, M. Eva Mungai  

**Link**: [PDF](https://arxiv.org/pdf/2506.01141)  

**Abstract**: This paper extends the fall prediction algorithm from Mungai et al.(2024) to a real-time/online setting, implemented in both hardware and simulation. This yields results comparable to the offline version, maintaining a zero false positive rate, sufficient lead time, and accurate lead time prediction. Additionally, it achieves a high recovery rate. The paper also evaluates the fall prediction algorithm against omnidirectional faults and introduces an improved algorithm capable of reliably predicting falls and lead times across a wider range of faults in full-sized robots. Compared to Mungai et al.(2024), the proposed algorithm performs significantly better across all metrics, such as false positive rate, lead time, accuracy, and response time, demonstrating the algorithm's efficacy for real-time fall prediction in bipedal robots. 

**Abstract (ZH)**: 这篇论文将Mungai等人（2024）的跌倒预测算法扩展到实时/在线设置，并在硬件和仿真中实现。这在保持零误报率、足够提前时间以及准确的提前时间预测的同时，产生了与离线版本相当的结果，并且实现了较高的恢复率。此外，该论文还评估了跌倒预测算法在全方位故障下的表现，并引入了一个改进的算法，能够在更大范围的故障下可靠地预测跌倒和提前时间，适用于全尺寸机器人。与Mungai等人（2024）相比，所提出的算法在误报率、提前时间、准确性和响应时间等所有指标上表现显著更好，证明了该算法在双足机器人实时跌倒预测中的有效性。 

---
# $\text{TREX}^2$: Dual-Reconstruction Framework for Teleoperated-Robot with EXtended Reality 

**Title (ZH)**: TREX²: 拓展现实下的双重建构框架用于远程操作机器人 

**Authors**: Ziliang Zhang, Cong Liu, Hyoseung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01135)  

**Abstract**: Robot teleoperation with extended reality (XR teleoperation) enables intuitive interaction by allowing remote robots to mimic user motions with real-time 3D feedback. However, existing systems face significant motion-to-motion (M2M) latency--the delay between the user's latest motion and the corresponding robot feedback--leading to high teleoperation error and mission completion time. This issue stems from the system's exclusive reliance on network communication, making it highly vulnerable to network degradation. To address these challenges, we introduce $\text{TREX}^2$, the first end-to-end, fully open-sourced XR teleoperation framework that decouples robot control and XR visualization from network dependencies. $\text{TREX}^2$ leverages local sensing data to reconstruct delayed or missing information of the counterpart, thereby significantly reducing network-induced issues. This approach allows both the XR and robot to run concurrently with network transmission while maintaining high robot planning accuracy. $\text{TREX}^2$ also features contention-aware scheduling to mitigate GPU contention and bandwidth-adaptive point cloud scaling to cope with limited bandwidth. We implement $\text{TREX}^2$ across three hardware settings, including simulated and physical robots, and evaluate it on 9,500 real-world teleoperation trials from the RoboSet dataset \cite{kumar2024robohive}, covering single- and multi-step missions. Compared to state-of-the-art XR teleoperation frameworks, $\text{TREX}^2$ reduces teleoperation error by up to 69.8\% on WLAN and 73.1\% on cellular networks with only 6.7\% maximum runtime overhead. It also improves completion time by up to 47.7\%, enabling smoother teleoperation. A real-world case study on ten stationary and mobile missions further shows $\text{TREX}^2$ achieves up to 37.7\% faster completion while lowering average teleoperation error by up to 57.2\%. 

**Abstract (ZH)**: XR增强现实遥操作框架TREX²：面向端到端开源的减少网络依赖的遥操作技术 

---
# iRonCub 3: The Jet-Powered Flying Humanoid Robot 

**Title (ZH)**: iRonCub 3：喷气动力飞行类人机器人 

**Authors**: Davide Gorbani, Hosameldin Awadalla Omer Mohamed, Giuseppe L'Erario, Gabriele Nava, Punith Reddy Vanteddu, Shabarish Purushothaman Pillai, Antonello Paolino, Fabio Bergonti, Saverio Taliani, Alessandro Croci, Nicholas James Tremaroli, Silvio Traversaro, Bruno Vittorio Trombetta, Daniele Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2506.01125)  

**Abstract**: This article presents iRonCub 3, a jet-powered humanoid robot, and its first flight experiments. Unlike traditional aerial vehicles, iRonCub 3 aims to achieve flight using a full-body humanoid form, which poses unique challenges in control, estimation, and system integration. We highlight the robot's current mechanical and software architecture, including its propulsion system, control framework, and experimental infrastructure. The control and estimation framework is first validated in simulation by performing a takeoff and tracking a reference trajectory. Then, we demonstrate, for the first time, a liftoff of a jet-powered humanoid robot - an initial but significant step toward aerial humanoid mobility. Also, we detail how the experimental area around a jet-powered humanoid robot should be designed in order to deal with a level of complexity that is substantially superior than indoor humanoid robot experiments. 

**Abstract (ZH)**: 本文介绍了iRonCub 3，一种喷气动力人形机器人及其首次飞行实验。不同于传统的航空器，iRonCub 3旨在通过全身人形形式实现飞行，这在控制、估计和系统集成方面提出了独特的挑战。我们强调了该机器人的当前机械和软件架构，包括其推进系统、控制框架和实验基础设施。控制和估计框架首先在仿真中通过执行起飞和跟踪参考轨迹进行了验证。然后，我们首次展示了喷气动力人形机器人离地飞行的过程——这是向空中人形机器人移动迈出的重要一步。此外，我们还详细介绍了如何设计围绕喷气动力人形机器人的实验区域，以应对远超室内外人形机器人实验的复杂性。 

---
# STATE-NAV: Stability-Aware Traversability Estimation for Bipedal Navigation on Rough Terrain 

**Title (ZH)**: STATE-NAV: 考虑稳定性的粗糙地形 bipedal 导航可通行性估计 

**Authors**: Ziwon Yoon, Lawrence Y. Zhu, Lu Gan, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01046)  

**Abstract**: Bipedal robots have advantages in maneuvering human-centered environments, but face greater failure risk compared to other stable mobile plarforms such as wheeled or quadrupedal robots. While learning-based traversability has been widely studied for these platforms, bipedal traversability has instead relied on manually designed rules with limited consideration of locomotion stability on rough terrain. In this work, we present the first learning-based traversability estimation and risk-sensitive navigation framework for bipedal robots operating in diverse, uneven environments. 

**Abstract (ZH)**: 基于学习的双足机器人在多样化不平坦环境中的通过性估测与风险敏感导航框架 

---
# RoboTwin: A Robotic Teleoperation Framework Using Digital Twins 

**Title (ZH)**: RoboTwin: 基于数字孪生的机器人远程操作框架 

**Authors**: Harsha Yelchuri, Diwakar Kumar Singh, Nithish Krishnabharathi Gnani, T V Prabhakar, Chandramani Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.01027)  

**Abstract**: Robotic surgery imposes a significant cognitive burden on the surgeon. This cognitive burden increases in the case of remote robotic surgeries due to latency between entities and thus might affect the quality of surgery. Here, the patient side and the surgeon side are geographically separated by hundreds to thousands of kilometres. Real-time teleoperation of robots requires strict latency bounds for control and feedback. We propose a dual digital twin (DT) framework and explain the simulation environment and teleoperation framework. Here, the doctor visually controls the locally available DT of the patient side and thus experiences minimum latency. The second digital twin serves two purposes. Firstly, it provides a layer of safety for operator-related mishaps, and secondly, it conveys the coordinates of known and unknown objects back to the operator's side digital twin. We show that teleoperation accuracy and user experience are enhanced with our approach. Experimental results using the NASA-TLX metric show that the quality of surgery is vastly improved with DT, perhaps due to reduced cognitive burden. The network data rate for identifying objects at the operator side is 25x lower than normal. 

**Abstract (ZH)**: 机器人手术对外科医生产生了显著的认知负担。远程机器人手术由于实体之间的延迟增加，这种认知负担可能会进一步影响手术质量。在这种情况下，患者端和医生端相隔数百至数千公里。实时遥控机器人需要严格控制延迟和反馈的时限要求。我们提出了一种双数字孪生（DT）框架，并解释了模拟环境和遥控框架。医生通过远程控制患者端的本地可用数字孪生体，从而体验最小的延迟。第二个数字孪生体有两个作用：首先，为操作员相关失误提供一层安全保护；其次，将已知和未知物体的坐标信息传回到操作员端的数字孪生体。我们的方法提高了遥控操作的准确性和用户体验。使用NASA-TLX指标的实验结果表明，使用数字孪生体提高了手术质量，可能是由于减少了认知负担。操作端识别物体的网络数据传输速率降低了25倍。 

---
# Robust and Safe Multi-Agent Reinforcement Learning Framework with Communication for Autonomous Vehicles 

**Title (ZH)**: 具备通信的鲁棒且安全的多智能体强化学习框架：自主车辆领域 

**Authors**: Keshawn Smith, Zhili Zhang, H M Sabbir Ahmad, Ehsan Sabouni, Maniak Mondal, Song Han, Wenchao Li, Fei Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00982)  

**Abstract**: Deep multi-agent reinforcement learning (MARL) has been demonstrated effectively in simulations for many multi-robot problems. For autonomous vehicles, the development of vehicle-to-vehicle (V2V) communication technologies provide opportunities to further enhance safety of the system. However, zero-shot transfer of simulator-trained MARL policies to hardware dynamic systems remains challenging, and how to leverage communication and shared information for MARL has limited demonstrations on hardware. This problem is challenged by discrepancies between simulated and physical states, system state and model uncertainties, practical shared information design, and the need for safety guarantees in both simulation and hardware. This paper introduces RSR-RSMARL, a novel Robust and Safe MARL framework that supports Real-Sim-Real (RSR) policy adaptation for multi-agent systems with communication among agents, with both simulation and hardware demonstrations. RSR-RSMARL leverages state (includes shared state information among agents) and action representations considering real system complexities for MARL formulation. The MARL policy is trained with robust MARL algorithm to enable zero-shot transfer to hardware considering the sim-to-real gap. A safety shield module using Control Barrier Functions (CBFs) provides safety guarantee for each individual agent. Experiment results on F1/10th-scale autonomous vehicles with V2V communication demonstrate the ability of RSR-RSMARL framework to enhance driving safety and coordination across multiple configurations. These findings emphasize the importance of jointly designing robust policy representations and modular safety architectures to enable scalable, generalizable RSR transfer in multi-agent autonomy. 

**Abstract (ZH)**: 一种适用于多agent系统的Robust和Safe MARL框架：基于通信的Real-Sim-Real策略适应 

---
# Globally Consistent RGB-D SLAM with 2D Gaussian Splatting 

**Title (ZH)**: 全局一致的RGB-D SLAM与2D高斯点云匹配 

**Authors**: Xingguang Zhong, Yue Pan, Liren Jin, Marija Popović, Jens Behley, Cyrill Stachniss  

**Link**: [PDF](https://arxiv.org/pdf/2506.00970)  

**Abstract**: Recently, 3D Gaussian splatting-based RGB-D SLAM displays remarkable performance of high-fidelity 3D reconstruction. However, the lack of depth rendering consistency and efficient loop closure limits the quality of its geometric reconstructions and its ability to perform globally consistent mapping online. In this paper, we present 2DGS-SLAM, an RGB-D SLAM system using 2D Gaussian splatting as the map representation. By leveraging the depth-consistent rendering property of the 2D variant, we propose an accurate camera pose optimization method and achieve geometrically accurate 3D reconstruction. In addition, we implement efficient loop detection and camera relocalization by leveraging MASt3R, a 3D foundation model, and achieve efficient map updates by maintaining a local active map. Experiments show that our 2DGS-SLAM approach achieves superior tracking accuracy, higher surface reconstruction quality, and more consistent global map reconstruction compared to existing rendering-based SLAM methods, while maintaining high-fidelity image rendering and improved computational efficiency. 

**Abstract (ZH)**: 基于2D高斯溅射的RGB-D SLAM：几何精确的3D重建与高效循环闭合 

---
# Max Entropy Moment Kalman Filter for Polynomial Systems with Arbitrary Noise 

**Title (ZH)**: 多项式系统中任意噪声下的最大熵矩卡尔曼滤波器 

**Authors**: Sangli Teng, Harry Zhang, David Jin, Ashkan Jasour, Ram Vasudevan, Maani Ghaffari, Luca Carlone  

**Link**: [PDF](https://arxiv.org/pdf/2506.00838)  

**Abstract**: Designing optimal Bayes filters for nonlinear non-Gaussian systems is a challenging task. The main difficulties are: 1) representing complex beliefs, 2) handling non-Gaussian noise, and 3) marginalizing past states. To address these challenges, we focus on polynomial systems and propose the Max Entropy Moment Kalman Filter (MEM-KF). To address 1), we represent arbitrary beliefs by a Moment-Constrained Max-Entropy Distribution (MED). The MED can asymptotically approximate almost any distribution given an increasing number of moment constraints. To address 2), we model the noise in the process and observation model as MED. To address 3), we propagate the moments through the process model and recover the distribution as MED, thus avoiding symbolic integration, which is generally intractable. All the steps in MEM-KF, including the extraction of a point estimate, can be solved via convex optimization. We showcase the MEM-KF in challenging robotics tasks, such as localization with unknown data association. 

**Abstract (ZH)**: 设计非线性非高斯系统的最优贝叶斯滤波器是一项具有挑战性的任务。主要困难包括：1）表示复杂的信念，2）处理非高斯噪声，3）消除过去状态的影响。为了应对这些挑战，我们专注于多项式系统，并提出了最大熵矩卡尔曼滤波器（MEM-KF）。为了应对1），我们使用矩约束最大熵分布（MED）来表示任意的信念。MED在给定越来越多的矩约束时可以渐近地逼近几乎所有分布。为了应对2），我们将过程和观测模型中的噪声建模为MED。为了应对3），我们通过过程模型传播矩并在必要时恢复为MED，从而避免了通常难以处理的符号积分。MEM-KF的所有步骤，包括提取点估计，都可以通过凸优化来解决。我们在挑战性的机器人任务中展示了MEM-KF，例如具有未知数据关联的定位任务。 

---
# Improving Multi-Vehicle Perception Fusion with Millimeter-Wave Radar Assistance 

**Title (ZH)**: 利用毫米波雷达辅助改进多车辆感知融合 

**Authors**: Zhiqing Luo, Yi Wang, Yingying He, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00837)  

**Abstract**: Cooperative perception enables vehicles to share sensor readings and has become a new paradigm to improve driving safety, where the key enabling technology for realizing this vision is to real-time and accurately align and fuse the perceptions. Recent advances to align the views rely on high-density LiDAR data or fine-grained image feature representations, which however fail to meet the requirements of accuracy, real-time, and adaptability for autonomous driving. To this end, we present MMatch, a lightweight system that enables accurate and real-time perception fusion with mmWave radar point clouds. The key insight is that fine-grained spatial information provided by the radar present unique associations with all the vehicles even in two separate views. As a result, by capturing and understanding the unique local and global position of the targets in this association, we can quickly find out all the co-visible vehicles for view alignment. We implement MMatch on both the datasets collected from the CARLA platform and the real-world traffic with over 15,000 radar point cloud pairs. Experimental results show that MMatch achieves decimeter-level accuracy within 59ms, which significantly improves the reliability for autonomous driving. 

**Abstract (ZH)**: 一种基于毫米波雷达点云的轻量级实时准确感知融合系统：MMatch 

---
# DriveMind: A Dual-VLM based Reinforcement Learning Framework for Autonomous Driving 

**Title (ZH)**: DriveMind：基于双多模态语言模型的自主驾驶强化学习框架 

**Authors**: Dawood Wasif, Terrence J Moore, Chandan K Reddy, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2506.00819)  

**Abstract**: End-to-end autonomous driving systems map sensor data directly to control commands, but remain opaque, lack interpretability, and offer no formal safety guarantees. While recent vision-language-guided reinforcement learning (RL) methods introduce semantic feedback, they often rely on static prompts and fixed objectives, limiting adaptability to dynamic driving scenes. We present DriveMind, a unified semantic reward framework that integrates: (i) a contrastive Vision-Language Model (VLM) encoder for stepwise semantic anchoring; (ii) a novelty-triggered VLM encoder-decoder, fine-tuned via chain-of-thought (CoT) distillation, for dynamic prompt generation upon semantic drift; (iii) a hierarchical safety module enforcing kinematic constraints (e.g., speed, lane centering, stability); and (iv) a compact predictive world model to reward alignment with anticipated ideal states. DriveMind achieves 19.4 +/- 2.3 km/h average speed, 0.98 +/- 0.03 route completion, and near-zero collisions in CARLA Town 2, outperforming baselines by over 4% in success rate. Its semantic reward generalizes zero-shot to real dash-cam data with minimal distributional shift, demonstrating robust cross-domain alignment and potential for real-world deployment. 

**Abstract (ZH)**: 面向驾驶场景的统一语义奖励框架：DriveMind 

---
# AWML: An Open-Source ML-based Robotics Perception Framework to Deploy for ROS-based Autonomous Driving Software 

**Title (ZH)**: AWML: 一种基于ROS的自主驾驶软件的开源机器学习导向的机器人感知框架 

**Authors**: Satoshi Tanaka, Samrat Thapa, Kok Seang Tan, Amadeusz Szymko, Lobos Kenzo, Koji Minoda, Shintaro Tomie, Kotaro Uetake, Guolong Zhang, Isamu Yamashita, Takamasa Horibe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00645)  

**Abstract**: In recent years, machine learning technologies have played an important role in robotics, particularly in the development of autonomous robots and self-driving vehicles. As the industry matures, robotics frameworks like ROS 2 have been developed and provides a broad range of applications from research to production. In this work, we introduce AWML, a framework designed to support MLOps for robotics. AWML provides a machine learning infrastructure for autonomous driving, supporting not only the deployment of trained models to robotic systems, but also an active learning pipeline that incorporates auto-labeling, semi-auto-labeling, and data mining techniques. 

**Abstract (ZH)**: 近年来，机器学习技术在机器人领域发挥了重要作用，特别是在自主机器人和自动驾驶车辆的发展中。随着行业的成熟，像ROS 2这样的机器人框架被开发出来，并提供了从研究到生产的广泛应用。在本工作中，我们介绍了AWML，这是一个用于机器人领域的MLOps框架。AWML提供了一种机器学习基础设施，不仅支持将训练好的模型部署到机器人系统中，还提供了一个集成自动标注、半自动标注和数据挖掘技术的主动学习管道。 

---
# Evaluating Robot Policies in a World Model 

**Title (ZH)**: 在世界模型中评估机器人策略 

**Authors**: Julian Quevedo, Percy Liang, Sherry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00613)  

**Abstract**: Robotics has broad applications from automating house chores to taking care of patients. However, evaluating robot control policies is challenging, as real-world testing is expensive, while handcrafted simulations often fail to accurately reflect real-world conditions, resulting in poor correlation between simulated evaluation and real-world outcomes. In this work, we investigate World-model-based Policy Evaluation (WPE). We first train an action-conditioned video generation model as a proxy to real-world environments. To enable efficient rollouts of hundreds of interactive steps while mitigating error accumulation in the world model, we propose an inference scheme which we call Blockwise-Autoregressive Diffusion Transformer with adjustable context and decoding horizon lengths. To ensure that the world model indeed follows action input, we propose metrics based on the agreement between the ground truth video and generated video conditioned on the same sequence of actions to evaluate the world model. We then use the world model for policy evaluation by performing Monte Carlo rollouts in the world model while employing a vision-language model (VLM) as a reward function. Interestingly, we found that WPE tends to underestimate the policy values for in-distribution actions and overestimate policy values for out-of-distribution actions. Nevertheless, WPE preserves the relative rankings of different policies. In emulating real robot executions, WPE achieves high fidelity in mimicing robot arm movements as in real videos, while emulating highly realistic object interaction remains challenging. Despite this limitation, we show that a world model can serve as a starting point for evaluating robot policies before real-world deployment. 

**Abstract (ZH)**: 基于世界模型的策略评估：从家庭家务到患者护理的机器人应用挑战与解决方案 

---
# Constrained Stein Variational Gradient Descent for Robot Perception, Planning, and Identification 

**Title (ZH)**: 受约束的Stein变分梯度下降方法在机器人感知、规划和识别中的应用 

**Authors**: Griffin Tabor, Tucker Hermans  

**Link**: [PDF](https://arxiv.org/pdf/2506.00589)  

**Abstract**: Many core problems in robotics can be framed as constrained optimization problems. Often on these problems, the robotic system has uncertainty, or it would be advantageous to identify multiple high quality feasible solutions. To enable this, we present two novel frameworks for applying principles of constrained optimization to the new variational inference algorithm Stein variational gradient descent. Our general framework supports multiple types of constrained optimizers and can handle arbitrary constraints. We demonstrate on a variety of problems that we are able to learn to approximate distributions without violating constraints. Specifically, we show that we can build distributions of: robot motion plans that exactly avoid collisions, robot arm joint angles on the SE(3) manifold with exact table placement constraints, and object poses from point clouds with table placement constraints. 

**Abstract (ZH)**: 许多机器人领域的核心问题可以框架为约束优化问题。在这种问题上，机器人系统通常存在不确定性，或者识别多个高质量的可行解是很有优势的。为了实现这一点，我们提出了将约束优化原理应用于新的变分推理算法Stein variational gradient descent的两种新型框架。我们的通用框架支持多种类型的约束优化器，并能处理任意类型的约束。我们通过多种问题的演示表明，能够学习到不违反约束的近似分布。具体来说，我们展示了如何构建满足以下条件的分布：机器人运动计划完全避免碰撞、SE(3)流形上的机器人手臂关节角度带精确桌面放置约束、以及带有桌面放置约束的点云物体姿态分布。 

---
# Using Diffusion Ensembles to Estimate Uncertainty for End-to-End Autonomous Driving 

**Title (ZH)**: 使用扩散集成估计端到端自动驾驶中的不确定性 

**Authors**: Florian Wintel, Sigmund H. Høeg, Gabriel Kiss, Frank Lindseth  

**Link**: [PDF](https://arxiv.org/pdf/2506.00560)  

**Abstract**: End-to-end planning systems for autonomous driving are improving rapidly, especially in closed-loop simulation environments like CARLA. Many such driving systems either do not consider uncertainty as part of the plan itself, or obtain it by using specialized representations that do not generalize. In this paper, we propose EnDfuser, an end-to-end driving system that uses a diffusion model as the trajectory planner. EnDfuser effectively leverages complex perception information like fused camera and LiDAR features, through combining attention pooling and trajectory planning into a single diffusion transformer module. Instead of committing to a single plan, EnDfuser produces a distribution of candidate trajectories (128 for our case) from a single perception frame through ensemble diffusion. By observing the full set of candidate trajectories, EnDfuser provides interpretability for uncertain, multi-modal future trajectory spaces, where there are multiple plausible options. EnDfuser achieves a competitive driving score of 70.1 on the Longest6 benchmark in CARLA with minimal concessions on inference speed. Our findings suggest that ensemble diffusion, used as a drop-in replacement for traditional point-estimate trajectory planning modules, can help improve the safety of driving decisions by modeling the uncertainty of the posterior trajectory distribution. 

**Abstract (ZH)**: 端到端融合系统在自主驾驶中的快速进步，尤其是在CARLA等闭环模拟环境中。许多此类驾驶系统要么不将不确定性作为计划的一部分，要么使用特殊表示法而不具备泛化能力。本文提出了一种名为EnDfuser的端到端驾驶系统，该系统使用扩散模型作为轨迹规划器。EnDfuser有效利用了融合的感知信息，如融合的相机和LiDAR特征，通过将注意力池化和轨迹规划结合到一个单一的扩散变换器模块中来实现。EnDfuser不是仅生成一个确定性计划，而是通过集合扩散从单个感知帧生成候选轨迹分布（例如，128条候选轨迹）。通过观察候选轨迹的完整集合，EnDfuser提供了对具有多个可能选项的不确定性和多模态未来轨迹空间的可解释性。EnDfuser在CARLA的Longest6基准测试中取得了70.1的竞争力驾驶分数，同时在推理速度上几乎没有妥协。我们的研究发现表明，将集合扩散用作传统点估计轨迹规划模块的即插即用替代品，可以有助于通过建模后轨迹分布的不确定性来提高驾驶决策的安全性。 

---
# Flying Co-Stereo: Enabling Long-Range Aerial Dense Mapping via Collaborative Stereo Vision of Dynamic-Baseline 

**Title (ZH)**: 飞行共视差：通过动态基线协作立体视觉实现长距离航拍密集映射 

**Authors**: Zhaoying Wang, Xingxing Zuo, Wei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.00546)  

**Abstract**: Lightweight long-range mapping is critical for safe navigation of UAV swarms in large-scale unknown environments. Traditional stereo vision systems with fixed short baselines face limited perception ranges. To address this, we propose Flying Co-Stereo, a cross-agent collaborative stereo vision system that leverages the wide-baseline spatial configuration of two UAVs for long-range dense mapping. Key innovations include: (1) a dual-spectrum visual-inertial-ranging estimator for robust baseline estimation; (2) a hybrid feature association strategy combining deep learning-based cross-agent matching and optical-flow-based intra-agent tracking; (3) A sparse-to-dense depth recovery scheme,refining dense monocular depth predictions using exponential fitting of long-range triangulated sparse landmarks for precise metric-scale mapping. Experiments demonstrate the Flying Co-Stereo system achieves dense 3D mapping up to 70 meters with 2.3%-9.7% relative error, outperforming conventional systems by up to 350% in depth range and 450% in coverage area. The project webpage: this https URL 

**Abstract (ZH)**: 轻量级长距离建图对于大型未知环境中小型无人机群的安全导航至关重要。传统具有固定短基线的立体视觉系统面临有限的感知范围。为了解决这一问题，我们提出了一种Flying Co-Stereo飞行协同立体视觉系统，该系统利用两架无人机的宽基线空间配置实现长距离密集建图。关键创新包括：（1）一种双谱视觉-惯性测距估算器，用于稳健的基线估计；（2）一种结合基于深度学习的跨代理匹配和基于光学流的内部代理跟踪的混合特征关联策略；（3）一种稀疏到密集的深度恢复方案，使用长距离三角测量稀疏地标点的指数拟合来细化单目深度预测，实现精确的米级尺度建图。实验结果表明，Flying Co-Stereo系统实现的最大70米范围内的密集三维建图相对误差为2.3%-9.7%，在深度范围和覆盖面积方面分别比传统系统高出350%和450%。项目网页：this https URL 

---
# Multi-Objective Neural Network Assisted Design Optimization of Soft Fin-Ray Grippers for Enhanced Grasping Performance 

**Title (ZH)**: 软鳍刺夹爪增强抓取性能的多目标神经网络辅助设计优化 

**Authors**: Ali Ghanizadeh, Ali Ahmadi, Arash Bahrami  

**Link**: [PDF](https://arxiv.org/pdf/2506.00494)  

**Abstract**: Soft Fin-Ray grippers can perform delicate and careful manipulation, which has caused notable attention in different fields. These grippers can handle objects of various forms and sizes safely. The internal structure of the Fin-Ray finger plays a significant role in its adaptability and grasping performance. However, modeling the non-linear grasp force and deformation behaviors for design purposes is challenging. Moreover, when the Fin-Ray finger becomes more rigid and capable of exerting higher forces, it becomes less delicate in handling objects. The contrast between these two objectives gives rise to a multi-objective optimization problem. In this study, we employ finite element method (FEM) to estimate the deflections and contact forces of the Fin-Ray, grasping cylindrical objects. This dataset is then used to construct a multilayer perception (MLP) for prediction of the contact force and the tip displacement. The FEM dataset consists of three input and four target features. The three input features of the MLP and optimization design variables are the thickness of the front and supporting beams, the thickness of the cross beams, and the equal spacing between the cross beams. In addition, the target features are the maximum contact forces and maximum tip displacements in x- and y-directions. The magnitude of maximum contact force and magnitude of maximum tip displacement are the two objectives, showing the trade-off between force and delicate manipulation in soft Fin-Ray grippers. Furthermore, the optimized set of solutions are found using multi-objective optimal techniques. We use non-dominated sorting genetic algorithm (NSGA-II) method for this purpose. Our findings demonstrate that our methodologies can be used to improve the design and gripping performance of soft robotic grippers, helping us to choose a design not only for delicate grasping but also for high-force applications. 

**Abstract (ZH)**: 软鳍状射线 grippers 能执行精细和谨慎的操作，已在不同领域引起了广泛关注。这些 grippers 可安全地拾取各种形状和大小的对象。Fin-Ray 指节的内部结构在其实现适应性和抓取性能方面起着重要作用。然而，为了设计目的建模非线性抓取力和变形行为极具挑战性。此外，当 Fin-Ray 指节变得更为刚性和能够施加更大的力时，其在处理对象方面会更加不精细。这些相互矛盾的目标导致了一个多目标优化问题。在这项研究中，我们采用有限元方法（FEM）来估算 Fin-Ray 在抓取圆柱形物体时的位移和接触力。然后，使用该数据集构建多层感知器（MLP）以预测接触力和尖端位移。FEM 数据集包含三个输入特征和四个目标特征。MLP 和优化设计变量的三个输入特征是前梁和支撑梁的厚度、横梁的厚度以及横梁之间的等间距。另外，目标特征是 x- 和 y-方向上的最大接触力和最大尖端位移。最大接触力的大小和最大尖端位移的大小是两个目标，显示了软鳍状射线 grippers 在力和精细操作之间的权衡。此外，通过多目标优化技术找到了优化的解决方案。我们使用非支配排序遗传算法（NSGA-II）方法进行这一目的。我们的研究结果表明，我们的方法可以用来改进软机器人 grippers 的设计和抓取性能，帮助我们选择一个不仅适合精细抓取，也适用于高力应用的设计。 

---
# Disturbance-Aware Adaptive Compensation in Hybrid Force-Position Locomotion Policy for Legged Robots 

**Title (ZH)**: 带扰动感知自适应补偿的腿式机器人混合力位运动政策 

**Authors**: Yang Zhang, Buqing Nie, Zhanxiang Cao, Yangqing Fu, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00472)  

**Abstract**: Reinforcement Learning (RL)-based methods have significantly improved the locomotion performance of legged robots. However, these motion policies face significant challenges when deployed in the real world. Robots operating in uncertain environments struggle to adapt to payload variations and external disturbances, resulting in severe degradation of motion performance. In this work, we propose a novel Hybrid Force-Position Locomotion Policy (HFPLP) learning framework, where the action space of the policy is defined as a combination of target joint positions and feedforward torques, enabling the robot to rapidly respond to payload variations and external disturbances. In addition, the proposed Disturbance-Aware Adaptive Compensation (DAAC) provides compensation actions in the torque space based on external disturbance estimation, enhancing the robot's adaptability to dynamic environmental changes. We validate our approach in both simulation and real-world deployment, demonstrating that it outperforms existing methods in carrying payloads and resisting disturbances. 

**Abstract (ZH)**: 基于强化学习的混合力-位置运动策略框架在提升腿式机器人运动性能方面的研究：扰动感知自适应补偿方法 

---
# Diffusion Models for Increasing Accuracy in Olfaction Sensors and Datasets 

**Title (ZH)**: 基于扩散模型提高嗅觉传感器和数据集准确性 

**Authors**: Kordel K. France, Ovidiu Daescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00455)  

**Abstract**: Robotic odour source localization (OSL) is a critical capability for autonomous systems operating in complex environments. However, current OSL methods often suffer from ambiguities, particularly when robots misattribute odours to incorrect objects due to limitations in olfactory datasets and sensor resolutions. To address this challenge, we introduce a novel machine learning method using diffusion-based molecular generation to enhance odour localization accuracy that can be used by itself or with automated olfactory dataset construction pipelines with vision-language models (VLMs) This generative process of our diffusion model expands the chemical space beyond the limitations of both current olfactory datasets and the training data of VLMs, enabling the identification of potential odourant molecules not previously documented. The generated molecules can then be more accurately validated using advanced olfactory sensors which emulate human olfactory recognition through electronic sensor arrays. By integrating visual analysis, language processing, and molecular generation, our framework enhances the ability of olfaction-vision models on robots to accurately associate odours with their correct sources, thereby improving navigation and decision-making in environments where olfactory cues are essential. Our methodology represents a foundational advancement in the field of robotic olfaction, offering a scalable solution to the challenges posed by limited olfactory data and sensor ambiguities. 

**Abstract (ZH)**: 基于扩散机制分子生成的机器人气味源定位方法 

---
# LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks 

**Title (ZH)**: LoHoVLA：统一的视觉-语言-动作模型用于长时 horizon 体态任务 

**Authors**: Yi Yang, Jiaxuan Sun, Siqi Kou, Yihan Wang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00411)  

**Abstract**: Real-world embodied agents face long-horizon tasks, characterized by high-level goals demanding multi-step solutions beyond single actions. Successfully navigating these requires both high-level task planning (i.e., decomposing goals into sub-tasks) and low-level motion control (i.e., generating precise robot actions). While existing vision language action (VLA) models and hierarchical architectures offer potential in embodied tasks, the former often falter in planning, and the latter can suffer from coordination issues, both hampering performance. We introduce a new unified VLA framework for long-horizon tasks, dubbed LoHoVLA, to overcome these limitations. LoHoVLA leverages a large pretrained vision language model (VLM) as the backbone to jointly generate language and action tokens for sub-task generation and robot action prediction, respectively. This shared representation promotes better generalization across tasks. Additionally, LoHoVLA embraces a hierarchical closed-loop control mechanism to mitigate errors originating from both high-level planning and low-level control. To train LoHoVLA, we introduce LoHoSet, a dataset built on the Ravens simulator, containing 20 long-horizon tasks, each with 1,000 expert demonstrations composed of visual observations, linguistic goals, sub-tasks, and robot actions. Experimental results show that LoHoVLA significantly surpasses both hierarchical and standard VLA approaches on long-horizon embodied tasks in the Ravens simulator. These findings underscore the promise of unified architectures for advancing generalizable embodied intelligence. 

**Abstract (ZH)**: 长 horizon 任务中的实物代理：LoHoVLA 统一视觉语言行动框架 

---
# Tunable Virtual IMU Frame by Weighted Averaging of Multiple Non-Collocated IMUs 

**Title (ZH)**: 可调虚拟IMU框架通过多个非对齐IMU加权平均实现 

**Authors**: Yizhou Gao, Tim Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2506.00371)  

**Abstract**: We present a new method to combine several rigidly connected but physically separated IMUs through a weighted average into a single virtual IMU (VIMU). This has the benefits of (i) reducing process noise through averaging, and (ii) allowing for tuning the location of the VIMU. The VIMU can be placed to be coincident with, for example, a camera frame or GNSS frame, thereby offering a quality-of-life improvement for users. Specifically, our VIMU removes the need to consider any lever-arm terms in the propagation model. We also present a quadratic programming method for selecting the weights to minimize the noise of the VIMU while still selecting the placement of its reference frame. We tested our method in simulation and validated it on a real dataset. The results show that our averaging technique works for IMUs with large separation and performance gain is observed in both the simulation and the real experiment compared to using only a single IMU. 

**Abstract (ZH)**: 我们提出了一种通过加权平均将多个物理上分离但刚性连接的IMU组合成单一虚拟IMU（VIMU）的新方法。这种方法的好处包括：（i）通过平均减少过程噪声，（ii）允许调整VIMU的位置。VIMU可以放置为与，例如，相机框架或GNSS框架重合，从而为用户提供生活质量的改善。具体而言，我们的VIMU消除了在传播模型中考虑任何力臂项的需求。我们还提出了一种二次规划方法来选择权重，以最小化VIMU的噪声同时选择其参考框架的位置。我们在仿真中测试了该方法并在实际数据集上进行了验证。结果显示，对于具有较大分离度的IMU，我们的平均技术是有效的，在仿真和实际实验中都观察到了使用单一IMU的性能提升。 

---
# Haptic Rapidly-Exploring Random Trees: A Sampling-based Planner for Quasi-static Manipulation Tasks 

**Title (ZH)**: 触觉快速探索随机树：一种基于采样的准静态操作任务规划器 

**Authors**: Lin Yang, Huu-Thiet Nguyen, Donghan Yu, Chen Lv, Domenico Campolo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00351)  

**Abstract**: In this work, we explore how conventional motion planning algorithms can be reapplied to contact-rich manipulation tasks. Rather than focusing solely on efficiency, we investigate how manipulation aspects can be recast in terms of conventional motion-planning algorithms. Conventional motion planners, such as Rapidly-Exploring Random Trees (RRT), typically compute collision-free paths in configuration space. However, in manipulation tasks, intentional contact is often necessary. For example, when dealing with a crowded bookshelf, a robot must strategically push books aside before inserting a new one. In such scenarios, classical motion planners often fail because of insufficient space. As such, we presents Haptic Rapidly-Exploring Random Trees (HapticRRT), a planning algorithm that incorporates a recently proposed optimality measure in the context of \textit{quasi-static} manipulation, based on the (squared) Hessian of manipulation potential. The key contributions are i) adapting classical RRT to a framework that re-frames quasi-static manipulation as a planning problem on an implicit equilibrium manifold; ii) discovering multiple manipulation strategies, corresponding to branches of the equilibrium manifold. iii) providing deeper insight to haptic obstacle and haptic metric, enhancing interpretability. We validate our approach on a simulated pendulum and a real-world crowded bookshelf task, demonstrating its ability to autonomously discover strategic wedging-in policies and multiple branches. The video can be found at this https URL 

**Abstract (ZH)**: 基于触觉的快速扩展随机树在接触富有的 manipulation 任务中的应用 

---
# Music-driven Robot Swarm Painting 

**Title (ZH)**: 音乐驱动的机器人 swarm 绘画 

**Authors**: Jingde Cheng, Gennaro Notomista  

**Link**: [PDF](https://arxiv.org/pdf/2506.00326)  

**Abstract**: This paper proposes a novel control framework for robotic swarms capable of turning a musical input into a painting. The approach connects the two artistic domains, music and painting, leveraging their respective connections to fundamental emotions. The robotic units of the swarm are controlled in a coordinated fashion using a heterogeneous coverage policy to control the motion of the robots which continuously release traces of color in the environment. The results of extensive simulations performed starting from different musical inputs and with different color equipments are reported. Finally, the proposed framework has been implemented on real robots equipped with LED lights and capable of light-painting. 

**Abstract (ZH)**: 本文提出了一种新颖的控制框架，能够将 musical 输入转化为绘画作品，该框架将音乐和绘画这两种艺术领域连接起来，利用它们与基本情感的关联。集群中的机器人单元以协调的方式受控，并使用异构覆盖策略控制机器人的运动，使其在环境中不断释放颜色痕迹。从不同的音乐输入和不同颜色设备出发进行的大量仿真结果被报道。最后，提出的框架已在配备 LED 灯光并能够进行光绘的实体机器人上实现。 

---
# Learning Aerodynamics for the Control of Flying Humanoid Robots 

**Title (ZH)**: 学习气动特性以控制飞行类人机器人 

**Authors**: Antonello Paolino, Gabriele Nava, Fabio Di Natale, Fabio Bergonti, Punith Reddy Vanteddu, Donato Grassi, Luca Riccobene, Alex Zanotti, Renato Tognaccini, Gianluca Iaccarino, Daniele Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2506.00305)  

**Abstract**: Robots with multi-modal locomotion are an active research field due to their versatility in diverse environments. In this context, additional actuation can provide humanoid robots with aerial capabilities. Flying humanoid robots face challenges in modeling and control, particularly with aerodynamic forces. This paper addresses these challenges from a technological and scientific standpoint. The technological contribution includes the mechanical design of iRonCub-Mk1, a jet-powered humanoid robot, optimized for jet engine integration, and hardware modifications for wind tunnel experiments on humanoid robots for precise aerodynamic forces and surface pressure measurements. The scientific contribution offers a comprehensive approach to model and control aerodynamic forces using classical and learning techniques. Computational Fluid Dynamics (CFD) simulations calculate aerodynamic forces, validated through wind tunnel experiments on iRonCub-Mk1. An automated CFD framework expands the aerodynamic dataset, enabling the training of a Deep Neural Network and a linear regression model. These models are integrated into a simulator for designing aerodynamic-aware controllers, validated through flight simulations and balancing experiments on the iRonCub-Mk1 physical prototype. 

**Abstract (ZH)**: 具有多模态运动的机器人是活跃的研究领域，由于其在多种环境中的灵活性。在这种背景下，额外的动力装置可以为类人机器人提供飞行能力。飞行类人机器人在建模和控制方面面临挑战，尤其是在气动力方面。本文从技术和科学的角度解决了这些挑战。技术贡献包括iRonCub-Mk1机械设计，这是一种配备喷气发动机的类人机器人，优化了喷气发动机的集成，并进行了硬件修改以在类人机器人上进行风洞实验，以精确测量气动力和表面压力。科学贡献提供了一种全面的气动力建模和控制方法，结合了经典技术和学习技术。通过计算流体动力学（CFD）模拟计算气动力，并通过iRonCub-Mk1的风洞实验进行验证。自动化的CFD框架扩展了气动力数据集，使深度神经网络和线性回归模型的训练成为可能。这些模型被集成到一个模拟器中，用于设计气动力感知控制器，并通过飞行模拟和iRonCub-Mk1物理原型的平衡实验进行了验证。 

---
# Lazy Heuristic Search for Solving POMDPs with Expensive-to-Compute Belief Transitions 

**Title (ZH)**: 昂贵信念转移计算的POMDPs懒惰启发式搜索 

**Authors**: Muhammad Suhail Saleem, Rishi Veerapaneni, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2506.00285)  

**Abstract**: Heuristic search solvers like RTDP-Bel and LAO* have proven effective for computing optimal and bounded sub-optimal solutions for Partially Observable Markov Decision Processes (POMDPs), which are typically formulated as belief MDPs. A belief represents a probability distribution over possible system states. Given a parent belief and an action, computing belief state transitions involves Bayesian updates that combine the transition and observation models of the POMDP to determine successor beliefs and their transition probabilities. However, there is a class of problems, specifically in robotics, where computing these transitions can be prohibitively expensive due to costly physics simulations, raycasting, or expensive collision checks required by the underlying transition and observation models, leading to long planning times. To address this challenge, we propose Lazy RTDP-Bel and Lazy LAO*, which defer computing expensive belief state transitions by leveraging Q-value estimation, significantly reducing planning time. We demonstrate the superior performance of the proposed lazy planners in domains such as contact-rich manipulation for pose estimation, outdoor navigation in rough terrain, and indoor navigation with a 1-D LiDAR sensor. Additionally, we discuss practical Q-value estimation techniques for commonly encountered problem classes that our lazy planners can leverage. Our results show that lazy heuristic search methods dramatically improve planning speed by postponing expensive belief transition evaluations while maintaining solution quality. 

**Abstract (ZH)**: Heuristic搜索求解器如RTDP-Bel和LAO*在计算部分可观测马尔可夫决策过程（POMDPs）的最优和次优解方面 proven有效，这些过程通常被公式化为信念MDP。一个信念表示可能系统状态的概率分布。给定一个父信念和一个动作，计算信念状态转移涉及贝叶斯更新，结合POMDP的转移模型和观测模型来确定后续信念及其转移概率。然而，在机器人领域存在一类问题，由于涉及昂贵的物理模拟、射线投射或基础转移和观测模型所需的昂贵碰撞检测，计算这些转移可能会导致规划时间过长。为了解决这一挑战，我们提出了Lazy RTDP-Bel和Lazy LAO*，通过利用Q值估算推迟计算昂贵的信念状态转移，显著减少了规划时间。我们在接触丰富的操作姿态估计、崎岖地形户外导航和配备一维LiDAR的室内导航等领域展示了所提懒惰规划器的优越性能。此外，我们还讨论了懒惰规划器可以利用的常见问题类别中的实用Q值估算技术。我们的结果表明，懒惰启发式搜索方法通过推迟昂贵的信念转移评估显著提高了规划速度，同时保持了解的质量。 

---
# RoboMoRe: LLM-based Robot Co-design via Joint Optimization of Morphology and Reward 

**Title (ZH)**: RoboMoRe：基于LLM的机器人联合形态与奖励优化协同设计 

**Authors**: Jiawei Fang, Yuxuan Sun, Chengtian Ma, Qiuyu Lu, Lining Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00276)  

**Abstract**: Robot co-design, jointly optimizing morphology and control policy, remains a longstanding challenge in the robotics community, where many promising robots have been developed. However, a key limitation lies in its tendency to converge to sub-optimal designs due to the use of fixed reward functions, which fail to explore the diverse motion modes suitable for different morphologies. Here we propose RoboMoRe, a large language model (LLM)-driven framework that integrates morphology and reward shaping for co-optimization within the robot co-design loop. RoboMoRe performs a dual-stage optimization: in the coarse optimization stage, an LLM-based diversity reflection mechanism generates both diverse and high-quality morphology-reward pairs and efficiently explores their distribution. In the fine optimization stage, top candidates are iteratively refined through alternating LLM-guided reward and morphology gradient updates. RoboMoRe can optimize both efficient robot morphologies and their suited motion behaviors through reward shaping. Results demonstrate that without any task-specific prompting or predefined reward/morphology templates, RoboMoRe significantly outperforms human-engineered designs and competing methods across eight different tasks. 

**Abstract (ZH)**: 机器人协同设计中的形态与控制策略联合优化仍是一个长期挑战，尽管已经开发了许多有前景的机器人，但其固定奖励函数的使用导致了亚最优设计的收敛，未能探索适合不同形态的多样化运动模式。为此，我们提出了一种以大型语言模型驱动的框架RoboMoRe，该框架将形态与奖励塑造结合起来，在机器人协同设计循环中共同优化。RoboMoRe执行两阶段优化：在粗优化阶段，基于大型语言模型的多样性反射机制生成多样且高质量的形态-奖励配对，并高效探索它们的分布；在细优化阶段，通过交替的基于大型语言模型的奖励和形态梯度更新，逐步优化顶级候选方案。RoboMoRe能够通过奖励塑造优化高效的机器人形态及其相应的运动行为。结果表明，无需任何特定任务的提示或预定义的奖励/形态模板，RoboMoRe在八个不同任务上显著优于人工设计和竞争方法。 

---
# Understanding while Exploring: Semantics-driven Active Mapping 

**Title (ZH)**: 探索中的理解：语义驱动的主动建图 

**Authors**: Liyan Chen, Huangying Zhan, Hairong Yin, Yi Xu, Philippos Mordohai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00225)  

**Abstract**: Effective robotic autonomy in unknown environments demands proactive exploration and precise understanding of both geometry and semantics. In this paper, we propose ActiveSGM, an active semantic mapping framework designed to predict the informativeness of potential observations before execution. Built upon a 3D Gaussian Splatting (3DGS) mapping backbone, our approach employs semantic and geometric uncertainty quantification, coupled with a sparse semantic representation, to guide exploration. By enabling robots to strategically select the most beneficial viewpoints, ActiveSGM efficiently enhances mapping completeness, accuracy, and robustness to noisy semantic data, ultimately supporting more adaptive scene exploration. Our experiments on the Replica and Matterport3D datasets highlight the effectiveness of ActiveSGM in active semantic mapping tasks. 

**Abstract (ZH)**: 有效的自主机器人在未知环境中的探索需要主动探索和对几何和语义的精确理解。本文提出了一种名为ActiveSGM的主动语义映射框架，该框架能够在执行前预测潜在观察的信息量。基于3D高斯点云（3DGS）映射骨干网络，该方法结合语义和几何不确定性量化以及稀疏语义表示来引导探索。通过使机器人能够战略性地选择最具益处的视角，ActiveSGM有效地增强了映射的完整性和准确性，并提高了对噪声语义数据的鲁棒性，最终支持更适应的场景探索。我们在Replica和Matterport3D数据集上的实验展示了ActiveSGM在主动语义映射任务中的有效性。 

---
# AniTrack: A Power-Efficient, Time-Slotted and Robust UWB Localization System for Animal Tracking in a Controlled Setting 

**Title (ZH)**: AniTrack：一种适用于受控环境中的动物跟踪的高效、时-slot化和稳健的UWB定位系统 

**Authors**: Victor Luder, Lukas Schulthess, Silvano Cortesi, Leyla Rivero Davis, Michele Magno  

**Link**: [PDF](https://arxiv.org/pdf/2506.00216)  

**Abstract**: Accurate localization is essential for a wide range of applications, including asset tracking, smart agriculture, and an- imal monitoring. While traditional localization methods, such as Global Navigation Satellite System (GNSS), Wi-Fi, and Bluetooth Low Energy (BLE), offer varying levels of accuracy and coverage, they have drawbacks regarding power consumption, infrastruc- ture requirements, and deployment flexibility. Ultra-Wideband (UWB) is emerging as an alternative, offering centimeter-level accuracy and energy efficiency, especially suitable for medium to large field monitoring with capabilities to work indoors and outdoors. However, existing UWB localization systems require infrastructure with mains power to supply the anchors, which impedes their scalability and ease of deployment. This under- scores the need for a fully battery-powered and energy-efficient localization system. This paper presents an energy-optimized, battery-operated UWB localization system that leverages Long Range Wide Area Network (LoRaWAN) for data transmission to a server backend. By employing single-sided two-way ranging (SS-TWR) in a time- slotted localization approach, the power consumption both on the anchor and the tag is reduced, while maintaining high accuracy. With a low average power consumption of 20.44 mW per anchor and 7.19 mW per tag, the system allows fully battery- powered operation for up to 25 days, achieving average accuracy of 13.96 cm with self-localizing anchors on a 600 m2 testing ground. To validate its effectiveness and ease of installation in a challenging application scenario, ten anchors and two tags were successfully deployed in a tropical zoological biome where they could be used to track Aldabra Giant Tortoises (Aldabrachelys gigantea). 

**Abstract (ZH)**: 一种基于LoRaWAN的数据传输的低功耗UWB定位系统及其在热带生物圈的应用研究 

---
# Interactive Imitation Learning for Dexterous Robotic Manipulation: Challenges and Perspectives -- A Survey 

**Title (ZH)**: Dexterous 机器人操作的交互式模仿学习：挑战与展望——综述 

**Authors**: Edgar Welte, Rania Rayyes  

**Link**: [PDF](https://arxiv.org/pdf/2506.00098)  

**Abstract**: Dexterous manipulation is a crucial yet highly complex challenge in humanoid robotics, demanding precise, adaptable, and sample-efficient learning methods. As humanoid robots are usually designed to operate in human-centric environments and interact with everyday objects, mastering dexterous manipulation is critical for real-world deployment. Traditional approaches, such as reinforcement learning and imitation learning, have made significant strides, but they often struggle due to the unique challenges of real-world dexterous manipulation, including high-dimensional control, limited training data, and covariate shift. This survey provides a comprehensive overview of these challenges and reviews existing learning-based methods for dexterous manipulation, spanning imitation learning, reinforcement learning, and hybrid approaches. A promising yet underexplored direction is interactive imitation learning, where human feedback actively refines a robot's behavior during training. While interactive imitation learning has shown success in various robotic tasks, its application to dexterous manipulation remains limited. To address this gap, we examine current interactive imitation learning techniques applied to other robotic tasks and discuss how these methods can be adapted to enhance dexterous manipulation. By synthesizing state-of-the-art research, this paper highlights key challenges, identifies gaps in current methodologies, and outlines potential directions for leveraging interactive imitation learning to improve dexterous robotic skills. 

**Abstract (ZH)**: 灵巧 manipulation 是类人机器人研究中一个关键但极具挑战性的课题，要求精确、适应性强且样本高效的算法。由于类人机器人通常设计用于人类中心环境并操作日常物体，因此掌握灵巧 manipulation 对于实际部署至关重要。传统方法，如强化学习和模仿学习，虽取得了显著进展，但在现实世界的灵巧 manipulation 中面临的高维控制、有限训练数据和协变量偏移等挑战常常难以克服。本文综述了这些挑战，并回顾了现有的基于学习的灵巧 manipulation 方法，涵盖模仿学习、强化学习及其混合方法。一个有前景但尚未充分探索的方向是互动模仿学习，其中人类反馈在训练过程中主动优化机器人的行为。虽然互动模仿学习在各种机器人任务中取得了成功，但在灵巧 manipulation 中的应用仍有限。为解决这一差距，本文讨论了当前应用于其他机器人任务的互动模仿学习技术，并探讨了如何调整这些方法以增强灵巧 manipulation。通过综合最新的研究成果，本文指出了关键挑战，识别了当前方法中的不足，并概述了利用互动模仿学习提高机器人灵巧技能的潜在方向。 

---
# Navigation of a Three-Link Microswimmer via Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的三连杆微型游泳器导航 

**Authors**: Yuyang Lai, Sina Heydari, On Shun Pak, Yi Man  

**Link**: [PDF](https://arxiv.org/pdf/2506.00084)  

**Abstract**: Motile microorganisms develop effective swimming gaits to adapt to complex biological environments. Translating this adaptability to smart microrobots presents significant challenges in motion planning and stroke design. In this work, we explore the use of reinforcement learning (RL) to develop stroke patterns for targeted navigation in a three-link swimmer model at low Reynolds numbers. Specifically, we design two RL-based strategies: one focusing on maximizing velocity (Velocity-Focused Strategy) and another balancing velocity with energy consumption (Energy-Aware Strategy). Our results demonstrate how the use of different reward functions influences the resulting stroke patterns developed via RL, which are compared with those obtained from traditional optimization methods. Furthermore, we showcase the capability of the RL-powered swimmer in adapting its stroke patterns in performing different navigation tasks, including tracing complex trajectories and pursuing moving targets. Taken together, this work highlights the potential of reinforcement learning as a versatile tool for designing efficient and adaptive microswimmers capable of sophisticated maneuvers in complex environments. 

**Abstract (ZH)**: 可游动微生物通过发展有效的游泳姿态来适应复杂的生物环境。将这种适应性移植到智能微机器人中在运动规划和摆动设计方面提出了重大挑战。在本工作中，我们探索使用强化学习（RL）来为低雷诺数下的三链接游泳者模型开发定向导航的摆动模式。具体地，我们设计了两种基于RL的策略：一种专注于最大化速度（速度导向策略）和另一种在速度与能耗之间寻求平衡（能量感知策略）。我们的结果表明，不同的奖励函数如何影响通过RL产生的摆动模式，并将这些结果与传统优化方法获得的结果进行比较。此外，我们展示了基于RL的游泳者在执行不同导航任务（包括跟踪复杂轨迹和追逐移动目标）时调整其摆动模式的能力。总体而言，本工作强调了强化学习作为设计高效且适应性强的微游泳者工具的潜力，这些微游泳者能够在复杂环境中执行复杂的操控动作。 

---
# Hi-Dyna Graph: Hierarchical Dynamic Scene Graph for Robotic Autonomy in Human-Centric Environments 

**Title (ZH)**: Hi-Dyna 图：以人为本环境中的分层动态场景图用于机器人自主性 

**Authors**: Jiawei Hou, Xiangyang Xue, Taiping Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00083)  

**Abstract**: Autonomous operation of service robotics in human-centric scenes remains challenging due to the need for understanding of changing environments and context-aware decision-making. While existing approaches like topological maps offer efficient spatial priors, they fail to model transient object relationships, whereas dense neural representations (e.g., NeRF) incur prohibitive computational costs. Inspired by the hierarchical scene representation and video scene graph generation works, we propose Hi-Dyna Graph, a hierarchical dynamic scene graph architecture that integrates persistent global layouts with localized dynamic semantics for embodied robotic autonomy. Our framework constructs a global topological graph from posed RGB-D inputs, encoding room-scale connectivity and large static objects (e.g., furniture), while environmental and egocentric cameras populate dynamic subgraphs with object position relations and human-object interaction patterns. A hybrid architecture is conducted by anchoring these subgraphs to the global topology using semantic and spatial constraints, enabling seamless updates as the environment evolves. An agent powered by large language models (LLMs) is employed to interpret the unified graph, infer latent task triggers, and generate executable instructions grounded in robotic affordances. We conduct complex experiments to demonstrate Hi-Dyna Grap's superior scene representation effectiveness. Real-world deployments validate the system's practicality with a mobile manipulator: robotics autonomously complete complex tasks with no further training or complex rewarding in a dynamic scene as cafeteria assistant. See this https URL for video demonstration and more details. 

**Abstract (ZH)**: 基于人类中心场景的服务机器人自主操作仍具有挑战性，因其需要理解和进行情境意识下的决策。虽然现有的方法如拓扑地图提供有效的空间先验，但无法建模瞬态对象关系，而密集神经表示（例如，NeRF）则会带来高昂的计算成本。受分层场景表示和视频场景图生成工作的启发，我们提出了一种分层动态场景图架构Hi-Dyna Graph，该架构整合了持久的全局布局与局部动态语义，以实现沉浸式机器人自主性。我们的框架从姿态化的RGB-D输入构建全局拓扑图，编码房间尺度的连接性和大规模静态对象（例如，家具），同时环境和视角摄像头填充动态子图，包括对象位置关系和人机交互模式。该分层架构通过语义和空间约束将这些子图锚定到全局拓扑中，从而在环境变化时实现无缝更新。基于大型语言模型的代理使用该统一图进行解释，推断潜在的任务触发，并生成以机器人功能性为基础的可执行指令。我们进行了复杂的实验以展示Hi-Dyna Graph在场景表示上的优越效果。实际部署验证了该系统的实用性，通过一个移动 manipulator，机器人在动态场景中（例如在食堂助手角色中）无需进一步训练或复杂的奖励机制就能自主完成复杂任务。更多细节和视频演示请参见：见此链接。 

---
# Reducing Latency in LLM-Based Natural Language Commands Processing for Robot Navigation 

**Title (ZH)**: 基于LLM的自然语言命令处理中降低机器人导航延迟 

**Authors**: Diego Pollini, Bruna V. Guterres, Rodrigo S. Guerra, Ricardo B. Grando  

**Link**: [PDF](https://arxiv.org/pdf/2506.00075)  

**Abstract**: The integration of Large Language Models (LLMs), such as GPT, in industrial robotics enhances operational efficiency and human-robot collaboration. However, the computational complexity and size of these models often provide latency problems in request and response times. This study explores the integration of the ChatGPT natural language model with the Robot Operating System 2 (ROS 2) to mitigate interaction latency and improve robotic system control within a simulated Gazebo environment. We present an architecture that integrates these technologies without requiring a middleware transport platform, detailing how a simulated mobile robot responds to text and voice commands. Experimental results demonstrate that this integration improves execution speed, usability, and accessibility of the human-robot interaction by decreasing the communication latency by 7.01\% on average. Such improvements facilitate smoother, real-time robot operations, which are crucial for industrial automation and precision tasks. 

**Abstract (ZH)**: 大语言模型（LLMs）如GPT在工业机器人中的集成增强了操作效率和人机协作。然而，这些模型的计算复杂度和大小常导致请求和响应时间的延迟问题。本研究探讨了将ChatGPT自然语言模型与Robot Operating System 2（ROS 2）集成以减轻交互延迟并改善模拟Gazebo环境中的机器人系统控制。我们提出了一种无需中间件传输平台的技术架构，详细说明了模拟移动机器人如何响应文本和语音命令。实验结果表明，这种集成通过平均减少7.01%的通信延迟，提高了人机交互的执行速度、可用性和访问性，从而促进更顺畅、实时的机器人操作，这对于工业自动化和精确任务至关重要。 

---
# Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics 

**Title (ZH)**: Robot-R1: 强化学习增强机器人具身推理 

**Authors**: Dongyoung Kim, Sumin Park, Huiwon Jang, Jinwoo Shin, Jaehyung Kim, Younggyo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00070)  

**Abstract**: Large Vision-Language Models (LVLMs) have recently shown great promise in advancing robotics by combining embodied reasoning with robot control. A common approach involves training on embodied reasoning tasks related to robot control using Supervised Fine-Tuning (SFT). However, SFT datasets are often heuristically constructed and not explicitly optimized for improving robot control. Furthermore, SFT often leads to issues such as catastrophic forgetting and reduced generalization performance. To address these limitations, we introduce Robot-R1, a novel framework that leverages reinforcement learning to enhance embodied reasoning specifically for robot control. Robot-R1 learns to predict the next keypoint state required for task completion, conditioned on the current scene image and environment metadata derived from expert demonstrations. Inspired by the DeepSeek-R1 learning approach, Robot-R1 samples reasoning-based responses and reinforces those that lead to more accurate predictions. Our experiments show that models trained with Robot-R1 outperform SFT methods on embodied reasoning tasks. Despite having only 7B parameters, Robot-R1 even surpasses GPT-4o on reasoning tasks related to low-level action control, such as spatial and primitive movement reasoning. 

**Abstract (ZH)**: Large Vision-Language Models for Robotics via Reinforcement Learning (Robot-R1) 

---
# From Motion to Behavior: Hierarchical Modeling of Humanoid Generative Behavior Control 

**Title (ZH)**: 从运动到行为：类人生成行为控制的层次模型 

**Authors**: Jusheng Zhang, Jinzhou Tang, Sidi Liu, Mingyan Li, Sheng Zhang, Jian Wang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00043)  

**Abstract**: Human motion generative modeling or synthesis aims to characterize complicated human motions of daily activities in diverse real-world environments. However, current research predominantly focuses on either low-level, short-period motions or high-level action planning, without taking into account the hierarchical goal-oriented nature of human activities. In this work, we take a step forward from human motion generation to human behavior modeling, which is inspired by cognitive science. We present a unified framework, dubbed Generative Behavior Control (GBC), to model diverse human motions driven by various high-level intentions by aligning motions with hierarchical behavior plans generated by large language models (LLMs). Our insight is that human motions can be jointly controlled by task and motion planning in robotics, but guided by LLMs to achieve improved motion diversity and physical fidelity. Meanwhile, to overcome the limitations of existing benchmarks, i.e., lack of behavioral plans, we propose GBC-100K dataset annotated with a hierarchical granularity of semantic and motion plans driven by target goals. Our experiments demonstrate that GBC can generate more diverse and purposeful high-quality human motions with 10* longer horizons compared with existing methods when trained on GBC-100K, laying a foundation for future research on behavioral modeling of human motions. Our dataset and source code will be made publicly available. 

**Abstract (ZH)**: 基于生成的行为控制 modeling human motions via large language models and hierarchical behavior planning 

---
# GaussianFusion: Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving 

**Title (ZH)**: GaussianFusion: 基于高斯过程的端到端多传感器融合 

**Authors**: Shuai Liu, Quanmin Liang, Zefeng Li, Boyang Li, Kai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00034)  

**Abstract**: Multi-sensor fusion is crucial for improving the performance and robustness of end-to-end autonomous driving systems. Existing methods predominantly adopt either attention-based flatten fusion or bird's eye view fusion through geometric transformations. However, these approaches often suffer from limited interpretability or dense computational overhead. In this paper, we introduce GaussianFusion, a Gaussian-based multi-sensor fusion framework for end-to-end autonomous driving. Our method employs intuitive and compact Gaussian representations as intermediate carriers to aggregate information from diverse sensors. Specifically, we initialize a set of 2D Gaussians uniformly across the driving scene, where each Gaussian is parameterized by physical attributes and equipped with explicit and implicit features. These Gaussians are progressively refined by integrating multi-modal features. The explicit features capture rich semantic and spatial information about the traffic scene, while the implicit features provide complementary cues beneficial for trajectory planning. To fully exploit rich spatial and semantic information in Gaussians, we design a cascade planning head that iteratively refines trajectory predictions through interactions with Gaussians. Extensive experiments on the NAVSIM and Bench2Drive benchmarks demonstrate the effectiveness and robustness of the proposed GaussianFusion framework. The source code will be released at this https URL. 

**Abstract (ZH)**: 基于高斯分布的端到端自主驾驶多传感器融合框架 

---
# Buoyant Choreographies: Harmonies of Light, Sound, and Human Connection 

**Title (ZH)**: 浮力 choreography：光、音与人联结的和谐 

**Authors**: Dennis Hong, Yusuke Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2506.00021)  

**Abstract**: BALLU, the Buoyancy Assisted Lightweight Legged Unit, is a unique legged robot with a helium balloon body and articulated legs \fig{fig:fig1}. Since it is buoyant-assisted, BALLU is inherently stable, never falling over, while being able to walk, jump, and interact safely with people. The BALLU art installation builds on this playful platform to express fluidity, serendipity, and connection. It transforms robotic motion into an artistic visual and acoustic experience, merging technology and creativity into a dynamic, interactive display. This exhibition intentionally does not have a physical boundary for the robots, emphasizing the harmony of the technologies and humanity. This work significantly extends BALLU's existing permanent exhibition in the Seoul Robotics & Artificial Intelligence Museum, Seoul RAIM (this https URL), emphasizing the harmony of robotics and humanity through visual, acoustic, and physical expression. 

**Abstract (ZH)**: BALLU，一种带有氦气球身体和可动腿的浮力辅助轻型腿足单元，是一种独特的腿足机器人 \fig{fig:fig1}。由于它是浮力辅助的， BALLU 原本就具备稳定性能，不会摔倒，同时能够行走、跳跃，并且能够安全地与人互动。BALLU 艺术装置在此基础上构建了一个充满趣味的平台，表达流动性、偶然性和连接性。它将机器人的运动转变为一种艺术性的视觉和听觉体验，将技术和创意融为一体，形成一个动态、互动的展示。此次展览故意没有为机器人设置物理边界，强调技术和人类的和谐共生。这项工作显著扩展了BALLU在首尔机器人与人工智能博物馆（Seoul RAIM）的永久展览（链接：这个 https URL），通过视觉、听觉和身体上的表达，强调了机器人与人类的和谐共生。 

---
# Online Competitive Information Gathering for Partially Observable Trajectory Games 

**Title (ZH)**: 部分可观测轨迹博弈中的在线竞争性信息收集 

**Authors**: Mel Krusniak, Hang Xu, Parker Palermo, Forrest Laine  

**Link**: [PDF](https://arxiv.org/pdf/2506.01927)  

**Abstract**: Game-theoretic agents must make plans that optimally gather information about their opponents. These problems are modeled by partially observable stochastic games (POSGs), but planning in fully continuous POSGs is intractable without heavy offline computation or assumptions on the order of belief maintained by each player. We formulate a finite history/horizon refinement of POSGs which admits competitive information gathering behavior in trajectory space, and through a series of approximations, we present an online method for computing rational trajectory plans in these games which leverages particle-based estimations of the joint state space and performs stochastic gradient play. We also provide the necessary adjustments required to deploy this method on individual agents. The method is tested in continuous pursuit-evasion and warehouse-pickup scenarios (alongside extensions to $N > 2$ players and to more complex environments with visual and physical obstacles), demonstrating evidence of active information gathering and outperforming passive competitors. 

**Abstract (ZH)**: 游戏论代理必须制定最优信息收集其对手的信息的计划。这些问题通过部分可观测随机博弈（POSGs）进行建模，但在完全连续的POSGs中进行计划需要大量的离线计算或对每个玩家保持belief顺序的假设。我们提出了POSGs的一个有限历史/时间剖分，它在轨迹空间中支持竞争性的信息收集行为，并通过一系列近似，我们提出了一个在线方法来计算这些博弈中的理性轨迹计划，该方法利用基于粒子的状态空间联合估计，并进行随机梯度博弈。我们还提供了在单个代理上部署此方法所需的必要调整。该方法在连续的追逐-逃避和仓库取货场景中进行了测试（并扩展到N>2个玩家以及更具复杂性的环境中，包含视觉和物理障碍），证明了积极信息收集的证据并优于被动竞争对手。 

---
# SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics 

**Title (ZH)**: SmolVLA：一种经济高效的人机协作模型 

**Authors**: Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, Remi Cadene  

**Link**: [PDF](https://arxiv.org/pdf/2506.01844)  

**Abstract**: Vision-language models (VLMs) pretrained on large-scale multimodal datasets encode rich visual and linguistic knowledge, making them a strong foundation for robotics. Rather than training robotic policies from scratch, recent approaches adapt VLMs into vision-language-action (VLA) models that enable natural language-driven perception and control. However, existing VLAs are typically massive--often with billions of parameters--leading to high training costs and limited real-world deployability. Moreover, they rely on academic and industrial datasets, overlooking the growing availability of community-collected data from affordable robotic platforms. In this work, we present SmolVLA, a small, efficient, and community-driven VLA that drastically reduces both training and inference costs, while retaining competitive performance. SmolVLA is designed to be trained on a single GPU and deployed on consumer-grade GPUs or even CPUs. To further improve responsiveness, we introduce an asynchronous inference stack decoupling perception and action prediction from action execution, allowing higher control rates with chunked action generation. Despite its compact size, SmolVLA achieves performance comparable to VLAs that are 10x larger. We evaluate SmolVLA on a range of both simulated as well as real-world robotic benchmarks and release all code, pretrained models, and training data. 

**Abstract (ZH)**: 基于视觉-语言的小型高效社区驱动机器人模型：SmolVLA 

---
# unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning 

**Title (ZH)**: 无监督多对象分割中心-边界推理 

**Authors**: Yafei Yang, Zihui Zhang, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01778)  

**Abstract**: We study the challenging problem of unsupervised multi-object segmentation on single images. Existing methods, which rely on image reconstruction objectives to learn objectness or leverage pretrained image features to group similar pixels, often succeed only in segmenting simple synthetic objects or discovering a limited number of real-world objects. In this paper, we introduce unMORE, a novel two-stage pipeline designed to identify many complex objects in real-world images. The key to our approach involves explicitly learning three levels of carefully defined object-centric representations in the first stage. Subsequently, our multi-object reasoning module utilizes these learned object priors to discover multiple objects in the second stage. Notably, this reasoning module is entirely network-free and does not require human labels. Extensive experiments demonstrate that unMORE significantly outperforms all existing unsupervised methods across 6 real-world benchmark datasets, including the challenging COCO dataset, achieving state-of-the-art object segmentation results. Remarkably, our method excels in crowded images where all baselines collapse. 

**Abstract (ZH)**: 我们研究单张图像上无监督多对象分割的具有挑战性问题。现有方法依赖于图像重构目标来学习对象性或利用预训练图像特征来聚类相似像素，往往只能成功分割简单的合成对象或发现少量的真实世界对象。在本文中，我们引入了unMORE，这是一种新颖的两阶段管道，旨在识别真实世界图像中的许多复杂对象。我们的方法的关键在于在第一阶段明确学习三个层次的对象中心表示。随后，我们的多对象推理模块利用这些学习到的对象先验在第二阶段发现多个对象。值得注意的是，此推理模块完全不依赖网络，也不需要人工标签。广泛的实验结果表明，unMORE在包括具有挑战性的COCO数据集在内的6个真实世界基准数据集上显著优于所有现有的无监督方法，达到了最先进的对象分割结果。更重要的是，我们的方法在所有基线方法失效的密集场景中表现出色。 

---
# Provably Safe Reinforcement Learning from Analytic Gradients 

**Title (ZH)**: 可验证安全的基于分析梯度的强化学习 

**Authors**: Tim Walter, Hannah Markgraf, Jonathan Külz, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.01665)  

**Abstract**: Deploying autonomous robots in safety-critical applications requires safety guarantees. Provably safe reinforcement learning is an active field of research which aims to provide such guarantees using safeguards. These safeguards should be integrated during training to prevent a large sim-to-real gap. While there are several approaches for safeguarding sampling-based reinforcement learning, analytic gradient-based reinforcement learning often achieves superior performance and sample efficiency. However, there is no safeguarding approach for this learning paradigm yet. Our work addresses this gap by developing the first effective safeguard for analytic gradient-based reinforcement learning. We analyse existing, differentiable safeguards, adapt them through modified mappings and gradient formulations, and integrate them with a state-of-the-art learning algorithm and a differentiable simulation. We evaluate how different safeguards affect policy optimisation using numerical experiments on two classical control tasks. The results demonstrate safeguarded training without compromising performance. 

**Abstract (ZH)**: 部署自主机器人于安全关键应用需要安全保证。可验证安全性强化学习是研究活跃领域，旨在通过防护措施提供此类保证。这些防护措施应在训练期间集成，以防止模拟与现实之间的差距。虽然已有几种基于采样的强化学习防护方法，但基于分析梯度的强化学习通常能够实现更优的性能和采样效率。然而，尚无针对此学习范式的防护方法。我们的工作通过开发第一个有效的基于分析梯度的强化学习防护措施来填补这一空白。我们分析了现有的可微防护措施，通过修改映射和梯度公式进行调整，并将它们与最先进的学习算法和可微模拟集成。我们通过两个经典控制任务的数值实验评估不同防护措施对策略优化的影响。结果表明，防护措施在不牺牲性能的情况下可以实现有效的训练。 

---
# General agents need world models 

**Title (ZH)**: 通用代理需要世界模型 

**Authors**: Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt  

**Link**: [PDF](https://arxiv.org/pdf/2506.01622)  

**Abstract**: Are world models a necessary ingredient for flexible, goal-directed behaviour, or is model-free learning sufficient? We provide a formal answer to this question, showing that any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment. We show that this model can be extracted from the agent's policy, and that increasing the agents performance or the complexity of the goals it can achieve requires learning increasingly accurate world models. This has a number of consequences: from developing safe and general agents, to bounding agent capabilities in complex environments, and providing new algorithms for eliciting world models from agents. 

**Abstract (ZH)**: 世界模型是实现灵活、目标导向行为的必要成分，还是无模型学习就已足够？我们提供了对该问题的正式回答，表明任何能够泛化到多步目标导向任务的智能体都必须学习到其环境的预测模型。我们展示了可以从智能体的策略中提取该模型，并且提高智能体的性能或增加其可以实现的目标复杂性需要更为准确的世界模型。这一结论具有多个重要后果：从开发安全和通用的智能体，到在复杂环境中限定智能体的能力，以及为从智能体中提取世界模型提供新的算法。 

---
# Trajectory First: A Curriculum for Discovering Diverse Policies 

**Title (ZH)**: 先轨迹：一种发现多样化策略的课程学习方法 

**Authors**: Cornelius V. Braun, Sayantan Auddy, Marc Toussaint  

**Link**: [PDF](https://arxiv.org/pdf/2506.01568)  

**Abstract**: Being able to solve a task in diverse ways makes agents more robust to task variations and less prone to local optima. In this context, constrained diversity optimization has emerged as a powerful reinforcement learning (RL) framework to train a diverse set of agents in parallel. However, existing constrained-diversity RL methods often under-explore in complex tasks such as robotic manipulation, leading to a lack in policy diversity. To improve diversity optimization in RL, we therefore propose a curriculum that first explores at the trajectory level before learning step-based policies. In our empirical evaluation, we provide novel insights into the shortcoming of skill-based diversity optimization, and demonstrate empirically that our curriculum improves the diversity of the learned skills. 

**Abstract (ZH)**: 具备多种解决任务的方法可以使智能体更 robust 并减少陷入局部最优解的可能性。在这种背景下，约束多样性优化已成为一种强大的强化学习（RL）框架，用于并行训练一组多样性的智能体。然而，现有的约束多样性 RL 方法往往在诸如机器人 manipulation 等复杂任务中探索不足，导致策略多样性不足。为了改进 RL 中的多样性优化，我们提出了一种课程学习方法，首先在轨迹层面探索，然后再学习基于步骤的策略。在我们的实验评估中，我们提供了关于技能基多样性优化缺陷的新见解，并通过实验证明我们的课程学习方法可以提升学习技能的多样性。 

---
# Captivity-Escape Games as a Means for Safety in Online Motion Generation 

**Title (ZH)**: 基于捕获-逃脱游戏的在线运动生成安全方法 

**Authors**: Christopher Bohn, Manuel Hess, Sören Hohmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.01399)  

**Abstract**: This paper presents a method that addresses the conservatism, computational effort, and limited numerical accuracy of existing frameworks and methods that ensure safety in online model-based motion generation, commonly referred to as fast and safe tracking. Computational limitations restrict online motion planning to low-fidelity models. However, planning with low-fidelity models compromises safety, as the dynamic feasibility of resulting reference trajectories is not ensured. This potentially leads to unavoidable tracking errors that may cause safety-critical constraint violations. Existing frameworks mitigate this safety risk by augmenting safety-critical constraints in motion planning by a safety margin that prevents constraint violations under worst-case tracking errors. However, the methods employed in these frameworks determine the safety margin based on a heuristically selected performance of the planning model, which likely results in overly conservative reference trajectories. Furthermore, these methods are computationally intensive, and the state-of-the-art method is limited in numerical accuracy. We adopt a different perspective and address these limitations with a method that mitigates conservatism in existing frameworks by adapting the planning model performance to a given safety margin. Our method achieves numerical accuracy and requires significantly less computation time than existing methods by leveraging a captivity-escape game, which is a specific zero-sum differential game formulated in this paper. We demonstrate our method using a numerical example and compare it to the state of the art. 

**Abstract (ZH)**: 本文提出了一种方法，用于解决现有确保在线模型导向运动生成安全性的框架和方法中存在的保守性、计算效率低以及数值精度有限的问题，这些框架和方法通常被称为快速安全跟踪。计算限制使得在线运动规划局限于低保真模型。然而，使用低保真模型进行规划会牺牲安全性，因为生成的参考轨迹的动态可行性无法得到保证。这可能导致不可避免的跟踪误差，进而引发安全关键约束的违犯。现有的框架通过在运动规划中增加安全裕度来缓解这种安全性风险，以防止在最坏情况下的跟踪误差导致约束违犯。然而，这些框架中使用的方法是基于规划模型的启发式性能来确定安全裕度的，这可能导致过于保守的参考轨迹。此外，这些方法计算量大，最先进的方法在数值精度上也有局限。本文从不同角度出发，并采用了一种方法来解决这些问题，通过将规划模型的性能调整到给定的安全裕度，缓解现有框架的保守性。本文提出的方法通过利用 captivity-escape 游戏（一种在本文中具体定义的零和微分博弈）实现了数值精度，并比现有方法所需的计算时间显著减少。我们使用数值示例展示了该方法，并将其与最先进的方法进行了比较。 

---
# Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion 

**Title (ZH)**: 基于Zubov采样和迭代领域扩展的两阶段稳定神经控制器学习 

**Authors**: Haoyu Li, Xiangru Zhong, Bin Hu, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01356)  

**Abstract**: Learning-based neural network (NN) control policies have shown impressive empirical performance. However, obtaining stability guarantees and estimations of the region of attraction of these learned neural controllers is challenging due to the lack of stable and scalable training and verification algorithms. Although previous works in this area have achieved great success, much conservatism remains in their framework. In this work, we propose a novel two-stage training framework to jointly synthesize the controller and Lyapunov function for continuous-time systems. By leveraging a Zubov-inspired region of attraction characterization to directly estimate stability boundaries, we propose a novel training data sampling strategy and a domain updating mechanism that significantly reduces the conservatism in training. Moreover, unlike existing works on continuous-time systems that rely on an SMT solver to formally verify the Lyapunov condition, we extend state-of-the-art neural network verifier $\alpha,\!\beta$-CROWN with the capability of performing automatic bound propagation through the Jacobian of dynamical systems and a novel verification scheme that avoids expensive bisection. To demonstrate the effectiveness of our approach, we conduct numerical experiments by synthesizing and verifying controllers on several challenging nonlinear systems across multiple dimensions. We show that our training can yield region of attractions with volume $5 - 1.5\cdot 10^{5}$ times larger compared to the baselines, and our verification on continuous systems can be up to $40-10000$ times faster compared to the traditional SMT solver dReal. Our code is available at this https URL. 

**Abstract (ZH)**: 基于学习的神经网络控制策略在实验性能上表现出色。然而，由于缺乏稳定的和可扩展的训练与验证算法，获得这些学习到的神经网络控制器的稳定性保证以及其吸引域估计仍具有挑战性。尽管该领域已有许多成功的工作，但在其框架中仍存在大量的保守性。本文提出了一种新颖的两阶段训练框架，用于联合合成连续时间系统的控制器和李亚普诺夫函数。通过利用Zubov启发式吸引域表征直接估计稳定性边界，我们提出了一种新的训练数据采样策略和领域更新机制，显著减少了训练的保守性。此外，与现有依赖SMT求解器形式验证李亚普诺夫条件的连续时间系统方法不同，我们扩展了最先进的神经网络验证器α,β-CROWN，使其能够通过动力学系统的雅可比进行自动边界传播，并采用了一种新的验证方案，避免了昂贵的二分法。为了展示我们方法的有效性，我们在多个维度上的多个具有挑战性的非线性系统上合成了并验证了控制器。实验结果表明，我们的训练可以使得吸引域的体积比 baselines 大5到$1.5 \times 10^5$倍，而对连续系统的验证比传统SMT求解器dReal 快40到10000倍。我们的代码可在此网址获得。 

---
# Variational Adaptive Noise and Dropout towards Stable Recurrent Neural Networks 

**Title (ZH)**: 变分自适应噪声和dropout以实现稳定的递归神经网络 

**Authors**: Taisuke Kobayashi, Shingo Murata  

**Link**: [PDF](https://arxiv.org/pdf/2506.01350)  

**Abstract**: This paper proposes a novel stable learning theory for recurrent neural networks (RNNs), so-called variational adaptive noise and dropout (VAND). As stabilizing factors for RNNs, noise and dropout on the internal state of RNNs have been separately confirmed in previous studies. We reinterpret the optimization problem of RNNs as variational inference, showing that noise and dropout can be derived simultaneously by transforming the explicit regularization term arising in the optimization problem into implicit regularization. Their scale and ratio can also be adjusted appropriately to optimize the main objective of RNNs, respectively. In an imitation learning scenario with a mobile manipulator, only VAND is able to imitate sequential and periodic behaviors as instructed. this https URL 

**Abstract (ZH)**: 这种论文提出了一种新的循环神经网络（RNN）稳定学习理论，称为变分自适应噪声和 dropout（VAND）。作为 RNN 的稳定因素，先前研究已分别确认噪声和 dropout 对 RNN 内部状态的稳定性有影响。我们将 RNN 的优化问题重新解读为变分推断，展示了可以通过将优化问题中出现的显式正则化项转换为隐式正则化来同时推导噪声和 dropout。它们的规模和比例也可以适当地调整以优化 RNN 的主要目标。在带有移动 manipulator 的模仿学习场景中，只有 VAND 能够模仿所指示的序列和周期行为。 this https://doi.org/10.1109/IEEECONF.2023.XXXXXX 

---
# Test Automation for Interactive Scenarios via Promptable Traffic Simulation 

**Title (ZH)**: 基于可指令 Traffic 模拟的交互式场景自动化测试 

**Authors**: Augusto Mondelli, Yueshan Li, Alessandro Zanardi, Emilio Frazzoli  

**Link**: [PDF](https://arxiv.org/pdf/2506.01199)  

**Abstract**: Autonomous vehicle (AV) planners must undergo rigorous evaluation before widespread deployment on public roads, particularly to assess their robustness against the uncertainty of human behaviors. While recent advancements in data-driven scenario generation enable the simulation of realistic human behaviors in interactive settings, leveraging these models to construct comprehensive tests for AV planners remains an open challenge. In this work, we introduce an automated method to efficiently generate realistic and safety-critical human behaviors for AV planner evaluation in interactive scenarios. We parameterize complex human behaviors using low-dimensional goal positions, which are then fed into a promptable traffic simulator, ProSim, to guide the behaviors of simulated agents. To automate test generation, we introduce a prompt generation module that explores the goal domain and efficiently identifies safety-critical behaviors using Bayesian optimization. We apply our method to the evaluation of an optimization-based planner and demonstrate its effectiveness and efficiency in automatically generating diverse and realistic driving behaviors across scenarios with varying initial conditions. 

**Abstract (ZH)**: 自主驾驶车辆（AV）规划器在公共道路大规模部署前必须经过严格的评估，特别是评估其在应对人类行为不确定性方面的鲁棒性。虽然近期基于数据驱动的场景生成技术能够模拟现实的人类行为，在交互环境中表现出色，但如何利用这些模型为AV规划器构建全面的测试仍是一项开放性的挑战。在本文中，我们介绍了一种自动方法，用于高效生成现实且安全关键的人类行为，以评估交互场景中的AV规划器。我们使用低维目标位置参数化复杂的人类行为，然后将这些参数输入到可提示的交通模拟器ProSim中，以指导模拟代理的行为。为了自动化测试生成，我们引入了一个提示生成模块，通过贝叶斯优化探索目标域并高效识别安全关键行为。我们将该方法应用于基于优化的规划器评估，并展示了其在自动生成各种初始条件下多样化且现实的驾驶行为方面的有效性和效率。 

---
# Accelerated Learning with Linear Temporal Logic using Differentiable Simulation 

**Title (ZH)**: 使用可微模拟的线性时序逻辑加速学习 

**Authors**: Alper Kamil Bozkurt, Calin Belta, Ming C. Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01167)  

**Abstract**: To ensure learned controllers comply with safety and reliability requirements for reinforcement learning in real-world settings remains challenging. Traditional safety assurance approaches, such as state avoidance and constrained Markov decision processes, often inadequately capture trajectory requirements or may result in overly conservative behaviors. To address these limitations, recent studies advocate the use of formal specification languages such as linear temporal logic (LTL), enabling the derivation of correct-by-construction learning objectives from the specified requirements. However, the sparse rewards associated with LTL specifications make learning extremely difficult, whereas dense heuristic-based rewards risk compromising correctness. In this work, we propose the first method, to our knowledge, that integrates LTL with differentiable simulators, facilitating efficient gradient-based learning directly from LTL specifications by coupling with differentiable paradigms. Our approach introduces soft labeling to achieve differentiable rewards and states, effectively mitigating the sparse-reward issue intrinsic to LTL without compromising objective correctness. We validate the efficacy of our method through experiments, demonstrating significant improvements in both reward attainment and training time compared to the discrete methods. 

**Abstract (ZH)**: 确保在实际应用场景中学习到的控制器符合安全性和可靠性要求仍然是一个挑战。传统的安全保证方法，如状态规避和约束马尔可夫决策过程，往往无法充分捕捉轨迹要求，或者可能导致过度保守的行为。为了解决这些限制，最近的研究提倡使用形式化规范语言，如线性时序逻辑（LTL），从而从指定的要求中推导出构造正确的学习目标。然而，与LTL规范相关的稀疏奖励使得学习变得极其困难，而基于启发式的密集奖励则可能影响正确性。在本文中，我们提出了第一个，据我们所知，将LTL与可微模拟器集成的方法，通过与可微范式耦合，实现直接从LTL规范进行高效梯度学习。我们的方法引入软标签以实现可微奖励和状态，有效缓解了LTL固有的稀疏奖励问题，同时不牺牲目标的正确性。通过实验验证了我们方法的有效性，显示了与离散方法相比，在奖励获取和训练时间方面取得了显著改进。 

---
# Towards Predicting Any Human Trajectory In Context 

**Title (ZH)**: 面向情境预测任意人体轨迹 

**Authors**: Ryo Fujii, Hideo Saito, Ryo Hachiuma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00871)  

**Abstract**: Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, this process is often impractical on edge devices due to constrained computational resources. To address this challenge, we introduce TrajICL, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables rapid adaptation without fine-tuning on the scenario-specific data. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. The code will be released at this https URL. 

**Abstract (ZH)**: 基于上下文学习的行人轨迹预测框架TrajICL：实现快速适应而无需针对场景进行微调 

---
# Adaptive Traffic-Following Scheme for Orderly Distributed Control of Multi-Vehicle Systems 

**Title (ZH)**: 多车辆系统有序分布式控制的自适应交通跟随方案 

**Authors**: Anahita Jain, Husni Idris, John-Paul Clarke, Daniel Delahaye  

**Link**: [PDF](https://arxiv.org/pdf/2506.00703)  

**Abstract**: We present an adaptive control scheme to enable the emergence of order within distributed, autonomous multi-agent systems. Past studies showed that under high-density conditions, order generated from traffic-following behavior reduces travel times, while under low densities, choosing direct paths is more beneficial. In this paper, we leveraged those findings to allow aircraft to independently and dynamically adjust their degree of traffic-following behavior based on the current state of the airspace. This enables aircraft to follow other traffic only when beneficial. Quantitative analyses revealed that dynamic traffic-following behavior results in lower aircraft travel times at the cost of minimal levels of additional disorder to the airspace. The sensitivity of these benefits to temporal and spatial horizons was also investigated. Overall, this work highlights the benefits, and potential necessity, of incorporating self-organizing behavior in making distributed, autonomous multi-agent systems scalable. 

**Abstract (ZH)**: 我们提出一种自适应控制方案以促进分布式自主多agent系统中秩序的出现。 

---
# Position: Olfaction Standardization is Essential for the Advancement of Embodied Artificial Intelligence 

**Title (ZH)**: 位置：嗅觉标准化对于 embodied 人工智的 advancement 至关重要 

**Authors**: Kordel K. France, Rohith Peddi, Nik Dennler, Ovidiu Daescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00398)  

**Abstract**: Despite extraordinary progress in artificial intelligence (AI), modern systems remain incomplete representations of human cognition. Vision, audition, and language have received disproportionate attention due to well-defined benchmarks, standardized datasets, and consensus-driven scientific foundations. In contrast, olfaction - a high-bandwidth, evolutionarily critical sense - has been largely overlooked. This omission presents a foundational gap in the construction of truly embodied and ethically aligned super-human intelligence. We argue that the exclusion of olfactory perception from AI architectures is not due to irrelevance but to structural challenges: unresolved scientific theories of smell, heterogeneous sensor technologies, lack of standardized olfactory datasets, absence of AI-oriented benchmarks, and difficulty in evaluating sub-perceptual signal processing. These obstacles have hindered the development of machine olfaction despite its tight coupling with memory, emotion, and contextual reasoning in biological systems. In this position paper, we assert that meaningful progress toward general and embodied intelligence requires serious investment in olfactory research by the AI community. We call for cross-disciplinary collaboration - spanning neuroscience, robotics, machine learning, and ethics - to formalize olfactory benchmarks, develop multimodal datasets, and define the sensory capabilities necessary for machines to understand, navigate, and act within human environments. Recognizing olfaction as a core modality is essential not only for scientific completeness, but for building AI systems that are ethically grounded in the full scope of the human experience. 

**Abstract (ZH)**: 尽管在人工智能（AI）领域取得了 extraordinary 进步，现代系统仍不完善的人类认知表示。由于定义明确的基准、标准化的数据集和共识驱动的科学基础，视觉、听觉和语言备受关注。相比之下，具有高带宽和进化重要性的嗅觉被很大程度忽视。这一缺失在构建真正具身和伦理上一致的超级智能方面留下了基础性缺口。我们认为，将嗅觉感知排除在 AI 架构之外并非因为不相关，而是由于结构性挑战：未解决的嗅觉科学理论、异构传感器技术、缺乏标准化的嗅觉数据集、缺乏面向 AI 的基准，以及评估亚感知信号处理的难度。这些障碍阻碍了机器嗅觉的发展，尽管它与生物系统中的记忆、情绪和情境推理密切相关。在本文中，我们认为，真正意义上的向通用和具身智能迈进需要人工智能社区在嗅觉研究上进行重大投入。我们呼吁跨学科合作——涵盖神经科学、机器人学、机器学习和伦理学——以正式化嗅觉基准、开发多模态数据集，并定义机器理解、导航和作用于人类环境所需的感觉能力。将嗅觉视为核心模态不仅是出于科学完整性考虑，更是为了构建在人类体验全貌基础上伦理上根植的 AI 系统。 

---
# Ctrl-Crash: Controllable Diffusion for Realistic Car Crashes 

**Title (ZH)**: Ctrl-Crash: 可控扩散模型生成真实车祸场景 

**Authors**: Anthony Gosselin, Ge Ya Luo, Luis Lara, Florian Golemo, Derek Nowrouzezahrai, Liam Paull, Alexia Jolicoeur-Martineau, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00227)  

**Abstract**: Video diffusion techniques have advanced significantly in recent years; however, they struggle to generate realistic imagery of car crashes due to the scarcity of accident events in most driving datasets. Improving traffic safety requires realistic and controllable accident simulations. To tackle the problem, we propose Ctrl-Crash, a controllable car crash video generation model that conditions on signals such as bounding boxes, crash types, and an initial image frame. Our approach enables counterfactual scenario generation where minor variations in input can lead to dramatically different crash outcomes. To support fine-grained control at inference time, we leverage classifier-free guidance with independently tunable scales for each conditioning signal. Ctrl-Crash achieves state-of-the-art performance across quantitative video quality metrics (e.g., FVD and JEDi) and qualitative measurements based on a human-evaluation of physical realism and video quality compared to prior diffusion-based methods. 

**Abstract (ZH)**: 控 Crash：一种基于可控条件的汽车碰撞视频生成模型 

---
# Curate, Connect, Inquire: A System for Findable Accessible Interoperable and Reusable (FAIR) Human-Robot Centered Datasets 

**Title (ZH)**: Curate, Connect, Inquire: 一个可发现、accessible、互操作和可重用（FAIR）的人机中心化数据集系统 

**Authors**: Xingru Zhou, Sadanand Modak, Yao-Cheng Chan, Zhiyun Deng, Luis Sentis, Maria Esteva  

**Link**: [PDF](https://arxiv.org/pdf/2506.00220)  

**Abstract**: The rapid growth of AI in robotics has amplified the need for high-quality, reusable datasets, particularly in human-robot interaction (HRI) and AI-embedded robotics. While more robotics datasets are being created, the landscape of open data in the field is uneven. This is due to a lack of curation standards and consistent publication practices, which makes it difficult to discover, access, and reuse robotics data. To address these challenges, this paper presents a curation and access system with two main contributions: (1) a structured methodology to curate, publish, and integrate FAIR (Findable, Accessible, Interoperable, Reusable) human-centered robotics datasets; and (2) a ChatGPT-powered conversational interface trained with the curated datasets metadata and documentation to enable exploration, comparison robotics datasets and data retrieval using natural language. Developed based on practical experience curating datasets from robotics labs within Texas Robotics at the University of Texas at Austin, the system demonstrates the value of standardized curation and persistent publication of robotics data. The system's evaluation suggests that access and understandability of human-robotics data are significantly improved. This work directly aligns with the goals of the HCRL @ ICRA 2025 workshop and represents a step towards more human-centered access to data for embodied AI. 

**Abstract (ZH)**: 人工智能在机器人领域的迅速发展加剧了对高质量、可重用数据集的需求，特别是在人机交互（HRI）和嵌入式人工智能机器人领域。尽管正在创建更多的机器人数据集，但该领域的开放数据 landscape 仍不均衡。这主要是由于缺乏标准化的编目标准和一致的发布实践，使得发现、访问和重用机器人数据变得困难。为了应对这些挑战，本文提出了一套编目和访问系统，并包括两个主要贡献：（1）一种结构化的方法来编目、发布和整合符合 FAIR（可查找、可访问、可互操作、可重用）标准的人机中心机器人数据集；以及（2）一个以编目数据集的元数据和文档为训练内容的 ChatGPT 驱动对话接口，以通过自然语言实现机器人数据集的探索、比较和数据检索。该系统基于在德克萨斯大学奥斯汀分校 Texas Robotics 实验室中编目数据集的实践经验，证明了标准化编目和持久发布机器人数据的价值。系统的评估表明，人机交互数据的访问性和可理解性得到了显著提高。这项工作直接与 HCRL @ ICRA 2025 会议的目标相一致，并代表了实现更以人为中心的机器人数据访问的一步。 

---
# MotionPersona: Characteristics-aware Locomotion Control 

**Title (ZH)**: MotionPersona：特征感知的运动控制 

**Authors**: Mingyi Shi, Wei Liu, Jidong Mei, Wangpok Tse, Rui Chen, Xuelin Chen, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2506.00173)  

**Abstract**: We present MotionPersona, a novel real-time character controller that allows users to characterize a character by specifying attributes such as physical traits, mental states, and demographics, and projects these properties into the generated motions for animating the character. In contrast to existing deep learning-based controllers, which typically produce homogeneous animations tailored to a single, predefined character, MotionPersona accounts for the impact of various traits on human motion as observed in the real world. To achieve this, we develop a block autoregressive motion diffusion model conditioned on SMPLX parameters, textual prompts, and user-defined locomotion control signals. We also curate a comprehensive dataset featuring a wide range of locomotion types and actor traits to enable the training of this characteristic-aware controller. Unlike prior work, MotionPersona is the first method capable of generating motion that faithfully reflects user-specified characteristics (e.g., an elderly person's shuffling gait) while responding in real time to dynamic control inputs. Additionally, we introduce a few-shot characterization technique as a complementary conditioning mechanism, enabling customization via short motion clips when language prompts fall short. Through extensive experiments, we demonstrate that MotionPersona outperforms existing methods in characteristics-aware locomotion control, achieving superior motion quality and diversity. Results, code, and demo can be found at: this https URL. 

**Abstract (ZH)**: MotionPersona：一种新型实时角色控制器 

---
# Autonomous Behavior and Whole-Brain Dynamics Emerge in Embodied Zebrafish Agents with Model-based Intrinsic Motivation 

**Title (ZH)**: 具模型内在动机的 embodied 斑马鱼代理涌现自主行为和全脑动力学 

**Authors**: Reece Keller, Alyn Tornell, Felix Pei, Xaq Pitkow, Leo Kozachkov, Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00138)  

**Abstract**: Autonomy is a hallmark of animal intelligence, enabling adaptive and intelligent behavior in complex environments without relying on external reward or task structure. Existing reinforcement learning approaches to exploration in sparse reward and reward-free environments, including class of methods known as intrinsic motivation, exhibit inconsistent exploration patterns and thus fail to produce robust autonomous behaviors observed in animals. Moreover, systems neuroscience has largely overlooked the neural basis of autonomy, focusing instead on experimental paradigms where animals are motivated by external reward rather than engaging in unconstrained, naturalistic and task-independent behavior. To bridge these gaps, we introduce a novel model-based intrinsic drive explicitly designed to capture robust autonomous exploration observed in animals. Our method (3M-Progress) motivates naturalistic behavior by tracking divergence between the agent's current world model and an ethological prior. We demonstrate that artificial embodied agents trained with 3M-Progress capture the explainable variance in behavioral patterns and whole-brain neural-glial dynamics recorded from autonomously-behaving larval zebrafish, introducing the first goal-driven, population-level model of neural-glial computation. Our findings establish a computational framework connecting model-based intrinsic motivation to naturalistic behavior, providing a foundation for building artificial agents with animal-like autonomy. 

**Abstract (ZH)**: 自主性是动物智能的一个标志，使动物能够在复杂环境中无需依赖外部奖励或任务结构即可实现适应性和智能行为。现有的用于稀疏奖励和无奖励环境中的探索的强化学习方法，包括内在动机这一类方法，表现出不一致的探索模式，因此无法产生在动物中观察到的稳健自主行为。此外，系统神经科学主要忽视了自主性的神经基础，而是专注于以外部奖励为动机的实验范式，而不是关注不受约束的、自然主义的和任务独立的行为。为了弥合这些差距，我们引入了一个新型模型导向的内在驱动力，明确设计用于捕捉动物中观察到的稳健自主探索行为。我们的方法（3M-Progress）通过追踪智能体当前世界模型与生态学先验之间的差异来激励自然主义行为。我们证明，使用3M-Progress训练的虚拟生物体能够捕捉自主行为的斑马鱼在体内记录的行为模式的可解释方差以及整个大脑的神经-胶质动力学变化，引入了首个目标驱动的群体水平神经-胶质计算模型。我们的发现建立了一个将模型导向的内在动机与自然行为联系起来的计算框架，为构建具有类似动物自主性的智能体奠定了基础。 

---
# Visual Embodied Brain: Let Multimodal Large Language Models See, Think, and Control in Spaces 

**Title (ZH)**: 视觉 bodyswarm 脑: 让多模态大型语言模型在空间中观察、思考和控制 

**Authors**: Gen Luo, Ganlin Yang, Ziyang Gong, Guanzhou Chen, Haonan Duan, Erfei Cui, Ronglei Tong, Zhi Hou, Tianyi Zhang, Zhe Chen, Shenglong Ye, Lewei Lu, Jingbo Wang, Wenhai Wang, Jifeng Dai, Yu Qiao, Rongrong Ji, Xizhou Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00123)  

**Abstract**: The remarkable progress of Multimodal Large Language Models (MLLMs) has attracted increasing attention to extend them to physical entities like legged robot. This typically requires MLLMs to not only grasp multimodal understanding abilities, but also integrate visual-spatial reasoning and physical interaction capabilities. Nevertheless,existing methods struggle to unify these capabilities due to their fundamental this http URL this paper, we present the Visual Embodied Brain (VeBrain), a unified framework for perception, reasoning, and control in real world. VeBrain reformulates robotic control into common text-based MLLM tasks in the 2D visual space, thus unifying the objectives and mapping spaces of different tasks. Then, a novel robotic adapter is proposed to convert textual control signals from MLLMs to motion policies of real robots. From the data perspective, we further introduce VeBrain-600k, a high-quality instruction dataset encompassing various capabilities of VeBrain. In VeBrain-600k, we take hundreds of hours to collect, curate and annotate the data, and adopt multimodal chain-of-thought(CoT) to mix the different capabilities into a single conversation. Extensive experiments on 13 multimodal benchmarks and 5 spatial intelligence benchmarks demonstrate the superior performance of VeBrain to existing MLLMs like Qwen2.5-VL. When deployed to legged robots and robotic arms, VeBrain shows strong adaptability, flexibility, and compositional capabilities compared to existing methods. For example, compared to Qwen2.5-VL, VeBrain not only achieves substantial gains on MMVet by +5.6%, but also excels in legged robot tasks with +50% average gains. 

**Abstract (ZH)**: Multimodal Large Language Models for Physical Entities: The Visual Embodied Brain (VeBrain)Unified Framework for Perception, Reasoning, and Control 

---
# Human sensory-musculoskeletal modeling and control of whole-body movements 

**Title (ZH)**: 人体感官-肌肉骨骼建模与全身运动控制 

**Authors**: Chenhui Zuo, Guohao Lin, Chen Zhang, Shanning Zhuang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.00071)  

**Abstract**: Coordinated human movement depends on the integration of multisensory inputs, sensorimotor transformation, and motor execution, as well as sensory feedback resulting from body-environment interaction. Building dynamic models of the sensory-musculoskeletal system is essential for understanding movement control and investigating human behaviours. Here, we report a human sensory-musculoskeletal model, termed SMS-Human, that integrates precise anatomical representations of bones, joints, and muscle-tendon units with multimodal sensory inputs involving visual, vestibular, proprioceptive, and tactile components. A stage-wise hierarchical deep reinforcement learning framework was developed to address the inherent challenges of high-dimensional control in musculoskeletal systems with integrated multisensory information. Using this framework, we demonstrated the simulation of three representative movement tasks, including bipedal locomotion, vision-guided object manipulation, and human-machine interaction during bicycling. Our results showed a close resemblance between natural and simulated human motor behaviours. The simulation also revealed musculoskeletal dynamics that could not be directly measured. This work sheds deeper insights into the sensorimotor dynamics of human movements, facilitates quantitative understanding of human behaviours in interactive contexts, and informs the design of systems with embodied intelligence. 

**Abstract (ZH)**: 协调的人类运动依赖于多种感官输入的整合、感觉运动转换和运动执行，以及由身体与环境相互作用产生的感觉反馈。构建感觉-肌肉骨骼系统的动态模型对于理解运动控制和探究人类行为至关重要。在这里，我们报告了一个称为SMS-Human的人类感觉-肌肉骨骼模型，该模型整合了精确的骨骼、关节和肌腱单元的解剖学表示与涉及视觉、前庭、本体感觉和触觉的多模态感官输入。我们开发了一种分阶段的层次化深度强化学习框架，以解决包含多感官信息的肌肉骨骼系统固有的高维控制难题。利用这一框架，我们展示了三项代表性运动任务的模拟，包括双足行走、视觉引导的物体操作以及骑自行车过程中的人机交互。我们的结果表明，自然与模拟的人类运动行为十分相似。模拟还揭示了无法直接测量的肌肉骨骼动力学。这项工作深入探讨了人类运动的感觉运动动力学，有助于在交互环境中定量理解人类行为，并为具有体态智能的系统设计提供指导。 

---
