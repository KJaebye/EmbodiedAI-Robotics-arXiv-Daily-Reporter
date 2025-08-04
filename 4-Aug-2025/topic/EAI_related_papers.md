# Video Generators are Robot Policies 

**Title (ZH)**: 视频生成器是机器人策略 

**Authors**: Junbang Liang, Pavel Tokmakov, Ruoshi Liu, Sruthi Sudhakar, Paarth Shah, Rares Ambrus, Carl Vondrick  

**Link**: [PDF](https://arxiv.org/pdf/2508.00795)  

**Abstract**: Despite tremendous progress in dexterous manipulation, current visuomotor policies remain fundamentally limited by two challenges: they struggle to generalize under perceptual or behavioral distribution shifts, and their performance is constrained by the size of human demonstration data. In this paper, we use video generation as a proxy for robot policy learning to address both limitations simultaneously. We propose Video Policy, a modular framework that combines video and action generation that can be trained end-to-end. Our results demonstrate that learning to generate videos of robot behavior allows for the extraction of policies with minimal demonstration data, significantly improving robustness and sample efficiency. Our method shows strong generalization to unseen objects, backgrounds, and tasks, both in simulation and the real world. We further highlight that task success is closely tied to the generated video, with action-free video data providing critical benefits for generalizing to novel tasks. By leveraging large-scale video generative models, we achieve superior performance compared to traditional behavior cloning, paving the way for more scalable and data-efficient robot policy learning. 

**Abstract (ZH)**: 尽管在灵巧操作方面取得了巨大进展，当前的视觉-运动策略仍受两大挑战的限制：它们在知觉或行为分布转移时难以泛化，且性能受限于人类演示数据的规模。本文中，我们使用视频生成作为机器人策略学习的代理，同时解决这两种限制。我们提出Video Policy，这是一种结合视频和动作生成的模块化框架，可以进行端到端训练。实验结果表明，学习生成机器人行为的视频能够利用少量的演示数据提取策略，显著提高鲁棒性和样本效率。我们的方法在仿真和现实世界中对未见物体、背景和任务都显示出强大的泛化能力。进一步研究表明，任务成功与生成的视频密切相关，无动作的视频数据提供了对新任务泛化的关键益处。通过利用大规模的视频生成模型，我们取得了优于传统行为克隆的表现，为更具扩展性和数据效率的机器人策略学习开辟了新途径。 

---
# On-Device Diffusion Transformer Policy for Efficient Robot Manipulation 

**Title (ZH)**: 设备端扩散变换器策略高效机器人操作 

**Authors**: Yiming Wu, Huan Wang, Zhenghao Chen, Jianxin Pang, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00697)  

**Abstract**: Diffusion Policies have significantly advanced robotic manipulation tasks via imitation learning, but their application on resource-constrained mobile platforms remains challenging due to computational inefficiency and extensive memory footprint. In this paper, we propose LightDP, a novel framework specifically designed to accelerate Diffusion Policies for real-time deployment on mobile devices. LightDP addresses the computational bottleneck through two core strategies: network compression of the denoising modules and reduction of the required sampling steps. We first conduct an extensive computational analysis on existing Diffusion Policy architectures, identifying the denoising network as the primary contributor to latency. To overcome performance degradation typically associated with conventional pruning methods, we introduce a unified pruning and retraining pipeline, optimizing the model's post-pruning recoverability explicitly. Furthermore, we combine pruning techniques with consistency distillation to effectively reduce sampling steps while maintaining action prediction accuracy. Experimental evaluations on the standard datasets, \ie, PushT, Robomimic, CALVIN, and LIBERO, demonstrate that LightDP achieves real-time action prediction on mobile devices with competitive performance, marking an important step toward practical deployment of diffusion-based policies in resource-limited environments. Extensive real-world experiments also show the proposed LightDP can achieve performance comparable to state-of-the-art Diffusion Policies. 

**Abstract (ZH)**: 轻量级扩散策略：一种针对移动平台实时部署的新型加速框架 

---
# A control scheme for collaborative object transportation between a human and a quadruped robot using the MIGHTY suction cup 

**Title (ZH)**: 基于MIGHTY吸附杯的人与四足机器人协作物体运输的控制方案 

**Authors**: Konstantinos Plotas, Emmanouil Papadakis, Drosakis Drosakis, Panos Trahanias, Dimitrios Papageorgiou  

**Link**: [PDF](https://arxiv.org/pdf/2508.00584)  

**Abstract**: In this work, a control scheme for human-robot collaborative object transportation is proposed, considering a quadruped robot equipped with the MIGHTY suction cup that serves both as a gripper for holding the object and a force/torque sensor. The proposed control scheme is based on the notion of admittance control, and incorporates a variable damping term aiming towards increasing the controllability of the human and, at the same time, decreasing her/his effort. Furthermore, to ensure that the object is not detached from the suction cup during the collaboration, an additional control signal is proposed, which is based on a barrier artificial potential. The proposed control scheme is proven to be passive and its performance is demonstrated through experimental evaluations conducted using the Unitree Go1 robot equipped with the MIGHTY suction cup. 

**Abstract (ZH)**: 基于MIGHTY吸附杯的四足机器人协作物体运输控制方案 

---
# SubCDM: Collective Decision-Making with a Swarm Subset 

**Title (ZH)**: 集体决策中的子群集决策Making: SubCDM with a Swarm Subset 

**Authors**: Samratul Fuady, Danesh Tarapore, Mohammad D. Soorati  

**Link**: [PDF](https://arxiv.org/pdf/2508.00467)  

**Abstract**: Collective decision-making is a key function of autonomous robot swarms, enabling them to reach a consensus on actions based on environmental features. Existing strategies require the participation of all robots in the decision-making process, which is resource-intensive and prevents the swarm from allocating the robots to any other tasks. We propose Subset-Based Collective Decision-Making (SubCDM), which enables decisions using only a swarm subset. The construction of the subset is dynamic and decentralized, relying solely on local information. Our method allows the swarm to adaptively determine the size of the subset for accurate decision-making, depending on the difficulty of reaching a consensus. Simulation results using one hundred robots show that our approach achieves accuracy comparable to using the entire swarm while reducing the number of robots required to perform collective decision-making, making it a resource-efficient solution for collective decision-making in swarm robotics. 

**Abstract (ZH)**: 基于子集的集体决策-making in Autonomous Robot Swarms via Subset-Based Collective Decision-Making (SubCDM) 

---
# A Whole-Body Motion Imitation Framework from Human Data for Full-Size Humanoid Robot 

**Title (ZH)**: 基于人类数据的全身运动模仿框架用于全尺寸人形机器人 

**Authors**: Zhenghan Chen, Haodong Zhang, Dongqi Wang, Jiyu Yu, Haocheng Xu, Yue Wang, Rong Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00362)  

**Abstract**: Motion imitation is a pivotal and effective approach for humanoid robots to achieve a more diverse range of complex and expressive movements, making their performances more human-like. However, the significant differences in kinematics and dynamics between humanoid robots and humans present a major challenge in accurately imitating motion while maintaining balance. In this paper, we propose a novel whole-body motion imitation framework for a full-size humanoid robot. The proposed method employs contact-aware whole-body motion retargeting to mimic human motion and provide initial values for reference trajectories, and the non-linear centroidal model predictive controller ensures the motion accuracy while maintaining balance and overcoming external disturbances in real time. The assistance of the whole-body controller allows for more precise torque control. Experiments have been conducted to imitate a variety of human motions both in simulation and in a real-world humanoid robot. These experiments demonstrate the capability of performing with accuracy and adaptability, which validates the effectiveness of our approach. 

**Abstract (ZH)**: 基于全身体态的仿人运动模仿框架 

---
# TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots 

**Title (ZH)**: TOP: 用于人类机器人稳定准确站立操作的时间优化策略 

**Authors**: Zhenghan Chen, Haocheng Xu, Haodong Zhang, Liang Zhang, He Li, Dongqi Wang, Jiyu Yu, Yifei Yang, Zhongxiang Zhou, Rong Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00355)  

**Abstract**: Humanoid robots have the potential capability to perform a diverse range of manipulation tasks, but this is based on a robust and precise standing controller. Existing methods are either ill-suited to precisely control high-dimensional upper-body joints, or difficult to ensure both robustness and accuracy, especially when upper-body motions are fast. This paper proposes a novel time optimization policy (TOP), to train a standing manipulation control model that ensures balance, precision, and time efficiency simultaneously, with the idea of adjusting the time trajectory of upper-body motions but not only strengthening the disturbance resistance of the lower-body. Our approach consists of three parts. Firstly, we utilize motion prior to represent upper-body motions to enhance the coordination ability between the upper and lower-body by training a variational autoencoder (VAE). Then we decouple the whole-body control into an upper-body PD controller for precision and a lower-body RL controller to enhance robust stability. Finally, we train TOP method in conjunction with the decoupled controller and VAE to reduce the balance burden resulting from fast upper-body motions that would destabilize the robot and exceed the capabilities of the lower-body RL policy. The effectiveness of the proposed approach is evaluated via both simulation and real world experiments, which demonstrate the superiority on standing manipulation tasks stably and accurately. The project page can be found at this https URL. 

**Abstract (ZH)**: 类人机器人具有执行多种操作任务的潜力，但这基于一个稳健且精确的站立控制器。现有方法要么不适合精确控制高维度上半身关节，要么难以同时保证稳健性和精确性，尤其是在上半身运动快速时。本文提出了一种新颖的时间优化策略（TOP），旨在训练一个同时确保平衡、精确性和时间效率的站立操作控制模型，通过调整上半身运动的时间轨迹，而不只是增强下半身的抗干扰能力。我们的方法包含三个部分：首先，利用运动先验来表示上半身运动，通过训练变分自编码器（VAE）增强上下半身的协调能力；然后，将全身控制分解为一个用于精确性的上半身PD控制器和一个用于增强鲁棒稳定的下半身RL控制器；最后，结合解耦控制器和VAE共同训练TOP方法，以降低因快速上半身运动导致的平衡负担，防止机器人失稳并超出下半身RL策略的能力范围。提出的这种方法通过仿真和现实世界的实验进行了评估，表明在稳定而准确的站立操作任务上具有优势。项目页面可访问 [此链接]。 

---
# UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents 

**Title (ZH)**: UAV-ON: 一种基于无人机的开放世界目标导航基准数据集 

**Authors**: Jianqiang Xiao, Yuexuan Sun, Yixin Shao, Boxi Gan, Rongqiang Liu, Yanjing Wu, Weili Gua, Xiang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00288)  

**Abstract**: Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments. 

**Abstract (ZH)**: 基于物体目标的无人机空中导航：一个开放世界环境中的基准（UAV-ON） 

---
# CHILD (Controller for Humanoid Imitation and Live Demonstration): a Whole-Body Humanoid Teleoperation System 

**Title (ZH)**: CHILD (用于模仿和现场演示的人形控制器): 一个全身人形远程操作系统 

**Authors**: Noboru Myers, Obin Kwon, Sankalp Yamsani, Joohyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.00162)  

**Abstract**: Recent advances in teleoperation have demonstrated robots performing complex manipulation tasks. However, existing works rarely support whole-body joint-level teleoperation for humanoid robots, limiting the diversity of tasks that can be accomplished. This work presents Controller for Humanoid Imitation and Live Demonstration (CHILD), a compact reconfigurable teleoperation system that enables joint level control over humanoid robots. CHILD fits within a standard baby carrier, allowing the operator control over all four limbs, and supports both direct joint mapping for full-body control and loco-manipulation. Adaptive force feedback is incorporated to enhance operator experience and prevent unsafe joint movements. We validate the capabilities of this system by conducting loco-manipulation and full-body control examples on a humanoid robot and multiple dual-arm systems. Lastly, we open-source the design of the hardware promoting accessibility and reproducibility. Additional details and open-source information are available at our project website: this https URL. 

**Abstract (ZH)**: Recent Advances in Teleoperation Have Demonstrated Robots Performing Complex Manipulation Tasks. However, Existing Works Rarely Support Whole-Body Joint-Level Teleoperation for Humanoid Robots, Limiting the Diversity of Tasks That Can Be Accomplished. This Work Presents Controller for Humanoid Imitation and Live Demonstration (CHILD), a Compact Reconfigurable Teleoperation System That Enables Joint-Level Control Over Humanoid Robots. 

---
# XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation 

**Title (ZH)**: XRoboToolkit：一种跨平台机器人远程操作框架 

**Authors**: Zhigen Zhao, Liuchuan Yu, Ke Jing, Ning Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00097)  

**Abstract**: The rapid advancement of Vision-Language-Action models has created an urgent need for large-scale, high-quality robot demonstration datasets. Although teleoperation is the predominant method for data collection, current approaches suffer from limited scalability, complex setup procedures, and suboptimal data quality. This paper presents XRoboToolkit, a cross-platform framework for extended reality based robot teleoperation built on the OpenXR standard. The system features low-latency stereoscopic visual feedback, optimization-based inverse kinematics, and support for diverse tracking modalities including head, controller, hand, and auxiliary motion trackers. XRoboToolkit's modular architecture enables seamless integration across robotic platforms and simulation environments, spanning precision manipulators, mobile robots, and dexterous hands. We demonstrate the framework's effectiveness through precision manipulation tasks and validate data quality by training VLA models that exhibit robust autonomous performance. 

**Abstract (ZH)**: 基于扩展现实的跨平台机器人远程操作工具包：XRoboToolkit 

---
# The Monado SLAM Dataset for Egocentric Visual-Inertial Tracking 

**Title (ZH)**: Monado SLAM数据集：第一人称视觉-惯性追踪 

**Authors**: Mateo de Mayo, Daniel Cremers, Taihú Pire  

**Link**: [PDF](https://arxiv.org/pdf/2508.00088)  

**Abstract**: Humanoid robots and mixed reality headsets benefit from the use of head-mounted sensors for tracking. While advancements in visual-inertial odometry (VIO) and simultaneous localization and mapping (SLAM) have produced new and high-quality state-of-the-art tracking systems, we show that these are still unable to gracefully handle many of the challenging settings presented in the head-mounted use cases. Common scenarios like high-intensity motions, dynamic occlusions, long tracking sessions, low-textured areas, adverse lighting conditions, saturation of sensors, to name a few, continue to be covered poorly by existing datasets in the literature. In this way, systems may inadvertently overlook these essential real-world issues. To address this, we present the Monado SLAM dataset, a set of real sequences taken from multiple virtual reality headsets. We release the dataset under a permissive CC BY 4.0 license, to drive advancements in VIO/SLAM research and development. 

**Abstract (ZH)**: 头部位姿传感器在人形机器人和混合现实头显中的应用：Monado SLAM数据集促进视觉惯性里程计和同时定位与建图的研究 

---
# Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training 

**Title (ZH)**: 认知内核增强：一种深度研究代理及代理基础模型训练的框架 

**Authors**: Tianqing Fang, Zhisong Zhang, Xiaoyang Wang, Rui Wang, Can Qin, Yuxuan Wan, Jun-Yu Ma, Ce Zhang, Jiaqi Chen, Xiyun Li, Hongming Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00414)  

**Abstract**: General AI Agents are increasingly recognized as foundational frameworks for the next generation of artificial intelligence, enabling complex reasoning, web interaction, coding, and autonomous research capabilities. However, current agent systems are either closed-source or heavily reliant on a variety of paid APIs and proprietary tools, limiting accessibility and reproducibility for the research community. In this work, we present \textbf{Cognitive Kernel-Pro}, a fully open-source and (to the maximum extent) free multi-module agent framework designed to democratize the development and evaluation of advanced AI agents. Within Cognitive Kernel-Pro, we systematically investigate the curation of high-quality training data for Agent Foundation Models, focusing on the construction of queries, trajectories, and verifiable answers across four key domains: web, file, code, and general reasoning. Furthermore, we explore novel strategies for agent test-time reflection and voting to enhance agent robustness and performance. We evaluate Cognitive Kernel-Pro on GAIA, achieving state-of-the-art results among open-source and free agents. Notably, our 8B-parameter open-source model surpasses previous leading systems such as WebDancer and WebSailor, establishing a new performance standard for accessible, high-capability AI agents. Code is available at this https URL 

**Abstract (ZH)**: 通用人工智能代理被日益认为是下一代人工智能的基础框架，能够实现复杂的推理、网络交互、编程和自主研究能力。然而，当前的代理系统要么是封闭源代码的，要么严重依赖多种付费API和专有工具，这限制了研究社区的可访问性和可重复性。在此项工作中，我们提出了Cognitive Kernel-Pro，一个完全开源且最大程度上免费的多模块代理框架，旨在普及高级AI代理的开发与评估。在Cognitive Kernel-Pro中，我们系统地研究了代理基础模型高质量训练数据的收集，重点关注在四个关键领域（网络、文件、代码和一般推理）构建查询、轨迹和可验证答案。此外，我们探讨了新的代理测试时反思和投票策略，以提高代理的鲁棒性和性能。我们对Cognitive Kernel-Pro进行了评估，并在GAIA上取得了开源和免费代理的最优结果。值得注意的是，我们8B参数的开源模型超越了之前的WebDancer和WebSailor等领先系统，确立了可访问且高性能AI代理的新标准。代码可用于此链接。 

---
# Theory of Mind Using Active Inference: A Framework for Multi-Agent Cooperation 

**Title (ZH)**: 基于主动推断的理论理解：多代理合作的框架 

**Authors**: Riddhi J. Pitliya, Ozan Catal, Toon Van de Maele, Corrado Pezzato, Tim Verbelen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00401)  

**Abstract**: We present a novel approach to multi-agent cooperation by implementing theory of mind (ToM) within active inference. ToM - the ability to understand that others can have differing knowledge and goals - enables agents to reason about others' beliefs while planning their own actions. Unlike previous active inference approaches to multi-agent cooperation, our method neither relies on task-specific shared generative models nor requires explicit communication, while being generalisable. In our framework, the ToM-equipped agent maintains distinct representations of its own and others' beliefs and goals. We extend the sophisticated inference tree-based planning algorithm to systematically explore joint policy spaces through recursive reasoning. Our approach is evaluated through collision avoidance and foraging task simulations. Results demonstrate that ToM-equipped agents cooperate better compared to non-ToM counterparts by being able to avoid collisions and reduce redundant efforts. Crucially, ToM agents accomplish this by inferring others' beliefs solely from observable behaviour. This work advances practical applications in artificial intelligence while providing computational insights into ToM. 

**Abstract (ZH)**: 我们在主动推断框架中实现理论心智以实现多智能体合作的新方法 

---
# Hyperproperty-Constrained Secure Reinforcement Learning 

**Title (ZH)**: Hyper属性约束的安全强化学习 

**Authors**: Ernest Bonnah, Luan Viet Nguyen, Khaza Anuarul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2508.00106)  

**Abstract**: Hyperproperties for Time Window Temporal Logic (HyperTWTL) is a domain-specific formal specification language known for its effectiveness in compactly representing security, opacity, and concurrency properties for robotics applications. This paper focuses on HyperTWTL-constrained secure reinforcement learning (SecRL). Although temporal logic-constrained safe reinforcement learning (SRL) is an evolving research problem with several existing literature, there is a significant research gap in exploring security-aware reinforcement learning (RL) using hyperproperties. Given the dynamics of an agent as a Markov Decision Process (MDP) and opacity/security constraints formalized as HyperTWTL, we propose an approach for learning security-aware optimal policies using dynamic Boltzmann softmax RL while satisfying the HyperTWTL constraints. The effectiveness and scalability of our proposed approach are demonstrated using a pick-up and delivery robotic mission case study. We also compare our results with two other baseline RL algorithms, showing that our proposed method outperforms them. 

**Abstract (ZH)**: 时间窗口时态逻辑下的超性质约束安全强化学习（HyperTWTL-constrained Secure Reinforcement Learning） 

---
# Composable OS Kernel Architectures for Autonomous Intelligence 

**Title (ZH)**: 可组合自主智能操作系统内核架构 

**Authors**: Rajpreet Singh, Vidhi Kothari  

**Link**: [PDF](https://arxiv.org/pdf/2508.00604)  

**Abstract**: As intelligent systems permeate edge devices, cloud infrastructure, and embedded real-time environments, this research proposes a new OS kernel architecture for intelligent systems, transforming kernels from static resource managers to adaptive, AI-integrated platforms. Key contributions include: (1) treating Loadable Kernel Modules (LKMs) as AI-oriented computation units for fast sensory and cognitive processing in kernel space; (2) expanding the Linux kernel into an AI-native environment with built-in deep learning inference, floating-point acceleration, and real-time adaptive scheduling for efficient ML workloads; and (3) introducing a Neurosymbolic kernel design leveraging Category Theory and Homotopy Type Theory to unify symbolic reasoning and differentiable logic within OS internals. Together, these approaches enable operating systems to proactively anticipate and adapt to the cognitive needs of autonomous intelligent applications. 

**Abstract (ZH)**: 随着智能系统渗透到边缘设备、云计算基础设施和嵌入式实时环境中，本研究提出了一种新的OS内核架构，将内核从静态资源管理器转变为适应性强、集成人工智能的平台。主要贡献包括：（1）将可加载内核模块（LKMs）视为面向人工智能的计算单元，以实现快速内核空间中的感知和认知处理；（2）将Linux内核扩展为内置深度学习推理、浮点加速和实时自适应调度的原生AI环境，以高效处理机器学习工作负载；（3）引入基于范畴论和同伦类型论的神经符号性内核设计，以内核内部统一符号推理和可微逻辑。这些方法共同使操作系统能够主动预见并适应自主智能应用的认知需求。 

---
# DeformTune: A Deformable XAI Music Prototype for Non-Musicians 

**Title (ZH)**: DeformTune: 一种面向非音乐家的可形变解释音乐原型 

**Authors**: Ziqing Xu, Nick Bryan-Kinns  

**Link**: [PDF](https://arxiv.org/pdf/2508.00160)  

**Abstract**: Many existing AI music generation tools rely on text prompts, complex interfaces, or instrument-like controls, which may require musical or technical knowledge that non-musicians do not possess. This paper introduces DeformTune, a prototype system that combines a tactile deformable interface with the MeasureVAE model to explore more intuitive, embodied, and explainable AI interaction. We conducted a preliminary study with 11 adult participants without formal musical training to investigate their experience with AI-assisted music creation. Thematic analysis of their feedback revealed recurring challenge--including unclear control mappings, limited expressive range, and the need for guidance throughout use. We discuss several design opportunities for enhancing explainability of AI, including multimodal feedback and progressive interaction support. These findings contribute early insights toward making AI music systems more explainable and empowering for novice users. 

**Abstract (ZH)**: DeformTune：一种结合可变形界面与MeasureVAE模型的直观、具身化和可解释的AI音乐创作 prototype 系统 

---
# A Mixed User-Centered Approach to Enable Augmented Intelligence in Intelligent Tutoring Systems: The Case of MathAIde app 

**Title (ZH)**: 一种混合用户中心的方法以在智能 Tutoring 系统中实现增强人工智能：MathAIde 应用案例 

**Authors**: Guilherme Guerino, Luiz Rodrigues, Luana Bianchiniand Mariana Alves, Marcelo Marinho, Thomaz Veloso, Valmir Macario, Diego Dermeval, Thales Vieira, Ig Bittencourt, Seiji Isotani  

**Link**: [PDF](https://arxiv.org/pdf/2508.00103)  

**Abstract**: Integrating Artificial Intelligence in Education (AIED) aims to enhance learning experiences through technologies like Intelligent Tutoring Systems (ITS), offering personalized learning, increased engagement, and improved retention rates. However, AIED faces three main challenges: the critical role of teachers in the design process, the limitations and reliability of AI tools, and the accessibility of technological resources. Augmented Intelligence (AuI) addresses these challenges by enhancing human capabilities rather than replacing them, allowing systems to suggest solutions. In contrast, humans provide final assessments, thus improving AI over time. In this sense, this study focuses on designing, developing, and evaluating MathAIde, an ITS that corrects mathematics exercises using computer vision and AI and provides feedback based on photos of student work. The methodology included brainstorming sessions with potential users, high-fidelity prototyping, A/B testing, and a case study involving real-world classroom environments for teachers and students. Our research identified several design possibilities for implementing AuI in ITSs, emphasizing a balance between user needs and technological feasibility. Prioritization and validation through prototyping and testing highlighted the importance of efficiency metrics, ultimately leading to a solution that offers pre-defined remediation alternatives for teachers. Real-world deployment demonstrated the usefulness of the proposed solution. Our research contributes to the literature by providing a usable, teacher-centered design approach that involves teachers in all design phases. As a practical implication, we highlight that the user-centered design approach increases the usefulness and adoption potential of AIED systems, especially in resource-limited environments. 

**Abstract (ZH)**: 将人工智能整合于教育（AIED）旨在通过智能辅导系统（ITS）等技术提升学习体验，实现个性化学习、增强参与度并提高留存率。然而，AIED面临三大挑战：教师在设计过程中的关键作用、AI工具的局限性和可靠性，以及技术资源的可访问性。增强智能（Augmented Intelligence, AuI）通过增强人类能力而非替代人类来应对这些挑战，使系统能够提出建议解决方案。相比之下，人类提供最终评估，从而随着时间改善AI。在此意义上，本研究重点在于设计、开发和评估一种名为MathAIde的ITS，该系统使用计算机视觉和AI纠正数学练习，并根据学生作业的照片提供反馈。研究方法包括与潜在用户进行头脑风暴会、高保真原型制作、A/B测试，以及涉及真实教室环境的案例研究。我们的研究指出了在ITS中实施AuI的设计可能性，强调了用户需求和技术可行性之间的平衡。通过原型制作和测试的优先级与验证突显了效率指标的重要性，最终导致一种为教师提供预定义补救选项的解决方案。实际部署表明所提案解决方案的实用性。我们的研究为文献贡献了一种由教师为中心的设计方法，该方法在整个设计阶段涉及教师。从实践角度来看，我们强调用户为中心的设计方法提高了AIED系统的有用性和采用潜力，特别是在资源有限的环境中。 

---
# Agent Network Protocol Technical White Paper 

**Title (ZH)**: 智能体网络协议技术白皮书 

**Authors**: Gaowei Chang, Eidan Lin, Chengxuan Yuan, Rizhao Cai, Binbin Chen, Xuan Xie, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00007)  

**Abstract**: With the development of large models and autonomous decision-making AI, agents are rapidly becoming the new entities of the internet, following mobile apps. However, existing internet infrastructure is primarily designed for human interaction, creating data silos, unfriendly interfaces, and high collaboration costs among agents, making it difficult to support the needs for large-scale agent interconnection and collaboration. The internet is undergoing a profound transformation, showing four core trends: agents replacing traditional software, universal agent interconnection, native protocol-based connections, and autonomous agent organization and collaboration. To align with these trends, Agent Network Protocol (ANP) proposes a new generation of communication protocols for the Agentic Web. ANP adheres to AI-native design, maintains compatibility with existing internet protocols, adopts a modular composable architecture, follows minimalist yet extensible principles, and enables rapid deployment based on existing infrastructure. Through a three-layer protocol system--identity and encrypted communication layer, meta-protocol negotiation layer, and application protocol layer--ANP. systematically solves the problems of agent identity authentication, dynamic negotiation, and capability discovery interoperability. 

**Abstract (ZH)**: 伴随大型模型和自主决策AI的发展，代理正在成为互联网上的新实体，继移动应用之后。然而，现有的互联网基础设施主要为人类交互设计，导致了代理间的数据孤岛、不友好的界面以及高协作成本，难以支持大规模代理互联和协作的需求。互联网正经历深刻的转型，展现出了四大核心趋势：代理取代传统软件、普遍的代理互联、基于原生协议的连接以及自主代理组织和协作。为了顺应这些趋势，代理网络协议（ANP）提出了一代适用于Agentic Web的通信协议。ANP坚持AI原生设计，保持与现有互联网协议的兼容性，采用模块化可组合架构，遵循简洁且可扩展的原则，并能基于现有基础设施实现快速部署。通过三层协议系统——身份与加密通信层、元协议协商层和应用协议层，ANP系统性地解决了代理身份认证、动态协商和能力发现互操作性的问题。 

---
