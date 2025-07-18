# Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop 

**Title (ZH)**: Eye, Robot: 学习观察以行动——基于BC-RL知觉-动作环路 

**Authors**: Justin Kerr, Kush Hari, Ethan Weber, Chung Min Kim, Brent Yi, Tyler Bonnen, Ken Goldberg, Angjoo Kanazawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.10968)  

**Abstract**: Humans do not passively observe the visual world -- we actively look in order to act. Motivated by this principle, we introduce EyeRobot, a robotic system with gaze behavior that emerges from the need to complete real-world tasks. We develop a mechanical eyeball that can freely rotate to observe its surroundings and train a gaze policy to control it using reinforcement learning. We accomplish this by first collecting teleoperated demonstrations paired with a 360 camera. This data is imported into a simulation environment that supports rendering arbitrary eyeball viewpoints, allowing episode rollouts of eye gaze on top of robot demonstrations. We then introduce a BC-RL loop to train the hand and eye jointly: the hand (BC) agent is trained from rendered eye observations, and the eye (RL) agent is rewarded when the hand produces correct action predictions. In this way, hand-eye coordination emerges as the eye looks towards regions which allow the hand to complete the task. EyeRobot implements a foveal-inspired policy architecture allowing high resolution with a small compute budget, which we find also leads to the emergence of more stable fixation as well as improved ability to track objects and ignore distractors. We evaluate EyeRobot on five panoramic workspace manipulation tasks requiring manipulation in an arc surrounding the robot arm. Our experiments suggest EyeRobot exhibits hand-eye coordination behaviors which effectively facilitate manipulation over large workspaces with a single camera. See project site for videos: this https URL 

**Abstract (ZH)**: 基于目光行为的人工智能机器人EyeRobot：从真实世界任务中 emerges from the need to complete real-world tasks 

---
# GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation 

**Title (ZH)**: GENMANIP: 由大规模语言模型驱动的通用指令遵循操作模拟 

**Authors**: Ning Gao, Yilun Chen, Shuai Yang, Xinyi Chen, Yang Tian, Hao Li, Haifeng Huang, Hanqing Wang, Tai Wang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10966)  

**Abstract**: Robotic manipulation in real-world settings remains challenging, especially regarding robust generalization. Existing simulation platforms lack sufficient support for exploring how policies adapt to varied instructions and scenarios. Thus, they lag behind the growing interest in instruction-following foundation models like LLMs, whose adaptability is crucial yet remains underexplored in fair comparisons. To bridge this gap, we introduce GenManip, a realistic tabletop simulation platform tailored for policy generalization studies. It features an automatic pipeline via LLM-driven task-oriented scene graph to synthesize large-scale, diverse tasks using 10K annotated 3D object assets. To systematically assess generalization, we present GenManip-Bench, a benchmark of 200 scenarios refined via human-in-the-loop corrections. We evaluate two policy types: (1) modular manipulation systems integrating foundation models for perception, reasoning, and planning, and (2) end-to-end policies trained through scalable data collection. Results show that while data scaling benefits end-to-end methods, modular systems enhanced with foundation models generalize more effectively across diverse scenarios. We anticipate this platform to facilitate critical insights for advancing policy generalization in realistic conditions. Project Page: this https URL. 

**Abstract (ZH)**: 现实场景中的机器人操作依然具有挑战性，特别是在鲁棒泛化方面。现有的模拟平台缺乏足够的支持来探索策略如何适应多变的指令和场景。因此，它们在与越来越受到关注的指令遵循基础模型（如LLMs）的适应性进行公平比较时落后了。为弥合这一差距，我们引入了GenManip，这是一个针对策略泛化研究定制的现实桌面模拟平台。它通过LLM驱动的任务导向场景图提供了一种自动化的流水线，使用10K注释的3D对象资产来合成大规模、多样化的任务。为了系统地评估泛化能力，我们提出了GenManip-Bench，这是一个包含200个场景的基准，这些场景通过人类在环路校正进行精炼。我们评估了两种策略类型：(1) 结合基础模型的模块化操作系统，用于感知、推理和规划，以及(2) 通过可扩展的数据收集训练的端到端策略。结果表明，尽管数据量的增加有利于端到端方法，但结合基础模型的模块化系统在多样化的场景中泛化更有效。我们期望该平台能促进在现实条件下的策略泛化研究。项目页面：这个 <https://>网址。 

---
# Modeling Trust Dynamics in Robot-Assisted Delivery: Impact of Trust Repair Strategies 

**Title (ZH)**: 基于机器人辅助配送的信任动态建模：信任修复策略的影响 

**Authors**: Dong Hae Mangalindan, Karthik Kandikonda, Ericka Rovira, Vaibhav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2506.10884)  

**Abstract**: With increasing efficiency and reliability, autonomous systems are becoming valuable assistants to humans in various tasks. In the context of robot-assisted delivery, we investigate how robot performance and trust repair strategies impact human trust. In this task, while handling a secondary task, humans can choose to either send the robot to deliver autonomously or manually control it. The trust repair strategies examined include short and long explanations, apology and promise, and denial.
Using data from human participants, we model human behavior using an Input-Output Hidden Markov Model (IOHMM) to capture the dynamics of trust and human action probabilities. Our findings indicate that humans are more likely to deploy the robot autonomously when their trust is high. Furthermore, state transition estimates show that long explanations are the most effective at repairing trust following a failure, while denial is most effective at preventing trust loss.
We also demonstrate that the trust estimates generated by our model are isomorphic to self-reported trust values, making them interpretable. This model lays the groundwork for developing optimal policies that facilitate real-time adjustment of human trust in autonomous systems. 

**Abstract (ZH)**: 随着自主系统的效率和可靠性不断提高，它们在各种任务中 becoming valuable assistants to humans。在机器人辅助配送的背景下，我们探讨了机器人性能和信任修复策略对人类信任的影响。在这一任务中，当处理次要任务时，人类可以选择让机器人自主配送或手动控制。我们研究的信任修复策略包括简短和详尽的解释、道歉和承诺以及否认。

通过人类参与者的数据，我们使用输入-输出隐马尔可夫模型（IOHMM）来建模人类行为，以捕捉信任动态和人类行动的概率。研究发现，当人类的信任度较高时，他们更倾向于让机器人自主配送。此外，状态转换估计表明，在故障后，详尽的解释是最有效的信任修复策略，而否认是最有效的防止信任损失的策略。

我们还展示了由该模型生成的信任估计值与自我报告的信任值等价，使其具有可解释性。该模型为开发促进自主系统中实时调整人类信任的最优政策奠定了基础。 

---
# Data-Driven Prediction of Dynamic Interactions Between Robot Appendage and Granular Material 

**Title (ZH)**: 基于数据驱动的机器人肢体与颗粒材料动态相互作用预测 

**Authors**: Guanjin Wang, Xiangxue Zhao, Shapour Azarm, Balakumar Balachandran  

**Link**: [PDF](https://arxiv.org/pdf/2506.10875)  

**Abstract**: An alternative data-driven modeling approach has been proposed and employed to gain fundamental insights into robot motion interaction with granular terrain at certain length scales. The approach is based on an integration of dimension reduction (Sequentially Truncated Higher-Order Singular Value Decomposition), surrogate modeling (Gaussian Process), and data assimilation techniques (Reduced Order Particle Filter). This approach can be used online and is based on offline data, obtained from the offline collection of high-fidelity simulation data and a set of sparse experimental data. The results have shown that orders of magnitude reduction in computational time can be obtained from the proposed data-driven modeling approach compared with physics-based high-fidelity simulations. With only simulation data as input, the data-driven prediction technique can generate predictions that have comparable accuracy as simulations. With both simulation data and sparse physical experimental measurement as input, the data-driven approach with its embedded data assimilation techniques has the potential in outperforming only high-fidelity simulations for the long-horizon predictions. In addition, it is demonstrated that the data-driven modeling approach can also reproduce the scaling relationship recovered by physics-based simulations for maximum resistive forces, which may indicate its general predictability beyond a case-by-case basis. The results are expected to help robot navigation and exploration in unknown and complex terrains during both online and offline phases. 

**Abstract (ZH)**: 一种数据驱动建模方法被提出并应用于在特定长度尺度下获得机器人运动与颗粒地形相互作用的基本见解。该方法基于降维（顺序截断高阶singular值分解）、代理建模（高斯过程）和数据同化技术（降阶粒子滤波）的集成。该方法可以在线使用，并基于离线收集的高保真仿真数据和一组稀疏的实验数据。结果表明，与基于物理的高保真仿真相比，所提出的数据驱动建模方法可以大幅减少计算时间。仅使用仿真数据作为输入，数据驱动预测技术可以生成与仿真具有可比准确性的预测。同时，利用仿真数据和稀疏物理实验测量作为输入，数据驱动方法结合嵌入的数据同化技术有望在长时预测中优于仅基于高保真仿真的方法。此外，还展示了数据驱动建模方法可以重现基于物理仿真的最大阻力的标度关系，这可能表明其在个例之外的一般预测能力。预计这些结果将有助于机器人在未知和复杂地形中的导航与勘探，在离线和在线阶段均适用。 

---
# RationalVLA: A Rational Vision-Language-Action Model with Dual System 

**Title (ZH)**: 理性的多模态模型：带有双系统的作用视知觉模型 

**Authors**: Wenxuan Song, Jiayi Chen, Wenxue Li, Xu He, Han Zhao, Pengxiang Ding Shiyan Su, Feilong Tang, Xuelian Cheng, Donglin Wang, Zongyuan Ge, Xinhu Zheng, Zhe Liu, Hesheng Wang, Yunhui Liu, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10826)  

**Abstract**: A fundamental requirement for real-world robotic deployment is the ability to understand and respond to natural language instructions. Existing language-conditioned manipulation tasks typically assume that instructions are perfectly aligned with the environment. This assumption limits robustness and generalization in realistic scenarios where instructions may be ambiguous, irrelevant, or infeasible. To address this problem, we introduce RAtional MAnipulation (RAMA), a new benchmark that challenges models with both unseen executable instructions and defective ones that should be rejected. In RAMA, we construct a dataset with over 14,000 samples, including diverse defective instructions spanning six dimensions: visual, physical, semantic, motion, safety, and out-of-context. We further propose the Rational Vision-Language-Action model (RationalVLA). It is a dual system for robotic arms that integrates the high-level vision-language model with the low-level manipulation policy by introducing learnable latent space embeddings. This design enables RationalVLA to reason over instructions, reject infeasible commands, and execute manipulation effectively. Experiments demonstrate that RationalVLA outperforms state-of-the-art baselines on RAMA by a 14.5% higher success rate and 0.94 average task length, while maintaining competitive performance on standard manipulation tasks. Real-world trials further validate its effectiveness and robustness in practical applications. Our project page is this https URL. 

**Abstract (ZH)**: 现实世界机器人部署的基本要求是能够理解和响应自然语言指令。现有基于语言的操纵任务通常假设指令与环境完全对齐。这种假设限制了在实际场景中的鲁棒性和泛化性，因为指令可能具有模糊性、无关性或不可行性。为了解决这一问题，我们引入了RAtional MAnipulation (RAMA)，这是一个新的基准，挑战模型处理未见过的可执行指令和应被拒绝的错误指令。在RAMA中，我们构建了一个包含超过14,000个样本的数据集，涵盖六维缺陷指令：视觉、物理、语义、运动、安全和脱节。我们进一步提出了Rational Vision-Language-Action模型（RationalVLA）。这是一种双系统，通过引入可学习的潜在空间嵌入将高层视觉-语言模型与低层操纵策略相结合。这种设计使RationalVLA能够处理指令、拒绝不可行的命令并有效执行操纵。实验表明，RationalVLA在RAMA上的成功率比最先进的基线高出14.5%，平均任务长度缩短0.94个单位，并且在标准操纵任务上保持了竞争力。在现实世界的试验证实了其在实际应用中的有效性和鲁棒性。我们的项目页面为 this https URL。 

---
# In-Hand Object Pose Estimation via Visual-Tactile Fusion 

**Title (ZH)**: 基于视觉-触觉融合的手持物体姿态估计 

**Authors**: Felix Nonnengießer, Alap Kshirsagar, Boris Belousov, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2506.10787)  

**Abstract**: Accurate in-hand pose estimation is crucial for robotic object manipulation, but visual occlusion remains a major challenge for vision-based approaches. This paper presents an approach to robotic in-hand object pose estimation, combining visual and tactile information to accurately determine the position and orientation of objects grasped by a robotic hand. We address the challenge of visual occlusion by fusing visual information from a wrist-mounted RGB-D camera with tactile information from vision-based tactile sensors mounted on the fingertips of a robotic gripper. Our approach employs a weighting and sensor fusion module to combine point clouds from heterogeneous sensor types and control each modality's contribution to the pose estimation process. We use an augmented Iterative Closest Point (ICP) algorithm adapted for weighted point clouds to estimate the 6D object pose. Our experiments show that incorporating tactile information significantly improves pose estimation accuracy, particularly when occlusion is high. Our method achieves an average pose estimation error of 7.5 mm and 16.7 degrees, outperforming vision-only baselines by up to 20%. We also demonstrate the ability of our method to perform precise object manipulation in a real-world insertion task. 

**Abstract (ZH)**: 基于视觉和触觉信息的在手物体姿态估计方法：克服视觉遮挡挑战 

---
# Grounded Vision-Language Navigation for UAVs with Open-Vocabulary Goal Understanding 

**Title (ZH)**: 基于开放词汇目标理解的无人机接地视觉语言导航 

**Authors**: Yuhang Zhang, Haosheng Yu, Jiaping Xiao, Mir Feroskhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10756)  

**Abstract**: Vision-and-language navigation (VLN) is a long-standing challenge in autonomous robotics, aiming to empower agents with the ability to follow human instructions while navigating complex environments. Two key bottlenecks remain in this field: generalization to out-of-distribution environments and reliance on fixed discrete action spaces. To address these challenges, we propose Vision-Language Fly (VLFly), a framework tailored for Unmanned Aerial Vehicles (UAVs) to execute language-guided flight. Without the requirement for localization or active ranging sensors, VLFly outputs continuous velocity commands purely from egocentric observations captured by an onboard monocular camera. The VLFly integrates three modules: an instruction encoder based on a large language model (LLM) that reformulates high-level language into structured prompts, a goal retriever powered by a vision-language model (VLM) that matches these prompts to goal images via vision-language similarity, and a waypoint planner that generates executable trajectories for real-time UAV control. VLFly is evaluated across diverse simulation environments without additional fine-tuning and consistently outperforms all baselines. Moreover, real-world VLN tasks in indoor and outdoor environments under direct and indirect instructions demonstrate that VLFly achieves robust open-vocabulary goal understanding and generalized navigation capabilities, even in the presence of abstract language input. 

**Abstract (ZH)**: 视觉-语言导航（VLN）是自主机器人领域的长期挑战，旨在赋予智能体在复杂环境中遵循人类指令的能力。该领域存在的两大瓶颈是跨分布环境泛化和依赖固定离散动作空间。为了解决这些挑战，我们提出了适用于无人驾驶飞行器（UAV）的视觉-语言飞行动作框架（VLFly），该框架能够执行语言引导的飞行。VLFly 不需要定位或主动测距传感器，仅通过机载单目相机拍摄的主观观测数据输出连续速度命令。VLFly 集成了三个模块：一个基于大型语言模型（LLM）的指令编码器，将高层次语言重新格式化为结构化提示；一个由视觉-语言模型（VLM）驱动的目标检索器，通过视觉-语言相似性将这些提示匹配到目标图像；以及一个航点规划器，生成用于实时UAV控制的可执行轨迹。VLFly 在多种模拟环境中进行了评估，无需额外微调，并且始终优于所有基线。此外，在室内和室外环境下的直接和间接指令下的真实世界VLN任务表明，VLFly 能够实现鲁棒性的开放词汇目标理解以及泛化的导航能力，即使在出现抽象语言输入的情况下也是如此。 

---
# EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence 

**Title (ZH)**: EmbodiedGen: 通往生成式三维世界引擎的体态智能之路 

**Authors**: Wang Xinjie, Liu Liu, Cao Yu, Wu Ruiqi, Qin Wenkang, Wang Dehui, Sui Wei, Su Zhizhong  

**Link**: [PDF](https://arxiv.org/pdf/2506.10600)  

**Abstract**: Constructing a physically realistic and accurately scaled simulated 3D world is crucial for the training and evaluation of embodied intelligence tasks. The diversity, realism, low cost accessibility and affordability of 3D data assets are critical for achieving generalization and scalability in embodied AI. However, most current embodied intelligence tasks still rely heavily on traditional 3D computer graphics assets manually created and annotated, which suffer from high production costs and limited realism. These limitations significantly hinder the scalability of data driven approaches. We present EmbodiedGen, a foundational platform for interactive 3D world generation. It enables the scalable generation of high-quality, controllable and photorealistic 3D assets with accurate physical properties and real-world scale in the Unified Robotics Description Format (URDF) at low cost. These assets can be directly imported into various physics simulation engines for fine-grained physical control, supporting downstream tasks in training and evaluation. EmbodiedGen is an easy-to-use, full-featured toolkit composed of six key modules: Image-to-3D, Text-to-3D, Texture Generation, Articulated Object Generation, Scene Generation and Layout Generation. EmbodiedGen generates diverse and interactive 3D worlds composed of generative 3D assets, leveraging generative AI to address the challenges of generalization and evaluation to the needs of embodied intelligence related research. Code is available at this https URL. 

**Abstract (ZH)**: 构建一个物理真实且准确缩放的三维世界对于训练和评估具身智能任务至关重要。三维数据资产的多样性、逼真度、低成本和易获取性对于实现具身人工智能的泛化和可扩展性至关重要。然而，当前大多数具身智能任务仍 heavily 依赖于手工创建和标注的传统三维计算机图形资产，这些资产存在高昂的制作成本和有限的逼真度问题。这些局限性显著阻碍了数据驱动方法的可扩展性。我们提出 EmbodiedGen，一个交互式三维世界生成的基础平台。它能够以低成本生成高质量、可控制和照片级真实的三维资产，这些资产具有准确的物理属性和真实世界比例，符合统一机器人描述格式（URDF）。这些资产可以直接导入各种物理仿真引擎，以支持精确的物理控制，从而支持训练和评估中的下游任务。EmbodiedGen 是一个易于使用且功能齐全的工具包，由六个关键模块组成：图像到三维、文本到三维、纹理生成、连杆对象生成、场景生成和布局生成。EmbodiedGen 利用生成式人工智能生成多样化的交互式三维世界，以应对具身智能相关研究中泛化和评估的挑战。代码可从以下链接获取。 

---
# Are We Generalizing from the Exception? An In-the-Wild Study on Group-Sensitive Conversation Design in Human-Agent Interactions 

**Title (ZH)**: 我们从例外中泛化吗？关于人类-代理交互中分组敏感对话设计的野外研究 

**Authors**: Ana Müller, Sabina Jeschke, Anja Richert  

**Link**: [PDF](https://arxiv.org/pdf/2506.10462)  

**Abstract**: This paper investigates the impact of a group-adaptive conversation design in two socially interactive agents (SIAs) through two real-world studies. Both SIAs - Furhat, a social robot, and MetaHuman, a virtual agent - were equipped with a conversational artificial intelligence (CAI) backend combining hybrid retrieval and generative models. The studies were carried out in an in-the-wild setting with a total of $N = 188$ participants who interacted with the SIAs - in dyads, triads or larger groups - at a German museum. Although the results did not reveal a significant effect of the group-sensitive conversation design on perceived satisfaction, the findings provide valuable insights into the challenges of adapting CAI for multi-party interactions and across different embodiments (robot vs.\ virtual agent), highlighting the need for multimodal strategies beyond linguistic pluralization. These insights contribute to the fields of Human-Agent Interaction (HAI), Human-Robot Interaction (HRI), and broader Human-Machine Interaction (HMI), providing insights for future research on effective dialogue adaptation in group settings. 

**Abstract (ZH)**: 本文通过两项实地研究，探讨了一种群体自适应对话设计在两个社会互动代理（SIAs）中的影响。在德国博物馆，共有188名参与者与Furhat（一种社会机器人）和MetaHuman（一种虚拟代理）等SIAs进行了二元、三元或更大规模的互动。尽管研究结果未能显著显示群体敏感对话设计对满意度感知的影响，但 findings 为多方面适应会话人工智能（CAI）以及不同类型实体（机器人 vs. 虚拟代理）之间的交互挑战提供了有价值的见解，突显了超越语言多样化的多模态策略的必要性。这些见解为人类-代理交互（HAI）、人类-机器人交互（HRI）和更广泛的人机交互（HMI）领域未来关于群组环境中有效对话适应的研究提供了指导。 

---
# A Navigation Framework Utilizing Vision-Language Models 

**Title (ZH)**: 利用视觉语言模型的导航框架 

**Authors**: Yicheng Duan, Kaiyu tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10172)  

**Abstract**: Vision-and-Language Navigation (VLN) presents a complex challenge in embodied AI, requiring agents to interpret natural language instructions and navigate through visually rich, unfamiliar environments. Recent advances in large vision-language models (LVLMs), such as CLIP and Flamingo, have significantly improved multimodal understanding but introduced new challenges related to computational cost and real-time deployment. In this project, we propose a modular, plug-and-play navigation framework that decouples vision-language understanding from action planning. By integrating a frozen vision-language model, Qwen2.5-VL-7B-Instruct, with lightweight planning logic, we aim to achieve flexible, fast, and adaptable navigation without extensive model fine-tuning. Our framework leverages prompt engineering, structured history management, and a two-frame visual input strategy to enhance decision-making continuity across navigation steps. We evaluate our system on the Room-to-Room benchmark within the VLN-CE setting using the Matterport3D dataset and Habitat-Lab simulation environment. Although our initial results reveal challenges in generalizing to unseen environments under strict evaluation settings, our modular approach lays a foundation for scalable and efficient navigation systems, highlighting promising directions for future improvement through enhanced environmental priors and expanded multimodal input integration. 

**Abstract (ZH)**: 基于视觉-语言的导航（VLN）在实体AI中提出了复杂挑战，要求代理解释自然语言指示并导航通过视觉丰富且不熟悉的环境。大型视觉-语言模型（LVLM）的最新进展，例如CLIP和Flamingo，显著提高了多模态理解能力，但也引入了计算成本和实时部署的新挑战。在本项目中，我们提出了一种模块化、即插即用的导航框架，将视觉-语言理解与动作规划解耦。通过将 frozen 视觉-语言模型 Qwen2.5-VL-7B-Instruct 与轻量级规划逻辑集成，我们旨在在无需大量模型微调的情况下实现灵活、快速且适应性强的导航。我们的框架利用提示工程、结构化历史管理以及双帧视觉输入策略来增强导航步骤间的决策连贯性。我们使用Matterport3D数据集和Habitat-Lab仿真环境，在VLN-CE设置下的Room-to-Room基准上评估了我们的系统。尽管初始结果在严格评估条件下显示了向未见环境泛化的挑战，但我们的模块化方法为可扩展和高效的导航系统奠定了基础，揭示了通过增强环境先验和扩展多模态输入集成的未来改进方向。 

---
# Cybernetic Marionette: Channeling Collective Agency Through a Wearable Robot in a Live Dancer-Robot Duet 

**Title (ZH)**: 机械傀儡控制装置：通过穿戴式机器人在真人与机器人舞者双人舞中引导集体行为 

**Authors**: Anup Sathya, Jiasheng Li, Zeyu Yan, Adriane Fang, Bill Kules, Jonathan David Martin, Huaishu Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10079)  

**Abstract**: We describe DANCE^2, an interactive dance performance in which audience members channel their collective agency into a dancer-robot duet by voting on the behavior of a wearable robot affixed to the dancer's body. At key moments during the performance, the audience is invited to either continue the choreography or override it, shaping the unfolding interaction through real-time collective input. While post-performance surveys revealed that participants felt their choices meaningfully influenced the performance, voting data across four public performances exhibited strikingly consistent patterns. This tension between what audience members do, what they feel, and what actually changes highlights a complex interplay between agentive behavior, the experience of agency, and power. We reflect on how choreography, interaction design, and the structure of the performance mediate this relationship, offering a live analogy for algorithmically curated digital systems where agency is felt, but not exercised. 

**Abstract (ZH)**: DANCE^2：一种通过投票控制舞者-机器人搭档行为的互动舞蹈表演 

---
# Optimus-3: Towards Generalist Multimodal Minecraft Agents with Scalable Task Experts 

**Title (ZH)**: Optimus-3: 向Towards通用 multimodal Minecraft 代理的过渡，配备可扩展的任务专家。 

**Authors**: Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Weili Guan, Dongmei Jiang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.10357)  

**Abstract**: Recently, agents based on multimodal large language models (MLLMs) have achieved remarkable progress across various domains. However, building a generalist agent with capabilities such as perception, planning, action, grounding, and reflection in open-world environments like Minecraft remains challenges: insufficient domain-specific data, interference among heterogeneous tasks, and visual diversity in open-world settings. In this paper, we address these challenges through three key contributions. 1) We propose a knowledge-enhanced data generation pipeline to provide scalable and high-quality training data for agent development. 2) To mitigate interference among heterogeneous tasks, we introduce a Mixture-of-Experts (MoE) architecture with task-level routing. 3) We develop a Multimodal Reasoning-Augmented Reinforcement Learning approach to enhance the agent's reasoning ability for visual diversity in Minecraft. Built upon these innovations, we present Optimus-3, a general-purpose agent for Minecraft. Extensive experimental results demonstrate that Optimus-3 surpasses both generalist multimodal large language models and existing state-of-the-art agents across a wide range of tasks in the Minecraft environment. Project page: this https URL 

**Abstract (ZH)**: 近日，基于多模态大型语言模型的代理已在多个领域取得了显著进展。然而，在像Minecraft这样的开放世界环境中构建具备感知、规划、行动、地面化和反思等能力的一般性代理依然面临挑战：领域特定数据不足、异构任务间的干扰以及开放世界环境中的视觉多样性。在本文中，我们通过三项关键贡献应对这些挑战。1) 我们提出了一种知识增强的数据生成管道，为代理开发提供可扩展且高质量的训练数据。2) 为减少异构任务间的干扰，我们引入了任务级路由的Mixture-of-Experts (MoE) 架构。3) 我们开发了一种多模态推理增强的强化学习方法，以增强代理在Minecraft中的视觉多样性推理能力。基于这些创新，我们介绍了一种通用型代理Optimus-3。广泛的经验结果表明，Optimus-3在Minecraft环境中的多种任务上均超越了现有的泛化多模态大型语言模型和先进代理。项目页面：this https URL 

---
# VideoDeepResearch: Long Video Understanding With Agentic Tool Using 

**Title (ZH)**: VideoDeepResearch: 使用代理工具理解长视频 

**Authors**: Huaying Yuan, Zheng Liu, Junjie Zhou, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.10821)  

**Abstract**: Long video understanding (LVU) presents a significant challenge for current multi-modal large language models (MLLMs) due to the task's inherent complexity and context window constraint. It is widely assumed that addressing LVU tasks requires foundation MLLMs with extended context windows, strong visual perception capabilities, and proficient domain expertise. In this work, we challenge this common belief by introducing VideoDeepResearch, a novel agentic framework for long video understanding. Our approach relies solely on a text-only large reasoning model (LRM) combined with a modular multi-modal toolkit, including multimodal retrievers and visual perceivers, all of which are readily available in practice. For each LVU task, the system formulates a problem-solving strategy through reasoning, while selectively accessing and utilizing essential video content via tool using. We conduct extensive experiments on popular LVU benchmarks, including MLVU, Video-MME, and LVBench. Our results demonstrate that VideoDeepResearch achieves substantial improvements over existing MLLM baselines, surpassing the previous state-of-the-art by 9.6%, 6.6%, and 3.9% on MLVU (test), LVBench, and LongVideoBench, respectively. These findings highlight the promise of agentic systems in overcoming key challenges in LVU problems. 

**Abstract (ZH)**: 长视频理解（LVU）对于当前的多模态大型语言模型（MLLMs）而言 poses 一项显著挑战，主要是由于该任务固有的复杂性和上下文窗口限制。人们普遍认为，解决LVU任务需要具有扩展上下文窗口、强大视觉感知能力和深厚领域专业知识的基础MLLMs。在本工作中，我们通过引入VideoDeepResearch，一种新颖的自主框架来挑战这一常见信念，VideoDeepResearch为长视频理解提供了解决方案。我们的方法仅依赖于一个纯文本大型推理模型（LRM），并结合了一个模块化的多模态工具包，包括多模态检索器和视觉感知器，它们在实践中都是现成可用的。对于每个LVU任务，系统通过推理来制定问题解决策略，并根据需要访问和利用关键视频内容。我们在流行的LVU基准测试中进行了广泛的实验，包括MLVU、Video-MME和LVBench。我们的结果表明，VideoDeepResearch在MLVU（测试）、LVBench和LongVideoBench上的表现分别优于现有MLLM基线9.6%、6.6%和3.9%，这些发现突显了自主系统在克服LVU问题核心挑战方面的潜力。 

---
# SlotPi: Physics-informed Object-centric Reasoning Models 

**Title (ZH)**: SlotPi：物理信息导向的对象中心推理模型 

**Authors**: Jian Li, Wan Han, Ning Lin, Yu-Liang Zhan, Ruizhi Chengze, Haining Wang, Yi Zhang, Hongsheng Liu, Zidong Wang, Fan Yu, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.10778)  

**Abstract**: Understanding and reasoning about dynamics governed by physical laws through visual observation, akin to human capabilities in the real world, poses significant challenges. Currently, object-centric dynamic simulation methods, which emulate human behavior, have achieved notable progress but overlook two critical aspects: 1) the integration of physical knowledge into models. Humans gain physical insights by observing the world and apply this knowledge to accurately reason about various dynamic scenarios; 2) the validation of model adaptability across diverse scenarios. Real-world dynamics, especially those involving fluids and objects, demand models that not only capture object interactions but also simulate fluid flow characteristics. To address these gaps, we introduce SlotPi, a slot-based physics-informed object-centric reasoning model. SlotPi integrates a physical module based on Hamiltonian principles with a spatio-temporal prediction module for dynamic forecasting. Our experiments highlight the model's strengths in tasks such as prediction and Visual Question Answering (VQA) on benchmark and fluid datasets. Furthermore, we have created a real-world dataset encompassing object interactions, fluid dynamics, and fluid-object interactions, on which we validated our model's capabilities. The model's robust performance across all datasets underscores its strong adaptability, laying a foundation for developing more advanced world models. 

**Abstract (ZH)**: 通过视觉观察理解由物理定律支配的动力学并进行推理，类似于人类在现实世界中的能力，提出了重大的挑战。当前，以对象为中心的动力学仿真方法尽管模仿了人类行为并在领域内取得了显著进展，但仍忽视了两个关键方面：1）将物理知识集成到模型中。人类通过观察世界获得物理直觉，并将这些知识应用于对各种动力学场景的准确推理；2）验证模型在不同场景下的适应性。尤其是涉及流体和物体的现实世界动力学要求模型不仅捕捉对象间的相互作用，还能模拟流体流动特性。为解决这些差距，我们提出了SlotPi，一个基于槽位的物理知情对象中心推理模型。SlotPi 结合了基于Hamilton原理的物理模块和时空预测模块进行动态预测。我们的实验展示了该模型在预测和视觉问答（VQA）任务中的优势，并在基准数据集和流体数据集上进行了验证。此外，我们创建了一个包含对象相互作用、流体动力学和流体-对象相互作用的现实世界数据集，并验证了该模型的能力。模型在所有数据集上的稳健表现突显了其强大的适应性，为开发更高级的世界模型奠定了基础。 

---
# Task Adaptation from Skills: Information Geometry, Disentanglement, and New Objectives for Unsupervised Reinforcement Learning 

**Title (ZH)**: 技能适配任务：信息几何、解缠绕及无监督强化学习的新目标 

**Authors**: Yucheng Yang, Tianyi Zhou, Qiang He, Lei Han, Mykola Pechenizkiy, Meng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10629)  

**Abstract**: Unsupervised reinforcement learning (URL) aims to learn general skills for unseen downstream tasks. Mutual Information Skill Learning (MISL) addresses URL by maximizing the mutual information between states and skills but lacks sufficient theoretical analysis, e.g., how well its learned skills can initialize a downstream task's policy. Our new theoretical analysis in this paper shows that the diversity and separability of learned skills are fundamentally critical to downstream task adaptation but MISL does not necessarily guarantee these properties. To complement MISL, we propose a novel disentanglement metric LSEPIN. Moreover, we build an information-geometric connection between LSEPIN and downstream task adaptation cost. For better geometric properties, we investigate a new strategy that replaces the KL divergence in information geometry with Wasserstein distance. We extend the geometric analysis to it, which leads to a novel skill-learning objective WSEP. It is theoretically justified to be helpful to downstream task adaptation and it is capable of discovering more initial policies for downstream tasks than MISL. We finally propose another Wasserstein distance-based algorithm PWSEP that can theoretically discover all optimal initial policies. 

**Abstract (ZH)**: 无监督强化学习中的互信息技能学习及其理论分析：一个新的几何视角 

---
