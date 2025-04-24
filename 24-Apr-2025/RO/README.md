# Latent Diffusion Planning for Imitation Learning 

**Title (ZH)**: 潜在扩散规划用于模仿学习 

**Authors**: Amber Xie, Oleh Rybkin, Dorsa Sadigh, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2504.16925)  

**Abstract**: Recent progress in imitation learning has been enabled by policy architectures that scale to complex visuomotor tasks, multimodal distributions, and large datasets. However, these methods often rely on learning from large amount of expert demonstrations. To address these shortcomings, we propose Latent Diffusion Planning (LDP), a modular approach consisting of a planner which can leverage action-free demonstrations, and an inverse dynamics model which can leverage suboptimal data, that both operate over a learned latent space. First, we learn a compact latent space through a variational autoencoder, enabling effective forecasting of future states in image-based domains. Then, we train a planner and an inverse dynamics model with diffusion objectives. By separating planning from action prediction, LDP can benefit from the denser supervision signals of suboptimal and action-free data. On simulated visual robotic manipulation tasks, LDP outperforms state-of-the-art imitation learning approaches, as they cannot leverage such additional data. 

**Abstract (ZH)**: 近期imitation learning的进步得益于能够处理复杂视觉运动任务、多模态分布和大数据集的策略架构。然而，这些方法往往依赖于从大量专家演示中学习。为此，我们提出了潜在扩散规划（LDP），这是一种模块化的方法，包括一个可以利用无需动作演示的规划器，以及一个可以利用亚最优数据的动力学逆模型，两者都操作在一个学习到的潜在空间中。首先，我们通过变分自编码器学习一个紧凑的潜在空间，使基于图像的领域能够有效预测未来状态。然后，我们用扩散目标训练规划器和动力学逆模型。通过将规划与动作预测分离，LDP可以从亚最优和无需动作的数据中获得更密集的监督信号。在模拟视觉机器人操作任务中，LDP优于当前最先进的imitation learning方法，因为它们无法利用此类额外数据。 

---
# Meta-Learning Online Dynamics Model Adaptation in Off-Road Autonomous Driving 

**Title (ZH)**: 离路自动驾驶中元学习在线动力模型适应 

**Authors**: Jacob Levy, Jason Gibson, Bogdan Vlahov, Erica Tevere, Evangelos Theodorou, David Fridovich-Keil, Patrick Spieler  

**Link**: [PDF](https://arxiv.org/pdf/2504.16923)  

**Abstract**: High-speed off-road autonomous driving presents unique challenges due to complex, evolving terrain characteristics and the difficulty of accurately modeling terrain-vehicle interactions. While dynamics models used in model-based control can be learned from real-world data, they often struggle to generalize to unseen terrain, making real-time adaptation essential. We propose a novel framework that combines a Kalman filter-based online adaptation scheme with meta-learned parameters to address these challenges. Offline meta-learning optimizes the basis functions along which adaptation occurs, as well as the adaptation parameters, while online adaptation dynamically adjusts the onboard dynamics model in real time for model-based control. We validate our approach through extensive experiments, including real-world testing on a full-scale autonomous off-road vehicle, demonstrating that our method outperforms baseline approaches in prediction accuracy, performance, and safety metrics, particularly in safety-critical scenarios. Our results underscore the effectiveness of meta-learned dynamics model adaptation, advancing the development of reliable autonomous systems capable of navigating diverse and unseen environments. Video is available at: this https URL 

**Abstract (ZH)**: 高速离线路面自主驾驶由于复杂多变的路面特性和地面车辆交互建模的难度而面临独特挑战。我们提出了一种新颖框架，该框架结合了基于卡尔曼滤波的在线自适应方案和元学习得到的参数，以应对这些挑战。离线元学习优化了自适应过程中使用的基函数，以及自适应参数，而在线自适应则在实时控制中动态调整车载动力学模型。我们通过广泛实验验证了该方法，包括在全尺寸自主离线车辆上进行实地测试，结果表明，与基线方法相比，我们的方法在预测精度、性能和安全性指标方面表现更优，特别是在安全关键场景中。我们的结果强调了元学习动力学模型自适应的有效性，促进了可靠自主系统的开发，这些系统能够导航多样性和未知环境。视频见: [this https URL] 

---
# Zero-shot Sim-to-Real Transfer for Reinforcement Learning-based Visual Servoing of Soft Continuum Arms 

**Title (ZH)**: 基于强化学习的软连续臂视觉伺服零样本模拟到现实转移 

**Authors**: Hsin-Jung Yang, Mahsa Khosravi, Benjamin Walt, Girish Krishnan, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.16916)  

**Abstract**: Soft continuum arms (SCAs) soft and deformable nature presents challenges in modeling and control due to their infinite degrees of freedom and non-linear behavior. This work introduces a reinforcement learning (RL)-based framework for visual servoing tasks on SCAs with zero-shot sim-to-real transfer capabilities, demonstrated on a single section pneumatic manipulator capable of bending and twisting. The framework decouples kinematics from mechanical properties using an RL kinematic controller for motion planning and a local controller for actuation refinement, leveraging minimal sensing with visual feedback. Trained entirely in simulation, the RL controller achieved a 99.8% success rate. When deployed on hardware, it achieved a 67% success rate in zero-shot sim-to-real transfer, demonstrating robustness and adaptability. This approach offers a scalable solution for SCAs in 3D visual servoing, with potential for further refinement and expanded applications. 

**Abstract (ZH)**: 基于强化学习的软连续臂视觉伺服方法：零样本模拟到现实的转移能力 

---
# MorphoNavi: Aerial-Ground Robot Navigation with Object Oriented Mapping in Digital Twin 

**Title (ZH)**: MorphoNavi: 基于对象导向建模的空地机器人导航在数字孖体中 

**Authors**: Sausar Karaf, Mikhail Martynov, Oleg Sautenkov, Zhanibek Darush, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2504.16914)  

**Abstract**: This paper presents a novel mapping approach for a universal aerial-ground robotic system utilizing a single monocular camera. The proposed system is capable of detecting a diverse range of objects and estimating their positions without requiring fine-tuning for specific environments. The system's performance was evaluated through a simulated search-and-rescue scenario, where the MorphoGear robot successfully located a robotic dog while an operator monitored the process. This work contributes to the development of intelligent, multimodal robotic systems capable of operating in unstructured environments. 

**Abstract (ZH)**: 本文提出了一种利用单目摄像头对通用空地机器人系统进行新型映射的方法。所提出的系统能够检测多样化的物体并估计其位置，无需针对特定环境进行微调。通过模拟的搜救场景评估了该系统的性能， MorphoGear 机器人成功定位了一只机械狗，操作员在一旁监控整个过程。本文为开发能够在非结构化环境中运行的智能多模态机器人系统做出了贡献。 

---
# Physically Consistent Humanoid Loco-Manipulation using Latent Diffusion Models 

**Title (ZH)**: 物理一致的人形Loco-Manipulation Using Latent Diffusion Models 

**Authors**: Ilyass Taouil, Haizhou Zhao, Angela Dai, Majid Khadiv  

**Link**: [PDF](https://arxiv.org/pdf/2504.16843)  

**Abstract**: This paper uses the capabilities of latent diffusion models (LDMs) to generate realistic RGB human-object interaction scenes to guide humanoid loco-manipulation planning. To do so, we extract from the generated images both the contact locations and robot configurations that are then used inside a whole-body trajectory optimization (TO) formulation to generate physically consistent trajectories for humanoids. We validate our full pipeline in simulation for different long-horizon loco-manipulation scenarios and perform an extensive analysis of the proposed contact and robot configuration extraction pipeline. Our results show that using the information extracted from LDMs, we can generate physically consistent trajectories that require long-horizon reasoning. 

**Abstract (ZH)**: 本文利用潜在扩散模型（LDMs）的能力生成逼真的RGB人体-对象交互场景以指导类人行走-操作规划。为此，我们从生成的图像中提取接触位置和机器人配置，然后在全身轨迹优化（TO）公式中使用这些信息生成物理上一致的类人机器人轨迹。我们在仿真中验证了整个管道在不同的长期 horizon 行走-操作场景中的有效性，并对提出的接触和机器人配置提取管道进行了广泛的分析。结果显示，利用LDMs提取的信息，可以生成需要长期 horizon 推理的物理上一致的轨迹。 

---
# Graph2Nav: 3D Object-Relation Graph Generation to Robot Navigation 

**Title (ZH)**: Graph2Nav：3D对象关系图生成在机器人导航中的应用 

**Authors**: Tixiao Shan, Abhinav Rajvanshi, Niluthpol Mithun, Han-Pang Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16782)  

**Abstract**: We propose Graph2Nav, a real-time 3D object-relation graph generation framework, for autonomous navigation in the real world. Our framework fully generates and exploits both 3D objects and a rich set of semantic relationships among objects in a 3D layered scene graph, which is applicable to both indoor and outdoor scenes. It learns to generate 3D semantic relations among objects, by leveraging and advancing state-of-the-art 2D panoptic scene graph works into the 3D world via 3D semantic mapping techniques. This approach avoids previous training data constraints in learning 3D scene graphs directly from 3D data. We conduct experiments to validate the accuracy in locating 3D objects and labeling object-relations in our 3D scene graphs. We also evaluate the impact of Graph2Nav via integration with SayNav, a state-of-the-art planner based on large language models, on an unmanned ground robot to object search tasks in real environments. Our results demonstrate that modeling object relations in our scene graphs improves search efficiency in these navigation tasks. 

**Abstract (ZH)**: 我们提出Graph2Nav，一种实时3D对象关系图生成框架，用于现实世界的自主导航。 

---
# MOSAIC: A Skill-Centric Algorithmic Framework for Long-Horizon Manipulation Planning 

**Title (ZH)**: MOSAIC: 以技能为中心的长期 horizons 操作规划算法框架 

**Authors**: Itamar Mishani, Yorai Shaoul, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2504.16738)  

**Abstract**: Planning long-horizon motions using a set of predefined skills is a key challenge in robotics and AI. Addressing this challenge requires methods that systematically explore skill combinations to uncover task-solving sequences, harness generic, easy-to-learn skills (e.g., pushing, grasping) to generalize across unseen tasks, and bypass reliance on symbolic world representations that demand extensive domain and task-specific knowledge. Despite significant progress, these elements remain largely disjoint in existing approaches, leaving a critical gap in achieving robust, scalable solutions for complex, long-horizon problems. In this work, we present MOSAIC, a skill-centric framework that unifies these elements by using the skills themselves to guide the planning process. MOSAIC uses two families of skills: Generators compute executable trajectories and world configurations, and Connectors link these independently generated skill trajectories by solving boundary value problems, enabling progress toward completing the overall task. By breaking away from the conventional paradigm of incrementally discovering skills from predefined start or goal states--a limitation that significantly restricts exploration--MOSAIC focuses planning efforts on regions where skills are inherently effective. We demonstrate the efficacy of MOSAIC in both simulated and real-world robotic manipulation tasks, showcasing its ability to solve complex long-horizon planning problems using a diverse set of skills incorporating generative diffusion models, motion planning algorithms, and manipulation-specific models. Visit this https URL for demonstrations and examples. 

**Abstract (ZH)**: 使用预定义技能规划长期 horizon 动作是机器人技术和人工智能中的一个关键挑战。克服这一挑战需要系统地探索技能组合的方法，以发现任务解决序列，利用通用且易于学习的技能（例如推拉、抓取）来泛化到未见过的任务，并避免依赖于要求广泛领域和任务特定知识的符号世界表示。尽管取得了显著进展，但这些元素在现有方法中仍然分离，留下了实现复杂、长期 horizons 问题稳健且可扩展解决方案的关键差距。在本文中，我们提出了 MOSAIC，这是一种以技能为中心的框架，通过使用技能本身来指导规划过程，将这些元素统一起来。MOSAIC 使用两类技能：技能生成器计算可执行轨迹和世界配置，技能链接器通过解决边界值问题将独立生成的技能轨迹连接起来，使整体任务的完成成为可能。通过打破从预定义的起始或目标状态逐步发现技能的常规范式--这种限制显著限制了探索--MOSAIC 将规划努力集中在技能本来就有效的区域。我们在模拟和真实世界的机器人操作任务中展示了 MOSAIC 的有效性，展示了其利用生成扩散模型、运动规划算法和特定于操作的模型来解决复杂长期 horizons 规划问题的能力。请访问此链接查看更多演示和示例。 

---
# DYNUS: Uncertainty-aware Trajectory Planner in Dynamic Unknown Environments 

**Title (ZH)**: DYNUS: 动态未知环境中的不确定性感知轨迹规划 

**Authors**: Kota Kondo, Mason Peterson, Nicholas Rober, Juan Rached Viso, Lucas Jia, Jialin Chen, Harvey Merton, Jonathan P. How  

**Link**: [PDF](https://arxiv.org/pdf/2504.16734)  

**Abstract**: This paper introduces DYNUS, an uncertainty-aware trajectory planner designed for dynamic unknown environments. Operating in such settings presents many challenges -- most notably, because the agent cannot predict the ground-truth future paths of obstacles, a previously planned trajectory can become unsafe at any moment, requiring rapid replanning to avoid collisions.
Recently developed planners have used soft-constraint approaches to achieve the necessary fast computation times; however, these methods do not guarantee collision-free paths even with static obstacles. In contrast, hard-constraint methods ensure collision-free safety, but typically have longer computation times.
To address these issues, we propose three key contributions. First, the DYNUS Global Planner (DGP) and Temporal Safe Corridor Generation operate in spatio-temporal space and handle both static and dynamic obstacles in the 3D environment. Second, the Safe Planning Framework leverages a combination of exploratory, safe, and contingency trajectories to flexibly re-route when potential future collisions with dynamic obstacles are detected. Finally, the Fast Hard-Constraint Local Trajectory Formulation uses a variable elimination approach to reduce the problem size and enable faster computation by pre-computing dependencies between free and dependent variables while still ensuring collision-free trajectories.
We evaluated DYNUS in a variety of simulations, including dense forests, confined office spaces, cave systems, and dynamic environments. Our experiments show that DYNUS achieves a success rate of 100% and travel times that are approximately 25.0% faster than state-of-the-art methods. We also evaluated DYNUS on multiple platforms -- a quadrotor, a wheeled robot, and a quadruped -- in both simulation and hardware experiments. 

**Abstract (ZH)**: 一种面向动态未知环境的 awareness 未知的路径规划器 DYNUS 

---
# Offline Robotic World Model: Learning Robotic Policies without a Physics Simulator 

**Title (ZH)**: 离线机器人世界模型：无需物理模拟器学习机器人策略 

**Authors**: Chenhao Li, Andreas Krause, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2504.16680)  

**Abstract**: Reinforcement Learning (RL) has demonstrated impressive capabilities in robotic control but remains challenging due to high sample complexity, safety concerns, and the sim-to-real gap. While offline RL eliminates the need for risky real-world exploration by learning from pre-collected data, it suffers from distributional shift, limiting policy generalization. Model-Based RL (MBRL) addresses this by leveraging predictive models for synthetic rollouts, yet existing approaches often lack robust uncertainty estimation, leading to compounding errors in offline settings. We introduce Offline Robotic World Model (RWM-O), a model-based approach that explicitly estimates epistemic uncertainty to improve policy learning without reliance on a physics simulator. By integrating these uncertainty estimates into policy optimization, our approach penalizes unreliable transitions, reducing overfitting to model errors and enhancing stability. Experimental results show that RWM-O improves generalization and safety, enabling policy learning purely from real-world data and advancing scalable, data-efficient RL for robotics. 

**Abstract (ZH)**: 基于模型的机器人世界模型（RWM-O）：通过显式估计 epistemic 不确定性提高政策学习 

---
# PP-Tac: Paper Picking Using Tactile Feedback in Dexterous Robotic Hands 

**Title (ZH)**: PP-Tac: 使用触觉反馈的灵巧机器人手拣纸方法 

**Authors**: Pei Lin, Yuzhe Huang, Wanlin Li, Jianpeng Ma, Chenxi Xiao, Ziyuan Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.16649)  

**Abstract**: Robots are increasingly envisioned as human companions, assisting with everyday tasks that often involve manipulating deformable objects. Although recent advances in robotic hardware and embodied AI have expanded their capabilities, current systems still struggle with handling thin, flat, and deformable objects such as paper and fabric. This limitation arises from the lack of suitable perception techniques for robust state estimation under diverse object appearances, as well as the absence of planning techniques for generating appropriate grasp motions. To bridge these gaps, this paper introduces PP-Tac, a robotic system for picking up paper-like objects. PP-Tac features a multi-fingered robotic hand with high-resolution omnidirectional tactile sensors \sensorname. This hardware configuration enables real-time slip detection and online frictional force control that mitigates such slips. Furthermore, grasp motion generation is achieved through a trajectory synthesis pipeline, which first constructs a dataset of finger's pinching motions. Based on this dataset, a diffusion-based policy is trained to control the hand-arm robotic system. Experiments demonstrate that PP-Tac can effectively grasp paper-like objects of varying material, thickness, and stiffness, achieving an overall success rate of 87.5\%. To our knowledge, this work is the first attempt to grasp paper-like deformable objects using a tactile dexterous hand. Our project webpage can be found at: this https URL 

**Abstract (ZH)**: 机器人作为人类伴侣，越来越多地被设想用于辅助日常任务，这些任务经常涉及操作柔性物体。尽管最近机器人硬件和体态AI的进步扩展了它们的能力，但现有系统仍然难以处理纸张和织物等薄、平且可变形的物体。这一局限性源于缺乏适合多种物体外观的感知技术来进行稳健的状态估计，以及缺乏生成适当抓取动作的规划技术。为了弥合这些差距，本文引入了PP-Tac，一种用于拾取纸张类似物体的机器人系统。PP-Tac配备了一个具有高分辨率全方位触觉传感器的多指机器人手\sensorname。这种硬件配置能够实现实时打滑检测和在线摩擦力控制，以减轻打滑现象。此外，通过轨迹合成管道生成抓取动作，该管道首先构建了手指捏持动作的数据集。基于此数据集，采用扩散模型训练策略以控制手-臂机器人系统。实验表明，PP-Tac能够有效抓取不同材料、厚度和刚度的纸张类似物体，总体成功率达到了87.5%。据我们所知，这是首次尝试使用触觉灵巧手抓取纸张类似可变形物体的研究。更多项目信息请访问：this https URL 

---
# HERB: Human-augmented Efficient Reinforcement learning for Bin-packing 

**Title (ZH)**: HERB：增强的人类辅助高效装箱强化学习 

**Authors**: Gojko Perovic, Nuno Ferreira Duarte, Atabak Dehban, Gonçalo Teixeira, Egidio Falotico, José Santos-Victor  

**Link**: [PDF](https://arxiv.org/pdf/2504.16595)  

**Abstract**: Packing objects efficiently is a fundamental problem in logistics, warehouse automation, and robotics. While traditional packing solutions focus on geometric optimization, packing irregular, 3D objects presents significant challenges due to variations in shape and stability. Reinforcement Learning~(RL) has gained popularity in robotic packing tasks, but training purely from simulation can be inefficient and computationally expensive. In this work, we propose HERB, a human-augmented RL framework for packing irregular objects. We first leverage human demonstrations to learn the best sequence of objects to pack, incorporating latent factors such as space optimization, stability, and object relationships that are difficult to model explicitly. Next, we train a placement algorithm that uses visual information to determine the optimal object positioning inside a packing container. Our approach is validated through extensive performance evaluations, analyzing both packing efficiency and latency. Finally, we demonstrate the real-world feasibility of our method on a robotic system. Experimental results show that our method outperforms geometric and purely RL-based approaches by leveraging human intuition, improving both packing robustness and adaptability. This work highlights the potential of combining human expertise-driven RL to tackle complex real-world packing challenges in robotic systems. 

**Abstract (ZH)**: 有效地包装物体是物流、仓库自动化和机器人技术中的基本问题。传统的包装解决方案主要集中于几何优化，而包装不规则的3D物体由于形状和稳定性上的变化带来了显著的挑战。强化学习(RL)在机器人包装任务中得到了广泛应用，但仅从模拟中进行训练可能效率低下且计算成本高昂。在本文中，我们提出了一种名为HERB的人机增强RL框架，用于包装不规则物体。我们首先利用人类演示学习出最佳的物体打包顺序，同时考虑了难以明确建模的空间优化、稳定性和物体关系等潜在因素。接下来，我们训练了一个使用视觉信息来确定物体在包装容器内最佳位置的放置算法。通过广泛的性能评估，分析了包装效率和延迟。最后，我们在一个机器人系统上验证了我们方法的现实可行性。实验结果表明，通过利用人类直觉，我们的方法在包装鲁棒性和适应性方面优于基于几何的方法和纯RL方法。本工作突显了将基于人类专业知识的RL结合用于解决机器人系统中复杂包装挑战的潜力。 

---
# The Dodecacopter: a Versatile Multirotor System of Dodecahedron-Shaped Modules 

**Title (ZH)**: 十二面体旋翼机：一种多面体模块的多功能多旋翼系统 

**Authors**: Kévin Garanger, Thanakorn Khamvilai, Jeremy Epps, Eric Feron  

**Link**: [PDF](https://arxiv.org/pdf/2504.16475)  

**Abstract**: With the promise of greater safety and adaptability, modular reconfigurable uncrewed air vehicles have been proposed as unique, versatile platforms holding the potential to replace multiple types of monolithic vehicles at once. State-of-the-art rigidly assembled modular vehicles are generally two-dimensional configurations in which the rotors are coplanar and assume the shape of a "flight array". We introduce the Dodecacopter, a new type of modular rotorcraft where all modules take the shape of a regular dodecahedron, allowing the creation of richer sets of configurations beyond flight arrays. In particular, we show how the chosen module design can be used to create three-dimensional and fully actuated configurations. We justify the relevance of these types of configurations in terms of their structural and actuation properties with various performance indicators. Given the broad range of configurations and capabilities that can be achieved with our proposed design, we formulate tractable optimization programs to find optimal configurations given structural and actuation constraints. Finally, a prototype of such a vehicle is presented along with results of performed flights in multiple configurations. 

**Abstract (ZH)**: 具有更高安全性和适应性的模块化可重构无人驾驶航空器因其独特的多功能平台潜力而被提出，有望同时取代多种类型的整体式航空器。最先进的刚性组装模块化飞行器通常是二维配置，其中旋翼共面并形成“飞行阵列”。我们介绍了一种新的模块化旋翼机——十二面体旋翼机，其中所有模块均具有正十二面体的形状，允许创建超越飞行阵列的更丰富配置集。特别是，我们展示了所选模块设计如何用于创建三维和完全驱动的配置。我们通过各种性能指标从结构和驱动特性角度论证了这些配置类型的相关性。鉴于我们提出的这种设计能够实现广泛的配置和能力，我们制定了可行的优化程序，以在结构和驱动约束条件下找到最优配置。最后，我们呈现了一种此类飞行器的原型及其在多种配置下进行的飞行实验结果。 

---
# ManipDreamer: Boosting Robotic Manipulation World Model with Action Tree and Visual Guidance 

**Title (ZH)**: ManipDreamer: 通过动作树和视觉引导增强机器人操作世界模型 

**Authors**: Ying Li, Xiaobao Wei, Xiaowei Chi, Yuming Li, Zhongyu Zhao, Hao Wang, Ningning Ma, Ming Lu, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16464)  

**Abstract**: While recent advancements in robotic manipulation video synthesis have shown promise, significant challenges persist in ensuring effective instruction-following and achieving high visual quality. Recent methods, like RoboDreamer, utilize linguistic decomposition to divide instructions into separate lower-level primitives, conditioning the world model on these primitives to achieve compositional instruction-following. However, these separate primitives do not consider the relationships that exist between them. Furthermore, recent methods neglect valuable visual guidance, including depth and semantic guidance, both crucial for enhancing visual quality. This paper introduces ManipDreamer, an advanced world model based on the action tree and visual guidance. To better learn the relationships between instruction primitives, we represent the instruction as the action tree and assign embeddings to tree nodes, each instruction can acquire its embeddings by navigating through the action tree. The instruction embeddings can be used to guide the world model. To enhance visual quality, we combine depth and semantic guidance by introducing a visual guidance adapter compatible with the world model. This visual adapter enhances both the temporal and physical consistency of video generation. Based on the action tree and visual guidance, ManipDreamer significantly boosts the instruction-following ability and visual quality. Comprehensive evaluations on robotic manipulation benchmarks reveal that ManipDreamer achieves large improvements in video quality metrics in both seen and unseen tasks, with PSNR improved from 19.55 to 21.05, SSIM improved from 0.7474 to 0.7982 and reduced Flow Error from 3.506 to 3.201 in unseen tasks, compared to the recent RoboDreamer model. Additionally, our method increases the success rate of robotic manipulation tasks by 2.5% in 6 RLbench tasks on average. 

**Abstract (ZH)**: 基于动作树和视觉引导的ManipDreamer：提升机器人 manipulation 视频合成的有效指令遵循能力和视觉质量 

---
# Long Exposure Localization in Darkness Using Consumer Cameras 

**Title (ZH)**: 在黑暗中使用消费级相机进行长时间曝光定位 

**Authors**: Michael Milford, Ian Turner, Peter Corke  

**Link**: [PDF](https://arxiv.org/pdf/2504.16406)  

**Abstract**: In this paper we evaluate performance of the SeqSLAM algorithm for passive vision-based localization in very dark environments with low-cost cameras that result in massively blurred images. We evaluate the effect of motion blur from exposure times up to 10,000 ms from a moving car, and the performance of localization in day time from routes learned at night in two different environments. Finally we perform a statistical analysis that compares the baseline performance of matching unprocessed grayscale images to using patch normalization and local neighborhood normalization - the two key SeqSLAM components. Our results and analysis show for the first time why the SeqSLAM algorithm is effective, and demonstrate the potential for cheap camera-based localization systems that function despite extreme appearance change. 

**Abstract (ZH)**: 本文评估了SeqSLAM算法在极暗环境中使用低成本相机进行被动视觉定位的表现，这些相机产生的图像极度模糊。我们评估了移动汽车在曝光时间长达10,000毫秒时的运动模糊对定位效果的影响，并研究了在夜间学习路径后白天定位的性能在两种不同环境中的表现。最后，我们进行了一项统计分析，比较了直接匹配未处理的灰度图像与使用局部补丁规范化和局部邻域规范化之间的基线性能——SeqSLAM算法的两个关键组件。我们的结果和分析首次揭示了SeqSLAM算法为何有效，并展示了尽管存在极端的外观变化，低成本相机定位系统仍有应用潜力。 

---
# Fast and Modular Whole-Body Lagrangian Dynamics of Legged Robots with Changing Morphology 

**Title (ZH)**: 快速且模块化的腿式机器人动态模型及其形态变化下的整体现量拉格朗日动力学 

**Authors**: Sahand Farghdani, Omar Abdelrahman, Robin Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2504.16383)  

**Abstract**: Fast and modular modeling of multi-legged robots (MLRs) is essential for resilient control, particularly under significant morphological changes caused by mechanical damage. Conventional fixed-structure models, often developed with simplifying assumptions for nominal gaits, lack the flexibility to adapt to such scenarios. To address this, we propose a fast modular whole-body modeling framework using Boltzmann-Hamel equations and screw theory, in which each leg's dynamics is modeled independently and assembled based on the current robot morphology. This singularity-free, closed-form formulation enables efficient design of model-based controllers and damage identification algorithms. Its modularity allows autonomous adaptation to various damage configurations without manual re-derivation or retraining of neural networks. We validate the proposed framework using a custom simulation engine that integrates contact dynamics, a gait generator, and local leg control. Comparative simulations against hardware tests on a hexapod robot with multiple leg damage confirm the model's accuracy and adaptability. Additionally, runtime analyses reveal that the proposed model is approximately three times faster than real-time, making it suitable for real-time applications in damage identification and recovery. 

**Abstract (ZH)**: 快速且模块化建模多足机器人（MLRs）对于在显著形态变化引起的机械损伤情况下实现鲁棒控制至关重要。传统的固定结构模型通常基于对名义步态的简化假设进行开发，缺乏适应此类场景的灵活性。为此，我们提出了一种使用玻尔兹曼-哈梅尔方程和轴矢理论构建的快速模块化全身模型框架，其中每个腿的动力学独立建模，并基于当前机器人的形态进行组装。这种无奇点的闭式表述使基于模型的控制器和损伤识别算法的设计变得高效。其模块化特性允许自主适应各种损伤配置，而无需手动重新推导或重新训练神经网络。我们利用一个集成了接触动力学、步态生成器和局部腿控制的定制仿真引擎验证了所提出的方法。与对一个具有多足损伤的六足机器人进行的硬件测试的对比仿真实验表明了模型的准确性和适应性。此外，运行时分析表明，所提出模型的运行速度大约是实时的三倍，使其适用于损伤识别和恢复的实时应用。 

---
# SILM: A Subjective Intent Based Low-Latency Framework for Multiple Traffic Participants Joint Trajectory Prediction 

**Title (ZH)**: SILM：一种基于主观意图的低延迟多交通参与者联合轨迹预测框架 

**Authors**: Qu Weiming, Wang Jia, Du Jiawei, Zhu Yuanhao, Yu Jianfeng, Xia Rui, Cao Song, Wu Xihong, Luo Dingsheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.16377)  

**Abstract**: Trajectory prediction is a fundamental technology for advanced autonomous driving systems and represents one of the most challenging problems in the field of cognitive intelligence. Accurately predicting the future trajectories of each traffic participant is a prerequisite for building high safety and high reliability decision-making, planning, and control capabilities in autonomous driving. However, existing methods often focus solely on the motion of other traffic participants without considering the underlying intent behind that motion, which increases the uncertainty in trajectory prediction. Autonomous vehicles operate in real-time environments, meaning that trajectory prediction algorithms must be able to process data and generate predictions in real-time. While many existing methods achieve high accuracy, they often struggle to effectively handle heterogeneous traffic scenarios. In this paper, we propose a Subjective Intent-based Low-latency framework for Multiple traffic participants joint trajectory prediction. Our method explicitly incorporates the subjective intent of traffic participants based on their key points, and predicts the future trajectories jointly without map, which ensures promising performance while significantly reducing the prediction latency. Additionally, we introduce a novel dataset designed specifically for trajectory prediction. Related code and dataset will be available soon. 

**Abstract (ZH)**: 基于主体意图的低延迟多交通参与者联合轨迹预测框架 

---
# DPGP: A Hybrid 2D-3D Dual Path Potential Ghost Probe Zone Prediction Framework for Safe Autonomous Driving 

**Title (ZH)**: DPGP：一种用于安全自动驾驶的二维-三维双路径潜在鬼探头区域预测混合框架 

**Authors**: Weiming Qu, Jiawei Du, Shenghai Yuan, Jia Wang, Yang Sun, Shengyi Liu, Yuanhao Zhu, Jianfeng Yu, Song Cao, Rui Xia, Xiaoyu Tang, Xihong Wu, Dingsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.16374)  

**Abstract**: Modern robots must coexist with humans in dense urban environments. A key challenge is the ghost probe problem, where pedestrians or objects unexpectedly rush into traffic paths. This issue affects both autonomous vehicles and human drivers. Existing works propose vehicle-to-everything (V2X) strategies and non-line-of-sight (NLOS) imaging for ghost probe zone detection. However, most require high computational power or specialized hardware, limiting real-world feasibility. Additionally, many methods do not explicitly address this issue. To tackle this, we propose DPGP, a hybrid 2D-3D fusion framework for ghost probe zone prediction using only a monocular camera during training and inference. With unsupervised depth prediction, we observe ghost probe zones align with depth discontinuities, but different depth representations offer varying robustness. To exploit this, we fuse multiple feature embeddings to improve prediction. To validate our approach, we created a 12K-image dataset annotated with ghost probe zones, carefully sourced and cross-checked for accuracy. Experimental results show our framework outperforms existing methods while remaining cost-effective. To our knowledge, this is the first work extending ghost probe zone prediction beyond vehicles, addressing diverse non-vehicle objects. We will open-source our code and dataset for community benefit. 

**Abstract (ZH)**: 现代机器人必须在稠密的城市环境中与人类共存。一个关键挑战是幽灵探测问题，其中行人或物体意外冲入交通路径。这个问题影响自动驾驶车辆和人类驾驶员。现有工作提出了车对外界通信（V2X）策略和非视线（NLOS）成像方法用于幽灵探测区检测。然而，大多数方法需要高计算能力或专用硬件，限制了其实用性。此外，许多方法并没有明确解决这一问题。为了解决这一问题，我们提出了一种基于单一摄像头的混合2D-3D融合框架DPGP，用于幽灵探测区预测。通过无监督深度预测，我们发现幽灵探测区与深度不连续处对齐，但不同的深度表示具有不同的鲁棒性。为充分利用这一点，我们融合多个特征嵌入以改进预测。为了验证我们的方法，我们创建了一个包含12000张标注有幽灵探测区的图像数据集，精心选择并交叉核对以确保准确性。实验结果表明，我们的框架在保持成本效益的同时优于现有方法。据我们所知，这是首项将幽灵探测区预测扩展到车辆之外，解决多样化非车辆对象的工作。我们将开源我们的代码和数据集以惠及社区。 

---
# Fast Online Adaptive Neural MPC via Meta-Learning 

**Title (ZH)**: 快速在线自适应神经MPC通过元学习 

**Authors**: Yu Mei, Xinyu Zhou, Shuyang Yu, Vaibhav Srivastava, Xiaobo Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.16369)  

**Abstract**: Data-driven model predictive control (MPC) has demonstrated significant potential for improving robot control performance in the presence of model uncertainties. However, existing approaches often require extensive offline data collection and computationally intensive training, limiting their ability to adapt online. To address these challenges, this paper presents a fast online adaptive MPC framework that leverages neural networks integrated with Model-Agnostic Meta-Learning (MAML). Our approach focuses on few-shot adaptation of residual dynamics - capturing the discrepancy between nominal and true system behavior - using minimal online data and gradient steps. By embedding these meta-learned residual models into a computationally efficient L4CasADi-based MPC pipeline, the proposed method enables rapid model correction, enhances predictive accuracy, and improves real-time control performance. We validate the framework through simulation studies on a Van der Pol oscillator, a Cart-Pole system, and a 2D quadrotor. Results show significant gains in adaptation speed and prediction accuracy over both nominal MPC and nominal MPC augmented with a freshly initialized neural network, underscoring the effectiveness of our approach for real-time adaptive robot control. 

**Abstract (ZH)**: 基于数据驱动模型预测控制的快速在线自适应框架：融合模型无感知元学习的残差动力学适应 

---
# Road Similarity-Based BEV-Satellite Image Matching for UGV Localization 

**Title (ZH)**: 基于道路相似性的BEV-卫星图像匹配在UGV定位中的应用 

**Authors**: Zhenping Sun, Chuang Yang, Yafeng Bu, Bokai Liu, Jun Zeng, Xiaohui Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.16346)  

**Abstract**: To address the challenge of autonomous UGV localization in GNSS-denied off-road environments,this study proposes a matching-based localization method that leverages BEV perception image and satellite map within a road similarity space to achieve high-precision this http URL first implement a robust LiDAR-inertial odometry system, followed by the fusion of LiDAR and image data to generate a local BEV perception image of the UGV. This approach mitigates the significant viewpoint discrepancy between ground-view images and satellite map. The BEV image and satellite map are then projected into the road similarity space, where normalized cross correlation (NCC) is computed to assess the matching this http URL, a particle filter is employed to estimate the probability distribution of the vehicle's this http URL comparing with GNSS ground truth, our localization system demonstrated stability without divergence over a long-distance test of 10 km, achieving an average lateral error of only 0.89 meters and an average planar Euclidean error of 3.41 meters. Furthermore, it maintained accurate and stable global localization even under nighttime conditions, further validating its robustness and adaptability. 

**Abstract (ZH)**: 自主无人地面车辆在GNSS受限非道路环境中的基于匹配的局部化方法 

---
# PCF-Grasp: Converting Point Completion to Geometry Feature to Enhance 6-DoF Grasp 

**Title (ZH)**: PCF-抓取：将点完成转换为几何特征以增强6自由度抓取 

**Authors**: Yaofeng Cheng, Fusheng Zha, Wei Guo, Pengfei Wang, Chao Zeng, Lining Sun, Chenguang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16320)  

**Abstract**: The 6-Degree of Freedom (DoF) grasp method based on point clouds has shown significant potential in enabling robots to grasp target objects. However, most existing methods are based on the point clouds (2.5D points) generated from single-view depth images. These point clouds only have one surface side of the object providing incomplete geometry information, which mislead the grasping algorithm to judge the shape of the target object, resulting in low grasping accuracy. Humans can accurately grasp objects from a single view by leveraging their geometry experience to estimate object shapes. Inspired by humans, we propose a novel 6-DoF grasping framework that converts the point completion results as object shape features to train the 6-DoF grasp network. Here, point completion can generate approximate complete points from the 2.5D points similar to the human geometry experience, and converting it as shape features is the way to utilize it to improve grasp efficiency. Furthermore, due to the gap between the network generation and actual execution, we integrate a score filter into our framework to select more executable grasp proposals for the real robot. This enables our method to maintain a high grasp quality in any camera viewpoint. Extensive experiments demonstrate that utilizing complete point features enables the generation of significantly more accurate grasp proposals and the inclusion of a score filter greatly enhances the credibility of real-world robot grasping. Our method achieves a 17.8\% success rate higher than the state-of-the-art method in real-world experiments. 

**Abstract (ZH)**: 基于点云的6自由度抓取方法在使机器人抓取目标物体方面展现出了显著潜力。然而，现有大多数方法基于单视角深度图像生成的2.5D点云。这些点云仅提供物体一面的几何信息，导致抓取算法错误判断目标物体的形状，从而降低了抓取准确性。受到人类单视角准确抓取物体的经验启发，我们提出了一种新颖的6自由度抓取框架，通过将点云补全结果转换为物体形状特征来训练6自由度抓取网络。点云补全可以产生与人类几何经验类似的近似完整点，将其转换为形状特征是利用其提升抓取效率的一种方式。此外，由于网络生成与实际执行之间的差距，我们在框架中整合了一个评分滤波器，以选择更适合实际机器人执行的抓取方案。这使得我们方法在任何相机视角下都能维持较高的抓取质量。大量实验表明，利用完整点特征生成的抓取提案更加准确，同时加入评分滤波器极大地提升了实际机器人抓取的可信度。在实际实验中，我们的方法实现了比现有最佳方法高出17.8%的成功率。 

---
# Vision Controlled Orthotic Hand Exoskeleton 

**Title (ZH)**: 视觉控制的假手外骨骼 

**Authors**: Connor Blais, Md Abdul Baset Sarker, Masudul H. Imtiaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.16319)  

**Abstract**: This paper presents the design and implementation of an AI vision-controlled orthotic hand exoskeleton to enhance rehabilitation and assistive functionality for individuals with hand mobility impairments. The system leverages a Google Coral Dev Board Micro with an Edge TPU to enable real-time object detection using a customized MobileNet\_V2 model trained on a six-class dataset. The exoskeleton autonomously detects objects, estimates proximity, and triggers pneumatic actuation for grasp-and-release tasks, eliminating the need for user-specific calibration needed in traditional EMG-based systems. The design prioritizes compactness, featuring an internal battery. It achieves an 8-hour runtime with a 1300 mAh battery. Experimental results demonstrate a 51ms inference speed, a significant improvement over prior iterations, though challenges persist in model robustness under varying lighting conditions and object orientations. While the most recent YOLO model (YOLOv11) showed potential with 15.4 FPS performance, quantization issues hindered deployment. The prototype underscores the viability of vision-controlled exoskeletons for real-world assistive applications, balancing portability, efficiency, and real-time responsiveness, while highlighting future directions for model optimization and hardware miniaturization. 

**Abstract (ZH)**: 基于AI视觉控制的手部外骨骼设计与实现：增强手部运动障碍个体的康复和辅助功能 

---
# Mass-Adaptive Admittance Control for Robotic Manipulators 

**Title (ZH)**: 基于质量自适应 admittance 控制的机器人 manipulator 

**Authors**: Hossein Gholampour, Jonathon E. Slightam, Logan E. Beaver  

**Link**: [PDF](https://arxiv.org/pdf/2504.16224)  

**Abstract**: Handling objects with unknown or changing masses is a common challenge in robotics, often leading to errors or instability if the control system cannot adapt in real-time. In this paper, we present a novel approach that enables a six-degrees-of-freedom robotic manipulator to reliably follow waypoints while automatically estimating and compensating for unknown payload weight. Our method integrates an admittance control framework with a mass estimator, allowing the robot to dynamically update an excitation force to compensate for the payload mass. This strategy mitigates end-effector sagging and preserves stability when handling objects of unknown weights. We experimentally validated our approach in a challenging pick-and-place task on a shelf with a crossbar, improved accuracy in reaching waypoints and compliant motion compared to a baseline admittance-control scheme. By safely accommodating unknown payloads, our work enhances flexibility in robotic automation and represents a significant step forward in adaptive control for uncertain environments. 

**Abstract (ZH)**: 处理未知或变化质量的物体是机器人技术中一个常见的挑战，如果控制系统无法实时适应，通常会导致错误或不稳定。在本文中，我们提出了一种新颖的方法，使六自由度机器人 manipulator 能够可靠地跟随路径点，同时自动估计并补偿未知负载质量。该方法结合了阻抗控制框架和质量估算器，允许机器人动态更新激励力以补偿负载质量。该策略在处理未知重量物体时可以缓解末端执行器下垂并保持稳定性。我们通过在带有横梁的货架上执行一项具有挑战性的拾取和放置任务，实验验证了该方法，并优于基准阻抗控制方案，提高了到达路径点的准确性和顺应运动。通过安全地容纳未知负载，我们的工作增强了机器人自动化灵活性，并在不确定环境中的自适应控制方面取得了重要进展。 

---
# Measuring Uncertainty in Shape Completion to Improve Grasp Quality 

**Title (ZH)**: 基于形状补全测量不确定性以提高抓取质量 

**Authors**: Nuno Ferreira Duarte, Seyed S. Mohammadi, Plinio Moreno, Alessio Del Bue, Jose Santos-Victor  

**Link**: [PDF](https://arxiv.org/pdf/2504.16183)  

**Abstract**: Shape completion networks have been used recently in real-world robotic experiments to complete the missing/hidden information in environments where objects are only observed in one or few instances where self-occlusions are bound to occur. Nowadays, most approaches rely on deep neural networks that handle rich 3D point cloud data that lead to more precise and realistic object geometries. However, these models still suffer from inaccuracies due to its nondeterministic/stochastic inferences which could lead to poor performance in grasping scenarios where these errors compound to unsuccessful grasps. We present an approach to calculate the uncertainty of a 3D shape completion model during inference of single view point clouds of an object on a table top. In addition, we propose an update to grasp pose algorithms quality score by introducing the uncertainty of the completed point cloud present in the grasp candidates. To test our full pipeline we perform real world grasping with a 7dof robotic arm with a 2 finger gripper on a large set of household objects and compare against previous approaches that do not measure uncertainty. Our approach ranks the grasp quality better, leading to higher grasp success rate for the rank 5 grasp candidates compared to state of the art. 

**Abstract (ZH)**: 基于单视角点云的3D形状完成模型的不确定性评估及抓取质量分数更新方法 

---
# PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation 

**Title (ZH)**: PIN-WM: 学习物理导向的世界模型以实现非摄取性 manipulation 

**Authors**: Wenxuan Li, Hang Zhao, Zhiyuan Yu, Yu Du, Qin Zou, Ruizhen Hu, Kai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16693)  

**Abstract**: While non-prehensile manipulation (e.g., controlled pushing/poking) constitutes a foundational robotic skill, its learning remains challenging due to the high sensitivity to complex physical interactions involving friction and restitution. To achieve robust policy learning and generalization, we opt to learn a world model of the 3D rigid body dynamics involved in non-prehensile manipulations and use it for model-based reinforcement learning. We propose PIN-WM, a Physics-INformed World Model that enables efficient end-to-end identification of a 3D rigid body dynamical system from visual observations. Adopting differentiable physics simulation, PIN-WM can be learned with only few-shot and task-agnostic physical interaction trajectories. Further, PIN-WM is learned with observational loss induced by Gaussian Splatting without needing state estimation. To bridge Sim2Real gaps, we turn the learned PIN-WM into a group of Digital Cousins via physics-aware randomizations which perturb physics and rendering parameters to generate diverse and meaningful variations of the PIN-WM. Extensive evaluations on both simulation and real-world tests demonstrate that PIN-WM, enhanced with physics-aware digital cousins, facilitates learning robust non-prehensile manipulation skills with Sim2Real transfer, surpassing the Real2Sim2Real state-of-the-arts. 

**Abstract (ZH)**: 基于物理信息的世界模型在非抓持操作中的Sim2Real迁移学习 

---
# Insect-Computer Hybrid Speaker: Speaker using Chirp of the Cicada Controlled by Electrical Muscle Stimulation 

**Title (ZH)**: 昆虫-计算机混合语音装置：通过电肌肉刺激控制的蝉鸣语音发生器 

**Authors**: Yuga Tsukuda, Naoto Nishida, Jun Lu, Yoichi Ochiai  

**Link**: [PDF](https://arxiv.org/pdf/2504.16459)  

**Abstract**: We propose "Insect-Computer Hybrid Speaker", which enables us to make musics made from combinations of computer and insects. Lots of studies have proposed methods and interfaces for controlling insects and obtaining feedback. However, there have been less research on the use of insects for interaction with third parties. In this paper, we propose a method in which cicadas are used as speakers triggered by using Electrical Muscle Stimulation (EMS). We explored and investigated the suitable waveform of chirp to be controlled, the appropriate voltage range, and the maximum pitch at which cicadas can chirp. 

**Abstract (ZH)**: 昆虫-计算机混合音箱：基于电肌肉刺激的蝉鸣音乐生成方法 

---
# Eigendecomposition Parameterization of Penalty Matrices for Enhanced Control Design: Aerospace Applications 

**Title (ZH)**: Penalty矩阵特征分解参数化方法在航空航天控制设计中的增强应用 

**Authors**: Nicholas P. Nurre, Ehsan Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2504.16328)  

**Abstract**: Modern control algorithms require tuning of square weight/penalty matrices appearing in quadratic functions/costs to improve performance and/or stability output. Due to simplicity in gain-tuning and enforcing positive-definiteness, diagonal penalty matrices are used extensively in control methods such as linear quadratic regulator (LQR), model predictive control, and Lyapunov-based control. In this paper, we propose an eigendecomposition approach to parameterize penalty matrices, allowing positive-definiteness with non-zero off-diagonal entries to be implicitly satisfied, which not only offers notable computational and implementation advantages, but broadens the class of achievable controls. We solve three control problems: 1) a variation of Zermelo's navigation problem, 2) minimum-energy spacecraft attitude control using both LQR and Lyapunov-based methods, and 3) minimum-fuel and minimum-time Lyapunov-based low-thrust trajectory design. Particle swarm optimization is used to optimize the decision variables, which will parameterize the penalty matrices. The results demonstrate improvements of up to 65% in the performance objective in the example problems utilizing the proposed method. 

**Abstract (ZH)**: 现代控制算法需要调节出现在二次函数/成本中的方权重/惩罚矩阵以提高性能和/或稳定性输出。由于增益调节简单且能保证正定性，对角惩罚矩阵在线性二次调节（LQR）、模型预测控制和Lyapunov基于的控制方法中广泛使用。本文提出了一种特征值分解方法来参数化惩罚矩阵，允许在非零非对角元素下隐式满足正定性，不仅提供了显著的计算和实现优势，还扩大了可实现控制的范围。我们解决了三个控制问题：1) Zermelo航行问题的变体；2) 使用LQR和Lyapunov基于方法的最小能量航天器姿态控制；3) 最小燃料和最小时间Lyapunov基于的小推力轨道设计。使用粒子群优化来优化决策变量，这些变量将参数化惩罚矩阵。结果显示，在使用所提方法的示例问题中，性能目标提高了最多65%。 

---
# MARFT: Multi-Agent Reinforcement Fine-Tuning 

**Title (ZH)**: 多智能体强化学习微调：MARFT 

**Authors**: Junwei Liao, Muning Wen, Jun Wang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.16129)  

**Abstract**: LLM-based Multi-Agent Systems have demonstrated remarkable capabilities in addressing complex, agentic tasks requiring multifaceted reasoning and collaboration, from generating high-quality presentation slides to conducting sophisticated scientific research. Meanwhile, RL has been widely recognized for its effectiveness in enhancing agent intelligence, but limited research has investigated the fine-tuning of LaMAS using foundational RL techniques. Moreover, the direct application of MARL methodologies to LaMAS introduces significant challenges, stemming from the unique characteristics and mechanisms inherent to LaMAS. To address these challenges, this article presents a comprehensive study of LLM-based MARL and proposes a novel paradigm termed Multi-Agent Reinforcement Fine-Tuning (MARFT). We introduce a universal algorithmic framework tailored for LaMAS, outlining the conceptual foundations, key distinctions, and practical implementation strategies. We begin by reviewing the evolution from RL to Reinforcement Fine-Tuning, setting the stage for a parallel analysis in the multi-agent domain. In the context of LaMAS, we elucidate critical differences between MARL and MARFT. These differences motivate a transition toward a novel, LaMAS-oriented formulation of RFT. Central to this work is the presentation of a robust and scalable MARFT framework. We detail the core algorithm and provide a complete, open-source implementation to facilitate adoption and further research. The latter sections of the paper explore real-world application perspectives and opening challenges in MARFT. By bridging theoretical underpinnings with practical methodologies, this work aims to serve as a roadmap for researchers seeking to advance MARFT toward resilient and adaptive solutions in agentic systems. Our implementation of the proposed framework is publicly available at: this https URL. 

**Abstract (ZH)**: 基于大语言模型的多agent系统在处理需要多维推理和协作的复杂任务方面展现了显著的能力，从生成高质量的演示文稿到开展复杂的科学研究。同时，强化学习因其提升代理智能的效用而受到广泛认可，但有关使用基础强化学习技术微调LaMAS的研究相对有限。此外，直接将MARL方法应用于LaMAS带来了重大挑战，源于LaMAS固有的独特特性和机制。为应对这些挑战，本文对基于大语言模型的多agent系统 reinforcement learning 进行了全面研究，并提出了一种新的范式，即多agent强化学习微调（MARFT）。我们介绍了一个通用的算法框架，适用于LaMAS，概述了其概念基础、关键区别和实用实施方案。我们首先回顾从强化学习到强化学习微调的发展，为多agent领域的平行分析奠定基础。在LaMAS背景下，我们阐明了MARL与MARFT之间的关键差异，这些差异促使我们向一种新的、面向LaMAS的强化学习微调形式转变。本文的核心在于展现了一个稳健和可扩展的MARFT框架。我们详细描述了核心算法，并提供了完整的开源实现，以促进其采用和进一步研究。论文后半部分探讨了MARFT在实际应用中的前景和挑战。通过结合理论基础与实践方法，本文旨在为希望将MARFT推向弹性且适应性强的多agent系统解决方案的研究人员提供指导。我们提出的框架的实现已公开发布于：this https URL。 

---
# MonoTher-Depth: Enhancing Thermal Depth Estimation via Confidence-Aware Distillation 

**Title (ZH)**: MonoTher-Depth：通过信心 Awareness 蒸馏增强热深度估计 

**Authors**: Xingxing Zuo, Nikhil Ranganathan, Connor Lee, Georgia Gkioxari, Soon-Jo Chung  

**Link**: [PDF](https://arxiv.org/pdf/2504.16127)  

**Abstract**: Monocular depth estimation (MDE) from thermal images is a crucial technology for robotic systems operating in challenging conditions such as fog, smoke, and low light. The limited availability of labeled thermal data constrains the generalization capabilities of thermal MDE models compared to foundational RGB MDE models, which benefit from datasets of millions of images across diverse scenarios. To address this challenge, we introduce a novel pipeline that enhances thermal MDE through knowledge distillation from a versatile RGB MDE model. Our approach features a confidence-aware distillation method that utilizes the predicted confidence of the RGB MDE to selectively strengthen the thermal MDE model, capitalizing on the strengths of the RGB model while mitigating its weaknesses. Our method significantly improves the accuracy of the thermal MDE, independent of the availability of labeled depth supervision, and greatly expands its applicability to new scenarios. In our experiments on new scenarios without labeled depth, the proposed confidence-aware distillation method reduces the absolute relative error of thermal MDE by 22.88\% compared to the baseline without distillation. 

**Abstract (ZH)**: 单目热成像深度估计：通过知识蒸馏提高在复杂环境中的鲁棒性 

---
# Shape Your Ground: Refining Road Surfaces Beyond Planar Representations 

**Title (ZH)**: 塑造地面形态：超越平面表示的道路表面精Refining 改进 

**Authors**: Oussema Dhaouadi, Johannes Meier, Jacques Kaiser, Daniel Cremers  

**Link**: [PDF](https://arxiv.org/pdf/2504.16103)  

**Abstract**: Road surface reconstruction from aerial images is fundamental for autonomous driving, urban planning, and virtual simulation, where smoothness, compactness, and accuracy are critical quality factors. Existing reconstruction methods often produce artifacts and inconsistencies that limit usability, while downstream tasks have a tendency to represent roads as planes for simplicity but at the cost of accuracy. We introduce FlexRoad, the first framework to directly address road surface smoothing by fitting Non-Uniform Rational B-Splines (NURBS) surfaces to 3D road points obtained from photogrammetric reconstructions or geodata providers. Our method at its core utilizes the Elevation-Constrained Spatial Road Clustering (ECSRC) algorithm for robust anomaly correction, significantly reducing surface roughness and fitting errors. To facilitate quantitative comparison between road surface reconstruction methods, we present GeoRoad Dataset (GeRoD), a diverse collection of road surface and terrain profiles derived from openly accessible geodata. Experiments on GeRoD and the photogrammetry-based DeepScenario Open 3D Dataset (DSC3D) demonstrate that FlexRoad considerably surpasses commonly used road surface representations across various metrics while being insensitive to various input sources, terrains, and noise types. By performing ablation studies, we identify the key role of each component towards high-quality reconstruction performance, making FlexRoad a generic method for realistic road surface modeling. 

**Abstract (ZH)**: 基于航拍图像的道路表面重建对于自动驾驶、城市规划和虚拟 simulation 至关重要，平滑度、紧凑性和准确性是关键质量因素。现有的重建方法常常产生伪影和不一致性，限制了其可用性，而下游任务倾向于简化地将道路表示为平面以提高效率，但会牺牲准确性。我们引入了 FlexRoad，这是首个通过使用非均匀有理B样条（NURBS）表面拟合三维道路点来直接解决道路表面平滑问题的框架，这些点来自光达重建或地理数据服务商。该方法的核心采用了高斯约束空间道路聚类（ECSRC）算法进行稳健的异常值修正，显著减少了表面粗糙度和拟合误差。为了便于道路表面重建方法的定量比较，我们提出了 GeoRoad 数据集（GeRoD），该数据集包含来自开放获取地理数据的地表和地形剖面。实验表明，FlexRoad 在各种指标上显著超越了常用的道路表面表示方法，且对不同输入源、地形和噪声类型不敏感。通过进行消融研究，我们确定了每个组件在高质量重建性能中的关键作用，使 FlexRoad 成为一种通用的道路表面建模方法。 

---
# Audio and Multiscale Visual Cues Driven Cross-modal Transformer for Idling Vehicle Detection 

**Title (ZH)**: 基于音频和多尺度视觉线索驱动的跨模态变压器在Idle车辆检测中的应用 

**Authors**: Xiwen Li, Ross Whitaker, Tolga Tasdizen  

**Link**: [PDF](https://arxiv.org/pdf/2504.16102)  

**Abstract**: Idling vehicle detection (IVD) supports real-time systems that reduce pollution and emissions by dynamically messaging drivers to curb excess idling behavior. In computer vision, IVD has become an emerging task that leverages video from surveillance cameras and audio from remote microphones to localize and classify vehicles in each frame as moving, idling, or engine-off. As with other cross-modal tasks, the key challenge lies in modeling the correspondence between audio and visual modalities, which differ in representation but provide complementary cues -- video offers spatial and motion context, while audio conveys engine activity beyond the visual field. The previous end-to-end model, which uses a basic attention mechanism, struggles to align these modalities effectively, often missing vehicle detections. To address this issue, we propose AVIVDNetv2, a transformer-based end-to-end detection network. It incorporates a cross-modal transformer with global patch-level learning, a multiscale visual feature fusion module, and decoupled detection heads. Extensive experiments show that AVIVDNetv2 improves mAP by 7.66 over the disjoint baseline and 9.42 over the E2E baseline, with consistent AP gains across all vehicle categories. Furthermore, AVIVDNetv2 outperforms the state-of-the-art method for sounding object localization, establishing a new performance benchmark on the AVIVD dataset. 

**Abstract (ZH)**: 基于多模态的车辆怠速检测（AVIVDNetv2）：一种端到端的检测网络 

---
