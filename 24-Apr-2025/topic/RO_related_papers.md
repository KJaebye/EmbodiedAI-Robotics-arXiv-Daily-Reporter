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
# PP-Tac: Paper Picking Using Tactile Feedback in Dexterous Robotic Hands 

**Title (ZH)**: PP-Tac: 使用触觉反馈的灵巧机器人手拣纸方法 

**Authors**: Pei Lin, Yuzhe Huang, Wanlin Li, Jianpeng Ma, Chenxi Xiao, Ziyuan Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.16649)  

**Abstract**: Robots are increasingly envisioned as human companions, assisting with everyday tasks that often involve manipulating deformable objects. Although recent advances in robotic hardware and embodied AI have expanded their capabilities, current systems still struggle with handling thin, flat, and deformable objects such as paper and fabric. This limitation arises from the lack of suitable perception techniques for robust state estimation under diverse object appearances, as well as the absence of planning techniques for generating appropriate grasp motions. To bridge these gaps, this paper introduces PP-Tac, a robotic system for picking up paper-like objects. PP-Tac features a multi-fingered robotic hand with high-resolution omnidirectional tactile sensors \sensorname. This hardware configuration enables real-time slip detection and online frictional force control that mitigates such slips. Furthermore, grasp motion generation is achieved through a trajectory synthesis pipeline, which first constructs a dataset of finger's pinching motions. Based on this dataset, a diffusion-based policy is trained to control the hand-arm robotic system. Experiments demonstrate that PP-Tac can effectively grasp paper-like objects of varying material, thickness, and stiffness, achieving an overall success rate of 87.5\%. To our knowledge, this work is the first attempt to grasp paper-like deformable objects using a tactile dexterous hand. Our project webpage can be found at: this https URL 

**Abstract (ZH)**: 机器人作为人类伴侣，越来越多地被设想用于辅助日常任务，这些任务经常涉及操作柔性物体。尽管最近机器人硬件和体态AI的进步扩展了它们的能力，但现有系统仍然难以处理纸张和织物等薄、平且可变形的物体。这一局限性源于缺乏适合多种物体外观的感知技术来进行稳健的状态估计，以及缺乏生成适当抓取动作的规划技术。为了弥合这些差距，本文引入了PP-Tac，一种用于拾取纸张类似物体的机器人系统。PP-Tac配备了一个具有高分辨率全方位触觉传感器的多指机器人手\sensorname。这种硬件配置能够实现实时打滑检测和在线摩擦力控制，以减轻打滑现象。此外，通过轨迹合成管道生成抓取动作，该管道首先构建了手指捏持动作的数据集。基于此数据集，采用扩散模型训练策略以控制手-臂机器人系统。实验表明，PP-Tac能够有效抓取不同材料、厚度和刚度的纸张类似物体，总体成功率达到了87.5%。据我们所知，这是首次尝试使用触觉灵巧手抓取纸张类似可变形物体的研究。更多项目信息请访问：this https URL 

---
# The Dodecacopter: a Versatile Multirotor System of Dodecahedron-Shaped Modules 

**Title (ZH)**: 十二面体旋翼机：一种多面体模块的多功能多旋翼系统 

**Authors**: Kévin Garanger, Thanakorn Khamvilai, Jeremy Epps, Eric Feron  

**Link**: [PDF](https://arxiv.org/pdf/2504.16475)  

**Abstract**: With the promise of greater safety and adaptability, modular reconfigurable uncrewed air vehicles have been proposed as unique, versatile platforms holding the potential to replace multiple types of monolithic vehicles at once. State-of-the-art rigidly assembled modular vehicles are generally two-dimensional configurations in which the rotors are coplanar and assume the shape of a "flight array". We introduce the Dodecacopter, a new type of modular rotorcraft where all modules take the shape of a regular dodecahedron, allowing the creation of richer sets of configurations beyond flight arrays. In particular, we show how the chosen module design can be used to create three-dimensional and fully actuated configurations. We justify the relevance of these types of configurations in terms of their structural and actuation properties with various performance indicators. Given the broad range of configurations and capabilities that can be achieved with our proposed design, we formulate tractable optimization programs to find optimal configurations given structural and actuation constraints. Finally, a prototype of such a vehicle is presented along with results of performed flights in multiple configurations. 

**Abstract (ZH)**: 具有更高安全性和适应性的模块化可重构无人驾驶航空器因其独特的多功能平台潜力而被提出，有望同时取代多种类型的整体式航空器。最先进的刚性组装模块化飞行器通常是二维配置，其中旋翼共面并形成“飞行阵列”。我们介绍了一种新的模块化旋翼机——十二面体旋翼机，其中所有模块均具有正十二面体的形状，允许创建超越飞行阵列的更丰富配置集。特别是，我们展示了所选模块设计如何用于创建三维和完全驱动的配置。我们通过各种性能指标从结构和驱动特性角度论证了这些配置类型的相关性。鉴于我们提出的这种设计能够实现广泛的配置和能力，我们制定了可行的优化程序，以在结构和驱动约束条件下找到最优配置。最后，我们呈现了一种此类飞行器的原型及其在多种配置下进行的飞行实验结果。 

---
# Fast and Modular Whole-Body Lagrangian Dynamics of Legged Robots with Changing Morphology 

**Title (ZH)**: 快速且模块化的腿式机器人动态模型及其形态变化下的整体现量拉格朗日动力学 

**Authors**: Sahand Farghdani, Omar Abdelrahman, Robin Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2504.16383)  

**Abstract**: Fast and modular modeling of multi-legged robots (MLRs) is essential for resilient control, particularly under significant morphological changes caused by mechanical damage. Conventional fixed-structure models, often developed with simplifying assumptions for nominal gaits, lack the flexibility to adapt to such scenarios. To address this, we propose a fast modular whole-body modeling framework using Boltzmann-Hamel equations and screw theory, in which each leg's dynamics is modeled independently and assembled based on the current robot morphology. This singularity-free, closed-form formulation enables efficient design of model-based controllers and damage identification algorithms. Its modularity allows autonomous adaptation to various damage configurations without manual re-derivation or retraining of neural networks. We validate the proposed framework using a custom simulation engine that integrates contact dynamics, a gait generator, and local leg control. Comparative simulations against hardware tests on a hexapod robot with multiple leg damage confirm the model's accuracy and adaptability. Additionally, runtime analyses reveal that the proposed model is approximately three times faster than real-time, making it suitable for real-time applications in damage identification and recovery. 

**Abstract (ZH)**: 快速且模块化建模多足机器人（MLRs）对于在显著形态变化引起的机械损伤情况下实现鲁棒控制至关重要。传统的固定结构模型通常基于对名义步态的简化假设进行开发，缺乏适应此类场景的灵活性。为此，我们提出了一种使用玻尔兹曼-哈梅尔方程和轴矢理论构建的快速模块化全身模型框架，其中每个腿的动力学独立建模，并基于当前机器人的形态进行组装。这种无奇点的闭式表述使基于模型的控制器和损伤识别算法的设计变得高效。其模块化特性允许自主适应各种损伤配置，而无需手动重新推导或重新训练神经网络。我们利用一个集成了接触动力学、步态生成器和局部腿控制的定制仿真引擎验证了所提出的方法。与对一个具有多足损伤的六足机器人进行的硬件测试的对比仿真实验表明了模型的准确性和适应性。此外，运行时分析表明，所提出模型的运行速度大约是实时的三倍，使其适用于损伤识别和恢复的实时应用。 

---
# Road Similarity-Based BEV-Satellite Image Matching for UGV Localization 

**Title (ZH)**: 基于道路相似性的BEV-卫星图像匹配在UGV定位中的应用 

**Authors**: Zhenping Sun, Chuang Yang, Yafeng Bu, Bokai Liu, Jun Zeng, Xiaohui Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.16346)  

**Abstract**: To address the challenge of autonomous UGV localization in GNSS-denied off-road environments,this study proposes a matching-based localization method that leverages BEV perception image and satellite map within a road similarity space to achieve high-precision this http URL first implement a robust LiDAR-inertial odometry system, followed by the fusion of LiDAR and image data to generate a local BEV perception image of the UGV. This approach mitigates the significant viewpoint discrepancy between ground-view images and satellite map. The BEV image and satellite map are then projected into the road similarity space, where normalized cross correlation (NCC) is computed to assess the matching this http URL, a particle filter is employed to estimate the probability distribution of the vehicle's this http URL comparing with GNSS ground truth, our localization system demonstrated stability without divergence over a long-distance test of 10 km, achieving an average lateral error of only 0.89 meters and an average planar Euclidean error of 3.41 meters. Furthermore, it maintained accurate and stable global localization even under nighttime conditions, further validating its robustness and adaptability. 

**Abstract (ZH)**: 自主无人地面车辆在GNSS受限非道路环境中的基于匹配的局部化方法 

---
# Mass-Adaptive Admittance Control for Robotic Manipulators 

**Title (ZH)**: 基于质量自适应 admittance 控制的机器人 manipulator 

**Authors**: Hossein Gholampour, Jonathon E. Slightam, Logan E. Beaver  

**Link**: [PDF](https://arxiv.org/pdf/2504.16224)  

**Abstract**: Handling objects with unknown or changing masses is a common challenge in robotics, often leading to errors or instability if the control system cannot adapt in real-time. In this paper, we present a novel approach that enables a six-degrees-of-freedom robotic manipulator to reliably follow waypoints while automatically estimating and compensating for unknown payload weight. Our method integrates an admittance control framework with a mass estimator, allowing the robot to dynamically update an excitation force to compensate for the payload mass. This strategy mitigates end-effector sagging and preserves stability when handling objects of unknown weights. We experimentally validated our approach in a challenging pick-and-place task on a shelf with a crossbar, improved accuracy in reaching waypoints and compliant motion compared to a baseline admittance-control scheme. By safely accommodating unknown payloads, our work enhances flexibility in robotic automation and represents a significant step forward in adaptive control for uncertain environments. 

**Abstract (ZH)**: 处理未知或变化质量的物体是机器人技术中一个常见的挑战，如果控制系统无法实时适应，通常会导致错误或不稳定。在本文中，我们提出了一种新颖的方法，使六自由度机器人 manipulator 能够可靠地跟随路径点，同时自动估计并补偿未知负载质量。该方法结合了阻抗控制框架和质量估算器，允许机器人动态更新激励力以补偿负载质量。该策略在处理未知重量物体时可以缓解末端执行器下垂并保持稳定性。我们通过在带有横梁的货架上执行一项具有挑战性的拾取和放置任务，实验验证了该方法，并优于基准阻抗控制方案，提高了到达路径点的准确性和顺应运动。通过安全地容纳未知负载，我们的工作增强了机器人自动化灵活性，并在不确定环境中的自适应控制方面取得了重要进展。 

---
