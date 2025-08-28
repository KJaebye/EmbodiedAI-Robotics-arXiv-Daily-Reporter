# A Standing Support Mobility Robot for Enhancing Independence in Elderly Daily Living 

**Title (ZH)**: 站立支撑型移动机器人，用于提高老年人日常生活中的独立性 

**Authors**: Ricardo J. Manríquez-Cisterna, Ankit A. Ravankar, Jose V. Salazar Luces, Takuro Hatsukari, Yasuhisa Hirata  

**Link**: [PDF](https://arxiv.org/pdf/2508.19816)  

**Abstract**: This paper presents a standing support mobility robot "Moby" developed to enhance independence and safety for elderly individuals during daily activities such as toilet transfers. Unlike conventional seated mobility aids, the robot maintains users in an upright posture, reducing physical strain, supporting natural social interaction at eye level, and fostering a greater sense of self-efficacy. Moby offers a novel alternative by functioning both passively and with mobility support, enabling users to perform daily tasks more independently. Its main advantages include ease of use, lightweight design, comfort, versatility, and effective sit-to-stand assistance. The robot leverages the Robot Operating System (ROS) for seamless control, featuring manual and autonomous operation modes. A custom control system enables safe and intuitive interaction, while the integration with NAV2 and LiDAR allows for robust navigation capabilities. This paper reviews existing mobility solutions and compares them to Moby, details the robot's design, and presents objective and subjective experimental results using the NASA-TLX method and time comparisons to other methods to validate our design criteria and demonstrate the advantages of our contribution. 

**Abstract (ZH)**: 本文介绍了一种站立支持移动机器人“Moby”，旨在提升老年人在如如厕转移等日常活动中的独立性和安全性。与传统的坐式助动装置不同，该机器人保持用户立姿，减轻身体负担，支持自然的平视社交互动，增强自我效能感。Moby 提供了一种新型替代方案，既能被动使用也能辅助移动，使用户能更独立地完成日常任务。其主要优势包括易于使用、轻巧设计、舒适性、多功能性和有效的坐起辅助。该机器人利用Robot Operating System (ROS) 进行无缝控制，具备手动和自主操作模式。自定义控制系统确保安全直观的交互，而与NAV2和LiDAR的集成赋予了强大的导航能力。本文回顾了现有的移动解决方案，与Moby 进行比较，详述了机器人的设计，并通过采用NASA-TLX 方法和时间比较其他方法的客观和主观实验结果来验证我们的设计标准，展示我们贡献的优势。 

---
# APT*: Asymptotically Optimal Motion Planning via Adaptively Prolated Elliptical R-Nearest Neighbors 

**Title (ZH)**: APT*: 通过自适应拉长椭圆R-最近邻实现渐近最优运动规划 

**Authors**: Liding Zhang, Sicheng Wang, Kuanqi Cai, Zhenshan Bing, Fan Wu, Chaoqun Wang, Sami Haddadin, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.19790)  

**Abstract**: Optimal path planning aims to determine a sequence of states from a start to a goal while accounting for planning objectives. Popular methods often integrate fixed batch sizes and neglect information on obstacles, which is not problem-specific. This study introduces Adaptively Prolated Trees (APT*), a novel sampling-based motion planner that extends based on Force Direction Informed Trees (FDIT*), integrating adaptive batch-sizing and elliptical $r$-nearest neighbor modules to dynamically modulate the path searching process based on environmental feedback. APT* adjusts batch sizes based on the hypervolume of the informed sets and considers vertices as electric charges that obey Coulomb's law to define virtual forces via neighbor samples, thereby refining the prolate nearest neighbor selection. These modules employ non-linear prolate methods to adaptively adjust the electric charges of vertices for force definition, thereby improving the convergence rate with lower solution costs. Comparative analyses show that APT* outperforms existing single-query sampling-based planners in dimensions from $\mathbb{R}^4$ to $\mathbb{R}^{16}$, and it was further validated through a real-world robot manipulation task. A video showcasing our experimental results is available at: this https URL 

**Abstract (ZH)**: 自适应拉长树（APT*）：一种基于力方向引导树的动态采样路径规划方法 

---
# Tree-Based Grafting Approach for Bidirectional Motion Planning with Local Subsets Optimization 

**Title (ZH)**: 基于树的方法在局部子集优化下的双向运动规划接枝策略 

**Authors**: Liding Zhang, Yao Ling, Zhenshan Bing, Fan Wu, Sami Haddadin, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.19776)  

**Abstract**: Bidirectional motion planning often reduces planning time compared to its unidirectional counterparts. It requires connecting the forward and reverse search trees to form a continuous path. However, this process could fail and restart the asymmetric bidirectional search due to the limitations of lazy-reverse search. To address this challenge, we propose Greedy GuILD Grafting Trees (G3T*), a novel path planner that grafts invalid edge connections at both ends to re-establish tree-based connectivity, enabling rapid path convergence. G3T* employs a greedy approach using the minimum Lebesgue measure of guided incremental local densification (GuILD) subsets to optimize paths efficiently. Furthermore, G3T* dynamically adjusts the sampling distribution between the informed set and GuILD subsets based on historical and current cost improvements, ensuring asymptotic optimality. These features enhance the forward search's growth towards the reverse tree, achieving faster convergence and lower solution costs. Benchmark experiments across dimensions from R^2 to R^8 and real-world robotic evaluations demonstrate G3T*'s superior performance compared to existing single-query sampling-based planners. A video showcasing our experimental results is available at: this https URL 

**Abstract (ZH)**: 双向运动规划通常比其单向 counterparts 更能减少规划时间。它需要连接正向和反向搜索树以形成连续路径。然而，这一过程可能由于懒惰反向搜索的限制而失败，从而导致异步双向搜索的重新开始。为了解决这一挑战，我们提出了一种新型路径规划器 Greedy GuILD Grafting Trees (G3T*)，该规划器在两端嫁接无效边连接以重新建立基于树的连通性，从而实现快速路径收敛。G3T* 使用最小勒贝格测度的引导增量局部稠密化 (GuILD) 子集的贪婪方法来高效优化路径。此外，G3T* 根据历史和当前成本改进动态调整已知集合与 GuILD 子集之间的采样分布，确保渐近最优性。这些功能增强了正向搜索向反向树的生长，实现了更快的收敛和更低的解决方案成本。来自 R^2 到 R^8 的多维度基准实验和实际机器人评估表明，G3T* 在与现有单查询基于采样的规划器相比时表现出更优性能。我们实验结果的视频请参见：this https URL。 

---
# Elliptical K-Nearest Neighbors -- Path Optimization via Coulomb's Law and Invalid Vertices in C-space Obstacles 

**Title (ZH)**: 椭圆K最近邻——通过库仑定律和C空间障碍中的无效顶点进行路径优化 

**Authors**: Liding Zhang, Zhenshan Bing, Yu Zhang, Kuanqi Cai, Lingyun Chen, Fan Wu, Sami Haddadin, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.19771)  

**Abstract**: Path planning has long been an important and active research area in robotics. To address challenges in high-dimensional motion planning, this study introduces the Force Direction Informed Trees (FDIT*), a sampling-based planner designed to enhance speed and cost-effectiveness in pathfinding. FDIT* builds upon the state-of-the-art informed sampling planner, the Effort Informed Trees (EIT*), by capitalizing on often-overlooked information in invalid vertices. It incorporates principles of physical force, particularly Coulomb's law. This approach proposes the elliptical $k$-nearest neighbors search method, enabling fast convergence navigation and avoiding high solution cost or infeasible paths by exploring more problem-specific search-worthy areas. It demonstrates benefits in search efficiency and cost reduction, particularly in confined, high-dimensional environments. It can be viewed as an extension of nearest neighbors search techniques. Fusing invalid vertex data with physical dynamics facilitates force-direction-based search regions, resulting in an improved convergence rate to the optimum. FDIT* outperforms existing single-query, sampling-based planners on the tested problems in R^4 to R^16 and has been demonstrated on a real-world mobile manipulation task. 

**Abstract (ZH)**: 基于力方向的树（FDIT*）采样路径规划算法 

---
# Autonomous Aerial Manipulation at Arbitrary Pose in SE(3) with Robust Control and Whole-body Planning 

**Title (ZH)**: SE(3)中任意姿态自主 aerial 操作及具备鲁棒控制与全身规划的方法 

**Authors**: Dongjae Lee, Byeongjun Kim, H. Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.19608)  

**Abstract**: Aerial manipulators based on conventional multirotors can conduct manipulation only in small roll and pitch angles due to the underactuatedness of the multirotor base. If the multirotor base is capable of hovering at arbitrary orientation, the robot can freely locate itself at any point in $\mathsf{SE}(3)$, significantly extending its manipulation workspace and enabling a manipulation task that was originally not viable. In this work, we present a geometric robust control and whole-body motion planning framework for an omnidirectional aerial manipulator (OAM). To maximize the strength of OAM, we first propose a geometric robust controller for a floating base. Since the motion of the robotic arm and the interaction forces during manipulation affect the stability of the floating base, the base should be capable of mitigating these adverse effects while controlling its 6D pose. We then design a two-step optimization-based whole-body motion planner, jointly considering the pose of the floating base and the joint angles of the robotic arm to harness the entire configuration space. The devised two-step approach facilitates real-time applicability and enhances convergence of the optimization problem with non-convex and non-Euclidean search space. The proposed approach enables the base to be stationary at any 6D pose while autonomously carrying out sophisticated manipulation near obstacles without any collision. We demonstrate the effectiveness of the proposed framework through experiments in which an OAM performs grasping and pulling of an object in multiple scenarios, including near $90^\circ$ and even $180^\circ$ pitch angles. 

**Abstract (ZH)**: 基于传统多旋翼的空中 manipulator 由于多旋翼底座的欠驱动特性，只能在小滚转和俯仰角范围内执行操作。如果多旋翼底座能够以任意姿态悬停，机器人可以自由地在 $\mathsf{SE}(3)$ 中定位自己，显著扩展其操作空间，并使原本无法实现的操作任务变得可行。在本文中，我们提出了一个用于全向空中 manipulator（OAM）的几何鲁棒控制和全身运动规划框架。为了最大化 OAM 的效能，我们首先提出了一种用于浮动底座的几何鲁棒控制器，由于在操作过程中机械臂的运动和相互作用力会影响浮动底座的稳定性，因此底座应能够减轻这些负面影响并控制其6维姿态。然后，我们设计了一种基于两步优化的整体运动规划方法，同时考虑浮动底座的姿态和机械臂的关节角度，以利用整个配置空间。采用的两步方法促进了实时应用并增强了优化问题在非凸和非欧几里得搜索空间中的收敛性。所提出的方法使得底座能够在任何6维姿态下保持静止，同时自主地在障碍物附近执行复杂的操作而不发生碰撞。通过实验展示了所提出的框架的有效性，实验中，OAM 在多种场景下执行对象的抓取和拉取操作，包括近90°和甚至180°的俯仰角。 

---
# Gentle Object Retraction in Dense Clutter Using Multimodal Force Sensing and Imitation Learning 

**Title (ZH)**: 在稠密杂乱环境中使用多模态力感知和模仿学习的温柔物体拾取 

**Authors**: Dane Brouwer, Joshua Citron, Heather Nolte, Jeannette Bohg, Mark Cutkosky  

**Link**: [PDF](https://arxiv.org/pdf/2508.19476)  

**Abstract**: Dense collections of movable objects are common in everyday spaces -- from cabinets in a home to shelves in a warehouse. Safely retracting objects from such collections is difficult for robots, yet people do it easily, using non-prehensile tactile sensing on the sides and backs of their hands and arms. We investigate the role of such sensing for training robots to gently reach into constrained clutter and extract objects. The available sensing modalities are (1) "eye-in-hand" vision, (2) proprioception, (3) non-prehensile triaxial tactile sensing, (4) contact wrenches estimated from joint torques, and (5) a measure of successful object acquisition obtained by monitoring the vacuum line of a suction cup. We use imitation learning to train policies from a set of demonstrations on randomly generated scenes, then conduct an ablation study of wrench and tactile information. We evaluate each policy's performance across 40 unseen environment configurations. Policies employing any force sensing show fewer excessive force failures, an increased overall success rate, and faster completion times. The best performance is achieved using both tactile and wrench information, producing an 80% improvement above the baseline without force information. 

**Abstract (ZH)**: 密集摆放可移动物体的集合在日常空间中普遍存在——从家庭橱柜到仓库的货架。机器人安全地从这类集合中回收物体是困难的，但人们可以轻松地完成这一任务，使用非抓握触觉感知来感知手和手臂的侧面和背面。我们研究此类感知在训练机器人轻轻进入受限杂乱环境并提取物体中的作用。可用的感知模态包括：（1）手持视觉，（2）本体感受，（3）非抓握三轴触觉感知，（4）从关节扭矩估算的接触力矩，以及（5）通过监测真空吸盘的真空管线获得的成功抓取物体的度量。我们使用模仿学习从随机生成的场景的演示集中训练策略，然后进行力矩和触觉信息的消融研究。我们评估每个策略在40个未见过的环境配置中的性能。任何使用力感知的策略都表现出较少的过度力失败、较高的整体成功率和更快的完成时间。同时使用触觉和力矩信息的策略实现了基线无力感知策略80%的性能改进。 

---
# FlipWalker: Jacob's Ladder toy-inspired robot for locomotion across diverse, complex terrain 

**Title (ZH)**: FlipWalker: 依据雅各布的梯子玩具设计的适用于多样化复杂地形移动的翻转行走机器人 

**Authors**: Diancheng Li, Nia Ralston, Bastiaan Hagen, Phoebe Tan, Matthew A. Robertson  

**Link**: [PDF](https://arxiv.org/pdf/2508.19380)  

**Abstract**: This paper introduces FlipWalker, a novel underactuated robot locomotion system inspired by Jacob's Ladder illusion toy, designed to traverse challenging terrains where wheeled robots often struggle. Like the Jacob's Ladder toy, FlipWalker features two interconnected segments joined by flexible cables, enabling it to pivot and flip around singularities in a manner reminiscent of the toy's cascading motion. Actuation is provided by motor-driven legs within each segment that push off either the ground or the opposing segment, depending on the robot's current configuration. A physics-based model of the underactuated flipping dynamics is formulated to elucidate the critical design parameters governing forward motion and obstacle clearance or climbing. The untethered prototype weighs 0.78 kg, achieves a maximum flipping speed of 0.2 body lengths per second. Experimental trials on artificial grass, river rocks, and snow demonstrate that FlipWalker's flipping strategy, which relies on ground reaction forces applied normal to the surface, offers a promising alternative to traditional locomotion for navigating irregular outdoor terrain. 

**Abstract (ZH)**: FlipWalker：一种受Jacob's Ladder幻觉玩具启发的新型欠驱动轮式机器人平衡行走系统及其在复杂地形中的应用 

---
