# CRESSim-MPM: A Material Point Method Library for Surgical Soft Body Simulation with Cutting and Suturing 

**Title (ZH)**: CRESSim-MPM: 一种用于手术软组织模拟（包括切割和缝合）的物料点方法库 

**Authors**: Yafei Ou, Mahdi Tavakoli  

**Link**: [PDF](https://arxiv.org/pdf/2502.18437)  

**Abstract**: A number of recent studies have focused on developing surgical simulation platforms to train machine learning (ML) agents or models with synthetic data for surgical assistance. While existing platforms excel at tasks such as rigid body manipulation and soft body deformation, they struggle to simulate more complex soft body behaviors like cutting and suturing. A key challenge lies in modeling soft body fracture and splitting using the finite-element method (FEM), which is the predominant approach in current platforms. Additionally, the two-way suture needle/thread contact inside a soft body is further complicated when using FEM. In this work, we use the material point method (MPM) for such challenging simulations and propose new rigid geometries and soft-rigid contact methods specifically designed for them. We introduce CRESSim-MPM, a GPU-accelerated MPM library that integrates multiple MPM solvers and incorporates surgical geometries for cutting and suturing, serving as a specialized physics engine for surgical applications. It is further integrated into Unity, requiring minimal modifications to existing projects for soft body simulation. We demonstrate the simulator's capabilities in real-time simulation of cutting and suturing on soft tissue and provide an initial performance evaluation of different MPM solvers when simulating varying numbers of particles. 

**Abstract (ZH)**: 最近的研究集中在开发手术模拟平台，利用合成数据训练机器学习代理或模型以提供手术辅助。虽然现有的平台在刚体操作和软体变形等任务上表现出色，但在模拟切割和缝合等更复杂的软体行为方面存在局限。关键挑战在于使用有限元方法（FEM）建模软体的骨折和分裂。此外，使用FEM模拟软体内缝合针/线的双向接触更为复杂。在本项工作中，我们采用物质点方法（MPM）进行此类具有挑战性的模拟，并提出专门为这些任务设计的新刚性几何形状和软-刚性接触方法。我们介绍了CRESSim-MPM，这是一个GPU加速的MPM库，集成了多个MPM求解器并包含了切开和缝合的手术几何，作为专门针对手术应用的物理引擎。该库进一步集成到Unity中，仅需少量修改即可实现现有项目的软体模拟。我们展示了模拟器在软组织切割和缝合的实时模拟能力，并提供了不同MPM求解器在模拟不同粒子数量时的初步性能评估。 

---
# Retrieval Dexterity: Efficient Object Retrieval in Clutters with Dexterous Hand 

**Title (ZH)**: 灵巧手指在杂乱环境中高效物体检索 

**Authors**: Fengshuo Bai, Yu Li, Jie Chu, Tawei Chou, Runchuan Zhu, Ying Wen, Yaodong Yang, Yuanpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18423)  

**Abstract**: Retrieving objects buried beneath multiple objects is not only challenging but also time-consuming. Performing manipulation in such environments presents significant difficulty due to complex contact relationships. Existing methods typically address this task by sequentially grasping and removing each occluding object, resulting in lengthy execution times and requiring impractical grasping capabilities for every occluding object. In this paper, we present a dexterous arm-hand system for efficient object retrieval in multi-object stacked environments. Our approach leverages large-scale parallel reinforcement learning within diverse and carefully designed cluttered environments to train policies. These policies demonstrate emergent manipulation skills (e.g., pushing, stirring, and poking) that efficiently clear occluding objects to expose sufficient surface area of the target object. We conduct extensive evaluations across a set of over 10 household objects in diverse clutter configurations, demonstrating superior retrieval performance and efficiency for both trained and unseen objects. Furthermore, we successfully transfer the learned policies to a real-world dexterous multi-fingered robot system, validating their practical applicability in real-world scenarios. Videos can be found on our project website this https URL. 

**Abstract (ZH)**: 埋藏在多重物体下的物体检索不仅具有挑战性且耗时，此类环境中的操作由于复杂的接触关系而极具难度。现有方法通常通过依次抓取并移除每个遮挡物体来解决此任务，导致执行时间较长，并且对每一个遮挡物体都要求不切实际的抓取能力。在这篇论文中，我们提出了一种灵巧的手臂-手系统，用于多物体堆积环境下的高效物体检索。我们的方法利用大规模并行强化学习在多样且精心设计的杂乱环境中进行训练，以训练策略。这些策略展示了 Emergent 操作技能（例如推、搅拌和戳），能够有效地清除遮挡物体以暴露目标物体的足够表面积。我们在超过 10 种不同家庭用品在多种杂乱配置中的广泛评估中，展示了对已训练和未见过的物体都有更好的检索性能和效率。此外，我们成功将所学策略转移到真实世界的多指灵巧机器人系统中，验证了其在实际应用场景中的实用适用性。更多信息请参见我们的项目网站：<https://>。 

---
# Stretchable Capacitive and Resistive Strain Sensors: Accessible Manufacturing Using Direct Ink Writing 

**Title (ZH)**: 可拉伸电容式和电阻式应变传感器：直接墨水书写实现简便制造 

**Authors**: Lukas Cha, Sonja Groß, Shuai Mao, Tim Braun, Sami Haddadin, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2502.18363)  

**Abstract**: As robotics advances toward integrating soft structures, anthropomorphic shapes, and complex tasks, soft and highly stretchable mechanotransducers are becoming essential. To reliably measure tactile and proprioceptive data while ensuring shape conformability, stretchability, and adaptability, researchers have explored diverse transduction principles alongside scalable and versatile manufacturing techniques. Nonetheless, many current methods for stretchable sensors are designed to produce a single sensor configuration, thereby limiting design flexibility. Here, we present an accessible, flexible, printing-based fabrication approach for customizable, stretchable sensors. Our method employs a custom-built printhead integrated with a commercial 3D printer to enable direct ink writing (DIW) of conductive ink onto cured silicone substrates. A layer-wise fabrication process, facilitated by stackable trays, allows for the deposition of multiple liquid conductive ink layers within a silicone matrix. To demonstrate the method's capacity for high design flexibility, we fabricate and evaluate both capacitive and resistive strain sensor morphologies. Experimental characterization showed that the capacitive strain sensor possesses high linearity (R^2 = 0.99), high sensitivity near the 1.0 theoretical limit (GF = 0.95), minimal hysteresis (DH = 1.36%), and large stretchability (550%), comparable to state-of-the-art stretchable strain sensors reported in the literature. 

**Abstract (ZH)**: 随着机器人技术向柔性结构、类人形态和复杂任务的集成发展，可穿戴的高延展性力觉和本体感觉传感器变得至关重要。为了在确保形状适应性、延展性和可调适性的同时可靠地测量触觉和本体感觉数据，研究人员探索了多种转换原理，并结合了可扩展和多功能的制造技术。尽管如此，许多现有的可延展传感器方法仍然设计为生成单一的传感器配置，限制了设计的灵活性。在这里，我们提出了一种可访问的、灵活的、基于打印的可定制可延展传感器的制造方法。该方法结合了定制打印头和商用3D打印机，以实现导电墨水的直接墨水书写（DIW）印制到固化的硅胶基板上。通过使用可堆叠的托盘进行逐层制造过程，可以在硅胶矩阵中沉积多层液体导电墨水。为了展示该方法的高度设计灵活性，我们制造并评估了电容式和电阻式应变传感器几何结构。实验表征表明，电容式应变传感器具有高线性度（R² = 0.99）、接近1.0理论极限的高灵敏度（GF = 0.95）、最小的滞回特性（DH = 1.36%）和大延展性（550%），这些性能与文献中报道的最先进的可延展应变传感器相当。 

---
# Pre-Surgical Planner for Robot-Assisted Vitreoretinal Surgery: Integrating Eye Posture, Robot Position and Insertion Point 

**Title (ZH)**: 机器人辅助玻璃体视网膜手术的术前规划系统：整合眼球姿势、机器人位置和穿刺点 

**Authors**: Satoshi Inagaki, Alireza Alikhani, Nassir Navab, Peter C. Issa, M. Ali Nasseri  

**Link**: [PDF](https://arxiv.org/pdf/2502.18230)  

**Abstract**: Several robotic frameworks have been recently developed to assist ophthalmic surgeons in performing complex vitreoretinal procedures such as subretinal injection of advanced therapeutics. These surgical robots show promising capabilities; however, most of them have to limit their working volume to achieve maximum accuracy. Moreover, the visible area seen through the surgical microscope is limited and solely depends on the eye posture. If the eye posture, trocar position, and robot configuration are not correctly arranged, the instrument may not reach the target position, and the preparation will have to be redone. Therefore, this paper proposes the optimization framework of the eye tilting and the robot positioning to reach various target areas for different patients. Our method was validated with an adjustable phantom eye model, and the error of this workflow was 0.13 +/- 1.65 deg (rotational joint around Y axis), -1.40 +/- 1.13 deg (around X axis), and 1.80 +/- 1.51 mm (depth, Z). The potential error sources are also analyzed in the discussion section. 

**Abstract (ZH)**: 眼科手术机器人框架优化以实现复杂玻璃体视网膜手术中的目标区域精准定位 

---
# iTrash: Incentivized Token Rewards for Automated Sorting and Handling 

**Title (ZH)**: iTrash: 基于激励代币奖励的自动分类与处理 

**Authors**: Pablo Ortega, Eduardo Castelló Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2502.18161)  

**Abstract**: As robotic systems (RS) become more autonomous, they are becoming increasingly used in small spaces and offices to automate tasks such as cleaning, infrastructure maintenance, or resource management. In this paper, we propose iTrash, an intelligent trashcan that aims to improve recycling rates in small office spaces. For that, we ran a 5 day experiment and found that iTrash can produce an efficiency increase of more than 30% compared to traditional trashcans. The findings derived from this work, point to the fact that using iTrash not only increase recyclying rates, but also provides valuable data such as users behaviour or bin usage patterns, which cannot be taken from a normal trashcan. This information can be used to predict and optimize some tasks in these spaces. Finally, we explored the potential of using blockchain technology to create economic incentives for recycling, following a Save-as-you-Throw (SAYT) model. 

**Abstract (ZH)**: 随着机器人系统（RS）变得更加自主，它们在小空间和办公室中越来越多地被用于自动化诸如清洁、基础设施维护或资源管理等任务。本文提出了一种智能垃圾桶iTrash，旨在改善小办公室空间的回收率。为此，我们进行了为期五天的实验，并发现iTrash相比传统垃圾桶的效率提高了超过30%。本研究的发现表明，使用iTrash不仅能提高回收率，还能提供诸如用户行为或垃圾桶使用模式等有价值的数据，这些数据普通垃圾桶无法提供。这些信息可用于预测和优化这些空间中的某些任务。最后，我们探讨了使用区块链技术以节省即丢弃（SAYT）模型创建回收经济激励的潜力。 

---
# A Real-time Spatio-Temporal Trajectory Planner for Autonomous Vehicles with Semantic Graph Optimization 

**Title (ZH)**: 基于语义图优化的自主车辆实时时空轨迹规划者 

**Authors**: Shan He, Yalong Ma, Tao Song, Yongzhi Jiang, Xinkai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18151)  

**Abstract**: Planning a safe and feasible trajectory for autonomous vehicles in real-time by fully utilizing perceptual information in complex urban environments is challenging. In this paper, we propose a spatio-temporal trajectory planning method based on graph optimization. It efficiently extracts the multi-modal information of the perception module by constructing a semantic spatio-temporal map through separation processing of static and dynamic obstacles, and then quickly generates feasible trajectories via sparse graph optimization based on a semantic spatio-temporal hypergraph. Extensive experiments have proven that the proposed method can effectively handle complex urban public road scenarios and perform in real time. We will also release our codes to accommodate benchmarking for the research community 

**Abstract (ZH)**: 基于图优化的时空轨迹规划方法：在复杂城市环境中实时规划安全可行的自主车辆轨迹 

---
# Enhancing Reusability of Learned Skills for Robot Manipulation via Gaze and Bottleneck 

**Title (ZH)**: 通过凝视和瓶颈增强机器人操作中学习技能的可重用性 

**Authors**: Ryo Takizawa, Izumi Karino, Koki Nakagawa, Yoshiyuki Ohmura, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18121)  

**Abstract**: Autonomous agents capable of diverse object manipulations should be able to acquire a wide range of manipulation skills with high reusability. Although advances in deep learning have made it increasingly feasible to replicate the dexterity of human teleoperation in robots, generalizing these acquired skills to previously unseen scenarios remains a significant challenge. In this study, we propose a novel algorithm, Gaze-based Bottleneck-aware Robot Manipulation (GazeBot), which enables high reusability of the learned motions even when the object positions and end-effector poses differ from those in the provided demonstrations. By leveraging gaze information and motion bottlenecks, both crucial features for object manipulation, GazeBot achieves high generalization performance compared with state-of-the-art imitation learning methods, without sacrificing its dexterity and reactivity. Furthermore, the training process of GazeBot is entirely data-driven once a demonstration dataset with gaze data is provided. Videos and code are available at this https URL. 

**Abstract (ZH)**: 能够执行多样化物体操作的自主代理应该能够获得广泛且高度可重用的操纵技能。尽管深度学习的进步使得在机器人上复制人类远程操作的灵巧性变得越来越可行，但将这些学到的技能推广到以前未见过的场景中仍然是一项重大挑战。在本研究中，我们提出了一种新型算法——基于视线的瓶颈感知机器人操纵（GazeBot），该算法能够在物体位置和末端执行器姿态与提供的示范不同的情况下，仍然实现高可重用性。通过利用视线信息和操纵瓶颈，GazeBot 在与最先进的模拟学习方法相比时，其泛化性能更高，同时不牺牲其灵巧性和反应性。此外，一旦提供了包含视线数据的示范数据集，GazeBot 的训练过程完全是数据驱动的。更多信息和代码请访问这个网址。 

---
# MRBTP: Efficient Multi-Robot Behavior Tree Planning and Collaboration 

**Title (ZH)**: MRBTP：高效的多机器人行为树规划与协作 

**Authors**: Yishuai Cai, Xinglin Chen, Zhongxuan Cai, Yunxin Mao, Minglong Li, Wenjing Yang, Ji Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18072)  

**Abstract**: Multi-robot task planning and collaboration are critical challenges in robotics. While Behavior Trees (BTs) have been established as a popular control architecture and are plannable for a single robot, the development of effective multi-robot BT planning algorithms remains challenging due to the complexity of coordinating diverse action spaces. We propose the Multi-Robot Behavior Tree Planning (MRBTP) algorithm, with theoretical guarantees of both soundness and completeness. MRBTP features cross-tree expansion to coordinate heterogeneous actions across different BTs to achieve the team's goal. For homogeneous actions, we retain backup structures among BTs to ensure robustness and prevent redundant execution through intention sharing. While MRBTP is capable of generating BTs for both homogeneous and heterogeneous robot teams, its efficiency can be further improved. We then propose an optional plugin for MRBTP when Large Language Models (LLMs) are available to reason goal-related actions for each robot. These relevant actions can be pre-planned to form long-horizon subtrees, significantly enhancing the planning speed and collaboration efficiency of MRBTP. We evaluate our algorithm in warehouse management and everyday service scenarios. Results demonstrate MRBTP's robustness and execution efficiency under varying settings, as well as the ability of the pre-trained LLM to generate effective task-specific subtrees for MRBTP. 

**Abstract (ZH)**: 多机器人任务规划与协作是机器人领域中的关键挑战。多机器人行为树规划算法（MRBTP）提供了soundness和completeness的理论保证，通过跨树扩展协调不同行为树中的异构行动以实现团队目标。对于同构行动，保留行为树间的backup结构以确保鲁棒性并通过意图共享防止冗余执行。尽管MRBTP能够生成适用于同构和异构机器人团队的行为树，其效率仍可进一步提高。当大型语言模型（LLMs）可用时，我们提出了一种可选插件以推断每个机器人的相关行动，这些行动可以通过预规划形成长期 horizons 的子树，显著增强MRBTP的规划速度和协作效率。我们在仓库管理和日常服务场景中评估了该算法。结果表明，MRBTP在不同设置下表现出鲁棒性和执行效率，并展示了预训练LLM为MRBTP生成有效任务特定子树的能力。 

---
# Ordered Genetic Algorithm for Entrance Dependent Vehicle Routing Problem in Farms 

**Title (ZH)**: 基于入口依赖的农场车辆路径问题的有序遗传算法 

**Authors**: Haotian Xu, Xiaohui Fan, Jialin Zhu, Qing Zhuo, Tao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18062)  

**Abstract**: Vehicle Routing Problems (VRP) are widely studied issues that play important roles in many production scenarios. We have noticed that in some practical scenarios of VRP, the size of cities and their entrances can significantly influence the optimization process. To address this, we have constructed the Entrance Dependent VRP (EDVRP) to describe such problems. We provide a mathematical formulation for the EDVRP in farms and propose an Ordered Genetic Algorithm (OGA) to solve it. The effectiveness of OGA is demonstrated through our experiments, which involve a multitude of randomly generated cases. The results indicate that OGA offers certain advantages compared to a random strategy baseline and a genetic algorithm without ordering. Furthermore, the novel operators introduced in this paper have been validated through ablation experiments, proving their effectiveness in enhancing the performance of the algorithm. 

**Abstract (ZH)**: 入口依赖车辆路线问题及其有序遗传算法求解 

---
# S-Graphs 2.0 -- A Hierarchical-Semantic Optimization and Loop Closure for SLAM 

**Title (ZH)**: S-Graphs 2.0 — 基于层次语义优化和闭环检测的SLAM 

**Authors**: Hriday Bavle, Jose Luis Sanchez-Lopez, Muhammad Shaheer, Javier Civera, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2502.18044)  

**Abstract**: Works based on localization and mapping do not exploit the inherent semantic-relational information from the environment for faster and efficient management and optimization of the robot poses and its map elements, often leading to pose and map inaccuracies and computational inefficiencies in large scale environments. 3D scene graph representations which distributes the environment in an hierarchical manner can be exploited to enhance the management/optimization of underlying robot poses and its map.
In this direction, we present our work Situational Graphs 2.0, which leverages the hierarchical structure of indoor scenes for efficient data management and optimization. Our algorithm begins by constructing a situational graph that organizes the environment into four layers: Keyframes, Walls, Rooms, and Floors. Our first novelty lies in the front-end which includes a floor detection module capable of identifying stairways and assigning a floor-level semantic-relations to the underlying layers. This floor-level semantic enables a floor-based loop closure strategy, rejecting false-positive loop closures in visually similar areas on different floors. Our second novelty is in exploiting the hierarchy for an improved optimization. It consists of: (1) local optimization, optimizing a window of recent keyframes and their connected components, (2) floor-global optimization, which focuses only on keyframes and their connections within the current floor during loop closures, and (3) room-local optimization, marginalizing redundant keyframes that share observations within the room.
We validate our algorithm extensively in different real multi-floor environments. Our approach can demonstrate state-of-art-art results in large scale multi-floor environments creating hierarchical maps while bounding the computational complexity where several baseline works fail to execute efficiently. 

**Abstract (ZH)**: 基于局部化和建图的工作未能充分利用环境中的固有语义关系信息，以实现更快更高效的机器人姿态管理和地图元素优化，常导致大型环境中姿态和地图的不准确性及计算效率低下。可以通过层次化的3D场景图表示来增强底层机器人姿态及其地图的管理和优化。

Situational Graphs 2.0：利用室内场景的层次结构进行高效的数据管理和优化 

---
# From planning to policy: distilling $\texttt{Skill-RRT}$ for long-horizon prehensile and non-prehensile manipulation 

**Title (ZH)**: 从规划到政策：提炼Skill-RRT用于长期 horizon 的抓取和非抓取操作 manipulation 

**Authors**: Haewon Jung, Donguk Lee, Haecheol Park, JunHyeop Kim, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18015)  

**Abstract**: Current robots face challenges in manipulation tasks that require a long sequence of prehensile and non-prehensile skills. This involves handling contact-rich interactions and chaining multiple skills while considering their long-term consequences. This paper presents a framework that leverages imitation learning to distill a planning algorithm, capable of solving long-horizon problems but requiring extensive computation time, into a policy for efficient action inference. We introduce $\texttt{Skill-RRT}$, an extension of the rapidly-exploring random tree (RRT) that incorporates skill applicability checks and intermediate object pose sampling for efficient long-horizon planning. To enable skill chaining, we propose $\textit{connectors}$, goal-conditioned policies that transition between skills while minimizing object disturbance. Using lazy planning, connectors are selectively trained on relevant transitions, reducing the cost of training. High-quality demonstrations are generated with $\texttt{Skill-RRT}$ and refined by a noise-based replay mechanism to ensure robust policy performance. The distilled policy, trained entirely in simulation, zero-shot transfer to the real world, and achieves over 80% success rates across three challenging manipulation tasks. In simulation, our approach outperforms the state-of-the-art skill-based reinforcement learning method, $\texttt{MAPLE}$, and $\texttt{Skill-RRT}$. 

**Abstract (ZH)**: 当前的机器人在执行需要长时间序列的抓持和非抓持技能的操作任务时面临挑战。这涉及到处理丰富的接触交互，并在考虑其长期后果的同时串联多种技能。本文提出了一种框架，利用模仿学习将能够解决长期问题但需要大量计算时间的规划算法提炼为一种高效的策略。我们引入了$\texttt{Skill-RRT}$，这是一种扩展的快速扩展随机树（RRT），结合了技能适用性检查和中间物体姿态采样，以实现高效的长期规划。为了实现技能串联，我们提出了连接器（$\textit{connectors}$），这是一种条件于目标的策略，可在最小化物体扰动的情况下在技能之间进行过渡。通过懒规划，连接器仅在相关过渡上进行训练，从而降低训练成本。高质量的演示由$\texttt{Skill-RRT}$生成，并通过基于噪声的重放机制进一步优化，以确保策略性能的稳健。该提炼出的策略完全在仿真中训练，并实现了一种零样本的现实世界转移，在三项具有挑战性的操作任务中成功率超过80%。在仿真中，我们的方法在技能基强化学习方法$\texttt{MAPLE}$和$\texttt{Skill-RRT}$的基础上表现出色。 

---
# Multimodal Interaction and Intention Communication for Industrial Robots 

**Title (ZH)**: 工业机器人多模态交互与意图通信 

**Authors**: Tim Schreiter, Andrey Rudenko, Jens V. Rüppel, Martin Magnusson, Achim J. Lilienthal  

**Link**: [PDF](https://arxiv.org/pdf/2502.17971)  

**Abstract**: Successful adoption of industrial robots will strongly depend on their ability to safely and efficiently operate in human environments, engage in natural communication, understand their users, and express intentions intuitively while avoiding unnecessary distractions. To achieve this advanced level of Human-Robot Interaction (HRI), robots need to acquire and incorporate knowledge of their users' tasks and environment and adopt multimodal communication approaches with expressive cues that combine speech, movement, gazes, and other modalities. This paper presents several methods to design, enhance, and evaluate expressive HRI systems for non-humanoid industrial robots. We present the concept of a small anthropomorphic robot communicating as a proxy for its non-humanoid host, such as a forklift. We developed a multimodal and LLM-enhanced communication framework for this robot and evaluated it in several lab experiments, using gaze tracking and motion capture to quantify how users perceive the robot and measure the task progress. 

**Abstract (ZH)**: 工业机器人在人类环境中的成功应用将强烈依赖于其安全高效地操作、自然沟通、理解用户以及以直观方式表达意图的能力，同时避免不必要的干扰。为了实现这一高级水平的人机交互（HRI），机器人需要获取并整合用户任务和环境的知识，并采用结合语音、动作、目光等多种模态的表达性交互方法。本文提出了几种设计、增强和评估表达性HRI系统的办法。我们介绍了作为非类人工业机器人代理的小型拟人化机器人概念，并开发了一种多模态和大语言模型增强的通信框架，通过实验评估了该框架，使用眼动追踪和运动捕捉量化用户对机器人的感知并测量任务进度。 

---
# Quadrotor Neural Dead Reckoning in Periodic Trajectories 

**Title (ZH)**: 四旋翼神经动力学航位推算在周期性轨迹中 

**Authors**: Shira Massas, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2502.17964)  

**Abstract**: In real world scenarios, due to environmental or hardware constraints, the quadrotor is forced to navigate in pure inertial navigation mode while operating indoors or outdoors. To mitigate inertial drift, end-to-end neural network approaches combined with quadrotor periodic trajectories were suggested. There, the quadrotor distance is regressed and combined with inertial model-based heading estimation, the quadrotor position vector is estimated. To further enhance positioning performance, in this paper we propose a quadrotor neural dead reckoning approach for quadrotors flying on periodic trajectories. In this case, the inertial readings are fed into a simple and efficient network to directly estimate the quadrotor position vector. Our approach was evaluated on two different quadrotors, one operating indoors while the other outdoors. Our approach improves the positioning accuracy of other deep-learning approaches, achieving an average 27% reduction in error outdoors and an average 79% reduction indoors, while requiring only software modifications. With the improved positioning accuracy achieved by our method, the quadrotor can seamlessly perform its tasks. 

**Abstract (ZH)**: 在实际场景中，由于环境或硬件限制，旋翼无人机被迫在室内或室外以纯惯性导航模式导航。为了减轻惯性漂移，建议使用端到端神经网络方法结合旋翼无人机的周期性轨迹。在那里，旋翼无人机的距离被回归并与基于惯性模型的方向估计结合，估计旋翼无人机的位置向量。为进一步增强定位性能，本文提出了一种适用于沿周期性轨迹飞行的旋翼无人机的神经死 reckoning 方法。在这种情况下，惯性读数被输入到一个简单且高效的网络中，直接估计旋翼无人机的位置向量。我们的方法在两种不同的旋翼无人机上进行了评估，一种在室内操作，另一种在室外操作。与其它深度学习方法相比，我们的方法提高了定位精度，室外平均误差减少了27%，室内平均误差减少了79%，仅需软件修改。通过我们方法实现的改进的定位精度，旋翼无人机可以无缝执行其任务。 

---
# InVDriver: Intra-Instance Aware Vectorized Query-Based Autonomous Driving Transformer 

**Title (ZH)**: InVDriver: Awareness of 内存实例向量查询驱动自主驾驶变换器 

**Authors**: Bo Zhang, Heye Huang, Chunyang Liu, Yaqin Zhang, Zhenhua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17949)  

**Abstract**: End-to-end autonomous driving with its holistic optimization capabilities, has gained increasing traction in academia and industry. Vectorized representations, which preserve instance-level topological information while reducing computational overhead, have emerged as a promising paradigm. While existing vectorized query-based frameworks often overlook the inherent spatial correlations among intra-instance points, resulting in geometrically inconsistent outputs (e.g., fragmented HD map elements or oscillatory trajectories). To address these limitations, we propose InVDriver, a novel vectorized query-based system that systematically models intra-instance spatial dependencies through masked self-attention layers, thereby enhancing planning accuracy and trajectory smoothness. Across all core modules, i.e., perception, prediction, and planning, InVDriver incorporates masked self-attention mechanisms that restrict attention to intra-instance point interactions, enabling coordinated refinement of structural elements while suppressing irrelevant inter-instance noise. Experimental results on the nuScenes benchmark demonstrate that InVDriver achieves state-of-the-art performance, surpassing prior methods in both accuracy and safety, while maintaining high computational efficiency. Our work validates that explicit modeling of intra-instance geometric coherence is critical for advancing vectorized autonomous driving systems, bridging the gap between theoretical advantages of end-to-end frameworks and practical deployment requirements. 

**Abstract (ZH)**: 端到端自主驾驶系统通过其整体优化能力，在学术界和工业界获得了越来越多的关注。矢量化表示在保留实例级拓扑信息的同时减少计算开销，已成为一个有前途的范式。尽管现有的基于矢量化查询的框架往往忽视了实例内部点之间的固有空间相关性，导致几何不一致的输出（例如，断裂的高精度地图元素或振荡轨迹）。为了解决这些限制，我们提出了一种新颖的基于矢量化查询的系统InVDriver，该系统通过掩码自注意力层系统地建模实例内部的空间依赖性，从而提高规划准确性和轨迹平滑度。在感知、预测和规划的所有核心模块中，InVDriver 集成了掩码自注意力机制，限制注意力仅关注实例内部点之间的交互，从而在抑制无关实例间噪声的同时实现结构元素的协调精化。在nuScenes基准测试上的实验结果表明，InVDriver 达到了最先进的性能，在准确性和安全性方面均超过了先前的方法，同时保持了高效的计算效率。我们的工作验证了明确建模实例内部几何一致性对于推动矢量化自主驾驶系统的发展至关重要，弥合了端到端框架理论优势与实际部署需求之间的差距。 

---
# FetchBot: Object Fetching in Cluttered Shelves via Zero-Shot Sim2Real 

**Title (ZH)**: FetchBot: 在杂乱货架上进行零样本Sim2Real对象抓取 

**Authors**: Weiheng Liu, Yuxuan Wan, Jilong Wang, Yuxuan Kuang, Xuesong Shi, Haoran Li, Dongbin Zhao, Zhizheng Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17894)  

**Abstract**: Object fetching from cluttered shelves is an important capability for robots to assist humans in real-world scenarios. Achieving this task demands robotic behaviors that prioritize safety by minimizing disturbances to surrounding objects, an essential but highly challenging requirement due to restricted motion space, limited fields of view, and complex object dynamics. In this paper, we introduce FetchBot, a sim-to-real framework designed to enable zero-shot generalizable and safety-aware object fetching from cluttered shelves in real-world settings. To address data scarcity, we propose an efficient voxel-based method for generating diverse simulated cluttered shelf scenes at scale and train a dynamics-aware reinforcement learning (RL) policy to generate object fetching trajectories within these scenes. This RL policy, which leverages oracle information, is subsequently distilled into a vision-based policy for real-world deployment. Considering that sim-to-real discrepancies stem from texture variations mostly while from geometric dimensions rarely, we propose to adopt depth information estimated by full-fledged depth foundation models as the input for the vision-based policy to mitigate sim-to-real gap. To tackle the challenge of limited views, we design a novel architecture for learning multi-view representations, allowing for comprehensive encoding of cluttered shelf scenes. This enables FetchBot to effectively minimize collisions while fetching objects from varying positions and depths, ensuring robust and safety-aware operation. Both simulation and real-robot experiments demonstrate FetchBot's superior generalization ability, particularly in handling a broad range of real-world scenarios, includ 

**Abstract (ZH)**: 从杂乱货架上抓取物体是机器人在实际场景中协助人类的重要能力。实现这一任务需要机器人行为优先确保安全，通过最小化对周围物体的干扰，这一要求由于受限的运动空间、有限的视野和复杂的物体动力学而变得至关重要。本文介绍了FetchBot，一个用于实现实用场景中杂乱货架上零样本泛化和安全意识物体抓取的从仿真到现实的框架。为了解决数据稀缺问题，我们提出了一种高效的方法，用于生成大规模多样的模拟杂乱货架场景，并训练一种动力学感知的强化学习（RL）策略来在这些场景中生成物体抓取轨迹。该RL策略利用先验信息，进而被提取为基于视觉的策略以部署到现实世界。考虑到仿真实验与现实之间的差异主要源自纹理变化而非几何尺寸变化，我们提出采用由完整深度基础模型估计的深度信息作为基于视觉策略的输入，以减轻仿真实验与现实的差距。为应对有限视野的挑战，我们设计了一种新的架构以学习多视角表示，使得可以全面编码杂乱货架场景。这使FetchBot能够在不同位置和深度抓取物体时有效减少碰撞，确保稳健和安全的操作。仿真实验和真实机器人实验均证明了FetchBot在处理各种实用场景中的优越泛化能力。 

---
# corobos: A Design for Mobile Robots Enabling Cooperative Transitions between Table and Wall Surfaces 

**Title (ZH)**: Corobos: 移动机器人的一种设计，实现桌子和墙面表面之间的协作过渡 

**Authors**: Changyo Han, Yosuke Nakagawa, Takeshi Naemura  

**Link**: [PDF](https://arxiv.org/pdf/2502.17868)  

**Abstract**: Swarm User Interfaces allow dynamic arrangement of user environments through the use of multiple mobile robots, but their operational range is typically confined to a single plane due to constraints imposed by their two-wheel propulsion systems. We present corobos, a proof-of-concept design that enables these robots to cooperatively transition between table (horizontal) and wall (vertical) surfaces seamlessly, without human intervention. Each robot is equipped with a uniquely designed slope structure that facilitates smooth rotation when another robot pushes it toward a target surface. Notably, this design relies solely on passive mechanical elements, eliminating the need for additional active electrical components. We investigated the design parameters of this structure and evaluated its transition success rate through experiments. Furthermore, we demonstrate various application examples to showcase the potential of corobos in enhancing user environments. 

**Abstract (ZH)**: 基于协作过渡设计的 swarm 用户界面允许多移动机器人在水平和垂直表面之间无缝切换，无需人工干预。 

---
# Impact of Object Weight in Handovers: Inspiring Robotic Grip Release and Motion from Human Handovers 

**Title (ZH)**: 手持物体重量对切换影响：启发机器人手部释放与运动的人手切换 

**Authors**: Parag Khanna, Mårten Björkman, Christian Smith  

**Link**: [PDF](https://arxiv.org/pdf/2502.17834)  

**Abstract**: This work explores the effect of object weight on human motion and grip release during handovers to enhance the naturalness, safety, and efficiency of robot-human interactions. We introduce adaptive robotic strategies based on the analysis of human handover behavior with varying object weights. The key contributions of this work includes the development of an adaptive grip-release strategy for robots, a detailed analysis of how object weight influences human motion to guide robotic motion adaptations, and the creation of handover-datasets incorporating various object weights, including the YCB handover dataset. By aligning robotic grip release and motion with human behavior, this work aims to improve robot-human handovers for different weighted objects. We also evaluate these human-inspired adaptive robotic strategies in robot-to-human handovers to assess their effectiveness and performance and demonstrate that they outperform the baseline approaches in terms of naturalness, efficiency, and user perception. 

**Abstract (ZH)**: 本研究探讨物体重量对人类传递过程中的运动和握持释放的影响，以增强机器人与人类互动的自然性、安全性和效率。我们基于不同物体重量的人类传递行为分析，提出适应性机器人策略。本文的关键贡献包括开发适应性握持释放策略、详细分析物体重量如何影响人类运动以指导机器人的运动适应、以及创建包含不同物体重量的传递数据集（如YCB传递数据集）。通过使机器人的握持释放和运动与人类行为相一致，本研究旨在改进不同重量物体的机器人与人类之间的传递。我们还在机器人到人类的传递中评估这些受人类启发的适应性机器人策略，以评估其有效性和性能，并证明它们在自然性、效率和用户感知方面优于基准方法。 

---
# CAML: Collaborative Auxiliary Modality Learning for Multi-Agent Systems 

**Title (ZH)**: CAML：多agent系统中的协作辅助模态学习 

**Authors**: Rui Liu, Yu Shen, Peng Gao, Pratap Tokekar, Ming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.17821)  

**Abstract**: Multi-modality learning has become a crucial technique for improving the performance of machine learning applications across domains such as autonomous driving, robotics, and perception systems. While existing frameworks such as Auxiliary Modality Learning (AML) effectively utilize multiple data sources during training and enable inference with reduced modalities, they primarily operate in a single-agent context. This limitation is particularly critical in dynamic environments, such as connected autonomous vehicles (CAV), where incomplete data coverage can lead to decision-making blind spots. To address these challenges, we propose Collaborative Auxiliary Modality Learning ($\textbf{CAML}$), a novel multi-agent multi-modality framework that enables agents to collaborate and share multimodal data during training while allowing inference with reduced modalities per agent during testing. We systematically analyze the effectiveness of $\textbf{CAML}$ from the perspective of uncertainty reduction and data coverage, providing theoretical insights into its advantages over AML. Experimental results in collaborative decision-making for CAV in accident-prone scenarios demonstrate that \ours~achieves up to a ${\bf 58.13}\%$ improvement in accident detection. Additionally, we validate $\textbf{CAML}$ on real-world aerial-ground robot data for collaborative semantic segmentation, achieving up to a ${\bf 10.61}\%$ improvement in mIoU. 

**Abstract (ZH)**: 协作辅助模态学习（CAML）：一种多agent多模态框架 

---
# Safe Multi-Agent Navigation guided by Goal-Conditioned Safe Reinforcement Learning 

**Title (ZH)**: 基于目标导向的安全强化学习引导的多智能体导航 

**Authors**: Meng Feng, Viraj Parimi, Brian Williams  

**Link**: [PDF](https://arxiv.org/pdf/2502.17813)  

**Abstract**: Safe navigation is essential for autonomous systems operating in hazardous environments. Traditional planning methods excel at long-horizon tasks but rely on a predefined graph with fixed distance metrics. In contrast, safe Reinforcement Learning (RL) can learn complex behaviors without relying on manual heuristics but fails to solve long-horizon tasks, particularly in goal-conditioned and multi-agent scenarios.
In this paper, we introduce a novel method that integrates the strengths of both planning and safe RL. Our method leverages goal-conditioned RL and safe RL to learn a goal-conditioned policy for navigation while concurrently estimating cumulative distance and safety levels using learned value functions via an automated self-training algorithm. By constructing a graph with states from the replay buffer, our method prunes unsafe edges and generates a waypoint-based plan that the agent follows until reaching its goal, effectively balancing faster and safer routes over extended distances.
Utilizing this unified high-level graph and a shared low-level goal-conditioned safe RL policy, we extend this approach to address the multi-agent safe navigation problem. In particular, we leverage Conflict-Based Search (CBS) to create waypoint-based plans for multiple agents allowing for their safe navigation over extended horizons. This integration enhances the scalability of goal-conditioned safe RL in multi-agent scenarios, enabling efficient coordination among agents.
Extensive benchmarking against state-of-the-art baselines demonstrates the effectiveness of our method in achieving distance goals safely for multiple agents in complex and hazardous environments. Our code will be released to support future research. 

**Abstract (ZH)**: 安全导航对于在危险环境中操作的自主系统至关重要。传统的规划方法在长期任务方面表现出色，但依赖于预先定义的具有固定距离度量的图。相比之下，安全的强化学习（RL）可以在不依赖手动启发式的情况下学习复杂的行为，但在解决长期任务方面尤其失败，特别是在目标条件和多agent场景中。
在本文中，我们提出了一种结合规划和安全RL优点的新方法。该方法利用目标条件RL和安全RL学习导航的目标条件策略，同时通过自训练算法利用学习的价值函数估计累积距离和安全水平。通过从回放缓冲区构建图，该方法删除不安全的边，生成agent遵循直至到达目标的基于路径点的计划，从而在较远的距离上实现更快更安全的路径。
利用这一统一的高层图和共享的目标条件安全RL策略，我们将此方法扩展以解决多agent安全导航问题。具体而言，我们利用冲突基搜索（CBS）为多个agent创建基于路径点的计划，使它们能够跨越长时间安全导航。这种集成增强了目标条件安全RL在多agent场景中的可扩展性，使agent之间能够实现高效的协调。
与最先进的基线方法的广泛基准测试显示，我们的方法在复杂和危险环境中能够有效地为多agent安全地实现距离目标。我们的代码将向未来的研究开放。 

---
# Design of a Breakaway Utensil Attachment for Enhanced Safety in Robot-Assisted Feeding 

**Title (ZH)**: 机器人辅助喂食中具有增强安全性的脱手餐具附件设计 

**Authors**: Hau Wen Chang, J-Anne Yow, Lek Syn Lim, Wei Tech Ang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17774)  

**Abstract**: Robot-assisted feeding systems enhance the independence of individuals with motor impairments and alleviate caregiver burden. While existing systems predominantly rely on software-based safety features to mitigate risks during unforeseen collisions, this study explores the use of a mechanical fail-safe to improve safety. We designed a breakaway utensil attachment that decouples forces exerted by the robot on the user when excessive forces occur. Finite element analysis (FEA) simulations were performed to predict failure points under various loading conditions, followed by experimental validation using 3D-printed attachments with variations in slot depth and wall loops. To facilitate testing, a drop test rig was developed and validated. Our results demonstrated a consistent failure point at the slot of the attachment, with a slot depth of 1 mm and three wall loops achieving failure at the target force of 65 N. Additionally, the parameters can be tailored to customize the breakaway force based on user-specific factors, such as comfort and pain tolerance. CAD files and utensil assembly instructions can be found here: this https URL 

**Abstract (ZH)**: 机器人辅助进食系统增强运动障碍个体的独立性并减轻护理负担。为了提高安全性，本研究探讨了使用机械失效安全措施的应用，设计了一种断开式餐具附件，在过度力量作用时脱离机器人对使用者的施力。进行了有限元分析（FEA）仿真预测在不同负载条件下的失效点，随后使用3D打印附件并通过改变槽深和壁环的变数进行了实验验证。为便于测试，开发并验证了一个跌落试验台。结果显示，附件的槽部位为1 mm深度和三个壁环参数在目标力65 N时实现失效。此外，参数可以根据用户的特定因素（如舒适度和疼痛耐受度）进行定制。CAD文件和餐具组装说明可在此获取：this https URL。 

---
# Toward 6-DOF Autonomous Underwater Vehicle Energy-Aware Position Control based on Deep Reinforcement Learning: Preliminary Results 

**Title (ZH)**: 基于深度强化学习的6自由度自主水下车辆能量感知位置控制：初步结果 

**Authors**: Gustavo Boré, Vicente Sufán, Sebastián Rodríguez-Martínez, Giancarlo Troni  

**Link**: [PDF](https://arxiv.org/pdf/2502.17742)  

**Abstract**: The use of autonomous underwater vehicles (AUVs) for surveying, mapping, and inspecting unexplored underwater areas plays a crucial role, where maneuverability and power efficiency are key factors for extending the use of these platforms, making six degrees of freedom (6-DOF) holonomic platforms essential tools. Although Proportional-Integral-Derivative (PID) and Model Predictive Control controllers are widely used in these applications, they often require accurate system knowledge, struggle with repeatability when facing payload or configuration changes, and can be time-consuming to fine-tune. While more advanced methods based on Deep Reinforcement Learning (DRL) have been proposed, they are typically limited to operating in fewer degrees of freedom. This paper proposes a novel DRL-based approach for controlling holonomic 6-DOF AUVs using the Truncated Quantile Critics (TQC) algorithm, which does not require manual tuning and directly feeds commands to the thrusters without prior knowledge of their configuration. Furthermore, it incorporates power consumption directly into the reward function. Simulation results show that the TQC High-Performance method achieves better performance to a fine-tuned PID controller when reaching a goal point, while the TQC Energy-Aware method demonstrates slightly lower performance but consumes 30% less power on average. 

**Abstract (ZH)**: 基于Truncated Quantile Critics算法的深度强化学习控制六自由度水下自主航行器方法 

---
# The Geometry of Optimal Gait Families for Steering Kinematic Locomoting Systems 

**Title (ZH)**: 最优行进家族几何学： steering 运动学移动系统指导下的行进方式几何 

**Authors**: Jinwoo Choi, Siming Deng, Nathan Justus, Noah J. Cowan, Ross L. Hatton  

**Link**: [PDF](https://arxiv.org/pdf/2502.17672)  

**Abstract**: Motion planning for locomotion systems typically requires translating high-level rigid-body tasks into low-level joint trajectories-a process that is straightforward for car-like robots with fixed, unbounded actuation inputs but more challenging for systems like snake robots, where the mapping depends on the current configuration and is constrained by joint limits. In this paper, we focus on generating continuous families of optimal gaits-collections of gaits parameterized by step size or steering rate-to enhance controllability and maneuverability. We uncover the underlying geometric structure of these optimal gait families and propose methods for constructing them using both global and local search strategies, where the local method and the global method compensate each other. The global search approach is robust to nonsmooth behavior, albeit yielding reduced-order solutions, while the local search provides higher accuracy but can be unstable near nonsmooth regions. To demonstrate our framework, we generate optimal gait families for viscous and perfect-fluid three-link swimmers. This work lays a foundation for integrating low-level joint controllers with higher-level motion planners in complex locomotion systems. 

**Abstract (ZH)**: 基于运动规划的连续最优步态族生成及其在仿生游泳器中的应用 

---
# SET-PAiREd: Designing for Parental Involvement in Learning with an AI-Assisted Educational Robot 

**Title (ZH)**: SET-PAiREd: 设计一种基于AI辅助教育机器人的家长参与学习方法 

**Authors**: Hui-Ru Ho, Nitigya Kargeti, Ziqi Liu, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17623)  

**Abstract**: AI-assisted learning companion robots are increasingly used in early education. Many parents express concerns about content appropriateness, while they also value how AI and robots could supplement their limited skill, time, and energy to support their children's learning. We designed a card-based kit, SET, to systematically capture scenarios that have different extents of parental involvement. We developed a prototype interface, PAiREd, with a learning companion robot to deliver LLM-generated educational content that can be reviewed and revised by parents. Parents can flexibly adjust their involvement in the activity by determining what they want the robot to help with. We conducted an in-home field study involving 20 families with children aged 3-5. Our work contributes to an empirical understanding of the level of support parents with different expectations may need from AI and robots and a prototype that demonstrates an innovative interaction paradigm for flexibly including parents in supporting their children. 

**Abstract (ZH)**: AI辅助的学习伴侣机器人在幼儿教育中的应用：父母参与程度的系统化捕捉及创新交互范式的原型设计 

---
# Learning Decentralized Swarms Using Rotation Equivariant Graph Neural Networks 

**Title (ZH)**: 使用旋转等变图神经网络学习去中心化 swarm 

**Authors**: Taos Transue, Bao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17612)  

**Abstract**: The orchestration of agents to optimize a collective objective without centralized control is challenging yet crucial for applications such as controlling autonomous fleets, and surveillance and reconnaissance using sensor networks. Decentralized controller design has been inspired by self-organization found in nature, with a prominent source of inspiration being flocking; however, decentralized controllers struggle to maintain flock cohesion. The graph neural network (GNN) architecture has emerged as an indispensable machine learning tool for developing decentralized controllers capable of maintaining flock cohesion, but they fail to exploit the symmetries present in flocking dynamics, hindering their generalizability. We enforce rotation equivariance and translation invariance symmetries in decentralized flocking GNN controllers and achieve comparable flocking control with 70% less training data and 75% fewer trainable weights than existing GNN controllers without these symmetries enforced. We also show that our symmetry-aware controller generalizes better than existing GNN controllers. Code and animations are available at this http URL. 

**Abstract (ZH)**: 无需协调中心控制的代理 orchestrating 以优化集体目标在自主车队控制和传感器网络监视与侦察等应用中具有挑战性但至关重要。缺乏中心控制的控制器设计受到自然界中自我组织现象的启发，其中飞行动物群集是最显著的灵感来源之一；然而，缺乏对群集凝聚力的维持。图神经网络架构已成为开发能够维持群集凝聚力的分散控制器不可或缺的机器学习工具，但它们未能利用群集动态中存在的对称性，阻碍了其泛化能力。我们强制执行旋转等变性和平移不变性对称性，在分散群集GNN控制器中实现类似性能的群集控制，所需训练数据减少70%，可训练参数减少75%，优于未强制执行这些对称性的现有GNN控制器。我们还展示了我们的对称性感知控制器比现有GNN控制器具有更好的泛化能力。代码和动画请访问此网址。 

---
# Self-Supervised Data Generation for Precision Agriculture: Blending Simulated Environments with Real Imagery 

**Title (ZH)**: 自监督数据生成在精准农业中的应用：模拟环境与实际影像的融合 

**Authors**: Leonardo Saraceni, Ionut Marian Motoi, Daniele Nardi, Thomas Alessandro Ciarfuglia  

**Link**: [PDF](https://arxiv.org/pdf/2502.18320)  

**Abstract**: In precision agriculture, the scarcity of labeled data and significant covariate shifts pose unique challenges for training machine learning models. This scarcity is particularly problematic due to the dynamic nature of the environment and the evolving appearance of agricultural subjects as living things. We propose a novel system for generating realistic synthetic data to address these challenges. Utilizing a vineyard simulator based on the Unity engine, our system employs a cut-and-paste technique with geometrical consistency considerations to produce accurate photo-realistic images and labels from synthetic environments to train detection algorithms. This approach generates diverse data samples across various viewpoints and lighting conditions. We demonstrate considerable performance improvements in training a state-of-the-art detector by applying our method to table grapes cultivation. The combination of techniques can be easily automated, an increasingly important consideration for adoption in agricultural practice. 

**Abstract (ZH)**: 在精准农业中，标注数据的稀缺性和显著的协变量转移为训练机器学习模型带来了独特的挑战。由于环境的动态性质和农业对象作为生物体的不断演变，这种稀缺性尤为成问题。我们提出了一种新颖的系统，用于生成现实主义的合成数据以应对这些挑战。基于Unity引擎的葡萄园模拟器，我们的系统采用带有几何一致性考虑的拼接技术，从合成环境中生成准确的逼真图像和标签，用于训练检测算法。这种方法能够在多种视角和光照条件下生成多样化的数据样本。我们将该方法应用于葡萄栽培，展示了对最先进的检测器训练性能的显著改进。该方法的结合可以轻松实现自动化，对于在农业实践中采用来说越来越重要。 

---
# HEROS-GAN: Honed-Energy Regularized and Optimal Supervised GAN for Enhancing Accuracy and Range of Low-Cost Accelerometers 

**Title (ZH)**: HEROS-GAN：磨砺能量正则化和最优监督生成网络，用于提高低成本加速度计的准确度和量程范围 

**Authors**: Yifeng Wang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18064)  

**Abstract**: Low-cost accelerometers play a crucial role in modern society due to their advantages of small size, ease of integration, wearability, and mass production, making them widely applicable in automotive systems, aerospace, and wearable technology. However, this widely used sensor suffers from severe accuracy and range limitations. To this end, we propose a honed-energy regularized and optimal supervised GAN (HEROS-GAN), which transforms low-cost sensor signals into high-cost equivalents, thereby overcoming the precision and range limitations of low-cost accelerometers. Due to the lack of frame-level paired low-cost and high-cost signals for training, we propose an Optimal Transport Supervision (OTS), which leverages optimal transport theory to explore potential consistency between unpaired data, thereby maximizing supervisory information. Moreover, we propose a Modulated Laplace Energy (MLE), which injects appropriate energy into the generator to encourage it to break range limitations, enhance local changes, and enrich signal details. Given the absence of a dedicated dataset, we specifically establish a Low-cost Accelerometer Signal Enhancement Dataset (LASED) containing tens of thousands of samples, which is the first dataset serving to improve the accuracy and range of accelerometers and is released in Github. Experimental results demonstrate that a GAN combined with either OTS or MLE alone can surpass the previous signal enhancement SOTA methods by an order of magnitude. Integrating both OTS and MLE, the HEROS-GAN achieves remarkable results, which doubles the accelerometer range while reducing signal noise by two orders of magnitude, establishing a benchmark in the accelerometer signal processing. 

**Abstract (ZH)**: 一种优化运输监督和调制拉普拉斯能量正则化优化生成对抗网络在低成本加速度计信号增强中的应用 

---
# OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation 

**Title (ZH)**: OpenFly：一种多功能工具链及大规模benchmark用于航空视觉语言导航 

**Authors**: Yunpeng Gao, Chenhui Li, Zhongrui You, Junli Liu, Zhen Li, Pengan Chen, Qizhi Chen, Zhonghan Tang, Liansheng Wang, Penghui Yang, Yiwen Tang, Yuhang Tang, Shuai Liang, Songyi Zhu, Ziqin Xiong, Yifei Su, Xinyi Ye, Jianan Li, Yan Ding, Dong Wang, Zhigang Wang, Bin Zhao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18041)  

**Abstract**: Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced. 

**Abstract (ZH)**: 开放飞行：面向高空导航的多功能工具链和大规模基准平台 

---
# ConvoyLLM: Dynamic Multi-Lane Convoy Control Using LLMs 

**Title (ZH)**: ConvoyLLM：使用大规模语言模型进行动态多车道车队控制 

**Authors**: Liping Lu, Zhican He, Duanfeng Chu, Rukang Wang, Saiqian Peng, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.17529)  

**Abstract**: This paper proposes a novel method for multi-lane convoy formation control that uses large language models (LLMs) to tackle coordination challenges in dynamic highway environments. Each connected and autonomous vehicle in the convoy uses a knowledge-driven approach to make real-time adaptive decisions based on various scenarios. Our method enables vehicles to dynamically perform tasks, including obstacle avoidance, convoy joining/leaving, and escort formation switching, all while maintaining the overall convoy structure. We design a Interlaced formation control strategy based on locally dynamic distributed graphs, ensuring the convoy remains stable and flexible. We conduct extensive experiments in the SUMO simulation platform across multiple traffic scenarios, and the results demonstrate that the proposed method is effective, robust, and adaptable to dynamic environments. The code is available at: this https URL. 

**Abstract (ZH)**: 本文提出了一种使用大型语言模型（LLMs）解决动态高速公路上多车道车队编队控制协调难题的新方法。每辆连接的自动驾驶车辆使用知识驱动的方法，根据各种场景做出实时适应性决策。我们的方法使车辆能够动态执行包括障碍物避免、车队加入/离开以及护航编队切换等任务，同时保持整个车队结构的稳定。我们设计了一种基于局部动态分散图的穿插编队控制策略，确保车队保持稳定和灵活性。我们在SUMO仿真平台上对多种交通场景进行了广泛的实验，结果表明所提出的方法是有效的、 robust的，并且能够适应动态环境。代码可在以下链接获取：this https URL。 

---
