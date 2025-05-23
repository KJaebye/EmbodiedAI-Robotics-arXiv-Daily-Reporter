# SeGuE: Semantic Guided Exploration for Mobile Robots 

**Title (ZH)**: 基于语义引导的移动机器人探索方法 

**Authors**: Cody Simons, Aritra Samanta, Amit K. Roy-Chowdhury, Konstantinos Karydis  

**Link**: [PDF](https://arxiv.org/pdf/2504.03629)  

**Abstract**: The rise of embodied AI applications has enabled robots to perform complex tasks which require a sophisticated understanding of their environment. To enable successful robot operation in such settings, maps must be constructed so that they include semantic information, in addition to geometric information. In this paper, we address the novel problem of semantic exploration, whereby a mobile robot must autonomously explore an environment to fully map both its structure and the semantic appearance of features. We develop a method based on next-best-view exploration, where potential poses are scored based on the semantic features visible from that pose. We explore two alternative methods for sampling potential views and demonstrate the effectiveness of our framework in both simulation and physical experiments. Automatic creation of high-quality semantic maps can enable robots to better understand and interact with their environments and enable future embodied AI applications to be more easily deployed. 

**Abstract (ZH)**: 随着具身AI应用的兴起，机器人能够执行需要对其环境进行复杂理解的任务。为了在这些环境中成功操作机器人，必须构建包含语义信息的地图，而不仅仅是几何信息。在本文中，我们探讨了一种新颖的语义探索问题，即移动机器人必须自主探索环境以完整地映射其结构和特征的语义外观。我们开发了一种基于最佳视角探索的方法，其中潜在姿势的评分基于从该姿势可见的语义特征。我们研究了两种潜在视图抽样方法，并在仿真和实际实验中展示了我们框架的有效性。自动创建高质量的语义地图可以使得机器人更好地理解并与其环境交互，并使未来具身AI应用更容易部署。 

---
# Real-is-Sim: Bridging the Sim-to-Real Gap with a Dynamic Digital Twin for Real-World Robot Policy Evaluation 

**Title (ZH)**: 实境即仿真：通过动态数字分身弥合仿真到实境的差距以评估真实世界机器人政策 

**Authors**: Jad Abou-Chakra, Lingfeng Sun, Krishan Rana, Brandon May, Karl Schmeckpeper, Maria Vittoria Minniti, Laura Herlant  

**Link**: [PDF](https://arxiv.org/pdf/2504.03597)  

**Abstract**: Recent advancements in behavior cloning have enabled robots to perform complex manipulation tasks. However, accurately assessing training performance remains challenging, particularly for real-world applications, as behavior cloning losses often correlate poorly with actual task success. Consequently, researchers resort to success rate metrics derived from costly and time-consuming real-world evaluations, making the identification of optimal policies and detection of overfitting or underfitting impractical. To address these issues, we propose real-is-sim, a novel behavior cloning framework that incorporates a dynamic digital twin (based on Embodied Gaussians) throughout the entire policy development pipeline: data collection, training, and deployment. By continuously aligning the simulated world with the physical world, demonstrations can be collected in the real world with states extracted from the simulator. The simulator enables flexible state representations by rendering image inputs from any viewpoint or extracting low-level state information from objects embodied within the scene. During training, policies can be directly evaluated within the simulator in an offline and highly parallelizable manner. Finally, during deployment, policies are run within the simulator where the real robot directly tracks the simulated robot's joints, effectively decoupling policy execution from real hardware and mitigating traditional domain-transfer challenges. We validate real-is-sim on the PushT manipulation task, demonstrating strong correlation between success rates obtained in the simulator and real-world evaluations. Videos of our system can be found at this https URL. 

**Abstract (ZH)**: 基于行为克隆的Recent advancements in behavior cloning have enabled robots to perform complex manipulation tasks.然而，准确评估训练性能仍然具有挑战性，尤其是在实际应用中，因为行为克隆损失与实际任务成功率的相关性较差。因此，研究人员不得不依赖昂贵且耗时的实地评估所获得的成功率指标，这使得识别最佳策略和检测过拟合或欠拟合变得 impractical。为了解决这些问题，我们提出了一种名为real-is-sim的新颖行为克隆框架，该框架在整个策略开发流程（数据收集、训练和部署）中引入了一个动态的数字孪生体（基于Embodied Gaussians）：通过不断使模拟世界与物理世界保持一致，可以在真实世界中收集演示，同时从模拟器中提取状态。模拟器通过从场景中体现的物体中提取低级状态信息或从任何视角渲染图像输入，实现了灵活的状态表示。在训练过程中，策略可以在模拟器中离线且高度并行地直接评估。最后，在部署阶段，策略在模拟器中运行，其中真实机器人直接跟踪模拟机器人的关节，有效解耦策略执行与实际硬件，从而减缓传统的域迁移挑战。我们在PushT抓取任务上验证了real-is-sim，证明了模拟器中获得的成功率与实地评估之间存在强烈的相关性。我们的系统视频可在以下链接找到：this https URL。 

---
# Dexterous Manipulation through Imitation Learning: A Survey 

**Title (ZH)**: 灵巧 manipulation 通过模仿学习：一种综述 

**Authors**: Shan An, Ziyu Meng, Chao Tang, Yuning Zhou, Tengyu Liu, Fangqiang Ding, Shufang Zhang, Yao Mu, Ran Song, Wei Zhang, Zeng-Guang Hou, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03515)  

**Abstract**: Dexterous manipulation, which refers to the ability of a robotic hand or multi-fingered end-effector to skillfully control, reorient, and manipulate objects through precise, coordinated finger movements and adaptive force modulation, enables complex interactions similar to human hand dexterity. With recent advances in robotics and machine learning, there is a growing demand for these systems to operate in complex and unstructured environments. Traditional model-based approaches struggle to generalize across tasks and object variations due to the high-dimensionality and complex contact dynamics of dexterous manipulation. Although model-free methods such as reinforcement learning (RL) show promise, they require extensive training, large-scale interaction data, and carefully designed rewards for stability and effectiveness. Imitation learning (IL) offers an alternative by allowing robots to acquire dexterous manipulation skills directly from expert demonstrations, capturing fine-grained coordination and contact dynamics while bypassing the need for explicit modeling and large-scale trial-and-error. This survey provides an overview of dexterous manipulation methods based on imitation learning (IL), details recent advances, and addresses key challenges in the field. Additionally, it explores potential research directions to enhance IL-driven dexterous manipulation. Our goal is to offer researchers and practitioners a comprehensive introduction to this rapidly evolving domain. 

**Abstract (ZH)**: Dexterous Manipulation基于模仿学习的方法：概述、 Recent Advances与关键挑战及潜在研究方向 

---
# Learning Dual-Arm Coordination for Grasping Large Flat Objects 

**Title (ZH)**: 学习双臂协调抓取大型平坦物体 

**Authors**: Yongliang Wang, Hamidreza Kasaei  

**Link**: [PDF](https://arxiv.org/pdf/2504.03500)  

**Abstract**: Grasping large flat objects, such as books or keyboards lying horizontally, presents significant challenges for single-arm robotic systems, often requiring extra actions like pushing objects against walls or moving them to the edge of a surface to facilitate grasping. In contrast, dual-arm manipulation, inspired by human dexterity, offers a more refined solution by directly coordinating both arms to lift and grasp the object without the need for complex repositioning. In this paper, we propose a model-free deep reinforcement learning (DRL) framework to enable dual-arm coordination for grasping large flat objects. We utilize a large-scale grasp pose detection model as a backbone to extract high-dimensional features from input images, which are then used as the state representation in a reinforcement learning (RL) model. A CNN-based Proximal Policy Optimization (PPO) algorithm with shared Actor-Critic layers is employed to learn coordinated dual-arm grasp actions. The system is trained and tested in Isaac Gym and deployed to real robots. Experimental results demonstrate that our policy can effectively grasp large flat objects without requiring additional maneuvers. Furthermore, the policy exhibits strong generalization capabilities, successfully handling unseen objects. Importantly, it can be directly transferred to real robots without fine-tuning, consistently outperforming baseline methods. 

**Abstract (ZH)**: 基于模型 free 深度强化学习的双臂协调抓取大型平板物体方法 

---
# MultiClear: Multimodal Soft Exoskeleton Glove for Transparent Object Grasping Assistance 

**Title (ZH)**: MultiClear：多模态柔软外骨骼手套，用于透明物体抓取辅助 

**Authors**: Chen Hu, Timothy Neate, Shan Luo, Letizia Gionfrida  

**Link**: [PDF](https://arxiv.org/pdf/2504.03379)  

**Abstract**: Grasping is a fundamental skill for interacting with the environment. However, this ability can be difficult for some (e.g. due to disability). Wearable robotic solutions can enhance or restore hand function, and recent advances have leveraged computer vision to improve grasping capabilities. However, grasping transparent objects remains challenging due to their poor visual contrast and ambiguous depth cues. Furthermore, while multimodal control strategies incorporating tactile and auditory feedback have been explored to grasp transparent objects, the integration of vision with these modalities remains underdeveloped. This paper introduces MultiClear, a multimodal framework designed to enhance grasping assistance in a wearable soft exoskeleton glove for transparent objects by fusing RGB data, depth data, and auditory signals. The exoskeleton glove integrates a tendon-driven actuator with an RGB-D camera and a built-in microphone. To achieve precise and adaptive control, a hierarchical control architecture is proposed. For the proposed hierarchical control architecture, a high-level control layer provides contextual awareness, a mid-level control layer processes multimodal sensory inputs, and a low-level control executes PID motor control for fine-tuned grasping adjustments. The challenge of transparent object segmentation was managed by introducing a vision foundation model for zero-shot segmentation. The proposed system achieves a Grasping Ability Score of 70.37%, demonstrating its effectiveness in transparent object manipulation. 

**Abstract (ZH)**: 多模态框架MultiClear增强穿戴式软exo手套在透明物体抓取中的辅助能力 

---
# Point Cloud-based Grasping for Soft Hand Exoskeleton 

**Title (ZH)**: 基于点云的软手外骨骼抓取技术 

**Authors**: Chen Hu, Enrica Tricomi, Eojin Rho, Daekyum Kim, Lorenzo Masia, Shan Luo, Letizia Gionfrida  

**Link**: [PDF](https://arxiv.org/pdf/2504.03369)  

**Abstract**: Grasping is a fundamental skill for interacting with and manipulating objects in the environment. However, this ability can be challenging for individuals with hand impairments. Soft hand exoskeletons designed to assist grasping can enhance or restore essential hand functions, yet controlling these soft exoskeletons to support users effectively remains difficult due to the complexity of understanding the environment. This study presents a vision-based predictive control framework that leverages contextual awareness from depth perception to predict the grasping target and determine the next control state for activation. Unlike data-driven approaches that require extensive labelled datasets and struggle with generalizability, our method is grounded in geometric modelling, enabling robust adaptation across diverse grasping scenarios. The Grasping Ability Score (GAS) was used to evaluate performance, with our system achieving a state-of-the-art GAS of 91% across 15 objects and healthy participants, demonstrating its effectiveness across different object types. The proposed approach maintained reconstruction success for unseen objects, underscoring its enhanced generalizability compared to learning-based models. 

**Abstract (ZH)**: 基于视觉的预测控制框架：利用深度感知的上下文意识预测抓取目标并确定下一控制状态 

---
# Dynamic Objective MPC for Motion Planning of Seamless Docking Maneuvers 

**Title (ZH)**: 无缝对接机动规划的动态目标模型预测控制 

**Authors**: Oliver Schumann, Michael Buchholz, Klaus Dietmayer  

**Link**: [PDF](https://arxiv.org/pdf/2504.03280)  

**Abstract**: Automated vehicles and logistics robots must often position themselves in narrow environments with high precision in front of a specific target, such as a package or their charging station. Often, these docking scenarios are solved in two steps: path following and rough positioning followed by a high-precision motion planning algorithm. This can generate suboptimal trajectories caused by bad positioning in the first phase and, therefore, prolong the time it takes to reach the goal. In this work, we propose a unified approach, which is based on a Model Predictive Control (MPC) that unifies the advantages of Model Predictive Contouring Control (MPCC) with a Cartesian MPC to reach a specific goal pose. The paper's main contributions are the adaption of the dynamic weight allocation method to reach path ends and goal poses inside driving corridors, and the development of the so-called dynamic objective MPC. The latter is an improvement of the dynamic weight allocation method, which can inherently switch state-dependent from an MPCC to a Cartesian MPC to solve the path-following problem and the high-precision positioning tasks independently of the location of the goal pose seamlessly by one algorithm. This leads to foresighted, feasible, and safe motion plans, which can decrease the mission time and result in smoother trajectories. 

**Abstract (ZH)**: 自动驾驶车辆和物流机器人经常需要在狭窄环境中以高精度定位到特定目标前方，如包裹或充电站。通常，这些对接场景分为两个步骤：路径跟随和粗定位，随后是高精度运动规划算法。这一过程可能导致由于初始阶段定位不准确而产生的次优轨迹，从而延长达到目标所需时间。本文提出了一种统一方法，基于模型预测控制（MPC），结合了模型预测轮廓控制（MPCC）和笛卡尔MPC的优点，以达到特定的目标姿态。本文的主要贡献是将动态权重分配方法适应于在驾驶走廊内达到路径末端和目标姿态，并开发了所谓的动态目标MPC。后者是动态权重分配方法的改进，可以固有地根据状态从MPCC切换到笛卡尔MPC，独立地解决路径跟随问题和高精度定位任务，无需算法切换。这导致前瞻性的、可行的和安全的运动计划，可以减少任务时间并产生更平滑的轨迹。 

---
# Gradient Field-Based Dynamic Window Approach for Collision Avoidance in Complex Environments 

**Title (ZH)**: 基于梯度场的动态窗口碰撞 avoidance 方法在复杂环境中的应用 

**Authors**: Ze Zhang, Yifan Xue, Nadia Figueroa, Knut Åkesson  

**Link**: [PDF](https://arxiv.org/pdf/2504.03260)  

**Abstract**: For safe and flexible navigation in multi-robot systems, this paper presents an enhanced and predictive sampling-based trajectory planning approach in complex environments, the Gradient Field-based Dynamic Window Approach (GF-DWA). Building upon the dynamic window approach, the proposed method utilizes gradient information of obstacle distances as a new cost term to anticipate potential collisions. This enhancement enables the robot to improve awareness of obstacles, including those with non-convex shapes. The gradient field is derived from the Gaussian process distance field, which generates both the distance field and gradient field by leveraging Gaussian process regression to model the spatial structure of the environment. Through several obstacle avoidance and fleet collision avoidance scenarios, the proposed GF-DWA is shown to outperform other popular trajectory planning and control methods in terms of safety and flexibility, especially in complex environments with non-convex obstacles. 

**Abstract (ZH)**: 基于梯度场的动态窗口方法（GF-DWA）：多机器人系统在复杂环境中的安全可灵活导航的增强预测采样轨迹规划 

---
# GraphSeg: Segmented 3D Representations via Graph Edge Addition and Contraction 

**Title (ZH)**: GraphSeg: 基于图边添加与收缩的分段3D表示 

**Authors**: Haozhan Tang, Tianyi Zhang, Oliver Kroemer, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2504.03129)  

**Abstract**: Robots operating in unstructured environments often require accurate and consistent object-level representations. This typically requires segmenting individual objects from the robot's surroundings. While recent large models such as Segment Anything (SAM) offer strong performance in 2D image segmentation. These advances do not translate directly to performance in the physical 3D world, where they often over-segment objects and fail to produce consistent mask correspondences across views. In this paper, we present GraphSeg, a framework for generating consistent 3D object segmentations from a sparse set of 2D images of the environment without any depth information. GraphSeg adds edges to graphs and constructs dual correspondence graphs: one from 2D pixel-level similarities and one from inferred 3D structure. We formulate segmentation as a problem of edge addition, then subsequent graph contraction, which merges multiple 2D masks into unified object-level segmentations. We can then leverage \emph{3D foundation models} to produce segmented 3D representations. GraphSeg achieves robust segmentation with significantly fewer images and greater accuracy than prior methods. We demonstrate state-of-the-art performance on tabletop scenes and show that GraphSeg enables improved performance on downstream robotic manipulation tasks. Code available at this https URL. 

**Abstract (ZH)**: 机器人在非结构化环境中的操作往往需要精确且一致的对象级表示。这通常需要将单个物体从机器人的周围环境中分离出来。虽然近期的大模型如Segment Anything (SAM) 在2D图像分割方面表现出色，但这些进步并不直接转化为物理3D世界中的性能，因为在3D世界中，它们往往会导致过度分割物体，并且难以在不同视角下产生一致的掩码对应关系。本文提出GraphSeg框架，用于从环境的稀疏2D图像集生成一致的3D对象分割，无需任何深度信息。GraphSeg通过添加图中的边并构建双重对应图：一个是基于2D像素级相似性，另一个是基于推断的3D结构。我们将分割问题形式化为边添加的问题，然后通过随后的图收缩，将多个2D掩码合并为统一的对象级分割。然后，可以利用3D基础模型生成分割的3D表示。GraphSeg在显著减少图像数量和提高分割准确性方面表现出了鲁棒性。我们在桌面场景上展示了最先进的性能，并证明GraphSeg能够提高下游机器人操作任务的表现。代码可在此链接获取：this https URL。 

---
# The Use of Gaze-Derived Confidence of Inferred Operator Intent in Adjusting Safety-Conscious Haptic Assistance 

**Title (ZH)**: 基于凝视推断的操作员意图置信度在调整意识安全触觉辅助中的应用 

**Authors**: Jeremy D. Webb, Michael Bowman, Songpo Li, Xiaoli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03098)  

**Abstract**: Humans directly completing tasks in dangerous or hazardous conditions is not always possible where these tasks are increasingly be performed remotely by teleoperated robots. However, teleoperation is difficult since the operator feels a disconnect with the robot caused by missing feedback from several senses, including touch, and the lack of depth in the video feedback presented to the operator. To overcome this problem, the proposed system actively infers the operator's intent and provides assistance based on the predicted intent. Furthermore, a novel method of calculating confidence in the inferred intent modifies the human-in-the-loop control. The operator's gaze is employed to intuitively indicate the target before the manipulation with the robot begins. A potential field method is used to provide a guiding force towards the intended target, and a safety boundary reduces risk of damage. Modifying these assistances based on the confidence level in the operator's intent makes the control more natural, and gives the robot an intuitive understanding of its human master. Initial validation results show the ability of the system to improve accuracy, execution time, and reduce operator error. 

**Abstract (ZH)**: 人类无法在危险条件下直接完成任务，这些任务越来越多地由远程操控机器人完成。然而，远程操控因操作者缺乏触觉反馈等多感官反馈以及视频反馈的缺乏深度而难以实现。为此，所提系统主动推断操作者的意图，并根据预测的意图提供辅助。此外，一种新颖的计算推断意图置信度的方法修改了人机在环控制。操作者的凝视被用来直观指示目标，机器人操作开始前。使用势场方法提供一个引导力，使机器人趋向预期目标，并通过安全边界减少损伤风险。根据操作者意图的置信度水平调整这些辅助手段，使控制更加自然，并赋予机器人对人类主人的直观理解。初步验证结果表明，该系统能够提高精确度、缩短执行时间并降低操作员错误率。 

---
# Statics of continuum planar grasping 

**Title (ZH)**: 连续平面抓取的静力学分析 

**Authors**: Udit Halder  

**Link**: [PDF](https://arxiv.org/pdf/2504.03067)  

**Abstract**: Continuum robotic grasping, inspired by biological appendages such as octopus arms and elephant trunks, provides a versatile and adaptive approach to object manipulation. Unlike conventional rigid-body grasping, continuum robots leverage distributed compliance and whole-body contact to achieve robust and dexterous grasping. This paper presents a control-theoretic framework for analyzing the statics of continuous contact with a planar object. The governing equations of static equilibrium of the object are formulated as a linear control system, where the distributed contact forces act as control inputs. To optimize the grasping performance, a constrained optimal control problem is posed to minimize contact forces required to achieve a static grasp, with solutions derived using the Pontryagin Maximum Principle. Furthermore, two optimization problems are introduced: (i) for assigning a measure to the quality of a particular grasp, which generalizes a (rigid-body) grasp quality metric in the continuum case, and (ii) for finding the best grasping configuration that maximizes the continuum grasp quality. Several numerical results are also provided to elucidate our methods. 

**Abstract (ZH)**: 连续体机器人抓取：受八足和象鼻生物附肢启发的物体 manipulation 提供了灵活适应的抓取方法。不同于传统的刚体抓取，连续体机器人利用分布式柔顺性和全身接触实现稳定的灵巧抓取。本文提出了一种控制理论框架，用于分析平面物体连续接触的静力学。物体的静态平衡方程被表述为一个线性控制系统，其中分布式接触力作为控制输入。为了优化抓取性能，提出了一个约束最优控制问题，旨在最小化实现静态抓取所需的接触力，并使用庞特里亚金最大原理求解。此外，引入了两个优化问题：(i) 为特定抓取质量赋予测度，该测度在连续体情况下推广了刚体抓取质量度量，(ii) 寻找最大化连续体抓取质量的最佳抓取配置。还提供了若干数值结果以阐明我们的方法。 

---
# Push-Grasp Policy Learning Using Equivariant Models and Grasp Score Optimization 

**Title (ZH)**: 基于不变模型和夹持评分优化的推-抓取策略学习 

**Authors**: Boce Hu, Heng Tian, Dian Wang, Haojie Huang, Xupeng Zhu, Robin Walters, Robert Platt  

**Link**: [PDF](https://arxiv.org/pdf/2504.03053)  

**Abstract**: Goal-conditioned robotic grasping in cluttered environments remains a challenging problem due to occlusions caused by surrounding objects, which prevent direct access to the target object. A promising solution to mitigate this issue is combining pushing and grasping policies, enabling active rearrangement of the scene to facilitate target retrieval. However, existing methods often overlook the rich geometric structures inherent in such tasks, thus limiting their effectiveness in complex, heavily cluttered scenarios. To address this, we propose the Equivariant Push-Grasp Network, a novel framework for joint pushing and grasping policy learning. Our contributions are twofold: (1) leveraging SE(2)-equivariance to improve both pushing and grasping performance and (2) a grasp score optimization-based training strategy that simplifies the joint learning process. Experimental results show that our method improves grasp success rates by 49% in simulation and by 35% in real-world scenarios compared to strong baselines, representing a significant advancement in push-grasp policy learning. 

**Abstract (ZH)**: 目标导向的机器人抓取在复杂环境中的挑战由于周围物体造成的遮挡，难以直接访问目标物体。一种有希望的解决方案是结合推拉策略，使机器人能够主动重组场景以利于目标物体的获取。然而，现有方法往往忽略了此类任务中固有的丰富几何结构，从而限制了其在复杂、高度拥挤场景中的有效性。为了解决这一问题，我们提出了等变推拉网络（Equivariant Push-Grasp Network），这是一种新的联合推拉策略学习框架。我们的贡献主要有两点：（1）利用SE(2)-等变性提高推拉性能；（2）基于抓取得分优化的训练策略简化联合学习过程。实验结果表明，与强基线相比，我们的方法在模拟环境中提高了49%的抓取成功率，在真实世界场景中提高了35%，标志着推拉策略学习的一个重要进步。 

---
# How to Adapt Control Barrier Functions? A Learning-Based Approach with Applications to a VTOL Quadplane 

**Title (ZH)**: 基于学习的方法如何适应控制屏障函数？应用于垂直起降四旋翼无人机的实例探究 

**Authors**: Taekyung Kim, Randal W. Beard, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2504.03038)  

**Abstract**: In this paper, we present a novel theoretical framework for online adaptation of Control Barrier Function (CBF) parameters, i.e., of the class K functions included in the CBF condition, under input constraints. We introduce the concept of locally validated CBF parameters, which are adapted online to guarantee finite-horizon safety, based on conditions derived from Nagumo's theorem and tangent cone analysis. To identify these parameters online, we integrate a learning-based approach with an uncertainty-aware verification process that account for both epistemic and aleatoric uncertainties inherent in neural network predictions. Our method is demonstrated on a VTOL quadplane model during challenging transition and landing maneuvers, showcasing enhanced performance while maintaining safety. 

**Abstract (ZH)**: 本文提出了一种新型的理论框架，用于在输入约束条件下在线适应控制屏障函数（CBF）参数，即CBF条件中包含的类K函数参数。我们引入了局部验证CBF参数的概念，这些参数通过从Nagumo定理和切锥分析中导出的条件进行在线调整，以保证有限时限的安全性。为了在线识别这些参数，我们将基于学习的方法与一个同时考虑神经网络预测中固有的先验不确定性和偶然不确定性的验证过程相结合。我们的方法在VTOL四旋翼模型的挑战性过渡和着陆动作中进行了演示，展示了在保持安全的前提下性能的提升。 

---
# AuDeRe: Automated Strategy Decision and Realization in Robot Planning and Control via LLMs 

**Title (ZH)**: AuDeRe：通过大型语言模型在机器人规划与控制中的自动策略决策与实现 

**Authors**: Yue Meng, Fei Chen, Yongchao Chen, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2504.03015)  

**Abstract**: Recent advancements in large language models (LLMs) have shown significant promise in various domains, especially robotics. However, most prior LLM-based work in robotic applications either directly predicts waypoints or applies LLMs within fixed tool integration frameworks, offering limited flexibility in exploring and configuring solutions best suited to different tasks. In this work, we propose a framework that leverages LLMs to select appropriate planning and control strategies based on task descriptions, environmental constraints, and system dynamics. These strategies are then executed by calling the available comprehensive planning and control APIs. Our approach employs iterative LLM-based reasoning with performance feedback to refine the algorithm selection. We validate our approach through extensive experiments across tasks of varying complexity, from simple tracking to complex planning scenarios involving spatiotemporal constraints. The results demonstrate that using LLMs to determine planning and control strategies from natural language descriptions significantly enhances robotic autonomy while reducing the need for extensive manual tuning and expert knowledge. Furthermore, our framework maintains generalizability across different tasks and notably outperforms baseline methods that rely on LLMs for direct trajectory, control sequence, or code generation. 

**Abstract (ZH)**: 近期大型语言模型的进展在各个领域显示出显著的潜力，特别是在机器人领域。然而，大多数基于大型语言模型的机器人应用程序工作要么直接预测航点，要么在固定工具集成框架内应用大型语言模型，这在探索和配置最适合不同任务的解决方案方面提供了有限的灵活性。在本研究中，我们提出了一种框架，该框架利用大型语言模型根据任务描述、环境约束和系统动力学选择合适的规划和控制策略。然后通过调用全面的规划和控制API来执行这些策略。我们的方法采用迭代的基于大型语言模型的推理并结合性能反馈来精炼算法选择。通过涵盖从简单跟踪到涉及时空约束的复杂规划场景的广泛实验，我们验证了这种方法。实验结果表明，从自然语言描述中确定规划和控制策略可以显著增强机器人的自主性，同时减少对大量手动调整和专家知识的需求。此外，我们的框架在不同任务之间保持了一定的通用性，并且在依赖大型语言模型直接生成轨迹、控制序列或代码的基本方法中表现更优。 

---
# Autonomy Architectures for Safe Planning in Unknown Environments Under Budget Constraints 

**Title (ZH)**: 预算约束下未知环境中的安全规划自主架构 

**Authors**: Daniel M. Cherenson, Devansh R. Agrawal, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2504.03001)  

**Abstract**: Mission planning can often be formulated as a constrained control problem under multiple path constraints (i.e., safety constraints) and budget constraints (i.e., resource expenditure constraints). In a priori unknown environments, verifying that an offline solution will satisfy the constraints for all time can be difficult, if not impossible. Our contributions are as follows: 1) We propose an online method, building on our previous work "gatekeeper", to guarantee safety and satisfy budget constraints of the system trajectory at all times throughout a mission. 2) Next, we prove that our algorithm is recursively feasible and correct. 3) Finally, instead of using a heuristically designed backup controller, we propose a sampling-based method to construct backup trajectories that both minimize resource expenditure and reach budget renewal sets, in which path constraints are satisfied and the constrained resources are renewed. We demonstrate our approach in simulation with a fixed-wing UAV in a GNSS-denied environment with a budget constraint on localization error that can be renewed at visual landmarks. 

**Abstract (ZH)**: 任务规划往往可以被形式化为在多重路径约束（即安全约束）和预算约束（即资源消耗约束）下的受限控制问题。在先验未知的环境中，验证离线解在整个时间内的约束满足性可能是困难的，甚至不可能。我们的贡献如下：1）我们提出了一个在线方法，基于我们之前的工作“gatekeeper”，以确保任务执行过程中系统的轨迹始终满足安全和预算约束；2）我们证明了该算法是递归可行且正确的；3）我们提出了一种基于采样的方法来构建备份轨迹，该方法在最小化资源消耗的同时，达到预算更新集，在这些集内路径约束得到满足且受限资源得到更新。我们在缺乏GPS的环境中通过视觉地标更新定位误差预算的固定翼无人机仿真中展示了该方法。 

---
# RANa: Retrieval-Augmented Navigation 

**Title (ZH)**: RANa: 检索增强导航 

**Authors**: Gianluca Monaci, Rafael S. Rezende, Romain Deffayet, Gabriela Csurka, Guillaume Bono, Hervé Déjean, Stéphane Clinchant, Christian Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2504.03524)  

**Abstract**: Methods for navigation based on large-scale learning typically treat each episode as a new problem, where the agent is spawned with a clean memory in an unknown environment. While these generalization capabilities to an unknown environment are extremely important, we claim that, in a realistic setting, an agent should have the capacity of exploiting information collected during earlier robot operations. We address this by introducing a new retrieval-augmented agent, trained with RL, capable of querying a database collected from previous episodes in the same environment and learning how to integrate this additional context information. We introduce a unique agent architecture for the general navigation task, evaluated on ObjectNav, ImageNav and Instance-ImageNav. Our retrieval and context encoding methods are data-driven and heavily employ vision foundation models (FM) for both semantic and geometric understanding. We propose new benchmarks for these settings and we show that retrieval allows zero-shot transfer across tasks and environments while significantly improving performance. 

**Abstract (ZH)**: 基于大规模学习的导航方法通常将每个episode视为一个新的问题，其中智能体在未知环境中以清空记忆的状态出现。虽然这种对未知环境的泛化能力非常重要，但我们认为在实际环境中，智能体应该有能力利用之前机器人操作中收集的信息。为此，我们提出了一种新的检索增强智能体，通过强化学习训练，能够在相同环境的先前episode中查询数据库，并学习如何整合这些额外的上下文信息。我们为通用导航任务引入了一种独特的智能体架构，并在ObjectNav、ImageNav和Instance-ImageNav上进行了评估。我们的检索和上下文编码方法是数据驱动的，并大量使用视觉基础模型（FM）来进行语义和几何理解。我们为这些环境提出了新的基准，并展示了检索允许在任务和环境之间进行零-shot迁移，并显著提高了性能。 

---
# DML-RAM: Deep Multimodal Learning Framework for Robotic Arm Manipulation using Pre-trained Models 

**Title (ZH)**: 基于预训练模型的深度多模态学习框架：应用于机器人臂操作 

**Authors**: Sathish Kumar, Swaroop Damodaran, Naveen Kumar Kuruba, Sumit Jha, Arvind Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.03423)  

**Abstract**: This paper presents a novel deep learning framework for robotic arm manipulation that integrates multimodal inputs using a late-fusion strategy. Unlike traditional end-to-end or reinforcement learning approaches, our method processes image sequences with pre-trained models and robot state data with machine learning algorithms, fusing their outputs to predict continuous action values for control. Evaluated on BridgeData V2 and Kuka datasets, the best configuration (VGG16 + Random Forest) achieved MSEs of 0.0021 and 0.0028, respectively, demonstrating strong predictive performance and robustness. The framework supports modularity, interpretability, and real-time decision-making, aligning with the goals of adaptive, human-in-the-loop cyber-physical systems. 

**Abstract (ZH)**: 一种集成多模态输入的新型深度学习机器人手臂 manipulation 框架：基于晚融合策略的研究 

---
# An Efficient GPU-based Implementation for Noise Robust Sound Source Localization 

**Title (ZH)**: 基于GPU的噪声鲁棒声源定位高效实现 

**Authors**: Zirui Lin, Masayuki Takigahira, Naoya Terakado, Haris Gulzar, Monikka Roslianna Busto, Takeharu Eda, Katsutoshi Itoyama, Kazuhiro Nakadai, Hideharu Amano  

**Link**: [PDF](https://arxiv.org/pdf/2504.03373)  

**Abstract**: Robot audition, encompassing Sound Source Localization (SSL), Sound Source Separation (SSS), and Automatic Speech Recognition (ASR), enables robots and smart devices to acquire auditory capabilities similar to human hearing. Despite their wide applicability, processing multi-channel audio signals from microphone arrays in SSL involves computationally intensive matrix operations, which can hinder efficient deployment on Central Processing Units (CPUs), particularly in embedded systems with limited CPU resources. This paper introduces a GPU-based implementation of SSL for robot audition, utilizing the Generalized Singular Value Decomposition-based Multiple Signal Classification (GSVD-MUSIC), a noise-robust algorithm, within the HARK platform, an open-source software suite. For a 60-channel microphone array, the proposed implementation achieves significant performance improvements. On the Jetson AGX Orin, an embedded device powered by an NVIDIA GPU and ARM Cortex-A78AE v8.2 64-bit CPUs, we observe speedups of 4645.1x for GSVD calculations and 8.8x for the SSL module, while speedups of 2223.4x for GSVD calculation and 8.95x for the entire SSL module on a server configured with an NVIDIA A100 GPU and AMD EPYC 7352 CPUs, making real-time processing feasible for large-scale microphone arrays and providing ample capacity for real-time processing of potential subsequent machine learning or deep learning tasks. 

**Abstract (ZH)**: 基于GPU的机器人听觉声源定位实现：利用HARK平台的噪稳健广义奇异值分解多重信号分类算法 

---
# Energy Aware and Safe Path Planning for Unmanned Aircraft Systems 

**Title (ZH)**: 能源aware和安全路径规划for无人驾驶航空系统 

**Authors**: Sebastian Gasche, Christian Kallies, Andreas Himmel, Rolf Findeisen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03271)  

**Abstract**: This paper proposes a path planning algorithm for multi-agent unmanned aircraft systems (UASs) to autonomously cover a search area, while considering obstacle avoidance, as well as the capabilities and energy consumption of the employed unmanned aerial vehicles. The path planning is optimized in terms of energy efficiency to prefer low energy-consuming maneuvers. In scenarios where a UAS is low on energy, it autonomously returns to its initial position for a safe landing, thus preventing potential battery damage. To accomplish this, an energy-aware multicopter model is integrated into a path planning algorithm based on model predictive control and mixed integer linear programming. Besides factoring in energy consumption, the planning is improved by dynamically defining feasible regions for each UAS to prevent obstacle corner-cutting or over-jumping. 

**Abstract (ZH)**: 本文提出了一种多无人机系统(UASs)的路径规划算法，使其能够自主覆盖搜索区域，同时考虑避障、以及所用无人机的能力和能耗。路径规划在能效方面进行了优化，倾向于选择低能耗操作。在无人机能量不足的情况下，它能够自主返回初始位置安全降落，防止潜在的电池损坏。为此，基于模型预测控制和混合整数线性规划的路径规划算法中集成了一种能量感知的多旋翼模型。此外，通过动态定义每个无人机的可行区域来防止障碍物角端绕行或过跳，从而提高规划质量。 

---
# A Modular Energy Aware Framework for Multicopter Modeling in Control and Planning Applications 

**Title (ZH)**: 面向控制与规划应用的多旋翼 Modeling 的模块化能源意识框架 

**Authors**: Sebastian Gasche, Christian Kallies, Andreas Himmel, Rolf Findeisen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03256)  

**Abstract**: Unmanned aerial vehicles (UAVs), especially multicopters, have recently gained popularity for use in surveillance, monitoring, inspection, and search and rescue missions. Their maneuverability and ability to operate in confined spaces make them particularly useful in cluttered environments. For advanced control and mission planning applications, accurate and resource-efficient modeling of UAVs and their capabilities is essential. This study presents a modular approach to multicopter modeling that considers vehicle dynamics, energy consumption, and sensor integration. The power train model includes detailed descriptions of key components such as the lithium-ion battery, electronic speed controllers, and brushless DC motors. Their models are validated with real test flight data. In addition, sensor models, including LiDAR and cameras, are integrated to describe the equipment often used in surveillance and monitoring missions. The individual models are combined into an energy-aware multicopter model, which provide the basis for a companion study on path planning for unmanned aircaft system (UAS) swarms performing search and rescue missions in cluttered and dynamic environments. The flexible modeling approach enables easy description of different UAVs in a heterogeneous UAS swarm, allowing for energy-efficient operations and autonomous decision making for a reliable mission performance. 

**Abstract (ZH)**: 无人直升机（UAVs），尤其是多旋翼无人机，近年来因其在监控、监测、检查及搜索救援任务中的应用而备受青睐。其机动性及在受限空间内操作的能力使得它们特别适用于复杂环境中。为了实现先进的控制和任务规划应用，对UAV及其能力进行准确且节省资源的建模至关重要。本研究提出了一种模块化的多旋翼无人机建模方法，该方法考虑了无人机的动力学、能耗以及传感器集成。动力系统模型详细描述了关键组件，如锂离子电池、电子调速器和无刷直流电机，并通过实际试飞数据进行了验证。此外，还集成了一系列传感器模型，包括LiDAR和摄像头，以描述通常用于监控和监测任务的设备。各个模型被整合成一种能源意识型多旋翼无人机模型，为执行搜索和救援任务的无人驾驶航空器（UAS）群在复杂和动态环境中的航迹规划研究奠定了基础。灵活的建模方法可方便地描述不同类型的UAV在异构UAS群中的特性，从而实现高效操作和自主决策，以确保任务执行的可靠性。 

---
# Robot Localization Using a Learned Keypoint Detector and Descriptor with a Floor Camera and a Feature Rich Industrial Floor 

**Title (ZH)**: 使用学习到的特征点检测器和描述符的地面上的相机和特征丰富的工业地面的机器人定位方法 

**Authors**: Piet Brömmel, Dominik Brämer, Oliver Urbann, Diana Kleingarn  

**Link**: [PDF](https://arxiv.org/pdf/2504.03249)  

**Abstract**: The localization of moving robots depends on the availability of good features from the environment. Sensor systems like Lidar are popular, but unique features can also be extracted from images of the ground. This work presents the Keypoint Localization Framework (KOALA), which utilizes deep neural networks that extract sufficient features from an industrial floor for accurate localization without having readable markers. For this purpose, we use a floor covering that can be produced as cheaply as common industrial floors. Although we do not use any filtering, prior, or temporal information, we can estimate our position in 75.7 % of all images with a mean position error of 2 cm and a rotation error of 2.4 %. Thus, the robot kidnapping problem can be solved with high precision in every frame, even while the robot is moving. Furthermore, we show that our framework with our detector and descriptor combination is able to outperform comparable approaches. 

**Abstract (ZH)**: 移动机器人定位依赖于环境中的良好特征。虽然像激光雷达这样的传感器系统很流行，但从地面图像中提取的独特特征也可以用于定位。本文提出了一种关键点定位框架（KOALA），利用深度神经网络从工业地板中提取足够的特征，无需可读标记即可实现精确定位。为此，我们使用了一种可低成本生产的地面覆盖物。尽管我们未使用任何过滤、先验或时间信息，但在所有图像中的75.7%的情况下，我们能够将位置误差估计为2 cm，旋转误差为2.4%。因此，即使机器人在移动过程中，也可以以高精度解决机器人绑架问题。此外，我们展示了我们的框架及其检测器和描述子组合能够优于其他可比方法。 

---
# Seeing is Believing: Belief-Space Planning with Foundation Models as Uncertainty Estimators 

**Title (ZH)**: 眼见为实：基于基础模型的信念空间规划 

**Authors**: Linfeng Zhao, Willie McClinton, Aidan Curtis, Nishanth Kumar, Tom Silver, Leslie Pack Kaelbling, Lawson L.S. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03245)  

**Abstract**: Generalizable robotic mobile manipulation in open-world environments poses significant challenges due to long horizons, complex goals, and partial observability. A promising approach to address these challenges involves planning with a library of parameterized skills, where a task planner sequences these skills to achieve goals specified in structured languages, such as logical expressions over symbolic facts. While vision-language models (VLMs) can be used to ground these expressions, they often assume full observability, leading to suboptimal behavior when the agent lacks sufficient information to evaluate facts with certainty. This paper introduces a novel framework that leverages VLMs as a perception module to estimate uncertainty and facilitate symbolic grounding. Our approach constructs a symbolic belief representation and uses a belief-space planner to generate uncertainty-aware plans that incorporate strategic information gathering. This enables the agent to effectively reason about partial observability and property uncertainty. We demonstrate our system on a range of challenging real-world tasks that require reasoning in partially observable environments. Simulated evaluations show that our approach outperforms both vanilla VLM-based end-to-end planning or VLM-based state estimation baselines by planning for and executing strategic information gathering. This work highlights the potential of VLMs to construct belief-space symbolic scene representations, enabling downstream tasks such as uncertainty-aware planning. 

**Abstract (ZH)**: 通用型机器人移动操控在开放世界环境中的泛化面临显著挑战，由于远期视角、复杂目标和部分可观测性。一种有希望的应对方法是使用参数化技能库进行规划，其中任务规划器将这些技能序列化，以实现通过结构化语言（如符号事实的逻辑表达式）指定的目标。虽然视觉-语言模型（VLMs）可以用于将这些表达式关联到具体场景，但它们通常假设完全可观测性，导致当代理缺乏足够信息以确定性评估事实时表现出次优行为。本文引入了一种新颖的框架，利用VLMs作为感知模块来估计不确定性并促进符号化关联。我们的方法构建了符号化的信念表示，并使用信念空间规划器生成考虑了信息收集策略的不确定性意识型规划。这使代理能够有效地推理部分可观测性和属性不确定性。我们在一系列需要部分可观测环境推理的具有挑战性真实世界任务上展示了我们的系统。模拟评估表明，与基于VLM的端到端规划或基于VLM的状态估计基准相比，我们的方法通过计划和执行战略信息收集来改进性能。本文突显了VLMs在构建信念空间符号化场景表示方面的潜力，从而支持下游任务如不确定性意识型规划。 

---
# Real-Time Roadway Obstacle Detection for Electric Scooters Using Deep Learning and Multi-Sensor Fusion 

**Title (ZH)**: 基于深度学习和多传感器融合的电动滑板车实时道路障碍检测 

**Authors**: Zeyang Zheng, Arman Hosseini, Dong Chen, Omid Shoghli, Arsalan Heydarian  

**Link**: [PDF](https://arxiv.org/pdf/2504.03171)  

**Abstract**: The increasing adoption of electric scooters (e-scooters) in urban areas has coincided with a rise in traffic accidents and injuries, largely due to their small wheels, lack of suspension, and sensitivity to uneven surfaces. While deep learning-based object detection has been widely used to improve automobile safety, its application for e-scooter obstacle detection remains unexplored. This study introduces a novel ground obstacle detection system for e-scooters, integrating an RGB camera, and a depth camera to enhance real-time road hazard detection. Additionally, the Inertial Measurement Unit (IMU) measures linear vertical acceleration to identify surface vibrations, guiding the selection of six obstacle categories: tree branches, manhole covers, potholes, pine cones, non-directional cracks, and truncated domes. All sensors, including the RGB camera, depth camera, and IMU, are integrated within the Intel RealSense Camera D435i. A deep learning model powered by YOLO detects road hazards and utilizes depth data to estimate obstacle proximity. Evaluated on the seven hours of naturalistic riding dataset, the system achieves a high mean average precision (mAP) of 0.827 and demonstrates excellent real-time performance. This approach provides an effective solution to enhance e-scooter safety through advanced computer vision and data fusion. The dataset is accessible at this https URL, and the project code is hosted on this https URL. 

**Abstract (ZH)**: 城市区域电动滑板车（e-scooter）的日益采用与交通事故和伤害的增加密切相关，主要原因是它们的轮子较小、缺乏减震以及对不平路面的敏感性。尽管基于深度学习的目标检测已在提高汽车安全性方面得到了广泛应用，但其在电动滑板车障碍检测中的应用尚未被探索。本研究介绍了一种针对电动滑板车的新型地面障碍检测系统，结合RGB相机和深度相机以增强实时路面威胁检测。此外，惯性测量单元（IMU）测量线性垂直加速度以识别路面振动，并指导选择六类障碍物：树枝、井盖、坑洞、松果、无定向裂缝和切除圆顶。所有传感器，包括RGB相机、深度相机和IMU，均集成在Intel RealSense Camera D435i中。由YOLO驱动的深度学习模型检测道路威胁并利用深度数据估算障碍物距离。在七小时的自然骑乘数据集上进行评估，该系统实现了高平均精度均值（mAP）0.827，并展示了卓越的实时性能。该方法通过先进的计算机视觉和数据融合增强电动滑板车安全。数据集可通过该链接访问，项目代码托管在该链接上。 

---
# Taming High-Dimensional Dynamics: Learning Optimal Projections onto Spectral Submanifolds 

**Title (ZH)**: 驯化高维动力学：学习最优投影到谱子流形上 

**Authors**: Hugo Buurmeijer, Luis A. Pabon, John Irvin Alora, Roshan S. Kaundinya, George Haller, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2504.03157)  

**Abstract**: High-dimensional nonlinear systems pose considerable challenges for modeling and control across many domains, from fluid mechanics to advanced robotics. Such systems are typically approximated with reduced order models, which often rely on orthogonal projections, a simplification that may lead to large prediction errors. In this work, we derive optimality of fiber-aligned projections onto spectral submanifolds, preserving the nonlinear geometric structure and minimizing long-term prediction error. We propose a computationally tractable procedure to approximate these projections from data, and show how the effect of control can be incorporated. For a 180-dimensional robotic system, we demonstrate that our reduced-order models outperform previous state-of-the-art approaches by up to fivefold in trajectory tracking accuracy under model predictive control. 

**Abstract (ZH)**: 高维非线性系统在流体力学和先进机器人等领域建模与控制中提出 considerable 挑战，通常通过降阶模型进行逼近，但可能产生较大预测误差。本文我们推导了沿纤维对谱子流形进行对齐投影的最优性，保留了非线性几何结构并最小化长期预测误差。提出了一种基于数据的计算上可行的方法来近似这些投影，并展示了如何纳入控制效应。对于一个180维的机器人系统，在模型预测控制下，我们的降阶模型在轨迹跟踪准确性上比现有最佳方法提高了一倍多。 

---
# Distributed Linear Quadratic Gaussian for Multi-Robot Coordination with Localization Uncertainty 

**Title (ZH)**: 分布式线性二次加权滤波器在存在定位不确定性条件下多机器人协调控制 

**Authors**: Tohid Kargar Tasooji, Sakineh Khodadadi  

**Link**: [PDF](https://arxiv.org/pdf/2504.03126)  

**Abstract**: This paper addresses the problem of distributed coordination control for multi-robot systems (MRSs) in the presence of localization uncertainty using a Linear Quadratic Gaussian (LQG) approach. We introduce a stochastic LQG control strategy that ensures the coordination of mobile robots while optimizing a performance criterion. The proposed control framework accounts for the inherent uncertainty in localization measurements, enabling robust decision-making and coordination. We analyze the stability of the system under the proposed control protocol, deriving conditions for the convergence of the multi-robot network. The effectiveness of the proposed approach is demonstrated through experimental validation using Robotrium simulation experiments, showcasing the practical applicability of the control strategy in real-world scenarios with localization uncertainty. 

**Abstract (ZH)**: 基于线性二次高斯方法的多机器人系统局部化不确定性下分布式协调控制研究 

---
# Event-Based Distributed Linear Quadratic Gaussian for Multi-Robot Coordination with Localization Uncertainty 

**Title (ZH)**: 事件驱动的分布式线性二次高斯控制用于具有定位不确定性多机器人协调 

**Authors**: Tohid Kargar Tasooji, Sakineh Khodadadi  

**Link**: [PDF](https://arxiv.org/pdf/2504.03125)  

**Abstract**: This paper addresses the problem of event-based distributed Linear Quadratic Gaussian (LQG) control for multirobot coordination under localization uncertainty. An event-triggered LQG rendezvous control strategy is proposed to ensure coordinated motion while reducing communication overhead. The design framework decouples the LQG controller from the event-triggering mechanism, although the scheduler parameters critically influence rendezvous performance. We establish stochastic stability for the closed-loop multi-robot system and demonstrate that a carefully tuned event-triggering scheduler can effectively balance rendezvous accuracy with communication efficiency by limiting the upper bound of the rendezvous error while minimizing the average transmission rate. Experimental results using a group of Robotarium mobile robots validate the proposed approach, confirming its efficacy in achieving robust coordination under uncertainty. 

**Abstract (ZH)**: 基于事件驱动的分布式线性二次高斯控制在定位不确定性下的多机器人协调问题研究 

---
# Event-Triggered Nonlinear Model Predictive Control for Cooperative Cable-Suspended Payload Transportation with Multi-Quadrotors 

**Title (ZH)**: 基于事件触发的非线性模型预测控制在多旋翼协同缆索悬挂载荷运输中的应用 

**Authors**: Tohid Kargar Tasooji, Sakineh Khodadadi, Guangjun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03123)  

**Abstract**: Autonomous Micro Aerial Vehicles (MAVs), particularly quadrotors, have shown significant potential in assisting humans with tasks such as construction and package delivery. These applications benefit greatly from the use of cables for manipulation mechanisms due to their lightweight, low-cost, and simple design. However, designing effective control and planning strategies for cable-suspended systems presents several challenges, including indirect load actuation, nonlinear configuration space, and highly coupled system dynamics. In this paper, we introduce a novel event-triggered distributed Nonlinear Model Predictive Control (NMPC) method specifically designed for cooperative transportation involving multiple quadrotors manipulating a cable-suspended payload. This approach addresses key challenges such as payload manipulation, inter-robot separation, obstacle avoidance, and trajectory tracking, all while optimizing the use of computational and communication resources. By integrating an event-triggered mechanism, our NMPC method reduces unnecessary computations and communication, enhancing energy efficiency and extending the operational range of MAVs. The proposed method employs a lightweight state vector parametrization that focuses on payload states in all six degrees of freedom, enabling efficient planning of trajectories on the SE(3) manifold. This not only reduces planning complexity but also ensures real-time computational feasibility. Our approach is validated through extensive simulation, demonstrating its efficacy in dynamic and resource-constrained environments. 

**Abstract (ZH)**: 自主微空中车辆（MAVs），特别是四旋翼飞行器，在协助人类完成建筑和包裹配送任务方面展现出显著潜力。这些应用由于电缆操纵机制的轻量化、低成本和简易设计而获益良多。然而，为电缆悬吊系统设计有效的控制和规划策略面临着诸多挑战，包括间接负载操作、非线性配置空间和高度耦合的动力学特性。本文介绍了一种新颖的事件触发分布式非线性模型预测控制（NMPC）方法，专门用于多四旋翼飞行器协同运输电缆悬吊负载的任务。该方法解决了负载操作、机器人间距离控制、障碍物避让和轨迹跟踪等关键挑战，同时优化了计算和通信资源的使用。通过集成事件触发机制，我们的NMPC方法降低了不必要的计算和通信量，增强了能量效率并扩展了MAVs的操作范围。所提出的方法采用轻量级状态向量参数化，专注于所有六个自由度的负载状态，在SE(3)流形上进行轨迹规划，这不仅减少了规划复杂性，还确保了实时计算的可行性。通过广泛的仿真实验，我们的方法在动态和资源受限的环境中展现了其有效性。 

---
# Distributed Resilience-Aware Control in Multi-Robot Networks 

**Title (ZH)**: 多机器人网络中的分布式鲁棒性aware控制 

**Authors**: Haejoon Lee, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2504.03120)  

**Abstract**: Ensuring resilient consensus in multi-robot systems with misbehaving agents remains a challenge, as many existing network resilience properties are inherently combinatorial and globally defined. While previous works have proposed control laws to enhance or preserve resilience in multi-robot networks, they often assume a fixed topology with known resilience properties, or require global state knowledge. These assumptions may be impractical in physically-constrained environments, where safety and resilience requirements are conflicting, or when misbehaving agents corrupt the shared information. In this work, we propose a distributed control law that enables each robot to guarantee resilient consensus and safety during its navigation without fixed topologies using only locally available information. To this end, we establish a new sufficient condition for resilient consensus in time-varying networks based on the degree of non-misbehaving or normal agents. Using this condition, we design a Control Barrier Function (CBF)-based controller that guarantees resilient consensus and collision avoidance without requiring estimates of global state and/or control actions of all other robots. Finally, we validate our method through simulations. 

**Abstract (ZH)**: 在具有异常行为代理的多机器人系统中确保复呡共识依然是一项挑战，因为许多现有的网络鲁棒性属性本质上是组合性和全局定义的。尽管此前已有工作提出了增强或保持多机器人网络鲁棒性的控制律，但这些方法往往假设具有固定且已知鲁棒性属性的拓扑结构，或者需要全局状态信息。在物理约束环境下，这些假设可能不切实际，因为安全性和鲁棒性要求可能存在冲突，或者当异常行为代理破坏共享信息时。在本工作中，我们提出了一种分布式的控制律，使得每个机器人能够在不依赖固定拓扑结构的情况下，仅通过局部可用信息来保证鲁棒共识和安全性。为此，我们基于正常或非异常行为代理的度建立了一个新的鲁棒共识充分条件。基于此条件，我们设计了一个基于控制屏障函数（CBF）的控制器，能够保证鲁棒共识和碰撞避免，而无需估计所有其他机器人的全局状态和/或控制动作。最后，我们通过仿真实验验证了该方法。 

---
# What People Share With a Robot When Feeling Lonely and Stressed and How It Helps Over Time 

**Title (ZH)**: 当人们感到孤独和压力时与机器人分享的内容及这些分享如何随时间帮助他们 

**Authors**: Guy Laban, Sophie Chiang, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2504.02991)  

**Abstract**: Loneliness and stress are prevalent among young adults and are linked to significant psychological and health-related consequences. Social robots may offer a promising avenue for emotional support, especially when considering the ongoing advancements in conversational AI. This study investigates how repeated interactions with a social robot influence feelings of loneliness and perceived stress, and how such feelings are reflected in the themes of user disclosures towards the robot. Participants engaged in a five-session robot-led intervention, where a large language model powered QTrobot facilitated structured conversations designed to support cognitive reappraisal. Results from linear mixed-effects models show significant reductions in both loneliness and perceived stress over time. Additionally, semantic clustering of 560 user disclosures towards the robot revealed six distinct conversational themes. Results from a Kruskal-Wallis H-test demonstrate that participants reporting higher loneliness and stress more frequently engaged in socially focused disclosures, such as friendship and connection, whereas lower distress was associated with introspective and goal-oriented themes (e.g., academic ambitions). By exploring both how the intervention affects well-being, as well as how well-being shapes the content of robot-directed conversations, we aim to capture the dynamic nature of emotional support in huma-robot interaction. 

**Abstract (ZH)**: 年轻人中普遍存在孤独和压力，并与重要的心理和健康后果相关。社交机器人可能为情感支持提供有希望的途径，特别是在考虑对话型AI的持续进步时。本研究探讨了重复与社交机器人的互动如何影响孤独和感知到的压力感，以及这些感觉如何反映在用户对机器人的披露主题中。参与者参与了一个由五个会话组成的机器人引导干预，其中由配备大型语言模型的QTrobot主导的结构化对话旨在支持认知重评。线性混合效应模型的结果显示，随着时间的推移，孤独和感知到的压力都有显著减少。此外，对560个用户对机器人的披露进行语义聚类，揭示了六种不同的对话主题。霍奇森-K武克尔曼检验的结果表明，报告更高孤独和压力的参与者更频繁地进行了以社交为重点的披露，例如友谊和连接，而较低的压力则与内省和目标导向的主题相关（如学术抱负）。通过探索干预如何影响福祉以及福祉如何塑造机器人导向对话的内容，我们旨在捕捉人类-机器人互动中情感支持的动态性质。 

---
# Distributionally Robust Predictive Runtime Verification under Spatio-Temporal Logic Specifications 

**Title (ZH)**: 基于时空逻辑规范的分布鲁棒预测运行时验证 

**Authors**: Yiqi Zhao, Emily Zhu, Bardh Hoxha, Georgios Fainekos, Jyotirmoy V. Deshmukh, Lars Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.02964)  

**Abstract**: Cyber-physical systems designed in simulators, often consisting of multiple interacting agents, behave differently in the real-world. We would like to verify these systems during runtime when they are deployed. Thus, we propose robust predictive runtime verification (RPRV) algorithms for: (1) general stochastic CPS under signal temporal logic (STL) tasks, and (2) stochastic multi-agent systems (MAS) under spatio-temporal logic tasks. The RPRV problem presents the following challenges: (1) there may not be sufficient data on the behavior of the deployed CPS, (2) predictive models based on design phase system trajectories may encounter distribution shift during real-world deployment, and (3) the algorithms need to scale to the complexity of MAS and be applicable to spatio-temporal logic tasks. To address these challenges, we assume knowledge of an upper bound on the statistical distance (in terms of an f-divergence) between the trajectory distributions of the system at deployment and design time. We are motivated by our prior work [1, 2] where we proposed an accurate and an interpretable RPRV algorithm for general CPS, which we here extend to the MAS setting and spatio-temporal logic tasks. Specifically, we use a learned predictive model to estimate the system behavior at runtime and robust conformal prediction to obtain probabilistic guarantees by accounting for distribution shifts. Building on [1], we perform robust conformal prediction over the robust semantics of spatio-temporal reach and escape logic (STREL) to obtain centralized RPRV algorithms for MAS. We empirically validate our results in a drone swarm simulator, where we show the scalability of our RPRV algorithms to MAS and analyze the impact of different trajectory predictors on the verification result. To the best of our knowledge, these are the first statistically valid algorithms for MAS under distribution shift. 

**Abstract (ZH)**: 针对分布偏移的鲁棒预测运行时验证算法：面向随机物理系统与空间-时间逻辑任务 

---
# Curvature-Constrained Vector Field for Motion Planning of Nonholonomic Robots 

**Title (ZH)**: 非完整机器人运动规划的曲率约束矢量场 

**Authors**: Yike Qiao, Xiaodong He, An Zhuo, Zhiyong Sun, Weimin Bao, Zhongkui Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.02852)  

**Abstract**: Vector fields are advantageous in handling nonholonomic motion planning as they provide reference orientation for robots. However, additionally incorporating curvature constraints becomes challenging, due to the interconnection between the design of the curvature-bounded vector field and the tracking controller under underactuation. In this paper, we present a novel framework to co-develop the vector field and the control laws, guiding the nonholonomic robot to the target configuration with curvature-bounded trajectory. First, we formulate the problem by introducing the target positive limit set, which allows the robot to converge to or pass through the target configuration, depending on different dynamics and tasks. Next, we construct a curvature-constrained vector field (CVF) via blending and distributing basic flow fields in workspace and propose the saturated control laws with a dynamic gain, under which the tracking error's magnitude decreases even when saturation occurs. Under the control laws, kinematically constrained nonholonomic robots are guaranteed to track the reference CVF and converge to the target positive limit set with bounded trajectory curvature. Numerical simulations show that the proposed CVF method outperforms other vector-field-based algorithms. Experiments on Ackermann UGVs and semi-physical fixed-wing UAVs demonstrate that the method can be effectively implemented in real-world scenarios. 

**Abstract (ZH)**: 非holonomic运动规划中带有 curvature 约束的向量场及其控制律的联合设计框架 

---
# A Class of Hierarchical Sliding Mode Control based on Extended Kalman filter for Quadrotor UAVs 

**Title (ZH)**: 基于扩展卡尔曼滤波的 quadrotor 无人机分层滑模控制方法 

**Authors**: Van Chung Nguyen, Hung Manh La  

**Link**: [PDF](https://arxiv.org/pdf/2504.02851)  

**Abstract**: This study introduces a novel methodology for controlling Quadrotor Unmanned Aerial Vehicles, focusing on Hierarchical Sliding Mode Control strategies and an Extended Kalman Filter. Initially, an EKF is proposed to enhance robustness in estimating UAV states, thereby reducing the impact of measured noises and external disturbances. By locally linearizing UAV systems, the EKF can mitigate the disadvantages of the Kalman filter and reduce the computational cost of other nonlinear observers. Subsequently, in comparison to other related work in terms of stability and computational cost, the HSMC framework shows its outperformance in allowing the quadrotor UAVs to track the references. Three types of HSMC Aggregated HSMC, Incremental HSMC, and Combining HSMC are investigated for their effectiveness in tracking reference trajectories. Moreover, the stability of the quadrotor UAVs is rigorously analyzed using the Lyapunov stability principle. Finally, experimental results and comparative analyses demonstrate the efficacy and feasibility of the proposed methodologies. 

**Abstract (ZH)**: 本研究介绍了一种新型的四旋翼无人机控制方法，专注于分层滑模控制策略和扩展卡尔曼滤波器。首先，提出了一种扩展卡尔曼滤波器（EKF）以增强对无人机状态估计的稳健性，从而减少测量噪声和外部干扰的影响。通过对无人机系统进行局部线性化，EKF可以缓解卡尔曼滤波器的缺点，并减少其他非线性观测器的计算成本。随后，从稳定性和计算成本的角度与其他相关工作进行对比，滑模控制（HSMC）框架展示了其在使四旋翼无人机跟踪参考轨迹方面的优越性能。研究了三种类型的HSMC：聚合滑模控制、增量滑模控制和综合滑模控制，以评估其跟踪参考轨迹的效果。此外，利用李雅普诺夫稳定性原理严谨分析了四旋翼无人机的稳定性。最后，实验结果和对比分析验证了所提方法的有效性和可行性。 

---
