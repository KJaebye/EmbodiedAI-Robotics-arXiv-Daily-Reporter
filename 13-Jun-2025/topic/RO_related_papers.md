# Vib2Move: In-Hand Object Reconfiguration via Fingertip Micro-Vibrations 

**Title (ZH)**: Vib2Move：通过指尖微振动实现手内物体重构 

**Authors**: Xili Yi, Nima Fazeli  

**Link**: [PDF](https://arxiv.org/pdf/2506.10923)  

**Abstract**: We introduce Vib2Move, a novel approach for in-hand object reconfiguration that uses fingertip micro-vibrations and gravity to precisely reposition planar objects. Our framework comprises three key innovations. First, we design a vibration-based actuator that dynamically modulates the effective finger-object friction coefficient, effectively emulating changes in gripping force. Second, we derive a sliding motion model for objects clamped in a parallel gripper with two symmetric, variable-friction contact patches. Third, we propose a motion planner that coordinates end-effector finger trajectories and fingertip vibrations to achieve the desired object pose. In real-world trials, Vib2Move consistently yields final positioning errors below 6 mm, demonstrating reliable, high-precision manipulation across a variety of planar objects. For more results and information, please visit this https URL. 

**Abstract (ZH)**: Vib2Move：一种基于指尖微振动和重力的在手物体重构新方法 

---
# Invariant Extended Kalman Filter for Autonomous Surface Vessels with Partial Orientation Measurements 

**Title (ZH)**: 部分姿态测量的自治水面船舶不变扩展卡尔曼滤波器 

**Authors**: Derek Benham, Easton Potokar, Joshua G. Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2506.10850)  

**Abstract**: Autonomous surface vessels (ASVs) are increasingly vital for marine science, offering robust platforms for underwater mapping and inspection. Accurate state estimation, particularly of vehicle pose, is paramount for precise seafloor mapping, as even small surface deviations can have significant consequences when sensing the seafloor below. To address this challenge, we propose an Invariant Extended Kalman Filter (InEKF) framework designed to integrate partial orientation measurements. While conventional estimation often relies on relative position measurements to fixed landmarks, open ocean ASVs primarily observe a receding horizon. We leverage forward-facing monocular cameras to estimate roll and pitch with respect to this horizon, which provides yaw-ambiguous partial orientation information. To effectively utilize these measurements within the InEKF, we introduce a novel framework for incorporating such partial orientation data. This approach contrasts with traditional InEKF implementations that assume full orientation measurements and is particularly relevant for planar vehicle motion constrained to a "seafaring plane." This paper details the developed InEKF framework; its integration with horizon-based roll/pitch observations and dual-antenna GPS heading measurements for ASV state estimation; and provides a comparative analysis against the InEKF using full orientation and a Multiplicative EKF (MEKF). Our results demonstrate the efficacy and robustness of the proposed partial orientation measurements for accurate ASV state estimation in open ocean environments. 

**Abstract (ZH)**: 自主水面船（ASVs）在海洋科学中日益重要，提供了一种用于水下测绘和检查的坚稳平台。精确的状态估计，尤其是车辆姿态的估计，对于精确的海底测绘至关重要，因为即使是很小的表面偏差也可能对海底传感产生重大影响。为了应对这一挑战，我们提出了一种不变广义卡尔曼滤波器（InEKF）框架，用于整合部分姿态测量。传统的估计方法通常依赖于相对于固定陆标的位置测量，而开阔海域中的ASVs主要观察的是逐渐远离的天际线。我们利用面向前方的单目摄像头来估计相对于天际线的横滚角和俯仰角，从而获得具有航向不确定性的一部分姿态信息。为了有效利用这些测量值在InEKF框架中，我们提出了一种新的整合部分姿态数据的框架。这一方法与传统的假设具有完整姿态测量的InEKF实现不同，并且特别适用于平面车辆运动受限于“航海平面”的情况。本文详细介绍了所开发的InEKF框架；其与基于天际线的横滚/俯仰观察以及双天线GPS航向测量的集成，用于ASV状态估计；并且提供了与具有完整姿态的InEKF和乘法卡尔曼滤波器（MEKF）的比较分析。我们的结果证明了所提出的部分姿态测量在开阔海域中进行准确的ASV状态估计的有效性和鲁棒性。 

---
# An $O(n$)-Algorithm for the Higher-Order Kinematics and Inverse Dynamics of Serial Manipulators using Spatial Representation of Twists 

**Title (ZH)**: 一种基于刚体运动表示的串联 manipulator 的高阶运动学和逆动力学的 O(n) 算法 

**Authors**: Andreas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2506.10686)  

**Abstract**: Optimal control in general, and flatness-based control in particular, of robotic arms necessitate to compute the first and second time derivatives of the joint torques/forces required to achieve a desired motion. In view of the required computational efficiency, recursive $O(n)$-algorithms were proposed to this end. Aiming at compact yet efficient formulations, a Lie group formulation was recently proposed, making use of body-fixed and hybrid representation of twists and wrenches. In this paper a formulation is introduced using the spatial representation. The second-order inverse dynamics algorithm is accompanied by a fourth-order forward and inverse kinematics algorithm. An advantage of all Lie group formulations is that they can be parameterized in terms of vectorial quantities that are readily available. The method is demonstrated for the 7 DOF Franka Emika Panda robot. 

**Abstract (ZH)**: 基于空间表示的冗余自由度机械臂的最优控制及基于平坦性控制 

---
# RICE: Reactive Interaction Controller for Cluttered Canopy Environment 

**Title (ZH)**: 稻草人控制器：杂乱植被环境下的反应性交互控制 

**Authors**: Nidhi Homey Parayil, Thierry Peynot, Chris Lehnert  

**Link**: [PDF](https://arxiv.org/pdf/2506.10383)  

**Abstract**: Robotic navigation in dense, cluttered environments such as agricultural canopies presents significant challenges due to physical and visual occlusion caused by leaves and branches. Traditional vision-based or model-dependent approaches often fail in these settings, where physical interaction without damaging foliage and branches is necessary to reach a target. We present a novel reactive controller that enables safe navigation for a robotic arm in a contact-rich, cluttered, deformable environment using end-effector position and real-time tactile feedback. Our proposed framework's interaction strategy is based on a trade-off between minimizing disturbance by maneuvering around obstacles and pushing through them to move towards the target. We show that over 35 trials in 3 experimental plant setups with an occluded target, the proposed controller successfully reached the target in all trials without breaking any branch and outperformed the state-of-the-art model-free controller in robustness and adaptability. This work lays the foundation for safe, adaptive interaction in cluttered, contact-rich deformable environments, enabling future agricultural tasks such as pruning and harvesting in plant canopies. 

**Abstract (ZH)**: 密集遮蔽环境中基于机器人的导航研究：面向农业冠层的鲁棒适应性交互控制 

---
# A Novel Feedforward Youla Parameterization Method for Avoiding Local Minima in Stereo Image Based Visual Servoing Control 

**Title (ZH)**: 基于立体图像视觉伺服控制中避免局部极小值的新型前馈Youla参数化方法 

**Authors**: Rongfei Li, Francis Assadian  

**Link**: [PDF](https://arxiv.org/pdf/2506.10252)  

**Abstract**: In robot navigation and manipulation, accurately determining the camera's pose relative to the environment is crucial for effective task execution. In this paper, we systematically prove that this problem corresponds to the Perspective-3-Point (P3P) formulation, where exactly three known 3D points and their corresponding 2D image projections are used to estimate the pose of a stereo camera. In image-based visual servoing (IBVS) control, the system becomes overdetermined, as the 6 degrees of freedom (DoF) of the stereo camera must align with 9 observed 2D features in the scene. When more constraints are imposed than available DoFs, global stability cannot be guaranteed, as the camera may become trapped in a local minimum far from the desired configuration during servoing. To address this issue, we propose a novel control strategy for accurately positioning a calibrated stereo camera. Our approach integrates a feedforward controller with a Youla parameterization-based feedback controller, ensuring robust servoing performance. Through simulations, we demonstrate that our method effectively avoids local minima and enables the camera to reach the desired pose accurately and efficiently. 

**Abstract (ZH)**: 机器人导航与操作中，准确确定相机相对于环境的姿态对于有效执行任务至关重要。本文系统地证明了这一问题等同于基于三点视角（P3P）的求解方式，即利用三个已知的3D点及其对应的2D图像投影来估计立体相机的姿态。在基于图像的视觉伺服（IBVS）控制中，系统变得过定，因为立体相机的6个自由度必须与场景中观察到的9个2D特征对齐。当施加的约束条件超过可自由度时，全局稳定性不能得到保证，相机在伺服过程中可能会被困在远离期望配置的局部极小值中。为解决这一问题，我们提出了一种新的用于精确定位校准立体相机的控制策略。该方法将前馈控制器与基于Youla参数化的反馈控制器相结合，确保视觉伺服性能的鲁棒性。通过仿真实验，我们展示了该方法有效地避免了局部极小值，并使相机能够准确高效地达到期望姿态。 

---
# Innovative Adaptive Imaged Based Visual Servoing Control of 6 DoFs Industrial Robot Manipulators 

**Title (ZH)**: 基于图像的自适应视觉伺服控制的6轴工业机器人 manipulator创新技术 

**Authors**: Rongfei Li, Francis Assadian  

**Link**: [PDF](https://arxiv.org/pdf/2506.10240)  

**Abstract**: Image-based visual servoing (IBVS) methods have been well developed and used in many applications, especially in pose (position and orientation) alignment. However, most research papers focused on developing control solutions when 3D point features can be detected inside the field of view. This work proposes an innovative feedforward-feedback adaptive control algorithm structure with the Youla Parameterization method. A designed feature estimation loop ensures stable and fast motion control when point features are outside the field of view. As 3D point features move inside the field of view, the IBVS feedback loop preserves the precision of the pose at the end of the control period. Also, an adaptive controller is developed in the feedback loop to stabilize the system in the entire range of operations. The nonlinear camera and robot manipulator model is linearized and decoupled online by an adaptive algorithm. The adaptive controller is then computed based on the linearized model evaluated at current linearized point. The proposed solution is robust and easy to implement in different industrial robotic systems. Various scenarios are used in simulations to validate the effectiveness and robust performance of the proposed controller. 

**Abstract (ZH)**: 基于图像的视觉伺服（IBVS）方法已在许多应用中得到发展和使用，尤其是在姿态（位置和方向）对准方面。然而，大多数研究论文集中在可以在视野内检测到3D点特征时开发控制解决方案。本文提出了一个创新的前馈-反馈自适应控制算法结构，并采用了尤拉参数化方法。设计的特征估计环路确保了当点特征位于视野外时的稳定和快速运动控制。当3D点特征移动到视野内时，IBVS反馈环路在控制期末期保持姿态精度。此外，在反馈环路中开发了自适应控制器，以在操作的整个范围内稳定系统。非线性的相机和机器人 manipulator 模型在线上通过自适应算法进行线性化和解耦。然后基于在当前线性化点上评价的线性化模型计算自适应控制器。所提出的解决方案在不同工业机器人系统中具有鲁棒性和易于实现的特点。通过各种场景在 simulations 中验证了所提出控制器的有效性和鲁棒性能。 

---
# A Unified Framework for Probabilistic Dynamic-, Trajectory- and Vision-based Virtual Fixtures 

**Title (ZH)**: 一种统一框架：基于概率动态、轨迹和视觉的虚拟 fixtures 

**Authors**: Maximilian Mühlbauer, Freek Stulp, Sylvain Calinon, Alin Albu-Schäffer, João Silvério  

**Link**: [PDF](https://arxiv.org/pdf/2506.10239)  

**Abstract**: Probabilistic Virtual Fixtures (VFs) enable the adaptive selection of the most suitable haptic feedback for each phase of a task, based on learned or perceived uncertainty. While keeping the human in the loop remains essential, for instance, to ensure high precision, partial automation of certain task phases is critical for productivity. We present a unified framework for probabilistic VFs that seamlessly switches between manual fixtures, semi-automated fixtures (with the human handling precise tasks), and full autonomy. We introduce a novel probabilistic Dynamical System-based VF for coarse guidance, enabling the robot to autonomously complete certain task phases while keeping the human operator in the loop. For tasks requiring precise guidance, we extend probabilistic position-based trajectory fixtures with automation allowing for seamless human interaction as well as geometry-awareness and optimal impedance gains. For manual tasks requiring very precise guidance, we also extend visual servoing fixtures with the same geometry-awareness and impedance behaviour. We validate our approach experimentally on different robots, showcasing multiple operation modes and the ease of programming fixtures. 

**Abstract (ZH)**: 概率虚拟 fixtures (VFs) 允许根据学习到的或感知到的不确定性，在任务的各个阶段选择最合适的触觉反馈。尽管保持人类在环内对于确保高精度仍然是必要的，但对某些任务阶段的部分自动化对于提高生产力至关重要。我们提出了一种统一的概率虚拟 fixtures 框架，该框架可以在手动 fixtures、半自动化 fixtures（人类处理精确任务）和完全自主之间无缝切换。我们引入了一种基于概率动力系统的新颖虚拟 fixtures，用于粗略指导，使机器人能够在保持人类操作员在环内的同时自主完成某些任务阶段。对于需要精确指导的任务，我们扩展了基于概率位置的轨迹 fixtures，引入了自动化功能，使其能够无缝地与人类交互，并具备几何感知能力和最优阻抗增益。对于需要非常精确指导的手动任务，我们还扩展了视觉伺服 fixtures，使其具备相同的几何感知能力和阻抗行为。我们在不同的机器人上实验验证了我们的方法，展示了多种操作模式以及 fixtures 编程的简单性。 

---
# Impacts between multibody systems and deformable structures 

**Title (ZH)**: 多体系统与可变形结构之间的相互作用影响 

**Authors**: Lipinski Krzysztof  

**Link**: [PDF](https://arxiv.org/pdf/2506.10034)  

**Abstract**: Collisions and impacts are the principal reasons for impulsive motions, which we frequently see in dynamic responses of systems. Precise modelling of impacts is a challenging problem due to the lack of the accurate and commonly accepted constitutive law that governs their mechanics. Rigid-body approach and soft contact methods are discussed in this paper and examined in the presented numerical examples. The main focus is set to impacts in systems with multiple unilateral contacts and collisions with elastic elements of the reference. Parameters of interconnecting unilateral springs are under discussion. 

**Abstract (ZH)**: 碰撞和冲击是引起瞬态运动的主要原因，我们经常在系统的动态响应中见到它们。准确模拟能动体的碰撞是一个难题，主要原因是缺乏能够准确描述其力学规律的公认的本构关系。本文讨论了刚体方法和软接触方法，并通过示例数值分析进行了检验。重点放在涉及多个单向接触的系统及其与参考弹性元件的碰撞上，探讨了相互连接的单向弹簧参数。 

---
