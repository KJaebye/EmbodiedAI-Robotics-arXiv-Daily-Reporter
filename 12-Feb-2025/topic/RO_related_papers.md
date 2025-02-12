# DOGlove: Dexterous Manipulation with a Low-Cost Open-Source Haptic Force Feedback Glove 

**Title (ZH)**: DOGlove：低成本开源触觉力反馈手套的灵巧操作 

**Authors**: Han Zhang, Songbo Hu, Zhecheng Yuan, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.07730)  

**Abstract**: Dexterous hand teleoperation plays a pivotal role in enabling robots to achieve human-level manipulation dexterity. However, current teleoperation systems often rely on expensive equipment and lack multi-modal sensory feedback, restricting human operators' ability to perceive object properties and perform complex manipulation tasks. To address these limitations, we present DOGlove, a low-cost, precise, and haptic force feedback glove system for teleoperation and manipulation. DoGlove can be assembled in hours at a cost under 600 USD. It features a customized joint structure for 21-DoF motion capture, a compact cable-driven torque transmission mechanism for 5-DoF multidirectional force feedback, and a linear resonate actuator for 5-DoF fingertip haptic feedback. Leveraging action and haptic force retargeting, DOGlove enables precise and immersive teleoperation of dexterous robotic hands, achieving high success rates in complex, contact-rich tasks. We further evaluate DOGlove in scenarios without visual feedback, demonstrating the critical role of haptic force feedback in task performance. In addition, we utilize the collected demonstrations to train imitation learning policies, highlighting the potential and effectiveness of DOGlove. DOGlove's hardware and software system will be fully open-sourced at this https URL. 

**Abstract (ZH)**: 灵巧手远程操作对于使机器人实现人类级别的操作灵巧性发挥着关键作用。然而，当前的远程操作系统往往依赖昂贵的设备，并缺乏多模态感官反馈，限制了操作者感知物体属性和执行复杂操作任务的能力。为了解决这些局限性，我们提出了一种低成本、精准且具有触觉力反馈的手套系统——DOGlove。DOGlove可在不到600美元的成本下，数小时内组装完成。它配备了21-自由度运动捕捉的定制关节结构、用于5-自由度多向力反馈的紧凑型电缆驱动扭矩传输机制，以及用于5-自由度指尖触觉反馈的线性共振执行器。借助动作和触觉力反馈重定位，DOGlove实现了对灵巧机器人手的精确且沉浸式的远程操作，在复杂的、接触丰富的任务中取得了高成功率。此外，我们还在无视觉反馈的场景中评估了DOGlove，展示了触觉力反馈在任务性能中的关键作用。同时，我们利用收集的演示数据训练了模仿学习策略，突显了DOGlove的潜力和有效性。DOGlove的硬件和软件系统将在以下网址开源：这个https://链接。 

---
# GaRLIO: Gravity enhanced Radar-LiDAR-Inertial Odometry 

**Title (ZH)**: GaRLIO: 重力增强的雷达-LiDAR-惯性里程计 

**Authors**: Chiyun Noh, Wooseong Yang, Minwoo Jung, Sangwoo Jung, Ayoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.07703)  

**Abstract**: Recently, gravity has been highlighted as a crucial constraint for state estimation to alleviate potential vertical drift. Existing online gravity estimation methods rely on pose estimation combined with IMU measurements, which is considered best practice when direct velocity measurements are unavailable. However, with radar sensors providing direct velocity data-a measurement not yet utilized for gravity estimation-we found a significant opportunity to improve gravity estimation accuracy substantially. GaRLIO, the proposed gravity-enhanced Radar-LiDAR-Inertial Odometry, can robustly predict gravity to reduce vertical drift while simultaneously enhancing state estimation performance using pointwise velocity measurements. Furthermore, GaRLIO ensures robustness in dynamic environments by utilizing radar to remove dynamic objects from LiDAR point clouds. Our method is validated through experiments in various environments prone to vertical drift, demonstrating superior performance compared to traditional LiDAR-Inertial Odometry methods. We make our source code publicly available to encourage further research and development. this https URL 

**Abstract (ZH)**: 最近，重力被突出强调为状态估计中的关键约束，以缓解潜在的垂直漂移。现有的在线重力估计方法依赖于姿态估计结合IMU测量，这在直接速度测量不可用时被认为是最佳实践。然而，随着雷达传感器提供直接速度数据——这一测量尚未用于重力估计——我们发现了一个大幅提高重力估计精度的重大机会。GaRLIO，所提出的重力增强雷达-lidar-惯性里程计，能够稳健地预测重力以减少垂直漂移，同时利用点速度测量提升状态估计性能。此外，GaRLIO通过利用雷达去除LiDAR点云中的动态对象，确保在动态环境中的鲁棒性。我们的方法通过在各种容易发生垂直漂移的环境中进行的实验得到验证，显示出与传统lidar-惯性里程计方法相比的优越性能。我们公开了我们的源代码，以促进进一步的研究和开发。this https URL 

---
# Dual Arm Steering of Deformable Linear Objects in 2-D and 3-D Environments Using Euler's Elastica Solutions 

**Title (ZH)**: 使用欧拉 elastica 解决方案在二维和三维环境中的可变形线性物体的双臂操纵 

**Authors**: Aharon Levin, Itay Grinberg, Elon Rimon, Amir Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2502.07509)  

**Abstract**: This paper describes a method for steering deformable linear objects using two robot hands in environments populated by sparsely spaced obstacles. The approach involves manipulating an elastic inextensible rod by varying the gripping endpoint positions and tangents. Closed form solutions that describe the flexible linear object shape in planar environments, Euler's elastica, are described. The paper uses these solutions to formulate criteria for non self-intersection, stability and obstacle avoidance. These criteria are formulated as constraints in the flexible object six-dimensional configuration space that represents the robot gripping endpoint positions and tangents. In particular, this paper introduces a novel criterion that ensures the flexible object stability during steering. All safety criteria are integrated into a scheme for steering flexible linear objects in planar environments, which is lifted into a steering scheme in three-dimensional environments populated by sparsely spaced obstacles. Experiments with a dual-arm robot demonstrate the method. 

**Abstract (ZH)**: 本文描述了一种在稀疏障碍环境中使用两只机器人手操控可变形线性物体的方法。该方法通过改变夹持端点位置和切线来操纵弹性的不可伸长杆。在平面环境中，文章描述了描述柔性线性物体形状的闭式解Euler's elastica。这些解用于制定非自交、稳定性和避障的判断标准。这些判断标准被形式化为柔性物体六维配置空间中的约束条件，该空间代表了机器人夹持端点位置和切线。特别地，本文引入了一个新的判断标准，以确保柔性物体在操控过程中的稳定性。所有安全判断标准被整合到一个平面环境中操控柔性线性物体的方案中，并被提升到一个三维环境中稀疏障碍物存在的操控方案中。实验使用双臂机器人验证了该方法。 

---
# Robotic In-Hand Manipulation for Large-Range Precise Object Movement: The RGMC Champion Solution 

**Title (ZH)**: 手持机器人在大范围精确物体 manipulation中的冠军solution：RGMC 方法 

**Authors**: Mingrui Yu, Yongpeng Jiang, Chen Chen, Yongyi Jia, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.07472)  

**Abstract**: In-hand manipulation using multiple dexterous fingers is a critical robotic skill that can reduce the reliance on large arm motions, thereby saving space and energy. This letter focuses on in-grasp object movement, which refers to manipulating an object to a desired pose through only finger motions within a stable grasp. The key challenge lies in simultaneously achieving high precision and large-range movements while maintaining a constant stable grasp. To address this problem, we propose a simple and practical approach based on kinematic trajectory optimization with no need for pretraining or object geometries, which can be easily applied to novel objects in real-world scenarios. Adopting this approach, we won the championship for the in-hand manipulation track at the 9th Robotic Grasping and Manipulation Competition (RGMC) held at ICRA 2024. Implementation details, discussion, and further quantitative experimental results are presented in this letter, which aims to comprehensively evaluate our approach and share our key takeaways from the competition. Supplementary materials including video and code are available at this https URL . 

**Abstract (ZH)**: 基于多灵巧手指的手在把握中的操作是一项关键的机器人技能，可以减少对大臂运动的依赖，从而节省空间和能量。本信关注握持中的对象移动，即仅通过手指运动在稳定握持下将对象移至 desired pose 的操作。关键挑战在于同时实现高精度和大范围运动，同时保持恒定的稳定握持。为解决这一问题，我们提出了一种基于运动学轨迹优化的简单实用方法，该方法无需进行预训练或对象几何形状的建模，并且可以轻松应用于现实世界中的新型对象。采用此方法，我们在第九届机器人抓取与操作竞赛（RGMC）ICRA 2024 竞赛的手在把握轨道中赢得了冠军。本信中介绍了该方法的实现细节、讨论和进一步的定量实验结果，旨在全面评估我们的方法并分享我们在竞赛中的关键见解。有关补充材料包括视频和代码可在以下链接获取。 

---
# Demonstrating Wheeled Lab: Modern Sim2Real for Low-cost, Open-source Wheeled Robotics 

**Title (ZH)**: 演示轮式实验室：面向低成本、开源轮式机器人的现代Sim2Real技术 

**Authors**: Tyler Han, Preet Shah, Sidharth Rajagopal, Yanda Bao, Sanghun Jung, Sidharth Talia, Gabriel Guo, Bryan Xu, Bhaumik Mehta, Emma Romig, Rosario Scalise, Byron Boots  

**Link**: [PDF](https://arxiv.org/pdf/2502.07380)  

**Abstract**: Simulation has been pivotal in recent robotics milestones and is poised to play a prominent role in the field's future. However, recent robotic advances often rely on expensive and high-maintenance platforms, limiting access to broader robotics audiences. This work introduces Wheeled Lab, a framework for the low-cost, open-source wheeled platforms that are already widely established in education and research. Through integration with Isaac Lab, Wheeled Lab introduces modern techniques in Sim2Real, such as domain randomization, sensor simulation, and end-to-end learning, to new user communities. To kickstart education and demonstrate the framework's capabilities, we develop three state-of-the-art policies for small-scale RC cars: controlled drifting, elevation traversal, and visual navigation, each trained in simulation and deployed in the real world. By bridging the gap between advanced Sim2Real methods and affordable, available robotics, Wheeled Lab aims to democratize access to cutting-edge tools, fostering innovation and education in a broader robotics context. The full stack, from hardware to software, is low cost and open-source. 

**Abstract (ZH)**: Wheeled Lab: 一种低成本开源轮式平台框架及其在Sim2Real中的应用 

---
# Leader-follower formation enabled by pressure sensing in free-swimming undulatory robotic fish 

**Title (ZH)**: 基于压力感知的自由游动波动型机器人鱼的领导者-跟随者群体形成立体armor 

**Authors**: Kundan Panta, Hankun Deng, Micah DeLattre, Bo Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.07282)  

**Abstract**: Fish use their lateral lines to sense flows and pressure gradients, enabling them to detect nearby objects and organisms. Towards replicating this capability, we demonstrated successful leader-follower formation swimming using flow pressure sensing in our undulatory robotic fish ($\mu$Bot/MUBot). The follower $\mu$Bot is equipped at its head with bilateral pressure sensors to detect signals excited by both its own and the leader's movements. First, using experiments with static formations between an undulating leader and a stationary follower, we determined the formation that resulted in strong pressure variations measured by the follower. This formation was then selected as the desired formation in free swimming for obtaining an expert policy. Next, a long short-term memory neural network was used as the control policy that maps the pressure signals along with the robot motor commands and the Euler angles (measured by the onboard IMU) to the steering command. The policy was trained to imitate the expert policy using behavior cloning and Dataset Aggregation (DAgger). The results show that with merely two bilateral pressure sensors and less than one hour of training data, the follower effectively tracked the leader within distances of up to 200 mm (= 1 body length) while swimming at speeds of 155 mm/s (= 0.8 body lengths/s). This work highlights the potential of fish-inspired robots to effectively navigate fluid environments and achieve formation swimming through the use of flow pressure feedback. 

**Abstract (ZH)**: 鱼类利用侧线感知水流和压力梯度，以检测附近的物体和生物。为复制这一能力，我们展示了使用流动压力感知实现波状机器人鱼（$\mu$Bot/MUBot）中的领先者-跟随者队形游泳。跟随者$\mu$Bot在其头部配备了双侧压力传感器，以检测自身和领先者运动引起的信号。首先，通过静止队形实验，即波状运动的领先者和静止的跟随者之间的队形，我们确定了能引起强压力变化的队形。然后，将此队形作为自由游泳状态下所需的队形，以获得专家策略。接下来，使用长短期记忆神经网络作为控制策略，将压力信号与机器人电机指令和IMU测量的欧拉角映射到转向指令。该策略使用行为克隆和数据集聚合（DAgger）训练以模仿专家策略。结果显示，仅使用两个双侧压力传感器和少于一小时的训练数据下，跟随者在距领先者200毫米（相当于1个身体长度）之内并在155毫米/秒（0.8个身体长度/秒）的速度下有效跟踪领先者。本工作突显了仿鱼机器人通过流动压力反馈在流体环境中有效导航并实现队形游泳的潜力。 

---
# Parameter Optimization of Optical Six-Axis Force/Torque Sensor for Legged Robots 

**Title (ZH)**: 基于腿式机器人光六轴力/力矩传感器参数优化 

**Authors**: Hyun-Bin Kim, Byeong-Il Ham, Keun-Ha Choi, Kyung-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.07196)  

**Abstract**: This paper introduces a novel six-axis force/torque sensor tailored for compact and lightweight legged robots. Unlike traditional strain gauge-based sensors, the proposed non-contact design employs photocouplers, enhancing resistance to physical impacts and reducing damage risk. This approach simplifies manufacturing, lowers costs, and meets the demands of legged robots by combining small size, light weight, and a wide force measurement range. A methodology for optimizing sensor parameters is also presented, focusing on maximizing sensitivity and minimizing error. Precise modeling and analysis of objective functions enabled the derivation of optimal design parameters. The sensor's performance was validated through extensive testing and integration into quadruped robots, demonstrating alignment with theoretical modeling. The sensor's precise measurement capabilities make it suitable for diverse robotic environments, particularly in analyzing interactions between robot feet and the ground. This innovation addresses existing sensor limitations while contributing to advancements in robotics and sensor technology, paving the way for future applications in robotic systems. 

**Abstract (ZH)**: 一种适用于紧凑轻型腿足机器人的一体化六轴力/力矩传感器及其优化设计方法 

---
# A Safe Hybrid Control Framework for Car-like Robot with Guaranteed Global Path-Invariance using a Control Barrier Function 

**Title (ZH)**: 一种使用控制障碍函数保证全局路径不变性的越野车样机器人混合控制框架 

**Authors**: Nan Wang, Adeel Akhtar, Ricardo G. Sanfelice  

**Link**: [PDF](https://arxiv.org/pdf/2502.07136)  

**Abstract**: This work proposes a hybrid framework for car-like robots with obstacle avoidance, global convergence, and safety, where safety is interpreted as path invariance, namely, once the robot converges to the path, it never leaves the path. Given a priori obstacle-free feasible path where obstacles can be around the path, the task is to avoid obstacles while reaching the path and then staying on the path without leaving it. The problem is solved in two stages. Firstly, we define a ``tight'' obstacle-free neighborhood along the path and design a local controller to ensure convergence to the path and path invariance. The control barrier function technology is involved in the control design to steer the system away from its singularity points, where the local path invariant controller is not defined. Secondly, we design a hybrid control framework that integrates this local path-invariant controller with any global tracking controller from the existing literature without path invariance guarantee, ensuring convergence from any position to the desired path, namely, global convergence. This framework guarantees path invariance and robustness to sensor noise. Detailed simulation results affirm the effectiveness of the proposed scheme. 

**Abstract (ZH)**: 一种兼顾障碍 Avoidance、全局收敛与路径不变性的混合框架：该框架确保一旦机器人收敛于路径，它将始终保持在该路径上，无论路径周围是否存在障碍物。首先定义路径的“紧凑无障碍”邻域，并设计局部控制器以确保路径收敛与路径不变性。其次，将此局部路径不变控制器与现有文献中的任意全局跟踪控制器结合，确保从任意位置到目标路径的全局收敛性，同时保证路径不变性和对传感器噪声的鲁棒性。详细仿真结果验证了所提方案的有效性。 

---
# Cross-platform Learning-based Fault Tolerant Surfacing Controller for Underwater Robots 

**Title (ZH)**: 跨平台基于学习的容错 Surfacing 控制器 for 水下机器人 

**Authors**: Yuya Hamamatsu, Walid Remmas, Jaan Rebane, Maarja Kruusmaa, Asko Ristolainen  

**Link**: [PDF](https://arxiv.org/pdf/2502.07133)  

**Abstract**: In this paper, we propose a novel cross-platform fault-tolerant surfacing controller for underwater robots, based on reinforcement learning (RL). Unlike conventional approaches, which require explicit identification of malfunctioning actuators, our method allows the robot to surface using only the remaining operational actuators without needing to pinpoint the failures. The proposed controller learns a robust policy capable of handling diverse failure scenarios across different actuator configurations. Moreover, we introduce a transfer learning mechanism that shares a part of the control policy across various underwater robots with different actuators, thus improving learning efficiency and generalization across platforms. To validate our approach, we conduct simulations on three different types of underwater robots: a hovering-type AUV, a torpedo shaped AUV, and a turtle-shaped robot (U-CAT). Additionally, real-world experiments are performed, successfully transferring the learned policy from simulation to a physical U-CAT in a controlled environment. Our RL-based controller demonstrates superior performance in terms of stability and success rate compared to a baseline controller, achieving an 85.7 percent success rate in real-world tests compared to 57.1 percent with a baseline controller. This research provides a scalable and efficient solution for fault-tolerant control for diverse underwater platforms, with potential applications in real-world aquatic missions. 

**Abstract (ZH)**: 基于强化学习的跨平台容错水面控制装置 

---
# Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects 

**Title (ZH)**: 几何感知的_rl_在变形和变化形状物体 manipulation 中的应用 

**Authors**: Tai Hoang, Huy Le, Philipp Becker, Vien Anh Ngo, Gerhard Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2502.07005)  

**Abstract**: Manipulating objects with varying geometries and deformable objects is a major challenge in robotics. Tasks such as insertion with different objects or cloth hanging require precise control and effective modelling of complex dynamics. In this work, we frame this problem through the lens of a heterogeneous graph that comprises smaller sub-graphs, such as actuators and objects, accompanied by different edge types describing their interactions. This graph representation serves as a unified structure for both rigid and deformable objects tasks, and can be extended further to tasks comprising multiple actuators. To evaluate this setup, we present a novel and challenging reinforcement learning benchmark, including rigid insertion of diverse objects, as well as rope and cloth manipulation with multiple end-effectors. These tasks present a large search space, as both the initial and target configurations are uniformly sampled in 3D space. To address this issue, we propose a novel graph-based policy model, dubbed Heterogeneous Equivariant Policy (HEPi), utilizing $SE(3)$
equivariant message passing networks as the main backbone to exploit the geometric symmetry. In addition, by modeling explicit heterogeneity, HEPi can outperform Transformer-based and non-heterogeneous equivariant policies in terms of average returns, sample efficiency, and generalization to unseen objects. 

**Abstract (ZH)**: 操纵几何形状各异的对象和可变形物体是机器人技术中的一个重大挑战。插入不同物体或布料悬挂等任务需要精确控制和有效的复杂动力学建模。在本工作中，我们通过包含较小子图的异质图框架来阐述这一问题，这些子图如效应器和对象，并伴有描述其相互作用的不同边类型。这种图表示形式为刚性及可变形物体任务提供了一个统一结构，并可进一步扩展至包含多个效应器的任务。为了评估此设置，我们提出了一种新型且具有挑战性的强化学习基准，包括多种物体的刚性插入以及多末端执行器操作绳子和布料。这些任务具有较大的搜索空间，因为初始和目标配置在3D空间中随机采样。为解决这一问题，我们提出了一种基于图的策略模型，名为异质等变策略（HEPi），利用$SE(3)$等变消息传递网络作为主要骨干以利用几何对称性。此外，通过建模显式的异质性，HEPi 在平均回报、样本效率以及对未见过的对象的泛化能力方面优于基于Transformer和非异质等变策略。 

---
