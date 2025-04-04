# A Planning Framework for Stable Robust Multi-Contact Manipulation 

**Title (ZH)**: 一种稳定鲁棒多接触操作规划框架 

**Authors**: Lin Yang, Sri Harsha Turlapati, Zhuoyi Lu, Chen Lv, Domenico Campolo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02516)  

**Abstract**: While modeling multi-contact manipulation as a quasi-static mechanical process transitioning between different contact equilibria, we propose formulating it as a planning and optimization problem, explicitly evaluating (i) contact stability and (ii) robustness to sensor noise. Specifically, we conduct a comprehensive study on multi-manipulator control strategies, focusing on dual-arm execution in a planar peg-in-hole task and extending it to the Multi-Manipulator Multiple Peg-in-Hole (MMPiH) problem to explore increased task complexity. Our framework employs Dynamic Movement Primitives (DMPs) to parameterize desired trajectories and Black-Box Optimization (BBO) with a comprehensive cost function incorporating friction cone constraints, squeeze forces, and stability considerations. By integrating parallel scenario training, we enhance the robustness of the learned policies. To evaluate the friction cone cost in experiments, we test the optimal trajectories computed for various contact surfaces, i.e., with different coefficients of friction. The stability cost is analytical explained and tested its necessity in simulation. The robustness performance is quantified through variations of hole pose and chamfer size in simulation and experiment. Results demonstrate that our approach achieves consistently high success rates in both the single peg-in-hole and multiple peg-in-hole tasks, confirming its effectiveness and generalizability. The video can be found at this https URL. 

**Abstract (ZH)**: 将多点接触操作建模为准静态机械过程，在不同接触平衡之间进行转换，我们提出将其表述为一个规划和优化问题，明确评估（i）接触稳定性及（ii）对传感器噪声的鲁棒性。具体而言，我们对多 manipulator 控制策略进行了全面研究，重点关注平面孔配任务的双臂执行，并将其扩展到多 manipulator 多孔配问题（MMPiH），以探索任务复杂性的增加。我们的框架使用动态运动 primitives (DMPs) 参数化期望轨迹，并使用包含摩擦锥约束、挤压力和稳定性考虑的黑盒优化（BBO）方法。通过集成并行场景训练，增强学习策略的鲁棒性。为了在实验中评估摩擦锥成本，我们对各种接触表面（具有不同的摩擦系数）计算出的最优轨迹进行了测试。稳定性成本进行了详细的分析解释，并在仿真中测试了其必要性。鲁棒性性能通过仿真和实验中孔位姿和倒角尺寸的变化进行量化。结果表明，我们的方法在单孔配和多孔配任务中都实现了高一致的成功率，证实了其有效性和泛化能力。视频链接见此 https URL。 

---
# Adaptive path planning for efficient object search by UAVs in agricultural fields 

**Title (ZH)**: 适应性路径规划以实现农业田地内无人机高效物体搜索 

**Authors**: Rick van Essen, Eldert van Henten, Lammert Kooistra, Gert Kootstra  

**Link**: [PDF](https://arxiv.org/pdf/2504.02473)  

**Abstract**: This paper presents an adaptive path planner for object search in agricultural fields using UAVs. The path planner uses a high-altitude coverage flight path and plans additional low-altitude inspections when the detection network is uncertain. The path planner was evaluated in an offline simulation environment containing real-world images. We trained a YOLOv8 detection network to detect artificial plants placed in grass fields to showcase the potential of our path planner. We evaluated the effect of different detection certainty measures, optimized the path planning parameters, investigated the effects of localization errors and different numbers of objects in the field. The YOLOv8 detection confidence worked best to differentiate between true and false positive detections and was therefore used in the adaptive planner. The optimal parameters of the path planner depended on the distribution of objects in the field, when the objects were uniformly distributed, more low-altitude inspections were needed compared to a non-uniform distribution of objects, resulting in a longer path length. The adaptive planner proved to be robust against localization uncertainty. When increasing the number of objects, the flight path length increased, especially when the objects were uniformly distributed. When the objects were non-uniformly distributed, the adaptive path planner yielded a shorter path than a low-altitude coverage path, even with high number of objects. Overall, the presented adaptive path planner allowed to find non-uniformly distributed objects in a field faster than a coverage path planner and resulted in a compatible detection accuracy. The path planner is made available at this https URL. 

**Abstract (ZH)**: 基于UAV的农业田地目标搜索自适应路径规划方法 

---
# Bipedal Robust Walking on Uneven Footholds: Piecewise Slope LIPM with Discrete Model Predictive Control 

**Title (ZH)**: 双足稳健不平地面行走：分段斜率LIPM结合离散模型预测控制 

**Authors**: Yapeng Shi, Sishu Li, Yongqiang Wu, Junjie Liu, Xiaokun Leng, Xizhe Zang, Songhao Piao  

**Link**: [PDF](https://arxiv.org/pdf/2504.02255)  

**Abstract**: This study presents an enhanced theoretical formulation for bipedal hierarchical control frameworks under uneven terrain conditions. Specifically, owing to the inherent limitations of the Linear Inverted Pendulum Model (LIPM) in handling terrain elevation variations, we develop a Piecewise Slope LIPM (PS-LIPM). This innovative model enables dynamic adjustment of the Center of Mass (CoM) height to align with topographical undulations during single-step cycles. Another contribution is proposed a generalized Angular Momentum-based LIPM (G-ALIP) for CoM velocity compensation using Centroidal Angular Momentum (CAM) regulation. Building upon these advancements, we derive the DCM step-to-step dynamics for Model Predictive Control MPC formulation, enabling simultaneous optimization of step position and step duration. A hierarchical control framework integrating MPC with a Whole-Body Controller (WBC) is implemented for bipedal locomotion across uneven stepping stones. The results validate the efficacy of the proposed hierarchical control framework and the theoretical formulation. 

**Abstract (ZH)**: 本研究提出了一种在不平地形条件下增强的双足分层控制框架的理论模型。由于线性倒摆模型（LIPM）在处理地形高度变化时的固有限制，我们开发了分段斜坡线性倒摆模型（PS-LIPM），该创新模型能够在单步周期中动态调整质心高度以与地形起伏保持一致。另一项贡献是提出了一种基于角动量的广义LIPM（G-ALIP）模型，用于通过质心角动量（CAM）调节补偿质心速度。在此基础上，我们推导出了基于模型预测控制（MPC）的动态中心动量（DCM）步态到步态动力学，实现了同时优化步位和步长的优化。将MPC与整体体控制器（WBC）结合的分层控制框架被实施以实现跨越不平踏石的双足行走。研究结果验证了所提出的分层控制框架和理论模型的有效性。 

---
# Evaluation of Flight Parameters in UAV-based 3D Reconstruction for Rooftop Infrastructure Assessment 

**Title (ZH)**: 基于无人机3D重建的屋顶基础设施评估中飞行参数评价 

**Authors**: Nick Chodura, Melissa Greeff, Joshua Woods  

**Link**: [PDF](https://arxiv.org/pdf/2504.02084)  

**Abstract**: Rooftop 3D reconstruction using UAV-based photogrammetry offers a promising solution for infrastructure assessment, but existing methods often require high percentages of image overlap and extended flight times to ensure model accuracy when using autonomous flight paths. This study systematically evaluates key flight parameters-ground sampling distance (GSD) and image overlap-to optimize the 3D reconstruction of complex rooftop infrastructure. Controlled UAV flights were conducted over a multi-segment rooftop at Queen's University using a DJI Phantom 4 Pro V2, with varied GSD and overlap settings. The collected data were processed using Reality Capture software and evaluated against ground truth models generated from UAV-based LiDAR and terrestrial laser scanning (TLS). Experimental results indicate that a GSD range of 0.75-1.26 cm combined with 85% image overlap achieves a high degree of model accuracy, while minimizing images collected and flight time. These findings provide guidance for planning autonomous UAV flight paths for efficient rooftop assessments. 

**Abstract (ZH)**: 基于无人机摄影测量的屋顶三维重建方法在基础设施评估中的应用：系统评价关键飞行参数以优化复杂屋顶结构的三维重建 

---
# Distance Estimation to Support Assistive Drones for the Visually Impaired using Robust Calibration 

**Title (ZH)**: 基于鲁棒标定支持视障人士的助航无人机的距离估计 

**Authors**: Suman Raj, Bhavani A Madhabhavi, Madhav Kumar, Prabhav Gupta, Yogesh Simmhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01988)  

**Abstract**: Autonomous navigation by drones using onboard sensors, combined with deep learning and computer vision algorithms, is impacting a number of domains. We examine the use of drones to autonomously assist Visually Impaired People (VIPs) in navigating outdoor environments while avoiding obstacles. Here, we present NOVA, a robust calibration technique using depth maps to estimate absolute distances to obstacles in a campus environment. NOVA uses a dynamic-update method that can adapt to adversarial scenarios. We compare NOVA with SOTA depth map approaches, and with geometric and regression-based baseline models, for distance estimation to VIPs and other obstacles in diverse and dynamic conditions. We also provide exhaustive evaluations to validate the robustness and generalizability of our methods. NOVA predicts distances to VIP with an error <30cm and to different obstacles like cars and bicycles with a maximum of 60cm error, which are better than the baselines. NOVA also clearly out-performs SOTA depth map methods, by upto 5.3-14.6x. 

**Abstract (ZH)**: 基于机载传感器结合深度学习和计算机视觉算法的无人机自主导航正在影响多个领域。我们研究了无人机在避免障碍物的同时自主协助视障人士在户外环境中导航的应用。在此，我们提出了NOVA，一种基于深度图的稳健校准技术，用于校园环境中的障碍物绝对距离估计。NOVA采用动态更新方法，可适应对抗场景。我们将NOVA与当前最佳深度图方法以及基于几何和回归的基本模型进行比较，以估计视障人士和其他障碍物在多种动态条件下的距离。我们还提供了详尽的评估以验证我们方法的稳健性和通用性。NOVA预测视障人士的距离误差小于30厘米，对汽车和自行车等不同障碍物的距离误差最大为60厘米，均优于基线模型。NOVA还比当前最佳深度图方法高5.3-14.6倍地优于这些方法。 

---
# Information Gain Is Not All You Need 

**Title (ZH)**: 信息增益并非 sufficient 

**Authors**: Ludvig Ericson, José Pedro, Patric Jensfelt  

**Link**: [PDF](https://arxiv.org/pdf/2504.01980)  

**Abstract**: Autonomous exploration in mobile robotics is driven by two competing objectives: coverage, to exhaustively observe the environment; and path length, to do so with the shortest path possible. Though it is difficult to evaluate the best course of action without knowing the unknown, the unknown can often be understood through models, maps, or common sense. However, previous work has shown that improving estimates of information gain through such prior knowledge leads to greedy behavior and ultimately causes backtracking, which degrades coverage performance. In fact, any information gain maximization will exhibit this behavior, even without prior knowledge. Information gained at task completion is constant, and cannot be maximized for. It is therefore an unsuitable choice as an optimization objective. Instead, information gain is a decision criterion for determining which candidate states should still be considered for exploration. The task therefore becomes to reach completion with the shortest total path. Since determining the shortest path is typically intractable, it is necessary to rely on a heuristic or estimate to identify candidate states that minimize the total path length. To address this, we propose a heuristic that reduces backtracking by preferring candidate states that are close to the robot, but far away from other candidate states. We evaluate the performance of the proposed heuristic in simulation against an information gain-based approach and frontier exploration, and show that our method significantly decreases total path length, both with and without prior knowledge of the environment. 

**Abstract (ZH)**: 移动机器人自主探索受两条竞争目标驱动：覆盖率，以全面观察环境；路径长度，以使用最短路径实现这一目标。 

---
# Designing Effective Human-Swarm Interaction Interfaces: Insights from a User Study on Task Performance 

**Title (ZH)**: 基于任务绩效用户研究的设计有效的人群-蜂群交互界面的见解 

**Authors**: Wasura D. Wattearachchi, Erandi Lakshika, Kathryn Kasmarik, Michael Barlow  

**Link**: [PDF](https://arxiv.org/pdf/2504.02250)  

**Abstract**: In this paper, we present a systematic method of design for human-swarm interaction interfaces, combining theoretical insights with empirical evaluation. We first derive ten design principles from existing literature, apply them to key information dimensions identified through goal-directed task analysis and developed a tablet-based interface for a target search task. We then conducted a user study with 31 participants where humans were required to guide a robotic swarm to a target in the presence of three types of hazards that pose a risk to the robots: Distributed, Moving, and Spreading. Performance was measured based on the proximity of the robots to the target and the number of deactivated robots at the end of the task. Results indicate that at least one robot was bought closer to the target in 98% of tasks, demonstrating the interface's success fulfilling the primary objective of the task. Additionally, in nearly 67% of tasks, more than 50% of the robots reached the target. Moreover, particularly better performance was noted in moving hazards. Additionally, the interface appeared to help minimize robot deactivation, as evidenced by nearly 94% of tasks where participants managed to keep more than 50% of the robots active, ensuring that most of the swarm remained operational. However, its effectiveness varied across hazards, with robot deactivation being lowest in distributed hazard scenarios, suggesting that the interface provided the most support in these conditions. 

**Abstract (ZH)**: 本文提出了一种结合理论洞察与实证评估的人机群组交互界面设计系统方法。首先从现有文献中推导出十项设计原则，并将其应用于通过目标导向任务分析识别的关键信息维度，开发了一个基于平板的界面用于目标搜索任务。然后，通过一项涉及31名参与者的用户研究，评估人类引导机器人群组在三种类型的威胁下（分布式、移动和蔓延）到达目标的表现。性能通过机器人与目标的接近程度和任务结束时未被激活的机器人数量来衡量。结果表明，在98%的任务中，至少有一台机器人的位置更接近目标，表明该界面成功实现了主要目标。此外，在约67%的任务中，超过50%的机器人能够到达目标。特别是在移动威胁方面，表现尤为出色。此外，界面似乎有助于减少机器人失能，因为在约94%的任务中，参与者能够保持至少50%的机器人活跃，确保大部分群组保持运行。然而，其效果在不同威胁类型下有所差异，分散威胁场景下机器人失能最少，表明该界面在这些条件下提供了最佳支持。 

---
# System Identification and Adaptive Input Estimation on the Jaiabot Micro Autonomous Underwater Vehicle 

**Title (ZH)**: Jaiabot 微型自主水下车辆的系统辨识与自适应输入估计 

**Authors**: Ioannis Faros, Herbert G. Tanner  

**Link**: [PDF](https://arxiv.org/pdf/2504.02005)  

**Abstract**: This paper reports an attempt to model the system dynamics and estimate both the unknown internal control input and the state of a recently developed marine autonomous vehicle, the Jaiabot. Although the Jaiabot has shown promise in many applications, process and sensor noise necessitates state estimation and noise filtering. In this work, we present the first surge and heading linear dynamical model for Jaiabots derived from real data collected during field testing. An adaptive input estimation algorithm is implemented to accurately estimate the control input and hence the state. For validation, this approach is compared to the classical Kalman filter, highlighting its advantages in handling unknown control inputs. 

**Abstract (ZH)**: 本研究报道了尝试建立Jaiabot海下自主车辆系统动力学模型，并估计其未知内部控制输入和状态的尝试。虽然Jaiabot在许多应用中表现出潜力，但过程噪声和传感器噪声需要进行状态估计和噪声滤波。在这项工作中，我们首次基于实地测试收集的实际数据建立了Jaiabot的纵荡和航向线性动态模型，并实现了一种自适应输入估计算法，以准确估计控制输入和状态。为了验证这种方法，将其与经典的卡尔曼滤波器进行了比较，突显了其在处理未知控制输入方面的优势。 

---
# Impedance and Stability Targeted Adaptation for Aerial Manipulator with Unknown Coupling Dynamics 

**Title (ZH)**: 未知耦合动力学条件下阻抗和稳定性目标适应性控制方法 

**Authors**: Amitabh Sharma, Saksham Gupta, Shivansh Pratap Singh, Rishabh Dev Yadav, Hongyu Song, Wei Pan, Spandan Roy, Simone Baldi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01983)  

**Abstract**: Stable aerial manipulation during dynamic tasks such as object catching, perching, or contact with rigid surfaces necessarily requires compliant behavior, which is often achieved via impedance control. Successful manipulation depends on how effectively the impedance control can tackle the unavoidable coupling forces between the aerial vehicle and the manipulator. However, the existing impedance controllers for aerial manipulator either ignore these coupling forces (in partitioned system compliance methods) or require their precise knowledge (in complete system compliance methods). Unfortunately, such forces are very difficult to model, if at all possible. To solve this long-standing control challenge, we introduce an impedance controller for aerial manipulator which does not rely on a priori knowledge of the system dynamics and of the coupling forces. The impedance control design can address unknown coupling forces, along with system parametric uncertainties, via suitably designed adaptive laws. The closed-loop system stability is proved analytically and experimental results with a payload-catching scenario demonstrate significant improvements in overall stability and tracking over the state-of-the-art impedance controllers using either partitioned or complete system compliance. 

**Abstract (ZH)**: 空中 manipulator 在动态任务如物体抓取、栖息或与刚性表面接触过程中稳定操作，必然需要具备顺应性行为，这通常通过阻抗控制实现。成功的操作取决于阻抗控制如何有效应对空中机器人与 manipulator 之间的不可避免的耦合力。然而，现有空中 manipulator 的阻抗控制器要么忽视这些耦合力（在分系统顺应性方法中），要么需要精确知道这些耦合力（在整体系统顺应性方法中）。不幸的是，这些力非常难以建模，甚至可能根本不可能建模。为解决这一长期存在的控制挑战，我们 introduce 一种无需预先了解系统动力学和耦合力的空中 manipulator 阻抗控制器。阻抗控制设计可以通过适当地设计自适应律来处理未知的耦合力以及系统参数不确定性。分析证明闭环系统稳定性，实验结果（以负载抓取场景为例）显示与分系统或整体系统顺应性方法的现有阻抗控制器相比在整体稳定性和跟踪方面有显著改进。 

---
