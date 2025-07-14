# Robotic Calibration Based on Haptic Feedback Improves Sim-to-Real Transfer 

**Title (ZH)**: 基于触觉反馈的机器人标定改进从模拟到现实的迁移 

**Authors**: Juraj Gavura, Michal Vavrecka, Igor Farkas, Connor Gade  

**Link**: [PDF](https://arxiv.org/pdf/2507.08572)  

**Abstract**: When inverse kinematics (IK) is adopted to control robotic arms in manipulation tasks, there is often a discrepancy between the end effector (EE) position of the robot model in the simulator and the physical EE in reality. In most robotic scenarios with sim-to-real transfer, we have information about joint positions in both simulation and reality, but the EE position is only available in simulation. We developed a novel method to overcome this difficulty based on haptic feedback calibration, using a touchscreen in front of the robot that provides information on the EE position in the real environment. During the calibration procedure, the robot touches specific points on the screen, and the information is stored. In the next stage, we build a transformation function from the data based on linear transformation and neural networks that is capable of outputting all missing variables from any partial input (simulated/real joint/EE position). Our results demonstrate that a fully nonlinear neural network model performs best, significantly reducing positioning errors. 

**Abstract (ZH)**: 基于触控屏的触觉反馈校准的逆运动学仿真到现实转移方法 

---
# Joint Optimization-based Targetless Extrinsic Calibration for Multiple LiDARs and GNSS-Aided INS of Ground Vehicles 

**Title (ZH)**: 基于联合优化的目标导向外标定方法及其在GNSS辅助INS与多LiDAR装备地面车辆中的应用 

**Authors**: Junhui Wang, Yan Qiao, Chao Gao, Naiqi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08349)  

**Abstract**: Accurate extrinsic calibration between multiple LiDAR sensors and a GNSS-aided inertial navigation system (GINS) is essential for achieving reliable sensor fusion in intelligent mining environments. Such calibration enables vehicle-road collaboration by aligning perception data from vehicle-mounted sensors to a unified global reference frame. However, existing methods often depend on artificial targets, overlapping fields of view, or precise trajectory estimation, which are assumptions that may not hold in practice. Moreover, the planar motion of mining vehicles leads to observability issues that degrade calibration performance. This paper presents a targetless extrinsic calibration method that aligns multiple onboard LiDAR sensors to the GINS coordinate system without requiring overlapping sensor views or external targets. The proposed approach introduces an observation model based on the known installation height of the GINS unit to constrain unobservable calibration parameters under planar motion. A joint optimization framework is developed to refine both the extrinsic parameters and GINS trajectory by integrating multiple constraints derived from geometric correspondences and motion consistency. The proposed method is applicable to heterogeneous LiDAR configurations, including both mechanical and solid-state sensors. Extensive experiments on simulated and real-world datasets demonstrate the accuracy, robustness, and practical applicability of the approach under diverse sensor setups. 

**Abstract (ZH)**: 多LiDAR传感器与GNSS辅助惯性导航系统(GINS)之间的无目标外部标定对于实现智能采矿环境中可靠的传感器融合至关重要。提出的方法无需重叠传感器视场或外部目标即可将多个车载LiDAR传感器对准GINS坐标系。该方法基于GINS单元的已知安装高度引入观测模型，以在平动运动下约束不可观测的标定参数。开发了一种联合优化框架，通过集成来自几何对应和运动一致性推导出的多个约束来细化外部参数和GINS轨迹。该方法适用于包括机械和固态传感器在内的异构LiDAR配置。在模拟和实际数据集上的广泛实验表明，该方法在不同传感器配置下具有高精度、鲁棒性和实用适用性。 

---
# Noise-Enabled Goal Attainment in Crowded Collectives 

**Title (ZH)**: 噪声驱动的目标达成在拥挤集体中 

**Authors**: Lucy Liu, Justin Werfel, Federico Toschi, L. Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2507.08100)  

**Abstract**: In crowded environments, individuals must navigate around other occupants to reach their destinations. Understanding and controlling traffic flows in these spaces is relevant to coordinating robot swarms and designing infrastructure for dense populations. Here, we combine simulations, theory, and robotic experiments to study how noisy motion can disrupt traffic jams and enable flow as agents travel to individual goals. Above a critical noise level, large jams do not persist. From this observation, we analytically approximate the goal attainment rate as a function of the noise level, then solve for the optimal agent density and noise level that maximize the swarm's goal attainment rate. We perform robotic experiments to corroborate our simulated and theoretical results. Finally, we compare simple, local navigation approaches with a sophisticated but computationally costly central planner. A simple reactive scheme performs well up to moderate densities and is far more computationally efficient than a planner, suggesting lessons for real-world problems. 

**Abstract (ZH)**: 在拥挤环境中，个体必须在其他占有人周围导航以到达目的地。理解并控制这些空间中的交通流对于协调机器人集群和设计密集人口的基础设施是相关的。在这里，我们结合模拟、理论和机器人实验来研究噪声运动如何破坏交通拥堵并促进个体目标导向过程中的流量。当噪声水平超过临界值时，大型拥堵不会持续。基于这一观察，我们分析地将目标达成率近似为噪声水平的函数，然后求解使集群目标达成率最大化的最优代理密度和噪声水平。我们进行机器人实验以验证我们的模拟和理论结果。最后，我们将简单的局部导航方法与复杂的但计算成本高的中央规划者进行比较。一个简单的反应方案在中等密度下表现良好，并且比规划者更加计算高效，这为现实世界问题提供了启示。 

---
