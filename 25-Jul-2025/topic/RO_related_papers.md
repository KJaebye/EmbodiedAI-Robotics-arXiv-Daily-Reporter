# A Novel Monte-Carlo Compressed Sensing and Dictionary Learning Method for the Efficient Path Planning of Remote Sensing Robots 

**Title (ZH)**: 一种用于遥感机器人高效路径规划的新型蒙特卡洛压缩感知与字典学习方法 

**Authors**: Alghalya Al-Hajri, Ejmen Al-Ubejdij, Aiman Erbad, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2507.18462)  

**Abstract**: In recent years, Compressed Sensing (CS) has gained significant interest as a technique for acquiring high-resolution sensory data using fewer measurements than traditional Nyquist sampling requires. At the same time, autonomous robotic platforms such as drones and rovers have become increasingly popular tools for remote sensing and environmental monitoring tasks, including measurements of temperature, humidity, and air quality. Within this context, this paper presents, to the best of our knowledge, the first investigation into how the structure of CS measurement matrices can be exploited to design optimized sampling trajectories for robotic environmental data collection. We propose a novel Monte Carlo optimization framework that generates measurement matrices designed to minimize both the robot's traversal path length and the signal reconstruction error within the CS framework. Central to our approach is the application of Dictionary Learning (DL) to obtain a data-driven sparsifying transform, which enhances reconstruction accuracy while further reducing the number of samples that the robot needs to collect. We demonstrate the effectiveness of our method through experiments reconstructing $NO_2$ pollution maps over the Gulf region. The results indicate that our approach can reduce robot travel distance to less than $10\%$ of a full-coverage path, while improving reconstruction accuracy by over a factor of five compared to traditional CS methods based on DCT and polynomial dictionaries, as well as by a factor of two compared to previously-proposed Informative Path Planning (IPP) methods. 

**Abstract (ZH)**: 近年来，压缩感知（CS）作为一种能够在少于传统奈奎斯特采样所需测量次数的情况下获取高分辨率传感器数据的技术，逐渐引起了广泛关注。与此同时，无人机和探测车等自主机器人平台因其在遥感和环境监测任务中的广泛应用（包括温度、湿度和空气质量的测量）而越来越流行。在此背景下，本文据我们所知，首次探讨了如何利用CS测量矩阵的结构来设计优化的机器人环境数据采集采样轨迹。我们提出了一种新颖的蒙特卡洛优化框架，用于生成既能最小化机器人行进路径长度又能最小化信号重构误差的测量矩阵。该方法的核心在于应用字典学习（DL）以获取数据驱动的稀疏化变换，从而提高重构准确性并进一步减少机器人需要采集的样本数量。通过在海湾地区重建$NO_2$污染图的实验，我们展示了该方法的有效性。结果表明，与基于DCT和多项式字典的传统CS方法相比，我们的方法可以将机器人行驶距离减少至全覆盖路径的不到10%，同时将重构准确性提高超过五倍，与先前提出的冗余路径规划（IPP）方法相比，重构准确性提高了一倍。 

---
# Evaluating the Pre-Dressing Step: Unfolding Medical Garments Via Imitation Learning 

**Title (ZH)**: 评估预穿衣步骤：通过类比学习展开医疗garments 

**Authors**: David Blanco-Mulero, Júlia Borràs, Carme Torras  

**Link**: [PDF](https://arxiv.org/pdf/2507.18436)  

**Abstract**: Robotic-assisted dressing has the potential to significantly aid both patients as well as healthcare personnel, reducing the workload and improving the efficiency in clinical settings. While substantial progress has been made in robotic dressing assistance, prior works typically assume that garments are already unfolded and ready for use. However, in medical applications gowns and aprons are often stored in a folded configuration, requiring an additional unfolding step. In this paper, we introduce the pre-dressing step, the process of unfolding garments prior to assisted dressing. We leverage imitation learning for learning three manipulation primitives, including both high and low acceleration motions. In addition, we employ a visual classifier to categorise the garment state as closed, partly opened, and fully opened. We conduct an empirical evaluation of the learned manipulation primitives as well as their combinations. Our results show that highly dynamic motions are not effective for unfolding freshly unpacked garments, where the combination of motions can efficiently enhance the opening configuration. 

**Abstract (ZH)**: 机器人辅助穿衣有潜力显著帮助患者和医疗人员，减少工作量并提高临床环境中的效率。虽然在机器人穿衣辅助方面取得了显著进展，但以往的研究通常假设衣物已经打开并准备好使用。然而，在医疗应用中，连衣袍和围裙 often 存储在折叠配置中，需要额外进行展开步骤。在本文中，我们引入了穿衣前的预处理步骤，即在辅助穿衣前展开衣物的过程。我们利用模仿学习学习三种操作基本单元，包括高加速和低加速运动。此外，我们采用视觉分类器将衣物状态分类为关闭、部分打开和完全打开。我们对学习的操作基本单元及其组合进行了实证评估。结果显示，对于刚取出的衣物，高度动态的运动并不有效，而运动的组合能有效地提升衣物的展开配置。 

---
# MoRPI-PINN: A Physics-Informed Framework for Mobile Robot Pure Inertial Navigation 

**Title (ZH)**: MoRPI-PINN: 一种移动机器人纯惯性导航的物理约束框架 

**Authors**: Arup Kumar Sahoo, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2507.18206)  

**Abstract**: A fundamental requirement for full autonomy in mobile robots is accurate navigation even in situations where satellite navigation or cameras are unavailable. In such practical situations, relying only on inertial sensors will result in navigation solution drift due to the sensors' inherent noise and error terms. One of the emerging solutions to mitigate drift is to maneuver the robot in a snake-like slithering motion to increase the inertial signal-to-noise ratio, allowing the regression of the mobile robot position. In this work, we propose MoRPI-PINN as a physics-informed neural network framework for accurate inertial-based mobile robot navigation. By embedding physical laws and constraints into the training process, MoRPI-PINN is capable of providing an accurate and robust navigation solution. Using real-world experiments, we show accuracy improvements of over 85% compared to other approaches. MoRPI-PINN is a lightweight approach that can be implemented even on edge devices and used in any typical mobile robot application. 

**Abstract (ZH)**: 全自主移动机器人中准确导航的基本要求是在卫星导航或摄像头不可用的情况下仍能实现精确导航。在这种实际情况下，仅依赖惯性传感器会导致由于传感器固有的噪声和误差而导致的导航解算偏移。为了减轻偏移，一种新兴的解决方案是让机器人以蛇形滑动运动来增加惯性信号信噪比，从而允许对移动机器人位置进行回归。本文提出了一种嵌入物理定律和约束的神经网络框架MoRPI-PINN，用于准确的基于惯性的移动机器人导航。通过在训练过程中嵌入物理定律和约束，MoRPI-PINN能够提供准确且稳健的导航解算。通过实际实验，我们展示了与其它方法相比超过85%的精度改善。MoRPI-PINN是一种轻量级的方法，甚至可以在边缘设备上实现，并可用于任何典型的移动机器人应用。 

---
# A Modular Residual Learning Framework to Enhance Model-Based Approach for Robust Locomotion 

**Title (ZH)**: 一种模块化残差学习框架以增强基于模型的方法以实现稳健的运动控制 

**Authors**: Min-Gyu Kim, Dongyun Kang, Hajun Kim, Hae-Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.18138)  

**Abstract**: This paper presents a novel approach that combines the advantages of both model-based and learning-based frameworks to achieve robust locomotion. The residual modules are integrated with each corresponding part of the model-based framework, a footstep planner and dynamic model designed using heuristics, to complement performance degradation caused by a model mismatch. By utilizing a modular structure and selecting the appropriate learning-based method for each residual module, our framework demonstrates improved control performance in environments with high uncertainty, while also achieving higher learning efficiency compared to baseline methods. Moreover, we observed that our proposed methodology not only enhances control performance but also provides additional benefits, such as making nominal controllers more robust to parameter tuning. To investigate the feasibility of our framework, we demonstrated residual modules combined with model predictive control in a real quadrupedal robot. Despite uncertainties beyond the simulation, the robot successfully maintains balance and tracks the commanded velocity. 

**Abstract (ZH)**: 本文提出了一种结合模型导向和学习导向框架优势的新方法，以实现稳健的运动控制。通过将残差模块与模型导向框架中的足步规划器和基于启发式的动态模型相结合，弥补由模型不匹配引起的表现退化。利用模块化结构并为每个残差模块选择合适的基于学习的方法，我们的框架在高不确定性环境中的控制性能得到提高，并且与基准方法相比具有更高的学习效率。此外，我们发现所提出的方法不仅提高了控制性能，还提供了额外的好处，如使名义控制器更 robust 至参数调整。为了验证框架的可行性，我们在实际四足机器人上展示了结合模型预测控制的残差模块。尽管存在模拟之外的不确定性，机器人仍能保持平衡并跟踪命令的速度。 

---
# Modular Robot and Landmark Localisation Using Relative Bearing Measurements 

**Title (ZH)**: 模块化机器人和基于相对方位测量的目标定位 

**Authors**: Behzad Zamani, Jochen Trumpf, Chris Manzie  

**Link**: [PDF](https://arxiv.org/pdf/2507.18070)  

**Abstract**: In this paper we propose a modular nonlinear least squares filtering approach for systems composed of independent subsystems. The state and error covariance estimate of each subsystem is updated independently, even when a relative measurement simultaneously depends on the states of multiple subsystems. We integrate the Covariance Intersection (CI) algorithm as part of our solution in order to prevent double counting of information when subsystems share estimates with each other. An alternative derivation of the CI algorithm based on least squares estimation makes this integration possible. We particularise the proposed approach to the robot-landmark localization problem. In this problem, noisy measurements of the bearing angle to a stationary landmark position measured relative to the SE(2) pose of a moving robot couple the estimation problems for the robot pose and the landmark position. In a randomized simulation study, we benchmark the proposed modular method against a monolithic joint state filter to elucidate their respective trade-offs. In this study we also include variants of the proposed method that achieve a graceful degradation of performance with reduced communication and bandwidth requirements. 

**Abstract (ZH)**: 模块化非线性最小二乘滤波方法及其在独立子系统组合系统中的应用：基于协方差融合的机器人地标定位问题及其性能评估 

---
# A Step-by-step Guide on Nonlinear Model Predictive Control for Safe Mobile Robot Navigation 

**Title (ZH)**: 非线性模型预测控制的逐步指南：安全移动机器人导航 

**Authors**: Dennis Benders, Laura Ferranti, Johannes Köhler  

**Link**: [PDF](https://arxiv.org/pdf/2507.17856)  

**Abstract**: Designing a Model Predictive Control (MPC) scheme that enables a mobile robot to safely navigate through an obstacle-filled environment is a complicated yet essential task in robotics. In this technical report, safety refers to ensuring that the robot respects state and input constraints while avoiding collisions with obstacles despite the presence of disturbances and measurement noise. This report offers a step-by-step approach to implementing Nonlinear Model Predictive Control (NMPC) schemes addressing these safety requirements. Numerous books and survey papers provide comprehensive overviews of linear MPC (LMPC) \cite{bemporad2007robust,kouvaritakis2016model}, NMPC \cite{rawlings2017model,allgower2004nonlinear,mayne2014model,grune2017nonlinear,saltik2018outlook}, and their applications in various domains, including robotics \cite{nascimento2018nonholonomic,nguyen2021model,shi2021advanced,wei2022mpc}. This report does not aim to replicate those exhaustive reviews. Instead, it focuses specifically on NMPC as a foundation for safe mobile robot navigation. The goal is to provide a practical and accessible path from theoretical concepts to mathematical proofs and implementation, emphasizing safety and performance guarantees. It is intended for researchers, robotics engineers, and practitioners seeking to bridge the gap between theoretical NMPC formulations and real-world robotic applications.
This report is not necessarily meant to remain fixed over time. If someone finds an error in the presented theory, please reach out via the given email addresses. We are happy to update the document if necessary. 

**Abstract (ZH)**: 设计一种非线性模型预测控制（NMPC）方案，使移动机器人能够安全地导航通过充满障碍物的环境是一项复杂但至关重要的任务。本技术报告中的安全性是指确保机器人在受到干扰和测量噪声影响的情况下，遵守状态和输入约束并避免与障碍物发生碰撞。本报告提供了实现非线性模型预测控制（NMPC）方案的逐步方法，以满足这些安全要求。 

---
# ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense 

**Title (ZH)**: ARBoids：自适应剩余强化学习与Boids模型在协同多USV目标防御中的应用 

**Authors**: Jiyue Tao, Tongsheng Shen, Dexin Zhao, Feitian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18549)  

**Abstract**: The target defense problem (TDP) for unmanned surface vehicles (USVs) concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, this letter introduces ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. The proposed approach is validated in a high-fidelity Gazebo simulation environment, demonstrating superior performance over traditional interception strategies, including pure force-based approaches and vanilla DRL policies. Furthermore, the learned policy exhibits strong adaptability to attackers with diverse maneuverability profiles, highlighting its robustness and generalization capability. The code of ARBoids will be released upon acceptance of this letter. 

**Abstract (ZH)**: 无人水面车辆的目标防御问题（TDP）涉及使用一个或多个防御型无人水面车辆在敌对无人水面车辆突破指定目标区域之前对其进行拦截。当攻击者具备优于防御者的机动性能时，这一挑战尤为严峻，极大地复杂化了有效的拦截过程。为应对这一挑战，本文引入了ARBoids，这是一种将深度强化学习（DRL）与受生物启发的力基Boids模型结合的新型自适应残差强化学习框架。该框架中的Boids模型作为多agent协调的高效基线策略，而DRL则学习一个残差策略以自适应地细化和优化防御者的行为。所提出的方法在高保真Gazebo仿真环境中得到了验证，并在传统拦截策略，包括基于纯粹力的方法和标准DRL策略中表现出更优性能。此外，学习到的策略对具有不同机动性能的攻击者表现出强大的适应性，突显了其鲁棒性和通用性能力。本文接受后将公开ARBoids的代码。 

---
