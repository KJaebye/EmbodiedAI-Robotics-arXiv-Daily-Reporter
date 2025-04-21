# DiffOG: Differentiable Policy Trajectory Optimization with Generalizability 

**Title (ZH)**: DiffOG: 具有泛化能力的可微分策略轨迹优化 

**Authors**: Zhengtong Xu, Zichen Miao, Qiang Qiu, Zhe Zhang, Yu She  

**Link**: [PDF](https://arxiv.org/pdf/2504.13807)  

**Abstract**: Imitation learning-based visuomotor policies excel at manipulation tasks but often produce suboptimal action trajectories compared to model-based methods. Directly mapping camera data to actions via neural networks can result in jerky motions and difficulties in meeting critical constraints, compromising safety and robustness in real-world deployment. For tasks that require high robustness or strict adherence to constraints, ensuring trajectory quality is crucial. However, the lack of interpretability in neural networks makes it challenging to generate constraint-compliant actions in a controlled manner. This paper introduces differentiable policy trajectory optimization with generalizability (DiffOG), a learning-based trajectory optimization framework designed to enhance visuomotor policies. By leveraging the proposed differentiable formulation of trajectory optimization with transformer, DiffOG seamlessly integrates policies with a generalizable optimization layer. Visuomotor policies enhanced by DiffOG generate smoother, constraint-compliant action trajectories in a more interpretable way. DiffOG exhibits strong generalization capabilities and high flexibility. We evaluated DiffOG across 11 simulated tasks and 2 real-world tasks. The results demonstrate that DiffOG significantly enhances the trajectory quality of visuomotor policies while having minimal impact on policy performance, outperforming trajectory processing baselines such as greedy constraint clipping and penalty-based trajectory optimization. Furthermore, DiffOG achieves superior performance compared to existing constrained visuomotor policy. 

**Abstract (ZH)**: 基于模仿学习的视觉运动策略在操作任务中表现出色，但与基于模型的方法相比，往往会生成次优的动作轨迹。直接通过神经网络将相机数据映射到动作可能会导致动作不连贯，并且难以满足关键约束，从而在实际部署中影响安全性和鲁棒性。对于需要高鲁棒性或严格遵守约束的任务，确保轨迹质量至关重要。然而，神经网络缺乏可解释性，使其难以以受控的方式生成符合约束的动作。本文提出了可泛化的可微策略轨迹优化（DiffOG），这是一种基于学习的轨迹优化框架，旨在提升视觉运动策略。通过利用带有变换器的可微轨迹优化公式，DiffOG 平滑地将策略与一个可泛化的优化层结合在一起。经过 DiffOG 提升的视觉运动策略生成更具可解释性的、符合约束的动作轨迹。DiffOG 具有较强的泛化能力和高度的灵活性。我们在 11 个模拟任务和 2 个真实世界任务上评估了 DiffOG。结果表明，DiffOG 显著提升了视觉运动策略的轨迹质量，同时对策略性能的影响极小，优于诸如贪婪约束截断和基于惩罚的轨迹优化等轨迹处理基线。此外，DiffOG 在现有的受约束的视觉运动策略中表现出优越的性能。 

---
# Imitation Learning with Precisely Labeled Human Demonstrations 

**Title (ZH)**: 精确标注的人类示范的 imitation 学习 

**Authors**: Yilong Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.13803)  

**Abstract**: Within the imitation learning paradigm, training generalist robots requires large-scale datasets obtainable only through diverse curation. Due to the relative ease to collect, human demonstrations constitute a valuable addition when incorporated appropriately. However, existing methods utilizing human demonstrations face challenges in inferring precise actions, ameliorating embodiment gaps, and fusing with frontier generalist robot training pipelines. In this work, building on prior studies that demonstrate the viability of using hand-held grippers for efficient data collection, we leverage the user's control over the gripper's appearance--specifically by assigning it a unique, easily segmentable color--to enable simple and reliable application of the RANSAC and ICP registration method for precise end-effector pose estimation. We show in simulation that precisely labeled human demonstrations on their own allow policies to reach on average 88.1% of the performance of using robot demonstrations, and boost policy performance when combined with robot demonstrations, despite the inherent embodiment gap. 

**Abstract (ZH)**: 基于模仿学习范式，在利用手持 gripper 效率收集数据的基础上，通过赋予其独特的易分割颜色以实现精确末端执行器姿态估计，无需大量标注数据即可提高通用机器人政策性能。 

---
# Unified Manipulability and Compliance Analysis of Modular Soft-Rigid Hybrid Fingers 

**Title (ZH)**: 模块化软硬混合手指的统一操作与顺应性分析 

**Authors**: Jianshu Zhou, Boyuan Liang, Junda Huang, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2504.13800)  

**Abstract**: This paper presents a unified framework to analyze the manipulability and compliance of modular soft-rigid hybrid robotic fingers. The approach applies to both hydraulic and pneumatic actuation systems. A Jacobian-based formulation maps actuator inputs to joint and task-space responses. Hydraulic actuators are modeled under incompressible assumptions, while pneumatic actuators are described using nonlinear pressure-volume relations. The framework enables consistent evaluation of manipulability ellipsoids and compliance matrices across actuation modes. We validate the analysis using two representative hands: DexCo (hydraulic) and Edgy-2 (pneumatic). Results highlight actuation-dependent trade-offs in dexterity and passive stiffness. These findings provide insights for structure-aware design and actuator selection in soft-rigid robotic fingers. 

**Abstract (ZH)**: 本文提出了一种统一框架来分析模块化软硬混合机器人手指的可控性和顺应性。该方法适用于液压和气动驱动系统。基于雅可比的表示将驱动器输入映射到关节空间和任务空间的响应。假定不可压缩流体条件建模液压驱动器，而气动驱动器则使用非线性压力-体积关系进行描述。该框架使得可以在不同的驱动模式下一致地评估可控性椭球体和顺应性矩阵。我们使用两种代表性手部结构对分析进行了验证：DexCo（液压驱动）和Edgy-2（气动驱动）。结果强调了不同驱动方式对灵巧性和被动刚度之间的权衡。这些发现为软硬混合机器人手指的结构意识设计和驱动器选择提供了见解。 

---
# Learning Through Retrospection: Improving Trajectory Prediction for Automated Driving with Error Feedback 

**Title (ZH)**: 基于反思学习：通过误差反馈提高自动驾驶轨迹预测 

**Authors**: Steffen Hagedorn, Aron Distelzweig, Marcel Hallgarten, Alexandru P. Condurache  

**Link**: [PDF](https://arxiv.org/pdf/2504.13785)  

**Abstract**: In automated driving, predicting trajectories of surrounding vehicles supports reasoning about scene dynamics and enables safe planning for the ego vehicle. However, existing models handle predictions as an instantaneous task of forecasting future trajectories based on observed information. As time proceeds, the next prediction is made independently of the previous one, which means that the model cannot correct its errors during inference and will repeat them. To alleviate this problem and better leverage temporal data, we propose a novel retrospection technique. Through training on closed-loop rollouts the model learns to use aggregated feedback. Given new observations it reflects on previous predictions and analyzes its errors to improve the quality of subsequent predictions. Thus, the model can learn to correct systematic errors during inference. Comprehensive experiments on nuScenes and Argoverse demonstrate a considerable decrease in minimum Average Displacement Error of up to 31.9% compared to the state-of-the-art baseline without retrospection. We further showcase the robustness of our technique by demonstrating a better handling of out-of-distribution scenarios with undetected road-users. 

**Abstract (ZH)**: 在自动驾驶中，预测周围车辆的轨迹有助于场景动力学的推理并使自我车辆的安全规划成为可能。然而，现有的模型将预测视为基于观测信息即时预测未来轨迹的任务。随着时间的推移，后续预测是独立于先前预测进行的，这意味着模型在推理过程中无法纠正错误并重复这些错误。为了解决这一问题并更好地利用时间数据，我们提出了一种新颖的回溯技术。通过在闭环滚动中训练模型，使其学会使用累积反馈。在获得新观测值时，模型会反思之前的预测并分析其错误以提高后续预测的质量。因此，模型可以学习在推理过程中纠正系统性错误。在nuScenes和Argoverse上的全面实验表明，与没有回溯的最新基准相比，最小平均位移误差减少了多达31.9%。我们还通过展示在未检测到的道路使用者情况下更好地处理出分布场景来展示我们技术的鲁棒性。 

---
# SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM 

**Title (ZH)**: SLAM&渲染：神经渲染、高斯点积和SLAM交叉领域的基准测试 

**Authors**: Samuel Cerezo, Gaetano Meli, Tomás Berriel Martins, Kirill Safronov, Javier Civera  

**Link**: [PDF](https://arxiv.org/pdf/2504.13713)  

**Abstract**: Models and methods originally developed for novel view synthesis and scene rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as multimodality and sequentiality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. To bridge this gap, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM and novel view rendering. It consists of 40 sequences with synchronized RGB, depth, IMU, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of novel SLAM strategies when applied to robot manipulators. The dataset sequences span five different setups featuring consumer and industrial objects under four different lighting conditions, with separate training and test trajectories per scene, as well as object rearrangements. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area. 

**Abstract (ZH)**: SLAM与新颖视图渲染交集中的SLAM&Render数据集 

---
# Self-Mixing Laser Interferometry: In Search of an Ambient Noise-Resilient Alternative to Acoustic Sensing 

**Title (ZH)**: 自混激光干涉测量：寻找一种抗环境噪声的 acoustic sensing 替代方案 

**Authors**: Remko Proesmans, Thomas Lips, Francis wyffels  

**Link**: [PDF](https://arxiv.org/pdf/2504.13711)  

**Abstract**: Self-mixing interferometry (SMI) has been lauded for its sensitivity in detecting microvibrations, while requiring no physical contact with its target. Microvibrations, i.e., sounds, have recently been used as a salient indicator of extrinsic contact in robotic manipulation. In previous work, we presented a robotic fingertip using SMI for extrinsic contact sensing as an ambient-noise-resilient alternative to acoustic sensing. Here, we extend the validation experiments to the frequency domain. We find that for broadband ambient noise, SMI still outperforms acoustic sensing, but the difference is less pronounced than in time-domain analyses. For targeted noise disturbances, analogous to multiple robots simultaneously collecting data for the same task, SMI is still the clear winner. Lastly, we show how motor noise affects SMI sensing more so than acoustic sensing, and that a higher SMI readout frequency is important for future work. Design and data files are available at this https URL. 

**Abstract (ZH)**: 自混合干涉ometry (SMI) 由于其在检测微振动方面的高灵敏度而备受赞扬，且无需与目标物理接触。微振动，即声音，最近被用作机器人操作中外来接触的一个显著指标。在以往的工作中，我们提出了一种使用 SMI 的机器人指尖，用作对背景噪声具有抗性的替代声学感知方法。在此，我们将验证实验扩展到频域。我们发现，对于宽带背景噪声，SMI 仍然优于声学感知，但差异小于时域分析中的情况。对于针对噪声干扰，类似于多台机器人同时为同一任务收集数据的情况，SMI 仍然是明显的优胜者。最后，我们展示了电机噪声如何比声学噪声更影响 SMI 的感知，并表明未来工作中较高的 SMI 读数频率很重要。设计和数据文件可在以下链接获取。 

---
# Green Robotic Mixed Reality with Gaussian Splatting 

**Title (ZH)**: 绿色机器人混合现实技术：基于高斯光斑渲染 

**Authors**: Chenxuan Liu, He Li, Zongze Li, Shuai Wang, Wei Xu, Kejiang Ye, Derrick Wing Kwan Ng, Chengzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13697)  

**Abstract**: Realizing green communication in robotic mixed reality (RoboMR) systems presents a challenge, due to the necessity of uploading high-resolution images at high frequencies through wireless channels. This paper proposes Gaussian splatting (GS) RoboMR (GSRMR), which achieves a lower energy consumption and makes a concrete step towards green RoboMR. The crux to GSRMR is to build a GS model which enables the simulator to opportunistically render a photo-realistic view from the robot's pose, thereby reducing the need for excessive image uploads. Since the GS model may involve discrepancies compared to the actual environments, a GS cross-layer optimization (GSCLO) framework is further proposed, which jointly optimizes content switching (i.e., deciding whether to upload image or not) and power allocation across different frames. The GSCLO problem is solved by an accelerated penalty optimization (APO) algorithm. Experiments demonstrate that the proposed GSRMR reduces the communication energy by over 10x compared with RoboMR. Furthermore, the proposed GSRMR with APO outperforms extensive baseline schemes, in terms of peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM). 

**Abstract (ZH)**: 实现机器人混合现实（RoboMR）系统的绿色通信 presents a challenge due to the necessity of uploading high-resolution images at high frequencies through wireless channels. This paper proposes Gaussian splatting (GS) RoboMR (GSRMR), which achieves lower energy consumption and takes a concrete step towards green RoboMR. The key to GSRMR is to build a GS model that enables the simulator to opportunistically render a photo-realistic view from the robot's pose, thereby reducing the need for excessive image uploads. Since the GS model may involve discrepancies compared to the actual environments, a GS cross-layer optimization (GSCLO) framework is further proposed, which jointly optimizes content switching (i.e., deciding whether to upload image or not) and power allocation across different frames. The GSCLO problem is solved by an accelerated penalty optimization (APO) algorithm. Experiments demonstrate that the proposed GSRMR reduces communication energy by over 10 times compared with RoboMR. Furthermore, the proposed GSRMR with APO outperforms extensive baseline schemes in terms of peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM). 

---
# Magnecko: Design and Control of a Quadrupedal Magnetic Climbing Robot 

**Title (ZH)**: Magnecko：四足磁吸附爬行机器人设计与控制 

**Authors**: Stefan Leuthard, Timo Eugster, Nicolas Faesch, Riccardo Feingold, Connor Flynn, Michael Fritsche, Nicolas Hürlimann, Elena Morbach, Fabian Tischhauser, Matthias Müller, Markus Montenegro, Valerio Schelbert, Jia-Ruei Chiu, Philip Arm, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2504.13672)  

**Abstract**: Climbing robots hold significant promise for applications such as industrial inspection and maintenance, particularly in hazardous or hard-to-reach environments. This paper describes the quadrupedal climbing robot Magnecko, developed with the major goal of providing a research platform for legged climbing locomotion. With its 12 actuated degrees of freedom arranged in an insect-style joint configuration, Magnecko's high manipulability and high range of motion allow it to handle challenging environments like overcoming concave 90 degree corners. A model predictive controller enables Magnecko to crawl on the ground on horizontal overhangs and on vertical walls. Thanks to the custom actuators and the electro-permanent magnets that are used for adhesion on ferrous surfaces, the system is powerful enough to carry additional payloads of at least 65 percent of its own weight in all orientations. The Magnecko platform serves as a foundation for climbing locomotion in complex three-dimensional environments. 

**Abstract (ZH)**: 攀爬机器人在工业检测与维护等领域应用前景显著，尤其在危险或难以到达的环境中。本文介绍了以提供腿式攀爬运动研究平台为主要目标的四足攀爬机器人Magnecko。Magnecko拥有12个可 actuated 的自由度，排列方式类似于昆虫关节，使其具备高操作灵活性和大活动范围，能够应对如克服凹面直角等挑战性环境。模型预测控制算法使Magnecko能够在水平悬挑和垂直墙面爬行。得益于定制化执行器和用于铁磁表面附着的电磁铁，该系统足以在所有姿态下承载自身重量至少65%的额外负载。Magnecko平台为复杂三维环境下的攀爬运动提供了基础。 

---
# Performance Analysis of a Mass-Spring-Damper Deformable Linear Object Model in Robotic Simulation Frameworks 

**Title (ZH)**: 基于机器人模拟框架的质点-弹簧-阻尼可变形线性物体模型性能分析 

**Authors**: Andrea Govoni, Nadia Zubair, Simone Soprani, Gianluca Palli  

**Link**: [PDF](https://arxiv.org/pdf/2504.13659)  

**Abstract**: The modelling of Deformable Linear Objects (DLOs) such as cables, wires, and strings presents significant challenges due to their flexible and deformable nature. In robotics, accurately simulating the dynamic behavior of DLOs is essential for automating tasks like wire handling and assembly. The presented study is a preliminary analysis aimed at force data collection through domain randomization (DR) for training a robot in simulation, using a Mass-Spring-Damper (MSD) system as the reference model. The study aims to assess the impact of model parameter variations on DLO dynamics, using Isaac Sim and Gazebo to validate the applicability of DR technique in these scenarios. 

**Abstract (ZH)**: 柔性线性物体（如电缆、导线和绳索）的建模因其柔性可变形的性质面临重大挑战。在机器人学中，准确模拟柔性线性物体的动力学行为对于自动化线材处理和组装任务至关重要。本研究是一项初步分析，旨在通过领域随机化（DR）收集力数据以在模拟中训练机器人，并使用质量-弹簧-阻尼（MSD）系统作为参考模型。本研究旨在评估模型参数变化对柔性线性物体动力学的影响，并使用Isaac Sim和Gazebo验证DR技术在这些场景中的适用性。 

---
# Lightweight LiDAR-Camera 3D Dynamic Object Detection and Multi-Class Trajectory Prediction 

**Title (ZH)**: 轻量级LiDAR-摄像头3D动态物体检测与多类轨迹预测 

**Authors**: Yushen He, Lei Zhao, Tianchen Deng, Zipeng Fang, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13647)  

**Abstract**: Service mobile robots are often required to avoid dynamic objects while performing their tasks, but they usually have only limited computational resources. So we present a lightweight multi-modal framework for 3D object detection and trajectory prediction. Our system synergistically integrates LiDAR and camera inputs to achieve real-time perception of pedestrians, vehicles, and riders in 3D space. The framework proposes two novel modules: 1) a Cross-Modal Deformable Transformer (CMDT) for object detection with high accuracy and acceptable amount of computation, and 2) a Reference Trajectory-based Multi-Class Transformer (RTMCT) for efficient and diverse trajectory prediction of mult-class objects with flexible trajectory lengths. Evaluations on the CODa benchmark demonstrate superior performance over existing methods across detection (+2.03% in mAP) and trajectory prediction (-0.408m in minADE5 of pedestrians) metrics. Remarkably, the system exhibits exceptional deployability - when implemented on a wheelchair robot with an entry-level NVIDIA 3060 GPU, it achieves real-time inference at 13.2 fps. To facilitate reproducibility and practical deployment, we release the related code of the method at this https URL and its ROS inference version at this https URL. 

**Abstract (ZH)**: 一种轻量级多模态的3D物体检测与轨迹预测框架 

---
# Robot Navigation in Dynamic Environments using Acceleration Obstacles 

**Title (ZH)**: 动态环境中国基于加速度障碍的机器人导航 

**Authors**: Asher Stern, Zvi Shiller  

**Link**: [PDF](https://arxiv.org/pdf/2504.13637)  

**Abstract**: This paper addresses the issue of motion planning in dynamic environments by extending the concept of Velocity Obstacle and Nonlinear Velocity Obstacle to Acceleration Obstacle AO and Nonlinear Acceleration Obstacle NAO. Similarly to VO and NLVO, the AO and NAO represent the set of colliding constant accelerations of the maneuvering robot with obstacles moving along linear and nonlinear trajectories, respectively. Contrary to prior works, we derive analytically the exact boundaries of AO and NAO. To enhance an intuitive understanding of these representations, we first derive the AO in several steps: first extending the VO to the Basic Acceleration Obstacle BAO that consists of the set of constant accelerations of the robot that would collide with an obstacle moving at constant accelerations, while assuming zero initial velocities of the robot and obstacle. This is then extended to the AO while assuming arbitrary initial velocities of the robot and obstacle. And finally, we derive the NAO that in addition to the prior assumptions, accounts for obstacles moving along arbitrary trajectories. The introduction of NAO allows the generation of safe avoidance maneuvers that directly account for the robot's second-order dynamics, with acceleration as its control input. The AO and NAO are demonstrated in several examples of selecting avoidance maneuvers in challenging road traffic. It is shown that the use of NAO drastically reduces the adjustment rate of the maneuvering robot's acceleration while moving in complex road traffic scenarios. The presented approach enables reactive and efficient navigation for multiple robots, with potential application for autonomous vehicles operating in complex dynamic environments. 

**Abstract (ZH)**: 基于加速度障碍和非线性加速度障碍的动态环境下的运动规划 

---
# Robust Humanoid Walking on Compliant and Uneven Terrain with Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的鲁棒类人形机器人在非刚性不平地形上的行走 

**Authors**: Rohan P. Singh, Mitsuharu Morisawa, Mehdi Benallegue, Zhaoming Xie, Fumio Kanehiro  

**Link**: [PDF](https://arxiv.org/pdf/2504.13619)  

**Abstract**: For the deployment of legged robots in real-world environments, it is essential to develop robust locomotion control methods for challenging terrains that may exhibit unexpected deformability and irregularity. In this paper, we explore the application of sim-to-real deep reinforcement learning (RL) for the design of bipedal locomotion controllers for humanoid robots on compliant and uneven terrains. Our key contribution is to show that a simple training curriculum for exposing the RL agent to randomized terrains in simulation can achieve robust walking on a real humanoid robot using only proprioceptive feedback. We train an end-to-end bipedal locomotion policy using the proposed approach, and show extensive real-robot demonstration on the HRP-5P humanoid over several difficult terrains inside and outside the lab environment. Further, we argue that the robustness of a bipedal walking policy can be improved if the robot is allowed to exhibit aperiodic motion with variable stepping frequency. We propose a new control policy to enable modification of the observed clock signal, leading to adaptive gait frequencies depending on the terrain and command velocity. Through simulation experiments, we show the effectiveness of this policy specifically for walking over challenging terrains by controlling swing and stance durations. The code for training and evaluation is available online at this https URL. Demo video is available at this https URL. 

**Abstract (ZH)**: 基于仿真实验到现实应用的强化学习 userList 完人力足运动控制器设计：应对 compliant 和 uneven 地形 

---
# On the Importance of Tactile Sensing for Imitation Learning: A Case Study on Robotic Match Lighting 

**Title (ZH)**: 触觉感知对于模仿学习的重要性：一项关于机器人击灯任务的研究案例 

**Authors**: Niklas Funk, Changqi Chen, Tim Schneider, Georgia Chalvatzaki, Roberto Calandra, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2504.13618)  

**Abstract**: The field of robotic manipulation has advanced significantly in the last years. At the sensing level, several novel tactile sensors have been developed, capable of providing accurate contact information. On a methodological level, learning from demonstrations has proven an efficient paradigm to obtain performant robotic manipulation policies. The combination of both holds the promise to extract crucial contact-related information from the demonstration data and actively exploit it during policy rollouts. However, despite its potential, it remains an underexplored direction. This work therefore proposes a multimodal, visuotactile imitation learning framework capable of efficiently learning fast and dexterous manipulation policies. We evaluate our framework on the dynamic, contact-rich task of robotic match lighting - a task in which tactile feedback influences human manipulation performance. The experimental results show that adding tactile information into the policies significantly improves performance by over 40%, thereby underlining the importance of tactile sensing for contact-rich manipulation tasks. Project website: this https URL . 

**Abstract (ZH)**: 机器人操控领域在过去几年取得了显著进展。在传感层面，开发出了多种新型触觉传感器，能够提供准确的接触信息。在方法层面，从演示中学习已被证明是一种有效的范式，可以获取高性能的机器人操控策略。这两种方法的结合有望从演示数据中提取关键的接触相关信息，并在策略实施中积极加以利用。然而，尽管具有巨大潜力，这一方向 still remains largely unexplored。本工作因此提出了一种多模态的视触觉模仿学习框架，能够高效地学习快速灵巧的操控策略。我们在一个动态且触觉信息丰富的机器人火柴点火任务中评估了该框架，这是一个触觉反馈影响人类操控性能的任务。实验结果表明，将触觉信息加入到策略中可以显著提高性能，超过40%，从而突显了触觉传感对于触觉丰富操控任务的重要性。项目网站: this https URL。 

---
# Hysteresis-Aware Neural Network Modeling and Whole-Body Reinforcement Learning Control of Soft Robots 

**Title (ZH)**: 具有滞回效应意识的神经网络建模与软机器人全身强化学习控制 

**Authors**: Zongyuan Chen, Yan Xia, Jiayuan Liu, Jijia Liu, Wenhao Tang, Jiayu Chen, Feng Gao, Longfei Ma, Hongen Liao, Yu Wang, Chao Yu, Boyu Zhang, Fei Xing  

**Link**: [PDF](https://arxiv.org/pdf/2504.13582)  

**Abstract**: Soft robots exhibit inherent compliance and safety, which makes them particularly suitable for applications requiring direct physical interaction with humans, such as surgical procedures. However, their nonlinear and hysteretic behavior, resulting from the properties of soft materials, presents substantial challenges for accurate modeling and control. In this study, we present a soft robotic system designed for surgical applications and propose a hysteresis-aware whole-body neural network model that accurately captures and predicts the soft robot's whole-body motion, including its hysteretic behavior. Building upon the high-precision dynamic model, we construct a highly parallel simulation environment for soft robot control and apply an on-policy reinforcement learning algorithm to efficiently train whole-body motion control strategies. Based on the trained control policy, we developed a soft robotic system for surgical applications and validated it through phantom-based laser ablation experiments in a physical environment. The results demonstrate that the hysteresis-aware modeling reduces the Mean Squared Error (MSE) by 84.95 percent compared to traditional modeling methods. The deployed control algorithm achieved a trajectory tracking error ranging from 0.126 to 0.250 mm on the real soft robot, highlighting its precision in real-world conditions. The proposed method showed strong performance in phantom-based surgical experiments and demonstrates its potential for complex scenarios, including future real-world clinical applications. 

**Abstract (ZH)**: 软体机器人具有固有的柔顺性和安全性，特别适合需要与人类直接物理互动的应用，如手术程序。然而，由软材料性质引起的非线性和滞回行为给准确建模和控制带来了重大挑战。在本研究中，我们提出了一种设计用于手术应用的软体机器人系统，并提出了一种滞回意识的全身神经网络模型，能够准确捕捉和预测软体机器人的全身运动，包括其滞回行为。基于高精度动力学模型，我们构建了一个高度并行的软体机器人控制仿真环境，并应用了基于策略的强化学习算法高效训练全身运动控制策略。基于训练得到的控制策略，我们开发了一种用于手术应用的软体机器人系统，并通过物理环境中基于仿体的激光消融实验进行了验证。结果显示，滞回意识建模相比传统建模方法将均方误差（MSE）降低了84.95%。部署的控制算法在实际软体机器人上的轨迹跟踪误差范围为0.126至0.250毫米，突显了其在现实条件下的精确性。所提出的方法在基于仿体的手术实验中表现出色，并展示了其在未来复杂场景，包括实际临床应用中的潜力。 

---
# An Addendum to NeBula: Towards Extending TEAM CoSTAR's Solution to Larger Scale Environments 

**Title (ZH)**: NeBula的补充：朝向扩展TEAM CoSTAR解决方案以应对更大规模环境的方向 

**Authors**: Ali Agha, Kyohei Otsu, Benjamin Morrell, David D. Fan, Sung-Kyun Kim, Muhammad Fadhil Ginting, Xianmei Lei, Jeffrey Edlund, Seyed Fakoorian, Amanda Bouman, Fernando Chavez, Taeyeon Kim, Gustavo J. Correa, Maira Saboia, Angel Santamaria-Navarro, Brett Lopez, Boseong Kim, Chanyoung Jung, Mamoru Sobue, Oriana Claudia Peltzer, Joshua Ott, Robert Trybula, Thomas Touma, Marcel Kaufmann, Tiago Stegun Vaquero, Torkom Pailevanian, Matteo Palieri, Yun Chang, Andrzej Reinke, Matthew Anderson, Frederik E.T. Schöller, Patrick Spieler, Lillian M. Clark, Avak Archanian, Kenny Chen, Hovhannes Melikyan, Anushri Dixit, Harrison Delecki, Daniel Pastor, Barry Ridge, Nicolas Marchal, Jose Uribe, Sharmita Dey, Kamak Ebadi, Kyle Coble, Alexander Nikitas Dimopoulos, Vivek Thangavelu, Vivek S. Varadharajan, Nicholas Palomo, Antoni Rosinol, Arghya Chatterjee, Christoforos Kanellakis, Bjorn Lindqvist, Micah Corah, Kyle Strickland, Ryan Stonebraker, Michael Milano, Christopher E. Denniston, Sami Sahnoune, Thomas Claudet, Seungwook Lee, Gautam Salhotra, Edward Terry, Rithvik Musuku, Robin Schmid, Tony Tran, Ara Kourchians, Justin Schachter, Hector Azpurua, Levi Resende, Arash Kalantari, Jeremy Nash, Josh Lee, Christopher Patterson, Jennifer G. Blank, Kartik Patath, Yuki Kubo, Ryan Alimo, Yasin Almalioglu, Aaron Curtis, Jacqueline Sly, Tesla Wells, Nhut T. Ho, Mykel Kochenderfer, Giovanni Beltrame, George Nikolakopoulos, David Shim, Luca Carlone, Joel Burdick  

**Link**: [PDF](https://arxiv.org/pdf/2504.13461)  

**Abstract**: This paper presents an appendix to the original NeBula autonomy solution developed by the TEAM CoSTAR (Collaborative SubTerranean Autonomous Robots), participating in the DARPA Subterranean Challenge. Specifically, this paper presents extensions to NeBula's hardware, software, and algorithmic components that focus on increasing the range and scale of the exploration environment. From the algorithmic perspective, we discuss the following extensions to the original NeBula framework: (i) large-scale geometric and semantic environment mapping; (ii) an adaptive positioning system; (iii) probabilistic traversability analysis and local planning; (iv) large-scale POMDP-based global motion planning and exploration behavior; (v) large-scale networking and decentralized reasoning; (vi) communication-aware mission planning; and (vii) multi-modal ground-aerial exploration solutions. We demonstrate the application and deployment of the presented systems and solutions in various large-scale underground environments, including limestone mine exploration scenarios as well as deployment in the DARPA Subterranean challenge. 

**Abstract (ZH)**: 本文提供了由TEAM CoSTAR（协作地下自治机器人团队）开发的原始NeBula自主解决方案的附录，参与了DARPA地下挑战赛。具体而言，本文介绍了针对NeBula硬件、软件和算法组件的扩展，重点在于扩大探索环境的范围和规模。从算法角度来看，我们讨论了原始NeBula框架的以下扩展：(i) 大区域几何和语义环境建图；(ii) 适应性定位系统；(iii) 或然可通行性分析和局部规划；(iv) 基于大规模POMDP的全局运动规划和探索行为；(v) 大规模网络和去中心化推理；(vi) 通信意识的任务规划；以及(vii) 多模态地面-空中探索解决方案。我们展示了所提出系统的应用和部署在各种大型地下环境中，包括石灰岩矿井探索场景以及DARPA地下挑战赛中的部署。 

---
# Testing the Fault-Tolerance of Multi-Sensor Fusion Perception in Autonomous Driving Systems 

**Title (ZH)**: 测试自动驾驶系统中多传感器融合感知的容错性 

**Authors**: Haoxiang Tian, Wenqiang Ding, Xingshuo Han, Guoquan Wu, An Guo, Junqi Zhang. Wei Chen, Jun Wei, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13420)  

**Abstract**: High-level Autonomous Driving Systems (ADSs), such as Google Waymo and Baidu Apollo, typically rely on multi-sensor fusion (MSF) based approaches to perceive their surroundings. This strategy increases perception robustness by combining the respective strengths of the camera and LiDAR and directly affects the safety-critical driving decisions of autonomous vehicles (AVs). However, in real-world autonomous driving scenarios, cameras and LiDAR are subject to various faults, which can probably significantly impact the decision-making and behaviors of ADSs. Existing MSF testing approaches only discovered corner cases that the MSF-based perception cannot accurately detected by MSF-based perception, while lacking research on how sensor faults affect the system-level behaviors of ADSs.
To address this gap, we conduct the first exploration of the fault tolerance of MSF perception-based ADS for sensor faults. In this paper, we systematically and comprehensively build fault models for cameras and LiDAR in AVs and inject them into the MSF perception-based ADS to test its behaviors in test scenarios. To effectively and efficiently explore the parameter spaces of sensor fault models, we design a feedback-guided differential fuzzer to discover the safety violations of MSF perception-based ADS caused by the injected sensor faults. We evaluate FADE on the representative and practical industrial ADS, Baidu Apollo. Our evaluation results demonstrate the effectiveness and efficiency of FADE, and we conclude some useful findings from the experimental results. To validate the findings in the physical world, we use a real Baidu Apollo 6.0 EDU autonomous vehicle to conduct the physical experiments, and the results show the practical significance of our findings. 

**Abstract (ZH)**: 高层次自动驾驶系统中基于多传感器融合的容错研究：以摄像头和LiDAR故障为例 

---
# LangCoop: Collaborative Driving with Language 

**Title (ZH)**: LangCoop:协作式语言驱动行驶 

**Authors**: Xiangbo Gao, Yuheng Wu, Rujia Wang, Chenxi Liu, Yang Zhou, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13406)  

**Abstract**: Multi-agent collaboration holds great promise for enhancing the safety, reliability, and mobility of autonomous driving systems by enabling information sharing among multiple connected agents. However, existing multi-agent communication approaches are hindered by limitations of existing communication media, including high bandwidth demands, agent heterogeneity, and information loss. To address these challenges, we introduce LangCoop, a new paradigm for collaborative autonomous driving that leverages natural language as a compact yet expressive medium for inter-agent communication. LangCoop features two key innovations: Mixture Model Modular Chain-of-thought (M$^3$CoT) for structured zero-shot vision-language reasoning and Natural Language Information Packaging (LangPack) for efficiently packaging information into concise, language-based messages. Through extensive experiments conducted in the CARLA simulations, we demonstrate that LangCoop achieves a remarkable 96\% reduction in communication bandwidth (< 2KB per message) compared to image-based communication, while maintaining competitive driving performance in the closed-loop evaluation. 

**Abstract (ZH)**: 多代理合作通过利用自然语言作为一种紧凑且表达能力强的中介通信手段，极大地促进了自主驾驶系统的安全、可靠性和移动性。然而，现有的多代理通信方法受限于现有通信介质的局限性，包括高带宽需求、代理异构性和信息丢失。为应对这些挑战，我们提出了LangCoop，一种利用自然语言进行代理间通信的新范式。LangCoop 特征在于两种创新：混合模型模块化链式思考（M$^3$CoT）用于结构化零样本视觉语言推理，以及自然语言信息封装（LangPack）用于高效地将信息打包成简洁的语言消息。通过在CARLA仿真环境中进行广泛实验，我们表明，LangCoop 相较于基于图像的通信，在通信带宽上实现了96%的显著减少（每条消息少于2KB），同时在闭环评估中保持了具有竞争力的驾驶性能。 

---
# Multi-Sensor Fusion-Based Mobile Manipulator Remote Control for Intelligent Smart Home Assistance 

**Title (ZH)**: 基于多传感器融合的移动 manipulator 远程控制技术及其在智能智能家居辅助中的应用 

**Authors**: Xiao Jin, Bo Xiao, Huijiang Wang, Wendong Wang, Zhenhua Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13370)  

**Abstract**: This paper proposes a wearable-controlled mobile manipulator system for intelligent smart home assistance, integrating MEMS capacitive microphones, IMU sensors, vibration motors, and pressure feedback to enhance human-robot interaction. The wearable device captures forearm muscle activity and converts it into real-time control signals for mobile manipulation. The wearable device achieves an offline classification accuracy of 88.33\%\ across six distinct movement-force classes for hand gestures by using a CNN-LSTM model, while real-world experiments involving five participants yield a practical accuracy of 83.33\%\ with an average system response time of 1.2 seconds. In Human-Robot synergy in navigation and grasping tasks, the robot achieved a 98\%\ task success rate with an average trajectory deviation of only 3.6 cm. Finally, the wearable-controlled mobile manipulator system achieved a 93.3\%\ gripping success rate, a transfer success of 95.6\%\, and a full-task success rate of 91.1\%\ during object grasping and transfer tests, in which a total of 9 object-texture combinations were evaluated. These three experiments' results validate the effectiveness of MEMS-based wearable sensing combined with multi-sensor fusion for reliable and intuitive control of assistive robots in smart home scenarios. 

**Abstract (ZH)**: 基于MEMS传感器的可穿戴控制移动 manipulator系统智能智能家居辅助设计与实验研究 

---
# Chain-of-Modality: Learning Manipulation Programs from Multimodal Human Videos with Vision-Language-Models 

**Title (ZH)**: 多模态人体视频中基于视觉-语言模型的操控程序学习 

**Authors**: Chen Wang, Fei Xia, Wenhao Yu, Tingnan Zhang, Ruohan Zhang, C. Karen Liu, Li Fei-Fei, Jie Tan, Jacky Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13351)  

**Abstract**: Learning to perform manipulation tasks from human videos is a promising approach for teaching robots. However, many manipulation tasks require changing control parameters during task execution, such as force, which visual data alone cannot capture. In this work, we leverage sensing devices such as armbands that measure human muscle activities and microphones that record sound, to capture the details in the human manipulation process, and enable robots to extract task plans and control parameters to perform the same task. To achieve this, we introduce Chain-of-Modality (CoM), a prompting strategy that enables Vision Language Models to reason about multimodal human demonstration data -- videos coupled with muscle or audio signals. By progressively integrating information from each modality, CoM refines a task plan and generates detailed control parameters, enabling robots to perform manipulation tasks based on a single multimodal human video prompt. Our experiments show that CoM delivers a threefold improvement in accuracy for extracting task plans and control parameters compared to baselines, with strong generalization to new task setups and objects in real-world robot experiments. Videos and code are available at this https URL 

**Abstract (ZH)**: 从人类视频学习执行操作任务是教机器人的一项有前景的方法。然而，许多操作任务在执行过程中需要改变控制参数，例如力，而仅靠视觉数据无法捕捉到这些信息。在本文中，我们利用臂带等传感设备来测量人类的肌肉活动，以及录音设备来记录声音，以捕捉人类操作过程的细节，并使机器人能够提取任务计划和控制参数以执行相同的任务。为此，我们引入了一种链式模态（CoM）策略，该策略允许视觉语言模型对多模态人类演示数据进行推理——视频配上肌肉或音频信号。通过逐步整合每种模态的信息，CoM细化任务计划并生成详细的控制参数，使机器人能够根据单一的多模态人类视频提示执行操作任务。我们的实验结果显示，与基线方法相比，CoM在提取任务计划和控制参数的准确性上提高了三倍，并在真实世界机器人实验中对新任务布置和对象具有较强的泛化能力。更多视频和代码可访问：this https URL 

---
# Physical Reservoir Computing in Hook-Shaped Rover Wheel Spokes for Real-Time Terrain Identification 

**Title (ZH)**: 钩形机器人车轮辐条上的物理储槽计算用于实时地形识别 

**Authors**: Xiao Jin, Zihan Wang, Zhenhua Yu, Changrak Choi, Kalind Carpenter, Thrishantha Nanayakkara  

**Link**: [PDF](https://arxiv.org/pdf/2504.13348)  

**Abstract**: Effective terrain detection in unknown environments is crucial for safe and efficient robotic navigation. Traditional methods often rely on computationally intensive data processing, requiring extensive onboard computational capacity and limiting real-time performance for rovers. This study presents a novel approach that combines physical reservoir computing with piezoelectric sensors embedded in rover wheel spokes for real-time terrain identification. By leveraging wheel dynamics, terrain-induced vibrations are transformed into high-dimensional features for machine learning-based classification. Experimental results show that strategically placing three sensors on the wheel spokes achieves 90$\%$ classification accuracy, which demonstrates the accuracy and feasibility of the proposed method. The experiment results also showed that the system can effectively distinguish known terrains and identify unknown terrains by analyzing their similarity to learned categories. This method provides a robust, low-power framework for real-time terrain classification and roughness estimation in unstructured environments, enhancing rover autonomy and adaptability. 

**Abstract (ZH)**: 在未知环境中的有效地形检测对于机器人导航的安全与效率至关重要。传统方法往往依赖于计算密集型的数据处理，需要大量的车载计算能力，从而限制了漫游车的实时性能。本研究提出了一种新颖的方法，结合物理蓄水池计算与嵌入漫游车轮辐中的压电传感器进行实时地形识别。通过利用轮动动力学，地形引起的振动被转换为高维度特征用于机器学习分类。实验结果表明，在轮辐上战略位置放置三个传感器可实现90%的分类准确性，证明了所提方法的准确性和可行性。实验结果还显示，该系统可以通过分析未知地形与已学习类别之间的相似性，有效地区分已知地形和未知地形。该方法为在无结构环境中提供了一种稳健且低功耗的实时地形分类和粗糙度估计框架，增强了漫游车的自主性和适应性。 

---
# LMPOcc: 3D Semantic Occupancy Prediction Utilizing Long-Term Memory Prior from Historical Traversals 

**Title (ZH)**: LMPOcc: 利用历史通行先验的长期记忆进行3D语义占用预测 

**Authors**: Shanshuai Yuan, Julong Wei, Muer Tie, Xiangyun Ren, Zhongxue Gan, Wenchao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.13596)  

**Abstract**: Vision-based 3D semantic occupancy prediction is critical for autonomous driving, enabling unified modeling of static infrastructure and dynamic agents. In practice, autonomous vehicles may repeatedly traverse identical geographic locations under varying environmental conditions, such as weather fluctuations and illumination changes. Existing methods in 3D occupancy prediction predominantly integrate adjacent temporal contexts. However, these works neglect to leverage perceptual information, which is acquired from historical traversals of identical geographic locations. In this paper, we propose Longterm Memory Prior Occupancy (LMPOcc), the first 3D occupancy prediction methodology that exploits long-term memory priors derived from historical traversal perceptual outputs. We introduce a plug-and-play architecture that integrates long-term memory priors to enhance local perception while simultaneously constructing global occupancy representations. To adaptively aggregate prior features and current features, we develop an efficient lightweight Current-Prior Fusion module. Moreover, we propose a model-agnostic prior format to ensure compatibility across diverse occupancy prediction baselines. LMPOcc achieves state-of-the-art performance validated on the Occ3D-nuScenes benchmark, especially on static semantic categories. Additionally, experimental results demonstrate LMPOcc's ability to construct global occupancy through multi-vehicle crowdsourcing. 

**Abstract (ZH)**: 基于视觉的三维语义占用预测对于自动驾驶至关重要，能够实现静态基础设施和动态代理的统一建模。在实践中，自主车辆可能在不同环境条件下反复穿越相同的地理位置，例如天气变化和光照改变。现有的三维占用预测方法主要整合相邻的时空上下文。然而，这些方法忽略了利用从相同地理位置的历史穿越中获得的感觉信息。在本文中，我们提出了一种名为Longterm Memory Prior Occupancy (LMPOcc)的三维占用预测方法，这是首个利用历史穿越感觉输出中提取的长期记忆先验的三维占用预测方法。我们引入了一种即插即用架构，将长期记忆先验整合进来，增强局部感知的同时构建全局占用表示。为了适应性地聚合先验特征和当前特征，我们开发了一种高效的轻量级当前-先验融合模块。此外，我们提出了一个模型通用的先验格式，以确保与其他各种占用预测基线的兼容性。LMPOcc在Occ3D-nuScenes基准上实现了最先进的性能，特别是在静态语义类别方面。实验结果还展示了LMPOcc通过多车辆众包构建全局占用的能力。 

---
# Task Assignment and Exploration Optimization for Low Altitude UAV Rescue via Generative AI Enhanced Multi-agent Reinforcement Learning 

**Title (ZH)**: 基于生成AI增强的多agent强化学习的低空无人机救援任务分配与探索优化 

**Authors**: Xin Tang, Qian Chen, Wenjie Weng, Chao Jin, Zhang Liu, Jiacheng Wang, Geng Sun, Xiaohuan Li, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2504.13554)  

**Abstract**: Artificial Intelligence (AI)-driven convolutional neural networks enhance rescue, inspection, and surveillance tasks performed by low-altitude uncrewed aerial vehicles (UAVs) and ground computing nodes (GCNs) in unknown environments. However, their high computational demands often exceed a single UAV's capacity, leading to system instability, further exacerbated by the limited and dynamic resources of GCNs. To address these challenges, this paper proposes a novel cooperation framework involving UAVs, ground-embedded robots (GERs), and high-altitude platforms (HAPs), which enable resource pooling through UAV-to-GER (U2G) and UAV-to-HAP (U2H) communications to provide computing services for UAV offloaded tasks. Specifically, we formulate the multi-objective optimization problem of task assignment and exploration optimization in UAVs as a dynamic long-term optimization problem. Our objective is to minimize task completion time and energy consumption while ensuring system stability over time. To achieve this, we first employ the Lyapunov optimization technique to transform the original problem, with stability constraints, into a per-slot deterministic problem. We then propose an algorithm named HG-MADDPG, which combines the Hungarian algorithm with a generative diffusion model (GDM)-based multi-agent deep deterministic policy gradient (MADDPG) approach. We first introduce the Hungarian algorithm as a method for exploration area selection, enhancing UAV efficiency in interacting with the environment. We then innovatively integrate the GDM and multi-agent deep deterministic policy gradient (MADDPG) to optimize task assignment decisions, such as task offloading and resource allocation. Simulation results demonstrate the effectiveness of the proposed approach, with significant improvements in task offloading efficiency, latency reduction, and system stability compared to baseline methods. 

**Abstract (ZH)**: 基于人工智能驱动的卷积神经网络增强低空无人机和地面计算节点在未知环境中的救援、检查和 surveillance 任务，但其高计算需求通常超出单个无人机的能力，导致系统不稳定，进一步加剧了地面计算节点有限且动态的资源限制。为解决这些挑战，本文提出一种涉及无人机、地面嵌入式机器人和高空平台的新型合作框架，通过无人机到地面嵌入式机器人（U2G）和无人机到高空平台（U2H）通信实现资源池化，为卸载到无人机的任务提供计算服务。具体而言，我们将无人机中的任务分配和探索优化问题形式化为动态长期优化问题。我们的目标是在确保系统长期稳定的同时，最小化任务完成时间和能耗。为此，我们首先采用李雅普诺夫优化技术将原始问题（带有稳定约束）转化为每时段的确定性问题。然后，我们提出了一个名为HG-MADDPG的算法，该算法结合了匈牙利算法和基于生成扩散模型（GDM）的多智能体深度确定性策略梯度（MADDPG）方法。我们首先介绍了匈牙利算法作为探索区域选择的方法，增强无人机与环境的交互效率。然后，我们创新性地将GDM和多智能体深度确定性策略梯度（MADDPG）结合，优化任务分配决策，如任务卸载和资源分配。仿真结果证明了所提出方法的有效性，与基准方法相比，在任务卸载效率、延迟减少和系统稳定性方面取得了显著改进。 

---
# SwitchMT: An Adaptive Context Switching Methodology for Scalable Multi-Task Learning in Intelligent Autonomous Agents 

**Title (ZH)**: SwitchMT：一种针对智能自主代理可扩展多任务学习的自适应上下文切换方法ological框架 

**Authors**: Avaneesh Devkota, Rachmad Vidya Wicaksana Putra, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2504.13541)  

**Abstract**: The ability to train intelligent autonomous agents (such as mobile robots) on multiple tasks is crucial for adapting to dynamic real-world environments. However, state-of-the-art reinforcement learning (RL) methods only excel in single-task settings, and still struggle to generalize across multiple tasks due to task interference. Moreover, real-world environments also demand the agents to have data stream processing capabilities. Toward this, a state-of-the-art work employs Spiking Neural Networks (SNNs) to improve multi-task learning by exploiting temporal information in data stream, while enabling lowpower/energy event-based operations. However, it relies on fixed context/task-switching intervals during its training, hence limiting the scalability and effectiveness of multi-task learning. To address these limitations, we propose SwitchMT, a novel adaptive task-switching methodology for RL-based multi-task learning in autonomous agents. Specifically, SwitchMT employs the following key ideas: (1) a Deep Spiking Q-Network with active dendrites and dueling structure, that utilizes task-specific context signals to create specialized sub-networks; and (2) an adaptive task-switching policy that leverages both rewards and internal dynamics of the network parameters. Experimental results demonstrate that SwitchMT achieves superior performance in multi-task learning compared to state-of-the-art methods. It achieves competitive scores in multiple Atari games (i.e., Pong: -8.8, Breakout: 5.6, and Enduro: 355.2) compared to the state-of-the-art, showing its better generalized learning capability. These results highlight the effectiveness of our SwitchMT methodology in addressing task interference while enabling multi-task learning automation through adaptive task switching, thereby paving the way for more efficient generalist agents with scalable multi-task learning capabilities. 

**Abstract (ZH)**: 多重任务学习中智能自主代理的适应性任务切换方法 

---
# A Model-Based Approach to Imitation Learning through Multi-Step Predictions 

**Title (ZH)**: 基于模型的方法通过多步预测进行模仿学习 

**Authors**: Haldun Balim, Yang Hu, Yuyang Zhang, Na Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.13413)  

**Abstract**: Imitation learning is a widely used approach for training agents to replicate expert behavior in complex decision-making tasks. However, existing methods often struggle with compounding errors and limited generalization, due to the inherent challenge of error correction and the distribution shift between training and deployment. In this paper, we present a novel model-based imitation learning framework inspired by model predictive control, which addresses these limitations by integrating predictive modeling through multi-step state predictions. Our method outperforms traditional behavior cloning numerical benchmarks, demonstrating superior robustness to distribution shift and measurement noise both in available data and during execution. Furthermore, we provide theoretical guarantees on the sample complexity and error bounds of our method, offering insights into its convergence properties. 

**Abstract (ZH)**: 基于模型的预测控制启发 imitation 学习框架：通过多步状态预测克服分布偏移和测量噪声 

---
# Integration of a Graph-Based Path Planner and Mixed-Integer MPC for Robot Navigation in Cluttered Environments 

**Title (ZH)**: 基于图的路径规划器与混合整数MPC在杂乱环境中的机器人导航集成 

**Authors**: Joshua A. Robbins, Stephen J. Harnett, Andrew F. Thompson, Sean Brennan, Herschel C. Pangborn  

**Link**: [PDF](https://arxiv.org/pdf/2504.13372)  

**Abstract**: The ability to update a path plan is a required capability for autonomous mobile robots navigating through uncertain environments. This paper proposes a re-planning strategy using a multilayer planning and control framework for cases where the robot's environment is partially known. A medial axis graph-based planner defines a global path plan based on known obstacles where each edge in the graph corresponds to a unique corridor. A mixed-integer model predictive control (MPC) method detects if a terminal constraint derived from the global plan is infeasible, subject to a non-convex description of the local environment. Infeasibility detection is used to trigger efficient global re-planning via medial axis graph edge deletion. The proposed re-planning strategy is demonstrated experimentally. 

**Abstract (ZH)**: 自主移动机器人在部分已知环境中的路径更新策略研究 

---
# BEV-GS: Feed-forward Gaussian Splatting in Bird's-Eye-View for Road Reconstruction 

**Title (ZH)**: BEV-GS: 鸟瞰视角下的前馈高斯绘制方法用于道路重建 

**Authors**: Wenhua Wu, Tong Zhao, Chensheng Peng, Lei Yang, Yintao Wei, Zhe Liu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13207)  

**Abstract**: Road surface is the sole contact medium for wheels or robot feet. Reconstructing road surface is crucial for unmanned vehicles and mobile robots. Recent studies on Neural Radiance Fields (NeRF) and Gaussian Splatting (GS) have achieved remarkable results in scene reconstruction. However, they typically rely on multi-view image inputs and require prolonged optimization times. In this paper, we propose BEV-GS, a real-time single-frame road surface reconstruction method based on feed-forward Gaussian splatting. BEV-GS consists of a prediction module and a rendering module. The prediction module introduces separate geometry and texture networks following Bird's-Eye-View paradigm. Geometric and texture parameters are directly estimated from a single frame, avoiding per-scene optimization. In the rendering module, we utilize grid Gaussian for road surface representation and novel view synthesis, which better aligns with road surface characteristics. Our method achieves state-of-the-art performance on the real-world dataset RSRD. The road elevation error reduces to 1.73 cm, and the PSNR of novel view synthesis reaches 28.36 dB. The prediction and rendering FPS is 26, and 2061, respectively, enabling high-accuracy and real-time applications. The code will be available at: \href{this https URL}{\texttt{this https URL}} 

**Abstract (ZH)**: 基于前向高斯散射的鸟瞰图单帧道路表面重建方法 

---
