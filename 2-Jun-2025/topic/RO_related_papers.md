# PB&J: Peanut Butter and Joints for Damped Articulation 

**Title (ZH)**: PB&J: 花生酱和关节用于阻尼 articulated 运动 

**Authors**: Avery S. Williamson, Michael J. Bennington, Ravesh Sukhnandan, Mrinali Nakhre, Yuemin Mao, Victoria A. Webster-Wood  

**Link**: [PDF](https://arxiv.org/pdf/2505.24860)  

**Abstract**: Many bioinspired robots mimic the rigid articulated joint structure of the human hand for grasping tasks, but experience high-frequency mechanical perturbations that can destabilize the system and negatively affect precision without a high-frequency controller. Despite having bandwidth-limited controllers that experience time delays between sensing and actuation, biological systems can respond successfully to and mitigate these high-frequency perturbations. Human joints include damping and stiffness that many rigid articulated bioinspired hand robots lack. To enable researchers to explore the effects of joint viscoelasticity in joint control, we developed a human-hand-inspired grasping robot with viscoelastic structures that utilizes accessible and bioderived materials to reduce the economic and environmental impact of prototyping novel robotic systems. We demonstrate that an elastic element at the finger joints is necessary to achieve concurrent flexion, which enables secure grasping of spherical objects. To significantly damp the manufactured finger joints, we modeled, manufactured, and characterized rotary dampers using peanut butter as an organic analog joint working fluid. Finally, we demonstrated that a real-time position-based controller could be used to successfully catch a lightweight falling ball. We developed this open-source, low-cost grasping platform that abstracts the morphological and mechanical properties of the human hand to enable researchers to explore questions about biomechanics in roboto that would otherwise be difficult to test in simulation or modeling. 

**Abstract (ZH)**: 一种受人类手部启发并具有黏弹结构的抓取机器人及其控制方法 

---
# System-integrated intrinsic static-dynamic pressure sensing enabled by charge excitation and 3D gradient engineering for autonomous robotic interaction 

**Title (ZH)**: 基于电荷激发和三维梯度工程的系统集成内在静动态压力感知技术及其在自主机器人交互中的应用 

**Authors**: Kequan Xia, Song Yang, Jianguo Lu, Min Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24645)  

**Abstract**: High-resolution pressure sensing that distinguishes static and dynamic inputs is vital for intelligent robotics but remains challenging for self-powered sensors. We present a self-powered intrinsic static-dynamic pressure sensor (iSD Sensor) that integrates charge excitation with a 3D gradient-engineered structure, achieving enhanced voltage outputs-over 25X for static and 15X for dynamic modes. The sensor exhibits multi-region sensitivities (up to 34.7 V/kPa static, 48.4 V/kPa dynamic), a low detection limit of 6.13 Pa, and rapid response/recovery times (83/43 ms). This design enables nuanced tactile perception and supports dual-mode robotic control: proportional actuation via static signals and fast triggering via dynamic inputs. Integrated into a wireless closed-loop system, the iSD Sensor enables precise functions such as finger bending, object grasping, and sign language output. 

**Abstract (ZH)**: 自供电内在静动态压力传感器（iSD传感器）：实现高压差检测与快速响应 

---
# DTR: Delaunay Triangulation-based Racing for Scaled Autonomous Racing 

**Title (ZH)**: 基于Delaunay三角剖分的缩放自主赛车技术 

**Authors**: Luca Tognoni, Neil Reichlin, Edoardo Ghignone, Nicolas Baumann, Steven Marty, Liam Boyle, Michele Magno  

**Link**: [PDF](https://arxiv.org/pdf/2505.24320)  

**Abstract**: Reactive controllers for autonomous racing avoid the computational overhead of full ee-Think-Act autonomy stacks by directly mapping sensor input to control actions, eliminating the need for localization and planning. A widely used reactive strategy is FTG, which identifies gaps in LiDAR range measurements and steers toward a chosen one. While effective on fully bounded circuits, FTG fails in scenarios with incomplete boundaries and is prone to driving into dead-ends, known as FTG-traps. This work presents DTR, a reactive controller that combines Delaunay triangulation, from raw LiDAR readings, with track boundary segmentation to extract a centerline while systematically avoiding FTG-traps. Compared to FTG, the proposed method achieves lap times that are 70\% faster and approaches the performance of map-dependent methods. With a latency of 8.95 ms and CPU usage of only 38.85\% on the robot's OBC, DTR is real-time capable and has been successfully deployed and evaluated in field experiments. 

**Abstract (ZH)**: 基于Delaunay三角剖分的响应式控制器：结合LiDAR读数和赛道边界分割规避FTG陷阱实现实时自动驾驶赛车控制 

---
# Safety-Aware Robust Model Predictive Control for Robotic Arms in Dynamic Environments 

**Title (ZH)**: 动态环境中具有安全意识的鲁棒模型预测控制方法研究（臂式机器人） 

**Authors**: Sanghyeon Nam, Dongmin Kim, Seung-Hwan Choi, Chang-Hyun Kim, Hyoeun Kwon, Hiroaki Kawamoto, Suwoong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.24209)  

**Abstract**: Robotic manipulators are essential for precise industrial pick-and-place operations, yet planning collision-free trajectories in dynamic environments remains challenging due to uncertainties such as sensor noise and time-varying delays. Conventional control methods often fail under these conditions, motivating the development of Robust MPC (RMPC) strategies with constraint tightening. In this paper, we propose a novel RMPC framework that integrates phase-based nominal control with a robust safety mode, allowing smooth transitions between safe and nominal operations. Our approach dynamically adjusts constraints based on real-time predictions of moving obstacles\textemdash whether human, robot, or other dynamic objects\textemdash thus ensuring continuous, collision-free operation. Simulation studies demonstrate that our controller improves both motion naturalness and safety, achieving faster task completion than conventional methods. 

**Abstract (ZH)**: 机器人 manipulators 对精密工业取放操作至关重要，但在动态环境中规划无碰撞路径仍因传感器噪声和时间变化的延迟等不确定性而具有挑战性。传统控制方法在此条件下常常失效，推动了具有约束紧化策略的鲁棒模型预测控制（RMPC）方法的发展。本文提出了一种新颖的 RMPC 框架，该框架集成了基于阶段的名义控制与鲁棒安全模式，允许在安全操作和名义操作之间平滑过渡。我们的方法根据移动障碍物（无论是人类、机器人还是其他动态物体）的实时预测动态调整约束，从而确保连续且无碰撞的操作。仿真研究显示，与传统方法相比，我们的控制器不仅能提高运动的自然性，还能提高安全性并实现更快的任务完成。 

---
# Exploiting Euclidean Distance Field Properties for Fast and Safe 3D planning with a modified Lazy Theta* 

**Title (ZH)**: 利用欧几里得距离场性质进行快速安全的3D规划与修改后的Lazy Theta*算法 

**Authors**: Jose A. Cobano, L. Merino, F. Caballero  

**Link**: [PDF](https://arxiv.org/pdf/2505.24024)  

**Abstract**: Graph search planners have been widely used for 3D path planning in the literature, and Euclidean Distance Fields (EDFs) are increasingly being used as a representation of the environment. However, to the best of our knowledge, the integration of EDFs into heuristic planning has been carried out in a loosely coupled fashion, dismissing EDF properties that can be used to accelerate/improve the planning process and enhance the safety margins of the resultant trajectories. This paper presents a fast graph search planner based on a modified Lazy Theta* planning algorithm for aerial robots in challenging 3D environments that exploits the EDF properties. The proposed planner outperforms classic graph search planners in terms of path smoothness and safety. It integrates EDFs as environment representation and directly generates fast and smooth paths avoiding the use of post-processing methods; it also considers the analytical properties of EDFs to obtain an approximation of the EDF cost along the line-of-sight segments and to reduce the number of visibility neighbours, which directly impacts the computation time. Moreover, we demonstrate that the proposed EDF-based cost function satisfies the triangle inequality, which reduces calculations during exploration and, hence, computation time. Many experiments and comparatives are carried out in 3D challenging indoor and outdoor simulation environments to evaluate and validate the proposed planner. The results show an efficient and safe planner in these environments. 

**Abstract (ZH)**: 基于改进Lazy Theta*算法的利用欧几里得距离场的快速图搜索规划器 

---
# Ergonomic Assessment of Work Activities for an Industrial-oriented Wrist Exoskeleton 

**Title (ZH)**: 面向工业应用的手腕外骨骼工作活动人机工程学评估 

**Authors**: Roberto F. Pitzalis, Nicholas Cartocci, Christian Di Natali, Luigi Monica, Darwin G. Caldwell, Giovanni Berselli, Jesús Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2505.20939)  

**Abstract**: Musculoskeletal disorders (MSD) are the most common cause of work-related injuries and lost production involving approximately 1.7 billion people worldwide and mainly affect low back (more than 50%) and upper limbs (more than 40%). It has a profound effect on both the workers affected and the company. This paper provides an ergonomic assessment of different work activities in a horse saddle-making company, involving 5 workers. This aim guides the design of a wrist exoskeleton to reduce the risk of musculoskeletal diseases wherever it is impossible to automate the production process. This evaluation is done either through subjective and objective measurement, respectively using questionnaires and by measurement of muscle activation with sEMG sensors. 

**Abstract (ZH)**: 肌肉骨骼疾病（MSD）是导致工作相关伤害和生产损失的最常见原因，全球约有17亿人受到影响，主要影响低背部（超过50%）和上肢（超过40%）。这些疾病对受影响的工人和公司都有深远的影响。本文对一家马鞍制造公司的不同工作活动进行了人机工程学评估，涉及5名工人。该评估旨在通过主观和客观测量（分别使用问卷调查和sEMG传感器测量肌肉激活）来指导设计一种腕部外骨骼，以降低在无法自动化生产过程中的肌肉骨骼疾病风险。 

---
# DiG-Net: Enhancing Quality of Life through Hyper-Range Dynamic Gesture Recognition in Assistive Robotics 

**Title (ZH)**: DiG-Net: 通过超远距离动态手势识别提升辅助机器人服务质量 

**Authors**: Eran Bamani Beeri, Eden Nissinman, Avishai Sintov  

**Link**: [PDF](https://arxiv.org/pdf/2505.24786)  

**Abstract**: Dynamic hand gestures play a pivotal role in assistive human-robot interaction (HRI), facilitating intuitive, non-verbal communication, particularly for individuals with mobility constraints or those operating robots remotely. Current gesture recognition methods are mostly limited to short-range interactions, reducing their utility in scenarios demanding robust assistive communication from afar. In this paper, we introduce a novel approach designed specifically for assistive robotics, enabling dynamic gesture recognition at extended distances of up to 30 meters, thereby significantly improving accessibility and quality of life. Our proposed Distance-aware Gesture Network (DiG-Net) effectively combines Depth-Conditioned Deformable Alignment (DADA) blocks with Spatio-Temporal Graph modules, enabling robust processing and classification of gesture sequences captured under challenging conditions, including significant physical attenuation, reduced resolution, and dynamic gesture variations commonly experienced in real-world assistive environments. We further introduce the Radiometric Spatio-Temporal Depth Attenuation Loss (RSTDAL), shown to enhance learning and strengthen model robustness across varying distances. Our model demonstrates significant performance improvement over state-of-the-art gesture recognition frameworks, achieving a recognition accuracy of 97.3% on a diverse dataset with challenging hyper-range gestures. By effectively interpreting gestures from considerable distances, DiG-Net significantly enhances the usability of assistive robots in home healthcare, industrial safety, and remote assistance scenarios, enabling seamless and intuitive interactions for users regardless of physical limitations 

**Abstract (ZH)**: 动态手势在辅助人机交互（HRI）中扮演关键角色，促进直观的非言语通信，特别适用于行动受限的个体或远程操作机器人的人。当前的手势识别方法 mostly 限制在短距离交互，这在需要远程稳健辅助通信的场景中降低了实用性。本文提出了一种针对辅助机器人设计的新方法，能够在30米以上的距离实现动态手势识别，从而显著提高无障碍性和生活质量。我们提出的距离感知手势网络（DiG-Net）有效结合了深度条件变形对齐（DADA）模块和时空图模块，能够在包括显著物理衰减、降低分辨率和常见的动态手势变异等挑战条件下，实现稳健的手势序列处理和分类。此外，我们引入了辐射度时空深度衰减损失（RSTDAL），证明了其能够增强学习并提高模型在不同距离下的鲁棒性。实验结果表明，DiG-Net 在一个包含具有挑战性的超远程手势的多样数据集上的识别准确率达到97.3%，显著优于当前最先进的手势识别框架。通过有效解读远距离手势，DiG-Net 显著提升了在家用护理、工业安全和远程协助场景中辅助机器人的可用性，使用户能够实现无障碍的无缝和直观交互。 

---
# 4,500 Seconds: Small Data Training Approaches for Deep UAV Audio Classification 

**Title (ZH)**: 4,500秒：面向深度无人机音频分类的小数据训练方法 

**Authors**: Andrew P. Berg, Qian Zhang, Mia Y. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23782)  

**Abstract**: Unmanned aerial vehicle (UAV) usage is expected to surge in the coming decade, raising the need for heightened security measures to prevent airspace violations and security threats. This study investigates deep learning approaches to UAV classification focusing on the key issue of data scarcity. To investigate this we opted to train the models using a total of 4,500 seconds of audio samples, evenly distributed across a 9-class dataset. We leveraged parameter efficient fine-tuning (PEFT) and data augmentations to mitigate the data scarcity. This paper implements and compares the use of convolutional neural networks (CNNs) and attention-based transformers. Our results show that, CNNs outperform transformers by 1-2\% accuracy, while still being more computationally efficient. These early findings, however, point to potential in using transformers models; suggesting that with more data and further optimizations they could outperform CNNs. Future works aims to upscale the dataset to better understand the trade-offs between these approaches. 

**Abstract (ZH)**: 无人飞行器(UAV)的使用预计将在未来十年大幅增长，从而提高了对 airspace违规和安全威胁预防的更高安全措施的需求。本研究探讨了深度学习方法在无人机分类中的应用，重点关注数据稀缺性这一关键问题。为研究这一问题，我们使用了总计4500秒的音频样本进行模型训练，这些样本被均匀分配在九类数据集中。我们利用参数高效微调(PEFT)和数据增强来缓解数据稀缺性。本文实现了并比较了卷积神经网络(CNNs)和基于注意力的变换器的使用。结果显示，卷积神经网络的准确率比变换器高1-2%，同时计算效率更高。然而，这些初步发现表明，变换器模型具有潜力，随着数据量的增加和进一步优化，它们可能会超越卷积神经网络。未来的工作旨在扩大数据集，以更好地理解这些方法之间的权衡。 

---
