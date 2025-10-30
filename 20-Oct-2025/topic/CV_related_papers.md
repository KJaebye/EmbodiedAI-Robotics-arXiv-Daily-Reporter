# Dynamic Recalibration in LiDAR SLAM: Integrating AI and Geometric Methods with Real-Time Feedback Using INAF Fusion 

**Title (ZH)**: 基于INAF融合的实时反馈结合AI和几何方法的LiDAR SLAM动态校准 

**Authors**: Zahra Arjmandi, Gunho Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2510.15803)  

**Abstract**: This paper presents a novel fusion technique for LiDAR Simultaneous Localization and Mapping (SLAM), aimed at improving localization and 3D mapping using LiDAR sensor. Our approach centers on the Inferred Attention Fusion (INAF) module, which integrates AI with geometric odometry. Utilizing the KITTI dataset's LiDAR data, INAF dynamically adjusts attention weights based on environmental feedback, enhancing the system's adaptability and measurement accuracy. This method advances the precision of both localization and 3D mapping, demonstrating the potential of our fusion technique to enhance autonomous navigation systems in complex scenarios. 

**Abstract (ZH)**: 本文提出了一种新颖的LiDAR SLAM融合技术，旨在通过LiDAR传感器提高定位和三维 mapping 的准确性。我们的方法集中在推理注意力融合（INAF）模块上，该模块结合了AI与几何位姿估计。利用KITTI数据集的LiDAR数据，INAF模块根据环境反馈动态调整注意力权重，增强系统的适应性和测量精度。该方法提高了定位和三维mapping的精确度，展示了我们的融合技术在复杂场景中增强自主导航系统的潜力。 

---
# Freehand 3D Ultrasound Imaging: Sim-in-the-Loop Probe Pose Optimization via Visual Servoing 

**Title (ZH)**: 自由手3D超声成像：基于视觉伺服的模拟在环探头姿态优化 

**Authors**: Yameng Zhang, Dianye Huang, Max Q.-H. Meng, Nassir Navab, Zhongliang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15668)  

**Abstract**: Freehand 3D ultrasound (US) imaging using conventional 2D probes offers flexibility and accessibility for diverse clinical applications but faces challenges in accurate probe pose estimation. Traditional methods depend on costly tracking systems, while neural network-based methods struggle with image noise and error accumulation, compromising reconstruction precision. We propose a cost-effective and versatile solution that leverages lightweight cameras and visual servoing in simulated environments for precise 3D US imaging. These cameras capture visual feedback from a textured planar workspace. To counter occlusions and lighting issues, we introduce an image restoration method that reconstructs occluded regions by matching surrounding texture patterns. For pose estimation, we develop a simulation-in-the-loop approach, which replicates the system setup in simulation and iteratively minimizes pose errors between simulated and real-world observations. A visual servoing controller refines the alignment of camera views, improving translational estimation by optimizing image alignment. Validations on a soft vascular phantom, a 3D-printed conical model, and a human arm demonstrate the robustness and accuracy of our approach, with Hausdorff distances to the reference reconstructions of 0.359 mm, 1.171 mm, and 0.858 mm, respectively. These results confirm the method's potential for reliable freehand 3D US reconstruction. 

**Abstract (ZH)**: 基于轻量级相机和仿真环境的视觉伺服自由手3D超声成像 

---
# VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation 

**Title (ZH)**: VO-DP: 基于语义-几何自适应扩散策略的纯视觉机器人 manipulation 

**Authors**: Zehao Ni, Yonghao He, Lingfeng Qian, Jilei Mao, Fa Fu, Wei Sui, Hu Su, Junran Peng, Zhipeng Wang, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2510.15530)  

**Abstract**: In the context of imitation learning, visuomotor-based diffusion policy learning is one of the main directions in robotic manipulation. Most of these approaches rely on point clouds as observation inputs and construct scene representations through point clouds feature learning, which enables them to achieve remarkable accuracy. However, the existing literature lacks an in-depth exploration of vision-only solutions that have significant potential. In this paper, we propose a Vision-Only and single-view Diffusion Policy learning method (VO-DP) that leverages pretrained visual foundation models to achieve effective fusion of semantic and geometric features. We utilize intermediate features from VGGT incorporating semantic features from DINOv2 and geometric features from Alternating Attention blocks. Features are fused via cross-attention and spatially compressed with a CNN to form the input to the policy head. Extensive experiments demonstrate that VO-DP not only outperforms the vision-only baseline DP significantly but also exhibits distinct performance trends against the point cloud-based method DP3: in simulation tasks, VO-DP achieves an average success rate of 64.6% on par with DP3 64.0% and far higher than DP 34.8%, while in real-world tasks, it reaches 87.9%, outperforming both DP3 67.5% and DP 11.2% by a notable margin. Further robustness evaluations confirm that VO-DP remains highly stable under varying conditions including color, size, background, and lighting. Lastly, we open-source a training library for robotic manipulation. Built on Accelerate, this library supports multi-machine and multi-GPU parallel training, as well as mixed precision training. It is compatible with visuomotor policies such as DP, DP3 and VO-DP, and also supports the RoboTwin simulator. 

**Abstract (ZH)**: 基于视觉的单视角扩散策略学习方法（VO-DP）：一种无需点云的数据驱动机器人 manipulation 方法 

---
