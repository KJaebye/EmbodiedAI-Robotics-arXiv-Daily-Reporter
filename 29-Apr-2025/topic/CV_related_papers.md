# GAN-SLAM: Real-Time GAN Aided Floor Plan Creation Through SLAM 

**Title (ZH)**: GAN-SLAM：通过SLAM的实时GAN辅助平面图创建 

**Authors**: Leon Davies, Baihua Li, Mohamad Saada, Simon Sølvsten, Qinggang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.19653)  

**Abstract**: SLAM is a fundamental component of modern autonomous systems, providing robots and their operators with a deeper understanding of their environment. SLAM systems often encounter challenges due to the dynamic nature of robotic motion, leading to inaccuracies in mapping quality, particularly in 2D representations such as Occupancy Grid Maps. These errors can significantly degrade map quality, hindering the effectiveness of specific downstream tasks such as floor plan creation. To address this challenge, we introduce our novel 'GAN-SLAM', a new SLAM approach that leverages Generative Adversarial Networks to clean and complete occupancy grids during the SLAM process, reducing the impact of noise and inaccuracies introduced on the output map. We adapt and integrate accurate pose estimation techniques typically used for 3D SLAM into a 2D form. This enables the quality improvement 3D LiDAR-odometry has seen in recent years to be effective for 2D representations. Our results demonstrate substantial improvements in map fidelity and quality, with minimal noise and errors, affirming the effectiveness of GAN-SLAM for real-world mapping applications within large-scale complex environments. We validate our approach on real-world data operating in real-time, and on famous examples of 2D maps. The improved quality of the output map enables new downstream tasks, such as floor plan drafting, further enhancing the capabilities of autonomous systems. Our novel approach to SLAM offers a significant step forward in the field, improving the usability for SLAM in mapping-based tasks, and offers insight into the usage of GANs for OGM error correction. 

**Abstract (ZH)**: 基于GAN的SLAM：.dynamic环境下的 occupancy网格净化与完成 

---
# ARTEMIS: Autoregressive End-to-End Trajectory Planning with Mixture of Experts for Autonomous Driving 

**Title (ZH)**: ARTEMIS：基于专家混合的自回归端到端轨迹规划自主驾驶 

**Authors**: Renju Feng, Ning Xi, Duanfeng Chu, Rukang Wang, Zejian Deng, Anzheng Wang, Liping Lu, Jinxiang Wang, Yanjun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19580)  

**Abstract**: This paper presents ARTEMIS, an end-to-end autonomous driving framework that combines autoregressive trajectory planning with Mixture-of-Experts (MoE). Traditional modular methods suffer from error propagation, while existing end-to-end models typically employ static one-shot inference paradigms that inadequately capture the dynamic changes of the environment. ARTEMIS takes a different method by generating trajectory waypoints sequentially, preserves critical temporal dependencies while dynamically routing scene-specific queries to specialized expert networks. It effectively relieves trajectory quality degradation issues encountered when guidance information is ambiguous, and overcomes the inherent representational limitations of singular network architectures when processing diverse driving scenarios. Additionally, we use a lightweight batch reallocation strategy that significantly improves the training speed of the Mixture-of-Experts model. Through experiments on the NAVSIM dataset, ARTEMIS exhibits superior competitive performance, achieving 87.0 PDMS and 83.1 EPDMS with ResNet-34 backbone, demonstrates state-of-the-art performance on multiple metrics. 

**Abstract (ZH)**: ARTEMIS：一种结合自回归轨迹规划与Mixture-of-Experts的端到端自主驾驶框架 

---
# GSFF-SLAM: 3D Semantic Gaussian Splatting SLAM via Feature Field 

**Title (ZH)**: 基于特征场的3D语义高斯点云SLAM：GSFF-SLAM 

**Authors**: Zuxing Lu, Xin Yuan, Shaowen Yang, Jingyu Liu, Jiawei Wang, Changyin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.19409)  

**Abstract**: Semantic-aware 3D scene reconstruction is essential for autonomous robots to perform complex interactions. Semantic SLAM, an online approach, integrates pose tracking, geometric reconstruction, and semantic mapping into a unified framework, shows significant potential. However, existing systems, which rely on 2D ground truth priors for supervision, are often limited by the sparsity and noise of these signals in real-world environments. To address this challenge, we propose GSFF-SLAM, a novel dense semantic SLAM system based on 3D Gaussian Splatting that leverages feature fields to achieve joint rendering of appearance, geometry, and N-dimensional semantic features. By independently optimizing feature gradients, our method supports semantic reconstruction using various forms of 2D priors, particularly sparse and noisy signals. Experimental results demonstrate that our approach outperforms previous methods in both tracking accuracy and photorealistic rendering quality. When utilizing 2D ground truth priors, GSFF-SLAM achieves state-of-the-art semantic segmentation performance with 95.03\% mIoU, while achieving up to 2.9$\times$ speedup with only marginal performance degradation. 

**Abstract (ZH)**: 基于3D高斯喷溅的语义SLAM：GSFF-SLAM系统的语义意识3D场景重建 

---
# NANO-SLAM : Natural Gradient Gaussian Approximation for Vehicle SLAM 

**Title (ZH)**: NANO-SLAM：车辆SLAM的自然梯度高斯近似 

**Authors**: Tianyi Zhang, Wenhan Cao, Chang Liu, Feihong Zhang, Wei Wu, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.19195)  

**Abstract**: Accurate localization is a challenging task for autonomous vehicles, particularly in GPS-denied environments such as urban canyons and tunnels. In these scenarios, simultaneous localization and mapping (SLAM) offers a more robust alternative to GPS-based positioning, enabling vehicles to determine their position using onboard sensors and surrounding environment's landmarks. Among various vehicle SLAM approaches, Rao-Blackwellized particle filter (RBPF) stands out as one of the most widely adopted methods due to its efficient solution with logarithmic complexity relative to the map size. RBPF approximates the posterior distribution of the vehicle pose using a set of Monte Carlo particles through two main steps: sampling and importance weighting. The key to effective sampling lies in solving a distribution that closely approximates the posterior, known as the sampling distribution, to accelerate convergence. Existing methods typically derive this distribution via linearization, which introduces significant approximation errors due to the inherent nonlinearity of the system. To address this limitation, we propose a novel vehicle SLAM method called \textit{N}atural Gr\textit{a}dient Gaussia\textit{n} Appr\textit{o}ximation (NANO)-SLAM, which avoids linearization errors by modeling the sampling distribution as the solution to an optimization problem over Gaussian parameters and solving it using natural gradient descent. This approach improves the accuracy of the sampling distribution and consequently enhances localization performance. Experimental results on the long-distance Sydney Victoria Park vehicle SLAM dataset show that NANO-SLAM achieves over 50\% improvement in localization accuracy compared to the most widely used vehicle SLAM algorithms, with minimal additional computational cost. 

**Abstract (ZH)**: 自然梯度高斯近似自主车辆SLAM（NANO-SLAM） 

---
# Vysics: Object Reconstruction Under Occlusion by Fusing Vision and Contact-Rich Physics 

**Title (ZH)**: Vysics: 在接触丰富物理融合下的物体遮挡重建 

**Authors**: Bibit Bianchini, Minghan Zhu, Mengti Sun, Bowen Jiang, Camillo J. Taylor, Michael Posa  

**Link**: [PDF](https://arxiv.org/pdf/2504.18719)  

**Abstract**: We introduce Vysics, a vision-and-physics framework for a robot to build an expressive geometry and dynamics model of a single rigid body, using a seconds-long RGBD video and the robot's proprioception. While the computer vision community has built powerful visual 3D perception algorithms, cluttered environments with heavy occlusions can limit the visibility of objects of interest. However, observed motion of partially occluded objects can imply physical interactions took place, such as contact with a robot or the environment. These inferred contacts can supplement the visible geometry with "physible geometry," which best explains the observed object motion through physics. Vysics uses a vision-based tracking and reconstruction method, BundleSDF, to estimate the trajectory and the visible geometry from an RGBD video, and an odometry-based model learning method, Physics Learning Library (PLL), to infer the "physible" geometry from the trajectory through implicit contact dynamics optimization. The visible and "physible" geometries jointly factor into optimizing a signed distance function (SDF) to represent the object shape. Vysics does not require pretraining, nor tactile or force sensors. Compared with vision-only methods, Vysics yields object models with higher geometric accuracy and better dynamics prediction in experiments where the object interacts with the robot and the environment under heavy occlusion. Project page: this https URL 

**Abstract (ZH)**: 视觉与物理框架Vysics：基于秒级RGBD视频和机器人本体感受构建单个刚体的表达几何与动力学模型 

---
# Decentralized Fusion of 3D Extended Object Tracking based on a B-Spline Shape Model 

**Title (ZH)**: 基于B-Spline形状模型的分布式三维扩展目标跟踪融合 

**Authors**: Longfei Han, Klaus Kefferpütz, Jürgen Beyerer  

**Link**: [PDF](https://arxiv.org/pdf/2504.18708)  

**Abstract**: Extended Object Tracking (EOT) exploits the high resolution of modern sensors for detailed environmental perception. Combined with decentralized fusion, it contributes to a more scalable and robust perception system. This paper investigates the decentralized fusion of 3D EOT using a B-spline curve based model. The spline curve is used to represent the side-view profile, which is then extruded with a width to form a 3D shape. We use covariance intersection (CI) for the decentralized fusion and discuss the challenge of applying it to EOT. We further evaluate the tracking result of the decentralized fusion with simulated and real datasets of traffic scenarios. We show that the CI-based fusion can significantly improve the tracking performance for sensors with unfavorable perspective. 

**Abstract (ZH)**: 基于B-样条曲线模型的分布式融合扩展目标跟踪 

---
# MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion 

**Title (ZH)**: MP-SfM: 单目表面先验用于稳健的结构从运动重建 

**Authors**: Zador Pataki, Paul-Edouard Sarlin, Johannes L. Schönberger, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2504.20040)  

**Abstract**: While Structure-from-Motion (SfM) has seen much progress over the years, state-of-the-art systems are prone to failure when facing extreme viewpoint changes in low-overlap, low-parallax or high-symmetry scenarios. Because capturing images that avoid these pitfalls is challenging, this severely limits the wider use of SfM, especially by non-expert users. We overcome these limitations by augmenting the classical SfM paradigm with monocular depth and normal priors inferred by deep neural networks. Thanks to a tight integration of monocular and multi-view constraints, our approach significantly outperforms existing ones under extreme viewpoint changes, while maintaining strong performance in standard conditions. We also show that monocular priors can help reject faulty associations due to symmetries, which is a long-standing problem for SfM. This makes our approach the first capable of reliably reconstructing challenging indoor environments from few images. Through principled uncertainty propagation, it is robust to errors in the priors, can handle priors inferred by different models with little tuning, and will thus easily benefit from future progress in monocular depth and normal estimation. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 尽管结构从运动（SfM）技术在过去几年取得了显著进步，最新的系统在低重叠、低视角差或高对称场景下的极端视角变化下容易失败。由于避免这些难题捕捉图像具有挑战性，这严重限制了SfM的广泛应用，尤其是对非专家用户。我们通过结合经典的SfM paradigm与由深度神经网络推断出的单目深度和法线先验，克服了这些限制。得益于单目和多视图约束的紧密整合，我们的方法在极端视角变化下显著优于现有方法，同时在标准条件下保持了强大的性能。我们还展示了单目先验可以帮助拒绝由于对称性引起的错误关联，这是SfM长期以来的一个难题。这使得我们的方法成为首个能够可靠地从少量图像重建具有挑战性的室内环境的方法。通过原理上的不确定性传播，该方法对于先验中的错误具有鲁棒性，能够处理由不同模型推断出的先验，且不需要大量调整，因此将容易从未来的单目深度和法线估计进展中受益。我们的代码已在以下网址公开：this https URL。 

---
# Category-Level and Open-Set Object Pose Estimation for Robotics 

**Title (ZH)**: 机器人领域类别级和开放集物体姿态估计 

**Authors**: Peter Hönig, Matthias Hirschmanner, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2504.19572)  

**Abstract**: Object pose estimation enables a variety of tasks in computer vision and robotics, including scene understanding and robotic grasping. The complexity of a pose estimation task depends on the unknown variables related to the target object. While instance-level methods already excel for opaque and Lambertian objects, category-level and open-set methods, where texture, shape, and size are partially or entirely unknown, still struggle with these basic material properties. Since texture is unknown in these scenarios, it cannot be used for disambiguating object symmetries, another core challenge of 6D object pose estimation. The complexity of estimating 6D poses with such a manifold of unknowns led to various datasets, accuracy metrics, and algorithmic solutions. This paper compares datasets, accuracy metrics, and algorithms for solving 6D pose estimation on the category-level. Based on this comparison, we analyze how to bridge category-level and open-set object pose estimation to reach generalization and provide actionable recommendations. 

**Abstract (ZH)**: 类别级6D对象姿态估计数据集、准确度指标和算法比较：如何弥合类别级与开放集对象姿态估计差距以实现泛化并提供可操作建议 

---
# OPAL: Visibility-aware LiDAR-to-OpenStreetMap Place Recognition via Adaptive Radial Fusion 

**Title (ZH)**: OPAL：基于可见性aware的LiDAR到OpenStreetMap的地物识别 via 自适应径向融合 

**Authors**: Shuhao Kang, Martin Y. Liao, Yan Xia, Olaf Wysocki, Boris Jutzi, Daniel Cremers  

**Link**: [PDF](https://arxiv.org/pdf/2504.19258)  

**Abstract**: LiDAR place recognition is a critical capability for autonomous navigation and cross-modal localization in large-scale outdoor environments. Existing approaches predominantly depend on pre-built 3D dense maps or aerial imagery, which impose significant storage overhead and lack real-time adaptability. In this paper, we propose OPAL, a novel network for LiDAR place recognition that leverages OpenStreetMap as a lightweight and up-to-date prior. Our key innovation lies in bridging the domain disparity between sparse LiDAR scans and structured OSM data through two carefully designed components: a cross-modal visibility mask that identifies maximal observable regions from both modalities to guide feature learning, and an adaptive radial fusion module that dynamically consolidates multiscale radial features into discriminative global descriptors. Extensive experiments on the augmented KITTI and KITTI-360 datasets demonstrate OPAL's superiority, achieving 15.98% higher recall at @1m threshold for top-1 retrieved matches while operating at 12x faster inference speeds compared to state-of-the-art approaches. Code and datasets are publicly available at: this https URL . 

**Abstract (ZH)**: 基于OpenStreetMap的LiDAR场所识别网络OPAL 

---
# LM-MCVT: A Lightweight Multi-modal Multi-view Convolutional-Vision Transformer Approach for 3D Object Recognition 

**Title (ZH)**: LM-MCVT：一种轻量级多模态多视图卷积-视觉变换器方法用于三维物体识别 

**Authors**: Songsong Xiong, Hamidreza Kasaei  

**Link**: [PDF](https://arxiv.org/pdf/2504.19256)  

**Abstract**: In human-centered environments such as restaurants, homes, and warehouses, robots often face challenges in accurately recognizing 3D objects. These challenges stem from the complexity and variability of these environments, including diverse object shapes. In this paper, we propose a novel Lightweight Multi-modal Multi-view Convolutional-Vision Transformer network (LM-MCVT) to enhance 3D object recognition in robotic applications. Our approach leverages the Globally Entropy-based Embeddings Fusion (GEEF) method to integrate multi-views efficiently. The LM-MCVT architecture incorporates pre- and mid-level convolutional encoders and local and global transformers to enhance feature extraction and recognition accuracy. We evaluate our method on the synthetic ModelNet40 dataset and achieve a recognition accuracy of 95.6% using a four-view setup, surpassing existing state-of-the-art methods. To further validate its effectiveness, we conduct 5-fold cross-validation on the real-world OmniObject3D dataset using the same configuration. Results consistently show superior performance, demonstrating the method's robustness in 3D object recognition across synthetic and real-world 3D data. 

**Abstract (ZH)**: 面向餐厅、家庭和仓库等以人为中心环境中的机器人3D物体识别挑战：一种轻量级多模态多视图卷积-视觉变换器网络（LM-MCVT）的研究 

---
# LRFusionPR: A Polar BEV-Based LiDAR-Radar Fusion Network for Place Recognition 

**Title (ZH)**: LRFusionPR: 一种基于极坐标BEV的LiDAR-雷达融合网络用于场所识别 

**Authors**: Zhangshuo Qi, Luqi Cheng, Zijie Zhou, Guangming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2504.19186)  

**Abstract**: In autonomous driving, place recognition is critical for global localization in GPS-denied environments. LiDAR and radar-based place recognition methods have garnered increasing attention, as LiDAR provides precise ranging, whereas radar excels in adverse weather resilience. However, effectively leveraging LiDAR-radar fusion for place recognition remains challenging. The noisy and sparse nature of radar data limits its potential to further improve recognition accuracy. In addition, heterogeneous radar configurations complicate the development of unified cross-modality fusion frameworks. In this paper, we propose LRFusionPR, which improves recognition accuracy and robustness by fusing LiDAR with either single-chip or scanning radar. Technically, a dual-branch network is proposed to fuse different modalities within the unified polar coordinate bird's eye view (BEV) representation. In the fusion branch, cross-attention is utilized to perform cross-modality feature interactions. The knowledge from the fusion branch is simultaneously transferred to the distillation branch, which takes radar as its only input to further improve the robustness. Ultimately, the descriptors from both branches are concatenated, producing the multimodal global descriptor for place retrieval. Extensive evaluations on multiple datasets demonstrate that our LRFusionPR achieves accurate place recognition, while maintaining robustness under varying weather conditions. Our open-source code will be released at this https URL. 

**Abstract (ZH)**: 基于LiDAR和雷达融合的自主驾驶场景识别方法 

---
# Towards Latency-Aware 3D Streaming Perception for Autonomous Driving 

**Title (ZH)**: 面向延迟感知的3D流式自动驾驶感知 

**Authors**: Jiaqi Peng, Tai Wang, Jiangmiao Pang, Yuan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.19115)  

**Abstract**: Although existing 3D perception algorithms have demonstrated significant improvements in performance, their deployment on edge devices continues to encounter critical challenges due to substantial runtime latency. We propose a new benchmark tailored for online evaluation by considering runtime latency. Based on the benchmark, we build a Latency-Aware 3D Streaming Perception (LASP) framework that addresses the latency issue through two primary components: 1) latency-aware history integration, which extends query propagation into a continuous process, ensuring the integration of historical feature regardless of varying latency; 2) latency-aware predictive detection, a module that compensates the detection results with the predicted trajectory and the posterior accessed latency. By incorporating the latency-aware mechanism, our method shows generalization across various latency levels, achieving an online performance that closely aligns with 80\% of its offline evaluation on the Jetson AGX Orin without any acceleration techniques. 

**Abstract (ZH)**: 虽然现有的3D感知算法在性能上已经取得了显著进步，但它们在边缘设备上的部署仍然面临由于运行时延迟巨大的挑战。我们提出了一种新的基准，旨在考虑运行时延迟进行在线评估。基于该基准，我们构建了一种latency-aware 3D流式感知（LASP）框架，该框架通过两个主要组成部分来解决延迟问题：1) latency-aware历史集成，将查询传播扩展为连续过程，确保在各种延迟情况下历史特征的集成；2) latency-aware预测检测模块，该模块利用预测轨迹和后验访问的延迟来补偿检测结果。通过引入latency-aware机制，我们的方法在各种延迟水平下表现出通用性，在Jetson AGX Orin上实现的在线性能与80%的离线评估结果相匹配，无需任何加速技术。 

---
# WLTCL: Wide Field-of-View 3-D LiDAR Truck Compartment Automatic Localization System 

**Title (ZH)**: WLTCL: 广视野3D LiDAR货箱自动定位系统 

**Authors**: Guodong Sun, Mingjing Li, Dingjie Liu, Mingxuan Liu, Bo Wu, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18870)  

**Abstract**: As an essential component of logistics automation, the automated loading system is becoming a critical technology for enhancing operational efficiency and safety. Precise automatic positioning of the truck compartment, which serves as the loading area, is the primary step in automated loading. However, existing methods have difficulty adapting to truck compartments of various sizes, do not establish a unified coordinate system for LiDAR and mobile manipulators, and often exhibit reliability issues in cluttered environments. To address these limitations, our study focuses on achieving precise automatic positioning of key points in large, medium, and small fence-style truck compartments in cluttered scenarios. We propose an innovative wide field-of-view 3-D LiDAR vehicle compartment automatic localization system. For vehicles of various sizes, this system leverages the LiDAR to generate high-density point clouds within an extensive field-of-view range. By incorporating parking area constraints, our vehicle point cloud segmentation method more effectively segments vehicle point clouds within the scene. Our compartment key point positioning algorithm utilizes the geometric features of the compartments to accurately locate the corner points, providing stackable spatial regions. Extensive experiments on our collected data and public datasets demonstrate that this system offers reliable positioning accuracy and reduced computational resource consumption, leading to its application and promotion in relevant fields. 

**Abstract (ZH)**: 物流自动化中必不可少的自动加载系统已成为提升操作效率和安全性的关键技术。卡车货箱的精确自动定位作为装载区域的第一步，至关重要。然而，现有的方法难以适应不同尺寸的卡车货箱，未能为LiDAR和移动 manipulator建立统一的坐标系统，且在复杂环境中可靠性差。为弥补这些局限性，本研究专注于在复杂场景中实现大、中、小型围栏式卡车货箱的关键点的精确自动定位。我们提出了一种创新的广角3D LiDAR车辆货箱自动定位系统。对于不同尺寸的车辆，该系统利用LiDAR在大范围视场内生成高密度点云。通过结合停车区域约束，我们的车辆点云分割方法更有效地在场景中分割车辆点云。我们提出的货箱关键点定位算法利用货箱的几何特征准确定位角点，提供堆叠的空间区域。在我们收集的数据集和公开数据集上的 extensive 实验显示，该系统提供了可靠的定位精度并减少了计算资源消耗，从而在相关领域得到了应用和推广。 

---
# LIRM: Large Inverse Rendering Model for Progressive Reconstruction of Shape, Materials and View-dependent Radiance Fields 

**Title (ZH)**: LIRM：大规模逆向渲染模型用于渐进重建形状、材料和视依赖辐射场 

**Authors**: Zhengqin Li, Dilin Wang, Ka Chen, Zhaoyang Lv, Thu Nguyen-Phuoc, Milim Lee, Jia-Bin Huang, Lei Xiao, Cheng Zhang, Yufeng Zhu, Carl S. Marshall, Yufeng Ren, Richard Newcombe, Zhao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.20026)  

**Abstract**: We present Large Inverse Rendering Model (LIRM), a transformer architecture that jointly reconstructs high-quality shape, materials, and radiance fields with view-dependent effects in less than a second. Our model builds upon the recent Large Reconstruction Models (LRMs) that achieve state-of-the-art sparse-view reconstruction quality. However, existing LRMs struggle to reconstruct unseen parts accurately and cannot recover glossy appearance or generate relightable 3D contents that can be consumed by standard Graphics engines. To address these limitations, we make three key technical contributions to build a more practical multi-view 3D reconstruction framework. First, we introduce an update model that allows us to progressively add more input views to improve our reconstruction. Second, we propose a hexa-plane neural SDF representation to better recover detailed textures, geometry and material parameters. Third, we develop a novel neural directional-embedding mechanism to handle view-dependent effects. Trained on a large-scale shape and material dataset with a tailored coarse-to-fine training scheme, our model achieves compelling results. It compares favorably to optimization-based dense-view inverse rendering methods in terms of geometry and relighting accuracy, while requiring only a fraction of the inference time. 

**Abstract (ZH)**: 大型逆渲染模型（LIRM）：一种在秒级时间内联合重建视点相关效果下的高质量形状、材料和辐射场的变换器架构 

---
# Breast Cancer Detection from Multi-View Screening Mammograms with Visual Prompt Tuning 

**Title (ZH)**: 多视角筛查乳腺X线摄影中的乳腺癌检测与视觉提示调谐 

**Authors**: Han Chen, Anne L. Martel  

**Link**: [PDF](https://arxiv.org/pdf/2504.19900)  

**Abstract**: Accurate detection of breast cancer from high-resolution mammograms is crucial for early diagnosis and effective treatment planning. Previous studies have shown the potential of using single-view mammograms for breast cancer detection. However, incorporating multi-view data can provide more comprehensive insights. Multi-view classification, especially in medical imaging, presents unique challenges, particularly when dealing with large-scale, high-resolution data. In this work, we propose a novel Multi-view Visual Prompt Tuning Network (MVPT-NET) for analyzing multiple screening mammograms. We first pretrain a robust single-view classification model on high-resolution mammograms and then innovatively adapt multi-view feature learning into a task-specific prompt tuning process. This technique selectively tunes a minimal set of trainable parameters (7\%) while retaining the robustness of the pre-trained single-view model, enabling efficient integration of multi-view data without the need for aggressive downsampling. Our approach offers an efficient alternative to traditional feature fusion methods, providing a more robust, scalable, and efficient solution for high-resolution mammogram analysis. Experimental results on a large multi-institution dataset demonstrate that our method outperforms conventional approaches while maintaining detection efficiency, achieving an AUROC of 0.852 for distinguishing between Benign, DCIS, and Invasive classes. This work highlights the potential of MVPT-NET for medical imaging tasks and provides a scalable solution for integrating multi-view data in breast cancer detection. 

**Abstract (ZH)**: 准确检测高分辨率乳腺X线摄影中的乳腺癌对于早期诊断和有效治疗规划至关重要。先前的研究表明，使用单视角乳腺X线摄影进行乳腺癌检测具有潜力。然而，结合多视角数据可以提供更全面的洞察。在医学影像中，多视角分类尤其具有挑战性，特别是处理大规模、高分辨率数据时。在本文中，我们提出了一种新型的多视角视觉提示调优网络（MVPT-NET）用于分析多个筛查乳腺X线摄影图像。我们首先在高分辨率乳腺X线摄影数据上预训练一个稳健的单视角分类模型，然后创新地将多视角特征学习适应到特定任务的提示调优过程中。该技术仅选择性地调优少量可训练参数（7%）以保留预训练单视角模型的稳健性，从而在不需要剧烈下采样的情况下高效地整合多视角数据。我们的方法为传统的特征融合方法提供了高效的替代方案，提供了对高分辨率乳腺X线摄影分析更加稳健、可扩展和高效的解决方案。在大规模多机构数据集上的实验结果显示，我们的方法在保持检测效率的同时超过了传统方法，实现了0.852的AUROC，用于区分良性、DCIS和浸润类。本工作突显了MVPT-NET在医学影像任务中的潜力，并提供了在乳腺癌检测中整合多视角数据的可扩展解决方案。 

---
# Towards Ball Spin and Trajectory Analysis in Table Tennis Broadcast Videos via Physically Grounded Synthetic-to-Real Transfer 

**Title (ZH)**: 基于物理接地的合成到现实转移的乒乓球旋转和轨迹分析在广播视频中的研究 

**Authors**: Daniel Kienzle, Robin Schön, Rainer Lienhart, Shin'Ichi Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2504.19863)  

**Abstract**: Analyzing a player's technique in table tennis requires knowledge of the ball's 3D trajectory and spin. While, the spin is not directly observable in standard broadcasting videos, we show that it can be inferred from the ball's trajectory in the video. We present a novel method to infer the initial spin and 3D trajectory from the corresponding 2D trajectory in a video. Without ground truth labels for broadcast videos, we train a neural network solely on synthetic data. Due to the choice of our input data representation, physically correct synthetic training data, and using targeted augmentations, the network naturally generalizes to real data. Notably, these simple techniques are sufficient to achieve generalization. No real data at all is required for training. To the best of our knowledge, we are the first to present a method for spin and trajectory prediction in simple monocular broadcast videos, achieving an accuracy of 92.0% in spin classification and a 2D reprojection error of 0.19% of the image diagonal. 

**Abstract (ZH)**: 分析乒乓球选手的技术需要了解球的3D轨迹和旋转。虽然旋转在标准广播视频中不可直接观察，但我们展示了可以通过视频中的球的轨迹推断出旋转。我们提出了一种新颖的方法，通过对应的2D轨迹推断出球的初始旋转和3D轨迹。由于我们选择的数据表示方式、物理上正确的合成训练数据以及目标导向的数据增强，网络自然能够泛化到实际数据中。值得注意的是，这些简单的方法足以实现泛化。在训练过程中并不需要使用实际数据。据我们所知，这是我们首次提出在简单的单目广播视频中预测旋转和轨迹的方法，在旋转分类准确率上达到92.0%，2D再投影误差为图像对角线的0.19%。 

---
# Foundation Model-Driven Framework for Human-Object Interaction Prediction with Segmentation Mask Integration 

**Title (ZH)**: 基于分割掩码整合的foundation模型驱动的人机物体交互预测框架 

**Authors**: Juhan Park, Kyungjae Lee, Hyung Jin Chang, Jungchan Cho  

**Link**: [PDF](https://arxiv.org/pdf/2504.19847)  

**Abstract**: In this work, we introduce Segmentation to Human-Object Interaction (\textit{\textbf{Seg2HOI}}) approach, a novel framework that integrates segmentation-based vision foundation models with the human-object interaction task, distinguished from traditional detection-based Human-Object Interaction (HOI) methods. Our approach enhances HOI detection by not only predicting the standard triplets but also introducing quadruplets, which extend HOI triplets by including segmentation masks for human-object pairs. More specifically, Seg2HOI inherits the properties of the vision foundation model (e.g., promptable and interactive mechanisms) and incorporates a decoder that applies these attributes to HOI task. Despite training only for HOI, without additional training mechanisms for these properties, the framework demonstrates that such features still operate efficiently. Extensive experiments on two public benchmark datasets demonstrate that Seg2HOI achieves performance comparable to state-of-the-art methods, even in zero-shot scenarios. Lastly, we propose that Seg2HOI can generate HOI quadruplets and interactive HOI segmentation from novel text and visual prompts that were not used during training, making it versatile for a wide range of applications by leveraging this flexibility. 

**Abstract (ZH)**: Segmentation to Human-Object Interaction (Seg2HOI) 方法：一种将基于分割的视觉基础模型与人-物交互任务结合的新型框架 

---
# Image Generation Method Based on Heat Diffusion Models 

**Title (ZH)**: 基于热扩散模型的图像生成方法 

**Authors**: Pengfei Zhang, Shouqing Jia  

**Link**: [PDF](https://arxiv.org/pdf/2504.19600)  

**Abstract**: Denoising Diffusion Probabilistic Models (DDPMs) achieve high-quality image generation without adversarial training, but they process images as a whole. Since adjacent pixels are highly likely to belong to the same object, we propose the Heat Diffusion Model (HDM) to further preserve image details and generate more realistic images. HDM is a model that incorporates pixel-level operations while maintaining the same training process as DDPM. In HDM, the discrete form of the two-dimensional heat equation is integrated into the diffusion and generation formulas of DDPM, enabling the model to compute relationships between neighboring pixels during image processing. Our experiments demonstrate that HDM can generate higher-quality samples compared to models such as DDPM, Consistency Diffusion Models (CDM), Latent Diffusion Models (LDM), and Vector Quantized Generative Adversarial Networks (VQGAN). 

**Abstract (ZH)**: 热扩散模型（HDM）在保留图像细节和生成更逼真图像方面的去噪扩散概率模型（DDPMs）实现高质量图像生成而不采用对抗训练，但在处理图像时是整体进行的。由于相邻像素很可能属于同一个对象，我们提出了一种热扩散模型（HDM），以进一步保留图像细节并生成更真实的图像。HDM是一种在保持与DDPM相同训练过程的同时包含像素级操作的模型。在HDM中，二维热方程的离散形式被整合到DDPM的扩散和生成公式中，使得模型在图像处理过程中能够计算相邻像素之间的关系。我们的实验表明，HDM的样本质量高于诸如DDPM、一致性扩散模型（CDM）、潜在扩散模型（LDM）和向量量化生成对抗网络（VQGAN）的模型。 

---
# Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video 

**Title (ZH)**: Prisma: 一个开源工具包，用于视觉和视频的机制可解释性 

**Authors**: Sonia Joseph, Praneet Suresh, Lorenz Hufe, Edward Stevinson, Robert Graham, Yash Vadi, Danilo Bzdok, Sebastian Lapuschkin, Lee Sharkey, Blake Aaron Richards  

**Link**: [PDF](https://arxiv.org/pdf/2504.19475)  

**Abstract**: Robust tooling and publicly available pre-trained models have helped drive recent advances in mechanistic interpretability for language models. However, similar progress in vision mechanistic interpretability has been hindered by the lack of accessible frameworks and pre-trained weights. We present Prisma (Access the codebase here: this https URL), an open-source framework designed to accelerate vision mechanistic interpretability research, providing a unified toolkit for accessing 75+ vision and video transformers; support for sparse autoencoder (SAE), transcoder, and crosscoder training; a suite of 80+ pre-trained SAE weights; activation caching, circuit analysis tools, and visualization tools; and educational resources. Our analysis reveals surprising findings, including that effective vision SAEs can exhibit substantially lower sparsity patterns than language SAEs, and that in some instances, SAE reconstructions can decrease model loss. Prisma enables new research directions for understanding vision model internals while lowering barriers to entry in this emerging field. 

**Abstract (ZH)**: Robista工具和公开可用的预训练模型有助于推动语言模型机械可解释性的 recent 进展。然而，视觉模型机械可解释性方面的类似进展因缺乏可访问的框架和预训练权重而受阻。我们介绍了Prisma（代码库在此访问：this https URL），这是一个开源框架，旨在加速视觉模型机械可解释性研究，提供了一个统一的工具包以访问75多种视觉和视频变压器；支持稀疏自编码器（SAE）、编码器-解码器和跨编码器训练；包含80多种预训练SAE权重；激活缓存、电路分析工具和可视化工具；以及教育资源。我们的分析揭示了一些令人惊讶的发现，包括有效的视觉SAE可以表现出显著 lower 的稀疏模式，以及在某些情况下，SAE重构可以降低模型损失。Prisma 为理解视觉模型内部结构提供了新的研究方向，同时也降低了进入这一新兴领域门槛。 

---
# A Real-Time Gesture-Based Control Framework 

**Title (ZH)**: 一种基于手势的实时控制框架 

**Authors**: Mahya Khazaei, Ali Bahrani, George Tzanetakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.19460)  

**Abstract**: We introduce a real-time, human-in-the-loop gesture control framework that can dynamically adapt audio and music based on human movement by analyzing live video input. By creating a responsive connection between visual and auditory stimuli, this system enables dancers and performers to not only respond to music but also influence it through their movements. Designed for live performances, interactive installations, and personal use, it offers an immersive experience where users can shape the music in real time.
The framework integrates computer vision and machine learning techniques to track and interpret motion, allowing users to manipulate audio elements such as tempo, pitch, effects, and playback sequence. With ongoing training, it achieves user-independent functionality, requiring as few as 50 to 80 samples to label simple gestures. This framework combines gesture training, cue mapping, and audio manipulation to create a dynamic, interactive experience. Gestures are interpreted as input signals, mapped to sound control commands, and used to naturally adjust music elements, showcasing the seamless interplay between human interaction and machine response. 

**Abstract (ZH)**: 一种实时、有人参与回路的手势控制框架：基于人体运动动态适应音频和音乐 

---
# EarthMapper: Visual Autoregressive Models for Controllable Bidirectional Satellite-Map Translation 

**Title (ZH)**: EarthMapper：视觉自回归模型在可控双向卫星图地图转换中的应用 

**Authors**: Zhe Dong, Yuzhe Sun, Tianzhu Liu, Wangmeng Zuo, Yanfeng Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.19432)  

**Abstract**: Satellite imagery and maps, as two fundamental data modalities in remote sensing, offer direct observations of the Earth's surface and human-interpretable geographic abstractions, respectively. The task of bidirectional translation between satellite images and maps (BSMT) holds significant potential for applications in urban planning and disaster response. However, this task presents two major challenges: first, the absence of precise pixel-wise alignment between the two modalities substantially complicates the translation process; second, it requires achieving both high-level abstraction of geographic features and high-quality visual synthesis, which further elevates the technical complexity. To address these limitations, we introduce EarthMapper, a novel autoregressive framework for controllable bidirectional satellite-map translation. EarthMapper employs geographic coordinate embeddings to anchor generation, ensuring region-specific adaptability, and leverages multi-scale feature alignment within a geo-conditioned joint scale autoregression (GJSA) process to unify bidirectional translation in a single training cycle. A semantic infusion (SI) mechanism is introduced to enhance feature-level consistency, while a key point adaptive guidance (KPAG) mechanism is proposed to dynamically balance diversity and precision during inference. We further contribute CNSatMap, a large-scale dataset comprising 302,132 precisely aligned satellite-map pairs across 38 Chinese cities, enabling robust benchmarking. Extensive experiments on CNSatMap and the New York dataset demonstrate EarthMapper's superior performance, achieving significant improvements in visual realism, semantic consistency, and structural fidelity over state-of-the-art methods. Additionally, EarthMapper excels in zero-shot tasks like in-painting, out-painting and coordinate-conditional generation, underscoring its versatility. 

**Abstract (ZH)**: 卫星遥感图像和地图之间的双向翻译：一种基于地理坐标嵌入的自回归框架及其应用 

---
# Mitigating Bias in Facial Recognition Systems: Centroid Fairness Loss Optimization 

**Title (ZH)**: Facial识别系统中偏见的缓解：质心公平损失优化 

**Authors**: Jean-Rémy Conti, Stéphan Clémençon  

**Link**: [PDF](https://arxiv.org/pdf/2504.19370)  

**Abstract**: The urging societal demand for fair AI systems has put pressure on the research community to develop predictive models that are not only globally accurate but also meet new fairness criteria, reflecting the lack of disparate mistreatment with respect to sensitive attributes ($\textit{e.g.}$ gender, ethnicity, age). In particular, the variability of the errors made by certain Facial Recognition (FR) systems across specific segments of the population compromises the deployment of the latter, and was judged unacceptable by regulatory authorities. Designing fair FR systems is a very challenging problem, mainly due to the complex and functional nature of the performance measure used in this domain ($\textit{i.e.}$ ROC curves) and because of the huge heterogeneity of the face image datasets usually available for training. In this paper, we propose a novel post-processing approach to improve the fairness of pre-trained FR models by optimizing a regression loss which acts on centroid-based scores. Beyond the computational advantages of the method, we present numerical experiments providing strong empirical evidence of the gain in fairness and of the ability to preserve global accuracy. 

**Abstract (ZH)**: 迫切的社会需求推动公平AI系统的发展，要求研究社区开发不仅全球准确而且满足新的公平标准的预测模型，反映对敏感属性（例如性别、种族、年龄）的不对等歧视的缺乏。特别是，某些 Facial Recognition 系统在特定人口群体中的错误变异程度影响了这些系统的部署，并被监管机构认为不可接受。设计公平的面部识别系统是一个非常具有挑战性的问题，主要由于该领域使用的性能指标（例如ROC曲线）的复杂性和功能性，以及通常可用于训练的面部图像数据集的巨大异质性。在本文中，我们提出了一种新颖的后处理方法来通过优化作用于基于质心的分数的回归损失来改进预训练的面部识别模型的公平性。除了该方法的计算优势外，我们还展示了一定量化的实验，提供了增强公平性的强大实证证据，并证明了保持全局准确性的能力。 

---
# Low-Rank Adaptive Structural Priors for Generalizable Diabetic Retinopathy Grading 

**Title (ZH)**: 低秩自适应结构先验用于糖尿病视网膜病变的一般化分级 

**Authors**: Yunxuan Wang, Ray Yin, Yumei Tan, Hao Chen, Haiying Xia  

**Link**: [PDF](https://arxiv.org/pdf/2504.19362)  

**Abstract**: Diabetic retinopathy (DR), a serious ocular complication of diabetes, is one of the primary causes of vision loss among retinal vascular diseases. Deep learning methods have been extensively applied in the grading of diabetic retinopathy (DR). However, their performance declines significantly when applied to data outside the training distribution due to domain shifts. Domain generalization (DG) has emerged as a solution to this challenge. However, most existing DG methods overlook lesion-specific features, resulting in insufficient accuracy. In this paper, we propose a novel approach that enhances existing DG methods by incorporating structural priors, inspired by the observation that DR grading is heavily dependent on vessel and lesion structures. We introduce Low-rank Adaptive Structural Priors (LoASP), a plug-and-play framework designed for seamless integration with existing DG models. LoASP improves generalization by learning adaptive structural representations that are finely tuned to the complexities of DR diagnosis. Extensive experiments on eight diverse datasets validate its effectiveness in both single-source and multi-source domain scenarios. Furthermore, visualizations reveal that the learned structural priors intuitively align with the intricate architecture of the vessels and lesions, providing compelling insights into their interpretability and diagnostic relevance. 

**Abstract (ZH)**: 糖尿病视网膜病变（DR）是一种严重的糖尿病眼并发症，是视网膜血管疾病导致视力丧失的主要原因。深度学习方法在糖尿病视网膜病变（DR）分级中得到了广泛应用。然而，当应用于训练分布之外的数据时，其性能会显著下降，原因是领域偏差的存在。领域泛化（DG）方法因此应运而生，但大多数现有DG方法忽视了病变特异性特征，导致准确率不足。本文提出了一种新的方法，通过引入结构性先验来增强现有的DG方法，该方法受到DR分级高度依赖于血管和病变结构的观察启发。我们引入了低秩自适应结构性先验（LoASP），这是一种插件框架，旨在无缝集成到现有的DG模型中。LoASP通过学习适应性结构表示来提高泛化能力，这些表示能精细地适应DR诊断的复杂性。在八个多样化的数据集上进行的广泛实验验证了其在单源和多源领域场景中的有效性。此外，可视化结果表明，学习到的结构性先验直观地与血管和病变的复杂结构对齐，提供了其可解释性和诊断相关性的有力见解。 

---
# CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis 

**Title (ZH)**: CARL：无需相机的光谱图像分析表示学习 

**Authors**: Alexander Baumann, Leonardo Ayala, Silvia Seidlitz, Jan Sellner, Alexander Studier-Fischer, Berkin Özdemir, Lena Maier-Hein, Slobodan Ilic  

**Link**: [PDF](https://arxiv.org/pdf/2504.19223)  

**Abstract**: Spectral imaging offers promising applications across diverse domains, including medicine and urban scene understanding, and is already established as a critical modality in remote sensing. However, variability in channel dimensionality and captured wavelengths among spectral cameras impede the development of AI-driven methodologies, leading to camera-specific models with limited generalizability and inadequate cross-camera applicability. To address this bottleneck, we introduce $\textbf{CARL}$, a model for $\textbf{C}$amera-$\textbf{A}$gnostic $\textbf{R}$epresentation $\textbf{L}$earning across RGB, multispectral, and hyperspectral imaging modalities. To enable the conversion of a spectral image with any channel dimensionality to a camera-agnostic embedding, we introduce wavelength positional encoding and a self-attention-cross-attention mechanism to compress spectral information into learned query representations. Spectral-spatial pre-training is achieved with a novel spectral self-supervised JEPA-inspired strategy tailored to CARL. Large-scale experiments across the domains of medical imaging, autonomous driving, and satellite imaging demonstrate our model's unique robustness to spectral heterogeneity, outperforming on datasets with simulated and real-world cross-camera spectral variations. The scalability and versatility of the proposed approach position our model as a backbone for future spectral foundation models. 

**Abstract (ZH)**: 跨RGB、多光谱和高光谱成像模态的相机无感知表示学习模型CARL 

---
# PAD: Phase-Amplitude Decoupling Fusion for Multi-Modal Land Cover Classification 

**Title (ZH)**: PAD: 相位-振幅解耦融合在多模态土地覆盖分类中的应用 

**Authors**: Huiling Zheng, Xian Zhong, Bin Liu, Yi Xiao, Bihan Wen, Xiaofeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.19136)  

**Abstract**: The fusion of Synthetic Aperture Radar (SAR) and RGB imagery for land cover classification remains challenging due to modality heterogeneity and the underutilization of spectral complementarity. Existing methods often fail to decouple shared structural features from modality-specific radiometric attributes, leading to feature conflicts and information loss. To address this issue, we propose Phase-Amplitude Decoupling (PAD), a frequency-aware framework that separates phase (modality-shared) and amplitude (modality-specific) components in the Fourier domain. Specifically, PAD consists of two key components: 1) Phase Spectrum Correction (PSC), which aligns cross-modal phase features through convolution-guided scaling to enhance geometric consistency, and 2) Amplitude Spectrum Fusion (ASF), which dynamically integrates high-frequency details and low-frequency structures using frequency-adaptive multilayer perceptrons. This approach leverages SAR's sensitivity to morphological features and RGB's spectral richness. Extensive experiments on WHU-OPT-SAR and DDHR-SK datasets demonstrate state-of-the-art performance. Our work establishes a new paradigm for physics-aware multi-modal fusion in remote sensing. The code will be available at this https URL. 

**Abstract (ZH)**: 合成孔径雷达(SAR)与RGB图像融合在土地覆盖分类中的应用仍面临挑战，主要是由于模态异质性和谱相互补性的未充分利用。现有方法往往难以分离共享的结构特征与模态特定的辐射度属性，导致特征冲突和信息损失。为解决这一问题，我们提出了一种频率感知的相位-幅度分离框架Phase-Amplitude Decoupling (PAD)，该框架在傅里叶域中分离相位（模态共享）和幅度（模态特定）分量。具体而言，PAD包含两个关键组件：1）相位谱校正（PSC），通过卷积引导的缩放对跨模态相位特征进行对齐，以增强几何一致性；2）幅度谱融合（ASF），使用频率自适应多层感知机动态整合高频细节和低频结构。该方法利用了SAR对形态特征的敏感性和RGB的丰富光谱特性。在WHU-OPT-SAR和DDHR-SK数据集上的 extensive 实验展示了卓越的性能。我们的工作建立了遥感中物理感知多模态融合的新范式。代码将在此处提供：this https URL。 

---
# Generative AI for Character Animation: A Comprehensive Survey of Techniques, Applications, and Future Directions 

**Title (ZH)**: 生成式AI在角色动画中的应用：技术、应用及未来方向综述 

**Authors**: Mohammad Mahdi Abootorabi, Omid Ghahroodi, Pardis Sadat Zahraei, Hossein Behzadasl, Alireza Mirrokni, Mobina Salimipanah, Arash Rasouli, Bahar Behzadipour, Sara Azarnoush, Benyamin Maleki, Erfan Sadraiye, Kiarash Kiani Feriz, Mahdi Teymouri Nahad, Ali Moghadasi, Abolfazl Eshagh Abianeh, Nizi Nazar, Hamid R. Rabiee, Mahdieh Soleymani Baghshah, Meisam Ahmadi, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2504.19056)  

**Abstract**: Generative AI is reshaping art, gaming, and most notably animation. Recent breakthroughs in foundation and diffusion models have reduced the time and cost of producing animated content. Characters are central animation components, involving motion, emotions, gestures, and facial expressions. The pace and breadth of advances in recent months make it difficult to maintain a coherent view of the field, motivating the need for an integrative review. Unlike earlier overviews that treat avatars, gestures, or facial animation in isolation, this survey offers a single, comprehensive perspective on all the main generative AI applications for character animation. We begin by examining the state-of-the-art in facial animation, expression rendering, image synthesis, avatar creation, gesture modeling, motion synthesis, object generation, and texture synthesis. We highlight leading research, practical deployments, commonly used datasets, and emerging trends for each area. To support newcomers, we also provide a comprehensive background section that introduces foundational models and evaluation metrics, equipping readers with the knowledge needed to enter the field. We discuss open challenges and map future research directions, providing a roadmap to advance AI-driven character-animation technologies. This survey is intended as a resource for researchers and developers entering the field of generative AI animation or adjacent fields. Resources are available at: this https URL. 

**Abstract (ZH)**: 生成式AI正在重塑艺术、游戏和尤其动画领域。近期基础模型和扩散模型的突破降低了生成动画内容的时间和成本。角色是动画的核心组件，涉及运动、情感、手势和面部表情。近期几个月的发展速度和广度使得难以保持对该领域的连贯理解，因此迫切需要一个综合性的回顾。与以往单独讨论Avatar、手势或面部动画的综述不同，本调查提供了一个全面的视角，涵盖了所有主要的生成式AI在角色动画中的应用。我们首先审视了面部动画、表情渲染、图像合成、Avatar创建、手势建模、运动合成、对象生成和纹理合成的最新状态。我们强调了每个领域的领先研究、实际部署、常用数据集以及新兴趋势。为了帮助初学者，我们还提供了一个全面的基础背景部分，介绍了基础模型和评估指标，使读者能够掌握进入该领域的知识。我们讨论了开放的挑战并规划了未来的研究方向，提供了一条通向驱动角色动画技术发展的道路。本调查旨在为进入生成式AI动画或相关领域的研究人员和开发人员提供资源。更多信息请参阅：this https URL。 

---
# VISUALCENT: Visual Human Analysis using Dynamic Centroid Representation 

**Title (ZH)**: VISUALCENT：基于动态质心表示的人体视觉分析 

**Authors**: Niaz Ahmad, Youngmoon Lee, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19032)  

**Abstract**: We introduce VISUALCENT, a unified human pose and instance segmentation framework to address generalizability and scalability limitations to multi person visual human analysis. VISUALCENT leverages centroid based bottom up keypoint detection paradigm and uses Keypoint Heatmap incorporating Disk Representation and KeyCentroid to identify the optimal keypoint coordinates. For the unified segmentation task, an explicit keypoint is defined as a dynamic centroid called MaskCentroid to swiftly cluster pixels to specific human instance during rapid changes in human body movement or significantly occluded environment. Experimental results on COCO and OCHuman datasets demonstrate VISUALCENTs accuracy and real time performance advantages, outperforming existing methods in mAP scores and execution frame rate per second. The implementation is available on the project page. 

**Abstract (ZH)**: VISUALCENT：一种统一的人体姿态和实例分割框架，以应对多人视觉人体分析的一般化和可扩展性限制 

---
# Improving Pretrained YAMNet for Enhanced Speech Command Detection via Transfer Learning 

**Title (ZH)**: 通过迁移学习改进预训练YAMNet以增强语音命令检测 

**Authors**: Sidahmed Lachenani, Hamza Kheddar, Mohamed Ouldzmirli  

**Link**: [PDF](https://arxiv.org/pdf/2504.19030)  

**Abstract**: This work addresses the need for enhanced accuracy and efficiency in speech command recognition systems, a critical component for improving user interaction in various smart applications. Leveraging the robust pretrained YAMNet model and transfer learning, this study develops a method that significantly improves speech command recognition. We adapt and train a YAMNet deep learning model to effectively detect and interpret speech commands from audio signals. Using the extensively annotated Speech Commands dataset (speech_commands_v0.01), our approach demonstrates the practical application of transfer learning to accurately recognize a predefined set of speech commands. The dataset is meticulously augmented, and features are strategically extracted to boost model performance. As a result, the final model achieved a recognition accuracy of 95.28%, underscoring the impact of advanced machine learning techniques on speech command recognition. This achievement marks substantial progress in audio processing technologies and establishes a new benchmark for future research in the field. 

**Abstract (ZH)**: 本研究旨在通过增强语音命令识别系统的准确性和效率来改善各种智能应用中的用户交互。借助稳健的预训练YAMNet模型和迁移学习，本研究开发了一种显著提高语音命令识别的方法。我们适应并训练了YAMNet深度学习模型，以有效检测和解释来自音频信号的语音命令。利用广泛标注的Speech Commands数据集（speech_commands_v0.01），本方法展示了迁移学习在准确识别预定义语音命令方面的实用应用。数据集经过精心扩充，特征被战略性地提取以提升模型性能。最终，模型的识别准确率达到95.28%，突显了先进机器学习技术对语音命令识别的影响。这一成就标志着音频处理技术的显著进步，并为未来的研究设立了新的标杆。 

---
# Surgeons vs. Computer Vision: A comparative analysis on surgical phase recognition capabilities 

**Title (ZH)**: 外科医生 vs. 计算机视觉：手术阶段识别能力的 comparative analysis 比较分析 

**Authors**: Marco Mezzina, Pieter De Backer, Tom Vercauteren, Matthew Blaschko, Alexandre Mottrie, Tinne Tuytelaars  

**Link**: [PDF](https://arxiv.org/pdf/2504.18954)  

**Abstract**: Purpose: Automated Surgical Phase Recognition (SPR) uses Artificial Intelligence (AI) to segment the surgical workflow into its key events, functioning as a building block for efficient video review, surgical education as well as skill assessment. Previous research has focused on short and linear surgical procedures and has not explored if temporal context influences experts' ability to better classify surgical phases. This research addresses these gaps, focusing on Robot-Assisted Partial Nephrectomy (RAPN) as a highly non-linear procedure. Methods: Urologists of varying expertise were grouped and tasked to indicate the surgical phase for RAPN on both single frames and video snippets using a custom-made web platform. Participants reported their confidence levels and the visual landmarks used in their decision-making. AI architectures without and with temporal context as trained and benchmarked on the Cholec80 dataset were subsequently trained on this RAPN dataset. Results: Video snippets and presence of specific visual landmarks improved phase classification accuracy across all groups. Surgeons displayed high confidence in their classifications and outperformed novices, who struggled discriminating phases. The performance of the AI models is comparable to the surgeons in the survey, with improvements when temporal context was incorporated in both cases. Conclusion: SPR is an inherently complex task for expert surgeons and computer vision, where both perform equally well when given the same context. Performance increases when temporal information is provided. Surgical tools and organs form the key landmarks for human interpretation and are expected to shape the future of automated SPR. 

**Abstract (ZH)**: 目的：自动化手术阶段识别（SPR）利用人工智能（AI）将手术工作流程分割为关键事件，作为高效视频审查、手术教育及技能评估的基础模块。以往研究主要关注短期和线性手术程序，未探索时间上下文对专家更好地分类手术阶段能力的影响。本研究填补了这些空白，重点关注高度非线性的机器人辅助部分肾切除术（RAPN）。方法：不同水平的泌尿科专家使用自定义网络平台，对RAPN进行单帧和视频片段的手术阶段标注，并报告他们的置信水平和决策依据。使用Cholec80数据集训练和基准测试的具有和不具有时间上下文的AI架构随后在该RAPN数据集上进行训练。结果：视频片段和特定视觉标志物的存在在所有组别中均提高了阶段分类准确性。外科医生在分类中表现出高水平的信心，并优于在区分阶段方面遇到困难的初级医生。在包含时间上下文的情况下，两种AI模型的性能与调查中的外科医生相当，且有所提升。结论：SPR是一项对外科医生和计算机视觉都极具挑战性的任务，在提供相同上下文的情况下，两者表现相当，提供时间信息时性能提升。手术工具和器官构成了人类解释的关键地标，预计将成为自动化SPR的未来方向。 

---
# Exploiting Multiple Representations: 3D Face Biometrics Fusion with Application to Surveillance 

**Title (ZH)**: 利用多重表示：3D 面部生物特征融合及其在监控中的应用 

**Authors**: Simone Maurizio La Cava, Roberto Casula, Sara Concas, Giulia Orrù, Ruben Tolosana, Martin Drahansky, Julian Fierrez, Gian Luca Marcialis  

**Link**: [PDF](https://arxiv.org/pdf/2504.18886)  

**Abstract**: 3D face reconstruction (3DFR) algorithms are based on specific assumptions tailored to the limits and characteristics of the different application scenarios. In this study, we investigate how multiple state-of-the-art 3DFR algorithms can be used to generate a better representation of subjects, with the final goal of improving the performance of face recognition systems in challenging uncontrolled scenarios. We also explore how different parametric and non-parametric score-level fusion methods can exploit the unique strengths of multiple 3DFR algorithms to enhance biometric recognition robustness. With this goal, we propose a comprehensive analysis of several face recognition systems across diverse conditions, such as varying distances and camera setups, intra-dataset and cross-dataset, to assess the robustness of the proposed ensemble method. The results demonstrate that the distinct information provided by different 3DFR algorithms can alleviate the problem of generalizing over multiple application scenarios. In addition, the present study highlights the potential of advanced fusion strategies to enhance the reliability of 3DFR-based face recognition systems, providing the research community with key insights to exploit them in real-world applications effectively. Although the experiments are carried out in a specific face verification setup, our proposed fusion-based 3DFR methods may be applied to other tasks around face biometrics that are not strictly related to identity recognition. 

**Abstract (ZH)**: 基于多3DFR算法的综合分析以改善面部识别系统在复杂非控制场景中的性能 

---
# Imitation Learning for Autonomous Driving: Insights from Real-World Testing 

**Title (ZH)**: 自主驾驶中的 imitation 学习：来自实际测试的见解 

**Authors**: Hidayet Ersin Dursun, Yusuf Güven, Tufan Kumbasar  

**Link**: [PDF](https://arxiv.org/pdf/2504.18847)  

**Abstract**: This work focuses on the design of a deep learning-based autonomous driving system deployed and tested on the real-world MIT Racecar to assess its effectiveness in driving scenarios. The Deep Neural Network (DNN) translates raw image inputs into real-time steering commands in an end-to-end learning fashion, following the imitation learning framework. The key design challenge is to ensure that DNN predictions are accurate and fast enough, at a high sampling frequency, and result in smooth vehicle operation under different operating conditions. In this study, we design and compare various DNNs, to identify the most effective approach for real-time autonomous driving. In designing the DNNs, we adopted an incremental design approach that involved enhancing the model capacity and dataset to address the challenges of real-world driving scenarios. We designed a PD system, CNN, CNN-LSTM, and CNN-NODE, and evaluated their performance on the real-world MIT Racecar. While the PD system handled basic lane following, it struggled with sharp turns and lighting variations. The CNN improved steering but lacked temporal awareness, which the CNN-LSTM addressed as it resulted in smooth driving performance. The CNN-NODE performed similarly to the CNN-LSTM in handling driving dynamics, yet with slightly better driving performance. The findings of this research highlight the importance of iterative design processes in developing robust DNNs for autonomous driving applications. The experimental video is available at this https URL. 

**Abstract (ZH)**: 基于深度学习的自动驾驶系统设计与在MIT Racing车上实测及评估 

---
# Video CLIP Model for Multi-View Echocardiography Interpretation 

**Title (ZH)**: 多视角超声心动图解释的Video CLIP模型 

**Authors**: Ryo Takizawa, Satoshi Kodera, Tempei Kabayama, Ryo Matsuoka, Yuta Ando, Yuto Nakamura, Haruki Settai, Norihiko Takeda  

**Link**: [PDF](https://arxiv.org/pdf/2504.18800)  

**Abstract**: Echocardiography involves recording videos of the heart using ultrasound, enabling clinicians to evaluate its condition. Recent advances in large-scale vision-language models (VLMs) have garnered attention for automating the interpretation of echocardiographic videos. However, most existing VLMs proposed for medical interpretation thus far rely on single-frame (i.e., image) inputs. Consequently, these image-based models often exhibit lower diagnostic accuracy for conditions identifiable through cardiac motion. Moreover, echocardiographic videos are recorded from various views that depend on the direction of ultrasound emission, and certain views are more suitable than others for interpreting specific conditions. Incorporating multiple views could potentially yield further improvements in accuracy. In this study, we developed a video-language model that takes five different views and full video sequences as input, training it on pairs of echocardiographic videos and clinical reports from 60,747 cases. Our experiments demonstrate that this expanded approach achieves higher interpretation accuracy than models trained with only single-view videos or with still images. 

**Abstract (ZH)**: 基于多视角和完整视频序列的心脏超声视频语言模型研究 

---
# IoT Botnet Detection: Application of Vision Transformer to Classification of Network Flow Traffic 

**Title (ZH)**: 基于视觉变换器的物联网僵尸网络检测：网络流量分类应用 

**Authors**: Hassan Wasswa, Timothy Lynar, Aziida Nanyonga, Hussein Abbass  

**Link**: [PDF](https://arxiv.org/pdf/2504.18781)  

**Abstract**: Despite the demonstrated effectiveness of transformer models in NLP, and image and video classification, the available tools for extracting features from captured IoT network flow packets fail to capture sequential patterns in addition to the absence of spatial patterns consequently limiting transformer model application. This work introduces a novel preprocessing method to adapt transformer models, the vision transformer (ViT) in particular, for IoT botnet attack detection using network flow packets. The approach involves feature extraction from .pcap files and transforming each instance into a 1-channel 2D image shape, enabling ViT-based classification. Also, the ViT model was enhanced to allow use any classifier besides Multilayer Perceptron (MLP) that was deployed in the initial ViT paper. Models including the conventional feed forward Deep Neural Network (DNN), LSTM and Bidirectional-LSTM (BLSTM) demonstrated competitive performance in terms of precision, recall, and F1-score for multiclass-based attack detection when evaluated on two IoT attack datasets. 

**Abstract (ZH)**: 尽管变压器模型在自然语言处理、图像和视频分类中已被证明有效，但现有用于从捕获的物联网网络流包中提取特征的工具无法捕捉序列模式，且缺乏空间模式，这限制了变压器模型的应用。本研究介绍了一种新的预处理方法，以适应变压器模型，特别是视觉变压器（ViT），用于使用网络流包检测物联网僵尸网络攻击。该方法包括从.pcap文件中提取特征，并将每个实例转换为1通道2D图像形状，以使基于ViT的分类成为可能。此外，ViT模型得到了增强，允许使用除了初始ViT论文中部署的多层感知机（MLP）之外的任何分类器。包括传统的前馈深度神经网络（DNN）、LSTM和双向LSTM（BLSTM）的模型在两个物联网攻击数据集上评估时，在多类攻击检测方面展现出了竞争性的精确度、召回率和F1分数。 

---
# PyViT-FUSE: A Foundation Model for Multi-Sensor Earth Observation Data 

**Title (ZH)**: PyViT-FUSE：多传感器地球观测数据的基础模型 

**Authors**: Manuel Weber, Carly Beneke  

**Link**: [PDF](https://arxiv.org/pdf/2504.18770)  

**Abstract**: We propose PyViT-FUSE, a foundation model for earth observation data explicitly designed to handle multi-modal imagery by learning to fuse an arbitrary number of mixed-resolution input bands into a single representation through an attention mechanism. The learned patch tokens are further processed by a stack of vision transformers with a novel pyramidal structure. We train the model on a globally sampled dataset in a self-supervised manner, leveraging core concepts of the SwAV algorithm. We show the interpretability of the fusion mechanism by visualization of the attention scores and the models applicability to downstream tasks. 

**Abstract (ZH)**: 我们提出PyViT-FUSE，一种专门为处理多模态图像的地观测数据设计的基础模型，通过注意机制学习将任意数量的混合分辨率输入波段融合为单一表示。学习得到的 patch tokens 进一步通过具有新颖分层结构的视觉变换器堆栈进行处理。我们采用全局采样的数据集以自监督方式训练该模型，利用SwAV算法的核心概念。通过可视化注意力分数展示了融合机制的可解释性，并展示了该模型对下游任务的应用性。 

---
# HierSum: A Global and Local Attention Mechanism for Video Summarization 

**Title (ZH)**: HierSum：视频摘要的全局和局部注意力机制 

**Authors**: Apoorva Beedu, Irfan Essa  

**Link**: [PDF](https://arxiv.org/pdf/2504.18689)  

**Abstract**: Video summarization creates an abridged version (i.e., a summary) that provides a quick overview of the video while retaining pertinent information. In this work, we focus on summarizing instructional videos and propose a method for breaking down a video into meaningful segments, each corresponding to essential steps in the video. We propose \textbf{HierSum}, a hierarchical approach that integrates fine-grained local cues from subtitles with global contextual information provided by video-level instructions. Our approach utilizes the ``most replayed" statistic as a supervisory signal to identify critical segments, thereby improving the effectiveness of the summary. We evaluate on benchmark datasets such as TVSum, BLiSS, this http URL, and the WikiHow test set, and show that HierSum consistently outperforms existing methods in key metrics such as F1-score and rank correlation. We also curate a new multi-modal dataset using WikiHow and EHow videos and associated articles containing step-by-step instructions. Through extensive ablation studies, we demonstrate that training on this dataset significantly enhances summarization on the target datasets. 

**Abstract (ZH)**: 视频总结创建一个简要版本（即摘要），以提供视频的快速概览并保留相关信息。在本文中，我们专注于总结教学视频，并提出了一种将视频分解为有意义段落的方法，每个段落对应视频中的关键步骤。我们提出了一种名为HierSum的分层方法，该方法将字幕中的细粒度局部线索与视频级别指令提供的全局上下文信息结合起来。我们的方法利用“最常回放”统计数据作为监督信号来识别关键段落，从而提高摘要的有效性。我们在TVSum、BLiSS、维基何及E何测试集等基准数据集上进行评估，并展示了HierSum在F1分数和相关性排名等关键指标上始终优于现有方法。我们还使用维基何和E何视频及包含逐步指南的关联文章创建了一个新的多模态数据集。通过广泛的消融研究，我们证明了在此数据集上进行训练显著提高了目标数据集上的总结效果。 

---
