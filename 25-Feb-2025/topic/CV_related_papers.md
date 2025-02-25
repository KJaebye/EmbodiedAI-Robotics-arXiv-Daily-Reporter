# Tidiness Score-Guided Monte Carlo Tree Search for Visual Tabletop Rearrangement 

**Title (ZH)**: 基于整洁度评分的蒙特卡洛树搜索方法用于视觉桌面整理 

**Authors**: Hogun Kee, Wooseok Oh, Minjae Kang, Hyemin Ahn, Songhwai Oh  

**Link**: [PDF](https://arxiv.org/pdf/2502.17235)  

**Abstract**: In this paper, we present the tidiness score-guided Monte Carlo tree search (TSMCTS), a novel framework designed to address the tabletop tidying up problem using only an RGB-D camera. We address two major problems for tabletop tidying up problem: (1) the lack of public datasets and benchmarks, and (2) the difficulty of specifying the goal configuration of unseen objects. We address the former by presenting the tabletop tidying up (TTU) dataset, a structured dataset collected in simulation. Using this dataset, we train a vision-based discriminator capable of predicting the tidiness score. This discriminator can consistently evaluate the degree of tidiness across unseen configurations, including real-world scenes. Addressing the second problem, we employ Monte Carlo tree search (MCTS) to find tidying trajectories without specifying explicit goals. Instead of providing specific goals, we demonstrate that our MCTS-based planner can find diverse tidied configurations using the tidiness score as a guidance. Consequently, we propose TSMCTS, which integrates a tidiness discriminator with an MCTS-based tidying planner to find optimal tidied arrangements. TSMCTS has successfully demonstrated its capability across various environments, including coffee tables, dining tables, office desks, and bathrooms. The TTU dataset is available at: this https URL. 

**Abstract (ZH)**: 基于整洁度评分引导的蒙特卡洛树搜索（TSMCTS）：利用RGB-D相机解决桌面整理问题的新框架 

---
# Task-Oriented 6-DoF Grasp Pose Detection in Clutters 

**Title (ZH)**: 面向任务的6自由度抓取姿态检测在杂乱环境中的应用 

**Authors**: An-Lan Wang, Nuo Chen, Kun-Yu Lin, Li Yuan-Ming, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.16976)  

**Abstract**: In general, humans would grasp an object differently for different tasks, e.g., "grasping the handle of a knife to cut" vs. "grasping the blade to hand over". In the field of robotic grasp pose detection research, some existing works consider this task-oriented grasping and made some progress, but they are generally constrained by low-DoF gripper type or non-cluttered setting, which is not applicable for human assistance in real life. With an aim to get more general and practical grasp models, in this paper, we investigate the problem named Task-Oriented 6-DoF Grasp Pose Detection in Clutters (TO6DGC), which extends the task-oriented problem to a more general 6-DOF Grasp Pose Detection in Cluttered (multi-object) scenario. To this end, we construct a large-scale 6-DoF task-oriented grasping dataset, 6-DoF Task Grasp (6DTG), which features 4391 cluttered scenes with over 2 million 6-DoF grasp poses. Each grasp is annotated with a specific task, involving 6 tasks and 198 objects in total. Moreover, we propose One-Stage TaskGrasp (OSTG), a strong baseline to address the TO6DGC problem. Our OSTG adopts a task-oriented point selection strategy to detect where to grasp, and a task-oriented grasp generation module to decide how to grasp given a specific task. To evaluate the effectiveness of OSTG, extensive experiments are conducted on 6DTG. The results show that our method outperforms various baselines on multiple metrics. Real robot experiments also verify that our OSTG has a better perception of the task-oriented grasp points and 6-DoF grasp poses. 

**Abstract (ZH)**: 面向任务的堆叠物中6-自由度抓取姿态检测（TO6DGC） 

---
# DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning 

**Title (ZH)**: DemoGen: 数据高效的视觉运动策略学习合成示范生成 

**Authors**: Zhengrong Xue, Shuying Deng, Zhenyang Chen, Yixuan Wang, Zhecheng Yuan, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16932)  

**Abstract**: Visuomotor policies have shown great promise in robotic manipulation but often require substantial amounts of human-collected data for effective performance. A key reason underlying the data demands is their limited spatial generalization capability, which necessitates extensive data collection across different object configurations. In this work, we present DemoGen, a low-cost, fully synthetic approach for automatic demonstration generation. Using only one human-collected demonstration per task, DemoGen generates spatially augmented demonstrations by adapting the demonstrated action trajectory to novel object configurations. Visual observations are synthesized by leveraging 3D point clouds as the modality and rearranging the subjects in the scene via 3D editing. Empirically, DemoGen significantly enhances policy performance across a diverse range of real-world manipulation tasks, showing its applicability even in challenging scenarios involving deformable objects, dexterous hand end-effectors, and bimanual platforms. Furthermore, DemoGen can be extended to enable additional out-of-distribution capabilities, including disturbance resistance and obstacle avoidance. 

**Abstract (ZH)**: 基于视觉运动策略的自动演示生成：一种低成本的全合成方法 

---
# Improving Monocular Visual-Inertial Initialization with Structureless Visual-Inertial Bundle Adjustment 

**Title (ZH)**: 单目视觉-惯性初始化的改进：无结构视觉-惯性束调整 

**Authors**: Junlin Song, Antoine Richard, Miguel Olivares-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2502.16598)  

**Abstract**: Monocular visual inertial odometry (VIO) has facilitated a wide range of real-time motion tracking applications, thanks to the small size of the sensor suite and low power consumption. To successfully bootstrap VIO algorithms, the initialization module is extremely important. Most initialization methods rely on the reconstruction of 3D visual point clouds. These methods suffer from high computational cost as state vector contains both motion states and 3D feature points. To address this issue, some researchers recently proposed a structureless initialization method, which can solve the initial state without recovering 3D structure. However, this method potentially compromises performance due to the decoupled estimation of rotation and translation, as well as linear constraints. To improve its accuracy, we propose novel structureless visual-inertial bundle adjustment to further refine previous structureless solution. Extensive experiments on real-world datasets show our method significantly improves the VIO initialization accuracy, while maintaining real-time performance. 

**Abstract (ZH)**: 单目视觉惯性里程计（VIO）由于传感器套件小巧和低功耗，已广泛推动各类实时运动跟踪应用。为了成功初始化VIO算法，初始化模块非常重要。大多数初始化方法依赖于构建3D视觉点云。这些方法由于状态向量同时包含运动状态和3D特征点，计算成本较高。为解决此问题，一些研究人员最近提出了一种无结构化初始化方法，该方法可以在不解构3D结构的情况下解决初始状态。然而，该方法由于旋转和平移的解耦估计以及线性约束，其性能可能受到影响。为了提高其准确性，我们提出了一种新型的无结构化视觉惯性Bundle调整，以进一步细化之前的无结构化解决方案。在实际数据集上的广泛实验表明，我们的方法在保持实时性能的同时显著提高了VIO初始化的准确性。 

---
# OpenVox: Real-time Instance-level Open-vocabulary Probabilistic Voxel Representation 

**Title (ZH)**: OpenVox：实时实例级开放词汇概率体素表示 

**Authors**: Yinan Deng, Bicheng Yao, Yihang Tang, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.16528)  

**Abstract**: In recent years, vision-language models (VLMs) have advanced open-vocabulary mapping, enabling mobile robots to simultaneously achieve environmental reconstruction and high-level semantic understanding. While integrated object cognition helps mitigate semantic ambiguity in point-wise feature maps, efficiently obtaining rich semantic understanding and robust incremental reconstruction at the instance-level remains challenging. To address these challenges, we introduce OpenVox, a real-time incremental open-vocabulary probabilistic instance voxel representation. In the front-end, we design an efficient instance segmentation and comprehension pipeline that enhances language reasoning through encoding captions. In the back-end, we implement probabilistic instance voxels and formulate the cross-frame incremental fusion process into two subtasks: instance association and live map evolution, ensuring robustness to sensor and segmentation noise. Extensive evaluations across multiple datasets demonstrate that OpenVox achieves state-of-the-art performance in zero-shot instance segmentation, semantic segmentation, and open-vocabulary retrieval. Furthermore, real-world robotics experiments validate OpenVox's capability for stable, real-time operation. 

**Abstract (ZH)**: 近年来，视觉-语言模型（VLMs）增强了开放词汇映射能力，使移动机器人能够同时实现环境重建和高层次语义理解。虽然集成对象认知有助于缓解点特征图中的语义模糊性，但在实例级别高效获得丰富语义理解并实现稳健的增量重建仍然具有挑战性。为应对这些挑战，我们介绍了OpenVox，一种实时增量开放词汇概率实例体素表示方法。在前端，我们设计了一个高效的实例分割和理解管道，通过编码标题来增强语言推理。在后端，我们实现了概率实例体素，并将跨帧增量融合过程分解为两个子任务：实例关联和实时地图演化，从而确保对传感器和分割噪声的鲁棒性。在多个数据集上进行的广泛评估表明，OpenVox 在零样本实例分割、语义分割和开放词汇检索方面达到了最佳性能。此外，实际机器人实验验证了OpenVox 能够实现稳定、实时的操作。 

---
# Gaussian Process Regression for Improved Underwater Navigation 

**Title (ZH)**: 高斯过程回归改进水下导航 

**Authors**: Nadav Cohen, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2502.16510)  

**Abstract**: Accurate underwater navigation is a challenging task due to the absence of global navigation satellite system signals and the reliance on inertial navigation systems that suffer from drift over time. Doppler velocity logs (DVLs) are typically used to mitigate this drift through velocity measurements, which are commonly estimated using a parameter estimation approach such as least squares (LS). However, LS works under the assumption of ideal conditions and does not account for sensor biases, leading to suboptimal performance. This paper proposes a data-driven alternative based on multi-output Gaussian process regression (MOGPR) to improve DVL velocity estimation. MOGPR provides velocity estimates and associated measurement covariances, enabling an adaptive integration within an error-state Extended Kalman Filter (EKF). We evaluate our proposed approach using real-world AUV data and compare it against LS and a state-of-the-art deep learning model, BeamsNet. Results demonstrate that MOGPR reduces velocity estimation errors by approximately 20% while simultaneously enhancing overall navigation accuracy, particularly in the orientation states. Additionally, the incorporation of uncertainty estimates from MOGPR enables an adaptive EKF framework, improving navigation robustness in dynamic underwater environments. 

**Abstract (ZH)**: 基于多输出高斯过程回归的 Doppler 速度日志速度估计方法 

---
# Supermarket-6DoF: A Real-World Grasping Dataset and Grasp Pose Representation Analysis 

**Title (ZH)**: Supermarket-6DoF: 一个真实世界抓取数据集及 grasping 姿态表示分析 

**Authors**: Jason Toskov, Akansel Cosgun  

**Link**: [PDF](https://arxiv.org/pdf/2502.16311)  

**Abstract**: We present Supermarket-6DoF, a real-world dataset of 1500 grasp attempts across 20 supermarket objects with publicly available 3D models. Unlike most existing grasping datasets that rely on analytical metrics or simulation for grasp labeling, our dataset provides ground-truth outcomes from physical robot executions. Among the few real-world grasping datasets, wile more modest in size, Supermarket-6DoF uniquely features full 6-DoF grasp poses annotated with both initial grasp success and post-grasp stability under external perturbation. We demonstrate the dataset's utility by analyzing three grasp pose representations for grasp success prediction from point clouds. Our results show that representing the gripper geometry explicitly as a point cloud achieves higher prediction accuracy compared to conventional quaternion-based grasp pose encoding. 

**Abstract (ZH)**: 超市-6自由度：一个包含20种超市物体1500次抓取尝试的真实世界数据集，附带公开的3D模型和物理机器人执行的真实抓取结果 

---
# DeProPose: Deficiency-Proof 3D Human Pose Estimation via Adaptive Multi-View Fusion 

**Title (ZH)**: DeProPose: 防缺陷的适应性多视图融合三维人体姿态估计 

**Authors**: Jianbin Jiao, Xina Cheng, Kailun Yang, Xiangrong Zhang, Licheng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.16419)  

**Abstract**: 3D human pose estimation has wide applications in fields such as intelligent surveillance, motion capture, and virtual reality. However, in real-world scenarios, issues such as occlusion, noise interference, and missing viewpoints can severely affect pose estimation. To address these challenges, we introduce the task of Deficiency-Aware 3D Pose Estimation. Traditional 3D pose estimation methods often rely on multi-stage networks and modular combinations, which can lead to cumulative errors and increased training complexity, making them unable to effectively address deficiency-aware estimation. To this end, we propose DeProPose, a flexible method that simplifies the network architecture to reduce training complexity and avoid information loss in multi-stage designs. Additionally, the model innovatively introduces a multi-view feature fusion mechanism based on relative projection error, which effectively utilizes information from multiple viewpoints and dynamically assigns weights, enabling efficient integration and enhanced robustness to overcome deficiency-aware 3D Pose Estimation challenges. Furthermore, to thoroughly evaluate this end-to-end multi-view 3D human pose estimation model and to advance research on occlusion-related challenges, we have developed a novel 3D human pose estimation dataset, termed the Deficiency-Aware 3D Pose Estimation (DA-3DPE) dataset. This dataset encompasses a wide range of deficiency scenarios, including noise interference, missing viewpoints, and occlusion challenges. Compared to state-of-the-art methods, DeProPose not only excels in addressing the deficiency-aware problem but also shows improvement in conventional scenarios, providing a powerful and user-friendly solution for 3D human pose estimation. The source code will be available at this https URL. 

**Abstract (ZH)**: 三维人体姿态估计在智能监控、动作捕捉和虚拟现实等领域具有广泛应用。然而，在现实场景中，遮挡、噪声干扰和视角缺失等问题严重影响姿态估计。为应对这些挑战，我们提出了缺陷感知三维姿态估计任务。传统的三维姿态估计方法往往依赖于多阶段网络和模块组合，这可能导致累积误差和训练复杂度增加，限制了其对缺陷感知估计的有效解决能力。为此，我们提出了一种灵活的方法——DeProPose，简化网络架构以降低训练复杂度并避免多阶段设计中的信息损失。此外，模型创新性地引入了基于相对投影误差的多视角特征融合机制，有效利用多视角信息并动态分配权重，实现高效集成和增强的鲁棒性，以克服缺陷感知三维姿态估计挑战。为进一步评估这一端到端的多视角三维人体姿态估计模型并推动遮挡相关挑战的研究，我们开发了一个新的三维人体姿态估计数据集，称为缺陷感知三维姿态估计（DA-3DPE）数据集。该数据集涵盖了广泛的缺陷场景，包括噪声干扰、视角缺失和遮挡挑战。与现有最先进的方法相比，DeProPose 不仅在解决缺陷感知问题方面表现出色，还在常规场景中也有所改进，提供了一种强大且用户友好的三维人体姿态估计解决方案。源代码将在此链接处提供：https://github.com/alibaba/Qwen-DeProPose。 

---
# RELICT: A Replica Detection Framework for Medical Image Generation 

**Title (ZH)**: RE.fillText: 医学图像生成中的副本检测框架 

**Authors**: Orhun Utku Aydin, Alexander Koch, Adam Hilbert, Jana Rieger, Felix Lohrke, Fujimaro Ishida, Satoru Tanioka, Dietmar Frey  

**Link**: [PDF](https://arxiv.org/pdf/2502.17360)  

**Abstract**: Despite the potential of synthetic medical data for augmenting and improving the generalizability of deep learning models, memorization in generative models can lead to unintended leakage of sensitive patient information and limit model utility. Thus, the use of memorizing generative models in the medical domain can jeopardize patient privacy. We propose a framework for identifying replicas, i.e. nearly identical copies of the training data, in synthetic medical image datasets. Our REpLIca deteCTion (RELICT) framework for medical image generative models evaluates image similarity using three complementary approaches: (1) voxel-level analysis, (2) feature-level analysis by a pretrained medical foundation model, and (3) segmentation-level analysis. Two clinically relevant 3D generative modelling use cases were investigated: non-contrast head CT with intracerebral hemorrhage (N=774) and time-of-flight MR angiography of the Circle of Willis (N=1,782). Expert visual scoring was used as the reference standard to assess the presence of replicas. We report the balanced accuracy at the optimal threshold to assess replica classification performance. The reference visual rating identified 45 of 50 and 5 of 50 generated images as replicas for the NCCT and TOF-MRA use cases, respectively. Image-level and feature-level measures perfectly classified replicas with a balanced accuracy of 1 when an optimal threshold was selected for the NCCT use case. A perfect classification of replicas for the TOF-MRA case was not possible at any threshold, with the segmentation-level analysis achieving a balanced accuracy of 0.79. Replica detection is a crucial but neglected validation step for the development of generative models in medical imaging. The proposed RELICT framework provides a standardized, easy-to-use tool for replica detection and aims to facilitate responsible and ethical medical image synthesis. 

**Abstract (ZH)**: 一种用于医疗图像生成模型的重复检测框架：REpLIca deteCTion (RELICT) 

---
# AnyTop: Character Animation Diffusion with Any Topology 

**Title (ZH)**: AnyTop: 具有任意拓扑的动画扩散角色动画生成 

**Authors**: Inbar Gat, Sigal Raab, Guy Tevet, Yuval Reshef, Amit H. Bermano, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2502.17327)  

**Abstract**: Generating motion for arbitrary skeletons is a longstanding challenge in computer graphics, remaining largely unexplored due to the scarcity of diverse datasets and the irregular nature of the data. In this work, we introduce AnyTop, a diffusion model that generates motions for diverse characters with distinct motion dynamics, using only their skeletal structure as input. Our work features a transformer-based denoising network, tailored for arbitrary skeleton learning, integrating topology information into the traditional attention mechanism. Additionally, by incorporating textual joint descriptions into the latent feature representation, AnyTop learns semantic correspondences between joints across diverse skeletons. Our evaluation demonstrates that AnyTop generalizes well, even with as few as three training examples per topology, and can produce motions for unseen skeletons as well. Furthermore, our model's latent space is highly informative, enabling downstream tasks such as joint correspondence, temporal segmentation and motion editing. Our webpage, this https URL, includes links to videos and code. 

**Abstract (ZH)**: 任意骨架的动捕生成是一个长期存在的计算机图形学挑战，由于缺乏多样化的数据集和数据的不规则性，该领域尚未得到充分探索。在本文中，我们提出了一种名为AnyTop的扩散模型，该模型仅通过输入骨架结构即可生成具有不同运动特性的多样角色的动捕。我们的工作采用基于变压器的去噪网络，该网络专门用于任意骨架的学习，并将拓扑信息整合到传统的注意力机制中。通过将文本关节描述融入到潜在特征表示中，AnyTop能够学习跨不同骨架的关节语义对应关系。我们的评估表明，即使是每种拓扑结构只有三个训练示例，AnyTop也能很好地泛化，并能够生成未见过的骨架的动捕。此外，我们的模型的潜在空间具有高度信息性，能够支持关节对应、时间分割和动捕编辑等下游任务。我们的网页包括视频和代码链接：这个https URL。 

---
# Disentangling Visual Transformers: Patch-level Interpretability for Image Classification 

**Title (ZH)**: 视觉变换器解构：图像分类的 patch 级别可解释性 

**Authors**: Guillaume Jeanneret, Loïc Simon, Frédéric Jurie  

**Link**: [PDF](https://arxiv.org/pdf/2502.17196)  

**Abstract**: Visual transformers have achieved remarkable performance in image classification tasks, but this performance gain has come at the cost of interpretability. One of the main obstacles to the interpretation of transformers is the self-attention mechanism, which mixes visual information across the whole image in a complex way. In this paper, we propose Hindered Transformer (HiT), a novel interpretable by design architecture inspired by visual transformers. Our proposed architecture rethinks the design of transformers to better disentangle patch influences at the classification stage. Ultimately, HiT can be interpreted as a linear combination of patch-level information. We show that the advantages of our approach in terms of explicability come with a reasonable trade-off in performance, making it an attractive alternative for applications where interpretability is paramount. 

**Abstract (ZH)**: 视觉变换器在图像分类任务中取得了 remarkable 的性能，但这一性能的提升是以牺牲可解释性为代价的。变换器的可解释性障碍之一在于其复杂的自注意力机制，该机制以复杂的方式在整个图像中混合视觉信息。本文提出了一种名为 Hindered Transformer (HiT) 的新型可设计可解释架构，该架构受到视觉变换器的启发。我们提出的架构重新思考了变换器的设计，在分类阶段更好地解耦patch的影响。最终，HiT 可以被解释为 patch 级信息的线性组合。我们展示了我们的方法在可解释性方面的优点伴随着性能上的合理权衡，使其成为注重可解释性的应用中一个有吸引力的替代方案。 

---
# MaxGlaViT: A novel lightweight vision transformer-based approach for early diagnosis of glaucoma stages from fundus images 

**Title (ZH)**: MaxGlaViT: 一种基于轻量级视觉变换器的早期青光眼阶段诊断方法基于视网膜影像 

**Authors**: Mustafa Yurdakul, Kubra Uyar, Sakir Tasdemir  

**Link**: [PDF](https://arxiv.org/pdf/2502.17154)  

**Abstract**: Glaucoma is a prevalent eye disease that progresses silently without symptoms. If not detected and treated early, it can cause permanent vision loss. Computer-assisted diagnosis systems play a crucial role in timely and efficient identification. This study introduces MaxGlaViT, a lightweight model based on the restructured Multi-Axis Vision Transformer (MaxViT) for early glaucoma detection. First, MaxViT was scaled to optimize block and channel numbers, resulting in a lighter architecture. Second, the stem was enhanced by adding attention mechanisms (CBAM, ECA, SE) after convolution layers to improve feature learning. Third, MBConv structures in MaxViT blocks were replaced by advanced DL blocks (ConvNeXt, ConvNeXtV2, InceptionNeXt). The model was evaluated using the HDV1 dataset, containing fundus images of different glaucoma stages. Additionally, 40 CNN and 40 ViT models were tested on HDV1 to validate MaxGlaViT's efficiency. Among CNN models, EfficientB6 achieved the highest accuracy (84.91%), while among ViT models, MaxViT-Tiny performed best (86.42%). The scaled MaxViT reached 87.93% accuracy. Adding ECA to the stem block increased accuracy to 89.01%. Replacing MBConv with ConvNeXtV2 further improved it to 89.87%. Finally, integrating ECA in the stem and ConvNeXtV2 in MaxViT blocks resulted in 92.03% accuracy. Testing 80 DL models for glaucoma stage classification, this study presents a comprehensive and comparative analysis. MaxGlaViT outperforms experimental and state-of-the-art models, achieving 92.03% accuracy, 92.33% precision, 92.03% recall, 92.13% f1-score, and 87.12% Cohen's kappa score. 

**Abstract (ZH)**: 无症状进展的青光眼是一种常见的致盲性眼病。如果未能早期检测和治疗，将导致永久性视力丧失。基于重构的Multi-Axis Vision Transformer (MaxViT) 的轻量化模型MaxGlaViT在早期青光眼检测中的应用研究 

---
# SFLD: Reducing the content bias for AI-generated Image Detection 

**Title (ZH)**: SFLD: 减少AI生成图像检测中的内容偏见 

**Authors**: Seoyeon Gye, Junwon Ko, Hyounguk Shon, Minchan Kwon, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.17105)  

**Abstract**: Identifying AI-generated content is critical for the safe and ethical use of generative AI. Recent research has focused on developing detectors that generalize to unknown generators, with popular methods relying either on high-level features or low-level fingerprints. However, these methods have clear limitations: biased towards unseen content, or vulnerable to common image degradations, such as JPEG compression. To address these issues, we propose a novel approach, SFLD, which incorporates PatchShuffle to integrate high-level semantic and low-level textural information. SFLD applies PatchShuffle at multiple levels, improving robustness and generalization across various generative models. Additionally, current benchmarks face challenges such as low image quality, insufficient content preservation, and limited class diversity. In response, we introduce TwinSynths, a new benchmark generation methodology that constructs visually near-identical pairs of real and synthetic images to ensure high quality and content preservation. Our extensive experiments and analysis show that SFLD outperforms existing methods on detecting a wide variety of fake images sourced from GANs, diffusion models, and TwinSynths, demonstrating the state-of-the-art performance and generalization capabilities to novel generative models. 

**Abstract (ZH)**: 识别AI生成的内容对于安全和伦理使用生成型AI至关重要。近期的研究集中在开发能够泛化到未知生成器的检测器，现有方法主要依赖高层特征或低层指纹。然而，这些方法存在明显的局限性：要么偏向未见内容，要么易受JPEG压缩等常见图像降级的影响。为解决这些问题，我们提出了一个新颖的方法SFLD，该方法结合PatchShuffle以整合高层语义和低层纹理信息。SFLD在多个层面应用PatchShuffle，增强了模型的鲁棒性和泛化能力。此外，当前的基准测试存在图像质量低、内容保真度不足以及类别多样性有限等挑战。为此，我们引入了一种新的基准生成方法TwinSynths，构建了视觉上几乎相同的真伪图像对，以确保高质量和内容保真度。广泛的实验和分析表明，SFLD在检测来自GAN、扩散模型和TwinSynths的多种伪造图像方面优于现有方法，展示了其优异的性能和对新型生成模型的泛化能力。 

---
# ENACT-Heart -- ENsemble-based Assessment Using CNN and Transformer on Heart Sounds 

**Title (ZH)**: ENACT-Heart —— 基于集合评估的CNN和变压器在心音分析中的应用 

**Authors**: Jiho Han, Adnan Shaout  

**Link**: [PDF](https://arxiv.org/pdf/2502.16914)  

**Abstract**: This study explores the application of Vision Transformer (ViT) principles in audio analysis, specifically focusing on heart sounds. This paper introduces ENACT-Heart - a novel ensemble approach that leverages the complementary strengths of Convolutional Neural Networks (CNN) and ViT through a Mixture of Experts (MoE) framework, achieving a remarkable classification accuracy of 97.52%. This outperforms the individual contributions of ViT (93.88%) and CNN (95.45%), demonstrating the potential for enhanced diagnostic accuracy in cardiovascular health monitoring. These results demonstrate the potential of ensemble methods in enhancing classification performance for cardiovascular health monitoring and diagnosis. 

**Abstract (ZH)**: 本研究探讨了视觉变换器（ViT）原理在音频分析中的应用，特别聚焦于心音分析。本文介绍了ENACT-Heart——一种新颖的集成方法，该方法通过专家混合（MoE）框架利用卷积神经网络（CNN）和ViT的互补优势，实现了97.52%的显著分类准确率，优于单独的ViT（93.88%）和CNN（95.45%），展示了在心血管健康监测中增强诊断准确性的潜力。这些结果表明，集成方法在提高心血管健康监测和诊断的分类性能方面的潜在价值。 

---
# MambaFlow: A Novel and Flow-guided State Space Model for Scene Flow Estimation 

**Title (ZH)**: MambaFlow：一种新型的流引导场景流估计状态空间模型 

**Authors**: Jiehao Luo, Jintao Cheng, Xiaoyu Tang, Qingwen Zhang, Bohuan Xue, Rui Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.16907)  

**Abstract**: Scene flow estimation aims to predict 3D motion from consecutive point cloud frames, which is of great interest in autonomous driving field. Existing methods face challenges such as insufficient spatio-temporal modeling and inherent loss of fine-grained feature during voxelization. However, the success of Mamba, a representative state space model (SSM) that enables global modeling with linear complexity, provides a promising solution. In this paper, we propose MambaFlow, a novel scene flow estimation network with a mamba-based decoder. It enables deep interaction and coupling of spatio-temporal features using a well-designed backbone. Innovatively, we steer the global attention modeling of voxel-based features with point offset information using an efficient Mamba-based decoder, learning voxel-to-point patterns that are used to devoxelize shared voxel representations into point-wise features. To further enhance the model's generalization capabilities across diverse scenarios, we propose a novel scene-adaptive loss function that automatically adapts to different motion this http URL experiments on the Argoverse 2 benchmark demonstrate that MambaFlow achieves state-of-the-art performance with real-time inference speed among existing works, enabling accurate flow estimation in real-world urban scenarios. The code is available at this https URL. 

**Abstract (ZH)**: 场景流估计旨在从连续点云帧中预测3D运动，这是自主驾驶领域的热点问题。现有方法面临时空建模不足和体元化过程中固有的细粒度特征损失等挑战。然而，Mamba的成功，这是一种允许全局建模且具有线性复杂度的代表性状态空间模型（SSM），提供了前景解决方案。在本文中，我们提出了一种基于Mamba解码器的新型场景流估计网络MambaFlow，它通过精心设计的骨干网络实现了时空特征的深度交互和耦合。创新地，我们使用高效的Mamba基于解码器以点偏移信息引导体元特征的全局注意力建模，学习体元到点的模式，将其用于将共享体元表示去体元化为点 wise特征。为了进一步增强模型在不同场景中的泛化能力，我们提出了一种新的场景自适应损失函数，该函数能够自动适应不同的运动模式。在Argoverse 2基准测试上的实验表明，MambaFlow在现有工作中实现了实时推理速度下的最佳性能，能够在现实世界的城市场景中实现准确的流估计。代码可在此处获取。 

---
# Unposed Sparse Views Room Layout Reconstruction in the Age of Pretrain Model 

**Title (ZH)**: 预训练模型时代基于稀疏视角的室内布局重建 

**Authors**: Yaxuan Huang, Xili Dai, Jianan Wang, Xianbiao Qi, Yixing Yuan, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.16779)  

**Abstract**: Room layout estimation from multiple-perspective images is poorly investigated due to the complexities that emerge from multi-view geometry, which requires muti-step solutions such as camera intrinsic and extrinsic estimation, image matching, and triangulation. However, in 3D reconstruction, the advancement of recent 3D foundation models such as DUSt3R has shifted the paradigm from the traditional multi-step structure-from-motion process to an end-to-end single-step approach. To this end, we introduce Plane-DUSt3R}, a novel method for multi-view room layout estimation leveraging the 3D foundation model DUSt3R. Plane-DUSt3R incorporates the DUSt3R framework and fine-tunes on a room layout dataset (Structure3D) with a modified objective to estimate structural planes. By generating uniform and parsimonious results, Plane-DUSt3R enables room layout estimation with only a single post-processing step and 2D detection results. Unlike previous methods that rely on single-perspective or panorama image, Plane-DUSt3R extends the setting to handle multiple-perspective images. Moreover, it offers a streamlined, end-to-end solution that simplifies the process and reduces error accumulation. Experimental results demonstrate that Plane-DUSt3R not only outperforms state-of-the-art methods on the synthetic dataset but also proves robust and effective on in the wild data with different image styles such as cartoon. 

**Abstract (ZH)**: 基于多视角图像的房间布局估计由于多视角几何学带来的复杂性而研究不足，这需要多步解决方案，如相机内参和外参估计、图像匹配和三角测量。然而，在三维重建中，近期三维基础模型（如DUSt3R）的进步已经将范式从传统的多步结构从运动过程转变为端到端的一步式方法。为了解决这一问题，我们引入了Plane-DUSt3R，这是一种利用三维基础模型DUSt3R的新方法，用于多视角房间布局估计。Plane-DUSt3R结合了DUSt3R框架，并在房间布局数据集（Structure3D）上进行了微调，以估计结构平面。通过生成均匀且简洁的结果，Plane-DUSt3R使得房间布局估计只需一个后处理步骤和二维检测结果即可。Plane-DUSt3R不同于依赖单一视角或全景图像的先前方法，它可以处理多视角图像。此外，它提供了一种简化的过程和减少累积误差的端到端解决方案。实验结果表明，Plane-DUSt3R不仅在合成数据集上优于现有方法，而且在不同图像风格（如卡通）的野外数据上也表现出鲁棒性和有效性。 

---
# DOSE3 : Diffusion-based Out-of-distribution detection on SE(3) trajectories 

**Title (ZH)**: DOSE3：基于扩散的SE(3)轨迹异类检测 

**Authors**: Hongzhe Cheng, Tianyou Zheng, Tianyi Zhang, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2502.16725)  

**Abstract**: Out-of-Distribution(OOD) detection, a fundamental machine learning task aimed at identifying abnormal samples, traditionally requires model retraining for different inlier distributions. While recent research demonstrates the applicability of diffusion models to OOD detection, existing approaches are limited to Euclidean or latent image spaces. Our work extends OOD detection to trajectories in the Special Euclidean Group in 3D ($\mathbb{SE}(3)$), addressing a critical need in computer vision, robotics, and engineering applications that process object pose sequences in $\mathbb{SE}(3)$. We present $\textbf{D}$iffusion-based $\textbf{O}$ut-of-distribution detection on $\mathbb{SE}(3)$ ($\mathbf{DOSE3}$), a novel OOD framework that extends diffusion to a unified sample space of $\mathbb{SE}(3)$ pose sequences. Through extensive validation on multiple benchmark datasets, we demonstrate $\mathbf{DOSE3}$'s superior performance compared to state-of-the-art OOD detection frameworks. 

**Abstract (ZH)**: 基于特殊欧几里得群$\mathbb{SE}(3)$的扩散模型异常检测（$\mathbf{DOSE3}$） 

---
# Can Large Vision-Language Models Detect Images Copyright Infringement from GenAI? 

**Title (ZH)**: 大规模vision-language模型能否检测来自GenAI的图像版权侵权？ 

**Authors**: Qipan Xu, Zhenting Wang, Xiaoxiao He, Ligong Han, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16618)  

**Abstract**: Generative AI models, renowned for their ability to synthesize high-quality content, have sparked growing concerns over the improper generation of copyright-protected material. While recent studies have proposed various approaches to address copyright issues, the capability of large vision-language models (LVLMs) to detect copyright infringements remains largely unexplored. In this work, we focus on evaluating the copyright detection abilities of state-of-the-art LVLMs using a various set of image samples. Recognizing the absence of a comprehensive dataset that includes both IP-infringement samples and ambiguous non-infringement negative samples, we construct a benchmark dataset comprising positive samples that violate the copyright protection of well-known IP figures, as well as negative samples that resemble these figures but do not raise copyright concerns. This dataset is created using advanced prompt engineering techniques. We then evaluate leading LVLMs using our benchmark dataset. Our experimental results reveal that LVLMs are prone to overfitting, leading to the misclassification of some negative samples as IP-infringement cases. In the final section, we analyze these failure cases and propose potential solutions to mitigate the overfitting problem. 

**Abstract (ZH)**: 生成式AI模型因其生成高质量内容的能力而闻名，但过度生成受版权保护的材料引发了日益增长的担忧。虽然近期研究提出了一些解决版权问题的方法，但大型视觉-语言模型（LVLMs）检测版权侵权的能力尚未得到充分探索。在本研究中，我们利用多种图像样本评估最先进的LVLMs的版权检测能力。鉴于缺乏一个包含IP侵权样本和模糊的非侵权负样本的全面数据集，我们构建了一个基准数据集，该数据集包含侵犯知名IP形象版权的正样本，以及相似但不引发版权担忧的负样本。该数据集使用高级提示工程技巧构建。然后，我们使用基准数据集评估领先的LVLMs。实验结果表明，LVLMs容易过拟合，导致一些负样本被误分类为IP侵权案例。在最终部分，我们分析了这些失败案例，并提出可能的解决方案以缓解过拟合问题。 

---
# AdverX-Ray: Ensuring X-Ray Integrity Through Frequency-Sensitive Adversarial VAEs 

**Title (ZH)**: AdverX-Ray：通过频率敏感对抗变分自编码器确保X射线完整性 

**Authors**: Francisco Caetano, Christiaan Viviers, Lena Filatova, Peter H. N. de With, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2502.16610)  

**Abstract**: Ensuring the quality and integrity of medical images is crucial for maintaining diagnostic accuracy in deep learning-based Computer-Aided Diagnosis and Computer-Aided Detection (CAD) systems. Covariate shifts are subtle variations in the data distribution caused by different imaging devices or settings and can severely degrade model performance, similar to the effects of adversarial attacks. Therefore, it is vital to have a lightweight and fast method to assess the quality of these images prior to using CAD models. AdverX-Ray addresses this need by serving as an image-quality assessment layer, designed to detect covariate shifts effectively. This Adversarial Variational Autoencoder prioritizes the discriminator's role, using the suboptimal outputs of the generator as negative samples to fine-tune the discriminator's ability to identify high-frequency artifacts. Images generated by adversarial networks often exhibit severe high-frequency artifacts, guiding the discriminator to focus excessively on these components. This makes the discriminator ideal for this approach. Trained on patches from X-ray images of specific machine models, AdverX-Ray can evaluate whether a scan matches the training distribution, or if a scan from the same machine is captured under different settings. Extensive comparisons with various OOD detection methods show that AdverX-Ray significantly outperforms existing techniques, achieving a 96.2% average AUROC using only 64 random patches from an X-ray. Its lightweight and fast architecture makes it suitable for real-time applications, enhancing the reliability of medical imaging systems. The code and pretrained models are publicly available. 

**Abstract (ZH)**: 确保医学图像的质量和完整性对于维持基于深度学习的计算机辅助诊断和检测(CAD)系统的诊断准确性至关重要。AdverX-Ray通过 Serving as an 图像质量评估层，有效检测协变量偏移，解决这一需求。基于对抗变分自编码器，该方法优先考虑判别器的作用，利用生成器的次优输出作为负样本，以精细化判别器识别高频伪影的能力。由对抗网络生成的图像常常表现出严重的高频伪影，这使判别器过度关注这些成分。AdverX-Ray 仅使用 X 射线特定机器模型图像的 64 个随机补丁块进行训练，即可评估扫描是否匹配训练分布，或在相同机器的不同设置下捕捉扫描。与各种OOD检测方法的广泛比较显示，AdverX-Ray 显著优于现有技术，平均AUROC达到96.2%。其轻量级和快速架构使其适合实时应用，提高医学成像系统的可靠性。代码和预训练模型已开源。 

---
# Co-MTP: A Cooperative Trajectory Prediction Framework with Multi-Temporal Fusion for Autonomous Driving 

**Title (ZH)**: Co-MTP：一种基于多时态融合的协同轨迹预测框架用于自动驾驶 

**Authors**: Xinyu Zhang, Zewei Zhou, Zhaoyi Wang, Yangjie Ji, Yanjun Huang, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.16589)  

**Abstract**: Vehicle-to-everything technologies (V2X) have become an ideal paradigm to extend the perception range and see through the occlusion. Exiting efforts focus on single-frame cooperative perception, however, how to capture the temporal cue between frames with V2X to facilitate the prediction task even the planning task is still underexplored. In this paper, we introduce the Co-MTP, a general cooperative trajectory prediction framework with multi-temporal fusion for autonomous driving, which leverages the V2X system to fully capture the interaction among agents in both history and future domains to benefit the planning. In the history domain, V2X can complement the incomplete history trajectory in single-vehicle perception, and we design a heterogeneous graph transformer to learn the fusion of the history feature from multiple agents and capture the history interaction. Moreover, the goal of prediction is to support future planning. Thus, in the future domain, V2X can provide the prediction results of surrounding objects, and we further extend the graph transformer to capture the future interaction among the ego planning and the other vehicles' intentions and obtain the final future scenario state under a certain planning action. We evaluate the Co-MTP framework on the real-world dataset V2X-Seq, and the results show that Co-MTP achieves state-of-the-art performance and that both history and future fusion can greatly benefit prediction. 

**Abstract (ZH)**: Vehicle-to-Everything技术（V2X）已成为扩展感知范围并穿透遮挡的理想范式。现有努力集中在单帧协同感知上，然而，如何利用V2X捕捉帧间的时间线索以辅助预测任务乃至规划任务仍然有待探索。在本文中，我们提出了Co-MTP，一种基于多时序融合的自主驾驶通用协同轨迹预测框架，利用V2X系统全面捕捉历史和未来领域中的 agent 交互，以利于规划。在历史领域，V2X可以补充单车辆感知中的不完整历史轨迹，我们设计了一个异构图变换器来学习多agent的历史特征融合并捕捉历史交互。此外，预测的目标是支持未来的规划。因此，在未来领域，V2X可以提供周围物体的预测结果，并进一步扩展图变换器以捕捉自主规划与其他车辆意图之间的未来交互，从而在特定规划行动下获取最终的未来场景状态。我们在现实世界数据集V2X-Seq上评估了Co-MTP框架，结果表明Co-MTP取得了最先进的性能，并且历史和未来融合可以显著提高预测效果。 

---
# On Computational Limits of FlowAR Models: Expressivity and Efficiency 

**Title (ZH)**: 关于FlowAR模型的计算限制：表达能力和效率 

**Authors**: Chengyue Gong, Yekun Ke, Xiaoyu Li, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.16490)  

**Abstract**: The expressive power and computational complexity of deep visual generative models, such as flow-based and autoregressive (AR) models, have gained considerable interest for their wide-ranging applications in generative tasks. However, the theoretical characterization of their expressiveness through the lens of circuit complexity remains underexplored, particularly for the state-of-the-art architecture like FlowAR proposed by [Ren et al., 2024], which integrates flow-based and autoregressive mechanisms. This gap limits our understanding of their inherent computational limits and practical efficiency. In this study, we address this gap by analyzing the circuit complexity of the FlowAR architecture. We demonstrate that when the largest feature map produced by the FlowAR model has dimensions $n \times n \times c$, the FlowAR model is simulable by a family of threshold circuits $\mathsf{TC}^0$, which have constant depth $O(1)$ and polynomial width $\mathrm{poly}(n)$. This is the first study to rigorously highlight the limitations in the expressive power of FlowAR models. Furthermore, we identify the conditions under which the FlowAR model computations can achieve almost quadratic time. To validate our theoretical findings, we present efficient model variant constructions based on low-rank approximations that align with the derived criteria. Our work provides a foundation for future comparisons with other generative paradigms and guides the development of more efficient and expressive implementations. 

**Abstract (ZH)**: 基于电路复杂性的深度视觉生成模型，如流动基于和自回归模型表达能力和计算复杂性的理论刻画研究——以Ren等人提出的FlowAR架构为例 

---
# Dragen3D: Multiview Geometry Consistent 3D Gaussian Generation with Drag-Based Control 

**Title (ZH)**: Dragen3D：基于拖拽控制的多视图几何一致的3D高斯生成 

**Authors**: Jinbo Yan, Alan Zhao, Yixin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16475)  

**Abstract**: Single-image 3D generation has emerged as a prominent research topic, playing a vital role in virtual reality, 3D modeling, and digital content creation. However, existing methods face challenges such as a lack of multi-view geometric consistency and limited controllability during the generation process, which significantly restrict their usability. % To tackle these challenges, we introduce Dragen3D, a novel approach that achieves geometrically consistent and controllable 3D generation leveraging 3D Gaussian Splatting (3DGS). We introduce the Anchor-Gaussian Variational Autoencoder (Anchor-GS VAE), which encodes a point cloud and a single image into anchor latents and decode these latents into 3DGS, enabling efficient latent-space generation. To enable multi-view geometry consistent and controllable generation, we propose a Seed-Point-Driven strategy: first generate sparse seed points as a coarse geometry representation, then map them to anchor latents via the Seed-Anchor Mapping Module. Geometric consistency is ensured by the easily learned sparse seed points, and users can intuitively drag the seed points to deform the final 3DGS geometry, with changes propagated through the anchor latents. To the best of our knowledge, we are the first to achieve geometrically controllable 3D Gaussian generation and editing without relying on 2D diffusion priors, delivering comparable 3D generation quality to state-of-the-art methods. 

**Abstract (ZH)**: 单幅图像的单次3D生成：一种基于3D高斯点扩散的几何一致性和可控性方法 

---
# Deep learning approaches to surgical video segmentation and object detection: A Scoping Review 

**Title (ZH)**: 深度学习在手术视频分割和对象检测中的应用：一个综述研究 

**Authors**: Devanish N. Kamtam, Joseph B. Shrager, Satya Deepya Malla, Nicole Lin, Juan J. Cardona, Jake J. Kim, Clarence Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16459)  

**Abstract**: Introduction: Computer vision (CV) has had a transformative impact in biomedical fields such as radiology, dermatology, and pathology. Its real-world adoption in surgical applications, however, remains limited. We review the current state-of-the-art performance of deep learning (DL)-based CV models for segmentation and object detection of anatomical structures in videos obtained during surgical procedures.
Methods: We conducted a scoping review of studies on semantic segmentation and object detection of anatomical structures published between 2014 and 2024 from 3 major databases - PubMed, Embase, and IEEE Xplore. The primary objective was to evaluate the state-of-the-art performance of semantic segmentation in surgical videos. Secondary objectives included examining DL models, progress toward clinical applications, and the specific challenges with segmentation of organs/tissues in surgical videos.
Results: We identified 58 relevant published studies. These focused predominantly on procedures from general surgery [20(34.4%)], colorectal surgery [9(15.5%)], and neurosurgery [8(13.8%)]. Cholecystectomy [14(24.1%)] and low anterior rectal resection [5(8.6%)] were the most common procedures addressed. Semantic segmentation [47(81%)] was the primary CV task. U-Net [14(24.1%)] and DeepLab [13(22.4%)] were the most widely used models. Larger organs such as the liver (Dice score: 0.88) had higher accuracy compared to smaller structures such as nerves (Dice score: 0.49). Models demonstrated real-time inference potential ranging from 5-298 frames-per-second (fps).
Conclusion: This review highlights the significant progress made in DL-based semantic segmentation for surgical videos with real-time applicability, particularly for larger organs. Addressing challenges with smaller structures, data availability, and generalizability remains crucial for future advancements. 

**Abstract (ZH)**: 计算机视觉在手术视频中解剖结构分割与对象检测的现状：从2014年至2024年的综述 

---
# Iterative Flow Matching -- Path Correction and Gradual Refinement for Enhanced Generative Modeling 

**Title (ZH)**: 迭代流匹配——路径修正与逐步细化以增强生成建模 

**Authors**: Eldad Haber, Shadab Ahamed, Md. Shahriar Rahim Siddiqui, Niloufar Zakariaei, Moshe Eliasof  

**Link**: [PDF](https://arxiv.org/pdf/2502.16445)  

**Abstract**: Generative models for image generation are now commonly used for a wide variety of applications, ranging from guided image generation for entertainment to solving inverse problems. Nonetheless, training a generator is a non-trivial feat that requires fine-tuning and can lead to so-called hallucinations, that is, the generation of images that are unrealistic. In this work, we explore image generation using flow matching. We explain and demonstrate why flow matching can generate hallucinations, and propose an iterative process to improve the generation process. Our iterative process can be integrated into virtually $\textit{any}$ generative modeling technique, thereby enhancing the performance and robustness of image synthesis systems. 

**Abstract (ZH)**: 图像生成的流匹配生成模型：探索、分析及迭代改进方法 

---
# An Expert Ensemble for Detecting Anomalous Scenes, Interactions, and Behaviors in Autonomous Driving 

**Title (ZH)**: 基于专家集成的自动驾驶中异常场景、交互和行为检测方法 

**Authors**: Tianchen Ji, Neeloy Chakraborty, Andre Schreiber, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2502.16389)  

**Abstract**: As automated vehicles enter public roads, safety in a near-infinite number of driving scenarios becomes one of the major concerns for the widespread adoption of fully autonomous driving. The ability to detect anomalous situations outside of the operational design domain is a key component in self-driving cars, enabling us to mitigate the impact of abnormal ego behaviors and to realize trustworthy driving systems. On-road anomaly detection in egocentric videos remains a challenging problem due to the difficulties introduced by complex and interactive scenarios. We conduct a holistic analysis of common on-road anomaly patterns, from which we propose three unsupervised anomaly detection experts: a scene expert that focuses on frame-level appearances to detect abnormal scenes and unexpected scene motions; an interaction expert that models normal relative motions between two road participants and raises alarms whenever anomalous interactions emerge; and a behavior expert which monitors abnormal behaviors of individual objects by future trajectory prediction. To combine the strengths of all the modules, we propose an expert ensemble (Xen) using a Kalman filter, in which the final anomaly score is absorbed as one of the states and the observations are generated by the experts. Our experiments employ a novel evaluation protocol for realistic model performance, demonstrate superior anomaly detection performance than previous methods, and show that our framework has potential in classifying anomaly types using unsupervised learning on a large-scale on-road anomaly dataset. 

**Abstract (ZH)**: 随着自动驾驶车辆进入公共道路，无限多驾驶场景中的安全性成为全面推广完全自动驾驶技术的主要关切。检测超出操作设计域的异常情况能力是自动驾驶汽车的关键组成部分，这使我们能够减轻异常自我行为的影响并实现可信赖的驾驶系统。由于复杂且交互性的场景带来的困难，基于第一人称视频的道路异常检测仍然是一个具有挑战性的问题。我们对常见的道路异常模式进行了全面分析，并提出三种无监督异常检测专家：场景专家专注于帧级外观以检测异常场景和意外场景运动；交互专家建模两个道路参与者之间的正常相对运动，并在异常交互出现时发出警报；行为专家通过未来轨迹预测监控个体对象的异常行为。为了整合各模块的优势，我们提出了一种专家集成（Xen），使用卡尔曼滤波器，在其中最终的异常分数作为状态之一，并由专家生成观察值。我们的实验采用了新的评估协议以展示现实模型性能，表明我们的方法在异常检测性能上优于以往方法，并展示了我们在大规模道路上异常数据集上使用无监督学习分类异常类型方面的潜力。 

---
# SalM$2$: An Extremely Lightweight Saliency Mamba Model for Real-Time Cognitive Awareness of Driver Attention 

**Title (ZH)**: SalM$2$: 一种极轻量级的驾驶注意力实时认知 Awareness 模型 

**Authors**: Chunyu Zhao, Wentao Mu, Xian Zhou, Wenbo Liu, Fei Yan, Tao Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.16214)  

**Abstract**: Driver attention recognition in driving scenarios is a popular direction in traffic scene perception technology. It aims to understand human driver attention to focus on specific targets/objects in the driving scene. However, traffic scenes contain not only a large amount of visual information but also semantic information related to driving tasks. Existing methods lack attention to the actual semantic information present in driving scenes. Additionally, the traffic scene is a complex and dynamic process that requires constant attention to objects related to the current driving task. Existing models, influenced by their foundational frameworks, tend to have large parameter counts and complex structures. Therefore, this paper proposes a real-time saliency Mamba network based on the latest Mamba framework. As shown in Figure 1, our model uses very few parameters (0.08M, only 0.09~11.16% of other models), while maintaining SOTA performance or achieving over 98% of the SOTA model's performance. 

**Abstract (ZH)**: 驾驶场景中驾驶员注意力识别是交通场景感知技术的一个热门方向 

---
# Robust Dynamic Facial Expression Recognition 

**Title (ZH)**: 鲁棒动态面部表情识别 

**Authors**: Feng Liu, Hanyang Wang, Siyuan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.16129)  

**Abstract**: The study of Dynamic Facial Expression Recognition (DFER) is a nascent field of research that involves the automated recognition of facial expressions in video data. Although existing research has primarily focused on learning representations under noisy and hard samples, the issue of the coexistence of both types of samples remains unresolved. In order to overcome this challenge, this paper proposes a robust method of distinguishing between hard and noisy samples. This is achieved by evaluating the prediction agreement of the model on different sampled clips of the video. Subsequently, methodologies that reinforce the learning of hard samples and mitigate the impact of noisy samples can be employed. Moreover, to identify the principal expression in a video and enhance the model's capacity for representation learning, comprising a key expression re-sampling framework and a dual-stream hierarchical network is proposed, namely Robust Dynamic Facial Expression Recognition (RDFER). The key expression re-sampling framework is designed to identify the key expression, thereby mitigating the potential confusion caused by non-target expressions. RDFER employs two sequence models with the objective of disentangling short-term facial movements and long-term emotional changes. The proposed method has been shown to outperform current State-Of-The-Art approaches in DFER through extensive experimentation on benchmark datasets such as DFEW and FERV39K. A comprehensive analysis provides valuable insights and observations regarding the proposed agreement. This work has significant implications for the field of dynamic facial expression recognition and promotes the further development of the field of noise-consistent robust learning in dynamic facial expression recognition. The code is available from [this https URL]. 

**Abstract (ZH)**: 动态面部表情识别中的坚实动态面部表情识别（RDFER）：噪声与困难样本共存下的表情识别研究 

---
# Cross-Model Transferability of Adversarial Patches in Real-time Segmentation for Autonomous Driving 

**Title (ZH)**: 实时分割领域中自动驾驶中对抗性补丁的跨模型通用性研究 

**Authors**: Prashant Shekhar, Bidur Devkota, Dumindu Samaraweera, Laxima Niure Kandel, Manoj Babu  

**Link**: [PDF](https://arxiv.org/pdf/2502.16012)  

**Abstract**: Adversarial attacks pose a significant threat to deep learning models, particularly in safety-critical applications like healthcare and autonomous driving. Recently, patch based attacks have demonstrated effectiveness in real-time inference scenarios owing to their 'drag and drop' nature. Following this idea for Semantic Segmentation (SS), here we propose a novel Expectation Over Transformation (EOT) based adversarial patch attack that is more realistic for autonomous vehicles. To effectively train this attack we also propose a 'simplified' loss function that is easy to analyze and implement. Using this attack as our basis, we investigate whether adversarial patches once optimized on a specific SS model, can fool other models or architectures. We conduct a comprehensive cross-model transferability analysis of adversarial patches trained on SOTA Convolutional Neural Network (CNN) models such PIDNet-S, PIDNet-M and PIDNet-L, among others. Additionally, we also include the Segformer model to study transferability to Vision Transformers (ViTs). All of our analysis is conducted on the widely used Cityscapes dataset. Our study reveals key insights into how model architectures (CNN vs CNN or CNN vs. Transformer-based) influence attack susceptibility. In particular, we conclude that although the transferability (effectiveness) of attacks on unseen images of any dimension is really high, the attacks trained against one particular model are minimally effective on other models. And this was found to be true for both ViT and CNN based models. Additionally our results also indicate that for CNN-based models, the repercussions of patch attacks are local, unlike ViTs. Per-class analysis reveals that simple-classes like 'sky' suffer less misclassification than others. The code for the project is available at: this https URL 

**Abstract (ZH)**: 基于斑马技术的对抗攻击：一种面向自主车辆更加现实的期望转换（EOT）方法及其跨模型可转移性分析 

---
# Graph Attention Convolutional U-NET: A Semantic Segmentation Model for Identifying Flooded Areas 

**Title (ZH)**: 基于图注意力卷积的U-NET：一种识别淹没区域的语义分割模型 

**Authors**: Muhammad Umair Danish, Madhushan Buwaneswaran, Tehara Fonseka, Katarina Grolinger  

**Link**: [PDF](https://arxiv.org/pdf/2502.15907)  

**Abstract**: The increasing impact of human-induced climate change and unplanned urban constructions has increased flooding incidents in recent years. Accurate identification of flooded areas is crucial for effective disaster management and urban planning. While few works have utilized convolutional neural networks and transformer-based semantic segmentation techniques for identifying flooded areas from aerial footage, recent developments in graph neural networks have created improvement opportunities. This paper proposes an innovative approach, the Graph Attention Convolutional U-NET (GAC-UNET) model, based on graph neural networks for automated identification of flooded areas. The model incorporates a graph attention mechanism and Chebyshev layers into the U-Net architecture. Furthermore, this paper explores the applicability of transfer learning and model reprogramming to enhance the accuracy of flood area segmentation models. Empirical results demonstrate that the proposed GAC-UNET model, outperforms other approaches with 91\% mAP, 94\% dice score, and 89\% IoU, providing valuable insights for informed decision-making and better planning of future infrastructures in flood-prone areas. 

**Abstract (ZH)**: 基于图神经网络的图注意力卷积U-NET模型在自动识别浸水区域的应用 

---
# Generative AI Framework for 3D Object Generation in Augmented Reality 

**Title (ZH)**: 三维对象生成的生成AI框架在增强现实中的应用 

**Authors**: Majid Behravan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15869)  

**Abstract**: This thesis presents a framework that integrates state-of-the-art generative AI models for real-time creation of three-dimensional (3D) objects in augmented reality (AR) environments. The primary goal is to convert diverse inputs, such as images and speech, into accurate 3D models, enhancing user interaction and immersion. Key components include advanced object detection algorithms, user-friendly interaction techniques, and robust AI models like Shap-E for 3D generation. Leveraging Vision Language Models (VLMs) and Large Language Models (LLMs), the system captures spatial details from images and processes textual information to generate comprehensive 3D objects, seamlessly integrating virtual objects into real-world environments. The framework demonstrates applications across industries such as gaming, education, retail, and interior design. It allows players to create personalized in-game assets, customers to see products in their environments before purchase, and designers to convert real-world objects into 3D models for real-time visualization. A significant contribution is democratizing 3D model creation, making advanced AI tools accessible to a broader audience, fostering creativity and innovation. The framework addresses challenges like handling multilingual inputs, diverse visual data, and complex environments, improving object detection and model generation accuracy, as well as loading 3D models in AR space in real-time. In conclusion, this thesis integrates generative AI and AR for efficient 3D model generation, enhancing accessibility and paving the way for innovative applications and improved user interactions in AR environments. 

**Abstract (ZH)**: 一种将前沿生成式AI模型集成以实现实时生成增强现实环境中三维对象的框架：多模态输入到精确三维模型的转换与应用 

---
# Spiking Point Transformer for Point Cloud Classification 

**Title (ZH)**: 基于尖峰点变换器的点云分类 

**Authors**: Peixi Wu, Bosong Chai, Hebei Li, Menghua Zheng, Yansong Peng, Zeyu Wang, Xuan Nie, Yueyi Zhang, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.15811)  

**Abstract**: Spiking Neural Networks (SNNs) offer an attractive and energy-efficient alternative to conventional Artificial Neural Networks (ANNs) due to their sparse binary activation. When SNN meets Transformer, it shows great potential in 2D image processing. However, their application for 3D point cloud remains underexplored. To this end, we present Spiking Point Transformer (SPT), the first transformer-based SNN framework for point cloud classification. Specifically, we first design Queue-Driven Sampling Direct Encoding for point cloud to reduce computational costs while retaining the most effective support points at each time step. We introduce the Hybrid Dynamics Integrate-and-Fire Neuron (HD-IF), designed to simulate selective neuron activation and reduce over-reliance on specific artificial neurons. SPT attains state-of-the-art results on three benchmark datasets that span both real-world and synthetic datasets in the SNN domain. Meanwhile, the theoretical energy consumption of SPT is at least 6.4$\times$ less than its ANN counterpart. 

**Abstract (ZH)**: 基于脉冲的点云变换器（SPT）：点云分类的变压器基脉冲神经网络框架 

---
# A Performance Analysis of You Only Look Once Models for Deployment on Constrained Computational Edge Devices in Drone Applications 

**Title (ZH)**: 约束计算边缘设备上无人机应用中You Only Look Once模型的性能分析 

**Authors**: Lucas Rey, Ana M. Bernardos, Andrzej D. Dobrzycki, David Carramiñana, Luca Bergesio, Juan A. Besada, José Ramón Casar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15737)  

**Abstract**: Advancements in embedded systems and Artificial Intelligence (AI) have enhanced the capabilities of Unmanned Aircraft Vehicles (UAVs) in computer vision. However, the integration of AI techniques o-nboard drones is constrained by their processing capabilities. In this sense, this study evaluates the deployment of object detection models (YOLOv8n and YOLOv8s) on both resource-constrained edge devices and cloud environments. The objective is to carry out a comparative performance analysis using a representative real-time UAV image processing pipeline. Specifically, the NVIDIA Jetson Orin Nano, Orin NX, and Raspberry Pi 5 (RPI5) devices have been tested to measure their detection accuracy, inference speed, and energy consumption, and the effects of post-training quantization (PTQ). The results show that YOLOv8n surpasses YOLOv8s in its inference speed, achieving 52 FPS on the Jetson Orin NX and 65 fps with INT8 quantization. Conversely, the RPI5 failed to satisfy the real-time processing needs in spite of its suitability for low-energy consumption applications. An analysis of both the cloud-based and edge-based end-to-end processing times showed that increased communication latencies hindered real-time applications, revealing trade-offs between edge (low latency) and cloud processing (quick processing). Overall, these findings contribute to providing recommendations and optimization strategies for the deployment of AI models on UAVs. 

**Abstract (ZH)**: 嵌入式系统和人工智能的进步提升了无人机在计算机视觉方面的能力。然而，无人机上集成人工智能技术受其计算能力的限制。在这种情况下，本研究评估了将YOLOv8n和YOLOv8s目标检测模型部署在资源受限的边缘设备和云环境中。目的是通过一个有代表性的实时无人机图像处理流水线进行性能比较分析。具体来说，测试了NVIDIA Jetson Orin Nano、Orin NX和Raspberry Pi 5 (RPI5)设备，以测量其检测精度、推理速度和能源消耗，并分析了后训练量化(PTQ)的影响。结果表明，YOLOv8n在推理速度上优于YOLOv8s，在Jetson Orin NX上的帧率达到了52 FPS，并通过INT8量化实现了65 fps。相反，RPI5未能满足实时处理需求，尽管其适合低能耗应用。云环境和边缘环境端到端处理时间的分析表明，增加的通信延迟阻碍了实时应用，揭示了边缘（低延迟）和云处理（快速处理）之间的权衡。总体来说，这些发现为无人机上部署AI模型提供了建议和优化策略。 

---
