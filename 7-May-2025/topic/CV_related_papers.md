# AquaticVision: Benchmarking Visual SLAM in Underwater Environment with Events and Frames 

**Title (ZH)**: AquaticVision: 在水下环境中基于事件和帧的视觉SLAM基准测试 

**Authors**: Yifan Peng, Yuze Hong, Ziyang Hong, Apple Pui-Yi Chui, Junfeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03448)  

**Abstract**: Many underwater applications, such as offshore asset inspections, rely on visual inspection and detailed 3D reconstruction. Recent advancements in underwater visual SLAM systems for aquatic environments have garnered significant attention in marine robotics research. However, existing underwater visual SLAM datasets often lack groundtruth trajectory data, making it difficult to objectively compare the performance of different SLAM algorithms based solely on qualitative results or COLMAP reconstruction. In this paper, we present a novel underwater dataset that includes ground truth trajectory data obtained using a motion capture system. Additionally, for the first time, we release visual data that includes both events and frames for benchmarking underwater visual positioning. By providing event camera data, we aim to facilitate the development of more robust and advanced underwater visual SLAM algorithms. The use of event cameras can help mitigate challenges posed by extremely low light or hazy underwater conditions. The webpage of our dataset is this https URL. 

**Abstract (ZH)**: 许多水下应用，如离岸资产检查，依赖于视觉检查和详细的3D重建。近年来，适用于水下环境的视觉SLAM系统在水下机器人研究领域引起了广泛关注。然而，现有的水下视觉SLAM数据集往往缺乏地面truth轨迹数据，这使得仅凭定性结果或COLMAP重建难以客观比较不同SLAM算法的性能。在这篇论文中，我们提出了一种新的水下数据集，其中包含使用运动捕捉系统获得的地面truth轨迹数据。此外，我们首次发布了包含事件和帧数据的视觉数据，用于水下视觉定位的基准测试。通过提供事件摄像头数据，我们旨在促进更 robust和先进的水下视觉SLAM算法的发展。事件摄像头的使用有助于缓解极低光照或浑浊水下条件带来的挑战。我们的数据集网址为：https://www.example.com。 

---
# HCOA*: Hierarchical Class-ordered A* for Navigation in Semantic Environments 

**Title (ZH)**: HCOA*: 分层类序A*在语义环境中的导航算法 

**Authors**: Evangelos Psomiadis, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2505.03128)  

**Abstract**: This paper addresses the problem of robot navigation in mixed geometric and semantic 3D environments. Given a hierarchical representation of the environment, the objective is to navigate from a start position to a goal while minimizing the computational cost. We introduce Hierarchical Class-ordered A* (HCOA*), an algorithm that leverages the environmental hierarchy for efficient path-planning in semantic graphs, significantly reducing computational effort. We use a total order over the semantic classes and prove theoretical performance guarantees for the algorithm. We propose two approaches for higher-layer node classification based on the node semantics of the lowest layer: a Graph Neural Network-based method and a Majority-Class method. We evaluate our approach through simulations on a 3D Scene Graph (3DSG), comparing it to the state-of-the-art and assessing its performance against our classification approaches. Results show that HCOA* can find the optimal path while reducing the number of expanded nodes by 25% and achieving a 16% reduction in computational time on the uHumans2 3DSG dataset. 

**Abstract (ZH)**: 本文探讨了在混合几何和语义3D环境下的机器人导航问题。给定环境的层次化表示，目标是从起始位置导航到目标位置同时最小化计算成本。我们引入了层次类别有序A* (HCOA*)算法，该算法利用环境层次信息进行语义图中的高效路径规划，显著降低计算开销。我们使用语义类的全序，并证明了该算法的理论性能保证。我们提出了两种基于低层节点语义的高层节点分类方法：基于图神经网络的方法和多数类方法。我们通过在3D场景图（3DSG）上的仿真实验评估了该方法，并将其与最先进的方法进行了比较，同时评估了其性能对分类方法的影响。结果表明，HCOA*能够找到最优路径，并在uHumans2 3DSG数据集中将展开节点数减少25%，同时计算时间减少16%。 

---
# Sim2Real Transfer for Vision-Based Grasp Verification 

**Title (ZH)**: 基于视觉的抓取验证的Sim2Real迁移学习 

**Authors**: Pau Amargant, Peter Hönig, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2505.03046)  

**Abstract**: The verification of successful grasps is a crucial aspect of robot manipulation, particularly when handling deformable objects. Traditional methods relying on force and tactile sensors often struggle with deformable and non-rigid objects. In this work, we present a vision-based approach for grasp verification to determine whether the robotic gripper has successfully grasped an object. Our method employs a two-stage architecture; first YOLO-based object detection model to detect and locate the robot's gripper and then a ResNet-based classifier determines the presence of an object. To address the limitations of real-world data capture, we introduce HSR-GraspSynth, a synthetic dataset designed to simulate diverse grasping scenarios. Furthermore, we explore the use of Visual Question Answering capabilities as a zero-shot baseline to which we compare our model. Experimental results demonstrate that our approach achieves high accuracy in real-world environments, with potential for integration into grasping pipelines. Code and datasets are publicly available at this https URL . 

**Abstract (ZH)**: 基于视觉的抓取验证在机器人操作中的研究：一种用于确定机器人成功抓取物体的方法 

---
# Matching Distance and Geometric Distribution Aided Learning Multiview Point Cloud Registration 

**Title (ZH)**: 基于匹配距离和几何分布辅助的多视点点云配准学习 

**Authors**: Shiqi Li, Jihua Zhu, Yifan Xie, Naiwen Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03692)  

**Abstract**: Multiview point cloud registration plays a crucial role in robotics, automation, and computer vision fields. This paper concentrates on pose graph construction and motion synchronization within multiview registration. Previous methods for pose graph construction often pruned fully connected graphs or constructed sparse graph using global feature aggregated from local descriptors, which may not consistently yield reliable results. To identify dependable pairs for pose graph construction, we design a network model that extracts information from the matching distance between point cloud pairs. For motion synchronization, we propose another neural network model to calculate the absolute pose in a data-driven manner, rather than optimizing inaccurate handcrafted loss functions. Our model takes into account geometric distribution information and employs a modified attention mechanism to facilitate flexible and reliable feature interaction. Experimental results on diverse indoor and outdoor datasets confirm the effectiveness and generalizability of our approach. The source code is available at this https URL. 

**Abstract (ZH)**: 多视图点云配准在机器人学、自动化和计算机视觉领域中起着关键作用。本文集中于多视图配准中的姿态图构造和运动同步。现有的姿态图构造方法通常会剪枝全连接图或将稀疏图构建基于局部描述子的全局特征聚合，这可能无法稳定地产生可靠的结果。为识别用于姿态图构造的可信配对，我们设计了一个网络模型，从中提取点云配对之间的匹配距离信息。对于运动同步，我们提出了一种基于数据驱动的方法来计算绝对姿态，而不是优化不准确的手工设计损失函数。我们的模型考虑了几何分布信息，并采用了修改后的注意机制来促进灵活可靠的特征交互。在多种室内和室外数据集上的实验结果证实了我们方法的有效性和泛化能力。相关源代码可在此处获取。 

---
# Panoramic Out-of-Distribution Segmentation 

**Title (ZH)**: 全景分布外分割 

**Authors**: Mengfei Duan, Kailun Yang, Yuheng Zhang, Yihong Cao, Fei Teng, Kai Luo, Jiaming Zhang, Zhiyong Li, Shutao Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03539)  

**Abstract**: Panoramic imaging enables capturing 360° images with an ultra-wide Field-of-View (FoV) for dense omnidirectional perception. However, current panoramic semantic segmentation methods fail to identify outliers, and pinhole Out-of-distribution Segmentation (OoS) models perform unsatisfactorily in the panoramic domain due to background clutter and pixel distortions. To address these issues, we introduce a new task, Panoramic Out-of-distribution Segmentation (PanOoS), achieving OoS for panoramas. Furthermore, we propose the first solution, POS, which adapts to the characteristics of panoramic images through text-guided prompt distribution learning. Specifically, POS integrates a disentanglement strategy designed to materialize the cross-domain generalization capability of CLIP. The proposed Prompt-based Restoration Attention (PRA) optimizes semantic decoding by prompt guidance and self-adaptive correction, while Bilevel Prompt Distribution Learning (BPDL) refines the manifold of per-pixel mask embeddings via semantic prototype supervision. Besides, to compensate for the scarcity of PanOoS datasets, we establish two benchmarks: DenseOoS, which features diverse outliers in complex environments, and QuadOoS, captured by a quadruped robot with a panoramic annular lens system. Extensive experiments demonstrate superior performance of POS, with AuPRC improving by 34.25% and FPR95 decreasing by 21.42% on DenseOoS, outperforming state-of-the-art pinhole-OoS methods. Moreover, POS achieves leading closed-set segmentation capabilities. Code and datasets will be available at this https URL. 

**Abstract (ZH)**: 全景影像 Enables 360° 图像的超广视野密集全方位感知。然而，现有的全景语义分割方法无法识别异常值，而针孔相机域外分割模型由于背景杂乱和像素失真，在全景域表现不佳。为解决这些问题，我们引入了一个新的任务——全景域外分割（PanOoS），以实现全景域的域外分割。此外，我们提出了第一个解决方案——POS，通过基于文本的提示分布学习适应全景图像的特性。具体来说，POS 结合了一种脱离纠缠策略，旨在实现 CLIP 的跨域泛化能力。提出的基于提示的恢复注意力（PRA）通过对提示引导和自适应校正优化语义解码，而双层提示分布学习（BPDL）通过语义原型监督细化每个像素掩码嵌入流形。另外，为弥补 PanOoS 数据集的稀缺性，我们建立了两个基准： DenseOoS，其中包含复杂环境中的多种异常值；以及由四足机器人使用全景环形镜头系统拍摄的 QuadOoS。大量实验证明了 POS 的优越性能，在 DenseOoS 上，AuPRC 提高了 34.25%，FPR95 降低了 21.42%，超越了最先进的针孔域外分割方法。此外，POS 达到了领先的数据闭集分割能力。代码和数据集将在此链接提供。 

---
# LiftFeat: 3D Geometry-Aware Local Feature Matching 

**Title (ZH)**: LiftFeat: 3D几何感知局部特征匹配 

**Authors**: Yepeng Liu, Wenpeng Lai, Zhou Zhao, Yuxuan Xiong, Jinchi Zhu, Jun Cheng, Yongchao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03422)  

**Abstract**: Robust and efficient local feature matching plays a crucial role in applications such as SLAM and visual localization for robotics. Despite great progress, it is still very challenging to extract robust and discriminative visual features in scenarios with drastic lighting changes, low texture areas, or repetitive patterns. In this paper, we propose a new lightweight network called \textit{LiftFeat}, which lifts the robustness of raw descriptor by aggregating 3D geometric feature. Specifically, we first adopt a pre-trained monocular depth estimation model to generate pseudo surface normal label, supervising the extraction of 3D geometric feature in terms of predicted surface normal. We then design a 3D geometry-aware feature lifting module to fuse surface normal feature with raw 2D descriptor feature. Integrating such 3D geometric feature enhances the discriminative ability of 2D feature description in extreme conditions. Extensive experimental results on relative pose estimation, homography estimation, and visual localization tasks, demonstrate that our LiftFeat outperforms some lightweight state-of-the-art methods. Code will be released at : this https URL. 

**Abstract (ZH)**: 鲁棒高效的局部特征匹配在机器人SLAM和视觉定位应用中起着关键作用。尽管取得了很大进展，但在剧烈光照变化、低纹理区域或重复模式场景中提取鲁棒性和区分性视觉特征仍极具挑战性。本文提出了一种新的轻量级网络LiftFeat，通过聚合三维几何特征增强原始描述子的鲁棒性。具体而言，我们首先采用预训练的单目深度估计模型生成伪表面法线标签，监督预测表面法线的三维几何特征提取。然后设计了一个三维几何感知特征提升模块，将表面法线特征与原始二维描述子特征融合。结合这种三维几何特征在极端条件下游提升了二维特征描述的区分能力。在相对姿态估计、仿射变换估计和视觉定位任务上的广泛实验结果表明，我们的LiftFeat在某些轻量级的最先进的方法中表现出色。代码将在以下链接发布：this https URL。 

---
# OccCylindrical: Multi-Modal Fusion with Cylindrical Representation for 3D Semantic Occupancy Prediction 

**Title (ZH)**: OccCylindrical: 基于圆柱表示的多模态融合三维语义 occupancy 预测 

**Authors**: Zhenxing Ming, Julie Stephany Berrio, Mao Shan, Yaoqi Huang, Hongyu Lyu, Nguyen Hoang Khoi Tran, Tzu-Yun Tseng, Stewart Worrall  

**Link**: [PDF](https://arxiv.org/pdf/2505.03284)  

**Abstract**: The safe operation of autonomous vehicles (AVs) is highly dependent on their understanding of the surroundings. For this, the task of 3D semantic occupancy prediction divides the space around the sensors into voxels, and labels each voxel with both occupancy and semantic information. Recent perception models have used multisensor fusion to perform this task. However, existing multisensor fusion-based approaches focus mainly on using sensor information in the Cartesian coordinate system. This ignores the distribution of the sensor readings, leading to a loss of fine-grained details and performance degradation. In this paper, we propose OccCylindrical that merges and refines the different modality features under cylindrical coordinates. Our method preserves more fine-grained geometry detail that leads to better performance. Extensive experiments conducted on the nuScenes dataset, including challenging rainy and nighttime scenarios, confirm our approach's effectiveness and state-of-the-art performance. The code will be available at: this https URL 

**Abstract (ZH)**: 自动驾驶车辆的安全运行高度依赖于其对周围环境的理解。为此，3D语义占位预测任务将传感器周围的空间划分成体素，并为每个体素标注兼具占用和语义信息的内容。最近的感知模型利用多传感器融合来执行此任务。然而，现有的基于多传感器融合的方法主要集中在使用笛卡尔坐标系中的传感器信息上。这忽视了传感器读数的分布，导致丢失了细粒度的细节并导致性能下降。本文提出OccCylindrical，在圆柱坐标系下合并和细化不同模态的特征。我们的方法保留了更多的细粒度几何细节，从而提高了性能。通过对nuScenes数据集进行广泛实验，包括挑战性的雨天和夜间场景，证实了我们方法的有效性和最先进的性能。代码将于以下链接提供：this https URL 

---
# Is AI currently capable of identifying wild oysters? A comparison of human annotators against the AI model, ODYSSEE 

**Title (ZH)**: AI当前是否有能力识别野生牡蛎？人类标注员与ODYSSEE模型的比较研究 

**Authors**: Brendan Campbell, Alan Williams, Kleio Baxevani, Alyssa Campbell, Rushabh Dhoke, Rileigh E. Hudock, Xiaomin Lin, Vivek Mange, Bernhard Neuberger, Arjun Suresh, Alhim Vera, Arthur Trembanis, Herbert G. Tanner, Edward Hale  

**Link**: [PDF](https://arxiv.org/pdf/2505.03108)  

**Abstract**: Oysters are ecologically and commercially important species that require frequent monitoring to track population demographics (e.g. abundance, growth, mortality). Current methods of monitoring oyster reefs often require destructive sampling methods and extensive manual effort. Therefore, they are suboptimal for small-scale or sensitive environments. A recent alternative, the ODYSSEE model, was developed to use deep learning techniques to identify live oysters using video or images taken in the field of oyster reefs to assess abundance. The validity of this model in identifying live oysters on a reef was compared to expert and non-expert annotators. In addition, we identified potential sources of prediction error. Although the model can make inferences significantly faster than expert and non-expert annotators (39.6 s, $2.34 \pm 0.61$ h, $4.50 \pm 1.46$ h, respectively), the model overpredicted the number of live oysters, achieving lower accuracy (63\%) in identifying live oysters compared to experts (74\%) and non-experts (75\%) alike. Image quality was an important factor in determining the accuracy of the model and the annotators. Better quality images improved human accuracy and worsened model accuracy. Although ODYSSEE was not sufficiently accurate, we anticipate that future training on higher-quality images, utilizing additional live imagery, and incorporating additional annotation training classes will greatly improve the model's predictive power based on the results of this analysis. Future research should address methods that improve the detection of living vs. dead oysters. 

**Abstract (ZH)**: 牡蛎是具有重要生态和商业价值的物种，需要频繁监测以追踪种群动态（如丰度、生长、死亡率）。目前牡蛎礁的监测方法往往需要破坏性采样和大量的手工努力。因此，这些方法对于小型或敏感环境并不理想。最近一种替代方案，ODYSSEE模型，利用深度学习技术通过现场拍摄的牡蛎礁视频或图像识别活牡蛎，以评估丰度。将该模型在识别礁上活牡蛎方面的有效性与专家和非专家标注员进行了比较，并确定了预测误差的潜在来源。虽然该模型比专家和非专家标注员快得多（分别为39.6秒，2.34±0.61小时，4.50±1.46小时），但它高估了活牡蛎的数量，在识别活牡蛎的准确性上低于专家（74%）和非专家（75%）的73%。图像质量是决定模型和标注员准确性的重要因素。高质量的图像提高了人类准确性和降低了模型准确性。尽管ODYSSEE不够准确，我们预计根据本次分析的结果，通过在更高质量图像上进行进一步训练，利用更多的活体影像，并结合额外的标注训练类别，将显著提高模型的预测能力。未来的研究应解决提高活牡蛎与死牡蛎检测的方法。 

---
# FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios 

**Title (ZH)**: FlexiAct: 向泛化场景下的灵活动作控制迈进 

**Authors**: Shiyi Zhang, Junhao Zhuang, Zhaoyang Zhang, Ying Shan, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03730)  

**Abstract**: Action customization involves generating videos where the subject performs actions dictated by input control signals. Current methods use pose-guided or global motion customization but are limited by strict constraints on spatial structure, such as layout, skeleton, and viewpoint consistency, reducing adaptability across diverse subjects and scenarios. To overcome these limitations, we propose FlexiAct, which transfers actions from a reference video to an arbitrary target image. Unlike existing methods, FlexiAct allows for variations in layout, viewpoint, and skeletal structure between the subject of the reference video and the target image, while maintaining identity consistency. Achieving this requires precise action control, spatial structure adaptation, and consistency preservation. To this end, we introduce RefAdapter, a lightweight image-conditioned adapter that excels in spatial adaptation and consistency preservation, surpassing existing methods in balancing appearance consistency and structural flexibility. Additionally, based on our observations, the denoising process exhibits varying levels of attention to motion (low frequency) and appearance details (high frequency) at different timesteps. So we propose FAE (Frequency-aware Action Extraction), which, unlike existing methods that rely on separate spatial-temporal architectures, directly achieves action extraction during the denoising process. Experiments demonstrate that our method effectively transfers actions to subjects with diverse layouts, skeletons, and viewpoints. We release our code and model weights to support further research at this https URL 

**Abstract (ZH)**: Action定制涉及生成视频，其中主体根据输入控制信号执行动作。现有方法使用姿态引导或全局运动定制，但受限于布局、骨架和视角一致性等严格的空间结构限制，降低了在多样化主体和场景中的适应性。为克服这些限制，我们提出FlexiAct，该方法将参考视频中的动作转移到任意目标图像中。与现有方法不同，FlexiAct 允许参考视频中的主体与目标图像之间的布局、视角和骨架结构发生变化，同时保持身份一致性。实现这一点需要精确的动作控制、空间结构适应和一致性保持。为此，我们引入了RefAdapter，这是一种轻量级的条件图像适配器，在空间适应和一致性保持方面表现出色，优于现有方法在外观一致性与结构灵活性之间的平衡。此外，基于我们的观察，去噪过程在不同时间步长中对运动（低频）和外观细节（高频）的关注程度不同。因此，我们提出了FAE（频域意识动作提取），它与现有方法依赖于独立的空间-时间架构的做法不同，在去噪过程中直接实现动作提取。实验表明，我们的方法能有效将动作转移到具有不同布局、骨架和视角的主体上。我们将在以下链接发布我们的代码和模型权重以支持进一步的研究：这个https URL。 

---
# Revolutionizing Brain Tumor Imaging: Generating Synthetic 3D FA Maps from T1-Weighted MRI using CycleGAN Models 

**Title (ZH)**: 革新脑肿瘤成像：使用CycleGAN模型从T1加权MRI生成合成3D FA图谱 

**Authors**: Xin Du, Francesca M. Cozzi, Rajesh Jena  

**Link**: [PDF](https://arxiv.org/pdf/2505.03662)  

**Abstract**: Fractional anisotropy (FA) and directionally encoded colour (DEC) maps are essential for evaluating white matter integrity and structural connectivity in neuroimaging. However, the spatial misalignment between FA maps and tractography atlases hinders their effective integration into predictive models. To address this issue, we propose a CycleGAN based approach for generating FA maps directly from T1-weighted MRI scans, representing the first application of this technique to both healthy and tumour-affected tissues. Our model, trained on unpaired data, produces high fidelity maps, which have been rigorously evaluated using Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR), demonstrating particularly robust performance in tumour regions. Radiological assessments further underscore the model's potential to enhance clinical workflows by providing an AI-driven alternative that reduces the necessity for additional scans. 

**Abstract (ZH)**: 基于CycleGAN的T1加权MRI扫描转换为FA图的 方法及其在健康和肿瘤组织中的应用 

---
# Real-Time Person Image Synthesis Using a Flow Matching Model 

**Title (ZH)**: 基于流匹配模型的实时人体图像合成 

**Authors**: Jiwoo Jeong, Kirok Kim, Wooju Kim, Nam-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.03562)  

**Abstract**: Pose-Guided Person Image Synthesis (PGPIS) generates realistic person images conditioned on a target pose and a source image. This task plays a key role in various real-world applications, such as sign language video generation, AR/VR, gaming, and live streaming. In these scenarios, real-time PGPIS is critical for providing immediate visual feedback and maintaining user this http URL, achieving real-time performance remains a significant challenge due to the complexity of synthesizing high-fidelity images from diverse and dynamic human poses. Recent diffusion-based methods have shown impressive image quality in PGPIS, but their slow sampling speeds hinder deployment in time-sensitive applications. This latency is particularly problematic in tasks like generating sign language videos during live broadcasts, where rapid image updates are required. Therefore, developing a fast and reliable PGPIS model is a crucial step toward enabling real-time interactive systems. To address this challenge, we propose a generative model based on flow matching (FM). Our approach enables faster, more stable, and more efficient training and sampling. Furthermore, the proposed model supports conditional generation and can operate in latent space, making it especially suitable for real-time PGPIS applications where both speed and quality are critical. We evaluate our proposed method, Real-Time Person Image Synthesis Using a Flow Matching Model (RPFM), on the widely used DeepFashion dataset for PGPIS tasks. Our results show that RPFM achieves near-real-time sampling speeds while maintaining performance comparable to the state-of-the-art models. Our methodology trades off a slight, acceptable decrease in generated-image accuracy for over a twofold increase in generation speed, thereby ensuring real-time performance. 

**Abstract (ZH)**: 基于流匹配的实时人体姿态导向图像合成 (Flow-Matching Guided Real-Time Person Image Synthesis, FM-GRTPIS) 

---
# Generating Synthetic Data via Augmentations for Improved Facial Resemblance in DreamBooth and InstantID 

**Title (ZH)**: 通过增强技术生成合成数据以改善DreamBooth和InstantID中的面部相似性 

**Authors**: Koray Ulusan, Benjamin Kiefer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03557)  

**Abstract**: The personalization of Stable Diffusion for generating professional portraits from amateur photographs is a burgeoning area, with applications in various downstream contexts. This paper investigates the impact of augmentations on improving facial resemblance when using two prominent personalization techniques: DreamBooth and InstantID. Through a series of experiments with diverse subject datasets, we assessed the effectiveness of various augmentation strategies on the generated headshots' fidelity to the original subject. We introduce FaceDistance, a wrapper around FaceNet, to rank the generations based on facial similarity, which aided in our assessment. Ultimately, this research provides insights into the role of augmentations in enhancing facial resemblance in SDXL-generated portraits, informing strategies for their effective deployment in downstream applications. 

**Abstract (ZH)**: 基于增强技术在稳定扩散模型中生成专业肖像的个性化研究：DreamBooth和InstantID的应用分析 

---
# Blending 3D Geometry and Machine Learning for Multi-View Stereopsis 

**Title (ZH)**: 融合3D几何和机器学习的多视图立体视觉 

**Authors**: Vibhas Vats, Md. Alimoor Reza, David Crandall, Soon-heung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.03470)  

**Abstract**: Traditional multi-view stereo (MVS) methods primarily depend on photometric and geometric consistency constraints. In contrast, modern learning-based algorithms often rely on the plane sweep algorithm to infer 3D geometry, applying explicit geometric consistency (GC) checks only as a post-processing step, with no impact on the learning process itself. In this work, we introduce GC MVSNet plus plus, a novel approach that actively enforces geometric consistency of reference view depth maps across multiple source views (multi view) and at various scales (multi scale) during the learning phase (see Fig. 1). This integrated GC check significantly accelerates the learning process by directly penalizing geometrically inconsistent pixels, effectively halving the number of training iterations compared to other MVS methods. Furthermore, we introduce a densely connected cost regularization network with two distinct block designs simple and feature dense optimized to harness dense feature connections for enhanced regularization. Extensive experiments demonstrate that our approach achieves a new state of the art on the DTU and BlendedMVS datasets and secures second place on the Tanks and Temples benchmark. To our knowledge, GC MVSNet plus plus is the first method to enforce multi-view, multi-scale supervised geometric consistency during learning. Our code is available. 

**Abstract (ZH)**: GC MVSNet++：一种在学习阶段主动强制多视图多尺度几何一致性的新方法 

---
# Very High-Resolution Forest Mapping with TanDEM-X InSAR Data and Self-Supervised Learning 

**Title (ZH)**: Very High-Resolution Forest Mapping with TanDEM-X InSAR Data and Self-Supervised Learning 

**Authors**: José-Luis Bueso-Bello, Benjamin Chauvel, Daniel Carcereri, Philipp Posovszky, Pietro Milillo, Jennifer Ruiz, Juan-Carlos Fernández-Diaz, Carolina González, Michele Martone, Ronny Hänsch, Paola Rizzoli  

**Link**: [PDF](https://arxiv.org/pdf/2505.03327)  

**Abstract**: Deep learning models have shown encouraging capabilities for mapping accurately forests at medium resolution with TanDEM-X interferometric SAR data. Such models, as most of current state-of-the-art deep learning techniques in remote sensing, are trained in a fully-supervised way, which requires a large amount of labeled data for training and validation. In this work, our aim is to exploit the high-resolution capabilities of the TanDEM-X mission to map forests at 6 m. The goal is to overcome the intrinsic limitations posed by midresolution products, which affect, e.g., the detection of narrow roads within vegetated areas and the precise delineation of forested regions contours. To cope with the lack of extended reliable reference datasets at such a high resolution, we investigate self-supervised learning techniques for extracting highly informative representations from the input features, followed by a supervised training step with a significantly smaller number of reliable labels. A 1 m resolution forest/non-forest reference map over Pennsylvania, USA, allows for comparing different training approaches for the development of an effective forest mapping framework with limited labeled samples. We select the best-performing approach over this test region and apply it in a real-case forest mapping scenario over the Amazon rainforest, where only very few labeled data at high resolution are available. In this challenging scenario, the proposed self-supervised framework significantly enhances the classification accuracy with respect to fully-supervised methods, trained using the same amount of labeled data, representing an extremely promising starting point for large-scale, very high-resolution forest mapping with TanDEM-X data. 

**Abstract (ZH)**: 利用TanDEM-X干涉雷达SAR数据进行6米分辨率森林制图的深度学习模型研究：自监督学习方法在高分辨率森林制图中的应用 

---
# SD-VSum: A Method and Dataset for Script-Driven Video Summarization 

**Title (ZH)**: SD-VSum: 一种基于脚本的视频摘要方法及数据集 

**Authors**: Manolis Mylonas, Evlampios Apostolidis, Vasileios Mezaris  

**Link**: [PDF](https://arxiv.org/pdf/2505.03319)  

**Abstract**: In this work, we introduce the task of script-driven video summarization, which aims to produce a summary of the full-length video by selecting the parts that are most relevant to a user-provided script outlining the visual content of the desired summary. Following, we extend a recently-introduced large-scale dataset for generic video summarization (VideoXum) by producing natural language descriptions of the different human-annotated summaries that are available per video. In this way we make it compatible with the introduced task, since the available triplets of ``video, summary and summary description'' can be used for training a method that is able to produce different summaries for a given video, driven by the provided script about the content of each summary. Finally, we develop a new network architecture for script-driven video summarization (SD-VSum), that relies on the use of a cross-modal attention mechanism for aligning and fusing information from the visual and text modalities. Our experimental evaluations demonstrate the advanced performance of SD-VSum against state-of-the-art approaches for query-driven and generic (unimodal and multimodal) summarization from the literature, and document its capacity to produce video summaries that are adapted to each user's needs about their content. 

**Abstract (ZH)**: 基于脚本的视频摘要任务及其在网络中实现的研究 

---
# Towards Efficient Benchmarking of Foundation Models in Remote Sensing: A Capabilities Encoding Approach 

**Title (ZH)**: 面向遥感领域的基础模型高效基准测试：一种能力编码方法 

**Authors**: Pierre Adorni, Minh-Tan Pham, Stéphane May, Sébastien Lefèvre  

**Link**: [PDF](https://arxiv.org/pdf/2505.03299)  

**Abstract**: Foundation models constitute a significant advancement in computer vision: after a single, albeit costly, training phase, they can address a wide array of tasks. In the field of Earth observation, over 75 remote sensing vision foundation models have been developed in the past four years. However, none has consistently outperformed the others across all available downstream tasks. To facilitate their comparison, we propose a cost-effective method for predicting a model's performance on multiple downstream tasks without the need for fine-tuning on each one. This method is based on what we call "capabilities encoding." The utility of this novel approach is twofold: we demonstrate its potential to simplify the selection of a foundation model for a given new task, and we employ it to offer a fresh perspective on the existing literature, suggesting avenues for future research. Codes are available at this https URL. 

**Abstract (ZH)**: 基于成本效益的方法预测遥感视觉基础模型在多种下游任务上的性能：一种能力编码的新途径 

---
# DCS-ST for Classification of Breast Cancer Histopathology Images with Limited Annotations 

**Title (ZH)**: DCS-ST在有限标注下的乳腺癌组织病理图像分类中应用 

**Authors**: Liu Suxing, Byungwon Min  

**Link**: [PDF](https://arxiv.org/pdf/2505.03204)  

**Abstract**: Deep learning methods have shown promise in classifying breast cancer histopathology images, but their performance often declines with limited annotated data, a critical challenge in medical imaging due to the high cost and expertise required for annotations. 

**Abstract (ZH)**: 深度学习方法在分类乳腺癌组织病理学图像方面显示出潜力，但在标注数据有限的情况下其性能往往会下降，这在医疗成像领域是一个关键挑战，因为标注数据需要较高的成本和专业知识。 

---
# Lesion-Aware Generative Artificial Intelligence for Virtual Contrast-Enhanced Mammography in Breast Cancer 

**Title (ZH)**: 基于病灶aware的生成人工智能在乳腺癌中虚拟对比增强乳房X线摄影的应用 

**Authors**: Aurora Rofena, Arianna Manchia, Claudia Lucia Piccolo, Bruno Beomonte Zobel, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03018)  

**Abstract**: Contrast-Enhanced Spectral Mammography (CESM) is a dual-energy mammographic technique that improves lesion visibility through the administration of an iodinated contrast agent. It acquires both a low-energy image, comparable to standard mammography, and a high-energy image, which are then combined to produce a dual-energy subtracted image highlighting lesion contrast enhancement. While CESM offers superior diagnostic accuracy compared to standard mammography, its use entails higher radiation exposure and potential side effects associated with the contrast medium. To address these limitations, we propose Seg-CycleGAN, a generative deep learning framework for Virtual Contrast Enhancement in CESM. The model synthesizes high-fidelity dual-energy subtracted images from low-energy images, leveraging lesion segmentation maps to guide the generative process and improve lesion reconstruction. Building upon the standard CycleGAN architecture, Seg-CycleGAN introduces localized loss terms focused on lesion areas, enhancing the synthesis of diagnostically relevant regions. Experiments on the CESM@UCBM dataset demonstrate that Seg-CycleGAN outperforms the baseline in terms of PSNR and SSIM, while maintaining competitive MSE and VIF. Qualitative evaluations further confirm improved lesion fidelity in the generated images. These results suggest that segmentation-aware generative models offer a viable pathway toward contrast-free CESM alternatives. 

**Abstract (ZH)**: 基于分割的CycleGAN在CEMG中的虚拟对比增强 

---
# Generating Narrated Lecture Videos from Slides with Synchronized Highlights 

**Title (ZH)**: 从幻灯片生成同步高亮的讲述视频 

**Authors**: Alexander Holmberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.02966)  

**Abstract**: Turning static slides into engaging video lectures takes considerable time and effort, requiring presenters to record explanations and visually guide their audience through the material. We introduce an end-to-end system designed to automate this process entirely. Given a slide deck, this system synthesizes a video lecture featuring AI-generated narration synchronized precisely with dynamic visual highlights. These highlights automatically draw attention to the specific concept being discussed, much like an effective presenter would. The core technical contribution is a novel highlight alignment module. This module accurately maps spoken phrases to locations on a given slide using diverse strategies (e.g., Levenshtein distance, LLM-based semantic analysis) at selectable granularities (line or word level) and utilizes timestamp-providing Text-to-Speech (TTS) for timing synchronization. We demonstrate the system's effectiveness through a technical evaluation using a manually annotated slide dataset with 1000 samples, finding that LLM-based alignment achieves high location accuracy (F1 > 92%), significantly outperforming simpler methods, especially on complex, math-heavy content. Furthermore, the calculated generation cost averages under $1 per hour of video, offering potential savings of two orders of magnitude compared to conservative estimates of manual production costs. This combination of high accuracy and extremely low cost positions this approach as a practical and scalable tool for transforming static slides into effective, visually-guided video lectures. 

**Abstract (ZH)**: 将静态幻灯片转化为引人入胜的视频讲座需要大量时间和努力，要求讲者录制解释并引导观众观看内容。我们介绍了一个端到端系统，旨在完全自动化这一过程。给定一个幻灯片集，该系统生成一个包含AI生成解说的视频讲座，解说与动态视觉高光精准同步。这些高光自动将注意力集中在正在讨论的具体概念上，类似于有效的讲者。核心技术贡献是新型高光对齐模块。该模块使用多种策略（如Levenshtein距离、基于LLM的语义分析）在可选择的粒度（行或单词级别）上准确地将语音短语映射到给定幻灯片上的位置，并利用提供时间戳的文本到语音（TTS）进行时间同步。通过使用包含1000个样例的手动标注幻灯片数据集进行技术评估，我们发现基于LLM的对齐在位置准确性方面取得高得分（F1 > 92%），显著优于简单方法，特别是在复杂、数学密集的内容方面。此外，计算生成成本平均每小时不到1美元，与保守估计的手动生产成本相比，潜在节省达两个数量级。这种高精度和极低成本的结合使该方法成为将静态幻灯片转化为有效、视觉引导的视频讲座的实用且可扩展工具。 

---
