# SDDiff: Boost Radar Perception via Spatial-Doppler Diffusion 

**Title (ZH)**: SDDiff: 通过空间-多普勒扩散增强雷达感知 

**Authors**: Shengpeng Wang, Xin Luo, Yulong Xie, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16936)  

**Abstract**: Point cloud extraction (PCE) and ego velocity estimation (EVE) are key capabilities gaining attention in 3D radar perception. However, existing work typically treats these two tasks independently, which may neglect the interplay between radar's spatial and Doppler domain features, potentially introducing additional bias. In this paper, we observe an underlying correlation between 3D points and ego velocity, which offers reciprocal benefits for PCE and EVE. To fully unlock such inspiring potential, we take the first step to design a Spatial-Doppler Diffusion (SDDiff) model for simultaneously dense PCE and accurate EVE. To seamlessly tailor it to radar perception, SDDiff improves the conventional latent diffusion process in three major aspects. First, we introduce a representation that embodies both spatial occupancy and Doppler features. Second, we design a directional diffusion with radar priors to streamline the sampling. Third, we propose Iterative Doppler Refinement to enhance the model's adaptability to density variations and ghosting effects. Extensive evaluations show that SDDiff significantly outperforms state-of-the-art baselines by achieving 59% higher in EVE accuracy, 4X greater in valid generation density while boosting PCE effectiveness and reliability. 

**Abstract (ZH)**: 基于空时扩散的点云提取与 ego 速度估计 

---
# Semantic and Feature Guided Uncertainty Quantification of Visual Localization for Autonomous Vehicles 

**Title (ZH)**: 基于语义和特征引导的不确定性量化在自动驾驶车辆视觉定位中的应用 

**Authors**: Qiyuan Wu, Mark Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.15851)  

**Abstract**: The uncertainty quantification of sensor measurements coupled with deep learning networks is crucial for many robotics systems, especially for safety-critical applications such as self-driving cars. This paper develops an uncertainty quantification approach in the context of visual localization for autonomous driving, where locations are selected based on images. Key to our approach is to learn the measurement uncertainty using light-weight sensor error model, which maps both image feature and semantic information to 2-dimensional error distribution. Our approach enables uncertainty estimation conditioned on the specific context of the matched image pair, implicitly capturing other critical, unannotated factors (e.g., city vs highway, dynamic vs static scenes, winter vs summer) in a latent manner. We demonstrate the accuracy of our uncertainty prediction framework using the Ithaca365 dataset, which includes variations in lighting and weather (sunny, night, snowy). Both the uncertainty quantification of the sensor+network is evaluated, along with Bayesian localization filters using unique sensor gating method. Results show that the measurement error does not follow a Gaussian distribution with poor weather and lighting conditions, and is better predicted by our Gaussian Mixture model. 

**Abstract (ZH)**: 传感器测量与深度学习网络结合的不确定性量化对于许多机器人系统至关重要，尤其是在自动驾驶等安全关键应用中。本文在视觉定位的自主驾驶背景下开发了一种不确定性量化方法，其中位置基于图像选择。我们的方法的关键在于使用轻量化传感器误差模型学习测量不确定性，该模型将图像特征和语义信息映射到二维误差分布。我们的方法能够在匹配图像对的具体上下文中进行不确定性估计，隐含地捕捉其他关键但未标注的因素（例如城市 vs 高速公路、动态 vs 静态场景、冬季 vs 夏季）。我们使用Ithaca365数据集展示了我们的不确定性预测框架的准确性，该数据集包含不同光照和天气条件（晴天、夜晚、雪天）。我们评估了传感器+网络的不确定性量化以及使用唯一传感器门控方法的贝叶斯定位滤波器。结果表明，在恶劣天气和光照条件下，测量误差不遵循正态分布，并且我们的混合高斯模型能够更好地预测这些误差。 

---
# Part$^{2}$GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting 

**Title (ZH)**: Part$^{2}$GS: Part-aware Modeling of Articulated Objects Using 3D Gaussian Splatting 

**Authors**: Tianjiao Yu, Vedant Shah, Muntasir Wahed, Ying Shen, Kiet A. Nguyen, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17212)  

**Abstract**: Articulated objects are common in the real world, yet modeling their structure and motion remains a challenging task for 3D reconstruction methods. In this work, we introduce Part$^{2}$GS, a novel framework for modeling articulated digital twins of multi-part objects with high-fidelity geometry and physically consistent articulation. Part$^{2}$GS leverages a part-aware 3D Gaussian representation that encodes articulated components with learnable attributes, enabling structured, disentangled transformations that preserve high-fidelity geometry. To ensure physically consistent motion, we propose a motion-aware canonical representation guided by physics-based constraints, including contact enforcement, velocity consistency, and vector-field alignment. Furthermore, we introduce a field of repel points to prevent part collisions and maintain stable articulation paths, significantly improving motion coherence over baselines. Extensive evaluations on both synthetic and real-world datasets show that Part$^{2}$GS consistently outperforms state-of-the-art methods by up to 10$\times$ in Chamfer Distance for movable parts. 

**Abstract (ZH)**: articulated对象在现实世界中非常普遍，但对其结构和运动建模仍然是3D重建方法的挑战性任务。本文引入了Part$^{2}$GS，一种新型框架，用于建模多部件对象的高保真几何和物理一致的 articulated 数字孪生。Part$^{2}$GS 利用一种部件感知的3D高斯表示，编码了可学习属性的articulated组件，从而支持结构化、独立的变换，保持高保真几何。为了确保物理一致的运动，我们提出了一种由基于物理约束引导的运动感知规范表示，包括接触约束、速度一致性以及矢量场对齐。此外，我们引入了一个排斥点场以防止部件碰撞并保持稳定的articulation路径，显著提高了运动连贯性。在合成和真实世界数据集上的广泛评估显示，Part$^{2}$GS 在可移动部分上的均方差距离上比最新方法最多提高10倍。 

---
# RGBTrack: Fast, Robust Depth-Free 6D Pose Estimation and Tracking 

**Title (ZH)**: RGBTrack：快速可靠的无深度6D姿态估计与跟踪 

**Authors**: Teng Guo, Jingjin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17119)  

**Abstract**: We introduce a robust framework, RGBTrack, for real-time 6D pose estimation and tracking that operates solely on RGB data, thereby eliminating the need for depth input for such dynamic and precise object pose tracking tasks. Building on the FoundationPose architecture, we devise a novel binary search strategy combined with a render-and-compare mechanism to efficiently infer depth and generate robust pose hypotheses from true-scale CAD models. To maintain stable tracking in dynamic scenarios, including rapid movements and occlusions, RGBTrack integrates state-of-the-art 2D object tracking (XMem) with a Kalman filter and a state machine for proactive object pose recovery. In addition, RGBTrack's scale recovery module dynamically adapts CAD models of unknown scale using an initial depth estimate, enabling seamless integration with modern generative reconstruction techniques. Extensive evaluations on benchmark datasets demonstrate that RGBTrack's novel depth-free approach achieves competitive accuracy and real-time performance, making it a promising practical solution candidate for application areas including robotics, augmented reality, and computer vision.
The source code for our implementation will be made publicly available at this https URL. 

**Abstract (ZH)**: 我们提出了一种鲁棒框架RGBTrack，用于仅基于RGB数据进行实时6D姿态估计与跟踪，从而消除动态且精确物体姿态跟踪任务中对深度输入的需求。该框架基于FoundationPose架构，设计了一种新颖的二分搜索策略结合渲染与比对机制，以高效地从真实尺度的CAD模型中推断深度并生成稳健的姿态假设。为了在包括快速运动和遮挡在内的动态场景中保持稳定的跟踪，RGBTrack将最先进的2D物体跟踪（XMem）与卡尔曼滤波器和状态机集成，以实现前瞻性的物体姿态恢复。此外，RGBTrack的尺度恢复模块能够根据初始深度估计动态适应未知尺度的CAD模型，使其能够无缝集成现代生成重建技术。在基准数据集上的广泛评估表明，RGBTrack的新颖无深度方法在精度和实时性能方面具有竞争力，使其成为机器人技术、增强现实和计算机视觉等领域具有潜力的实用解决方案候选者。源代码将在该网址公开发布：https://example.com。 

---
# Camera Calibration via Circular Patterns: A Comprehensive Framework with Measurement Uncertainty and Unbiased Projection Model 

**Title (ZH)**: 基于圆特征的相机标定：包含测量不确定性及无偏投影模型的综合框架 

**Authors**: Chaehyeon Song, Dongjae Lee, Jongwoo Lim, Ayoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.16842)  

**Abstract**: Camera calibration using planar targets has been widely favored, and two types of control points have been mainly considered as measurements: the corners of the checkerboard and the centroid of circles. Since a centroid is derived from numerous pixels, the circular pattern provides more precise measurements than the checkerboard. However, the existing projection model of circle centroids is biased under lens distortion, resulting in low performance. To surmount this limitation, we propose an unbiased projection model of the circular pattern and demonstrate its superior accuracy compared to the checkerboard. Complementing this, we introduce uncertainty into circular patterns to enhance calibration robustness and completeness. Defining centroid uncertainty improves the performance of calibration components, including pattern detection, optimization, and evaluation metrics. We also provide guidelines for performing good camera calibration based on the evaluation metric. The core concept of this approach is to model the boundary points of a two-dimensional shape as a Markov random field, considering its connectivity. The shape distribution is propagated to the centroid uncertainty through an appropriate shape representation based on the Green theorem. Consequently, the resulting framework achieves marked gains in calibration accuracy and robustness. The complete source code and demonstration video are available at this https URL. 

**Abstract (ZH)**: 使用平面靶标进行相机校准已被广泛青睐，主要考虑的两种测量控制点是棋盘格的角点和圆的质心。由于质心是从众多像素中得出的，因此圆的模式提供了比棋盘格更精确的测量。然而现有的圆心投影模型在镜头畸变下存在偏差，导致性能较低。为了克服这一限制，我们提出了一个无偏的圆的投影模型，并证明其在标定精度上优于棋盘格。此外，我们引入圆中的不确定性来增强标定的稳健性和完整性。定义质心不确定性可以提高包括图案检测、优化和评估指标在内的标定组件的性能。我们还基于评估指标提供了进行良好相机校准的准则。该方法的核心概念是将二维形状的边界点建模为马尔可夫随机场，考虑其连通性。形状分布通过基于格林定理的适当形状表示传播到质心不确定性中。因此，该框架实现了显著的标定精度和稳健性的提升。完整的源代码和演示视频可在此处访问：this https URL。 

---
# Dense 3D Displacement Estimation for Landslide Monitoring via Fusion of TLS Point Clouds and Embedded RGB Images 

**Title (ZH)**: 基于TLS点云和嵌入RGB图像融合的滑坡监测密集3D位移估计 

**Authors**: Zhaoyi Wang, Jemil Avers Butt, Shengyu Huang, Tomislav Medic, Andreas Wieser  

**Link**: [PDF](https://arxiv.org/pdf/2506.16265)  

**Abstract**: Landslide monitoring is essential for understanding geohazards and mitigating associated risks. However, existing point cloud-based methods typically rely on either geometric or radiometric information and often yield sparse or non-3D displacement estimates. In this paper, we propose a hierarchical partition-based coarse-to-fine approach that fuses 3D point clouds and co-registered RGB images to estimate dense 3D displacement vector fields. We construct patch-level matches using both 3D geometry and 2D image features. These matches are refined via geometric consistency checks, followed by rigid transformation estimation per match. Experimental results on two real-world landslide datasets demonstrate that our method produces 3D displacement estimates with high spatial coverage (79% and 97%) and high accuracy. Deviations in displacement magnitude with respect to external measurements (total station or GNSS observations) are 0.15 m and 0.25 m on the two datasets, respectively, and only 0.07 m and 0.20 m compared to manually derived references. These values are below the average scan resolutions (0.08 m and 0.30 m). Our method outperforms the state-of-the-art method F2S3 in spatial coverage while maintaining comparable accuracy. Our approach offers a practical and adaptable solution for TLS-based landslide monitoring and is extensible to other types of point clouds and monitoring tasks. Our example data and source code are publicly available at this https URL. 

**Abstract (ZH)**: 基于层次分区的粗细结合方法：融合三维点云和共注册RGB图像进行密集三维位移场估计 

---
# EndoMUST: Monocular Depth Estimation for Robotic Endoscopy via End-to-end Multi-step Self-supervised Training 

**Title (ZH)**: EndoMUST：基于端到端多步自监督训练的单目深度估计在机器人内窥镜中的应用 

**Authors**: Liangjing Shao, Linxin Bai, Chenkang Du, Xinrong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16017)  

**Abstract**: Monocular depth estimation and ego-motion estimation are significant tasks for scene perception and navigation in stable, accurate and efficient robot-assisted endoscopy. To tackle lighting variations and sparse textures in endoscopic scenes, multiple techniques including optical flow, appearance flow and intrinsic image decomposition have been introduced into the existing methods. However, the effective training strategy for multiple modules are still critical to deal with both illumination issues and information interference for self-supervised depth estimation in endoscopy. Therefore, a novel framework with multistep efficient finetuning is proposed in this work. In each epoch of end-to-end training, the process is divided into three steps, including optical flow registration, multiscale image decomposition and multiple transformation alignments. At each step, only the related networks are trained without interference of irrelevant information. Based on parameter-efficient finetuning on the foundation model, the proposed method achieves state-of-the-art performance on self-supervised depth estimation on SCARED dataset and zero-shot depth estimation on Hamlyn dataset, with 4\%$\sim$10\% lower error. The evaluation code of this work has been published on this https URL. 

**Abstract (ZH)**: 单目深度估计和自我运动估计是稳定、准确、高效机器人辅助内窥镜场景感知与导航的重要任务。为此，在处理内窥镜场景中的光照变化和稀疏纹理时，已将光流、外观流和固有图像分解等多种技术引入现有方法。然而，如何有效训练多个模块以解决照明问题和信息干扰对于自监督深度估计仍至关重要。因此，本文提出了一种具有多步骤高效微调的新框架。在端到端训练的每个周期中，过程分为三步：光流注册、多尺度图像分解和多种变换对齐。在每一步中，只训练相关的网络以避免无关信息的干扰。基于对基础模型的参数高效微调，所提出的方法在SCARED数据集上的自监督深度估计和Hamlyn数据集上的零样本深度估计中均取得了最佳性能，误差降低了4%~10%。本文的评估代码已发布在https://...。 

---
# Adversarial Attacks and Detection in Visual Place Recognition for Safer Robot Navigation 

**Title (ZH)**: 视觉地点识别中的对抗性攻击与检测以实现更安全的机器人导航 

**Authors**: Connor Malone, Owen Claxton, Iman Shames, Michael Milford  

**Link**: [PDF](https://arxiv.org/pdf/2506.15988)  

**Abstract**: Stand-alone Visual Place Recognition (VPR) systems have little defence against a well-designed adversarial attack, which can lead to disastrous consequences when deployed for robot navigation. This paper extensively analyzes the effect of four adversarial attacks common in other perception tasks and four novel VPR-specific attacks on VPR localization performance. We then propose how to close the loop between VPR, an Adversarial Attack Detector (AAD), and active navigation decisions by demonstrating the performance benefit of simulated AADs in a novel experiment paradigm -- which we detail for the robotics community to use as a system framework. In the proposed experiment paradigm, we see the addition of AADs across a range of detection accuracies can improve performance over baseline; demonstrating a significant improvement -- such as a ~50% reduction in the mean along-track localization error -- can be achieved with True Positive and False Positive detection rates of only 75% and up to 25% respectively. We examine a variety of metrics including: Along-Track Error, Percentage of Time Attacked, Percentage of Time in an `Unsafe' State, and Longest Continuous Time Under Attack. Expanding further on these results, we provide the first investigation into the efficacy of the Fast Gradient Sign Method (FGSM) adversarial attack for VPR. The analysis in this work highlights the need for AADs in real-world systems for trustworthy navigation, and informs quantitative requirements for system design. 

**Abstract (ZH)**: 独立视觉位置识别(VPR)系统对精心设计的对抗攻击缺乏防御能力，这可能导致灾难性的后果，特别是在机器人导航中部署时。本文广泛分析了四种常见于其他感知任务的对抗攻击和四种新型VPR特定攻击对VPR定位性能的影响。随后，我们提出了一种通过展示模拟对抗攻击检测器(AAD)在一种新型实验范式中的性能优势（我们对此范式进行了详细说明，供机器人社区作为系统框架使用）来闭环VPR和主动导航决策的方法。在所提出的实验范式中，我们发现不同检测准确度的AADs的添加可以提高性能；仅通过True Positive和False Positive检测率分别为75%和25%即可实现显著的性能提升，如沿航向定位误差降低约50%。我们考察了包含以下指标在内的多种指标：沿航向误差、受攻击时间百分比、不可用状态时间百分比以及连续受攻击时间最长。进一步扩展这些结果，我们首次探讨了快速梯度符号方法(FGSM)对抗攻击在VPR中的效用。本文的分析强调了在实际系统中使用对抗攻击检测器以实现值得信赖的导航的必要性，并提供了系统设计的量化要求。 

---
# AI's Blind Spots: Geographic Knowledge and Diversity Deficit in Generated Urban Scenario 

**Title (ZH)**: AI的盲点：生成的城市场景中的地理知识匮乏与多元化缺失 

**Authors**: Ciro Beneduce, Massimiliano Luca, Bruno Lepri  

**Link**: [PDF](https://arxiv.org/pdf/2506.16898)  

**Abstract**: Image generation models are revolutionizing many domains, and urban analysis and design is no exception. While such models are widely adopted, there is a limited literature exploring their geographic knowledge, along with the biases they embed. In this work, we generated 150 synthetic images for each state in the USA and related capitals using FLUX 1 and Stable Diffusion 3.5, two state-of-the-art models for image generation. We embed each image using DINO-v2 ViT-S/14 and the Fréchet Inception Distances to measure the similarity between the generated images. We found that while these models have implicitly learned aspects of USA geography, if we prompt the models to generate an image for "United States" instead of specific cities or states, the models exhibit a strong representative bias toward metropolis-like areas, excluding rural states and smaller cities. {\color{black} In addition, we found that models systematically exhibit some entity-disambiguation issues with European-sounding names like Frankfort or Devon. 

**Abstract (ZH)**: 图像生成模型正在革新许多领域，城市分析与设计也不例外。尽管这些模型被广泛采用，但有关它们嵌入的地理知识及其偏见的研究仍然有限。在本工作中，我们使用FLUX 1和Stable Diffusion 3.5两种最先进的图像生成模型，分别为美国各州和首府生成了150张合成图像。我们使用DINO-v2 ViT-S/14和弗雷切尔-Incendtime 距离来衡量生成图像之间的相似度。研究发现，虽然这些模型在一定程度上隐式学习了美国的地理特征，但若将提示语从具体的城镇或州改为“美国”，模型倾向于生成类似大都市区的图像，而忽视了农村州和较小的城镇。此外，我们还发现模型在处理具有欧洲风格名称的实体时存在一定程度的语义歧义问题，例如Frankfort或Devon。 

---
# Facial Landmark Visualization and Emotion Recognition Through Neural Networks 

**Title (ZH)**: 通过神经网络实现面部特征点可视化与情绪识别 

**Authors**: Israel Juárez-Jiménez, Tiffany Guadalupe Martínez Paredes, Jesús García-Ramírez, Eric Ramos Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17191)  

**Abstract**: Emotion recognition from facial images is a crucial task in human-computer interaction, enabling machines to learn human emotions through facial expressions. Previous studies have shown that facial images can be used to train deep learning models; however, most of these studies do not include a through dataset analysis. Visualizing facial landmarks can be challenging when extracting meaningful dataset insights; to address this issue, we propose facial landmark box plots, a visualization technique designed to identify outliers in facial datasets. Additionally, we compare two sets of facial landmark features: (i) the landmarks' absolute positions and (ii) their displacements from a neutral expression to the peak of an emotional expression. Our results indicate that a neural network achieves better performance than a random forest classifier. 

**Abstract (ZH)**: 基于面部图像的情感识别是人机交互中的关键任务，使机器能够通过面部表情学习人类情绪。虽然以往研究已证明可以利用面部图像训练深度学习模型，但大部分研究并未进行详尽的数据集分析。在提取有意义的数据集洞察时，可视化面部特征点可能会遇到挑战；为解决这一问题，我们提出了一种面部特征点箱线图可视化技术，旨在识别面部数据集中的异常值。此外，我们比较了两种面部特征点特征集：（i）特征点的绝对位置，以及（ii）它们从中性表情到情感表情峰值的位移。结果显示，神经网络的性能优于随机森林分类器。 

---
# Proportional Sensitivity in Generative Adversarial Network (GAN)-Augmented Brain Tumor Classification Using Convolutional Neural Network 

**Title (ZH)**: 基于生成对抗网络（GAN）增强的卷积神经网络在脑肿瘤分类中的比例敏感性 

**Authors**: Mahin Montasir Afif, Abdullah Al Noman, K. M. Tahsin Kabir, Md. Mortuza Ahmmed, Md. Mostafizur Rahman, Mufti Mahmud, Md. Ashraful Babu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17165)  

**Abstract**: Generative Adversarial Networks (GAN) have shown potential in expanding limited medical imaging datasets. This study explores how different ratios of GAN-generated and real brain tumor MRI images impact the performance of a CNN in classifying healthy vs. tumorous scans. A DCGAN was used to create synthetic images which were mixed with real ones at various ratios to train a custom CNN. The CNN was then evaluated on a separate real-world test set. Our results indicate that the model maintains high sensitivity and precision in tumor classification, even when trained predominantly on synthetic data. When only a small portion of GAN data was added, such as 900 real images and 100 GAN images, the model achieved excellent performance, with test accuracy reaching 95.2%, and precision, recall, and F1-score all exceeding 95%. However, as the proportion of GAN images increased further, performance gradually declined. This study suggests that while GANs are useful for augmenting limited datasets especially when real data is scarce, too much synthetic data can introduce artifacts that affect the model's ability to generalize to real world cases. 

**Abstract (ZH)**: 生成对抗网络（GAN）在扩展有限医疗影像数据集方面的潜力已得到展现。本研究探讨了不同比例的GAN生成和真实脑肿瘤MRI图像对CNN分类健康与肿瘤扫描性能的影响。使用DCGAN创建合成图像，并以不同比例与真实图像混合训练自定义CNN。然后对该CNN在独立的现实世界测试集上进行评估。结果显示，即使主要使用合成数据训练，模型在肿瘤分类方面的灵敏度和精确度仍保持在较高水平。当仅添加少量GAN数据，例如900张真实图像和100张GAN图像时，模型达到了 excellent 的性能，测试准确率高达95.2%，并且精确度、召回率和F1分数均超过95%。然而，随着GAN图像比例进一步增加，性能逐渐下降。本研究表明，虽然GAN对于补充稀缺的真实数据集特别有用，但过多的合成数据可能会引入影响模型泛化能力的伪影。 

---
# Loupe: A Generalizable and Adaptive Framework for Image Forgery Detection 

**Title (ZH)**: Loupe: 一种通用且适应性强的图像伪造检测框架 

**Authors**: Yuchu Jiang, Jiaming Chu, Jian Zhao, Xin Zhang, Xu Yang, Lei Jin, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16819)  

**Abstract**: The proliferation of generative models has raised serious concerns about visual content forgery. Existing deepfake detection methods primarily target either image-level classification or pixel-wise localization. While some achieve high accuracy, they often suffer from limited generalization across manipulation types or rely on complex architectures. In this paper, we propose Loupe, a lightweight yet effective framework for joint deepfake detection and localization. Loupe integrates a patch-aware classifier and a segmentation module with conditional queries, allowing simultaneous global authenticity classification and fine-grained mask prediction. To enhance robustness against distribution shifts of test set, Loupe introduces a pseudo-label-guided test-time adaptation mechanism by leveraging patch-level predictions to supervise the segmentation head. Extensive experiments on the DDL dataset demonstrate that Loupe achieves state-of-the-art performance, securing the first place in the IJCAI 2025 Deepfake Detection and Localization Challenge with an overall score of 0.846. Our results validate the effectiveness of the proposed patch-level fusion and conditional query design in improving both classification accuracy and spatial localization under diverse forgery patterns. The code is available at this https URL. 

**Abstract (ZH)**: 生成模型的 proliferaton 提高了视觉内容伪造的严重性。现有的深度假信息检测方法主要针对图像级分类或像素级定位。尽管一些方法达到了高精度，但它们往往在跨伪造类型的一般化方面表现有限，或者依赖于复杂的架构。在本文中，我们提出了一种轻量而有效的框架 Loupe，用于联合假信息检测与定位。Loupe 结合了patch-aware分类器和具有条件查询的分割模块，允许同时进行全局真实性和分类和精细粒度的掩码预测。为了增强对测试集分布偏移的鲁棒性，Loupe 引入了一种基于 patch-level 预测的伪标签指导测试时适应机制，以监督分割头部。在 DDL 数据集上的广泛实验显示出，Loupe 达到了最先进的性能，在IJCAI 2025 深度假信息检测与定位挑战中以综合得分为0.846获得第一名。我们的结果验证了在多变的伪造模式下，提出的 patch-level 融合和条件查询设计在提高分类准确性和空间定位方面的有效性。代码可在如下链接获取。 

---
# Hybrid Attention Network for Accurate Breast Tumor Segmentation in Ultrasound Images 

**Title (ZH)**: 超声图像中乳腺肿瘤分割的混合注意力网络 

**Authors**: Muhammad Azeem Aslam, Asim Naveed, Nisar Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.16592)  

**Abstract**: Breast ultrasound imaging is a valuable tool for early breast cancer detection, but automated tumor segmentation is challenging due to inherent noise, variations in scale of lesions, and fuzzy boundaries. To address these challenges, we propose a novel hybrid attention-based network for lesion segmentation. Our proposed architecture integrates a pre-trained DenseNet121 in the encoder part for robust feature extraction with a multi-branch attention-enhanced decoder tailored for breast ultrasound images. The bottleneck incorporates Global Spatial Attention (GSA), Position Encoding (PE), and Scaled Dot-Product Attention (SDPA) to learn global context, spatial relationships, and relative positional features. The Spatial Feature Enhancement Block (SFEB) is embedded at skip connections to refine and enhance spatial features, enabling the network to focus more effectively on tumor regions. A hybrid loss function combining Binary Cross-Entropy (BCE) and Jaccard Index loss optimizes both pixel-level accuracy and region-level overlap metrics, enhancing robustness to class imbalance and irregular tumor shapes. Experiments on public datasets demonstrate that our method outperforms existing approaches, highlighting its potential to assist radiologists in early and accurate breast cancer diagnosis. 

**Abstract (ZH)**: 基于注意力机制的混合网络在乳腺超声图像中肿块分割中的应用：一种克服固有噪声、病变尺度变化和模糊边界挑战的新方法 

---
# Spatially-Aware Evaluation of Segmentation Uncertainty 

**Title (ZH)**: 空间感知分割不确定性评估 

**Authors**: Tal Zeevi, Eléonore V. Lieffrig, Lawrence H. Staib, John A. Onofrey  

**Link**: [PDF](https://arxiv.org/pdf/2506.16589)  

**Abstract**: Uncertainty maps highlight unreliable regions in segmentation predictions. However, most uncertainty evaluation metrics treat voxels independently, ignoring spatial context and anatomical structure. As a result, they may assign identical scores to qualitatively distinct patterns (e.g., scattered vs. boundary-aligned uncertainty). We propose three spatially aware metrics that incorporate structural and boundary information and conduct a thorough validation on medical imaging data from the prostate zonal segmentation challenge within the Medical Segmentation Decathlon. Our results demonstrate improved alignment with clinically important factors and better discrimination between meaningful and spurious uncertainty patterns. 

**Abstract (ZH)**: 空间感知不确定性度量突显分割预测中的不可靠区域，并考虑结构和边界信息， medical imaging data from the prostate zonal segmentation challenge within the Medical Segmentation Decathlon上的验证表明，这些度量能更好地与临床重要因素对齐，并能更有效地区分有意义和虚假的不确定性模式。 

---
# From Semantic To Instance: A Semi-Self-Supervised Learning Approach 

**Title (ZH)**: 从语义到实例：一种半自监督学习方法 

**Authors**: Keyhan Najafian, Farhad Maleki, Lingling Jin, Ian Stavness  

**Link**: [PDF](https://arxiv.org/pdf/2506.16563)  

**Abstract**: Instance segmentation is essential for applications such as automated monitoring of plant health, growth, and yield. However, extensive effort is required to create large-scale datasets with pixel-level annotations of each object instance for developing instance segmentation models that restrict the use of deep learning in these areas. This challenge is more significant in images with densely packed, self-occluded objects, which are common in agriculture. To address this challenge, we propose a semi-self-supervised learning approach that requires minimal manual annotation to develop a high-performing instance segmentation model. We design GLMask, an image-mask representation for the model to focus on shape, texture, and pattern while minimizing its dependence on color features. We develop a pipeline to generate semantic segmentation and then transform it into instance-level segmentation. The proposed approach substantially outperforms the conventional instance segmentation models, establishing a state-of-the-art wheat head instance segmentation model with mAP@50 of 98.5%. Additionally, we assessed the proposed methodology on the general-purpose Microsoft COCO dataset, achieving a significant performance improvement of over 12.6% mAP@50. This highlights that the utility of our proposed approach extends beyond precision agriculture and applies to other domains, specifically those with similar data characteristics. 

**Abstract (ZH)**: 基于实例的分割对于植物健康、生长和产量的自动化监控等应用至关重要。然而，为了开发实例分割模型，需要大量带有像素级标注的实例数据集，这限制了深度学习在这些领域的应用。这一挑战在密集堆叠且相互遮挡的物体图像中尤为显著，而这类图像在农业中非常常见。为应对这一挑战，我们提出了一种半自监督学习方法，该方法只需少量手动标注即可训练高性能的实例分割模型。我们设计了GLMask，该模型的图像-掩膜表示专注于形状、纹理和模式，同时减少对颜色特征的依赖。我们开发了一个流水线，用于生成语义分割，然后将其转换为实例级分割。所提出的方法显著优于传统的实例分割模型，建立了基于mAP@50为98.5%的高性能小麦穗实例分割模型。此外，我们在通用的Microsoft COCO数据集上评估了所提出的方法，获得了超过12.6%的mAP@50性能提升。这表明我们提出的方法不仅适用于精确农业，还可以应用于其他具有类似数据特征的领域。 

---
# Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details 

**Title (ZH)**: 混沌3D 2.5：迈向极致细节的高保真3D资产生成 

**Authors**: Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, Sheng Zhang, Xin Huang, Di Luo, Fan Yang, Fang Yang, Lifu Wang, Sicong Liu, Yixuan Tang, Yulin Cai, Zebin He, Tian Liu, Yuhong Liu, Jie Jiang, Linus, Jingwei Huang, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.16504)  

**Abstract**: In this report, we present Hunyuan3D 2.5, a robust suite of 3D diffusion models aimed at generating high-fidelity and detailed textured 3D assets. Hunyuan3D 2.5 follows two-stages pipeline of its previous version Hunyuan3D 2.0, while demonstrating substantial advancements in both shape and texture generation. In terms of shape generation, we introduce a new shape foundation model -- LATTICE, which is trained with scaled high-quality datasets, model-size, and compute. Our largest model reaches 10B parameters and generates sharp and detailed 3D shape with precise image-3D following while keeping mesh surface clean and smooth, significantly closing the gap between generated and handcrafted 3D shapes. In terms of texture generation, it is upgraded with phyiscal-based rendering (PBR) via a novel multi-view architecture extended from Hunyuan3D 2.0 Paint model. Our extensive evaluation shows that Hunyuan3D 2.5 significantly outperforms previous methods in both shape and end-to-end texture generation. 

**Abstract (ZH)**: 本报告介绍了Hunyuan3D 2.5，这是一个稳健的3D扩散模型套件，旨在生成高保真度和详细纹理的3D资产。Hunyuan3D 2.5沿用了其前一个版本Hunyuan3D 2.0的两阶段管道，同时在形状和纹理生成方面取得了显著进步。在形状生成方面，我们引入了一个新的形状基础模型——LATTICE，该模型使用放缩后的高质量数据集、模型大小和计算资源进行训练。我们的最大模型达到10B参数，并生成了清晰且细节丰富的3D形状，保真度高，同时保持了网格表面的清洁和平滑，显著缩小了生成与手工制作的3D形状之间的差距。在纹理生成方面，通过从Hunyuan3D 2.0 Paint模型扩展出的新颖多视角架构引入基于物理的渲染（PBR）。广泛的评估表明，Hunyuan3D 2.5在形状和端到端纹理生成方面显著优于以前的方法。 

---
# Spotting tell-tale visual artifacts in face swapping videos: strengths and pitfalls of CNN detectors 

**Title (ZH)**: 检测人脸换脸视频中的 tell-tale 视觉 artefacts：CNN 检测器的优势与局限 

**Authors**: Riccardo Ziglio, Cecilia Pasquini, Silvio Ranise  

**Link**: [PDF](https://arxiv.org/pdf/2506.16497)  

**Abstract**: Face swapping manipulations in video streams represents an increasing threat in remote video communications, due to advances
in automated and real-time tools. Recent literature proposes to characterize and exploit visual artifacts introduced in video frames
by swapping algorithms when dealing with challenging physical scenes, such as face occlusions. This paper investigates the
effectiveness of this approach by benchmarking CNN-based data-driven models on two data corpora (including a newly collected
one) and analyzing generalization capabilities with respect to different acquisition sources and swapping algorithms. The results
confirm excellent performance of general-purpose CNN architectures when operating within the same data source, but a significant
difficulty in robustly characterizing occlusion-based visual cues across datasets. This highlights the need for specialized detection
strategies to deal with such artifacts. 

**Abstract (ZH)**: 视频流中的面部置换操作对远程视频通信构成了日益严重的威胁，由于自动化和实时工具的发展。近期文献提出，在处理复杂的物理场景（如面部遮挡）时，可以通过表征和利用面部置换算法在视频帧中引入的视觉artifact来应对这一挑战。本文通过在两个数据集（包括一个新收集的数据集）上基准测试基于CNN的数据驱动模型，并分析其在不同采集源和置换算法下的泛化能力，来探讨该方法的有效性。结果表明，通用的CNN架构在同一数据源下表现出色，但在跨数据集表征基于遮挡的视觉线索方面面临显著困难。这强调了需要专门的检测策略来应对这些artifact。 

---
# Efficient Transformations in Deep Learning Convolutional Neural Networks 

**Title (ZH)**: 深度学习卷积神经网络中的高效变换 

**Authors**: Berk Yilmaz, Daniel Fidel Harvey, Prajit Dhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.16418)  

**Abstract**: This study investigates the integration of signal processing transformations -- Fast Fourier Transform (FFT), Walsh-Hadamard Transform (WHT), and Discrete Cosine Transform (DCT) -- within the ResNet50 convolutional neural network (CNN) model for image classification. The primary objective is to assess the trade-offs between computational efficiency, energy consumption, and classification accuracy during training and inference. Using the CIFAR-100 dataset (100 classes, 60,000 images), experiments demonstrated that incorporating WHT significantly reduced energy consumption while improving accuracy. Specifically, a baseline ResNet50 model achieved a testing accuracy of 66%, consuming an average of 25,606 kJ per model. In contrast, a modified ResNet50 incorporating WHT in the early convolutional layers achieved 74% accuracy, and an enhanced version with WHT applied to both early and late layers achieved 79% accuracy, with an average energy consumption of only 39 kJ per model. These results demonstrate the potential of WHT as a highly efficient and effective approach for energy-constrained CNN applications. 

**Abstract (ZH)**: 本研究探讨了在ResNet50卷积神经网络模型中整合快速傅里叶变换（FFT）、沃尔什-哈达玛变换（WHT）和离散余弦变换（DCT）对图像分类的影响，主要评估训练和推理过程中计算效率、能耗与分类准确性之间的权衡。通过使用CIFAR-100数据集（100个类别，60,000张图像）的实验，研究表明引入WHT显著降低了能耗并提高了准确性。具体而言，基线ResNet50模型在测试集上的准确率为66%，平均每模型能耗为25,606 kJ。相比之下，一个在早期卷积层引入WHT的修改版ResNet50模型达到了74%的准确率，而一个在早期和晚期层均应用WHT的增强版ResNet50模型则达到了79%的准确率，平均每模型能耗仅为39 kJ。这些结果表明WHT作为能耗受限的卷积神经网络应用的高效而有效的方案具有巨大潜力。 

---
# Robustness Evaluation of OCR-based Visual Document Understanding under Multi-Modal Adversarial Attacks 

**Title (ZH)**: 基于多模态对抗攻击的OCR驱动视觉文档理解鲁棒性评估 

**Authors**: Dong Nguyen Tien, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2506.16407)  

**Abstract**: Visual Document Understanding (VDU) systems have achieved strong performance in information extraction by integrating textual, layout, and visual signals. However, their robustness under realistic adversarial perturbations remains insufficiently explored. We introduce the first unified framework for generating and evaluating multi-modal adversarial attacks on OCR-based VDU models. Our method covers six gradient-based layout attack scenarios, incorporating manipulations of OCR bounding boxes, pixels, and texts across both word and line granularities, with constraints on layout perturbation budget (e.g., IoU >= 0.6) to preserve plausibility.
Experimental results across four datasets (FUNSD, CORD, SROIE, DocVQA) and six model families demonstrate that line-level attacks and compound perturbations (BBox + Pixel + Text) yield the most severe performance degradation. Projected Gradient Descent (PGD)-based BBox perturbations outperform random-shift baselines in all investigated models. Ablation studies further validate the impact of layout budget, text modification, and adversarial transferability. 

**Abstract (ZH)**: 基于OCR的视觉文档理解（VDU）模型的多模态 adversarial 攻击生成与评估统一框架 

---
# CLIP-MG: Guiding Semantic Attention with Skeletal Pose Features and RGB Data for Micro-Gesture Recognition on the iMiGUE Dataset 

**Title (ZH)**: CLIP-MG：基于骨架姿态特征和RGB数据的语义注意力引导微型手势识别方法 

**Authors**: Santosh Patapati, Trisanth Srinivasan, Amith Adiraju  

**Link**: [PDF](https://arxiv.org/pdf/2506.16385)  

**Abstract**: Micro-gesture recognition is a challenging task in affective computing due to the subtle, involuntary nature of the gestures and their low movement amplitude. In this paper, we introduce a Pose-Guided Semantics-Aware CLIP-based architecture, or CLIP for Micro-Gesture recognition (CLIP-MG), a modified CLIP model tailored for micro-gesture classification on the iMiGUE dataset. CLIP-MG integrates human pose (skeleton) information into the CLIP-based recognition pipeline through pose-guided semantic query generation and a gated multi-modal fusion mechanism. The proposed model achieves a Top-1 accuracy of 61.82%. These results demonstrate both the potential of our approach and the remaining difficulty in fully adapting vision-language models like CLIP for micro-gesture recognition. 

**Abstract (ZH)**: 基于姿态引导的语义感知CLIP框架：微手势识别（CLIP-MG） 

---
# Watermarking Autoregressive Image Generation 

**Title (ZH)**: 自回归图像生成中的水印技术 

**Authors**: Nikola Jovanović, Ismail Labiad, Tomáš Souček, Martin Vechev, Pierre Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.16349)  

**Abstract**: Watermarking the outputs of generative models has emerged as a promising approach for tracking their provenance. Despite significant interest in autoregressive image generation models and their potential for misuse, no prior work has attempted to watermark their outputs at the token level. In this work, we present the first such approach by adapting language model watermarking techniques to this setting. We identify a key challenge: the lack of reverse cycle-consistency (RCC), wherein re-tokenizing generated image tokens significantly alters the token sequence, effectively erasing the watermark. To address this and to make our method robust to common image transformations, neural compression, and removal attacks, we introduce (i) a custom tokenizer-detokenizer finetuning procedure that improves RCC, and (ii) a complementary watermark synchronization layer. As our experiments demonstrate, our approach enables reliable and robust watermark detection with theoretically grounded p-values. 

**Abstract (ZH)**: 基于生成模型输出的水印嵌入已成为追踪其溯源的一种有前景的方法。尽管自回归图像生成模型及其潜在的滥用引起了广泛关注，但以往没有任何研究在标记级尝试对这些模型的输出进行水印嵌入。在本工作中，我们通过将语言模型水印技术适应到此领域，提出了首个此类方法。我们识别出一个关键挑战：缺乏反向循环一致性（RCC），即重新标记生成的图像标记会大幅改变标记序列，从而有效擦除水印。为解决这一问题，并使我们的方法能够抵抗常见的图像变换、神经压缩和删除攻击，我们引入了（i）一种自定义的标记器-逆标记器微调程序，以提高RCC，以及（ii）一个互补的水印同步层。如我们的实验所展示，我们的方法能够实现可靠且具有理论依据的水印检测。 

---
# Segment Anything for Satellite Imagery: A Strong Baseline and a Regional Dataset for Automatic Field Delineation 

**Title (ZH)**: 面向卫星影像的Segment Anything：一种强大的基线模型和区域数据集，用于自动农田界定 

**Authors**: Carmelo Scribano, Elena Govi, Paolo bertellini, Simone Parisi, Giorgia Franchini, Marko Bertogna  

**Link**: [PDF](https://arxiv.org/pdf/2506.16318)  

**Abstract**: Accurate mapping of agricultural field boundaries is essential for the efficient operation of agriculture. Automatic extraction from high-resolution satellite imagery, supported by computer vision techniques, can avoid costly ground surveys. In this paper, we present a pipeline for field delineation based on the Segment Anything Model (SAM), introducing a fine-tuning strategy to adapt SAM to this task. In addition to using published datasets, we describe a method for acquiring a complementary regional dataset that covers areas beyond current sources. Extensive experiments assess segmentation accuracy and evaluate the generalization capabilities. Our approach provides a robust baseline for automated field delineation. The new regional dataset, known as ERAS, is now publicly available. 

**Abstract (ZH)**: 准确的农田边界映射对于农业高效运作至关重要。基于高分辨率卫星影像和支持计算机视觉技术的自动提取可避免昂贵的实地调查。本文提出了一种基于段一切换模型（SAM）的农田界定管道，并介绍了一种细调策略以适应这一任务。除了使用公开的数据集，我们还描述了一种方法，用于获取一个补充性的区域数据集，该数据集涵盖了当前来源之外的区域。广泛的实验评估了分割准确性并考察了泛化能力。我们的方法为自动化农田界定提供了一个稳健的基准。新的区域数据集名为ERAS，现已公开可用。 

---
# Learning Multi-scale Spatial-frequency Features for Image Denoising 

**Title (ZH)**: 学习多尺度空间频率特征的图像去噪方法 

**Authors**: Xu Zhao, Chen Zhao, Xiantao Hu, Hongliang Zhang, Ying Tai, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16307)  

**Abstract**: Recent advancements in multi-scale architectures have demonstrated exceptional performance in image denoising tasks. However, existing architectures mainly depends on a fixed single-input single-output Unet architecture, ignoring the multi-scale representations of pixel level. In addition, previous methods treat the frequency domain uniformly, ignoring the different characteristics of high-frequency and low-frequency noise. In this paper, we propose a novel multi-scale adaptive dual-domain network (MADNet) for image denoising. We use image pyramid inputs to restore noise-free results from low-resolution images. In order to realize the interaction of high-frequency and low-frequency information, we design an adaptive spatial-frequency learning unit (ASFU), where a learnable mask is used to separate the information into high-frequency and low-frequency components. In the skip connections, we design a global feature fusion block to enhance the features at different scales. Extensive experiments on both synthetic and real noisy image datasets verify the effectiveness of MADNet compared with current state-of-the-art denoising approaches. 

**Abstract (ZH)**: Recent advancements in multi-scale architectures have demonstrated exceptional performance in image denoising tasks. However, existing architectures mainly depend on a fixed single-input single-output Unet architecture, ignoring the multi-scale representations of pixel level. In addition, previous methods treat the frequency domain uniformly, ignoring the different characteristics of high-frequency and low-frequency noise. In this paper, we propose a novel multi-scale adaptive dual-domain network (MADNet) for image denoising. We use image pyramid inputs to restore noise-free results from low-resolution images. In order to realize the interaction of high-frequency and low-frequency information, we design an adaptive spatial-frequency learning unit (ASFU), where a learnable mask is used to separate the information into high-frequency and low-frequency components. In the skip connections, we design a global feature fusion block to enhance the features at different scales. Extensive experiments on both synthetic and real noisy image datasets verify the effectiveness of MADNet compared with current state-of-the-art denoising approaches. 

---
# SycnMapV2: Robust and Adaptive Unsupervised Segmentation 

**Title (ZH)**: SycnMapV2: 坚韧且自适应的无监督分割 

**Authors**: Heng Zhang, Zikang Wan, Danilo Vasconcellos Vargas  

**Link**: [PDF](https://arxiv.org/pdf/2506.16297)  

**Abstract**: Human vision excels at segmenting visual cues without the need for explicit training, and it remains remarkably robust even as noise severity increases. In contrast, existing AI algorithms struggle to maintain accuracy under similar conditions. Here, we present SyncMapV2, the first to solve unsupervised segmentation with state-of-the-art robustness. SyncMapV2 exhibits a minimal drop in mIoU, only 0.01%, under digital corruption, compared to a 23.8% drop observed in SOTA this http URL superior performance extends across various types of corruption: noise (7.3% vs. 37.7%), weather (7.5% vs. 33.8%), and blur (7.0% vs. 29.5%). Notably, SyncMapV2 accomplishes this without any robust training, supervision, or loss functions. It is based on a learning paradigm that uses self-organizing dynamical equations combined with concepts from random networks. Moreover,unlike conventional methods that require re-initialization for each new input, SyncMapV2 adapts online, mimicking the continuous adaptability of human vision. Thus, we go beyond the accurate and robust results, and present the first algorithm that can do all the above online, adapting to input rather than re-initializing. In adaptability tests, SyncMapV2 demonstrates near-zero performance degradation, which motivates and fosters a new generation of robust and adaptive intelligence in the near future. 

**Abstract (ZH)**: SyncMapV2：第一种在多种噪声条件下具备卓越鲁棒性的无监督分割算法 

---
# Category-based Galaxy Image Generation via Diffusion Models 

**Title (ZH)**: 基于类别生成的银河图像生成方法 

**Authors**: Xingzhong Fan, Hongming Tang, Yue Zeng, M.B.N.Kouwenhoven, Guangquan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.16255)  

**Abstract**: Conventional galaxy generation methods rely on semi-analytical models and hydrodynamic simulations, which are highly dependent on physical assumptions and parameter tuning. In contrast, data-driven generative models do not have explicit physical parameters pre-determined, and instead learn them efficiently from observational data, making them alternative solutions to galaxy generation. Among these, diffusion models outperform Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) in quality and diversity. Leveraging physical prior knowledge to these models can further enhance their capabilities. In this work, we present GalCatDiff, the first framework in astronomy to leverage both galaxy image features and astrophysical properties in the network design of diffusion models. GalCatDiff incorporates an enhanced U-Net and a novel block entitled Astro-RAB (Residual Attention Block), which dynamically combines attention mechanisms with convolution operations to ensure global consistency and local feature fidelity. Moreover, GalCatDiff uses category embeddings for class-specific galaxy generation, avoiding the high computational costs of training separate models for each category. Our experimental results demonstrate that GalCatDiff significantly outperforms existing methods in terms of the consistency of sample color and size distributions, and the generated galaxies are both visually realistic and physically consistent. This framework will enhance the reliability of galaxy simulations and can potentially serve as a data augmentor to support future galaxy classification algorithm development. 

**Abstract (ZH)**: 基于扩散模型的GalCatDiff：结合星系图像特征和天体物理属性的星系生成框架 

---
# CF-Seg: Counterfactuals meet Segmentation 

**Title (ZH)**: CF-Seg: 反事实推理结合分割 

**Authors**: Raghav Mehta, Fabio De Sousa Ribeiro, Tian Xia, Melanie Roschewitz, Ainkaran Santhirasekaram, Dominic C. Marshall, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2506.16213)  

**Abstract**: Segmenting anatomical structures in medical images plays an important role in the quantitative assessment of various diseases. However, accurate segmentation becomes significantly more challenging in the presence of disease. Disease patterns can alter the appearance of surrounding healthy tissues, introduce ambiguous boundaries, or even obscure critical anatomical structures. As such, segmentation models trained on real-world datasets may struggle to provide good anatomical segmentation, leading to potential misdiagnosis. In this paper, we generate counterfactual (CF) images to simulate how the same anatomy would appear in the absence of disease without altering the underlying structure. We then use these CF images to segment structures of interest, without requiring any changes to the underlying segmentation model. Our experiments on two real-world clinical chest X-ray datasets show that the use of counterfactual images improves anatomical segmentation, thereby aiding downstream clinical decision-making. 

**Abstract (ZH)**: 生成反事实图像以在无病状态下模拟 Anatomy 在无病状态下出现的情况，从而改善解剖学分割，辅助下游临床决策。 

---
# Advanced Sign Language Video Generation with Compressed and Quantized Multi-Condition Tokenization 

**Title (ZH)**: 基于压缩和量化多条件令牌化的方法改进手语视频生成 

**Authors**: Cong Wang, Zexuan Deng, Zhiwei Jiang, Fei Shen, Yafeng Yin, Shiwei Gan, Zifeng Cheng, Shiping Ge, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15980)  

**Abstract**: Sign Language Video Generation (SLVG) seeks to generate identity-preserving sign language videos from spoken language texts. Existing methods primarily rely on the single coarse condition (\eg, skeleton sequences) as the intermediary to bridge the translation model and the video generation model, which limits both the naturalness and expressiveness of the generated videos. To overcome these limitations, we propose SignViP, a novel SLVG framework that incorporates multiple fine-grained conditions for improved generation fidelity. Rather than directly translating error-prone high-dimensional conditions, SignViP adopts a discrete tokenization paradigm to integrate and represent fine-grained conditions (\ie, fine-grained poses and 3D hands). SignViP contains three core components. (1) Sign Video Diffusion Model is jointly trained with a multi-condition encoder to learn continuous embeddings that encapsulate fine-grained motion and appearance. (2) Finite Scalar Quantization (FSQ) Autoencoder is further trained to compress and quantize these embeddings into discrete tokens for compact representation of the conditions. (3) Multi-Condition Token Translator is trained to translate spoken language text to discrete multi-condition tokens. During inference, Multi-Condition Token Translator first translates the spoken language text into discrete multi-condition tokens. These tokens are then decoded to continuous embeddings by FSQ Autoencoder, which are subsequently injected into Sign Video Diffusion Model to guide video generation. Experimental results show that SignViP achieves state-of-the-art performance across metrics, including video quality, temporal coherence, and semantic fidelity. The code is available at this https URL. 

**Abstract (ZH)**: 手语视频生成（SLVG）旨在从口头语言文本生成保有身份的手语视频。现有的方法主要依赖单一粗粒度条件（例如，骨架序列）作为中介来连接翻译模型和视频生成模型，这限制了生成视频的自然性和表现力。为克服这些限制，我们提出了一种新颖的手语视频生成框架SignViP，该框架结合了多细粒度条件以提高生成保真度。SignViP 不直接翻译易出错的高维条件，而是采用离散标记化范式来整合和表示细粒度条件（即，细粒度姿态和3D手部）。SignViP 包含三个核心组件：（1）手语视频扩散模型，与多条件编码器联合训练，学习包含细粒度运动和外观的连续嵌入；（2）有限标量量化（FSQ）自编码器进一步训练以压缩和量化这些嵌入成离散标记，以紧凑地表示条件；（3）多条件标记翻译器，训练以将口头语言文本翻译为离散的多条件标记。在推理过程中，多条件标记翻译器首先将口头语言文本翻译为离散的多条件标记，这些标记由FSQ自编码器解码为连续嵌入，并随后注入到手语视频扩散模型中以指导视频生成。实验结果表明，SignViP 在包括视频质量、时间连贯性和语义保真度的指标上达到了最先进的性能。代码可在以下链接中获取：this https URL。 

---
# Beyond Audio and Pose: A General-Purpose Framework for Video Synchronization 

**Title (ZH)**: 超越音频和姿势：一种通用视频同步框架 

**Authors**: Yosub Shin, Igor Molybog  

**Link**: [PDF](https://arxiv.org/pdf/2506.15937)  

**Abstract**: Video synchronization-aligning multiple video streams capturing the same event from different angles-is crucial for applications such as reality TV show production, sports analysis, surveillance, and autonomous systems. Prior work has heavily relied on audio cues or specific visual events, limiting applicability in diverse settings where such signals may be unreliable or absent. Additionally, existing benchmarks for video synchronization lack generality and reproducibility, restricting progress in the field. In this work, we introduce VideoSync, a video synchronization framework that operates independently of specific feature extraction methods, such as human pose estimation, enabling broader applicability across different content types. We evaluate our system on newly composed datasets covering single-human, multi-human, and non-human scenarios, providing both the methodology and code for dataset creation to establish reproducible benchmarks. Our analysis reveals biases in prior SOTA work, particularly in SeSyn-Net's preprocessing pipeline, leading to inflated performance claims. We correct these biases and propose a more rigorous evaluation framework, demonstrating that VideoSync outperforms existing approaches, including SeSyn-Net, under fair experimental conditions. Additionally, we explore various synchronization offset prediction methods, identifying a convolutional neural network (CNN)-based model as the most effective. Our findings advance video synchronization beyond domain-specific constraints, making it more generalizable and robust for real-world applications. 

**Abstract (ZH)**: 视频同步：多角度捕捉同一事件的视频流同步对于实景电视节目制作、体育分析、监控和自主系统等应用至关重要。先前的工作主要依赖于音频线索或特定的视觉事件，限制了其在信号不可靠或缺失的多样环境中的适用性。此外，现有的视频同步基准缺乏通用性和可重复性，限制了该领域的进步。在本工作中，我们提出了VideoSync，这是一种独立于特定特征提取方法的视频同步框架，如人体姿态估计，使其在不同类型的内容中具有更广泛的适用性。我们使用新编纂的数据集评估了系统的表现，这些数据集覆盖了单人、多人和非人场景，提供了数据集创建的方法和代码，以建立可重复的基准。我们的分析揭示了先前最佳方法中的偏见，特别是在SeSyn-Net的预处理管道中，导致了夸大了的性能声称。我们纠正了这些偏见，并提出了一种更严格的评估框架，展示了在公平的实验条件下，VideoSync优于现有方法，包括SeSyn-Net。此外，我们探索了各种同步偏移预测方法，发现基于卷积神经网络（CNN）的模型最有效。我们的发现推动了视频同步的发展，使其超越了特定领域的限制，更具通用性和鲁棒性，适用于实际应用。 

---
# MoiréXNet: Adaptive Multi-Scale Demoiréing with Linear Attention Test-Time Training and Truncated Flow Matching Prior 

**Title (ZH)**: MoiréXNet：自适应多尺度消moire处理的线性注意力测试时训练及截断流匹配先验 

**Authors**: Liangyan Li, Yimo Ning, Kevin Le, Wei Dong, Yunzhe Li, Jun Chen, Xiaohong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15929)  

**Abstract**: This paper introduces a novel framework for image and video demoiréing by integrating Maximum A Posteriori (MAP) estimation with advanced deep learning techniques. Demoiréing addresses inherently nonlinear degradation processes, which pose significant challenges for existing methods.
Traditional supervised learning approaches either fail to remove moiré patterns completely or produce overly smooth results. This stems from constrained model capacity and scarce training data, which inadequately represent the clean image distribution and hinder accurate reconstruction of ground-truth images. While generative models excel in image restoration for linear degradations, they struggle with nonlinear cases such as demoiréing and often introduce artifacts.
To address these limitations, we propose a hybrid MAP-based framework that integrates two complementary components. The first is a supervised learning model enhanced with efficient linear attention Test-Time Training (TTT) modules, which directly learn nonlinear mappings for RAW-to-sRGB demoiréing. The second is a Truncated Flow Matching Prior (TFMP) that further refines the outputs by aligning them with the clean image distribution, effectively restoring high-frequency details and suppressing artifacts. These two components combine the computational efficiency of linear attention with the refinement abilities of generative models, resulting in improved restoration performance. 

**Abstract (ZH)**: 一种结合最大后验估计与先进深度学习技术的新型图像和视频反摩尔纹框架 

---
# Cross-Modality Learning for Predicting IHC Biomarkers from H&E-Stained Whole-Slide Images 

**Title (ZH)**: 从H&E染色全切片图像中预测IHC生物标志物的跨模态学习 

**Authors**: Amit Das, Naofumi Tomita, Kyle J. Syme, Weijie Ma, Paige O'Connor, Kristin N. Corbett, Bing Ren, Xiaoying Liu, Saeed Hassanpour  

**Link**: [PDF](https://arxiv.org/pdf/2506.15853)  

**Abstract**: Hematoxylin and Eosin (H&E) staining is a cornerstone of pathological analysis, offering reliable visualization of cellular morphology and tissue architecture for cancer diagnosis, subtyping, and grading. Immunohistochemistry (IHC) staining provides molecular insights by detecting specific proteins within tissues, enhancing diagnostic accuracy, and improving treatment planning. However, IHC staining is costly, time-consuming, and resource-intensive, requiring specialized expertise. To address these limitations, this study proposes HistoStainAlign, a novel deep learning framework that predicts IHC staining patterns directly from H&E whole-slide images (WSIs) by learning joint representations of morphological and molecular features. The framework integrates paired H&E and IHC embeddings through a contrastive training strategy, capturing complementary features across staining modalities without patch-level annotations or tissue registration. The model was evaluated on gastrointestinal and lung tissue WSIs with three commonly used IHC stains: P53, PD-L1, and Ki-67. HistoStainAlign achieved weighted F1 scores of 0.735 [95% Confidence Interval (CI): 0.670-0.799], 0.830 [95% CI: 0.772-0.886], and 0.723 [95% CI: 0.607-0.836], respectively for these three IHC stains. Embedding analyses demonstrated the robustness of the contrastive alignment in capturing meaningful cross-stain relationships. Comparisons with a baseline model further highlight the advantage of incorporating contrastive learning for improved stain pattern prediction. This study demonstrates the potential of computational approaches to serve as a pre-screening tool, helping prioritize cases for IHC staining and improving workflow efficiency. 

**Abstract (ZH)**: HE和苏木精- eosin (H&E) 染色是病理分析的基石，为癌症诊断、亚型分类和分级提供可靠的细胞形态和组织结构可视化。免疫组织化学（IHC）染色通过检测组织内的特定蛋白质，提供分子洞察，增强诊断准确性并改善治疗规划。然而，IHC染色成本高、耗时且资源密集，需要专门的 expertise。为解决这些局限性，本研究提出了一种名为HistoStainAlign的新型深度学习框架，该框架可以直接从H&E全切片图像（WSI）中预测IHC染色模式，通过学习形态和分子特征的联合表示。该框架通过对比训练策略结合配对的H&E和IHC嵌入，捕获不同染色模式下的互补特征，无需斑块级注释或组织注册。该模型在胃肠道和肺组织的WSI上进行了评估，使用三种常用的IHC染色：P53、PD-L1和Ki-67。HistoStainAlign分别在这三种IHC染色中实现了加权F1分数为0.735 [95% 置信区间（CI）：0.670-0.799]、0.830 [95% CI：0.772-0.886] 和0.723 [95% CI：0.607-0.836]。嵌入分析表明对比对齐在捕获有意义的跨染色关系方面的稳健性。与基准模型的比较进一步突出了结合对比学习的优越性，以改善染色模式预测。本研究展示了计算方法作为预筛选工具的潜力，帮助优先处理需要IHC染色的病例并提高工作流程效率。 

---
# MoNetV2: Enhanced Motion Network for Freehand 3D Ultrasound Reconstruction 

**Title (ZH)**: MoNetV2: 提升的-motion 网络用于自由手绘制三维超声重建 

**Authors**: Mingyuan Luo, Xin Yang, Zhongnuo Yan, Yan Cao, Yuanji Zhang, Xindi Hu, Jin Wang, Haoxuan Ding, Wei Han, Litao Sun, Dong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.15835)  

**Abstract**: Three-dimensional (3D) ultrasound (US) aims to provide sonographers with the spatial relationships of anatomical structures, playing a crucial role in clinical diagnosis. Recently, deep-learning-based freehand 3D US has made significant advancements. It reconstructs volumes by estimating transformations between images without external tracking. However, image-only reconstruction poses difficulties in reducing cumulative drift and further improving reconstruction accuracy, particularly in scenarios involving complex motion trajectories. In this context, we propose an enhanced motion network (MoNetV2) to enhance the accuracy and generalizability of reconstruction under diverse scanning velocities and tactics. First, we propose a sensor-based temporal and multi-branch structure that fuses image and motion information from a velocity perspective to improve image-only reconstruction accuracy. Second, we devise an online multi-level consistency constraint that exploits the inherent consistency of scans to handle various scanning velocities and tactics. This constraint exploits both scan-level velocity consistency, path-level appearance consistency, and patch-level motion consistency to supervise inter-frame transformation estimation. Third, we distill an online multi-modal self-supervised strategy that leverages the correlation between network estimation and motion information to further reduce cumulative errors. Extensive experiments clearly demonstrate that MoNetV2 surpasses existing methods in both reconstruction quality and generalizability performance across three large datasets. 

**Abstract (ZH)**: 基于运动网络V2的三维超声成像增强方法 

---
# VEIGAR: View-consistent Explicit Inpainting and Geometry Alignment for 3D object Removal 

**Title (ZH)**: VEIGAR：视图一致的显式修复和几何对齐以去除3D物体 

**Authors**: Pham Khai Nguyen Do, Bao Nguyen Tran, Nam Nguyen, Duc Dung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15821)  

**Abstract**: Recent advances in Novel View Synthesis (NVS) and 3D generation have significantly improved editing tasks, with a primary emphasis on maintaining cross-view consistency throughout the generative process. Contemporary methods typically address this challenge using a dual-strategy framework: performing consistent 2D inpainting across all views guided by embedded priors either explicitly in pixel space or implicitly in latent space; and conducting 3D reconstruction with additional consistency guidance. Previous strategies, in particular, often require an initial 3D reconstruction phase to establish geometric structure, introducing considerable computational overhead. Even with the added cost, the resulting reconstruction quality often remains suboptimal. In this paper, we present VEIGAR, a computationally efficient framework that outperforms existing methods without relying on an initial reconstruction phase. VEIGAR leverages a lightweight foundation model to reliably align priors explicitly in the pixel space. In addition, we introduce a novel supervision strategy based on scale-invariant depth loss, which removes the need for traditional scale-and-shift operations in monocular depth regularization. Through extensive experimentation, VEIGAR establishes a new state-of-the-art benchmark in reconstruction quality and cross-view consistency, while achieving a threefold reduction in training time compared to the fastest existing method, highlighting its superior balance of efficiency and effectiveness. 

**Abstract (ZH)**: Recent Advances in Novel View Synthesis (NVS) and 3D Generation Have Significantly Improved Editing Tasks While Maintaining Cross-View Consistency Without Relying on an Initial Reconstruction Phase 

---
