# Diffusion-Based Imaginative Coordination for Bimanual Manipulation 

**Title (ZH)**: 基于扩散的想象性协调实现双臂操作 

**Authors**: Huilin Xu, Jian Ding, Jiakun Xu, Ruixiang Wang, Jun Chen, Jinjie Mai, Yanwei Fu, Bernard Ghanem, Feng Xu, Mohamed Elhoseiny  

**Link**: [PDF](https://arxiv.org/pdf/2507.11296)  

**Abstract**: Bimanual manipulation is crucial in robotics, enabling complex tasks in industrial automation and household services. However, it poses significant challenges due to the high-dimensional action space and intricate coordination requirements. While video prediction has been recently studied for representation learning and control, leveraging its ability to capture rich dynamic and behavioral information, its potential for enhancing bimanual coordination remains underexplored. To bridge this gap, we propose a unified diffusion-based framework for the joint optimization of video and action prediction. Specifically, we propose a multi-frame latent prediction strategy that encodes future states in a compressed latent space, preserving task-relevant features. Furthermore, we introduce a unidirectional attention mechanism where video prediction is conditioned on the action, while action prediction remains independent of video prediction. This design allows us to omit video prediction during inference, significantly enhancing efficiency. Experiments on two simulated benchmarks and a real-world setting demonstrate a significant improvement in the success rate over the strong baseline ACT using our method, achieving a \textbf{24.9\%} increase on ALOHA, an \textbf{11.1\%} increase on RoboTwin, and a \textbf{32.5\%} increase in real-world experiments. Our models and code are publicly available at this https URL. 

**Abstract (ZH)**: 双臂 manipulation 对机器人技术至关重要，能够在工业自动化和家庭服务中执行复杂任务。然而，由于高维动作空间和复杂的协调要求，它提出了重大挑战。虽然视频预测已被研究用于表示学习和控制，并利用其捕捉丰富动态和行为信息的能力，但其在提升双臂协调方面的潜力尚未得到充分开发。为了解决这一问题，我们提出了一种统一的基于扩散的框架，用于联合优化视频和动作预测。具体来说，我们提出了一种多帧潜在预测策略，将未来状态编码在一个压缩的潜在空间中，保留任务相关特征。此外，我们引入了一种单向注意力机制，其中视频预测依赖于动作，而动作预测与视频预测独立。这种设计允许我们在推理过程中省略视频预测，显著提高效率。在两个模拟基准和一个实际场景中的实验表明，与强大的基线ACT相比，我们的方法显著提高了成功率，在ALOHA中提高了24.9%，在RoboTwin中提高了11.1%，在实际实验中提高了32.5%。我们的模型和代码已在以下网址公开。 

---
# TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update 

**Title (ZH)**: TRAN-D：基于2D高斯散斑的稀疏视图透明对象深度重构方法及其在场景更新中的物理模拟 

**Authors**: Jeongyun Kim, Seunghoon Jeong, Giseop Kim, Myung-Hwan Jeon, Eunji Jun, Ayoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.11069)  

**Abstract**: Understanding the 3D geometry of transparent objects from RGB images is challenging due to their inherent physical properties, such as reflection and refraction. To address these difficulties, especially in scenarios with sparse views and dynamic environments, we introduce TRAN-D, a novel 2D Gaussian Splatting-based depth reconstruction method for transparent objects. Our key insight lies in separating transparent objects from the background, enabling focused optimization of Gaussians corresponding to the object. We mitigate artifacts with an object-aware loss that places Gaussians in obscured regions, ensuring coverage of invisible surfaces while reducing overfitting. Furthermore, we incorporate a physics-based simulation that refines the reconstruction in just a few seconds, effectively handling object removal and chain-reaction movement of remaining objects without the need for rescanning. TRAN-D is evaluated on both synthetic and real-world sequences, and it consistently demonstrated robust improvements over existing GS-based state-of-the-art methods. In comparison with baselines, TRAN-D reduces the mean absolute error by over 39% for the synthetic TRansPose sequences. Furthermore, despite being updated using only one image, TRAN-D reaches a {\delta} < 2.5 cm accuracy of 48.46%, over 1.5 times that of baselines, which uses six images. Code and more results are available at this https URL. 

**Abstract (ZH)**: 从RGB图像理解透明物体的3D几何形状具有挑战性，因为透明物体具有固有的物理属性，如反射和折射。为了应对这些困难，特别是在稀疏视角和动态环境场景中，我们引入了TRAN-D，一种新型的基于2D高斯斑点的透明物体深度重建方法。我们的关键见解在于将透明物体与背景分离，从而使我们能够专注于优化与物体对应的高斯函数。我们通过一种对象感知的损失来减轻伪影，该损失使高斯函数定位在被遮挡的区域，从而确保覆盖看不见的表面同时减少过拟合。此外，我们结合了基于物理的模拟，在几秒钟内细化重建结果，有效处理物体去除和剩余物体连带运动，而无需重新扫描。TRAN-D在合成序列和真实世界序列上进行了评估，并且在与现有基于高斯斑点的最先进的方法相比时，表现出一致的稳健改进。相比基线方法，TRAN-D在合成TRansPose序列上的均方绝对误差降低了超过39%。此外，即使只使用了一张图像进行更新，TRAN-D的δ < 2.5 cm精度达到了48.46%，超过了使用六张图像的基线方法1.5倍以上的精度。相关代码和更多结果详见此链接。 

---
# Task-Oriented Human Grasp Synthesis via Context- and Task-Aware Diffusers 

**Title (ZH)**: 面向任务的人手抓取合成：基于上下文和任务的扩散模型 

**Authors**: An-Lun Liu, Yu-Wei Chao, Yi-Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11287)  

**Abstract**: In this paper, we study task-oriented human grasp synthesis, a new grasp synthesis task that demands both task and context awareness. At the core of our method is the task-aware contact maps. Unlike traditional contact maps that only reason about the manipulated object and its relation with the hand, our enhanced maps take into account scene and task information. This comprehensive map is critical for hand-object interaction, enabling accurate grasping poses that align with the task. We propose a two-stage pipeline that first constructs a task-aware contact map informed by the scene and task. In the subsequent stage, we use this contact map to synthesize task-oriented human grasps. We introduce a new dataset and a metric for the proposed task to evaluate our approach. Our experiments validate the importance of modeling both scene and task, demonstrating significant improvements over existing methods in both grasp quality and task performance. See our project page for more details: this https URL 

**Abstract (ZH)**: 本文研究面向任务的人机抓取合成，这是一个既需要任务意识又需要上下文意识的新抓取合成任务。我们方法的核心是任务意识接触图。不同于传统接触图仅考虑操作对象及其与手的关系，我们的增强接触图还考虑场景和任务信息。这种综合性的接触图对于手-物交互至关重要，能够生成与任务相匹配的准确抓取姿态。我们提出了一种两阶段管道，首先基于场景和任务构建任务意识接触图，随后使用该接触图合成面向任务的人机抓取。我们引入了新的数据集和评价指标来评估所提出的方法。实验结果验证了同时建模场景和任务的重要性，在抓取质量与任务性能方面显著优于现有方法。更多详情请参见我们的项目页面：this https URL 

---
# Streaming 4D Visual Geometry Transformer 

**Title (ZH)**: Streaming 4D Visual Geometry Transformer 

**Authors**: Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11539)  

**Abstract**: Perceiving and reconstructing 4D spatial-temporal geometry from videos is a fundamental yet challenging computer vision task. To facilitate interactive and real-time applications, we propose a streaming 4D visual geometry transformer that shares a similar philosophy with autoregressive large language models. We explore a simple and efficient design and employ a causal transformer architecture to process the input sequence in an online manner. We use temporal causal attention and cache the historical keys and values as implicit memory to enable efficient streaming long-term 4D reconstruction. This design can handle real-time 4D reconstruction by incrementally integrating historical information while maintaining high-quality spatial consistency. For efficient training, we propose to distill knowledge from the dense bidirectional visual geometry grounded transformer (VGGT) to our causal model. For inference, our model supports the migration of optimized efficient attention operator (e.g., FlashAttention) from the field of large language models. Extensive experiments on various 4D geometry perception benchmarks demonstrate that our model increases the inference speed in online scenarios while maintaining competitive performance, paving the way for scalable and interactive 4D vision systems. Code is available at: this https URL. 

**Abstract (ZH)**: 从视频中感知和重构4D时空几何是一个基本但具有挑战性的计算机视觉任务。为了支持交互式和实时应用，我们提出了一种流式4D视觉几何变换器，其设计理念与自回归大规模语言模型类似。我们探索了一个简单而高效的架构，并采用因果变换器架构在线处理输入序列。我们利用时间因果注意力，并通过隐式记忆缓存历史键值对，以实现高效流式4D重建。该设计能够在增量集成历史信息的同时保持高质量的空间一致性。为了提高训练效率，我们提出从密集双向视觉几何基础变换器（VGGT）中提炼知识到我们的因果模型中。对于推理，我们的模型支持从大规模语言模型领域迁移优化的高效注意力算子（例如，FlashAttention）。在各种4D几何感知基准上的广泛实验表明，我们的模型在保持竞争力的同时提升了在线场景下的推理速度，为可扩展和交互式的4D视觉系统铺平了道路。代码可在以下链接获取：this https URL。 

---
# U-RWKV: Lightweight medical image segmentation with direction-adaptive RWKV 

**Title (ZH)**: U-RWKV：具有方向自适应性的小型化医学图像分割方法 

**Authors**: Hongbo Ye, Fenghe Tang, Peiang Zhao, Zhen Huang, Dexin Zhao, Minghao Bian, S.Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.11415)  

**Abstract**: Achieving equity in healthcare accessibility requires lightweight yet high-performance solutions for medical image segmentation, particularly in resource-limited settings. Existing methods like U-Net and its variants often suffer from limited global Effective Receptive Fields (ERFs), hindering their ability to capture long-range dependencies. To address this, we propose U-RWKV, a novel framework leveraging the Recurrent Weighted Key-Value(RWKV) architecture, which achieves efficient long-range modeling at O(N) computational cost. The framework introduces two key innovations: the Direction-Adaptive RWKV Module(DARM) and the Stage-Adaptive Squeeze-and-Excitation Module(SASE). DARM employs Dual-RWKV and QuadScan mechanisms to aggregate contextual cues across images, mitigating directional bias while preserving global context and maintaining high computational efficiency. SASE dynamically adapts its architecture to different feature extraction stages, balancing high-resolution detail preservation and semantic relationship capture. Experiments demonstrate that U-RWKV achieves state-of-the-art segmentation performance with high computational efficiency, offering a practical solution for democratizing advanced medical imaging technologies in resource-constrained environments. The code is available at this https URL. 

**Abstract (ZH)**: 实现医疗服务公平性要求在资源受限环境中采用轻量且高性能的医疗图像分割解决方案。现有的方法如U-Net及其变体常常受到有限全局有效感受野(ERFs)的限制，影响其捕捉长程依赖的能力。为解决这一问题，我们提出了U-RWKV，这是一种利用循环加权键值(RWKV)架构的新框架，能够在O(N)计算代价下实现高效的长程建模。该框架引入了两大创新：方向自适应RWKV模块(DARM)和阶段自适应Squeeze-and-Excitation模块(SASE)。DARM利用双RWKV和四扫描机制在图像间聚合上下文线索，减轻方向偏见同时保留全局上下文并维持高计算效率。SASE动态适应不同特征提取阶段的架构，平衡高分辨率细节保真和语义关系捕捉。实验表明，U-RWKV在保持高计算效率的同时实现了最先进的分割性能，提供了一种实用的解决方案，以在资源受限环境中普及高级医疗成像技术。相关代码可在以下链接访问。 

---
# Attributes Shape the Embedding Space of Face Recognition Models 

**Title (ZH)**: 属性塑造面部识别模型的嵌入空间 

**Authors**: Pierrick Leroy, Antonio Mastropietro, Marco Nurisso, Francesco Vaccarino  

**Link**: [PDF](https://arxiv.org/pdf/2507.11372)  

**Abstract**: Face Recognition (FR) tasks have made significant progress with the advent of Deep Neural Networks, particularly through margin-based triplet losses that embed facial images into high-dimensional feature spaces. During training, these contrastive losses focus exclusively on identity information as labels. However, we observe a multiscale geometric structure emerging in the embedding space, influenced by interpretable facial (e.g., hair color) and image attributes (e.g., contrast). We propose a geometric approach to describe the dependence or invariance of FR models to these attributes and introduce a physics-inspired alignment metric. We evaluate the proposed metric on controlled, simplified models and widely used FR models fine-tuned with synthetic data for targeted attribute augmentation. Our findings reveal that the models exhibit varying degrees of invariance across different attributes, providing insight into their strengths and weaknesses and enabling deeper interpretability. Code available here: this https URL}{this https URL 

**Abstract (ZH)**: 基于深度神经网络的Face Recognition任务通过基于边际的三元组损失将面部图像嵌入到高维特征空间中取得了显著进展。在训练过程中，这些对比损失专注于身份信息作为标签。然而，我们观察到嵌入空间中出现了一种多尺度几何结构，受到可解释的面部特征（例如，发色）和图像属性（例如，对比度）的影响。我们提出了一种几何方法来描述Face Recognition模型对这些属性的依赖性或不变性，并引入了一种受物理启发的对齐度量。我们在受控的简化模型和广泛使用的通过合成数据微调以针对特定属性增强的Face Recognition模型上评估了提出的度量标准。我们的研究结果表明，模型在不同属性上的不变性程度不同，这为深入了解模型的优势和劣势提供了见解，并使解释更加深入。代码可在此获取：this https URL 

---
# HANS-Net: Hyperbolic Convolution and Adaptive Temporal Attention for Accurate and Generalizable Liver and Tumor Segmentation in CT Imaging 

**Title (ZH)**: HANS-Net：双曲卷积与自适应时空注意力在CT成像中用于准确且泛化能力强的肝脏和肿瘤分割 

**Authors**: Arefin Ittesafun Abian, Ripon Kumar Debnath, Md. Abdur Rahman, Mohaimenul Azam Khan Raiaan, Md Rafiqul Islam, Asif Karim, Reem E. Mohamed, Sami Azam  

**Link**: [PDF](https://arxiv.org/pdf/2507.11325)  

**Abstract**: Accurate liver and tumor segmentation on abdominal CT images is critical for reliable diagnosis and treatment planning, but remains challenging due to complex anatomical structures, variability in tumor appearance, and limited annotated data. To address these issues, we introduce Hyperbolic-convolutions Adaptive-temporal-attention with Neural-representation and Synaptic-plasticity Network (HANS-Net), a novel segmentation framework that synergistically combines hyperbolic convolutions for hierarchical geometric representation, a wavelet-inspired decomposition module for multi-scale texture learning, a biologically motivated synaptic plasticity mechanism for adaptive feature enhancement, and an implicit neural representation branch to model fine-grained and continuous anatomical boundaries. Additionally, we incorporate uncertainty-aware Monte Carlo dropout to quantify prediction confidence and lightweight temporal attention to improve inter-slice consistency without sacrificing efficiency. Extensive evaluations of the LiTS dataset demonstrate that HANS-Net achieves a mean Dice score of 93.26%, an IoU of 88.09%, an average symmetric surface distance (ASSD) of 0.72 mm, and a volume overlap error (VOE) of 11.91%. Furthermore, cross-dataset validation on the 3D-IRCADb-01 dataset obtains an average Dice of 87.45%, IoU of 80.30%, ASSD of 1.525 mm, and VOE of 19.71%, indicating strong generalization across different datasets. These results confirm the effectiveness and robustness of HANS-Net in providing anatomically consistent, accurate, and confident liver and tumor segmentation. 

**Abstract (ZH)**: Hyperbolic-convolutions Adaptive-temporal-attention with Neural-representation and Synaptic-plasticity Network for Accurate Liver and Tumor Segmentation in Abdominal CT Images 

---
# YOLOatr : Deep Learning Based Automatic Target Detection and Localization in Thermal Infrared Imagery 

**Title (ZH)**: YOLOatr：基于深度学习的热红外图像自动目标检测与定位 

**Authors**: Aon Safdar, Usman Akram, Waseem Anwar, Basit Malik, Mian Ibad Ali  

**Link**: [PDF](https://arxiv.org/pdf/2507.11267)  

**Abstract**: Automatic Target Detection (ATD) and Recognition (ATR) from Thermal Infrared (TI) imagery in the defense and surveillance domain is a challenging computer vision (CV) task in comparison to the commercial autonomous vehicle perception domain. Limited datasets, peculiar domain-specific and TI modality-specific challenges, i.e., limited hardware, scale invariance issues due to greater distances, deliberate occlusion by tactical vehicles, lower sensor resolution and resultant lack of structural information in targets, effects of weather, temperature, and time of day variations, and varying target to clutter ratios all result in increased intra-class variability and higher inter-class similarity, making accurate real-time ATR a challenging CV task. Resultantly, contemporary state-of-the-art (SOTA) deep learning architectures underperform in the ATR domain. We propose a modified anchor-based single-stage detector, called YOLOatr, based on a modified YOLOv5s, with optimal modifications to the detection heads, feature fusion in the neck, and a custom augmentation profile. We evaluate the performance of our proposed model on a comprehensive DSIAC MWIR dataset for real-time ATR over both correlated and decorrelated testing protocols. The results demonstrate that our proposed model achieves state-of-the-art ATR performance of up to 99.6%. 

**Abstract (ZH)**: 自动目标检测（ATD）和识别（ATR）从红外（TI）图像在防御与监控领域的挑战性计算机视觉（CV）任务：基于商业自动驾驶汽车感知领域的比较 

---
# Assessing Color Vision Test in Large Vision-language Models 

**Title (ZH)**: 评估大型视觉语言模型的颜色视觉测试 

**Authors**: Hongfei Ye, Bin Chen, Wenxi Liu, Yu Zhang, Zhao Li, Dandan Ni, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11153)  

**Abstract**: With the widespread adoption of large vision-language models, the capacity for color vision in these models is crucial. However, the color vision abilities of large visual-language models have not yet been thoroughly explored. To address this gap, we define a color vision testing task for large vision-language models and construct a dataset \footnote{Anonymous Github Showing some of the data this https URL} that covers multiple categories of test questions and tasks of varying difficulty levels. Furthermore, we analyze the types of errors made by large vision-language models and propose fine-tuning strategies to enhance their performance in color vision tests. 

**Abstract (ZH)**: 随着大型多模态模型的广泛应用，这些模型的色彩识别能力至关重要。然而，大型视觉-语言模型的色彩识别能力尚未得到充分研究。为弥补这一空白，我们定义了一个针对大型视觉-语言模型的色彩识别测试任务，并构建了一个数据集（匿名GitHub链接，部分内容可查看：https://github.com/），该数据集涵盖了多种类别、不同难度级别的测试问题和任务。此外，我们分析了大型视觉-语言模型在色彩识别测试中犯下的错误类型，并提出了一些微调策略以提高其在色彩识别测试中的表现。 

---
# Automatic Road Subsurface Distress Recognition from Ground Penetrating Radar Images using Deep Learning-based Cross-verification 

**Title (ZH)**: 基于深度学习交叉验证的地面穿透雷达图像自动道路地下病害识别 

**Authors**: Chang Peng, Bao Yang, Meiqi Li, Ge Zhang, Hui Sun, Zhenyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11081)  

**Abstract**: Ground penetrating radar (GPR) has become a rapid and non-destructive solution for road subsurface distress (RSD) detection. However, RSD recognition from GPR images is labor-intensive and heavily relies on inspectors' expertise. Deep learning offers the possibility for automatic RSD recognition, but its current performance is limited by two factors: Scarcity of high-quality dataset for network training and insufficient capability of network to distinguish RSD. In this study, a rigorously validated 3D GPR dataset containing 2134 samples of diverse types was constructed through field scanning. Based on the finding that the YOLO model trained with one of the three scans of GPR images exhibits varying sensitivity to specific type of RSD, we proposed a novel cross-verification strategy with outstanding accuracy in RSD recognition, achieving recall over 98.6% in field tests. The approach, integrated into an online RSD detection system, can reduce the labor of inspection by around 90%. 

**Abstract (ZH)**: 地面穿透雷达(GPR)已成为一种快速且非破坏性的解方案，用于道路地下病害(RSD)检测。然而，从GPR图像中识别RSD劳动密集且高度依赖检查人员的专业知识。深度学习为自动RSD识别提供了可能性，但其当前表现受限于两个因素：用于网络训练的高质量数据集稀缺以及网络区分RSD的能力不足。在此研究中，通过现场扫描构建了一个严格的验证3D GPR数据集，包含2134个不同类型样本。基于训练其中一个GPR图像扫描的YOLO模型对特定类型RSD表现出不同程度敏感性的发现，我们提出了一个准确的跨验证策略，在现场测试中RSD识别召回率超过98.6%。该方法整合到在线RSD检测系统中，可将检查工作量减少约90%。 

---
# Joint angle model based learning to refine kinematic human pose estimation 

**Title (ZH)**: 基于关节角度模型的学习以细化人体姿态估计 

**Authors**: Chang Peng, Yifei Zhou, Huifeng Xi, Shiqing Huang, Chuangye Chen, Jianming Yang, Bao Yang, Zhenyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11075)  

**Abstract**: Marker-free human pose estimation (HPE) has found increasing applications in various fields. Current HPE suffers from occasional errors in keypoint recognition and random fluctuation in keypoint trajectories when analyzing kinematic human poses. The performance of existing deep learning-based models for HPE refinement is considerably limited by inaccurate training datasets in which the keypoints are manually annotated. This paper proposed a novel method to overcome the difficulty through joint angle-based modeling. The key techniques include: (i) A joint angle-based model of human pose, which is robust to describe kinematic human poses; (ii) Approximating temporal variation of joint angles through high order Fourier series to get reliable "ground truth"; (iii) A bidirectional recurrent network is designed as a post-processing module to refine the estimation of well-established HRNet. Trained with the high-quality dataset constructed using our method, the network demonstrates outstanding performance to correct wrongly recognized joints and smooth their spatiotemporal trajectories. Tests show that joint angle-based refinement (JAR) outperforms the state-of-the-art HPE refinement network in challenging cases like figure skating and breaking. 

**Abstract (ZH)**: 基于关节角度的人体姿态估计方法 

---
# Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling 

**Title (ZH)**: 带正则化评分蒸馏采样的鲁棒3D-掩码部分级编辑在3D高斯点绘制中 

**Authors**: Hayeon Kim, Ji Ha Jang, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2507.11061)  

**Abstract**: Recent advances in 3D neural representations and instance-level editing models have enabled the efficient creation of high-quality 3D content. However, achieving precise local 3D edits remains challenging, especially for Gaussian Splatting, due to inconsistent multi-view 2D part segmentations and inherently ambiguous nature of Score Distillation Sampling (SDS) loss. To address these limitations, we propose RoMaP, a novel local 3D Gaussian editing framework that enables precise and drastic part-level modifications. First, we introduce a robust 3D mask generation module with our 3D-Geometry Aware Label Prediction (3D-GALP), which uses spherical harmonics (SH) coefficients to model view-dependent label variations and soft-label property, yielding accurate and consistent part segmentations across viewpoints. Second, we propose a regularized SDS loss that combines the standard SDS loss with additional regularizers. In particular, an L1 anchor loss is introduced via our Scheduled Latent Mixing and Part (SLaMP) editing method, which generates high-quality part-edited 2D images and confines modifications only to the target region while preserving contextual coherence. Additional regularizers, such as Gaussian prior removal, further improve flexibility by allowing changes beyond the existing context, and robust 3D masking prevents unintended edits. Experimental results demonstrate that our RoMaP achieves state-of-the-art local 3D editing on both reconstructed and generated Gaussian scenes and objects qualitatively and quantitatively, making it possible for more robust and flexible part-level 3D Gaussian editing. 

**Abstract (ZH)**: Recent Advances in 3D神经表示和实例级编辑模型使得高效创建高质量3D内容成为可能，然而，实现精确的局部3D编辑仍然具有挑战性，尤其是在高斯成簇方面，这主要是由于多视角2D部件分割的一致性问题和Score Distillation Sampling (SDS)损失固有的含糊性。为解决这些限制，我们提出RoMaP，一种新颖的局部3D高斯编辑框架，能够实现精确和剧烈的部件级修改。首先，我们引入了一种鲁棒的3D掩码生成模块，使用球谐系数（SH）来建模视角依赖的标签变化和软标签特性，从而在不同视角下获得准确且一致的部件分割。其次，我们提出了一种正则化的SDS损失，将标准的SDS损失与附加正则化器相结合。特别是，通过我们的Scheduled Latent Mixing and Part (SLaMP)编辑方法，引入了L1锚定损失，该方法生成高质量的部件编辑2D图像，并将修改仅限于目标区域，同时保持上下文的一致性。附加的正则化器，如高斯先验去除，进一步提高了灵活性，允许超出现有上下文的更改，并且鲁棒的3D掩码防止了非预期的修改。实验结果表明，我们的RoMaP在重构和生成的高斯场景和对象的局部3D编辑方面达到了最先进的水平，无论是定性还是定量评估，都使其成为更鲁棒和灵活的部件级3D高斯编辑的可能性。 

---
# SpaRTAN: Spatial Reinforcement Token-based Aggregation Network for Visual Recognition 

**Title (ZH)**: SpaRTAN: 基于空间强化标记的聚合网络 för 视觉识别 

**Authors**: Quan Bi Pay, Vishnu Monn Baskaran, Junn Yong Loo, KokSheik Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2507.10999)  

**Abstract**: The resurgence of convolutional neural networks (CNNs) in visual recognition tasks, exemplified by ConvNeXt, has demonstrated their capability to rival transformer-based architectures through advanced training methodologies and ViT-inspired design principles. However, both CNNs and transformers exhibit a simplicity bias, favoring straightforward features over complex structural representations. Furthermore, modern CNNs often integrate MLP-like blocks akin to those in transformers, but these blocks suffer from significant information redundancies, necessitating high expansion ratios to sustain competitive performance. To address these limitations, we propose SpaRTAN, a lightweight architectural design that enhances spatial and channel-wise information processing. SpaRTAN employs kernels with varying receptive fields, controlled by kernel size and dilation factor, to capture discriminative multi-order spatial features effectively. A wave-based channel aggregation module further modulates and reinforces pixel interactions, mitigating channel-wise redundancies. Combining the two modules, the proposed network can efficiently gather and dynamically contextualize discriminative features. Experimental results in ImageNet and COCO demonstrate that SpaRTAN achieves remarkable parameter efficiency while maintaining competitive performance. In particular, on the ImageNet-1k benchmark, SpaRTAN achieves 77. 7% accuracy with only 3.8M parameters and approximately 1.0 GFLOPs, demonstrating its ability to deliver strong performance through an efficient design. On the COCO benchmark, it achieves 50.0% AP, surpassing the previous benchmark by 1.2% with only 21.5M parameters. The code is publicly available at [this https URL]. 

**Abstract (ZH)**: 卷积神经网络（CNNs）在视觉识别任务中的 resurgence，以 ConvNeXt 为例，证明了其通过先进的训练方法和受 ViT 启发的设计原则，能够与基于变换器的架构相媲美。然而，这两种方法都存在简化偏见，偏好简单的特征而非复杂的结构表示。此外，现代 CNN 经常整合类似于变换器中的 MLP 块，但这些块存在着显著的信息冗余，需要高的扩展比以维持竞争力的表现。为了解决这些限制，我们提出了 SpaRTAN，一种轻量级的架构设计，增强空间和通道级的信息处理。SpaRTAN 使用可由kernel大小和膨胀因子控制的具有不同接收域的 kernel，以有效地捕捉具有鉴别性的多阶空间特征。一种基于波的通道聚合模块进一步调节和强化像素间的交互，缓解通道级的冗余。结合这两个模块，提出的网络能够高效地收集和动态地上下文化鉴别特征。实验结果显示，SpaRTAN 在保持竞争力的同时实现了显著的参数效率，在 ImageNet 和 COCO 挑战中分别以仅 3.8M 参数和约 1.0 GFLOPs 达到 77.7% 的准确率，并在 COCO 挑战中以仅 21.5M 参数达到 50.0% 的 AP，超过了之前的最佳结果。相关的代码已公开发布。 

---
# Conceptualizing Multi-scale Wavelet Attention and Ray-based Encoding for Human-Object Interaction Detection 

**Title (ZH)**: 多尺度小波注意力与基于射线的编码对人体-物体交互检测的概念化 

**Authors**: Quan Bi Pay, Vishnu Monn Baskaran, Junn Yong Loo, KokSheik Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2507.10977)  

**Abstract**: Human-object interaction (HOI) detection is essential for accurately localizing and characterizing interactions between humans and objects, providing a comprehensive understanding of complex visual scenes across various domains. However, existing HOI detectors often struggle to deliver reliable predictions efficiently, relying on resource-intensive training methods and inefficient architectures. To address these challenges, we conceptualize a wavelet attention-like backbone and a novel ray-based encoder architecture tailored for HOI detection. Our wavelet backbone addresses the limitations of expressing middle-order interactions by aggregating discriminative features from the low- and high-order interactions extracted from diverse convolutional filters. Concurrently, the ray-based encoder facilitates multi-scale attention by optimizing the focus of the decoder on relevant regions of interest and mitigating computational overhead. As a result of harnessing the attenuated intensity of learnable ray origins, our decoder aligns query embeddings with emphasized regions of interest for accurate predictions. Experimental results on benchmark datasets, including ImageNet and HICO-DET, showcase the potential of our proposed architecture. The code is publicly available at [this https URL]. 

**Abstract (ZH)**: 人体-物体交互（HOI）检测对于准确定位和描述人类与物体之间的交互至关重要，有助于对各种领域中复杂视觉场景进行全面理解。然而，现有的HOI检测器往往难以高效地提供可靠的预测，依赖于资源密集型的训练方法和低效的网络结构。为了解决这些问题，我们提出了一种小波注意力_like主干和一种新的基于射线的编码器架构，专为HOI检测设计。我们的小波主干通过从多种卷积滤波器中提取的低阶和高阶交互中聚合 discriminative 特征，解决了表达中阶交互的局限性。同时，基于射线的编码器通过优化解码器对相关感兴趣区域的关注并减轻计算开销，实现了多尺度注意力。通过利用可学习射线原点的衰减强度，我们的解码器将查询嵌入与强调的感兴趣区域对齐，以实现准确的预测。在ImageNet和HICO-DET等基准数据集上的实验结果展示了我们提出架构的潜力。相关代码已在[此链接]公开。 

---
# A Lightweight and Robust Framework for Real-Time Colorectal Polyp Detection Using LOF-Based Preprocessing and YOLO-v11n 

**Title (ZH)**: 基于LOF预处理和YOLO-v11n的轻量级稳健实时结直肠息肉检测框架 

**Authors**: Saadat Behzadi, Danial Sharifrazi, Bita Mesbahzadeh, Javad Hassannataj Joloudarid, Roohallah Alizadehsani  

**Link**: [PDF](https://arxiv.org/pdf/2507.10864)  

**Abstract**: Objectives: Timely and accurate detection of colorectal polyps plays a crucial role in diagnosing and preventing colorectal cancer, a major cause of mortality worldwide. This study introduces a new, lightweight, and efficient framework for polyp detection that combines the Local Outlier Factor (LOF) algorithm for filtering noisy data with the YOLO-v11n deep learning model.
Study design: An experimental study leveraging deep learning and outlier removal techniques across multiple public datasets.
Methods: The proposed approach was tested on five diverse and publicly available datasets: CVC-ColonDB, CVC-ClinicDB, Kvasir-SEG, ETIS, and EndoScene. Since these datasets originally lacked bounding box annotations, we converted their segmentation masks into suitable detection labels. To enhance the robustness and generalizability of our model, we apply 5-fold cross-validation and remove anomalous samples using the LOF method configured with 30 neighbors and a contamination ratio of 5%. Cleaned data are then fed into YOLO-v11n, a fast and resource-efficient object detection architecture optimized for real-time applications. We train the model using a combination of modern augmentation strategies to improve detection accuracy under diverse conditions.
Results: Our approach significantly improves polyp localization performance, achieving a precision of 95.83%, recall of 91.85%, F1-score of 93.48%, mAP@0.5 of 96.48%, and mAP@0.5:0.95 of 77.75%. Compared to previous YOLO-based methods, our model demonstrates enhanced accuracy and efficiency.
Conclusions: These results suggest that the proposed method is well-suited for real-time colonoscopy support in clinical settings. Overall, the study underscores how crucial data preprocessing and model efficiency are when designing effective AI systems for medical imaging. 

**Abstract (ZH)**: 目的：及时准确地检测结肠息肉在诊断和预防结肠癌中起着关键作用，结肠癌是全球的主要死因之一。本研究介绍了一种新的轻量级高效息肉检测框架，该框架通过局部异常因子（LOF）算法过滤噪声数据并与YOLO-v11n深度学习模型相结合。

研究设计：利用深度学习和离群值去除技术，在多个公开数据集中进行的实验研究。

方法：所提出的方法在CVC-ColonDB、CVC-ClinicDB、Kvasir-SEG、ETIS和EndoScene这五个多样且公开可得的数据集上进行了测试。由于这些数据集原本缺乏边界框注释，我们将它们的分割掩膜转换为合适的检测标签。为了增强模型的鲁棒性和通用性，我们采用了5折交叉验证，并使用LOF方法（配置为30个邻居和污染比率为5%）移除异常样本。清洗后的数据被输入到YOLO-v11n中，这是一种为实时应用优化的快速且资源高效的物体检测架构。我们通过结合现代数据增强策略来训练模型，以在不同条件下提高检测准确性。

结果：我们的方法显著提高了息肉定位性能，实现了精度95.83%、召回率91.85%、F1分值93.48%、mAP@0.5 96.48%和mAP@0.5:0.95 77.75%。与之前的YOLO基线方法相比，我们的模型显示出更高的准确性和效率。

结论：这些结果表明，所提出的方法适用于临床环境中的实时结肠镜检查支持。总体而言，本研究强调了在设计有效的医疗成像AI系统时，数据预处理和模型效率的重要性。 

---
# Winsor-CAM: Human-Tunable Visual Explanations from Deep Networks via Layer-Wise Winsorization 

**Title (ZH)**: Winsor-CAM:通过层wise Winsorization的人类可调节视觉解释 

**Authors**: Casey Wall, Longwei Wang, Rodrigue Rizk, KC Santosh  

**Link**: [PDF](https://arxiv.org/pdf/2507.10846)  

**Abstract**: Interpreting the decision-making process of Convolutional Neural Networks (CNNs) is critical for deploying models in high-stakes domains. Gradient-weighted Class Activation Mapping (Grad-CAM) is a widely used method for visual explanations, yet it typically focuses on the final convolutional layer or naïvely averages across layers, strategies that can obscure important semantic cues or amplify irrelevant noise. We propose Winsor-CAM, a novel, human-tunable extension of Grad-CAM that generates robust and coherent saliency maps by aggregating information across all convolutional layers. To mitigate the influence of noisy or extreme attribution values, Winsor-CAM applies Winsorization, a percentile-based outlier attenuation technique. A user-controllable threshold allows for semantic-level tuning, enabling flexible exploration of model behavior across representational hierarchies. Evaluations on standard architectures (ResNet50, DenseNet121, VGG16, InceptionV3) using the PASCAL VOC 2012 dataset demonstrate that Winsor-CAM produces more interpretable heatmaps and achieves superior performance in localization metrics, including intersection-over-union and center-of-mass alignment, when compared to Grad-CAM and uniform layer-averaging baselines. Winsor-CAM advances the goal of trustworthy AI by offering interpretable, multi-layer insights with human-in-the-loop control. 

**Abstract (ZH)**: Winsor-CAM: 一种基于Winsor化的人工可控多层解释方法 

---
# A New Dataset and Performance Benchmark for Real-time Spacecraft Segmentation in Onboard Flight Computers 

**Title (ZH)**: 一种新的数据集及实时空间目标分割性能基准评估方法在机载飞行计算机中应用 

**Authors**: Jeffrey Joan Sam, Janhavi Sathe, Nikhil Chigali, Naman Gupta, Radhey Ruparel, Yicheng Jiang, Janmajay Singh, James W. Berck, Arko Barman  

**Link**: [PDF](https://arxiv.org/pdf/2507.10775)  

**Abstract**: Spacecraft deployed in outer space are routinely subjected to various forms of damage due to exposure to hazardous environments. In addition, there are significant risks to the subsequent process of in-space repairs through human extravehicular activity or robotic manipulation, incurring substantial operational costs. Recent developments in image segmentation could enable the development of reliable and cost-effective autonomous inspection systems. While these models often require large amounts of training data to achieve satisfactory results, publicly available annotated spacecraft segmentation data are very scarce. Here, we present a new dataset of nearly 64k annotated spacecraft images that was created using real spacecraft models, superimposed on a mixture of real and synthetic backgrounds generated using NASA's TTALOS pipeline. To mimic camera distortions and noise in real-world image acquisition, we also added different types of noise and distortion to the images. Finally, we finetuned YOLOv8 and YOLOv11 segmentation models to generate performance benchmarks for the dataset under well-defined hardware and inference time constraints to mimic real-world image segmentation challenges for real-time onboard applications in space on NASA's inspector spacecraft. The resulting models, when tested under these constraints, achieved a Dice score of 0.92, Hausdorff distance of 0.69, and an inference time of about 0.5 second. The dataset and models for performance benchmark are available at this https URL. 

**Abstract (ZH)**: 在外空间部署的航天器由于暴露在有害环境中，往往会遭受各种形式的损伤。此外，通过宇航员出舱活动或机器人操作进行在轨维修过程存在较高的风险，导致大量运营成本。最近图像分割技术的发展能够促进可靠且低成本的自主检测系统的开发。尽管这些模型通常需要大量训练数据才能达到满意的结果，但公开可用的标注航天器分割数据极为稀缺。在此，我们使用真实航天器模型，并结合NASA TTALOS管道生成的现实和合成背景混合图像，创建了一个近64,000张标注航天器图像的新数据集。为了模拟实际图像获取中的相机失真和噪声，我们还向图像中添加了不同类型的噪声和失真。最后，我们针对特定硬件和推断时间约束对YOLOv8和YOLOv11分割模型进行了微调，以生成该数据集下的性能基准，模拟NASA检查员航天器在太空中实时应用时的实际图像分割挑战。在这些约束条件下进行测试后，所得模型的Dice分数为0.92，Hausdorff距离为0.69，推理时间为约0.5秒。该数据集和用于性能基准的模型可在以下链接获取：this https URL。 

---
# Comparative Analysis of Vision Transformers and Traditional Deep Learning Approaches for Automated Pneumonia Detection in Chest X-Rays 

**Title (ZH)**: 基于胸部X光片的自动化肺炎检测：视觉-transformer与传统深度学习方法的 comparative analysis 

**Authors**: Gaurav Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.10589)  

**Abstract**: Pneumonia, particularly when induced by diseases like COVID-19, remains a critical global health challenge requiring rapid and accurate diagnosis. This study presents a comprehensive comparison of traditional machine learning and state-of-the-art deep learning approaches for automated pneumonia detection using chest X-rays (CXRs). We evaluate multiple methodologies, ranging from conventional machine learning techniques (PCA-based clustering, Logistic Regression, and Support Vector Classification) to advanced deep learning architectures including Convolutional Neural Networks (Modified LeNet, DenseNet-121) and various Vision Transformer (ViT) implementations (Deep-ViT, Compact Convolutional Transformer, and Cross-ViT). Using a dataset of 5,856 pediatric CXR images, we demonstrate that Vision Transformers, particularly the Cross-ViT architecture, achieve superior performance with 88.25% accuracy and 99.42% recall, surpassing traditional CNN approaches. Our analysis reveals that architectural choices impact performance more significantly than model size, with Cross-ViT's 75M parameters outperforming larger models. The study also addresses practical considerations including computational efficiency, training requirements, and the critical balance between precision and recall in medical diagnostics. Our findings suggest that Vision Transformers offer a promising direction for automated pneumonia detection, potentially enabling more rapid and accurate diagnosis during health crises. 

**Abstract (ZH)**: 肺炎，尤其是由COVID-19等疾病引起的肺炎，仍然是一个关键的全球健康挑战，需要快速而准确的诊断。本研究全面比较了传统机器学习方法和最新的深度学习方法在胸部X光片（CXR）上自动检测肺炎的应用。我们评估了多种方法，从传统的机器学习技术（基于PCA的聚类、逻辑回归和支持向量分类）到先进的深度学习架构（包括修正后的LeNet、DenseNet-121），以及多种视觉变换器（ViT）实现（Deep-ViT、紧凑卷积变换器和Cross-ViT）。使用包含5,856张儿科CXR图像的数据集，我们展示了视觉变换器，尤其是Cross-ViT架构，取得了88.25%的准确率和99.42%的召回率，超越了传统的CNN方法。我们的分析表明，架构选择比模型大小对性能影响更大，Cross-ViT的75M参数在性能上超过了更大的模型。本研究还讨论了计算效率、训练需求以及医学诊断中精确度和召回率之间的关键平衡。我们的研究结果表明，视觉变换器为自动肺炎检测提供了有前途的方向，可能在健康危机期间实现更快更准确的诊断。 

---
