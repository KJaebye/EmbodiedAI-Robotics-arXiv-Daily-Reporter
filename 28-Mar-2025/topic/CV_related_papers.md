# Enhancing Underwater Navigation through Cross-Correlation-Aware Deep INS/DVL Fusion 

**Title (ZH)**: 基于交叉相关性意识的深耦合INS/DVL融合 underwater navigation enhancement 

**Authors**: Nadav Cohen, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2503.21727)  

**Abstract**: The accurate navigation of autonomous underwater vehicles critically depends on the precision of Doppler velocity log (DVL) velocity measurements. Recent advancements in deep learning have demonstrated significant potential in improving DVL outputs by leveraging spatiotemporal dependencies across multiple sensor modalities. However, integrating these estimates into model-based filters, such as the extended Kalman filter, introduces statistical inconsistencies, most notably, cross-correlations between process and measurement noise. This paper addresses this challenge by proposing a cross-correlation-aware deep INS/DVL fusion framework. Building upon BeamsNet, a convolutional neural network designed to estimate AUV velocity using DVL and inertial data, we integrate its output into a navigation filter that explicitly accounts for the cross-correlation induced between the noise sources. This approach improves filter consistency and better reflects the underlying sensor error structure. Evaluated on two real-world underwater trajectories, the proposed method outperforms both least squares and cross-correlation-neglecting approaches in terms of state uncertainty. Notably, improvements exceed 10% in velocity and misalignment angle confidence metrics. Beyond demonstrating empirical performance, this framework provides a theoretically principled mechanism for embedding deep learning outputs within stochastic filters. 

**Abstract (ZH)**: 精确自主水下车辆的导航依赖于Doppler速度计(DVL)速度测量的精确性。基于深度学习的 Recent 进展展示了通过利用多传感器模态的空间时间依赖性来改进 DVL 输出的巨大潜力。然而，将这些估计值整合到模型滤波器中，如扩展卡尔曼滤波器中，会引入统计不一致性，尤其是在过程噪声和测量噪声之间的交叉相关性。本文通过提出一种考虑交叉相关的深度 INS/DVL 融合框架来应对这一挑战。在此基础上，我们利用为利用 DVL 和惯性数据估计 AUV 速度而设计的 BeamsNet 卷积神经网络的输出，将其整合进一个导航滤波器中，该滤波器明确考虑了由噪声源引起的交叉相关性。该方法提高了滤波器的一致性，更好地反映了传感器误差结构。在两个实际水下轨迹上的评估表明，与最小二乘法和忽视交叉相关性的方法相比，所提出的方法在状态不确定性上表现出更佳性能。特别是在速度和偏移角度的置信度指标上，改进超过10%。除了展示实证性能外，该框架还提供了一种在随机滤波器中嵌入深度学习输出的理论基础机制。 

---
# UGNA-VPR: A Novel Training Paradigm for Visual Place Recognition Based on Uncertainty-Guided NeRF Augmentation 

**Title (ZH)**: UGNA-VPR：基于不确定性引导的NeRF增强的视觉地方识别新型训练范式 

**Authors**: Yehui Shen, Lei Zhang, Qingqiu Li, Xiongwei Zhao, Yue Wang, Huimin Lu, Xieyuanli Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.21338)  

**Abstract**: Visual place recognition (VPR) is crucial for robots to identify previously visited locations, playing an important role in autonomous navigation in both indoor and outdoor environments. However, most existing VPR datasets are limited to single-viewpoint scenarios, leading to reduced recognition accuracy, particularly in multi-directional driving or feature-sparse scenes. Moreover, obtaining additional data to mitigate these limitations is often expensive. This paper introduces a novel training paradigm to improve the performance of existing VPR networks by enhancing multi-view diversity within current datasets through uncertainty estimation and NeRF-based data augmentation. Specifically, we initially train NeRF using the existing VPR dataset. Then, our devised self-supervised uncertainty estimation network identifies places with high uncertainty. The poses of these uncertain places are input into NeRF to generate new synthetic observations for further training of VPR networks. Additionally, we propose an improved storage method for efficient organization of augmented and original training data. We conducted extensive experiments on three datasets and tested three different VPR backbone networks. The results demonstrate that our proposed training paradigm significantly improves VPR performance by fully utilizing existing data, outperforming other training approaches. We further validated the effectiveness of our approach on self-recorded indoor and outdoor datasets, consistently demonstrating superior results. Our dataset and code have been released at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 视觉场所识别（VPR）对于机器人识别之前访问的位置至关重要，对于室内外环境的自主导航起着重要作用。然而，现有的大多数VPR数据集局限于单视角场景，导致识别准确性降低，特别是在多方向驾驶或特征稀疏场景中。此外，获得额外数据以应对这些局限性往往代价昂贵。本文提出了一种新的训练 paradigm，通过在当前数据集中增强多视角多样性来提高现有VPR网络的性能，这种方法利用不确定性估计和基于NeRF的数据增强。我们首先使用现有VPR数据集训练NeRF。然后，我们设计的自监督不确定性估计网络识别出高不确定性的地方。将这些不确定地点的姿态输入NeRF生成新的合成观测数据，进一步用于VPR网络的训练。此外，我们提出了改进的数据存储方法，以高效组织增强和原始训练数据。我们在三个数据集上进行了广泛的实验，并测试了三种不同的VPR骨干网络。结果表明，我们提出的训练 paradigm 显着提高了VPR性能，充分利用了现有数据，优于其他训练方法。我们在自行录制的室内外数据集上进一步验证了我们方法的有效性，结果一致表现出色。我们的数据集和代码已发布在 \href{this https URL}{this https URL}。 

---
# Stable-SCore: A Stable Registration-based Framework for 3D Shape Correspondence 

**Title (ZH)**: 稳定注册框架下的3D形状对应：Stable-SCore 

**Authors**: Haolin Liu, Xiaohang Zhan, Zizheng Yan, Zhongjin Luo, Yuxin Wen, Xiaoguang Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.21766)  

**Abstract**: Establishing character shape correspondence is a critical and fundamental task in computer vision and graphics, with diverse applications including re-topology, attribute transfer, and shape interpolation. Current dominant functional map methods, while effective in controlled scenarios, struggle in real situations with more complex challenges such as non-isometric shape discrepancies. In response, we revisit registration-for-correspondence methods and tap their potential for more stable shape correspondence estimation. To overcome their common issues including unstable deformations and the necessity for careful pre-alignment or high-quality initial 3D correspondences, we introduce Stable-SCore: A Stable Registration-based Framework for 3D Shape Correspondence. We first re-purpose a foundation model for 2D character correspondence that ensures reliable and stable 2D mappings. Crucially, we propose a novel Semantic Flow Guided Registration approach that leverages 2D correspondence to guide mesh deformations. Our framework significantly surpasses existing methods in challenging scenarios, and brings possibilities for a wide array of real applications, as demonstrated in our results. 

**Abstract (ZH)**: 建立字符形状对应关系是计算机视觉和图形学中一个关键且基础的任务，广泛应用于重新拓扑、属性迁移和形状插值等领域。针对当前主导的功能映射方法在应对非等参形变等复杂挑战时的局限性，我们重新审视了基于注册的对应方法，并挖掘其潜在优势以获得更稳定的形状对应估计。为克服其常见的不稳定形变和需要精细预对齐或高质量初值3D对应等问题，我们提出了一种稳定的基于注册的3D形状对应框架——Stable-SCore。我们首先将一个基础模型重新用于2D字符对应，确保可靠的2D映射。关键的是，我们提出了一种基于语义流的注册方法，利用2D对应关系指导网格变形。我们的框架在挑战性场景中显著超越了现有方法，并为广泛的实际应用提供了可能性，如我们在实验结果中所展示的。 

---
# Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video 

**Title (ZH)**: Uni4D: 统一单视频来源的4D建模视觉基础模型 

**Authors**: David Yifan Yao, Albert J. Zhai, Shenlong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21761)  

**Abstract**: This paper presents a unified approach to understanding dynamic scenes from casual videos. Large pretrained vision foundation models, such as vision-language, video depth prediction, motion tracking, and segmentation models, offer promising capabilities. However, training a single model for comprehensive 4D understanding remains challenging. We introduce Uni4D, a multi-stage optimization framework that harnesses multiple pretrained models to advance dynamic 3D modeling, including static/dynamic reconstruction, camera pose estimation, and dense 3D motion tracking. Our results show state-of-the-art performance in dynamic 4D modeling with superior visual quality. Notably, Uni4D requires no retraining or fine-tuning, highlighting the effectiveness of repurposing visual foundation models for 4D understanding. 

**Abstract (ZH)**: 本文提出了一种统一方法，以理解来自随手拍摄视频中的动态场景。大型预训练视觉基础模型，如视觉语言模型、视频深度预测模型、运动跟踪模型和分割模型提供了有力的能力。然而，为全面的4D理解训练单一模型仍然具有挑战性。我们引入了Uni4D，这是一种多阶段优化框架，利用多个预训练模型推进动态3D建模，包括静态/动态重建、相机姿态估计和密集3D运动跟踪。我们的结果表明，Uni4D在动态4D建模方面取得了最先进的性能，并且具有更优秀的视觉质量。值得注意的是，Uni4D无需重新训练或微调，突显了重新利用视觉基础模型进行4D理解的有效性。 

---
# Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation without 3D Data 

**Title (ZH)**: 逐级渲染精炼：adapt Stable Diffusion 以实现无需3D数据的即时文本到网格生成 

**Authors**: Zhiyuan Ma, Xinyue Liang, Rongyuan Wu, Xiangyu Zhu, Zhen Lei, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21694)  

**Abstract**: It is highly desirable to obtain a model that can generate high-quality 3D meshes from text prompts in just seconds. While recent attempts have adapted pre-trained text-to-image diffusion models, such as Stable Diffusion (SD), into generators of 3D representations (e.g., Triplane), they often suffer from poor quality due to the lack of sufficient high-quality 3D training data. Aiming at overcoming the data shortage, we propose a novel training scheme, termed as Progressive Rendering Distillation (PRD), eliminating the need for 3D ground-truths by distilling multi-view diffusion models and adapting SD into a native 3D generator. In each iteration of training, PRD uses the U-Net to progressively denoise the latent from random noise for a few steps, and in each step it decodes the denoised latent into 3D output. Multi-view diffusion models, including MVDream and RichDreamer, are used in joint with SD to distill text-consistent textures and geometries into the 3D outputs through score distillation. Since PRD supports training without 3D ground-truths, we can easily scale up the training data and improve generation quality for challenging text prompts with creative concepts. Meanwhile, PRD can accelerate the inference speed of the generation model in just a few steps. With PRD, we train a Triplane generator, namely TriplaneTurbo, which adds only $2.5\%$ trainable parameters to adapt SD for Triplane generation. TriplaneTurbo outperforms previous text-to-3D generators in both efficiency and quality. Specifically, it can produce high-quality 3D meshes in 1.2 seconds and generalize well for challenging text input. The code is available at this https URL. 

**Abstract (ZH)**: 高质异地图生成：一种渐进渲染蒸馏方法 

---
# AlignDiff: Learning Physically-Grounded Camera Alignment via Diffusion 

**Title (ZH)**: AlignDiff: 学习基于物理的相机对准通过扩散 

**Authors**: Liuyue Xie, Jiancong Guo, Ozan Cakmakci, Andre Araujo, Laszlo A. Jeni, Zhiheng Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.21581)  

**Abstract**: Accurate camera calibration is a fundamental task for 3D perception, especially when dealing with real-world, in-the-wild environments where complex optical distortions are common. Existing methods often rely on pre-rectified images or calibration patterns, which limits their applicability and flexibility. In this work, we introduce a novel framework that addresses these challenges by jointly modeling camera intrinsic and extrinsic parameters using a generic ray camera model. Unlike previous approaches, AlignDiff shifts focus from semantic to geometric features, enabling more accurate modeling of local distortions. We propose AlignDiff, a diffusion model conditioned on geometric priors, enabling the simultaneous estimation of camera distortions and scene geometry. To enhance distortion prediction, we incorporate edge-aware attention, focusing the model on geometric features around image edges, rather than semantic content. Furthermore, to enhance generalizability to real-world captures, we incorporate a large database of ray-traced lenses containing over three thousand samples. This database characterizes the distortion inherent in a diverse variety of lens forms. Our experiments demonstrate that the proposed method significantly reduces the angular error of estimated ray bundles by ~8.2 degrees and overall calibration accuracy, outperforming existing approaches on challenging, real-world datasets. 

**Abstract (ZH)**: 准确的相机标定是三维感知的基础任务，尤其是在处理复杂光学畸变常见的自然环境时。现有方法往往依赖于预校正图像或校准图案，这限制了它们的适用性和灵活性。在本文中，我们提出了一种新的框架，通过使用通用射线相机模型联合建模相机固有参数和外部参数来解决这些挑战。与之前的 Approaches 不同，AlignDiff 转而关注几何特征，从而更准确地建模局部畸变。我们提出了一种条件于几何先验的扩散模型 AlignDiff，能够同时估计相机畸变和场景几何。为了增强畸变预测，我们融入了边缘感知注意力机制，使模型专注于图像边缘周围的几何特征，而非语义内容。此外，为了增强对自然环境捕捉的泛化能力，我们引入了一个包含三千多个样本的大型射线追踪镜头数据库。该数据库表征了各种镜头形式固有的畸变特征。我们的实验表明，所提出的方法能够显著降低估计射线束的角度误差 (~8.2 度) 和整体标定精度，在挑战性的现实世界数据集上优于现有方法。 

---
# Retinal Fundus Multi-Disease Image Classification using Hybrid CNN-Transformer-Ensemble Architectures 

**Title (ZH)**: 基于混合CNN-Transformer-集成架构的视网膜 fundus 多病种图像分类 

**Authors**: Deependra Singh, Saksham Agarwal, Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2503.21465)  

**Abstract**: Our research is motivated by the urgent global issue of a large population affected by retinal diseases, which are evenly distributed but underserved by specialized medical expertise, particularly in non-urban areas. Our primary objective is to bridge this healthcare gap by developing a comprehensive diagnostic system capable of accurately predicting retinal diseases solely from fundus images. However, we faced significant challenges due to limited, diverse datasets and imbalanced class distributions. To overcome these issues, we have devised innovative strategies. Our research introduces novel approaches, utilizing hybrid models combining deeper Convolutional Neural Networks (CNNs), Transformer encoders, and ensemble architectures sequentially and in parallel to classify retinal fundus images into 20 disease labels. Our overarching goal is to assess these advanced models' potential in practical applications, with a strong focus on enhancing retinal disease diagnosis accuracy across a broader spectrum of conditions. Importantly, our efforts have surpassed baseline model results, with the C-Tran ensemble model emerging as the leader, achieving a remarkable model score of 0.9166, surpassing the baseline score of 0.9. Additionally, experiments with the IEViT model showcased equally promising outcomes with improved computational efficiency. We've also demonstrated the effectiveness of dynamic patch extraction and the integration of domain knowledge in computer vision tasks. In summary, our research strives to contribute significantly to retinal disease diagnosis, addressing the critical need for accessible healthcare solutions in underserved regions while aiming for comprehensive and accurate disease prediction. 

**Abstract (ZH)**: 我们的研究受到全球数百万人患视网膜疾病这一迫切问题的启发，这些疾病分布均衡但缺乏专科医疗 expertise，尤其是在非都市区域。我们的主要目标是通过开发一种全面的诊断系统来弥合这一医疗缺口，该系统仅通过视网膜Fundus图像即可准确预测视网膜疾病。然而，由于受限于有限且多样化的数据集以及类别分布不平衡的问题，我们面临了重大挑战。为克服这些问题，我们提出了一些创新策略。我们的研究引入了新颖的方法，利用混合模型结合更深的卷积神经网络（CNNs）、Transformer编码器以及按序列和并行方式构建的集成架构，将视网膜Fundus图像分类为20种疾病标签。我们的总体目标是评估这些先进模型在实际应用中的潜力，特别注重提高各类疾病诊断的准确性。重要的是，我们的努力超越了基准模型的结果，C-Tran集成模型脱颖而出，实现了0.9166的卓越模型得分，超过了基准得分0.9。此外，IEViT模型的实验同样取得了令人鼓舞的结果，展示了更好的计算效率。我们还展示了动态补丁提取和在计算机视觉任务中集成领域知识的有效性。总之，我们的研究致力于在视网膜疾病诊断领域作出重大贡献，旨在为缺乏医疗服务的地区提供可及的健康解决方案，同时着眼于实现全面和准确的疾病预测。 

---
# Multi-Scale Invertible Neural Network for Wide-Range Variable-Rate Learned Image Compression 

**Title (ZH)**: 多尺度可逆神经网络用于宽动态范围可变率学习图像压缩 

**Authors**: Hanyue Tu, Siqi Wu, Li Li, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21284)  

**Abstract**: Autoencoder-based structures have dominated recent learned image compression methods. However, the inherent information loss associated with autoencoders limits their rate-distortion performance at high bit rates and restricts their flexibility of rate adaptation. In this paper, we present a variable-rate image compression model based on invertible transform to overcome these limitations. Specifically, we design a lightweight multi-scale invertible neural network, which bijectively maps the input image into multi-scale latent representations. To improve the compression efficiency, a multi-scale spatial-channel context model with extended gain units is devised to estimate the entropy of the latent representation from high to low levels. Experimental results demonstrate that the proposed method achieves state-of-the-art performance compared to existing variable-rate methods, and remains competitive with recent multi-model approaches. Notably, our method is the first learned image compression solution that outperforms VVC across a very wide range of bit rates using a single model, especially at high bit this http URL source code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于可逆变换的变率图像压缩模型：克服自动编码器的固有局限 

---
# GenFusion: Closing the Loop between Reconstruction and Generation via Videos 

**Title (ZH)**: GenFusion: 通过视频在重建与生成之间形成闭环 

**Authors**: Sibo Wu, Congrong Xu, Binbin Huang, Andreas Geiger, Anpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.21219)  

**Abstract**: Recently, 3D reconstruction and generation have demonstrated impressive novel view synthesis results, achieving high fidelity and efficiency. However, a notable conditioning gap can be observed between these two fields, e.g., scalable 3D scene reconstruction often requires densely captured views, whereas 3D generation typically relies on a single or no input view, which significantly limits their applications. We found that the source of this phenomenon lies in the misalignment between 3D constraints and generative priors. To address this problem, we propose a reconstruction-driven video diffusion model that learns to condition video frames on artifact-prone RGB-D renderings. Moreover, we propose a cyclical fusion pipeline that iteratively adds restoration frames from the generative model to the training set, enabling progressive expansion and addressing the viewpoint saturation limitations seen in previous reconstruction and generation pipelines. Our evaluation, including view synthesis from sparse view and masked input, validates the effectiveness of our approach. 

**Abstract (ZH)**: 近期，3D重建与生成展示了令人印象深刻的新型视图合成结果，实现了高保真度和效率。然而，这两个领域之间存在显著的条件差异，例如，可扩展的3D场景重建通常需要密集捕获的视角，而3D生成通常依赖于单个或没有输入视角，这大大限制了它们的应用。我们发现，这一现象源于3D约束与生成先验之间的不匹配。为解决这一问题，我们提出了一种以重建为导向的视频扩散模型，该模型学习在易产生伪影的RGB-D渲染上条件化视频帧。此外，我们提出了一种循环融合管道，该管道迭代地将生成模型的修复帧添加到训练集，从而实现渐进扩展并解决先前重建和生成管道中视角饱和的限制。我们的评估，包括从稀疏视角和带掩码输入的视图合成，验证了我们方法的有效性。 

---
# The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation 

**Title (ZH)**: 低级特征藏玄机：跨域少量样本分割 

**Authors**: Yuhan Liu, Yixiong Zou, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21150)  

**Abstract**: Cross-Domain Few-Shot Segmentation (CDFSS) is proposed to transfer the pixel-level segmentation capabilities learned from large-scale source-domain datasets to downstream target-domain datasets, with only a few annotated images per class. In this paper, we focus on a well-observed but unresolved phenomenon in CDFSS: for target domains, particularly those distant from the source domain, segmentation performance peaks at the very early epochs, and declines sharply as the source-domain training proceeds. We delve into this phenomenon for an interpretation: low-level features are vulnerable to domain shifts, leading to sharper loss landscapes during the source-domain training, which is the devil of CDFSS. Based on this phenomenon and interpretation, we further propose a method that includes two plug-and-play modules: one to flatten the loss landscapes for low-level features during source-domain training as a novel sharpness-aware minimization method, and the other to directly supplement target-domain information to the model during target-domain testing by low-level-based calibration. Extensive experiments on four target datasets validate our rationale and demonstrate that our method surpasses the state-of-the-art method in CDFSS signifcantly by 3.71% and 5.34% average MIoU in 1-shot and 5-shot scenarios, respectively. 

**Abstract (ZH)**: 跨域少样本分割（CDFSS） 

---
# Rerouting Connection: Hybrid Computer Vision Analysis Reveals Visual Similarity Between Indus and Tibetan-Yi Corridor Writing Systems 

**Title (ZH)**: 重定向连接：混合计算机视觉分析揭示印度河与藏彝走廊书写系统之间的视觉相似性 

**Authors**: Ooha Lakkadi Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.21074)  

**Abstract**: This thesis employs a hybrid CNN-Transformer architecture, in conjunction with a detailed anthropological framework, to investigate potential historical connections between the visual morphology of the Indus Valley script and pictographic systems of the Tibetan-Yi Corridor. Through an ensemble methodology of three target scripts across 15 independently trained models, we demonstrate that Tibetan-Yi Corridor scripts exhibit approximately six-fold higher visual similarity to the Indus script (61.7%-63.5%) than to the Bronze Age Proto-Cuneiform (10.2%-10.9%) or Proto-Elamite (7.6%-8.7%) systems. Additionally and contrarily to our current understanding of the networks of the Indus Valley Civilization, the Indus script unexpectedly maps closer to Tibetan-Yi Corridor scripts, with a mean cosine similarity of 0.629, than to the aforementioned contemporaneous West Asian signaries, both of which recorded mean cosine similarities of 0.104 and 0.080 despite their close geographic proximity and evident trade relations. Across various dimensionality reduction practices and clustering methodologies, the Indus script consistently clusters closest to Tibetan-Yi Corridor scripts. Our computational results align with qualitative observations of specific pictorial parallels in numeral systems, gender markers, and key iconographic elements; this is further supported by archaeological evidence of sustained contact networks along the ancient Shu-Shendu road in tandem with the Indus Valley Civilization's decline, providing a plausible transmission pathway. While alternative explanations cannot be ruled out, the specificity and consistency of observed similarities challenge conventional narratives of isolated script development and suggest more complex ancient cultural transmission networks between South and East Asia than previously recognized. 

**Abstract (ZH)**: 本论文采用混合CNN-Transformer架构，结合详细的人类学框架，探讨印度河谷文字与西藏-彝走廊象形系统的视觉形态之间潜在的历史联系。通过针对三个目标文字训练的15个独立模型的集成方法，我们证明了西藏-彝走廊文字与印度河谷文字的视觉相似度大约高六倍（61.7%-63.5%），远高于青铜时代楔形文字初型（10.2%-10.9%）或埃兰文字初型（7.6%-8.7%）系统。此外，与我们对印度河文明网络的理解相反，印度河谷文字意外地与西藏-彝走廊文字更接近，平均余弦相似度为0.629，而上述同时期的西亚洲符号系统分别为0.104和0.080，尽管它们地理上接近且存在贸易关系。在各种降维实践和聚类方法中，印度河谷文字始终与西藏-彝走廊文字聚类最近。我们的计算结果与特定数字符号、性别标志和关键图象学元素的定性观察相符；这进一步得到了古代松都路沿线持续接触网络以及印度河谷文明衰落的考古证据的支持，提供了可能的传播途径。虽然可以提出替代解释，但观察到的特定和一致的相似性挑战了孤立文字发展的传统叙述，并暗示南亚与东亚之间比以前认识的更加复杂的文化传播网络。 

---
# LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos 

**Title (ZH)**: LATTE-MV: 从单目视频学习预测乒乓球击球动作 

**Authors**: Daniel Etaat, Dvij Kalaria, Nima Rahmanian, Shankar Sastry  

**Link**: [PDF](https://arxiv.org/pdf/2503.20936)  

**Abstract**: Physical agility is a necessary skill in competitive table tennis, but by no means sufficient. Champions excel in this fast-paced and highly dynamic environment by anticipating their opponent's intent - buying themselves the necessary time to react. In this work, we take one step towards designing such an anticipatory agent. Previous works have developed systems capable of real-time table tennis gameplay, though they often do not leverage anticipation. Among the works that forecast opponent actions, their approaches are limited by dataset size and variety. Our paper contributes (1) a scalable system for reconstructing monocular video of table tennis matches in 3D and (2) an uncertainty-aware controller that anticipates opponent actions. We demonstrate in simulation that our policy improves the ball return rate against high-speed hits from 49.9% to 59.0% as compared to a baseline non-anticipatory policy. 

**Abstract (ZH)**: 物理敏捷性是竞争性乒乓球比赛中的一项必要技能，但绝非充分条件。冠军在这一快速且高度动态的环境中表现出色，通过预测对手的意图为自己争取必要的反应时间。在这项工作中，我们朝着设计这样一种预见性代理迈出了一步。先前的研究开发了能够实现实时乒乓球游戏的系统，尽管它们往往没有利用预见性。在预测对手动作的研究中，这些方法受限于数据集的大小和多样性。我们的论文贡献了（1）一个可扩展的系统，可以重建3D乒乓球比赛的单目视频，以及（2）一种具备不确定性意识的控制器，能够预见对手的动作。我们通过模拟证明，与基准的非预见性策略相比，我们的策略在面对高速击球时的还击率从49.9%提高到了59.0%。 

---
# VinaBench: Benchmark for Faithful and Consistent Visual Narratives 

**Title (ZH)**: VinaBench：视觉叙述忠实性和一致性基准 

**Authors**: Silin Gao, Sheryl Mathew, Li Mi, Sepideh Mamooler, Mengjie Zhao, Hiromi Wakaki, Yuki Mitsufuji, Syrielle Montariol, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2503.20871)  

**Abstract**: Visual narrative generation transforms textual narratives into sequences of images illustrating the content of the text. However, generating visual narratives that are faithful to the input text and self-consistent across generated images remains an open challenge, due to the lack of knowledge constraints used for planning the stories. In this work, we propose a new benchmark, VinaBench, to address this challenge. Our benchmark annotates the underlying commonsense and discourse constraints in visual narrative samples, offering systematic scaffolds for learning the implicit strategies of visual storytelling. Based on the incorporated narrative constraints, we further propose novel metrics to closely evaluate the consistency of generated narrative images and the alignment of generations with the input textual narrative. Our results across three generative vision models demonstrate that learning with VinaBench's knowledge constraints effectively improves the faithfulness and cohesion of generated visual narratives. 

**Abstract (ZH)**: 视觉叙事生成将文本叙事转换为展现文本内容的图像序列。然而，生成既忠实于输入文本又能跨图像保持一致性的视觉叙事仍然是一个开放性的挑战，这主要是由于缺乏用于规划故事的知识约束。在此工作中，我们提出一个新的基准VinaBench以应对这一挑战。我们的基准标注了视觉叙事样本中的底层常识和话语约束，为学习视觉叙事中的隐含策略提供系统性框架。基于纳入的故事约束，我们进一步提出新的评估指标，以更紧密地评估生成叙事图像的一致性及生成与输入文本叙事的一致性。我们的结果表明，在VinaBench知识约束下学习能够有效提高生成的视觉叙事的真实性和连贯性。 

---
# Exploiting Temporal State Space Sharing for Video Semantic Segmentation 

**Title (ZH)**: 利用时空状态空间共享进行视频语义分割 

**Authors**: Syed Ariff Syed Hesham, Yun Liu, Guolei Sun, Henghui Ding, Jing Yang, Ender Konukoglu, Xue Geng, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20824)  

**Abstract**: Video semantic segmentation (VSS) plays a vital role in understanding the temporal evolution of scenes. Traditional methods often segment videos frame-by-frame or in a short temporal window, leading to limited temporal context, redundant computations, and heavy memory requirements. To this end, we introduce a Temporal Video State Space Sharing (TV3S) architecture to leverage Mamba state space models for temporal feature sharing. Our model features a selective gating mechanism that efficiently propagates relevant information across video frames, eliminating the need for a memory-heavy feature pool. By processing spatial patches independently and incorporating shifted operation, TV3S supports highly parallel computation in both training and inference stages, which reduces the delay in sequential state space processing and improves the scalability for long video sequences. Moreover, TV3S incorporates information from prior frames during inference, achieving long-range temporal coherence and superior adaptability to extended sequences. Evaluations on the VSPW and Cityscapes datasets reveal that our approach outperforms current state-of-the-art methods, establishing a new standard for VSS with consistent results across long video sequences. By achieving a good balance between accuracy and efficiency, TV3S shows a significant advancement in spatiotemporal modeling, paving the way for efficient video analysis. The code is publicly available at this https URL. 

**Abstract (ZH)**: 基于Mamba状态空间模型的时空视频状态空间共享架构在视频语义分割中的应用 

---
# Synthetic Video Enhances Physical Fidelity in Video Synthesis 

**Title (ZH)**: 合成视频在视频合成中增强物理保真度 

**Authors**: Qi Zhao, Xingyu Ni, Ziyu Wang, Feng Cheng, Ziyan Yang, Lu Jiang, Bohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20822)  

**Abstract**: We investigate how to enhance the physical fidelity of video generation models by leveraging synthetic videos derived from computer graphics pipelines. These rendered videos respect real-world physics, such as maintaining 3D consistency, and serve as a valuable resource that can potentially improve video generation models. To harness this potential, we propose a solution that curates and integrates synthetic data while introducing a method to transfer its physical realism to the model, significantly reducing unwanted artifacts. Through experiments on three representative tasks emphasizing physical consistency, we demonstrate its efficacy in enhancing physical fidelity. While our model still lacks a deep understanding of physics, our work offers one of the first empirical demonstrations that synthetic video enhances physical fidelity in video synthesis. Website: this https URL 

**Abstract (ZH)**: 我们探究如何通过利用源自计算机图形管道的合成视频来增强视频生成模型的物理保真度。这些渲染视频遵循现实世界的物理法则，如保持三维一致性，并且可以作为有价值的资源，潜在地提高视频生成模型的性能。为充分利用这一潜力，我们提出了一个解决方案，以整理和整合合成数据，并引入了一种方法将其实验现实性转移到模型中，从而显著减少不必要的伪影。通过在三个强调物理一致性的代表性任务上的实验，我们展示了其在增强物理保真度方面的有效性。尽管我们的模型尚未对物理现象有深入的理解，但我们的工作提供了第一个实证证明合成视频在视频合成中增强物理保真度的示例。 

---
