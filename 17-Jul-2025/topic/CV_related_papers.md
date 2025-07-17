# UniLGL: Learning Uniform Place Recognition for FOV-limited/Panoramic LiDAR Global Localization 

**Title (ZH)**: UniLGL：学习统一视角限制/全景LiDAR全局定位中的场地识别 

**Authors**: Hongming Shen, Xun Chen, Yulin Hui, Zhenyu Wu, Wei Wang, Qiyang Lyu, Tianchen Deng, Danwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12194)  

**Abstract**: Existing LGL methods typically consider only partial information (e.g., geometric features) from LiDAR observations or are designed for homogeneous LiDAR sensors, overlooking the uniformity in LGL. In this work, a uniform LGL method is proposed, termed UniLGL, which simultaneously achieves spatial and material uniformity, as well as sensor-type uniformity. The key idea of the proposed method is to encode the complete point cloud, which contains both geometric and material information, into a pair of BEV images (i.e., a spatial BEV image and an intensity BEV image). An end-to-end multi-BEV fusion network is designed to extract uniform features, equipping UniLGL with spatial and material uniformity. To ensure robust LGL across heterogeneous LiDAR sensors, a viewpoint invariance hypothesis is introduced, which replaces the conventional translation equivariance assumption commonly used in existing LPR networks and supervises UniLGL to achieve sensor-type uniformity in both global descriptors and local feature representations. Finally, based on the mapping between local features on the 2D BEV image and the point cloud, a robust global pose estimator is derived that determines the global minimum of the global pose on SE(3) without requiring additional registration. To validate the effectiveness of the proposed uniform LGL, extensive benchmarks are conducted in real-world environments, and the results show that the proposed UniLGL is demonstratively competitive compared to other State-of-the-Art LGL methods. Furthermore, UniLGL has been deployed on diverse platforms, including full-size trucks and agile Micro Aerial Vehicles (MAVs), to enable high-precision localization and mapping as well as multi-MAV collaborative exploration in port and forest environments, demonstrating the applicability of UniLGL in industrial and field scenarios. 

**Abstract (ZH)**: 一种同时实现空间均匀性、材料均匀性和传感器类型均匀性的统一LiDAR全局定位方法 

---
# IANN-MPPI: Interaction-Aware Neural Network-Enhanced Model Predictive Path Integral Approach for Autonomous Driving 

**Title (ZH)**: IANN-MPPI: 具有交互意识的神经网络增强路径积分模型预测控制方法在自动驾驶中的应用 

**Authors**: Kanghyun Ryu, Minjun Sung, Piyush Gupta, Jovin D'sa, Faizan M. Tariq, David Isele, Sangjae Bae  

**Link**: [PDF](https://arxiv.org/pdf/2507.11940)  

**Abstract**: Motion planning for autonomous vehicles (AVs) in dense traffic is challenging, often leading to overly conservative behavior and unmet planning objectives. This challenge stems from the AVs' limited ability to anticipate and respond to the interactive behavior of surrounding agents. Traditional decoupled prediction and planning pipelines rely on non-interactive predictions that overlook the fact that agents often adapt their behavior in response to the AV's actions. To address this, we propose Interaction-Aware Neural Network-Enhanced Model Predictive Path Integral (IANN-MPPI) control, which enables interactive trajectory planning by predicting how surrounding agents may react to each control sequence sampled by MPPI. To improve performance in structured lane environments, we introduce a spline-based prior for the MPPI sampling distribution, enabling efficient lane-changing behavior. We evaluate IANN-MPPI in a dense traffic merging scenario, demonstrating its ability to perform efficient merging maneuvers. Our project website is available at this https URL 

**Abstract (ZH)**: 自主车辆在稠密交通中的运动规划具有挑战性，往往导致行为过于保守且无法满足规划目标。这一挑战源于自主车辆对周围代理交互行为的预测和响应能力有限。传统分离式的预测和规划管道依赖非交互式的预测，忽略了代理通常会根据自主车辆的行为调整自身行为的事实。为此，我们提出了一种交互感知的神经网络增强模型预测路径积分（IANN-MPPI）控制方法，该方法通过预测周围代理对每个由MPPI采样的控制序列的可能反应，实现了交互式轨迹规划。为了在结构化车道环境中提高性能，我们引入了一种基于样条函数的先验用于MPPI的采样分布，从而能够高效地实现车道变更行为。我们在稠密交通汇入场景中评估了IANN-MPPI，展示了其执行高效汇入动作的能力。项目网站见这个 <https> 地址。 

---
# AutoVDC: Automated Vision Data Cleaning Using Vision-Language Models 

**Title (ZH)**: AutoVDC：使用视觉-语言模型的自动化视觉数据清洗 

**Authors**: Santosh Vasa, Aditi Ramadwar, Jnana Rama Krishna Darabattula, Md Zafar Anwar, Stanislaw Antol, Andrei Vatavu, Thomas Monninger, Sihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.12414)  

**Abstract**: Training of autonomous driving systems requires extensive datasets with precise annotations to attain robust performance. Human annotations suffer from imperfections, and multiple iterations are often needed to produce high-quality datasets. However, manually reviewing large datasets is laborious and expensive. In this paper, we introduce AutoVDC (Automated Vision Data Cleaning) framework and investigate the utilization of Vision-Language Models (VLMs) to automatically identify erroneous annotations in vision datasets, thereby enabling users to eliminate these errors and enhance data quality. We validate our approach using the KITTI and nuImages datasets, which contain object detection benchmarks for autonomous driving. To test the effectiveness of AutoVDC, we create dataset variants with intentionally injected erroneous annotations and observe the error detection rate of our approach. Additionally, we compare the detection rates using different VLMs and explore the impact of VLM fine-tuning on our pipeline. The results demonstrate our method's high performance in error detection and data cleaning experiments, indicating its potential to significantly improve the reliability and accuracy of large-scale production datasets in autonomous driving. 

**Abstract (ZH)**: 自动驾驶系统训练需要具有精确标注的大规模数据集以实现 robust 性能。人工标注存在缺陷，往往需要多轮迭代来生成高质量数据集。然而，人工审查大量数据集是劳神费财的。本文提出 AutoVDC（自动视觉数据清洗）框架，并探索视觉语言模型（VLMs）自动识别视觉数据集中的错误标注，从而让用户能够消除这些错误并提高数据质量。我们使用包含自主驾驶检测基准的 KITTI 和 nuImages 数据集验证了我们的方法。为了测试 AutoVDC 的有效性，我们创建了故意注入错误标注的数据集变体，并观察了我们方法的错误检测率。此外，我们比较了不同 VLMs 的检测率，并探讨了 VLM 微调对我们管道的影响。结果表明，我们的方法在错误检测和数据清洗实验中表现出高性能，表明其有可能极大地提高大规模生产数据集中自主驾驶的可靠性和准确性。 

---
# SGLoc: Semantic Localization System for Camera Pose Estimation from 3D Gaussian Splatting Representation 

**Title (ZH)**: SGLoc：基于3D 高斯点云表示的相机姿态估计语义定位系统 

**Authors**: Beining Xu, Siting Zhu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12027)  

**Abstract**: We propose SGLoc, a novel localization system that directly regresses camera poses from 3D Gaussian Splatting (3DGS) representation by leveraging semantic information. Our method utilizes the semantic relationship between 2D image and 3D scene representation to estimate the 6DoF pose without prior pose information. In this system, we introduce a multi-level pose regression strategy that progressively estimates and refines the pose of query image from the global 3DGS map, without requiring initial pose priors. Moreover, we introduce a semantic-based global retrieval algorithm that establishes correspondences between 2D (image) and 3D (3DGS map). By matching the extracted scene semantic descriptors of 2D query image and 3DGS semantic representation, we align the image with the local region of the global 3DGS map, thereby obtaining a coarse pose estimation. Subsequently, we refine the coarse pose by iteratively optimizing the difference between the query image and the rendered image from 3DGS. Our SGLoc demonstrates superior performance over baselines on 12scenes and 7scenes datasets, showing excellent capabilities in global localization without initial pose prior. Code will be available at this https URL. 

**Abstract (ZH)**: 我们提出SGLoc，一种利用语义信息直接从3D高斯斑点表示（3DGS）回归相机姿态的新型定位系统。该方法利用2D图像与3D场景表示之间的语义关系，无需先验姿态信息即可估计6DoF姿态。在该系统中，我们引入了一种多层次的姿态回归策略，逐步从全局3DGS图中估计和细化查询图像的姿态。此外，我们引入了一种基于语义的全局检索算法，建立2D（图像）与3D（3DGS图）之间的对应关系。通过匹配2D查询图像提取的场景语义描述符与3DGS语义表示，使图像与全局3DGS图的局部区域对齐，从而获得粗略的姿态估计。随后，我们通过迭代优化查询图像与3DGS渲染图像之间的差异来细化粗略姿态。SGLoc在12scenes和7scenes数据集上表现出优于基线模型的性能，展示了在无需初始姿态先验的情况下出色的全局定位能力。代码将在此网址提供：这个https URL。 

---
# MOSPA: Human Motion Generation Driven by Spatial Audio 

**Title (ZH)**: MOSPA: 由空间音频驱动的人体运动生成 

**Authors**: Shuyang Xu, Zhiyang Dou, Mingyi Shi, Liang Pan, Leo Ho, Jingbo Wang, Yuan Liu, Cheng Lin, Yuexin Ma, Wenping Wang, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2507.11949)  

**Abstract**: Enabling virtual humans to dynamically and realistically respond to diverse auditory stimuli remains a key challenge in character animation, demanding the integration of perceptual modeling and motion synthesis. Despite its significance, this task remains largely unexplored. Most previous works have primarily focused on mapping modalities like speech, audio, and music to generate human motion. As of yet, these models typically overlook the impact of spatial features encoded in spatial audio signals on human motion. To bridge this gap and enable high-quality modeling of human movements in response to spatial audio, we introduce the first comprehensive Spatial Audio-Driven Human Motion (SAM) dataset, which contains diverse and high-quality spatial audio and motion data. For benchmarking, we develop a simple yet effective diffusion-based generative framework for human MOtion generation driven by SPatial Audio, termed MOSPA, which faithfully captures the relationship between body motion and spatial audio through an effective fusion mechanism. Once trained, MOSPA could generate diverse realistic human motions conditioned on varying spatial audio inputs. We perform a thorough investigation of the proposed dataset and conduct extensive experiments for benchmarking, where our method achieves state-of-the-art performance on this task. Our model and dataset will be open-sourced upon acceptance. Please refer to our supplementary video for more details. 

**Abstract (ZH)**: 使虚拟人类能够动态且真实地响应多样的听觉刺激仍然是角色动画中的一个关键挑战，这需要感知建模与运动合成的结合。尽管这项任务非常重要，但仍被很大程度上忽略了。大多数先前的工作主要集中在将语音、音频和音乐等模态映射到人类运动的生成上。截至目前，这些模型通常忽略了空间音频信号中编码的空间特征对人类运动的影响。为弥合这一差距，以实现空间音频驱动的人类运动的高质量建模，我们首次引入了全面的Spatial Audio-Driven Human Motion (SAM)数据集，该数据集包含多样且高质量的空间音频和运动数据。为了进行基准测试，我们开发了一个简单而有效的基于扩散的生成框架，用于由空间音频驱动的人类运动生成，该框架称为MOSPA，并通过有效的融合机制准确捕捉了身体运动与空间音频之间的关系。训练完成后，MOSPA能够在不同空间音频输入的条件下生成多样且逼真的人类运动。我们对提出的数据集进行了详尽的研究，并进行了广泛的基准测试，其中我们的方法已在该任务上实现了最先进的表现。我们的模型和数据集将在接受后开源。更多信息请参见我们的补充视频。 

---
# VISTA: Monocular Segmentation-Based Mapping for Appearance and View-Invariant Global Localization 

**Title (ZH)**: VISTA：基于单目分割的外观和视角不变全局定位映射 

**Authors**: Hannah Shafferman, Annika Thomas, Jouko Kinnari, Michael Ricard, Jose Nino, Jonathan How  

**Link**: [PDF](https://arxiv.org/pdf/2507.11653)  

**Abstract**: Global localization is critical for autonomous navigation, particularly in scenarios where an agent must localize within a map generated in a different session or by another agent, as agents often have no prior knowledge about the correlation between reference frames. However, this task remains challenging in unstructured environments due to appearance changes induced by viewpoint variation, seasonal changes, spatial aliasing, and occlusions -- known failure modes for traditional place recognition methods. To address these challenges, we propose VISTA (View-Invariant Segmentation-Based Tracking for Frame Alignment), a novel open-set, monocular global localization framework that combines: 1) a front-end, object-based, segmentation and tracking pipeline, followed by 2) a submap correspondence search, which exploits geometric consistencies between environment maps to align vehicle reference frames. VISTA enables consistent localization across diverse camera viewpoints and seasonal changes, without requiring any domain-specific training or finetuning. We evaluate VISTA on seasonal and oblique-angle aerial datasets, achieving up to a 69% improvement in recall over baseline methods. Furthermore, we maintain a compact object-based map that is only 0.6% the size of the most memory-conservative baseline, making our approach capable of real-time implementation on resource-constrained platforms. 

**Abstract (ZH)**: 全局定位对于自主导航至关重要，特别是在代理必须在其不同会话生成的地图或由其他代理生成的地图中进行定位的场景中，因为代理通常对参考坐标系之间的相关性没有任何先验知识。然而，由于视角变化、季节变化、空间混叠和遮挡导致的外观变化，这一任务在未结构化的环境中仍然极具挑战性——这些是传统位置识别方法已知的失败模式。为了解决这些挑战，我们提出了VISTA（基于视角不变分割和跟踪的帧对齐框架），一种新型的开放集单目全局定位框架，结合了：1）基于对象的前端分割和跟踪流水线，随后是2）子地图对应搜索，利用环境地图之间的几何一致性对车辆参考坐标系进行对齐。VISTA能够在多样化的摄像头视角和季节变化中实现一致的定位，无需任何领域特定的训练或微调。我们在季节性和斜视角航空数据集上评估了VISTA，相对于基线方法实现了高达69%的召回率提升。此外，我们维护了一个紧凑的对象基地图，其大小仅为最省内存基线的0.6%，使我们的方法能够在资源受限的平台上实现实时部署。 

---
# Unit-Based Histopathology Tissue Segmentation via Multi-Level Feature Representation 

**Title (ZH)**: 基于单元的组织病理学组织分割：多级特征表示 

**Authors**: Ashkan Shakarami, Azade Farshad, Yousef Yeganeh, Lorenzo Nicole, Peter Schuffler, Stefano Ghidoni, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2507.12427)  

**Abstract**: We propose UTS, a unit-based tissue segmentation framework for histopathology that classifies each fixed-size 32 * 32 tile, rather than each pixel, as the segmentation unit. This approach reduces annotation effort and improves computational efficiency without compromising accuracy. To implement this approach, we introduce a Multi-Level Vision Transformer (L-ViT), which benefits the multi-level feature representation to capture both fine-grained morphology and global tissue context. Trained to segment breast tissue into three categories (infiltrating tumor, non-neoplastic stroma, and fat), UTS supports clinically relevant tasks such as tumor-stroma quantification and surgical margin assessment. Evaluated on 386,371 tiles from 459 H&E-stained regions, it outperforms U-Net variants and transformer-based baselines. Code and Dataset will be available at GitHub. 

**Abstract (ZH)**: 基于单元的病理组织分割框架UTS：一种以32×32单元格为分割单位的细粒度形态与全局组织上下文捕获方法 

---
# Revealing the Ancient Beauty: Digital Reconstruction of Temple Tiles using Computer Vision 

**Title (ZH)**: 揭示古代之美：计算机视觉在寺庙瓷砖数字重建中的应用 

**Authors**: Arkaprabha Basu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12195)  

**Abstract**: Modern digitised approaches have dramatically changed the preservation and restoration of cultural treasures, integrating computer scientists into multidisciplinary projects with ease. Machine learning, deep learning, and computer vision techniques have revolutionised developing sectors like 3D reconstruction, picture inpainting,IoT-based methods, genetic algorithms, and image processing with the integration of computer scientists into multidisciplinary initiatives. We suggest three cutting-edge techniques in recognition of the special qualities of Indian monuments, which are famous for their architectural skill and aesthetic appeal. First is the Fractal Convolution methodology, a segmentation method based on image processing that successfully reveals subtle architectural patterns within these irreplaceable cultural buildings. The second is a revolutionary Self-Sensitive Tile Filling (SSTF) method created especially for West Bengal's mesmerising Bankura Terracotta Temples with a brand-new data augmentation method called MosaicSlice on the third. Furthermore, we delve deeper into the Super Resolution strategy to upscale the images without losing significant amount of quality. Our methods allow for the development of seamless region-filling and highly detailed tiles while maintaining authenticity using a novel data augmentation strategy within affordable costs introducing automation. By providing effective solutions that preserve the delicate balance between tradition and innovation, this study improves the subject and eventually ensures unrivalled efficiency and aesthetic excellence in cultural heritage protection. The suggested approaches advance the field into an era of unmatched efficiency and aesthetic quality while carefully upholding the delicate equilibrium between tradition and innovation. 

**Abstract (ZH)**: 现代数字化方法极大地改变了文化珍宝的保护与恢复，将计算机科学家融入多学科项目变得轻而易举。机器学习、深度学习和计算机视觉技术通过将计算机科学家融入多学科举措，重塑了3D重建、图像修复、物联网方法、遗传算法和图像处理等领域。我们提出了三种针对印度著名建筑艺术和美学特征的前沿技术。首先是基于图像处理的分形卷积方法，成功揭示了这些不可替代的文化建筑中的细微建筑模式。其次是专为西孟加拉邦迷人的 Bankura 陶器庙宇设计的革命性自我敏感瓷砖填充（SSTF）方法，引入了新的数据扩增方法 MosaicSlice。此外，我们更深入研究了超分辨率策略，以提高图像质量而不丢失大量细节。我们的方法通过引入新颖的数据扩增策略，在保证成本效益的同时实现了无缝区域填充和高度详细的瓷砖制作，维护了真实感并实现自动化。通过提供有效的解决方案，平衡传统与创新，本研究提升了主题，最终确保了文化遗址保护的无与伦比的效率和美学卓越。建议的方法将领域带入一个前所未有的高效和美学质量的时代，同时谨慎地维护了传统与创新之间的微妙平衡。 

---
# Wavelet-based Decoupling Framework for low-light Stereo Image Enhancement 

**Title (ZH)**: 基于小波的低光照立体图像解耦增强框架 

**Authors**: Shuangli Du, Siming Yan, Zhenghao Shi, Zhenzhen You, Lu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.12188)  

**Abstract**: Low-light images suffer from complex degradation, and existing enhancement methods often encode all degradation factors within a single latent space. This leads to highly entangled features and strong black-box characteristics, making the model prone to shortcut learning. To mitigate the above issues, this paper proposes a wavelet-based low-light stereo image enhancement method with feature space decoupling. Our insight comes from the following findings: (1) Wavelet transform enables the independent processing of low-frequency and high-frequency information. (2) Illumination adjustment can be achieved by adjusting the low-frequency component of a low-light image, extracted through multi-level wavelet decomposition. Thus, by using wavelet transform the feature space is decomposed into a low-frequency branch for illumination adjustment and multiple high-frequency branches for texture enhancement. Additionally, stereo low-light image enhancement can extract useful cues from another view to improve enhancement. To this end, we propose a novel high-frequency guided cross-view interaction module (HF-CIM) that operates within high-frequency branches rather than across the entire feature space, effectively extracting valuable image details from the other view. Furthermore, to enhance the high-frequency information, a detail and texture enhancement module (DTEM) is proposed based on cross-attention mechanism. The model is trained on a dataset consisting of images with uniform illumination and images with non-uniform illumination. Experimental results on both real and synthetic images indicate that our algorithm offers significant advantages in light adjustment while effectively recovering high-frequency information. The code and dataset are publicly available at: this https URL. 

**Abstract (ZH)**: 基于小波变换的低光照立体图像增强方法及特征空间分解 

---
# InstructFLIP: Exploring Unified Vision-Language Model for Face Anti-spoofing 

**Title (ZH)**: InstructFLIP: 探索统一的视觉-语言模型在面部防欺骗中的应用 

**Authors**: Kun-Hsiang Lin, Yu-Wen Tseng, Kang-Yang Huang, Jhih-Ciang Wu, Wen-Huang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.12060)  

**Abstract**: Face anti-spoofing (FAS) aims to construct a robust system that can withstand diverse attacks. While recent efforts have concentrated mainly on cross-domain generalization, two significant challenges persist: limited semantic understanding of attack types and training redundancy across domains. We address the first by integrating vision-language models (VLMs) to enhance the perception of visual input. For the second challenge, we employ a meta-domain strategy to learn a unified model that generalizes well across multiple domains. Our proposed InstructFLIP is a novel instruction-tuned framework that leverages VLMs to enhance generalization via textual guidance trained solely on a single domain. At its core, InstructFLIP explicitly decouples instructions into content and style components, where content-based instructions focus on the essential semantics of spoofing, and style-based instructions consider variations related to the environment and camera characteristics. Extensive experiments demonstrate the effectiveness of InstructFLIP by outperforming SOTA models in accuracy and substantially reducing training redundancy across diverse domains in FAS. Project website is available at this https URL. 

**Abstract (ZH)**: 人脸识别防欺骗（Face Anti-Spoofing, FAS）旨在构建一个 robust 系统以应对各种攻击。尽管近期努力主要集中在跨域泛化上，但依然存在两大挑战：对攻击类型的有限语义理解以及不同领域间的训练冗余。我们通过整合视觉语言模型（VLMs）以增强对视觉输入的感知来应对第一个挑战。针对第二个挑战，我们采用元领域策略来学习一个能够在多个领域泛化的统一模型。我们提出的 InstructFLIP 是一种新颖的指令调优框架，通过仅在一个领域上进行文本指导，利用 VLMs 来提升泛化能力。InstructFLIP 在核心部分显式地将指令分解为内容和风格两个组件，其中基于内容的指令关注伪装的关键语义，基于风格的指令考虑环境和摄像机特性等方面的变化。大量实验结果表明，InstructFLIP 在准确率上优于当前最优模型，并且在 FAS 的多个领域中大幅减少了训练冗余。项目网址可访问此 <https://>。 

---
# SS-DC: Spatial-Spectral Decoupling and Coupling Across Visible-Infrared Gap for Domain Adaptive Object Detection 

**Title (ZH)**: SS-DC: 可见光-红外区间跨域适配目标检测中的空间-谱域解藕与耦合 

**Authors**: Xiwei Zhang, Chunjin Yang, Yiming Xiao, Runtong Zhang, Fanman Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.12017)  

**Abstract**: Unsupervised domain adaptive object detection (UDAOD) from the visible domain to the infrared (RGB-IR) domain is challenging. Existing methods regard the RGB domain as a unified domain and neglect the multiple subdomains within it, such as daytime, nighttime, and foggy scenes. We argue that decoupling the domain-invariant (DI) and domain-specific (DS) features across these multiple subdomains is beneficial for RGB-IR domain adaptation. To this end, this paper proposes a new SS-DC framework based on a decoupling-coupling strategy. In terms of decoupling, we design a Spectral Adaptive Idempotent Decoupling (SAID) module in the aspect of spectral decomposition. Due to the style and content information being highly embedded in different frequency bands, this module can decouple DI and DS components more accurately and interpretably. A novel filter bank-based spectral processing paradigm and a self-distillation-driven decoupling loss are proposed to improve the spectral domain decoupling. In terms of coupling, a new spatial-spectral coupling method is proposed, which realizes joint coupling through spatial and spectral DI feature pyramids. Meanwhile, this paper introduces DS from decoupling to reduce the domain bias. Extensive experiments demonstrate that our method can significantly improve the baseline performance and outperform existing UDAOD methods on multiple RGB-IR datasets, including a new experimental protocol proposed in this paper based on the FLIR-ADAS dataset. 

**Abstract (ZH)**: 无监督域自适应可见光到红外域目标检测（UDAOD）：从可见光域到红外域的无监督域自适应目标检测 

---
# Identifying Signatures of Image Phenotypes to Track Treatment Response in Liver Disease 

**Title (ZH)**: 识别图像表型的特征以追踪肝病治疗反应 

**Authors**: Matthias Perkonigg, Nina Bastati, Ahmed Ba-Ssalamah, Peter Mesenbrink, Alexander Goehler, Miljen Martic, Xiaofei Zhou, Michael Trauner, Georg Langs  

**Link**: [PDF](https://arxiv.org/pdf/2507.12012)  

**Abstract**: Quantifiable image patterns associated with disease progression and treatment response are critical tools for guiding individual treatment, and for developing novel therapies. Here, we show that unsupervised machine learning can identify a pattern vocabulary of liver tissue in magnetic resonance images that quantifies treatment response in diffuse liver disease. Deep clustering networks simultaneously encode and cluster patches of medical images into a low-dimensional latent space to establish a tissue vocabulary. The resulting tissue types capture differential tissue change and its location in the liver associated with treatment response. We demonstrate the utility of the vocabulary on a randomized controlled trial cohort of non-alcoholic steatohepatitis patients. First, we use the vocabulary to compare longitudinal liver change in a placebo and a treatment cohort. Results show that the method identifies specific liver tissue change pathways associated with treatment, and enables a better separation between treatment groups than established non-imaging measures. Moreover, we show that the vocabulary can predict biopsy derived features from non-invasive imaging data. We validate the method on a separate replication cohort to demonstrate the applicability of the proposed method. 

**Abstract (ZH)**: 可量化图像模式与疾病进展和治疗反应相关，是指导个性化治疗和开发新型疗法的重要工具。在这里，我们展示了无监督机器学习可以识别磁共振图像中肝组织的模式词汇，量化弥漫性肝病的治疗反应。深度聚类网络同时将医学图像的块编码并聚类到低维潜在空间中，建立组织词汇。产生的组织类型捕获与治疗反应相关的组织变化及其在肝脏中的位置。我们通过一项随机对照试验队列的非酒精性 steatohepatitis 患者组来展示词汇的实用性。首先，我们使用词汇比较安慰剂组和治疗组的纵向肝组织变化。结果显示，该方法识别出与治疗相关的特定肝脏组织变化路径，并能比现有非影像学指标更好地分离治疗组。此外，我们展示了该词汇可以预测从非侵入性影像数据推导出的活检特征。我们在独立的复制队列上验证该方法，以证明所提出方法的应用性。 

---
# Dual form Complementary Masking for Domain-Adaptive Image Segmentation 

**Title (ZH)**: 域适应图像分割的双重形式互补掩蔽 

**Authors**: Jiawen Wang, Yinda Chen, Xiaoyu Liu, Che Liu, Dong Liu, Jianqing Gao, Zhiwei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12008)  

**Abstract**: Recent works have correlated Masked Image Modeling (MIM) with consistency regularization in Unsupervised Domain Adaptation (UDA). However, they merely treat masking as a special form of deformation on the input images and neglect the theoretical analysis, which leads to a superficial understanding of masked reconstruction and insufficient exploitation of its potential in enhancing feature extraction and representation learning. In this paper, we reframe masked reconstruction as a sparse signal reconstruction problem and theoretically prove that the dual form of complementary masks possesses superior capabilities in extracting domain-agnostic image features. Based on this compelling insight, we propose MaskTwins, a simple yet effective UDA framework that integrates masked reconstruction directly into the main training pipeline. MaskTwins uncovers intrinsic structural patterns that persist across disparate domains by enforcing consistency between predictions of images masked in complementary ways, enabling domain generalization in an end-to-end manner. Extensive experiments verify the superiority of MaskTwins over baseline methods in natural and biological image segmentation. These results demonstrate the significant advantages of MaskTwins in extracting domain-invariant features without the need for separate pre-training, offering a new paradigm for domain-adaptive segmentation. 

**Abstract (ZH)**: 近期 studies 将 Masked Image Modeling (MIM) 与 Unsupervised Domain Adaptation (UDA) 中的一致性正则化联系了起来。然而，这些研究仅仅将掩蔽视为输入图像的一种特殊变形形式，并忽视了理论分析，导致对掩蔽重建的理解肤浅，无法充分发掘其在增强特征提取和表示学习方面的潜力。本文重新将掩蔽重建视为稀疏信号重构问题，并理论证明互补掩蔽的对偶形式在提取领域不变图像特征方面具有优越的能力。基于这一令人信服的见解，我们提出了 MaskTwins，一种将掩蔽重建直接集成到主训练管道中的简单而有效的UDA框架。MaskTwins 通过在以不同方式掩蔽的图像之间强制一致性来揭示跨不同领域内在的结构模式，从而以端到端的方式实现领域泛化。广泛的实验验证了 MaskTwins 在自然和生物图像分割中的优越性，证明了 MaskTwins 在无需单独预训练的情况下提取领域不变特征的优势，为领域适应分割提供了新的范式。 

---
# Frequency-Dynamic Attention Modulation for Dense Prediction 

**Title (ZH)**: 频域动态注意力调制dense预测 

**Authors**: Linwei Chen, Lin Gu, Ying Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12006)  

**Abstract**: Vision Transformers (ViTs) have significantly advanced computer vision, demonstrating strong performance across various tasks. However, the attention mechanism in ViTs makes each layer function as a low-pass filter, and the stacked-layer architecture in existing transformers suffers from frequency vanishing. This leads to the loss of critical details and textures. We propose a novel, circuit-theory-inspired strategy called Frequency-Dynamic Attention Modulation (FDAM), which can be easily plugged into ViTs. FDAM directly modulates the overall frequency response of ViTs and consists of two techniques: Attention Inversion (AttInv) and Frequency Dynamic Scaling (FreqScale). Since circuit theory uses low-pass filters as fundamental elements, we introduce AttInv, a method that generates complementary high-pass filtering by inverting the low-pass filter in the attention matrix, and dynamically combining the two. We further design FreqScale to weight different frequency components for fine-grained adjustments to the target response function. Through feature similarity analysis and effective rank evaluation, we demonstrate that our approach avoids representation collapse, leading to consistent performance improvements across various models, including SegFormer, DeiT, and MaskDINO. These improvements are evident in tasks such as semantic segmentation, object detection, and instance segmentation. Additionally, we apply our method to remote sensing detection, achieving state-of-the-art results in single-scale settings. The code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于电路理论的频域动态注意力调制（FDAM）：一种适用于ViTs的新型策略 

---
# Effective Fine-Tuning of Vision Transformers with Low-Rank Adaptation for Privacy-Preserving Image Classification 

**Title (ZH)**: 有效细调视觉变换器以实现隐私保护图像分类的低秩适应方法 

**Authors**: Haiwei Lin, Shoko Imaizumi, Hitoshi Kiya  

**Link**: [PDF](https://arxiv.org/pdf/2507.11943)  

**Abstract**: We propose a low-rank adaptation method for training privacy-preserving vision transformer (ViT) models that efficiently freezes pre-trained ViT model weights. In the proposed method, trainable rank decomposition matrices are injected into each layer of the ViT architecture, and moreover, the patch embedding layer is not frozen, unlike in the case of the conventional low-rank adaptation methods. The proposed method allows us not only to reduce the number of trainable parameters but to also maintain almost the same accuracy as that of full-time tuning. 

**Abstract (ZH)**: 我们提出了一种低秩适应方法，用于训练保护隐私的视觉变换器（ViT）模型，该方法可以有效地冻结预训练的ViT模型权重。在所提出的方法中，可训练的秩分解矩阵被注入到ViT架构的每一层中，而且 Patch 嵌入层没有被冻结，这与传统低秩适应方法不同。所提出的方法不仅减少了可训练参数的数量，还几乎保持了全时段调优的相同准确率。 

---
# Spatial Frequency Modulation for Semantic Segmentation 

**Title (ZH)**: 空间频率调制用于语义分割 

**Authors**: Linwei Chen, Ying Fu, Lin Gu, Dezhi Zheng, Jifeng Dai  

**Link**: [PDF](https://arxiv.org/pdf/2507.11893)  

**Abstract**: High spatial frequency information, including fine details like textures, significantly contributes to the accuracy of semantic segmentation. However, according to the Nyquist-Shannon Sampling Theorem, high-frequency components are vulnerable to aliasing or distortion when propagating through downsampling layers such as strided-convolution. Here, we propose a novel Spatial Frequency Modulation (SFM) that modulates high-frequency features to a lower frequency before downsampling and then demodulates them back during upsampling. Specifically, we implement modulation through adaptive resampling (ARS) and design a lightweight add-on that can densely sample the high-frequency areas to scale up the signal, thereby lowering its frequency in accordance with the Frequency Scaling Property. We also propose Multi-Scale Adaptive Upsampling (MSAU) to demodulate the modulated feature and recover high-frequency information through non-uniform upsampling This module further improves segmentation by explicitly exploiting information interaction between densely and sparsely resampled areas at multiple scales. Both modules can seamlessly integrate with various architectures, extending from convolutional neural networks to transformers. Feature visualization and analysis confirm that our method effectively alleviates aliasing while successfully retaining details after demodulation. Finally, we validate the broad applicability and effectiveness of SFM by extending it to image classification, adversarial robustness, instance segmentation, and panoptic segmentation tasks. The code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 高频空间信息，包括纹理等精细细节，显著提高了语义分割的准确性。然而，根据奈奎斯特-香农采样定理，高频成分在经过卷积下采样层（如跨步卷积）传播时容易发生混叠或失真。为此，我们提出了一种新型的空间频率调制（SFM），在下采样前将高频特征调制至较低频率，然后在上采样时进行反调制。具体地，我们通过自适应重采样（ARS）实现调制，并设计了一个轻量级附加模块，密集采样高频区域以放大信号，从而根据频域缩放特性降低其频率。此外，我们提出了多尺度自适应上采样（MSAU），用于反调制调制特征并通过非均匀上采样恢复高频信息。该模块进一步通过在多尺度上显式利用密集和稀疏重采样区域之间的信息交互来改善分割。这两个模块可以无缝集成到各种架构中，从卷积神经网络扩展到变压器。特征可视化和分析表明，我们的方法有效地缓解了混叠问题，并在反调制后成功保留了细节。最后，我们通过将其扩展到图像分类、对抗鲁棒性、实例分割和全景分割任务，验证了SFM的广泛应用性和有效性。代码可在\href{this https URL}{此链接}获得。 

---
# Beyond Task-Specific Reasoning: A Unified Conditional Generative Framework for Abstract Visual Reasoning 

**Title (ZH)**: 超越任务特定推理：一种统一的条件生成框架用于抽象视觉推理 

**Authors**: Fan Shi, Bin Li, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.11761)  

**Abstract**: Abstract visual reasoning (AVR) enables humans to quickly discover and generalize abstract rules to new scenarios. Designing intelligent systems with human-like AVR abilities has been a long-standing topic in the artificial intelligence community. Deep AVR solvers have recently achieved remarkable success in various AVR tasks. However, they usually use task-specific designs or parameters in different tasks. In such a paradigm, solving new tasks often means retraining the model, and sometimes retuning the model architectures, which increases the cost of solving AVR problems. In contrast to task-specific approaches, this paper proposes a novel Unified Conditional Generative Solver (UCGS), aiming to address multiple AVR tasks in a unified framework. First, we prove that some well-known AVR tasks can be reformulated as the problem of estimating the predictability of target images in problem panels. Then, we illustrate that, under the proposed framework, training one conditional generative model can solve various AVR tasks. The experiments show that with a single round of multi-task training, UCGS demonstrates abstract reasoning ability across various AVR tasks. Especially, UCGS exhibits the ability of zero-shot reasoning, enabling it to perform abstract reasoning on problems from unseen AVR tasks in the testing phase. 

**Abstract (ZH)**: 统一条件生成求解器（UCGS）：解决多种抽象视觉推理任务的统一框架 

---
# Galaxy image simplification using Generative AI 

**Title (ZH)**: 使用生成式AI进行星系图像简化 

**Authors**: Sai Teja Erukude, Lior Shamir  

**Link**: [PDF](https://arxiv.org/pdf/2507.11692)  

**Abstract**: Modern digital sky surveys have been acquiring images of billions of galaxies. While these images often provide sufficient details to analyze the shape of the galaxies, accurate analysis of such high volumes of images requires effective automation. Current solutions often rely on machine learning annotation of the galaxy images based on a set of pre-defined classes. Here we introduce a new approach to galaxy image analysis that is based on generative AI. The method simplifies the galaxy images and automatically converts them into a ``skeletonized" form. The simplified images allow accurate measurements of the galaxy shapes and analysis that is not limited to a certain pre-defined set of classes. We demonstrate the method by applying it to galaxy images acquired by the DESI Legacy Survey. The code and data are publicly available. The method was applied to 125,000 DESI Legacy Survey images, and the catalog of the simplified images is publicly available. 

**Abstract (ZH)**: 现代数字天空调查正在获取数十亿星系的图像。虽然这些图像通常提供了足够的细节来分析星系的形状，但对如此大量图像的准确分析需要有效的自动化手段。当前的解决方案往往依赖于基于预定义类别的机器学习对星系图像进行标注。在这里，我们介绍了一种基于生成式AI的星系图像分析方法。该方法简化了星系图像，并自动将其转换为“骨架化”形式。简化的图像允许精确测量星系形状，并且分析不限于某些预定义的类别。我们通过将该方法应用于DESI遗留给星系图像来演示这种方法。该代码和数据均可公开获取。该方法应用于125,000张DESI遗留给星系图像，并公开发布了简化的图像目录。 

---
# What cat is that? A re-id model for feral cats 

**Title (ZH)**: 那cats是什么？一种针对无家猫的再识别模型 

**Authors**: Victor Caquilpan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11575)  

**Abstract**: Feral cats exert a substantial and detrimental impact on Australian wildlife, placing them among the most dangerous invasive species worldwide. Therefore, closely monitoring these cats is essential labour in minimising their effects. In this context, the potential application of Re-Identification (re-ID) emerges to enhance monitoring activities for these animals, utilising images captured by camera traps. This project explores different CV approaches to create a re-ID model able to identify individual feral cats in the wild. The main approach consists of modifying a part-pose guided network (PPGNet) model, initially used in the re-ID of Amur tigers, to be applicable for feral cats. This adaptation, resulting in PPGNet-Cat, which incorporates specific modifications to suit the characteristics of feral cats images. Additionally, various experiments were conducted, particularly exploring contrastive learning approaches such as ArcFace loss. The main results indicate that PPGNet-Cat excels in identifying feral cats, achieving high performance with a mean Average Precision (mAP) of 0.86 and a rank-1 accuracy of 0.95. These outcomes establish PPGNet-Cat as a competitive model within the realm of re-ID. 

**Abstract (ZH)**: 野猫对澳大利亚野生动植物造成重大且有害的影响，将其列为全球最危险的入侵物种之一。因此，密切监测这些猫对减轻其影响至关重要。在此背景下，重新识别（re-ID）的应用有望提高对这些动物的监测活动，利用相机陷阱拍摄的图像。本项目探索不同的计算机视觉（CV）方法，以创建一个能够识别野生环境中 individual 野猫的 re-ID 模型。主要方法是修改最初用于 Amur 豹猫重新识别的 part-pose 指导网络（PPGNet）模型，使其适用于野猫。这种适应结果产生了 PPGNet-Cat，其中包含了特定的修改以适应野猫图像的特性。此外，还进行了多种实验，特别是探索对比学习方法，如 ArcFace 损失。主要结果表明，PPGNet-Cat 在识别野猫方面表现出色，平均精度（mAP）达到 0.86，排名为 1 的准确性为 0.95。这些结果使 PPGNet-Cat 成为 re-ID 领域中一个竞争力较强的模型。 

---
# Are Vision Foundation Models Ready for Out-of-the-Box Medical Image Registration? 

**Title (ZH)**: Vision基础模型准备好开箱即用的医疗图像配准了吗？ 

**Authors**: Hanxue Gu, Yaqian Chen, Nicholas Konz, Qihang Li, Maciej A. Mazurowski  

**Link**: [PDF](https://arxiv.org/pdf/2507.11569)  

**Abstract**: Foundation models, pre-trained on large image datasets and capable of capturing rich feature representations, have recently shown potential for zero-shot image registration. However, their performance has mostly been tested in the context of rigid or less complex structures, such as the brain or abdominal organs, and it remains unclear whether these models can handle more challenging, deformable anatomy. Breast MRI registration is particularly difficult due to significant anatomical variation between patients, deformation caused by patient positioning, and the presence of thin and complex internal structure of fibroglandular tissue, where accurate alignment is crucial. Whether foundation model-based registration algorithms can address this level of complexity remains an open question. In this study, we provide a comprehensive evaluation of foundation model-based registration algorithms for breast MRI. We assess five pre-trained encoders, including DINO-v2, SAM, MedSAM, SSLSAM, and MedCLIP, across four key breast registration tasks that capture variations in different years and dates, sequences, modalities, and patient disease status (lesion versus no lesion). Our results show that foundation model-based algorithms such as SAM outperform traditional registration baselines for overall breast alignment, especially under large domain shifts, but struggle with capturing fine details of fibroglandular tissue. Interestingly, additional pre-training or fine-tuning on medical or breast-specific images in MedSAM and SSLSAM, does not improve registration performance and may even decrease it in some cases. Further work is needed to understand how domain-specific training influences registration and to explore targeted strategies that improve both global alignment and fine structure accuracy. We also publicly release our code at \href{this https URL}{Github}. 

**Abstract (ZH)**: 基于基础模型的乳腺MRI配准算法全面评估 

---
# Expert Operational GANS: Towards Real-Color Underwater Image Restoration 

**Title (ZH)**: 专家操作GANS：Towards 实际色彩 underwater 图像恢复 

**Authors**: Ozer Can Devecioglu, Serkan Kiranyaz, Mehmet Yamac, Moncef Gabbouj  

**Link**: [PDF](https://arxiv.org/pdf/2507.11562)  

**Abstract**: The wide range of deformation artifacts that arise from complex light propagation, scattering, and depth-dependent attenuation makes the underwater image restoration to remain a challenging problem. Like other single deep regressor networks, conventional GAN-based restoration methods struggle to perform well across this heterogeneous domain, since a single generator network is typically insufficient to capture the full range of visual degradations. In order to overcome this limitation, we propose xOp-GAN, a novel GAN model with several expert generator networks, each trained solely on a particular subset with a certain image quality. Thus, each generator can learn to maximize its restoration performance for a particular quality range. Once a xOp-GAN is trained, each generator can restore the input image and the best restored image can then be selected by the discriminator based on its perceptual confidence score. As a result, xOP-GAN is the first GAN model with multiple generators where the discriminator is being used during the inference of the regression task. Experimental results on benchmark Large Scale Underwater Image (LSUI) dataset demonstrates that xOp-GAN achieves PSNR levels up to 25.16 dB, surpassing all single-regressor models by a large margin even, with reduced complexity. 

**Abstract (ZH)**: 复杂光传播、散射及深度相关衰减引起的广泛变形伪影使得水下图像恢复仍是一个具有挑战性的问题。由于传统的基于生成对抗网络（GAN）的恢复方法难以在这种异质领域中表现出色，单一生成网络通常无法捕捉到视觉退化的全部范围，我们提出xOp-GAN，一种具有多个专门训练于特定图像质量子集的专家生成器网络的新型GAN模型，每个生成器可以学习在特定质量范围内最大化其恢复性能。训练完成后，每个生成器可以恢复输入图像，鉴别器将基于感知置信分数选择最佳恢复图像。实验结果表明，xOp-GAN在基准Large Scale Underwater Image (LSUI)数据集上的PSNR达到25.16 dB，即使复杂度降低，也显著超越了所有单一回归模型。 

---
# Predicting Pulmonary Hypertension in Newborns: A Multi-view VAE Approach 

**Title (ZH)**: 新出生婴儿肺动脉高压预测：多视图VAE方法 

**Authors**: Lucas Erlacher, Samuel Ruipérez-Campillo, Holger Michel, Sven Wellmann, Thomas M. Sutter, Ece Ozkan, Julia E. Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2507.11561)  

**Abstract**: Pulmonary hypertension (PH) in newborns is a critical condition characterized by elevated pressure in the pulmonary arteries, leading to right ventricular strain and heart failure. While right heart catheterization (RHC) is the diagnostic gold standard, echocardiography is preferred due to its non-invasive nature, safety, and accessibility. However, its accuracy highly depends on the operator, making PH assessment subjective. While automated detection methods have been explored, most models focus on adults and rely on single-view echocardiographic frames, limiting their performance in diagnosing PH in newborns. While multi-view echocardiography has shown promise in improving PH assessment, existing models struggle with generalizability. In this work, we employ a multi-view variational autoencoder (VAE) for PH prediction using echocardiographic videos. By leveraging the VAE framework, our model captures complex latent representations, improving feature extraction and robustness. We compare its performance against single-view and supervised learning approaches. Our results show improved generalization and classification accuracy, highlighting the effectiveness of multi-view learning for robust PH assessment in newborns. 

**Abstract (ZH)**: 新生儿肺动脉高压的多视角变分自编码器预测研究 

---
# Reprogramming Vision Foundation Models for Spatio-Temporal Forecasting 

**Title (ZH)**: 重塑视觉基础模型以实现空时预测 

**Authors**: Changlu Chen, Yanbin Liu, Chaoxi Niu, Ling Chen, Tianqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11558)  

**Abstract**: Foundation models have achieved remarkable success in natural language processing and computer vision, demonstrating strong capabilities in modeling complex patterns. While recent efforts have explored adapting large language models (LLMs) for time-series forecasting, LLMs primarily capture one-dimensional sequential dependencies and struggle to model the richer spatio-temporal (ST) correlations essential for accurate ST forecasting. In this paper, we present \textbf{ST-VFM}, a novel framework that systematically reprograms Vision Foundation Models (VFMs) for general-purpose spatio-temporal forecasting. While VFMs offer powerful spatial priors, two key challenges arise when applying them to ST tasks: (1) the lack of inherent temporal modeling capacity and (2) the modality gap between visual and ST data. To address these, ST-VFM adopts a \emph{dual-branch architecture} that integrates raw ST inputs with auxiliary ST flow inputs, where the flow encodes lightweight temporal difference signals interpretable as dynamic spatial cues. To effectively process these dual-branch inputs, ST-VFM introduces two dedicated reprogramming stages. The \emph{pre-VFM reprogramming} stage applies a Temporal-Aware Token Adapter to embed temporal context and align both branches into VFM-compatible feature spaces. The \emph{post-VFM reprogramming} stage introduces a Bilateral Cross-Prompt Coordination module, enabling dynamic interaction between branches through prompt-based conditioning, thus enriching joint representation learning without modifying the frozen VFM backbone. Extensive experiments on ten spatio-temporal datasets show that ST-VFM outperforms state-of-the-art baselines, demonstrating effectiveness and robustness across VFM backbones (e.g., DINO, CLIP, DEIT) and ablation studies, establishing it as a strong general framework for spatio-temporal forecasting. 

**Abstract (ZH)**: ST-VFM：一种系统性重构视觉基础模型的新型框架，用于通用时空预测 

---
# 3D Wavelet Latent Diffusion Model for Whole-Body MR-to-CT Modality Translation 

**Title (ZH)**: 三维小波潜在扩散模型在全身MR到CT模态转化中的应用 

**Authors**: Jiaxu Zheng, Meiman He, Xuhui Tang, Xiong Wang, Tuoyu Cao, Tianyi Zeng, Lichi Zhang, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2507.11557)  

**Abstract**: Magnetic Resonance (MR) imaging plays an essential role in contemporary clinical diagnostics. It is increasingly integrated into advanced therapeutic workflows, such as hybrid Positron Emission Tomography/Magnetic Resonance (PET/MR) imaging and MR-only radiation therapy. These integrated approaches are critically dependent on accurate estimation of radiation attenuation, which is typically facilitated by synthesizing Computed Tomography (CT) images from MR scans to generate attenuation maps. However, existing MR-to-CT synthesis methods for whole-body imaging often suffer from poor spatial alignment between the generated CT and input MR images, and insufficient image quality for reliable use in downstream clinical tasks. In this paper, we present a novel 3D Wavelet Latent Diffusion Model (3D-WLDM) that addresses these limitations by performing modality translation in a learned latent space. By incorporating a Wavelet Residual Module into the encoder-decoder architecture, we enhance the capture and reconstruction of fine-scale features across image and latent spaces. To preserve anatomical integrity during the diffusion process, we disentangle structural and modality-specific characteristics and anchor the structural component to prevent warping. We also introduce a Dual Skip Connection Attention mechanism within the diffusion model, enabling the generation of high-resolution CT images with improved representation of bony structures and soft-tissue contrast. 

**Abstract (ZH)**: 磁共振成像（MR）在当代临床诊断中扮演着重要角色，越来越多地被集成到先进的治疗工作流中，如混合正电子发射断层扫描/磁共振成像（PET/MR）和磁共振引导的放射治疗。这些集成方法通常依赖于准确的辐射衰减估计，这通常通过从MR扫描合成计算机断层扫描（CT）图像来生成衰减图来实现。然而，现有的全身成像的MR到CT合成方法往往在生成的CT和输入的MR图像之间产生不良的空间对齐，并且图像质量不够可靠，无法在下游临床任务中可靠使用。本文中，我们提出了一种新颖的三维小波潜在扩散模型（3D-WLDM），通过在学习的潜在空间中执行模态转换来解决这些限制。通过将小波残差模块融入编码器-解码器架构中，我们增强了跨图像和潜在空间的细尺度特征的捕捉和重建。为了在扩散过程中保持解剖完整性，我们分离了结构和模态特异性特征，并固定结构成分以防止变形。我们还在扩散模型中引入了双跳跃连接注意力机制，从而能够生成高分辨率CT图像，并改善骨骼结构和软组织对比度的表示。 

---
# Landmark Detection for Medical Images using a General-purpose Segmentation Model 

**Title (ZH)**: 使用通用分割模型进行医学图像 landmarks 检测 

**Authors**: Ekaterina Stansfield, Jennifer A. Mitterer, Abdulrahman Altahhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11551)  

**Abstract**: Radiographic images are a cornerstone of medical diagnostics in orthopaedics, with anatomical landmark detection serving as a crucial intermediate step for information extraction. General-purpose foundational segmentation models, such as SAM (Segment Anything Model), do not support landmark segmentation out of the box and require prompts to function. However, in medical imaging, the prompts for landmarks are highly specific. Since SAM has not been trained to recognize such landmarks, it cannot generate accurate landmark segmentations for diagnostic purposes. Even MedSAM, a medically adapted variant of SAM, has been trained to identify larger anatomical structures, such as organs and their parts, and lacks the fine-grained precision required for orthopaedic pelvic landmarks. To address this limitation, we propose leveraging another general-purpose, non-foundational model: YOLO. YOLO excels in object detection and can provide bounding boxes that serve as input prompts for SAM. While YOLO is efficient at detection, it is significantly outperformed by SAM in segmenting complex structures. In combination, these two models form a reliable pipeline capable of segmenting not only a small pilot set of eight anatomical landmarks but also an expanded set of 72 landmarks and 16 regions with complex outlines, such as the femoral cortical bone and the pelvic inlet. By using YOLO-generated bounding boxes to guide SAM, we trained the hybrid model to accurately segment orthopaedic pelvic radiographs. Our results show that the proposed combination of YOLO and SAM yields excellent performance in detecting anatomical landmarks and intricate outlines in orthopaedic pelvic radiographs. 

**Abstract (ZH)**: 放射学图像被誉为骨科医学诊断的基石，解剖标志点检测是信息提取的关键中间步骤。通用的基础分割模型，如SAM（Segment Anything Model），不支持即用型的标志点分割，需要提示以发挥作用。然而，在医学成像中，标志点的提示非常具体。由于SAM未被训练来识别此类标志点，因此无法生成用于诊断目的的准确标志点分割。即使是专门针对医学应用调整的MedSAM，也仅被训练识别较大的解剖结构，如器官及其部分，缺乏骨科骨盆标志点所需的细粒度精度。为解决这一限制，我们提出利用另一种通用非基础模型YOLO：YOLO在目标检测方面表现优异，可以提供作为SAM输入提示的边界框。虽然YOLO在检测方面非常高效，但在分割复杂结构方面远逊于SAM。结合使用这两种模型形成了一个可靠的流水线，不仅能分割骨盆的8个解剖标志点试点集，还能分割72个标志点和包括股骨皮质骨和骨盆入口在内的16个具有复杂边缘的区域。通过使用YOLO生成的边界框指导SAM，我们训练了这种混合模型以准确分割骨科骨盆放射学图像。我们的结果表明，提出的YOLO与SAM的组合在检测骨科骨盆放射学图像中的解剖标志点和复杂边缘方面表现卓越。 

---
# An Memory-Efficient Framework for Deformable Transformer with Neural Architecture Search 

**Title (ZH)**: 一种基于神经架构搜索的内存高效变形 transformer 框架 

**Authors**: Wendong Mao, Mingfan Zhao, Jianfeng Guan, Qiwei Dong, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11549)  

**Abstract**: Deformable Attention Transformers (DAT) have shown remarkable performance in computer vision tasks by adaptively focusing on informative image regions. However, their data-dependent sampling mechanism introduces irregular memory access patterns, posing significant challenges for efficient hardware deployment. Existing acceleration methods either incur high hardware overhead or compromise model accuracy. To address these issues, this paper proposes a hardware-friendly optimization framework for DAT. First, a neural architecture search (NAS)-based method with a new slicing strategy is proposed to automatically divide the input feature into uniform patches during the inference process, avoiding memory conflicts without modifying model architecture. The method explores the optimal slice configuration by jointly optimizing hardware cost and inference accuracy. Secondly, an FPGA-based verification system is designed to test the performance of this framework on edge-side hardware. Algorithm experiments on the ImageNet-1K dataset demonstrate that our hardware-friendly framework can maintain have only 0.2% accuracy drop compared to the baseline DAT. Hardware experiments on Xilinx FPGA show the proposed method reduces DRAM access times to 18% compared with existing DAT acceleration methods. 

**Abstract (ZH)**: 变形注意力变压器（DAT）在计算机视觉任务中通过自适应关注信息性的图像区域展现了卓越的表现。然而，其数据依赖的采样机制引入了不规则的内存访问模式，给高效的硬件部署带来了重大挑战。现有的加速方法要么会导致高硬件开销，要么会损害模型的准确性。为解决这些问题，本文提出了一种面向硬件的DAT优化框架。首先，提出了一种基于神经架构搜索（NAS）的新切片策略，在推理过程中自动生成均匀的特征片，避免内存冲突，同时不修改模型架构。该方法通过联合优化硬件成本和推理准确性来探索最佳切片配置。其次，设计了一种基于FPGA的验证系统，用以测试该框架在边缘硬件上的性能。图像集ImageNet-1K上的算法实验表明，我们的面向硬件的框架与基线DAT相比仅Accuracy下降0.2%。Xilinx FPGA上的硬件实验表明，所提出的方法将DRAM访问时间减少了18%，优于现有DAT加速方法。 

---
