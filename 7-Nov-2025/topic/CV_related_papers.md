# GraspView: Active Perception Scoring and Best-View Optimization for Robotic Grasping in Cluttered Environments 

**Title (ZH)**: GraspView: 基于主动感知评分与杂乱环境中最佳视角优化的机器人抓取方法 

**Authors**: Shenglin Wang, Mingtong Dai, Jingxuan Su, Lingbo Liu, Chunjie Chen, Xinyu Wu, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.04199)  

**Abstract**: Robotic grasping is a fundamental capability for autonomous manipulation, yet remains highly challenging in cluttered environments where occlusion, poor perception quality, and inconsistent 3D reconstructions often lead to unstable or failed grasps. Conventional pipelines have widely relied on RGB-D cameras to provide geometric information, which fail on transparent or glossy objects and degrade at close range. We present GraspView, an RGB-only robotic grasping pipeline that achieves accurate manipulation in cluttered environments without depth sensors. Our framework integrates three key components: (i) global perception scene reconstruction, which provides locally consistent, up-to-scale geometry from a single RGB view and fuses multi-view projections into a coherent global 3D scene; (ii) a render-and-score active perception strategy, which dynamically selects next-best-views to reveal occluded regions; and (iii) an online metric alignment module that calibrates VGGT predictions against robot kinematics to ensure physical scale consistency. Building on these tailor-designed modules, GraspView performs best-view global grasping, fusing multi-view reconstructions and leveraging GraspNet for robust execution. Experiments on diverse tabletop objects demonstrate that GraspView significantly outperforms both RGB-D and single-view RGB baselines, especially under heavy occlusion, near-field sensing, and with transparent objects. These results highlight GraspView as a practical and versatile alternative to RGB-D pipelines, enabling reliable grasping in unstructured real-world environments. 

**Abstract (ZH)**: 基于RGB的机器人抓取pipeline：在杂乱环境中实现准确操作的GraspView 

---
# BoRe-Depth: Self-supervised Monocular Depth Estimation with Boundary Refinement for Embedded Systems 

**Title (ZH)**: BoRe-Depth: 面向嵌入式系统的边界细化自监督单目深度估计 

**Authors**: Chang Liu, Juan Li, Sheng Zhang, Chang Liu, Jie Li, Xu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04388)  

**Abstract**: Depth estimation is one of the key technologies for realizing 3D perception in unmanned systems. Monocular depth estimation has been widely researched because of its low?cost advantage, but the existing methods face the challenges of poor depth estimation performance and blurred object boundaries on embedded systems. In this paper, we propose a novel monocular depth estimation model, BoRe-Depth, which contains only 8.7M parameters. It can accurately estimate depth maps on embedded systems and significantly improves boundary quality. Firstly, we design an Enhanced Feature Adaptive Fusion Module (EFAF) which adaptively fuses depth features to enhance boundary detail representation. Secondly, we integrate semantic knowledge into the encoder to improve the object recognition and boundary perception capabilities. Finally, BoRe-Depth is deployed on NVIDIA Jetson Orin, and runs efficiently at 50.7 FPS. We demonstrate that the proposed model significantly outperforms previous lightweight models on multiple challenging datasets, and we provide detailed ablation studies for the proposed methods. The code is available at this https URL. 

**Abstract (ZH)**: 单目深度估计是实现无人系统三维感知的关键技术之一。虽然单目深度估计因其低成本优势而受到广泛研究，但现有方法在嵌入式系统上面临着深度估计性能差和物体边界模糊的挑战。本文提出了一种新型单目深度估计模型BoRe-Depth，该模型仅包含8.7M参数，可以在嵌入式系统上准确估计深度图，并显著提高边界质量。首先，我们设计了一种增强特征自适应融合模块(EFAF)，以自适应融合深度特征，增强边界细节表示。其次，我们将语义知识集成到编码器中，以提高物体识别和边界感知能力。最后，BoRe-Depth 在 NVIDIA Jetson Orin 上部署，并以每秒50.7帧的速度高效运行。实验结果表明，所提模型在多个具有挑战性的数据集上显著优于 Previous Lightweight 模型，并提供了详尽的方法消融研究。代码可在以下链接获取。 

---
# GUI-360: A Comprehensive Dataset and Benchmark for Computer-Using Agents 

**Title (ZH)**: GUI-360: 一个全面的数据集和评估基准供计算机使用代理使用 

**Authors**: Jian Mu, Chaoyun Zhang, Chiming Ni, Lu Wang, Bo Qiao, Kartik Mathur, Qianhui Wu, Yuhang Xie, Xiaojun Ma, Mengyu Zhou, Si Qin, Liqun Li, Yu Kang, Minghua Ma, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04307)  

**Abstract**: We introduce GUI-360$^\circ$, a large-scale, comprehensive dataset and benchmark suite designed to advance computer-using agents (CUAs). CUAs present unique challenges and is constrained by three persistent gaps: a scarcity of real-world CUA tasks, the lack of automated collection-and-annotation pipelines for multi-modal trajectories, and the absence of a unified benchmark that jointly evaluates GUI grounding, screen parsing, and action prediction.
GUI-360$^\circ$ addresses these gaps with an LLM-augmented, largely automated pipeline for query sourcing, environment-template construction, task instantiation, batched execution, and LLM-driven quality filtering. The released corpus contains over 1.2M executed action steps across thousands of trajectories in popular Windows office applications, and includes full-resolution screenshots, accessibility metadata when available, instantiated goals, intermediate reasoning traces, and both successful and failed action trajectories. The dataset supports three canonical tasks, GUI grounding, screen parsing, and action prediction, and a hybrid GUI+API action space that reflects modern agent designs. Benchmarking state-of-the-art vision--language models on GUI-360$^\circ$ reveals substantial out-of-the-box shortcomings in grounding and action prediction; supervised fine-tuning and reinforcement learning yield significant gains but do not close the gap to human-level reliability. We release GUI-360$^\circ$ and accompanying code to facilitate reproducible research and accelerate progress on robust desktop CUAs.
The full dataset has been made public on this https URL. 

**Abstract (ZH)**: GUI-360$^\circ$: 一个用于促进计算机使用代理（CUA）发展的大规模综合性数据集和基准套件 

---
# MusRec: Zero-Shot Text-to-Music Editing via Rectified Flow and Diffusion Transformers 

**Title (ZH)**: MusRec: 零样本文本到音乐编辑通过矫正流和扩散变换器 

**Authors**: Ali Boudaghi, Hadi Zare  

**Link**: [PDF](https://arxiv.org/pdf/2511.04376)  

**Abstract**: Music editing has emerged as an important and practical area of artificial intelligence, with applications ranging from video game and film music production to personalizing existing tracks according to user preferences. However, existing models face significant limitations, such as being restricted to editing synthesized music generated by their own models, requiring highly precise prompts, or necessitating task-specific retraining, thus lacking true zero-shot capability. Leveraging recent advances in rectified flow and diffusion transformers, we introduce MusRec, the first zero-shot text-to-music editing model capable of performing diverse editing tasks on real-world music efficiently and effectively. Experimental results demonstrate that our approach outperforms existing methods in preserving musical content, structural consistency, and editing fidelity, establishing a strong foundation for controllable music editing in real-world scenarios. 

**Abstract (ZH)**: 音乐编辑已成为人工智能的一个重要且实用的研究领域，应用范围从视频游戏和电影音乐制作到根据用户偏好个性化现有曲目。然而，现有模型面临显著限制，如仅能编辑自身模型生成的合成音乐、需要高度精确的提示或需要针对特定任务重新训练，因而缺乏真正的零样本能力。借助最近在校正流和扩散变换器方面的进展，我们提出了MusRec，这是第一个能够在高效且有效地对真实音乐进行多样化编辑任务的零样本文本到音乐编辑模型。实验结果表明，我们的方法在保留音乐内容、结构一致性和编辑保真度方面优于现有方法，为现实场景中的可控音乐编辑奠定了坚实的基础。 

---
# Deep learning-based object detection of offshore platforms on Sentinel-1 Imagery and the impact of synthetic training data 

**Title (ZH)**: 基于深度学习的Sentinel-1影像海上平台目标检测及其合成训练数据影响研究 

**Authors**: Robin Spanier, Thorsten Hoeser, Claudia Kuenzer  

**Link**: [PDF](https://arxiv.org/pdf/2511.04304)  

**Abstract**: The recent and ongoing expansion of marine infrastructure, including offshore wind farms, oil and gas platforms, artificial islands, and aquaculture facilities, highlights the need for effective monitoring systems. The development of robust models for offshore infrastructure detection relies on comprehensive, balanced datasets, but falls short when samples are scarce, particularly for underrepresented object classes, shapes, and sizes. By training deep learning-based YOLOv10 object detection models with a combination of synthetic and real Sentinel-1 satellite imagery acquired in the fourth quarter of 2023 from four regions (Caspian Sea, South China Sea, Gulf of Guinea, and Coast of Brazil), this study investigates the use of synthetic training data to enhance model performance. We evaluated this approach by applying the model to detect offshore platforms in three unseen regions (Gulf of Mexico, North Sea, Persian Gulf) and thereby assess geographic transferability. This region-holdout evaluation demonstrated that the model generalises beyond the training areas. In total, 3,529 offshore platforms were detected, including 411 in the North Sea, 1,519 in the Gulf of Mexico, and 1,593 in the Persian Gulf. The model achieved an F1 score of 0.85, which improved to 0.90 upon incorporating synthetic data. We analysed how synthetic data enhances the representation of unbalanced classes and overall model performance, taking a first step toward globally transferable detection of offshore infrastructure. This study underscores the importance of balanced datasets and highlights synthetic data generation as an effective strategy to address common challenges in remote sensing, demonstrating the potential of deep learning for scalable, global offshore infrastructure monitoring. 

**Abstract (ZH)**: 最近和正在进行的海洋基础设施扩展，包括海上风电场、石油和天然气平台、人工岛和水产养殖设施，突显了有效监测系统的需求。基于综合平衡数据集开发 robust 的离岸基础设施检测模型依赖于全面的样本，但在样本稀缺时仍然不足，特别是在未充分代表的物体类别、形状和大小方面。通过使用在2023年第四季度从四个区域（里海、南中国海、几内亚湾和巴西海岸）获取的合成和真实 Sentinel-1 卫星图像训练基于深度学习的 YOLOv10 对象检测模型，本研究探讨了使用合成训练数据以增强模型性能的方法。我们通过将模型应用于三个未见区域（墨西哥湾、北海、波斯湾）的离岸平台检测来评估其地理转移性。这种区域保留评估表明，模型可以泛化到训练区域之外。总共检测到3,529个离岸平台，包括北海的411个、墨西哥湾的1,519个和波斯湾的1,593个。模型的F1分数为0.85，在结合合成数据后提升至0.90。我们分析了合成数据如何增强不平衡类别的表示以及整体模型性能，为迈向全球离岸基础设施检测奠定了第一步。本研究强调了平衡数据集的重要性，并突显了生成合成数据作为一种有效策略，以应对遥感中常见的挑战，展示了深度学习在可扩展和全球离岸基础设施监测中的潜力。 

---
# Proto-LeakNet: Towards Signal-Leak Aware Attribution in Synthetic Human Face Imagery 

**Title (ZH)**: Proto-LeakNet：面向合成人类面部图像中的信号泄漏可追溯性分析 

**Authors**: Claudio Giusti, Luca Guarnera, Sebastiano Battiato  

**Link**: [PDF](https://arxiv.org/pdf/2511.04260)  

**Abstract**: The growing sophistication of synthetic image and deepfake generation models has turned source attribution and authenticity verification into a critical challenge for modern computer vision systems. Recent studies suggest that diffusion pipelines unintentionally imprint persistent statistical traces, known as signal leaks, within their outputs, particularly in latent representations. Building on this observation, we propose Proto-LeakNet, a signal-leak-aware and interpretable attribution framework that integrates closed-set classification with a density-based open-set evaluation on the learned embeddings, enabling analysis of unseen generators without retraining. Operating in the latent domain of diffusion models, our method re-simulates partial forward diffusion to expose residual generator-specific cues. A temporal attention encoder aggregates multi-step latent features, while a feature-weighted prototype head structures the embedding space and enables transparent attribution. Trained solely on closed data and achieving a Macro AUC of 98.13%, Proto-LeakNet learns a latent geometry that remains robust under post-processing, surpassing state-of-the-art methods, and achieves strong separability between known and unseen generators. These results demonstrate that modeling signal-leak bias in latent space enables reliable and interpretable AI-image and deepfake forensics. The code for the whole work will be available upon submission. 

**Abstract (ZH)**: 合成图像和深度伪造生成模型日益 sophistication 的发展使得源归属和真实性验证成为现代计算机视觉系统中的关键挑战。基于这一观察，我们提出了一种信号泄漏意识和可解释的归属框架 Proto-LeakNet，该框架结合了基于密度的开放集评估和已学习嵌入的封闭集分类，能够在无需重新训练的情况下分析未见过的生成器。我们的方法在扩散模型的潜在域中运行，通过部分前向扩散重新模拟来暴露残余的生成器特定线索。时间注意力编码器聚合多步潜在特征，而特征加权原型头部结构化嵌入空间并实现透明归属。Proto-LeakNet 仅在封闭数据上训练，并实现宏AUC为98.13%，学习的潜在几何形态在后处理下保持稳健，超越了现有方法，并在已知和未见过的生成器之间实现了强大的可分性。这些结果表明，在潜在空间中建模信号泄漏偏差能够实现可靠和可解释的AI图像和深度伪造取证。完整代码将在提交后提供。 

---
# MedSapiens: Taking a Pose to Rethink Medical Imaging Landmark Detection 

**Title (ZH)**: MedSapiens: 以新颖视角重思医学成像标志点检测 

**Authors**: Marawan Elbatel, Anbang Wang, Keyuan Liu, Kaouther Mouheb, Enrique Almar-Munoz, Lizhuo Lin, Yanqi Yang, Karim Lekadir, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.04255)  

**Abstract**: This paper does not introduce a novel architecture; instead, it revisits a fundamental yet overlooked baseline: adapting human-centric foundation models for anatomical landmark detection in medical imaging. While landmark detection has traditionally relied on domain-specific models, the emergence of large-scale pre-trained vision models presents new opportunities. In this study, we investigate the adaptation of Sapiens, a human-centric foundation model designed for pose estimation, to medical imaging through multi-dataset pretraining, establishing a new state of the art across multiple datasets. Our proposed model, MedSapiens, demonstrates that human-centric foundation models, inherently optimized for spatial pose localization, provide strong priors for anatomical landmark detection, yet this potential has remained largely untapped. We benchmark MedSapiens against existing state-of-the-art models, achieving up to 5.26% improvement over generalist models and up to 21.81% improvement over specialist models in the average success detection rate (SDR). To further assess MedSapiens adaptability to novel downstream tasks with few annotations, we evaluate its performance in limited-data settings, achieving 2.69% improvement over the few-shot state of the art in SDR. Code and model weights are available at this https URL . 

**Abstract (ZH)**: 本文并未引入新的架构，而是重新审视了一个被忽视的基础baseline：将以人类为中心的基础模型适应于医学成像中的解剖标志点检测。尽管传统的地标检测依赖于领域特定模型，但大规模预训练视觉模型的出现带来了新的机遇。本研究通过多数据集预训练探讨了Sapiens（一种为姿态估计设计的人类为中心的基础模型）在医学成像中的适应性，在多个数据集上达到了新的最佳性能。我们提出的MedSapiens模型表明，以人类为中心的基础模型，由于其在空间姿态定位方面的优化，为解剖标志点检测提供了强烈先验，但这一潜力仍远未得到开发。我们在现有最佳模型上对MedSapiens进行基准测试，平均成功检测率（SDR）相较于通用模型提高了5.26%，相较于专家模型提高了21.81%。为更进一步评估MedSapiens在少量标注数据下对新下游任务的适应性，我们在有限数据设置下对其性能进行了评估，相较于少量标注数据的最佳表现提高了2.69%的SDR。代码和模型权重可从以下链接获取。 

---
# Systematic Evaluation of Preprocessing Techniques for Accurate Image Registration in Digital Pathology 

**Title (ZH)**: 数字病理学中准确图像配准预处理技术的系统评估 

**Authors**: Fatemehzahra Darzi, Rodrigo Escobar Diaz Guerrero, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2511.04171)  

**Abstract**: Image registration refers to the process of spatially aligning two or more images by mapping them into a common coordinate system, so that corresponding anatomical or tissue structures are matched across images. In digital pathology, registration enables direct comparison and integration of information from different stains or imaging modalities, sup-porting applications such as biomarker analysis and tissue reconstruction. Accurate registration of images from different modalities is an essential step in digital pathology. In this study, we investigated how various color transformation techniques affect image registration between hematoxylin and eosin (H&E) stained images and non-linear multimodal images. We used a dataset of 20 tissue sample pairs, with each pair undergoing several preprocessing steps, including different color transformation (CycleGAN, Macenko, Reinhard, Vahadane), inversion, contrast adjustment, intensity normalization, and denoising. All images were registered using the VALIS registration method, which first applies rigid registration and then performs non-rigid registration in two steps on both low and high-resolution images. Registration performance was evaluated using the relative Target Registration Error (rTRE). We reported the median of median rTRE values (MMrTRE) and the average of median rTRE values (AMrTRE) for each method. In addition, we performed a custom point-based evaluation using ten manually selected key points. Registration was done separately for two scenarios, using either the original or inverted multimodal images. In both scenarios, CycleGAN color transformation achieved the lowest registration errors, while the other methods showed higher errors. These findings show that applying color transformation before registration improves alignment between images from different modalities and supports more reliable analysis in digital pathology. 

**Abstract (ZH)**: 图像配准指的是将两张或多张图像在共同坐标系中进行空间对齐的过程，使得图像中的相应解剖结构或组织结构在图像间匹配。在数字病理学中，配准使得不同染色或成像模态的信息可以直接进行比较和整合，支持如生物标记物分析和组织重建等应用。不同模态图像的准确配准是数字病理学中的一个关键步骤。在本研究中，我们探讨了各种颜色变换技术如何影响苏木精和曙红（H&E）染色图像与非线性多模态图像之间的配准效果。我们使用包含20对组织样本的数据集进行研究，每对样本都经过了不同的预处理步骤，包括不同的颜色变换（CycleGAN、Macenko、Reinhard、Vahadane）、反转、对比度调整、强度归一化和去噪。所有图像均使用VALIS配准方法进行配准，该方法首先进行刚性配准，然后分两步在低分辨率和高分辨率图像上进行非刚性配准。配准性能通过相对目标配准误差（rTRE）进行评估。我们报告了每种方法的中位数中值rTRE（MMrTRE）和中值rTRE的平均值（AMrTRE）。此外，我们还使用了十个手动选择的关键点进行了定制的基于点的评估。配准分别在两种场景下进行，使用原始图像或反转后的多模态图像。在两种场景下，CycleGAN颜色变换方法都实现了最低的配准误差，而其他方法的误差较高。这些发现表明，在配准前应用颜色变换可以提高不同模态图像之间的对齐效果，并在数字病理学中支持更可靠的分析。 

---
# Learning from Online Videos at Inference Time for Computer-Use Agents 

**Title (ZH)**: 基于推理时在线视频学习的计算机使用代理 

**Authors**: Yujian Liu, Ze Wang, Hao Chen, Ximeng Sun, Xiaodong Yu, Jialian Wu, Jiang Liu, Emad Barsoum, Zicheng Liu, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04137)  

**Abstract**: Computer-use agents can operate computers and automate laborious tasks, but despite recent rapid progress, they still lag behind human users, especially when tasks require domain-specific procedural knowledge about particular applications, platforms, and multi-step workflows. Humans can bridge this gap by watching video tutorials: we search, skim, and selectively imitate short segments that match our current subgoal. In this paper, we study how to enable computer-use agents to learn from online videos at inference time effectively. We propose a framework that retrieves and filters tutorial videos, converts them into structured demonstration trajectories, and dynamically selects trajectories as in-context guidance during execution. Particularly, using a VLM, we infer UI actions, segment videos into short subsequences of actions, and assign each subsequence a textual objective. At inference time, a two-stage selection mechanism dynamically chooses a single trajectory to add in context at each step, focusing the agent on the most helpful local guidance for its next decision. Experiments on two widely used benchmarks show that our framework consistently outperforms strong base agents and variants that use only textual tutorials or transcripts. Analyses highlight the importance of trajectory segmentation and selection, action filtering, and visual information, suggesting that abundant online videos can be systematically distilled into actionable guidance that improves computer-use agents at inference time. Our code is available at this https URL. 

**Abstract (ZH)**: 基于在线视频的计算机使用代理学习框架 

---
# DMSORT: An efficient parallel maritime multi-object tracking architecture for unmanned vessel platforms 

**Title (ZH)**: DMSORT: 一种高效的并行海上多目标跟踪架构适用于无人船平台 

**Authors**: Shengyu Tang, Zeyuan Lu, Jiazhi Dong, Changdong Yu, Xiaoyu Wang, Yaohui Lyu, Weihao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2511.04128)  

**Abstract**: Accurate perception of the marine environment through robust multi-object tracking (MOT) is essential for ensuring safe vessel navigation and effective maritime surveillance. However, the complicated maritime environment often causes camera motion and subsequent visual degradation, posing significant challenges to MOT. To address this challenge, we propose an efficient Dual-branch Maritime SORT (DMSORT) method for maritime MOT. The core of the framework is a parallel tracker with affine compensation, which incorporates an object detection and re-identification (ReID) branch, along with a dedicated branch for dynamic camera motion estimation. Specifically, a Reversible Columnar Detection Network (RCDN) is integrated into the detection module to leverage multi-level visual features for robust object detection. Furthermore, a lightweight Transformer-based appearance extractor (Li-TAE) is designed to capture global contextual information and generate robust appearance features. Another branch decouples platform-induced and target-intrinsic motion by constructing a projective transformation, applying platform-motion compensation within the Kalman filter, and thereby stabilizing true object trajectories. Finally, a clustering-optimized feature fusion module effectively combines motion and appearance cues to ensure identity consistency under noise, occlusion, and drift. Extensive evaluations on the Singapore Maritime Dataset demonstrate that DMSORT achieves state-of-the-art performance. Notably, DMSORT attains the fastest runtime among existing ReID-based MOT frameworks while maintaining high identity consistency and robustness to jitter and occlusion. Code is available at: this https URL. 

**Abstract (ZH)**: 通过稳健多目标跟踪实现准确的海洋环境感知对于确保船舶安全导航和有效的海上监控至关重要。然而，复杂的海洋环境常常导致相机运动和随后的视觉退化，对多目标跟踪（MOT）构成了重大挑战。为应对这一挑战，我们提出了一种高效的双分支海上 SORT（DMSORT）方法用于海上MOT。该框架的核心是一个并行跟踪器，该跟踪器包含仿射补偿，并结合了对象检测和再识别（ReID）分支，以及一个专门用于动态相机运动估计的分支。具体来说，将可逆柱状检测网络（RCDN）集成到检测模块中，利用多级视觉特征实现稳健的对象检测。此外，设计了一种轻量级的基于Transformer的外观提取器（Li-TAE），以捕获全局上下文信息并生成稳健的外观特征。另一个分支通过构建投影变换，将平台运动补偿应用于卡尔曼滤波器内，从而稳定真实对象轨迹。最终，一个聚类优化特征融合模块有效地结合了运动和外观线索，以确保在噪声、遮挡和漂移情况下身份一致性。在新加坡海上数据集上的广泛评估表明，DMSORT 达到了最先进的性能。值得注意的是，DMSORT 在现有的ReID基MOT框架中实现了最快的运行时间，同时保持了高水平的身份一致性和对抖动和遮挡的鲁棒性。代码可从以下链接获得：this https URL。 

---
# Automated Tennis Player and Ball Tracking with Court Keypoints Detection (Hawk Eye System) 

**Title (ZH)**: 基于球场关键点检测的自动网球运动员和球追踪（Hawk Eye系统） 

**Authors**: Venkata Manikanta Desu, Syed Fawaz Ali  

**Link**: [PDF](https://arxiv.org/pdf/2511.04126)  

**Abstract**: This study presents a complete pipeline for automated tennis match analysis. Our framework integrates multiple deep learning models to detect and track players and the tennis ball in real time, while also identifying court keypoints for spatial reference. Using YOLOv8 for player detection, a custom-trained YOLOv5 model for ball tracking, and a ResNet50-based architecture for court keypoint detection, our system provides detailed analytics including player movement patterns, ball speed, shot accuracy, and player reaction times. The experimental results demonstrate robust performance in varying court conditions and match scenarios. The model outputs an annotated video along with detailed performance metrics, enabling coaches, broadcasters, and players to gain actionable insights into the dynamics of the game. 

**Abstract (ZH)**: 本研究提出了一套完整的自动化网球比赛分析pipeline。我们的框架结合了多个深度学习模型，用于实时检测和追踪球员及网球，并识别比赛场地的关键点以便于空间参考。通过使用YOLOv8进行球员检测、自训练的YOLOv5模型进行网球追踪，以及基于ResNet50的架构进行关键点检测，我们的系统提供了包括球员运动模式、击球速度、击球准确性和球员反应时间在内的详细分析。实验结果表明，该模型在不同的场地条件和比赛场景下表现出色。模型输出带有标注的视频和详细的表现指标，使得教练、广播员和运动员能够获得有关比赛动态的 actionable 洞察。 

---
# Pediatric Appendicitis Detection from Ultrasound Images 

**Title (ZH)**: 儿童阑尾炎的超声图像检测 

**Authors**: Fatemeh Hosseinabadi, Seyedhassan Sharifi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04069)  

**Abstract**: Pediatric appendicitis remains one of the most common causes of acute abdominal pain in children, and its diagnosis continues to challenge clinicians due to overlapping symptoms and variable imaging quality. This study aims to develop and evaluate a deep learning model based on a pretrained ResNet architecture for automated detection of appendicitis from ultrasound images. We used the Regensburg Pediatric Appendicitis Dataset, which includes ultrasound scans, laboratory data, and clinical scores from pediatric patients admitted with abdominal pain to Children Hospital. Hedwig in Regensburg, Germany. Each subject had 1 to 15 ultrasound views covering the right lower quadrant, appendix, lymph nodes, and related structures. For the image based classification task, ResNet was fine tuned to distinguish appendicitis from non-appendicitis cases. Images were preprocessed by normalization, resizing, and augmentation to enhance generalization. The proposed ResNet model achieved an overall accuracy of 93.44, precision of 91.53, and recall of 89.8, demonstrating strong performance in identifying appendicitis across heterogeneous ultrasound views. The model effectively learned discriminative spatial features, overcoming challenges posed by low contrast, speckle noise, and anatomical variability in pediatric imaging. 

**Abstract (ZH)**: 儿童阑尾炎仍然是导致儿童急性腹痛的最常见原因之一，其诊断由于症状重叠和影像质量变异仍挑战着临床医生。本研究旨在基于预训练的ResNet架构开发并评估一种自动化检测阑尾炎的深度学习模型，该模型基于儿童医院Heidwig在德国雷根斯堡收集的儿科阑尾炎数据集中的超声图像、实验室数据和临床评分。每个受试者有1到15张覆盖右下腹部、阑尾、淋巴结及相关结构的超声图像。对于基于图像的分类任务，ResNet被微调以区分阑尾炎与非阑尾炎病例。图像经过归一化、调整大小和增强预处理，以提高泛化能力。所提出的ResNet模型在分类任务上的总体准确率为93.44%，精确率为91.53%，召回率为89.8%，展示了在不同超声视图中识别阑尾炎的强大性能。模型有效地学习了区分性空间特征，克服了低对比度、 speckle 噪声和儿童影像中解剖变异带来的挑战。 

---
# Improving Multi-View Reconstruction via Texture-Guided Gaussian-Mesh Joint Optimization 

**Title (ZH)**: 基于纹理引导的高斯网格联合优化的多视图重建改进 

**Authors**: Zhejia Cai, Puhua Jiang, Shiwei Mao, Hongkun Cao, Ruqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.03950)  

**Abstract**: Reconstructing real-world objects from multi-view images is essential for applications in 3D editing, AR/VR, and digital content creation. Existing methods typically prioritize either geometric accuracy (Multi-View Stereo) or photorealistic rendering (Novel View Synthesis), often decoupling geometry and appearance optimization, which hinders downstream editing tasks. This paper advocates an unified treatment on geometry and appearance optimization for seamless Gaussian-mesh joint optimization. More specifically, we propose a novel framework that simultaneously optimizes mesh geometry (vertex positions and faces) and vertex colors via Gaussian-guided mesh differentiable rendering, leveraging photometric consistency from input images and geometric regularization from normal and depth maps. The obtained high-quality 3D reconstruction can be further exploit in down-stream editing tasks, such as relighting and shape deformation. The code will be publicly available upon acceptance. 

**Abstract (ZH)**: 从多视角图像重建真实世界对象对于三维编辑、AR/VR和数字内容创作的应用至关重要。现有方法通常在几何准确性（多视图立体）和光orealistic渲染（新颖视图合成）之间权衡，常常将几何优化和外观优化解耦，这阻碍了下游编辑任务。本文提倡对几何和外观优化进行统一处理，实现无缝的高斯网格联合优化。具体而言，我们提出了一种新型框架，通过基于高斯的网格可微渲染同时优化网格几何（顶点位置和面）和顶点颜色，并利用输入图像的光度一致性及法线图和深度图的几何正则化。获得的高质量3D重建可以进一步用于下游编辑任务，如重新光照和形状变形。接受后代码将公开发布。 

---
# CORE - A Cell-Level Coarse-to-Fine Image Registration Engine for Multi-stain Image Alignment 

**Title (ZH)**: CORE - 一种细胞级粗细粒度图像配准引擎用于多染色图像对齐 

**Authors**: Esha Sadia Nasir, Behnaz Elhaminia, Mark Eastwood, Catherine King, Owen Cain, Lorraine Harper, Paul Moss, Dimitrios Chanouzas, David Snead, Nasir Rajpoot, Adam Shephard, Shan E Ahmed Raza  

**Link**: [PDF](https://arxiv.org/pdf/2511.03826)  

**Abstract**: Accurate and efficient registration of whole slide images (WSIs) is essential for high-resolution, nuclei-level analysis in multi-stained tissue slides. We propose a novel coarse-to-fine framework CORE for accurate nuclei-level registration across diverse multimodal whole-slide image (WSI) datasets. The coarse registration stage leverages prompt-based tissue mask extraction to effectively filter out artefacts and non-tissue regions, followed by global alignment using tissue morphology and ac- celerated dense feature matching with a pre-trained feature extractor. From the coarsely aligned slides, nuclei centroids are detected and subjected to fine-grained rigid registration using a custom, shape-aware point-set registration model. Finally, non-rigid alignment at the cellular level is achieved by estimating a non-linear dis- placement field using Coherent Point Drift (CPD). Our approach benefits from automatically generated nuclei that enhance the accuracy of deformable registra- tion and ensure precise nuclei-level correspondence across modalities. The pro- posed model is evaluated on three publicly available WSI registration datasets, and two private datasets. We show that CORE outperforms current state-of-the-art methods in terms of generalisability, precision, and robustness in bright-field and immunofluorescence microscopy WSIs 

**Abstract (ZH)**: 准确高效的全slide图像(WSI)注册对于多染色组织切片的高分辨率、核水平分析至关重要。我们提出了一种新的自上而下的框架CORE，用于跨多样化的多模态WSI数据集实现精确的核水平注册。粗注册阶段利用基于提示的组织掩模提取有效滤除伪影和非组织区域，随后使用组织形态进行全局对齐，并利用预训练的特征提取器加速密集特征匹配。从粗对齐后的切片中，检测核质心并使用自定义的形状感知点集注册模型进行细致的刚性对齐。最后，通过相干点漂移(CPD)估计非线性位移场，实现细胞水平的非刚性对齐。我们的方法得益于自动生成的核，这些核提高了可变形注册的准确性，并确保了模态间的精确核水平对应关系。所提模型在三个公开的WSI注册数据集和两个私有数据集上进行了评估，结果显示CORE在明场和免疫荧光显微镜WSI中的泛化性、精确性和鲁棒性方面优于当前最先进的方法。 

---
# A convolutional neural network deep learning method for model class selection 

**Title (ZH)**: 基于卷积神经网络的深度学习模型类选择方法 

**Authors**: Marios Impraimakis  

**Link**: [PDF](https://arxiv.org/pdf/2511.03743)  

**Abstract**: The response-only model class selection capability of a novel deep convolutional neural network method is examined herein in a simple, yet effective, manner. Specifically, the responses from a unique degree of freedom along with their class information train and validate a one-dimensional convolutional neural network. In doing so, the network selects the model class of new and unlabeled signals without the need of the system input information, or full system identification. An optional physics-based algorithm enhancement is also examined using the Kalman filter to fuse the system response signals using the kinematics constraints of the acceleration and displacement data. Importantly, the method is shown to select the model class in slight signal variations attributed to the damping behavior or hysteresis behavior on both linear and nonlinear dynamic systems, as well as on a 3D building finite element model, providing a powerful tool for structural health monitoring applications. 

**Abstract (ZH)**: 一种新型深度卷积神经网络方法的仅响应模型类选择能力研究 

---
