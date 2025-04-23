# Visual Place Cell Encoding: A Computational Model for Spatial Representation and Cognitive Mapping 

**Title (ZH)**: 视觉位置细胞编码：空间表示与认知映射的计算模型 

**Authors**: Chance J. Hamilton, Alfredo Weitzenfeld  

**Link**: [PDF](https://arxiv.org/pdf/2504.15953)  

**Abstract**: This paper presents the Visual Place Cell Encoding (VPCE) model, a biologically inspired computational framework for simulating place cell-like activation using visual input. Drawing on evidence that visual landmarks play a central role in spatial encoding, the proposed VPCE model activates visual place cells by clustering high-dimensional appearance features extracted from images captured by a robot-mounted camera. Each cluster center defines a receptive field, and activation is computed based on visual similarity using a radial basis function. We evaluate whether the resulting activation patterns correlate with key properties of biological place cells, including spatial proximity, orientation alignment, and boundary differentiation. Experiments demonstrate that the VPCE can distinguish between visually similar yet spatially distinct locations and adapt to environment changes such as the insertion or removal of walls. These results suggest that structured visual input, even in the absence of motion cues or reward-driven learning, is sufficient to generate place-cell-like spatial representations and support biologically inspired cognitive mapping. 

**Abstract (ZH)**: 基于视觉的地点细胞编码模型：生物学启发的视觉地点细胞激活计算框架 

---
# DERD-Net: Learning Depth from Event-based Ray Densities 

**Title (ZH)**: DERD-Net: 从事件式射线密度学习深度 

**Authors**: Diego de Oliveira Hitzges, Suman Ghosh, Guillermo Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2504.15863)  

**Abstract**: Event cameras offer a promising avenue for multi-view stereo depth estimation and Simultaneous Localization And Mapping (SLAM) due to their ability to detect blur-free 3D edges at high-speed and over broad illumination conditions. However, traditional deep learning frameworks designed for conventional cameras struggle with the asynchronous, stream-like nature of event data, as their architectures are optimized for discrete, image-like inputs. We propose a scalable, flexible and adaptable framework for pixel-wise depth estimation with event cameras in both monocular and stereo setups. The 3D scene structure is encoded into disparity space images (DSIs), representing spatial densities of rays obtained by back-projecting events into space via known camera poses. Our neural network processes local subregions of the DSIs combining 3D convolutions and a recurrent structure to recognize valuable patterns for depth prediction. Local processing enables fast inference with full parallelization and ensures constant ultra-low model complexity and memory costs, regardless of camera resolution. Experiments on standard benchmarks (MVSEC and DSEC datasets) demonstrate unprecedented effectiveness: (i) using purely monocular data, our method achieves comparable results to existing stereo methods; (ii) when applied to stereo data, it strongly outperforms all state-of-the-art (SOTA) approaches, reducing the mean absolute error by at least 42%; (iii) our method also allows for increases in depth completeness by more than 3-fold while still yielding a reduction in median absolute error of at least 30%. Given its remarkable performance and effective processing of event-data, our framework holds strong potential to become a standard approach for using deep learning for event-based depth estimation and SLAM. Project page: this https URL 

**Abstract (ZH)**: 事件摄像头为基于多视图立体深度估计和SLAM的应用提供了有前景的方法，由于其能在高速和广泛光照条件下检测无模糊3D边缘的能力。然而，传统为常规摄像头设计的深度学习框架难以处理事件数据的异步、流式性质，因为其架构优化是为了离散的、图像-like的输入。我们提出了一种适用于单目和立体设置的事件摄像头像素级深度估计的可扩展、灵活和适应性框架。3D场景结构编码在视差空间图像(DSIs)中表示，代表通过将事件回投影到空间中获得的光线的空间密度，其中已知相机姿态。我们的神经网络处理DSI的局部子区域，结合3D卷积和递归结构来识别深度预测有价值的模式。局部处理能使推理快速并实现全并行化，确保无论相机分辨率如何，模型复杂度和内存成本始终保持在超低水平。在标准基准数据集（MVSEC和DSEC）上的实验展示了前所未有的效果：(i) 使用纯单目数据，我们的方法达到与现有立体方法相当的结果；(ii) 当应用于立体数据时，它显著优于所有最新方法，绝对均值误差降低至少42%；(iii) 我们的方法还能使深度完整性提高超过三倍，同时仍能减少至少30%的中值绝对误差。鉴于其卓越的性能和对事件数据的有效处理，我们的框架具有成为深度学习用于事件驱动深度估计和SLAM的标准方法的强大潜力。项目页面: [这个链接](this https URL)。 

---
# Pose Optimization for Autonomous Driving Datasets using Neural Rendering Models 

**Title (ZH)**: 使用神经渲染模型的自主驾驶数据集姿态优化 

**Authors**: Quentin Herau, Nathan Piasco, Moussab Bennehar, Luis Rolado, Dzmitry Tsishkou, Bingbing Liu, Cyrille Migniot, Pascal Vasseur, Cédric Demonceaux  

**Link**: [PDF](https://arxiv.org/pdf/2504.15776)  

**Abstract**: Autonomous driving systems rely on accurate perception and localization of the ego car to ensure safety and reliability in challenging real-world driving scenarios. Public datasets play a vital role in benchmarking and guiding advancement in research by providing standardized resources for model development and evaluation. However, potential inaccuracies in sensor calibration and vehicle poses within these datasets can lead to erroneous evaluations of downstream tasks, adversely impacting the reliability and performance of the autonomous systems. To address this challenge, we propose a robust optimization method based on Neural Radiance Fields (NeRF) to refine sensor poses and calibration parameters, enhancing the integrity of dataset benchmarks. To validate improvement in accuracy of our optimized poses without ground truth, we present a thorough evaluation process, relying on reprojection metrics, Novel View Synthesis rendering quality, and geometric alignment. We demonstrate that our method achieves significant improvements in sensor pose accuracy. By optimizing these critical parameters, our approach not only improves the utility of existing datasets but also paves the way for more reliable autonomous driving models. To foster continued progress in this field, we make the optimized sensor poses publicly available, providing a valuable resource for the research community. 

**Abstract (ZH)**: 自主驾驶系统依赖于对 ego 车辆准确感知和定位，以确保在复杂的真实驾驶场景中实现安全性和可靠性。公共数据集在基准测试和指导研究方面发挥着重要作用，通过提供标准化资源支持模型开发和评估。然而，这些数据集中传感器校准和车辆姿态的潜在不准确可能会导致下游任务评估错误，从而影响自主系统的可靠性和性能。为解决这一挑战，我们提出一种基于神经辐射场（NeRF）的稳健优化方法，以 refinement 传感器姿态和校准参数，提高数据集基准的完整性。为验证优化姿态准确性的提升，我们采用了详细的评估过程，基于重投影指标、新颖视角合成渲染质量以及几何对齐。我们证明，我们的方法在传感器姿态准确性方面取得了显著提升。通过优化这些关键参数，我们的方法不仅提高了现有数据集的实用性，也为更可靠的自主驾驶模型铺平了道路。为了促进该领域的持续进展，我们公开发布了优化后的传感器姿态，为研究社区提供了一项宝贵资源。 

---
# Solving New Tasks by Adapting Internet Video Knowledge 

**Title (ZH)**: 通过适应互联网视频知识解决新任务 

**Authors**: Calvin Luo, Zilai Zeng, Yilun Du, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.15369)  

**Abstract**: Video generative models demonstrate great promise in robotics by serving as visual planners or as policy supervisors. When pretrained on internet-scale data, such video models intimately understand alignment with natural language, and can thus facilitate generalization to novel downstream behavior through text-conditioning. However, they may not be sensitive to the specificities of the particular environment the agent inhabits. On the other hand, training video models on in-domain examples of robotic behavior naturally encodes environment-specific intricacies, but the scale of available demonstrations may not be sufficient to support generalization to unseen tasks via natural language specification. In this work, we investigate different adaptation techniques that integrate in-domain information with large-scale pretrained video models, and explore the extent to which they enable novel text-conditioned generalization for robotic tasks, while also considering their independent data and resource considerations. We successfully demonstrate across robotic environments that adapting powerful video models with small scales of example data can successfully facilitate generalization to novel behaviors. In particular, we present a novel adaptation strategy, termed Inverse Probabilistic Adaptation, that not only consistently achieves strong generalization performance across robotic tasks and settings, but also exhibits robustness to the quality of adaptation data, successfully solving novel tasks even when only suboptimal in-domain demonstrations are available. 

**Abstract (ZH)**: 视频生成模型在机器人领域通过作为视觉规划者或策略监督者展现出巨大潜力。当基于互联网规模的数据进行预训练时，这些视频模型能够深刻理解自然语言的对齐关系，并可以通过文本条件化促进对新颖下游行为的一般化。然而，它们可能对代理所处特定环境的具体特性不够敏感。另一方面，基于机器人行为领域的示例训练视频模型会自然地编码环境特定的复杂性，但可用的演示数据规模可能不足以支持通过自然语言规范的一般化到未见过的任务。在本工作中，我们研究了将领域内信息与大规模预训练视频模型集成的不同适应技术，并探讨了它们在何种程度上能够支持针对机器人任务的新颖文本条件化一般化，同时考虑各自的独立数据和资源要求。我们成功地展示了在多种机器人环境中，通过使用小规模的示例数据适应强大的视频模型能够有效促进对新颖行为的一般化。特别是，我们提出了一种新颖的适应策略，称为逆概率适应，这种策略不仅在机器人任务和环境的不同条件下持续实现了强泛化性能，而且对适应数据的质量表现出稳定性，在仅有限的次优领域内演示可用的情况下，成功解决了新颖任务。 

---
# Describe Anything: Detailed Localized Image and Video Captioning 

**Title (ZH)**: 描述万物：详尽局部的图像和视频字幕生成 

**Authors**: Long Lian, Yifan Ding, Yunhao Ge, Sifei Liu, Hanzi Mao, Boyi Li, Marco Pavone, Ming-Yu Liu, Trevor Darrell, Adam Yala, Yin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.16072)  

**Abstract**: Generating detailed and accurate descriptions for specific regions in images and videos remains a fundamental challenge for vision-language models. We introduce the Describe Anything Model (DAM), a model designed for detailed localized captioning (DLC). DAM preserves both local details and global context through two key innovations: a focal prompt, which ensures high-resolution encoding of targeted regions, and a localized vision backbone, which integrates precise localization with its broader context. To tackle the scarcity of high-quality DLC data, we propose a Semi-supervised learning (SSL)-based Data Pipeline (DLC-SDP). DLC-SDP starts with existing segmentation datasets and expands to unlabeled web images using SSL. We introduce DLC-Bench, a benchmark designed to evaluate DLC without relying on reference captions. DAM sets new state-of-the-art on 7 benchmarks spanning keyword-level, phrase-level, and detailed multi-sentence localized image and video captioning. 

**Abstract (ZH)**: 生成图像和视频中特定区域的详细和准确描述仍然是视觉-语言模型的一个基本挑战。我们介绍了详细的局部描述模型（DAM），该模型旨在进行详细的局部描述（DLC）。DAM通过两项关键创新保留了局部细节和全局上下文：焦点提示，确保目标区域的高分辨率编码；以及局部视觉主干，将精确定位与其更广泛的上下文相结合。为了解决高质量DLC数据的稀缺性，我们提出了一种基于半监督学习（SSL）的数据管道（DLC-SDP）。DLC-SDP从现有的分割数据集开始，并使用SSL扩展到未标注的网络图像。我们提出了DLC-Bench，这是一个无需依赖参考描述进行评估的标准测试平台。DAM在涵盖关键词级、短语级以及详细的多句局部图像和视频描述在内的7个基准测试中均取得了新的最佳性能。 

---
# Evaluating Vision Language Models (VLMs) for Radiology: A Comprehensive Analysis 

**Title (ZH)**: 评估视觉语言模型（VLMs）在放射学中的应用：一项全面分析 

**Authors**: Frank Li, Hari Trivedi, Bardia Khosravi, Theo Dapamede, Mohammadreza Chavoshi, Abdulhameed Dere, Rohan Satya Isaac, Aawez Mansuri, Janice Newsome, Saptarshi Purkayastha, Judy Gichoya  

**Link**: [PDF](https://arxiv.org/pdf/2504.16047)  

**Abstract**: Foundation models, trained on vast amounts of data using self-supervised techniques, have emerged as a promising frontier for advancing artificial intelligence (AI) applications in medicine. This study evaluates three different vision-language foundation models (RAD-DINO, CheXagent, and BiomedCLIP) on their ability to capture fine-grained imaging features for radiology tasks. The models were assessed across classification, segmentation, and regression tasks for pneumothorax and cardiomegaly on chest radiographs. Self-supervised RAD-DINO consistently excelled in segmentation tasks, while text-supervised CheXagent demonstrated superior classification performance. BiomedCLIP showed inconsistent performance across tasks. A custom segmentation model that integrates global and local features substantially improved performance for all foundation models, particularly for challenging pneumothorax segmentation. The findings highlight that pre-training methodology significantly influences model performance on specific downstream tasks. For fine-grained segmentation tasks, models trained without text supervision performed better, while text-supervised models offered advantages in classification and interpretability. These insights provide guidance for selecting foundation models based on specific clinical applications in radiology. 

**Abstract (ZH)**: 基于大规模数据自监督训练的视觉-语言基础模型在医学影像学应用中的进展：RAD-DINO、CheXagent和BiomedCLIP在肺炎和心脏肥大分割与分类任务中的性能评估 

---
# Ask2Loc: Learning to Locate Instructional Visual Answers by Asking Questions 

**Title (ZH)**: Ask2Loc: 学习通过提问来定位指令性视觉答案 

**Authors**: Chang Zong, Bin Li, Shoujun Zhou, Jian Wan, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15918)  

**Abstract**: Locating specific segments within an instructional video is an efficient way to acquire guiding knowledge. Generally, the task of obtaining video segments for both verbal explanations and visual demonstrations is known as visual answer localization (VAL). However, users often need multiple interactions to obtain answers that align with their expectations when using the system. During these interactions, humans deepen their understanding of the video content by asking themselves questions, thereby accurately identifying the location. Therefore, we propose a new task, named In-VAL, to simulate the multiple interactions between humans and videos in the procedure of obtaining visual answers. The In-VAL task requires interactively addressing several semantic gap issues, including 1) the ambiguity of user intent in the input questions, 2) the incompleteness of language in video subtitles, and 3) the fragmentation of content in video segments. To address these issues, we propose Ask2Loc, a framework for resolving In-VAL by asking questions. It includes three key modules: 1) a chatting module to refine initial questions and uncover clear intentions, 2) a rewriting module to generate fluent language and create complete descriptions, and 3) a searching module to broaden local context and provide integrated content. We conduct extensive experiments on three reconstructed In-VAL datasets. Compared to traditional end-to-end and two-stage methods, our proposed Ask2Loc can improve performance by up to 14.91 (mIoU) on the In-VAL task. Our code and datasets can be accessed at this https URL. 

**Abstract (ZH)**: 在获取视觉答案过程中的多轮交互定位（In-VAL）及Ask2Loc框架 

---
# Integrating Non-Linear Radon Transformation for Diabetic Retinopathy Grading 

**Title (ZH)**: 将非线性Radon变换集成用于糖尿病视网膜病变分级 

**Authors**: Farida Mohsen, Samir Belhaouari, Zubair Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.15883)  

**Abstract**: Diabetic retinopathy is a serious ocular complication that poses a significant threat to patients' vision and overall health. Early detection and accurate grading are essential to prevent vision loss. Current automatic grading methods rely heavily on deep learning applied to retinal fundus images, but the complex, irregular patterns of lesions in these images, which vary in shape and distribution, make it difficult to capture subtle changes. This study introduces RadFuse, a multi-representation deep learning framework that integrates non-linear RadEx-transformed sinogram images with traditional fundus images to enhance diabetic retinopathy detection and grading. Our RadEx transformation, an optimized non-linear extension of the Radon transform, generates sinogram representations to capture complex retinal lesion patterns. By leveraging both spatial and transformed domain information, RadFuse enriches the feature set available to deep learning models, improving the differentiation of severity levels. We conducted extensive experiments on two benchmark datasets, APTOS-2019 and DDR, using three convolutional neural networks (CNNs): ResNeXt-50, MobileNetV2, and VGG19. RadFuse showed significant improvements over fundus-image-only models across all three CNN architectures and outperformed state-of-the-art methods on both datasets. For severity grading across five stages, RadFuse achieved a quadratic weighted kappa of 93.24%, an accuracy of 87.07%, and an F1-score of 87.17%. In binary classification between healthy and diabetic retinopathy cases, the method reached an accuracy of 99.09%, precision of 98.58%, and recall of 99.6%, surpassing previously established models. These results demonstrate RadFuse's capacity to capture complex non-linear features, advancing diabetic retinopathy classification and promoting the integration of advanced mathematical transforms in medical image analysis. 

**Abstract (ZH)**: 糖尿病视网膜病变是一种严重的眼部并发症，对患者的视力和整体健康构成重大威胁。早期检测和准确分级是预防视力丧失的关键。当前的自动分级方法主要依赖于深度学习在视网膜 Fundus 图像上的应用，但这些图像中病变的复杂、不规则模式，其形状和分布各异，使得捕捉细微变化变得困难。本研究引入了 RadFuse，这是一种多表示深度学习框架，将非线性 RadEx 变换后的 sinogram 图像与传统 Fundus 图像相结合，以增强糖尿病视网膜病变的检测和分级。我们的 RadEx 变换是 Radon 变换的一种优化的非线性扩展，它生成 sinogram 表示以捕捉复杂的视网膜病变模式。通过利用空域和变换域信息，RadFuse 丰富了可供深度学习模型使用的特征集，从而提高了严重程度级别的区分能力。我们在两个基准数据集 APTOS-2019 和 DDR 上使用了三种卷积神经网络（CNN）：ResNeXt-50、MobileNetV2 和 VGG19 进行了广泛实验。RadFuse 在所有三种 CNN 架构中都显著优于仅使用 Fundus 图像的模型，并在两个数据集上均超过了最先进的方法。对于五级严重程度分级，RadFuse 达到了 93.24% 的加权二次 κ 值、87.07% 的准确率和 87.17% 的 F1 分数。在健康与糖尿病视网膜病变二分类中，该方法达到了 99.09% 的准确率、98.58% 的精确率和 99.6% 的召回率，超越了先前建立的模型。这些结果表明 RadFuse 在捕捉复杂非线性特征方面的能力，促进了糖尿病视网膜病变分类的发展，并推动了高级数学变换在医学图像分析中的应用。 

---
# Dynamic Intent Queries for Motion Transformer-based Trajectory Prediction 

**Title (ZH)**: 基于运动变换器的轨迹预测中的动态意图查询 

**Authors**: Tobias Demmler, Lennart Hartung, Andreas Tamke, Thao Dang, Alexander Hegai, Karsten Haug, Lars Mikelsons  

**Link**: [PDF](https://arxiv.org/pdf/2504.15766)  

**Abstract**: In autonomous driving, accurately predicting the movements of other traffic participants is crucial, as it significantly influences a vehicle's planning processes. Modern trajectory prediction models strive to interpret complex patterns and dependencies from agent and map data. The Motion Transformer (MTR) architecture and subsequent work define the most accurate methods in common benchmarks such as the Waymo Open Motion Benchmark. The MTR model employs pre-generated static intention points as initial goal points for trajectory prediction. However, the static nature of these points frequently leads to misalignment with map data in specific traffic scenarios, resulting in unfeasible or unrealistic goal points. Our research addresses this limitation by integrating scene-specific dynamic intention points into the MTR model. This adaptation of the MTR model was trained and evaluated on the Waymo Open Motion Dataset. Our findings demonstrate that incorporating dynamic intention points has a significant positive impact on trajectory prediction accuracy, especially for predictions over long time horizons. Furthermore, we analyze the impact on ground truth trajectories which are not compliant with the map data or are illegal maneuvers. 

**Abstract (ZH)**: 自主驾驶中，准确预测其他交通参与者的运动至关重要，因为这显著影响车辆的规划过程。现代轨迹预测模型力求从代理和地图数据中解释复杂的模式和依赖关系。Motion Transformer (MTR) 架构及其后续工作定义了在 Waymo 开放运动基准等通用基准测试中最准确的方法。MTR 模型使用预生成的静态意图点作为轨迹预测的初始目标点。然而，这些静态点的性质在特定交通场景中经常与地图数据不匹配，导致不现实的目标点。我们的研究通过将场景特定的动态意图点集成到 MTR 模型中，解决了这一限制。我们对 Waymo 开放运动数据集进行了训练和评估，结果显示，引入动态意图点对长时程轨迹预测准确性有显著的积极影响。此外，我们分析了这些动态意图点对不符合地图数据或不合法的操作的真实轨迹的影响。 

---
# CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting 

**Title (ZH)**: CAPTURe: 通过遮挡物体计数评估视觉语言模型的空间推理能力 

**Authors**: Atin Pothiraj, Elias Stengel-Eskin, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.15485)  

**Abstract**: Recognizing and reasoning about occluded (partially or fully hidden) objects is vital to understanding visual scenes, as occlusions frequently occur in real-world environments and act as obstacles for spatial comprehension. To test models' ability to reason about multiple occluded objects, we introduce a novel task, Counting Amodally for Patterns Through Unseen REgions (CAPTURe), which requires a model to count objects arranged in a pattern by inferring how the pattern continues behind an occluder (an object which blocks parts of the scene). CAPTURe requires both recognizing visual patterns and reasoning, making it a useful testbed for evaluating vision-language models (VLMs) on whether they understand occluded patterns and possess spatial understanding skills. By requiring models to reason about occluded objects, CAPTURe also tests VLMs' ability to form world models that would allow them to fill in missing information. CAPTURe consists of two parts: (1) CAPTURe-real, with manually filtered images of real objects in patterns and (2) CAPTURe-synthetic, a controlled diagnostic with generated patterned images. We evaluate four strong VLMs (GPT-4o, Intern-VL2, Molmo, and Qwen2-VL) on CAPTURe, finding that models struggle to count on both occluded and unoccluded patterns. Crucially, we find that models perform worse with occlusion, suggesting that VLMs are also deficient in inferring unseen spatial relationships: even the strongest VLMs like GPT-4o fail to count with occlusion. In contrast, we find that humans achieve very little error on CAPTURe. We also find that providing auxiliary information of occluded object locations increases performance, underscoring that the model error comes both from an inability to handle occlusion as well as difficulty counting in images. 

**Abstract (ZH)**: 识别和推理被遮挡（部分或完全遮挡）的对象对于理解视觉场景至关重要，因为遮挡在现实环境中经常发生，是空间理解的障碍。为了测试模型推理多个被遮挡对象的能力，我们引入了一个新的任务——通过未知区域识别模式中的物体数量（CAPTURe），该任务要求模型通过推断遮挡背后（阻止场景部分区域的对象）的模式如何继续来识别并计数以模式排列的物体。CAPTURe 要求同时识别视觉模式并推理，使其成为一个评估视觉语言模型（VLMs）是否理解被遮挡的模式和具备空间理解能力的有效测试平台。通过要求模型推理被遮挡的对象，CAPTURe 也测试了 VLMs 形成允许填补缺失信息的世界模型的能力。CAPTURe 包含两个部分：（1）CAPTURe-real，包含人工筛选的真实对象模式图像；（2）CAPTURe-synthetic，一个由生成的模式图像组成的受控诊断。我们评估了四种强大的 VLMs（GPT-4o、Intern-VL2、Molmo 和 Qwen2-VL）的 CAPTURe 性能，发现模型在被遮挡和未被遮挡的模式上都难以计数。关键的是，我们发现模型在遮挡情况下表现更差，表明 VLMs 在推断看不见的空间关系方面也存在缺陷：即使是最强的 VLMs 如 GPT-4o 也无法在遮挡情况下计数。相比之下，我们发现人类在 CAPTURe 上几乎没有错误。我们还发现，提供被遮挡物体位置的辅助信息可以提高性能，这表明模型错误不仅来自处理遮挡的能力不足，还来自于在图像中计数的难度。 

---
# Towards Understanding Camera Motions in Any Video 

**Title (ZH)**: 理解任意视频中的相机运动 toward理解任意视频中的相机运动 

**Authors**: Zhiqiu Lin, Siyuan Cen, Daniel Jiang, Jay Karhade, Hewei Wang, Chancharik Mitra, Tiffany Ling, Yuhan Huang, Sifan Liu, Mingyu Chen, Rushikesh Zawar, Xue Bai, Yilun Du, Chuang Gan, Deva Ramanan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15376)  

**Abstract**: We introduce CameraBench, a large-scale dataset and benchmark designed to assess and improve camera motion understanding. CameraBench consists of ~3,000 diverse internet videos, annotated by experts through a rigorous multi-stage quality control process. One of our contributions is a taxonomy of camera motion primitives, designed in collaboration with cinematographers. We find, for example, that some motions like "follow" (or tracking) require understanding scene content like moving subjects. We conduct a large-scale human study to quantify human annotation performance, revealing that domain expertise and tutorial-based training can significantly enhance accuracy. For example, a novice may confuse zoom-in (a change of intrinsics) with translating forward (a change of extrinsics), but can be trained to differentiate the two. Using CameraBench, we evaluate Structure-from-Motion (SfM) and Video-Language Models (VLMs), finding that SfM models struggle to capture semantic primitives that depend on scene content, while VLMs struggle to capture geometric primitives that require precise estimation of trajectories. We then fine-tune a generative VLM on CameraBench to achieve the best of both worlds and showcase its applications, including motion-augmented captioning, video question answering, and video-text retrieval. We hope our taxonomy, benchmark, and tutorials will drive future efforts towards the ultimate goal of understanding camera motions in any video. 

**Abstract (ZH)**: CameraBench：一个用于评估和提升相机运动理解的大规模数据集和基准 

---
# Enhancing DR Classification with Swin Transformer and Shifted Window Attention 

**Title (ZH)**: 使用Swin Transformer和移窗注意力增强DR分类 

**Authors**: Meher Boulaabi, Takwa Ben Aïcha Gader, Afef Kacem Echi, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2504.15317)  

**Abstract**: Diabetic retinopathy (DR) is a leading cause of blindness worldwide, underscoring the importance of early detection for effective treatment. However, automated DR classification remains challenging due to variations in image quality, class imbalance, and pixel-level similarities that hinder model training. To address these issues, we propose a robust preprocessing pipeline incorporating image cropping, Contrast-Limited Adaptive Histogram Equalization (CLAHE), and targeted data augmentation to improve model generalization and resilience. Our approach leverages the Swin Transformer, which utilizes hierarchical token processing and shifted window attention to efficiently capture fine-grained features while maintaining linear computational complexity. We validate our method on the Aptos and IDRiD datasets for multi-class DR classification, achieving accuracy rates of 89.65% and 97.40%, respectively. These results demonstrate the effectiveness of our model, particularly in detecting early-stage DR, highlighting its potential for improving automated retinal screening in clinical settings. 

**Abstract (ZH)**: 糖尿病视网膜病变（DR）是全球导致失明的主要原因，强调了早期检测的重要性以实现有效治疗。然而，由于图像质量、类别不平衡和像素级相似性导致的挑战，自动DR分类仍然困难。为解决这些问题，我们提出了一种鲁棒的预处理管道，结合了图像裁剪、限制对比度自适应直方图均衡化（CLAHE）和目标数据增强，以提高模型的泛化能力和鲁棒性。我们的方法利用了Swin Transformer，该方法通过分层令牌处理和移窗注意力高效地捕捉细粒度特征，并保持线性计算复杂度。我们在Aptos和IDRiD数据集上对多类DR分类进行了验证，分别达到了89.65%和97.40%的准确率。这些结果表明了我们模型的有效性，特别是在检测早期DR方面，强调了其在临床环境中的自动视网膜筛查中的潜在应用价值。 

---
