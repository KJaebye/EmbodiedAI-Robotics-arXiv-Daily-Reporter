# Data Shift of Object Detection in Autonomous Driving 

**Title (ZH)**: 自主驾驶中目标检测的数据偏移 

**Authors**: Lida Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11868)  

**Abstract**: With the widespread adoption of machine learning technologies in autonomous driving systems, their role in addressing complex environmental perception challenges has become increasingly crucial. However, existing machine learning models exhibit significant vulnerability, as their performance critically depends on the fundamental assumption that training and testing data satisfy the independent and identically distributed condition, which is difficult to guarantee in real-world applications. Dynamic variations in data distribution caused by seasonal changes, weather fluctuations lead to data shift problems in autonomous driving systems. This study investigates the data shift problem in autonomous driving object detection tasks, systematically analyzing its complexity and diverse manifestations. We conduct a comprehensive review of data shift detection methods and employ shift detection analysis techniques to perform dataset categorization and balancing. Building upon this foundation, we construct an object detection model. To validate our approach, we optimize the model by integrating CycleGAN-based data augmentation techniques with the YOLOv5 framework. Experimental results demonstrate that our method achieves superior performance compared to baseline models on the BDD100K dataset. 

**Abstract (ZH)**: 随着机器学习技术在自动驾驶系统中的广泛应用，其在应对复杂环境感知挑战中的作用变得日益重要。然而，现有的机器学习模型表现出明显的脆弱性，因为它们的性能高度依赖于训练和测试数据独立同分布的基本假设，在实际应用中难以保证。由季节变化和天气波动引起的数据分布动态变化导致了自动驾驶系统中的数据偏移问题。本研究探讨了自动驾驶目标检测任务中的数据偏移问题，系统地分析了其复杂性和多种表现形式。我们全面回顾了数据偏移检测方法，并运用偏移检测分析技术对数据集进行分类和平衡。在此基础上，我们构建了一个目标检测模型。为了验证我们的方法，我们通过将CycleGAN基于的数据增强技术与YOLOv5框架结合，优化了该模型。实验结果表明，我们的方法在BDD100K数据集上的性能优于基准模型。 

---
# Precise Action-to-Video Generation Through Visual Action Prompts 

**Title (ZH)**: 通过视觉动作提示实现精确的动作到视频生成 

**Authors**: Yuang Wang, Chao Wen, Haoyu Guo, Sida Peng, Minghan Qin, Hujun Bao, Xiaowei Zhou, Ruizhen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13104)  

**Abstract**: We present visual action prompts, a unified action representation for action-to-video generation of complex high-DoF interactions while maintaining transferable visual dynamics across domains. Action-driven video generation faces a precision-generality trade-off: existing methods using text, primitive actions, or coarse masks offer generality but lack precision, while agent-centric action signals provide precision at the cost of cross-domain transferability. To balance action precision and dynamic transferability, we propose to "render" actions into precise visual prompts as domain-agnostic representations that preserve both geometric precision and cross-domain adaptability for complex actions; specifically, we choose visual skeletons for their generality and accessibility. We propose robust pipelines to construct skeletons from two interaction-rich data sources - human-object interactions (HOI) and dexterous robotic manipulation - enabling cross-domain training of action-driven generative models. By integrating visual skeletons into pretrained video generation models via lightweight fine-tuning, we enable precise action control of complex interaction while preserving the learning of cross-domain dynamics. Experiments on EgoVid, RT-1 and DROID demonstrate the effectiveness of our proposed approach. Project page: this https URL. 

**Abstract (ZH)**: 我们提出了视觉动作提示，这是一种统一的动作表示，用于生成复杂高自由度交互的动作到视频转换，同时保持跨域的可转移视觉动力学。动作驱动的视频生成面临着精度与通用性的权衡：现有方法使用文本、原始动作或粗糙掩码虽然具有通用性但缺乏精度，而以代理为中心的动作信号则以牺牲跨域可转移性为代价提供了精度。为了平衡动作精度与动态可转移性，我们提出将动作“渲染”为精确的视觉提示，作为域无关的表示，既保存几何精度又保持跨域适应性；具体而言，我们选择了通用且易访问的视觉骨架。我们提出了健壮的工作流，从两种富含交互的数据源构建骨架——人-物交互（HOI）和灵巧的机器人操作，从而实现动作驱动生成模型的跨域训练。通过将视觉骨架轻量级微调到预训练的视频生成模型中，我们能够在保持跨域动力学学习的同时实现对复杂交互的精确动作控制。在EgoVid、RT-1和DROID上的实验结果表明了我们提出方法的有效性。项目页面：this https URL。 

---
# DynamicPose: Real-time and Robust 6D Object Pose Tracking for Fast-Moving Cameras and Objects 

**Title (ZH)**: DynamicPose：快速移动相机和物体的实时 robust 6D对象姿态跟踪 

**Authors**: Tingbang Liang, Yixin Zeng, Jiatong Xie, Boyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.11950)  

**Abstract**: We present DynamicPose, a retraining-free 6D pose tracking framework that improves tracking robustness in fast-moving camera and object scenarios. Previous work is mainly applicable to static or quasi-static scenes, and its performance significantly deteriorates when both the object and the camera move rapidly. To overcome these challenges, we propose three synergistic components: (1) A visual-inertial odometry compensates for the shift in the Region of Interest (ROI) caused by camera motion; (2) A depth-informed 2D tracker corrects ROI deviations caused by large object translation; (3) A VIO-guided Kalman filter predicts object rotation, generates multiple candidate poses, and then obtains the final pose by hierarchical refinement. The 6D pose tracking results guide subsequent 2D tracking and Kalman filter updates, forming a closed-loop system that ensures accurate pose initialization and precise pose tracking. Simulation and real-world experiments demonstrate the effectiveness of our method, achieving real-time and robust 6D pose tracking for fast-moving cameras and objects. 

**Abstract (ZH)**: DynamicPose：一种无需重新训练的6D姿态跟踪框架，适用于快速移动的相机和物体场景 

---
# From Transthoracic to Transesophageal: Cross-Modality Generation using LoRA Diffusion 

**Title (ZH)**: 从胸壁到食道：基于LoRA扩散模型的跨模态生成 

**Authors**: Emmanuel Oladokun, Yuxuan Ou, Anna Novikova, Daria Kulikova, Sarina Thomas, Jurica Šprem, Vicente Grau  

**Link**: [PDF](https://arxiv.org/pdf/2508.13077)  

**Abstract**: Deep diffusion models excel at realistic image synthesis but demand large training sets-an obstacle in data-scarce domains like transesophageal echocardiography (TEE). While synthetic augmentation has boosted performance in transthoracic echo (TTE), TEE remains critically underrepresented, limiting the reach of deep learning in this high-impact modality.
We address this gap by adapting a TTE-trained, mask-conditioned diffusion backbone to TEE with only a limited number of new cases and adapters as small as $10^5$ parameters. Our pipeline combines Low-Rank Adaptation with MaskR$^2$, a lightweight remapping layer that aligns novel mask formats with the pretrained model's conditioning channels. This design lets users adapt models to new datasets with a different set of anatomical structures to the base model's original set.
Through a targeted adaptation strategy, we find that adapting only MLP layers suffices for high-fidelity TEE synthesis. Finally, mixing less than 200 real TEE frames with our synthetic echoes improves the dice score on a multiclass segmentation task, particularly boosting performance on underrepresented right-heart structures. Our results demonstrate that (1) semantically controlled TEE images can be generated with low overhead, (2) MaskR$^2$ effectively transforms unseen mask formats into compatible formats without damaging downstream task performance, and (3) our method generates images that are effective for improving performance on a downstream task of multiclass segmentation. 

**Abstract (ZH)**: 深度扩散模型在图像合成方面表现出色，但需要庞大的训练集——这在超声心动图食道部分（TEE）这样的数据稀缺领域是一大障碍。虽然合成增强在经胸超声（TTE）中提升了性能，TEE仍严重不足，限制了深度学习在这一高影响成像模式中的应用范围。

我们通过仅使用少量的新案例和仅有 $10^5$ 参数的适配器，将TTE训练的掩码条件扩散主干模型应用到TEE，来填补这一差距。我们的流程结合了低秩适配和MaskR$^2$，这是一种轻量级的重映射层，能够将新的掩码格式与预训练模型的条件通道对齐。这种设计允许用户用不同的解剖结构集将模型适应到新的数据集中。

通过一种有针对性的适配策略，我们发现仅适配MLP层就足以实现高保真TEE合成。最后，将不到200个真实的TEE帧与我们的合成回声混合，可以提高多类分割任务的Dice分数，尤其是在提升未充分代表的右心结构的性能方面尤为明显。我们的结果表明：（1）通过低开销可以生成具有语义控制的TEE图像；（2）MaskR$^2$ 能够有效地将未见过的掩码格式转换为兼容格式，而不损害下游任务的性能；（3）我们的方法生成的图像对下游任务的多类分割性能提升有效。 

---
# Multi-Phase Automated Segmentation of Dental Structures in CBCT Using a Lightweight Auto3DSeg and SegResNet Implementation 

**Title (ZH)**: 基于 Lightweight Auto3DSeg 和 SegResNet 的CBCT牙结构多阶段自动化分割实现 

**Authors**: Dominic LaBella, Keshav Jha, Jared Robbins, Esther Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12962)  

**Abstract**: Cone-beam computed tomography (CBCT) has become an invaluable imaging modality in dentistry, enabling 3D visualization of teeth and surrounding structures for diagnosis and treatment planning. Automated segmentation of dental structures in CBCT can efficiently assist in identifying pathology (e.g., pulpal or periapical lesions) and facilitate radiation therapy planning in head and neck cancer patients. We describe the DLaBella29 team's approach for the MICCAI 2025 ToothFairy3 Challenge, which involves a deep learning pipeline for multi-class tooth segmentation. We utilized the MONAI Auto3DSeg framework with a 3D SegResNet architecture, trained on a subset of the ToothFairy3 dataset (63 CBCT scans) with 5-fold cross-validation. Key preprocessing steps included image resampling to 0.6 mm isotropic resolution and intensity clipping. We applied an ensemble fusion using Multi-Label STAPLE on the 5-fold predictions to infer a Phase 1 segmentation and then conducted tight cropping around the easily segmented Phase 1 mandible to perform Phase 2 segmentation on the smaller nerve structures. Our method achieved an average Dice of 0.87 on the ToothFairy3 challenge out-of-sample validation set. This paper details the clinical context, data preparation, model development, results of our approach, and discusses the relevance of automated dental segmentation for improving patient care in radiation oncology. 

**Abstract (ZH)**: 锥束计算机断层成像（CBCT）已成为牙科中不可或缺的成像技术，能够实现牙齿及其周围结构的3D可视化，用于诊断和治疗计划。在头颈部癌症患者中，CBCT中牙齿结构的自动分割可以有效辅助识别病理（如牙髓或根尖周病变）并促进放射治疗计划。我们描述了DLaBella29团队参加MICCAI 2025 ToothFairy3挑战赛的方法，涉及一种用于多类牙齿分割的深度学习管道。我们使用MONAI Auto3DSeg框架结合3D SegResNet架构，在ToothFairy3数据集的子集（63例CBCT扫描）上进行了5折交叉验证训练。关键技术预处理步骤包括将图像重采样为0.6 mm等向性分辨率并进行强度剪裁。我们采用Multi-Label STAPLE对5折预测进行集成融合以推断第一阶段分割，然后在第一阶段容易分割的下颌周围进行精确裁剪，以在较小的神经结构上执行第二阶段分割。我们的方法在ToothFairy3挑战赛的外部验证集上实现了平均Dice值为0.87。本文详细介绍了临床背景、数据准备、模型开发、方法效果和自动化牙齿分割在放射肿瘤学中改善患者护理方面的相关性。 

---
# CTFlow: Video-Inspired Latent Flow Matching for 3D CT Synthesis 

**Title (ZH)**: CTFlow: 由视频启发的潜空间流匹配三维CT合成 

**Authors**: Jiayi Wang, Hadrien Reynaud, Franciskus Xaverius Erick, Bernhard Kainz  

**Link**: [PDF](https://arxiv.org/pdf/2508.12900)  

**Abstract**: Generative modelling of entire CT volumes conditioned on clinical reports has the potential to accelerate research through data augmentation, privacy-preserving synthesis and reducing regulator-constraints on patient data while preserving diagnostic signals. With the recent release of CT-RATE, a large-scale collection of 3D CT volumes paired with their respective clinical reports, training large text-conditioned CT volume generation models has become achievable. In this work, we introduce CTFlow, a 0.5B latent flow matching transformer model, conditioned on clinical reports. We leverage the A-VAE from FLUX to define our latent space, and rely on the CT-Clip text encoder to encode the clinical reports. To generate consistent whole CT volumes while keeping the memory constraints tractable, we rely on a custom autoregressive approach, where the model predicts the first sequence of slices of the volume from text-only, and then relies on the previously generated sequence of slices and the text, to predict the following sequence. We evaluate our results against state-of-the-art generative CT model, and demonstrate the superiority of our approach in terms of temporal coherence, image diversity and text-image alignment, with FID, FVD, IS scores and CLIP score. 

**Abstract (ZH)**: 基于临床报告条件下的CT整卷生成模型有望通过数据增强、隐私保护合成以及减少患者数据监管约束来加速研究，同时保留诊断信号。随着CT-RATE的推出，一个大规模的3D CT整卷及其相应的临床报告集合，训练大型文本条件下的CT整卷生成模型变得可行。在此工作中，我们介绍CTFlow，一个基于临床报告条件下的0.5B潜空间流动匹配转换器模型。我们利用FLUX的A-VAE来定义潜空间，并依赖CT-Clip文本编码器对临床报告进行编码。为了生成一致的完整CT整卷并保持内存约束可处理，我们依赖一种自回归方法，其中模型首先仅从文本中预测体积的第一个序列切片，然后依赖之前生成的序列切片和文本来预测后续序列。我们将我们的结果与最先进的生成CT模型进行评估，并在时空连贯性、图像多样性及文本-图像对齐方面展示了我们方法的优越性，通过FID、FVD、IS评分和CLIP评分进行验证。 

---
# Next Visual Granularity Generation 

**Title (ZH)**: 下一级视觉粒度生成 

**Authors**: Yikai Wang, Zhouxia Wang, Zhonghua Wu, Qingyi Tao, Kang Liao, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2508.12811)  

**Abstract**: We propose a novel approach to image generation by decomposing an image into a structured sequence, where each element in the sequence shares the same spatial resolution but differs in the number of unique tokens used, capturing different level of visual granularity. Image generation is carried out through our newly introduced Next Visual Granularity (NVG) generation framework, which generates a visual granularity sequence beginning from an empty image and progressively refines it, from global layout to fine details, in a structured manner. This iterative process encodes a hierarchical, layered representation that offers fine-grained control over the generation process across multiple granularity levels. We train a series of NVG models for class-conditional image generation on the ImageNet dataset and observe clear scaling behavior. Compared to the VAR series, NVG consistently outperforms it in terms of FID scores (3.30 -> 3.03, 2.57 ->2.44, 2.09 -> 2.06). We also conduct extensive analysis to showcase the capability and potential of the NVG framework. Our code and models will be released. 

**Abstract (ZH)**: 我们提出了一种新颖的图像生成方法，通过将图像分解为结构化序列，其中序列中的每个元素具有相同的空间分辨率但使用独特标记的数量不同，从而捕捉不同级别的视觉粒度。图像生成通过我们新引入的Next Visual Granularity（NVG）生成框架进行，该框架从空白图像开始生成视觉粒度序列，并以结构化的方式逐步细化，从全局布局到细项。这一迭代过程编码了一个分层表示，提供了对多粒度级别生成过程的细粒度控制。我们在ImageNet数据集上训练了一系列NVG模型进行类别条件图像生成，并观察到明显的缩放行为。与VAR系列相比，NVG在FID分数上始终表现更优（3.30 -> 3.03，2.57 -> 2.44，2.09 -> 2.06）。我们还进行了广泛的分析以展示NVG框架的能力和潜力。我们的代码和模型将开源发布。 

---
# Vehicle detection from GSV imagery: Predicting travel behaviour for cycling and motorcycling using Computer Vision 

**Title (ZH)**: 基于街景图像的车辆检测：利用计算机视觉预测自行车和摩托车出行行为 

**Authors**: Kyriaki, Kokka, Rahul Goel, Ali Abbas, Kerry A. Nice, Luca Martial, SM Labib, Rihuan Ke, Carola Bibiane Schönlieb, James Woodcock  

**Link**: [PDF](https://arxiv.org/pdf/2508.12794)  

**Abstract**: Transportation influence health by shaping exposure to physical activity, air pollution and injury this http URL data on cycling and motorcycling behaviours is scarce, particularly at a global this http URL view imagery, such as Google Street View (GSV), combined with computer vision, is a valuable resource for efficiently capturing travel behaviour this http URL study demonstrates a novel approach using deep learning on street view images to estimate cycling and motorcycling levels across diverse cities this http URL utilized data from 185 global this http URL data on mode shares of cycling and motorcycling estimated using travel surveys or this http URL used GSV images to detect cycles and motorcycles in sampled locations, using 8000 images per this http URL YOLOv4 model, fine-tuned using images from six cities, achieved a mean average precision of 89% for detecting cycles and motorcycles in GSV images.A global prediction model was developed using beta regression with city-level mode shares as outcome, with log transformed explanatory variables of counts of GSV-detected images with cycles and motorcycles, while controlling for population this http URL found strong correlations between GSV motorcycle counts and motorcycle mode share (0.78) and moderate correlations between GSV cycle counts and cycling mode share (0.51).Beta regression models predicted mode shares with $R^2$ values of 0.614 for cycling and 0.612 for motorcycling, achieving median absolute errors (MDAE) of 1.3% and 1.4%, this http URL demonstrated consistent prediction accuracy, though cities like Utrecht and Cali were this http URL model was applied to 60 cities globally for which we didn't have recent mode share this http URL provided estimates for some cities in the Middle East, Latin America and East this http URL computer vision, GSV images capture travel modes and activity, providing insights alongside traditional data sources. 

**Abstract (ZH)**: 交通通过塑造体力活动、空气污染暴露和伤害影响健康，特别是基于全球视角的骑行和摩托车行为数据稀缺，通过将遥感影像与计算机视觉结合，可以有效捕捉出行行为，本研究采用深度学习方法分析遥感影像，以估算不同城市的骑行和摩托车出行水平，利用来自185个全球城市的数据，通过计算机视觉检测遥感影像中的自行车和摩托车，在8000张遥感影像上使用YOLOv4模型，经过六个城市影像的微调，检测遥感影像中自行车和摩托车的平均精度达到89%。开发了一个使用贝塔回归模型的全球预测模型，城市级别的出行比例作为结果变量，通过转换后的遥感影像中检测到的自行车和摩托车数量的计数解释变量来控制人口因素，结果发现遥感影像中的摩托车数量与摩托车出行比例之间存在很强的相关性（0.78），而遥感影像中的自行车数量与自行车出行比例之间存在中等的相关性（0.51）。贝塔回归模型预测出行比例，自行车出行比例的$R^2$值为0.614，摩托车出行比例的$R^2$值为0.612，中位绝对误差分别为1.3%和1.4%，展示了模型具有稳定的预测准确性，尽管部分地区如乌得勒支和卡利的预测效果不佳。该模型应用于60个缺乏近期出行比例数据的城市，为中东、拉丁美洲和东亚的一些城市提供了出行比例的估计。遥感影像和计算机视觉捕捉出行模式和活动，为传统数据源提供了补充见解。 

---
# Harnessing Group-Oriented Consistency Constraints for Semi-Supervised Semantic Segmentation in CdZnTe Semiconductors 

**Title (ZH)**: 基于群导向一致约束的半监督语义分割在CdZnTe半导体中的应用 

**Authors**: Peihao Li, Yan Fang, Man Liu, Huihui Bai, Anhong Wang, Yunchao Wei, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12766)  

**Abstract**: Labeling Cadmium Zinc Telluride (CdZnTe) semiconductor images is challenging due to the low-contrast defect boundaries, necessitating annotators to cross-reference multiple views. These views share a single ground truth (GT), forming a unique ``many-to-one'' relationship. This characteristic renders advanced semi-supervised semantic segmentation (SSS) methods suboptimal, as they are generally limited by a ``one-to-one'' relationship, where each image is independently associated with its GT. Such limitation may lead to error accumulation in low-contrast regions, further exacerbating confirmation bias. To address this issue, we revisit the SSS pipeline from a group-oriented perspective and propose a human-inspired solution: the Intra-group Consistency Augmentation Framework (ICAF). First, we experimentally validate the inherent consistency constraints within CdZnTe groups, establishing a group-oriented baseline using the Intra-group View Sampling (IVS). Building on this insight, we introduce the Pseudo-label Correction Network (PCN) to enhance consistency representation, which consists of two key modules. The View Augmentation Module (VAM) improves boundary details by dynamically synthesizing a boundary-aware view through the aggregation of multiple views. In the View Correction Module (VCM), this synthesized view is paired with other views for information interaction, effectively emphasizing salient regions while minimizing noise. Extensive experiments demonstrate the effectiveness of our solution for CdZnTe materials. Leveraging DeepLabV3+ with a ResNet-101 backbone as our segmentation model, we achieve a 70.6\% mIoU on the CdZnTe dataset using only 2 group-annotated data (5\textperthousand). The code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于CdZnTe半导体图像的标签标注因其低对比度缺陷边界具有挑战性，需要标注人员参考多个视图。这些视图共享一个单一的真实标注（GT），形成了独特的“多对一”关系。这一特性使得先进的半监督语义分割（SSS）方法不够理想，因为它们通常受限于“一对一”关系，即每张图像独立关联一个GT。这种限制可能导致低对比度区域的错误积累，进一步加剧确认偏差。为解决这一问题，我们从群体导向的视角回顾了SSS管道，并提出了一种受人类启发的解决方案：Intra-group一致性增强框架（ICAF）。首先，我们通过Intra-group视图采样（IVS）实验验证了CdZnTe群体内的固有一致性约束，建立了群体导向的基础模型。在此基础上，我们引入了伪标签校准网络（PCN）来增强一致性表示，该网络由两个关键模块组成。视图增强模块（VAM）通过多个视图的聚合动态合成边界感知视图，以改进边界细节。在视图校准模块（VCM）中，该合成视图与其他视图配对进行信息交互，有效强调显著区域同时减少噪声。广泛的实验表明，我们的解决方案对CdZnTe材料的有效性。利用具有ResNet-101骨干网络的DeepLabV3+作为分割模型，在仅使用2组标注数据（0.5‰）的情况下，实现了CdZnTe数据集上的70.6% mIoU。代码可在\href{this https URL}{this https URL}获取。 

---
# CLAIRE-DSA: Fluoroscopic Image Classification for Quality Assurance of Computer Vision Pipelines in Acute Ischemic Stroke 

**Title (ZH)**: CLAIRE-DSA：急性缺血性中风计算机视觉管道质量保证的荧光透视图像分类 

**Authors**: Cristo J. van den Berg, Frank G. te Nijenhuis, Mirre J. Blaauboer, Daan T. W. van Erp, Carlijn M. Keppels, Matthijs van der Sluijs, Bob Roozenbeek, Wim van Zwam, Sandra Cornelissen, Danny Ruijters, Ruisheng Su, Theo van Walsum  

**Link**: [PDF](https://arxiv.org/pdf/2508.12755)  

**Abstract**: Computer vision models can be used to assist during mechanical thrombectomy (MT) for acute ischemic stroke (AIS), but poor image quality often degrades performance. This work presents CLAIRE-DSA, a deep learning--based framework designed to categorize key image properties in minimum intensity projections (MinIPs) acquired during MT for AIS, supporting downstream quality control and workflow optimization. CLAIRE-DSA uses pre-trained ResNet backbone models, fine-tuned to predict nine image properties (e.g., presence of contrast, projection angle, motion artefact severity). Separate classifiers were trained on an annotated dataset containing $1,758$ fluoroscopic MinIPs. The model achieved excellent performance on all labels, with ROC-AUC ranging from $0.91$ to $0.98$, and precision ranging from $0.70$ to $1.00$. The ability of CLAIRE-DSA to identify suitable images was evaluated on a segmentation task by filtering poor quality images and comparing segmentation performance on filtered and unfiltered datasets. Segmentation success rate increased from $42%$ to $69%$, $p < 0.001$. CLAIRE-DSA demonstrates strong potential as an automated tool for accurately classifying image properties in DSA series of acute ischemic stroke patients, supporting image annotation and quality control in clinical and research applications. Source code is available at this https URL. 

**Abstract (ZH)**: 基于深度学习的CLAIRE-DSA框架用于急性缺血性中风机械取栓过程中最小强度投影图像的关键图像属性分类 

---
# DCSCR: A Class-Specific Collaborative Representation based Network for Image Set Classification 

**Title (ZH)**: DCSCR：一种用于图像集分类的类特定协作表示表示网络 fod图像集分类yd 

**Authors**: Xizhan Gao, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12745)  

**Abstract**: Image set classification (ISC), which can be viewed as a task of comparing similarities between sets consisting of unordered heterogeneous images with variable quantities and qualities, has attracted growing research attention in recent years. How to learn effective feature representations and how to explore the similarities between different image sets are two key yet challenging issues in this field. However, existing traditional ISC methods classify image sets based on raw pixel features, ignoring the importance of feature learning. Existing deep ISC methods can learn deep features, but they fail to adaptively adjust the features when measuring set distances, resulting in limited performance in few-shot ISC. To address the above issues, this paper combines traditional ISC methods with deep models and proposes a novel few-shot ISC approach called Deep Class-specific Collaborative Representation (DCSCR) network to simultaneously learn the frame- and concept-level feature representations of each image set and the distance similarities between different sets. Specifically, DCSCR consists of a fully convolutional deep feature extractor module, a global feature learning module, and a class-specific collaborative representation-based metric learning module. The deep feature extractor and global feature learning modules are used to learn (local and global) frame-level feature representations, while the class-specific collaborative representation-based metric learning module is exploit to adaptively learn the concept-level feature representation of each image set and thus obtain the distance similarities between different sets by developing a new CSCR-based contrastive loss function. Extensive experiments on several well-known few-shot ISC datasets demonstrate the effectiveness of the proposed method compared with some state-of-the-art image set classification algorithms. 

**Abstract (ZH)**: 基于深度模型的Frame-和Concept-level协同表示的少样本图像集合分类（Deep Class-specific Collaborative Representation Network for Few-shot Image Set Classification） 

---
# OpenMoCap: Rethinking Optical Motion Capture under Real-world Occlusion 

**Title (ZH)**: OpenMoCap: 重新思考.real-world Occlusion下的光学运动捕捉 

**Authors**: Chen Qian, Danyang Li, Xinran Yu, Zheng Yang, Qiang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.12610)  

**Abstract**: Optical motion capture is a foundational technology driving advancements in cutting-edge fields such as virtual reality and film production. However, system performance suffers severely under large-scale marker occlusions common in real-world applications. An in-depth analysis identifies two primary limitations of current models: (i) the lack of training datasets accurately reflecting realistic marker occlusion patterns, and (ii) the absence of training strategies designed to capture long-range dependencies among markers. To tackle these challenges, we introduce the CMU-Occlu dataset, which incorporates ray tracing techniques to realistically simulate practical marker occlusion patterns. Furthermore, we propose OpenMoCap, a novel motion-solving model designed specifically for robust motion capture in environments with significant occlusions. Leveraging a marker-joint chain inference mechanism, OpenMoCap enables simultaneous optimization and construction of deep constraints between markers and joints. Extensive comparative experiments demonstrate that OpenMoCap consistently outperforms competing methods across diverse scenarios, while the CMU-Occlu dataset opens the door for future studies in robust motion solving. The proposed OpenMoCap is integrated into the MoSen MoCap system for practical deployment. The code is released at: this https URL. 

**Abstract (ZH)**: 光学运动捕捉是推动虚拟现实和电影制作等前沿领域发展的基础技术，但在实际应用中大规模标记物遮挡会严重影响系统性能。深入分析识别了当前模型的两大主要局限性：（i）缺乏能够准确反映真实标记物遮挡模式的训练数据集，（ii）缺少用于捕捉标记物之间长程依赖性的训练策略。为应对这些挑战，我们引入了CMU-Occlu数据集，利用射线 tracing 技术真实模拟实际的标记物遮挡模式。此外，我们提出了OpenMoCap，这是一种专为显著遮挡环境中稳健运动捕捉设计的新模型。利用标记物-关节链推理机制，OpenMoCap能够同时优化并构建标记和关节间的深层约束。广泛比较实验表明，OpenMoCap在多种场景中表现 superior 于现有方法，而CMU-Occlu数据集则为未来稳健运动求解研究打开了大门。提出的OpenMoCap已被集成到MoSen MoCap系统中用于实际部署。代码在此处发布：this https URL。 

---
# An Initial Study of Bird's-Eye View Generation for Autonomous Vehicles using Cross-View Transformers 

**Title (ZH)**: 基于交叉视图变换器的鸟瞰视图生成在自主车辆中的初步研究 

**Authors**: Felipe Carlos dos Santos, Eric Aislan Antonelo, Gustavo Claudio Karl Couto  

**Link**: [PDF](https://arxiv.org/pdf/2508.12520)  

**Abstract**: Bird's-Eye View (BEV) maps provide a structured, top-down abstraction that is crucial for autonomous-driving perception. In this work, we employ Cross-View Transformers (CVT) for learning to map camera images to three BEV's channels - road, lane markings, and planned trajectory - using a realistic simulator for urban driving. Our study examines generalization to unseen towns, the effect of different camera layouts, and two loss formulations (focal and L1). Using training data from only a town, a four-camera CVT trained with the L1 loss delivers the most robust test performance, evaluated in a new town. Overall, our results underscore CVT's promise for mapping camera inputs to reasonably accurate BEV maps. 

**Abstract (ZH)**: Bird's-Eye View Maps from Cross-View Transformers for Autonomous Driving Perception in Urban Scenarios 

---
# Adversarial Attacks on VQA-NLE: Exposing and Alleviating Inconsistencies in Visual Question Answering Explanations 

**Title (ZH)**: 针对VQA-NLE的对抗攻击：揭露并缓解视觉问答解释中的不一致性 

**Authors**: Yahsin Yeh, Yilun Wu, Bokai Ruan, Honghan Shuai  

**Link**: [PDF](https://arxiv.org/pdf/2508.12430)  

**Abstract**: Natural language explanations in visual question answering (VQA-NLE) aim to make black-box models more transparent by elucidating their decision-making processes. However, we find that existing VQA-NLE systems can produce inconsistent explanations and reach conclusions without genuinely understanding the underlying context, exposing weaknesses in either their inference pipeline or explanation-generation mechanism. To highlight these vulnerabilities, we not only leverage an existing adversarial strategy to perturb questions but also propose a novel strategy that minimally alters images to induce contradictory or spurious outputs. We further introduce a mitigation method that leverages external knowledge to alleviate these inconsistencies, thereby bolstering model robustness. Extensive evaluations on two standard benchmarks and two widely used VQA-NLE models underscore the effectiveness of our attacks and the potential of knowledge-based defenses, ultimately revealing pressing security and reliability concerns in current VQA-NLE systems. 

**Abstract (ZH)**: 自然语言解释在视觉问答（VQA）中的应用旨在通过阐明其决策过程使黑盒模型更加透明。然而，我们发现现有VQA-NLE系统会产生不一致的解释，并在未真正理解底层上下文的情况下得出结论，暴露了它们推理管道或解释生成机制中的弱点。为了突出这些弱点，我们不仅利用现有的对抗策略来扰动问题，还提出了一种新的策略，通过最小改变图像来诱导矛盾或虚假的输出。我们进一步引入了一种利用外部知识的方法来缓解这些不一致性，从而增强模型的稳健性。广泛的标准基准和广泛使用的VQA-NLE模型的评估证实了我们攻击的有效性和基于知识的防护潜力，最终揭示了当前VQA-NLE系统中存在的迫切的安全性和可靠性问题。 

---
# SRMA-Mamba: Spatial Reverse Mamba Attention Network for Pathological Liver Segmentation in MRI Volumes 

**Title (ZH)**: SRMA-Mamba：空间逆Mamba注意力网络在MRI体积中用于病理肝脏分割 

**Authors**: Jun Zeng, Yannan Huang, Elif Keles, Halil Ertugrul Aktas, Gorkem Durak, Nikhil Kumar Tomar, Quoc-Huy Trinh, Deepak Ranjan Nayak, Ulas Bagci, Debesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2508.12410)  

**Abstract**: Liver Cirrhosis plays a critical role in the prognosis of chronic liver disease. Early detection and timely intervention are critical in significantly reducing mortality rates. However, the intricate anatomical architecture and diverse pathological changes of liver tissue complicate the accurate detection and characterization of lesions in clinical settings. Existing methods underutilize the spatial anatomical details in volumetric MRI data, thereby hindering their clinical effectiveness and explainability. To address this challenge, we introduce a novel Mamba-based network, SRMA-Mamba, designed to model the spatial relationships within the complex anatomical structures of MRI volumes. By integrating the Spatial Anatomy-Based Mamba module (SABMamba), SRMA-Mamba performs selective Mamba scans within liver cirrhotic tissues and combines anatomical information from the sagittal, coronal, and axial planes to construct a global spatial context representation, enabling efficient volumetric segmentation of pathological liver structures. Furthermore, we introduce the Spatial Reverse Attention module (SRMA), designed to progressively refine cirrhotic details in the segmentation map, utilizing both the coarse segmentation map and hierarchical encoding features. Extensive experiments demonstrate that SRMA-Mamba surpasses state-of-the-art methods, delivering exceptional performance in 3D pathological liver segmentation. Our code is available for public: {\color{blue}{this https URL}}. 

**Abstract (ZH)**: 肝脏硬化在慢性肝病的预后中起着关键作用。早期检测和及时干预对于显著降低 mortality 率至关重要。然而，肝脏组织复杂的解剖结构和多种病理变化在临床检测和病变表征中增加了复杂性。现有方法在利用容积 MRI 数据的空间解剖细节方面存在不足，从而限制了其临床效果和可解释性。为了应对这一挑战，我们引入了一种基于 Mamba 的新型网络 SRMA-Mamba，用于建模 MRI 体积中复杂解剖结构内的空间关系。通过整合基于空间解剖学的 Mamba 模块（SABMamba），SRMA-Mamba 在肝脏硬化组织中执行选择性的 Mamba 扫描，并结合矢状面、冠状面和轴向面的解剖信息来构建全局空间上下文表示，从而实现病理肝脏结构的高效容积分割。此外，我们还引入了空间反向注意力模块（SRMA），该模块通过利用粗略分割图和层次编码特征来逐步细化分割图中的硬化细节。广泛实验表明，SRMA-Mamba 超过了最先进的方法，在 3D 病理肝脏分割任务中表现出色。我们的代码已公开：[this https URL]。 

---
# Semantic Discrepancy-aware Detector for Image Forgery Identification 

**Title (ZH)**: 语义不一致性感知检测器用于图像伪造识别 

**Authors**: Ziye Wang, Minghang Yu, Chunyan Xu, Zhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2508.12341)  

**Abstract**: With the rapid advancement of image generation techniques, robust forgery detection has become increasingly imperative to ensure the trustworthiness of digital media. Recent research indicates that the learned semantic concepts of pre-trained models are critical for identifying fake images. However, the misalignment between the forgery and semantic concept spaces hinders the model's forgery detection performance. To address this problem, we propose a novel Semantic Discrepancy-aware Detector (SDD) that leverages reconstruction learning to align the two spaces at a fine-grained visual level. By exploiting the conceptual knowledge embedded in the pre-trained vision language model, we specifically design a semantic token sampling module to mitigate the space shifts caused by features irrelevant to both forgery traces and semantic concepts. A concept-level forgery discrepancy learning module, built upon a visual reconstruction paradigm, is proposed to strengthen the interaction between visual semantic concepts and forgery traces, effectively capturing discrepancies under the concepts' guidance. Finally, the low-level forgery feature enhancemer integrates the learned concept level forgery discrepancies to minimize redundant forgery information. Experiments conducted on two standard image forgery datasets demonstrate the efficacy of the proposed SDD, which achieves superior results compared to existing methods. The code is available at this https URL. 

**Abstract (ZH)**: 随着图像生成技术的迅速发展，稳健的伪造检测已成为确保数字媒体可信性的日益重要任务。近期研究表明，预训练模型学习到的语义概念对于识别伪造图像至关重要。然而，伪造和语义概念空间之间的不一致阻碍了模型的伪造检测性能。为了解决这一问题，我们提出了一种新型的语义差异感知检测器（SDD），利用重建学习在细微视觉层面上对齐两个空间。通过利用预训练的视觉语言模型中嵌入的概念知识，我们特别设计了一个语义令牌采样模块，以缓解由与伪造痕迹和语义概念无关的特征引起的空间偏移。基于视觉重建范式的概念级伪造差异学习模块被提出，以加强视觉语义概念与伪造痕迹之间的交互，在概念的引导下有效捕捉差异。最后，低级别伪造特征增强模块整合了学习到的概念级伪造差异，以最小化冗余的伪造信息。实验结果显示，所提出的SDD在两个标准图像伪造数据集上具有优越性，其性能优于现有方法。代码可在此处获得。 

---
# TSLA: A Task-Specific Learning Adaptation for Semantic Segmentation on Autonomous Vehicles Platform 

**Title (ZH)**: TSLA：面向自主驾驶平台语义分割的任务特定学习适应 

**Authors**: Jun Liu, Zhenglun Kong, Pu Zhao, Weihao Zeng, Hao Tang, Xuan Shen, Changdi Yang, Wenbin Zhang, Geng Yuan, Wei Niu, Xue Lin, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12279)  

**Abstract**: Autonomous driving platforms encounter diverse driving scenarios, each with varying hardware resources and precision requirements. Given the computational limitations of embedded devices, it is crucial to consider computing costs when deploying on target platforms like the NVIDIA\textsuperscript{\textregistered} DRIVE PX 2. Our objective is to customize the semantic segmentation network according to the computing power and specific scenarios of autonomous driving hardware. We implement dynamic adaptability through a three-tier control mechanism -- width multiplier, classifier depth, and classifier kernel -- allowing fine-grained control over model components based on hardware constraints and task requirements. This adaptability facilitates broad model scaling, targeted refinement of the final layers, and scenario-specific optimization of kernel sizes, leading to improved resource allocation and performance.
Additionally, we leverage Bayesian Optimization with surrogate modeling to efficiently explore hyperparameter spaces under tight computational budgets. Our approach addresses scenario-specific and task-specific requirements through automatic parameter search, accommodating the unique computational complexity and accuracy needs of autonomous driving. It scales its Multiply-Accumulate Operations (MACs) for Task-Specific Learning Adaptation (TSLA), resulting in alternative configurations tailored to diverse self-driving tasks. These TSLA customizations maximize computational capacity and model accuracy, optimizing hardware utilization. 

**Abstract (ZH)**: 自主驾驶平台面临多样的驾驶场景，每个场景的硬件资源和精度要求各不相同。鉴于嵌入式设备的计算限制，在如NVIDIA® DRIVE PX 2等目标平台上部署时，考虑计算成本至关重要。我们的目标是根据自主驾驶硬件的计算能力及其特定场景定制语义分割网络。通过三层控制机制——宽度乘数、分类器深度和分类器核大小，实现细粒度控制，基于硬件约束和任务要求。这种适应性使得能够广泛调整模型规模、精炼最终层，并针对特定场景优化核大小，从而提高资源分配和性能。此外，我们利用贝叶斯优化和代理模型高效探索在紧苛计算预算下的超参数空间。我们的方法通过自动参数搜索解决特定场景和任务的具体需求，适应自主驾驶的独特计算复杂性和精确度需求。该方法针对任务特定学习适应性调整乘法累加操作（MACs），产生适应不同自动驾驶任务的替代配置，从而最大化计算能力和模型精度，优化硬件利用率。 

---
# STM3: Mixture of Multiscale Mamba for Long-Term Spatio-Temporal Time-Series Prediction 

**Title (ZH)**: STM3：多尺度Mamba混合模型的长时空间-时间序列预测 

**Authors**: Haolong Chen, Liang Zhang, Zhengyuan Xin, Guangxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12247)  

**Abstract**: Recently, spatio-temporal time-series prediction has developed rapidly, yet existing deep learning methods struggle with learning complex long-term spatio-temporal dependencies efficiently. The long-term spatio-temporal dependency learning brings two new challenges: 1) The long-term temporal sequence includes multiscale information naturally which is hard to extract efficiently; 2) The multiscale temporal information from different nodes is highly correlated and hard to model. To address these challenges, we propose an efficient \textit{\textbf{S}patio-\textbf{T}emporal \textbf{M}ultiscale \textbf{M}amba} (STM2) that includes a multiscale Mamba architecture to capture the multiscale information efficiently and simultaneously, and an adaptive graph causal convolution network to learn the complex multiscale spatio-temporal dependency. STM2 includes hierarchical information aggregation for different-scale information that guarantees their distinguishability. To capture diverse temporal dynamics across all spatial nodes more efficiently, we further propose an enhanced version termed \textit{\textbf{S}patio-\textbf{T}emporal \textbf{M}ixture of \textbf{M}ultiscale \textbf{M}amba} (STM3) that employs a special Mixture-of-Experts architecture, including a more stable routing strategy and a causal contrastive learning strategy to enhance the scale distinguishability. We prove that STM3 has much better routing smoothness and guarantees the pattern disentanglement for each expert successfully. Extensive experiments on real-world benchmarks demonstrate STM2/STM3's superior performance, achieving state-of-the-art results in long-term spatio-temporal time-series prediction. 

**Abstract (ZH)**: 高效的时空多尺度Mamba（STM2/STM3）方法及其在长期时空时间序列预测中的应用 

---
# Towards Generalizable Human Activity Recognition: A Survey 

**Title (ZH)**: 面向泛化的行人活动识别：一种综述 

**Authors**: Yize Cai, Baoshen Guo, Flora Salim, Zhiqing Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.12213)  

**Abstract**: As a critical component of Wearable AI, IMU-based Human Activity Recognition (HAR) has attracted increasing attention from both academia and industry in recent years. Although HAR performance has improved considerably in specific scenarios, its generalization capability remains a key barrier to widespread real-world adoption. For example, domain shifts caused by variations in users, sensor positions, or environments can significantly decrease the performance in practice. As a result, in this survey, we explore the rapidly evolving field of IMU-based generalizable HAR, reviewing 229 research papers alongside 25 publicly available datasets to provide a broad and insightful overview. We first present the background and overall framework of IMU-based HAR tasks, as well as the generalization-oriented training settings. Then, we categorize representative methodologies from two perspectives: (i) model-centric approaches, including pre-training method, end-to-end method, and large language model (LLM)-based learning method; and (ii) data-centric approaches, including multi-modal learning and data augmentation techniques. In addition, we summarize widely used datasets in this field, as well as relevant tools and benchmarks. Building on these methodological advances, the broad applicability of IMU-based HAR is also reviewed and discussed. Finally, we discuss persistent challenges (e.g., data scarcity, efficient training, and reliable evaluation) and also outline future directions for HAR, including the adoption of foundation and large language models, physics-informed and context-aware reasoning, generative modeling, and resource-efficient training and inference. The complete list of this survey is available at this https URL, which will be updated continuously. 

**Abstract (ZH)**: 基于IMU的人体活动识别：可泛化的挑战与方法综述 

---
# RealTalk: Realistic Emotion-Aware Lifelike Talking-Head Synthesis 

**Title (ZH)**: RealTalk: 真实情感意识的逼真头部合成 

**Authors**: Wenqing Wang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12163)  

**Abstract**: Emotion is a critical component of artificial social intelligence. However, while current methods excel in lip synchronization and image quality, they often fail to generate accurate and controllable emotional expressions while preserving the subject's identity. To address this challenge, we introduce RealTalk, a novel framework for synthesizing emotional talking heads with high emotion accuracy, enhanced emotion controllability, and robust identity preservation. RealTalk employs a variational autoencoder (VAE) to generate 3D facial landmarks from driving audio, which are concatenated with emotion-label embeddings using a ResNet-based landmark deformation model (LDM) to produce emotional landmarks. These landmarks and facial blendshape coefficients jointly condition a novel tri-plane attention Neural Radiance Field (NeRF) to synthesize highly realistic emotional talking heads. Extensive experiments demonstrate that RealTalk outperforms existing methods in emotion accuracy, controllability, and identity preservation, advancing the development of socially intelligent AI systems. 

**Abstract (ZH)**: 情绪是人工社会智能的关键组成部分。然而，虽然现有方法在唇同步和图像质量方面表现出色，但在生成准确且可控的情绪表达并同时保留主体身份方面常常失败。为解决这一挑战，我们提出了RealTalk，这是一个用于合成高情绪准确度、增强情绪可控性和稳健身份保留的情感面部模型的新框架。RealTalk 使用变分自编码器 (VAE) 从驱动音频中生成 3D 面部关键点，然后使用基于 ResNet 的面部关键点变形模型 (LDM) 将情绪标签嵌入与其他关键点连接，生成带有情绪信息的关键点。这些关键点与面部混合形状系数一起条件作用于一种新颖的三平面注意力神经辐射场 (NeRF)，以合成极为逼真的情感面部模型。大量实验表明，RealTalk 在情绪准确度、可控性和身份保留方面优于现有方法，推动了社会智能AI系统的开发。 

---
# Generic Event Boundary Detection via Denoising Diffusion 

**Title (ZH)**: 通用事件边界检测 via 去噪扩散 

**Authors**: Jaejun Hwang, Dayoung Gong, Manjin Kim, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.12084)  

**Abstract**: Generic event boundary detection (GEBD) aims to identify natural boundaries in a video, segmenting it into distinct and meaningful chunks. Despite the inherent subjectivity of event boundaries, previous methods have focused on deterministic predictions, overlooking the diversity of plausible solutions. In this paper, we introduce a novel diffusion-based boundary detection model, dubbed DiffGEBD, that tackles the problem of GEBD from a generative perspective. The proposed model encodes relevant changes across adjacent frames via temporal self-similarity and then iteratively decodes random noise into plausible event boundaries being conditioned on the encoded features. Classifier-free guidance allows the degree of diversity to be controlled in denoising diffusion. In addition, we introduce a new evaluation metric to assess the quality of predictions considering both diversity and fidelity. Experiments show that our method achieves strong performance on two standard benchmarks, Kinetics-GEBD and TAPOS, generating diverse and plausible event boundaries. 

**Abstract (ZH)**: 通用事件边界检测（GEBD）旨在识别视频中的自然边界，将其分割为独立且有意义的片段。尽管事件边界的主观性较强，之前的 方法集中在确定性预测上，忽视了可能解的多样性。本文提出了一种基于扩散的边界检测模型，名为DiffGEBD，该模型从生成的角度解决 GEBD 问题。提出的模型通过时间自相似性编码相邻帧的相关变化，然后逐步将随机噪声解码为以编码特征为条件的可能的事件边界。无分类器引导允许在去噪扩散过程中控制多样性的程度。此外，我们引入了一个新的评估指标，考虑多样性和保真度来评估预测质量。实验结果显示，我们的方法在两个标准基准数据集Kinetics-GEBD和TAPOS上表现出色，生成了多样且合理的事件边界。 

---
# Automated Model Evaluation for Object Detection via Prediction Consistency and Reliablity 

**Title (ZH)**: 基于预测一致性和可靠性的自动化目标检测模型评估 

**Authors**: Seungju Yoo, Hyuk Kwon, Joong-Won Hwang, Kibok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.12082)  

**Abstract**: Recent advances in computer vision have made training object detectors more efficient and effective; however, assessing their performance in real-world applications still relies on costly manual annotation. To address this limitation, we develop an automated model evaluation (AutoEval) framework for object detection. We propose Prediction Consistency and Reliability (PCR), which leverages the multiple candidate bounding boxes that conventional detectors generate before non-maximum suppression (NMS). PCR estimates detection performance without ground-truth labels by jointly measuring 1) the spatial consistency between boxes before and after NMS, and 2) the reliability of the retained boxes via the confidence scores of overlapping boxes. For a more realistic and scalable evaluation, we construct a meta-dataset by applying image corruptions of varying severity. Experimental results demonstrate that PCR yields more accurate performance estimates than existing AutoEval methods, and the proposed meta-dataset covers a wider range of detection performance. The code is available at this https URL. 

**Abstract (ZH)**: 最近计算机视觉的进步使目标检测器的训练更加高效和有效；然而，它们在实际应用中的性能评估仍然依赖于昂贵的手动注释。为了解决这一限制，我们开发了一种目标检测的自动模型评估（AutoEval）框架。我们提出了一种预测一致性与可靠性（PCR）方法，该方法利用了常规检测器在非最大抑制（NMS）之前生成的多个候选边界框。PCR通过共同测量1) NMS前后边界框的空间一致性，以及2) 保留边界框的可靠性的置信分数来估算检测性能，无需 ground-truth 标注。为了实现更加现实和可扩展的评估，我们通过应用不同程度的图像腐化构建了一个元数据集。实验结果表明，PCR比现有的自动评估方法提供了更准确的性能估计，并且提出的元数据集涵盖了更广泛的检测性能范围。代码可在此处访问：这个 https URL。 

---
# Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering 

**Title (ZH)**: Q-FSRU：量子增强频率谱融合在医学视觉问答中的应用 

**Authors**: Rakesh Thakur, Yusra Tariq  

**Link**: [PDF](https://arxiv.org/pdf/2508.12036)  

**Abstract**: Solving tough clinical questions that require both image and text understanding is still a major challenge in healthcare AI. In this work, we propose Q-FSRU, a new model that combines Frequency Spectrum Representation and Fusion (FSRU) with a method called Quantum Retrieval-Augmented Generation (Quantum RAG) for medical Visual Question Answering (VQA). The model takes in features from medical images and related text, then shifts them into the frequency domain using Fast Fourier Transform (FFT). This helps it focus on more meaningful data and filter out noise or less useful information. To improve accuracy and ensure that answers are based on real knowledge, we add a quantum-inspired retrieval system. It fetches useful medical facts from external sources using quantum-based similarity techniques. These details are then merged with the frequency-based features for stronger reasoning. We evaluated our model using the VQA-RAD dataset, which includes real radiology images and questions. The results showed that Q-FSRU outperforms earlier models, especially on complex cases needing image-text reasoning. The mix of frequency and quantum information improves both performance and explainability. Overall, this approach offers a promising way to build smart, clear, and helpful AI tools for doctors. 

**Abstract (ZH)**: 解决需要同时理解图像和文本的复杂临床问题仍是医疗AI领域的重大挑战。在此工作中，我们提出了一种新的Q-FSRU模型，该模型结合了频谱表示和融合（FSRU）与一种名为量子检索增强生成（Quantum RAG）的方法，用于医学视觉问答（VQA）。该模型接收医学图像和相关文本的特征，然后使用快速傅里叶变换（FFT）将其转换到频域，有助于它聚焦于更有意义的数据并过滤掉噪声或不重要的信息。为提高准确性和确保答案基于真实知识，我们添加了一个受量子启发的检索系统，利用基于量子的方法从外部来源获取有用的医学事实。这些细节随后与基于频谱的特征合并，以增强推理。我们使用包括真实放射学图像和问题的VQA-RAD数据集评估了该模型，结果显示Q-FSRU优于早期模型，尤其是在需要图像文本推理的复杂情况下。频域和量子信息的结合提高了性能和可解释性。总之，这种方法为构建智能、清晰且有助于医生的AI工具提供了有前景的方法。 

---
# UniDCF: A Foundation Model for Comprehensive Dentocraniofacial Hard Tissue Reconstruction 

**Title (ZH)**: UniDCF：全面颌面硬组织重建的基座模型 

**Authors**: Chunxia Ren, Ning Zhu, Yue Lai, Gui Chen, Ruijie Wang, Yangyi Hu, Suyao Liu, Shuwen Mao, Hong Su, Yu Zhang, Li Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11728)  

**Abstract**: Dentocraniofacial hard tissue defects profoundly affect patients' physiological functions, facial aesthetics, and psychological well-being, posing significant challenges for precise reconstruction. Current deep learning models are limited to single-tissue scenarios and modality-specific imaging inputs, resulting in poor generalizability and trade-offs between anatomical fidelity, computational efficiency, and cross-tissue adaptability. Here we introduce UniDCF, a unified framework capable of reconstructing multiple dentocraniofacial hard tissues through multimodal fusion encoding of point clouds and multi-view images. By leveraging the complementary strengths of each modality and incorporating a score-based denoising module to refine surface smoothness, UniDCF overcomes the limitations of prior single-modality approaches. We curated the largest multimodal dataset, comprising intraoral scans, CBCT, and CT from 6,609 patients, resulting in 54,555 annotated instances. Evaluations demonstrate that UniDCF outperforms existing state-of-the-art methods in terms of geometric precision, structural completeness, and spatial accuracy. Clinical simulations indicate UniDCF reduces reconstruction design time by 99% and achieves clinician-rated acceptability exceeding 94%. Overall, UniDCF enables rapid, automated, and high-fidelity reconstruction, supporting personalized and precise restorative treatments, streamlining clinical workflows, and enhancing patient outcomes. 

**Abstract (ZH)**: 统一多模态硬组织重建框架UniDCF：多模态点云和多视角图像融合在多发性牙颅面硬组织重建中的应用 

---
# Separating Knowledge and Perception with Procedural Data 

**Title (ZH)**: 分离知识和感知：基于过程化数据的方法 

**Authors**: Adrián Rodríguez-Muñoz, Manel Baradad, Phillip Isola, Antonio Torralba  

**Link**: [PDF](https://arxiv.org/pdf/2508.11697)  

**Abstract**: We train representation models with procedural data only, and apply them on visual similarity, classification, and semantic segmentation tasks without further training by using visual memory -- an explicit database of reference image embeddings. Unlike prior work on visual memory, our approach achieves full compartmentalization with respect to all real-world images while retaining strong performance. Compared to a model trained on Places, our procedural model performs within $1\%$ on NIGHTS visual similarity, outperforms by $8\%$ and $15\%$ on CUB200 and Flowers102 fine-grained classification, and is within $10\%$ on ImageNet-1K classification. It also demonstrates strong zero-shot segmentation, achieving an $R^2$ on COCO within $10\%$ of the models trained on real data. Finally, we analyze procedural versus real data models, showing that parts of the same object have dissimilar representations in procedural models, resulting in incorrect searches in memory and explaining the remaining performance gap. 

**Abstract (ZH)**: 我们仅使用过程生成数据训练表征模型，并通过视觉记忆（一个显式的参考图像嵌入数据库）在视觉相似性、分类和语义分割任务中应用这些模型，而无需进一步训练。与视觉记忆领域的先前工作相比，我们的方法实现了对所有真实世界图像的完全分隔化，同时保持了强大的性能。与在Places数据集上训练的模型相比，我们的过程生成模型在NIGHTS视觉相似性任务中表现相差1%，在CUB200和Flowers102细粒度分类任务中的表现分别超过8%和15%，在ImageNet-1K分类任务中的表现相差10%。该模型还展示了强大的零样本分割能力，在COCO上的$R^2$分数与基于真实数据训练的模型相差10%以内。最后，我们分析了过程生成数据模型与真实数据模型之间的差异，发现过程生成模型中同一对象的不同部分具有不同的表征，导致记忆搜索错误，并解释了剩余的性能差距。 

---
# RefAdGen: High-Fidelity Advertising Image Generation 

**Title (ZH)**: RefAdGen: 高保真广告图像生成 

**Authors**: Yiyun Chen, Weikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11695)  

**Abstract**: The rapid advancement of Artificial Intelligence Generated Content (AIGC) techniques has unlocked opportunities in generating diverse and compelling advertising images based on referenced product images and textual scene descriptions. This capability substantially reduces human labor and production costs in traditional marketing workflows. However, existing AIGC techniques either demand extensive fine-tuning for each referenced image to achieve high fidelity, or they struggle to maintain fidelity across diverse products, making them impractical for e-commerce and marketing industries. To tackle this limitation, we first construct AdProd-100K, a large-scale advertising image generation dataset. A key innovation in its construction is our dual data augmentation strategy, which fosters robust, 3D-aware representations crucial for realistic and high-fidelity image synthesis. Leveraging this dataset, we propose RefAdGen, a generation framework that achieves high fidelity through a decoupled design. The framework enforces precise spatial control by injecting a product mask at the U-Net input, and employs an efficient Attention Fusion Module (AFM) to integrate product features. This design effectively resolves the fidelity-efficiency dilemma present in existing methods. Extensive experiments demonstrate that RefAdGen achieves state-of-the-art performance, showcasing robust generalization by maintaining high fidelity and remarkable visual results for both unseen products and challenging real-world, in-the-wild images. This offers a scalable and cost-effective alternative to traditional workflows. Code and datasets are publicly available at this https URL. 

**Abstract (ZH)**: 基于引用产品图像和文本场景描述生成的快速进步的人工智能生成内容（AIGC）技术为生成多样化且引人注目的广告图像提供了机会。这一能力大大减少了传统营销工作流程中的人力和生产成本。然而，现有的AIGC技术要么需要对每个引用图像进行大量的微调才能达到高质量，要么无法在多样化的产品中保持高质量，这使它们在电子商务和营销行业中不切实际。为解决这一局限性，我们首先构建了AdProd-100K，一个大规模的广告图像生成数据集。其构建中的关键创新是我们的双重数据增强策略，这对于实现现实和高质量的图像合成是必不可少的。利用这一数据集，我们提出了RefAdGen生成框架，通过解耦设计实现高质量生成。框架通过在U-Net输入中注入产品遮罩来实现精确的空间控制，并采用高效的注意力融合模块（AFM）整合产品特征。这一设计有效解决了现有方法中存在的质量-效率权衡问题。大量实验表明，RefAdGen达到了最先进的性能，展示了在未见过的产品和具有挑战性的现实世界拍摄的图像中都保持高质量和出色的视觉效果的能力。这提供了一种可扩展且成本效益高的传统工作流程替代方案。代码和数据集可在以下链接公开获取。 

---
# Toward Practical Equilibrium Propagation: Brain-inspired Recurrent Neural Network with Feedback Regulation and Residual Connections 

**Title (ZH)**: 面向实用的均衡传播：具有反馈调节和残差连接的脑启发递归神经网络 

**Authors**: Zhuo Liu, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11659)  

**Abstract**: Brain-like intelligent systems need brain-like learning methods. Equilibrium Propagation (EP) is a biologically plausible learning framework with strong potential for brain-inspired computing hardware. However, existing im-plementations of EP suffer from instability and prohibi-tively high computational costs. Inspired by the structure and dynamics of the brain, we propose a biologically plau-sible Feedback-regulated REsidual recurrent neural network (FRE-RNN) and study its learning performance in EP framework. Feedback regulation enables rapid convergence by reducing the spectral radius. The improvement in con-vergence property reduces the computational cost and train-ing time of EP by orders of magnitude, delivering perfor-mance on par with backpropagation (BP) in benchmark tasks. Meanwhile, residual connections with brain-inspired topologies help alleviate the vanishing gradient problem that arises when feedback pathways are weak in deep RNNs. Our approach substantially enhances the applicabil-ity and practicality of EP in large-scale networks that un-derpin artificial intelligence. The techniques developed here also offer guidance to implementing in-situ learning in physical neural networks. 

**Abstract (ZH)**: 脑似的智能系统需要脑似的学习方法。Equilibrium Propagation (EP)是一种具有强大脑启发式计算硬件潜力的生物可实现学习框架。然而，现有的EP实现面临不稳定性问题和高昂的计算成本。受脑结构和动力学的启发，我们提出了一种生物可实现的反馈调节残差递归神经网络（FRE-RNN），并在EP框架下研究其学习性能。反馈调节通过减小特征值半径实现了快速收敛，从而显著提高了EP在收敛性方面的性能，降低了计算成本和培训时间，在基准任务中性能与反向传播（BP）相当。同时，具有脑启发式拓扑结构的残差连接有助于缓解深层RNN中反馈路径较弱时出现的梯度消失问题。我们的方法显著增强了EP在支撑人工智能的大规模网络中的应用性和实用性。所开发的技术也为在物理神经网络中实现就地学习提供了指导。 

---
