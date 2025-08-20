# The 9th AI City Challenge 

**Title (ZH)**: 第九届AI城市挑战赛 

**Authors**: Zheng Tang, Shuo Wang, David C. Anastasiu, Ming-Ching Chang, Anuj Sharma, Quan Kong, Norimasa Kobori, Munkhjargal Gochoo, Ganzorig Batnasan, Munkh-Erdene Otgonbold, Fady Alnajjar, Jun-Wei Hsieh, Tomasz Kornuta, Xiaolong Li, Yilin Zhao, Han Zhang, Subhashree Radhakrishnan, Arihant Jain, Ratnesh Kumar, Vidya N. Murali, Yuxing Wang, Sameer Satish Pusegaonkar, Yizhou Wang, Sujit Biswas, Xunlei Wu, Zhedong Zheng, Pranamesh Chakraborty, Rama Chellappa  

**Link**: [PDF](https://arxiv.org/pdf/2508.13564)  

**Abstract**: The ninth AI City Challenge continues to advance real-world applications of computer vision and AI in transportation, industrial automation, and public safety. The 2025 edition featured four tracks and saw a 17% increase in participation, with 245 teams from 15 countries registered on the evaluation server. Public release of challenge datasets led to over 30,000 downloads to date. Track 1 focused on multi-class 3D multi-camera tracking, involving people, humanoids, autonomous mobile robots, and forklifts, using detailed calibration and 3D bounding box annotations. Track 2 tackled video question answering in traffic safety, with multi-camera incident understanding enriched by 3D gaze labels. Track 3 addressed fine-grained spatial reasoning in dynamic warehouse environments, requiring AI systems to interpret RGB-D inputs and answer spatial questions that combine perception, geometry, and language. Both Track 1 and Track 3 datasets were generated in NVIDIA Omniverse. Track 4 emphasized efficient road object detection from fisheye cameras, supporting lightweight, real-time deployment on edge devices. The evaluation framework enforced submission limits and used a partially held-out test set to ensure fair benchmarking. Final rankings were revealed after the competition concluded, fostering reproducibility and mitigating overfitting. Several teams achieved top-tier results, setting new benchmarks in multiple tasks. 

**Abstract (ZH)**: 第九届AI城市挑战赛继续推动计算机视觉和人工智能在交通、工业自动化和公共安全领域的实际应用。2025年版设立了四个赛道，参赛队伍增加了17%，共有来自15个国家的245支队伍在评估服务器上注册。挑战赛数据集的公开发布迄今已超过30,000次下载。赛道1专注于多类3D多摄像头跟踪，涉及行人、类人机器人、自主移动机器人和叉车，使用详细的标定和3D边界框注释。赛道2解决交通安全性中的视频问答问题，通过3D凝视标签增强多摄像头事件理解。赛道3解决动态仓库环境中的细粒度空间推理问题，需要人工智能系统解释RGB-D输入并回答结合感知、几何和语言的空间问题。赛道1和赛道3的数据集均在NVIDIA Omniverse中生成。赛道4强调从鱼眼摄像头高效检测道路对象，支持在边缘设备上进行轻量级、实时部署。评估框架设置了提交限制，并使用部分保留的测试集确保公平基准测试。比赛结束后公布最终排名，促进可重复性并减轻过拟合问题。多支队伍取得了顶尖成绩，多个任务中设立了新的基准。 

---
# V2P: From Background Suppression to Center Peaking for Robust GUI Grounding Task 

**Title (ZH)**: V2P: 从背景抑制到中心峰化以实现稳健的GUI定位任务 

**Authors**: Jikai Chen, Long Chen, Dong Wang, Leilei Gan, Chenyi Zhuang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13634)  

**Abstract**: Precise localization of GUI elements is crucial for the development of GUI agents. Traditional methods rely on bounding box or center-point regression, neglecting spatial interaction uncertainty and visual-semantic hierarchies. Recent methods incorporate attention mechanisms but still face two key issues: (1) ignoring processing background regions causes attention drift from the desired area, and (2) uniform labeling fails to distinguish between center and edges of the target UI element, leading to click imprecision. Inspired by how humans visually process and interact with GUI elements, we propose the Valley-to-Peak (V2P) method to address these issues. To mitigate background distractions, V2P introduces a suppression attention mechanism that minimizes the model's focus on irrelevant regions to highlight the intended region. For the issue of center-edge distinction, V2P applies a Fitts' Law-inspired approach by modeling GUI interactions as 2D Gaussian heatmaps where the weight gradually decreases from the center towards the edges. The weight distribution follows a Gaussian function, with the variance determined by the target's size. Consequently, V2P effectively isolates the target area and teaches the model to concentrate on the most essential point of the UI element. The model trained by V2P achieves the performance with 92.3% and 50.5% on two benchmarks ScreenSpot-v2 and ScreenSpot-Pro. Ablations further confirm each component's contribution, highlighting V2P's generalizability for precise GUI grounding tasks. 

**Abstract (ZH)**: 谷到峰(SV到PF)方法在GUI元素精确定位中的应用 

---
# GeoSAM2: Unleashing the Power of SAM2 for 3D Part Segmentation 

**Title (ZH)**: GeoSAM2: 解锁SAM2在三维部件分割中的潜力 

**Authors**: Ken Deng, Yunhan Yang, Jingxiang Sun, Xihui Liu, Yebin Liu, Ding Liang, Yan-Pei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14036)  

**Abstract**: Modern 3D generation methods can rapidly create shapes from sparse or single views, but their outputs often lack geometric detail due to computational constraints. We present DetailGen3D, a generative approach specifically designed to enhance these generated 3D shapes. Our key insight is to model the coarse-to-fine transformation directly through data-dependent flows in latent space, avoiding the computational overhead of large-scale 3D generative models. We introduce a token matching strategy that ensures accurate spatial correspondence during refinement, enabling local detail synthesis while preserving global structure. By carefully designing our training data to match the characteristics of synthesized coarse shapes, our method can effectively enhance shapes produced by various 3D generation and reconstruction approaches, from single-view to sparse multi-view inputs. Extensive experiments demonstrate that DetailGen3D achieves high-fidelity geometric detail synthesis while maintaining efficiency in training. 

**Abstract (ZH)**: 现代的3D生成方法可以快速从稀疏或单视角生成形状，但由于计算约束，其输出往往缺乏几何细节。我们提出了DetailGen3D，一种专门用于增强这些生成的3D形状的生成方法。我们的关键见解是通过数据相关的流在潜在空间中直接建模从粗略到精细的变换，从而避免大规模3D生成模型的计算开销。我们引入了一种token匹配策略，确保在细化过程中准确的空间对应，从而实现局部细节合成的同时保留全局结构。通过精心设计训练数据以匹配合成粗略形状的特征，该方法可以有效地增强各种3D生成和重建方法产生的形状，从单视图到稀疏多视图输入。大量实验表明，DetailGen3D在保持训练效率的同时实现了高保真几何细节合成。 

---
# A Novel Attention-Augmented Wavelet YOLO System for Real-time Brain Vessel Segmentation on Transcranial Color-coded Doppler 

**Title (ZH)**: 一种用于经颅彩色编码多普勒实时脑血管分割的新型注意力增强小波YOLO系统 

**Authors**: Wenxuan Zhang, Shuai Li, Xinyi Wang, Yu Sun, Hongyu Kang, Pui Yuk Chryste Wan, Yong-Ping Zheng, Sai-Kit Lam  

**Link**: [PDF](https://arxiv.org/pdf/2508.13875)  

**Abstract**: The Circle of Willis (CoW), vital for ensuring consistent blood flow to the brain, is closely linked to ischemic stroke. Accurate assessment of the CoW is important for identifying individuals at risk and guiding appropriate clinical management. Among existing imaging methods, Transcranial Color-coded Doppler (TCCD) offers unique advantages due to its radiation-free nature, affordability, and accessibility. However, reliable TCCD assessments depend heavily on operator expertise for identifying anatomical landmarks and performing accurate angle correction, which limits its widespread adoption. To address this challenge, we propose an AI-powered, real-time CoW auto-segmentation system capable of efficiently capturing cerebral arteries. No prior studies have explored AI-driven cerebrovascular segmentation using TCCD. In this work, we introduce a novel Attention-Augmented Wavelet YOLO (AAW-YOLO) network tailored for TCCD data, designed to provide real-time guidance for brain vessel segmentation in the CoW. We prospectively collected TCCD data comprising 738 annotated frames and 3,419 labeled artery instances to establish a high-quality dataset for model training and evaluation. The proposed AAW-YOLO demonstrated strong performance in segmenting both ipsilateral and contralateral CoW vessels, achieving an average Dice score of 0.901, IoU of 0.823, precision of 0.882, recall of 0.926, and mAP of 0.953, with a per-frame inference speed of 14.199 ms. This system offers a practical solution to reduce reliance on operator experience in TCCD-based cerebrovascular screening, with potential applications in routine clinical workflows and resource-constrained settings. Future research will explore bilateral modeling and larger-scale validation. 

**Abstract (ZH)**: Willis环自动分割的注意力增强小波YOLO网络在经颅彩色编码多普勒成像中的应用 

---
# Comparing Conditional Diffusion Models for Synthesizing Contrast-Enhanced Breast MRI from Pre-Contrast Images 

**Title (ZH)**: 比较条件扩散模型在从非增强MRI合成增强乳腺MRI中的应用 

**Authors**: Sebastian Ibarra, Javier del Riego, Alessandro Catanese, Julian Cuba, Julian Cardona, Nataly Leon, Jonathan Infante, Karim Lekadir, Oliver Diaz, Richard Osuala  

**Link**: [PDF](https://arxiv.org/pdf/2508.13776)  

**Abstract**: Dynamic contrast-enhanced (DCE) MRI is essential for breast cancer diagnosis and treatment. However, its reliance on contrast agents introduces safety concerns, contraindications, increased cost, and workflow complexity. To this end, we present pre-contrast conditioned denoising diffusion probabilistic models to synthesize DCE-MRI, introducing, evaluating, and comparing a total of 22 generative model variants in both single-breast and full breast settings. Towards enhancing lesion fidelity, we introduce both tumor-aware loss functions and explicit tumor segmentation mask conditioning. Using a public multicenter dataset and comparing to respective pre-contrast baselines, we observe that subtraction image-based models consistently outperform post-contrast-based models across five complementary evaluation metrics. Apart from assessing the entire image, we also separately evaluate the region of interest, where both tumor-aware losses and segmentation mask inputs improve evaluation metrics. The latter notably enhance qualitative results capturing contrast uptake, albeit assuming access to tumor localization inputs that are not guaranteed to be available in screening settings. A reader study involving 2 radiologists and 4 MRI technologists confirms the high realism of the synthetic images, indicating an emerging clinical potential of generative contrast-enhancement. We share our codebase at this https URL. 

**Abstract (ZH)**: 基于对比剂的磁共振成像(DCE-MRI)对于乳腺癌诊断和治疗至关重要。然而，其对对比剂的依赖引入了安全问题、禁忌症、成本增加和工作流程复杂性。为此，我们提出了预对比条件下的一种去噪扩散概率模型来合成DCE-MRI，共介绍了、评估和比较了22种生成模型变体，在单侧乳腺和全乳腺设置中进行了研究。为了增强病灶保真度，我们引入了肿瘤感知损失函数和显式的肿瘤分割掩码条件。使用公开的多中心数据集，并与相应的预对比基线进行比较，我们观察到，减影像基模型在五个互补评估指标中始终优于基于后对比的模型。除了评估整个影像外，我们还分别评估了感兴趣区域，在该区域中，肿瘤感知损失和分割掩码输入均能提高评估指标。后者显著提升了捕捉对比剂摄取的定性结果，尽管假定可获得肿瘤定位输入，而这些输入在筛查环境中并不总是可获得的。两位放射科医生和四位MRI技术人员的读者研究证实了合成影像的高度真实性，表明生成对比增强在临床中具有潜在应用价值。我们已在如下链接共享了我们的代码库：this https URL。 

---
# EAvatar: Expression-Aware Head Avatar Reconstruction with Generative Geometry Priors 

**Title (ZH)**: EAvatar：带有生成几何先验的表达意识头部avatar重建 

**Authors**: Shikun Zhang, Cunjian Chen, Yiqun Wang, Qiuhong Ke, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13537)  

**Abstract**: High-fidelity head avatar reconstruction plays a crucial role in AR/VR, gaming, and multimedia content creation. Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated effectiveness in modeling complex geometry with real-time rendering capability and are now widely used in high-fidelity head avatar reconstruction tasks. However, existing 3DGS-based methods still face significant challenges in capturing fine-grained facial expressions and preserving local texture continuity, especially in highly deformable regions. To mitigate these limitations, we propose a novel 3DGS-based framework termed EAvatar for head reconstruction that is both expression-aware and deformation-aware. Our method introduces a sparse expression control mechanism, where a small number of key Gaussians are used to influence the deformation of their neighboring Gaussians, enabling accurate modeling of local deformations and fine-scale texture transitions. Furthermore, we leverage high-quality 3D priors from pretrained generative models to provide a more reliable facial geometry, offering structural guidance that improves convergence stability and shape accuracy during training. Experimental results demonstrate that our method produces more accurate and visually coherent head reconstructions with improved expression controllability and detail fidelity. 

**Abstract (ZH)**: 高保真头部 avatar 重建在 AR/VR、游戏和多媒体内容创作中发挥着关键作用。基于 3D 高斯点绘制（3DGS）的Recent进展显示出在实时渲染能力下模拟复杂几何形状的有效性，并且现在被广泛应用于高保真头部 avatar 重建任务中。然而，现有的基于 3DGS 的方法在捕捉细微表情变化和保持局部纹理连续性方面仍然面临重大挑战，特别是在高度可变形区域。为缓解这些限制，我们提出了一种名为 EAvatar 的新颖基于 3DGS 的框架，该框架既具备表情感知能力又具备变形感知能力。我们的方法引入了一种稀疏表情控制机制，使用少量关键高斯点影响邻近高斯点的变形，以实现局部变形和精细尺度纹理过渡的准确建模。此外，我们利用预训练生成模型提供的高质量 3D 先验知识，提供更可靠的面部几何结构，从结构上指导训练以提高收敛稳定性和形状准确性。实验结果表明，我们的方法生成了更准确且视觉连贯的头部重建，同时提高了表情可控性和细节保真度。 

---
# Evaluating Open-Source Vision Language Models for Facial Emotion Recognition against Traditional Deep Learning Models 

**Title (ZH)**: 评估开源视觉语言模型在面部情绪识别任务中与传统深度学习模型的性能对比 

**Authors**: Vamsi Krishna Mulukutla, Sai Supriya Pavarala, Srinivasa Raju Rudraraju, Sridevi Bonthu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13524)  

**Abstract**: Facial Emotion Recognition (FER) is crucial for applications such as human-computer interaction and mental health diagnostics. This study presents the first empirical comparison of open-source Vision-Language Models (VLMs), including Phi-3.5 Vision and CLIP, against traditional deep learning models VGG19, ResNet-50, and EfficientNet-B0 on the challenging FER-2013 dataset, which contains 35,887 low-resolution grayscale images across seven emotion classes. To address the mismatch between VLM training assumptions and the noisy nature of FER data, we introduce a novel pipeline that integrates GFPGAN-based image restoration with FER evaluation. Results show that traditional models, particularly EfficientNet-B0 (86.44%) and ResNet-50 (85.72%), significantly outperform VLMs like CLIP (64.07%) and Phi-3.5 Vision (51.66%), highlighting the limitations of VLMs in low-quality visual tasks. In addition to performance evaluation using precision, recall, F1-score, and accuracy, we provide a detailed computational cost analysis covering preprocessing, training, inference, and evaluation phases, offering practical insights for deployment. This work underscores the need for adapting VLMs to noisy environments and provides a reproducible benchmark for future research in emotion recognition. 

**Abstract (ZH)**: 开放源代码视觉-语言模型在FER-2013数据集上的面部情感识别对比研究：适应噪声环境的必要性与可重复基准 

---
# Structured Prompting and Multi-Agent Knowledge Distillation for Traffic Video Interpretation and Risk Inference 

**Title (ZH)**: 结构化提示与多agents知识蒸馏在交通视频解释与风险推理中的应用 

**Authors**: Yunxiang Yang, Ningning Xu, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13439)  

**Abstract**: Comprehensive highway scene understanding and robust traffic risk inference are vital for advancing Intelligent Transportation Systems (ITS) and autonomous driving. Traditional approaches often struggle with scalability and generalization, particularly under the complex and dynamic conditions of real-world environments. To address these challenges, we introduce a novel structured prompting and knowledge distillation framework that enables automatic generation of high-quality traffic scene annotations and contextual risk assessments. Our framework orchestrates two large Vision-Language Models (VLMs): GPT-4o and o3-mini, using a structured Chain-of-Thought (CoT) strategy to produce rich, multi-perspective outputs. These outputs serve as knowledge-enriched pseudo-annotations for supervised fine-tuning of a much smaller student VLM. The resulting compact 3B-scale model, named VISTA (Vision for Intelligent Scene and Traffic Analysis), is capable of understanding low-resolution traffic videos and generating semantically faithful, risk-aware captions. Despite its significantly reduced parameter count, VISTA achieves strong performance across established captioning metrics (BLEU-4, METEOR, ROUGE-L, and CIDEr) when benchmarked against its teacher models. This demonstrates that effective knowledge distillation and structured multi-agent supervision can empower lightweight VLMs to capture complex reasoning capabilities. The compact architecture of VISTA facilitates efficient deployment on edge devices, enabling real-time risk monitoring without requiring extensive infrastructure upgrades. 

**Abstract (ZH)**: 全面的高速公路场景理解和稳健的交通风险推断对于推动智能交通系统（ITS）和自动驾驶至关重要。传统的 approach 通常在处理真实环境下的复杂和动态条件时表现出可扩展性和泛化能力的不足。为了解决这些挑战，我们提出了一种新型的结构化提示和知识蒸馏框架，该框架能够自动生成高质量的交通场景标注和上下文风险评估。该框架利用结构化链式思考（CoT）策略协调两个大型ビジョンと言語モデル（VLMs）：GPT-4o和o3-mini，生成丰富、多视角的输出。这些输出作为知识丰富的伪标注，用于监督微调一个小得多的学生VLM。由此产生的紧凑3B量级模型，名为VISTA（视觉智能场景与交通分析），能够理解低分辨率的交通视频并生成语义忠实、风险意识强的描述。尽管参数量显著减少，但VISTA在基准测试中表现出色，其性能在现有描述生成指标（BLEU-4、METEOR、ROUGE-L和CIDEr）上达到了强劲的表现。这表明有效的知识蒸馏和结构化多代理监督可以使轻量级的VLMs具备复杂的推理能力。VISTA的紧凑架构便于在边缘设备上高效部署，无需进行广泛的基础设施升级即可实现实时风险监测。 

---
# GaitCrafter: Diffusion Model for Biometric Preserving Gait Synthesis 

**Title (ZH)**: 步态匠人：保留生物特征的步态合成扩散模型 

**Authors**: Sirshapan Mitra, Yogesh S. Rawat  

**Link**: [PDF](https://arxiv.org/pdf/2508.13300)  

**Abstract**: Gait recognition is a valuable biometric task that enables the identification of individuals from a distance based on their walking patterns. However, it remains limited by the lack of large-scale labeled datasets and the difficulty of collecting diverse gait samples for each individual while preserving privacy. To address these challenges, we propose GaitCrafter, a diffusion-based framework for synthesizing realistic gait sequences in the silhouette domain. Unlike prior works that rely on simulated environments or alternative generative models, GaitCrafter trains a video diffusion model from scratch, exclusively on gait silhouette data. Our approach enables the generation of temporally consistent and identity-preserving gait sequences. Moreover, the generation process is controllable-allowing conditioning on various covariates such as clothing, carried objects, and view angle. We show that incorporating synthetic samples generated by GaitCrafter into the gait recognition pipeline leads to improved performance, especially under challenging conditions. Additionally, we introduce a mechanism to generate novel identities-synthetic individuals not present in the original dataset-by interpolating identity embeddings. These novel identities exhibit unique, consistent gait patterns and are useful for training models while maintaining privacy of real subjects. Overall, our work takes an important step toward leveraging diffusion models for high-quality, controllable, and privacy-aware gait data generation. 

**Abstract (ZH)**: 基于扩散模型的 silhouette 领域实际步态序列合成框架 GaitCrafter 

---
# PreSem-Surf: RGB-D Surface Reconstruction with Progressive Semantic Modeling and SG-MLP Pre-Rendering Mechanism 

**Title (ZH)**: PreSem-Surf: 基于 progressive semantic modeling 和 SG-MLP 预渲染机制的 RGB-D 表面重建 

**Authors**: Yuyan Ye, Hang Xu, Yanghang Huang, Jiali Huang, Qian Weng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13228)  

**Abstract**: This paper proposes PreSem-Surf, an optimized method based on the Neural Radiance Field (NeRF) framework, capable of reconstructing high-quality scene surfaces from RGB-D sequences in a short time. The method integrates RGB, depth, and semantic information to improve reconstruction performance. Specifically, a novel SG-MLP sampling structure combined with PR-MLP (Preconditioning Multilayer Perceptron) is introduced for voxel pre-rendering, allowing the model to capture scene-related information earlier and better distinguish noise from local details. Furthermore, progressive semantic modeling is adopted to extract semantic information at increasing levels of precision, reducing training time while enhancing scene understanding. Experiments on seven synthetic scenes with six evaluation metrics show that PreSem-Surf achieves the best performance in C-L1, F-score, and IoU, while maintaining competitive results in NC, Accuracy, and Completeness, demonstrating its effectiveness and practical applicability. 

**Abstract (ZH)**: 基于Neural Radiance Field框架的PreSem-Surf：一种高效的RGB-D序列场景 surfaces 重建方法 

---
# MIRAGE: Towards AI-Generated Image Detection in the Wild 

**Title (ZH)**: MIRAGE:面向野生环境中的AI生成图像检测 

**Authors**: Cheng Xia, Manxi Lin, Jiexiang Tan, Xiaoxiong Du, Yang Qiu, Junjun Zheng, Xiangheng Kong, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13223)  

**Abstract**: The spreading of AI-generated images (AIGI), driven by advances in generative AI, poses a significant threat to information security and public trust. Existing AIGI detectors, while effective against images in clean laboratory settings, fail to generalize to in-the-wild scenarios. These real-world images are noisy, varying from ``obviously fake" images to realistic ones derived from multiple generative models and further edited for quality control. We address in-the-wild AIGI detection in this paper. We introduce Mirage, a challenging benchmark designed to emulate the complexity of in-the-wild AIGI. Mirage is constructed from two sources: (1) a large corpus of Internet-sourced AIGI verified by human experts, and (2) a synthesized dataset created through the collaboration between multiple expert generators, closely simulating the realistic AIGI in the wild. Building on this benchmark, we propose Mirage-R1, a vision-language model with heuristic-to-analytic reasoning, a reflective reasoning mechanism for AIGI detection. Mirage-R1 is trained in two stages: a supervised-fine-tuning cold start, followed by a reinforcement learning stage. By further adopting an inference-time adaptive thinking strategy, Mirage-R1 is able to provide either a quick judgment or a more robust and accurate conclusion, effectively balancing inference speed and performance. Extensive experiments show that our model leads state-of-the-art detectors by 5% and 10% on Mirage and the public benchmark, respectively. The benchmark and code will be made publicly available. 

**Abstract (ZH)**: AI生成图像在野检测：Mirage及其挑战基准 

---
