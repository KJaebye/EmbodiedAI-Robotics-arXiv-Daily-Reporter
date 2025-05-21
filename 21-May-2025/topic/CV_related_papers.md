# M3Depth: Wavelet-Enhanced Depth Estimation on Mars via Mutual Boosting of Dual-Modal Data 

**Title (ZH)**: M3Depth：通过双模数据相互增强的基于小波增强的火星深度估计 

**Authors**: Junjie Li, Jiawei Wang, Miyu Li, Yu Liu, Yumei Wang, Haitao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14159)  

**Abstract**: Depth estimation plays a great potential role in obstacle avoidance and navigation for further Mars exploration missions. Compared to traditional stereo matching, learning-based stereo depth estimation provides a data-driven approach to infer dense and precise depth maps from stereo image pairs. However, these methods always suffer performance degradation in environments with sparse textures and lacking geometric constraints, such as the unstructured terrain of Mars. To address these challenges, we propose M3Depth, a depth estimation model tailored for Mars rovers. Considering the sparse and smooth texture of Martian terrain, which is primarily composed of low-frequency features, our model incorporates a convolutional kernel based on wavelet transform that effectively captures low-frequency response and expands the receptive field. Additionally, we introduce a consistency loss that explicitly models the complementary relationship between depth map and surface normal map, utilizing the surface normal as a geometric constraint to enhance the accuracy of depth estimation. Besides, a pixel-wise refinement module with mutual boosting mechanism is designed to iteratively refine both depth and surface normal predictions. Experimental results on synthetic Mars datasets with depth annotations show that M3Depth achieves a significant 16% improvement in depth estimation accuracy compared to other state-of-the-art methods in depth estimation. Furthermore, the model demonstrates strong applicability in real-world Martian scenarios, offering a promising solution for future Mars exploration missions. 

**Abstract (ZH)**: 深度估计在进一步火星探测任务中的障碍 Avoidance 和导航中发挥着巨大的潜在作用。与传统的立体匹配方法相比，基于学习的立体深度估计提供了一种数据驱动的方法，可以从立体图像对中推断出密集和精确的深度图。然而，这些方法在纹理稀疏且缺乏几何约束的环境（如火星的无结构地形）中总是会表现出性能下降。为了解决这些挑战，我们提出了M3Depth，一种为火星探测车量身定制的深度估计模型。考虑到火星地形稀疏且平滑的纹理，主要由低频特征组成，我们的模型结合了一个基于小波变换的卷积核，有效捕捉低频响应并扩展了感受野。此外，我们引入了一种一致性损失，明确地表示了深度图和表面法线图之间的互补关系，利用表面法线作为几何约束来增强深度估计的准确性。此外，我们设计了一个像素级 refinements模块，并结合了互促机制，以迭代地细化深度和表面法线预测。实验结果表明，M3Depth在合成火星数据集上的深度估计精度比其他最先进的深度估计方法有显著16%的提升。此外，该模型在实际火星场景中表现出强大的适用性，为未来的火星探测任务提供了前景广阔的有效解决方案。 

---
# 4D-ROLLS: 4D Radar Occupancy Learning via LiDAR Supervision 

**Title (ZH)**: 4D-ROLLS：通过LiDAR监督的4D雷达占用学习 

**Authors**: Ruihan Liu, Xiaoyi Wu, Xijun Chen, Liang Hu, Yunjiang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2505.13905)  

**Abstract**: A comprehensive understanding of 3D scenes is essential for autonomous vehicles (AVs), and among various perception tasks, occupancy estimation plays a central role by providing a general representation of drivable and occupied space. However, most existing occupancy estimation methods rely on LiDAR or cameras, which perform poorly in degraded environments such as smoke, rain, snow, and fog. In this paper, we propose 4D-ROLLS, the first weakly supervised occupancy estimation method for 4D radar using the LiDAR point cloud as the supervisory signal. Specifically, we introduce a method for generating pseudo-LiDAR labels, including occupancy queries and LiDAR height maps, as multi-stage supervision to train the 4D radar occupancy estimation model. Then the model is aligned with the occupancy map produced by LiDAR, fine-tuning its accuracy in occupancy estimation. Extensive comparative experiments validate the exceptional performance of 4D-ROLLS. Its robustness in degraded environments and effectiveness in cross-dataset training are qualitatively demonstrated. The model is also seamlessly transferred to downstream tasks BEV segmentation and point cloud occupancy prediction, highlighting its potential for broader applications. The lightweight network enables 4D-ROLLS model to achieve fast inference speeds at about 30 Hz on a 4060 GPU. The code of 4D-ROLLS will be made available at this https URL. 

**Abstract (ZH)**: 一种用于4D雷达的弱监督占用估计方法：4D-ROLLS 

---
# GeoVLM: Improving Automated Vehicle Geolocalisation Using Vision-Language Matching 

**Title (ZH)**: GeoVLM：利用视觉-语言匹配提高自动车辆地理定位性能 

**Authors**: Barkin Dagda, Muhammad Awais, Saber Fallah  

**Link**: [PDF](https://arxiv.org/pdf/2505.13669)  

**Abstract**: Cross-view geo-localisation identifies coarse geographical position of an automated vehicle by matching a ground-level image to a geo-tagged satellite image from a database. Despite the advancements in Cross-view geo-localisation, significant challenges still persist such as similar looking scenes which makes it challenging to find the correct match as the top match. Existing approaches reach high recall rates but they still fail to rank the correct image as the top match. To address this challenge, this paper proposes GeoVLM, a novel approach which uses the zero-shot capabilities of vision language models to enable cross-view geo-localisation using interpretable cross-view language descriptions. GeoVLM is a trainable reranking approach which improves the best match accuracy of cross-view geo-localisation. GeoVLM is evaluated on standard benchmark VIGOR and University-1652 and also through real-life driving environments using Cross-View United Kingdom, a new benchmark dataset introduced in this paper. The results of the paper show that GeoVLM improves retrieval performance of cross-view geo-localisation compared to the state-of-the-art methods with the help of explainable natural language descriptions. The code is available at this https URL 

**Abstract (ZH)**: 跨视角地理定位通过将地面图像与数据库中的地理标记卫星图像匹配来识别自动驾驶车辆的大致地理位置。尽管跨视角地理定位取得了进展，但仍存在一些重大挑战，例如相似外观的场景，这使得找到正确的匹配作为最优匹配变得具有挑战性。现有的方法可以达到高的召回率，但仍无法将正确的图像排名为最优匹配。为了解决这一挑战，本文提出了一种名为GeoVLM的新方法，该方法利用视觉语言模型的零样本能力，使用可解释的跨视角语言描述来实现跨视角地理定位。GeoVLM 是一种可训练的重排方法，可以提高跨视角地理定位的最佳匹配准确性。GeoVLM 在标准基准 VIGOR 和 University-1652 以及通过使用本文介绍的新基准数据集 Cross-View United Kingdom 的真实驾驶环境进行了评估。本文的结果表明，GeoVLM 在可解释自然语言描述的帮助下，与现有方法相比，提高了跨视角地理定位的检索性能。代码可在以下链接获取。 

---
# Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation 

**Title (ZH)**: 移动代理-V：一种基于视频的轻松高效操作知识注入方法在移动自动化中的应用 

**Authors**: Junyang Wang, Haiyang Xu, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13887)  

**Abstract**: The exponential rise in mobile device usage necessitates streamlined automation for effective task management, yet many AI frameworks fall short due to inadequate operational expertise. While manually written knowledge can bridge this gap, it is often burdensome and inefficient. We introduce Mobile-Agent-V, an innovative framework that utilizes video as a guiding tool to effortlessly and efficiently inject operational knowledge into mobile automation processes. By deriving knowledge directly from video content, Mobile-Agent-V eliminates manual intervention, significantly reducing the effort and time required for knowledge acquisition. To rigorously evaluate this approach, we propose Mobile-Knowledge, a benchmark tailored to assess the impact of external knowledge on mobile agent performance. Our experimental findings demonstrate that Mobile-Agent-V enhances performance by 36% compared to existing methods, underscoring its effortless and efficient advantages in mobile automation. 

**Abstract (ZH)**: 移动设备使用量的指数增长 necessitates 精简自动化以有效进行任务管理，但许多AI框架因缺乏操作专业知识而不足以应对。虽然手动编写的知识可以弥补这一差距，但往往负担沉重且效率低下。我们介绍了 Mobile-Agent-V，这是一种创新框架，利用视频作为引导工具，轻松高效地将操作知识注入移动自动化过程。通过直接从视频内容中提取知识，Mobile-Agent-V 消除了手动干预，显著减少了知识获取所需的精力和时间。为了严格评估此方法，我们提出了 Mobile-Knowledge，这是一个专门用于评估外部知识对移动代理性能影响的基准。我们的实验结果表明，与现有方法相比，Mobile-Agent-V 的性能提高了36%，突显了其在移动自动化中的轻松高效优势。 

---
# Training-Free Watermarking for Autoregressive Image Generation 

**Title (ZH)**: 无需训练的自回归图像生成水印技术 

**Authors**: Yu Tong, Zihao Pan, Shuai Yang, Kaiyang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14673)  

**Abstract**: Invisible image watermarking can protect image ownership and prevent malicious misuse of visual generative models. However, existing generative watermarking methods are mainly designed for diffusion models while watermarking for autoregressive image generation models remains largely underexplored. We propose IndexMark, a training-free watermarking framework for autoregressive image generation models. IndexMark is inspired by the redundancy property of the codebook: replacing autoregressively generated indices with similar indices produces negligible visual differences. The core component in IndexMark is a simple yet effective match-then-replace method, which carefully selects watermark tokens from the codebook based on token similarity, and promotes the use of watermark tokens through token replacement, thereby embedding the watermark without affecting the image quality. Watermark verification is achieved by calculating the proportion of watermark tokens in generated images, with precision further improved by an Index Encoder. Furthermore, we introduce an auxiliary validation scheme to enhance robustness against cropping attacks. Experiments demonstrate that IndexMark achieves state-of-the-art performance in terms of image quality and verification accuracy, and exhibits robustness against various perturbations, including cropping, noises, Gaussian blur, random erasing, color jittering, and JPEG compression. 

**Abstract (ZH)**: 无迹图像水印可以保护图像所有权并防止视觉生成模型的恶意滥用。然而，现有的生成水印方法主要针对扩散模型，而针对自回归图像生成模型的水印研究仍相对不足。我们提出了IndexMark，一种无需训练的自回归图像生成模型水印框架。IndexMark 受码本冗余性启发：用相似的索引替换自回归生成的索引会产生微乎其微的视觉差异。IndexMark 的核心组件是一种简单有效的匹配然后替换方法，该方法根据标记相似性选择水印标记，并通过标记替换促进水印标记的使用，从而在不损害图像质量的情况下嵌入水印。通过计算生成图像中水印标记的比例实现水印验证，精确度进一步通过索引编码器提高。此外，我们引入了辅助验证方案以增强对裁剪攻击的鲁棒性。实验结果显示，IndexMark 在图像质量和验证准确性方面达到了最先进的性能，并且对抗各种扰动（包括裁剪、噪声、高斯模糊、随机擦除、色彩抖动和JPEG压缩）均表现出鲁棒性。 

---
# EmoGist: Efficient In-Context Learning for Visual Emotion Understanding 

**Title (ZH)**: EmoGist: 有效的基于上下文的学习方法用于视觉情感理解 

**Authors**: Ronald Seoh, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2505.14660)  

**Abstract**: In this paper, we introduce EmoGist, a training-free, in-context learning method for performing visual emotion classification with LVLMs. The key intuition of our approach is that context-dependent definition of emotion labels could allow more accurate predictions of emotions, as the ways in which emotions manifest within images are highly context dependent and nuanced. EmoGist pre-generates multiple explanations of emotion labels, by analyzing the clusters of example images belonging to each category. At test time, we retrieve a version of explanation based on embedding similarity, and feed it to a fast VLM for classification. Through our experiments, we show that EmoGist allows up to 13 points improvement in micro F1 scores with the multi-label Memotion dataset, and up to 8 points in macro F1 in the multi-class FI dataset. 

**Abstract (ZH)**: 基于上下文学习的EmoGist：一种无需训练的情感图像分类方法 

---
# CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code Generation 

**Title (ZH)**: CAD-Coder: 一种面向计算机辅助设计代码生成的开源视觉-语言模型 

**Authors**: Anna C. Doris, Md Ferdous Alam, Amin Heyrani Nobari, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2505.14646)  

**Abstract**: Efficient creation of accurate and editable 3D CAD models is critical in engineering design, significantly impacting cost and time-to-market in product innovation. Current manual workflows remain highly time-consuming and demand extensive user expertise. While recent developments in AI-driven CAD generation show promise, existing models are limited by incomplete representations of CAD operations, inability to generalize to real-world images, and low output accuracy. This paper introduces CAD-Coder, an open-source Vision-Language Model (VLM) explicitly fine-tuned to generate editable CAD code (CadQuery Python) directly from visual input. Leveraging a novel dataset that we created--GenCAD-Code, consisting of over 163k CAD-model image and code pairs--CAD-Coder outperforms state-of-the-art VLM baselines such as GPT-4.5 and Qwen2.5-VL-72B, achieving a 100% valid syntax rate and the highest accuracy in 3D solid similarity. Notably, our VLM demonstrates some signs of generalizability, successfully generating CAD code from real-world images and executing CAD operations unseen during fine-tuning. The performance and adaptability of CAD-Coder highlights the potential of VLMs fine-tuned on code to streamline CAD workflows for engineers and designers. CAD-Coder is publicly available at: this https URL. 

**Abstract (ZH)**: 高效创建准确可编辑的3D CAD模型对于工程设计至关重要，显著影响产品创新的成本和时间。当前的手动工作流程仍然高度耗时，并需要大量的用户专业知识。尽管AI驱动的CAD生成技术前景看好，但现有模型受限于不完整的CAD操作表示、无法泛化到真实世界图像以及较低的输出准确性。本文介绍了CAD-Coder，一个专门微调以直接从视觉输入生成可编辑CAD代码（CadQuery Python）的开源视觉语言模型（VLM）。利用我们创建的一个新颖的数据集GenCAD-Code，包含超过16.3万对CAD模型图像和代码——CAD-Coder在状态最先进VLM基线如GPT-4.5和Qwen2.5-VL-72B上表现出色，实现了100%的有效语法率和最高的三维实体相似性准确性。值得注意的是，我们的VLM显示出一些泛化能力，在微调期间未见过的CAD操作中成功从真实世界图像生成CAD代码并执行相应操作。CAD-Coder的性能和适应性突显了针对代码进行微调的VLMs为工程师和设计师简化CAD工作流的潜力。CAD-Coder可在以下链接获取：this https URL。 

---
# Replace in Translation: Boost Concept Alignment in Counterfactual Text-to-Image 

**Title (ZH)**: 替换翻译：提升反事实文本到图像的概念对齐 

**Authors**: Sifan Li, Ming Tao, Hao Zhao, Ling Shao, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14341)  

**Abstract**: Text-to-Image (T2I) has been prevalent in recent years, with most common condition tasks having been optimized nicely. Besides, counterfactual Text-to-Image is obstructing us from a more versatile AIGC experience. For those scenes that are impossible to happen in real world and anti-physics, we should spare no efforts in increasing the factual feel, which means synthesizing images that people think very likely to be happening, and concept alignment, which means all the required objects should be in the same frame. In this paper, we focus on concept alignment. As controllable T2I models have achieved satisfactory performance for real applications, we utilize this technology to replace the objects in a synthesized image in latent space step-by-step to change the image from a common scene to a counterfactual scene to meet the prompt. We propose a strategy to instruct this replacing process, which is called as Explicit Logical Narrative Prompt (ELNP), by using the newly SoTA language model DeepSeek to generate the instructions. Furthermore, to evaluate models' performance in counterfactual T2I, we design a metric to calculate how many required concepts in the prompt can be covered averagely in the synthesized images. The extensive experiments and qualitative comparisons demonstrate that our strategy can boost the concept alignment in counterfactual T2I. 

**Abstract (ZH)**: 文本到图像（T2I）近年来十分盛行，大多数常见条件任务已得到很好地优化。然而，反事实文本到图像限制了我们获得更丰富多样的AIGC体验。对于现实中不可能发生的场景和反物理场景，我们应该竭尽全力增加其实感感，即合成人们认为很可能会发生的图像，以及概念对齐，即所有所需的对象应在同一帧中。本文侧重于概念对齐。鉴于可控文本到图像模型在现实应用中已取得满意性能，我们利用此技术在潜在空间中逐步替换合成图像中的对象，将图像从常见场景转变为反事实场景以满足提示要求。我们提出了一种策略来指导这一替换过程，称为显式逻辑叙述提示（ELNP），并通过最新的最先进语言模型DeepSeek生成指令。此外，为了评估模型在反事实T2I中的性能，我们设计了一个度量标准来计算合成图像中平均能覆盖提示中所需概念的数量。广泛实验和定性比较表明，我们的策略能够提升反事实T2I中的概念对齐效果。 

---
# AppleGrowthVision: A large-scale stereo dataset for phenological analysis, fruit detection, and 3D reconstruction in apple orchards 

**Title (ZH)**: 苹果生长视觉：适用于苹果 orchards 阶段分析、果实检测及三维重建的大规模立体数据集 

**Authors**: Laura-Sophia von Hirschhausen, Jannes S. Magnusson, Mykyta Kovalenko, Fredrik Boye, Tanay Rawat, Peter Eisert, Anna Hilsmann, Sebastian Pretzsch, Sebastian Bosse  

**Link**: [PDF](https://arxiv.org/pdf/2505.14029)  

**Abstract**: Deep learning has transformed computer vision for precision agriculture, yet apple orchard monitoring remains limited by dataset constraints. The lack of diverse, realistic datasets and the difficulty of annotating dense, heterogeneous scenes. Existing datasets overlook different growth stages and stereo imagery, both essential for realistic 3D modeling of orchards and tasks like fruit localization, yield estimation, and structural analysis. To address these gaps, we present AppleGrowthVision, a large-scale dataset comprising two subsets. The first includes 9,317 high resolution stereo images collected from a farm in Brandenburg (Germany), covering six agriculturally validated growth stages over a full growth cycle. The second subset consists of 1,125 densely annotated images from the same farm in Brandenburg and one in Pillnitz (Germany), containing a total of 31,084 apple labels. AppleGrowthVision provides stereo-image data with agriculturally validated growth stages, enabling precise phenological analysis and 3D reconstructions. Extending MinneApple with our data improves YOLOv8 performance by 7.69 % in terms of F1-score, while adding it to MinneApple and MAD boosts Faster R-CNN F1-score by 31.06 %. Additionally, six BBCH stages were predicted with over 95 % accuracy using VGG16, ResNet152, DenseNet201, and MobileNetv2. AppleGrowthVision bridges the gap between agricultural science and computer vision, by enabling the development of robust models for fruit detection, growth modeling, and 3D analysis in precision agriculture. Future work includes improving annotation, enhancing 3D reconstruction, and extending multimodal analysis across all growth stages. 

**Abstract (ZH)**: Deep Learning已在精确农业中重塑了计算机视觉，但苹果园监测仍受限于数据集约束。缺少多样且现实的数据集以及标注密集且异质场景的难度。现有数据集忽略了不同生长阶段和立体影像，这两者对于真实的三维建模以及果实定位、产量估计和结构分析至关重要。为填补这些空白，我们提出了AppleGrowthVision，这是一个大规模数据集，包含两个子集。第一个子集包括9,317张高分辨率立体图像，从德国Brandenburg的一个农场收集，涵盖了整个生长周期的六个农学验证生长阶段。第二个子集包含来自Brandenburg和Pillnitz（德国）的同一个农场的1,125张密集标注图像，总共有31,084个苹果标签。AppleGrowthVision提供了具有农学验证生长阶段的立体图像数据，使精确的性状分析和三维重建成为可能。通过将我们的数据与MinneApple结合，YOLOv8的F1分数提高了7.69%，而添加到MinneApple和MAD中，则使Faster R-CNN的F1分数提高了31.06%。使用VGG16、ResNet152、DenseNet201和MobileNetv2，可以上述95%以上的准确率预测六种BBCH生长阶段。AppleGrowthVision弥合了农业科学与计算机视觉之间的差距，通过支持用于果实检测、生长建模和精确农业中三维分析的稳健模型的发展。未来的工作包括改进注释、增强三维重建，并在整个生长阶段扩展多模态分析。 

---
# XDementNET: An Explainable Attention Based Deep Convolutional Network to Detect Alzheimer Progression from MRI data 

**Title (ZH)**: XDementNET：一种用于检测MRI数据中阿尔茨海默病进展情况的可解释注意力基于深卷积网络 

**Authors**: Soyabul Islam Lincoln, Mirza Mohd Shahriar Maswood  

**Link**: [PDF](https://arxiv.org/pdf/2505.13906)  

**Abstract**: A common neurodegenerative disease, Alzheimer's disease requires a precise diagnosis and efficient treatment, particularly in light of escalating healthcare expenses and the expanding use of artificial intelligence in medical diagnostics. Many recent studies shows that the combination of brain Magnetic Resonance Imaging (MRI) and deep neural networks have achieved promising results for diagnosing AD. Using deep convolutional neural networks, this paper introduces a novel deep learning architecture that incorporates multiresidual blocks, specialized spatial attention blocks, grouped query attention, and multi-head attention. The study assessed the model's performance on four publicly accessible datasets and concentrated on identifying binary and multiclass issues across various categories. This paper also takes into account of the explainability of AD's progression and compared with state-of-the-art methods namely Gradient Class Activation Mapping (GradCAM), Score-CAM, Faster Score-CAM, and XGRADCAM. Our methodology consistently outperforms current approaches, achieving 99.66\% accuracy in 4-class classification, 99.63\% in 3-class classification, and 100\% in binary classification using Kaggle datasets. For Open Access Series of Imaging Studies (OASIS) datasets the accuracies are 99.92\%, 99.90\%, and 99.95\% respectively. The Alzheimer's Disease Neuroimaging Initiative-1 (ADNI-1) dataset was used for experiments in three planes (axial, sagittal, and coronal) and a combination of all planes. The study achieved accuracies of 99.08\% for axis, 99.85\% for sagittal, 99.5\% for coronal, and 99.17\% for all axis, and 97.79\% and 8.60\% respectively for ADNI-2. The network's ability to retrieve important information from MRI images is demonstrated by its excellent accuracy in categorizing AD stages. 

**Abstract (ZH)**: 一种常见的神经退行性疾病——阿尔茨海默病需要精确的诊断和高效的治疗，特别是在医疗费用不断上升和人工智能在医疗诊断中广泛应用的背景下。许多近期的研究表明，脑磁共振成像（MRI）与深度神经网络的结合在诊断AD方面取得了令人瞩目的成果。利用深度卷积神经网络，本文提出了一种新型深度学习架构，该架构结合了多残差块、专门的空间注意力块、分组查询注意力和多头注意力机制。研究在四个公开可用的数据集上评估了模型性能，并集中于不同类别下的二元和多元分类问题。本文还考虑了AD进展的可解释性问题，并与当前最先进的方法，如梯度类激活映射（GradCAM）、评分-CAM、快速评分-CAM和XGRADCAM进行了比较。我们的方法在Kaggle数据集上始终优于现有方法，分别实现99.66%的四分类准确率、99.63%的三分类准确率和100%的二分类准确率。对于Open Access Series of Imaging Studies（OASIS）数据集，准确率分别为99.92%、99.90%和99.95%。在Alzheimer's Disease Neuroimaging Initiative-1（ADNI-1）数据集上，在三个切面（轴位、矢状位和冠状位）及其组合条件下进行实验，准确率分别为99.08%、99.85%、99.5%和99.17%，以及8.60%的ADNI-2准确率。网络从MRI图像中提取重要信息的能力通过其在分类AD阶段方面的卓越准确率得到了展示。 

---
# Domain Adaptation of VLM for Soccer Video Understanding 

**Title (ZH)**: 足球视频理解中VLM的领域适应 

**Authors**: Tiancheng Jiang, Henry Wang, Md Sirajus Salekin, Parmida Atighehchian, Shinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13860)  

**Abstract**: Vision Language Models (VLMs) have demonstrated strong performance in multi-modal tasks by effectively aligning visual and textual representations. However, most video understanding VLM research has been domain-agnostic, leaving the understanding of their transfer learning capability to specialized domains under-explored. In this work, we address this by exploring the adaptability of open-source VLMs to specific domains, and focusing on soccer as an initial case study. Our approach uses large-scale soccer datasets and LLM to create instruction-following data, and use them to iteratively fine-tune the general-domain VLM in a curriculum learning fashion (first teaching the model key soccer concepts to then question answering tasks). The final adapted model, trained using a curated dataset of 20k video clips, exhibits significant improvement in soccer-specific tasks compared to the base model, with a 37.5% relative improvement for the visual question-answering task and an accuracy improvement from 11.8% to 63.5% for the downstream soccer action classification task. 

**Abstract (ZH)**: 开源视觉语言模型在特定领域中的可适应性研究：以足球为例 

---
# Learning Spatio-Temporal Dynamics for Trajectory Recovery via Time-Aware Transformer 

**Title (ZH)**: 基于时间意识变换器的时空动态学习及其在轨迹恢复中的应用 

**Authors**: Tian Sun, Yuqi Chen, Baihua Zheng, Weiwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.13857)  

**Abstract**: In real-world applications, GPS trajectories often suffer from low sampling rates, with large and irregular intervals between consecutive GPS points. This sparse characteristic presents challenges for their direct use in GPS-based systems. This paper addresses the task of map-constrained trajectory recovery, aiming to enhance trajectory sampling rates of GPS trajectories. Previous studies commonly adopt a sequence-to-sequence framework, where an encoder captures the trajectory patterns and a decoder reconstructs the target trajectory. Within this framework, effectively representing the road network and extracting relevant trajectory features are crucial for overall performance. Despite advancements in these models, they fail to fully leverage the complex spatio-temporal dynamics present in both the trajectory and the road network.
To overcome these limitations, we categorize the spatio-temporal dynamics of trajectory data into two distinct aspects: spatial-temporal traffic dynamics and trajectory dynamics. Furthermore, We propose TedTrajRec, a novel method for trajectory recovery. To capture spatio-temporal traffic dynamics, we introduce PD-GNN, which models periodic patterns and learns topologically aware dynamics concurrently for each road segment. For spatio-temporal trajectory dynamics, we present TedFormer, a time-aware Transformer that incorporates temporal dynamics for each GPS location by integrating closed-form neural ordinary differential equations into the attention mechanism. This allows TedFormer to effectively handle irregularly sampled data. Extensive experiments on three real-world datasets demonstrate the superior performance of TedTrajRec. The code is publicly available at this https URL. 

**Abstract (ZH)**: 基于地图约束的GPS轨迹恢复：时空动态建模与重构方法 

---
# ClapFM-EVC: High-Fidelity and Flexible Emotional Voice Conversion with Dual Control from Natural Language and Speech 

**Title (ZH)**: ClapFM-EVC：基于自然语言和语音双重控制的高质量和灵活情感语音转换 

**Authors**: Yu Pan, Yanni Hu, Yuguang Yang, Jixun Yao, Jianhao Ye, Hongbin Zhou, Lei Ma, Jianjun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13805)  

**Abstract**: Despite great advances, achieving high-fidelity emotional voice conversion (EVC) with flexible and interpretable control remains challenging. This paper introduces ClapFM-EVC, a novel EVC framework capable of generating high-quality converted speech driven by natural language prompts or reference speech with adjustable emotion intensity. We first propose EVC-CLAP, an emotional contrastive language-audio pre-training model, guided by natural language prompts and categorical labels, to extract and align fine-grained emotional elements across speech and text modalities. Then, a FuEncoder with an adaptive intensity gate is presented to seamless fuse emotional features with Phonetic PosteriorGrams from a pre-trained ASR model. To further improve emotion expressiveness and speech naturalness, we propose a flow matching model conditioned on these captured features to reconstruct Mel-spectrogram of source speech. Subjective and objective evaluations validate the effectiveness of ClapFM-EVC. 

**Abstract (ZH)**: 尽管取得了巨大进步，实现具有灵活可解释控制的高保真情感语音转换（EVC）仍然具有挑战性。本文介绍了一种新型EVC框架ClapFM-EVC，该框架能够根据自然语言提示或参考语音生成具有可调节情感强度的高质量转换语音。我们首先提出了由自然语言提示和类别标签引导的情感对比语言-音频预训练模型EVC-CLAP，以在语音和文本模态之间提取和对齐细粒度的情感元素。然后，提出了一个带有自适应强度门控的FuEncoder，以无缝融合情感特征与预训练ASR模型的音素后验gram。为进一步提高情感表达能力和语音自然度，我们提出了一种基于这些捕获特征的流匹配模型，以重建源语音的梅尔频谱图。主客观评估证明了ClapFM-EVC的有效性。 

---
# Learning Wavelet-Sparse FDK for 3D Cone-Beam CT Reconstruction 

**Title (ZH)**: 学习小波稀疏FDK算法在3D锥束CT重建中的应用 

**Authors**: Yipeng Sun, Linda-Sophie Schneider, Chengze Ye, Mingxuan Gu, Siyuan Mei, Siming Bayer, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2505.13579)  

**Abstract**: Cone-Beam Computed Tomography (CBCT) is essential in medical imaging, and the Feldkamp-Davis-Kress (FDK) algorithm is a popular choice for reconstruction due to its efficiency. However, FDK is susceptible to noise and artifacts. While recent deep learning methods offer improved image quality, they often increase computational complexity and lack the interpretability of traditional methods. In this paper, we introduce an enhanced FDK-based neural network that maintains the classical algorithm's interpretability by selectively integrating trainable elements into the cosine weighting and filtering stages. Recognizing the challenge of a large parameter space inherent in 3D CBCT data, we leverage wavelet transformations to create sparse representations of the cosine weights and filters. This strategic sparsification reduces the parameter count by $93.75\%$ without compromising performance, accelerates convergence, and importantly, maintains the inference computational cost equivalent to the classical FDK algorithm. Our method not only ensures volumetric consistency and boosts robustness to noise, but is also designed for straightforward integration into existing CT reconstruction pipelines. This presents a pragmatic enhancement that can benefit clinical applications, particularly in environments with computational limitations. 

**Abstract (ZH)**: 基于锥束计算断层成像的改进FDK神经网络：稀疏表示与计算效率的平衡 

---
